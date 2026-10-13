"""Audit actual model-consumed crops, including DataLoader worker output.

No images are resampled, labels are not inspected, and no random generators
are used. Retention is measured against the tumour mask in the full cache.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class CropAudit:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def update(self, batch: dict) -> None:
        metadata = batch.get("crop_audit")
        if metadata is None:
            raise ValueError("Crop audit enabled but the dataset returned no crop metadata")
        for index, patient_id in enumerate(batch["patient_id"]):
            row = {"patient_id": str(patient_id)}
            for name, values in metadata.items():
                value = values[index]
                row[name] = value.item() if hasattr(value, "item") else value
            # An empty full mask has undefined retention, not 0% or 100%.
            if row["tumor_voxels"] == 0:
                row["tumor_retention"] = None
            self.rows.append(row)

    def summary(self) -> dict:
        known = [row for row in self.rows if row["tumor_voxels"] > 0]
        retention = np.asarray([row["tumor_retention"] for row in known], dtype=float)
        below = sum(
            row["tumor_retention"] < row["min_tumor_retention"] for row in known
        )
        return {
            "n_crops": len(self.rows),
            "n_mask_available": len(known),
            "n_empty_masks": len(self.rows) - len(known),
            "n_tumor_guided": sum(row["branch"] == "tumor_guided" for row in self.rows),
            "n_random_brain": sum(row["branch"] == "random_brain" for row in self.rows),
            "n_fallback": sum(row["fallback"] for row in self.rows),
            "mean_tumor_retention": float(retention.mean()) if len(known) else None,
            "min_tumor_retention": float(retention.min()) if len(known) else None,
            "p05_tumor_retention": float(np.quantile(retention, 0.05)) if len(known) else None,
            "n_below_retention_target": below,
            "below_retention_target_fraction": below / len(known) if known else None,
            "n_zero_tumor_crops": int(np.count_nonzero(retention == 0.0)),
        }

    def write_epoch(self, path: Path, *, epoch: int, fold: int, task: str) -> dict:
        """Append small per-crop records; reset this run's CSV at epoch one."""
        summary = self.summary()
        if not self.rows:
            return summary
        path.parent.mkdir(parents=True, exist_ok=True)
        prefix = {"fold": fold, "task": task, "epoch": epoch}
        records = [{**prefix, **row} for row in self.rows]
        mode = "w" if epoch == 1 else "a"
        write_header = epoch == 1 or not path.exists()
        with path.open(mode, newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            if write_header:
                writer.writeheader()
            writer.writerows(records)
        return summary

    def log(self) -> str:
        stats = self.summary()
        mean = stats["mean_tumor_retention"]
        return (
            f"crop_retention_mean={mean:.4f} " if mean is not None else "crop_retention_mean=NA "
        ) + (
            f"crop_below_target={stats['n_below_retention_target']}/{stats['n_mask_available']} "
            f"crop_zero_tumor={stats['n_zero_tumor_crops']} "
            f"crop_empty_masks={stats['n_empty_masks']} "
            f"crop_fallbacks={stats['n_fallback']}"
        )

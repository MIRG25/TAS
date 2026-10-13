"""Fixed same-scale validation perturbations, never a crop/model selector."""
from __future__ import annotations

import numpy as np
import pandas as pd

from glioma_meta.data.dataset import GliomaMRIDataset


VERSION = "frozen_validation_crop_sensitivity_v1"
VIEWS = {
    "center": (0, 0, 0),
    "d_minus16": (-16, 0, 0), "d_plus16": (16, 0, 0),
    "h_minus16": (0, -16, 0), "h_plus16": (0, 16, 0),
    "w_minus16": (0, 0, -16), "w_plus16": (0, 0, 16),
}


def fixed_crop(image, mask, shape, offset):
    """Offset the *clipped production evaluation start*, in cached D/H/W voxels.

    No random draws, padding, resizing, intensity changes or retention-based
    selection. Boundaries clip requested offsets; actual starts are recorded.
    """
    shape = np.asarray(shape, dtype=int)
    offset = np.asarray(offset, dtype=int)
    if image.ndim != 4 or image.shape[0] != 4 or mask.shape != image.shape[1:]:
        raise ValueError("Expected aligned four-modality MRI and mask")
    if shape.shape != (3,) or offset.shape != (3,) or np.any(shape <= 0):
        raise ValueError("Expected positive three-dimensional crop shape and 3D offset")
    maximum = np.asarray(mask.shape) - shape
    if np.any(maximum < 0):
        raise ValueError("Crop exceeds cache: this audit never pads/resizes")
    mask = mask.astype(bool, copy=False)
    total = int(np.count_nonzero(mask))
    if total:
        coordinates = np.argwhere(mask)
        center = (coordinates.min(0) + coordinates.max(0)) / 2.0
        base = np.clip(np.rint(center - (shape - 1) / 2.0).astype(int), 0, maximum)
    else:
        base = maximum // 2
    start = np.clip(base + offset, 0, maximum)
    slices = tuple(slice(int(s), int(s + n)) for s, n in zip(start, shape))
    crop = np.array(image[(slice(None), *slices)], dtype=np.float32, copy=True, order="C")
    if not np.isfinite(crop).all():
        raise ValueError("Non-finite cached MRI values")
    metadata = {
        "tumor_voxels": total,
        "tumor_retention": float(np.count_nonzero(mask[slices]) / total) if total else float("nan"),
        "brain_fraction": float(np.any(crop != 0, axis=0).mean()),
        "offset_clipped": bool(np.any(start != base + offset)),
        "same_as_center": bool(np.array_equal(start, base)),
    }
    for axis, s, b, requested in zip("dhw", start, base, offset):
        metadata.update({f"start_{axis}": int(s), f"center_start_{axis}": int(b),
                         f"actual_offset_{axis}": int(s - b),
                         f"requested_offset_{axis}": int(requested)})
    return crop, metadata


class FixedCropDataset(GliomaMRIDataset):
    """Cache-only, validation-only dataset; raw-MRI fallback is forbidden."""

    def __init__(self, manifest, ids, config, task, view):
        data = config["data"]
        if data.get("roi_mode") != "full_volume" or not data.get("cache_dir"):
            raise ValueError("Crop audit requires the original full-volume cache")
        super().__init__(manifest, ids, crop_shape=data["crop_shape"],
                         cache_shape=data["cache_shape"], cache_dir=data["cache_dir"],
                         random_crop=False, included_tasks=[task], audit_crops=True)
        self.offset = VIEWS[view]

    def _image(self, row, audit=None):
        image_path = self.cache_dir / f"{row.patient_id}.npy"
        mask_path = self.cache_dir / f"{row.patient_id}_mask.npy"
        image = np.load(image_path, mmap_mode="r", allow_pickle=False)
        mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        if image.shape != (4, *self.cache_shape):
            raise ValueError(f"Cache/config shape mismatch: {image_path}")
        result, metadata = fixed_crop(image, mask, self.crop_shape, self.offset)
        if audit is not None:
            audit.update(metadata)
        return result


class RecordingLoader:
    """Retain only small crop metadata while streaming normal inference batches."""

    def __init__(self, loader, name):
        self.loader, self.name, self.rows = loader, name, []

    def __iter__(self):
        for batch_number, batch in enumerate(self.loader, 1):
            for i, patient in enumerate(batch["patient_id"]):
                self.rows.append({"patient_id": str(patient), **{
                    key: values[i].item() for key, values in batch["crop_audit"].items()}})
            if batch_number == 1 or batch_number % 20 == 0 or batch_number == len(self.loader):
                print(f"  {self.name}: batch={batch_number}/{len(self.loader)} "
                      f"patients={len(self.rows)}", flush=True)
            yield batch


def check_reproduction(expected, actual, task, tolerance=1e-5):
    """Original validation ranking/decisions must reproduce before perturbations."""
    keys = ("c_index",) if task == "survival" else ("acc", "auc", "f1", "sensitivity", "specificity")
    differences = {}
    for key in keys:
        a, b = expected.get(key), actual.get(key)
        if a is None or b is None or not np.isfinite([a, b]).all():
            raise ValueError(f"Cannot verify original validation metric: {key}")
        differences[key] = float(b - a)
        if abs(b - a) > tolerance:
            raise ValueError(f"Center reproduction failed for {key}: saved={a}, replay={b}. "
                             "No shifted crops evaluated; check cache/code/runtime consistency.")
    return differences


def sensitivity_summary(predictions: pd.DataFrame, task: str):
    """Unweighted patient summaries of UNIQUE actual crops, not an ensemble.

    Per-view rank correlations compare entire fixed patient cohorts. Individual
    patient variability/flip flags exclude boundary-duplicated crop starts.
    """
    if predictions.duplicated(["patient_id", "view"]).any():
        raise ValueError("Duplicate patient/view predictions")
    groups = predictions.groupby("view")
    if set(groups.groups) != set(VIEWS):
        raise ValueError("All seven prespecified views are required")
    center = groups.get_group("center").set_index("patient_id")
    score = "risk_score" if task == "survival" else "probability_class_1"
    if not np.isfinite(predictions[score]).all():
        raise ValueError("Non-finite prediction scores")
    for view, frame in groups:
        if set(frame.patient_id) != set(center.index):
            raise ValueError(f"Patient mismatch in view {view}")
    center_sd = float(center[score].std(ddof=0))
    rows = []
    for patient, frame in predictions.groupby("patient_id", sort=False):
        # Explicit order keeps center first even when input row order differs.
        ordered = frame.set_index("view").loc[list(VIEWS)].reset_index()
        unique = ordered.drop_duplicates(["start_d", "start_h", "start_w"])
        values = unique[score].to_numpy(float)
        reference = center.loc[patient]
        sd = float(values.std(ddof=0))
        row = {"patient_id": patient, "n_unique_crops": len(unique),
               "center_score": float(reference[score]), "score_std": sd,
               "score_range": float(np.ptp(values)),
               "max_abs_change_from_center": float(np.abs(values - reference[score]).max()),
               "score_std_over_between_patient_center_sd": sd / center_sd if center_sd > 0 else None,
               "center_tumor_retention": float(reference.tumor_retention),
               "min_tumor_retention": float(unique.tumor_retention.min()),
               "max_tumor_retention": float(unique.tumor_retention.max())}
        if task != "survival":
            row["any_class_flip"] = bool((unique.predicted_class != reference.predicted_class).any())
        rows.append(row)
    patients = pd.DataFrame(rows)
    ranks = {}
    for view, frame in groups:
        paired = frame.set_index("patient_id").loc[center.index]
        a, b = center[score].rank(), paired[score].rank()
        ranks[view] = float(a.corr(b)) if a.nunique() > 1 and b.nunique() > 1 else None
    result = {"n_patients": len(patients), "score": score,
              "between_patient_center_score_sd": center_sd,
              "mean_unique_crops": float(patients.n_unique_crops.mean()),
              "n_patients_with_no_distinct_shift": int((patients.n_unique_crops == 1).sum()),
              "mean_within_patient_score_std": float(patients.score_std.mean()),
              "median_max_abs_change": float(patients.max_abs_change_from_center.median()),
              "p95_max_abs_change": float(patients.max_abs_change_from_center.quantile(.95)),
              "mean_score_std_over_between_patient_center_sd":
                  float(patients.score_std.mean() / center_sd) if center_sd > 0 else None,
              "spearman_with_center_by_view": ranks,
              "center_tumor_retention_mean": float(patients.center_tumor_retention.mean())
                  if patients.center_tumor_retention.notna().any() else None,
              "n_patients_empty_tumor_mask": int(patients.center_tumor_retention.isna().sum()),
              "any_class_flip_fraction": float(patients.any_class_flip.mean()) if task != "survival" else None}
    return result, patients

"""Target-only fixed-fit weighting ablation; legacy weighted CE stays default."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from glioma_meta.training.runtime import class_weights
from glioma_meta.training.seed_repeat import (
    PATIENT_FIELDS, config_digest, file_sha256, source_state_digest,
)

KEY = "classification_loss_reduction"
LEGACY = "batch-weight-mean"
FIXED = "fit-patient-mean"
VERSION = "target_fit_patient_weighting_v1"


def fit_patient_weights(frame, task, device, config):
    """E_fit[w_y] = 1; only valid fit labels determine counts and scale."""
    raw = class_weights(frame, task, device, config)
    column = f"{task}_label"
    valid = frame[column].notna()
    if f"{task}_mask" in frame:
        valid &= frame[f"{task}_mask"].fillna(False).astype(bool)
    labels = frame.loc[valid, column].to_numpy(dtype=float)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Fixed-fit weights require binary valid labels")
    counts = np.bincount(labels.astype(int), minlength=2)
    probability = torch.as_tensor(counts / counts.sum(), device=device, dtype=raw.dtype)
    normalizer = (raw * probability).sum()
    if not torch.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("Non-finite/non-positive fit weight normalizer")
    weights = raw / normalizer
    return weights, {
        "version": VERSION, "reduction": FIXED, "counts": counts.tolist(),
        "raw_weights": raw.detach().cpu().tolist(),
        "fit_weight_normalizer": normalizer.item(),
        "normalized_weights": weights.detach().cpu().tolist(),
        "weight_reference": "target_fit_only",
        "fit_mean_weight": (weights * probability).sum().item(),
    }


def check_loss_ablation(baseline, *, baseline_path, config, source_state, split, target_task, fold):
    old_config = copy.deepcopy(baseline["config"])
    new_config = copy.deepcopy(config)
    if (old_config["adaptation"].pop(KEY, LEGACY) != LEGACY
            or new_config["adaptation"].pop(KEY, LEGACY) != FIXED
            or old_config != new_config):
        raise ValueError("Loss ablation must change ONLY adaptation.classification_loss_reduction")
    if (target_task not in {"idh", "who"} or baseline.get("target_task") != target_task
            or baseline.get("fold") != fold or baseline.get("target_initialization") != "source-shared"
            or config["adaptation"].get("target_initialization") != "source-shared"
            or config["adaptation"].get("refit_on_full_label_budget", True)):
        raise ValueError("Loss ablation requires matched classification task/fold, source-shared, no refit")
    old_split = baseline["split"]
    if (not split.get("protocol_id") or split["protocol_id"] != old_split.get("protocol_id")
            or split.get("label_seed") != old_split.get("label_seed")):
        raise ValueError("Loss ablation protocol/label seed differs")
    for field in PATIENT_FIELDS:
        old, new = old_split.get(field, []), split.get(field, [])
        if not old or len(set(old)) != len(old) or old != new:
            raise ValueError(f"Loss ablation patient IDs/order differ: {field}")
    if not source_state or not any(n.startswith("shared_adapters.") for n in source_state):
        raise ValueError("Missing source-shared tensors")
    for name, value in source_state.items():
        if name not in baseline["model_state"] or not torch.equal(value.cpu(), baseline["model_state"][name].cpu()):
            raise ValueError(f"Source tensor changed: {name}")
    return {
        "version": VERSION, "baseline_checkpoint": str(Path(baseline_path)),
        "baseline_checkpoint_sha256": file_sha256(baseline_path),
        "configuration_except_reduction_sha256": config_digest(old_config),
        "source_state_sha256": source_state_digest(source_state),
        "training_seed": config["training"].get("seed", 2026),
        "only_requested_training_change": f"adaptation.{KEY}",
        "same_source_weights": True, "same_patient_sets": True,
    }

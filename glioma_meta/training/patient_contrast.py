"""Training-only, label-free patient contrast for source-shared MRI features.

The positive is an intensity-perturbed view of the *same* online MRI crop.
Other fit patients supply negatives; clinical labels never define pairs. A
momentum key encoder and a patient-deduplicated queue support small 3-D batches.
This is a MoCo-style auxiliary objective, not a new inference branch.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
import math
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


VERSION = "source_patient_contrast_v1"
DEFAULTS = {
    "enabled": False, "weight": 0.05, "projection_dim": 128,
    "queue_size": 256, "temperature": 0.2, "momentum": 0.99,
    "gain_range": 0.1, "offset_std": 0.05, "noise_std": 0.02,
}


def contrast_settings(source_settings: dict[str, Any]) -> dict[str, Any]:
    supplied = source_settings.get("patient_contrast", {})
    if not isinstance(supplied, dict) or not isinstance(supplied.get("enabled", False), bool):
        raise ValueError("source_training.patient_contrast must be a mapping with a boolean enabled")
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown patient_contrast settings: {sorted(unknown)}")
    cfg = {**DEFAULTS, **supplied}
    for key in ("projection_dim", "queue_size"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"patient_contrast.{key} must be a positive integer")
    for key in ("weight", "temperature", "momentum", "gain_range", "offset_std", "noise_std"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"patient_contrast.{key} must be finite")
        if value < 0:
            raise ValueError(f"patient_contrast.{key} must be non-negative")
    if cfg["temperature"] <= 0 or (cfg["enabled"] and cfg["weight"] <= 0):
        raise ValueError("Enabled patient contrast requires positive weight and temperature")
    if cfg["momentum"] >= 1 or cfg["gain_range"] >= 1:
        raise ValueError("patient_contrast momentum and gain_range must be less than one")
    if cfg["enabled"] and not any(cfg[k] > 0 for k in ("gain_range", "offset_std", "noise_std")):
        raise ValueError("Patient contrast requires a non-identity second view")
    return cfg


def patient_ids_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(map(str, ids)), separators=(",", ":")).encode()).hexdigest()


def contrast_metadata(settings: dict[str, Any], fit_ids: Sequence[str]) -> dict[str, Any]:
    cfg = contrast_settings(settings)
    if not cfg["enabled"]:
        return {"enabled": False}
    return {**cfg, "version": VERSION, "fit_patient_count": len(fit_ids),
            "fit_patient_ids_sha256": patient_ids_hash(fit_ids),
            "uses_target_labels": False, "queue_scope": "source_fit_only",
            "view_rule": "same_online_crop_intensity_perturbation",
            "inference_module": False}


def validate_contrast_checkpoint(checkpoint: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Reject old/mismatched source checkpoints before expensive target training."""
    source_settings = config.get("source_training", {})
    enabled = contrast_settings(source_settings)["enabled"]
    expected = (contrast_metadata(source_settings, checkpoint["split"]["fit_ids"])
                if enabled else {"enabled": False})
    recorded = checkpoint.get("patient_contrast", {"enabled": False})
    if expected != recorded:
        raise ValueError("Source patient_contrast metadata does not match this experiment; "
                         "use the newly trained source checkpoint, not the legacy source root")
    return recorded


class PatientContrast:
    """Not attached to the predictive model: no inference/checkpoint key changes."""

    def __init__(self, model: nn.Module, settings: dict[str, Any], fit_ids: Sequence[str], seed: int):
        self.settings = contrast_settings(settings)
        if not self.settings["enabled"]:
            raise ValueError("Construct PatientContrast only when enabled")
        ids = list(map(str, fit_ids))
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError("Patient contrast needs at least two unique source-fit patients")
        self.allowed_ids = frozenset(ids)
        self.device = next(model.parameters()).device
        # Initializing an optional projector must not change the supervised RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 917)
            self.projector = nn.Sequential(nn.LayerNorm(model.feature_dim),
                nn.Linear(model.feature_dim, model.feature_dim), nn.GELU(),
                nn.Linear(model.feature_dim, self.settings["projection_dim"]))
        self.projector.to(self.device)
        self.teacher = copy.deepcopy(model).eval().requires_grad_(False)
        self.key_projector = copy.deepcopy(self.projector).eval().requires_grad_(False)
        self.generator = torch.Generator(device=self.device).manual_seed(seed + 1907)
        self.queue: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.updates = 0

    def _ids(self, patient_ids: Sequence[str]) -> list[str]:
        ids = list(map(str, patient_ids))
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("A patient-contrast batch must contain unique patients")
        if not set(ids).issubset(self.allowed_ids):
            raise ValueError("Patient-contrast queue/view access outside source fit is forbidden")
        return ids

    @torch.no_grad()
    def second_view(self, image: torch.Tensor) -> torch.Tensor:
        # Same geometry/modalities. No new ROI crop, channel dropout, label, or
        # segmentation input. Preserve exact-zero background after perturbation.
        shape = (*image.shape[:2], *([1] * (image.ndim - 2)))
        options = dict(device=image.device, dtype=torch.float32, generator=self.generator)
        gain = 1 + (torch.rand(shape, **options) * 2 - 1) * self.settings["gain_range"]
        offset = torch.randn(shape, **options) * self.settings["offset_std"]
        noise = torch.randn(image.shape, **options) * self.settings["noise_std"]
        return ((image.float() * gain + offset + noise) * (image != 0)).to(image.dtype)

    def loss(self, shared_features: torch.Tensor, image: torch.Tensor, patient_ids: Sequence[str]):
        ids = self._ids(patient_ids)
        if len(ids) != len(image) or len(ids) != len(shared_features):
            raise ValueError("Patient IDs and MRI batch sizes do not match")
        with torch.no_grad():
            key_features = self.teacher.encode_source_shared(self.second_view(image))
            keys = F.normalize(self.key_projector(key_features).float(), dim=-1)
        queries = F.normalize(self.projector(shared_features).float(), dim=-1)
        # Include current keys and the queue once per patient, preferring fresh
        # keys. A returning patient's stale key is neither another positive nor
        # a false negative. Stack creates a snapshot safe for backward/enqueue.
        candidates = OrderedDict(self.queue)
        candidates.update(zip(ids, keys.detach().unbind(0)))
        candidate_ids = list(candidates)
        bank = torch.stack(list(candidates.values()))
        with torch.autocast(device_type=image.device.type, enabled=False):
            positive = (queries * keys.detach()).sum(-1, keepdim=True)
            negatives = queries @ bank.float().T
            same_patient = torch.tensor([[pid == other for other in candidate_ids] for pid in ids],
                                        device=self.device, dtype=torch.bool)
            negatives = negatives.masked_fill(same_patient, -torch.inf)
            logits = torch.cat((positive, negatives), dim=1) / self.settings["temperature"]
            loss = F.cross_entropy(logits, torch.zeros(len(ids), dtype=torch.long, device=self.device))
        diagnostics = {"positive_cosine": float(positive.detach().mean()),
                       "negatives_per_query": float((~same_patient).sum(1).float().mean())}
        return loss, keys.detach(), diagnostics

    @torch.no_grad()
    def after_step(self, model: nn.Module, keys: torch.Tensor, patient_ids: Sequence[str]) -> None:
        """Call only after a successful optimizer step (not an AMP overflow)."""
        ids = self._ids(patient_ids)
        if len(keys) != len(ids) or not torch.isfinite(keys).all():
            raise ValueError("Invalid patient-contrast keys")
        momentum = self.settings["momentum"]
        # The MAE is frozen; only the shared adapters and projector are EMA
        # updated. Private clinical adapters/heads are not used by the teacher.
        for target, source in zip(self.teacher.shared_adapters.parameters(), model.shared_adapters.parameters()):
            target.lerp_(source.detach(), 1 - momentum)
        for target, source in zip(self.key_projector.parameters(), self.projector.parameters()):
            target.lerp_(source.detach(), 1 - momentum)
        for pid, key in zip(ids, keys):
            self.queue.pop(pid, None)
            self.queue[pid] = key.detach().clone()
        while len(self.queue) > self.settings["queue_size"]:
            self.queue.popitem(last=False)
        self.updates += 1

    def audit(self) -> dict[str, Any]:
        return {"updates": self.updates, "queue_patients": len(self.queue),
                "queue_patient_ids_sha256": patient_ids_hash(list(self.queue)),
                "queue_within_source_fit": set(self.queue).issubset(self.allowed_ids),
                "teacher_requires_grad": any(p.requires_grad for p in self.teacher.parameters())}

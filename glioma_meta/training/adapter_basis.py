"""Freeze adapter input projections without changing its forward or state keys."""
from __future__ import annotations

import hashlib
import json

import torch

KEY = "target_adapter_basis_trainable"
COUNT = "target_adapter_basis_frozen_parameters"
DIGEST = "target_adapter_basis_initial_sha256"
UNCHANGED = "target_adapter_basis_unchanged"


def validate_config(config):
    settings = config["adaptation"]
    enabled = settings.get(KEY, True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{KEY} must be a boolean")
    if not enabled and (settings.get("source_residuals", True)
                        or settings.get("head_warmstart_epochs", 0)
                        or settings.get("refit_on_full_label_budget", True)
                        or config["model"].get("compositional_adapter", {}).get("enabled", False)):
        raise ValueError("Fixed adapter basis requires direct adaptation, no warmstart, composition or refit")
    return enabled


def basis_state(state, task):
    prefix = f"target_adapters.{task}."
    return {k: v.detach().cpu().clone() for k, v in state.items()
            if k.startswith(prefix) and (".norm." in k or ".down." in k)}


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def apply_fixed_basis(model, task, config):
    if validate_config(config):
        return
    for adapter in model.target_adapters[task]:
        adapter.norm.requires_grad_(False)
        adapter.down.requires_grad_(False)
    snapshots = getattr(model, "_fixed_adapter_basis_snapshots", {})
    snapshots[task] = basis_state(model.state_dict(), task)
    model._fixed_adapter_basis_snapshots = snapshots


def metadata(model, task):
    snapshot = getattr(model, "_fixed_adapter_basis_snapshots", {}).get(task)
    if snapshot is None:
        return {KEY: True}
    current = basis_state(model.state_dict(), task)
    if current.keys() != snapshot.keys() or any(not torch.equal(current[k], snapshot[k]) for k in snapshot):
        raise RuntimeError("Frozen target adapter basis changed during adaptation")
    if any(p.requires_grad for a in model.target_adapters[task]
           for part in (a.norm, a.down) for p in part.parameters()):
        raise RuntimeError("Frozen target adapter basis was re-enabled")
    return {KEY: False, COUNT: sum(v.numel() for v in snapshot.values()),
            DIGEST: state_digest(snapshot), UNCHANGED: True}


def validate_checkpoint(checkpoint, config):
    enabled = validate_config(config)
    if checkpoint.get(KEY, True) is not enabled:
        raise ValueError("Adapter basis checkpoint/config metadata mismatch")
    if not enabled:
        state = basis_state(checkpoint["model_state"], checkpoint["target_task"])
        if (not state or checkpoint.get(UNCHANGED) is not True
                or checkpoint.get(COUNT) != sum(v.numel() for v in state.values())
                or checkpoint.get(DIGEST) != state_digest(state)):
            raise ValueError("Frozen adapter basis checkpoint fingerprint mismatch")


def checkpoint_metadata(checkpoint):
    return {k: checkpoint[k] for k in (KEY, COUNT, DIGEST, UNCHANGED) if k in checkpoint}

"""Patient-disjoint source/target development protocols.

Reserve one common validation patient pool before any supervised source fit.
Target fit AND validation annotations count towards the advertised label budget.
The source model never fits patients from that reserved pool, at any budget.
Default v1 is label-blind. The separate balanced-100pct version explicitly uses
cohort labels for allocation and must not be used for strict low-label claims.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from glioma_meta.training.transfer import label_budget_ids
from glioma_meta.data.who_labels import LEGACY_WHO, WHO_MAPS, manifest_who_definition, map_who_grades


VERSION = "patient-disjoint-transfer-v1"
LABEL_DEFINITION = {"idh": "0=wildtype,1=mutant", "who": LEGACY_WHO}


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def validate_binary_manifest(manifest: pd.DataFrame) -> None:
    definition = manifest_who_definition(manifest)
    if manifest.patient_id.isna().any() or manifest.patient_id.duplicated().any():
        raise ValueError("Manifest patient IDs must be present and unique")
    for task in ("idh", "who"):
        labels = pd.to_numeric(manifest[f"{task}_label"], errors="raise")
        if not set(labels.dropna().unique()).issubset({0, 1}):
            raise ValueError(f"{task} requires binary 0/1 labels; use artifacts/dataset_who_binary")
    if "WHO CNS Grade" in manifest:
        expected = map_who_grades(manifest["WHO CNS Grade"], definition)
        present = manifest.who_label.notna()
        if not (expected[present] == manifest.loc[present, "who_label"]).all():
            raise ValueError(f"WHO labels do not match original grades: {definition}")


def _validate_split(split: dict, outer: dict) -> None:
    sets = []
    for key in ("fit_ids", "validation_ids", "test_ids"):
        ids = split[key]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Protocol {key} must be nonempty and unique")
        sets.append(set(ids))
    fit, validation, test = sets
    if fit & validation or fit & test or validation & test:
        raise ValueError("Protocol patient sets overlap")
    if fit | validation != set(outer["train_ids"]) or test != set(outer["test_ids"]):
        raise ValueError("Protocol does not match the requested outer fold")


def build_transfer_protocol(folds: dict, validation_fraction: float = .2, seed: int = 2027,
                            who_definition: str = LEGACY_WHO) -> dict:
    if who_definition not in WHO_MAPS:
        raise ValueError("Unknown WHO label definition")
    if not 0 < validation_fraction < .5:
        raise ValueError("validation_fraction must lie in (0, 0.5)")
    records = []
    for index, outer in enumerate(folds["folds"]):
        ids = sorted(outer["train_ids"])
        shuffled = np.random.default_rng(seed + index).permutation(ids).tolist()
        count = max(1, math.ceil(len(ids) * validation_fraction))
        split = {"fold": index, "fit_ids": sorted(shuffled[count:]),
                 "validation_ids": sorted(shuffled[:count]), "test_ids": sorted(outer["test_ids"])}
        _validate_split(split, outer)
        records.append(split)
    payload = {"version": VERSION, "seed": seed, "validation_fraction": validation_fraction,
               "split_rule": "label_blind_patient_random_split",
               "label_definition": {**LABEL_DEFINITION, "who": who_definition},
               "folds": records}
    return {**payload, "protocol_id": _digest(payload)}


def load_protocol_split(path: str | Path, outer: dict, fold: int, manifest: pd.DataFrame) -> dict:
    validate_binary_manifest(manifest)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    identifier = payload.pop("protocol_id", None)
    from glioma_meta.data.balanced_protocol import VERSION as BALANCED_VERSION, validate_balanced_payload
    if payload.get("version") not in (VERSION, BALANCED_VERSION) or identifier != _digest(payload):
        raise ValueError("Unrecognized or modified transfer protocol; regenerate under a new path")
    if payload['version'] == BALANCED_VERSION:
        validate_balanced_payload(payload, manifest)
    definition = {**LABEL_DEFINITION, "who": manifest_who_definition(manifest)}
    if payload.get("label_definition") != definition:
        raise ValueError("Protocol label definitions do not match this model")
    split = payload["folds"][fold]
    if split["fold"] != fold:
        raise ValueError("Protocol fold index mismatch")
    _validate_split(split, outer)
    if not (set(outer["train_ids"]) | set(outer["test_ids"])).issubset(set(manifest.patient_id)):
        raise ValueError("Protocol contains patients missing from the manifest")
    extra = ({key: payload[key] for key in ('split_rule', 'allocation_label_usage',
              'minimum_label_fraction', 'allocation_labels_sha256')}
             if payload['version'] == BALANCED_VERSION else {})
    return {**split, **extra, "protocol_id": identifier, "protocol_version": payload['version'],
            "label_definition": definition, "protocol_path": str(path)}


def target_budget_split(manifest: pd.DataFrame, split: dict, task: str, fraction: float, seed: int
                        ) -> tuple[list[str], list[str], list[str]]:
    if not 0 < fraction <= 1:
        raise ValueError("label_fraction must lie in (0, 1]")
    if fraction < split.get('minimum_label_fraction', 0):
        raise ValueError('Label-informed balanced allocation requires 100% target labels; use a label-blind protocol for low-label experiments')
    fit_pool = manifest[manifest.patient_id.isin(split["fit_ids"])].sort_values("patient_id").reset_index(drop=True)
    val_pool = manifest[manifest.patient_id.isin(split["validation_ids"])].sort_values("patient_id").reset_index(drop=True)
    total = len(fit_pool) + len(val_pool)
    budget = int(round(total * fraction))
    val_count = max(1, int(round(budget * len(val_pool) / total)))
    fit_count = budget - val_count
    if fit_count < 2 or val_count < 2:
        raise ValueError("Budget too small for fit and validation; increase label_fraction")
    fit = label_budget_ids(fit_pool, task, fit_count / len(fit_pool), seed)
    validation = label_budget_ids(val_pool, task, val_count / len(val_pool), seed + 100003)
    validate_task_labels(manifest[manifest.patient_id.isin(fit)], task, "target fit")
    validate_task_labels(manifest[manifest.patient_id.isin(validation)], task, "target validation")
    return sorted(fit + validation), fit, validation


def validate_task_labels(frame: pd.DataFrame, task: str, role: str) -> None:
    columns = [f"{task}_label"] if task in {"idh", "who"} else ["survival_time", "survival_event"]
    if frame.empty or frame[columns].isna().any().any():
        raise ValueError(f"{role}: missing {task} labels cannot be treated as negative")
    if f"{task}_mask" in frame and not frame[f"{task}_mask"].fillna(False).astype(bool).all():
        raise ValueError(f"{role}: unavailable {task} annotations in the selected budget")
    if task in {"idh", "who"}:
        if set(frame[f"{task}_label"].unique()) != {0, 1}:
            raise ValueError(f"{role}: both {task} classes are required; increase the label budget")
    else:
        if not set(frame.survival_event.unique()).issubset({0, 1}):
            raise ValueError(f"{role}: survival_event must be 0/1")
        times = frame.survival_time.to_numpy(float)
        events = frame.survival_event.to_numpy(int)
        comparable = (events[:, None] == 1) & (times[:, None] < times[None, :])
        if not np.isfinite(times).all() or (times < 0).any() or not comparable.any():
            raise ValueError(f"{role}: valid follow-up and comparable survival pairs are required")


def validate_source_checkpoint(checkpoint: dict, protocol: dict) -> None:
    split = checkpoint.get("split", {})
    if (split.get("protocol_id") != protocol["protocol_id"]
            or split.get("label_definition") != protocol["label_definition"]):
        raise ValueError("Source checkpoint lacks this patient-disjoint protocol; retrain the source model")
    if "source_refit_ids" in split or checkpoint.get("method") == "source_guided_full_budget_refit":
        raise ValueError("Full-budget source refit is incompatible with held-out validation patients")
    for key in ("fit_ids", "validation_ids", "test_ids"):
        if set(split.get(key, [])) != set(protocol[key]):
            raise ValueError(f"Source checkpoint {key} mismatch; do not reuse a legacy checkpoint")


def require_no_refit(config: dict, section: str, flag: str) -> None:
    if bool(config[section].get(flag, True)):
        raise ValueError(f"Patient-disjoint development requires {section}.{flag}: false")

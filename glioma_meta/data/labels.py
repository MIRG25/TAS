"""Clinical label parsing and leakage-safe patient-level folds."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from glioma_meta.utils import save_json
from glioma_meta.data.who_labels import LOW_HIGH_WHO, map_who_grades

MODALITIES = ("T1", "T2", "T1c", "FLAIR")
TASK_COLUMNS = {"idh": "idh_label", "who": "who_label", "survival": "survival_time"}


def _single_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern))
    files = [f for f in files if "_bias" not in f.name.lower()]
    return files[0] if len(files) == 1 else None


def resolve_image_paths(data_root: str | Path, patient_id: str) -> dict[str, str] | None:
    directory = Path(data_root) / f"{patient_id}_nifti"
    if not directory.is_dir():
        return None
    names = {
        "t1_path": _single_file(directory, f"{patient_id}_T1.nii*"),
        "t2_path": _single_file(directory, f"{patient_id}_T2.nii*"),
        "t1c_path": _single_file(directory, f"{patient_id}_T1c.nii*"),
        "flair_path": _single_file(directory, f"{patient_id}_FLAIR.nii*"),
        "tumor_mask_path": _single_file(directory, f"{patient_id}_tumor_segmentation.nii*"),
    }
    if any(names[key] is None for key in ("t1_path", "t2_path", "t1c_path", "flair_path")):
        return None
    # POSIX separators remain valid on Windows and keep CSV manifests portable
    # when the project and sibling data directory are copied to Linux.
    return {key: value.as_posix() if value is not None else "" for key, value in names.items()}


def build_manifest(metadata_csv: str | Path, data_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a manifest; missing labels stay missing rather than being imputed."""
    frame = pd.read_csv(metadata_csv)
    required = {"ID", "IDH", "WHO CNS Grade", "1-dead 0-alive", "OS"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

    frame["patient_id"] = frame["ID"].astype(str)
    # Preserve missing IDH values as missing.  Treating every value other than
    # "wildtype" as mutant would silently turn NaN into a positive label.
    idh = frame["IDH"].astype("string").str.strip()
    frame["idh_label"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame.loc[idh.str.lower().eq("wildtype").fillna(False), "idh_label"] = 0
    frame.loc[idh.notna() & idh.ne(""), "idh_label"] = frame.loc[idh.notna() & idh.ne(""), "idh_label"].fillna(1)
    # User-defined low/high endpoint: I/II versus III/IV.
    # Keep the original "WHO CNS Grade" column in the manifest for auditability.
    frame["who_label"] = map_who_grades(frame["WHO CNS Grade"])
    frame["who_label_definition"] = LOW_HIGH_WHO
    frame["survival_time"] = pd.to_numeric(frame["OS"], errors="coerce")
    frame["survival_event"] = pd.to_numeric(frame["1-dead 0-alive"], errors="coerce")

    paths = [resolve_image_paths(data_root, patient_id) for patient_id in frame["patient_id"]]
    # Keep rows with unresolved paths so they are auditable exclusions rather than a DataFrame construction error.
    path_frame = pd.DataFrame([item if item is not None else {} for item in paths], index=frame.index)
    valid_images = path_frame["t1_path"].notna() if "t1_path" in path_frame else pd.Series(False, index=frame.index)
    label_columns = ["idh_label", "who_label", "survival_time", "survival_event"]
    valid_labels = frame[label_columns].notna().all(axis=1)
    keep = valid_images & valid_labels
    manifest = pd.concat([frame.reset_index(drop=True), path_frame.reset_index(drop=True)], axis=1).loc[keep].copy()
    manifest["idh_label"] = manifest["idh_label"].astype(int)
    manifest["who_label"] = manifest["who_label"].astype(int)
    manifest["survival_event"] = manifest["survival_event"].astype(int)
    manifest = manifest.reset_index(drop=True)

    excluded = frame.loc[~keep, ["ID"]].copy()
    excluded["reason"] = np.select(
        [
            (~valid_images.loc[~keep]) & (~valid_labels.loc[~keep]),
            ~valid_images.loc[~keep],
            ~valid_labels.loc[~keep],
        ],
        [
            "missing_modalities_and_labels",
            "missing_one_or_more_required_modalities",
            "missing_one_or_more_required_study_labels",
        ],
        default="unknown_exclusion",
    )
    return manifest, excluded


def make_patient_folds(manifest: pd.DataFrame, n_splits: int, seed: int) -> dict:
    """Five patient-level folds stratified jointly by IDH, WHO and survival event."""
    if manifest["patient_id"].duplicated().any():
        raise ValueError("Patient IDs are not unique; patient-level splitting is unsafe.")
    # Composite labels preserve the rare IDH-mutant/grade combinations better than a single task.
    strata = (
        manifest["idh_label"].astype(str)
        + "_"
        + manifest["who_label"].astype(str)
        + "_"
        + manifest["survival_event"].astype(str)
    )
    counts = strata.value_counts()
    if counts.min() < n_splits:
        # Stable fallback retains the most clinically important binary class balance.
        strata = manifest["idh_label"].astype(str) + "_" + manifest["survival_event"].astype(str)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(manifest, strata)):
        folds.append(
            {
                "fold": fold,
                "train_ids": manifest.iloc[train_idx]["patient_id"].tolist(),
                "test_ids": manifest.iloc[test_idx]["patient_id"].tolist(),
            }
        )
    return {"seed": seed, "n_splits": n_splits, "folds": folds}


def write_dataset_artifacts(manifest: pd.DataFrame, excluded: pd.DataFrame, output_dir: str | Path, n_splits: int, seed: int) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    excluded.to_csv(output_dir / "excluded_patients.csv", index=False)
    folds = make_patient_folds(manifest, n_splits=n_splits, seed=seed)
    save_json(folds, output_dir / "folds.json")

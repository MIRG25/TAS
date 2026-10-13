"""Leakage-safe inner validation splits within an outer patient fold."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


STRATIFICATION_COLUMNS = {"idh": "idh_label", "who": "who_label"}


def _survival_strata(frame: pd.DataFrame, time_bins: int = 4) -> pd.Series:
    """Jointly stratify survival splits by censoring status and follow-up time."""
    if len(frame) < 2:
        return frame["survival_event"].astype(str)
    bins = min(max(int(time_bins), 1), len(frame))
    time_group = pd.qcut(
        frame["survival_time"], q=bins, labels=False, duplicates="drop"
    )
    return (
        frame["survival_event"].astype(int).astype(str)
        + "_t"
        + time_group.fillna(-1).astype(int).astype(str)
    )


def make_inner_validation_split(
    frame: pd.DataFrame,
    tasks: Sequence[str],
    fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Create a task-aware fit/validation split inside one outer-training fold.

    Survival uses both censoring status and follow-up-time quartiles.  Event-only
    stratification can produce a validation set with a materially different time
    distribution and a noisy, optimistically selected C-index.
    """
    if not 0.0 < fraction < 0.5:
        raise ValueError(f"validation fraction must be in (0, 0.5); got {fraction}")
    if frame["patient_id"].duplicated().any():
        raise ValueError("Patient IDs must be unique before creating an inner validation split.")
    unique_tasks = list(dict.fromkeys(tasks))
    unknown = set(unique_tasks) - {"idh", "who", "survival"}
    if unknown:
        raise ValueError(f"Unknown tasks for split stratification: {sorted(unknown)}")

    components: list[pd.Series] = []
    for task in unique_tasks:
        if task == "survival":
            components.append(_survival_strata(frame))
        else:
            components.append(frame[STRATIFICATION_COLUMNS[task]].astype(str))

    candidates = [pd.concat(components, axis=1).astype(str).agg("_".join, axis=1)]
    if "survival" in unique_tasks:
        # Preserve censoring balance if joint event/time strata become too rare.
        candidates.append(frame["survival_event"].astype(int).astype(str))
    candidates.extend(component.astype(str) for component in components)
    n_validation = max(1, int(np.ceil(fraction * len(frame))))
    n_fit = len(frame) - n_validation

    def can_stratify(candidate: pd.Series) -> bool:
        counts = candidate.value_counts()
        return (
            counts.min() >= 2
            and len(counts) <= n_validation
            and len(counts) <= n_fit
        )

    strata = next(
        (candidate for candidate in candidates if can_stratify(candidate)),
        candidates[-1],
    )

    indices = np.arange(len(frame))
    if can_stratify(strata):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        fit_indices, validation_indices = next(splitter.split(indices, strata))
    else:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(indices)
        n_validation = max(1, int(round(fraction * len(frame))))
        validation_indices = shuffled[:n_validation]
        fit_indices = shuffled[n_validation:]
    return (
        frame.iloc[fit_indices]["patient_id"].astype(str).tolist(),
        frame.iloc[validation_indices]["patient_id"].astype(str).tolist(),
    )

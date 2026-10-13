"""Prespecified binary decision-threshold diagnostic, not probability calibration."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


VERSION = "validation_balanced_accuracy_v1"
RULE = {
    "version": VERSION,
    "selection_data": "target_validation_only",
    "objective": "balanced_accuracy",
    "decision": "probability_class_1 > threshold; equality is class 0",
    "candidates": "distinct validation scores plus 0.5 and all-positive endpoint",
    "tie_break": "closest to 0.5, then larger threshold",
    "baseline_threshold": 0.5,
}


def checked_arrays(labels, probabilities):
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Non-empty binary labels are required; missing labels are not negatives")
    if (p.shape != (len(y), 2) or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0)):
        raise ValueError("Expected finite N x 2 probabilities in [0,1] summing to one")
    # Fail explicitly rather than silently changing the project's argmax tie rule.
    if not np.array_equal(p.argmax(1), (p[:, 1] > .5).astype(int)):
        raise ValueError("Floating-point boundary mismatch between argmax and threshold 0.5")
    return y.astype(int), p


def binary_metrics(labels, probabilities, threshold=.5):
    y, p = checked_arrays(labels, probabilities)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    pred = p[:, 1] > threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    sen = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    return {
        "acc": (tp + tn) / len(y),
        "auc": float(roc_auc_score(y, p[:, 1])) if len(np.unique(y)) == 2 else None,
        "sensitivity": sen, "specificity": spec,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "balanced_accuracy": (sen + spec) / 2 if sen is not None and spec is not None else None,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def select_validation_threshold(validation_labels, validation_probabilities):
    """Accept validation data only. No test arrays or target metrics as arguments.

    The endpoint just below the smallest score permits the all-positive rule,
    even when the smallest score is zero. Selection comparisons use integer
    numerators, avoiding floating-point objective ties. 0.5 is always included.
    """
    y, p = checked_arrays(validation_labels, validation_probabilities)
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        raise ValueError("Validation must contain both classes; refusing test-derived or default fallback")
    candidates = np.unique(np.r_[p[:, 1], .5, np.nextafter(p[:, 1].min(), -np.inf)])
    rows = []
    for t in candidates:
        m = binary_metrics(y, p, float(t))
        rows.append({"threshold": float(t), **{k: m[k] for k in (
            "balanced_accuracy", "acc", "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp")},
            "objective_numerator": m["tp"] * negatives + m["tn"] * positives})
    selected = min(rows, key=lambda r: (-r["objective_numerator"], abs(r["threshold"] - .5), -r["threshold"]))
    for row in rows:
        row["selected"] = row["threshold"] == selected["threshold"]
    lock = {"rule": dict(RULE), "threshold": selected["threshold"],
            "validation_class_counts": {"0": negatives, "1": positives},
            "validation_default_metrics": binary_metrics(y, p),
            "validation_threshold_metrics": binary_metrics(y, p, selected["threshold"]),
            "candidate_count": len(rows)}
    return lock, rows

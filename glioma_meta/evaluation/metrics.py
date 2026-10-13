"""Research metrics required by the study protocol."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from glioma_meta.training.losses import SurvivalDiscretizer


DEFAULT_INTEGRATED_AUC_RANGE_DAYS = (365.0, 1825.0)
DEFAULT_INTEGRATED_BRIER_RANGE_DAYS = (365.0, 1825.0)
DEFAULT_SURVIVAL_AUC_HORIZONS_DAYS = (365.0, 1095.0, 1825.0)


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def classification_metrics(target: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    target = np.asarray(target, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities.argmax(axis=1)
    n_classes = probabilities.shape[1]
    result: dict[str, float | None] = {
        "acc": float(accuracy_score(target, predictions)),
        "f1": float(f1_score(target, predictions, average="binary" if n_classes == 2 else "macro", zero_division=0)),
    }
    try:
        result["auc"] = _finite(
            roc_auc_score(target, probabilities[:, 1])
            if n_classes == 2
            else roc_auc_score(target, probabilities, multi_class="ovr", average="macro")
        )
    except ValueError:
        result["auc"] = None
    matrix = confusion_matrix(target, predictions, labels=np.arange(n_classes))
    sensitivities, specificities = [], []
    for label in range(n_classes):
        tp = matrix[label, label]
        fn = matrix[label].sum() - tp
        fp = matrix[:, label].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        sensitivities.append(tp / (tp + fn) if tp + fn else np.nan)
        specificities.append(tn / (tn + fp) if tn + fp else np.nan)
    if n_classes == 2:
        result["sensitivity"] = _finite(sensitivities[1])
        result["specificity"] = _finite(specificities[1])
    else:
        result["sensitivity"] = _finite(float(np.nanmean(sensitivities)))
        result["specificity"] = _finite(float(np.nanmean(specificities)))
    return result


def harrell_c_index(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> float | None:
    """Harrell C-index; higher risk should correspond to shorter survival."""
    times, events, risks = map(np.asarray, (times, events, risks))
    concordant = comparable = 0.0
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if times[i] == times[j]:
                continue
            if times[i] < times[j] and events[i] == 1:
                comparable += 1
                concordant += 1 if risks[i] > risks[j] else 0.5 if risks[i] == risks[j] else 0
            elif times[j] < times[i] and events[j] == 1:
                comparable += 1
                concordant += 1 if risks[j] > risks[i] else 0.5 if risks[i] == risks[j] else 0
    return _finite(concordant / comparable) if comparable else None


def survival_probability_from_logits(logits: np.ndarray) -> np.ndarray:
    hazards = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    return np.cumprod(1.0 - hazards, axis=1)


def _step_probability(
    step_times: np.ndarray,
    step_probabilities: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Evaluate a right-continuous step function, with value one before its first step."""
    indices = np.searchsorted(step_times, query_times, side="right") - 1
    result = np.ones(np.asarray(query_times).shape, dtype=float)
    valid = indices >= 0
    result[valid] = step_probabilities[indices[valid]]
    return result


def _kaplan_meier_probabilities(
    times: np.ndarray,
    events: np.ndarray,
    query_times: np.ndarray,
    reverse: bool = False,
) -> np.ndarray:
    """Kaplan-Meier event survival, or reverse-KM censoring survival."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    unique_times = np.unique(times)
    probabilities = []
    current = 1.0
    for time in unique_times:
        at_risk = int(np.sum(times >= time))
        deaths = int(np.sum((times == time) & events))
        censored = int(np.sum((times == time) & ~events))
        if reverse:
            # With tied event/censoring times, observed events leave the risk set
            # before estimating the censoring hazard (reverse Kaplan-Meier).
            denominator = at_risk - deaths
            failures = censored
        else:
            denominator = at_risk
            failures = deaths
        if denominator > 0 and failures > 0:
            current *= 1.0 - failures / denominator
        probabilities.append(current)
    return _step_probability(unique_times, np.asarray(probabilities), np.asarray(query_times, dtype=float))


def _survival_risk_at_times(
    survival_probabilities: np.ndarray,
    discretizer: SurvivalDiscretizer,
    evaluation_times: np.ndarray,
) -> np.ndarray:
    """Return 1-S(t), matching a horizon to the interval ending at that cutpoint."""
    indices = np.searchsorted(discretizer.cutpoints, evaluation_times, side="left")
    indices = np.clip(indices, 0, survival_probabilities.shape[1] - 1)
    return 1.0 - survival_probabilities[:, indices]


def _weighted_cumulative_dynamic_auc(
    times: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
    evaluation_time: float,
    ipcw: np.ndarray,
    tied_tolerance: float = 1e-8,
) -> float | None:
    cases = (events == 1) & (times <= evaluation_time)
    controls = times > evaluation_time
    if not np.any(cases) or not np.any(controls):
        return None
    case_weights = ipcw[cases]
    if not np.any(case_weights > 0):
        return None
    differences = risks[cases, None] - risks[controls][None, :]
    concordance = (differences > tied_tolerance).astype(float)
    concordance[np.abs(differences) <= tied_tolerance] = 0.5
    value = np.sum(case_weights[:, None] * concordance) / (case_weights.sum() * controls.sum())
    return _finite(value)


def integrated_time_dependent_auc(
    times: np.ndarray,
    events: np.ndarray,
    survival_probabilities: np.ndarray,
    discretizer: SurvivalDiscretizer,
    reference_times: np.ndarray | None = None,
    reference_events: np.ndarray | None = None,
    time_range: tuple[float, float] = DEFAULT_INTEGRATED_AUC_RANGE_DAYS,
    audit_horizons: Iterable[float] = DEFAULT_SURVIVAL_AUC_HORIZONS_DAYS,
) -> dict[str, float | None]:
    """IPCW cumulative/dynamic AUC integrated over a prespecified time range.

    The integration measure is the Kaplan-Meier event-survival distribution of
    the evaluated cohort. The censoring distribution is estimated only from
    ``reference_times/reference_events`` (normally the corresponding outer
    training fold), never from model predictions or test-fold tuning.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    survival_probabilities = np.asarray(survival_probabilities, dtype=float)
    if len(times) != len(events) or survival_probabilities.shape[0] != len(times):
        raise ValueError("Survival outcomes and predictions must contain the same number of patients")
    start, end = map(float, time_range)
    if not 0 <= start < end:
        raise ValueError("Integrated-AUC time_range must satisfy 0 <= start < end")
    audit_horizons = tuple(float(value) for value in audit_horizons)
    if reference_times is None or reference_events is None:
        reference_times, reference_events = times, events
    reference_times = np.asarray(reference_times, dtype=float)
    reference_events = np.asarray(reference_events, dtype=int)
    if len(reference_times) != len(reference_events):
        raise ValueError("Reference survival times and events must have equal length")

    event_ipcw = np.zeros(len(times), dtype=float)
    maximum_evaluation_time = max(end, *audit_horizons)
    event_rows = (events == 1) & (times <= maximum_evaluation_time)
    if np.any(event_rows):
        censoring_survival = _kaplan_meier_probabilities(
            reference_times,
            reference_events,
            times[event_rows],
            reverse=True,
        )
        if np.any(censoring_survival <= 0):
            raise ValueError(
                "The reference-fold censoring survival is zero within the requested AUC range; "
                "shorten evaluation.integrated_auc_time_range_days."
            )
        event_ipcw[event_rows] = 1.0 / censoring_survival

    # AUC(t) is a step function that may change at each observed event time.
    integration_times = np.unique(times[(events == 1) & (times > start) & (times <= end)])
    integrated_auc: float | None = None
    if integration_times.size:
        risks = _survival_risk_at_times(survival_probabilities, discretizer, integration_times)
        auc_values = np.asarray(
            [
                _weighted_cumulative_dynamic_auc(times, events, risks[:, index], time, event_ipcw)
                for index, time in enumerate(integration_times)
            ],
            dtype=float,
        )
        survival_at_start = float(_kaplan_meier_probabilities(times, events, np.asarray([start]))[0])
        survival_at_events = _kaplan_meier_probabilities(times, events, integration_times)
        survival_at_end = float(_kaplan_meier_probabilities(times, events, np.asarray([end]))[0])
        probability_mass = -np.diff(np.r_[survival_at_start, survival_at_events])
        denominator = survival_at_start - survival_at_end
        if denominator > 0 and np.all(np.isfinite(auc_values)):
            integrated_auc = _finite(float(np.sum(auc_values * probability_mass) / denominator))

    results: dict[str, float | None] = {"integrated_auc": integrated_auc}
    horizons = np.asarray(audit_horizons, dtype=float)
    if horizons.size:
        horizon_risks = _survival_risk_at_times(survival_probabilities, discretizer, horizons)
        for index, horizon in enumerate(horizons):
            results[f"auc_at_{horizon:.0f}d"] = _weighted_cumulative_dynamic_auc(
                times,
                events,
                horizon_risks[:, index],
                horizon,
                event_ipcw,
            )
    return results


def integrated_brier_score(
    times: np.ndarray,
    events: np.ndarray,
    survival_probabilities: np.ndarray,
    discretizer: SurvivalDiscretizer,
    reference_times: np.ndarray | None = None,
    reference_events: np.ndarray | None = None,
    time_range: tuple[float, float] = DEFAULT_INTEGRATED_BRIER_RANGE_DAYS,
) -> float | None:
    """IPCW integrated Brier score over a prespecified time interval.

    At each evaluation time, observed events are weighted by the inverse
    censoring survival immediately before their event time and patients still
    under observation are weighted by the inverse censoring survival at the
    evaluation time. Patients censored before the evaluation time contribute
    zero, as in the standard IPCW Brier score. The censoring distribution is
    estimated exclusively from the supplied reference fold.

    The score is evaluated at the discrete survival model's cutpoints inside
    ``time_range`` (including both range boundaries) and integrated with the
    trapezoidal rule. Lower values indicate better calibrated survival
    probabilities; zero is perfect.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    survival_probabilities = np.asarray(survival_probabilities, dtype=float)
    if len(times) != len(events) or survival_probabilities.shape[0] != len(times):
        raise ValueError("Survival outcomes and predictions must contain the same number of patients")
    start, end = map(float, time_range)
    if not 0 <= start < end:
        raise ValueError("Integrated-Brier time_range must satisfy 0 <= start < end")
    if reference_times is None or reference_events is None:
        reference_times, reference_events = times, events
    reference_times = np.asarray(reference_times, dtype=float)
    reference_events = np.asarray(reference_events, dtype=int)
    if len(reference_times) != len(reference_events):
        raise ValueError("Reference survival times and events must have equal length")

    internal_cutpoints = discretizer.cutpoints[
        (discretizer.cutpoints > start) & (discretizer.cutpoints < end)
    ]
    evaluation_times = np.unique(np.r_[start, internal_cutpoints, end]).astype(float)
    predicted_survival = 1.0 - _survival_risk_at_times(
        survival_probabilities,
        discretizer,
        evaluation_times,
    )

    censoring_at_times = _kaplan_meier_probabilities(
        reference_times,
        reference_events,
        evaluation_times,
        reverse=True,
    )
    # Event terms require G(T_i-). np.nextafter obtains the left limit of the
    # right-continuous reverse-KM step function without an arbitrary epsilon.
    event_left_times = np.nextafter(times, -np.inf)
    censoring_before_event = _kaplan_meier_probabilities(
        reference_times,
        reference_events,
        event_left_times,
        reverse=True,
    )
    if np.any(censoring_at_times <= 0):
        return None

    scores = np.empty(len(evaluation_times), dtype=float)
    for index, evaluation_time in enumerate(evaluation_times):
        cases = (events == 1) & (times <= evaluation_time)
        controls = times > evaluation_time
        if np.any(cases & (censoring_before_event <= 0)):
            return None
        terms = np.zeros(len(times), dtype=float)
        if np.any(cases):
            terms[cases] = (
                np.square(predicted_survival[cases, index])
                / censoring_before_event[cases]
            )
        if np.any(controls):
            terms[controls] = (
                np.square(1.0 - predicted_survival[controls, index])
                / censoring_at_times[index]
            )
        scores[index] = float(np.mean(terms))

    value = np.trapz(scores, evaluation_times) / (end - start)
    return _finite(float(value))


def _km_curve(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    event_times = np.unique(times[events == 1])
    curve_times, survival = [0.0], [1.0]
    current = 1.0
    for time in event_times:
        at_risk = np.sum(times >= time)
        deaths = np.sum((times == time) & (events == 1))
        if at_risk:
            current *= 1.0 - deaths / at_risk
        curve_times.extend([time, time])
        survival.extend([survival[-1], current])
    return np.asarray(curve_times), np.asarray(survival)


def log_rank_test(times: np.ndarray, events: np.ndarray, groups: np.ndarray) -> dict[str, float | None]:
    """Two-group log-rank test using observed/expected deaths at each event time."""
    times, events, groups = map(np.asarray, (times, events, groups))
    if len(np.unique(groups)) != 2:
        return {"log_rank_chi2": None, "log_rank_p": None}
    observed_minus_expected = variance = 0.0
    for time in np.unique(times[events == 1]):
        n1 = np.sum((times >= time) & (groups == 1)); n0 = np.sum((times >= time) & (groups == 0))
        d1 = np.sum((times == time) & (events == 1) & (groups == 1)); d0 = np.sum((times == time) & (events == 1) & (groups == 0))
        n, deaths = n1 + n0, d1 + d0
        if n <= 1 or deaths == 0:
            continue
        expected = deaths * n1 / n
        variance_t = n1 * n0 * deaths * (n - deaths) / (n**2 * (n - 1))
        observed_minus_expected += d1 - expected
        variance += variance_t
    statistic = observed_minus_expected**2 / variance if variance > 0 else np.nan
    return {"log_rank_chi2": _finite(statistic), "log_rank_p": _finite(chi2.sf(statistic, 1))}


def evaluate_survival(
    times: np.ndarray,
    events: np.ndarray,
    logits: np.ndarray,
    discretizer: SurvivalDiscretizer,
    reference_times: np.ndarray | None = None,
    reference_events: np.ndarray | None = None,
    integrated_auc_time_range: tuple[float, float] = DEFAULT_INTEGRATED_AUC_RANGE_DAYS,
    integrated_brier_time_range: tuple[float, float] = DEFAULT_INTEGRATED_BRIER_RANGE_DAYS,
    auc_audit_horizons: Iterable[float] = DEFAULT_SURVIVAL_AUC_HORIZONS_DAYS,
    risk_horizon: float = 1825.0,
) -> tuple[dict, np.ndarray, np.ndarray]:
    survival = survival_probability_from_logits(logits)
    # Use a prespecified clinically meaningful horizon for the scalar risk.
    # The final open-ended output interval can be weakly supervised or entirely
    # unobserved in a fold; including it would inject arbitrary tail logits into
    # C-index and Kaplan-Meier risk grouping.
    risk_index = min(
        int(np.searchsorted(discretizer.cutpoints, float(risk_horizon), side="left")),
        survival.shape[1] - 1,
    )
    risks = -np.log(np.clip(survival[:, risk_index], 1e-8, 1.0))
    groups = (risks >= np.median(risks)).astype(int)
    metrics: dict = {"c_index": harrell_c_index(times, events, risks)}
    metrics.update(
        integrated_time_dependent_auc(
            times,
            events,
            survival,
            discretizer,
            reference_times=reference_times,
            reference_events=reference_events,
            time_range=integrated_auc_time_range,
            audit_horizons=auc_audit_horizons,
        )
    )
    metrics["integrated_brier_score"] = integrated_brier_score(
        times,
        events,
        survival,
        discretizer,
        reference_times=reference_times,
        reference_events=reference_events,
        time_range=integrated_brier_time_range,
    )
    metrics.update(log_rank_test(times, events, groups))
    return metrics, risks, groups


def save_kaplan_meier_plot(times: np.ndarray, events: np.ndarray, groups: np.ndarray, output_path: str | Path, title: str = "Predicted-risk Kaplan-Meier") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    for group, label, color in [(0, "Predicted low risk", "#2171b5"), (1, "Predicted high risk", "#cb181d")]:
        selected = groups == group
        x, y = _km_curve(np.asarray(times)[selected], np.asarray(events)[selected])
        axis.step(x, y, where="post", label=f"{label} (n={selected.sum()})", color=color, linewidth=2)
    axis.set(title=title, xlabel="Overall survival (days)", ylabel="Survival probability", ylim=(0, 1.05))
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

"""Masked task losses, including censoring-aware discrete-time survival NLL."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SurvivalDiscretizer:
    """Maps continuous OS days to a fixed set of discrete survival intervals."""

    cutpoints: np.ndarray
    loss_version: str = "legacy-v1"

    def __post_init__(self):
        if self.loss_version not in ("legacy-v1", "endpoint-v2"):
            raise ValueError(f"Unknown survival loss version: {self.loss_version!r}")
        if self.loss_version == "endpoint-v2":
            cuts = np.asarray(self.cutpoints)
            if (cuts.ndim != 1 or not len(cuts) or not np.isfinite(cuts).all()
                    or (cuts <= 0).any() or (np.diff(cuts) <= 0).any()):
                raise ValueError("Endpoint survival cutpoints must be finite, positive and strictly increasing")

    @classmethod
    def fit(cls, times: np.ndarray, n_bins: int) -> "SurvivalDiscretizer":
        times = np.asarray(times, dtype=float)
        if n_bins < 2 or len(times) < n_bins:
            raise ValueError("Need at least two survival bins and at least n_bins training cases.")
        quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
        cutpoints = np.quantile(times, quantiles)
        if len(np.unique(cutpoints)) != len(cutpoints):
            cutpoints = np.linspace(float(times.min()), float(times.max()), n_bins + 1)[1:-1]
        return cls(np.asarray(cutpoints, dtype=np.float32))

    @property
    def n_bins(self) -> int:
        return len(self.cutpoints) + 1

    def transform_numpy(self, times: np.ndarray) -> np.ndarray:
        side = "left" if self.loss_version == "endpoint-v2" else "right"
        return np.searchsorted(self.cutpoints, np.asarray(times), side=side).clip(0, self.n_bins - 1).astype(np.int64)

    def transform_tensor(self, times: torch.Tensor) -> torch.Tensor:
        edges = torch.as_tensor(self.cutpoints, dtype=times.dtype, device=times.device)
        return torch.bucketize(times, edges, right=self.loss_version == "legacy-v1").clamp(max=self.n_bins - 1)

    def to_dict(self) -> dict:
        result = {"cutpoints": self.cutpoints.tolist()}
        if self.loss_version != "legacy-v1":
            result["loss_version"] = self.loss_version
        return result

    @classmethod
    def from_dict(cls, value: dict) -> "SurvivalDiscretizer":
        return cls(np.asarray(value["cutpoints"], dtype=np.float32), value.get("loss_version", "legacy-v1"))


def discrete_time_nll(logits: torch.Tensor, times: torch.Tensor, events: torch.Tensor, discretizer: SurvivalDiscretizer) -> torch.Tensor:
    """Negative log-likelihood of a discrete hazard model for events and censoring."""
    if logits.ndim != 2 or logits.shape[1] != discretizer.n_bins:
        raise ValueError("Survival logits do not match the fitted discretizer.")
    if discretizer.loss_version == "endpoint-v2":
        if (times.shape != (len(logits),) or events.shape != times.shape or not len(logits)
                or not torch.isfinite(times).all() or (times < 0).any()
                or not ((events == 0) | (events == 1)).all()):
            raise ValueError("Endpoint survival loss requires nonempty finite nonnegative times and binary events")
    interval = discretizer.transform_tensor(times).long()
    log_hazard = F.logsigmoid(logits)
    log_survival = F.logsigmoid(-logits)
    cumulative_survival = torch.cumsum(log_survival, dim=1)
    # torch.where evaluates both branches, so clamp before gathering interval zero.
    previous_index = (interval - 1).clamp_min(0)
    previous_survival = cumulative_survival.gather(1, previous_index.unsqueeze(1)).squeeze(1)
    previous_survival = torch.where(interval > 0, previous_survival, torch.zeros_like(times))
    event_log_likelihood = previous_survival + log_hazard.gather(1, interval.unsqueeze(1)).squeeze(1)
    if discretizer.loss_version == "endpoint-v2":
        # Events round UP to an endpoint; censors round DOWN. A censor before
        # the first endpoint contributes log(1)=0, but remains in the patient
        # mean. The open terminal bin has no finite endpoint and is never
        # counted as fully survived. Do not supervise an unobserved interval.
        edges = torch.as_tensor(discretizer.cutpoints, dtype=times.dtype, device=times.device)
        completed = torch.bucketize(times, edges, right=True)
        padded = F.pad(cumulative_survival, (1, 0), value=0.0)
        censor_log_likelihood = padded.gather(1, completed.unsqueeze(1)).squeeze(1)
    else:
        censor_log_likelihood = cumulative_survival.gather(1, interval.unsqueeze(1)).squeeze(1)
    return -torch.where(events > 0.5, event_log_likelihood, censor_log_likelihood).mean()


def task_loss(
    task: str,
    logits: torch.Tensor,
    batch: dict,
    discretizer: SurvivalDiscretizer | None = None,
    class_weights: torch.Tensor | None = None,
    *,
    classification_reduction: str = "batch-weight-mean",
) -> torch.Tensor:
    if classification_reduction not in {"batch-weight-mean", "fit-patient-mean"}:
        raise ValueError(f"Unknown classification reduction: {classification_reduction}")
    if task in {"idh", "who"}:
        mask = batch.get(f"{task}_mask", torch.ones_like(batch[task], dtype=torch.bool)).bool()
        if not mask.any():
            return logits.sum() * 0.0
        if classification_reduction == "fit-patient-mean":
            # Weights must be normalized ONCE using target-fit class frequencies.
            # Never divide by this minibatch's random sum of class weights.
            return F.cross_entropy(
                logits[mask], batch[task][mask].long(), weight=class_weights, reduction="none"
            ).mean()
        return F.cross_entropy(logits[mask], batch[task][mask].long(), weight=class_weights)
    if task == "survival":
        if classification_reduction != "batch-weight-mean":
            raise ValueError("Classification weighting ablation does not change survival loss")
        if discretizer is None:
            raise ValueError("Survival loss needs a fitted SurvivalDiscretizer.")
        mask = batch.get("survival_mask", torch.ones_like(batch["survival_event"], dtype=torch.bool)).bool()
        if not mask.any():
            return logits.sum() * 0.0
        return discrete_time_nll(
            logits[mask], batch["survival_time"][mask], batch["survival_event"][mask], discretizer
        )
    raise ValueError(f"Unknown task: {task}")

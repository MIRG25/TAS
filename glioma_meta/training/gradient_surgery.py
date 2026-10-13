"""Gradient-conflict diagnostics and deterministic PCGrad projection."""
from __future__ import annotations

from collections.abc import Sequence

import torch


GradientTuple = tuple[torch.Tensor | None, ...]


def _dot(left: GradientTuple, right: GradientTuple) -> torch.Tensor:
    reference = next(
        (value for value in (*left, *right) if value is not None), None
    )
    if reference is None:
        raise ValueError("At least one gradient tensor is required")
    result = reference.new_zeros((), dtype=torch.float32)
    for left_value, right_value in zip(left, right):
        if left_value is not None and right_value is not None:
            result = result + (left_value.float() * right_value.float()).sum()
    return result


def _norm_squared(gradients: GradientTuple) -> torch.Tensor:
    return _dot(gradients, gradients)


def gradient_cosine(left: GradientTuple, right: GradientTuple) -> float:
    denominator = torch.sqrt(_norm_squared(left) * _norm_squared(right)).clamp_min(
        torch.finfo(torch.float32).eps
    )
    return float((_dot(left, right) / denominator).detach().cpu())


def _sum_gradients(gradients: Sequence[GradientTuple]) -> GradientTuple:
    if not gradients:
        raise ValueError("At least one task gradient is required")
    combined: list[torch.Tensor | None] = []
    for values in zip(*gradients):
        present = [value for value in values if value is not None]
        combined.append(sum(present) if present else None)
    return tuple(combined)


def pcgrad_project(
    task_gradients: Sequence[GradientTuple],
) -> tuple[GradientTuple, GradientTuple, dict[str, float]]:
    """Project conflicting task gradients before updating shared parameters.

    Returns the projected sum, the original sum, and pairwise diagnostics. The
    original sum lets a caller replace only the task-loss component of an
    already-computed gradient while retaining regularizer gradients.
    """
    if len(task_gradients) < 2:
        original = _sum_gradients(task_gradients)
        return original, original, {"mean_cosine": 1.0, "minimum_cosine": 1.0, "conflict_fraction": 0.0}

    originals = [
        tuple(value.detach().clone() if value is not None else None for value in gradients)
        for gradients in task_gradients
    ]
    projected = [list(gradients) for gradients in originals]
    cosines: list[float] = []
    conflicts = 0
    comparisons = 0
    for left_index in range(len(originals)):
        for right_index in range(left_index + 1, len(originals)):
            cosine = gradient_cosine(originals[left_index], originals[right_index])
            cosines.append(cosine)
            conflicts += int(cosine < 0.0)
            comparisons += 1

    epsilon = torch.finfo(torch.float32).eps
    for left_index, left in enumerate(originals):
        for right_index, right in enumerate(originals):
            if left_index == right_index:
                continue
            current = tuple(projected[left_index])
            dot = _dot(current, right)
            if float(dot.detach()) >= 0.0:
                continue
            coefficient = dot / _norm_squared(right).clamp_min(epsilon)
            for parameter_index, (left_value, right_value) in enumerate(
                zip(projected[left_index], right)
            ):
                if left_value is not None and right_value is not None:
                    projected[left_index][parameter_index] = (
                        left_value - coefficient.to(left_value.dtype) * right_value
                    )

    projected_sum = _sum_gradients([tuple(values) for values in projected])
    original_sum = _sum_gradients(originals)
    diagnostics = {
        "mean_cosine": sum(cosines) / len(cosines),
        "minimum_cosine": min(cosines),
        "conflict_fraction": conflicts / max(comparisons, 1),
    }
    return projected_sum, original_sum, diagnostics


def replace_task_gradients(
    parameters: Sequence[torch.nn.Parameter],
    projected: GradientTuple,
    original: GradientTuple,
) -> None:
    """Replace summed task gradients while preserving other loss gradients."""
    for parameter, projected_value, original_value in zip(
        parameters, projected, original
    ):
        if projected_value is None:
            continue
        if parameter.grad is None:
            parameter.grad = projected_value.clone()
        elif original_value is None:
            parameter.grad.add_(projected_value)
        else:
            parameter.grad.add_(projected_value - original_value)

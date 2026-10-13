"""Target-only second-order residual; no labels, batch statistics or extra views."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CovarianceResidualReadout(nn.Module):
    """Keep the original spatial mean and learn a low-rank covariance correction.

    Input is [patient, spatial token, feature], after the existing frozen LN,
    without CLS. Covariance is centered *within each patient*, divided by N,
    trace-normalized, then upper-triangle vectorized (sqrt(2) off diagonals).
    log1p(trace) preserves the scale removed by normalization. No matrix square
    root, eigen-decomposition, second encoder, or biological labels are used.
    Zero output weights preserve the exact original readout at initialization.
    """

    VERSION = "covariance_residual_v1"

    def __init__(self, feature_dim: int, rank: int = 32) -> None:
        super().__init__()
        if (isinstance(rank, bool) or not isinstance(rank, int)
                or not 1 <= rank <= feature_dim):
            raise ValueError("covariance_rank must be an integer in [1, feature_dim]")
        self.feature_dim, self.rank = feature_dim, rank
        self.eps = 1e-6
        # Adding this optional module must not move the legacy RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Linear(feature_dim, rank, bias=False)
        indices = torch.triu_indices(rank, rank)
        scales = torch.where(indices[0] == indices[1], 1.0, math.sqrt(2.0))
        self.register_buffer("indices", indices, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        self.output_weight = nn.Parameter(torch.zeros(feature_dim, indices.shape[1] + 1))

    def descriptor(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.feature_dim or tokens.shape[1] < 1:
            raise ValueError("Expected nonempty [batch, spatial tokens, feature_dim]")
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            projected = F.linear(tokens.float(), self.projection.weight.float())
            centered = projected - projected.mean(dim=1, keepdim=True)
            covariance = centered.transpose(1, 2) @ centered / tokens.shape[1]
            trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)
            normalized = covariance / (trace.unsqueeze(-1) + self.eps)
            vector = normalized[:, self.indices[0], self.indices[1]] * self.scales
            return torch.cat((vector, torch.log1p(trace)), dim=-1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        descriptor = self.descriptor(tokens)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            residual = F.linear(descriptor, self.output_weight.float())
        # Use exactly the baseline mean operation and dtype, including at init.
        return tokens.mean(dim=1) + residual.to(tokens.dtype)

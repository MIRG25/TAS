"""A spatial residual extension that preserves a copied clinical adapter."""
from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral

import torch
import torch.nn.functional as F
from torch import nn

from .multitask import ResidualFeatureAdapter


class SpatialResidualFeatureAdapter(nn.Module):
    """Reuse a target adapter; add zero local mixing before its nonlinearity.

    The fixed identity is outside the trainable convolution, so weight decay
    acts only on the learned delta. Construction consumes no random numbers.
    The supplied adapter must be a target-owned copy, never the source module.
    """

    def __init__(self, base: ResidualFeatureAdapter, grid_shape: Sequence[int]) -> None:
        super().__init__()
        grid = tuple(grid_shape)
        if len(grid) != 3 or any(
            isinstance(v, bool) or not isinstance(v, Integral) or v <= 0 for v in grid
        ):
            raise ValueError('grid_shape must contain three positive integers')
        self.grid_shape = tuple(int(v) for v in grid)
        self.norm, self.down, self.up = base.norm, base.down, base.up
        self.spatial_weight = nn.Parameter(base.down.weight.new_zeros(
            base.down.out_features, 1, 3, 3, 3
        ))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        expected = (1 + math.prod(self.grid_shape), self.down.in_features)
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(f'Expected tokens [B,{expected[0]},{expected[1]}]; got {tuple(tokens.shape)}')
        hidden = self.down(self.norm(tokens))
        spatial = hidden[:, 1:].transpose(1, 2).reshape(
            tokens.shape[0], self.down.out_features, *self.grid_shape
        )
        spatial = spatial + F.conv3d(
            spatial, self.spatial_weight, padding=1, groups=self.down.out_features
        )
        hidden = torch.cat([hidden[:, :1], spatial.flatten(2).transpose(1, 2)], dim=1)
        return tokens + self.up(F.gelu(hidden))

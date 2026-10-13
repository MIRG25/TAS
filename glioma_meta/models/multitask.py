"""Supervised MRI baselines built around a shared 3-D encoder."""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import MRIEncoder3D


class ResidualFeatureAdapter(nn.Module):
    """Zero-initialized bottleneck adapter for parameter-efficient fine-tuning."""

    def __init__(self, feature_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, feature_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        adapted = self.up(F.gelu(self.down(self.norm(features))))
        return features + adapted


class GliomaMultiTaskModel(nn.Module):
    def __init__(
        self,
        base_channels: int = 24,
        feature_dim: int = 192,
        dropout: float = 0.2,
        survival_bins: int = 12,
        adapter_dim: int = 48,
        tasks: Sequence[str] | None = None,
        pooling: str = "average",
        attention_hidden_channels: int = 64,
        feature_norm: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = MRIEncoder3D(
            base_channels=base_channels,
            feature_dim=feature_dim,
            dropout=dropout,
            pooling=pooling,
            attention_hidden_channels=attention_hidden_channels,
            feature_norm=feature_norm,
        )
        self.feature_dim = feature_dim
        self.survival_bins = survival_bins
        output_dims = {"idh": 2, "who": 2, "survival": survival_bins}
        requested_tasks = list(dict.fromkeys(tasks or output_dims.keys()))
        unknown = set(requested_tasks) - set(output_dims)
        if unknown:
            raise ValueError(f"Unknown task heads requested: {sorted(unknown)}")

        self.adapter = ResidualFeatureAdapter(feature_dim=feature_dim, bottleneck_dim=adapter_dim)
        self.heads = nn.ModuleDict({task: nn.Linear(feature_dim, output_dims[task]) for task in requested_tasks})

    def encode_backbone(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.encode_backbone(image))

    def forward_task(self, image: torch.Tensor, task: str) -> torch.Tensor:
        return self.heads[task](self.encode(image))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode(image)
        return {name: head(features) for name, head in self.heads.items()}

    def new_head(self, task: str) -> nn.Linear:
        output_dim = {"idh": 2, "who": 2, "survival": self.survival_bins}[task]
        return nn.Linear(self.feature_dim, output_dim)

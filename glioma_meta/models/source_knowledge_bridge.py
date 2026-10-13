"""Patient-conditioned retrieval and relation mapping of frozen source features."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SourceKnowledgeBridge(nn.Module):
    """A zero-initialized residual, not a fixed weighted sum of source experts.

    Memory contains one frozen shared vector and the frozen private residuals
    of the source tasks, all computed on this patient's same MRI crop. A null
    slot allows refusal of source evidence. Shared projections make the module
    independent of the number/order and output label spaces of the sources.
    There are no batch statistics, stored patient vectors, or source labels.
    """

    def __init__(self, feature_dim: int, bottleneck_dim: int = 64) -> None:
        super().__init__()
        if (isinstance(bottleneck_dim, bool) or not isinstance(bottleneck_dim, int)
                or bottleneck_dim < 1):
            raise ValueError("source_bridge_dim must be a positive integer")
        self.feature_dim = feature_dim
        self.bottleneck_dim = bottleneck_dim
        self.query = nn.Linear(feature_dim, bottleneck_dim, bias=False)
        self.key = nn.Linear(feature_dim, bottleneck_dim, bias=False)
        self.value = nn.Linear(feature_dim, bottleneck_dim, bias=False)
        self.null_logit = nn.Parameter(torch.zeros(()))
        self.relation = nn.Linear(2 * bottleneck_dim, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, feature_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def _retrieve(self, target: torch.Tensor, memory: torch.Tensor):
        if (target.ndim != 2 or memory.ndim != 3 or memory.shape[0] != target.shape[0]
                or memory.shape[1] < 1 or memory.shape[2] != self.feature_dim
                or target.shape[1] != self.feature_dim):
            raise ValueError("Bridge expects target [B,D] and source memory [B,S,D]")
        target_norm = F.layer_norm(target.float(), (self.feature_dim,))
        source_norm = F.layer_norm(memory.detach().float(), (self.feature_dim,))
        query = self.query(target_norm)
        scores = (self.key(source_norm) * query.unsqueeze(1)).sum(-1)
        scores = scores / math.sqrt(self.bottleneck_dim)
        logits = torch.cat([scores, self.null_logit.expand(len(target), 1)], dim=1)
        weights = logits.softmax(dim=1)
        # Null value is exactly zero, so its mass attenuates the entire bridge.
        context = (weights[:, :-1, None] * self.value(source_norm)).sum(dim=1)
        return query, context, weights

    def routing_weights(self, target: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=target.device.type, enabled=False):
            return self._retrieve(target, memory)[2]

    def forward(self, target: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Routing and relation algebra stay FP32 even with CUDA AMP. Bias-free
        # maps ensure absent/rejected evidence cannot become a target-only MLP.
        with torch.autocast(device_type=target.device.type, enabled=False):
            query, context, _ = self._retrieve(target, memory)
            interaction = torch.cat([context, query * context], dim=-1)
            correction = self.up(F.gelu(self.relation(interaction)))
        return target + correction.to(target.dtype)

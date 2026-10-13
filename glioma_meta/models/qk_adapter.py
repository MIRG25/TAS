"""Source-trained Q/K deltas, separate from immutable MAE parameter storage."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

VERSION = 'source-qk-low-rank-v1'


class QKLowRankAdapter(nn.Module):
    """Scale-one low-rank updates; V/output projections remain unchanged."""

    def __init__(self, dimension: int, rank: int):
        super().__init__()
        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= dimension:
            raise ValueError('Q/K rank must be an integer in [1, embed_dim]')
        self.rank = rank
        # CPU construction must not shift original adapters/heads or data RNG.
        with torch.random.fork_rng(devices=[]):
            self.q_down = nn.Linear(dimension, rank, bias=False)
            self.q_up = nn.Linear(rank, dimension, bias=False)
            self.k_down = nn.Linear(dimension, rank, bias=False)
            self.k_up = nn.Linear(rank, dimension, bias=False)
            nn.init.zeros_(self.q_up.weight)
            nn.init.zeros_(self.k_up.weight)

    def forward(self, block: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        attention = block.attention
        normalized = block.norm1(tokens).transpose(0, 1)
        # Functional MHA avoids eval/no_grad fastpaths bypassing the deltas.
        # Forming effective weights preserves the same fused QKV projection.
        # Disable autocast only for these tiny weight products: FP32 parameters
        # must not acquire FP16-rounded deltas before the regular MHA autocast.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            dq = self.q_up.weight @ self.q_down.weight
            dk = self.k_up.weight @ self.k_down.weight
            weight = attention.in_proj_weight + torch.cat((dq, dk, torch.zeros_like(dq)), dim=0)
        attended, _ = F.multi_head_attention_forward(
            normalized, normalized, normalized, attention.embed_dim, attention.num_heads,
            weight, attention.in_proj_bias, attention.bias_k, attention.bias_v,
            attention.add_zero_attn, attention.dropout,
            attention.out_proj.weight, attention.out_proj.bias,
            training=attention.training, need_weights=False)
        tokens = tokens + attended.transpose(0, 1)
        return tokens + block.mlp(block.norm2(tokens))


def qk_settings(config):
    settings = config.get('model', {}).get('qk_adapter', {})
    target_trainable = config.get('adaptation', {}).get('target_qk_trainable', True)
    if not isinstance(target_trainable, bool):
        raise ValueError('adaptation.target_qk_trainable requires a boolean')
    if not isinstance(settings, dict) or not isinstance(settings.get('enabled', False), bool):
        raise ValueError('model.qk_adapter requires boolean enabled')
    if not settings.get('enabled', False):
        if not target_trainable:
            raise ValueError('Frozen target Q/K requires model.qk_adapter.enabled')
        return 0
    rank = settings.get('rank', 8)
    dim = config.get('model', {}).get('embed_dim', 384)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= dim:
        raise ValueError('Q/K rank must be an integer in [1, embed_dim]')
    source, target = config.get('source_training', {}), config.get('adaptation', {})
    if (config.get('model', {}).get('compositional_adapter', {}).get('enabled', False)
            or source.get('mae_token_retention_weight', 0)
            or source.get('patient_contrast', {}).get('enabled', False)
            or source.get('refit_on_full_source_budget', False)
            or target.get('refit_on_full_label_budget', False)
            or target.get('sam', {}).get('enabled', False)
            or target.get('safe_transfer', {}).get('enabled', False)
            or any(target.get(k, False) for k in ('source_residuals', 'target_mae_bypass',
                'target_source_bridge', 'target_covariance_readout', 'head_warmstart_epochs'))
            or target.get('target_pooling', 'mean') != 'mean'):
        raise ValueError('Q/K v1 excludes composition, residuals, retention, contrast, refit and other ablations')
    return rank


def qk_metadata(model):
    if not getattr(model, 'qk_rank', 0):
        return {}
    return {'qk_adapter': {'version': VERSION, 'rank': model.qk_rank, 'scale': 1.0,
        'depth': len(model.shared_qk), 'projections': ['q', 'k'],
        'source_parameter_count': sum(p.numel() for p in model.shared_qk.parameters())},
        **({'target_qk_trainable': dict(model._target_qk_trainable)} if model.target_qk else {})}


def validate_qk_checkpoint(saved, config):
    rank = qk_settings(config)
    recorded = saved.get('qk_adapter')
    if bool(rank) != bool(recorded):
        raise ValueError('Q/K checkpoint mismatch: train new Q/K sources; legacy sources cannot be reused')
    if rank and (recorded.get('version') != VERSION or recorded.get('rank') != rank
                 or recorded.get('scale') != 1.0 or recorded.get('projections') != ['q', 'k']
                 or recorded.get('depth') != config['model'].get('depth', 12)
                 or qk_settings(saved.get('config', {})) != rank):
        raise ValueError('Q/K checkpoint metadata/config mismatch')


def source_qk_groups(model, learning_rate):
    if not model.qk_rank:
        return []
    return [('source_qk', list(model.shared_qk.named_parameters()), learning_rate)]


def source_shared_parameters(model):
    """Parameters on which source task conflicts are coordinated (PCGrad)."""
    return [p for module in (model.shared_adapters, model.shared_qk)
            for p in module.parameters() if p.requires_grad]

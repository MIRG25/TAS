"""Target-private, zero-initialized residual readout of intermediate MRI features."""
import torch
from torch import nn
import torch.nn.functional as F


class CrossDepthResidualReadout(nn.Module):
    VERSION = 'cross_depth_residual_v1'

    def __init__(self, feature_dim, layers, rank, encoder_depth):
        super().__init__()
        self.validate(feature_dim, layers, rank, encoder_depth)
        self.layers = tuple(layers)
        self.rank = rank
        with torch.random.fork_rng(devices=[]):
            self.down = nn.Linear(feature_dim * len(layers), rank, bias=False)
            self.up = nn.Linear(rank, feature_dim, bias=False)
            nn.init.zeros_(self.up.weight)

    @staticmethod
    def validate(feature_dim, layers, rank, encoder_depth):
        if (not isinstance(layers, (list, tuple)) or not layers
                or any(isinstance(i, bool) or not isinstance(i, int)
                       or not 1 <= i < encoder_depth for i in layers)
                or list(layers) != sorted(set(layers))):
            raise ValueError('cross_depth_layers must be sorted unique intermediate 1-based layers')
        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= feature_dim:
            raise ValueError('cross_depth_rank must be an integer in [1, feature_dim]')

    @staticmethod
    def summarize(tokens):
        # Patient-local normalization, no running statistics; exclude CLS.
        # FP32 reductions also keep the small readout stable under AMP.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            values = tokens[:, 1:].float()
            return F.layer_norm(values, (values.shape[-1],)).mean(dim=1)

    def forward(self, final_features, summaries):
        if len(summaries) != len(self.layers):
            raise ValueError('Missing intermediate cross-depth features')
        residual = self.up(F.gelu(self.down(torch.cat(summaries, dim=-1))))
        return final_features + residual.to(final_features.dtype)

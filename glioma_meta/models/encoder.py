"""A compact 3D ResNet encoder suitable for four-channel brain MRI."""
from __future__ import annotations

import torch
from torch import nn


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class GatedAttentionPool3D(nn.Module):
    """Learn a patient-level descriptor without supplying the tumour mask.

    Global average pooling can dilute a comparatively small enhancing or
    infiltrative component inside a 128^3 ROI.  This lightweight gated
    attention layer learns the spatial weighting solely from MRI features, so
    the inference model remains MRI-only.
    """

    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        hidden_channels = max(int(hidden_channels), 1)
        self.tanh_projection = nn.Conv3d(channels, hidden_channels, kernel_size=1)
        self.sigmoid_projection = nn.Conv3d(channels, hidden_channels, kernel_size=1)
        self.score = nn.Conv3d(hidden_channels, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attention = torch.tanh(self.tanh_projection(features))
        attention = attention * torch.sigmoid(self.sigmoid_projection(features))
        attention = self.score(attention).flatten(2)
        attention = torch.softmax(attention, dim=-1)
        tokens = features.flatten(2)
        return torch.sum(tokens * attention, dim=-1)


class MRIEncoder3D(nn.Module):
    def __init__(
        self,
        base_channels: int = 24,
        feature_dim: int = 192,
        dropout: float = 0.2,
        pooling: str = "average",
        attention_hidden_channels: int = 64,
        feature_norm: bool = False,
    ) -> None:
        super().__init__()
        c = base_channels
        if pooling not in {"average", "gated-attention"}:
            raise ValueError("model.pooling must be average or gated-attention")
        self.pooling = pooling
        self.stem = nn.Sequential(
            nn.Conv3d(4, c, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm3d(c, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.stage1 = nn.Sequential(ResidualBlock3D(c, c), ResidualBlock3D(c, c))
        self.stage2 = nn.Sequential(ResidualBlock3D(c, c * 2, stride=2), ResidualBlock3D(c * 2, c * 2))
        self.stage3 = nn.Sequential(ResidualBlock3D(c * 2, c * 4, stride=2), ResidualBlock3D(c * 4, c * 4))
        self.stage4 = nn.Sequential(ResidualBlock3D(c * 4, c * 6, stride=2), ResidualBlock3D(c * 6, c * 6))
        self.pool = (
            nn.AdaptiveAvgPool3d(1)
            if pooling == "average"
            else GatedAttentionPool3D(c * 6, attention_hidden_channels)
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 6, feature_dim),
            nn.LayerNorm(feature_dim) if feature_norm else nn.Identity(),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.projection(self.pool(x))

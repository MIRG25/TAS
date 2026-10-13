"""Target-private, zero-output voxel residual on the frozen MAE token grid."""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


class VoxelResidualTokenizer(nn.Module):
    """Small-kernel MRI stem; not a replacement for pretrained patch weights.

    For patch size 16: four stride-two 3x3x3 convolutions, with channels
    8/16/32/32, followed by a zero-initialized 1x1x1 projection. GroupNorm
    avoids micro-batch running statistics. Output order matches Conv3d's
    D/H/W flattening in MAEEncoder3D. No CLS token or position embedding is
    created here. Smaller power-of-two patches are supported for tiny tests.
    """

    VERSION = "voxel_residual_v1"

    def __init__(self, in_channels: int, embed_dim: int, patch_size: Sequence[int]) -> None:
        super().__init__()
        patch = tuple(patch_size)
        if (len(patch) != 3 or any(isinstance(p, bool) or not isinstance(p, int) for p in patch)
                or len(set(patch)) != 1 or patch[0] < 2 or patch[0] & (patch[0] - 1)):
            raise ValueError("Voxel tokenizer requires isotropic power-of-two patch_size >= 2")
        self.patch_size = patch
        self.in_channels = in_channels
        self.stage_channels = tuple(min(8 * 2**i, 32) for i in range(int(math.log2(patch[0]))))
        # Extra module construction must not shift the head, augmentation, or
        # dropout random stream relative to the baseline. Construction is CPU.
        with torch.random.fork_rng(devices=[]):
            layers = []
            previous = in_channels
            for channels in self.stage_channels:
                layers.extend([
                    nn.Conv3d(previous, channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(4, channels),
                    nn.GELU(),
                ])
                previous = channels
            self.stem = nn.Sequential(*layers)
            self.projection = nn.Conv3d(previous, embed_dim, kernel_size=1, bias=False)
            nn.init.zeros_(self.projection.weight)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if (images.ndim != 5 or images.shape[1] != self.in_channels
                or any(n % p for n, p in zip(images.shape[2:], self.patch_size))):
            raise ValueError("Voxel tokenizer expects B/C/D/H/W images divisible by patch_size")
        return self.projection(self.stem(images)).flatten(2).transpose(1, 2)

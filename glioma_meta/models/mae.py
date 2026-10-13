"""A compact 3-D masked autoencoder for four-modal glioma MRI."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _triple(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    values = tuple(int(item) for item in value)
    if len(values) != 3:
        raise ValueError("Expected three spatial dimensions")
    return values


def _sincos_1d(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim % 2:
        raise ValueError("One-dimensional sine/cosine embedding must have even width")
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10_000**omega)
    angles = np.outer(positions.reshape(-1), omega)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


def sincos_position_embedding_3d(
    embed_dim: int, grid_shape: tuple[int, int, int], include_cls_token: bool = True
) -> torch.Tensor:
    if embed_dim % 6:
        raise ValueError("3-D sine/cosine embedding width must be divisible by 6")
    grids = np.meshgrid(
        np.arange(grid_shape[0], dtype=np.float64),
        np.arange(grid_shape[1], dtype=np.float64),
        np.arange(grid_shape[2], dtype=np.float64),
        indexing="ij",
    )
    axis_width = embed_dim // 3
    embedding = np.concatenate(
        [_sincos_1d(axis_width, axis.reshape(-1)) for axis in grids], axis=1
    ).astype(np.float32)
    if include_cls_token:
        embedding = np.concatenate([np.zeros((1, embed_dim), dtype=np.float32), embedding], axis=0)
    return torch.from_numpy(embedding).unsqueeze(0)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(round(embed_dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class MaskedAutoencoderViT3D(nn.Module):
    def __init__(
        self,
        image_size: int | Sequence[int] = 128,
        patch_size: int | Sequence[int] = 16,
        in_channels: int = 4,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        decoder_embed_dim: int = 192,
        decoder_depth: int = 4,
        decoder_num_heads: int = 6,
        mlp_ratio: float = 4.0,
        norm_pix_loss: bool = True,
        dropout: float = 0.0,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = _triple(image_size)
        self.patch_size = _triple(patch_size)
        if any(image % patch for image, patch in zip(self.image_size, self.patch_size)):
            raise ValueError("Every image dimension must be divisible by its patch size")
        self.grid_shape = tuple(
            image // patch for image, patch in zip(self.image_size, self.patch_size)
        )
        self.num_patches = math.prod(self.grid_shape)
        self.in_channels = int(in_channels)
        self.patch_volume = math.prod(self.patch_size)
        self.patch_dim = self.in_channels * self.patch_volume
        self.norm_pix_loss = bool(norm_pix_loss)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.patch_embed = nn.Conv3d(
            self.in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer(
            "position_embedding",
            sincos_position_embedding_3d(embed_dim, self.grid_shape),
            persistent=True,
        )
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.register_buffer(
            "decoder_position_embedding",
            sincos_position_embedding_3d(decoder_embed_dim, self.grid_shape),
            persistent=True,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_prediction = nn.Linear(decoder_embed_dim, self.patch_dim)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        patch_weight = self.patch_embed.weight.data
        nn.init.xavier_uniform_(patch_weight.view(patch_weight.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = images.shape
        if (depth, height, width) != self.image_size or channels != self.in_channels:
            raise ValueError(
                f"Expected [B,{self.in_channels},{self.image_size[0]},{self.image_size[1]},{self.image_size[2]}]; "
                f"got {tuple(images.shape)}"
            )
        pd, ph, pw = self.patch_size
        gd, gh, gw = self.grid_shape
        patches = images.reshape(batch, channels, gd, pd, gh, ph, gw, pw)
        patches = patches.permute(0, 2, 4, 6, 3, 5, 7, 1)
        return patches.reshape(batch, self.num_patches, self.patch_dim)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch, number, dimension = patches.shape
        if number != self.num_patches or dimension != self.patch_dim:
            raise ValueError("Patch tensor has incompatible shape")
        pd, ph, pw = self.patch_size
        gd, gh, gw = self.grid_shape
        images = patches.reshape(batch, gd, gh, gw, pd, ph, pw, self.in_channels)
        images = images.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return images.reshape(batch, self.in_channels, *self.image_size)

    def random_masking(
        self, tokens: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must lie strictly between 0 and 1")
        batch, number, dimension = tokens.shape
        keep = max(int(round(number * (1.0 - mask_ratio))), 1)
        noise = torch.rand(batch, number, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        visible = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, dimension))
        mask = torch.ones(batch, number, device=tokens.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return visible, mask, ids_restore

    def _run_blocks(self, tokens: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        for block in blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        return tokens

    def forward_encoder(
        self, images: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embedding[:, 1:, :]
        visible, mask, ids_restore = self.random_masking(tokens, mask_ratio)
        cls = self.cls_token + self.position_embedding[:, :1, :]
        cls = cls.expand(images.shape[0], -1, -1)
        latent = torch.cat([cls, visible], dim=1)
        latent = self._run_blocks(latent, self.encoder_blocks)
        return self.encoder_norm(latent), mask, ids_restore

    def forward_decoder(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder_embed(latent)
        visible = decoded[:, 1:, :]
        missing_count = ids_restore.shape[1] - visible.shape[1]
        mask_tokens = self.mask_token.expand(decoded.shape[0], missing_count, -1)
        restored = torch.cat([visible, mask_tokens], dim=1)
        restored = torch.gather(
            restored,
            1,
            ids_restore.unsqueeze(-1).expand(-1, -1, restored.shape[-1]),
        )
        decoded = torch.cat([decoded[:, :1, :], restored], dim=1)
        decoded = decoded + self.decoder_position_embedding
        decoded = self._run_blocks(decoded, self.decoder_blocks)
        decoded = self.decoder_norm(decoded)
        return self.decoder_prediction(decoded[:, 1:, :])

    def reconstruction_loss(
        self, images: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        target = self.patchify(images)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            variance = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mean) / torch.sqrt(variance + 1e-6)
        loss = (prediction - target).pow(2).mean(dim=-1)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(
        self, images: torch.Tensor, mask_ratio: float = 0.75
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(images, mask_ratio)
        prediction = self.forward_decoder(latent, ids_restore)
        loss = self.reconstruction_loss(images, prediction, mask)
        return loss, prediction, mask

    def encode(self, images: torch.Tensor, pooling: str = "mean") -> torch.Tensor:
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embedding[:, 1:, :]
        cls = (self.cls_token + self.position_embedding[:, :1, :]).expand(
            images.shape[0], -1, -1
        )
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.encoder_norm(self._run_blocks(tokens, self.encoder_blocks))
        if pooling == "mean":
            return tokens[:, 1:, :].mean(dim=1)
        if pooling == "cls":
            return tokens[:, 0, :]
        if pooling == "tokens":
            return tokens[:, 1:, :]
        raise ValueError("pooling must be mean, cls or tokens")

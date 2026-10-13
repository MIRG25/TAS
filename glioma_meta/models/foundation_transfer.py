"""MAE encoder and clinical heads for downstream glioma MRI transfer."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .mae import TransformerBlock, _triple, sincos_position_embedding_3d
from .multitask import ResidualFeatureAdapter


class MAEEncoder3D(nn.Module):
    """Encoder-only counterpart of :class:`MaskedAutoencoderViT3D`.

    Parameter names intentionally match the ``encoder_state`` saved by MAE
    pretraining, while decoder parameters are omitted from downstream runs.
    """

    def __init__(
        self,
        image_size: int | Sequence[int] = 128,
        patch_size: int | Sequence[int] = 16,
        in_channels: int = 4,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
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
        self.in_channels = int(in_channels)
        self.feature_dim = int(embed_dim)
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
        self.initialize_weights()

    def initialize_weights(self) -> None:
        weight = self.patch_embed.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _run_blocks(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.encoder_blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        return tokens

    def prepare_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Patchify an MRI batch and prepend the positional class token."""
        expected = (self.in_channels, *self.image_size)
        if images.ndim != 5 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"Expected [B,{','.join(str(value) for value in expected)}]; "
                f"got {tuple(images.shape)}"
            )
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embedding[:, 1:, :]
        cls = (self.cls_token + self.position_embedding[:, :1, :]).expand(
            images.shape[0], -1, -1
        )
        return torch.cat([cls, tokens], dim=1)

    def pool_tokens(self, tokens: torch.Tensor, pooling: str) -> torch.Tensor:
        tokens = self.encoder_norm(tokens)
        if pooling == "mean":
            return tokens[:, 1:, :].mean(dim=1)
        if pooling == "cls":
            return tokens[:, 0, :]
        if pooling == "tokens":
            return tokens[:, 1:, :]
        raise ValueError("pooling must be mean, cls or tokens")

    def forward(self, images: torch.Tensor, pooling: str = "mean") -> torch.Tensor:
        return self.pool_tokens(self._run_blocks(self.prepare_tokens(images)), pooling)


class GliomaFoundationTransferModel(nn.Module):
    def __init__(
        self,
        image_size: int | Sequence[int] = 128,
        patch_size: int | Sequence[int] = 16,
        in_channels: int = 4,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        gradient_checkpointing: bool = True,
        pooling: str = "mean",
        survival_bins: int = 12,
        tasks: Sequence[str] | None = None,
        use_feature_adapter: bool = False,
        use_deep_adapters: bool = False,
        adapter_dim: int = 96,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "cls"}:
            raise ValueError("Downstream pooling must be mean or cls")
        if use_feature_adapter and use_deep_adapters:
            raise ValueError("Feature and deep adapters are mutually exclusive")
        self.encoder = MAEEncoder3D(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.pooling = pooling
        self.feature_dim = int(embed_dim)
        self.survival_bins = int(survival_bins)
        self.adapter = (
            ResidualFeatureAdapter(self.feature_dim, int(adapter_dim))
            if use_feature_adapter
            else nn.Identity()
        )
        self.deep_adapters = nn.ModuleList(
            [
                ResidualFeatureAdapter(self.feature_dim, int(adapter_dim))
                for _ in range(len(self.encoder.encoder_blocks))
            ]
            if use_deep_adapters
            else []
        )
        output_dims = {"idh": 2, "who": 2, "survival": self.survival_bins}
        requested_tasks = list(dict.fromkeys(tasks or output_dims.keys()))
        unknown = set(requested_tasks) - set(output_dims)
        if unknown:
            raise ValueError(f"Unknown task heads requested: {sorted(unknown)}")
        self.heads = nn.ModuleDict(
            {task: nn.Linear(self.feature_dim, output_dims[task]) for task in requested_tasks}
        )
        self.encoder_frozen = False

    def set_encoder_trainable(self, trainable: bool) -> None:
        self.encoder_frozen = not trainable
        self.encoder.requires_grad_(trainable)

    def load_pretrained_encoder(self, state: Mapping[str, torch.Tensor]) -> None:
        incompatible = self.encoder.load_state_dict(dict(state), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Incompatible MAE encoder: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    def encode_backbone(self, image: torch.Tensor) -> torch.Tensor:
        if self.deep_adapters:
            # Gradients must pass through frozen transformer blocks so that
            # adapters inserted in earlier layers can learn.  The encoder
            # parameters themselves remain requires_grad=False.
            tokens = self.encoder.prepare_tokens(image)
            for block, adapter in zip(
                self.encoder.encoder_blocks, self.deep_adapters
            ):
                if (
                    self.encoder.gradient_checkpointing
                    and self.training
                    and torch.is_grad_enabled()
                    and tokens.requires_grad
                ):
                    tokens = checkpoint(block, tokens, use_reentrant=False)
                else:
                    tokens = block(tokens)
                tokens = adapter(tokens)
            return self.encoder.pool_tokens(tokens, self.pooling)
        if self.encoder_frozen:
            with torch.no_grad():
                return self.encoder(image, pooling=self.pooling)
        return self.encoder(image, pooling=self.pooling)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.encode_backbone(image))

    def forward_task(self, image: torch.Tensor, task: str) -> torch.Tensor:
        return self.heads[task](self.encode(image))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode(image)
        return {name: head(features) for name, head in self.heads.items()}


class TaskExpandableFoundationModel(nn.Module):
    """Immutable foundation encoder with isolated adapter/head modules per task.

    Adding or fitting one task never changes the encoder or any previously
    registered task module.  This makes task expansion free from catastrophic
    forgetting by construction rather than by a soft regularization penalty.
    """

    OUTPUT_DIMS = {"idh": 2, "who": 2, "survival": None}

    def __init__(
        self,
        image_size: int | Sequence[int] = 128,
        patch_size: int | Sequence[int] = 16,
        in_channels: int = 4,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        gradient_checkpointing: bool = True,
        pooling: str = "mean",
        survival_bins: int = 12,
        adapter_dim: int = 96,
        adapter_type: str = "feature",
        tasks: Sequence[str] = (),
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "cls"}:
            raise ValueError("Downstream pooling must be mean or cls")
        self.encoder = MAEEncoder3D(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.pooling = pooling
        self.feature_dim = int(embed_dim)
        self.survival_bins = int(survival_bins)
        self.adapter_dim = int(adapter_dim)
        if adapter_type not in {"feature", "deep"}:
            raise ValueError("adapter_type must be feature or deep")
        self.adapter_type = adapter_type
        self.adapters = nn.ModuleDict()
        self.deep_adapters = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.encoder.requires_grad_(False)
        for task in dict.fromkeys(tasks):
            self.add_task(task)
        self.freeze_all()

    def _output_dim(self, task: str) -> int:
        if task not in self.OUTPUT_DIMS:
            raise ValueError(f"Unknown task: {task}")
        return self.survival_bins if task == "survival" else int(self.OUTPUT_DIMS[task])

    def load_pretrained_encoder(self, state: Mapping[str, torch.Tensor]) -> None:
        incompatible = self.encoder.load_state_dict(dict(state), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Incompatible MAE encoder: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        self.encoder.requires_grad_(False)

    def add_task(self, task: str) -> None:
        self._output_dim(task)
        if task in self.heads:
            raise ValueError(f"Task already exists: {task}")
        if self.adapter_type == "feature":
            self.adapters[task] = ResidualFeatureAdapter(
                self.feature_dim, self.adapter_dim
            )
        else:
            self.deep_adapters[task] = nn.ModuleList(
                [
                    ResidualFeatureAdapter(self.feature_dim, self.adapter_dim)
                    for _ in self.encoder.encoder_blocks
                ]
            )
        self.heads[task] = nn.Linear(self.feature_dim, self._output_dim(task))

    def freeze_all(self) -> None:
        self.requires_grad_(False)

    def set_trainable_task(self, task: str) -> None:
        if task not in self.heads:
            raise ValueError(f"Task has not been added: {task}")
        self.freeze_all()
        if self.adapter_type == "feature":
            self.adapters[task].requires_grad_(True)
        else:
            self.deep_adapters[task].requires_grad_(True)
        self.heads[task].requires_grad_(True)

    def _deep_task_features(self, image: torch.Tensor, task: str) -> torch.Tensor:
        tokens = self.encoder.prepare_tokens(image)
        for block, adapter in zip(
            self.encoder.encoder_blocks, self.deep_adapters[task]
        ):
            if (
                self.encoder.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
                and tokens.requires_grad
            ):
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            tokens = adapter(tokens)
        return self.encoder.pool_tokens(tokens, self.pooling)

    def forward_task(self, image: torch.Tensor, task: str) -> torch.Tensor:
        if task not in self.heads:
            raise ValueError(f"Task has not been added: {task}")
        if self.adapter_type == "deep":
            return self.heads[task](self._deep_task_features(image, task))
        with torch.no_grad():
            features = self.encoder(image, pooling=self.pooling)
        return self.heads[task](self.adapters[task](features))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.adapter_type == "deep":
            return {
                task: self.heads[task](self._deep_task_features(image, task))
                for task in self.heads
            }
        with torch.no_grad():
            features = self.encoder(image, pooling=self.pooling)
        return {
            task: self.heads[task](self.adapters[task](features))
            for task in self.heads
        }

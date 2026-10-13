"""Source-guided, expandable clinical-task model on a frozen 3-D MAE."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .foundation_transfer import MAEEncoder3D
from .multitask import ResidualFeatureAdapter
from .spatial_adapter import SpatialResidualFeatureAdapter
from .source_knowledge_bridge import SourceKnowledgeBridge
from .voxel_tokenizer import VoxelResidualTokenizer
from .covariance_readout import CovarianceResidualReadout
from .cross_depth_readout import CrossDepthResidualReadout
from .qk_adapter import QKLowRankAdapter


class SpatialAttentionPool(nn.Module):
    """Content-weighted spatial mean, initialized to the original uniform mean.

    One zero-initialized linear score per token; no bias (a common score offset
    cancels in softmax), no extra RNG draws and only feature_dim parameters.
    Input tokens are already normalized by the immutable encoder's LayerNorm.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, feature_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Keep scoring/softmax/reduction in FP32 under mixed precision. Cast
        # only the pooled feature back, without detaching the scoring gradient.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            values = tokens.float()
            scores = F.linear(values, self.weight.float()).squeeze(-1)
            weights = scores.softmax(dim=1)
            pooled = (values * weights.unsqueeze(-1)).sum(dim=1)
        return pooled.to(dtype=tokens.dtype)


class MAEFeatureBypass(nn.Module):
    """Extra columns of the target head, zero initialized without RNG draws."""

    def __init__(self, feature_dim: int, output_dim: int) -> None:
        super().__init__()
        # No second bias: the existing target head already owns the intercept.
        self.weight = nn.Parameter(torch.zeros(output_dim, feature_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.weight)


class SourceGuidedFoundationModel(nn.Module):
    """Separate transferable consensus features from source-specific features.

    The MAE encoder is immutable. Source tasks jointly update a deep shared
    adapter stack while retaining private feature adapters and heads. A newly
    added target can either copy the trained shared adapters (direct initialized
    adaptation) or learn gates over frozen residuals (legacy comparison).
    Existing tasks retain their original frozen parameters in both modes.
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
        private_adapter_dim: int | None = None,
        source_tasks: Sequence[str] = (),
        qk_rank: int = 0,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be mean or cls")
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
        self.private_adapter_dim = int(private_adapter_dim or adapter_dim)
        self.source_tasks = list(dict.fromkeys(source_tasks))
        unknown = set(self.source_tasks) - set(self.OUTPUT_DIMS)
        if unknown:
            raise ValueError(f"Unknown source tasks: {sorted(unknown)}")

        self.shared_adapters = nn.ModuleList(
            [
                ResidualFeatureAdapter(self.feature_dim, self.adapter_dim)
                for _ in self.encoder.encoder_blocks
            ]
        )
        self.private_adapters = nn.ModuleDict(
            {
                task: ResidualFeatureAdapter(
                    self.feature_dim, self.private_adapter_dim
                )
                for task in self.source_tasks
            }
        )
        self.target_adapters = nn.ModuleDict()
        # Empty for legacy mean/CLS checkpoints, so their state keys remain
        # unchanged. Pooling belongs to the target, never to a source branch.
        self.target_poolers = nn.ModuleDict()
        self._target_pooling: dict[str, str] = {}
        # An empty dictionary adds no keys to legacy checkpoints. Reuse the
        # immutable encoder, not a second parameter copy, for the raw MAE view.
        self.target_mae_heads = nn.ModuleDict()
        # Empty by default: existing source/target checkpoint keys are unchanged.
        self.target_source_bridges = nn.ModuleDict()
        self.target_voxel_tokenizers = nn.ModuleDict()
        self.target_covariance_readouts = nn.ModuleDict()
        self.target_cross_depth_readouts = nn.ModuleDict()
        # Each newly added endpoint may own a small copy of the final encoder
        # blocks.  The copies are initialized from the immutable MAE encoder,
        # then trained only for that endpoint.  Existing source tasks continue
        # to use the original frozen encoder, so expansion cannot change their
        # parameters or predictions.
        self.target_private_blocks = nn.ModuleDict()
        self.target_gate_logits = nn.ParameterDict()
        self.target_expert_gate_logits = nn.ParameterDict()
        # These are deliberately runtime controls rather than trainable
        # parameters.  Fine-grained signed gates are learned on the target-fit
        # split, while a global source scale is calibrated only on the inner
        # validation split.  A scale of zero is an exact null-source expert.
        self.target_source_scales = nn.ParameterDict()
        self._target_source_dropout: dict[str, float] = {}
        self._target_source_residuals: dict[str, bool] = {}
        self._target_initializations: dict[str, str] = {}
        self.heads = nn.ModuleDict(
            {
                task: nn.Linear(self.feature_dim, self._output_dim(task))
                for task in self.source_tasks
            }
        )
        self.encoder.requires_grad_(False)
        if isinstance(qk_rank, bool) or not isinstance(qk_rank, int) or not 0 <= qk_rank <= self.feature_dim:
            raise ValueError('qk_rank must be an integer in [0, feature_dim]')
        self.qk_rank = qk_rank
        self.shared_qk = nn.ModuleList([
            QKLowRankAdapter(self.feature_dim, qk_rank)
            for _ in self.encoder.encoder_blocks] if qk_rank else [])
        self.target_qk = nn.ModuleDict()
        self._target_qk_trainable: dict[str, bool] = {}

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

    def add_target_task(
        self,
        task: str,
        initial_source_gate: float = 0.0,
        source_path_dropout: float = 0.0,
        private_block_depth: int = 0,
        initialization: str = "fresh",
        source_residuals: bool = True,
        target_pooling: str | None = None,
        target_mae_bypass: bool = False,
        target_spatial_adapter: bool = False,
        target_source_bridge: bool = False,
        source_bridge_dim: int = 64,
        target_voxel_tokenizer: bool = False,
        target_covariance_readout: bool = False,
        covariance_rank: int = 32,
        target_qk_trainable: bool = True,
        target_cross_depth_readout: bool = False,
        cross_depth_layers: Sequence[int] = (4, 8, 10),
        cross_depth_rank: int = 64,
    ) -> None:
        self._output_dim(task)
        if not isinstance(target_qk_trainable, bool):
            raise ValueError('target_qk_trainable must be a boolean')
        if not target_qk_trainable and (not self.qk_rank or initialization != 'source-shared'):
            raise ValueError('Frozen target Q/K requires source-shared Q/K initialization')
        if self.qk_rank and (source_residuals or target_mae_bypass or target_source_bridge
                             or target_covariance_readout or target_pooling not in (None, 'mean')):
            raise ValueError('Q/K target requires the direct mean-pooling path, without other ablations')
        if task in self.heads:
            raise ValueError(f"Task already exists: {task}")
        target_pooling = self.pooling if target_pooling is None else target_pooling
        if not isinstance(target_cross_depth_readout, bool):
            raise ValueError('target_cross_depth_readout must be a boolean')
        if target_cross_depth_readout:
            if (target_pooling != 'mean' or source_residuals or target_mae_bypass
                    or target_source_bridge or target_covariance_readout or self.qk_rank):
                raise ValueError('Cross-depth readout requires direct mean pooling without other readout ablations')
            CrossDepthResidualReadout.validate(self.feature_dim, cross_depth_layers,
                                               cross_depth_rank, len(self.encoder.encoder_blocks))
        if target_pooling not in {"mean", "cls", "attention"}:
            raise ValueError("target_pooling must be mean, cls or attention")
        if not isinstance(target_mae_bypass, bool):
            raise ValueError("target_mae_bypass must be a boolean")
        if not isinstance(target_spatial_adapter, bool):
            raise ValueError("target_spatial_adapter must be a boolean")
        if not isinstance(target_source_bridge, bool):
            raise ValueError("target_source_bridge must be a boolean")
        if not isinstance(target_voxel_tokenizer, bool):
            raise ValueError("target_voxel_tokenizer must be a boolean")
        if not isinstance(target_covariance_readout, bool):
            raise ValueError("target_covariance_readout must be a boolean")
        if target_covariance_readout:
            if target_pooling != "mean" or source_residuals or target_mae_bypass or target_source_bridge:
                raise ValueError("Covariance readout requires mean pooling, no legacy residuals/bypass/bridge")
            if (isinstance(covariance_rank, bool) or not isinstance(covariance_rank, int)
                    or not 1 <= covariance_rank <= self.feature_dim):
                raise ValueError("covariance_rank must be an integer in [1, feature_dim]")
        if target_voxel_tokenizer and source_residuals:
            raise ValueError("Voxel tokenizer requires source_residuals=False; source input must stay unchanged")
        if target_source_bridge:
            if initialization != "source-shared" or source_residuals or not self.source_tasks:
                raise ValueError("Source bridge requires source-shared initialization, frozen sources, and no legacy residuals")
            if (isinstance(source_bridge_dim, bool) or not isinstance(source_bridge_dim, int)
                    or source_bridge_dim < 1):
                raise ValueError("source_bridge_dim must be a positive integer")
        if initialization not in {"fresh", "source-shared"}:
            raise ValueError("Target initialization must be fresh or source-shared")
        if initialization == "source-shared" and source_residuals:
            raise ValueError("Copied shared adapters must not also inject source residuals")
        if not source_residuals and (initial_source_gate != 0 or source_path_dropout != 0):
            raise ValueError("Disabled source residuals require zero gates and zero source dropout")
        if not -1.0 < initial_source_gate < 1.0:
            raise ValueError("initial_source_gate must lie strictly between -1 and 1")
        if not 0.0 <= source_path_dropout <= 1.0:
            raise ValueError("source_path_dropout must lie in [0, 1]")
        private_block_depth = int(private_block_depth)
        encoder_depth = len(self.encoder.encoder_blocks)
        if not 0 <= private_block_depth <= encoder_depth:
            raise ValueError(
                "private_block_depth must lie between zero and the encoder depth"
            )
        self.target_adapters[task] = nn.ModuleList(
            [
                ResidualFeatureAdapter(self.feature_dim, self.adapter_dim)
                for _ in self.encoder.encoder_blocks
            ]
        )
        # Construct the same modules in both arms before copying: this preserves
        # the random stream and gives the matched MAE control the identical head.
        # load_state_dict copies values, never ties the original source storage.
        if initialization == "source-shared":
            self.target_adapters[task].load_state_dict(self.shared_adapters.state_dict())
        if self.qk_rank:
            self._target_qk_trainable[task] = target_qk_trainable
            if initialization == 'source-shared':
                self.target_qk[task] = copy.deepcopy(self.shared_qk)
            else:
                self.target_qk[task] = nn.ModuleList([
                    QKLowRankAdapter(self.feature_dim, self.qk_rank)
                    for _ in self.encoder.encoder_blocks])
        if target_spatial_adapter:
            # Strictly copy the old state first; the zero extension does not
            # alter those names, values, or the following head's RNG stream.
            self.target_adapters[task] = nn.ModuleList([
                SpatialResidualFeatureAdapter(adapter, self.encoder.grid_shape)
                for adapter in self.target_adapters[task]
            ])
        self._target_initializations[task] = initialization
        self._target_source_residuals[task] = bool(source_residuals)
        self.target_private_blocks[task] = nn.ModuleList(
            [
                copy.deepcopy(block)
                for block in self.encoder.encoder_blocks[
                    encoder_depth - private_block_depth :
                ]
            ]
            if private_block_depth
            else []
        )
        # In the legacy fresh/gated path, tanh starts at zero while retaining a
        # non-zero derivative.  Consequently, adding a target task initially
        # reproduces a target-only deep adapter and source information is used
        # only when target labels provide evidence for it.  Signed gates also
        # let the target subtract a consistently biased source residual.
        initial_logit = math.atanh(initial_source_gate)
        self.target_gate_logits[task] = nn.Parameter(
            torch.full((len(self.encoder.encoder_blocks),), initial_logit)
        )
        self.target_expert_gate_logits[task] = nn.Parameter(
            torch.full((len(self.source_tasks),), initial_logit)
        )
        self.target_source_scales[task] = nn.Parameter(
            torch.tensor(1.0 if source_residuals else 0.0), requires_grad=False
        )
        self._target_source_dropout[task] = float(source_path_dropout)
        self._target_pooling[task] = target_pooling
        if target_pooling == "attention":
            self.target_poolers[task] = SpatialAttentionPool(self.feature_dim)
        self.heads[task] = nn.Linear(self.feature_dim, self._output_dim(task))
        if target_mae_bypass:
            self.target_mae_heads[task] = MAEFeatureBypass(self.feature_dim, self._output_dim(task))
        if target_source_bridge:
            # CPU construction, like the existing target head. Preserve the
            # baseline initialization and subsequent training random stream.
            with torch.random.fork_rng(devices=[]):
                self.target_source_bridges[task] = SourceKnowledgeBridge(
                    self.feature_dim, source_bridge_dim)
        if target_voxel_tokenizer:
            self.target_voxel_tokenizers[task] = VoxelResidualTokenizer(
                self.encoder.in_channels, self.feature_dim, self.encoder.patch_size)
        if target_covariance_readout:
            self.target_covariance_readouts[task] = CovarianceResidualReadout(
                self.feature_dim, covariance_rank)
        if target_cross_depth_readout:
            self.target_cross_depth_readouts[task] = CrossDepthResidualReadout(
                self.feature_dim, cross_depth_layers, cross_depth_rank, encoder_depth)

    def set_target_source_scale(self, task: str, scale: float) -> None:
        """Set validation-calibrated source strength for a target task.

        The scale is constrained to [0, 1].  It multiplies both the shared
        per-layer residual gates and the source-private expert gates.  Thus
        scale=0 disables explicit residual injection without modifying source
        parameters. It does NOT erase source knowledge already copied into a
        source-initialized target adapter.
        """
        if task not in self.target_adapters:
            raise ValueError(f"Target task has not been added: {task}")
        if not 0.0 <= float(scale) <= 1.0:
            raise ValueError("target source scale must lie in [0, 1]")
        if not self._target_source_residuals[task] and float(scale) != 0.0:
            raise ValueError("Source residual injection is disabled for this target")
        with torch.no_grad():
            self.target_source_scales[task].fill_(float(scale))

    def target_source_scale(self, task: str) -> float:
        if task not in self.target_adapters:
            raise ValueError(f"Target task has not been added: {task}")
        return float(self.target_source_scales[task].detach().cpu())

    def freeze_all(self) -> None:
        self.requires_grad_(False)

    def set_source_trainable(self) -> None:
        self.freeze_all()
        self.shared_adapters.requires_grad_(True)
        self.shared_qk.requires_grad_(True)
        for task in self.source_tasks:
            self.private_adapters[task].requires_grad_(True)
            self.heads[task].requires_grad_(True)

    def set_target_trainable(self, task: str) -> None:
        if task not in self.target_adapters:
            raise ValueError(f"Target task has not been added: {task}")
        self.freeze_all()
        self.target_adapters[task].requires_grad_(True)
        if task in self.target_qk:
            self.target_qk[task].requires_grad_(self._target_qk_trainable[task])
        self.target_private_blocks[task].requires_grad_(True)
        if task in self.target_poolers:
            self.target_poolers[task].requires_grad_(True)
        if task in self.target_mae_heads:
            self.target_mae_heads[task].requires_grad_(True)
        if task in self.target_source_bridges:
            self.target_source_bridges[task].requires_grad_(True)
        if task in self.target_voxel_tokenizers:
            self.target_voxel_tokenizers[task].requires_grad_(True)
        if task in self.target_covariance_readouts:
            self.target_covariance_readouts[task].requires_grad_(True)
        if task in self.target_cross_depth_readouts:
            self.target_cross_depth_readouts[task].requires_grad_(True)
        self.target_gate_logits[task].requires_grad_(self._target_source_residuals[task])
        self.target_expert_gate_logits[task].requires_grad_(self._target_source_residuals[task])
        self.heads[task].requires_grad_(True)

    def _run_block(self, block: nn.Module, tokens: torch.Tensor,
                   qk: QKLowRankAdapter | None = None) -> torch.Tensor:
        forward = block if qk is None else lambda value: qk(block, value)
        if (
            self.encoder.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and tokens.requires_grad
        ):
            return checkpoint(forward, tokens, use_reentrant=False)
        return forward(tokens)

    def _encode_source_shared_tokens(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.prepare_tokens(image)
        for index, (block, shared_adapter) in enumerate(zip(
            self.encoder.encoder_blocks, self.shared_adapters
        )):
            qk = self.shared_qk[index] if self.qk_rank else None
            tokens = shared_adapter(self._run_block(block, tokens, qk))
        return tokens

    def encode_source_shared(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder.pool_tokens(self._encode_source_shared_tokens(image), self.pooling)

    def source_components(
        self, image: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        shared = self.encode_source_shared(image)
        return self._source_readouts(shared)

    def _source_readouts(
        self, shared: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        private_deltas = {
            task: self.private_adapters[task](shared) - shared
            for task in self.source_tasks
        }
        outputs = {
            task: self.heads[task](shared + private_deltas[task])
            for task in self.source_tasks
        }
        return outputs, shared, private_deltas

    def source_components_with_retention(
        self, image: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Source supervision plus a label-free, same-crop spatial MAE anchor.

        One student forward supplies both the task graph and the token loss.
        Corresponding final normalized spatial tokens (never CLS or a pooled
        vector) are compared before source-private adapters. The reference is
        the immutable MAE without any adapter, evaluated without dropout or
        gradients. No target label, batch statistic, or extra model is needed.
        """
        tokens = self._encode_source_shared_tokens(image)
        outputs, shared, private = self._source_readouts(
            self.encoder.pool_tokens(tokens, self.pooling)
        )
        student = self.encoder.pool_tokens(tokens, "tokens")
        teacher = self._mae_reference(image, pooling="tokens")
        with torch.autocast(device_type=image.device.type, enabled=False):
            cosine = F.cosine_similarity(student.float(), teacher.detach().float(), dim=-1, eps=1e-6)
            # Round-off can produce cos > 1 at the identity initialization.
            retention = (1.0 - cosine).clamp_min(0.0).mean()
        return outputs, shared, private, retention

    def shared_private_orthogonality(
        self, shared: torch.Tensor, private_deltas: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if not private_deltas:
            return shared.new_zeros((), dtype=torch.float32)
        shared_float = shared.float()
        penalties = [
            F.cosine_similarity(shared_float, delta.float(), dim=-1, eps=1e-6)
            .pow(2)
            .mean()
            for delta in private_deltas.values()
        ]
        return torch.stack(penalties).mean()

    def target_pooling(self, task: str) -> str:
        if task not in self._target_pooling:
            raise ValueError(f"Target task has not been added: {task}")
        return self._target_pooling[task]

    def target_mae_bypass(self, task: str) -> bool:
        if task not in self.target_adapters:
            raise ValueError(f"Target task has not been added: {task}")
        return task in self.target_mae_heads

    def target_spatial_adapter(self, task: str) -> bool:
        if task not in self.target_adapters:
            raise ValueError(f"Target task has not been added: {task}")
        return all(isinstance(adapter, SpatialResidualFeatureAdapter)
                   for adapter in self.target_adapters[task])

    def target_voxel_tokenizer(self, task: str) -> bool:
        if task not in self.target_adapters:
            raise ValueError(f"Target task has not been added: {task}")
        return task in self.target_voxel_tokenizers

    def _mae_reference_features(self, image: torch.Tensor) -> torch.Tensor:
        """Same-crop raw spatial MAE mean for the optional target bypass."""
        return self._mae_reference(image, pooling="mean")

    def _source_bridge_memory(self, image: torch.Tensor) -> torch.Tensor:
        """Read trained source features on their own deterministic trajectory.

        Never apply frozen private experts to shifted target features. Restore
        every source submodule's mode before target checkpointed backward.
        """
        modules = [self.encoder, self.shared_adapters, self.private_adapters]
        modes = [(sub, sub.training) for module in modules for sub in module.modules()]
        try:
            for module in modules:
                module.eval()
            with torch.no_grad():
                shared = self.encode_source_shared(image)
                private = [self.private_adapters[t](shared) - shared for t in self.source_tasks]
                return torch.stack([shared, *private], dim=1).detach()
        finally:
            for module, training in modes:
                module.training = training

    def target_source_bridge(self, task: str) -> bool:
        if task not in self.target_adapters:
            raise ValueError(f"Target task has not been added: {task}")
        return task in self.target_source_bridges

    def _mae_reference(self, image: torch.Tensor, pooling: str) -> torch.Tensor:
        """Raw MAE reference, with no adapter, dropout or gradient.

        Restore each submodule's mode to leave the existing target path (and
        its checkpointed backward) untouched, including mixed train/eval modes.
        """
        modes = [(module, module.training) for module in self.encoder.modules()]
        try:
            self.encoder.eval()
            with torch.no_grad():
                return self.encoder(image, pooling=pooling)
        finally:
            for module, training in modes:
                module.training = training

    def _pool_target_tokens(self, tokens: torch.Tensor, task: str) -> torch.Tensor:
        pooling = self.target_pooling(task)
        if task in self.target_covariance_readouts:
            return self.target_covariance_readouts[task](self.encoder.pool_tokens(tokens, "tokens"))
        if pooling == "attention":
            # Identical normalization and CLS exclusion to the original mean.
            return self.target_poolers[task](self.encoder.pool_tokens(tokens, "tokens"))
        return self.encoder.pool_tokens(tokens, pooling)

    def encode_target(self, image: torch.Tensor, task: str) -> torch.Tensor:
        if task not in self.target_adapters:
            raise ValueError(f"Target task has not been added: {task}")
        if not self._target_source_residuals[task]:
            # Direct initialized adaptation: no source experts, no gates and no
            # extra random draws. Same graph/capacity as the fresh-MAE control.
            tokens = self.encoder.prepare_tokens(image)
            if task in self.target_voxel_tokenizers:
                delta = self.target_voxel_tokenizers[task](image).to(dtype=tokens.dtype)
                if delta.shape != tokens[:, 1:].shape:
                    raise RuntimeError("Voxel residual and MAE spatial token grids do not match")
                # The original CLS and position embedding are retained once.
                # Frozen blocks still propagate gradients to this input stem.
                tokens = torch.cat([tokens[:, :1], tokens[:, 1:] + delta], dim=1)
            private_blocks = self.target_private_blocks[task]
            private_start = len(self.encoder.encoder_blocks) - len(private_blocks)
            depth_readout = self.target_cross_depth_readouts[task] if task in self.target_cross_depth_readouts else None
            depth_summaries = []
            for index, (block, adapter) in enumerate(zip(
                self.encoder.encoder_blocks, self.target_adapters[task]
            )):
                target_block = private_blocks[index - private_start] if index >= private_start else block
                qk = self.target_qk[task][index] if self.qk_rank else None
                tokens = adapter(self._run_block(target_block, tokens, qk))
                if depth_readout is not None and index + 1 in depth_readout.layers:
                    depth_summaries.append(depth_readout.summarize(tokens))
            features = self._pool_target_tokens(tokens, task)
            if depth_readout is not None:
                features = depth_readout(features, depth_summaries)
            if task in self.target_source_bridges:
                features = self.target_source_bridges[task](features, self._source_bridge_memory(image))
            return features
        source_strength = self.target_source_scales[task].to(
            device=image.device, dtype=image.dtype
        )
        dropout_probability = self._target_source_dropout[task]
        if self.training and dropout_probability > 0.0:
            # No inverted-dropout rescaling is used: the purpose is to train an
            # actually usable null-source path, not merely preserve an
            # expectation.  The same draw controls every source contribution
            # within this forward pass.
            keep_source = (
                torch.rand((), device=image.device) >= dropout_probability
            ).to(dtype=image.dtype)
            source_strength = source_strength * keep_source
        gates = torch.tanh(self.target_gate_logits[task]) * source_strength
        target_tokens = self.encoder.prepare_tokens(image)
        private_blocks = self.target_private_blocks[task]
        private_start = len(self.encoder.encoder_blocks) - len(private_blocks)
        # Keep a separate frozen source trajectory.  A shared adapter was
        # trained on the output of the preceding shared adapter; applying it
        # directly to target-adapted tokens is an out-of-distribution
        # composition and was observed to cause negative transfer.  The
        # independent trajectory preserves the representation on which each
        # frozen source adapter was trained.
        source_tokens = target_tokens.detach()
        for index, (block, shared_adapter, target_adapter) in enumerate(
            zip(
                self.encoder.encoder_blocks,
                self.shared_adapters,
                self.target_adapters[task],
            )
        ):
            with torch.no_grad():
                source_base = self._run_block(block, source_tokens)
                source_next = shared_adapter(source_base)
                source_delta = source_next - source_base
            target_block = (
                private_blocks[index - private_start]
                if index >= private_start
                else block
            )
            target_base = self._run_block(target_block, target_tokens)
            gated = target_base + gates[index].to(
                dtype=target_base.dtype
            ) * source_delta.to(dtype=target_base.dtype)
            target_tokens = target_adapter(gated)
            source_tokens = source_next
        features = self._pool_target_tokens(target_tokens, task)
        source_features = self.encoder.pool_tokens(source_tokens, self.pooling)
        expert_gates = (
            torch.tanh(self.target_expert_gate_logits[task]) * source_strength
        )
        for index, source_task in enumerate(self.source_tasks):
            with torch.no_grad():
                private_delta = (
                    self.private_adapters[source_task](source_features)
                    - source_features
                )
            features = features + expert_gates[index].to(
                dtype=features.dtype
            ) * private_delta.to(dtype=features.dtype)
        return features

    def target_private_block_depth(self, task: str) -> int:
        if task not in self.target_private_blocks:
            raise ValueError(f"Target task has not been added: {task}")
        return len(self.target_private_blocks[task])

    def source_gate_values(self, task: str) -> torch.Tensor:
        if task not in self.target_gate_logits:
            raise ValueError(f"Target task has not been added: {task}")
        return torch.tanh(self.target_gate_logits[task])

    def source_expert_gate_values(self, task: str) -> torch.Tensor:
        if task not in self.target_expert_gate_logits:
            raise ValueError(f"Target task has not been added: {task}")
        return torch.tanh(self.target_expert_gate_logits[task])

    def effective_source_gate_values(self, task: str) -> torch.Tensor:
        return self.source_gate_values(task) * self.target_source_scale(task)

    def effective_source_expert_gate_values(self, task: str) -> torch.Tensor:
        return self.source_expert_gate_values(task) * self.target_source_scale(task)

    def forward_task(self, image: torch.Tensor, task: str) -> torch.Tensor:
        if task in self.source_tasks:
            shared = self.encode_source_shared(image)
            return self.heads[task](self.private_adapters[task](shared))
        if task in self.target_adapters:
            logits = self.heads[task](self.encode_target(image, task))
            if task in self.target_mae_heads:
                # Equivalent to a linear head on [adapted_features, raw_MAE],
                # retaining the original head and RNG stream exactly at init.
                logits = logits + self.target_mae_heads[task](self._mae_reference_features(image))
            return logits
        raise ValueError(f"Task has not been registered: {task}")

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs, _, _ = self.source_components(image)
        outputs.update(
            {
                task: self.forward_task(image, task)
                for task in self.target_adapters
            }
        )
        return outputs

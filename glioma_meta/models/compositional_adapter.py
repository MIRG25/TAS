"""Task-conditioned composition of a source-trained, reusable adapter basis.

This opt-in model is independent of the legacy residual gates. Target routing
uses MRI features and learned parameters only, never source label values.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from .multitask import ResidualFeatureAdapter
from .source_guided import SourceGuidedFoundationModel

VERSION = "compositional-source-v1"


class ComponentBank(ResidualFeatureAdapter):
    """Packed rank-r components; total bottleneck width matches the baseline."""

    def __init__(self, feature_dim, bottleneck_dim, components):
        super().__init__(feature_dim, bottleneck_dim)
        self.components = components

    def forward(self, tokens, logits, spatial_weight=None, grid_shape=None):
        hidden = self.down(self.norm(tokens))
        if spatial_weight is not None:
            if hidden.ndim != 3 or hidden.shape[1] != 1 + math.prod(grid_shape):
                raise ValueError("Compositional spatial adapter token/grid mismatch")
            spatial = hidden[:, 1:].transpose(1, 2).reshape(
                hidden.shape[0], hidden.shape[-1], *grid_shape)
            mixed = spatial + F.conv3d(spatial, spatial_weight, padding=1,
                                      groups=spatial.shape[1])
            hidden = torch.cat([hidden[:, :1], mixed.flatten(2).transpose(1, 2)], dim=1)
        hidden = F.gelu(hidden)
        # Uniform composition exactly recovers an ordinary width-D adapter.
        weights = (self.components * logits.float().softmax(-1)).to(hidden.dtype)
        hidden = (hidden.reshape(*hidden.shape[:-1], self.components, -1)
                  * weights.unsqueeze(-1)).flatten(-2)
        return tokens + self.up(hidden)


class TargetComposition(nn.Module):
    def __init__(self, bank, probabilities, grid_shape, spatial):
        super().__init__()
        # Separate storage: source paths are never aliased to target modules.
        self.bank = copy.deepcopy(bank).requires_grad_(False)
        self.logits = nn.Parameter(probabilities.detach().clamp_min(1e-12).log().clone())
        self.grid_shape = tuple(grid_shape)
        self.spatial_weight = (nn.Parameter(torch.zeros(bank.down.out_features, 1, 3, 3, 3))
                               if spatial else None)

    def forward(self, tokens):
        return self.bank(tokens, self.logits, self.spatial_weight, self.grid_shape)


class CompositionalSourceModel(SourceGuidedFoundationModel):
    def __init__(self, *args, components=4, composition_trainable_bank=False, **kwargs):
        if not isinstance(composition_trainable_bank, bool):
            raise ValueError('composition_trainable_bank requires a boolean')
        width = kwargs.get("adapter_dim", 96)
        if (isinstance(components, bool) or not isinstance(components, int)
                or components < 2 or width % components):
            raise ValueError("adapter_dim must be divisible by components (integer >= 2)")
        super().__init__(*args, **kwargs)
        if not self.source_tasks:
            raise ValueError("Compositional learning requires source tasks")
        self.components = components
        self.composition_trainable_bank = composition_trainable_bank
        self.shared_adapters = nn.ModuleList([
            ComponentBank(self.feature_dim, self.adapter_dim, components)
            for _ in self.encoder.encoder_blocks])
        self.source_routes = nn.ParameterDict()
        for i, task in enumerate(self.source_tasks):
            logits = torch.zeros(len(self.shared_adapters), components)
            # Small deterministic task-index offset; not a label-derived prior.
            logits[:, i % components] = .01
            self.source_routes[task] = nn.Parameter(logits)

    def set_source_trainable(self):
        super().set_source_trainable()
        self.source_routes.requires_grad_(True)

    def _source_tokens(self, image, logits):
        tokens = self.encoder.prepare_tokens(image)
        for i, (block, bank) in enumerate(zip(self.encoder.encoder_blocks, self.shared_adapters)):
            tokens = bank(self._run_block(block, tokens), logits[i])
        return tokens

    def mean_source_probabilities(self):
        return torch.stack([self.source_routes[t].softmax(-1) for t in self.source_tasks]).mean(0)

    def _encode_source_shared_tokens(self, image):
        return self._source_tokens(image, self.mean_source_probabilities().clamp_min(1e-12).log())

    def source_components(self, image):
        shared, private, outputs = {}, {}, {}
        for task in self.source_tasks:
            shared[task] = self.encoder.pool_tokens(
                self._source_tokens(image, self.source_routes[task]), self.pooling)
            private[task] = self.private_adapters[task](shared[task]) - shared[task]
            outputs[task] = self.heads[task](shared[task] + private[task])
        return outputs, shared, private

    def shared_private_orthogonality(self, shared, private_deltas):
        return torch.stack([F.cosine_similarity(shared[t].float(), delta.float(),
            dim=-1, eps=1e-6).square().mean() for t, delta in private_deltas.items()]).mean()

    def forward_task(self, image, task):
        if task in self.source_tasks:
            features = self.encoder.pool_tokens(
                self._source_tokens(image, self.source_routes[task]), self.pooling)
            return self.heads[task](self.private_adapters[task](features))
        return super().forward_task(image, task)

    def add_target_task(self, task, **kwargs):
        if kwargs.get("initialization") != "source-shared" or kwargs.get("source_residuals", True):
            raise ValueError("Composition requires source-shared initialization and no legacy residuals")
        for flag in ("target_source_bridge", "target_mae_bypass", "target_covariance_readout", "target_cross_depth_readout"):
            if kwargs.get(flag, False):
                raise ValueError(f"Composition v1 excludes {flag}")
        if kwargs.get("target_pooling", self.pooling) not in (None, "mean"):
            raise ValueError("Composition v1 requires mean pooling")
        spatial = kwargs.get("target_spatial_adapter", False)
        base_kwargs = dict(kwargs, initialization="fresh", target_spatial_adapter=False)
        super().add_target_task(task, **base_kwargs)
        probabilities = self.mean_source_probabilities().detach()
        self.target_adapters[task] = nn.ModuleList([
            TargetComposition(bank, probabilities[i], self.encoder.grid_shape, spatial)
            for i, bank in enumerate(self.shared_adapters)])
        for adapter in self.target_adapters[task]:
            adapter.bank.requires_grad_(self.composition_trainable_bank)
        self._target_initializations[task] = "source-shared"

    def set_target_trainable(self, task):
        super().set_target_trainable(task)
        for adapter in self.target_adapters[task]:
            adapter.bank.requires_grad_(self.composition_trainable_bank)

    def target_spatial_adapter(self, task):
        return all(a.spatial_weight is not None for a in self.target_adapters[task])


def composition_settings(config):
    settings = config.get("model", {}).get("compositional_adapter", {})
    if not isinstance(settings, dict) or not isinstance(settings.get("enabled", False), bool):
        raise ValueError("model.compositional_adapter requires a boolean enabled")
    return settings


def model_class_and_kwargs(config):
    from .qk_adapter import qk_settings
    qk_rank = qk_settings(config)
    settings = composition_settings(config)
    trainable_bank = config.get('adaptation', {}).get('composition_trainable_bank', False)
    if not isinstance(trainable_bank, bool):
        raise ValueError('adaptation.composition_trainable_bank requires a boolean')
    if not settings.get("enabled", False):
        if trainable_bank:
            raise ValueError('composition_trainable_bank requires compositional_adapter.enabled')
        return SourceGuidedFoundationModel, ({'qk_rank': qk_rank} if qk_rank else {})
    source = config.get("source_training", {})
    target = config.get("adaptation", {})
    if (source.get("mae_token_retention_weight", 0) or
            source.get("patient_contrast", {}).get("enabled", False) or
            source.get("refit_on_full_source_budget", False) or
            target.get("refit_on_full_label_budget", False)):
        raise ValueError("Composition v1 excludes retention, patient contrast and refit")
    return CompositionalSourceModel, {"components": settings.get("components", 4),
                                      "composition_trainable_bank": trainable_bank}


def composition_metadata(model):
    if not isinstance(model, CompositionalSourceModel):
        from .qk_adapter import qk_metadata
        return qk_metadata(model)
    return {"composition": {
        "version": VERSION, "components": model.components,
        "component_dim": model.adapter_dim // model.components,
        "source_probabilities": {t: model.source_routes[t].detach().softmax(-1).cpu().tolist()
                                 for t in model.source_tasks},
        "target_probabilities": {t: [a.logits.detach().softmax(-1).cpu().tolist() for a in stack]
                                 for t, stack in model.target_adapters.items()},
        "target_banks_frozen": all(not p.requires_grad for stack in model.target_adapters.values()
                                   for a in stack for p in a.bank.parameters()),
    }}


def validate_composition_checkpoint(saved, config):
    from .qk_adapter import validate_qk_checkpoint
    validate_qk_checkpoint(saved, config)
    expected = composition_settings(config)
    recorded = saved.get("composition")
    if bool(recorded) != expected.get("enabled", False):
        raise ValueError("Compositional source checkpoint required; legacy sources cannot be reused (or vice versa)")
    if recorded:
        components = expected.get("components", 4)
        if (recorded.get("version") != VERSION or recorded.get("components") != components or
                recorded.get("component_dim") * components != config["model"].get("adapter_dim", 96)):
            raise ValueError("Compositional checkpoint version or width mismatch")


def source_route_optimizer_groups(model, learning_rate):
    if not isinstance(model, CompositionalSourceModel):
        return []
    return [("source_routes", list(model.source_routes.named_parameters()), learning_rate)]

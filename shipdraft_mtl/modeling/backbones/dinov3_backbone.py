from __future__ import annotations

"""DINOv3 ViT and ConvNeXt backbone adapters for DraftFormer."""

import math
import os
from collections.abc import Mapping, Sequence
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from shipdraft_mtl.modeling.common import ShapeSpec

__all__ = [
    "DINOv3ViTBackbone",
    "DINOv3ConvNeXtBackbone",
    "ViTBackbone",
    "ConvNeXtBackbone",
]


_FEATURE_NAMES = ("res2", "res3", "res4", "res5")
_FEATURE_STRIDES = {
    "res2": 4,
    "res3": 8,
    "res4": 16,
    "res5": 32,
}
_CONVNEXT_CHANNELS = {
    "dinov3_convnext_tiny": (96, 192, 384, 768),
    "dinov3_convnext_small": (96, 192, 384, 768),
    "dinov3_convnext_base": (128, 256, 512, 1024),
    "dinov3_convnext_large": (192, 384, 768, 1536),
}


def _validate_out_features(out_features):
    out_features = list(out_features or _FEATURE_NAMES)
    unsupported = [name for name in out_features if name not in _FEATURE_NAMES]
    if unsupported:
        raise ValueError(
            f"Unsupported DINOv3 output features: {unsupported}. "
            f"Supported features are {list(_FEATURE_NAMES)}."
        )
    return out_features


def _load_dinov3_hub_model(
    *,
    model_name: str,
    repo_or_dir: str,
    source: str,
    pretrained: bool,
    weights: Optional[str],
    model_kwargs: Optional[dict],
    force_reload: bool,
    trust_repo: Optional[bool],
    skip_validation: bool,
    verbose: bool,
    allow_remote_code: bool,
):
    if source == "github" and not allow_remote_code:
        raise ValueError(
            "source='github' executes repository code. Clone the repository and use "
            "source='local', or explicitly set allow_remote_code=True for a trusted revision."
        )
    entrypoint_kwargs = dict(model_kwargs or {})
    entrypoint_kwargs["pretrained"] = pretrained
    if weights not in (None, ""):
        entrypoint_kwargs["weights"] = weights

    hub_kwargs = {
        "source": source,
        "force_reload": force_reload,
        "verbose": verbose,
    }
    if source == "github":
        hub_kwargs["skip_validation"] = skip_validation
        if trust_repo is not None:
            hub_kwargs["trust_repo"] = trust_repo

    try:
        return torch.hub.load(repo_or_dir, model_name, **hub_kwargs, **entrypoint_kwargs)
    except TypeError as exc:
        message = str(exc)
        if "trust_repo" in message or "skip_validation" in message:
            hub_kwargs.pop("trust_repo", None)
            hub_kwargs.pop("skip_validation", None)
            return torch.hub.load(repo_or_dir, model_name, **hub_kwargs, **entrypoint_kwargs)
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load DINOv3 model {model_name!r} from {repo_or_dir!r}. "
            "Clone https://github.com/facebookresearch/dinov3 and set "
            "repo_or_dir to that path with source='local', or use source='github' "
            "with network access. Pass a local .pth file through weights when "
            "you want to use downloaded DINOv3 pretrained weights."
        ) from exc


def _resolve_local_path(path: str) -> str:
    """Resolve a local checkpoint path relative to common project roots."""
    if not path or os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)

    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    candidates = [
        os.path.join(os.getcwd(), path),
        os.path.join(package_root, path),
        os.path.join(package_root, "shipdraft_mtl", path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return path


def _load_dinov3_huggingface_model(
    repo_or_dir: str,
    *,
    local_files_only: bool,
    trust_remote_code: bool,
):
    """Load a DINOv3 checkpoint saved in Hugging Face ``from_pretrained`` format."""
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "Loading DINOv3 Hugging Face checkpoints requires transformers>=4.56.0. "
            "Install the project dependencies with `pip install -r requirements.txt`."
        ) from exc

    resolved = _resolve_local_path(repo_or_dir)
    try:
        return AutoModel.from_pretrained(
            resolved,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load DINOv3 Hugging Face checkpoint from {repo_or_dir!r} "
            f"(resolved={resolved!r}). "
            "The directory must contain config.json and model.safetensors (or pytorch_model.bin)."
        ) from exc


def _make_projection(in_channels: int, out_channels: int):
    if in_channels == out_channels:
        return nn.Identity()
    projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    nn.init.xavier_uniform_(projection.weight)
    nn.init.constant_(projection.bias, 0)
    return projection


def _resolve_out_channels(out_channels, out_features, default_channels):
    if out_channels is None:
        return {name: int(default_channels[name]) for name in out_features}
    if isinstance(out_channels, int):
        return {name: out_channels for name in out_features}
    if isinstance(out_channels, Mapping):
        return {name: int(out_channels.get(name, default_channels[name])) for name in out_features}

    channels = list(out_channels)
    if len(channels) != len(out_features):
        raise ValueError(
            "out_channels must be an int, a mapping, or a sequence with the "
            "same length as out_features."
        )
    return {name: int(channel) for name, channel in zip(out_features, channels)}


def _freeze_module(module: nn.Module, frozen_modules):
    for parameter in module.parameters():
        parameter.requires_grad = False
    module.eval()
    frozen_modules.append(module)


def _resize_feature(x, size, mode):
    if x.shape[-2:] == size:
        return x
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        return F.interpolate(x, size=size, mode=mode, align_corners=False)
    return F.interpolate(x, size=size, mode=mode)


def _as_int_patch_size(patch_size):
    if isinstance(patch_size, Sequence) and not isinstance(patch_size, (str, bytes)):
        if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
            raise ValueError(f"Only square DINOv3 patch sizes are supported, got {patch_size}.")
        return int(patch_size[0])
    return int(patch_size)


def _evenly_spaced_indices(count: int, total: int):
    if total <= 0:
        raise ValueError("DINOv3 ViT model has no transformer blocks.")
    indices = []
    for idx in range(count):
        block_idx = int(round((idx + 1) * total / count) - 1)
        indices.append(max(0, min(total - 1, block_idx)))
    return indices


class _FrozenTrainMixin:
    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for module in getattr(self, "_frozen_modules", []):
                module.eval()
        return self


class DINOv3ViTBackbone(_FrozenTrainMixin, nn.Module):
    """DraftFormer adapter for DINOv3 Vision Transformer backbones.

    The official DINOv3 ViT produces patch-token maps at one native stride
    (normally 16). This adapter samples intermediate transformer blocks and
    resizes them into DraftFormer's conventional res2-res5 feature pyramid.
    """

    def __init__(
        self,
        in_channels=3,
        model_name="dinov3_vitb16",
        repo_or_dir="facebookresearch/dinov3",
        source="github",
        pretrained=True,
        weights=None,
        model_kwargs=None,
        out_features=None,
        layer_indices=None,
        out_channels=None,
        norm=True,
        auto_pad=True,
        interpolate_mode="bilinear",
        freeze_at=0,
        force_reload=False,
        trust_repo=True,
        skip_validation=False,
        verbose=True,
        allow_remote_code=False,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("DINOv3 pretrained ViT backbones expect in_channels=3.")

        self.out_features = _validate_out_features(out_features)
        self.norm = norm
        self.auto_pad = auto_pad
        self.interpolate_mode = interpolate_mode
        self._frozen_modules = []

        self.model = _load_dinov3_hub_model(
            model_name=model_name,
            repo_or_dir=repo_or_dir,
            source=source,
            pretrained=pretrained,
            weights=weights,
            model_kwargs=model_kwargs,
            force_reload=force_reload,
            trust_repo=trust_repo,
            skip_validation=skip_validation,
            verbose=verbose,
            allow_remote_code=allow_remote_code,
        )

        self.patch_size = _as_int_patch_size(getattr(self.model, "patch_size", 16))
        self._out_feature_strides = dict(_FEATURE_STRIDES)
        self._feature_layer_indices = self._resolve_layer_indices(layer_indices)

        embed_dim = int(getattr(self.model, "embed_dim", getattr(self.model, "num_features", 0)))
        if embed_dim <= 0:
            raise ValueError("Could not infer DINOv3 ViT embedding dimension.")
        default_channels = {name: embed_dim for name in _FEATURE_NAMES}
        self._out_feature_channels = _resolve_out_channels(
            out_channels, self.out_features, default_channels
        )
        self.projections = nn.ModuleDict(
            {
                name: _make_projection(embed_dim, self._out_feature_channels[name])
                for name in self.out_features
            }
        )

        self._freeze(freeze_at)

    def _resolve_layer_indices(self, layer_indices):
        total_blocks = int(
            getattr(
                self.model,
                "n_blocks",
                len(getattr(self.model, "blocks", [])),
            )
        )
        if layer_indices is None:
            canonical = dict(zip(_FEATURE_NAMES, _evenly_spaced_indices(4, total_blocks)))
            return {name: canonical[name] for name in self.out_features}
        if isinstance(layer_indices, Mapping):
            return {name: int(layer_indices[name]) for name in self.out_features}

        indices = list(layer_indices)
        if len(indices) != len(self.out_features):
            raise ValueError("layer_indices must match the length of out_features.")
        return {name: int(index) for name, index in zip(self.out_features, indices)}

    def _freeze(self, freeze_at: int):
        if freeze_at <= 0:
            return

        token_names = ("cls_token", "mask_token", "storage_tokens")
        for name in token_names:
            token = getattr(self.model, name, None)
            if token is not None:
                token.requires_grad = False

        patch_embed = getattr(self.model, "patch_embed", None)
        if patch_embed is not None:
            _freeze_module(patch_embed, self._frozen_modules)

        blocks = list(getattr(self.model, "blocks", []))
        for block in blocks[: max(0, freeze_at - 1)]:
            _freeze_module(block, self._frozen_modules)

        if freeze_at >= len(blocks) + 1:
            for name in ("norm", "cls_norm", "local_cls_norm"):
                module = getattr(self.model, name, None)
                if module is not None:
                    _freeze_module(module, self._frozen_modules)

    def _pad_input(self, x):
        if not self.auto_pad:
            return x, x.shape[-2:]
        height, width = x.shape[-2:]
        pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - width % self.patch_size) % self.patch_size
        if pad_h == 0 and pad_w == 0:
            return x, (height, width)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (height + pad_h, width + pad_w)

    @staticmethod
    def _target_size(padded_size, stride):
        height, width = padded_size
        return (
            max(1, int(math.ceil(float(height) / stride))),
            max(1, int(math.ceil(float(width) / stride))),
        )

    def forward(self, x):
        x, padded_size = self._pad_input(x)
        layer_indices = sorted(set(self._feature_layer_indices.values()))
        features = self.model.get_intermediate_layers(
            x,
            n=layer_indices,
            reshape=True,
            return_class_token=False,
            return_extra_tokens=False,
            norm=self.norm,
        )
        feature_by_layer = dict(zip(layer_indices, features))

        outputs = {}
        for name in self.out_features:
            feature = feature_by_layer[self._feature_layer_indices[name]]
            target_size = self._target_size(padded_size, self._out_feature_strides[name])
            feature = _resize_feature(feature, target_size, self.interpolate_mode)
            outputs[name] = self.projections[name](feature)
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self.out_features
        }


class DINOv3ConvNeXtBackbone(_FrozenTrainMixin, nn.Module):
    """DraftFormer adapter for DINOv3 ConvNeXt backbones."""

    def __init__(
        self,
        in_channels=3,
        model_name="dinov3_convnext_base",
        repo_or_dir="facebookresearch/dinov3",
        source="github",
        pretrained=True,
        weights=None,
        model_kwargs=None,
        out_features=None,
        out_channels=None,
        norm=True,
        freeze_at=0,
        force_reload=False,
        trust_repo=True,
        skip_validation=False,
        verbose=True,
        local_files_only=True,
        trust_remote_code=False,
        allow_remote_code=False,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("DINOv3 pretrained ConvNeXt backbones expect in_channels=3.")

        self.out_features = _validate_out_features(out_features)
        self.norm = norm
        self._frozen_modules = []

        self._huggingface_model = source.lower() in {"hf", "huggingface"}
        if self._huggingface_model:
            if not pretrained:
                raise ValueError("Hugging Face DINOv3 ConvNeXt loading requires pretrained=True.")
            if weights not in (None, "") or model_kwargs:
                raise ValueError(
                    "For source='huggingface', put the local checkpoint directory in "
                    "repo_or_dir instead of using weights or model_kwargs."
                )
            self.model = _load_dinov3_huggingface_model(
                repo_or_dir,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        else:
            self.model = _load_dinov3_hub_model(
                model_name=model_name,
                repo_or_dir=repo_or_dir,
                source=source,
                pretrained=pretrained,
                weights=weights,
                model_kwargs=model_kwargs,
                force_reload=force_reload,
                trust_repo=trust_repo,
                skip_validation=skip_validation,
                verbose=verbose,
                allow_remote_code=allow_remote_code,
            )

        self._stage_indices = {name: index for index, name in enumerate(_FEATURE_NAMES)}
        self._out_feature_strides = dict(_FEATURE_STRIDES)

        embed_dims = getattr(self.model, "embed_dims", None)
        if embed_dims is None:
            config = getattr(self.model, "config", None)
            hidden_sizes = getattr(config, "hidden_sizes", None) if config is not None else None
            if hidden_sizes is not None:
                embed_dims = tuple(int(v) for v in hidden_sizes)
        if embed_dims is None:
            embed_dims = _CONVNEXT_CHANNELS.get(model_name)
        if embed_dims is None:
            raise ValueError(
                "Could not infer DINOv3 ConvNeXt feature channels. "
                "Pass out_channels explicitly for custom ConvNeXt variants."
            )
        default_channels = {
            name: int(embed_dims[self._stage_indices[name]]) for name in _FEATURE_NAMES
        }
        self._backbone_feature_channels = default_channels
        self._out_feature_channels = _resolve_out_channels(
            out_channels, self.out_features, default_channels
        )
        self.projections = nn.ModuleDict(
            {
                name: _make_projection(
                    default_channels[name], self._out_feature_channels[name]
                )
                for name in self.out_features
            }
        )

        self._freeze(freeze_at)

    def _freeze(self, freeze_at: int):
        if freeze_at <= 0:
            return
        if self._huggingface_model:
            embeddings = getattr(self.model, "embeddings", None)
            if embeddings is not None:
                _freeze_module(embeddings, self._frozen_modules)
            encoder = getattr(self.model, "encoder", None)
            stages = list(getattr(encoder, "stages", []))
            for stage in stages[:freeze_at]:
                _freeze_module(stage, self._frozen_modules)
            return

        downsample_layers = list(getattr(self.model, "downsample_layers", []))
        stages = list(getattr(self.model, "stages", []))
        for index in range(min(freeze_at, len(stages))):
            if index < len(downsample_layers):
                _freeze_module(downsample_layers[index], self._frozen_modules)
            _freeze_module(stages[index], self._frozen_modules)

        if freeze_at >= len(stages):
            norm = getattr(self.model, "norm", None)
            if norm is not None:
                _freeze_module(norm, self._frozen_modules)

    def forward(self, x):
        if self._huggingface_model:
            return self._forward_huggingface(x)

        stage_indices = sorted({self._stage_indices[name] for name in self.out_features})
        features = self.model.get_intermediate_layers(
            x,
            n=stage_indices,
            reshape=True,
            return_class_token=False,
            norm=self.norm,
        )
        feature_by_stage = dict(zip(stage_indices, features))

        outputs = {}
        for name in self.out_features:
            feature = feature_by_stage[self._stage_indices[name]]
            outputs[name] = self.projections[name](feature)
        return outputs

    def _forward_huggingface(self, x):
        outputs = self.model(x, output_hidden_states=True, return_dict=True)
        hidden_states = tuple(
            feature for feature in (outputs.hidden_states or ()) if feature.ndim == 4
        )
        if not hidden_states:
            raise RuntimeError("DINOv3 Hugging Face model did not return 2D hidden states.")

        # HF ConvNeXt exposes the stem plus one state per stage.  Match by the
        # known stage channel width so an optional duplicate stem is ignored.
        stage_features = {}
        for name in self.out_features:
            expected_channels = self._backbone_feature_channels[name]
            candidates = [
                feature for feature in hidden_states if feature.shape[1] == expected_channels
            ]
            if not candidates:
                raise RuntimeError(
                    f"Could not find DINOv3 ConvNeXt stage {name} with "
                    f"{expected_channels} channels in Hugging Face outputs."
                )
            stage_features[name] = candidates[-1]

        return {
            name: self.projections[name](stage_features[name])
            for name in self.out_features
        }

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self.out_features
        }


ViTBackbone = DINOv3ViTBackbone
ConvNeXtBackbone = DINOv3ConvNeXtBackbone

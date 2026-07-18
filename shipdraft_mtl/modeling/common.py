from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ShapeSpec:
    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            scale = self.weight.reshape(1, -1, 1, 1) * (
                self.running_var.reshape(1, -1, 1, 1) + self.eps
            ).rsqrt()
            bias = self.bias.reshape(1, -1, 1, 1) - self.running_mean.reshape(1, -1, 1, 1) * scale
            return x * scale + bias
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=False,
            eps=self.eps,
        )


def get_norm(norm: Optional[str], out_channels: int):
    if norm is None or norm == "":
        return None
    norm = norm.upper()
    if norm == "BN":
        return nn.BatchNorm2d(out_channels)
    if norm == "SYNCBN":
        return nn.SyncBatchNorm(out_channels)
    if norm == "FROZENBN":
        return FrozenBatchNorm2d(out_channels)
    if norm == "GN":
        return nn.GroupNorm(32, out_channels)
    raise ValueError(f"Unsupported norm: {norm}")


class Conv2d(nn.Conv2d):
    def __init__(
        self,
        *args,
        norm: Optional[nn.Module] = None,
        activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.size_divisibility = 32

    def output_shape(self) -> Dict[str, ShapeSpec]:
        raise NotImplementedError


class CNNBlockBase(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride


class BasicStem(nn.Sequential):
    def __init__(self, in_channels=3, out_channels=64, norm="FrozenBN"):
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            get_norm(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]
        super().__init__(*[module for module in modules if module is not None])
        self.out_channels = out_channels
        self.stride = 4


class BottleneckBlock(CNNBlockBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int,
        stride: int = 1,
        num_groups: int = 1,
        norm: str = "FrozenBN",
        stride_in_1x1: bool = False,
    ):
        super().__init__(in_channels, out_channels, stride)
        stride_1x1 = stride if stride_in_1x1 else 1
        stride_3x3 = 1 if stride_in_1x1 else stride

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=stride_1x1, bias=False)
        self.norm1 = get_norm(norm, bottleneck_channels)
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1,
            bias=False,
            groups=num_groups,
        )
        self.norm2 = get_norm(norm, bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.norm3 = get_norm(norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = None
        if in_channels != out_channels or stride != 1:
            modules = [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                get_norm(norm, out_channels),
            ]
            self.shortcut = nn.Sequential(*[module for module in modules if module is not None])

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.norm3 is not None:
            out = self.norm3(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        return self.relu(out + identity)


class ResNet(Backbone):
    _STAGE_NAMES = ("res2", "res3", "res4", "res5")

    def __init__(
        self,
        stem: nn.Module,
        stages: Iterable[nn.Module],
        num_classes: Optional[int] = None,
        out_features: Optional[Iterable[str]] = None,
        freeze_at: int = 0,
    ):
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes
        self.out_features = list(out_features or self._STAGE_NAMES)

        current_stride = getattr(stem, "stride", 4)
        current_channels = getattr(stem, "out_channels", None)
        self._out_feature_strides: Dict[str, int] = {}
        self._out_feature_channels: Dict[str, int] = {}

        for name, stage in zip(self._STAGE_NAMES, stages):
            setattr(self, name, stage)
            stage_stride = 1
            stage_channels = current_channels
            for module in stage.modules():
                if isinstance(module, CNNBlockBase):
                    stage_stride *= module.stride
                    stage_channels = module.out_channels
                    break
            current_stride *= stage_stride
            current_channels = stage_channels
            self._out_feature_strides[name] = current_stride
            self._out_feature_channels[name] = current_channels

        self._freeze(freeze_at)
        self._init_weights()

    def _freeze(self, freeze_at: int):
        stages = [self.stem] + [getattr(self, name) for name in self._STAGE_NAMES]
        for stage in stages[: max(0, freeze_at)]:
            for param in stage.parameters():
                param.requires_grad = False
            stage.eval()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        for name in self._STAGE_NAMES:
            x = getattr(self, name)(x)
            if name in self.out_features:
                outputs[name] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self.out_features
        }

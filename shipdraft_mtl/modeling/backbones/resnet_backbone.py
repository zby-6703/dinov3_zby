from __future__ import annotations

"""ResNet backbone component for DraftFormer."""

import torch.nn as nn

from shipdraft_mtl.modeling.common import BasicStem, BottleneckBlock, ResNet, ShapeSpec

__all__ = ["ResNetBackbone"]


def _build_resnet(
    input_shape: ShapeSpec,
    *,
    freeze_at: int = 0,
    depth: int = 50,
    norm: str = "FrozenBN",
    stem_out_channels: int = 64,
    num_groups: int = 1,
    width_per_group: int = 64,
    stride_in_1x1: bool = False,
    res2_out_channels: int = 256,
    out_features=None,
):
    if depth == 50:
        num_blocks_per_stage = [3, 4, 6, 3]
    elif depth == 101:
        num_blocks_per_stage = [3, 4, 23, 3]
    else:
        raise ValueError(f"Unsupported ResNet depth: {depth}")

    stem = BasicStem(
        in_channels=input_shape.channels or 3,
        out_channels=stem_out_channels,
        norm=norm,
    )

    bottleneck_channels = num_groups * width_per_group
    stages = []
    in_channels = stem_out_channels
    out_channels = res2_out_channels
    for stage_idx, num_blocks in enumerate(num_blocks_per_stage):
        stage_stride = 1 if stage_idx == 0 else 2
        stage_blocks = []
        for block_idx in range(num_blocks):
            block_stride = stage_stride if block_idx == 0 else 1
            stage_blocks.append(
                BottleneckBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bottleneck_channels=bottleneck_channels * (2 ** stage_idx),
                    stride=block_stride,
                    num_groups=num_groups,
                    norm=norm,
                    stride_in_1x1=stride_in_1x1,
                )
            )
            in_channels = out_channels
        stages.append(nn.Sequential(*stage_blocks))
        out_channels *= 2

    return ResNet(
        stem=stem,
        stages=stages,
        num_classes=None,
        out_features=list(out_features or ["res2", "res3", "res4", "res5"]),
        freeze_at=freeze_at,
    )


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        in_channels=3,
        freeze_at=0,
        depth=50,
        norm="FrozenBN",
        stem_out_channels=64,
        num_groups=1,
        width_per_group=64,
        stride_in_1x1=False,
        res2_out_channels=256,
        out_features=None,
    ):
        super().__init__()
        self.backbone = _build_resnet(
            ShapeSpec(channels=in_channels),
            freeze_at=freeze_at,
            depth=depth,
            norm=norm,
            stem_out_channels=stem_out_channels,
            num_groups=num_groups,
            width_per_group=width_per_group,
            stride_in_1x1=stride_in_1x1,
            res2_out_channels=res2_out_channels,
            out_features=out_features,
        )

    def forward(self, x):
        return self.backbone(x)

    def output_shape(self):
        return self.backbone.output_shape()

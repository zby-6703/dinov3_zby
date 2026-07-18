# ------------------------------------------------------------------------------------------------
# Pure PyTorch multi-scale deformable attention.
# This module keeps the Deformable DETR MSDeformAttn interface, but does not
# depend on the compiled pixel_decoder/ops extension.
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import math
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


def _shape_as_int_tuple(shape):
    if isinstance(shape, torch.Tensor):
        shape = shape.detach().cpu().tolist()
    return tuple(int(v) for v in shape)


def _as_spatial_shapes(value_spatial_shapes):
    return [_shape_as_int_tuple(shape) for shape in value_spatial_shapes]


def _validate_flattened_length(value_len, spatial_shapes):
    split_shape = [H * W for H, W in spatial_shapes]
    if sum(split_shape) != value_len:
        raise ValueError(
            "value_spatial_shapes must describe the flattened value length: "
            "got {} from shapes, but value has length {}".format(sum(split_shape), value_len)
        )
    return split_shape


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Deformable-DETR style core. All feature levels use the same number of
    sampling points.

    Args:
        value: (N, sum(H_l * W_l), n_heads, head_dim)
        value_spatial_shapes: (n_levels, 2), each item is (H_l, W_l)
        sampling_locations: (N, Len_q, n_heads, n_levels, n_points, 2)
        attention_weights: (N, Len_q, n_heads, n_levels, n_points)

    Returns:
        Tensor with shape (N, Len_q, n_heads * head_dim).
    """
    N, value_len, n_heads, head_dim = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    spatial_shapes = _as_spatial_shapes(value_spatial_shapes)
    split_shape = _validate_flattened_length(value_len, spatial_shapes)

    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    for level, (H, W) in enumerate(spatial_shapes):
        value_l = value_list[level].flatten(2).transpose(1, 2).reshape(
            N * n_heads, head_dim, H, W
        )
        sampling_grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l = F.grid_sample(
            value_l,
            sampling_grid_l,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l)

    attention_weights = attention_weights.transpose(1, 2).reshape(
        N * n_heads, 1, Len_q, n_levels * n_points
    )
    output = (
        torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights
    ).sum(-1).view(N, n_heads * head_dim, Len_q)

    return output.transpose(1, 2).contiguous()


def ms_deform_attn_core_pytorch_v2(
    value,
    value_spatial_shapes,
    sampling_locations,
    attention_weights,
    num_points_list,
    method="default",
):
    """
    RT-DETRv2 style core. Sampling points are flattened across feature levels,
    so each level can use a different number of points.

    Args:
        value: (N, sum(H_l * W_l), n_heads, head_dim)
        value_spatial_shapes: (n_levels, 2), each item is (H_l, W_l)
        sampling_locations: (N, Len_q, n_heads, sum(num_points_list), 2)
        attention_weights: (N, Len_q, n_heads, sum(num_points_list))
        num_points_list: list[int], sampling points per level.
        method: "default" for bilinear grid_sample, "discrete" for integer lookup.

    Returns:
        Tensor with shape (N, Len_q, n_heads * head_dim).
    """
    N, value_len, n_heads, head_dim = value.shape
    _, Len_q, _, total_points, _ = sampling_locations.shape

    spatial_shapes = _as_spatial_shapes(value_spatial_shapes)
    num_points_list = [int(n) for n in num_points_list]
    if len(spatial_shapes) != len(num_points_list):
        raise ValueError(
            "value_spatial_shapes has {} levels, but num_points_list has {}".format(
                len(spatial_shapes), len(num_points_list)
            )
        )
    if sum(num_points_list) != total_points:
        raise ValueError(
            "sampling_locations has {} flattened points, but num_points_list sums to {}".format(
                total_points, sum(num_points_list)
            )
        )

    split_shape = _validate_flattened_length(value_len, spatial_shapes)
    value_list = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)

    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations
    else:
        raise ValueError("method must be 'default' or 'discrete', but got {}".format(method))

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_grids_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (H, W) in enumerate(spatial_shapes):
        value_l = value_list[level].reshape(N * n_heads, head_dim, H, W)
        sampling_grid_l = sampling_grids_list[level]

        if method == "default":
            sampling_value_l = F.grid_sample(
                value_l,
                sampling_grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        else:
            sampling_coord = (
                sampling_grid_l
                * torch.tensor([W, H], device=value.device, dtype=sampling_grid_l.dtype)
                + 0.5
            ).to(torch.int64)
            sampling_x = sampling_coord[..., 0].clamp(0, W - 1)
            sampling_y = sampling_coord[..., 1].clamp(0, H - 1)
            sampling_x = sampling_x.reshape(N * n_heads, Len_q * num_points_list[level])
            sampling_y = sampling_y.reshape(N * n_heads, Len_q * num_points_list[level])
            batch_idx = torch.arange(N * n_heads, device=value.device).unsqueeze(-1)
            batch_idx = batch_idx.expand_as(sampling_x)
            sampling_value_l = value_l[batch_idx, :, sampling_y, sampling_x]
            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(
                N * n_heads, head_dim, Len_q, num_points_list[level]
            )

        sampling_value_list.append(sampling_value_l)

    attention_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        N * n_heads, 1, Len_q, sum(num_points_list)
    )
    output = (torch.cat(sampling_value_list, dim=-1) * attention_weights).sum(-1)
    output = output.reshape(N, n_heads * head_dim, Len_q)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_levels=4,
        n_heads=8,
        n_points=4,
        method="default",
        offset_scale=0.5,
    ):
        """
        Multi-scale deformable attention module.

        Args:
            d_model: hidden dimension.
            n_levels: number of feature levels.
            n_heads: number of attention heads.
            n_points: int or list[int], sampling points per feature level.
            method: "default" uses bilinear sampling, "discrete" uses integer lookup.
            offset_scale: scale used for 4D box reference points, matching RT-DETRv2.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads, but got {} and {}".format(d_model, n_heads))
        if method not in ("default", "discrete"):
            raise ValueError("method must be 'default' or 'discrete', but got {}".format(method))

        d_per_head = d_model // n_heads
        if not _is_power_of_2(d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension "
                "of each attention head a power of 2 for better memory layout."
            )

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.method = method
        self.offset_scale = offset_scale

        if isinstance(n_points, (list, tuple)):
            if len(n_points) != n_levels:
                raise ValueError("n_points list length must equal n_levels")
            self.num_points_list = [int(n) for n in n_points]
        else:
            self.num_points_list = [int(n_points) for _ in range(n_levels)]

        num_points_scale = [1.0 / n for n in self.num_points_list for _ in range(n)]
        self.register_buffer(
            "num_points_scale",
            torch.tensor(num_points_scale, dtype=torch.float32),
        )
        self.total_points = sum(self.num_points_list)

        self.sampling_offsets = nn.Linear(d_model, n_heads * self.total_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * self.total_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()
        if self.method == "discrete":
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        ).view(self.n_heads, 1, 2).repeat(1, self.total_points, 1)
        scaling = torch.cat(
            [torch.arange(1, n + 1, dtype=torch.float32) for n in self.num_points_list]
        ).view(1, -1, 1)
        grid_init *= scaling
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def _sampling_locations(self, reference_points, sampling_offsets, spatial_shapes):
        if reference_points.shape[-1] == 2:
            sampling_locations = []
            start = 0
            for level, num_points in enumerate(self.num_points_list):
                end = start + num_points
                ref_level = level if reference_points.shape[2] > 1 else 0
                normalizer = torch.stack(
                    [spatial_shapes[level, 1], spatial_shapes[level, 0]], -1
                ).to(dtype=sampling_offsets.dtype)
                sampling_locations.append(
                    reference_points[:, :, None, ref_level, None, :]
                    + sampling_offsets[:, :, :, start:end, :] / normalizer
                )
                start = end
            return torch.cat(sampling_locations, dim=3)

        if reference_points.shape[-1] == 4:
            sampling_locations = []
            start = 0
            for level, num_points in enumerate(self.num_points_list):
                end = start + num_points
                ref_level = level if reference_points.shape[2] > 1 else 0
                scale = self.num_points_scale[start:end].to(dtype=sampling_offsets.dtype)
                sampling_locations.append(
                    reference_points[:, :, None, ref_level, None, :2]
                    + sampling_offsets[:, :, :, start:end, :]
                    * scale.view(1, 1, 1, -1, 1)
                    * reference_points[:, :, None, ref_level, None, 2:]
                    * self.offset_scale
                )
                start = end
            return torch.cat(sampling_locations, dim=3)

        raise ValueError(
            "Last dim of reference_points must be 2 or 4, but got {}".format(
                reference_points.shape[-1]
            )
        )

    def _forward_impl(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_padding_mask=None,
        padding_mask_is_valid=False,
    ):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        input_spatial_shapes = torch.as_tensor(
            input_spatial_shapes,
            dtype=torch.long,
            device=query.device,
        )

        expected_len = int((input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum().item())
        if expected_len != Len_in:
            raise ValueError(
                "input_spatial_shapes describe length {}, but input_flatten has length {}".format(
                    expected_len, Len_in
                )
            )

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            mask = input_padding_mask.to(dtype=value.dtype).unsqueeze(-1)
            if padding_mask_is_valid:
                value = value * mask
            else:
                value = value.masked_fill(mask.to(torch.bool), 0.0)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.total_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.total_points
        )
        attention_weights = F.softmax(attention_weights, -1)

        sampling_locations = self._sampling_locations(
            reference_points,
            sampling_offsets,
            input_spatial_shapes,
        )
        output = ms_deform_attn_core_pytorch_v2(
            value,
            input_spatial_shapes,
            sampling_locations,
            attention_weights,
            self.num_points_list,
            method=self.method,
        )
        return self.output_proj(output)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index=None,
        input_padding_mask=None,
    ):
        """
        Args:
            query: (N, Len_q, C)
            reference_points: (N, Len_q, n_levels, 2/4), normalized to [0, 1].
            input_flatten: (N, sum(H_l * W_l), C)
            input_spatial_shapes: (n_levels, 2), each item is (H_l, W_l).
            input_level_start_index: kept for API compatibility; not needed here.
            input_padding_mask: (N, sum(H_l * W_l)), True for padded elements.

        Returns:
            Tensor with shape (N, Len_q, C).
        """
        return self._forward_impl(
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_padding_mask=input_padding_mask,
            padding_mask_is_valid=False,
        )


class RTDETRMSDeformableAttention(MSDeformAttn):
    """RT-DETR/RT-DETRv2 compatible wrapper.

    The forward signature follows RT-DETR: ``value_mask`` is a valid-position
    mask, not a padding mask.
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method="default",
        offset_scale=0.5,
    ):
        super().__init__(
            d_model=embed_dim,
            n_levels=num_levels,
            n_heads=num_heads,
            n_points=num_points,
            method=method,
            offset_scale=offset_scale,
        )

    def forward(
        self,
        query,
        reference_points,
        value,
        value_spatial_shapes,
        value_mask=None,
    ):
        return self._forward_impl(
            query,
            reference_points,
            value,
            value_spatial_shapes,
            input_padding_mask=value_mask,
            padding_mask_is_valid=True,
        )


__all__ = [
    "MSDeformAttn",
    "RTDETRMSDeformableAttention",
    "ms_deform_attn_core_pytorch",
    "ms_deform_attn_core_pytorch_v2",
]

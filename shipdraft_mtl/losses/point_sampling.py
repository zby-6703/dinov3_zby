from __future__ import annotations

"""PointRend-style point sampling helpers used by mask matching and mask loss."""

import torch
import torch.nn.functional as F


def point_sample(input: torch.Tensor, point_coords: torch.Tensor, align_corners: bool = False):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(
        input,
        2.0 * point_coords - 1.0,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(
    coarse_logits: torch.Tensor,
    uncertainty_func,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
):
    if num_points <= 0:
        raise ValueError("num_points must be positive")

    num_boxes = coarse_logits.shape[0]
    num_sampled = max(int(num_points * oversample_ratio), num_points)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    uncertainties = uncertainty_func(point_logits).squeeze(1)

    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_uncertain_points = min(num_uncertain_points, num_points)
    num_random_points = num_points - num_uncertain_points

    if num_uncertain_points > 0:
        idx = torch.topk(uncertainties, k=num_uncertain_points, dim=1)[1]
        batch_idx = torch.arange(num_boxes, device=coarse_logits.device)[:, None]
        uncertain_coords = point_coords[batch_idx, idx]
    else:
        uncertain_coords = point_coords[:, :0]

    if num_random_points > 0:
        random_coords = torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device)
        return torch.cat([uncertain_coords, random_coords], dim=1)
    return uncertain_coords

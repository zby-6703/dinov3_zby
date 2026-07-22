# ------------------------------------------------------------------------
# DraftFormer Loss Criterion
# ------------------------------------------------------------------------
# Multi-task loss computation for detection and segmentation.
#
# Historical Note:
#   Originally derived from DINO/MaskDINO (IDEA, 2022). The denoising (DN)
#   training components have been removed for a cleaner implementation.
# ------------------------------------------------------------------------
"""
DraftFormer criterion for multi-task learning.
"""
import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from shipdraft_mtl.losses.point_sampling import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from shipdraft_mtl.losses.task_balancer import DynamicTaskBalancer
from shipdraft_mtl.losses.matcher import HungarianMatcher

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from shipdraft_mtl.utils import box_ops


def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss


    return loss.mean(1).sum() / num_boxes


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, 
                 # 接收两个匹配器
                 matcher_det, matcher_seg,
                 weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, 
                 # 接收多任务相关的参数
                 num_detection_queries,
                 num_classes_seg=1, 
                 semantic_ce_loss=False,
                 # 动态任务平衡器
                 task_balancer=None,
                 keypoint_mode=False,
                 curve_smooth_weight=0.5,
                 train_character=True,
                 train_waterline=True):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            num_classes_seg: number of segmentation task categories
            matcher_det: matcher for detection task (includes box cost)
            matcher_seg: matcher for segmentation task (no box cost)
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_detection_queries: number of queries assigned to detection task.
            keypoint_mode: pure 2D character keypoints + ordered waterline curve points
            train_character / train_waterline: multi-stage task switches
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_classes_seg = num_classes_seg
        
        # 存储匹配器
        self.matcher_det = matcher_det
        self.matcher_seg = matcher_seg
        
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.keypoint_mode = bool(keypoint_mode)
        self.curve_smooth_weight = float(curve_smooth_weight)
        self.train_character = bool(train_character)
        self.train_waterline = bool(train_waterline)
        
        # 为不同的任务创建不同的eos分类权重
        # 检测任务
        empty_weight_det = torch.ones(self.num_classes + 1)
        empty_weight_det[-1] = self.eos_coef
        self.register_buffer("empty_weight_det", empty_weight_det)
        # 分割任务
        empty_weight_seg = torch.ones(self.num_classes_seg + 1)
        empty_weight_seg[-1] = self.eos_coef
        self.register_buffer("empty_weight_seg", empty_weight_seg)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.focal_alpha = 0.25

        self.semantic_ce_loss = semantic_ce_loss
        
        # 记录检测任务的查询数量
        self.num_detection_queries = num_detection_queries
        
        # 🔥【新增】动态任务平衡器
        self.task_balancer = task_balancer
        if task_balancer is not None:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"✅ SetCriterion: 启用动态任务平衡器 (方法: {task_balancer.method})")


    def loss_labels_ce(self, outputs, targets, indices, num_masks, task='det'):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        if task == 'det':
            num_classes = self.num_classes
            empty_weight = self.empty_weight_det
        else: # seg
            num_classes = self.num_classes_seg
            empty_weight = self.empty_weight_seg

        target_classes = torch.full(
            src_logits.shape[:2], num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, task='det', log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        if task == 'det':
            num_classes = self.num_classes
        else: # seg
            num_classes = self.num_classes_seg

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:num_classes] # Use num_classes for slicing
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Box or pure-keypoint regression.

        Keypoint mode uses L1 on 2D points (from pred_points / boxes[...,:2]).
        Legacy mode uses L1 + GIoU on cxcywh boxes.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if self.keypoint_mode or (
            "pred_points" in outputs and outputs["pred_points"] is not None
            and outputs["pred_boxes"].shape[-1] == 2
        ):
            src_points = outputs.get("pred_points")
            if src_points is None:
                src_points = outputs["pred_boxes"][..., :2]
            src_points = src_points[idx]
            target_points = []
            for t, (_, i) in zip(targets, indices):
                if "points" in t and t["points"] is not None and len(t["points"]):
                    target_points.append(t["points"][i])
                else:
                    target_points.append(t["boxes"][i][..., :2])
            target_points = torch.cat(target_points, dim=0)
            loss_bbox = F.l1_loss(src_points, target_points, reduction="none")
            losses = {
                "loss_bbox": loss_bbox.sum() / num_boxes,
                "loss_giou": src_points.new_zeros(()),
            }
            return losses

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_waterline_curve(self, outputs, targets):
        """Dynamic-length ordered waterline curve supervision.

        Max-N ordered slots (left-to-right). Each sample has a validity mask of
        length N: the first M entries are real curve points (M adaptive), the
        rest are padding (existence=0). Coordinate / smooth losses only apply on
        valid slots — so effective curve length is dynamic at both train and test.
        """
        k = self.num_detection_queries
        pred_points = outputs.get("pred_points")
        if pred_points is None:
            pred_points = outputs["pred_boxes"][..., :2]
        pred_logits = outputs["pred_logits"]
        nq = pred_points.shape[1]
        zero = pred_points.new_zeros(())
        if k >= nq:
            return {"loss_ce": zero, "loss_curve": zero, "loss_curve_smooth": zero}

        pred_curve = pred_points[:, k:]
        pred_exist = pred_logits[:, k:, 0]
        num_slots = pred_curve.shape[1]

        exist_losses = []
        curve_losses = []
        smooth_losses = []
        for batch_index, target in enumerate(targets):
            has_waterline = target.get("has_waterline")
            curve_gt = target.get("waterline_curve_points")
            valid = target.get("waterline_curve_valid")
            if has_waterline is None:
                has_waterline = curve_gt is not None and valid is not None and bool(valid.any()) if valid is not None else (
                    curve_gt is not None and len(curve_gt) > 0
                )
            else:
                has_waterline = bool(has_waterline.item() if torch.is_tensor(has_waterline) else has_waterline)

            if has_waterline and curve_gt is not None and len(curve_gt) > 0:
                gt = curve_gt.to(device=pred_curve.device, dtype=pred_curve.dtype)
                if gt.shape[0] != num_slots:
                    if gt.shape[0] == 0:
                        exist_target = pred_exist.new_zeros(num_slots)
                        exist_losses.append(
                            F.binary_cross_entropy_with_logits(pred_exist[batch_index], exist_target)
                        )
                        continue
                    # Pad / truncate GT to slot count while preserving leading real points.
                    padded = gt.new_zeros(num_slots, 2)
                    n_copy = min(num_slots, gt.shape[0])
                    padded[:n_copy] = gt[:n_copy]
                    gt = padded
                    if valid is None:
                        valid = torch.zeros(num_slots, dtype=torch.bool, device=gt.device)
                        valid[:n_copy] = True
                if valid is None:
                    # Backward compatible: all slots real.
                    valid = torch.ones(num_slots, dtype=torch.bool, device=gt.device)
                else:
                    valid = valid.to(device=gt.device).bool()
                    if valid.numel() != num_slots:
                        v = torch.zeros(num_slots, dtype=torch.bool, device=gt.device)
                        n_copy = min(num_slots, valid.numel())
                        v[:n_copy] = valid[:n_copy]
                        valid = v

                exist_target = valid.float()
                exist_losses.append(
                    F.binary_cross_entropy_with_logits(pred_exist[batch_index], exist_target)
                )
                if valid.any():
                    curve_losses.append(
                        F.l1_loss(pred_curve[batch_index][valid], gt[valid], reduction="mean")
                    )
                    # Smooth only across consecutive *valid* pairs (usually a prefix).
                    if valid.sum() >= 2:
                        valid_idx = valid.nonzero(as_tuple=False).flatten()
                        # Prefer a contiguous prefix for ordered waterlines.
                        if valid_idx.numel() >= 2 and (valid_idx[-1] - valid_idx[0] + 1 == valid_idx.numel()):
                            sl = slice(int(valid_idx[0]), int(valid_idx[-1]) + 1)
                            pred_delta = pred_curve[batch_index, sl][1:] - pred_curve[batch_index, sl][:-1]
                            gt_delta = gt[sl][1:] - gt[sl][:-1]
                            smooth_losses.append(F.smooth_l1_loss(pred_delta, gt_delta, beta=0.01))
            else:
                exist_target = pred_exist.new_zeros(num_slots)
                exist_losses.append(
                    F.binary_cross_entropy_with_logits(pred_exist[batch_index], exist_target)
                )

        return {
            "loss_ce": torch.stack(exist_losses).mean() if exist_losses else zero,
            "loss_curve": torch.stack(curve_losses).mean() if curve_losses else zero,
            "loss_curve_smooth": torch.stack(smooth_losses).mean() if smooth_losses else zero,
        }

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def split_targets_by_task(self, targets):
        """
        将targets按任务类型分离（基于category_id）
        
        检测任务: category_id < num_classes (0-8)
        分割任务: category_id >= num_classes (9) -> 重映射到 0
        
        【跨目标多任务核心设计】：
        - 检测任务的masks虽然是从bbox生成的矩形，但匹配时只使用 cost=["cls", "box"]
        - 分割任务的boxes虽然是从多边形计算的粗略框，但匹配时只使用 cost=["cls", "mask"]
        - 因此两个任务完全独立，不会相互干扰
        """
        targets_det = []
        targets_seg = []
        
        for target_per_image in targets:
            # Ensure target_per_image is a dictionary with 'labels'
            if not isinstance(target_per_image, dict) or 'labels' not in target_per_image:
                continue

            gt_labels = target_per_image['labels']
            det_mask = gt_labels < self.num_classes
            seg_mask = gt_labels >= self.num_classes
            
            # Common properties
            device = gt_labels.device
            
            # Create detection target
            targets_det.append({
                'labels': gt_labels[det_mask],
                'boxes': target_per_image['boxes'][det_mask],
                'masks': target_per_image['masks'][det_mask],
            })

            # Waterline is a single semantic mask, not one mask per sampled
            # polyline point. The mapper stores the rasterized target separately.
            waterline_mask = target_per_image.get("waterline_mask")
            if waterline_mask is not None and waterline_mask.any():
                seg_labels_remapped = torch.zeros(1, dtype=torch.int64, device=device)
                seg_masks = waterline_mask.unsqueeze(0)
                seg_boxes = torch.tensor([[0.5, 0.5, 1.0, 1.0]], dtype=torch.float32, device=device)
            else:
                seg_labels_remapped = torch.zeros(0, dtype=torch.int64, device=device)
                seg_masks = target_per_image["masks"][:0]
                seg_boxes = target_per_image["boxes"][:0]
            targets_seg.append({
                'labels': seg_labels_remapped,
                'boxes': seg_boxes,
                'masks': seg_masks,
            })

        return targets_det, targets_seg
        

    def get_loss(self, loss, outputs, targets, indices, num_masks, task='det'):
        loss_map = {
            'labels': self.loss_labels_ce if self.semantic_ce_loss else self.loss_labels,
            'masks': self.loss_masks,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        # 传递任务类型'task'给loss_labels函数
        if loss == 'labels':
            return loss_map[loss](outputs, targets, indices, num_masks, task=task)
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, mask_dict=None):
        """Compute dual-task losses for detection and segmentation / curve points."""
        # 1. 分离 ground truth targets
        targets_det, targets_seg = self.split_targets_by_task(targets)
        
        # 2. 分离模型预测 (outputs)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs" and k != 'interm_outputs'}
        
        # 使用 num_detection_queries 来切分所有预测
        k = self.num_detection_queries
        pred_points_all = outputs_without_aux.get("pred_points")
        if pred_points_all is None:
            pred_points_all = outputs_without_aux["pred_boxes"][..., :2]
        
        # 检测任务的预测
        # 确保类别维度正确
        outputs_det = {
            'pred_logits': outputs_without_aux['pred_logits'][:, :k, :self.num_classes],
            'pred_masks': outputs_without_aux['pred_masks'][:, :k, :, :],
            'pred_boxes': outputs_without_aux['pred_boxes'][:, :k, :],
            'pred_points': pred_points_all[:, :k, :],
        }

        # 分割 / 水线任务的预测
        outputs_seg = {
            'pred_logits': outputs_without_aux['pred_logits'][:, k:, :self.num_classes_seg],
            'pred_masks': outputs_without_aux['pred_masks'][:, k:, :, :],
            'pred_boxes': outputs_without_aux['pred_boxes'][:, k:, :],
            'pred_points': pred_points_all[:, k:, :],
        }
        
        losses = {}
        det_match_cost = ["cls", "point"] if self.keypoint_mode else ["cls", "box"]
        device = next(iter(outputs.values())).device
        zero = torch.zeros((), device=device)

        # ════════════════════════════════════════════════════════
        # 3. 计算检测 / 字符关键点任务损失
        # ════════════════════════════════════════════════════════
        num_masks_det = 1.0
        indices_det = None
        if self.train_character:
            indices_det = self.matcher_det(outputs_det, targets_det, cost=det_match_cost)
            num_masks_det = sum(len(t["labels"]) for t in targets_det)
            num_masks_det = torch.as_tensor([num_masks_det], dtype=torch.float, device=device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_masks_det)
            num_masks_det = torch.clamp(num_masks_det / get_world_size(), min=1).item()

            for loss in ['labels', 'boxes']:
                if loss in self.losses:
                    l_dict = self.get_loss(loss, outputs_det, targets_det, indices_det, num_masks_det, task='det')
                    losses.update({key + '_det': v for key, v in l_dict.items()})

        # ════════════════════════════════════════════════════════
        # 4. 水线：keypoint_mode 下动态长度曲线点；否则 mask 分割
        # ════════════════════════════════════════════════════════
        nq = outputs_without_aux["pred_logits"].shape[1]
        has_seg = (self.num_classes_seg > 0) and (k < nq) and (outputs_seg["pred_logits"].shape[1] > 0)
        num_masks_seg = 1.0
        if has_seg and self.keypoint_mode and self.train_waterline:
            curve_losses = self.loss_waterline_curve(outputs_without_aux, targets)
            losses["loss_ce_seg"] = curve_losses["loss_ce"]
            losses["loss_curve_seg"] = curve_losses["loss_curve"]
            losses["loss_curve_smooth_seg"] = curve_losses["loss_curve_smooth"]
        elif has_seg and not self.keypoint_mode and self.train_waterline:
            indices_seg = self.matcher_seg(outputs_seg, targets_seg, cost=["cls", "mask"])
            num_masks_seg = sum(len(t["labels"]) for t in targets_seg)
            num_masks_seg = torch.as_tensor([num_masks_seg], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_masks_seg)
            num_masks_seg = torch.clamp(num_masks_seg / get_world_size(), min=1).item()
            for loss in ["labels", "masks"]:
                if loss in self.losses:
                    l_dict = self.get_loss(loss, outputs_seg, targets_seg, indices_seg, num_masks_seg, task="seg")
                    losses.update({key + "_seg": v for key, v in l_dict.items()})

        # ════════════════════════════════════════════════════════
        # 5. 计算辅助层损失 (Auxiliary outputs)
        # ════════════════════════════════════════════════════════
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_points = aux_outputs.get("pred_points")
                if aux_points is None:
                    aux_points = aux_outputs["pred_boxes"][..., :2]
                aux_outputs_det = {
                    'pred_logits': aux_outputs['pred_logits'][:, :k, :self.num_classes],
                    'pred_masks': aux_outputs['pred_masks'][:, :k, :, :],
                    'pred_boxes': aux_outputs['pred_boxes'][:, :k, :],
                    'pred_points': aux_points[:, :k, :],
                }
                aux_outputs_seg = {
                    'pred_logits': aux_outputs['pred_logits'][:, k:, :self.num_classes_seg],
                    'pred_masks': aux_outputs['pred_masks'][:, k:, :, :],
                    'pred_boxes': aux_outputs['pred_boxes'][:, k:, :],
                    'pred_points': aux_points[:, k:, :],
                }

                if self.train_character:
                    indices_det_aux = self.matcher_det(aux_outputs_det, targets_det, cost=det_match_cost)
                    for loss in ['labels', 'boxes']:
                        if loss in self.losses:
                            l_dict = self.get_loss(loss, aux_outputs_det, targets_det, indices_det_aux, num_masks_det, task='det')
                            losses.update({key + f'_det_{i}': v for key, v in l_dict.items()})

                if has_seg and self.keypoint_mode and self.train_waterline:
                    curve_losses = self.loss_waterline_curve(aux_outputs, targets)
                    losses[f"loss_ce_seg_{i}"] = curve_losses["loss_ce"]
                    losses[f"loss_curve_seg_{i}"] = curve_losses["loss_curve"]
                    losses[f"loss_curve_smooth_seg_{i}"] = curve_losses["loss_curve_smooth"]
                elif has_seg and not self.keypoint_mode and self.train_waterline:
                    indices_seg_aux = self.matcher_seg(aux_outputs_seg, targets_seg, cost=["cls", "mask"])
                    for loss in ["labels", "masks"]:
                        if loss in self.losses:
                            l_dict = self.get_loss(loss, aux_outputs_seg, targets_seg, indices_seg_aux, num_masks_seg, task="seg")
                            losses.update({key + f"_seg_{i}": v for key, v in l_dict.items()})
                
        # ════════════════════════════════════════════════════════
        # 6.计算 Contrastive Denoising Training 损失
        # ════════════════════════════════════════════════════════
        if 'dn_aux_outputs' in outputs and 'dn_meta' in outputs:
            dn_meta = outputs['dn_meta']
            # 获取 CDN 匹配索引（不需要匈牙利匹配，直接使用 GT 对应关系）
            dn_indices = self._get_cdn_matched_indices(dn_meta, targets_det)
            
            # 计算 denoising 的 num_boxes（乘以 group 数量）
            dn_num_boxes = num_masks_det * dn_meta['dn_num_group']
            
            # 对每个 denoising 辅助层计算损失
            for i, dn_aux_outputs in enumerate(outputs['dn_aux_outputs']):
                # Denoising outputs 已经是检测任务的输出
                # forward_prediction_heads 使用 class_embed_det，输出维度已经是 num_classes
                # 不需要额外的类别维度切片
                dn_outputs_det = {
                    'pred_logits': dn_aux_outputs['pred_logits'],
                    'pred_boxes': dn_aux_outputs['pred_boxes'],
                }
                
                # 计算 denoising 损失（只计算 labels 和 boxes）
                for loss in ['labels', 'boxes']:
                    if loss in self.losses:
                        l_dict = self.get_loss(loss, dn_outputs_det, targets_det, dn_indices, dn_num_boxes, task='det')
                        losses.update({k + f'_dn_{i}': v for k, v in l_dict.items()})

        # 直接返回原始损失，平衡逻辑移到 DraftFormer.forward
        return losses
    
    @staticmethod
    def _get_cdn_matched_indices(dn_meta, targets):
        """
        获取 Contrastive Denoising Training 的匹配索引
        
        CDN 不需要匈牙利匹配，因为 denoising queries 是直接从 GT 生成的，
        所以匹配关系是已知的。
        
        Args:
            dn_meta: dict containing:
                - dn_positive_idx: tuple of tensors, positive query indices per image
                - dn_num_group: number of denoising groups
            targets: list of target dicts
            
        Returns:
            indices: list of (src_idx, tgt_idx) tuples for each image
        """
        dn_positive_idx = dn_meta["dn_positive_idx"]
        dn_num_group = dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        
        # 从 dn_positive_idx 获取 device，确保所有 tensor 在同一设备
        # dn_positive_idx 是在 decoder 中生成的，一定在正确的设备上
        device = dn_positive_idx[0].device if len(dn_positive_idx) > 0 and len(dn_positive_idx[0]) > 0 else \
                 (targets[0]['labels'].device if len(targets) > 0 and len(targets[0]['labels']) > 0 else 'cuda:0')
        
        indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                # 每个 GT 在每个 group 中都有一个对应的 positive query
                # dn_positive_idx[i] 包含了第 i 张图片的所有 positive query 索引
                src_idx = dn_positive_idx[i]
                # 目标索引：每个 group 重复一次 GT 索引
                # 确保 tgt_idx 在与 src_idx 相同的设备上
                tgt_idx = torch.arange(num_gt, device=src_idx.device, dtype=torch.long).repeat(dn_num_group)
                indices.append((src_idx, tgt_idx))
            else:
                # 空 tensor 也需要在正确的设备上
                indices.append((torch.tensor([], dtype=torch.long, device=device),
                               torch.tensor([], dtype=torch.long, device=device)))
        
        return indices

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher_det: {}".format(self.matcher_det.__repr__(_repr_indent=8) if self.matcher_det else "None"),
            "matcher_seg: {}".format(self.matcher_seg.__repr__(_repr_indent=8) if self.matcher_seg else "None"),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "num_classes_seg: {}".format(self.num_classes_seg),
            "num_detection_queries: {}".format(self.num_detection_queries),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


DraftFormerLoss = SetCriterion


def build_task_balancer(config, task_names):
    if not config or not config.get("enabled", False):
        return None
    method = config.get("method", "uncertainty")
    kwargs = {}
    if method == "uncertainty":
        kwargs["initial_log_var"] = config.get("initial_log_var", 0.0)
    elif method == "gradnorm":
        kwargs["alpha"] = config.get("alpha", 1.5)
        kwargs["update_frequency"] = config.get("update_frequency", 10)
        kwargs["initial_weight"] = config.get("initial_weight", 1.0)
    elif method == "dwa":
        kwargs["temperature"] = config.get("temperature", 2.0)
        kwargs["window_size"] = config.get("window_size", 20)
    return DynamicTaskBalancer(task_names=task_names, method=method, **kwargs)


def build_loss(config, architecture_config):
    config = copy.deepcopy(config)
    head_cfg = architecture_config["Head"]
    decoder_cfg = architecture_config["Decoder"]
    keypoint_mode = bool(
        architecture_config.get(
            "keypoint_mode",
            head_cfg.get("keypoint_mode", architecture_config.get("point_mode", False)),
        )
    )

    matcher_det_cfg = copy.deepcopy(config.get("matcher_det", {}))
    matcher_seg_cfg = copy.deepcopy(config.get("matcher_seg", {}))
    num_points = config.get("num_points", decoder_cfg.get("train_num_points", 12544))

    matcher_det = HungarianMatcher(
        cost_class=matcher_det_cfg.get("cost_class", 4.0),
        cost_mask=matcher_det_cfg.get("cost_mask", 0.0),
        cost_dice=matcher_det_cfg.get("cost_dice", 0.0),
        cost_box=matcher_det_cfg.get("cost_box", matcher_det_cfg.get("cost_point", 5.0)),
        cost_giou=matcher_det_cfg.get("cost_giou", 0.0 if keypoint_mode else 2.0),
        num_points=matcher_det_cfg.get("num_points", num_points),
    )
    matcher_seg = HungarianMatcher(
        cost_class=matcher_seg_cfg.get("cost_class", 4.0),
        cost_mask=matcher_seg_cfg.get("cost_mask", 5.0),
        cost_dice=matcher_seg_cfg.get("cost_dice", 5.0),
        cost_box=matcher_seg_cfg.get("cost_box", 0.0),
        cost_giou=matcher_seg_cfg.get("cost_giou", 0.0),
        num_points=matcher_seg_cfg.get("num_points", num_points),
    )

    weight_dict = {
        "loss_ce_det": config.get("class_weight", 4.0),
        "loss_bbox_det": config.get("box_weight", config.get("point_weight", 5.0)),
        "loss_giou_det": config.get("giou_weight", 0.0 if keypoint_mode else 2.0),
        "loss_ce_seg": config.get("curve_exist_weight", config.get("class_weight", 4.0)),
    }
    if keypoint_mode:
        weight_dict.update(
            {
                "loss_curve_seg": config.get("curve_weight", 5.0),
                "loss_curve_smooth_seg": config.get("curve_smooth_weight", 0.5),
            }
        )
    else:
        weight_dict.update(
            {
                "loss_mask_seg": config.get("mask_weight", 5.0),
                "loss_dice_seg": config.get("dice_weight", 5.0),
            }
        )
    if head_cfg.get("deep_supervision", True):
        base_weight_dict = dict(weight_dict)
        for i in range(decoder_cfg.get("dec_layers", 6)):
            for key, value in base_weight_dict.items():
                weight_dict[f"{key}_{i}"] = value

    if keypoint_mode:
        losses = config.get("losses", ["labels", "boxes"])
    else:
        losses = config.get("losses", ["labels", "masks", "boxes"])
    task_train = architecture_config.get("TaskTrain") or {}
    task_names = []
    if task_train.get("train_character", True):
        task_names.append("det")
    if task_train.get("train_waterline", True):
        task_names.append("seg")
    depth_cfg = architecture_config.get("DirectDepth") or {}
    if depth_cfg.get("enabled", True) and task_train.get("train_depth", True):
        task_names.append("draft")
    task_balancer = build_task_balancer(config.get("task_balancer", {}), task_names)
    # Allow TaskTrain at root cfg via architecture_config passthrough from build_model.

    return DraftFormerLoss(
        num_classes=head_cfg["num_classes"],
        matcher_det=matcher_det,
        matcher_seg=matcher_seg,
        weight_dict=weight_dict,
        eos_coef=config.get("no_object_weight", 0.1),
        losses=losses,
        num_points=num_points,
        oversample_ratio=config.get("oversample_ratio", 3.0),
        importance_sample_ratio=config.get("importance_sample_ratio", 0.75),
        num_detection_queries=decoder_cfg["num_detection_queries"],
        num_classes_seg=head_cfg.get("num_classes_seg", 1),
        semantic_ce_loss=head_cfg.get("semantic_ce_loss", False),
        task_balancer=task_balancer,
        keypoint_mode=keypoint_mode,
        curve_smooth_weight=config.get("curve_smooth_weight", 0.5),
        train_character=task_train.get("train_character", True),
        train_waterline=task_train.get("train_waterline", True),
    )

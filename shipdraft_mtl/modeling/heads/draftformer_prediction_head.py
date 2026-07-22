from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from shipdraft_mtl.utils.utils import MLP


class DraftFormerPredictionHead(nn.Module):
    """Multi-task head for character keypoints and waterline curve points.

    When ``keypoint_mode=True`` (default for e2e draft reading):
      - Each query predicts a pure 2D coordinate ``(x, y)`` in [0, 1]
      - Character queries ``[0:num_detection_queries)`` predict multi-class logits
      - Waterline queries ``[num_detection_queries:)`` predict binary existence logits
      - No box (w,h) regression and no mask head

    Legacy box/mask mode is kept for older multitask configs.
    """

    def __init__(
        self,
        hidden_dim,
        mask_dim,
        num_classes,
        num_classes_seg,
        num_detection_queries,
        num_decoder_layers,
        semantic_ce_loss=False,
        deep_supervision=True,
        keypoint_mode=False,
        predict_masks=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask_dim = mask_dim
        self.num_classes = num_classes
        self.num_classes_seg = num_classes_seg
        self.num_detection_queries = num_detection_queries
        self.num_decoder_layers = num_decoder_layers
        self.semantic_ce_loss = semantic_ce_loss
        self.deep_supervision = deep_supervision
        self.keypoint_mode = bool(keypoint_mode)
        if predict_masks is None:
            predict_masks = not self.keypoint_mode
        self.predict_masks = bool(predict_masks) and num_classes_seg > 0
        self.point_dim = 2 if self.keypoint_mode else 4

        if semantic_ce_loss:
            self.class_embed_det = nn.Linear(hidden_dim, num_classes + 1)
            self.class_embed_seg = (
                nn.Linear(hidden_dim, max(num_classes_seg, 0) + 1) if num_classes_seg > 0 else None
            )
        else:
            self.class_embed_det = nn.Linear(hidden_dim, num_classes)
            self.class_embed_seg = nn.Linear(hidden_dim, num_classes_seg) if num_classes_seg > 0 else None

        self.mask_embed_seg = (
            MLP(hidden_dim, hidden_dim, mask_dim, 3) if self.predict_masks else None
        )

        point_heads = [MLP(hidden_dim, hidden_dim, self.point_dim, 3) for _ in range(num_decoder_layers)]
        for module in point_heads:
            nn.init.constant_(module.layers[-1].weight.data, 0)
            nn.init.constant_(module.layers[-1].bias.data, 0)
        # Keep historical attribute name used by the decoder refine path.
        self.bbox_embed_det = nn.ModuleList(point_heads)

    def get_proposal_class_head(self):
        return self.class_embed_det

    def get_proposal_box_head(self):
        return self.bbox_embed_det[0]

    def get_bbox_refine_heads(self):
        return self.bbox_embed_det

    def forward(self, decoder_outputs, mask_features, query_decoder):
        predictions_class = []
        predictions_mask = []

        initial_state = decoder_outputs.get("initial_state")
        if initial_state is not None:
            outputs_class, outputs_mask = self.forward_prediction_heads(
                initial_state,
                mask_features,
                query_decoder,
                pred_mask=self.predict_masks,
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        decoder_states = decoder_outputs["decoder_states"]
        for i, output in enumerate(decoder_states):
            outputs_class, outputs_mask = self.forward_prediction_heads(
                output,
                mask_features,
                query_decoder,
                pred_mask=self.predict_masks and (self.training or (i == len(decoder_states) - 1)),
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        reference_tensor = torch.stack(decoder_outputs["references"], dim=0)
        out_points = reference_tensor if decoder_outputs["initial_pred"] else reference_tensor[1:]

        # Decoder references may be 2D (keypoints) or 4D (legacy boxes).
        final_refs = out_points[-1]
        pred_points = final_refs[..., :2]
        if final_refs.shape[-1] >= 4:
            pred_boxes = final_refs[..., :4]
        else:
            # Compatibility: materialize tiny pseudo-boxes around pure points.
            wh = final_refs.new_full(final_refs.shape[:-1] + (2,), 0.05)
            pred_boxes = torch.cat([pred_points, wh], dim=-1)

        outputs = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "pred_boxes": pred_boxes,
            "pred_points": pred_points,
            "pred_references": final_refs,
            "decoder_hidden_states": decoder_states[-1].transpose(0, 1),
        }
        if self.deep_supervision:
            outputs["aux_outputs"] = self._set_aux_loss(predictions_class, predictions_mask, out_points)

        interm_state = decoder_outputs.get("interm_state")
        interm_reference = decoder_outputs.get("interm_reference")
        if interm_state is not None and interm_reference is not None:
            decoder_output = query_decoder.normalize_queries(interm_state).transpose(0, 1)
            interm_outputs_class = self.class_embed_det(decoder_output)
            bs, nq, _ = decoder_output.shape
            h, w = mask_features.shape[-2:]
            interm_outputs_mask = torch.zeros(
                bs,
                nq,
                h,
                w,
                device=mask_features.device,
                dtype=mask_features.dtype,
            )
            interm_pts = interm_reference[..., :2]
            if interm_reference.shape[-1] >= 4:
                interm_boxes = interm_reference[..., :4]
            else:
                wh = interm_reference.new_full(interm_reference.shape[:-1] + (2,), 0.05)
                interm_boxes = torch.cat([interm_pts, wh], dim=-1)
            outputs["interm_outputs"] = {
                "pred_logits": interm_outputs_class,
                "pred_boxes": interm_boxes,
                "pred_points": interm_pts,
                "pred_masks": interm_outputs_mask,
            }

        return outputs

    def forward_prediction_heads(self, query_state, mask_features, query_decoder, pred_mask=True):
        decoder_output = query_decoder.normalize_queries(query_state).transpose(0, 1)
        total_queries = decoder_output.shape[1]
        det_total = min(self.num_detection_queries, total_queries)

        decoder_output_det = decoder_output[:, :det_total, :]
        decoder_output_seg = decoder_output[:, det_total:, :]

        outputs_class_det = self.class_embed_det(decoder_output_det)
        if decoder_output_seg.shape[1] > 0 and self.class_embed_seg is not None:
            outputs_class_seg = self.class_embed_seg(decoder_output_seg)
            num_cls_det = outputs_class_det.shape[-1]
            num_cls_seg = outputs_class_seg.shape[-1]
            if num_cls_det > num_cls_seg:
                outputs_class_seg = F.pad(outputs_class_seg, (0, num_cls_det - num_cls_seg), value=-10.0)
            elif num_cls_seg > num_cls_det:
                outputs_class_det = F.pad(outputs_class_det, (0, num_cls_seg - num_cls_det), value=-10.0)
            outputs_class = torch.cat([outputs_class_det, outputs_class_seg], dim=1)
        else:
            outputs_class = outputs_class_det

        outputs_mask = None
        if pred_mask and self.mask_embed_seg is not None:
            bs = decoder_output_det.shape[0]
            h, w = mask_features.shape[-2:]
            det_masks = torch.zeros(
                bs,
                det_total,
                h,
                w,
                device=mask_features.device,
                dtype=mask_features.dtype,
            )
            if decoder_output_seg.shape[1] > 0:
                mask_embed_seg = self.mask_embed_seg(decoder_output_seg)
                seg_masks = torch.einsum("bqc,bchw->bqhw", mask_embed_seg, mask_features)
                outputs_mask = torch.cat([det_masks, seg_masks], dim=1)
            else:
                outputs_mask = det_masks
        elif pred_mask:
            bs = decoder_output.shape[0]
            h, w = mask_features.shape[-2:]
            outputs_mask = torch.zeros(
                bs,
                total_queries,
                h,
                w,
                device=mask_features.device,
                dtype=mask_features.dtype,
            )
        else:
            # Keypoint mode: no mask tensor; allocate an empty-compatible placeholder.
            bs = decoder_output.shape[0]
            h, w = mask_features.shape[-2:]
            outputs_mask = torch.zeros(
                bs,
                total_queries,
                h,
                w,
                device=mask_features.device,
                dtype=mask_features.dtype,
            )

        return outputs_class, outputs_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, out_refs=None):
        if out_refs is None:
            return [{"pred_logits": a, "pred_masks": b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        aux = []
        for a, b, refs in zip(outputs_class[:-1], outputs_seg_masks[:-1], out_refs[:-1]):
            pts = refs[..., :2]
            if refs.shape[-1] >= 4:
                boxes = refs[..., :4]
            else:
                wh = refs.new_full(refs.shape[:-1] + (2,), 0.05)
                boxes = torch.cat([pts, wh], dim=-1)
            aux.append(
                {
                    "pred_logits": a,
                    "pred_masks": b,
                    "pred_boxes": boxes,
                    "pred_points": pts,
                }
            )
        return aux

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from shipdraft_mtl.utils.utils import MLP


class DraftFormerPredictionHead(nn.Module):
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

        if semantic_ce_loss:
            self.class_embed_det = nn.Linear(hidden_dim, num_classes + 1)
            self.class_embed_seg = nn.Linear(hidden_dim, num_classes_seg + 1)
        else:
            self.class_embed_det = nn.Linear(hidden_dim, num_classes)
            self.class_embed_seg = nn.Linear(hidden_dim, num_classes_seg)
        self.mask_embed_seg = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        bbox_heads = [MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_decoder_layers)]
        for module in bbox_heads:
            nn.init.constant_(module.layers[-1].weight.data, 0)
            nn.init.constant_(module.layers[-1].bias.data, 0)
        self.bbox_embed_det = nn.ModuleList(bbox_heads)

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
                pred_mask=True,
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        decoder_states = decoder_outputs["decoder_states"]
        for i, output in enumerate(decoder_states):
            outputs_class, outputs_mask = self.forward_prediction_heads(
                output,
                mask_features,
                query_decoder,
                pred_mask=self.training or (i == len(decoder_states) - 1),
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        reference_tensor = torch.stack(decoder_outputs["references"], dim=0)
        out_boxes = reference_tensor if decoder_outputs["initial_pred"] else reference_tensor[1:]

        outputs = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "pred_boxes": out_boxes[-1],
            "decoder_hidden_states": decoder_states[-1].transpose(0, 1),
        }
        if self.deep_supervision:
            outputs["aux_outputs"] = self._set_aux_loss(predictions_class, predictions_mask, out_boxes)

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
            outputs["interm_outputs"] = {
                "pred_logits": interm_outputs_class,
                "pred_boxes": interm_reference,
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
        if decoder_output_seg.shape[1] > 0:
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
        if pred_mask:
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

        return outputs_class, outputs_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, out_boxes=None):
        if out_boxes is None:
            return [{"pred_logits": a, "pred_masks": b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        return [
            {"pred_logits": a, "pred_masks": b, "pred_boxes": c}
            for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1])
        ]

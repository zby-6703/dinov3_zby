from __future__ import annotations

"""Decoder component for DraftFormer."""

from typing import Optional

import fvcore.nn.weight_init as weight_init
import torch
from torch import Tensor, nn
from torch.amp import autocast

from shipdraft_mtl.modeling.common import Conv2d
from shipdraft_mtl.utils.utils import (
    MLP,
    _get_activation_fn,
    _get_clones,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
    inverse_sigmoid,
)

from ..encoders.ms_deform_attn import MSDeformAttn

__all__ = ["DraftFormerDecoder"]


class DraftFormerDecoder(nn.Module):
    """Two-stage deformable query decoder for DraftFormer."""

    def __init__(
        self,
        in_channels,
        hidden_dim,
        num_queries,
        num_detection_queries,
        nheads,
        dim_feedforward,
        dec_layers,
        enforce_input_project=False,
        two_stage=True,
        initialize_box_type="no",
        initial_pred=True,
        learn_tgt=False,
        total_num_feature_levels=4,
        dropout=0.0,
        activation="relu",
        dec_n_points=4,
        return_intermediate_dec=True,
        query_dim=4,
        dec_layer_share=False,
        use_task_embedding=False,
        task_embedding_scale=1.0,
    ):
        super().__init__()
        self.num_feature_levels = total_num_feature_levels
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_detection_queries = num_detection_queries
        self.num_layers = dec_layers
        self.initial_pred = initial_pred
        self.two_stage = two_stage
        self.learn_tgt = learn_tgt
        self.initialize_box_type = initialize_box_type
        self.task_embedding_scale = float(task_embedding_scale)
        self.task_embedding = nn.Embedding(2, hidden_dim) if use_task_embedding else None

        if not two_stage or self.learn_tgt:
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
        if not two_stage and initialize_box_type == "no":
            self.query_embed = nn.Embedding(num_queries, 4)
        if two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.output_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim,
            dim_feedforward,
            dropout,
            activation,
            self.num_feature_levels,
            nheads,
            dec_n_points,
        )
        self.decoder = TransformerDecoder(
            decoder_layer,
            self.num_layers,
            self.output_norm,
            return_intermediate=return_intermediate_dec,
            d_model=hidden_dim,
            query_dim=query_dim,
            num_feature_levels=self.num_feature_levels,
            dec_layer_share=dec_layer_share,
            num_detection_queries=self.num_detection_queries,
        )

    def normalize_queries(self, query_state):
        return self.output_norm(query_state)

    def add_task_embedding(self, tgt):
        if self.task_embedding is None:
            return tgt
        total_queries = tgt.shape[1]
        detection_end = min(int(self.num_detection_queries), total_queries)
        task_ids = torch.zeros(total_queries, dtype=torch.long, device=tgt.device)
        if detection_end < total_queries:
            task_ids[detection_end:] = 1
        task_embed = self.task_embedding(task_ids).to(dtype=tgt.dtype)
        return tgt + self.task_embedding_scale * task_embed.unsqueeze(0)

    def get_valid_ratio(self, mask):
        _, h, w = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_h.float() / h
        valid_ratio_w = valid_w.float() / w
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

    def forward(
        self,
        multi_scale_features,
        masks=None,
        proposal_class_head=None,
        proposal_box_head=None,
        bbox_refine_heads=None,
    ):
        assert len(multi_scale_features) == self.num_feature_levels
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []

        enable_mask = 0
        if masks is not None:
            for src in multi_scale_features:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [
                torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool)
                for src in multi_scale_features
            ]

        for i in range(self.num_feature_levels):
            idx = self.num_feature_levels - 1 - i
            src = multi_scale_features[idx]
            spatial_shapes.append(src.shape[-2:])
            src_flatten.append(self.input_proj[idx](src).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        bs = multi_scale_features[0].shape[0]
        interm_state = None
        interm_reference = None

        if self.two_stage:
            if proposal_class_head is None or proposal_box_head is None:
                raise ValueError("Two-stage decoder requires proposal_class_head and proposal_box_head")
            output_memory, output_proposals = gen_encoder_output_proposals(src_flatten, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            enc_outputs_class_unselected = proposal_class_head(output_memory)
            enc_outputs_coord_unselected = proposal_box_head(output_memory) + output_proposals

            topk = self.num_queries
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
            )
            refpoint_embed = refpoint_embed_undetach.detach()
            tgt_undetach = torch.gather(
                output_memory,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim),
            )
            interm_state = tgt_undetach.transpose(0, 1)
            interm_reference = refpoint_embed_undetach.sigmoid()

            tgt = tgt_undetach.detach()
            if self.learn_tgt:
                tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            if self.initialize_box_type != "no":
                refpoint_embed = refpoint_embed.detach()
        else:
            tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            refpoint_embed = self.query_embed.weight[None].repeat(bs, 1, 1)

        tgt = self.add_task_embedding(tgt)
        initial_state = tgt.transpose(0, 1) if self.initial_pred else None
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=None,
            bbox_embed=bbox_refine_heads,
            num_detection_queries=self.num_detection_queries,
        )

        return {
            "initial_pred": self.initial_pred,
            "initial_state": initial_state,
            "decoder_states": [state.transpose(0, 1) for state in hs],
            "references": references,
            "interm_state": interm_state,
            "interm_reference": interm_reference,
        }


class TransformerDecoder(nn.Module):
    """Multi-layer deformable decoder used by `DraftFormerDecoder`."""

    def __init__(self, decoder_layer, num_layers, norm=None,
                 return_intermediate=False,
                 d_model=256, query_dim=4,
                 modulate_hw_attn=True,
                 num_feature_levels=1,
                 deformable_decoder=True,
                 decoder_query_perturber=None,
                 dec_layer_number=None,  # number of queries each layer in decoder
                 rm_dec_query_scale=True,
                 dec_layer_share=False,
                 dec_layer_dropout_prob=None,
                 num_detection_queries: Optional[int] = None,
                 ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder
        self.num_detection_queries = num_detection_queries

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                # for memory
                level_start_index: Optional[Tensor] = None,  # num_levels
                spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None,
                bbox_embed=None,
                num_detection_queries: Optional[int] = None,
                ):
        """Run the deformable decoder and return all layer states and references."""
        output = tgt
        device = tgt.device
        bbox_embed = self.bbox_embed if bbox_embed is None else bbox_embed
        num_detection_queries = self.num_detection_queries if num_detection_queries is None else num_detection_queries

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid().to(device)
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(reference_points)

            reference_points_input = (
                reference_points[:, :, None]
                * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            )
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])

            raw_query_pos = self.ref_point_head(query_sine_embed)
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,

                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
            )

            if bbox_embed is not None:
                total_queries = output.shape[0]
                reference_before_sigmoid = inverse_sigmoid(reference_points)

                if num_detection_queries is None:
                    detection_end = total_queries
                else:
                    detection_end = min(num_detection_queries, total_queries)

                if detection_end > 0:
                    delta_unsig_det = bbox_embed[layer_id](output[:detection_end]).to(device)
                    outputs_unsig_det = delta_unsig_det + reference_before_sigmoid[:detection_end]
                    new_reference_points_det = outputs_unsig_det.sigmoid()
                    if detection_end < total_queries:
                        new_reference_points = torch.cat(
                            [new_reference_points_det, reference_points[detection_end:]], dim=0
                        )
                    else:
                        new_reference_points = new_reference_points_det
                else:
                    new_reference_points = reference_points

                reference_points = new_reference_points.detach()
                ref_points.append(new_reference_points)
            else:
                ref_points.append(reference_points)

            intermediate.append(self.norm(output))

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]
        ]


class DeformableTransformerDecoderLayer(nn.Module):
    """Single decoder layer with self-attention, deformable cross-attention, and FFN."""

    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 use_deformable_box_attn=False,
                 key_aware_type=None,
                 ):
        super().__init__()

        if use_deformable_box_attn:
            raise NotImplementedError
        else:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None

    def rm_self_attn_modules(self):
        """Remove self-attention modules for ablation experiments."""
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    @autocast(device_type='cuda', enabled=False)
    def forward(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None,  # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None,  # num_levels
                memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None,  # pos for memory

                # attention masks
                self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
                ):
        """Apply one decoder block."""
        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        if self.key_aware_type is not None:
            if self.key_aware_type == "mean":
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == "proj_mean":
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))

        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            tgt_reference_points.transpose(0, 1).contiguous(),
            memory.transpose(0, 1),
            memory_spatial_shapes,
            memory_level_start_index,
            memory_key_padding_mask,
        ).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt = self.forward_ffn(tgt)
        return tgt

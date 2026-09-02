"""Dual-branch maturity head for RT-DETR.

This module keeps the standard maturity detection branch and adds an
independent ordinal auxiliary branch:
    - pred_logits: [B, Q, num_classes]
    - pred_ord_logits: [B, Q, ordinal_num_classes - 1]
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from src.core import register
from .denoising import get_contrastive_denoising_training_group
from .maturity_head import Integral, MLP, TransformerDecoderLayer
from .utils import bias_init_with_prob, inverse_sigmoid


__all__ = ["DualBranchMaturityHead"]


class DualBranchTransformerDecoder(nn.Module):
    """Transformer decoder with extra ordinal branch outputs."""

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        tgt,
        ref_points_unact,
        memory,
        memory_spatial_shapes,
        memory_level_start_index,
        bbox_head,
        score_head,
        ord_head,
        query_pos_head,
        attn_mask=None,
        memory_mask=None,
        use_ordinal_branch=True,
    ):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_ord_logits = []
        dec_out_dists = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(
                output,
                ref_points_input,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask,
                memory_mask,
                query_pos_embed,
            )

            bbox_output = bbox_head[i](output)
            cls_output = score_head[i](output)
            ord_output = ord_head[i](output) if use_ordinal_branch else None

            if bbox_output.shape[-1] == 4:
                inter_ref_bbox = F.sigmoid(bbox_output + inverse_sigmoid(ref_points_detach))
                dist_output = None
            else:
                bbox_delta = bbox_output[..., :4]
                dist_output = bbox_output[..., 4:]
                inter_ref_bbox = F.sigmoid(bbox_delta + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(cls_output)
                if ord_output is not None:
                    dec_out_ord_logits.append(ord_output)
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    bbox_pred = bbox_head[i](output)
                    if bbox_pred.shape[-1] == 4:
                        dec_out_bboxes.append(F.sigmoid(bbox_pred + inverse_sigmoid(ref_points)))
                    else:
                        bbox_delta_i = bbox_pred[..., :4]
                        dec_out_bboxes.append(F.sigmoid(bbox_delta_i + inverse_sigmoid(ref_points)))
                if dist_output is not None:
                    dec_out_dists.append(dist_output)
            elif i == self.eval_idx:
                dec_out_logits.append(cls_output)
                if ord_output is not None:
                    dec_out_ord_logits.append(ord_output)
                dec_out_bboxes.append(inter_ref_bbox)
                if dist_output is not None:
                    dec_out_dists.append(dist_output)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

        ord_stack = torch.stack(dec_out_ord_logits) if dec_out_ord_logits else None
        if len(dec_out_dists) > 0:
            return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), torch.stack(dec_out_dists), ord_stack
        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), None, ord_stack


@register
class DualBranchMaturityHead(nn.Module):
    """RT-DETR detection head with an auxiliary ordinal branch."""

    __share__ = ["num_classes", "ordinal_num_classes"]

    def __init__(
        self,
        num_classes=4,
        hidden_dim=256,
        reg_max=16,
        num_queries=300,
        position_embed_type="sine",
        feat_channels=[256, 256, 256],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_decoder_points=4,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        use_ordinal=False,
        use_fdr=False,
        use_ordinal_branch=True,
        ordinal_branch_type="cumulative",
        ordinal_num_classes=4,
        num_ordinal_logits=None,
    ):
        super().__init__()
        assert position_embed_type in ["sine", "learned"]
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max
        self.use_ordinal = use_ordinal
        self.use_fdr = use_fdr
        self.use_ordinal_branch = bool(use_ordinal_branch)
        # Keep ordinal modules and checkpoint weights available, while allowing
        # deployment inference to skip their computation.
        self.inference_use_ordinal_branch = True
        self.ordinal_branch_type = ordinal_branch_type
        self.ordinal_num_classes = int(
            ordinal_num_classes if ordinal_num_classes is not None else num_classes
        )
        self.num_ordinal_logits = int(
            num_ordinal_logits
            if num_ordinal_logits is not None
            else (self.ordinal_num_classes - 1)
        )

        if self.ordinal_branch_type != "cumulative":
            raise ValueError(f"Only cumulative ordinal branch is supported, got {self.ordinal_branch_type}.")
        if self.ordinal_num_classes < 2:
            raise ValueError(
                f"Expected ordinal_num_classes >= 2, got {self.ordinal_num_classes}."
            )
        if self.ordinal_num_classes > num_classes:
            raise ValueError(
                f"Expected ordinal_num_classes <= num_classes, got "
                f"{self.ordinal_num_classes} > {num_classes}."
            )
        if self.num_ordinal_logits != self.ordinal_num_classes - 1:
            raise ValueError(
                f"Expected num_ordinal_logits={self.ordinal_num_classes - 1}, "
                f"got {self.num_ordinal_logits}."
            )
        if not self.use_ordinal_branch:
            raise ValueError("DualBranchMaturityHead requires use_ordinal_branch=True.")
        self._build_input_proj_layer(feat_channels)

        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points
        )
        self.decoder = DualBranchTransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)

        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_ord_head = self._build_ord_head(hidden_dim, self.num_ordinal_logits)

        if self.use_fdr:
            self.enc_bbox_head = self._build_fdr_head(hidden_dim, reg_max)
            self.integral = Integral(reg_max)
        else:
            self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            self.integral = None

        self.dec_score_head = nn.ModuleList()
        self.dec_ord_head = nn.ModuleList()
        self.dec_bbox_head = nn.ModuleList()
        for _ in range(num_decoder_layers):
            self.dec_score_head.append(nn.Linear(hidden_dim, num_classes))
            self.dec_ord_head.append(self._build_ord_head(hidden_dim, self.num_ordinal_logits))
            if self.use_fdr:
                self.dec_bbox_head.append(self._build_fdr_head(hidden_dim, reg_max))
            else:
                self.dec_bbox_head.append(MLP(hidden_dim, hidden_dim, 4, num_layers=3))

        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _build_ord_head(self, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _build_fdr_head(self, hidden_dim, reg_max):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 + 4 * (reg_max + 1)),
        )

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        init.constant_(self.enc_score_head.bias, bias)
        if self.use_fdr:
            init.constant_(self.enc_bbox_head[-1].weight, 0)
            init.constant_(self.enc_bbox_head[-1].bias, 0)
        else:
            init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
            init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_, reg_, ord_ in zip(self.dec_score_head, self.dec_bbox_head, self.dec_ord_head):
            init.constant_(cls_.bias, bias)
            init.constant_(ord_[-1].weight, 0)
            init.constant_(ord_[-1].bias, 0)
            if self.use_fdr:
                init.constant_(reg_[-1].weight, 0)
                init.constant_(reg_[-1].bias, 0)
            else:
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        init.constant_(self.enc_ord_head[-1].weight, 0)
        init.constant_(self.enc_ord_head[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    @torch.jit.unused
    def set_ordinal_inference(self, enabled=True):
        """Enable/disable ordinal-head computation for deployment inference."""
        self.inference_use_ordinal_branch = bool(enabled)
        return self

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("conv", nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                            ("norm", nn.BatchNorm2d(self.hidden_dim)),
                        ]
                    )
                )
            )
        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("conv", nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                            ("norm", nn.BatchNorm2d(self.hidden_dim)),
                        ]
                    )
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0]
        for feat in proj_feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
            level_start_index.append(h * w + level_start_index[-1])

        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return feat_flatten, spatial_shapes, level_start_index

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)] for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=h, dtype=dtype),
                torch.arange(end=w, dtype=dtype),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_wh = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_wh
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _get_decoder_input(
        self,
        memory,
        spatial_shapes,
        denoising_class=None,
        denoising_bbox_unact=None,
        use_ordinal_branch=True,
    ):
        bs, _, _ = memory.shape
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
            if anchors.device != memory.device:
                anchors = anchors.to(memory.device)
            if valid_mask.device != memory.device:
                valid_mask = valid_mask.to(memory.device)

        memory = valid_mask.to(memory.dtype) * memory
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_ord = self.enc_ord_head(output_memory) if use_ordinal_branch else None

        if self.use_fdr:
            enc_bbox_raw = self.enc_bbox_head(output_memory)
            enc_bbox_delta = enc_bbox_raw[..., :4]
            enc_outputs_coord_unact = enc_bbox_delta + anchors
        else:
            enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        if self.use_ordinal:
            enc_scores = torch.sigmoid(enc_outputs_class).max(-1).values
        else:
            enc_scores = enc_outputs_class.max(-1).values
        _, topk_ind = torch.topk(enc_scores, self.num_queries, dim=1)

        reference_points_unact = enc_outputs_coord_unact.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]),
        )
        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)

        enc_topk_logits = enc_outputs_class.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]),
        )
        enc_topk_ord_logits = None
        if enc_outputs_ord is not None:
            enc_topk_ord_logits = enc_outputs_ord.gather(
                dim=1,
                index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_ord.shape[-1]),
            )

        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(
                dim=1,
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]),
            )
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits, enc_topk_ord_logits

    def forward(self, feats, targets=None):
        memory, spatial_shapes, level_start_index = self._get_encoder_input(feats)

        use_ordinal_branch = bool(
            self.use_ordinal_branch
            and (self.training or self.inference_use_ordinal_branch)
        )

        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets,
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
            )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits, enc_topk_ord_logits = self._get_decoder_input(
            memory,
            spatial_shapes,
            denoising_class,
            denoising_bbox_unact,
            use_ordinal_branch=use_ordinal_branch,
        )

        out_bboxes, out_logits, out_dists, out_ord_logits = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.dec_ord_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            use_ordinal_branch=use_ordinal_branch,
        )

        out = {
            "pred_logits": out_logits[-1],
            "pred_boxes": out_bboxes[-1],
        }
        if out_ord_logits is not None:
            out["pred_ord_logits"] = out_ord_logits[-1]

        if self.use_fdr and out_dists is not None:
            last_dist = out_dists[-1]
            bsz, num_queries, _ = last_dist.shape
            out["pred_dist"] = last_dist.reshape(bsz, num_queries, 4, self.reg_max + 1)

        if self.training and self.aux_loss:
            if dn_meta is not None:
                dn_out_bboxes, out_bboxes_main = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
                dn_out_logits, out_logits_main = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)
                dn_out_ord_logits, out_ord_logits_main = torch.split(out_ord_logits, dn_meta["dn_num_split"], dim=2)

                out["pred_logits"] = out_logits_main[-1]
                out["pred_boxes"] = out_bboxes_main[-1]
                out["pred_ord_logits"] = out_ord_logits_main[-1]

                if out_dists is not None:
                    dn_out_dists, out_dists_main = torch.split(out_dists, dn_meta["dn_num_split"], dim=2)
                    last_dist_main = out_dists_main[-1]
                    bsz, num_queries, _ = last_dist_main.shape
                    out["pred_dist"] = last_dist_main.reshape(bsz, num_queries, 4, self.reg_max + 1)
                else:
                    dn_out_dists, out_dists_main = None, None

                out["aux_outputs"] = self._set_aux_loss(
                    out_logits_main[:-1],
                    out_bboxes_main[:-1],
                    out_ord_logits_main[:-1],
                    None if out_dists_main is None else out_dists_main[:-1],
                )
                out["dn_aux_outputs"] = self._set_aux_loss(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_ord_logits,
                    dn_out_dists,
                )
                out["dn_meta"] = dn_meta
            else:
                out["aux_outputs"] = self._set_aux_loss(
                    out_logits[:-1],
                    out_bboxes[:-1],
                    out_ord_logits[:-1],
                    None if out_dists is None else out_dists[:-1],
                )

            out["aux_outputs"].extend(
                self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes], [enc_topk_ord_logits])
            )

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_ord, outputs_dist=None):
        aux_outputs = []
        if outputs_dist is None:
            outputs_dist = [None] * len(outputs_class)

        for cls, box, ord_logits, dist in zip(outputs_class, outputs_coord, outputs_ord, outputs_dist):
            aux = {"pred_logits": cls, "pred_boxes": box, "pred_ord_logits": ord_logits}
            if dist is not None:
                bsz, num_queries, _ = dist.shape
                aux["pred_dist"] = dist.reshape(bsz, num_queries, 4, self.reg_max + 1)
            aux_outputs.append(aux)

        return aux_outputs

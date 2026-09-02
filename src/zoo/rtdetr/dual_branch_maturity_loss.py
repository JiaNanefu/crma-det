"""Dual-branch maturity loss for RT-DETR.

Standard detection losses (VFL classification, L1+GIoU bbox) plus ordinal
auxiliary supervision:
    - loss_ord_branch: BCEWithLogits on cumulative ordinal targets
      P(y > 0), ..., P(y > ordinal_num_classes - 2)
    - loss_ord_mono:   monotonicity regularization P(y>k) >= P(y>k+1)

This is the ONLY maturity loss used in the final model.
The original MaturityLoss base class has been merged in - no inheritance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import register
from src.misc.dist import get_world_size, is_dist_available_and_initialized
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou


__all__ = ["DualBranchMaturityLoss"]


@register
class DualBranchMaturityLoss(nn.Module):
    """Maturity detection criterion with the dual-branch ordinal head.

    Matcher produces (src_idx, tgt_idx) pairs; losses are computed only on
    matched predictions and averaged over `num_boxes` (world-size aware).

    Loss routing (configured via YAML `losses` list):
        "vfl"         -> Varifocal Loss on standard num_classes logits
        "boxes"       -> L1 + GIoU on bbox regressions
        "ord_branch"  -> cumulative BCE on ordinal logits
        "ord_mono"    -> monotonicity penalty (ReLU on probability reversal)
    """

    __share__ = ["num_classes", "ordinal_num_classes", "ordinal_ignore_labels"]
    __inject__ = ["matcher"]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.75,
        gamma=2.0,
        num_classes=4,
        ordinal_num_classes=4,
        ordinal_ignore_labels=None,
        use_ordinal_branch=True,
        apply_dn_ordinal=False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ordinal_num_classes = int(
            ordinal_num_classes if ordinal_num_classes is not None else num_classes
        )
        if self.ordinal_num_classes < 2:
            raise ValueError(
                f"Expected ordinal_num_classes >= 2, got {self.ordinal_num_classes}."
            )
        if self.ordinal_num_classes > self.num_classes:
            raise ValueError(
                f"Expected ordinal_num_classes <= num_classes, got "
                f"{self.ordinal_num_classes} > {self.num_classes}."
            )
        self.ordinal_ignore_labels = {
            int(label) for label in (ordinal_ignore_labels or [])
        }
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.alpha = alpha
        self.gamma = gamma
        self.use_ordinal_branch = bool(use_ordinal_branch)
        self.apply_dn_ordinal = bool(apply_dn_ordinal)

    # ------------------------------------------------------------------
    #  Core losses
    # ------------------------------------------------------------------

    def _loss_vfl(self, outputs, targets, indices, num_boxes):
        """Varifocal Loss on the standard class branch."""
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        device = outputs["pred_logits"].device

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs["pred_logits"]
        if src_logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"Expected pred_logits last dim to be {self.num_classes}, got {src_logits.shape[-1]}."
            )
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def _loss_boxes(self, outputs, targets, indices, num_boxes):
        """Bounding-box loss: L1 + GIoU."""
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_boxes
        loss_giou = (
            1
            - torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
                )
            )
        ).sum() / num_boxes

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    # ------------------------------------------------------------------
    #  Ordinal-branch losses
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_like_loss(outputs):
        if "pred_ord_logits" in outputs:
            return outputs["pred_ord_logits"].sum() * 0.0
        return outputs["pred_logits"].sum() * 0.0

    def _get_matched_targets(self, outputs, targets, indices):
        if "pred_ord_logits" not in outputs or not self.use_ordinal_branch:
            return None, None

        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            return None, None

        pred_ord_logits = outputs["pred_ord_logits"][idx]
        labels = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)], dim=0)
        valid = (labels >= 0) & (labels < self.ordinal_num_classes)
        if self.ordinal_ignore_labels:
            for label in self.ordinal_ignore_labels:
                valid = valid & (labels != label)
        if not valid.any():
            return None, None
        pred_ord_logits = pred_ord_logits[valid]
        labels = labels[valid]
        return pred_ord_logits, labels

    def _loss_ord_branch(self, outputs, targets, indices, num_boxes):
        """Cumulative BCE: P(y > k) for k in [0, ordinal_num_classes - 2]."""
        pred_ord_logits, labels = self._get_matched_targets(outputs, targets, indices)
        if pred_ord_logits is None:
            return {"loss_ord_branch": self._zero_like_loss(outputs)}

        num_ord = pred_ord_logits.shape[-1]
        expected_num_ord = self.ordinal_num_classes - 1
        if num_ord != expected_num_ord:
            raise ValueError(
                f"Expected pred_ord_logits last dim to be {expected_num_ord} "
                f"for ordinal_num_classes={self.ordinal_num_classes}, got {num_ord}."
            )
        thresholds = torch.arange(num_ord, device=labels.device)
        ord_targets = (labels[:, None] > thresholds[None, :]).float()
        loss = F.binary_cross_entropy_with_logits(pred_ord_logits, ord_targets, reduction="mean")
        return {"loss_ord_branch": loss}

    def _loss_ord_mono(self, outputs, targets, indices, num_boxes):
        """Monotonicity regulariser: P(y > k) >= P(y > k+1)."""
        pred_ord_logits, _ = self._get_matched_targets(outputs, targets, indices)
        if pred_ord_logits is None:
            return {"loss_ord_mono": self._zero_like_loss(outputs)}

        probs = pred_ord_logits.sigmoid()
        if probs.shape[-1] < 2:
            return {"loss_ord_mono": probs.sum() * 0.0}

        loss = F.relu(probs[..., 1:] - probs[..., :-1]).mean()
        return {"loss_ord_mono": loss}

    # ------------------------------------------------------------------
    #  Loss routing
    # ------------------------------------------------------------------

    def _route_loss(self, loss_name, outputs, targets, indices, num_boxes):
        mapping = {
            "vfl": self._loss_vfl,
            "boxes": self._loss_boxes,
            "ord_branch": self._loss_ord_branch,
            "ord_mono": self._loss_ord_mono,
        }
        if loss_name not in mapping:
            raise ValueError(
                f"Unknown loss '{loss_name}'. Available: {list(mapping.keys())}"
            )
        return mapping[loss_name](outputs, targets, indices, num_boxes)

    def _compute(self, outputs, targets, indices, num_boxes):
        losses = {}
        for name in self.losses:
            l_dict = self._route_loss(name, outputs, targets, indices, num_boxes)
            l_dict = {
                k: v * self.weight_dict.get(k, 1.0)
                for k, v in l_dict.items()
                if k in self.weight_dict
            }
            losses.update(l_dict)
        return losses

    # ------------------------------------------------------------------
    #  Forward (main + aux + denoising)
    # ------------------------------------------------------------------

    def forward(self, outputs, targets):
        outputs_main = {k: v for k, v in outputs.items() if "aux" not in k}
        indices = self.matcher(outputs_main, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = self._compute(outputs, targets, indices, num_boxes)

        # ---- aux outputs ----
        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux, targets)
                aux_dict = self._compute(aux, targets, aux_indices, num_boxes)
                aux_dict = {f"{k}_aux_{i}": v for k, v in aux_dict.items()}
                losses.update(aux_dict)

        # ---- denoising aux outputs ----
        if "dn_aux_outputs" in outputs:
            assert "dn_meta" in outputs
            dn_indices = self._get_cdn_matched_indices(outputs["dn_meta"], targets)
            num_boxes_dn = num_boxes * outputs["dn_meta"]["dn_num_group"]

            for i, aux in enumerate(outputs["dn_aux_outputs"]):
                dn_losses = {}
                for name in self.losses:
                    if not self.apply_dn_ordinal and name in ("ord_branch", "ord_mono"):
                        continue
                    l_dict = self._route_loss(name, aux, targets, dn_indices, num_boxes_dn)
                    l_dict = {
                        k: v * self.weight_dict.get(k, 1.0)
                        for k, v in l_dict.items()
                        if k in self.weight_dict
                    }
                    dn_losses.update(l_dict)
                dn_losses = {f"{k}_dn_{i}": v for k, v in dn_losses.items()}
                losses.update(dn_losses)

        return losses

    # ------------------------------------------------------------------
    #  Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx = dn_meta["dn_positive_idx"]
        dn_num_group = dn_meta["dn_num_group"]
        num_gts = [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((
                    torch.zeros(0, dtype=torch.int64, device=device),
                    torch.zeros(0, dtype=torch.int64, device=device),
                ))
        return dn_match_indices

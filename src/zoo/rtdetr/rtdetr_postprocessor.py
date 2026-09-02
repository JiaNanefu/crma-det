"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision

from src.core import register


__all__ = ['RTDETRPostProcessor']


@register
class RTDETRPostProcessor(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category']
    
    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
        ordinal_rerank_enable=False,
        ordinal_rerank_lambda=0.25,
        ordinal_rerank_max_gap=0.05,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category 
        self.ordinal_rerank_enable = bool(ordinal_rerank_enable)
        self.ordinal_rerank_lambda = float(ordinal_rerank_lambda)
        self.ordinal_rerank_max_gap = float(ordinal_rerank_max_gap)
        if not 0.0 <= self.ordinal_rerank_lambda <= 1.0:
            raise ValueError(
                f"ordinal_rerank_lambda must be in [0, 1], got {self.ordinal_rerank_lambda}."
            )
        if self.ordinal_rerank_max_gap < 0.0:
            raise ValueError(
                f"ordinal_rerank_max_gap must be non-negative, got {self.ordinal_rerank_max_gap}."
            )
        self.deploy_mode = False 

    def extra_repr(self) -> str:
        return (
            f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, '
            f'num_top_queries={self.num_top_queries}, '
            f'ordinal_rerank_enable={self.ordinal_rerank_enable}'
        )

    def _ordinal_class_scores(self, ord_logits):
        if ord_logits.shape[-1] != self.num_classes - 1:
            raise ValueError(
                f"Expected pred_ord_logits last dim to be {self.num_classes - 1}, "
                f"got {ord_logits.shape[-1]}."
            )

        ord_probs = torch.sigmoid(ord_logits)
        first = 1.0 - ord_probs[..., :1]
        middle = ord_probs[..., :-1] - ord_probs[..., 1:]
        last = ord_probs[..., -1:]
        class_scores = torch.cat([first, middle, last], dim=-1).clamp_min(0.0)
        denom = class_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return class_scores / denom

    def _apply_adjacent_ordinal_rerank(self, cls_scores, outputs):
        if not self.ordinal_rerank_enable or 'pred_ord_logits' not in outputs:
            return cls_scores

        ord_scores = self._ordinal_class_scores(outputs['pred_ord_logits'])
        if ord_scores.shape != cls_scores.shape:
            raise ValueError(
                f"Ordinal class scores shape {tuple(ord_scores.shape)} must match "
                f"class scores shape {tuple(cls_scores.shape)}."
            )

        top_scores, top_labels = torch.topk(cls_scores, k=2, dim=-1)
        adjacent = (top_labels[..., 0] - top_labels[..., 1]).abs() == 1
        close = (top_scores[..., 0] - top_scores[..., 1]) <= self.ordinal_rerank_max_gap
        rerank_mask = (adjacent & close).unsqueeze(-1)

        candidate_mask = torch.zeros_like(cls_scores, dtype=torch.bool)
        candidate_mask.scatter_(-1, top_labels[..., :1], True)
        candidate_mask.scatter_(-1, top_labels[..., 1:2], True)

        eps = 1e-6
        lam = self.ordinal_rerank_lambda
        fused_scores = torch.exp(
            (1.0 - lam) * torch.log(cls_scores.clamp_min(eps))
            + lam * torch.log(ord_scores.clamp_min(eps))
        )
        return torch.where(rerank_mask & candidate_mask, fused_scores, cls_scores)
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes):

        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores = self._apply_adjacent_ordinal_rerank(scores, outputs)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            boxes = bbox_pred
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.coco import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        
        return results
        

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 

    @property
    def iou_types(self, ):
        return ('bbox', )

# ------------------------------------------------------------------------
# Cuboid-DETR
# Copyright (c) 2025 t-34400
# Licensed under the Apache License, Version 2.0 (see LICENSE for details)
#
# Based on:
#  - RF-DETR, Copyright (c) 2025 Roboflow
#    Licensed under the Apache License, Version 2.0
#  - LW-DETR, Copyright (c) 2024 Baidu
#    Licensed under the Apache License, Version 2.0
#  - Conditional DETR, Copyright (c) 2021 Microsoft
#    Licensed under the Apache License, Version 2.0
#  - DETR, Copyright (c) Facebook, Inc. and its affiliates
#    Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

import copy
import math
from typing import Callable
import torch
from torch import nn

from .util.misc import (
    NestedTensor, nested_tensor_from_tensor_list
)
from .transformer import MLP


class CuboidDetr(nn.Module):

    def __init__(self, 
                 backbone, 
                 transformer, 
                 num_queries, 
                 group_detr=1,
                 aux_loss=False, 
                 lite_refpoint_refine=False, 
                 bbox_reparam=False):
        super().__init__()

        self.backbone = backbone
        self.transformer = transformer

        self.num_queries = num_queries
        self.group_detr = group_detr
        self.aux_loss = aux_loss
        self.lite_refpoint_refine = lite_refpoint_refine
        self.bbox_reparam = bbox_reparam
        self.two_stage = getattr(self.transformer, "two_stage", False)

        d_model = transformer.d_model
        num_classes = 2

        self.class_embed = nn.Linear(d_model, num_classes)

        self.bbox_embed = MLP(d_model, d_model, 4, 3)

        self.num_kp = 8
        self.kp_dim = self.num_kp * 2

        self.refpoint_dim = 4

        query_dim = self.refpoint_dim
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        self.query_feat = nn.Embedding(num_queries * group_detr, d_model)

        self.kp_head = MLP(d_model + 4, d_model, self.kp_dim, 3)

        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.refpoint_delta_head = MLP(d_model, d_model, self.refpoint_dim, 3)
        self.transformer.decoder.refpoint_embed = self.refpoint_delta_head

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        with torch.no_grad():
            self.class_embed.bias.data = torch.ones(num_classes) * bias_value

            nn.init.zeros_(self.bbox_embed.layers[-1].weight)
            nn.init.zeros_(self.bbox_embed.layers[-1].bias)

            nn.init.zeros_(self.refpoint_delta_head.layers[-1].weight)
            nn.init.zeros_(self.refpoint_delta_head.layers[-1].bias)

            nn.init.zeros_(self.kp_head.layers[-1].weight)
            nn.init.zeros_(self.kp_head.layers[-1].bias)

        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)])
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)])

        self._export = False

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for _, m in self.named_modules():
            if hasattr(m, "export") and isinstance(m.export, Callable) and hasattr(m, "_export") and not m._export:
                m.export()

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)
        srcs, masks = zip(*(feat.decompose() for feat in features))
        srcs, masks = list(srcs), list(masks)

        if self.training:
            ref_w, qry_w = self.refpoint_embed.weight, self.query_feat.weight
        else:
            ref_w = self.refpoint_embed.weight[:self.num_queries]
            qry_w = self.query_feat.weight[:self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(srcs, masks, poss, ref_w, qry_w)
        out = {}

        if hs is not None:
            has_layered_ref = (ref_unsigmoid is not None and ref_unsigmoid.dim() == 4)

            if has_layered_ref:
                ref_last = ref_unsigmoid[-1]
            else:
                ref_last = ref_unsigmoid

            out.update(self._predict_from(hs[-1], ref_last))

            if self.training and self.aux_loss:
                if has_layered_ref:
                    aux_refs = ref_unsigmoid[:-1]
                else:
                    aux_refs = [ref_last] * (hs.shape[0] - 1)

                out["aux_outputs"] = [
                    self._predict_from(h, r)
                    for h, r in zip(hs[:-1], aux_refs)
                ]

        if self.two_stage and (hs_enc is not None) and (ref_enc is not None):
            group = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group, dim=1)
            cls_enc = torch.cat(
                [self.transformer.enc_out_class_embed[g](hs_enc_list[g]) for g in range(group)],
                dim=1
            )
            out["enc_outputs"] = {"pred_logits": cls_enc, "pred_boxes": ref_enc}

        return out

    def forward_export(self, tensors):
        srcs, _, poss = self.backbone(tensors)
        ref_w = self.refpoint_embed.weight[:self.num_queries]
        qry_w = self.query_feat.weight[:self.num_queries]
        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(srcs, None, poss, ref_w, qry_w)

        if hs is not None:
            if ref_unsigmoid is not None and ref_unsigmoid.dim() == 4:
                ref_last = ref_unsigmoid[-1]
            else:
                ref_last = ref_unsigmoid
            out = self._predict_from(hs[-1], ref_last)
            return out["pred_logits"], out["pred_boxes"], out["pred_kp"]
        else:
            assert self.two_stage
            logits = self.transformer.enc_out_class_embed[0](hs_enc)
            return logits, ref_enc, None

    def _predict_from(self, h: torch.Tensor, ref: torch.Tensor):
        # ref: (B, Nq, 4)
        ref_box = ref[..., :4]

        if self.bbox_reparam:
            d = self.bbox_embed(h)               # (B, Nq, 4)
            cx = d[..., :2] * ref_box[..., 2:] + ref_box[..., :2]
            wh = d[..., 2:].exp() * ref_box[..., 2:]
            boxes = torch.cat([cx, wh], dim=-1)
        else:
            boxes = (self.bbox_embed(h) + ref_box).sigmoid()

        logits = self.class_embed(h)
        B, Nq, _ = h.shape

        kp_in = torch.cat([h, boxes], dim=-1)          # (B, Nq, d_model + 4)
        kp_logits = self.kp_head(kp_in)                # (B, Nq, 16)

        kp_rel = torch.tanh(kp_logits.view(B, Nq, self.num_kp, 2))  # (B, Nq, K, 2)

        box_center = boxes[..., :2].unsqueeze(2)       # (B, Nq, 1, 2)
        box_wh = boxes[..., 2:].unsqueeze(2)           # (B, Nq, 1, 2)

        pred_kp = box_center + 0.5 * kp_rel * box_wh   # (B, Nq, K, 2)

        return {"pred_logits": logits, "pred_boxes": boxes, "pred_kp": pred_kp}

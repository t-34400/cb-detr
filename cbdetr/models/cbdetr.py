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
import torch.nn.functional as F

from .util.misc import NestedTensor, nested_tensor_from_tensor_list
from .transformer import MLP


class KeypointNeck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_scales: int,
        upsample_factor: float = 1.0,
    ):
        super().__init__()
        self.upsample_factor = upsample_factor

        self.proj_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
                for _ in range(num_scales)
            ]
        )

        if upsample_factor != 1.0:
            self.refine = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.refine = None

    def forward(self, feats):
        if isinstance(feats, torch.Tensor):
            feats = [feats]

        target_h, target_w = feats[0].shape[-2:]
        fused = None
        dtype = feats[0].dtype

        for feat, conv in zip(feats, self.proj_convs):
            x = feat.to(conv.weight.dtype)
            x = conv(x)
            if x.shape[-2:] != (target_h, target_w):
                x = F.interpolate(
                    x,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            fused = x if fused is None else fused + x

        fused = fused.to(dtype)

        if self.upsample_factor != 1.0:
            fused = F.interpolate(
                fused,
                scale_factor=self.upsample_factor,
                mode="bilinear",
                align_corners=False,
            )
            fused = self.refine(fused)

        return fused


class CuboidJointDecoder(nn.Module):
    """Joint refinement for all cuboid vertices via self-attention."""

    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        num_kp: int = 8,
        out_dim: int = 2,
    ):
        super().__init__()
        self.num_kp = num_kp

        self.input_proj = nn.Linear(in_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.output_head = nn.Linear(d_model, out_dim)

        self.kp_type_embed = nn.Embedding(num_kp, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Nq, K, C_in)
        returns: (B, Nq, K, out_dim)
        """
        B, Nq, K, C_in = x.shape
        assert K == self.num_kp

        x = x.view(B * Nq, K, C_in)
        x = self.input_proj(x)

        idx = torch.arange(K, device=x.device).unsqueeze(0)
        kp_pos = self.kp_type_embed(idx)
        x = x + kp_pos

        x = self.encoder(x)
        delta = self.output_head(x)

        delta = delta.view(B, Nq, K, -1)
        return delta


class CuboidDetr(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        num_queries: int,
        kp_feat_dim: int = 128,
        group_detr: int = 1,
        aux_loss: bool = False,
        lite_refpoint_refine: bool = False,
        bbox_reparam: bool = False,
        kp_upsample_factor: float = 1.0,
    ):
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

        self.num_kp = 8
        self.kp_feat_dim = kp_feat_dim
        self.kp_dim = self.num_kp * 2
        self.refpoint_dim = 4

        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)

        query_dim = self.refpoint_dim
        total_queries = num_queries * group_detr
        self.refpoint_embed = nn.Embedding(total_queries, query_dim)
        self.query_feat = nn.Embedding(total_queries, d_model)

        self.kp_coarse_head = MLP(d_model + 4, d_model, self.kp_dim, 3)

        inner_backbone = backbone[0] if isinstance(backbone, nn.Sequential) else backbone

        if hasattr(inner_backbone, "projector"):
            kp_in_channels = getattr(inner_backbone.projector, "out_channels", d_model)
        else:
            kp_in_channels = d_model

        if hasattr(inner_backbone, "projector_scale"):
            kp_num_scales = len(inner_backbone.projector_scale)
        else:
            kp_num_scales = 1

        self.kp_neck = KeypointNeck(
            in_channels=kp_in_channels,
            out_channels=self.kp_feat_dim,
            num_scales=kp_num_scales,
            upsample_factor=kp_upsample_factor,
        )

        refine_in_dim = d_model + 4 + 2 + self.kp_feat_dim

        self.num_kp_samples = 4
        self.sample_offset_scale = 0.25
        self.kp_sample_offset_head = MLP(
            d_model + 4 + 2,
            d_model,
            self.num_kp_samples * 2,
            3,
        )
        self.kp_sample_weight_head = MLP(
            d_model + 4 + 2,
            d_model,
            self.num_kp_samples,
            3,
        )

        self.kp_joint_decoder = CuboidJointDecoder(
            in_dim=refine_in_dim,
            d_model=self.kp_feat_dim,
            num_layers=2,
            nhead=4,
            num_kp=self.num_kp,
            out_dim=2,
        )

        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.refpoint_delta_head = MLP(d_model, d_model, self.refpoint_dim, 3)
        self.transformer.decoder.refpoint_embed = self.refpoint_delta_head

        self._init_parameters()
        self._export = False

    def _init_parameters(self) -> None:
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        with torch.no_grad():
            self.class_embed.bias.data.fill_(bias_value)

            nn.init.zeros_(self.bbox_embed.layers[-1].weight)
            nn.init.zeros_(self.bbox_embed.layers[-1].bias)

            nn.init.zeros_(self.refpoint_delta_head.layers[-1].weight)
            nn.init.zeros_(self.refpoint_delta_head.layers[-1].bias)

            base_kp = torch.tensor(
                [
                    [-1, -1],
                    [1, -1],
                    [-1, 1],
                    [1, 1],
                    [-1, 0],
                    [1, 0],
                    [0, -1],
                    [0, 1],
                ]
            )

            nn.init.zeros_(self.kp_coarse_head.layers[-1].weight)
            self.kp_coarse_head.layers[-1].bias.copy_(base_kp.view(-1))

            nn.init.zeros_(self.kp_sample_offset_head.layers[-1].weight)
            nn.init.zeros_(self.kp_sample_offset_head.layers[-1].bias)
            nn.init.zeros_(self.kp_sample_weight_head.layers[-1].weight)
            nn.init.zeros_(self.kp_sample_weight_head.layers[-1].bias)

            nn.init.zeros_(self.kp_joint_decoder.output_head.weight)
            nn.init.zeros_(self.kp_joint_decoder.output_head.bias)

        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(self.group_detr)]
            )
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(self.group_detr)]
            )

    def export(self) -> None:
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

        for _, m in self.named_modules():
            if (
                hasattr(m, "export")
                and isinstance(m.export, Callable)
                and hasattr(m, "_export")
                and not m._export
            ):
                m.export()

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, poss = self.backbone(samples)
        srcs, masks = zip(*(feat.decompose() for feat in features))
        srcs = list(srcs)
        masks = list(masks)

        kp_feat_map = self.kp_neck([feat.tensors.detach() for feat in features])

        if self.training:
            ref_w = self.refpoint_embed.weight
            qry_w = self.query_feat.weight
        else:
            ref_w = self.refpoint_embed.weight[: self.num_queries]
            qry_w = self.query_feat.weight[: self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, ref_w, qry_w
        )

        out = {}

        if hs is not None:
            has_layered_ref = ref_unsigmoid is not None and ref_unsigmoid.dim() == 4
            ref_last = ref_unsigmoid[-1] if has_layered_ref else ref_unsigmoid

            out.update(self._predict_from(hs[-1], ref_last, kp_feat_map))

            if self.training and self.aux_loss:
                if has_layered_ref:
                    aux_refs = ref_unsigmoid[:-1]
                else:
                    aux_refs = [ref_last] * (hs.shape[0] - 1)

                out["aux_outputs"] = [
                    self._predict_from(h, r, kp_feat_map)
                    for h, r in zip(hs[:-1], aux_refs)
                ]

        if self.two_stage and (hs_enc is not None) and (ref_enc is not None):
            group = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group, dim=1)
            cls_enc = torch.cat(
                [
                    self.transformer.enc_out_class_embed[g](hs_enc_list[g])
                    for g in range(group)
                ],
                dim=1,
            )
            out["enc_outputs"] = {
                "pred_logits": cls_enc,
                "pred_boxes": ref_enc,
            }

        return out

    def forward_export(self, tensors: torch.Tensor):
        if isinstance(tensors, (list, torch.Tensor)):
            tensors = nested_tensor_from_tensor_list(tensors)

        features, poss = self.backbone(tensors)
        srcs, masks = zip(*(feat.decompose() for feat in features))
        srcs = list(srcs)
        masks = list(masks)

        kp_feat_map = self.kp_neck([feat.tensors for feat in features])

        ref_w = self.refpoint_embed.weight[: self.num_queries]
        qry_w = self.query_feat.weight[: self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, ref_w, qry_w
        )

        if hs is None:
            assert self.two_stage
            logits = self.transformer.enc_out_class_embed[0](hs_enc)
            return logits, ref_enc, None, None

        if ref_unsigmoid is not None and ref_unsigmoid.dim() == 4:
            ref_last = ref_unsigmoid[-1]
        else:
            ref_last = ref_unsigmoid

        out = self._predict_from(hs[-1], ref_last, kp_feat_map)
        return (
            out["pred_logits"],
            out["pred_boxes"],
            out["pred_kp"],
            out["pred_kp_coarse"],
        )

    def _predict_from(
        self,
        h: torch.Tensor,
        ref: torch.Tensor,
        kp_feat_map: torch.Tensor,
    ):
        ref_box = ref[..., :4]

        if self.bbox_reparam:
            d = self.bbox_embed(h)
            d = d.clamp(-10.0, 10.0)
            cx = d[..., :2] * ref_box[..., 2:] + ref_box[..., :2]
            wh = d[..., 2:].exp() * ref_box[..., 2:]
            boxes = torch.cat([cx, wh], dim=-1)
        else:
            boxes = (self.bbox_embed(h) + ref_box).sigmoid()

        logits = self.class_embed(h)
        B, Nq, _ = h.shape

        box_center_2d = boxes[..., :2]
        box_wh_2d = boxes[..., 2:]

        kp_in_coarse = torch.cat([h, boxes], dim=-1)
        kp_logits_coarse = self.kp_coarse_head(kp_in_coarse)
        kp_rel_coarse = torch.tanh(
            kp_logits_coarse.view(B, Nq, self.num_kp, 2)
        )

        pred_kp_coarse = (
            box_center_2d.unsqueeze(2)
            + 0.5 * kp_rel_coarse * box_wh_2d.unsqueeze(2)
        )

        box_center = box_center_2d.unsqueeze(2).unsqueeze(3)
        box_wh = box_wh_2d.unsqueeze(2).unsqueeze(3)

        h_exp = h.detach().unsqueeze(2).expand(-1, -1, self.num_kp, -1)
        box_exp = boxes.detach().unsqueeze(2).expand(-1, -1, self.num_kp, -1)
        sample_in = torch.cat(
            [h_exp, box_exp, kp_rel_coarse.detach()],
            dim=-1,
        )

        num_samples = self.num_kp_samples

        sample_in_flat = sample_in.view(B * Nq * self.num_kp, -1)
        sample_offset_flat = self.kp_sample_offset_head(sample_in_flat)
        sample_weight_flat = self.kp_sample_weight_head(sample_in_flat)

        sample_offset = torch.tanh(
            sample_offset_flat.view(B, Nq, self.num_kp, num_samples, 2)
        ) * self.sample_offset_scale
        sample_weight = sample_weight_flat.view(
            B, Nq, self.num_kp, num_samples
        )

        kp_rel_sample = kp_rel_coarse.detach().unsqueeze(3) + sample_offset

        sample_pos = box_center + 0.5 * kp_rel_sample * box_wh

        grid = sample_pos * 2.0 - 1.0
        grid = grid.view(B, Nq * self.num_kp, num_samples, 2)

        sampled_multi = F.grid_sample(
            kp_feat_map, grid, align_corners=False
        )

        C = self.kp_feat_dim
        sampled_multi = sampled_multi.view(
            B, C, Nq, self.num_kp, num_samples
        ).permute(0, 2, 3, 4, 1)

        att = F.softmax(sample_weight, dim=-1)
        sampled_flat = sampled_multi.reshape(
            B * Nq * self.num_kp, num_samples, C
        )
        att_flat = att.reshape(B * Nq * self.num_kp, num_samples, 1)
        sampled_agg = (att_flat * sampled_flat).sum(dim=1)
        sampled = sampled_agg.view(
            B, Nq, self.num_kp, C
        )

        refine_in = torch.cat(
            [h_exp, box_exp, kp_rel_coarse.detach(), sampled],
            dim=-1,
        )

        kp_delta = self.kp_joint_decoder(refine_in)
        kp_delta = torch.tanh(kp_delta)

        kp_rel_refined = kp_rel_coarse.detach() + kp_delta

        pred_kp = (
            box_center_2d.detach().unsqueeze(2)
            + 0.5 * kp_rel_refined * box_wh_2d.detach().unsqueeze(2)
        )

        return {
            "pred_logits": logits,
            "pred_boxes": boxes,
            "pred_kp": pred_kp,
            "pred_kp_coarse": pred_kp_coarse,
        }
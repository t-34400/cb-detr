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
from typing import Callable, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .util.misc import NestedTensor, nested_tensor_from_tensor_list
from .transformer import MLP

from cbdetr.util.cuboid_topology import EDGES


class HighFreqEnhancer(nn.Module):
    def __init__(self, channels: int, init_scale: float = 0.5):
        super().__init__()
        self.blur = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        nn.init.constant_((self.blur.weight), 1.0 / 9.0)
        self.blur.weight.requires_grad = False
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blur = self.blur(x)
        hf = x - blur
        return x + self.scale * hf


class ImageHFEncoder(nn.Module):
    """Lightweight line-aware encoder for high-frequency image features."""

    def __init__(self, out_channels: int):
        super().__init__()
        mid = max(out_channels // 2, 16)

        self.stem = nn.Sequential(
            nn.Conv2d(3, mid, 3, padding=1, bias=False),
            nn.GELU(),
        )

        # Depthwise convolutions emphasize horizontal and vertical line structures.
        self.dw_h = nn.Conv2d(
            mid,
            mid,
            kernel_size=(1, 3),
            padding=(0, 1),
            groups=mid,
            bias=False,
        )
        self.dw_v = nn.Conv2d(
            mid,
            mid,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=mid,
            bias=False,
        )

        self.proj = nn.Conv2d(mid, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        h = self.dw_h(x)
        v = self.dw_v(x)
        x = x + h + v
        x = self.proj(x)
        return F.gelu(x)


class KeypointNeck(nn.Module):
    """
    Multi-scale fusion neck for keypoint feature map.

    Design choices:
      - Fuse backbone scales at the resolution of the first feature.
      - No global upsampling; only local sampling is used later.
      - Optional small refinement conv block.
      - Final 2x upsampling to increase effective spatial resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_scales: int,
        refine: bool = True,
        use_hf_enhancer: bool = True,
    ):
        super().__init__()

        self.proj_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
                for _ in range(num_scales)
            ]
        )

        self.refine = (
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            if refine
            else None
        )

        self.hf = HighFreqEnhancer(out_channels) if use_hf_enhancer else None

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
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

        if self.refine is not None:
            fused = self.refine(fused)

        if self.hf is not None:
            fused = self.hf(fused)

        fused = F.interpolate(
            fused,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )

        return fused


class CuboidJointDecoder(nn.Module):
    """Joint refinement for all cuboid vertices via transformer encoder."""

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

        total_queries = num_queries * group_detr
        self.refpoint_embed = nn.Embedding(total_queries, self.refpoint_dim)
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
            refine=True,
            use_hf_enhancer=True,
        )

        self.img_hf_encoder = ImageHFEncoder(self.kp_feat_dim)
        self.kp_fuse = nn.Conv2d(self.kp_feat_dim * 2, self.kp_feat_dim, kernel_size=1)

        refine_in_dim = (
            d_model  # query feature
            + 4  # bbox(xywh)
            + 2  # coarse kp offset
            + self.kp_feat_dim  # point-based feature
            + self.kp_feat_dim  # edge-based feature
        )

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

        self.num_edge_samples = 8
        self.register_buffer("edges", EDGES)  # (E, 2)

        edge_in_dim = d_model + 4
        self.edge_sample_head = MLP(edge_in_dim, d_model, self.num_edge_samples, 3)

        K = self.num_kp
        E = EDGES.shape[0]
        vertex_edge_mask = torch.zeros(K, E, dtype=torch.float32)
        for e in range(E):
            i, j = EDGES[e]
            vertex_edge_mask[i, e] = 1.0
            vertex_edge_mask[j, e] = 1.0
        self.register_buffer("vertex_edge_mask", vertex_edge_mask)

        deg = vertex_edge_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        self.register_buffer("vertex_edge_deg_inv", 1.0 / deg)

        Ck = self.kp_feat_dim
        C_ctx = self.kp_feat_dim

        self.edge_att_q = nn.Linear(C_ctx, Ck // 2)
        self.edge_att_k = nn.Linear(Ck, Ck // 2)
        self.edge_att_v = nn.Linear(Ck, Ck)

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

    def _compute_kp_feat_map(
        self,
        features: List[NestedTensor],
        images: torch.Tensor,
        detach_backbone: bool,
    ) -> torch.Tensor:
        feat_tensors = [feat.tensors for feat in features]
        if detach_backbone:
            feat_tensors = [t.detach() for t in feat_tensors]

        kp_feat_map = self.kp_neck(feat_tensors)

        B, _, Hk, Wk = kp_feat_map.shape
        img_resized = F.interpolate(
            images,
            size=(Hk, Wk),
            mode="bilinear",
            align_corners=False,
        )
        img_hf = self.img_hf_encoder(img_resized)

        fused = torch.cat([kp_feat_map, img_hf], dim=1)
        kp_feat_map = self.kp_fuse(fused)
        return kp_feat_map

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        images = samples.tensors  # (B, 3, H, W)
        features, poss = self.backbone(samples)
        srcs, masks = zip(*(feat.decompose() for feat in features))
        srcs = list(srcs)
        masks = list(masks)

        kp_feat_map = self._compute_kp_feat_map(
            features=features, images=images, detach_backbone=True
        )

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

        images = tensors.tensors
        features, poss = self.backbone(tensors)
        srcs, masks = zip(*(feat.decompose() for feat in features))
        srcs = list(srcs)
        masks = list(masks)

        kp_feat_map = self._compute_kp_feat_map(
            features=features, images=images, detach_backbone=False
        )

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

    def _sample_kp_point_features(
        self,
        h: torch.Tensor,              # (B, Nq, Cq)
        boxes: torch.Tensor,          # (B, Nq, 4) in [0,1]
        kp_rel_coarse: torch.Tensor,  # (B, Nq, K, 2) in [-1,1]
        kp_feat_map: torch.Tensor,    # (B, Ck, Hk, Wk)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            sampled: (B, Nq, K, Ck)
            h_exp:   (B, Nq, K, Cq)  # query feature per KP
            box_exp: (B, Nq, K, 4)   # bbox per KP
        """
        B, Nq, _ = h.shape
        K = self.num_kp
        Ck = self.kp_feat_dim
        num_samples = self.num_kp_samples

        box_center_2d = boxes[..., :2]
        box_wh_2d = boxes[..., 2:]

        box_center = box_center_2d.unsqueeze(2).unsqueeze(3)  # (B, Nq, 1, 1, 2)
        box_wh = box_wh_2d.unsqueeze(2).unsqueeze(3)          # (B, Nq, 1, 1, 2)

        h_exp = h.unsqueeze(2).expand(-1, -1, K, -1)
        box_exp = boxes.unsqueeze(2).expand(-1, -1, K, -1)

        sample_in = torch.cat(
            [h_exp.detach(), box_exp.detach(), kp_rel_coarse.detach()],
            dim=-1,
        )  # (B, Nq, K, Cq+4+2)

        sample_in_flat = sample_in.view(B * Nq * K, -1)
        sample_offset_flat = self.kp_sample_offset_head(sample_in_flat)
        sample_weight_flat = self.kp_sample_weight_head(sample_in_flat)

        sample_offset = torch.tanh(
            sample_offset_flat.view(B, Nq, K, num_samples, 2)
        ) * self.sample_offset_scale
        sample_weight = sample_weight_flat.view(B, Nq, K, num_samples)

        kp_rel_sample = kp_rel_coarse.detach().unsqueeze(3) + sample_offset
        sample_pos = box_center + 0.5 * kp_rel_sample * box_wh  # (B,Nq,K,S,2), in [0,1]

        grid = sample_pos * 2.0 - 1.0
        grid = grid.view(B, Nq * K, num_samples, 2)

        sampled_multi = F.grid_sample(
            kp_feat_map, grid, align_corners=False
        )  # (B, Ck, Nq*K*S, 1)

        sampled_multi = sampled_multi.view(
            B, Ck, Nq, K, num_samples
        ).permute(0, 2, 3, 4, 1)  # (B, Nq, K, S, Ck)

        att = F.softmax(sample_weight, dim=-1)  # (B,Nq,K,S)
        sampled_flat = sampled_multi.reshape(
            B * Nq * K, num_samples, Ck
        )
        att_flat = att.reshape(B * Nq * K, num_samples, 1)
        sampled_agg = (att_flat * sampled_flat).sum(dim=1)
        sampled = sampled_agg.view(B, Nq, K, Ck)

        return sampled, h_exp, box_exp

    def _sample_edge_features_raw(
        self,
        kp_feat_map: torch.Tensor,     # (B, Ck, Hk, Wk)
        pred_kp_coarse: torch.Tensor,  # (B, Nq, K, 2)
        h: torch.Tensor,               # (B, Nq, Cq)
    ) -> torch.Tensor:
        """
        Sample edge features along predicted coarse keypoints using
        learnable sampling positions per edge and query.
        """
        B, Ck, Hk, Wk = kp_feat_map.shape
        B2, Nq, K, _ = pred_kp_coarse.shape
        assert B == B2 and K == self.num_kp

        edges = self.edges  # (E, 2)
        E = edges.shape[0]
        num_samples = self.num_edge_samples

        p_u = pred_kp_coarse[:, :, edges[:, 0], :]  # (B, Nq, E, 2)
        p_v = pred_kp_coarse[:, :, edges[:, 1], :]  # (B, Nq, E, 2)

        # Broadcast query features to per-edge context.
        h_edge = h.unsqueeze(2).expand(-1, -1, E, -1)  # (B, Nq, E, Cq)

        edge_in = torch.cat([h_edge, p_u, p_v], dim=-1)  # (B, Nq, E, Cq+4+4)
        edge_in_flat = edge_in.view(B * Nq * E, -1)

        s_raw = self.edge_sample_head(edge_in_flat)  # (B*Nq*E, S)
        s = torch.sigmoid(s_raw).view(
            B, Nq, E, num_samples, 1
        )  # (B, Nq, E, S, 1) in [0,1]

        p_line = (1.0 - s) * p_u.unsqueeze(3) + s * p_v.unsqueeze(3)

        grid = p_line.view(B, Nq * E * num_samples, 1, 2) * 2.0 - 1.0

        sampled = F.grid_sample(
            kp_feat_map, grid, align_corners=False
        )

        sampled = sampled.view(B, Ck, Nq, E, num_samples).permute(
            0, 2, 3, 4, 1
        )

        edge_feat = sampled.mean(dim=3)  # (B, Nq, E, Ck)
        return edge_feat

    def _aggregate_edge_features(
        self,
        edge_feat: torch.Tensor,     # (B, Nq, E, Ck)
        kp_point_feat: torch.Tensor, # (B, Nq, K, C_ctx)
    ) -> torch.Tensor:
        B, Nq, E, Ck = edge_feat.shape
        _, _, K, C_ctx = kp_point_feat.shape

        Q = self.edge_att_q(kp_point_feat)        # (B, Nq, K, D)
        K_e = self.edge_att_k(edge_feat)          # (B, Nq, E, D)
        V_e = self.edge_att_v(edge_feat)          # (B, Nq, E, Ck)

        Q = Q.unsqueeze(3)                        # (B, Nq, K, 1, D)
        K_e = K_e.unsqueeze(2)                    # (B, Nq, 1, E, D)
        logits = (Q * K_e).sum(-1)                # (B, Nq, K, E)

        mask = self.vertex_edge_mask.view(1, 1, K, E)
        logits = logits.masked_fill(mask == 0, -1e4)

        att = F.softmax(logits, dim=-1)          # (B, Nq, K, E)

        V_e = V_e.unsqueeze(2)                   # (B, Nq, 1, E, Ck)
        att = att.unsqueeze(-1)                  # (B, Nq, K, E, 1)

        kp_edge_feat = (att * V_e).sum(dim=3)    # (B, Nq, K, Ck)
        return kp_edge_feat

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

        # coarse kp
        kp_in_coarse = torch.cat([h, boxes], dim=-1)
        kp_logits_coarse = self.kp_coarse_head(kp_in_coarse)
        kp_rel_coarse = torch.tanh(
            kp_logits_coarse.view(B, Nq, self.num_kp, 2)
        )  # [-1,1]

        pred_kp_coarse = (
            box_center_2d.unsqueeze(2)
            + 0.5 * kp_rel_coarse * box_wh_2d.unsqueeze(2)
        )  # [0,1]

        kp_point_feat, h_exp, box_exp = self._sample_kp_point_features(
            h=h,
            boxes=boxes,
            kp_rel_coarse=kp_rel_coarse,
            kp_feat_map=kp_feat_map,
        )

        edge_feat = self._sample_edge_features_raw(
            kp_feat_map=kp_feat_map,
            pred_kp_coarse=pred_kp_coarse,
            h=h,
        )

        kp_edge_feat = self._aggregate_edge_features(
            edge_feat=edge_feat,
            kp_point_feat=kp_point_feat,
        )

        refine_in = torch.cat(
            [
                h_exp,                      # (B,Nq,K,Cq)
                box_exp,                    # (B,Nq,K,4)
                kp_rel_coarse.detach(),     # (B,Nq,K,2)
                kp_point_feat,              # (B,Nq,K,Ck)
                kp_edge_feat,               # (B,Nq,K,Ck)
            ],
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

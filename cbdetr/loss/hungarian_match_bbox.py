# ------------------------------------------------------------------------
# Cuboid-DETR
# Copyright (c) 2025 t-34400
# Licensed under the Apache License, Version 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from cbdetr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

@torch.no_grad()
def hungarian_match_bbox(outputs, targets,
                         cost_class=1.0, cost_bbox=1.0, cost_giou=1.0,
                         group_detr=1):
    """
    outputs: {
        "pred_logits": [B, Q_total, 2],  # [no-object, object]
        "pred_boxes": [B, Q_total, 4],   # (cx, cy, w, h)
    }
    Q_total = num_queries * group_detr
    """
    bs, num_queries_total = outputs["pred_logits"].shape[:2]
    if group_detr > 1:
        assert num_queries_total % group_detr == 0
        num_queries = num_queries_total // group_detr
    else:
        num_queries = num_queries_total

    flat_logits = outputs["pred_logits"].flatten(0, 1)  # [B*Q_total, 2]
    out_bbox = outputs["pred_boxes"].flatten(0, 1)      # [B*Q_total, 4]

    prob = flat_logits.softmax(-1)                      # [B*Q_total, 2]
    p_obj = prob[:, 1].clamp(min=1e-6, max=1.0 - 1e-6)

    tgt_bbox = torch.cat([v["boxes"] for v in targets])
    sizes = [len(v["boxes"]) for v in targets]

    cost_class_mat = -p_obj.log().unsqueeze(1).expand(-1, tgt_bbox.size(0))
    cost_bbox_mat = torch.cdist(out_bbox, tgt_bbox, p=1)
    giou = generalized_box_iou(
        box_cxcywh_to_xyxy(out_bbox),
        box_cxcywh_to_xyxy(tgt_bbox)
    )
    cost_giou_mat = -giou

    C = cost_bbox * cost_bbox_mat + cost_class * cost_class_mat + cost_giou * cost_giou_mat
    C = C.view(bs, num_queries_total, -1).cpu().float()

    indices = []
    start = 0
    for i, size in enumerate(sizes):
        if size == 0:
            indices.append((
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
            ))
            continue

        row_inds_all = []
        col_inds_all = []

        for g in range(group_detr):
            q_start = g * num_queries
            q_end = (g + 1) * num_queries

            c = C[i, q_start:q_end, start:start + size]  # [num_queries, size]

            if c.numel() == 0:
                continue

            row_ind, col_ind = linear_sum_assignment(c)

            row_ind = torch.as_tensor(row_ind + q_start, dtype=torch.int64)
            col_ind = torch.as_tensor(col_ind, dtype=torch.int64)

            row_inds_all.append(row_ind)
            col_inds_all.append(col_ind)

        if len(row_inds_all) == 0:
            indices.append((
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
            ))
        else:
            indices.append((
                torch.cat(row_inds_all, dim=0),
                torch.cat(col_inds_all, dim=0),
            ))

        start += size

    return indices

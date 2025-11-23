# ------------------------------------------------------------------------
# Cuboid-DETR
# Copyright (c) 2025 t-34400
# Licensed under the Apache License, Version 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------

import torch

# ---------- topology ----------
# 0: back-bottom-left
# 1: back-bottom-right
# 2: back-top-right
# 3: back-top-left
# 4: front-bottom-left
# 5: front-bottom-right
# 6: front-top-right
# 7: front-top-left

EDGES = torch.tensor([
    [0,1],[1,2],[2,3],[3,0],   # back face
    [4,5],[5,6],[6,7],[7,4],   # front face
    [0,4],[1,5],[2,6],[3,7],   # depth edges
], dtype=torch.long)

PARALLEL_GROUPS = [
    torch.tensor([[0,1],[3,2],[4,5],[7,6]], dtype=torch.long),  # x-like (left→right)
    torch.tensor([[0,3],[1,2],[4,7],[5,6]], dtype=torch.long),  # y-like (bottom→top)
    torch.tensor([[0,4],[1,5],[2,6],[3,7]], dtype=torch.long),  # z-like (back→front)
]

FACES = torch.tensor([
    [0,3,2,1],  # back
    [4,5,6,7],  # front
    [0,4,7,3],  # left
    [1,2,6,5],  # right
    [0,1,5,4],  # bottom
    [3,7,6,2],  # top
], dtype=torch.long)

# ---------- utils ----------
def bbox_from_kp(kp):
    """Tight axis-aligned bbox from 8 points.
    kp: (M,8,2) in *image-normalized* coords -> (M,4) (cx,cy,w,h) in same coords
    """
    x_min, _ = kp[..., 0].min(dim=1)
    y_min, _ = kp[..., 1].min(dim=1)
    x_max, _ = kp[..., 0].max(dim=1)
    y_max, _ = kp[..., 1].max(dim=1)
    w = (x_max - x_min).clamp_min(1e-6)
    h = (y_max - y_min).clamp_min(1e-6)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    return torch.stack([cx, cy, w, h], dim=-1)


def huber(x, delta=0.1, reduction="none"):
    """Smooth L1 with explicit delta."""
    absx = x.abs()
    loss = torch.where(absx < delta, 0.5 * (x ** 2) / delta, absx - 0.5 * delta)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def direction_loss(u, v, eps=1e-8):
    dot = (u * v).sum(-1)
    nu = u.norm(dim=-1).clamp_min(eps)
    nv = v.norm(dim=-1).clamp_min(eps)
    cos = dot / (nu * nv)
    cos = cos.clamp(-1.0, 1.0)
    return 1.0 - cos


# ---------- losses (image-normalized coordinates) ----------
def loss_kp_core(pred_kp, gt_kp_perm, use_huber=True, delta=0.05, coord_scale=1.0):
    """Core keypoint regression loss in image-normalized coordinates.

    pred_kp, gt_kp_perm: (M,8,2) in [0,1] image-normalized coords
    coord_scale: optional scale factor to roughly match pixel-domain magnitude
                 (e.g., max(H, W)). Defaults to 1.0.
    """
    diff = (pred_kp - gt_kp_perm) * coord_scale
    if use_huber:
        l = huber(diff, delta=delta * coord_scale, reduction="none").sum(-1)  # (M,8)
    else:
        l = diff.abs().sum(-1)  # L1 per point (M,8)
    return l.mean(-1)  # (M,)


def loss_bbox_from_kp_consistency(pred_bbox, pred_kp):
    """Encourage predicted bbox to agree with the tight bbox of predicted keypoints.

    All in the same image-normalized coordinate system.
    """
    tight = bbox_from_kp(pred_kp)               # (M,4)
    return (pred_bbox - tight).abs().sum(-1)    # (M,)


def loss_edges(pred_kp, gt_kp_perm):
    """Edge length + direction consistency in image-normalized space."""
    Pn = pred_kp
    Tn = gt_kp_perm

    p_e = Pn[:, EDGES[:, 0]] - Pn[:, EDGES[:, 1]]   # (M,12,2)
    t_e = Tn[:, EDGES[:, 0]] - Tn[:, EDGES[:, 1]]   # (M,12,2)

    len_p = p_e.norm(dim=-1)
    len_t = t_e.norm(dim=-1)
    l_len = (len_p - len_t).abs().mean(-1)          # (M,)

    dir_err = direction_loss(p_e, t_e)              # (M,12)
    l_dir = dir_err.mean(-1)                        # (M,)

    return l_len + l_dir


def loss_faces_convexity(pred_kp, gt_kp, eps=1e-6, alpha=0.1):
    def face_s(kp):
        polys = kp[:, FACES]          # (M,6,4,2)
        e0 = polys[..., 1, :] - polys[..., 0, :]
        e1 = polys[..., 2, :] - polys[..., 1, :]
        e2 = polys[..., 3, :] - polys[..., 2, :]
        e3 = polys[..., 0, :] - polys[..., 3, :]
        def cross_z(a, b):
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        return torch.stack([
            cross_z(e0, e1),
            cross_z(e1, e2),
            cross_z(e2, e3),
            cross_z(e3, e0),
        ], dim=-1)  # (M,6,4)

    s_p = face_s(pred_kp)
    s_g = face_s(gt_kp)

    area_p = s_p.abs()
    area_g = s_g.abs()

    ratio = area_p / (area_g + eps)
    pen_area = (alpha - ratio).clamp_min(0.0).mean(dim=(-1, -2))

    sign_p = torch.sign(s_p)
    sign_g = torch.sign(s_g)
    pen_sign = (sign_p - sign_g).abs().mean(dim=(-1, -2))

    return pen_area + pen_sign


def loss_repulsion(pred_kp, gt_kp, tau_ratio=0.5, eps=1e-6):
    P = pred_kp
    G = gt_kp
    diff_p = P[:, :, None, :] - P[:, None, :, :]
    diff_g = G[:, :, None, :] - G[:, None, :, :]
    dist_p = diff_p.norm(dim=-1)
    dist_g = diff_g.norm(dim=-1)

    thr = dist_g * tau_ratio
    mask = (dist_p < thr).float()

    invd = 1.0 / dist_p.clamp_min(eps)
    loss = (invd * mask).sum(dim=(1, 2)) / 8.0
    return loss


# ---------- master aggregation ----------
def cuboid_kp_losses(outputs, targets, kp_match_results,
                     lambda_kp=4.0, lambda_bbox_kp=1.0,
                     lambda_edge=0.5,
                     lambda_face=0.3, lambda_rep=0.05,
                     use_huber=True,
                     coord_scale=1.0):

    """Keypoint-related losses in image-normalized coordinates.

    outputs: dict with
      pred_boxes: (B,Q,4), pred_kp: (B,Q,8,2)   in [0,1] image coords
    kp_match_results: list len B with dicts from hungarian_match_kp()
    targets: list len B with dicts: boxes (T,4) in same coord system

    coord_scale: optional scale factor to roughly match pixel-domain L1.
    """
    device = outputs["pred_kp"].device
    loss_kp_all = []
    loss_bboxkp_all = []
    loss_edge_all = []
    loss_face_all = []
    loss_rep_all = []

    for b, mr in enumerate(kp_match_results):
        pred_idx = mr["pred_idx"]
        tgt_idx = mr["tgt_idx"]
        if pred_idx.numel() == 0:
            continue

        Pk = outputs["pred_kp"][b, pred_idx]              # (M,8,2)
        Pb = outputs["pred_boxes"][b, pred_idx]           # (M,4)
        Tk = mr["gt_kp_perm"]                             # (M,8,2), same coords
        Tb = targets[b]["boxes"][tgt_idx].to(device)      # (M,4), same coords

        # Core regression (image-normalized coords + scale factor)
        l_kp = loss_kp_core(Pk, Tk, use_huber=use_huber, delta=0.05,
                             coord_scale=coord_scale)                # (M,)
        # BBox consistency
        l_bboxkp = loss_bbox_from_kp_consistency(Pb, Pk)             # (M,)

        # Structure regularizers in image-normalized space
        l_edge = loss_edges(Pk, Tk)                                  # (M,)
        l_face = loss_faces_convexity(Pk, Tk)                        # (M,)
        l_rep  = loss_repulsion(Pk, Tk)                                  # (M,)

        loss_kp_all.append(l_kp)
        loss_bboxkp_all.append(l_bboxkp)
        loss_edge_all.append(l_edge)
        loss_face_all.append(l_face)
        loss_rep_all.append(l_rep)

    def _cat_mean(xs):
        if len(xs) == 0:
            return torch.tensor(0.0, device=device)
        return torch.cat(xs).mean()

    losses = {
        "loss_kp":      _cat_mean(loss_kp_all),
        "loss_bbox_kp": _cat_mean(loss_bboxkp_all),
        "loss_edge":    _cat_mean(loss_edge_all),
        "loss_face":    _cat_mean(loss_face_all),
        "loss_rep":     _cat_mean(loss_rep_all),
    }

    total = (lambda_kp      * losses["loss_kp"]
           + lambda_bbox_kp * losses["loss_bbox_kp"]
           + lambda_edge    * losses["loss_edge"]
           + lambda_face    * losses["loss_face"]
           + lambda_rep     * losses["loss_rep"])

    losses["loss_kp_total"] = total
    return losses

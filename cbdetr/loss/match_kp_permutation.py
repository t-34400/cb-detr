# ------------------------------------------------------------------------
# Cuboid-DETR
# Copyright (c) 2025 t-34400
# Licensed under the Apache License, Version 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------

import itertools
import torch

def _cube_24_perm_tensor(device=None, dtype=torch.long):
    def perm_parity(p):
        inv = 0
        for i in range(3):
            for j in range(i+1, 3):
                inv += (p[i] > p[j])
        return 1 if (inv % 2 == 0) else -1

    vs = torch.tensor([
        [-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1],
        [-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1],
    ], dtype=torch.float32, device=device)

    eye = torch.eye(3, device=device)
    mats = []
    for perm in itertools.permutations(range(3)):
        P = eye[:, list(perm)]
        psgn = perm_parity(perm)  # +1 or -1
        for s0 in (1, -1):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    # det = parity * s0 * s1 * s2
                    if psgn * s0 * s1 * s2 == 1:
                        D = torch.diag(torch.tensor([s0, s1, s2], dtype=torch.float32, device=device))
                        M = P @ D
                        mats.append(M)
    R = torch.stack(mats, dim=0)  # (24,3,3)

    perms = []
    for k in range(R.size(0)):
        v_rot = (vs @ R[k].T)  # (8,3)
        perm = []
        for i in range(8):
            diffs = (vs - v_rot[i]).abs().sum(dim=1)
            j = torch.argmin(diffs).item()
            perm.append(j)
        perms.append(perm)

    perm_tensor = torch.tensor(perms, dtype=dtype, device=device)  # (24,8)
    perm_tensor = torch.unique(perm_tensor, dim=0)
    assert perm_tensor.shape[0] == 24, f"Expected 24 perms, got {perm_tensor.shape[0]}"
    return perm_tensor


def _normalize_kp_by_box(kp, box, eps=1e-6):
    """
    Normalize keypoints by box: (x - cx)/w, (y - cy)/h.
    kp:  (..., 8, 2)
    box: (..., 4) in (cx, cy, w, h)
    returns: (..., 8, 2)
    """
    cx, cy, w, h = box.unbind(-1)
    w = w.clamp_min(eps)
    h = h.clamp_min(eps)
    x = (kp[..., 0] - cx[..., None]) / w[..., None]
    y = (kp[..., 1] - cy[..., None]) / h[..., None]
    return torch.stack([x, y], dim=-1)


@torch.no_grad()
def match_kp_permutation(outputs, targets, bbox_indices):
    """
    Use 24-way (cube rotation group) permutation-min loss per matched pair (from bbox assignment).

    outputs: {
        "pred_logits": [B, Q, 2],      # [object, no-object]
        "pred_boxes":  [B, Q, 4],      # (cx, cy, w, h) in image coords
        "pred_kp":     [B, Q, 8, 2],   # keypoints in image coords
    }
    targets: list of len B, each = {
        "boxes": [T, 4],               # (cx, cy, w, h) in image coords
        "kp":    [T, 8, 2],            # keypoints in image coords
    }
    bbox_indices: list of len B, each is (pred_idx, tgt_idx) from hungarian_match_bbox()

    Returns:
        result: list of len B, each is a dict:
            {
              "pred_idx":      LongTensor [M],
              "tgt_idx":       LongTensor [M],
              "perm_indices":  LongTensor [M],     # which of the 24 was selected
              "gt_kp_perm":    FloatTensor [M,8,2],# permuted GT keypoints (original coords)
              "costs":         FloatTensor [M],    # minimal L1 cost (normalized space)
            }
    """
    device = outputs["pred_kp"].device
    perm24 = _cube_24_perm_tensor(device=device)  # (24,8)

    B, Q = outputs["pred_kp"].shape[:2]
    pred_kp = outputs["pred_kp"]
    pred_boxes = outputs["pred_boxes"]

    results = []

    for b in range(B):
        pred_idx, tgt_idx = bbox_indices[b]
        if pred_idx.numel() == 0:
            results.append({
                "pred_idx": pred_idx,
                "tgt_idx": tgt_idx,
                "perm_indices": torch.empty(0, dtype=torch.long, device=device),
                "gt_kp_perm": torch.empty(0, 8, 2, dtype=torch.float32, device=device),
                "costs": torch.empty(0, dtype=torch.float32, device=device),
            })
            continue

        P = pred_kp[b, pred_idx]                 # (M,8,2), image coords
        Bp = pred_boxes[b, pred_idx]             # (M,4)
        T = targets[b]["kp"][tgt_idx].to(device) # (M,8,2), image coords
        Bt = targets[b]["boxes"][tgt_idx].to(device)  # (M,4)

        M = P.shape[0]

        Pn = _normalize_kp_by_box(P, Bt)  # (M,8,2)
        Tn = _normalize_kp_by_box(T, Bt)  # (M,8,2)

        Tn_perm = Tn[:, None, :, :].expand(M, perm24.size(0), 8, 2).clone()
        Tn_perm = Tn_perm.gather(dim=2, index=perm24[None, :, :, None].expand(M, -1, -1, 2))

        # Compute L1 cost per permutation: (M,24)
        diff = (Pn[:, None, :, :] - Tn_perm).abs().sum(dim=(-1, -2))  # sum over (8,2)
        costs, best_k = diff.min(dim=1)  # (M,), (M,)

        # Build permuted GT keypoints in ORIGINAL coords for downstream losses
        T_perm = T[:, None, :, :].expand(M, perm24.size(0), 8, 2).clone()
        T_perm = T_perm.gather(dim=2, index=perm24[None, :, :, None].expand(M, -1, -1, 2))

        # select best_k along dim=1
        gather_idx = best_k.view(M, 1, 1, 1).expand(M, 1, 8, 2)
        gt_kp_perm = T_perm.gather(dim=1, index=gather_idx).squeeze(1).contiguous()  # (M,8,2)

        results.append({
            "pred_idx": pred_idx.to(device),
            "tgt_idx": tgt_idx.to(device),
            "perm_indices": best_k.to(device),        # [M]
            "gt_kp_perm": gt_kp_perm.to(device),      # [M,8,2] original coords
            "costs": costs.to(device),                # [M] normalized L1
        })

    return results
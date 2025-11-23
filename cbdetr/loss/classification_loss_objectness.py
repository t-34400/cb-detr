# ------------------------------------------------------------------------
# Cuboid-DETR
# Copyright (c) 2025 t-34400
# Licensed under the Apache License, Version 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F

def classification_loss_objectness(outputs, bbox_indices, no_object_weight=0.1, use_focal=False, gamma=2.0):
    """
    Compute object/no-object classification loss for all queries.
    - Matched queries -> target = 1 (object)
    - Unmatched queries -> target = 0 (no-object)
    pred_logits: (B, Q, 2) in outputs
    """
    logits = outputs["pred_logits"]  # (B, Q, 2)
    B, Q, _ = logits.shape
    device = logits.device

    # Build targets: 1 for matched, 0 otherwise
    tgt = torch.zeros((B, Q), dtype=torch.long, device=device)
    for b, (pi, _) in enumerate(bbox_indices):
        if pi.numel() > 0:
            tgt[b, pi.to(device)] = 1

    if use_focal:
        # Binary focal loss on logits (convert to prob of class=1)
        p = logits.softmax(-1)[..., 1]  # (B, Q)
        pt = torch.where(tgt==1, p, 1-p)
        alpha = torch.where(tgt==1, torch.tensor(1.0, device=device), torch.tensor(no_object_weight, device=device))
        loss = -alpha * ((1-pt)**gamma) * pt.log().clamp_min(-20)
        return loss.mean()
    else:
        weight = torch.tensor([no_object_weight, 1.0], device=device)
        loss = F.cross_entropy(
            logits.flatten(0,1), 
            tgt.flatten(0,1),
            weight=weight
        )
        return loss
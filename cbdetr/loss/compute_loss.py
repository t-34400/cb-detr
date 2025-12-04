# ------------------------------------------------------------------------
# Cuboid-DETR
# Copyright (c) 2025 t-34400
# Licensed under the Apache License, Version 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from typing import Dict, Any

from cbdetr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou_matched
from cbdetr.loss.hungarian_match_bbox import hungarian_match_bbox
from cbdetr.loss.match_kp_permutation import match_kp_permutation
from cbdetr.loss.classification_loss_objectness import classification_loss_objectness
from cbdetr.loss.cuboid_kp_losses import cuboid_kp_losses

# ------------------------------------------------------------
# Assumptions:
#  - You have the following helpers defined in the same module:
#    - hungarian_match_bbox(outputs, targets, cost_class, cost_bbox, cost_giou)
#    - hungarian_match_kp(outputs, targets, bbox_indices)  # 24-way permutation inside
#    - cuboid_kp_losses(outputs, targets, kp_match_results, ...)  # KP + structure terms
#    - classification_loss_objectness(outputs, bbox_indices, no_object_weight, use_focal, gamma)
#    - box_cxcywh_to_xyxy, generalized_box_iou  (already imported in your code)
#  - outputs:
#       {
#         "pred_logits": (B,Q,2),
#         "pred_boxes":  (B,Q,4),
#         "pred_kp":     (B,Q,8,2),
#         # optional DETR-style auxiliary outputs:
#         "aux_outputs": list of dicts with the same keys (pred_logits/pred_boxes/pred_kp)
#       }
#  - targets: list len B, each = {
#         "boxes": (T,4),
#         "kp":    (T,8,2),
#       }
# ------------------------------------------------------------

def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Any,
    group_detr: int = 1,
    # Matcher costs (for bbox-level Hungarian)
    cost_class: float = 1.0,
    cost_bbox: float  = 5.0,
    cost_giou: float  = 2.0,
    # Classification loss config
    no_object_weight: float = 0.1,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    # BBox loss config
    lambda_bbox_l1  = 5.0,
    lambda_bbox_giou= 2.0,
    # KP/structure loss weights (passed through to cuboid_kp_losses)
    lambda_kp: float = 4.0,
    lambda_bbox_kp: float = 1.0,
    lambda_edge: float = 0.5,
    lambda_face: float = 0.3,
    lambda_rep: float = 0.05,
    lambda_kp_coarse: float = 1.0,
    lambda_edge_coarse: float = 0.2,
    use_huber: bool = True,
    # DETR-style: also train on intermediate decoder layers if present
    use_aux_losses: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute the total loss for one forward pass.
    Returns a dict of loss components and 'loss_total'.
    """

    device = outputs["pred_boxes"].device

    # -------- 1) Hungarian matching by bbox (instance-level, MAIN layer) --------
    bbox_indices = hungarian_match_bbox(
        outputs, targets,
        cost_class=cost_class,
        cost_bbox=cost_bbox,
        cost_giou=cost_giou,
        group_detr=group_detr
    )

    # -------- 2) Classification loss (object vs no-object) for all queries --------
    loss_cls = classification_loss_objectness(
        outputs=outputs,
        bbox_indices=bbox_indices,
        no_object_weight=no_object_weight,
        use_focal=use_focal,
        gamma=focal_gamma
    )

    # -------- 3) BBox regression losses (for matched queries only, MAIN layer) --------
    pred_boxes_main, tgt_boxes_main = [], []
    for b, (pi, ti) in enumerate(bbox_indices):
        if pi.numel() == 0:
            continue
        pred_boxes_main.append(outputs["pred_boxes"][b, pi])          # (M,4)
        tgt_boxes_main.append(targets[b]["boxes"][ti].to(device))     # (M,4)

    if len(pred_boxes_main) > 0:
        pred_boxes_main = torch.cat(pred_boxes_main, dim=0)
        tgt_boxes_main  = torch.cat(tgt_boxes_main,  dim=0)

        loss_bbox_l1 = F.l1_loss(pred_boxes_main, tgt_boxes_main, reduction='none').sum(-1).mean()

        giou = generalized_box_iou_matched(
            box_cxcywh_to_xyxy(pred_boxes_main),
            box_cxcywh_to_xyxy(tgt_boxes_main)
        )
        loss_giou = (1.0 - giou).mean()
    else:
        loss_bbox_l1 = torch.tensor(0.0, device=device)
        loss_giou    = torch.tensor(0.0, device=device)

    # -------- 4) KP matching (24-way) and KP/structure losses (MAIN layer) --------
    kp_match_results = match_kp_permutation(outputs, targets, bbox_indices)

    kp_losses_main = cuboid_kp_losses(
        outputs, targets, kp_match_results,
        lambda_kp=lambda_kp,
        lambda_bbox_kp=lambda_bbox_kp,
        lambda_edge=lambda_edge,
        lambda_face=lambda_face,
        lambda_rep=lambda_rep,
        lambda_kp_coarse=lambda_kp_coarse,
        lambda_edge_coarse=lambda_edge_coarse,
        use_huber=use_huber
    )

    # -------- 5) Aggregate losses (MAIN layer) --------
    losses = {
        "loss_cls": loss_cls,
        "loss_bbox_l1": lambda_bbox_l1   * loss_bbox_l1,
        "loss_bbox_giou": lambda_bbox_giou * loss_giou,
        "loss_kp": kp_losses_main["loss_kp"] * 1.0,
        "loss_bbox_kp": kp_losses_main["loss_bbox_kp"] * 1.0,
        "loss_edge": kp_losses_main["loss_edge"] * 1.0,
        "loss_face": kp_losses_main["loss_face"] * 1.0,
        "loss_rep":  kp_losses_main["loss_rep"] * 1.0,
        "loss_kp_total": kp_losses_main["loss_kp_total"] * 1.0
    }

    loss_total = (
         losses["loss_cls"]
        + losses["loss_bbox_l1"] + losses["loss_bbox_giou"]
        + losses["loss_kp_total"]
    )
    losses["loss_total"] = loss_total

    # -------- 6) Auxiliary losses (for intermediate decoder layers) --------
    if use_aux_losses and "aux_outputs" in outputs and outputs["aux_outputs"] is not None:
        aux_losses_total = []

        for i, aux in enumerate(outputs["aux_outputs"]):
            # ---- 6.1 cls for aux (reuse main bbox_indices) ----
            aux_loss_cls = classification_loss_objectness(
                outputs=aux,
                bbox_indices=bbox_indices,
                no_object_weight=no_object_weight,
                use_focal=use_focal,
                gamma=focal_gamma
            )

            # ---- 6.2 bbox regression for aux (same matching) ----
            pred_boxes_aux, tgt_boxes_aux = [], []
            for b, (pi, ti) in enumerate(bbox_indices):
                if pi.numel() == 0:
                    continue
                pred_boxes_aux.append(aux["pred_boxes"][b, pi])
                tgt_boxes_aux.append(targets[b]["boxes"][ti].to(device))

            if len(pred_boxes_aux) > 0:
                pred_boxes_aux = torch.cat(pred_boxes_aux, dim=0)
                tgt_boxes_aux  = torch.cat(tgt_boxes_aux,  dim=0)

                aux_l1 = F.l1_loss(pred_boxes_aux, tgt_boxes_aux, reduction='none').sum(-1).mean()
                aux_giou = (1.0 - generalized_box_iou_matched(
                    box_cxcywh_to_xyxy(pred_boxes_aux),
                    box_cxcywh_to_xyxy(tgt_boxes_aux)
                )).mean()
            else:
                aux_l1 = torch.tensor(0.0, device=device)
                aux_giou = torch.tensor(0.0, device=device)

            # ---- 6.3 kp + structure losses for aux (reuse kp_match_results) ----
            aux_kp_losses = cuboid_kp_losses(
                aux, targets, kp_match_results,
                lambda_kp=lambda_kp,
                lambda_bbox_kp=lambda_bbox_kp,
                lambda_edge=lambda_edge,
                lambda_face=lambda_face,
                lambda_rep=lambda_rep,
                lambda_kp_coarse=lambda_kp_coarse,
                lambda_edge_coarse=lambda_edge_coarse,
                use_huber=use_huber
            )

            aux_total = (
                aux_loss_cls
                + lambda_bbox_l1 * aux_l1
                + lambda_bbox_giou * aux_giou
                + aux_kp_losses["loss_kp_total"]
            )
            aux_losses_total.append(aux_total)

            losses[f"loss_aux_{i}_cls"] = aux_loss_cls
            losses[f"loss_aux_{i}_bbox_l1"] = lambda_bbox_l1 * aux_l1
            losses[f"loss_aux_{i}_bbox_giou"] = lambda_bbox_giou * aux_giou
            losses[f"loss_aux_{i}_kp_total"] = aux_kp_losses["loss_kp_total"]

        if len(aux_losses_total) > 0:
            losses["aux_loss"] = torch.stack(aux_losses_total).mean()
            losses["loss_total"] = losses["loss_total"] + losses["aux_loss"]

    return losses, bbox_indices, kp_match_results

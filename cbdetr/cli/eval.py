# ------------------------------------------------------------------------
# Cuboid-DETR
# Copyright (c) 2025 t-34400
# Licensed under the Apache License, Version 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------

import os
import json
import time
import argparse
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cbdetr.util.batch_utils import to_device_batch, build_targets_from_batch
from cbdetr.util.ckpt_utils import load_checkpoint

from cbdetr.dataset.dataset import CuboidDataset, cuboid_collate_fn
from cbdetr.models.build_model import (
    build_model, 
    add_model_transformer_args, 
    parse_model_transformer_args, 
)
from cbdetr.loss.compute_loss import compute_total_loss

# ----------------------------
# Evaluation
# ----------------------------
@torch.no_grad()
def eval_avg_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    matcher_costs: Dict[str, float],
    cls_cfg: Dict[str, Any],
    kp_cfg: Dict[str, float],
    use_amp: bool = True,
) -> float:
    model.eval()
    losses = []
    for batch in loader:
        batch = to_device_batch(batch, device)
        targets = build_targets_from_batch(batch, device=device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(batch["images"])
            ldict, _, _ = compute_total_loss(
                outputs, targets,
                cost_class=matcher_costs["cost_class"],
                cost_bbox=matcher_costs["cost_bbox"],
                cost_giou=matcher_costs["cost_giou"],
                no_object_weight=cls_cfg["no_object_weight"],
                use_focal=cls_cfg["use_focal"],
                focal_gamma=cls_cfg["focal_gamma"],
                lambda_kp=kp_cfg["lambda_kp"],
                lambda_bbox_kp=kp_cfg["lambda_bbox_kp"],
                lambda_edge=kp_cfg["lambda_edge"],
                lambda_face=kp_cfg["lambda_face"],
                lambda_rep=kp_cfg["lambda_rep"],
                use_huber=True,
                use_aux_losses=False,
            )
        losses.append(ldict["loss_total"].item())
    return float(np.mean(losses)) if len(losses) else 0.0


@torch.no_grad()
def dump_predictions_if_needed(
    *,
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    out_list: List[Dict[str, Any]],
    score_thresh: float,
    topk: int,
):
    """
    Make a simple JSON-friendly prediction list:
    - For each image: take top-k queries by objectness (max class prob excluding 'no-object')
    - Save boxes (xywh in pixels), keypoints (pixels), score, class_id
    This is a convenience dumper for quick inspection; not COCO-format.
    """
    if ("pred_logits" not in outputs) or ("pred_boxes" not in outputs):
        return  # model doesn't expose DETR-like heads

    B, _, H, W = batch["images"].shape
    logits = outputs["pred_logits"]          # (B, Q, C)
    pred_boxes = outputs["pred_boxes"]       # (B, Q, 4) in cx,cy,w,h (0..1)
    pred_kp = outputs.get("pred_kp", None)   # (B, Q, 8, 2) in (0..1), optional

    probs = logits.softmax(-1)               # (B,Q,C)
    # Assume last class is "no-object"
    obj_prob, cls_id = probs[..., :-1].max(-1)  # (B,Q), (B,Q)

    for b in range(B):
        # rank by score
        scores = obj_prob[b]  # (Q,)
        keep = torch.nonzero(scores >= score_thresh, as_tuple=False).squeeze(-1)
        if keep.numel() == 0:
            keep = torch.arange(scores.numel(), device=scores.device)
        scores_kept = scores[keep]
        _, order = torch.sort(scores_kept, descending=True)
        sel = keep[order][:topk]

        # convert boxes to pixel xywh
        cxcywh = pred_boxes[b, sel]  # (K,4)
        xywh = cxcywh.clone()
        xywh[:, 0] = (cxcywh[:, 0] - cxcywh[:, 2] * 0.5) * W
        xywh[:, 1] = (cxcywh[:, 1] - cxcywh[:, 3] * 0.5) * H
        xywh[:, 2] = cxcywh[:, 2] * W
        xywh[:, 3] = cxcywh[:, 3] * H

        # keypoints (optional) -> pixels
        kp_pix = None
        if pred_kp is not None:
            kp = pred_kp[b, sel]  # (K,8,2) in 0..1
            kp_pix = kp.clone()
            kp_pix[..., 0] = kp[..., 0] * W
            kp_pix[..., 1] = kp[..., 1] * H

        # image identifier (best effort)
        img_id = None
        # try common meta keys
        for k in ["paths", "img_paths", "image_paths", "ids", "image_ids"]:
            if k in batch and len(batch[k]) > 0:
                candidate = batch[k][b]
                img_id = candidate if isinstance(candidate, (str, int)) else str(candidate)
                break

        recs = []
        for i, q in enumerate(sel.tolist()):
            rec = {
                "score": float(scores[q].item()),
                "class_id": int(cls_id[b, q].item()),
                "bbox_xywh": [float(v) for v in xywh[i].tolist()],
            }
            if kp_pix is not None:
                rec["kp_xy"] = [[float(x), float(y)] for (x, y) in kp_pix[i].tolist()]
            recs.append(rec)

        out_list.append({
            "image_id": img_id if img_id is not None else f"batch{time.time_ns()}_{b}",
            "predictions": recs
        })


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_root", type=str, required=True, help="Root dir with .h5 validation data")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (best.pt or last.pt)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--out_dir", type=str, default="./runs/cuboid_eval")
    parser.add_argument("--dump_json", type=str, default=None, help="If set, dump predictions to this JSON path")
    parser.add_argument("--score_thresh", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=50)
    # loss configs (mirror training defaults)
    parser.add_argument("--cost_class", type=float, default=2.0)
    parser.add_argument("--cost_bbox", type=float, default=5.0)
    parser.add_argument("--cost_giou", type=float, default=2.0)
    parser.add_argument("--no_object_weight", type=float, default=1.0)
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--lambda_kp", type=float, default=4.0)
    parser.add_argument("--lambda_bbox_kp", type=float, default=1.0)
    parser.add_argument("--lambda_edge", type=float, default=0.5)
    parser.add_argument("--lambda_face", type=float, default=0.3)
    parser.add_argument("--lambda_rep", type=float, default=0.05)
    add_model_transformer_args(parser=parser)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    margs, targs = parse_model_transformer_args(args, device=device)
    model = build_model(margs, targs).to(device)

    # Dataset / loader
    size_multiple = margs.patch_size * margs.num_windows
    val_ds = CuboidDataset(root_dir=args.val_root, size_multiple=size_multiple, filter="bbox_center")
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=cuboid_collate_fn,
    )

    # Load checkpoint
    resume_epoch = load_checkpoint(args.ckpt, model)
    if resume_epoch is not None:
        print(f"Loaded checkpoint from epoch {resume_epoch}")
    model.eval()

    # Loss configs
    matcher_costs = dict(cost_class=args.cost_class, cost_bbox=args.cost_bbox, cost_giou=args.cost_giou)
    cls_cfg = dict(no_object_weight=args.no_object_weight, use_focal=args.use_focal, focal_gamma=args.focal_gamma)
    kp_cfg = dict(
        lambda_kp=args.lambda_kp,
        lambda_bbox_kp=args.lambda_bbox_kp,
        lambda_edge=args.lambda_edge,
        lambda_face=args.lambda_face,
        lambda_rep=args.lambda_rep,
    )

    # Evaluate average loss
    t0 = time.time()
    avg_loss = eval_avg_loss(
        model, val_loader, device,
        matcher_costs=matcher_costs, cls_cfg=cls_cfg, kp_cfg=kp_cfg, use_amp=args.amp
    )
    dt = time.time() - t0
    imgs = len(val_loader.dataset)
    print(f"[Eval] avg_loss={avg_loss:.4f}  imgs={imgs}  time={dt:.1f}s  img/s={imgs/max(dt,1e-6):.1f}")

    # Optional: dump predictions for quick eyeballing
    if args.dump_json:
        preds: List[Dict[str, Any]] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device_batch(batch, device)
                with torch.amp.autocast('cuda', enabled=args.amp):
                    outputs = model(batch["images"])
                dump_predictions_if_needed(
                    outputs=outputs, batch=batch, out_list=preds,
                    score_thresh=args.score_thresh, topk=args.topk
                )
        json_path = args.dump_json
        with open(json_path, "w") as f:
            json.dump({"avg_loss": avg_loss, "predictions": preds}, f, indent=2)
        print(f"[Eval] wrote predictions to: {json_path}")


if __name__ == "__main__":
    main()

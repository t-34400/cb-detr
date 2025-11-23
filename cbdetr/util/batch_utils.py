# ----------------------------
# Batch utilities
# ----------------------------
from typing import Any, Dict, List

import torch


def to_device_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move only tensors that must live on device."""
    batch["images"] = batch["images"].to(device, non_blocking=True)
    return batch


def build_targets_from_batch(batch: Dict[str, Any], device='cuda') -> List[Dict[str, torch.Tensor]]:
    """
    Convert dataset batch to DETR-style targets:
      - boxes: (T,4) in (cx,cy,w,h) normalized to [0,1]
      - kp   : (T,8,2) normalized to [0,1]
      - optional visible_mask: (T,8) -> float {0,1}
    """
    B, _, H, W = batch["images"].shape
    targets: List[Dict[str, torch.Tensor]] = []

    for b in range(B):
        ann = batch["anns"][b]
        boxes = ann.get("bbox", None)
        kps = ann.get("uv", None)

        if boxes is None or kps is None:
            targets.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "kp": torch.zeros((0, 8, 2), dtype=torch.float32),
            })
            continue

        boxes = boxes.to(device).float()   # (T,4)
        kps = kps.to(device).float()       # (T,8,2)

        boxes[:, 0] /= W; boxes[:, 1] /= H
        boxes[:, 2] /= W; boxes[:, 3] /= H

        x, y, w_box, h_box = boxes.unbind(-1)
        cx = x + 0.5 * w_box
        cy = y + 0.5 * h_box
        boxes_cxcywh = torch.stack([cx, cy, w_box, h_box], dim=-1)

        tgt: Dict[str, torch.Tensor] = {
            "boxes": boxes_cxcywh,
            "kp": kps,
        }
        vis = ann.get("visible_vertices", None)
        if vis is not None:
            tgt["visible_mask"] = vis.to(device=device, dtype=torch.bool).float()

        targets.append(tgt)

    return targets
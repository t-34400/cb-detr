# ------------------------------------------------------------------------
# Cuboid-DETR
# Copyright (c) 2025 t-34400
# Licensed under the Apache License, Version 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------

import os
import glob
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cbdetr.models.build_model import (
    build_model,
    add_model_transformer_args,
    parse_model_transformer_args,
)
from cbdetr.dataset.dataset import CuboidDataset, cuboid_collate_fn


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_path(p: str) -> bool:
    return os.path.splitext(p.lower())[1] in IMG_EXTS


def list_images(path: str) -> List[str]:
    if os.path.isdir(path):
        files: List[str] = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(path, f"*{ext}")))
        return sorted(files)
    elif os.path.isfile(path) and is_image_path(path):
        return [path]
    return []


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        paths: List[str],
        img_max_size: int = 1333,
        img_min_size: int = 480,
        size_multiple: int = 32,
        normalize: bool = False,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.paths = paths
        self.img_max = img_max_size
        self.img_min = img_min_size
        self.size_multiple = size_multiple
        self.normalize = normalize
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __len__(self) -> int:
        return len(self.paths)

    def _resize_and_align(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if self.img_min > 0:
            scale = min(self.img_max / max(h, w), self.img_min / min(h, w))
        else:
            scale = min(1.0, self.img_max / max(h, w))

        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        new_w = max(self.size_multiple, int(np.floor(new_w / self.size_multiple)) * self.size_multiple)
        new_h = max(self.size_multiple, int(np.floor(new_h / self.size_multiple)) * self.size_multiple)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = self._resize_and_align(bgr)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.normalize:
            rgb = (rgb - self.mean) / self.std
        chw = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        return {
            "images": chw,
            "paths": [path],
            "orig_hw": (bgr.shape[0], bgr.shape[1]),
            "proc_hw": (img.shape[0], img.shape[1]),
        }


def img_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    imgs = torch.stack([b["images"] for b in batch], dim=0)
    paths = [b["paths"][0] for b in batch]
    orig_hws = [b["orig_hw"] for b in batch]
    proc_hws = [b["proc_hw"] for b in batch]
    return {
        "images": imgs,
        "paths": paths,
        "orig_hw": orig_hws,
        "proc_hw": proc_hws,
        "anns": [{} for _ in batch],
    }


def load_checkpoint_model(path: str, model: nn.Module) -> Optional[int]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    return ckpt.get("epoch", None)


def select_top_queries(
    outputs: Dict[str, torch.Tensor],
    score_thresh: float,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = outputs["pred_logits"]
    probs = logits.softmax(-1)

    obj_prob, cls_id = probs[..., 1:].max(-1)
    cls_id = cls_id + 1

    B, Q = obj_prob.shape
    K = min(topk, Q)

    sel_idx_list: List[torch.Tensor] = []
    sel_scores_list: List[torch.Tensor] = []
    sel_cls_list: List[torch.Tensor] = []

    for b in range(B):
        scores = obj_prob[b]
        keep = torch.nonzero(scores >= score_thresh, as_tuple=False).squeeze(-1)

        if keep.numel() == 0:
            keep = torch.empty(0, dtype=torch.long, device=scores.device)

        kept_scores = scores[keep]
        order = torch.argsort(kept_scores, descending=True)
        sel = keep[order][:K]

        sel_idx_list.append(sel)
        sel_scores_list.append(scores[sel])
        sel_cls_list.append(cls_id[b, sel])

    def pad_to_k(x: torch.Tensor, K: int) -> torch.Tensor:
        if x.numel() == K:
            return x
        pad = torch.full((K - x.numel(),), -1, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    sel_idx = torch.stack([pad_to_k(si, K) for si in sel_idx_list], dim=0)
    sel_scores = torch.stack([pad_to_k(ss, K) for ss in sel_scores_list], dim=0)
    sel_cls = torch.stack([pad_to_k(sc, K) for sc in sel_cls_list], dim=0)

    return sel_idx, sel_scores, sel_cls


def boxes_to_xyxy(cxcywh: torch.Tensor, W: int, H: int) -> np.ndarray:
    xyxy = torch.zeros_like(cxcywh)
    xyxy[..., 0] = (cxcywh[..., 0] - cxcywh[..., 2] * 0.5) * W
    xyxy[..., 1] = (cxcywh[..., 1] - cxcywh[..., 3] * 0.5) * H
    xyxy[..., 2] = (cxcywh[..., 0] + cxcywh[..., 2] * 0.5) * W
    xyxy[..., 3] = (cxcywh[..., 1] + cxcywh[..., 3] * 0.5) * H
    return xyxy.cpu().numpy()


def kps_to_pixels(kps01: torch.Tensor, W: int, H: int) -> np.ndarray:
    kp = kps01.clone()
    kp[..., 0] *= W
    kp[..., 1] *= H
    return kp.cpu().numpy()


CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def tensor_to_bgr(chw: torch.Tensor) -> np.ndarray:
    arr = chw.detach().cpu().permute(1, 2, 0).float().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def draw_predictions(
    bgr: np.ndarray,
    boxes_xyxy: Optional[np.ndarray],
    kps_xy: Optional[np.ndarray],
    scores: Optional[np.ndarray],
    cls_ids: Optional[np.ndarray],
    kp_radius: int = 3,
    score_thresh: float = 0.0,
) -> np.ndarray:
    out = bgr.copy()
    H, W = out.shape[:2]

    if boxes_xyxy is None or scores is None or cls_ids is None:
        return out

    K = boxes_xyxy.shape[0]
    for i in range(K):
        if scores[i] < score_thresh:
            continue

        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"{int(cls_ids[i])}:{scores[i]:.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if kps_xy is not None:
            pts = kps_xy[i].astype(int)
            for p in pts:
                cv2.circle(out, (p[0], p[1]), kp_radius, (0, 0, 255), -1)
            if pts.shape[0] >= 8:
                for (a, b) in CUBOID_EDGES:
                    pa, pb = pts[a], pts[b]
                    cv2.line(out, (pa[0], pa[1]), (pb[0], pb[1]), (255, 0, 0), 2)

    return out


def build_deformable_heatmap_for_query(
    cross_attn: nn.Module,
    b: int,
    q: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    loc = getattr(cross_attn, "last_sampling_locations", None)
    w = getattr(cross_attn, "last_attention_weights", None)
    shapes = getattr(cross_attn, "last_spatial_shapes", None)

    if loc is None or w is None or shapes is None:
        raise RuntimeError("Deformable attention cache is empty. Ensure record_attn is enabled and a forward pass was run.")

    loc = loc[b, q]
    w = w[b, q]
    device = loc.device

    n_heads, n_levels, n_points, _ = loc.shape
    heat_all = torch.zeros((n_levels, out_h, out_w), device=device)

    for lvl in range(n_levels):
        H_l, W_l = shapes[lvl].tolist()
        loc_lvl = loc[:, lvl]
        w_lvl = w[:, lvl * n_points:(lvl + 1) * n_points]

        xs = loc_lvl[..., 0] * W_l
        ys = loc_lvl[..., 1] * H_l
        xs = xs.round().long().clamp(0, W_l - 1)
        ys = ys.round().long().clamp(0, H_l - 1)

        heat_lvl = torch.zeros((H_l, W_l), device=device)
        for h in range(n_heads):
            for p in range(n_points):
                heat_lvl[ys[h, p], xs[h, p]] += w_lvl[h, p]

        heat_lvl = heat_lvl.unsqueeze(0).unsqueeze(0)
        heat_up = torch.nn.functional.interpolate(
            heat_lvl, size=(out_h, out_w), mode="bilinear", align_corners=False
        )[0, 0]
        heat_all[lvl] = heat_up

    heat = heat_all.sum(0)
    heat = heat / (heat.max() + 1e-6)
    return heat.detach().cpu().numpy()


def overlay_heatmap_on_bgr(
    bgr: np.ndarray,
    heat: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    heat_norm = np.clip(heat, 0.0, 1.0)
    heat_u8 = (heat_norm * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.resize(heat_color, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    blended = cv2.addWeighted(bgr, 1.0 - alpha, heat_color, alpha, 0)
    return blended


def enable_last_decoder_attn_record(model: nn.Module, flag: bool = True) -> None:
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        return
    decoder = getattr(transformer, "decoder", None)
    if decoder is None or not hasattr(decoder, "layers") or len(decoder.layers) == 0:
        return
    last_layer = decoder.layers[-1]
    cross_attn = getattr(last_layer, "cross_attn", None)
    if cross_attn is None:
        return
    if hasattr(cross_attn, "record_attn"):
        cross_attn.record_attn = flag


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    out_dir: str,
    score_thresh: float,
    topk: int,
    save_json: str,
    from_h5: bool,
    vis_attn: bool = False,
    attn_dir: Optional[str] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    if vis_attn and attn_dir is not None:
        os.makedirs(attn_dir, exist_ok=True)

    all_preds: List[Dict[str, Any]] = []
    global_idx = 0
    warned_no_attn_cache = False

    for batch in loader:
        imgs = batch["images"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(imgs)

        if ("pred_logits" not in outputs) or ("pred_boxes" not in outputs):
            raise RuntimeError("Model outputs must contain 'pred_logits' and 'pred_boxes'.")

        if vis_attn:
            transformer = getattr(model, "transformer", None)
            decoder = getattr(transformer, "decoder", None) if transformer is not None else None
            last_cross = decoder.layers[-1].cross_attn if (decoder is not None and len(decoder.layers) > 0) else None
        else:
            last_cross = None

        sel_idx, sel_scores, sel_cls = select_top_queries(outputs, score_thresh, topk)
        B, K = sel_idx.shape
        pred_boxes = outputs["pred_boxes"]
        pred_kp = outputs.get("pred_kp", None)

        for b in range(B):
            if from_h5:
                src_name = None
                if "paths" in batch and len(batch["paths"]) > b and batch["paths"][b] is not None:
                    src_name = str(batch["paths"][b])
                img_id = src_name if src_name else f"h5_{global_idx}"
                _, H, W = imgs[b].shape
                overlay_src = tensor_to_bgr(imgs[b].cpu())
                can_draw = True
            else:
                H, W = batch["proc_hw"][b]
                src_path = batch["paths"][b]
                img_id = os.path.basename(src_path)
                can_draw = True
                overlay_src = cv2.imread(src_path, cv2.IMREAD_COLOR)
                if overlay_src is None:
                    overlay_src = np.zeros((H, W, 3), dtype=np.uint8)
                else:
                    overlay_src = cv2.resize(overlay_src, (W, H), interpolation=cv2.INTER_LINEAR)

            sel = sel_idx[b]
            valid = sel >= 0

            if valid.any():
                q_idx = sel[valid]
                boxes_xyxy = boxes_to_xyxy(pred_boxes[b, q_idx], W=W, H=H)
                if pred_kp is not None:
                    kp_xy = kps_to_pixels(pred_kp[b, q_idx], W=W, H=H)
                else:
                    kp_xy = None
                scores = sel_scores[b, valid].cpu().numpy()
                cls_ids = sel_cls[b, valid].cpu().numpy().astype(np.int32)
            else:
                q_idx = torch.empty(0, dtype=torch.long, device=imgs.device)
                boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
                kp_xy = np.zeros((0, 8, 2), dtype=np.float32) if pred_kp is not None else None
                scores = np.zeros((0,), dtype=np.float32)
                cls_ids = np.zeros((0,), dtype=np.int32)

            if can_draw and overlay_src is not None:
                overlay = draw_predictions(
                    overlay_src,
                    boxes_xyxy,
                    kp_xy,
                    scores,
                    cls_ids,
                    score_thresh=score_thresh,
                )
                out_path = os.path.join(
                    out_dir,
                    f"{os.path.splitext(os.path.basename(img_id))[0]}_overlay.jpg",
                )
                cv2.imwrite(out_path, overlay)

                if vis_attn and last_cross is not None and valid.any() and attn_dir is not None:
                    if (
                        getattr(last_cross, "last_sampling_locations", None) is None
                        or getattr(last_cross, "last_attention_weights", None) is None
                        or getattr(last_cross, "last_spatial_shapes", None) is None
                    ):
                        if not warned_no_attn_cache:
                            print("Warning: attention cache is empty. Ensure MSDeformAttn.record_attn=True.")
                            warned_no_attn_cache = True
                    else:
                        base_name = os.path.splitext(os.path.basename(img_id))[0]
                        num_dets = boxes_xyxy.shape[0]
                        for i in range(num_dets):
                            if scores[i] < score_thresh:
                                continue
                            q_int = int(q_idx[i].item())
                            heat = build_deformable_heatmap_for_query(
                                last_cross,
                                b=b,
                                q=q_int,
                                out_h=H,
                                out_w=W,
                            )
                            heat_img = overlay_heatmap_on_bgr(overlay_src, heat, alpha=0.5)
                            x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
                            x1 = max(0, min(W - 1, x1))
                            x2 = max(0, min(W - 1, x2))
                            y1 = max(0, min(H - 1, y1))
                            y2 = max(0, min(H - 1, y2))

                            cv2.rectangle(heat_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            label = f"{int(cls_ids[i])}:{scores[i]:.2f}"
                            cv2.putText(
                                heat_img,
                                label,
                                (x1, max(0, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )

                            attn_path = os.path.join(
                                attn_dir,
                                f"{base_name}_q{q_int:03d}_attn.jpg",
                            )
                            cv2.imwrite(attn_path, heat_img)

            recs: List[Dict[str, Any]] = []
            for i in range(boxes_xyxy.shape[0]):
                if scores[i] < score_thresh:
                    continue
                box = boxes_xyxy[i].tolist()
                item: Dict[str, Any] = {
                    "score": float(scores[i]),
                    "class_id": int(cls_ids[i]),
                    "bbox_xyxy": [float(v) for v in box],
                }
                if kp_xy is not None:
                    item["kp_xy"] = [[float(x), float(y)] for (x, y) in kp_xy[i].tolist()]
                recs.append(item)

            all_preds.append({"image_id": str(img_id), "predictions": recs})
            global_idx += 1

    with open(save_json, "w") as f:
        json.dump(all_preds, f, indent=2)
    print(f"Wrote JSON predictions: {save_json}")
    print(f"Overlay images saved to: {out_dir}")
    if vis_attn and attn_dir is not None:
        print(f"Attention maps saved to: {attn_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (best.pt or last.pt)")
    parser.add_argument("--images", type=str, default=None, help="Image path or directory")
    parser.add_argument("--img_min_size", type=int, default=480)
    parser.add_argument("--img_max_size", type=int, default=1333)
    parser.add_argument("--normalize", action="store_true", help="Apply ImageNet mean/std normalization")
    parser.add_argument("--h5_root", type=str, default=None, help="H5 dataset root for CuboidDataset")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--score_thresh", type=float, default=0.8)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="./runs/cuboid_infer/overlays")
    parser.add_argument("--out_json", type=str, default="./runs/cuboid_infer/preds.json")

    parser.add_argument(
        "--vis_attn",
        action="store_true",
        help="Visualize decoder cross-attention for all drawn queries",
    )
    parser.add_argument(
        "--vis_attn_dir",
        type=str,
        default=None,
        help="Directory to save attention maps (default: OUT_DIR/attn)",
    )

    add_model_transformer_args(parser=parser)
    args = parser.parse_args()

    if (args.images is None) == (args.h5_root is None):
        raise ValueError("Specify exactly one of --images or --h5_root")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    margs, targs = parse_model_transformer_args(args, device=device)
    model = build_model(margs, targs).to(device)
    epoch_loaded = load_checkpoint_model(args.ckpt, model)
    if epoch_loaded is not None:
        print(f"Loaded checkpoint from epoch {epoch_loaded}")
    model.eval()

    if args.vis_attn:
        enable_last_decoder_attn_record(model, flag=True)

    size_multiple = margs.patch_size * margs.num_windows
    from_h5 = False

    if args.images is not None:
        img_paths = list_images(args.images)
        if len(img_paths) == 0:
            raise FileNotFoundError(f"No images found under: {args.images}")
        ds = ImageFolderDataset(
            img_paths,
            img_max_size=args.img_max_size,
            img_min_size=args.img_min_size,
            normalize=args.normalize,
            size_multiple=size_multiple,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=img_collate_fn,
        )
        os.makedirs(args.out_dir, exist_ok=True)
    else:
        from_h5 = True
        ds = CuboidDataset(root_dir=args.h5_root, size_multiple=size_multiple, filter="bbox_center")
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=cuboid_collate_fn,
        )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    attn_dir = None
    if args.vis_attn:
        attn_dir = args.vis_attn_dir or os.path.join(args.out_dir, "attn")

    run_inference(
        model,
        loader,
        device,
        use_amp=args.amp,
        out_dir=args.out_dir,
        score_thresh=args.score_thresh,
        topk=args.topk,
        save_json=args.out_json,
        from_h5=from_h5,
        vis_attn=args.vis_attn,
        attn_dir=attn_dir,
    )


if __name__ == "__main__":
    main()

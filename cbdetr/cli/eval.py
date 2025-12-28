# ------------------------------------------------------------------------
# Cuboid-DETR Evaluation Script
# Copyright (c) 2025
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

import os
import time
import json
import argparse
from typing import Dict, Any, Tuple, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from cbdetr.util.batch_utils import to_device_batch, build_targets_from_batch
from cbdetr.util.ckpt_utils import load_checkpoint
from cbdetr.dataset.dataset import CuboidDataset, cuboid_collate_fn
from cbdetr.models.build_model import (
    build_model,
    add_model_transformer_args,
    parse_model_transformer_args,
)
from cbdetr.loss.compute_loss import compute_total_loss


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
    return_avg_components: bool = True,
    log_interval: int = 50,  # 追加
    tag: str = "val",       # 追加（val / val_easy / val_hard 用）
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_losses = []
    comp: Dict[str, list] = {}

    num_batches = len(loader)
    t0 = time.time()

    for it, batch in enumerate(loader, 1):
        batch = to_device_batch(batch, device)
        targets = build_targets_from_batch(batch)

        with torch.amp.autocast("cuda", enabled=use_amp):
            ldict, _, _ = compute_total_loss(
                model(batch["images"]),
                targets,
                group_detr=1,
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
                lambda_kp_coarse=kp_cfg.get("lambda_kp_coarse", 0.0),
                lambda_edge_coarse=kp_cfg.get("lambda_edge_coarse", 0.0),
                use_huber=True,
                use_aux_losses=False,
            )

        loss_val = float(ldict["loss_total"].item())
        total_losses.append(loss_val)

        if return_avg_components:
            for k, v in ldict.items():
                if k == "loss_total":
                    continue
                comp.setdefault(k, []).append(float(v.item()))

        # ---- progress log ----
        if log_interval > 0 and (it % log_interval == 0 or it == num_batches):
            elapsed = time.time() - t0
            avg_loss = float(np.mean(total_losses))
            print(
                f"[{tag}] iter {it:5d}/{num_batches} "
                f"avg_loss={avg_loss:.6f} "
                f"time={elapsed/60:.1f} min"
            )

    mean_total = float(np.mean(total_losses)) if total_losses else 0.0
    mean_components = (
        {k: float(np.mean(v)) for k, v in comp.items()}
        if return_avg_components else {}
    )
    return mean_total, mean_components


@torch.no_grad()
def eval_inference_speed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    warmup_iters: int = 10,
    measure_iters: int = 50,
) -> Dict[str, float]:
    model.eval()

    it = iter(loader)
    batches = []
    for _ in range(warmup_iters + measure_iters):
        try:
            batches.append(next(it))
        except StopIteration:
            break

    if not batches:
        return {"iters": 0, "latency_ms": 0.0, "fps": 0.0}

    def _run(batch):
        batch = to_device_batch(batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            _ = model(batch["images"])

    for b in batches[:warmup_iters]:
        _run(b)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n = 0
    for b in batches[warmup_iters : warmup_iters + measure_iters]:
        _run(b)
        n += 1

    if device.type == "cuda":
        torch.cuda.synchronize()

    dt = time.perf_counter() - t0
    if n == 0 or dt <= 0:
        return {"iters": float(n), "latency_ms": 0.0, "fps": 0.0}

    latency_ms = (dt / n) * 1000.0
    fps = n / dt
    return {"iters": float(n), "latency_ms": float(latency_ms), "fps": float(fps)}


def _set_ablation_flags(
    model: nn.Module,
    *,
    disable_img_hf: bool,
    disable_neck_hf: bool,
    disable_edge: bool,
    disable_joint: bool,
) -> Dict[str, Any]:
    restore: Dict[str, Any] = {}

    if disable_neck_hf and hasattr(model, "kp_neck"):
        restore["kp_neck_hf"] = getattr(model.kp_neck, "hf", None)
        model.kp_neck.hf = None

    if disable_img_hf and hasattr(model, "_compute_kp_feat_map"):
        restore["compute_kp_feat_map"] = model._compute_kp_feat_map

        def _compute_kp_feat_map_no_img_hf(features, images, detach_backbone: bool):
            feat_tensors = [feat.tensors for feat in features]
            if detach_backbone:
                feat_tensors = [t.detach() for t in feat_tensors]
            return model.kp_neck(feat_tensors)

        model._compute_kp_feat_map = _compute_kp_feat_map_no_img_hf  # type: ignore[method-assign]

    if disable_edge:
        restore["sample_edge_fn"] = getattr(model, "_sample_edge_features_raw", None)
        restore["aggregate_edge_fn"] = getattr(model, "_aggregate_edge_features", None)

        def _zeros_edge_feat(kp_feat_map, pred_kp_coarse, h):
            B, Nq, K, _ = pred_kp_coarse.shape
            E = int(getattr(model, "edges").shape[0]) if hasattr(model, "edges") else 0
            Ck = int(getattr(model, "kp_feat_dim", kp_feat_map.shape[1]))
            return torch.zeros((B, Nq, E, Ck), device=kp_feat_map.device, dtype=kp_feat_map.dtype)

        def _zeros_kp_edge(edge_feat, kp_point_feat):
            return torch.zeros_like(kp_point_feat)

        model._sample_edge_features_raw = _zeros_edge_feat  # type: ignore[method-assign]
        model._aggregate_edge_features = _zeros_kp_edge     # type: ignore[method-assign]

    if disable_joint:
        restore["kp_joint_decoder"] = getattr(model, "kp_joint_decoder", None)

        class _ZeroDelta(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.zeros((*x.shape[:-1], 2), device=x.device, dtype=x.dtype)

        model.kp_joint_decoder = _ZeroDelta()

    return restore


def _restore_ablation_flags(model: nn.Module, restore: Dict[str, Any]) -> None:
    if "kp_neck_hf" in restore:
        model.kp_neck.hf = restore["kp_neck_hf"]

    if "compute_kp_feat_map" in restore:
        model._compute_kp_feat_map = restore["compute_kp_feat_map"]  # type: ignore[method-assign]

    if "sample_edge_fn" in restore and restore["sample_edge_fn"] is not None:
        model._sample_edge_features_raw = restore["sample_edge_fn"]  # type: ignore[method-assign]

    if "aggregate_edge_fn" in restore and restore["aggregate_edge_fn"] is not None:
        model._aggregate_edge_features = restore["aggregate_edge_fn"]  # type: ignore[method-assign]

    if "kp_joint_decoder" in restore and restore["kp_joint_decoder"] is not None:
        model.kp_joint_decoder = restore["kp_joint_decoder"]


def _normalize_roots(single_root: Optional[str], multi_roots: Optional[List[str]]) -> List[str]:
    roots: List[str] = []
    if multi_roots:
        roots.extend([r for r in multi_roots if r])
    elif single_root:
        roots.append(single_root)
    return roots


def _build_cuboid_dataset(
    roots: List[str],
    *,
    output_size: int,
    object_filter: str = "bbox_center",
) -> Optional[Union[CuboidDataset, ConcatDataset]]:
    datasets = []
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        datasets.append(
            CuboidDataset(
                root_dir=r,
                object_filter=object_filter,
                output_size=output_size,
            )
        )
    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def build_val_loader(
    roots: List[str],
    batch_size: int,
    workers: int,
    output_size: int,
) -> Optional[DataLoader]:
    ds = _build_cuboid_dataset(roots, output_size=output_size)
    if ds is None:
        return None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, workers // 2),
        pin_memory=True,
        collate_fn=cuboid_collate_fn,
    )


def main():
    parser = argparse.ArgumentParser()

    # Legacy single-root (kept for backward compatibility)
    parser.add_argument("--val_root", type=str, default=None)

    # New multi-root
    parser.add_argument("--val_root_easy", type=str, nargs="*", default=None)
    parser.add_argument("--val_root_hard", type=str, nargs="*", default=None)

    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output_size", type=int, default=532)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--warmup_iters", type=int, default=10)
    parser.add_argument("--measure_iters", type=int, default=50)

    parser.add_argument("--print_components", action="store_true")
    parser.add_argument("--json_out", type=str, default=None)

    parser.add_argument("--disable_img_hf", action="store_true")
    parser.add_argument("--disable_neck_hf", action="store_true")
    parser.add_argument("--disable_edge", action="store_true")
    parser.add_argument("--disable_joint", action="store_true")

    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Print evaluation progress every N batches (0 = disable)",
    )

    add_model_transformer_args(parser)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    margs, targs = parse_model_transformer_args(args, device=device)
    model = build_model(margs, targs).to(device)

    dummy_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dummy_sch = torch.optim.lr_scheduler.LambdaLR(dummy_opt, lr_lambda=lambda _: 1.0)
    dummy_scaler = torch.amp.GradScaler("cuda", enabled=False)

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    epoch = load_checkpoint(args.ckpt, model, dummy_opt, dummy_sch, dummy_scaler)
    model.eval()

    matcher_costs = dict(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    cls_cfg = dict(no_object_weight=1.0, use_focal=False, focal_gamma=2.0)
    kp_cfg = dict(
        lambda_kp=20.0,
        lambda_bbox_kp=1.0,
        lambda_edge=4.0,
        lambda_face=0.3,
        lambda_rep=0.005,
        lambda_kp_coarse=10.0,
        lambda_edge_coarse=2.0,
    )

    easy_roots = _normalize_roots(None, args.val_root_easy)
    hard_roots = _normalize_roots(None, args.val_root_hard)
    legacy_val_roots = _normalize_roots(args.val_root, None)

    loaders: Dict[str, DataLoader] = {}

    use_split = (len(easy_roots) > 0) or (len(hard_roots) > 0)

    if not use_split:
        if not legacy_val_roots:
            raise ValueError("Provide --val_root or (--val_root_easy/--val_root_hard).")
        val_loader = build_val_loader(legacy_val_roots, args.batch_size, args.workers, args.output_size)
        if val_loader is None:
            raise RuntimeError("No valid val roots found for --val_root.")
        loaders["val"] = val_loader
    else:
        val_easy_loader = build_val_loader(easy_roots, args.batch_size, args.workers, args.output_size) if easy_roots else None
        val_hard_loader = build_val_loader(hard_roots, args.batch_size, args.workers, args.output_size) if hard_roots else None

        if val_easy_loader is not None:
            loaders["val_easy"] = val_easy_loader
        if val_hard_loader is not None:
            loaders["val_hard"] = val_hard_loader

        if val_easy_loader is not None and val_hard_loader is not None:
            mixed_ds = ConcatDataset([val_easy_loader.dataset, val_hard_loader.dataset])
            loaders["val"] = DataLoader(
                mixed_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=max(1, args.workers // 2),
                pin_memory=True,
                collate_fn=cuboid_collate_fn,
            )
        elif val_easy_loader is not None:
            loaders["val"] = val_easy_loader
        elif val_hard_loader is not None:
            loaders["val"] = val_hard_loader
        else:
            raise RuntimeError("Split enabled but no valid val roots found.")

    restore = _set_ablation_flags(
        model,
        disable_img_hf=args.disable_img_hf,
        disable_neck_hf=args.disable_neck_hf,
        disable_edge=args.disable_edge,
        disable_joint=args.disable_joint,
    )

    results: Dict[str, Any] = {
        "ckpt": args.ckpt,
        "loaded_epoch": int(epoch),
        "ablation": {
            "disable_img_hf": bool(args.disable_img_hf),
            "disable_neck_hf": bool(args.disable_neck_hf),
            "disable_edge": bool(args.disable_edge),
            "disable_joint": bool(args.disable_joint),
        },
        "loss": {},
        "speed": {},
        "val_roots": {
            "legacy_val_root": legacy_val_roots,
            "val_root_easy": easy_roots,
            "val_root_hard": hard_roots,
        },
    }

    for name, loader in loaders.items():
        mean_total, mean_comp = eval_avg_loss(
            model,
            loader,
            device,
            matcher_costs=matcher_costs,
            cls_cfg=cls_cfg,
            kp_cfg=kp_cfg,
            use_amp=args.amp,
            return_avg_components=args.print_components,
            log_interval=args.log_interval,
            tag=name,
        )
        results["loss"][name] = {"loss_total": float(mean_total), "components": mean_comp}

    if args.speed_test:
        results["speed"] = eval_inference_speed(
            model,
            loaders["val"],
            device,
            use_amp=args.amp,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
        )

    _restore_ablation_flags(model, restore)

    print("\n=== Evaluation Summary ===")
    print(f"ckpt: {args.ckpt}")
    print(f"loaded_epoch: {int(epoch)}")
    if use_split:
        if easy_roots:
            print(f"val_root_easy: {easy_roots}")
        if hard_roots:
            print(f"val_root_hard: {hard_roots}")
    else:
        print(f"val_root: {legacy_val_roots}")

    for k, v in results["ablation"].items():
        if v:
            print(f"ablation: {k}=True")

    print("")
    for split_name, d in results["loss"].items():
        print(f"[{split_name}] loss_total = {d['loss_total']:.6f}")
        if args.print_components and d["components"]:
            for kk, vv in sorted(d["components"].items()):
                print(f"  {kk}: {vv:.6f}")

    if args.speed_test and results["speed"]:
        s = results["speed"]
        print("\n[Speed]")
        print(f"  iters: {int(s['iters'])}")
        print(f"  latency_ms/iter: {s['latency_ms']:.3f}")
        print(f"  fps: {s['fps']:.3f}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {args.json_out}")


if __name__ == "__main__":
    main()
# ------------------------------------------------------------------------
# Cuboid-DETR
# Copyright (c) 2025 t-34400
# Licensed under the Apache License, Version 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------

from collections import defaultdict
import json
import os
import math
import time
import argparse
import random
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from cbdetr.util.batch_utils import to_device_batch, build_targets_from_batch
from cbdetr.util.ckpt_utils import save_checkpoint, load_checkpoint

from cbdetr.dataset.dataset import CuboidDataset, cuboid_collate_fn
from cbdetr.models.build_model import (
    build_model,
    add_model_transformer_args,
    parse_model_transformer_args,
)
from cbdetr.loss.compute_loss import compute_total_loss


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# LR scheduler (cosine + linear warmup)
# ----------------------------
def build_scheduler(
    optimizer: optim.Optimizer,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int = 1,
):
    assert steps_per_epoch > 0, "steps_per_epoch must be > 0"

    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ----------------------------
# Training / Evaluation
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    writer: SummaryWriter,
    global_step: int,
    *,
    matcher_costs: Dict[str, float],
    cls_cfg: Dict[str, Any],
    kp_cfg: Dict[str, float],
    grad_clip: float = 0.1,
    use_amp: bool = True,
    log_interval: int = 50,
) -> int:
    model.train()
    running = 0.0
    t0 = time.time()

    for it, batch in enumerate(loader, 1):
        batch = to_device_batch(batch, device)
        targets = build_targets_from_batch(batch)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(batch["images"])
            losses, _, _ = compute_total_loss(
                outputs,
                targets,
                group_detr=model.group_detr,
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
                use_aux_losses=True,
            )
            loss = losses["loss_total"]

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running += loss.item()
        lr_now = optimizer.param_groups[0]["lr"]

        writer.add_scalar("train/loss_total", loss.item(), global_step)
        writer.add_scalar("train/lr", lr_now, global_step)
        for k, v in losses.items():
            if k.startswith("loss_") and k != "loss_total":
                writer.add_scalar(f"train/{k}", float(v.item()), global_step)

        if it % log_interval == 0:
            avg = running / log_interval
            print(f"[iter {it:04d}/{len(loader)}] loss={avg:.4f}  lr={lr_now:.3e}")
            running = 0.0

        global_step += 1

    dt = time.time() - t0
    writer.add_scalar("train/epoch_time_min", dt / 60.0, global_step)
    print(f"train epoch done in {dt/60:.1f} min")
    return global_step


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
    return_avg_components: bool = False,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_losses = []
    comp_losses = defaultdict(list)

    for batch in loader:
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
                lambda_kp_coarse=kp_cfg["lambda_kp_coarse"],
                lambda_edge_coarse=kp_cfg["lambda_edge_coarse"],
                use_huber=True,
                use_aux_losses=False,
            )

        total_losses.append(ldict["loss_total"].item())

        if return_avg_components:
            for k, v in ldict.items():
                if k != "loss_total":
                    comp_losses[k].append(v.item())

    mean_total = float(np.mean(total_losses)) if total_losses else 0.0

    if not return_avg_components:
        return mean_total, {}

    mean_components = {k: float(np.mean(v)) for k, v in comp_losses.items()}
    return mean_total, mean_components


def _normalize_roots(single_root: Optional[str], multi_roots: Optional[List[str]]) -> List[str]:
    """Merge legacy single-root arg and new multi-root arg into a list of dirs."""
    roots: List[str] = []
    if multi_roots:
        roots.extend(multi_roots)
    elif single_root:
        roots.append(single_root)
    return roots


def _build_cuboid_dataset(
    roots: List[str],
    augment: bool,
    output_size: int,
    object_filter: str = "bbox_center",
) -> Optional[ConcatDataset]:
    """Build ConcatDataset from multiple root dirs. Returns None if no valid dir."""
    datasets = []
    for r in roots:
        if r is None:
            continue
        if not os.path.isdir(r):
            continue
        ds = CuboidDataset(
            root_dir=r,
            object_filter=object_filter,
            augment=augment,
            output_size=output_size,
        )
        datasets.append(ds)
    if len(datasets) == 0:
        return None
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def _get_phase(epoch_idx: int, e1: int, e2: int, e3: int) -> str:
    """Return training phase name for given 0-based epoch index."""
    if epoch_idx < e1:
        return "cls_bbox"
    if epoch_idx < e1 + e2:
        return "kp_coarse"
    if epoch_idx < e1 + e2 + e3:
        return "kp_refine"
    return "done"


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    # ---- train roots ----
    parser.add_argument(
        "--train_root",
        type=str,
        default=None,
        help="Root dir with .h5 training data (legacy single easy root)",
    )
    parser.add_argument(
        "--train_root_easy",
        type=str,
        nargs="*",
        default=None,
        help="One or more EASY training roots (overrides train_root if set)",
    )
    parser.add_argument(
        "--train_root_hard",
        type=str,
        nargs="*",
        default=None,
        help="One or more HARD training roots (enable curriculum if set)",
    )

    # ---- val roots ----
    parser.add_argument(
        "--val_root",
        type=str,
        default=None,
        help="Root dir with .h5 validation data (single set, legacy)",
    )
    parser.add_argument(
        "--val_root_easy",
        type=str,
        nargs="*",
        default=None,
        help="One or more EASY validation roots",
    )
    parser.add_argument(
        "--val_root_hard",
        type=str,
        nargs="*",
        default=None,
        help="One or more HARD validation roots",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./runs/cuboid",
        help="Output dir for logs/checkpoints",
    )

    # ---- training schedule ----
    parser.add_argument(
        "--epochs_stage1",
        type=int,
        default=10,
        help="Epochs for stage1: cls+bbox only",
    )
    parser.add_argument(
        "--epochs_stage2",
        type=int,
        default=20,
        help="Epochs for stage2: add KP coarse losses",
    )
    parser.add_argument(
        "--epochs_stage3",
        type=int,
        default=60,
        help="Epochs for stage3: full KP refine",
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume",
    )
    parser.add_argument("--warmup_epochs", type=int, default=1)

    # KP warmup is now stage-wise: coarse in stage2, full KP in stage3.
    parser.add_argument(
        "--kp_warmup_coarse_epochs",
        type=int,
        default=5,
        help="Number of warmup epochs for coarse KP losses in stage2",
    )
    parser.add_argument(
        "--kp_warmup_full_epochs",
        type=int,
        default=10,
        help="Number of warmup epochs for full KP losses in stage3",
    )

    # curriculum for easy/hard train
    parser.add_argument("--cl_easy_epochs", type=int, default=30)

    # ---- loss config (arg-ified) ----
    parser.add_argument("--cost_class", type=float, default=1.0)
    parser.add_argument("--cost_bbox", type=float, default=5.0)
    parser.add_argument("--cost_giou", type=float, default=2.0)

    parser.add_argument("--no_object_weight", type=float, default=1.0)
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument("--lambda_kp", type=float, default=20.0)
    parser.add_argument("--lambda_bbox_kp", type=float, default=1.0)
    parser.add_argument("--lambda_edge", type=float, default=4.0)
    parser.add_argument("--lambda_face", type=float, default=0.3)
    parser.add_argument("--lambda_rep", type=float, default=0.005)
    parser.add_argument("--lambda_kp_coarse", type=float, default=10.0)
    parser.add_argument("--lambda_edge_coarse", type=float, default=2.0)

    # ---- validation efficiency ----
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Run validation every N epochs (1 = every epoch)",
    )
    parser.add_argument(
        "--val_split_detail",
        action="store_true",
        help="If set, compute separate val on easy/hard; "
             "otherwise prefer mixed-only when available.",
    )

    parser.add_argument(
        "--print_loss_components",
        action="store_true",
        help="Print validation loss components",
    )

    add_model_transformer_args(parser=parser)
    args = parser.parse_args()

    # ---- derived schedule ----
    total_epochs = args.epochs_stage1 + args.epochs_stage2 + args.epochs_stage3
    print(
        f"Training schedule: stage1={args.epochs_stage1}, "
        f"stage2={args.epochs_stage2}, stage3={args.epochs_stage3}, "
        f"total={total_epochs}"
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model ----
    margs, targs = parse_model_transformer_args(args, device=device)
    model = build_model(margs, targs).to(device)

    # ----------------------------
    # Datasets & loaders (train with optional curriculum)
    # ----------------------------
    easy_train_roots = _normalize_roots(args.train_root, args.train_root_easy)
    hard_train_roots = _normalize_roots(None, args.train_root_hard)

    use_curriculum = len(hard_train_roots) > 0

    if not use_curriculum:
        train_ds = _build_cuboid_dataset(
            easy_train_roots,
            augment=True,
            output_size=532,
        )
        if train_ds is None:
            raise RuntimeError("No valid training roots provided.")
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=cuboid_collate_fn,
            drop_last=True,
        )
        easy_loader = mixed_loader = None
        steps_per_epoch = len(train_loader)
    else:
        easy_ds = _build_cuboid_dataset(
            easy_train_roots,
            augment=True,
            output_size=532,
        )
        hard_ds = _build_cuboid_dataset(
            hard_train_roots,
            augment=True,
            output_size=532,
        )
        if easy_ds is None or hard_ds is None:
            raise RuntimeError(
                "Both easy and hard roots must be valid when curriculum is enabled."
            )

        easy_loader = DataLoader(
            easy_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=cuboid_collate_fn,
            drop_last=True,
        )

        mixed_ds = ConcatDataset([easy_ds, hard_ds])
        mixed_loader = DataLoader(
            mixed_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=cuboid_collate_fn,
            drop_last=True,
        )

        steps_per_epoch = max(len(easy_loader), len(mixed_loader))
        train_loader = None

    # ----------------------------
    # Validation loaders
    # ----------------------------
    val_loader: Optional[DataLoader] = None
    val_easy_loader: Optional[DataLoader] = None
    val_hard_loader: Optional[DataLoader] = None

    easy_val_roots = _normalize_roots(args.val_root, args.val_root_easy)
    hard_val_roots = _normalize_roots(None, args.val_root_hard)

    use_val_split = (len(easy_val_roots) > 0 and len(hard_val_roots) > 0)

    if not use_val_split:
        if len(easy_val_roots) > 0:
            val_ds = _build_cuboid_dataset(
                easy_val_roots,
                augment=False,
                output_size=532,
            )
            if val_ds is not None:
                val_loader = DataLoader(
                    val_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=max(1, args.workers // 2),
                    pin_memory=True,
                    collate_fn=cuboid_collate_fn,
                )
    else:
        val_easy_ds = _build_cuboid_dataset(
            easy_val_roots,
            augment=False,
            output_size=532,
        )
        val_hard_ds = _build_cuboid_dataset(
            hard_val_roots,
            augment=False,
            output_size=532,
        )

        if val_easy_ds is not None:
            val_easy_loader = DataLoader(
                val_easy_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=max(1, args.workers // 2),
                pin_memory=True,
                collate_fn=cuboid_collate_fn,
            )

        if val_hard_ds is not None:
            val_hard_loader = DataLoader(
                val_hard_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=max(1, args.workers // 2),
                pin_memory=True,
                collate_fn=cuboid_collate_fn,
            )

    # ----------------------------
    # Optimizer / scheduler / AMP
    # ----------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        steps_per_epoch=steps_per_epoch,
        epochs=total_epochs,
        warmup_epochs=args.warmup_epochs,
    )
    scaler = GradScaler("cuda", enabled=args.amp)

    # TensorBoard writer
    tb_dir = os.path.join(args.out_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # ---- curriculum loader selection ----
    def choose_loader(epoch_idx: int) -> DataLoader:
        if not use_curriculum:
            return train_loader
        if epoch_idx < args.cl_easy_epochs:
            stage = "easy"
            loader = easy_loader
        else:
            stage = "mixed"
            loader = mixed_loader
        print(f"  curriculum stage: {stage}")
        return loader

    # ----------------------------
    # KP warmup helpers
    # ----------------------------
    def cosine_warmup(t: float) -> float:
        t = max(0.0, min(1.0, t))
        x = 0.5 - 0.5 * math.cos(math.pi * t)
        return x * x

    def build_kp_cfg_for_epoch(
        epoch_idx: int,
        phase: str,
        args,
        base_kp_cfg: Dict[str, float],
    ) -> Tuple[Dict[str, float], float, float]:
        """
        Returns (kp_cfg_epoch, kp_coarse_scale, kp_full_scale).
        """
        if phase == "cls_bbox":
            kp_cfg_epoch = {k: 0.0 for k in base_kp_cfg.keys()}
            return kp_cfg_epoch, 0.0, 0.0

        # stage2: coarse-only warmup
        if phase == "kp_coarse":
            phase_start = args.epochs_stage1
            phase_epoch = max(0, epoch_idx - phase_start)
            denom = max(1, args.kp_warmup_coarse_epochs)
            t = phase_epoch / float(denom)
            kp_coarse_scale = cosine_warmup(t)
            kp_full_scale = 0.0

            kp_cfg_epoch = {
                "lambda_kp": 0.0,
                "lambda_bbox_kp": 0.0,
                "lambda_edge": 0.0,
                "lambda_face": 0.0,
                "lambda_rep": 0.0,
                "lambda_kp_coarse": base_kp_cfg["lambda_kp_coarse"] * kp_coarse_scale,
                "lambda_edge_coarse": base_kp_cfg["lambda_edge_coarse"] * kp_coarse_scale,
            }
            return kp_cfg_epoch, kp_coarse_scale, kp_full_scale

        # stage3: full KP warmup, coarse fixed to 1.0
        phase_start = args.epochs_stage1 + args.epochs_stage2
        phase_epoch = max(0, epoch_idx - phase_start)
        denom = max(1, args.kp_warmup_full_epochs)
        t = phase_epoch / float(denom)
        kp_full_scale = cosine_warmup(t)
        kp_coarse_scale = 1.0

        kp_cfg_epoch = {
            "lambda_kp": base_kp_cfg["lambda_kp"] * kp_full_scale,
            "lambda_bbox_kp": base_kp_cfg["lambda_bbox_kp"] * kp_full_scale,
            "lambda_edge": base_kp_cfg["lambda_edge"] * kp_full_scale,
            "lambda_face": base_kp_cfg["lambda_face"] * kp_full_scale,
            "lambda_rep": base_kp_cfg["lambda_rep"] * kp_full_scale,
            "lambda_kp_coarse": base_kp_cfg["lambda_kp_coarse"] * kp_coarse_scale,
            "lambda_edge_coarse": base_kp_cfg["lambda_edge_coarse"] * kp_coarse_scale,
        }
        return kp_cfg_epoch, kp_coarse_scale, kp_full_scale

    # ----------------------------
    # Resume (optional)
    # ----------------------------
    start_epoch = 0
    best_val = float("inf")
    history: Dict[str, List] = {
        "epoch": [],
        "phase": [],
        "val_loss": [],
        "best_val": [],
    }

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        print(f"Resumed at epoch {start_epoch}")

    # ---- Loss configs from args ----
    matcher_costs = dict(
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou,
    )
    cls_cfg = dict(
        no_object_weight=args.no_object_weight,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
    )
    base_kp_cfg = dict(
        lambda_kp=args.lambda_kp,
        lambda_bbox_kp=args.lambda_bbox_kp,
        lambda_edge=args.lambda_edge,
        lambda_face=args.lambda_face,
        lambda_rep=args.lambda_rep,
        lambda_kp_coarse=args.lambda_kp_coarse,
        lambda_edge_coarse=args.lambda_edge_coarse,
    )

    # ----------------------------
    # Training loop
    # ----------------------------
    global_step = 0
    summary_path = os.path.join(args.out_dir, "summary.json")
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        for epoch in range(start_epoch + 1, total_epochs + 1):
            epoch_idx = epoch - 1
            phase = _get_phase(
                epoch_idx,
                args.epochs_stage1,
                args.epochs_stage2,
                args.epochs_stage3,
            )
            print(f"\n=== Epoch {epoch}/{total_epochs} (phase: {phase}) ===")
            t0 = time.time()

            kp_cfg_epoch, kp_coarse_scale, kp_full_scale = build_kp_cfg_for_epoch(
                epoch_idx=epoch_idx,
                phase=phase,
                args=args,
                base_kp_cfg=base_kp_cfg,
            )

            cur_loader = choose_loader(epoch_idx)

            global_step = train_one_epoch(
                model,
                cur_loader,
                optimizer,
                scheduler,
                scaler,
                device,
                writer,
                global_step,
                matcher_costs=matcher_costs,
                cls_cfg=cls_cfg,
                kp_cfg=kp_cfg_epoch,
                grad_clip=args.grad_clip,
                use_amp=args.amp,
                log_interval=50,
            )

            # ------------------------
            # Validation (interval-based)
            # ------------------------
            run_val = (
                (epoch % args.val_interval == 0)
                or (epoch == total_epochs)
            )

            val_loss_main = float("nan")
            avg_components_main: Dict[str, float] = {}

            val_easy_loss = None
            val_hard_loss = None
            avg_easy: Dict[str, float] = {}
            avg_hard: Dict[str, float] = {}

            if run_val:
                if val_loader is not None:
                    val_loss_main, avg_components_main = eval_avg_loss(
                        model,
                        val_loader,
                        device,
                        matcher_costs=matcher_costs,
                        cls_cfg=cls_cfg,
                        kp_cfg=base_kp_cfg,
                        use_amp=args.amp,
                        return_avg_components=args.print_loss_components,
                    )
                    writer.add_scalar("val/loss_total", val_loss_main, epoch)

                elif val_easy_loader is not None or val_hard_loader is not None:
                    Neasy = 0
                    Nhard = 0

                    if val_easy_loader is not None:
                        val_easy_loss, avg_easy = eval_avg_loss(
                            model,
                            val_easy_loader,
                            device,
                            matcher_costs=matcher_costs,
                            cls_cfg=cls_cfg,
                            kp_cfg=base_kp_cfg,
                            use_amp=args.amp,
                            return_avg_components=args.print_loss_components,
                        )
                        writer.add_scalar("val_easy/loss_total", val_easy_loss, epoch)
                        Neasy = len(val_easy_loader.dataset)
                    else:
                        avg_easy = {}

                    if val_hard_loader is not None:
                        val_hard_loss, avg_hard = eval_avg_loss(
                            model,
                            val_hard_loader,
                            device,
                            matcher_costs=matcher_costs,
                            cls_cfg=cls_cfg,
                            kp_cfg=base_kp_cfg,
                            use_amp=args.amp,
                            return_avg_components=args.print_loss_components,
                        )
                        writer.add_scalar("val_hard/loss_total", val_hard_loss, epoch)
                        Nhard = len(val_hard_loader.dataset)
                    else:
                        avg_hard = {}

                    if Neasy + Nhard > 0:
                        if Neasy > 0 and Nhard > 0:
                            val_loss_main = (
                                val_easy_loss * Neasy + val_hard_loss * Nhard
                            ) / float(Neasy + Nhard)

                            avg_components_main = {}
                            keys = set(avg_easy.keys()) | set(avg_hard.keys())
                            for k in keys:
                                ve = avg_easy.get(k, 0.0)
                                vh = avg_hard.get(k, 0.0)
                                avg_components_main[k] = (
                                    ve * Neasy + vh * Nhard
                                ) / float(max(1, Neasy + Nhard))

                        elif Neasy > 0:
                            val_loss_main = val_easy_loss
                            avg_components_main = avg_easy
                        else:
                            val_loss_main = val_hard_loss
                            avg_components_main = avg_hard

                        writer.add_scalar("val/loss_total", val_loss_main, epoch)

            epoch_time = time.time() - t0
            writer.add_scalar("misc/epoch_time_min", epoch_time / 60.0, epoch)
            writer.add_scalar("misc/epoch", epoch, epoch)
            writer.add_scalar("misc/kp_coarse_loss_scale", kp_coarse_scale, epoch)
            writer.add_scalar("misc/kp_full_loss_scale", kp_full_scale, epoch)

            print(
                f"kp_coarse_loss_scale = {kp_coarse_scale:.3f}, "
                f"kp_full_loss_scale = {kp_full_scale:.3f}"
            )
            if run_val:
                print(
                    f"Epoch {epoch}: val_loss={val_loss_main:.4f}, "
                    f"time={epoch_time/60:.1f} min"
                )
            else:
                print(
                    f"Epoch {epoch}: val skipped (interval={args.val_interval}), "
                    f"time={epoch_time/60:.1f} min"
                )

            if args.print_loss_components and run_val:
                if avg_components_main:
                    print("  Validation loss components (overall):")
                    for k, v in avg_components_main.items():
                        print(f"    {k}: {v:.4f}")
                if avg_easy:
                    print("  Validation loss components (easy):")
                    for k, v in avg_easy.items():
                        print(f"    {k}: {v:.4f}")
                if avg_hard:
                    print("  Validation loss components (hard):")
                    for k, v in avg_hard.items():
                        print(f"    {k}: {v:.4f}")

            # ------------------------
            # Save checkpoints
            # ------------------------
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val": best_val,
                },
                args.out_dir,
                "last.pt",
            )

            if run_val and (
                val_loader is not None
                or val_easy_loader is not None
                or val_hard_loader is not None
            ):
                if val_loss_main < best_val:
                    best_val = val_loss_main
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict(),
                            "best_val": best_val,
                        },
                        args.out_dir,
                        "best.pt",
                    )
                    print(f"Saved best checkpoint (val {best_val:.4f})")

            # ---- update summary (robust to interruption) ----
            history["epoch"].append(epoch)
            history["phase"].append(phase)
            history["val_loss"].append(val_loss_main)
            history["best_val"].append(best_val)
            try:
                with open(summary_path, "w") as f:
                    json.dump(history, f, indent=2)
            except Exception as e:
                print(f"Warning: failed to write summary.json: {e}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    finally:
        writer.close()
        print(f"Training finished or interrupted. TensorBoard logs at: {tb_dir}")
        print(f"Summary JSON: {summary_path}")
        print(f"Launch: tensorboard --logdir {tb_dir} --port 6006")


if __name__ == "__main__":
    main()
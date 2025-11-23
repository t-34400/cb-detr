# ------------------------------------------------------------------------
# Cuboid-DETR
# Copyright (c) 2025 t-34400
# Licensed under the Apache License, Version 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------

from collections import defaultdict
import os
import math
import time
import argparse
import random
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
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
def build_scheduler(optimizer: optim.Optimizer, steps_per_epoch: int, epochs: int, warmup_epochs: int = 1):
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

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(batch["images"])
            losses, _, _ = compute_total_loss(
                outputs, targets,
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
):
    model.eval()
    total_losses = []
    comp_losses = defaultdict(list)

    for batch in loader:
        batch = to_device_batch(batch, device)
        targets = build_targets_from_batch(batch)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(batch["images"])
            ldict, _, _ = compute_total_loss(
                outputs, targets,
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


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, required=True, help="Root dir with .h5 training data")
    parser.add_argument("--val_root", type=str, default=None, help="Root dir with .h5 validation data")
    parser.add_argument("--out_dir", type=str, default="./runs/cuboid", help="Output dir for logs/checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--kp_warmup_epochs", type=int, default=5, help="Number of epochs to warm up keypoint losses")
    parser.add_argument("--print_loss_components", action="store_true", help="Print validation loss components")
    add_model_transformer_args(parser=parser)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    margs, targs = parse_model_transformer_args(args, device=device)
    model = build_model(margs, targs).to(device)

    # Datasets & loaders
    size_multiple = margs.patch_size * margs.num_windows
    train_ds = CuboidDataset(root_dir=args.train_root, size_multiple=size_multiple, filter="bbox_center")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=cuboid_collate_fn,
        drop_last=True,
    )
    val_loader = None
    if args.val_root is not None and os.path.isdir(args.val_root):
        val_ds = CuboidDataset(root_dir=args.val_root, size_multiple=size_multiple, filter="bbox_center")
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(1, args.workers // 2),
            pin_memory=True,
            collate_fn=cuboid_collate_fn,
        )

    # Optimizer / scheduler / AMP
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, steps_per_epoch=len(train_loader), epochs=args.epochs, warmup_epochs=args.warmup_epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    # TensorBoard writer
    tb_dir = os.path.join(args.out_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # Resume (optional)
    start_epoch = 0
    best_val = float("inf")
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        print(f"Resumed at epoch {start_epoch}")

    # Loss configs
    matcher_costs = dict(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    cls_cfg = dict(no_object_weight=1.0, use_focal=False, focal_gamma=2.0)
    kp_cfg = dict(lambda_kp=4.0, lambda_bbox_kp=1.0, lambda_edge=0.5, lambda_face=0.3, lambda_rep=0.005)

    # Training loop
    global_step = 0
    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        t0 = time.time()

        kp_warmup = max(1, args.kp_warmup_epochs)
        kp_scale = min(1.0, epoch / kp_warmup)

        kp_cfg_epoch = {k: v * kp_scale for k, v in kp_cfg.items()}

        global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, writer, global_step,
            matcher_costs=matcher_costs, cls_cfg=cls_cfg, kp_cfg=kp_cfg_epoch,
            grad_clip=args.grad_clip, use_amp=args.amp, log_interval=50,
        )

        # Validation
        val_loss = 0.0
        avg_components = {}
        if val_loader is not None:
            val_loss, avg_components = eval_avg_loss(
                model, val_loader, device,
                matcher_costs=matcher_costs, cls_cfg=cls_cfg, kp_cfg=kp_cfg_epoch, use_amp=args.amp,
                return_avg_components=args.print_loss_components
            )
            writer.add_scalar("val/loss_total", val_loss, epoch)

        epoch_time = time.time() - t0
        writer.add_scalar("misc/epoch_time_min", epoch_time / 60.0, epoch)
        writer.add_scalar("misc/epoch", epoch, epoch)

        print(f"kp_loss_scale = {kp_scale:.3f}")
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}, time={epoch_time/60:.1f} min")
        if args.print_loss_components and avg_components:
            print("  Validation loss components:")
            for k, v in avg_components.items():
                print(f"    {k}: {v:.4f}")

        # Save last
        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val": best_val,
        }, args.out_dir, "last.pt")

        # Save best
        if val_loader is not None and val_loss < best_val:
            best_val = val_loss
            save_checkpoint({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val": best_val,
            }, args.out_dir, "best.pt")
            print(f"Saved best checkpoint (val {best_val:.4f})")

    writer.close()
    print(f"Training complete. TensorBoard logs at: {tb_dir}")
    print(f"Launch: tensorboard --logdir {tb_dir} --port 6006")


if __name__ == "__main__":
    main()

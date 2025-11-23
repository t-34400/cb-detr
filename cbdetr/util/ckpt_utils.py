# ----------------------------
# Checkpoint I/O
# ----------------------------
import os
from typing import Any, Dict

import torch
import torch.nn as nn


def save_checkpoint(state: Dict[str, Any], out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state, os.path.join(out_dir, name))


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None, scaler=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0))
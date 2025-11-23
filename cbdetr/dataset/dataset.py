# ------------------------------------------------------------------------
# Cuboid-DETR
# Copyright (c) 2025 t-34400. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import os
import glob
import math
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class CuboidDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        include_intrinsics: bool = True,
        include_extrinsics: bool = True,
        min_objects: int = 0,
        image_key: str = "images",
        intr_key: str = "intrinsics",
        extr_key: str = "extrinsics",
        ann_group: str = "ann",
        faces_key: str = "categories/cuboid_topology/faces",
        img2obj_key: str = "img2obj",

        size_multiple: int | None = None,
        size_mode: str = "ceil",            # "ceil" | "floor" | "round"
        interp: str = "bilinear",           # resize interpolation

        filter: str | None = None,          # None | "bbox_center" | "kp_in_image"

        center_margin: float = 0.0,
    ):
        self.root_dir = root_dir
        self.include_intrinsics = include_intrinsics
        self.include_extrinsics = include_extrinsics
        self.image_key = image_key
        self.intr_key = intr_key
        self.extr_key = extr_key
        self.ann_group = ann_group
        self.img2obj_key = img2obj_key
        self.faces = None

        self.size_multiple = size_multiple
        self.size_mode = size_mode
        self.interp = interp

        self.center_margin = float(center_margin)

        self.filter = None
        if filter == "bbox_center":
            self.filter = self._filter_objects_by_bbox_center
        elif filter == "kp_in_image":
            self.filter = self._filter_objects_by_kp_in_image

        self.h5_paths = sorted(
            glob.glob(os.path.join(root_dir, "**", "*.h5"), recursive=True)
        )
        if not self.h5_paths:
            raise FileNotFoundError(f"No .h5 files found under: {root_dir}")

        self.entries: List[Tuple[str, int, int, int]] = []
        for path in self.h5_paths:
            with h5py.File(path, "r") as f:
                if self.img2obj_key not in f:
                    raise KeyError(f"{self.img2obj_key} not found in {path}")
                img2obj = np.array(f[self.img2obj_key])  # shape=(num_images, 2)

                if self.image_key not in f:
                    raise KeyError(f"{self.image_key} not found in {path}")
                num_images = f[self.image_key].shape[0]
                if img2obj.shape[0] != num_images:
                    raise ValueError(
                        f"{path}: img2obj rows ({img2obj.shape[0]}) != images count ({num_images})"
                    )

                for img_idx in range(num_images):
                    start, count = int(img2obj[img_idx, 0]), int(img2obj[img_idx, 1])
                    if count >= min_objects:
                        self.entries.append((path, img_idx, start, count))
                
                if self.faces is None:
                    if faces_key in f:
                        self.faces = torch.from_numpy(f[faces_key][:])
                    else:
                        self.faces = torch.from_numpy(get_default_faces())

        if not self.entries:
            raise RuntimeError("No images matched the filtering conditions (e.g., min_objects).")

    def __len__(self) -> int:
        return len(self.entries)

    def _read_object_array(self, dset, start: int, end: int):
        slice_obj = dset[start:end]
        out = []
        for v in slice_obj:
            out.append(np.array(v))
        return out

    @staticmethod
    def _gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a

    def _compute_uniform_scale_to_multiple(self, H: int, W: int) -> Tuple[float, Tuple[int, int]]:
        """
        Find uniform scale s so that (s*H, s*W) are multiples of k.
        s = (k * t) / gcd(H,W), choose integer t by mode wrt s≈1.
        """
        k = self.size_multiple
        if not k or k <= 0:
            return 1.0, (H, W)

        g = self._gcd(H, W)
        # base t* so that s≈1 -> t≈g/k
        base = g / k
        if self.size_mode == "floor":
            t = max(1, int(math.floor(base)))
        elif self.size_mode == "round":
            t = max(1, int(round(base)))
        else:
            t = max(1, int(math.ceil(base)))
        s = (k * t) / g
        Hn = int(round(H * s))
        Wn = int(round(W * s))

        if Hn % k != 0: Hn += (k - (Hn % k))
        if Wn % k != 0: Wn += (k - (Wn % k))
        s = Hn / H  # uniform by construction
        return float(s), (Hn, Wn)

    def _resize_image(self, img_chw: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        # img_chw: (3,H,W) float[0..1]
        img_bchw = img_chw.unsqueeze(0)
        mode = self.interp
        align_corners = None
        if mode in ("bilinear", "bicubic"):
            align_corners = False
        out = F.interpolate(img_bchw, size=target_hw, mode=mode, align_corners=align_corners)
        return out.squeeze(0)

    def _scale_intrinsics(self, intr: np.ndarray | torch.Tensor | None, s: float):
        if intr is None:
            return None
        K = torch.as_tensor(intr).clone().float()

        if K.ndim == 2 and K.shape[0] >= 3 and K.shape[1] >= 3:
            K[0, 0] *= s  # fx
            K[1, 1] *= s  # fy
            K[0, 2] *= s  # cx
            K[1, 2] *= s  # cy
        return K

    def _scale_annotations(self, ann: Dict[str, Any], s: float):
        # scale pixel-domain annotations
        if "bbox" in ann:
            # (N,4) [x,y,w,h] in px
            b = torch.from_numpy(ann["bbox"]).float()
            b[:, 0:2] *= s
            b[:, 2:4] *= s
            ann["bbox"] = b

        if "uv" in ann:
            uv = torch.from_numpy(ann["uv"]).float()
            # if looks normalized, keep; else scale
            maxv = float(uv.abs().max())
            if maxv > 1.5:
                uv[:, 0] *= s
                uv[:, 1] *= s
            ann["uv"] = uv

        if "edge_len_px" in ann:
            v = torch.from_numpy(ann["edge_len_px"]).float()
            ann["edge_len_px"] = v * s

        # pixel areas (best guess)
        if "full_px" in ann:
            v = torch.from_numpy(ann["full_px"]).float()
            ann["full_px"] = v * (s * s)
        if "visible_px" in ann:
            v = torch.from_numpy(ann["visible_px"]).float()
            ann["visible_px"] = v * (s * s)

        # keep ratios/scores as-is

        if "visible_edges" in ann:
            ann["visible_edges"] = torch.from_numpy(ann["visible_edges"]).to(torch.uint8)
        if "visible_vertices" in ann:
            ann["visible_vertices"] = torch.from_numpy(ann["visible_vertices"]).to(torch.uint8)
        if "edge_vis_ratio" in ann:
            ann["edge_vis_ratio"] = torch.from_numpy(ann["edge_vis_ratio"]).float()
        if "vis_ratio" in ann:
            ann["vis_ratio"] = torch.from_numpy(ann["vis_ratio"]).float()
        if "pnp_cond_score" in ann:
            ann["pnp_cond_score"] = torch.from_numpy(ann["pnp_cond_score"]).float()

        return ann

    def _filter_objects_by_bbox_center(
        self,
        ann: Dict[str, Any],
        H: int,
        W: int,
    ) -> Dict[str, Any]:
        if "bbox" not in ann:
            return ann

        b = ann["bbox"]

        # bbox: (N,4) [x, y, w, h] in px
        if isinstance(b, torch.Tensor):
            b_t = b.float()
        elif isinstance(b, np.ndarray):
            b_t = torch.from_numpy(b).float()
        else:
            return ann

        if b_t.ndim != 2 or b_t.shape[1] < 4:
            return ann

        N = b_t.shape[0]
        if N == 0:
            return ann

        x = b_t[:, 0]
        y = b_t[:, 1]
        w = b_t[:, 2]
        h = b_t[:, 3]

        cx = x + 0.5 * w
        cy = y + 0.5 * h

        m = float(self.center_margin)

        valid_center = (cx >= -m) & (cx < float(W) + m) & \
                       (cy >= -m) & (cy < float(H) + m)

        if valid_center.all():
            return ann

        keep_idx = valid_center.nonzero(as_tuple=False).flatten()
        keep_idx_np = keep_idx.cpu().numpy().tolist()

        new_ann: Dict[str, Any] = {}
        for k, v in ann.items():
            if isinstance(v, torch.Tensor):
                if v.dim() >= 1 and v.shape[0] == N:
                    new_ann[k] = v[keep_idx]
                else:
                    new_ann[k] = v
            elif isinstance(v, np.ndarray):
                if v.ndim >= 1 and v.shape[0] == N:
                    new_ann[k] = v[keep_idx_np]
                else:
                    new_ann[k] = v
            elif isinstance(v, list):
                if len(v) == N:
                    new_ann[k] = [v[i] for i in keep_idx_np]
                else:
                    new_ann[k] = v
            else:
                new_ann[k] = v

        return new_ann

    def _filter_objects_by_kp_in_image(
        self,
        ann: Dict[str, Any],
        H: int,
        W: int,
    ) -> Dict[str, Any]:
        if "uv" not in ann:
            return ann

        uv = ann["uv"]
        if isinstance(uv, torch.Tensor):
            uv_t = uv
        elif isinstance(uv, np.ndarray):
            uv_t = torch.from_numpy(uv)
        else:
            return ann

        if uv_t.numel() == 0:
            return ann

        if uv_t.ndim == 2 and uv_t.shape[-1] == 2:
            # (N, 2) → (N, 1, 2)
            uv_t = uv_t.unsqueeze(1)
        elif uv_t.ndim == 3 and uv_t.shape[-1] == 2:
            pass
        else:
            return ann

        N = uv_t.shape[0]
        x = uv_t[..., 0]
        y = uv_t[..., 1]

        maxv = float(uv_t.abs().max())
        if maxv <= 1.5:
            mask_pts = (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)
        else:
            mask_pts = (x >= 0.0) & (x < float(W)) & (y >= 0.0) & (y < float(H))

        mask_obj = mask_pts.view(N, -1).all(dim=1)

        if mask_obj.all():
            return ann

        keep_idx = mask_obj.nonzero(as_tuple=False).flatten()
        keep_idx_np = keep_idx.cpu().numpy().tolist()

        new_ann: Dict[str, Any] = {}
        for k, v in ann.items():
            if isinstance(v, torch.Tensor):
                if v.dim() >= 1 and v.shape[0] == N:
                    new_ann[k] = v[keep_idx]
                else:
                    new_ann[k] = v
            elif isinstance(v, np.ndarray):
                if v.ndim >= 1 and v.shape[0] == N:
                    new_ann[k] = v[keep_idx_np]
                else:
                    new_ann[k] = v
            elif isinstance(v, list):
                if len(v) == N:
                    new_ann[k] = [v[i] for i in keep_idx_np]
                else:
                    new_ann[k] = v
            else:
                new_ann[k] = v

        return new_ann
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        h5_path, img_idx, start, count = self.entries[index]
        end = start + count

        with h5py.File(h5_path, "r") as f:
            img = f[self.image_key][img_idx]  # (H, W, 3), uint8
            H, W = int(img.shape[0]), int(img.shape[1])

            intr = f[self.intr_key][img_idx] if self.include_intrinsics and self.intr_key in f else None
            extr = f[self.extr_key][img_idx] if self.include_extrinsics and self.extr_key in f else None

            ann_grp = f[self.ann_group]
            ann = {}
            for key in ann_grp.keys():
                dset = ann_grp[key]
                if dset.dtype.kind == "O":
                    ann[key] = self._read_object_array(dset, start, end)
                else:
                    ann[key] = dset[start:end]

            img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0  # (3,H,W)

            s = 1.0
            target_hw = (H, W)
            if self.size_multiple:
                s, target_hw = self._compute_uniform_scale_to_multiple(H, W)
                if abs(s - 1.0) > 1e-6:
                    img = self._resize_image(img, target_hw)
                    H, W = target_hw

            K_scaled = self._scale_intrinsics(intr, s) if self.include_intrinsics else None
            ann = self._scale_annotations(ann, s)

            if self.filter is not None:
                ann = self.filter(ann, H, W)

            if "bbox" in ann and isinstance(ann["bbox"], torch.Tensor):
                obj_count_after = int(ann["bbox"].shape[0])
            else:
                obj_count_after = count  # fallback

            sample: Dict[str, Any] = {
                "image": img,  # (3,H,W)
                "intrinsics": K_scaled if K_scaled is not None else (torch.from_numpy(intr).float() if intr is not None else None),
                "extrinsics": torch.from_numpy(extr).float() if extr is not None else None,
                "ann": ann,           # dict: key -> Tensor or list
                "meta": {
                    "h5_path": h5_path,
                    "image_index_in_file": img_idx,
                    "obj_start": start,
                    "obj_count_raw": count,
                    "obj_count": obj_count_after,
                    "scale": s,
                    "orig_hw": (int(f[self.image_key].shape[1]), int(f[self.image_key].shape[2])) if False else None,
                    "resized_hw": (H, W),
                    "size_multiple": self.size_multiple,
                    "size_mode": self.size_mode,
                }
            }

            return sample


def cuboid_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)  # (B,3,H,W)
    intr_list = [b["intrinsics"] for b in batch]
    extr_list = [b["extrinsics"] for b in batch]
    meta_list = [b["meta"] for b in batch]
    anns = [b["ann"] for b in batch]
    return {
        "images": images,
        "intrinsics": intr_list,
        "extrinsics": extr_list,
        "anns": anns,
        "metas": meta_list,
    }


def get_default_faces():
    return np.array([
        [0, 3, 2, 1], [4, 5, 6, 7], [0, 4, 7, 3],
        [1, 2, 6, 5], [0, 1, 5, 4], [3, 7, 6, 2]
    ])


if __name__ == "__main__":
    root = "dataset"

    ds = CuboidDataset(
        root_dir=root,
        include_intrinsics=True,
        include_extrinsics=True,
        min_objects=0,

        # example: make sizes multiples of 32, up-scale only
        size_multiple=32,
        size_mode="ceil",       # or "floor"/"round"
        interp="bilinear",

        remove_oob_center = True,
    )

    print("=== Dataset summary ===")
    print(f"Found {len(ds.h5_paths)} h5 files")
    print(f"Total image samples: {len(ds)}")
    print(f"Example file paths:")
    for p in ds.h5_paths[:3]:
        print("  ", p)
    print("=======================\n")

    print(f"faces: {ds.faces}")
    print()

    dl = DataLoader(
        ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=cuboid_collate_fn,
    )

    for i, batch in enumerate(dl):
        print(f"--- Batch {i} ---")
        print("images:", batch["images"].shape, batch["images"].dtype)
        print("num samples in batch:", len(batch["anns"]))
        print()

        for bidx, ann in enumerate(batch["anns"]):
            meta = batch["metas"][bidx]
            print(f" Sample {bidx}: {meta['h5_path']} [img_idx={meta['image_index_in_file']}]")
            print(f"  num_objects: {meta['obj_count']}")
            print(f"  scale: {meta['scale']:.4f}, resized_hw: {meta['resized_hw']}, k={meta['size_multiple']}, mode={meta['size_mode']}")
            if "bbox" in ann:
                print(f"  bbox: shape {ann['bbox'].shape}, example {ann['bbox'][:2]}")
            if "uv" in ann:
                print(f"  uv: shape {ann['uv'].shape}, example {ann['uv'][:1]}")
            if "visible_faces" in ann:
                print(f"  visible_faces[0]: {ann['visible_faces'][0]}")
            print()

        break

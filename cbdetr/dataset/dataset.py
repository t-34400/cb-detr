# ------------------------------------------------------------------------
# Cuboid-DETR Dataset
# Copyright (c) 2025 t-34400
# Licensed under the Apache License, Version 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------

import os
import glob
from typing import Any, Dict, List, Tuple, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader


def get_default_faces() -> np.ndarray:
    return np.array(
        [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 4, 7, 3],
            [1, 2, 6, 5],
            [0, 1, 5, 4],
            [3, 7, 6, 2],
        ]
    )


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
        output_size: Optional[Tuple[int, int] | int] = None,
        resize_interp: str = "bilinear",
        augment: bool = False,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        random_crop: bool = True,
        random_hflip: bool = False,
        photometric_augment: bool = True,
        object_filter: Optional[str] = None,   # None | "bbox_center" | "kp_in_image"
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

        if output_size is None:
            self.output_size: Optional[Tuple[int, int]] = None
        elif isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = (int(output_size[0]), int(output_size[1]))

        self.resize_interp = resize_interp

        self.augment = augment
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.random_crop = random_crop
        self.random_hflip = random_hflip
        self.photometric_augment = photometric_augment

        self.center_margin = float(center_margin)

        self.filter = None
        if object_filter == "bbox_center":
            self.filter = self._filter_objects_by_bbox_center
        elif object_filter == "kp_in_image":
            self.filter = self._filter_objects_by_kp_in_image

        self.faces: Optional[torch.Tensor] = None

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
                img2obj = np.asarray(f[self.img2obj_key])

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
                        self.faces = torch.from_numpy(np.asarray(f[faces_key]))
                    else:
                        self.faces = torch.from_numpy(get_default_faces())

        if not self.entries:
            raise RuntimeError(
                "No images matched the filtering conditions (e.g., min_objects)."
            )

    def __len__(self) -> int:
        return len(self.entries)

    # ---------------------------------------------------------------------
    # low-level helpers
    # ---------------------------------------------------------------------

    def _read_object_array(self, dset, start: int, end: int) -> List[np.ndarray]:
        slice_obj = dset[start:end]
        out: List[np.ndarray] = []
        for v in slice_obj:
            out.append(np.array(v))
        return out

    def _resize_image(
        self, img_chw: torch.Tensor, target_hw: Tuple[int, int]
    ) -> torch.Tensor:
        img_bchw = img_chw.unsqueeze(0)
        mode = self.resize_interp
        align_corners = False if mode in ("bilinear", "bicubic") else None
        out = F.interpolate(img_bchw, size=target_hw, mode=mode, align_corners=align_corners)
        return out.squeeze(0)

    def _apply_photometric_transforms(self, img: torch.Tensor) -> torch.Tensor:
        photometric_augment = self.augment and self.photometric_augment

        if not photometric_augment:
            return img

        if torch.rand(1).item() < 0.8:
            b = 0.2
            c = 0.2
            s = 0.2

            factor = 1.0 + (torch.rand(1).item() * 2 * b - b)
            img = TF.adjust_brightness(img, factor)

            factor = 1.0 + (torch.rand(1).item() * 2 * c - c)
            img = TF.adjust_contrast(img, factor)

            factor = 1.0 + (torch.rand(1).item() * 2 * s - s)
            img = TF.adjust_saturation(img, factor)

        if torch.rand(1).item() < 0.5:
            gamma = torch.empty(1).uniform_(0.8, 1.2).item()
            img = TF.adjust_gamma(img, gamma)

        if torch.rand(1).item() < 0.5:
            std = 0.03
            noise = torch.randn_like(img) * std
            img = img + noise

        img = img.clamp(0.0, 1.0)
        return img

    def _scale_intrinsics(
        self,
        intr: Optional[np.ndarray | torch.Tensor],
        s: float,
    ) -> Optional[torch.Tensor]:
        if intr is None:
            return None
        K = torch.as_tensor(intr).clone().float()
        if K.ndim == 2 and K.shape[0] >= 3 and K.shape[1] >= 3:
            K[0, 0] *= s
            K[1, 1] *= s
            K[0, 2] *= s
            K[1, 2] *= s
        return K

    def _scale_intrinsics_anisotropic(
        self,
        intr: Optional[torch.Tensor],
        sx: float,
        sy: float,
    ) -> Optional[torch.Tensor]:
        if intr is None:
            return None
        K = intr.clone().float()
        if K.ndim == 2 and K.shape[0] >= 3 and K.shape[1] >= 3:
            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] *= sx
            K[1, 2] *= sy
        return K

    def _crop_intrinsics(
        self,
        K: Optional[torch.Tensor],
        x0: int,
        y0: int,
    ) -> Optional[torch.Tensor]:
        if K is None:
            return None
        K = K.clone()
        if K.ndim == 2 and K.shape[0] >= 3 and K.shape[1] >= 3:
            K[0, 2] -= float(x0)
            K[1, 2] -= float(y0)
        return K

    def _hflip_intrinsics(
        self,
        K: Optional[torch.Tensor],
        W: int,
    ) -> Optional[torch.Tensor]:
        if K is None:
            return None
        K = K.clone()
        if K.ndim == 2 and K.shape[0] >= 3 and K.shape[1] >= 3:
            cx = K[0, 2]
            K[0, 2] = (W - 1.0) - cx
        return K

    def _scale_annotations_uniform(
        self,
        ann: Dict[str, Any],
        s: float,
    ) -> Dict[str, Any]:
        if s == 1.0:
            if "visible_edges" in ann and isinstance(ann["visible_edges"], np.ndarray):
                ann["visible_edges"] = torch.from_numpy(ann["visible_edges"]).to(torch.uint8)
            if "visible_vertices" in ann and isinstance(ann["visible_vertices"], np.ndarray):
                ann["visible_vertices"] = torch.from_numpy(ann["visible_vertices"]).to(torch.uint8)
            if "edge_vis_ratio" in ann and isinstance(ann["edge_vis_ratio"], np.ndarray):
                ann["edge_vis_ratio"] = torch.from_numpy(ann["edge_vis_ratio"]).float()
            if "vis_ratio" in ann and isinstance(ann["vis_ratio"], np.ndarray):
                ann["vis_ratio"] = torch.from_numpy(ann["vis_ratio"]).float()
            if "pnp_cond_score" in ann and isinstance(ann["pnp_cond_score"], np.ndarray):
                ann["pnp_cond_score"] = torch.from_numpy(ann["pnp_cond_score"]).float()
            if "uv" in ann and isinstance(ann["uv"], np.ndarray):
                ann["uv"] = torch.from_numpy(ann["uv"]).float()
            return ann

        if "bbox" in ann:
            b_np = ann["bbox"]
            if isinstance(b_np, np.ndarray):
                b = torch.from_numpy(b_np).float()
            else:
                b = b_np.clone().float()
            b[:, 0:2] *= s
            b[:, 2:4] *= s
            ann["bbox"] = b

        if "uv" in ann:
            uv = ann["uv"]
            if isinstance(uv, np.ndarray):
                ann["uv"] = torch.from_numpy(uv).float()
            elif isinstance(uv, torch.Tensor):
                ann["uv"] = uv.float()

        if "edge_len_px" in ann:
            v_np = ann["edge_len_px"]
            if isinstance(v_np, np.ndarray):
                v = torch.from_numpy(v_np).float()
            else:
                v = v_np.clone().float()
            ann["edge_len_px"] = v * s

        if "full_px" in ann:
            v_np = ann["full_px"]
            if isinstance(v_np, np.ndarray):
                v = torch.from_numpy(v_np).float()
            else:
                v = v_np.clone().float()
            ann["full_px"] = v * (s * s)

        if "visible_px" in ann:
            v_np = ann["visible_px"]
            if isinstance(v_np, np.ndarray):
                v = torch.from_numpy(v_np).float()
            else:
                v = v_np.clone().float()
            ann["visible_px"] = v * (s * s)

        if "visible_edges" in ann:
            if isinstance(ann["visible_edges"], np.ndarray):
                ann["visible_edges"] = torch.from_numpy(ann["visible_edges"]).to(torch.uint8)
            else:
                ann["visible_edges"] = ann["visible_edges"].to(torch.uint8)
        if "visible_vertices" in ann:
            if isinstance(ann["visible_vertices"], np.ndarray):
                ann["visible_vertices"] = torch.from_numpy(ann["visible_vertices"]).to(torch.uint8)
            else:
                ann["visible_vertices"] = ann["visible_vertices"].to(torch.uint8)
        if "edge_vis_ratio" in ann:
            if isinstance(ann["edge_vis_ratio"], np.ndarray):
                ann["edge_vis_ratio"] = torch.from_numpy(ann["edge_vis_ratio"]).float()
            else:
                ann["edge_vis_ratio"] = ann["edge_vis_ratio"].float()
        if "vis_ratio" in ann:
            if isinstance(ann["vis_ratio"], np.ndarray):
                ann["vis_ratio"] = torch.from_numpy(ann["vis_ratio"]).float()
            else:
                ann["vis_ratio"] = ann["vis_ratio"].float()
        if "pnp_cond_score" in ann:
            if isinstance(ann["pnp_cond_score"], np.ndarray):
                ann["pnp_cond_score"] = torch.from_numpy(ann["pnp_cond_score"]).float()
            else:
                ann["pnp_cond_score"] = ann["pnp_cond_score"].float()

        return ann

    def _scale_annotations_anisotropic(
        self,
        ann: Dict[str, Any],
        sx: float,
        sy: float,
    ) -> Dict[str, Any]:
        if "bbox" in ann:
            b = ann["bbox"]
            if isinstance(b, np.ndarray):
                b_t = torch.from_numpy(b).float()
            else:
                b_t = b.clone().float()
            b_t[:, 0] *= sx
            b_t[:, 1] *= sy
            b_t[:, 2] *= sx
            b_t[:, 3] *= sy
            ann["bbox"] = b_t

        if "uv" in ann:
            uv = ann["uv"]
            if isinstance(uv, np.ndarray):
                ann["uv"] = torch.from_numpy(uv).float()
            elif isinstance(uv, torch.Tensor):
                ann["uv"] = uv.float()

        if "edge_len_px" in ann:
            v = ann["edge_len_px"]
            if isinstance(v, np.ndarray):
                v_t = torch.from_numpy(v).float()
            else:
                v_t = v.clone().float()
            ann["edge_len_px"] = v_t * float(0.5 * (sx + sy))

        if "full_px" in ann:
            v = ann["full_px"]
            if isinstance(v, np.ndarray):
                v_t = torch.from_numpy(v).float()
            else:
                v_t = v.clone().float()
            ann["full_px"] = v_t * float(sx * sy)

        if "visible_px" in ann:
            v = ann["visible_px"]
            if isinstance(v, np.ndarray):
                v_t = torch.from_numpy(v).float()
            else:
                v_t = v.clone().float()
            ann["visible_px"] = v_t * float(sx * sy)

        return ann

    def _crop_annotations(
        self,
        ann: Dict[str, Any],
        x0: int,
        y0: int,
        cropped_hw: Tuple[int, int],
        img_hw_before: Tuple[int, int],
    ) -> Dict[str, Any]:
        H_before, W_before = img_hw_before
        H_crop, W_crop = cropped_hw

        if "bbox" in ann:
            b = ann["bbox"]
            if isinstance(b, np.ndarray):
                b_t = torch.from_numpy(b).float()
            else:
                b_t = b.clone().float()
            b_t[:, 0] -= float(x0)
            b_t[:, 1] -= float(y0)
            ann["bbox"] = b_t

        if "uv" in ann:
            uv = ann["uv"]
            if isinstance(uv, np.ndarray):
                uv_t = torch.from_numpy(uv).float()
            else:
                uv_t = uv.clone().float()

            if uv_t.numel() > 0:
                x_pix = uv_t[..., 0] * float(W_before)
                y_pix = uv_t[..., 1] * float(H_before)
                x_pix -= float(x0)
                y_pix -= float(y0)
                uv_t[..., 0] = x_pix / float(W_crop)
                uv_t[..., 1] = y_pix / float(H_crop)
            ann["uv"] = uv_t

        return ann

    def _hflip_annotations(
        self,
        ann: Dict[str, Any],
        W: int,
    ) -> Dict[str, Any]:
        if "bbox" in ann:
            b = ann["bbox"]
            if isinstance(b, np.ndarray):
                b_t = torch.from_numpy(b).float()
            else:
                b_t = b.clone().float()
            x, y, w, h = b_t.unbind(dim=1)
            x_new = (W - (x + w))
            b_t = torch.stack([x_new, y, w, h], dim=1)
            ann["bbox"] = b_t

        if "uv" in ann:
            uv = ann["uv"]
            if isinstance(uv, np.ndarray):
                uv_t = torch.from_numpy(uv).float()
            else:
                uv_t = uv.clone().float()

            if uv_t.numel() > 0:
                uv_t[..., 0] = 1.0 - uv_t[..., 0]
            ann["uv"] = uv_t

        return ann

    def _random_or_center_crop_coords(
        self,
        H: int,
        W: int,
        out_h: int,
        out_w: int,
        random_crop: bool,
    ) -> Tuple[int, int]:
        if H <= out_h or W <= out_w:
            return 0, 0
        if random_crop:
            max_y = H - out_h
            max_x = W - out_w
            y0 = int(torch.randint(0, max_y + 1, (1,)).item())
            x0 = int(torch.randint(0, max_x + 1, (1,)).item())
        else:
            y0 = (H - out_h) // 2
            x0 = (W - out_w) // 2
        return y0, x0

    # ---------------------------------------------------------------------
    # object filters
    # ---------------------------------------------------------------------

    def _filter_objects_by_bbox_center(
        self,
        ann: Dict[str, Any],
        H: int,
        W: int,
    ) -> Dict[str, Any]:
        if "bbox" not in ann:
            return ann

        b = ann["bbox"]
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
        valid_center = (cx >= -m) & (cx < float(W) + m) & (cy >= -m) & (cy < float(H) + m)
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
            uv_t = uv.float()
        elif isinstance(uv, np.ndarray):
            uv_t = torch.from_numpy(uv).float()
        else:
            return ann

        if uv_t.numel() == 0:
            return ann

        if uv_t.ndim == 2 and uv_t.shape[-1] == 2:
            uv_t = uv_t.unsqueeze(1)
        elif uv_t.ndim == 3 and uv_t.shape[-1] == 2:
            pass
        else:
            return ann

        N = uv_t.shape[0]
        x = uv_t[..., 0]
        y = uv_t[..., 1]

        mask_pts = (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)

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

    # ---------------------------------------------------------------------
    # augmentation (geometry)
    # ---------------------------------------------------------------------

    def _sample_scale(self, H: int, W: int) -> float:
        s_min, s_max = self.scale_range
        if not self.augment:
            return 1.0
        if s_max <= 0.0:
            return 1.0
        if s_min <= 0.0:
            s_min = 1e-6
        r = torch.rand(1).item()
        return s_min + (s_max - s_min) * r

    def _apply_geometric_transforms(
        self,
        img: torch.Tensor,
        intr: Optional[np.ndarray | torch.Tensor],
        ann: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        _, H0, W0 = img.shape
        out_hw = self.output_size

        if self.augment:
            s = self._sample_scale(H0, W0)
        else:
            s = 1.0

        H1 = max(1, int(round(H0 * s)))
        W1 = max(1, int(round(W0 * s)))

        if s != 1.0:
            img = self._resize_image(img, (H1, W1))
        else:
            H1, W1 = H0, W0

        K = self._scale_intrinsics(intr, s) if self.include_intrinsics else None
        ann = self._scale_annotations_uniform(ann, s)

        if out_hw is not None:
            out_h, out_w = out_hw

            if H1 <= out_h and W1 <= out_w:
                y0, x0 = 0, 0
                crop_h, crop_w = H1, W1
            else:
                crop_h = min(out_h, H1)
                crop_w = min(out_w, W1)
                y0, x0 = self._random_or_center_crop_coords(
                    H1, W1, crop_h, crop_w, self.random_crop if self.augment else False
                )

            img = img[:, y0 : y0 + crop_h, x0 : x0 + crop_w]
            H2, W2 = crop_h, crop_w

            K = self._crop_intrinsics(K, x0, y0)
            ann = self._crop_annotations(ann, x0, y0, (H2, W2), (H1, W1))
        else:
            H2, W2 = H1, W1

        if out_hw is not None:
            out_h, out_w = out_hw
            sy = out_h / float(H2)
            sx = out_w / float(W2)

            img = self._resize_image(img, (out_h, out_w))
            K = self._scale_intrinsics_anisotropic(K, sx, sy)
            ann = self._scale_annotations_anisotropic(ann, sx, sy)
            H_final, W_final = out_h, out_w
        else:
            H_final, W_final = H2, W2

        if self.augment and self.random_hflip:
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])
                K = self._hflip_intrinsics(K, W_final)
                ann = self._hflip_annotations(ann, W_final)

        return img, K, ann

    # ---------------------------------------------------------------------
    # IO
    # ---------------------------------------------------------------------

    def _load_raw_sample(
        self, h5_path: str, img_idx: int, start: int, end: int
    ) -> Tuple[torch.Tensor, Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any], Tuple[int, int]]:
        with h5py.File(h5_path, "r") as f:
            img_np = np.asarray(f[self.image_key][img_idx])
            intr = (
                np.asarray(f[self.intr_key][img_idx])
                if self.include_intrinsics and self.intr_key in f
                else None
            )
            extr = (
                np.asarray(f[self.extr_key][img_idx])
                if self.include_extrinsics and self.extr_key in f
                else None
            )

            ann_grp = f[self.ann_group]
            ann: Dict[str, Any] = {}
            for key in ann_grp.keys():
                dset = ann_grp[key]
                if dset.dtype.kind == "O":
                    ann[key] = self._read_object_array(dset, start, end)
                else:
                    ann[key] = np.asarray(dset[start:end])

        img = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().float() / 255.0
        H, W = img.shape[1], img.shape[2]
        return img, intr, extr, ann, (H, W)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        h5_path, img_idx, start, count = self.entries[index]
        end = start + count

        img, intr, extr, ann, orig_hw = self._load_raw_sample(
            h5_path, img_idx, start, end
        )

        img, K_scaled, ann = self._apply_geometric_transforms(img, intr, ann)
        H_final, W_final = img.shape[1], img.shape[2]

        img = self._apply_photometric_transforms(img)

        if self.filter is not None:
            ann = self.filter(ann, H_final, W_final)

        if "bbox" in ann and isinstance(ann["bbox"], torch.Tensor):
            obj_count_after = int(ann["bbox"].shape[0])
        else:
            obj_count_after = count

        sample: Dict[str, Any] = {
            "image": img,
            "intrinsics": (
                K_scaled
                if K_scaled is not None
                else (torch.from_numpy(intr).float() if intr is not None else None)
            ),
            "extrinsics": torch.from_numpy(extr).float() if extr is not None else None,
            "ann": ann,
            "meta": {
                "h5_path": h5_path,
                "image_index_in_file": img_idx,
                "obj_start": start,
                "obj_count_raw": count,
                "obj_count": obj_count_after,
                "orig_hw": orig_hw,
                "final_hw": (H_final, W_final),
                "output_size": self.output_size,
                "augment": self.augment,
                "scale_range": self.scale_range,
            },
        }

        return sample


def cuboid_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
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


if __name__ == "__main__":
    root = "dataset"

    ds = CuboidDataset(
        root_dir=root,
        include_intrinsics=True,
        include_extrinsics=True,
        min_objects=0,
        output_size=(384, 640),
        resize_interp="bilinear",
        augment=True,
        scale_range=(0.8, 1.2),
        random_crop=True,
        random_hflip=True,
        object_filter="bbox_center",
        center_margin=0.0,
    )

    print("=== Dataset summary ===")
    print(f"Found {len(ds.h5_paths)} h5 files")
    print(f"Total image samples: {len(ds)}")
    print("Example file paths:")
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
            print(
                f" Sample {bidx}: {meta['h5_path']} "
                f"[img_idx={meta['image_index_in_file']}]"
            )
            print(f"  num_objects: {meta['obj_count']}")
            print(
                f"  orig_hw: {meta['orig_hw']}, "
                f"final_hw: {meta['final_hw']}, "
                f"output_size: {meta['output_size']}, "
                f"augment: {meta['augment']}, "
                f"scale_range: {meta['scale_range']}"
            )
            if "bbox" in ann:
                print(
                    f"  bbox: shape {ann['bbox'].shape}, "
                    f"example {ann['bbox'][:2]}"
                )
            if "uv" in ann:
                print(f"  uv: shape {ann['uv'].shape}, example {ann['uv'][:1]}")
            if "visible_faces" in ann:
                print(f"  visible_faces[0]: {ann['visible_faces'][0]}")
            print()

        break

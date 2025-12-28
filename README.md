# Cuboid-DETR

**Geometry-Driven Cuboid Detection with Symmetry-Aware Keypoints**

---

## Overview

Cuboid-DETR is a geometry-driven object detection framework that jointly predicts **2D bounding boxes and 8 projected cuboid keypoints** from a single RGB image.

The method explicitly incorporates **cuboid geometry, edge structure, and symmetry constraints** into both the model architecture and the loss design. It is built upon DETR-style query-based detection and supports **DINOv2** as a strong visual backbone.

This repository provides a complete pipeline for **training, evaluation, and inference**, together with visualization utilities.

---

## Key Features

* **Joint Cuboid Detection**
  * Simultaneous prediction of bounding boxes and 8 cuboid vertices

* **Coarse-to-Refined Keypoint Estimation**
  * Coarse keypoints are predicted from DETR queries and refined jointly using a transformer encoder

* **Geometry-Aware Feature Sampling**
  * Point-based sampling around keypoints
  * Learnable edge sampling along cuboid edges with attention-based aggregation

* **High-Frequency Image Cues**
  * High-frequency (edge/line-aware) image features are fused into the keypoint feature map

---

## Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate cuboid-detr
```

### Editable installation

```bash
pip install -e .
```

**Requirements**

* Python ≥ 3.10
* PyTorch ≥ 1.13
* CUDA-enabled GPU is strongly recommended
* Automatic Mixed Precision (AMP) is supported

All dependencies are specified in `pyproject.toml`.

---

## Dataset Format (HDF5)

This implementation uses a **custom HDF5-based dataset format**.

### File Discovery

* All `.h5` files under the specified root directory are recursively scanned.
* Each `.h5` file may contain multiple images and annotations.

### Required Structure (simplified)

```text
dataset.h5
├── images                  # (N, H, W, 3), uint8
├── img2obj                 # (N, 2): [start_idx, num_objects]
├── ann/
│   ├── bbox                # (M, 4)  [x, y, w, h] in pixel coordinates
│   ├── uv                  # (M, 8, 2) keypoints in normalized [0,1]
│   ├── visible_vertices    # (M, 8)
│   ├── visible_edges       # (M, E)
│   └── ...
├── intrinsics   (optional)
├── extrinsics   (optional)
└── categories/cuboid_topology/faces (optional)
```

* Bounding boxes are stored in pixel coordinates.
* Keypoints (`uv`) are stored in normalized image coordinates.
* If cuboid face topology is not provided, a default cuboid topology is used.

---

## Synthetic Dataset

The experiments in this repository are conducted using a **synthetic cuboid dataset**.

* The dataset contains rendered images with precise cuboid geometry and keypoint annotations.
* Camera intrinsics and extrinsics can be optionally included.

**The dataset generation project will be released in a separate repository in the near future.**
A link will be added here once it becomes publicly available.

---

## Training

```bash
python -m cbdetr.cli.train \
  --train_root_easy path/to/train_easy \
  --train_root_hard path/to/train_hard \
  --val_root_easy path/to/val_easy \
  --val_root_hard path/to/val_hard \
  --out_dir outputs/exp01
```

### Training Strategy

* Curriculum learning with **easy → hard** splits
* Three-stage training schedule:

  1. Bounding box + classification
  2. Coarse keypoint prediction
  3. Full joint keypoint refinement
* AMP and distributed training are supported
* TensorBoard logging is enabled by default

---

## Evaluation

```bash
python -m cbdetr.cli.eval \
  --val_root_easy path/to/val_easy \
  --val_root_hard path/to/val_hard \
  --weights outputs/exp01/best.pt
```

Evaluation focuses primarily on **keypoint-based losses**, including:

* vertex error
* edge consistency
* face consistency
* repulsion constraints

Inference speed (FPS) can also be measured.

---

## Inference and Visualization

Pretrained weights are available via GitHub Releases.
See the **Pretrained Models** section for details.

```bash
python -m cbdetr.cli.infer \
  --images path/to/images \
  --weights outputs/exp01/best.pt \
  --out_dir results/
```

Outputs include:

* Images with bounding boxes, keypoints, and edges overlaid
* Prediction results saved as JSON
* Optional visualization of decoder cross-attention maps

---

## Pretrained Models

We provide pretrained model weights via [**GitHub Releases**](https://github.com/t-34400/cb-detr/releases).

The released checkpoints are trained with the following configuration:

* `bbox_reparam = True`
* `two_stage = True`
* Automatic Mixed Precision (**AMP**) enabled
* Backbone: **DINOv2**
* Task: 2D cuboid detection with 8 keypoint prediction

### Available Models

| Model              | Backbone | bbox_reparam | two_stage | AMP | Notes                               |
| ------------------ | -------- | ------------ | --------- | --- | ----------------------------------- |
| Cuboid-DETR-DINOv2 | DINOv2   | ✓            | ✓         | ✓   | Trained on synthetic cuboid dataset |

Each release contains:

* model checkpoint (`.pt`)
* configuration summary
* inference-ready weights

Note: At the moment, pretrained weights are provided only for the
bbox-reparameterized, two-stage, AMP-enabled configuration.

---

## Loss Design

Cuboid-DETR employs geometry-aware losses centered on keypoints:

* Keypoint regression loss
* BBox–keypoint consistency loss
* Edge length and direction loss
* Face convexity and area constraints
* Vertex repulsion loss

Losses are applied to both coarse and refined keypoint predictions.

---

## Citation

Citation information will be added once the accompanying paper or technical report is released.

---

## License & Acknowledgements

This project is licensed under the **Apache License 2.0**.

Cuboid-DETR builds upon a number of influential works and open-source projects in object detection and representation learning. We would like to sincerely acknowledge the authors and maintainers of the following projects for making their code and research publicly available:

* [RF-DETR](https://github.com/roboflow/rf-detr)
* [LW-DETR](https://github.com/Atten4Vis/LW-DETR)
* [Conditional DETR](https://github.com/Atten4Vis/ConditionalDETR)
* [DETR](https://github.com/facebookresearch/detr)
* [DINOv2](https://github.com/facebookresearch/dinov2)

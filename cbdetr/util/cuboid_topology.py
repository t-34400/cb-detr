# ------------------------------------------------------------------------
# Cuboid-DETR
# Copyright (c) 2025 t-34400
# Licensed under the Apache License, Version 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------

import torch

# ---------- topology ----------
# 0: back-bottom-left
# 1: back-bottom-right
# 2: back-top-right
# 3: back-top-left
# 4: front-bottom-left
# 5: front-bottom-right
# 6: front-top-right
# 7: front-top-left

EDGES = torch.tensor([
    [0,1],[1,2],[2,3],[3,0],   # back face
    [4,5],[5,6],[6,7],[7,4],   # front face
    [0,4],[1,5],[2,6],[3,7],   # depth edges
], dtype=torch.long)

PARALLEL_GROUPS = [
    torch.tensor([[0,1],[3,2],[4,5],[7,6]], dtype=torch.long),  # x-like (left→right)
    torch.tensor([[0,3],[1,2],[4,7],[5,6]], dtype=torch.long),  # y-like (bottom→top)
    torch.tensor([[0,4],[1,5],[2,6],[3,7]], dtype=torch.long),  # z-like (back→front)
]

FACES = torch.tensor([
    [0,3,2,1],  # back
    [4,5,6,7],  # front
    [0,4,7,3],  # left
    [1,2,6,5],  # right
    [0,1,5,4],  # bottom
    [3,7,6,2],  # top
], dtype=torch.long)
# ------------------------------------------------------------------------
# Cuboid-DETR
# Copyright (c) 2025 t-34400
# Licensed under the Apache License, Version 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------

import argparse
from dataclasses import dataclass
import torch

from .backbone import build_backbone
from .transformer import build_transformer
from cbdetr.models.cbdetr import CuboidDetr


@dataclass
class TransformerArgs:
    d_model: int = 256
    sa_nheads: int = 8
    ca_nheads: int = 16
    num_queries: int = 100
    dropout: float = 0.1
    dim_feedforward: int = 2048
    dec_layers: int = 3
    return_intermediate_dec: bool = True
    group_detr: int = 4
    two_stage: bool = False
    num_feature_levels: int = 4
    dec_n_points: int = 4
    lite_refpoint_refine: bool = False
    decoder_norm: str = 'LN'
    bbox_reparam: bool = False

    @property
    def hidden_dim(self) -> int:
        return self.d_model


@dataclass
class ModelArgs:
    encoder: str = 'dinov2_windowed_small'
    d_model: int = 256
    out_feature_indexes: tuple[int, ...] = (3, 6, 9, 12)
    projector_scale: tuple[str, ...] = ("P4",)
    position_embedding: str = 'sine'
    freeze_encoder: bool = False
    layer_norm: bool = False
    target_shape: tuple[int, int] = (640, 640)
    rms_norm: bool = False
    gradient_checkpointing: bool = False
    load_dinov2_weights: bool = True
    patch_size: int = 14
    num_windows: int = 2
    positional_encoding_size: int = 37
    aux_loss: bool = True
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_model_transformer_args(parser: argparse.ArgumentParser):
    group_m = parser.add_argument_group("Model Arguments")
    group_m.add_argument("--encoder", type=str, default='dinov2_windowed_small')
    group_m.add_argument("--d_model", type=int, default=256)
    group_m.add_argument("--out_feature_indexes", type=int, nargs='+', default=[3, 6, 9, 12])
    group_m.add_argument("--projector_scale", type=str, nargs='+', default=["P4"])
    group_m.add_argument("--position_embedding", type=str, default='sine')
    group_m.add_argument("--freeze_encoder", action='store_true')
    group_m.add_argument("--layer_norm", action='store_true')
    group_m.add_argument("--target_shape", type=int, nargs=2, default=[640, 640])
    group_m.add_argument("--rms_norm", action='store_true')
    group_m.add_argument("--gradient_checkpointing", action='store_true')
    group_m.add_argument("--load_dinov2_weights", action='store_true')
    group_m.add_argument("--patch_size", type=int, default=14)
    group_m.add_argument("--num_windows", type=int, default=2)
    group_m.add_argument("--positional_encoding_size", type=int, default=37)
    group_m.add_argument("--aux_loss", action='store_true')

    group_t = parser.add_argument_group("Transformer Arguments")
    group_t.add_argument("--t_d_model", type=int, default=256)
    group_t.add_argument("--sa_nheads", type=int, default=8)
    group_t.add_argument("--ca_nheads", type=int, default=16)
    group_t.add_argument("--num_queries", type=int, default=30)
    group_t.add_argument("--dropout", type=float, default=0.1)
    group_t.add_argument("--dim_feedforward", type=int, default=2048)
    group_t.add_argument("--dec_layers", type=int, default=3)
    group_t.add_argument("--return_intermediate_dec", action='store_true')
    group_t.add_argument("--group_detr", type=int, default=1)
    group_t.add_argument("--two_stage", action='store_true')
    group_t.add_argument("--num_feature_levels", type=int, default=4)
    group_t.add_argument("--dec_n_points", type=int, default=4)
    group_t.add_argument("--lite_refpoint_refine", action='store_true')
    group_t.add_argument("--decoder_norm", type=str, default='LN')
    group_t.add_argument("--bbox_reparam", action='store_true')

    return parser


def parse_model_transformer_args(args, device):
    margs = ModelArgs(
        encoder=args.encoder,
        d_model=args.d_model,
        out_feature_indexes=args.out_feature_indexes,
        projector_scale=args.projector_scale,
        position_embedding=args.position_embedding,
        freeze_encoder=args.freeze_encoder,
        layer_norm=args.layer_norm,
        target_shape=tuple(args.target_shape),
        rms_norm=args.rms_norm,
        gradient_checkpointing=args.gradient_checkpointing,
        load_dinov2_weights=args.load_dinov2_weights,
        patch_size=args.patch_size,
        num_windows=args.num_windows,
        positional_encoding_size=args.positional_encoding_size,
        aux_loss=args.aux_loss,
        device=device
    )

    targs = TransformerArgs(
        d_model=args.t_d_model,
        sa_nheads=args.sa_nheads,
        ca_nheads=args.ca_nheads,
        num_queries=args.num_queries,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        dec_layers=args.dec_layers,
        return_intermediate_dec=args.return_intermediate_dec,
        group_detr=args.group_detr,
        two_stage=args.two_stage,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        lite_refpoint_refine=args.lite_refpoint_refine,
        decoder_norm=args.decoder_norm,
        bbox_reparam=args.bbox_reparam
    )

    return margs, targs


def build_model(args: ModelArgs, transformer_args: TransformerArgs):
    device = torch.device(args.device)

    backbone = build_backbone(
        encoder=args.encoder,
        vit_encoder_num_layers=None, # Not used
        pretrained_encoder=None,     # Not used
        window_block_indexes=None,   # Not used
        drop_path=None,              # Not used
        out_channels=args.d_model,
        out_feature_indexes=args.out_feature_indexes,
        projector_scale=args.projector_scale,
        use_cls_token=None,          # Not used
        hidden_dim=args.d_model,
        position_embedding=args.position_embedding,
        freeze_encoder=args.freeze_encoder,
        layer_norm=args.layer_norm,
        target_shape=args.target_shape,
        rms_norm=args.rms_norm,
        backbone_lora=False,
        force_no_pretrain=None,       # Not used
        gradient_checkpointing=args.gradient_checkpointing,
        load_dinov2_weights=args.load_dinov2_weights,
        patch_size=args.patch_size,
        num_windows=args.num_windows,
        positional_encoding_size=args.positional_encoding_size,
    )

    transformer_args.num_feature_levels = len(args.projector_scale)
    transformer_args.d_model = args.d_model

    transformer = build_transformer(transformer_args)

    model = CuboidDetr(
        backbone=backbone,
        transformer=transformer,
        num_queries=transformer.num_queries,
        group_detr=transformer_args.group_detr,
        aux_loss=args.aux_loss,
        lite_refpoint_refine=transformer_args.lite_refpoint_refine,
        bbox_reparam=transformer_args.bbox_reparam
    )
    model.to(device=device)

    return model
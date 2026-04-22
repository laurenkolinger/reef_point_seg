#!/usr/bin/env python3
"""Thin wrapper around Ultralytics' YOLO segmentation training.

Mirrors oceankind_CV/training/train_segment.py hyperparameters but exposes
every knob as a CLI argument so the orchestrator UI can forward user-tuned
values. Defaults are taken from train_segment.py so unchanged runs give
identical results to upstream.

`project` is accepted explicitly so training runs land inside the pipeline's
step-6 project folder instead of next to cwd (upstream hardcodes 'OK_CV').
"""

import argparse
import os
import sys

from ultralytics import YOLO


def arg_parse():
    p = argparse.ArgumentParser(description='Train YOLO segmentation model (TCRMP step 6)')

    # --- required plumbing ------------------------------------------------
    p.add_argument('--src', required=True, help='Path to data.yaml from the split step')
    p.add_argument('--name', required=True, help='Training run name (subfolder of --project)')
    p.add_argument('--project', required=True, help='Root directory for training runs')

    # --- core training ----------------------------------------------------
    p.add_argument('--model', default='yolo11m-seg.pt',
                   help='Base weights. e.g. yolo11{n,s,m,l,x}-seg.pt')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--batch', type=int, default=-1,
                   help='-1 = ultralytics auto-batch based on GPU memory')
    p.add_argument('--patience', type=int, default=50,
                   help='Early-stop: epochs with no improvement before stopping')
    p.add_argument('--device', default='0', help='CUDA device(s) — "0", "0,1", "cpu"')
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--optimizer', default='auto',
                   help='auto | SGD | Adam | AdamW | RMSProp')
    p.add_argument('--seed', type=int, default=0)

    # --- LR schedule ------------------------------------------------------
    p.add_argument('--lr0', type=float, default=0.01,
                   help='initial LR (SGD=1e-2, Adam=1e-3)')
    p.add_argument('--lrf', type=float, default=0.01,
                   help='final LR = lr0 * lrf')
    p.add_argument('--momentum', type=float, default=0.937)
    p.add_argument('--weight_decay', type=float, default=0.0005)
    p.add_argument('--warmup_epochs', type=float, default=3.0)
    p.add_argument('--warmup_momentum', type=float, default=0.8)
    p.add_argument('--warmup_bias_lr', type=float, default=0.1)

    # --- loss weights -----------------------------------------------------
    p.add_argument('--box', type=float, default=7.5)
    p.add_argument('--cls', type=float, default=0.5)
    p.add_argument('--dfl', type=float, default=1.5)
    p.add_argument('--pose', type=float, default=12.0)
    p.add_argument('--kobj', type=float, default=1.0)
    p.add_argument('--label_smoothing', type=float, default=0.0)
    p.add_argument('--nbs', type=int, default=64,
                   help='nominal batch size (used for scaling loss)')

    # --- color / hue augmentation (most impactful for underwater imagery) -
    p.add_argument('--hsv_h', type=float, default=0.2, help='image HSV-Hue jitter (fraction). Underwater tip: 0.02-0.05 to cover color-cast variation.')
    p.add_argument('--hsv_s', type=float, default=0.3, help='image HSV-Saturation jitter. Underwater tip: 0.6-0.9.')
    p.add_argument('--hsv_v', type=float, default=0.3, help='image HSV-Value/brightness jitter. Underwater tip: 0.5-0.6 for depth/lighting variation.')
    p.add_argument('--bgr', type=float, default=0.0, help='P(channel BGR<->RGB swap). Leave 0 unless the dataset mixes both conventions.')

    # --- geometric augmentation -------------------------------------------
    p.add_argument('--degrees', type=float, default=0.0, help='rotation +/- deg')
    p.add_argument('--translate', type=float, default=0.0, help='translation +/- fraction')
    p.add_argument('--scale', type=float, default=0.2, help='scale +/- gain')
    p.add_argument('--shear', type=float, default=0.0, help='shear +/- deg')
    p.add_argument('--perspective', type=float, default=0.0,
                   help='perspective (0-0.001 typical)')
    p.add_argument('--flipud', type=float, default=0.5, help='P(flip up-down)')
    p.add_argument('--fliplr', type=float, default=0.5, help='P(flip left-right)')

    # --- sample-mixing augmentation ---------------------------------------
    p.add_argument('--mosaic', type=float, default=0.0, help='P(mosaic)')
    p.add_argument('--mixup', type=float, default=0.0, help='P(mixup)')
    p.add_argument('--cutmix', type=float, default=0.0, help='P(cutmix). New-ish (8.4.x). 0.0-0.1 typical for seg.')
    p.add_argument('--copy_paste', type=float, default=0.0,
                   help='P(copy-paste seg augmentation). Very effective for rare classes in sparse coral data — try 0.1-0.3.')
    p.add_argument('--copy_paste_mode', default='flip',
                   help='copy_paste strategy: flip | mixup')
    p.add_argument('--erasing', type=float, default=0.0,
                   help='P(random erasing). Classification-only; ignored for seg in current ultralytics.')
    p.add_argument('--auto_augment', default='randaugment',
                   help='AutoAugment policy (classification-only in ultralytics): randaugment | autoaugment | augmix | None')

    # --- misc -------------------------------------------------------------
    p.add_argument('--cos_lr', type=str, default='false',
                   help='use cosine LR scheduler (true/false)')
    p.add_argument('--close_mosaic', type=int, default=10,
                   help='epochs before end to disable mosaic (improves final accuracy)')

    return p.parse_args()


def _as_bool(s):
    return str(s).strip().lower() in ('1', 'true', 't', 'yes', 'y')


def main():
    args = arg_parse()
    if not os.path.isfile(args.src):
        print(f'ERROR: data yaml not found: {args.src}', file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.project, exist_ok=True)

    # Echo the knobs that matter so the run is reproducible from logs alone.
    print(f'[train] data={args.src}')
    print(f'[train] project={args.project} name={args.name}')
    print(f'[train] model={args.model} epochs={args.epochs} imgsz={args.imgsz} '
          f'batch={args.batch} optimizer={args.optimizer}')
    print(f'[train] HSV hue={args.hsv_h} sat={args.hsv_s} val={args.hsv_v}')
    print(f'[train] geom rot={args.degrees} trans={args.translate} scale={args.scale} '
          f'shear={args.shear} persp={args.perspective} flipud={args.flipud} fliplr={args.fliplr}')
    print(f'[train] mix mosaic={args.mosaic} mixup={args.mixup} '
          f'copy_paste={args.copy_paste} erase={args.erasing} auto_aug={args.auto_augment}')

    kwargs = dict(
        data=args.src,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        save=True,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        val=True,
        optimizer=args.optimizer,
        seed=args.seed,
        cos_lr=_as_bool(args.cos_lr),
        close_mosaic=args.close_mosaic,
        # LR
        lr0=args.lr0, lrf=args.lrf,
        momentum=args.momentum, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        # loss
        box=args.box, cls=args.cls, dfl=args.dfl,
        pose=args.pose, kobj=args.kobj,
        label_smoothing=args.label_smoothing, nbs=args.nbs,
        # color
        hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v, bgr=args.bgr,
        # geometric
        degrees=args.degrees, translate=args.translate, scale=args.scale,
        shear=args.shear, perspective=args.perspective,
        flipud=args.flipud, fliplr=args.fliplr,
        # mixing
        mosaic=args.mosaic, mixup=args.mixup, cutmix=args.cutmix,
        copy_paste=args.copy_paste, copy_paste_mode=args.copy_paste_mode,
        erasing=args.erasing,
        auto_augment=args.auto_augment if args.auto_augment and args.auto_augment.lower() != 'none' else None,
    )

    model = YOLO(args.model)
    model.train(**kwargs)


if __name__ == '__main__':
    main()

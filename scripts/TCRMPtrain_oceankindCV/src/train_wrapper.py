#!/usr/bin/env python3
"""Thin wrapper around Ultralytics' YOLO training, segmentation or detection.

Mirrors oceankind_CV/training/train_segment.py hyperparameters but exposes
every knob as a CLI argument so the orchestrator UI can forward user-tuned
values. Defaults are taken from train_segment.py so unchanged runs give
identical results to upstream.

--task picks what is trained: segment (the default, unchanged behaviour) or
detect. The task changes exactly two things, the base weights (yolo11<size>.pt
instead of yolo11<size>-seg.pt) and the task handed to the YOLO constructor.
Every other knob, augmentation included, is task-agnostic and is forwarded the
same way for both.

The weights are what really decide the head: ultralytics reads the task of a
.pt checkpoint from the checkpoint and ignores the task it is handed (only a
.yaml config takes the task from the caller). --task is therefore the name of
the intent, and check_model_task() refuses the pairing that would quietly
train the wrong head.

`project` is accepted explicitly so training runs are written inside the
pipeline's step-6 project folder instead of next to cwd (upstream hardcodes
'OK_CV').
"""

import argparse
import os
import re
import sys

from ultralytics import YOLO

# What ultralytics calls each task, keyed by every spelling this platform uses.
_TASKS = {'segment': 'segment', 'seg': 'segment', 'segmentation': 'segment',
          'detect': 'detect', 'detection': 'detect', 'bbox': 'detect'}

# GPU traffic (docs/superpowers/specs/2026-09-28-gpu-traffic-design.md,
# section 7): when VICARIUS starts this training under the "vega" GPU
# right-of-way tier, stage_runner.py sets this environment variable to the
# CUDA device string for the card(s) it assigned, in place of --device
# (including a DDP device list such as "0,1"), because Ultralytics is the
# thing that reads CUDA_VISIBLE_DEVICES from its own --device string, and
# VICARIUS's per-card assignment may span both cards without ever setting
# that variable itself. Unset or empty (every run before this variable
# existed, and any run a build that predates it still launches): this script
# behaves exactly as it did before, training on --device as given.
GPU_DEVICE_ENV = 'VICARIUS_GPU_DEVICE'

# A CUDA device string: one or more non-negative whole numbers, comma
# separated (Ultralytics' own --device grammar: "0" for one card, "0,1" for
# DDP across two). Anything else is a malformed assignment and is refused
# rather than silently ignored or passed through to a training run that
# would then pick its own card.
# Matched with fullmatch(), never match(): match() alone would accept a
# trailing newline ("1\n"), which fullmatch refuses along with anything
# else the pattern does not account for.
_DEVICE_STRING_RE = re.compile(r'\d+(,\d+)*')


def resolve_device_override(cli_device, env_value):
    """The --device value to train with, honoring a VICARIUS GPU traffic override.

    Parameters:
        cli_device: the value --device was parsed to (or its default).
        env_value: os.environ.get(GPU_DEVICE_ENV), VICARIUS's per-card
            assignment for this run, or '' / None when it was not set.

    Returns:
        cli_device unchanged when env_value is falsy (unset or empty): this
        is the only path a run started before GPU_DEVICE_ENV existed takes,
        so its behaviour never changes. Otherwise env_value itself, once
        validated as a CUDA device string.

    Raises:
        ValueError naming GPU_DEVICE_ENV and the bad value when env_value is
        set to something that is not a comma-separated list of non-negative
        whole numbers: a typo'd assignment must fail loudly rather than
        quietly train on the wrong card, on no card restriction at all, or
        crash deep inside Ultralytics with a confusing message.
    """
    if not env_value:
        return cli_device
    if not _DEVICE_STRING_RE.fullmatch(env_value):
        raise ValueError(
            f'{GPU_DEVICE_ENV}={env_value!r} is not a valid CUDA device string '
            "(expected whole numbers separated by commas, e.g. '0' or '0,1'); "
            'refusing to guess which card to train on')
    return env_value


def apply_gpu_device_override(args):
    """Replace args.device with VICARIUS's per-card assignment, when given.

    Reads GPU_DEVICE_ENV once and mutates args.device in place; prints one
    line naming the override when it applies. Unset or empty, this makes no
    change and prints nothing, so a run with no such variable set is
    byte-for-byte the run this script made before GPU_DEVICE_ENV existed.

    Parameters:
        args: the parsed command line (argparse.Namespace); args.device may
            be replaced.

    Returns:
        None.

    Raises:
        ValueError: see resolve_device_override.
    """
    env_device = os.environ.get(GPU_DEVICE_ENV, '')
    if not env_device:
        return
    original = args.device
    args.device = resolve_device_override(args.device, env_device)
    print(f'[train] {GPU_DEVICE_ENV} overrides --device: training on '
          f'device={args.device} (was {original})')


def yolo_task(task):
    """The ultralytics task name for a task as the orchestrator spells it,
    case-insensitively. An unrecognized value reads as segment here; the CLI
    refuses it outright (see _task_arg), so no run silently gets the wrong
    task from a typo."""
    return _TASKS.get(str(task or '').strip().lower(), 'segment')


def _task_arg(value):
    """argparse type for --task: normalize the spelling and the case, and
    refuse anything else. Case matters because the caller is a shell env var:
    plain `choices=` would reject "Detect" AFTER run_step6.sh had already
    written dataset_detect/."""
    key = str(value).strip().lower()
    if key not in _TASKS:
        raise argparse.ArgumentTypeError(
            f"unknown task {value!r}; use segment or detect")
    return _TASKS[key]


def default_model_for(task, size='m'):
    """Base weights for a task when the caller named none.

    yolo11n/s/m are staged next to this tool for BOTH tasks, so the default
    (m) never downloads. The l and x sizes are staged for segmentation only,
    so a detection run at those sizes has ultralytics fetch the file, which
    needs the box to reach the network."""
    return f'yolo11{size}.pt' if yolo_task(task) == 'detect' else f'yolo11{size}-seg.pt'


def resolve_model(args):
    """The weights to train from: whatever --model named, else the task's
    default. An explicit path always wins, which is how a fine-tune continues
    from an earlier run's best.pt."""
    return args.model or default_model_for(args.task)


def check_model_task(model, task):
    """'' when the weights suit the task, else why they do not.

    This matters more than it looks: ultralytics reads the task of a .pt
    checkpoint from the checkpoint itself and ignores the `task` it is handed
    (Model._load), so a mismatched pair does not fail, it silently trains the
    OTHER head and writes it into this run's folder. Only the file name is
    evidence, so neutral names (best.pt, last.pt) are left to ultralytics.

    The check runs one way on purpose. A detection run handed -seg weights is
    always a mistake and is refused. The reverse (a segmentation task named
    over stock detection weights) is how the model lab drives a detection lab
    today, which passes detection weights without setting a task: refusing it
    would break a path that trains the right model. Once every caller sends
    TCRMP_STEP6_TASK, the check can run both ways."""
    base = os.path.basename(str(model or ''))
    if yolo_task(task) == 'detect' and '-seg' in base:
        return (f'task=detect cannot train from segmentation weights ({base}). '
                'Pick a detection base model, or fine-tune from a detection run.')
    return ''


def arg_parse():
    p = argparse.ArgumentParser(description='Train a YOLO model (TCRMP step 6)')

    # --- required plumbing ------------------------------------------------
    p.add_argument('--src', required=True, help='Path to data.yaml from the split step')
    p.add_argument('--name', required=True, help='Training run name (subfolder of --project)')
    p.add_argument('--project', required=True, help='Root directory for training runs')

    # --- core training ----------------------------------------------------
    p.add_argument('--task', default='segment', type=_task_arg,
                   help='What to train: segment (default) or detect, in any case. Sets the '
                        'base weights and the ultralytics task; every other knob is '
                        'task-agnostic.')
    p.add_argument('--model', default='',
                   help='Base weights. Unset uses the task default: yolo11m-seg.pt for '
                        'segment, yolo11m.pt for detect.')
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
    p.add_argument('--freeze', type=int, default=None,
                   help='Number of leading layers to freeze for fine-tuning from previous '
                        'weights. Unset = ultralytics default (no freezing).')

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


def build_train_kwargs(args):
    """Assemble the kwargs dict passed to model.train(**kwargs).

    Pulled out of main() so it can be unit-tested without a GPU or a real
    YOLO model load. Any argparse.Namespace with the same attributes as
    arg_parse() produces works (e.g. a hand-built one in tests).
    """
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
    if args.freeze is not None:
        kwargs['freeze'] = args.freeze
    return kwargs


def main():
    args = arg_parse()
    apply_gpu_device_override(args)
    if not os.path.isfile(args.src):
        print(f'ERROR: data yaml not found: {args.src}', file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.project, exist_ok=True)

    task = yolo_task(args.task)
    model_path = resolve_model(args)
    mismatch = check_model_task(model_path, task)
    if mismatch:
        print(f'ERROR: {mismatch}', file=sys.stderr)
        sys.exit(2)

    # Echo the knobs that matter so the run is reproducible from logs alone.
    print(f'[train] data={args.src}')
    print(f'[train] project={args.project} name={args.name}')
    print(f'[train] task={task} model={model_path} epochs={args.epochs} imgsz={args.imgsz} '
          f'batch={args.batch} optimizer={args.optimizer}')
    print(f'[train] HSV hue={args.hsv_h} sat={args.hsv_s} val={args.hsv_v}')
    print(f'[train] geom rot={args.degrees} trans={args.translate} scale={args.scale} '
          f'shear={args.shear} persp={args.perspective} flipud={args.flipud} fliplr={args.fliplr}')
    print(f'[train] mix mosaic={args.mosaic} mixup={args.mixup} '
          f'copy_paste={args.copy_paste} erase={args.erasing} auto_aug={args.auto_augment}')
    if args.freeze is not None:
        print(f'[train] freeze={args.freeze}')

    kwargs = build_train_kwargs(args)

    model = YOLO(model_path, task=task)
    model.train(**kwargs)


if __name__ == '__main__':
    main()

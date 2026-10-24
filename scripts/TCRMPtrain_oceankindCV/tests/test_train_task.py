"""Unit tests for train_wrapper.py task support (segment | detect).

Training needs a GPU, so this exercises only the pure paths: arg_parse() ->
resolve_model() / yolo_task() / check_model_task() / build_train_kwargs().
No YOLO model is loaded and nothing is trained.

The contract under test:
  * --task defaults to segment, and a segmentation run composes EXACTLY the
    kwargs it composed before detection existed (the frozen dict below).
  * detect swaps only the base weights and the ultralytics task; every other
    knob, augmentation included, is task-agnostic and forwarded identically.
  * an explicit --model always wins over the task default.
  * a detection run refuses segmentation weights by name, because ultralytics
    accepts the file and then fails deep inside the training loop.

Run: REPO/env/bin/python scripts/TCRMPtrain_oceankindCV/tests/test_train_task.py
"""
# Keep this suite off the live mask registry, expert library and clip index
# when it is run alone: scripts/tests/_isolation.py sets the isolating
# variables to temp dirs and blocks any write to the live stores.
import os as _iso_os
import sys as _iso_sys
_iso_dir = _iso_os.path.dirname(_iso_os.path.abspath(__file__))
while _iso_os.path.basename(_iso_dir) != 'scripts' and _iso_dir != _iso_os.path.dirname(_iso_dir):
    _iso_dir = _iso_os.path.dirname(_iso_dir)
_iso_sys.path.insert(0, _iso_os.path.join(_iso_dir, 'tests'))
import _isolation
_isolation.install()

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import train_wrapper as TW  # noqa: E402

_fail = 0


def check(cond, msg):
    global _fail
    if not cond:
        _fail += 1
        print(f"FAIL: {msg}")
    else:
        print(f"PASS: {msg}")


def _parse(argv):
    old_argv = sys.argv
    try:
        sys.argv = ["train_wrapper.py"] + argv
        return TW.arg_parse()
    finally:
        sys.argv = old_argv


base_argv = [
    "--src", "/tmp/fake_data.yaml",
    "--name", "run1",
    "--project", "/tmp/fake_project",
]

# ---- task parsing --------------------------------------------------------
args_default = _parse(base_argv)
check(args_default.task == "segment", f"--task defaults to segment (got {args_default.task!r})")
args_detect = _parse(base_argv + ["--task", "detect"])
check(args_detect.task == "detect", "--task detect parses")
check(TW.yolo_task(args_default.task) == "segment" and TW.yolo_task(args_detect.task) == "detect",
      "yolo_task hands ultralytics its own task names")
check(TW.yolo_task("seg") == "segment", "the short form seg is accepted as segment")
# The caller is a shell env var, so case and spelling are normalized by the
# parser itself. A plain choices= list would reject "Detect" after the driver
# had already written dataset_detect/.
check(_parse(base_argv + ["--task", "Detect"]).task == "detect",
      "--task Detect normalizes to detect")
check(_parse(base_argv + ["--task", "BBOX"]).task == "detect"
      and _parse(base_argv + ["--task", "Segmentation"]).task == "segment",
      "every accepted spelling parses to one of the two canonical names")
try:
    # argparse prints its usage on the way out; keep it off the suite's output.
    _devnull = open(os.devnull, "w")
    _old_stderr, sys.stderr = sys.stderr, _devnull
    try:
        _parse(base_argv + ["--task", "nonsense"])
        check(False, "an unknown task is refused by the parser")
    finally:
        sys.stderr = _old_stderr
        _devnull.close()
except SystemExit:
    check(True, "an unknown task is refused by the parser")

# ---- base weights --------------------------------------------------------
check(TW.resolve_model(args_default) == "yolo11m-seg.pt",
      f"a segmentation run keeps the historical default weights (got {TW.resolve_model(args_default)!r})")
check(TW.resolve_model(args_detect) == "yolo11m.pt",
      f"a detection run defaults to the detection weights (got {TW.resolve_model(args_detect)!r})")
for size in ("n", "s", "m", "l", "x"):
    check(TW.default_model_for("detect", size) == f"yolo11{size}.pt"
          and TW.default_model_for("segment", size) == f"yolo11{size}-seg.pt",
          f"size {size} names both task defaults")
explicit = _parse(base_argv + ["--task", "detect", "--model", "/abs/path/best.pt"])
check(TW.resolve_model(explicit) == "/abs/path/best.pt",
      "an explicit --model overrides the task default (fine-tune from earlier weights)")
explicit_seg = _parse(base_argv + ["--model", "yolo11n-seg.pt"])
check(TW.resolve_model(explicit_seg) == "yolo11n-seg.pt",
      "an explicit --model still wins on the segmentation path")

# ---- task and weights must agree ----------------------------------------
check(TW.check_model_task("yolo11m-seg.pt", "detect") != "",
      "a detection run refuses segmentation weights")
check(TW.check_model_task("yolo11m.pt", "detect") == "",
      "detection weights pass on a detection run")
check(TW.check_model_task("/labs/x/runs/r1/weights/best.pt", "detect") == "",
      "weights with a neutral name are left to ultralytics to judge")
check(TW.check_model_task("yolo11m-seg.pt", "segment") == "",
      "the segmentation path is unchanged")
# The check runs one way on purpose: the model lab drives a detection lab with
# detection weights and no task, so refusing that pairing would break a path
# that trains the right model. See check_model_task's docstring.
check(TW.check_model_task("yolo11m.pt", "segment") == "",
      "detection weights under the default task are left alone, which is how the "
      "model lab trains a detection lab today")

# ---- every knob, both tasks ---------------------------------------------
# One distinct value per knob so a mis-wired forward cannot pass by accident.
KNOBS = [
    ("--epochs", "7", "epochs", 7),
    ("--imgsz", "640", "imgsz", 640),
    ("--batch", "8", "batch", 8),
    ("--patience", "11", "patience", 11),
    ("--device", "0,1", "device", "0,1"),
    ("--workers", "3", "workers", 3),
    ("--optimizer", "AdamW", "optimizer", "AdamW"),
    ("--seed", "42", "seed", 42),
    ("--freeze", "10", "freeze", 10),
    ("--lr0", "0.005", "lr0", 0.005),
    ("--lrf", "0.02", "lrf", 0.02),
    ("--momentum", "0.9", "momentum", 0.9),
    ("--weight_decay", "0.001", "weight_decay", 0.001),
    ("--warmup_epochs", "4.0", "warmup_epochs", 4.0),
    ("--warmup_momentum", "0.7", "warmup_momentum", 0.7),
    ("--warmup_bias_lr", "0.2", "warmup_bias_lr", 0.2),
    ("--box", "8.5", "box", 8.5),
    ("--cls", "0.6", "cls", 0.6),
    ("--dfl", "1.6", "dfl", 1.6),
    ("--pose", "13.0", "pose", 13.0),
    ("--kobj", "1.1", "kobj", 1.1),
    ("--label_smoothing", "0.05", "label_smoothing", 0.05),
    ("--nbs", "32", "nbs", 32),
    ("--hsv_h", "0.03", "hsv_h", 0.03),
    ("--hsv_s", "0.7", "hsv_s", 0.7),
    ("--hsv_v", "0.55", "hsv_v", 0.55),
    ("--bgr", "0.1", "bgr", 0.1),
    ("--degrees", "15.0", "degrees", 15.0),
    ("--translate", "0.12", "translate", 0.12),
    ("--scale", "0.35", "scale", 0.35),
    ("--shear", "2.0", "shear", 2.0),
    ("--perspective", "0.0005", "perspective", 0.0005),
    ("--flipud", "0.25", "flipud", 0.25),
    ("--fliplr", "0.75", "fliplr", 0.75),
    ("--mosaic", "0.4", "mosaic", 0.4),
    ("--mixup", "0.05", "mixup", 0.05),
    ("--cutmix", "0.06", "cutmix", 0.06),
    ("--copy_paste", "0.2", "copy_paste", 0.2),
    ("--copy_paste_mode", "mixup", "copy_paste_mode", "mixup"),
    ("--erasing", "0.15", "erasing", 0.15),
    ("--auto_augment", "augmix", "auto_augment", "augmix"),
    ("--cos_lr", "true", "cos_lr", True),
    ("--close_mosaic", "5", "close_mosaic", 5),
]
full_argv = []
for flag, value, _key, _expected in KNOBS:
    full_argv += [flag, value]

kw_seg = TW.build_train_kwargs(_parse(base_argv + full_argv))
kw_det = TW.build_train_kwargs(_parse(base_argv + full_argv + ["--task", "detect"]))
for _flag, _value, key, expected in KNOBS:
    check(kw_seg.get(key) == expected,
          f"{key} forwarded on the segmentation path (got {kw_seg.get(key)!r})")
check(kw_seg == kw_det,
      "every knob composes identically for detect: the task changes the weights and the "
      "ultralytics task, never a hyperparameter")
check("task" not in kw_seg, "task is passed to the YOLO constructor, never into train kwargs")

# ---- the segmentation default is byte-for-byte what it always was --------
FROZEN_SEG_KWARGS = {
    "data": "/tmp/fake_data.yaml",
    "epochs": 500,
    "patience": 50,
    "batch": -1,
    "imgsz": 512,
    "save": True,
    "device": "0",
    "workers": 12,
    "project": "/tmp/fake_project",
    "name": "run1",
    "val": True,
    "optimizer": "auto",
    "seed": 0,
    "cos_lr": False,
    "close_mosaic": 10,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "pose": 12.0,
    "kobj": 1.0,
    "label_smoothing": 0.0,
    "nbs": 64,
    "hsv_h": 0.2,
    "hsv_s": 0.3,
    "hsv_v": 0.3,
    "bgr": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.2,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.5,
    "fliplr": 0.5,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "copy_paste_mode": "flip",
    "erasing": 0.0,
    "auto_augment": "randaugment",
}
kw_base = TW.build_train_kwargs(args_default)
check(kw_base == FROZEN_SEG_KWARGS,
      "a default segmentation run composes exactly the kwargs it did before detection existed")
check(TW.build_train_kwargs(args_detect) == FROZEN_SEG_KWARGS,
      "a default detection run composes the same kwargs: only the weights and the task differ")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)

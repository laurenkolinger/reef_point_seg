#!/usr/bin/env python
"""run_step6.sh task plumbing (segment | detect), driven with a stub python.

The driver is exercised for real (bash run_step6.sh ...), but TCRMP_STEP6_PYTHON
points at a stub that records its argv and prints a path instead of running
anything: no split runs, no model loads, no GPU is touched, no port is bound.

Covers:
  syntax          bash -n over the driver
  segment default the composed train_wrapper argv is unchanged, and no
                  detection dataset is prepared
  detect          the driver prepares dataset_detect via detect_dataset.py and
                  trains on the data.yaml that prepare printed
  base weights    task default, positional override, TCRMP_STEP6_MODEL_PATH
  knob forwarding every TCRMP_STEP6_* knob reaches train_wrapper identically
                  for both tasks
  all mode        split then train, with the task carried through

Run: REPO/env/bin/python scripts/TCRMPtrain_oceankindCV/tests/test_run_step6_task.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOL = os.path.abspath(os.path.join(_HERE, ".."))
_SCRIPT = os.path.join(_TOOL, "run_step6.sh")

_fail = 0


def check(cond, msg):
    global _fail
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _fail += 1


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$1" in
  *detect_dataset.py) printf '%s\\n' "$STUB_DETECT_YAML" ;;
esac
exit 0
"""

# Every knob run_step6.sh forwards, with the flag train_wrapper.py must see.
# A sentinel value per knob so a mis-wired forward cannot pass by accident.
KNOBS = [
    ("TCRMP_STEP6_DEVICE", "--device", "0,1"),
    ("TCRMP_STEP6_BATCH", "--batch", "8"),
    ("TCRMP_STEP6_PATIENCE", "--patience", "11"),
    ("TCRMP_STEP6_OPTIMIZER", "--optimizer", "AdamW"),
    ("TCRMP_STEP6_SEED", "--seed", "42"),
    ("TCRMP_STEP6_COS_LR", "--cos_lr", "true"),
    ("TCRMP_STEP6_CLOSE_MOSAIC", "--close_mosaic", "5"),
    ("TCRMP_STEP6_FREEZE", "--freeze", "10"),
    ("TCRMP_STEP6_LR0", "--lr0", "0.005"),
    ("TCRMP_STEP6_LRF", "--lrf", "0.02"),
    ("TCRMP_STEP6_MOMENTUM", "--momentum", "0.9"),
    ("TCRMP_STEP6_WEIGHT_DECAY", "--weight_decay", "0.001"),
    ("TCRMP_STEP6_WARMUP_EPOCHS", "--warmup_epochs", "4.0"),
    ("TCRMP_STEP6_BOX", "--box", "8.5"),
    ("TCRMP_STEP6_CLS", "--cls", "0.6"),
    ("TCRMP_STEP6_DFL", "--dfl", "1.6"),
    ("TCRMP_STEP6_LABEL_SMOOTHING", "--label_smoothing", "0.05"),
    ("TCRMP_STEP6_HSV_H", "--hsv_h", "0.03"),
    ("TCRMP_STEP6_HSV_S", "--hsv_s", "0.7"),
    ("TCRMP_STEP6_HSV_V", "--hsv_v", "0.55"),
    ("TCRMP_STEP6_DEGREES", "--degrees", "15.0"),
    ("TCRMP_STEP6_TRANSLATE", "--translate", "0.12"),
    ("TCRMP_STEP6_SCALE", "--scale", "0.35"),
    ("TCRMP_STEP6_SHEAR", "--shear", "2.0"),
    ("TCRMP_STEP6_PERSPECTIVE", "--perspective", "0.0005"),
    ("TCRMP_STEP6_FLIPUD", "--flipud", "0.25"),
    ("TCRMP_STEP6_FLIPLR", "--fliplr", "0.75"),
    ("TCRMP_STEP6_MOSAIC", "--mosaic", "0.4"),
    ("TCRMP_STEP6_MIXUP", "--mixup", "0.05"),
    ("TCRMP_STEP6_CUTMIX", "--cutmix", "0.06"),
    ("TCRMP_STEP6_COPY_PASTE", "--copy_paste", "0.2"),
    ("TCRMP_STEP6_COPY_PASTE_MODE", "--copy_paste_mode", "mixup"),
    ("TCRMP_STEP6_ERASING", "--erasing", "0.15"),
    ("TCRMP_STEP6_AUTO_AUGMENT", "--auto_augment", "augmix"),
    ("TCRMP_STEP6_BGR", "--bgr", "0.1"),
]


root = tempfile.mkdtemp(prefix="run_step6_task_test_")
try:
    stub = os.path.join(root, "stub_python.sh")
    write(stub, STUB)
    os.chmod(stub, 0o755)

    step5 = os.path.join(root, "step5_export")
    os.makedirs(os.path.join(step5, "all_images"), exist_ok=True)
    step6 = os.path.join(root, "step6_trainSegmentation")
    write(os.path.join(step6, "dataset", "data.yaml"), "path: .\ntrain: train/images\n")
    detect_yaml = os.path.join(step6, "dataset_detect", "data.yaml")
    write(detect_yaml, "path: .\ntrain: train/images\n")

    def run(argv, env_extra=None, log_name="stub.log"):
        """Run the driver with the stub python and return (proc, invocations)."""
        log = os.path.join(root, log_name)
        if os.path.exists(log):
            os.remove(log)
        env = dict(os.environ)
        env.pop("TCRMP_STEP6_TASK", None)
        env.update({
            "TCRMP_STEP6_PYTHON": stub,
            "STUB_LOG": log,
            "STUB_DETECT_YAML": detect_yaml,
        })
        env.update(env_extra or {})
        proc = subprocess.run(["bash", _SCRIPT] + argv, capture_output=True, text=True, env=env)
        lines = []
        if os.path.isfile(log):
            lines = [ln for ln in open(log).read().splitlines() if ln.strip()]
        return proc, lines

    def train_args(lines):
        for ln in lines:
            if "train_wrapper.py" in ln:
                return ln.split()
        return []

    def flag_value(argv, flag):
        return argv[argv.index(flag) + 1] if flag in argv else None

    # ---- syntax ---------------------------------------------------------
    print("\n-- syntax --")
    syn = subprocess.run(["bash", "-n", _SCRIPT], capture_output=True, text=True)
    check(syn.returncode == 0, f"bash -n accepts run_step6.sh ({syn.stderr.strip()})")

    # ---- segment stays exactly as it was --------------------------------
    print("\n-- segment (default) --")
    proc, lines = run(["train", step6, "run1", "7", "640"])
    argv = train_args(lines)
    check(proc.returncode == 0, f"train mode exits 0 (stderr: {proc.stderr.strip()[-200:]})")
    check(bool(argv), "train_wrapper.py is invoked")
    check(flag_value(argv, "--src") == os.path.join(step6, "dataset", "data.yaml"),
          f"a segmentation run trains on dataset/data.yaml (got {flag_value(argv, '--src')})")
    check(flag_value(argv, "--model") == "yolo11m-seg.pt",
          f"the segmentation default weights are unchanged (got {flag_value(argv, '--model')})")
    check(flag_value(argv, "--task") == "segment", "the task is stated explicitly")
    check(flag_value(argv, "--epochs") == "7" and flag_value(argv, "--imgsz") == "640",
          "the positional epochs and imgsz still reach the wrapper")
    check(flag_value(argv, "--project") == os.path.join(step6, "runs")
          and flag_value(argv, "--name") == "run1",
          "the run is written under the step6 runs dir")
    check(not any("detect_dataset.py" in ln for ln in lines),
          "a segmentation run prepares no detection dataset")

    # ---- detect ----------------------------------------------------------
    print("\n-- detect --")
    proc, lines = run(["train", step6, "run2"], {"TCRMP_STEP6_TASK": "detect"})
    argv = train_args(lines)
    check(proc.returncode == 0, f"detect train mode exits 0 (stderr: {proc.stderr.strip()[-200:]})")
    prep = [ln for ln in lines if "detect_dataset.py" in ln]
    check(len(prep) == 1 and "prepare" in prep[0] and os.path.join(step6, "dataset") in prep[0],
          f"the driver prepares the detection dataset first (got {prep})")
    check(os.path.join(step6, "dataset_detect") in prep[0],
          "dataset_detect is written beside dataset")
    check(flag_value(argv, "--src") == detect_yaml,
          f"training reads the data.yaml prepare printed (got {flag_value(argv, '--src')})")
    check(flag_value(argv, "--task") == "detect", "the wrapper is told to train a detector")
    check(flag_value(argv, "--model") == "yolo11m.pt",
          f"the detection default weights are the plain ones (got {flag_value(argv, '--model')})")

    # ---- base weights overrides -----------------------------------------
    print("\n-- base weights --")
    proc, lines = run(["train", step6, "run3", "7", "640", "yolo11n.pt"],
                      {"TCRMP_STEP6_TASK": "detect"})
    check(flag_value(train_args(lines), "--model") == "yolo11n.pt",
          "the positional model argument overrides the task default")
    proc, lines = run(["train", step6, "run4", "7", "640", "yolo11n.pt"],
                      {"TCRMP_STEP6_TASK": "detect",
                       "TCRMP_STEP6_MODEL_PATH": "/abs/best.pt"})
    check(flag_value(train_args(lines), "--model") == "/abs/best.pt",
          "TCRMP_STEP6_MODEL_PATH wins over everything (fine-tune from named weights)")
    proc, lines = run(["train", step6, "run5"], {"TCRMP_STEP6_TASK": "seg"})
    check(flag_value(train_args(lines), "--task") == "segment"
          and flag_value(train_args(lines), "--model") == "yolo11m-seg.pt",
          "the short spelling seg is normalized before it is forwarded, and keeps the seg weights")
    proc, lines = run(["train", step6, "run6"], {"TCRMP_STEP6_TASK": "nonsense"})
    check(proc.returncode != 0 and "task" in (proc.stderr + proc.stdout).lower(),
          f"an unknown task is refused (rc={proc.returncode})")
    check(not any("detect_dataset.py" in ln for ln in lines),
          "an unknown task is refused before anything is written")
    # Case and spelling are normalized in ONE place. Forwarding the raw value
    # would reach the wrapper's own parser after dataset_detect/ was written.
    proc, lines = run(["train", step6, "run6b"], {"TCRMP_STEP6_TASK": "Detect"})
    check(proc.returncode == 0 and flag_value(train_args(lines), "--task") == "detect"
          and flag_value(train_args(lines), "--model") == "yolo11m.pt",
          f"a capitalized spelling is normalized, not forwarded raw "
          f"(got {flag_value(train_args(lines), '--task')})")
    proc, lines = run(["train", step6, "run6c"], {"TCRMP_STEP6_TASK": "BBOX"})
    check(flag_value(train_args(lines), "--task") == "detect",
          "every accepted spelling reaches the wrapper as one of its two canonical names")

    # ---- every knob, both tasks -----------------------------------------
    print("\n-- knob forwarding --")
    knob_env = {var: value for var, _flag, value in KNOBS}
    _proc, seg_lines = run(["train", step6, "run7"], dict(knob_env))
    det_env = dict(knob_env)
    det_env["TCRMP_STEP6_TASK"] = "detect"
    _proc, det_lines = run(["train", step6, "run8"], det_env)
    seg_argv, det_argv = train_args(seg_lines), train_args(det_lines)
    for var, flag, value in KNOBS:
        ok = flag_value(seg_argv, flag) == value and flag_value(det_argv, flag) == value
        check(ok, f"{var} reaches {flag} for both tasks "
                  f"(seg={flag_value(seg_argv, flag)}, detect={flag_value(det_argv, flag)})")
    differ = {flag for flag in ("--src", "--task", "--model")
              if flag_value(seg_argv, flag) != flag_value(det_argv, flag)}
    check(differ == {"--src", "--task", "--model"},
          f"the data yaml, the task and the weights are the only differences (got {differ})")
    seg_rest = [a for a in seg_argv if a not in (flag_value(seg_argv, "--src"), "segment",
                                                 "yolo11m-seg.pt", "run7")]
    det_rest = [a for a in det_argv if a not in (flag_value(det_argv, "--src"), "detect",
                                                 "yolo11m.pt", "run8")]
    check(seg_rest == det_rest,
          "every other composed argument is byte-identical between the two tasks")

    # ---- all mode --------------------------------------------------------
    print("\n-- all mode --")
    proc, lines = run(["all", step5, step6, "run9", "7", "640"],
                      {"TCRMP_STEP6_TASK": "detect", "TCRMP_STEP6_PIN_SPLIT": "1"})
    check(proc.returncode == 0, f"all mode exits 0 (stderr: {proc.stderr.strip()[-200:]})")
    split = [ln for ln in lines if "pinned_split.py" in ln]
    check(len(split) == 1 and f"--src {step5}" in split[0],
          f"the pinned split still runs first, unchanged (got {split})")
    argv = train_args(lines)
    check(flag_value(argv, "--task") == "detect" and flag_value(argv, "--src") == detect_yaml,
          "the task carries through all mode into training")
    check(any("detect_dataset.py" in ln for ln in lines),
          "all mode derives the detection dataset after the split")
finally:
    shutil.rmtree(root, ignore_errors=True)

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)

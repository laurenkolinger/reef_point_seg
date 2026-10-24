"""Unit tests for the VICARIUS_GPU_DEVICE override in train_wrapper.py,
evaluate_run.py and run_inference.py (GPU traffic, docs/superpowers/specs/
2026-09-28-gpu-traffic-design.md, section 7).

VICARIUS's stage runner sets this environment variable to the CUDA device
string for the card(s) it assigned a "vega"-tier stage (training,
evaluation, inference), in place of --device. No GPU is touched here: only
the pure parsing/validation path (resolve_device_override) and the
args-mutating wrapper (apply_gpu_device_override) are exercised, on a
hand-built argparse.Namespace. train_wrapper.py imports `from ultralytics
import YOLO` at module level, exactly as the existing test_train_task.py
already does in this same suite; nothing here calls YOLO() or touches a
CUDA device.

The contract under test, per module:
  * unset, or set to an empty string: --device is left exactly as parsed
    (byte-for-byte identical to a run made before this variable existed).
  * a valid CUDA device string ("1", "0,1", or an arbitrarily large index)
    replaces --device outright and is printed once, naming the variable.
  * anything else (letters, a negative number, an empty list entry, a
    trailing comma, a decimal, embedded whitespace) raises ValueError
    naming VICARIUS_GPU_DEVICE and the bad value, rather than silently
    falling back to --device or to every card.

Run: REPO/env/bin/python scripts/TCRMPtrain_oceankindCV/tests/test_gpu_device_override.py
"""
import argparse
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import train_wrapper as TW  # noqa: E402
import evaluate_run as ER  # noqa: E402
import run_inference as RI  # noqa: E402

_fail = 0


def check(cond, msg):
    global _fail
    if not cond:
        _fail += 1
        print(f"FAIL: {msg}")
    else:
        print(f"PASS: {msg}")


# Values that read as unset. resolve_device_override treats both the same
# way as apply_gpu_device_override reads os.environ.get(name, ''): neither
# a missing key nor an explicitly empty string is a device assignment.
# "1\n" and "0,1\n" pin fullmatch() over match(): match() alone accepts a
# trailing newline because $ matches just before one, so a value read from a
# file or a shell substitution with a stray trailing newline would have
# silently passed as "1" or "0,1" instead of being refused.
_MALFORMED = ("a", "-1", "1,,2", "0, 1", "0.5", ",", "1,2,", "1 ", " 1", "0,-1",
              "1\n", "0,1\n")


def _check_module(mod, label):
    env_name = mod.GPU_DEVICE_ENV
    check(env_name == "VICARIUS_GPU_DEVICE",
          f"{label}: GPU_DEVICE_ENV names the platform's own variable (got {env_name!r})")

    try:
        # ---- resolve_device_override: the pure function -------------------
        check(mod.resolve_device_override("0", "") == "0",
              f"{label}: resolve_device_override('0', '') keeps the cli value")
        check(mod.resolve_device_override("0", None) == "0",
              f"{label}: resolve_device_override('0', None) keeps the cli value")
        check(mod.resolve_device_override("0", "1,2") == "1,2",
              f"{label}: resolve_device_override returns a valid env value verbatim")
        huge = "9" * 40
        check(mod.resolve_device_override("0", huge) == huge,
              f"{label}: a huge device index is accepted, not truncated or overflowed")
        for bad in _MALFORMED:
            try:
                mod.resolve_device_override("0", bad)
                check(False, f"{label}: resolve_device_override({bad!r}) must raise")
            except ValueError as exc:
                # repr(bad), not the raw bad string: a bad value that itself
                # contains a character repr() escapes (the trailing newline
                # in "1\n") would never appear as a raw substring of a
                # message built with an !r conversion, since the message
                # holds the escaped form.
                check(env_name in str(exc) and repr(bad) in str(exc),
                      f"{label}: the error for {bad!r} names {env_name} and the value "
                      f"(got {exc})")

        # ---- apply_gpu_device_override: unset / empty leave --device alone ----
        os.environ.pop(env_name, None)
        args = argparse.Namespace(device="0")
        mod.apply_gpu_device_override(args)
        check(args.device == "0", f"{label}: an unset env leaves --device untouched (got {args.device!r})")

        os.environ[env_name] = ""
        args = argparse.Namespace(device="0")
        mod.apply_gpu_device_override(args)
        check(args.device == "0", f"{label}: an empty env value leaves --device untouched (got {args.device!r})")

        # ---- valid overrides replace --device, including a DDP list -----------
        os.environ[env_name] = "1"
        args = argparse.Namespace(device="0")
        mod.apply_gpu_device_override(args)
        check(args.device == "1", f"{label}: env='1' overrides --device (got {args.device!r})")

        os.environ[env_name] = "0,1"
        args = argparse.Namespace(device="0")
        mod.apply_gpu_device_override(args)
        check(args.device == "0,1",
              f"{label}: env='0,1' overrides --device with a DDP device list (got {args.device!r})")

        os.environ[env_name] = huge
        args = argparse.Namespace(device="0")
        mod.apply_gpu_device_override(args)
        check(args.device == huge, f"{label}: a huge device index passes through as-is")

        # ---- malformed values raise loudly, never fall back silently ----------
        for bad in _MALFORMED:
            os.environ[env_name] = bad
            args = argparse.Namespace(device="0")
            try:
                mod.apply_gpu_device_override(args)
                check(False, f"{label}: malformed env {bad!r} must raise (got device={args.device!r})")
            except ValueError as exc:
                check(env_name in str(exc),
                      f"{label}: the raised error for {bad!r} names {env_name} (got {exc})")

        # ---- exactly one printed line, naming the override; none when unset ---
        os.environ[env_name] = "1"
        args = argparse.Namespace(device="0")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod.apply_gpu_device_override(args)
        printed = buf.getvalue()
        check(printed.count("\n") == 1 and env_name in printed and "1" in printed,
              f"{label}: exactly one line is printed, naming the override (got {printed!r})")

        os.environ.pop(env_name, None)
        args = argparse.Namespace(device="0")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod.apply_gpu_device_override(args)
        check(buf.getvalue() == "",
              f"{label}: no line is printed when the variable is unset (got {buf.getvalue()!r})")
    finally:
        os.environ.pop(env_name, None)


_check_module(TW, "train_wrapper")
_check_module(ER, "evaluate_run")
_check_module(RI, "run_inference")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)

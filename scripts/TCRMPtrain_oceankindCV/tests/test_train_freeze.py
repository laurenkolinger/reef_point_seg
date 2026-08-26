"""Unit tests for train_wrapper.py --freeze plumbing.

Trains need a GPU, so this exercises only the pure kwargs-assembly path:
arg_parse() -> build_train_kwargs(args). No YOLO model is loaded.

Run: REPO/env/bin/python scripts/TCRMPtrain_oceankindCV/tests/test_train_freeze.py
"""
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
    """Parse argv through train_wrapper's own arg_parse(), bypassing
    sys.argv so tests don't depend on process invocation."""
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

# --freeze passed -> present in kwargs with the given int value.
args_with_freeze = _parse(base_argv + ["--freeze", "10"])
check(args_with_freeze.freeze == 10, "arg_parse: --freeze 10 parses to int 10")

kwargs_with_freeze = TW.build_train_kwargs(args_with_freeze)
check("freeze" in kwargs_with_freeze, "build_train_kwargs: freeze key present when --freeze given")
check(kwargs_with_freeze.get("freeze") == 10, "build_train_kwargs: freeze value == 10")

# without --freeze -> default None, key entirely absent from kwargs.
args_no_freeze = _parse(base_argv)
check(args_no_freeze.freeze is None, "arg_parse: --freeze defaults to None when omitted")

kwargs_no_freeze = TW.build_train_kwargs(args_no_freeze)
check("freeze" not in kwargs_no_freeze, "build_train_kwargs: freeze key absent when --freeze omitted")

# sanity: a handful of other kwargs still resolve as expected (no regression
# in the extraction of build_train_kwargs out of main()).
check(kwargs_no_freeze["data"] == "/tmp/fake_data.yaml", "build_train_kwargs: data passthrough")
check(kwargs_no_freeze["project"] == "/tmp/fake_project", "build_train_kwargs: project passthrough")
check(kwargs_no_freeze["name"] == "run1", "build_train_kwargs: name passthrough")
check(kwargs_no_freeze["epochs"] == 500, "build_train_kwargs: epochs default passthrough")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)

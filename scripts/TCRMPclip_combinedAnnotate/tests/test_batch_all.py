"""
Task 3: TCRMP_REVIEW_BATCH_SIZE='all' must disable batching (one batch holding
every routed frame), not silently fall back to the default batch size.

NO-pytest harness. Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_batch_all.py

config.py reads the env at import, so each case runs in a fresh subprocess.
"""

import os
import sys
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
_PY = sys.executable

_RESULTS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _batch_size_for(rbs):
    """Import config in a fresh process with TCRMP_REVIEW_BATCH_SIZE=rbs and
    print REVIEW_BATCH_SIZE."""
    env = dict(os.environ)
    env["TCRMP_TARGET_SPECIES"] = env.get("TCRMP_TARGET_SPECIES", "MC")
    if rbs is None:
        env.pop("TCRMP_REVIEW_BATCH_SIZE", None)
    else:
        env["TCRMP_REVIEW_BATCH_SIZE"] = rbs
    code = (
        "import sys; sys.path.insert(0, %r); import config; "
        "print(config.REVIEW_BATCH_SIZE)" % _SRC
    )
    out = subprocess.run([_PY, "-c", code], env=env, capture_output=True, text=True)
    check(out.returncode == 0, f"config import failed for {rbs!r}: {out.stderr}")
    return int(out.stdout.strip().splitlines()[-1])


def run(fn):
    import traceback
    try:
        fn(); _RESULTS.append((fn.__name__, True, "")); print(f"  PASS {fn.__name__}")
    except Exception as e:
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


def test_all_disables_batching():
    bs = _batch_size_for("all")
    check(bs >= 10 ** 6, f"'all' should be a huge batch (no batching), got {bs}")


def test_all_case_insensitive():
    check(_batch_size_for("ALL") >= 10 ** 6, "'ALL' should also disable batching")


def test_numeric_still_works():
    check(_batch_size_for("7") == 7, "numeric batch size must be honored")


def test_bogus_falls_back_to_default():
    check(_batch_size_for("nonsense") == 10, "bogus value falls back to default 10")


if __name__ == "__main__":
    print("TASK 3 — 'All (no batching)' batch size\n")
    for fn in (test_all_disables_batching, test_all_case_insensitive,
               test_numeric_still_works, test_bogus_falls_back_to_default):
        run(fn)
    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    sys.exit(1 if failed else 0)

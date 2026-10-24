"""Step 3 applies the project's own species recode, and says so truthfully.

Until 2026-10-15 _load_species_remap looked in
<github_repo>/TCRMPcvr_recodeSpecies/output, a folder that does not exist, so
the pre-2020 points read from cpc_all were never remapped while the log still
said "remapped". The lookup now reads the remap log the orchestrator passes
(--remap-log, the project's step 2 remap_log.json).

Run: CUDA_VISIBLE_DEVICES="" env/bin/python scripts/TCRMPcvr_chooseImages/tests/test_species_remap_path.py
"""
import io
import contextlib
import json
import os
import shutil
import sys
import tempfile
import traceback

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
import select_images as si  # noqa: E402

_R = []
_TMP = []


def check(c, m):
    if not c:
        raise AssertionError(m)


def run(fn):
    try:
        fn()
        _R.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:  # noqa: BLE001
        _R.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


def _tmp():
    d = tempfile.mkdtemp(prefix="remap_path_")
    _TMP.append(d)
    return d


def _log(d, remaps, name="remap_log.json"):
    p = os.path.join(d, name)
    with open(p, "w") as fh:
        json.dump({"timestamp": "20261015_100000", "remaps": remaps, "excludes": []}, fh)
    return p


def test_project_log_is_found_and_applied():
    d = _tmp()
    p = _log(d, [{"old_code": "MACX", "new_code": "OX"},
                 {"old_code": "OFRA", "new_code": "OFRA", "action": "rename"},
                 {"old": "MFAV", "new": "OFAV"},
                 {"from": "MA", "to": "OA"},
                 {"old_code": "", "new_code": "ZZ"}])
    got = si._load_species_remap(p)
    check(got == {"MACX": "OX", "MFAV": "OFAV", "MA": "OA"}, f"mapping: {got}")


def test_remap_reaches_the_pre2020_points():
    d = _tmp()
    ids = os.path.join(d, "2016", "ids")
    os.makedirs(ids)
    pd.DataFrame([{"raw_image": "TCRMP20160815_clip_BWR_T101.jpg", "species_code": "MACX",
                   "category": "Coral", "label": "1", "x": 1, "y": 2},
                  {"raw_image": "TCRMP20160815_clip_BWR_T101.jpg", "species_code": "PA",
                   "category": "Coral", "label": "2", "x": 3, "y": 4}]
                 ).to_csv(os.path.join(ids, "point_coords.csv"), index=False)
    remap = si._load_species_remap(_log(_tmp(), [{"old_code": "MACX", "new_code": "OX"}]))
    out = si.load_cpc_species(d, [2016], remap)
    check(sorted(out["species_code"]) == ["OX", "PA"], f"pre-2020 codes remapped: {list(out['species_code'])}")


def test_no_path_applies_nothing_and_says_so():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got = si._load_species_remap(None)
    check(got == {}, f"no log, no remap: {got}")
    check("No species remap applied" in buf.getvalue(), f"plain line: {buf.getvalue()!r}")
    check("remapped" not in buf.getvalue().replace("No species remap applied", ""),
          "and never claims a remap")


def test_missing_file_applies_nothing_and_says_so():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got = si._load_species_remap(os.path.join(_tmp(), "gone.json"))
    check(got == {} and "No species remap applied" in buf.getvalue(), buf.getvalue())


def test_malformed_log_is_an_error_not_a_silent_skip():
    d = _tmp()
    p = os.path.join(d, "remap_log.json")
    open(p, "w").write("{broken")
    try:
        si._load_species_remap(p)
    except ValueError as e:
        check(p in str(e), f"names the file: {e}")
        return
    raise AssertionError("a log that exists but cannot be read must raise")


def test_wrong_shape_log_is_an_error():
    d = _tmp()
    p = os.path.join(d, "remap_log.json")
    json.dump(["not", "an", "object"], open(p, "w"))
    try:
        si._load_species_remap(p)
    except ValueError:
        return
    raise AssertionError("a log that is not an object must raise")


def test_empty_remap_list_is_a_noop_with_honest_line():
    p = _log(_tmp(), [])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got = si._load_species_remap(p)
    check(got == {}, str(got))
    check("No species remap applied" in buf.getvalue(), buf.getvalue())


def test_describe_never_says_remapped_without_a_remap():
    check("remapped" not in si.describe_cpc_load(10, {}), si.describe_cpc_load(10, {}))
    check("remapped" in si.describe_cpc_load(10, {"A": "B"}), "says remapped when it was")


def test_flag_is_wired_into_the_cli():
    src = open(os.path.join(os.path.dirname(_HERE), "src", "select_images.py"), encoding="utf-8").read()
    check('"--remap-log"' in src, "--remap-log flag exists")
    check('"TCRMPcvr_recodeSpecies", "output"' not in src.split("def _load_species_remap")[1].split("def load_cpc_species")[0],
          "the dead folder is no longer searched")


if __name__ == "__main__":
    for f in (test_project_log_is_found_and_applied, test_remap_reaches_the_pre2020_points,
              test_no_path_applies_nothing_and_says_so, test_missing_file_applies_nothing_and_says_so,
              test_malformed_log_is_an_error_not_a_silent_skip, test_wrong_shape_log_is_an_error,
              test_empty_remap_list_is_a_noop_with_honest_line,
              test_describe_never_says_remapped_without_a_remap, test_flag_is_wired_into_the_cli):
        run(f)
    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)
    left = [d for d in _TMP if os.path.exists(d)]
    print("fixtures removed" if not left else f"fixtures NOT removed: {left}")
    ok = sum(1 for _, p, _ in _R if p)
    for n, p, det in _R:
        if not p:
            print(f"\n--- {n}\n{det}")
    print(f"\n{ok}/{len(_R)} passed")
    sys.exit(0 if ok == len(_R) and not left else 1)

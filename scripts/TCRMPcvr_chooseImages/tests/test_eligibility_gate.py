"""Eligibility gate: a frame survives only if its image AND point source resolve.
Run: env/bin/python scripts/TCRMPcvr_chooseImages/tests/test_eligibility_gate.py
"""
import os, sys, traceback
import pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
import select_images as si  # noqa: E402

_R = []
def check(c, m):
    if not c: raise AssertionError(m)
def run(fn):
    try: fn(); _R.append((fn.__name__, True, "")); print(f"  PASS {fn.__name__}")
    except Exception as e: _R.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}")); print(f"  FAIL {fn.__name__}: {e}")

def _binary():
    # 4 frames: pre-2020 with/without cpc; 2020+ with/without pts. All have images.
    rows = [
        {"frame_id": "f_pre_ok",  "year_int": 2016, "date": "2016-08-15", "site": "BWR", "transect": 1, "frame": 1, "PA": 1},
        {"frame_id": "f_pre_no",  "year_int": 2016, "date": "2016-08-15", "site": "BWR", "transect": 1, "frame": 2, "PA": 1},
        {"frame_id": "f_post_ok", "year_int": 2022, "date": "2022-06-01", "site": "FLC", "transect": 3, "frame": 7, "PA": 1},
        {"frame_id": "f_post_no", "year_int": 2022, "date": "2022-06-01", "site": "FLC", "transect": 3, "frame": 8, "PA": 1},
    ]
    return pd.DataFrame(rows).set_index("frame_id")

def _names(b):
    return {fid: si.build_image_filename(str(b.loc[fid,"date"]), b.loc[fid,"site"], int(b.loc[fid,"transect"]), int(b.loc[fid,"frame"])) for fid in b.index}

def test_keeps_only_resolvable_frames():
    b = _binary(); nm = _names(b)
    # Every frame HAS an image; pre_no lacks CPC; post_no lacks _pts.
    clip_index = {}
    for fid, base in nm.items():
        clip_index[base] = f"/clip/{base}.jpg"
    clip_index[nm["f_post_ok"] + "_pts"] = f"/clip/{nm['f_post_ok']}_pts.jpg"
    cpc_basenames = {nm["f_pre_ok"]}
    keep, excl = si.frame_eligibility(b, clip_index, cpc_basenames, si.build_image_filename)
    check(set(keep) == {"f_pre_ok", "f_post_ok"}, f"kept wrong set: {keep}")
    check(excl["cpc_missing"] == 1, f"cpc_missing={excl['cpc_missing']}")
    check(excl["pts_missing"] == 1, f"pts_missing={excl['pts_missing']}")

def test_image_missing_excluded_even_with_points():
    b = _binary(); nm = _names(b)
    clip_index = {}  # NO images at all
    clip_index[nm["f_pre_ok"] + "_pts"] = "x"  # irrelevant
    cpc_basenames = {nm["f_pre_ok"], nm["f_pre_no"]}
    keep, excl = si.frame_eligibility(b, clip_index, cpc_basenames, si.build_image_filename)
    check(keep == [], f"no image -> nothing eligible, got {keep}")
    check(excl["image_missing"] == 4, f"image_missing={excl['image_missing']}")

if __name__ == "__main__":
    for fn in (test_keeps_only_resolvable_frames, test_image_missing_excluded_even_with_points):
        run(fn)
    ok = sum(1 for _, p, _ in _R if p)
    print(f"\n{ok}/{len(_R)} passed")
    sys.exit(0 if ok == len(_R) else 1)

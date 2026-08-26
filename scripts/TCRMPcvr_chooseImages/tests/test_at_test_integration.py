"""End-to-end Step 3 on a real-ish run: every selected row resolves image+points;
reserve is disjoint from selected. Skips gracefully if AT_test inputs are absent.
Run: env/bin/python scripts/TCRMPcvr_chooseImages/tests/test_at_test_integration.py
"""
import os, sys, subprocess, tempfile, csv

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
_MOD = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))  # github_repo
RUN = "/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/inprocess/run_AT_test_29June_20260629_644346"


def main():
    ap = None
    for cand in ("step2_recodeSpecies/all_points_recoded.csv", "all_points_recoded.csv"):
        p = os.path.join(RUN, cand)
        if os.path.exists(p): ap = p; break
    if ap is None:
        print("  SKIP: AT_test all_points not found"); return 0
    out = tempfile.mkdtemp(prefix="at_step3_")
    env = dict(os.environ, TCRMP_OUTPUT_DIR=out)
    cmd = [os.path.join(_MOD, "env/bin/python"),
           os.path.join(_SRC, "select_images.py"),
           "--species", "OFRA", "PA", "SS", "--target", "20",
           "--all-points", ap, "--min-year", "2014", "--max-year", "2025"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=_SRC, env=env)
    if r.returncode != 0:
        print("  FAIL: select_images errored:\n", r.stderr[-1500:]); return 1
    sf = os.path.join(out, "selected_frames.csv")
    with open(sf) as f:
        rows = list(csv.DictReader(f))
    bad = [x for x in rows if not x.get("source_image")]
    if bad:
        print(f"  FAIL: {len(bad)} selected rows lack a resolved image"); return 1
    for x in rows:
        if x["point_source"] == "pts" and not x["point_source_path"]:
            print(f"  FAIL: 2020+ row missing _pts: {x['basename']}"); return 1
    sel = {x["frame_id"] for x in rows}
    res_path = os.path.join(out, "reserve_frames.csv")
    if os.path.exists(res_path):
        with open(res_path) as f:
            res = {x["frame_id"] for x in csv.DictReader(f)}
        if sel & res:
            print(f"  FAIL: reserve overlaps selected: {sel & res}"); return 1
    print(f"  PASS: {len(rows)} selected rows all resolve image+points; reserve disjoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())

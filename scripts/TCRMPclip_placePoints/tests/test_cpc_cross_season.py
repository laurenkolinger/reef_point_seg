"""A pre-2020 frame's CPC points can sit in a season folder that does not
match the frame's own calendar year: a TCRMP season surveyed late in the
year carries dates into the following January through June, but
cpc_all/<year>/ids/point_coords.csv is named for the SEASON, not the date.
cpc_all/2017 holds 2,623 frames dated January to June 2018; the calendar-
year-only lookup in load_cpc_points dropped every one of them at routing
with "no CPC points found" (run_26-09-18_AT lost 57 frames this way).

Run with the unified env python (no pytest), standalone:
    env/bin/python scripts/TCRMPclip_placePoints/tests/test_cpc_cross_season.py

No GPU: this only imports app.py (EasyOCR loads lazily inside detect.py,
never at import time) and exercises find_cpc_rows_any_season / load_cpc_points
directly, plus one read-only check against real data.
"""
import builtins
import csv
import importlib.util
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PP_SRC = os.path.normpath(os.path.join(HERE, "..", "src"))

REAL_CPC_ROOT = (
    "/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026"
    "/supporting_data/cpc_all"
)
RUN_SELECTED_FRAMES = (
    "/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/inprocess"
    "/run_26-09-18_AT/step3_chooseImages/selected_frames.csv"
)
EXPECTED_RECOVERED = 57

passed = failed = 0


def ok(cond, msg):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print("  FAIL:", msg)


# ── Import app.py the way test_routing_report.py does: env vars set before
# import so config.py resolves them, module loaded by path so importing this
# test twice in one run (unlikely, but matches sibling tests) never collides
# with a real "app" module on sys.path. ──
tmp_env = tempfile.mkdtemp(prefix="pp_cpc_cross_season_env_")
os.environ["TCRMP_SELECTED_FRAMES"] = os.path.join(tmp_env, "selected_frames.csv")
os.environ["TCRMP_EXPORT_DIR"] = os.path.join(tmp_env, "export")
os.environ["TCRMP_CPC_DIR"] = os.path.join(tmp_env, "cpc_all")
os.environ["TCRMP_CLIP_DIR"] = os.path.join(tmp_env, "clip")
os.environ["TCRMP_TARGET_SPECIES"] = "PA"
os.environ["TCRMP_REMAP_LOG"] = os.path.join(tmp_env, "_no_remap.json")

prev_cwd = os.getcwd()
os.chdir(PP_SRC)
sys.path.insert(0, PP_SRC)
spec = importlib.util.spec_from_file_location(
    "placepoints_app_cross_season", os.path.join(PP_SRC, "app.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
os.chdir(prev_cwd)


def _write_point_coords(path, rows):
    """rows: list of (raw_image_basename, label, x, y) tuples."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "raw_image", "label", "species_code", "species_name",
            "category", "x", "y", "source"])
        w.writeheader()
        for (base, label, x, y) in rows:
            w.writerow({
                "raw_image": base + ".jpg", "label": label,
                "species_code": "PA", "species_name": "Porites astreoides",
                "category": "coral", "x": x, "y": y, "source": "cpc",
            })


def _season_csv(cpc_root, season):
    return os.path.join(cpc_root, str(season), "ids", "point_coords.csv")


# ── Test 1: frame dated into the next calendar year is only in the prior
# season's folder; the lookup must still find it and name that season. ──
def test_found_only_in_prior_season():
    cpc_root = tempfile.mkdtemp(prefix="pp_cross_season_t1_")
    base = "TCRMP20180118_clip_BWR_T101"
    _write_point_coords(_season_csv(cpc_root, 2017),
                         [(base, "A", 213.3, 1439.9), (base, "B", 313.7, 551.6)])
    # 2018's folder exists but does not have this frame.
    _write_point_coords(_season_csv(cpc_root, 2018), [])

    rows, season_used = mod.find_cpc_rows_any_season(base, 2018, cpc_root)
    ok(season_used == "2017", f"season_used={season_used!r} (expected '2017')")
    ok(len(rows) == 2, f"rows={len(rows)} (expected 2)")
    if rows:
        ok({r["label"] for r in rows} == {"A", "B"},
           f"labels={[r['label'] for r in rows]} (expected A, B)")


# ── Test 2: frame present in BOTH its calendar year and another season with
# different coordinates. The calendar year must win, and the conflict must
# be logged. ──
def test_calendar_year_wins_on_conflict():
    cpc_root = tempfile.mkdtemp(prefix="pp_cross_season_t2_")
    base = "TCRMP20180301_clip_BID_T202"
    _write_point_coords(_season_csv(cpc_root, 2017),
                         [(base, "A", 100.0, 100.0)])
    _write_point_coords(_season_csv(cpc_root, 2018),
                         [(base, "A", 999.0, 999.0)])

    logged = []
    orig_log = mod.log
    mod.log = lambda msg, *a, **kw: logged.append(msg)
    try:
        rows, season_used = mod.find_cpc_rows_any_season(base, 2018, cpc_root)
    finally:
        mod.log = orig_log

    ok(season_used == "2018", f"season_used={season_used!r} (expected '2018')")
    ok(len(rows) == 1 and float(rows[0]["x"]) == 999.0,
       f"rows={rows} (expected calendar-year x=999.0)")
    ok(any(base in m and "2017" in m for m in logged),
       f"no conflict logged for {base}: {logged}")


# ── Test 2b: frame absent from its calendar year but present in several
# other seasons. The previous season (year - 1, the late-survey case) must
# win over a numerically smaller season, and a real content conflict among
# the matching seasons must be logged. ──
def test_previous_season_preferred_over_earlier_and_later():
    cpc_root = tempfile.mkdtemp(prefix="pp_cross_season_t2b_")
    base = "TCRMP20180901_clip_GKT_T210"
    # 2018 (the calendar year) does not have this frame at all.
    _write_point_coords(_season_csv(cpc_root, 2018), [])
    # Three other seasons DO have it, with different rows in each, so an
    # ascending-order fallback (2016 first) would pick the wrong one.
    _write_point_coords(_season_csv(cpc_root, 2016), [(base, "A", 11.0, 11.0)])
    _write_point_coords(_season_csv(cpc_root, 2017), [(base, "A", 22.0, 22.0)])
    _write_point_coords(_season_csv(cpc_root, 2019), [(base, "A", 33.0, 33.0)])

    logged = []
    orig_log = mod.log
    mod.log = lambda msg, *a, **kw: logged.append(msg)
    try:
        rows, season_used = mod.find_cpc_rows_any_season(base, 2018, cpc_root)
    finally:
        mod.log = orig_log

    ok(season_used == "2017",
       f"season_used={season_used!r} (expected '2017', the previous season, "
       "not '2016' which sorts first)")
    ok(len(rows) == 1 and float(rows[0]["x"]) == 22.0,
       f"rows={rows} (expected the 2017 row, x=22.0)")
    ok(any(base in m and season_used in m for m in logged),
       f"no conflict logged naming {base} and {season_used}: {logged}")


# ── Test 3: frame absent from every season folder returns nothing, cleanly. ──
def test_absent_everywhere():
    cpc_root = tempfile.mkdtemp(prefix="pp_cross_season_t3_")
    _write_point_coords(_season_csv(cpc_root, 2017), [])
    _write_point_coords(_season_csv(cpc_root, 2018), [])

    rows, season_used = mod.find_cpc_rows_any_season(
        "TCRMP20170101_clip_GHOST_T999", 2017, cpc_root)
    ok(rows == [], f"rows={rows} (expected [])")
    ok(season_used is None, f"season_used={season_used!r} (expected None)")


# ── Test 4: a .cpc file is indexed for the frame but its raw image is not
# (a stale or partially-synced clip tree); point_coords.csv still supplies
# points instead of the frame being dropped. ──
def test_cpc_without_raw_image_still_returns_points():
    cpc_root = tempfile.mkdtemp(prefix="pp_cross_season_t4_")
    base = "TCRMP20180410_clip_FLC_T404"
    _write_point_coords(_season_csv(cpc_root, 2018),
                         [(base, "A", 50.0, 60.0)])

    cpc_file = os.path.join(tempfile.mkdtemp(prefix="pp_cross_season_t4_cpc_"),
                             base + ".cpc")
    # Minimal well-formed .cpc header + border block: only lines[0..4] are
    # read (header image name, then BL, BR, TR, TL corners); load_cpc_points
    # never uses the .cpc file's own coordinates or image size, only its
    # presence as a name/border sanity check.
    with open(cpc_file, "w") as f:
        f.write(f"code_path.cpc,{base}.jpg,5000,5000,3000,2000\n")
        f.write("0,480\n")     # BL
        f.write("640,480\n")   # BR
        f.write("640,0\n")     # TR
        f.write("0,0\n")       # TL

    prev_cpc_index = dict(mod.session.get("cpc_index") or {})
    prev_clip_index = dict(mod.session.get("clip_index") or {})
    mod.session["cpc_index"] = {base: cpc_file}
    mod.session["clip_index"] = {}  # no raw image indexed for this basename
    try:
        points = mod.load_cpc_points(base, 2018, cpc_root, None)
    finally:
        mod.session["cpc_index"] = prev_cpc_index
        mod.session["clip_index"] = prev_clip_index

    ok(len(points) == 1, f"points={points} (expected 1 point)")
    if points:
        ok(points[0]["label"] == "A", f"label={points[0]['label']} (expected A)")


# ── Test 5: each point_coords.csv is read from disk at most once per
# process, however many frames resolve against it. ──
def test_each_csv_opened_once():
    cpc_root = tempfile.mkdtemp(prefix="pp_cross_season_t5_")
    base_2017 = "TCRMP20171204_clip_BWR_T101"
    base_2018 = "TCRMP20180118_clip_BWR_T102"
    _write_point_coords(_season_csv(cpc_root, 2017),
                         [(base_2017, "A", 1.0, 2.0), (base_2018, "A", 3.0, 4.0)])
    _write_point_coords(_season_csv(cpc_root, 2018), [])

    open_counts = {}
    real_open = builtins.open

    def counting_open(path, *a, **kw):
        if str(path).endswith("point_coords.csv"):
            open_counts[str(path)] = open_counts.get(str(path), 0) + 1
        return real_open(path, *a, **kw)

    builtins.open = counting_open
    try:
        # Five lookups spanning both basenames and both calendar years;
        # every one of them touches the 2017 and/or 2018 CSV.
        mod.find_cpc_rows_any_season(base_2017, 2017, cpc_root)
        mod.find_cpc_rows_any_season(base_2017, 2017, cpc_root)
        mod.find_cpc_rows_any_season(base_2018, 2018, cpc_root)
        mod.find_cpc_rows_any_season(base_2018, 2018, cpc_root)
        mod.find_cpc_rows_any_season("nonexistent_frame", 2018, cpc_root)
    finally:
        builtins.open = real_open

    path_2017 = _season_csv(cpc_root, 2017)
    path_2018 = _season_csv(cpc_root, 2018)
    ok(open_counts.get(path_2017, 0) == 1,
       f"2017 point_coords.csv opened {open_counts.get(path_2017, 0)}x (expected 1)")
    ok(open_counts.get(path_2018, 0) == 1,
       f"2018 point_coords.csv opened {open_counts.get(path_2018, 0)}x (expected 1)")


# ── Test 6 (read-only, real data): count how many of run_26-09-18_AT's
# selected pre-2020 frames the OLD (calendar-year-only) lookup missed and
# the NEW (any-season) lookup recovers. Expected 57 (2026-09-27 audit). ──
def test_real_run_recovery_count():
    if not os.path.isdir(REAL_CPC_ROOT):
        print(f"  SKIP: real cpc root not present: {REAL_CPC_ROOT}")
        return
    if not os.path.isfile(RUN_SELECTED_FRAMES):
        print(f"  SKIP: run selection not present: {RUN_SELECTED_FRAMES}")
        return

    with open(RUN_SELECTED_FRAMES, newline="") as f:
        rows = list(csv.DictReader(f))

    pre_2020 = [r for r in rows if int(r["year"]) < 2020]
    recovered = []
    for r in pre_2020:
        base = r["basename"]
        year = int(r["year"])
        old_csv = _season_csv(REAL_CPC_ROOT, year)
        old_hit = False
        if os.path.exists(old_csv):
            with open(old_csv, newline="") as cf:
                for row in csv.DictReader(cf):
                    if os.path.splitext(row["raw_image"])[0] == base:
                        old_hit = True
                        break
        if old_hit:
            continue
        new_rows, new_season = mod.find_cpc_rows_any_season(base, year, REAL_CPC_ROOT)
        if new_rows:
            recovered.append((base, year, new_season))

    ok(len(recovered) == EXPECTED_RECOVERED,
       f"recovered={len(recovered)} (expected {EXPECTED_RECOVERED})")
    print(f"  pre-2020 selected frames checked: {len(pre_2020)}")
    print(f"  recovered by the any-season lookup: {len(recovered)}")
    for base, year, season in recovered[:5]:
        print(f"    {base} (calendar year {year}) -> season {season}")


test_found_only_in_prior_season()
test_calendar_year_wins_on_conflict()
test_previous_season_preferred_over_earlier_and_later()
test_absent_everywhere()
test_cpc_without_raw_image_still_returns_points()
test_each_csv_opened_once()
test_real_run_recovery_count()

print(f"test_cpc_cross_season: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)

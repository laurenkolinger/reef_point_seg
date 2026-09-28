"""Site-aware reserve refill: a replacement comes from the site that is most
behind its Step 3 allocation, not simply the lowest reserve rank.

Step 3 allocates frames per site under a share cap. When an OCR frame fails,
the old refill took the lowest-ranked species-matching reserve row, so a run
whose failures clustered at an already-full dominant site (Coral Bay in the
real data) pulled its replacements from that same site and pushed it further
past its share. The refill now reads Step 3's selection_summary.json, compares
each candidate site's routed count against its allocation, and takes the
largest deficit. With no selection_summary.json (an older project) the
behavior falls back to the previous rank order exactly.

No pytest, no GPU, no port binding, no service start. Run with the unified env
python:
    env/bin/python scripts/TCRMPclip_placePoints/tests/test_reserve_site_balance.py
"""
import os
import sys
import csv
import json
import shutil
import tempfile
import importlib.util
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PP_SRC = os.path.normpath(os.path.join(HERE, "..", "src"))

# Defensive: nothing in this test should ever spawn a process. Stub before the
# app module is imported so an accidental launch path cannot reach the shell.
subprocess.run = lambda *a, **k: (_ for _ in ()).throw(
    AssertionError("subprocess.run must not be called by this test"))
subprocess.Popen = lambda *a, **k: (_ for _ in ()).throw(
    AssertionError("subprocess.Popen must not be called by this test"))

passed = failed = 0


def ok(c, m):
    global passed, failed
    if c:
        passed += 1
    else:
        failed += 1
        print("  FAIL:", m)


SCRATCH = ("/tmp/claude-1000/-mnt-rip-vicarius-drive/"
           "ab8997e1-410f-4a89-a1a3-8a15baff3ddb/scratchpad")
tmp = tempfile.mkdtemp(prefix="pp_site_refill_",
                       dir=SCRATCH if os.path.isdir(SCRATCH) else None)
clip_dir = os.path.join(tmp, "clip")
os.makedirs(clip_dir)

TARGET = "PA"
OTHER = "OFAV"

# Reserve rows: rank ascending is CRB, BPT, CBS. CRB is the dominant site and
# already at its allocation, so rank order alone would keep feeding it.
RESERVE = [
    # (rank, site, transect, frame, species, route)
    (0, "CRB", 3, 11, TARGET, "ocr_needed"),
    (1, "BPT", 3, 12, TARGET, "ocr_needed"),
    (2, "CBS", 3, 13, OTHER, "ocr_needed"),
    (3, "CBS", 3, 14, TARGET, "ocr_needed"),
]


def _base(site, transect, frame):
    return f"TCRMP20220601_clip_{site}_T{transect}{frame:02d}"


def _touch(path):
    open(path, "w").close()


reserve_csv = os.path.join(tmp, "reserve_frames.csv")
with open(reserve_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame_id", "basename", "date", "year", "site", "transect",
                "frame", "species", "reserve_rank", "route",
                "source_image", "pts_image"])
    for rank, site, transect, frame, sp, route in RESERVE:
        base = _base(site, transect, frame)
        raw = os.path.join(clip_dir, base + ".jpg")
        pts = os.path.join(clip_dir, base + "_pts.jpg")
        _touch(raw)
        _touch(pts)
        w.writerow([base, base, "2022-06-01", 2022, site, transect, frame,
                    sp, rank, route, raw, pts])

# Step 3 said each of the three sites should hold 10 frames.
SUMMARY = {
    "per_site": {"CRB": 10, "BPT": 10, "CBS": 10},
    "per_species": {TARGET: {"target": 30, "achieved": 30, "reason": "ok"}},
    "cap_frac": 0.25,
    "cap_ok": True,
    "n_selected": 30,
    "shortfalls": [],
}
summary_json = os.path.join(tmp, "selection_summary.json")
with open(summary_json, "w") as f:
    json.dump(SUMMARY, f)

prev = os.getcwd()
try:
    os.chdir(PP_SRC)
    sys.path.insert(0, PP_SRC)
    spec = importlib.util.spec_from_file_location(
        "pp_site_refill_app", os.path.join(PP_SRC, "app.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.USE_GPU = False

    reserve_rows = mod._load_reserve(reserve_csv)
    ok(len(reserve_rows) == len(RESERVE),
       f"reserve loaded {len(reserve_rows)} rows, expected {len(RESERVE)}")

    def _reset(site_allocation, site_counts):
        mod.session["reserve_frames"] = mod._load_reserve(reserve_csv)
        mod.session["reserve_used"] = set()
        mod.session["clip_index"] = {}
        mod.session["site_allocation"] = dict(site_allocation)
        mod.session["site_counts"] = dict(site_counts)
        mod.session["routing_report"] = []
        mod.session["export_dir"] = ""

    # ── 1. Selection is skewed toward CRB: CBS is furthest behind. ──────────
    # deficits: CRB 10-10 = 0, BPT 10-9 = 1, CBS 10-4 = 6 -> CBS wins even
    # though CRB holds the lowest reserve rank.
    _reset(SUMMARY["per_site"], {"CRB": 10, "BPT": 9, "CBS": 4})
    repl = mod._pull_reserve_frame([TARGET])
    ok(repl is not None, "a reserve frame was returned")
    ok(repl and repl.get("site") == "CBS",
       f"most under-represented site should win: got {repl and repl.get('site')}")
    ok(repl and TARGET in (repl.get("species_present") or []),
       f"the species match must still hold: {repl and repl.get('species_present')}")

    # The chosen site's running count is bumped so a second failure in the
    # same run does not keep drawing from the same site.
    ok(mod.session["site_counts"].get("CBS") == 5,
       f"post-refill count for CBS should be 5: {mod.session['site_counts']}")

    # ── 2. The species filter outranks the site deficit. ────────────────────
    # CBS's only remaining TARGET row is consumed above, so asking again with
    # the deepest deficit at CBS must fall to BPT rather than take CBS's
    # wrong-species row.
    _reset(SUMMARY["per_site"], {"CRB": 10, "BPT": 9, "CBS": 4})
    first = mod._pull_reserve_frame([TARGET])
    second = mod._pull_reserve_frame([TARGET])
    ok(first and first.get("site") == "CBS", f"first pull is CBS: {first}")
    ok(second and second.get("site") == "BPT",
       f"second pull must skip CBS's wrong-species row and take BPT: "
       f"{second and second.get('site')}")

    # ── 3. No allocation on file: previous rank order, unchanged. ───────────
    _reset({}, {})
    repl = mod._pull_reserve_frame([TARGET])
    ok(repl and repl.get("site") == "CRB",
       f"without selection_summary.json the lowest rank wins: "
       f"{repl and repl.get('site')}")

    # ── 4. The allocation loader reads Step 3's JSON, and tolerates absence.
    got = mod._load_site_allocation(summary_json)
    ok(got == SUMMARY["per_site"],
       f"per_site should load verbatim: {got}")
    ok(mod._load_site_allocation(os.path.join(tmp, "nope.json")) == {},
       "a missing selection_summary.json must yield an empty allocation")
    bad = os.path.join(tmp, "bad.json")
    with open(bad, "w") as f:
        f.write("{not json")
    ok(mod._load_site_allocation(bad) == {},
       "a malformed selection_summary.json must not raise")

    # ── 5. The refill is recorded with its site and post-refill counts. ─────
    _reset(SUMMARY["per_site"], {"CRB": 10, "BPT": 9, "CBS": 4})
    repl = mod._pull_reserve_frame([TARGET])
    repl["refilled_from"] = "TCRMP20220601_clip_CRB_T301"
    repl["refill_reason"] = "ocr_failed"
    mod._record_refill_entry(repl)
    report = mod.session["routing_report"]
    ok(len(report) == 1, f"one refill row recorded: {report}")
    entry = report[0] if report else {}
    ok(entry.get("refill_site") == "CBS",
       f"the report names the site the refill came from: {entry}")
    counts_after = entry.get("site_counts_after") or ""
    ok("CBS=5" in str(counts_after),
       f"the report carries the post-refill per-site counts: {counts_after}")
    ok("refill_site" in mod._ROUTING_REPORT_FIELDS
       and "site_counts_after" in mod._ROUTING_REPORT_FIELDS,
       f"routing_report.csv must carry the new columns: "
       f"{mod._ROUTING_REPORT_FIELDS}")

except Exception as e:  # noqa: BLE001
    failed += 1
    import traceback
    print("  FAIL: site-balanced refill flow:", repr(e))
    traceback.print_exc()
finally:
    os.chdir(prev)
    shutil.rmtree(tmp, ignore_errors=True)

print(f"test_reserve_site_balance: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)

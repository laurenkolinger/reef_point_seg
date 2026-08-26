# scripts/pipeline_orchestrator/tests/test_flatten_routed_input.py
"""_flatten_routed_input collapses routed_input/<year>/{ids,raw,test_pts}
into flat routed_input/{ids/sam_click_prompts.json, raw/}.
No-pytest harness: run with REPO/env/bin/python <this file>.
"""
import json, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import app as appmod

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1


def _write(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        json.dump(obj, f)


# --- two per-year prompt files merge into one flat file ---
d = tempfile.mkdtemp(prefix="flatten_")
r = os.path.join(d, "routed_input")
_write(os.path.join(r, "2014", "ids", "sam_click_prompts.json"),
       {"TCRMP2014_a.jpeg": {"points": [1]}})
_write(os.path.join(r, "2015", "ids", "sam_click_prompts.json"),
       {"TCRMP2015_b.jpeg": {"points": [2]}})
os.makedirs(os.path.join(r, "2014", "raw"), exist_ok=True)
os.makedirs(os.path.join(r, "2015", "raw"), exist_ok=True)
open(os.path.join(r, "2014", "raw", "TCRMP2014_a.jpeg"), "w").close()
open(os.path.join(r, "2015", "raw", "TCRMP2015_b.jpeg"), "w").close()
os.makedirs(os.path.join(r, "2014", "test_pts"), exist_ok=True)

stats = appmod._flatten_routed_input(r)

flat_prompts_path = os.path.join(r, "ids", "sam_click_prompts.json")
check(os.path.isfile(flat_prompts_path), "flat ids/sam_click_prompts.json exists")
with open(flat_prompts_path) as fh:
    merged = json.load(fh)
check(set(merged) == {"TCRMP2014_a.jpeg", "TCRMP2015_b.jpeg"},
      f"both keys present in merged file, got {set(merged)}")
check(merged["TCRMP2014_a.jpeg"] == {"points": [1]}, "2014 entry intact (not rewritten)")
check(merged["TCRMP2015_b.jpeg"] == {"points": [2]}, "2015 entry intact (not rewritten)")

check(os.path.isfile(os.path.join(r, "raw", "TCRMP2014_a.jpeg")),
      "raw/TCRMP2014_a.jpeg moved to flat raw/")
check(os.path.isfile(os.path.join(r, "raw", "TCRMP2015_b.jpeg")),
      "raw/TCRMP2015_b.jpeg moved to flat raw/")

check(not os.path.exists(os.path.join(r, "2014")), "year dir 2014 removed")
check(not os.path.exists(os.path.join(r, "2015")), "year dir 2015 removed")

check(stats["merged"] == 2, f"stats.merged == 2, got {stats['merged']}")
check(stats["collisions"] == 0, f"stats.collisions == 0, got {stats['collisions']}")
check(stats["year_dirs_removed"] == 2,
      f"stats.year_dirs_removed == 2, got {stats['year_dirs_removed']}")

# --- collision: same filename in two year dirs -> skip second, count it ---
d2 = tempfile.mkdtemp(prefix="flatten_col_")
r2 = os.path.join(d2, "routed_input")
_write(os.path.join(r2, "2020", "ids", "sam_click_prompts.json"),
       {"SAME.jpeg": {"points": [10]}})
_write(os.path.join(r2, "2021", "ids", "sam_click_prompts.json"),
       {"SAME.jpeg": {"points": [20]}})
os.makedirs(os.path.join(r2, "2020", "raw"), exist_ok=True)
os.makedirs(os.path.join(r2, "2021", "raw"), exist_ok=True)
open(os.path.join(r2, "2020", "raw", "SAME.jpeg"), "w").close()
open(os.path.join(r2, "2021", "raw", "SAME.jpeg"), "w").close()

stats2 = appmod._flatten_routed_input(r2)
check(stats2["collisions"] == 1, f"collision counted (1 skip), got {stats2['collisions']}")
check(os.path.isfile(os.path.join(r2, "raw", "SAME.jpeg")), "first copy of SAME.jpeg survived")

# --- empty routed_input -> noop, returns zeros ---
d3 = tempfile.mkdtemp(prefix="flatten_empty_")
r3 = os.path.join(d3, "routed_input")
os.makedirs(r3)
stats3 = appmod._flatten_routed_input(r3)
check(stats3["merged"] == 0 and stats3["year_dirs_removed"] == 0,
      f"empty dir -> zeros, got {stats3}")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)

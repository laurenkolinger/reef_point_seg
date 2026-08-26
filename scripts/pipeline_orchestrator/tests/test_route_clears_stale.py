# scripts/pipeline_orchestrator/tests/test_route_clears_stale.py
"""A fresh route clears a STALE routed_input/<year>/ (Step 3 selection newer
than the cached prompts) so a prior run cannot leak frames into segmentation;
an up-to-date cache is preserved. Run with github_repo/env/bin/python.
"""
import os, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import app as appmod

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1

# --- _clear_routed_input removes year-dirs and is a safe no-op the 2nd time ---
d = tempfile.mkdtemp(prefix="routed_")
for y in ("2018", "2022"):
    ids = os.path.join(d, y, "ids"); os.makedirs(ids)
    open(os.path.join(ids, "sam_click_prompts.json"), "w").close()
n = appmod._clear_routed_input(d)
check(n == 2, f"cleared 2 year dirs, got {n}")
check(not os.path.exists(os.path.join(d, "2018")), "2018 route removed")
check(not os.path.exists(os.path.join(d, "2022")), "2022 route removed")
check(appmod._clear_routed_input(d) == 0, "second clear is a safe no-op")
check(appmod._clear_routed_input(os.path.join(d, "nope")) == 0, "missing dir -> 0")

# --- _routed_prompts_stale: selected_frames newer than flat prompt -> stale ---
base = tempfile.mkdtemp(prefix="stale_")
routed = os.path.join(base, "routed")
os.makedirs(os.path.join(routed, "ids"))  # flat layout (post-flatten)
prompt = os.path.join(routed, "ids", "sam_click_prompts.json")
sel = os.path.join(base, "selected_frames.csv")
# prompt written first (older), selection written after (newer)
open(prompt, "w").close()
time.sleep(0.02)
open(sel, "w").close()
os.utime(prompt, (1000, 1000))      # force prompt OLD
os.utime(sel, (2000, 2000))         # force selection NEW
check(appmod._routed_prompts_stale(routed, sel) is True, "selection newer than flat prompt -> stale")
# selection OLDER than prompt -> not stale (reuse is fine)
os.utime(sel, (500, 500))
check(appmod._routed_prompts_stale(routed, sel) is False, "selection older than flat prompt -> not stale")
# no prompts at all -> not stale (nothing to invalidate)
empty = os.path.join(base, "empty"); os.makedirs(empty)
check(appmod._routed_prompts_stale(empty, sel) is False, "no flat prompts -> not stale")
# missing selected_frames -> not stale
check(appmod._routed_prompts_stale(routed, os.path.join(base, "gone.csv")) is False, "missing selection -> not stale")

# --- lores marker: records the mode a route was built with so a toggle change
#     invalidates the cache (the "Low res only" checkbox actually takes effect) ---
lm = tempfile.mkdtemp(prefix="lores_")
check(appmod._routed_lores_mode(lm) is None, "no marker -> None (unknown; never forces a re-route)")
appmod._write_routed_lores_marker(lm, True)
check(appmod._routed_lores_mode(lm) is True, "marker written True -> True")
appmod._write_routed_lores_marker(lm, False)
check(appmod._routed_lores_mode(lm) is False, "marker overwritten False -> False")
# a marker survives _clear_routed_input (it removes year DIRS, not the flat file)
os.makedirs(os.path.join(lm, "2019", "ids"))
appmod._clear_routed_input(lm)
check(appmod._routed_lores_mode(lm) is False, "marker persists across _clear_routed_input")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)

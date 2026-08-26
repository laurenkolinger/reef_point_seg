"""Project folders are named YY-MM-DD_{name} (no random hash); creating a
project whose folder already exists raises (one folder per project — Open it
instead of duplicating). Run with github_repo/env/bin/python.
"""
import os, sys, tempfile, re
import datetime as _dt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import project_manager as pm

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1

base = tempfile.mkdtemp(prefix="proj_naming_")
today = _dt.datetime.now().strftime("%y-%m-%d")

st = pm.create_project("LO-30june", base)
dirname = os.path.basename(st["project_dir"])
check(dirname == f"{today}_LO-30june", f"folder is YY-MM-DD_name, got {dirname!r}")
check(st["id"] == f"{today}_LO-30june", f"id matches folder, got {st['id']!r}")
# No trailing 6-hex random hash like the old _92cf4b suffix.
check(re.search(r"_[0-9a-f]{6}$", dirname) is None, f"no random hash suffix in {dirname!r}")
check(os.path.isdir(st["project_dir"]), "project dir created on disk")

# Spaces collapse to underscores (existing behavior) but still no hash/date dup.
st2 = pm.create_project("My Reef", base)
check(os.path.basename(st2["project_dir"]) == f"{today}_My_Reef",
      f"spaces -> underscores, got {os.path.basename(st2['project_dir'])!r}")

# Error if the same project folder already exists (one folder per project).
raised = False
try:
    pm.create_project("LO-30june", base)
except ValueError as e:
    raised = True
    check("already exists" in str(e).lower(), f"error message names the conflict: {e}")
check(raised, "creating an existing project folder raises ValueError")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)

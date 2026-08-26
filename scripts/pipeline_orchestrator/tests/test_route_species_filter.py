"""The routing thread must be launched with the real target-species filter,
not a hardcoded []. Guards against the selection->segmentation leak regressing.
Run with env/bin/python.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(os.path.dirname(HERE), "app.py")

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1

src = open(APP).read()
i = src.index("target=_route_drive")
call = src[i:i + 220]
check("routed_dir, []" not in call,
      "route thread must NOT pass a hardcoded [] species filter")
check("species_list" in call or "_get_target_species_str" in call.replace(" ", "") or
      "species" in call,
      "route thread passes a species filter argument")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)

#!/usr/bin/env python
"""The Bounding Boxes initials field starts on the signed-in initials.

No-pytest house harness. Run with:
  env/bin/python scripts/_boxes/tests/test_initials_seed.py

Runs the real static/boxes.js in node against a fake DOM
(scripts/tests/initials_fake_dom.js) and checks #bx-project-by:
  - a valid ?initials parameter seeds it, upper-cased and trimmed
  - A, ABCDE, L0, <b>, empty and an invalid first of a repeated parameter
    leave it blank; there is no remembered value to fall to
  - an input handler rewrites lower case to upper case and keeps the caret
  - the template's LO placeholder stays display only (the value is never LO)
Exits nonzero on any failure.
"""
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
DRIVER = os.path.join(_SCRIPTS, "tests", "initials_fake_dom.js")
BOXES_JS = os.path.join(_SCRIPTS, "_boxes", "static", "boxes.js")
FIELD = "bx-project-by"

_FAILURES = []


def check(cond, msg):
    print(f"[{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        _FAILURES.append(msg)


def run(search, typed=None):
    """Run boxes.js in node; return the driver's JSON answer."""
    spec = {"file": BOXES_JS, "search": search, "ids": [FIELD], "type": typed or []}
    r = subprocess.run(["node", DRIVER, json.dumps(spec)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"node failed: {r.stderr[:300]}")
    return json.loads(r.stdout)


def main():
    for search, want in [("?initials=AT", "AT"), ("?initials=at", "AT"),
                         ("?initials=%20at%20", "AT"), ("?x=1&initials=abcd", "ABCD")]:
        check(run(search)["values"][FIELD] == want, f"{search} seeds {want}")
    for search in ["?initials=A", "?initials=ABCDE", "?initials=L0", "?initials=%3Cb%3E",
                   "?initials=", "?initials", "", "?initials=A&initials=AT"]:
        check(run(search)["values"][FIELD] == "", f"{search!r} is ignored: the field stays blank")
    out = run("", typed=[[FIELD, "abc", 2]])
    check(out["values"][FIELD] == "ABC", "typed 'abc' becomes ABC")
    check(out["carets"][FIELD] == 2, "the caret stays at 2")
    out = run("", typed=[[FIELD, "AB", 1]])
    check(out["values"][FIELD] == "AB" and out["carets"][FIELD] == 1, "upper case is left alone")
    check(run("")["values"][FIELD] != "LO", "never seeds LO")
    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()

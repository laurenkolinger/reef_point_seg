#!/usr/bin/env python
"""Initials fields of the coverage matrix start on the signed-in initials.

No-pytest house harness. Run with:
  env/bin/python scripts/_matrix/tests/test_initials_seed.py

Runs the real static/matrix_cell.js and static/matrix_stats.js in node against
a fake DOM (scripts/tests/initials_fake_dom.js). Covers the three fields
#mx-by, #mx-stats-by and #mx-train-by:
  - a valid ?initials parameter seeds them, upper-cased and trimmed
  - a parameter beats the remembered mx.by; no parameter falls to mx.by
  - A, ABCDE, L0, <b>, empty and a repeated parameter are ignored
  - seeding never writes mx.by
  - an input handler rewrites lower case to upper case and keeps the caret
  - blocked storage still seeds from the parameter
Exits nonzero on any failure.
"""
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
DRIVER = os.path.join(_SCRIPTS, "tests", "initials_fake_dom.js")
STATIC = os.path.join(_SCRIPTS, "_matrix", "static")
FIELDS = ["mx-by", "mx-stats-by", "mx-train-by"]
FILES = {"mx-by": "matrix_cell.js", "mx-stats-by": "matrix_stats.js", "mx-train-by": "matrix_stats.js"}

_FAILURES = []


def check(cond, msg):
    print(f"[{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        _FAILURES.append(msg)


def run(js_name, search, store=None, throws=False, typed=None):
    """Run one matrix script in node; return the driver's JSON answer."""
    spec = {"file": os.path.join(STATIC, js_name), "search": search, "store": store or {},
            "throws": throws, "ids": FIELDS, "type": typed or []}
    r = subprocess.run(["node", DRIVER, json.dumps(spec)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"node failed for {js_name}: {r.stderr[:300]}")
    return json.loads(r.stdout)


def seeded(search, store=None, throws=False):
    """The value each field starts on, read from the script that owns it."""
    out = {}
    for field in FIELDS:
        out[field] = run(FILES[field], search, store, throws)["values"][field]
    return out


def test_valid_parameter():
    for search, want in [("?initials=AT", "AT"), ("?initials=at", "AT"),
                         ("?initials=%20at%20", "AT"), ("?x=1&initials=abcd", "ABCD")]:
        got = seeded(search)
        check(all(v == want for v in got.values()), f"{search} seeds {want} on every field: {got}")


def test_invalid_parameter_ignored():
    for search in ["?initials=A", "?initials=ABCDE", "?initials=L0", "?initials=%3Cb%3E",
                   "?initials=", "?initials", "?", "", "?initials=%3Cscript%3E"]:
        got = seeded(search)
        check(all(v == "" for v in got.values()), f"{search!r} is ignored and the fields stay blank")
        got = seeded(search, {"mx.by": "ZZ"})
        check(all(v == "ZZ" for v in got.values()), f"{search!r} falls to mx.by")


def test_repeated_parameter_ignored():
    # Two initials parameters are ambiguous, so the page ignores both, even when both are valid.
    got = seeded("?initials=A&initials=AT", {"mx.by": "ZZ"})
    check(all(v == "ZZ" for v in got.values()), "an invalid first of two initials parameters falls to mx.by")
    got = seeded("?initials=AT&initials=XY", {"mx.by": "ZZ"})
    check(all(v == "ZZ" for v in got.values()), "two valid initials parameters fall to mx.by")
    got = seeded("?initials=AT&initials=%3Cb%3E")
    check(all(v == "" for v in got.values()), "a repeated parameter with a hostile value seeds blank")


def test_order_and_no_write():
    got = seeded("?initials=AT", {"mx.by": "ZZ"})
    check(all(v == "AT" for v in got.values()), "the parameter beats mx.by")
    got = seeded("", {"mx.by": "zz"})
    check(all(v == "zz" for v in got.values()), "no parameter: mx.by is the seed (as stored)")
    for js in ("matrix_cell.js", "matrix_stats.js"):
        out = run(js, "?initials=AT", {"mx.by": "ZZ"})
        check(out["writes"] == [], f"{js}: seeding writes no remembered key")
    got = seeded("?initials=AT", throws=True)
    check(all(v == "AT" for v in got.values()), "blocked storage still seeds from the parameter")
    got = seeded("", throws=True)
    check(all(v == "" for v in got.values()), "blocked storage and no parameter stay blank")
    check("LO" not in json.dumps(seeded("")), "never seeds LO")


def test_input_handler_uppercases():
    for field in FIELDS:
        out = run(FILES[field], "", typed=[[field, "abc", 2]])
        check(out["values"][field] == "ABC", f"{field}: typed 'abc' becomes ABC")
        check(out["carets"][field] == 2, f"{field}: the caret stays at 2")
        out = run(FILES[field], "", typed=[[field, "AB", 1]])
        check(out["values"][field] == "AB" and out["carets"][field] == 1, f"{field}: upper case is left alone")


def main():
    test_valid_parameter()
    test_invalid_parameter_ignored()
    test_repeated_parameter_ignored()
    test_order_and_no_write()
    test_input_handler_uppercases()
    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()

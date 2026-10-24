#!/usr/bin/env python
"""No-pytest harness for mask preferences (preferences.py): round trip,
validation, file placement and the choice validator.

Run with:
  env/bin/python scripts/_matrix/tests/test_preferences.py
"""

# Keep this suite off the live mask registry, expert library and clip index
# when it is run alone: scripts/tests/_isolation.py sets the isolating
# variables to temp dirs and blocks any write to the live stores.
import os as _iso_os
import sys as _iso_sys
_iso_dir = _iso_os.path.dirname(_iso_os.path.abspath(__file__))
while _iso_os.path.basename(_iso_dir) != 'scripts' and _iso_dir != _iso_os.path.dirname(_iso_dir):
    _iso_dir = _iso_os.path.dirname(_iso_dir)
_iso_sys.path.insert(0, _iso_os.path.join(_iso_dir, 'tests'))
import _isolation
_isolation.install()

import csv
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _matrix.preferences import (  # noqa: E402
    PREF_FIELDS,
    Preferences,
    exportable_masks,
    preferences_path,
    validate_choice,
)

_FAILURES = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        _FAILURES.append(msg)


def main():
    root = tempfile.mkdtemp(prefix="matrix_pref_test_")
    try:
        prefs = Preferences(root)
        check(prefs.path == os.path.join(root, "_mask_registry", "preferences.csv"),
              f"preferences live next to the registry (got {prefs.path})")
        check(preferences_path(root) == prefs.path, "preferences_path helper agrees")
        check(prefs.load() == [] and prefs.get("IMG", "PA") is None, "empty store loads as []")

        # set + get round trip
        row = prefs.set("IMG_one", "PA", "projA", ["u1", "u2", "u1", ""], by="lo", note="best edge")
        check(row["chosen_by"] == "LO", f"chosen_by uppercased (got {row['chosen_by']})")
        check(row["uids"] == ["u1", "u2"], f"uids de-duplicated, blanks dropped (got {row['uids']})")
        check(len(row["at"]) == 19 and row["at"][10] == " ", f"at stamped AST 24h 'YYYY-MM-DD HH:MM:SS' (got {row['at']})")
        got = prefs.get("IMG_one", "PA")
        check(got is not None and got["project_id"] == "projA" and got["uids"] == ["u1", "u2"]
              and got["note"] == "best edge", f"get returns the stored row (got {got})")

        with open(prefs.path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames
            raw = list(reader)
        check(header == PREF_FIELDS, f"csv header is the D3 column set (got {header})")
        check(raw[0]["uids"] == "u1;u2", f"uids semicolon-joined on disk (got {raw[0]['uids']})")

        # replace on the same key, second key independent
        prefs.set("IMG_one", "PA", "projB", [], by="AB")
        prefs.set("IMG_one", "OFAV", "projA", "u9", by="AB")
        rows = prefs.load()
        check(len(rows) == 2, f"one row per (basename, label) (got {len(rows)})")
        check(prefs.get("IMG_one", "PA")["project_id"] == "projB" and prefs.get("IMG_one", "PA")["uids"] == [],
              "set replaces the row for the same key")
        check(prefs.get("IMG_one", "OFAV")["uids"] == ["u9"], "a string uid is accepted as a single uid")

        # validation
        for bad in (dict(basename="", label="PA", project_id="p", by="LO"),
                    dict(basename="I", label="", project_id="p", by="LO"),
                    dict(basename="I", label="PA", project_id="", by="LO"),
                    dict(basename="I", label="PA", project_id="p", by="  ")):
            try:
                prefs.set(bad["basename"], bad["label"], bad["project_id"], [], by=bad["by"])
                check(False, f"set rejects blank fields ({bad})")
            except ValueError:
                check(True, f"set rejects blank fields ({bad})")

        # clear
        check(prefs.clear("IMG_one", "PA") is True, "clear removes an existing row")
        check(prefs.clear("IMG_one", "PA") is False, "clear on a missing row is a no-op")
        check([r["label"] for r in prefs.load()] == ["OFAV"], "the other row survives clear")

        # validate_choice
        masks = [
            {"uid": "a1", "species": "PA", "status": "accepted", "review": False},
            {"uid": "a2", "species": "PA", "status": "rejected", "review": False},
            {"uid": "a3", "species": "PA", "status": "accepted", "review": True},
            {"uid": "a4", "species": "OFAV", "status": "accepted", "review": False},
        ]
        check([m["uid"] for m in exportable_masks(masks, "PA")] == ["a1"],
              "exportable_masks keeps accepted, non-review masks of the label only")
        check(validate_choice(masks, "PA", []) == "", "valid: project has an accepted PA mask")
        check(validate_choice(masks, "PA", ["a1"]) == "", "valid: uid among the accepted PA masks")
        check(validate_choice(masks, "PA", ["a2"]) != "", "invalid: rejected uid")
        check(validate_choice(masks, "PA", ["zz"]) != "", "invalid: unknown uid")
        check(validate_choice(masks, "MC", []) != "", "invalid: no accepted mask of that label")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check(not os.path.exists(root), "temp fixture removed")

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL TESTS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()

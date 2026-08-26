#!/usr/bin/env python
"""
Recode regression + adversarial tests for the Label Manager "Re-run recode".

Run with the orchestrator env python:
  github_repo/env/bin/python scripts/pipeline_orchestrator/tests/test_remap_loader_recode.py

Background: the Label Manager (scripts/_labels) composes remap logs in the shape
{"old": CODE, "new": CODE} (code-only; names live in the canonical vocabulary).
The recode consumer remap_loader.apply_remaps historically read each remap entry
as rm["old_code"] / rm["new_code"] / rm["new_name"] / rm["new_category"], so a
manager-composed log crashed with KeyError('old_code') and the "Re-run recode"
popup reported a failure. The fix makes apply_remaps tolerant of the manager
shape AND derive missing names/categories from the master_codes vocabulary so a
code-only remap never blanks them. These tests pin that behavior.

No emoji, no em dashes. Uses a simple check()/print harness; exits nonzero on
any failure. Never touches real project data (everything in a tmp dir).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Make the orchestrator package importable (import remap_loader) and scripts/
# importable (import _labels) regardless of CWD.
THIS = Path(__file__).resolve()
ORCH_DIR = THIS.parent.parent          # .../scripts/pipeline_orchestrator
SCRIPTS_DIR = ORCH_DIR.parent          # .../scripts
sys.path.insert(0, str(ORCH_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import remap_loader  # noqa: E402
from _labels import labels_io  # noqa: E402


_RESULTS: list[str] = []


def check(label, ok, note=""):
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {label}" + (f" -- {note}" if note else "")
    _RESULTS.append((ok, line))
    print(line)


def _fixture(tmp):
    """Write a tiny all_points + master_codes and return their paths + out dir."""
    ap = os.path.join(tmp, "all_points.csv")
    mc = os.path.join(tmp, "master_codes.csv")
    out = os.path.join(tmp, "output")
    os.makedirs(out, exist_ok=True)
    # species_code/species_name/category are the columns apply_remaps touches.
    with open(ap, "w", encoding="utf-8") as fh:
        fh.write("species_code,species_name,category\n")
        fh.write("DIPL,Diploria,Coral\n")
        fh.write("DIPL,Diploria,Coral\n")
        fh.write("OFRA,Orbicella franksi,Coral\n")
        fh.write("BL,Bleaching,Condition\n")
    with open(mc, "w", encoding="utf-8") as fh:
        fh.write("code,category,name\n")
        fh.write("DIPL,Coral,Diploria\n")
        fh.write("OFRA,Coral,Orbicella franksi\n")
        fh.write("BL,Condition,Bleaching\n")
    return ap, mc, out


def _write_log(out, remaps, excludes=None, note="manager test"):
    """Write a remap log using the SAME core the Label Manager uses."""
    res = labels_io.write_new_remap_log(out, remaps, excludes or [], source_note=note)
    return res["path"]


def _recoded_points(out):
    return pd.read_csv(os.path.join(out, "all_points_recoded.csv"), dtype=str).fillna("")


def _recoded_codes(out):
    return pd.read_csv(os.path.join(out, "master_codes_recoded.csv"), dtype=str).fillna("")


# ── 1. Reproduction: manager shape {old,new} must not crash ───────────────────
def test_manager_shape_does_not_crash():
    with tempfile.TemporaryDirectory() as tmp:
        ap, mc, out = _fixture(tmp)
        log = _write_log(out, [{"old": "DIPL", "new": "OFRA"}])
        try:
            res = remap_loader.apply_remaps(ap, mc, log, out)
            crashed = False
        except Exception as exc:  # noqa: BLE001
            crashed = True
            res = {"error": repr(exc)}
        check("manager-shape {old,new} recode does not crash",
              not crashed and isinstance(res, dict) and res.get("success"),
              note=res.get("error", "ok"))


# ── 2. No-blank: a code-only remap adopts the new code's vocabulary name ───────
def test_manager_shape_fills_name_from_vocabulary():
    with tempfile.TemporaryDirectory() as tmp:
        ap, mc, out = _fixture(tmp)
        log = _write_log(out, [{"old": "DIPL", "new": "OFRA"}])
        try:
            remap_loader.apply_remaps(ap, mc, log, out)
            pts = _recoded_points(out)
            # All former DIPL rows are now OFRA with OFRA's canonical name/cat.
            no_dipl = "DIPL" not in set(pts["species_code"])
            remapped = pts[pts["species_code"] == "OFRA"]
            names_ok = set(remapped["species_name"]) == {"Orbicella franksi"}
            blanks = (remapped["species_name"] == "").any()
            ok = no_dipl and names_ok and not blanks
            note = f"names={sorted(set(remapped['species_name']))}"
        except Exception as exc:  # noqa: BLE001
            ok, note = False, repr(exc)
        check("code-only remap fills species_name from vocabulary (no blanks)", ok, note)


# ── 3. Canonical shape still works unchanged (regression guard) ───────────────
def test_canonical_shape_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        ap, mc, out = _fixture(tmp)
        log = os.path.join(out, "remap_log_canonical.json")
        json.dump({"timestamp": "20260625_000000", "excludes": [], "remaps": [
            {"old_code": "DIPL", "new_code": "OFRA",
             "new_name": "Orbicella franksi", "new_category": "Coral"},
        ]}, open(log, "w", encoding="utf-8"))
        try:
            res = remap_loader.apply_remaps(ap, mc, log, out)
            pts = _recoded_points(out)
            ok = (res.get("success") and "DIPL" not in set(pts["species_code"])
                  and set(pts[pts["species_code"] == "OFRA"]["species_name"]) == {"Orbicella franksi"})
            note = "ok"
        except Exception as exc:  # noqa: BLE001
            ok, note = False, repr(exc)
        check("canonical {old_code,new_code,...} shape still recodes", ok, note)


# ── 4. Alias {from,to} is accepted ────────────────────────────────────────────
def test_from_to_alias():
    with tempfile.TemporaryDirectory() as tmp:
        ap, mc, out = _fixture(tmp)
        log = os.path.join(out, "remap_log_fromto.json")
        json.dump({"timestamp": "20260625_000001", "excludes": [],
                   "remaps": [{"from": "DIPL", "to": "OFRA"}]},
                  open(log, "w", encoding="utf-8"))
        try:
            res = remap_loader.apply_remaps(ap, mc, log, out)
            pts = _recoded_points(out)
            ok = res.get("success") and "DIPL" not in set(pts["species_code"])
            note = "ok"
        except Exception as exc:  # noqa: BLE001
            ok, note = False, repr(exc)
        check("{from,to} alias is accepted", ok, note)


# ── 5. Malformed entry is skipped, not fatal; valid entries still apply ───────
def test_malformed_entry_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        ap, mc, out = _fixture(tmp)
        log = os.path.join(out, "remap_log_malformed.json")
        json.dump({"timestamp": "20260625_000002", "excludes": [], "remaps": [
            {"new": "OFRA"},          # no old code -> skip
            {},                        # empty -> skip
            {"old": "DIPL", "new": "OFRA"},  # valid -> apply
        ]}, open(log, "w", encoding="utf-8"))
        try:
            res = remap_loader.apply_remaps(ap, mc, log, out)
            pts = _recoded_points(out)
            ok = res.get("success") and "DIPL" not in set(pts["species_code"])
            note = "ok"
        except Exception as exc:  # noqa: BLE001
            ok, note = False, repr(exc)
        check("malformed remap entries are skipped, valid ones still apply", ok, note)


# ── 6. Exclude-only log (no remaps) does not crash ────────────────────────────
def test_exclude_only_log():
    with tempfile.TemporaryDirectory() as tmp:
        ap, mc, out = _fixture(tmp)
        log = _write_log(out, [], excludes=["BL"])
        try:
            res = remap_loader.apply_remaps(ap, mc, log, out)
            ok = bool(res.get("success"))
            note = "ok"
        except Exception as exc:  # noqa: BLE001
            ok, note = False, repr(exc)
        check("exclude-only remap log does not crash", ok, note)


# ── 7. Integration via labels_io.run_recode (the popup's backend path) ────────
def test_run_recode_integration_manager_log():
    with tempfile.TemporaryDirectory() as tmp:
        ap, mc, out = _fixture(tmp)
        log = _write_log(out, [{"old": "DIPL", "new": "OFRA"}])
        # run_recode takes an explicit remap log path (the blueprint resolves
        # "latest"); pass the manager-composed log we just wrote.
        res = labels_io.run_recode(ap, mc, log, out, backup=True)
        check("labels_io.run_recode succeeds on a manager-composed log",
              bool(res.get("ok")), note=res.get("error", "ok"))


# ── 8. End-to-end through the Flask blueprint /api/recode (the popup) ─────────
def test_blueprint_recode_route():
    from flask import Flask
    from _labels import make_blueprint
    with tempfile.TemporaryDirectory() as tmp:
        ap, mc, out = _fixture(tmp)
        _write_log(out, [{"old": "DIPL", "new": "OFRA"}])
        provider = lambda: {  # noqa: E731
            "master_codes_csv": mc,
            "duplicate_master_codes": "",
            "all_points_csv": ap,
            "recode_output_dir": out,
            "pipeline_yaml": "",
        }
        app = Flask(__name__)
        app.register_blueprint(make_blueprint(provider), url_prefix="/labels")
        client = app.test_client()
        r = client.post("/labels/api/recode", json={"confirm": True})
        body = r.get_json() or {}
        check("POST /labels/api/recode (confirmed) returns ok",
              r.status_code == 200 and body.get("ok") is True,
              note=f"status={r.status_code} body={ {k: body.get(k) for k in ('error','remap_log_path')} }")


def main():
    test_manager_shape_does_not_crash()
    test_manager_shape_fills_name_from_vocabulary()
    test_canonical_shape_unchanged()
    test_from_to_alias()
    test_malformed_entry_skipped()
    test_exclude_only_log()
    test_run_recode_integration_manager_log()
    test_blueprint_recode_route()
    print("\n=== Summary ===")
    failures = [line for ok, line in _RESULTS if not ok]
    for ok, line in _RESULTS:
        print(line)
    print(f"\n{len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

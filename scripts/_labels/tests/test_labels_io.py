#!/usr/bin/env python
"""
No-pytest test harness for labels_io.

Run with the unified env python:
  env/bin/python scripts/_labels/tests/test_labels_io.py

SAFETY: this test NEVER touches the real canonical or duplicate files. It
copies the real canonical CSV into a throwaway tmp dir, makes a tmp dup, and
exercises every mutation against those copies only. The real files are read
once (to copy) and asserted untouched at the end.

Uses a simple check()/print harness and exits nonzero on any failure.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Make the _labels package importable (scripts/ on sys.path -> import _labels).
THIS = Path(__file__).resolve()
SCRIPTS_DIR = THIS.parent.parent.parent  # .../scripts
sys.path.insert(0, str(SCRIPTS_DIR))

from _labels import labels_io  # noqa: E402


REAL_CANONICAL = Path(
    "/mnt/rip/vicarius_drive/vicarius/_METADATA/library/definitions/tcrmp_species_codes.csv"
)
REAL_DUP = Path(
    "/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/"
    "supporting_data/master_codes.csv"
)


# ───────────────────────────── tiny harness ──────────────────────────────────

_failures: list[str] = []
_passes = 0


def check(cond, label):
    global _passes
    if cond:
        _passes += 1
        print(f"  ok   {label}")
    else:
        _failures.append(label)
        print(f"  FAIL {label}")


def section(title):
    print(f"\n== {title} ==")


# ───────────────────────────── fixtures ──────────────────────────────────────

def make_tmp_copies(tmp):
    """Copy the real canonical into tmp canonical + tmp dup (byte-identical)."""
    tmp_canonical = Path(tmp) / "tcrmp_species_codes.csv"
    tmp_dup = Path(tmp) / "supporting_data" / "master_codes.csv"
    tmp_dup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REAL_CANONICAL, tmp_canonical)
    shutil.copy2(REAL_CANONICAL, tmp_dup)
    return tmp_canonical, tmp_dup


# ───────────────────────────── tests ─────────────────────────────────────────

def main():
    # Snapshot the real files up front so we can prove they are untouched.
    real_canonical_bytes = REAL_CANONICAL.read_bytes()
    real_dup_bytes = REAL_DUP.read_bytes()

    tmp = tempfile.mkdtemp(prefix="labels_io_test_")
    try:
        tmp_canonical, tmp_dup = make_tmp_copies(tmp)

        # ---- read_vocabulary ----
        section("read_vocabulary")
        vocab = labels_io.read_vocabulary(tmp_canonical)
        check(len(vocab) == 130, f"returns 130 dicts (got {len(vocab)})")
        check(
            all(set(r.keys()) == {"code", "category", "name"} for r in vocab),
            "every row has exactly keys code/category/name",
        )
        check(vocab[0]["code"] == "AA", f"first code is AA (got {vocab[0]['code']})")

        # ---- validate_row refusals ----
        section("validate_row refusals")
        existing = [r["code"] for r in vocab]

        ok, err = labels_io.validate_row(
            {"code": "AA", "category": "Coral", "name": "dup"},
            existing,
            is_new=True,
        )
        check(not ok, f"refuses duplicate code on add ({err!r})")

        ok, err = labels_io.validate_row(
            {"code": "ZZNEW", "category": "Coral", "name": ""},
            existing,
            is_new=True,
        )
        check(not ok, f"refuses empty name ({err!r})")

        ok, err = labels_io.validate_row(
            {"code": "ZZNEW", "category": "", "name": "x"},
            existing,
            is_new=True,
        )
        check(not ok, f"refuses empty category ({err!r})")

        ok, err = labels_io.validate_row(
            {"code": "zznew", "category": "Coral", "name": "x"},
            existing,
            is_new=True,
        )
        check(not ok, f"refuses lowercase code ({err!r})")

        # Code change / delete attempt: editing a code that does not exist is
        # how a "rename to a new code" or "delete then re-add" would surface,
        # and it is refused. (is_new=False requires the code to already exist.)
        ok, err = labels_io.validate_row(
            {"code": "DOESNOTEXIST", "category": "Coral", "name": "x"},
            existing,
            is_new=False,
        )
        check(not ok, f"refuses edit of a non-existent code / code change ({err!r})")

        # ---- add_or_edit: ADD a new row ----
        section("add_or_edit ADD")
        pre_write_bytes = tmp_canonical.read_bytes()
        res = labels_io.add_or_edit(
            tmp_canonical,
            tmp_dup,
            code="ZZTEST",
            category="Coral",
            name="Test species",
            is_new=True,
        )
        check(res["ok"] is True, "add returns ok=True")
        check(res["mirrored"] is True, "add reports mirrored=True")

        vocab2 = labels_io.read_vocabulary(tmp_canonical)
        check(len(vocab2) == 131, f"row appended -> 131 rows (got {len(vocab2)})")
        added = [r for r in vocab2 if r["code"] == "ZZTEST"]
        check(
            len(added) == 1
            and added[0]["category"] == "Coral"
            and added[0]["name"] == "Test species",
            "ZZTEST row present with correct category/name",
        )

        backup = Path(res["backup_path"])
        check(backup.exists(), f"backup file exists ({backup})")
        check(
            backup.read_bytes() == pre_write_bytes,
            "backup equals the pre-write canonical content",
        )
        check(
            tmp_canonical.read_bytes() == tmp_dup.read_bytes(),
            "tmp canonical == tmp dup byte-for-byte after ADD",
        )

        # ---- add_or_edit: EDIT an existing row ----
        section("add_or_edit EDIT")
        res2 = labels_io.add_or_edit(
            tmp_canonical,
            tmp_dup,
            code="AA",
            category="Coral",
            name="Agaricia agaricites EDITED",
            is_new=False,
        )
        check(res2["ok"] is True, "edit returns ok=True")
        vocab3 = labels_io.read_vocabulary(tmp_canonical)
        aa = [r for r in vocab3 if r["code"] == "AA"][0]
        check(aa["name"] == "Agaricia agaricites EDITED", "AA name updated")
        check(aa["code"] == "AA", "AA code unchanged")
        check(len(vocab3) == 131, f"edit did not add/remove rows ({len(vocab3)})")
        check(
            tmp_canonical.read_bytes() == tmp_dup.read_bytes(),
            "tmp canonical == tmp dup byte-for-byte after EDIT",
        )

        # ---- add_or_edit raises on validation failure ----
        section("add_or_edit raises ValueError")
        raised = False
        try:
            labels_io.add_or_edit(
                tmp_canonical, tmp_dup,
                code="AA", category="Coral", name="dup", is_new=True,
            )
        except ValueError:
            raised = True
        check(raised, "add_or_edit raises ValueError on duplicate-code add")

        # ---- remap logs ----
        section("remap logs")
        recode_dir = Path(tmp) / "recode_output"
        recode_dir.mkdir(parents=True, exist_ok=True)
        hand = recode_dir / "remap_log_x.json"
        hand.write_text(
            json.dumps(
                {
                    "timestamp": "20260101_000000",
                    "new_code_count": 42,
                    "remaps": [
                        {"old_code": "MA", "new_code": "MAU"},
                        {"old_code": "MB", "new_code": "MBU"},
                    ],
                    "excludes": ["BL"],
                }
            )
        )
        logs = labels_io.list_remap_logs(recode_dir)
        check(len(logs) == 1, f"list_remap_logs finds the hand-written log ({len(logs)})")
        check(logs[0]["n_remaps"] == 2, f"parsed n_remaps == 2 (got {logs[0]['n_remaps']})")
        check(
            logs[0]["new_code_count"] == 42,
            f"parsed new_code_count == 42 (got {logs[0]['new_code_count']})",
        )

        full = labels_io.read_remap_log(hand)
        check(full.get("timestamp") == "20260101_000000", "read_remap_log returns full dict")

        # ---- write_new_remap_log never overwrites ----
        section("write_new_remap_log")
        w1 = labels_io.write_new_remap_log(
            recode_dir,
            [{"old_code": "A", "new_code": "B"}],
            ["BL"],
            source_note="first",
        )
        w2 = labels_io.write_new_remap_log(
            recode_dir,
            [{"old_code": "C", "new_code": "D"}],
            [],
            source_note="second",
        )
        check(Path(w1["path"]).exists(), "first new remap log written")
        check(Path(w2["path"]).exists(), "second new remap log written")
        check(w1["path"] != w2["path"], "two calls produce two distinct files")
        new_logs = sorted(recode_dir.glob("remap_log_2*.json"))
        # exclude the hand-written remap_log_x.json (does not start with a digit)
        written = [p for p in recode_dir.glob("remap_log_*.json") if p.name != "remap_log_x.json"]
        check(len(written) == 2, f"exactly 2 newly written logs exist (got {len(written)})")

        # ---- get_locations ----
        section("get_locations")
        pipeline_yaml = (
            SCRIPTS_DIR.parent / "config" / "pipeline.yaml"
        )
        loc = labels_io.get_locations(pipeline_yaml)
        check(
            loc["master_codes_csv"] == str(REAL_CANONICAL),
            f"master_codes_csv resolves to canonical ({loc['master_codes_csv']})",
        )
        check(
            loc["duplicate_master_codes"] == str(REAL_DUP),
            f"duplicate_master_codes resolves to dup ({loc['duplicate_master_codes']})",
        )
        check(
            loc["recode_output_dir"].endswith("TCRMPcvr_recodeSpecies/output"),
            f"recode_output_dir points at recode output ({loc['recode_output_dir']})",
        )
        check(
            "${" not in (loc["all_points_csv"] or ""),
            "all_points_csv fully resolved (no ${...})",
        )

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- prove the REAL files are untouched ----
    section("real files untouched")
    check(
        REAL_CANONICAL.read_bytes() == real_canonical_bytes,
        "real canonical csv byte-identical to pre-test snapshot",
    )
    check(
        REAL_DUP.read_bytes() == real_dup_bytes,
        "real duplicate csv byte-identical to pre-test snapshot",
    )

    # ---- summary ----
    print(f"\n{_passes} passed, {len(_failures)} failed")
    if _failures:
        print("FAILURES:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

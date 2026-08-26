"""
labels_io.py — the testable IO core of the Reef Point Seg label manager.

Pure-python (stdlib only, plus an optional defensive import of pandas via
remap_loader). No Flask, no UI. Everything here is exercised by
tests/test_labels_io.py with throwaway tmp copies of the real files.

SAFETY POSTURE (binding, Lauren chose "add + edit, backup, manual recode"):
  - Vocabulary edits ADD new rows or EDIT the name/category of existing rows.
    Never delete a row. Never change an existing code. Codes are referenced
    everywhere downstream, so they are immutable once minted.
  - Every canonical write backs up the current canonical CSV first, to a
    timestamped (AST, fixed UTC-4) file under a backups/ dir beside it, then
    writes the new canonical atomically, then mirrors the exact bytes to the
    duplicate master_codes.csv so the two files stay byte-identical.
  - Recode re-run is manual and never automatic on save. run_recode backs up
    the prior recode outputs before regenerating.

All timestamps are AST (Atlantic Standard Time, fixed UTC-4, no DST).
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


# Atlantic Standard Time is a fixed UTC-4 offset year round (no daylight saving).
AST = timezone(timedelta(hours=-4), name="AST")

FIELDNAMES = ["code", "category", "name"]


# ───────────────────────────── time helpers ──────────────────────────────────

def _ast_now() -> datetime:
    return datetime.now(AST)


def ast_compact_stamp(now: datetime | None = None) -> str:
    """Compact AST stamp for filenames, e.g. 20260625_142233."""
    return (now or _ast_now()).strftime("%Y%m%d_%H%M%S")


# ───────────────────────────── vocabulary read ───────────────────────────────

def read_vocabulary(master_codes_csv) -> list[dict]:
    """Return [{"code","category","name"}, ...] preserving file order.

    Tolerant of extra columns in the source file; only the three canonical
    keys are returned. Header is required and must contain code/category/name.
    """
    rows: list[dict] = []
    with open(master_codes_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return rows
        missing = [c for c in FIELDNAMES if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"master_codes csv missing columns {missing}; "
                f"found {reader.fieldnames}"
            )
        for raw in reader:
            rows.append(
                {
                    "code": (raw.get("code") or "").strip(),
                    "category": (raw.get("category") or "").strip(),
                    "name": (raw.get("name") or "").strip(),
                }
            )
    return rows


# ───────────────────────────── validation ────────────────────────────────────

def validate_row(row, existing_codes, *, is_new) -> tuple[bool, str]:
    """Validate a single add/edit row.

    existing_codes: iterable of codes already present in the vocabulary.
    is_new=True  -> the code must NOT already exist (we are appending).
    is_new=False -> the code MUST already exist (we are editing name/category);
                    the code itself may not change and the row may not be
                    deleted (deletion is never offered by this core).

    Rules:
      - code: required, uppercase, non-empty, unique on add / present on edit.
      - name: required, non-empty.
      - category: required, non-empty.
    """
    code = (row.get("code") or "").strip()
    category = (row.get("category") or "").strip()
    name = (row.get("name") or "").strip()
    existing = set(existing_codes)

    if not code:
        return False, "code is required and cannot be empty"
    if code != code.upper():
        return False, f"code must be uppercase ('{code}' is not)"
    if not name:
        return False, "name is required and cannot be empty"
    if not category:
        return False, "category is required and cannot be empty"

    if is_new:
        if code in existing:
            return False, f"code '{code}' already exists; codes must be unique"
    else:
        if code not in existing:
            return False, (
                f"code '{code}' does not exist; cannot edit a missing code "
                f"(changing or deleting a code is not allowed)"
            )

    return True, ""


# ─────────────────────────── canonical mutation ──────────────────────────────

def _rows_to_csv_bytes(rows) -> bytes:
    """Serialize rows to canonical CSV bytes (LF newlines, header first)."""
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, "") for k in FIELDNAMES})
    return buf.getvalue().encode("utf-8")


def _atomic_write_bytes(target, data: bytes) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


def _backup_canonical(master_codes_csv, now: datetime | None = None) -> str:
    """Copy the current canonical CSV to backups/ with an AST timestamp.

    Returns the backup path. Backs up the bytes as they exist on disk right
    now (the pre-write content).
    """
    src = Path(master_codes_csv)
    backups_dir = src.parent / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    stamp = ast_compact_stamp(now)
    backup = backups_dir / f"{src.stem}.{stamp}.bak{src.suffix}"
    # Avoid clobbering if two writes land in the same second.
    n = 1
    while backup.exists():
        backup = backups_dir / f"{src.stem}.{stamp}_{n}.bak{src.suffix}"
        n += 1
    shutil.copy2(src, backup)
    return str(backup)


def add_or_edit(master_codes_csv, dup_csv, *, code, category, name, is_new) -> dict:
    """Add a new vocabulary row or edit an existing row's name/category.

    Steps, in order:
      1. Read current vocabulary, validate the requested change.
      2. Back up the current canonical CSV to backups/ (timestamped, AST).
      3. Apply the change in memory (append on is_new, else update the matching
         code's name/category; refuse any code change or deletion).
      4. Atomically write the new canonical CSV.
      5. Mirror the exact bytes to dup_csv so the two stay byte-identical.

    Returns {"ok", "backup_path", "row", "mirrored"}.
    Raises ValueError on validation failure (before any backup or write).
    """
    master_codes_csv = str(master_codes_csv)
    rows = read_vocabulary(master_codes_csv)
    existing = [r["code"] for r in rows]

    new_row = {
        "code": (code or "").strip(),
        "category": (category or "").strip(),
        "name": (name or "").strip(),
    }
    ok, err = validate_row(new_row, existing, is_new=is_new)
    if not ok:
        raise ValueError(err)

    # Back up BEFORE mutating anything on disk.
    backup_path = _backup_canonical(master_codes_csv)

    if is_new:
        rows.append(new_row)
    else:
        # Edit name/category of the matching code; the code itself is unchanged.
        matched = False
        for r in rows:
            if r["code"] == new_row["code"]:
                r["name"] = new_row["name"]
                r["category"] = new_row["category"]
                matched = True
                break
        if not matched:
            # Should not happen (validate_row already checked), but be defensive.
            raise ValueError(
                f"code '{new_row['code']}' vanished between validate and write"
            )

    data = _rows_to_csv_bytes(rows)
    _atomic_write_bytes(master_codes_csv, data)

    mirrored = False
    if dup_csv:
        _atomic_write_bytes(str(dup_csv), data)
        mirrored = True

    return {
        "ok": True,
        "backup_path": backup_path,
        "row": new_row,
        "mirrored": mirrored,
    }


# ───────────────────────────── remap logs ────────────────────────────────────

def read_remap_log(path) -> dict:
    """Load and return a full remap_log JSON."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _remap_log_meta(path) -> dict:
    """Parse the header of one remap_log_*.json into a list-row dict."""
    p = Path(path)
    try:
        data = read_remap_log(path)
        return {
            "path": str(p),
            "name": p.name,
            "timestamp": data.get("timestamp", ""),
            "n_remaps": len(data.get("remaps", []) or []),
            "new_code_count": data.get("new_code_count"),
        }
    except (json.JSONDecodeError, OSError):
        return {
            "path": str(p),
            "name": p.name,
            "timestamp": "",
            "n_remaps": 0,
            "new_code_count": None,
        }


def list_remap_logs(recode_output_dir) -> list[dict]:
    """List parsed remap_log_*.json headers, newest first.

    Sort key is the parsed timestamp string (compact AST stamps sort
    lexicographically by time); ties fall back to filename.
    """
    d = Path(recode_output_dir)
    if not d.is_dir():
        return []
    metas = [
        _remap_log_meta(p)
        for p in d.glob("remap_log_*.json")
        if p.is_file()
    ]
    metas.sort(key=lambda m: (m.get("timestamp") or "", m.get("name") or ""), reverse=True)
    return metas


def write_new_remap_log(recode_output_dir, remaps, excludes, *, source_note="") -> dict:
    """Write a NEW timestamped remap_log_<AST compact>.json. Never overwrites.

    If a file for the current second already exists, a numeric suffix is added
    so an existing log is never clobbered. Returns {"path","timestamp"}.
    """
    d = Path(recode_output_dir)
    d.mkdir(parents=True, exist_ok=True)
    now = _ast_now()
    stamp = ast_compact_stamp(now)

    target = d / f"remap_log_{stamp}.json"
    n = 1
    while target.exists():
        target = d / f"remap_log_{stamp}_{n}.json"
        n += 1

    payload = {
        "timestamp": stamp,
        "source_note": source_note,
        "excludes": list(excludes or []),
        "remaps": list(remaps or []),
    }
    _atomic_write_bytes(target, json.dumps(payload, indent=2).encode("utf-8"))
    return {"path": str(target), "timestamp": stamp}


# ───────────────────────────── locations ─────────────────────────────────────

def get_locations(pipeline_yaml) -> dict:
    """Resolve label-manager file locations from pipeline.yaml.

    Returns:
      {
        "master_codes_csv": <canonical csv>,
        "all_points_csv": <all points csv>,
        "supporting_data_dir": <dir>,
        "recode_output_dir": <TCRMPcvr_recodeSpecies/output>,
        "duplicate_master_codes": <supporting_data_dir/master_codes.csv>,
      }

    Resolves ${repo_root} and ${...} path references the same way the
    orchestrator does (repo_root is two dirs above the config file).
    """
    import re

    import yaml  # local import; only needed when reading the yaml

    cfg_path = Path(pipeline_yaml).resolve()
    raw = yaml.safe_load(cfg_path.read_text())
    repo_root = str(cfg_path.parent.parent)

    paths = dict(raw.get("paths", {}) or {})

    def interp(value, scope):
        if not isinstance(value, str):
            return value
        for _ in range(4):
            new = re.sub(
                r"\$\{(\w+)\}",
                lambda m: str(scope.get(m.group(1), m.group(0))),
                value,
            )
            if new == value:
                break
            value = new
        return value

    # Resolve the paths block against repo_root and itself (multi-pass).
    scope = {"repo_root": repo_root}
    resolved = {k: interp(v, scope) for k, v in paths.items()}
    for _ in range(4):
        scope = {"repo_root": repo_root, **resolved}
        new_resolved = {k: interp(v, scope) for k, v in resolved.items()}
        if new_resolved == resolved:
            break
        resolved = new_resolved

    supporting = resolved.get("supporting_data_dir", "")
    master = resolved.get("master_codes_csv", "")
    all_points = resolved.get("all_points_csv", "")
    recode_output_dir = str(
        Path(repo_root) / "scripts" / "TCRMPcvr_recodeSpecies" / "output"
    )
    duplicate = str(Path(supporting) / "master_codes.csv") if supporting else ""

    return {
        "master_codes_csv": master,
        "all_points_csv": all_points,
        "supporting_data_dir": supporting,
        "recode_output_dir": recode_output_dir,
        "duplicate_master_codes": duplicate,
    }


# ───────────────────────────── recode (manual) ───────────────────────────────

def _backup_prior_recode_outputs(output_dir, now: datetime | None = None) -> str | None:
    """Move/copy prior recode outputs into a timestamped subdir.

    Backs up the recoded CSVs and remap_log.json that apply_remaps would
    overwrite. Returns the backup dir path, or None if there was nothing to
    back up. The browseable remap_log_*.json files are left in place.
    """
    out = Path(output_dir)
    if not out.is_dir():
        return None
    targets = ["all_points_recoded.csv", "master_codes_recoded.csv", "remap_log.json"]
    present = [t for t in targets if (out / t).exists()]
    if not present:
        return None
    stamp = ast_compact_stamp(now)
    backup_dir = out / "backups" / stamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    for t in present:
        shutil.copy2(out / t, backup_dir / t)
    return str(backup_dir)


def run_recode(all_points_csv, master_codes_csv, remap_log_path, output_dir, *, backup=True) -> dict:
    """Manually re-run the recode by delegating to remap_loader.apply_remaps.

    This is the explicit, button-driven recode; it is never called on save.
    When backup=True the prior recode outputs are copied to a timestamped
    backups/ subdir before apply_remaps regenerates them.

    Returns {"ok", "outputs", "summary"}. On any failure (import or call) it
    returns {"ok": False, "error": ...} rather than raising, so the UI can
    surface the message without crashing.
    """
    backup_dir = None
    try:
        if backup:
            backup_dir = _backup_prior_recode_outputs(output_dir)
    except Exception as exc:  # noqa: BLE001 - defensive, report not crash
        return {
            "ok": False,
            "error": f"failed to back up prior recode outputs: {exc}",
            "outputs": {},
            "summary": {},
        }

    # Defensive import: remap_loader lives in the orchestrator package and
    # depends on pandas. Keep it lazy and wrapped.
    orch_dir = str(
        Path(__file__).resolve().parent.parent / "pipeline_orchestrator"
    )
    if orch_dir not in sys.path:
        sys.path.insert(0, orch_dir)
    try:
        import remap_loader  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"could not import remap_loader: {exc}",
            "outputs": {},
            "summary": {},
            "backup_dir": backup_dir,
        }

    try:
        result = remap_loader.apply_remaps(
            all_points_csv, master_codes_csv, remap_log_path, output_dir
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"apply_remaps failed: {exc}",
            "outputs": {},
            "summary": {},
            "backup_dir": backup_dir,
        }

    output_files = result.get("output_files", []) if isinstance(result, dict) else []
    outputs = {name: str(Path(output_dir) / name) for name in output_files}

    return {
        "ok": bool(result.get("success", True)) if isinstance(result, dict) else True,
        "outputs": outputs,
        "summary": result if isinstance(result, dict) else {"raw": result},
        "backup_dir": backup_dir,
    }

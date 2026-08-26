"""Pure-python aggregator for the cross-project image x label matrix.

`build_matrix(inprocess_root)` rescans the entire inprocess tree on every
call (no cache), so a deleted / purged project simply stops contributing.

Primary data source is each project's `label_provenance.csv` ledger(s).
A project with NO ledger anywhere falls back to deriving label outcomes
from its legacy step5 `segmentations.json` files.
"""

import csv
import glob
import json
import os
import sys
from datetime import datetime, timezone, timedelta

_PKG_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_SCRIPTS not in sys.path:
    sys.path.insert(0, _PKG_SCRIPTS)

from _reefreview.mask_registry import MaskRegistry  # noqa: E402

# Conflict-resolution strength: higher wins when >1 project covers the
# same (image, label) pair. pending_expert beats not_found only; any
# found_* beats pending; found_expert beats all.
_OUTCOME_STRENGTH = {
    "found_expert": 5,
    "found_manual": 4,
    "found_ai": 3,
    "found_model": 2,
    "pending_expert": 1,
    "not_found": 0,
}

# Mask source_types that count as a hand-drawn (manual) annotation.
_MANUAL_SOURCE_TYPES = {"manual_box", "manual_click"}


def _is_review_mask(m):
    return bool(m.get("review")) or m.get("species") == "REVIEW"


def _is_expert_mask(m):
    eid = m.get("expert_id")
    return isinstance(eid, dict) and eid.get("mode") == "EXPERT"


def _pending_labels(m):
    """Labels for a still-pending review mask: each distinct non-blank
    tentative code in mask.reviews[] excluding the synthetic 'overlap'
    reviewer, or ['REVIEW'] when there is none. Mirrors the provenance
    writers in TCRMPclip_combinedAnnotate / TCRMPclip_segmentImages."""
    codes = []
    for r in (m.get("reviews") or []):
        if not isinstance(r, dict):
            continue
        if (r.get("reviewer") or "").strip() == "overlap":
            continue
        code = (r.get("code") or "").strip()
        if code and code not in codes:
            codes.append(code)
    return codes or ["REVIEW"]

# Fixed Atlantic Standard Time (UTC-4, no DST).
_AST = timezone(timedelta(hours=-4))


def _ast_now_iso():
    """ISO timestamp pinned to AST (UTC-4, no daylight saving)."""
    return datetime.now(_AST).isoformat()


def _stronger(a, b):
    """Return the stronger of two outcome strings."""
    if a is None:
        return b
    if b is None:
        return a
    return a if _OUTCOME_STRENGTH.get(a, -1) >= _OUTCOME_STRENGTH.get(b, -1) else b


def _basename_stem(image_name):
    """Strip directory + extension; .jpeg/.jpg normalize to the same stem.

    Per the contract we simply take os.path.splitext on the basename and
    keep the stem as-is (the extension carries the .jpeg/.jpg difference,
    so dropping it normalizes them automatically).
    """
    return os.path.splitext(os.path.basename(image_name))[0]


def _read_project_json(project_dir):
    """Load project.json for a run dir, or None if absent / unreadable."""
    path = os.path.join(project_dir, "project.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _target_species(meta):
    """Parse steps.3.config.target_species ('OFRA, PA') -> ['OFRA','PA']."""
    raw = ""
    try:
        raw = meta["steps"]["3"]["config"].get("target_species", "") or ""
    except (KeyError, TypeError, AttributeError):
        raw = ""
    return [s.strip() for s in raw.split(",") if s.strip()]


def _has_model(project_dir):
    """True if any trained best.pt exists under step6_trainModel/runs."""
    pattern = os.path.join(project_dir, "step6_trainModel", "runs", "*", "weights", "best.pt")
    return bool(glob.glob(pattern))


def _ledger_rows(project_dir):
    """Yield provenance rows from every label_provenance.csv under a project.

    Returns a list of dicts: basename, label, outcome, source, reviewer, at.
    Returns None (not []) when NO ledger file exists anywhere, so callers
    can distinguish "no ledger" from "ledger with zero rows".
    """
    pattern = os.path.join(project_dir, "**", "label_provenance.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    rows = []
    for fpath in sorted(files):
        try:
            with open(fpath, "r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for raw in reader:
                    basename = _basename_stem((raw.get("basename") or "").strip())
                    label = (raw.get("label") or "").strip()
                    outcome = (raw.get("outcome") or "").strip()
                    if not basename or not label or not outcome:
                        continue
                    rows.append({
                        "basename": basename,
                        "label": label,
                        "outcome": outcome,
                        "source": (raw.get("source") or "").strip(),
                        "reviewer": (raw.get("reviewer") or "").strip(),
                        "at": (raw.get("at") or "").strip(),
                    })
        except OSError:
            continue
    return rows


def _legacy_rows(project_dir, target_species):
    """Derive provenance rows from legacy step5 segmentations.json files.

    Only used for a project with NO ledger. For each reviewed-or-exported
    image we emit one row per accepted (non-review) species (found_expert
    when an accepted expert ID is stamped), one pending_expert row per
    tentative code on each non-rejected review mask, plus a not_found row
    for every target species with no accepted or pending presence.
    """
    flat = glob.glob(os.path.join(project_dir, "**", "segmentations", "segmentations.json"), recursive=True)
    nested = glob.glob(os.path.join(project_dir, "**", "segmentations", "*", "segmentations.json"), recursive=True)
    files = sorted(set(flat) | set(nested))
    rows = []
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                seg_doc = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(seg_doc, dict):
            continue
        for img_name, seg in seg_doc.items():
            if not isinstance(seg, dict):
                continue
            if not (seg.get("reviewed") is True or seg.get("exported") is True):
                continue
            basename = _basename_stem(seg.get("image_path") or img_name)
            at = seg.get("processed_at", "") or ""

            masks = [m for m in (seg.get("masks") or []) if isinstance(m, dict)]
            accepted = [
                m for m in masks
                if m.get("status") == "accepted" and not _is_review_mask(m)
            ]
            # Review masks stay OUT of found; they only yield pending_expert.
            pending = [
                m for m in masks
                if _is_review_mask(m) and m.get("status") != "rejected"
            ]

            # Per accepted species: expert beats manual beats ai.
            by_species = {}  # code -> {"manual": bool, "expert": bool}
            for m in accepted:
                code = m.get("species")
                if not code:
                    continue
                rec = by_species.setdefault(code, {"manual": False, "expert": False})
                if m.get("source_type") in _MANUAL_SOURCE_TYPES:
                    rec["manual"] = True
                if _is_expert_mask(m):
                    rec["expert"] = True

            pending_codes = set()
            for m in pending:
                pending_codes.update(_pending_labels(m))

            present = set()
            for code, rec in by_species.items():
                present.add(code)
                if rec["expert"]:
                    outcome = "found_expert"
                elif rec["manual"]:
                    outcome = "found_manual"
                else:
                    outcome = "found_ai"
                rows.append({
                    "basename": basename,
                    "label": code,
                    "outcome": outcome,
                    "source": "derived",
                    "reviewer": "",
                    "at": at,
                })

            # Pending review codes with no accepted mask -> pending_expert.
            for code in sorted(pending_codes - present):
                rows.append({
                    "basename": basename,
                    "label": code,
                    "outcome": "pending_expert",
                    "source": "derived",
                    "reviewer": "",
                    "at": at,
                })

            # Target species with no accepted or pending mask -> not_found.
            for code in target_species:
                if code in present or code in pending_codes:
                    continue
                rows.append({
                    "basename": basename,
                    "label": code,
                    "outcome": "not_found",
                    "source": "derived",
                    "reviewer": "",
                    "at": at,
                })
    return rows


def _registry_rows(inprocess_root):
    """Yield provenance-shaped rows from the canonical cross-project mask
    registry (`<inprocess_root>/_mask_registry`).

    Returns a list of dicts: basename, label, outcome, source, reviewer, at,
    project_id. Returns None (not []) when the registry is absent or holds
    zero rows, so callers can fall back to the ledger/legacy chain exactly
    as they do today.

    outcome is derived from the same stored facts / strength semantics the
    rest of the builder uses:
      expert_mode == '1'                       -> found_expert
      review == '1' and expert_mode != '1'      -> pending_expert
      source_type in manual (click/box)         -> found_manual
      source_type in {auto, exemplar, model}    -> found_ai
    Rows with status != 'accepted' are skipped UNLESS they are review masks
    (review == '1'), which surface as pending_expert regardless of status.
    """
    registry_dir = os.path.join(inprocess_root, "_mask_registry")
    reg = MaskRegistry(registry_dir)
    try:
        records = reg.rows()
    except OSError:
        return None
    if not records:
        return None

    rows = []
    for rec in records:
        basename = _basename_stem((rec.get("basename") or rec.get("source_image") or "").strip())
        label = (rec.get("species") or rec.get("category") or "").strip()
        if not basename or not label:
            continue

        is_review = rec.get("review") == "1"
        is_expert = rec.get("expert_mode") == "1"
        status = (rec.get("status") or "").strip()
        source_type = (rec.get("source_type") or "").strip()

        if is_expert:
            outcome = "found_expert"
        elif is_review:
            outcome = "pending_expert"
        elif status != "accepted":
            # Non-review mask that never reached accepted status contributes
            # nothing (mirrors the ledger/legacy contract: only accepted or
            # pending masks are represented).
            continue
        elif source_type in _MANUAL_SOURCE_TYPES:
            outcome = "found_manual"
        elif source_type in {"auto", "exemplar", "model"}:
            outcome = "found_ai"
        else:
            # Accepted, non-manual, non-recognized source_type: treat as AI
            # derived (matches the legacy fallback's default for accepted,
            # non-manual, non-expert masks).
            outcome = "found_ai"

        rows.append({
            "basename": basename,
            "label": label,
            "outcome": outcome,
            "source": "registry",
            "reviewer": "",
            "at": (rec.get("updated_at") or rec.get("created_at") or "").strip(),
            "project_id": (rec.get("project_id") or "").strip(),
        })
    return rows


def build_matrix(inprocess_root):
    """Aggregate every project under inprocess_root into a sparse matrix.

    Returns the dict described by the matrix JSON contract. Rescans the
    tree on every call (no cache), so purged projects vanish naturally.
    """
    projects = []
    cells = {}  # basename -> label -> {"outcome", "sources"}

    registry_rows = _registry_rows(inprocess_root)
    registry_by_project = {}
    if registry_rows:
        for row in registry_rows:
            registry_by_project.setdefault(row["project_id"], []).append(row)

    run_dirs = sorted(glob.glob(os.path.join(inprocess_root, "run_*")))
    for project_dir in run_dirs:
        if not os.path.isdir(project_dir):
            continue
        meta = _read_project_json(project_dir)
        if meta is None:
            # No project.json -> not a real project; ignore (e.g. junk dirs).
            continue

        dir_name = os.path.basename(project_dir)
        project_id = meta.get("id") or dir_name
        name = meta.get("name") or dir_name
        target_species = _target_species(meta)
        has_model = _has_model(project_dir)

        projects.append({
            "project_id": project_id,
            "name": name,
            "target_species": target_species,
            "has_model": has_model,
        })

        rows = _ledger_rows(project_dir)
        if rows is None:
            rows = _legacy_rows(project_dir, target_species)

        if project_id in registry_by_project:
            # This project has registry rows: the registry is the source of
            # truth for found-*/pending-* cells for THIS project only. A
            # sibling project with no registry rows of its own keeps its
            # ledger/legacy result untouched (registry preference is
            # per-project, not a single global switch).
            # not_found still comes from the existing ledger/legacy chain
            # (target_species x reviewed image set), exactly as today.
            not_found_rows = [r for r in rows if r["outcome"] == "not_found"]
            rows = list(registry_by_project[project_id]) + not_found_rows

        for row in rows:
            basename = row["basename"]
            label = row["label"]
            outcome = row["outcome"]
            cell = cells.setdefault(basename, {}).setdefault(
                label, {"outcome": None, "sources": []}
            )
            cell["sources"].append({
                "project_id": project_id,
                "name": name,
                "source": row["source"],
                "reviewer": row["reviewer"],
                "at": row["at"],
                "outcome": outcome,
            })
            cell["outcome"] = _stronger(cell["outcome"], outcome)

    # Unions.
    labels = set()
    images = set()
    for basename, by_label in cells.items():
        images.add(basename)
        for label in by_label:
            labels.add(label)

    labels_sorted = sorted(labels)
    images_sorted = sorted(images)

    # Stats.
    by_outcome = {
        "found_expert": 0,
        "found_manual": 0,
        "found_ai": 0,
        "found_model": 0,
        "pending_expert": 0,
        "not_found": 0,
    }
    n_reviewed_cells = 0
    for by_label in cells.values():
        for cell in by_label.values():
            outcome = cell["outcome"]
            if outcome in by_outcome:
                by_outcome[outcome] += 1
            n_reviewed_cells += 1

    return {
        "generated_at": _ast_now_iso(),
        "labels": labels_sorted,
        "images": images_sorted,
        "projects": projects,
        "cells": cells,
        "stats": {
            "n_images": len(images_sorted),
            "n_labels": len(labels_sorted),
            "n_projects": len(projects),
            "n_reviewed_cells": n_reviewed_cells,
            "by_outcome": by_outcome,
        },
    }

"""Pure-python aggregator for the cross-project image x label matrix.

`build_matrix(inprocess_root)` rescans the entire inprocess tree on every
call (no cache), so a deleted / purged project simply stops contributing.

Primary data source is each project's `label_provenance.csv` ledger(s).
A processing directory with no ledger of its own falls back to deriving
label outcomes from the legacy `segmentations.json` files under it.

Registry rows never replace a project wholesale: they are upserted over
their own (basename, label) cells, so a project whose registry holds five
of its masks still shows every other cell the ledger/legacy chain knows.
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

from _reefreview.manual_sources import MANUAL_SOURCE_TYPES  # noqa: E402
from _superseded import is_superseded_path  # noqa: E402
from _reefreview.mask_registry import MaskRegistry  # noqa: E402
# The one list of project.json project_type values that mean "trains from a
# prepared source, holds no annotation work". Imported rather than copied so a
# third kind of training source cannot be added there and silently start
# feeding the matrix here. training_projects.py imports nothing from _matrix,
# so this direction carries no cycle.
from _reefreview.training_projects import TRAINING_PROJECT_TYPES  # noqa: E402

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

# Mask source_types that count as a hand-made (manual) annotation: the one
# shared list, `merged` included (see _reefreview/manual_sources.py).
_MANUAL_SOURCE_TYPES = MANUAL_SOURCE_TYPES

# project.json project_type of a run dir minted from a prepared training
# source. Two kinds exist and both are skipped here:
#
#   composite_training  minted from a composite training set. Its step4test
#                       dir is a symlink INTO the composite it trains on, so
#                       scanning it would count every mask in that set a
#                       second time and feed the matrix its own output. The
#                       matrix learns about these from composite.json's
#                       linked_projects list instead.
#   bbox_training       minted from a staged detection dataset. That imagery
#                       is not TCRMP survey data and must never reach the mask
#                       registry, the coverage matrix or STATS at all.
#
# TRAINING_PROJECT_TYPES is the one list, owned by
# _reefreview/training_projects.py. Nothing here names a single type: a
# comparison against one would silently stop skipping the other.


def is_training_project(meta):
    """True for a project.json that trains from a prepared source, of either
    kind. Every scan of annotation work skips these."""
    return (meta or {}).get("project_type") in TRAINING_PROJECT_TYPES


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


def _upper_initials(value):
    return str(value or "").strip().upper()


def project_identity(meta):
    """(creator_initials, opened_by) for a project.json document.

    creator_initials: project.json ``initials``, else the initials of the
    first access_log entry whose action is "create".
    opened_by: the initials of the LATEST access_log entry (open or create,
    the launcher writes both), falling back to the creator initials. Both
    uppercased; '' when nothing is recorded. Nothing is inferred into the
    data: this only reads what the launcher already wrote.
    """
    meta = meta or {}
    log = [e for e in (meta.get("access_log") or []) if isinstance(e, dict)]
    creator = _upper_initials(meta.get("initials"))
    if not creator:
        for entry in log:
            if entry.get("action") == "create":
                creator = _upper_initials(entry.get("initials"))
                break
    opened_by = ""
    for entry in reversed(log):
        ini = _upper_initials(entry.get("initials"))
        if ini:
            opened_by = ini
            break
    return creator, opened_by or creator


def scan_projects(inprocess_root):
    """Every live run_* project directly under inprocess_root that carries a
    readable project.json, sorted by directory name. Each entry:
    {project_id, name, run_dir, project_dir, initials, opened_by,
     created_at, target_species, has_model, meta}. Names come from
    project.json at scan time (auto names change as Step 3 runs).

    Training projects (project_type "composite_training" or "bbox_training")
    are NOT returned: they are downstream consumers of a prepared training
    source, never a source of annotation work. A composite project's step4test
    dir is a symlink into the composite itself, and a bbox project trains on
    imagery from outside the TCRMP survey. See is_training_project."""
    out = []
    for project_dir in sorted(glob.glob(os.path.join(inprocess_root or "", "run_*"))):
        if not os.path.isdir(project_dir):
            continue
        meta = _read_project_json(project_dir)
        if meta is None:
            continue
        if is_training_project(meta):
            continue
        dir_name = os.path.basename(project_dir)
        initials, opened_by = project_identity(meta)
        out.append({
            "project_id": meta.get("id") or dir_name,
            "name": meta.get("name") or dir_name,
            "run_dir": dir_name,
            "project_dir": os.path.abspath(project_dir),
            "initials": initials,
            "opened_by": opened_by,
            "created_at": meta.get("created_at") or "",
            "target_species": _target_species(meta),
            "has_model": _has_model(project_dir),
            "meta": meta,
        })
    return out


def _ledger_files(project_dir):
    """Every label_provenance.csv under a project, sorted."""
    return sorted(p for p in glob.glob(
        os.path.join(project_dir, "**", "label_provenance.csv"), recursive=True)
        if not is_superseded_path(p, root=project_dir))


def _ledger_dirs(project_dir):
    """Processing directories that carry their own label_provenance.csv.

    A directory in this set is ledger driven: the legacy derivation skips
    the segmentations.json files under it, because the ledger is the
    authoritative record for the work that ran there. A sibling directory
    with segmentations and no ledger of its own still derives.
    """
    return {os.path.dirname(p) for p in _ledger_files(project_dir)}


def _processing_dir(seg_file):
    """The step directory a segmentations.json belongs to.

    Handles both shapes the pipeline writes:
      <step>/segmentations/segmentations.json
      <step>/segmentations/<batch>/segmentations.json
    Returns '' when the path holds no 'segmentations' directory.
    """
    d = os.path.dirname(seg_file)
    while d and os.path.basename(d) != "segmentations":
        parent = os.path.dirname(d)
        if parent == d:
            return ""
        d = parent
    return os.path.dirname(d) if d else ""


def _ledger_rows(project_dir):
    """Yield provenance rows from every label_provenance.csv under a project.

    Returns a list of dicts: basename, label, outcome, source, reviewer, at.
    Returns None (not []) when NO ledger file exists anywhere, so callers
    can distinguish "no ledger" from "ledger with zero rows".
    """
    files = _ledger_files(project_dir)
    if not files:
        return None
    rows = []
    for fpath in files:
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
                        "uid": "",
                        "status": "",
                        "source_type": "",
                    })
        except OSError:
            continue
    return rows


def _legacy_rows(project_dir, target_species, skip_dirs=None):
    """Derive provenance rows from legacy segmentations.json files.

    Used for every processing directory that has no label_provenance.csv of
    its own (`skip_dirs` names the ones that do). For each reviewed-or-
    exported image we emit one row per accepted (non-review) species
    (found_expert when an accepted expert ID is stamped), one pending_expert
    row per tentative code on each non-rejected review mask, plus a not_found
    row for every target species with no accepted or pending presence.
    """
    flat = glob.glob(os.path.join(project_dir, "**", "segmentations", "segmentations.json"), recursive=True)
    nested = glob.glob(os.path.join(project_dir, "**", "segmentations", "*", "segmentations.json"), recursive=True)
    files = sorted(p for p in set(flat) | set(nested)
                   if not is_superseded_path(p, root=project_dir))
    skip_dirs = set(skip_dirs or ())
    rows = []
    for fpath in files:
        if skip_dirs and _processing_dir(fpath) in skip_dirs:
            continue
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
                rec = by_species.setdefault(
                    code, {"manual": False, "expert": False, "reviewer": "", "source_type": ""})
                if m.get("source_type") in _MANUAL_SOURCE_TYPES:
                    rec["manual"] = True
                if _is_expert_mask(m):
                    rec["expert"] = True
                if not rec["reviewer"]:
                    rec["reviewer"] = str(m.get("reviewer") or "").strip().upper()
                if not rec["source_type"]:
                    rec["source_type"] = str(m.get("source_type") or "")

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
                    "reviewer": rec.get("reviewer", ""),
                    "at": at,
                    "uid": "",
                    "status": "accepted",
                    "source_type": rec.get("source_type", ""),
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
                    "uid": "",
                    "status": "pending",
                    "source_type": "",
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
                    "uid": "",
                    "status": "",
                    "source_type": "",
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
            # The reviewer column arrives with the identity work; a manifest
            # written before it simply has no such key.
            "reviewer": (rec.get("reviewer") or "").strip().upper(),
            "at": (rec.get("updated_at") or rec.get("created_at") or "").strip(),
            "project_id": (rec.get("project_id") or "").strip(),
            "uid": (rec.get("uid") or "").strip(),
            "status": status,
            "source_type": source_type,
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

    # Dirs without a project.json are not projects and are ignored (junk).
    for info in scan_projects(inprocess_root):
        project_dir = info["project_dir"]
        project_id = info["project_id"]
        name = info["name"]
        target_species = info["target_species"]

        # Ledger and legacy are combined per processing directory: a step
        # dir with its own label_provenance.csv is ledger driven, a sibling
        # dir holding only segmentations.json still derives. Before this a
        # single ledger anywhere in the project silenced every segmentations
        # file in it, which hid whole step5 populations from the grid.
        ledger = _ledger_rows(project_dir)
        rows = list(ledger or [])
        rows.extend(_legacy_rows(project_dir, target_species,
                                 skip_dirs=_ledger_dirs(project_dir)))

        # Per-cell merge, never a project-level replace. The registry is the
        # source of truth for the (basename, label) cells it actually holds
        # for THIS project; every other cell keeps its ledger/legacy row, so
        # a project with five registered masks and a thousand annotated ones
        # still reports all thousand. not_found keeps coming from the
        # ledger/legacy chain only (target_species x reviewed image set).
        reg_rows = registry_by_project.get(project_id) or []
        reg_cells = {(r["basename"], r["label"]) for r in reg_rows}
        base_cells = {(r["basename"], r["label"]) for r in rows
                      if r["outcome"] != "not_found"}
        if reg_rows:
            rows = list(reg_rows) + [
                r for r in rows
                if r["outcome"] == "not_found"
                or (r["basename"], r["label"]) not in reg_cells
            ]

        # partially_registered: the registry covers some but not all of the
        # project's found/pending cells, so the grid mixes registry rows with
        # derived ones. A project with no registry rows at all is not
        # partially registered: its whole contribution is one consistent
        # chain. Task 5 renders this as a badge in the project isolate list.
        n_cells = len(base_cells | reg_cells)
        projects.append({
            "project_id": project_id,
            "name": name,
            "target_species": target_species,
            "has_model": info["has_model"],
            "initials": info["initials"],
            "opened_by": info["opened_by"],
            "run_dir": info["run_dir"],
            "project_dir": project_dir,
            "created_at": info["created_at"],
            "partially_registered": bool(reg_cells) and len(reg_cells) < n_cells,
            "n_registry_cells": len(reg_cells),
            "n_cells": n_cells,
        })

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
                "reviewer": row.get("reviewer", ""),
                "at": row["at"],
                "outcome": outcome,
                "uid": row.get("uid", ""),
                "status": row.get("status", ""),
                "source_type": row.get("source_type", ""),
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

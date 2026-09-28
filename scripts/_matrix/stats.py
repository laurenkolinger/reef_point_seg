"""Overall annotation statistics for the matrix STATS tab (spec D6, D7).

Counted rows: registry rows whose project_id matches a live run_* project
(project.json id), merged per mask with that project's segmentations.json
masks (uid minted with mask_uid). The registry wins the masks it holds; a
mask the registry never recorded still counts, so a project with five
registered masks and a thousand annotated ones reports all thousand. This
mirrors the grid's per-cell merge at the granularity STATS counts in.
Rows from unknown project ids are counted and reported in `excluded`,
never silently dropped; so are live-project rows with a blank species
(`excluded.unlabeled_rows`, per project in `excluded.unlabeled_by_project`).
Status filter: 'accepted' (accepted, non-review masks, the default) or
'all'.

Registry hygiene: `purge_test_rows` removes rows whose project_id matches
the test-fixture patterns and has no run_* dir, after backing up
manifest.csv.
"""

import os
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone

_PKG_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_SCRIPTS not in sys.path:
    sys.path.insert(0, _PKG_SCRIPTS)

from _reefreview import filename_parse  # noqa: E402
from _reefreview.mask_registry import MaskRegistry  # noqa: E402
from _reefreview.uid import mask_uid  # noqa: E402

from .builder import _ast_now_iso, _is_expert_mask, _is_review_mask, scan_projects  # noqa: E402
from .cellinfo import load_project_segmentations  # noqa: E402

TEST_RESIDUE_PATTERNS = [
    re.compile(r"^tmp$"),
    re.compile(r"^exp_"),
    re.compile(r"^e2e_"),
    re.compile(r"^combined_smoke_"),
    re.compile(r"^label_guard_"),
    re.compile(r"^expert_lock_test_"),
]
UNRECORDED = "unrecorded"
_AST = timezone(timedelta(hours=-4))


def is_test_residue_id(project_id):
    pid = str(project_id or "")
    return any(p.search(pid) for p in TEST_RESIDUE_PATTERNS)


def registry_dir(inprocess_root):
    return os.path.join(inprocess_root or "", "_mask_registry")


def _int_or_none(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _record_from_registry(rec):
    basename = (rec.get("basename") or "").strip() or os.path.splitext(
        (rec.get("source_image") or "").strip())[0]
    parsed = filename_parse.parse(rec.get("source_image") or basename)
    site = (rec.get("site") or "").strip().upper() or (parsed.get("site") or "")
    year = _int_or_none(rec.get("year"))
    if year is None:
        year = parsed.get("year")
    return {
        "uid": (rec.get("uid") or "").strip(),
        "basename": basename,
        "species": (rec.get("species") or "").strip(),
        "status": (rec.get("status") or "").strip(),
        "review": rec.get("review") == "1",
        "expert": rec.get("expert_mode") == "1",
        "source_type": (rec.get("source_type") or "").strip(),
        "reviewer": (rec.get("reviewer") or "").strip().upper(),
        "project_id": (rec.get("project_id") or "").strip(),
        "site": site,
        "year": year,
        "origin": "registry",
    }


def _records_from_segmentations(info):
    out = []
    for stem, entry in load_project_segmentations(info["project_dir"]).items():
        seg = entry["seg"]
        filename = entry["filename"]
        parsed = filename_parse.parse(filename)
        for m in (seg.get("masks") or []):
            if not isinstance(m, dict):
                continue
            out.append({
                "uid": mask_uid(filename, m, parsed),
                "basename": stem,
                "species": (m.get("species") or "").strip(),
                "status": (m.get("status") or "").strip(),
                "review": _is_review_mask(m),
                "expert": _is_expert_mask(m),
                "source_type": (m.get("source_type") or "").strip(),
                "reviewer": str(m.get("reviewer") or "").strip().upper(),
                "project_id": info["project_id"],
                "site": parsed.get("site") or "",
                "year": parsed.get("year"),
                "origin": "legacy",
            })
    return out


def collect_records(inprocess_root):
    """(records, excluded) before any status filter.

    Annotation sources only: scan_projects already drops every training
    project (builder.TRAINING_PROJECT_TYPES, so both "composite_training" and
    "bbox_training"). A composite project's step4test dir is a symlink into
    the training set it trains on, and counting one would report every mask in
    that set twice; a bbox project trains on imagery from outside the TCRMP
    survey, which must never be counted here at all. A registry row that
    somehow carries such a project's id is not silently dropped either: its id
    is not live, so the row surfaces under excluded.unknown_project_rows."""
    projects = scan_projects(inprocess_root)
    live_ids = {p["project_id"] for p in projects}
    reg = MaskRegistry(registry_dir(inprocess_root))
    try:
        reg_rows = reg.rows()
    except OSError:
        reg_rows = []

    records = []
    unknown_rows = 0
    unknown_ids = set()
    residue_rows = 0
    registered_uids = {}
    for rec in reg_rows:
        pid = (rec.get("project_id") or "").strip()
        if pid not in live_ids:
            unknown_rows += 1
            unknown_ids.add(pid)
            if is_test_residue_id(pid):
                residue_rows += 1
            continue
        uid = (rec.get("uid") or "").strip()
        if uid:
            registered_uids.setdefault(pid, set()).add(uid)
        records.append(_record_from_registry(rec))

    legacy_projects = []
    for info in projects:
        # Per-mask merge: a mask the registry already holds is counted once,
        # from the registry. Every other mask on disk still counts, instead
        # of the whole project's segmentations being dropped the moment one
        # of its masks reached the registry.
        covered = registered_uids.get(info["project_id"]) or set()
        derived = [r for r in _records_from_segmentations(info)
                   if r["uid"] not in covered]
        if derived:
            legacy_projects.append(info["project_id"])
            records.extend(derived)

    excluded = {
        "unknown_project_rows": unknown_rows,
        "unknown_project_ids": sorted(unknown_ids),
        "test_residue_rows": residue_rows,
    }
    return records, excluded, legacy_projects


def _passes(rec, status):
    if status == "all":
        return True
    return rec["status"] == "accepted" and not rec["review"]


def compute_stats(inprocess_root, status="accepted"):
    status = "all" if str(status or "").lower() == "all" else "accepted"
    records, excluded, legacy_projects = collect_records(inprocess_root)
    counted = [r for r in records if _passes(r, status) and r["species"]]
    # A live-project mask with no species code cannot be counted under any
    # label, but it is never dropped silently: report it next to the
    # unknown-project rows (the composite side counts the same masks as
    # blocked_unlabeled).
    unlabeled = [r for r in records if _passes(r, status) and not r["species"]]
    unlabeled_by_project = {}
    for r in unlabeled:
        unlabeled_by_project[r["project_id"]] = unlabeled_by_project.get(r["project_id"], 0) + 1
    excluded = dict(excluded)
    excluded["unlabeled_rows"] = len(unlabeled)
    excluded["unlabeled_by_project"] = dict(sorted(unlabeled_by_project.items()))

    species = {}
    by_year = {}
    by_site = {}
    images_by_year = {}
    images_by_site = {}
    reviewers = {}
    images = set()
    projects = set()
    unknown_year = unknown_site = 0
    years = set()
    sites = set()

    for r in counted:
        code = r["species"]
        images.add(r["basename"])
        projects.add(r["project_id"])
        sp = species.setdefault(code, {"code": code, "n_masks": 0, "images": set(), "projects": set()})
        sp["n_masks"] += 1
        sp["images"].add(r["basename"])
        sp["projects"].add(r["project_id"])

        if r["year"] is None:
            unknown_year += 1
            ykey = "unknown"
        else:
            years.add(int(r["year"]))
            ykey = str(int(r["year"]))
        by_year.setdefault(code, {})[ykey] = by_year.setdefault(code, {}).get(ykey, 0) + 1
        images_by_year.setdefault(ykey, set()).add(r["basename"])

        if not r["site"]:
            unknown_site += 1
            skey = "unknown"
        else:
            sites.add(r["site"])
            skey = r["site"]
        by_site.setdefault(code, {})[skey] = by_site.setdefault(code, {}).get(skey, 0) + 1
        images_by_site.setdefault(skey, set()).add(r["basename"])

        ini = r["reviewer"] or UNRECORDED
        reviewers[ini] = reviewers.get(ini, 0) + 1

    species_out = sorted(
        ({"code": s["code"], "n_masks": s["n_masks"], "n_images": len(s["images"]),
          "n_projects": len(s["projects"])} for s in species.values()),
        key=lambda s: (-s["n_masks"], s["code"]),
    )
    reviewers_out = sorted(
        ({"initials": k, "n_masks": v} for k, v in reviewers.items()),
        key=lambda r: (-r["n_masks"], r["initials"]),
    )
    return {
        "generated_at": _ast_now_iso(),
        "status": status,
        "totals": {
            "masks": len(counted),
            "images": len(images),
            "projects": len(projects),
            "reviewers": len([k for k in reviewers if k != UNRECORDED]),
            "labels": len(species),
        },
        "species": species_out,
        "years": sorted(years),
        "sites": sorted(sites),
        "by_year": by_year,
        "by_site": by_site,
        "images_by_year": {k: len(v) for k, v in images_by_year.items()},
        "images_by_site": {k: len(v) for k, v in images_by_site.items()},
        "reviewers": reviewers_out,
        "unknown_year_masks": unknown_year,
        "unknown_site_masks": unknown_site,
        "legacy_projects": legacy_projects,
        "excluded": excluded,
    }


def purge_test_rows(inprocess_root):
    """(removed, backup_path). Delegates to MaskRegistry.purge_test_rows."""
    known = {p["project_id"] for p in scan_projects(inprocess_root)}
    reg = MaskRegistry(registry_dir(inprocess_root))
    # Single implementation: MaskRegistry.purge_test_rows counts under the
    # lock, writes a uniquely named backup only when rows go, then drops them.
    removed, backup = reg.purge_test_rows(known)
    return int(removed or 0), backup or ""

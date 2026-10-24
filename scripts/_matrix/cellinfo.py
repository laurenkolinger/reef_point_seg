"""Per-cell drill-down for the matrix.

Given an (image, label) cell this module answers: which live projects hold
that image in a `segmentations.json` (flat `segmentations/segmentations.json`
or nested `segmentations/<batch>/segmentations.json`), every mask each of
those projects holds for the image (uid minted with `_reefreview.uid.mask_uid`
so it matches the registry), the project's outcome for that label, and the
absolute image file the `/img/<project_id>/<basename>` route serves.

The project list itself rescans on every call, like the builder: no cache, so
a purged project simply stops answering. What IS cached is the parse of each
individual `segmentations.json` (see `_load_json` below) since re-parsing the
same tens-of-MB file on every drawer open dominated latency; the cache keys
off the file's (mtime, size) so an edited file is always re-read.
"""

import collections
import csv
import glob
import json
import os
import sys
import threading

_PKG_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_SCRIPTS not in sys.path:
    sys.path.insert(0, _PKG_SCRIPTS)

from _reefreview.uid import mask_uid  # noqa: E402

from .builder import (  # noqa: E402
    _basename_stem,
    _is_expert_mask,
    _is_review_mask,
    _pending_labels,
    _stronger,
    build_matrix,
    scan_projects,
)
from .preferences import Preferences  # noqa: E402
from _superseded import is_superseded_path  # noqa: E402


def segmentation_files(project_dir):
    """Every segmentations.json under a project: flat and per-batch nested."""
    flat = glob.glob(os.path.join(project_dir, "**", "segmentations", "segmentations.json"),
                     recursive=True)
    nested = glob.glob(os.path.join(project_dir, "**", "segmentations", "*", "segmentations.json"),
                       recursive=True)
    return sorted(p for p in set(flat) | set(nested)
                  if not is_superseded_path(p, root=project_dir))


# mtime+size keyed parse cache for segmentations.json files. Bounded LRU
# (module-level OrderedDict) so a season of projects cannot grow it without
# limit; guarded by a lock since Flask serves this threaded. Keyed by the
# absolute file path so composite.py and stats.py, which import
# `load_project_segmentations` from here, share the same cache for free.
_SEG_CACHE = collections.OrderedDict()  # abspath -> (mtime, size, doc)
_SEG_CACHE_LOCK = threading.Lock()
_SEG_CACHE_MAX = 64


def _parse_json_file(path):
    """Uncached parse of one segmentations.json. Split out from `_load_json`
    so tests can monkeypatch this to count real re-parses."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _load_json(path):
    """`_parse_json_file(path)`, cached by (mtime, size) of `path`. A file
    whose mtime and size both match the cached entry is assumed unchanged
    and the parsed result is reused; anything else (edited, replaced,
    removed) is re-parsed and the cache entry refreshed."""
    abspath = os.path.abspath(path)
    try:
        st = os.stat(abspath)
    except OSError:
        with _SEG_CACHE_LOCK:
            _SEG_CACHE.pop(abspath, None)
        return None
    stat_key = (st.st_mtime, st.st_size)

    with _SEG_CACHE_LOCK:
        cached = _SEG_CACHE.get(abspath)
        if cached is not None and (cached[0], cached[1]) == stat_key:
            _SEG_CACHE.move_to_end(abspath)
            return cached[2]

    doc = _parse_json_file(abspath)

    with _SEG_CACHE_LOCK:
        _SEG_CACHE[abspath] = (stat_key[0], stat_key[1], doc)
        _SEG_CACHE.move_to_end(abspath)
        while len(_SEG_CACHE) > _SEG_CACHE_MAX:
            _SEG_CACHE.popitem(last=False)

    return doc


# Composite training sets live under <inprocess_root>/_composite/<name>/ and
# record one composite_manifest.csv row per exported mask (basename, label,
# project_id, ...). The drawer shows which sets a cell already trains in, so
# the manifests are indexed the same way segmentations.json files are: an
# mtime+size keyed parse, bounded, guarded by its own lock. The dirname and
# the two column names mirror composite.py (COMPOSITE_DIRNAME,
# MANIFEST_FIELDS); they are repeated here rather than imported because
# composite.py imports THIS module.
_COMPOSITE_DIRNAME = "_composite"
_MANIFEST_NAME = "composite_manifest.csv"
_MANIFEST_CACHE = collections.OrderedDict()  # abspath -> (mtime, size, cells set)
_MANIFEST_CACHE_LOCK = threading.Lock()
_MANIFEST_CACHE_MAX = 32


def _parse_manifest_cells(path):
    """Uncached read of one composite_manifest.csv into a set of
    (basename, label) pairs. Split out so tests can count real re-reads."""
    cells = set()
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                basename = _basename_stem((row.get("basename") or "").strip())
                label = (row.get("label") or "").strip()
                if basename and label:
                    cells.add((basename, label))
    except (OSError, ValueError):
        return set()
    return cells


def _manifest_cells(path):
    """`_parse_manifest_cells(path)`, cached by (mtime, size) of `path`, so a
    drawer open never re-reads a manifest that has not changed. A rebuilt
    composite writes a new file, which changes both, so the index refreshes."""
    abspath = os.path.abspath(path)
    try:
        st = os.stat(abspath)
    except OSError:
        with _MANIFEST_CACHE_LOCK:
            _MANIFEST_CACHE.pop(abspath, None)
        return set()
    stat_key = (st.st_mtime, st.st_size)

    with _MANIFEST_CACHE_LOCK:
        cached = _MANIFEST_CACHE.get(abspath)
        if cached is not None and (cached[0], cached[1]) == stat_key:
            _MANIFEST_CACHE.move_to_end(abspath)
            return cached[2]

    cells = _parse_manifest_cells(abspath)

    with _MANIFEST_CACHE_LOCK:
        _MANIFEST_CACHE[abspath] = (stat_key[0], stat_key[1], cells)
        _MANIFEST_CACHE.move_to_end(abspath)
        while len(_MANIFEST_CACHE) > _MANIFEST_CACHE_MAX:
            _MANIFEST_CACHE.popitem(last=False)
    return cells


def composites_for_cell(inprocess_root, stem, label):
    """Sorted names of the composite training sets whose manifest holds a row
    for this (basename, label). A composite with no manifest (a partial or
    hand-made directory) contributes nothing rather than raising."""
    root = os.path.join(inprocess_root or "", _COMPOSITE_DIRNAME)
    if not stem or not label or not os.path.isdir(root):
        return []
    out = []
    for path in sorted(glob.glob(os.path.join(root, "*", _MANIFEST_NAME))):
        if (stem, label) in _manifest_cells(path):
            out.append(os.path.basename(os.path.dirname(path)))
    return sorted(out)


def _is_reviewed(seg):
    return seg.get("reviewed") is True or seg.get("exported") is True


def load_project_segmentations(project_dir):
    """{stem: {"filename", "seg", "seg_file"}} across every segmentations.json
    of the project. When the same stem appears in more than one file the
    reviewed/exported entry wins; otherwise the later file (sorted) wins."""
    out = {}
    for seg_file in segmentation_files(project_dir):
        doc = _load_json(seg_file)
        if not doc:
            continue
        for filename, seg in doc.items():
            if not isinstance(seg, dict):
                continue
            stem = _basename_stem(seg.get("image_path") or filename)
            entry = {"filename": os.path.basename(filename), "seg": seg, "seg_file": seg_file}
            prev = out.get(stem)
            if prev is None or _is_reviewed(seg) or not _is_reviewed(prev["seg"]):
                out[stem] = entry
    return out


def _mask_at(mask, seg):
    outcomes = seg.get("label_outcomes") or {}
    species = mask.get("species") or ""
    lo = outcomes.get(species) if isinstance(outcomes, dict) else None
    lo_at = lo.get("at") if isinstance(lo, dict) else ""
    return (mask.get("reviewed_at") or mask.get("updated_at") or lo_at
            or seg.get("processed_at") or "")


def mask_record(filename, mask, seg, label=None):
    """The drawer's per-mask record (spec 4.1). `is_label` is true for a mask
    of the cell's label, including a pending review mask whose tentative
    codes carry that label."""
    species = mask.get("species") or ""
    review = _is_review_mask(mask)
    is_label = bool(label) and (species == label or (review and label in _pending_labels(mask)))
    return {
        "uid": mask_uid(filename, mask),
        "mask_id": mask.get("id"),
        "species": species,
        "status": mask.get("status") or "",
        "source_type": mask.get("source_type") or "",
        "area": mask.get("area"),
        "bbox": mask.get("bbox"),
        "polygon_px": mask.get("polygon_px") or [],
        "reviewer": str(mask.get("reviewer") or "").strip().upper(),
        "at": _mask_at(mask, seg),
        "is_label": is_label,
        "review": review,
        "expert": _is_expert_mask(mask),
    }


def _find_project(inprocess_root, project_id):
    pid = str(project_id or "")
    for info in scan_projects(inprocess_root):
        if info["project_id"] == pid or info["run_dir"] == pid:
            return info
    return None


def image_path_for(inprocess_root, project_id, basename):
    """Absolute image path recorded in that project's segmentations.json for
    the basename (stem or filename), or None when the project or the image
    is unknown. Never derived from the query string."""
    info = _find_project(inprocess_root, project_id)
    if info is None:
        return None
    entry = load_project_segmentations(info["project_dir"]).get(_basename_stem(basename))
    if entry is None:
        return None
    path = entry["seg"].get("image_path_abs") or ""
    return path or None


def cell_info(inprocess_root, image, label, matrix=None, image_url_base="/matrix/img"):
    """The drawer payload for one cell, or None when no project holds the
    image and the matrix has no cell for it (the route answers 404)."""
    stem = _basename_stem(image)
    label = str(label or "").strip()
    matrix = matrix or build_matrix(inprocess_root)
    cell = (matrix.get("cells") or {}).get(stem, {}).get(label)
    sources = list(cell["sources"]) if cell else []

    projects = []
    for info in scan_projects(inprocess_root):
        entry = load_project_segmentations(info["project_dir"]).get(stem)
        if entry is None:
            continue
        seg = entry["seg"]
        outcome = None
        for s in sources:
            if s.get("project_id") == info["project_id"]:
                outcome = _stronger(outcome, s.get("outcome"))
        masks = [mask_record(entry["filename"], m, seg, label)
                 for m in (seg.get("masks") or []) if isinstance(m, dict)]
        projects.append({
            "project_id": info["project_id"],
            "name": info["name"],
            "run_dir": info["run_dir"],
            "project_dir": info["project_dir"],
            "opened_by": info["opened_by"],
            "initials": info["initials"],
            "outcome_for_label": outcome,
            "image_url": f"{image_url_base.rstrip('/')}/{info['project_id']}/{stem}",
            "filename": entry["filename"],
            "image_width": seg.get("image_width"),
            "image_height": seg.get("image_height"),
            "reviewed": _is_reviewed(seg),
            "masks": masks,
        })

    if cell is None and not projects:
        return None

    pref = Preferences(inprocess_root).get(stem, label)
    if pref is not None:
        # A saved choice whose project no longer holds the image (purged or
        # renamed) is stale: the drawer says so and the composite falls back
        # to the strongest project.
        pref = dict(pref)
        pref["stale"] = not any(p["project_id"] == pref["project_id"] for p in projects)
    return {
        "image": stem,
        "label": label,
        "outcome": cell["outcome"] if cell else None,
        "sources": sources,
        "preference": pref,
        "projects": projects,
        # Training sets this cell already exports into. Empty when no
        # composite holds it, which is the common case.
        "composites": composites_for_cell(inprocess_root, stem, label),
    }

"""Composite YOLO export assembled across projects (spec D4).

    <inprocess_root>/_composite/<name>/
        all_images/               symlinks to the source images
        all_labels/<stem>.txt     YOLO-seg lines (empty file = negative)
        data.yaml                 YOLO dataset config
        class_map.json            label -> class id (sorted label order)
        composite_manifest.csv    one row per exported mask
        composite.json            listing metadata (name, counts, labels)
        README.md                 what was assembled and how

Resolution per (image, label): the preference row if present, else the
project whose outcome is strongest under the matrix ladder
(found_expert > found_manual > found_ai > found_model), ties broken by the
latest `at`. Only accepted, non-review, labeled masks export (the same
filter as export_yolo.export_batch). An image with no exported mask is a
negative only when some project reviewed it and no project holds a pending
REVIEW mask for it; otherwise it is skipped.

A composite can then be picked up as a regular pipeline project that starts
at Train: create_training_project() mints a run_* dir whose
step4test_combinedAnnotate is a relative SYMLINK to the composite, so the
orchestrator reads the same all_images/ + all_labels/ + data.yaml the Train
step already expects and no image is ever copied. The link runs one way:
composite.json remembers every training project it fed (linked_projects),
and those run dirs are skipped by every scan that gathers annotation work,
so a training set can never feed the matrix back into itself.
"""

import csv
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

try:
    import fcntl  # POSIX advisory file locking (this box is Linux)
    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover
    _HAVE_FCNTL = False

_PKG_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_SCRIPTS not in sys.path:
    sys.path.insert(0, _PKG_SCRIPTS)

from _reefreview.clock import now_ast  # noqa: E402
from _reefreview.eml_writer import write_eml  # noqa: E402
# The single place a training project's run dir is written, for either kind of
# source. RunDirExhausted is imported, not redefined: one class, so
# _matrix/blueprint.py's 409 mapping catches a mint raised from either module.
from _reefreview.training_projects import (  # noqa: E402
    RunDirExhausted,
    mint_training_project,
)

log = logging.getLogger(__name__)

from .builder import (  # noqa: E402
    _MANUAL_SOURCE_TYPES,
    _OUTCOME_STRENGTH,
    _is_expert_mask,
    _is_review_mask,
    build_matrix,
    scan_projects,
)
from .cellinfo import load_project_segmentations, mask_record  # noqa: E402
from .preferences import Preferences  # noqa: E402

COMPOSITE_DIRNAME = "_composite"
MANIFEST_FIELDS = ["basename", "label", "project_id", "uid", "status", "source_type",
                   "reviewer", "chosen_by_preference", "at"]
_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_AST = timezone(timedelta(hours=-4))

# Mask source_type values the composite's source_types filter accepts (spec
# Task 6). The route validates a request's list against this same set. The
# manual ones come from the shared list (`merged` included), so the filter
# can select exactly the masks the grid counts as manual.
SOURCE_TYPES = set(_MANUAL_SOURCE_TYPES) | {"auto", "exemplar", "model", "edit"}
# A mask predating source_type tracking, or one written by a path that never
# set it, carries '' / missing. It is a hand-placed click in every such case
# on record, so the filter treats it as manual_click (composite.json records
# this rule explicitly so a reader never has to guess).
_DEFAULT_SOURCE_TYPE = "manual_click"
# TCRMPyyyymmdd_clip_SITE_Tnnn -> SITE; anything else buckets as 'unknown'.
_SITE_RE = re.compile(r"^TCRMP\d{8}_clip_([A-Za-z0-9]+)_T\d+", re.IGNORECASE)
# Sibling dirs used for the rebuild swap (build into _tmp, then move the old
# out_dir to _backup so the rename target is free, then move _tmp into place).
_REBUILD_TMP_SUFFIX = ".rebuild-tmp"
_REBUILD_BACKUP_SUFFIX = ".rebuild-backup"

# --- training projects minted from a composite -----------------------------
# A run dir minted here trains from a composite instead of holding annotation
# work of its own, and builder.scan_projects skips it so it never contributes
# cells, stats records or composite candidates. The project_type marker
# (COMPOSITE_PROJECT_TYPE) and the step4test symlink name both live in
# _reefreview/training_projects.py, which writes the run dir; nothing is
# restated here, so the two cannot drift.
#
# Initials accepted for the `by` of a training project (same shape the
# matrix and the launcher use for reviewer initials). The shared minter
# enforces the same rule; the check runs here first so a bad `by` is refused
# before the composite name is even resolved, which is the order the route's
# error messages were written against.
_INITIALS_RE = re.compile(r"^[A-Za-z]{1,4}$")

# join_polygons comes from the annotator's exporter so a multi-part mask is
# bridged into one YOLO line exactly as the per-project export does. The
# module is loaded by file path under a private name: its import is
# side-effect free, and loading it this way keeps the three same-named
# export_yolo.py copies (combinedAnnotate / editMasks / segmentImages) from
# colliding in sys.modules.
_EXPORT_YOLO_PATH = os.path.join(_PKG_SCRIPTS, "TCRMPclip_combinedAnnotate", "src", "export_yolo.py")


def _load_join_polygons():
    try:
        spec = importlib.util.spec_from_file_location("_matrix_export_yolo", _EXPORT_YOLO_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.join_polygons
    except Exception:  # pragma: no cover - only when the annotator tree is missing
        return _join_polygons_fallback


def _join_polygons_fallback(polygons):
    """Copy of TCRMPclip_combinedAnnotate/src/export_yolo.py::join_polygons,
    used only when that file cannot be loaded."""
    polys = [list(p) for p in polygons if p and len(p) >= 6]
    if not polys:
        return []
    if len(polys) == 1:
        return polys[0]
    polys.sort(key=len, reverse=True)
    merged = polys[0]
    for poly in polys[1:]:
        mx, my = merged[0::2], merged[1::2]
        px, py = poly[0::2], poly[1::2]
        best = None
        for i in range(len(mx)):
            for j in range(len(px)):
                d = (mx[i] - px[j]) ** 2 + (my[i] - py[j]) ** 2
                if best is None or d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        rotated = poly[2 * j:] + poly[:2 * j]
        bridge = rotated + rotated[:2]
        merged = merged[:2 * i + 2] + bridge + merged[2 * i:]
    return merged


join_polygons = _load_join_polygons()


class CompositeExists(Exception):
    """The composite directory already exists (route answers 409)."""


def composite_root(inprocess_root):
    return os.path.join(inprocess_root or "", COMPOSITE_DIRNAME)


# Lock file for every read-modify-write of a composite.json, and for the
# rebuild swap. It lives in the composite ROOT, not inside a composite dir, so
# a rebuild (which replaces the whole directory) cannot pull the lock file out
# from under a waiting writer. Same flock pattern as
# _reefreview.mask_registry.MaskRegistry._locked.
_LOCK_FILENAME = ".composite.lock"


@contextmanager
def composite_lock(inprocess_root):
    """Serialize composite.json writers across processes and threads.

    Without it, two concurrent create-project calls read the same
    linked_projects list and the second write silently drops the first entry,
    and a reader can observe a composite.json mid-truncation. Held only around
    short metadata writes, never around a build."""
    root = composite_root(inprocess_root)
    os.makedirs(root, exist_ok=True)
    if not _HAVE_FCNTL:  # pragma: no cover - POSIX only box
        yield
        return
    fh = open(os.path.join(root, _LOCK_FILENAME), "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def _write_json_atomic(path, doc):
    """Write JSON so a concurrent reader sees either the old file or the whole
    new one, never a truncated one: a temp file in the SAME directory (so the
    rename cannot cross a filesystem) then os.replace."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".composite-", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def default_name():
    return "composite_" + datetime.now(_AST).strftime("%y-%m-%d_%H%M")


def sanitize_name(name):
    """Keep [A-Za-z0-9._-]; blank or dot-only names fall back to the default."""
    cleaned = _NAME_RE.sub("_", str(name or "").strip()).strip("._")
    return cleaned or default_name()


def _at_key(value):
    """Naive datetime for any `at` shape a source carries: registry
    'YYYY-MM-DD HH:MM:SS', ledger ISO with an offset, legacy naive ISO.
    Blank or unparsable -> datetime.min, so a real stamp always ranks above
    a missing one. Offsets are dropped (every stamp is AST wall clock)."""
    text = str(value or "").strip()
    if not text:
        return datetime.min
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except ValueError:
        return datetime.min


def _fmt_ast(value):
    """'YYYY-MM-DD HH:MM:SS' AST for any stamp shape; an aware stamp is
    converted to AST first, a naive one is taken as AST already. Blank or
    unparsable -> ''."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return ""
    if dt.tzinfo is not None:
        dt = dt.astimezone(_AST)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _norm_polygons(mask, width, height):
    """Normalized rings for a mask: polygon_norm when present, else
    polygon_px scaled by the image size."""
    rings = mask.get("polygon_norm")
    if rings:
        return [list(r) for r in rings if r]
    px = mask.get("polygon_px") or []
    if not px or not width or not height:
        return []
    out = []
    for ring in px:
        flat = []
        for i in range(0, len(ring) - 1, 2):
            flat.append(float(ring[i]) / float(width))
            flat.append(float(ring[i + 1]) / float(height))
        if flat:
            out.append(flat)
    return out


def _mask_source_type(m):
    """The source_type used for the source_types filter: the mask's own
    value, or _DEFAULT_SOURCE_TYPE when missing/empty."""
    return str((m or {}).get("source_type") or "").strip() or _DEFAULT_SOURCE_TYPE


def _filter_by_source_types(masks, source_types):
    """masks whose _mask_source_type is in source_types; every mask is kept
    when source_types is None (no filter). An explicit empty list matches
    nothing, same as any other empty allow-set."""
    if source_types is None:
        return list(masks)
    allowed = set(source_types)
    return [m for m in masks if _mask_source_type(m) in allowed]


def _site_for_stem(stem):
    """Site code parsed from the TCRMPyyyymmdd_clip_SITE_Tnnn basename
    pattern; 'unknown' for anything that does not match."""
    m = _SITE_RE.match(stem or "")
    return m.group(1) if m else "unknown"


def _read_composite_json(out_dir):
    """The composite.json of an existing composite dir, or None when it is
    absent or unreadable (a partial/corrupt directory still lets a rebuild
    proceed, just without history to carry forward)."""
    path = os.path.join(out_dir, "composite.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _yolo_line(polygon_norm, class_id):
    coords = " ".join(f"{v:.6f}" for v in polygon_norm)
    return f"{class_id} {coords}"


def _derived_outcome(masks):
    if any(_is_expert_mask(m) for m in masks):
        return "found_expert"
    if any(m.get("source_type") in _MANUAL_SOURCE_TYPES for m in masks):
        return "found_manual"
    return "found_ai"


def collect(inprocess_root, matrix=None, source_types=None, project_ids=None):
    """Scan every live project's segmentations into per-image candidates.

    source_types, when not None, keeps only accepted masks whose
    _mask_source_type is in that set (candidate-mask-level filter, spec
    Task 6): a project with zero matching masks for an (image, label) simply
    contributes no candidate for it, and an accepted-but-unlabeled mask that
    the filter excludes does not count toward blocked_unlabeled either,
    since it was never going to export regardless of its missing label.

    project_ids, when not None, is a project-level filter: only the named
    projects contribute candidates at all, so an image no listed project
    holds is absent from the result entirely (rather than exported as a
    negative). An explicit empty list matches no project.

    Returns {stem: {"reviewed", "pending_review", "blocked_unlabeled",
    "paths": [(project_id, filename, image_path_abs, width, height)],
    "by_label": {label: [candidate]}}} where a candidate is
    {project_id, name, filename, image_path, width, height, masks (raw),
     outcome, strength, at}.
    """
    matrix = matrix or build_matrix(inprocess_root)
    cells = matrix.get("cells") or {}
    allowed = None if project_ids is None else set(project_ids)
    images = {}
    for info in scan_projects(inprocess_root):
        pid = info["project_id"]
        if allowed is not None and pid not in allowed:
            continue
        for stem, entry in load_project_segmentations(info["project_dir"]).items():
            seg = entry["seg"]
            if seg.get("scrapped"):
                continue
            img = images.setdefault(stem, {
                "reviewed": False, "pending_review": False, "blocked_unlabeled": 0,
                "paths": [], "by_label": {},
            })
            if seg.get("reviewed") is True or seg.get("exported") is True:
                img["reviewed"] = True
            width = seg.get("image_width")
            height = seg.get("image_height")
            img["paths"].append((pid, entry["filename"], seg.get("image_path_abs") or "", width, height))
            masks = [m for m in (seg.get("masks") or []) if isinstance(m, dict)]
            if any(_is_review_mask(m) and m.get("status") != "rejected" for m in masks):
                img["pending_review"] = True
            accepted = [m for m in masks if m.get("status") == "accepted" and not _is_review_mask(m)]
            accepted = _filter_by_source_types(accepted, source_types)
            img["blocked_unlabeled"] += sum(1 for m in accepted if not (m.get("species") or ""))
            grouped = {}
            for m in accepted:
                label = m.get("species") or ""
                if label:
                    grouped.setdefault(label, []).append(m)
            for label, ms in grouped.items():
                outcome = None
                at = ""
                for s in (cells.get(stem, {}).get(label, {}).get("sources") or []):
                    if s.get("project_id") != pid:
                        continue
                    o = s.get("outcome")
                    if _OUTCOME_STRENGTH.get(o, -1) > _OUTCOME_STRENGTH.get(outcome, -1):
                        outcome = o
                    at = max([at, s.get("at") or ""], key=_at_key)
                if outcome is None or not outcome.startswith("found"):
                    outcome = _derived_outcome(ms)
                if not at:
                    at = max([mask_record(entry["filename"], m, seg)["at"] for m in ms] or [""], key=_at_key)
                img["by_label"].setdefault(label, []).append({
                    "project_id": pid,
                    "name": info["name"],
                    "filename": entry["filename"],
                    "image_path": seg.get("image_path_abs") or "",
                    "width": width,
                    "height": height,
                    "masks": ms,
                    "outcome": outcome,
                    "strength": _OUTCOME_STRENGTH.get(outcome, 0),
                    "at": at,
                })
    return images


def resolve(inprocess_root, labels=None, matrix=None, images=None, source_types=None,
            project_ids=None):
    """Decide, per (image, label), which project's masks export.

    source_types and project_ids are forwarded to collect() when images is
    not supplied directly (a caller passing images= already applied its own
    filtering). A preference row naming a project the filter excluded can no
    longer be honored: it finds no candidate, counts once under
    preference_unresolved, and the cell falls back to the strongest mask
    among the projects that ARE in the filter.

    Returns (decisions, images, counters). Each decision:
    {stem, label, candidate, masks, by_preference, preference}.
    """
    images = images if images is not None else collect(inprocess_root, matrix, source_types,
                                                       project_ids)
    wanted = set(labels) if labels else None
    prefs = Preferences(inprocess_root).index()
    decisions = []
    counters = {"n_preferred": 0, "preference_unresolved": 0}
    for stem in sorted(images):
        for label in sorted(images[stem]["by_label"]):
            if wanted is not None and label not in wanted:
                continue
            cands = images[stem]["by_label"][label]
            pref = prefs.get((stem, label))
            chosen = None
            masks = None
            by_pref = False
            if pref:
                cand = next((c for c in cands if c["project_id"] == pref["project_id"]), None)
                if cand:
                    ms = cand["masks"]
                    if pref["uids"]:
                        keep = set(pref["uids"])
                        ms = [m for m in ms
                              if mask_record(cand["filename"], m, {})["uid"] in keep]
                    if ms:
                        chosen, masks, by_pref = cand, ms, True
                        counters["n_preferred"] += 1
                if not chosen:
                    counters["preference_unresolved"] += 1
            if not chosen:
                ranked = sorted(cands, key=lambda c: (-c["strength"], _at_key(c["at"]), c["project_id"]))
                best = ranked[0]
                # Highest strength first; among equals the LATEST at wins.
                # `at` shapes differ per source (registry space form, ledger
                # ISO with offset, legacy naive ISO) so compare parsed
                # datetimes, never the raw strings.
                ties = [c for c in ranked if c["strength"] == best["strength"]]
                chosen = max(ties, key=lambda c: (_at_key(c["at"]), c["project_id"]))
                masks = chosen["masks"]
            decisions.append({
                "stem": stem, "label": label, "candidate": chosen, "masks": masks,
                "by_preference": by_pref, "preference": pref if by_pref else None,
            })
    return decisions, images, counters


def _write_data_yaml(write_dir, class_map, path_dir=None):
    """Write data.yaml under write_dir. path_dir (default write_dir) is the
    location recorded in the `path:` field: during a rebuild swap the file
    is written under the temp build dir but must still claim the final
    out_dir, since that is where it will live once the swap completes."""
    path_dir = write_dir if path_dir is None else path_dir
    names = sorted(class_map.items(), key=lambda kv: kv[1])
    lines = [
        f"path: {os.path.abspath(path_dir)}",
        "train: all_images",
        "val: all_images",
        f"nc: {len(names)}",
        "names:",
    ]
    for label, cid in names:
        lines.append(f"  {cid}: {label}")
    with open(os.path.join(write_dir, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def normalize_project_ids(inprocess_root, project_ids):
    """Validate a project filter against the projects actually on disk.

    Accepts the ids the matrix shows (project.json ``id``) and, for
    convenience, run_* directory names, which resolve to the same id the way
    the blueprint's own lookups do. Returns a sorted list of canonical
    project ids, or None when project_ids is None (no filter). Raises
    ValueError naming every unknown id, so a typo fails before a single file
    is written.
    """
    if project_ids is None:
        return None
    wanted = [str(p).strip() for p in project_ids if str(p).strip()]
    known = {}
    for info in scan_projects(inprocess_root):
        known[info["project_id"]] = info["project_id"]
        known[info["run_dir"]] = info["project_id"]
    out, unknown = [], []
    for pid in wanted:
        if pid in known:
            if known[pid] not in out:
                out.append(known[pid])
        else:
            unknown.append(pid)
    if unknown:
        raise ValueError("unknown project id: " + ", ".join(sorted(set(unknown))))
    return sorted(out)


def _source_projects(manifest):
    """[{"project_id", "n_masks"}] for the rows that actually exported, most
    masks first. Always recorded, filter or no filter, so a composite says
    where its training data came from without re-reading the manifest."""
    counts = {}
    for row in manifest:
        pid = row.get("project_id") or ""
        counts[pid] = counts.get(pid, 0) + 1
    return [{"project_id": pid, "n_masks": counts[pid]}
            for pid in sorted(counts, key=lambda p: (-counts[p], p))]


def build_composite(inprocess_root, name=None, labels=None, by="", matrix=None,
                     source_types=None, rebuild=False, project_ids=None):
    """Assemble a composite export. Returns the route result dict (spec 4.1).
    Raises CompositeExists when the directory already exists and rebuild is
    not True, and ValueError (before anything is written) when the resolved
    label set is empty or the project filter names an unknown project.

    source_types: None (default) exports every source type; an explicit
    list restricts candidate masks to those (spec Task 6), see
    _filter_by_source_types / _mask_source_type for the missing-value rule.
    project_ids: None (default) draws on every project; an explicit list
    restricts the candidates to those projects, and is recorded in
    composite.json as "project_filter". Either way "source_projects" records
    the projects the export actually drew from.
    rebuild: when True and the name already exists, the new export is built
    into a temp sibling dir and swapped into place with os.replace (the old
    build's metadata is preserved in composite.json's "history" list)
    instead of raising CompositeExists.
    """
    if source_types is not None:
        source_types = sorted({str(s).strip() for s in source_types if str(s).strip()})
    project_ids = normalize_project_ids(inprocess_root, project_ids)
    name = sanitize_name(name)
    labels = sorted({str(l).strip() for l in (labels or []) if str(l).strip()}) or None
    out_dir = os.path.join(composite_root(inprocess_root), name)
    exists = os.path.exists(out_dir)
    if exists and not rebuild:
        raise CompositeExists(out_dir)

    previous_meta = _read_composite_json(out_dir) if exists else None
    tmp_dir = None
    if exists:
        tmp_dir = out_dir + _REBUILD_TMP_SUFFIX
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
    build_dir = tmp_dir if exists else out_dir

    decisions, images, counters = resolve(inprocess_root, labels, matrix,
                                          source_types=source_types,
                                          project_ids=project_ids)
    label_set = labels or sorted({d["label"] for d in decisions})
    if not label_set:
        raise ValueError("nothing to export: no project holds an accepted, labeled mask "
                         "(a composite needs at least one label)")
    # An explicit label list whose labels resolve to no exportable mask (for
    # example a species whose masks are all rejected, or all excluded by
    # source_types) must not produce a positives-free export that Step 6
    # would then offer as a training source.
    n_resolved = sum(len(d.get("masks") or []) for d in decisions)
    if not n_resolved:
        raise ValueError("nothing to export: the chosen labels resolve to no accepted, "
                         "labeled mask in any project")
    class_map = {label: i for i, label in enumerate(label_set)}

    images_dir = os.path.join(build_dir, "all_images")
    labels_dir = os.path.join(build_dir, "all_labels")
    os.makedirs(images_dir)
    os.makedirs(labels_dir)

    by_stem = {}
    for d in decisions:
        by_stem.setdefault(d["stem"], []).append(d)

    manifest = []
    negatives = []
    by_label = {label: 0 for label in label_set}
    by_site = {}
    n_images = n_masks = 0
    skipped_review_only = skipped_unreviewed = skipped_missing_image = 0
    blocked_unlabeled = sum(img["blocked_unlabeled"] for img in images.values())

    def _link_image(stem, filename, image_path):
        dest = os.path.join(images_dir, filename or (stem + ".jpg"))
        if not os.path.exists(dest):
            os.symlink(os.path.abspath(image_path), dest)
        return dest

    for stem in sorted(images):
        img = images[stem]
        lines = []
        rows = []
        chosen_path = None
        for d in by_stem.get(stem, []):
            cand = d["candidate"]
            cid = class_map[d["label"]]
            # Second, independent source_types check on the chosen project's
            # mask list (defense in depth alongside the collect()-level
            # filter above; d["masks"] should already only hold matching
            # masks, but this guarantees no non-matching mask is ever
            # written regardless of how it reached resolve()).
            for m in _filter_by_source_types(d["masks"], source_types):
                joined = join_polygons(_norm_polygons(m, cand["width"], cand["height"]))
                if len(joined) < 6:
                    continue
                rec = mask_record(cand["filename"], m, {})
                lines.append(_yolo_line(joined, cid))
                rows.append({
                    "basename": stem,
                    "label": d["label"],
                    "project_id": cand["project_id"],
                    "uid": rec["uid"],
                    "status": rec["status"],
                    "source_type": rec["source_type"],
                    "reviewer": rec["reviewer"],
                    "chosen_by_preference": "1" if d["by_preference"] else "",
                    "at": cand["at"],
                })
                by_label[d["label"]] += 1
            if lines and chosen_path is None and cand["image_path"] and os.path.exists(cand["image_path"]):
                chosen_path = (cand["filename"], cand["image_path"])
        if not lines:
            if img["pending_review"]:
                skipped_review_only += 1
                continue
            if not img["reviewed"]:
                skipped_unreviewed += 1
                continue
        if chosen_path is None:
            for _pid, filename, path, _w, _h in img["paths"]:
                if path and os.path.exists(path):
                    chosen_path = (filename, path)
                    break
        if chosen_path is None:
            skipped_missing_image += 1
            continue
        _link_image(stem, chosen_path[0], chosen_path[1])
        with open(os.path.join(labels_dir, stem + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n" if lines else "")
        n_images += 1
        n_masks += len(lines)
        manifest.extend(rows)
        site = _site_for_stem(stem)
        by_site[site] = by_site.get(site, 0) + 1
        if not lines:
            negatives.append(stem)

    with open(os.path.join(build_dir, "class_map.json"), "w", encoding="utf-8") as fh:
        json.dump(class_map, fh, indent=2)
    _write_data_yaml(build_dir, class_map, path_dir=out_dir)
    with open(os.path.join(build_dir, "composite_manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for row in manifest:
            w.writerow(row)

    # Fold the previous build's own metadata (minus its own history, so the
    # chain stays flat rather than nesting) onto the end of the carried
    # history list, oldest first.
    history = []
    if previous_meta:
        history = list(previous_meta.get("history") or [])
        history.append({k: v for k, v in previous_meta.items() if k != "history"})

    created_at = now_ast()
    meta = {
        "name": name,
        "created_at": created_at,
        "by": str(by or "").strip().upper(),
        "labels": label_set,
        "source_types": source_types,
        "source_type_missing_default": _DEFAULT_SOURCE_TYPE,
        "project_filter": project_ids,
        "source_projects": _source_projects(manifest),
        "n_images": n_images,
        "n_masks": n_masks,
        "n_negatives": len(negatives),
        "negatives": negatives,
        "by_label": by_label,
        "by_site": dict(sorted(by_site.items())),
        "skipped_review_only": skipped_review_only,
        "skipped_unreviewed": skipped_unreviewed,
        "skipped_missing_image": skipped_missing_image,
        "blocked_unlabeled": blocked_unlabeled,
        "n_preferred": counters["n_preferred"],
        "preference_unresolved": counters["preference_unresolved"],
        "projects": sorted({r["project_id"] for r in manifest}),
        # Training projects minted from this set (run dir basenames, oldest
        # first). A rebuild keeps the list: those projects still point their
        # step4test symlink here and train on whatever the set now holds.
        "linked_projects": list((previous_meta or {}).get("linked_projects") or []),
        "history": history,
    }
    # Atomic, like every other writer of this file: a create-project call
    # reading it must see the old document or the new one, never a truncated
    # one (it is the file that carries linked_projects).
    _write_json_atomic(os.path.join(build_dir, "composite.json"), meta)
    with open(os.path.join(build_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(_readme(meta, out_dir))

    try:
        description = "Composite export %s: %d images, %d masks. Labels: %s. %s" % (
            meta["name"], meta["n_images"], meta["n_masks"],
            ", ".join(meta["labels"]) or "none", _source_filter_line(meta),
        )
        write_eml(build_dir, meta["name"], description, rows=manifest or None)
    except Exception as e:
        log.warning("EML write failed for composite %s (non-fatal): %s", out_dir, e)

    if exists:
        # Atomic swap: rename() cannot replace a non-empty directory, so the
        # old out_dir is moved aside first (freeing the name), then the new
        # build is moved into place. Each step is a single os.replace (one
        # rename syscall); on any failure the old content is restored.
        #
        # Held under composite_lock so a create-project call cannot read the
        # old composite.json, have the directory swapped underneath it, and
        # then write its linked_projects append into a directory that is about
        # to be removed. The lock covers only the two renames, never the build.
        with composite_lock(inprocess_root):
            # linked_projects may have grown while this build ran; re-read it
            # under the lock so a project minted mid-build is not dropped.
            live_meta = _read_composite_json(out_dir)
            live_linked = list((live_meta or {}).get("linked_projects") or [])
            for run_name in live_linked:
                if run_name not in meta["linked_projects"]:
                    meta["linked_projects"].append(run_name)
            if live_linked != (previous_meta or {}).get("linked_projects"):
                _write_json_atomic(os.path.join(build_dir, "composite.json"), meta)
            backup_dir = out_dir + _REBUILD_BACKUP_SUFFIX
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            os.replace(out_dir, backup_dir)
            try:
                os.replace(build_dir, out_dir)
            except Exception:
                if os.path.exists(out_dir):
                    shutil.rmtree(out_dir, ignore_errors=True)
                os.replace(backup_dir, out_dir)
                raise
            else:
                shutil.rmtree(backup_dir, ignore_errors=True)

    result = {"success": True, "out_dir": out_dir}
    result.update(meta)
    return result


def _source_filter_line(meta):
    """The README line describing the source_types filter applied to this
    build, and the missing-source_type-defaults-to-manual_click rule. An
    empty source_types list has no branch here: it matches no mask anywhere,
    so build_composite always raises "nothing to export" before a README is
    ever written for that case."""
    types = meta.get("source_types")
    if types is None:
        return "Source filter: none, every source type is included."
    default = meta.get("source_type_missing_default", _DEFAULT_SOURCE_TYPE)
    return (f"Source filter: {', '.join(types)}. A mask with a missing or empty "
            f"source_type is treated as {default} for this filter.")


def _project_filter_line(meta):
    """The README line describing the project filter applied to this build."""
    pids = meta.get("project_filter")
    if pids is None:
        return "Project filter: none, every project was considered."
    return "Project filter: " + (", ".join(pids) or "none") + "."


def _site_table_lines(meta):
    """Markdown table rows for the per-site image count, site parsed from
    the TCRMPyyyymmdd_clip_SITE_Tnnn basename pattern (see _site_for_stem)."""
    by_site = meta.get("by_site") or {}
    lines = ["| site | images |", "| --- | --- |"]
    for site in sorted(by_site):
        lines.append(f"| {site} | {by_site[site]} |")
    return lines


def _readme(meta, out_dir):
    lines = [
        f"# Composite export {meta['name']}",
        "",
        f"Built {meta['created_at']} AST" + (f" by {meta['by']}" if meta["by"] else "") + ".",
        "",
        "A YOLO-seg training set assembled across projects from the coverage",
        "matrix. For each image and label the masks come from the project a",
        "reviewer chose in the matrix (preferences.csv), otherwise from the",
        "project whose outcome ranks highest (found_expert > found_manual >",
        "found_ai > found_model), ties broken by the latest annotation time.",
        "Only accepted, non-review, labeled masks export. Images with no",
        "exported mask are negatives when some project reviewed them and no",
        "project still holds a pending review mask; otherwise they are skipped.",
        "",
        _source_filter_line(meta),
        "",
        _project_filter_line(meta),
        "",
        "## Contents",
        "",
        "- all_images/: symlinks to the source images",
        "- all_labels/: one YOLO-seg .txt per image (empty file = negative)",
        "- data.yaml, class_map.json: dataset config, class ids in sorted label order",
        "- composite_manifest.csv: one row per exported mask with its source project, uid, reviewer",
        "- composite.json: the counts below, machine readable",
        "",
        "## Counts",
        "",
        f"- images: {meta['n_images']} (negatives: {meta['n_negatives']})",
        f"- masks: {meta['n_masks']}",
        f"- labels: {', '.join(meta['labels']) or 'none'}",
        f"- chosen by preference: {meta['n_preferred']} (unresolved preferences: {meta['preference_unresolved']})",
        f"- skipped, review pending: {meta['skipped_review_only']}",
        f"- skipped, never reviewed: {meta['skipped_unreviewed']}",
        f"- skipped, image file missing: {meta['skipped_missing_image']}",
        f"- accepted masks without a label (never exported): {meta['blocked_unlabeled']}",
        f"- source projects: {', '.join(meta['projects']) or 'none'}",
        "- masks per source project: " + (", ".join(
            f"{s['project_id']} {s['n_masks']}" for s in meta.get("source_projects") or []
        ) or "none"),
        "",
        "## Per-site image counts",
        "",
        *_site_table_lines(meta),
        "",
        f"Path: {out_dir}",
        "",
    ]
    return "\n".join(lines)


def source_projects_for(out_dir, meta=None):
    """[{"project_id", "n_masks"}] for a composite on disk. Reads
    composite.json's own source_projects when it has one; a composite built
    before that key existed is counted from composite_manifest.csv instead of
    being reported as having no sources."""
    meta = _read_composite_json(out_dir) if meta is None else meta
    listed = (meta or {}).get("source_projects")
    if isinstance(listed, list) and listed:
        return listed
    path = os.path.join(out_dir, "composite_manifest.csv")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return _source_projects(list(csv.DictReader(fh)))
    except OSError:
        return []


def _append_linked_project(inprocess_root, out_dir, run_name):
    """Record a training project in the composite's composite.json
    linked_projects list. Returns True when the file was updated.

    The whole read-append-write runs under composite_lock and finishes with an
    atomic replace, so concurrent minters cannot drop each other's entries and
    no reader ever sees a half-written file.

    An UNREADABLE composite.json is left exactly as it is: replacing it with a
    fresh minimal document would throw away whatever the corrupt file still
    holds. The training project is created either way; only the back link is
    missing, and a rebuild of the set restores a readable file.
    """
    path = os.path.join(out_dir, "composite.json")
    with composite_lock(inprocess_root):
        meta = _read_composite_json(out_dir)
        if meta is None:
            if os.path.exists(path):
                log.warning("composite.json at %s is unreadable; linked_projects not updated",
                            out_dir)
                return False
            meta = {"name": os.path.basename(out_dir)}
        linked = list(meta.get("linked_projects") or [])
        if run_name not in linked:
            linked.append(run_name)
        meta["linked_projects"] = linked
        _write_json_atomic(path, meta)
    return True


def _training_readme(composite_name, sources, run_name, created_at):
    """The run dir's README: what this project trains on, in plain language."""
    lines = [
        f"# {run_name}",
        "",
        f"Created {created_at} AST.",
        "",
        f"This project trains on the training set {composite_name}, assembled on the",
        "coverage matrix from masks that were annotated in other projects. It holds no",
        "annotation work of its own: steps 1 to 4 are skipped, and the project opens on",
        "Train.",
        "",
        "## Where the training data lives",
        "",
        f"step4test_combinedAnnotate is a symlink to inprocess/_composite/{composite_name},",
        "so Train reads that set's all_images/, all_labels/ and data.yaml directly. No",
        "image is copied. Editing or rebuilding the training set changes what this",
        "project trains on, and deleting the set leaves this project with nothing to",
        "read.",
        "",
        "That link is relative, so this project stays paired with its set if the whole",
        "inprocess folder moves. The set's own all_images/ entries are absolute symlinks",
        "to the source frames, so moving the survey imagery breaks the set until it is",
        "rebuilt from the matrix.",
        "",
        "## Source projects",
        "",
    ]
    if sources:
        lines.append("| project | id | masks |")
        lines.append("| --- | --- | --- |")
        for s in sources:
            lines.append("| %s | %s | %s |" % (s.get("name") or s.get("project_id", ""),
                                               s.get("project_id", ""), s.get("n_masks", 0)))
    else:
        lines.append("No source project was recorded for this training set.")
    lines.extend([
        "",
        "The masks stay owned by the projects above. The coverage matrix keeps naming",
        "them as the place each annotation was made, and names this project as one of",
        "the training projects built from the set.",
        "",
    ])
    return "\n".join(lines)


def create_training_project(inprocess_root, name, by):
    """Mint a pipeline project that trains from the composite `name`.

    Creates <inprocess_root>/run_{YY-MM-DD}_{INITIALS}_train_{name} (with a
    _2 .. _99 suffix on a same-day collision) holding:

      step4test_combinedAnnotate -> ../_composite/<name>   (relative symlink)
      project.json    the regular project shape plus project_type
                      "composite_training" and a composite block; steps 1 to
                      4 are "skipped" and current_step is 6 (Train)
      README.md       what the project trains on and where it came from

    The composite's composite.json gains the new run dir basename in its
    linked_projects list, which is how the matrix knows the downstream
    training projects (nothing scans for them).

    Returns {"run_dir" (absolute path), "run_dir_name" (basename),
    "project_id", "display_name"}. Raises FileNotFoundError when the
    composite does not exist, ValueError on bad initials or on a set that
    holds no images, and RunDirExhausted when no run dir is free today.

    The run dir itself is written by
    _reefreview.training_projects.mint_training_project, which mints both
    kinds of training project (composite and bbox) so the two cannot drift.
    Four things stay here because only a composite caller can do them:

      1. resolving `name` inside the composite root, refusing a separator, a
         traversal or a symlink out (the name arrives from a request body);
      2. refusing a set whose all_images/ is empty, before anything is minted;
      3. enriching the recorded source projects with the display name of every
         source still on disk;
      4. the composite.json linked_projects back link, which is this module's
         own file and is written under its own lock.

    Concurrency: the run dir is claimed by the mkdir inside the shared minter,
    and the composite.json back link is appended under composite_lock with an
    atomic replace, so simultaneous calls cannot collide on a name or lose
    each other's linked_projects entries.
    """
    initials = str(by or "").strip()
    if not _INITIALS_RE.match(initials):
        raise ValueError("initials must be 1 to 4 letters")
    initials = initials.upper()

    # The name arrives from a request body, so it is resolved and checked the
    # way list_composites checks an entry: a real directory sitting directly
    # inside the composite root, with a data.yaml. Anything else (a path
    # separator, a traversal, a symlink out) is simply unknown.
    # The two scratch siblings a rebuild leaves mid-swap are refused by name
    # here, not only hidden by list_composites: one is a half-written export
    # and the other is about to be removed, so a project minted from either
    # would train on something that is not there minutes later. list_composites
    # never offers them, so a name reaching this point is a hand-typed or
    # stale request.
    composite_name = str(name or "").strip()
    root = composite_root(inprocess_root)
    out_dir = os.path.join(root, composite_name)
    if not composite_name or os.sep in composite_name \
            or composite_name.endswith(_REBUILD_TMP_SUFFIX) \
            or composite_name.endswith(_REBUILD_BACKUP_SUFFIX) \
            or os.path.dirname(os.path.realpath(out_dir)) != os.path.realpath(root) \
            or not os.path.isdir(out_dir) \
            or not os.path.isfile(os.path.join(out_dir, "data.yaml")):
        raise FileNotFoundError(f"unknown training set: {composite_name}")

    # A set with no images trains on nothing: refuse here with a message that
    # says what to do, rather than minting a project whose Step 6 fails its own
    # preflight later with a path the reader has to work out.
    images_dir = os.path.join(out_dir, "all_images")
    if not os.path.isdir(images_dir) or not os.listdir(images_dir):
        raise ValueError(f"training set {composite_name} holds no images; rebuild it from "
                         "the matrix before making a project from it")

    meta = _read_composite_json(out_dir) or {}
    # The composite records ids and mask counts. Carry the display name of
    # every source project still on disk into the project's own copy, so the
    # Train panel can name the projects a reader recognizes rather than ids.
    # A purged source simply keeps its id and no name.
    project_names = {info["project_id"]: info["name"] for info in scan_projects(inprocess_root)}
    sources = []
    for s in source_projects_for(out_dir, meta):
        row = dict(s)
        display = project_names.get(row.get("project_id"))
        if display and display != row.get("project_id"):
            row["name"] = display
        sources.append(row)

    # Everything after this point is the shared minter's job: the run dir
    # claim, the relative step4test symlink, the four skipped step folders,
    # project.json (written with an atomic replace) and a README. The
    # composite block it writes carries `sources` exactly as enriched above,
    # and path_label is passed explicitly so this module keeps naming its own
    # source the way the composite contract always has.
    inprocess_root = os.path.abspath(inprocess_root)
    minted = mint_training_project(
        inprocess_root, "composite", composite_name, out_dir, initials,
        extra={
            "source_projects": sources,
            "path_label": os.path.join("inprocess", COMPOSITE_DIRNAME, composite_name),
        },
    )
    run_dir, run_name = minted["run_dir"], minted["run_dir_name"]

    # The README is the one artifact this module still writes itself: it names
    # the source projects by display name, and explains what the relative link
    # does and does not survive, neither of which the shared minter knows.
    # Overwriting the one the mint wrote keeps a composite project's README
    # byte-identical to what it was before this delegation.
    _training_readme_path = os.path.join(run_dir, "README.md")
    with open(_training_readme_path, "w", encoding="utf-8") as fh:
        fh.write(_training_readme(composite_name, sources, run_name,
                                  _read_created_at(run_dir)))

    _append_linked_project(inprocess_root, out_dir, run_name)

    return dict(minted)


def _read_created_at(run_dir):
    """The composite block's created_at stamp the mint just wrote, so the
    README carries the same one rather than a second call to the clock a
    second later. '' when project.json cannot be read (the README then says
    nothing false; the mint itself already succeeded)."""
    try:
        with open(os.path.join(run_dir, "project.json"), encoding="utf-8") as fh:
            return str(((json.load(fh) or {}).get("composite") or {}).get("created_at") or "")
    except (OSError, ValueError):
        return ""


def list_composites(inprocess_root):
    """Every composite under <inprocess_root>/_composite that carries a
    data.yaml, newest first. An entry whose real path resolves outside the
    composite root (a symlink out) is skipped, mirroring the orchestrator's
    training-source guard, and `created_at` is always
    'YYYY-MM-DD HH:MM:SS' AST.

    The two scratch siblings a rebuild uses (<name>.rebuild-tmp while the new
    export is being written, <name>.rebuild-backup while the swap is in
    flight) are skipped as well. Both carry a data.yaml, so without this a
    rebuild running in another tab offers a half-written set as a training
    source, and the backup vanishes under any project minted from it.

    Each entry also carries `linked_project_dirs`: the absolute run-dir path
    for every basename in `linked_projects`, same order, `os.path.join`
    against `inprocess_root`. A basename that no longer exists on disk is
    still listed; the caller decides how to show a missing run."""
    root = composite_root(inprocess_root)
    out = []
    if not os.path.isdir(root):
        return out
    real_root = os.path.realpath(root)
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if name.endswith(_REBUILD_TMP_SUFFIX) or name.endswith(_REBUILD_BACKUP_SUFFIX):
            continue
        if not os.path.isdir(path) or not os.path.isfile(os.path.join(path, "data.yaml")):
            continue
        if not os.path.realpath(path).startswith(real_root + os.sep):
            continue
        meta = {}
        meta_path = os.path.join(path, "composite.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh) or {}
            except (OSError, ValueError):
                meta = {}
        labels = meta.get("labels")
        if labels is None:
            labels = []
            cm_path = os.path.join(path, "class_map.json")
            if os.path.isfile(cm_path):
                try:
                    with open(cm_path, "r", encoding="utf-8") as fh:
                        cm = json.load(fh) or {}
                    labels = [k for k, _v in sorted(cm.items(), key=lambda kv: kv[1])]
                except (OSError, ValueError):
                    labels = []
        n_images = meta.get("n_images")
        if n_images is None:
            img_dir = os.path.join(path, "all_images")
            n_images = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else 0
        n_masks = meta.get("n_masks")
        if n_masks is None:
            n_masks = 0
            man = os.path.join(path, "composite_manifest.csv")
            if os.path.isfile(man):
                try:
                    with open(man, newline="", encoding="utf-8") as fh:
                        n_masks = sum(1 for _ in csv.DictReader(fh))
                except OSError:
                    n_masks = 0
        created_at = _fmt_ast(meta.get("created_at"))
        if not created_at:
            try:
                created_at = datetime.fromtimestamp(os.path.getmtime(path), _AST).strftime("%Y-%m-%d %H:%M:%S")
            except OSError:
                created_at = ""
        linked = meta.get("linked_projects") or []
        out.append({
            "name": name,
            "path": path,
            "created_at": created_at,
            "n_images": n_images,
            "n_masks": n_masks,
            "labels": labels,
            "n_negatives": meta.get("n_negatives", 0),
            "by": meta.get("by", ""),
            # Downstream training projects minted from this set, and the
            # projects the masks came from. Both are read from composite.json
            # only; nothing scans the tree for them.
            "linked_projects": linked,
            "linked_project_dirs": [os.path.join(inprocess_root, b) for b in linked],
            "source_projects": source_projects_for(path, meta),
            "project_filter": meta.get("project_filter"),
        })
    out.sort(key=lambda c: (c["created_at"], c["name"]), reverse=True)
    return out

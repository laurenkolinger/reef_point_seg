"""Cross-project no-re-screening ledger.

2026-09-27 VEGA annotator tidy-up plan (Lauren, verbatim): "If an image has
previously been screened for all target classes, do not include it in the
image selection options. If an image has been previously screened for any
target classes, do include it ... and include any masks as previews in the
annotator for the present project ... I don't want people retreading each
other's work."

Definitions (docs/superpowers/plans/2026-09-27-vega-annotator-tidy-and-no-
rescreening.md):
  Screened        an image is screened for a label when a project OTHER than
                  the current one recorded an outcome for that label on it:
                  found_expert, found_manual, found_ai, pending_expert or
                  not_found. found_model is deliberately excluded: it is
                  reserved for an unreviewed model prediction (no writer
                  emits it today), which is not a human screening decision
                  and must never suppress re-annotation.
  Fully screened  every one of the new project's target labels is screened.
  Partly screened at least one target label is screened, but not all.

Everything here reads the coverage matrix (`_matrix.builder.build_matrix`),
which already aggregates every inprocess project's label outcomes per image
stem from each project's label_provenance.csv ledger, its legacy
segmentations.json fallback, or the cross-project mask registry. This module
adds no new aggregation: it asks build_matrix's own cells "which labels does
some OTHER project already have an outcome for", and assembles the two
on-disk contracts Step 3's selector and the combined annotator read.

Two entry points:
  screened_by_stem(runs_root, exclude_project_id, exclude_project_dir)
      {stem: sorted [labels]} screened by some other project. Step 3's
      selector (TCRMPcvr_chooseImages/src/select_images.py) uses this,
      through the {stem: [labels]} JSON the orchestrator writes as
      prior_screening_input.json, to drop fully-screened frames before
      selection ever runs.
  write_prior_screening(project_dir, frames, runs_root, master_codes_csv=)
      Writes <project_dir>/step4test_combinedAnnotate/prior_screening.json:
      for every one of `frames` with at least one screened target label
      (whether that leaves it partly screened or fully screened), which
      labels are already screened (by whom, what outcome) and every accepted
      mask an OTHER project holds for those labels, so the combined
      annotator (W3) can draw them as locked previews instead of asking
      someone to redo work another reviewer already finished. A frame with
      zero screened labels gets no entry.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

_PKG_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_SCRIPTS not in sys.path:
    sys.path.insert(0, _PKG_SCRIPTS)

from _matrix import names as _names  # noqa: E402
from _matrix.builder import (  # noqa: E402
    _MANUAL_SOURCE_TYPES,
    _OUTCOME_STRENGTH,
    _is_expert_mask,
    _read_project_json,
    _target_species,
    build_matrix,
    scan_projects,
)
from _matrix.cellinfo import load_project_segmentations, mask_record  # noqa: E402
from _matrix.composite import _norm_polygons  # noqa: E402

# Outcomes that count as "screened" per the plan's definition (see module
# docstring). A set, not the full _OUTCOME_STRENGTH mapping, because
# membership is all this module ever asks; found_model is the one strength
# key deliberately left out.
SCREENED_OUTCOMES = frozenset(
    {"found_expert", "found_manual", "found_ai", "pending_expert", "not_found"}
)

# Directory name of the combined annotator's export dir, where
# prior_screening.json is written. Mirrors pipeline_orchestrator/
# project_manager.py's STEP_DIRS["step4test"]; named here as a literal
# constant rather than imported, because pipeline_orchestrator is a CONSUMER
# of this package (the orchestrator calls write_prior_screening) and must
# never be a dependency of it. reef_point_seg/github_repo/CLAUDE.md: "the
# internal step keys and output directories ... never change."
STEP4TEST_DIRNAME = "step4test_combinedAnnotate"

# Schema version of prior_screening.json and prior_screening_input.json.
CONTRACT_VERSION = 1

# Fixed Atlantic Standard Time (UTC-4, no DST), matching every other AST
# timestamp helper in this package (builder.py, composite.py, app.py).
_AST = timezone(timedelta(hours=-4))


def _ast_now_iso():
    """ISO-8601 timestamp pinned to Atlantic Standard Time (UTC-4, no DST)."""
    return datetime.now(_AST).isoformat()


def _current_project_identity(project_dir):
    """(project_id, target_species) for the project rooted at project_dir,
    read straight from its own project.json with the same id-or-dirname
    fallback scan_projects() applies, so the project this module excludes
    from "screened elsewhere" is always the one the coverage matrix would
    name for that directory.

    Parameters:
        project_dir: the project's root directory (holds project.json).
    Returns:
        (project_id, target_species): project_id is "" when project_dir
        holds no readable project.json (the safe default is then to exclude
        nothing, so every project's coverage counts, rather than raising for
        a project not yet fully written to disk); target_species is the
        list parsed from steps.3.config.target_species, or [] when unset.
    """
    meta = _read_project_json(project_dir)
    if meta is None:
        return "", []
    project_id = str(meta.get("id") or os.path.basename(os.path.normpath(project_dir)))
    return project_id, _target_species(meta)


def _resolve_exclude_ids(runs_root, exclude_project_id, exclude_project_dir):
    """project_id values a cell's source must never match to count as
    "another project" for screening purposes.

    Always includes exclude_project_id verbatim (stringified) when given.
    Also resolves exclude_project_dir to ITS OWN project_id via
    scan_projects(runs_root), so a project whose project.json 'id' field
    disagrees with, or is absent from, the caller's notion of "current" (
    scan_projects falls back to the run directory's basename in that case)
    is still excluded correctly. Either argument may be omitted; omitting
    both excludes nothing, so every project's coverage counts.

    Parameters:
        runs_root: the module's inprocess/ root, rescanned via scan_projects.
        exclude_project_id: the current project's id, or None/"".
        exclude_project_dir: the current project's directory, or None/"".
    Returns:
        set[str]: every project_id to exclude (0, 1 or 2 entries). Never
        raises: an unreadable inprocess tree resolves through
        scan_projects' own tolerant scanning to no match found.
    """
    ids = set()
    if exclude_project_id:
        ids.add(str(exclude_project_id))
    if exclude_project_dir:
        norm = os.path.abspath(exclude_project_dir)
        for info in scan_projects(runs_root):
            if os.path.abspath(info["project_dir"]) == norm:
                ids.add(info["project_id"])
                break
    return ids


def _screened_labels_for_stem(by_label, exclude_ids):
    """Labels of one stem's cell-map (matrix['cells'][stem]) screened by a
    project outside exclude_ids. The one place that walks a cell's sources
    against SCREENED_OUTCOMES, shared by screened_by_stem and
    write_prior_screening so the two can never disagree on what "screened"
    means for the same stem."""
    labels = set()
    for label, cell in (by_label or {}).items():
        for src in (cell.get("sources") or []):
            if str(src.get("project_id") or "") in exclude_ids:
                continue
            if src.get("outcome") in SCREENED_OUTCOMES:
                labels.add(label)
                break
    return labels


def screened_by_stem(runs_root, exclude_project_id=None, exclude_project_dir=None):
    """{stem: sorted [labels]} screened by some OTHER project, across every
    inprocess/run_* project build_matrix(runs_root) knows about.

    A label counts for a stem when at least one of that cell's sources
    belongs to a project other than the excluded one(s) and its outcome is
    in SCREENED_OUTCOMES. Archived, standalone and composite/box-training
    projects never enter here: build_matrix's own scan_projects already
    leaves them out (composite/box training via is_training_project;
    standalone and archived runs because they are not under inprocess/run_*
    at all).

    Parameters:
        runs_root: the module's inprocess/ root (build_matrix rescans it on
            every call; nothing here caches across calls).
        exclude_project_id: the current project's id (or run-dir fallback,
            matching what scan_projects would report for it); never counted
            as "another project". May be omitted.
        exclude_project_dir: the current project's directory, a second,
            independent way to name it (see _resolve_exclude_ids). Either or
            both of the two exclude_* arguments may be given; omitting both
            is only correct when there is no current project yet, and
            returns coverage as any project would see it.

    Returns:
        dict[str, list[str]]: stem -> sorted distinct labels, present only
        for stems with at least one screened label. Never raises: an
        unreadable inprocess tree resolves through build_matrix's own
        tolerant scanning to an empty matrix, which yields {}.
    """
    exclude_ids = _resolve_exclude_ids(runs_root, exclude_project_id, exclude_project_dir)
    matrix = build_matrix(runs_root)
    result = {}
    for stem, by_label in (matrix.get("cells") or {}).items():
        labels = _screened_labels_for_stem(by_label, exclude_ids)
        if labels:
            result[stem] = sorted(labels)
    return result


def _strongest_other_source(sources, exclude_ids):
    """The strongest SCREENED_OUTCOMES source among `sources` whose
    project_id is not in exclude_ids, or None when none qualifies.
    'Strongest' mirrors builder._OUTCOME_STRENGTH; a tie keeps whichever
    source is encountered first, which is deterministic because build_matrix
    orders a cell's sources by scan_projects' own directory-name sort."""
    best = None
    for src in sources or []:
        if str(src.get("project_id") or "") in exclude_ids:
            continue
        outcome = src.get("outcome")
        if outcome not in SCREENED_OUTCOMES:
            continue
        if best is None or _OUTCOME_STRENGTH.get(outcome, -1) > _OUTCOME_STRENGTH.get(best["outcome"], -1):
            best = src
    return best


def _mask_outcome(mask):
    """found_expert/found_manual/found_ai for one accepted, non-review mask,
    mirroring the same three-way split builder._legacy_rows applies to every
    accepted mask when deriving a project's own provenance rows (expert
    beats manual beats ai)."""
    if _is_expert_mask(mask):
        return "found_expert"
    if str(mask.get("source_type") or "") in _MANUAL_SOURCE_TYPES:
        return "found_manual"
    return "found_ai"


def write_prior_screening(project_dir, frames, runs_root, *, master_codes_csv=None):
    """Write <project_dir>/step4test_combinedAnnotate/prior_screening.json:
    the read-only "already screened elsewhere" ledger the combined annotator
    loads at configure and resume, and the Step 3 completion panel counts
    against. Regenerated in full on every call (never merged with a prior
    file), and covers every one of this project's frames (among `frames`)
    that carries at least one label screened by another project, whether
    that leaves the frame partly screened or fully screened; only a frame
    with zero screened labels gets no entry.

    A brand-new Step 3 selection never contains a fully-screened frame
    (select_images.py's own --prior-screening gate drops it before selection
    runs), but `frames` here is not limited to a fresh selection: a frame
    this project selected before that gate existed, or before some other
    project finished screening every target label on it, can be fully
    screened by the time an annotator launch calls this function, and it
    still needs its previews. Excluding it here would silently hide another
    reviewer's finished work from someone about to open that exact frame.

    Parameters:
        project_dir: the current project's root directory (holds
            project.json; its own id and target_species, read from there,
            decide what "fully" vs "partly" screened means for this call).
            The file is written under this project's step4test export dir,
            created if it does not already exist.
        frames: an iterable of image stems (as build_matrix keys them: the
            filename without its directory or extension) belonging to this
            project to check. Duplicates are ignored; a falsy stem is
            skipped. Typically the current project's Step 3 selected-frames
            basenames.
        runs_root: the module's inprocess/ root that build_matrix rescans.
        master_codes_csv: optional code,category,name CSV resolving a
            screened label to its full species name for each mask's "name"
            field (the same table the coverage matrix and Label Manager
            already read via _matrix.names.read_label_names). Omitted or
            unreadable: every mask's "name" is "", and nothing raises
            (names.py's own tolerant contract).

    Returns:
        str: the path prior_screening.json was written to.

    Raises:
        ValueError: project_dir or runs_root is falsy.
        OSError: the export directory could not be created, or the file
            could not be written. Deciding that such a failure must never
            block Step 3 or an annotator launch is the CALLER's job (the
            orchestrator logs it via vic.note and proceeds); this function
            itself always either writes a complete, valid file or raises.
    """
    if not project_dir:
        raise ValueError("write_prior_screening: project_dir is required")
    if not runs_root:
        raise ValueError("write_prior_screening: runs_root is required")

    project_id, target_species = _current_project_identity(project_dir)
    exclude_ids = _resolve_exclude_ids(runs_root, project_id, project_dir)
    target_set = set(target_species)

    matrix = build_matrix(runs_root)
    cells = matrix.get("cells") or {}
    label_names = _names.read_label_names(master_codes_csv) if master_codes_csv else {}

    # Every OTHER project's parsed segmentations, loaded once and reused
    # across every stem this call checks rather than once per stem.
    other_projects = [p for p in (matrix.get("projects") or [])
                      if p["project_id"] not in exclude_ids]
    segs_by_project = {p["project_id"]: load_project_segmentations(p["project_dir"])
                       for p in other_projects}
    names_by_project = {p["project_id"]: p["name"] for p in other_projects}

    frames_out = {}
    for stem in sorted({s for s in (frames or ()) if s}):
        by_label = cells.get(stem) or {}
        screened_labels = _screened_labels_for_stem(by_label, exclude_ids) & target_set
        if not screened_labels:
            # Nothing screened by any other project: no entry, partial or
            # full. A frame fully screened elsewhere DOES get an entry below
            # (same schema as a partly-screened one): a brand-new Step 3
            # selection never includes such a frame (select_images.py's own
            # --prior-screening gate drops it before selection runs), but
            # `frames` here is whatever the caller says belongs to THIS
            # project (typically its own selected_frames.csv), which can
            # carry a frame selected before this gate existed, or before some
            # other project finished screening every target label on it. On
            # the real inprocess tree, run_26-09-18_AT alone has 101 of its
            # 750 selected frames already screened for PA elsewhere; every
            # one of those still needs its previews, not silence.
            continue

        screened = {}
        for label in sorted(screened_labels):
            src = _strongest_other_source((by_label.get(label) or {}).get("sources"), exclude_ids)
            if src is None:
                continue
            screened[label] = {
                "outcome": src.get("outcome"),
                "project_id": src.get("project_id", ""),
                "project_name": src.get("name", ""),
                "reviewer": str(src.get("reviewer") or "").strip().upper(),
                "at": src.get("at", ""),
            }

        masks = []
        for pid, segs in segs_by_project.items():
            entry = segs.get(stem)
            if entry is None:
                continue
            seg = entry["seg"]
            width, height = seg.get("image_width"), seg.get("image_height")
            for m in (seg.get("masks") or []):
                if not isinstance(m, dict):
                    continue
                label = m.get("species") or ""
                if label not in screened_labels or m.get("status") != "accepted":
                    continue
                rec = mask_record(entry["filename"], m, seg, label)
                masks.append({
                    "label": label,
                    "name": label_names.get(label, ""),
                    "polygon_norm": _norm_polygons(m, width, height),
                    "project_id": pid,
                    "project_name": names_by_project.get(pid, ""),
                    "reviewer": rec["reviewer"],
                    "at": rec["at"],
                    "outcome": _mask_outcome(m),
                })
        masks.sort(key=lambda r: (r["label"], r["project_id"], r.get("at", "")))

        frames_out[stem] = {"screened": screened, "masks": masks}

    doc = {
        "version": CONTRACT_VERSION,
        "generated_at": _ast_now_iso(),
        "project_id": project_id,
        "frames": frames_out,
    }

    out_dir = os.path.join(project_dir, STEP4TEST_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prior_screening.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return out_path

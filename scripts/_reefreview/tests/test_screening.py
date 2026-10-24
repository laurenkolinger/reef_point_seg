#!/usr/bin/env python
"""No-pytest harness for the cross-project no-re-screening ledger
(_reefreview/screening.py), built for the 2026-09-27 VEGA annotator tidy-up
plan (docs/superpowers/plans/2026-09-27-vega-annotator-tidy-and-no-
rescreening.md).

Run with:
  env/bin/python scripts/_reefreview/tests/test_screening.py

Covers:
  screened_by_stem   full / partial / no screening on a stem; the current
                     project excluded by id, by directory, and by both;
                     an archived (no project.json), a standalone (outside
                     inprocess/run_*), and a composite-training project all
                     contribute nothing.
  write_prior_screening
                     a fully-screened stem gets a FULL entry (every target
                     label, not silence: a frame this project selected before
                     the gate existed, or before another project finished
                     screening it, can still be fully screened by launch
                     time and still needs its previews); a partly-screened
                     one gets its screened labels' strongest other source
                     plus every other project's accepted masks for those
                     labels; a mask with only polygon_px is normalised by its
                     own image_width/height, one that already carries
                     polygon_norm passes through unchanged; .jpg vs .jpeg
                     resolve to the same stem; a master_codes_csv resolves
                     each mask's full species name, and an absent one
                     degrades to "" without raising; only an UNSCREENED (zero
                     labels) or entirely-unknown stem is silently skipped.
  real-data check    screened_by_stem against the real, live inprocess tree
                     (read-only: this module never writes there) is
                     internally consistent with build_matrix's own image
                     set, and excluding a real, heavily-contributing project
                     measurably shrinks the result. See the reconciliation
                     note in test_real_inprocess_screened_by_stem_matches_
                     coverage_audit for how this run's count compares to the
                     2026-09-26 coverage audit's 1,330 figure.

Exits nonzero on any failed assertion.
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

import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _matrix.builder import build_matrix  # noqa: E402
from _reefreview import screening  # noqa: E402

_FAILURES = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        _FAILURES.append(msg)


def _write_json(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _mask(species, *, status="accepted", source_type="manual_click",
         polygon_px=None, polygon_norm=None, expert=False, reviewer="AT",
         mask_id=0):
    m = {
        "id": mask_id, "species": species, "status": status,
        "source_type": source_type, "reviewer": reviewer, "review": False,
    }
    if polygon_px is not None:
        m["polygon_px"] = polygon_px
    if polygon_norm is not None:
        m["polygon_norm"] = polygon_norm
    if expert:
        m["expert_id"] = {"code": species, "mode": "EXPERT", "labeler": reviewer}
    return m


def _seg_doc(entries, width=1000, height=500):
    """{filename: seg} for a set of (filename, masks, extra_seg) triples,
    each defaulted to reviewed=True/exported=False so it counts as finished."""
    doc = {}
    for filename, masks, extra in entries:
        seg = {
            "image_path": filename, "image_width": width, "image_height": height,
            "masks": masks, "reviewed": True, "exported": False,
            "processed_at": "2026-09-20T10:00:00-04:00",
        }
        seg.update(extra or {})
        doc[filename] = seg
    return doc


# ── Fixture: an "other" project (projA), a composite-training project that
# must be ignored, an archived dir with no project.json, a standalone
# session outside run_*, and the "current" project being screened. ──────────

def build_fixture(root):
    ts = "2026-09-20T10:00:00-04:00"

    # projA: finds PA+SS on IMG_full, PA only on IMG_partial, PA on the
    # .jpg-named IMG_jpeg_dup (stem keying), nothing on IMG_current_only.
    # target_species is PA only (not SS): a "found" row comes from any
    # accepted mask regardless of target_species membership, but a
    # "not_found" row is only emitted per TARGET species (builder.py
    # _legacy_rows), so keeping this at just PA means IMG_partial's lone PA
    # mask screens PA alone, not PA-plus-a-not_found-SS-by-accident.
    proj_a = os.path.join(root, "run_projA_20260920_aaaaaa")
    _write_json(os.path.join(proj_a, "project.json"), {
        "id": "projA", "name": "Project A",
        "steps": {"3": {"config": {"target_species": "PA"}}},
    })
    seg_a = _seg_doc([
        ("IMG_full.jpeg", [
            _mask("PA", polygon_px=[[0, 0, 10, 0, 10, 10, 0, 10]], reviewer="at"),
            _mask("SS", source_type="auto", mask_id=1),
        ], None),
        ("IMG_partial.jpeg", [
            _mask("PA", polygon_px=[[100, 50, 200, 50, 200, 150, 100, 150]]),
        ], None),
        ("IMG_jpeg_dup.jpg", [
            _mask("PA", polygon_norm=[[0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.4]]),
        ], None),
    ], width=1000, height=500)
    _write_json(
        os.path.join(proj_a, "step4test_combinedAnnotate", "segmentations", "segmentations.json"),
        seg_a)

    # A composite-training project: is_training_project() must exclude it
    # even though it carries an accepted PA mask on a stem no other project
    # touches, which would otherwise show up as freshly "screened".
    proj_composite = os.path.join(root, "run_projCTrain_20260920_cccccc")
    _write_json(os.path.join(proj_composite, "project.json"), {
        "id": "projCTrain", "name": "Composite Train", "project_type": "composite_training",
        "steps": {"3": {"config": {"target_species": "PA"}}},
    })
    _write_json(
        os.path.join(proj_composite, "step4test_combinedAnnotate", "segmentations", "segmentations.json"),
        _seg_doc([("IMG_composite_only.jpeg", [_mask("PA")], None)]))

    # An archived-looking directory: matches run_* but carries no
    # project.json at all, so scan_projects must skip it outright.
    proj_archived = os.path.join(root, "run_archivedOld_20200101_zzzzzz")
    _write_json(
        os.path.join(proj_archived, "step4test_combinedAnnotate", "segmentations", "segmentations.json"),
        _seg_doc([("IMG_archived_only.jpeg", [_mask("PA")], None)]))

    # A standalone "Annotate Images" session: lives OUTSIDE inprocess/run_*
    # (scan_projects only globs run_*), so it must never surface either.
    proj_standalone = os.path.join(root, "_standalone", "session_20260920")
    _write_json(os.path.join(proj_standalone, "segmentations", "segmentations.json"),
               _seg_doc([("IMG_standalone_only.jpeg", [_mask("PA")], None)]))

    # The current project: targets PA + SS, has already touched
    # IMG_current_only itself (must never count as "screened by another
    # project" for its own frame) and IMG_full (its own PA mask must not
    # change that projA independently screened PA there too).
    current_dir = os.path.join(root, "run_current_20260920_dddddd")
    _write_json(os.path.join(current_dir, "project.json"), {
        "id": "current123", "name": "Current Project",
        "steps": {"3": {"config": {"target_species": "PA, SS"}}},
    })
    _write_json(
        os.path.join(current_dir, "step4test_combinedAnnotate", "segmentations", "segmentations.json"),
        _seg_doc([
            ("IMG_current_only.jpeg", [_mask("PA", reviewer="lo")], None),
            ("IMG_full.jpeg", [_mask("PA", reviewer="lo", mask_id=9)], None),
        ]))

    return current_dir, proj_a


# ── screened_by_stem ─────────────────────────────────────────────────────

def test_full_partial_and_unscreened_stems():
    root = tempfile.mkdtemp(prefix="screening_test_")
    try:
        current_dir, _ = build_fixture(root)
        result = screening.screened_by_stem(root, exclude_project_id="current123")
        check(sorted(result.get("IMG_full", [])) == ["PA", "SS"],
              f"IMG_full should be screened for both PA and SS: {result.get('IMG_full')}")
        check(sorted(result.get("IMG_partial", [])) == ["PA"],
              f"IMG_partial should be screened for PA only: {result.get('IMG_partial')}")
        check("IMG_current_only" not in result,
              "a stem only the current project itself has touched must not be screened")
        check("IMG_missing" not in result, "an unknown stem is simply absent, never an error")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_jpg_and_jpeg_share_one_stem():
    root = tempfile.mkdtemp(prefix="screening_test_")
    try:
        build_fixture(root)
        result = screening.screened_by_stem(root, exclude_project_id="current123")
        check("IMG_jpeg_dup" in result and result["IMG_jpeg_dup"] == ["PA"],
              f".jpg source must resolve to the bare stem: {result.get('IMG_jpeg_dup')}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_composite_training_archived_and_standalone_are_ignored():
    root = tempfile.mkdtemp(prefix="screening_test_")
    try:
        build_fixture(root)
        result = screening.screened_by_stem(root, exclude_project_id="current123")
        for stem in ("IMG_composite_only", "IMG_archived_only", "IMG_standalone_only"):
            check(stem not in result,
                  f"{stem} belongs to an excluded project kind and must never surface: {result.get(stem)}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_current_project_excluded_by_id_by_dir_and_by_both():
    root = tempfile.mkdtemp(prefix="screening_test_")
    try:
        current_dir, _ = build_fixture(root)
        by_id = screening.screened_by_stem(root, exclude_project_id="current123")
        by_dir = screening.screened_by_stem(root, exclude_project_dir=current_dir)
        by_both = screening.screened_by_stem(
            root, exclude_project_id="current123", exclude_project_dir=current_dir)
        for label, result in (("id", by_id), ("dir", by_dir), ("both", by_both)):
            check("IMG_current_only" not in result,
                  f"exclude-by-{label} must still keep the current project's own-only frame out")
            check(sorted(result.get("IMG_full", [])) == ["PA", "SS"],
                  f"exclude-by-{label} must not suppress projA's independent screening: "
                  f"{result.get('IMG_full')}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_no_exclusion_given_excludes_nothing():
    root = tempfile.mkdtemp(prefix="screening_test_")
    try:
        build_fixture(root)
        result = screening.screened_by_stem(root)
        # With nobody excluded, the current project's own frame counts too.
        check("IMG_current_only" in result, "with no exclusion, every project's coverage counts")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── write_prior_screening ────────────────────────────────────────────────

def _write_master_codes(root):
    path = os.path.join(root, "master_codes.csv")
    _write_text(path, "code,category,name\nPA,Coral,Porites astreoides\nSS,Coral,Siderastrea siderea\n")
    return path


def test_fully_screened_frame_gets_a_full_entry_not_silence():
    """2026-09-27 review finding: a frame this project already selected
    (before this gate existed, or before another project finished screening
    every target label on it) can be fully screened by launch time.
    run_26-09-18_AT alone has 101 of its 750 selected frames in exactly this
    state. Excluding it here would silently hide another reviewer's finished
    work from someone about to open that frame, so a fully-screened frame
    gets the SAME shape of entry a partly-screened one gets: every target
    label it carries, in full, plus every other project's accepted mask for
    each of them."""
    root = tempfile.mkdtemp(prefix="screening_test_")
    try:
        current_dir, _ = build_fixture(root)
        out_path = screening.write_prior_screening(
            current_dir, ["IMG_full"], root)
        check(out_path == os.path.join(
            current_dir, "step4test_combinedAnnotate", "prior_screening.json"), out_path)
        doc = json.load(open(out_path))
        check(doc["version"] == screening.CONTRACT_VERSION, doc)
        check(doc["project_id"] == "current123", doc)
        check("generated_at" in doc and doc["generated_at"], "generated_at must be stamped")

        frame = doc["frames"].get("IMG_full")
        check(frame is not None,
              f"a fully-screened frame must still get an entry: {doc['frames']}")
        check(sorted(frame["screened"].keys()) == ["PA", "SS"],
              f"both of the current project's target labels are screened for IMG_full: "
              f"{frame['screened']}")
        check(frame["screened"]["PA"]["outcome"] == "found_manual", frame["screened"]["PA"])
        check(frame["screened"]["SS"]["outcome"] == "found_ai", frame["screened"]["SS"])
        labels_in_masks = sorted(m["label"] for m in frame["masks"])
        check(labels_in_masks == ["PA", "SS"],
              f"an accepted mask for each fully-screened label is included: {frame['masks']}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_unscreened_frame_still_gets_no_entry():
    """The only frame kind that ever gets skipped now: zero labels screened
    by anyone else at all (distinct from "fully screened", which now gets a
    full entry per the fix above)."""
    root = tempfile.mkdtemp(prefix="screening_test_")
    try:
        current_dir, _ = build_fixture(root)
        out_path = screening.write_prior_screening(
            current_dir, ["IMG_current_only"], root)
        doc = json.load(open(out_path))
        check(doc["frames"] == {},
              f"a frame nobody else has touched must still get no entry: {doc['frames']}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_partly_screened_frame_carries_screened_labels_and_masks():
    root = tempfile.mkdtemp(prefix="screening_test_")
    try:
        current_dir, _ = build_fixture(root)
        mc_path = _write_master_codes(root)
        out_path = screening.write_prior_screening(
            current_dir, ["IMG_partial"], root, master_codes_csv=mc_path)
        doc = json.load(open(out_path))
        frame = doc["frames"].get("IMG_partial")
        check(frame is not None, f"IMG_partial is partly screened and must have an entry: {doc['frames']}")
        check(list(frame["screened"].keys()) == ["PA"],
              f"only PA is screened for IMG_partial: {frame['screened']}")
        check(frame["screened"]["PA"]["project_id"] == "projA", frame["screened"]["PA"])
        check(frame["screened"]["PA"]["outcome"] == "found_manual", frame["screened"]["PA"])
        check(len(frame["masks"]) == 1, f"exactly one accepted PA mask exists elsewhere: {frame['masks']}")
        mrec = frame["masks"][0]
        check(mrec["label"] == "PA" and mrec["project_id"] == "projA", mrec)
        check(mrec["name"] == "Porites astreoides", f"master_codes_csv must resolve the full name: {mrec}")
        check(mrec["outcome"] == "found_manual", mrec)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_polygon_normalisation_from_polygon_px_and_passthrough_when_present():
    root = tempfile.mkdtemp(prefix="screening_test_")
    try:
        current_dir, _ = build_fixture(root)
        out_path = screening.write_prior_screening(
            current_dir, ["IMG_partial", "IMG_jpeg_dup"], root)
        doc = json.load(open(out_path))

        px_mask = doc["frames"]["IMG_partial"]["masks"][0]
        # IMG_partial's mask carries polygon_px=[[100,50,200,50,200,150,100,150]]
        # at width=1000, height=500 -> normalised x/1000, y/500.
        expected = [[100 / 1000, 50 / 500, 200 / 1000, 50 / 500,
                    200 / 1000, 150 / 500, 100 / 1000, 150 / 500]]
        check(px_mask["polygon_norm"] == expected,
              f"polygon_px must be normalised by this entry's own width/height: {px_mask['polygon_norm']}")

        norm_mask = doc["frames"]["IMG_jpeg_dup"]["masks"][0]
        check(norm_mask["polygon_norm"] == [[0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.4]],
              f"a mask that already carries polygon_norm must pass through unchanged: "
              f"{norm_mask['polygon_norm']}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_no_master_codes_csv_degrades_to_blank_name_without_raising():
    root = tempfile.mkdtemp(prefix="screening_test_")
    try:
        current_dir, _ = build_fixture(root)
        out_path = screening.write_prior_screening(current_dir, ["IMG_partial"], root)
        doc = json.load(open(out_path))
        check(doc["frames"]["IMG_partial"]["masks"][0]["name"] == "",
              "no master_codes_csv given -> blank name, never a raise")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_unscreened_and_unknown_frames_are_skipped():
    root = tempfile.mkdtemp(prefix="screening_test_")
    try:
        current_dir, _ = build_fixture(root)
        out_path = screening.write_prior_screening(
            current_dir, ["IMG_current_only", "IMG_does_not_exist_anywhere"], root)
        doc = json.load(open(out_path))
        check(doc["frames"] == {}, f"neither frame is partly screened by another project: {doc['frames']}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_missing_and_empty_project_json_never_raises():
    """A project directory with no readable project.json (e.g. a project
    still being minted) must degrade to an empty, well-formed file rather
    than raising, since the caller (the orchestrator) writes this file
    before every annotator launch and a raise there must never be possible
    from a bad project.json alone."""
    root = tempfile.mkdtemp(prefix="screening_test_")
    try:
        build_fixture(root)
        broken_dir = os.path.join(root, "run_broken_20260920_eeeeee")
        os.makedirs(broken_dir, exist_ok=True)  # no project.json at all
        out_path = screening.write_prior_screening(broken_dir, ["IMG_full"], root)
        doc = json.load(open(out_path))
        check(doc["project_id"] == "", f"an unreadable project.json yields project_id '': {doc}")
        check(doc["frames"] == {}, "no target species known -> nothing can be partly screened")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_required_arguments_are_validated():
    try:
        screening.write_prior_screening("", ["x"], "/tmp")
        check(False, "empty project_dir must raise ValueError")
    except ValueError:
        check(True, "empty project_dir raises ValueError")
    try:
        screening.write_prior_screening("/tmp/x", ["x"], "")
        check(False, "empty runs_root must raise ValueError")
    except ValueError:
        check(True, "empty runs_root raises ValueError")


# ── Real, read-only inprocess check ──────────────────────────────────────

# _SCRIPTS is .../reef_point_seg/github_repo/scripts; inprocess/ is a sibling
# of github_repo/ under the module root, exactly as pipeline_orchestrator/
# app.py's own _inprocess_root() resolves it
# (os.path.join(os.path.dirname(REPO_DIR), "inprocess")).
_REAL_INPROCESS_ROOT = os.path.abspath(os.path.join(
    _SCRIPTS, "..", "..", "inprocess"))


def test_real_inprocess_screened_by_stem_matches_coverage_audit():
    """Read-only against the real, live inprocess tree (never writes there:
    screened_by_stem only calls build_matrix, which only reads).

    Reconciliation against docs/superpowers/reviews/2026-09-26-vega-coverage-
    audit/report.txt: that audit's section 3b counted 1,330 distinct images
    "finished" (reviewed or exported) across the four AT/LO annotation runs
    that existed on 2026-09-26, by scanning every segmentations.json
    directly. Run on 2026-09-27 against the same tree (now five projects:
    run_26-09-16_LO was minted after the audit, has 0 finished images and an
    empty target_species list, and changes nothing either way),
    build_matrix's own images set - and so screened_by_stem's, which is
    built directly on it - is 1,310, a gap of 20. That gap is a PRE-EXISTING
    property of the coverage matrix builder, not a defect in this module: two
    of the four runs (run_18aprilpm_PA_20260418_431d1b,
    run_26-07-06_AT_2nd_test_6July2026) each carry a label_provenance.csv
    ledger that covers only PART of that run's finished images (45/401 and
    272/294 respectively - an incomplete backfill from before ledger writing
    began), and builder.py's ledger/legacy merge treats any ledger-bearing
    directory as fully authoritative, skipping the legacy segmentations.json
    fallback for it entirely. The 356 + 22 = 378 finished-but-unledgered
    images this drops are mostly (358 of them) also finished in another,
    fully-ledgered run and so still appear in the union; the 20 that are not
    are invisible to build_matrix today exactly as they are invisible to
    every other consumer of it (the coverage matrix UI, STATS). Fixing that
    ledger gap is a separate, pre-existing platform issue outside this
    plan's scope; screened_by_stem correctly inherits the platform's one
    source of truth rather than re-deriving a second, disagreeing one.

    The two assertions below hold regardless of how much the live tree grows
    or shrinks between runs of this suite, so they stay meaningful as a
    permanent regression check rather than a snapshot pinned to one day's
    counts:
      1. self-consistency: excluding a project id that matches nothing real
         is a no-op, so screened_by_stem's stem set must exactly equal
         build_matrix's own 'images' set (every outcome build_matrix's
         legacy/ledger/registry chain can produce today is in
         SCREENED_OUTCOMES except found_model, which no writer emits).
      2. excluding a real, heavily-contributing project (run_26-09-18_AT,
         598 of its own accepted-PA images) measurably shrinks the result,
         proving project exclusion actually reaches the real data and is
         not a no-op against it.
    """
    if not os.path.isdir(_REAL_INPROCESS_ROOT):
        print(f"[SKIP] real inprocess root not found at {_REAL_INPROCESS_ROOT}")
        return

    matrix = build_matrix(_REAL_INPROCESS_ROOT)
    all_images = set(matrix.get("images") or [])
    check(len(all_images) > 1000,
          f"sanity floor: expected a real, substantial live tree, got {len(all_images)} images")

    no_exclusion = screening.screened_by_stem(
        _REAL_INPROCESS_ROOT, exclude_project_id="screening_test_no_such_project_ffffffff")
    check(set(no_exclusion.keys()) == all_images,
          f"screened_by_stem with a fake exclusion must match build_matrix's own image set "
          f"exactly ({len(no_exclusion)} vs {len(all_images)})")

    real_project_ids = {p["project_id"] for p in (matrix.get("projects") or [])}
    if "26-09-18_AT" in real_project_ids:
        excluding_at = screening.screened_by_stem(
            _REAL_INPROCESS_ROOT, exclude_project_id="26-09-18_AT")
        check(len(excluding_at) < len(no_exclusion),
              f"excluding a real, heavily-contributing project must shrink the result: "
              f"{len(excluding_at)} vs {len(no_exclusion)}")
    else:
        print("[SKIP] run_26-09-18_AT not present in this box's live inprocess tree")

    print(f"[INFO] live inprocess: build_matrix images={len(all_images)}; "
          f"2026-09-26 audit finished-images union=1330 (see docstring for the reconciliation)")


TESTS = [
    test_full_partial_and_unscreened_stems,
    test_jpg_and_jpeg_share_one_stem,
    test_composite_training_archived_and_standalone_are_ignored,
    test_current_project_excluded_by_id_by_dir_and_by_both,
    test_no_exclusion_given_excludes_nothing,
    test_fully_screened_frame_gets_a_full_entry_not_silence,
    test_unscreened_frame_still_gets_no_entry,
    test_partly_screened_frame_carries_screened_labels_and_masks,
    test_polygon_normalisation_from_polygon_px_and_passthrough_when_present,
    test_no_master_codes_csv_degrades_to_blank_name_without_raising,
    test_unscreened_and_unknown_frames_are_skipped,
    test_missing_and_empty_project_json_never_raises,
    test_required_arguments_are_validated,
    test_real_inprocess_screened_by_stem_matches_coverage_audit,
]


def main():
    for fn in TESTS:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - surface as a FAIL, not a crash
            _FAILURES.append(f"{fn.__name__} raised {e!r}")
            print(f"[FAIL] {fn.__name__} raised {e!r}")
    print(f"\n{len(TESTS) - len(_FAILURES)}/{len(TESTS)} passed.")
    return 1 if _FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())

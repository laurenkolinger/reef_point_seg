#!/usr/bin/env python
"""Mint the two ready-made projects the VEGA guided tutorial starts from.

Module:  scripts/_tutorial/build_tutorial_projects.py
Purpose: Give a student one project per track, already pointed at the right
         tutorial fixture and already parked on Train, so the tutorial opens on
         the work rather than on setting the work up. One project reads the
         staged bounding-box dataset tutorial_boxes (the detection track) and
         one reads the training set of outlines tutorial_outlines (the
         segmentation track). Both are minted by the platform's own minter,
         _reefreview.training_projects.mint_training_project, so a tutorial
         project is the same object as any other training project and nothing
         has to special-case it.
Inputs:  --inprocess (the module's run folder root), --track (boxes, outlines
         or both), --by (the initials recorded as the maker), --rebuild
         (remove the tutorial project of each chosen track and mint it fresh),
         --quiet.
Outputs: <inprocess>/run_{YY-MM-DD}_{INITIALS}_train_tutorial_boxes/ and
         <inprocess>/run_{YY-MM-DD}_{INITIALS}_train_tutorial_outlines/, each
         holding project.json, README.md, the four skipped step folders, the
         marker file tutorial_project.json, and (the outlines project only) the
         relative step4test_combinedAnnotate symlink to its training set. The
         function returns one record per project.

Why a student gets a ready-made project. The tour walks a real surface and a
real project, and the first thing a student would otherwise do is invent a
project name and pick a training source from a list of six. That is a decision
about someone else's survey, made before the student knows what a training
source is. The two projects here remove it, and a student who wants to make
their own afterwards has watched one being used first.

What --rebuild will and will not remove. A student who wrecks a tutorial
project gets a clean one from --rebuild, and nothing else on the box moves.
Before a folder is removed it has to pass every one of these: it sits directly
inside the run folder root, it is a real directory rather than a link, its name
starts with run_, it carries this tool's own marker file naming the same
fixture, and its project.json, when it can be read at all, declares the training
project type that fixture mints and records that fixture as its source. A folder
failing any one of them is left exactly where it is and the run stops with a
message saying which check failed. Real annotation projects carry no marker
file, so none of them can reach the removal at all.

An unreadable or missing project.json does not block the removal, because a
student whose project.json was wrecked is exactly the student --rebuild exists
for. The marker is what authorises a removal; a readable project.json that
disagrees with it is what vetoes one.

The fixtures themselves are never touched. The outlines project reaches its
training set through a symlink and the boxes project through a recorded path,
so removing a tutorial project removes a run folder and nothing else. The
tests prove that by counting the fixture's frames after a rebuild.

Both projects are training projects, which means the coverage matrix and STATS
skip them the way they skip every training project
(docs/adr/0001-non-tcrmp-imagery-never-enters-the-stores.md). The tutorial
therefore adds nothing to the numbers the TCRMP program reports about its own
survey, no matter how many students run it.

Usage:
  env/bin/python scripts/_tutorial/build_tutorial_projects.py
  env/bin/python scripts/_tutorial/build_tutorial_projects.py --rebuild --track outlines
"""

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import textwrap
from collections import Counter, namedtuple
from datetime import datetime, timedelta, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SCRIPTS)
_MODULE_ROOT = os.path.dirname(_REPO)

if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# The single place a training project's run folder is written, for either kind
# of source. Imported, never reimplemented: a hand-written project.json would
# drift from the shape the orchestrator, the launcher and Vesper.llm read.
from _reefreview.training_projects import (  # noqa: E402
    BBOX_PROJECT_TYPE,
    COMPOSITE_PROJECT_TYPE,
    RunDirExhausted,
    TRAINING_LINK_DIRNAME,
    mint_training_project,
)
# The composite root and its cross-process lock, so the back link this tool
# appends to a training set's composite.json cannot race the matrix's own.
from _matrix.composite import composite_lock  # noqa: E402
from _boxes import ingest_yolo_boxes as ing  # noqa: E402
import build_tutorial_outlines as outlines  # noqa: E402

AST = timezone(timedelta(hours=-4))


# ── What this tool makes ─────────────────────────────────────────────────

# One record per tutorial track. `key` is what --track takes, `kind` is the
# source kind the minter understands, `name` is the fixture folder, `root` says
# where that folder lives (None means the training sets under _composite/), and
# `display_name` is what the launcher card shows under the run folder name.
#
# The display names are the tour's own two track names, so a student choosing
# "Outlines (segmentation)" on the tour's first screen opens the project that
# says the same words. Both start with TUTORIAL, so the two sort next to each
# other in the launcher's project list and apart from a survey project, and
# the run folder names do the same thing on disk: every tutorial folder is
# run_<date>_<initials>_train_tutorial_<track>.
Fixture = namedtuple("Fixture", "key kind name root display_name task_default track")

BBOX_ROOT = os.path.join(_REPO, "supporting_data", "bbox_datasets")
COMPOSITE_ROOT_NAME = outlines.COMPOSITE_DIRNAME
DEFAULT_INPROCESS = os.path.join(_MODULE_ROOT, "inprocess")

FIXTURES = (
    Fixture(key="boxes", kind="bbox", name="tutorial_boxes", root=BBOX_ROOT,
            display_name="TUTORIAL boxes (detection)", task_default="detect",
            track="Boxes (detection)"),
    Fixture(key="outlines", kind="composite", name=outlines.DEFAULT_NAME,
            root=None, display_name="TUTORIAL outlines (segmentation)",
            task_default=None, track="Outlines (segmentation)"),
)

FIXTURE_KEYS = tuple(f.key for f in FIXTURES)
TRACK_CHOICES = FIXTURE_KEYS + ("both",)

# The initials recorded as the maker of a tutorial project. Agent work is
# signed LO across every VICARIUS log, ledger and project.
DEFAULT_INITIALS = "LO"

# The marker this tool writes inside every project it mints, and the only file
# --rebuild trusts. Its name is deliberately not project.json: the marker says
# who made this folder and which fixture it belongs to, and project.json says
# what the folder is, and conflating the two would let an edit to one quietly
# authorise a removal.
MARKER_NAME = "tutorial_project.json"
MARKER_SCHEMA_VERSION = 1
MARKER_TOOL = os.path.join("scripts", "_tutorial", os.path.basename(__file__))

# The project.json type each source kind mints, so a marker cannot authorise
# the removal of a folder whose project.json says something else.
PROJECT_TYPE_FOR_KIND = {"bbox": BBOX_PROJECT_TYPE, "composite": COMPOSITE_PROJECT_TYPE}

# How wide the README paragraphs this tool appends are wrapped, the same
# width the fixture READMEs use.
README_WIDTH = outlines.README_WIDTH

# The step the tutorial projects open on, mirrored from the minter so a report
# can say it without reading project.json twice.
TRAIN_STEP = 6


class TutorialProjectError(RuntimeError):
    """Anything that stops the run before it mints or removes a folder."""


def ast_now():
    """'YYYY-MM-DD HH:MM:SS' in Atlantic Standard Time, 24 hour."""
    return datetime.now(AST).strftime("%Y-%m-%d %H:%M:%S")


def fixture_path(fixture, inprocess_root):
    """Where one fixture's folder is.

    Parameters:
      fixture: a Fixture.
      inprocess_root: the module's run folder root, which also holds the
        training sets under _composite/.
    Returns:
      an absolute path.
    """
    root = fixture.root or os.path.join(inprocess_root, COMPOSITE_ROOT_NAME)
    return os.path.abspath(os.path.join(root, fixture.name))


def chosen_fixtures(track):
    """The fixtures --track names.

    Parameters:
      track: 'boxes', 'outlines' or 'both'.
    Returns:
      a tuple of Fixture.
    Raises:
      TutorialProjectError naming the valid choices.
    """
    if track == "both":
        return FIXTURES
    for fixture in FIXTURES:
        if fixture.key == track:
            return (fixture,)
    raise TutorialProjectError(
        f"--track takes one of {', '.join(TRACK_CHOICES)}, got {track!r}")


# ── Checking a fixture before anything is minted ─────────────────────────

def check_fixture(fixture, inprocess_root):
    """Refuse a fixture that cannot be trained on, before a folder is made.

    The minter documents that the caller owns the source-specific guards,
    because what makes a source usable differs by kind. These are the same
    guards the coverage matrix and the Bounding Boxes page apply to their own
    sources.

    Parameters:
      fixture: a Fixture.
      inprocess_root: the module's run folder root.
    Returns:
      the fixture's absolute path.
    Raises:
      TutorialProjectError naming the fixture, what is missing, and the tool
      that builds it.
    """
    path = fixture_path(fixture, inprocess_root)
    builder = ("scripts/_tutorial/build_tutorial_set.py" if fixture.kind == "bbox"
               else "scripts/_tutorial/build_tutorial_outlines.py")
    if not os.path.isdir(path):
        raise TutorialProjectError(
            f"the tutorial fixture {fixture.name} is not at {path}; build it "
            f"with {builder} before minting a project from it")
    if fixture.kind == "composite":
        root = os.path.realpath(os.path.join(inprocess_root, COMPOSITE_ROOT_NAME))
        if os.path.dirname(os.path.realpath(path)) != root:
            raise TutorialProjectError(
                f"{path} does not resolve to a folder directly inside {root}; a "
                "training source is read where training sets live, never through "
                "a link out of that folder")
        if not os.path.isfile(os.path.join(path, outlines.DATA_YAML_NAME)):
            raise TutorialProjectError(
                f"{path} has no {outlines.DATA_YAML_NAME}, so the Train step "
                f"cannot read it; rebuild it with {builder}")
        images = os.path.join(path, outlines.IMAGES_DIR)
        if not os.path.isdir(images) or not os.listdir(images):
            raise TutorialProjectError(
                f"{path} holds no frames under {outlines.IMAGES_DIR}/, so a "
                f"project made from it would train on nothing; rebuild it with "
                f"{builder}")
        return path
    if not os.path.realpath(path).startswith(os.path.realpath(BBOX_ROOT) + os.sep):
        raise TutorialProjectError(
            f"{path} does not resolve to a folder inside {BBOX_ROOT}; a staged "
            "dataset is read where it is staged")
    pairs, unlabelled = _load_detect_dataset().bbox_pairs(path)
    if not pairs:
        raise TutorialProjectError(
            f"{path} holds no picture with a label file beside it "
            f"({unlabelled} picture(s) with none), so a project made from it "
            f"would train on nothing; rebuild it with {builder}")
    return path


def _load_detect_dataset():
    """The detection dataset builder, loaded from its own folder by path.

    scripts/TCRMPtrain_oceankindCV/src/ is not an importable package, so the
    module is loaded by file path. Its bbox_pairs is the same pairing check
    the Train step makes, imported rather than copied so a fixture this tool
    accepts is a fixture Train accepts.

    Returns:
      the loaded module.
    Raises:
      TutorialProjectError naming the file when it cannot be loaded.
    """
    path = os.path.join(_SCRIPTS, "TCRMPtrain_oceankindCV", "src", "detect_dataset.py")
    try:
        spec = importlib.util.spec_from_file_location("_tutorial_detect_dataset", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except (OSError, ImportError, AttributeError, ValueError) as exc:
        raise TutorialProjectError(
            f"the detection dataset builder at {path} could not be loaded "
            f"({exc}); this tool reads its pairing check so a fixture it accepts "
            "is a fixture the Train step accepts")


# ── The marker, and what it authorises ───────────────────────────────────

def write_marker(run_dir, fixture, source_path, by):
    """Write the marker that says this folder is a tutorial project.

    Parameters:
      run_dir: the minted run folder.
      fixture: the Fixture it was minted from.
      source_path: where that fixture lives.
      by: the initials recorded as the maker.
    Returns:
      the marker document written.
    """
    doc = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "tool": MARKER_TOOL,
        "fixture": fixture.name,
        "fixture_kind": fixture.kind,
        "fixture_path": os.path.abspath(source_path),
        "track": fixture.track,
        "minted_at": ast_now(),
        "by": by,
        "note": ("This folder is a VEGA tutorial project. "
                 f"{MARKER_TOOL} --rebuild removes and remakes a folder "
                 "carrying this file and no other."),
    }
    _write_json_atomic(os.path.join(run_dir, MARKER_NAME), doc)
    return doc


def read_marker(run_dir):
    """The marker inside one run folder, or None when there is none.

    An unreadable marker returns None rather than raising, so a corrupt file
    can never authorise a removal.

    Parameters:
      run_dir: the folder to read.
    Returns:
      the marker document, or None.
    """
    try:
        with open(os.path.join(run_dir, MARKER_NAME), encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def find_tutorial_projects(inprocess_root, fixture):
    """Every run folder this tool minted for one fixture, oldest name first.

    Parameters:
      inprocess_root: the module's run folder root.
      fixture: the Fixture.
    Returns:
      a sorted list of absolute run folder paths.
    """
    found = []
    try:
        names = sorted(os.listdir(inprocess_root))
    except OSError:
        return found
    for name in names:
        run_dir = os.path.join(inprocess_root, name)
        if not name.startswith("run_") or not os.path.isdir(run_dir):
            continue
        marker = read_marker(run_dir)
        if marker and marker.get("tool") == MARKER_TOOL \
                and marker.get("fixture") == fixture.name:
            found.append(run_dir)
    return found


def check_removable(inprocess_root, run_dir, fixture):
    """Refuse to remove anything that is not a tutorial project of this fixture.

    Every check has to pass. A folder failing any one of them is left exactly
    where it is, because the alternative is a flag that deletes a reviewer's
    annotation work. The marker is what authorises a removal; a readable
    project.json that disagrees with it is what vetoes one, and an unreadable
    project.json neither authorises nor vetoes.

    Parameters:
      inprocess_root: the module's run folder root.
      run_dir: the folder --rebuild would remove.
      fixture: the Fixture it is supposed to belong to.
    Raises:
      TutorialProjectError naming the check that failed and saying nothing was
      removed.
    """
    tail = " Nothing was removed."
    real_root = os.path.realpath(inprocess_root)
    if os.path.islink(run_dir):
        raise TutorialProjectError(
            f"{run_dir} is a link, not a run folder, and this tool removes only "
            "a real folder it minted." + tail)
    if not os.path.isdir(run_dir):
        raise TutorialProjectError(f"{run_dir} is not a folder." + tail)
    if os.path.dirname(os.path.realpath(run_dir)) != real_root:
        raise TutorialProjectError(
            f"{run_dir} does not sit directly inside {real_root}, so it is not "
            "one of this module's run folders." + tail)
    if not os.path.basename(run_dir).startswith("run_"):
        raise TutorialProjectError(
            f"{run_dir} is not named like a run folder (a run folder's name "
            "starts with run_)." + tail)

    marker = read_marker(run_dir)
    if marker is None:
        raise TutorialProjectError(
            f"{run_dir} carries no readable {MARKER_NAME}, so it was not minted "
            "by this tool. A project holding annotation work never carries "
            "one." + tail)
    if marker.get("tool") != MARKER_TOOL:
        raise TutorialProjectError(
            f"{run_dir} carries a {MARKER_NAME} written by "
            f"{marker.get('tool')!r}, not by {MARKER_TOOL}." + tail)
    if marker.get("fixture") != fixture.name:
        raise TutorialProjectError(
            f"{run_dir} is a tutorial project of {marker.get('fixture')!r}, and "
            f"this rebuild is for {fixture.name!r}." + tail)

    # A student whose project.json was wrecked is exactly the student --rebuild
    # exists for, so an unreadable or absent project.json does not block the
    # removal: the marker already said this folder is a tutorial project of
    # this fixture, and a project holding annotation work carries no marker. A
    # project.json that IS readable and disagrees still refuses, because that
    # means the folder was repurposed into something else.
    meta = _read_project_json(run_dir)
    if meta is None:
        return
    wanted_type = PROJECT_TYPE_FOR_KIND[fixture.kind]
    if meta.get("project_type") != wanted_type:
        raise TutorialProjectError(
            f"{run_dir} declares project_type {meta.get('project_type')!r} and a "
            f"tutorial project of {fixture.name} declares {wanted_type!r}." + tail)
    recorded = _recorded_source_name(meta)
    if recorded != fixture.name:
        raise TutorialProjectError(
            f"{run_dir} records its training source as {recorded!r}, not "
            f"{fixture.name!r}." + tail)


def _recorded_source_name(meta):
    """The training source name a training project's project.json records.

    Parameters:
      meta: the project.json document.
    Returns:
      the name, or '' when the document records none.
    """
    for key in ("composite", "training_source"):
        block = meta.get(key)
        if isinstance(block, dict) and block.get("name"):
            return str(block["name"])
    return ""


def _read_project_json(run_dir):
    """One run folder's project.json, or None when it cannot be read."""
    try:
        with open(os.path.join(run_dir, "project.json"), encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def remove_tutorial_project(inprocess_root, run_dir, fixture, log):
    """Remove one tutorial project, after every guard has passed.

    shutil.rmtree unlinks a symlinked folder rather than descending into it,
    so the outlines project's step4test_combinedAnnotate link is removed and
    the training set it points at is untouched. The tests count the fixture's
    frames afterwards to prove it.

    Parameters:
      inprocess_root: the module's run folder root.
      run_dir: the folder to remove.
      fixture: the Fixture it belongs to.
      log: where the line goes.
    Raises:
      TutorialProjectError from check_removable, or naming the failure when
      the removal itself fails.
    """
    check_removable(inprocess_root, run_dir, fixture)
    try:
        shutil.rmtree(run_dir)
    except OSError as exc:
        raise TutorialProjectError(f"{run_dir} could not be removed ({exc})")
    if os.path.exists(run_dir):
        raise TutorialProjectError(f"{run_dir} is still on disk after the removal")
    log(f"removed the old tutorial project at {run_dir}")


# ── Minting ──────────────────────────────────────────────────────────────

def mint(inprocess_root, fixture, by, log):
    """Mint one tutorial project from one fixture.

    Parameters:
      inprocess_root: the module's run folder root.
      fixture: the Fixture.
      by: the initials recorded as the maker.
      log: where the lines go.
    Returns:
      {'run_dir', 'run_dir_name', 'project_id', 'display_name', 'fixture',
       'source_path'}.
    Raises:
      TutorialProjectError naming the failure.
    """
    source_path = check_fixture(fixture, inprocess_root)
    extra = {"display_name": fixture.display_name}
    if fixture.kind == "bbox":
        extra["task_default"] = fixture.task_default
    else:
        extra["path_label"] = os.path.join("inprocess", COMPOSITE_ROOT_NAME,
                                           fixture.name)
        extra["source_projects"] = _source_projects(source_path)
    try:
        minted = mint_training_project(inprocess_root, fixture.kind, fixture.name,
                                       source_path, by, extra=extra)
    except RunDirExhausted as exc:
        raise TutorialProjectError(
            f"no free run folder for {fixture.name} today ({exc}); archive one "
            "and run this again")
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise TutorialProjectError(
            f"the tutorial project for {fixture.name} could not be minted ({exc})")

    write_marker(minted["run_dir"], fixture, source_path, by)
    append_readme_note(minted["run_dir"], fixture, minted["run_dir_name"])
    if fixture.kind == "composite":
        append_linked_project(inprocess_root, source_path, minted["run_dir_name"])
    log(f"minted {minted['run_dir_name']} for {fixture.name}, opening on Train")
    record = dict(minted)
    record["fixture"] = fixture
    record["source_path"] = source_path
    return record


def _source_projects(composite_dir):
    """[{'project_id', 'n_masks'}] a training set records about its sources.

    Parameters:
      composite_dir: the training set folder.
    Returns:
      the list from its composite.json, or [] when it records none. The
      project's README names these, so a student can see the outlines came
      from reviewed survey work rather than from nowhere.
    """
    try:
        with open(os.path.join(composite_dir, outlines.COMPOSITE_JSON_NAME),
                  encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, UnicodeDecodeError, ValueError):
        return []
    listed = (doc or {}).get("source_projects")
    return listed if isinstance(listed, list) else []


def append_linked_project(inprocess_root, composite_dir, run_name):
    """Record the new project in its training set's composite.json.

    The coverage matrix names a training set's downstream projects from this
    list; nothing scans for them. The read, the append and the write run under
    the matrix's own cross-process lock and finish with an atomic replace, so
    this cannot drop an entry the matrix wrote a moment earlier.

    Parameters:
      inprocess_root: the module's run folder root.
      composite_dir: the training set folder.
      run_name: the new run folder's basename.
    Returns:
      True when the file was updated, False when composite.json is unreadable
      (the project is minted either way; only the back link is missing).
    """
    path = os.path.join(composite_dir, outlines.COMPOSITE_JSON_NAME)
    with composite_lock(inprocess_root):
        try:
            with open(path, encoding="utf-8") as fh:
                meta = json.load(fh)
            if not isinstance(meta, dict):
                return False
        except (OSError, UnicodeDecodeError, ValueError):
            return False
        linked = list(meta.get("linked_projects") or [])
        if run_name not in linked:
            linked.append(run_name)
        meta["linked_projects"] = linked
        _write_json_atomic(path, meta)
    return True


def append_readme_note(run_dir, fixture, run_name):
    """Add the tutorial paragraph to the README the minter wrote.

    Appended rather than written over, so the minter's own account of where
    the training data lives survives and this only says what the folder is for.

    Parameters:
      run_dir: the minted run folder.
      fixture: the Fixture.
      run_name: the run folder's basename.
    """
    lines = [
        "",
        "## This is a tutorial project",
        "",
        _fill(
            f"{run_name} was made by {MARKER_TOOL} for the VEGA guided "
            f"tutorial's {fixture.track} track. It is a working project: a "
            "student trains in it, evaluates in it and runs inference in it, "
            "and everything it produces is written inside this folder."),
        "",
        _fill(
            f"The fixture {fixture.name} is read only. Nothing a student does "
            "here changes it, so the next student starts from the same frames "
            "and the same held-out split."),
        "",
        _fill(
            f"To start over with a clean project, run {MARKER_TOOL} --rebuild "
            f"--track {fixture.key}. It removes this folder and mints a new "
            f"one, and it refuses to remove any folder that does not carry "
            f"{MARKER_NAME}."),
        "",
        "Lauren Olinger, lauren.olinger@uvi.edu",
        "",
    ]
    path = os.path.join(run_dir, "README.md")
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
    except OSError as exc:
        raise TutorialProjectError(f"{path} could not be added to ({exc})")


def _fill(text):
    """Wrap one paragraph to README_WIDTH."""
    return textwrap.fill(text, width=README_WIDTH)


def _write_json_atomic(path, doc):
    """Write JSON so a concurrent reader sees either the old file or the whole
    new one: a temp file in the same folder, then os.replace.

    Raises:
      TutorialProjectError naming the file when the write fails.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    try:
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tutorial-", suffix=".json.tmp")
    except OSError as exc:
        raise TutorialProjectError(f"{path} could not be written ({exc})")
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


# ── The run ──────────────────────────────────────────────────────────────

def build_tutorial_projects(inprocess_root, fixtures, *, by=DEFAULT_INITIALS,
                            rebuild=False, log=print):
    """Make sure each fixture has one tutorial project, and return them.

    Without --rebuild this is idempotent: a fixture that already has a
    tutorial project keeps it and nothing is written. With --rebuild every
    tutorial project of each chosen fixture is removed first, after the guards
    pass, and one fresh project is minted.

    Parameters:
      inprocess_root: the module's run folder root.
      fixtures: the Fixtures to cover.
      by: the initials recorded as the maker.
      rebuild: remove and remake rather than keep.
      log: where the lines go.
    Returns:
      a list of records, one per fixture, each with 'run_dir', 'fixture',
      'source_path' and 'action' ('minted', 'rebuilt' or 'kept').
    Raises:
      TutorialProjectError naming the failure.
    """
    if not os.path.isdir(inprocess_root):
        raise TutorialProjectError(
            f"{inprocess_root} is not a folder; --inprocess takes the module's "
            "run folder root")
    records = []
    for fixture in fixtures:
        existing = find_tutorial_projects(inprocess_root, fixture)
        if existing and not rebuild:
            log(f"{fixture.name} already has a tutorial project at "
                f"{existing[0]}; left as it is")
            record = _describe_existing(existing[0], fixture, inprocess_root)
            record["action"] = "kept"
            records.append(record)
            continue
        for run_dir in existing:
            remove_tutorial_project(inprocess_root, run_dir, fixture, log)
        record = mint(inprocess_root, fixture, by, log)
        record["action"] = "rebuilt" if existing else "minted"
        records.append(record)
    log(f"Step 1 complete: {len(records)} tutorial project(s) in place")
    return records


def _describe_existing(run_dir, fixture, inprocess_root):
    """The record shape mint returns, for a project that was already there."""
    meta = _read_project_json(run_dir) or {}
    return {
        "run_dir": run_dir,
        "run_dir_name": os.path.basename(run_dir),
        "project_id": str(meta.get("id") or ""),
        "display_name": str(meta.get("name") or ""),
        "fixture": fixture,
        "source_path": fixture_path(fixture, inprocess_root),
    }


# ── Verification ─────────────────────────────────────────────────────────

def describe_fixture(fixture, source_path):
    """Count what one fixture holds, read off disk.

    Parameters:
      fixture: the Fixture.
      source_path: its folder.
    Returns:
      {'n_images', 'n_labels', 'n_pairs', 'n_annotations', 'by_split',
       'by_class_split', 'classes'} where by_class_split counts pictures, one
      per class the picture carries, keyed by (class token, split).
    Raises:
      TutorialProjectError when the fixture cannot be read.
    """
    if fixture.kind == "composite":
        return _describe_composite(source_path)
    return _describe_bbox(source_path)


def _describe_composite(path):
    """Counts for a training set of outlines."""
    names = outlines.read_parent_classes(path)
    pins = _read_pins(path)
    images = sorted(name for name in os.listdir(os.path.join(path, outlines.IMAGES_DIR))
                    if os.path.splitext(name)[1].lower() in ing.IMAGE_EXTS)
    labels = sorted(name for name in os.listdir(os.path.join(path, outlines.LABELS_DIR))
                    if name.endswith(outlines.LABEL_EXT))
    stems = {os.path.splitext(n)[0] for n in images}
    pairs = sorted(stems & {n[:-len(outlines.LABEL_EXT)] for n in labels})
    counts = outlines.count_set(path, names, {n: pins[n] for n in images if n in pins})
    return {
        "n_images": len(images), "n_labels": len(labels), "n_pairs": len(pairs),
        "n_annotations": counts["n_masks"],
        "by_split": dict(Counter(pins[n] for n in images if n in pins)),
        "by_class_split": dict(counts["by_label_split"]),
        "classes": [names[c] for c in sorted(names)],
        "n_negatives": len(counts["negatives"]),
    }


def _describe_bbox(path):
    """Counts for a staged bounding-box dataset."""
    detect = _load_detect_dataset()
    pairs, _unlabelled = detect.bbox_pairs(path)
    pins = _read_pins(path)
    classes = [line.strip() for line
               in _read_text(os.path.join(path, "classes.txt")).splitlines()
               if line.strip()]
    images = sorted(name for name in os.listdir(os.path.join(path, "images"))
                    if os.path.splitext(name)[1].lower() in ing.IMAGE_EXTS)
    label_dir = detect.bbox_label_dir(path)
    labels = sorted(name for name in os.listdir(label_dir) if name.endswith(".txt"))
    by_class_split, n_boxes, n_negatives = Counter(), 0, 0
    for name, _image_path, label_path in pairs:
        split = pins.get(name, "")
        boxes = ing.parse_boxes(_read_text(label_path), len(classes),
                                         os.path.basename(label_path))
        n_boxes += len(boxes)
        if not boxes:
            n_negatives += 1
        for class_id in {c for c, _coords in boxes}:
            by_class_split[(classes[class_id], split)] += 1
    return {
        "n_images": len(images), "n_labels": len(labels), "n_pairs": len(pairs),
        "n_annotations": n_boxes,
        "by_split": dict(Counter(pins[n] for n in images if n in pins)),
        "by_class_split": dict(by_class_split), "classes": classes,
        "n_negatives": n_negatives,
    }


def _read_pins(path):
    """{picture: split} from a fixture's split_pins.json; {} when it has none."""
    try:
        with open(os.path.join(path, outlines.SPLIT_PINS_NAME), encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, UnicodeDecodeError, ValueError):
        return {}
    pins = (doc or {}).get("pins")
    return pins if isinstance(pins, dict) else {}


def _read_text(path):
    """One text file, naming it in the error when it cannot be read."""
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise TutorialProjectError(f"{path} could not be read ({exc})")


def verify(inprocess_root, records, log):
    """Read every minted project back off disk and check it is what it claims.

    Parameters:
      inprocess_root: the module's run folder root.
      records: what build_tutorial_projects returned.
      log: where the counts go.
    Returns:
      a list of {'record', 'project_type', 'source_name', 'source_path',
      'counts'}.
    Raises:
      TutorialProjectError naming the disagreement.
    """
    checked = []
    for record in records:
        fixture = record["fixture"]
        run_dir = record["run_dir"]
        meta = _read_project_json(run_dir)
        if meta is None:
            raise TutorialProjectError(f"{run_dir} has no readable project.json")
        wanted = PROJECT_TYPE_FOR_KIND[fixture.kind]
        if meta.get("project_type") != wanted:
            raise TutorialProjectError(
                f"{run_dir} declares project_type {meta.get('project_type')!r}, "
                f"and a project made from {fixture.name} declares {wanted!r}")
        if _recorded_source_name(meta) != fixture.name:
            raise TutorialProjectError(
                f"{run_dir} records its training source as "
                f"{_recorded_source_name(meta)!r}, not {fixture.name!r}")
        if meta.get("current_step") != TRAIN_STEP:
            raise TutorialProjectError(
                f"{run_dir} opens on step {meta.get('current_step')!r} and a "
                f"training project opens on {TRAIN_STEP} (Train)")
        if read_marker(run_dir) is None:
            raise TutorialProjectError(f"{run_dir} carries no readable {MARKER_NAME}")

        if fixture.kind == "composite":
            link = os.path.join(run_dir, TRAINING_LINK_DIRNAME)
            if not os.path.islink(link):
                raise TutorialProjectError(
                    f"{link} is not a symlink, and the Train step reads the "
                    "training set through it")
            if os.path.realpath(link) != os.path.realpath(record["source_path"]):
                raise TutorialProjectError(
                    f"{link} points at {os.path.realpath(link)} and the fixture "
                    f"is at {os.path.realpath(record['source_path'])}")
        else:
            recorded = str((meta.get("training_source") or {}).get("path") or "")
            if os.path.realpath(recorded) != os.path.realpath(record["source_path"]):
                raise TutorialProjectError(
                    f"{run_dir} records its dataset at {recorded} and the "
                    f"fixture is at {record['source_path']}")
            for entry in os.listdir(run_dir):
                if os.path.islink(os.path.join(run_dir, entry)):
                    raise TutorialProjectError(
                        f"{run_dir} holds the link {entry}; a detection project "
                        "copies and links no imagery at all")

        counts = describe_fixture(fixture, record["source_path"])
        checked.append({"record": record, "project_type": meta["project_type"],
                        "source_name": fixture.name,
                        "source_path": record["source_path"], "counts": counts})
        log(f"verified {os.path.basename(run_dir)}: {meta['project_type']}, "
            f"source {fixture.name}, opens on Train")
    log("Step 2 complete: every tutorial project on disk is what it claims")
    return checked


def report(checked, log):
    """Print the counts a reader needs to trust the fixtures.

    Parameters:
      checked: what verify returned.
      log: where the lines go.
    """
    for item in checked:
        counts = item["counts"]
        record = item["record"]
        log("")
        log(f"{record['run_dir_name']}  ({record['action']})")
        log(f"  display name    {record['display_name']}")
        log(f"  project type    {item['project_type']}")
        log(f"  training source {item['source_name']} at {item['source_path']}")
        log(f"  pictures        {counts['n_images']}")
        log(f"  label files     {counts['n_labels']}")
        log(f"  pairs matched   {counts['n_pairs']}")
        log(f"  annotations     {counts['n_annotations']} "
            f"({counts['n_negatives']} picture(s) with none)")
        log("  split sizes     " + ", ".join(
            f"{split} {counts['by_split'].get(split, 0)}"
            for split in ("train", "valid", "test")))
        for token in counts["classes"]:
            row = ", ".join(
                f"{split} {counts['by_class_split'].get((token, split), 0)}"
                for split in ("train", "valid", "test"))
            log(f"    {token:<28} {row}")


# ── Command line ─────────────────────────────────────────────────────────

def build_parser():
    """The command-line parser, split out so the flags read in one place."""
    parser = argparse.ArgumentParser(
        description="Mint the two ready-made projects the VEGA guided tutorial "
                    "starts from.")
    parser.add_argument(
        "--inprocess", default=DEFAULT_INPROCESS,
        help=f"the module's run folder root (default: {DEFAULT_INPROCESS})")
    parser.add_argument(
        "--track", default="both", choices=list(TRACK_CHOICES),
        help="which tutorial track to make a project for (default: both)")
    parser.add_argument(
        "--by", default=DEFAULT_INITIALS,
        help=f"the initials recorded as the maker (default: {DEFAULT_INITIALS})")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="remove the tutorial project of each chosen track and mint it "
             "fresh; refuses to remove any folder without this tool's marker file")
    parser.add_argument("--quiet", action="store_true",
                        help="print only the refusals and the final line")
    return parser


def main(argv=None):
    """Parse, validate, mint, and verify.

    Parameters:
      argv: the command line, or None for sys.argv[1:].
    Returns:
      0 when every chosen project is in place, 1 when the run was refused.
    """
    args = build_parser().parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else print
    try:
        fixtures = chosen_fixtures(args.track)
        inprocess_root = os.path.abspath(args.inprocess)
        records = build_tutorial_projects(inprocess_root, fixtures, by=args.by,
                                          rebuild=args.rebuild, log=log)
        checked = verify(inprocess_root, records, log)
        report(checked, log)
    except TutorialProjectError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print("; ".join(f"{r['run_dir_name']} {r['action']}" for r in records))
    return 0


if __name__ == "__main__":
    sys.exit(main())

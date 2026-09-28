"""
Flask application for the Reef Point Seg Orchestrator.
"""

import os
import re
import sys
import csv
import json
import glob
import time
import shutil
import threading
import traceback
from datetime import datetime, timezone, timedelta
from urllib.parse import unquote, urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError

from flask import (
    Flask, render_template, render_template_string, request, jsonify,
    send_from_directory, send_file,
)

import lock_status
import project_manager as pm
import host_open
import remap_loader
import vicarius_bridge as vic
from stage_runner import StageRunner, busy_result, find_free_port
from orchestrator_config import (
    REPO_DIR, PYTHON_PATHS, ENTRY_POINTS, WORKING_DIRS, STAGE_PORTS,
    PRESET_DIRS, PATHS, CONFIG_PATH, ORCHESTRATOR_PORT, DEFAULT_STEP_CONFIGS,
)

# Atlantic Standard Time, fixed UTC-4, no daylight saving. Byte-identical
# constant to TCRMPclip_combinedAnnotate/src/provenance.py:AST and
# TCRMPtrain_oceankindCV/src/evaluate_run.py's own copy.
_AST = timezone(timedelta(hours=-4))


def _ast_now():
    """AST ISO-8601 timestamp, second precision, e.g. 2026-06-25T08:13:42-04:00."""
    return datetime.now(_AST).isoformat(timespec="seconds")


def _ast_stamp(epoch):
    """Format a file time as an Atlantic Standard Time stamp.

    Parameters:
        epoch: seconds since the epoch, as os.path.getmtime returns.
    Returns:
        'YYYY-MM-DD HH:MM' in AST, the house form for a folder's time.
    """
    return datetime.fromtimestamp(epoch, tz=_AST).strftime("%Y-%m-%d %H:%M")

# scripts/ on path for the shared Add-Expert-IDs blueprint (mounted natively
# below — no subprocess, no iframe).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _expertids import make_blueprint as _make_expertids_bp
# Cross-project Label Coverage Matrix — also mounted natively (its own full
# page under /matrix, no iframe/subprocess), same as the expert-IDs blueprint.
from _matrix import make_blueprint as _make_matrix_bp
# Label Manager — canonical species vocabulary editor + remap composer + manual
# recode, mounted natively as its own full page under /labels.
from _labels import make_blueprint as _make_labels_bp
# The cross-run library: every trained run the module can see, the promoted
# champion, and whether two runs' scores may be ranked against each other.
# Library code only; the Train + Infer page that once mounted beside it is
# retired and its capabilities live in the main window's step panels.
from _modellab import labruns as _labruns
# The one list of project.json project_type values that mean "this run dir
# trains from a prepared source and holds no annotation work of its own", and
# the module that mints both kinds.
from _reefreview.training_projects import (
    BBOX_PROJECT_TYPE as _BBOX_PROJECT_TYPE,
    COMPOSITE_PROJECT_TYPE as _COMPOSITE_PROJECT_TYPE,
    TRAINING_PROJECT_TYPES,
)
# The 2026-09-27 VEGA annotator tidy-up plan's no-re-screening ledger: builds
# {stem: [labels]} from the coverage matrix for Step 3's --prior-screening
# flag (screened_by_stem) and prior_screening.json for the combined
# annotator, refreshed before every launch (write_prior_screening).
from _reefreview import screening as _screening

# ---------------------------------------------------------------------------
# Asset cache-bust + user-facing step numbering
# ---------------------------------------------------------------------------
# One bump point for both /static assets. The template renders it into the
# style.css AND orchestrator.js query strings, so a release only has to change
# this string once.
ASSET_VERSION = "20260921-start"

# Internal step key -> the number the sidebar shows. The chain slots 4 and 5
# were archived when the combined annotator (route key step4test) took over
# both, so every step after it displays one lower than its internal id. Every
# user-facing string goes through step_ref() rather than hard-coding a number.
STEP_DISPLAY = {
    "1": "1",
    "2": "2",
    "3": "3",
    "step4test": "4",
    "6": "5",
    "7": "6",
    "8": "7",
}

# Sidebar title per step key, so an error names the tile the reader can see.
STEP_TITLES = {
    "1": "Fetch Points",
    "2": "Fetch Labels",
    "3": "Choose Images",
    "step4test": "Place + Segment",
    "6": "Train Model",
    "7": "Evaluate Model",
    "8": "Model Inference",
}

# HTTP status for a launch refused because the step's process is still alive:
# the request conflicts with the step's current state and nothing was spawned.
HTTP_BAD_REQUEST = 400
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_CONFLICT = 409
# HTTP status for a launch that started and failed, or raised.
HTTP_SERVER_ERROR = 500

# What a press refused as busy is told to do, after the sidebar label: a CLI
# stage finishes on its own; a Flask sub-app never does and has a Stop button.
BUSY_ADVICE = "Wait for it to finish before starting it again."
FLASK_BUSY_ADVICE = "Open the one that is running, or press Stop, before starting it again."
# Appended when the live process outlived an orchestrator restart, so the
# reader can find the one process the orchestrator can no longer stop.
ORPHAN_NOTE = (" It is process {pid}, which outlived an orchestrator restart, so the "
               "orchestrator cannot stop it.")

# How much of an error message the VICARIUS process_end note keeps.
_VIC_NOTE_MAX_CHARS = 500

# Where the kernel lists live processes. A Step 6 driver that outlived an
# orchestrator restart is found here, because the runner forgets its
# children when the orchestrator is killed.
PROC_ROOT = "/proc"
STEP6_DRIVER_NAME = os.path.basename(ENTRY_POINTS[6])


def step_ref(step):
    """User-facing name for a step: sidebar title plus its displayed number.

    Parameters:
        step: the internal step key, an int or a string such as "step4test".
    Returns:
        "<title> (step <n> in the sidebar)", the bare title for a key with no
        sidebar number, or "step <key>" for a key with no title.

    Example: step_ref(7) -> "Evaluate Model (step 6 in the sidebar)".
    """
    key = str(step)
    title = STEP_TITLES.get(key)
    num = STEP_DISPLAY.get(key)
    if not title:
        return f"step {key}"
    return f"{title} (step {num} in the sidebar)" if num else title


# ---------------------------------------------------------------------------
# Training projects (composite training sets and staged box datasets)
# ---------------------------------------------------------------------------
# A run dir minted from a prepared training source holds no annotation work of
# its own: it opens on Train and every annotation route refuses. Two kinds
# reach this orchestrator and both behave identically here:
#
#   composite_training  minted from the coverage matrix ("Make a project from
#                       this set"). project.json carries a "composite" block
#                       and step4test_combinedAnnotate is a SYMLINK to
#                       <inprocess>/_composite/<set>, which Step 6 trains from
#                       directly. See _matrix/composite.py.
#   bbox_training       minted from the Bounding Boxes page. project.json
#                       carries a "training_source" block naming a staged
#                       detection dataset, which Step 6 builds its own split
#                       from. NO imagery enters the run dir. See
#                       _boxes/blueprint.py.
#
# Both are minted by _reefreview/training_projects.py, which owns
# TRAINING_PROJECT_TYPES.
COMPOSITE_PROJECT_TYPE = _COMPOSITE_PROJECT_TYPE
BBOX_PROJECT_TYPE = _BBOX_PROJECT_TYPE
COMPOSITE_TRAIN_STEP = 6
COMPOSITE_REFUSAL = ("Not available for a training project. "
                     "This project trains from the training set or box dataset named "
                     "in its README.")

# What the Train banner calls each kind of source, in plain language.
_TRAINING_SOURCE_LABELS = {
    COMPOSITE_PROJECT_TYPE: "training set",
    BBOX_PROJECT_TYPE: "box dataset",
}


def _is_composite_training(state):
    """True when a loaded project trains from a prepared source, of either
    kind. Named for the first kind that existed; every guard below reads it,
    so a bbox project inherits the same refusals without a second code path."""
    return bool(state) and (state or {}).get("project_type") in TRAINING_PROJECT_TYPES


def _training_source_of(state):
    """(label, name) for a training project's source, ('', '') for an ordinary
    project. Reads whichever block the project.json carries: "composite" for a
    training set, "training_source" for a staged box dataset."""
    if not _is_composite_training(state):
        return "", ""
    ptype = (state or {}).get("project_type")
    block = (state.get("training_source") if ptype == BBOX_PROJECT_TYPE
             else state.get("composite")) or {}
    return _TRAINING_SOURCE_LABELS.get(ptype, "training source"), str(block.get("name") or "")


def _state_payload(state):
    """project.json as the SPA reads it, plus five DERIVED keys the browser
    would otherwise have to work out per project type:

        is_training_project       the SPA's disabled-tile and banner switch
        training_source_label     "training set" or "box dataset"
        training_source_name      the set or dataset this project trains on
        eval_conf_upgraded_from   the Evaluate confidence replaced when this
                                  project was opened, absent when none was
        eval_settings_off_recommended
                                  every Evaluate setting this project has
                                  saved that is not the recommended one, as
                                  {key: {saved, recommended}}, so the panel
                                  can say which of the numbers it opens on
                                  differ and what they decide; {} when every
                                  saved setting matches

    Derived on the way out and never saved, so project.json keeps exactly the
    keys the minter wrote and a stale copy of these can never be read back."""
    if not state:
        return state
    out = dict(state)
    label, name = _training_source_of(state)
    out["is_training_project"] = _is_composite_training(state)
    out["training_source_label"] = label
    out["training_source_name"] = name
    out["eval_settings_off_recommended"] = _eval_settings_off_recommended(state)
    upgraded = _EVAL_CONF_UPGRADES.get(os.path.abspath(state.get("project_dir") or ""))
    if upgraded is not None:
        out["eval_conf_upgraded_from"] = upgraded
    return out


def _saved_current_step(project_dir):
    """current_step exactly as project.json holds it, read BEFORE
    pm.load_project reconciles it. The reconcile walks to the first
    non-completed step, which for a training project is always the skipped
    Step 1, so the value a returning user left behind is only visible here.
    0 when the file is missing or unreadable."""
    try:
        with open(os.path.join(project_dir, "project.json")) as fh:
            return int((json.load(fh) or {}).get("current_step") or 0)
    except (OSError, ValueError, TypeError):
        return 0


def _apply_composite_training_state(state, saved_step=0):
    """Post-load fixups for a training project, applied after
    pm.load_project(): Train is unlocked from the start (nothing upstream will
    ever complete to unlock it), and the project opens on Train rather than on
    the first non-completed step, which would be the skipped Step 1.

    A user who has moved PAST Train reopens where they left off: the opening
    step is only forced to Train when the step they left on was below it, so
    Evaluate and Inference are not thrown away on every reopen.

    Returns the same state object; a no-op for an ordinary project."""
    if not _is_composite_training(state):
        return state
    s6 = (state.get("steps") or {}).get(str(COMPOSITE_TRAIN_STEP))
    if s6 is not None and s6.get("status") == "locked":
        s6["status"] = "pending"
    # A project whose source says "detect" opens the Train panel on the
    # detector, base weights included. Without this the step-6 defaults grafted
    # on first open carry the segmentation model, and a box dataset (which has
    # no outlines) would refuse its own first launch over a default nobody
    # chose. Only the untouched segmentation default is replaced: a deliberate
    # later pick stays, and a genuinely mismatched one is still refused at
    # launch by check_model_task.
    if s6 is not None and _project_default_task(state) == "detect":
        cfg6 = s6.setdefault("config", {})
        if not _normalize_task(cfg6.get("task")):
            cfg6["task"] = "detect"
        if not cfg6.get("model") or cfg6.get("model") == default_model_for_task("segment"):
            cfg6["model"] = default_model_for_task("detect")
    last_step = int(pm.STEP_KEYS[-1])
    try:
        saved_step = int(saved_step or 0)
    except (TypeError, ValueError):
        saved_step = 0
    state["current_step"] = (min(saved_step, last_step)
                             if saved_step > COMPOSITE_TRAIN_STEP
                             else COMPOSITE_TRAIN_STEP)
    pm.save_project(state)
    return state


def _load_project(project_dir):
    """pm.load_project plus the composite-training fixups and the Evaluate
    confidence upgrade. Every path that opens a project goes through this, so
    a training project always arrives on Train (or on the later step it was
    left on) with Step 6 reachable, and no project still opens the Evaluate
    panel on the retired measurement cut-off.

    _upgrade_retired_eval_conf is defined beside the Evaluate defaults it
    reads, further down this file."""
    saved_step = _saved_current_step(project_dir)
    state = _apply_composite_training_state(pm.load_project(project_dir), saved_step)
    return _upgrade_retired_eval_conf(state)


# ---------------------------------------------------------------------------
# Step 6 training task
# ---------------------------------------------------------------------------
# Segmentation is the default everywhere. "detect" trains a bounding-box model
# instead, from the export's polygons reduced to their boxes (an annotation
# project) or from a box dataset directly (a bbox training project).
#
# Every spelling the platform accepts is normalized ONCE here, exactly as
# TCRMPtrain_oceankindCV/run_step6.sh and src/train_wrapper.py::_task_arg do,
# so a value that reaches the engine can never be rejected there after the
# orchestrator has already committed to a launch.
STEP6_TASKS = {"segment": "segment", "seg": "segment", "segmentation": "segment",
               "detect": "detect", "detection": "detect", "bbox": "detect"}
STEP6_DEFAULT_TASK = "segment"


def _normalize_task(value):
    """The engine's spelling of a task, or '' when the value is unknown."""
    return STEP6_TASKS.get(str(value or "").strip().lower(), "")


def _project_default_task(state):
    """The task a project's Train panel opens on: "detect" only for a project
    whose project.json training_source says so (a box dataset has no outlines
    to learn from), "segment" for every other project."""
    src = (state or {}).get("training_source") or {}
    return _normalize_task(src.get("task_default")) or STEP6_DEFAULT_TASK


# ---------------------------------------------------------------------------
# Step 6 training recipes: what a recipe is FOR
# ---------------------------------------------------------------------------
# A recipe (a preset YAML in config/presets/) declares the two facts that
# decide whether it suits the run in front of the operator: the task it trains
# (the STEP6_TASKS vocabulary above) and the subject it was written for. The
# Train panel carries one control per fact, lists only the recipes matching
# both, and applies one only when exactly one matches.
#
# Nothing here reads a file name. Selection by file order was removed on
# 2026-09-15: the recipe that won was whichever name sorted first among those
# carrying a flag, so adding a coral recipe beside the fish one would have
# switched box-dataset training onto the coral recipe with nothing on screen
# saying so.
#
# A recipe written for every subject declares STEP6_SUBJECT_ANY. The panel's
# own control adds STEP6_SUBJECT_OTHER for a subject no recipe names, which
# therefore matches the "any" recipes and nothing else.
STEP6_SUBJECT_ANY = "any"
STEP6_SUBJECT_OTHER = "other"
# The subjects a dataset and a recipe can both name.
STEP6_SUBJECTS = ("coral", "fish", "algae")
# Everything a recipe's _meta.subject may say.
STEP6_PRESET_SUBJECTS = STEP6_SUBJECTS + (STEP6_SUBJECT_ANY,)
# Everything the Train panel's Subject control may hold.
STEP6_PANEL_SUBJECTS = STEP6_SUBJECTS + (STEP6_SUBJECT_OTHER,)
# What a staged box dataset records itself in, beside its images and labels.
BBOX_MANIFEST_NAME = "ingest_manifest.json"


def _normalize_subject(value, allowed):
    """The vocabulary spelling of a subject, or '' when it is not in it.

    value: whatever a preset file or a dataset manifest holds, of any type.
    allowed: the tuple of spellings this caller accepts (STEP6_SUBJECTS for a
      dataset, STEP6_PRESET_SUBJECTS for a recipe).
    Returns the stripped lower-case spelling when allowed holds it, else "".
    A value that is not a string names no subject and answers "" as well,
    rather than failing the caller that reads a file somebody hand-edited.
    """
    if not isinstance(value, str):
        return ""
    want = value.strip().lower()
    return want if want in allowed else ""


def _step6_recipe_declaration(meta):
    """What a training recipe's _meta block says the recipe is for.

    meta: the _meta mapping of a preset file, or {} when the file carries none
      or carries something other than a mapping.
    Returns (task, subject, warning):
      task     the engine's spelling of _meta.task, "" when absent or unknown
      subject  the vocabulary spelling of _meta.subject, "" likewise
      warning  "" when both are known, else one sentence naming what the file
               said and what the key accepts
    A recipe with a warning is unclassified: the Train panel lists it so it can
    still be imported by hand, and never selects it on its own.

    Example: {"task": "Detect", "subject": "coral"} gives ("detect", "coral", "").
    """
    task = _normalize_task(meta.get("task"))
    subject = _normalize_subject(meta.get("subject"), STEP6_PRESET_SUBJECTS)
    if task and subject:
        return task, subject, ""
    problems = []
    if not task:
        problems.append(f"_meta.task is {meta.get('task')!r}, not one of "
                        f"{', '.join(sorted(set(STEP6_TASKS.values())))}")
    if not subject:
        problems.append(f"_meta.subject is {meta.get('subject')!r}, not one of "
                        f"{', '.join(STEP6_PRESET_SUBJECTS)}")
    return task, subject, (
        "this recipe is unclassified, so VICARIUS never selects it on its own: "
        + "; ".join(problems))


def _project_dataset_subject(state):
    """The subject a box-dataset project's staged dataset recorded for itself.

    state: a project.json dict, or None when no project is open.
    Returns (subject, warning):
      subject  the vocabulary spelling of the manifest's "subject" key; "" for
               a project that trains from no staged box dataset, for a dataset
               that recorded no subject, and for a value outside the vocabulary
      warning  "" unless the manifest is on disk and could not be read, in
               which case one sentence naming the file and what went wrong
    The Train panel uses this as the INITIAL value of its Subject control. It
    never overrides a value the operator set, and it never selects a recipe:
    the two controls do that. _recorded_training_source, which resolves the
    staged dataset folder, is defined further down this file.
    """
    if not state or state.get("project_type") != BBOX_PROJECT_TYPE:
        return "", ""
    dataset_dir = _recorded_training_source(state)
    if not dataset_dir:
        return "", ""
    path = os.path.join(dataset_dir, BBOX_MANIFEST_NAME)
    if not os.path.isfile(path):
        return "", ""
    try:
        with open(path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError) as exc:
        return "", (f"the subject recorded with this project's box dataset could not be "
                    f"read from {path}: {exc}")
    if not isinstance(manifest, dict):
        return "", (f"{path} holds {type(manifest).__name__} where a staged dataset "
                    f"manifest is a mapping, so it records no subject")
    return _normalize_subject(manifest.get("subject"), STEP6_SUBJECTS), ""


def default_model_for_task(task, size="m"):
    """Base weights when the panel named none. Byte-identical rule to
    train_wrapper.py::default_model_for, which is the source of truth:
    yolo11<size>.pt for detect, yolo11<size>-seg.pt for segment."""
    return f"yolo11{size}.pt" if task == "detect" else f"yolo11{size}-seg.pt"


def check_model_task(model, task):
    """'' when the weights suit the task, else why they do not.

    A COPY of train_wrapper.py::check_model_task, wording included, so the
    orchestrator can refuse the pairing before it launches anything instead of
    letting the run die a few seconds into its own start-up. The engine keeps
    its own copy as the last line of defence for callers that never come
    through here (the model lab, a hand-run script); test_step6_task.py pins
    the two strings against each other.

    Why it matters: ultralytics reads the task of a .pt checkpoint from the
    checkpoint and ignores the task it is handed, so a mismatched pair does not
    fail, it silently trains the OTHER head into this run's folder. Only the
    file name is evidence, so neutral names (best.pt, last.pt) are left alone.
    """
    base = os.path.basename(str(model or ""))
    if task == "detect" and "-seg" in base:
        return (f"task=detect cannot train from segmentation weights ({base}). "
                "Pick a detection base model, or fine-tune from a detection run.")
    return ""


# The split builder for a staged box dataset. It lives with the training tool
# (TCRMPtrain_oceankindCV/src/detect_dataset.py) so the model lab and the
# pipeline build byte-identical splits; loaded by file path under a private
# name for the same reason _modellab/labruns.py does, and lazily so a missing
# training tool cannot stop this app from importing.
_DETECT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "TCRMPtrain_oceankindCV", "src", "detect_dataset.py")
_detect_dataset_mod = None


def _load_detect_dataset():
    global _detect_dataset_mod
    if _detect_dataset_mod is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_orch_detect_dataset",
                                                      _DETECT_DATASET_PATH)
        if spec is None or spec.loader is None or not os.path.isfile(_DETECT_DATASET_PATH):
            raise ImportError(f"cannot load the detection dataset builder at "
                              f"{_DETECT_DATASET_PATH}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _detect_dataset_mod = mod
    return _detect_dataset_mod


# The outline (segmentation) splitter's transect rule, loaded the same lazy
# importlib.util way as _load_detect_dataset just above, so the Step 6
# training-data summary can replay parse_transect without pinned_split.py
# becoming a package-level import of this app.
_PINNED_SPLIT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "TCRMPtrain_oceankindCV", "src", "pinned_split.py")
_pinned_split_mod = None


def _load_pinned_split():
    global _pinned_split_mod
    if _pinned_split_mod is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_orch_pinned_split",
                                                      _PINNED_SPLIT_PATH)
        if spec is None or spec.loader is None or not os.path.isfile(_PINNED_SPLIT_PATH):
            raise ImportError(f"cannot load the transect split rule at "
                              f"{_PINNED_SPLIT_PATH}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _pinned_split_mod = mod
    return _pinned_split_mod


def _build_detect_dataset(source_dir, dataset_dir, seed_name, *, valid_frac, test_frac):
    """Build <dataset_dir> (train/valid/test + data.yaml) from a staged box
    dataset.

    Parameters:
        source_dir: the staged dataset (images/ beside labels/).
        dataset_dir: where the split dataset goes, replaced whole.
        seed_name: the staged dataset's folder name; both callers (this
            window and the Bounding Boxes quick trainer) seed on it, so the
            same dataset splits the same way wherever it is trained.
        valid_frac, test_frac: the held-out fractions from the Train panel,
            applied to images not yet pinned by an earlier split.
    Returns:
        the split manifest the builder wrote (seed, fractions, fingerprint,
        counts). Raises ValueError when the source holds no image and label
        pair, the seed is empty, a fraction is out of range, or the saved or
        staged split names a photo the dataset cannot train on (VEGA never
        re-draws a held-out photo in silence; _launch_step6 shows the reason).
    """
    return _load_detect_dataset().build_detect_dataset(
        source_dir, dataset_dir, seed_name, valid_frac=valid_frac, test_frac=test_frac)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
runner = StageRunner()
current_project = None  # dict (project.json contents)

# One Start Training press at a time. The busy probe at the top of
# _launch_step6 and the runner's own check at its end bracket a dataset build
# that removes and replaces <step6>/dataset, so two presses a few hundred
# milliseconds apart would both pass the probe and both rebuild the dataset a
# live trainer is reading. The lock makes the second press wait until the
# first has registered its process, so the probe refuses it.
_step6_launch_lock = threading.Lock()

# SAM3 driver state, updated by a background thread per step 5 launch so the
# orchestrator UI can show progress without the user having to open the sub-app.
sam3_status = {
    "phase": "idle",       # idle | launching | waiting_for_sub_app | configuring |
                           # segmenting | review_ready | error
    "processed": 0,
    "total": 0,
    "running": False,
    "error": None,
    "message": "",
    "updated_at": None,
}
sam3_lock = threading.Lock()
sam3_thread = None

# Combined-annotator (step4test) launch state - this non-chain tile has no
# project step in STEP_KEYS, so it tracks its own port + status here.
step4test_state = {"running": False, "port": None, "started_at": None, "out_dir": None,
                   "project_id": None, "input_dir": None}

# Combined-annotator config collected from the #panel-step4test inputs and POSTed
# with /api/step/step4test/run|reroute. This non-chain tile has no project step,
# so its settings live here (transient, per-launch). _launch_step4test_ui reads
# these and falls back to the inherited Step-5 saved config for any unset key.
step4test_cfg = {}

# Refine loop (step4loop) - the active-learning resume-mode annotator. Same
# shape as step4test_state/step4test_cfg, but tracked separately because it is
# a distinct runner key ("step4loop") that can run independently of step4test
# (though _kill_all_annotators still enforces one annotator process at a time).
step4loop_state = {"running": False, "port": None, "started_at": None, "out_dir": None,
                   "project_id": None, "input_dir": None}

# step4loop panel config (#panel-step4loop inputs), collected client-side and
# POSTed with /api/step/step4loop/run. Transient, per-launch, same pattern as
# step4test_cfg.
step4loop_cfg = {}

# Edit Masks (editmasks) - launches the standalone TCRMPclip_editMasks app
# against the CURRENT project's step4test export, in "edit" session mode
# (boots straight into the review queue via /api/resume, same as step4loop,
# but with no seeding pass: it opens on whatever masks Step 4 already wrote).
# Same shape as step4test_state/step4loop_state, but its own runner key
# ("editmasks") so it can run independently (though _kill_all_annotators
# still enforces one annotator-family app at a time).
editmasks_state = {"running": False, "port": None, "started_at": None, "out_dir": None,
                   "project_id": None, "input_dir": None}

# Annotate Images (standalone) - launches TCRMPclip_combinedAnnotate with no
# project attached, for an operator who just wants to import their own images
# and label them. Own runner key ("annotate") and own port (5096, deliberately
# outside the 5080-5089 sweep _kill_all_annotators uses below, and clear of
# 5090 which is the VICARIUS UI, 5092 which belongs to the TagFab
# orchestrator, and 5077 which is reserved). The guarantee runs both ways:
# _launch_standalone_annotator never calls _kill_all_annotators, so a
# standalone launch never kills a project annotator; and
# _kill_all_annotators (run by every project-mode launch) skips any process
# whose environment carries TCRMP_SESSION_MODE=standalone, so a project-mode
# launch cannot kill a running standalone session either, by name or by port.
ANNOTATE_PORT = 5096
annotate_state = {"running": False, "port": None, "started_at": None, "out_dir": None}

# ---------------------------------------------------------------------------
# Whose run the Evaluate step is scoring
# ---------------------------------------------------------------------------
# Evaluate is the one step that can be launched against a run the open project
# does not own: its runs card lists every run the module can see (a model lab
# under inprocess/_models/, another project's runs/), and _run_step7 scores
# whichever was picked on that run's own dataset, writing its ledger row
# beside it. One poll later, step_status has to decide whether the open
# project's own Evaluate step may be recorded as completed, and only the
# launch knows whose run it was.
#
# That fact is kept here, in the orchestrator process, and never in
# project.json: project.json is the record of the project's own work, and
# whose run is being scored right now is transient state that dies with the
# process that started it. The dict is keyed by the step key, so a later
# launch of that step replaces an earlier one and no other step can read it.
EVALUATE_STEP = 7
_evaluation_launches = {}

# The status an Evaluate step is left in when a foreign evaluation finishes
# and nothing records what the step looked like before it started: "not done
# yet", the same neutral state a stopped step is left in. A step left reading
# "running" with nothing running would be a false record of its own.
EVALUATE_NEUTRAL_STATUS = "pending"

# The status a step carries while its process is in flight. Named so
# _restore_evaluation_step can say, by name, the one status it never writes
# back: the process is over by the time it runs.
RUNNING_STATUS = "running"

# The sentence the status poll carries when the evaluation that finished
# scored a run the open project does not own. Same voice as the warnings
# _run_step7 returns: it names the run and says what was left alone.
FOREIGN_EVALUATION_NOTE = (
    "The run '{run_name}' is not part of the open project, so VEGA left this project's "
    "{step_ref} exactly as it was; the score was written to that run's own ledger."
)
# The same sentence for a poll that arrives with nothing left to say whose run
# was scored. Two things can leave that behind: an orchestrator restart during
# the evaluation, and a poll taken while a different project was open, which
# reads the record, sees it belongs to another project and records nothing.
# The sentence names neither, because this code cannot tell them apart and
# stating the wrong one is its own false claim. It says only what is known and
# what to do next. Nothing is recorded, because nothing here can tell whose
# work it was, and a wrong guess would write a claim that never happened.
UNKNOWN_EVALUATION_NOTE = (
    "VEGA cannot tell which training run this evaluation scored, so this project's "
    "{step_ref} was left exactly as it was. Run it again on one of this project's own "
    "runs to record it here."
)


def _remember_evaluation_launch(owned, run_name, out_dir):
    """Record whose run the Evaluate step has just been launched against.

    Parameters:
        owned: True when the run being scored sits under the open project's
            own runs folder, False for a model lab's run or another project's.
        run_name: the run folder's name, for the sentence the reader gets when
            the evaluation turns out to have been someone else's.
        out_dir: the evaluation folder this launch writes into, so the
            VICARIUS process event can name the artifact it really produced.
    Returns:
        None. The record replaces any earlier one for this step and is taken,
        once, by the status poll of the project that made it; a poll made
        while a different project is open reads it, records nothing for that
        project, and leaves it where this project's own poll can still take
        it.
    """
    _evaluation_launches[str(EVALUATE_STEP)] = {
        "owned": bool(owned),
        "run_name": str(run_name or ""),
        "out_dir": str(out_dir or ""),
        "project_id": str((current_project or {}).get("id") or ""),
        "prior_status": None,
        "prior_started_at": None,
    }


def _remember_evaluation_prior_state(prior):
    """Add to the remembered launch what the step looked like before it ran.

    step_run stamps a step as running and records a start time before it
    dispatches, because a launch normally succeeds. When the launch turns out
    to have scored someone else's run, those two writes are the only marks it
    left on the open project, and _restore_evaluation_step undoes them from
    this record.

    This runs in the same request as the launch, two statements after it, so
    a status poll can only read a record without a prior state if it arrives
    in that microsecond window; the step is then left EVALUATE_NEUTRAL_STATUS
    rather than at the status it held, which claims no work either way.

    Parameters:
        prior: the Evaluate step's "status" and "started_at" as step_run read
            them before the launch, None where a key was absent.
    Returns:
        None. A launch that was refused left no record and is left alone.
    """
    record = _evaluation_launches.get(str(EVALUATE_STEP))
    if record is None:
        return
    record["prior_status"] = prior.get("status")
    record["prior_started_at"] = prior.get("started_at")


def _take_evaluation_launch(project_id):
    """Take the record of the Evaluate launch that has just finished.

    The record is read before it is removed, and it is removed only when it
    belongs to the project that is open now. A poll can arrive for a different
    project: open project A, start an evaluation of one of A's own runs, then
    open project B whose Evaluate step still reads "running" from an earlier
    crash. B's poller sees A's process finish. Nothing may be recorded for B,
    which is what the None answer does, but A's record must survive that poll,
    or A's genuine evaluation is never recorded when A is reopened and the
    operator has to press Evaluate again for work that already ran.

    Parameters:
        project_id: the id of the project that is open now.
    Returns:
        the record when one was made for this project, else None. None means
        either that no launch was remembered (the orchestrator was restarted
        while the evaluation ran) or that the launch was made while another
        project was open. Neither can say that the open project did this work.
    """
    record = _evaluation_launches.get(str(EVALUATE_STEP))
    if record is None:
        return None
    if record.get("project_id") != str(project_id or ""):
        return None
    return _evaluation_launches.pop(str(EVALUATE_STEP), None)


def _forget_evaluation_launch():
    """Drop the record of a finished Evaluate launch without reading it.

    Called when the evaluation exited non-zero: nothing is recorded as done,
    and the spent record must not be read by a later poll.
    """
    _evaluation_launches.pop(str(EVALUATE_STEP), None)


def _restore_evaluation_step(launch):
    """Put the open project's Evaluate step back the way the launch found it.

    An evaluation of someone else's run leaves exactly two marks on the open
    project, both written by step_run before it dispatched: the step reads
    "running" and carries a fresh start time. Both are written back here and
    project.json is saved. The completion stamp, the output pointers and the
    project's current step are never touched: this path completes nothing, so
    it has nothing to say about them.

    A saved status of "running" is the one thing never written back. The step
    can be pressed over a stale "running" mark, because pm.load_project does
    not normalise one left behind by a crash, and the launch then remembers
    it. The process that mark named is over by the time this runs, so writing
    it back would leave the step reading "running" with nothing running: a
    false record of its own, and one the next status poll reads as a step
    still in flight, sending it back through the completion branch to write a
    second success row for a single process. Such a step is left
    EVALUATE_NEUTRAL_STATUS, which claims only that the step is not done.

    Parameters:
        launch: the record the launch left, or None when nothing remembers it.
            Without a record the step is left EVALUATE_NEUTRAL_STATUS, which
            claims only that the step is not done.
    Returns:
        None. current_project is updated in memory and saved to disk.
    """
    if current_project is None:
        return
    st = current_project["steps"].get(str(EVALUATE_STEP))
    if st is None:
        return
    prior_status = str((launch or {}).get("prior_status") or "")
    st["status"] = (EVALUATE_NEUTRAL_STATUS
                    if prior_status in ("", RUNNING_STATUS) else prior_status)
    prior_started = (launch or {}).get("prior_started_at")
    if prior_started:
        st["started_at"] = prior_started
    else:
        st.pop("started_at", None)
    pm.save_project(current_project)


def _finish_foreign_evaluation(step, status):
    """Close a finished evaluation the open project must not be credited with.

    A run that is not the open project's is scored on its own dataset and its
    ledger row is written beside it (see _run_step7), so the open project did
    no work: its Evaluate step keeps the status, the completion stamp, the
    output pointers and the current step it had before the launch, and
    nothing is auto-linked or unlocked behind it. The process itself really
    ran, so its VICARIUS event is closed as a success against the folder it
    wrote, and the poll still reports a finished, zero-exit process, which is
    what stops the panel polling and reloads its report card on the score.

    Parameters:
        step: the step key this status poll is about. Only the Evaluate step
            can be launched against another owner's run, so every other step
            is handed straight back to the caller.
        status: the poll payload about to be returned. When this function
            takes the poll it sets "completed" to False and appends one plain
            sentence to "warnings".
    Returns:
        True when the poll was handled here and the caller must not complete
        the step; False when the evaluation was the open project's own work,
        or this is another step, and the caller's normal completion applies.
    """
    if step != EVALUATE_STEP or current_project is None:
        return False
    launch = _take_evaluation_launch(current_project.get("id"))
    if launch is not None and launch.get("owned"):
        return False
    if launch is None:
        note = UNKNOWN_EVALUATION_NOTE.format(step_ref=step_ref(EVALUATE_STEP))
    else:
        note = FOREIGN_EVALUATION_NOTE.format(
            run_name=launch.get("run_name") or "(unnamed)",
            step_ref=step_ref(EVALUATE_STEP))
    out_dir = str((launch or {}).get("out_dir") or "")
    _vic_end_step(step, "success", notes=note[:_VIC_NOTE_MAX_CHARS],
                  outputs=[out_dir] if out_dir else [])
    _restore_evaluation_step(launch)
    status["completed"] = False
    status["warnings"] = list(status.get("warnings") or []) + [note]
    return True


# ---------------------------------------------------------------------------
# Which project a step's process belongs to
# ---------------------------------------------------------------------------
# The stage runner holds ONE process per step key for the whole orchestrator,
# and the status poll used to record whichever project happened to be open
# when that process finished. Open project A, start its training, open project
# B while it runs, and B's own poller wrote A's training into B's project.json
# as completed work B never did.
#
# The launch records whose work it is, here, in this process: project.json is
# the record of a project's own work, and which project a live process belongs
# to is transient state that dies with the orchestrator. The dict is keyed by
# the step key, so a later launch of that step replaces an earlier one.
#
# The Evaluate step keeps its own richer record (_evaluation_launches) because
# it can be launched against another owner's RUN while still being the open
# project's own work; this one answers the plainer question of which PROJECT
# pressed the button.
_step_launches = {}

# What the status poll says when the step that finished belongs to another
# project. It names that project, because the reader is looking at a window
# where nothing changed and needs to know where the result went.
FOREIGN_STEP_NOTE = (
    "{step_ref} was started by project '{project_name}', not the one open here, so VEGA "
    "recorded the result there and left this project exactly as it was."
)
# The same poll when that project's own record could not be written.
FOREIGN_STEP_UNWRITABLE_NOTE = (
    "{step_ref} was started by project '{project_name}', not the one open here, and VEGA "
    "could not record the result there: {reason}. Open that project and check "
    "{step_ref} before running it again."
)
# What the poll says when the orchestrator was restarted while the step's
# process was running: the runner forgets its children when it is killed, so
# it can no longer say whether that process finished, failed or is still
# going. Nothing is recorded, because every one of those would be a guess.
RESTARTED_STEP_NOTE = (
    "VEGA was restarted while {step_ref} was running, so it no longer owns that process "
    "and cannot say how it ended. Nothing was recorded. Check {step_dir}, then press Stop "
    "to clear the mark or run the step again."
)
# The same restart, but the process is still alive: the kernel's own process
# list says so, and the pid rides along because the runner can no longer stop
# it.
ORPHAN_STEP_NOTE = (
    "{step_ref} is still running as process {pid}, which outlived a VEGA restart, so the "
    "orchestrator cannot stop it. Its output goes on being written to {step_dir}."
)


def _remember_step_launch(step, pid):
    """Record that the open project is the one that launched this step.

    Parameters:
        step: the step key whose process has just been dispatched.
        pid: the process id the runner answered with, which ties this record
            to one process. Without it a record left by an earlier launch
            that nobody polled would be read as the owner of a later,
            unrelated process.
    Returns:
        None. Any earlier record for the step is replaced, because the runner
        holds one process per step and the newest launch owns it.
    """
    _step_launches[str(step)] = {
        "project_id": str((current_project or {}).get("id") or ""),
        "project_dir": str((current_project or {}).get("project_dir") or ""),
        "project_name": str((current_project or {}).get("name") or ""),
        "pid": _as_int(pid),
    }


def _runner_pid(step):
    """The process id the runner currently holds for a step, or None.

    The runner exposes no accessor for it, so its process table is read
    directly, the same way this file already reads runner.ports. Only the pid
    is taken and nothing is written. None when the runner holds no process
    for the step, and None for a stand-in runner with no readable process
    table: a pid that cannot be read cannot confirm that a launch record
    describes the process this poll is reporting on, and an unconfirmed
    record is treated as stale rather than credited to another project.
    """
    processes = getattr(runner, "processes", None)
    proc = processes.get(step) if isinstance(processes, dict) else None
    return getattr(proc, "pid", None) if proc is not None else None


def _take_step_launch(step):
    """Take the record of the step launch the poll is reporting on.

    Parameters:
        step: the step key the status poll is about.
    Returns:
        the record when it describes the process the runner is holding for
        this step, else None. None means one of three things, and none of
        them may credit another project: nothing remembers the launch (an
        orchestrator restart), or the record belongs to an earlier launch
        that nobody ever polled, whose pid no longer matches the process the
        runner holds now. A record that is read, or found stale, is removed:
        the process it described is over.
    """
    record = _step_launches.get(str(step))
    if record is None:
        return None
    _step_launches.pop(str(step), None)
    if record.get("pid") is not None and record["pid"] != _runner_pid(step):
        return None
    return record


def _forget_step_launch(step):
    """Drop a step's launch record without reading it. Used when a launch was
    refused or undone, so a later poll cannot credit a process that never ran."""
    _step_launches.pop(str(step), None)


def _proc_environ(pid):
    """Raw bytes of /proc/<pid>/environ (NUL-separated KEY=VALUE entries), or
    b'' when unreadable (process already gone, permission denied, /proc
    absent). Kept as its own module-level function - not inlined into
    _kill_all_annotators - so tests can monkeypatch it without touching a
    real /proc entry."""
    try:
        with open(f"/proc/{pid}/environ", "rb") as fh:
            return fh.read()
    except Exception:
        return b""


def _proc_cmdline(pid):
    """The command line of a pid as one string (the NUL separators in
    /proc/<pid>/cmdline become spaces), or '' when unreadable. Its own
    module-level function so tests can monkeypatch it, and so
    _kill_all_annotators can fail closed on '' rather than guess."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().decode("utf-8", "replace").replace("\0", " ")
    except Exception:
        return ""


def _port_listener_pids(ss_output, port):
    """PIDs listening on `port`, read out of `ss -ltnp` output.

    An `ss -ltnp` row looks like
    `LISTEN 0 128 0.0.0.0:5080 0.0.0.0:* users:(("python",pid=1583163,fd=49))`,
    so the 4th field carries the local address and every `pid=<n>` in the row
    belongs to that listener. Returns [] for anything it cannot parse (no
    output, a header row, no pid= field, a stubbed subprocess.run in tests)
    and never raises."""
    pids = []
    try:
        for line in (ss_output or "").splitlines():
            fields = line.split()
            if len(fields) < 4 or not fields[3].endswith(f":{port}"):
                continue
            for m in re.finditer(r"pid=(\d+)", line):
                pid = int(m.group(1))
                if pid not in pids:
                    pids.append(pid)
    except Exception:
        return []
    return pids


def _annotator_pids(entry_match):
    """PIDs of processes whose command line matches entry_match, via
    `pgrep -f`. Returns [] on any failure (no pgrep, no matches, or a stubbed
    subprocess.run in tests) rather than raising. Replaces the old bare
    `pkill -f` call in _kill_all_annotators so each matching pid can be
    inspected (and possibly spared, if it is a standalone session) before
    being killed."""
    import subprocess
    try:
        result = subprocess.run(["pgrep", "-f", entry_match],
                                 capture_output=True, text=True)
        out = getattr(result, "stdout", "") or ""
        return [int(tok) for tok in out.split() if tok.strip().isdigit()]
    except Exception:
        return []


def _listener_port_for_pid(ss_output, pid):
    """The local TCP port `pid` is listening on, read out of the same
    `ss -ltnp` snapshot _port_listener_pids parses, or None when this pid is
    not a listener in that snapshot (already exited, not yet bound, or ss
    failed). Reverse of _port_listener_pids (port -> pids): this goes
    pid -> port, so a zombie annotator matched only by _annotator_pids (no
    runner-tracked port) can still be flushed by port before it is
    signalled. Never raises."""
    needle = re.compile(rf"pid={pid}\b")
    try:
        for line in (ss_output or "").splitlines():
            if not needle.search(line):
                continue
            fields = line.split()
            if len(fields) < 4 or ":" not in fields[3]:
                continue
            try:
                return int(fields[3].rsplit(":", 1)[1])
            except ValueError:
                continue
    except Exception as exc:
        vic.note(f"[orch] _listener_port_for_pid could not parse ss output "
                 f"for pid {pid}: {exc}")
        return None
    return None


# Timeout for the "save first" annotator flush (2026-09-27 annotation
# safety-net plan): long enough for a local JSON write, short enough that a
# process that will never answer cannot hang a stop for more than this.
_ANNOTATOR_FLUSH_TIMEOUT_S = 5


def _flush_annotator(port):
    """Ask the annotator or Edit Masks process listening on `port` to save its
    in-memory session to disk (POST /api/flush) before it is stopped.

    The plan's "save first" decision: every path that stops a combined
    annotator or Edit Masks process calls this before signalling it, so an
    unsaved review edit reaches disk before the process is cut off. Never
    raises: no listener, a timeout, and a non-2xx status are all just a
    failed flush, so a stop is never blocked by a process that will not
    answer. Returns (True, None) on any 2xx response, else
    (False, <short reason>) for the caller to log and move on.
    """
    if not port:
        return False, "no port"
    url = f"http://127.0.0.1:{port}/api/flush"
    try:
        req = Request(url, data=b"", method="POST")
        with urlopen(req, timeout=_ANNOTATOR_FLUSH_TIMEOUT_S) as resp:
            status = resp.status if hasattr(resp, "status") else resp.getcode()
            if 200 <= status < 300:
                return True, None
            return False, f"HTTP {status}"
    except URLError as exc:
        code = getattr(exc, "code", None)
        if code is not None:
            return False, f"HTTP {code}"
        return False, str(getattr(exc, "reason", None) or exc)
    except Exception as exc:
        return False, str(exc)


def _live_annotator_port(step):
    """The port to flush for a StageRunner key (step4test, step4loop or
    editmasks), or None when there is nothing safe to flush.

    runner.ports[step] survives after the process behind it exits (kill()
    never clears its own ports map entry), so on its own it can name a port
    an unrelated process has since bound (vicariusLLM on 5085, or
    benthic_species_viewer's serve_edit.py on 5081, on this shared box
    today). This returns a port only when runner.is_running(step) confirms
    that SAME key still owns a live, runner-tracked process; getattr-safe
    for a test double that omits either attribute, in which case nothing is
    returned rather than guessed at.
    """
    is_running = getattr(runner, "is_running", None)
    if is_running is None:
        return None
    try:
        alive = bool(is_running(step))
    except Exception as exc:
        vic.note(f"[orch] _live_annotator_port could not confirm {step!r} "
                 f"is running: {exc}")
        return None
    if not alive:
        return None
    return (getattr(runner, "ports", None) or {}).get(step)


def _project_champion(step6_dir):
    """This project's own gate baseline: the champion record beside its
    ledger, normalised, or None when there is none.

    Scope: project (CHAMPION_SCOPE_PROJECT). evaluate_run.py reads
    {step6_dir}/champion.json to decide what a new round is gated against, so
    this record answers "what were this project's gate columns measured
    against", never "which model does VEGA use by default". That second
    question has one answer for the whole module and it is read from
    labruns.load_champion; see the champion-scope note further down this file.

    Parameters:
        step6_dir: the project's Train step folder, which holds its ledger.
    Returns:
        the record labruns.read_champion_file normalises, or None when
        step6_dir is empty, the file is absent, or it cannot be read.
    """
    if not step6_dir:
        return None
    return _labruns.read_champion_file(
        os.path.join(step6_dir, _labruns.CHAMPION_FILE_NAME))


def _step4test_export_dir():
    """The active project's 4.test combined-annotator export dir: holds
    all_images/, all_labels/, data.yaml, class_map.json, segmentations/.
    4.test replaces Steps 4+5, so this is the canonical training + review source.
    Empty string when no project is loaded. The dir name lives only in
    pm.STEP_DIRS["step4test"], so this helper is the single derivation point
    (re-used by _routed_dir, _expertids_paths, and the folder/reset routes).

    In a composite training project this path is a SYMLINK to the training set
    under <inprocess>/_composite. The path is returned as-is, not resolved:
    os.path.isdir / os.path.isfile follow the link, so the Step 6 preflight
    reads all_images/ and data.yaml straight through it, while every caller
    that reports a path still shows one inside the project. Only the annotation
    routes would ever WRITE here, and they refuse on a training project."""
    if current_project is None:
        return ""
    return os.path.join(current_project["project_dir"], pm.STEP_DIRS["step4test"])


# Local full-set clip dir under the repo's supporting_data. This is the SAME
# tree Step 3 reads (TCRMPcvr_chooseImages config.CLIP_DIR), holding the full
# year/period structure the user installed. Routing reads images only from here
# (no Dropbox dependency).
LOCAL_CLIP_DIR = os.path.join(REPO_DIR, "supporting_data", "TCRMP_clip")


def _resolve_routing_clip_dir(step4test_cfg, s4cfg):
    """Clip image source for the 4.test routing pass. The 4.test panel value
    wins, then the saved Step-4 value, then the LOCAL full-set clip dir. Any
    candidate that does not exist on disk is skipped, so stale Dropbox paths
    saved in older projects no longer break routing."""
    for cand in ((step4test_cfg or {}).get("clip_dir"),
                 (s4cfg or {}).get("clip_dir"),
                 LOCAL_CLIP_DIR):
        if cand and os.path.isdir(cand):
            return cand
    return LOCAL_CLIP_DIR


def _clear_routed_input(routed_dir):
    """Remove every routed_input/<year>/ subtree so a fresh route starts clean.
    A prior route leaves sam_click_prompts.json on disk; reusing it after the
    Step 3 selection changed leaks the old run's frames into the annotator
    regardless of the current species filter. Returns the number of year-dirs
    removed; no-op (0) if routed_dir is absent/empty.
    """
    import shutil as _shutil
    if not routed_dir or not os.path.isdir(routed_dir):
        return 0
    n = 0
    for entry in sorted(os.listdir(routed_dir)):
        p = os.path.join(routed_dir, entry)
        if os.path.isdir(p):
            _shutil.rmtree(p, ignore_errors=True)
            n += 1
    return n


_LORES_MARKER = ".lores_mode"


def _write_routed_lores_marker(routed_dir, on):
    """Record which lores_mode a route was produced with, so a later reuse can
    detect a toggle change and re-route instead of serving the wrong resolution."""
    try:
        with open(os.path.join(routed_dir, _LORES_MARKER), "w") as f:
            f.write("1" if on else "0")
    except OSError:
        pass


def _routed_lores_mode(routed_dir):
    """The lores_mode a cached route was built with (True/False), or None for an
    older cache with no marker (treated as 'unknown', never forces a re-route)."""
    p = os.path.join(routed_dir, _LORES_MARKER)
    if not os.path.isfile(p):
        return None
    try:
        return open(p).read().strip() == "1"
    except OSError:
        return None


def _routed_prompts_stale(routed_dir, selected_frames):
    """True iff cached routing prompts exist but are OLDER than the Step 3
    selection -- i.e. the user re-chose frames since the last route, so the
    cache must be invalidated. False when no prompts exist (nothing to
    invalidate) or selected_frames is missing.
    """
    if not selected_frames or not os.path.isfile(selected_frames):
        return False
    prompt = os.path.join(routed_dir, "ids", "sam_click_prompts.json")
    if not os.path.isfile(prompt):
        return False
    return os.path.getmtime(selected_frames) > os.path.getmtime(prompt)


def _flatten_routed_input(routed_dir):
    """Collapse routed_input/<year>/{ids,raw,test_pts} into flat routed_input/
    {ids/sam_click_prompts.json, raw/}. placePoints writes per-year; the combined
    annotator reads flat. Basenames embed the year so cross-year collisions are
    effectively impossible; a dest collision is skipped + counted. test_pts/ is
    discarded (regenerated downstream). Returns {merged, collisions,
    year_dirs_removed}."""
    import glob as _glob
    import shutil as _shutil
    merged = {}
    year_dirs = []
    for prompts in sorted(_glob.glob(os.path.join(routed_dir, "*", "ids", "sam_click_prompts.json"))):
        year_dir = os.path.dirname(os.path.dirname(prompts))
        year_dirs.append(year_dir)
        with open(prompts) as f:
            merged.update(json.load(f))

    flat_ids = os.path.join(routed_dir, "ids")
    flat_raw = os.path.join(routed_dir, "raw")
    os.makedirs(flat_ids, exist_ok=True)
    os.makedirs(flat_raw, exist_ok=True)
    with open(os.path.join(flat_ids, "sam_click_prompts.json"), "w") as f:
        json.dump(merged, f, indent=1)

    collisions = 0
    for year_dir in year_dirs:
        raw_dir = os.path.join(year_dir, "raw")
        if os.path.isdir(raw_dir):
            for name in os.listdir(raw_dir):
                dst = os.path.join(flat_raw, name)
                if os.path.exists(dst):
                    collisions += 1
                    continue
                _shutil.move(os.path.join(raw_dir, name), dst)
        _shutil.rmtree(year_dir, ignore_errors=True)  # drops test_pts + emptied ids/raw

    if collisions:
        print(f"[flatten] WARNING: {collisions} raw filename collision(s) skipped")
    return {"merged": len(merged), "collisions": collisions,
            "year_dirs_removed": len(year_dirs)}


def _sam3_set(**kw):
    with sam3_lock:
        # Same lifecycle rule as _route_set: leaving the error phase clears the
        # process-death residue written by _reconcile_liveness.
        if kw.get("phase") not in (None, "error"):
            sam3_status.pop("died", None)
            sam3_status.pop("log_tail", None)
            if "error" not in kw:
                sam3_status["error"] = None
        sam3_status.update(kw)
        sam3_status["updated_at"] = datetime.now().isoformat()


# Routing driver state (step4test) - a background thread drives placePoints
# headlessly (CPC/OCR detection -> sam_click_prompts.json) before the combined
# annotator window opens. The frontend polls /api/step/step4test/route_status.
route_status = {
    "phase": "idle",       # idle | launching | routing_configure | routing_ocr |
                           # exporting | route_ready | ui_ready | error
    "processed": 0,
    "total": 0,
    "message": "",
    "ui_port": None,
    "error": None,
    # Routing accountability (Task 2): human-readable skip notices + the raw
    # per-frame dropped list from placePoints' configure response, surfaced in
    # the 4.test launch overlay. Reset on each fresh routing pass.
    "warnings": [],
    "dropped": [],
    "updated_at": None,
}
route_lock = threading.Lock()
route_thread = None

# Cancellation for an in-flight routing pass. The orchestrator UI is the ONLY way
# these tools are ever launched, so closing the launch window must mean the work
# stops: no GPU held by a routing sub-app nobody is watching, and no annotator
# kill+relaunch fired by a pass the operator walked away from. _route_drive polls
# this between every step of the pass; the routes that start a pass clear it.
#
# Cancelling is cheap: placePoints flushes detections.json every 50 frames and its
# /api/configure skips frames it already has (outcome "already_processed"), so a
# re-opened route resumes and only redoes the handful of frames since the last
# flush - it never restarts the whole OCR pass from zero.
route_cancel = threading.Event()


def _route_set(**kw):
    with route_lock:
        # Any transition to a non-error phase starts a new lifecycle: clear the
        # process-death residue (marker, log tail, stale error string) so an
        # old death cannot haunt a fresh launch or a healthy recovery.
        if kw.get("phase") not in (None, "error"):
            route_status.pop("died", None)
            route_status.pop("log_tail", None)
            if "error" not in kw:
                route_status["error"] = None
        route_status.update(kw)
        route_status["updated_at"] = datetime.now().isoformat()


def _reconcile_liveness(lock, status, alive_phases, runner_key, label):
    """A cached phase that claims a living process must be verified against
    that process at every read. The 2026-08-13 incident: the annotator died
    at SAM3 load after ui_ready was cached, and this API reported a dead port
    as "Combined annotator ready" for hours while the UI polled it. Flip to
    error (written back under the lock) the moment the process is gone; a
    dead process cannot come back, so the flip is safe and idempotent.

    Ownership: route_status is SHARED by three launch modes (step4test,
    step4loop, editmasks), so the phase setter stamps runner_key/runner_label
    and this check verifies the process that actually owns the current phase;
    runner_key/label arguments are only the fallback for pre-stamp states.

    Races: the runner poll happens outside the status lock (poll_status takes
    the runner's own lock; holding both invites ordering deadlocks), so the
    final write re-checks phase+updated_at and yields to any transition that
    landed in between (a finishing relaunch, a deliberate Stop)."""
    with lock:
        phase = status.get("phase")
        seen_at = status.get("updated_at")
        owner = status.get("runner_key") or runner_key
        owner_label = status.get("runner_label") or label
    if phase not in alive_phases:
        with lock:
            return dict(status)
    if runner.is_running(owner):
        with lock:
            return dict(status)
    proc = runner.poll_status(owner)
    if proc.get("running"):
        with lock:
            return dict(status)
    exit_code = proc.get("exit_code")
    detail = (f"exited (code {exit_code})" if exit_code is not None
              else "is not running (never started or already cleaned up)")
    with lock:
        if (status.get("phase"), status.get("updated_at")) != (phase, seen_at):
            return dict(status)
        status.update(
            phase="error", ui_port=None, died=True,
            error=f"{owner_label} process {detail}",
            message="The process died. See the log tail below, then relaunch.",
            log_tail=list(proc.get("log_tail") or [])[-12:],
        )
        status["updated_at"] = datetime.now().isoformat()
        return dict(status)


def _route_mark_stopped_if_owner(runner_key, message):
    """Deliberate Stop/Reset must never read as a crash, but only the mode
    that OWNS the current ui_ready may rewrite it: step4test's Stop killing
    its own keys must not claim an Edit Masks or Refine session stopped."""
    with route_lock:
        is_owner = (route_status.get("phase") == "ui_ready"
                    and (route_status.get("runner_key") or "step4test") == runner_key)
    if is_owner:
        _route_set(phase="stopped", ui_port=None, error=None, message=message)


def _sam3_mark_stopped(message):
    """Deliberate-Stop guard for the step-5 surface, mirroring
    _route_mark_stopped_if_owner: without this, stopping or completing step 5
    leaves review_ready cached and the liveness check reports the kill as a
    crash."""
    with sam3_lock:
        alive = sam3_status.get("phase") in ("review_ready", "segmenting",
                                             "configuring",
                                             "waiting_for_sub_app")
    if alive:
        _sam3_set(phase="stopped", running=False, error=None, message=message)


def _reset_transient_status(message):
    """Project open/quit invalidates every cached claim about running
    processes: without this, project B's tile inherits project A's ui_ready
    and port (2026-08-14 critic gap). Deliberate lifecycle event, so plain
    idle, never a crash error."""
    _route_set(phase="idle", ui_port=None, error=None, message=message)
    _sam3_set(phase="idle", running=False, error=None, message="")
    # The annotator launch dicts remember the LAST launch's out_dir/project.
    # Left alone across a project switch, project B's "Open Folder" resolved
    # project A's export dir and the path guard refused it with 403
    # (2026-08-26, "I can not open folder"). Forget the binding; keep the
    # process claims (running/port) for the liveness reconcile.
    for st in (step4test_state, step4loop_state, editmasks_state):
        st.update(out_dir=None, project_id=None, input_dir=None)


def _gui_env():
    """Thin delegate: the host GUI environment scrub lives in host_open.py so
    the orchestrator's folder routes and the coverage-matrix blueprint share
    one implementation (2026-08-26)."""
    return host_open.gui_env()


def _open_in_file_manager(path):
    """Thin delegate to host_open.open_in_file_manager(path) -> (ok, err)."""
    return host_open.open_in_file_manager(path)


def _reviewer_initials():
    """Reviewer identity for annotator launches (spec D2): the latest
    access_log initials of the loaded project, else its creator initials;
    '' when no project is loaded or nothing is recorded."""
    if current_project is None:
        return ""
    try:
        return pm.reviewer_for_state(current_project)
    except Exception:
        return ""


def _post_reviewer(port, reviewer, timeout=2.0):
    """Best-effort POST /api/reviewer {reviewer} to an annotator on ``port``.
    Annotator processes are reused across opens, so the launch env alone
    cannot carry a changed reviewer; this call updates a live one. Never
    raises; returns True only on a 2xx answer."""
    if not port or not reviewer:
        return False
    try:
        req = Request(f"http://127.0.0.1:{int(port)}/api/reviewer",
                      data=json.dumps({"reviewer": reviewer}).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=timeout) as resp:
            return 200 <= int(getattr(resp, "status", 0) or 0) < 300
    except Exception:
        return False


def _push_reviewer_after_health(port, reviewer, health_fn, *, deadline_s=240,
                                interval_s=2.0, background=True):
    """Wait until ``health_fn(port)`` is true, then POST the reviewer once.
    A fresh annotator loads SAM3 for about a minute, so the wait runs on a
    daemon thread by default; ``background=False`` runs it inline (tests).
    Gives up silently at the deadline; never raises."""
    if not port or not reviewer:
        return None

    def _work():
        end = time.time() + deadline_s
        while time.time() < end:
            try:
                if health_fn(port):
                    _post_reviewer(port, reviewer)
                    return
            except Exception:
                pass
            time.sleep(interval_s)

    if not background:
        _work()
        return None
    t = threading.Thread(target=_work, daemon=True, name=f"reviewer-push-{port}")
    t.start()
    return t


def _composite_root():
    """Where composite YOLO exports live: <module>/inprocess/_composite/.
    Own function (not inlined) so tests can point it at a temp dir."""
    return os.path.join(os.path.dirname(REPO_DIR), "inprocess", "_composite")


def _validate_composite_source(source_dir):
    """Spec D5 guard for Step 6's training source. Returns (path, '') when
    ``source_dir`` resolves to a directory under _composite_root() that holds
    all_images/ and data.yaml; else ('', reason)."""
    root = os.path.realpath(_composite_root())
    path = os.path.realpath(str(source_dir or "").strip())
    if not path.startswith(root + os.sep):
        return "", f"training source must be a composite export under {root}: {path}"
    if not os.path.isdir(path):
        return "", f"training source is not a directory: {path}"
    if not os.path.isdir(os.path.join(path, "all_images")):
        return "", f"training source is missing all_images/: {path}"
    if not os.path.isfile(os.path.join(path, "data.yaml")):
        return "", f"training source is missing data.yaml: {path}"
    return path, ""


def _standalone_root():
    """Where Annotate Images sessions live: <module>/inprocess/_standalone/.

    Returns:
        The folder that holds one sub-folder per standalone session. Own
        function so tests can point it at a temp dir.
    """
    return os.path.join(os.path.dirname(REPO_DIR), "inprocess", "_standalone")


def _validate_standalone_source(source_dir):
    """Decision record 0006: an Annotate Images session export as a training source.

    Parameters:
        source_dir: the folder the Train step was pointed at.
    Returns:
        (path, '') when source_dir resolves to a folder under _standalone_root()
        that holds all_images/ and data.yaml; else ('', reason).
    """
    root = os.path.realpath(_standalone_root())
    path = os.path.realpath(str(source_dir or "").strip())
    if not _contained(path, root) or path == root:
        return "", "training source must be an Annotate Images session export under the standalone folder"
    if not os.path.isdir(path):
        return "", f"training source is not a directory: {path}"
    if not os.path.isdir(os.path.join(path, "all_images")):
        return "", f"training source is missing all_images/: {path}"
    if not os.path.isfile(os.path.join(path, "data.yaml")):
        return "", f"training source is missing data.yaml: {path}"
    return path, ""


def _validate_outline_source(source_dir):
    """Validate a training source that holds outlines: a training set or a session.

    Parameters:
        source_dir: the folder the Train step was pointed at.
    Returns:
        (path, '') or ('', reason), from _validate_standalone_source when the
        folder sits under the standalone root, else from
        _validate_composite_source.
    """
    real = os.path.realpath(str(source_dir or "").strip())
    if _contained(real, os.path.realpath(_standalone_root())):
        return _validate_standalone_source(source_dir)
    return _validate_composite_source(source_dir)


def _count_polygon_lines(labels_dir):
    """Count the label lines across a YOLO labels folder.

    Parameters:
        labels_dir: the folder of .txt label files.
    Returns:
        The number of non-empty lines over every .txt file, one per outline;
        0 when the folder cannot be listed. A file that cannot be read is
        skipped rather than stopping the count.
    """
    total = 0
    try:
        names = os.listdir(labels_dir)
    except OSError:
        return 0
    for name in names:
        if not name.endswith(".txt"):
            continue
        try:
            with open(os.path.join(labels_dir, name), "r", encoding="utf-8", errors="ignore") as fh:
                total += sum(1 for line in fh if line.strip())
        except OSError:
            continue
    return total


def _list_standalone_exports():
    """Annotate Images sessions that hold a YOLO export, newest first.

    Returns:
        A list of source rows in the shape /api/composites uses: name (the
        session folder), path (the export folder), kind "standalone", task
        "seg", n_images (files in all_images/), n_masks (polygon lines in
        all_labels/), created (AST). A session without data.yaml and
        all_images/ is not an export and is left out.
    """
    root = _standalone_root()
    real_root = os.path.realpath(root)
    rows = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return rows
    for name in names:
        session = os.path.join(root, name)
        # A session that is a link pointing outside the root is not offered:
        # the launch would refuse it, so listing it would only show a choice
        # that cannot be made.
        if not os.path.isdir(session) or not _contained(os.path.realpath(session), real_root):
            continue
        if not os.path.isfile(os.path.join(session, "data.yaml")):
            continue
        images_dir = os.path.join(session, "all_images")
        if not os.path.isdir(images_dir):
            continue
        try:
            n_images = sum(1 for f in os.listdir(images_dir) if not f.startswith("."))
        except OSError:
            n_images = 0
        record = _record_json(os.path.join(session, "session.json"))
        created_at = _text(record.get("created"))
        if not created_at:
            try:
                created_at = _ast_stamp(os.path.getmtime(session))
            except OSError:
                created_at = ""
        rows.append({"name": name, "path": session, "kind": "standalone", "task": "seg",
                     "n_images": n_images, "n_masks": _count_polygon_lines(os.path.join(session, "all_labels")),
                     "labels": _data_yaml_names(os.path.join(session, "data.yaml")),
                     "created_at": created_at, "reviewer": _text(record.get("initials")) or _text(record.get("reviewer")),
                     "origin": "Annotate Images session"})
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows

def _data_yaml_names(path):
    """Class names from a YOLO data.yaml (list or {id: name} mapping); [] when
    unreadable. PyYAML when present, else a tolerant line parser."""
    try:
        import yaml  # noqa: WPS433 (optional dependency)
        with open(path) as fh:
            doc = yaml.safe_load(fh) or {}
        names = doc.get("names")
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
        if isinstance(names, list):
            return [str(n) for n in names]
        return []
    except Exception:
        pass
    names = []
    try:
        in_names = False
        for line in open(path):
            if line.startswith("names:"):
                in_names = True
                tail = line.split(":", 1)[1].strip()
                if tail.startswith("["):
                    return [t.strip().strip("'\"") for t in tail.strip("[]").split(",") if t.strip()]
                continue
            if in_names:
                mm = re.match(r"^\s*(?:-|\d+:)\s*(.+?)\s*$", line)
                if mm:
                    names.append(mm.group(1).strip("'\""))
                elif line.strip():
                    break
    except Exception:
        return []
    return names


def _step6_is_unnamed_class(name):
    """True when a Step 6 class name should be dropped from training: blank,
    or the literal "(unnamed)" (case-insensitive, any surrounding
    whitespace). Shared by /api/step/6/list_classes, the Start-Training
    guard, and the training-data summary, so the three agree on what an
    unnamed class is."""
    if name is None:
        return True
    s = str(name).strip()
    if not s:
        return True
    return re.match(r"^\(unnamed\)$", s, flags=re.IGNORECASE) is not None


def _data_yaml_id_names(path):
    """{class_id: name} from a YOLO data.yaml (list or {id: name} mapping);
    {} when unreadable. Ids are kept as written so a mapping with gaps
    ({0: MC, 2: PA}) resolves to the ids the label files carry."""
    try:
        import yaml  # noqa: WPS433 (optional dependency)
        with open(path) as fh:
            doc = yaml.safe_load(fh) or {}
        names = doc.get("names")
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}
        return {}
    except Exception:
        pass
    out = {}
    try:
        in_names = False
        for line in open(path):
            if line.startswith("names:"):
                in_names = True
                tail = line.split(":", 1)[1].strip()
                if tail.startswith("["):
                    items = [t.strip().strip("'\"") for t in tail.strip("[]").split(",") if t.strip()]
                    return {i: n for i, n in enumerate(items)}
                continue
            if in_names:
                mm = re.match(r"^\s*(?:-|(\d+):)\s*(.+?)\s*$", line)
                if mm:
                    cid = int(mm.group(1)) if mm.group(1) is not None else len(out)
                    out[cid] = mm.group(2).strip("'\"")
                elif line.strip():
                    break
    except Exception:
        return {}
    return out


def _translate_class_ids(ids, src_names, dst_names):
    """Carry a class-id selection from one YOLO export to another by NAME
    (H2, 2026-08-26): ids chosen against ``src_names`` ({id: name}) become
    the ids the same names carry in ``dst_names``. Returns
    (mapped_ids, dropped_names); a name the target lacks is dropped and
    reported, never guessed, and an id with no source name drops as '#<id>'.
    Names compare case-insensitively after stripping."""
    by_name = {}
    for cid, nm in (dst_names or {}).items():
        key = str(nm).strip().lower()
        if key and key not in by_name:
            by_name[key] = int(cid)
    mapped, dropped = [], []
    for raw in ids or []:
        try:
            cid = int(raw)
        except (TypeError, ValueError):
            continue
        nm = (src_names or {}).get(cid)
        key = str(nm).strip().lower() if nm is not None else ""
        if not key:
            dropped.append(f"#{cid}")
        elif key in by_name:
            if by_name[key] not in mapped:
                mapped.append(by_name[key])
        else:
            dropped.append(str(nm).strip())
    return mapped, dropped


def _class_picker_source(include_classes_source):
    """Directory the Step 6 class picker was loaded from. Blank, and a legacy
    config with no include_classes_source at all, mean this project's Step 4
    export; anything else is a composite export validated exactly like the
    training source. Returns (dir, error)."""
    src = str(include_classes_source or "").strip()
    if not src:
        return _step4test_export_dir(), ""
    return _validate_outline_source(src)


def _step6_picker_names(picker_dir, picker_src):
    """{id: name} for the source the class picker was loaded from. The
    project export falls back to Step 3's target_species roster, the same
    roster /api/step/6/list_classes shows before Step 4 has exported."""
    yaml_path = os.path.join(picker_dir, "data.yaml") if picker_dir else ""
    if yaml_path and os.path.isfile(yaml_path):
        return _data_yaml_id_names(yaml_path)
    if picker_src:
        return {}
    s3_cfg = ((current_project or {}).get("steps", {}).get("3", {}) or {}).get("config", {}) or {}
    species = [s.strip() for s in str(s3_cfg.get("target_species") or "").split(",") if s.strip()]
    return {i: sp for i, sp in enumerate(species)}


def _list_composites():
    """Composite exports under _composite_root(), newest first: the same
    shape the matrix blueprint's /matrix/api/composites returns
    ({name, path, created_at, n_images, n_masks, labels})."""
    root = _composite_root()
    out = []
    if not os.path.isdir(root):
        return out
    real_root = os.path.realpath(root)
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d) or not os.path.isfile(os.path.join(d, "data.yaml")):
            continue
        # Mirror _validate_composite_source: an entry whose real path sits
        # outside the composite root (a symlink out) would be offered in the
        # Step 6 select and then refused at run time, so skip it up front.
        if not os.path.realpath(d).startswith(real_root + os.sep):
            continue
        imgs = os.path.join(d, "all_images")
        lbls = os.path.join(d, "all_labels")
        n_images = len(os.listdir(imgs)) if os.path.isdir(imgs) else 0
        n_masks = 0
        if os.path.isdir(lbls):
            for fn in os.listdir(lbls):
                if fn.endswith(".txt"):
                    try:
                        with open(os.path.join(lbls, fn)) as fh:
                            n_masks += sum(1 for ln in fh if ln.strip())
                    except Exception:
                        pass
        created_at = ""
        # Prefer the build stamp the matrix wrote (composite.json created_at,
        # already 'YYYY-MM-DD HH:MM:SS' AST) so both listings agree; fall back
        # to data.yaml's mtime for a composite written without it.
        try:
            with open(os.path.join(d, "composite.json")) as fh:
                created_at = str((json.load(fh) or {}).get("created_at") or "")[:19]
        except Exception:
            created_at = ""
        if not created_at:
            try:
                created = datetime.fromtimestamp(os.path.getmtime(os.path.join(d, "data.yaml")), _AST)
                created_at = created.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                created_at = ""
        out.append({"name": name, "path": d, "created_at": created_at,
                    "n_images": n_images, "n_masks": n_masks,
                    "labels": _data_yaml_names(os.path.join(d, "data.yaml"))})
    out.sort(key=lambda c: c.get("created_at", ""), reverse=True)
    return out


# ---------------------------------------------------------------------------
# Cross-project reads: allowed roots, path containment, timestamps
# ---------------------------------------------------------------------------
# Every path a client hands to a read route (a run folder, an evaluation
# folder, an inference output folder) must sit under one of two roots: the
# open project's folder and the module's inprocess/ folder, which also holds
# the model labs under _models/. Containment is checked with
# os.path.commonpath on real paths, so neither a ".." segment, a symbolic
# link, nor a sibling folder whose name merely extends an allowed one can
# reach outside. Rows and files read from disk are data written by earlier
# runs, never instructions.

# Names the evaluator (TCRMPtrain_oceankindCV/src/evaluate_run.py) and the
# inference tool (src/run_inference.py) write. Change them here when a
# writer renames them.
REPORT_MD_NAME = "report.md"
METRICS_JSON_NAME = "metrics.json"
REPORT_PDF_NAME = "report.pdf"
PREVIEWS_DIRNAME = "previews"
VAL_RUN_DIRNAME = "val_run"
MANIFEST_JSON_NAME = "manifest.json"
VERIFICATIONS_CSV_NAME = "verifications.csv"
ROUNDS_CSV_NAME = "rounds.csv"
WEIGHTS_REL = os.path.join("weights", "best.pt")
# An evaluation launched from Step 6 writes <step7>/eval_<run name>; a
# folder stamped by a later writer may carry _YYYYMMDD_HHMMSS after it, and
# the lab evaluator writes _models/_eval/<run name>_<stamp>.
EVAL_FOLDER_PREFIX = "eval_"
_EVAL_STAMP_PATTERN = r"\d{8}_\d{6}"
# The figures the report card shows, as (contract key, file under val_run/).
REPORT_FIGURES = (
    ("box_pr", "BoxPR_curve.png"),
    ("box_f1", "BoxF1_curve.png"),
    ("mask_pr", "MaskPR_curve.png"),
    ("mask_f1", "MaskF1_curve.png"),
    ("confusion_norm", "confusion_matrix_normalized.png"),
    ("confusion_abs", "confusion_matrix.png"),
)
# ultralytics writes the training curves into the run folder itself. The
# step 7 image route reads exactly these files from the run folder, under
# the "run/" prefix; everything else it serves comes from the evaluation
# folder.
RUN_FIGURE_REL_PREFIX = "run/"
RUN_FIGURES = (("results", "results.png"),)
METRICS_KEYS = ("overall", "per_class", "per_class_instances", "absent_classes",
                "metric_family", "class_names", "class_full_names")
# A metrics.json from before 2026-09-04 carries no metric_family: it is
# mask when any mask number under overall is present, else box.
_MASK_OVERALL_PREFIX = "mask_"
STEP7_IMAGE_ROUTE = "/api/step/7/image"
STEP7_PDF_ROUTE = "/api/step/7/pdf"
STEP8_IMAGE_ROUTE = "/api/step/8/image"
# Last-resort evaluation image size when the run's args.yaml records none.
EVAL_IMGSZ_FALLBACK = 1024


def _above_measurement_ceiling(conf):
    """Whether a confidence cut-off is too high for an honest measurement.

    Parameters:
        conf: the value the panel holds, which a hand-edited project.json can
            make a string or something that is not a number at all.
    Returns:
        True when it reads as a number above EVAL_CONF_MEASUREMENT_CEILING.
        A value that is not a number returns False: the evaluator's own
        command line refuses it, and a warning about a number nobody typed
        would only confuse.
    """
    try:
        return float(conf) > EVAL_CONF_MEASUREMENT_CEILING
    except (TypeError, ValueError):
        return False


def _eval_default(key, cast):
    """One Evaluate-step default, read from config/pipeline.yaml step_defaults.

    The panel's Use recommended defaults answer, the launch fallback and the
    default a new project is created with all read this one place, so they
    cannot drift apart.

    Parameters:
        key: the step_defaults key under "7", for example "conf_threshold".
        cast: float or int, the type the evaluator's command line takes.
    Returns:
        The cast value.
    Raises:
        RuntimeError naming the config file and the key when it is absent or
        is not a number.
    """
    raw = DEFAULT_STEP_CONFIGS.get("7", {}).get(key)
    try:
        return cast(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"step_defaults '7'.{key} in {CONFIG_PATH} is {raw!r}, which is not a number; "
            f"the Evaluate panel cannot offer a default it cannot read"
        ) from exc


# What the Evaluate step measures with. Confidence sits at 0.001 because mean
# average precision is the area under the whole precision-recall curve: a
# higher cut-off removes the low-confidence tail the curve needs and reports a
# score lower than the weights earn. Step 8 (inference) keeps its own, higher,
# deployment threshold and is not read from here.
EVAL_CONF_DEFAULT = _eval_default("conf_threshold", float)
EVAL_IOU_DEFAULT = _eval_default("iou_threshold", float)
EVAL_PREVIEW_COUNT_DEFAULT = _eval_default("preview_count", int)
EVAL_SPLIT_DEFAULT = str(DEFAULT_STEP_CONFIGS.get("7", {}).get("split") or "test")
# The settings the Evaluate panel's Use recommended defaults button offers,
# in one place, so the button, the value a new project is created with, the
# launch fallback and the "your saved value differs" note can never disagree.
EVAL_RECOMMENDED = {
    "split": EVAL_SPLIT_DEFAULT,
    "conf_threshold": EVAL_CONF_DEFAULT,
    "iou_threshold": EVAL_IOU_DEFAULT,
    "preview_count": EVAL_PREVIEW_COUNT_DEFAULT,
}
# How close two of those settings count as the same value. Both are read
# back from JSON, so this absorbs a round trip and nothing else.
EVAL_SETTING_TOLERANCE = 1e-9
# Above this confidence an evaluation has stopped measuring the whole
# precision-recall curve and is reporting a deployment score instead, which
# reads lower than the weights earn. A project created before 2026-09-15 saved
# 0.25 into its own config and keeps it until the panel is refilled, so the
# launch says so rather than letting the number pass unremarked.
EVAL_CONF_MEASUREMENT_CEILING = 0.01
# The confidence a project saved into its own step 7 config while the panel's
# default was a deployment threshold. pm.load_project grafts defaults only
# onto a step block that is missing altogether, so this number survives every
# reopen: six of the seven runs under inprocess/ still held it at
# 2026-09-15 20:30 AST, the Coral Intel subset project included, because the
# running orchestrator grafts the defaults it read at startup. Opening a
# project replaces exactly
# this value, which is exactly representable in binary, so the comparison
# below is an equality. Any other value the operator chose is left alone.
RETIRED_EVAL_CONF = 0.25
# project_dir -> the confidence that was replaced when it was opened, so the
# Evaluate panel can say what changed and why. Kept in memory only: nothing
# here is written into project.json, which holds exactly the keys the minter
# and the panel wrote.
_EVAL_CONF_UPGRADES = {}


def _upgrade_retired_eval_conf(state):
    """Replace an Evaluate confidence saved before the measurement default was fixed.

    A score measured at RETIRED_EVAL_CONF leaves the low-confidence tail of
    the predictions out of the mean average precision and reads about 0.06
    mAP50 below what the same weights earn (measured on
    TRAIN_coral_intel_20260902_224223, 2026-09-15). Without this the panel
    would open on that number while the tooltip beside it says to measure at
    EVAL_CONF_DEFAULT, and the multi-minute evaluation would be away before
    the launch warning could be read.

    Parameters:
        state: a project dict as pm.load_project returns it.
    Returns:
        The same dict, with steps.7.config.conf_threshold replaced and saved
        to project.json when it held exactly the retired value. A value that
        is absent, chosen, or not a number at all is left untouched.
    """
    config = ((state.get("steps") or {}).get("7") or {}).get("config")
    if not isinstance(config, dict):
        return state
    try:
        saved = float(config.get("conf_threshold"))
    except (TypeError, ValueError):
        # A hand-edited project.json can hold anything. The evaluator's own
        # command line refuses a confidence that is not a number, by name, so
        # nothing is guessed here.
        return state
    if saved != RETIRED_EVAL_CONF:
        return state
    config["conf_threshold"] = EVAL_CONF_DEFAULT
    _EVAL_CONF_UPGRADES[os.path.abspath(state.get("project_dir") or "")] = RETIRED_EVAL_CONF
    pm.save_project(state)
    vic.note(
        f"[orch] {step_ref(7)}: the saved confidence {RETIRED_EVAL_CONF} was replaced with "
        f"{EVAL_CONF_DEFAULT} in project '{state.get('name', '')}', so the score keeps the "
        f"whole precision-recall curve"
    )
    return state


def _differs_from_recommended(key, value):
    """Whether a saved Evaluate setting is not the value VEGA recommends.

    Parameters:
        key: the step 7 config key, one of EVAL_RECOMMENDED.
        value: what the project has saved for it.
    Returns:
        True when the two differ. A numeric pair is compared within
        EVAL_SETTING_TOLERANCE; anything that is not a number is compared as
        text, so a hand-edited project.json reads as different rather than
        raising. False for a key VEGA recommends nothing for.
    Raises:
        nothing: a value of any type is handled.
    """
    if key not in EVAL_RECOMMENDED:
        return False
    recommended = EVAL_RECOMMENDED[key]
    if isinstance(value, bool) or isinstance(recommended, bool):
        return value != recommended
    try:
        return abs(float(value) - float(recommended)) > EVAL_SETTING_TOLERANCE
    except (TypeError, ValueError):
        return str(value) != str(recommended)


def _eval_settings_off_recommended(state):
    """Every Evaluate setting a project has saved that is not the recommended one.

    The Evaluate panel opens on the project's own saved settings, and a
    project created before those defaults were fixed still holds the old
    overlap of 0.7 while the copy beside the field says 0.6 is standard. The
    ledger records what each score was measured with, so a value that differs
    is not a preference, it decides which other runs the score may be ranked
    against. This is what the panel says so out of.

    Parameters:
        state: a project dict, or None when no project is open.
    Returns:
        {key: {"saved": value, "recommended": value}} for each differing key,
        {} when every saved setting matches or the project has no step 7
        config yet.
    """
    config = ((state or {}).get("steps") or {}).get("7") or {}
    config = config.get("config") if isinstance(config, dict) else None
    if not isinstance(config, dict):
        return {}
    off = {}
    for key, recommended in EVAL_RECOMMENDED.items():
        if key not in config:
            continue
        if _differs_from_recommended(key, config[key]):
            off[key] = {"saved": config[key], "recommended": recommended}
    return off


# How the report routes say a run has not been evaluated; the report card
# renders a 404 carrying "ready": false as its one-line empty state.
NO_EVALUATION_PREFIX = "no evaluation found for"
# What an inference run's source_type means, in words, for the run list.
_INFERENCE_DIR_SOURCE = "dir"
_INFERENCE_SOURCE_LABELS = {"unused": "unused TCRMP frames", "full": "full TCRMP set"}
VERDICTS = ("yes", "no", "skip")
# Ledger column suffix per metric family (map50_M, map50_95_B, ...).
_FAMILY_SUFFIX = {_labruns.METRIC_FAMILY_MASK: "M", _labruns.METRIC_FAMILY_BOX: "B"}
# Where a run's scores in the runs card came from: its rounds.csv row, or the
# newest evaluation folder on disk when no ledger row names the run. Only a
# run with a ledger row may be promoted (POST /api/step/7/promote), so the
# card reads this to say why a scored run still cannot be made champion.
SCORE_SOURCE_LEDGER = "ledger"
SCORE_SOURCE_EVALUATION = "evaluation"
# Why a run cannot be ranked against the champion, verbatim for the runs card.
REASON_DIFFERENT_FAMILY = "different metric family"
REASON_SPLIT_NOT_RECORDED = "split not recorded"
REASON_DIFFERENT_SPLIT = "different held-out images"
REASON_MEASUREMENT_NOT_RECORDED = "measurement settings not recorded"
REASON_DIFFERENT_MEASUREMENT = "different measurement settings"
# What a score was measured with, as the ledger's own column names. mean
# average precision is the area under the whole precision-recall curve, so
# the same weights score differently at another confidence cut-off, at
# another non-maximum-suppression overlap, and at another image size. Two
# rows are a comparison only when all three agree, which is why every one of
# them is read here and not just the confidence.
# Byte-identical to evaluate_run.py's ROUNDS_SETTINGS_COLUMNS, which writes
# the cells this reads.
MEASUREMENT_COLUMNS = ("conf", "iou", "imgsz")
# Two recorded confidences (or overlaps) count as one measurement within this
# much, which absorbs a CSV round trip and nothing else. Same tolerance as
# evaluate_run.py's CONF_MATCH_TOLERANCE.
MEASUREMENT_MATCH_TOLERANCE = 1e-9
# The gate columns evaluate_run.py writes into every ledger row, and what to
# call each one in a sentence a reader sees. A verdict is "pass", "fail",
# "flag" (optionally followed by ':' and the species codes that dropped),
# "unpinned" (the split was not pinned, so the gates could not be trusted) or
# "incomparable" (the baseline was never a comparison). A cell that was never
# written is GATE_NOT_RECORDED.
GATE_COLUMNS = ("gate_map", "gate_class", "gate_recall")
GATE_LABELS = {
    "gate_map": "the mean average precision gate",
    "gate_class": "the per-class gate",
    "gate_recall": "the recall gate",
}
# What a cell that was never written reads as. Not a verdict: a ledger that
# recorded nothing has stated no failure.
GATE_NOT_RECORDED = ""
# The verdicts that make promotion a decision rather than a formality, each
# matched on its first word so "flag:PA,MC" counts as a flag. "fail" says the
# score went backwards against the baseline, "flag" names classes or a recall
# that dropped, and "incomparable" says the two rounds were never a
# comparison. "unpinned" is deliberately not one of them: it says the split
# was loose so the gates never ran, which is the state of almost every ledger
# written before splits were pinned, and refusing on it would make every one
# of those runs unpromotable without saying anything about the run itself.
GATE_BLOCKING_VERDICTS = ("fail", "flag", "incomparable")
# Files a staged box dataset must carry to count as finished, the same rule
# labruns.list_bbox_sources applies when it offers one.
_BBOX_SOURCE_FILES = ("data.yaml", BBOX_MANIFEST_NAME)


def _inprocess_root():
    """The module's inprocess/ folder, holding every project and the model
    labs (_models/). Read from pipeline.yaml paths.projects_dir, else the
    sibling of the repo. Own function so tests can point it at a fixture."""
    return PATHS.get("projects_dir") or os.path.join(os.path.dirname(REPO_DIR), "inprocess")


def _bbox_root():
    """Where the Bounding Boxes page stages detection datasets
    (supporting_data/bbox_datasets under the repo). Own function so tests can
    point it at a fixture."""
    return os.path.join(REPO_DIR, "supporting_data", "bbox_datasets")


def _allowed_roots():
    """Real paths of the folders a client-supplied path may sit under: the
    open project's folder (when one is open) and the inprocess root."""
    roots = []
    if current_project is not None and current_project.get("project_dir"):
        roots.append(os.path.realpath(current_project["project_dir"]))
    roots.append(os.path.realpath(_inprocess_root()))
    return roots


def _contained(real_path, real_root):
    """True when real_path is real_root or sits beneath it, both already
    resolved with os.path.realpath. A sibling whose name extends the root's
    name is not contained, which is what a startswith test got wrong."""
    try:
        return os.path.commonpath([real_root, real_path]) == real_root
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Path containment (2026-09-17, LO). VEGA answers on 0.0.0.0 with no login,
# so a route that reads or writes wherever a request points is an open door
# to every machine on the lab network. Reads stay inside the module folder;
# the PDF export keeps its real use (a folder Lauren chooses) but only under
# a root VEGA is allowed to write to.
# ---------------------------------------------------------------------------

EXPORT_ROOTS_ENV = "VEGA_EXPORT_ROOTS"
_PATH_UP = ".."


def _module_root():
    """The module folder that holds github_repo/ and inprocess/, resolved.

    Returns:
        The real path of the parent of the inprocess root. Every file a route
        reads on a request's behalf must sit under it.
    """
    return os.path.realpath(os.path.dirname(os.path.normpath(_inprocess_root())))


def _export_roots():
    """Folders the PDF export may write under.

    Returns:
        A list of real absolute paths: the module folder, the VICARIUS drive
        that holds it, the home directory, and every absolute entry of the
        colon-separated VEGA_EXPORT_ROOTS variable (for NAS shares). A
        relative entry is ignored rather than resolved against the working
        directory, which would move with the process.
    """
    module = _module_root()
    drive = os.path.dirname(os.path.dirname(os.path.dirname(module)))
    candidates = [module, drive, os.path.realpath(os.path.expanduser("~"))]
    for entry in os.environ.get(EXPORT_ROOTS_ENV, "").split(os.pathsep):
        entry = entry.strip()
        if entry and os.path.isabs(entry):
            candidates.append(os.path.realpath(entry))
    roots = []
    for root in candidates:
        # A root of "/" would allow every path on the box: an empty HOME makes
        # expanduser("~") return "/", and a typo can put "/" in the variable.
        if os.path.realpath(root) == os.sep:
            vic.note(f"[orch] " + ("export root %r resolves to the filesystem root and is ignored" % root))
            continue
        roots.append(root)
    return roots


def _real_destination(raw):
    """Resolve a destination the way the kernel will, before it exists.

    Parameters:
        raw: the folder a request named.
    Returns:
        (real_path, '') when raw is an absolute path whose deepest existing
        ancestor resolves cleanly and whose unborn tail holds no '..'; else
        ('', reason). Resolving the existing part through symlinks first
        means a link that points outside a root is seen as where it points,
        and refusing '..' in the tail means a folder that does not exist yet
        cannot climb back out.
    """
    if not isinstance(raw, str) or not raw.strip():
        return "", "a folder path is required"
    if "\x00" in raw:
        return "", "the folder path holds a null byte"
    if not os.path.isabs(raw):
        return "", "the folder must be an absolute path"
    if _PATH_UP in raw.split(os.sep):
        # Checked on the raw segments: normpath would collapse a ".." away, but
        # a request that climbs is refused rather than quietly straightened.
        return "", "the folder path may not climb with '..'"
    parts = [seg for seg in os.path.normpath(raw).split(os.sep) if seg]
    existing = os.sep
    tail = []
    for index in range(len(parts), -1, -1):
        candidate = os.sep + os.sep.join(parts[:index])
        if os.path.exists(candidate):
            existing, tail = candidate, parts[index:]
            break
    return os.path.join(os.path.realpath(existing), *tail), ""


def _contained_in_any(real_path, roots):
    """Find which allowed root a resolved path sits under.

    Parameters:
        real_path: a path already resolved with os.path.realpath.
        roots: real paths a route may use.
    Returns:
        The first root that contains real_path, or '' when none does.
    """
    for root in roots:
        if _contained(real_path, root):
            return root
    return ""


def _read_path_inside_module(raw, what):
    """Admit a path a route will read only when it sits inside the module.

    Parameters:
        raw: the path a request named.
        what: what the path is, for the refusal ('a remap log', 'a run folder').
    Returns:
        (real_path, '', 0) when admitted; else ('', reason, http_status).
    """
    if not isinstance(raw, str) or not raw.strip():
        return "", f"{what} path is required", HTTP_BAD_REQUEST
    if "\x00" in raw:
        return "", f"{what} path holds a null byte", HTTP_BAD_REQUEST
    root = _module_root()
    real = os.path.realpath(raw)
    if not _contained(real, root):
        return "", f"{what} is read only from inside the module folder", HTTP_FORBIDDEN
    return real, "", 0

def _validate_client_dir(raw, what):
    """(absolute path, '', 0) when raw names a folder inside an allowed root.

    Parameters:
        raw: the path as the client sent it.
        what: the parameter name for the error text.
    Returns:
        ('', reason, status) otherwise: 400 for an empty value, 403 when the
        real path sits outside every allowed root, 404 when the folder does
        not exist.
    """
    clean = str(raw or "").strip()
    if not clean:
        return "", f"{what} is required", HTTP_BAD_REQUEST
    path = os.path.abspath(clean)
    real = os.path.realpath(path)
    if not any(_contained(real, root) for root in _allowed_roots()):
        return "", (f"{what} is outside the folders this window may read (the open "
                    f"project and {_inprocess_root()}): {path}"), HTTP_FORBIDDEN
    if not os.path.isdir(path):
        return "", f"{what} not found: {path}", HTTP_NOT_FOUND
    return path, "", 0


def _validate_rel(rel):
    """(rel, '', 0) when rel may name a file inside a served folder; else
    ('', reason, status): 400 when empty, 403 when absolute or holding a
    '..' segment."""
    clean = str(rel or "").strip()
    if not clean:
        return "", "rel (the file's path inside the folder) is required", HTTP_BAD_REQUEST
    if os.path.isabs(clean) or ".." in clean.split("/"):
        return "", f"rel must be a relative path inside the folder with no '..' segment: {clean}", HTTP_FORBIDDEN
    return clean, "", 0


def _file_inside(folder, rel):
    """(full path, '', 0) for the regular file rel under folder, resolved
    through symbolic links and confined to the folder; else ('', reason,
    status): 403 when it resolves outside, 404 when it is not a file."""
    real_root = os.path.realpath(folder)
    full = os.path.realpath(os.path.join(folder, rel))
    if not _contained(full, real_root):
        return "", f"{rel} resolves outside {folder}", HTTP_FORBIDDEN
    if not os.path.isfile(full):
        return "", f"no such file under {folder}: {rel}", HTTP_NOT_FOUND
    return full, "", 0


def _iso_ast(value):
    """A timestamp as ISO 8601 in AST with the offset at second precision
    ('2026-09-05T10:00:00-04:00'), or None when value is empty or not a
    timestamp. Accepts 'YYYY-MM-DD HH:MM:SS', ISO with or without fractional
    seconds, and ISO with an offset (converted to AST). A naive value is
    taken as AST, which is how every writer here stamps."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=_AST)
    return stamp.astimezone(_AST).isoformat(timespec="seconds")


def _iso_ast_from_mtime(path):
    """The modification time of path as ISO AST, or None when it cannot be read."""
    try:
        return datetime.fromtimestamp(os.path.getmtime(path), _AST).isoformat(timespec="seconds")
    except OSError:
        return None


def _read_json_object(path):
    """(dict, '') for a JSON object file, else ({}, reason). Never raises:
    a damaged file written by an earlier run is reported, not fatal."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        return {}, f"cannot read {path}: {exc}"
    if not isinstance(data, dict):
        return {}, f"{path} is not a JSON object"
    return data, ""


def _write_json_atomic(path, record):
    """Write record as JSON to path through a temp file and an atomic
    replace, creating the folder. Raises OSError naming the path."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        os.replace(tmp_path, path)
    except OSError as exc:
        raise OSError(f"cannot write {path}: {exc}") from exc


def _as_int(value):
    """value as an int, or None when it is not a whole number."""
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Ledger and champion
# ---------------------------------------------------------------------------
# A champion has a SCOPE, and every route that shows one says which it is
# showing. Two records exist on disk and they answer two different questions:
#
#   module   <inprocess>/_models/champion.json, written by POST
#            /api/step/7/promote and by the library's save_champion. This is
#            THE promoted model: the weights Inference pre-selects, the base
#            the fine-tune recipe continues from, and the run every other run
#            in the Trained runs card is ranked against. There is exactly one
#            for the whole module, whichever project or model lab trained it.
#   project  <step6_dir>/champion.json beside a project's own rounds ledger.
#            evaluate_run.py reads it to choose the baseline its gate columns
#            are measured against, so it says what this project's ledger was
#            gated on and nothing more.
#
# Before 2026-09-16 the Trained runs card read the first and the Rounds block
# read the second, both calling what they found "the champion", so one window
# showed two champions and the Evaluate panel could say no champion was
# promoted while Train and Inference were both honouring one. Every reader now
# shows the module record, labelled, and the project record rides beside it
# under its own name (gate_baseline).
CHAMPION_SCOPE_MODULE = "module"
CHAMPION_SCOPE_PROJECT = "project"


def _module_champion():
    """The promoted model for the whole module, normalised, or None.

    Scope: module (CHAMPION_SCOPE_MODULE). Read from
    <inprocess>/_models/champion.json through the shared library, so the
    Trained runs card, the Rounds block, the Inference run picker and the
    fine-tune recipe cannot disagree about which run was promoted.

    Returns:
        the record labruns.read_champion_file normalises, or None when no run
        has been promoted or the record cannot be read.
    """
    return _labruns.load_champion(_inprocess_root())


def _ledger_row_for_run(run_dir):
    """The rounds.csv row for run_dir from the ledger beside its runs/
    folder (labruns.rounds_dir_for_run), matched by run_dir then run_name;
    None when the run was never evaluated. The last matching row wins."""
    ledger = _labruns.rounds_dir_for_run(run_dir)
    rows = _labruns.read_rounds(os.path.join(ledger, ROUNDS_CSV_NAME)) if ledger else []
    want_dir = os.path.abspath(run_dir)
    want_name = os.path.basename(os.path.normpath(run_dir))
    match = None
    for row in rows:
        row_dir = str(row.get("run_dir") or "").strip()
        if (row_dir and os.path.abspath(row_dir) == want_dir) or row.get("run_name") == want_name:
            match = row
    return match


def _family_for_run(run_dir, row=None, task=""):
    """The metric family a run is scored in: the ledger row's metric_family
    when it names one, else the family of task (the run's task as read from
    disk when not given)."""
    recorded = str((row or {}).get("metric_family") or "").strip()
    if recorded in _FAMILY_SUFFIX:
        return recorded
    return _labruns.metric_family_for_task(task or _labruns.task_for_run_dir(run_dir))


def _family_column(family, metric):
    """The ledger column for metric ('map50' or 'map50_95') in family."""
    return f"{metric}_{_FAMILY_SUFFIX[family]}"


def _overall_key(family, metric):
    """The metrics.json overall key for metric ('map50' or 'map50_95') in
    family: mask_map50 for a segmenter, box_map50 for a detector. The
    evaluator names these by family and the ledger names its own columns by
    suffix (map50_M), so the two never share a spelling."""
    return f"{family}_{metric}"


def _row_float(row, key):
    """row[key] as a float, or None when blank or not a number."""
    value = (row or {}).get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_folder_missing(run_dir):
    """True when run_dir names no folder on disk.

    Parameters:
        run_dir: the run folder as a champion record or a ledger row names it.
    Returns:
        True when the folder was deleted, moved or renamed. Nothing about
        such a run can be read: not its task, not its split, not its weights,
        so anything derived from it would be a guess.
    """
    return not os.path.isdir(str(run_dir or ""))


def _champion_view(record, scope):
    """The champion as the runs table shows it (run_dir, run_name,
    promoted_at as ISO AST, score, family), from the record
    labruns.read_champion_file normalises. None for None.

    Parameters:
        record: the normalised champion record, or None.
        scope: which champion this is, CHAMPION_SCOPE_MODULE (the promoted
            model for the whole module) or CHAMPION_SCOPE_PROJECT (one
            project's own ledger baseline). It rides in the payload so a
            reader never has to guess which of the two it was handed.
    Returns:
        the five contract keys plus two: scope, and missing, True when the
        promoted run's folder is no longer on disk. The record survives a
        deleted run folder, so without that flag the card shows a champion
        nobody can read and every other run is measured against a guess.
    Raises:
        ValueError when scope is neither of the two scopes, because a payload
        that names no scope is the defect this parameter exists to close.
    """
    if scope not in (CHAMPION_SCOPE_MODULE, CHAMPION_SCOPE_PROJECT):
        raise ValueError(
            f"a champion must say which scope it is: {CHAMPION_SCOPE_MODULE!r} or "
            f"{CHAMPION_SCOPE_PROJECT!r}, got {scope!r}")
    if not record:
        return None
    run_dir = record["run_dir"]
    return {
        "run_dir": run_dir,
        "run_name": os.path.basename(os.path.normpath(run_dir)),
        "promoted_at": _iso_ast(record.get("promoted_at")),
        "score": record.get("score"),
        "family": record.get("metric_family"),
        "missing": _run_folder_missing(run_dir),
        "scope": scope,
    }


def _measurement_of(row):
    """What a ledger row says its score was measured with.

    Parameters:
        row: a rounds.csv row, or None when no ledger names the run.
    Returns:
        {"conf": float|None, "iou": float|None, "imgsz": int|None}. A cell
        that is blank or is not a number reads as None, which means "not
        recorded": a ledger written before those columns existed carries
        three empty cells, and guessing a number for one of them would
        invent the very fact the columns were added to record.
    """
    measured = {}
    for column in MEASUREMENT_COLUMNS:
        value = (row or {}).get(column)
        measured[column] = (_as_int(value) if column == "imgsz"
                            else _row_float(row, column))
    return measured


def _same_measurement(own, champion):
    """Whether two runs' scores were measured the same way.

    Parameters:
        own, champion: measurement dicts from _measurement_of.
    Returns:
        (True, '') when every one of MEASUREMENT_COLUMNS agrees, including
        the case where neither row records any of them, which is how every
        ledger written before 2026-09-15 reads and which stays rankable.
        (False, REASON_MEASUREMENT_NOT_RECORDED) when one row records a
        setting the other does not, so nobody can tell whether the two
        numbers came from the same measurement.
        (False, REASON_DIFFERENT_MEASUREMENT) when both record one and they
        differ: the same weights score differently at another confidence,
        another overlap or another image size, and ranking those two numbers
        credits the change of setting to the model.
    """
    for column in MEASUREMENT_COLUMNS:
        mine, theirs = own.get(column), champion.get(column)
        if mine is None and theirs is None:
            continue
        if mine is None or theirs is None:
            return False, REASON_MEASUREMENT_NOT_RECORDED
        if abs(float(mine) - float(theirs)) > MEASUREMENT_MATCH_TOLERANCE:
            return False, REASON_DIFFERENT_MEASUREMENT
    return True, ""


def _gate_verdicts(row):
    """The three gate verdicts a ledger row recorded for a run.

    Parameters:
        row: a rounds.csv row, or None when no ledger names the run.
    Returns:
        {column: verdict} over GATE_COLUMNS, each stripped and lowercased.
        A cell that was never written reads GATE_NOT_RECORDED (''), which is
        how every ledger written before the gates existed reads and which is
        not a stated failure.
    """
    return {column: str((row or {}).get(column) or GATE_NOT_RECORDED).strip().lower()
            for column in GATE_COLUMNS}


def _gate_needs_confirmation(gate):
    """Whether promoting this run is a decision the operator must state.

    Parameters:
        gate: the verdicts from _gate_verdicts.
    Returns:
        True when any gate recorded one of GATE_BLOCKING_VERDICTS. False when
        every recorded verdict is a pass, when a gate could not run
        ("unpinned"), and when none was recorded at all: a ledger that wrote
        no verdict has stated no failure, and refusing on silence would block
        every run evaluated before the gates existed.
    """
    return any(verdict.split(":", 1)[0] in GATE_BLOCKING_VERDICTS
               for verdict in gate.values())


def _gate_sentence(gate):
    """The gate verdicts as one plain clause for a refusal or a log line.

    Parameters:
        gate: the verdicts from _gate_verdicts.
    Returns:
        "the mean average precision gate says fail, the per-class gate says
        pass, the recall gate recorded no verdict" and the like, in the order
        of GATE_COLUMNS.
    """
    parts = []
    for column in GATE_COLUMNS:
        verdict = gate.get(column) or GATE_NOT_RECORDED
        parts.append(f"{GATE_LABELS[column]} says {verdict}"
                     if verdict != GATE_NOT_RECORDED
                     else f"{GATE_LABELS[column]} recorded no verdict")
    return ", ".join(parts)


def _comparability_reason(run_dir, family, champion_dir):
    """Why run_dir cannot be ranked against the champion, as one of the
    REASON_* strings; '' when it can. Reads the same facts runs_comparable
    reads: the metric families and the split fingerprints.

    None when the champion's own run folder is gone: its task, and so its
    metric family, can no longer be read, and reporting the family of the
    fallback task as the reason would name a difference nobody measured.
    """
    if _run_folder_missing(champion_dir):
        return None
    if family != _labruns.metric_family_for_task(_labruns.task_for_run_dir(champion_dir)):
        return REASON_DIFFERENT_FAMILY
    own = _labruns.split_fingerprint_for_run(run_dir)
    champ = _labruns.split_fingerprint_for_run(champion_dir)
    if not own[0] or not champ[0]:
        return REASON_SPLIT_NOT_RECORDED
    if own != champ:
        return REASON_DIFFERENT_SPLIT
    return ""


def _evaluation_scores(run_dir, family):
    """The scores on disk for a run no ledger row names.

    Parameters:
        run_dir: the run folder.
        family: the run's metric family, 'mask' or 'box', which names the
            overall keys to read (mask_map50 and mask_map50_95, or the box
            pair).
    Returns:
        (map50, map50_95, evaluated_at) from the newest evaluation folder of
        run_dir, evaluated_at as ISO AST. (None, None, None) when the run has
        no evaluation folder, when its metrics.json is missing or damaged
        (the reason is noted), or when the file carries neither score in this
        family: an evaluation nobody can read is not a score.
    """
    folders = _eval_folders_for_run(run_dir)
    if not folders:
        return None, None, None
    out_dir = folders[0]
    doc, err = _read_json_object(os.path.join(out_dir, METRICS_JSON_NAME))
    if err:
        vic.note(f"[orch] {err}")
        return None, None, None
    overall = _report_metrics(doc)["overall"]
    map50 = _row_float(overall, _overall_key(family, "map50"))
    map50_95 = _row_float(overall, _overall_key(family, "map50_95"))
    if map50 is None and map50_95 is None:
        return None, None, None
    return map50, map50_95, (_iso_ast(doc.get("generated_at")) or _iso_ast_from_mtime(out_dir))


def _run_scores(run_dir, row, family):
    """A run's scores for the runs card, and where they came from.

    Parameters:
        run_dir: the run folder.
        row: its rounds.csv row, or None when no ledger names it.
        family: the run's metric family, 'mask' or 'box'.
    Returns:
        (map50, map50_95, evaluated_at, source). source is
        SCORE_SOURCE_LEDGER whenever a ledger row exists, because that row is
        what promotion is allowed on, even when its own family columns are
        blank; SCORE_SOURCE_EVALUATION when the numbers were read from the
        newest evaluation folder instead; '' when the run has neither.
    """
    if row is not None:
        return (_row_float(row, _family_column(family, "map50")),
                _row_float(row, _family_column(family, "map50_95")),
                _iso_ast(row.get("at")),
                SCORE_SOURCE_LEDGER)
    map50, map50_95, evaluated_at = _evaluation_scores(run_dir, family)
    scored = map50 is not None or map50_95 is not None
    return map50, map50_95, evaluated_at, (SCORE_SOURCE_EVALUATION if scored else "")


# ---------------------------------------------------------------------------
# What a run was trained on (2026-09-17, LO). Lauren: "make a table that
# includes the training data subset (so we can filter by those)". The runs
# card renders trained_on and filters by it, so every row names the data the
# run learned from, read from the records that recorded it and never guessed
# from a folder name. Empty means "not recorded" and the card says so.
# ---------------------------------------------------------------------------

_TRAINED_ON_KINDS = ("bbox", "composite", "standalone", "export")
_RUN_ORIGINS = ("lab", "project")
_BBOX_DATASETS_SEGMENT = "bbox_datasets"
_NO_PROVENANCE = {"origin_kind": "", "imported_from": "", "imported_at": ""}


def _record_json(path):
    """Read a JSON object record, tolerating its absence or corruption.

    Parameters:
        path: the file to read.
    Returns:
        The parsed object, or {} when the file is missing, unreadable, not
        valid JSON, not a JSON object, or nested too deeply to parse. A
        record that cannot be read must never take the runs table down with
        it, so nothing here raises.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, RecursionError):
        return {}
    return data if isinstance(data, dict) else {}


def _text(value):
    """Coerce a hand-typed record field to text.

    Parameters:
        value: any field read from a JSON record.
    Returns:
        The value when it is a string, else '' (a number, list or null in a
        text field is treated as unrecorded rather than stringified).
    """
    return value if isinstance(value, str) else ""


def _dataset_from_bbox_path(path):
    """The staged dataset a path names when it sits under bbox_datasets/<name>/.

    Parameters:
        path: any path-like string, relative or absolute.
    Returns:
        <name>, or '' when the path does not pass through a bbox_datasets
        folder. A path elsewhere is not evidence of a dataset.
    """
    parts = [seg for seg in _text(path).replace("\\", "/").split("/") if seg]
    if _BBOX_DATASETS_SEGMENT in parts:
        # The last occurrence: the staged root is the innermost bbox_datasets
        # folder, so a clone whose own path holds that word cannot misname a run.
        index = len(parts) - 1 - parts[::-1].index(_BBOX_DATASETS_SEGMENT)
        if index + 1 < len(parts):
            return parts[index + 1]
    return ""


def _trained_on_for_run(run_dir, origin):
    """Name the training data a run learned from, and what kind of data it was.

    Parameters:
        run_dir: the run folder (.../runs/<name>).
        origin: 'lab' or 'project', from the list_all_runs entry.
    Returns:
        (name, kind). kind is 'bbox' (a staged box dataset), 'composite' (a
        training set from the coverage matrix), 'export' (the project's own
        annotation export), or '' when unrecorded. Read order, first hit
        wins: a lab's lab.json source; a project's training_source, then its
        composite, then its split manifest's label_dir under bbox_datasets,
        then its own export folder. ('', '') means not recorded.
    Example:
        a lab whose lab.json source ends in fish_photo_28feb25 with
        source_kind 'bbox' gives ('fish_photo_28feb25', 'bbox').
    """
    if origin not in _RUN_ORIGINS:
        raise ValueError(f"run origin must be one of {_RUN_ORIGINS}, got {origin!r} for {run_dir}")
    parent = _labruns.rounds_dir_for_run(run_dir)
    if origin == "lab":
        lab = _record_json(os.path.join(parent, "lab.json"))
        source = _text(lab.get("source")).strip()
        if not source.strip("/"):
            return "", ""
        kind = _text(lab.get("source_kind"))
        return os.path.basename(os.path.normpath(source)), (kind if kind in _TRAINED_ON_KINDS else "")
    project_dir = os.path.dirname(parent)
    project = _record_json(os.path.join(project_dir, "project.json"))
    training_source = project.get("training_source")
    if isinstance(training_source, dict) and _text(training_source.get("name")):
        kind = _text(training_source.get("kind"))
        return training_source["name"], (kind if kind in _TRAINED_ON_KINDS else "bbox")
    composite = project.get("composite")
    if isinstance(composite, dict) and _text(composite.get("name")):
        return composite["name"], "composite"
    chosen = _source_dir_recorded_for_step6(project)
    if chosen:
        real = os.path.realpath(chosen)
        if _contained(real, os.path.realpath(_standalone_root())):
            return os.path.basename(real), "standalone"
        if _contained(real, os.path.realpath(_composite_root())):
            return os.path.basename(real), "composite"
    manifest = _record_json(os.path.join(parent, "dataset", "split_manifest.json"))
    dataset = _dataset_from_bbox_path(manifest.get("label_dir"))
    if dataset:
        return dataset, "bbox"
    if os.path.isdir(os.path.join(project_dir, "step4test_combinedAnnotate")) and _text(project.get("name")):
        return project["name"], "export"
    return "", ""


def _source_dir_recorded_for_step6(project):
    """The training source the Train step was pointed at, from the project record.

    Parameters:
        project: the parsed project.json.
    Returns:
        steps.6.config.source_dir as text, or '' when the project never chose
        one (an ordinary project training on its own export).
    """
    steps = project.get("steps")
    step6 = steps.get("6") if isinstance(steps, dict) else None
    config = step6.get("config") if isinstance(step6, dict) else None
    return _text(config.get("source_dir")).strip() if isinstance(config, dict) else ""


def _provenance_for_run(run_dir, origin):
    """Where an imported run came from, for the runs card's imported marker.

    Parameters:
        run_dir: the run folder.
        origin: 'lab' or 'project'.
    Returns:
        {origin_kind, imported_from, imported_at}: from the run's own
        provenance.json when it records an import, else from the lab record
        when the lab lists the run among imported_runs, else all empty for a
        run trained here.
    """
    if origin not in _RUN_ORIGINS:
        raise ValueError(f"run origin must be one of {_RUN_ORIGINS}, got {origin!r} for {run_dir}")
    provenance = _record_json(os.path.join(run_dir, "provenance.json"))
    imported_from = _text(provenance.get("imported_from"))
    imported_at = _text(provenance.get("imported_at"))
    if imported_from or imported_at:
        return {"origin_kind": "imported", "imported_from": imported_from, "imported_at": imported_at}
    if origin == "lab":
        lab = _record_json(os.path.join(_labruns.rounds_dir_for_run(run_dir), "lab.json"))
        listed = lab.get("imported_runs")
        if isinstance(listed, list) and os.path.basename(os.path.normpath(run_dir)) in listed:
            return {"origin_kind": "imported", "imported_from": "", "imported_at": _text(lab.get("created_at"))}
    return dict(_NO_PROVENANCE)


def _all_runs_row(run, champion_dir, champion_measured):
    """One /api/step/6/all_runs row from a labruns.list_all_runs entry.

    Parameters:
        run: the entry (run_dir, run_name, origin 'lab:<name>' or
            'project:<folder>', task, has_weights, champion).
        champion_dir: the champion's run_dir, '' when none is promoted.
        champion_measured: what the champion's own ledger row says its score
            was measured with (_measurement_of), read once by the caller
            rather than once per row.
    Returns:
        the contract row plus _folder_time (the run folder's modification
        time) for sorting. Scores come from the family's own ledger columns,
        or from the run's newest evaluation folder when no ledger row names
        it, with score_source saying which. measured_at carries the three
        settings this run's score was measured with. The champion row is
        always comparable to itself, and comparability is null (unknown) both
        when no champion is promoted and when the promoted run's folder is no
        longer on disk.
    """
    run_dir = run["run_dir"]
    origin, _, owner = str(run.get("origin") or "").partition(":")
    row = _ledger_row_for_run(run_dir)
    family = _family_for_run(run_dir, row, run.get("task"))
    map50, map50_95, evaluated_at, score_source = _run_scores(run_dir, row, family)
    measured = _measurement_of(row)
    is_champion = bool(run.get("champion"))
    comparable = None
    reason = ""
    if not champion_dir or _run_folder_missing(champion_dir):
        # No champion, or a champion nobody can read: comparability stays
        # unknown, and an unknown comparison is never reported as a refusal.
        pass
    elif is_champion:
        comparable = True
    else:
        comparable = _labruns.runs_comparable(run_dir, champion_dir)[0]
        reason = "" if comparable else _comparability_reason(run_dir, family, champion_dir)
        if comparable:
            # Same family, same held-out images: the last thing between the
            # two numbers is how each was measured.
            comparable, reason = _same_measurement(measured, champion_measured)
    try:
        folder_time = os.path.getmtime(run_dir)
    except OSError:
        folder_time = 0.0
    trained_on, trained_on_kind = _trained_on_for_run(run_dir, origin)
    provenance = _provenance_for_run(run_dir, origin)
    return {
        "run_dir": run_dir,
        "run_name": run["run_name"],
        "origin": origin,
        "project_name": owner or None,
        "task": run.get("task"),
        "family": family,
        "map50": map50,
        "map50_95": map50_95,
        "evaluated_at": evaluated_at,
        "score_source": score_source,
        "measured_at": measured,
        "has_weights": bool(run.get("has_weights")),
        "champion": is_champion,
        "imgsz": _labruns.run_imgsz(run_dir, default=0) or None,
        "split_fingerprint": _labruns.split_fingerprint_for_run(run_dir)[0],
        "comparable_to_champion": comparable,
        "comparability_reason": reason,
        "trained_on": trained_on,
        "trained_on_kind": trained_on_kind,
        **provenance,
        "_folder_time": folder_time,
    }


# ---------------------------------------------------------------------------
# Evaluation folders and the report
# ---------------------------------------------------------------------------

def _is_eval_folder(folder):
    """True when folder holds an evaluator output (report.md or metrics.json)."""
    return os.path.isdir(folder) and (os.path.isfile(os.path.join(folder, REPORT_MD_NAME))
                                      or os.path.isfile(os.path.join(folder, METRICS_JSON_NAME)))


def _eval_folder_time(folder):
    """When an evaluation folder was last written: the newest of the folder
    itself, its metrics.json and its report.md, as a POSIX time. A
    re-evaluation into the same folder rewrites the files without touching
    the folder's own time, which is why the files count."""
    times = []
    for path in (folder, os.path.join(folder, METRICS_JSON_NAME), os.path.join(folder, REPORT_MD_NAME)):
        try:
            times.append(os.path.getmtime(path))
        except OSError:
            continue
    return max(times) if times else 0.0


def _matching_eval_folders(parent, pattern):
    """Sub-folders of parent whose names fully match the compiled pattern
    and hold an evaluator output; [] when parent is absent or unlistable."""
    if not os.path.isdir(parent):
        return []
    try:
        names = os.listdir(parent)
    except OSError as exc:
        vic.note(f"[orch] cannot list {parent}: {exc}")
        return []
    return [os.path.join(parent, n) for n in sorted(names)
            if pattern.fullmatch(n) and _is_eval_folder(os.path.join(parent, n))]


def _step7_dirs_for_run(run_dir):
    """The step 7 folders an evaluation of run_dir may have been written to:
    the run's own project (when it is a project run) and the open project,
    since an evaluation launched here writes into the open project whatever
    run it scores. Deduplicated by real path."""
    dirs = []
    ledger = _labruns.rounds_dir_for_run(run_dir)
    if os.path.basename(ledger) == pm.STEP_DIRS["6"]:
        dirs.append(os.path.join(os.path.dirname(ledger), pm.STEP_DIRS["7"]))
    if current_project is not None:
        dirs.append(current_project["steps"].get("7", {}).get("dir")
                    or os.path.join(current_project["project_dir"], pm.STEP_DIRS["7"]))
    seen = set()
    unique = []
    for d in dirs:
        real = os.path.realpath(d)
        if real not in seen:
            seen.add(real)
            unique.append(d)
    return unique


def _eval_folders_for_run(run_dir):
    """Every evaluation folder written for run_dir, newest first by
    _eval_folder_time: eval_<run> and eval_<run>_<stamp> under the step 7
    folders _step7_dirs_for_run names, and <run>_<stamp> under
    inprocess/_models/_eval. A folder whose name merely extends the run's
    name (another run's evaluation) does not match."""
    run_name = os.path.basename(os.path.normpath(run_dir))
    project_pattern = re.compile(re.escape(EVAL_FOLDER_PREFIX + run_name) + rf"(_{_EVAL_STAMP_PATTERN})?")
    lab_pattern = re.compile(re.escape(_labruns.safe_note(run_name)) + rf"_{_EVAL_STAMP_PATTERN}")
    found = []
    for step7_dir in _step7_dirs_for_run(run_dir):
        found.extend(_matching_eval_folders(step7_dir, project_pattern))
    found.extend(_matching_eval_folders(_labruns.eval_root(_inprocess_root()), lab_pattern))
    found.sort(key=_eval_folder_time, reverse=True)
    return found


def _eval_folder_run_name(out_dir):
    """The run an evaluation folder was written for, read from its own name.

    Parameters:
        out_dir: an evaluation folder, named eval_<run name> by _run_step7.
    Returns:
        the run name, or '' when the folder is not one of those (a lab
        evaluation under inprocess/_models/_eval is named <run>_<stamp> and
        is never one of a project's saved pointers).
    """
    leaf = os.path.basename(os.path.normpath(str(out_dir or "")))
    return leaf[len(EVAL_FOLDER_PREFIX):] if leaf.startswith(EVAL_FOLDER_PREFIX) else ""


def _resolve_evaluation(raw_run_dir):
    """(run_dir, out_dir, '', 0) for the evaluation to show.

    Parameters:
        raw_run_dir: the run_dir the client sent, or empty.
    Returns:
        with a run_dir, the newest evaluation folder for that run; without
        one, the open project's saved Step 7 pointer (steps.7.outputs.out_dir
        with steps.7.config.run_dir), so a project evaluated before reports
        were keyed by run still shows its report. Else ('', '', reason,
        status): 400 with no project, 403 outside the roots, 404 when there
        is no evaluation.

    Without a run_dir the name always comes from the folder that holds the
    numbers: the saved selection is used only when it names that same run.
    A project.json whose two saved pointers disagree (a selection saved by
    the client for a launch that was then refused, which older builds left
    behind) would otherwise serve this project's numbers under a run that was
    never scored.
    """
    if str(raw_run_dir or "").strip():
        run_dir, err, status = _validate_client_dir(raw_run_dir, "run_dir")
        if err:
            return "", "", err, status
        folders = _eval_folders_for_run(run_dir)
        if not folders:
            return "", "", (f"{NO_EVALUATION_PREFIX} {os.path.basename(run_dir)}: run "
                            f"{step_ref(7)} on it first"), HTTP_NOT_FOUND
        return run_dir, folders[0], "", 0
    if current_project is None:
        return "", "", "No project loaded", HTTP_BAD_REQUEST
    step7 = current_project["steps"].get("7", {})
    out_dir = str((step7.get("outputs") or {}).get("out_dir") or "").strip()
    if not out_dir or not _is_eval_folder(out_dir):
        return "", "", f"{NO_EVALUATION_PREFIX} this project yet: run {step_ref(7)} first", HTTP_NOT_FOUND
    run_dir = str((step7.get("config") or {}).get("run_dir") or "").strip()
    folder_run_name = _eval_folder_run_name(out_dir)
    if folder_run_name and os.path.basename(os.path.normpath(run_dir)) != folder_run_name:
        # Either no run was saved, or the one saved is not the run this
        # folder holds the numbers for. The folder decides, and the runs live
        # under step 6.
        run_dir = os.path.join(current_project["steps"].get("6", {}).get("dir", ""),
                               "runs", folder_run_name)
    elif not run_dir:
        # A folder that is not named eval_<run> and no saved run: the leaf is
        # the best name there is.
        leaf = os.path.basename(os.path.normpath(out_dir))
        run_dir = os.path.join(current_project["steps"].get("6", {}).get("dir", ""), "runs", leaf)
    return run_dir, out_dir, "", 0


def _image_url(route, **params):
    """A url on route with params query-encoded."""
    return f"{route}?{urlencode(params)}"


def _report_metrics(doc):
    """The contract's metrics block from a metrics.json document: the
    METRICS_KEYS under its 'metrics' object, with an empty value for any the
    evaluator did not write, and the family derived from the numbers when
    the file predates the metric_family key. class_full_names (class id to
    full species name, written from a staged dataset's names table) is
    forwarded so the report card can show a species name beside a class token
    no species code table names; an evaluation from before that key forwards
    an empty map."""
    inner = doc.get("metrics") if isinstance(doc.get("metrics"), dict) else {}

    def _mapping(key):
        return inner.get(key) if isinstance(inner.get(key), dict) else {}

    overall = _mapping("overall")
    family = str(inner.get("metric_family") or "").strip()
    if family not in _FAMILY_SUFFIX:
        has_mask = any(k.startswith(_MASK_OVERALL_PREFIX) and v is not None for k, v in overall.items())
        family = _labruns.METRIC_FAMILY_MASK if has_mask else _labruns.METRIC_FAMILY_BOX
    return {
        "overall": overall,
        "per_class": _mapping("per_class"),
        "per_class_instances": _mapping("per_class_instances"),
        "absent_classes": [str(x) for x in (inner.get("absent_classes") or [])],
        "metric_family": family,
        "class_names": _mapping("class_names"),
        "class_full_names": _mapping("class_full_names"),
    }


def _report_previews(run_dir, out_dir, doc):
    """The preview entries the evaluator recorded whose overlay file exists
    under <out_dir>/previews: filename, image url, n_gt and n_pred."""
    out = []
    for item in doc.get("previews") or []:
        if not isinstance(item, dict):
            continue
        overlay_name = os.path.basename(str(item.get("overlay") or ""))
        if not overlay_name or not os.path.isfile(os.path.join(out_dir, PREVIEWS_DIRNAME, overlay_name)):
            continue
        out.append({
            "filename": str(item.get("filename") or overlay_name),
            "url": _image_url(STEP7_IMAGE_ROUTE, run_dir=run_dir, rel=f"{PREVIEWS_DIRNAME}/{overlay_name}"),
            "n_gt": _as_int(item.get("n_gt")),
            "n_pred": _as_int(item.get("n_pred")),
        })
    return out


def _report_figures(run_dir, out_dir):
    """One url per contract figure key, None where the file is missing: the
    curves and confusion matrices under val_run/, results.png from the run."""
    figures = {}
    for key, name in REPORT_FIGURES:
        rel = f"{VAL_RUN_DIRNAME}/{name}"
        present = os.path.isfile(os.path.join(out_dir, rel))
        figures[key] = _image_url(STEP7_IMAGE_ROUTE, run_dir=run_dir, rel=rel) if present else None
    for key, name in RUN_FIGURES:
        present = os.path.isfile(os.path.join(run_dir, name))
        figures[key] = (_image_url(STEP7_IMAGE_ROUTE, run_dir=run_dir, rel=RUN_FIGURE_REL_PREFIX + name)
                        if present else None)
    return figures


def _report_payload(run_dir, out_dir):
    """The GET /api/step/7/report body for the evaluation folder out_dir of
    run_dir, per the wave 2 contract. A missing or unreadable report.md
    reads as an empty string and a missing or damaged metrics.json as empty
    metrics, each with the reason noted, so an interrupted evaluation still
    shows what it wrote."""
    md_path = os.path.join(out_dir, REPORT_MD_NAME)
    report_md = ""
    if os.path.isfile(md_path):
        try:
            with open(md_path, encoding="utf-8") as fh:
                report_md = fh.read()
        except (OSError, UnicodeDecodeError) as exc:
            vic.note(f"[orch] cannot read {md_path}: {exc}")
    doc = {}
    metrics_path = os.path.join(out_dir, METRICS_JSON_NAME)
    if os.path.isfile(metrics_path):
        doc, doc_err = _read_json_object(metrics_path)
        if doc_err:
            vic.note(f"[orch] {doc_err}")
    pdf_exists = os.path.isfile(os.path.join(out_dir, REPORT_PDF_NAME))
    return {
        "ok": True,
        "run_name": os.path.basename(os.path.normpath(run_dir)),
        "run_dir": run_dir,
        "out_dir": out_dir,
        "evaluated_at": _iso_ast(doc.get("generated_at")) or _iso_ast_from_mtime(out_dir),
        "report_md": report_md,
        "metrics": _report_metrics(doc),
        "previews": _report_previews(run_dir, out_dir, doc),
        "figures": _report_figures(run_dir, out_dir),
        "pdf_url": _image_url(STEP7_PDF_ROUTE, run_dir=run_dir) if pdf_exists else None,
    }


# ---------------------------------------------------------------------------
# Inference runs
# ---------------------------------------------------------------------------

def _inference_source_label(args):
    """What an inference run read, in words: the folder for a directory
    source, the sampling source's name otherwise, the raw source_type when
    it is neither."""
    source_type = str(args.get("source_type") or "")
    if source_type == _INFERENCE_DIR_SOURCE:
        return f"folder {args.get('source_dir') or ''}".strip()
    return _INFERENCE_SOURCE_LABELS.get(source_type, source_type)


def _inference_row(out_dir):
    """(row, '') for /api/step/8/runs from <out_dir>/manifest.json, or
    (None, reason) when the manifest is missing or unreadable."""
    manifest_path = os.path.join(out_dir, MANIFEST_JSON_NAME)
    if not os.path.isfile(manifest_path):
        return None, f"no {MANIFEST_JSON_NAME} in {out_dir}"
    doc, err = _read_json_object(manifest_path)
    if err:
        return None, err
    args = doc.get("args") if isinstance(doc.get("args"), dict) else {}
    items = [i for i in (doc.get("items") or []) if isinstance(i, dict)]
    n_detections = _as_int(doc.get("total_predictions"))
    if n_detections is None:
        n_detections = sum(_as_int(i.get("n_detections")) or 0 for i in items)
    return {
        "out_dir": out_dir,
        "name": os.path.basename(os.path.normpath(out_dir)),
        "created_at": _iso_ast(doc.get("generated_at")) or _iso_ast_from_mtime(manifest_path),
        "model_run_name": os.path.basename(os.path.normpath(str(args.get("run_dir") or ""))),
        "n_images": len(items),
        "n_detections": n_detections,
        "source_label": _inference_source_label(args),
    }, ""


def _inference_parents():
    """The folders whose sub-folders are inference runs: the open project's
    step 8 folder (when a project is open) and inprocess/_models/_inference."""
    parents = []
    if current_project is not None:
        parents.append(current_project["steps"].get("8", {}).get("dir")
                       or os.path.join(current_project["project_dir"], pm.STEP_DIRS["8"]))
    parents.append(_labruns.inference_root(_inprocess_root()))
    return parents


def _list_inference_runs():
    """Every inference run under _inference_parents with a readable
    manifest, newest first by created_at. A folder with no manifest yet (a
    run still in flight) is skipped quietly; one whose manifest cannot be
    read is skipped with the reason noted."""
    rows = []
    for parent in _inference_parents():
        if not os.path.isdir(parent):
            continue
        try:
            names = sorted(os.listdir(parent))
        except OSError as exc:
            vic.note(f"[orch] cannot list {parent}: {exc}")
            continue
        for name in names:
            out_dir = os.path.join(parent, name)
            if not os.path.isdir(out_dir) or not os.path.isfile(os.path.join(out_dir, MANIFEST_JSON_NAME)):
                continue
            row, err = _inference_row(out_dir)
            if row is None:
                vic.note(f"[orch] skipping inference folder: {err}")
                continue
            rows.append(row)
    rows.sort(key=lambda r: r["created_at"] or "", reverse=True)
    return rows


def _resolve_inference_dir(raw_out_dir):
    """(out_dir, '', 0) for the inference run to read: raw_out_dir checked
    against the allowed roots when given, else the open project's saved
    Step 8 pointer (steps.8.outputs.out_dir). Else ('', reason, status)."""
    if str(raw_out_dir or "").strip():
        return _validate_client_dir(raw_out_dir, "out_dir")
    if current_project is None:
        return "", "No project loaded", HTTP_BAD_REQUEST
    out_dir = str((current_project["steps"].get("8", {}).get("outputs") or {}).get("out_dir") or "").strip()
    if not out_dir or not os.path.isdir(out_dir):
        return "", f"No inference run in this project yet: run {step_ref(8)} first", HTTP_NOT_FOUND
    return out_dir, "", 0


def _read_verdicts(out_dir):
    """The verdicts recorded for an inference run, filename to 'yes', 'no'
    or 'skip', from <out_dir>/verifications.csv (written by
    POST /api/step/8/verify). The last row per filename wins, so a changed
    mind replaces the earlier press. {} when the file is absent; {} with the
    reason noted when it cannot be read; a row whose verdict is not one of
    VERDICTS is skipped with the reason noted."""
    path = os.path.join(out_dir, VERIFICATIONS_CSV_NAME)
    if not os.path.isfile(path):
        return {}
    verdicts = {}
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                filename = str(row.get("filename") or "").strip()
                verdict = str(row.get("verdict") or "").strip()
                if not filename or verdict not in VERDICTS:
                    vic.note(f"[orch] skipping a verdict row in {path}: filename {filename!r}, verdict {verdict!r}")
                    continue
                verdicts[filename] = verdict
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        vic.note(f"[orch] cannot read {path}: {exc}")
        return {}
    return verdicts


def _raw_frame_path(out_dir, filename):
    """(path, '', 0) of the source image the run's manifest recorded for
    filename, resolved against out_dir when recorded relative.

    The client never supplies the path itself. Returns ('', reason, status):
    404 when the manifest is unreadable or lists no such filename, 403 when
    the recorded path is not a regular file this process may read.
    """
    manifest_path = os.path.join(out_dir, MANIFEST_JSON_NAME)
    doc, err = _read_json_object(manifest_path)
    if err:
        return "", err, HTTP_NOT_FOUND
    for item in doc.get("items") or []:
        if not isinstance(item, dict) or item.get("filename") != filename:
            continue
        raw = str(item.get("raw") or "")
        if not raw:
            return "", f"the manifest records no source path for {filename}", HTTP_NOT_FOUND
        path = raw if os.path.isabs(raw) else os.path.join(out_dir, raw)
        if not os.path.isfile(path) or not os.access(path, os.R_OK):
            return "", f"the source image recorded for {filename} is not a readable file: {path}", HTTP_FORBIDDEN
        return path, "", 0
    return "", f"{filename} is not listed in {manifest_path}", HTTP_NOT_FOUND


# ---------------------------------------------------------------------------
# Step 6 training sources
# ---------------------------------------------------------------------------

def _under_bbox_root(source_dir):
    """True when source_dir resolves to a folder under the staged box-dataset root."""
    return _contained(os.path.realpath(str(source_dir or "")), os.path.realpath(_bbox_root()))


def _validate_bbox_source(source_dir):
    """(path, '') when source_dir is a finished staged box dataset (a folder
    holding every file in _BBOX_SOURCE_FILES), else ('', reason)."""
    path = os.path.abspath(str(source_dir or "").strip())
    if not os.path.isdir(path):
        return "", f"box dataset folder not found: {path}"
    for name in _BBOX_SOURCE_FILES:
        if not os.path.isfile(os.path.join(path, name)):
            return "", (f"{path} is not a finished staged box dataset: it has no {name}. "
                        "Stage it again from the Bounding Boxes page.")
    return path, ""


def _recorded_training_source(state):
    """The real path of the source a training project was minted to train
    from: the composite behind the step4test link for a composite project,
    training_source.path for a box-dataset project; '' for an ordinary
    project, which records none, or when the source is gone."""
    if not _is_composite_training(state):
        return ""
    if state.get("project_type") == BBOX_PROJECT_TYPE:
        path = str((state.get("training_source") or {}).get("path") or "")
    else:
        path = os.path.join(state["project_dir"], pm.STEP_DIRS["step4test"])
    return os.path.realpath(path) if path and os.path.isdir(path) else ""


def _step6_fraction(cfg, key, label):
    """(value, '') for the panel's held-out fraction cfg[key], the panel
    default from pipeline.yaml when the config carries none; (None, reason)
    when it is not a number. The range check belongs to the split builder."""
    raw = cfg.get(key)
    if raw in (None, ""):
        raw = DEFAULT_STEP_CONFIGS.get("6", {}).get(key)
    if isinstance(raw, bool):
        return None, f"{label} must be a number between 0 and 1 (got {raw!r})"
    try:
        return float(raw), ""
    except (TypeError, ValueError):
        return None, f"{label} must be a number between 0 and 1 (got {raw!r})"


# ---------------------------------------------------------------------------
# Step 6 training-data summary (spec 2026-09-21-vega-starting-model-and-
# training-data-design.md section 7). GET /api/step/6/source_summary reads
# these to answer, before Start Training builds anything, how many images and
# labels a source holds and what its split would look like. Every helper here
# is read-only: none writes a file, and none replaces the real splitters
# (pinned_split.py, detect_dataset.py), which still decide the split that
# actually gets built.
# ---------------------------------------------------------------------------

# The transect numbers pinned_split.py's default policy holds out, mirrored
# here because the summary route takes no val-transects/test-transects query
# (the panel never exposes them): 5 for validation, 6 for test, matching
# run_step6.sh's own TCRMP_STEP6_VAL_TRANSECTS/TCRMP_STEP6_TEST_TRANSECTS
# defaults.
_STEP6_SUMMARY_VALID_TRANSECT = 5
_STEP6_SUMMARY_TEST_TRANSECT = 6


def _step6_summary_resolve_source(source_dir_param):
    """Resolve the source the Train panel's data summary reads.

    Mirrors _run_step6's own resolution, without launching anything: a blank
    source_dir means the open project's recorded source (its own step4test
    export, or the box dataset a bbox_training project was minted against); a
    non-blank source_dir is validated exactly as a picked one is validated at
    launch (_validate_bbox_source under the staged box-dataset root, else
    _validate_outline_source).

    Parameters:
        source_dir_param: the raw ?source_dir= query value.
    Returns:
        (path, kind, name, error): kind one of "export", "composite",
        "standalone" or "bbox"; name is what the summary calls the source (the
        project's own name for its own export or recorded box dataset, else
        the source folder's basename); ("", "", "", reason) when the source
        cannot be resolved (a missing folder, a path outside the allowed
        roots, or no project's own source is recorded).
    """
    source_dir = str(source_dir_param or "").strip()
    if not source_dir:
        if (current_project or {}).get("project_type") == BBOX_PROJECT_TYPE:
            path = _recorded_training_source(current_project)
            if not path:
                return "", "", "", "This project's recorded box dataset is missing or unreadable."
            return path, "bbox", current_project.get("name", ""), ""
        path = _step4test_export_dir()
        if not path or not os.path.isdir(path):
            return "", "", "", (f"{step_ref('step4test')} has no output yet at "
                                f"{path or '(unset)'}. Run it first.")
        return path, "export", current_project.get("name", ""), ""
    if _under_bbox_root(source_dir):
        path, err = _validate_bbox_source(source_dir)
        if err:
            return "", "", "", err
        return path, "bbox", os.path.basename(path), ""
    path, err = _validate_outline_source(source_dir)
    if err:
        return "", "", "", err
    kind = ("standalone" if _contained(os.path.realpath(path), os.path.realpath(_standalone_root()))
            else "composite")
    return path, kind, os.path.basename(path), ""


def _step6_summary_task(kind, name, path, raw_task):
    """The normalized task the summary route previews, or a refusal.

    Parameters:
        kind: the resolved source's kind ("export", "composite",
            "standalone" or "bbox").
        name, path: the resolved source's display name and path, for the
            refusal message.
        raw_task: the raw ?task= query value, '' or None for unspecified.
    Returns:
        (task, error): task is "segment", "detect" or '' (unspecified);
        error is '' unless raw_task names an unknown spelling, or names
        "segment" against a resolved box source, which holds no outlines.
    """
    task = ""
    if raw_task not in (None, ""):
        task = _normalize_task(raw_task)
        if not task:
            return "", (f"Unknown training task {raw_task!r}. Use segment "
                        "(the default) or detect.")
    if kind == "bbox" and task == "segment":
        return "", (f"{name or os.path.basename(path)} holds boxes and no outlines to learn "
                    f"from, so it can only preview a detector's split. Set Training task to "
                    f"Detector in the {step_ref(6)} advanced settings.")
    return task, ""


def _step6_summary_ratios(request_args):
    """The held-out fractions the summary route previews with.

    Parameters:
        request_args: the request's query string (any mapping with .get).
    Returns:
        (valid_ratio, test_ratio, error): the panel's pipeline.yaml step 6
        defaults when the query names none; error is '' unless a present
        value is malformed or the two fractions leave no image for training.
    """
    try:
        valid_ratio = _step6_summary_fraction(
            request_args, "valid_ratio", "Validation fraction",
            DEFAULT_STEP_CONFIGS.get("6", {}).get("valid_ratio", 0.2))
        test_ratio = _step6_summary_fraction(
            request_args, "test_ratio", "Test fraction",
            DEFAULT_STEP_CONFIGS.get("6", {}).get("test_ratio", 0.1))
    except ValueError as e:
        return None, None, str(e)
    if valid_ratio + test_ratio >= 1:
        return None, None, (
            f"Validation fraction + Test fraction must leave images for training "
            f"(got {valid_ratio} + {test_ratio}, which is 1 or more)")
    return valid_ratio, test_ratio, ""


def _step6_summary_include_ids(request_args):
    """The class-id filter the summary route previews with.

    Parameters:
        request_args: the request's query string (any mapping with .get).
    Returns:
        (include_ids, error): a set of class ids from ?include_classes=,
        None for blank or unparseable-to-empty (every named class); error
        is '' unless a present value is not comma-separated integers.
    """
    raw = (request_args.get("include_classes") or "").strip()
    if not raw:
        return None, ""
    try:
        include_ids = {int(x) for x in raw.split(",") if x.strip()}
    except ValueError:
        return None, f"include_classes must be comma-separated integers, got {raw!r}"
    return (include_ids or None), ""


def _step6_summary_fraction(request_args, key, label, default):
    """A held-out fraction from the summary route's query string.

    Parameters:
        request_args: the request's query string (any mapping with .get).
        key: the query parameter name.
        label: the field name an error message names.
        default: the value to use when the query carries none.
    Returns:
        the query value as a float in [0, 1) when present, else default.
    Raises:
        ValueError naming `label` when a present value is not a number, or is
        outside [0, 1).
    """
    raw = request_args.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number between 0 and 1 (got {raw!r})")
    if not 0 <= value < 1:
        raise ValueError(f"{label} must be at least 0 and below 1 (got {value!r})")
    return value


def _step6_summary_class_names(src_dir, include_ids):
    """Named classes an outline source's data.yaml declares, for the summary.

    Parameters:
        src_dir: an outline source directory (holds data.yaml).
        include_ids: a set of class ids to keep, or None for every named
            class.
    Returns:
        {class_id: name}, unnamed classes dropped (_step6_is_unnamed_class,
        matching /api/step/6/list_classes), further restricted to
        include_ids when given.
    """
    names = _data_yaml_id_names(os.path.join(src_dir, "data.yaml"))
    named = {cid: nm for cid, nm in names.items() if not _step6_is_unnamed_class(nm)}
    if include_ids is not None:
        named = {cid: nm for cid, nm in named.items() if cid in include_ids}
    return named


def _step6_outline_label_lines(labels_dir, basename, wanted_ids):
    """Included label-line count for one outline image.

    Parameters:
        labels_dir: the source's all_labels/ folder.
        basename: the image's file name in all_images/.
        wanted_ids: a set of class ids to count, or None to count every line
            whose first field parses as an integer.
    Returns:
        the count of matching lines; 0 when the label file is absent,
        unreadable, or empty. A line whose first field is not an integer is
        skipped, the same tolerance step6_list_classes applies.
    """
    path = os.path.join(labels_dir, os.path.splitext(basename)[0] + ".txt")
    if not os.path.isfile(path):
        return 0
    n = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    cid = int(line.split()[0])
                except ValueError:
                    continue
                if wanted_ids is None or cid in wanted_ids:
                    n += 1
    except OSError:
        return 0
    return n


def _step6_outline_counts(src_dir, include_ids):
    """Per-image included-label counts for an outline source (a segmentation
    export, composite export or Annotate Images session).

    An image counts only when it carries at least one label line of an
    included class (design section 7's counting rule), so a frame with no
    matching label is left out of both the image count and the split preview
    below, the same way a fully unlabeled frame never enters a training run
    that only reads all_labels/.

    Parameters:
        src_dir: a validated outline source directory.
        include_ids: a set of class ids to count, or None for every named
            class.
    Returns:
        (counted, n_labels, n_classes): counted is {basename: n_lines} for
        every qualifying image; n_labels is the sum of those lines; n_classes
        is how many named classes the count covers.
    """
    n_classes = len(_step6_summary_class_names(src_dir, include_ids))
    images_dir = os.path.join(src_dir, "all_images")
    labels_dir = os.path.join(src_dir, "all_labels")
    try:
        basenames = sorted(os.listdir(images_dir))
    except OSError:
        basenames = []
    counted, n_labels = {}, 0
    for bn in basenames:
        n = _step6_outline_label_lines(labels_dir, bn, include_ids)
        if n:
            counted[bn] = n
            n_labels += n
    return counted, n_labels, n_classes


def _step6_random_counts(n_images, valid_ratio, test_ratio):
    """(n_train, n_valid, n_test) the random-split rule reports.

    The held-out fractions multiplied by n_images and rounded, the same
    arithmetic a seeded shuffle averages to, without replaying the shuffle
    (design section 7: "the random rule multiplies by the fractions and
    rounds"). Never negative: rounding both fractions up cannot cost more
    images than n_images holds, so the test count is trimmed first.
    """
    n_valid = round(n_images * valid_ratio)
    n_test = round(n_images * test_ratio)
    if n_valid + n_test > n_images:
        n_test = max(0, n_images - n_valid)
    n_train = n_images - n_valid - n_test
    return n_train, n_valid, n_test


def _step6_transect_pins(basenames):
    """{basename: 'train'|'valid'|'test'} by pinned_split.parse_transect,
    the default holdout (transect 5 validation, transect 6 test, everything
    else, including an unparseable name, train), the same rule
    assign_fresh_transect applies before any degrade check."""
    ps = _load_pinned_split()
    pins = {}
    for bn in basenames:
        t = ps.parse_transect(bn)
        if t == _STEP6_SUMMARY_TEST_TRANSECT:
            pins[bn] = "test"
        elif t == _STEP6_SUMMARY_VALID_TRANSECT:
            pins[bn] = "valid"
        else:
            pins[bn] = "train"
    return pins


def _step6_outline_split(counted_basenames, pin_split, valid_ratio, test_ratio):
    """The split preview for an outline source.

    Parameters:
        counted_basenames: the images the Pictures line counts.
        pin_split: True for the panel's default (Holdout split ticked,
            TCRMP_STEP6_PIN_SPLIT=1: transect holdout), False for the legacy
            random split.
        valid_ratio, test_ratio: the held-out fractions.
    Returns:
        (rule, n_train, n_valid, n_test, warning): rule is "transect" or
        "random"; warning is '' unless a transect split degrades because the
        source carries no transect-5 or no transect-6 image, in which case it
        names the fallback, replaying main()'s own degrade check in
        pinned_split.py without writing anything.
    """
    n_images = len(counted_basenames)
    if not pin_split:
        n_train, n_valid, n_test = _step6_random_counts(n_images, valid_ratio, test_ratio)
        return "random", n_train, n_valid, n_test, ""
    pins = _step6_transect_pins(counted_basenames)
    counts = {"train": 0, "valid": 0, "test": 0}
    for split in pins.values():
        counts[split] += 1
    if counts["valid"] and counts["test"]:
        return "transect", counts["train"], counts["valid"], counts["test"], ""
    missing = []
    if not counts["valid"]:
        missing.append("transect 5 (validation)")
    if not counts["test"]:
        missing.append("transect 6 (test)")
    warning = (f"No images in this source parse to {' or '.join(missing)}, so the pinned split "
              "would fall back to a random holdout on the real run too; showing that random "
              "split here instead.")
    n_train, n_valid, n_test = _step6_random_counts(n_images, valid_ratio, test_ratio)
    return "random", n_train, n_valid, n_test, warning


def _step6_bbox_counts(src_dir):
    """Every image's box-line count for a staged detection dataset.

    A box source counts every image in images/, whether or not it has a
    label file (design section 7), unlike an outline source which counts
    only images with a matching label line.

    Parameters:
        src_dir: a validated staged box dataset.
    Returns:
        (basenames, n_labels, n_background, n_classes): basenames is every
        image file in images/ (matching detect_dataset.IMAGE_EXTS); n_labels
        is the sum of box lines across their label files; n_background is
        how many of them carry zero box lines (no label file, or an empty
        one); n_classes is the count of names in classes.txt, else
        data.yaml.
    """
    dd = _load_detect_dataset()
    image_dir = os.path.join(src_dir, "images")
    label_dir = dd.bbox_label_dir(src_dir)
    try:
        names = sorted(os.listdir(image_dir))
    except OSError:
        names = []
    basenames, n_labels, n_background = [], 0, 0
    for fn in names:
        if os.path.splitext(fn)[1].lower() not in dd.IMAGE_EXTS:
            continue
        if not os.path.isfile(os.path.join(image_dir, fn)):
            continue
        basenames.append(fn)
        n = 0
        if label_dir:
            label_path = os.path.join(label_dir, os.path.splitext(fn)[0] + ".txt")
            if os.path.isfile(label_path):
                try:
                    with open(label_path, "r", encoding="utf-8", errors="ignore") as fh:
                        n = sum(1 for ln in fh if ln.strip())
                except OSError:
                    n = 0
        n_labels += n
        if not n:
            n_background += 1
    class_names = dd.read_classes_txt(os.path.join(src_dir, "classes.txt")) \
        or dd.data_yaml_labels(os.path.join(src_dir, "data.yaml"))
    return basenames, n_labels, n_background, len(class_names)


def _step6_bbox_split(src_dir, basenames, valid_ratio, test_ratio):
    """The split preview for a staged box dataset.

    Parameters:
        src_dir: the staged box dataset.
        basenames: the images _step6_bbox_counts found.
        valid_ratio, test_ratio: the held-out fractions.
    Returns:
        (rule, n_train, n_valid, n_test, approximate, warning): rule is
        "pinned" when the staged split_pins.json is readable and pins at
        least one present image (exact counts, approximate False), else
        "random" (approximate True). warning names an unreadable pins file,
        so the random fallback is explained rather than silent.
    """
    dd = _load_detect_dataset()
    try:
        pins = dd.read_source_pins(src_dir)
    except ValueError as e:
        n_train, n_valid, n_test = _step6_random_counts(len(basenames), valid_ratio, test_ratio)
        return ("random", n_train, n_valid, n_test, True,
                f"The staged split pins could not be read ({e}); showing a random split instead.")
    present = set(basenames)
    counts = {"train": 0, "valid": 0, "test": 0}
    for bn, split in pins.items():
        if bn in present:
            counts[split] += 1
    if any(counts.values()):
        return "pinned", counts["train"], counts["valid"], counts["test"], False, ""
    n_train, n_valid, n_test = _step6_random_counts(len(basenames), valid_ratio, test_ratio)
    return "random", n_train, n_valid, n_test, True, ""


# The Pictures sentence's irregular plurals (label_noun already carries the
# source's own plural spelling, "outlines" or "boxes"; "box" is not "boxs").
_STEP6_LABEL_SINGULAR = {"outlines": "outline", "boxes": "box"}


def _step6_count_word(n, singular, plural):
    """"1 <singular>" for n == 1, else "<n:,> <plural>" with thousands
    separators, so "1 image" reads right beside "946 outlines"."""
    return f"{n:,} {singular if n == 1 else plural}"


def _step6_pictures_sentence(kind, n_images, n_labels, label_noun, n_classes, n_background):
    """The Pictures line's sentence, numbers formatted with thousands
    separators and each count singular or plural on its own (design section
    3.1): "1,036 images, 3,412 outlines, 2 classes", or "1 image, 1 outline,
    1 class". The "<n> with no box" clause (a box source only) appears only
    when n_background is at least 1, and reads "1 with no box" for one; the
    word "box" there never pluralizes, because the clause counts IMAGES with
    no box, not boxes."""
    label_singular = _STEP6_LABEL_SINGULAR.get(label_noun, label_noun)
    base = (f"{_step6_count_word(n_images, 'image', 'images')}, "
            f"{_step6_count_word(n_labels, label_singular, label_noun)}, "
            f"{_step6_count_word(n_classes, 'class', 'classes')}")
    if kind == "bbox" and n_background:
        base += f", {n_background:,} with no box"
    return base


def _step6_split_sentence(rule, n_train, n_valid, n_test, approximate, valid_ratio, test_ratio):
    """The Split line's sentence for the resolved rule (design section 3.1),
    numbers formatted with thousands separators; "about " prefixes the
    counts of an approximate rule (transect, random), never the pinned
    rule's exact counts.

    The pinned rule's sentence drops its trailing "images" noun when any of
    the three counts is 1 ("3 train, 1 validation, 1 test."): a single
    generic noun cannot agree with three counts at once, so "1 test images"
    is a grammar error, and the simpler fix is to drop the noun rather than
    inflect it per clause. With every count above 1 the noun stays, matching
    the design's own example ("5,300 train, 1,514 validation, 758 test
    images."). The transect and random sentences keep "images" unconditionally:
    they name one N / N / N group, not three individually-labeled counts, so
    "about 1 / 1 / 1 images" is not a grammar error worth a special case.
    """
    about = "about " if approximate else ""
    if rule == "transect":
        return (f"Split by transect: transects 1 to 4 train, transect 5 validation, "
                f"transect 6 test, {about}{n_train:,} / {n_valid:,} / {n_test:,} images.")
    if rule == "pinned":
        counts = f"{n_train:,} train, {n_valid:,} validation, {n_test:,} test"
        noun = "" if 1 in (n_train, n_valid, n_test) else " images"
        return f"Split pinned by the dataset: {counts}{noun}."
    valid_pct = round(valid_ratio * 100)
    test_pct = round(test_ratio * 100)
    train_pct = max(0, 100 - valid_pct - test_pct)
    return (f"Random split: {train_pct}% train, {valid_pct}% validation, {test_pct}% test, "
            f"{about}{n_train:,} / {n_valid:,} / {n_test:,} images.")


def _parse_eval_imgsz(raw):
    """(size, '') for the Step 7 image size the panel sent: 0 when blank
    (the client sends an empty string for a blank field) or 0, meaning
    'use the size the run trained at'; else the whole number of pixels.
    (None, reason) for anything that is not a whole number or is negative."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return 0, ""
    size = None
    if isinstance(raw, bool):
        size = None
    elif isinstance(raw, int):
        size = raw
    elif isinstance(raw, float) and raw.is_integer():
        size = int(raw)
    elif isinstance(raw, str) and re.fullmatch(r"-?\d+", raw.strip()):
        size = int(raw.strip())
    if size is None or size < 0:
        return None, (f"Image size must be a whole number of pixels, or blank to use the size "
                      f"the run trained at (got {raw!r})")
    return size, ""



def _folder_response(path, project_dir):
    """Shared tail of every open-folder route: path guard (inside the current
    project or the repo) then the file-manager spawn. Every error body carries
    the resolved path so the UI can show it instead of a bare message."""
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return jsonify({"error": f"not a directory: {path}", "path": path}), 400
    project_dir = os.path.abspath(project_dir)
    repo_dir = os.path.abspath(REPO_DIR)
    if not (path.startswith(project_dir + os.sep) or path == project_dir
            or path.startswith(repo_dir + os.sep) or path == repo_dir):
        return jsonify({"error": "path outside project/repo", "path": path}), 403
    ok, err = _open_in_file_manager(path)
    if not ok:
        return jsonify({"error": f"file manager did not open: {err}", "path": path}), 500
    return jsonify({"success": True, "path": path})


def _vic_end_step(step, status, notes="", outputs=None, state=None):
    """Emit a VICARIUS process_end event linked to the step's process_start.

    Parameters:
        step: the step key whose event is being closed.
        status: how the process ended, "success", "failed" or "cancelled".
        notes: the note the event carries.
        outputs: the artifact paths this process really produced. None means
            the step's own recorded outputs, which is where every step that
            writes into the open project keeps them; an evaluation of another
            owner's run passes its own folder instead, because the project's
            recorded outputs are not what that process wrote.
        state: the project whose step is being closed. None means the open
            one, which is every caller but the one that records a finished
            step against the project that launched it while a different
            project is open.
    Returns:
        None. No-op when vicarius is disabled or the step was never started
        via the orchestrator (e.g. stale state).
    """
    state = current_project if state is None else state
    if state is None:
        return
    s = str(step)
    st = state["steps"].get(s, {})
    event_id = st.pop("_vic_event_id", None)
    started = st.get("started_at")
    duration = 0.0
    if started:
        try:
            t0 = datetime.fromisoformat(started).timestamp()
            duration = max(0.0, datetime.now().timestamp() - t0)
        except Exception:
            duration = 0.0
    if outputs is None:
        outputs = list(st.get("outputs", {}).values())
    step_name = f"step{step}_{pm.STEP_DIRS.get(s, '').split('_', 1)[-1] or 'stage'}"
    vic.process_end(step_name, event_id, status, duration,
                    outputs=outputs, notes=notes)


def _record_step_for_owner(step, launch, succeeded, exit_code):
    """Record a finished step against the project that launched it.

    The runner holds one process per step for the whole orchestrator, so a
    training started in project A finishes while project B may be open. The
    result belongs to A: its project.json is where the completion (or the
    error) is written, and B, which did no work, is left untouched.

    Parameters:
        step: the step key that finished.
        launch: the record _take_step_launch returned, naming the owner.
        succeeded: True when the process exited zero.
        exit_code: the process exit code, for the failure note.
    Returns:
        '' when the owner's record was written, else one sentence saying why
        it could not be. The owner's project.json is loaded fresh from disk
        and saved back: it is not the open project, so no in-memory copy of
        it exists here to keep in step.
    """
    project_dir = str((launch or {}).get("project_dir") or "")
    if not project_dir:
        return "that project's folder is not on record"
    try:
        owner = pm.load_project(project_dir)
    except (OSError, ValueError, KeyError) as exc:
        return f"its project.json at {project_dir} could not be read ({exc})"
    st = (owner.get("steps") or {}).get(str(step))
    if st is None:
        return f"its project.json at {project_dir} has no step {step}"
    try:
        if succeeded:
            _vic_end_step(step, "success", state=owner)
            pm.complete_step(owner, step)
            _stamp_finished_training_run(owner, step, st)
        else:
            _vic_end_step(step, "failed", notes=f"exit_code={exit_code}", state=owner)
            st["status"] = "error"
            pm.save_project(owner)
    except OSError as exc:
        return f"its project.json at {project_dir} could not be written ({exc})"
    return ""


def _stamp_finished_training_run(owner, step, step_state):
    """Record the split a just-finished Train run was scored on, into the run itself.

    Parameters:
        owner: the project record that owns the run.
        step: the step that finished.
        step_state: that step's record (dir and config).
    Returns:
        None. Only a finished Train step (internal step 6) is stamped; the run
        folder is <step dir>/runs/<run_name> from the config the launch
        saved. A missing folder or an unwritable record is noted and never
        fails the completion, because the completion is already on disk.
    """
    if str(step) != "6":
        return
    config = step_state.get("config") if isinstance(step_state, dict) else None
    run_name = str((config or {}).get("run_name") or "").strip()
    step_dir = str((step_state or {}).get("dir") or "")
    if not run_name or not step_dir:
        vic.note(f"[orch] finished Train run of {owner.get('name')} carries no run_name or dir to stamp")
        return
    run_dir = os.path.join(step_dir, "runs", run_name)
    if not os.path.isdir(run_dir):
        vic.note(f"[orch] finished Train run folder {run_dir} is not on disk, so its split was not stamped")
        return
    try:
        _labruns.record_split_fingerprint(run_dir, source=_labruns.SPLIT_RECORD_SOURCE_TRAINING)
    except (OSError, ValueError) as exc:
        vic.note(f"[orch] could not stamp the split of {run_dir}: {exc}")


def _undo_refused_launch(step, refusal, prior):
    """Put a step's recorded state back after its launch was refused as busy.

    step_run stamps the step as running, records a start time and opens a
    VICARIUS process event before it dispatches, because a launch normally
    succeeds. A refusal because the step's process is still alive spawned
    nothing, so those writes are undone here: the event this request opened,
    if any, is closed as failed with the refusal as its note, and the status, start
    time and event id the step carried before the request are written back.
    Without this a second Start Training press marked the live run as errored,
    and the status poller, which completes only a step still marked running,
    could never finish it.

    Parameters:
        step: the integer step key that was refused.
        refusal: the result dict carrying the busy "error" text.
        prior: the step's "status", "started_at" and "_vic_event_id" as they
            were before the request, None where a key was absent.
    Returns:
        None. current_project is updated in memory and saved to disk.
    """
    if current_project is None:
        return
    st = current_project["steps"].get(str(step), {})
    # Close only an event this request opened. process_start answers None
    # when the bridge is off or its write failed, and then the step still
    # carries the live run's own id, which has to stay open.
    opened = st.get("_vic_event_id")
    if opened is not None and opened != prior.get("_vic_event_id"):
        _vic_end_step(step, "failed", notes=str(refusal.get("error", ""))[:_VIC_NOTE_MAX_CHARS])
    st["status"] = prior["status"]
    for key in ("started_at", "_vic_event_id"):
        if prior.get(key) is None:
            st.pop(key, None)
        else:
            st[key] = prior[key]
    pm.save_project(current_project)


def _busy_refusal_for(step, result):
    """The user-facing body for a launch refused because `step` is busy.

    Names the tile the reader sees, says what to do (a CLI stage finishes on
    its own; a Flask sub-app never does and has a Stop button), and, when the
    live process outlived an orchestrator restart, names its pid so it can be
    stopped by hand, since the runner no longer owns it.

    Parameters:
        step: the integer step key that was refused.
        result: the busy result from the probe or the runner, carrying
            "busy": True and, for an orphan, "orphan_pid".
    Returns:
        a dict with "error", "busy": True and, for an orphan, "orphan_pid".
    """
    advice = FLASK_BUSY_ADVICE if runner.ports.get(step) else BUSY_ADVICE
    refusal = busy_result(step, label=step_ref(step), advice=advice)
    orphan_pid = result.get("orphan_pid")
    if orphan_pid:
        refusal["orphan_pid"] = orphan_pid
        refusal["error"] += ORPHAN_NOTE.format(pid=orphan_pid)
    return refusal


def _step6_driver_alive_outside_runner(step_dir, proc_root=None):
    """The pid of a Step 6 driver still running over step_dir that the
    runner does not know about, or None.

    The restart script kills the orchestrator with SIGKILL, and a training
    child spawned in its own session survives that. After the restart the
    runner's process table is empty, so the busy probe alone would let a new
    press rebuild the dataset under the surviving trainer. This scans the
    kernel's process list for a command line that names both the Step 6
    driver script and this step directory as separate arguments, which is
    how the orchestrator invokes it.

    Parameters:
        step_dir: path of the step's output directory.
        proc_root: the process listing to scan; None means PROC_ROOT. Tests
            point it at a fixture.
    Returns:
        the matching pid as an int, or None when no such process exists or
        the listing cannot be read.
    """
    root = proc_root or PROC_ROOT
    wanted_dir = os.path.normpath(step_dir)
    try:
        entries = os.listdir(root)
    except OSError:
        return None
    for entry in entries:
        if not (entry.isascii() and entry.isdigit()):
            continue
        try:
            with open(os.path.join(root, entry, "cmdline"), "rb") as fh:
                argv = [a.decode("utf-8", "replace") for a in fh.read().split(b"\0") if a]
        except OSError:
            continue
        names_driver = any(os.path.basename(a) == STEP6_DRIVER_NAME for a in argv)
        names_dir = any(os.path.normpath(a) == wanted_dir for a in argv)
        if names_driver and names_dir:
            return int(entry)
    return None


def _newest_run_name(step_dir):
    """The name of the newest run folder under <step_dir>/runs, or ''.

    Parameters:
        step_dir: the Train step's output folder.
    Returns:
        the folder name with the most recent modification time, which is the
        run a live trainer is writing into, and '' when the folder holds no
        run or cannot be listed. Used only to name a run in a message, never
        to decide what to do with it.
    """
    runs_dir = os.path.join(step_dir or "", "runs")
    try:
        names = [n for n in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, n))]
    except OSError:
        return ""
    newest, newest_mtime = "", -1.0
    for name in names:
        try:
            mtime = os.path.getmtime(os.path.join(runs_dir, name))
        except OSError:
            continue
        if mtime > newest_mtime:
            newest, newest_mtime = name, mtime
    return newest


def _live_training_run(step, step_dir, cfg=None):
    """The process writing into step_dir right now, or None.

    Two processes count, and Start Training already refuses for both: one the
    runner owns, and one that outlived an orchestrator restart and is found in
    the kernel's process list (_step6_driver_alive_outside_runner, which
    recognises the Train step's driver by name and by the step folder it was
    handed). Reset, Stop and the status poll consult this one probe, so the
    four cannot disagree about whether something is alive.

    Parameters:
        step: the step key being probed.
        step_dir: that step's output folder.
        cfg: the step's saved config, whose run_name names the run the last
            launch started. None means look the name up on disk only.
    Returns:
        {"run_name": str, "pid": int or None, "owned": bool} when a trainer
        is alive: owned True for a process the runner can still stop, False
        for one that outlived a restart, whose pid is then the only way to
        stop it. None when nothing is writing into step_dir.
    """
    if runner.is_running(step):
        return {"run_name": str((cfg or {}).get("run_name") or "") or _newest_run_name(step_dir),
                "pid": None, "owned": True}
    if not step_dir:
        # Without a folder there is nothing to match a command line against,
        # and an empty path normalises to "." which could match an unrelated
        # process. A step with no recorded folder has no orphan by definition.
        return None
    orphan_pid = _step6_driver_alive_outside_runner(step_dir)
    if orphan_pid:
        return {"run_name": str((cfg or {}).get("run_name") or "") or _newest_run_name(step_dir),
                "pid": orphan_pid, "owned": False}
    return None


def _sam3_drive(port, input_dir, export_dir, categories, batch_size):
    """Background driver: wait for sub-app health, configure, process, poll.

    Runs after _run_step5 launches the sub-app. Pushes progress into
    `sam3_status` which the frontend polls via /api/step/5/sam3_status.
    """
    try:
        _sam3_set(phase="waiting_for_sub_app", running=True, error=None,
                  message="Waiting for SAM3 sub-app (loading models, can take minutes)...")
        t0 = time.time()
        healthy = False
        while time.time() - t0 < 600:  # 10 min ceiling for model loading
            try:
                with urlopen(f"http://localhost:{port}/api/status", timeout=3) as r:
                    if r.status == 200:
                        healthy = True
                        break
            except Exception:
                pass
            time.sleep(2)
        if not healthy:
            _sam3_set(phase="error", running=False,
                      error="SAM3 sub-app did not become healthy within 10 minutes")
            return

        _sam3_set(phase="configuring", message="Loading sam_click_prompts and building queue...")
        body = json.dumps({
            "input_dir": input_dir,
            "export_dir": export_dir,
            "categories": categories,
            "review_batch_size": batch_size,
        }).encode()
        req = Request(f"http://localhost:{port}/api/configure", data=body,
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=120) as r:
            cfg_data = json.loads(r.read())
        if cfg_data.get("error"):
            _sam3_set(phase="error", running=False,
                      error=f"configure failed: {cfg_data['error']}")
            return

        total = int(cfg_data.get("to_process") or 0)
        already = int(cfg_data.get("already_processed") or 0)
        _sam3_set(total=total,
                  message=f"{total} to segment, {already} already done. Starting...")

        # If everything is already done, skip the process call.
        if cfg_data.get("phase") == "review" or total == 0:
            _sam3_set(phase="review_ready", running=False,
                      message=f"Already segmented. Ready for review ({cfg_data.get('review_count', 0)} frames).")
            return

        # Kick off background processing on the sub-app.
        _sam3_set(phase="segmenting", message="SAM3 running...")
        req = Request(f"http://localhost:{port}/api/process", data=b"{}",
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=15):
            pass

        # Poll progress until done.
        while True:
            try:
                with urlopen(f"http://localhost:{port}/api/process_status", timeout=10) as r:
                    d = json.loads(r.read())
                _sam3_set(
                    processed=int(d.get("processed") or 0),
                    total=int(d.get("total") or 0),
                    running=bool(d.get("running", False)),
                )
                if d.get("done") or (not d.get("running") and d.get("processed", 0) >= d.get("total", 0) > 0):
                    _sam3_set(phase="review_ready", running=False,
                              message="Segmentation complete. Ready for review.")
                    return
            except Exception:
                # Transient errors are fine — the sub-app may be busy with GPU work.
                pass
            time.sleep(3)
    except Exception as e:
        _sam3_set(phase="error", running=False, error=str(e))


def _expertids_paths():
    """Effective paths for the native Add-Expert-IDs blueprint: the active
    project's step-5 export dir + the pipeline's review/library config. Read live
    each request so it tracks the currently-loaded project."""
    # Prefer the 4.test combined-annotator export (step4test replaces Steps 4+5)
    # when that project has a segmentations/ tree; else fall back to the legacy
    # Step-5 dir so older Step-5-annotated projects still round-trip.
    export_dir = ""
    if current_project is not None:
        d4 = _step4test_export_dir()
        # Prefer 4.test only when it actually HOLDS segmentation data (an empty /
        # aborted-routing step4test/segmentations/ dir must not shadow a real
        # Step-5 export). Matches importer._iter_segmentation_files' access pattern.
        if d4 and glob.glob(os.path.join(d4, "segmentations", "*", "segmentations.json")):
            export_dir = d4
        else:
            export_dir = current_project["steps"].get("5", {}).get("dir", "") or ""
    # Projects root for multi-project CSV routing (resolve project_id -> export
    # dir). The blueprint falls back to importer.default_projects_root() if
    # absent, but provide it explicitly + the site-codes table for full names.
    _projects_root = os.path.join(os.path.dirname(REPO_DIR), "inprocess")
    _site_codes = os.path.join(REPO_DIR, "supporting_data", "site_codes.csv")
    return {
        "export_dir": export_dir,
        "review_dir": PATHS.get("review_dir", ""),
        "library_dir": PATHS.get("expert_library_dir", "") or "",
        "review_repo_url": PATHS.get("review_repo_url", ""),
        "master_codes": PATHS.get("master_codes_csv", ""),
        "overlap_thresh": float(PATHS.get("review_overlap_thresh", 0.5) or 0.5),
        "git_push": os.environ.get("TCRMP_REVIEW_GIT_PUSH", "1") == "1",
        "projects_root": _projects_root if os.path.isdir(_projects_root) else "",
        "site_codes": _site_codes if os.path.isfile(_site_codes) else "",
    }


def _matrix_paths():
    """Effective paths for the native Label Coverage Matrix blueprint. The
    builder rescans this `inprocess` projects root on every /matrix/api/data
    call, so a purged project simply stops contributing. Same inprocess root
    that _expertids_paths() computes for its projects_root. master_codes_csv
    and site_codes_csv resolve the bare codes the matrix shows to full names
    (spec 2026-09-02-workflow-page-design.md section 6)."""
    return {
        "inprocess_root": os.path.join(os.path.dirname(REPO_DIR), "inprocess"),
        "master_codes_csv": PATHS.get("master_codes_csv", ""),
        "site_codes_csv": os.path.join(REPO_DIR, "supporting_data", "site_codes.csv"),
    }


def _labels_paths():
    """Effective paths for the native Label Manager blueprint. The canonical
    species vocabulary + all-points CSV come straight from the orchestrator's
    resolved pipeline.yaml PATHS (so they track config), with the recode output
    dir and the supporting_data duplicate mirror anchored to the repo."""
    return {
        "master_codes_csv": PATHS.get("master_codes_csv", ""),
        "duplicate_master_codes": PATHS.get("duplicate_master_codes", ""),
        "all_points_csv": PATHS.get("all_points_csv", ""),
        "recode_output_dir": os.path.join(REPO_DIR, "scripts", "TCRMPcvr_recodeSpecies", "output"),
        "pipeline_yaml": CONFIG_PATH,
    }


# Minimal standalone page served on every non-API path while the module lock
# is active (see lock_status.py). Inline CSS + inline SVG only: while locked,
# nothing but /static is served, so the page must carry everything itself.
_LOCK_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reef Point Seg (locked)</title>
<style>
  html, body { margin: 0; height: 100%; }
  body { background: #10151c; color: #dfe7ef;
         font-family: system-ui, -apple-system, sans-serif;
         display: flex; align-items: center; justify-content: center; }
  .vic-lock-page { text-align: center; padding: 40px; max-width: 520px; }
  .vic-lock-page svg { display: block; margin: 0 auto 18px; }
  .vic-lock-page h1 { font-size: 22px; font-weight: 600; margin: 0 0 12px; }
  .vic-lock-page p { font-size: 15px; line-height: 1.6; margin: 0; color: #b8c4d0; }
  .vic-lock-page a { color: #7fb3e8; }
  .vic-lock-footer { margin-top: 28px; font-size: 12px; color: #5a6673; }
</style>
</head>
<body>
<div class="vic-lock-page">
  <svg xmlns="http://www.w3.org/2000/svg" width="56" height="56"
       viewBox="0 0 24 24" fill="none" stroke="#8fa4b8" stroke-width="1.6"
       stroke-linecap="round" stroke-linejoin="round" role="img"
       aria-label="Locked">
    <rect x="4" y="10.5" width="16" height="10" rx="2"></rect>
    <path d="M8 10.5V7a4 4 0 0 1 8 0v3.5"></path>
    <circle cx="12" cy="15.5" r="1.4" fill="#8fa4b8" stroke="none"></circle>
  </svg>
  <h1>Reef Point Seg</h1>
  <p>Under development. Please come back later.
     <a href="mailto:lauren.olinger@uvi.edu">Email Lauren</a> for questions.</p>
  <div class="vic-lock-footer">VICARIUS module lock</div>
</div>
</body>
</html>"""


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )

    # ── Module-lock gate (VICARIUS platform contract) ──────────────────
    # Registered before the blueprints mount so every route in the app,
    # blueprint or not, is refused while the platform registry marks this
    # module locked (lock_status.is_locked(), read fresh per request).
    # Because before_request runs ahead of any view, index()'s ?project_dir=
    # auto-open can never execute while locked. /static stays served so the
    # lock page itself can render assets.
    def _is_browser_navigation():
        """Whether this request is a person opening a page, not code calling an API.

        Only a GET can be one. An API path never is: this window's own data
        routes sit under /api/, and every dashboard mounted beside it
        (/labels/, /matrix/, /expertids/, /boxes/) is a page at the root of
        its blueprint with its data routes under <prefix>/api/, so the two
        are told apart by the shape of the path and never by a list of page
        names that a new dashboard would silently fall off.

        What is left is a page unless the caller said otherwise: a request
        that names an Accept type and leaves HTML out of it is code asking
        for data, and it needs a status code rather than a web page. A
        request that names no Accept type at all is treated as a page, which
        is what a bare fetch of a page path is.

        Returns:
            True for a page a browser is opening, which must be answered with
            the lock page; False for everything else, which must be answered
            with 423.
        """
        if request.method != "GET":
            return False
        if request.path.startswith("/api/") or "/api/" in request.path:
            return False
        if len(request.accept_mimetypes) and not request.accept_mimetypes.accept_html:
            return False
        return True

    @app.before_request
    def _module_lock_gate():
        if not lock_status.is_locked():
            return None
        if request.path.startswith("/static/"):
            return None
        # A person navigating to any page of this module gets the page the
        # platform contract names, whether that page is the main window or
        # one of the dashboards mounted beside it. Three of the four used to
        # answer a navigation with raw JSON.
        if _is_browser_navigation():
            return render_template_string(_LOCK_PAGE_HTML)
        return (
            jsonify({"error": "locked", "message": lock_status.LOCK_MESSAGE}),
            423,
        )

    # Add Expert IDs — mounted natively (its routes live under /expertids, its UI
    # is included into the step-6 panel). No subprocess, no iframe.
    app.register_blueprint(
        _make_expertids_bp(_expertids_paths,
                           log_fn=lambda m: vic.note(f"[expertids] {m}")),
        url_prefix="/expertids",
    )

    # Label Coverage Matrix — mounted natively as its own full page under
    # /matrix (cross-project image x label grid, live from inprocess/).
    # "Open project folder" from a matrix cell runs the same host spawn as the
    # orchestrator's own folder routes (host_open.open_in_file_manager, D8).
    app.register_blueprint(
        _make_matrix_bp(_matrix_paths,
                        log_fn=lambda m: vic.note(f"[matrix] {m}"),
                        open_folder_fn=host_open.open_in_file_manager),
        url_prefix="/matrix",
    )

    # Label Manager — mounted natively as its own full page under /labels
    # (canonical species vocabulary editor + remap composer + manual recode).
    app.register_blueprint(
        _make_labels_bp(_labels_paths,
                        log_fn=lambda m: vic.note(f"[labels] {m}")),
        url_prefix="/labels",
    )

    # Bounding Boxes: external box datasets, mounted natively as its own full
    # page under /boxes. It stages a third-party detection dataset into
    # supporting_data/bbox_datasets, converts its boxes to outlines with SAM3
    # into inprocess/_bbox, trains a detector on the staged dataset with its
    # own project-free job runner, and mints a training project so the same
    # dataset can be trained from this window (the Step 5 source list offers
    # staged box datasets beside the composites). None of that imagery is
    # TCRMP survey data: it never enters a project, the per-mask registry,
    # the coverage matrix, or STATS. Nothing here shares the pipeline
    # StageRunner or its step keys.
    from _boxes import make_blueprint as _make_boxes_bp

    def _boxes_paths():
        return {
            "bbox_root": _bbox_root(),
            "seg_root": os.path.join(_inprocess_root(), "_bbox"),
            "scripts_dir": os.path.join(REPO_DIR, "scripts"),
            "python": PYTHON_PATHS.get(5) or os.path.join(REPO_DIR, "env", "bin", "python"),
            # Where "Create a training project" mints its run_* dir. The same
            # folder the VICARIUS launcher mints into, so a project made there
            # opens the same way as any other.
            "inprocess_root": _inprocess_root(),
        }

    app.register_blueprint(
        _make_boxes_bp(_boxes_paths, log_fn=lambda m: vic.note(f"[boxes] {m}")),
        url_prefix="/boxes",
    )

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    # Path prefix that any auto-open project_dir must live under. This
    # prevents a crafted ?project_dir= URL param from pointing at arbitrary
    # filesystem locations (e.g. /etc/passwd). Derived from the configured
    # projects_dir (pipeline.yaml paths.projects_dir), falling back to the
    # module-standard inprocess/ sibling of the repo.
    _AUTOOPEN_ALLOWED_ROOT = (
        PATHS.get("projects_dir")
        or os.path.join(os.path.dirname(REPO_DIR), "inprocess")
    ).rstrip("/") + "/"

    @app.route("/")
    def index():
        global current_project
        from orchestrator_config import PATHS

        # Optional auto-open: /?project_dir=<abs-path>[&purpose=<text>]
        raw = request.args.get("project_dir", "")
        if raw:
            try:
                candidate = os.path.abspath(unquote(raw))
                # Guardrails: must live under the module's inprocess/ tree
                # and contain a project.json to be considered valid.
                if (
                    candidate.startswith(_AUTOOPEN_ALLOWED_ROOT)
                    and os.path.isfile(os.path.join(candidate, "project.json"))
                ):
                    # Idempotent: don't reload if the same project is already open.
                    already = (
                        current_project is not None
                        and current_project.get("project_dir") == candidate
                    )
                    if not already:
                        current_project = _load_project(candidate)
                        vic.note(
                            f"[orch] project '{current_project['name']}' "
                            f"opened via URL param"
                        )
                        raw_purpose = request.args.get("purpose", "")
                        if raw_purpose:
                            purpose = unquote(raw_purpose).strip()
                            if purpose:
                                vic.note(f"[orch] auto-open purpose: {purpose}")
            except Exception:
                # Never block page render on auto-open failure.
                pass

        # When a project is already loaded (either via the ?project_dir= auto-open
        # above, or via a prior POST /api/project/open from the VICARIUS UI), tell
        # the template to skip the landing/intro screen and render the project view
        # directly. The intro block is still emitted but pre-hidden so the
        # "Switch Project" button can re-reveal it without a page reload.
        return render_template(
            "index.html",
            asset_version=ASSET_VERSION,
            repo_root=REPO_DIR,
            projects_dir=PATHS.get("projects_dir", os.path.join(REPO_DIR, "projects")),
            supporting_data_dir=PATHS.get("supporting_data_dir", ""),
            all_points_csv=PATHS.get("all_points_csv", ""),
            master_codes_csv=PATHS.get("master_codes_csv", ""),
            project_loaded=(current_project is not None),
            project_state=_state_payload(current_project),
        )

    # ------------------------------------------------------------------
    # Project endpoints
    # ------------------------------------------------------------------
    # NOTE: there is deliberately no /api/project/create route. Projects are
    # minted as run_* dirs by the VICARIUS launcher (vicarius_ui_os
    # module_views.api_module_new_run) and arrive here via /api/project/open
    # or the guarded GET /?project_dir=... auto-open. The old in-orchestrator
    # create path (removed 2026-08-14) used an incompatible YY-MM-DD_{name}
    # naming scheme and had no remaining caller.
    @app.route("/api/project/open", methods=["POST"])
    def project_open():
        global current_project
        data = request.get_json(force=True)
        project_dir = data.get("project_dir", "")
        project_dir = project_dir.strip() if isinstance(project_dir, str) else ""
        if not project_dir:
            return jsonify({"error": "Project directory is required"}), 400
        if "\x00" in project_dir:
            return jsonify({"error": "Project directory holds a null byte"}), 400
        # Containment to the inprocess root is deliberately not applied here:
        # house suites open fixtures from temp folders and projects may sit
        # under _TEST/. Recorded as TODO T20260917-152000 for a decision.
        try:
            switching = (
                current_project is not None
                and os.path.abspath(current_project.get("project_dir", ""))
                != os.path.abspath(project_dir)
            )
            current_project = _load_project(project_dir)
            if switching:
                # A reopened SAME project keeps its live annotator state; a
                # DIFFERENT project must not inherit the old one's ui_ready.
                _reset_transient_status("Project switched.")
            vic.note(f"[orch] project '{current_project['name']}' reopened")
            return jsonify({"success": True, "state": _state_payload(current_project)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/state")
    def project_state():
        if current_project is None:
            return jsonify({"loaded": False})
        return jsonify({"loaded": True, "state": _state_payload(current_project)})

    @app.route("/api/project/config/<int:step>", methods=["PUT"])
    def project_config(step):
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        data = request.get_json(force=True)
        s = str(step)
        if s not in current_project["steps"]:
            return jsonify({"error": f"Invalid step {step}"}), 400
        current_project["steps"][s]["config"].update(data)
        pm.save_project(current_project)
        # save_project may have rebuilt an auto project name (name_auto);
        # hand the current name back so the UI can refresh the topbar.
        return jsonify({"success": True, "name": current_project.get("name", "")})

    @app.route("/api/project/name-preview")
    def project_name_preview():
        """Preview the auto project name for a candidate species selection.

        ?species=PA,SS -> {"auto": true, "name": "LO_PA-SS_714img_26-08-26"}
        computed from the query's species plus the project's real initials,
        creation date, and current selected_frames.csv count. Never mutates or
        saves current_project: the real name is rebuilt by pm.save_project on
        the next config PUT / step completion. Hand-named projects (no
        name_auto) answer {"auto": false, "name": <current name>}. Gated by
        the module-lock before_request like every other /api/ route.
        """
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        current_name = current_project.get("name", "")
        if not current_project.get("name_auto"):
            return jsonify({"auto": False, "name": current_name})
        initials = pm.initials_for_state(current_project)
        if not initials:
            return jsonify({"auto": False, "name": current_name})
        species = pm.species_from_config(request.args.get("species", ""))
        # The image count belongs to the labels Step 3 actually ran with: show
        # it only while Step 3 is completed AND the candidate selection matches
        # the saved target_species. A different pick previews without a count
        # rather than pairing new labels with the old selection's number.
        saved = pm.species_from_config(
            (((current_project.get("steps") or {}).get("3") or {}).get("config") or {})
            .get("target_species"))
        n_images = (pm.n_images_for_state(current_project)
                    if set(species) == set(saved) else None)
        name = pm.auto_project_name(
            initials, species, n_images, current_project.get("created_at"))
        return jsonify({"auto": True, "name": name})

    @app.route("/api/project/quit", methods=["POST"])
    def project_quit():
        global current_project
        # Save first (D1): runner.kill_all() below also stops any live
        # combined annotator, Refine annotator or Edit Masks, so each is
        # flushed before that kill reaches it. _live_annotator_port confirms
        # runner.is_running(key) first, so a dead key's stale port (since
        # reused by an unrelated process) is never flushed.
        for _annotator_key in ("step4test", "step4loop", "editmasks"):
            _port = _live_annotator_port(_annotator_key)
            if _port:
                _ok, _reason = _flush_annotator(_port)
                if not _ok:
                    vic.note(f"[orch] annotator flush failed on port {_port} "
                             f"before quit: {_reason}")
        runner.kill_all()
        _reset_transient_status("Project closed.")
        if current_project:
            vic.note(f"[orch] project '{current_project['name']}' closed")
            pm.save_project(current_project)
        current_project = None
        return jsonify({"success": True})

    # ------------------------------------------------------------------
    # Step execution endpoints
    # ------------------------------------------------------------------
    # Steps a composite training project never runs. Its annotation work was
    # done in the source projects, so the point-fetch / label-fetch / image-
    # choosing steps have nothing to act on. Train (6), Evaluate (7) and
    # Inference (8) stay fully available.
    _COMPOSITE_REFUSED_STEPS = {1, 2, 3}
    # Reset refuses further than run does. pm.reset_step clears every step
    # from the given one to the last AND relocks the ones after it, so a reset
    # of any pre-Train slot would relock Train itself and wipe the "skipped"
    # markers this project is built on. Train (6) and everything after it stay
    # resettable, which is what a bad training run needs.
    _COMPOSITE_REFUSED_RESET_STEPS = {1, 2, 3, 4, 5}

    def _refuse_for_composite():
        """The 400 body every annotation route answers on a training project,
        or None when the current project is an ordinary one."""
        if _is_composite_training(current_project):
            return jsonify({"error": COMPOSITE_REFUSAL}), 400
        return None

    @app.route("/api/step/<int:step>/run", methods=["POST"])
    def step_run(step):
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        if step in _COMPOSITE_REFUSED_STEPS:
            refused = _refuse_for_composite()
            if refused is not None:
                return refused

        s = str(step)
        st = current_project["steps"].get(s)
        if not st:
            return jsonify({"error": f"Invalid step {step}"}), 400
        if st["status"] == "locked":
            return jsonify({"error": f"Step {step} is locked"}), 400

        cfg = st["config"]
        step_dir = st["dir"]
        os.makedirs(step_dir, exist_ok=True)

        # What a refused launch gives back. The step is stamped as running
        # and a VICARIUS event is opened before anyone knows whether the
        # step's process from an earlier press is still alive; see
        # _undo_refused_launch.
        prior = {"status": st["status"],
                 "started_at": st.get("started_at"),
                 "_vic_event_id": st.get("_vic_event_id")}

        st["status"] = "running"
        st["started_at"] = datetime.now().isoformat()

        # VICARIUS process_start — stash event_id so step completion can link to it.
        step_name = f"step{step}_{pm.STEP_DIRS.get(s, '').split('_', 1)[-1] or 'stage'}"
        purpose = cfg.get("purpose") or f"{st.get('name', 'step')}" + (
            f" (project '{current_project.get('name', '')}')"
        )
        event_id = vic.process_start(step_name, purpose,
                                     inputs=[str(v) for v in cfg.values() if isinstance(v, str)],
                                     notes="")
        if event_id is not None:
            st["_vic_event_id"] = event_id
        pm.save_project(current_project)

        try:
            if step == 1:
                result = _run_step1(step, cfg, step_dir)
            elif step == 2:
                result = _run_step2(step, cfg, step_dir)
            elif step == 3:
                result = _run_step3(step, cfg, step_dir)
            elif step == 4:
                result = _run_step4(step, cfg, step_dir)
            elif step == 5:
                result = _run_step5(step, cfg, step_dir)
            elif step == 6:
                result = _run_step6(step, cfg, step_dir)
            elif step == 7:
                result = _run_step7(step, cfg, step_dir)
            elif step == 8:
                result = _run_step8(step, cfg, step_dir)
            else:
                return jsonify({"error": "Invalid step"}), 400

            if "error" in result:
                if result.get("busy"):
                    # The step's process from an earlier press is still
                    # alive and nothing was spawned: name the tile the
                    # reader sees, give back what this request stamped
                    # over, and leave the live run alone. The launch record
                    # belongs to the press that really started that process
                    # and is left exactly as it is.
                    refusal = _busy_refusal_for(step, result)
                    _undo_refused_launch(step, refusal, prior)
                    return jsonify(refusal), HTTP_CONFLICT
                st["status"] = "error"
                _vic_end_step(step, "failed",
                              notes=str(result.get("error", ""))[:_VIC_NOTE_MAX_CHARS])
                pm.save_project(current_project)
                http_status = int(result.pop("http_status", HTTP_SERVER_ERROR) or HTTP_SERVER_ERROR)
                return jsonify(result), http_status

            # The process is away. Which project it belongs to is decided
            # here, once, so the status poll records the result against the
            # project that pressed the button and not against whichever one
            # happens to be open when it finishes. The pid ties the record to
            # this one process.
            _remember_step_launch(step, result.get("pid"))

            if step == EVALUATE_STEP:
                # The evaluation is away. Add what the step looked like before
                # this request stamped it running, so the poll that sees it
                # finish can put the project's own step back exactly as it
                # was when the run turns out to be someone else's.
                _remember_evaluation_prior_state(prior)
            return jsonify({"success": True, **result})
        except Exception as e:
            st["status"] = "error"
            _vic_end_step(step, "failed", notes=str(e)[:_VIC_NOTE_MAX_CHARS])
            pm.save_project(current_project)
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), HTTP_SERVER_ERROR

    # Add Expert IDs is mounted natively via the _expertids blueprint registered
    # above (routes under /expertids, UI included into the step-6 panel). The old
    # subprocess-launch + status endpoints were retired in favor of that.

    def _run_step1(step, cfg, step_dir):
        py = PYTHON_PATHS.get(1, "python3")
        if not os.path.isfile(py):
            py = "python3"
        cmd = [py, ENTRY_POINTS[1], cfg.get("input_dir", ""), step_dir]
        return runner.run_cli_stage(step, cmd, cwd=WORKING_DIRS[1])

    def _run_step2(step, cfg, step_dir):
        port = find_free_port(cfg.get("port", 5055))
        py = PYTHON_PATHS.get(2, "python3")
        if not os.path.isfile(py):
            py = "python3"
        ap = cfg.get("all_points", "")
        mc = cfg.get("master_codes", "")
        cmd = [py, ENTRY_POINTS[2], "--port", str(port), "--all-points", ap, "--master-codes", mc]
        result = runner.run_flask_stage(step, cmd, port, cwd=WORKING_DIRS[2])
        # A refused launch spawned nothing, so the port picked above is not
        # the one the live sub-app answers on and must not be saved.
        if "error" not in result:
            current_project["steps"]["2"]["config"]["port"] = port
            pm.save_project(current_project)
        return result

    def _resolve_max_site_frac(raw):
        """Validate the Max share per site value before it reaches the CLI.

        /api/project/config PUT accepts any JSON value for max_site_frac with
        no server-side check, so `raw` here could be a stray string, None, a
        negative number, or something >1 (bad manual API call, corrupted
        saved project, a future UI bug) - the HTML min/max attributes and
        collectConfig's isNaN guard are cosmetic/client-only and never reach
        this path. Non-numeric/NaN falls back to select_images.py's own
        default (0.25); a numeric value <=0 or >1 clamps to the nearest bound
        of [0.05, 1.0] rather than handing select_images.py's --max-site-frac
        (its own argparse takes any float) a value that breaks the cap math.
        Returns (value, clamped) so the caller can log when it had to step in.
        """
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return 0.25, True
        if v != v:  # NaN != NaN
            return 0.25, True
        if v <= 0:
            return 0.05, True
        if v > 1:
            return 1.0, True
        return v, False

    def _write_prior_screening_input(step_dir):
        """Write <step_dir>/prior_screening_input.json ({stem: [labels]}),
        from the coverage matrix, for Step 3's --prior-screening flag: which
        labels are already screened, in another project, for a frame this
        project might select (2026-09-27 VEGA annotator tidy-up plan).
        Excludes the open project by both id and directory (see
        _reefreview.screening.screened_by_stem).

        Never raises: a matrix read problem, or having no project open at
        all, is logged via vic.note and answered with "", which
        _run_step3 reads as "skip the flag, run exactly as before this
        feature existed."

        Parameters:
            step_dir: Step 3's output directory (already created by the
                caller); the file is written directly under it.
        Returns:
            str: the path written, or "" on any failure.
        """
        if current_project is None:
            return ""
        try:
            mapping = _screening.screened_by_stem(
                _inprocess_root(),
                exclude_project_id=current_project.get("id", ""),
                exclude_project_dir=current_project.get("project_dir", ""),
            )
            path = os.path.join(step_dir, "prior_screening_input.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(mapping, fh, indent=2)
            return path
        except Exception as e:
            vic.note(f"[orch] failed to write prior_screening_input.json for step 3: {e}")
            return ""

    def _run_step3(step, cfg, step_dir):
        py = PYTHON_PATHS.get(3, "python3")
        if not os.path.isfile(py):
            py = "python3"
        species = [s.strip() for s in cfg.get("target_species", "").split(",") if s.strip()]
        cmd = [
            py, ENTRY_POINTS[3],
            "--species", *species,
            "--target", str(cfg.get("target_instances", 1000)),
            "--all-points", cfg.get("all_points", ""),
            "--master-codes", cfg.get("master_codes", ""),
        ]
        # Year bounds now flow from the UI form through to the CLI.
        if cfg.get("min_year") is not None:
            cmd += ["--min-year", str(cfg.get("min_year"))]
        if cfg.get("max_year") is not None:
            cmd += ["--max-year", str(cfg.get("max_year"))]

        # Site balance is ON by default in select_images.py; only pass the
        # disable flag when the user actually unchecked it. --max-site-frac is
        # always passed explicitly so the run's config_snapshot/selection_summary
        # always reflect the value the UI showed, not the script's own default.
        if cfg.get("site_balance") is False:
            cmd += ["--no-site-balance"]
        raw_max_site_frac = cfg.get("max_site_frac", 0.25)
        max_site_frac, clamped = _resolve_max_site_frac(raw_max_site_frac)
        if clamped:
            vic.note(f"[orch] step 3 max_site_frac {raw_max_site_frac!r} is not a usable "
                     f"value in (0, 1]; using {max_site_frac} instead")
        cmd += ["--max-site-frac", str(max_site_frac)]

        # No-re-screening gate (2026-09-27 VEGA annotator tidy-up plan): a
        # write failure is already logged inside the helper and answered
        # with "", so a matrix problem here degrades to "run Step 3 exactly
        # as it ran before this feature existed" rather than blocking it.
        prior_screening_path = _write_prior_screening_input(step_dir)
        if prior_screening_path:
            cmd += ["--prior-screening", prior_screening_path]

        env_extra = {"TCRMP_OUTPUT_DIR": step_dir}
        return runner.run_cli_stage(step, cmd, cwd=WORKING_DIRS[3], env_extra=env_extra)

    def _get_target_species_str():
        """Get comma-separated target species from step 3 config.

        No hardcoded fallback: target labels are chosen by the user in the
        Step 3 picker (nothing is selected by default). The UI blocks running
        Step 3 with an empty selection, so an empty string here means the user
        ran without picking labels rather than a default set to silently inject.
        """
        s3 = current_project["steps"].get("3", {}).get("config", {})
        return s3.get("target_species", "")

    def _get_remap_log_path():
        """Find the remap_log.json from step 2 output."""
        s2_dir = current_project["steps"].get("2", {}).get("dir", "")
        remap = os.path.join(s2_dir, "remap_log.json")
        if os.path.isfile(remap):
            return remap
        # Fall back to the source remap log configured in step 2
        src = current_project["steps"].get("2", {}).get("config", {}).get("remap_log_source", "")
        if src and os.path.isfile(src):
            return src
        return ""

    def _resolve_path(step_num, config_key, output_key=None):
        """Resolve a path from step config, falling back to step outputs."""
        s = str(step_num)
        step = current_project["steps"].get(s, {})
        val = step.get("config", {}).get(config_key, "")
        if not val and output_key:
            val = step.get("outputs", {}).get(output_key, "")
        return val or ""

    def _run_step4(step, cfg, step_dir):
        port = find_free_port(cfg.get("port", 5065))
        py = PYTHON_PATHS.get(4, "python3")
        if not os.path.isfile(py):
            py = "python3"
        cmd = [py, ENTRY_POINTS[4], "--port", str(port)]

        # Resolve selected_frames from step 3 (config or outputs)
        selected_frames = (
            cfg.get("selected_frames", "")
            or _resolve_path(3, "selected_frames", "selected_frames")
        )
        # Resolve all_points from step 2 (config or outputs)
        all_points = (
            cfg.get("all_points", "")
            or _resolve_path(2, "all_points", "all_points_recoded")
        )

        # Orchestrator URL the sub-app can call back to (e.g. to head-start SAM3).
        # We're inside a Flask request context so `request.host` is safe.
        host_port = request.host.split(":")[-1] if ":" in request.host else "80"
        orchestrator_url = f"http://localhost:{host_port}"

        # Pass ALL project paths so the sub-app reflects current settings
        env_extra = {
            "TCRMP_SELECTED_FRAMES": selected_frames,
            "TCRMP_EXPORT_DIR": step_dir,
            "TCRMP_CLIP_DIR": cfg.get("clip_dir", ""),
            "TCRMP_CPC_DIR": PATHS.get("cpc_all_dir", "") or os.path.join(REPO_DIR, "supporting_data", "cpc_all"),
            "TCRMP_TARGET_SPECIES": _get_target_species_str(),
            "TCRMP_REMAP_LOG": _get_remap_log_path(),
            "TCRMP_ALL_POINTS": all_points,
            "TCRMP_ORCHESTRATOR_URL": orchestrator_url,
            "TCRMP_AUTO_START_SAM3": "1" if cfg.get("auto_start_sam3") else "",
            # These used to be user-selectable on the sub-app startup form.
            # Orchestrator-driven auto-boot bypasses that form, so we relay the
            # values here for the sub-app to pre-populate before calling configure.
            "TCRMP_REFERENCE_MODE": "1" if cfg.get("reference_mode") else "",
            "TCRMP_SHUFFLE": "1" if cfg.get("shuffle") else "",
            "TCRMP_REVIEW_BATCH_SIZE": str(cfg.get("review_batch_size") or "10"),
        }

        # Remove empty values so sub-app uses its own defaults
        env_extra = {k: v for k, v in env_extra.items() if v}

        result = runner.run_flask_stage(step, cmd, port, cwd=WORKING_DIRS[4], env_extra=env_extra)
        current_project["steps"]["4"]["config"]["port"] = port
        pm.save_project(current_project)
        return result

    def _run_step5(step, cfg, step_dir):
        global sam3_thread
        port = find_free_port(cfg.get("port", 5070))
        py = PYTHON_PATHS.get(5, "python3")
        if not os.path.isfile(py):
            py = "python3"
        cmd = [py, ENTRY_POINTS[5], "--port", str(port)]

        # Resolve input_dir: step 4's output dir
        input_dir = cfg.get("input_dir", "")
        if not input_dir:
            s4_dir = current_project["steps"].get("4", {}).get("dir", "")
            if s4_dir:
                input_dir = s4_dir

        # Pass ALL project paths + SAM3 settings so sub-app reflects current config
        env_extra = {
            "TCRMP_INPUT_DIR": input_dir,
            "TCRMP_EXPORT_DIR": step_dir,
            # Canonical project identity (project.json id + human name), so review
            # items + library records are tagged by project instead of being
            # path-derived. Used by the multi-project review site + CSV routing.
            "TCRMP_PROJECT_ID": current_project.get("id", "") if current_project else "",
            "TCRMP_PROJECT_NAME": current_project.get("name", "") if current_project else "",
            "TCRMP_TARGET_SPECIES": _get_target_species_str(),
            "TCRMP_SAM3_DEVICE_TRACKER": cfg.get("sam3_device_tracker", ""),
            "TCRMP_SAM3_DEVICE_EXEMPLAR": cfg.get("sam3_device_exemplar", ""),
            "TCRMP_CONFIDENCE_THRESHOLD": str(cfg.get("confidence_threshold", "")),
            "TCRMP_MIN_MASK_AREA_PX": str(cfg.get("min_mask_area_px", "")),
            "TCRMP_MERGE_DISTANCE_PX": str(cfg.get("merge_distance_px", "")),
            "TCRMP_OVERLAP_STRATEGY": cfg.get("overlap_strategy", ""),
            # Expert-review export targets (REVIEW-flagged masks).
            "TCRMP_REVIEW_DIR": PATHS.get("review_dir", ""),
            "TCRMP_REVIEW_REPO_URL": PATHS.get("review_repo_url", ""),
            "TCRMP_EXPERT_LIBRARY_DIR": PATHS.get("expert_library_dir", ""),
            "TCRMP_REVIEW_CONTACTS": cfg.get("review_contacts", ""),
            # Who is annotating (spec D2): stamped into label_outcomes,
            # label_provenance.csv, each mask and the registry.
            "TCRMP_REVIEWER": _reviewer_initials(),
        }

        # Remove empty values so sub-app uses its own defaults
        env_extra = {k: v for k, v in env_extra.items() if v}

        # Recoded species dictionary (Step 2 output) so RECODED names (e.g.
        # Orbicella, not the obsolete Montastraea annularis) carry through into
        # the step-5 labels + the expert-review codes.json + the library. Without
        # this, step 5 falls back to the canonical pre-recode supporting_data
        # dictionary and leaks un-recoded names to reviewers. Falls back to the
        # canonical dict only when Step 2 hasn't produced a recoded one.
        _s2 = current_project["steps"].get("2", {}) if current_project else {}
        _mc = (_s2.get("outputs", {}).get("master_codes_recoded")
               or _s2.get("outputs", {}).get("master_codes"))
        if not (_mc and os.path.isfile(_mc)):
            _s2dir = _s2.get("dir", "")
            for _cand in ("master_codes_recoded.csv", "master_codes.csv"):
                _p = os.path.join(_s2dir, _cand) if _s2dir else ""
                if _p and os.path.isfile(_p):
                    _mc = _p
                    break
        if _mc and os.path.isfile(_mc):
            env_extra["TCRMP_MASTER_CODES"] = _mc

        result = runner.run_flask_stage(step, cmd, port, cwd=WORKING_DIRS[5], env_extra=env_extra)
        if not (isinstance(result, dict) and result.get("error")):
            _push_reviewer_after_health(port, _reviewer_initials(), runner.health_check)
        current_project["steps"]["5"]["config"]["port"] = port
        pm.save_project(current_project)

        # Kick off the orchestrator-side driver so SAM3 configures and starts
        # segmentation without the user touching the sub-app UI.
        categories = cfg.get("categories") or ["Target species only"]
        # Review batch size is user-selectable in the Step 5 panel; "all" is a
        # sentinel meaning "no batching" (the sub-app slices files[offset:+N],
        # so a huge N effectively returns everything).
        raw_batch = cfg.get("review_batch_size", "10")
        if isinstance(raw_batch, str) and raw_batch.strip().lower() == "all":
            batch_size = 999999
        else:
            try:
                batch_size = int(raw_batch)
            except (TypeError, ValueError):
                batch_size = 10
        _sam3_set(phase="launching", processed=0, total=0, running=True, error=None,
                  message="Launching SAM3 sub-app...")
        sam3_thread = threading.Thread(
            target=_sam3_drive,
            args=(port, input_dir, step_dir, categories, batch_size),
            daemon=True,
        )
        sam3_thread.start()
        return result

    def _routed_dir():
        """The 4.test-owned dir that placePoints routes detections into and the
        combined annotator then consumes (sam_click_prompts.json lives here)."""
        return os.path.join(_step4test_export_dir(), "routed_input")

    def _routed_prompts_exist(routed_dir):
        """True when a prior routing pass already wrote the flat prompts file."""
        return os.path.isfile(os.path.join(routed_dir, "ids", "sam_click_prompts.json"))

    def _route_drive(pp_port, selected_frames, routed_dir, species_filter, project_ctx):
        """Background driver: drive placePoints headlessly to produce
        sam_click_prompts.json, then launch the combined annotator.

        Clone of _sam3_drive but for the routing (CPC/OCR detection) pass:
        wait for health, configure, loop process until done, export_all, kill
        the placePoints stage (frees the GPU), then open the combined UI. Pushes
        progress into `route_status` which the frontend polls via
        /api/step/step4test/route_status.
        """
        def _abort_if_cancelled():
            """True (and cleans up) when the operator cancelled this pass.

            Checked between every step of the route. Killing placePoints frees the
            GPU immediately; the detections it already flushed stay on disk, so the
            next Open resumes from there instead of re-OCRing everything."""
            if not route_cancel.is_set():
                return False
            runner.kill("step4test_route")
            _route_set(phase="cancelled", ui_port=None, error=None,
                       message="Routing cancelled. Reopening resumes where it stopped.")
            return True

        try:
            _route_set(message="Waiting for routing sub-app (placePoints)...")
            t0 = time.time()
            healthy = False
            while time.time() - t0 < 600:  # 10 min ceiling
                if _abort_if_cancelled():
                    return
                try:
                    with urlopen(f"http://localhost:{pp_port}/api/status", timeout=3) as r:
                        if r.status == 200:
                            healthy = True
                            break
                except Exception:
                    pass
                time.sleep(2)
            if _abort_if_cancelled():
                return
            if not healthy:
                _route_set(phase="error",
                           error="Routing sub-app did not become healthy within 10 minutes")
                return

            _route_set(phase="routing_configure",
                       message="Loading selected frames and splitting CPC/OCR...")
            body = json.dumps({
                "selected_frames": selected_frames,
                "export_dir": routed_dir,
                "species_filter": species_filter or [],
                "review_batch_size": 999999,
                "reference_mode": False,
                "shuffle": False,
            }).encode()
            req = Request(f"http://localhost:{pp_port}/api/configure", data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=300) as r:
                cfg_data = json.loads(r.read())
            if cfg_data.get("error"):
                _route_set(phase="error", error=f"configure failed: {cfg_data['error']}")
                return

            # Routing accountability (Task 2): placePoints' configure response
            # reports per-frame drops (dropped[]) plus cpc_missing/image_missing
            # counts. Surface them as human warnings + the raw dropped list so
            # the 4.test overlay/panel can show "N of M frames skipped" without
            # auto-recovery (descoped). Absent fields default to empty/0 so this
            # stays resilient if the placePoints side hasn't landed the report yet.
            dropped = cfg_data.get("dropped") or []
            cpc_missing = int(cfg_data.get("cpc_missing") or 0)
            image_missing = int(cfg_data.get("image_missing") or 0)
            warnings = []
            n_dropped = len(dropped) if isinstance(dropped, list) else 0
            if n_dropped or cpc_missing or image_missing:
                bits = []
                if cpc_missing:
                    bits.append(f"{cpc_missing} missing CPC points")
                if image_missing:
                    bits.append(f"{image_missing} missing images")
                detail = f" ({', '.join(bits)})" if bits else ""
                warnings.append(
                    f"{n_dropped or (cpc_missing + image_missing)} frame(s) "
                    f"skipped during routing{detail}; see routing_report.json."
                )
            _route_set(warnings=warnings,
                       dropped=dropped if isinstance(dropped, list) else [])

            total = int(cfg_data.get("ocr_remaining") or 0)
            _route_set(phase="routing_ocr", processed=0, total=total,
                       message=f"Routing chosen images... 0/{total}")

            # Loop process (one OCR frame per call) until done. CPC-only frames
            # need no OCR, so total may be 0 and this loop exits immediately.
            while True:
                # The long leg of the pass (minutes to tens of minutes), so this is
                # where a cancel almost always lands.
                if _abort_if_cancelled():
                    return
                req = Request(f"http://localhost:{pp_port}/api/process", data=b"{}",
                              headers={"Content-Type": "application/json"}, method="POST")
                try:
                    with urlopen(req, timeout=120) as r:
                        d = json.loads(r.read())
                except Exception:
                    # Transient error (GPU busy) - retry after a beat.
                    time.sleep(2)
                    continue
                if d.get("error"):
                    _route_set(phase="error", error=f"process failed: {d['error']}")
                    return
                proc = int(d.get("processed") or 0)
                tot = int(d.get("total") or total)
                _route_set(processed=proc, total=tot,
                           message=f"Routing chosen images... {proc}/{tot}")
                if d.get("done"):
                    break

            if _abort_if_cancelled():
                return

            _route_set(phase="exporting",
                       message="Writing sam_click_prompts.json for all routed frames...")
            req = Request(f"http://localhost:{pp_port}/api/export_all", data=b"{}",
                          headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=600) as r:
                exp = json.loads(r.read())
            if exp.get("error"):
                _route_set(phase="error", error=f"export_all failed: {exp['error']}")
                return

            # Flatten the per-year routed prompts into the flat layout the
            # combined annotator now reads.
            fstats = _flatten_routed_input(routed_dir)
            _write_routed_lores_marker(
                routed_dir, bool((step4test_cfg or {}).get("lores_mode", True)))
            _route_set(message=(f"Flattened routed input: {fstats['merged']} frames, "
                                f"{fstats['year_dirs_removed']} year folder(s) collapsed."))

            # Free the GPU before the combined annotator (SAM3) opens - routing
            # and segmentation must never contend for the device.
            runner.kill("step4test_route")

            # Last gate before the launch. The operator closed the launch window,
            # so do NOT spring an annotator on them now - and above all do not run
            # _launch_step4test_ui, which kills every running annotator first. That
            # is exactly how an abandoned pass killed a live annotator out from
            # under the UI (2026-07-14). The routed prompts are written and cached,
            # so the next Open is instant.
            if route_cancel.is_set():
                _route_set(phase="cancelled", ui_port=None, error=None,
                           message="Routing finished, but the launch window was closed. "
                                   "Click Open to start the annotator.")
                return

            # This pass routed project_ctx's frames into project_ctx's
            # routed_input, but a route can outlive the project that started it
            # (the operator can open another project mid-route). _launch_step4test_ui
            # reads the GLOBAL current_project, so launching now would bind the
            # annotator to a DIFFERENT project's routed_input - one this pass never
            # wrote, so empty or half-written. Stop here and let the operator start
            # a route for the project they actually have open.
            cur_id = (current_project or {}).get("id")
            ctx_id = (project_ctx or {}).get("id")
            if cur_id != ctx_id:
                _route_set(phase="error",
                           error=(f"Project changed during routing ({ctx_id} -> "
                                  f"{cur_id}). The routed prompts for {ctx_id} are "
                                  f"written, but no annotator was launched. Reopen "
                                  f"that project, or click Open to route {cur_id}."))
                return

            _route_set(phase="route_ready",
                       message="Routing complete. Launching combined annotator...")

            # Open the combined window only after routing + placePoints kill.
            ui_result = _launch_step4test_ui()
            if isinstance(ui_result, dict) and ui_result.get("error"):
                _route_set(phase="error", error=ui_result["error"])
                return
        except Exception as e:
            _route_set(phase="error", error=str(e))

    def _annotator_reusable(routed_dir):
        """True if a combined annotator is already running, healthy, and bound to
        the CURRENT project + routed_dir, so we can open its existing window
        without a kill + relaunch.

        Reuse is only safe when the running process is the one launched for this
        project and this input dir: the annotator freezes its export_dir at first
        /api/configure and never rebinds, so a process launched for a different
        project would misroute every export. Any mismatch (or an unhealthy /
        vanished process) returns False, so the caller launches fresh."""
        if not step4test_state.get("running"):
            return False
        port = runner.ports.get("step4test") or step4test_state.get("port")
        if not port:
            return False
        pid = current_project.get("id") if current_project else None
        if step4test_state.get("project_id") != pid:
            return False
        if step4test_state.get("input_dir") != routed_dir:
            return False
        return runner.health_check(port)

    def _run_step4test(force_reroute=False):
        """Two-phase launch of the combined annotator (TCRMPclip_combinedAnnotate)
        as a full replacement for Step 4.

        Phase 1 (routing): drive placePoints headlessly over Step 3's chosen
        frames to produce sam_click_prompts.json in routed_input/, streaming
        progress via route_status. Phase 2: launch the combined annotator pointed
        at routed_input/ (done by _route_drive's tail, or directly on reuse).

        If routed_input/ already holds prompts and not force_reroute, skip
        routing and launch the UI instantly.
        """
        global route_thread

        # A routing pass already in flight OWNS the annotator launch: its tail
        # (_route_drive) kills every annotator and relaunches one when it
        # finishes. Starting a second launch path here races it - the older
        # thread would kill the annotator this call just launched, orphaning the
        # UI's wait on a port that never comes back up. Attach the UI to the
        # running pass instead, so it polls route_status and opens when that
        # pass lands. (Closing the launch popup abandons the frontend poll but
        # does NOT stop the thread, so an in-flight pass can outlive its click.)
        if route_thread is not None and route_thread.is_alive():
            with route_lock:
                phase = route_status.get("phase") or "routing_ocr"
            return {"phase": phase, "in_flight": True}

        routed_dir = _routed_dir()
        os.makedirs(routed_dir, exist_ok=True)

        # Resolve Step 3's selected frames (config or outputs).
        selected_frames = _resolve_path(3, "selected_frames", "selected_frames")
        if not selected_frames or not os.path.isfile(selected_frames):
            return {"error": f"Run {step_ref(3)} first: no selected_frames found."}

        # Reuse only a FRESH cache: if the Step 3 selection changed since the
        # last route, the cached prompts are stale and would leak the prior
        # run's frames into segmentation. Clear them and re-route. Otherwise
        # open instantly.
        want_lores = bool((step4test_cfg or {}).get("lores_mode", True))
        if not force_reroute and _routed_prompts_exist(routed_dir):
            cached_lores = _routed_lores_mode(routed_dir)
            lores_changed = cached_lores is not None and cached_lores != want_lores
            if _routed_prompts_stale(routed_dir, selected_frames) or lores_changed:
                cleared = _clear_routed_input(routed_dir)
                _reason = ("Low-res setting changed since last route; "
                           if lores_changed
                           else "Step 3 selection changed since last route; ")
                _route_set(phase="routing", processed=0, total=0, error=None,
                           message=(f"{_reason}cleared {cleared} stale route(s), "
                                    f"re-routing..."))
                # fall through to the routing path below (do NOT return)
            else:
                # Prompts are fresh. If a combined annotator is ALREADY running,
                # healthy, and bound to THIS project + routed_dir, reuse it as-is
                # instead of killing + relaunching. A relaunch reloads SAM3 (~1
                # min) and drifts the annotator's port, which pushes the browser's
                # window.open() past its user-activation window so the popup gets
                # silently blocked (the "Open does nothing" bug). Reuse keeps the
                # open instant, so the popup fires inside the click's activation.
                if _annotator_reusable(routed_dir):
                    reuse_port = runner.ports.get("step4test") or step4test_state.get("port")
                    _push_reviewer_after_health(reuse_port, _reviewer_initials(), runner.health_check)
                    _route_set(phase="ui_ready", ui_port=reuse_port,
                               runner_key="step4test",
                               runner_label="Combined annotator",
                               message="Combined annotator already running; reusing it.")
                    return {"phase": "ui_ready", "ui_port": reuse_port}
                _route_set(phase="route_ready", processed=0, total=0, error=None,
                           message="Reusing routed images. Launching combined annotator...")
                return _launch_step4test_ui()

        # Build the Step-3 routing env exactly like _run_step4, but exporting
        # into routed_dir (placePoints writes prompts under {routed_dir}/{year}/ids/).
        s4cfg = current_project["steps"].get("4", {}).get("config", {})
        all_points = _resolve_path(2, "all_points", "all_points_recoded")
        # Routing clip source: the 4.test panel's clip-dir wins, else the (now
        # archived) Step-4 saved clip_dir, else the LOCAL full-set clip dir.
        # A stale/missing saved path (e.g. an old Dropbox path) is skipped, so
        # routing always reads from the same supporting_data tree as Step 3.
        _clip_dir = _resolve_routing_clip_dir(step4test_cfg, s4cfg)
        routing_env = {
            "TCRMP_SELECTED_FRAMES": selected_frames,
            "TCRMP_EXPORT_DIR": routed_dir,
            "TCRMP_CLIP_DIR": _clip_dir,
            # Low-res delivery (step-4 panel "Low res only", default on): route
            # every >1920px frame from its lores twin with points scaled to match.
            "TCRMP_LORES_MODE": "1" if (step4test_cfg or {}).get("lores_mode", True) else "",
            "TCRMP_CPC_DIR": PATHS.get("cpc_all_dir", "") or os.path.join(REPO_DIR, "supporting_data", "cpc_all"),
            "TCRMP_TARGET_SPECIES": _get_target_species_str(),
            "TCRMP_REMAP_LOG": _get_remap_log_path(),
            "TCRMP_ALL_POINTS": all_points,
        }
        routing_env = {k: v for k, v in routing_env.items() if v}

        # Launch placePoints headless on a free port and drive it.
        py = PYTHON_PATHS.get(4, "python3")
        if not os.path.isfile(py):
            py = "python3"
        pp_entry = ENTRY_POINTS[4]
        pp_dir = WORKING_DIRS[4]
        pp_port = find_free_port(5066)
        _route_set(phase="launching", processed=0, total=0, ui_port=None, error=None,
                   warnings=[], dropped=[],
                   message="Launching routing sub-app (placePoints)...")
        cmd = [py, pp_entry, "--port", str(pp_port)]

        # No live driver thread got us here (the in-flight guard above returns
        # first), so any routing stage still holding this slot is STRANDED - a
        # placePoints left by a pass whose driver died, or one inherited across an
        # orchestrator restart. It will never finish or launch anything. Reclaim the
        # slot instead of letting run_flask_stage refuse with "already running",
        # which would fail every Open click until someone killed it by hand.
        runner.kill("step4test_route")

        result = runner.run_flask_stage("step4test_route", cmd, pp_port,
                                        cwd=pp_dir, env_extra=routing_env)
        if isinstance(result, dict) and result.get("error"):
            _route_set(phase="error", error=result["error"])
            return result

        # A fresh pass starts uncancelled (a previous pass may have left the flag
        # set when its window was closed).
        route_cancel.clear()

        species_list = [s.strip() for s in _get_target_species_str().split(",") if s.strip()]
        route_thread = threading.Thread(
            target=_route_drive,
            args=(pp_port, selected_frames, routed_dir, species_list, current_project),
            daemon=True,
        )
        route_thread.start()
        return {"phase": "launching"}

    def _kill_all_annotators():
        """Kill EVERY running PROJECT annotator-family app before launching a
        fresh one, so the new process binds to the CURRENT project's export
        dir. Spares any process whose environment carries
        TCRMP_SESSION_MODE=standalone (the "Annotate Images" launch) - see
        the comment above annotate_state for the two-way guarantee this
        keeps with _launch_standalone_annotator.

        The annotator is one long-lived process whose session['export_dir'] is
        frozen at its first /api/configure and never rebinds. If a stale one
        keeps running, every export lands in ITS original project instead of the
        active one. Covers (a) the stages this orchestrator tracks (step4test,
        step4loop share the combinedAnnotate entry; editmasks is a separate
        entry script) AND (b) zombies from a prior orchestrator session (not in
        runner.processes) by matching both entry scripts (via pgrep, one pid at
        a time, so a standalone pid can be skipped) and freeing the shared
        annotator port range. No lsof on this box; pgrep and ss only.

        2026-09-27: every kill below is preceded by a flush (_flush_annotator,
        POST /api/flush) of that same port, so an operator's unsaved review
        edit reaches disk before the process is cut off. A failed flush (no
        listener, timeout, non-2xx) is logged to vic.note and the kill
        proceeds regardless, since a process that will not answer must never
        block a stop. Each port is flushed at most once per call
        (flushed_ports), since the pgrep pass and the port sweep below can
        both name the same live pid."""
        import subprocess
        import signal
        flushed_ports = set()

        def _flush_before_kill(port):
            """Flush `port` (POST /api/flush) unless it is falsy or was
            already flushed earlier in this call, then remember it either
            way so a later call in the same sweep never re-flushes it. A
            failed flush is logged with its port and reason; the caller
            still proceeds to kill regardless."""
            if not port or port in flushed_ports:
                return
            flushed_ports.add(port)
            ok, reason = _flush_annotator(port)
            if not ok:
                vic.note(f"[orch] annotator flush failed on port {port} before kill: {reason}")

        _flush_before_kill(_live_annotator_port("step4test"))
        try:
            runner.kill("step4test")
        except Exception:
            pass
        _flush_before_kill(_live_annotator_port("editmasks"))
        try:
            runner.kill("editmasks")
        except Exception:
            pass
        entry_matches = (
            os.path.join("TCRMPclip_combinedAnnotate", "src", "app.py"),
            os.path.join("TCRMPclip_editMasks", "src", "app.py"),
        )
        # ss is fetched once, up front, so both the pgrep pass right below (a
        # pid -> port reverse lookup, for a zombie process runner.ports never
        # tracked) and the port sweep after it read the SAME snapshot.
        try:
            _ss = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=10)
            _ss_out = getattr(_ss, "stdout", "") or ""
        except Exception:
            _ss_out = ""
        for entry_match in entry_matches:
            for pid in _annotator_pids(entry_match):
                environ = _proc_environ(pid)
                if b"TCRMP_SESSION_MODE=standalone" in environ:
                    continue  # standalone session - never killed by a project launch
                _flush_before_kill(_listener_port_for_pid(_ss_out, pid))
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass
        # Port half. The 5080-5089 range is SHARED with the rest of the
        # platform: vicariusLLM autostarts on 5085. Edit Masks now prefers 5083
        # (T20260826-223636 resolved on the reef side), so it no longer
        # collides. The old blind `fuser -k` loop over the whole range
        # therefore killed vicariusLLM on 2026-08-26 21:49. Each listener is now
        # identified first: only a pid whose /proc/<pid>/cmdline names one of
        # the two annotator entry scripts is signalled, and an unreadable
        # cmdline fails closed (the process is left running). ANNOTATE_PORT
        # (5096) is deliberately outside 5080-5089, so this sweep can never hit
        # the standalone annotator's port either.
        for _p in range(5080, 5090):
            for pid in _port_listener_pids(_ss_out, _p):
                cmdline = _proc_cmdline(pid)
                if not any(entry in cmdline for entry in entry_matches):
                    continue  # not an annotator (vicariusLLM on 5085, say), or unreadable
                if b"TCRMP_SESSION_MODE=standalone" in _proc_environ(pid):
                    continue  # standalone session - never killed by a project launch
                _flush_before_kill(_p)
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass
        time.sleep(0.5)  # let the killed process release its port

    def _resolve_master_codes():
        """Recoded master codes CSV path for the active project - same
        resolution used by both annotator launch paths (step4test, step4loop)
        and the step4loop seeder call: step 2's recoded output if present on
        disk, else the plain master_codes.csv, else the on-disk fallback
        filenames under the step-2 dir. Returns '' when nothing resolves."""
        _s2 = current_project["steps"].get("2", {})
        _mc = (_s2.get("outputs", {}).get("master_codes_recoded")
               or _s2.get("outputs", {}).get("master_codes"))
        if not (_mc and os.path.isfile(_mc)):
            _s2dir = _s2.get("dir", "")
            for _cand in ("master_codes_recoded.csv", "master_codes.csv"):
                _p = os.path.join(_s2dir, _cand) if _s2dir else ""
                if _p and os.path.isfile(_p):
                    _mc = _p
                    break
        return _mc if (_mc and os.path.isfile(_mc)) else ""

    def _selected_frame_stems():
        """Every basename Step 3 selected for the open project, read from
        its own selected_frames.csv (the 'basename' column select_images.py
        writes). Empty list when Step 3 has not produced one yet or the file
        cannot be read; never raises, so a missing selection never blocks an
        annotator launch, it just means prior_screening.json covers zero
        frames this time."""
        path = _resolve_path(3, "selected_frames", "selected_frames")
        if not path or not os.path.isfile(path):
            return []
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                return [row["basename"] for row in csv.DictReader(fh) if row.get("basename")]
        except (OSError, csv.Error, KeyError) as e:
            vic.note(f"[orch] failed to read {path} for prior_screening.json: {e}")
            return []

    def _write_prior_screening_for_launch():
        """Refresh <step4test export dir>/prior_screening.json before an
        annotator launch (2026-09-27 VEGA annotator tidy-up plan): which of
        this project's selected frames are already screened, in another
        project, for some but not all of its target labels, and every
        accepted mask another project holds for those labels.

        Never raises: a matrix read problem, a missing selected_frames.csv,
        or an unwritable export dir is logged via vic.note and the launch
        proceeds regardless, with whatever prior_screening.json (possibly
        none, or a now-stale one from an earlier launch) already sits on
        disk. Called for every mode (step4test, step4loop, editmasks) from
        the one shared launch path below."""
        if current_project is None:
            return
        try:
            _screening.write_prior_screening(
                current_project["project_dir"],
                _selected_frame_stems(),
                _inprocess_root(),
                master_codes_csv=_resolve_master_codes() or PATHS.get("master_codes_csv", ""),
            )
        except Exception as e:
            vic.note(f"[orch] failed to write prior_screening.json: {e}")

    def _launch_step4test_ui(mode="step4test", cfg=None):
        """Launch the combined annotator (TCRMPclip_combinedAnnotate) pointed at
        routed_input/ (the routed-prompts dir, NOT step 4's output dir).

        Inherits step 5's SAM3 settings + recoded master codes; stays in
        MANUAL_ANNOTATE mode with provenance source step4test. The routed points
        become the read-only reference layer in the UI.

        mode="step4loop" (the Refine active-learning loop) shares every path
        below EXCEPT: the cfg source (step4loop_cfg instead of step4test_cfg),
        the runner key ("step4loop"), the state dict (step4loop_state), and two
        env overrides that put the annotator into resume mode against the
        segmentations the seeder already wrote (TCRMP_SESSION_MODE=resume,
        TCRMP_PROVENANCE_SOURCE=step4loop) instead of the fresh-configure
        MANUAL_ANNOTATE flow step4test uses.

        mode="editmasks" launches a DIFFERENT app entirely: the standalone
        TCRMPclip_editMasks tool (not combinedAnnotate), against the current
        project's step4test export. Like step4loop it boots straight into the
        review queue via /api/resume (TCRMP_SESSION_MODE=edit,
        TCRMP_PROVENANCE_SOURCE=edit) rather than fresh-configure, but there is
        no seeding pass first: it opens on whatever masks Step 4 already wrote.
        Runner key "editmasks", state dict editmasks_state, own preferred port
        5083 (editMasks' own default PORT in config.py; moved off 5085 to stop
        colliding with vicariusLLM's autostart port, TODO T20260826-223636).
        """
        runner_key = {"step4loop": "step4loop", "editmasks": "editmasks"}.get(mode, "step4test")
        state_dict = {"step4loop": step4loop_state,
                      "editmasks": editmasks_state}.get(mode, step4test_state)

        # One annotator at a time: kill any already-running annotator (any of
        # combinedAnnotate/editMasks) so this launch binds to the current
        # project's export dir (not a stale one).
        _kill_all_annotators()

        # Hardcoded entry point + cwd (config dicts are int-keyed; this tile has
        # no integer key, so we build paths from REPO_DIR directly). editmasks
        # runs a different app (TCRMPclip_editMasks) than step4test/step4loop
        # (TCRMPclip_combinedAnnotate).
        if mode == "editmasks":
            entry = os.path.join(
                REPO_DIR, "scripts", "TCRMPclip_editMasks", "src", "app.py")
            cwd = os.path.join(REPO_DIR, "scripts", "TCRMPclip_editMasks")
            # TODO T20260826-223636: was 5085, which collides with
            # vicariusLLM's autostart port; vicariusLLM keeps 5085.
            preferred_port = 5083
            app_label = "edit-masks app"
        else:
            entry = os.path.join(
                REPO_DIR, "scripts", "TCRMPclip_combinedAnnotate", "src", "app.py")
            cwd = os.path.join(REPO_DIR, "scripts", "TCRMPclip_combinedAnnotate")
            preferred_port = 5080
            app_label = "combined annotator"
        py = PYTHON_PATHS.get(5, "python3")  # same unified env as step 5
        if not os.path.isfile(py):
            py = "python3"
        if not os.path.isfile(entry):
            return {"error": f"{app_label} not found at {entry}"}

        port = find_free_port(preferred_port)

        # Input = the routed-prompts dir produced by the routing pass.
        input_dir = _routed_dir()

        # Output dir = sibling of the step-5 dir, under the project dir so
        # /api/fs/open's guardrails allow opening it.
        project_dir = current_project["project_dir"]
        out_dir = _step4test_export_dir()
        os.makedirs(out_dir, exist_ok=True)

        # Inherit step 5's SAM3 settings verbatim, but let the step4test panel
        # override per-launch. `step4test_cfg` carries the #panel-step4test inputs
        # (collected client-side and POSTed with /run|/reroute); each key falls
        # back to the inherited Step-5 saved config so a 4.test-only flow still
        # works even if Step 5 was never opened. mode="step4loop" reads from
        # step4loop_cfg instead (or an explicit `cfg` override, if given).
        # mode="editmasks" has no panel config of its own (Edit Masks is a
        # straight launch, no settings to collect); it falls back to the
        # inherited Step-5 config like a bare cfg={} would.
        s5cfg = current_project["steps"].get("5", {}).get("config", {})
        default_cfg = step4loop_cfg if mode == "step4loop" else step4test_cfg
        t4 = cfg if cfg is not None else (default_cfg or {})

        def _pick(key, default=""):
            """step4test panel value, else inherited Step-5 value, else default."""
            v = t4.get(key)
            if v is None or v == "":
                v = s5cfg.get(key)
            return v if (v is not None and v != "") else default

        # SETTINGS ENV CONTRACT (config.py reads these; the orchestrated
        # /api/configure is called with an empty body, so env defaults must apply).
        # Target-only + read-only-reference default ON ("1"); batch size default 10.
        target_only = "1" if t4.get("target_species_only", True) else "0"
        reference_default = "1" if t4.get("reference_default", True) else "0"
        raw_batch = _pick("review_batch_size", "10")
        batch_size = str(raw_batch).strip() or "10"

        # Expert-review export targets for a 4.test-only flow (Task 5): inherit
        # Step-5 config, else fall back to PATHS + the canonical repo url so REVIEW
        # export works even when Step 5 was never opened.
        review_dir = _pick("review_dir") or PATHS.get("review_dir", "")
        review_repo = (t4.get("review_repo_url") or s5cfg.get("review_repo_url")
                       or PATHS.get("review_repo_url", ""))
        expert_lib = (t4.get("expert_library_dir") or s5cfg.get("expert_library_dir")
                      or PATHS.get("expert_library_dir", ""))
        review_contacts = _pick("review_contacts")

        env_extra = {
            "TCRMP_INPUT_DIR": input_dir,
            "TCRMP_EXPORT_DIR": out_dir,
            "TCRMP_PROJECT_ID": current_project.get("id", "") if current_project else "",
            "TCRMP_PROJECT_NAME": current_project.get("name", "") if current_project else "",
            "TCRMP_TARGET_SPECIES": _get_target_species_str(),
            # Combined Step-4+5 settings (panel-overridable, Step-5-inherited).
            "TCRMP_TARGET_SPECIES_ONLY": target_only,
            "TCRMP_REFERENCE_DEFAULT": reference_default,
            "TCRMP_REVIEW_BATCH_SIZE": batch_size,
            "TCRMP_SAM3_DEVICE_TRACKER": _pick("sam3_device_tracker"),
            "TCRMP_SAM3_DEVICE_EXEMPLAR": _pick("sam3_device_exemplar"),
            "TCRMP_CONFIDENCE_THRESHOLD": str(_pick("confidence_threshold")),
            "TCRMP_MIN_MASK_AREA_PX": str(_pick("min_mask_area_px")),
            "TCRMP_MERGE_DISTANCE_PX": str(_pick("merge_distance_px")),
            "TCRMP_OVERLAP_STRATEGY": _pick("overlap_strategy"),
            # NEW env overrides plumbed through to combinedAnnotate config.py.
            "TCRMP_THIN_MASK_RATIO": str(_pick("thin_mask_ratio")),
            "TCRMP_POLYGON_SIMPLIFY_EPSILON": str(_pick("polygon_simplify_epsilon")),
            # Symlink-vs-copy of exported images (panel control; always sent so an
            # explicit "0" survives the empty-value sweep below).
            "TCRMP_SYMLINK_IMAGES": "1" if t4.get("symlink_images", True) else "0",
            # Expert-review export targets (REVIEW-flagged masks).
            "TCRMP_REVIEW_DIR": review_dir,
            "TCRMP_REVIEW_REPO_URL": review_repo,
            "TCRMP_EXPERT_LIBRARY_DIR": expert_lib,
            "TCRMP_REVIEW_CONTACTS": review_contacts,
            # Who is annotating (spec D2): latest access_log initials of the
            # project, else its creator. The annotator stamps it on every
            # mask it touches and on the registry row.
            "TCRMP_REVIEWER": _reviewer_initials(),
        }
        # Target-only + reference default are an explicit "1"/"0" contract; keep
        # them even when "0". The rest drop out when blank so config.py defaults win.
        _always = {"TCRMP_TARGET_SPECIES_ONLY", "TCRMP_REFERENCE_DEFAULT",
                   "TCRMP_SYMLINK_IMAGES"}
        env_extra = {k: v for k, v in env_extra.items() if v or k in _always}

        # Combined-annotator mode + provenance source. These are NOT filtered
        # by the empty-value sweep above (they are always set).
        if mode == "step4loop":
            # Refine loop: the seeder already wrote segmentations.json (Task 3),
            # so the annotator must boot in resume mode (/api/resume, which reads
            # only segmentations.json) rather than fresh-configure MANUAL_ANNOTATE.
            env_extra["TCRMP_SESSION_MODE"] = "resume"
            env_extra["TCRMP_PROVENANCE_SOURCE"] = "step4loop"
        elif mode == "editmasks":
            # Edit Masks: no seeding pass, just open on the export's existing
            # segmentations.json. editMasks' own config.py already defaults
            # SESSION_MODE/PROVENANCE_SOURCE to "edit"; set them explicitly here
            # so the orchestrated launch never depends on that default drifting.
            env_extra["TCRMP_SESSION_MODE"] = "edit"
            env_extra["TCRMP_PROVENANCE_SOURCE"] = "edit"
        else:
            env_extra["TCRMP_MANUAL_ANNOTATE"] = "1"
            env_extra["TCRMP_PROVENANCE_SOURCE"] = "step4test"

        # Recoded master codes - same resolution as step 5.
        _mc = _resolve_master_codes()
        if _mc:
            env_extra["TCRMP_MASTER_CODES"] = _mc

        # No-re-screening ledger (2026-09-27 VEGA annotator tidy-up plan):
        # refreshed before every annotator launch, configure or resume alike,
        # so a partly-screened frame's locked previews are never more than
        # one launch stale. A write failure is logged inside the helper and
        # never blocks the launch (see its docstring).
        _write_prior_screening_for_launch()

        cmd = [py, entry, "--port", str(port)]
        result = runner.run_flask_stage(runner_key, cmd, port, cwd=cwd, env_extra=env_extra)
        if isinstance(result, dict) and result.get("error"):
            return result
        # The env carries TCRMP_REVIEWER for a fresh process; the POST below
        # is what updates a process that outlives a project switch.
        _push_reviewer_after_health(port, _reviewer_initials(), runner.health_check)
        state_dict.update(running=True, port=port,
                          started_at=datetime.now().isoformat(), out_dir=out_dir,
                          project_id=(current_project.get("id") if current_project else None),
                          input_dir=input_dir)
        ready_message = "Edit Masks ready." if mode == "editmasks" else "Combined annotator ready."
        # Stamp which process owns this ui_ready: route_status is shared by all
        # three launch modes and the liveness check must poll the right one.
        owner_label = {"step4loop": "Refine annotator",
                       "editmasks": "Edit Masks app"}.get(mode, "Combined annotator")
        _route_set(phase="ui_ready", ui_port=port,
                   runner_key=runner_key, runner_label=owner_label,
                   message=ready_message)
        return result

    def _run_step6(step, cfg, step_dir):
        """Launch training for one Start Training press at a time.

        Holds the module-level launch lock around _launch_step6 so the busy
        probe at its top and the launch at its end cannot interleave with a
        second press.

        Parameters:
            step: the integer step key (6).
            cfg: the saved Step 6 config.
            step_dir: the step's output directory.
        Returns:
            the runner's launch result, or an error dict from _launch_step6.
        """
        with _step6_launch_lock:
            return _launch_step6(step, cfg, step_dir)

    def _launch_step6(step, cfg, step_dir):
        """Run dataset split + YOLO training via oceankind_CV.

        Parameters:
            step: the integer step key (6).
            cfg: the saved Step 6 config.
            step_dir: the step's output directory.
        Returns:
            the runner's launch result ({"pid": ...} plus task, split and
            class details), or an error dict carrying an "error" message,
            an optional "http_status", and "busy": True when refused
            because a trainer is still running.

        step_dir will receive:
            dataset/   train/valid/test split + data.yaml + test.yaml
            runs/      ultralytics training runs, one subdir per --name

        Two shapes of training source reach here:

          a YOLO-seg export     this project's Step 4 export, or a composite
                                export from the coverage matrix. The engine is
                                driven in "all" mode: split, then train.
          a staged box dataset  a bbox training project. There is nothing for
                                pinned_split.py to read (it wants all_images/
                                and pins the holdout by TCRMP transect), so
                                the split is built here from the dataset where
                                it is staged and the engine is driven in
                                "train" mode over the finished dataset. Same
                                invocation the model lab uses, so a dataset
                                trained in either place splits the same way.

        Task: segmentation unless the panel asked for a detector. Every other
        knob is task-agnostic and is forwarded identically for both.

        Refuses with 409 before anything is written when this step's process
        is still alive: the dataset build below removes and replaces
        <step_dir>/dataset, which a live trainer is reading.
        """
        # Busy probe first, before any filesystem write. The runner's own
        # check at the end of this function comes after the dataset build,
        # which is too late for a box dataset. The recorded status is not
        # consulted: after an orchestrator restart project.json can still say
        # running while nothing is. The kernel's process list is, because a
        # driver spawned in its own session outlives that restart and the
        # runner no longer knows it; its pid rides along so the refusal can
        # name the process to stop. The route names the tile and answers 409.
        if runner.is_running(step):
            return busy_result(step)
        orphan_pid = _step6_driver_alive_outside_runner(step_dir)
        if orphan_pid:
            return {**busy_result(step), "orphan_pid": orphan_pid}

        # What this run trains, normalized once. An unknown spelling is refused
        # here rather than at the engine, which would already have written a
        # dataset by the time its own parser rejected it.
        raw_task = cfg.get("task")
        if raw_task in (None, ""):
            task = _project_default_task(current_project)
        else:
            task = _normalize_task(raw_task)
            if not task:
                return {"error": (f"Unknown training task {raw_task!r}. Use segment "
                                  "(the default) or detect."),
                        "http_status": 400}

        # A bbox training project reads its dataset where it is staged; nothing
        # is copied or symlinked into the run dir. An ordinary project may also
        # pick a staged box dataset as its training source (the Step 5 source
        # list offers them beside the composites) and is handled the same way.
        bbox_source = None
        source_dir = str(cfg.get("source_dir") or "").strip()
        if (current_project or {}).get("project_type") == BBOX_PROJECT_TYPE:
            bbox_source = dict((current_project.get("training_source") or {}))
        elif source_dir and _under_bbox_root(source_dir):
            bbox_path, bbox_err = _validate_bbox_source(source_dir)
            if bbox_err:
                return {"error": bbox_err, "http_status": HTTP_BAD_REQUEST}
            # Only a box training project may train from a staged box
            # dataset. In any other project the build below replaces
            # <step6>/dataset whole, which is the split this project's own
            # outline runs were trained and scored on: those runs lose the
            # dataset Evaluate reads for them, and the ledger beside them
            # starts ranking box scores against mask scores. A box dataset
            # gets its own project, which the Bounding Boxes page mints.
            return {"error": (
                f"This project trains on outlines, so it cannot train from the box dataset "
                f"{os.path.basename(os.path.normpath(bbox_path))}: building that split would "
                f"replace the dataset this project's own runs were scored on. Open the "
                f"Bounding Boxes page and create a training project for that dataset instead."),
                "http_status": HTTP_CONFLICT}
        if bbox_source is not None:
            # A box dataset holds no outlines, so a blank task follows the
            # source (detect) and an explicit segmentation ask is refused.
            if raw_task in (None, ""):
                task = "detect"
            if task != "detect":
                return {"error": (
                    "This project trains from a box dataset, which holds boxes and no "
                    "outlines to learn from, so it can only train a detector. Set "
                    f"Training task back to Detector in the {step_ref(6)} advanced "
                    "settings."), "http_status": HTTP_BAD_REQUEST}

        # A training project was minted to train from one source, and its
        # ledger compares rounds on that source. Training it again from the
        # SAME source adds a round; pointing it at another source needs a new
        # project, or the ledger would rank incomparable rounds together.
        # Same guard, same wording, as the retired model lab's lab.json check.
        recorded_source = _recorded_training_source(current_project)
        if recorded_source and source_dir and os.path.realpath(source_dir) != recorded_source:
            return {"error": (f"project {current_project.get('name', '')} already trains from a "
                              f"different source ({recorded_source}). Pick that source again, "
                              "or create a new training project for the other one."),
                    "http_status": HTTP_CONFLICT}

        src_dir = ""
        if bbox_source is None:
            # Input: the 4.test combined-annotator export dir (all_images/, all_labels/,
            # data.yaml). 4.test fully replaces Steps 4+5, so train always reads there.
            src_dir = _step4test_export_dir()
            # Training source (spec D5, 2026-08-26): a composite export assembled
            # across projects on the coverage matrix may replace this project's
            # own Step 4 export. Only a dir under <inprocess>/_composite/ holding
            # all_images/ + data.yaml is accepted; anything else answers 400.
            if source_dir:
                src_dir, src_err = _validate_outline_source(source_dir)
                if src_err:
                    return {"error": src_err, "http_status": 400}
            if not src_dir or not os.path.isdir(src_dir):
                return {"error": f"{step_ref('step4test')} output not found at {src_dir}. Run it first."}
            # Quick sanity: a meaningful training payload needs the exported images.
            if not os.path.isdir(os.path.join(src_dir, "all_images")):
                return {"error": (
                    f"{step_ref('step4test')} wrote no all_images/ into {src_dir}. "
                    "Export at least one batch from the Place + Segment UI to generate training data."
                )}

        # Resolve a training run name.
        run_name = (cfg.get("run_name") or "").strip()
        if not run_name:
            run_name = f"{current_project['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            current_project["steps"]["6"]["config"]["run_name"] = run_name

        # Base weights: an explicit pick wins, an empty one follows the task
        # (yolo11m-seg.pt for segment, yolo11m.pt for detect), same rule the
        # engine's own default uses.
        model = cfg.get("model") or default_model_for_task(task)
        epochs = str(cfg.get("epochs", 500))
        imgsz = str(cfg.get("imgsz", 512))

        # Fine-tune source: an explicit absolute weights path (e.g. a previous
        # run's best.pt, or the promoted champion) overrides the catalog model
        # name above. Preflight so a stale/typo'd path fails here with a clear
        # message instead of surfacing as an opaque ultralytics traceback.
        model_path = (cfg.get("model_path") or "").strip()
        if model_path:
            if not os.path.isfile(model_path):
                return {"error": f"Base weights not found: {model_path}"}
            model = model_path

        # Weights and task must agree. Ultralytics takes the head from the
        # checkpoint and ignores the task it is handed, so a detector asked to
        # start from segmentation weights would quietly train a segmentation
        # model into this run's folder. Refuse here, before anything is split
        # or written, with the engine's own wording.
        mismatch = check_model_task(model, task)
        if mismatch:
            return {"error": mismatch, "http_status": 400}

        # Pre-flight: ultralytics refuses batch<1 (auto) under DDP. Catch it
        # here so the user sees a helpful error instead of the subprocess
        # traceback.
        device_preview = (cfg.get("device") or "").strip()
        batch_preview = cfg.get("batch", -1)
        try:
            batch_int = int(batch_preview)
        except (TypeError, ValueError):
            batch_int = -1
        if "," in device_preview and batch_int < 1:
            gpu_count = len([d for d in device_preview.split(",") if d.strip()])
            suggested = max(2, 8 * gpu_count)
            return {"error": (
                f"Multi-GPU training (device='{device_preview}') needs an explicit "
                f"batch size: ultralytics does not support auto-batch (batch=-1) "
                f"under DDP. Set Batch to a multiple of {gpu_count} in the "
                f"{step_ref(6)} panel (e.g. {suggested}) and re-run."
            )}

        # Invoke our bash driver. A YOLO-seg export goes through "all" mode
        # (split, then train). A staged box dataset has its split built here
        # first, because pinned_split.py cannot read images/ + labels/ and has
        # no TCRMP transect to pin a holdout by, and then goes through "train"
        # mode over the finished dataset. The split is seeded on the staged
        # dataset's folder name, so the same dataset splits identically wherever
        # it is trained (Lauren's ruling of 2026-09-04; the retired Train +
        # Infer page seeded on the lab name, which is why older lab runs read as
        # not comparable).
        split_summary = None
        if bbox_source is not None:
            raw_source_path = str(bbox_source.get("path") or "")
            source_path = os.path.normpath(raw_source_path) if raw_source_path else ""
            # The split is seeded on the staged dataset's folder name, the
            # same string the Bounding Boxes quick trainer seeds on, so the
            # same dataset splits the same images the same way wherever it is
            # trained. The path is normalised first: a trailing slash once
            # gave an empty seed. Existing pins in split_manifest.json are
            # replayed, so retraining never moves an image between splits.
            source_name = os.path.basename(source_path)
            if not source_path or not os.path.isdir(source_path):
                return {"error": (
                    f"The box dataset {source_name or bbox_source.get('name') or ''} is no longer "
                    f"at {raw_source_path}. Stage it again from the Bounding Boxes page, then re-run."),
                    "http_status": HTTP_BAD_REQUEST}
            if not source_name:
                return {"error": (f"The box dataset path {raw_source_path} has no folder name to "
                                  "seed the split on. Stage the dataset under its own folder."),
                        "http_status": HTTP_BAD_REQUEST}
            valid_frac, frac_err = _step6_fraction(cfg, "valid_ratio", "Validation fraction")
            if frac_err:
                return {"error": frac_err, "http_status": HTTP_BAD_REQUEST}
            test_frac, frac_err = _step6_fraction(cfg, "test_ratio", "Test fraction")
            if frac_err:
                return {"error": frac_err, "http_status": HTTP_BAD_REQUEST}
            dataset_dir = os.path.join(step_dir, "dataset")
            try:
                split_summary = _build_detect_dataset(source_path, dataset_dir, source_name,
                                                      valid_frac=valid_frac, test_frac=test_frac)
            except ValueError as e:
                return {"error": str(e), "http_status": HTTP_BAD_REQUEST}
            except Exception as e:
                vic.note(f"[orch] build_detect_dataset failed for {source_path}: {e}")
                return {"error": f"Could not build the training split: {e}",
                        "http_status": 500}
            cmd = [
                ENTRY_POINTS[6], "train",
                step_dir, run_name,
                epochs, imgsz, model,
            ]
        else:
            cmd = [
                ENTRY_POINTS[6], "all",
                src_dir, step_dir, run_name,
                epochs, imgsz, model,
            ]

        # Forward everything the user might have tweaked via env vars. The
        # shell driver only appends flags when the corresponding var is set,
        # so missing keys here just mean "use wrapper defaults".
        def _e(key, cfg_key, default=None):
            v = cfg.get(cfg_key, default)
            return ("", str(v)) if v is None else (key, str(v))

        # Pinned-split (frozen by-transect holdout) knobs. Unlike the rest of
        # _e()'s list below (omitted when unset, so run_step6.sh falls back to
        # its own default), these four are ALWAYS forwarded with an explicit
        # default so the Refine loop's seeder (which requires split_manifest.json
        # to exist) and pinned_split.py agree on the same defaults even when the
        # Step 6 panel never touched them.
        env_pin_split = {
            "TCRMP_STEP6_PIN_SPLIT": str(cfg.get("pin_split", "1")),
            "TCRMP_STEP6_VAL_TRANSECTS": str(cfg.get("val_transects", "5")),
            "TCRMP_STEP6_TEST_TRANSECTS": str(cfg.get("test_transects", "6")),
            "TCRMP_STEP6_HOLDOUT_MODE": str(cfg.get("holdout_mode", "transect")),
        }

        env_extra_raw = dict(_e(k, ck) for k, ck in [
            ("TCRMP_STEP6_VALID", "valid_ratio"),
            ("TCRMP_STEP6_TEST", "test_ratio"),
            ("TCRMP_STEP6_MIN_SAMPLES", "min_samples"),
            # Training plumbing
            ("TCRMP_STEP6_BATCH", "batch"),
            ("TCRMP_STEP6_PATIENCE", "patience"),
            ("TCRMP_STEP6_OPTIMIZER", "optimizer"),
            ("TCRMP_STEP6_SEED", "seed"),
            ("TCRMP_STEP6_COS_LR", "cos_lr"),
            ("TCRMP_STEP6_CLOSE_MOSAIC", "close_mosaic"),
            # LR
            ("TCRMP_STEP6_LR0", "lr0"),
            ("TCRMP_STEP6_LRF", "lrf"),
            ("TCRMP_STEP6_MOMENTUM", "momentum"),
            ("TCRMP_STEP6_WEIGHT_DECAY", "weight_decay"),
            ("TCRMP_STEP6_WARMUP_EPOCHS", "warmup_epochs"),
            # Loss
            ("TCRMP_STEP6_BOX", "box"),
            ("TCRMP_STEP6_CLS", "cls"),
            ("TCRMP_STEP6_DFL", "dfl"),
            ("TCRMP_STEP6_LABEL_SMOOTHING", "label_smoothing"),
            # Color
            ("TCRMP_STEP6_HSV_H", "hsv_h"),
            ("TCRMP_STEP6_HSV_S", "hsv_s"),
            ("TCRMP_STEP6_HSV_V", "hsv_v"),
            ("TCRMP_STEP6_BGR", "bgr"),
            # Geometric
            ("TCRMP_STEP6_DEGREES", "degrees"),
            ("TCRMP_STEP6_TRANSLATE", "translate"),
            ("TCRMP_STEP6_SCALE", "scale"),
            ("TCRMP_STEP6_SHEAR", "shear"),
            ("TCRMP_STEP6_PERSPECTIVE", "perspective"),
            ("TCRMP_STEP6_FLIPUD", "flipud"),
            ("TCRMP_STEP6_FLIPLR", "fliplr"),
            # Mixing
            ("TCRMP_STEP6_MOSAIC", "mosaic"),
            ("TCRMP_STEP6_MIXUP", "mixup"),
            ("TCRMP_STEP6_CUTMIX", "cutmix"),
            ("TCRMP_STEP6_COPY_PASTE", "copy_paste"),
            ("TCRMP_STEP6_COPY_PASTE_MODE", "copy_paste_mode"),
            ("TCRMP_STEP6_ERASING", "erasing"),
            ("TCRMP_STEP6_AUTO_AUGMENT", "auto_augment"),
        ])
        env_extra = {k: v for k, v in env_extra_raw.items() if k and v not in ("", "None")}
        env_extra.update(env_pin_split)
        # What to train. Always forwarded so a run's env says what it was,
        # rather than leaning on the engine's own default. Every knob above and
        # below is task-agnostic and reaches the engine identically for both.
        env_extra["TCRMP_STEP6_TASK"] = task

        # Also forward the fine-tune freeze depth, when set: TCRMP_STEP6_FREEZE
        # (int; number of leading layers to freeze). Unset means no freeze
        # (train_wrapper.py's --freeze default is None, i.e. train everything).
        freeze = cfg.get("freeze")
        if freeze not in (None, ""):
            env_extra["TCRMP_STEP6_FREEZE"] = str(freeze)

        # Class-inclusion filter: if the user picked a subset of classes in the
        # orchestrator, forward them as a CSV env var. Empty / None / "all" means
        # "train on every class present in step 5".
        #
        # A staged box dataset has no class picker (the panel's picker reads a
        # YOLO-seg export's data.yaml + all_labels/, which it has neither of):
        # its classes come from its own classes.txt and every one of them
        # trains. Anything a stale saved config still carries is dropped here
        # rather than forwarded against the wrong id space.
        include_classes = None if bbox_source is not None else cfg.get("include_classes")

        # H2 (2026-08-26): the picker's ids belong to the id space of the
        # source it was loaded from (cfg.include_classes_source; a legacy
        # config with no such key means this project's Step 4 export). When
        # that differs from the training source, carry the selection over by
        # NAME so a composite whose ids follow sorted label order keeps the
        # species the user ticked. Names the training source lacks are
        # dropped and reported in the response.
        dropped_classes = []
        if isinstance(include_classes, (list, tuple)) and len(include_classes) > 0:
            picker_src = str(cfg.get("include_classes_source") or "").strip()
            picker_dir, picker_err = _class_picker_source(picker_src)
            if picker_err:
                return {"error": ("The class picker was loaded from a training source that is "
                                  f"no longer valid ({picker_err}). Reopen {step_ref(6)}, pick the "
                                  "training source again and re-tick the classes."),
                        "http_status": 400}
            if os.path.realpath(picker_dir) != os.path.realpath(src_dir):
                include_classes, dropped_classes = _translate_class_ids(
                    include_classes, _step6_picker_names(picker_dir, picker_src),
                    _data_yaml_id_names(os.path.join(src_dir, "data.yaml")))
                if not include_classes:
                    return {"error": ("None of the selected classes exist in the training source "
                                      f"{src_dir} (missing: {', '.join(dropped_classes) or 'none named'}). "
                                      "Re-tick the classes against that source."),
                            "http_status": 400}

        # Guard: strip any unnamed classes from the training source's data.yaml so the
        # training payload never carries them, even if the user manually
        # re-added one via the UI or from a stale saved project. Matches
        # _step6_is_unnamed_class (empty / "(unnamed)" case-insensitive).
        try:
            import yaml as _y
            _src_yaml = os.path.join(src_dir, "data.yaml") if src_dir else ""
            if _src_yaml and os.path.isfile(_src_yaml):
                with open(_src_yaml) as f:
                    _d = _y.safe_load(f) or {}
                _raw_names = _d.get("names") or {}
                if isinstance(_raw_names, dict):
                    _names = {int(k): v for k, v in _raw_names.items()}
                elif isinstance(_raw_names, list):
                    _names = {i: v for i, v in enumerate(_raw_names)}
                else:
                    _names = {}
                _unnamed_ids = {cid for cid, nm in _names.items()
                                if nm is None or not str(nm).strip() or
                                re.match(r"^\(unnamed\)$", str(nm).strip(),
                                         flags=re.IGNORECASE)}
                if _unnamed_ids:
                    if isinstance(include_classes, (list, tuple)):
                        include_classes = [int(x) for x in include_classes
                                           if int(x) not in _unnamed_ids]
                    else:
                        # "All classes present in the Step 4 (4.test) export"
                        # -> narrow to the explicit named subset so unnamed
                        # classes are stripped from the split.
                        include_classes = [cid for cid in sorted(_names.keys())
                                           if cid not in _unnamed_ids]
                    if not include_classes:
                        return {"error": (
                            "All classes resolved to unnamed / empty labels. "
                            f"Re-annotate in {step_ref('step4test')} with proper class names in "
                            f"{_src_yaml}."
                        )}
        except Exception as _exc:
            # Non-fatal: fall back to what the UI sent if the 4.test data.yaml is
            # unreadable. (Distinct name from the _e() env-builder helper above.)
            pass

        if isinstance(include_classes, (list, tuple)) and len(include_classes) > 0:
            csv = ",".join(str(int(x)) for x in include_classes)
            env_extra["TCRMP_STEP6_INCLUDE_CLASSES"] = csv

        # Device selection — forwarded as ultralytics `device=`. A comma-separated
        # value (e.g. "0,1") triggers DDP across those GPUs. Blank means "wrapper
        # default" (cuda:0).
        device = (cfg.get("device") or "").strip()
        if device:
            env_extra["TCRMP_STEP6_DEVICE"] = device

        current_project["steps"]["6"]["config"].update({
            "run_name": run_name,
            "task": task,
            "model": model,
            "model_path": model_path or None,
            "freeze": freeze if freeze not in (None, "") else None,
            "epochs": int(epochs),
            "imgsz": int(imgsz),
            "include_classes": include_classes if isinstance(include_classes, list) else None,
            # The ids above are now in the training source's id space.
            "include_classes_source": source_dir or None,
            "device": device or None,
            "source_dir": source_dir or None,
        })
        pm.save_project(current_project)
        result = runner.run_cli_stage(step, cmd, cwd=WORKING_DIRS[6], env_extra=env_extra)
        if isinstance(result, dict) and "error" not in result:
            result["task"] = task
            if dropped_classes:
                result["dropped_classes"] = dropped_classes
            if isinstance(include_classes, list):
                result["include_classes"] = list(include_classes)
            if split_summary:
                result["split"] = {"n_pairs": split_summary.get("n_pairs"),
                                   "n_unlabelled": split_summary.get("n_unlabelled"),
                                   "counts": split_summary.get("counts")}
        return result

    def _run_step7(step, cfg, step_dir):
        """Run ultralytics val on a trained run + emit metrics JSON + PDF report.

        Parameters:
            step: the internal step key (7), as the stage runner records it.
            cfg: the saved Step 7 config; cfg["run_dir"] names the run to score.
            step_dir: the open project's step 7 output folder.
        Returns:
            the stage runner's result dict, with any advisory sentences under
            "warnings"; or {"error": ..., "http_status": ...} when the
            selection cannot be scored and nothing was launched.

        The run the operator picked decides every path. A run of the open
        project is scored on the project's own dataset, writes its report to
        <step 7>/eval_<run name>, and appends its ledger row to the project's
        rounds.csv, exactly as this route has always done. A run that belongs
        to someone else (a model lab under inprocess/_models/, or another
        project) is scored on its own dataset, writes its report to
        inprocess/_models/_eval/<run name>_<stamp> (the folder the report
        route already searches), and appends its ledger row to its own
        rounds.csv, so the runs card can show its score. Nothing outside the
        open project is ever recorded as one of the project's Step 7 pointers,
        neither the output folders nor the saved selection.
        """
        step6_dir = current_project["steps"].get("6", {}).get("dir", "")
        raw_run_dir = (cfg.get("run_dir") or "").strip()
        pick_a_run = ("Select a training run to evaluate. Expected a path like "
                      f"{step6_dir}/runs/<run_name>/")

        def _refuse(message, http_status=HTTP_BAD_REQUEST):
            """Refuse this launch and forget the run that was picked.

            The client saves the pick into steps.7.config.run_dir (PUT
            /api/project/config/7) before this route sees it. Nothing was
            launched, so steps.7.outputs still describes this project's own
            last evaluation, and a refused pick left beside it would be
            paired with that folder by _resolve_evaluation on the next report
            card, PDF download or PDF copy asked without a run_dir, which is
            what the panel asks after a page reload: this project's numbers
            would be served under a run that was never scored. Forgetting the
            pick restores the pair, because the fallback reads the run back
            out of the evaluation folder's own eval_<run name> leaf.

            Parameters:
                message: what the reader is told.
                http_status: the status the route answers with.
            Returns:
                the result dict step_run turns into the response.
            """
            current_project["steps"]["7"]["config"]["run_dir"] = ""
            pm.save_project(current_project)
            return {"error": message, "http_status": http_status}

        if not raw_run_dir:
            return _refuse(pick_a_run)

        # Whose run is this? A run under the open project's own runs/ folder
        # keeps every path it has always had; anything else is foreign.
        # Containment is decided on real paths, the way labruns._contained
        # does it, so neither a '..' segment nor a symlink can pass a foreign
        # run off as one of this project's. A project with no step 6 folder
        # (nothing loaded yet) owns nothing.
        project_runs_dir = os.path.join(step6_dir, "runs") if step6_dir else ""
        owned = bool(project_runs_dir) and _contained(
            os.path.realpath(raw_run_dir), os.path.realpath(project_runs_dir))

        if not owned:
            # The client writes the picked run into this project's Step 7
            # config (PUT /api/project/config/7) before this route sees it, so
            # a run that is not this project's is sitting there right now. It
            # cannot stay. steps.7.outputs still points at this project's own
            # last evaluation, and _resolve_evaluation pairs that folder with
            # steps.7.config.run_dir every time the report card, the PDF
            # download or the PDF copy is asked without a run_dir, which is
            # what the panel asks after a page reload. Paired with a foreign
            # name, this project's own numbers would be served under another
            # run's name. Forgetting the selection restores the pair: the
            # fallback reads the run back out of the evaluation folder's own
            # eval_<run name> leaf. Nothing the operator can see is lost,
            # because the runs card re-selects a foreign run on every Evaluate
            # press, and the run list the panel rebuilds on a reload holds
            # this project's own runs only. A foreign pick that is then
            # refused is forgotten here as well as by _refuse, which does the
            # same for a refused pick of one of this project's own runs.
            current_project["steps"]["7"]["config"]["run_dir"] = ""
            pm.save_project(current_project)

        if owned:
            # The project's own run: the gate stays what it was, so a run
            # folder that exists but has no weights yet is still refused by
            # the evaluator with its own wording rather than here.
            if not os.path.isdir(raw_run_dir):
                return _refuse(pick_a_run)
            run_dir = raw_run_dir
        else:
            # Someone else's run. validate_run_dir resolves the path, refuses
            # anything outside the module's inprocess folder, and requires
            # weights/best.pt, so a typed path can neither escape the module
            # nor name a folder that is not a finished run.
            run_dir, reason = _labruns.validate_run_dir(
                raw_run_dir, root=_inprocess_root())
            if reason:
                return _refuse(reason)
            # An evaluation writes a ledger row beside the run it scored, so
            # the run's own folder must really sit inside the module. A link
            # from outside the module that points into a lab would put that
            # row outside the module, so it is refused rather than followed.
            ledger_dir = _labruns.rounds_dir_for_run(run_dir)
            if not _contained(os.path.realpath(ledger_dir),
                              os.path.realpath(_inprocess_root())):
                return _refuse(
                    f"The run folder is reached through a link from outside "
                    f"{_inprocess_root()}, so VEGA cannot write its evaluation "
                    f"ledger row beside it. Pick the run where it really sits, "
                    f"under that folder.")

        run_name = os.path.basename(run_dir.rstrip("/"))

        # Dataset, evaluation folder and ledger, each following the run.
        if owned:
            # Whatever the split step wrote for this project.
            dataset_dir = os.path.join(step6_dir, "dataset")
            if not os.path.isdir(dataset_dir):
                return _refuse(f"No dataset at {dataset_dir}. Has {step_ref(6)} run?")
            rounds_dir = step6_dir
        else:
            # The dataset beside the run's own runs/ parent: a lab's dataset/
            # for a lab run, that project's dataset/ for another project's.
            dataset_dir = _labruns.dataset_for_run(run_dir)
            if not dataset_dir:
                return _refuse(
                    f"The run '{run_name}' is not part of the open project and its own "
                    f"dataset folder is missing: VEGA expected "
                    f"{os.path.join(_labruns.rounds_dir_for_run(run_dir), 'dataset')} "
                    f"with a data.yaml inside it. A run is scored only on the dataset "
                    f"it was trained on.")
            rounds_dir = _labruns.rounds_dir_for_run(run_dir)

        py = PYTHON_PATHS.get(7)
        if not py or not os.path.isfile(py):
            return _refuse(
                f"{step_ref(7)} env not found: run TCRMPtrain_oceankindCV/setup_env.sh",
                HTTP_SERVER_ERROR)

        # Place evaluation outputs in a dedicated subdir tagged with the run name
        # so multiple evaluations don't clobber each other. A foreign run's
        # evaluation goes to the module-wide _eval folder instead of into this
        # project, and is stamped so repeated evaluations sit side by side;
        # GET /api/step/7/report searches both places for a run's newest.
        if owned:
            out_dir = os.path.join(step_dir, f"{EVAL_FOLDER_PREFIX}{run_name}")
        else:
            out_dir = _labruns.eval_out_dir(_inprocess_root(), run_name)
        os.makedirs(out_dir, exist_ok=True)

        # --------------------------------------------------------------
        # Image size: a blank field (the client sends an empty string) or 0
        # means the size the run trained at, read from its args.yaml through
        # the shared library; a value that is not a whole number is refused
        # before anything runs. A typed size that differs from the training
        # size by more than 2x still runs, with a warning in the response.
        # --------------------------------------------------------------
        trained_imgsz = _labruns.run_imgsz(run_dir, default=0)
        trained_task = _labruns.task_for_run_dir(run_dir)

        user_imgsz, imgsz_err = _parse_eval_imgsz(cfg.get("imgsz"))
        if imgsz_err:
            return _refuse(imgsz_err)

        warnings = []
        if not owned:
            # Say plainly whose data this score came from, so a number read
            # in this project's window is never taken for this project's.
            warnings.append(
                f"This run is not part of the open project, so VEGA scored it on its own "
                f"dataset ({dataset_dir}) and wrote its ledger row to "
                f"{os.path.join(rounds_dir, ROUNDS_CSV_NAME)}, not this project's."
            )
        if user_imgsz == 0:
            chosen_imgsz = trained_imgsz or EVAL_IMGSZ_FALLBACK
            warnings.append(
                f"Image size was blank; VEGA filled it from the run's args.yaml: {chosen_imgsz}."
                if trained_imgsz else
                f"Image size was blank and the run's args.yaml records no image size; "
                f"VEGA used {EVAL_IMGSZ_FALLBACK}."
            )
        else:
            chosen_imgsz = user_imgsz
            if trained_imgsz and (
                chosen_imgsz > 2 * trained_imgsz or trained_imgsz > 2 * chosen_imgsz
            ):
                warnings.append(
                    f"imgsz={chosen_imgsz} differs from training imgsz={trained_imgsz} "
                    f"by more than 2x. Metrics may be distorted; consider matching the "
                    f"training value."
                )

        # A detector is scored in the box family; the report and the ledger
        # row carry box numbers, and the ledger's mask columns stay blank
        # because a detector predicts no outline. Say which family applies so
        # the blank mask cells in the Rounds table read as expected.
        if trained_task == "detect":
            warnings.append(
                "This run trained a detector (boxes only), so it is scored on box metrics: "
                "box mAP, precision and recall. The mask columns of the ledger stay blank "
                "because a detector predicts no outlines."
            )

        # A panel saved before the measurement defaults were fixed carries no
        # confidence at all, or the old 0.25; fall back to the config, never to
        # a hardcoded number that could disagree with it, and say plainly when
        # the value in hand cannot measure the whole curve.
        chosen_conf = cfg.get("conf_threshold", EVAL_CONF_DEFAULT)
        if _above_measurement_ceiling(chosen_conf):
            warnings.append(
                f"Confidence {chosen_conf} leaves the low-confidence tail of the predictions "
                f"out of the score, so the mAP below reads lower than these weights earn, and "
                f"it cannot be compared with a run measured at {EVAL_CONF_DEFAULT}. Press Use "
                f"recommended defaults to measure at {EVAL_CONF_DEFAULT} instead."
            )

        # The overlap this score is measured at, recorded beside it. A
        # project created before the Evaluate defaults were fixed still holds
        # the old 0.7 in its own config, and the ledger records the overlap
        # with every score, so a run measured at one overlap is never ranked
        # against a run measured at another. Say so at the launch rather than
        # letting the number pass unremarked.
        chosen_iou = cfg.get("iou_threshold", EVAL_IOU_DEFAULT)
        if _differs_from_recommended("iou_threshold", chosen_iou):
            warnings.append(
                f"Overlap {chosen_iou} is not the {EVAL_IOU_DEFAULT} VEGA recommends, so this "
                f"score is recorded as measured at {chosen_iou} and is not ranked against a run "
                f"measured at {EVAL_IOU_DEFAULT}. Press Use recommended defaults to measure at "
                f"{EVAL_IOU_DEFAULT} instead."
            )

        cmd = [
            py, ENTRY_POINTS[7],
            "--run_dir", run_dir,
            "--dataset_dir", dataset_dir,
            "--out_dir", out_dir,
            "--split", str(cfg.get("split", EVAL_SPLIT_DEFAULT)),
            "--imgsz", str(chosen_imgsz),
            "--batch", str(cfg.get("batch", -1)),
            "--conf", str(chosen_conf),
            "--iou", str(chosen_iou),
            "--preview_count", str(cfg.get("preview_count", EVAL_PREVIEW_COUNT_DEFAULT)),
        ]
        pdf_export_dir = (cfg.get("pdf_export_dir") or "").strip()
        if pdf_export_dir:
            cmd += ["--pdf_export_dir", pdf_export_dir]

        # Active-learning rounds ledger: every eval appends/updates a row in
        # {rounds_dir}/rounds.csv (keyed by run_name, so re-evaluating a run
        # updates its row rather than duplicating it) and gates this round
        # against the current champion.json. rounds.csv + champion.json both
        # live directly under the run's own runs/ parent (a sibling of
        # dataset/), not under this eval's own out_dir, so every round's eval
        # shares one ledger and the runs card reads a lab run's score from the
        # lab's own ledger.
        cmd += ["--rounds_dir", rounds_dir]

        # Persist the image size and the overlap the evaluation really used,
        # so the panel reopens on the numbers the ledger row will carry. The
        # Step 7 pointer pair is written below, after the evaluation is
        # really away.
        current_project["steps"]["7"]["config"]["imgsz"] = chosen_imgsz
        current_project["steps"]["7"]["config"]["iou_threshold"] = chosen_iou
        pm.save_project(current_project)
        result = runner.run_cli_stage(step, cmd, cwd=WORKING_DIRS[7])
        if not isinstance(result, dict) or "error" in result:
            # The runner refused the launch, because the step's process from
            # an earlier press is still alive. Nothing was spawned, so the
            # project's saved evaluation stays the one that holds numbers and
            # the pick that was refused is forgotten beside it, exactly as in
            # _refuse. The result itself is handed back untouched: step_run
            # reads its "busy" marker to answer 409 and to undo what it
            # stamped on the step.
            current_project["steps"]["7"]["config"]["run_dir"] = ""
            pm.save_project(current_project)
            return result

        # The evaluation is away. A run of the open project records both
        # halves of its Step 7 pointer pair now, so the saved selection and
        # the saved evaluation folder always describe one and the same
        # evaluation: config.run_dir and outputs.out_dir are the pair
        # _resolve_evaluation falls back to when no run_dir is given. A
        # foreign run's selection and folders are deliberately recorded in
        # neither place; its report is found by run name under
        # inprocess/_models/_eval.
        if owned:
            current_project["steps"]["7"]["config"]["run_dir"] = run_dir
            outputs = current_project["steps"]["7"]["outputs"]
            outputs["out_dir"] = out_dir
            outputs["report_md"] = os.path.join(out_dir, "report.md")
            outputs["report_pdf"] = os.path.join(out_dir, "report.pdf")
            outputs["metrics_json"] = os.path.join(out_dir, "metrics.json")
            outputs["rounds_csv"] = os.path.join(rounds_dir, ROUNDS_CSV_NAME)
            outputs["champion_json"] = os.path.join(rounds_dir, _labruns.CHAMPION_FILE_NAME)
            pm.save_project(current_project)

        # Whose run this is decides, one status poll later, whether the open
        # project's own Evaluate step may be recorded as completed. Kept in
        # this process only; project.json records the project's own work.
        _remember_evaluation_launch(owned, run_name, out_dir)

        # Attach any warnings so the UI can surface them (non-blocking).
        if warnings:
            existing = result.get("warnings") or []
            if isinstance(existing, list):
                result["warnings"] = existing + warnings
            else:
                result["warnings"] = warnings
        return result

    def _run_step8(step, cfg, step_dir):
        """Run model inference on a user-selected source and save overlays + manifest."""
        run_dir = (cfg.get("run_dir") or "").strip()
        if not run_dir or not os.path.isdir(run_dir):
            return {"error": "Select a training run whose weights to use."}

        py = PYTHON_PATHS.get(8)
        if not py or not os.path.isfile(py):
            return {"error": f"{step_ref(8)} env not found: run TCRMPtrain_oceankindCV/setup_env.sh"}

        # User-supplied run name — sanitize and stamp into the output folder.
        import re as _re
        run_name = (cfg.get("run_name") or "").strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name:
            safe = _re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name)
            out_leaf = f"{safe}_{timestamp}"
        else:
            safe = ""
            out_leaf = f"infer_{timestamp}"
        out_dir = os.path.join(step_dir, out_leaf)
        os.makedirs(out_dir, exist_ok=True)

        source_type = cfg.get("source_type", "unused")
        cmd = [
            py, ENTRY_POINTS[8],
            "--run_dir", run_dir,
            "--out_dir", out_dir,
            "--source_type", source_type,
            "--conf", str(cfg.get("conf_threshold", 0.25)),
            "--iou", str(cfg.get("iou_threshold", 0.7)),
            "--imgsz", str(cfg.get("imgsz", 512)),
            "--sample_count", str(cfg.get("sample_count", 100)),
            "--sample_mode", str(cfg.get("sample_mode", "random")),
            "--mask_alpha", str(cfg.get("mask_alpha", 0.45)),
            "--run_name", run_name or out_leaf,
            "--project_name", current_project.get("name", ""),
        ]
        if cfg.get("draw_boxes"):
            cmd.append("--draw_boxes")
        if source_type == "dir":
            source_dir = (cfg.get("source_dir") or "").strip()
            if not source_dir:
                return {"error": "source_type=dir requires source_dir"}
            cmd += ["--source_dir", source_dir]
        elif source_type in ("unused", "full"):
            # Pass the step 3 selected_frames.csv so we can exclude those for "unused".
            selected_frames = (
                current_project["steps"].get("3", {}).get("outputs", {}).get("selected_frames")
                or current_project["steps"].get("3", {}).get("config", {}).get("selected_frames")
                or ""
            )
            # Resolve the clip dir the same way routing does: a stale saved
            # Step-4 clip_dir (e.g. an old Dropbox path) falls back to the local
            # full set, so visualization reads the same tree as the rest.
            clip_dir = _resolve_routing_clip_dir(
                {}, current_project["steps"].get("4", {}).get("config", {}))
            if selected_frames:
                cmd += ["--selected_frames", selected_frames]
            if clip_dir:
                cmd += ["--clip_dir", clip_dir]
        if cfg.get("save_overlays"):
            cmd.append("--save_overlays")
        if cfg.get("include_pts_variant"):
            cmd.append("--include_pts_variant")
        # Always persist per-detection polygons to predictions.json. The Refine
        # loop (step4loop) seeder reads this file to turn model detections into
        # pending review masks, so every inference run must produce it, not just
        # ones the user happens to flag for it.
        cmd.append("--save_predictions")
        # The GPU the panel chose; blank leaves the engine's own default.
        device = str(cfg.get("device") or "").strip()
        if device:
            cmd += ["--device", device]
        # The species code list (pipeline.yaml paths.master_codes_csv) gives
        # the overlay chips and the gallery their full names. A configured
        # file that is missing is reported, not forwarded, so the run still
        # completes. The run's own dataset can still name its classes
        # (run_inference.names_full_for_run reads names_full from the split
        # dataset's data.yaml), so the warning says the code list is the
        # source that is missing, not that every class is nameless.
        warnings = []
        codes_csv = str(PATHS.get("master_codes_csv") or "").strip()
        if codes_csv and os.path.isfile(codes_csv):
            cmd += ["--codes_csv", codes_csv]
        elif codes_csv:
            warnings.append(
                f"The species code list at {codes_csv} (pipeline.yaml paths.master_codes_csv) "
                "is missing, so overlays and the gallery show codes without names, "
                "unless the run's own dataset names them.")

        current_project["steps"]["8"]["config"]["run_dir"] = run_dir
        current_project["steps"]["8"]["outputs"]["out_dir"] = out_dir
        current_project["steps"]["8"]["outputs"]["manifest"] = os.path.join(out_dir, MANIFEST_JSON_NAME)
        pm.save_project(current_project)
        result = runner.run_cli_stage(step, cmd, cwd=WORKING_DIRS[8])
        if warnings and isinstance(result, dict) and "error" not in result:
            result["warnings"] = list(result.get("warnings") or []) + warnings
        return result

    # ------------------------------------------------------------------
    # Link existing outputs (skip running a step)
    # ------------------------------------------------------------------
    @app.route("/api/step/1/link", methods=["POST"])
    def step1_link():
        """Link existing all_points + master_codes to skip Step 1."""
        global current_project
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        # Step 1's other entry point, refused for the same reason /run is.
        refused = _refuse_for_composite()
        if refused is not None:
            return refused

        data = request.get_json(force=True)
        ap_path = data.get("all_points", "").strip()
        mc_path = data.get("master_codes", "").strip()

        if not ap_path or not os.path.isfile(ap_path):
            return jsonify({"error": f"all_points file not found: {ap_path}"}), 400

        # master_codes: try to auto-detect from same directory if not given
        if not mc_path:
            mc_path = os.path.join(os.path.dirname(ap_path), "master_codes.csv")
        if not os.path.isfile(mc_path):
            return jsonify({"error": f"master_codes file not found: {mc_path}"}), 400

        step1 = current_project["steps"]["1"]
        step_dir = step1["dir"]
        os.makedirs(step_dir, exist_ok=True)

        # Symlink or copy into the project step dir
        import shutil
        for src, name in [(ap_path, os.path.basename(ap_path)), (mc_path, os.path.basename(mc_path))]:
            dst = os.path.join(step_dir, name)
            if not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copy2(src, dst)

        step1["outputs"]["all_points"] = ap_path
        step1["outputs"]["master_codes"] = mc_path

        # Count rows for feedback
        rows = None
        try:
            with open(ap_path) as f:
                rows = sum(1 for _ in f) - 1  # minus header
        except Exception:
            pass

        pm.complete_step(current_project, 1)

        return jsonify({
            "success": True,
            "rows": rows,
            "state": _state_payload(current_project),
        })

    # ------------------------------------------------------------------
    # Step status / control
    # ------------------------------------------------------------------
    def _warn(status, note):
        """Append one plain sentence to a status payload's warnings list."""
        status["warnings"] = list(status.get("warnings") or []) + [note]

    def _survive_restart(step, status, step_dir):
        """Answer a poll for a step whose process the runner no longer holds.

        The restart script kills the orchestrator with SIGKILL and a training
        child spawned in its own session survives it, so after a restart
        poll_status reports the step as not running with no exit code at all.
        Reading that as a failure is how a training still writing into its run
        folder was marked errored on the first poll after every restart.

        Parameters:
            step: the step key being polled.
            status: the poll payload, updated in place.
            step_dir: the step's output folder, named in both notes.
        Returns:
            None. The payload gains running True plus orphan_pid when the
            kernel says the process is alive, else untracked True, and a
            sentence saying what is known either way. Nothing is written to
            project.json: whether that process finished or died cannot be
            read from here, and both answers would be a guess.
        """
        live = _live_training_run(step, step_dir)
        if live and live["pid"]:
            status["running"] = True
            status["orphan_pid"] = live["pid"]
            _warn(status, ORPHAN_STEP_NOTE.format(step_ref=step_ref(step),
                                                  pid=live["pid"], step_dir=step_dir))
            return
        status["untracked"] = True
        _warn(status, RESTARTED_STEP_NOTE.format(step_ref=step_ref(step), step_dir=step_dir))

    def _finish_for_owner(step, status, launch):
        """Record a finished step against another project and say so here.

        Parameters:
            step: the step key that finished.
            status: the poll payload, updated in place.
            launch: the launch record naming the project that started it.
        Returns:
            None. completed is False for the open project, which did no work,
            and one sentence names the project the result went to.
        """
        succeeded = status.get("exit_code") == 0
        reason = _record_step_for_owner(step, launch, succeeded, status.get("exit_code"))
        note = (FOREIGN_STEP_UNWRITABLE_NOTE if reason else FOREIGN_STEP_NOTE).format(
            step_ref=step_ref(step),
            project_name=launch.get("project_name") or launch.get("project_id") or "(unnamed)",
            reason=reason)
        status["completed"] = False
        _warn(status, note)

    @app.route("/api/step/<int:step>/status")
    def step_status(step):
        """Whether a step's process is running, and what it left behind.

        A finished process is recorded against the project that LAUNCHED it,
        which is not always the one open now: the runner holds one process per
        step for the whole orchestrator, so a training started in one project
        can finish while another is open. A poll taken in that other window
        records nothing there and says where the result went.

        A step the runner no longer holds at all (an orchestrator restart
        while it ran) is neither completed nor failed here: _survive_restart
        consults the kernel for the process and reports what is actually
        known.
        """
        status = runner.poll_status(step)
        # For Flask stages, add health check
        port = runner.ports.get(step)
        if port:
            status["healthy"] = runner.health_check(port)
            status["port"] = port

        # Auto-detect CLI completion
        if current_project:
            s = str(step)
            st = current_project["steps"].get(s, {})
            if st.get("status") == "running" and not status["running"]:
                if status["exit_code"] is None:
                    # poll_status reports no exit code only when the runner
                    # holds no process for this step, which after a launch
                    # means the orchestrator was restarted under it.
                    _survive_restart(step, status, st.get("dir", ""))
                    return jsonify(status)
                launch = _take_step_launch(step)
                # Evaluate is left to its own record: it can be launched
                # against a run the launching project does not own, and only
                # that record knows whether the project may be credited at
                # all. Every other step belongs to whoever pressed it.
                if (step != EVALUATE_STEP and launch
                        and launch.get("project_id") != str(current_project.get("id") or "")):
                    _finish_for_owner(step, status, launch)
                    return jsonify(status)
                if status["exit_code"] == 0:
                    # An Evaluate run that scored a run the open project does
                    # not own finishes here and nowhere else: the project did
                    # no work, so nothing about it may be recorded as done.
                    if _finish_foreign_evaluation(step, status):
                        return jsonify(status)
                    _vic_end_step(step, "success")
                    pm.complete_step(current_project, step)
                    status["completed"] = True
                    if step == 3:
                        # select_images.py writes output/selection_summary.json
                        # (per_site, per_species, cap_frac, cap_ok,
                        # cap_satisfiable, n_selected, shortfalls) next to the
                        # .txt summary. Surfaced here so the UI can render a
                        # site breakdown instead of a bare success message;
                        # omitted cleanly (no key) for a legacy run with no JSON.
                        summary_path = os.path.join(st.get("dir", ""), "selection_summary.json")
                        if os.path.isfile(summary_path):
                            try:
                                with open(summary_path) as f:
                                    status["site_summary"] = json.load(f)
                            except (OSError, ValueError):
                                pass
                else:
                    if step == EVALUATE_STEP:
                        # The evaluation is over, so its record is spent
                        # whether it succeeded or not; a later poll must not
                        # read it.
                        _forget_evaluation_launch()
                    st["status"] = "error"
                    _vic_end_step(step, "failed",
                                  notes=f"exit_code={status.get('exit_code')}")
                    pm.save_project(current_project)

        return jsonify(status)

    @app.route("/api/step/<int:step>/log")
    def step_log(step):
        offset = request.args.get("offset", 0, type=int)
        lines, new_offset, dropped = runner.get_log(step, offset)
        return jsonify({
            "lines": lines,
            "offset": new_offset,
            "dropped": dropped,
        })

    @app.route("/api/step/<int:step>/done", methods=["POST"])
    def step_done(step):
        """User marks an interactive (Flask) step as complete."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        # Kill the sub-app
        runner.kill(step)
        if step == 5:
            _sam3_mark_stopped("Step 5 marked done.")
        _vic_end_step(step, "success", notes="marked done via UI")
        pm.complete_step(current_project, step)
        return jsonify({"success": True, "state": _state_payload(current_project)})

    @app.route("/api/step/<int:step>/stop", methods=["POST"])
    def step_stop(step):
        """Stop a step's process, and say what was really stopped.

        Three answers, and the press gets the one that happened:
          stopped True     the runner held the process and killed it.
          stopped False    nothing was running. The step's stale "running"
                           mark is cleared, because nothing is.
          HTTP_CONFLICT    a process is alive that outlived an orchestrator
                           restart, so the runner cannot stop it. Its pid is
                           named and the step's mark is left alone: reporting
                           success while a trainer goes on writing is the
                           lie this answers.
        """
        s = str(step)
        st = (current_project or {}).get("steps", {}).get(s, {})
        step_dir = st.get("dir", "")
        held = runner.is_running(step)
        runner.kill(step)
        if step == 5:
            _sam3_mark_stopped("Step 5 stopped.")
        if not held:
            live = _live_training_run(step, step_dir, st.get("config"))
            if live and live["pid"]:
                return jsonify({
                    "success": False,
                    "stopped": False,
                    "orphan_pid": live["pid"],
                    "error": (f"{step_ref(step)} is still running as process {live['pid']}, which "
                              f"outlived a VEGA restart, so the orchestrator cannot stop it. "
                              f"Stop process {live['pid']} yourself, then press Stop again."),
                }), HTTP_CONFLICT
        if current_project and st.get("status") == "running":
            st["status"] = "pending"
            _vic_end_step(step, "cancelled",
                          notes="stopped via UI" if held else "stale running mark cleared via UI")
            pm.save_project(current_project)
        _forget_step_launch(step)
        return jsonify({"success": True, "stopped": held,
                        "message": ("Stopped." if held
                                    else f"Nothing was running for {step_ref(step)}.")})

    @app.route("/api/step/<int:step>/reset", methods=["POST"])
    def step_reset(step):
        """Clear a step's outputs and every step after it.

        Refused with HTTP_CONFLICT while a training run is alive in any step
        the reset would clear, because pm.reset_step empties those folders:
        the dataset the trainer is reading and the run folder it is writing
        weights into both go, and the run dies mid-epoch with its output half
        removed. The refusal names the run. Start Training refuses over the
        same probe, so the two cannot disagree about whether a trainer is
        alive.
        """
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        if step in _COMPOSITE_REFUSED_RESET_STEPS:
            refused = _refuse_for_composite()
            if refused is not None:
                return refused
        last = int(pm.STEP_KEYS[-1])
        # COMPOSITE_TRAIN_STEP is the Train step's internal key: the one step
        # whose folder a long-running process writes into for hours.
        if step <= COMPOSITE_TRAIN_STEP <= last:
            train_step = current_project["steps"].get(str(COMPOSITE_TRAIN_STEP), {})
            live = _live_training_run(COMPOSITE_TRAIN_STEP, train_step.get("dir", ""),
                                      train_step.get("config"))
            if live:
                where = (f" It is process {live['pid']}, which outlived a VEGA restart, so the "
                         f"orchestrator cannot stop it." if live["pid"] else
                         " Press Stop on that step first.")
                return jsonify({
                    "busy": True,
                    "orphan_pid": live["pid"],
                    "error": (f"The run '{live['run_name'] or '(unnamed)'}' is still training in "
                              f"{step_ref(COMPOSITE_TRAIN_STEP)}, and this reset would delete the "
                              f"dataset it is reading and the folder it is writing to.{where}"),
                }), HTTP_CONFLICT
        # Kill any running processes for this and later steps
        for s in range(step, last + 1):
            runner.kill(s)
        if step <= 5:
            _sam3_mark_stopped("Step reset.")
        pm.reset_step(current_project, step)
        for s in range(step, last + 1):
            _forget_step_launch(s)
        return jsonify({"success": True, "state": _state_payload(current_project)})

    # ------------------------------------------------------------------
    # Remap endpoints
    # ------------------------------------------------------------------
    @app.route("/api/remap/browse")
    def remap_browse():
        extra = []
        if current_project:
            extra.append(current_project["project_dir"])
        logs = remap_loader.browse_remap_logs(REPO_DIR, extra_dirs=extra)
        return jsonify({"logs": logs})

    @app.route("/api/remap/load", methods=["POST"])
    def remap_load():
        """Load a saved label mapping. The file must sit inside the module folder."""
        data = request.get_json(force=True, silent=True) or {}
        path, err, status = _read_path_inside_module(data.get("path"), "a remap log")
        if err:
            return jsonify({"error": err}), status
        if not os.path.isfile(path):
            return jsonify({"error": "File not found"}), 404
        try:
            log = remap_loader.load_remap_log(path)
            return jsonify({"success": True, "remap_log": log})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/remap/apply", methods=["POST"])
    def remap_apply():
        """Auto-apply remaps without launching interactive UI."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400

        data = request.get_json(force=True)
        remap_log_path, err, status = _read_path_inside_module(data.get("remap_log_path"), "a remap log")
        if err:
            return jsonify({"error": err}), status
        if not os.path.isfile(remap_log_path):
            return jsonify({"error": "Remap log file not found"}), 404

        step2 = current_project["steps"]["2"]
        ap = step2["config"].get("all_points", "")
        mc = step2["config"].get("master_codes", "")
        if not ap or not mc:
            return jsonify({"error": f"{step_ref(1)} outputs are not linked (all_points / master_codes)"}), 400

        output_dir = step2["dir"]
        try:
            step2["status"] = "running"
            pm.save_project(current_project)

            result = remap_loader.apply_remaps(ap, mc, remap_log_path, output_dir)

            step2["config"]["remap_log_source"] = remap_log_path
            pm.complete_step(current_project, 2)

            return jsonify({"success": True, **result, "state": _state_payload(current_project)})
        except Exception as e:
            step2["status"] = "error"
            pm.save_project(current_project)
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------
    @app.route("/api/fs/open", methods=["POST"])
    def fs_open():
        """Open a directory in the desktop file manager via xdg-open.

        Accepts { path } and only opens paths inside the current project or
        the repo — we do not want random server-side path access.
        """
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        data = request.get_json(force=True) or {}
        path = (data.get("path") or "").strip()
        if not path:
            return jsonify({"error": "not a directory: (empty path)", "path": ""}), 400
        return _folder_response(path, current_project["project_dir"])

    # ------------------------------------------------------------------
    # Step 3 helpers — label picker feed
    # ------------------------------------------------------------------
    @app.route("/api/step/3/target_labels")
    def step3_target_labels():
        """Return the label universe for the Step-3 picker.

        This feed is AUTO-LOADED by the Step 3 panel when the user enters the
        step (there is no manual "load labels" button); it is the sole way
        labels reach Step 3. The frontend groups the returned labels by their
        `category` string (empty/missing category is bucketed under
        "Uncategorized" client-side) and re-sorts within each category, so the
        server-side ordering below is harmless.

        Source of truth resolution (so the correct labels load straight from
        Step 2 when a recode has happened):
          1. Step 2's `master_codes_recoded.csv` if it exists (post-recode).
          2. Step 2's configured `master_codes`.
          3. Step 1's `master_codes.csv` (outputs, else config) otherwise.
        Also returns the excluded-code set from Step 2's remap log so the UI
        can mark or hide them. Row counts per label come from the current
        all_points CSV (step 2 recoded, else step 1) for sorting.

        Response shape (the client depends on these keys):
          labels[], excludes[], master_codes_path, all_points_path,
          remap_log_path. Every label always carries a clean `category`
          string (NaN is normalized away via .fillna("")). A missing master
          codes file returns {"error": ...}, 404; an unreadable one returns
          {"error": ...}, 500 rather than a raw stack trace.
        """
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        s1 = current_project["steps"]["1"]
        s2 = current_project["steps"]["2"]
        # Master codes: prefer step 2 output, fall back to step 1.
        mc_path = (
            s2.get("outputs", {}).get("master_codes_recoded")
            or s2.get("config", {}).get("master_codes")
            or s1.get("outputs", {}).get("master_codes")
            or s1.get("config", {}).get("master_codes", "")
        )
        if not mc_path or not os.path.isfile(mc_path):
            return jsonify({"error": f"master_codes not found: {mc_path}"}), 404
        # All points: prefer recoded.
        ap_path = (
            s2.get("outputs", {}).get("all_points_recoded")
            or s2.get("config", {}).get("all_points")
            or s1.get("outputs", {}).get("all_points", "")
        )
        # Excludes: look in the remap log that was applied.
        excludes: list[str] = []
        remap_log = s2.get("config", {}).get("remap_log_source", "")
        if remap_log and os.path.isfile(remap_log):
            try:
                with open(remap_log) as f:
                    excludes = list(json.load(f).get("excludes", []) or [])
            except Exception:
                excludes = []
        try:
            import pandas as _pd
            mc = _pd.read_csv(mc_path)
            mc = mc.fillna("")
            counts: dict = {}
            frame_counts: dict = {}
            if ap_path and os.path.isfile(ap_path):
                try:
                    ap = _pd.read_csv(
                        ap_path,
                        usecols=["species_code", "date", "site", "transect", "frame"],
                    )
                    counts = ap["species_code"].value_counts().to_dict()
                    ap["__fid"] = (
                        ap["date"].astype(str) + "|" + ap["site"].astype(str) + "|"
                        + ap["transect"].astype(str) + "|" + ap["frame"].astype(str)
                    )
                    frame_counts = (
                        ap.groupby("species_code")["__fid"].nunique().to_dict()
                    )
                except Exception:
                    counts = {}
                    frame_counts = {}
            labels = []
            for _, row in mc.iterrows():
                code = str(row.get("code", ""))
                if not code:
                    continue
                labels.append({
                    "code": code,
                    "name": str(row.get("name", "")),
                    "category": str(row.get("category", "")),
                    "count": int(counts.get(code, 0)),
                    "frames": int(frame_counts.get(code, 0)),
                    "excluded": code in excludes,
                })
            labels.sort(key=lambda r: (r["frames"], r["count"]), reverse=True)
            return jsonify({
                "labels": labels,
                "excludes": excludes,
                "master_codes_path": mc_path,
                "all_points_path": ap_path,
                "remap_log_path": remap_log,
            })
        except Exception as exc:
            return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500

    # ------------------------------------------------------------------
    # Step 7/8: listings, reports, downloads
    # ------------------------------------------------------------------
    @app.route("/api/step/6/presets")
    def step6_list_presets():
        """List the Step 6 training recipes, with what each one declares it is for.

        Every .yaml or .yml file shipped in config/presets/ (full path
        /mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/
        config/presets/, resolved through PRESET_DIRS["train"]). The two
        built-in presets (Fine-tune preset, Full retrain preset) were removed
        2026-09-21: the Train panel starts from a base model by default (the
        old Full retrain preset's shape) and the Continue from a VEGA model
        disclosure covers the old Fine-tune preset, so a recipe file is the
        only kind of preset left to list.

        A recipe's id is its file name with the extension, which is the same
        spelling /api/step/6/presets/<id> takes.

        This route CHOOSES NOTHING. It reports what each file declares, and
        the Train panel matches those declarations against its two controls,
        Training task and Subject, applying a recipe only when exactly one
        matches. Selection by file name was removed on 2026-09-15 with the
        `default_for_source` and `default: true` flags it resolved, because
        whichever name sorted first won and the panel could not say why.

        Every entry carries:
          task        "detect", "segment", or "" when _meta.task says neither
          subject     "coral", "fish", "algae", "any", or "" likewise
          classified  True only when both are known. A False entry is listed
                      so it can still be imported by hand, and is never
                      matched against the panel's controls.
          warning     why an unclassified recipe is unclassified, naming what
                      the file said and what the key accepts

        The payload also carries `dataset_subject`: the subject the open
        project's staged box dataset recorded in its ingest_manifest.json, ""
        for every other project. The panel uses it as the initial value of its
        Subject control and for nothing else. `dataset_subject_warning` says
        why when that file is there and could not be read.

        Files are listed in name order so the picker is stable between loads.
        Nothing is decided by that order.

        One damaged file never takes out the listing: a file that will not
        parse, one that parses to something other than a mapping, and one
        whose `_meta` section is not a mapping are each reported in place, so
        every other recipe still reaches the picker.
        """
        out = []
        preset_dir = PRESET_DIRS.get("train", "")
        if os.path.isdir(preset_dir):
            import yaml as _y
            for fname in sorted(os.listdir(preset_dir)):
                if not fname.endswith(".yaml") and not fname.endswith(".yml"):
                    continue
                fpath = os.path.join(preset_dir, fname)
                # A damaged entry still carries the documented keys, so a
                # consumer reading task and subject on every entry does not
                # break on one unreadable file.
                damaged = {"id": fname, "task": "", "subject": "", "classified": False}
                try:
                    with open(fpath) as f:
                        data = _y.safe_load(f) or {}
                except Exception as e:
                    out.append(dict(damaged, error=f"parse error: {e}"))
                    continue
                if not isinstance(data, dict):
                    out.append(dict(damaged, error=(
                        f"parse error: a preset must be a mapping with _meta and "
                        f"params sections, {fpath} holds {type(data).__name__}")))
                    continue
                # A _meta section that is not a mapping (a scalar, a list) is
                # unusable but says nothing about the params below it: treat it
                # as absent rather than failing the whole listing on one file.
                raw_meta = data.get("_meta")
                meta = raw_meta if isinstance(raw_meta, dict) else {}
                task, subject, warning = _step6_recipe_declaration(meta)
                entry = {
                    "id": fname,
                    "name": meta.get("name") or fname,
                    "description": meta.get("description") or "",
                    "task": task,
                    "subject": subject,
                    "classified": not warning,
                }
                if warning:
                    entry["warning"] = warning
                out.append(entry)

        dataset_subject, dataset_warning = _project_dataset_subject(current_project)
        payload = {"presets": out, "dir": preset_dir,
                   "dataset_subject": dataset_subject}
        if dataset_warning:
            payload["dataset_subject_warning"] = dataset_warning
        return jsonify(payload)

    @app.route("/api/step/6/presets/<path:preset_id>")
    def step6_get_preset(preset_id):
        """Return the parsed params dict for one preset.

        preset_id is the file name WITH its extension (the same spelling
        /api/step/6/presets lists).

        The listing renders a damaged preset as a selectable option, so this
        route is one click from every shape that listing tolerates. A file that
        is not a mapping, and a `params` section that is not a mapping, each
        answer a JSON error naming the file and what it holds; an unusable
        `_meta` section is treated as absent, since the params below it still
        load. The client reads `error` and shows it, so no case is silent."""
        # Reject any path traversal attempt: only a plain file name in the dir.
        safe_name = os.path.basename(preset_id)
        if safe_name != preset_id:
            return jsonify({"error": "invalid preset id"}), 400
        preset_dir = PRESET_DIRS.get("train", "")
        fpath = os.path.join(preset_dir, safe_name)
        if not os.path.isfile(fpath):
            return jsonify({"error": f"preset not found: {safe_name}"}), 404
        import yaml as _y
        try:
            with open(fpath) as f:
                data = _y.safe_load(f) or {}
        except Exception as e:
            return jsonify({"error": f"parse error: {e}"}), 500
        if not isinstance(data, dict):
            return jsonify({"error": (
                f"parse error: a preset must be a mapping with _meta and params "
                f"sections, {fpath} holds {type(data).__name__}")}), 500
        # An unusable _meta section costs the preset its name and description,
        # not its params: fall back to the file name, as the listing does.
        raw_meta = data.get("_meta")
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        raw_params = data.get("params")
        if raw_params is not None and not isinstance(raw_params, dict):
            # A list here would reach the client as numeric keys and apply
            # nothing, in silence. Refuse and say what the section holds.
            return jsonify({"error": (
                f"parse error: the params section of a preset must be a mapping of "
                f"form field to value, {fpath} holds "
                f"{type(raw_params).__name__}")}), 500
        params = raw_params or {}
        return jsonify({
            "id": safe_name,
            "name": meta.get("name") or safe_name,
            "description": meta.get("description") or "",
            "params": params,
        })

    # ------------------------------------------------------------------
    # Step 7: preset loader (mirrors step 6 preset endpoints) + run_info
    # ------------------------------------------------------------------
    @app.route("/api/step/7/presets")
    def step7_list_presets():
        """List Step 7 evaluation presets shipped in config/eval_presets/ (full
        path /mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/
        github_repo/config/eval_presets/, resolved through
        PRESET_DIRS["eval"])."""
        preset_dir = PRESET_DIRS.get("eval", "")
        out = []
        if os.path.isdir(preset_dir):
            import yaml as _y
            for fname in sorted(os.listdir(preset_dir)):
                if not fname.endswith(".yaml") and not fname.endswith(".yml"):
                    continue
                fpath = os.path.join(preset_dir, fname)
                try:
                    with open(fpath) as f:
                        data = _y.safe_load(f) or {}
                except Exception as e:
                    out.append({"id": fname, "error": f"parse error: {e}"})
                    continue
                meta = data.get("_meta") or {}
                out.append({
                    "id": fname,
                    "name": meta.get("name") or fname,
                    "description": meta.get("description") or "",
                })
        return jsonify({"presets": out, "dir": preset_dir})

    @app.route("/api/step/7/presets/<path:preset_id>")
    def step7_get_preset(preset_id):
        """Return the parsed params dict for one Step 7 preset. preset_id is the filename."""
        safe_name = os.path.basename(preset_id)
        if safe_name != preset_id:
            return jsonify({"error": "invalid preset id"}), 400
        preset_dir = PRESET_DIRS.get("eval", "")
        fpath = os.path.join(preset_dir, safe_name)
        if not os.path.isfile(fpath):
            return jsonify({"error": f"preset not found: {safe_name}"}), 404
        import yaml as _y
        try:
            with open(fpath) as f:
                data = _y.safe_load(f) or {}
        except Exception as e:
            return jsonify({"error": f"parse error: {e}"}), 500
        meta = data.get("_meta") or {}
        params = data.get("params") or {}
        return jsonify({
            "id": safe_name,
            "name": meta.get("name") or safe_name,
            "description": meta.get("description") or "",
            "params": params,
        })

    def _eval_recommended():
        """The settings the Evaluate panel's Use recommended defaults offers.

        Read from config/pipeline.yaml so the button, the value a new project
        is created with and the launch fallback can never disagree. The
        confidence is deliberately near zero: average precision is the area
        under the whole precision-recall curve, so a higher cut-off would
        remove its low-confidence tail and understate the score.

        Returns:
            A copy of EVAL_RECOMMENDED: split, conf_threshold, iou_threshold
            and preview_count.
        """
        return dict(EVAL_RECOMMENDED)

    @app.route("/api/step/7/run_info")
    def step7_run_info():
        """Peek at a training run's args.yaml so the Step 7 Quick-start banner
        can auto-fill `imgsz` (among a few other useful fields), and hand back
        the recommended measurement settings the same banner fills in, plus
        off_recommended: every setting the open project has SAVED that is not
        the recommended one, as {key: {saved, recommended}}, so the panel can
        say which of the numbers it opened on differ. Returns the subset of
        fields the UI needs; missing file -> 200 with empty values."""
        raw = (request.args.get("path") or "").strip()
        if not raw:
            return jsonify({"error": "missing ?path=<run_dir>"}), 400
        run_dir, err, status = _read_path_inside_module(raw, "a run folder")
        if err:
            return jsonify({"error": err}), status
        if not os.path.isdir(run_dir):
            return jsonify({"error": f"not a directory: {run_dir}"}), 404
        args_path = os.path.join(run_dir, "args.yaml")
        info = {"model": None, "imgsz": None, "epochs": None,
                "recommended": _eval_recommended(),
                "off_recommended": _eval_settings_off_recommended(current_project)}
        if os.path.isfile(args_path):
            import yaml as _y
            try:
                with open(args_path) as f:
                    data = _y.safe_load(f) or {}
            except Exception as e:
                return jsonify({"error": f"parse error: {e}", **info}), 500
            info["model"] = data.get("model")
            info["imgsz"] = data.get("imgsz")
            info["epochs"] = data.get("epochs")
        return jsonify({"run_dir": run_dir, "args_yaml_found": os.path.isfile(args_path), **info})

    @app.route("/api/composites")
    def api_composites():
        """The Step 6 "Training source" list: composite YOLO exports under
        <inprocess>/_composite/ (built on the coverage matrix; kind
        "composite", task seg, the same item shape as
        /matrix/api/composites), then the Annotate Images sessions under
        <inprocess>/_standalone/ that hold an export (kind "standalone",
        task seg, decision record 0006), then the staged box datasets under
        supporting_data/bbox_datasets (kind "bbox", task detect). A box row
        carries no mask count (n_masks is null): its labels are boxes.

        Every row says whether the OPEN project may train from it
        (usable_here), and a row it may not carries the one-sentence reason
        the launch would give. Only a box training project may train from a
        staged box dataset: building that split replaces the open project's
        own dataset folder, which is the split its outline runs were scored
        on. Without this the list offered a choice that Start Training always
        refuses.
        """
        box_refusal = ""
        if (current_project or {}).get("project_type") != BBOX_PROJECT_TYPE:
            box_refusal = ("This project trains on outlines. Open the Bounding Boxes page and "
                           "create a training project for a box dataset.")
        composites = [dict(c, kind="composite", task="seg", usable_here=True, refusal="")
                      for c in _list_composites()]
        # Decision record 0006: Annotate Images sessions are a training source.
        composites += [dict(c, usable_here=True, refusal="") for c in _list_standalone_exports()]
        boxes = [{"kind": "bbox", "task": "detect", "name": b["name"], "path": b["path"],
                  "created_at": b["built_at"], "n_images": b["n_images"], "n_masks": None,
                  "labels": b["labels"], "usable_here": not box_refusal,
                  "refusal": box_refusal}
                 for b in _labruns.list_bbox_sources(_bbox_root())]
        return jsonify({"composites": composites + boxes, "root": _composite_root(),
                        "bbox_root": _bbox_root()})

    @app.route("/api/step/6/list_classes")
    def step6_list_classes():
        """Enumerate classes present in the Step 4 (4.test) export so the user can
        pick which ones to include in training. Returns class id, name (from
        data.yaml), and per-class label-instance count from all_labels/.

        Unnamed/empty-name classes (including the literal "(unnamed)") are
        filtered out — they have leaked into prior training runs and the user
        wants them suppressed at source.

        Fallback: if Step 4 hasn't exported yet (no data.yaml), synthesize the
        class list from step 3's target_species so the panel is still usable.
        """
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400

        # H2 (2026-08-26): the picker follows the selected training source.
        # ?source_dir= blank -> this project's Step 4 export; else a composite
        # export validated exactly like _run_step6's cfg.source_dir.
        picker_source = str(request.args.get("source_dir") or "").strip()
        if picker_source:
            src_dir, src_err = _validate_outline_source(picker_source)
            if src_err:
                return jsonify({"error": src_err, "source_dir": picker_source}), 400
        else:
            src_dir = _step4test_export_dir()
        yaml_path = os.path.join(src_dir, "data.yaml") if src_dir else ""
        src_ready = bool(src_dir and os.path.isfile(yaml_path))

        names = {}
        if src_ready:
            try:
                import yaml as _y
                with open(yaml_path) as f:
                    data = _y.safe_load(f) or {}
                raw = data.get("names") or {}
                if isinstance(raw, dict):
                    names = {int(k): v for k, v in raw.items()}
                elif isinstance(raw, list):
                    names = {i: v for i, v in enumerate(raw)}
            except Exception as e:
                return jsonify({"error": f"Could not parse {yaml_path}: {e}"}), 500

        counts = {}
        image_counts = {}
        if src_ready:
            label_dir = os.path.join(src_dir, "all_labels")
            if os.path.isdir(label_dir):
                for fname in os.listdir(label_dir):
                    fpath = os.path.join(label_dir, fname)
                    seen_in_file = set()
                    try:
                        with open(fpath) as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    cid = int(line.split()[0])
                                except ValueError:
                                    continue
                                counts[cid] = counts.get(cid, 0) + 1
                                seen_in_file.add(cid)
                    except OSError:
                        continue
                    for cid in seen_in_file:
                        image_counts[cid] = image_counts.get(cid, 0) + 1

        # If step 5 never ran, fall back to step 3's configured target_species
        # so the panel can at least preview the expected class roster. No
        # counts available in this mode.
        source = "step4"
        if not names and not counts:
            s3_cfg = current_project["steps"].get("3", {}).get("config", {}) or {}
            species_raw = s3_cfg.get("target_species") or ""
            species = [s.strip() for s in species_raw.split(",") if s.strip()]
            if species:
                names = {i: sp for i, sp in enumerate(species)}
                source = "step3_fallback"
            else:
                return jsonify({
                    "error": (f"{step_ref('step4test')} has no output yet and "
                              f"{step_ref(3)} has no target labels. Run one of them first."),
                }), 400

        all_ids = sorted(set(names.keys()) | set(counts.keys()))
        classes = []
        dropped_unnamed = []
        for cid in all_ids:
            nm = names.get(cid)
            if _step6_is_unnamed_class(nm):
                dropped_unnamed.append(cid)
                continue
            classes.append({
                "id": cid,
                "name": str(nm).strip(),
                "instance_count": counts.get(cid, 0),
                "image_count": image_counts.get(cid, 0),
            })
        s6cfg = current_project["steps"].get("6", {}).get("config", {}) or {}
        saved = s6cfg.get("include_classes")
        saved_dropped = []
        if isinstance(saved, (list, tuple)):
            # Saved ids live in the id space of the source the picker was
            # loaded from (legacy configs: the project export). Carry them
            # by name into the source requested now; names it lacks are
            # dropped, and a vanished saved source means every class ticked.
            saved_src = str(s6cfg.get("include_classes_source") or "").strip()
            saved_dir, saved_err = _class_picker_source(saved_src)
            if saved_err:
                saved = None
            elif os.path.realpath(saved_dir) != os.path.realpath(src_dir or ""):
                saved, saved_dropped = _translate_class_ids(
                    saved, _step6_picker_names(saved_dir, saved_src), names)
        return jsonify({
            "classes": classes,
            "src_dir": src_dir,
            "include_classes": saved,
            "include_classes_source": picker_source,
            "saved_classes_dropped": saved_dropped,
            "source": source,
            "dropped_unnamed_ids": dropped_unnamed,
        })

    @app.route("/api/step/6/source_summary")
    def step6_source_summary():
        """Read-only preview of the Train panel's Training data section: how
        many images and labels a source holds, and what its train, validation
        and test split would look like, all read before Start Training builds
        anything (design 2026-09-21-vega-starting-model-and-training-data-
        design.md section 7).

        Query:
          source_dir       blank for the open project's own recorded source,
                            else a path _step6_summary_resolve_source accepts
                            (the same rules _run_step6 validates a picked
                            source by).
          pin_split         "1" (default) or "0": outline sources only,
                            mirrors the Holdout split checkbox and
                            TCRMP_STEP6_PIN_SPLIT.
          valid_ratio, test_ratio: the held-out fractions, defaulting to the
                            panel's own pipeline.yaml step 6 defaults.
          task              "segment" or "detect"; refused when it names a
                            task the resolved source cannot train (a box
                            dataset holds no outlines).
          include_classes   comma-separated class ids; outline sources only,
                            blank means every named class.

        Returns 200 {ok, name, kind, path, n_images, n_labels, label_noun,
        n_classes, n_background, pictures_sentence, split: {rule, sentence,
        n_train, n_valid, n_test, approximate}, warnings}, kind one of
        "export", "composite", "standalone", "bbox". n_background is always 0
        for an outline source: the spec defines it only for a box source
        ("the number with no box"), where an image with no label file still
        counts toward n_images.

        Returns 400 {error} for no project, a missing folder, a source
        outside the allowed roots, a malformed query value, or a task the
        source cannot train.

        Never writes: no split is built and no project or run folder is
        touched. A segmentation source asked for task=detect is reported
        using its own outline counts unchanged (deriving the box view a
        detect run would train from is the trainer's job, out of scope for
        this preview per design section 6).
        """
        if current_project is None:
            return jsonify({"error": "No project loaded"}), HTTP_BAD_REQUEST
        path, kind, name, err = _step6_summary_resolve_source(request.args.get("source_dir"))
        if err:
            return jsonify({"error": err}), HTTP_BAD_REQUEST
        _task, err = _step6_summary_task(kind, name, path, request.args.get("task"))
        if err:
            return jsonify({"error": err}), HTTP_BAD_REQUEST
        valid_ratio, test_ratio, err = _step6_summary_ratios(request.args)
        if err:
            return jsonify({"error": err}), HTTP_BAD_REQUEST
        include_ids, err = _step6_summary_include_ids(request.args)
        if err:
            return jsonify({"error": err}), HTTP_BAD_REQUEST
        pin_split = (request.args.get("pin_split") or "1").strip() != "0"

        warnings = []
        if kind == "bbox":
            basenames, n_labels, n_background, n_classes = _step6_bbox_counts(path)
            n_images = len(basenames)
            label_noun = "boxes"
            rule, n_train, n_valid, n_test, approximate, warn = _step6_bbox_split(
                path, basenames, valid_ratio, test_ratio)
        else:
            counted, n_labels, n_classes = _step6_outline_counts(path, include_ids)
            n_images, n_background, label_noun = len(counted), 0, "outlines"
            rule, n_train, n_valid, n_test, warn = _step6_outline_split(
                list(counted.keys()), pin_split, valid_ratio, test_ratio)
            approximate = True
        if warn:
            warnings.append(warn)

        return jsonify({
            "ok": True, "name": name, "kind": kind, "path": path,
            "n_images": n_images, "n_labels": n_labels, "label_noun": label_noun,
            "n_classes": n_classes, "n_background": n_background,
            "pictures_sentence": _step6_pictures_sentence(
                kind, n_images, n_labels, label_noun, n_classes, n_background),
            "split": {
                "rule": rule,
                "sentence": _step6_split_sentence(rule, n_train, n_valid, n_test, approximate,
                                                  valid_ratio, test_ratio),
                "n_train": n_train, "n_valid": n_valid, "n_test": n_test,
                "approximate": approximate,
            },
            "warnings": warnings,
        })

    @app.route("/api/step/6/list_runs")
    def step6_list_runs():
        """List training runs that step 7/8 can pick from.

        Each row carries the run's own task ('seg' or 'detect') and metric
        family, and the champion rides with champion_run_dir, champion_task
        and champion_scope, so the Train panel can show only the runs whose
        weights suit the training task now selected. champion_task is ''
        when no champion is promoted or its folder is gone, because the task
        is read from the run folder.

        The champion here is the MODULE one (scope "module"): this list feeds
        the Inference run picker and the Train panel's continue-from-a-run
        group, and promotion promises exactly that those two follow the
        promoted run, whichever project or lab trained it. A run of another
        owner is not in this list, so no row is badged as the champion then,
        and champion_run_dir still names it.
        """
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        step6_dir = current_project["steps"].get("6", {}).get("dir", "")
        runs_dir = os.path.join(step6_dir, "runs")
        champion = _module_champion()
        champion_run_dir = champion.get("run_dir") if champion else None
        champion_task = ("" if not champion_run_dir or _run_folder_missing(champion_run_dir)
                         else _labruns.task_for_run_dir(champion_run_dir))
        if not os.path.isdir(runs_dir):
            return jsonify({"runs": [], "champion_run_dir": champion_run_dir,
                            "champion_task": champion_task,
                            "champion_scope": CHAMPION_SCOPE_MODULE})
        out = []
        for nm in sorted(os.listdir(runs_dir)):
            rdir = os.path.join(runs_dir, nm)
            if not os.path.isdir(rdir):
                continue
            weights_dir = os.path.join(rdir, "weights")
            best = os.path.join(weights_dir, "best.pt")
            last = os.path.join(weights_dir, "last.pt")
            has_best = os.path.isfile(best)
            has_last = os.path.isfile(last)
            # Pull a small subset of training args for the dropdown tooltip.
            args_yaml = os.path.join(rdir, "args.yaml")
            model = epochs = None
            if os.path.isfile(args_yaml):
                try:
                    import yaml as _y
                    with open(args_yaml) as f:
                        a = _y.safe_load(f) or {}
                    model = a.get("model")
                    epochs = a.get("epochs")
                except Exception:
                    pass
            task = _labruns.task_for_run_dir(rdir)
            out.append({
                "name": nm,
                "path": rdir,
                "has_best": has_best,
                "has_last": has_last,
                "model": model,
                "epochs": epochs,
                "task": task,
                "family": _labruns.metric_family_for_task(task),
                "is_champion": bool(champion_run_dir) and os.path.abspath(rdir) == os.path.abspath(champion_run_dir),
            })
        return jsonify({"runs": out, "runs_dir": runs_dir,
                        "champion_run_dir": champion_run_dir,
                        "champion_task": champion_task,
                        "champion_scope": CHAMPION_SCOPE_MODULE})

    @app.route("/api/step/6/all_runs")
    def step6_all_runs():
        """Every trained run the module can see, for the Step 6 runs card:
        the model labs under inprocess/_models/<lab>/runs and every project's
        step6_trainModel/runs, newest run folder first, each with its task,
        metric family, ledger scores, image size, split fingerprint, what its
        score was measured with, and whether it may be ranked against the
        module champion (inprocess/_models/champion.json, scope "module").
        Project-free: no open project is needed. Per the wave 2 contract."""
        root = _inprocess_root()
        champion = _champion_view(_module_champion(), CHAMPION_SCOPE_MODULE)
        champion_dir = champion["run_dir"] if champion else ""
        champion_measured = _measurement_of(
            _ledger_row_for_run(champion_dir) if champion_dir else None)
        rows = [_all_runs_row(run, champion_dir, champion_measured)
                for run in _labruns.list_all_runs(root)]
        rows.sort(key=lambda r: r.pop("_folder_time"), reverse=True)
        return jsonify({"ok": True, "runs": rows, "champion": champion,
                        "champion_scope": CHAMPION_SCOPE_MODULE,
                        "champion_measured_at": champion_measured})

    # Rounds ledger numeric columns (Task 9): cast on the way out of rounds.csv
    # so the frontend never has to parseFloat/parseInt csv strings itself.
    # imgsz and conf/iou say what a row was measured with; a row written
    # before those columns existed carries empty cells, which cast to None so
    # the Rounds table can tell "not recorded" from a real number.
    _ROUNDS_INT_COLS = ("round", "n_train", "n_valid", "n_test", "imgsz")
    _ROUNDS_FLOAT_COLS = ("map50_M", "map50_95_M", "recall_M", "precision_M",
                          "map50_B", "map50_95_B", "recall_B", "precision_B",
                          "conf", "iou")

    def _cast_round_row(row):
        out = dict(row)
        for k in _ROUNDS_INT_COLS:
            v = out.get(k)
            try:
                out[k] = int(v) if v not in (None, "") else None
            except (TypeError, ValueError):
                out[k] = None
        for k in _ROUNDS_FLOAT_COLS:
            v = out.get(k)
            try:
                out[k] = float(v) if v not in (None, "") else None
            except (TypeError, ValueError):
                out[k] = None
        return out

    @app.route("/api/step/7/rounds")
    def step7_rounds():
        """Rounds ledger for the active learning loop (Task 9): every row of
        {step6_dir}/rounds.csv (numeric columns cast) plus the two champion
        records, each labelled with its scope, so the Evaluate panel's Rounds
        table and the Trained runs card can never show two different
        champions for one window.

        champion is the MODULE record (champion_scope "module"): the promoted
        model, the same one the Trained runs card ranks against and the one
        Inference and the fine-tune recipe start from.
        gate_baseline is this project's own record (gate_baseline_scope
        "project"): what evaluate_run.py measured this ledger's gate columns
        against. It is null for a project that has never promoted a run of
        its own, which says nothing about whether a module champion exists.
        """
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        step6_dir = current_project["steps"].get("6", {}).get("dir", "")
        rounds_csv_path = os.path.join(step6_dir, "rounds.csv")
        rows = []
        if os.path.isfile(rounds_csv_path):
            with open(rounds_csv_path, newline="") as f:
                rows = [_cast_round_row(r) for r in csv.DictReader(f)]
        return jsonify({
            "rounds": rows,
            "champion": _module_champion(),
            "champion_scope": CHAMPION_SCOPE_MODULE,
            "gate_baseline": _project_champion(step6_dir),
            "gate_baseline_scope": CHAMPION_SCOPE_PROJECT,
        })

    @app.route("/api/step/7/promote", methods=["POST"])
    def step7_promote():
        """Promote a trained run to champion.

        Accepts any run under the allowed roots (the open project and
        inprocess/, which holds the model labs), so the Step 6 runs card can
        promote a run from another project or a lab. The run needs
        weights/best.pt and a row in the ledger beside it
        (labruns.rounds_dir_for_run): a run nobody has evaluated has no
        basis for being the best.

        A run whose own ledger row records one of GATE_BLOCKING_VERDICTS is
        refused with HTTP_CONFLICT until the caller sends confirm_gate: true,
        and the refusal states every gate's verdict. "fail" says the score
        went backwards against the baseline, "flag" names classes or a recall
        that dropped, and "incomparable" says the two rounds were never a
        comparison; promoting over any of them is a choice, so it is made
        explicitly rather than by a press that says nothing.

        Writes the family-aware record
        {run_dir, promoted_at, metric_family, map50_95_M or map50_95_B, note,
        gate} twice: beside that ledger (scope "project", where
        evaluate_run.py reads its gate baseline), and at
        inprocess/_models/champion.json (scope "module", the promoted model
        every reader shows). Answers {"ok", "champion", "scopes_written",
        "gate"}, the champion as the runs card shows it (wave 2 contract).
        """
        body = request.get_json(force=True, silent=True) or {}
        run_dir, err, status = _validate_client_dir(body.get("run_dir"), "run_dir")
        if err:
            return jsonify({"error": err}), status
        if not os.path.isfile(os.path.join(run_dir, WEIGHTS_REL)):
            return jsonify({"error": (f"Run has no {WEIGHTS_REL} yet, so it cannot be "
                                      f"promoted: {run_dir}")}), HTTP_BAD_REQUEST
        run_name = os.path.basename(os.path.normpath(run_dir))
        row = _ledger_row_for_run(run_dir)
        if row is None:
            return jsonify({"error": (
                f"No evaluation on record for {run_name}. Run {step_ref(7)} "
                "on this run before promoting it."
            )}), HTTP_BAD_REQUEST

        gate = _gate_verdicts(row)
        if _gate_needs_confirmation(gate) and body.get("confirm_gate") is not True:
            return jsonify({
                "error": (f"The evaluation of {run_name} did not pass its own gates "
                          f"({_gate_sentence(gate)}). Promoting it makes it the default "
                          "weights for Inference and fine-tuning anyway. Send "
                          "confirm_gate: true to promote it over that verdict."),
                "needs_confirmation": True,
                "gate": gate,
            }), HTTP_CONFLICT

        family = _family_for_run(run_dir, row)
        score_key = _family_column(family, "map50_95")
        record = {
            "run_dir": run_dir,
            "promoted_at": _ast_now(),
            "metric_family": family,
            score_key: _row_float(row, score_key),
            "note": str(body.get("note") or ""),
            "gate": gate,
        }
        ledger_path = os.path.join(_labruns.rounds_dir_for_run(run_dir), _labruns.CHAMPION_FILE_NAME)
        module_path = _labruns.champion_path(_inprocess_root())
        try:
            _write_json_atomic(ledger_path, record)
            _write_json_atomic(module_path, record)
        except OSError as exc:
            return jsonify({"error": str(exc)}), HTTP_SERVER_ERROR
        vic.note(f"[orch] promoted {run_name} to champion ({family} mAP50-95={record[score_key]}, "
                 f"{_gate_sentence(gate)})")
        return jsonify({
            "ok": True,
            "champion": _champion_view(_labruns.read_champion_file(module_path),
                                       CHAMPION_SCOPE_MODULE),
            "scopes_written": [CHAMPION_SCOPE_PROJECT, CHAMPION_SCOPE_MODULE],
            "gate": gate,
        })

    @app.route("/api/step/7/report")
    def step7_report():
        """The evaluation report for one run, per the wave 2 contract:
        report_md, the metrics keys from metrics.json (METRICS_KEYS, which
        carry the species names the evaluation resolved), the previews with
        image urls and the counts the evaluator recorded, one url or null
        per figure, and pdf_url when report.pdf exists. With ?run_dir= the
        newest evaluation folder for that run wins (its own project's
        step7_evaluateModel/eval_*, the open project's, and
        inprocess/_models/_eval/<run>_*); without it, the project's saved
        Step 7 pointer, so a project evaluated before today still shows its
        report. 404 with the reason and "ready": false when no evaluation
        exists yet, which the report card shows as its empty state."""
        run_dir, out_dir, err, status = _resolve_evaluation(request.args.get("run_dir"))
        if err:
            body = {"error": err}
            if err.startswith(NO_EVALUATION_PREFIX):
                body["ready"] = False
            return jsonify(body), status
        return jsonify(_report_payload(run_dir, out_dir))

    @app.route("/api/step/7/pdf")
    def step7_pdf():
        """The evaluation PDF for a run as an attachment named
        <run>_report.pdf, resolved like /api/step/7/report; 404 JSON when
        that evaluation wrote no PDF."""
        run_dir, out_dir, err, status = _resolve_evaluation(request.args.get("run_dir"))
        if err:
            return jsonify({"error": err}), status
        pdf = os.path.join(out_dir, REPORT_PDF_NAME)
        if not os.path.isfile(pdf):
            return jsonify({"error": f"no {REPORT_PDF_NAME} in {out_dir}: that evaluation wrote no PDF"}), HTTP_NOT_FOUND
        run_name = os.path.basename(os.path.normpath(run_dir))
        return send_file(pdf, as_attachment=True, download_name=f"{run_name}_report.pdf")

    @app.route("/api/step/7/copy_pdf", methods=["POST"])
    def step7_copy_pdf():
        """Copy the evaluation PDF for a run (body run_dir, resolved like
        /api/step/7/report) into the folder the body names as dst_dir, as
        <run>_report.pdf so copies of several runs can share a folder."""
        data = request.get_json(force=True, silent=True) or {}
        dst_dir, dst_err = _real_destination(data.get("dst_dir"))
        if dst_err:
            return jsonify({"error": f"dst_dir (the folder to copy the PDF into): {dst_err}"}), HTTP_BAD_REQUEST
        roots = _export_roots()
        if not _contained_in_any(dst_dir, roots):
            return jsonify({"error": "an export folder must sit under an allowed root: the module folder, the "
                                     f"VICARIUS drive, the home folder, or a share listed in {EXPORT_ROOTS_ENV}"}), HTTP_FORBIDDEN
        run_dir, out_dir, err, status = _resolve_evaluation(data.get("run_dir"))
        if err:
            return jsonify({"error": err}), status
        pdf = os.path.join(out_dir, REPORT_PDF_NAME)
        if not os.path.isfile(pdf):
            return jsonify({"error": f"no {REPORT_PDF_NAME} in {out_dir}: that evaluation wrote no PDF"}), HTTP_NOT_FOUND
        run_name = os.path.basename(os.path.normpath(run_dir))
        dst = os.path.join(dst_dir, f"{run_name}_report.pdf")
        if os.path.exists(dst_dir) and not os.path.isdir(dst_dir):
            return jsonify({"error": "the export destination names a file, not a folder"}), HTTP_CONFLICT
        try:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(pdf, dst)
        except OSError as exc:
            return jsonify({"error": f"cannot copy {pdf} to {dst}: {exc}"}), HTTP_SERVER_ERROR
        return jsonify({"ok": True, "dst": dst})

    @app.route("/api/step/7/image")
    def step7_image():
        """One file from a run's evaluation folder (resolved like
        /api/step/7/report): ?rel= names it relative to that folder, may not
        be absolute or hold a '..' segment, and anything resolving outside
        the folder is refused with 403. The one exception is the run's own
        training curve: rel "run/results.png" reads results.png from the
        validated run folder, and no other file under that prefix."""
        rel, err, status = _validate_rel(request.args.get("rel"))
        if err:
            return jsonify({"error": err}), status
        run_dir, out_dir, err, status = _resolve_evaluation(request.args.get("run_dir"))
        if err:
            return jsonify({"error": err}), status
        if rel.startswith(RUN_FIGURE_REL_PREFIX):
            name = rel[len(RUN_FIGURE_REL_PREFIX):]
            allowed = [n for _k, n in RUN_FIGURES]
            if name not in allowed:
                return jsonify({"error": (f"only {', '.join(allowed)} may be read from the run "
                                          f"folder: {rel}")}), HTTP_FORBIDDEN
            full, err, status = _file_inside(run_dir, name)
        else:
            full, err, status = _file_inside(out_dir, rel)
        if err:
            return jsonify({"error": err}), status
        return send_file(full)

    @app.route("/api/step/8/runs")
    def step8_runs():
        """Every inference run the gallery can show, newest first: the open
        project's step8_inference/* and inprocess/_models/_inference/*, each
        with the counts read from its manifest.json (wave 2 contract).
        Project-free: without a project only the lab runs are listed."""
        return jsonify({"ok": True, "runs": _list_inference_runs()})

    @app.route("/api/step/8/manifest")
    def step8_manifest():
        """The manifest.json of one inference run, plus ok, image_base (the
        /api/step/8/image url prefix the overlay's relative path is appended
        to) and verdicts (filename to yes, no or skip, from that run's
        verifications.csv, last row per filename). ?out_dir= is checked
        against the allowed roots; without it the open project's saved
        Step 8 pointer is read, as before."""
        out_dir, err, status = _resolve_inference_dir(request.args.get("out_dir"))
        if err:
            return jsonify({"error": err}), status
        manifest_path = os.path.join(out_dir, MANIFEST_JSON_NAME)
        if not os.path.isfile(manifest_path):
            return jsonify({"error": (f"no {MANIFEST_JSON_NAME} in {out_dir}: the inference run has "
                                      "not finished, or wrote nothing")}), HTTP_NOT_FOUND
        data, err = _read_json_object(manifest_path)
        if err:
            return jsonify({"error": err}), HTTP_SERVER_ERROR
        data["ok"] = True
        data["image_base"] = f"{STEP8_IMAGE_ROUTE}?{urlencode({'out_dir': out_dir})}&rel="
        data["verdicts"] = _read_verdicts(out_dir)
        return jsonify(data)

    @app.route("/api/step/8/image")
    def step8_image():
        """One image of an inference run. ?out_dir= names the run (checked
        against the allowed roots; the project's saved pointer without it).
        ?rel= names an overlay relative to that folder, confined to it with
        commonpath so a '..' segment, a symbolic link out, or a sibling
        folder whose name extends this one is refused (403). ?raw=1 with
        ?filename= serves the source frame instead: the filename is looked
        up in the run's own manifest and the recorded path served, 404 for
        a filename the manifest does not list, 403 when the recorded path
        is not a readable file. A client-supplied path is never resolved."""
        out_dir, err, status = _resolve_inference_dir(request.args.get("out_dir"))
        if err:
            return jsonify({"error": err}), status
        if str(request.args.get("raw") or "") == "1":
            filename = str(request.args.get("filename") or "").strip()
            if not filename:
                return jsonify({"error": "filename (the tile's file name in the manifest) is required with raw=1"}), HTTP_BAD_REQUEST
            full, err, status = _raw_frame_path(out_dir, filename)
        else:
            rel, err, status = _validate_rel(request.args.get("rel"))
            if not err:
                full, err, status = _file_inside(out_dir, rel)
        if err:
            return jsonify({"error": err}), status
        return send_file(full)

    @app.route("/api/step/8/verify", methods=["POST"])
    def step8_verify():
        """Record a verdict (yes, no or skip) on one inference tile in
        <out_dir>/verifications.csv, stamped in AST with the offset. The body's
        out_dir names the run (checked against the allowed roots) so a
        verdict on an older run reaches that run's file; without it the
        project's saved pointer is used."""
        data = request.get_json(force=True, silent=True) or {}
        filename = str(data.get("filename") or "").strip()
        verdict = str(data.get("verdict") or "").strip()
        notes = str(data.get("notes") or "")
        if not filename:
            return jsonify({"error": "filename (the tile's file name) is required"}), HTTP_BAD_REQUEST
        if verdict not in VERDICTS:
            return jsonify({"error": f"verdict must be one of {', '.join(VERDICTS)} (got {verdict!r})"}), HTTP_BAD_REQUEST
        out_dir, err, status = _resolve_inference_dir(data.get("out_dir"))
        if err:
            return jsonify({"error": err}), status
        log_path = os.path.join(out_dir, VERIFICATIONS_CSV_NAME)
        is_new = not os.path.isfile(log_path)
        try:
            with open(log_path, "a", newline="") as f:
                w = csv.writer(f)
                if is_new:
                    w.writerow(["filename", "verdict", "notes", "recorded_at"])
                w.writerow([filename, verdict, notes, _ast_now()])
        except OSError as exc:
            return jsonify({"error": f"cannot write {log_path}: {exc}"}), HTTP_SERVER_ERROR
        return jsonify({"ok": True, "log_path": log_path})

    # ------------------------------------------------------------------
    # SAM3 driver status (orchestrator-driven)
    # ------------------------------------------------------------------
    @app.route("/api/step/5/sam3_status")
    def step5_sam3_status():
        # These phases all claim a live step-5 sub-app process. The driver
        # thread reports its own failures, but it cannot speak for the sub-app
        # dying after review_ready (or mid-drive between its HTTP polls).
        # waiting_for_sub_app is included: the process exists from launch, and
        # dying DURING model load was the original 2026-08-13 incident window.
        return jsonify(_reconcile_liveness(
            sam3_lock, sam3_status,
            ("review_ready", "segmenting", "configuring",
             "waiting_for_sub_app"), 5,
            "Step 5 segmentation sub-app"))

    @app.route("/api/step/5/kick", methods=["POST"])
    def step5_kick():
        """Head-start SAM3. Called by step 4 sub-app after each batch export
        (when auto_start_sam3 is on) and by any other orchestrator-side flow
        that wants to nudge segmentation forward.

        - If step 5 sub-app is not running: launch it + start the SAM3 driver.
        - If it's running and the driver is idle/done: re-run configure+process
          so newly exported frames get picked up.
        - If the driver is still working: no-op (the sub-app's queue is rebuilt
          on the next cycle; new frames will be picked up then).
        """
        global sam3_thread
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400

        s5 = current_project["steps"].get("5")
        if not s5:
            return jsonify({"error": "Step 5 not in project"}), 400

        # Case 1: sub-app not running — do a full launch (same as Start button).
        if not runner.is_running(5):
            cfg = s5["config"]
            step_dir = s5["dir"]
            os.makedirs(step_dir, exist_ok=True)
            s5["status"] = "running"
            s5["started_at"] = datetime.now().isoformat()
            pm.save_project(current_project)
            try:
                result = _run_step5(5, cfg, step_dir)
                if "error" in result:
                    s5["status"] = "error"
                    pm.save_project(current_project)
                    return jsonify(result), 500
                return jsonify({"launched": True, **result})
            except Exception as e:
                s5["status"] = "error"
                pm.save_project(current_project)
                return jsonify({"error": str(e)}), 500

        # Case 2: sub-app running, driver thread still active — nothing to do.
        if sam3_thread is not None and sam3_thread.is_alive():
            return jsonify({"status": "already_driving"})

        # Case 3: sub-app running, driver idle — re-kick with fresh configure.
        port = runner.ports.get(5)
        if not port:
            return jsonify({"error": "No port recorded for running step 5"}), 500
        cfg = s5["config"]
        step_dir = s5["dir"]
        input_dir = cfg.get("input_dir") or current_project["steps"].get("4", {}).get("dir", "")
        categories = cfg.get("categories") or ["Target species only"]
        batch_size = int(cfg.get("review_batch_size") or 10)
        _sam3_set(phase="re_kicking", running=True, error=None,
                  message="New frames exported; re-configuring SAM3 queue...")
        sam3_thread = threading.Thread(
            target=_sam3_drive,
            args=(port, input_dir, step_dir, categories, batch_size),
            daemon=True,
        )
        sam3_thread.start()
        return jsonify({"status": "re_kicked"})

    # ------------------------------------------------------------------
    # step4test - combined annotator (place + SAM3 segment). Non-chain tile,
    # launched on its own port; NOT a STEP_KEYS project step. The int-keyed
    # /api/step/<int:step>/* routes 404 a string segment, so these are
    # dedicated string routes.
    # ------------------------------------------------------------------
    @app.route("/api/step/step4test/run", methods=["POST"])
    def step4test_run():
        global step4test_cfg
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        refused = _refuse_for_composite()
        if refused is not None:
            return refused
        # Capture the #panel-step4test settings (Combined Step-4+5 config). The
        # body may be empty (legacy callers) - then the launch inherits Step-5.
        body = request.get_json(silent=True) or {}
        if isinstance(body.get("config"), dict):
            step4test_cfg = body["config"]
        try:
            result = _run_step4test(force_reroute=False)
            if isinstance(result, dict) and "error" in result:
                # run_flask_stage reports "Step <key> is already running", and the
                # ANNOTATOR and the ROUTING sub-app are different keys:
                #   step4test       -> the annotator process is up (possibly still
                #                      loading SAM3). Not a failure: hand the UI its
                #                      port and let it wait for health.
                #   step4test_route -> a ROUTING pass is in flight. That is NOT a
                #                      ready UI, and runner.ports["step4test"] here
                #                      is a STALE port from an earlier launch whose
                #                      process is long dead.
                # Matching the bare substring "already running" conflated the two and
                # answered ui_ready on that stale port, so the browser polled a dead
                # port for its whole deadline and reported "Combined Annotator did not
                # become ready". _run_step4test's in-flight guard now returns a routing
                # phase before this branch is ever reached; the narrowed match keeps a
                # routing-key error from ever being read as a live UI again.
                err = str(result.get("error", "")).lower()
                if "step4test is already running" in err:
                    # The error itself proves the annotator process is alive
                    # (run_flask_stage only raises it when poll() is None), so this
                    # port belongs to a running process - no health gate needed, and
                    # a mid-SAM3-load annotator still gets waited on by the UI.
                    port = runner.ports.get("step4test") or step4test_state.get("port")
                    if port:
                        _push_reviewer_after_health(port, _reviewer_initials(), runner.health_check)
                        return jsonify({"success": True, "already_running": True,
                                        "phase": "ui_ready", "ui_port": port})
                return jsonify(result), 500
            # Routing kicked (poll route_status) or instant reuse (ui_ready).
            with route_lock:
                phase = route_status.get("phase")
                ui_port = route_status.get("ui_port")
            payload = {"success": True, "phase": phase}
            if phase == "ui_ready" and ui_port:
                payload["ui_port"] = ui_port
            return jsonify(payload)
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    @app.route("/api/step/step4test/route_status")
    def step4test_route_status():
        # The routing phases are owned by the driver THREAD; if that thread
        # died without writing a phase (hard crash), the status would stick at
        # a routing phase forever with nothing checking it. route_thread.ident
        # filters the created-but-not-yet-started window.
        with route_lock:
            _phase = route_status.get("phase")
        if (_phase in ("routing", "routing_configure", "routing_ocr",
                       "exporting")
                and route_thread is not None
                and route_thread.ident is not None
                and not route_thread.is_alive()):
            _route_set(phase="error", ui_port=None, died=True,
                       error="Routing driver thread died mid-pass.",
                       message="Routing stopped unexpectedly. Reopen to "
                               "resume where it left off.")
        # ui_ready is the only phase claiming a live annotator process; the
        # setter stamps which mode's process owns it.
        return jsonify(_reconcile_liveness(
            route_lock, route_status, ("ui_ready",), "step4test",
            "Combined annotator"))

    @app.route("/api/step/step4test/cancel", methods=["POST"])
    def step4test_cancel():
        """Cancel an in-flight routing pass. Called when the operator closes the
        launch window, and by Stop.

        The UI is the only way this pipeline is ever launched, so a closed window
        must mean the work stops - not a routing sub-app left holding the GPU and
        later killing a live annotator to launch one nobody asked for.

        Cheap by construction: placePoints flushes detections every 50 frames and
        skips frames it already has, so reopening resumes rather than restarting.
        """
        route_cancel.set()
        # Kill the routing sub-app now rather than waiting for the driver thread to
        # notice: it is the thing holding the GPU.
        runner.kill("step4test_route")
        with route_lock:
            phase = route_status.get("phase")
        # Only rewrite the phase if a pass was actually underway - never clobber a
        # ui_ready/idle state.
        if phase in ("launching", "routing", "routing_configure", "routing_ocr",
                     "exporting", "route_ready"):
            _route_set(phase="cancelled", ui_port=None, error=None,
                       message="Routing cancelled. Reopening resumes where it stopped.")
        return jsonify({"success": True, "cancelled": True})

    @app.route("/api/step/step4test/status")
    def step4test_status():
        # runner keys by hashable step id; we use the string "step4test".
        status = runner.poll_status("step4test")
        port = runner.ports.get("step4test") or step4test_state.get("port")
        if port:
            status["healthy"] = runner.health_check(port)
            status["port"] = port
        return jsonify(status)

    @app.route("/api/step/step4test/log")
    def step4test_log():
        offset = request.args.get("offset", 0, type=int)
        lines, new_offset, dropped = runner.get_log("step4test", offset)
        return jsonify({"lines": lines, "offset": new_offset, "dropped": dropped})

    @app.route("/api/step/step4test/stop", methods=["POST"])
    def step4test_stop():
        """Stop everything this tile owns: the annotator AND any routing pass.

        Stopping used to kill only the annotator, leaving a routing pass running
        invisibly - which would then kill a later annotator to launch its own.
        Stop means stop.
        """
        route_cancel.set()
        runner.kill("step4test_route")
        # _live_annotator_port confirms runner.is_running("step4test") first,
        # so a dead key's stale port (since reused by an unrelated process)
        # is never flushed.
        port = _live_annotator_port("step4test")
        if port:
            ok, reason = _flush_annotator(port)
            if not ok:
                vic.note(f"[orch] annotator flush failed on port {port} before stop: {reason}")
        runner.kill("step4test")
        with route_lock:
            phase = route_status.get("phase")
        if phase in ("launching", "routing", "routing_configure", "routing_ocr",
                     "exporting", "route_ready"):
            _route_set(phase="cancelled", ui_port=None, error=None,
                       message="Stopped. Reopening resumes where routing stopped.")
        else:
            # A deliberate Stop must not read as a crash: the liveness check
            # flips a lingering ui_ready over a dead process to a red
            # "process exited" error, so rewrite the phase first - but only
            # when step4test owns it (Edit Masks / Refine keep running).
            _route_mark_stopped_if_owner("step4test", "Annotator stopped.")
        step4test_state.update(running=False, project_id=None, input_dir=None)
        return jsonify({"success": True})

    @app.route("/api/step/step4test/folder", methods=["POST"])
    def step4test_folder():
        """Open the combined annotator's output dir in the file manager.

        Same guardrail as /api/fs/open: only opens paths inside the current
        project or the repo.
        """
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        out_dir = step4test_state.get("out_dir")
        # Trust the remembered launch dir only when it belongs to THIS project;
        # otherwise (not launched yet, or launched for another project) use the
        # canonical export dir, which is always under project_dir.
        if not out_dir or step4test_state.get("project_id") != current_project.get("id"):
            out_dir = _step4test_export_dir()
        return _folder_response(out_dir, current_project["project_dir"])

    @app.route("/api/step/step4test/reset", methods=["POST"])
    def step4test_reset():
        """Reset the combined annotator (4.test): stop its runners and delete the
        whole step4test_combinedAnnotate dir (routed_input + all_images/all_labels/
        segmentations) so the next Open re-routes from Step 3 and starts fresh.

        Same project-dir guardrail as /folder: only a path inside the current
        project is ever removed.
        """
        global step4test_cfg
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        # A training project's step4test dir is a SYMLINK into the shared
        # composite. Refuse before anything is killed or removed: rmtree would
        # refuse the link itself, but the reset has no meaning here anyway.
        refused = _refuse_for_composite()
        if refused is not None:
            return refused
        # Kill both runners (the headless routing pass + the combined UI) so
        # nothing is mid-write while we remove the tree. Flush the combined UI
        # first (save first, D1), moot for this route's own rmtree below, but
        # every other stop of a live annotator follows the same rule.
        # _live_annotator_port confirms runner.is_running("step4test") first,
        # so a dead key's stale port is never flushed.
        port = _live_annotator_port("step4test")
        if port:
            ok, reason = _flush_annotator(port)
            if not ok:
                vic.note(f"[orch] annotator flush failed on port {port} before reset: {reason}")
        runner.kill("step4test")
        runner.kill("step4test_route")
        # Both sides resolved: an export dir that is a symlink OUT of the
        # project fails this check on its real path instead of passing on a
        # path that only looks contained. Structural, so it holds even if the
        # refusal above were ever removed.
        project_dir = os.path.realpath(current_project["project_dir"])
        out_dir = os.path.realpath(_step4test_export_dir())
        # Guardrail: the target must live strictly under the project dir.
        if not out_dir.startswith(project_dir + os.sep):
            return jsonify({"error": "path outside project"}), 403
        try:
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        # Reset transient state so the panel reflects a clean slate.
        step4test_state.update(running=False, port=None,
                               started_at=None, out_dir=None,
                               project_id=None, input_dir=None)
        step4test_cfg = {}
        _route_set(phase="idle", processed=0, total=0, ui_port=None, error=None,
                   warnings=[], dropped=[], message="")
        vic.note("[orch] step4test reset - removed step4test_combinedAnnotate")
        return jsonify({"success": True, "state": _state_payload(current_project)})

    # ------------------------------------------------------------------
    # ---- step4loop (Refine loop) string routes ----
    #
    # Active-learning review loop: Step 8 inference writes predictions.json;
    # this tile seeds those detections into the SAME step4test export
    # (segmentations.json) as pending model masks via seed_from_predictions.py,
    # then opens the combined annotator in resume mode so the reviewer accepts/
    # rejects/edits them like any other mask. Non-chain string routes for the
    # same reason step4test's are: /api/step/<int:step>/* can't hold "step4loop".
    # ------------------------------------------------------------------

    def _loop_rounds_dir():
        return os.path.join(_step4test_export_dir(), "loop_rounds")

    @app.route("/api/step/step4loop/inference_runs")
    def step4loop_inference_runs():
        """List step8_inference/*/ dirs that contain a predictions.json, newest
        first, so the Refine panel can offer a run to seed from."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        step8_dir = current_project["steps"].get("8", {}).get("dir", "")
        runs = []
        if step8_dir and os.path.isdir(step8_dir):
            for nm in os.listdir(step8_dir):
                rdir = os.path.join(step8_dir, nm)
                pred_path = os.path.join(rdir, "predictions.json")
                if not os.path.isfile(pred_path):
                    continue
                n_items = 0
                generated_at = None
                try:
                    with open(pred_path) as f:
                        data = json.load(f)
                    n_items = len(data.get("items") or [])
                    generated_at = data.get("generated_at")
                except Exception:
                    pass
                runs.append({
                    "dir": rdir,
                    "name": nm,
                    "n_items": n_items,
                    "generated_at": generated_at,
                    "_mtime": os.path.getmtime(pred_path),
                })
        runs.sort(key=lambda r: r["_mtime"], reverse=True)
        for r in runs:
            r.pop("_mtime", None)
        return jsonify({"runs": runs})

    @app.route("/api/step/step4loop/run", methods=["POST"])
    def step4loop_run():
        """Seed a chosen inference run's predictions.json into the step4test
        export as pending review masks, then open the combined annotator in
        resume mode against exactly those (plus any prior) segmentations.

        Fail-closed on the frozen split (D8): the seeder requires
        {step6_dir}/dataset/split_manifest.json so it never seeds a frame the
        training split pinned to valid/test. Train a model (Step 6) before
        this route will do anything.
        """
        global step4loop_cfg
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        refused = _refuse_for_composite()
        if refused is not None:
            return refused

        body = request.get_json(silent=True) or {}
        cfg = body.get("config") if isinstance(body.get("config"), dict) else {}
        step4loop_cfg = cfg

        predictions_dir = (cfg.get("predictions_dir") or "").strip()
        if not predictions_dir or not os.path.isdir(predictions_dir):
            return jsonify({"success": False,
                            "error": f"predictions_dir not found: {predictions_dir}"}), 400
        predictions_dir = os.path.abspath(predictions_dir)
        step8_dir = os.path.abspath(current_project["steps"].get("8", {}).get("dir", "") or "")
        if step8_dir and not (predictions_dir == step8_dir
                              or predictions_dir.startswith(step8_dir + os.sep)):
            return jsonify({"success": False,
                            "error": f"predictions_dir sits outside this project's "
                                     f"{step_ref(8)} output"}), 400
        predictions_path = os.path.join(predictions_dir, "predictions.json")
        if not os.path.isfile(predictions_path):
            return jsonify({"success": False,
                            "error": f"No predictions.json in {predictions_dir}. Re-run {step_ref(8)}."}), 400

        # Fail-closed (D8): the seeder must never seed a frame the training
        # split already pinned to valid/test, so the split has to exist first.
        step6_dir = current_project["steps"].get("6", {}).get("dir", "")
        split_manifest = os.path.join(step6_dir, "dataset", "split_manifest.json") if step6_dir else ""
        if not split_manifest or not os.path.isfile(split_manifest):
            return jsonify({"success": False, "error": (
                f"No frozen split yet: run {step_ref(6)} before Refine so "
                "the holdout is pinned."
            )}), 400

        codes_csv = _resolve_master_codes()
        py = PYTHON_PATHS.get(5, "python3")
        if not os.path.isfile(py):
            py = "python3"
        seeder = os.path.join(REPO_DIR, "scripts", "TCRMPclip_combinedAnnotate",
                              "src", "seed_from_predictions.py")
        if not os.path.isfile(seeder):
            return jsonify({"success": False, "error": f"seeder script not found: {seeder}"}), 500

        export_dir = _step4test_export_dir()
        os.makedirs(export_dir, exist_ok=True)

        conf_min = cfg.get("conf_min", 0.25)
        max_frames = cfg.get("max_frames", 0)
        skip_empty = bool(cfg.get("skip_empty", False))

        cmd = [
            py, seeder,
            "--predictions", predictions_path,
            "--export_dir", export_dir,
            "--codes_csv", codes_csv,
            "--split_manifest", split_manifest,
            "--conf_min", str(conf_min),
            "--max_frames", str(max_frames),
        ]
        if skip_empty:
            cmd.append("--skip_empty")
        lores_manifest = os.path.join(REPO_DIR, "supporting_data", "TCRMP_clip_lores",
                                      "lores_manifest.csv")
        if os.path.isfile(lores_manifest):
            cmd += ["--lores_manifest", lores_manifest]

        # Save first, THEN stop, before the seeder writes (2026-09-27 safety
        # net, D1). The seeder below rewrites segmentations.json directly on
        # disk; flush_all() in the annotator is a blind dump of its in-memory
        # segmentations dict (never a merge), so if that process were still
        # alive when the seeder finished, any later save from it, the generic
        # flush _launch_step4test_ui's _kill_all_annotators performs right
        # before its own kill, or the process's own save-on-quit handler,
        # would overwrite the seeder's freshly seeded masks with the stale
        # pre-seed content. Fully stopping it here, before the seeder ever
        # runs, means nothing is left alive to re-save that stale copy;
        # _kill_all_annotators still runs its normal sweep afterward for
        # zombies, but finds nothing tracked left to flush twice.
        # _live_annotator_port confirms runner.is_running(key) first, so a
        # dead key's stale port is never flushed.
        for _annotator_key in ("step4test", "step4loop"):
            _port = _live_annotator_port(_annotator_key)
            if _port:
                _ok, _reason = _flush_annotator(_port)
                if not _ok:
                    vic.note(f"[orch] annotator flush failed on port {_port} "
                             f"before seeding: {_reason}")
            try:
                runner.kill(_annotator_key)
            except Exception as _exc:
                vic.note(f"[orch] runner.kill({_annotator_key!r}) failed "
                         f"before seeding: {_exc}")

        import subprocess as _sp
        try:
            proc = _sp.run(cmd, capture_output=True, text=True, timeout=300)
        except _sp.TimeoutExpired:
            return jsonify({"success": False, "error": "Seeder timed out after 300s"}), 500
        except Exception as exc:
            return jsonify({"success": False, "error": f"Seeder failed to launch: {exc}"}), 500

        if proc.returncode != 0:
            return jsonify({"success": False, "error": (
                f"Seeder exited {proc.returncode}: "
                f"{(proc.stderr or proc.stdout or '').strip()[-2000:]}"
            )}), 500

        # The seeder prints exactly one JSON line (its summary) to stdout.
        summary = None
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                summary = json.loads(line)
            except Exception:
                continue
            break
        if summary is None:
            return jsonify({"success": False, "error": (
                f"Seeder produced no parseable summary. stdout tail: {proc.stdout[-2000:]}"
            )}), 500

        vic.note(
            f"[orch] step4loop seeded frames={summary.get('seeded_frames')} "
            f"masks={summary.get('seeded_masks')} "
            f"skipped_holdout={summary.get('skipped_holdout')} "
            f"skipped_existing={summary.get('skipped_existing')} "
            f"skipped_below_conf={summary.get('skipped_below_conf')}"
        )

        if not summary.get("seeded_frames"):
            return jsonify({"success": False, "seeded": summary, "error": (
                "Nothing new to review (all frames already in the labelset, or "
                "all remaining frames are on a held-out transect)."
            )}), 400

        # Preflight for the resume-mode launch: /api/resume reads only
        # segmentations.json (never sam_click_prompts.json), so the only thing
        # to assert here is that the seeder actually wrote it.
        seg_path = os.path.join(export_dir, "segmentations", "segmentations.json")
        if not os.path.isfile(seg_path) or os.path.getsize(seg_path) == 0:
            return jsonify({"success": False, "error": (
                f"Seeder reported {summary.get('seeded_frames')} seeded frames but "
                f"{seg_path} is missing or empty."
            )}), 500

        try:
            result = _launch_step4test_ui(mode="step4loop", cfg=cfg)
        except Exception as e:
            return jsonify({"success": False, "error": str(e),
                            "traceback": traceback.format_exc()}), 500
        if isinstance(result, dict) and result.get("error"):
            if "already running" in str(result.get("error", "")).lower():
                port = runner.ports.get("step4loop") or step4loop_state.get("port")
                _push_reviewer_after_health(port, _reviewer_initials(), runner.health_check)
                return jsonify({"success": True, "seeded": summary, "already_running": True,
                                "phase": "ui_ready", "ui_port": port})
            return jsonify({"success": False, "error": result["error"]}), 500

        port = runner.ports.get("step4loop") or step4loop_state.get("port")
        return jsonify({"success": True, "seeded": summary,
                        "phase": "ui_ready", "ui_port": port})

    @app.route("/api/step/step4loop/status")
    def step4loop_status():
        status = runner.poll_status("step4loop")
        port = runner.ports.get("step4loop") or step4loop_state.get("port")
        if port:
            status["healthy"] = runner.health_check(port)
            status["port"] = port
        return jsonify(status)

    @app.route("/api/step/step4loop/log")
    def step4loop_log():
        offset = request.args.get("offset", 0, type=int)
        lines, new_offset, dropped = runner.get_log("step4loop", offset)
        return jsonify({"lines": lines, "offset": new_offset, "dropped": dropped})

    @app.route("/api/step/step4loop/stop", methods=["POST"])
    def step4loop_stop():
        runner.kill("step4loop")
        _route_mark_stopped_if_owner("step4loop", "Refine annotator stopped.")
        step4loop_state.update(running=False, project_id=None, input_dir=None)
        return jsonify({"success": True})

    @app.route("/api/step/step4loop/folder", methods=["POST"])
    def step4loop_folder():
        """Open loop_rounds/ (or the export dir if it doesn't exist yet) in the
        file manager. Same guardrail as step4test's /folder."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        path = os.path.abspath(_loop_rounds_dir())
        if not os.path.isdir(path):
            path = os.path.abspath(_step4test_export_dir())
        return _folder_response(path, current_project["project_dir"])

    @app.route("/api/step/step4loop/reset", methods=["POST"])
    def step4loop_reset():
        """Surgically undo unreviewed seeding, never rmtree.

        For every loop_rounds/round_*.json manifest: for each seeded frame that
        is still unexported, drop the PENDING model-sourced masks this round
        added (matched by id); if that empties the frame's mask list AND it was
        never reviewed, remove the seg entry entirely (plus its routed_input/
        raw/ symlink, only if it really is a symlink - never touch a real
        file). Frames already exported, or carrying any accepted/manual mask,
        are left completely alone and reported back as `kept` so the user can
        see what reset intentionally did not touch. Round manifests that were
        fully consumed by this pass are deleted; segmentations.json is written
        atomically (tmp + os.replace).
        """
        global step4loop_cfg
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        # A training project's export dir is a symlink into the SHARED
        # composite, so this route must never run there: it deletes masks and
        # rewrites segmentations.json. Refused before anything is killed.
        refused = _refuse_for_composite()
        if refused is not None:
            return refused
        # Save first (D1), before the kill: this route reads and rewrites
        # segmentations.json itself right below, so the live process must be
        # both flushed AND stopped before that read, or its later exit would
        # overwrite what this route is about to write. _live_annotator_port
        # confirms runner.is_running("step4loop") first, so a dead key's
        # stale port is never flushed.
        port = _live_annotator_port("step4loop")
        if port:
            ok, reason = _flush_annotator(port)
            if not ok:
                vic.note(f"[orch] annotator flush failed on port {port} before reset: {reason}")
        runner.kill("step4loop")
        _route_mark_stopped_if_owner("step4loop", "Refine annotator stopped.")

        # Containment is structural, not accidental: resolve BOTH sides before
        # comparing, so a symlinked export can never satisfy the check by
        # sitting at a path that merely looks like it is inside the project.
        export_dir = os.path.realpath(_step4test_export_dir())
        project_dir = os.path.realpath(current_project["project_dir"])
        if not export_dir.startswith(project_dir + os.sep):
            return jsonify({"error": "path outside project"}), 403

        rounds_dir = os.path.join(export_dir, "loop_rounds")
        seg_path = os.path.join(export_dir, "segmentations", "segmentations.json")

        dropped_masks = 0
        removed_frames = []
        kept_frames = []
        consumed_manifests = []

        seg_dict = {}
        if os.path.isfile(seg_path):
            try:
                with open(seg_path) as f:
                    seg_dict = json.load(f)
            except Exception as exc:
                return jsonify({"error": f"could not read segmentations.json: {exc}"}), 500

        round_manifests = sorted(glob.glob(os.path.join(rounds_dir, "round_*.json"))) \
            if os.path.isdir(rounds_dir) else []

        for manifest_path in round_manifests:
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
            except Exception:
                continue
            seeded = manifest.get("seeded") or {}
            manifest_fully_consumed = True

            for filename, mask_ids in seeded.items():
                seg = seg_dict.get(filename)
                if seg is None:
                    continue  # already removed by an earlier pass/manifest
                if seg.get("exported"):
                    kept_frames.append(filename)
                    manifest_fully_consumed = False
                    continue

                mask_id_set = set(mask_ids or [])
                before = seg.get("masks") or []
                keep_masks = []
                for m in before:
                    is_seeded_pending_model = (
                        m.get("id") in mask_id_set
                        and m.get("source_type") == "model"
                        and m.get("status") == "pending"
                    )
                    if is_seeded_pending_model:
                        dropped_masks += 1
                    else:
                        keep_masks.append(m)
                seg["masks"] = keep_masks

                if keep_masks or seg.get("reviewed"):
                    # Either a real (accepted/manual/other) mask remains, or a
                    # human already reviewed this frame - never silently delete
                    # review work. Leave the (now-trimmed) entry in place.
                    kept_frames.append(filename)
                    manifest_fully_consumed = False
                    continue

                # No masks left, never reviewed, and this frame was created by
                # seeding: remove the seg entry and its raw/ symlink.
                del seg_dict[filename]
                removed_frames.append(filename)
                link_path = os.path.join(export_dir, "routed_input", "raw", filename)
                if os.path.islink(link_path):
                    try:
                        os.unlink(link_path)
                    except OSError:
                        pass

            if manifest_fully_consumed:
                consumed_manifests.append(manifest_path)

        if round_manifests:
            os.makedirs(os.path.dirname(seg_path), exist_ok=True)
            import tempfile as _tempfile
            fd, tmp_path = _tempfile.mkstemp(
                dir=os.path.dirname(seg_path), suffix=".tmp")
            os.close(fd)
            with open(tmp_path, "w") as f:
                json.dump(seg_dict, f, indent=2)
            os.replace(tmp_path, seg_path)

            for mp in consumed_manifests:
                try:
                    os.remove(mp)
                except OSError:
                    pass

        step4loop_state.update(running=False, port=None,
                               started_at=None, out_dir=None,
                               project_id=None, input_dir=None)
        step4loop_cfg = {}
        vic.note(
            f"[orch] step4loop reset - dropped {dropped_masks} pending model "
            f"mask(s), removed {len(removed_frames)} frame(s), "
            f"kept {len(kept_frames)} frame(s), "
            f"consumed {len(consumed_manifests)} round manifest(s)"
        )
        return jsonify({
            "success": True,
            "dropped_masks": dropped_masks,
            "removed_frames": sorted(set(removed_frames)),
            "kept_frames": sorted(set(kept_frames)),
            "consumed_manifests": [os.path.basename(m) for m in consumed_manifests],
        })

    # ------------------------------------------------------------------
    # ---- editmasks (standalone Edit Masks) string routes ----
    #
    # Opens TCRMPclip_editMasks against the CURRENT project's step4test export
    # so a reviewer can relabel/fix masks Step 4 already wrote, without
    # re-running Step 4. Unlike step4loop there is no seeding pass: /run just
    # preflights that the export has masks to edit, then launches. Non-chain
    # string routes for the same reason step4test's/step4loop's are:
    # /api/step/<int:step>/* can't hold "editmasks".
    # ------------------------------------------------------------------

    @app.route("/api/step/editmasks/run", methods=["POST"])
    def editmasks_run():
        """Preflight + launch. The export must already have masks to edit
        (segmentations.json from a prior Step 4 session) - editmasks never
        creates a fresh export the way step4test/step4loop can."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        refused = _refuse_for_composite()
        if refused is not None:
            return refused

        export_dir = _step4test_export_dir()
        seg_path = os.path.join(export_dir, "segmentations", "segmentations.json")
        if not os.path.isfile(seg_path):
            return jsonify({"success": False, "error": (
                f"No masks to edit yet: run {step_ref('step4test')} first."
            )}), 400

        try:
            result = _launch_step4test_ui(mode="editmasks")
        except Exception as e:
            return jsonify({"success": False, "error": str(e),
                            "traceback": traceback.format_exc()}), 500
        if isinstance(result, dict) and result.get("error"):
            if "already running" in str(result.get("error", "")).lower():
                port = runner.ports.get("editmasks") or editmasks_state.get("port")
                _push_reviewer_after_health(port, _reviewer_initials(), runner.health_check)
                return jsonify({"success": True, "already_running": True,
                                "phase": "ui_ready", "ui_port": port})
            return jsonify({"success": False, "error": result["error"]}), 500

        port = runner.ports.get("editmasks") or editmasks_state.get("port")
        return jsonify({"success": True, "phase": "ui_ready", "ui_port": port})

    @app.route("/api/step/editmasks/status")
    def editmasks_status():
        status = runner.poll_status("editmasks")
        port = runner.ports.get("editmasks") or editmasks_state.get("port")
        if port:
            status["healthy"] = runner.health_check(port)
            status["port"] = port
        return jsonify(status)

    @app.route("/api/step/editmasks/log")
    def editmasks_log():
        offset = request.args.get("offset", 0, type=int)
        lines, new_offset, dropped = runner.get_log("editmasks", offset)
        return jsonify({"lines": lines, "offset": new_offset, "dropped": dropped})

    @app.route("/api/step/editmasks/stop", methods=["POST"])
    def editmasks_stop():
        # _live_annotator_port confirms runner.is_running("editmasks") first,
        # so a dead key's stale port is never flushed.
        port = _live_annotator_port("editmasks")
        if port:
            ok, reason = _flush_annotator(port)
            if not ok:
                vic.note(f"[orch] annotator flush failed on port {port} before stop: {reason}")
        runner.kill("editmasks")
        _route_mark_stopped_if_owner("editmasks", "Edit Masks stopped.")
        editmasks_state.update(running=False, project_id=None, input_dir=None)
        return jsonify({"success": True})

    @app.route("/api/step/editmasks/folder", methods=["POST"])
    def editmasks_folder():
        """Open the step4test export dir (the masks Edit Masks operates on) in
        the file manager. Same guardrail as step4test's/step4loop's /folder."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        path = os.path.abspath(_step4test_export_dir())
        return _folder_response(path, current_project["project_dir"])

    @app.route("/api/step/editmasks/reset", methods=["POST"])
    def editmasks_reset():
        """Stop the Edit Masks service and clear its transient launch state.

        Unlike step4test's/step4loop's reset, this NEVER touches the export
        dir or segmentations.json: editmasks only ever opens the export Step 4
        (and Refine) already built, it never owns that data, so "reset" here
        means only "stop the service and forget the port" - the masks
        themselves are untouched.
        """
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        # _live_annotator_port confirms runner.is_running("editmasks") first,
        # so a dead key's stale port is never flushed.
        port = _live_annotator_port("editmasks")
        if port:
            ok, reason = _flush_annotator(port)
            if not ok:
                vic.note(f"[orch] annotator flush failed on port {port} before reset: {reason}")
        runner.kill("editmasks")
        _route_mark_stopped_if_owner("editmasks", "Edit Masks stopped.")
        editmasks_state.update(running=False, port=None,
                               started_at=None, out_dir=None,
                               project_id=None, input_dir=None)
        vic.note("[orch] editmasks reset - stopped service, export left untouched")
        return jsonify({"success": True, "state": _state_payload(current_project)})

    # ---- Annotate Images (standalone, no project) -------------------------
    # Launches combinedAnnotate in standalone mode: the operator imports their
    # own images and labels them outside any project. Own runner key + fixed
    # port; it NEVER calls _kill_all_annotators, so a project annotator that
    # is open stays open. The reverse also holds: _kill_all_annotators (run by
    # every project-mode launch) skips any process whose environment carries
    # TCRMP_SESSION_MODE=standalone, so a project-mode launch cannot kill a
    # running standalone session either.
    def _launch_standalone_annotator():
        global annotate_state
        port_now = runner.ports.get("annotate") or annotate_state.get("port")
        if port_now and runner.poll_status("annotate").get("running") and runner.health_check(port_now):
            return {"phase": "ui_ready", "ui_port": port_now, "already_running": True}
        try:
            runner.kill("annotate")
        except Exception:
            pass
        entry = os.path.join(REPO_DIR, "scripts", "TCRMPclip_combinedAnnotate", "src", "app.py")
        cwd = os.path.join(REPO_DIR, "scripts", "TCRMPclip_combinedAnnotate")
        if not os.path.isfile(entry):
            return {"error": f"combined annotator not found at {entry}"}
        py = PYTHON_PATHS.get(5, "python3")
        if not os.path.isfile(py):
            py = "python3"
        port = find_free_port(ANNOTATE_PORT)
        root = _standalone_root()
        os.makedirs(root, exist_ok=True)
        env_extra = {
            "TCRMP_SESSION_MODE": "standalone",
            "TCRMP_PROVENANCE_SOURCE": "standalone",
            "TCRMP_STANDALONE_ROOT": root,
            "TCRMP_EXEMPLAR_WARMUP": "1",
            # ORCHESTRATOR_PORT (not STAGE_PORTS, which is int-step-keyed, not
            # name-keyed) is the orchestrator's own base URL.
            "TCRMP_ORCHESTRATOR_URL": f"http://127.0.0.1:{ORCHESTRATOR_PORT}",
        }
        # Best-effort reuse of the current project's recoded master codes, if
        # one happens to be loaded; standalone mode has no project of its own,
        # so a project-less launch (current_project is None) skips this.
        _mc = ""
        if current_project is not None:
            try:
                _mc = _resolve_master_codes()
            except Exception:
                _mc = ""
        if _mc:
            env_extra["TCRMP_MASTER_CODES"] = _mc
        env_extra = {k: v for k, v in env_extra.items() if v}
        result = runner.run_flask_stage("annotate", [py, entry, "--port", str(port)], port,
                                        cwd=cwd, env_extra=env_extra)
        if isinstance(result, dict) and result.get("error"):
            return result
        annotate_state.update(running=True, port=port, started_at=datetime.now().isoformat(), out_dir=root)
        vic.note(f"[annotate] standalone annotator launched on :{port} (root {root})")
        return {"phase": "ui_ready", "ui_port": port}

    @app.route("/annotate")
    def annotate_page():
        return render_template("annotate.html", asset_version=ASSET_VERSION)

    @app.route("/api/annotate/run", methods=["POST"])
    def annotate_run():
        try:
            result = _launch_standalone_annotator()
        except Exception as e:
            return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500
        if result.get("error"):
            return jsonify({"success": False, "error": result["error"]}), 500
        return jsonify({"success": True, **result})

    @app.route("/api/annotate/status")
    def annotate_status():
        status = runner.poll_status("annotate")
        port = runner.ports.get("annotate") or annotate_state.get("port")
        if port:
            status["healthy"] = runner.health_check(port)
            status["port"] = port
        return jsonify(status)

    @app.route("/api/annotate/stop", methods=["POST"])
    def annotate_stop():
        runner.kill("annotate")
        annotate_state.update(running=False, port=None)
        return jsonify({"success": True})

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    return app

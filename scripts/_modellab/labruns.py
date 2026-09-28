"""Shared cross-run library and background job runner the VEGA main window
imports: run discovery, champion records, split comparability and the
detection-dataset wrappers.

Everything here works off explicit paths, so the orchestrator (and its tests)
can point at a temp tree. Nothing in this module imports the orchestrator app,
Flask, or ultralytics: every function below runs in a bare interpreter with
no application context and no request object. Load it by file path or as
``_modellab.labruns``; the package ``__init__`` imports nothing, so neither
route brings Flask along. The project-free Train + Infer page that first owned
this file was retired on 2026-09-05 (docs/adr/0002).

Layout it owns, all under the module's `inprocess/` root:

    _models/<lab_name>/lab.json     what this lab trains (source, task, model)
    _models/<lab_name>/dataset/     pinned_split output (data.yaml, test.yaml)
    _models/<lab_name>/runs/<name>/ ultralytics training runs, each holding a
                                    split_record.json naming the split that
                                    run was trained on, once that split has
                                    been recorded on it
    _models/<lab_name>/rounds.csv   evaluation ledger appended by evaluate_run
    _models/_eval/<run>_<ts>/       evaluation reports
    _models/_inference/<ts>_<note>/ inference overlays + manifest
    _models/champion.json           {run_dir, at} the promoted model (the
                                    orchestrator writes its own
                                    step6_trainModel/champion.json in two
                                    richer shapes; read_champion_file reads
                                    all three)

Training sources are read-only inputs from two places:

    <inprocess>/_composite/<name>/          segmentation composites built on
                                            the coverage matrix (task seg)
    <supporting_data>/bbox_datasets/<name>/ staged detection datasets
                                            (task detect)

Non-TCRMP bbox imagery never enters inprocess/run_*, the mask registry, the
coverage matrix, or STATS: a bbox source is read where it is staged and its
outputs stay under _models/.
"""

import csv
import importlib.util
import json
import logging
import math
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone

# The detection-dataset builder lives with the training tool it feeds
# (TCRMPtrain_oceankindCV/src/detect_dataset.py) so the pipeline's Step 6 and
# this lab build byte-identical splits from one implementation. It is loaded by
# file path under a private name: the training tool is a script dir, not an
# importable package, and loading it this way keeps the name out of the way of
# anything else in sys.modules. Its import is side-effect free.
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DETECT_DATASET_PATH = os.path.join(_SCRIPTS_DIR, "TCRMPtrain_oceankindCV", "src",
                                    "detect_dataset.py")


def _load_detect_dataset():
    """The shared builder module, loaded eagerly at import.

    Eager and hard-failing on purpose. The file is a sibling in this same
    checkout, so its absence means a broken install, and the alternatives are
    worse: a lazy load would surface the same breakage in the middle of a
    training request, and a silent fallback copy would let the lab and the
    pipeline drift into building different splits from the same source.

    A missing file raises FileNotFoundError out of exec_module (naming the
    path), not ImportError: spec_from_file_location does not check that the
    path exists, so the spec-None branch below only covers a loader that
    refuses the suffix.
    """
    spec = importlib.util.spec_from_file_location("_modellab_detect_dataset",
                                                  _DETECT_DATASET_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the detection dataset builder at {_DETECT_DATASET_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_DD = _load_detect_dataset()

# Atlantic Standard Time, fixed UTC-4, no daylight saving (house rule: every
# timestamp this platform writes is AST 24h).
AST = timezone(timedelta(hours=-4))

_LOG = logging.getLogger(__name__)

LAB_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,60}$")
MODEL_SIZES = ("n", "s", "m", "l", "x")
TASKS = ("seg", "detect")

# What a run is filed as when nothing on disk says otherwise, and the image
# size assumed when args.yaml does not record one.
DEFAULT_TASK = "seg"
DEFAULT_IMGSZ = 512

# Split fractions the detection wrappers below default to. Read from the
# builder so the pipeline's Step 6 and every caller of this library split on
# the same pair of numbers; the values are never restated here.
DEFAULT_VALID_FRAC = _DD.DEFAULT_VALID_FRAC
DEFAULT_TEST_FRAC = _DD.DEFAULT_TEST_FRAC

# ultralytics writes one flat ``key: value`` file per run.
ARGS_YAML_NAME = "args.yaml"
# The task vocabulary ultralytics writes into args.yaml, mapped onto TASKS.
_ULTRALYTICS_TASK_NAMES = {"segment": "seg", "seg": "seg", "detect": "detect"}
# A base-weights file name carrying this text (yolo11m-seg.pt) is a segmenter.
_SEG_MODEL_NAME_MARK = "-seg"

# The promoted model. Three shapes on disk, one normalised record: see
# read_champion_file. CHAMPION_FIELDS is the published shape of that record,
# for callers and tests; CHAMPION_SCORE_KEYS maps a metric family to the key
# that carries its mAP50-95.
CHAMPION_FILE_NAME = "champion.json"
CHAMPION_FIELDS = ("run_dir", "promoted_at", "metric_family", "map50_95_M", "map50_95_B", "score", "note")
CHAMPION_SCORE_KEYS = {"mask": "map50_95_M", "box": "map50_95_B"}

# Comparability. The split fingerprint and the scheme it was hashed under are
# written by the dataset builder (TCRMPtrain_oceankindCV/src/detect_dataset.py)
# into the dataset's split manifest. These three names are the only manifest
# fields this library reads; change them here if the writer renames them.
SPLIT_MANIFEST_NAME = "split_manifest.json"
SPLIT_FINGERPRINT_FIELD = "split_fingerprint"
SPLIT_FINGERPRINT_SCHEME_FIELD = "split_fingerprint_scheme"

# A run's own split record, written inside the run folder. The dataset folder
# beside a run is rebuilt every time its training set changes, so its manifest
# says which split the NEXT run would train on, not the one this run was
# scored on. Reading it live made every run of one project report the same
# fingerprint and let an old run claim a split it had never seen, so the
# fingerprint is recorded once, into the run, and read from there afterwards.
RUN_SPLIT_RECORD_NAME = "split_record.json"
# The three fields a record carries beside the fingerprint and its scheme.
# recorded_at is written by ast_now, so it is Atlantic Standard Time on a
# 24 hour clock, the one shape every timestamp in this module is written in.
# RUN_SPLIT_RECORD_FIELDS is spelled from these names and the two manifest
# field names above, so the published field list and the writer cannot drift.
RUN_SPLIT_RECORDED_AT_FIELD = "recorded_at"
RUN_SPLIT_RECORDED_SOURCE_FIELD = "recorded_source"
RUN_SPLIT_DATASET_DIR_FIELD = "dataset_dir"
RUN_SPLIT_RECORD_FIELDS = (SPLIT_FINGERPRINT_FIELD, SPLIT_FINGERPRINT_SCHEME_FIELD,
                           RUN_SPLIT_RECORDED_AT_FIELD, RUN_SPLIT_RECORDED_SOURCE_FIELD,
                           RUN_SPLIT_DATASET_DIR_FIELD)
# A record is written through a temporary file in the run folder and moved
# over the target. One left behind by a process killed between the open and
# the move is swept on the next write, once it is older than this. A live
# writer's temporary file is milliseconds old, so an hour never takes one out
# from under a writer that is still using it.
RUN_SPLIT_RECORD_TMP_SUFFIX = ".tmp"
STALE_RECORD_TMP_AGE_SECONDS = 3600
# What occasion wrote a record: the training path stamping a run it has just
# finished, or the first read of a run trained before records existed.
SPLIT_RECORD_SOURCE_TRAINING = "training"
SPLIT_RECORD_SOURCE_FIRST_READ = "first-read"
SPLIT_RECORD_SOURCES = (SPLIT_RECORD_SOURCE_TRAINING, SPLIT_RECORD_SOURCE_FIRST_READ)

# A segmenter is scored on mask metrics (map50_M and kin), a detector on box
# metrics (map50_B). The two families are never ranked against each other.
METRIC_FAMILY_MASK = "mask"
METRIC_FAMILY_BOX = "box"
_TASK_TO_METRIC_FAMILY = {"seg": METRIC_FAMILY_MASK, "detect": METRIC_FAMILY_BOX}

# Folder names under _models/ that are buckets, not labs.
_RESERVED_LAB_NAMES = ("_inference", "_eval")

_IMAGE_DIRS = ("all_images", "images", os.path.join("train", "images"))
_IMAGE_EXTS = _DD.IMAGE_EXTS


def ast_now():
    """Timestamp the way every VICARIUS surface writes it: AST, 24h."""
    return datetime.now(AST).strftime("%Y-%m-%d %H:%M:%S")


# ── Roots ────────────────────────────────────────────────────────────────

def models_root(inprocess_root):
    return os.path.join(inprocess_root, "_models")


def composite_root(inprocess_root):
    return os.path.join(inprocess_root, "_composite")


def inference_root(inprocess_root):
    return os.path.join(models_root(inprocess_root), "_inference")


def eval_root(inprocess_root):
    return os.path.join(models_root(inprocess_root), "_eval")


def champion_path(inprocess_root):
    return os.path.join(models_root(inprocess_root), CHAMPION_FILE_NAME)


# ── Names and paths ──────────────────────────────────────────────────────

def validate_lab_name(name):
    """(name, '') for an acceptable lab name, else ('', reason). Letters,
    digits, underscore and hyphen only, so a name can never carry a path
    separator or a traversal segment."""
    clean = str(name or "").strip()
    if not clean:
        return "", "a name for this model lab is required"
    if not LAB_NAME_RE.match(clean):
        return "", ("name must be 1 to 60 characters of letters, digits, "
                    "underscore or hyphen: " + clean)
    return clean, ""


def lab_paths(inprocess_root, lab_name):
    """Every path one lab owns. Raises ValueError on an unacceptable name, so
    no caller can build a path outside _models/."""
    clean, err = validate_lab_name(lab_name)
    if err:
        raise ValueError(err)
    lab_dir = os.path.join(models_root(inprocess_root), clean)
    return {
        "lab_name": clean,
        "lab_dir": lab_dir,
        "dataset_dir": os.path.join(lab_dir, "dataset"),
        "runs_dir": os.path.join(lab_dir, "runs"),
        "rounds_csv": os.path.join(lab_dir, "rounds.csv"),
        "lab_json": os.path.join(lab_dir, "lab.json"),
    }


def safe_note(note):
    """A folder-safe leaf for an inference note. Anything outside
    [A-Za-z0-9_.-] collapses to an underscore, so separators and traversal
    segments can never survive into a path."""
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(note or "").strip())[:60]
    return clean or "infer"


def inference_out_dir(inprocess_root, note, ts=None):
    stamp = ts or time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(inference_root(inprocess_root), f"{stamp}_{safe_note(note)}")


def eval_out_dir(inprocess_root, run_name, ts=None):
    stamp = ts or time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(eval_root(inprocess_root), f"{safe_note(run_name)}_{stamp}")


# ── data.yaml ────────────────────────────────────────────────────────────

def data_yaml_labels(path):
    """Class names from a YOLO data.yaml (list or {id: name} mapping); []
    when the file is missing or unreadable. PyYAML when it imports, else a
    tolerant line parser, so a lab listing never fails on a stray file."""
    return _DD.data_yaml_labels(path)


def _count_images(dataset_dir):
    for rel in _IMAGE_DIRS:
        d = os.path.join(dataset_dir, rel)
        if os.path.isdir(d):
            try:
                return sum(1 for fn in os.listdir(d)
                           if os.path.splitext(fn)[1].lower() in _IMAGE_EXTS)
            except OSError:
                return 0
    return 0


def _mtime_stamp(path):
    try:
        return datetime.fromtimestamp(os.path.getmtime(path), AST).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return ""


# ── Training sources ─────────────────────────────────────────────────────

def list_composite_sources(inprocess_root):
    """Composite YOLO-seg exports under <inprocess>/_composite. Same discovery
    rule the orchestrator's Step 6 source select uses: a directory holding
    data.yaml whose real path stays inside the composite root."""
    root = composite_root(inprocess_root or "")
    if not os.path.isdir(root):
        return []
    real_root = os.path.realpath(root)
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        yaml_path = os.path.join(d, "data.yaml")
        if not os.path.isdir(d) or not os.path.isfile(yaml_path):
            continue
        if not os.path.realpath(d).startswith(real_root + os.sep):
            continue
        built_at = ""
        try:
            with open(os.path.join(d, "composite.json")) as fh:
                built_at = str((json.load(fh) or {}).get("created_at") or "")[:19]
        except Exception:
            built_at = ""
        out.append({
            "kind": "composite",
            "task": "seg",
            "name": name,
            "path": d,
            "labels": data_yaml_labels(yaml_path),
            "n_images": _count_images(d),
            "built_at": built_at or _mtime_stamp(yaml_path),
        })
    return out


def list_bbox_sources(bbox_root):
    """Staged detection datasets under supporting_data/bbox_datasets. A
    finished dataset carries both data.yaml and ingest_manifest.json; a
    half-staged folder is skipped rather than offered and then refused."""
    if not bbox_root or not os.path.isdir(bbox_root):
        return []
    out = []
    for name in sorted(os.listdir(bbox_root)):
        d = os.path.join(bbox_root, name)
        yaml_path = os.path.join(d, "data.yaml")
        manifest_path = os.path.join(d, "ingest_manifest.json")
        if not os.path.isdir(d) or not os.path.isfile(yaml_path) or not os.path.isfile(manifest_path):
            continue
        manifest = {}
        try:
            with open(manifest_path) as fh:
                manifest = json.load(fh) or {}
        except Exception:
            manifest = {}
        labels = manifest.get("classes")
        if not isinstance(labels, list) or not labels:
            labels = data_yaml_labels(yaml_path)
        try:
            n_images = int(manifest.get("n_images"))
        except (TypeError, ValueError):
            n_images = _count_images(d)
        built_at = str(manifest.get("ingested_at") or manifest.get("built_at") or "")[:19]
        out.append({
            "kind": "bbox",
            "task": "detect",
            "name": name,
            "path": d,
            "labels": [str(x) for x in labels],
            "n_images": n_images,
            "built_at": built_at or _mtime_stamp(manifest_path),
        })
    return out


def list_sources(inprocess_root, bbox_root):
    return list_composite_sources(inprocess_root) + list_bbox_sources(bbox_root)


def validate_composite_source(source_dir):
    """('', reason) when a composite cannot be trained from. Same two checks
    the orchestrator's Step 6 makes (app.py _validate_composite_source): a
    composite the matrix started but never filled has a data.yaml and no
    images, and reaches ultralytics as an opaque traceback."""
    path = str(source_dir or "")
    if not os.path.isdir(os.path.join(path, "all_images")):
        return "", (f"this training set has no all_images/ yet: {path}. Rebuild it on the "
                    "coverage matrix, Training Sets tab.")
    if not os.path.isfile(os.path.join(path, "data.yaml")):
        return "", f"this training set has no data.yaml: {path}"
    return path, ""


def find_source(sources, path):
    """The listed source whose path is ``path``, else None. Training only ever
    starts from a source the listing already offered."""
    raw = str(path or "").strip()
    if not raw:
        return None
    want = os.path.realpath(raw)
    for s in sources:
        if os.path.realpath(s["path"]) == want:
            return s
    return None


# ── Detection dataset build ──────────────────────────────────
#
# A staged detection dataset is {images/, labels/ (or labels_localized/),
# data.yaml, classes.txt, ingest_manifest.json}. pinned_split.py reads only
# the composite shape (all_images/ + all_labels/) and pins the holdout by
# TCRMP transect, which a non-TCRMP dataset has no notion of. So the split is
# built here instead, in the SAME on-disk shape pinned_split produces
# (train/valid/test with a data.yaml whose val is valid/images), and
# run_step6.sh's "train" mode is driven over it. pinned_split.py is untouched.
#
# The implementation moved to TCRMPtrain_oceankindCV/src/detect_dataset.py so
# the pipeline's Step 6 builds detection datasets from the same code the lab
# does. These names stay as they were: every caller and every test that used
# them keeps working, and the behaviour is the module's.

BBOX_LABEL_DIRS = _DD.BBOX_LABEL_DIRS
SPLIT_DIRS = _DD.SPLIT_DIRS


def bbox_label_dir(dataset_dir):
    """Where a staged detection dataset keeps its YOLO label files.
    labels_localized/ (degenerate full-image boxes dropped) wins over
    labels/ when the ingest wrote both. '' when neither exists."""
    return _DD.bbox_label_dir(dataset_dir)


def read_classes_txt(path):
    """Class names, one per line, from a pinned classes.txt; [] when absent."""
    return _DD.read_classes_txt(path)


def bbox_pairs(dataset_dir):
    """(pairs, unlabelled): every image with a label file beside it, and how
    many images had none. An image with no label file is a background image,
    not a training pair, and a staged dataset that lost its labels must fail
    loudly rather than train on nothing."""
    return _DD.bbox_pairs(dataset_dir)


def assign_split(names, seed_text, existing=None, valid_frac=DEFAULT_VALID_FRAC,
                 test_frac=DEFAULT_TEST_FRAC):
    """basename -> train|valid|test, seeded on the lab name so the same lab
    always splits the same images the same way. ``existing`` pins are replayed
    unchanged, so retraining a lab never moves an image between splits and
    only assigns images the dataset gained since. The fractions default to
    the builder's DEFAULT_VALID_FRAC and DEFAULT_TEST_FRAC."""
    return _DD.assign_split(names, seed_text, existing, valid_frac, test_frac)


def _place(src, dst):
    """Hardlink when the filesystem allows it (a staged dataset is read-only
    input, and a 10k-image link costs nothing), else copy."""
    return _DD._place(src, dst)


def build_detect_dataset(source_dir, dataset_dir, lab_name, valid_frac=DEFAULT_VALID_FRAC,
                         test_frac=DEFAULT_TEST_FRAC):
    """Build <lab>/dataset/ from a staged detection dataset and return a
    summary. Raises ValueError when the source holds no image/label pair, or
    when a saved or staged split names a photo the source cannot train on.
    The fractions default to the builder's DEFAULT_VALID_FRAC and
    DEFAULT_TEST_FRAC."""
    return _DD.build_detect_dataset(source_dir, dataset_dir, lab_name, valid_frac, test_frac)


# ── Runs ─────────────────────────────────────────────────────────────────

def read_rounds(rounds_csv: str) -> list[dict]:
    """Rows of a rounds.csv evaluation ledger as dicts, in file order.

    Parameters: ``rounds_csv``, the ledger path. Returns [] when the path is
    empty or the file is missing, and [] with the reason logged when the file
    cannot be read or parsed as CSV, so one damaged ledger never stops a scan.
    A row shorter than the header carries None for its missing cells; a cell
    holding quotes, newlines or a spreadsheet formula is returned as the
    plain text it is.
    """
    if not rounds_csv or not os.path.isfile(rounds_csv):
        return []
    try:
        with open(rounds_csv, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        _LOG.warning("cannot read the rounds ledger %s: %s", rounds_csv, exc)
        return []


def _map50_for(rows, run_dir, run_name):
    """mAP50 of one run from its rounds.csv row. Mask mAP first (map50_M);
    a detection run has no mask metrics at all, so its row carries the box
    column (map50_B) instead and the table would otherwise read blank."""
    for row in rows:
        if row.get("run_dir") == run_dir or row.get("run_name") == run_name:
            for column in ("map50_M", "map50_B"):
                try:
                    return float(row.get(column))
                except (TypeError, ValueError):
                    continue
            return None
    return None


def _args_yaml_field(run_dir: str, key: str) -> str:
    """The value of one top-level ``key:`` line in <run_dir>/args.yaml.

    ultralytics writes args.yaml as flat ``key: value`` lines, so a line
    scan is enough and keeps this library free of a YAML dependency. Only a
    line starting at column one counts; an indented key belongs to a nested
    block. Surrounding quotes are stripped. Returns '' when the file is
    missing, empty, or has no such line, and '' with the reason logged when
    the file cannot be read as text.
    """
    path = os.path.join(run_dir, ARGS_YAML_NAME)
    if not os.path.isfile(path):
        return ""
    prefix = key + ":"
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(prefix):
                    return line[len(prefix):].strip().strip("'\"")
    except (OSError, UnicodeDecodeError) as exc:
        _LOG.warning("cannot read %s while looking for %r: %s", path, key, exc)
        return ""
    return ""


def _args_yaml_model(run_dir: str) -> str:
    """The base weights a run started from (the ``model:`` line of its
    args.yaml), '' when unknown."""
    return _args_yaml_field(run_dir, "model")


def _args_yaml_task(run_dir: str) -> str:
    """What ultralytics says a run trained, from the ``task:`` line of its
    args.yaml, in this library's vocabulary: 'seg', 'detect', or '' when the
    line is absent or carries a value this library does not know (logged,
    then treated as absent so the caller falls back)."""
    raw = _args_yaml_field(run_dir, "task")
    if not raw:
        return ""
    task = _ULTRALYTICS_TASK_NAMES.get(raw.lower())
    if task is None:
        _LOG.warning("%s has an unrecognised task %r; falling back to lab.json, then the model-name guess",
                     os.path.join(run_dir, ARGS_YAML_NAME), raw)
        return ""
    return task


def _task_from_model_name(model: str) -> str:
    """A GUESS at a run's task from its base-weights file name: a name
    carrying '-seg' (yolo11m-seg.pt) is a segmenter, anything else a
    detector. The guess is wrong for a run continued from an earlier run's
    weights/best.pt, whose name says nothing about the task, which is why
    the args.yaml task line is consulted first. '' when there is no name."""
    if not model:
        return ""
    return "seg" if _SEG_MODEL_NAME_MARK in model else "detect"


def run_imgsz(run_dir: str, default: int = DEFAULT_IMGSZ) -> int:
    """The square image size a run was trained at, from the ``imgsz:`` line
    of its args.yaml. Returns ``default`` when the file or the line is
    missing, or the value is not a finite positive number (logged), so an
    evaluation never silently changes resolution."""
    raw = _args_yaml_field(run_dir, "imgsz")
    if not raw:
        return default
    try:
        size = int(float(raw))
    except (ValueError, OverflowError):
        size = 0
    if size <= 0:
        _LOG.warning("%s has an unusable imgsz %r; using %d",
                     os.path.join(run_dir, ARGS_YAML_NAME), raw, default)
        return default
    return size


def _lab_task(lab_dir: str) -> str:
    """The task a lab was configured to train, from <lab_dir>/lab.json;
    '' when the file is missing, unreadable (logged) or names no known task."""
    path = os.path.join(lab_dir, "lab.json")
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        _LOG.warning("cannot read the lab record %s: %s", path, exc)
        return ""
    if not isinstance(record, dict):
        _LOG.warning("lab record %s is not a JSON object", path)
        return ""
    task = str(record.get("task") or "")
    return task if task in TASKS else ""


def _task_for_run(run_dir: str, lab_task: str) -> str:
    """One run's task ('seg' or 'detect'), from the most to the least
    trustworthy source: the ``task:`` line ultralytics wrote into the run's
    args.yaml; then the owning lab's lab.json (``lab_task``, '' for a project
    run); then a guess from the base-weights file name; then DEFAULT_TASK."""
    return (_args_yaml_task(run_dir)
            or lab_task
            or _task_from_model_name(_args_yaml_model(run_dir))
            or DEFAULT_TASK)


def dataset_for_run(run_dir):
    """The dataset a run was trained on: dataset/ next to its runs/ parent
    (true for both a lab and a project step6 dir). '' when there is none."""
    if not run_dir:
        return ""
    parent = os.path.dirname(os.path.dirname(os.path.abspath(run_dir)))
    ds = os.path.join(parent, "dataset")
    if os.path.isdir(ds) and os.path.isfile(os.path.join(ds, "data.yaml")):
        return ds
    return ""


def rounds_dir_for_run(run_dir):
    """Where this run's evaluation ledger lives: the lab dir for a lab run,
    the step6 dir for a project run. Both are the runs/ parent."""
    if not run_dir:
        return ""
    return os.path.dirname(os.path.dirname(os.path.abspath(run_dir)))


def _subdirs(path: str) -> list[str]:
    """Sorted names of the sub-directories of ``path``; [] when it cannot be
    listed, with the reason logged, so one unreadable folder is skipped
    visibly instead of stopping a whole scan."""
    try:
        names = os.listdir(path)
    except OSError as exc:
        _LOG.warning("cannot list %s: %s", path, exc)
        return []
    return sorted(n for n in names if os.path.isdir(os.path.join(path, n)))


def _run_dirs(runs_dir: str, real_root: str) -> list[str]:
    """The run folders under ``runs_dir`` that really sit inside
    ``real_root`` (already resolved with os.path.realpath). A run folder
    that is a symlink pointing outside the root is skipped with the reason
    logged: every later action on it would be refused by validate_run_dir,
    so listing it would only show a row nobody can act on."""
    out = []
    for name in _subdirs(runs_dir):
        run_dir = os.path.join(runs_dir, name)
        if not _contained(os.path.realpath(run_dir), real_root):
            _LOG.warning("skipping %s: it resolves outside the runs root %s", run_dir, real_root)
            continue
        out.append(run_dir)
    return out


def _run_row(run_dir, origin, task, rows, champion_dir):
    run_name = os.path.basename(os.path.normpath(run_dir))
    return {
        "run_dir": run_dir,
        "run_name": run_name,
        "origin": origin,
        "task": task,
        "map50": _map50_for(rows, run_dir, run_name),
        "has_weights": os.path.isfile(os.path.join(run_dir, "weights", "best.pt")),
        "champion": bool(champion_dir) and os.path.realpath(run_dir) == champion_dir,
    }


def list_all_runs(inprocess_root: str) -> list[dict]:
    """Every trained run the module can see, from both roots:
    _models/<lab>/runs/* (origin lab:<name>) and
    run_*/step6_trainModel/runs/* (origin project:<run folder>).

    Parameters: ``inprocess_root``, the module's inprocess/ folder. Returns
    one dict per run with run_dir, run_name, origin, task ('seg' or
    'detect'), map50 (float from the adjacent rounds.csv, else None),
    has_weights and champion (True for the module champion). A missing root
    scans to []; a folder the process cannot list, or a run folder that is a
    symlink out of the root, is skipped with the reason logged. Needs no
    Flask context.
    """
    root = inprocess_root or ""
    if not os.path.isdir(root):
        return []
    real_root = os.path.realpath(root)
    champ = load_champion(root) or {}
    champion_dir = os.path.realpath(champ.get("run_dir")) if champ.get("run_dir") else ""
    out = []

    mroot = models_root(root)
    if os.path.isdir(mroot):
        for lab in _subdirs(mroot):
            if lab in _RESERVED_LAB_NAMES:
                continue
            lab_dir = os.path.join(mroot, lab)
            runs_dir = os.path.join(lab_dir, "runs")
            if not os.path.isdir(runs_dir):
                continue
            rows = read_rounds(os.path.join(lab_dir, "rounds.csv"))
            task = _lab_task(lab_dir)
            for rd in _run_dirs(runs_dir, real_root):
                out.append(_run_row(rd, f"lab:{lab}", _task_for_run(rd, task), rows, champion_dir))

    for entry in _subdirs(root):
        if not entry.startswith("run_"):
            continue
        step6 = os.path.join(root, entry, "step6_trainModel")
        runs_dir = os.path.join(step6, "runs")
        if not os.path.isdir(runs_dir):
            continue
        rows = read_rounds(os.path.join(step6, "rounds.csv"))
        for rd in _run_dirs(runs_dir, real_root):
            out.append(_run_row(rd, f"project:{entry}", _task_for_run(rd, ""), rows, champion_dir))
    return out


def _contained(real_path: str, real_root: str) -> bool:
    """True when ``real_path`` is ``real_root`` or sits beneath it. Both
    arguments are paths already resolved with os.path.realpath, so neither
    a '..' segment nor a symlink can place a path outside the root. A
    sibling whose name merely extends the root's name is not contained."""
    try:
        return os.path.commonpath([real_root, real_path]) == real_root
    except ValueError:
        return False


def validate_run_dir(run_dir: str, root: str = "") -> tuple[str, str]:
    """(absolute path, '') when ``run_dir`` is a run folder holding
    weights/best.pt inside ``root``, else ('', reason).

    Parameters: ``run_dir``, the folder to check; ``root``, the folder every
    acceptable run must sit under (the module's inprocess/ folder). Both are
    resolved through symlinks before the containment check, so a path that
    escapes the root with '..' or through a link is refused, as is the root
    itself. Only the literal default '' skips the containment check: that
    default is a leftover of the retired Train + Infer page (2026-09-05),
    which predated the guard; no live caller relies on it. A root of None is
    refused, so a caller whose configuration lost the root hears about it
    instead of losing the guard. Every caller, the VEGA main window
    included, MUST pass the root.
    """
    clean = str(run_dir or "").strip()
    if not clean:
        return "", "pick a model run first"
    path = os.path.abspath(clean)
    if root is None:
        return "", "the runs root is not set; pass the module's inprocess folder"
    if root:
        real_root = os.path.realpath(root)
        if not os.path.exists(real_root):
            return "", f"the runs root does not exist: {root}"
        if not os.path.isdir(real_root):
            return "", f"the runs root is not a folder: {root}"
        real_path = os.path.realpath(path)
        if real_path == real_root:
            return "", f"the runs root itself is not a run folder: {root}"
        if not _contained(real_path, real_root):
            return "", f"run folder is outside the runs root ({root}): {path}"
    if not os.path.isdir(path):
        return "", f"run folder not found: {path}"
    best = os.path.join(path, "weights", "best.pt")
    if not os.path.isfile(best):
        return "", f"this run has no weights/best.pt yet: {path}"
    return path, ""


def validate_infer_source(inprocess_root, source_dir):
    """(abs path, '') for a folder inference may read, else ('', reason).

    Refuses any folder inside a pipeline project's step output
    (inprocess/run_*/step*): those are pipeline state, and pointing a
    standalone inference run at them mixes lab output into project data."""
    clean = str(source_dir or "").strip()
    if not clean:
        return "", "type the folder of images to run the model on"
    path = os.path.abspath(clean)
    if not os.path.isdir(path):
        return "", f"image folder not found: {path}"
    root = os.path.realpath(inprocess_root or "")
    real = os.path.realpath(path)
    if root and (real == root or real.startswith(root + os.sep)):
        parts = os.path.relpath(real, root).split(os.sep)
        if len(parts) >= 2 and parts[0].startswith("run_") and parts[1].startswith("step"):
            return "", ("this folder is inside a pipeline project step output "
                        f"({parts[0]}/{parts[1]}); pick a folder outside the projects")
    return path, ""


def base_model_for(task, size, train_dir):
    """The base weights to train from: the file already in the training tool
    directory when it is there (no download), else the plain name for
    ultralytics to fetch. A detection source trains a detection model."""
    name = f"yolo11{size}.pt" if task == "detect" else f"yolo11{size}-seg.pt"
    path = os.path.join(train_dir or "", name)
    return path if os.path.isfile(path) else name


def task_for_run_dir(run_dir: str) -> str:
    """What one run predicts ('seg' or 'detect') without scanning every root.

    Parameters: ``run_dir``, the run folder. Trusts the ``task:`` line
    ultralytics wrote into the run's args.yaml first; then the lab.json of
    the lab that owns the run, if any; then a guess from the base-weights
    file name (a guess, wrong for a run continued from another run's
    best.pt); then DEFAULT_TASK. '' only for an empty ``run_dir``.
    """
    if not run_dir:
        return ""
    path = os.path.abspath(run_dir)
    return _task_for_run(path, _lab_task(rounds_dir_for_run(path)))


def newest_weights(inprocess_root, lab_name="", task=""):
    """Best weights to continue from: the promoted champion's best.pt, else
    the newest best.pt in this lab, else ''.

    ``task`` filters by what the weights predict. A segmentation run must
    never continue from a detection champion (and the reverse): ultralytics
    accepts the file and then fails deep inside the training loop, so the
    mismatch is dropped here and the route reports it plainly.
    """
    champ = load_champion(inprocess_root) or {}
    champ_run = str(champ.get("run_dir") or "")
    best = os.path.join(champ_run, "weights", "best.pt")
    if champ_run and os.path.isfile(best):
        if not task or task_for_run_dir(champ_run) == task:
            return best
    if not lab_name:
        return ""
    try:
        runs_dir = lab_paths(inprocess_root, lab_name)["runs_dir"]
    except ValueError:
        return ""
    newest, newest_mtime = "", -1.0
    if os.path.isdir(runs_dir):
        for nm in sorted(os.listdir(runs_dir)):
            run_dir = os.path.join(runs_dir, nm)
            cand = os.path.join(run_dir, "weights", "best.pt")
            if not os.path.isfile(cand):
                continue
            if task and task_for_run_dir(run_dir) != task:
                continue
            mtime = os.path.getmtime(cand)
            if mtime > newest_mtime:
                newest, newest_mtime = cand, mtime
    return newest


# ── Champion ─────────────────────────────────────────────────────────────

def read_champion_file(path: str) -> dict | None:
    """The champion record at ``path`` in one normalised shape, or None.

    Three shapes exist on disk. All stay readable here and no file is
    rewritten by a read:

      {run_dir, at}
          written by save_champion in this file: the module-wide champion at
          _models/champion.json and its mirror in a lab folder
          (mirror_champion_to_lab)
      {run_dir, promoted_at, map50_95_M, note}
          written by the orchestrator's POST /api/step/7/promote before
          2026-09-04 into a project's step6_trainModel/champion.json; mask
          score only, so a detector champion carried no score
      {run_dir, promoted_at, metric_family, map50_95_M or map50_95_B, note}
          written by the widened promote route from 2026-09-04: the family
          ('mask' or 'box') and the one score key that matches it

    Parameters: ``path``, the champion.json to read. Returns a dict with the
    keys of CHAMPION_FIELDS: run_dir (the path as written, stripped);
    promoted_at (``promoted_at`` else ``at`` else ''; save_champion writes
    'YYYY-MM-DD HH:MM:SS' and the orchestrator writes ISO with the offset,
    'YYYY-MM-DDTHH:MM:SS-04:00', so parse for either); metric_family
    ('mask' or 'box', from the file when it carries a valid one, else derived
    from the run's task on disk); map50_95_M and map50_95_B (each a finite
    float or None); score (the family's own score, so map50_95_M for a mask
    champion and map50_95_B for a box champion, None when that key is absent;
    a file whose only score belongs to the other family is logged and reads
    as score None); and note (str). A missing file is None; a file that is
    not JSON, not an object, or has no run_dir string is None with the
    reason logged.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        _LOG.warning("cannot read the champion record %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        _LOG.warning("champion record %s is not a JSON object", path)
        return None
    run_dir = data.get("run_dir")
    if not isinstance(run_dir, str) or not run_dir.strip():
        _LOG.warning("champion record %s names no run_dir", path)
        return None
    run_dir = run_dir.strip()
    family = _champion_family(data.get("metric_family"), run_dir, path)
    scores = {key: _finite_score(data.get(key), path) for key in CHAMPION_SCORE_KEYS.values()}
    return {
        "run_dir": run_dir,
        "promoted_at": str(data.get("promoted_at") or data.get("at") or ""),
        "metric_family": family,
        "map50_95_M": scores["map50_95_M"],
        "map50_95_B": scores["map50_95_B"],
        "score": _family_score(family, scores, path),
        "note": str(data.get("note") or ""),
    }


def _champion_family(recorded, run_dir: str, path: str) -> str:
    """The metric family of a champion: ``recorded`` (the file's
    metric_family) when it is 'mask' or 'box', else the family of the
    run's task as read from disk (metric_family_for_task of
    task_for_run_dir), which is how the two older shapes are filed. A
    recorded value that is neither family is logged before the fallback."""
    if recorded is not None:
        if isinstance(recorded, str) and recorded in CHAMPION_SCORE_KEYS:
            return recorded
        _LOG.warning("champion record %s has an unknown metric_family %r; deriving it from the run",
                     path, recorded)
    return metric_family_for_task(task_for_run_dir(run_dir))


def _family_score(family: str, scores: dict, path: str) -> float | None:
    """The score that belongs to ``family`` out of ``scores`` (map50_95_M
    and map50_95_B, each a finite float or None). None when the family's
    key is absent; when only the other family's score is present the
    mismatch is logged, because a mask score says nothing about a box
    champion and the reverse."""
    own = scores.get(CHAMPION_SCORE_KEYS.get(family, ""))
    if own is None:
        others = [k for k, v in scores.items() if v is not None]
        if others:
            _LOG.warning("champion record %s is a %s champion but carries only %s; score reads as None",
                         path, family, ", ".join(others))
    return own


def _finite_score(score, path: str) -> float | None:
    """``score`` as a finite float, or None when it is absent or not a finite
    number (a bool, NaN, infinity or text), with the reason logged against
    the champion file at ``path``."""
    if score is None:
        return None
    if isinstance(score, bool):
        _LOG.warning("champion record %s has a boolean map50_95_M %r", path, score)
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        _LOG.warning("champion record %s has a non-numeric map50_95_M %r", path, score)
        return None
    if not math.isfinite(value):
        _LOG.warning("champion record %s has a non-finite map50_95_M %r", path, score)
        return None
    return value


def load_champion(inprocess_root: str) -> dict | None:
    """The module-wide champion at <inprocess>/_models/champion.json as the
    normalised record read_champion_file returns, or None when there is
    none. Needs no Flask context."""
    return read_champion_file(champion_path(inprocess_root or ""))


def save_champion(inprocess_root, run_dir):
    """Write _models/champion.json atomically and return the record."""
    record = {"run_dir": os.path.abspath(str(run_dir)), "at": ast_now()}
    path = champion_path(inprocess_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(record, fh, indent=2)
    os.replace(tmp, path)
    return record


def mirror_champion_to_lab(inprocess_root, run_dir, record):
    """Copy the champion record into the run's own ledger dir when the run
    belongs to a lab under <inprocess>/_models. evaluate_run.py reads
    <rounds_dir>/champion.json to choose its gate baseline, so a promotion
    has to reach that file too. Returns the path written, or ''.

    Containment is decided with _contained on realpath-resolved paths, so a
    lab folder symlinked out of _models is refused. The _models root itself
    is refused too: a run at _models/runs/<name> would resolve its ledger to
    the root, and the champion.json there is the module-wide record that
    save_champion owns. A project run is left alone: its ledger is the
    pipeline's, and the orchestrator's own promote route owns that file.
    """
    ledger = rounds_dir_for_run(run_dir)
    if not ledger:
        return ""
    mroot = os.path.realpath(models_root(inprocess_root or ""))
    real_ledger = os.path.realpath(ledger)
    if real_ledger == mroot or not _contained(real_ledger, mroot):
        return ""
    if not os.path.isdir(ledger):
        return ""
    path = os.path.join(ledger, CHAMPION_FILE_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(record), fh, indent=2)
    os.replace(tmp, path)
    return path


# ── Comparability ────────────────────────────────────────────────────────

def metric_family_for_task(task: str) -> str:
    """The metric family a task is scored in: METRIC_FAMILY_MASK ('mask')
    for a segmenter, METRIC_FAMILY_BOX ('box') for a detector, '' for
    anything else. Mask and box scores are never ranked against each other."""
    return _TASK_TO_METRIC_FAMILY.get(str(task or ""), "")


def run_split_record_path(run_dir: str | None) -> str:
    """Where one run keeps its own split record: RUN_SPLIT_RECORD_NAME inside
    the run folder.

    Parameters: ``run_dir``, the run folder, or None where a caller has no run
    in hand. Returns the full path, or '' for an empty or missing run folder,
    so no caller can turn a missing value into a record path in whatever
    directory the process happens to stand in.
    """
    clean = str(run_dir or "").strip()
    if not clean:
        return ""
    return os.path.join(clean, RUN_SPLIT_RECORD_NAME)


def _dataset_split_fingerprint(dataset_dir: str) -> tuple[str, str]:
    """(fingerprint, scheme) a dataset folder carries RIGHT NOW, read from the
    SPLIT_FINGERPRINT_FIELD and SPLIT_FINGERPRINT_SCHEME_FIELD of
    <dataset_dir>/SPLIT_MANIFEST_NAME.

    Parameters: ``dataset_dir``, the dataset folder. Returns ('', '') when the
    manifest is missing or unreadable (logged), is not a JSON object (logged),
    or either field is absent, not a string (logged) or blank.

    This is what the dataset holds today, which is the split a run trained on
    only at the moment that run is trained: the folder is rebuilt whenever its
    training set changes. Every other reader takes the run's own record
    instead, through split_fingerprint_for_run.
    """
    unknown = ("", "")
    if not dataset_dir:
        return unknown
    path = os.path.join(dataset_dir, SPLIT_MANIFEST_NAME)
    if not os.path.isfile(path):
        return unknown
    try:
        with open(path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        _LOG.warning("cannot read the split manifest %s: %s", path, exc)
        return unknown
    if not isinstance(manifest, dict):
        _LOG.warning("split manifest %s is not a JSON object", path)
        return unknown
    fingerprint = manifest.get(SPLIT_FINGERPRINT_FIELD)
    scheme = manifest.get(SPLIT_FINGERPRINT_SCHEME_FIELD)
    if fingerprint is None or scheme is None:
        return unknown
    if not isinstance(fingerprint, str) or not isinstance(scheme, str):
        _LOG.warning("split manifest %s: %s and %s must be strings, got %r and %r",
                     path, SPLIT_FINGERPRINT_FIELD, SPLIT_FINGERPRINT_SCHEME_FIELD, fingerprint, scheme)
        return unknown
    if not fingerprint.strip() or not scheme.strip():
        return unknown
    return fingerprint.strip(), scheme.strip()


def _read_run_split_record(run_dir: str | None) -> tuple[str, str]:
    """(fingerprint, scheme) out of a run's own split record.

    Parameters: ``run_dir``, the run folder, or None. Returns ('', '') when the run has
    no record at all, and ('', '') with the reason logged when the record
    cannot be read, is not a JSON object, or carries a fingerprint or a scheme
    that is not a non-empty string. ('', '') on its own does not mean the
    split is unknown: only the caller knows whether a run with no record may
    still have one filled in, so it checks run_split_record_path itself.
    """
    unknown = ("", "")
    path = run_split_record_path(run_dir)
    if not path or not os.path.isfile(path):
        return unknown
    try:
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        _LOG.warning("cannot read the split record %s: %s", path, exc)
        return unknown
    if not isinstance(record, dict):
        _LOG.warning("split record %s is not a JSON object", path)
        return unknown
    fingerprint = record.get(SPLIT_FINGERPRINT_FIELD)
    scheme = record.get(SPLIT_FINGERPRINT_SCHEME_FIELD)
    if not isinstance(fingerprint, str) or not isinstance(scheme, str) \
            or not fingerprint.strip() or not scheme.strip():
        _LOG.warning("split record %s carries no usable %s and %s (got %r and %r); "
                     "this run reports an unrecorded split until the file is repaired",
                     path, SPLIT_FINGERPRINT_FIELD, SPLIT_FINGERPRINT_SCHEME_FIELD,
                     fingerprint, scheme)
        return unknown
    return fingerprint.strip(), scheme.strip()


def _is_record_tmp(name: str) -> bool:
    """True when one directory entry is a temporary file of a split record
    write.

    Parameters: ``name``, one entry name as os.listdir gives it, never a path.
    Returns True only for the shape _write_run_split_record creates
    (RUN_SPLIT_RECORD_NAME, the writer's process and thread, then
    RUN_SPLIT_RECORD_TMP_SUFFIX), so the record itself and every file the
    training wrote are left alone.
    """
    return (name.startswith(RUN_SPLIT_RECORD_NAME + ".")
            and name.endswith(RUN_SPLIT_RECORD_TMP_SUFFIX))


def _sweep_stale_record_tmp(folder: str, now: float) -> None:
    """Remove split record temporaries a killed writer left in ``folder``.

    Parameters: ``folder``, the run folder; ``now``, the clock in seconds
    since the epoch, passed in so the age test is one comparison against one
    reading. Returns None.

    A process that dies between the open and the move leaves its temporary
    file in the run folder, and nothing else would ever remove it. Only a file
    older than STALE_RECORD_TMP_AGE_SECONDS is swept, so the temporary another
    writer is holding open right now (milliseconds old) is never taken from
    under it. A leftover another writer swept first is simply gone; a folder
    that cannot be listed, or a file that cannot be removed, is left as it is
    with the reason logged, because a leftover costs bytes and writing past an
    error nobody saw costs data.
    """
    try:
        names = os.listdir(folder)
    except OSError as exc:
        _LOG.warning("cannot list %s to sweep stale split record temporaries: %s", folder, exc)
        return
    for name in names:
        if not _is_record_tmp(name):
            continue
        stale = os.path.join(folder, name)
        try:
            if now - os.path.getmtime(stale) < STALE_RECORD_TMP_AGE_SECONDS:
                continue
            os.remove(stale)
        except FileNotFoundError:
            # Another writer swept the same leftover between the listing and
            # here. Nothing is wrong: the file is gone, which is the point.
            continue
        except OSError as exc:
            _LOG.warning("cannot remove the stale split record temporary %s: %s", stale, exc)
            continue
        _LOG.info("removed the stale split record temporary %s", stale)


def _split_record_document(fingerprint: str, scheme: str, source: str,
                           dataset_dir: str) -> dict:
    """One split record in its published shape, ready to be written.

    Parameters: ``fingerprint`` and ``scheme``, the pair read from the
    dataset's split manifest; ``source``, one of SPLIT_RECORD_SOURCES, naming
    the occasion that recorded it; ``dataset_dir``, the dataset folder the
    pair was read from. Returns the record as a dict whose keys are exactly
    RUN_SPLIT_RECORD_FIELDS, stamped with ast_now (Atlantic Standard Time,
    24 hour, the shape every timestamp in this module carries).

    Every key here is the same constant RUN_SPLIT_RECORD_FIELDS is built
    from, so a caller reading that tuple always sees what this writes.
    """
    return {
        SPLIT_FINGERPRINT_FIELD: fingerprint,
        SPLIT_FINGERPRINT_SCHEME_FIELD: scheme,
        RUN_SPLIT_RECORDED_AT_FIELD: ast_now(),
        RUN_SPLIT_RECORDED_SOURCE_FIELD: source,
        RUN_SPLIT_DATASET_DIR_FIELD: os.path.abspath(dataset_dir),
    }


def _restore_folder_times(folder: str, before: os.stat_result | None) -> None:
    """Put a folder's access and modification times back to what they were.

    Parameters: ``folder``, the folder that was written to; ``before``, the
    os.stat_result read before the write, or None when it could not be read.
    Returns None. A failure is logged and nothing is raised: a card that sorts
    one run by the wrong time is a smaller fault than a split record that
    refuses to be written.
    """
    if before is None:
        return
    try:
        os.utime(folder, ns=(before.st_atime_ns, before.st_mtime_ns))
    except OSError as exc:
        _LOG.warning("cannot put back the times of %s after recording its split, so the Trained "
                     "runs card sorts that run as though it changed now: %s", folder, exc)


def _write_run_split_record(path: str, record: dict) -> bool:
    """Write one split record to ``path``, replacing whatever is there, and
    leave the run folder's own times exactly as they were.

    Parameters: ``path``, the record file inside the run folder; ``record``,
    the fields to write, as _split_record_document builds them.
    Returns True when the file is in place, and False with the reason logged
    when it cannot be written (a read-only run folder, a full disk).

    The write goes to a temporary file named for this process and thread and
    is then moved over the target, so two readers stamping the same run at
    once each write a whole file and the loser's leftover never appears as a
    half-written record. Temporaries older than STALE_RECORD_TMP_AGE_SECONDS,
    left by a writer that was killed mid-write, are swept first.

    Creating the record, sweeping a temporary and moving a file over the
    target each change the modification time of the run folder that holds
    them, and that time is the only sort key of the Trained runs card (newest
    first). The folder's access and modification times are therefore read
    before the write and put back after it, so recording a split never
    reorders the card.
    """
    folder = os.path.dirname(path)
    try:
        before = os.stat(folder)
    except OSError as exc:
        before = None
        _LOG.warning("cannot read the times of %s before recording its split, so that folder will "
                     "carry the time of this write: %s", folder, exc)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}{RUN_SPLIT_RECORD_TMP_SUFFIX}"
    try:
        _sweep_stale_record_tmp(folder, time.time())
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        os.replace(tmp, path)
        return True
    except OSError as exc:
        _LOG.warning("cannot write the split record %s: %s", path, exc)
        try:
            os.remove(tmp)
        except FileNotFoundError:
            # The write failed before the file was created. Nothing is left.
            pass
        except OSError as cleanup_exc:
            _LOG.warning("the temporary file %s of that failed write could not be removed either; "
                         "it is swept by the next write, once it is an hour old: %s",
                         tmp, cleanup_exc)
        return False
    finally:
        _restore_folder_times(folder, before)


def _newest_run_file_mtime(run_dir: str) -> float:
    """When anything inside a run folder was last written, in seconds since
    the epoch.

    Parameters: ``run_dir``, the run folder. Returns the newest modification
    time among the files it holds at any depth (weights/best.pt included), or
    -1.0 when it holds no file at all, or none whose time can be read. The
    run's own split record and the temporaries of a record write are skipped:
    they are written by whoever reads the run, not by the training that made
    it, so counting them would let a read vouch for itself.

    What a finished run leaves behind is training's own work: ultralytics
    writes args.yaml when training starts and results.csv and the weights as
    it goes, and an imported run carries the provenance.json written when it
    was copied in. Nothing writes into a run folder after that, so the newest
    file is when the run stopped being written.
    """
    clean = str(run_dir or "").strip()
    if not clean:
        return -1.0
    root = os.path.normpath(clean)
    newest = -1.0
    for folder, _dirs, names in os.walk(root):
        for name in names:
            if folder == root and (name == RUN_SPLIT_RECORD_NAME or _is_record_tmp(name)):
                continue
            entry = os.path.join(folder, name)
            try:
                mtime = os.path.getmtime(entry)
            except OSError as exc:
                # One entry whose time cannot be read (a dangling symlink, a
                # file removed while the folder was being walked) is skipped,
                # not fatal: the answer is still the newest of the rest, and
                # the caller logs the verdict it reaches. Debug, because a run
                # folder full of them would otherwise flood the log.
                _LOG.debug("cannot read the time of %s: %s", entry, exc)
                continue
            newest = max(newest, mtime)
    return newest


def _dataset_predates_run(dataset_dir: str, run_dir: str) -> tuple[bool, str]:
    """(True, why) when the dataset beside a run was already in place the last
    time that run wrote anything, (False, why) when it was not.

    Being already in place is what makes a dataset's split manifest evidence
    of the split the run trained on. A dataset folder is rebuilt where it
    stands whenever its training set changes, so a manifest written after the
    run stopped can name a split that run never saw.

    Parameters: ``dataset_dir``, the dataset folder beside the run;
    ``run_dir``, the run folder. Returns False when either time cannot be read
    (an unreadable manifest, a run folder holding no file), and ``why`` as one
    plain sentence naming both times in AST, fit for a log line.

    The one window this cannot see is a rebuild during training, which leaves
    a manifest newer than args.yaml but older than the last weights written.
    Recording the split as the run finishes (record_split_fingerprint) closes
    that window; nothing a later reader can measure does.
    """
    manifest = os.path.join(dataset_dir, SPLIT_MANIFEST_NAME)
    try:
        built = os.path.getmtime(manifest)
    except OSError as exc:
        _LOG.warning("cannot read the time of the split manifest %s: %s", manifest, exc)
        return False, f"the time of {manifest} cannot be read"
    newest = _newest_run_file_mtime(run_dir)
    if newest < 0:
        return False, f"{run_dir} holds no file whose time can be read"
    when = (f"the dataset was built at {_ast_stamp(built)} and the run last wrote "
            f"at {_ast_stamp(newest)}")
    return built <= newest, when


def _ast_stamp(epoch_seconds: float) -> str:
    """One epoch time as this platform writes times: AST, 24 hour, with the
    zone named.

    Parameters: ``epoch_seconds``, seconds since the epoch as os.path.getmtime
    returns them. Returns for example '2026-09-16 01:27:08 AST'. Used in log
    lines, where the reader has no other way to tell which zone a file time
    was read in.
    """
    return datetime.fromtimestamp(epoch_seconds, AST).strftime("%Y-%m-%d %H:%M:%S") + " AST"


def record_split_fingerprint(run_dir: str | None,
                             source: str = SPLIT_RECORD_SOURCE_TRAINING) -> tuple[str, str]:
    """Record the split one run was trained on into that run's own folder and
    return the recorded (fingerprint, scheme).

    Call this once, as soon as a run finishes training, while the dataset
    beside it still holds the split the run was scored on. The value is read
    from that dataset's split manifest (_dataset_split_fingerprint) and
    written to run_split_record_path(run_dir); every later read takes the
    recorded value, so rebuilding the dataset cannot move what an older run
    claims. This is the only call that is certain of what it records: every
    other path has to infer it from file times after the fact.

    Parameters: ``run_dir``, the run folder, or None; ``source``, one of
    SPLIT_RECORD_SOURCES, naming the occasion that wrote the record.
    Returns ('', '') when there is nothing to record: no run folder on disk,
    no dataset beside it, or a dataset whose manifest carries no usable
    fingerprint. Any record already there is replaced, so the caller, not this
    function, decides when a run is stamped.
    Raises ValueError for a ``source`` outside SPLIT_RECORD_SOURCES, which is
    a wiring mistake rather than a state on disk.

    A record that cannot be written is logged and its fingerprint is still
    returned, so a run folder nobody may write to degrades to the old live
    read of the dataset instead of losing its split entirely.
    """
    if source not in SPLIT_RECORD_SOURCES:
        raise ValueError(f"a split record's source must be one of {SPLIT_RECORD_SOURCES}, got {source!r}")
    path = run_split_record_path(run_dir)
    if not path or not os.path.isdir(os.path.dirname(path)):
        return "", ""
    dataset_dir = dataset_for_run(run_dir)
    fingerprint, scheme = _dataset_split_fingerprint(dataset_dir)
    if not fingerprint:
        return "", ""
    _write_run_split_record(path, _split_record_document(fingerprint, scheme, source, dataset_dir))
    return fingerprint, scheme


def split_fingerprint_for_run(run_dir: str | None) -> tuple[str, str]:
    """(fingerprint, scheme) of the split a run was really trained on.

    Parameters: ``run_dir``, the run folder, or None.

    A run that carries its own split record reports that pair, and reports it
    for the rest of its life: the dataset beside it is never read again, so
    rebuilding the training set cannot move what the run claims.

    A run trained before records existed carries none, and its split is read
    from the dataset still beside it. That reading is recorded onto the run
    only when the dataset was already in place the last time the run wrote
    anything (_dataset_predates_run). A manifest written after the run stopped
    belongs to a rebuild, and a rebuild can change the split, so that pair is
    reported for the card and deliberately not recorded: freezing it would put
    a split the run may never have seen beyond the reach of every later
    correction, including the correct one.

    Returns ('', '') when there is nothing to read: no run folder, no dataset
    beside it, no fingerprint in its manifest, or a record on disk that is
    damaged (logged, and left untouched so whoever repairs it can still see
    it). ('', '') means the split was not recorded, and an unrecorded split is
    never comparable.

    The certain value comes from recording the split as the run finishes
    training, with record_split_fingerprint(run_dir): while a run is being
    trained, the dataset beside it still holds the split it is scored on.
    """
    fingerprint, scheme = _read_run_split_record(run_dir)
    if fingerprint:
        return fingerprint, scheme
    path = run_split_record_path(run_dir)
    if not path or os.path.isfile(path):
        # Either no run folder to read at all, or a record that is there and
        # unusable. A damaged record is left exactly as it is: a read must
        # never write over the only copy of what somebody has to repair.
        return "", ""
    if not os.path.isdir(os.path.dirname(path)):
        return "", ""
    dataset_dir = dataset_for_run(run_dir)
    fingerprint, scheme = _dataset_split_fingerprint(dataset_dir)
    if not fingerprint:
        return "", ""
    predates, why = _dataset_predates_run(dataset_dir, run_dir)
    if predates:
        _write_run_split_record(path, _split_record_document(
            fingerprint, scheme, SPLIT_RECORD_SOURCE_FIRST_READ, dataset_dir))
        return fingerprint, scheme
    _LOG.warning("the split of the run at %s is not recorded on it: the dataset at %s may have "
                 "been rebuilt since that run trained (%s). VEGA reports the fingerprint the "
                 "dataset holds now and records nothing, because a rebuild can change the split. "
                 "Record the split when a run finishes training to close this.",
                 run_dir, dataset_dir, why)
    return fingerprint, scheme


def runs_comparable(run_dir_a: str, run_dir_b: str) -> tuple[bool, str]:
    """(True, why) when two runs' scores may be ranked against each other,
    (False, why) when they may not.

    Two runs are comparable only when they score in the same metric family
    (box against box, mask against mask; see metric_family_for_task) and
    trained on the same split, shown by an identical split fingerprint under
    the same hashing scheme, each taken from the run's own recorded split
    (split_fingerprint_for_run). A run whose split was never recorded is never
    comparable, itself included: without it nobody can tell whether the two
    were scored on the same held-out images.
    A run folder that does not exist is refused by name. ``why`` is one plain
    sentence fit for a table cell.
    """
    for run_dir in (run_dir_a, run_dir_b):
        if not run_dir or not os.path.isdir(str(run_dir)):
            return False, f"run folder not found: {run_dir}"
    family_a = metric_family_for_task(task_for_run_dir(run_dir_a))
    family_b = metric_family_for_task(task_for_run_dir(run_dir_b))
    if family_a != family_b:
        return False, f"different metric families ({family_a} against {family_b})"
    split_a = split_fingerprint_for_run(run_dir_a)
    split_b = split_fingerprint_for_run(run_dir_b)
    for run_dir, split in ((run_dir_a, split_a), (run_dir_b, split_b)):
        if not split[0]:
            return False, (f"split fingerprint unknown for {os.path.basename(os.path.normpath(run_dir))}; "
                           "runs on an unrecorded split cannot be compared")
    if split_a[0] != split_b[0]:
        return False, "different splits (the split fingerprints differ)"
    if split_a[1] != split_b[1]:
        return False, "different splits (the same fingerprint under different hashing schemes)"
    return True, f"same {family_a} metrics on the same split"


# ── Background job runner ────────────────────────────────────────────────

_LOG_MAXLEN = 20000


class LabRunner:
    """One background job at a time (train, evaluate or infer).

    Deliberately small: the orchestrator's StageRunner manages a whole
    pipeline keyed by step and is bound to orchestrator_config, so the lab
    keeps its own instance rather than importing the app. The liveness idea is
    copied verbatim: every status read polls the process, and a job whose
    process has exited flips out of "running" on that read, with its exit
    code, instead of leaving a stale claim on the page.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._log = deque(maxlen=_LOG_MAXLEN)
        self._state = {
            "phase": "idle",
            "kind": "",
            "lab_name": "",
            "run_name": "",
            "run_dir": "",
            "out_dir": "",
            "exit_code": None,
            "started_at": "",
            "message": "",
        }

    # -- internals --------------------------------------------------
    def _capture(self, proc, log):
        stream = proc.stdout
        if stream is None:
            return
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                log.append(line.rstrip("\n"))
        except (ValueError, OSError):
            pass

    def _reconcile(self):
        """Called under the lock on every read: a job whose process has
        exited leaves the running phase here, never on a later write."""
        proc = self._proc
        if proc is None or self._state["phase"] != "running":
            return
        code = proc.poll()
        if code is None:
            return
        self._state["exit_code"] = code
        self._state["phase"] = "done" if code == 0 else "error"
        if code != 0:
            self._state["message"] = f"{self._state['kind'] or 'job'} exited with code {code}"
        else:
            self._state["message"] = f"{self._state['kind'] or 'job'} finished"

    # -- public -----------------------------------------------------
    def is_busy(self):
        with self._lock:
            self._reconcile()
            return self._state["phase"] == "running"

    def busy_message(self):
        with self._lock:
            name = self._state["lab_name"] or self._state["run_name"] or "a job"
            kind = self._state["kind"] or "job"
        return (f"another job is already running ({kind}: {name}). Wait for it to finish, "
                "or press Stop in Progress.")

    def start(self, cmd, *, kind, lab_name="", run_name="", run_dir="", out_dir="",
              cwd=None, env_extra=None):
        """Launch one job. Returns (info, '') or (None, reason)."""
        with self._lock:
            self._reconcile()
            if self._state["phase"] == "running":
                return None, "busy"
            env = os.environ.copy()
            if env_extra:
                env.update({k: str(v) for k, v in env_extra.items()})
            self._log = deque(maxlen=_LOG_MAXLEN)
            try:
                proc = subprocess.Popen(
                    list(cmd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=cwd,
                    env=env,
                    text=True,
                    start_new_session=True,
                )
            except Exception as exc:
                self._state.update(phase="error", kind=kind, lab_name=lab_name,
                                   run_name=run_name, exit_code=None,
                                   message=f"could not start {kind}: {exc}")
                return None, str(exc)
            self._proc = proc
            self._state.update(phase="running", kind=kind, lab_name=lab_name,
                               run_name=run_name, run_dir=run_dir, out_dir=out_dir,
                               exit_code=None, started_at=ast_now(),
                               message=f"{kind} running")
            log = self._log
        threading.Thread(target=self._capture, args=(proc, log), daemon=True).start()
        return {"pid": proc.pid, "started_at": self._state["started_at"]}, ""

    def status(self, tail=12):
        with self._lock:
            self._reconcile()
            state = dict(self._state)
            lines = list(self._log)[-int(tail or 0):] if tail else []
        state["log_tail"] = lines
        return state

    def log(self, lines=200):
        with self._lock:
            self._reconcile()
            window = list(self._log)
            phase = self._state["phase"]
        try:
            n = int(lines)
        except (TypeError, ValueError):
            n = 200
        n = max(1, min(n, _LOG_MAXLEN))
        return {"lines": window[-n:], "phase": phase, "total": len(window)}

    def stop(self):
        """Stop the running job. A deliberate stop reports idle, never an
        error: the orchestrator learned the hard way that a killed process
        reconciled as a crash reads as a bug to whoever pressed the button."""
        with self._lock:
            proc = self._proc
            # Reconcile FIRST: a job that finished or crashed between polls
            # must keep its own outcome and exit code. Stamping "stopped by
            # you" over a crash hides the failure from whoever pressed Stop.
            self._reconcile()
            already_gone = proc is None or proc.poll() is not None
            phase = self._state["phase"]
            exit_code = self._state["exit_code"]
        if already_gone:
            return {"stopped": False, "phase": phase, "exit_code": exit_code}

        killed = False
        try:
            # start_new_session=True makes the child its own group leader,
            # so pgid == pid identifies OUR group. Anything else means the
            # pid is not ours to signal: fall back to the process only.
            pgid = os.getpgid(proc.pid)
            if pgid == proc.pid:
                os.killpg(pgid, signal.SIGTERM)
                killed = True
        except Exception:
            killed = False
        if not killed:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        with self._lock:
            self._proc = None
            self._state.update(phase="idle", exit_code=None,
                               message=f"stopped by you at {ast_now()}")
            phase = self._state["phase"]
        return {"stopped": True, "phase": phase}

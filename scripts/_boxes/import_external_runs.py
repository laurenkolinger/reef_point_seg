#!/usr/bin/env python
"""Import finished training runs from outside VEGA as a model lab.

Module:  scripts/_boxes/import_external_runs.py
Purpose: Turn ultralytics detection runs trained outside this module (an
         experiment run from a shell, on a laptop, or by a collaborator) into
         one lab under inprocess/_models/<lab>/, so the VEGA main window lists
         them on the Train step's runs card, the Evaluate step scores them and
         the Promote step can make one of them the champion. The lab's dataset
         is rebuilt from the staged dataset those runs were trained on, and,
         when that staged folder carries split_pins.json, on exactly the split
         the original experiment used, so an imported run and a VEGA retrain
         are scored on the same held-out images.
Inputs:  --dataset (a staged box dataset under supporting_data/bbox_datasets),
         --lab (the lab folder name), --runs (one or more finished run folders
         holding weights/best.pt and args.yaml). Optional: --inprocess-root,
         --note, --allow-fresh-split, --quiet.
Outputs: <inprocess>/_models/<lab>/ holding lab.json (the lab name, the task,
         the staged source and its kind, origin_kind "imported", the note and
         the imported run names), dataset/ (the split this lab is scored on)
         and runs/<run name>/ with weights/best.pt, args.yaml, results.csv when
         the source had one, and provenance.json. Every path written into a
         record is absolute. The function returns what it wrote.

No score is invented here. The importer writes no rounds.csv row and no
champion record: a score reaches the ledger only from VEGA's own Evaluate step,
which measures it and records the metric family and the split fingerprint
beside it. A number written here would be a number VEGA never measured.

Nothing is written until every check has passed, and the lab folder itself is
the claim: two imports of one lab name cannot both write, and any failure after
the claim removes the whole folder, because a half-written lab is worse than
none.

This imagery is outside the TCRMP survey. The staged dataset is read where it
is staged and every artifact stays under _models/: nothing here writes to
inprocess/run_*, the mask registry, the coverage matrix, or STATS.

Usage:
  env/bin/python scripts/_boxes/import_external_runs.py \
      --dataset supporting_data/bbox_datasets/fish_photo_28feb25 \
      --lab fish_photo_28feb25_paper \
      --runs <experiment>/training_runs/run3_heavy_augment \
             <experiment>/training_runs/run6_1024px_augment
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from collections import namedtuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
REPO_DIR = os.path.dirname(_SCRIPTS)
MODULE_DIR = os.path.dirname(REPO_DIR)

# The two libraries this tool leans on live outside the package: the shared run
# library (_modellab/labruns.py) owns the lab layout and the lab-name rule, and
# the detection dataset builder (TCRMPtrain_oceankindCV/src/detect_dataset.py)
# owns the split. Both are loaded the way the rest of this package loads its
# neighbours: the scripts folder is put on the path for the package import, and
# the training tool, which is a script folder rather than a package, is loaded
# by file path under a private name. Neither import starts a process, reads a
# dataset, or pulls Flask along.
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _modellab import labruns  # noqa: E402

_DETECT_DATASET_PATH = os.path.join(_SCRIPTS, "TCRMPtrain_oceankindCV", "src",
                                    "detect_dataset.py")


def _load_detect_dataset():
    """The shared split builder, loaded from its file at import.

    Eager and hard-failing on purpose, the way labruns loads the same file: it
    is a sibling in this checkout, so its absence means a broken install, and a
    lazy load would surface that breakage half way through an import that has
    already claimed a lab folder.

    Returns:
      the loaded detect_dataset module.
    Raises:
      ImportError naming the path when the loader refuses the file;
      FileNotFoundError naming the path when the file is not there.
    """
    spec = importlib.util.spec_from_file_location("_boxes_detect_dataset",
                                                  _DETECT_DATASET_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the detection dataset builder at "
                          f"{_DETECT_DATASET_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


detect_dataset = _load_detect_dataset()

# Where the module keeps its projects and its labs when the caller names no
# other root. The orchestrator resolves the same folder (app.py, projects_dir).
DEFAULT_INPROCESS_ROOT = os.path.join(MODULE_DIR, "inprocess")

# The held-out fractions an imported lab builds with. They reach only images
# the staged dataset does not pin, so a fully pinned dataset (the case this
# tool exists for) ignores them. The values are the fish preset's
# (config/presets/fish_heavy_augment_640.yaml) valid_ratio and test_ratio, so
# an unpinned image is split the same way by a VEGA retrain only when that
# project applied that preset: a project left on the Train panel's own fallback
# splits it 0.1 and 0.1 (detect_dataset.DEFAULT_VALID_FRAC and
# DEFAULT_TEST_FRAC, which these names deliberately do not shadow, because the
# split builder publishes those as the module's one source of truth for the
# fallback and these are a different number for a different caller).
IMPORT_VALID_FRAC = 0.2
IMPORT_TEST_FRAC = 0.1

# What an imported lab holds. Every run in one lab is a detector trained on one
# box dataset: the task is written into lab.json so a run whose own args.yaml
# names no task still reads as a detector on the runs card.
LAB_TASK = "detect"

# The kind of training source the lab was built from. This is the meaning
# source_kind carries everywhere else in the module (the two live labs on disk,
# _reefreview/training_projects.SOURCE_KINDS, the staged dataset manifests and
# the Bounding Boxes card), and an imported detection lab is always built from
# a staged bounding-box dataset, so it is the one value written here. Where the
# runs themselves came from is a different fact and gets its own key below,
# rather than a second vocabulary inside one key.
LAB_SOURCE_KIND = "bbox"

# Who wrote this lab and where its runs were trained. A lab written by the
# retired Train + Infer dashboard carries no origin_kind, so its absence reads
# as a lab trained in this module.
LAB_ORIGIN_KIND = "imported"

# The run folder name becomes a folder name under the lab, a run_name in the
# evaluation ledger, an evaluation folder name and a value in the runs card's
# links, so it stays inside the plain-name rule the rest of the module uses for
# a lab, a dataset and a set of initials. A leading dot is out: a run folder
# nobody can see in a listing is a run folder nobody can act on. The cap is the
# module's own name cap (labruns.LAB_NAME_RE), which is also where
# labruns.safe_note truncates an evaluation folder name: two longer names that
# agree on their first RUN_NAME_MAX_CHARS characters would give two evaluation
# folders a reader cannot tell apart.
RUN_NAME_MAX_CHARS = 60
RUN_NAME_RE = re.compile(
    rf"^[A-Za-z0-9][A-Za-z0-9._-]{{0,{RUN_NAME_MAX_CHARS - 1}}}$")

# What a finished ultralytics run must carry to be worth importing: the weights
# the evaluator loads, and the arguments file the runs card reads the task and
# the image size from.
WEIGHTS_REL = os.path.join("weights", "best.pt")
ARGS_YAML_NAME = "args.yaml"
RESULTS_CSV_NAME = "results.csv"
PROVENANCE_NAME = "provenance.json"
DATA_YAML_NAME = "data.yaml"

# The args.yaml lines this tool reads. The task decides whether the run belongs
# in a detection lab at all; data and save_dir are recorded in provenance
# because they name folders on the machine the run was trained on, which no
# longer exist here and which VEGA never reads (the evaluator is given the
# dataset explicitly).
ARGS_TASK_KEY = "task"
ARGS_KEYS_RECORDED = ("data", "save_dir")

# args.yaml task values this importer accepts. An empty line is accepted too:
# the lab record then says what the run is. Anything else is refused rather
# than imported, because a segmentation run scored on a box dataset would give
# a mask-family number nobody can rank against the box scores beside it.
ACCEPTED_ARGS_TASKS = ("detect",)

# How much of a weights file is read at a time when it is hashed. A detection
# checkpoint is about 51 MB, so it is read in pieces rather than into memory.
HASH_CHUNK_BYTES = 1 << 20

# The longest note the lab record and every provenance record carry. A note is
# one plain sentence about where these runs came from, not a report.
NOTE_MAX_CHARS = 500

# What the command line answers when it refuses.
EXIT_REFUSED = 2

# One source run: the name it is imported under, the path the operator named,
# the path that name resolves to, the args.yaml lines already read from it, and
# whether it carries a results.csv.
RunSource = namedtuple("RunSource", "name given real args has_results")


class ImportRefused(RuntimeError):
    """Anything that stops an import, either before it writes the lab or
    before the lab is complete. The message names what failed, where, and
    why, and is written to be read by the operator who ran the tool."""


# -- small helpers --------------------------------------------------------

def sha256_of(path):
    """The SHA-256 digest of a file, read in HASH_CHUNK_BYTES pieces.

    Parameters:
      path: the file to hash.
    Returns:
      the digest as 64 lower-case hex characters.
    Raises:
      ImportRefused naming the file when it cannot be read.
    """
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ImportRefused(f"{path} could not be read to check the weights "
                            f"against their source ({exc})") from exc
    return digest.hexdigest()


def _args_lines(args_path, keys):
    """The values of top-level ``key:`` lines in one ultralytics args.yaml.

    ultralytics writes args.yaml as flat ``key: value`` lines, so a line scan
    is enough and keeps this tool free of a YAML dependency for a file it only
    reads three lines of. Only a line starting at column one counts; an
    indented key belongs to a nested block. Surrounding quotes are stripped.
    The shared run library scans the same way (labruns._args_yaml_field), but
    that function takes a finished run folder inside the module, and this one
    reads a file in an experiment folder that is not a lab run yet.

    Parameters:
      args_path: the args.yaml to read.
      keys: the top-level keys wanted.
    Returns:
      {key: value} with '' for a key the file does not carry.
    Raises:
      ImportRefused naming the file when it cannot be read as UTF-8 text.
    """
    found = {key: "" for key in keys}
    try:
        with open(args_path, encoding="utf-8") as fh:
            for line in fh:
                for key in keys:
                    prefix = key + ":"
                    if not found[key] and line.startswith(prefix):
                        found[key] = line[len(prefix):].strip().strip("'\"")
    except UnicodeDecodeError as exc:
        raise ImportRefused(f"{args_path} is not UTF-8 text ({exc}); only a "
                            "finished ultralytics run, whose args.yaml this "
                            "tool reads the task and the image size from, can "
                            "be imported") from exc
    except OSError as exc:
        raise ImportRefused(f"{args_path} could not be read ({exc}); it is "
                            "where the run says what it trained") from exc
    return found


def _absolute(path):
    """One filesystem path, expanded and made absolute.

    Every path this tool records or reports is absolute. A provenance record is
    the archival account of where a model came from, and a progress line is
    read by whoever comes after, so a path that resolves only from the folder
    the operator happened to stand in is not an answer either of them can use.

    Parameters:
      path: the path as it was given.
    Returns:
      the expanded, absolute, normalised path. A trailing separator is dropped,
      because a staged folder's own name is the split seed. Symbolic links are
      left in place: resolving them is a separate step (os.path.realpath),
      recorded separately.
    """
    return os.path.abspath(os.path.expanduser(str(path).strip()))


def _write_json(path, doc):
    """Write one JSON record, pretty printed, with a closing newline.

    Parameters:
      path: the file to write.
      doc: the object to serialise.
    Returns:
      None.
    Raises:
      ImportRefused naming the file when it cannot be written.
    """
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except OSError as exc:
        raise ImportRefused(f"{path} could not be written ({exc})") from exc


def _remove_claimed_lab(lab_dir, log):
    """Remove a lab folder this import claimed but could not finish.

    A half-written lab is worse than none: the runs card would list runs with
    no dataset beside them, and the next import of the same name would refuse
    because the folder is there.

    Parameters:
      lab_dir: the folder to remove.
      log: where a folder that cannot be removed is reported.
    Returns:
      None. A removal that fails is reported through ``log`` and never hides
      the error that caused the rollback.
    """
    try:
        shutil.rmtree(lab_dir)
    except FileNotFoundError:
        return
    except OSError as exc:
        log(f"the half-written lab at {lab_dir} could not be removed ({exc}); "
            "remove it by hand before importing into that name again")


# -- validation -----------------------------------------------------------

def _check_lab_name(lab_name):
    """The lab folder name, checked against the shared rule.

    Parameters:
      lab_name: the name the operator gave.
    Returns:
      the cleaned name.
    Raises:
      ImportRefused naming the fault when the name is empty, carries anything
      but letters, digits, underscore or hyphen, or is one of the two names
      the module keeps for its evaluation and inference folders.
    """
    clean, why = labruns.validate_lab_name(lab_name)
    if why:
        raise ImportRefused(f"the lab name {lab_name!r} cannot be used: {why}")
    if clean in labruns._RESERVED_LAB_NAMES:
        raise ImportRefused(
            f"the lab name {clean!r} is kept by the module for its own folder "
            f"under _models (the reserved names are "
            f"{', '.join(labruns._RESERVED_LAB_NAMES)}); pick another name")
    return clean


def _check_dataset_dir(dataset_dir):
    """The staged dataset the imported runs are scored on.

    Parameters:
      dataset_dir: the staged folder the operator named. A trailing separator
        is dropped, because the folder name is the split seed and a trailing
        separator once left that seed empty.
    Returns:
      the absolute, normalised folder path.
    Raises:
      ImportRefused when the path is empty, is not a folder, or holds no
      data.yaml, which is what makes a folder a staged box dataset.
    """
    if not isinstance(dataset_dir, str) or not dataset_dir.strip():
        raise ImportRefused("name the staged box dataset these runs were "
                            "trained on (--dataset); the lab is scored on it")
    clean = _absolute(dataset_dir)
    if not os.path.isdir(clean):
        raise ImportRefused(f"{clean} is not a folder, so it is not a staged "
                            "box dataset; stage the dataset first with "
                            "scripts/_boxes/ingest_yolo_boxes.py")
    if not os.path.isfile(os.path.join(clean, DATA_YAML_NAME)):
        raise ImportRefused(f"{clean} is not a staged box dataset: it carries "
                            f"no {DATA_YAML_NAME}, which names the classes the "
                            "runs were trained on")
    return clean


def _check_note(note):
    """The note the lab record and every provenance record carry.

    Parameters:
      note: the operator's note, a plain sentence.
    Returns:
      the note, stripped.
    Raises:
      ImportRefused when the note is not text, and when it is longer than
      NOTE_MAX_CHARS: cutting it here would drop part of the one sentence that
      says where these runs came from, and the record would read as if that was
      all the operator wrote.
    """
    if not isinstance(note, str):
        raise ImportRefused(f"the note must be text, got {type(note).__name__}; "
                            "it is one sentence saying where these runs came from")
    clean = note.strip()
    if len(clean) > NOTE_MAX_CHARS:
        raise ImportRefused(
            f"the note is {len(clean)} characters and the lab record keeps at "
            f"most {NOTE_MAX_CHARS}; shorten it to one sentence saying where "
            "these runs came from, and put the longer account in a note file "
            "under _METADATA/notes")
    return clean


def _check_run_source(run_dir):
    """One source run folder, checked before anything is written.

    Parameters:
      run_dir: the folder the operator named. A symbolic link is followed, and
        the run is imported under the name the operator named it by.
    Returns:
      a RunSource.
    Raises:
      ImportRefused when the path is empty or not text, when the name it gives
      the run cannot be a run name, when the path is not a folder, when the
      folder holds no weights/best.pt or no args.yaml, or when its args.yaml
      says it trained something other than a detector.
    """
    if not isinstance(run_dir, str) or not run_dir.strip():
        raise ImportRefused("every --runs entry must be the path of a finished "
                            f"run folder, got {run_dir!r}")
    given = run_dir.strip()
    name = os.path.basename(os.path.normpath(os.path.expanduser(given)))
    if not RUN_NAME_RE.match(name):
        raise ImportRefused(
            f"the run folder {given} would be imported as {name!r}, which "
            f"cannot be a run name: a run name carries 1 to "
            f"{RUN_NAME_MAX_CHARS} characters of letters, digits, underscore, "
            "hyphen or dot, and starts with a letter or a digit, because it "
            "names a folder, a row in the evaluation ledger and a link on the "
            "runs card. Rename the folder, or copy it under a usable name, and "
            "import that")
    real = os.path.realpath(os.path.expanduser(given))
    if not os.path.isdir(real):
        raise ImportRefused(f"the run folder {given} is not a folder "
                            f"(it resolves to {real})")
    for rel in (WEIGHTS_REL, ARGS_YAML_NAME):
        if not os.path.isfile(os.path.join(real, rel)):
            raise ImportRefused(
                f"the run folder {given} has no {rel}; only a finished "
                "ultralytics run can be imported, and VEGA needs the weights "
                "to evaluate it and the arguments file to read its task and "
                "image size")
    args = _args_lines(os.path.join(real, ARGS_YAML_NAME),
                       (ARGS_TASK_KEY,) + ARGS_KEYS_RECORDED)
    task = args[ARGS_TASK_KEY].lower()
    if task and task not in ACCEPTED_ARGS_TASKS:
        raise ImportRefused(
            f"the run folder {given} says it trained {args[ARGS_TASK_KEY]!r} in "
            f"its {ARGS_YAML_NAME}, and an imported lab holds detectors only: a "
            "run of another task scored on a box dataset would give a number "
            "nobody can rank against the box scores beside it")
    return RunSource(name=name, given=given, real=real, args=args,
                     has_results=os.path.isfile(os.path.join(real, RESULTS_CSV_NAME)))


def _check_run_sources(run_dirs):
    """Every source run folder, and the names they would be imported under.

    Parameters:
      run_dirs: the list of run folders the operator named.
    Returns:
      a list of RunSource in the order given.
    Raises:
      ImportRefused when the list is a bare string, is empty, holds a folder
      _check_run_source refuses, or holds two folders that would be imported
      under one name, where the second copy would quietly replace the first.
    """
    if isinstance(run_dirs, (str, bytes)):
        raise ImportRefused("the run folders must be given as a list of paths, "
                            "not as one string")
    try:
        given = list(run_dirs)
    except TypeError as exc:
        raise ImportRefused("the run folders must be given as a list of paths "
                            f"(got {type(run_dirs).__name__})") from exc
    if not given:
        raise ImportRefused("name at least one finished run folder to import "
                            "(--runs); an empty lab would list nothing")
    sources = []
    seen = {}
    for entry in given:
        source = _check_run_source(entry)
        if source.name in seen:
            raise ImportRefused(
                f"two run folders would both be imported as {source.name!r}: "
                f"{seen[source.name]} and {source.given}. One would replace the "
                "other in the lab, so rename one of them, or import them into "
                "two labs")
        seen[source.name] = source.given
        sources.append(source)
    return sources


# -- the import -----------------------------------------------------------

def _build_lab_dataset(dataset_dir, lab_dataset_dir, valid_frac, test_frac,
                       allow_fresh_split):
    """Build the split the lab is scored on, and refuse a split nobody chose.

    Parameters:
      dataset_dir: the staged dataset.
      lab_dataset_dir: <lab>/dataset, written whole.
      valid_frac, test_frac: the fractions applied to images the staged
        dataset does not pin.
      allow_fresh_split: True to accept a split drawn here rather than
        replayed from the staged dataset.
    Returns:
      the split manifest the builder wrote.
    Raises:
      ImportRefused when the builder refuses the source, and when the built
      split is a fresh draw and the override was not given, because a run
      scored on a holdout it was never chosen on gives a misleading number.
    """
    seed_name = os.path.basename(dataset_dir)
    try:
        manifest = detect_dataset.build_detect_dataset(
            dataset_dir, lab_dataset_dir, seed_name,
            valid_frac=valid_frac, test_frac=test_frac)
    except (ValueError, OSError) as exc:
        raise ImportRefused("the lab's dataset could not be built from "
                            f"{dataset_dir}: {exc}") from exc
    if (manifest.get("pins_source") == detect_dataset.PINS_SOURCE_FRESH
            and not allow_fresh_split):
        raise ImportRefused(
            f"{dataset_dir} carries no {detect_dataset.SOURCE_PINS_NAME}, so "
            "this lab would hold out a fresh set of images drawn here, and "
            "these runs were never chosen on it: every score would be measured "
            "on images some of these runs trained on, and the number would "
            "mislead anyone who read it. Stage the dataset again with its "
            "original split (--split-from), or pass --allow-fresh-split when "
            "the fresh split is what is wanted")
    return manifest


def _copy_run(source, lab_dir, note, log):
    """Copy one finished run into the lab and record where it came from.

    Parameters:
      source: the RunSource to copy.
      lab_dir: the lab folder.
      note: the note recorded beside the run.
      log: where the per-run line goes.
    Returns:
      {"run_name", "run_dir", "best_pt_sha256"} for the copy.
    Raises:
      ImportRefused when a file cannot be copied, or when the copied weights
      do not hash to what the source weights hash to.
    """
    run_dir = os.path.join(lab_dir, "runs", source.name)
    weights_dst = os.path.join(run_dir, WEIGHTS_REL)
    try:
        os.makedirs(os.path.dirname(weights_dst))
        shutil.copy2(os.path.join(source.real, WEIGHTS_REL), weights_dst)
        shutil.copy2(os.path.join(source.real, ARGS_YAML_NAME),
                     os.path.join(run_dir, ARGS_YAML_NAME))
        if source.has_results:
            shutil.copy2(os.path.join(source.real, RESULTS_CSV_NAME),
                         os.path.join(run_dir, RESULTS_CSV_NAME))
    except OSError as exc:
        raise ImportRefused(f"the run {source.given} could not be copied into "
                            f"{run_dir} ({exc})") from exc
    source_digest = sha256_of(os.path.join(source.real, WEIGHTS_REL))
    copied_digest = sha256_of(weights_dst)
    if source_digest != copied_digest:
        raise ImportRefused(
            f"the weights copied for {source.name} do not match their source: "
            f"{source.real} hashes to {source_digest} and the copy to "
            f"{copied_digest}. The copy was cut short, so nothing of this "
            "import is kept")
    size = os.path.getsize(weights_dst)
    _write_json(os.path.join(run_dir, PROVENANCE_NAME), {
        "imported_from": source.real,
        # The path the operator named, made absolute against the folder they
        # typed it from. The verbatim string reaches the refusal messages, where
        # the operator recognises what they typed; the record keeps the path a
        # later reader can resolve.
        "imported_from_given": _absolute(source.given),
        "imported_at": labruns.ast_now(),
        "best_pt_sha256": copied_digest,
        "best_pt_bytes": size,
        "args_data_original": source.args["data"],
        "args_save_dir_original": source.args["save_dir"],
        "results_csv": source.has_results,
        "note": note,
    })
    log(f"imported {source.name}: weights {size} bytes, sha256 {copied_digest[:12]}, "
        f"results.csv {'copied' if source.has_results else 'absent in the source'}")
    return {"run_name": source.name, "run_dir": run_dir,
            "best_pt_sha256": copied_digest}


def import_runs(inprocess_root, dataset_dir, lab_name, run_dirs, *, note="",
                valid_frac=IMPORT_VALID_FRAC, test_frac=IMPORT_TEST_FRAC,
                allow_fresh_split=False, log=print):
    """Import finished detection runs as one lab under inprocess/_models.

    Parameters:
      inprocess_root: the module's projects folder, which holds _models. A
        relative path is made absolute here, so every path this function
        returns, logs or records resolves from anywhere.
      dataset_dir: the staged box dataset the runs were trained on; the lab's
        own split is built from it, on its split_pins.json when it has one.
      lab_name: the lab folder name, checked against the shared lab-name rule.
      run_dirs: the finished run folders to import, each holding
        weights/best.pt and args.yaml. Each is imported under its folder name.
      note: one sentence saying where these runs came from, recorded in
        lab.json and in every provenance.json.
      valid_frac, test_frac: the held-out fractions for images the staged
        dataset does not pin.
      allow_fresh_split: True to accept a lab whose split was drawn here
        rather than replayed from the staged dataset.
      log: where the progress lines go; pass a function that ignores them to
        import quietly.
    Returns:
      {"lab_dir", "dataset_dir", "split_fingerprint", "runs"}, where runs is
      one {"run_name", "run_dir", "best_pt_sha256"} per imported run, in the
      order given.
    Raises:
      ImportRefused, naming what failed and why, for every fault: an unusable
      lab name, a lab folder that already exists, an unstaged dataset, an
      unusable run folder, two runs that would share one name, a fresh split
      without the override, and a copy that does not match its source. Nothing
      is left behind by a refusal: the lab folder is removed whole.

    No score is written. The lab carries no rounds.csv and no champion record:
    VEGA's Evaluate step measures the score and records it with the metric
    family and the split fingerprint.

    Example:
      import_runs("/.../inprocess", "/.../bbox_datasets/fish_photo_28feb25",
                  "fish_photo_28feb25_paper", ["/.../run3_heavy_augment"],
                  note="October 2025 ablation")
    """
    if not callable(log):
        raise ImportRefused("log must be a function that takes one line of text")
    if not isinstance(inprocess_root, str) or not inprocess_root.strip():
        raise ImportRefused("name the module's projects folder (inprocess), "
                            "which holds the _models folder a lab is written to")
    if not isinstance(allow_fresh_split, bool):
        raise ImportRefused("allow_fresh_split must be True or False, got "
                            f"{allow_fresh_split!r}: a value that is merely "
                            "truthy would quietly accept a split nobody chose")
    clean_lab = _check_lab_name(lab_name)
    clean_root = _absolute(inprocess_root)
    clean_dataset = _check_dataset_dir(dataset_dir)
    clean_note = _check_note(note)
    sources = _check_run_sources(run_dirs)
    log(f"Step 1 complete: {len(sources)} run folder(s) checked, "
        f"dataset {clean_dataset}")

    paths = labruns.lab_paths(clean_root, clean_lab)
    lab_dir = paths["lab_dir"]
    try:
        # The folder creation is the claim: two imports racing for one name
        # cannot both write, and the one that arrives second refuses here,
        # before it has touched anything the winner owns.
        os.makedirs(lab_dir)
    except FileExistsError as exc:
        raise ImportRefused(
            f"the lab {clean_lab} already exists at {lab_dir}; import into "
            "another name, or remove that folder first when it is no longer "
            "wanted") from exc
    except OSError as exc:
        raise ImportRefused(f"the lab folder {lab_dir} could not be created "
                            f"({exc})") from exc

    try:
        manifest = _build_lab_dataset(clean_dataset, paths["dataset_dir"],
                                      valid_frac, test_frac, allow_fresh_split)
        log(f"Step 2 complete: dataset built from {manifest['n_pairs']} image "
            f"and label pairs, counts {manifest['counts']}, pins from "
            f"{manifest['pins_source']}, fingerprint {manifest['split_fingerprint']}")
        imported = [_copy_run(source, lab_dir, clean_note, log) for source in sources]
        stamp = labruns.ast_now()
        _write_json(paths["lab_json"], {
            # The lab name under both spellings: lab_name is what
            # labruns.lab_paths returns and what every lab on disk carries, and
            # name is the spelling this tool's design named, so a reader of
            # either finds it.
            "lab_name": clean_lab,
            "name": clean_lab,
            "task": LAB_TASK,
            "source": clean_dataset,
            "source_kind": LAB_SOURCE_KIND,
            "origin_kind": LAB_ORIGIN_KIND,
            # This lab trained nothing, so it has no base weights of its own.
            # Each run's args.yaml names the weights that run started from, and
            # its provenance.json keeps the folders it was trained in.
            "model": "",
            # One stamp under both spellings: created_at because an imported lab
            # is written once and never edited, updated_at because that is the
            # key every lab on disk carries and the one a reader sorts labs by.
            "created_at": stamp,
            "updated_at": stamp,
            "note": clean_note,
            "imported_runs": [source.name for source in sources],
        })
        on_disk = sorted(name for name in os.listdir(paths["runs_dir"])
                         if os.path.isfile(os.path.join(paths["runs_dir"], name,
                                                        WEIGHTS_REL)))
        if len(on_disk) != len(sources):
            raise ImportRefused(
                f"the lab at {lab_dir} holds {len(on_disk)} run(s) with weights "
                f"but {len(sources)} were imported ({on_disk}); nothing of this "
                "import is kept")
    except BaseException:
        # Including a keyboard interrupt: a lab half way through its runs would
        # be listed by the runs card as a set of models nobody can evaluate.
        _remove_claimed_lab(lab_dir, log)
        raise

    log(f"Step 3 complete: {len(on_disk)} run(s) imported to {lab_dir}; "
        "no score was written, run Evaluate in VEGA to score them")
    return {"lab_dir": lab_dir,
            "dataset_dir": paths["dataset_dir"],
            "split_fingerprint": manifest["split_fingerprint"],
            "runs": imported}


# -- CLI ------------------------------------------------------------------

def build_parser():
    """The command-line parser, split out so the flags can be read in one place.

    Returns:
      an argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Import finished ultralytics detection runs trained "
                    "outside VEGA as a model lab the main window lists.")
    parser.add_argument("--dataset", required=True,
                        help="the staged box dataset these runs were trained "
                             "on, under supporting_data/bbox_datasets")
    parser.add_argument("--lab", required=True,
                        help="the lab folder name under inprocess/_models, 1 "
                             "to 60 characters from A-Z a-z 0-9 underscore and "
                             "hyphen")
    parser.add_argument("--runs", required=True, nargs="+",
                        help="the finished run folders to import, each holding "
                             "weights/best.pt and args.yaml")
    parser.add_argument("--inprocess-root", default=DEFAULT_INPROCESS_ROOT,
                        help="the module's projects folder (default the "
                             "inprocess folder beside this clone)")
    parser.add_argument("--note", default="",
                        help="one sentence saying where these runs came from, "
                             "recorded in the lab record and beside every run")
    parser.add_argument("--allow-fresh-split", action="store_true",
                        help="accept a lab whose held-out images were drawn "
                             "here, for a staged dataset that pins no split")
    parser.add_argument("--quiet", action="store_true", help="print nothing")
    return parser


def main(argv=None):
    """Parse the arguments, import the runs, report what was written.

    Parameters:
      argv: the argument list, or None to read sys.argv.
    Returns:
      the dict import_runs returned.
    Raises:
      ImportRefused naming what failed, where, and why.
    """
    args = build_parser().parse_args(argv)
    log = (lambda _line: None) if args.quiet else print
    return import_runs(args.inprocess_root, args.dataset, args.lab, args.runs,
                       note=args.note, allow_fresh_split=args.allow_fresh_split,
                       log=log)


if __name__ == "__main__":
    try:
        main()
    except ImportRefused as exc:
        print(f"import stopped: {exc}", file=sys.stderr)
        sys.exit(EXIT_REFUSED)

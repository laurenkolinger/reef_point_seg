#!/usr/bin/env python3
"""Detection (bounding-box) datasets for step 6, in the shape training reads.

Two shapes of training source reach the Train step, and this module turns both
into the on-disk layout pinned_split.py already produces (train/valid/test,
each with images/ and labels/, plus a data.yaml whose `val` is valid/images):

  1. A staged detection dataset (images/ + labels_localized/ or labels/,
     classes.txt, data.yaml, and optionally split_pins.json and names.csv).
     pinned_split.py cannot read it: it wants all_images/ + all_labels/ and
     pins the holdout by TCRMP transect, which a non-TCRMP dataset has no
     notion of. build_detect_dataset() builds the split itself, seeded on a
     name so the same source always splits the same way. The two optional
     files carry what the staging tool knew and the folder layout cannot say:
     split_pins.json is the split the dataset's original experiment used,
     replayed before any shuffle so a model imported from that experiment and
     a retrain here hold the same images out, and names.csv is the full
     species name for every class token, copied into the split dataset's
     data.yaml as names_full.

     Two manifest fields answer two different questions about that split, and
     a reader who confuses them will misread a rebuild. holdout_mode says
     which policy chose the held-out images: "staged" when the staged
     split_pins.json supplied them, "lab-random" when the seeded shuffle drew
     them, and on a rebuild that replays a manifest's pins whatever that
     manifest already recorded, because replaying a pin never relabels the
     holdout that made it. A build that replays no pin records the draw it
     made, whatever the folder's previous manifest named, so a policy never
     outlives the images it chose. pins_source says which file THIS build read
     the pins from. Before 2026-09-15 holdout_mode was the literal
     "lab-random" on every build, which misdescribed a dataset whose every
     image was replayed from a published experiment.

  2. A segmentation export already split by pinned_split.py into
     <step6>/dataset/. build_detect_from_seg_dataset() writes a parallel
     <step6>/dataset_detect/ whose images are relative SYMLINKS back into
     dataset/ and whose labels are the polygons reduced to their bounding
     boxes. The seg dataset is never rewritten, so the same step-6 folder can
     still train a segmentation model afterwards.

prepare() picks between the two: a dataset whose labels are already boxes
trains from its own data.yaml, one whose labels are polygons gets a
dataset_detect/ built beside it. run_step6.sh calls the CLI form of it
(`detect_dataset.py prepare --dataset <dir>`), which prints the resolved
data.yaml path on stdout and everything else on stderr.

Non-TCRMP imagery never enters inprocess/run_*, the mask registry, the
coverage matrix or STATS: a staged detection dataset is read where it is
staged, and what is written goes to the caller's dataset dir.

The split logic in bbox_pairs/assign_split/build_detect_dataset came from
scripts/_modellab/labruns.py, which now delegates here. Sharing the code does
not by itself make the two splits identical: the split depends on the seed
string, and the model lab seeds on the lab name while the pipeline seeds on
the dataset name, so the same images get two different holdouts (measured on
this box, 2026-09-04: most of one run's test images sat in the other run's
training set). Every manifest therefore records the seed, the fractions and a
split fingerprint (split_fingerprint), and read_split_record hands them back
so a later reader can mark two runs as comparable or not instead of ranking
them together.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Atlantic Standard Time, fixed UTC-4, no daylight saving (house rule: every
# timestamp this platform writes is AST 24h).
AST = timezone(timedelta(hours=-4))

BBOX_LABEL_DIRS = ("labels_localized", "labels")
SPLIT_DIRS = ("train", "valid", "test")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")

# Held-out fractions of a lab-random split. Both callers read these constants:
# scripts/pipeline_orchestrator/app.py forwards the Train panel's validation
# and test fractions and falls back to these; scripts/_modellab/labruns.py's
# wrappers import them by name (since 2026-09-05), so there is one source of
# truth. Both parameters stay overridable.
DEFAULT_VALID_FRAC = 0.1
DEFAULT_TEST_FRAC = 0.1

# Version of the split_manifest.json build_detect_dataset writes. Version 2
# (2026-09-04) added valid_frac, test_frac, split_fingerprint,
# split_fingerprint_scheme, n_replayed and n_fresh. A version 1 manifest is
# still read and its pins still replayed; read_split_record reports it as not
# recorded.
SPLIT_MANIFEST_SCHEMA_VERSION = 2

# How split_fingerprint hashes an assignment, written beside every fingerprint
# so a reader can tell whether two fingerprints are comparable at all. Change
# it whenever the hashing changes.
SPLIT_FINGERPRINT_SCHEME = "sha256-16/netstring-utf8-sorted-v1"
SPLIT_FINGERPRINT_HEX_CHARS = 16

# The manifest keys read_split_record needs before it calls a split recorded.
SPLIT_RECORD_KEYS = ("seed", "valid_frac", "test_frac", "split_fingerprint")

# Two optional files a staged detection dataset may carry beside images/ and
# labels/ (written by scripts/_boxes/ingest_yolo_boxes.py). split_pins.json
# holds the split the dataset's original experiment used, keyed by staged
# basename; names.csv holds the full species name for every class token.
# Neither is required, and the Coral Intel dataset carries neither.
SOURCE_PINS_NAME = "split_pins.json"
NAMES_CSV_NAME = "names.csv"

# What the manifest's pins_source field records: which of the three sources
# of pins a build actually used. The dataset's own manifest wins (a rebuild
# never moves an image), then the staged split_pins.json (so an imported
# model and a retrain hold the same images out), then a seeded shuffle.
PINS_SOURCE_DATASET = "dataset split_manifest.json"
PINS_SOURCE_STAGED = "staged split_pins.json"
PINS_SOURCE_FRESH = "fresh"
PINS_SOURCES = (PINS_SOURCE_DATASET, PINS_SOURCE_STAGED, PINS_SOURCE_FRESH)

# What the manifest's holdout_mode field records: the policy that chose a
# dataset's held-out images in the first place. HOLDOUT_MODE_LAB_RANDOM means
# the seeded shuffle in this module drew them. HOLDOUT_MODE_STAGED (added
# 2026-09-15) means the staged dataset's split_pins.json carried the split its
# original experiment used and this build replayed it, so nothing was drawn
# here; before that date such a build wrote "lab-random", which misdescribed
# every replayed image. A rebuild that keeps a holdout writes neither: it
# carries forward whatever the destination manifest already recorded, so these
# two are what a build choosing for itself picks between
# (holdout_mode_for_build).
#
# A staged file that pins part of a source records HOLDOUT_MODE_STAGED for the
# whole build, because the images it named are the split the original
# experiment chose and this module redrew none of them; n_replayed against
# n_pairs in the same manifest is what says how much of the holdout the staged
# file reached, and every manifest carries both.
#
# pinned_split.py names its own four policies in its own HOLDOUT_MODES and
# refuses any manifest naming one of these two, which is correct: it reads only
# the segmentation splitter's manifests and never a dataset folder this
# function builds. The reverse meeting does happen, because both splitters
# write split_manifest.json into <step6>/dataset: a detection build can find
# the segmentation splitter's manifest there, replays nothing from it (its pin
# table is "pinned") and records its own policy rather than that file's.
HOLDOUT_MODE_LAB_RANDOM = "lab-random"
HOLDOUT_MODE_STAGED = "staged"
HOLDOUT_MODES = (HOLDOUT_MODE_LAB_RANDOM, HOLDOUT_MODE_STAGED)

# The header read_names_csv requires of a staged names.csv.
NAMES_CSV_COLUMNS = ("token", "name")

# Prefixes of the temp folders a build stages beside its destination: the
# build in progress, and the previous dataset moved aside during the swap.
_BUILD_DIR_PREFIX = ".dataset_build_"
_DETECT_BUILD_DIR_PREFIX = ".dataset_detect_build_"
_OLD_DIR_PREFIX = ".dataset_old_"

# A polygon needs three points (six coordinates) to enclose any area. A label
# line shorter than that is degenerate and is dropped by the box conversion.
_MIN_POLYGON_COORDS = 6


def ast_now():
    """Timestamp the way every VICARIUS surface writes it: AST, 24h."""
    return datetime.now(AST).strftime("%Y-%m-%d %H:%M:%S")


# ── data.yaml ────────────────────────────────────────────────────────────

def data_yaml_labels(path):
    """Class names from a YOLO data.yaml (list or {id: name} mapping); []
    when the file is missing or unreadable. PyYAML when it imports, else a
    tolerant line parser, so a listing never fails on a stray file."""
    if not os.path.isfile(path):
        return []
    try:
        import yaml
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
    import re
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


def read_data_yaml(path):
    """The whole data.yaml as a dict; {} when it cannot be read."""
    try:
        import yaml
        with open(path) as fh:
            doc = yaml.safe_load(fh) or {}
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


# ── Staged detection sources ─────────────────────────────────────────────

def bbox_label_dir(dataset_dir):
    """Where a staged detection dataset keeps its YOLO label files.
    labels_localized/ (degenerate full-image boxes dropped) wins over
    labels/ when the ingest wrote both. '' when neither exists."""
    for rel in BBOX_LABEL_DIRS:
        d = os.path.join(dataset_dir, rel)
        if os.path.isdir(d):
            return d
    return ""


def read_classes_txt(path):
    """Class names, one per line, from a pinned classes.txt; [] when absent."""
    if not os.path.isfile(path):
        return []
    try:
        with open(path) as fh:
            return [ln.strip() for ln in fh if ln.strip()]
    except OSError:
        return []


def bbox_pairs(source_dir):
    """(pairs, unlabelled): every image with a label file beside it, and how
    many images had none. An image with no label file is a background image,
    not a training pair, and a staged dataset that lost its labels must fail
    loudly rather than train on nothing."""
    image_dir = os.path.join(source_dir, "images")
    label_dir = bbox_label_dir(source_dir)
    if not os.path.isdir(image_dir) or not label_dir:
        return [], 0
    pairs, unlabelled = [], 0
    for fn in sorted(os.listdir(image_dir)):
        if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
            continue
        # A folder named like an image is not an image: skipped the way a
        # stray non-image file is, rather than failing later on the copy.
        if not os.path.isfile(os.path.join(image_dir, fn)):
            continue
        label = os.path.join(label_dir, os.path.splitext(fn)[0] + ".txt")
        if os.path.isfile(label):
            pairs.append((fn, os.path.join(image_dir, fn), label))
        else:
            unlabelled += 1
    return pairs, unlabelled


def _validate_fractions(valid_frac: float, test_frac: float) -> None:
    """Refuse a held-out fraction that is not a real number in [0, 1), and a
    pair that leaves no image for training. A bool is refused too: True would
    silently read as 1.0."""
    for label, value in (("valid_frac", valid_frac), ("test_frac", test_frac)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
            raise ValueError(f"{label} must be a number between 0 and 1 (got {value!r})")
        if not 0 <= value < 1:
            raise ValueError(f"{label} must be at least 0 and below 1 (got {value!r})")
    if valid_frac + test_frac >= 1:
        raise ValueError(f"valid_frac + test_frac must leave images for training "
                         f"(got {valid_frac} + {test_frac}, which is 1 or more)")


def assign_split(names: Iterable[str], seed_text: object, existing: Mapping | None = None,
                 valid_frac: float = DEFAULT_VALID_FRAC,
                 test_frac: float = DEFAULT_TEST_FRAC) -> dict:
    """basename -> train|valid|test.

    Every basename with a pin in ``existing`` keeps it unchanged: retraining
    never moves an image between splits, and only images with no pin yet are
    assigned. The fresh images are shuffled by random.Random(str(seed_text)),
    so the same seed over the same fresh images always gives the same split
    and a different seed gives a different one; the first
    round(n_fresh * test_frac) shuffled images go to test, the next
    round(n_fresh * valid_frac) to valid, the rest to train.

    Parameters: names, the basenames to assign, any iterable, read once;
    seed_text, what the shuffle is seeded on (coerced with str()); existing,
    {basename: split} pins to replay, where a pin for a basename not in names
    or with an unknown split is ignored; valid_frac and test_frac, the
    held-out fractions of the fresh images, checked by _validate_fractions.

    Returns {basename: split} covering every basename in names.
    """
    _validate_fractions(valid_frac, test_frac)
    names = list(names)
    wanted = set(names)
    pins = {k: v for k, v in (existing or {}).items() if k in wanted and v in SPLIT_DIRS}
    fresh = [n for n in sorted(names) if n not in pins]
    if not fresh:
        return pins
    rng = random.Random(str(seed_text))
    shuffled = list(fresh)
    rng.shuffle(shuffled)
    n_test = int(round(len(fresh) * test_frac))
    n_valid = int(round(len(fresh) * valid_frac))
    for i, nm in enumerate(shuffled):
        if i < n_test:
            pins[nm] = "test"
        elif i < n_test + n_valid:
            pins[nm] = "valid"
        else:
            pins[nm] = "train"
    return pins


def _place(src, dst):
    """Hardlink when the filesystem allows it (a staged dataset is read-only
    input, and a 10k-image link costs nothing), else copy."""
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _make_build_dir(final: str, prefix: str) -> str:
    """A fresh temp folder beside ``final`` for staging a build, so the swap
    into place is one rename on one filesystem. Raises OSError naming the
    parent folder when it cannot be created there."""
    parent = os.path.dirname(final) or "."
    try:
        os.makedirs(parent, exist_ok=True)
        return tempfile.mkdtemp(prefix=prefix, dir=parent)
    except OSError as e:
        raise type(e)(f"cannot create a build folder for {final} under {parent}: "
                      f"{e.strerror or e}") from e


def _swap_into_place(tmp_dir: str, final: str) -> None:
    """Put the staged build at ``tmp_dir`` where ``final`` belongs.

    A previous dataset at ``final`` is moved aside first and removed only
    after the new one is in place, so a failure never leaves ``final``
    half-deleted: either the old dataset is still there whole, or the new one
    is. Raises OSError naming the paths when a rename fails, or when the new
    dataset is in place but the old copy could not be removed (it stays
    beside it under a .dataset_old_ folder for Lauren to delete by hand).
    """
    old_dir = ""
    if os.path.isdir(final):
        old_dir = _make_build_dir(final, _OLD_DIR_PREFIX)
        try:
            os.rename(final, os.path.join(old_dir, "dataset"))
        except OSError as e:
            leftover = ""
            try:
                os.rmdir(old_dir)
            except OSError:
                leftover = f"; the empty folder {old_dir} was left behind"
            raise type(e)(f"cannot move the previous dataset {final} aside to {old_dir}: "
                          f"{e.strerror or e}{leftover}") from e
    try:
        os.rename(tmp_dir, final)
    except OSError as e:
        raise _rollback_after_failed_swap(e, tmp_dir, final, old_dir) from e
    if not old_dir:
        return
    try:
        shutil.rmtree(old_dir)
    except OSError as e:
        raise type(e)(f"the new dataset is in place at {final}, but its previous copy could "
                      f"not be removed from {old_dir}: {e.strerror or e}; delete that folder "
                      "by hand") from e


def _rollback_after_failed_swap(error: OSError, tmp_dir: str, final: str, old_dir: str) -> OSError:
    """Put the previous dataset back after the built one could not be moved
    into ``final``, and return the OSError to raise: one naming every path
    involved, and saying where the previous dataset sits when even the
    rollback fails. ``old_dir`` is empty when there was no previous dataset."""
    message = (f"cannot move the built dataset {tmp_dir} into place at {final}: "
               f"{error.strerror or error}")
    if not old_dir:
        return type(error)(message)
    kept = os.path.join(old_dir, "dataset")
    try:
        os.rename(kept, final)
    except OSError as rollback_error:
        return type(error)(
            f"{message}; the previous dataset is whole at {kept} and could not be moved back "
            f"({rollback_error.strerror or rollback_error}): move it back to {final} by hand")
    try:
        os.rmdir(old_dir)
    except OSError as rmdir_error:
        return type(error)(
            f"{message}; the previous dataset was put back, and the empty folder {old_dir} "
            f"could not be removed ({rmdir_error.strerror or rmdir_error}) and was left behind")
    return type(error)(f"{message}; the previous dataset was put back")


def _refuse_symlink(final: str, what: str) -> None:
    """Refuse to build into a symbolic link: the swap would replace the link
    with a real folder and leave the link's target holding a stale dataset."""
    if os.path.islink(final):
        raise ValueError(f"{what} {final} is a symbolic link; point at the real folder instead")


def _load_json_object(path: str, advice: str) -> dict | None:
    """The JSON object in the file at ``path``; None when there is no such
    file. Raises ValueError, ending with ``advice``, when the file cannot be
    read or does not hold a JSON object."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise ValueError(f"{path} cannot be read as JSON ({e}); {advice}") from e
    if not isinstance(doc, dict):
        raise ValueError(f"{path} must hold a JSON object, got {type(doc).__name__}; {advice}")
    return doc


def _write_json(path: str, doc: dict) -> None:
    """Write ``doc`` to ``path`` as indented, key-sorted UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)


def _write_yaml_pair(tmp_dir: str, doc: dict) -> None:
    """Write data.yaml (val: valid/images) and test.yaml (val: test/images)
    from ``doc`` into ``tmp_dir``. test.yaml is vestigial, kept for parity
    with pinned_split.py, and read by nothing."""
    # PyYAML is imported here and not at module top, so read_split_record and
    # split_fingerprint import with no third-party package at all.
    import yaml
    with open(os.path.join(tmp_dir, "data.yaml"), "w", encoding="utf-8") as fh:
        yaml.dump(doc, fh, sort_keys=False)
    test_doc = dict(doc)
    test_doc["val"] = "test/images"
    with open(os.path.join(tmp_dir, "test.yaml"), "w", encoding="utf-8") as fh:
        yaml.dump(test_doc, fh, sort_keys=False)


def _place_pairs(tmp_dir: str, pairs: list, pins: Mapping) -> None:
    """Create train/valid/test under ``tmp_dir``, each with images/ and
    labels/, and place every (basename, image path, label path) pair from
    ``pairs`` into the split ``pins`` assigns it (hardlink or copy)."""
    for split in SPLIT_DIRS:
        os.makedirs(os.path.join(tmp_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(tmp_dir, split, "labels"), exist_ok=True)
    for basename, img_path, lbl_path in pairs:
        split = pins[basename]
        _place(img_path, os.path.join(tmp_dir, split, "images", basename))
        _place(lbl_path, os.path.join(tmp_dir, split, "labels",
                                      os.path.splitext(basename)[0] + ".txt"))


def _data_yaml_doc(final: str, source_dir: str) -> dict:
    """The data.yaml document for a split dataset at ``final``, with class
    names from the source's classes.txt, else its data.yaml.

    When the staged source carries a names.csv, the document also carries
    names_full, {class id: full name}, for every token that table names.
    Ultralytics reads only the keys it knows, so the extra key is inert for
    training and gives every later reader (the evaluator, the inference
    overlay, the report card) the full species name without re-reading the
    staged folder. A token with no row in names.csv, or one whose name cell is
    blank, is left out rather than guessed or named by an empty string, and a
    source with no names.csv gets the document it always got.
    """
    names_list = read_classes_txt(os.path.join(source_dir, "classes.txt")) \
        or data_yaml_labels(os.path.join(source_dir, "data.yaml"))
    doc = {
        "path": final,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {i: nm for i, nm in enumerate(names_list)},
    }
    full = read_names_csv(source_dir)
    if full:
        doc["names_full"] = {i: full[nm] for i, nm in enumerate(names_list) if nm in full}
    return doc


def _read_existing_split(manifest_path: str) -> tuple[dict, str | None]:
    """What an earlier build already decided at ``manifest_path``: its pins and
    the holdout policy it recorded.

    Parameters: manifest_path, the split_manifest.json of the dataset folder
    being rebuilt, which need not exist.

    Returns (pins, holdout_mode): {basename: split} and the recorded policy,
    ({}, None) when no manifest exists yet and (pins, None) when the manifest
    records no holdout_mode. A manifest written by the segmentation splitter
    reads as ({}, its policy), because its pin table is "pinned" and only
    "pins" is read here. The two travel together because a rebuild that
    replays the pins keeps the policy that chose them, and the caller pairs
    them (holdout_mode_for_build) rather than carrying a policy whose pins
    decided nothing; reading them in one pass also parses a manifest of many
    thousand pins once.

    Raises ValueError, naming the file, when the manifest exists but is not a
    JSON object, its pins are not an object, its pins carry a split outside
    SPLIT_DIRS, or its holdout_mode is anything but a non-empty string.
    Ignoring a damaged manifest would re-split those images and move them
    between train, valid and test, which is exactly what the pins exist to
    prevent, and carrying a holdout_mode that is not a string forward would
    write a manifest no reader can use.
    """
    advice = "fix or remove it before rebuilding, because ignoring it would re-split every image"
    doc = _load_json_object(manifest_path, advice)
    if doc is None:
        return {}, None
    holdout_mode = doc.get("holdout_mode")
    if holdout_mode is not None and (not isinstance(holdout_mode, str) or not holdout_mode.strip()):
        raise ValueError(
            f"{manifest_path}: holdout_mode must be a non-empty string naming the policy that "
            f"chose this holdout, got {holdout_mode!r}; a rebuild carries that value forward, so "
            "fix or remove the manifest before rebuilding")
    pins = doc.get("pins")
    if pins is None:
        return {}, holdout_mode
    if not isinstance(pins, dict):
        raise ValueError(f"{manifest_path}: pins must be an object of basename to split, got "
                         f"{type(pins).__name__}; {advice}")
    bad = [(k, v) for k, v in pins.items() if v not in SPLIT_DIRS]
    if bad:
        raise ValueError(
            f"{manifest_path}: {len(bad)} pin(s) carry a split that is not one of {SPLIT_DIRS}, "
            f"for example {bad[:3]!r}; fix or remove the manifest before rebuilding, because "
            "ignoring them would re-split those images")
    return pins, holdout_mode


def holdout_mode_for_build(recorded_mode: str | None, pins_source: str,
                           n_replayed: int) -> str:
    """The holdout policy a build records: what actually chose its held-out
    images, not what this build would have chosen.

    Parameters: recorded_mode, the holdout_mode the destination manifest
    already carries (None when there is no manifest, or it records none);
    pins_source, which of the three sources of pins this build used, one of
    PINS_SOURCE_DATASET, PINS_SOURCE_STAGED or PINS_SOURCE_FRESH; n_replayed,
    how many of this build's images took their split from those pins rather
    than from the seeded shuffle.

    Returns recorded_mode unchanged when this build replayed the pins of the
    manifest that recorded it (pins_source PINS_SOURCE_DATASET and n_replayed
    above zero), because those images kept the splits that manifest pinned and
    the policy that produced them is still the earlier one; else
    HOLDOUT_MODE_STAGED when the staged dataset's split_pins.json supplied at
    least one pin, because the original experiment chose those held-out images
    and this module drew none of them (n_replayed against n_pairs in the
    manifest says how many the staged file reached, which is every image on a
    fully pinned import such as the fish dataset); else
    HOLDOUT_MODE_LAB_RANDOM, the seeded draw.

    Two cases turn on n_replayed rather than on the policy string, and both
    are reachable from the orchestrator's Step 5 source list. A pin table that
    decided none of this build's images describes nothing on disk, so a
    manifest left by a source that has since been switched away, or a staged
    file naming only images the source no longer holds, records the seeded
    draw that really happened. And a recorded policy this build replayed no
    pin from never travels: the segmentation splitter writes its own
    split_manifest.json into the same <step6>/dataset folder, with the frozen
    transect holdout under holdout_mode "transect" and its pin table under
    "pinned", which this module never reads; stamping "transect" on a seeded
    shuffle would make the strongest claim in the module's vocabulary about a
    split nothing froze. A destination manifest carrying pins and no
    holdout_mode (the shape written before 2026-09-15, or a hand-edited file)
    records lab-random, which is what every detection build wrote before this
    change and so claims nothing the file did not already claim.

    Raises ValueError when pins_source is none of the three constants, because
    a policy chosen from an unknown source would misdescribe the holdout, and
    TypeError or ValueError when n_replayed is not a whole number of images at
    or above zero, because a miscounted replay decides the wrong branch.

    Example: holdout_mode_for_build(None, PINS_SOURCE_STAGED, 7572) is
    "staged"; holdout_mode_for_build("staged", PINS_SOURCE_DATASET, 7572) is
    "staged" again on the rebuild that replays those pins; and
    holdout_mode_for_build("transect", PINS_SOURCE_FRESH, 0) is "lab-random",
    because a policy from the other writer's manifest chose none of these
    images.
    """
    if pins_source not in PINS_SOURCES:
        raise ValueError(f"pins_source must be one of {PINS_SOURCES} (got {pins_source!r}): "
                         "holdout_mode records the policy that source stands for")
    if isinstance(n_replayed, bool) or not isinstance(n_replayed, int):
        raise TypeError(f"n_replayed must be an int (got {type(n_replayed).__name__}): it is "
                        "the count of images a pin decided, and it chooses which policy "
                        "holdout_mode records")
    if n_replayed < 0:
        raise ValueError(f"n_replayed must be zero or more (got {n_replayed}): it is the count "
                         "of images a pin decided, and it chooses which policy holdout_mode "
                         "records")
    if not n_replayed:
        return HOLDOUT_MODE_LAB_RANDOM
    if pins_source == PINS_SOURCE_DATASET and recorded_mode:
        return recorded_mode
    if pins_source == PINS_SOURCE_STAGED:
        return HOLDOUT_MODE_STAGED
    return HOLDOUT_MODE_LAB_RANDOM


def read_source_pins(source_dir: str) -> dict:
    """The split a staged dataset pins for its own images: {basename: split}
    from <source_dir>/split_pins.json, {} when the file is absent.

    A staged dataset written by scripts/_boxes/ingest_yolo_boxes.py carries
    the split its original experiment used, so a model imported from that
    experiment and a retrain in VEGA hold the same images out and their
    fingerprints match.

    Parameters: source_dir, the staged dataset folder.

    Returns {basename: split} with every split one of SPLIT_DIRS; the
    basenames are the staged ones, so build_detect_dataset can replay them
    without a second lookup. A pin for a basename the source does not hold is
    returned here and ignored by assign_split.

    Raises ValueError, naming the file, when it cannot be read as JSON, does
    not hold a JSON object, carries no pins object, or pins a split outside
    SPLIT_DIRS: ignoring such a file would re-split every image it names,
    which is exactly what the pins exist to prevent. Every refusal reaches the
    Train panel as it is written here, so each says the fault in plain words.
    """
    path = os.path.join(source_dir, SOURCE_PINS_NAME)
    advice = ("fix or remove it before building, because ignoring it would re-split every "
              "image it names")
    doc = _load_json_object(path, advice)
    if doc is None:
        return {}
    pins = doc.get("pins")
    if pins is None:
        raise ValueError(f"{path} carries no pins object, so it names neither an image nor a "
                         f"split; the file must hold a pins object of image basename to one of "
                         f"{SPLIT_DIRS}. Fix or remove it before building.")
    if not isinstance(pins, dict):
        raise ValueError(f"{path}: pins must be an object of basename to split, got "
                         f"{type(pins).__name__}; {advice}")
    bad = [(k, v) for k, v in pins.items() if v not in SPLIT_DIRS]
    if bad:
        raise ValueError(f"{path}: {len(bad)} pin(s) name a split that is not one of "
                         f"{SPLIT_DIRS}, for example {bad[:3]!r}; {advice}")
    return {str(k): str(v) for k, v in pins.items()}


def read_names_csv(source_dir: str) -> dict:
    """The full name of every class token a staged dataset declares:
    {token: full name} from <source_dir>/names.csv, {} when absent.

    The platform shows the full name beside any code it can resolve, and a
    staged dataset's tokens (KYSE_Normal) resolve against nothing else, so
    the staged table is what build_detect_dataset copies into the split
    dataset's data.yaml as names_full.

    Parameters: source_dir, the staged dataset folder.

    Returns {token: full name} with both sides stripped. A row whose token
    cell is blank, and a row whose name cell is blank, are both skipped, so
    that class falls back to its token exactly as a class with no row at all
    does; a class named by an empty string would show a reader a species
    column with nothing in it. A byte-order mark, which a spreadsheet writes
    when it saves a CSV, is read and discarded.

    Raises ValueError, naming the file, when the header does not carry both
    of NAMES_CSV_COLUMNS, when one token is named twice with two different
    names, when the file is not valid UTF-8, or when it cannot be parsed as
    CSV; OSError naming the file when it cannot be opened or read. A table
    nobody can read is refused rather than skipped, because a silently empty
    table would show every class as a bare token, and a token named twice
    would change a species label wherever the class is shown with no trace of
    which row won.

    The staging tool (scripts/_boxes/ingest_yolo_boxes.py) carries a function
    of the same name that takes the CSV path itself and returns a list of
    (token, name) pairs in file order. Code that needs both reads them as
    detect_dataset.read_names_csv and ingest_yolo_boxes.read_names_csv, never
    by the bare name.
    """
    path = os.path.join(source_dir, NAMES_CSV_NAME)
    if not os.path.isfile(path):
        return {}
    fix_it = "fix or remove it before building"
    try:
        # utf-8-sig reads a plain UTF-8 file unchanged and discards a
        # byte-order mark when one is there, so a table saved from a
        # spreadsheet is not refused for a header nobody can see.
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or []
            missing = [c for c in NAMES_CSV_COLUMNS if c not in fields]
            if missing:
                raise ValueError(f"{path} lacks the column(s) {missing} (header read as "
                                 f"{fields}); the header must be token,name")
            rows = list(reader)
    except UnicodeDecodeError as e:
        raise ValueError(f"{path} is not valid UTF-8 ({e}); it holds the full name of every "
                         f"class token, so {fix_it}") from e
    except csv.Error as e:
        raise ValueError(f"{path} could not be parsed as CSV ({e}); {fix_it}") from e
    except OSError as e:
        raise OSError(f"{path} could not be read ({e}); it holds the full name of every class "
                      f"token, so {fix_it}") from e
    table = {}
    for row in rows:
        token = (row.get("token") or "").strip()
        name = (row.get("name") or "").strip()
        if not token or not name:
            continue
        if token in table and table[token] != name:
            raise ValueError(f"{path} names the token {token!r} twice, as {table[token]!r} and "
                             f"{name!r}, so the species shown for that class would be whichever "
                             f"row is read last; {fix_it}")
        table[token] = name
    return table


def _no_train_split_message(counts: Mapping[str, int], n_images: int, n_replayed: int,
                            valid_frac: float, test_frac: float, pins_source: str,
                            source_dir: str, manifest_path: str) -> str:
    """The refusal an operator reads when a split leaves the train folder
    empty, naming whatever actually made that decision.

    The orchestrator returns this text verbatim as the Train panel's error
    (scripts/pipeline_orchestrator/app.py), so it ends in the remedy that can
    change the outcome: the fractions only when the seeded shuffle assigned
    every image, and otherwise the pin file that held them out, because
    lowering a fraction that reached no image changes nothing.

    Parameters: counts, images per split in the refused assignment; n_images,
    how many image and label pairs the source holds; n_replayed, how many of
    them a pin decided; valid_frac and test_frac, the fractions this build was
    given; pins_source, which of the three sources of pins the build used, one
    of PINS_SOURCE_DATASET, PINS_SOURCE_STAGED or PINS_SOURCE_FRESH;
    source_dir, the staged dataset; manifest_path, the split_manifest.json of
    the dataset folder being built.

    Returns the whole message as one line of plain words.
    Raises ValueError when pins_source is none of the three constants, because
    a message that named the wrong file would send the operator to fix a file
    that decided nothing.
    """
    if pins_source not in PINS_SOURCES:
        raise ValueError(f"pins_source must be one of {PINS_SOURCES} (got {pins_source!r}): "
                         "the refusal names the file that source stands for")
    head = f"the split leaves no training image: {n_images} image(s) give {counts}"
    fractions = f"valid_frac {valid_frac} and test_frac {test_frac}"
    if pins_source == PINS_SOURCE_FRESH:
        return f"{head} at {fractions}; lower the fractions or add images"
    if pins_source == PINS_SOURCE_STAGED:
        pin_file = os.path.join(source_dir, SOURCE_PINS_NAME)
        whose = "the staged split pins at"
    else:
        pin_file = manifest_path
        whose = "the split already recorded at"
    n_fresh = n_images - n_replayed
    if n_fresh:
        remedy = (f"so fix or remove that file, or lower the fractions ({fractions}), which "
                  f"reached only the {n_fresh} image(s) it does not name")
    else:
        remedy = f"so fix or remove that file: {fractions} reached no image at all"
    return (f"{head}; {whose} {pin_file} assigned {n_replayed} of them and none to train, "
            f"{remedy}")


def _manifest_source(manifest_path: str) -> str:
    """The staged dataset folder a saved split recorded under "source", resolved
    through symbolic links; '' when the manifest records none (written before
    that key existed) or does not exist."""
    doc = _load_json_object(manifest_path, "fix or remove it before rebuilding")
    recorded = (doc or {}).get("source")
    if not isinstance(recorded, str) or not recorded.strip():
        return ""
    return os.path.realpath(recorded)


def saved_split_applies(pins: Mapping, names: Iterable[str], manifest_path: str,
                        source_dir: str) -> bool:
    """Whether a project's saved split is this dataset's split and must be kept.

    It applies when it recorded this folder (compared with links resolved),
    recorded no folder at all, or shares any photo name with this dataset: the
    same photos reached through a moved, renamed or linked folder keep their
    split. Only a split naming none of these photos, recorded from another
    folder, is set aside, because the project switched to a different dataset.

    Parameters: pins, the saved {basename: split}; names, the image basenames
    this dataset holds; manifest_path, the saved split's file; source_dir, the
    staged dataset being built.
    Returns True to replay (and protect) the saved split, False to set it aside.
    """
    recorded = _manifest_source(manifest_path)
    if not recorded or recorded == os.path.realpath(source_dir):
        return True
    return bool(set(pins) & set(names))


# How many missing photo names a refusal quotes, so the message stays readable.
ABSENT_PIN_EXAMPLES = 3


def _quote(names: list) -> str:
    """The first ABSENT_PIN_EXAMPLES names, quoted, for an error message."""
    return ", ".join(repr(n) for n in names[:ABSENT_PIN_EXAMPLES])


def refuse_pins_for_absent_images(pins: Mapping, names: Iterable[str], pins_path: str,
                                  source_dir: str, from_project: bool) -> None:
    """Refuse a split that names a photo the staged dataset cannot train on.

    A held-out set is only preserved if every photo it names is still there
    under the same name with its labels. Before 2026-09-26 such an entry was
    dropped in silence, so a pin file keyed by the wrong spelling (a stem with
    no extension) matched nothing, every photo was re-drawn at random, and the
    split record still named the pin file as its source. Now the build stops.

    Parameters: pins, the {basename: split} being replayed; names, every
    image basename the dataset pairs with a label file; pins_path, the file
    the split came from; source_dir, the staged dataset; from_project, True
    when pins_path is the project's own saved split (the advice differs:
    that file is never edited by hand).

    Raises ValueError, separating names with no image in images/ from images
    whose label file is gone. Photos the dataset gained since the split was
    made are not refused: they have no entry, and assign_split places them
    without moving any photo that has one.

    Example: pins {"a.jpg": "test", "b": "train"} over names ["a.jpg", "b.jpg"]
    raises, quoting "b".
    """
    held = set(names)
    absent = sorted(k for k in pins if k not in held)
    if not absent:
        return
    image_dir = os.path.join(source_dir, "images")
    # A name carrying a folder part is never a staged photo, whatever the disk says.
    unlabelled = [n for n in absent if os.path.basename(n) == n
                  and os.path.isfile(os.path.join(image_dir, n))]
    unlabelled_set = set(unlabelled)
    missing = [n for n in absent if n not in unlabelled_set]
    parts = []
    if missing:
        parts.append(f"{len(missing)} are not in {image_dir} (for example {_quote(missing)})")
    if unlabelled:
        parts.append(f"{len(unlabelled)} are in images/ but have no label file "
                     f"(for example {_quote(unlabelled)})")
    if from_project:
        advice = ("Put those photos and labels back, or start a new project: dropping a "
                  "photo from a saved split changes the test set every earlier model was "
                  "scored on. Do not edit this file by hand.")
    else:
        advice = ("Each entry must name a photo exactly as it sits in images/, extension "
                  "included, and the photo needs its label file.")
    raise ValueError(
        f"The split in {pins_path} names {len(pins)} photos, and {len(absent)} of them "
        f"cannot be trained on: " + "; ".join(parts) + ". VEGA stopped rather than "
        f"change which photos are held out. {advice}")


def build_detect_dataset(source_dir: str, dataset_dir: str, seed_name: str,
                         valid_frac: float = DEFAULT_VALID_FRAC,
                         test_frac: float = DEFAULT_TEST_FRAC) -> dict:
    """Build <dataset_dir>/ (train/valid/test, data.yaml, test.yaml,
    split_manifest.json) from a staged detection dataset and return the
    manifest that was written.

    Parameters: source_dir, the staged dataset (images/ beside labels/ or
    labels_localized/, with classes.txt or data.yaml for the class names);
    dataset_dir, where the split dataset goes, replaced whole when it exists;
    seed_name, a non-empty string the split shuffle is seeded on (both callers
    must pass the same string for the same source, or the same images get two
    different holdouts); valid_frac and test_frac, the held-out fractions
    applied to images that have no pin yet, DEFAULT_VALID_FRAC and
    DEFAULT_TEST_FRAC unless given.

    Pins come from the first of three sources that has any, and the manifest
    records which one in pins_source. First, <dataset_dir>/split_manifest.json
    when it already exists: every image it pins keeps its split whatever seed
    or fractions this call carries, and only images the source gained since
    are assigned (PINS_SOURCE_DATASET). Second, <source_dir>/split_pins.json,
    the split the staged dataset's original experiment used, so an imported
    model and a retrain are scored on the same held-out images and their
    fingerprints match; images it does not name are assigned by the seeded
    shuffle as usual (PINS_SOURCE_STAGED). Either source that names a photo
    the dataset does not hold, or holds without a label file, is refused
    (refuse_pins_for_absent_images), so a held-out photo can never be re-drawn
    in silence. Third, nothing, and the whole
    source is assigned by the seeded shuffle (PINS_SOURCE_FRESH). Either file
    is refused rather than ignored when it cannot be read (see
    _read_existing_split and read_source_pins).

    The manifest's holdout_mode says which policy chose the holdout, decided
    by holdout_mode_for_build from where the assignment actually came:
    whatever an existing manifest recorded, for as long as this build replays
    that manifest's pins (a rebuild never relabels a holdout it kept), else
    HOLDOUT_MODE_STAGED when the staged split_pins.json supplied at least one
    pin, else HOLDOUT_MODE_LAB_RANDOM for the seeded draw. It answers a
    different question from pins_source, which names the file THIS build read
    the pins from, so on a rebuild the two differ on purpose: a staged holdout
    rebuilt in place records holdout_mode "staged" beside pins_source
    "dataset split_manifest.json". A build that replays no pin at all records
    its own seeded draw whatever the folder's previous manifest named, which
    is what happens over the segmentation splitter's manifest (its pin table
    is "pinned", which this module does not read) and over a manifest left by
    a source the operator has since switched away from.

    The manifest also records the seed, the fractions, a fingerprint of the
    final assignment (split_fingerprint), how many pins were replayed
    (n_replayed) and how many images were freshly assigned (n_fresh), so a
    reader can tell whether the seed reached any image at all.

    Raises ValueError for a saved or staged split naming a photo the source
    cannot train on (refuse_pins_for_absent_images), a source with no image
    and label pair, a bad seed,
    bad fractions, a split that leaves no training image (the message names
    the pin file or the fractions, whichever held every image out), a symbolic
    link as dataset_dir, an unreadable manifest, a manifest whose holdout_mode
    is not a non-empty string, a malformed split_pins.json, a names.csv whose
    header is not token,name, or a names.csv naming one token twice; OSError
    when the build folder cannot be created, when the previous dataset cannot
    be replaced, or when a staged names.csv cannot be read.
    """
    _validate_fractions(valid_frac, test_frac)
    if not isinstance(seed_name, str) or not seed_name.strip():
        raise ValueError(
            f"seed_name must be a non-empty string (got {seed_name!r}): the split shuffle is "
            "seeded on it and it is recorded so the split can be reproduced")
    final = os.path.abspath(dataset_dir)
    _refuse_symlink(final, "dataset_dir")

    pairs, unlabelled = bbox_pairs(source_dir)
    if not pairs:
        raise ValueError(f"no image and label pairs found in {source_dir}: expected images/ "
                         "beside labels/ or labels_localized/")

    manifest_path = os.path.join(dataset_dir, "split_manifest.json")
    existing, recorded_mode = _read_existing_split(manifest_path)
    if existing and not saved_split_applies(existing, [p[0] for p in pairs],
                                            manifest_path, source_dir):
        # The project switched to a different staged dataset: that split names
        # none of these photos and decides nothing here.
        existing = {}
    pins_source = PINS_SOURCE_DATASET if existing else PINS_SOURCE_FRESH
    if not existing:
        staged = read_source_pins(source_dir)
        if staged:
            existing, pins_source = staged, PINS_SOURCE_STAGED
    names = [p[0] for p in pairs]
    if existing:
        pins_path = (manifest_path if pins_source == PINS_SOURCE_DATASET
                     else os.path.join(source_dir, SOURCE_PINS_NAME))
        refuse_pins_for_absent_images(existing, names, pins_path, source_dir,
                                      from_project=pins_source == PINS_SOURCE_DATASET)
    pins = assign_split(names, seed_name, existing, valid_frac, test_frac)
    n_replayed = sum(1 for n in names if existing.get(n) in SPLIT_DIRS)
    counts = {s: sum(1 for v in pins.values() if v == s) for s in SPLIT_DIRS}
    if counts["train"] == 0:
        raise ValueError(_no_train_split_message(counts, len(names), n_replayed, valid_frac,
                                                 test_frac, pins_source, source_dir,
                                                 manifest_path))

    summary = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        # The policy that chose these held-out images, which is not always the
        # one this build would have used: a rebuild keeps the policy recorded
        # by the manifest whose pins it replays, and only while it replays
        # them (n_replayed).
        "holdout_mode": holdout_mode_for_build(recorded_mode, pins_source, n_replayed),
        # Which of the three sources of pins this build used, so a reader can
        # tell a replayed experiment split from a seeded draw of its own.
        "pins_source": pins_source,
        "seed": seed_name,
        # The fractions this call was given. They reached only the n_fresh
        # images assigned now; the n_replayed pins kept whatever split they
        # had. The fingerprint, not these two numbers, says what is on disk.
        "valid_frac": float(valid_frac),
        "test_frac": float(test_frac),
        "split_fingerprint": split_fingerprint(pins),
        "split_fingerprint_scheme": SPLIT_FINGERPRINT_SCHEME,
        "created_at": ast_now(),
        "source": os.path.realpath(source_dir),
        "label_dir": bbox_label_dir(source_dir),
        "n_pairs": len(pairs),
        "n_unlabelled": unlabelled,
        "n_replayed": n_replayed,
        "n_fresh": len(names) - n_replayed,
        "counts": counts,
        # The per-image split, under the key this writer has always used. The
        # segmentation splitter records the same table as "pinned", and every
        # reader accepts both spellings rather than either writer renaming:
        # read_split_record below, evaluate_run._split_pinned (step 7), and the
        # Refine loop's seeder (scripts/TCRMPclip_combinedAnnotate/src/
        # seed_from_predictions.py), whose held-out guard read only "pinned"
        # until 2026-09-16 and so excluded nothing on a detection project.
        # Renaming this key would strand every manifest already on disk:
        # _read_existing_split replays a rebuild from "pins" alone, so a
        # rebuild would redraw a holdout it was meant to keep. The key stays,
        # the readers read both.
        "pins": pins,
    }
    tmp_dir = _make_build_dir(final, _BUILD_DIR_PREFIX)
    try:
        _place_pairs(tmp_dir, pairs, pins)
        _write_yaml_pair(tmp_dir, _data_yaml_doc(final, source_dir))
        _write_json(os.path.join(tmp_dir, "split_manifest.json"), summary)
        _swap_into_place(tmp_dir, final)
        tmp_dir = ""
    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return summary


# ── Split record: what a manifest says about its split ───────────────────

def split_fingerprint(pins: Mapping[str, str]) -> str:
    """A short, stable fingerprint of a split assignment: two datasets with
    the same fingerprint hold the same images, by basename string, in the
    same splits.

    Scheme (SPLIT_FINGERPRINT_SCHEME): the basenames are sorted by Unicode
    code point; for each basename in that order, the basename and then its
    split are fed to SHA-256 as netstrings, ``<byte length>:<utf-8 bytes>,``;
    the fingerprint is the first SPLIT_FINGERPRINT_HEX_CHARS characters of the
    lowercase hex digest. Every step is fixed by a standard (SHA-256, UTF-8,
    code point order) and none depends on dict order, locale, platform, hash
    randomization or Python version, so the same assignment gives the same
    fingerprint on any machine under any Python 3.10 or newer. The guarantee
    is over the same basename strings: a filesystem that normalises Unicode
    names differently yields different strings and a different fingerprint.
    The netstring framing means a basename holding a tab or newline cannot
    collide with two separate basenames. A basename carrying undecodable
    bytes (a surrogateescape string from os.listdir) is encoded back to those
    bytes.

    Parameters: pins, {basename: split} with every split one of SPLIT_DIRS.
    Returns the fingerprint. Raises TypeError for a non-mapping or a
    non-string basename, ValueError for a split outside SPLIT_DIRS.

    Example: split_fingerprint({"b.jpg": "test", "a.jpg": "train"}) is
    "15b781a844dcfe7e", the SHA-256 of b"5:a.jpg,5:train,5:b.jpg,4:test,".
    """
    if not isinstance(pins, Mapping):
        raise TypeError(f"pins must be a mapping of basename to split, got {type(pins).__name__}")
    not_str = [k for k in pins if not isinstance(k, str)]
    if not_str:
        raise TypeError(f"every pinned basename must be a string, got {not_str[:3]!r}")
    digest = hashlib.sha256()
    for name in sorted(pins):
        split = pins[name]
        if split not in SPLIT_DIRS:
            raise ValueError(f"pin for {name!r} is {split!r}, expected one of {SPLIT_DIRS}")
        for field in (name, split):
            data = field.encode("utf-8", errors="surrogateescape")
            digest.update(f"{len(data)}:".encode("ascii") + data + b",")
    return digest.hexdigest()[:SPLIT_FINGERPRINT_HEX_CHARS]


@dataclass(frozen=True)
class SplitRecord:
    """What a split_manifest.json says about how its images were split.

    ``recorded`` is True only when the manifest carries every key in
    SPLIT_RECORD_KEYS (seed, valid_frac, test_frac, split_fingerprint), which
    build_detect_dataset writes since manifest schema version 2. A manifest
    from before that, or one written by pinned_split.py (transect holdout),
    reads as ``recorded`` False with ``missing`` naming the absent keys and
    each absent value None. ``n_pins`` counts the pinned basenames under
    "pins" (build_detect_dataset) or "pinned" (pinned_split.py), 0 when
    neither is present.
    """
    path: str
    recorded: bool
    missing: tuple
    seed: str | None
    valid_frac: float | None
    test_frac: float | None
    split_fingerprint: str | None
    split_fingerprint_scheme: str | None
    holdout_mode: str | None
    schema_version: int | None
    n_pins: int

    def same_split_as(self, other: "SplitRecord") -> bool:
        """True only when both records are recorded and carry the same
        fingerprint under the same scheme: the same images in the same
        splits. An unrecorded manifest is never the same split as anything,
        itself included, because nothing proves what it holds."""
        return bool(self.recorded and other.recorded
                    and self.split_fingerprint == other.split_fingerprint
                    and self.split_fingerprint_scheme == other.split_fingerprint_scheme)


def _manifest_field(doc: dict, key: str, kinds: tuple, path: Path, what: str) -> object:
    """doc[key] when it is present and of one of ``kinds``; None when absent
    or null; ValueError naming key and file otherwise. A bool never passes as
    a number."""
    value = doc.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, kinds):
        raise ValueError(f"split manifest at {path}: {key} must be {what}, got {value!r}")
    return value


def read_split_record(manifest_path: str | os.PathLike) -> SplitRecord:
    """Read what a split_manifest.json recorded about its split, touching
    nothing else.

    Parameters: manifest_path, the split_manifest.json, as str or Path.
    Returns a SplitRecord; its docstring says what ``recorded`` means.
    Raises FileNotFoundError when there is no such file, ValueError when the
    file is not a JSON object or a recorded value has the wrong type.

    Pure stdlib, so it imports with no Flask context and no ultralytics:
    later code decides whether two runs are comparable by reading both
    manifests and calling SplitRecord.same_split_as.

    The record deliberately stops at what makes two runs comparable, so
    pins_source (which of the three sources of pins a detection build used) is
    no part of a SplitRecord; a reader that needs it reads the manifest JSON
    itself. The fingerprint, not the source of the pins, is what says two
    datasets hold the same images in the same splits.

    Example: on a manifest written before 2026-09-04,
    read_split_record(step6_dir / "dataset" / "split_manifest.json") returns
    recorded=False and missing=("valid_frac", "test_frac", "split_fingerprint").
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"no split_manifest.json at {path}: the dataset has not been "
                                "split yet, or this is not a dataset folder")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"split manifest at {path} cannot be read as JSON: {e}") from e
    if not isinstance(doc, dict):
        raise ValueError(f"split manifest at {path} must be a JSON object, got "
                         f"{type(doc).__name__}")
    seed = _manifest_field(doc, "seed", (str,), path, "a string")
    valid_frac = _manifest_field(doc, "valid_frac", (int, float), path, "a number")
    test_frac = _manifest_field(doc, "test_frac", (int, float), path, "a number")
    fingerprint = _manifest_field(doc, "split_fingerprint", (str,), path, "a string")
    scheme = _manifest_field(doc, "split_fingerprint_scheme", (str,), path, "a string")
    holdout_mode = _manifest_field(doc, "holdout_mode", (str,), path, "a string")
    schema_version = _manifest_field(doc, "schema_version", (int,), path, "an integer")
    pins_key = "pins" if doc.get("pins") is not None else "pinned"
    pins = _manifest_field(doc, pins_key, (dict,), path, "an object of basename to split")
    values = {"seed": seed, "valid_frac": valid_frac, "test_frac": test_frac,
              "split_fingerprint": fingerprint}
    missing = tuple(k for k in SPLIT_RECORD_KEYS if values[k] is None)
    return SplitRecord(
        path=str(path),
        recorded=not missing,
        missing=missing,
        seed=seed,
        valid_frac=None if valid_frac is None else float(valid_frac),
        test_frac=None if test_frac is None else float(test_frac),
        split_fingerprint=fingerprint,
        split_fingerprint_scheme=scheme,
        holdout_mode=holdout_mode,
        schema_version=schema_version,
        n_pins=len(pins or {}),
    )


# ── Segmentation labels reduced to boxes ─────────────────────────────────

def _fmt(v):
    """A normalized coordinate the way YOLO label files carry it: six decimals."""
    return f"{v:.6f}"


def polygon_to_box(coords):
    """(cx, cy, w, h) of the extent of a normalized YOLO-seg polygon, every
    coordinate clamped to [0,1]. None when the line cannot enclose an area:
    fewer than three points, an odd coordinate count, or zero width or
    height once clamped.

    Clamping is a deliberate divergence from ultralytics, which asserts
    `points.max() <= 1.01` and refuses the whole label file otherwise
    (ultralytics/data/utils.py::verify_image_label). A seg export can carry a
    vertex a hair outside the frame, and dropping a real colony over it would
    be worse than trimming the box to the frame. A coordinate far outside
    [0,1] is trimmed too, so a genuinely non-normalized label (pixel
    coordinates, say) converts to a full-frame box here instead of failing
    loudly the way ultralytics would."""
    if len(coords) < _MIN_POLYGON_COORDS or len(coords) % 2:
        return None
    xs = [min(1.0, max(0.0, v)) for v in coords[0::2]]
    ys = [min(1.0, max(0.0, v)) for v in coords[1::2]]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    return (x0 + w / 2.0, y0 + h / 2.0, w, h)


def seg_labels_to_boxes(labels_dir, out_dir):
    """Write a box-label copy of every YOLO-seg label file in ``labels_dir``
    into ``out_dir``: each polygon line `class x1 y1 x2 y2 ...` becomes
    `class cx cy w h` from the polygon's bounding box.

    An empty label file stays empty: a frame with no mask is a negative and
    stays one. A line that cannot enclose an area, or whose class id is not a
    number, is dropped and counted.

    The three ways a label file can end up with no boxes are counted apart,
    because they mean different things to a reader:

        n_negative    the file was already empty. A real negative frame.
        n_emptied     the file held lines and EVERY one of them was dropped.
                      The frame silently becomes a background image, which is
                      a data problem worth seeing in the manifest.
        n_unreadable  the file could not be read. No output file is written
                      for it, and build_detect_from_seg_dataset refuses to
                      finish, because a missing label file would otherwise
                      turn an annotated frame into a background image with
                      nothing on record.

    Returns {n_files, n_boxes, n_dropped, n_negative, n_emptied, n_unreadable}.
    """
    os.makedirs(out_dir, exist_ok=True)
    counts = {"n_files": 0, "n_boxes": 0, "n_dropped": 0,
              "n_negative": 0, "n_emptied": 0, "n_unreadable": 0}
    if not os.path.isdir(labels_dir):
        return counts
    for fn in sorted(os.listdir(labels_dir)):
        if not fn.endswith(".txt"):
            continue
        counts["n_files"] += 1
        out_lines = []
        n_lines = 0
        try:
            with open(os.path.join(labels_dir, fn)) as fh:
                for line in fh:
                    parts = line.split()
                    if not parts:
                        continue
                    n_lines += 1
                    try:
                        cid = int(float(parts[0]))
                        coords = [float(v) for v in parts[1:]]
                    except ValueError:
                        counts["n_dropped"] += 1
                        continue
                    box = polygon_to_box(coords)
                    if box is None:
                        counts["n_dropped"] += 1
                        continue
                    out_lines.append(f"{cid} " + " ".join(_fmt(v) for v in box))
                    counts["n_boxes"] += 1
        except OSError:
            counts["n_unreadable"] += 1
            continue
        if not out_lines:
            counts["n_emptied" if n_lines else "n_negative"] += 1
        with open(os.path.join(out_dir, fn), "w") as fh:
            fh.write("".join(ln + "\n" for ln in out_lines))
    return counts


def dataset_label_kind(dataset_dir):
    """What the labels of a split dataset already are: "segment" when any line
    holds a polygon (three points or more), "detect" when every line is a
    four-number box, "empty" when no label file holds a single line.

    Every label file is read until the first polygon line settles it. There is
    deliberately no scan cap: reef clips are negative-heavy, so a cap large
    enough to be cheap is still small enough to hit a run of empty files and
    call a segmentation dataset "empty". The files are a few bytes each and
    this runs once, before a training run that takes hours.
    """
    seen_box = False
    for split in SPLIT_DIRS:
        label_dir = os.path.join(dataset_dir, split, "labels")
        if not os.path.isdir(label_dir):
            continue
        for fn in sorted(os.listdir(label_dir)):
            if not fn.endswith(".txt"):
                continue
            try:
                with open(os.path.join(label_dir, fn)) as fh:
                    for line in fh:
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        if len(parts) - 1 >= _MIN_POLYGON_COORDS:
                            return "segment"
                        seen_box = True
            except OSError:
                continue
    return "detect" if seen_box else "empty"


def _link(src, dst):
    """Relative symlink dst -> src, replacing whatever sat there."""
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)


def build_detect_from_seg_dataset(dataset_dir, detect_dir):
    """Write ``detect_dir`` as the detection view of the segmentation dataset
    at ``dataset_dir``: the same splits and the same images (relative symlinks,
    never copies), with every polygon label reduced to its bounding box.

    ``dataset_dir`` is not touched, so the same step-6 folder still trains a
    segmentation model from it afterwards. Returns
    {counts, conversions, n_dropped, data_yaml}."""
    src_yaml = os.path.join(dataset_dir, "data.yaml")
    if not os.path.isfile(src_yaml):
        raise ValueError(f"no data.yaml in {dataset_dir}: run the split step first")
    final = os.path.abspath(detect_dir)
    _refuse_symlink(final, "detect_dir")
    # The splits are the seg dataset's own, so its holdout record carries
    # across unchanged: an evaluation of a detection run reads the same
    # pinned or unpinned answer it would read for a segmentation run. A
    # record that cannot be read is refused, because carrying nothing across
    # would make the detection view read as an unpinned holdout.
    manifest = _load_json_object(
        os.path.join(dataset_dir, "split_manifest.json"),
        "fix or remove it before building the detection view, or its holdout record is lost") or {}

    tmp_dir = _make_build_dir(final, _DETECT_BUILD_DIR_PREFIX)
    try:
        counts = {}
        conversions = {}
        for split in SPLIT_DIRS:
            src_images = os.path.join(dataset_dir, split, "images")
            if not os.path.isdir(src_images):
                continue
            dst_images = os.path.join(tmp_dir, split, "images")
            os.makedirs(dst_images, exist_ok=True)
            n = 0
            for fn in sorted(os.listdir(src_images)):
                if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
                    continue
                # Relative to the FINAL dir, which sits at the same depth as
                # the temp build dir, so the link is correct after the rename.
                _link(os.path.join(os.path.abspath(src_images), fn),
                      os.path.join(dst_images, fn))
                n += 1
            counts[split] = n
            conv = seg_labels_to_boxes(os.path.join(dataset_dir, split, "labels"),
                                       os.path.join(tmp_dir, split, "labels"))
            # A label file that could not be read has no converted copy, and an
            # image with no label file trains as a background image. Refuse
            # rather than quietly drop annotated frames out of the training set.
            if conv["n_unreadable"]:
                raise ValueError(
                    f"{conv['n_unreadable']} label file(s) in "
                    f"{os.path.join(dataset_dir, split, 'labels')} could not be read; "
                    "fix or remove them before training a detector on this dataset")
            conversions[split] = conv

        doc = read_data_yaml(src_yaml)
        doc["path"] = final
        doc.setdefault("train", "train/images")
        doc["val"] = "valid/images"
        doc.setdefault("test", "test/images")
        _write_yaml_pair(tmp_dir, doc)

        manifest["derived_from"] = os.path.abspath(dataset_dir)
        manifest["task"] = "detect"
        manifest["derived_at"] = ast_now()
        _write_json(os.path.join(tmp_dir, "split_manifest.json"), manifest)

        summary = {
            "schema_version": 1,
            "created_at": ast_now(),
            "source_dataset": os.path.abspath(dataset_dir),
            "counts": counts,
            "conversions": conversions,
            "n_dropped": sum(c["n_dropped"] for c in conversions.values()),
            "n_emptied": sum(c["n_emptied"] for c in conversions.values()),
            "n_negative": sum(c["n_negative"] for c in conversions.values()),
            "data_yaml": os.path.join(final, "data.yaml"),
        }
        _write_json(os.path.join(tmp_dir, "detect_manifest.json"), summary)
        _swap_into_place(tmp_dir, final)
        tmp_dir = ""
    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return summary


def detect_dir_for(dataset_dir):
    """Where the detection view of ``dataset_dir`` belongs: dataset_detect/
    beside it."""
    return os.path.join(os.path.dirname(os.path.abspath(dataset_dir)), "dataset_detect")


def prepare(dataset_dir, detect_dir=""):
    """(data_yaml, summary) for a detection run over ``dataset_dir``.

    A dataset whose labels are already boxes trains from its own data.yaml and
    nothing is written (summary None). Anything else gets a dataset_detect/
    built beside it, and that data.yaml is returned.

    A dataset with no label line at all is built rather than passed through:
    it cannot be told apart from a segmentation dataset whose frames are all
    negatives, and converting all-empty labels yields all-empty labels, so
    building is correct either way while passing through would train a
    segmentation dataset's polygons as if they were boxes."""
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"dataset not found: {dataset_dir}")
    kind = dataset_label_kind(dataset_dir)
    if kind == "detect":
        return os.path.join(os.path.abspath(dataset_dir), "data.yaml"), None
    out = detect_dir or detect_dir_for(dataset_dir)
    summary = build_detect_from_seg_dataset(dataset_dir, out)
    return summary["data_yaml"], summary


def main(argv=None):
    """CLI entry: parse ``argv``, run the sub-command, return the exit code
    (0 on success, 2 when the dataset cannot be prepared)."""
    p = argparse.ArgumentParser(description="Detection dataset helpers for step 6")
    sub = p.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare", help="resolve (and build when needed) the data.yaml "
                                          "a detection run trains on")
    prep.add_argument("--dataset", required=True, help="the split dataset dir (<step6>/dataset)")
    prep.add_argument("--out", default="", help="where to write the detection view "
                                                "(default: dataset_detect beside --dataset)")
    args = p.parse_args(argv)

    try:
        data_yaml, summary = prepare(args.dataset, args.out)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if summary is None:
        print(f"[detect_dataset] {args.dataset} already holds box labels; training on its own "
              "data.yaml", file=sys.stderr)
    else:
        print(f"[detect_dataset] built {os.path.dirname(data_yaml)} from {args.dataset}: "
              f"{summary['counts']} images, {summary['n_dropped']} label lines dropped",
              file=sys.stderr)
    print(data_yaml)
    return 0


if __name__ == "__main__":
    sys.exit(main())

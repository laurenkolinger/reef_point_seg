#!/usr/bin/env python
"""Stage any YOLO-format bounding-box dataset into the module.

Module:  scripts/_boxes/ingest_yolo_boxes.py
Purpose: Copy an external object-detection dataset (an images folder, a labels
         folder holding one "class cx cy w h" text file per image, and a class
         list) into supporting_data/bbox_datasets/<name>/ in the same layout
         the Coral Intel stager writes, plus two optional files the split
         builder honours: names.csv (the full name behind every class token)
         and split_pins.json (the split every image had in its original
         experiment). Source file names are normalised on the way in, because
         a name carrying spaces, a non-ascii character or a .jfif extension is
         not a name the trainer can read back.
Inputs:  --images and --labels (the source folders), --name (the staged folder
         name), and at least one of --data-yaml (the dataset's class list) or
         --names (a token,name table). Optional: --split-from (a folder
         holding train/, valid/ and test/, each with images/ inside),
         --subject (coral, fish or algae, recorded for the Train panel's
         Subject control), --citation, --license, --dest-root, --limit,
         --quiet.
Outputs: supporting_data/bbox_datasets/<name>/ holding images/, labels/,
         classes.txt, data.yaml, names.csv, split_pins.json, rename_map.csv,
         LICENSE.md and ingest_manifest.json. The function returns the
         manifest it wrote.

Every staged basename is also unique ignoring letter case. The whole
supporting_data tree is kept inside a Synology Drive sync whose destination
cannot hold two names that differ only in case: at 01:26 AST on 2026-09-15 the
sync renamed labels/nurse_shark.txt, four minutes after this stager wrote it
beside labels/Nurse_shark.txt, and the image that label described was left with
none, so the dataset held one labelled pair fewer than its manifest counted and
every split fingerprint computed from it changed. Two source names that reduce
to one staged name are still refused, because there the operator has to look at
the source; two that differ only in case are both real, distinct files, so the
later one in sorted source-name order takes a numbered suffix on its stem
(_2, then _3), the label follows the stem, rename_map.csv records it like any
other rename, and the manifest counts it as n_case_disambiguated. The suffix is
the one step of staging that makes a name longer than the source name it came
from, so a suffix that would pass MAX_STAGED_NAME_BYTES is refused while the
run is still planning. The rule also covers the staged dataset folder itself,
which cannot match a sibling dataset folder in the same staging root only in
letter case.

Nothing is written until every check has passed. Each of these stops the run
with a message naming what failed, where, and why: a basename collision, two
records that would share one staged label file, a label line that is not
exactly five fields (a truncated line, or a segmentation polygon), a class id
outside the class list, a label with no image, a class token with no full name,
a missing split folder, an image pinned twice, a pinned image that is not among
the inputs, a case clash whose numbered suffix would pass the file-name limit,
a dataset folder matching a sibling dataset folder only in letter case, a
destination that already holds a name differing from one this run stages only
in letter case, and a destination that already holds files this run would not
write (an older stage, a Coral Intel labels_localized/ folder, or a stale
names.csv or split_pins.json).

This imagery is outside the TCRMP survey. Staging never writes to
inprocess/run_*, the mask registry, label provenance, the coverage matrix, or
STATS: a staged dataset is read where it is staged.

Usage:
  env/bin/python scripts/_boxes/ingest_yolo_boxes.py \
      --images <dir>/all_images --labels <dir>/all_labels \
      --data-yaml <dir>/data.yaml \
      --names scripts/_boxes/names/fish_photo_28feb25.csv \
      --split-from <dir>/split --name fish_photo_28feb25
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import namedtuple
from datetime import datetime, timedelta, timezone

import yaml

# Atlantic Standard Time, fixed UTC-4, no daylight saving (house rule: every
# timestamp this platform writes is AST 24h).
AST = timezone(timedelta(hours=-4))

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
REPO_DIR = os.path.dirname(_SCRIPTS)

# The Coral Intel stager owns the staged layout: its copy-when-changed rule,
# its data.yaml writer and its whole-frame box rule are imported rather than
# copied, so both staged datasets stay the same shape as that file changes.
if __package__:
    from . import ingest_coral_intel as coral
else:  # run directly as a script: the package root is not on sys.path yet
    if _SCRIPTS not in sys.path:
        sys.path.insert(0, _SCRIPTS)
    from _boxes import ingest_coral_intel as coral

# Image extensions the module's trainer reads back (detect_dataset.bbox_pairs).
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

# JFIF is JPEG under another extension; neither the trainer's extension list
# nor ultralytics 8.4 lists .jfif, so the staged copy carries .jpg.
EXT_RENAMES = {".jfif": ".jpg"}

# Everything outside this set becomes one underscore in a staged basename.
SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

# Two normalised characters side by side read as one separator, so "a_ñ_b"
# stages as "a_b" rather than "a__b".
UNDERSCORE_RUN = re.compile(r"_{2,}")

# A stem with nothing safe left in it still needs a name.
FALLBACK_STEM = "image"

# The first suffix a staged stem takes when an earlier source file already
# claimed that stem ignoring letter case: "nurse_shark" then "nurse_shark_2".
CASE_SUFFIX_START = 2

# The longest file name a staged copy may carry, in bytes: the POSIX NAME_MAX
# every filesystem under this tree honours. Every other step of safe_basename
# shortens a name or keeps its length, and a source file that exists on disk is
# already inside the limit, so the case suffix is the one step of staging that
# can reach it.
MAX_STAGED_NAME_BYTES = 255

SPLIT_DIRS = ("train", "valid", "test")

# The staged folder name is used in run folder names and in URLs, so it stays
# inside the platform's plain-name rule.
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,60}$")

SOURCE_KIND = "yolo"
NAMES_HEADER = ("token", "name")
SPLIT_PINS_SCHEMA_VERSION = 1
LABEL_EXT = ".txt"
PROGRESS_EVERY = 2000

# A YOLO detection label line is exactly "class cx cy w h". Four fields is a
# truncated line; more than five is a segmentation polygon, which describes a
# different geometry and must be converted to boxes before staging.
LABEL_FIELDS = 5

# Files the common annotation tools write into a labels folder beside the
# labels themselves (labelImg and Roboflow both write classes.txt). They are
# passed over rather than read as a label whose image is missing, because the
# raw source shares are read only and the operator cannot delete them there.
LABEL_SIDECARS = ("classes.txt", "notes.json", "data.yaml")

# The Coral Intel stager's extra labels folder. The split builder prefers it
# over labels/ (detect_dataset.BBOX_LABEL_DIRS), so a generic stage must never
# appear beside one left by an earlier Coral Intel stage.
LOCALIZED_LABEL_DIR = "labels_localized"

# What the operator does about a destination that already holds another run's
# files. Quoted in every leftover refusal so the remedy is always the same.
STALE_REMEDY = ("Stage into a different folder, or remove the staged copy "
                "first.")

# How many offending names a refusal lists before it counts the rest.
REFUSAL_EXAMPLES = 5

DEFAULT_DEST_ROOT = os.path.join(REPO_DIR, "supporting_data", "bbox_datasets")

# What a staged dataset may record itself as being about, written into the
# manifest as "subject". The Train panel reads it to start its own Subject
# control on the right value, which is half of what chooses a training recipe
# (the other half being the training task). The same three words are
# app.py's STEP6_SUBJECTS, and test_ingest_yolo_boxes.py holds the two lists
# against each other so they cannot drift apart. "" means a dataset that
# records no subject, which is every dataset staged before 2026-09-15.
DATASET_SUBJECTS = ("coral", "fish", "algae")
NO_SUBJECT = ""

NO_LICENSE_NOTE = ("No license recorded for this dataset. Local training only "
                   "until the terms are recorded.")
NO_CITATION = "none recorded"
COVERAGE_NOTE = ("Outside the TCRMP survey. This dataset never enters "
                 "inprocess/run_*, the mask registry, the label coverage "
                 "matrix, or STATS.")

# One staged record: the source basename, the staged basename that source
# basename would take on its own (before any case disambiguation), where the
# two source files were read from, and the basename the record is staged under.
Record = namedtuple("Record", "original natural staged image_path label_path")


class IngestError(RuntimeError):
    """Anything that stops the ingest before it stages a file."""


def ast_now():
    """The current Atlantic Standard Time as YYYY-MM-DD HH:MM:SS."""
    return datetime.now(AST).strftime("%Y-%m-%d %H:%M:%S")


# ── Names ────────────────────────────────────────────────────────────────

def safe_basename(filename):
    """The basename a source file is staged under.

    Runs of characters outside A-Za-z0-9._- become one underscore, two or more
    underscores in a row collapse to one, leading and trailing dots and
    underscores go, the extension is lower-cased, and .jfif becomes .jpg.

    Parameters:
      filename: a source basename, for example "Screen Shot 2023.png".
    Returns:
      the staged basename, for example "Screen_Shot_2023.png".
    """
    stem, ext = os.path.splitext(str(filename))
    ext = EXT_RENAMES.get(ext.lower(), ext.lower())
    stem = UNDERSCORE_RUN.sub("_", SAFE_CHARS.sub("_", stem)).strip("._")
    return (stem or FALLBACK_STEM) + ext


def check_dataset_name(name):
    """Refuse a staged folder name outside the platform's plain-name rule.

    Parameters:
      name: the folder name the dataset would be staged under.
    Returns:
      the same name, as a string.
    Raises:
      IngestError naming the offending value.
    """
    text = str(name)
    if not NAME_RE.match(text):
        raise IngestError(
            f"{text!r} is not a usable dataset folder name: use 1 to 60 "
            "characters from A to Z, a to z, 0 to 9, underscore and hyphen, "
            "because this name appears in run folder names and in URLs")
    return text


def check_subject(value):
    """Refuse a subject outside the vocabulary the Train panel reads.

    A misspelled subject would be recorded, carried into every subset, and
    then match no recipe, with nothing to say why, so it is refused while the
    operator is still at the command line.

    Parameters:
      value: what --subject was given, or "" or None for a dataset that
        records none.
    Returns:
      the lower-case vocabulary spelling, or "" when nothing was given.
    Raises:
      IngestError naming the offending value and the words that are accepted.

    Example: check_subject(" Coral ") returns "coral".
    """
    text = str(value or "").strip().lower()
    if not text:
        return NO_SUBJECT
    if text not in DATASET_SUBJECTS:
        raise IngestError(
            f"{value!r} is not a subject VICARIUS knows: use one of "
            f"{', '.join(DATASET_SUBJECTS)}, or leave --subject off to record "
            "none. The Train panel matches this word against what each "
            "training recipe declares it is for.")
    return text


# ── Readers ──────────────────────────────────────────────────────────────

def parse_boxes(text, nc, where):
    """Every box in one label file, as (class_id, [cx, cy, w, h]).

    Line-based on purpose: a label file with no trailing newline is common in
    exported datasets, so counting boxes by lines read is the only count that
    matches the source.

    Parameters:
      text: the label file's content.
      nc: how many classes the dataset declares.
      where: the label file's name, quoted back in any refusal.
    Returns:
      a list of (class_id, [cx, cy, w, h]) in file order.
    Raises:
      IngestError naming the file and the line number when a line does not
      carry exactly LABEL_FIELDS fields (a truncated line, or a segmentation
      polygon), when a field is not a number, or when the class id is outside
      the class list.
    """
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != LABEL_FIELDS:
            raise IngestError(
                f"{where} line {n}: expected {LABEL_FIELDS} fields "
                f"'class cx cy w h', got {len(fields)}. A file of polygon "
                "(segmentation) labels must be converted to boxes before it "
                "is staged")
        try:
            cls = int(fields[0])
            coords = [float(v) for v in fields[1:LABEL_FIELDS]]
        except ValueError:
            raise IngestError(
                f"{where} line {n}: expected 'class cx cy w h', got {line.strip()!r}")
        if not 0 <= cls < nc:
            raise IngestError(
                f"{where} line {n}: class id {cls} is outside the {nc}-class list")
        out.append((cls, coords))
    return out


def read_names_csv(path):
    """The full name behind every class token, in file order.

    Parameters:
      path: a CSV whose first row is the header token,name.
    Returns:
      a list of (token, full name).
    Raises:
      IngestError when the file is missing, empty, carries another header, has
      a row that is not two fields, or names one token twice.
    """
    full = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(full):
        raise IngestError(f"names table not found: {full}")
    try:
        with open(full, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise IngestError(f"names table {full} could not be read: {exc}")
    if not rows:
        raise IngestError(
            f"names table {full} is empty: it needs a token,name header and "
            "one row per class token")
    header = tuple(cell.strip().lower() for cell in rows[0][:2])
    if header != NAMES_HEADER:
        raise IngestError(
            f"names table {full} must start with a token,name header, got "
            f"{','.join(rows[0]) or '(an empty first row)'}")
    out = []
    seen = {}
    for n, row in enumerate(rows[1:], 2):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) < 2:
            raise IngestError(
                f"{full} line {n}: expected 'token,name', got {','.join(row)!r}")
        token, name = row[0].strip(), row[1].strip()
        if not token or not name:
            raise IngestError(
                f"{full} line {n}: both the token and the full name are "
                f"required, got {row[:2]!r}")
        if token in seen:
            raise IngestError(
                f"{full} line {n}: token {token!r} is already named on line "
                f"{seen[token]}, so the full name would be ambiguous")
        seen[token] = n
        out.append((token, name))
    if not out:
        raise IngestError(
            f"names table {full} carries a header but no rows")
    return out


def read_class_list(data_yaml_path):
    """The dataset's class tokens in class-id order.

    Parameters:
      data_yaml_path: an ultralytics data.yaml whose names is a mapping keyed
        by class id or a list in class-id order.
    Returns:
      a list of class tokens, index equal to the class id.
    Raises:
      IngestError when the file is missing or unreadable, declares no names,
      keys them by anything but integers, skips an id, or declares an nc that
      disagrees with the names it lists.
    """
    full = os.path.abspath(os.path.expanduser(str(data_yaml_path)))
    if not os.path.isfile(full):
        raise IngestError(f"data.yaml not found: {full}")
    try:
        with open(full, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise IngestError(f"data.yaml {full} could not be read: {exc}")
    if not isinstance(doc, dict):
        raise IngestError(
            f"data.yaml {full} is not a YAML mapping, so it declares no class list")
    names = doc.get("names")
    if names is None:
        raise IngestError(
            f"data.yaml {full} declares no names, so the class list cannot be read")
    classes = _classes_from_names(names, full)
    if not classes or any(not token for token in classes):
        raise IngestError(
            f"data.yaml {full}: every class needs a non-empty token, got {classes}")
    duplicates = sorted({t for t in classes if classes.count(t) > 1})
    if duplicates:
        raise IngestError(
            f"data.yaml {full} names the same token twice ({', '.join(duplicates)}), "
            "so a box could not be traced back to one class")
    nc = doc.get("nc")
    if nc is not None and str(nc) != str(len(classes)):
        raise IngestError(
            f"data.yaml {full}: nc is {nc} but the names list holds "
            f"{len(classes)} classes")
    return classes


def _classes_from_names(names, where):
    """Class tokens in id order from a data.yaml names mapping or list.

    Parameters:
      names: the value of the data.yaml names key.
      where: the data.yaml path, quoted back in any refusal.
    Returns:
      a list of class tokens.
    Raises:
      IngestError when the mapping is not keyed by contiguous integers from 0,
      or when names is neither a mapping nor a list.
    """
    if isinstance(names, dict):
        try:
            keys = sorted(names, key=lambda k: int(k))
            ids = [int(k) for k in keys]
        except (TypeError, ValueError):
            raise IngestError(
                f"data.yaml {where}: the names mapping must be keyed by "
                f"integer class ids, got {sorted(map(str, names))}")
        if ids != list(range(len(ids))):
            raise IngestError(
                f"data.yaml {where}: the class ids must be contiguous from 0, "
                f"got {ids}")
        return [str(names[k]).strip() for k in keys]
    if isinstance(names, list):
        return [str(value).strip() for value in names]
    raise IngestError(
        f"data.yaml {where}: names must be a mapping of class id to token or "
        f"a list in class-id order, got {type(names).__name__}")


def normalise_limit(limit):
    """The record cap as a whole number, 0 meaning every record.

    Parameters:
      limit: None, 0, a whole number, or a string holding one.
    Returns:
      an int: 0 for every record, otherwise how many records to stage.
    Raises:
      IngestError naming the value when it is not a whole number or is
      negative.
    """
    if limit is None:
        return 0
    try:
        cap = int(limit)
    except (TypeError, ValueError):
        raise IngestError(
            f"the record cap {limit!r} is not a whole number: give the number "
            "of records to stage, or 0 for every record")
    if cap < 0:
        raise IngestError(
            f"the record cap is {cap}: give the number of records to stage, "
            "or 0 for every record")
    return cap


def plan_records(images_dir, labels_dir, limit=None):
    """Pair every source image with its label file and its staged basename.

    Two passes: _natural_names gives every source file the basename
    safe_basename makes of it on its own and refuses the two collisions that
    only the operator can resolve, then _case_safe_names settles the names that
    clash with one another ignoring letter case. Validation always reads every
    file in the folders, so a limited run carries the same refusals as a full
    one, and a limited run's names are the names the full run would give, so a
    prefix of the dataset is named the same way the whole of it is.

    Parameters:
      images_dir: folder holding the source images.
      labels_dir: folder holding one YOLO text file per labelled image.
      limit: stage only the first N records in sorted source-name order (None
        or 0 stages every record).
    Returns:
      (records, skipped): records is a list of Record in sorted source-name
      order, skipped is the sorted list of names this stager passed over, each
      prefixed with the folder it sits in.
    Raises:
      IngestError when either folder is missing, when limit is not a whole
      number or is negative, when the images folder holds no image, when two
      names collide after normalisation, when two records would share one
      staged label file, when a case clash cannot take a numbered suffix and
      still fit inside MAX_STAGED_NAME_BYTES, or when a label file has no
      image.
    """
    cap = normalise_limit(limit)
    img_dir = os.path.abspath(os.path.expanduser(str(images_dir)))
    lab_dir = os.path.abspath(os.path.expanduser(str(labels_dir)))
    if not os.path.isdir(img_dir):
        raise IngestError(f"images folder not found: {img_dir}")
    if not os.path.isdir(lab_dir):
        raise IngestError(f"labels folder not found: {lab_dir}")

    natural, skipped = _natural_names(img_dir)
    if not natural:
        raise IngestError(
            f"no image files in {img_dir}: this stager reads "
            f"{', '.join(IMAGE_EXTS + tuple(EXT_RENAMES))}")
    staged_by_source = _case_safe_names(natural)

    records = []
    for filename, first_choice in natural:
        label = os.path.join(lab_dir, os.path.splitext(filename)[0] + LABEL_EXT)
        records.append(Record(filename, first_choice,
                              staged_by_source[filename],
                              os.path.join(img_dir, filename),
                              label if os.path.isfile(label) else ""))

    sidecars = _check_orphan_labels(lab_dir, {r.original for r in records})
    skipped += ["labels/" + name for name in sidecars]
    if cap:
        records = records[:cap]
    return records, sorted(skipped)


def _natural_names(img_dir):
    """Every source image paired with the basename it would stage under alone.

    The name safe_basename gives a file on its own, before any case
    disambiguation. The two collisions the operator has to resolve are refused
    here, because a run that cannot name its records has nothing to settle.

    Parameters:
      img_dir: folder holding the source images.
    Returns:
      (natural, skipped): natural is [(source basename, staged basename)] in
      sorted source-name order, skipped is the list of names in this folder
      that are not images, each prefixed with "images/".
    Raises:
      IngestError when two source names reduce to one staged name, or when two
      source names reduce to one staged stem and would therefore share one
      staged label file.
    """
    natural, skipped, claimed, stems = [], [], {}, {}
    for filename in sorted(os.listdir(img_dir)):
        path = os.path.join(img_dir, filename)
        ext = os.path.splitext(filename)[1].lower()
        if not os.path.isfile(path) or (
                ext not in IMAGE_EXTS and ext not in EXT_RENAMES):
            skipped.append("images/" + filename)
            continue
        staged = safe_basename(filename)
        if staged in claimed:
            raise IngestError(
                f"two source files would be staged as {staged!r}: "
                f"{claimed[staged]!r} and {filename!r}. Rename one at the "
                "source, then stage again")
        claimed[staged] = filename
        stem = os.path.splitext(staged)[0]
        if stem in stems:
            raise IngestError(
                f"two source files would be staged under the stem {stem!r}, so "
                f"the one staged label file {stem + LABEL_EXT!r} would have to "
                f"describe both {stems[stem]!r} and {filename!r}, and one image "
                "would train on the other image's boxes. Rename one at the "
                "source, then stage again")
        stems[stem] = filename
        natural.append((filename, staged))
    return natural, skipped


def _case_safe_names(natural):
    """The staged basename every source file takes, unique ignoring case.

    Of the files whose staged stems differ only in letter case, the first in
    sorted source-name order keeps its stem and every later one takes the first
    free numbered suffix. Sorted source-name order is the only input, so
    staging one folder twice gives the same staged names both times, and a
    suffix never takes a stem another source file already owns.

    Parameters:
      natural: [(source basename, staged basename)] in sorted source-name
        order, as _natural_names returned it.
    Returns:
      {source basename: staged basename}.
    Raises:
      IngestError naming the source file when no free suffix was found, or
      when the suffix it needs would make a file name longer than
      MAX_STAGED_NAME_BYTES.

    Example:
      _case_safe_names([("Nurse shark.JPG", "Nurse_shark.jpg"),
                        ("nurse shark.png", "nurse_shark.png")])
      gives {"Nurse shark.JPG": "Nurse_shark.jpg",
             "nurse shark.png": "nurse_shark_2.png"}.
    """
    taken = {os.path.splitext(staged)[0].lower() for _source, staged in natural}
    settled, assigned = set(), {}
    for filename, staged in natural:
        stem, ext = os.path.splitext(staged)
        key = stem.lower()
        if key in settled:
            stem = _free_stem(stem, ext, taken, filename)
            taken.add(stem.lower())
            staged = stem + ext
        else:
            settled.add(key)
        assigned[filename] = staged
    return assigned


def _free_stem(stem, ext, taken, filename):
    """The first numbered form of a stem that no staged name uses and fits on disk.

    Parameters:
      stem: the staged stem an earlier source file already claimed.
      ext: the staged extension this record carries, so a candidate is
        measured as the file name it will become.
      taken: every staged stem already in use, lower-cased.
      filename: the source file being named, quoted back in any refusal.
    Returns:
      the free stem, for example "nurse_shark_2".
    Raises:
      IngestError naming the source file when the first free candidate would
      make a file name longer than MAX_STAGED_NAME_BYTES, or when every
      candidate was in use, which the search below cannot reach.
    """
    # Each candidate that fails rules out one distinct name in taken, so one
    # more candidate than taken holds is always enough.
    ceiling = CASE_SUFFIX_START + len(taken) + 1
    for suffix in range(CASE_SUFFIX_START, ceiling):
        candidate = f"{stem}_{suffix}"
        if candidate.lower() in taken:
            continue
        _check_staged_name_fits(candidate, ext, suffix, filename)
        return candidate
    raise IngestError(
        f"{filename!r} could not be given a staged name that is unique "
        f"ignoring letter case: {stem!r} and every numbered form of it up to "
        f"{ceiling - 1} are already in use. Rename it at the source, then "
        "stage again")


def _check_staged_name_fits(stem, ext, suffix, filename):
    """Refuse a numbered stem whose file name would not fit on disk.

    A record is staged as two files, the image under stem + ext and its label
    under stem + LABEL_EXT, so the longer of the two is what has to fit. The
    check runs here, while the run is still planning its names, because a name
    refused during the copy would stop the run part way through and leave a
    half-staged folder behind.

    Parameters:
      stem: the numbered stem the suffix search settled on.
      ext: the staged extension of the image.
      suffix: the number the search added, quoted back in the refusal.
      filename: the source file being named, quoted back in the refusal.
    Returns:
      None.
    Raises:
      IngestError naming the source file, the suffix it needed, the length its
      staged name reached and the limit.
    """
    reached = max(len((stem + ext).encode("utf-8")),
                  len((stem + LABEL_EXT).encode("utf-8")))
    if reached <= MAX_STAGED_NAME_BYTES:
        return
    raise IngestError(
        f"{filename!r} shares a staged stem with an earlier source file "
        f"ignoring letter case, so it needs the suffix '_{suffix}' to be "
        f"staged beside it, and its staged name would then be {reached} bytes "
        f"against the {MAX_STAGED_NAME_BYTES}-byte file-name limit this "
        "filesystem holds. Shorten the name at the source, then stage again")


def _check_orphan_labels(labels_dir, image_names):
    """Refuse a label file whose image is not among the inputs.

    A known annotation-tool sidecar (LABEL_SIDECARS) is not a label file and is
    reported as passed over instead of refused.

    Parameters:
      labels_dir: folder holding the label files.
      image_names: the source image basenames that were found.
    Returns:
      the sorted sidecar names this folder holds, so the caller can report
      them.
    Raises:
      IngestError naming the first orphan label files, how many there are, and
      the remedy.
    """
    stems = {os.path.splitext(name)[0] for name in image_names}
    try:
        present = sorted(os.listdir(labels_dir))
    except OSError as exc:
        raise IngestError(f"labels folder {labels_dir} could not be read: {exc}")
    sidecars = [name for name in present if name.lower() in LABEL_SIDECARS]
    orphans = [name for name in present
               if name.endswith(LABEL_EXT)
               and name.lower() not in LABEL_SIDECARS
               and os.path.splitext(name)[0] not in stems]
    if not orphans:
        return sidecars
    raise IngestError(
        f"{len(orphans)} label file(s) in {labels_dir} have no image among the "
        "inputs, so their boxes would train on nothing: "
        + ", ".join(orphans[:REFUSAL_EXAMPLES])
        + (f" (and {len(orphans) - REFUSAL_EXAMPLES} more)"
           if len(orphans) > REFUSAL_EXAMPLES else "")
        + ". Remove them, or point --labels at a folder holding only the label "
          "files for these images")


def read_split_pins(split_root, staged_by_first_choice, drop_unstaged=False):
    """The split every image was assigned in its original experiment.

    A split tree names its images by their source basenames, so a pin is
    matched on the name safe_basename makes of that basename and then follows
    the record to the staged name it settled on, which for a case clash carries
    a numbered suffix the split tree knows nothing about.

    Parameters:
      split_root: a folder holding train/, valid/ and test/, each with an
        images/ folder inside holding that split's image files.
      staged_by_first_choice: {the name safe_basename gives a source file on
        its own: the staged basename this ingest is about to write}.
      drop_unstaged: pass over a pinned image that is not being staged instead
        of refusing it. Only a limited run sets this, because there the
        operator asked for a prefix of the dataset on purpose.
    Returns:
      {staged basename: "train" | "valid" | "test"}.
    Raises:
      IngestError when the split root or one of its three folders is missing,
      when an image is pinned to two splits, when a pinned image is not among
      the staged names, or when the tree pins nothing at all.
    """
    root = os.path.abspath(os.path.expanduser(str(split_root)))
    if not os.path.isdir(root):
        raise IngestError(f"split folder not found: {root}")
    known = dict(staged_by_first_choice)
    pins = {}
    for split in SPLIT_DIRS:
        folder = os.path.join(root, split, "images")
        if not os.path.isdir(folder):
            raise IngestError(
                f"split folder not found: {folder}. A pinned dataset needs "
                f"{', '.join(SPLIT_DIRS)}, each holding an images folder")
        for filename in sorted(os.listdir(folder)):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMAGE_EXTS and ext not in EXT_RENAMES:
                continue
            first_choice = safe_basename(filename)
            if first_choice not in known:
                if drop_unstaged:
                    continue
                raise IngestError(
                    f"{split}/images/{filename} is pinned to the {split} "
                    f"split but {first_choice!r} is not among the images being "
                    "staged, so the pin would name nothing")
            staged = known[first_choice]
            if staged in pins:
                raise IngestError(
                    f"{staged!r} is pinned to two splits, {pins[staged]} and "
                    f"{split}, so its held-out status would be ambiguous")
            pins[staged] = split
    if not pins:
        raise IngestError(
            f"the split tree under {root} pins no image that is being staged")
    return pins


# ── Staging ──────────────────────────────────────────────────────────────

def _scan_labels(records, nc):
    """Count what the label files hold, refusing any line that is not a box.

    Parameters:
      records: the Record list this ingest will stage.
      nc: how many classes the dataset declares.
    Returns:
      a dict with n_boxes, n_degenerate, n_degenerate_exact, n_background and
      n_duplicate_lines. A record with no label file, or an empty one, is
      background: it carries no box and trains the model on what a reef
      without the target looks like.
    Raises:
      IngestError from parse_boxes, naming the file and the line.
    """
    counts = {"n_boxes": 0, "n_degenerate": 0, "n_degenerate_exact": 0,
              "n_background": 0, "n_duplicate_lines": 0}
    for record in records:
        if not record.label_path:
            counts["n_background"] += 1
            continue
        try:
            with open(record.label_path, encoding="utf-8") as fh:
                text = fh.read()
        except (OSError, UnicodeDecodeError) as exc:
            raise IngestError(
                f"label file {record.label_path} could not be read: {exc}")
        where = os.path.basename(record.label_path)
        boxes = parse_boxes(text, nc, where)
        if not boxes:
            counts["n_background"] += 1
        counts["n_boxes"] += len(boxes)
        counts["n_degenerate"] += sum(1 for _cls, coords in boxes
                                      if coral.is_degenerate(coords))
        lines = [line for line in text.splitlines() if line.split()]
        counts["n_degenerate_exact"] += sum(
            1 for line in lines if coral.is_exact_degenerate_line(line))
        counts["n_duplicate_lines"] += _count_repeats(lines)
    return counts


def _count_repeats(lines):
    """How many label lines repeat a line already seen in the same file.

    Whitespace between fields is ignored, so "0 0.5 0.5 1 1" and
    "0  0.5 0.5 1 1" count as the same box written twice.

    Parameters:
      lines: the file's non-empty lines.
    Returns:
      the number of lines beyond the first occurrence of each distinct box.
    """
    seen, repeats = set(), 0
    for line in lines:
        key = " ".join(line.split())
        if key in seen:
            repeats += 1
        else:
            seen.add(key)
    return repeats


def _write_csv(path, header, rows):
    """Write a small CSV with POSIX line endings.

    Parameters:
      path: the file to write.
      header: the header row.
      rows: the data rows.
    Returns:
      None.
    Raises:
      IngestError naming the file when the write fails.
    """
    try:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
    except OSError as exc:
        raise IngestError(f"{path} could not be written: {exc}")


def _write_json(path, doc):
    """Write a JSON document through a temp file, so a reader never sees half of one.

    The temp file carries this process's id, so two stagers running at once
    cannot interleave their bytes into one another's temp file.

    Parameters:
      path: the file to write.
      doc: the object to serialise.
    Returns:
      None.
    Raises:
      IngestError naming the file when the write fails.
    """
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        os.replace(tmp, path)
    except OSError as exc:
        raise IngestError(f"{path} could not be written: {exc}")


def _check_destination_clean(dest, records, names, pins):
    """Refuse a destination that already holds another run's files.

    A staged dataset is described by its manifest and trained on as a folder,
    so a folder holding files this run would not write would train on more
    images than its card and its manifest report. Every check runs before the
    first copy.

    Parameters:
      dest: the staged dataset folder.
      records: the Record list this run will stage.
      names: [(token, full name)] this run will write, or an empty list.
      pins: {staged basename: split} this run will write, empty for none.
    Returns:
      None.
    Raises:
      IngestError naming the folder, how many files it holds that this run
      would not write, the first of them, and the remedy; or naming a sibling
      dataset folder that matches this one only in letter case.
    """
    _check_folder_case_twin(dest)
    localized = os.path.join(dest, LOCALIZED_LABEL_DIR)
    if os.path.isdir(localized):
        raise IngestError(
            f"{localized} was written by the Coral Intel stager, and the split "
            f"builder prefers {LOCALIZED_LABEL_DIR}/ over labels/, so a dataset "
            f"staged here would train on the older labels. {STALE_REMEDY}")

    expected = {
        "images": {record.staged for record in records},
        "labels": {os.path.splitext(record.staged)[0] + LABEL_EXT
                   for record in records if record.label_path},
    }
    for sub, keep in expected.items():
        folder = os.path.join(dest, sub)
        if not os.path.isdir(folder):
            continue
        try:
            present = sorted(os.listdir(folder))
        except OSError as exc:
            raise IngestError(f"{folder} could not be read: {exc}")
        _check_case_twins(folder, keep, present)
        extras = [name for name in present if name not in keep]
        if extras:
            raise IngestError(
                f"{folder} already holds {len(extras)} file(s) this run would "
                f"not write, starting with {extras[0]!r}. They would stay where "
                f"they are while the manifest counted only the {len(keep)} file(s) "
                "this run stages, so the trainer would train on more data than "
                f"the dataset card describes. {STALE_REMEDY}")

    for sidecar, writing in (("names.csv", bool(names)),
                             ("split_pins.json", bool(pins))):
        path = os.path.join(dest, sidecar)
        if os.path.exists(path) and not writing:
            raise IngestError(
                f"{path} was written by an earlier run and this run writes no "
                f"{sidecar}, so the split builder would read the older file "
                f"against these images. {STALE_REMEDY}")


def _check_folder_case_twin(dest):
    """Refuse a dataset folder that matches a sibling folder only by case.

    The same rule as _check_case_twins, one level up. Two staged datasets whose
    folder names differ only in letter case are renamed by the sync exactly as
    two staged files would be, and every reader that looks a dataset up by its
    folder name, the Boxes dataset list and the training source list, stops
    finding the one that was renamed.

    Parameters:
      dest: the staged dataset folder this run would write.
    Returns:
      None.
    Raises:
      IngestError naming the folder on disk, the folder this run would create,
      and why the two cannot sit side by side.
    """
    parent = os.path.dirname(dest)
    name = os.path.basename(dest)
    if not os.path.isdir(parent):
        return
    try:
        siblings = sorted(os.listdir(parent))
    except OSError as exc:
        raise IngestError(f"{parent} could not be read: {exc}")
    twins = [entry for entry in siblings
             if entry != name and entry.lower() == name.lower()]
    if not twins:
        return
    raise IngestError(
        f"{os.path.join(parent, twins[0])} differs from {name!r}, the dataset "
        "folder this run would create beside it, only in letter case. This "
        "tree is kept inside a Synology Drive sync whose destination cannot "
        "hold both names, so one of the two folders would be renamed minutes "
        "after the stage and every reader that looks a dataset up by its "
        f"folder name would stop finding it. {STALE_REMEDY}")


def _check_case_twins(folder, keep, present):
    """Refuse a staged file that differs from one this run writes only by case.

    This run's own names are already unique ignoring letter case, so the one
    remaining way to end with two names the sync cannot both hold is a file an
    earlier run left in the destination under the other case. Claiming a name
    means claiming it against the destination as well as against this run's own
    choices, and a name already on disk is claimed by whoever put it there.

    Parameters:
      folder: the staged images/ or labels/ folder, quoted back in the refusal.
      keep: the basenames this run writes into that folder.
      present: the basenames the folder already holds.
    Returns:
      None.
    Raises:
      IngestError naming the file on disk, the name this run stages, and why
      the two cannot sit side by side.
    """
    ours = {name.lower(): name for name in keep}
    twins = [(name, ours[name.lower()]) for name in present
             if name not in keep and name.lower() in ours]
    if not twins:
        return
    on_disk, writing = twins[0]
    raise IngestError(
        f"{os.path.join(folder, on_disk)} differs from {writing!r}, which this "
        f"run stages into the same folder, only in letter case, and {len(twins)} "
        "name(s) here are in that position. This tree is kept inside a Synology "
        "Drive sync whose destination cannot hold both names, so one of the two "
        f"would be renamed minutes after the stage. {STALE_REMEDY}")


def _make_staging_dirs(dest):
    """Create the staged dataset's images/ and labels/ folders.

    Parameters:
      dest: the staged dataset folder.
    Returns:
      None.
    Raises:
      IngestError naming the folder that could not be created and why.
    """
    for sub in ("images", "labels"):
        folder = os.path.join(dest, sub)
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as exc:
            raise IngestError(f"{folder} could not be created: {exc}")


def _write_text(path, text):
    """Write one small text file as utf-8.

    Parameters:
      path: the file to write.
      text: its whole content.
    Returns:
      None.
    Raises:
      IngestError naming the file when the write fails.
    """
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    except OSError as exc:
        raise IngestError(f"{path} could not be written: {exc}")


def _copy_records(dest, records, log):
    """Copy every record's image and label into the staged folder.

    Parameters:
      dest: the staged dataset folder.
      records: the Record list to stage.
      log: where progress lines go.
    Returns:
      how many files were copied this run (a file already staged and unchanged
      is not copied again).
    Raises:
      IngestError naming the file when a copy fails.
    """
    copied = 0
    for i, record in enumerate(records, 1):
        staged_label = os.path.splitext(record.staged)[0] + LABEL_EXT
        pairs = [(record.image_path, os.path.join(dest, "images", record.staged))]
        if record.label_path:
            pairs.append((record.label_path,
                          os.path.join(dest, "labels", staged_label)))
        for src, dst in pairs:
            try:
                if coral._copy_if_changed(src, dst):
                    copied += 1
            except OSError as exc:
                raise IngestError(f"{src} could not be staged as {dst}: {exc}")
        if i % PROGRESS_EVERY == 0:
            log(f"staged {i} of {len(records)} records")
    return copied


def ingest_yolo(images_dir, labels_dir, dest, *, classes, names, names_csv="",
                split_root=None, citation="", license_note="", limit=None,
                subject="", log=print):
    """Stage a YOLO box dataset and return the manifest it wrote.

    Nothing is written until every check has passed, and a second run over the
    same destination copies only the files that changed. A destination already
    holding files this run would not write is refused rather than added to,
    because those files would stay on disk and be trained on while the manifest
    counted only what this run staged. Every staged basename is unique ignoring
    letter case, both among this run's own names and against what the
    destination already holds, because the staged tree is kept inside a
    Synology Drive sync whose destination cannot hold two names that differ
    only in case.

    Parameters:
      images_dir: folder holding the source images (symlinks are followed and
        their targets copied, so the staged dataset stands on its own).
      labels_dir: folder holding one YOLO text file per labelled image.
      dest: the staged dataset folder; its basename is the dataset name.
      classes: class tokens in class-id order.
      names: [(token, full name)] covering every class token, or an empty list
        when the dataset ships no names table.
      names_csv: where those rows were read from, recorded in the manifest.
      split_root: a folder holding train/, valid/ and test/ with images/
        inside each, or None for a dataset with no pinned split.
      citation: how the dataset must be cited, or "" for none recorded.
      license_note: the license terms, or "" for none recorded.
      limit: stage only the first N records in sorted source-name order; None
        or 0 stages every record.
      subject: what the dataset is about, one of DATASET_SUBJECTS, or "" to
        record none. The Train panel starts its Subject control on it.
      log: where progress lines go.
    Returns:
      the manifest dict, identical to the ingest_manifest.json it wrote.
    Raises:
      IngestError naming what failed, where, and why: a bad class list, a bad
      record cap, any refusal from plan_records, _scan_labels or
      read_split_pins, a case clash whose numbered suffix would make a file
      name longer than MAX_STAGED_NAME_BYTES, a dataset folder matching a
      sibling dataset folder only in letter case, a destination already
      holding a name that differs from one this run stages only in letter
      case, a destination holding another run's files, or a file that could
      not be written.
    """
    dest = os.path.abspath(os.path.expanduser(str(dest)))
    name = check_dataset_name(os.path.basename(dest.rstrip(os.sep)))
    subject = check_subject(subject)
    classes = [str(token).strip() for token in (classes or [])]
    if not classes:
        raise IngestError(
            "the class list is empty: give --data-yaml or a names table, "
            "because a dataset with no classes trains nothing")
    duplicates = sorted({t for t in classes if classes.count(t) > 1})
    if duplicates:
        raise IngestError(
            f"the class list names the same token twice ({', '.join(duplicates)})")
    _check_names_cover(classes, names)

    cap = normalise_limit(limit)
    records, skipped = plan_records(images_dir, labels_dir, limit=cap)
    counts = _scan_labels(records, len(classes))
    staged_by_first_choice = {record.natural: record.staged
                              for record in records}
    pins = (read_split_pins(split_root, staged_by_first_choice,
                            drop_unstaged=bool(cap))
            if split_root else {})
    _check_destination_clean(dest, records, names, pins)
    if skipped:
        log(f"{len(skipped)} file(s) in the source folders are not label or "
            f"image files and were passed over, starting with {skipped[0]}")

    _make_staging_dirs(dest)
    copied = _copy_records(dest, records, log)

    renamed = [(r.original, r.staged) for r in records if r.original != r.staged]
    disambiguated = [r for r in records if r.staged != r.natural]
    _write_text(os.path.join(dest, "classes.txt"),
                "".join(token + "\n" for token in classes))
    try:
        coral._write_data_yaml(dest, classes)
    except OSError as exc:
        raise IngestError(
            f"{os.path.join(dest, 'data.yaml')} could not be written: {exc}")
    if names:
        _write_csv(os.path.join(dest, "names.csv"), NAMES_HEADER, names)
    _write_csv(os.path.join(dest, "rename_map.csv"), ("original", "staged"), renamed)
    if pins:
        _write_json(os.path.join(dest, "split_pins.json"),
                    {"schema_version": SPLIT_PINS_SCHEMA_VERSION,
                     "source": os.path.abspath(os.path.expanduser(str(split_root))),
                     "pins": pins})
    license_text = license_note.strip() or NO_LICENSE_NOTE
    _write_text(os.path.join(dest, "LICENSE.md"),
                "# License\n\n" + license_text + "\n")

    manifest = _build_manifest(
        name=name, dest=dest, records=records, classes=classes, counts=counts,
        pins=pins, renamed=renamed, n_case_disambiguated=len(disambiguated),
        limit=cap, citation=citation, license_text=license_text, names=names,
        names_csv=names_csv, images_dir=images_dir, labels_dir=labels_dir,
        subject=subject)
    _write_json(os.path.join(dest, "ingest_manifest.json"), manifest)

    log(f"staged {manifest['n_images']} images, {manifest['n_boxes']} boxes "
        f"({manifest['n_degenerate']} whole-frame, {manifest['n_background']} "
        f"background), {manifest['n_classes']} classes -> {dest}")
    log(f"{copied} file(s) copied this run, {len(renamed)} basename(s) normalised")
    if disambiguated:
        log(f"{len(disambiguated)} staged name(s) took a numbered suffix so that "
            "no two staged names differ only in letter case: this tree is kept "
            "inside a Synology Drive sync whose destination cannot hold both "
            "names, and renames one of them minutes after the stage. "
            f"rename_map.csv records each one, starting with "
            f"{disambiguated[0].original!r} staged as {disambiguated[0].staged!r}")
    _log_pin_counts(manifest, log)
    return manifest


def _log_pin_counts(manifest, log):
    """Say how much of the staged dataset carries a split pin.

    Parameters:
      manifest: the manifest _build_manifest returned.
      log: where the line goes.
    Returns:
      None.
    """
    if not manifest["split_pins"]:
        log("no split pins: the split builder assigns every image by its "
            "seeded shuffle")
        return
    by_split = manifest["pins_by_split"]
    line = (f"{manifest['n_pinned']} of {manifest['n_images']} image(s) pinned "
            f"to a split (train {by_split['train']}, valid {by_split['valid']}, "
            f"test {by_split['test']})")
    if manifest["n_pinned"] < manifest["n_images"]:
        line += (", and the rest are assigned by the split builder's seeded "
                 "shuffle")
    log(line)


def _check_names_cover(classes, names):
    """Refuse a names table that does not name every class token.

    Parameters:
      classes: class tokens in class-id order.
      names: [(token, full name)], or an empty list for no names table.
    Returns:
      None.
    Raises:
      IngestError naming the first tokens with no full name.
    """
    if not names:
        return
    known = {token for token, _full in names}
    missing = [token for token in classes if token not in known]
    if missing:
        raise IngestError(
            "these class tokens have no full name in the names table: "
            + ", ".join(missing[:REFUSAL_EXAMPLES])
            + (f" (and {len(missing) - REFUSAL_EXAMPLES} more)"
               if len(missing) > REFUSAL_EXAMPLES else "")
            + ". Add a row for each, or stage without --names")


def _build_manifest(*, name, dest, records, classes, counts, pins, renamed,
                    n_case_disambiguated, limit, citation, license_text, names,
                    names_csv, images_dir, labels_dir, subject):
    """Assemble ingest_manifest.json.

    Parameters:
      name: the staged dataset folder name.
      dest: the staged dataset folder.
      records: the Record list that was staged.
      classes: class tokens in class-id order.
      counts: the dict _scan_labels returned.
      pins: {staged basename: split}, empty when the dataset has no pins.
      renamed: [(original, staged)] for every normalised basename.
      n_case_disambiguated: how many of those names took a numbered suffix
        because an earlier source file had already claimed the stem ignoring
        letter case.
      limit: the record cap this run used, 0 for every record.
      citation: the citation text, or "" for none recorded.
      license_text: the license text as written to LICENSE.md.
      names: [(token, full name)], or an empty list.
      names_csv: where the names rows were read from, or "".
      images_dir, labels_dir: the source folders.
      subject: the checked subject, or "" when the dataset records none.
    Returns:
      the manifest dict. The Coral Intel keys come first so both staged
      datasets read the same way; the eleven keys after them are what a
      generic staged dataset adds.
    """
    by_split = {split: sum(1 for value in pins.values() if value == split)
                for split in SPLIT_DIRS}
    return {
        "dataset": name,
        "n_images": len(records),
        "n_boxes": counts["n_boxes"],
        "n_degenerate": counts["n_degenerate"],
        "n_degenerate_exact": counts["n_degenerate_exact"],
        "n_classes": len(classes),
        "classes": list(classes),
        "source_path": os.path.abspath(os.path.expanduser(str(images_dir))),
        "ingested_at": ast_now(),
        "limit": int(limit) or None,
        "citation": citation.strip() or NO_CITATION,
        "license_note": license_text,
        "coverage_note": COVERAGE_NOTE,
        "source_kind": SOURCE_KIND,
        "labels_source": os.path.abspath(os.path.expanduser(str(labels_dir))),
        "names_csv": (os.path.abspath(os.path.expanduser(str(names_csv)))
                      if names and names_csv else ""),
        "split_pins": bool(pins),
        "n_pinned": len(pins),
        "pins_by_split": by_split,
        "n_renamed": len(renamed),
        "n_case_disambiguated": n_case_disambiguated,
        "n_background": counts["n_background"],
        "n_duplicate_lines": counts["n_duplicate_lines"],
        # What the dataset is about, for the Train panel's Subject control.
        # "" for a dataset staged without --subject.
        "subject": subject,
    }


# ── CLI ──────────────────────────────────────────────────────────────────

def build_parser():
    """The command-line parser, split out so the flags can be read in one place.

    Returns:
      an argparse.ArgumentParser.
    """
    ap = argparse.ArgumentParser(
        description="Stage a YOLO-format bounding-box dataset into the module.")
    ap.add_argument("--images", required=True,
                    help="folder holding the source images")
    ap.add_argument("--labels", required=True,
                    help="folder holding one YOLO 'class cx cy w h' text file "
                         "per labelled image")
    ap.add_argument("--name", required=True,
                    help="the staged dataset folder name, 1 to 60 characters "
                         "from A-Z a-z 0-9 underscore and hyphen")
    ap.add_argument("--names", default="",
                    help="a token,name CSV giving the full name behind every "
                         "class token; staged as names.csv")
    ap.add_argument("--data-yaml", default="",
                    help="the dataset's data.yaml, read for the class list; "
                         "without it the class list is the names CSV order")
    ap.add_argument("--split-from", default="",
                    help="a folder holding train/, valid/ and test/, each with "
                         "an images/ folder, pinned into split_pins.json")
    ap.add_argument("--citation", default="",
                    help="how this dataset must be cited; recorded as "
                         f"{NO_CITATION!r} when absent")
    ap.add_argument("--license", dest="license_note", default="",
                    help="the license terms; a standing note saying none is "
                         "recorded is written when absent")
    ap.add_argument("--dest-root", default=DEFAULT_DEST_ROOT,
                    help="where staged datasets live (default "
                         "supporting_data/bbox_datasets in this clone)")
    ap.add_argument("--subject", default="",
                    help="what this dataset is about: "
                         f"{', '.join(DATASET_SUBJECTS)}. Recorded in the "
                         "manifest, where the Train panel reads it to start "
                         "its Subject control on the right value; omit it to "
                         "record none")
    ap.add_argument("--limit", type=int, default=0,
                    help="stage only the first N records in sorted name order "
                         "(0 stages every record)")
    ap.add_argument("--quiet", action="store_true", help="print nothing")
    return ap


def main(argv=None):
    """Parse the arguments, validate them, stage the dataset, report the counts.

    Parameters:
      argv: the argument list, or None to read sys.argv.
    Returns:
      the manifest dict ingest_yolo wrote.
    Raises:
      IngestError naming what failed, where, and why.
    """
    args = build_parser().parse_args(argv)
    log = (lambda _m: None) if args.quiet else print
    check_dataset_name(args.name)
    check_subject(args.subject)
    names = read_names_csv(args.names) if args.names else []
    if args.data_yaml:
        classes = read_class_list(args.data_yaml)
    elif names:
        classes = [token for token, _full in names]
    else:
        raise IngestError(
            "give --data-yaml (the dataset's class list) or --names (a "
            "token,name table): the class list cannot be guessed from the "
            "label files, because a class id that appears nowhere would be lost")
    dest = os.path.join(os.path.abspath(os.path.expanduser(args.dest_root)),
                        args.name)
    return ingest_yolo(args.images, args.labels, dest, classes=classes,
                       names=names, names_csv=args.names,
                       split_root=args.split_from or None,
                       citation=args.citation, license_note=args.license_note,
                       limit=args.limit, subject=args.subject, log=log)


if __name__ == "__main__":
    try:
        main()
    except IngestError as exc:
        print(f"ingest stopped: {exc}", file=sys.stderr)
        sys.exit(2)

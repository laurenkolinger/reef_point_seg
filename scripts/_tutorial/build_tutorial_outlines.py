#!/usr/bin/env python
"""Build the small training set of outlines the VEGA guided tutorial trains on.

Module:  scripts/_tutorial/build_tutorial_outlines.py
Purpose: Write a NEW training set, in the shape the coverage matrix writes one,
         holding a few dozen reviewed frames of each species plus a matching
         share of frames that carry no outline, so a student can run a whole
         train, evaluate and infer cycle on the segmentation track in about a
         minute instead of an afternoon. The result is a training set in every
         respect: the Train step's source picker lists it and the split builder
         reads its all_images/, all_labels/ and data.yaml exactly as it reads
         the set this one came from.
Inputs:  --parent (a training set folder holding all_images/, all_labels/ and
         data.yaml), --name (the new folder name), --dest-root (where training
         sets live, the module's inprocess/_composite/). Optional:
         --species-quota, --negative-quota, --seed, --quiet.
Outputs: <dest-root>/<name>/ holding all_images/ (symlinks to the survey
         frames), all_labels/ (one YOLO polygon file per frame), data.yaml,
         class_map.json, composite_manifest.csv, composite.json,
         split_pins.json, README.md and eml.xml. The function returns the
         composite.json document it wrote.

Why this tool exists beside the coverage matrix's own builder, which is not
edited here. That builder assembles every reviewed mask the matrix can reach
and gives the result whatever size the survey has: the two sets on disk hold
809 and 816 frames. A tutorial set needs three things the builder cannot
express: a cap on how many frames of each species are taken (so the run
finishes while the student is still watching), a deliberate share of frames
that carry no outline (so the student meets an empty frame as training data
rather than as a mistake), and a split written down in the set itself, so two
students are taught on the same held-out frames.

How a frame is filed. Every frame goes to exactly one bucket: the species that
owns most of its outlines, or the negatives bucket when it carries none. That
is the same notion the split builder's own --min_samples uses (it counts train
frames whose label MODE is a class), so a quota met here is a quota that
survives the split.

The split is pinned by the survey's own transect rule, not drawn. VEGA holds
transect 5 out for validation and transect 6 for test at every site, and files
transects 1 to 4 as train
(scripts/TCRMPtrain_oceankindCV/src/pinned_split.py::assign_fresh_transect,
imported here rather than copied). Which frames enter this set therefore fixes
the split completely, and split_pins.json writes that split down so a reader
can check it without running the splitter. The tests run the real splitter over
the built set and compare its pins to that file.

Class ids are the parent's, unchanged. This tool keeps every class the parent
declares, so no label line is rewritten and a prediction from this set means
the same thing as a prediction from the parent. A set holding only some of the
parent's classes would need renumbering, which is the box staging tool's job,
not this one's.

A frame whose image link does not resolve to a file is passed over and counted.
The parent sets on disk carry such links (the survey frame was moved or was
never copied to this box), the split builder skips a missing image silently,
and a pinned split whose counts do not match what is on disk would teach the
student a number that is not true.

Nothing is written until every check has passed. Each of these stops the run
with a message naming what failed, where, and why: a parent that is not a
training set, a parent whose data.yaml declares no class, a quota that is not a
positive whole number, a bucket with fewer usable frames than its quota asks
for, a train quota below the split builder's own minimum for a class, a name
outside the platform's plain-name rule, a destination whose holding folder is
not there, a destination that already exists, a label line that is not a
polygon, a class id the parent does not declare, a coordinate outside 0 to 1,
and a label file with no frame.

This is TCRMP survey imagery and it stays inside the survey's own stores: the
set is written under inprocess/_composite/ where every training set lives, and
the training projects made from it are skipped by the coverage matrix and by
STATS the way every training project is
(docs/adr/0001-non-tcrmp-imagery-never-enters-the-stores.md).

Usage:
  env/bin/python scripts/_tutorial/build_tutorial_outlines.py
"""

import argparse
import csv
import importlib.util
import json
import os
import random
import shutil
import sys
import textwrap
from collections import Counter
from datetime import datetime, timedelta, timezone
from statistics import mode as _mode

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SCRIPTS)

# The staged-dataset name rule, the refusal type and the image extension list
# are the box stager's, imported rather than copied so one rule covers both
# tutorial fixtures. Nothing in scripts/_boxes/ is edited here.
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from _boxes import ingest_yolo_boxes as ing  # noqa: E402
from _reefreview.eml_writer import write_eml  # noqa: E402

# Atlantic Standard Time, fixed at UTC-4, no daylight saving, the way every
# VICARIUS surface writes a timestamp.
AST = timezone(timedelta(hours=-4))


def _load_pinned_split():
    """The split builder module, loaded from its own folder by path.

    scripts/TCRMPtrain_oceankindCV/src/ is not an importable package, so the
    module is loaded by file path the way scripts/_matrix/composite.py loads
    its own sibling helpers. Importing it (rather than copying its transect
    rule) means this fixture's pins follow the splitter if that rule ever
    changes.

    Returns:
      the loaded module.
    Raises:
      TutorialSetError naming the file when it cannot be loaded.
    """
    path = os.path.join(_SCRIPTS, "TCRMPtrain_oceankindCV", "src", "pinned_split.py")
    try:
        spec = importlib.util.spec_from_file_location("_tutorial_pinned_split", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except (OSError, ImportError, AttributeError, ValueError) as exc:
        raise TutorialSetError(
            f"the split builder at {path} could not be loaded ({exc}); this tool "
            "reads its transect rule so the pins it writes are the pins the "
            "splitter will make")


# ── What this tool writes and what it defaults to ────────────────────────

# The training set this tool writes by default, and the set it reads by
# default. The parent is the two-species set on the coverage matrix: a set with
# one species gives the evaluation report one row, and the tutorial's
# evaluation lesson is about reading a row per species.
DEFAULT_NAME = "tutorial_outlines"
DEFAULT_PARENT_NAME = "qs20260902_pa_ss"
COMPOSITE_DIRNAME = "_composite"
DEFAULT_DEST_ROOT = os.path.join(os.path.dirname(_REPO), "inprocess", COMPOSITE_DIRNAME)
DEFAULT_PARENT = os.path.join(DEFAULT_DEST_ROOT, DEFAULT_PARENT_NAME)

# How many frames of each species the set holds, per split, and how many
# frames carrying no outline. The species numbers are set by the scarcer
# species in the parent: of the frames whose image link still resolves,
# Porites astreoides owns most of the outlines in 38 train, 6 validation and 8
# test frames, so a quota above those cannot be filled. The negatives quota is
# twice the species quota, which puts frames with no outline at about half the
# set, close to the share the parent carries (53 per cent), so the student
# meets an empty reef frame as ordinary training data.
SPECIES_QUOTAS = {"train": 30, "valid": 5, "test": 6}
NEGATIVE_QUOTAS = {"train": 60, "valid": 10, "test": 12}

# The bucket every frame with no outline goes to. Not a class id, so it can
# never collide with one.
NEGATIVE_BUCKET = "negatives"

# What the per-bucket shuffle is seeded on. Fixed, and recorded in
# composite.json and split_pins.json, so this tool run twice writes the same
# set with the same held-out frames: a tutorial whose holdout moved between two
# students would make their scores incomparable.
DEFAULT_SEED = "vega-tutorial-outlines"

# The smallest number of train frames the split builder accepts for a class
# before it drops that class from training (its --min_samples default, in
# scripts/TCRMPtrain_oceankindCV/src/pinned_split.py). A species quota below
# this would build a set whose own splitter silently trains on one species.
# tests/test_tutorial_projects.py asserts this equals the splitter's default.
PINNED_SPLIT_MIN_SAMPLES = 10

# The files a training set owes a reader, in the shape the coverage matrix
# writes them.
IMAGES_DIR = "all_images"
LABELS_DIR = "all_labels"
LABEL_EXT = ".txt"
DATA_YAML_NAME = "data.yaml"
CLASS_MAP_NAME = "class_map.json"
MANIFEST_CSV_NAME = "composite_manifest.csv"
COMPOSITE_JSON_NAME = "composite.json"
SPLIT_PINS_NAME = "split_pins.json"
README_NAME = "README.md"

# The version of split_pins.json this tool writes, matching the box fixture's.
SPLIT_PINS_SCHEMA_VERSION = 1

# The smallest YOLO polygon line: a class id and three points.
MIN_POLYGON_FIELDS = 7

# How wide the README's paragraphs are wrapped.
README_WIDTH = 79

# How many offending names a refusal lists before it counts the rest.
REFUSAL_EXAMPLES = ing.REFUSAL_EXAMPLES

# The full species name behind every label code this set can carry, read from
# the module's own code registry so the README and this tool never carry a
# second spelling of a binomial.
MASTER_CODES_CSV = os.path.join(_REPO, "supporting_data", "master_codes.csv")


class TutorialSetError(ing.IngestError):
    """Anything that stops the build before it writes a file."""


def ast_now():
    """'YYYY-MM-DD HH:MM:SS' in Atlantic Standard Time, 24 hour."""
    return datetime.now(AST).strftime("%Y-%m-%d %H:%M:%S")


def full_names(path=MASTER_CODES_CSV):
    """{code: full name} from the module's species code registry.

    Parameters:
      path: the master_codes.csv to read.
    Returns:
      a dict; empty when the file is absent or unreadable, which only costs
      the README its binomials.
    """
    names = {}
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                code = str(row.get("code") or "").strip()
                name = str(row.get("name") or "").strip()
                if code and name:
                    names[code] = name
    except (OSError, UnicodeDecodeError, csv.Error):
        return {}
    return names


# ── Reading the parent set ───────────────────────────────────────────────

def read_parent_classes(parent_dir):
    """{class id: token} from the parent's data.yaml, in class-id order.

    Parameters:
      parent_dir: the training set folder.
    Returns:
      a dict of int id to token.
    Raises:
      TutorialSetError when data.yaml is absent, unreadable, or declares no
      class, because a set with no class trains on nothing.
    """
    path = os.path.join(parent_dir, DATA_YAML_NAME)
    if not os.path.isfile(path):
        raise TutorialSetError(
            f"{parent_dir} is not a training set: it has no {DATA_YAML_NAME}. "
            "Point --parent at a folder under inprocess/_composite/.")
    try:
        import yaml
        with open(path, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise TutorialSetError(f"{path} could not be read ({exc}); rebuild the "
                               "parent set from the coverage matrix")
    names = doc.get("names")
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    if not isinstance(names, dict) or not names:
        raise TutorialSetError(
            f"{path} declares no class under names:, so there is nothing to "
            "take a balanced subset of")
    out = {}
    for key, value in names.items():
        try:
            out[int(key)] = str(value).strip()
        except (TypeError, ValueError):
            raise TutorialSetError(
                f"{path} names a class under key {key!r}, which is not a class "
                "id; the parent set is not in the shape the split builder reads")
    return dict(sorted(out.items()))


def check_parent_layout(parent_dir):
    """Refuse a parent that is not a training set.

    Parameters:
      parent_dir: the folder --parent named.
    Returns:
      (image_dir, label_dir) as absolute paths.
    Raises:
      TutorialSetError naming the missing folder.
    """
    if not os.path.isdir(parent_dir):
        raise TutorialSetError(f"{parent_dir} is not a folder; --parent takes a "
                               "training set under inprocess/_composite/")
    image_dir = os.path.join(parent_dir, IMAGES_DIR)
    label_dir = os.path.join(parent_dir, LABELS_DIR)
    for folder in (image_dir, label_dir):
        if not os.path.isdir(folder):
            raise TutorialSetError(
                f"{parent_dir} is not a training set: it has no "
                f"{os.path.basename(folder)}/ folder")
    return os.path.abspath(image_dir), os.path.abspath(label_dir)


def parse_polygons(text, n_classes, where):
    """Class ids from one YOLO polygon label file, one per outline.

    Parameters:
      text: the file's contents.
      n_classes: how many classes the parent declares.
      where: the file path, quoted back in a refusal.
    Returns:
      a list of int class ids, in file order; empty for a frame with no
      outline.
    Raises:
      TutorialSetError naming the file and the line when a line is not a
      polygon, names a class the parent does not declare, or carries a
      coordinate outside 0 to 1.
    """
    ids = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < MIN_POLYGON_FIELDS or (len(parts) - 1) % 2:
            raise TutorialSetError(
                f"{where} line {lineno} is not a polygon: a YOLO outline is a "
                f"class id and at least three x y pairs, and this line has "
                f"{len(parts) - 1} coordinate(s)")
        try:
            class_id = int(parts[0])
        except ValueError:
            raise TutorialSetError(
                f"{where} line {lineno} starts with {parts[0]!r}, which is not "
                "a class id")
        if not 0 <= class_id < n_classes:
            raise TutorialSetError(
                f"{where} line {lineno} names class {class_id}, and the parent "
                f"declares {n_classes} class(es) (0 to {n_classes - 1})")
        for field in parts[1:]:
            try:
                value = float(field)
            except ValueError:
                raise TutorialSetError(
                    f"{where} line {lineno} carries {field!r}, which is not a "
                    "coordinate")
            if not 0.0 <= value <= 1.0:
                raise TutorialSetError(
                    f"{where} line {lineno} carries the coordinate {value}, and "
                    "a YOLO coordinate is a fraction of the frame between 0 and 1")
        ids.append(class_id)
    return ids


def bucket_of(class_ids):
    """The bucket a frame belongs to: the class owning most of its outlines,
    or NEGATIVE_BUCKET when it carries none.

    The mode is taken the way the split builder's --min_samples takes it
    (statistics.mode over the label file's class ids), so a quota met here is
    a quota that survives the split. A tie goes to the class that appears
    first in the file, which is what statistics.mode does.

    Parameters:
      class_ids: the class ids of the frame's outlines, in file order.
    Returns:
      an int class id, or NEGATIVE_BUCKET.
    """
    return _mode(class_ids) if class_ids else NEGATIVE_BUCKET


def split_of(basename, pinned_split):
    """The split the survey's transect rule gives one frame.

    Parameters:
      basename: the image file name.
      pinned_split: the loaded split builder module.
    Returns:
      'train', 'valid' or 'test'.
    """
    return pinned_split.assign_fresh_transect([basename], {5}, {6})[basename]


def scan_parent(image_dir, label_dir, n_classes, pinned_split, log):
    """File every frame of the parent set by bucket and split.

    Reads each label file once and checks the image link resolves, so the set
    this tool builds holds only frames a training run will actually find.

    Parameters:
      image_dir: the parent's all_images/.
      label_dir: the parent's all_labels/.
      n_classes: how many classes the parent declares.
      pinned_split: the loaded split builder module.
      log: where the counts go.
    Returns:
      (candidates, skipped) where candidates is {(bucket, split): [basename]}
      sorted by basename, and skipped is a list of the basenames whose image
      link does not resolve.
    Raises:
      TutorialSetError when a label file has no frame or a label line is not a
      polygon.
    """
    images = sorted(name for name in os.listdir(image_dir)
                    if os.path.splitext(name)[1].lower() in ing.IMAGE_EXTS)
    stems = {os.path.splitext(name)[0] for name in images}
    orphans = sorted(name for name in os.listdir(label_dir)
                     if name.endswith(LABEL_EXT) and name[:-len(LABEL_EXT)] not in stems)
    if orphans:
        raise TutorialSetError(
            f"{label_dir} holds {len(orphans)} label file(s) with no frame in "
            f"{image_dir}, for example {', '.join(orphans[:REFUSAL_EXAMPLES])}; "
            "rebuild the parent set from the coverage matrix")

    candidates, skipped = {}, []
    for name in images:
        if not os.path.exists(os.path.join(image_dir, name)):
            skipped.append(name)
            continue
        label_path = os.path.join(label_dir, os.path.splitext(name)[0] + LABEL_EXT)
        text = ""
        if os.path.isfile(label_path):
            try:
                with open(label_path, encoding="utf-8") as fh:
                    text = fh.read()
            except (OSError, UnicodeDecodeError) as exc:
                raise TutorialSetError(f"{label_path} could not be read ({exc})")
        key = (bucket_of(parse_polygons(text, n_classes, label_path)),
               split_of(name, pinned_split))
        candidates.setdefault(key, []).append(name)

    log(f"read {len(images)} frame(s) of {image_dir}: {len(images) - len(skipped)} "
        f"usable, {len(skipped)} whose image link does not resolve")
    log("Step 1 complete: every parent frame is filed by species and split")
    return candidates, skipped


# ── Choosing the subset ──────────────────────────────────────────────────

def check_quotas(quotas, flag):
    """Refuse a quota that is not a positive whole number.

    A bool is refused too: True would silently read as 1.

    Parameters:
      quotas: {'train': n, 'valid': n, 'test': n}.
      flag: the command-line flag, quoted back in the refusal.
    Returns:
      the same quotas, with int values.
    Raises:
      TutorialSetError naming the flag and the split.
    """
    out = {}
    for split in ("train", "valid", "test"):
        value = quotas.get(split)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TutorialSetError(
                f"{flag} {split} must be a whole number of frames above zero, "
                f"got {value!r}; a split with no frame leaves the evaluation "
                "report without a row to read")
        out[split] = int(value)
    return out


def check_train_quota_trains(species_quota, buckets):
    """Refuse a train quota the split builder would then drop a class for.

    Parameters:
      species_quota: how many train frames each species bucket takes.
      buckets: the class tokens, for the refusal's wording.
    Raises:
      TutorialSetError naming the minimum and where it comes from.
    """
    if species_quota < PINNED_SPLIT_MIN_SAMPLES:
        raise TutorialSetError(
            f"--species-quota asks for {species_quota} train frame(s) of each of "
            f"{', '.join(buckets)}, and the split builder drops a class with "
            f"fewer than {PINNED_SPLIT_MIN_SAMPLES} train frames "
            "(pinned_split.py --min_samples); the set would train on fewer "
            "species than it holds")


def check_enough(candidates, class_ids, species_quotas, negative_quotas, names, where):
    """Refuse a parent that cannot fill every bucket in every split.

    Parameters:
      candidates: {(bucket, split): [basename]} from scan_parent.
      class_ids: the parent's class ids.
      species_quotas: per-split quota for each species bucket.
      negative_quotas: per-split quota for the negatives bucket.
      names: {class id: token}, for the refusal's wording.
      where: the parent folder, quoted back in the refusal.
    Raises:
      TutorialSetError listing every bucket and split that falls short.
    """
    short = []
    for bucket in list(class_ids) + [NEGATIVE_BUCKET]:
        quotas = negative_quotas if bucket == NEGATIVE_BUCKET else species_quotas
        label = NEGATIVE_BUCKET if bucket == NEGATIVE_BUCKET else names[bucket]
        for split, wanted in quotas.items():
            have = len(candidates.get((bucket, split), []))
            if have < wanted:
                short.append(f"{label} {split}: {have} usable frame(s), "
                             f"{wanted} asked for")
    if short:
        raise TutorialSetError(
            f"{where} cannot fill the tutorial set: " + "; ".join(short) +
            ". Lower the quota, or assemble a larger parent set on the coverage "
            "matrix first.")


def select_and_pin(candidates, class_ids, species_quotas, negative_quotas, seed):
    """Choose the frames and write down the split of each.

    Each bucket and split is shuffled on its own seed, derived from ``seed``
    and the bucket and split names, so changing one quota never reshuffles
    another bucket and a rebuild reproduces this set exactly.

    Parameters:
      candidates: {(bucket, split): [basename]} from scan_parent.
      class_ids: the parent's class ids.
      species_quotas: per-split quota for each species bucket.
      negative_quotas: per-split quota for the negatives bucket.
      seed: what the shuffles are seeded on.
    Returns:
      (pins, by_bucket_split) where pins is {basename: split} and
      by_bucket_split is {(bucket, split): count}.
    """
    pins, counts = {}, {}
    for bucket in list(class_ids) + [NEGATIVE_BUCKET]:
        quotas = negative_quotas if bucket == NEGATIVE_BUCKET else species_quotas
        for split, wanted in sorted(quotas.items()):
            pool = list(candidates.get((bucket, split), []))
            random.Random(f"{seed}|{bucket}|{split}").shuffle(pool)
            for name in sorted(pool[:wanted]):
                pins[name] = split
            counts[(bucket, split)] = wanted
    return pins, counts


# ── Writing the set ──────────────────────────────────────────────────────

def make_dest(dest):
    """Create the destination folder, refusing one that is already there.

    Parameters:
      dest: the folder to write.
    Raises:
      TutorialSetError when the holding folder is absent or the destination
      exists. A fixture is never written over: a student's project points at
      this folder, and replacing it under them would change what they trained
      on without a word.
    """
    holder = os.path.dirname(os.path.abspath(dest))
    if not os.path.isdir(holder):
        raise TutorialSetError(
            f"{holder} is not there, so {dest} cannot be written; --dest-root "
            "takes the module's inprocess/_composite/ folder")
    if os.path.exists(dest):
        raise TutorialSetError(
            f"{dest} already exists. This set is a fixture and is never written "
            "over; remove that folder by hand first if it really is to be rebuilt.")
    os.makedirs(os.path.join(dest, IMAGES_DIR))
    os.makedirs(os.path.join(dest, LABELS_DIR))


def link_target(image_dir, name):
    """Where one of the parent's frame links points.

    Parameters:
      image_dir: the parent's all_images/.
      name: the image file name.
    Returns:
      an absolute path to the survey frame. A parent entry that is a real file
      rather than a link yields that file's own path, so a parent written
      either way is copied the same.
    """
    path = os.path.join(image_dir, name)
    if os.path.islink(path):
        target = os.readlink(path)
        if not os.path.isabs(target):
            target = os.path.join(image_dir, target)
        return os.path.abspath(target)
    return os.path.abspath(path)


def write_pairs(image_dir, label_dir, dest, pins, log):
    """Write one frame link and one label file for every pinned frame.

    The frame is a symlink to the survey image, absolute, the way the coverage
    matrix writes one, so no imagery is duplicated. The label file is copied,
    because it is small and because a copy makes the set readable when the
    parent is removed.

    Parameters:
      image_dir: the parent's all_images/.
      label_dir: the parent's all_labels/.
      dest: the folder being written.
      pins: {basename: split}.
      log: where the counts go.
    Returns:
      the number of pairs written.
    Raises:
      TutorialSetError naming the file that could not be written.
    """
    written = 0
    for name in sorted(pins):
        stem = os.path.splitext(name)[0]
        try:
            os.symlink(link_target(image_dir, name),
                       os.path.join(dest, IMAGES_DIR, name))
        except OSError as exc:
            raise TutorialSetError(f"the frame link for {name} could not be "
                                   f"written under {dest} ({exc})")
        source_label = os.path.join(label_dir, stem + LABEL_EXT)
        target_label = os.path.join(dest, LABELS_DIR, stem + LABEL_EXT)
        try:
            if os.path.isfile(source_label):
                shutil.copyfile(source_label, target_label)
            else:
                # A frame the matrix exported with no outline carries no label
                # file in some parents. An empty file is what YOLO reads as a
                # negative, so one is written rather than left out.
                with open(target_label, "w", encoding="utf-8"):
                    pass
        except OSError as exc:
            raise TutorialSetError(f"the label file for {name} could not be "
                                   f"written under {dest} ({exc})")
        written += 1
    log(f"wrote {written} frame link(s) and {written} label file(s) under {dest}")
    log("Step 2 complete: every chosen frame is on disk with its outlines")
    return written


def write_data_yaml(dest, names):
    """Write data.yaml in the shape the coverage matrix writes it.

    Parameters:
      dest: the folder being written.
      names: {class id: token}.
    """
    lines = [f"path: {os.path.abspath(dest)}", f"train: {IMAGES_DIR}",
             f"val: {IMAGES_DIR}", f"nc: {len(names)}", "names:"]
    for class_id, token in sorted(names.items()):
        lines.append(f"  {class_id}: {token}")
    _write(os.path.join(dest, DATA_YAML_NAME), "\n".join(lines) + "\n")


def write_class_map(dest, names):
    """Write class_map.json, token to class id, the matrix's own shape.

    Parameters:
      dest: the folder being written.
      names: {class id: token}.
    """
    _write(os.path.join(dest, CLASS_MAP_NAME),
           json.dumps({token: class_id for class_id, token in sorted(names.items())},
                      indent=2) + "\n")


def read_manifest_rows(parent_dir, pins):
    """The parent's manifest rows for the chosen frames.

    composite_manifest.csv names, for every exported outline, the project it
    was annotated in, its uid and its reviewer. Carrying the rows keeps that
    provenance with the subset instead of asking a reader to go back to the
    parent.

    Read before the destination folder is made, so a parent whose manifest is
    corrupt stops the run with nothing half written on disk.

    Parameters:
      parent_dir: the parent set.
      pins: {basename: split}.
    Returns:
      (fields, rows): the column names and the matching rows. ([], []) when
      the parent carries no manifest, which is legitimate for a set assembled
      before the manifest existed.
    Raises:
      TutorialSetError naming the file when it cannot be read.
    """
    source = os.path.join(parent_dir, MANIFEST_CSV_NAME)
    if not os.path.isfile(source):
        return [], []
    stems = {os.path.splitext(name)[0] for name in pins}
    try:
        with open(source, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fields = list(reader.fieldnames or [])
            rows = [row for row in reader if str(row.get("basename") or "") in stems]
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise TutorialSetError(f"{source} could not be read ({exc}); rebuild the "
                               "parent set from the coverage matrix")
    return fields, rows


def write_manifest_rows(dest, fields, rows):
    """Write the carried manifest rows beside the set.

    Parameters:
      dest: the folder being written.
      fields: the column names.
      rows: the rows to write.
    Raises:
      TutorialSetError naming the file when the write fails.
    """
    if not fields:
        return
    try:
        with open(os.path.join(dest, MANIFEST_CSV_NAME), "w", newline="",
                  encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError) as exc:
        raise TutorialSetError(
            f"{os.path.join(dest, MANIFEST_CSV_NAME)} could not be written ({exc})")


def site_of(basename, composite_site_of):
    """The survey site code in one frame name, or 'unknown'."""
    return composite_site_of(os.path.splitext(basename)[0])


def count_set(dest, names, pins):
    """Count what is on disk: outlines per class, frames per class per split.

    Parameters:
      dest: the folder being written or checked.
      names: {class id: token}.
      pins: {basename: split}.
    Returns:
      {'n_masks', 'by_label', 'by_label_split', 'negatives'} where by_label
      counts outlines, by_label_split counts frames, and negatives lists the
      stems with no outline.
    """
    by_label, by_label_split, negatives, n_masks = Counter(), Counter(), [], 0
    for name in sorted(pins):
        stem = os.path.splitext(name)[0]
        path = os.path.join(dest, LABELS_DIR, stem + LABEL_EXT)
        with open(path, encoding="utf-8") as fh:
            ids = parse_polygons(fh.read(), len(names), path)
        n_masks += len(ids)
        if not ids:
            negatives.append(stem)
            continue
        for class_id in ids:
            by_label[names[class_id]] += 1
        for class_id in set(ids):
            by_label_split[(names[class_id], pins[name])] += 1
    return {"n_masks": n_masks, "by_label": dict(by_label),
            "by_label_split": by_label_split, "negatives": negatives}


def write_split_pins(dest, pins, parent_dir, seed):
    """Write split_pins.json: the split of every frame, keyed by frame name.

    Parameters:
      dest: the folder being written.
      pins: {basename: split}.
      parent_dir: the parent set, named in the file's source line.
      seed: what the shuffles were seeded on.
    """
    doc = {
        "schema_version": SPLIT_PINS_SCHEMA_VERSION,
        "source": (f"{os.path.abspath(parent_dir)} (pinned by "
                   f"{os.path.basename(__file__)}, seed {seed}, by the survey's "
                   "transect rule: transect 5 validates, transect 6 tests, "
                   "transects 1 to 4 train)"),
        "pins": {name: pins[name] for name in sorted(pins)},
    }
    _write(os.path.join(dest, SPLIT_PINS_NAME), json.dumps(doc, indent=2) + "\n")


def build_composite_json(*, name, parent_dir, parent_name, names, pins, counts,
                         manifest_rows, skipped, seed, species_quotas,
                         negative_quotas, composite_site_of):
    """The composite.json document, in the shape the coverage matrix writes it
    plus a tutorial block saying how this one was made.

    Parameters:
      name: the set's folder name.
      parent_dir: the parent set's path.
      parent_name: the parent set's folder name.
      names: {class id: token}.
      pins: {basename: split}.
      counts: the dict count_set returned.
      manifest_rows: the carried manifest rows.
      skipped: parent frames whose image link does not resolve.
      seed: what the shuffles were seeded on.
      species_quotas: the per-split species quota.
      negative_quotas: the per-split negatives quota.
      composite_site_of: the matrix's site parser.
    Returns:
      the document, ready to write.
    """
    by_project = Counter(str(row.get("project_id") or "") for row in manifest_rows)
    by_site = Counter(site_of(name_, composite_site_of) for name_ in pins)
    by_split = Counter(pins.values())
    return {
        "name": name,
        "created_at": ast_now(),
        "by": "LO",
        "labels": [names[class_id] for class_id in sorted(names)],
        "n_images": len(pins),
        "n_masks": counts["n_masks"],
        "n_negatives": len(counts["negatives"]),
        "negatives": sorted(counts["negatives"]),
        "by_label": counts["by_label"],
        "by_site": dict(sorted(by_site.items())),
        "by_split": {split: by_split.get(split, 0)
                     for split in ("train", "valid", "test")},
        "source_projects": [{"project_id": pid, "n_masks": by_project[pid]}
                            for pid in sorted(by_project,
                                              key=lambda p: (-by_project[p], p))],
        "projects": sorted(pid for pid in by_project if pid),
        "linked_projects": [],
        "tutorial": {
            "tool": os.path.join("scripts", "_tutorial", os.path.basename(__file__)),
            "parent": parent_name,
            "parent_path": os.path.abspath(parent_dir),
            "seed": seed,
            "species_quotas": dict(species_quotas),
            "negative_quotas": dict(negative_quotas),
            "split_policy": "transect (5 validates, 6 tests, 1 to 4 train)",
            "parent_frames_skipped_unresolved_link": len(skipped),
        },
    }


def _write(path, text):
    """Write one text file, naming it in the error when the write fails."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    except OSError as exc:
        raise TutorialSetError(f"{path} could not be written ({exc})")


def _fill(text):
    """Wrap one paragraph to README_WIDTH."""
    return textwrap.fill(text, width=README_WIDTH)


def render_readme(meta, names, dest):
    """The set's README: what it holds, how it was chosen, what it is for.

    Parameters:
      meta: the composite.json document.
      names: {class id: token}.
      dest: the folder being written.
    Returns:
      the README text.
    """
    binomials = full_names()
    tutorial = meta["tutorial"]
    species_rows = []
    for class_id in sorted(names):
        token = names[class_id]
        full = binomials.get(token, "")
        shown = f"*{full}* ({token})" if full else token
        species_rows.append(f"| {shown} | {meta['by_label'].get(token, 0)} |")
    species_list = " and ".join(
        f"*{binomials[names[c]]}* ({names[c]})" if names[c] in binomials else names[c]
        for c in sorted(names))

    lines = [
        f"# {meta['name']}",
        "",
        "The small training set of outlines the VEGA guided tutorial trains on.",
        "",
        f"- Made: {meta['created_at']} AST by {meta['by']}",
        f"- Tool: `{tutorial['tool']}`",
        f"- Parent set: `{tutorial['parent']}`, at `{tutorial['parent_path']}`",
        f"- Shuffle seed: `{tutorial['seed']}`",
        "",
        "## Read only",
        "",
        _fill("This folder is a fixture. Nobody annotates into it and nobody "
              "trains into it. A student's tutorial project points at it and "
              "writes everything it produces inside that project, so the next "
              "student starts from exactly the same frames, the same outlines "
              "and the same held-out split. Rebuild it only with the tool named "
              "above, over a folder that is not there yet."),
        "",
        "## What this set is",
        "",
        _fill(f"A training set in the layout the coverage matrix writes: "
              f"{IMAGES_DIR}/, {LABELS_DIR}/, {DATA_YAML_NAME}, "
              f"{CLASS_MAP_NAME}, {MANIFEST_CSV_NAME} and "
              f"{COMPOSITE_JSON_NAME}, plus {SPLIT_PINS_NAME} which this tool "
              f"adds. It holds {meta['n_images']} frames and "
              f"{meta['n_masks']} outlines of {species_list}. Nothing was "
              "re-annotated and no outline moved: every outline here is an "
              "outline of the parent set, under the same class id, and every "
              "frame is the parent's frame of the same name."),
        "",
        _fill("It is this small on purpose. The tutorial teaches the cycle of "
              "training a model, scoring it on frames it never saw and running "
              "it on new ones, and a student learns that cycle no better from "
              "eight hundred frames than from a couple of hundred, while "
              "waiting several times as long for it."),
        "",
        "## The two species and the empty frames",
        "",
        "| Species | Outlines |",
        "| --- | --- |",
    ]
    lines.extend(species_rows)
    lines.extend([
        f"| frames with no outline | {meta['n_negatives']} |",
        "",
        _fill(f"Each species carries the same number of frames in each split "
              f"({tutorial['species_quotas']['train']} train, "
              f"{tutorial['species_quotas']['valid']} validation, "
              f"{tutorial['species_quotas']['test']} test), so a score is not "
              "read through a class imbalance the student has not met yet. "
              "Real survey data is not balanced, and the tutorial says so "
              "where it matters."),
        "",
        _fill("About half the set carries no outline at all. Those are frames "
              "a reviewer looked at and found no target coral in, which is "
              "what most of a survey transect looks like, and they are "
              "training data: they teach the model what an empty reef is. The "
              "parent set is 53 per cent such frames and this one is close to "
              "that on purpose."),
        "",
        _fill("A frame is filed under the species that owns most of its "
              "outlines. That is the same rule the split builder uses when it "
              "decides whether a class has enough training frames "
              "(pinned_split.py --min_samples), so a quota met here is a quota "
              "that survives the split."),
        "",
        "## The split is pinned by transect, and written down",
        "",
        _fill(f"{SPLIT_PINS_NAME} names the split of every frame: "
              f"{meta['by_split']['train']} train, "
              f"{meta['by_split']['valid']} validation and "
              f"{meta['by_split']['test']} test. VEGA holds transect 5 out for "
              "validation and transect 6 for test at every site, and files "
              "transects 1 to 4 as train, so which frames entered this set "
              "fixed the split completely. The file is written so a reader can "
              "check the split without running the splitter, and so a rebuild "
              "can be compared against it frame by frame."),
        "",
        _fill("Both species appear in all three splits, which is what gives "
              "the evaluation report a row per species for the student to "
              "read."),
        "",
        _fill("The held-out frames stay held out. A model that has seen a "
              "frame in training will find its coral again, so a score "
              "measured on those frames says how well the model remembers, not "
              "how well it works. That is the lesson the test split exists to "
              "teach, and it holds only while the split stays where it is."),
        "",
        "## Frames the parent could not supply",
        "",
        _fill(f"{tutorial['parent_frames_skipped_unresolved_link']} frame(s) of "
              "the parent set were passed over because the image link does not "
              "resolve to a file on this box. The split builder skips a missing "
              "frame without a word, so a set built around one would promise "
              "the student a count that is not on disk."),
        "",
        "## What a score measured here means",
        "",
        _fill("Two species and a couple of dozen held-out frames make a number "
              "with wide error bars. Quote a score from this set only beside "
              "another score from this same set, and never beside one measured "
              "on the parent set or on the whole survey."),
        "",
        "## Coverage",
        "",
        _fill("Inside the TCRMP survey. These are survey frames and their "
              "outlines stay owned by the projects they were annotated in, "
              "named in " + MANIFEST_CSV_NAME + ". A training project made from "
              "this set is skipped by the coverage matrix and by STATS, the way "
              "every training project is, so nothing here double counts."),
        "",
        "## What each file holds",
        "",
        f"- `{IMAGES_DIR}/`: symlinks to the survey frames, absolute, nothing copied.",
        f"- `{LABELS_DIR}/`: one YOLO polygon file per frame, class then x y pairs;",
        "  an empty file is a frame with no outline.",
        f"- `{DATA_YAML_NAME}`: the Ultralytics dataset config, names keyed by class id.",
        f"- `{CLASS_MAP_NAME}`: label code to class id.",
        f"- `{MANIFEST_CSV_NAME}`: one row per outline with the project it was",
        "  annotated in, its uid, its reviewer and when.",
        f"- `{COMPOSITE_JSON_NAME}`: the counts above, machine readable, plus the",
        "  tutorial block naming the parent, the seed and the quotas.",
        f"- `{SPLIT_PINS_NAME}`: the train, valid or test split of every frame.",
        "- `eml.xml`: the ecological metadata record for this export.",
        "",
        f"Path: {os.path.abspath(dest)}",
        "",
        "Lauren Olinger, lauren.olinger@uvi.edu",
        "",
    ])
    return "\n".join(lines)


# ── The build ────────────────────────────────────────────────────────────

def build_tutorial_outlines(parent_dir, dest, *, species_quotas=None,
                            negative_quotas=None, seed=DEFAULT_SEED, log=print):
    """Build the tutorial outlines set and return its composite.json document.

    Parameters:
      parent_dir: the training set to take frames from.
      dest: the folder to write.
      species_quotas: per-split frames for each species bucket.
      negative_quotas: per-split frames with no outline.
      seed: what the per-bucket shuffles are seeded on.
      log: where progress goes; pass a no-op to silence it.
    Returns:
      the composite.json document written.
    Raises:
      TutorialSetError naming what failed, where and why, before anything is
      written whenever the failure can be seen up front.
    """
    species_quotas = check_quotas(dict(species_quotas or SPECIES_QUOTAS),
                                  "--species-quota")
    negative_quotas = check_quotas(dict(negative_quotas or NEGATIVE_QUOTAS),
                                   "--negative-quota")
    pinned_split = _load_pinned_split()
    composite_site_of = _load_site_parser()

    parent_dir = os.path.abspath(parent_dir)
    image_dir, label_dir = check_parent_layout(parent_dir)
    names = read_parent_classes(parent_dir)
    check_train_quota_trains(species_quotas["train"],
                             [names[c] for c in sorted(names)])
    candidates, skipped = scan_parent(image_dir, label_dir, len(names),
                                      pinned_split, log)
    check_enough(candidates, sorted(names), species_quotas, negative_quotas,
                 names, parent_dir)
    pins, _bucket_counts = select_and_pin(candidates, sorted(names),
                                          species_quotas, negative_quotas, seed)
    log(f"chose {len(pins)} frame(s) of {sum(len(v) for v in candidates.values())} "
        f"usable, seeded on {seed!r}")

    manifest_fields, manifest_rows = read_manifest_rows(parent_dir, pins)

    # From here on the destination folder exists, so any failure removes it
    # again: a half-written set would be offered in the Train step's source
    # picker and would block the retry that fixes it.
    make_dest(dest)
    try:
        write_pairs(image_dir, label_dir, dest, pins, log)
        write_data_yaml(dest, names)
        write_class_map(dest, names)
        write_manifest_rows(dest, manifest_fields, manifest_rows)
        counts = count_set(dest, names, pins)
        meta = build_composite_json(
            name=os.path.basename(os.path.abspath(dest)), parent_dir=parent_dir,
            parent_name=os.path.basename(parent_dir), names=names, pins=pins,
            counts=counts, manifest_rows=manifest_rows, skipped=skipped, seed=seed,
            species_quotas=species_quotas, negative_quotas=negative_quotas,
            composite_site_of=composite_site_of)
        _write(os.path.join(dest, COMPOSITE_JSON_NAME),
               json.dumps(meta, indent=2) + "\n")
        write_split_pins(dest, pins, parent_dir, seed)
        _write(os.path.join(dest, README_NAME), render_readme(meta, names, dest))
        _write_eml(dest, meta, log)
        log(f"wrote {DATA_YAML_NAME}, {CLASS_MAP_NAME}, {MANIFEST_CSV_NAME}, "
            f"{COMPOSITE_JSON_NAME}, {SPLIT_PINS_NAME} and {README_NAME}")
        log("Step 3 complete: the set's own records are written")
        verify(dest, names, species_quotas, negative_quotas, pinned_split, log)
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return meta


def _load_site_parser():
    """The coverage matrix's own site-code parser.

    Imported rather than copied so the per-site counts this tool writes are
    the counts the matrix would write.

    Returns:
      a callable taking a frame stem and returning a site code or 'unknown'.
    Raises:
      TutorialSetError when the matrix module cannot be imported.
    """
    try:
        from _matrix.composite import _site_for_stem
        return _site_for_stem
    except ImportError as exc:
        raise TutorialSetError(
            f"the coverage matrix's composite module could not be imported "
            f"({exc}); this tool reads its site parser so the per-site counts "
            "here match the matrix's")


def _write_eml(dest, meta, log):
    """Write eml.xml beside the set, the way the matrix's builder does.

    A failure is logged and not raised: the set is complete without it, and
    the matrix's own builder treats an EML failure the same way.
    """
    try:
        description = (f"reef_point_seg tutorial training set {meta['name']}: "
                       f"{meta['n_images']} frames, {meta['n_masks']} outlines, "
                       f"labels {', '.join(meta['labels']) or 'none'}.")
        write_eml(dest, meta["name"], description)
    except Exception as exc:  # noqa: BLE001 - never let metadata stop a fixture
        log(f"note: eml.xml was not written ({exc}); the set is complete without it")


# ── Verification ─────────────────────────────────────────────────────────

def verify(dest, names, species_quotas, negative_quotas, pinned_split, log):
    """Read the written folder back and check every invariant the tutorial needs.

    Read off disk rather than off the build's own bookkeeping, so this proves
    what a student's training run will find rather than what this process
    believed it wrote.

    Parameters:
      dest: the folder just written.
      names: {class id: token}.
      species_quotas: the per-split species quota.
      negative_quotas: the per-split negatives quota.
      pinned_split: the loaded split builder module.
      log: where the counts go.
    Returns:
      the counts it read.
    Raises:
      TutorialSetError naming the disagreement.
    """
    meta = _read_json(os.path.join(dest, COMPOSITE_JSON_NAME))
    pins = _read_json(os.path.join(dest, SPLIT_PINS_NAME))["pins"]
    images = sorted(name for name in os.listdir(os.path.join(dest, IMAGES_DIR))
                    if os.path.splitext(name)[1].lower() in ing.IMAGE_EXTS)
    labels = sorted(name for name in os.listdir(os.path.join(dest, LABELS_DIR))
                    if name.endswith(LABEL_EXT))

    if len(images) != len(labels):
        raise TutorialSetError(f"{dest} holds {len(images)} frame(s) and "
                               f"{len(labels)} label file(s)")
    unpaired = sorted(set(os.path.splitext(n)[0] for n in images)
                      ^ set(n[:-len(LABEL_EXT)] for n in labels))
    if unpaired:
        raise TutorialSetError(
            f"{dest} holds {len(unpaired)} frame(s) or label file(s) without a "
            f"partner, for example {', '.join(unpaired[:REFUSAL_EXAMPLES])}")
    if sorted(pins) != images:
        raise TutorialSetError(f"{SPLIT_PINS_NAME} pins {len(pins)} frame(s) and "
                               f"{dest} holds {len(images)}")
    if len(images) != meta["n_images"]:
        raise TutorialSetError(f"{dest} holds {len(images)} frame(s) and "
                               f"{COMPOSITE_JSON_NAME} says {meta['n_images']}")

    broken = [name for name in images
              if not os.path.exists(os.path.join(dest, IMAGES_DIR, name))]
    if broken:
        raise TutorialSetError(
            f"{len(broken)} frame link(s) under {dest} do not resolve to a file, "
            f"for example {', '.join(broken[:REFUSAL_EXAMPLES])}")

    # This tree is mirrored by a Synology sync that folds two names differing
    # only in letter case into one file, which would break the frame and label
    # pair the set is built on. Caught here rather than discovered as a missing
    # frame weeks later.
    folded = Counter(name.lower() for name in images)
    clashing = sorted(name for name in folded if folded[name] > 1)
    if clashing:
        raise TutorialSetError(
            f"{dest} holds {len(clashing)} frame name(s) that differ from "
            f"another only in letter case, for example "
            f"{', '.join(clashing[:REFUSAL_EXAMPLES])}; the sync on this tree "
            "folds such a pair into one file and the set would lose a frame")

    replayed = pinned_split.assign_fresh_transect(images, {5}, {6})
    moved = sorted(name for name in images if replayed[name] != pins[name])
    if moved:
        raise TutorialSetError(
            f"{SPLIT_PINS_NAME} disagrees with the transect rule for "
            f"{len(moved)} frame(s), for example "
            f"{', '.join(moved[:REFUSAL_EXAMPLES])}")

    counts = count_set(dest, names, pins)
    if counts["n_masks"] != meta["n_masks"]:
        raise TutorialSetError(f"{dest} holds {counts['n_masks']} outline(s) and "
                               f"{COMPOSITE_JSON_NAME} says {meta['n_masks']}")
    if len(counts["negatives"]) != negative_quotas["train"] + negative_quotas["valid"] \
            + negative_quotas["test"]:
        raise TutorialSetError(
            f"{dest} holds {len(counts['negatives'])} frame(s) with no outline "
            f"and the negatives quota asks for "
            f"{sum(negative_quotas.values())}")
    missing = []
    for class_id in sorted(names):
        for split in ("train", "valid", "test"):
            have = counts["by_label_split"].get((names[class_id], split), 0)
            if have < species_quotas[split]:
                missing.append(f"{names[class_id]} in {split}: {have} frame(s), "
                               f"{species_quotas[split]} asked for")
    if missing:
        raise TutorialSetError(
            f"{dest} does not carry every species in every split: " +
            "; ".join(missing) + ". The evaluation report reads one row per "
            "species per split, and a missing row is the number the tutorial "
            "asks the student to read.")

    by_split = Counter(pins.values())
    log(f"verified {len(images)} frame(s) beside {len(labels)} label file(s), "
        f"{counts['n_masks']} outline(s), train {by_split['train']}, valid "
        f"{by_split['valid']}, test {by_split['test']}")
    log("Step 4 complete: the folder on disk holds what its records claim")
    return counts


def _read_json(path):
    """One JSON document this tool just wrote.

    Raises:
      TutorialSetError naming the file when it cannot be read or parsed.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise TutorialSetError(f"{path} could not be read back: {exc}")


# ── Command line ─────────────────────────────────────────────────────────

def build_parser():
    """The command-line parser, split out so the flags read in one place."""
    parser = argparse.ArgumentParser(
        description="Build the small training set of outlines the VEGA guided "
                    "tutorial trains on.")
    parser.add_argument(
        "--parent", default=DEFAULT_PARENT,
        help=f"the training set to take frames from (default: {DEFAULT_PARENT})")
    parser.add_argument(
        "--name", default=DEFAULT_NAME,
        help=f"the folder to write (default: {DEFAULT_NAME})")
    parser.add_argument(
        "--dest-root", default=DEFAULT_DEST_ROOT,
        help=f"where training sets live (default: {DEFAULT_DEST_ROOT})")
    parser.add_argument(
        "--species-quota", nargs=3, type=int, metavar=("TRAIN", "VALID", "TEST"),
        default=[SPECIES_QUOTAS["train"], SPECIES_QUOTAS["valid"],
                 SPECIES_QUOTAS["test"]],
        help="how many frames of each species go to train, validation and test "
             f"(default: {SPECIES_QUOTAS['train']} "
             f"{SPECIES_QUOTAS['valid']} {SPECIES_QUOTAS['test']})")
    parser.add_argument(
        "--negative-quota", nargs=3, type=int, metavar=("TRAIN", "VALID", "TEST"),
        default=[NEGATIVE_QUOTAS["train"], NEGATIVE_QUOTAS["valid"],
                 NEGATIVE_QUOTAS["test"]],
        help="how many frames with no outline go to train, validation and test "
             f"(default: {NEGATIVE_QUOTAS['train']} "
             f"{NEGATIVE_QUOTAS['valid']} {NEGATIVE_QUOTAS['test']})")
    parser.add_argument(
        "--seed", default=DEFAULT_SEED,
        help=f"what the per-bucket shuffles are seeded on (default: {DEFAULT_SEED})")
    parser.add_argument("--quiet", action="store_true",
                        help="print only the refusals and the final line")
    return parser


def _quotas(triple):
    """{'train': n, 'valid': n, 'test': n} from a --*-quota triple."""
    return {"train": triple[0], "valid": triple[1], "test": triple[2]}


def main(argv=None):
    """Parse, validate, build, and report.

    Parameters:
      argv: the command line, or None for sys.argv[1:].
    Returns:
      0 when the set was written, 1 when the build was refused.
    """
    args = build_parser().parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else print
    try:
        ing.check_dataset_name(args.name)
        root = os.path.abspath(args.dest_root)
        meta = build_tutorial_outlines(
            args.parent, os.path.join(root, args.name),
            species_quotas=_quotas(args.species_quota),
            negative_quotas=_quotas(args.negative_quota),
            seed=args.seed, log=log)
    except ing.IngestError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(f"{meta['name']}: {meta['n_images']} frames, {meta['n_masks']} outlines, "
          f"{len(meta['labels'])} species, {meta['n_negatives']} frames with no "
          f"outline, pinned train {meta['by_split']['train']}, valid "
          f"{meta['by_split']['valid']}, test {meta['by_split']['test']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

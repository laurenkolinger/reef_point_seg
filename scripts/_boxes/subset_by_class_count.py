#!/usr/bin/env python
"""Write a class-count subset of a staged bounding-box dataset.

Module:  scripts/_boxes/subset_by_class_count.py
Purpose: Copy a staged box dataset into a NEW staged dataset holding only the
         classes that carry at least N boxes. A detector trained on a class
         with a handful of boxes learns that class badly and drags the whole
         run's score down with it, so an operator who wants a training set of
         the classes the data can actually support asks for one here rather
         than editing label files by hand. The copy is a staged dataset in
         every respect: the Bounding Boxes page lists it, and the training
         split builder reads it, exactly as they read the parent.
Inputs:  --parent (a staged dataset folder holding data.yaml and
         ingest_manifest.json), --name (the new staged folder name) and
         --min-boxes (how many boxes a class needs to be kept). Optional:
         --dest-root, --quiet.
Outputs: <dest-root>/<name>/ holding images/, labels/, classes.txt, data.yaml,
         class_map.csv, dropped_classes.csv, LICENSE.md, README.md and
         ingest_manifest.json, plus names.csv and split_pins.json narrowed to
         what survived when the parent carries them. The function returns the
         manifest it wrote.

Class ids are renumbered. Ultralytics refuses a label file carrying a class id
at or above the number of classes the dataset declares, so a subset that kept
the parent's ids would not train at all: the surviving classes are numbered
again from 0 in their parent order, and every kept label line is rewritten with
its new id. class_map.csv is the record that maps a prediction from this subset
back to the parent, and dropped_classes.csv names every class that did not meet
the minimum, so the omission is documented rather than silent.

The licence follows the data. The parent's LICENSE.md is copied verbatim,
because a staged dataset whose copy loses its terms is a licensing problem, not
a tidiness problem; a parent with no LICENSE.md but a license note in its
manifest gets one written from that note, and a parent with neither stops the
run.

Nothing is written until every check has passed. Each of these stops the run
with a message naming what failed, where, and why: a parent that is not a
staged dataset, a minimum that is not a positive whole number, a minimum that
would keep no class, an output name outside the platform's plain-name rule, an
output folder whose holding folder is not there, an output folder that already
exists, a label line that is not exactly five
fields, a class id outside the parent class list, a parent whose classes.txt
and data.yaml disagree, a label file with no image, a split_pins.json that
cannot be read, a parent that records no licence terms, a parent recording a
subject outside the vocabulary the Train panel reads, and two names that
differ only in letter case (the staged tree is kept inside a Synology Drive
sync whose destination cannot hold both, and renames one of them minutes after
the copy).

The subject follows the data too. A subset is its parent's images with some
classes removed, so it is about what the parent was about: the parent's
"subject" key is checked and written into the subset's own manifest, and the
Train panel starts its Subject control on it either way.

Label files are read one at a time in both passes, so a parent of any size is
subset without its labels ever being in memory together.

This imagery is outside the TCRMP survey. A subset never writes to
inprocess/run_*, the mask registry, label provenance, the coverage matrix, or
STATS: a staged dataset is read where it is staged.

Usage:
  env/bin/python scripts/_boxes/subset_by_class_count.py \
      --parent supporting_data/bbox_datasets/coral_intel \
      --name coral_intel_subset --min-boxes 100
"""

import argparse
import csv
import json
import os
import shutil
import sys
import textwrap

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)

# The two stagers own the staged layout: their class-list reader, name rule,
# label parser, data.yaml writer, whole-frame box rule and copy-when-changed
# rule are imported rather than copied, so a subset stays the same shape as a
# staged dataset as those files change.
if __package__:
    from . import ingest_coral_intel as coral
    from . import ingest_yolo_boxes as ing
else:  # run directly as a script: the package root is not on sys.path yet
    if _SCRIPTS not in sys.path:
        sys.path.insert(0, _SCRIPTS)
    from _boxes import ingest_coral_intel as coral
    from _boxes import ingest_yolo_boxes as ing

# What the manifest records about how this dataset was made. The Bounding
# Boxes page shows it beside the dataset's counts.
SOURCE_KIND = "subset"

# The staged files a parent must carry to be read as a staged dataset, and the
# ones a subset writes.
MANIFEST_NAME = "ingest_manifest.json"
DATA_YAML_NAME = "data.yaml"
CLASSES_TXT_NAME = "classes.txt"
LICENSE_NAME = "LICENSE.md"
README_NAME = "README.md"
NAMES_CSV_NAME = "names.csv"
SPLIT_PINS_NAME = "split_pins.json"
CLASS_MAP_NAME = "class_map.csv"
DROPPED_NAME = "dropped_classes.csv"
IMAGES_DIR = "images"
LABELS_DIR = "labels"

CLASS_MAP_HEADER = ("old_id", "new_id", "name", "boxes")
DROPPED_HEADER = ("old_id", "name", "boxes")

# Which parent label folder is read, in the order the training split builder
# prefers them (detect_dataset.BBOX_LABEL_DIRS). labels_localized/ is the
# Coral Intel stager's folder of boxes minus the whole-frame ones, and it is
# what training reads, so it is what a subset of that dataset must read too.
LABEL_DIR_PREFERENCE = ("labels_localized", "labels")

# The label folder a subset writes. One folder only: a subset that wrote both
# would leave the split builder reading a labels_localized/ this tool never
# filtered.
SUBSET_LABEL_DIR = LABELS_DIR

# How many records go by between progress lines on a long run.
PROGRESS_EVERY = ing.PROGRESS_EVERY

# How wide the README's paragraphs are wrapped. The counts go into the prose
# before it is wrapped, so a four-image subset and a ten-thousand-image one
# read the same way.
README_WIDTH = 79

LABEL_EXT = ing.LABEL_EXT
LABEL_FIELDS = ing.LABEL_FIELDS

# How many offending names a refusal lists before it counts the rest.
REFUSAL_EXAMPLES = ing.REFUSAL_EXAMPLES

# What the operator does about an output folder that is already there. This
# tool never writes into an existing folder: the older files would stay on disk
# and be trained on while the manifest counted only what this run wrote.
EXISTS_REMEDY = ("Choose another name, or remove that folder first: this tool "
                 "never writes into a folder that already exists.")

# Why two names that differ only in letter case cannot both be staged. The
# staged tree is kept inside a Synology Drive sync whose destination cannot
# hold both, and it renames one of them minutes after the copy, leaving an
# image with no label.
CASE_RULE = ("This tree is kept inside a Synology Drive sync whose destination "
             "cannot hold two names that differ only in letter case, so one of "
             "the two would be renamed minutes after the copy and the image "
             "and label pair it belongs to would be broken.")

DEFAULT_DEST_ROOT = ing.DEFAULT_DEST_ROOT

# The three splits a pin may name, the split builder's own folder names. A
# subset narrows the parent's pins to the images it keeps and counts them by
# split in the manifest.
SPLIT_NAMES = ing.SPLIT_DIRS

# The staged layout's three writers, held under this module's own names rather
# than copied into it. A subset writes the same text, CSV and JSON files the
# stagers write, and two copies of a writer drift apart the next time one of
# them is fixed. They raise IngestError, which SubsetError subclasses and the
# command line already catches, so a failed write reaches the operator through
# the same channel as every refusal here.
_write_text = ing._write_text
_write_csv = ing._write_csv
_write_json = ing._write_json


class SubsetError(ing.IngestError):
    """Anything that stops the subset before it writes a file.

    A subclass of the stagers' IngestError so a caller that already catches
    that one catches a refusal raised here as well, and so the refusals this
    tool borrows from the stagers (the name rule, the label parser, the
    class-list reader) reach the operator through the same channel.
    """


# ── Readers and validation ───────────────────────────────────────────────

def normalise_min_boxes(value):
    """The class-count minimum as a positive whole number.

    Parameters:
      value: a whole number, or a string holding one.
    Returns:
      an int of 1 or more.
    Raises:
      SubsetError naming the value when it is not a whole number, or is below
      1. A minimum of 0 keeps every class, which is a copy rather than a
      subset, and is refused so the operator says what they meant.

    Example:
      normalise_min_boxes("100") == 100
    """
    try:
        count = int(str(value).strip())
    except (TypeError, ValueError):
        raise SubsetError(
            f"the class-count minimum {value!r} is not a whole number: give "
            "the number of boxes a class needs to be kept, for example 100")
    if count < 1:
        raise SubsetError(
            f"the class-count minimum is {count}: a class needs at least 1 "
            "box to be kept, and a minimum of 0 would keep every class, which "
            "is a copy of the parent rather than a subset of it")
    return count


def read_manifest(parent_dir):
    """The parent dataset's ingest_manifest.json.

    Parameters:
      parent_dir: the staged parent dataset folder.
    Returns:
      the manifest as a dict.
    Raises:
      SubsetError naming the file when it is missing, unreadable, or is not a
      JSON object.
    """
    path = os.path.join(parent_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise SubsetError(
            f"{parent_dir} is not a staged bounding-box dataset: {MANIFEST_NAME} "
            "is not there. A staged dataset carries data.yaml and "
            f"{MANIFEST_NAME} beside its images/ and labels/ folders")
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise SubsetError(f"{path} could not be read: {exc}")
    if not isinstance(doc, dict):
        raise SubsetError(
            f"{path} is not a JSON object, so it records none of the parent's "
            f"counts, licence or citation, got {type(doc).__name__}")
    return doc


def parent_class_list(parent_dir):
    """The parent's class tokens in class-id order, agreed by both its lists.

    A staged dataset declares its classes twice, in data.yaml and in
    classes.txt, and the two are written together. When they disagree, which
    of them the label ids were numbered against is unknowable, and a subset
    that guessed would rename every class it kept.

    Parameters:
      parent_dir: the staged parent dataset folder.
    Returns:
      a list of class tokens, index equal to the class id.
    Raises:
      SubsetError when data.yaml is missing or unreadable (through
      ingest_yolo_boxes.read_class_list), or when classes.txt disagrees with
      it.
    """
    data_yaml = os.path.join(parent_dir, DATA_YAML_NAME)
    if not os.path.isfile(data_yaml):
        raise SubsetError(
            f"{parent_dir} is not a staged bounding-box dataset: "
            f"{DATA_YAML_NAME} is not there. A staged dataset carries "
            f"{DATA_YAML_NAME} and {MANIFEST_NAME} beside its images/ and "
            "labels/ folders")
    classes = ing.read_class_list(data_yaml)
    listed = _read_classes_txt(os.path.join(parent_dir, CLASSES_TXT_NAME))
    if listed and listed != classes:
        raise SubsetError(
            f"{parent_dir}: {CLASSES_TXT_NAME} lists {len(listed)} class(es) "
            f"and {DATA_YAML_NAME} lists {len(classes)}, and the two do not "
            "agree, so which list the label ids were numbered against cannot "
            f"be told. First difference: {_first_difference(listed, classes)}. "
            "Fix the parent's class lists, then subset again")
    return classes


def _read_classes_txt(path):
    """The class tokens in a staged classes.txt, one per line.

    Parameters:
      path: the classes.txt to read.
    Returns:
      a list of tokens with blank lines removed, or [] when the file is not
      there.
    Raises:
      SubsetError naming the file when it cannot be read.
    """
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip()]
    except (OSError, UnicodeDecodeError) as exc:
        raise SubsetError(f"{path} could not be read: {exc}")


def _first_difference(listed, declared):
    """The first class id where two class lists say different things.

    Parameters:
      listed: the classes.txt tokens.
      declared: the data.yaml tokens.
    Returns:
      a short phrase naming the id and both tokens, for reading inside a
      refusal.
    """
    for i in range(max(len(listed), len(declared))):
        left = listed[i] if i < len(listed) else "(nothing)"
        right = declared[i] if i < len(declared) else "(nothing)"
        if left != right:
            return (f"class {i} is {left!r} in {CLASSES_TXT_NAME} and {right!r} "
                    f"in {DATA_YAML_NAME}")
    return "the two lists agree line for line"


def choose_label_dir(parent_dir):
    """Which of the parent's label folders this subset reads.

    labels_localized/ comes first, because that is the folder the training
    split builder prefers, so it is the one a training run would have used.

    Parameters:
      parent_dir: the staged parent dataset folder.
    Returns:
      (relative name, absolute path, sorted list of label file names).
    Raises:
      SubsetError naming the folders it looked in when neither holds a label
      file, because a parent with no labels has nothing to subset.
    """
    for rel in LABEL_DIR_PREFERENCE:
        folder = os.path.join(parent_dir, rel)
        if not os.path.isdir(folder):
            continue
        files = label_file_names(folder)
        if files:
            return rel, folder, files
    raise SubsetError(
        f"{parent_dir} holds no label file in "
        f"{' or '.join(LABEL_DIR_PREFERENCE)}, so there is nothing to subset")


def label_file_names(folder):
    """Every label file in one folder, in sorted order.

    The sidecars the common annotation tools write beside the labels
    (ingest_yolo_boxes.LABEL_SIDECARS: classes.txt, notes.json, data.yaml) are
    passed over rather than read as a label.

    Parameters:
      folder: a staged label folder.
    Returns:
      a sorted list of label file basenames.
    Raises:
      SubsetError naming the folder when it cannot be read.
    """
    try:
        names = sorted(os.listdir(folder))
    except OSError as exc:
        raise SubsetError(f"{folder} could not be read: {exc}")
    return [name for name in names
            if name.endswith(LABEL_EXT) and name not in ing.LABEL_SIDECARS]


def image_index(parent_dir):
    """Every parent image, keyed by the stem its label file shares.

    Parameters:
      parent_dir: the staged parent dataset folder.
    Returns:
      {stem: image basename} over the parent's images/ folder.
    Raises:
      SubsetError when images/ is missing or unreadable, or when two images
      share one stem, because then the label file naming that stem describes
      two images and neither could be copied with confidence.
    """
    folder = os.path.join(parent_dir, IMAGES_DIR)
    if not os.path.isdir(folder):
        raise SubsetError(
            f"{parent_dir} is not a staged bounding-box dataset: {folder} is "
            "not there")
    try:
        names = sorted(os.listdir(folder))
    except OSError as exc:
        raise SubsetError(f"{folder} could not be read: {exc}")
    index = {}
    for name in names:
        stem, ext = os.path.splitext(name)
        if ext.lower() not in ing.IMAGE_EXTS:
            continue
        if stem in index:
            raise SubsetError(
                f"{folder} holds both {index[stem]!r} and {name!r}, two images "
                f"under the stem {stem!r}, so the one label file "
                f"{stem + LABEL_EXT!r} would describe both. Fix the parent, "
                "then subset again")
        index[stem] = name
    if not index:
        raise SubsetError(
            f"{folder} holds no image this module reads "
            f"({', '.join(ing.IMAGE_EXTS)})")
    return index


def check_case_unique(names, where):
    """Refuse two names in one folder that differ only in letter case.

    A subset inherits every basename from its parent, so this is asserted
    over the parent's own folders: when no two parent names clash, no two
    names this run writes can clash either.

    Parameters:
      names: the basenames in one folder.
      where: that folder, quoted back in the refusal.
    Returns:
      None.
    Raises:
      SubsetError naming both files and the rule.
    """
    seen = {}
    for name in sorted(names):
        key = name.lower()
        if key in seen:
            raise SubsetError(
                f"{where} holds {seen[key]!r} and {name!r}, two names that "
                f"differ only in letter case. {CASE_RULE} Rename one in the "
                "parent dataset, then subset again")
        seen[key] = name


def check_every_label_has_an_image(files, images, label_dir):
    """Refuse a label file whose image is not in the parent's images/ folder.

    Parameters:
      files: the label file basenames this run read.
      images: {stem: image basename} from image_index.
      label_dir: the label folder, quoted back in the refusal.
    Returns:
      None.
    Raises:
      SubsetError naming the first label files with no image.
    """
    missing = [name for name in files
               if name[:-len(LABEL_EXT)] not in images]
    if not missing:
        return
    raise SubsetError(
        f"{len(missing)} label file(s) in {label_dir} have no image in the "
        "parent's images/ folder, so the boxes describe a picture that is not "
        "there: " + ", ".join(missing[:REFUSAL_EXAMPLES])
        + (f" (and {len(missing) - REFUSAL_EXAMPLES} more)"
           if len(missing) > REFUSAL_EXAMPLES else ""))


def check_dest_holder(dest):
    """Refuse a subset folder whose holding folder is not there.

    os.makedirs creates every intermediate folder a path names, so a caller
    that passed "<root>/typo/coral_intel_subset" would get a "typo" folder
    made for it and a staged dataset the Bounding Boxes page never lists. The
    command line joins a checked name to --dest-root and so is already safe;
    this check is what makes the library call safe for an in-process caller,
    for example a route that builds a subset from the page.

    Parameters:
      dest: the subset folder this run would write.
    Returns:
      None.
    Raises:
      SubsetError naming the holding folder when it is not an existing folder.
    """
    holder = os.path.dirname(dest.rstrip(os.sep))
    if not os.path.isdir(holder):
        raise SubsetError(
            f"{holder} is not there, so {dest} cannot be written inside it. "
            "Name a destination folder that already exists, or create that "
            "folder first: this tool makes the subset folder and nothing "
            "above it")


def check_output_free(dest):
    """Refuse an output path that is already taken.

    Parameters:
      dest: the folder this run would write.
    Returns:
      None.
    Raises:
      SubsetError naming the path and the remedy, or naming a sibling folder
      that matches this one only in letter case.
    """
    if os.path.exists(dest):
        raise SubsetError(
            f"{dest} already exists, so this run would write a second dataset "
            f"over the first. {EXISTS_REMEDY}")
    ing._check_folder_case_twin(dest)


# ── Counting and the class map ───────────────────────────────────────────

def count_boxes_by_class(label_dir, files, n_classes):
    """How many boxes each parent class carries, over every label file.

    One file is read at a time and its text is released before the next is
    opened, so a parent of any size is counted without its labels ever being
    in memory together.

    Label files carrying no box at all are counted separately. They were never
    candidates for the class-count rule, so a subset that reported them among
    the images the rule dropped would name the wrong reason for their absence.

    Parameters:
      label_dir: the parent label folder being read.
      files: the label file basenames in it.
      n_classes: how many classes the parent declares.
    Returns:
      ({class id: box count} over every class id seen, total boxes, how many
      label files held no box at all).
    Raises:
      SubsetError naming the file and the line for a label line that is not
      exactly five fields, carries a field that is not a number, or carries a
      class id outside the parent class list.
    """
    counts = {}
    total = 0
    n_background = 0
    for name in files:
        text = _read_label(os.path.join(label_dir, name))
        boxes_here = 0
        for cls, _coords in ing.parse_boxes(text, n_classes, name):
            counts[cls] = counts.get(cls, 0) + 1
            boxes_here += 1
        total += boxes_here
        if not boxes_here:
            n_background += 1
    return counts, total, n_background


def _read_label(path):
    """One label file's text.

    Parameters:
      path: the label file.
    Returns:
      its content as a string.
    Raises:
      SubsetError naming the file when it cannot be read.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise SubsetError(f"label file {path} could not be read: {exc}")


def split_classes(classes, counts, min_boxes, where):
    """Which classes are kept and which are dropped, in parent order.

    Parameters:
      classes: the parent class tokens in class-id order.
      counts: {class id: box count} from count_boxes_by_class.
      min_boxes: how many boxes a class needs to be kept.
      where: the parent folder, quoted back in the refusal.
    Returns:
      (kept, dropped): kept is [(old_id, new_id, name, boxes)] with new_id
      running from 0 in parent order, dropped is [(old_id, name, boxes)].
    Raises:
      SubsetError, naming the largest class, when no class meets the minimum,
      because a dataset with no class trains nothing.
    """
    kept, dropped = [], []
    for old_id, name in enumerate(classes):
        boxes = counts.get(old_id, 0)
        if boxes >= min_boxes:
            kept.append((old_id, len(kept), name, boxes))
        else:
            dropped.append((old_id, name, boxes))
    if not kept:
        largest = max(counts.values()) if counts else 0
        raise SubsetError(
            f"a minimum of {min_boxes} boxes keeps no class of {where}: its "
            f"largest class carries {largest} box(es). Give a minimum of "
            f"{largest} or fewer")
    return kept, dropped


# ── Writing ──────────────────────────────────────────────────────────────

def write_labels_and_images(parent_dir, dest, label_dir, files, images, id_map,
                            log):
    """Write the subset's labels and copy the images that keep a box.

    One label file is read, filtered and written at a time. An image whose
    every box belonged to a dropped class is not copied at all: an image with
    no box trains the detector on an empty scene rather than on the species it
    shows.

    Parameters:
      parent_dir: the staged parent dataset folder.
      dest: the subset folder, already created.
      label_dir: the parent label folder being read.
      files: the label file basenames in it.
      images: {stem: image basename} from image_index.
      id_map: {parent class id: subset class id} for the surviving classes.
      log: where progress lines go.
    Returns:
      (counts, staged_images): counts carries n_images, n_boxes, n_degenerate,
      n_degenerate_exact and n_duplicate_lines for what was written;
      staged_images is the sorted list of image basenames that were copied.
    Raises:
      SubsetError naming the file when a label cannot be read or written, an
      image cannot be copied, or a label line stopped being a box between the
      counting pass and this one (a parent that changed while the run was
      reading it).
    """
    counts = {"n_images": 0, "n_boxes": 0, "n_degenerate": 0,
              "n_degenerate_exact": 0, "n_duplicate_lines": 0}
    staged_images = []
    for i, name in enumerate(files, 1):
        stem = name[:-len(LABEL_EXT)]
        text = _read_label(os.path.join(label_dir, name))
        lines, n_degenerate, n_exact = _kept_lines(text, id_map, name)
        if not lines:
            continue
        _write_text(os.path.join(dest, SUBSET_LABEL_DIR, name),
                    "".join(line + "\n" for line in lines))
        source_image = os.path.join(parent_dir, IMAGES_DIR, images[stem])
        try:
            shutil.copy2(source_image,
                         os.path.join(dest, IMAGES_DIR, images[stem]))
        except OSError as exc:
            raise SubsetError(
                f"{source_image} could not be copied into {dest}: {exc}")
        staged_images.append(images[stem])
        counts["n_images"] += 1
        counts["n_boxes"] += len(lines)
        counts["n_degenerate"] += n_degenerate
        counts["n_degenerate_exact"] += n_exact
        counts["n_duplicate_lines"] += ing._count_repeats(lines)
        if i % PROGRESS_EVERY == 0:
            log(f"read {i} of {len(files)} parent label file(s)")
    return counts, sorted(staged_images)


def _kept_lines(text, id_map, where):
    """The label lines of one file that survive, rewritten with their new ids.

    A kept line is written as "new_id cx cy w h" with one space between
    fields, so a parent line whose fields were separated by a tab or by
    several spaces is written here in the one form every reader parses.

    Parameters:
      text: the parent label file's content.
      id_map: {parent class id: subset class id} for the surviving classes.
      where: the label file name, quoted back in any refusal.
    Returns:
      (lines, n_degenerate, n_degenerate_exact): the rewritten lines in file
      order, how many of them cover the whole frame, and how many are written
      in the literal whole-frame form.
    Raises:
      SubsetError naming the file and the line when a line stopped being a
      box between the counting pass and this one.
    """
    lines, n_degenerate, n_exact = [], 0, 0
    for n, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != LABEL_FIELDS:
            raise SubsetError(
                f"{where} line {n}: expected {LABEL_FIELDS} fields "
                f"'class cx cy w h', got {len(fields)}. The parent changed "
                "while this run was reading it")
        try:
            cls = int(fields[0])
            coords = [float(value) for value in fields[1:LABEL_FIELDS]]
        except ValueError:
            raise SubsetError(
                f"{where} line {n}: expected 'class cx cy w h', got "
                f"{line.strip()!r}. The parent changed while this run was "
                "reading it")
        if cls not in id_map:
            continue
        rewritten = " ".join([str(id_map[cls])] + fields[1:LABEL_FIELDS])
        lines.append(rewritten)
        if coral.is_degenerate(coords):
            n_degenerate += 1
        if coral.is_exact_degenerate_line(rewritten):
            n_exact += 1
    return lines, n_degenerate, n_exact


def read_parent_names(parent_dir):
    """The parent's names table, read before anything is written.

    Read during the checks rather than at writing time, so a names table the
    parent cannot be read from stops the run while the output folder is still
    empty.

    Parameters:
      parent_dir: the staged parent dataset folder.
    Returns:
      (rows, path): [(token, full name)] in file order and the absolute path
      they came from, or ([], "") when the parent carries no names table.
    Raises:
      SubsetError naming the file when it cannot be read, carries another
      header, or names one token twice (through
      ingest_yolo_boxes.read_names_csv).
    """
    source = os.path.join(parent_dir, NAMES_CSV_NAME)
    if not os.path.isfile(source):
        return [], ""
    return ing.read_names_csv(source), os.path.abspath(source)


def narrow_names_csv(dest, rows, kept_names):
    """Write the parent's names table, narrowed to the surviving classes.

    Parameters:
      dest: the subset folder.
      rows: [(token, full name)] from read_parent_names.
      kept_names: the surviving class tokens, in subset class-id order.
    Returns:
      True when a names table was written, False when the parent carries none
      or names none of the surviving classes.
    Raises:
      SubsetError naming the file when the write fails.
    """
    keep = set(kept_names)
    narrowed = [(token, full) for token, full in rows if token in keep]
    if not narrowed:
        return False
    _write_csv(os.path.join(dest, NAMES_CSV_NAME), ing.NAMES_HEADER, narrowed)
    return True


def read_parent_pins(parent_dir):
    """The parent's split pins, as the document it wrote.

    Parameters:
      parent_dir: the staged parent dataset folder.
    Returns:
      the split_pins.json document as a dict, or {} when the parent has none.
    Raises:
      SubsetError naming the file when it cannot be read, is not a JSON
      object, carries no pins object, or pins a split outside the three the
      split builder knows. Such a file is refused rather than passed over,
      because a subset that dropped it would re-split every image it names.
    """
    path = os.path.join(parent_dir, SPLIT_PINS_NAME)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise SubsetError(
            f"{path} could not be read: {exc}. Fix or remove it in the parent "
            "before subsetting, because dropping it would re-split every "
            "image it names")
    if not isinstance(doc, dict) or not isinstance(doc.get("pins"), dict):
        raise SubsetError(
            f"{path} must hold a JSON object carrying a pins object of image "
            f"basename to one of {SPLIT_NAMES}, got "
            f"{type(doc).__name__}. Fix or remove it in the parent before "
            "subsetting, because dropping it would re-split every image it "
            "names")
    bad = [(key, value) for key, value in doc["pins"].items()
           if value not in SPLIT_NAMES]
    if bad:
        raise SubsetError(
            f"{path}: {len(bad)} pin(s) name a split that is not one of "
            f"{SPLIT_NAMES}, for example {bad[:3]!r}. Fix or remove it in the "
            "parent before subsetting")
    return doc


def narrow_split_pins(doc, dest, staged_images):
    """Write the parent's split pins, narrowed to the images that survive.

    Parameters:
      doc: the parent split_pins.json document, or {} for none.
      dest: the subset folder.
      staged_images: the image basenames this run copied.
    Returns:
      {image basename: split} as written, or {} when nothing was written.
    Raises:
      SubsetError naming the file when the write fails.
    """
    if not doc:
        return {}
    keep = set(staged_images)
    pins = {name: split for name, split in sorted(doc["pins"].items())
            if name in keep}
    if not pins:
        return {}
    out = {"schema_version": doc.get("schema_version",
                                     ing.SPLIT_PINS_SCHEMA_VERSION),
           "source": doc.get("source", ""),
           "pins": pins}
    path = os.path.join(dest, SPLIT_PINS_NAME)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    except OSError as exc:
        raise SubsetError(f"{path} could not be written: {exc}")
    return pins


def resolve_license(parent_dir, manifest):
    """Decide where the subset's licence text comes from, before anything is written.

    Parameters:
      parent_dir: the staged parent dataset folder.
      manifest: the parent manifest.
    Returns:
      (source_path, text): the parent LICENSE.md to copy verbatim, or the text
      to write when the parent has no LICENSE.md but records its terms in the
      manifest. Exactly one of the two is non-empty.
    Raises:
      SubsetError naming the file when the parent records no terms at all: a
      copy that loses the terms the data came under is a licensing problem,
      not a tidiness problem.
    """
    source = os.path.join(parent_dir, LICENSE_NAME)
    if os.path.isfile(source):
        return source, ""
    note = manifest.get("license_note")
    note = note.strip() if isinstance(note, str) else ""
    if not note:
        raise SubsetError(
            f"{source} is not there and the parent manifest records no "
            "license_note, so this subset would carry no license terms at "
            "all. Write the parent's LICENSE.md, or record its terms in the "
            "parent manifest, then subset again")
    return "", "# License\n\n" + note + "\n"


def write_license(dest, source, text):
    """Put the licence the parent carries into the subset.

    The file is copied or written and nothing is read back: a read-back is
    wasted work on every run, and the one reader this module has is the label
    reader, whose failure message would call LICENSE.md a label file.

    Parameters:
      dest: the subset folder.
      source: the parent LICENSE.md to copy verbatim, or "".
      text: the licence text to write when there is no file to copy.
    Returns:
      None.
    Raises:
      SubsetError naming the file when the copy fails, or IngestError naming
      it when the write fails.
    """
    target = os.path.join(dest, LICENSE_NAME)
    if not source:
        _write_text(target, text)
        return
    try:
        shutil.copy2(source, target)
    except OSError as exc:
        raise SubsetError(f"{source} could not be copied to {target}: {exc}")


# ── The README ───────────────────────────────────────────────────────────

def render_readme(facts):
    """The subset's README.md, in plain words for a reader who was not here.

    Every paragraph is wrapped after the counts are put into it, so a dataset
    of four images and one of ten thousand read the same way. The parent's
    licence is quoted from the one paragraph its manifest records rather than
    from LICENSE.md, whose own headings would otherwise take over this
    document; LICENSE.md sits beside this file and carries the full terms.

    Parameters:
      facts: the dict subset_by_class_count assembles, carrying name,
        parent_name, parent_dir, made, min_boxes, label_rel, localized,
        n_classes, n_classes_parent, n_classes_dropped, n_boxes,
        n_boxes_parent, n_boxes_dropped, n_images, n_images_parent,
        n_images_dropped, n_background_parent, license_note, citation,
        has_names and has_pins.
    Returns:
      the README text.
    """
    parent = facts["parent_name"]
    blocks = [
        f"# {facts['name']}",
        _fill(f"A class-count subset of the staged bounding-box dataset "
              f"`{parent}`."),
        _readme_header_list(facts),
        "## What this dataset is",
        _fill(f"This folder is a staged bounding-box dataset in the layout "
              f"every other staged dataset uses: `images/`, `labels/`, "
              f"`classes.txt`, `data.yaml` and `ingest_manifest.json`. It "
              f"holds the {facts['n_classes']} classes of `{parent}` whose "
              f"labels carry at least {facts['min_boxes']} boxes, and the "
              f"{facts['n_images']} parent images that still have at least "
              f"one box once the other classes were removed. Nothing was "
              f"re-annotated and no box moved: every box here is the parent's "
              f"box, under a new class id."),
        _readme_upstream_note(facts),
        "## What was kept and what was dropped",
        _readme_table(facts),
        _readme_table_note(facts),
        _fill("An image whose every box belonged to a dropped class is not "
              "copied here at all. `dropped_classes.csv` lists every class "
              "that did not meet the minimum and how many boxes it had, so "
              "what is missing is on the record rather than left to be "
              "noticed."),
        _readme_background(facts),
        "## Class ids were renumbered",
        _fill("Ultralytics refuses a label file carrying a class id at or "
              "above the number of classes the dataset declares, so the "
              "surviving classes are numbered again from 0, in the order they "
              "had in the parent, and every kept label line was rewritten "
              "with its new id. `class_map.csv` holds the mapping: one row "
              "per surviving class with its id in the parent (`old_id`), its "
              "id here (`new_id`), its name and its box count. Read any "
              "prediction from a model trained on this subset through that "
              "file before setting it beside anything measured against the "
              "parent."),
        "## What a score measured here means",
        _fill(f"Mean average precision is an average over classes. The "
              f"classes this rule removes are the ones with the fewest boxes, "
              f"which are also the classes a detector does worst on, so a "
              f"model trained and scored on this subset reports a higher "
              f"number than the same model scored against `{parent}` without "
              f"being any better at finding corals. A score measured here is "
              f"comparable only with another score measured on this same "
              f"subset, at this same minimum. Never quote it beside a score "
              f"measured on the parent."),
        "## License and citation",
        _readme_license(facts),
        "## What each file holds",
        _readme_file_list(facts),
        _fill(f"Basenames are the parent's, unchanged, so a record here is "
              f"the record of the same name in `{parent}`."),
    ]
    return "\n\n".join(block for block in blocks if block) + "\n"


def _fill(text):
    """One paragraph wrapped to README_WIDTH, leaving long paths whole.

    Parameters:
      text: the paragraph, already carrying its numbers.
    Returns:
      the wrapped paragraph.
    """
    return textwrap.fill(text, width=README_WIDTH, break_long_words=False,
                         break_on_hyphens=False)


def _readme_header_list(facts):
    """The README's opening list: when this was made, from what, under what rule.

    Parameters:
      facts: the dict render_readme was given.
    Returns:
      the markdown list, one fact per line.
    """
    return "\n".join([
        f"- Made: {facts['made']} by LO",
        f"- Tool: `scripts/_boxes/subset_by_class_count.py`",
        f"- Parent dataset: `{facts['parent_name']}`, at `{facts['parent_dir']}`",
        f"- Parent labels read from: `{facts['label_rel']}/`",
        f"- Rule: a class is kept when the parent's labels carry at least "
        f"{facts['min_boxes']} boxes of it.",
    ])


def _readme_upstream_note(facts):
    """What the parent's labels_localized/ folder already left out, when it was read.

    Parameters:
      facts: the dict render_readme was given.
    Returns:
      the paragraph, or "" when the parent's plain labels/ folder was read and
      no upstream filtering happened.
    """
    if not facts["localized"]:
        return ""
    return _fill(
        "These labels were read from the parent's `labels_localized/` folder, "
        "which is the folder the training split builder prefers. It holds the "
        "parent's boxes minus the ones that cover the entire frame and so "
        "point at no particular colony, which means those whole-frame boxes "
        "were already excluded upstream and none of them reached this subset.")


def _readme_table(facts):
    """The kept, dropped and parent-total counts as a markdown table.

    Parameters:
      facts: the dict render_readme was given.
    Returns:
      the table.
    """
    return "\n".join([
        "| | kept | dropped | parent total |",
        "|---|---|---|---|",
        f"| classes | {facts['n_classes']} | {facts['n_classes_dropped']} | "
        f"{facts['n_classes_parent']} |",
        f"| boxes in {facts['label_rel']}/ | {facts['n_boxes']} | "
        f"{facts['n_boxes_dropped']} | {facts['n_boxes_parent']} |",
        f"| images | {facts['n_images']} | {facts['n_images_dropped']} | "
        f"{facts['n_images_parent']} |",
    ])


def _readme_table_note(facts):
    """Why the parent's card can show a larger box total than the table does.

    The table counts the label folder this run read. A Coral Intel parent's
    card on the Bounding Boxes page counts its labels/ folder, which still
    carries the whole-frame boxes, so the two documents give two totals for one
    parent and the difference has to be said where the number is quoted.

    Parameters:
      facts: the dict render_readme was given.
    Returns:
      the paragraph, or "" when the parent's plain labels/ folder was read and
      there is no second total to reconcile.
    """
    if not facts["localized"]:
        return ""
    return _fill(
        f"The box totals in that table count the parent's "
        f"`{facts['label_rel']}/` folder. The parent's own card on the "
        f"Bounding Boxes page counts its `labels/` folder instead, so the box "
        f"total shown there is the larger of the two: it still includes the "
        f"whole-frame boxes that `{facts['label_rel']}/` leaves out.")


def _plural(count, word):
    """A count and its noun, with the noun made plural when the count is not one.

    Parameters:
      count: how many.
      word: the singular noun, for example "parent image".
    Returns:
      the phrase, for example "740 parent images" or "1 parent image".
    """
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _readme_background(facts):
    """The parent images that carried no box at all, and why none is here.

    Parameters:
      facts: the dict render_readme was given.
    Returns:
      the paragraph, or "" when every parent label file carried at least one
      box and there is nothing to account for.
    """
    if not facts["n_background_parent"]:
        return ""
    lost_boxes = facts["n_images_dropped"] - facts["n_background_parent"]
    if facts["localized"]:
        why = (" A parent image is in that state when every box it carried "
               "covered the whole frame and the parent's own filter removed "
               "it, so the picture still shows a coral and staging it as a "
               "frame with nothing to find would teach a detector the "
               "opposite.")
    else:
        why = (" A parent image is in that state when it was staged as a "
               "background image, to show a detector what a frame with "
               "nothing to find looks like. Whether a training set wants "
               "those is a decision for whoever stages it, so this subset "
               "leaves them out rather than making that decision quietly.")
    return _fill(
        f"{facts['n_background_parent']} of the {facts['n_images_parent']} "
        f"parent images carry no box at all in `{facts['label_rel']}/`, and "
        f"none of them is copied here." + why +
        f" `ingest_manifest.json` records that count as "
        f"`n_background_dropped`, apart from the "
        f"{_plural(lost_boxes, 'parent image')} that lost every box they had "
        f"to the class-count rule.")


def _readme_license(facts):
    """The licence and citation the subset carries from its parent.

    Parameters:
      facts: the dict render_readme was given.
    Returns:
      the paragraphs, the licence note quoted only when the parent manifest
      records one.
    """
    opening = (f"Carried from `{facts['parent_name']}` and unchanged. The "
               f"full text is in `LICENSE.md` beside this file.")
    parts = [_fill(opening)]
    if facts["license_note"]:
        parts.append(_fill(facts["license_note"]))
    parts.append(_fill(f"Citation: {facts['citation']}"))
    return "\n\n".join(parts)


def _readme_file_list(facts):
    """What each file in the subset holds, one line per file.

    Parameters:
      facts: the dict render_readme was given.
    Returns:
      the markdown list.
    """
    lines = [
        "- `images/`: one copy of every parent image that kept at least one box.",
        "- `labels/`: one YOLO text file per image, lines \"class cx cy w h\",",
        "  with the new class ids.",
        "- `classes.txt`: the surviving class names, one per line, in new-id order.",
        "- `data.yaml`: the Ultralytics dataset config, names keyed by the new ids.",
        "- `class_map.csv`: old_id, new_id, name, boxes for every surviving class.",
        "- `dropped_classes.csv`: old_id, name, boxes for every class that was",
        "  dropped.",
        "- `LICENSE.md`: the parent's terms, carried verbatim.",
        "- `ingest_manifest.json`: the counts above, the class list, the parent",
        "  this came from and the AST timestamp it was made.",
    ]
    return "\n".join(lines) + _readme_sidecar_lines(facts)


def _readme_sidecar_lines(facts):
    """The README lines for the two optional files, when the subset has them.

    Parameters:
      facts: the dict render_readme was given.
    Returns:
      the extra markdown list items, or "" when the subset carries neither
      file.
    """
    out = ""
    if facts["has_names"]:
        out += ("\n- `names.csv`: the full name behind every surviving class "
                "token, narrowed from the parent's table.")
    if facts["has_pins"]:
        out += ("\n- `split_pins.json`: the train, valid or test split each "
                "surviving image had in the experiment the parent came from, "
                "narrowed to the images that are here.")
    return out


# ── The manifest ─────────────────────────────────────────────────────────

def parent_subject(parent_manifest, parent_dir):
    """The subject the parent dataset recorded, checked against the vocabulary.

    A subset is its parent's data with some classes removed, so it is about
    exactly what the parent was about and the Train panel has to read the same
    word off both.

    Parameters:
      parent_manifest: the parent's manifest dict.
      parent_dir: the staged parent folder, named in a refusal.
    Returns:
      one of ingest_yolo_boxes.DATASET_SUBJECTS, or "" when the parent records
      no subject, which is every dataset staged before 2026-09-15.
    Raises:
      SubsetError naming the file and the value when the parent records a
      subject outside the vocabulary: copying that forward would give the
      subset a word no training recipe can ever match, and the fix belongs in
      the parent.

    Example: a parent manifest holding {"subject": "coral"} gives "coral".
    """
    raw = parent_manifest.get("subject") or ""
    try:
        return ing.check_subject(raw)
    except ing.IngestError as exc:
        raise SubsetError(
            f"{os.path.join(parent_dir, MANIFEST_NAME)} records subject {raw!r}, "
            f"which this subset would carry forward: {exc}")


def build_manifest(*, parent_manifest, facts, dest, parent_dir, label_dir,
                   kept, counts, pins, names_csv, made_at, subject):
    """Assemble the subset's ingest_manifest.json.

    Every key the parent manifest carries is copied through, so the Bounding
    Boxes page renders a subset's card exactly as it renders its parent's;
    every count is then recomputed for what this run actually wrote, and the
    eleven subset keys are added on top.

    Parameters:
      parent_manifest: the parent's manifest.
      facts: the dict of counts and names render_readme was given.
      dest: the subset folder.
      parent_dir: the staged parent dataset folder.
      label_dir: the parent label folder that was read.
      kept: [(old_id, new_id, name, boxes)] for the surviving classes.
      counts: what write_labels_and_images counted.
      pins: {image basename: split} as written, or {}.
      names_csv: the parent names table that was read, or "".
      made_at: the AST timestamp this subset was made.
      subject: what parent_subject read off the parent, "" for none.
    Returns:
      the manifest dict.
    """
    manifest = dict(parent_manifest)
    manifest.update({
        "dataset": os.path.basename(dest.rstrip(os.sep)),
        "n_images": counts["n_images"],
        "n_boxes": counts["n_boxes"],
        "n_degenerate": counts["n_degenerate"],
        "n_degenerate_exact": counts["n_degenerate_exact"],
        "n_classes": len(kept),
        "classes": [name for _old, _new, name, _boxes in kept],
        "source_path": parent_dir,
        "ingested_at": made_at,
        "limit": None,
        "source_kind": SOURCE_KIND,
        "labels_source": label_dir,
        "names_csv": names_csv,
        "split_pins": bool(pins),
        "n_pinned": len(pins),
        "pins_by_split": {split: sum(1 for value in pins.values()
                                     if value == split)
                          for split in SPLIT_NAMES},
        "n_renamed": 0,
        "n_case_disambiguated": 0,
        # Zero by construction: a subset copies an image only when at least one
        # of its boxes survived, so it never writes an empty label file. The
        # parent's own background images are counted in n_background_dropped.
        "n_background": 0,
        "n_duplicate_lines": counts["n_duplicate_lines"],
        "parent_dataset": facts["parent_name"],
        "parent_source_path": str(parent_manifest.get("source_path") or ""),
        "min_boxes": facts["min_boxes"],
        "n_classes_parent": facts["n_classes_parent"],
        "n_classes_dropped": facts["n_classes_dropped"],
        "n_images_dropped": facts["n_images_dropped"],
        "n_background_dropped": facts["n_background_parent"],
        "n_boxes_dropped": facts["n_boxes_dropped"],
        "label_source": facts["label_rel"],
        "subset_created_at": made_at,
        # The parent's own subject, checked and written again rather than left
        # to the dict copy above, so the key is visible here beside every other
        # recomputed value and a hand-edited parent cannot slip a word past.
        "subject": subject,
    })
    return manifest


# ── The run ──────────────────────────────────────────────────────────────

def subset_by_class_count(parent_dir, dest, min_boxes, log=print):
    """Write a class-count subset of a staged box dataset and return its manifest.

    Nothing is written until every check has passed: the parent is read twice,
    once to count the boxes each class carries and once to write the labels
    that survive, and every refusal happens in or before the counting pass.

    Parameters:
      parent_dir: a staged dataset folder holding data.yaml and
        ingest_manifest.json beside images/ and a label folder.
      dest: the subset folder to write; its basename is the dataset name.
      min_boxes: how many boxes a class needs to be kept.
      log: where progress lines go.
    Returns:
      the manifest dict, identical to the ingest_manifest.json it wrote.
    Raises:
      SubsetError naming what failed, where, and why, for every refusal listed
      in this module's docstring, or IngestError naming the file when a write
      fails.
    """
    parent_dir = os.path.abspath(os.path.expanduser(str(parent_dir)))
    dest = os.path.abspath(os.path.expanduser(str(dest)))
    name = ing.check_dataset_name(os.path.basename(dest.rstrip(os.sep)))
    min_boxes = normalise_min_boxes(min_boxes)
    if not os.path.isdir(parent_dir):
        raise SubsetError(f"parent dataset folder not found: {parent_dir}")

    parent_manifest = read_manifest(parent_dir)
    # Read before anything is written, like every other refusal in this run.
    subject = parent_subject(parent_manifest, parent_dir)
    classes = parent_class_list(parent_dir)
    label_rel, label_dir, files = choose_label_dir(parent_dir)
    images = image_index(parent_dir)
    check_case_unique(images.values(), os.path.join(parent_dir, IMAGES_DIR))
    check_case_unique(files, label_dir)
    check_every_label_has_an_image(files, images, label_dir)
    pins_doc = read_parent_pins(parent_dir)
    name_rows, names_csv = read_parent_names(parent_dir)
    license_source, license_text = resolve_license(parent_dir, parent_manifest)
    check_dest_holder(dest)
    check_output_free(dest)

    log(f"reading {len(files)} label file(s) from {label_rel}/ of "
        f"{os.path.basename(parent_dir)}")
    by_class, n_boxes_parent, n_background_parent = count_boxes_by_class(
        label_dir, files, len(classes))
    kept, dropped = split_classes(classes, by_class, min_boxes, parent_dir)
    log(f"{len(classes)} parent class(es) carry {n_boxes_parent} box(es): "
        f"keeping {len(kept)} at {min_boxes} or more, dropping {len(dropped)}")

    id_map = {old: new for old, new, _name, _boxes in kept}
    _make_dirs(dest)
    counts, staged_images = write_labels_and_images(
        parent_dir, dest, label_dir, files, images, id_map, log)
    _verify_counts(counts, kept, dest)

    facts = _facts(name=name, parent_dir=parent_dir, label_rel=label_rel,
                   min_boxes=min_boxes, classes=classes, kept=kept,
                   dropped=dropped, counts=counts, n_images_parent=len(images),
                   n_boxes_parent=n_boxes_parent,
                   n_background_parent=n_background_parent,
                   parent_manifest=parent_manifest)
    wrote_names = narrow_names_csv(dest, name_rows, facts["classes_kept"])
    pins = narrow_split_pins(pins_doc, dest, staged_images)
    facts["has_names"] = wrote_names
    facts["has_pins"] = bool(pins)
    write_license(dest, license_source, license_text)
    _write_class_files(dest, kept, dropped, facts["classes_kept"])
    _write_text(os.path.join(dest, README_NAME), render_readme(facts))

    manifest = build_manifest(parent_manifest=parent_manifest, facts=facts,
                              dest=dest, parent_dir=parent_dir,
                              label_dir=label_dir, kept=kept, counts=counts,
                              pins=pins,
                              names_csv=names_csv if wrote_names else "",
                              made_at=facts["made_at"], subject=subject)
    _write_json(os.path.join(dest, MANIFEST_NAME), manifest)

    log(f"wrote {counts['n_images']} image(s), {counts['n_boxes']} box(es) and "
        f"{len(kept)} class(es) -> {dest}")
    log(f"left out {facts['n_images_dropped']} parent image(s) and "
        f"{facts['n_boxes_dropped']} box(es) of {len(dropped)} class(es) below "
        f"{min_boxes}; dropped_classes.csv names them")
    log("class ids were renumbered from 0; class_map.csv maps them back to "
        f"{os.path.basename(parent_dir)}")
    return manifest


def _facts(*, name, parent_dir, label_rel, min_boxes, classes, kept, dropped,
           counts, n_images_parent, n_boxes_parent, n_background_parent,
           parent_manifest):
    """The counts and names the README and the manifest both read.

    Parameters:
      name: the subset dataset name.
      parent_dir: the staged parent dataset folder.
      label_rel: which parent label folder was read.
      min_boxes: the class-count minimum this run used.
      classes: the parent class tokens in class-id order.
      kept, dropped: what split_classes returned.
      counts: what write_labels_and_images counted.
      n_images_parent: how many images the parent holds.
      n_boxes_parent: how many boxes the parent label folder holds.
      n_background_parent: how many parent label files held no box at all.
      parent_manifest: the parent's manifest, read for its citation.
    Returns:
      a dict of the facts, including made_at (the AST timestamp) and
      classes_kept (the surviving class tokens in new-id order).
    """
    made_at = ing.ast_now()
    citation = parent_manifest.get("citation")
    return {
        "name": name,
        "parent_name": os.path.basename(parent_dir.rstrip(os.sep)),
        "parent_dir": parent_dir,
        "made_at": made_at,
        "made": made_at[:16] + " AST",
        "min_boxes": min_boxes,
        "label_rel": label_rel,
        "localized": label_rel != LABELS_DIR,
        "classes_kept": [class_name for _o, _n, class_name, _b in kept],
        "n_classes": len(kept),
        "n_classes_parent": len(classes),
        "n_classes_dropped": len(dropped),
        "n_boxes": counts["n_boxes"],
        "n_boxes_parent": n_boxes_parent,
        "n_boxes_dropped": n_boxes_parent - counts["n_boxes"],
        "n_images": counts["n_images"],
        "n_images_parent": n_images_parent,
        "n_images_dropped": n_images_parent - counts["n_images"],
        "n_background_parent": n_background_parent,
        "citation": str(citation or "none recorded"),
        "license_note": _text_field(parent_manifest.get("license_note")),
        "has_names": False,
        "has_pins": False,
    }


def _text_field(value):
    """One manifest text field as prose can use it.

    Parameters:
      value: what the manifest carries, of any type.
    Returns:
      the text with its surrounding whitespace removed, or "" when the field
      is absent or is not text.
    """
    return value.strip() if isinstance(value, str) else ""


def _make_dirs(dest):
    """Create the subset's images/ and labels/ folders.

    Parameters:
      dest: the subset folder.
    Returns:
      None.
    Raises:
      SubsetError naming the folder that could not be created and why.
    """
    for sub in (IMAGES_DIR, SUBSET_LABEL_DIR):
        folder = os.path.join(dest, sub)
        try:
            os.makedirs(folder)
        except OSError as exc:
            raise SubsetError(f"{folder} could not be created: {exc}")


def _verify_counts(counts, kept, dest):
    """Check that the labels written carry exactly the boxes that were counted.

    The counting pass and the writing pass read the parent's label files
    separately, so this is what proves the two agree before the manifest
    quotes either of them.

    Parameters:
      counts: what write_labels_and_images counted.
      kept: [(old_id, new_id, name, boxes)] for the surviving classes.
      dest: the subset folder, quoted back in the refusal.
    Returns:
      None.
    Raises:
      SubsetError naming both counts when they disagree, which means the
      parent changed while this run was reading it.
    """
    expected = sum(boxes for _old, _new, _name, boxes in kept)
    if counts["n_boxes"] != expected:
        raise SubsetError(
            f"{dest}: the counting pass found {expected} box(es) in the "
            f"surviving classes and the writing pass wrote {counts['n_boxes']}. "
            "The parent changed while this run was reading it, so this folder "
            "is half written and must be removed before another run")


def _write_class_files(dest, kept, dropped, class_names):
    """Write classes.txt, data.yaml, class_map.csv and dropped_classes.csv.

    Parameters:
      dest: the subset folder.
      kept: [(old_id, new_id, name, boxes)] for the surviving classes.
      dropped: [(old_id, name, boxes)] for the classes below the minimum.
      class_names: the surviving class tokens in new-id order.
    Returns:
      None.
    Raises:
      SubsetError naming the file when a write fails.
    """
    _write_text(os.path.join(dest, CLASSES_TXT_NAME),
                "".join(class_name + "\n" for class_name in class_names))
    try:
        coral._write_data_yaml(dest, class_names)
    except OSError as exc:
        raise SubsetError(
            f"{os.path.join(dest, DATA_YAML_NAME)} could not be written: {exc}")
    _write_csv(os.path.join(dest, CLASS_MAP_NAME), CLASS_MAP_HEADER, kept)
    _write_csv(os.path.join(dest, DROPPED_NAME), DROPPED_HEADER, dropped)


# ── CLI ──────────────────────────────────────────────────────────────────

def ensure_dest_root(dest_root):
    """Create the staged tree the command line was pointed at, when it is not there.

    The library call makes the subset folder and nothing above it, because an
    in-process caller can reach it with a path nobody typed. --dest-root is a
    different thing: the operator named the staged tree on the command line, so
    this tool makes it, exactly as the generic YOLO stager makes it.

    Parameters:
      dest_root: where staged datasets live.
    Returns:
      the absolute path of the staged tree.
    Raises:
      SubsetError naming the path when it is there and is not a folder, or
      when the folder cannot be created.
    """
    root = os.path.abspath(os.path.expanduser(str(dest_root)))
    if os.path.exists(root) and not os.path.isdir(root):
        raise SubsetError(
            f"{root} is there and is not a folder, so staged datasets cannot "
            "be written inside it. Give a --dest-root that names a folder")
    try:
        os.makedirs(root, exist_ok=True)
    except OSError as exc:
        raise SubsetError(f"{root} could not be created: {exc}")
    return root


def build_parser():
    """The command-line parser, split out so the flags can be read in one place.

    Returns:
      an argparse.ArgumentParser.
    """
    ap = argparse.ArgumentParser(
        description="Write a class-count subset of a staged bounding-box "
                    "dataset: only the classes carrying at least N boxes, with "
                    "the class ids renumbered from 0.")
    ap.add_argument("--parent", required=True,
                    help="the staged dataset folder to subset, holding "
                         "data.yaml and ingest_manifest.json")
    ap.add_argument("--name", required=True,
                    help="the new staged dataset folder name, 1 to 60 "
                         "characters from A-Z a-z 0-9 underscore and hyphen")
    ap.add_argument("--min-boxes", required=True,
                    help="how many boxes a class needs in the parent's labels "
                         "to be kept, for example 100")
    ap.add_argument("--dest-root", default=DEFAULT_DEST_ROOT,
                    help="where staged datasets live (default "
                         "supporting_data/bbox_datasets in this clone)")
    ap.add_argument("--quiet", action="store_true", help="print nothing")
    return ap


def main(argv=None):
    """Parse the arguments, validate them, write the subset, report the counts.

    Parameters:
      argv: the argument list, or None to read sys.argv.
    Returns:
      the manifest dict subset_by_class_count wrote.
    Raises:
      SubsetError naming what failed, where, and why, or IngestError naming
      the file when a write fails.
    """
    args = build_parser().parse_args(argv)
    log = (lambda _m: None) if args.quiet else print
    ing.check_dataset_name(args.name)
    dest = os.path.join(ensure_dest_root(args.dest_root), args.name)
    return subset_by_class_count(args.parent, dest, args.min_boxes, log=log)


if __name__ == "__main__":
    try:
        main()
    except ing.IngestError as exc:
        print(f"subset stopped: {exc}", file=sys.stderr)
        sys.exit(2)

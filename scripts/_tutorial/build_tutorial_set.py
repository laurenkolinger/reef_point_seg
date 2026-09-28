#!/usr/bin/env python
"""Build the small bounding-box dataset the VEGA guided tutorial trains on.

Module:  scripts/_tutorial/build_tutorial_set.py
Purpose: Write a NEW staged bounding-box dataset holding a few named classes
         and a fixed number of pictures of each, so that a student can run a
         whole train, evaluate and infer cycle in a few minutes instead of a
         few hours. The tutorial's lesson is the cycle, not the score, and a
         set of eight thousand pictures teaches the cycle no better while
         making the student sit through it. The result is a staged dataset in
         every respect: the Bounding Boxes page lists it and the training
         split builder reads it exactly as they read the dataset it came from.
Inputs:  --parent (a staged dataset folder holding data.yaml and
         ingest_manifest.json beside images/ and labels/), --name (the new
         staged folder name), --classes (the class tokens to keep, in the
         order they are numbered here), --per-class (how many pictures of each
         class to take). Optional: --dest-root, --train-per-class,
         --valid-per-class, --test-per-class, --seed, --quiet.
Outputs: <dest-root>/<name>/ holding images/, labels/, classes.txt, data.yaml,
         names.csv, class_map.csv, dropped_classes.csv, split_pins.json,
         LICENSE.md, README.md and ingest_manifest.json. The function returns
         the manifest it wrote.

Why this tool exists beside scripts/_boxes/subset_by_class_count.py, which is
protected and is not edited here. That tool keeps every class carrying at
least so many boxes, keeps every picture of them, and carries the parent's
split pins forward. A tutorial set needs three things it cannot express:
classes chosen by name rather than by box count (a teaching set wants species
a student can tell apart, and the most numerous classes in a coral dataset are
not those), a cap on how many pictures of each class are taken (so the run
finishes while the student is still watching), and a split pinned by this tool
when the parent pins nothing (so every student holds out the same pictures and
one student's score means the same thing as another's).

The split is pinned, not drawn. split_pins.json names the train, valid or test
split of every picture, so VEGA's split builder replays it instead of
shuffling: two students, and the same student twice, evaluate on the same
held-out pictures. Every class appears in all three splits, in the same
proportion, so the evaluation report carries a row for each of them and the
tutorial's evaluation lesson has something to read.

Class ids are renumbered. Ultralytics refuses a label file carrying a class id
at or above the number of classes the dataset declares, so the kept classes
are numbered from 0 in the order they were asked for, and every kept label
line is rewritten with its new id. class_map.csv maps a prediction from this
dataset back to the parent, and dropped_classes.csv names every parent class
that is not here, so the omission is on the record.

A picture is taken only when every box it carries belongs to a kept class. A
picture holding a coral of some other species would show the detector that
species with no box on it, teaching it that the species is background, so such
a picture is passed over and counted rather than copied with part of its
annotation missing.

Nothing is written until every check has passed. Each of these stops the run
with a message naming what failed, where, and why: a parent that is not a
staged dataset, a class name the parent does not declare, the same class named
twice, fewer than two classes, a per-class count that is not a positive whole
number, split counts that do not add up to the per-class count, a class with
fewer eligible pictures than the count asks for, an output name outside the
platform's plain-name rule, an output folder whose holding folder is not
there, an output folder that already exists, a label line that is not exactly
five fields, a class id outside the parent class list, a label file with no
picture, two parent names that differ only in letter case, and a parent that
records no licence terms.

This imagery is outside the TCRMP survey. The dataset this writes never goes
to inprocess/run_*, the mask registry, label provenance, the coverage matrix
or STATS: a staged dataset is read where it is staged.

Usage:
  env/bin/python scripts/_tutorial/build_tutorial_set.py \
      --parent supporting_data/bbox_datasets/coral_intel_subset \
      --name tutorial_boxes
"""

import argparse
import json
import os
import random
import shutil
import sys
import textwrap

try:
    from PIL import Image
except ImportError:  # the README's one pixel-size sentence is skipped without it
    Image = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)

# The staged layout is owned by the two stagers and the class-count subset
# tool under scripts/_boxes/. Their readers, their name rule, their label
# parser, their data.yaml writer and their file writers are imported rather
# than copied, so this fixture stays the same shape as every other staged
# dataset as those files change. Nothing in scripts/_boxes/ is edited here.
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from _boxes import ingest_coral_intel as coral  # noqa: E402
from _boxes import ingest_yolo_boxes as ing  # noqa: E402
from _boxes import subset_by_class_count as sub  # noqa: E402


# ── What this tool writes and what it defaults to ────────────────────────

# What the manifest records about how this dataset was made. The Bounding
# Boxes page shows it beside the dataset's counts. It is deliberately not the
# class-count subset tool's word, because that tool did not write this and a
# reader following that card's rebuild line would run the wrong command.
SOURCE_KIND = "tutorial"

# The staged folder this tool writes by default, and the staged dataset it
# reads by default.
DEFAULT_NAME = "tutorial_boxes"
DEFAULT_PARENT = os.path.join("supporting_data", "bbox_datasets",
                              "coral_intel_subset")

# The four species the tutorial teaches on, in the order they are numbered
# here. Chosen for a student, not for a leaderboard: each carries well over
# 250 pictures in the parent, and the four growth forms are as different from
# one another as Caribbean corals get, so a detector separates them in a few
# epochs and a student can see from the pictures why. The parent's confusable
# pairs are deliberately absent: Orbicella annularis beside Orbicella
# faveolata, Siderastrea siderea beside Siderastrea radians, the three
# Agaricia species, and Acropora cervicornis beside Acropora palmata.
TUTORIAL_CLASSES = ("Acropora-palmata",
                    "Dendrogyra-cylindrus",
                    "Diploria-labyrinthiformis",
                    "Eusmilia-fastigiata")

# The full name behind every token this tool writes into names.csv, so the
# Train panel and the evaluation report show the binomial rather than the
# hyphenated token. A token with no row here is written with its hyphens
# turned into spaces, which is the right answer for every token in the parent.
FULL_NAMES = {
    "Acropora-palmata": "Acropora palmata",
    "Dendrogyra-cylindrus": "Dendrogyra cylindrus",
    "Diploria-labyrinthiformis": "Diploria labyrinthiformis",
    "Eusmilia-fastigiata": "Eusmilia fastigiata",
}

# The common name of each species, for the README's one descriptive sentence.
# A token with no row here is described by its binomial alone.
COMMON_NAMES = {
    "Acropora-palmata": "elkhorn coral",
    "Dendrogyra-cylindrus": "pillar coral",
    "Diploria-labyrinthiformis": "grooved brain coral",
    "Eusmilia-fastigiata": "smooth flower coral",
}

# How many pictures of each class the set holds, and how those split. Equal
# quotas per class are a teaching point in themselves: the student meets a
# balanced set first and class imbalance later, on real survey data. The three
# split counts must add up to the per-class count, which check_quotas asserts.
IMAGES_PER_CLASS = 100
TRAIN_PER_CLASS = 70
VALID_PER_CLASS = 15
TEST_PER_CLASS = 15

# What the per-class shuffle is seeded on. Fixed, and recorded in the
# manifest, so this tool run twice writes the same dataset with the same
# held-out pictures: a tutorial whose holdout moved between two students would
# make their scores incomparable.
DEFAULT_SEED = "vega-tutorial-boxes"

# The order the splits are filled from the shuffled list. Held out first, the
# order detect_dataset.assign_split fills its own splits in, so a reader of
# both files meets one convention.
SPLIT_FILL_ORDER = ("test", "valid", "train")

# How wide the README's paragraphs are wrapped, matching the class-count
# subset tool's README.
README_WIDTH = 79

# How many offending names a refusal lists before it counts the rest.
REFUSAL_EXAMPLES = ing.REFUSAL_EXAMPLES

# The staged file names, held under this module's own names so a rename in the
# subset tool reaches this one.
MANIFEST_NAME = sub.MANIFEST_NAME
CLASSES_TXT_NAME = sub.CLASSES_TXT_NAME
NAMES_CSV_NAME = sub.NAMES_CSV_NAME
SPLIT_PINS_NAME = sub.SPLIT_PINS_NAME
CLASS_MAP_NAME = sub.CLASS_MAP_NAME
DROPPED_NAME = sub.DROPPED_NAME
README_NAME = sub.README_NAME
IMAGES_DIR = sub.IMAGES_DIR
LABELS_DIR = sub.SUBSET_LABEL_DIR
LABEL_EXT = sub.LABEL_EXT
SPLIT_NAMES = sub.SPLIT_NAMES

CLASS_MAP_HEADER = sub.CLASS_MAP_HEADER
DROPPED_HEADER = sub.DROPPED_HEADER

_write_text = ing._write_text
_write_csv = ing._write_csv
_write_json = ing._write_json


class TutorialError(sub.SubsetError):
    """Anything that stops the build before it writes a file.

    A subclass of the class-count subset tool's error, which is itself a
    subclass of the stagers' IngestError, so a caller that already catches
    either one catches a refusal raised here as well.
    """


# ── Validation ───────────────────────────────────────────────────────────

def check_class_choice(parent_classes, wanted, where):
    """Which parent classes this build keeps, numbered again from 0.

    Parameters:
      parent_classes: the parent class tokens in class-id order.
      wanted: the class tokens asked for, in the order they are to be
        numbered here.
      where: the parent folder, quoted back in a refusal.
    Returns:
      [(old_id, new_id, token)] in the order they were asked for.
    Raises:
      TutorialError when fewer than two classes are asked for, when a token is
      named twice, or when a token is not one the parent declares. A detector
      trained on one class shows a student no per-class report, and a token
      the parent never heard of would silently write an empty class.

    Example: check_class_choice(["a", "b", "c"], ["c", "a"], "/d") gives
    [(2, 0, "c"), (0, 1, "a")].
    """
    tokens = [str(token).strip() for token in wanted]
    if len(tokens) < 2:
        raise TutorialError(
            "a tutorial set needs at least two classes so the evaluation "
            f"report has something to compare, got {len(tokens)}")
    seen = {}
    for position, token in enumerate(tokens, 1):
        if token in seen:
            raise TutorialError(
                f"class {token!r} is named twice, at position {seen[token]} "
                f"and position {position}, so it would take two class ids")
        seen[token] = position
    missing = [token for token in tokens if token not in parent_classes]
    if missing:
        raise TutorialError(
            f"{where} declares no class named "
            + ", ".join(repr(token) for token in missing[:REFUSAL_EXAMPLES])
            + (f" (and {len(missing) - REFUSAL_EXAMPLES} more)"
               if len(missing) > REFUSAL_EXAMPLES else "")
            + f". Its {len(parent_classes)} class(es) are listed in "
            f"{CLASSES_TXT_NAME} inside that folder")
    return [(parent_classes.index(token), new_id, token)
            for new_id, token in enumerate(tokens)]


def check_quotas(per_class, train, valid, test):
    """Refuse split counts that do not describe the per-class count.

    Parameters:
      per_class: how many pictures of each class the set holds.
      train, valid, test: how many of those go to each split.
    Returns:
      {"train": train, "valid": valid, "test": test}.
    Raises:
      TutorialError when a count is not a positive whole number, or when the
      three do not add up to the per-class count. A split with no picture of a
      class leaves that class out of the report the split feeds, and an
      evaluation lesson read off a report with a missing row teaches nothing.

    Example: check_quotas(80, 56, 12, 12) gives
    {"train": 56, "valid": 12, "test": 12}.
    """
    quotas = {"train": train, "valid": valid, "test": test}
    for label, value in [("per-class", per_class)] + sorted(quotas.items()):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TutorialError(
                f"the {label} picture count must be a positive whole number, "
                f"got {value!r}: every class needs at least one picture in "
                "every split")
    total = train + valid + test
    if total != per_class:
        raise TutorialError(
            f"the split counts add up to {total} picture(s) per class "
            f"(train {train}, valid {valid}, test {test}) and the per-class "
            f"count is {per_class}. Give three counts that add up to it, so "
            "no picture of a class is taken and then left out of every split")
    return quotas


# ── Reading the parent ───────────────────────────────────────────────────

def scan_parent_labels(label_dir, files, n_classes, kept_ids):
    """Read every parent label file once: what it holds and whether it is eligible.

    One file is read at a time and its text is released before the next is
    opened, so a parent of any size is read without its labels ever being in
    memory together.

    Parameters:
      label_dir: the parent label folder.
      files: the label file basenames in that folder, sorted.
      n_classes: how many classes the parent declares.
      kept_ids: the parent class ids this build keeps.
    Returns:
      (eligible, boxes_by_class, n_mixed). eligible is {kept class id: [label
      stem]} in file order, holding every picture whose every box belongs to a
      kept class, each picture offered to exactly one class (the kept class
      carrying the most of its boxes, ties going to the lowest parent class
      id). boxes_by_class is {parent class id: box count} over the whole
      parent, which class_map.csv and dropped_classes.csv report. n_mixed is
      how many pictures carry a kept class beside a dropped one and were
      therefore passed over.
    Raises:
      TutorialError, through the stagers' label parser, naming the file and
      the line when a line is not a box or names a class the parent does not
      declare.
    """
    kept = set(kept_ids)
    eligible = {class_id: [] for class_id in kept_ids}
    boxes_by_class = {}
    n_mixed = 0
    for name in files:
        stem = name[:-len(LABEL_EXT)]
        text = sub._read_label(os.path.join(label_dir, name))
        here = {}
        for class_id, _coords in ing.parse_boxes(text, n_classes, name):
            here[class_id] = here.get(class_id, 0) + 1
            boxes_by_class[class_id] = boxes_by_class.get(class_id, 0) + 1
        mine = {class_id: n for class_id, n in here.items() if class_id in kept}
        if not mine:
            continue
        if len(mine) != len(here):
            n_mixed += 1
            continue
        primary = min(mine, key=lambda class_id: (-mine[class_id], class_id))
        eligible[primary].append(stem)
    return eligible, boxes_by_class, n_mixed


def check_enough_images(eligible, per_class, kept, where):
    """Refuse a class the parent cannot fill the quota for.

    Parameters:
      eligible: {parent class id: [label stem]} from scan_parent_labels.
      per_class: how many pictures of each class the set holds.
      kept: [(old_id, new_id, token)] from check_class_choice.
      where: the parent folder, quoted back in the refusal.
    Returns:
      None.
    Raises:
      TutorialError naming every short class and how many pictures it has, so
      an operator can lower --per-class or choose another class in one go
      rather than one refusal at a time.
    """
    short = [(token, len(eligible.get(old_id, [])))
             for old_id, _new_id, token in kept
             if len(eligible.get(old_id, [])) < per_class]
    if not short:
        return
    raise TutorialError(
        f"{where} cannot fill {per_class} picture(s) of every class asked "
        "for: "
        + ", ".join(f"{token} has {count}" for token, count in short)
        + ". Lower --per-class to the smallest of those, or choose classes "
        "the parent carries more of. A picture counts for a class only when "
        "every box it carries belongs to a class this build keeps")


# ── Choosing the pictures and pinning their splits ───────────────────────

def select_and_pin(eligible, kept, per_class, quotas, seed):
    """Choose the pictures of each class and pin each one to a split.

    The shuffle is seeded on the seed text and the class token together, so
    the same call always chooses the same pictures and pins the same holdout,
    and adding a class does not move the pictures already chosen for the
    others.

    Parameters:
      eligible: {parent class id: [label stem]} from scan_parent_labels.
      kept: [(old_id, new_id, token)] from check_class_choice.
      per_class: how many pictures of each class to take.
      quotas: {"train": n, "valid": n, "test": n} from check_quotas.
      seed: the text the shuffle is seeded on.
    Returns:
      (pins, primary): pins is {label stem: "train" | "valid" | "test"} over
      every chosen picture, and primary is {label stem: tutorial class id},
      the class the picture was chosen for.
    Raises:
      TutorialError when a class has fewer eligible pictures than the quota,
      which check_enough_images normally catches first.

    Example: with one class of ten eligible stems, per_class 4 and quotas of
    two train, one valid and one test, the result names four of those ten and
    gives one of them "test".
    """
    pins, primary = {}, {}
    for old_id, new_id, token in kept:
        names = sorted(eligible.get(old_id, []))
        if len(names) < per_class:
            raise TutorialError(
                f"class {token!r} has {len(names)} eligible picture(s) and "
                f"{per_class} were asked for")
        shuffled = list(names)
        random.Random(f"{seed}:{token}").shuffle(shuffled)
        taken = shuffled[:per_class]
        at = 0
        for split in SPLIT_FILL_ORDER:
            for stem in taken[at:at + quotas[split]]:
                pins[stem] = split
                primary[stem] = new_id
            at += quotas[split]
    return pins, primary


# ── Writing ──────────────────────────────────────────────────────────────

def write_pairs(parent_dir, dest, label_dir, pins, images, id_map, log):
    """Copy the chosen pictures and write their labels under the new class ids.

    One label file is read, rewritten and written at a time, and its picture is
    copied beside it, so a picture is never written without its label.

    Parameters:
      parent_dir: the staged parent dataset folder.
      dest: the tutorial folder, already created.
      label_dir: the parent label folder being read.
      pins: {label stem: split} from select_and_pin.
      images: {stem: image basename} from the subset tool's image_index.
      id_map: {parent class id: tutorial class id}.
      log: where progress lines go.
    Returns:
      {"n_boxes": n, "n_degenerate": n, "n_degenerate_exact": n,
       "by_class": {tutorial class id: box count},
       "images": [image basename]}.
    Raises:
      TutorialError naming the file when a read, a write or a copy fails, or
      when a label file lost every box between the two passes.
    """
    counts = {"n_boxes": 0, "n_degenerate": 0, "n_degenerate_exact": 0,
              "by_class": {}, "images": []}
    for stem in sorted(pins):
        text = sub._read_label(os.path.join(label_dir, stem + LABEL_EXT))
        lines, n_degenerate, n_exact = sub._kept_lines(
            text, id_map, stem + LABEL_EXT)
        if not lines:
            raise TutorialError(
                f"{stem + LABEL_EXT} lost every box between the two passes, "
                "so its picture would be copied with nothing to find. The "
                "parent changed while this run was reading it")
        _write_text(os.path.join(dest, LABELS_DIR, stem + LABEL_EXT),
                    "".join(line + "\n" for line in lines))
        image_name = images[stem]
        source = os.path.join(parent_dir, IMAGES_DIR, image_name)
        target = os.path.join(dest, IMAGES_DIR, image_name)
        try:
            shutil.copy2(source, target)
        except OSError as exc:
            raise TutorialError(
                f"{source} could not be copied to {target}: {exc}")
        counts["images"].append(image_name)
        counts["n_boxes"] += len(lines)
        counts["n_degenerate"] += n_degenerate
        counts["n_degenerate_exact"] += n_exact
        for line in lines:
            class_id = int(line.split()[0])
            counts["by_class"][class_id] = counts["by_class"].get(class_id, 0) + 1
    log(f"wrote {len(counts['images'])} picture and label pair(s), "
        f"{counts['n_boxes']} box(es)")
    return counts


def write_split_pins(dest, pins, images, source):
    """Write split_pins.json keyed by the staged image basename.

    The split builder (TCRMPtrain_oceankindCV/src/detect_dataset.py,
    read_source_pins) keys its pins on the image file name, extension and all,
    because that is what it sees when it walks images/. A pins file keyed on
    the label stem would name nothing and every picture would be shuffled
    again.

    Parameters:
      dest: the tutorial folder.
      pins: {label stem: split} from select_and_pin.
      images: {stem: image basename}.
      source: what the pins came from, recorded in the file.
    Returns:
      {image basename: split} as written.
    Raises:
      TutorialError naming the file when the write fails.
    """
    by_image = {images[stem]: split for stem, split in sorted(pins.items())}
    _write_json(os.path.join(dest, SPLIT_PINS_NAME),
                {"schema_version": ing.SPLIT_PINS_SCHEMA_VERSION,
                 "source": source,
                 "pins": by_image})
    return by_image


def write_class_files(dest, kept, dropped, boxes_here):
    """Write classes.txt, data.yaml, names.csv, class_map.csv and dropped_classes.csv.

    Parameters:
      dest: the tutorial folder.
      kept: [(old_id, new_id, token)] from check_class_choice.
      dropped: [(old_id, token, boxes in the parent)] in parent order.
      boxes_here: {tutorial class id: box count} written by write_pairs.
    Returns:
      None.
    Raises:
      TutorialError naming data.yaml when it cannot be written; the other
      writers raise IngestError naming their own file.
    """
    tokens = [token for _old, _new, token in kept]
    _write_text(os.path.join(dest, CLASSES_TXT_NAME),
                "".join(token + "\n" for token in tokens))
    try:
        coral._write_data_yaml(dest, tokens)
    except OSError as exc:
        raise TutorialError(
            f"{os.path.join(dest, 'data.yaml')} could not be written: {exc}")
    _write_csv(os.path.join(dest, NAMES_CSV_NAME), ing.NAMES_HEADER,
               [(token, full_name(token)) for token in tokens])
    _write_csv(os.path.join(dest, CLASS_MAP_NAME), CLASS_MAP_HEADER,
               [(old_id, new_id, token, boxes_here.get(new_id, 0))
                for old_id, new_id, token in kept])
    _write_csv(os.path.join(dest, DROPPED_NAME), DROPPED_HEADER, dropped)


def full_name(token):
    """The binomial behind one class token.

    Parameters:
      token: a class token, for example "Acropora-palmata".
    Returns:
      the full name from FULL_NAMES, or the token with its hyphens turned
      into spaces, which is the right answer for every token in the parent
      this tool reads.

    Example: full_name("Porites-astreoides") gives "Porites astreoides".
    """
    text = str(token).strip()
    return FULL_NAMES.get(text, text.replace("-", " "))


# ── The README ───────────────────────────────────────────────────────────

def render_readme(facts):
    """The tutorial dataset's README.md, in plain words for a reader who was not here.

    Parameters:
      facts: the dict _facts assembled, naming the dataset, its parent, its
        classes, its counts, its splits and the timestamp it was made.
    Returns:
      the README text, ending in a newline.
    """
    rows = "\n".join(
        f"| *{full_name(token)}* | `{token}` | "
        f"{facts['boxes_by_new'].get(new_id, 0)} | {facts['quotas']['train']} "
        f"| {facts['quotas']['valid']} | {facts['quotas']['test']} |"
        for _old, new_id, token in facts["kept"])
    parts = [
        f"# {facts['name']}",
        "",
        "The small bounding-box dataset the VEGA guided tutorial trains on.",
        "",
        f"- Made: {facts['made_at'][:16]} AST by LO",
        "- Tool: `scripts/_tutorial/build_tutorial_set.py`",
        f"- Parent dataset: `{facts['parent_name']}`, at `{facts['parent_dir']}`",
        f"- Parent labels read from: `{facts['label_rel']}/`",
        f"- Shuffle seed: `{facts['seed']}`",
        "",
        "## Read only",
        "",
        _fill(
            "This folder is a fixture. Nobody edits it, nobody annotates into "
            "it and nobody trains into it. A student's tutorial run makes its "
            "own training project from this dataset and writes everything it "
            "produces inside that project, so the next student starts from "
            "exactly the same pictures, the same boxes and the same held-out "
            "split. Rebuild it only with the tool named above, over a folder "
            "that is not there yet."),
        "",
        "## What this dataset is",
        "",
        _fill(
            "A staged bounding-box dataset in the layout every other staged "
            "dataset uses: images/, labels/, classes.txt, data.yaml, "
            "names.csv, split_pins.json and ingest_manifest.json. It holds "
            f"{_word(facts['n_classes'])} coral species, {facts['per_class']} "
            f"pictures of each, {facts['n_images']} pictures and "
            f"{facts['n_boxes']} boxes in all. Nothing was re-annotated and "
            "no box moved: every box here is a box of "
            f"`{facts['parent_name']}`, under a new class id. The pictures "
            f"are {facts['image_note']}."),
        "",
        _fill(
            "It is this small on purpose. The tutorial teaches the cycle of "
            "training a model, scoring it on pictures it never saw and "
            "running it on new ones, and a student learns that cycle no "
            "better from eight thousand pictures than from a few hundred, "
            "while waiting hours instead of minutes for it."),
        "",
        f"## The {_word(facts['n_classes'])} species",
        "",
        _fill(facts["species_sentence"]),
        "",
        _fill(
            "They were chosen because a student can tell them apart by eye, "
            "which is what makes a short training run's result readable "
            "rather than mysterious. The parent's confusable pairs were left "
            "out on purpose: the two *Orbicella* species, the two "
            "*Siderastrea* species, the three *Agaricia* species, and "
            "staghorn coral beside elkhorn coral."),
        "",
        "| Species | Class token | Boxes | Train pictures | Valid pictures "
        "| Test pictures |",
        "|---|---|---|---|---|---|",
        rows,
        "",
        _fill(
            f"Every species carries the same {facts['per_class']} pictures, "
            "split the same way, so a score is not read through a class "
            "imbalance the student has not met yet. Real survey data is not "
            "balanced, and the tutorial says so where it matters."),
        "",
        "## The split is pinned, not drawn",
        "",
        _fill(
            "split_pins.json names the split of every picture: "
            f"{facts['split_totals']['train']} train, "
            f"{facts['split_totals']['valid']} valid and "
            f"{facts['split_totals']['test']} test. VEGA's split builder "
            "replays those pins instead of shuffling, so every student holds "
            "out the same pictures and two students' scores mean the same "
            f"thing. All {_word(facts['n_classes'])} species appear in all three "
            "splits, which is what gives the evaluation report a row per "
            "species for the student to read."),
        "",
        _fill(
            "The held-out pictures stay held out. A model that has seen a "
            "picture in training will find its coral again, so a score "
            "measured on those pictures says how well the model remembers, "
            "not how well it works. That is the lesson the test split exists "
            "to teach, and it holds only while the split stays where it is."),
        "",
        "## Class ids were renumbered",
        "",
        _fill(
            "Ultralytics refuses a label file carrying a class id at or "
            "above the number of classes the dataset declares, so the "
            f"{_word(facts['n_classes'])} kept classes are numbered again "
            "from 0, in the order this tool was given them, and every kept "
            "label line was rewritten with its new id. class_map.csv holds "
            "the mapping: one row per class with its id in "
            f"`{facts['parent_name']}` (old_id), its id here (new_id), its "
            "token and how many boxes of it this dataset carries. "
            f"dropped_classes.csv names all {facts['n_dropped']} parent "
            "classes that are not here and how many boxes each had in the "
            "parent."),
        "",
        "## What a score measured here means",
        "",
        _fill(
            f"Nothing outside this folder. {_word(facts['n_classes']).capitalize()} "
            "species a student can tell apart, "
            f"{facts['quotas']['test']} test pictures of each, and boxes that "
            "cover much of the frame together make a number that looks better "
            "than the same model would look on a real survey frame. Quote a "
            "score from this dataset only beside another score from this same "
            "dataset, and never beside one measured on "
            f"`{facts['parent_name']}` or on TCRMP imagery."),
        "",
        "## Coverage",
        "",
        _fill(
            "Outside the TCRMP survey. This dataset never goes to a run "
            "folder under inprocess/, the mask registry, label provenance, "
            "the coverage matrix or STATS. It is read where it is staged."),
        "",
        "## License and citation",
        "",
        _fill(f"Carried from `{facts['parent_name']}` and unchanged. The full "
              "text is in LICENSE.md beside this file."),
        "",
    ]
    if facts["license_note"]:
        parts += [_fill(facts["license_note"]), ""]
    if facts["citation"]:
        parts += [_fill("Citation: " + facts["citation"]), ""]
    parts += [
        "## What each file holds",
        "",
        "- `images/`: the chosen pictures, copied from the parent unchanged.",
        '- `labels/`: one YOLO text file per picture, lines "class cx cy w h",',
        "  with the new class ids.",
        "- `classes.txt`: the class tokens, one per line, in new-id order.",
        "- `data.yaml`: the Ultralytics dataset config, names keyed by the new ids.",
        "- `names.csv`: token,name for each class, so VEGA shows the binomial",
        "  rather than the hyphenated token.",
        "- `split_pins.json`: the train, valid or test split of every picture,",
        "  keyed by the image file name.",
        "- `class_map.csv`: old_id, new_id, name, boxes for every class kept.",
        "- `dropped_classes.csv`: old_id, name, boxes for every parent class",
        "  that is not here.",
        "- `LICENSE.md`: the parent's terms, carried verbatim.",
        "- `ingest_manifest.json`: the counts above, the class list, the parent",
        "  this came from, the seed and the AST timestamp it was made.",
        "",
        _fill("Basenames are the parent's, unchanged, so a picture here is "
              f"the picture of the same name in `{facts['parent_name']}`."),
        "",
        "Lauren Olinger, lauren.olinger@uvi.edu",
    ]
    return "\n".join(parts) + "\n"


def species_sentence(kept):
    """One sentence describing what each kept species looks like.

    Parameters:
      kept: [(old_id, new_id, token)] from check_class_choice.
    Returns:
      a sentence naming each species in italic with its common name and its
      growth form, for a class with a row in COMMON_NAMES, and naming the
      binomial alone for one without.
    """
    shapes = {
        "Acropora-palmata": "grows as broad flattened branches",
        "Dendrogyra-cylindrus": "grows as upright columns with its polyps out "
                                "by day",
        "Diploria-labyrinthiformis": "grows as a boulder cut by winding "
                                     "valleys",
        "Eusmilia-fastigiata": "grows as clusters of separate fleshy cups",
    }
    clauses = []
    for _old, _new, token in kept:
        common = COMMON_NAMES.get(token, "")
        shape = shapes.get(token, "")
        lead = f"*{full_name(token)}*"
        if common:
            lead += f" ({common})"
        clauses.append(f"{lead} {shape}" if shape else lead)
    if len(clauses) == 1:
        return clauses[0] + "."
    return ", ".join(clauses[:-1]) + ", and " + clauses[-1] + "."


def _word(count):
    """A small whole number in words, for prose that would otherwise open with a digit.

    Parameters:
      count: a whole number.
    Returns:
      the English word for two through ten, or the number as text.
    """
    words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
             7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    return words.get(count, str(count))


def _fill(text):
    """Wrap one paragraph to README_WIDTH.

    Parameters:
      text: the paragraph, already assembled with its numbers in it.
    Returns:
      the wrapped paragraph with no trailing newline.
    """
    return textwrap.fill(text, width=README_WIDTH)


# ── The manifest ─────────────────────────────────────────────────────────

def build_manifest(*, parent_manifest, facts, parent_dir, label_rel, counts,
                   pins_by_image, subject):
    """Assemble the tutorial dataset's ingest_manifest.json.

    Parameters:
      parent_manifest: the parent's manifest dict.
      facts: the dict _facts assembled.
      parent_dir: the staged parent dataset folder.
      label_rel: which parent label folder was read.
      counts: what write_pairs returned.
      pins_by_image: {image basename: split} as written.
      subject: what the dataset is about, carried from the parent.
    Returns:
      the manifest dict, in the shape the Bounding Boxes page and the split
      builder read.
    """
    by_split = {split: sum(1 for value in pins_by_image.values()
                           if value == split)
                for split in SPLIT_NAMES}
    return {
        "dataset": facts["name"],
        "n_images": facts["n_images"],
        "n_boxes": counts["n_boxes"],
        "n_degenerate": counts["n_degenerate"],
        "n_degenerate_exact": counts["n_degenerate_exact"],
        "n_classes": facts["n_classes"],
        "classes": [token for _old, _new, token in facts["kept"]],
        "source_path": os.path.abspath(parent_dir),
        "ingested_at": facts["made_at"],
        "limit": None,
        "citation": facts["citation"],
        "license_note": facts["license_note"],
        "coverage_note": ing.COVERAGE_NOTE,
        "subject": subject,
        "source_kind": SOURCE_KIND,
        "labels_source": os.path.join(os.path.abspath(parent_dir), label_rel),
        "names_csv": NAMES_CSV_NAME,
        "split_pins": True,
        "n_pinned": len(pins_by_image),
        "pins_by_split": by_split,
        "n_renamed": 0,
        "n_case_disambiguated": 0,
        "n_background": 0,
        "n_duplicate_lines": 0,
        "parent_dataset": facts["parent_name"],
        "parent_source_path": str(parent_manifest.get("source_path") or ""),
        "label_source": label_rel,
        "purpose": "vega_tutorial",
        "read_only": True,
        "images_per_class": facts["per_class"],
        "images_per_class_by_split": facts["quotas"],
        "split_seed": facts["seed"],
        "n_classes_parent": len(facts["parent_classes"]),
        "n_classes_dropped": facts["n_dropped"],
        "n_images_parent": int(parent_manifest.get("n_images") or 0),
        "n_images_mixed_class_skipped": facts["n_mixed"],
    }


def _facts(*, name, parent_dir, parent_name, parent_classes, label_rel, kept,
           dropped, per_class, quotas, seed, counts, pins, made_at, citation,
           license_note, n_mixed, image_note):
    """Everything the README and the manifest both need, assembled once.

    Parameters: the build's own values, each already checked.
    Returns:
      a dict of plain values, so the README renderer and the manifest builder
      read one set of numbers rather than each counting for itself.
    """
    split_totals = {split: sum(1 for value in pins.values() if value == split)
                    for split in SPLIT_NAMES}
    return {
        "name": name,
        "parent_dir": os.path.abspath(parent_dir),
        "parent_name": parent_name,
        "parent_classes": list(parent_classes),
        "label_rel": label_rel,
        "kept": list(kept),
        "n_classes": len(kept),
        "n_dropped": len(dropped),
        "per_class": per_class,
        "quotas": dict(quotas),
        "seed": seed,
        "n_images": len(pins),
        "n_boxes": counts["n_boxes"],
        "boxes_by_new": dict(counts["by_class"]),
        "split_totals": split_totals,
        "made_at": made_at,
        "citation": citation,
        "license_note": license_note,
        "n_mixed": n_mixed,
        "image_note": image_note,
        "species_sentence": species_sentence(kept),
    }


# ── The build ────────────────────────────────────────────────────────────

def build_tutorial_set(parent_dir, dest, *, classes=TUTORIAL_CLASSES,
                       per_class=IMAGES_PER_CLASS, train=TRAIN_PER_CLASS,
                       valid=VALID_PER_CLASS, test=TEST_PER_CLASS,
                       seed=DEFAULT_SEED, log=print):
    """Write the tutorial dataset and return the manifest it wrote.

    Nothing is written until every check has passed, so a refusal leaves no
    half-built folder for the Bounding Boxes page to list.

    Parameters:
      parent_dir: a staged bounding-box dataset folder.
      dest: the tutorial folder to write. Its holding folder must exist and
        the folder itself must not.
      classes: the class tokens to keep, in the order they are numbered here.
      per_class: how many pictures of each class to take.
      train, valid, test: how many of those go to each split.
      seed: what the per-class shuffle is seeded on.
      log: where progress lines go.
    Returns:
      the manifest dict written to ingest_manifest.json.
    Raises:
      TutorialError, or the class-count subset tool's SubsetError and the
      stagers' IngestError which it subclasses, naming what failed and why.
    """
    parent_dir = os.path.abspath(os.path.expanduser(str(parent_dir)))
    dest = os.path.abspath(os.path.expanduser(str(dest)))
    name = os.path.basename(dest.rstrip(os.sep))
    ing.check_dataset_name(name)

    parent_manifest = sub.read_manifest(parent_dir)
    parent_classes = sub.parent_class_list(parent_dir)
    subject = sub.parent_subject(parent_manifest, parent_dir)
    kept = check_class_choice(parent_classes, classes, parent_dir)
    quotas = check_quotas(per_class, train, valid, test)
    license_source, license_text = sub.resolve_license(parent_dir,
                                                       parent_manifest)
    label_rel, label_dir, files = sub.choose_label_dir(parent_dir)
    images = sub.image_index(parent_dir)
    sub.check_every_label_has_an_image(files, images, label_dir)
    sub.check_case_unique(list(images.values()),
                          os.path.join(parent_dir, IMAGES_DIR))
    sub.check_dest_holder(dest)
    sub.check_output_free(dest)
    log(f"reading {len(files)} label file(s) in {label_rel}/ of {parent_dir}")

    kept_ids = [old_id for old_id, _new, _token in kept]
    eligible, boxes_by_class, n_mixed = scan_parent_labels(
        label_dir, files, len(parent_classes), kept_ids)
    check_enough_images(eligible, per_class, kept, parent_dir)
    pins, _primary = select_and_pin(eligible, kept, per_class, quotas, seed)
    log(f"chose {len(pins)} picture(s): {per_class} of each of {len(kept)} "
        f"class(es); {n_mixed} picture(s) carrying a kept class beside a "
        "dropped one were passed over")
    log("Step 1 complete: the parent was read and the pictures were chosen")

    id_map = {old_id: new_id for old_id, new_id, _token in kept}
    dropped = [(old_id, token, boxes_by_class.get(old_id, 0))
               for old_id, token in enumerate(parent_classes)
               if old_id not in id_map]

    sub._make_dirs(dest)
    counts = write_pairs(parent_dir, dest, label_dir, pins, images, id_map, log)
    pins_by_image = write_split_pins(
        dest, pins, images,
        f"{parent_dir} (pinned by build_tutorial_set.py, seed {seed})")
    write_class_files(dest, kept, dropped, counts["by_class"])
    sub.write_license(dest, license_source, license_text)
    log("Step 2 complete: pictures, labels, class files and split pins written")

    facts = _facts(
        name=name, parent_dir=parent_dir,
        parent_name=os.path.basename(parent_dir.rstrip(os.sep)),
        parent_classes=parent_classes, label_rel=label_rel, kept=kept,
        dropped=dropped, per_class=per_class, quotas=quotas, seed=seed,
        counts=counts, pins=pins, made_at=ing.ast_now(),
        citation=str(parent_manifest.get("citation") or ""),
        license_note=str(parent_manifest.get("license_note") or ""),
        n_mixed=n_mixed, image_note=_image_note(parent_dir, counts["images"]))
    _write_text(os.path.join(dest, README_NAME), render_readme(facts))
    manifest = build_manifest(
        parent_manifest=parent_manifest, facts=facts, parent_dir=parent_dir,
        label_rel=label_rel, counts=counts, pins_by_image=pins_by_image,
        subject=subject)
    _write_json(os.path.join(dest, MANIFEST_NAME), manifest)
    log("Step 3 complete: the README and the manifest were written")

    verify(dest, quotas, log)
    log(f"staged {manifest['n_images']} pictures, {manifest['n_boxes']} "
        f"boxes, {manifest['n_classes']} classes -> {dest}")
    return manifest


def _image_note(parent_dir, image_names):
    """One clause saying what the pictures are, for the README.

    Parameters:
      parent_dir: the staged parent dataset folder.
      image_names: the image basenames this run copied.
    Returns:
      a clause naming the pixel size when every picture shares one, and a
      plainer clause when they differ or cannot be read.
    """
    plain = "the parent's pictures, copied unchanged"
    if Image is None or not image_names:
        return plain
    sizes = set()
    for name in image_names:
        try:
            with Image.open(os.path.join(parent_dir, IMAGES_DIR, name)) as im:
                sizes.add(im.size)
        except OSError:
            return plain
        if len(sizes) > 1:
            return plain + ", at mixed sizes"
    width, height = sizes.pop()
    return f"{plain}, every one of them {width} by {height} pixels"


# ── Verification ─────────────────────────────────────────────────────────

def verify(dest, quotas, log):
    """Read the written folder back and check every invariant the tutorial needs.

    Read off disk rather than off the build's own bookkeeping, so this proves
    what a student's training run will find rather than what this process
    believed it wrote.

    Parameters:
      dest: the tutorial folder just written.
      quotas: {"train": n, "valid": n, "test": n}.
      log: where the counts go.
    Returns:
      a dict of the counts it read: n_images, n_labels, n_boxes, by_class,
      by_split and by_class_split.
    Raises:
      TutorialError naming the disagreement when the folder does not hold what
      the manifest says, when a picture has no label or a label no picture,
      when a class is missing from a split, or when a split count is wrong.
    """
    manifest = _read_json(os.path.join(dest, MANIFEST_NAME))
    classes = [line.strip() for line
               in open(os.path.join(dest, CLASSES_TXT_NAME),
                       encoding="utf-8").read().splitlines() if line.strip()]
    images = sorted(name for name in os.listdir(os.path.join(dest, IMAGES_DIR))
                    if os.path.splitext(name)[1].lower() in ing.IMAGE_EXTS)
    labels = sorted(name for name in os.listdir(os.path.join(dest, LABELS_DIR))
                    if name.endswith(LABEL_EXT))
    _check_pairing(dest, images, labels, manifest)
    pins = _read_json(os.path.join(dest, SPLIT_PINS_NAME))["pins"]
    _check_pins_cover(dest, images, pins)
    counts = _count_written(dest, labels, len(classes), pins)
    _check_every_class_in_every_split(dest, classes, counts["by_class_split"],
                                      quotas)
    if counts["n_boxes"] != manifest["n_boxes"]:
        raise TutorialError(
            f"{dest} holds {counts['n_boxes']} box(es) and the manifest says "
            f"{manifest['n_boxes']}")
    log(f"verified {len(images)} picture(s) beside {len(labels)} label "
        f"file(s), {counts['n_boxes']} box(es), train "
        f"{counts['by_split']['train']}, valid {counts['by_split']['valid']}, "
        f"test {counts['by_split']['test']}")
    log("Step 4 complete: the folder on disk holds what the manifest claims")
    return counts


def _read_json(path):
    """One JSON document this tool just wrote.

    Parameters:
      path: the file to read.
    Returns:
      the parsed document.
    Raises:
      TutorialError naming the file when it cannot be read or parsed.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise TutorialError(f"{path} could not be read back: {exc}")


def _check_pairing(dest, images, labels, manifest):
    """Refuse a folder where a picture has no label or a label no picture.

    Parameters:
      dest: the tutorial folder, quoted back in the refusal.
      images: the image basenames written.
      labels: the label basenames written.
      manifest: the manifest written beside them.
    Returns:
      None.
    Raises:
      TutorialError naming the count and the first few lonely names.
    """
    if len(images) != manifest["n_images"] or len(labels) != manifest["n_images"]:
        raise TutorialError(
            f"{dest} holds {len(images)} picture(s) and {len(labels)} label "
            f"file(s) and the manifest says {manifest['n_images']} of each")
    lonely = ({os.path.splitext(name)[0] for name in images}
              ^ {os.path.splitext(name)[0] for name in labels})
    if lonely:
        raise TutorialError(
            f"{dest}: {len(lonely)} name(s) have a picture without a label or "
            f"a label without a picture: "
            + ", ".join(sorted(lonely)[:REFUSAL_EXAMPLES]))


def _check_pins_cover(dest, images, pins):
    """Refuse a pins file that misses a picture or names one that is not there.

    Parameters:
      dest: the tutorial folder, quoted back in the refusal.
      images: the image basenames written.
      pins: {image basename: split} read back from split_pins.json.
    Returns:
      None.
    Raises:
      TutorialError naming what is unpinned, what is pinned and absent, or
      what carries a split the split builder does not know.
    """
    unpinned = sorted(set(images) - set(pins))
    if unpinned:
        raise TutorialError(
            f"{dest}: {len(unpinned)} picture(s) carry no split pin, so the "
            "split builder would shuffle them and the holdout would move "
            "between students: "
            + ", ".join(unpinned[:REFUSAL_EXAMPLES]))
    absent = sorted(set(pins) - set(images))
    if absent:
        raise TutorialError(
            f"{dest}: {len(absent)} pin(s) name a picture that is not in "
            "images/: " + ", ".join(absent[:REFUSAL_EXAMPLES]))
    unknown = sorted(name for name, split in pins.items()
                     if split not in SPLIT_NAMES)
    if unknown:
        raise TutorialError(
            f"{dest}: {len(unknown)} pin(s) name a split outside "
            f"{SPLIT_NAMES}: " + ", ".join(unknown[:REFUSAL_EXAMPLES]))


def _count_written(dest, labels, n_classes, pins):
    """Count what the written labels hold, by class and by split.

    Parameters:
      dest: the tutorial folder.
      labels: the label basenames written.
      n_classes: how many classes the folder declares.
      pins: {image basename: split} read back from split_pins.json.
    Returns:
      {"n_boxes": n, "by_class": {class id: boxes},
       "by_split": {split: pictures},
       "by_class_split": {(class id, split): pictures}}.
    Raises:
      TutorialError naming the file and the line when a written label line is
      not a box, or when a written label file holds no box at all.
    """
    stem_split = {os.path.splitext(name)[0]: split
                  for name, split in pins.items()}
    counts = {"n_boxes": 0, "by_class": {}, "by_split": {s: 0 for s in SPLIT_NAMES},
              "by_class_split": {}}
    for name in labels:
        stem = name[:-len(LABEL_EXT)]
        text = sub._read_label(os.path.join(dest, LABELS_DIR, name))
        boxes = ing.parse_boxes(text, n_classes, name)
        if not boxes:
            raise TutorialError(
                f"{os.path.join(dest, LABELS_DIR, name)} holds no box, so its "
                "picture would train the detector on an empty scene")
        split = stem_split[stem]
        counts["by_split"][split] += 1
        counts["n_boxes"] += len(boxes)
        seen = set()
        for class_id, _coords in boxes:
            counts["by_class"][class_id] = counts["by_class"].get(class_id, 0) + 1
            seen.add(class_id)
        for class_id in seen:
            key = (class_id, split)
            counts["by_class_split"][key] = counts["by_class_split"].get(key, 0) + 1
    return counts


def _check_every_class_in_every_split(dest, classes, by_class_split, quotas):
    """Refuse a class that is missing from a split, or short in one.

    A class absent from the test split has no row in the evaluation report,
    and the tutorial's evaluation lesson reads that report.

    Parameters:
      dest: the tutorial folder, quoted back in the refusal.
      classes: the class tokens the folder declares, in class-id order.
      by_class_split: {(class id, split): pictures} from _count_written.
      quotas: {"train": n, "valid": n, "test": n}.
    Returns:
      None.
    Raises:
      TutorialError naming every class and split that does not hold its quota.
    """
    wrong = []
    for class_id, token in enumerate(classes):
        for split in SPLIT_NAMES:
            got = by_class_split.get((class_id, split), 0)
            if got != quotas[split]:
                wrong.append(f"{token} has {got} picture(s) in the {split} "
                             f"split and {quotas[split]} were asked for")
    if wrong:
        raise TutorialError(
            f"{dest}: the split does not carry every class evenly, so the "
            "evaluation report would be missing a row: " + "; ".join(wrong))


# ── Command line ─────────────────────────────────────────────────────────

def build_parser():
    """The command-line parser, split out so the flags can be read in one place.

    Returns:
      an argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Build the small bounding-box dataset the VEGA guided "
                    "tutorial trains on.")
    parser.add_argument(
        "--parent", default=DEFAULT_PARENT,
        help="the staged bounding-box dataset to take pictures from "
             f"(default: {DEFAULT_PARENT})")
    parser.add_argument(
        "--name", default=DEFAULT_NAME,
        help=f"the staged folder to write (default: {DEFAULT_NAME})")
    parser.add_argument(
        "--dest-root", default=ing.DEFAULT_DEST_ROOT,
        help=f"where staged datasets live (default: {ing.DEFAULT_DEST_ROOT})")
    parser.add_argument(
        "--classes", nargs="+", default=list(TUTORIAL_CLASSES),
        help="the class tokens to keep, in the order they are numbered here "
             f"(default: {' '.join(TUTORIAL_CLASSES)})")
    parser.add_argument(
        "--per-class", type=int, default=IMAGES_PER_CLASS,
        help="how many pictures of each class to take (default: "
             f"{IMAGES_PER_CLASS})")
    parser.add_argument(
        "--train-per-class", type=int, default=TRAIN_PER_CLASS,
        help=f"how many of those train (default: {TRAIN_PER_CLASS})")
    parser.add_argument(
        "--valid-per-class", type=int, default=VALID_PER_CLASS,
        help=f"how many of those validate (default: {VALID_PER_CLASS})")
    parser.add_argument(
        "--test-per-class", type=int, default=TEST_PER_CLASS,
        help="how many of those are held out for the test split (default: "
             f"{TEST_PER_CLASS})")
    parser.add_argument(
        "--seed", default=DEFAULT_SEED,
        help=f"what the per-class shuffle is seeded on (default: {DEFAULT_SEED})")
    parser.add_argument("--quiet", action="store_true",
                        help="print only the refusals and the final line")
    return parser


def main(argv=None):
    """Parse, validate, build, and report.

    Parameters:
      argv: the command line, or None for sys.argv[1:].
    Returns:
      0 when the dataset was written, 1 when the build was refused.
    """
    args = build_parser().parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else print
    try:
        root = sub.ensure_dest_root(args.dest_root)
        ing.check_dataset_name(args.name)
        manifest = build_tutorial_set(
            args.parent, os.path.join(root, args.name),
            classes=args.classes, per_class=args.per_class,
            train=args.train_per_class, valid=args.valid_per_class,
            test=args.test_per_class, seed=args.seed, log=log)
    except ing.IngestError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(f"{manifest['dataset']}: {manifest['n_images']} pictures, "
          f"{manifest['n_boxes']} boxes, {manifest['n_classes']} classes, "
          f"pinned train {manifest['pins_by_split']['train']}, valid "
          f"{manifest['pins_by_split']['valid']}, test "
          f"{manifest['pins_by_split']['test']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

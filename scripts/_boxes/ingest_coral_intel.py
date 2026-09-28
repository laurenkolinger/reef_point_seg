#!/usr/bin/env python
"""Stage the Coral Intel object-detection dataset into the module.

Source (a collaborator distribution, read only):

    <Dropbox>/Coral Intel Object Detection Dataset/Coral Intel Object Detection Dataset/
        images/       one jpg per record
        annotations/  one YOLO detection txt per record, lines "class cx cy w h"
        LICENSE.md

Staged copy (what this script writes):

    supporting_data/bbox_datasets/coral_intel/
        images/             verbatim image copies
        labels/             verbatim annotation copies
        labels_localized/   the same labels minus whole-frame boxes; a record
                            whose every box is whole-frame keeps an empty file
        classes.txt         one species name per line, alphabetical
        data.yaml           path, names, nc
        LICENSE.md          copied verbatim
        ingest_manifest.json counts, class list, source path, AST timestamp

The class map is derived, not trusted: the species is read off each filename
(Genus-species_id.txt), the sorted set of species names becomes the class list,
and every class id in every annotation file must equal its species' index in
that list. A filename that does not carry a hyphenated species name, or an id
that disagrees, aborts the run before anything is staged.

Licensing, carried into the staged copy and the manifest: the annotations are
CC BY 4.0 and require citing Cordero et al. 2026 (doi 10.1007/s44295-026-00107-5).
The images are third-party photographs (GBIF, CoralNet, Corals of the World),
excluded from that grant. They are for local training only: never re-serve them
publicly and never bundle them into a shared export.

This imagery is outside the TCRMP survey. Nothing here writes to inprocess/run_*,
the mask registry, label provenance, the coverage matrix, or STATS.

Usage:
  env/bin/python scripts/_boxes/ingest_coral_intel.py
  env/bin/python scripts/_boxes/ingest_coral_intel.py --limit 50
  env/bin/python scripts/_boxes/ingest_coral_intel.py --source <dir> --dest <dir>
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone

# Atlantic Standard Time, fixed UTC-4, no daylight saving (house rule: every
# timestamp this platform writes is AST 24h).
AST = timezone(timedelta(hours=-4))

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
REPO_DIR = os.path.dirname(_SCRIPTS)

# Where the collaborator distribution sits on this machine. Overridable so a
# second box, or a test, never has to have that path.
DEFAULT_SOURCE = os.environ.get("TCRMP_CORAL_INTEL_SOURCE", "") or os.path.join(
    os.path.expanduser("~"),
    "UVI Dropbox", "Lauren Olinger",
    "Coral Intel Object Detection Dataset",
    "Coral Intel Object Detection Dataset",
)

DATASET_NAME = "coral_intel"


def default_dest(repo_dir=REPO_DIR, name=DATASET_NAME):
    return os.path.join(repo_dir, "supporting_data", "bbox_datasets", name)


# Genus-species_id: at least one hyphen in the species name, then an underscore
# and the record id. Anything else is a filename this script does not understand.
SPECIES_RE = re.compile(r"^([A-Za-z]+(?:-[A-Za-z]+)+)_([^_]+)$")

# A box that covers the whole frame carries no localisation. The distribution
# writes those as "0.5 0.5 1 1"; a handful are written with slightly different
# centres or rounded extents, so the rule is geometric, not textual.
FULL_FRAME_MIN = 0.999

# The literal written form, counted separately so the staged manifest can be
# checked against the number quoted for this distribution.
EXACT_DEGENERATE = ("0.5", "0.5", "1", "1")

LICENSE_NOTE = (
    "Annotations are CC BY 4.0 and require citing Cordero, E., Sherman, C. E., "
    "Yadav, P., Raju, V. K., Townsend, J. E., and Courtney, T. A. (2026). "
    "Coral Intel: a YOLOv8 deep learning framework for monitoring Caribbean "
    "corals. Intelligent Marine Technology and Systems, 4, Article 18. "
    "https://doi.org/10.1007/s44295-026-00107-5 . The images are third-party "
    "photographs (GBIF, CoralNet, Corals of the World) and are excluded from "
    "that grant: use them for local training only, never re-serve them "
    "publicly and never bundle them into a shared export."
)

CITATION = (
    "Cordero, E., Sherman, C. E., Yadav, P., Raju, V. K., Townsend, J. E., and "
    "Courtney, T. A. (2026). Coral Intel: a YOLOv8 deep learning framework for "
    "monitoring Caribbean corals. Intelligent Marine Technology and Systems, 4, "
    "Article 18. https://doi.org/10.1007/s44295-026-00107-5"
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


class IngestError(RuntimeError):
    """Anything that stops the ingest before it stages a file."""


def ast_now():
    return datetime.now(AST).strftime("%Y-%m-%d %H:%M:%S")


# ── Parsing ──────────────────────────────────────────────────────────────

def species_of(stem):
    """The hyphenated species name in a record stem, else '' when the stem
    does not follow Genus-species_id."""
    m = SPECIES_RE.match(stem)
    return m.group(1) if m else ""


def parse_boxes(text):
    """[(class_id, [cx, cy, w, h], raw_fields)] for every non-empty line.

    Line-based on purpose: many files in this distribution have no trailing
    newline, so counting boxes by lines read is the only count that matches.
    """
    out = []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 5:
            raise ValueError(f"expected 'class cx cy w h', got {line.strip()!r}")
        try:
            cls = int(fields[0])
            coords = [float(v) for v in fields[1:5]]
        except ValueError:
            raise ValueError(f"expected 'class cx cy w h', got {line.strip()!r}")
        out.append((cls, coords, fields))
    return out


def is_degenerate(coords):
    """True when a box covers the whole frame, so it localises nothing."""
    _, _, w, h = coords
    return w >= FULL_FRAME_MIN and h >= FULL_FRAME_MIN


def is_exact_degenerate_line(line):
    """True for the literal '<class> 0.5 0.5 1 1' form this distribution uses."""
    fields = line.split()
    return len(fields) >= 5 and tuple(fields[1:5]) == EXACT_DEGENERATE


# ── Class map ────────────────────────────────────────────────────────────

def derive_classes(stems):
    """The alphabetical species list behind a set of record stems. Raises
    IngestError naming every stem that does not carry a species name."""
    bad = [s for s in stems if not species_of(s)]
    if bad:
        raise IngestError(
            "these annotation filenames do not follow Genus-species_id.txt, so "
            "the class map cannot be derived: "
            + ", ".join(sorted(b + ".txt" for b in bad)[:5])
            + (f" (and {len(bad) - 5} more)" if len(bad) > 5 else "")
        )
    return sorted({species_of(s) for s in stems})


def verify_class_ids(ann_dir, stems, classes):
    """Every class id in every annotation file must equal its species' index in
    ``classes``. Raises IngestError listing the first disagreements."""
    index = {name: i for i, name in enumerate(classes)}
    problems = []
    for stem in stems:
        expect = index[species_of(stem)]
        path = os.path.join(ann_dir, stem + ".txt")
        try:
            with open(path) as fh:
                boxes = parse_boxes(fh.read())
        except ValueError as e:
            problems.append(f"{stem}.txt: {e}")
            continue
        for cls, _coords, _fields in boxes:
            if cls != expect:
                problems.append(
                    f"{stem}.txt: class id {cls} but {species_of(stem)} is "
                    f"class {expect} in alphabetical order")
                break
        if len(problems) >= 8:
            break
    if problems:
        raise IngestError(
            "the class ids in this distribution disagree with the alphabetical "
            "species order, so the derived class map would mislabel the data:\n  "
            + "\n  ".join(problems))


# ── Staging ──────────────────────────────────────────────────────────────

def _copy_if_changed(src, dst):
    """Copy only when the destination is missing or differs in size or mtime,
    so a re-run over 10k files is cheap and idempotent."""
    try:
        s, d = os.stat(src), os.stat(dst)
        if s.st_size == d.st_size and int(s.st_mtime) == int(d.st_mtime):
            return False
    except OSError:
        pass
    shutil.copy2(src, dst)
    return True


def _staged_count(dest):
    """How many records are already staged in ``dest``."""
    try:
        return sum(1 for fn in os.listdir(os.path.join(dest, "labels"))
                   if fn.endswith(".txt"))
    except OSError:
        return 0


def _write_data_yaml(dest, classes):
    lines = ["# YOLO dataset config for a staged bounding-box dataset.",
             "path: .",
             "names:"]
    lines += [f"  {i}: {name}" for i, name in enumerate(classes)]
    lines.append(f"nc: {len(classes)}")
    with open(os.path.join(dest, "data.yaml"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _image_for(img_dir, stem):
    for ext in IMAGE_EXTS:
        cand = os.path.join(img_dir, stem + ext)
        if os.path.isfile(cand):
            return cand
    return ""


def ingest(source, dest, limit=None, log=print):
    """Stage ``source`` into ``dest`` and return the manifest it wrote.

    Idempotent: staging the same source twice copies nothing new and rewrites
    the same manifest. Nothing is staged until the class map verifies.
    """
    source = os.path.abspath(os.path.expanduser(str(source)))
    dest = os.path.abspath(os.path.expanduser(str(dest)))
    img_dir = os.path.join(source, "images")
    ann_dir = os.path.join(source, "annotations")
    if not os.path.isdir(img_dir):
        raise IngestError(f"source has no images/ folder: {img_dir}")
    if not os.path.isdir(ann_dir):
        raise IngestError(f"source has no annotations/ folder: {ann_dir}")

    stems = sorted(fn[:-4] for fn in os.listdir(ann_dir) if fn.endswith(".txt"))
    if not stems:
        raise IngestError(f"no annotation files in {ann_dir}")

    # The class map always comes from EVERY annotation file, so a limited
    # ingest carries the same class ids as a full one.
    classes = derive_classes(stems)
    verify_class_ids(ann_dir, stems, classes)
    log(f"class map verified: {len(classes)} species, "
        f"{classes[0]} .. {classes[-1]}")

    staged = stems if limit in (None, 0) else stems[:int(limit)]
    if limit not in (None, 0):
        already = _staged_count(dest)
        if already > len(staged):
            raise IngestError(
                f"{dest} already holds {already} staged records, more than the "
                f"{len(staged)} this run would write. The older files would stay "
                "where they are and the manifest would undercount them. Run "
                "without --limit, or stage into a different folder.")
    missing = [s for s in staged if not _image_for(img_dir, s)]
    if missing:
        raise IngestError(
            "these records have an annotation but no image: "
            + ", ".join(missing[:5])
            + (f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""))

    for sub in ("images", "labels", "labels_localized"):
        os.makedirs(os.path.join(dest, sub), exist_ok=True)

    n_boxes = n_degenerate = n_exact = 0
    copied = 0
    for i, stem in enumerate(staged, 1):
        src_img = _image_for(img_dir, stem)
        if _copy_if_changed(src_img, os.path.join(dest, "images",
                                                  os.path.basename(src_img))):
            copied += 1
        src_lab = os.path.join(ann_dir, stem + ".txt")
        _copy_if_changed(src_lab, os.path.join(dest, "labels", stem + ".txt"))

        with open(src_lab) as fh:
            text = fh.read()
        kept = []
        for line in text.splitlines():
            if not line.split():
                continue
            n_boxes += 1
            if is_exact_degenerate_line(line):
                n_exact += 1
            _cls, coords, _fields = parse_boxes(line)[0]
            if is_degenerate(coords):
                n_degenerate += 1
            else:
                kept.append(line.strip())
        with open(os.path.join(dest, "labels_localized", stem + ".txt"), "w") as fh:
            fh.write("".join(ln + "\n" for ln in kept))

        if i % 2000 == 0:
            log(f"staged {i} of {len(staged)} records")

    with open(os.path.join(dest, "classes.txt"), "w") as fh:
        fh.write("".join(name + "\n" for name in classes))
    _write_data_yaml(dest, classes)

    src_license = os.path.join(source, "LICENSE.md")
    if os.path.isfile(src_license):
        _copy_if_changed(src_license, os.path.join(dest, "LICENSE.md"))
    else:
        with open(os.path.join(dest, "LICENSE.md"), "w") as fh:
            fh.write("# License\n\n" + LICENSE_NOTE + "\n")

    manifest = {
        "dataset": os.path.basename(dest.rstrip(os.sep)),
        "n_images": len(staged),
        "n_boxes": n_boxes,
        "n_degenerate": n_degenerate,
        "n_degenerate_exact": n_exact,
        "n_classes": len(classes),
        "classes": classes,
        "source_path": source,
        "ingested_at": ast_now(),
        "limit": int(limit) if limit not in (None, 0) else None,
        "citation": CITATION,
        "license_note": LICENSE_NOTE,
        "coverage_note": ("Outside the TCRMP survey. This dataset never enters "
                          "inprocess/run_*, the mask registry, the label "
                          "coverage matrix, or STATS."),
    }
    tmp = os.path.join(dest, "ingest_manifest.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(tmp, os.path.join(dest, "ingest_manifest.json"))

    log(f"staged {len(staged)} images, {n_boxes} boxes "
        f"({n_degenerate} whole-frame), {len(classes)} classes -> {dest}")
    log(f"{copied} image files copied this run (the rest were already staged)")
    return manifest


# ── CLI ──────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Stage the Coral Intel bounding-box dataset into the module.")
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="the distribution folder holding images/ and annotations/")
    ap.add_argument("--dest", default=default_dest(),
                    help="where to stage it (supporting_data/bbox_datasets/coral_intel)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stage only the first N records (0 stages every record)")
    ap.add_argument("--quiet", action="store_true", help="print nothing")
    args = ap.parse_args(argv)
    log = (lambda m: None) if args.quiet else print
    return ingest(args.source, args.dest, limit=args.limit or None, log=log)


if __name__ == "__main__":
    try:
        main()
    except IngestError as exc:
        print(f"ingest stopped: {exc}", file=sys.stderr)
        sys.exit(2)

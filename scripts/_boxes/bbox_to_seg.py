#!/usr/bin/env python
"""Turn staged bounding boxes into YOLO segmentation polygons with SAM3.

Reads a dataset staged by ingest_coral_intel.py, prompts the existing SAM3
engine with each box, and writes the resulting outlines as a YOLO-seg dataset
in the same shape the composite exports use:

    <out>/all_images/    symlinks back to the staged images (never copies)
    <out>/all_labels/    one txt per image, one polygon line per box
    <out>/data.yaml      path, train, val, nc, names
    <out>/manifest.csv   one row per source box: what was converted and how
    <out>/README.md      where the data came from and what may be done with it

The engine is the one the annotator already uses
(TCRMPclip_segmentImages/src/sam_engine.py), imported by path and never
modified. Tests inject a stand-in through the ``engine`` argument or the
module-level ``ENGINE`` attribute, so no test loads a model.

Boxes that cover the whole frame carry no outline worth having; they are
skipped unless --keep-degenerate is passed. A record that produces no polygon
is left out of the dataset entirely rather than shipped as a background image,
and its boxes stay in manifest.csv with the reason.

Scope: this tool writes only inside its own output folder. It never touches the
per-mask registry, annotation provenance, a pipeline project, the label
coverage matrix, or STATS. The imagery is outside the TCRMP survey.

Usage:
  env/bin/python scripts/_boxes/bbox_to_seg.py --limit 5
  env/bin/python scripts/_boxes/bbox_to_seg.py --dataset <staged> --out <dir>
"""

import argparse
import csv
import importlib
import os
import sys
import time
from datetime import datetime, timedelta, timezone

AST = timezone(timedelta(hours=-4))

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
REPO_DIR = os.path.dirname(_SCRIPTS)
MODULE_ROOT = os.path.dirname(REPO_DIR)

# The annotator's segmentation sources: sam_engine (SAM3 wrapper) and mask_ops
# (mask to polygon). Imported by path, read only.
SEG_SRC = os.path.join(_SCRIPTS, "TCRMPclip_segmentImages", "src")

SOURCE_TYPE = "bbox_sam3"

MANIFEST_FIELDS = [
    "basename", "class_id", "class_name",
    "box_x_min", "box_y_min", "box_x_max", "box_y_max",
    "polygon_points", "sam3_score", "source_type", "status",
]

# Set by a test to stand in for SAM3. Left None in every other situation.
ENGINE = None


class ConvertError(RuntimeError):
    """Anything that stops the conversion before it writes a dataset."""


def ast_now():
    return datetime.now(AST).strftime("%Y-%m-%d %H:%M:%S")


def default_dataset(repo_dir=REPO_DIR, name="coral_intel"):
    return os.path.join(repo_dir, "supporting_data", "bbox_datasets", name)


def default_out_root(module_root=MODULE_ROOT):
    return os.path.join(module_root, "inprocess", "_bbox")


# ── Imports from the annotator's segmentation sources ────────────────────

def _seg_module(name):
    if SEG_SRC not in sys.path:
        sys.path.insert(0, SEG_SRC)
    return importlib.import_module(name)


def load_engine(device_tracker="", log=print):
    """The real SAM3 engine, configured from the annotator's config module.

    Imported here rather than at module import so the CLI, the tests and the
    blueprint can all load this file without pulling torch.
    """
    if device_tracker:
        os.environ["TCRMP_SAM3_DEVICE_TRACKER"] = device_tracker
    config = _seg_module("config")
    sam_engine = _seg_module("sam_engine")
    log(f"loading SAM3 on {config.SAM3_DEVICE_TRACKER} "
        f"(mask size {config.SAM3_MASK_SIZE})")
    return sam_engine.SAM3Engine(config)


# ── Geometry ─────────────────────────────────────────────────────────────

def box_to_pixels(coords, width, height):
    """A normalised YOLO box (cx, cy, w, h) as integer pixel corners, clamped
    to the frame."""
    cx, cy, bw, bh = coords
    x_min = int(round((cx - bw / 2.0) * width))
    y_min = int(round((cy - bh / 2.0) * height))
    x_max = int(round((cx + bw / 2.0) * width))
    y_max = int(round((cy + bh / 2.0) * height))
    x_min = max(0, min(x_min, width - 1))
    y_min = max(0, min(y_min, height - 1))
    x_max = max(x_min + 1, min(x_max, width - 1))
    y_max = max(y_min + 1, min(y_max, height - 1))
    return x_min, y_min, x_max, y_max


def _polygon_area(flat):
    """Shoelace area of a flat [x1, y1, x2, y2, ...] ring."""
    pts = list(zip(flat[0::2], flat[1::2]))
    total = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def largest_polygon(polys_px, polys_norm):
    """The biggest ring of a multi-part mask, normalised. YOLO-seg carries one
    ring per line, so a mask that broke into pieces contributes its main body."""
    best, best_area = None, -1.0
    for px, norm in zip(polys_px, polys_norm):
        if len(norm) < 6:
            continue
        area = _polygon_area(px)
        if area > best_area:
            best, best_area = norm, area
    return best


# ── Staged dataset ───────────────────────────────────────────────────────

def read_classes(dataset_dir):
    path = os.path.join(dataset_dir, "classes.txt")
    if not os.path.isfile(path):
        raise ConvertError(f"{dataset_dir} is not staged: no classes.txt. "
                           "Run the Coral Intel ingest first.")
    return [ln for ln in open(path).read().splitlines() if ln.strip()]


def _labels_dir(dataset_dir):
    """Which staged label folder to read: always the verbatim copy.
    labels_localized/ already has the whole-frame boxes removed, so reading it
    would hide them from the ledger; the skip rule below decides instead."""
    return os.path.join(dataset_dir, "labels")


# ── Conversion ───────────────────────────────────────────────────────────

def convert(dataset_dir, out_dir, *, limit=None, keep_degenerate=False,
            engine=None, device="", log=print):
    """Convert a staged box dataset into a YOLO-seg dataset. Returns a summary."""
    ingest = _boxes_ingest()
    dataset_dir = os.path.abspath(os.path.expanduser(str(dataset_dir)))
    out_dir = os.path.abspath(os.path.expanduser(str(out_dir)))
    if not os.path.isfile(os.path.join(dataset_dir, "data.yaml")):
        raise ConvertError(f"{dataset_dir} is not staged: no data.yaml. "
                           "Run the Coral Intel ingest first.")
    classes = read_classes(dataset_dir)
    img_dir = os.path.join(dataset_dir, "images")
    lab_dir = _labels_dir(dataset_dir)
    if not os.path.isdir(img_dir) or not os.path.isdir(lab_dir):
        raise ConvertError(f"{dataset_dir} is not staged: images/ and labels/ "
                           "must both be present.")

    stems = sorted(fn[:-4] for fn in os.listdir(lab_dir) if fn.endswith(".txt"))
    if limit not in (None, 0):
        stems = stems[:int(limit)]

    mask_ops = _seg_module("mask_ops")
    eng = engine or ENGINE
    owns_engine = False
    if eng is None:
        eng = load_engine(device, log=log)
        owns_engine = True

    os.makedirs(os.path.join(out_dir, "all_images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "all_labels"), exist_ok=True)

    summary = {
        "dataset_dir": dataset_dir,
        "out_dir": out_dir,
        "n_records_read": len(stems),
        "n_images": 0,
        "n_boxes": 0,
        "n_polygons": 0,
        "n_skipped_degenerate": 0,
        "n_no_mask": 0,
        "keep_degenerate": bool(keep_degenerate),
        "started_at": ast_now(),
        "seconds": 0.0,
    }
    started = time.time()
    rows = []

    try:
        for i, stem in enumerate(stems, 1):
            image_path = _image_for(img_dir, stem)
            if not image_path:
                log(f"no image for {stem}, skipping")
                continue
            basename = os.path.basename(image_path)
            with open(os.path.join(lab_dir, stem + ".txt")) as fh:
                boxes = ingest.parse_boxes(fh.read())
            if not boxes:
                continue

            width, height = eng.set_image(image_path)
            lines = []
            for cls, coords, _fields in boxes:
                summary["n_boxes"] += 1
                name = classes[cls] if 0 <= cls < len(classes) else str(cls)
                corners = box_to_pixels(coords, width, height)
                row = {
                    "basename": basename,
                    "class_id": cls,
                    "class_name": name,
                    "box_x_min": corners[0], "box_y_min": corners[1],
                    "box_x_max": corners[2], "box_y_max": corners[3],
                    "polygon_points": 0,
                    "sam3_score": "",
                    "source_type": SOURCE_TYPE,
                    "status": "ok",
                }
                if ingest.is_degenerate(coords) and not keep_degenerate:
                    row["status"] = "skipped_degenerate"
                    summary["n_skipped_degenerate"] += 1
                    rows.append(row)
                    continue
                result = eng.segment_box(*corners)
                polygon = None
                if result is not None and result.get("mask") is not None:
                    px, norm = mask_ops.mask_to_polygon(result["mask"])
                    polygon = largest_polygon(px, norm)
                if polygon is None:
                    row["status"] = "no_mask"
                    summary["n_no_mask"] += 1
                    rows.append(row)
                    continue
                row["polygon_points"] = len(polygon) // 2
                row["sam3_score"] = round(float(result.get("score") or 0.0), 4)
                rows.append(row)
                lines.append(str(cls) + " "
                             + " ".join(f"{v:.6f}" for v in polygon))
                summary["n_polygons"] += 1

            if lines:
                with open(os.path.join(out_dir, "all_labels", stem + ".txt"), "w") as fh:
                    fh.write("".join(ln + "\n" for ln in lines))
                _symlink(image_path, os.path.join(out_dir, "all_images", basename))
                summary["n_images"] += 1
            if i % 50 == 0:
                log(f"converted {i} of {len(stems)} records, "
                    f"{summary['n_polygons']} polygons so far")
    finally:
        if owns_engine:
            try:
                eng.release()
            except Exception:
                pass

    summary["seconds"] = round(time.time() - started, 2)
    _write_manifest(out_dir, rows)
    _write_data_yaml(out_dir, classes)
    _write_readme(out_dir, dataset_dir, classes, summary)
    log(f"{summary['n_polygons']} polygons over {summary['n_images']} images "
        f"in {summary['seconds']}s -> {out_dir}")
    return summary


def _boxes_ingest():
    """The ingest module beside this one, however this file was started
    (imported as _boxes.ingest_coral_intel, or run straight from the folder)."""
    if __package__:
        return importlib.import_module("." + "ingest_coral_intel", __package__)
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    return importlib.import_module("ingest_coral_intel")


def _image_for(img_dir, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        cand = os.path.join(img_dir, stem + ext)
        if os.path.isfile(cand):
            return cand
    return ""


def _symlink(target, link_path):
    if os.path.islink(link_path) or os.path.exists(link_path):
        try:
            os.unlink(link_path)
        except OSError:
            return
    os.symlink(target, link_path)


def _write_manifest(out_dir, rows):
    path = os.path.join(out_dir, "manifest.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_data_yaml(out_dir, classes):
    lines = [f"path: {out_dir}", "train: all_images", "val: all_images",
             f"nc: {len(classes)}", "names:"]
    lines += [f"  {i}: {name}" for i, name in enumerate(classes)]
    with open(os.path.join(out_dir, "data.yaml"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_readme(out_dir, dataset_dir, classes, summary):
    ingest = _boxes_ingest()
    text = f"""# Boxes converted to masks

Built {summary['started_at']} AST by scripts/_boxes/bbox_to_seg.py.

## What this is

Every polygon here was produced by prompting SAM3 with one bounding box from a
staged detection dataset. Mask source type: `{SOURCE_TYPE}`. No person drew or
checked these outlines, so treat them as a starting point for training, not as
reviewed labels.

## Where it came from

Staged dataset: `{dataset_dir}`
Records read: {summary['n_records_read']}
Images written: {summary['n_images']}
Polygons: {summary['n_polygons']}
Whole-frame boxes skipped: {summary['n_skipped_degenerate']}
Boxes SAM3 returned nothing for: {summary['n_no_mask']}
Classes: {len(classes)}

`all_images/` holds symlinks back to the staged images. Copying this folder
somewhere else without the staged dataset leaves the links broken.

## License and citation

{ingest.LICENSE_NOTE}

Citation:

> {ingest.CITATION}

## Scope

These images are outside the TCRMP survey. This dataset never enters the
pipeline projects, the per-mask registry, annotation provenance, the label
coverage matrix, or STATS. It exists to train detection and segmentation
models beside the survey data.
"""
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write(text)


# ── CLI ──────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert staged bounding boxes to YOLO-seg polygons with SAM3.")
    ap.add_argument("--dataset", default=default_dataset(),
                    help="staged dataset folder (supporting_data/bbox_datasets/<name>)")
    ap.add_argument("--name", default="",
                    help="output folder name under inprocess/_bbox "
                         "(default: <dataset>_<stamp>)")
    ap.add_argument("--out", default="",
                    help="full output path, overriding --name")
    ap.add_argument("--limit", type=int, default=0,
                    help="convert only the first N records (0 converts every record)")
    ap.add_argument("--keep-degenerate", action="store_true",
                    help="also prompt with boxes that cover the whole frame")
    ap.add_argument("--device", default="",
                    help="tracker device for SAM3, e.g. cuda:0 (default: the "
                         "annotator's configured device)")
    ap.add_argument("--quiet", action="store_true", help="print nothing")
    args = ap.parse_args(argv)

    out = args.out
    if not out:
        name = args.name or (os.path.basename(args.dataset.rstrip(os.sep))
                             + "_" + time.strftime("%Y%m%d_%H%M%S"))
        out = os.path.join(default_out_root(), name)
    log = (lambda m: None) if args.quiet else print
    return convert(args.dataset, out, limit=args.limit or None,
                   keep_degenerate=args.keep_degenerate, device=args.device, log=log)


if __name__ == "__main__":
    try:
        main()
    except ConvertError as exc:
        print(f"conversion stopped: {exc}", file=sys.stderr)
        sys.exit(2)

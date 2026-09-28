#!/usr/bin/env python
"""Coral Intel ingest: staged layout, verified class map, degenerate handling.

No-pytest house harness. Every fixture is a synthetic source tree built in a
temp dir and removed in the finally block: no test here reads the real Dropbox
dataset, and nothing is written outside the temp dirs.

Covers:
  layout      images/ labels/ labels_localized/ data.yaml classes.txt
              LICENSE.md ingest_manifest.json, plus the all_images/all_labels
              compatibility links the training tool reads
  class map   alphabetical index of the hyphenated species name, verified
              against every annotation file
  aborts      a filename that does not carry a hyphenated species name, a
              class id that disagrees with the alphabetical index, a source
              missing images/ or annotations/
  labels      labels/ is byte-identical to the source; labels_localized/ drops
              full-frame boxes and keeps an empty file when every box goes
  manifest    counts, class list, source path, AST timestamp, license note
  cli         --limit stages a prefix, a second run is idempotent

Run with:
  env/bin/python scripts/_boxes/tests/test_ingest_coral_intel.py
"""

import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _boxes import ingest_coral_intel as ing  # noqa: E402

_fail = 0


def check(cond, msg):
    global _fail
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _fail += 1


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def write_jpeg(path, size=(32, 32), color=(20, 90, 140)):
    """A real (tiny) JPEG, so the box-to-seg tests can open the staged copy."""
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")


# Three species. Alphabetical order pins the class ids:
#   Alpha-beta 0, Mid-gamma 1, Zeta-alpha 2
FIXTURE = {
    # stem: annotation text (deliberately mixed trailing-newline styles)
    "Alpha-beta_1": "0 0.25 0.25 0.2 0.2\n0 0.75 0.75 0.3 0.3\n",
    "Alpha-beta_2": "0 0.5 0.5 1 1\n",                       # every box full frame
    "Mid-gamma_10": "1 0.4 0.4 0.5 0.5\n1 0.5 0.5 1 1\n",    # one real, one full frame
    "Zeta-alpha_7": "2 0.6 0.6 0.25 0.25",                   # no trailing newline
}


def build_source(root, license_text=None):
    """A synthetic stand-in for the Dropbox dataset directory."""
    for stem, text in FIXTURE.items():
        write_jpeg(os.path.join(root, "images", stem + ".jpg"))
        write(os.path.join(root, "annotations", stem + ".txt"), text)
    write(os.path.join(root, "LICENSE.md"),
          license_text if license_text is not None
          else "# Fixture license\n\nCite Cordero et al. 2026.\n")
    return root


def main():
    src = tempfile.mkdtemp(prefix="boxes_ingest_src_")
    dest = tempfile.mkdtemp(prefix="boxes_ingest_dest_")
    bad = tempfile.mkdtemp(prefix="boxes_ingest_bad_")
    try:
        build_source(src)
        out = os.path.join(dest, "coral_intel")

        # ---- happy path -------------------------------------------------
        manifest = ing.ingest(src, out, log=lambda m: None)

        for rel in ("images", "labels", "labels_localized", "data.yaml",
                    "classes.txt", "LICENSE.md", "ingest_manifest.json"):
            check(os.path.exists(os.path.join(out, rel)),
                  f"staged dataset carries {rel}")
        check(sorted(os.listdir(os.path.join(out, "images"))) ==
              sorted(s + ".jpg" for s in FIXTURE),
              "every source image is staged")
        check(sorted(os.listdir(os.path.join(out, "labels"))) ==
              sorted(s + ".txt" for s in FIXTURE),
              "every source annotation is staged")

        # ---- class map ---------------------------------------------------
        classes = open(os.path.join(out, "classes.txt")).read().splitlines()
        check(classes == ["Alpha-beta", "Mid-gamma", "Zeta-alpha"],
              f"classes.txt is the alphabetical species list (got {classes})")
        yaml_text = open(os.path.join(out, "data.yaml")).read()
        check("nc: 3" in yaml_text, f"data.yaml pins nc (got {yaml_text!r})")
        check("path: ." in yaml_text, "data.yaml pins path: .")
        for i, nm in enumerate(classes):
            check(f"  {i}: {nm}" in yaml_text, f"data.yaml names carries {i}: {nm}")

        # ---- labels ------------------------------------------------------
        for stem, text in FIXTURE.items():
            staged = open(os.path.join(out, "labels", stem + ".txt")).read()
            check(staged == text, f"labels/{stem}.txt is a verbatim copy")
        loc = open(os.path.join(out, "labels_localized", "Alpha-beta_2.txt")).read()
        check(loc == "", "a file whose every box is full frame stays as an empty file")
        loc = open(os.path.join(out, "labels_localized", "Mid-gamma_10.txt")).read()
        check(loc == "1 0.4 0.4 0.5 0.5\n",
              f"labels_localized drops only the full-frame box (got {loc!r})")
        loc = open(os.path.join(out, "labels_localized", "Zeta-alpha_7.txt")).read()
        check(loc == "2 0.6 0.6 0.25 0.25\n",
              f"a source file with no trailing newline is normalised (got {loc!r})")
        check(sorted(os.listdir(os.path.join(out, "labels_localized"))) ==
              sorted(s + ".txt" for s in FIXTURE),
              "labels_localized holds one file per image, empty ones included")

        # ---- manifest -----------------------------------------------------
        disk = json.load(open(os.path.join(out, "ingest_manifest.json")))
        check(disk == manifest, "ingest returns exactly what it wrote to disk")
        check(manifest["n_images"] == 4, f"n_images (got {manifest['n_images']})")
        check(manifest["n_boxes"] == 6, f"n_boxes counts parsed lines (got {manifest['n_boxes']})")
        check(manifest["n_degenerate"] == 2, f"n_degenerate (got {manifest['n_degenerate']})")
        check(manifest["n_degenerate_exact"] == 2,
              f"n_degenerate_exact (got {manifest['n_degenerate_exact']})")
        check(manifest["n_classes"] == 3 and manifest["classes"] == classes,
              "manifest carries the class count and the class list")
        check(manifest["source_path"] == src, "manifest records the source path")
        check(len(manifest["ingested_at"]) == 19 and manifest["ingested_at"][4] == "-",
              f"ingested_at is an AST 24h stamp (got {manifest['ingested_at']!r})")
        check("Cordero" in manifest["license_note"] and "images" in manifest["license_note"],
              "license_note names the citation and the image exclusion")
        check("—" not in manifest["license_note"], "license_note has no em dash")

        # ---- no compatibility links ------------------------------------------
        # The staged layout is exactly the pinned one. all_images/all_labels
        # aliases would let a bbox dataset slip into the Step 6 split tool that
        # was never meant to read it, quietly, instead of failing where a person
        # can see it; they would also double the tree under a naive copy.
        for link in ("all_images", "all_labels"):
            p = os.path.join(out, link)
            check(not os.path.exists(p) and not os.path.islink(p),
                  f"the staged dataset carries no {link} alias")
        check(sorted(os.listdir(out)) ==
              ["LICENSE.md", "classes.txt", "data.yaml", "images",
               "ingest_manifest.json", "labels", "labels_localized"],
              f"the staged folder holds exactly the pinned entries (got {sorted(os.listdir(out))})")

        # ---- LICENSE ---------------------------------------------------------
        check(open(os.path.join(out, "LICENSE.md")).read() ==
              open(os.path.join(src, "LICENSE.md")).read(),
              "LICENSE.md is copied verbatim")

        # ---- idempotent -------------------------------------------------------
        again = ing.ingest(src, out, log=lambda m: None)
        check(again["n_images"] == 4 and again["n_boxes"] == 6,
              "a second ingest into the same folder repeats the same counts")
        check(len(os.listdir(os.path.join(out, "labels"))) == 4,
              "a second ingest does not duplicate files")

        # ---- limit -------------------------------------------------------------
        small = os.path.join(dest, "small")
        m2 = ing.main(["--source", src, "--dest", small, "--limit", "2", "--quiet"])
        check(m2["n_images"] == 2, f"--limit stages a prefix (got {m2['n_images']})")
        check(m2["limit"] == 2, "the manifest records the limit that produced it")
        check(m2["classes"] == classes,
              "a limited ingest still derives the class map from every annotation file")
        # A limited run over a fuller folder would leave the older records in
        # place and write a manifest that undercounts them: refuse instead.
        try:
            ing.ingest(src, out, limit=1, log=lambda m: None)
            check(False, "a limited ingest over a fuller staged folder aborts")
        except ing.IngestError as e:
            check("already holds 4" in str(e) and "--limit" in str(e),
                  f"the abort message names the counts and the way out (got {e})")
        check(len(os.listdir(os.path.join(out, "labels"))) == 4,
              "the refused limited ingest changed nothing already staged")
        m3 = ing.main(["--source", src, "--dest", small, "--limit", "2", "--quiet"])
        check(m3["n_images"] == 2, "re-running the same limit over the same folder is fine")

        # ---- aborts ---------------------------------------------------------------
        build_source(bad)
        write(os.path.join(bad, "annotations", "nohyphen_3.txt"), "0 0.1 0.1 0.1 0.1\n")
        write_jpeg(os.path.join(bad, "images", "nohyphen_3.jpg"))
        try:
            ing.ingest(bad, os.path.join(dest, "bad"), log=lambda m: None)
            check(False, "a filename with no hyphenated species name aborts")
        except ing.IngestError as e:
            check("nohyphen_3.txt" in str(e),
                  f"the abort message names the offending file (got {e})")
        check(not os.path.exists(os.path.join(dest, "bad", "images")),
              "an aborted ingest stages nothing")

        shutil.rmtree(bad, ignore_errors=True)
        build_source(bad)
        write(os.path.join(bad, "annotations", "Alpha-beta_1.txt"), "2 0.25 0.25 0.2 0.2\n")
        try:
            ing.ingest(bad, os.path.join(dest, "bad2"), log=lambda m: None)
            check(False, "a class id that disagrees with the alphabetical index aborts")
        except ing.IngestError as e:
            check("Alpha-beta_1.txt" in str(e) and "2" in str(e),
                  f"the abort message names the file and the id (got {e})")

        empty = tempfile.mkdtemp(prefix="boxes_ingest_empty_")
        try:
            ing.ingest(empty, os.path.join(dest, "bad3"), log=lambda m: None)
            check(False, "a source with no images/ or annotations/ aborts")
        except ing.IngestError as e:
            check("annotations" in str(e) or "images" in str(e),
                  f"the abort message names the missing folder (got {e})")
        finally:
            shutil.rmtree(empty, ignore_errors=True)

        # ---- degenerate rule ---------------------------------------------------------
        check(ing.is_degenerate([0.5, 0.5, 1.0, 1.0]) is True, "0.5 0.5 1 1 is degenerate")
        check(ing.is_degenerate([0.5, 0.4998, 1.0, 0.9996]) is True,
              "a box within a thousandth of the whole frame is degenerate too")
        check(ing.is_degenerate([0.5, 0.5, 0.9, 1.0]) is False,
              "a box that localises in x is not degenerate")
        check(ing.is_exact_degenerate_line("0 0.5 0.5 1 1") is True,
              "the written 0.5 0.5 1 1 form is counted separately")
        check(ing.is_exact_degenerate_line("0 0.5 0.5 1.0 1.0") is False,
              "only the literal written form counts as exact")
    finally:
        for d in (src, dest, bad):
            shutil.rmtree(d, ignore_errors=True)
    check(not os.path.exists(src) and not os.path.exists(dest) and not os.path.exists(bad),
          "temp fixtures removed")

    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()

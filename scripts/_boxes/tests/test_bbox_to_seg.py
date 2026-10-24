#!/usr/bin/env python
"""Box-to-mask bridge: staged boxes become YOLO-seg polygons via SAM3.

No-pytest house harness. A fake engine stands in for SAM3 (constructor
parameter in one pass, module attribute in another), so no test here loads a
model, touches a GPU, or reads the real Dropbox dataset. The staged dataset the
test converts is built by the ingest tests' synthetic fixture.

Covers:
  layout      all_images symlinks, all_labels, data.yaml, manifest.csv, README
  polygons    one YOLO-seg line per converted box, class id preserved,
              coordinates normalised inside 0..1
  degenerate  whole-frame boxes skipped by default, kept with --keep-degenerate
  limit       --limit stages a prefix
  manifest    one ledger row per source box with its status
  no mask     an engine that returns nothing writes no label line
  isolation   the module never names the mask registry, label provenance, or a
              run_ folder, and writes nothing outside its output folder

Run with:
  env/bin/python scripts/_boxes/tests/test_bbox_to_seg.py
"""

# Keep this suite off the live mask registry, expert library and clip index
# when it is run alone: scripts/tests/_isolation.py sets the isolating
# variables to temp dirs and blocks any write to the live stores.
import os as _iso_os
import sys as _iso_sys
_iso_dir = _iso_os.path.dirname(_iso_os.path.abspath(__file__))
while _iso_os.path.basename(_iso_dir) != 'scripts' and _iso_dir != _iso_os.path.dirname(_iso_dir):
    _iso_dir = _iso_os.path.dirname(_iso_dir)
_iso_sys.path.insert(0, _iso_os.path.join(_iso_dir, 'tests'))
import _isolation
_isolation.install()

import csv
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import numpy as np  # noqa: E402

from _boxes import bbox_to_seg as b2s  # noqa: E402
from _boxes import ingest_coral_intel as ing  # noqa: E402
from _boxes.tests.test_ingest_coral_intel import build_source  # noqa: E402

_fail = 0


def check(cond, msg):
    global _fail
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _fail += 1


class FakeEngine:
    """Answers a box prompt with a filled rectangle just inside that box."""

    def __init__(self, answer=True):
        self.answer = answer
        self.size = (32, 32)
        self.calls = []
        self.images = []

    def set_image(self, path):
        self.images.append(path)
        return self.size

    def segment_box(self, x_min, y_min, x_max, y_max):
        self.calls.append((x_min, y_min, x_max, y_max))
        if not self.answer:
            return None
        w, h = self.size
        mask = np.zeros((h, w), dtype=bool)
        x0, x1 = max(0, int(x_min) + 1), min(w, int(x_max) - 1)
        y0, y1 = max(0, int(y_min) + 1), min(h, int(y_max) - 1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        mask[y0:y1, x0:x1] = True
        return {"mask": mask, "score": 0.83, "bbox": None}

    def release(self):
        pass


def rows_of(out):
    with open(os.path.join(out, "manifest.csv"), newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    src = tempfile.mkdtemp(prefix="boxes_seg_src_")
    stage = tempfile.mkdtemp(prefix="boxes_seg_stage_")
    outs = tempfile.mkdtemp(prefix="boxes_seg_out_")
    try:
        build_source(src)
        dataset = os.path.join(stage, "coral_intel")
        ing.ingest(src, dataset, log=lambda m: None)

        # ---- happy path -------------------------------------------------
        eng = FakeEngine()
        out = os.path.join(outs, "run_a")
        summary = b2s.convert(dataset, out, engine=eng, log=lambda m: None)

        for rel in ("all_images", "all_labels", "data.yaml", "manifest.csv", "README.md"):
            check(os.path.exists(os.path.join(out, rel)), f"output carries {rel}")

        labels = sorted(os.listdir(os.path.join(out, "all_labels")))
        check(labels == ["Alpha-beta_1.txt", "Mid-gamma_10.txt", "Zeta-alpha_7.txt"],
              f"a record whose every box is whole frame is left out (got {labels})")
        images = sorted(os.listdir(os.path.join(out, "all_images")))
        check(images == ["Alpha-beta_1.jpg", "Mid-gamma_10.jpg", "Zeta-alpha_7.jpg"],
              f"one image per converted record (got {images})")
        first = os.path.join(out, "all_images", "Alpha-beta_1.jpg")
        check(os.path.islink(first) and os.path.realpath(first) == os.path.realpath(
            os.path.join(dataset, "images", "Alpha-beta_1.jpg")),
            "images are symlinks back to the staged dataset, not copies")

        lines = open(os.path.join(out, "all_labels", "Alpha-beta_1.txt")).read().splitlines()
        check(len(lines) == 2, f"both real boxes became polygons (got {len(lines)})")
        for ln in lines:
            parts = ln.split()
            coords = [float(v) for v in parts[1:]]
            check(parts[0] == "0", f"class id is preserved (got {parts[0]})")
            check(len(coords) >= 6 and len(coords) % 2 == 0,
                  f"a polygon carries at least 3 points (got {len(coords) // 2})")
            check(all(0.0 <= v <= 1.0 for v in coords),
                  "polygon coordinates are normalised inside 0 to 1")
        mid = open(os.path.join(out, "all_labels", "Mid-gamma_10.txt")).read().splitlines()
        check(len(mid) == 1 and mid[0].split()[0] == "1",
              f"the whole-frame box in a mixed record is skipped (got {mid})")

        check(summary["n_images"] == 3 and summary["n_polygons"] == 4,
              f"summary counts (got {summary})")
        check(summary["n_skipped_degenerate"] == 2,
              f"summary counts the skipped whole-frame boxes (got {summary})")
        check(len(eng.images) == 4, f"every staged record was opened once (got {len(eng.images)})")

        # ---- box geometry handed to the engine ------------------------------
        # Alpha-beta_1 box one is cx .25 cy .25 w .2 h .2 on a 32 px frame:
        # x from (0.25-0.1)*32 = 4.8 to (0.25+0.1)*32 = 11.2
        check(eng.calls[0] == (5, 5, 11, 11),
              f"normalised boxes become pixel corners (got {eng.calls[0]})")

        # ---- manifest -------------------------------------------------------
        rows = rows_of(out)
        check([r["basename"] for r in rows].count("Alpha-beta_2.jpg") == 1,
              "the manifest keeps a row for a record that produced nothing")
        check(len(rows) == 6, f"one ledger row per source box (got {len(rows)})")
        ok = [r for r in rows if r["status"] == "ok"]
        check(len(ok) == 4 and all(r["source_type"] == "bbox_sam3" for r in ok),
              "converted rows are marked bbox_sam3")
        check(all(int(r["polygon_points"]) >= 3 for r in ok),
              "each converted row records its polygon point count")
        check(all(float(r["sam3_score"]) == 0.83 for r in ok),
              "each converted row records the score the engine returned")
        check(ok[0]["class_name"] == "Alpha-beta",
              f"rows carry the species name, not just the id (got {ok[0]['class_name']})")
        check([r["status"] for r in rows].count("skipped_degenerate") == 2,
              "skipped whole-frame boxes are recorded as such")
        box = (float(ok[0]["box_x_min"]), float(ok[0]["box_y_min"]),
               float(ok[0]["box_x_max"]), float(ok[0]["box_y_max"]))
        check(box == (5.0, 5.0, 11.0, 11.0), f"the source box is in the ledger (got {box})")

        # ---- data.yaml + README ------------------------------------------------
        yml = open(os.path.join(out, "data.yaml")).read()
        check("nc: 3" in yml and "0: Alpha-beta" in yml,
              "data.yaml carries the source dataset's class map")
        readme = open(os.path.join(out, "README.md")).read()
        check("Cordero" in readme and "coverage matrix" in readme,
              "README documents the citation and the coverage-matrix exclusion")
        check(dataset in readme and "bbox_sam3" in readme,
              "README names the source dataset and the mask source type")
        check("—" not in readme and "–" not in readme,
              "README has no em or en dashes")

        # ---- keep degenerate -----------------------------------------------------
        out2 = os.path.join(outs, "run_b")
        s2 = b2s.convert(dataset, out2, engine=FakeEngine(), keep_degenerate=True,
                         log=lambda m: None)
        check(s2["n_skipped_degenerate"] == 0 and s2["n_images"] == 4,
              f"--keep-degenerate converts the whole-frame boxes too (got {s2})")
        check(os.path.isfile(os.path.join(out2, "all_labels", "Alpha-beta_2.txt")),
              "the all-whole-frame record appears once its box is kept")

        # ---- limit ------------------------------------------------------------------
        out3 = os.path.join(outs, "run_c")
        s3 = b2s.convert(dataset, out3, engine=FakeEngine(), limit=2, log=lambda m: None)
        check(s3["n_records_read"] == 2, f"--limit reads a prefix (got {s3})")

        # ---- engine injected as a module attribute -------------------------------------
        out4 = os.path.join(outs, "run_d")
        b2s.ENGINE = FakeEngine()
        try:
            s4 = b2s.convert(dataset, out4, limit=1, log=lambda m: None)
        finally:
            b2s.ENGINE = None
        check(s4["n_polygons"] == 2, f"the module-level engine is used (got {s4})")

        # ---- an engine that finds nothing -------------------------------------------------
        out5 = os.path.join(outs, "run_e")
        s5 = b2s.convert(dataset, out5, engine=FakeEngine(answer=False), log=lambda m: None)
        check(s5["n_polygons"] == 0 and s5["n_no_mask"] == 4,
              f"an engine that returns nothing writes no polygons (got {s5})")
        check(not os.listdir(os.path.join(out5, "all_labels")),
              "no label files are written when nothing segmented")
        check(all(r["status"] in ("no_mask", "skipped_degenerate") for r in rows_of(out5)),
              "the ledger records every box as no_mask or skipped")

        # ---- cli ---------------------------------------------------------------------------
        out6 = os.path.join(outs, "run_f")
        b2s.ENGINE = FakeEngine()
        try:
            s6 = b2s.main(["--dataset", dataset, "--out", out6, "--limit", "1", "--quiet"])
        finally:
            b2s.ENGINE = None
        check(s6["n_records_read"] == 1 and os.path.isdir(os.path.join(out6, "all_labels")),
              f"the CLI drives the same conversion (got {s6})")

        # ---- refuses a dataset that is not staged --------------------------------------------
        try:
            b2s.convert(os.path.join(stage, "nosuch"), os.path.join(outs, "run_g"),
                        engine=FakeEngine(), log=lambda m: None)
            check(False, "an unstaged dataset is refused")
        except b2s.ConvertError as e:
            check("data.yaml" in str(e) or "not staged" in str(e),
                  f"the refusal explains what is missing (got {e})")

        # ---- isolation ------------------------------------------------------------------------
        source_text = open(b2s.__file__.replace(".pyc", ".py")).read()
        for banned in ("_mask_registry", "label_provenance", "mask_registry"):
            check(banned not in source_text,
                  f"the converter never names {banned}")
        check("inprocess/run_" not in source_text and "step4test" not in source_text,
              "the converter never names a pipeline run folder")
    finally:
        for d in (src, stage, outs):
            shutil.rmtree(d, ignore_errors=True)
    check(not os.path.exists(src) and not os.path.exists(stage) and not os.path.exists(outs),
          "temp fixtures removed")

    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()

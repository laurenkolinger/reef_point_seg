"""Tests for evaluate_run.py's metric extraction, label parsing, metric-family
choice and family-aware rounds-ledger gates. Every case builds a fake
ultralytics results object with the same shape as the real one, so no GPU
and no ultralytics import is needed.
Run: env/bin/python scripts/TCRMPtrain_oceankindCV/tests/test_evaluate_metrics.py"""
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

import contextlib
import csv
import io
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import types

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import evaluate_run as ER

_results = []


def check(cond, msg):
    """Raise AssertionError with msg when cond is false."""
    if not cond:
        raise AssertionError(msg)


def run(name, fn):
    """Run one named case, print PASS or FAIL, and record the outcome."""
    try:
        fn()
        print(f"PASS {name}")
        _results.append((name, True))
    except Exception as exc:
        print(f"FAIL {name}: {exc}")
        traceback.print_exc()
        _results.append((name, False))


# ---------------------------------------------------------------------------
# Fixture: the real Coral Intel class list (59 names, ids 0..58)
# ---------------------------------------------------------------------------

CORAL_INTEL_NAMES = [
    "Acropora-cervicornis", "Acropora-palmata", "Acropora-prolifera",
    "Agaricia-agaricites", "Agaricia-fragilis", "Agaricia-grahamae",
    "Agaricia-humilis", "Agaricia-lamarcki", "Agaricia-tenuifolia",
    "Agaricia-undata", "Cladocora-arbuscula", "Colpophyllia-natans",
    "Dendrogyra-cylindrus", "Dichocoenia-stokesi", "Diploria-clivosa",
    "Diploria-labyrinthiformis", "Diploria-strigosa", "Eusmilia-fastigiata",
    "Favia-fragum", "Isophyllia-rigida", "Isophyllia-sinuosa",
    "Leptoseris-cailleti", "Leptoseris-cucullata", "Madracis-carmabi",
    "Madracis-decactis", "Madracis-formosa", "Madracis-mirabilis",
    "Madracis-pharensis", "Madracis-senaria", "Manicina-areolata",
    "Meandrina-danae", "Meandrina-jacksoni", "Meandrina-meandrites",
    "Montastraea-cavernosa", "Mussa-angulosa", "Mycetophyllia-aliciae",
    "Mycetophyllia-danaana", "Mycetophyllia-ferox", "Mycetophyllia-lamarckiana",
    "Mycetophyllia-reesi", "Oculina-diffusa", "Oculina-valenciennesi",
    "Oculina-varicosa", "Orbicella-annularis", "Orbicella-faveolata",
    "Orbicella-franksi", "Porites-astreoides", "Porites-branneri",
    "Porites-colonensis", "Porites-divaricata", "Porites-furcata",
    "Porites-porites", "Scolymia-cubensis", "Siderastrea-radians",
    "Siderastrea-siderea", "Siderastrea-stellata", "Solenastrea-bournoni",
    "Solenastrea-hyades", "Stephanocoenia-michelini",
]
CORAL_INTEL_NAME_MAP = {i: nm for i, nm in enumerate(CORAL_INTEL_NAMES)}
# Measured on the real detector's test split (2026-09-05): four classes have
# no ground truth there, ids 16 (Diploria strigosa), 23 (Madracis carmabi),
# 30 (Meandrina danae) and 39 (Mycetophyllia reesi), so 59 names map onto 55
# slots. The old code indexed slots by class id, which put slot 16 (really
# Eusmilia fastigiata, id 17) under Diploria strigosa and shifted every later
# row, then raised on ids 55 to 58 (swallowed, so those species vanished).
ABSENT_IDS = (16, 23, 30, 39)
PRESENT_IDS = [i for i in range(len(CORAL_INTEL_NAMES)) if i not in ABSENT_IDS]


def slot_numbers(slot):
    """Distinct, recognisable numbers for one result slot."""
    return (0.100 + slot * 0.001, 0.200 + slot * 0.001,
            0.300 + slot * 0.001, 0.400 + slot * 0.001)


class FakeMetric:
    """Same shape as ultralytics.utils.metrics.Metric: arrays indexed by
    slot, ap_class_index mapping slot to class id."""

    def __init__(self, ap_class_index, n_slots=None):
        """Build slot arrays for ap_class_index, n_slots long when given (shorter arrays reproduce the IndexError)."""
        self.ap_class_index = np.array(list(ap_class_index), dtype=int)
        n = len(self.ap_class_index) if n_slots is None else n_slots
        vals = [slot_numbers(s) for s in range(n)]
        self.p = np.array([v[0] for v in vals])
        self.r = np.array([v[1] for v in vals])
        self.ap50 = np.array([v[2] for v in vals])
        self.ap = np.array([v[3] for v in vals])
        self.nc = n

    def class_result(self, i):
        """Slot i's (precision, recall, ap50, ap), as ultralytics returns them."""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def mp(self):
        """Mean precision over slots."""
        return float(self.p.mean()) if len(self.p) else 0.0

    @property
    def mr(self):
        """Mean recall over slots."""
        return float(self.r.mean()) if len(self.r) else 0.0

    @property
    def map50(self):
        """Mean AP at IoU 0.5 over slots."""
        return float(self.ap50.mean()) if len(self.ap50) else 0.0

    @property
    def map(self):
        """Mean AP over IoU 0.5 to 0.95 over slots."""
        return float(self.ap.mean()) if len(self.ap) else 0.0


class FakeConfusionMatrix:
    """matrix[predicted][true], with the trailing background row and column."""

    def __init__(self, matrix):
        """Store the matrix as floats, predicted rows by true columns."""
        self.matrix = np.asarray(matrix, dtype=float)


class FakeResults:
    """Same fields evaluate_run reads off a DetMetrics / SegmentMetrics."""

    def __init__(self, names, box=None, seg=None, nt_per_class=None,
                 confusion_matrix=None, fitness=0.5,
                 speed=None):
        """Attach only the fields given, so a missing metric object or count source stays missing."""
        self.names = names
        if box is not None:
            self.box = box
        if seg is not None:
            self.seg = seg
        if nt_per_class is not None:
            self.nt_per_class = np.asarray(nt_per_class)
        if confusion_matrix is not None:
            self.confusion_matrix = confusion_matrix
        self.fitness = fitness
        self.speed = speed if speed is not None else {"inference": 1.5}


def val_out_for(results):
    """Wrap a fake results object the way run_val() returns it."""
    return {"results": results, "weights": "/w/best.pt",
            "data_yaml": "/d/data.yaml", "wall_seconds": 2.0}


def nt_for(names, counts_by_id):
    """A validator-style nt_per_class array (bincount over target ids)."""
    arr = np.zeros(max(len(names), max(counts_by_id) + 1 if counts_by_id else 0), dtype=int)
    for cid, n in counts_by_id.items():
        arr[cid] = n
    return arr


def detector_results(names=None, ap_class_index=None, counts=None):
    """A detector results object with the Coral Intel names and the measured present ids by default."""
    names = names if names is not None else CORAL_INTEL_NAME_MAP
    ap_class_index = ap_class_index if ap_class_index is not None else PRESENT_IDS
    counts = counts if counts is not None else {cid: 10 + cid for cid in ap_class_index}
    return FakeResults(names, box=FakeMetric(ap_class_index),
                       nt_per_class=nt_for(names, counts))


# ---------------------------------------------------------------------------
# Defect 1: per-class rows follow ap_class_index
# ---------------------------------------------------------------------------

def test_per_class_rows_follow_ap_class_index():
    """Every scored class carries the numbers of its own slot per ap_class_index, never of its class id."""
    m = ER.extract_metrics(val_out_for(detector_results()))
    pc = m["per_class"]
    check(len(pc) == len(PRESENT_IDS), f"expected {len(PRESENT_IDS)} scored classes, got {len(pc)}")
    for slot, cid in enumerate(PRESENT_IDS):
        nm = CORAL_INTEL_NAMES[cid]
        p, r, ap50, ap = slot_numbers(slot)
        row = pc.get(nm)
        check(row is not None, f"{nm} (id {cid}) has no row")
        check(abs(row["box_precision"] - p) < 1e-9
              and abs(row["box_recall"] - r) < 1e-9
              and abs(row["box_map50"] - ap50) < 1e-9
              and abs(row["box_map50_95"] - ap) < 1e-9,
              f"{nm} (id {cid}) must carry slot {slot}'s numbers, got {row}")
    # The exact measured bug: id 17 sits in slot 16 because id 16 is absent.
    eus = pc["Eusmilia-fastigiata"]
    p16, r16, ap50_16, ap_16 = slot_numbers(16)
    check(abs(eus["box_map50_95"] - ap_16) < 1e-9 and abs(eus["box_precision"] - p16) < 1e-9,
          f"Eusmilia-fastigiata must carry slot 16's numbers, got {eus}")
    check("Diploria-strigosa" not in pc, "Diploria-strigosa has no ground truth and must not be scored")
    for cid in ABSENT_IDS:
        check(CORAL_INTEL_NAMES[cid] not in pc, f"id {cid} is absent and must not be scored")
    check(CORAL_INTEL_NAMES[58] in pc, "the highest id (58) is present and must keep its row")
    check(m["metric_family"] == ER.METRIC_FAMILY_BOX, f"detector family must be box, got {m['metric_family']}")


def test_absent_class_is_reported_absent_not_zero():
    # Slot 0 scores exactly zero; id 16 is absent. The two must be told apart.
    """A class with no ground truth is listed absent, distinct from a class that scored zero."""
    names = {0: "Zero-scorer", 16: "Diploria-strigosa", 17: "Eusmilia-fastigiata"}
    box = FakeMetric([0, 17])
    box.p[0] = box.r[0] = box.ap50[0] = box.ap[0] = 0.0
    res = FakeResults(names, box=box, nt_per_class=nt_for(names, {0: 3, 17: 4}))
    m = ER.extract_metrics(val_out_for(res))
    check(m["absent_classes"] == ["Diploria-strigosa"],
          f"absent list must name exactly the unscored class, got {m['absent_classes']}")
    check("Diploria-strigosa" not in m["per_class"], "absent class must not carry a per-class row")
    check(m["per_class"]["Zero-scorer"]["box_map50_95"] == 0.0, "a real zero score must stay a zero")
    check(m["per_class_instances"]["Diploria-strigosa"] == 0, "absent class has zero instances")

    tmp = tempfile.mkdtemp(prefix="eval_metrics_absent_")
    try:
        md = ER.build_report_md(m, fake_ctx(), [], fake_args(tmp))
        lines = [ln for ln in md.splitlines() if "Diploria-strigosa" in ln]
        check(len(lines) == 1 and ER.ABSENT_CLASS_NOTE in lines[0],
              f"report row for the absent class must say '{ER.ABSENT_CLASS_NOTE}', got {lines}")
        zero_lines = [ln for ln in md.splitlines() if "Zero-scorer" in ln]
        check(len(zero_lines) == 1 and "0.000" in zero_lines[0] and ER.ABSENT_CLASS_NOTE not in zero_lines[0],
              f"a zero score must render as 0.000, not as absent: {zero_lines}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_index_error_is_swallowed():
    # Arrays shorter than ap_class_index: class_result raises IndexError inside.
    """Short score arrays and a slot pointing at an unknown class id raise ValueError instead of being skipped."""
    broken = FakeMetric(PRESENT_IDS, n_slots=50)
    res = FakeResults(CORAL_INTEL_NAME_MAP, box=broken,
                      nt_per_class=nt_for(CORAL_INTEL_NAME_MAP, {0: 1}))
    raised = False
    try:
        ER.extract_metrics(val_out_for(res))
    except ValueError:
        raised = True
    check(raised, "a results object whose arrays are shorter than ap_class_index must raise")

    # A slot pointing at a class id that names does not know must raise too.
    res2 = FakeResults({0: "A", 1: "B"}, box=FakeMetric([0, 1, 9]),
                       nt_per_class=nt_for({0: "A", 1: "B"}, {0: 1, 1: 1}))
    raised2 = False
    try:
        ER.extract_metrics(val_out_for(res2))
    except ValueError:
        raised2 = True
    check(raised2, "a slot whose class id is not in names must raise, not be skipped")


# ---------------------------------------------------------------------------
# Defect 2: instance counts
# ---------------------------------------------------------------------------

def test_instance_counts_match_the_validator():
    """Instance counts come from nt_per_class when present, else the confusion matrix including its background row."""
    names = {0: "A", 1: "B", 2: "C"}
    # Validator counts: A 5, B 0 (absent), C 7.
    nt = nt_for(names, {0: 5, 2: 7})
    # Confusion matrix (predicted rows, true columns, background last) that
    # disagrees with the validator on purpose, to prove the validator wins.
    cm = FakeConfusionMatrix([[1, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 2, 0],
                              [0, 0, 0, 0]])
    res = FakeResults(names, box=FakeMetric([0, 2]), nt_per_class=nt, confusion_matrix=cm)
    m = ER.extract_metrics(val_out_for(res))
    check(m["per_class_instances"] == {"A": 5, "B": 0, "C": 7},
          f"validator counts must be used verbatim, got {m['per_class_instances']}")
    check(m["per_class_instances_source"] == ER.INSTANCE_SOURCE_VALIDATOR,
          f"source must say validator, got {m['per_class_instances_source']}")

    # Fallback: no nt_per_class. The confusion matrix carries 3 missed A
    # instances in the background row; the old column sum over class rows
    # only saw 2 for A and 0 for C (C was missed every time).
    cm2 = FakeConfusionMatrix([[2, 0, 0, 1],   # predicted A: 2 true A, 1 false alarm
                               [0, 0, 0, 0],   # predicted B: nothing
                               [0, 0, 0, 0],   # predicted C: nothing
                               [3, 0, 4, 0]])  # background row: 3 missed A, 4 missed C
    res2 = FakeResults(names, box=FakeMetric([0, 2]), confusion_matrix=cm2)
    m2 = ER.extract_metrics(val_out_for(res2))
    check(m2["per_class_instances"] == {"A": 5, "B": 0, "C": 4},
          f"confusion-matrix fallback must include the background row, got {m2['per_class_instances']}")
    check(m2["per_class_instances_source"] == ER.INSTANCE_SOURCE_CONFUSION_MATRIX,
          f"source must say confusion matrix, got {m2['per_class_instances_source']}")

    # Neither source: a clear error, never a silent empty dict.
    res3 = FakeResults(names, box=FakeMetric([0, 2]))
    raised = False
    try:
        ER.extract_metrics(val_out_for(res3))
    except ValueError as exc:
        raised = "instance" in str(exc).lower()
    check(raised, "no instance source at all must raise a ValueError naming the problem")


# ---------------------------------------------------------------------------
# Defect 4: label parser reads both shapes
# ---------------------------------------------------------------------------

def write_label(path, text):
    """Write a YOLO label file with the given text."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def test_label_parser_reads_box_and_polygon_files():
    """The label parser reads five-field box lines and polygon lines and reports malformed lines."""
    tmp = tempfile.mkdtemp(prefix="eval_metrics_labels_")
    try:
        W, H = 640, 480
        # 1. Real detection line from disk: class cx cy w h.
        box_path = os.path.join(tmp, "box.txt")
        write_label(box_path, "18 0.56640625 0.55859375 0.380859375 0.40625\n")
        parsed = ER.parse_yolo_label_file(box_path, W, H)
        inst = parsed["instances"]
        check(len(inst) == 1 and parsed["skipped"] == [], f"one box, nothing skipped: {parsed}")
        cid, shape, pts = inst[0]
        check(cid == 18 and shape == ER.LABEL_SHAPE_BOX, f"class 18 box expected, got {cid} {shape}")
        x0 = (0.56640625 - 0.380859375 / 2) * W
        y0 = (0.55859375 - 0.40625 / 2) * H
        x1 = (0.56640625 + 0.380859375 / 2) * W
        y1 = (0.55859375 + 0.40625 / 2) * H
        check(np.allclose(pts, [[x0, y0], [x1, y1]], atol=1e-6),
              f"box corners must be pixel [[x0,y0],[x1,y1]], got {pts.tolist()}")

        # 2. Polygon line: class + three x,y pairs.
        poly_path = os.path.join(tmp, "poly.txt")
        write_label(poly_path, "3 0.1 0.1 0.5 0.1 0.5 0.5\n")
        parsed = ER.parse_yolo_label_file(poly_path, W, H)
        check(len(parsed["instances"]) == 1 and parsed["skipped"] == [], f"one polygon: {parsed}")
        cid, shape, pts = parsed["instances"][0]
        check(cid == 3 and shape == ER.LABEL_SHAPE_POLYGON and pts.shape == (3, 2),
              f"polygon expected, got {cid} {shape} {pts.shape}")
        check(np.allclose(pts, [[64, 48], [320, 48], [320, 240]]), f"polygon pixels wrong: {pts.tolist()}")

        # 3. Blank file and a file of blank lines: zero instances, no skips.
        blank = os.path.join(tmp, "blank.txt")
        write_label(blank, "")
        parsed = ER.parse_yolo_label_file(blank, W, H)
        check(parsed["instances"] == [] and parsed["skipped"] == [], f"blank file: {parsed}")
        blank_lines = os.path.join(tmp, "blank_lines.txt")
        write_label(blank_lines, "\n\n   \n")
        parsed = ER.parse_yolo_label_file(blank_lines, W, H)
        check(parsed["instances"] == [] and parsed["skipped"] == [], f"blank lines: {parsed}")

        # 4. Missing file: an unlabeled image, zero instances.
        parsed = ER.parse_yolo_label_file(os.path.join(tmp, "nope.txt"), W, H)
        check(parsed["instances"] == [] and parsed["skipped"] == [], f"missing file: {parsed}")

        # 5. Mixed file: malformed, odd coordinate count, non-int class, then a good box.
        mixed = os.path.join(tmp, "mixed.txt")
        write_label(mixed,
                    "banana\n"                       # line 1: malformed
                    "2 0.1 0.1 0.5 0.1 0.5\n"       # line 2: odd coordinate count (5)
                    "x 0.5 0.5 0.2 0.2\n"            # line 3: class id not an int
                    "1.5 0.5 0.5 0.2 0.2\n"          # line 4: class id not an int
                    "7 0.5 0.5 0.2 0.2\n"            # line 5: good box
                    "4 0.5 0.5 abc 0.2\n"            # line 6: coordinate not a number
                    "-1 0.5 0.5 0.2 0.2\n"           # line 7: negative class id
                    "5 0.5 0.5 0.0 0.2\n"            # line 8: box with no area
                    "6 1.5 0.5 0.2 0.2\n")           # line 9: out of the 0..1 range
        parsed = ER.parse_yolo_label_file(mixed, W, H)
        check(len(parsed["instances"]) == 1 and parsed["instances"][0][0] == 7,
              f"only the good box survives: {parsed['instances']}")
        skipped_lines = sorted(n for n, _ in parsed["skipped"])
        check(skipped_lines == [1, 2, 3, 4, 6, 7, 8, 9],
              f"every bad line is reported by number, got {parsed['skipped']}")
        check(all(reason for _, reason in parsed["skipped"]), "every skip carries a reason")

        # 6. Bad image size is refused at entry.
        raised = False
        try:
            ER.parse_yolo_label_file(box_path, 0, H)
        except ValueError:
            raised = True
        check(raised, "a zero image width must raise ValueError")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_label_file_with_hostile_content():
    """A huge, malformed or non-numeric label file is parsed or refused without hanging."""
    tmp = tempfile.mkdtemp(prefix="eval_metrics_hostile_")
    try:
        W, H = 100, 100
        hostile = os.path.join(tmp, "hostile.txt")
        # 50,000 vertices (100,000 coordinates): the documented upper bound.
        long_line = "9 " + " ".join(["0.5"] * 100000)
        too_long = "9 " + "0" * (ER.LABEL_MAX_LINE_CHARS + 10)
        with open(hostile, "w", encoding="utf-8") as fh:
            fh.write('"quoted" 0.5 0.5 0.2 0.2\n')
            fh.write("1 0.5\x00 0.5 0.2 0.2\n")
            fh.write("\x1b[31m2\x1b[0m 0.5 0.5 0.2 0.2\n")
            fh.write("3 0.5 0.5 0.2 0.2\n")
            fh.write(long_line + "\n")
            fh.write(too_long + "\n")
            fh.write("'; DROP TABLE labels; -- 0.5 0.5 0.2 0.2\n")
        with open(hostile, "ab") as fh:
            fh.write(b"\xff\xfe 0.5 0.5 0.2 0.2\n")
        parsed = ER.parse_yolo_label_file(hostile, W, H)
        ids = [cid for cid, _, _ in parsed["instances"]]
        check(3 in ids, f"the clean line must survive hostile neighbours, got {ids}")
        check(9 in ids and any(shape == ER.LABEL_SHAPE_POLYGON and pts.shape[0] == 50000
                               for cid, shape, pts in parsed["instances"] if cid == 9),
              "a very long but well-formed polygon line is parsed")
        check(len(parsed["skipped"]) >= 5, f"hostile lines are reported, got {parsed['skipped']}")
        check(any("exceeds" in reason for _, reason in parsed["skipped"]),
              f"the over-long line is reported as exceeding the cap: {parsed['skipped']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Defects 3 and 5: metric family, box gates, cross-family comparisons
# ---------------------------------------------------------------------------

def fake_ctx():
    """A context dict with fabricated run paths, training arguments and split sizes."""
    return {"run_dir": "/x/runs/run1", "dataset_dir": "/x/dataset",
            "train_args": {"model": "yolo11n.pt", "imgsz": 256, "batch": -1, "epochs": 20},
            "split_counts": {"train": 10, "valid": 2, "test": 2},
            "epochs_completed": 20, "best_epoch": 18}


def fake_args(out_dir):
    """The argparse namespace fields the report builders read."""
    return types.SimpleNamespace(split="test", conf=0.001, iou=0.6, imgsz=256, out_dir=out_dir,
                                 preview_conf=0.25)


def box_metrics(map50_95, recall, per_class_ap):
    """A detector-shaped metrics dict with the given overall and per-class box scores."""
    per_class = {nm: {"box_map50_95": ap, "box_map50": min(1.0, ap + 0.1),
                      "box_precision": 0.6, "box_recall": recall}
                 for nm, ap in per_class_ap.items()}
    return {"overall": {"box_map50": min(1.0, map50_95 + 0.1), "box_map50_95": map50_95,
                        "box_precision": 0.61, "box_recall": recall,
                        "mask_map50": None, "mask_map50_95": None,
                        "mask_precision": None, "mask_recall": None},
            "per_class": per_class, "per_class_instances": {nm: 5 for nm in per_class_ap},
            "class_names": {i: nm for i, nm in enumerate(per_class_ap)},
            "absent_classes": [], "weights": "/w/best.pt", "data_yaml": "/d/data.yaml",
            "wall_seconds": 3.0}


def mask_metrics(map50_95, recall, per_class_ap):
    """A segmentation-shaped metrics dict with mask and box scores."""
    per_class = {nm: {"mask_map50_95": ap, "mask_map50": min(1.0, ap + 0.1),
                      "mask_precision": 0.7, "mask_recall": recall,
                      "box_map50_95": ap + 0.02, "box_map50": min(1.0, ap + 0.12),
                      "box_precision": 0.72, "box_recall": recall}
                 for nm, ap in per_class_ap.items()}
    return {"overall": {"mask_map50": min(1.0, map50_95 + 0.1), "mask_map50_95": map50_95,
                        "mask_precision": 0.72, "mask_recall": recall,
                        "box_map50": min(1.0, map50_95 + 0.12), "box_map50_95": map50_95 + 0.02,
                        "box_precision": 0.74, "box_recall": recall},
            "per_class": per_class, "per_class_instances": {nm: 5 for nm in per_class_ap},
            "class_names": {i: nm for i, nm in enumerate(per_class_ap)},
            "absent_classes": [], "weights": "/w/best.pt", "data_yaml": "/d/data.yaml",
            "wall_seconds": 3.0}


def make_box_dataset(root, classes, with_manifest=True):
    """A detection dataset the way detect_dataset.py stages one: data.yaml,
    train/valid/test image folders, train labels in box form, and a
    split_manifest.json with holdout_mode lab-random and a pins table."""
    import yaml
    os.makedirs(root, exist_ok=True)
    names = {i: c for i, c in enumerate(classes)}
    with open(os.path.join(root, "data.yaml"), "w") as fh:
        yaml.dump({"path": root, "train": "train/images", "val": "valid/images",
                   "test": "test/images", "names": names}, fh)
    pins = {}
    for split, n in (("train", 6), ("valid", 2), ("test", 2)):
        img_dir = os.path.join(root, split, "images")
        os.makedirs(img_dir, exist_ok=True)
        for i in range(n):
            fn = f"{split}_{i}.jpg"
            with open(os.path.join(img_dir, fn), "wb") as fh:
                fh.write(b"\xff\xd8\xff\xd9")
            pins[fn] = split
    lbl_dir = os.path.join(root, "train", "labels")
    os.makedirs(lbl_dir, exist_ok=True)
    for i in range(6):
        with open(os.path.join(lbl_dir, f"train_{i}.txt"), "w") as fh:
            fh.write(f"{i % len(classes)} 0.5 0.5 0.2 0.2\n")
    if with_manifest:
        with open(os.path.join(root, "split_manifest.json"), "w") as fh:
            json.dump({"schema_version": 1, "holdout_mode": "lab-random",
                       "seed": "coral_intel", "counts": {"train": 6, "valid": 2, "test": 2},
                       "pins": pins}, fh)
    return root


def make_run(root, name):
    """Create a detector run folder with an args.yaml naming its base model."""
    run_dir = os.path.join(root, name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.yaml"), "w") as fh:
        fh.write("model: yolo11n.pt\ntask: detect\n")
    return run_dir


def read_rows(rounds_dir):
    """Read every row of rounds.csv in a ledger folder."""
    path = os.path.join(rounds_dir, "rounds.csv")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_metric_family_is_chosen_from_the_numbers():
    """metric_family() reads mask when mask numbers exist, box otherwise, and refuses a dict with neither."""
    check(ER.metric_family(box_metrics(0.3, 0.4, {"A": 0.3})) == ER.METRIC_FAMILY_BOX, "box-only is box")
    check(ER.metric_family(mask_metrics(0.3, 0.4, {"A": 0.3})) == ER.METRIC_FAMILY_MASK,
          "mask plus box is the mask (segmentation) family")
    neither = {"overall": {"box_map50": None, "mask_map50": None}}
    raised = False
    try:
        ER.metric_family(neither)
    except ValueError:
        raised = True
    check(raised, "a metrics dict with neither family must raise ValueError")
    raised = False
    try:
        ER.metric_family({"overall": "not a dict"})
    except ValueError:
        raised = True
    check(raised, "a malformed overall block must raise ValueError")


def test_detector_metrics_produce_box_gates():
    """A detector's ledger row fills the box columns and its gates compare box numbers."""
    tmp = tempfile.mkdtemp(prefix="eval_metrics_boxgates_")
    try:
        rounds_dir = os.path.join(tmp, "lab")
        dataset = make_box_dataset(os.path.join(tmp, "dataset"), ("Acropora-palmata", "Porites-astreoides"))
        check(ER._split_pinned(dataset), "a detection dataset with a pins table is a frozen split")

        row1 = ER.append_round_row(rounds_dir, make_run(tmp, "run1"),
                                   box_metrics(0.50, 0.60, {"Acropora-palmata": 0.5, "Porites-astreoides": 0.5}),
                                   dataset)
        check(row1["split_pinned"] == "true", f"split_pinned must be true, got {row1['split_pinned']}")
        check(row1["metric_family"] == ER.METRIC_FAMILY_BOX, f"row family must be box, got {row1['metric_family']}")
        check(row1["gate_map"] == ER.GATE_PASS and row1["gate_class"] == ER.GATE_PASS
              and row1["gate_recall"] == ER.GATE_PASS,
              f"first round has no baseline and passes, got {row1['gate_map']} {row1['gate_class']} {row1['gate_recall']}")
        check(abs(float(row1["map50_95_B"]) - 0.50) < 1e-9 and abs(float(row1["recall_B"]) - 0.60) < 1e-9
              and row1["precision_B"] not in ("", None),
              f"box columns must be populated: {row1}")
        check(row1["map50_95_M"] in ("", None), f"mask columns stay empty for a detector, got {row1['map50_95_M']}")

        # Round 2 regresses: box mAP50-95 0.50 -> 0.40, recall 0.60 -> 0.40, PA drops 0.5 -> 0.3.
        row2 = ER.append_round_row(rounds_dir, make_run(tmp, "run2"),
                                   box_metrics(0.40, 0.40, {"Acropora-palmata": 0.5, "Porites-astreoides": 0.3}),
                                   dataset)
        check(row2["gate_map"] == ER.GATE_FAIL, f"box mAP drop must fail gate_map, got {row2['gate_map']}")
        check(row2["gate_recall"] == ER.GATE_FLAG, f"box recall drop must flag gate_recall, got {row2['gate_recall']}")
        check(row2["gate_class"].startswith(ER.GATE_FLAG) and "PA" in row2["gate_class"],
              f"Porites astreoides drop must flag gate_class with PA, got {row2['gate_class']}")

        rows = read_rows(rounds_dir)
        check(len(rows) == 2, f"two rows expected, got {len(rows)}")
        hdr = list(rows[0].keys())
        for col in ("metric_family", "recall_B", "precision_B", "metrics_schema"):
            check(col in hdr, f"ledger header must carry {col}, got {hdr}")
        check(rows[0]["metrics_schema"] == str(ER.ROUNDS_METRICS_SCHEMA),
              f"schema marker must be written, got {rows[0]['metrics_schema']}")

        # Without a manifest the split is not frozen: gates read unpinned.
        loose = make_box_dataset(os.path.join(tmp, "loose"), ("A", "B"), with_manifest=False)
        row3 = ER.append_round_row(os.path.join(tmp, "lab2"), make_run(tmp, "run3"),
                                   box_metrics(0.5, 0.5, {"A": 0.5, "B": 0.5}), loose)
        check(row3["gate_map"] == ER.GATE_UNPINNED, f"no manifest must read unpinned, got {row3['gate_map']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cross_family_comparison_is_marked_impossible():
    """A baseline of the other metric family marks all three gates incomparable."""
    tmp = tempfile.mkdtemp(prefix="eval_metrics_xfamily_")
    try:
        rounds_dir = os.path.join(tmp, "lab")
        dataset = make_box_dataset(os.path.join(tmp, "dataset"), ("A", "B"))
        # Round 1 is a segmentation run, round 2 a detector, on the same split.
        row1 = ER.append_round_row(rounds_dir, make_run(tmp, "seg1"),
                                   mask_metrics(0.60, 0.70, {"A": 0.6, "B": 0.6}), dataset)
        check(row1["metric_family"] == ER.METRIC_FAMILY_MASK, f"segmentation row is mask, got {row1['metric_family']}")
        row2 = ER.append_round_row(rounds_dir, make_run(tmp, "det1"),
                                   box_metrics(0.20, 0.30, {"A": 0.2, "B": 0.2}), dataset)
        check(row2["gate_map"] == ER.GATE_INCOMPARABLE and row2["gate_class"] == ER.GATE_INCOMPARABLE
              and row2["gate_recall"] == ER.GATE_INCOMPARABLE,
              f"a box round after a mask round must be incomparable, got {row2['gate_map']} {row2['gate_class']} {row2['gate_recall']}")
        # A second detector round compares against the detector, not the segmentation run.
        row3 = ER.append_round_row(rounds_dir, make_run(tmp, "det2"),
                                   box_metrics(0.25, 0.30, {"A": 0.25, "B": 0.25}), dataset)
        check(row3["gate_map"] == ER.GATE_PASS, f"same-family baseline must be used, got {row3['gate_map']}")

        # A champion of the other family makes the comparison impossible too.
        with open(os.path.join(rounds_dir, "champion.json"), "w") as fh:
            json.dump({"run_dir": os.path.join(tmp, "seg1"), "promoted_at": "2026-09-04T10:00:00-04:00",
                       "map50_95_M": 0.60, "note": ""}, fh)
        row4 = ER.append_round_row(rounds_dir, make_run(tmp, "det3"),
                                   box_metrics(0.90, 0.90, {"A": 0.9, "B": 0.9}), dataset)
        check(row4["gate_map"] == ER.GATE_INCOMPARABLE,
              f"a mask champion cannot judge a box round, got {row4['gate_map']}")

        # Legacy rows without a metric_family column are classified from their numbers.
        legacy = {"map50_95_M": "", "map50_95_B": "0.3"}
        check(ER._row_family(legacy) == ER.METRIC_FAMILY_BOX, "legacy box row inferred from map50_95_B")
        legacy_m = {"map50_95_M": "0.5", "map50_95_B": "0.6"}
        check(ER._row_family(legacy_m) == ER.METRIC_FAMILY_MASK, "legacy mask row inferred from map50_95_M")
        check(ER._row_family({"map50_95_M": "", "map50_95_B": ""}) is None, "an empty row has no family")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_markdown_report_labels_the_family():
    """The markdown report names the run's metric family and lists absent classes."""
    tmp = tempfile.mkdtemp(prefix="eval_metrics_md_")
    try:
        m = ER.extract_metrics(val_out_for(detector_results()))
        md = ER.build_report_md(m, fake_ctx(), [], fake_args(tmp))
        check("box scores" in md.lower(), "a detector report names the box family")
        check("not present in this split" in md, "absent classes are said to be absent")
        per_class = md.split("## Per-class")[1].split("## ")[0]
        eus_line = [ln for ln in per_class.splitlines() if "Eusmilia-fastigiata" in ln][0]
        p16 = slot_numbers(16)[0]
        check(f"{p16:.3f}" in eus_line, f"Eusmilia fastigiata row carries slot 16 numbers: {eus_line}")
        check("N/A" not in eus_line, f"a scored class must not render N/A: {eus_line}")
        check("*Eusmilia fastigiata*" in eus_line, f"binomial is shown in full and italic: {eus_line}")
        check("`Eusmilia-fastigiata`" in eus_line, f"the class token stays beside the name: {eus_line}")
        check("\u2014" not in md and "\u2013" not in md, "no em or en dashes in the report")
        check(" AST" in md, "the report timestamp is Atlantic Standard Time")

        # Segmentation report keeps the two-column top line and outline wording.
        seg = FakeResults({0: "SS", 1: "MCAV"}, box=FakeMetric([0, 1]), seg=FakeMetric([0, 1]),
                          nt_per_class=nt_for({0: "SS", 1: "MCAV"}, {0: 2, 1: 3}))
        ms = ER.extract_metrics(val_out_for(seg))
        check(ms["metric_family"] == ER.METRIC_FAMILY_MASK, "seg results are the mask family")
        md2 = ER.build_report_md(ms, fake_ctx(), [], fake_args(tmp))
        check("outline" in md2.lower(), "a segmentation report names the outline family")
        check("*Siderastrea siderea*" in md2 and "`SS`" in md2,
              "a registry code is shown with its full name beside it")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Adversarial cases
# ---------------------------------------------------------------------------

def test_empty_names_mapping_is_refused():
    """A results object with no class names raises ValueError."""
    res = FakeResults({}, box=FakeMetric([]), nt_per_class=np.zeros(0, dtype=int))
    raised = False
    try:
        ER.extract_metrics(val_out_for(res))
    except ValueError as exc:
        raised = "class name" in str(exc).lower()
    check(raised, "empty names mapping must raise a ValueError about class names")


def test_empty_ap_class_index_marks_every_class_absent():
    """An empty ap_class_index scores nothing and marks every class absent."""
    names = {0: "A", 1: "B"}
    res = FakeResults(names, box=FakeMetric([]), nt_per_class=nt_for(names, {}))
    m = ER.extract_metrics(val_out_for(res))
    check(m["per_class"] == {}, f"no slots means no scored classes, got {m['per_class']}")
    check(m["absent_classes"] == ["A", "B"], f"every class is absent, got {m['absent_classes']}")
    check(m["overall"]["box_map50"] == 0.0, "an empty split yields a zero mean, not a crash")


def test_non_contiguous_class_ids():
    """Class ids with gaps map each slot to the right name."""
    names = {0: "A", 3: "B", 7: "C"}
    # Only A and C have ground truth; C's id (7) exceeds len(names).
    res = FakeResults(names, box=FakeMetric([0, 7]), nt_per_class=nt_for(names, {0: 4, 7: 9}))
    m = ER.extract_metrics(val_out_for(res))
    check(set(m["per_class"]) == {"A", "C"}, f"A and C scored, got {set(m['per_class'])}")
    check(abs(m["per_class"]["C"]["box_map50_95"] - slot_numbers(1)[3]) < 1e-9, "C carries slot 1")
    check(m["absent_classes"] == ["B"], f"B absent, got {m['absent_classes']}")
    check(m["per_class_instances"] == {"A": 4, "B": 0, "C": 9}, f"counts by id: {m['per_class_instances']}")
    # A validator array shorter than the highest id: that id simply has no targets.
    res2 = FakeResults(names, box=FakeMetric([0]), nt_per_class=np.array([4, 0, 0]))
    m2 = ER.extract_metrics(val_out_for(res2))
    check(m2["per_class_instances"] == {"A": 4, "B": 0, "C": 0}, f"short array reads as zero: {m2['per_class_instances']}")


def test_results_with_both_and_with_neither_family():
    """Both families are kept for a segmentation run; a results object with neither raises."""
    names = {0: "A", 1: "B"}
    both = FakeResults(names, box=FakeMetric([0, 1]), seg=FakeMetric([0, 1]),
                       nt_per_class=nt_for(names, {0: 1, 1: 1}))
    m = ER.extract_metrics(val_out_for(both))
    check(m["metric_family"] == ER.METRIC_FAMILY_MASK, "both families present is a segmentation run")
    check(all(k in m["per_class"]["A"] for k in ("box_map50_95", "mask_map50_95")),
          "both families' per-class keys are kept")
    check(m["overall"]["mask_map50"] is not None and m["overall"]["box_map50"] is not None,
          "both overall families are populated")

    neither = FakeResults(names, nt_per_class=nt_for(names, {0: 1}))
    raised = False
    try:
        ER.extract_metrics(val_out_for(neither))
    except ValueError as exc:
        raised = "neither" in str(exc).lower()
    check(raised, "a results object with neither box nor seg metrics must raise")


def test_pdf_content_builders_render_both_families():
    """The page builders build_pdf calls are exercised here directly, without
    Chrome, for a detector and a segmentation run."""
    tmp = tempfile.mkdtemp(prefix="eval_metrics_pdfpages_")
    try:
        args = fake_args(tmp)
        det = ER.extract_metrics(val_out_for(detector_results()))
        seg = FakeResults({0: "SS", 1: "MCAV", 2: "PAST"}, box=FakeMetric([0, 2]), seg=FakeMetric([0, 2]),
                          nt_per_class=nt_for({0: "SS", 1: "MCAV", 2: "PAST"}, {0: 2, 2: 3}))
        segm = ER.extract_metrics(val_out_for(seg))
        for label, metrics in (("detector", det), ("segmentation", segm)):
            family = ER.metric_family(metrics)
            bar = ER._plot_per_class_bar(metrics["per_class"], os.path.join(tmp, f"{label}_bar.png"),
                                         metric_key=ER.family_key(family, "map50_95"))
            check(bar is not None and os.path.getsize(bar) > 0, f"{label}: bar chart written")
            figures = {"per_class_bar": bar}
            plan = ER._plan_pages(len(metrics["per_class"]) + len(metrics["absent_classes"]), 0)
            page5 = "".join(ER._render_perclass(metrics, args, figures, plan))
            check("Page 5: Per-class breakdown" in page5, f"{label}: per-class page title")
            check(ER.FAMILY_LABELS[family] in page5, f"{label}: per-class page names its family")
            check("N/A" not in page5.split("<table")[1].split("</table>")[0],
                  f"{label}: no N/A cell in the per-class table")
            page3 = ER._render_metric_page("box", 3, metrics, fake_ctx(), args, figures)
            page4 = ER._render_metric_page("mask", 4, metrics, fake_ctx(), args, figures)
            cover = ER._render_cover(metrics, fake_ctx(), args, [("Cover", 1)], "2026-09-04 15:00 AST", "run1")
            for name, html in (("page3", page3), ("page4", page4), ("cover", cover), ("page5", page5)):
                check("\u2014" not in html and "\u2013" not in html, f"{label} {name}: no em or en dash")
        det_plan = ER._plan_pages(len(det["per_class"]) + len(det["absent_classes"]), 0)
        det_page5 = "".join(ER._render_perclass(det, args, {}, det_plan))
        check(ER.ABSENT_CLASS_NOTE in det_page5 and "Diploria strigosa" in det_page5,
              "detector per-class page lists the absent species by full name")
        check("<i>Eusmilia fastigiata</i>" in det_page5, "binomials are italic in the PDF table")
        det_page4 = ER._render_metric_page("mask", 4, det, fake_ctx(), args, {})
        check(ER.DETECTOR_NO_MASK_NOTE in det_page4, "detector mask page says why it is empty")
        check("incomplete" not in det_page4, "detector mask page does not call the run incomplete")
        seg_page4 = ER._render_metric_page("mask", 4, segm, fake_ctx(), args, {})
        check(ER.DETECTOR_NO_MASK_NOTE not in seg_page4, "segmentation mask page keeps its interpretation")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_display_name_resolution():
    """class_display_name renders binomials italic with the code beside them and leaves unknown tokens as code."""
    check(ER.class_display_name("Acropora-cervicornis") == "*Acropora cervicornis* (`Acropora-cervicornis`)",
          f"dashed binomial: {ER.class_display_name('Acropora-cervicornis')}")
    check(ER.class_display_name("Porites_astreoides") == "*Porites astreoides* (`Porites_astreoides`)",
          f"underscored binomial: {ER.class_display_name('Porites_astreoides')}")
    ss = ER.class_display_name("SS")
    check(ss == "*Siderastrea siderea* (`SS`)", f"registry code resolves to its binomial: {ss}")
    check(ER.class_display_name("sand") == "`sand`", f"an unresolvable token stays a code span: {ER.class_display_name('sand')}")
    check(ER.class_display_name("") == "`?`", "empty name renders as a placeholder code span")


def test_counts_take_a_singular_or_plural_noun():
    """The preview captions read "1 instance" and "2 detections", never "(s)".

    The same helper writes the caption baked into the side-by-side image, the
    line in report.md and the caption in the PDF, so one check covers all
    three renderings.
    """
    check(ER.count_noun(1, "instance") == "1 instance", "one takes the singular")
    check(ER.count_noun(2, "detection") == "2 detections", "two takes the plural")
    check(ER.count_noun(0, "detection") == "0 detections", "zero takes the plural")
    check(ER.count_noun(-1, "instance") == "-1 instances", "a negative count takes the plural")
    check(ER.count_noun("1", "instance") == "1 instance", "a count that arrives as text still reads as one")
    check(ER.count_noun("?", "instance") == "? instances", "an unknown count takes the plural and is never dropped")
    check(ER.count_noun(None, "instance") == "None instances", "a missing count is shown, never hidden")
    check(ER.count_noun(1.0, "instance") == "1.0 instances",
          "a float is not the integer one: only an exact integer one takes the singular")
    for bad in ("", "  "):
        try:
            ER.count_noun(1, bad)
            raise AssertionError(f"a blank noun must be refused, got {bad!r}")
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Full species names from the staged dataset (T6)
# ---------------------------------------------------------------------------

# One fish class token and its full name, as the staged fish dataset writes
# them: a token that resolves through neither the module's code registry nor
# the binomial pattern.
FISH_TOKEN = "EPST_Dark"
FISH_FULL_NAME = "Epinephelus striatus, Nassau grouper, dark phase"
FISH_TOKEN_2 = "KYSE_Normal"
FISH_FULL_NAME_2 = "Kyphosus sectatrix, Bermuda chub"
FISH_NAME_MAP = {FISH_TOKEN: FISH_FULL_NAME, FISH_TOKEN_2: FISH_FULL_NAME_2}


def write_text(path, text):
    """Write text to path as UTF-8, creating the parent folder."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def fish_results():
    """A detector results object over the two fish classes, both scored."""
    names = {0: FISH_TOKEN, 1: FISH_TOKEN_2}
    return FakeResults(names, box=FakeMetric([0, 1]),
                       nt_per_class=nt_for(names, {0: 12, 1: 7}))


def test_names_full_wins_over_registry_and_binomial():
    """A staged dataset's names beat the code registry, the binomial rule still resolves, and an unknown token stays a token."""
    ER.set_names_full({FISH_TOKEN: FISH_FULL_NAME, "SS": "A staged name for SS"})
    try:
        check(ER.class_full_name(FISH_TOKEN) == FISH_FULL_NAME,
              f"dataset name resolves a bare code: {ER.class_full_name(FISH_TOKEN)}")
        check(ER.class_display_name(FISH_TOKEN)
              == f"*Epinephelus striatus*, Nassau grouper, dark phase (`{FISH_TOKEN}`)",
              f"the report italicises the binomial and keeps the qualifiers upright: {ER.class_display_name(FISH_TOKEN)}")
        check(ER.class_display_name("Acropora-cervicornis") == "*Acropora cervicornis* (`Acropora-cervicornis`)",
              "a name that is a binomial on its own is still italic in full")
        check(ER.class_full_name("SS") == "A staged name for SS",
              f"the dataset wins over the code registry: {ER.class_full_name('SS')}")
        check(ER.class_full_name("Acropora-cervicornis") == "Acropora cervicornis",
              "binomial fallback still works")
        check(ER.class_full_name("ZZZZ") == "ZZZZ", "unknown token stays a token")
    finally:
        ER.set_names_full({})
    check(ER.class_full_name(FISH_TOKEN) == FISH_TOKEN,
          "clearing the map puts the token back")
    check(ER.class_full_name("SS") == "Siderastrea siderea",
          f"clearing the map puts the code registry back: {ER.class_full_name('SS')}")


def test_names_full_from_data_yaml():
    """names and names_full are joined on the class id, in every shape a data.yaml writes them."""
    root = tempfile.mkdtemp(prefix="t6_names_full_")
    try:
        mapped = write_text(os.path.join(root, "mapped", "data.yaml"),
                            "names:\n  0: A_N\n  1: B_TP\nnames_full:\n  0: Genus a\n")
        check(ER.names_full_from_data_yaml(mapped) == {"A_N": "Genus a"},
              f"joined on id, ids names does not carry are skipped: {ER.names_full_from_data_yaml(mapped)}")
        listed = write_text(os.path.join(root, "listed", "data.yaml"),
                            "names:\n  - A_N\n  - B_TP\nnames_full:\n  - Genus a\n  - Genus b\n")
        check(ER.names_full_from_data_yaml(listed) == {"A_N": "Genus a", "B_TP": "Genus b"},
              "a class map written as a list is read in id order")
        quoted = write_text(os.path.join(root, "quoted", "data.yaml"),
                            "names:\n  '0': A_N\nnames_full:\n  '0': Genus a\n")
        check(ER.names_full_from_data_yaml(quoted) == {"A_N": "Genus a"},
              "quoted class ids join the same way")
        blanks = write_text(os.path.join(root, "blanks", "data.yaml"),
                            "names:\n  0: A_N\n  1: B_TP\nnames_full:\n  0: '  '\n  1: '  Genus b  '\n")
        check(ER.names_full_from_data_yaml(blanks) == {"B_TP": "Genus b"},
              "a blank name is left out and a padded one is trimmed")
        plain = write_text(os.path.join(root, "plain", "data.yaml"), "names:\n  0: A_N\n")
        check(ER.names_full_from_data_yaml(plain) == {},
              "a dataset without names_full names nothing, which is how every older dataset reads")
        empty = write_text(os.path.join(root, "empty", "data.yaml"), "")
        check(ER.names_full_from_data_yaml(empty) == {}, "an empty data.yaml names nothing")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_dataset_without_names_full_is_never_refused_on_its_names_map():
    """A data.yaml that carries no names_full names nothing, whatever shape its names map has, because this key is optional context."""
    root = tempfile.mkdtemp(prefix="t6_names_optional_")
    try:
        for folder, text in [
            ("scalar_names", "names: seven\n"),
            ("bad_id", "names:\n  zero: A_N\n"),
            ("null_full", "names: seven\nnames_full:\n"),
        ]:
            path = write_text(os.path.join(root, folder, "data.yaml"), text)
            check(ER.names_full_from_data_yaml(path) == {},
                  f"{folder}: a dataset with no names_full names nothing rather than stopping the evaluation")
        # A dataset that does carry names_full is still refused on a damaged names map,
        # because then the two keys were written to be read together.
        carried = write_text(os.path.join(root, "carried", "data.yaml"),
                             "names: seven\nnames_full:\n  0: Genus a\n")
        try:
            ER.names_full_from_data_yaml(carried)
            raise AssertionError("a dataset carrying names_full is still refused on a damaged names map")
        except ValueError as exc:
            check(carried in str(exc), f"the refusal names the file: {exc}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_vertical_bar_in_a_name_stays_inside_its_markdown_cell():
    """A name carrying a vertical bar is escaped for the per-class table, so the row keeps its six columns."""
    ER.set_names_full({"BAR_Tok": "Genus species, a | b"})
    try:
        shown = ER.class_display_name("BAR_Tok")
        check(shown == "*Genus species*, a \\| b (`BAR_Tok`)",
              f"the vertical bar in the name is escaped: {shown}")
        metrics = {
            "class_names": {0: "BAR_Tok"},
            "per_class": {"BAR_Tok": {"box_precision": 0.5, "box_recall": 0.5,
                                      "box_map50": 0.5, "box_map50_95": 0.4}},
            "per_class_instances": {"BAR_Tok": 3},
            "absent_classes": [],
        }
        lines = ER._md_per_class_section(metrics, "box", "test")
        row = [ln for ln in lines if ln.startswith("| *Genus species*")]
        check(len(row) == 1, f"the per-class table holds one row for the class: {lines}")
        unescaped = row[0].replace("\\|", "")
        check(unescaped.count("|") == 7,
              f"the row still splits into six cells: {unescaped.count('|')} bars in {row[0]}")
    finally:
        ER.set_names_full({})


def test_names_full_from_data_yaml_refuses_a_malformed_file():
    """A missing file, a document that is not a mapping, a class map of the wrong type and a non-integer id are each refused by name."""
    root = tempfile.mkdtemp(prefix="t6_names_bad_")
    try:
        missing = os.path.join(root, "gone", "data.yaml")
        try:
            ER.names_full_from_data_yaml(missing)
            raise AssertionError("a missing data.yaml must be refused")
        except FileNotFoundError as exc:
            check(missing in str(exc), f"the error names the missing file: {exc}")
        cases = [
            ("list_doc", "- one\n- two\n", "does not hold a mapping"),
            ("scalar_names", "names: seven\nnames_full:\n  0: Genus a\n", "names"),
            ("scalar_full", "names:\n  0: A_N\nnames_full: seven\n", "names_full"),
            ("bad_id", "names:\n  0: A_N\nnames_full:\n  zero: Genus a\n", "not an integer"),
        ]
        for folder, text, needle in cases:
            path = write_text(os.path.join(root, folder, "data.yaml"), text)
            try:
                ER.names_full_from_data_yaml(path)
                raise AssertionError(f"{folder} must be refused, not read as empty")
            except ValueError as exc:
                check(path in str(exc) and needle in str(exc),
                      f"{folder}: the error names the file and what is wrong ({exc})")
        huge = write_text(os.path.join(root, "huge", "data.yaml"),
                          "names:\n" + "".join(f"  {i}: T{i}\n" for i in range(5000))
                          + "names_full:\n" + "".join(f"  {i}: Genus {i}\n" for i in range(5000)))
        joined = ER.names_full_from_data_yaml(huge)
        check(len(joined) == 5000 and joined["T4999"] == "Genus 4999",
              f"five thousand classes join without loss: {len(joined)}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_metrics_name_the_classes_only_when_they_are_codes():
    """metrics carry class_full_names for tokens that are codes, and no such key when every token is already its own name."""
    ER.set_names_full(FISH_NAME_MAP)
    try:
        fish = ER.extract_metrics(val_out_for(fish_results()))
        check(fish["class_full_names"] == {0: FISH_FULL_NAME, 1: FISH_FULL_NAME_2},
              f"the fish classes carry their full names by class id: {fish.get('class_full_names')}")
    finally:
        ER.set_names_full({})
    registry_names = {0: "SS", 1: "PA"}
    coded = ER.extract_metrics(val_out_for(
        FakeResults(registry_names, box=FakeMetric([0, 1]),
                    nt_per_class=nt_for(registry_names, {0: 3, 1: 4}))))
    check(coded["class_full_names"] == {0: "Siderastrea siderea", 1: "Porites astreoides"},
          f"a code the module's registry knows is named too: {coded.get('class_full_names')}")
    binomials = ER.extract_metrics(val_out_for(detector_results()))
    check("class_full_names" not in binomials,
          "a dataset whose tokens are binomials writes no names map: every reader expands them already")
    plain_names = {0: "sand", 1: "rubble"}
    plain = ER.extract_metrics(val_out_for(
        FakeResults(plain_names, box=FakeMetric([0, 1]),
                    nt_per_class=nt_for(plain_names, {0: 1, 1: 2}))))
    check("class_full_names" not in plain,
          "tokens that resolve to nothing write no new key, so an older run's file is unchanged")


def test_main_writes_class_full_names_from_the_dataset():
    """main reads the evaluated dataset's names_full and metrics.json carries the names; a dataset without them writes no key."""
    root = tempfile.mkdtemp(prefix="t6_main_names_")
    saved = {k: getattr(ER, k) for k in ("run_val", "render_previews", "build_pdf")}
    saved_argv = sys.argv
    try:
        run_dir = os.path.join(root, "runs", "run_fish")
        os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
        write_text(os.path.join(run_dir, "args.yaml"), "model: yolo11l.pt\ntask: detect\n")
        named = os.path.join(root, "dataset_named")
        write_text(os.path.join(named, "data.yaml"),
                   f"path: .\nnames:\n  0: {FISH_TOKEN}\n  1: {FISH_TOKEN_2}\n"
                   f"names_full:\n  0: {FISH_FULL_NAME}\n  1: {FISH_FULL_NAME_2}\n")
        plain = os.path.join(root, "dataset_plain")
        write_text(os.path.join(plain, "data.yaml"),
                   f"path: .\nnames:\n  0: {FISH_TOKEN}\n  1: {FISH_TOKEN_2}\n")

        ER.render_previews = lambda *a, **k: []
        ER.build_pdf = lambda *a, **k: None

        def read_metrics(dataset_dir, out_name):
            """Run main over one dataset dir with a fabricated val pass and return its metrics.json."""
            out_dir = os.path.join(root, out_name)
            ER.run_val = lambda args: {"results": fish_results(), "weights": os.path.join(run_dir, "weights", "best.pt"),
                                       "data_yaml": os.path.join(dataset_dir, "data.yaml"), "wall_seconds": 1.0,
                                       "model": None, "val_dir": args.out_dir}
            sys.argv = ["evaluate_run.py", "--run_dir", run_dir, "--dataset_dir", dataset_dir,
                        "--out_dir", out_dir, "--split", "test"]
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                code = ER.main()
            check(code == ER.EXIT_OK, f"{out_name}: main must exit {ER.EXIT_OK}, got {code}")
            with open(os.path.join(out_dir, "metrics.json"), encoding="utf-8") as fh:
                return json.load(fh)["metrics"], buf.getvalue()

        named_metrics, named_out = read_metrics(named, "out_named")
        check(named_metrics["class_full_names"] == {"0": FISH_FULL_NAME, "1": FISH_FULL_NAME_2},
              f"metrics.json carries the dataset's names by class id: {named_metrics.get('class_full_names')}")
        check("2 class names" in named_out,
              f"the console says how many names the dataset supplied: {named_out[:400]!r}")
        plain_metrics, _ = read_metrics(plain, "out_plain")
        check("class_full_names" not in plain_metrics,
              "a dataset without names_full leaves metrics.json as it was before this key existed")
    finally:
        for k, v in saved.items():
            setattr(ER, k, v)
        sys.argv = saved_argv
        ER.set_names_full({})
        shutil.rmtree(root, ignore_errors=True)


def test_names_full_reads_what_the_split_builder_writes():
    """The data.yaml the split builder writes from a staged names table is read back by the evaluator unchanged."""
    import detect_dataset as DD
    root = tempfile.mkdtemp(prefix="t6_writer_reader_")
    try:
        source = os.path.join(root, "staged")
        write_text(os.path.join(source, "classes.txt"),
                   f"{FISH_TOKEN}\n{FISH_TOKEN_2}\nUnnamed_Token\n")
        # The real table (scripts/_boxes/names/fish_photo_28feb25.csv) quotes every
        # name, because a fish name carries commas; an unquoted name would be cut
        # at its first comma by any CSV reader.
        write_text(os.path.join(source, "names.csv"),
                   f'token,name\n{FISH_TOKEN},"{FISH_FULL_NAME}"\n{FISH_TOKEN_2},"{FISH_FULL_NAME_2}"\n')
        doc = DD._data_yaml_doc(os.path.join(root, "dataset"), source)
        data_yaml = os.path.join(root, "dataset", "data.yaml")
        os.makedirs(os.path.dirname(data_yaml), exist_ok=True)
        with open(data_yaml, "w", encoding="utf-8") as fh:
            yaml.safe_dump(doc, fh, sort_keys=False)
        joined = ER.names_full_from_data_yaml(data_yaml)
        check(joined == {FISH_TOKEN: FISH_FULL_NAME, FISH_TOKEN_2: FISH_FULL_NAME_2},
              f"the evaluator reads back exactly the names the split builder wrote: {joined}")
        ER.set_names_full(joined)
        try:
            check(ER.class_display_html(FISH_TOKEN)
                  == f"<i>Epinephelus striatus</i>, Nassau grouper, dark phase "
                     f"<span class=\"mono\">({FISH_TOKEN})</span>",
                  f"the report's HTML italicises the binomial alone: {ER.class_display_html(FISH_TOKEN)}")
            check(ER.class_full_name("Unnamed_Token") == "Unnamed_Token",
                  "a token the names table does not carry stays a token")
        finally:
            ER.set_names_full({})
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_names_setters_refuse_the_wrong_type():
    """set_names_full and class_full_names_for refuse anything that is not a mapping, naming what arrived."""
    for bad in (["EPST_Dark", FISH_FULL_NAME], "EPST_Dark", 7):
        try:
            ER.set_names_full(bad)
            ER.set_names_full({})
            raise AssertionError(f"set_names_full must refuse {type(bad).__name__}")
        except TypeError as exc:
            check(type(bad).__name__ in str(exc), f"set_names_full names the type it got: {exc}")
        try:
            ER.class_full_names_for(bad)
            raise AssertionError(f"class_full_names_for must refuse {type(bad).__name__}")
        except TypeError as exc:
            check(type(bad).__name__ in str(exc), f"class_full_names_for names the type it got: {exc}")
    check(ER.class_full_names_for(None) == {} and ER.class_full_names_for({}) == {},
          "no classes name nothing")
    ER.set_names_full({"  ": "blank token", FISH_TOKEN: "   "})
    try:
        check(ER.class_full_name(FISH_TOKEN) == FISH_TOKEN,
              "a blank name is dropped, so the token is shown instead of an empty species")
    finally:
        ER.set_names_full({})


# ---------------------------------------------------------------------------
# The subject noun: what the explanatory prose calls the thing being scored
#
# This module was written for TCRMP coral photographs, so every explanation
# said "coral". The fish detector evaluated on 2026-09-15 02:43 AST scored 34
# fish species and colour phases, and its ten-page report still told the
# reader it was about corals 42 times over. The prose now uses a neutral noun
# that is true of any subject a run is trained on, and a dataset may declare
# its own noun in its data.yaml.
# ---------------------------------------------------------------------------

# Three of the fish detector's real class tokens and full names, from
# inprocess/_models/fish_photo_28feb25_paper/dataset/data.yaml.
FISH_NAMES = {0: "EPST_Dark", 1: "PTVO_Normal", 2: "GICI_Normal"}
FISH_NAMES_FULL = {
    "EPST_Dark": "Epinephelus striatus, Nassau grouper, dark phase",
    "PTVO_Normal": "Pterois volitans, red lionfish",
    "GICI_Normal": "Ginglymostoma cirratum, nurse shark",
}
CORAL_NAMES = {0: "Acropora-cervicornis", 1: "Orbicella-annularis", 2: "Porites-astreoides"}

# Sentences copied verbatim out of the fish detector's report on disk at
# inprocess/_models/_eval/run3_heavy_augment_20260915_024311/report.md. None
# of them may appear in a report for a run whose subjects are not corals.
CORAL_CLAIMS_FROM_THE_FISH_REPORT = (
    "Recall (R) is the fraction of the real corals the model finds",
    "the model only has to place a box roughly over each coral",
    "The fraction of the real corals (ground-truth instances) that the model found",
    "If 100 corals exist and the model locates 80",
    "drawn tightly around each coral",
    "80 match a real coral",
    "averaged across all coral classes",
    "the model is not reliably detecting corals at all",
    "even when it does find a coral",
    "the model finds corals but draws their outlines poorly",
    "duplicate detections of the same coral",
    "broken out by coral species",
)

# Today's coral segmentation report, captured out of evaluate_run.py at
# 2026-09-15 04:05 AST, before the subject noun existed, and re-pinned at
# 2026-09-15 20:40 AST for the two threshold paragraphs (the samples are drawn
# at the display threshold, the score is measured at 0.001). The fixtures that
# reproduce it are CORAL_NAMES, fake_ctx(), fake_args() and one preview;
# the generated-on stamp is the only volatile field and is masked.
CORAL_MARKDOWN_BEFORE_THE_FIX = """\
# Evaluation report for run `run1`

This report was generated on GENERATED_STAMP. It evaluates the model on the 'test' split, and the full evaluation took 2s of wall-clock time. The per-class scores below are outline (mask) scores.

## Top-line metrics

Precision (P) is the fraction of the model's predictions that are correct. Recall (R) is the fraction of the real corals the model finds. Mean average precision at intersection-over-union 0.5 (mAP50) is the lenient detection score; averaged from 0.5 to 0.95 (mAP50-95) is the strict score. Box columns score the model's predicted rectangles; mask columns score its pixel outlines.

| Metric | Box | Mask | Good range |
|---|---|---|---|
| mAP50 | 0.301 | 0.301 | <0.35 poor | 0.35-0.55 weak | 0.55-0.75 good | 0.75-0.95 very good | >0.95 excellent |
| mAP50-95 | 0.401 | 0.401 | <0.2 poor | 0.2-0.4 weak | 0.4-0.6 good | 0.6-0.8 very good | >0.8 excellent |
| Precision | 0.101 | 0.101 | <0.5 poor | 0.5-0.7 OK | 0.7-0.85 good | >0.85 excellent |
| Recall | 0.201 | 0.201 | <0.5 poor | 0.5-0.7 OK | 0.7-0.85 good | >0.85 excellent |

## Training context

- model=yolo11n.pt, imgsz=256, batch=-1
- epochs_requested=20, epochs_completed=20, best_epoch=18
- splits: train=10 valid=2 test=2
- eval_args: conf=0.001 iou=0.6 imgsz=256

## Per-class (outline (mask) scores)

Each row scores one class on its outline (mask) predictions. Instances counts the ground-truth objects of that class in the 'test' split (source: validator nt_per_class). A class with no ground truth in this split cannot be scored and is marked 'not present in this split'; that is not a score of zero.

| Class | Instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| *Acropora cervicornis* (`Acropora-cervicornis`) | 4 | 0.100 | 0.200 | 0.300 | 0.400 |
| *Orbicella annularis* (`Orbicella-annularis`) | 5 | 0.101 | 0.201 | 0.301 | 0.401 |
| *Porites astreoides* (`Porites-astreoides`) | 6 | 0.102 | 0.202 | 0.302 | 0.402 |

## Prediction samples (1 images)

Each image is shown with the human-drawn ground-truth (GT) outlines on the left and the model's predicted outlines on the right.

These samples are drawn at a display threshold of 0.25, so the right panel shows the predictions an operator would act on. The scores above were measured at 0.001, which keeps the whole precision-recall curve; the two numbers answer different questions and are not expected to match.

- **x.jpg**: the human labeler drew 2 ground-truth instances; the model produced 3 detections.
  ![x.jpg](x.jpg)

## Glossary

Every term and abbreviation used above is defined below as a complete sentence. The 'Good' column gives a healthy band for each metric on this kind of data; the 'Red flag' column names the symptom you see when it is too low.

| Term | Definition | Good | Red flag |
|---|---|---|---|
| Ground truth (GT) | The human-drawn labels the model is being compared against; the 'correct answer' the model is graded on. | Good ground-truth labels are consistent across images and drawn tightly around each coral. | Noisy or inconsistent ground-truth labels cause every metric below to look worse than the model actually is. |
| Intersection over union (IoU) | The ratio of the area where a prediction and the ground-truth region overlap to the area they jointly cover; values run from 0 (no overlap) to 1 (pixel-perfect match). | An IoU of 0.5 or higher counts as a loose match; 0.75 or higher counts as a strict match. | If the typical IoU is below 0.3, the model is putting predictions in the wrong part of the image. |
| Precision (P) | The fraction of the model's predictions that are correct. If the model makes 100 predictions and 80 match a real coral, precision is 0.80. | A precision above 0.85 is excellent; 0.70 to 0.85 is good. | A precision below 0.5 means the model raises too many false alarms to be trusted without filtering. |
| Recall (R) | The fraction of the real corals (ground-truth instances) that the model found. If 100 corals exist and the model locates 80, recall is 0.80. | A recall above 0.85 is excellent; 0.70 to 0.85 is good. | A recall below 0.5 means the model is missing more real corals than it finds. |
| mAP50 | The area under the precision-recall curve when a prediction is considered correct at an IoU of 0.5 or higher, averaged across all coral classes. It is the lenient detection score. | An mAP50 between 0.55 and 0.75 is good; above 0.75 is very good. | An mAP50 below 0.35 means the model is not reliably detecting corals at all. |
| mAP50-95 | The mean average precision computed at ten IoU thresholds from 0.5 to 0.95 in steps of 0.05, then averaged. It is the strict score because it rewards tight, accurate localization. | An mAP50-95 between 0.4 and 0.6 is good; above 0.6 is very good. | An mAP50-95 below 0.2 means the model's boundaries are very loose even when it does find a coral. |
| Box vs. mask metrics | Box metrics judge the model by the overlap of predicted and true bounding rectangles. Mask metrics judge it by the overlap of predicted and true pixel outlines; mask metrics are strictly harder. | Mask scores are expected to be slightly lower than box scores on the same run. | If mask scores are much lower than box scores, the model finds corals but draws their outlines poorly. |
| Confidence threshold | The minimum score a prediction needs to be kept. This evaluation keeps almost every prediction, because mean average precision is the area under the whole precision-recall curve and a higher cut-off removes its tail. | 0.001 is the right value for a measurement, and is what the ultralytics validator uses when none is given; the inference step uses a high threshold instead, where only confident calls are wanted. | A score measured above about 0.01 reads lower than the weights earn, and cannot be ranked against a score measured at 0.001. |
| Non-maximum suppression (NMS) IoU | A cleanup rule: when two predictions of the same class overlap by more than this IoU, only the higher-confidence one is kept. | Values between 0.6 and 0.7 are standard. | A value set too low leaves duplicate detections of the same coral on the output. |
| Fitness | Ultralytics's internal checkpoint-selector score, a weighted combination of mAP values used to pick which epoch to save as 'best.pt'. | Useful for comparing two runs of the same architecture on the same data. | Not directly comparable across different model architectures. |
| Confidence | The model's self-reported certainty for each prediction, expressed as a number between 0 (no confidence) and 1 (maximum confidence). Predictions whose confidence score falls below the chosen threshold are discarded before the rest of the metrics are computed. | The scores in this report were measured at 0.001, so the whole precision-recall curve is counted; the F1-vs-confidence curves on pages 3 and 4 show the value to deploy this run at. | If every prediction is close to 1.0 the model is overconfident; if every prediction sits near 0.5 the model has not learned to distinguish confident calls from guesses. |
| F1 score | The harmonic mean (balanced average) of precision and recall. It is computed as 2 x P x R / (P + R) and rewards models only when both precision and recall are high together; a model with precision 0.9 but recall 0.1 earns an F1 near 0.18, not 0.5. | An F1 above 0.75 is good; above 0.85 is excellent. | An F1 below 0.5 means the model cannot be relied on without substantial manual correction. |

_Weights_: `/w/best.pt`  _Data_: `/d/data.yaml`"""

# Every difference the subject noun is allowed to make to that report. The
# left side of each pair is the pre-edit wording above; the right side is what
# the neutral noun renders. A coral run that changes in any other way is a
# regression in its wording, and the equality check below catches it.
SUBJECT_NOUN_SWAP = (
    ("the fraction of the real corals the model finds",
     "the fraction of the real objects the model finds"),
    ("drawn tightly around each coral",
     "drawn tightly around each object"),
    ("80 match a real coral",
     "80 match a real object"),
    ("The fraction of the real corals (ground-truth instances)",
     "The fraction of the real objects (ground-truth instances)"),
    ("If 100 corals exist",
     "If 100 objects exist"),
    ("missing more real corals than it finds",
     "missing more real objects than it finds"),
    ("averaged across all coral classes",
     "averaged across all classes"),
    ("the model is not reliably detecting corals at all",
     "the model is not reliably detecting objects at all"),
    ("even when it does find a coral",
     "even when it does find a real object"),
    ("the model finds corals but draws their outlines poorly",
     "the model finds objects but draws their outlines poorly"),
    ("duplicate detections of the same coral",
     "duplicate detections of the same object"),
)

GENERATED_STAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} AST")


def first_difference(got, expected):
    """A readable description of where two report texts first disagree."""
    for i, (a, b) in enumerate(zip(got, expected)):
        if a != b:
            return f"at character {i}: got ...{got[max(0, i - 60):i + 60]!r}, expected ...{expected[max(0, i - 60):i + 60]!r}"
    if len(got) != len(expected):
        return f"lengths differ: got {len(got)} characters, expected {len(expected)}"
    return "no difference"


def wording_metrics(names, seg):
    """A metrics dict for three classes; segmentation when seg is true."""
    counts = {0: 4, 1: 5, 2: 6}
    kwargs = {"box": FakeMetric([0, 1, 2]), "nt_per_class": nt_for(names, counts)}
    if seg:
        kwargs["seg"] = FakeMetric([0, 1, 2])
    return ER.extract_metrics(val_out_for(FakeResults(names, **kwargs)))


def wording_previews(out_dir):
    """One preview record whose overlay sits directly in out_dir."""
    return [{"filename": "x.jpg", "overlay": os.path.join(out_dir, "x.jpg"),
             "n_gt": 2, "n_pred": 3}]


def render_every_page(metrics, ctx, args, previews):
    """Every report builder's output for one run, joined into one string.

    Parameters:
        metrics: extract_metrics() output.
        ctx: a context dict shaped like gather_context()'s.
        args: the argparse namespace the builders read.
        previews: render_previews()-shaped records.
    Returns:
        The markdown report, every PDF section and the stylesheet, joined.
    """
    family = ER.metric_family(metrics)
    plan = ER._plan_pages(len(metrics["per_class"]) + len(metrics["absent_classes"]), len(previews))
    parts = [
        ER.build_report_md(metrics, ctx, previews, args),
        ER._render_cover(metrics, ctx, args, [("Cover", 1)], "2026-09-15 10:00 AST", "run1"),
        ER._render_training(metrics, ctx, args, {}, plan["training"]),
        ER._render_metric_page("box", plan["box"], metrics, ctx, args, {}),
        ER._render_metric_page("mask", plan["mask"], metrics, ctx, args, {}),
        "".join(ER._render_perclass(metrics, args, {}, plan)),
        ER._render_confusion(metrics, ctx, args, {}, plan["confusion"]),
        "".join(ER._render_samples(previews, args, plan["confusion"] + 1, family)[0]),
        ER._render_glossary(metrics, ctx, args, plan["glossary"], "2026-09-15 10:00 AST"),
        ER._build_css(),
    ]
    return "\n".join(parts)


def test_a_detector_report_never_calls_its_subjects_corals():
    """A box detector trained on fish is described with a neutral noun: no
    page calls its subjects corals, and no page calls its boxes outlines."""
    tmp = tempfile.mkdtemp(prefix="eval_metrics_fish_")
    try:
        ER.set_names_full(FISH_NAMES_FULL)
        ER.set_subject_noun(None)
        m = wording_metrics(FISH_NAMES, seg=False)
        args = fake_args(tmp)
        text = render_every_page(m, fake_ctx(), args, wording_previews(tmp))

        offenders = [ln for ln in text.splitlines() if "coral" in ln.lower()][:3]
        check("coral" not in text.lower(),
              f"a fish detector's report must not say coral anywhere: {offenders}")
        for claim in CORAL_CLAIMS_FROM_THE_FISH_REPORT:
            check(claim not in text, f"the report still carries the coral claim {claim!r}")
        # The sentences were reworded, not deleted.
        check("the fraction of the real objects the model finds" in text,
              "the top-line prose still explains recall, with the neutral noun")
        check("place a box roughly over each object" in text,
              "the box page still explains the lenient view, with the neutral noun")
        # The top-line table has a Box column only, so the prose above it
        # must not tell the reader how to read a Mask column.
        check("mask columns score" not in text,
              "a detector's top-line prose describes a mask column its table has not got")
        # The subjects are still named where the dataset names them.
        check("Pterois volitans" in text and "red lionfish" in text,
              "the per-class rows still name the species the dataset declares")
        # An article never disagrees with a noun that starts with a vowel.
        check(" a object" not in text and " a objects" not in text and " an objects" not in text,
              "an article disagrees with the neutral noun somewhere in the report")
        # No unfilled placeholder reaches a reader.
        check("{subject" not in text, "an unfilled subject placeholder reached the report")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        ER.set_names_full({})
        ER.set_subject_noun(None)


def test_a_detector_is_never_described_as_drawing_outlines():
    """Outline wording belongs to a segmentation run: a detector's sample
    page, mask page and training page all say box instead."""
    tmp = tempfile.mkdtemp(prefix="eval_metrics_boxwords_")
    try:
        ER.set_subject_noun(None)
        args = fake_args(tmp)
        det = wording_metrics(FISH_NAMES, seg=False)
        seg = wording_metrics(CORAL_NAMES, seg=True)

        det_samples = "".join(ER._render_samples(wording_previews(tmp), args, 8, ER.METRIC_FAMILY_BOX)[0])
        check("outline" not in det_samples.lower(),
              f"a detector's sample page must talk about boxes, not outlines: {det_samples}")
        check("ground-truth (GT) boxes" in det_samples,
              f"a detector's sample page names the ground truth as boxes: {det_samples}")
        seg_samples = "".join(ER._render_samples(wording_previews(tmp), args, 8, ER.METRIC_FAMILY_MASK)[0])
        check("ground-truth (GT) outlines" in seg_samples,
              f"a segmentation run's sample page still names outlines: {seg_samples}")

        det_mask = ER._render_metric_page("mask", 4, det, fake_ctx(), args, {})
        check(ER.DETECTOR_NO_MASK_NOTE in det_mask,
              "the detector's mask page still says why it is empty")
        check("has to trace each" not in det_mask,
              f"the detector's mask page must not describe tracing outlines: {det_mask}")
        check("pixel-level outlines" not in det_mask,
              f"the detector's mask page must not explain outline curves it has none of: {det_mask}")
        # The page says there is nothing to score, so it must not go on to
        # head two curve sections and call their figures unavailable: an
        # outline curve is not missing for a detector, it does not exist.
        check("Precision-recall curve" not in det_mask,
              f"the detector's mask page heads a curve section it can never fill: {det_mask}")
        check("F1 vs confidence curve" not in det_mask,
              f"the detector's mask page heads an F1 section it can never fill: {det_mask}")
        check("unavailable" not in det_mask,
              f"unavailable tells the reader a plot file went missing: {det_mask}")
        seg_mask = ER._render_metric_page("mask", 4, seg, fake_ctx(), args, {})
        check("has to trace each" in seg_mask,
              f"a segmentation run's mask page keeps the strict view: {seg_mask}")
        check(ER.DETECTOR_NO_MASK_NOTE not in seg_mask,
              "a segmentation run's mask page keeps its interpretation")
        check("Precision-recall curve" in seg_mask and "F1 vs confidence curve" in seg_mask,
              f"a segmentation run's mask page keeps both curve sections: {seg_mask}")

        det_train = ER._render_training(det, fake_ctx(), args, {}, 2)
        check("seg (how well masks fit)" not in det_train,
              f"a detector tracks no segmentation loss: {det_train}")
        # The figure caption under the loss curves is chosen by family too,
        # or page 2 of a detector's report contradicts its own subtitle.
        check("segmentation (mask) loss" not in det_train,
              f"a detector's loss caption must not name a mask loss: {det_train}")
        check("four training losses" not in det_train,
              f"a detector records three training losses, not four: {det_train}")
        check("three training losses" in det_train,
              f"a detector's loss caption still names the losses it does record: {det_train}")
        seg_train = ER._render_training(seg, fake_ctx(), args, {}, 2)
        check("seg (how well masks fit)" in seg_train,
              f"a segmentation run still lists its segmentation loss: {seg_train}")
        check("four training losses" in seg_train and "segmentation (mask) loss" in seg_train,
              f"a segmentation run's loss caption is unchanged: {seg_train}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        ER.set_subject_noun(None)


def test_a_coral_report_changes_only_the_subject_noun():
    """A coral segmentation run renders exactly the report it rendered before
    this change, with the subject noun swapped in and nothing else moved."""
    tmp = tempfile.mkdtemp(prefix="eval_metrics_coral_")
    try:
        ER.set_names_full({})
        ER.set_subject_noun(None)
        m = wording_metrics(CORAL_NAMES, seg=True)
        md = ER.build_report_md(m, fake_ctx(), wording_previews(tmp), fake_args(tmp))
        got = GENERATED_STAMP_RE.sub("GENERATED_STAMP", md)

        expected = CORAL_MARKDOWN_BEFORE_THE_FIX
        for before, after in SUBJECT_NOUN_SWAP:
            check(before in expected, f"the frozen report does not carry {before!r}")
            expected = expected.replace(before, after)
        check(got == expected,
              "a coral run's report changed beyond the subject noun, "
              + first_difference(got, expected))
        check("*Acropora cervicornis* (`Acropora-cervicornis`)" in got,
              "the per-class table still names each coral species in full")
        check("outline (mask) scores" in got,
              "the segmentation run still reads as an outline run")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        ER.set_subject_noun(None)


def test_a_dataset_can_declare_the_noun_for_its_own_subjects():
    """data.yaml's optional subject key names the subjects in the prose; a
    malformed or hostile value is refused and the neutral noun stands."""
    tmp = tempfile.mkdtemp(prefix="eval_metrics_subject_")
    try:
        check(ER.subject_nouns() == ER.SUBJECT_NOUN_DEFAULT,
              f"the neutral noun stands until a dataset declares one: {ER.subject_nouns()}")

        def write(doc):
            """Write one data.yaml under tmp and return its path."""
            path = os.path.join(tmp, "data.yaml")
            with open(path, "w", encoding="utf-8") as fh:
                yaml.dump(doc, fh)
            return path

        base = {"path": tmp, "names": {0: "A"}}
        check(ER.subject_noun_from_data_yaml(write(base)) is None,
              "a dataset that declares nothing gets no noun")
        pair = ER.subject_noun_from_data_yaml(
            write({**base, "subject": {"singular": "fish", "plural": "fish"}}))
        check(pair == ("fish", "fish"), f"both declared forms are used verbatim: {pair}")

        # A bare string is refused rather than pluralised by guesswork, which
        # would write "fishs" into every page of a fish report.
        for bad in ("fish", 123, [], {"singular": "fish"}, {"plural": "fish"},
                    {"singular": "  ", "plural": "fish"},
                    {"singular": "<b>fish</b>", "plural": "fish"},
                    {"singular": "f" * 100, "plural": "fish"},
                    {"singular": "fish", "plural": 7}):
            raised = False
            try:
                ER.subject_noun_from_data_yaml(write({**base, "subject": bad}))
            except ValueError:
                raised = True
            check(raised, f"subject {bad!r} must be refused, not rendered into the report")

        ER.set_subject_noun(("fish", "fish"))
        md = ER.build_report_md(wording_metrics(FISH_NAMES, seg=False), fake_ctx(), [], fake_args(tmp))
        check("the fraction of the real fish the model finds" in md,
              f"the declared noun reaches the metric explanations: {md[:400]}")
        check("object" not in md, f"the declared noun replaces the neutral one everywhere: {md[:400]}")

        ER.set_subject_noun(None)
        check(ER.subject_nouns() == ER.SUBJECT_NOUN_DEFAULT, "clearing restores the neutral noun")
        for bad in ("fish", ("fish",), ("fish", 1), ("", "fish"), 7):
            raised = False
            try:
                ER.set_subject_noun(bad)
            except (TypeError, ValueError):
                raised = True
            check(raised, f"set_subject_noun must refuse {bad!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        ER.set_subject_noun(None)


def test_a_detector_glossary_matches_the_pages_it_points_at():
    """A detector's glossary defines the mask rows its cover prints as N/A
    without claiming the run drew outlines, and sends the reader to the one
    page that carries an F1 curve. A segmentation run's glossary is the one
    shipped today, word for word."""
    tmp = tempfile.mkdtemp(prefix="eval_metrics_glossary_")
    try:
        ER.set_names_full({})
        ER.set_subject_noun(None)
        args = fake_args(tmp)
        det = wording_metrics(FISH_NAMES, seg=False)
        seg = wording_metrics(CORAL_NAMES, seg=True)
        stamp = "2026-09-15 10:00 AST"
        det_text = (ER.build_report_md(det, fake_ctx(), [], args)
                    + ER._render_glossary(det, fake_ctx(), args, 9, stamp))
        seg_text = (ER.build_report_md(seg, fake_ctx(), [], args)
                    + ER._render_glossary(seg, fake_ctx(), args, 9, stamp))

        check("pages 3 and 4" not in det_text,
              "a detector's glossary sends the reader to page 4, which has nothing to show")
        check("the F1-vs-confidence curve on page 3" in det_text,
              f"a detector's glossary names the page its F1 curve is on: {det_text[-3000:]}")
        check("draws their outlines poorly" not in det_text,
              "a detector's glossary must not describe outlines the run never drew")
        check(ER.DETECTOR_MASK_GLOSSARY_NOTE in det_text,
              f"a detector's glossary defines the mask rows its cover prints as N/A: {det_text[-3000:]}")
        # Every glossary row still reaches both readers.
        check(det_text.count("Box vs. mask metrics") == 2 and seg_text.count("Box vs. mask metrics") == 2,
              "the row is reworded for a detector, never dropped from one of the two renderers")

        check("pages 3 and 4" in seg_text,
              "a segmentation run's glossary still names both curve pages")
        check("the model finds objects but draws their outlines poorly" in seg_text,
              "a segmentation run keeps the box-versus-mask red flag it has today")
        check(ER.DETECTOR_MASK_GLOSSARY_NOTE not in seg_text,
              "a segmentation run's glossary says nothing about a detector")
        check(ER.glossary_rows(ER.METRIC_FAMILY_MASK) == list(ER.GLOSSARY_TABLE),
              "a segmentation run reads the glossary exactly as it is written")
        # A family the module does not know is refused rather than guessed at,
        # because a guessed glossary tells the reader the wrong thing.
        for bad in ("segment", "", None, 7):
            raised = False
            try:
                ER.glossary_rows(bad)
            except ValueError:
                raised = True
            check(raised, f"glossary_rows must refuse the family {bad!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        ER.set_names_full({})
        ER.set_subject_noun(None)


def test_a_rebuilt_dataset_keeps_the_noun_its_source_declares():
    """A split dataset is written from scratch every time it is built, so a
    subject key added to its data.yaml by hand is erased on the next build.
    The evaluator therefore also reads the staged source that
    split_manifest.json records, which no rebuild rewrites."""
    tmp = tempfile.mkdtemp(prefix="eval_metrics_source_subject_")
    try:
        dataset = os.path.join(tmp, "dataset")
        source = os.path.join(tmp, "staged_source")
        os.makedirs(dataset, exist_ok=True)
        os.makedirs(source, exist_ok=True)

        def write_yaml(folder, doc):
            """Write one data.yaml into folder and return its path."""
            path = os.path.join(folder, "data.yaml")
            with open(path, "w", encoding="utf-8") as fh:
                yaml.dump(doc, fh)
            return path

        def write_manifest(doc):
            """Write the dataset's split_manifest.json from a dict."""
            with open(os.path.join(dataset, ER.SPLIT_MANIFEST_NAME), "w", encoding="utf-8") as fh:
                json.dump(doc, fh)

        def write_raw_manifest(text):
            """Write split_manifest.json verbatim, valid JSON or not."""
            with open(os.path.join(dataset, ER.SPLIT_MANIFEST_NAME), "w", encoding="utf-8") as fh:
                fh.write(text)

        split_yaml = write_yaml(dataset, {"path": dataset, "names": {0: "A"}})
        write_yaml(source, {"names": {0: "A"},
                            "subject": {"singular": "fish", "plural": "fish"}})

        check(ER.subject_noun_for_dataset(dataset, split_yaml) is None,
              "a dataset whose folder holds no manifest declares nothing")

        write_manifest({"source": source})
        check(ER.subject_noun_for_dataset(dataset, split_yaml) == ("fish", "fish"),
              "the staged source's declaration survives a rebuild of the split dataset")

        # The split dataset's own declaration is the nearer one and wins.
        own = write_yaml(dataset, {"path": dataset, "names": {0: "A"},
                                   "subject": {"singular": "coral", "plural": "corals"}})
        check(ER.subject_noun_for_dataset(dataset, own) == ("coral", "corals"),
              "a split dataset that declares its own noun keeps it")
        write_yaml(dataset, {"path": dataset, "names": {0: "A"}})

        # A source that has moved away leaves the neutral noun standing.
        write_manifest({"source": os.path.join(tmp, "moved_away")})
        check(ER.subject_noun_for_dataset(dataset, split_yaml) is None,
              "a source folder no longer on disk is not a declaration and not a crash")

        # A source folder that holds no data.yaml declares nothing either.
        bare = os.path.join(tmp, "bare_source")
        os.makedirs(bare, exist_ok=True)
        write_manifest({"source": bare})
        check(ER.subject_noun_for_dataset(dataset, split_yaml) is None,
              "a staged source with no data.yaml declares no subject")

        # A manifest that records no source, or records a value that is not a
        # path, is read past rather than argued with.
        for doc in ({"holdout_mode": "transect"}, {"source": 7}, {"source": ""}, []):
            write_manifest(doc)
            check(ER.subject_noun_for_dataset(dataset, split_yaml) is None,
                  f"a manifest of {doc!r} declares no subject")
        write_raw_manifest("{not json at all")
        check(ER.subject_noun_for_dataset(dataset, split_yaml) is None,
              "an unreadable manifest declares no subject and does not stop the evaluation")

        # A declaration the operator did write, but wrote badly, is refused by
        # name instead of reaching the report as markup.
        write_yaml(source, {"names": {0: "A"},
                            "subject": {"singular": "<b>fish</b>", "plural": "fish"}})
        write_manifest({"source": source})
        raised = ""
        try:
            ER.subject_noun_for_dataset(dataset, split_yaml)
        except ValueError as exc:
            raised = str(exc)
        check(source in raised,
              f"a malformed declaration in the staged source is refused by name, got {raised!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        ER.set_subject_noun(None)


def test_split_counts_read_every_image_type_the_trainer_reads():
    """gather_context counts a split with the same image-type list the scorer
    uses, so the report's splits line and metrics.json agree with the ledger
    instead of dropping WebP and the other less common formats."""
    tmp = tempfile.mkdtemp(prefix="eval_metrics_splitcount_")
    try:
        dataset = os.path.join(tmp, "dataset")
        expected = {}
        for split, names in (("train", ("a.jpg", "b.webp", "c.avif")),
                             ("valid", ("d.png", "e.heic")),
                             ("test", ("f.jpeg", "g.jp2", "h.mpo"))):
            img_dir = os.path.join(dataset, split, "images")
            os.makedirs(img_dir, exist_ok=True)
            for nm in names:
                with open(os.path.join(img_dir, nm), "wb") as fh:
                    fh.write(b"\xff\xd8\xff\xd9")
            # Neither a label file nor a folder whose name ends in an image
            # suffix is an image the scorer can open.
            with open(os.path.join(img_dir, "a.txt"), "w", encoding="utf-8") as fh:
                fh.write("0 0.5 0.5 0.1 0.1\n")
            os.makedirs(os.path.join(img_dir, "spare.png"), exist_ok=True)
            expected[split] = len(names)

        run_dir = os.path.join(tmp, "run")
        os.makedirs(run_dir, exist_ok=True)
        ctx = ER.gather_context(run_dir, dataset)
        check(ctx["split_counts"] == expected,
              f"gather_context must count every readable image type, got "
              f"{ctx['split_counts']}, expected {expected}")
        check(ctx["split_counts"] == {s: ER._count_images(dataset, s) for s in expected},
              "the report's split sizes and the ledger's must come from one list")

        md = ER.build_report_md(wording_metrics(CORAL_NAMES, seg=True), ctx, [], fake_args(tmp))
        check("- splits: train=3 valid=2 test=3" in md,
              f"the report's splits line carries the full counts: "
              f"{[ln for ln in md.splitlines() if ln.startswith('- splits')]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_score_keeps_the_tail_while_the_previews_stay_readable():
    """One number cannot do two jobs: the score is measured with almost every
    prediction kept, and the preview images are drawn at the display
    threshold, or a 256 by 256 frame carries up to 300 labelled boxes."""
    root = tempfile.mkdtemp(prefix="t6_preview_conf_")
    saved = {k: getattr(ER, k) for k in ("run_val", "render_previews", "build_pdf")}
    saved_argv = sys.argv
    try:
        run_dir = os.path.join(root, "runs", "run_fish")
        os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
        write_text(os.path.join(run_dir, "args.yaml"), "model: yolo11l.pt\ntask: detect\n")
        dataset = os.path.join(root, "dataset")
        write_text(os.path.join(dataset, "data.yaml"),
                   f"path: .\nnames:\n  0: {FISH_TOKEN}\n  1: {FISH_TOKEN_2}\n")
        seen = {}

        def fake_val(args):
            """Record the confidence the score was measured with."""
            seen["measured_at"] = args.conf
            return {"results": fish_results(),
                    "weights": os.path.join(run_dir, "weights", "best.pt"),
                    "data_yaml": os.path.join(dataset, "data.yaml"),
                    "wall_seconds": 1.0, "model": None, "val_dir": args.out_dir}

        def fake_previews(model, dataset_dir, split, out_dir, count, conf, iou, imgsz):
            """Record the confidence the sample images were drawn at."""
            seen["drawn_at"] = conf
            return []

        ER.run_val = fake_val
        ER.render_previews = fake_previews
        ER.build_pdf = lambda *a, **k: None

        base_argv = ["evaluate_run.py", "--run_dir", run_dir, "--dataset_dir", dataset,
                     "--split", "test"]
        sys.argv = base_argv + ["--out_dir", os.path.join(root, "out_default")]
        parsed = ER.arg_parse()
        check(parsed.conf == 0.001,
              f"the evaluator's own command line measures at 0.001, got {parsed.conf}")
        check(parsed.preview_conf == ER.PREVIEW_CONF_DEFAULT == 0.25,
              f"and draws its previews at the display threshold, got {parsed.preview_conf}")

        with contextlib.redirect_stdout(io.StringIO()):
            code = ER.main()
        check(code == ER.EXIT_OK, f"main must exit {ER.EXIT_OK}, got {code}")
        check(seen["measured_at"] == 0.001,
              f"the score keeps the whole curve, got {seen['measured_at']}")
        check(seen["drawn_at"] == ER.PREVIEW_CONF_DEFAULT,
              f"the previews are drawn at the display threshold, got {seen['drawn_at']}")

        sys.argv = base_argv + ["--out_dir", os.path.join(root, "out_typed"),
                                "--conf", "0.002", "--preview_conf", "0.4"]
        with contextlib.redirect_stdout(io.StringIO()):
            ER.main()
        check((seen["measured_at"], seen["drawn_at"]) == (0.002, 0.4),
              f"both thresholds are honoured as typed, got {(seen['measured_at'], seen['drawn_at'])}")
    finally:
        for k, v in saved.items():
            setattr(ER, k, v)
        sys.argv = saved_argv
        ER.set_names_full({})
        shutil.rmtree(root, ignore_errors=True)


def test_the_report_says_which_threshold_produced_which_number():
    """The report's glossary and its sample pages carry the measurement
    confidence and the display threshold as two different numbers, and no
    copy anywhere still calls the retired cut-off the right default."""
    tmp = tempfile.mkdtemp(prefix="eval_metrics_thresholds_")
    try:
        ER.set_names_full({})
        ER.set_subject_noun(None)
        args = fake_args(tmp)
        for metrics, label in ((wording_metrics(FISH_NAMES, seg=False), "detector"),
                               (wording_metrics(CORAL_NAMES, seg=True), "segmenter")):
            text = (ER.build_report_md(metrics, fake_ctx(), wording_previews(tmp), args)
                    + ER._render_glossary(metrics, fake_ctx(), args, 9, "2026-09-15 10:00 AST"))
            check("A default of 0.25 works for most runs" not in text,
                  f"{label}: the report stops calling the retired cut-off the right default")
            check("confidence threshold between 0.20 and 0.40" not in text,
                  f"{label}: and stops recommending a band that understates every score")
            check("0.001" in text,
                  f"{label}: the glossary names the confidence the score was measured at")
            check("whole precision-recall curve" in text,
                  f"{label}: and says why it is that low")
            check(f"{ER.PREVIEW_CONF_DEFAULT}" in text,
                  f"{label}: the sample images say what they were drawn at")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        ER.set_names_full({})
        ER.set_subject_noun(None)


if __name__ == "__main__":
    cases = [
        ("test_per_class_rows_follow_ap_class_index", test_per_class_rows_follow_ap_class_index),
        ("test_absent_class_is_reported_absent_not_zero", test_absent_class_is_reported_absent_not_zero),
        ("test_no_index_error_is_swallowed", test_no_index_error_is_swallowed),
        ("test_instance_counts_match_the_validator", test_instance_counts_match_the_validator),
        ("test_label_parser_reads_box_and_polygon_files", test_label_parser_reads_box_and_polygon_files),
        ("test_label_file_with_hostile_content", test_label_file_with_hostile_content),
        ("test_metric_family_is_chosen_from_the_numbers", test_metric_family_is_chosen_from_the_numbers),
        ("test_detector_metrics_produce_box_gates", test_detector_metrics_produce_box_gates),
        ("test_cross_family_comparison_is_marked_impossible", test_cross_family_comparison_is_marked_impossible),
        ("test_markdown_report_labels_the_family", test_markdown_report_labels_the_family),
        ("test_empty_names_mapping_is_refused", test_empty_names_mapping_is_refused),
        ("test_empty_ap_class_index_marks_every_class_absent", test_empty_ap_class_index_marks_every_class_absent),
        ("test_non_contiguous_class_ids", test_non_contiguous_class_ids),
        ("test_results_with_both_and_with_neither_family", test_results_with_both_and_with_neither_family),
        ("test_pdf_content_builders_render_both_families", test_pdf_content_builders_render_both_families),
        ("test_display_name_resolution", test_display_name_resolution),
        ("test_counts_take_a_singular_or_plural_noun", test_counts_take_a_singular_or_plural_noun),
        ("test_names_full_wins_over_registry_and_binomial", test_names_full_wins_over_registry_and_binomial),
        ("test_names_full_from_data_yaml", test_names_full_from_data_yaml),
        ("test_names_full_from_data_yaml_refuses_a_malformed_file", test_names_full_from_data_yaml_refuses_a_malformed_file),
        ("test_a_dataset_without_names_full_is_never_refused_on_its_names_map", test_a_dataset_without_names_full_is_never_refused_on_its_names_map),
        ("test_a_vertical_bar_in_a_name_stays_inside_its_markdown_cell", test_a_vertical_bar_in_a_name_stays_inside_its_markdown_cell),
        ("test_metrics_name_the_classes_only_when_they_are_codes", test_metrics_name_the_classes_only_when_they_are_codes),
        ("test_main_writes_class_full_names_from_the_dataset", test_main_writes_class_full_names_from_the_dataset),
        ("test_names_full_reads_what_the_split_builder_writes", test_names_full_reads_what_the_split_builder_writes),
        ("test_the_names_setters_refuse_the_wrong_type", test_the_names_setters_refuse_the_wrong_type),
        ("test_a_detector_report_never_calls_its_subjects_corals", test_a_detector_report_never_calls_its_subjects_corals),
        ("test_a_detector_is_never_described_as_drawing_outlines", test_a_detector_is_never_described_as_drawing_outlines),
        ("test_a_coral_report_changes_only_the_subject_noun", test_a_coral_report_changes_only_the_subject_noun),
        ("test_a_dataset_can_declare_the_noun_for_its_own_subjects", test_a_dataset_can_declare_the_noun_for_its_own_subjects),
        ("test_a_detector_glossary_matches_the_pages_it_points_at", test_a_detector_glossary_matches_the_pages_it_points_at),
        ("test_a_rebuilt_dataset_keeps_the_noun_its_source_declares", test_a_rebuilt_dataset_keeps_the_noun_its_source_declares),
        ("test_split_counts_read_every_image_type_the_trainer_reads", test_split_counts_read_every_image_type_the_trainer_reads),
        ("test_the_score_keeps_the_tail_while_the_previews_stay_readable", test_the_score_keeps_the_tail_while_the_previews_stay_readable),
        ("test_the_report_says_which_threshold_produced_which_number", test_the_report_says_which_threshold_produced_which_number),
    ]
    for name, fn in cases:
        run(name, fn)
    n_fail = sum(1 for _, ok in _results if not ok)
    print(f"{len(_results) - n_fail} passed, {n_fail} failed")
    print("PASS" if n_fail == 0 else f"{n_fail} FAILED")
    sys.exit(1 if n_fail else 0)

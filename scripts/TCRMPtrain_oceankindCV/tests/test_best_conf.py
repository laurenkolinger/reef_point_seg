"""Tests for the best confidence the Evaluate step records and the model card
it refreshes: best_conf_fields (box and segmentation curves, missing and empty
curves), the extracted metrics carrying best_conf and best_conf_f1, main
writing them to metrics.json, and the card beside best.pt rebuilt whole after
a test-split evaluation (scores, report link, checksum, best confidence) with
refreshed_by "VEGA Evaluate", person fields kept, a broken card left alone.
No model, no GPU, no network.
Run: env/bin/python scripts/TCRMPtrain_oceankindCV/tests/test_best_conf.py"""
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
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import evaluate_run as ER
import test_evaluate_metrics as TEM

_results = []

PX = [0.0, 0.25, 0.5, 0.75, 1.0]
BOX_CURVE = [[0.0, 0.5, 0.8, 0.2, 0.0], [0.0, 0.3, 0.6, 0.4, 0.0]]
SEG_CURVE = [[0.0, 0.1, 0.2, 0.9, 0.0]]
BOX_PEAK = (0.5, 0.7)
SEG_PEAK = (0.75, 0.9)


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


def with_curve(block, px, curve):
    """Give a fake metric block the px grid and f1_curve ultralytics exposes."""
    block.px = px
    block.f1_curve = curve
    return block


def detector_with_curve():
    """A detector results stub whose box metrics carry an F1 curve."""
    res = TEM.detector_results()
    with_curve(res.box, PX, BOX_CURVE)
    return res


def test_box_curve_gives_the_peak_of_the_mean():
    got = ER.best_conf_fields(detector_with_curve())
    check(got == {"best_conf": BOX_PEAK[0], "best_conf_f1": BOX_PEAK[1]}, f"box peak: {got}")


def test_segmentation_reads_the_seg_curve_not_the_box_curve():
    res = TEM.detector_results()
    res.seg = TEM.FakeMetric(TEM.PRESENT_IDS)
    with_curve(res.box, PX, BOX_CURVE)
    with_curve(res.seg, PX, SEG_CURVE)
    res.task = "segment"
    got = ER.best_conf_fields(res)
    check(got == {"best_conf": SEG_PEAK[0], "best_conf_f1": SEG_PEAK[1]}, f"seg peak: {got}")
    res.task = "detect"
    got = ER.best_conf_fields(res)
    check(got["best_conf"] == BOX_PEAK[0], f"a detect task reads the box curve: {got}")


def test_numpy_curves_are_accepted():
    import numpy as np
    res = TEM.detector_results()
    with_curve(res.box, np.array(PX), np.array(BOX_CURVE))
    got = ER.best_conf_fields(res)
    check(got["best_conf"] == BOX_PEAK[0], f"numpy arrays: {got}")


def test_missing_or_empty_curve_gives_no_fields():
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        check(ER.best_conf_fields(TEM.detector_results()) == {}, "no px or f1_curve gives {}")
        empty = TEM.detector_results()
        with_curve(empty.box, [], [])
        check(ER.best_conf_fields(empty) == {}, "an empty curve gives {}")
        ragged = TEM.detector_results()
        with_curve(ragged.box, PX, [[0.1, 0.2]])
        check(ER.best_conf_fields(ragged) == {}, "a curve of the wrong length gives {}")
    check("best confidence" in buf.getvalue(), "each refusal prints one line naming best confidence")


def test_extracted_metrics_carry_the_two_keys():
    with contextlib.redirect_stdout(io.StringIO()):
        m = ER.extract_metrics(TEM.val_out_for(detector_with_curve()))
        bare = ER.extract_metrics(TEM.val_out_for(TEM.detector_results()))
    check(m["best_conf"] == BOX_PEAK[0] and m["best_conf_f1"] == BOX_PEAK[1], f"metrics keys: {m}")
    check("best_conf" not in bare, "results without a curve leave the metrics as they were")


def test_main_writes_metrics_json_and_refreshes_the_card():
    root = tempfile.mkdtemp(prefix="t8_best_conf_main_")
    saved = {k: getattr(ER, k) for k in ("run_val", "render_previews", "build_pdf")}
    saved_argv = sys.argv
    try:
        run_dir = os.path.join(root, "runs", "run_one")
        weights = os.path.join(run_dir, "weights", "best.pt")
        os.makedirs(os.path.dirname(weights))
        with open(weights, "wb") as fh:
            fh.write(b"fake-weights")
        TEM.write_text(os.path.join(run_dir, "args.yaml"), "model: yolo11l.pt\ntask: detect\n")
        dataset = os.path.join(root, "dataset")
        TEM.write_text(os.path.join(dataset, "data.yaml"), "path: .\nnames:\n  0: A\n  1: B\n")
        ER.render_previews = lambda *a, **k: []
        ER.build_pdf = lambda *a, **k: None
        ER.run_val = lambda args: {"results": detector_with_curve(), "weights": weights,
                                   "data_yaml": os.path.join(dataset, "data.yaml"), "wall_seconds": 1.0,
                                   "model": None, "val_dir": args.out_dir}
        out_dir = os.path.join(root, "out")
        sys.argv = ["evaluate_run.py", "--run_dir", run_dir, "--dataset_dir", dataset,
                    "--out_dir", out_dir, "--split", "test"]
        with contextlib.redirect_stdout(io.StringIO()):
            code = ER.main()
        check(code == ER.EXIT_OK, f"main exits {ER.EXIT_OK}, got {code}")
        with open(os.path.join(out_dir, "metrics.json"), encoding="utf-8") as fh:
            metrics = json.load(fh)["metrics"]
        check(metrics["best_conf"] == BOX_PEAK[0] and metrics["best_conf_f1"] == BOX_PEAK[1],
              f"metrics.json keys: {metrics.get('best_conf')}, {metrics.get('best_conf_f1')}")
        _, card = ER.load_model_card_modules()
        doc = card.read_card(card.card_path_for(__import__("pathlib").Path(weights)))
        record = doc["vicarius"]["record"]
        check(record["best_conf"] == BOX_PEAK[0] and record["best_conf_f1"] == BOX_PEAK[1], f"card record: {record}")
        check(record["best_conf_source"].startswith("VEGA evaluation "), f"source: {record['best_conf_source']}")
        check(doc["vicarius"]["refreshed_by"] == "VEGA Evaluate", f"writer: {doc['vicarius']['refreshed_by']}")
    finally:
        for k, v in saved.items():
            setattr(ER, k, v)
        sys.argv = saved_argv
        ER.set_names_full({})
        shutil.rmtree(root, ignore_errors=True)


def test_card_is_not_refreshed_for_another_split_or_last_pt():
    root = tempfile.mkdtemp(prefix="t8_best_conf_split_")
    try:
        weights = os.path.join(root, "best.pt")
        last = os.path.join(root, "last.pt")
        for path in (weights, last):
            with open(path, "wb") as fh:
                fh.write(b"x")
        metrics = {"best_conf": 0.4, "best_conf_f1": 0.8}
        with contextlib.redirect_stdout(io.StringIO()):
            check(ER.refresh_model_card(weights, metrics, "val") is None, "a val split writes no card")
            check(ER.refresh_model_card(last, metrics, "test") is None, "last.pt writes no card")
        check(not os.path.exists(weights + ".card.md") and not os.path.exists(last + ".card.md"), "no card file")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_one_dimensional_curve_or_missing_attribute_leaves_the_metrics_written():
    flat = TEM.detector_results()
    with_curve(flat.box, PX, [0.1, 0.5, 0.9, 0.5, 0.1])
    bare = TEM.detector_results()
    bare.box = types.SimpleNamespace(px=PX)
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        check(ER.best_conf_fields(flat) == {}, "a 1-D curve gives no fields")
        check(ER.best_conf_fields(bare) == {}, "a block with no f1_curve gives no fields")
        m = ER.extract_metrics(TEM.val_out_for(flat))
    check("best_conf" not in m and "overall" in m, "the evaluation's metrics are still extracted")
    check(buf.getvalue().count("best confidence") >= 2, "each refusal prints a line")


def test_a_refresh_that_raises_never_fails_the_evaluation():
    root = tempfile.mkdtemp(prefix="t8_best_conf_raise_")
    saved = {k: getattr(ER, k) for k in ("run_val", "render_previews", "build_pdf", "refresh_model_card")}
    saved_argv = sys.argv
    try:
        run_dir = os.path.join(root, "runs", "run_one")
        weights = os.path.join(run_dir, "weights", "best.pt")
        os.makedirs(os.path.dirname(weights))
        with open(weights, "wb") as fh:
            fh.write(b"fake-weights")
        TEM.write_text(os.path.join(run_dir, "args.yaml"), "model: yolo11l.pt\ntask: detect\n")
        dataset = os.path.join(root, "dataset")
        TEM.write_text(os.path.join(dataset, "data.yaml"), "path: .\nnames:\n  0: A\n  1: B\n")

        def boom(*_a, **_k):
            raise RuntimeError("card exploded")

        ER.render_previews = lambda *a, **k: []
        ER.build_pdf = lambda *a, **k: None
        ER.refresh_model_card = boom
        ER.run_val = lambda args: {"results": detector_with_curve(), "weights": weights,
                                   "data_yaml": os.path.join(dataset, "data.yaml"), "wall_seconds": 1.0,
                                   "model": None, "val_dir": args.out_dir}
        out_dir = os.path.join(root, "out")
        sys.argv = ["evaluate_run.py", "--run_dir", run_dir, "--dataset_dir", dataset,
                    "--out_dir", out_dir, "--split", "test"]
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            code = ER.main()
        check(code == ER.EXIT_OK, f"main still exits {ER.EXIT_OK}, got {code}")
        check(os.path.isfile(os.path.join(out_dir, "metrics.json")), "metrics.json is written")
        check("card exploded" in buf.getvalue(), "one line says why the card was not refreshed")
    finally:
        for k, v in saved.items():
            setattr(ER, k, v)
        sys.argv = saved_argv
        ER.set_names_full({})
        shutil.rmtree(root, ignore_errors=True)


def test_a_missing_card_code_folder_is_reported_not_raised():
    saved = os.environ.get("VICARIUS_ROOT")
    os.environ["VICARIUS_ROOT"] = "/nonexistent/vicarius_root"
    try:
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            check(ER.best_conf_fields(detector_with_curve()) == {}, "fields are empty without the card code")
            check(ER.refresh_model_card("/x/best.pt", {"best_conf": 0.3, "best_conf_f1": 0.5}, "test") is None,
                  "no card is written without the card code")
        check("not at /nonexistent/vicarius_root" in buf.getvalue(), buf.getvalue())
    finally:
        if saved is None:
            os.environ.pop("VICARIUS_ROOT", None)
        else:
            os.environ["VICARIUS_ROOT"] = saved


# ---------------------------------------------------------------------------
# The Evaluate step rebuilds the card's whole record (final review I2)
# ---------------------------------------------------------------------------

VEGA_PROJECT = "run_26-10-17_LO_demo"
VEGA_RUN = "demo_20261017_090000"
PDF_STUB = b"%PDF-1.4 stub"


def make_vega_tree(root):
    """A VEGA project with one trained run, its dataset and the step 7 folder the route evaluates into."""
    project = os.path.join(root, "inprocess", VEGA_PROJECT)
    step6 = os.path.join(project, "step6_trainModel")
    run_dir = os.path.join(step6, "runs", VEGA_RUN)
    weights = os.path.join(run_dir, "weights", "best.pt")
    os.makedirs(os.path.dirname(weights))
    with open(weights, "wb") as fh:
        fh.write(b"fake-weights")
    dataset = os.path.join(step6, "dataset")
    TEM.write_text(os.path.join(dataset, "data.yaml"), "path: .\nnames:\n  0: A\n  1: B\n")
    TEM.write_text(os.path.join(run_dir, "args.yaml"),
                   f"model: yolo11l.pt\ntask: detect\ndata: {dataset}/data.yaml\nepochs: 3\nimgsz: 640\n")
    out_dir = os.path.join(project, "step7_evaluateModel", f"eval_{VEGA_RUN}")
    return {"run_dir": run_dir, "weights": weights, "dataset": dataset, "out_dir": out_dir}


def write_pdf_stub(_metrics, _ctx, _previews, _args, pdf_path):
    """Stand in for build_pdf: write a few PDF bytes where the report goes."""
    with open(pdf_path, "wb") as fh:
        fh.write(PDF_STUB)


def run_main_on(tree, results, split="test"):
    """Run ER.main on a tree with validation, previews and the PDF stubbed; return (exit code, printed text)."""
    saved = {k: getattr(ER, k) for k in ("run_val", "render_previews", "build_pdf")}
    saved_argv = sys.argv
    try:
        ER.render_previews = lambda *a, **k: []
        ER.build_pdf = write_pdf_stub
        ER.run_val = lambda args: {"results": results, "weights": tree["weights"],
                                   "data_yaml": os.path.join(tree["dataset"], "data.yaml"), "wall_seconds": 1.0,
                                   "model": None, "val_dir": args.out_dir}
        sys.argv = ["evaluate_run.py", "--run_dir", tree["run_dir"], "--dataset_dir", tree["dataset"],
                    "--out_dir", tree["out_dir"], "--split", split]
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            code = ER.main()
        return code, buf.getvalue()
    finally:
        for k, v in saved.items():
            setattr(ER, k, v)
        sys.argv = saved_argv
        ER.set_names_full({})


def sha256_hex(path):
    """Lowercase hex SHA-256 of a small file."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def read_card_of(weights):
    """The card dict beside a weights file."""
    _, card = ER.load_model_card_modules()
    return card.read_card(card.card_path_for(Path(weights)))


def write_train_card(weights, sha, extra=None):
    """The card VEGA's Train step leaves: a record with no scores, plus a person summary."""
    _, card = ER.load_model_card_modules()
    record = {"weights": {"sha256": sha}, "task": "detect", "default_name": "demo", **(extra or {})}
    card.write_record(Path(weights), record, "VEGA Train")
    card.edit_person(Path(weights), "summary", "Finds two test classes.", "LO")


def test_evaluate_rebuilds_the_whole_card_record():
    root = tempfile.mkdtemp(prefix="i2_card_refresh_")
    try:
        tree = make_vega_tree(root)
        sha = sha256_hex(tree["weights"])
        write_train_card(tree["weights"], sha)
        check("performance" not in read_card_of(tree["weights"])["vicarius"]["record"], "the Train card has no scores")
        code, out = run_main_on(tree, detector_with_curve())
        check(code == ER.EXIT_OK, f"main exits {ER.EXIT_OK}, got {code}: {out}")
        with open(os.path.join(tree["out_dir"], "metrics.json"), encoding="utf-8") as fh:
            overall = json.load(fh)["metrics"]["overall"]
        doc = read_card_of(tree["weights"])
        block, record = doc["vicarius"], doc["vicarius"]["record"]
        perf = record.get("performance") or {}
        check(perf.get("map50") == round(overall["box_map50"], 3), f"map50 on the card: {perf}")
        check(perf.get("map50_95") == round(overall["box_map50_95"], 3), f"map50_95 on the card: {perf}")
        check(perf.get("metric_family") == "box" and perf.get("split") == "test", f"family and split: {perf}")
        check(perf.get("evaluation_dir") == tree["out_dir"], f"evaluation folder: {perf.get('evaluation_dir')}")
        check(record.get("evaluation_report") == os.path.join(tree["out_dir"], "report.pdf"),
              f"evaluation link: {record.get('evaluation_report')}")
        check(record["best_conf"] == BOX_PEAK[0] and record["best_conf_f1"] == BOX_PEAK[1], f"best conf: {record}")
        check(record["best_conf_source"].startswith("VEGA evaluation "), f"source: {record['best_conf_source']}")
        check(record["weights"]["sha256"] == sha, f"checksum: {record['weights']}")
        check(record["weights"]["file"] == os.path.realpath(tree["weights"]), f"file: {record['weights']}")
        check(record.get("trained_at") == "2026-10-17T09:00:00-04:00", f"trained_at: {record.get('trained_at')}")
        check(record.get("default_name") == "26-10-17_LO_demo / " + VEGA_RUN, f"name: {record.get('default_name')}")
        check(block["refreshed_by"] == "VEGA Evaluate", f"writer: {block['refreshed_by']}")
        check(block["person"].get("summary") == "Finds two test classes." and len(block["history"]) == 1,
              f"person part kept: {block['person']}, {block['history']}")
        check("model card refreshed" in out, "one line names the refreshed card")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_new_weights_drop_the_old_measurement_and_copies_and_scores_still_arrive():
    root = tempfile.mkdtemp(prefix="i2_card_new_weights_")
    try:
        tree = make_vega_tree(root)
        old = {"best_conf": 0.9, "best_conf_f1": 0.95, "best_conf_source": "VISION measurement 2026-10-16 10:00:00 AST",
               "also_at": ["/elsewhere/best.pt"]}
        write_train_card(tree["weights"], "0" * 64, old)
        code, out = run_main_on(tree, TEM.detector_results())
        check(code == ER.EXIT_OK, f"main exits {ER.EXIT_OK}, got {code}")
        record = read_card_of(tree["weights"])["vicarius"]["record"]
        check("map50" in (record.get("performance") or {}), f"scores arrive without a best confidence: {record}")
        check(not {"best_conf", "best_conf_f1", "best_conf_source", "also_at"} & set(record),
              f"another file's measurement and copies are dropped: {sorted(record)}")
        check(record["weights"]["sha256"] == sha256_hex(tree["weights"]), "the checksum is this file's")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_same_weights_keep_the_copies_a_backfill_found():
    root = tempfile.mkdtemp(prefix="i2_card_same_weights_")
    try:
        tree = make_vega_tree(root)
        write_train_card(tree["weights"], sha256_hex(tree["weights"]), {"also_at": ["/lab/runs/x/weights/best.pt"]})
        code, _ = run_main_on(tree, detector_with_curve())
        record = read_card_of(tree["weights"])["vicarius"]["record"]
        check(code == ER.EXIT_OK and record.get("also_at") == ["/lab/runs/x/weights/best.pt"], f"also_at: {record}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_broken_card_is_left_alone_and_the_evaluation_passes():
    root = tempfile.mkdtemp(prefix="i2_card_broken_")
    try:
        tree = make_vega_tree(root)
        card_path = tree["weights"] + ".card.md"
        with open(card_path, "w", encoding="utf-8") as fh:
            fh.write("no header here\n")
        code, out = run_main_on(tree, detector_with_curve())
        check(code == ER.EXIT_OK, f"main still exits {ER.EXIT_OK}, got {code}")
        with open(card_path, encoding="utf-8") as fh:
            check(fh.read() == "no header here\n", "the broken card is never overwritten")
        lines = [ln for ln in out.splitlines() if "card not refreshed" in ln]
        check(len(lines) == 1 and "broken header" in lines[0], f"one line says why: {lines}")
        check(os.path.isfile(os.path.join(tree["out_dir"], "metrics.json")), "metrics.json is written")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_val_split_evaluation_leaves_the_card_as_it_was():
    root = tempfile.mkdtemp(prefix="i2_card_val_")
    try:
        tree = make_vega_tree(root)
        write_train_card(tree["weights"], sha256_hex(tree["weights"]))
        before = read_card_of(tree["weights"])
        code, out = run_main_on(tree, detector_with_curve(), split="val")
        check(code == ER.EXIT_OK and read_card_of(tree["weights"]) == before, "a val evaluation changes no card")
        check("test split evaluation only" in out, f"one line says why: {out[-300:]}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_card_layout_follows_the_run_folder():
    layouts = ER.load_card_builder().LAYOUTS
    cases = {
        "/x/inprocess/run_a/step6_trainModel/runs/r1/weights/best.pt": "vega_run",
        "/x/archive/run_a/step6_trainModel/runs/r1/weights/best.pt": "vega_archived_run",
        "/x/inprocess/_models/coral_lab/runs/r1/weights/best.pt": "vega_lab_run",
        "/x/somewhere/r1/weights/best.pt": "vega_run",
        "best.pt": "vega_run",
    }
    for path, expected in cases.items():
        got = ER.card_layout_for(path)
        check(got == expected and got in layouts, f"{path}: {got}")


if __name__ == "__main__":
    cases = [
        ("test_box_curve_gives_the_peak_of_the_mean", test_box_curve_gives_the_peak_of_the_mean),
        ("test_segmentation_reads_the_seg_curve_not_the_box_curve", test_segmentation_reads_the_seg_curve_not_the_box_curve),
        ("test_numpy_curves_are_accepted", test_numpy_curves_are_accepted),
        ("test_missing_or_empty_curve_gives_no_fields", test_missing_or_empty_curve_gives_no_fields),
        ("test_extracted_metrics_carry_the_two_keys", test_extracted_metrics_carry_the_two_keys),
        ("test_main_writes_metrics_json_and_refreshes_the_card", test_main_writes_metrics_json_and_refreshes_the_card),
        ("test_card_is_not_refreshed_for_another_split_or_last_pt", test_card_is_not_refreshed_for_another_split_or_last_pt),
        ("test_a_one_dimensional_curve_or_missing_attribute_leaves_the_metrics_written", test_a_one_dimensional_curve_or_missing_attribute_leaves_the_metrics_written),
        ("test_a_refresh_that_raises_never_fails_the_evaluation", test_a_refresh_that_raises_never_fails_the_evaluation),
        ("test_a_missing_card_code_folder_is_reported_not_raised", test_a_missing_card_code_folder_is_reported_not_raised),
        ("test_evaluate_rebuilds_the_whole_card_record", test_evaluate_rebuilds_the_whole_card_record),
        ("test_new_weights_drop_the_old_measurement_and_copies_and_scores_still_arrive", test_new_weights_drop_the_old_measurement_and_copies_and_scores_still_arrive),
        ("test_same_weights_keep_the_copies_a_backfill_found", test_same_weights_keep_the_copies_a_backfill_found),
        ("test_a_broken_card_is_left_alone_and_the_evaluation_passes", test_a_broken_card_is_left_alone_and_the_evaluation_passes),
        ("test_a_val_split_evaluation_leaves_the_card_as_it_was", test_a_val_split_evaluation_leaves_the_card_as_it_was),
        ("test_card_layout_follows_the_run_folder", test_card_layout_follows_the_run_folder),
    ]
    for name, fn in cases:
        run(name, fn)
    n_fail = sum(1 for _, ok in _results if not ok)
    print(f"{len(_results) - n_fail} passed, {n_fail} failed")
    print("PASS" if n_fail == 0 else f"{n_fail} FAILED")
    sys.exit(1 if n_fail else 0)

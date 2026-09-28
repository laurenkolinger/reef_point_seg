#!/usr/bin/env python3
"""House tests for the Coral Intel study's shared folder report.

Module:  scripts/_ablation/tests/test_coral_intel_study_report.py
Purpose: Prove the study registry keeps one project per study, that each
         model's status is read correctly from its project on disk (queued,
         training with epoch and best validation score, trained, scored,
         failed), that the models-and-status page names every model with why
         it runs, and that a refresh fills the share folder with the status
         page, each scored model's evaluation PDF, the summary and the
         methods, without listing a queued model as a failure.
Inputs:  none; projects are built in a temporary directory, PDFs are faked.
Outputs: a pass or fail count on stdout, exit code 1 on any failure.

Run:     github_repo/env/bin/python scripts/_ablation/tests/test_coral_intel_study_report.py
"""

import html
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coral_intel_study_report as rep  # noqa: E402
import coral_intel_vega_driver as cd  # noqa: E402

FAILURES = 0
CHECKS = 0


def check(condition: bool, message: str) -> None:
    """Record one assertion, printing PASS or FAIL and counting failures."""
    global FAILURES, CHECKS
    CHECKS += 1
    if condition:
        print(f"PASS {message}")
    else:
        FAILURES += 1
        print(f"FAIL {message}")


def section(title: str) -> None:
    """Print a section heading."""
    print(f"\n== {title}")


def make_project(root: Path, name: str) -> Path:
    """A minimal VEGA project folder."""
    project = root / name
    (project / cd.STUDY_DIRNAME).mkdir(parents=True)
    (project / "project.json").write_text("{}")
    return project


def add_run(project: Path, arm_id: str, run: str, epochs: int, trained: bool,
            scored: bool) -> None:
    """Record an arm's run with ``epochs`` rows of results and optional scores."""
    state = cd.read_json(project / cd.STUDY_DIRNAME / "state.json", {}) or {}
    state[arm_id] = {"run_name": run, "trained": trained}
    cd.write_json(project / cd.STUDY_DIRNAME / "state.json", state)
    run_dir = project / "step6_trainModel" / "runs" / run
    run_dir.mkdir(parents=True)
    lines = ["epoch,metrics/mAP50(B),metrics/mAP50-95(B)"]
    lines += [f"{e},{0.5 + e / 1000:.4f},{0.4 + (e if e < 30 else 60 - e) / 1000:.4f}"
              for e in range(1, epochs + 1)]
    (run_dir / "results.csv").write_text("\n".join(lines) + "\n")
    if scored:
        out = project / "step7_evaluateModel" / f"eval_{run}"
        out.mkdir(parents=True)
        (out / "report.pdf").write_bytes(b"%PDF")
        (out / "metrics.json").write_text(json.dumps({
            "metrics": {"overall": {"box_map50": 0.66, "box_map50_95": 0.58,
                                    "box_precision": 0.7, "box_recall": 0.6},
                        "per_class": {}, "per_class_instances": {}},
            "context": {"epochs_completed": epochs, "best_epoch": 30,
                        "split_counts": {"train": 21771, "valid": 1552, "test": 1552}}}))


tmp = Path(tempfile.mkdtemp(prefix="coral_report_test_"))
try:
    main = make_project(tmp, "main")
    leaky = make_project(tmp, "leaky")

    section("the registry")
    check(rep.registered(main) == {"main": main}, "the main project is always registered")
    rep.register(main, "leaky", leaky)
    check(rep.registered(main)["leaky"] == leaky, "the leaky project is recorded")
    try:
        rep.register(main, "bogus", leaky)
        check(False, "an unknown study is refused")
    except ValueError:
        check(True, "an unknown study is refused")

    section("status read from disk")
    arms = {a["id"]: a for a in cd.resolve_arms(rep.STUDIES["main"]["presets"])}
    add_run(main, "A0_replicate", "RUN_A0", 178, trained=True, scored=True)
    add_run(main, "B1_yolo11x", "RUN_B1", 133, trained=True, scored=False)
    add_run(main, "B2_imgsz1024", "RUN_B2", 37, trained=False, scored=False)
    a0 = rep.arm_status(main, arms["A0_replicate"])
    check(a0["state"] == "scored" and a0["scores"]["map50"] == 0.66
          and "178 epochs" in a0["status"], f"a scored model: {a0['status']}")
    b1 = rep.arm_status(main, arms["B1_yolo11x"])
    check(b1["state"] == "trained" and "waiting to be scored" in b1["status"],
          f"a trained, unscored model: {b1['status']}")
    b2 = rep.arm_status(main, arms["B2_imgsz1024"])
    check(b2["state"] == "training" and "epoch 37 of up to 300" in b2["status"]
          and b2["progress"]["best_epoch"] == 30, f"a training model: {b2['status']}")
    b3 = rep.arm_status(main, arms["B3_offline_only"])
    check(b3["state"] == "queued", "a model with no run is queued")
    check(rep.arm_status(None, arms["A0_replicate"])["state"] == "queued",
          "a study with no project yet is queued")
    (main / "step6_trainModel" / "runs" / "RUN_B2" / "results.csv").write_text("garbage\n")
    check(rep.arm_status(main, arms["B2_imgsz1024"])["status"] == "Training: starting.",
          "an unreadable results file does not crash the status")
    cd.Progress(main / cd.STUDY_DIRNAME / "progress.jsonl").write("train_failed",
                                                                  arm="B1_yolo11x")
    check(rep.arm_status(main, arms["B1_yolo11x"])["state"] == "failed",
          "a failed training is reported as failed")

    section("the models-and-status page")
    models = rep.collect(main)
    ids = [m["arm"]["id"] for m in models]
    check(ids == ["A0_replicate", "B1_yolo11x", "B2_imgsz1024", "B3_offline_only",
                  "L0_leaky_replicate"], f"every model of both studies, in order: {ids}")
    page = rep.status_html(models, cd.ast_now())
    for m in models:
        check(m["arm"]["why"] and html.escape(m["arm"]["why"][:40]) in page,
              f"{m['arm']['id']} says why it runs")
    check("Queued." in page and "A0_replicate_evaluation.pdf" in page.replace("<wbr>", ""),
          "statuses and report names appear")
    check("—" not in page and "–" not in page and "Sam" not in page,
          "no dashes and no personal names beyond the author")

    section("a refresh fills the share folder")
    printed = []
    share = rep.refresh(main, printer=lambda h, p: (printed.append(p.name), p.write_bytes(b"%PDF")))
    names = sorted(p.name for p in share.iterdir())
    check(names == sorted(["A0_replicate_evaluation.pdf", rep.STATUS_PDF_NAME,
                           cd.SUMMARY_PDF_NAME, cd.METHODS_PDF_NAME]),
          f"status, summary, methods and the one scored evaluation: {names}")
    summary = (main / cd.STUDY_DIRNAME / "summary.html").read_text()
    fail_section = summary.split("did not produce a score")[1].split("</ul>")[0] \
        if "did not produce a score" in summary else ""
    check("B1_yolo11x" in fail_section and "B3_offline_only" not in fail_section
          and "B2_imgsz1024" not in fail_section,
          "the failed model is listed as a failure; queued and training models are not")
    check("L0 repeats A0" in summary, "the summary explains the leaky model")
    methods = (main / cd.STUDY_DIRNAME / "methods.html").read_text()
    check("Test of the published split order" in methods and "L0_leaky" not in
          methods.split("Change from A0")[1].split("</table>")[0],
          "the methods describe L0 in its own paragraph, not as a challenger")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
    check(not tmp.exists(), "the temporary fixtures were removed")

print(f"\n{CHECKS - FAILURES}/{CHECKS} passed.")
sys.exit(1 if FAILURES else 0)

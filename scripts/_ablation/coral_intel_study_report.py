#!/usr/bin/env python3
"""The shared output folder of the Coral Intel study: every model, why, and where it stands.

Module:  scripts/_ablation/coral_intel_study_report.py
Purpose: Keep one folder, coralintel_reproduce_compete/ inside the main study
         project, complete and current. The study runs as more than one VEGA
         project (the four-arm reproduce-and-compete project, and the leaky
         replicate, which trains on a different dataset and so needs its own
         project). Each registers its project folder here; every refresh
         rebuilds the folder from what is on disk, so any driver, or the
         30-minute check, can bring it up to date:

           coralintel_reproduce_compete_models_and_status.pdf
                                     every model, what it is, why it runs,
                                     its status and its scores so far
           <arm>_evaluation.pdf      VEGA's evaluation of each scored model
           coralintel_reproduce_compete_summary.pdf
           coralintel_reproduce_compete_methods.pdf

Inputs:  the registry coral_intel_study/studies.json in the main project, and
         each registered project's coral_intel_study/state.json, run folders
         and evaluation folders.
Outputs: the files above, in <main project>/coralintel_reproduce_compete/.

Run:     github_repo/env/bin/python scripts/_ablation/coral_intel_study_report.py \\
             --share_project <main project>      (refresh only; used by the check)
"""

from __future__ import annotations

import argparse
import csv
import html
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coral_intel_vega_driver as cd  # noqa: E402

REGISTRY_NAME = "studies.json"
STATUS_PDF_NAME = "coralintel_reproduce_compete_models_and_status.pdf"
# The studies, in the order their models are listed. "main" holds the share folder.
STUDIES: dict[str, dict[str, Any]] = {
    "main": {"dataset": "coral_intel_t2aug", "presets": cd.PRESET_FILES,
             "split": "photos split before augmenting (the strict order)"},
    "leaky": {"dataset": "coral_intel_t2aug_leaky",
              "presets": [cd.STUDY_PRESET_DIR / "coral_intel_L0_leaky_replicate.yaml"],
              "split": "images split after augmenting (copies of a photo can be in training and test)"},
}


def registry_path(share_project: Path) -> Path:
    """Where the list of study projects is kept, inside the main project."""
    return Path(share_project) / cd.STUDY_DIRNAME / REGISTRY_NAME


def register(share_project: Path, study: str, project_dir: Path) -> None:
    """Record which VEGA project holds a study's runs.

    Raises ValueError for a study name not in STUDIES.
    """
    if study not in STUDIES:
        raise ValueError(f"unknown study {study!r}; expected one of {sorted(STUDIES)}")
    doc = cd.read_json(registry_path(share_project), {}) or {}
    doc[study] = str(project_dir)
    cd.write_json(registry_path(share_project), doc)


def registered(share_project: Path) -> dict[str, Path]:
    """The study to project map, with the main project always present."""
    doc = cd.read_json(registry_path(share_project), {}) or {}
    out = {k: Path(v) for k, v in doc.items() if k in STUDIES}
    out.setdefault("main", Path(share_project))
    return out


def training_progress(run_dir: Path) -> dict[str, Any]:
    """Epochs finished and the best validation scores so far, from results.csv.

    Returns {} when the run has written no epoch yet or the file is unreadable.
    """
    path = Path(run_dir) / "results.csv"
    try:
        with path.open() as handle:
            rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(handle)]
    except OSError:
        return {}
    try:
        best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
        return {"epochs_done": int(float(rows[-1]["epoch"])),
                "best_epoch": int(float(best["epoch"])),
                "val_map50": float(best["metrics/mAP50(B)"]),
                "val_map50_95": float(best["metrics/mAP50-95(B)"])}
    except (ValueError, KeyError):
        return {}


def arm_status(project_dir: Path | None, arm: Mapping[str, Any]) -> dict[str, Any]:
    """Where one model stands, read from its project on disk.

    Returns a dict with state (queued, training, trained, scored, failed),
    a plain status sentence, the run name, validation progress and, once
    scored, the test scores.
    """
    row: dict[str, Any] = {"id": arm["id"], "state": "queued", "status": "Queued."}
    if project_dir is None or not (Path(project_dir) / "project.json").is_file():
        return row
    project_dir = Path(project_dir)
    record = (cd.read_json(project_dir / cd.STUDY_DIRNAME / "state.json", {}) or {}).get(arm["id"]) or {}
    run_name = record.get("run_name")
    if not run_name:
        return row
    row["run_name"] = run_name
    progress = training_progress(project_dir / "step6_trainModel" / "runs" / run_name)
    row["progress"] = progress
    metrics = project_dir / "step7_evaluateModel" / f"eval_{run_name}" / "metrics.json"
    total = arm["config"].get("epochs")
    if metrics.is_file():
        row["state"], row["scores"] = "scored", cd.read_scores(metrics)
        row["status"] = (f"Scored. Trained {progress.get('epochs_done', '?')} epochs "
                         f"(best at epoch {progress.get('best_epoch', '?')}).")
    elif record.get("trained"):
        row["state"] = "trained"
        row["status"] = (f"Trained ({progress.get('epochs_done', '?')} epochs); "
                         "waiting to be scored.")
    elif progress:
        row["state"] = "training"
        row["status"] = (f"Training: epoch {progress['epochs_done']} of up to {total}; stops "
                         f"early after 100 epochs without improvement.")
    else:
        row["state"], row["status"] = "training", "Training: starting."
    failures = [e for e in _events(project_dir) if e.get("arm") == arm["id"]
                and e.get("event") in ("train_failed", "eval_failed")]
    if failures and row["state"] not in ("scored", "training"):
        row["state"], row["status"] = "failed", f"Failed: {failures[-1]['event'].replace('_', ' ')}."
    return row


def _events(project_dir: Path) -> list[dict[str, Any]]:
    """The study's progress log as a list of events; [] when absent."""
    import json  # noqa: PLC0415
    path = project_dir / cd.STUDY_DIRNAME / "progress.jsonl"
    try:
        return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    except (OSError, ValueError):
        return []


def collect(share_project: Path) -> list[dict[str, Any]]:
    """Every model of every study, in listing order, with its study and status."""
    projects = registered(share_project)
    models = []
    for study, spec in STUDIES.items():
        for arm in cd.resolve_arms(spec["presets"]):
            status = arm_status(projects.get(study), arm)
            models.append({"study": study, "split": spec["split"], "dataset": spec["dataset"],
                           "arm": arm, "row": status, "project": projects.get(study)})
    return models


def status_html(models: list[Mapping[str, Any]], created: datetime) -> str:
    """Render the models-and-status page.

    Parameters:
        models: the output of collect().
        created: when the page was written.
    Returns:
        the HTML document as a string.
    """
    esc = html.escape

    def f3(value: Any) -> str:
        return f"{value:.3f}" if isinstance(value, (int, float)) else ""

    rows = []
    for m in models:
        arm, row = m["arm"], m["row"]
        prog, scores = row.get("progress") or {}, row.get("scores") or {}
        val = (f"{f3(prog.get('val_map50'))} / {f3(prog.get('val_map50_95'))} "
               f"(epoch {prog.get('best_epoch')})" if prog else "")
        test = f"{f3(scores.get('map50'))} / {f3(scores.get('map50_95'))}" if scores else ""
        pdf = f"{arm['id']}_evaluation.pdf" if row["state"] == "scored" else ""
        rows.append(
            f"<tr><td><b>{esc(arm['id'])}</b><br>{esc(arm['label'])}</td>"
            f"<td>{esc(arm['why'])}<br><span class='meta'>Data: {esc(m['split'])}.</span></td>"
            f"<td>{esc(row['status'])}</td><td>{esc(val)}</td><td>{esc(test)}</td>"
            f"<td>{esc(pdf).replace('_', '_<wbr>')}</td></tr>")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{esc(cd.STUDY_TITLE)}</title><style>{cd.PAGE_STYLE}
@page {{ size: letter landscape; margin: 12mm; }}
table {{ table-layout: fixed; }}
col.m {{ width: 17%; }} col.w {{ width: 35%; }} col.s {{ width: 18%; }}
col.v {{ width: 10%; }} col.t {{ width: 8%; }} col.r {{ width: 12%; }}
td {{ word-wrap: break-word; }}</style></head><body>
<h1>{esc(cd.STUDY_TITLE)}: models and status</h1>
<div class="meta">VICARIUS, VEGA. Updated {esc(cd.stamp(created))}. Lauren Olinger</div>
<p>Every model in this study, what it tests and where it stands. VEGA trains one model at a
time on two RTX 5090 cards, in the order listed. Validation scores guide training; test scores
are the final measure and come from the evaluation PDF of each scored model.</p>
<table><colgroup><col class="m"><col class="w"><col class="s"><col class="v"><col class="t"><col class="r"></colgroup>
<tr><th>Model</th><th>Why it runs</th><th>Status</th>
<th>Validation, best (mAP50 / 50-95)</th><th>Test (mAP50 / 50-95)</th><th>Report</th></tr>
{''.join(rows)}</table>
<p>A0 to B3 share one split of 10,361 photographs made before augmentation (7,257 training,
1,552 validation, 1,552 test), so their test scores compare directly. L0 uses the same
settings as A0 on images split after augmenting, where 91 percent of test images have a copy
of the same photograph in training; its test score is not comparable with A0 to B3 and exists
to show how much that overlap raises a score. The published Coral Intel test mAP50 is
0.938.</p>
<p class="meta">Lauren Olinger</p>
</body></html>"""


def refresh(share_project: Path, printer: Callable[[Path, Path], None] | None = None) -> Path:
    """Rebuild the share folder from disk: status page, evaluations, summary, methods.

    Returns the share folder path. A printing failure is raised to the caller,
    which decides whether it may stop the study (the drivers log and carry on).
    """
    printer = printer or cd.print_pdf
    share_project = Path(share_project)
    share = share_project / cd.SHARE_DIRNAME
    share.mkdir(parents=True, exist_ok=True)
    work = share_project / cd.STUDY_DIRNAME
    work.mkdir(parents=True, exist_ok=True)
    models = collect(share_project)
    for m in models:
        row = m["row"]
        if row["state"] == "scored":
            src = (m["project"] / "step7_evaluateModel" / f"eval_{row['run_name']}" / "report.pdf")
            if src.is_file():
                shutil.copy2(src, share / cd.share_pdf_name(m["arm"]))
    (work / "status.html").write_text(status_html(models, cd.ast_now()))
    printer(work / "status.html", share / STATUS_PDF_NAME)
    protocol = cd.load_eval_protocol()
    arms = [m["arm"] for m in models]
    results = [dict(m["row"], status="ok") if m["row"]["state"] == "scored"
               else {"id": m["arm"]["id"], "status": m["row"]["status"],
                     "pending": m["row"]["state"] != "failed"} for m in models]
    split_counts = next((r["scores"].get("split_counts") for r in results
                         if r.get("status") == "ok" and r.get("scores")), None)
    (work / "summary.html").write_text(cd.summary_html(
        arms, results, split_counts, cd.ast_now(), protocol, cd.read_photo_counts()))
    printer(work / "summary.html", share / cd.SUMMARY_PDF_NAME)
    cd.write_methods(share_project, arms, protocol, printer)
    return share


def main(argv: list[str] | None = None) -> int:
    """CLI entry: refresh the share folder of a main study project."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--share_project", required=True,
                        help="the main study project, which holds the share folder")
    args = parser.parse_args(argv)
    share_project = Path(args.share_project)
    if not (share_project / "project.json").is_file():
        print(f"ERROR: {share_project} is not a VEGA project (no project.json)", file=sys.stderr)
        return 2
    share = refresh(share_project)
    print(f"Step 1 complete: {share} refreshed, "
          f"{len(list(share.iterdir()))} files", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

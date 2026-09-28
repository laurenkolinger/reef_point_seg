#!/usr/bin/env python3
"""Backend ablation runner for the fish bounding-box detector.

Module:  scripts/_ablation/fish_ablation.py
Purpose: Run a controlled, sequential ablation of the fish detector outside the
         VEGA orchestrator, writing every artifact into the folder shape a VEGA
         training project uses, so the finished work can be adopted by the
         platform later without being rebuilt.

         The ablation answers one question: does anything beat
         run3_heavy_augment, which scored test mAP50 0.904 and mAP50-95 0.835
         on its own held-out split. Each arm changes exactly one thing from the
         arm before it, so a difference between two neighbouring arms has one
         candidate cause:

           A1 replicate      run3's arguments, unchanged, under this machine's
                             ultralytics. Isolates the library version, because
                             run3 trained under 8.3.222 and this runs under
                             8.4.38.
           A2 finish         A1 with patience raised above the epoch count.
                             run3 stopped at epoch 225 of a scheduled 350 while
                             close_mosaic 10 turns four-image stitching off only
                             for epochs 341 to 350, so run3 never trained a
                             single epoch on whole photographs. Isolates that.
           A3 finish_lowhue  A2 with hue jitter cut from 0.30 to 0.05. Twelve of
                             the 34 classes are colour phases of one species, so
                             a large hue shift may erase the signal that
                             separates them. Isolates hue.

Inputs:  a staged YOLO box dataset (images/, labels/, classes.txt, names.csv,
         split_pins.json). The pins are replayed, so every arm holds out the
         same images run3 held out and the scores are comparable.
Outputs: <project>/step6_trainModel/dataset/   one split dataset, shared by all arms
         <project>/step6_trainModel/runs/<arm>/  ultralytics run per arm
         <project>/step7_evaluateModel/eval_*/   one VEGA evaluation per arm
         <project>/ablation/arms.json            the design, as executed
         <project>/ablation/progress.jsonl       one row per state change
         <project>/ablation/REPORT.md            the written comparison
         <project>/project.json, README.md       the VEGA project shape

This runner never touches the orchestrator on port 5050 and never starts a
VEGA step. It keeps the project's step 6 status at "pending" while it works,
because VEGA marks a step it finds in "running" state with no process of its
own as failed, and this project has no process of VEGA's.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

AST = timezone(timedelta(hours=-4))

REPO_DIR = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_DIR / "scripts" / "TCRMPtrain_oceankindCV"
PYTHON = REPO_DIR / "env" / "bin" / "python"
TRAIN_WRAPPER = TRAIN_DIR / "src" / "train_wrapper.py"
EVALUATE_RUN = TRAIN_DIR / "src" / "evaluate_run.py"

DEFAULT_SOURCE = REPO_DIR / "supporting_data" / "bbox_datasets" / "fish_photo_28feb25"
SPLIT_SEED_NAME = "fish_photo_28feb25"

# Scoring protocol, identical for every arm and for the reference model.
# Confidence 0.001 keeps the whole precision-recall curve, which is what mean
# average precision is the area under. 0.7 is the NMS IoU run3's own test
# evaluation used, so the reference number is reproduced rather than re-derived.
EVAL_CONF = 0.001
EVAL_IOU = 0.7
EVAL_BATCH = 8

# The published reference this ablation is measured against.
REFERENCE_NAME = "run3_heavy_augment"
REFERENCE_WEIGHTS = Path(
    "/home/bizon/Desktop/ai/oceankind_CV/fish_training/training_runs/"
    "run3_heavy_augment/weights/best.pt"
)
REFERENCE_MAP50 = 0.904
REFERENCE_MAP5095 = 0.835
# The preflight gate: the reference weights re-scored here must land within this
# many mAP50 points of the published number, or the dataset, the class order or
# the protocol is wrong and no arm would be comparable.
REFERENCE_TOLERANCE = 0.02

# run3_heavy_augment's arguments, read from its own args.yaml on 2026-09-16 AST.
BASE_ARGS: dict[str, Any] = {
    "task": "detect",
    "model": "yolo11l.pt",
    "epochs": 350,
    "imgsz": 640,
    "batch": 32,
    "patience": 50,
    "device": "0,1",
    "workers": 12,
    "optimizer": "auto",
    "seed": 0,
    "cos_lr": "false",
    "close_mosaic": 10,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "nbs": 64,
    "hsv_h": 0.3,
    "hsv_s": 0.5,
    "hsv_v": 0.4,
    "bgr": 0.0,
    "degrees": 15.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 2.0,
    "perspective": 0.0001,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.15,
    "copy_paste": 0.1,
    "copy_paste_mode": "flip",
    "erasing": 0.0,
    "auto_augment": "randaugment",
}

# Each arm names the keys it changes from the arm before it. An empty "changes"
# marks the replication. Nothing else may differ, and verify_arm_chain proves it.
ARM_CHAIN: list[dict[str, Any]] = [
    {
        "id": "A1_replicate",
        "changes": {},
        "isolates": "the ultralytics version, 8.3.222 for run3 against 8.4.38 here",
        "hypothesis": "scores within measurement noise of run3, confirming the "
                      "comparison chain is sound",
    },
    {
        "id": "A2_finish",
        "changes": {"patience": 400},
        "isolates": "whether the schedule completing, so the last 10 epochs train "
                    "with four-image stitching switched off, is worth anything",
        "hypothesis": "beats A1, because run3 never trained one epoch on whole "
                      "undistorted photographs and inference only ever sees those",
    },
    {
        "id": "A3_finish_lowhue",
        "changes": {"hsv_h": 0.05},
        "isolates": "hue jitter, cut from 0.30 to 0.05",
        "hypothesis": "beats A2 on the colour-phase classes, because 12 of the 34 "
                      "classes separate on colour and a 0.30 hue shift erases it",
    },
]

# Wall-clock guard. The runner refuses to start an arm it cannot finish.
DEFAULT_BUDGET_HOURS = 12.0
# Measured from run3: 225 epochs in 1.80 hours on this hardware. Padded, because
# the newer library and the wider evaluation cost a little more.
SECONDS_PER_EPOCH_ESTIMATE = 34.0
EVAL_SECONDS_ESTIMATE = 300.0


def ast_now() -> datetime:
    """Return the current time in Atlantic Standard Time."""
    return datetime.now(AST)


def stamp(when: datetime | None = None) -> str:
    """Return a human timestamp, ``YYYY-MM-DD HH:MM AST``."""
    return (when or ast_now()).strftime("%Y-%m-%d %H:%M AST")


def compact_stamp(when: datetime | None = None) -> str:
    """Return a filename-safe timestamp, ``YYYYMMDD-HHMMSS``."""
    return (when or ast_now()).strftime("%Y%m%d-%H%M%S")


def resolve_arms(chain: Iterable[Mapping[str, Any]] = ARM_CHAIN,
                 base: Mapping[str, Any] = BASE_ARGS) -> list[dict[str, Any]]:
    """Turn the chain of single-key changes into one full argument set per arm.

    Each arm inherits every argument from the arm before it and applies only the
    keys in its own ``changes``. The first arm inherits ``base`` unchanged.

    Parameters:
        chain: the ordered arm definitions, each with ``id`` and ``changes``.
        base: the reference argument set the first arm replicates.
    Returns:
        a list of dicts, each with id, args, changes, isolates and hypothesis.
    Raises:
        ValueError: when an arm changes a key that is not in the base argument
            set, which would silently introduce an argument the reference never
            had, or when two arms share an id.
    """
    resolved: list[dict[str, Any]] = []
    current = dict(base)
    seen: set[str] = set()
    for entry in chain:
        arm_id = entry["id"]
        if arm_id in seen:
            raise ValueError(f"two arms share the id {arm_id!r}")
        seen.add(arm_id)
        unknown = sorted(set(entry["changes"]) - set(base))
        if unknown:
            raise ValueError(
                f"arm {arm_id!r} changes {unknown}, which the reference argument "
                "set does not contain, so the change would not be an ablation of it"
            )
        current = {**current, **entry["changes"]}
        resolved.append({
            "id": arm_id,
            "args": dict(current),
            "changes": dict(entry["changes"]),
            "isolates": entry.get("isolates", ""),
            "hypothesis": entry.get("hypothesis", ""),
        })
    return resolved


def verify_arm_chain(arms: list[Mapping[str, Any]]) -> list[str]:
    """Return one plain sentence per arm naming exactly what differs from its
    predecessor, and raise when the difference is not what the arm claimed.

    This is the guard that keeps the ablation controlled: an arm whose resolved
    arguments differ from its predecessor in a key it did not declare is a
    confounded arm, and a result from it would not mean anything.

    Parameters:
        arms: the output of resolve_arms.
    Returns:
        a list of sentences, one per arm, in order.
    Raises:
        ValueError: when an arm's real difference is not its declared change.
    """
    lines: list[str] = []
    for i, arm in enumerate(arms):
        if i == 0:
            lines.append(f"{arm['id']}: replicates the reference exactly, no argument changed")
            continue
        prev = arms[i - 1]["args"]
        cur = arm["args"]
        real = {k for k in set(prev) | set(cur) if prev.get(k) != cur.get(k)}
        declared = set(arm["changes"])
        if real != declared:
            raise ValueError(
                f"arm {arm['id']!r} declares it changes {sorted(declared)} but really "
                f"changes {sorted(real)}; the arm is confounded and would prove nothing"
            )
        parts = ", ".join(
            f"{k} {prev.get(k)} to {cur.get(k)}" for k in sorted(real)
        )
        lines.append(f"{arm['id']}: from {arms[i - 1]['id']}, {parts}")
    return lines


def estimate_arm_seconds(args: Mapping[str, Any]) -> float:
    """Estimate the wall-clock cost of one arm, training plus its evaluation.

    An arm whose patience is below its epoch count may stop early, so the
    estimate is the full schedule, which is the worst case and the right basis
    for a budget guard.
    """
    return float(args["epochs"]) * SECONDS_PER_EPOCH_ESTIMATE + EVAL_SECONDS_ESTIMATE


class Progress:
    """Append-only progress log, one JSON object per line.

    Every state change is written and flushed immediately, so a reader watching
    the file sees the run advance without waiting for the process to end.
    """

    def __init__(self, path: Path) -> None:
        """Open the log at ``path``, creating its parent directory."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> None:
        """Append one event row carrying ``event`` and any extra fields."""
        row = {"ts": ast_now().isoformat(), "event": event, **fields}
        try:
            with self.path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
        except OSError as exc:
            raise OSError(f"cannot append to the progress log {self.path}: {exc}") from exc
        print(f"[{stamp()}] {event} {json.dumps(fields, default=str)}", flush=True)


def train_command(arm: Mapping[str, Any], data_yaml: Path, runs_dir: Path,
                  run_name: str) -> list[str]:
    """Build the argument list that trains one arm through VEGA's own wrapper.

    Using the platform's wrapper rather than calling ultralytics directly is
    deliberate: the run folder, the arguments recorded in it and the dataset
    contract are then identical to a run VEGA produced itself.
    """
    cmd = [str(PYTHON), str(TRAIN_WRAPPER),
           "--src", str(data_yaml),
           "--name", run_name,
           "--project", str(runs_dir)]
    for key, value in arm["args"].items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def evaluate_command(run_dir: Path, dataset_dir: Path, out_dir: Path,
                     imgsz: int) -> list[str]:
    """Build the argument list that scores one finished run on the test split."""
    return [str(PYTHON), str(EVALUATE_RUN),
            "--run_dir", str(run_dir),
            "--dataset_dir", str(dataset_dir),
            "--out_dir", str(out_dir),
            "--split", "test",
            "--imgsz", str(imgsz),
            "--batch", str(EVAL_BATCH),
            "--conf", str(EVAL_CONF),
            "--iou", str(EVAL_IOU),
            "--device", "0",
            "--preview_count", "0"]


def run_step(cmd: list[str], log_path: Path, cwd: Path) -> int:
    """Run one subprocess, streaming its output to ``log_path``.

    Returns the exit code. The child gets its own process group so a signal
    aimed at this runner does not reach a training job mid-epoch.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("a") as handle:
            handle.write(f"\n=== {stamp()} :: {' '.join(cmd)}\n")
            handle.flush()
            proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                    cwd=str(cwd), start_new_session=True)
            return proc.wait()
    except OSError as exc:
        raise OSError(f"cannot run {cmd[0]} with its log at {log_path}: {exc}") from exc


def read_metrics(metrics_path: Path) -> dict[str, float | None]:
    """Read the four headline box numbers out of a VEGA evaluation.

    Returns a dict with map50, map50_95, precision and recall, each None when
    the file does not carry it.
    """
    try:
        doc = json.loads(Path(metrics_path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read the evaluation at {metrics_path}: {exc}") from exc
    overall = doc.get("metrics", {}).get("overall", {})
    return {
        "map50": overall.get("box_map50"),
        "map50_95": overall.get("box_map50_95"),
        "precision": overall.get("box_precision"),
        "recall": overall.get("box_recall"),
    }


def epochs_completed(results_csv: Path) -> int:
    """Return how many epochs a run actually finished, 0 when it wrote nothing."""
    try:
        with Path(results_csv).open() as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except OSError:
        return 0


def project_json(project_dir: Path, name: str, source_dir: Path) -> dict[str, Any]:
    """Build the VEGA project record for this ablation.

    Step 6 is left "pending" on purpose while the ablation runs. VEGA decides a
    step is running by holding its child process in memory, and it marks a step
    recorded as running with no such process as failed. This project's training
    is not VEGA's child, so claiming "running" would make VEGA write an error
    into a healthy run the moment anyone opened the page.
    """
    now = ast_now().isoformat()
    step_names = {
        1: "Make All Points", 2: "Recode Species", 3: "Choose Images",
        4: "Place Points", 5: "Segment Images (SAM3)", 6: "Train Model",
        7: "Evaluate Model", 8: "Run Inference",
    }
    steps: dict[str, Any] = {}
    for number, label in step_names.items():
        if number <= 4:
            status, note = "skipped", f"from detection dataset {source_dir.name}"
        elif number == 5:
            status, note = "locked", "not used by a bounding-box training project"
        elif number == 6:
            status, note = "pending", "trained by the backend ablation runner, not by VEGA"
        else:
            status, note = "pending", ""
        steps[str(number)] = {
            "status": status,
            "name": label,
            "dir": str(project_dir / f"step{number}_{_step_slug(number)}"),
            "config": {},
            "outputs": {},
            "note": note,
        }
    return {
        "id": project_dir.name.replace("run_", "", 1),
        "name": name,
        "created_at": now,
        "updated_at": now,
        "project_dir": str(project_dir),
        "repo_dir": str(REPO_DIR),
        "project_type": "bbox_training",
        "current_step": 6,
        "steps": steps,
        "access_log": [{
            "ts": now,
            "initials": "LO",
            "action": "created",
            "purpose": "backend fish detector ablation, three arms, run outside VEGA",
        }],
    }


def _step_slug(number: int) -> str:
    """Return the folder suffix VEGA uses for a step number."""
    return {
        1: "makeAllPoints", 2: "recodeSpecies", 3: "chooseImages",
        4: "routeChosenImages", 5: "segmentImages", 6: "trainModel",
        7: "evaluateModel", 8: "runInference",
    }[number]


def write_json(path: Path, doc: Mapping[str, Any]) -> None:
    """Write ``doc`` to ``path`` as indented JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    except OSError as exc:
        raise OSError(f"cannot write {path}: {exc}") from exc


def build_report(results: list[Mapping[str, Any]], reference: Mapping[str, Any],
                 arms: list[Mapping[str, Any]], budget_hours: float,
                 started: datetime) -> str:
    """Render the comparison report a reader can act on without the raw files.

    Parameters:
        results: one row per arm, with id, status, metrics, epochs and hours.
        reference: the re-scored reference model's row.
        arms: the resolved arm definitions, for the design section.
        budget_hours: the wall-clock budget the run was given.
        started: when the run began.
    Returns:
        the report as markdown.
    """
    lines: list[str] = []
    ref_m = reference.get("metrics") or {}
    lines.append("# Fish detector ablation: does anything beat run3_heavy_augment")
    lines.append("")
    lines.append(f"Run by LO on the backend, outside VEGA. Started {stamp(started)}, "
                 f"budget {budget_hours:g} hours.")
    lines.append("")
    lines.append("Every arm trained on the same images and was scored on the same "
                 "held-out test split, at confidence 0.001 and non-maximum-suppression "
                 "IoU 0.7. The reference model's own stored weights were re-scored here "
                 "under the same library and the same protocol, so the comparison does "
                 "not rest on a number measured elsewhere.")
    lines.append("")

    lines.append("## The answer")
    lines.append("")
    scored = [r for r in results if (r.get("metrics") or {}).get("map50") is not None]
    base = ref_m.get("map50")
    if not scored:
        lines.append("No arm produced a score. The detail below says why.")
    elif base is None:
        lines.append("**No comparison is possible.** The reference model could not be "
                     "re-scored here, so nothing in the table below can be judged "
                     "against it. The arms' own scores still stand relative to each "
                     "other. The detail below says what failed.")
    else:
        best = max(scored, key=lambda r: r["metrics"]["map50"])
        delta = best["metrics"]["map50"] - base
        if delta > 0.005:
            lines.append(f"**{best['id']} beats the reference** by {delta * 100:.1f} "
                         f"mAP50 points, {best['metrics']['map50']:.4f} against "
                         f"{base:.4f}.")
        elif delta < -0.005:
            lines.append(f"**Nothing beat the reference.** The best arm, {best['id']}, "
                         f"scored {best['metrics']['map50']:.4f} against {base:.4f}, "
                         f"which is {abs(delta) * 100:.1f} points worse.")
        else:
            lines.append(f"**Nothing separated from the reference.** The best arm, "
                         f"{best['id']}, scored {best['metrics']['map50']:.4f} against "
                         f"{base:.4f}, a difference of {delta * 100:+.1f} points, which "
                         "one seed per arm cannot distinguish from noise.")
    lines.append("")

    lines.append("## Scores on the held-out test split")
    lines.append("")
    lines.append("| Model | mAP50 | mAP50-95 | Precision | Recall | Epochs | Hours |")
    lines.append("|---|---|---|---|---|---|---|")
    rows = [dict(reference, id=f"{REFERENCE_NAME} (reference, re-scored)")] + list(results)
    for row in rows:
        m = row.get("metrics") or {}
        def fmt(key: str) -> str:
            value = m.get(key)
            return f"{value:.4f}" if isinstance(value, (int, float)) else "not scored"
        ep = row.get("epochs")
        hrs = row.get("hours")
        lines.append(
            f"| {row.get('id')} | {fmt('map50')} | {fmt('map50_95')} | "
            f"{fmt('precision')} | {fmt('recall')} | "
            f"{ep if ep else 'n/a'} | {f'{hrs:.2f}' if isinstance(hrs, (int, float)) else 'n/a'} |"
        )
    lines.append("")
    lines.append(f"The reference model's published test scores were mAP50 "
                 f"{REFERENCE_MAP50:.3f} and mAP50-95 {REFERENCE_MAP5095:.3f}, measured "
                 "under ultralytics 8.3.222. The row above is the same weights re-scored "
                 "here under 8.4.38.")
    lines.append("")

    lines.append("## What each arm isolates")
    lines.append("")
    for i, arm in enumerate(arms):
        if i == 0:
            change = "no argument changed, this is the replication"
        else:
            change = ", ".join(f"{k} set to {v}" for k, v in arm["changes"].items())
        lines.append(f"- **{arm['id']}**: {change}. Isolates {arm['isolates']}. "
                     f"Expected: {arm['hypothesis']}.")
    lines.append("")

    lines.append("## Reading these numbers honestly")
    lines.append("")
    lines.append("- One seed per arm. A difference under about half a point is not "
                 "distinguishable from run-to-run variation, and nothing here was "
                 "repeated to measure that variation.")
    lines.append("- The arms form a chain, so each comparison is against its immediate "
                 "predecessor. A2 against A1 isolates the schedule. A3 against A2 "
                 "isolates hue. A1 against the reference isolates the library version.")
    lines.append("- The test split is the one the reference used, replayed from the "
                 "staged dataset's pins, so the held-out images are identical.")
    lines.append("- That split was made by image and not grouped by photographer or "
                 "capture session, so near-duplicate frames can sit on both sides. Every "
                 "score in the table, the reference included, is inflated by the same "
                 "unknown amount. The comparison between arms is still sound; the "
                 "absolute values are optimistic.")
    lines.append("")

    failures = [r for r in results if r.get("status") != "ok"]
    if failures:
        lines.append("## Arms that did not produce a score")
        lines.append("")
        for row in failures:
            lines.append(f"- **{row['id']}**: {row.get('status')}. "
                         f"{row.get('detail', '')}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry: build the project, run each arm, score it, write the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project_dir", required=True,
                        help="the VEGA-shaped project folder to create and fill")
    parser.add_argument("--source_dir", default=str(DEFAULT_SOURCE),
                        help="the staged YOLO box dataset the arms train on")
    parser.add_argument("--budget_hours", type=float, default=DEFAULT_BUDGET_HOURS,
                        help="wall-clock ceiling for the whole ablation")
    parser.add_argument("--skip_preflight", action="store_true",
                        help="do not re-score the reference model first (not advised: "
                             "the preflight is what proves the arms are comparable)")
    args = parser.parse_args(argv)

    project_dir = Path(args.project_dir)
    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print(f"ERROR: the staged dataset {source_dir} does not exist", file=sys.stderr)
        return 2

    started = ast_now()
    deadline = time.monotonic() + args.budget_hours * 3600.0
    progress = Progress(project_dir / "ablation" / "progress.jsonl")

    try:
        arms = resolve_arms()
        chain_lines = verify_arm_chain(arms)
    except ValueError as exc:
        print(f"ERROR: the ablation design is not controlled: {exc}", file=sys.stderr)
        return 2
    for line in chain_lines:
        progress.write("design", detail=line)

    step6 = project_dir / "step6_trainModel"
    step7 = project_dir / "step7_evaluateModel"
    dataset_dir = step6 / "dataset"
    runs_dir = step6 / "runs"
    logs_dir = project_dir / "ablation" / "logs"
    for folder in (step6, step7, runs_dir, logs_dir):
        folder.mkdir(parents=True, exist_ok=True)
    for number in range(1, 9):
        (project_dir / f"step{number}_{_step_slug(number)}").mkdir(exist_ok=True)

    write_json(project_dir / "project.json",
               project_json(project_dir, "TRAIN_fish_ablation", source_dir))
    write_json(project_dir / "ablation" / "arms.json", {
        "reference": {"name": REFERENCE_NAME, "weights": str(REFERENCE_WEIGHTS),
                      "published_map50": REFERENCE_MAP50,
                      "published_map50_95": REFERENCE_MAP5095},
        "protocol": {"split": "test", "conf": EVAL_CONF, "iou": EVAL_IOU},
        "chain": chain_lines,
        "arms": arms,
        "started": started.isoformat(),
        "budget_hours": args.budget_hours,
    })

    progress.write("dataset_build_start", source=str(source_dir))
    sys.path.insert(0, str(TRAIN_DIR / "src"))
    try:
        import detect_dataset  # noqa: PLC0415, imported here so the CLI stays importable
        manifest = detect_dataset.build_detect_dataset(
            str(source_dir), str(dataset_dir), SPLIT_SEED_NAME)
    except Exception as exc:  # the builder raises several kinds; none is recoverable here
        progress.write("dataset_build_failed", error=str(exc))
        print(f"ERROR: cannot build the split dataset: {exc}", file=sys.stderr)
        return 2
    progress.write("dataset_build_done",
                   counts=manifest.get("counts"),
                   pins_source=manifest.get("pins_source"),
                   holdout_mode=manifest.get("holdout_mode"),
                   fingerprint=manifest.get("split_fingerprint"))

    data_yaml = dataset_dir / "data.yaml"
    reference_row: dict[str, Any] = {"id": REFERENCE_NAME, "status": "skipped",
                                     "metrics": {}, "epochs": None, "hours": None}

    if not args.skip_preflight:
        if not REFERENCE_WEIGHTS.is_file():
            progress.write("preflight_failed", error=f"{REFERENCE_WEIGHTS} is missing")
            print(f"ERROR: the reference weights {REFERENCE_WEIGHTS} are missing; "
                  "without them no arm can be compared", file=sys.stderr)
            return 2
        ref_run_dir = runs_dir / f"REF_{REFERENCE_NAME}"
        (ref_run_dir / "weights").mkdir(parents=True, exist_ok=True)
        target = ref_run_dir / "weights" / "best.pt"
        if not target.exists():
            os.symlink(REFERENCE_WEIGHTS, target)
        ref_out = step7 / f"eval_REF_{REFERENCE_NAME}"
        progress.write("preflight_start", weights=str(REFERENCE_WEIGHTS))
        code = run_step(evaluate_command(ref_run_dir, dataset_dir, ref_out,
                                         BASE_ARGS["imgsz"]),
                        logs_dir / "preflight.log", REPO_DIR)
        if code != 0:
            progress.write("preflight_failed", exit_code=code)
            print("ERROR: the reference model could not be re-scored; stopping before "
                  "spending training hours on a comparison that would not hold",
                  file=sys.stderr)
            return 2
        metrics = read_metrics(ref_out / "metrics.json")
        reference_row = {"id": REFERENCE_NAME, "status": "ok", "metrics": metrics,
                         "epochs": None, "hours": None}
        gap = abs((metrics.get("map50") or 0.0) - REFERENCE_MAP50)
        progress.write("preflight_done", metrics=metrics, gap_from_published=round(gap, 4))
        if gap > REFERENCE_TOLERANCE:
            progress.write("preflight_gate_failed", gap=round(gap, 4),
                           tolerance=REFERENCE_TOLERANCE)
            print(f"ERROR: the reference re-scored at mAP50 {metrics.get('map50')}, "
                  f"{gap:.4f} away from its published {REFERENCE_MAP50}. The dataset, "
                  "the class order or the protocol does not match, so no arm would be "
                  "comparable. Stopping.", file=sys.stderr)
            return 2

    results: list[dict[str, Any]] = []
    for arm in arms:
        need = estimate_arm_seconds(arm["args"])
        left = deadline - time.monotonic()
        if left < need:
            progress.write("arm_skipped", arm=arm["id"],
                           seconds_left=round(left), seconds_needed=round(need))
            results.append({"id": arm["id"], "status": "skipped for time",
                            "detail": f"needed about {need / 3600:.1f} hours and "
                                      f"{left / 3600:.1f} remained in the budget",
                            "metrics": {}, "epochs": None, "hours": None})
            continue

        run_name = f"TRAIN_fish_{arm['id']}_{compact_stamp()}"
        arm_started = time.monotonic()
        progress.write("arm_train_start", arm=arm["id"], run_name=run_name,
                       changes=arm["changes"])
        code = run_step(train_command(arm, data_yaml, runs_dir, run_name),
                        logs_dir / f"{arm['id']}_train.log", TRAIN_DIR)
        run_dir = runs_dir / run_name
        done = epochs_completed(run_dir / "results.csv")
        hours = (time.monotonic() - arm_started) / 3600.0
        if code != 0 or not (run_dir / "weights" / "best.pt").is_file():
            progress.write("arm_train_failed", arm=arm["id"], exit_code=code,
                           epochs=done)
            results.append({"id": arm["id"], "status": "training failed",
                            "detail": f"the trainer exited {code} after {done} epochs; "
                                      f"its log is ablation/logs/{arm['id']}_train.log",
                            "metrics": {}, "epochs": done, "hours": round(hours, 2)})
            continue
        progress.write("arm_train_done", arm=arm["id"], epochs=done,
                       hours=round(hours, 2))

        out_dir = step7 / f"eval_{run_name}"
        progress.write("arm_eval_start", arm=arm["id"])
        code = run_step(evaluate_command(run_dir, dataset_dir, out_dir,
                                         arm["args"]["imgsz"]),
                        logs_dir / f"{arm['id']}_eval.log", REPO_DIR)
        if code != 0:
            progress.write("arm_eval_failed", arm=arm["id"], exit_code=code)
            results.append({"id": arm["id"], "status": "evaluation failed",
                            "detail": f"training finished but scoring exited {code}",
                            "metrics": {}, "epochs": done, "hours": round(hours, 2)})
            continue
        metrics = read_metrics(out_dir / "metrics.json")
        progress.write("arm_eval_done", arm=arm["id"], metrics=metrics)
        results.append({"id": arm["id"], "status": "ok", "metrics": metrics,
                        "epochs": done, "hours": round(hours, 2),
                        "run_name": run_name})

        report_path = project_dir / "ablation" / "REPORT.md"
        report_path.write_text(build_report(results, reference_row, arms,
                                            args.budget_hours, started))

    report_path = project_dir / "ablation" / "REPORT.md"
    report_path.write_text(build_report(results, reference_row, arms,
                                        args.budget_hours, started))
    write_json(project_dir / "ablation" / "results.json",
               {"reference": reference_row, "arms": results,
                "finished": ast_now().isoformat()})

    doc = json.loads((project_dir / "project.json").read_text())
    scored = [r for r in results if r.get("status") == "ok"]
    doc["steps"]["6"]["status"] = "completed" if scored else "error"
    doc["steps"]["7"]["status"] = "completed" if scored else "pending"
    doc["current_step"] = 7 if scored else 6
    doc["updated_at"] = ast_now().isoformat()
    doc["access_log"].append({
        "ts": ast_now().isoformat(), "initials": "LO", "action": "ablation finished",
        "purpose": f"{len(scored)} of {len(arms)} arms scored",
    })
    write_json(project_dir / "project.json", doc)

    progress.write("finished", scored=len(scored), of=len(arms),
                   report=str(report_path))
    print(f"Step complete: {len(scored)} of {len(arms)} arms scored. "
          f"Report at {report_path}")
    return 0 if scored else 1


if __name__ == "__main__":
    sys.exit(main())

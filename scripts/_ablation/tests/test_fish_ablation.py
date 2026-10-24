#!/usr/bin/env python3
"""House tests for the fish detector ablation runner.

Module:  scripts/_ablation/tests/test_fish_ablation.py
Purpose: Prove the ablation is controlled (each arm differs from its neighbour
         in exactly the keys it declares), that the budget guard refuses an arm
         it cannot finish, that the commands handed to the trainer and the
         scorer carry the right flags, that the project record is the shape VEGA
         reads, and that the report tells the truth when arms fail.
Inputs:  none; every fixture is built in a temporary directory.
Outputs: a pass or fail count on stdout, exit code 1 on any failure.

Run:     github_repo/env/bin/python scripts/_ablation/tests/test_fish_ablation.py
         (/usr/bin/python3 also works; nothing here needs a third-party package)
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

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fish_ablation as fa  # noqa: E402

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
    """Print a section heading so a failure is easy to place."""
    print(f"\n--- {title}")


# ── the design is controlled ────────────────────────────────────────────────
section("the arm chain is a controlled ablation")

arms = fa.resolve_arms()
check(len(arms) == 3, "three arms resolve")
check([a["id"] for a in arms] == ["A1_replicate", "A2_finish", "A3_finish_lowhue"],
      "the arms are in the declared order")

check(arms[0]["args"] == fa.BASE_ARGS,
      "A1 replicates the reference argument set exactly, key for key")

a1, a2, a3 = (a["args"] for a in arms)
diff_12 = {k for k in set(a1) | set(a2) if a1.get(k) != a2.get(k)}
diff_23 = {k for k in set(a2) | set(a3) if a2.get(k) != a3.get(k)}
check(diff_12 == {"patience"}, f"A2 differs from A1 in patience alone, got {sorted(diff_12)}")
check(diff_23 == {"hsv_h"}, f"A3 differs from A2 in hue alone, got {sorted(diff_23)}")

check(a2["patience"] > a2["epochs"],
      "A2's patience exceeds its epoch count, so early stopping cannot fire and the "
      "close_mosaic phase is reached")
check(a1["patience"] < a1["epochs"],
      "A1 keeps the reference's patience, which is what let the reference stop early")
check(a3["hsv_h"] < a2["hsv_h"], "A3 lowers hue rather than raising it")

lines = fa.verify_arm_chain(arms)
check(len(lines) == 3, "the chain describes every arm")
check("replicates the reference exactly" in lines[0], "A1 is described as the replication")
check("patience 50 to 400" in lines[1], f"A2's line names the real change, got {lines[1]!r}")
check("hsv_h 0.3 to 0.05" in lines[2], f"A3's line names the real change, got {lines[2]!r}")


section("a confounded or malformed design is refused, not run")

try:
    fa.resolve_arms([{"id": "X", "changes": {"not_a_real_argument": 1}}])
    check(False, "an arm changing an argument the reference lacks is refused")
except ValueError as exc:
    check("does not contain" in str(exc),
          "changing an unknown argument is refused with a message naming it")

try:
    fa.resolve_arms([{"id": "X", "changes": {}}, {"id": "X", "changes": {}}])
    check(False, "two arms sharing an id are refused")
except ValueError as exc:
    check("share the id" in str(exc), "a duplicated arm id is refused")

confounded = [
    {"id": "B1", "args": {"epochs": 10, "hsv_h": 0.3}, "changes": {}},
    # declares it changes epochs, really changes hue as well
    {"id": "B2", "args": {"epochs": 20, "hsv_h": 0.1}, "changes": {"epochs": 20}},
]
try:
    fa.verify_arm_chain(confounded)
    check(False, "a confounded arm is refused")
except ValueError as exc:
    check("confounded" in str(exc) and "hsv_h" in str(exc),
          "a confounded arm is refused with the undeclared key named")

single = fa.resolve_arms([{"id": "only", "changes": {}}], {"epochs": 5})
check(fa.verify_arm_chain(single) == ["only: replicates the reference exactly, no argument changed"],
      "a one-arm chain is described without reaching for a predecessor")


# ── the budget guard ────────────────────────────────────────────────────────
section("the wall-clock guard")

cost = fa.estimate_arm_seconds({"epochs": 350})
check(cost > 350 * 30, "an arm's estimate covers the full schedule, not the early stop")
check(cost == 350 * fa.SECONDS_PER_EPOCH_ESTIMATE + fa.EVAL_SECONDS_ESTIMATE,
      "the estimate is training plus one evaluation")
check(fa.estimate_arm_seconds({"epochs": 700}) > fa.estimate_arm_seconds({"epochs": 350}),
      "a longer schedule estimates higher")
check(fa.estimate_arm_seconds({"epochs": 0}) == fa.EVAL_SECONDS_ESTIMATE,
      "a zero-epoch arm still costs its evaluation")

three_arms_hours = sum(fa.estimate_arm_seconds(a["args"]) for a in arms) / 3600.0
check(three_arms_hours < fa.DEFAULT_BUDGET_HOURS,
      f"the three arms fit the default budget, estimate {three_arms_hours:.1f} hours "
      f"against {fa.DEFAULT_BUDGET_HOURS:g}")


# ── the commands handed to the platform's own tools ─────────────────────────
section("the trainer command")

tmp = Path(tempfile.mkdtemp(prefix="fish_abl_"))
try:
    cmd = fa.train_command(arms[1], tmp / "data.yaml", tmp / "runs", "TRAIN_x")
    check(str(fa.TRAIN_WRAPPER) in cmd, "the platform's own train wrapper is invoked")
    check("--src" in cmd and str(tmp / "data.yaml") in cmd, "the dataset is passed")
    check("--name" in cmd and "TRAIN_x" in cmd, "the run name is passed")
    check(cmd[cmd.index("--patience") + 1] == "400", "A2's patience reaches the trainer")
    check(cmd[cmd.index("--hsv_h") + 1] == "0.3", "A2 keeps the reference hue")
    check(cmd[cmd.index("--device") + 1] == "0,1", "both cards are used")
    check(cmd[cmd.index("--close_mosaic") + 1] == "10",
          "close_mosaic reaches the trainer, which is the whole point of A2")

    missing = [k for k in fa.BASE_ARGS if f"--{k}" not in cmd]
    check(not missing, f"every argument in the design reaches the trainer, missing {missing}")

    cmd3 = fa.train_command(arms[2], tmp / "data.yaml", tmp / "runs", "TRAIN_y")
    check(cmd3[cmd3.index("--hsv_h") + 1] == "0.05", "A3's lowered hue reaches the trainer")

    section("the scorer command")
    ev = fa.evaluate_command(tmp / "run", tmp / "dataset", tmp / "out", 640)
    check(str(fa.EVALUATE_RUN) in ev, "the platform's own evaluator is invoked")
    check(ev[ev.index("--split") + 1] == "test", "scoring uses the held-out test split")
    check(ev[ev.index("--conf") + 1] == "0.001",
          "scoring keeps the whole precision-recall curve")
    check(ev[ev.index("--iou") + 1] == "0.7",
          "scoring uses the same NMS IoU the reference's own test evaluation used")
    check(ev[ev.index("--imgsz") + 1] == "640", "the arm's own resolution is scored at")

    # ── reading a VEGA evaluation back ──────────────────────────────────────
    section("reading an evaluation")
    good = tmp / "metrics.json"
    good.write_text(json.dumps({"metrics": {"overall": {
        "box_map50": 0.9042, "box_map50_95": 0.8351,
        "box_precision": 0.895, "box_recall": 0.835}}}))
    m = fa.read_metrics(good)
    check(m["map50"] == 0.9042 and m["recall"] == 0.835, "the four headline numbers are read")

    empty = tmp / "empty.json"
    empty.write_text(json.dumps({"metrics": {"overall": {}}}))
    check(fa.read_metrics(empty)["map50"] is None,
          "an evaluation with no box numbers reads as None, not as zero")

    seg_only = tmp / "seg.json"
    seg_only.write_text(json.dumps({"metrics": {"overall": {
        "mask_map50": 0.5, "box_map50": None}}}))
    check(fa.read_metrics(seg_only)["map50"] is None,
          "a mask-only evaluation does not masquerade as a box score")

    for bad, label in ((tmp / "nope.json", "a missing file"),
                       (tmp / "bad.json", "a malformed file")):
        if label == "a malformed file":
            bad.write_text("{not json")
        try:
            fa.read_metrics(bad)
            check(False, f"{label} is refused")
        except ValueError as exc:
            check("cannot read the evaluation" in str(exc),
                  f"{label} is refused with a message naming the path")

    section("counting epochs")
    csv = tmp / "results.csv"
    csv.write_text("epoch,time\n1,10\n2,20\n3,30\n")
    check(fa.epochs_completed(csv) == 3, "a results file counts its data rows")
    csv.write_text("epoch,time\n")
    check(fa.epochs_completed(csv) == 0, "a header-only results file counts zero")
    check(fa.epochs_completed(tmp / "absent.csv") == 0,
          "a missing results file counts zero rather than raising")

    # ── the VEGA project record ─────────────────────────────────────────────
    section("the project record is the shape VEGA reads")
    proj = tmp / "run_26-09-16_LO_fish_ablation_backend"
    doc = fa.project_json(proj, "TRAIN_fish_ablation", Path("/x/fish_photo_28feb25"))
    for key in ("id", "name", "created_at", "updated_at", "project_dir",
                "current_step", "steps", "access_log"):
        check(key in doc, f"project.json carries {key}")
    check(doc["id"] == "26-09-16_LO_fish_ablation_backend",
          "the project id drops the run_ prefix, as VEGA's do")
    check(set(doc["steps"]) == {str(n) for n in range(1, 9)},
          "all eight steps are recorded")
    check(doc["steps"]["6"]["status"] == "pending",
          "step 6 is NOT recorded as running, which would make VEGA mark this "
          "healthy run as failed the moment the page was opened")
    check(doc["steps"]["6"]["status"] != "running", "step 6 never claims to be VEGA's child")
    check(doc["steps"]["5"]["status"] == "locked", "the segmentation step is locked")
    check(all(doc["steps"][str(n)]["status"] == "skipped" for n in range(1, 5)),
          "the point-annotation steps are skipped, as they are for any box project")
    check(doc["steps"]["6"]["dir"].endswith("step6_trainModel"),
          "step 6 points at the folder name VEGA expects")
    check(doc["access_log"][0]["initials"] == "LO", "the access log is signed LO")
    check("-04:00" in doc["created_at"], "timestamps carry the Atlantic Standard offset")

    # ── the report ──────────────────────────────────────────────────────────
    section("the report")
    ref = {"id": fa.REFERENCE_NAME, "status": "ok", "metrics":
           {"map50": 0.9042, "map50_95": 0.8351, "precision": 0.895, "recall": 0.835},
           "epochs": None, "hours": None}
    started = datetime(2026, 9, 16, 1, 0, tzinfo=timezone(timedelta(hours=-4)))

    won = [{"id": "A2_finish", "status": "ok", "epochs": 350, "hours": 3.2,
            "metrics": {"map50": 0.9210, "map50_95": 0.8500,
                        "precision": 0.9, "recall": 0.85}}]
    text = fa.build_report(won, ref, arms, 12.0, started)
    check("beats the reference" in text, "a real gain is reported as a win")
    check("1.7 mAP50 points" in text, f"the size of the win is stated, got:\n{text[:400]}")

    lost = [{"id": "A2_finish", "status": "ok", "epochs": 350, "hours": 3.2,
             "metrics": {"map50": 0.8600, "map50_95": 0.80,
                         "precision": 0.9, "recall": 0.85}}]
    check("Nothing beat the reference" in fa.build_report(lost, ref, arms, 12.0, started),
          "a loss is reported as a loss, not spun")

    tie = [{"id": "A2_finish", "status": "ok", "epochs": 350, "hours": 3.2,
            "metrics": {"map50": 0.9050, "map50_95": 0.836,
                        "precision": 0.9, "recall": 0.85}}]
    tie_text = fa.build_report(tie, ref, arms, 12.0, started)
    check("Nothing separated from the reference" in tie_text,
          "a difference inside noise is called a tie, not a win")
    check("one seed per arm" in tie_text, "the tie verdict says why it cannot be called")

    failed = [{"id": "A1_replicate", "status": "training failed", "epochs": 12,
               "hours": 0.2, "metrics": {}, "detail": "the trainer exited 1"},
              {"id": "A2_finish", "status": "skipped for time", "epochs": None,
               "hours": None, "metrics": {}, "detail": "needed 3.3 hours"}]
    ftext = fa.build_report(failed, ref, arms, 12.0, started)
    check("No arm produced a score" in ftext, "an all-failed run says so plainly")
    check("Arms that did not produce a score" in ftext, "failures get their own section")
    check("the trainer exited 1" in ftext and "needed 3.3 hours" in ftext,
          "each failure's reason is carried into the report")
    check("not scored" in ftext, "a failed arm shows as not scored rather than as zero")

    no_ref = dict(ref, metrics={}, status="skipped")
    no_ref_text = fa.build_report(won, no_ref, arms, 12.0, started)
    check("No comparison is possible" in no_ref_text,
          "without the reference re-scored, the report refuses to declare a winner")
    check("beats the reference" not in no_ref_text,
          "and it does not quietly declare a win against a reference it never measured")

    check("near-duplicate frames can sit on both sides" in text,
          "the report carries the split-leakage caveat, so the score is not overread")
    check("mAP50 0.904" in text, "the published reference number is stated for contrast")

    for arm in arms:
        check(arm["id"] in text, f"the design section names {arm['id']}")

    section("writing json")
    target = tmp / "deep" / "nested" / "out.json"
    fa.write_json(target, {"a": 1})
    check(json.loads(target.read_text())["a"] == 1,
          "write_json creates parent directories and writes readable json")

    section("the progress log")
    log = fa.Progress(tmp / "prog" / "progress.jsonl")
    log.write("arm_train_start", arm="A1_replicate")
    log.write("arm_train_done", arm="A1_replicate", epochs=225)
    rows = [json.loads(l) for l in (tmp / "prog" / "progress.jsonl").read_text().splitlines()]
    check(len(rows) == 2, "each event is one line")
    check(rows[0]["event"] == "arm_train_start" and rows[1]["epochs"] == 225,
          "events carry their fields")
    check("-04:00" in rows[0]["ts"], "progress rows are timestamped in Atlantic Standard Time")

    section("timestamps")
    check(fa.ast_now().utcoffset() == timedelta(hours=-4), "now() is Atlantic Standard Time")
    check(len(fa.compact_stamp()) == 15 and "-" in fa.compact_stamp(),
          "the compact stamp is YYYYMMDD-HHMMSS")
    check(fa.stamp(started) == "2026-09-16 01:00 AST", "the human stamp is AST 24-hour")

    section("the tools this runner leans on exist")
    check(fa.PYTHON.is_file(), f"the module python is at {fa.PYTHON}")
    check(fa.TRAIN_WRAPPER.is_file(), f"the train wrapper is at {fa.TRAIN_WRAPPER}")
    check(fa.EVALUATE_RUN.is_file(), f"the evaluator is at {fa.EVALUATE_RUN}")
    check((fa.TRAIN_DIR / "yolo11l.pt").is_file(),
          "yolo11l.pt is staged where the trainer resolves a bare file name")

finally:
    shutil.rmtree(tmp, ignore_errors=True)
    check(not tmp.exists(), "the temporary fixtures were removed")

print(f"\n{CHECKS - FAILURES}/{CHECKS} passed.")
sys.exit(1 if FAILURES else 0)

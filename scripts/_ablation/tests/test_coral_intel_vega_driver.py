#!/usr/bin/env python3
"""House tests for the Coral Intel VEGA driver.

Module:  scripts/_ablation/tests/test_coral_intel_vega_driver.py
Purpose: Prove the design is controlled (the replicate changes nothing, every
         challenger changes one factor), that the driver talks to VEGA the way
         the Train and Evaluate panels do (open, save settings, Start, poll),
         that it waits out a busy answer and a Metashape run, that it resumes
         without retraining a finished arm and without launching a second
         trainer over a live one, that failures and a lost process are
         reported and never scored, and that the share folder gets one PDF per
         scored arm plus the summary.
Inputs:  none; VEGA is an in-process fake and every fixture sits in a temp dir.
Outputs: a pass or fail count on stdout, exit code 1 on any failure.

Run:     github_repo/env/bin/python scripts/_ablation/tests/test_coral_intel_vega_driver.py
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

import io
import json
import shutil
import sys
import tempfile
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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


def raises(fn, exc_type) -> bool:
    """Return True when fn() raises exc_type."""
    try:
        fn()
    except exc_type:
        return True
    return False


class FakeResponse(io.BytesIO):
    """A urlopen response: readable bytes with a status and context manager."""

    def __init__(self, status: int, doc: dict) -> None:
        super().__init__(json.dumps(doc).encode())
        self.status = status


class FakeVega:
    """An in-process stand-in for the orchestrator's web API.

    Training runs for ``train_polls`` status polls, then exits with
    ``train_exit``; on exit 0 it writes weights/best.pt. Evaluation writes a
    metrics.json and report.pdf. ``busy_once`` answers the first Start with 409.
    """

    def __init__(self, root: Path, train_polls: int = 2, train_exit: int = 0,
                 eval_exit: int = 0, busy_once: bool = False, lose_process: bool = False,
                 hostile_body: bool = False, refuse_409: bool = False,
                 metrics_on_exit2: bool = False, restart_once: bool = False) -> None:
        self.root = root
        self.train_polls = train_polls
        self.train_exit = train_exit
        self.eval_exit = eval_exit
        self.busy_once = busy_once
        self.lose_process = lose_process
        self.hostile_body = hostile_body
        self.refuse_409 = refuse_409
        self.metrics_on_exit2 = metrics_on_exit2
        self.restart_once = restart_once
        self.calls: list[tuple[str, str, dict]] = []
        self.open_dir = ""
        self.opened: list[str] = []
        # Per project: whether step 6 is recorded running and step 7 unlocked,
        # as VEGA keeps them in each project.json.
        self.step6_running: dict[str, bool] = {}
        self.step7_unlocked: dict[str, bool] = {}
        self.configs: dict[str, dict] = {"6": {}, "7": {}}
        self.proc: dict[int, dict] = {}
        self.starts = {6: 0, 7: 0}

    def __call__(self, request, timeout=None):
        method = request.get_method()
        path = request.full_url.replace(cd.VEGA_URL, "")
        body = json.loads(request.data.decode()) if request.data else {}
        self.calls.append((method, path, body))
        status, doc = self.route(method, path, body)
        if status >= 400:
            payload = b"<html>not json</html>" if self.hostile_body else json.dumps(doc).encode()
            raise urllib.error.HTTPError(request.full_url, status, "err", {}, io.BytesIO(payload))
        return FakeResponse(status, doc)

    def route(self, method, path, body):
        if path == "/boxes/api/create-project":
            run = self.root / "run_26-09-26_LO_train_coral_intel_t2aug"
            run.mkdir(parents=True, exist_ok=True)
            (run / "project.json").write_text("{}")
            return 200, {"ok": True, "run_dir": str(run)}
        if path == "/api/project/state":
            return 200, ({"loaded": True, "state": {"project_dir": self.open_dir}}
                         if self.open_dir else {"loaded": False})
        if path == "/api/project/open":
            self.opened.append(body["project_dir"])
            self.open_dir = body["project_dir"]
            return 200, {"success": True, "state": {"project_dir": self.open_dir}}
        if path.startswith("/api/project/config/"):
            self.configs[path.rsplit("/", 1)[1]].update(body)
            return 200, {"success": True}
        if path.endswith("/run"):
            step = int(path.split("/")[3])
            self.starts[step] += 1
            if self.busy_once and self.starts[step] == 1:
                return 409, {"error": "Train Model is already running", "busy": True}
            if self.refuse_409:
                return 409, {"error": "this project already trains from another source"}
            if step == 7 and not self.step7_unlocked.get(self.open_dir):
                return 400, {"error": "Step 7 is locked"}
            if step == 6:
                self.step6_running[self.open_dir] = True
            # Like VEGA, results go to the project that pressed Start.
            self.proc[step] = {"polls": 0, "owner": self.open_dir}
            return 200, {"success": True, "pid": 4242}
        if path.endswith("/status"):
            return 200, self.poll(int(path.split("/")[3]))
        return 404, {"error": "no such route"}

    def poll(self, step):
        proc = self.proc.get(step)
        if proc is None:
            return {"running": False, "exit_code": None}
        if "result" in proc:
            # VEGA keeps a finished process's exit code; the finish is recorded
            # only on a poll while its owner is open and still marked running.
            self.record(step, proc)
            return dict(proc["result"])
        proc["polls"] += 1
        if step == 6 and self.restart_once and proc["polls"] == 1:
            return {"running": False, "exit_code": None}
        if step == 6 and self.restart_once and proc["polls"] == 2:
            return {"running": True, "exit_code": None, "orphan_pid": 777}
        if step == 6 and proc["polls"] < self.train_polls:
            return {"running": True, "exit_code": None}
        if self.lose_process:
            del self.proc[step]
            return {"running": False, "exit_code": None, "untracked": True}
        project = Path(proc.get("owner") or self.open_dir)
        if step == 6:
            run = project / "step6_trainModel" / "runs" / self.configs["6"]["run_name"]
            if self.train_exit == 0:
                (run / "weights").mkdir(parents=True, exist_ok=True)
                (run / "weights" / "best.pt").write_bytes(b"w")
            proc["result"] = {"running": False, "exit_code": self.train_exit}
            self.record(step, proc)
            return dict(proc["result"])
        run_name = Path(self.configs["7"]["run_dir"]).name
        out = project / "step7_evaluateModel" / f"eval_{run_name}"
        if self.eval_exit == 0 or self.metrics_on_exit2:
            out.mkdir(parents=True, exist_ok=True)
            (out / "report.pdf").write_bytes(b"%PDF-fake")
            (out / "metrics.json").write_text(json.dumps({
                "metrics": {"overall": {"box_map50": 0.8, "box_map50_95": 0.7,
                                        "box_precision": 0.85, "box_recall": 0.75},
                            "per_class": {"Diploria-clivosa": {"box_map50": 0.5}},
                            "per_class_instances": {"Diploria-clivosa": 9}},
                "context": {"epochs_completed": 300, "best_epoch": 250,
                            "split_counts": {"train": 21000, "valid": 1554, "test": 1554}}}))
        del self.proc[step]
        return {"running": False, "exit_code": self.eval_exit}

    def record(self, step, proc):
        """VEGA's step_status: record a step 6 finish for the open project only."""
        owner = proc.get("owner") or self.open_dir
        if step != 6 or self.open_dir != owner or not self.step6_running.get(owner):
            return
        self.step6_running[owner] = False
        if proc["result"]["exit_code"] == 0:
            self.step7_unlocked[owner] = True


PROTOCOL = cd.load_eval_protocol()


def make_driver(fake: FakeVega, project: Path, blocker=lambda: "",
                finished=lambda run_dir: False, annotators=lambda: "",
                complete=None) -> cd.Driver:
    """Build a Driver over the fake with instant sleeps and no real ss or pgrep.

    By default recording a recovered finish unlocks Evaluate in the fake, as
    VEGA's complete_step does in project.json."""
    client = cd.VegaClient(cd.VEGA_URL, opener=fake)
    progress = cd.Progress(project / cd.STUDY_DIRNAME / "progress.jsonl")
    if complete is None:
        def complete(project_dir):
            fake.step7_unlocked[str(project_dir)] = True
    return cd.Driver(client, project, progress, PROTOCOL, sleep=lambda s: None,
                     blocker=blocker, finished=finished, annotators=annotators,
                     complete=complete)


def write_preset(folder: Path, name: str, meta: dict, params: dict) -> Path:
    """Write a preset YAML fixture and return its path."""
    import yaml
    path = folder / name
    path.write_text(yaml.safe_dump({"_meta": meta, "params": params}))
    return path


def events(project: Path) -> list[str]:
    """Return the event names written to the progress log, in order."""
    path = project / cd.STUDY_DIRNAME / "progress.jsonl"
    return [json.loads(ln)["event"] for ln in path.read_text().splitlines()]


tmp = Path(tempfile.mkdtemp(prefix="coral_driver_test_"))
try:
    section("the design is read from the preset files and is controlled")
    arms = cd.resolve_arms()
    check([a["id"] for a in arms] == ["A0_replicate", "B1_yolo11x", "B2_imgsz1024",
                                      "B3_offline_only"], "four arms, replicate first")
    check(all(Path(a["path"]).parent == cd.STUDY_PRESET_DIR for a in arms),
          "every arm comes from a file in the study's params folder")
    check(not list((cd.CONFIG_DIR / "presets").glob("coral_intel_*")),
          "no study recipe sits in the Train panel's own recipe folder")
    base = arms[0]["config"]
    check(arms[0]["changes"] == {}, "the replicate changes nothing")
    check(arms[1]["changes"] == {"model": "yolo11x.pt"}, "B1 changes only the model")
    check(arms[2]["changes"] == {"imgsz": 1024}, "B2 changes only the image size")
    check(set(arms[3]["changes"]) == {"hsv_h", "hsv_s", "hsv_v", "translate", "scale",
                                      "fliplr", "mosaic"}
          and all(v == 0 for v in arms[3]["changes"].values()),
          "B3 changes only online augmentation knobs, all to zero")
    check((base["model"], base["epochs"], base["imgsz"], base["batch"], base["seed"])
          == ("yolov8x.pt", 300, 640, 16, 32), "the replicate carries Table 2's values")
    check(base["task"] == "detect" and base["device"] == "0,1" and base["batch"] >= 1,
          "detect task, both cards, an explicit batch (VEGA refuses auto batch under DDP)")
    check(base["mosaic"] == 1.0 and base["hsv_s"] == 0.7 and base["flipud"] == 0.0
          and base["patience"] == 100,
          "online augmentation and patience are Ultralytics' defaults, not VEGA's panel defaults")
    check(base["valid_ratio"] == 0.15 and base["test_ratio"] == 0.15, "70/15/15 fractions")
    check(cd.describe_change(arms[2]) == "imgsz 640 to 1024", "the change reads in plain words")
    check(PROTOCOL["split"] == "test" and PROTOCOL["conf_threshold"] == 0.001
          and PROTOCOL["iou_threshold"] == 0.6 and PROTOCOL["batch"] == 8,
          "the Evaluate preset is the study protocol")

    section("a broken or confounded preset is refused")
    pdir = tmp / "presets"
    pdir.mkdir()
    good = dict(base)
    a0 = write_preset(pdir, "a0.yaml", {"arm": "A0"}, good)
    def refused(*files) -> bool:
        return raises(lambda: cd.resolve_arms(list(files)), ValueError)
    check(not refused(a0), "a lone good replicate loads")
    check(refused(pdir / "missing.yaml"), "a missing file is refused")
    bad_yaml = pdir / "bad.yaml"
    bad_yaml.write_text("_meta: [unclosed\n")
    check(refused(bad_yaml), "a file that does not parse is refused")
    listy = pdir / "list.yaml"
    listy.write_text("- 1\n- 2\n")
    check(refused(listy), "a file holding a list is refused")
    check(refused(write_preset(pdir, "noparams.yaml", {"arm": "A0"}, {})),
          "empty params are refused")
    check(refused(a0, write_preset(pdir, "undeclared.yaml", {"arm": "B"},
                                   dict(good, imgsz=1024))),
          "a real change the file does not declare is refused")
    check(refused(a0, write_preset(pdir, "confounded.yaml",
                                   {"arm": "B", "changes_from_replicate": ["imgsz"]},
                                   dict(good, imgsz=1024, batch=8))),
          "a second, undeclared change confounds the arm and is refused")
    check(refused(a0, write_preset(pdir, "fake.yaml",
                                   {"arm": "B", "changes_from_replicate": ["imgsz"]}, good)),
          "a declared change that is not real is refused")
    check(refused(a0, write_preset(pdir, "extra.yaml",
                                   {"arm": "B", "changes_from_replicate": ["dropout"]},
                                   dict(good, dropout=0.1))),
          "a key the replicate lacks is refused")
    check(refused(write_preset(pdir, "a0c.yaml",
                               {"arm": "A0", "changes_from_replicate": ["imgsz"]}, good)),
          "a replicate that declares changes is refused")
    check(refused(a0, write_preset(pdir, "dup.yaml",
                                   {"arm": "A0", "changes_from_replicate": ["imgsz"]},
                                   dict(good, imgsz=800))),
          "duplicate arm ids are refused")
    check(refused(write_preset(pdir, "seg.yaml", {"arm": "S"},
                               dict(good, model="yolo11m-seg.pt"))),
          "segmentation weights are refused in a detector study")
    check(raises(lambda: cd.resolve_arms([]), ValueError), "an empty design is refused")
    check(raises(lambda: cd.load_eval_protocol(write_preset(pdir, "ev.yaml", {},
                                                            {"split": "test"})), ValueError),
          "an Evaluate preset missing a scoring key is refused")

    section("one arm runs through VEGA like a panel press")
    project = tmp / "p1"
    project.mkdir()
    fake = FakeVega(tmp)
    driver = make_driver(fake, project)
    row = driver.run_arm(arms[0])
    check(row["status"] == "ok" and row["scores"]["map50"] == 0.8, "the arm is scored")
    run_name = row["run_name"]
    check(run_name.startswith("TRAIN_coral_intel_t2aug_A0_replicate_"), "a unique run name")
    check(fake.configs["6"]["run_name"] == run_name, "the run name is saved before Start")
    check(fake.configs["6"]["model"] == "yolov8x.pt" and fake.configs["6"]["seed"] == 32,
          "the full config is saved for step 6")
    check(all(fake.configs["6"][k] == v for k, v in arms[0]["config"].items()),
          "every key of the replicate preset reaches VEGA unchanged")
    seq = [(m, p) for m, p, _ in fake.calls]
    i_open = seq.index(("POST", "/api/project/open"))
    i_put = seq.index(("PUT", "/api/project/config/6"))
    i_run = seq.index(("POST", "/api/step/6/run"))
    check(i_open < i_put < i_run, "open, then save settings, then Start")
    check(fake.configs["7"]["run_dir"].endswith(run_name) and fake.configs["7"]["split"] == "test"
          and fake.configs["7"]["iou_threshold"] == 0.6 and fake.configs["7"]["imgsz"] == 640,
          "evaluation targets this run on the test split at the arm's image size")
    state = json.loads((project / cd.STUDY_DIRNAME / "state.json").read_text())
    check(state["A0_replicate"] == {"run_name": run_name, "trained": True},
          "the arm is recorded as trained only after VEGA said so")
    check("train_done" in events(project) and "eval_done" in events(project),
          "progress log has train_done and eval_done")

    section("another person's open project is handed back")
    project_g = tmp / "p_giveback"
    project_g.mkdir()
    fake_g = FakeVega(tmp)
    fake_g.open_dir = "/somewhere/run_26-09-18_AT"
    row_g = make_driver(fake_g, project_g).run_arm(arms[0])
    check(row_g["status"] == "ok", "the arm still trains and scores")
    check(fake_g.open_dir == "/somewhere/run_26-09-18_AT",
          "VEGA ends with the other person's project open")
    check(fake_g.opened.count("/somewhere/run_26-09-18_AT") >= 2,
          "it is reopened after Start and again after the evaluation")

    section("a finish seen while another project is open is still recorded (2026-09-27)")
    project_q = tmp / "p_record"
    project_q.mkdir()
    fake_q = FakeVega(tmp, train_polls=3)
    fake_q.open_dir = "/somewhere/run_26-09-18_AT"
    row_q = make_driver(fake_q, project_q).run_arm(arms[0])
    check(row_q["status"] == "ok" and fake_q.step7_unlocked.get(str(project_q)),
          "Evaluate is unlocked and the replicate is scored")
    check(fake_q.open_dir == "/somewhere/run_26-09-18_AT", "and the other project is handed back")

    section("an arm refused at Evaluate is scored once the next finish unlocks it")
    project_u = tmp / "p_unscored"
    project_u.mkdir()
    fake_u = FakeVega(tmp)
    fake_u.open_dir = "/somewhere/run_26-09-18_AT"
    driver_u = make_driver(fake_u, project_u)
    real_record = driver_u.record_finish
    driver_u.record_finish = lambda status: status   # the 2026-09-26 behaviour
    rows_u = cd.run_all(driver_u, arms[:1], [], driver_u.progress, project_u)
    check(rows_u[0]["status"] == "error" and not fake_u.step7_unlocked.get(str(project_u)),
          "reproduced: the replicate's finish went unrecorded and Evaluate refused it")
    driver_u.record_finish = real_record
    rows_u2 = cd.run_all(driver_u, arms[:2], [], driver_u.progress, project_u)
    check([r["status"] for r in rows_u2] == ["ok", "ok"],
          "rerun with the fix: the next finish is recorded, and both models are scored")
    check("eval_retry" in events(project_u), "the replicate's retry is logged")
    check(fake_u.starts[6] == 2, "the replicate was not retrained, only the next arm trained")

    section("resuming does not retrain or re-score a finished arm")
    starts_before = dict(fake.starts)
    row2 = driver.run_arm(arms[0])
    check(fake.starts == starts_before and row2["run_name"] == run_name,
          "no Start pressed for an arm already trained and scored")
    (project / "step7_evaluateModel" / f"eval_{run_name}" / "metrics.json").unlink()
    row3 = driver.run_arm(arms[0])
    check(fake.starts[6] == starts_before[6] and fake.starts[7] == starts_before[7] + 1
          and row3["status"] == "ok", "a trained arm missing its evaluation is only re-scored")

    section("a driver restart mid-training waits on the live trainer")
    project_r = tmp / "p_resume"
    project_r.mkdir()
    fake_r = FakeVega(tmp, train_polls=3)
    driver_r = make_driver(fake_r, project_r)
    driver_r.remember("A0_replicate", "TRAIN_live", trained=False)
    fake_r.open_dir = str(project_r)
    fake_r.configs["6"]["run_name"] = "TRAIN_live"
    fake_r.proc[6] = {"polls": 0}
    fake_r.step6_running[str(project_r)] = True
    (project_r / "step6_trainModel" / "runs" / "TRAIN_live").mkdir(parents=True)
    fake_r.open_dir = "/somewhere/run_26-09-18_AT"
    fake_r.proc[6]["owner"] = str(project_r)
    row_r = driver_r.run_arm(arms[0])
    check(fake_r.starts[6] == 0 and row_r["run_name"] == "TRAIN_live" and row_r["status"] == "ok",
          "no second trainer is launched; the live one is awaited and scored")
    check(fake_r.open_dir == "/somewhere/run_26-09-18_AT",
          "resuming hands the other person's project back")

    section("VEGA busy with a run that is not this arm's is not adopted")
    project_o = tmp / "p_other"
    project_o.mkdir()
    fake_o = FakeVega(tmp, train_polls=2)
    driver_o = make_driver(fake_o, project_o)
    driver_o.remember("A0_replicate", "TRAIN_never_started", trained=False)
    fake_o.proc[6] = {"polls": 0, "owner": "/somewhere/else"}
    row_o = driver_o.run_arm(arms[0])
    check(fake_o.starts[6] == 1 and row_o["run_name"] != "TRAIN_never_started",
          "with no run folder of its own, the arm is launched afresh")

    section("a half-trained arm with best.pt but no VEGA verdict is retrained")
    project_h = tmp / "p_half"
    (project_h / "step6_trainModel" / "runs" / "TRAIN_dead" / "weights").mkdir(parents=True)
    (project_h / "step6_trainModel" / "runs" / "TRAIN_dead" / "weights" / "best.pt").write_bytes(b"x")
    fake_h = FakeVega(tmp)
    driver_h = make_driver(fake_h, project_h)
    driver_h.remember("A0_replicate", "TRAIN_dead", trained=False)
    row_h = driver_h.run_arm(arms[0])
    check(fake_h.starts[6] == 1 and row_h["run_name"] != "TRAIN_dead",
          "an early best.pt is not mistaken for a finished run")

    section("an arm that finished while the driver was down is kept, not retrained")
    project_k = tmp / "p_kept"
    (project_k / "step6_trainModel" / "runs" / "TRAIN_done" / "weights").mkdir(parents=True)
    (project_k / "step6_trainModel" / "runs" / "TRAIN_done" / "weights" / "best.pt").write_bytes(b"w")
    fake_k = FakeVega(tmp)
    driver_k = make_driver(fake_k, project_k, finished=lambda run_dir: run_dir.name == "TRAIN_done")
    driver_k.remember("A0_replicate", "TRAIN_done", trained=False)
    row_k = driver_k.run_arm(arms[0])
    check(fake_k.step7_unlocked.get(str(project_k)),
          "the kept run is recorded as finished, which unlocks Evaluate in a new project")
    check(fake_k.starts[6] == 0 and row_k["run_name"] == "TRAIN_done" and row_k["status"] == "ok",
          "no retraining; the finished run is evaluated")
    check(driver_k.state()["A0_replicate"]["trained"] is True, "and recorded as trained")

    section("a VEGA restart mid-training is survived")
    project_s = tmp / "p_restart"
    project_s.mkdir()
    fake_s = FakeVega(tmp, train_polls=4, restart_once=True)
    row_s = make_driver(fake_s, project_s).run_arm(arms[0])
    check(row_s["status"] == "ok" and fake_s.starts[6] == 1,
          "a poll with no exit code is re-asked with the project open and the survivor awaited")

    section("a 409 that is not busy is refused, not retried forever")
    project_n = tmp / "p_refuse"
    project_n.mkdir()
    fake_n = FakeVega(tmp, refuse_409=True)
    check(raises(lambda: make_driver(fake_n, project_n).run_arm(arms[0]), cd.VegaError)
          and fake_n.starts[6] == 1, "a non-busy 409 raises after one press")

    section("an evaluation whose only failure was the PDF is still scored")
    project_p = tmp / "p_pdf"
    project_p.mkdir()
    row_p = make_driver(FakeVega(tmp, eval_exit=2, metrics_on_exit2=True), project_p).run_arm(arms[0])
    check(row_p["status"] == "ok" and "eval_pdf_missing" in events(project_p),
          "exit 2 with metrics.json is scored and the missing PDF is logged")

    section("busy VEGA and the GPU ladder are waited out")
    project_b = tmp / "p_busy"
    project_b.mkdir()
    fake_b = FakeVega(tmp, busy_once=True)
    ladder = iter(["123 metashape --run", ""])
    driver_b = make_driver(fake_b, project_b, blocker=lambda: next(ladder, ""))
    row_b = driver_b.run_arm(arms[1])
    ev = events(project_b)
    check(row_b["status"] == "ok" and fake_b.starts[6] == 2, "a 409 is retried, then succeeds")
    check("waiting_for_gpu_ladder" in ev and "launch_busy" in ev,
          "both waits are logged")
    check(ev.index("waiting_for_gpu_ladder") < ev.index("train_launched"),
          "nothing launches while Metashape runs")
    check(fake_b.configs["6"]["model"] == "yolo11x.pt", "B1's model reaches VEGA")

    section("an open annotator holding a card is waited out, never swept")
    project_a = tmp / "p_annot"
    project_a.mkdir()
    fake_a = FakeVega(tmp)
    annot = iter(["an annotator on port 5080 (pid 42) holds a GPU",
                  "an annotator on port 5080 (pid 42) holds a GPU", ""])
    row_a = make_driver(fake_a, project_a, annotators=lambda: next(annot, "")).run_arm(arms[0])
    ev_a = events(project_a)
    check(row_a["status"] == "ok" and ev_a.count("waiting_for_gpu_ladder") >= 2
          and ev_a.index("waiting_for_gpu_ladder") < ev_a.index("train_launched"),
          "no Start is pressed until the annotator closes")

    section("the annotator check reads ss and nvidia-smi")
    SS = ("LISTEN 0 5 127.0.0.1:5080 0.0.0.0:* users:((\"python\",pid=42,fd=3))\n"
          "LISTEN 0 5 127.0.0.1:5081 0.0.0.0:* users:((\"python3\",pid=77,fd=3))\n"
          "LISTEN 0 5 127.0.0.1:5050 0.0.0.0:* users:((\"python\",pid=99,fd=3))\n")
    def fake_cmds(gpu_pids):
        def run(cmd, **k):
            return R(SS if cmd[0] == "ss" else "\n".join(gpu_pids) + "\n")
        return run
    class R:
        def __init__(self, out): self.stdout = out
    check("5080" in cd.annotator_blocker(fake_cmds(["42", "7045"])),
          "an annotator port whose pid holds a GPU blocks")
    check(cd.annotator_blocker(fake_cmds(["7045"])) == "",
          "an annotator port holding no GPU does not block")
    check(cd.annotator_blocker(fake_cmds(["77"])) != "",
          "any listener in 5080 to 5089 holding a GPU blocks")
    check(cd.annotator_blocker(fake_cmds(["99"])) == "",
          "a GPU holder outside 5080 to 5089 (VEGA itself on 5050) does not block")
    def no_ss(cmd, **k):
        raise FileNotFoundError(cmd[0])
    check(raises(lambda: cd.annotator_blocker(no_ss), cd.VegaError),
          "a missing ss or nvidia-smi refuses rather than guessing")

    section("failures are reported and never scored")
    project_f = tmp / "p_fail"
    project_f.mkdir()
    row_f = make_driver(FakeVega(tmp, train_exit=1), project_f).run_arm(arms[0])
    check(row_f["status"] == "training failed" and "scores" not in row_f,
          "a trainer exit 1 is a failed arm with no score")
    project_e = tmp / "p_evalfail"
    project_e.mkdir()
    row_e = make_driver(FakeVega(tmp, eval_exit=2), project_e).run_arm(arms[0])
    check(row_e["status"] == "evaluation failed", "an evaluation exit 2 is reported")
    project_l = tmp / "p_lost"
    project_l.mkdir()
    check(raises(lambda: make_driver(FakeVega(tmp, lose_process=True), project_l).run_arm(arms[0]),
                 cd.VegaError), "a process VEGA lost, with no finished run on disk, raises")
    project_v = tmp / "p_recover"
    project_v.mkdir()
    fake_v = FakeVega(tmp, lose_process=True)
    driver_v = make_driver(fake_v, project_v, finished=lambda run_dir: True)
    def recover():
        run = project_v / "step6_trainModel" / "runs" / fake_v.configs["6"]["run_name"] / "weights"
        run.mkdir(parents=True, exist_ok=True)
        (run / "best.pt").write_bytes(b"w")
    orig_poll = fake_v.poll
    def poll(step):
        out = orig_poll(step)
        if step == 6 and out.get("untracked"):
            recover()
            fake_v.lose_process = False
        return out
    fake_v.poll = poll
    row_v = driver_v.run_arm(arms[0])
    check(row_v["status"] == "ok" and "train_recovered" in events(project_v),
          "a run that finished while VEGA restarted is recovered and scored")

    section("finished-run detection reads the optimizer state")
    rd = tmp / "rd"
    (rd / "weights").mkdir(parents=True)
    check(not cd.training_finished(rd, loader=lambda p: {}), "no last.pt, not finished")
    (rd / "weights" / "last.pt").write_bytes(b"x")
    check(cd.training_finished(rd, loader=lambda p: {"optimizer": None}),
          "optimizer stripped means finished")
    check(not cd.training_finished(rd, loader=lambda p: {"optimizer": {"state": 1}}),
          "optimizer present means cut off")
    def boom(p):
        raise RuntimeError("truncated")
    check(not cd.training_finished(rd, loader=boom), "an unreadable checkpoint is not finished")
    check(not cd.training_finished(rd, loader=lambda p: [1]), "a non-dict checkpoint is not finished")

    section("the client handles VEGA's answers")
    fake_x = FakeVega(tmp, hostile_body=True)
    client = cd.VegaClient(cd.VEGA_URL, opener=fake_x)
    code, doc = client.call("GET", "/nope")
    check(code == 404 and "not json" in doc["error"], "a non-JSON error body is wrapped, not raised")
    check(raises(lambda: client.require("GET", "/nope"), cd.VegaError), "require raises on 404")

    def refuse(request, timeout=None):
        raise urllib.error.URLError("connection refused")
    check(raises(lambda: cd.VegaClient(opener=refuse).status(6), cd.VegaError),
          "VEGA down raises VegaError with the URL")

    def wrong_open(request, timeout=None):
        return FakeResponse(200, {"state": {"project_dir": "/somewhere/else"}})
    check(raises(lambda: cd.VegaClient(opener=wrong_open).open_project(tmp), cd.VegaError),
          "opening the wrong project is caught")
    check(cd._decode(b"[1,2]") == {"value": [1, 2]} and cd._decode(b"") == {},
          "odd JSON bodies decode to dicts")
    project_c = cd.find_or_create_project(cd.VegaClient(opener=FakeVega(tmp)), "")
    check((project_c / "project.json").is_file(), "a fresh project is minted with reuse false")
    check(raises(lambda: cd.find_or_create_project(client, str(tmp / "nothing")), cd.VegaError),
          "resuming a folder with no project.json is refused")

    section("calls carry the backend key that passes a module lock")
    seen = []
    def capture(request, timeout=None):
        seen.append(dict(request.header_items()))
        return FakeResponse(200, {"running": False})
    cd.VegaClient(cd.VEGA_URL, opener=capture,
                  headers={"X-VEGA-Backend-Token": "k" * 64}).status(6)
    check(any(v == "k" * 64 for v in seen[0].values()),
          "the key header is sent on every request")
    import os as _os
    _os.environ["VEGA_BACKEND_TOKEN_PATH"] = str(tmp / "cfg" / "vega_backend_token")
    hdr = cd.backend_headers()
    check(list(hdr) == ["X-VEGA-Backend-Token"] and len(next(iter(hdr.values()))) == 64,
          "backend_headers creates and returns the key")
    _os.environ.pop("VEGA_BACKEND_TOKEN_PATH")

    section("a recovered finish is recorded with VEGA's own project code")
    real = tmp / "run_real"
    (real / "step6_trainModel").mkdir(parents=True)
    (real / "project.json").write_text(json.dumps({
        "id": "x", "name": "TRAIN_x", "project_dir": str(real), "current_step": 6,
        "project_type": "bbox_training",
        "steps": {"6": {"status": "running", "name": "Train Model",
                        "dir": str(real / "step6_trainModel"), "config": {}, "outputs": {}},
                  "7": {"status": "locked", "name": "Evaluate Model",
                        "dir": str(real / "step7_evaluateModel"), "config": {}, "outputs": {}}},
        "access_log": []}))
    cd.mark_training_complete(real)
    after = json.loads((real / "project.json").read_text())
    check(after["steps"]["6"]["status"] == "completed", "step 6 is recorded completed")
    check(after["steps"]["7"]["status"] != "locked", "and Evaluate is unlocked")

    section("the GPU ladder check")

    class R:
        def __init__(self, out): self.stdout = out
    check(cd.ladder_blocker(lambda *a, **k: R("")) == "", "nothing running, no blocker")
    check(cd.ladder_blocker(lambda *a, **k: R("99 pgrep -af metashape\n")) == "",
          "pgrep's own line is ignored")
    noise = ("5 /bin/bash -c pgrep -af metashape|run_phase1\n"
             "6 env/bin/python scripts/_ablation/coral_intel_vega_driver.py --allow_metashape_pid 7\n"
             "9 tail -f /mnt/logs/metashape_run.log\n")
    check(cd.ladder_blocker(lambda *a, **k: R(noise)) == "",
          "shells, log viewers and this driver's own flags never count as Metashape (2026-09-29)")
    check(cd.ladder_blocker(lambda *a, **k: R("11 /usr/bin/python3 /opt/3D/src/run_phase1.py --site MRS\n"))
          .startswith("11"), "a run_phase1.py reconstruction blocks")
    check(cd.ladder_blocker(lambda *a, **k: R("12 /home/b/metashape-pro/metashape.sh -r x.py\n"))
          .startswith("12"), "the metashape.sh launcher blocks")
    edit = "13 /opt/metashape-pro/metashape /mnt/rip/driver_staging/editbench/FLC_T6_3D/FLC_T6.psx\n"
    scripted = "14 /opt/metashape-pro/metashape -r /x/align.py /mnt/rip/driver_staging/editbench/a.psx\n"
    check(cd.ladder_blocker(lambda *a, **k: R(edit)).startswith("13"),
          "a manual-edit window blocks by default")
    check(cd.ladder_blocker(lambda *a, **k: R(edit), allow_manual_edit=True) == "",
          "with the allowance, an interactive manual-edit window does not block")
    check(cd.ladder_blocker(lambda *a, **k: R(scripted), allow_manual_edit=True).startswith("14"),
          "a scripted Metashape run blocks even with the allowance")
    check(cd.ladder_blocker(lambda *a, **k: R("7 /opt/metashape/metashape -r x\n")).startswith("7"),
          "a Metashape process blocks")
    two = "7 /opt/metashape/metashape edit.psx\n8 /opt/metashape/metashape -r run.py\n"
    check(cd.ladder_blocker(lambda *a, **k: R(two), allowed_pids=frozenset({7})).startswith("8"),
          "an allowed Metashape window is ignored, any other Metashape still blocks")
    check(cd.ladder_blocker(lambda *a, **k: R("7 /opt/metashape/metashape edit.psx\n"),
                            allowed_pids=frozenset({7})) == "",
          "with only the allowed window open, nothing blocks")
    def no_pgrep(*a, **k):
        raise FileNotFoundError("pgrep")
    check(raises(lambda: cd.ladder_blocker(no_pgrep), cd.VegaError),
          "a missing pgrep refuses rather than skipping the ladder check")

    section("scores and the share folder")
    check(raises(lambda: cd.read_scores(tmp / "missing.json"), ValueError),
          "a missing evaluation raises")
    bad = tmp / "bad.json"
    bad.write_text("{nope")
    check(raises(lambda: cd.read_scores(bad), ValueError), "a corrupt evaluation raises")
    printed = []
    results = [row, dict(row_f, id="B1_yolo11x")]
    share = cd.assemble_share(project, arms, results, PROTOCOL,
                              printer=lambda h, p: (printed.append(h), p.write_bytes(b"%PDF")))
    names = sorted(p.name for p in share.iterdir())
    check(names == sorted(["A0_replicate_evaluation.pdf", cd.SUMMARY_PDF_NAME, cd.METHODS_PDF_NAME]),
          "one PDF per scored arm plus the summary and the methods, nothing for the failed arm")
    page = cd.summary_html(arms, results, {"test": 1554, "valid": 1554, "train": 21000},
                           cd.ast_now(), PROTOCOL)
    check("0.938" in page and "0.800" in page and "1,554" in page, "summary carries paper and arm scores")
    check("Confidence\n0.001" in page and "IoU 0.6" in page, "summary quotes the Evaluate preset")
    check("Open question for Elvin Cordero" in page, "the split-order question for Cordero is in the report")
    check("training failed" in page, "summary names the failed arm")
    page2 = cd.summary_html(arms, results, {"test": 1552, "valid": 1552, "train": 21771},
                            cd.ast_now(), PROTOCOL, {"train": 7257, "valid": 1552, "test": 1552})
    check(page2.index("strictest version") < page2.index("Scores on the held-out test split"),
          "the split explanation comes before the scores")
    check("before any augmentation" in page2 and "10,361 photographs" in page2
          and "never trained on in any form" in page2,
          "it states the photo counts and that test photos were never augmented or trained on")
    check("very likely" not in page2 and "either order" in page2,
          "the paper's split order is described as unknown, not assumed")
    check("\u2014" not in page and "\u2013" not in page and "UVI" not in page,
          "no em or en dashes and no UVI in the summary")
    check("Lauren Olinger" in page, "the summary carries the author line")
    hostile = [dict(arms[0], label="<script>x</script>")]
    check("<script>x" not in cd.summary_html(hostile, [], None, cd.ast_now(), PROTOCOL),
          "labels are HTML-escaped")

    section("the methods write-up comes from the params that ran")
    aug = cd.load_preset(cd.AUGMENT_PARAMS_FILE)["params"]
    manifest = {"counts": {"source_images_per_split": {"train": 7257, "valid": 1552, "test": 1552},
                           "output_images_per_split": {"train": 21771, "valid": 1552, "test": 1552},
                           "boxes_per_split": {"train": 29746, "valid": 2132, "test": 2192}},
                "boxes_dropped_by_clipping": 23}
    mpage = cd.methods_html(arms, PROTOCOL, aug, manifest,
                            {"ultralytics": "8.4.38", "torch": "2.11.0"}, cd.ast_now())
    import re as _re
    mtext = _re.sub(r"<[^>]+>", "", mpage)
    for fragment in ("before any augmentation", "70/15/15", "7,257 training", "1,552 test",
                     "21,771 training images", "YOLOv8x at 640", "batch size 16", "seed 32",
                     "13 degrees", "at most 14 degrees", "Ultralytics 8.4.38",
                     "23 of 34,070 boxes", "confidence threshold of 0.001", "Dropout"):
        check(fragment in " ".join(mtext.split()), f"the methods state {fragment!r}")
    check("\u2014" not in mtext and "\u2013" not in mtext and "Sam" not in mtext
          and "UVI" not in mtext, "no dashes, no personal names beyond the author, no UVI")
    check(all(a["id"] in mtext for a in arms[1:]), "every challenger is in the methods table")
    printed_m = []
    pdf = cd.write_methods(project, arms, PROTOCOL,
                           printer=lambda h, p: (printed_m.append(h), p.write_bytes(b"%PDF")))
    check(pdf.name == cd.METHODS_PDF_NAME and pdf.parent.name == cd.SHARE_DIRNAME,
          "the methods PDF goes into the share folder under the study name")

    section("staged weights exist where VEGA resolves bare names")
    wdir = cd.REPO_DIR / "scripts" / "TCRMPtrain_oceankindCV"
    check((wdir / "yolov8x.pt").is_file() and (wdir / "yolo11x.pt").is_file(),
          "yolov8x.pt and yolo11x.pt are staged")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
    check(not tmp.exists(), "the temporary fixtures were removed")

print(f"\n{CHECKS - FAILURES}/{CHECKS} passed.")
sys.exit(1 if FAILURES else 0)

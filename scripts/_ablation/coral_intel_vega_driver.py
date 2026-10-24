#!/usr/bin/env python3
"""Drive VEGA headlessly through the Coral Intel replication and three challengers.

Module:  scripts/_ablation/coral_intel_vega_driver.py
Purpose: Train and evaluate four bounding-box detectors inside ONE VEGA training
         project by calling VEGA's own web API on port 5050, exactly as the Train
         Model and Evaluate Model panels do when a person presses Start. VEGA
         builds the split dataset, owns every training process, writes every
         run into the project's step6_trainModel/runs/ folder (which is what its
         model pickers scan), and prints its own evaluation PDF. This driver only
         fills in the panel settings, presses Start, waits, and at the end copies
         the four evaluation PDFs plus one comparison PDF into a share folder.

           A0_replicate     Coral Intel Table 2 (Cordero et al. 2026): YOLOv8x,
                            640 px, batch 16, seed 32, 300 epochs, trained on the
                            offline Table 2 variants in coral_intel_t2aug, with
                            Ultralytics' own default online augmentation.
           B1_yolo11x       A0 with the YOLO11x backbone.
           B2_imgsz1024     A0 at 1024 px.
           B3_offline_only  A0 with online augmentation switched off, the other
                            reading of Table 2.

         Every challenger changes exactly one factor from A0; resolve_arms
         refuses a design where that is not true.

Inputs:  the study's params files in config/study_presets/coral_intel_replication/
         (four Train recipes and one Evaluate protocol, VEGA preset format),
         the staged dataset supporting_data/bbox_datasets/coral_intel_t2aug
         (built by coral_intel_offline_aug.py, carrying split_pins.json), a live
         VEGA orchestrator on port 5050, the YOLOv8x and YOLO11x weights in
         scripts/TCRMPtrain_oceankindCV/.
Outputs: <project>/step6_trainModel/runs/<run>/      one VEGA training run per arm
         <project>/step7_evaluateModel/eval_<run>/   one VEGA evaluation per arm
         <project>/coral_intel_study/arms.json       the design as executed
         <project>/coral_intel_study/state.json      arm -> run name, for resuming
         <project>/coral_intel_study/progress.jsonl  one row per state change
         <project>/coral_intel_study/results.json    the scores
         <project>/coralintel_reproduce_compete/                   four evaluation PDFs and
                                                     coralintel_reproduce_compete_summary.pdf

Run:     github_repo/env/bin/python scripts/_ablation/coral_intel_vega_driver.py
         (the module python: the recovery path reads checkpoints with torch and
         the summary PDF prints through evaluate_run.py)

Resuming: rerun the same command with --project_dir. An arm whose run already has weights and an
evaluation is skipped; an arm whose run has weights but no evaluation is
evaluated; anything else is trained from the start.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

AST = timezone(timedelta(hours=-4))

REPO_DIR = Path(__file__).resolve().parents[2]
INPROCESS_DIR = REPO_DIR.parent / "inprocess"
EVALUATE_SRC = REPO_DIR / "scripts" / "TCRMPtrain_oceankindCV" / "src"
ORCHESTRATOR_DIR = REPO_DIR / "scripts" / "pipeline_orchestrator"


def _orchestrator_module(name: str) -> Any:
    """Import one of VEGA's own orchestrator modules (lock_status, project_manager)."""
    if str(ORCHESTRATOR_DIR) not in sys.path:
        sys.path.insert(0, str(ORCHESTRATOR_DIR))
    import importlib  # noqa: PLC0415
    return importlib.import_module(name)


def backend_headers() -> dict[str, str]:
    """The header that lets this driver's calls through a module lock.

    Lauren locks VEGA to keep students off the annotation tools while a
    study runs (2026-09-28); the lock gate lets a local call carrying the
    backend key through. The key is created on first use, owner-only.
    """
    lock_status = _orchestrator_module("lock_status")
    return {lock_status.BACKEND_HEADER: lock_status.backend_token(create=True)}


def mark_training_complete(project_dir: Path) -> None:
    """Record a finished training in project.json with VEGA's own code.

    Used only when VEGA was restarted during a training that then finished:
    VEGA no longer holds the process, so no status poll can record it, and a
    new project's Evaluate step would stay locked. project_manager.complete_step
    is what VEGA itself calls on a finish (marks step 6 completed, unlocks the
    next step, links outputs).
    """
    pm = _orchestrator_module("project_manager")
    state = pm.load_project(str(project_dir))
    if state["steps"]["6"].get("status") != "completed":
        pm.complete_step(state, 6)

VEGA_URL = "http://127.0.0.1:5050"
DATASET_NAME = "coral_intel_t2aug"
LEAKY_DATASET_NAME = "coral_intel_t2aug_leaky"
INITIALS = "LO"
STUDY_DIRNAME = "coral_intel_study"
SHARE_DIRNAME = "coralintel_reproduce_compete"
SUMMARY_PDF_NAME = "coralintel_reproduce_compete_summary.pdf"
METHODS_PDF_NAME = "coralintel_reproduce_compete_methods.pdf"
STUDY_TITLE = "Coral Intel reproduce and compete"

HTTP_TIMEOUT_SECONDS = 60
HTTP_CONFLICT = 409
POLL_SECONDS = 60
# A 409 at launch means VEGA already has a trainer running or the GPU right-of-way
# gate said no. Either clears on its own; the driver waits this long between tries.
BUSY_RETRY_SECONDS = 600
# The GPU ladder puts a Metashape reconstruction above every VEGA run. The driver
# checks for one before each launch and waits while it runs.
LADDER_PATTERN = "metashape|run_phase1"
LADDER_RETRY_SECONDS = 900
# Lauren's manual-edit workspace. An interactive Metashape window open on a
# project there (no -r script) may stay open beside a VEGA run when she allows
# it (--allow_manual_edit_windows, 2026-09-29); scripted runs still hold VEGA.
MANUAL_EDIT_ROOT = "/mnt/rip/driver_staging/editbench/"
# A person's annotator (the SAM annotators on 5080 to 5089) holding a card is
# swept by VICARIUS's GPU right of way the moment a VEGA step claims the cards.
# Lauren ruled on 2026-09-26 that the study waits for it to close instead.
ANNOTATOR_PORTS = range(5080, 5090)

# Every setting lives in a params file, never in this script: four Train Model
# recipes and one Evaluate Model protocol, in VEGA's preset format, kept in
# their own study folder (the folder's README says why not config/presets/).
CONFIG_DIR = REPO_DIR / "config"
STUDY_PRESET_DIR = CONFIG_DIR / "study_presets" / "coral_intel_replication"
PRESET_FILES = [STUDY_PRESET_DIR / f"coral_intel_{arm}.yaml" for arm in
                ("A0_replicate", "B1_yolo11x", "B2_imgsz1024", "B3_offline_only")]
EVAL_PRESET_FILE = STUDY_PRESET_DIR / "evaluate_test_split.yaml"
AUGMENT_PARAMS_FILE = CONFIG_DIR / "augment_presets" / "coral_intel_table2.yaml"
# The Evaluate keys the preset must carry, besides the image size the driver
# replaces with each run's own training size.
EVAL_KEYS = ("split", "imgsz", "batch", "conf_threshold", "iou_threshold", "preview_count")
# evaluate_run.py exits 2 when the scores were written and only the PDF print
# failed: the arm is scored and its missing PDF is reported.
EVAL_EXIT_PDF_FAILED = 2

# Published Coral Intel species model, held-out evaluation (paper Table 4, sec. 3.2).
PAPER = {"map50": 0.938, "map50_95": 0.851, "precision": 0.954, "recall": 0.890}
PAPER_WEAKEST = {"Diploria-clivosa": 0.637, "Porites-colonensis": 0.641,
                 "Oculina-varicosa": 0.687}


def ast_now() -> datetime:
    """Return the current time in Atlantic Standard Time."""
    return datetime.now(AST)


def stamp(when: datetime | None = None) -> str:
    """Return a human timestamp, ``YYYY-MM-DD HH:MM AST``."""
    return (when or ast_now()).strftime("%Y-%m-%d %H:%M AST")


def compact_stamp(when: datetime | None = None) -> str:
    """Return a filename-safe timestamp, ``YYYYMMDD-HHMMSS``."""
    return (when or ast_now()).strftime("%Y%m%d-%H%M%S")


def load_preset(path: Path) -> dict[str, Any]:
    """Read one preset file and return its _meta and params mappings.

    Raises:
        ValueError: when the file is missing, will not parse, or lacks a
            mapping _meta or params section.
    """
    import yaml  # noqa: PLC0415, kept local so the module imports without PyYAML
    path = Path(path)
    try:
        doc = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise ValueError(f"cannot read the preset {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"the preset {path} does not parse: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"the preset {path} holds {type(doc).__name__}, not a mapping")
    meta, params = doc.get("_meta"), doc.get("params")
    if not isinstance(meta, dict) or not isinstance(params, dict) or not params:
        raise ValueError(f"the preset {path} needs a _meta mapping and a non-empty "
                         "params mapping")
    return {"meta": meta, "params": params, "path": str(path)}


def resolve_arms(paths: list[Path] | None = None) -> list[dict[str, Any]]:
    """Load the arm presets and prove the design is controlled.

    The first file is the replicate and must declare no changes. Every other
    file must carry exactly the replicate's keys, and the keys whose values
    differ from the replicate's must be exactly the ones its
    _meta.changes_from_replicate declares.

    Parameters:
        paths: the preset files in run order; the study's four by default.
    Returns:
        a list of dicts with id, factor, label, why, changes, config and path.
    Raises:
        ValueError: on a missing or malformed file, a duplicate arm id, a key
            set that differs from the replicate's, a real difference the file
            does not declare, a declared change that is not real, a
            replicate that declares changes, or a model that is not a detector.
    """
    files = PRESET_FILES if paths is None else list(paths)
    if not files:
        raise ValueError("the design has no arms")
    loaded = [load_preset(p) for p in files]
    base = loaded[0]["params"]
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, preset in enumerate(loaded):
        meta, params = preset["meta"], preset["params"]
        arm_id = str(meta.get("arm") or "")
        if not arm_id or arm_id in seen:
            raise ValueError(f"{preset['path']}: _meta.arm is missing or repeats {arm_id!r}")
        seen.add(arm_id)
        if str(params.get("task")) != "detect" or str(params.get("model", "")).endswith("-seg.pt"):
            raise ValueError(f"{preset['path']}: this study trains box detectors only")
        if set(params) != set(base):
            raise ValueError(f"{preset['path']}: its keys differ from the replicate's by "
                             f"{sorted(set(params) ^ set(base))}")
        declared = set(meta.get("changes_from_replicate") or [])
        real = {k for k in params if params[k] != base[k]}
        if index == 0 and declared:
            raise ValueError(f"{preset['path']}: the replicate must declare no changes")
        if index > 0 and not real:
            raise ValueError(f"{preset['path']}: {arm_id} changes nothing from the replicate")
        if real != declared:
            raise ValueError(f"{preset['path']}: {arm_id} declares changes {sorted(declared)} "
                             f"but really changes {sorted(real)}; the arm would be confounded")
        resolved.append({
            "id": arm_id,
            "factor": str(meta.get("factor", "")),
            "label": str(meta.get("name", arm_id)),
            "why": str(meta.get("why") or meta.get("description", "")),
            "changes": {k: params[k] for k in sorted(real)},
            "config": dict(params),
            "base": dict(base),
            "path": preset["path"],
        })
    return resolved


def load_eval_protocol(path: Path = EVAL_PRESET_FILE) -> dict[str, Any]:
    """Read the Evaluate preset and check it carries every scoring key."""
    params = load_preset(path)["params"]
    missing = [k for k in EVAL_KEYS if k not in params]
    if missing:
        raise ValueError(f"the evaluation preset {path} lacks {missing}")
    return dict(params)


def describe_change(arm: Mapping[str, Any]) -> str:
    """Return one plain phrase naming what an arm changed from the replicate."""
    if not arm["changes"]:
        return "none (the replicate)"
    return ", ".join(f"{k} {arm['base'][k]} to {v}" for k, v in arm["changes"].items())


class VegaError(RuntimeError):
    """Raised when VEGA answers with an error the driver cannot wait out."""


class VegaClient:
    """A thin JSON client for the VEGA orchestrator's web API."""

    def __init__(self, base_url: str = VEGA_URL,
                 opener: Callable[..., Any] = urllib.request.urlopen,
                 headers: Mapping[str, str] | None = None) -> None:
        """Remember the base URL, the opener (a fake in tests) and extra headers
        (the backend key that passes a module lock)."""
        self.base_url = base_url.rstrip("/")
        self._open = opener
        self.headers = dict(headers or {})

    def call(self, method: str, path: str, body: Mapping[str, Any] | None = None
             ) -> tuple[int, dict[str, Any]]:
        """Send one request and return (HTTP status, decoded JSON body).

        A non-2xx answer is returned, not raised, so callers can tell a busy
        409 from a real failure. A connection failure raises VegaError.
        """
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method,
            headers={"Content-Type": "application/json", **self.headers})
        try:
            with self._open(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                return response.status, _decode(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, _decode(exc.read())
        except (urllib.error.URLError, OSError) as exc:
            raise VegaError(f"VEGA at {self.base_url} did not answer {method} {path}: "
                            f"{exc}") from exc

    def require(self, method: str, path: str, body: Mapping[str, Any] | None = None
                ) -> dict[str, Any]:
        """Send one request and raise VegaError unless it answered 2xx."""
        status, doc = self.call(method, path, body)
        if not 200 <= status < 300:
            raise VegaError(f"VEGA refused {method} {path} with {status}: "
                            f"{doc.get('error') or doc}")
        return doc

    def create_project(self, dataset: str, initials: str) -> dict[str, Any]:
        """Mint a new training project for a staged dataset (never reuse one)."""
        return self.require("POST", "/boxes/api/create-project",
                            {"dataset": dataset, "by": initials, "reuse": False})

    def open_project(self, project_dir: Path) -> None:
        """Make ``project_dir`` VEGA's open project and prove it took."""
        doc = self.require("POST", "/api/project/open", {"project_dir": str(project_dir)})
        opened = (doc.get("state") or {}).get("project_dir") or ""
        if opened and Path(opened).resolve() != Path(project_dir).resolve():
            raise VegaError(f"VEGA opened {opened} when asked for {project_dir}")

    def open_project_dir(self) -> str:
        """Return the folder of the project VEGA has open, '' when none is."""
        doc = self.require("GET", "/api/project/state")
        return str((doc.get("state") or {}).get("project_dir") or "") if doc.get("loaded") else ""

    def put_config(self, step: int, config: Mapping[str, Any]) -> None:
        """Save a step's panel settings into the open project."""
        self.require("PUT", f"/api/project/config/{step}", dict(config))

    def start(self, step: int) -> tuple[int, dict[str, Any]]:
        """Press Start on a step; returns VEGA's status code and answer."""
        return self.call("POST", f"/api/step/{step}/run", {})

    def status(self, step: int) -> dict[str, Any]:
        """Poll a step's process; this poll is also what records completion."""
        return self.require("GET", f"/api/step/{step}/status")


def _decode(raw: bytes) -> dict[str, Any]:
    """Decode a JSON body, wrapping a non-JSON answer so it can be reported."""
    try:
        doc = json.loads(raw.decode() or "{}")
    except (UnicodeDecodeError, ValueError):
        return {"error": raw[:300].decode(errors="replace")}
    return doc if isinstance(doc, dict) else {"value": doc}


class Progress:
    """Append-only progress log, one JSON object per line, flushed each write."""

    def __init__(self, path: Path) -> None:
        """Open the log at ``path``, creating its parent directory."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> None:
        """Append one event row and echo it to stdout."""
        row = {"ts": ast_now().isoformat(), "event": event, **fields}
        try:
            with self.path.open("a") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
        except OSError as exc:
            raise OSError(f"cannot append to the progress log {self.path}: {exc}") from exc
        print(f"[{stamp()}] {event} {json.dumps(fields, default=str)}", flush=True)


def ladder_blocker(runner: Callable[..., Any] = subprocess.run,
                   allowed_pids: frozenset[int] = frozenset(),
                   allow_manual_edit: bool = False) -> str:
    """Return the command line of a running Metashape job, or '' when none runs.

    A Metashape reconstruction outranks every VEGA run on the GPU ladder, so no
    arm is launched while one is alive. ``allowed_pids`` are Metashape
    processes Lauren has explicitly allowed to stay open beside a run (an idle
    manual-edit window, 2026-09-29); every other one still holds the run.
    """
    try:
        result = runner(["pgrep", "-af", LADDER_PATTERN], capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise VegaError("pgrep is missing, so the GPU ladder cannot be checked; "
                        "no arm is launched without that check") from exc
    for line in (result.stdout or "").splitlines():
        if is_ladder_process(line, allowed_pids, allow_manual_edit):
            return line
    return ""


def is_ladder_process(line: str, allowed_pids: frozenset[int] = frozenset(),
                      allow_manual_edit: bool = False) -> bool:
    """Whether one ``pgrep -af`` line is a real Metashape or VOYAGER run.

    Only the program itself counts: a Metashape executable, or a
    run_phase1.py script. A shell, a log viewer or this driver whose command
    line merely contains the word (such as --allow_metashape_pid) does not;
    on 2026-09-29 the driver held itself for half an hour that way.
    """
    parts = line.split()
    if len(parts) < 2 or not parts[0].isdigit():
        return False
    if int(parts[0]) in allowed_pids:
        return False
    program = os.path.basename(parts[1]).lower()
    if program.startswith("metashape"):
        interactive_edit = ("-r" not in parts[2:]
                            and any(tok.startswith(MANUAL_EDIT_ROOT) for tok in parts[2:]))
        return not (allow_manual_edit and interactive_edit)
    return any(os.path.basename(tok) == "run_phase1.py" for tok in parts[2:])


def annotator_blocker(runner: Callable[..., Any] = subprocess.run) -> str:
    """Return a description of an open annotator holding a GPU, '' when none.

    An annotator is a process listening on a port in ANNOTATOR_PORTS whose
    pid also appears in nvidia-smi's compute list: exactly the human-tier
    holder VICARIUS's right-of-way sweep would stop. A listener holding no
    card (a plain editor server) does not block, and neither does Vesper.llm
    on 5085, whose card memory belongs to the Ollama service, not that pid.

    Raises:
        VegaError: when ss or nvidia-smi cannot be run, because launching
            without knowing would risk stopping someone's session.
    """
    try:
        listening = runner(["ss", "-ltnpH"], capture_output=True, text=True).stdout or ""
        gpu = runner(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout or ""
    except FileNotFoundError as exc:
        raise VegaError(f"cannot check for an open annotator ({exc}); no arm is launched "
                        "without that check") from exc
    gpu_pids = {ln.strip() for ln in gpu.splitlines() if ln.strip().isdigit()}
    for line in listening.splitlines():
        parts = line.split()
        port = parts[3].rsplit(":", 1)[-1] if len(parts) > 3 else ""
        if not port.isdigit() or int(port) not in ANNOTATOR_PORTS:
            continue
        for pid in re.findall(r"pid=(\d+)", line):
            if pid in gpu_pids:
                return f"an annotator on port {port} (pid {pid}) holds a GPU"
    return ""


def read_json(path: Path, default: Any = None) -> Any:
    """Read a JSON file, returning ``default`` when it does not exist."""
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def write_json(path: Path, doc: Any) -> None:
    """Write ``doc`` as indented JSON atomically, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
        tmp.replace(path)
    except OSError as exc:
        raise OSError(f"cannot write {path}: {exc}") from exc


def read_scores(metrics_path: Path) -> dict[str, Any]:
    """Read headline and per-class box scores out of a VEGA evaluation.

    Returns a dict with map50, map50_95, precision, recall, per_class (name to
    mAP50) and instances (name to test instance count).
    """
    doc = read_json(metrics_path)
    if doc is None:
        raise ValueError(f"the evaluation {metrics_path} does not exist")
    metrics = doc.get("metrics") or {}
    overall = metrics.get("overall") or {}
    per_class = {name: (row or {}).get("box_map50")
                 for name, row in (metrics.get("per_class") or {}).items()}
    context = doc.get("context") or {}
    return {
        "map50": overall.get("box_map50"),
        "map50_95": overall.get("box_map50_95"),
        "precision": overall.get("box_precision"),
        "recall": overall.get("box_recall"),
        "per_class": per_class,
        "instances": metrics.get("per_class_instances") or {},
        "epochs_completed": context.get("epochs_completed"),
        "best_epoch": context.get("best_epoch"),
        "split_counts": context.get("split_counts"),
    }


def training_finished(run_dir: Path,
                      loader: Callable[[Path], Any] | None = None) -> bool:
    """Whether Ultralytics finished a run, read from the run folder alone.

    Ultralytics strips the optimizer state out of weights/last.pt only when
    training ends (the schedule completed or patience stopped it), so a
    last.pt whose optimizer is None is a finished run and one still carrying
    it is a run that was cut off. Used only when VEGA lost the process.

    Parameters:
        run_dir: the Ultralytics run folder.
        loader: reads a checkpoint into a dict; torch.load by default.
    Returns:
        True when last.pt exists and carries no optimizer state.
    """
    last = Path(run_dir) / "weights" / "last.pt"
    if not last.is_file():
        return False
    if loader is None:
        import torch  # noqa: PLC0415, only needed on this recovery path
        def loader(path: Path) -> Any:
            return torch.load(str(path), map_location="cpu", weights_only=False)
    try:
        checkpoint = loader(last)
    except Exception as exc:  # a truncated or foreign file is not a finished run
        print(f"cannot read {last}: {exc}", file=sys.stderr)
        return False
    return isinstance(checkpoint, dict) and checkpoint.get("optimizer") is None


class Driver:
    """Runs the arms one after another through a VegaClient."""

    def __init__(self, client: VegaClient, project_dir: Path, progress: Progress,
                 protocol: Mapping[str, Any],
                 sleep: Callable[[float], None] = time.sleep,
                 blocker: Callable[[], str] = ladder_blocker,
                 annotators: Callable[[], str] = annotator_blocker,
                 finished: Callable[[Path], bool] = training_finished,
                 on_change: Callable[[], None] = lambda: None,
                 complete: Callable[[Path], None] = mark_training_complete) -> None:
        """Bind the client, the project, the log, the Evaluate preset's
        settings, and the waiting functions."""
        self.client = client
        self.protocol = dict(protocol)
        self.finished = finished
        self.on_change = on_change
        self.complete = complete
        self.previous = ""
        self.project_dir = Path(project_dir)
        self.progress = progress
        self.sleep = sleep
        self.blocker = blocker
        self.annotators = annotators
        self.study_dir = self.project_dir / STUDY_DIRNAME
        self.runs_dir = self.project_dir / "step6_trainModel" / "runs"
        self.eval_dir = self.project_dir / "step7_evaluateModel"

    def state(self) -> dict[str, Any]:
        """Return the per-arm record ({arm: {run_name, trained}}), empty when fresh."""
        return read_json(self.study_dir / "state.json", {}) or {}

    def remember(self, arm_id: str, run_name: str, trained: bool) -> None:
        """Record an arm's run folder and whether VEGA reported its training done.

        The weights file alone cannot say that: Ultralytics writes best.pt after
        the first epoch, so a half-trained run already has one.
        """
        doc = self.state()
        doc[arm_id] = {"run_name": run_name, "trained": trained}
        write_json(self.study_dir / "state.json", doc)

    def wait_for_ladder(self) -> None:
        """Hold while a Metashape reconstruction runs or an annotator holds a card.

        Metashape outranks VEGA on the GPU ladder. An open annotator ranks below
        VEGA and VICARIUS would stop it at launch, so the study waits for the
        person to close it instead (Lauren, 2026-09-26).
        """
        while True:
            blocker = self.blocker() or self.annotators()
            if not blocker:
                return
            self.progress.write("waiting_for_gpu_ladder", running=blocker[:200])
            self.sleep(LADDER_RETRY_SECONDS)

    def give_back(self, previous: str) -> None:
        """Reopen the project someone else had open before the driver took over.

        VEGA holds one open project for everyone, so pressing Start here moves
        every browser on VEGA to this study. Handing the previous project back
        straight after keeps a person's session where they left it.
        """
        if previous and Path(previous).resolve() != self.project_dir.resolve():
            try:
                self.client.open_project(Path(previous))
                self.progress.write("gave_back_project", project_dir=previous)
            except VegaError as exc:
                self.progress.write("give_back_failed", project_dir=previous, error=str(exc))

    def launch(self, step: int, config: Mapping[str, Any]) -> dict[str, Any]:
        """Open the project, save the step's settings, press Start, hand back.

        A busy 409 (a trainer already running, or the GPU gate refusing) is
        waited out; any other refusal raises VegaError. For training, the
        project that was open before is reopened once Start is pressed; for an
        evaluation that happens after it finishes, in run_arm.
        """
        while True:
            self.wait_for_ladder()
            self.previous = self.client.open_project_dir()
            self.client.open_project(self.project_dir)
            self.client.put_config(step, config)
            code, doc = self.client.start(step)
            if 200 <= code < 300:
                if step == 6:
                    self.give_back(self.previous)
                return doc
            self.give_back(self.previous)
            if code != HTTP_CONFLICT or not doc.get("busy"):
                # Only a busy refusal (a live trainer, or the GPU gate) clears
                # by waiting; any other 409, such as a project that already
                # trains from another source, never will.
                raise VegaError(f"VEGA refused to start step {step} with {code}: "
                                f"{doc.get('error') or doc}")
            self.progress.write("launch_busy", step=step, reason=str(doc.get("error"))[:300])
            self.sleep(BUSY_RETRY_SECONDS)

    def wait(self, step: int) -> dict[str, Any]:
        """Poll a step until its process ends and return the final status.

        Training polls do not reopen the project, so a person using VEGA in a
        browser is not pulled off whatever project they have open; VEGA records
        a finished training against the project that launched it either way.
        An evaluation lasts minutes and VEGA credits it only to the open
        project, so for step 7 the project is reopened before each poll.

        A poll with no exit code means the orchestrator no longer holds the
        process (it was restarted). VEGA only looks for a trainer that survived
        a restart while a project is open, so the project is opened and the
        step polled again: a surviving trainer answers running (with its pid)
        and is waited on. Raises VegaError when it still answers no exit code,
        because whether that process finished cannot be read from VEGA.
        """
        while True:
            self.sleep(POLL_SECONDS)
            if step == 7:
                self.client.open_project(self.project_dir)
            status = self.client.status(step)
            if status.get("exit_code") is None and not status.get("running"):
                self.client.open_project(self.project_dir)
                status = self.client.status(step)
            if status.get("running"):
                continue
            if status.get("exit_code") is None:
                raise VegaError(f"VEGA no longer holds the step {step} process "
                                "(the orchestrator was restarted under it)")
            if step == 6:
                return self.record_finish(status)
            return status

    def record_finish(self, status: dict[str, Any]) -> dict[str, Any]:
        """Poll once more with this project open, so VEGA records the finish.

        VEGA writes a finished training into a project (step 6 completed,
        Evaluate unlocked) only on a status poll made while that project is
        the open one; a poll made while another project is open records
        nothing (found 2026-09-27, when the replicate's Evaluate stayed
        locked). The runner keeps the finished process's exit code, so a
        second poll here still sees it. The previous project is handed back.
        """
        previous = self.client.open_project_dir()
        self.client.open_project(self.project_dir)
        try:
            again = self.client.status(6)
        finally:
            self.give_back(previous)
        return again if again.get("exit_code") is not None else status

    def run_arm(self, arm: Mapping[str, Any]) -> dict[str, Any]:
        """Train (unless VEGA already finished it) and evaluate one arm; return its row.

        On a rerun, an arm recorded as launched but not trained is resumed in
        one of three ways: VEGA still runs it (wait on it), it ran to its end
        while the driver was down (keep it), or it died (train it afresh
        under a new run name). A run kept from disk was never marked complete
        by VEGA, so it is recorded with VEGA's own project code
        (mark_training_complete) before it is scored; otherwise a new
        project's Evaluate step would stay locked.
        """
        record = self.state().get(arm["id"]) or {}
        run_name = record.get("run_name")
        trained = bool(record.get("trained") and run_name)
        resume_live = False
        started = time.monotonic()
        if run_name and not trained:
            previous = self.client.open_project_dir()
            self.client.open_project(self.project_dir)
            live = self.client.status(6).get("running")
            self.give_back(previous)
            if (live
                    and (self.runs_dir / run_name).is_dir()):
                # VEGA is training and this arm's run folder exists: it is ours.
                resume_live = True
                self.progress.write("train_resumed_wait", arm=arm["id"], run_name=run_name)
            elif self.finished(self.runs_dir / run_name):
                trained = True
                self.complete(self.project_dir)
                self.remember(arm["id"], run_name, trained=True)
                self.progress.write("train_recovered", arm=arm["id"], run_name=run_name)
        if not trained:
            if not resume_live:
                run_name = f"TRAIN_coral_intel_t2aug_{arm['id']}_{compact_stamp()}"
                self.remember(arm["id"], run_name, trained=False)
                self.progress.write("train_start", arm=arm["id"], run_name=run_name,
                                    changes=arm["changes"])
                launched = self.launch(6, {**arm["config"], "run_name": run_name})
                self.progress.write("train_launched", arm=arm["id"], pid=launched.get("pid"))
                self.on_change()
            try:
                status = self.wait(6)
            except VegaError:
                if not self.finished(self.runs_dir / run_name):
                    raise
                # VEGA was restarted under a trainer that then ran to its end.
                self.complete(self.project_dir)
                self.progress.write("train_recovered", arm=arm["id"], run_name=run_name)
                status = {"exit_code": 0}
            weights = self.runs_dir / run_name / "weights" / "best.pt"
            if status.get("exit_code") != 0 or not weights.is_file():
                self.progress.write("train_failed", arm=arm["id"],
                                    exit_code=status.get("exit_code"))
                return {"id": arm["id"], "status": "training failed", "run_name": run_name,
                        "detail": f"VEGA's trainer exited {status.get('exit_code')}"}
            self.remember(arm["id"], run_name, trained=True)
            self.progress.write("train_done", arm=arm["id"],
                                hours=round((time.monotonic() - started) / 3600, 2))
        metrics_path = self.eval_dir / f"eval_{run_name}" / "metrics.json"
        if not metrics_path.is_file():
            self.progress.write("eval_start", arm=arm["id"], run_name=run_name)
            self.launch(7, {**self.protocol, "run_dir": str(self.runs_dir / run_name),
                            "imgsz": arm["config"]["imgsz"]})
            try:
                status = self.wait(7)
            finally:
                self.give_back(self.previous)
            code = status.get("exit_code")
            if not metrics_path.is_file() or code not in (0, EVAL_EXIT_PDF_FAILED):
                self.progress.write("eval_failed", arm=arm["id"], exit_code=code)
                return {"id": arm["id"], "status": "evaluation failed", "run_name": run_name,
                        "detail": f"VEGA's evaluation exited {code}"}
            if code == EVAL_EXIT_PDF_FAILED:
                self.progress.write("eval_pdf_missing", arm=arm["id"])
        scores = read_scores(metrics_path)
        self.progress.write("eval_done", arm=arm["id"],
                            map50=scores["map50"], map50_95=scores["map50_95"])
        self.on_change()
        return {"id": arm["id"], "status": "ok", "run_name": run_name, "scores": scores}


def run_all(driver: "Driver", arms: list[dict[str, Any]], wanted: list[str],
            progress: Progress, project_dir: Path) -> list[dict[str, Any]]:
    """Run every arm in order and return one result row per arm.

    Before each arm, and once more at the end, any earlier arm that trained
    but was not scored (an evaluation VEGA refused) is evaluated again, so a
    finished model is scored before the next one takes the cards.
    """
    rows: dict[str, dict[str, Any]] = {}
    order = [a for a in arms if not wanted or a["id"] in wanted]

    def attempt(arm: Mapping[str, Any]) -> None:
        try:
            rows[arm["id"]] = driver.run_arm(arm)
        except VegaError as exc:
            progress.write("arm_error", arm=arm["id"], error=str(exc))
            rows[arm["id"]] = {"id": arm["id"], "status": "error", "detail": str(exc)}
        write_json(project_dir / STUDY_DIRNAME / "results.json",
                   {"arms": [rows[a["id"]] for a in order if a["id"] in rows],
                    "updated": ast_now().isoformat()})

    def retry_unscored() -> None:
        for arm in order:
            row = rows.get(arm["id"])
            trained = (driver.state().get(arm["id"]) or {}).get("trained")
            if row and row.get("status") != "ok" and trained:
                progress.write("eval_retry", arm=arm["id"])
                attempt(arm)

    for arm in order:
        retry_unscored()
        attempt(arm)
    retry_unscored()
    return [rows[a["id"]] for a in order]


def share_pdf_name(arm: Mapping[str, Any]) -> str:
    """Return the file name an arm's evaluation PDF gets in the share folder."""
    return f"{arm['id']}_evaluation.pdf"


def fmt(value: Any, digits: int = 3) -> str:
    """Format a score, or say it is missing."""
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def summary_html(arms: list[Mapping[str, Any]], results: list[Mapping[str, Any]],
                 split_counts: Mapping[str, Any] | None, created: datetime,
                 protocol: Mapping[str, Any],
                 photo_counts: Mapping[str, Any] | None = None) -> str:
    """Render the comparison summary as a self-contained HTML page for printing.

    Parameters:
        arms: the resolved arms, for the design table.
        results: one row per arm from Driver.run_arm.
        split_counts: the test/valid/train image counts VEGA reported.
        created: when the summary was written.
        protocol: the Evaluate preset's settings, quoted in the text.
        photo_counts: source photographs per split, before augmentation, from
            the dataset's augment_manifest.json; None prints n/a.
    Returns:
        the HTML document as a string.
    """
    by_id = {r["id"]: r for r in results}
    esc = html.escape
    rows = [f"<tr><td>Coral Intel, published</td><td>{fmt(PAPER['map50'])}</td>"
            f"<td>{fmt(PAPER['map50_95'])}</td><td>{fmt(PAPER['precision'])}</td>"
            f"<td>{fmt(PAPER['recall'])}</td><td>n/a</td></tr>"]
    for arm in arms:
        row = by_id.get(arm["id"], {})
        s = row.get("scores") or {}
        epochs = (f"{s.get('epochs_completed')} (best {s.get('best_epoch')})"
                  if s.get("epochs_completed") else "n/a")
        rows.append(f"<tr><td>{esc(arm['id'])}: {esc(arm['label'])}</td>"
                    f"<td>{fmt(s.get('map50'))}</td><td>{fmt(s.get('map50_95'))}</td>"
                    f"<td>{fmt(s.get('precision'))}</td><td>{fmt(s.get('recall'))}</td>"
                    f"<td>{esc(epochs)}</td></tr>")
    design = "".join(
        f"<tr><td>{esc(a['id'])}</td><td>{esc(a['factor'])}</td>"
        f"<td>{esc(describe_change(a))}</td><td>{esc(a['why'])}</td></tr>" for a in arms)
    weak_rows = []
    for name, paper_value in PAPER_WEAKEST.items():
        cells = "".join(
            f"<td>{fmt(((by_id.get(a['id']) or {}).get('scores') or {}).get('per_class', {}).get(name))}</td>"
            for a in arms)
        weak_rows.append(f"<tr><td><i>{esc(name.replace('-', ' '))}</i></td>"
                         f"<td>{fmt(paper_value)}</td>{cells}</tr>")
    arm_heads = "".join(f"<th>{esc(a['id'])}</th>" for a in arms)
    failures = "".join(f"<li>{esc(r['id'])}: {esc(r.get('status', ''))}. "
                       f"{esc(r.get('detail', ''))}</li>"
                       for r in results if r.get("status") != "ok" and not r.get("pending"))
    counts = split_counts or {}
    def num(value: Any) -> str:
        """An integer with thousands separators, else the value as text."""
        return f"{value:,}" if isinstance(value, int) else "n/a"
    raw_photos = photo_counts or {}
    photos = {k: num(v) for k, v in raw_photos.items()}
    counts = {k: num(v) for k, v in counts.items()}
    n_photos = num(sum(v for v in raw_photos.values() if isinstance(v, int)) or None)
    pdfs = "".join(f"<li>{esc(share_pdf_name(a))}: VEGA's full evaluation of {esc(a['id'])}"
                   "</li>" for a in arms)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{esc(STUDY_TITLE)}</title><style>
body{{font:10.5pt/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1d2530;margin:28px}}
h1{{font-size:17pt;margin:0 0 4px}} h2{{font-size:12.5pt;margin:20px 0 6px}}
table{{border-collapse:collapse;width:100%;margin:6px 0}}
th,td{{border:1px solid #c9d1db;padding:4px 6px;text-align:left;vertical-align:top}}
th{{background:#eef2f6}} .meta{{color:#5a6573;font-size:9pt}}
</style></head><body>
<h1>{esc(STUDY_TITLE)}: results summary</h1>
<div class="meta">VICARIUS, VEGA project {esc(DATASET_NAME)}. Written {esc(stamp(created))}. Lauren Olinger</div>

<h2>How the photos were split, and why this is the strictest version</h2>
<p>All {n_photos} photographs were divided <b>before any augmentation</b>:
{photos.get('train', 'n/a')} for training, {photos.get('valid', 'n/a')} for validation and
{photos.get('test', 'n/a')} for testing (70/15/15, balanced on each photograph's rarest species,
seed 32). Only the training photographs were then augmented, each appearing as the original
plus two Table 2 copies ({counts.get('train', 'n/a')} training images). Validation and test are
the original photographs, never augmented and never trained on in any form. Every model in this
folder is scored on exactly the same {photos.get('test', 'n/a')} test photographs.</p>
<p>The paper states a 70/15/15 split but not whether it split before or after augmenting. Its
evaluation set of 4,843 images is about 15 percent of the 32,423 augmented images, which fits
either order. If it augmented first and then split, copies of the same photograph can sit in
both training and test, and the model is partly tested on near-copies of photographs it has
learned; that raises scores. If it split first and then tripled every set, nothing leaks, but
the test set is three copies of each held-out photograph. The split here is at least as strict
as either: a score below measures photographs the model has never seen, in any version. Scores
are therefore expected to sit below the published ones for reasons that have nothing to do
with the model, and the published row below should be read with that in mind.</p>
<p><b>Open question for Elvin Cordero:</b> were the 32,423 images augmented before or after the
70/15/15 split, and can he share the train, validation and test file lists? If augmentation
came first, copies of test photographs were in training and the published scores are
inflated.</p>

<h2>Scores on the held-out test split</h2>
<p>Box detection on the {counts.get('test', 'n/a')} test photographs. Confidence
{protocol.get('conf_threshold')}, NMS IoU {protocol.get('iou_threshold')}, each model at its own
training image size.</p>
<table><tr><th>Model</th><th>mAP50</th><th>mAP50-95</th><th>Precision</th><th>Recall</th>
<th>Epochs</th></tr>{''.join(rows)}</table>
<p>The published row comes from the paper's own evaluation set, whose split order is not
stated (see above), so it is not a like-for-like comparison with the rows beneath it.
{'L0 repeats A0 on images split after augmenting and is scored on its own test images, most of which have a copy of the same photograph in training; it shows how far that overlap alone raises the score and is not comparable with A0 to B3.' if any(a['id'].startswith('L') for a in arms) else ''}</p>

<h2>Weakest species in the paper</h2>
<table><tr><th>Species</th><th>Paper</th>{arm_heads}</tr>{''.join(weak_rows)}</table>

<h2>Design: each challenger changes one factor from the replicate</h2>
<table><tr><th>Arm</th><th>Factor</th><th>Change</th><th>Reason</th></tr>{design}</table>

<h2>How the replicate follows Table 2</h2>
<ul>
<li>YOLOv8x from COCO weights, 300 epochs, 640 px, batch 16, seed 32. Patience 100 and every
other training setting at the Ultralytics default, since the paper reports none.</li>
<li>Offline augmentation: each training photograph appears as the original plus two variants,
each drawing flips, a 90 degree rotation, rotation up to 13 degrees, shear up to 10 degrees
on both axes, hue up to 15 degrees, saturation up to 25 percent, exposure up to 10 percent,
blur up to 1.5 px and noise on up to 0.1 percent of pixels. Validation and test are untouched
originals.</li>
<li>Split by photograph before augmenting, as described at the top.</li>
<li>Not carried: dropout 0.10 (in Ultralytics it acts on classification heads only, so it has
no effect on a detector), 8 data-loader workers (VEGA uses 12; this changes speed, not the
model), and the 12 images the paper has that our copy lacks. Trained on two RTX 5090 cards,
batch 16 in total.</li>
<li>Kept as the paper did: 1,278 whole-frame boxes, flagged at ingest as possible placeholders.
1,273 photos carry only such a box, so they count as one coral filling the frame. They are
easy to score, which raises mAP50-95 slightly for every model alike.</li>
<li>One seed per arm. A difference under about one mAP50 point is not distinguishable from
run-to-run variation.</li>
</ul>
{f'<h2>Arms that did not produce a score</h2><ul>{failures}</ul>' if failures else ''}
<h2>Files in this folder</h2><ul>{pdfs}<li>{esc(SUMMARY_PDF_NAME)}: this summary</li>
<li>{esc(METHODS_PDF_NAME)}: the methods, written as for a manuscript</li></ul>
<p class="meta">Paper: Cordero E, Sherman CE, Yadav P, Kumar Raju V, Townsend JE, Courtney TA
(2026). Coral Intel: a YOLOv8 deep learning framework for monitoring Caribbean corals.
Intelligent Marine Technology and Systems 4:18. doi:10.1007/s44295-026-00107-5. Annotations
CC BY 4.0; the photographs are third-party and are for local training only.</p>
<p class="meta">Lauren Olinger</p>
</body></html>"""


def print_pdf(html_path: Path, pdf_path: Path) -> None:
    """Print an HTML file to PDF with the same headless Chrome VEGA's reports use."""
    sys.path.insert(0, str(EVALUATE_SRC))
    import evaluate_run  # noqa: PLC0415, heavy import kept out of the test path
    evaluate_run.print_html_to_pdf_with_chrome(str(html_path), str(pdf_path))
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise OSError(f"Chrome wrote no PDF at {pdf_path}")


PAGE_STYLE = """body{font:10.5pt/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1d2530;margin:28px}
h1{font-size:17pt;margin:0 0 4px} h2{font-size:12.5pt;margin:20px 0 6px}
table{border-collapse:collapse;width:100%;margin:6px 0}
th,td{border:1px solid #c9d1db;padding:4px 6px;text-align:left;vertical-align:top}
th{background:#eef2f6} .meta{color:#5a6573;font-size:9pt}"""


def library_versions() -> dict[str, str]:
    """The Ultralytics and PyTorch versions this interpreter carries, 'unknown'
    when a package cannot be found (the methods then say so rather than guess)."""
    import importlib.metadata as md  # noqa: PLC0415
    versions = {}
    for package in ("ultralytics", "torch"):
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def dataset_manifest(dataset_dir: Path | None = None) -> dict[str, Any]:
    """The staged dataset's augment_manifest.json, {} when it cannot be read."""
    path = (dataset_dir or REPO_DIR / "supporting_data" / "bbox_datasets" / DATASET_NAME) \
        / "augment_manifest.json"
    try:
        return read_json(path) or {}
    except ValueError:
        return {}


def methods_html(arms: list[Mapping[str, Any]], protocol: Mapping[str, Any],
                 aug: Mapping[str, Any], manifest: Mapping[str, Any],
                 versions: Mapping[str, str], created: datetime) -> str:
    """Render a manuscript-style Methods section for the study.

    Every number comes from what ran: the augmentation params file, the
    replicate's Train recipe, the Evaluate preset and the staged dataset's
    manifest, so the text cannot drift from the runs.

    Parameters:
        arms: the resolved arms; arms[0] is the replicate.
        protocol: the Evaluate preset's settings.
        aug: the augmentation params file's params mapping.
        manifest: the staged dataset's augment_manifest.json.
        versions: library versions from library_versions().
        created: when the page was written.
    Returns:
        the HTML document as a string.
    """
    esc = html.escape
    base = arms[0]["config"]
    counts = manifest.get("counts") or {}
    photos = counts.get("source_images_per_split") or {}
    images = counts.get("output_images_per_split") or {}
    boxes = counts.get("boxes_per_split") or {}

    def n(value: Any) -> str:
        return f"{value:,}" if isinstance(value, int) else "n/a"

    n_photos = n(sum(v for v in photos.values() if isinstance(v, int)) or None)
    n_boxes = n(sum(v for v in boxes.values() if isinstance(v, int)) or None)
    challengers = "".join(
        f"<tr><td>{esc(a['id'])}</td><td>{esc(a['factor'])}</td>"
        f"<td>{esc(describe_change(a))}</td></tr>" for a in arms[1:] if not a["id"].startswith("L"))
    leaky_paragraph = ""
    if any(a["id"].startswith("L") for a in arms):
        lm = dataset_manifest(REPO_DIR / "supporting_data" / "bbox_datasets" / LEAKY_DATASET_NAME)
        lc = (lm.get("counts") or {}).get("output_images_per_split") or {}
        lt = ((lm.get("sibling_leak") or {}).get("test") or {})
        leaky_paragraph = f"""<h2>Test of the published split order</h2>
<p>To measure how much a split made after augmentation raises the scores, we trained a fifth
model (L0) with the A0 settings on a second dataset built from the same photographs in the other
order. L0 trained on a single RTX 5090 with the whole batch of {base['batch']} on it, as the paper
trained on one GPU; A0 split the batch across two cards. We wrote {aug['variants']} Table 2 copies of every photograph, including those
that would be held out, and then split the {n(sum(v for v in lc.values() if isinstance(v, int)) or None)}
resulting images one by one into training, validation and test (70/15/15, stratified on each
image's rarest species, seed {aug['seed']}), giving {n(lc.get('train'))}, {n(lc.get('valid'))}
and {n(lc.get('test'))} images. In that test set, {n(lt.get('with_copy_in_train'))} of
{n(lt.get('images'))} images had at least one copy of the same photograph in training. We
scored L0 on its own test set with the protocol below. Its score measures the effect of the
overlap and is not comparable with those of A0 to B3.</p>"""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{esc(STUDY_TITLE)}</title><style>{PAGE_STYLE}</style></head><body>
<h1>{esc(STUDY_TITLE)}: methods</h1>
<div class="meta">VICARIUS, VEGA project {esc(DATASET_NAME)}. Written {esc(stamp(created))}. Lauren Olinger</div>

<h2>Data</h2>
<p>We used the Coral Intel object-detection dataset (Cordero et al. 2026) as distributed by
its authors: {n_photos} photographs of 59 Caribbean coral species, each coral marked with a
species-labelled bounding box in YOLO format. The published dataset held 10,373 photographs;
12 were absent from the copy we received. The annotations carry a CC BY 4.0 licence; the
photographs come from GBIF, CoralNet and Corals of the World and were used for local
training only. Of the boxes, 1,278 span the whole frame (1,273 photographs carry only such a
box). We kept them, as the original study did, so that a photograph showing one colony that
fills the frame is labelled as that species rather than as an empty scene.</p>

<h2>Partitioning into training, validation and test sets</h2>
<p>We assigned every photograph to one set before any augmentation, in the proportions
{round((1 - aug['valid_frac'] - aug['test_frac']) * 100)}/{round(aug['valid_frac'] * 100)}/{round(aug['test_frac'] * 100)}
(training, validation, test) used by Cordero et al. To keep rare species represented in
every set, we stratified on each photograph's rarest species, defined as the species that
appears in the fewest photographs across the dataset. Within each stratum we shuffled the
photographs with seed {aug['seed']} and allocated them by the three proportions, placing at
least one photograph in validation and one in test for every stratum of three or more. This
produced {n(photos.get('train'))} training, {n(photos.get('valid'))} validation and
{n(photos.get('test'))} test photographs. We froze the assignment in a split file that VEGA
replayed for every model, so all models were selected on the same validation photographs and
scored on the same test photographs.</p>
<p>Because the split preceded augmentation, no version of a validation or test photograph was
ever used in training. This is the strictest ordering available. Cordero et al. report a
70/15/15 split without stating whether it preceded augmentation; their evaluation set of 4,843
images is about 15 percent of their 32,423 augmented images, which is consistent with either
order.</p>

<h2>Offline augmentation</h2>
<p>We reproduced the augmentations listed in Table 2 of Cordero et al. as an offline step.
For each training photograph we wrote {aug['variants']} augmented copies, so each training
photograph appeared {aug['variants'] + 1} times ({n(images.get('train'))} training images).
Validation and test photographs were not augmented. Each copy drew every transform
independently: a horizontal flip and a vertical flip, each with probability
{aug['flip_horizontal_p']}; one of no rotation, 90 degrees clockwise, 90 degrees
counter-clockwise or 180 degrees, chosen uniformly; a rotation drawn uniformly from
{aug['rotation_deg']} degrees either way; horizontal and vertical shear, each drawn uniformly
from {aug['shear_deg_x']} degrees either way; a hue shift drawn uniformly from
{aug['hue_deg']} degrees either way and applied in the 2-degree steps of 8-bit HSV images,
rounded toward zero (so at most {int(aug['hue_deg'] // 2) * 2} degrees);
a saturation change of up to {round(aug['saturation_frac'] * 100)} percent; an exposure
(brightness) change of up to {round(aug['exposure_frac'] * 100)} percent; a Gaussian blur
with standard deviation up to {aug['blur_sigma_max_px']} pixels; and salt-and-pepper noise on
up to {aug['noise_pixel_frac_max'] * 100:g} percent of pixels. We applied every geometric
transform to the four corners of each box and kept the enclosing box, clipped to the frame; a
box narrower than {aug['min_box_px']} pixels or keeping less than
{round(aug['min_visible_area_frac'] * 100)} percent of its area after clipping was dropped
({n(manifest.get('boxes_dropped_by_clipping'))} of {n_boxes} boxes). Each copy's random draws
were seeded from its photograph's name and copy number, so the augmented set can be rebuilt
exactly. Copies were saved as JPEG at quality {aug['jpeg_quality']}.</p>

<h2>Model training</h2>
<p>We trained every model through VEGA, the model-training module of the VICARIUS platform,
which runs Ultralytics {esc(versions.get('ultralytics', 'unknown'))} on PyTorch
{esc(versions.get('torch', 'unknown'))}. All models started from COCO-pretrained weights.
The reproduction (A0) followed Table 2: YOLOv8x at {base['imgsz']} by {base['imgsz']}
pixels, batch size {base['batch']}, seed {base['seed']}, up to {base['epochs']} epochs. The
paper does not report an optimizer, learning rate, early stopping or other settings, so we
used the Ultralytics defaults: automatic optimizer selection, initial learning rate
{base['lr0']} decaying linearly to {base['lrf']} of that value, momentum {base['momentum']},
weight decay {base['weight_decay']}, {base['warmup_epochs']:g} warm-up epochs, loss gains of
{base['box']} (box), {base['cls']} (class) and {base['dfl']} (distribution focal), and early
stopping after {base['patience']} epochs without improvement in validation fitness. We also
kept the default online augmentation applied during training: mosaic with probability
{base['mosaic']} (switched off for the last {base['close_mosaic']} epochs), hue, saturation
and value jitter of {base['hsv_h']}, {base['hsv_s']} and {base['hsv_v']}, translation of
{base['translate']}, scaling of {base['scale']} and horizontal flips with probability
{base['fliplr']}. The weights kept for each model came from the epoch with the highest
validation fitness (0.1 times mAP50 plus 0.9 times mAP50-95).</p>
<p>Three settings differed from Table 2. Dropout (0.10 in the paper) was not applied, because
Ultralytics applies dropout only to classification models and it has no effect on a detector.
The data loader used 12 workers rather than 8, which changes loading speed only. Training ran
on two NVIDIA RTX 5090 GPUs in data-parallel mode, with the batch of {base['batch']} split
evenly between them, where the paper used one RTX 5070 Ti.</p>
<p>Each challenger changed one factor from A0 and kept every other setting:</p>
<table><tr><th>Model</th><th>Factor</th><th>Change from A0</th></tr>{challengers}</table>

{leaky_paragraph}
<h2>Evaluation</h2>
<p>We scored every model on the {n(photos.get('test'))} test photographs with the
Ultralytics validator at a confidence threshold of {protocol.get('conf_threshold')} and a
non-maximum-suppression IoU threshold of {protocol.get('iou_threshold')}, each model at its
own training image size. We report box precision, recall, mAP50 (mean average precision at an
IoU of 0.5, averaged over species) and mAP50-95 (averaged over IoU thresholds from 0.5 to
0.95), overall and per species. Each configuration was trained once, so differences smaller
than about one mAP50 point fall within run-to-run variation. VEGA wrote a full evaluation
report for each model.</p>

<h2>Reference</h2>
<p class="meta">Cordero E, Sherman CE, Yadav P, Kumar Raju V, Townsend JE, Courtney TA (2026).
Coral Intel: a YOLOv8 deep learning framework for monitoring Caribbean corals. Intelligent
Marine Technology and Systems 4:18. doi:10.1007/s44295-026-00107-5</p>
<p class="meta">Lauren Olinger</p>
</body></html>"""


def write_methods(project_dir: Path, arms: list[Mapping[str, Any]],
                  protocol: Mapping[str, Any],
                  printer: Callable[[Path, Path], None] | None = None) -> Path:
    """Write the methods page into the share folder and return its PDF path."""
    share = Path(project_dir) / SHARE_DIRNAME
    share.mkdir(parents=True, exist_ok=True)
    aug = load_preset(AUGMENT_PARAMS_FILE)["params"]
    page = methods_html(arms, protocol, aug, dataset_manifest(), library_versions(), ast_now())
    html_path = Path(project_dir) / STUDY_DIRNAME / "methods.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page)
    pdf_path = share / METHODS_PDF_NAME
    (printer or print_pdf)(html_path, pdf_path)
    return pdf_path


def read_photo_counts(dataset_dir: Path | None = None) -> dict[str, Any] | None:
    """Source photographs per split, before augmentation, from the staged
    dataset's augment_manifest.json; None when it cannot be read."""
    path = (dataset_dir or REPO_DIR / "supporting_data" / "bbox_datasets" / DATASET_NAME) \
        / "augment_manifest.json"
    try:
        doc = read_json(path)
    except ValueError:
        return None
    return ((doc or {}).get("counts") or {}).get("source_images_per_split")


def assemble_share(project_dir: Path, arms: list[Mapping[str, Any]],
                   results: list[Mapping[str, Any]], protocol: Mapping[str, Any],
                   printer: Callable[[Path, Path], None] = print_pdf) -> Path:
    """Copy each scored arm's VEGA PDF into the share folder and add the summary.

    Returns the share folder path.
    """
    share = Path(project_dir) / SHARE_DIRNAME
    share.mkdir(parents=True, exist_ok=True)
    split_counts = None
    for row in results:
        if row.get("status") != "ok":
            continue
        source = Path(project_dir) / "step7_evaluateModel" / f"eval_{row['run_name']}" / "report.pdf"
        arm = next(a for a in arms if a["id"] == row["id"])
        if source.is_file():
            shutil.copy2(source, share / share_pdf_name(arm))
        split_counts = split_counts or (row.get("scores") or {}).get("split_counts")
    html_path = Path(project_dir) / STUDY_DIRNAME / "summary.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(summary_html(arms, results, split_counts, ast_now(), protocol,
                                      read_photo_counts()))
    printer(html_path, share / SUMMARY_PDF_NAME)
    write_methods(project_dir, arms, protocol, printer)
    return share


def find_or_create_project(client: VegaClient, project_dir: str | None,
                           dataset: str = DATASET_NAME) -> Path:
    """Return the project to fill: the one named, or a freshly minted one."""
    if project_dir:
        path = Path(project_dir)
        if not (path / "project.json").is_file():
            raise VegaError(f"{path} is not a VEGA project (no project.json)")
        return path
    minted = client.create_project(dataset, INITIALS)
    return Path(minted["run_dir"])


def wait_for_pid(pid: int, progress: Progress,
                 sleep: Callable[[float], None] = time.sleep) -> None:
    """Hold until process ``pid`` has exited (a study queued behind another)."""
    import os  # noqa: PLC0415
    logged = False
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            pass
        if not logged:
            progress.write("queued_behind", pid=pid)
            logged = True
        sleep(BUSY_RETRY_SECONDS)


def refresh_share(share_project: Path, progress: Progress) -> None:
    """Bring the shared folder up to date; a failure is logged, never fatal."""
    try:
        import coral_intel_study_report as report  # noqa: PLC0415
        report.refresh(share_project)
    except Exception as exc:  # noqa: BLE001, a report problem must not stop training
        progress.write("share_refresh_failed", error=str(exc)[:300])


def main(argv: list[str] | None = None) -> int:
    """CLI entry: mint or resume a study's project, run its arms, refresh the share folder."""
    import coral_intel_study_report as report  # noqa: PLC0415
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", default="main", choices=sorted(report.STUDIES),
                        help="which study to run: main (A0 to B3) or leaky (L0)")
    parser.add_argument("--project_dir", default="",
                        help="resume this project; omitted, a new one is minted")
    parser.add_argument("--share_project", default="",
                        help="the main study project holding the share folder "
                             "(default: this project, for the main study)")
    parser.add_argument("--after_pid", type=int, default=0,
                        help="wait until this process has exited before starting")
    parser.add_argument("--allow_metashape_pid", type=int, action="append", default=[],
                        help="a Metashape process Lauren has allowed to stay open beside "
                             "this run; may be given more than once")
    parser.add_argument("--allow_manual_edit_windows", action="store_true",
                        help="let interactive Metashape windows on the manual-edit workspace "
                             "stay open beside this run (scripted runs still hold it)")
    parser.add_argument("--vega_url", default=VEGA_URL)
    parser.add_argument("--arms", default="",
                        help="comma-separated arm ids to run (default: all, in order)")
    args = parser.parse_args(argv)

    spec = report.STUDIES[args.study]
    try:
        arms = resolve_arms(spec["presets"])
        protocol = load_eval_protocol()
    except ValueError as exc:
        print(f"ERROR: the design is not controlled: {exc}", file=sys.stderr)
        return 2
    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = sorted(set(wanted) - {a["id"] for a in arms})
    if unknown:
        print(f"ERROR: unknown arm ids {unknown}", file=sys.stderr)
        return 2
    if args.study != "main" and not args.share_project:
        print("ERROR: --share_project (the main study project) is required for this study",
              file=sys.stderr)
        return 2

    client = VegaClient(args.vega_url, headers=backend_headers())
    if args.after_pid:
        wait_for_pid(args.after_pid, Progress(Path(args.share_project or args.project_dir or ".")
                                              / STUDY_DIRNAME / f"{args.study}_queue.jsonl"))
    try:
        project_dir = find_or_create_project(client, args.project_dir, spec["dataset"])
    except VegaError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    share_project = Path(args.share_project) if args.share_project else project_dir
    report.register(share_project, args.study, project_dir)
    progress = Progress(project_dir / STUDY_DIRNAME / "progress.jsonl")
    progress.write("start", project_dir=str(project_dir), study=args.study,
                   arms=[a["id"] for a in arms])
    write_json(project_dir / STUDY_DIRNAME / "arms.json",
               {"dataset": spec["dataset"], "protocol": protocol,
                "eval_preset": str(EVAL_PRESET_FILE),
                "arms": arms, "written": ast_now().isoformat()})
    print("Step 1 complete: design resolved and project ready", flush=True)

    allowed = frozenset(args.allow_metashape_pid)
    if allowed or args.allow_manual_edit_windows:
        progress.write("ladder_override", allowed_metashape_pids=sorted(allowed),
                       manual_edit_windows=args.allow_manual_edit_windows, by="LO")
    driver = Driver(client, project_dir, progress, protocol,
                    blocker=lambda: ladder_blocker(allowed_pids=allowed,
                                                   allow_manual_edit=args.allow_manual_edit_windows),
                    on_change=lambda: refresh_share(share_project, progress))
    refresh_share(share_project, progress)
    results = run_all(driver, arms, wanted, progress, project_dir)
    print(f"Step 2 complete: {sum(r['status'] == 'ok' for r in results)} of "
          f"{len(results)} arms scored", flush=True)

    refresh_share(share_project, progress)
    share = share_project / SHARE_DIRNAME
    progress.write("finished", share=str(share),
                   files=sorted(p.name for p in share.iterdir()) if share.is_dir() else [])
    print(f"Step 3 complete: share folder at {share}", flush=True)
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

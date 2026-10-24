"""Bounding Boxes: the external box-dataset page as a self-contained blueprint.

The orchestrator mounts this natively at /boxes (no iframe, no page of its
own). The host supplies a ``paths_provider`` callable returning:

    source_path   the collaborator distribution of the Coral Intel dataset on
                  this machine (images/ + annotations/ + LICENSE.md)
    bbox_root     <repo>/supporting_data/bbox_datasets, where datasets are
                  staged for training
    seg_root      <module>/inprocess/_bbox, holding box-to-mask conversions
    scripts_dir   <repo>/scripts, holding _boxes/ingest_coral_intel.py and
                  _boxes/bbox_to_seg.py
    python        the interpreter those tools run under
    inprocess_root  the orchestrator's projects folder (<module>/inprocess by
                  default), where a training project minted from a staged
                  dataset is written and where existing ones are looked for.
                  Optional: without it the handoff answers 500 and nothing
                  else changes.

One background job runs at a time; a second request answers 409 rather than
queueing. Everything this page touches sits outside the TCRMP survey: staged
datasets under supporting_data, conversions under inprocess/_bbox, and nothing
in the per-mask registry, the coverage matrix, or STATS.

A conversion job (bbox_to_seg.py) loads a SAM3 model onto a card, so it asks
the platform's GPU right-of-way gate before it starts, the same tier="human"
check every other card-holding reef_point_seg stage asks
(docs/superpowers/specs/2026-09-28-gpu-traffic-design.md); an ingest job
touches no card and asks nothing.

The one thing this page does write into a project folder is a TRAINING
project: create-project mints a run_* dir that carries the dataset's PATH and
no imagery at all (see _reefreview/training_projects.py). Every scan of
annotation work skips it, so those non-TCRMP frames still reach none of the
above.

Training itself never happens here. The VEGA main window is the one training
surface (Lauren, 2026-09-04): "Train in VEGA" hands a staged dataset to it by
minting that training project (or reusing the one the dataset already has)
and answering the URL that opens it in the main window on Train Model. The
main window's index route honours ?project_dir=<absolute run dir> for any
folder under its projects folder that holds a project.json; inprocess_root
above IS that folder, so every project minted here passes that guard.
"""

import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from flask import Blueprint, jsonify, render_template, request, url_for

from . import ingest_coral_intel as ingest

_PKG = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_PKG)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# The single place a training project's run dir is written, for either kind of
# training source. RunDirExhausted maps to 409, the same shape the VICARIUS
# launcher answers when a day's run folders for one set of initials run out.
from _reefreview.training_projects import (  # noqa: E402
    BBOX_PROJECT_TYPE,
    RunDirExhausted,
    mint_training_project,
)

# gpu_claim.py (GPU right-of-way / GPU traffic client, docs/superpowers/
# specs/2026-09-28-gpu-traffic-design.md) lives one level down from
# _SCRIPTS_DIR, in pipeline_orchestrator/, not directly under it, so it needs
# its own sys.path entry: when this blueprint is mounted inside the running
# orchestrator (app.py), pipeline_orchestrator is already on sys.path (it is
# app.py's own directory), but this module must also import cleanly on its
# own, e.g. under a test that imports _boxes.blueprint directly.
_ORCHESTRATOR_DIR = os.path.join(_SCRIPTS_DIR, "pipeline_orchestrator")
if _ORCHESTRATOR_DIR not in sys.path:
    sys.path.insert(0, _ORCHESTRATOR_DIR)
import gpu_claim  # noqa: E402

# The page's asset cache-bust is written into the template itself
# (boxes/index.html, both ?v= strings), the same way the coverage matrix page
# carries its own. Nothing here needs to know it.
DEFAULT_CONVERT_LIMIT = 200
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,60}$")
# Reviewer initials, the shape the matrix and the launcher use. Checked here
# so the refusal names the field the reader can see; the shared minter
# enforces the same rule.
INITIALS_RE = re.compile(r"^[A-Za-z]{1,4}$")
# Where a staged dataset keeps its YOLO label files. Mirrors
# TCRMPtrain_oceankindCV/src/detect_dataset.py::BBOX_LABEL_DIRS; this module
# only has to answer "are there labels at all", so it does not load that tool.
BBOX_LABEL_DIRS = ("labels_localized", "labels")

# How a project folder is recognised under inprocess_root: the launcher and
# the training-project minter both name run dirs run_* and write project.json
# at their top level.
RUN_DIR_PREFIX = "run_"
PROJECT_JSON = "project.json"
# The main window's auto-open contract (pipeline_orchestrator/app.py index()):
# GET <app root>/?project_dir=<absolute run dir> opens that project on render.
MAIN_WINDOW_PATH = "/"
PROJECT_PARAM = "project_dir"
# How much of a hostile dataset name is echoed back in a refusal.
NAME_ECHO_LIMIT = 60
# The collision suffix the minter adds to a same-day run dir (run_x_2 ..
# run_x_99); a name without one is the first project of its day.
RUN_SUFFIX_RE = re.compile(r"_(\d+)$")
FIRST_RUN_SUFFIX = 1

AST = timezone(timedelta(hours=-4))
_LOG_MAXLEN = 20000

COVERAGE_NOTE = ("These images are outside the TCRMP survey and never appear "
                 "in the label coverage matrix.")

# GPU right-of-way / GPU traffic (docs/superpowers/specs/2026-09-28-gpu-
# traffic-design.md): bbox_to_seg.py loads a SAM3 model onto a card, exactly
# like every other tier="human" reef_point_seg stage (module.yaml's
# gpu.stages block), but its Popen sits outside stage_runner.py's runner-key
# gate: this blueprint spawns it directly through JobRunner, so it asks the
# same gate itself rather than through a runner key.
GPU_CLAIM_MODULE = "reef_point_seg"
GPU_CLAIM_TIER = "human"
# Every automated backend claim is signed with this VICARIUS-wide agent
# initials (platform rule: "Agent actions are signed LO"); this route has no
# per-request operator identity (api_convert takes no "by" field) to attach
# a claim to instead.
GPU_CLAIM_BY = "LO"
# gpu_claim.check() decisions that mean "spawn". ADR 0008 / gpu_claims.py's
# own contract is that only "granted" ever means proceed; "unregistered"
# (port 5090 down) is the sole exception. This is deliberately an allow-list,
# matching stage_runner.py's own rule: a decision word this file has never
# seen ("wait", one added after this file was written, or any other), and an
# answer with no "decision" key at all, must both refuse by default, the same
# as "refused" and "needs_pin" (a human-tier job asked to bypass a model-tier
# holder, which this route has no admin-session concept to grant a PIN
# bypass for). A block-list would do the opposite: it would start the
# conversion on any word it had not been told to recognise.
_GPU_STARTABLE_DECISIONS = frozenset({"granted", "unregistered"})

# The refusal shown when gpu_claim.check() answers a decision this file must
# not start on, and that answer carries no "message" of its own (an unknown
# decision word, or a reply with no "decision" key at all).
GPU_NO_USABLE_ANSWER_MESSAGE = "GPU right of way returned no usable answer."


def ast_now():
    return datetime.now(AST).strftime("%Y-%m-%d %H:%M:%S")


def _mtime_stamp(path):
    try:
        return datetime.fromtimestamp(os.path.getmtime(path), AST).strftime(
            "%Y-%m-%d %H:%M:%S")
    except OSError:
        return ""


def _json_body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _folder_name(path):
    """The last segment of a folder path, with any trailing separators
    removed.

    The Coral Intel distribution sits in a personal cloud-drive tree on this
    machine, and the page shows only which folder it is, never where it is.
    Answering the name here keeps the path on the server: it is never sent to
    a browser, so it cannot be read out of a network panel or picked up by a
    later reader of this route.

    Parameters:
      path: an absolute or relative folder path, or "" when the host has none.
    Returns:
      the folder's own name, or "" when path is empty or is the root.

    Example:
      _folder_name("/a/b/Coral Intel Object Detection Dataset/") ->
      "Coral Intel Object Detection Dataset"
    """
    return os.path.basename(str(path or "").rstrip("/" + os.sep))


def _echo_name(name):
    """A dataset name as a refusal shows it: '(none)' when blank, cut to
    NAME_ECHO_LIMIT characters when longer, so a hostile body cannot fill
    the page with itself."""
    name = str(name or "")
    if not name:
        return "(none)"
    return name if len(name) <= NAME_ECHO_LIMIT else name[:NAME_ECHO_LIMIT] + "..."


def _parse_created(value):
    """project.json's created_at (ISO 8601 with offset) as an aware datetime
    in AST, or None when it is missing, not a string, or does not parse, so
    a hand-edited file can neither crash the listing nor sort ahead of a
    real project."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).astimezone(AST)
    except ValueError:
        return None


def _folder_stamp(path):
    """The folder's modification time as an aware datetime in AST: the sort
    key for a project whose created_at does not parse. The epoch when even
    that cannot be read, which sorts such a folder last."""
    try:
        return datetime.fromtimestamp(os.path.getmtime(path), AST)
    except OSError:
        return datetime.fromtimestamp(0, AST)


def _run_suffix(name):
    """The numeric collision suffix of a run dir name (run_x_10 gives 10),
    FIRST_RUN_SUFFIX when there is none, so same-second ties order by number
    rather than as strings (where _10 would sit under _2)."""
    match = RUN_SUFFIX_RE.search(name)
    return int(match.group(1)) if match else FIRST_RUN_SUFFIX


def _inside_projects_root(run_dir, real_root):
    """True when run_dir is a real directory inside the projects folder:
    not a link, and resolving to a path under real_root. The main window's
    auto-open guard compares unresolved paths, so a link planted under the
    projects folder could otherwise open a folder from anywhere on disk."""
    if os.path.islink(run_dir):
        return False
    real = os.path.realpath(run_dir)
    return real != real_root and os.path.commonpath([real_root, real]) == real_root


def _read_project_meta(meta_path, say):
    """The parsed project.json at meta_path, or None when it cannot be read
    or parsed (reported through say) or is not a JSON object."""
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError) as exc:
        say(f"skipping {meta_path}: project.json could not be read ({exc})")
        return None
    return meta if isinstance(meta, dict) else None


def _trains_on(meta, want, meta_path, say):
    """True when meta is a box-training project whose training_source.path
    resolves to want. A path that cannot be resolved (a NUL byte, an
    unreadable mount) is reported through say and counts as no match rather
    than raising out of the listing."""
    if meta.get("project_type") != BBOX_PROJECT_TYPE:
        return False
    source = meta.get("training_source")
    src_path = str(source.get("path") or "") if isinstance(source, dict) else ""
    if not src_path:
        return False
    try:
        return os.path.realpath(src_path) == want
    except (OSError, ValueError) as exc:
        say(f"skipping {meta_path}: training_source.path cannot be resolved ({exc})")
        return False


def _project_entry(run_dir, name, meta):
    """The listing row for one matching project, with its sort key under
    ``_sort``: the parsed created_at (the folder's own time when it does not
    parse) and the numeric run suffix."""
    created = _parse_created(meta.get("created_at"))
    return {
        "run_dir": os.path.abspath(run_dir),
        "run_dir_name": name,
        "project_id": str(meta.get("id") or name[len(RUN_DIR_PREFIX):]),
        "display_name": str(meta.get("name") or name),
        "created_at": created.strftime("%Y-%m-%d %H:%M:%S") if created else "",
        "initials": str(meta.get("initials") or ""),
        "current_step": meta.get("current_step"),
        "_sort": (created or _folder_stamp(run_dir), _run_suffix(name)),
    }


def main_window_url(run_dir):
    """Root-relative URL that opens ``run_dir`` in the VEGA main window.

    The orchestrator's index route honours ?project_dir=<absolute run dir>
    and opens that project on render, provided the folder sits under its
    projects folder (the inprocess_root the host passes to this blueprint)
    and holds a project.json. request.script_root is the prefix the whole
    app is mounted under, empty when it is served at the root, so the usual
    result is "/?project_dir=%2Fmnt%2F...%2Frun_26-09-04_LO_train_coral_intel".
    """
    return (f"{request.script_root}{MAIN_WINDOW_PATH}"
            f"?{PROJECT_PARAM}={quote(str(run_dir), safe='')}")


def list_training_projects(inprocess_root, dataset_path, log=None):
    """Training projects already minted for one staged box dataset, newest
    first.

    Scans the run_* folders directly under ``inprocess_root`` (the folder
    the VICARIUS launcher and this page both mint into) and keeps every
    project.json whose project_type is the box-training type and whose
    training_source.path resolves to ``dataset_path``. Nothing below the
    first level is read. Skipped, and reported through ``log``: a run_*
    entry that is a link or resolves outside the projects folder (the main
    window's guard would open it unresolved), a project.json that cannot be
    read or parsed, and a source path that cannot be resolved. A run folder
    with no project.json is not a project and is skipped silently.

    Returns a list of dicts: run_dir (absolute), run_dir_name, project_id,
    display_name, created_at (AST 'YYYY-MM-DD HH:MM:SS', blank when the
    stored value does not parse), initials and current_step. Order: the
    parsed created_at, newest first, a project with an unparseable stamp
    sorting by its folder's own time; same-second ties by the numeric run
    suffix, highest first. Empty when the root is unset or missing, or when
    no project matches.
    """
    if not inprocess_root or not dataset_path or not os.path.isdir(inprocess_root):
        return []
    say = log or (lambda m: None)
    real_root = os.path.realpath(inprocess_root)
    want = os.path.realpath(dataset_path)
    found = []
    for name in sorted(os.listdir(inprocess_root)):
        if not name.startswith(RUN_DIR_PREFIX):
            continue
        run_dir = os.path.join(inprocess_root, name)
        meta_path = os.path.join(run_dir, PROJECT_JSON)
        if not os.path.isdir(run_dir) or not os.path.isfile(meta_path):
            continue
        if not _inside_projects_root(run_dir, real_root):
            say(f"skipping {run_dir}: a link, or a folder outside the projects folder, "
                "which the main window must not open")
            continue
        meta = _read_project_meta(meta_path, say)
        if meta is None or not _trains_on(meta, want, meta_path, say):
            continue
        found.append(_project_entry(run_dir, name, meta))
    found.sort(key=lambda proj: proj["_sort"], reverse=True)
    for proj in found:
        del proj["_sort"]
    return found


def _gpu_refusal():
    """Ask the GPU right-of-way gate before a conversion job's Popen.

    Returns:
        (refusal, answer). refusal is None when the job may start
        (answer["decision"] is in _GPU_STARTABLE_DECISIONS, checked as an
        allow-list: a decision word this file has never seen, and an answer
        with no "decision" key at all, both refuse rather than starting the
        job on a word nobody taught this file to recognise). Otherwise
        refusal is the message to answer the request with (the answer's own
        "message" when it has one, else GPU_NO_USABLE_ANSWER_MESSAGE),
        exactly the way this blueprint already reports a failed start (the
        same jsonify({"error": ...}), a 4xx status shape api_convert already
        uses for a busy JobRunner). answer is the decision dict
        gpu_claim.check() returned, kept so the caller can build the child's
        environment from it (gpu_claim.launch_env) and, after Popen, claim
        the same cards with the child's own pid (gpu_claim.card_uuids).
    """
    answer = gpu_claim.check(GPU_CLAIM_MODULE, tier=GPU_CLAIM_TIER, by=GPU_CLAIM_BY)
    if answer.get("decision") not in _GPU_STARTABLE_DECISIONS:
        message = answer.get("message") or GPU_NO_USABLE_ANSWER_MESSAGE
        return message, answer
    return None, answer


def _gpu_claim_after_spawn(pid, answer, log):
    """Claim the card for a freshly spawned bbox-to-seg conversion.

    Called immediately after Popen returns, with the child's own pid, so the
    registry protects the process that actually holds the card rather than
    the orchestrator's own pid.

    No release is ever sent for this claim: the 5090 watcher clears a claim
    whose pid has exited, on its own liveness sweep, the same way it
    reconciles any other unclaimed or dead holder. This one-shot background
    job needs no release thread of its own, unlike stage_runner.py's gated
    stages, which release explicitly because their log-reader thread already
    watches for the child's exit for an unrelated reason (draining stdout).

    Parameters:
        pid: the child process's pid.
        answer: the decision dict from the pre-spawn _gpu_refusal() call,
            used to name the same cards on the claim (cards=).
        log: callable(str) for a claim that came back anything other than
            "granted" or "unregistered" (a race, or 5090 going down between
            the check and the claim): the process already exists, so
            refusing the claim after the fact cannot un-spawn it, and this
            blueprint has no policy left to enforce at this point.

    Returns:
        None always. Sends None, never an empty list, when the answer named
        no cards at all: the claims registry reads an empty list and "no
        restriction" as two different things, and this is the latter.
    """
    cards = gpu_claim.card_uuids(answer) or None
    result = gpu_claim.claim(GPU_CLAIM_MODULE, tier=GPU_CLAIM_TIER, run_id=str(pid),
                             pid=pid, by=GPU_CLAIM_BY, cards=cards)
    decision = result.get("decision")
    if decision not in ("granted", "unregistered"):
        log(f"GPU claim for the conversion job (pid {pid}) was not granted after "
            f"spawn ({decision!r}); the process is already running and proceeds "
            "without a claim")


# ── Background job runner ────────────────────────────────────────────────

class JobRunner:
    """One background job at a time (ingest or convert).

    The same small shape the model lab's runner uses: every status read polls
    the process, so a job whose process has exited flips out of "running" on
    that read, with its exit code, instead of leaving a stale claim on the
    page. A deliberate stop reports idle, never an error.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._log = deque(maxlen=_LOG_MAXLEN)
        self._state = {
            "phase": "idle",
            "kind": "",
            "name": "",
            "out_dir": "",
            "exit_code": None,
            "started_at": "",
            "message": "",
        }

    def _capture(self, proc, log):
        stream = proc.stdout
        if stream is None:
            return
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                log.append(line.rstrip("\n"))
        except (ValueError, OSError):
            pass

    def _reconcile(self):
        proc = self._proc
        if proc is None or self._state["phase"] != "running":
            return
        code = proc.poll()
        if code is None:
            return
        self._state["exit_code"] = code
        self._state["phase"] = "done" if code == 0 else "error"
        kind = self._state["kind"] or "job"
        self._state["message"] = (f"{kind} finished" if code == 0
                                  else f"{kind} exited with code {code}")

    def is_busy(self):
        with self._lock:
            self._reconcile()
            return self._state["phase"] == "running"

    def busy_message(self):
        with self._lock:
            kind = self._state["kind"] or "job"
            name = self._state["name"] or "a job"
        return (f"another job is already running ({kind}: {name}). Wait for it to "
                "finish, or press Stop in Progress.")

    def start(self, cmd, *, kind, name="", out_dir="", cwd=None, env_extra=None):
        """Start a background job. env_extra, when given, is layered over a
        copy of this process's own environment (GPU right-of-way variables
        for a job that was gated; omitted, this is unchanged from before
        that gate existed)."""
        with self._lock:
            self._reconcile()
            if self._state["phase"] == "running":
                return None, "busy"
            self._log = deque(maxlen=_LOG_MAXLEN)
            env = os.environ.copy()
            if env_extra:
                env.update(env_extra)
            try:
                proc = subprocess.Popen(
                    list(cmd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=cwd,
                    env=env,
                    text=True,
                    start_new_session=True,
                )
            except Exception as exc:
                self._state.update(phase="error", kind=kind, name=name,
                                   exit_code=None,
                                   message=f"could not start {kind}: {exc}")
                return None, str(exc)
            self._proc = proc
            self._state.update(phase="running", kind=kind, name=name,
                               out_dir=out_dir, exit_code=None,
                               started_at=ast_now(), message=f"{kind} running")
            log = self._log
        threading.Thread(target=self._capture, args=(proc, log), daemon=True).start()
        return {"pid": proc.pid, "started_at": self._state["started_at"]}, ""

    def status(self, tail=12):
        with self._lock:
            self._reconcile()
            state = dict(self._state)
            lines = list(self._log)[-int(tail or 0):] if tail else []
        state["log_tail"] = lines
        return state

    def stop(self):
        with self._lock:
            proc = self._proc
            was_running = self._state["phase"] == "running"
        if proc is None or proc.poll() is not None:
            with self._lock:
                self._reconcile()
                if not was_running:
                    return {"stopped": False, "phase": self._state["phase"]}
        else:
            killed = False
            try:
                # start_new_session=True makes the child its own group leader,
                # so pgid == pid identifies OUR group. Anything else means the
                # pid is not ours to signal: fall back to the process only.
                pgid = os.getpgid(proc.pid)
                if pgid == proc.pid:
                    os.killpg(pgid, signal.SIGTERM)
                    killed = True
            except Exception:
                killed = False
            if not killed:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        with self._lock:
            self._proc = None
            self._state.update(phase="idle", exit_code=None,
                               message=f"stopped by you at {ast_now()}")
            phase = self._state["phase"]
        return {"stopped": True, "phase": phase}


# ── Listings ─────────────────────────────────────────────────────────────

# The two optional sidecars the generic YOLO stager writes beside a staged
# dataset (scripts/_boxes/ingest_yolo_boxes.py): the full name behind every
# class token, and the split every image had in the experiment the dataset
# came from. The Coral Intel stager writes neither.
NAMES_CSV = "names.csv"
SPLIT_PINS_JSON = "split_pins.json"
# The header a names table must carry, mirroring ingest_yolo_boxes.NAMES_HEADER.
# Spelled out here rather than imported so a stager that fails to import can
# never stop this page from rendering.
NAMES_CSV_HEADER = ("token", "name")
# The three splits a pin may name, in the order the card shows them. Mirrors
# ingest_yolo_boxes.SPLIT_DIRS and the split builder's folder names.
SPLIT_NAMES = ("train", "valid", "test")
# How much of one sidecar this page reads. A names table runs to a few
# kilobytes and a pins file to roughly 40 bytes per image, so 8 MB covers
# about 200,000 pinned images; a file larger than that is not a sidecar this
# page can summarise, and the card says so rather than reading it all.
SIDECAR_MAX_BYTES = 8 * 1024 * 1024

# What the generic YOLO stager writes into a manifest when the operator
# supplied no citation and no license: its NO_CITATION and NO_LICENSE_NOTE.
# Both mean the dataset records nothing, so this page answers "" for either
# and the card says none recorded once, rather than presenting the stager's
# own placeholder as the dataset's terms. Spelled out here rather than
# imported for the reason above; test_boxes_routes.py holds the copies
# against the stager that writes them.
NO_CITATION_RECORDED = "none recorded"
NO_LICENSE_RECORDED = ("No license recorded for this dataset. Local training "
                       "only until the terms are recorded.")


def _sidecar_text(path):
    """The text of one optional sidecar inside a staged dataset.

    The read is what enforces the cap: SIDECAR_MAX_BYTES + 1 bytes are asked
    for and a longer answer is refused, so a sidecar a stager is still
    growing cannot be read past the cap between a size check and the read.
    Anything that is not a regular file is refused before it is opened, so a
    folder, a named pipe or a device node in a sidecar's place answers at
    once instead of blocking a page request on a reader that never finishes.

    Parameters:
      path: the sidecar file to read.
    Returns:
      the file's text, or "" when it is absent, is not a regular file, holds
      more than SIDECAR_MAX_BYTES, or is not valid UTF-8. A damaged sidecar
      never stops the listing: the card shows what could be read, and the
      split builder is the one that refuses a malformed pins file when a
      training project is built from the dataset.
    """
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(SIDECAR_MAX_BYTES + 1)
    except OSError:
        # A sidecar this process cannot open or read (a permission, a vanished
        # file, an I/O error) is treated as one that says nothing, which is
        # what every caller below already handles.
        return ""
    if len(raw) > SIDECAR_MAX_BYTES:
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        # Bytes that are not text at all: a card reads none rather than the
        # page answering with an error.
        return ""


def count_names(dataset_dir):
    """How many class tokens the dataset's names table gives a full name.

    Parameters:
      dataset_dir: a staged dataset folder.
    Returns:
      the number of complete token,name rows in names.csv, and 0 when the
      file is absent, unreadable, too large, or does not open with the
      token,name header the stager writes. A row missing either field is not
      counted, because it names nothing.
    """
    text = _sidecar_text(os.path.join(dataset_dir, NAMES_CSV))
    if not text:
        return 0
    try:
        rows = list(csv.reader(text.splitlines()))
    except csv.Error:
        return 0
    if not rows:
        return 0
    header = tuple(cell.strip().lower() for cell in rows[0][:2])
    if header != NAMES_CSV_HEADER:
        return 0
    return sum(1 for row in rows[1:]
               if len(row) >= 2 and row[0].strip() and row[1].strip())


def count_split_pins(dataset_dir):
    """How many images the dataset pins to each split, and whether a pins
    file is there that this page could not read.

    The split builder is the other reader of this file
    (detect_dataset.read_source_pins) and it refuses a file it cannot read
    rather than ignoring it: not JSON, not an object, no pins object, or one
    pin naming a split outside SPLIT_NAMES. This reader is exactly as strict,
    so a file the builder would refuse is never counted as pins here and is
    never reported as absent either. Absent is the benign state, the one
    where VEGA splits the dataset fresh and the operator need do nothing.

    Parameters:
      dataset_dir: a staged dataset folder.
    Returns:
      (counts, unreadable). counts is {"train": n, "valid": n, "test": n}
      when the file pins at least one image, else None. unreadable is True
      when a regular split_pins.json is there and no pins could be read out
      of it, which covers every case the split builder refuses plus a file
      past SIDECAR_MAX_BYTES, which this page declines to read. A file that
      is absent, and one the builder also passes over because it is not a
      regular file, answer (None, False); so does a file that is valid and
      pins nothing, because that is a dataset VEGA splits fresh.
    """
    path = os.path.join(dataset_dir, SPLIT_PINS_JSON)
    if not os.path.isfile(path):
        return None, False
    text = _sidecar_text(path)
    if not text:
        return None, True
    try:
        doc = json.loads(text)
    except ValueError:
        return None, True
    if not isinstance(doc, dict):
        return None, True
    pins = doc.get("pins")
    if not isinstance(pins, dict):
        return None, True
    counts = {split: 0 for split in SPLIT_NAMES}
    for split in pins.values():
        if not isinstance(split, str) or split not in counts:
            return None, True
        counts[split] += 1
    return (counts if any(counts.values()) else None), False


def _own_text(value, nothing_recorded):
    """One manifest text field as the card should show it.

    Parameters:
      value: what the manifest carries, of any type.
      nothing_recorded: the exact text a stager writes to mean the dataset
        records nothing (NO_CITATION_RECORDED, NO_LICENSE_RECORDED).
    Returns:
      the text with its surrounding whitespace removed, or "" when the field
      is absent, is not text, is empty, or is the stager's own words for
      nothing recorded. The card reads "" as none recorded, so a dataset that
      records no terms says that once rather than showing a placeholder as
      though it were a license.
    """
    if not isinstance(value, str):
        return ""
    text = value.strip()
    return "" if text == nothing_recorded.strip() else text


def _count_or_none(value):
    """One optional manifest count as the card should show it.

    Parameters:
      value: what the manifest carries, of any type.
    Returns:
      the count as an int when the manifest carries a whole number, else
      None. None and 0 are different facts on a card: 0 means the stager
      counted none, None means it counted nothing at all, and only the second
      may be shown as unknown.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def list_datasets(bbox_root):
    """Staged box datasets. A finished dataset carries both data.yaml and
    ingest_manifest.json; a half-staged folder is skipped rather than offered
    and then refused.

    Parameters:
      bbox_root: supporting_data/bbox_datasets, or "" when the host has none.
    Returns:
      one dict per staged dataset, sorted by folder name, carrying the
      manifest counts plus the facts read from the folder itself: which
      stager wrote it, how many class tokens its names table names, how many
      images it pins to each split, whether a pins file is there that could
      not be read, and how many staged images carry no box at all
      (n_background, None when the stager recorded no such count).

    The folder a dataset was staged FROM is not among them. Every manifest on
    this box records a path through somebody's personal tree, the page shows
    no such path, and no reader asked for one: it stays in the manifest, where
    the stager wrote it, rather than being sent to a browser for nothing.

    Only the dataset the shipped Coral Intel stager writes falls back to that
    paper's citation and terms when its manifest carries none. Every other
    dataset shows its own text or an empty string, which the card reads as
    none recorded: a borrowed citation would credit a paper the dataset has
    nothing to do with. The two texts a stager writes to mean nothing
    recorded answer as an empty string too, so a card never presents a
    placeholder as the terms the data came under.
    """
    if not bbox_root or not os.path.isdir(bbox_root):
        return []
    out = []
    for name in sorted(os.listdir(bbox_root)):
        d = os.path.join(bbox_root, name)
        manifest_path = os.path.join(d, "ingest_manifest.json")
        if not os.path.isdir(d) or not os.path.isfile(manifest_path):
            continue
        if not os.path.isfile(os.path.join(d, "data.yaml")):
            continue
        try:
            with open(manifest_path) as fh:
                m = json.load(fh) or {}
        except (OSError, ValueError):
            # A manifest that cannot be opened, is not valid UTF-8, or is not
            # JSON. Spec section 6: this listing never throws on a damaged
            # manifest, so the card shows the folder with the facts it could
            # read and none of the rest. Anything other than these two is a
            # fault this page has no answer for and is left to surface.
            m = {}
        if not isinstance(m, dict):
            # A manifest that parses but is not an object (a list, a number, a
            # bare string) names none of the keys below. The card then shows
            # the folder and its counts as zero, rather than the whole page
            # answering with an error.
            m = {}
        is_coral = name == ingest.DATASET_NAME
        n_names = count_names(d)
        split_counts, split_pins_unreadable = count_split_pins(d)
        citation = _own_text(m.get("citation"), NO_CITATION_RECORDED)
        license_note = _own_text(m.get("license_note"), NO_LICENSE_RECORDED)
        out.append({
            "name": name,
            "path": d,
            "n_images": m.get("n_images", 0),
            "n_boxes": m.get("n_boxes", 0),
            "n_degenerate": m.get("n_degenerate", 0),
            "n_degenerate_exact": m.get("n_degenerate_exact", 0),
            "n_classes": m.get("n_classes", len(m.get("classes") or [])),
            "classes": [str(x) for x in (m.get("classes") or [])],
            "ingested_at": str(m.get("ingested_at") or "") or _mtime_stamp(manifest_path),
            "limit": m.get("limit"),
            "citation": citation or (ingest.CITATION if is_coral else ""),
            "license_note": license_note or (ingest.LICENSE_NOTE if is_coral else ""),
            "source_kind": str(m.get("source_kind") or ("coral_intel" if is_coral else "")),
            "has_names": n_names > 0,
            "n_names": n_names,
            "has_split_pins": split_counts is not None,
            "split_counts": split_counts,
            "split_pins_unreadable": split_pins_unreadable,
            "n_background": _count_or_none(m.get("n_background")),
            "has_license_file": os.path.isfile(os.path.join(d, "LICENSE.md")),
            "has_localized": os.path.isdir(os.path.join(d, "labels_localized")),
        })
    return out


def trainable_labels(dataset_dir):
    """(label_dir, n_label_files) for a staged dataset: where its YOLO boxes
    are and how many files it holds. ('', 0) when there is no label folder or
    it holds no .txt at all, which is a dataset that would train on nothing."""
    for rel in BBOX_LABEL_DIRS:
        d = os.path.join(dataset_dir, rel)
        if not os.path.isdir(d):
            continue
        try:
            n = sum(1 for fn in os.listdir(d) if fn.endswith(".txt"))
        except OSError:
            n = 0
        if n:
            return d, n
    return "", 0


def _count_rows(path):
    """(label files, ledger rows marked ok) for one converted dataset."""
    n_polygons = 0
    manifest = os.path.join(path, "manifest.csv")
    if os.path.isfile(manifest):
        try:
            with open(manifest, newline="") as fh:
                n_polygons = sum(1 for row in csv.DictReader(fh)
                                 if row.get("status") == "ok")
        except Exception:
            n_polygons = 0
    labels_dir = os.path.join(path, "all_labels")
    try:
        n_labels = sum(1 for fn in os.listdir(labels_dir) if fn.endswith(".txt"))
    except OSError:
        n_labels = 0
    return n_labels, n_polygons


def list_seg_outputs(seg_root):
    """Box-to-mask conversions under inprocess/_bbox."""
    if not seg_root or not os.path.isdir(seg_root):
        return []
    out = []
    for name in sorted(os.listdir(seg_root)):
        d = os.path.join(seg_root, name)
        if not os.path.isdir(d) or not os.path.isdir(os.path.join(d, "all_labels")):
            continue
        n_labels, n_polygons = _count_rows(d)
        out.append({
            "name": name,
            "path": d,
            "n_labels": n_labels,
            "n_polygons": n_polygons,
            "has_readme": os.path.isfile(os.path.join(d, "README.md")),
            "built_at": _mtime_stamp(os.path.join(d, "manifest.csv"))
                        or _mtime_stamp(d),
        })
    return out


# ── Blueprint ────────────────────────────────────────────────────────────

def make_blueprint(paths_provider, *, name="boxes", log_fn=None):
    log = log_fn or (lambda m: print(f"[boxes] {m}"))
    runner = JobRunner()

    bp = Blueprint(name, __name__,
                   template_folder=os.path.join(_PKG, "templates"),
                   static_folder=os.path.join(_PKG, "static"),
                   static_url_path="static")

    def P():
        try:
            p = paths_provider() or {}
        except Exception as e:
            log(f"paths_provider failed: {e}")
            p = {}
        return {
            "source_path": p.get("source_path") or ingest.DEFAULT_SOURCE,
            "bbox_root": p.get("bbox_root", "") or "",
            "seg_root": p.get("seg_root", "") or "",
            "scripts_dir": p.get("scripts_dir", "") or "",
            "python": p.get("python", "") or "",
            "inprocess_root": p.get("inprocess_root", "") or "",
        }

    def _tool(p, filename):
        return os.path.join(p["scripts_dir"], "_boxes", filename)

    # ── Page ───────────────────────────────────────────────────────────
    @bp.route("/")
    def index():
        static_base = url_for(f"{name}.static", filename="_")
        base = static_base.rsplit("/static/", 1)[0]
        return render_template("boxes/index.html", base=base,
                               default_limit=DEFAULT_CONVERT_LIMIT,
                               coverage_note=COVERAGE_NOTE)

    # ── Datasets ───────────────────────────────────────────────────────
    @bp.route("/api/datasets")
    def api_datasets():
        """The staged dataset cards, and whether the Coral Intel distribution
        is on this machine.

        Two facts per answer and nothing else. A citation and terms belong to
        one dataset, never to the page: a page-level pair was here once, held
        the Coral Intel paper, and was read by nothing, which is how a later
        reader would have credited that paper on a dataset with no connection
        to it. Terms and citation are per dataset now, and each one carries
        its own or nothing.

        The source answers its folder NAME. The path runs through a personal
        cloud-drive tree, and the page renders the name alone, so the path is
        never sent. The page's own coverage note, the staging root and the
        conversion root are read by nothing here either: index() passes the
        note into the template, and the two roots are the server's business.
        """
        p = P()
        source = p["source_path"]
        present = bool(source) and os.path.isdir(os.path.join(source, "annotations"))
        return jsonify({
            "ok": True,
            "datasets": list_datasets(p["bbox_root"]),
            "source": {"name": _folder_name(source), "present": present},
        })

    # ── Training project: the handoff to the VEGA main window ──────────
    # The run dir records the dataset's PATH; no image and no symlink is
    # written into it, so non-TCRMP frames never reach a project folder.
    # Step 6 builds its own train/valid/test split from that path when the
    # project is trained, in the main window, with the full training form.

    def _staged_by_name(p, name):
        """The staged dataset the page listed under exactly this name, or
        None. A path, a traversal or a separator matches nothing, so a
        request can never point the minter or the project scan elsewhere."""
        for d in list_datasets(p["bbox_root"]):
            if d["name"] == name:
                return d
        return None

    def _not_staged(name):
        """The 400 answer for a dataset name the page did not list, echoing
        at most NAME_ECHO_LIMIT characters of it."""
        return jsonify({"error": "pick a staged dataset from the list; "
                                 f"{_echo_name(name)} is not one of them. Run the "
                                 "ingest first."}), 400

    @bp.route("/api/training-projects")
    def api_training_projects():
        """Training projects that already exist for one staged dataset,
        newest first, each with the URL that opens it in the main window.
        Query: dataset=<staged name>. A name the page did not list is 400."""
        p = P()
        name = str(request.args.get("dataset") or "").strip()
        chosen = _staged_by_name(p, name)
        if chosen is None:
            return _not_staged(name)
        projects = list_training_projects(p["inprocess_root"], chosen["path"], log)
        for proj in projects:
            proj["open_url"] = main_window_url(proj["run_dir"])
        return jsonify({"ok": True, "dataset": name, "projects": projects})

    @bp.route("/api/create-project", methods=["POST"])
    def api_create_project():
        """Hand a staged dataset to VEGA: answer the training project to open.

        Body: dataset (a staged name), by (initials, 1 to 4 letters), reuse
        (boolean, default true). With reuse true and a training project
        already minted for the dataset, the newest one is answered with
        reused=true and nothing is written; initials are not needed to open
        an existing project. Otherwise a project is minted (initials
        required) and answered with reused=false. Either answer carries
        open_url, the main-window URL that opens the project on Train Model.
        """
        p = P()
        body = _json_body()

        name = str(body.get("dataset") or "").strip()
        chosen = _staged_by_name(p, name)
        if chosen is None:
            return _not_staged(name)

        reuse = body.get("reuse", True)
        if not isinstance(reuse, bool):
            return jsonify({"error": "reuse must be true or false"}), 400

        existing = list_training_projects(p["inprocess_root"], chosen["path"], log)
        if reuse and existing:
            newest = existing[0]
            log(f"handing {name} to the main window as existing project "
                f"{newest['run_dir_name']} ({len(existing)} on disk)")
            return jsonify({"ok": True, "reused": True, "dataset": name,
                            "run_dir": newest["run_dir"],
                            "run_dir_name": newest["run_dir_name"],
                            "project_id": newest["project_id"],
                            "display_name": newest["display_name"],
                            "open_url": main_window_url(newest["run_dir"]),
                            "n_existing": len(existing)})

        by = str(body.get("by") or "").strip()
        if not INITIALS_RE.match(by):
            return jsonify({"error": "your initials must be 1 to 4 letters, so the "
                                     "project folder says who made it"}), 400

        label_dir, n_labels = trainable_labels(chosen["path"])
        images_dir = os.path.join(chosen["path"], "images")
        if not os.path.isdir(images_dir) or not label_dir:
            return jsonify({"error": f"{name} has no images beside labels to train on. "
                                     "Run the ingest again, then try this."}), 400

        if not p["inprocess_root"]:
            return jsonify({"error": "no projects folder is configured for this host"}), 500

        try:
            minted = mint_training_project(
                p["inprocess_root"], "bbox", name, chosen["path"], by,
                extra={"task_default": "detect"})
        except RunDirExhausted as e:
            # Same shape the VICARIUS launcher answers at its own ceiling.
            return jsonify({"error": str(e), "dataset": name}), 409
        except (ValueError, FileNotFoundError) as e:
            return jsonify({"error": str(e), "dataset": name}), 400
        except Exception as e:
            # Never hand a raw exception (and the filesystem path inside it) to
            # the browser: the detail goes to the orchestrator log.
            log(f"create-project failed for {name!r}: {e}")
            return jsonify({"error": "could not make the project. See the orchestrator "
                                     "log for what went wrong.", "dataset": name}), 500

        log(f"training project {minted['run_dir_name']} from {chosen['path']} "
            f"({n_labels} label files), handed to the main window")
        return jsonify({"ok": True, "reused": False, "dataset": name,
                        "run_dir": minted["run_dir"],
                        "run_dir_name": minted["run_dir_name"],
                        "project_id": minted["project_id"],
                        "display_name": minted["display_name"],
                        "open_url": main_window_url(minted["run_dir"]),
                        "n_existing": len(existing), "n_labels": n_labels})

    # ── Ingest ─────────────────────────────────────────────────────────
    @bp.route("/api/ingest", methods=["POST"])
    def api_ingest():
        p = P()
        body = _json_body()
        source = p["source_path"]
        if not source or not os.path.isdir(os.path.join(source, "annotations")):
            # The path itself stays out of the message: it runs through a
            # personal Dropbox tree, and this string is rendered on the page.
            return jsonify({"error": "the Coral Intel distribution folder is not on "
                                     "this machine where this host expects it. Put it "
                                     "there, or point TCRMP_CORAL_INTEL_SOURCE at it, "
                                     "then try again."}), 400
        if not p["bbox_root"]:
            return jsonify({"error": "no staging folder is configured for this host"}), 500

        limit = body.get("limit")
        try:
            limit = int(limit) if limit not in (None, "", 0) else 0
        except (TypeError, ValueError):
            return jsonify({"error": "limit must be a whole number of records"}), 400
        if limit < 0:
            return jsonify({"error": "limit must be a whole number of records"}), 400

        if runner.is_busy():
            return jsonify({"error": runner.busy_message()}), 409

        dest = os.path.join(p["bbox_root"], ingest.DATASET_NAME)
        cmd = [p["python"], _tool(p, "ingest_coral_intel.py"),
               "--source", source, "--dest", dest]
        if limit:
            cmd += ["--limit", str(limit)]
        info, why = runner.start(cmd, kind="ingest", name=ingest.DATASET_NAME,
                                 out_dir=dest)
        if info is None:
            code = 409 if why == "busy" else 500
            return jsonify({"error": runner.busy_message() if why == "busy" else why}), code
        # The source path goes to the orchestrator log and no further: the
        # answer is rendered in the browser, and that path runs through a
        # personal cloud-drive tree.
        log(f"ingest {source} -> {dest}")
        return jsonify({"ok": True, "dest": dest, "pid": info["pid"]})

    # ── Convert ────────────────────────────────────────────────────────
    @bp.route("/api/convert", methods=["POST"])
    def api_convert():
        p = P()
        body = _json_body()

        want = os.path.realpath(str(body.get("dataset") or "").strip())
        listed = list_datasets(p["bbox_root"])
        chosen = None
        for d in listed:
            if os.path.realpath(d["path"]) == want:
                chosen = d
                break
        if chosen is None:
            return jsonify({"error": "pick a staged dataset from the list; "
                                     f"{body.get('dataset') or '(none)'} is not one "
                                     "of them. Run the ingest first."}), 400

        try:
            limit = int(body.get("limit"))
        except (TypeError, ValueError):
            limit = 0
        if limit < 1 or limit > 100000:
            return jsonify({"error": "how many images to convert must be a whole "
                                     "number from 1 to 100000"}), 400

        out_name = str(body.get("name") or "").strip()
        if out_name and not NAME_RE.match(out_name):
            return jsonify({"error": "output name must be 1 to 60 characters of "
                                     "letters, digits, underscore or hyphen: "
                                     + out_name}), 400
        if not out_name:
            out_name = f"{chosen['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
        if not p["seg_root"]:
            return jsonify({"error": "no output folder is configured for this host"}), 500

        if runner.is_busy():
            return jsonify({"error": runner.busy_message()}), 409

        gpu_refusal, gpu_answer = _gpu_refusal()
        if gpu_refusal is not None:
            return jsonify({"error": gpu_refusal}), 409

        out_dir = os.path.join(p["seg_root"], out_name)
        cmd = [p["python"], _tool(p, "bbox_to_seg.py"),
               "--dataset", chosen["path"], "--out", out_dir,
               "--limit", str(limit)]
        if bool(body.get("keep_degenerate")):
            cmd.append("--keep-degenerate")
        info, why = runner.start(cmd, kind="convert", name=out_name, out_dir=out_dir,
                                 env_extra=gpu_claim.launch_env(gpu_answer))
        if info is None:
            code = 409 if why == "busy" else 500
            return jsonify({"error": runner.busy_message() if why == "busy" else why}), code
        _gpu_claim_after_spawn(info["pid"], gpu_answer, log)
        log(f"convert {chosen['name']} ({limit} images) -> {out_dir}")
        return jsonify({"ok": True, "out_dir": out_dir, "name": out_name,
                        "dataset": chosen["path"], "limit": limit,
                        "pid": info["pid"]})

    # ── Job state ──────────────────────────────────────────────────────
    @bp.route("/api/status")
    def api_status():
        state = runner.status(12)
        state["ok"] = True
        return jsonify(state)

    # There is no separate log route. The page's log pane is filled from the
    # log_tail field of /api/status, which every poll already reads, so a
    # second route would be a live address no control on the page reaches.

    @bp.route("/api/stop", methods=["POST"])
    def api_stop():
        out = runner.stop()
        out["ok"] = True
        return jsonify(out)

    # ── Converted datasets ─────────────────────────────────────────────
    @bp.route("/api/seg_outputs")
    def api_seg_outputs():
        p = P()
        return jsonify({"ok": True, "outputs": list_seg_outputs(p["seg_root"]),
                        "seg_root": p["seg_root"]})

    return bp

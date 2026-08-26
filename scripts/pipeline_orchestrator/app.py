"""
Flask application for the Reef Point Seg Orchestrator.
"""

import os
import re
import sys
import csv
import json
import glob
import time
import shutil
import threading
import traceback
from datetime import datetime, timezone, timedelta
from urllib.parse import unquote
from urllib.request import Request, urlopen
from urllib.error import URLError

from flask import (
    Flask, render_template, render_template_string, request, jsonify,
    send_from_directory,
)

import lock_status
import project_manager as pm
import remap_loader
import vicarius_bridge as vic
from stage_runner import StageRunner, find_free_port
from orchestrator_config import (
    REPO_DIR, PYTHON_PATHS, ENTRY_POINTS, WORKING_DIRS, STAGE_PORTS,
    PRESET_DIRS, PATHS, CONFIG_PATH,
)

# Atlantic Standard Time, fixed UTC-4, no daylight saving. Byte-identical
# constant to TCRMPclip_combinedAnnotate/src/provenance.py:AST and
# TCRMPtrain_oceankindCV/src/evaluate_run.py's own copy.
_AST = timezone(timedelta(hours=-4))


def _ast_now():
    """AST ISO-8601 timestamp, second precision, e.g. 2026-06-25T08:13:42-04:00."""
    return datetime.now(_AST).isoformat(timespec="seconds")

# scripts/ on path for the shared Add-Expert-IDs blueprint (mounted natively
# below — no subprocess, no iframe).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _expertids import make_blueprint as _make_expertids_bp
# Cross-project Label Coverage Matrix — also mounted natively (its own full
# page under /matrix, no iframe/subprocess), same as the expert-IDs blueprint.
from _matrix import make_blueprint as _make_matrix_bp
# Label Manager — canonical species vocabulary editor + remap composer + manual
# recode, mounted natively as its own full page under /labels.
from _labels import make_blueprint as _make_labels_bp

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
runner = StageRunner()
current_project = None  # dict (project.json contents)

# SAM3 driver state — updated by a background thread per step 5 launch so the
# orchestrator UI can show progress without the user having to open the sub-app.
sam3_status = {
    "phase": "idle",       # idle | launching | waiting_for_sub_app | configuring |
                           # segmenting | review_ready | error
    "processed": 0,
    "total": 0,
    "running": False,
    "error": None,
    "message": "",
    "updated_at": None,
}
sam3_lock = threading.Lock()
sam3_thread = None

# Combined-annotator (step4test) launch state - this non-chain tile has no
# project step in STEP_KEYS, so it tracks its own port + status here.
step4test_state = {"running": False, "port": None, "started_at": None, "out_dir": None,
                   "project_id": None, "input_dir": None}

# Combined-annotator config collected from the #panel-step4test inputs and POSTed
# with /api/step/step4test/run|reroute. This non-chain tile has no project step,
# so its settings live here (transient, per-launch). _launch_step4test_ui reads
# these and falls back to the inherited Step-5 saved config for any unset key.
step4test_cfg = {}

# Refine loop (step4loop) - the active-learning resume-mode annotator. Same
# shape as step4test_state/step4test_cfg, but tracked separately because it is
# a distinct runner key ("step4loop") that can run independently of step4test
# (though _kill_all_annotators still enforces one annotator process at a time).
step4loop_state = {"running": False, "port": None, "started_at": None, "out_dir": None,
                   "project_id": None, "input_dir": None}

# step4loop panel config (#panel-step4loop inputs), collected client-side and
# POSTed with /api/step/step4loop/run. Transient, per-launch, same pattern as
# step4test_cfg.
step4loop_cfg = {}

# Edit Masks (editmasks) - launches the standalone TCRMPclip_editMasks app
# against the CURRENT project's step4test export, in "edit" session mode
# (boots straight into the review queue via /api/resume, same as step4loop,
# but with no seeding pass: it opens on whatever masks Step 4 already wrote).
# Same shape as step4test_state/step4loop_state, but its own runner key
# ("editmasks") so it can run independently (though _kill_all_annotators
# still enforces one annotator-family app at a time).
editmasks_state = {"running": False, "port": None, "started_at": None, "out_dir": None,
                   "project_id": None, "input_dir": None}


def _load_champion_json(step6_dir):
    """Parsed {step6_dir}/champion.json, or None when absent/unreadable.
    Single read point reused by the fine-tune preset, the step6/8 run
    pickers, and the rounds routes (Task 8 + Task 9)."""
    if not step6_dir:
        return None
    path = os.path.join(step6_dir, "champion.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _step4test_export_dir():
    """The active project's 4.test combined-annotator export dir: holds
    all_images/, all_labels/, data.yaml, class_map.json, segmentations/.
    4.test replaces Steps 4+5, so this is the canonical training + review source.
    Empty string when no project is loaded. The dir name lives only in
    pm.STEP_DIRS["step4test"], so this helper is the single derivation point
    (re-used by _routed_dir, _expertids_paths, and the folder/reset routes)."""
    if current_project is None:
        return ""
    return os.path.join(current_project["project_dir"], pm.STEP_DIRS["step4test"])


# Local full-set clip dir under the repo's supporting_data. This is the SAME
# tree Step 3 reads (TCRMPcvr_chooseImages config.CLIP_DIR), holding the full
# year/period structure the user installed. Routing reads images only from here
# (no Dropbox dependency).
LOCAL_CLIP_DIR = os.path.join(REPO_DIR, "supporting_data", "TCRMP_clip")


def _resolve_routing_clip_dir(step4test_cfg, s4cfg):
    """Clip image source for the 4.test routing pass. The 4.test panel value
    wins, then the saved Step-4 value, then the LOCAL full-set clip dir. Any
    candidate that does not exist on disk is skipped, so stale Dropbox paths
    saved in older projects no longer break routing."""
    for cand in ((step4test_cfg or {}).get("clip_dir"),
                 (s4cfg or {}).get("clip_dir"),
                 LOCAL_CLIP_DIR):
        if cand and os.path.isdir(cand):
            return cand
    return LOCAL_CLIP_DIR


def _clear_routed_input(routed_dir):
    """Remove every routed_input/<year>/ subtree so a fresh route starts clean.
    A prior route leaves sam_click_prompts.json on disk; reusing it after the
    Step 3 selection changed leaks the old run's frames into the annotator
    regardless of the current species filter. Returns the number of year-dirs
    removed; no-op (0) if routed_dir is absent/empty.
    """
    import shutil as _shutil
    if not routed_dir or not os.path.isdir(routed_dir):
        return 0
    n = 0
    for entry in sorted(os.listdir(routed_dir)):
        p = os.path.join(routed_dir, entry)
        if os.path.isdir(p):
            _shutil.rmtree(p, ignore_errors=True)
            n += 1
    return n


_LORES_MARKER = ".lores_mode"


def _write_routed_lores_marker(routed_dir, on):
    """Record which lores_mode a route was produced with, so a later reuse can
    detect a toggle change and re-route instead of serving the wrong resolution."""
    try:
        with open(os.path.join(routed_dir, _LORES_MARKER), "w") as f:
            f.write("1" if on else "0")
    except OSError:
        pass


def _routed_lores_mode(routed_dir):
    """The lores_mode a cached route was built with (True/False), or None for an
    older cache with no marker (treated as 'unknown', never forces a re-route)."""
    p = os.path.join(routed_dir, _LORES_MARKER)
    if not os.path.isfile(p):
        return None
    try:
        return open(p).read().strip() == "1"
    except OSError:
        return None


def _routed_prompts_stale(routed_dir, selected_frames):
    """True iff cached routing prompts exist but are OLDER than the Step 3
    selection -- i.e. the user re-chose frames since the last route, so the
    cache must be invalidated. False when no prompts exist (nothing to
    invalidate) or selected_frames is missing.
    """
    if not selected_frames or not os.path.isfile(selected_frames):
        return False
    prompt = os.path.join(routed_dir, "ids", "sam_click_prompts.json")
    if not os.path.isfile(prompt):
        return False
    return os.path.getmtime(selected_frames) > os.path.getmtime(prompt)


def _flatten_routed_input(routed_dir):
    """Collapse routed_input/<year>/{ids,raw,test_pts} into flat routed_input/
    {ids/sam_click_prompts.json, raw/}. placePoints writes per-year; the combined
    annotator reads flat. Basenames embed the year so cross-year collisions are
    effectively impossible; a dest collision is skipped + counted. test_pts/ is
    discarded (regenerated downstream). Returns {merged, collisions,
    year_dirs_removed}."""
    import glob as _glob
    import shutil as _shutil
    merged = {}
    year_dirs = []
    for prompts in sorted(_glob.glob(os.path.join(routed_dir, "*", "ids", "sam_click_prompts.json"))):
        year_dir = os.path.dirname(os.path.dirname(prompts))
        year_dirs.append(year_dir)
        with open(prompts) as f:
            merged.update(json.load(f))

    flat_ids = os.path.join(routed_dir, "ids")
    flat_raw = os.path.join(routed_dir, "raw")
    os.makedirs(flat_ids, exist_ok=True)
    os.makedirs(flat_raw, exist_ok=True)
    with open(os.path.join(flat_ids, "sam_click_prompts.json"), "w") as f:
        json.dump(merged, f, indent=1)

    collisions = 0
    for year_dir in year_dirs:
        raw_dir = os.path.join(year_dir, "raw")
        if os.path.isdir(raw_dir):
            for name in os.listdir(raw_dir):
                dst = os.path.join(flat_raw, name)
                if os.path.exists(dst):
                    collisions += 1
                    continue
                _shutil.move(os.path.join(raw_dir, name), dst)
        _shutil.rmtree(year_dir, ignore_errors=True)  # drops test_pts + emptied ids/raw

    if collisions:
        print(f"[flatten] WARNING: {collisions} raw filename collision(s) skipped")
    return {"merged": len(merged), "collisions": collisions,
            "year_dirs_removed": len(year_dirs)}


def _sam3_set(**kw):
    with sam3_lock:
        # Same lifecycle rule as _route_set: leaving the error phase clears the
        # process-death residue written by _reconcile_liveness.
        if kw.get("phase") not in (None, "error"):
            sam3_status.pop("died", None)
            sam3_status.pop("log_tail", None)
            if "error" not in kw:
                sam3_status["error"] = None
        sam3_status.update(kw)
        sam3_status["updated_at"] = datetime.now().isoformat()


# Routing driver state (step4test) - a background thread drives placePoints
# headlessly (CPC/OCR detection -> sam_click_prompts.json) before the combined
# annotator window opens. The frontend polls /api/step/step4test/route_status.
route_status = {
    "phase": "idle",       # idle | launching | routing_configure | routing_ocr |
                           # exporting | route_ready | ui_ready | error
    "processed": 0,
    "total": 0,
    "message": "",
    "ui_port": None,
    "error": None,
    # Routing accountability (Task 2): human-readable skip notices + the raw
    # per-frame dropped list from placePoints' configure response, surfaced in
    # the 4.test launch overlay. Reset on each fresh routing pass.
    "warnings": [],
    "dropped": [],
    "updated_at": None,
}
route_lock = threading.Lock()
route_thread = None

# Cancellation for an in-flight routing pass. The orchestrator UI is the ONLY way
# these tools are ever launched, so closing the launch window must mean the work
# stops: no GPU held by a routing sub-app nobody is watching, and no annotator
# kill+relaunch fired by a pass the operator walked away from. _route_drive polls
# this between every step of the pass; the routes that start a pass clear it.
#
# Cancelling is cheap: placePoints flushes detections.json every 50 frames and its
# /api/configure skips frames it already has (outcome "already_processed"), so a
# re-opened route resumes and only redoes the handful of frames since the last
# flush - it never restarts the whole OCR pass from zero.
route_cancel = threading.Event()


def _route_set(**kw):
    with route_lock:
        # Any transition to a non-error phase starts a new lifecycle: clear the
        # process-death residue (marker, log tail, stale error string) so an
        # old death cannot haunt a fresh launch or a healthy recovery.
        if kw.get("phase") not in (None, "error"):
            route_status.pop("died", None)
            route_status.pop("log_tail", None)
            if "error" not in kw:
                route_status["error"] = None
        route_status.update(kw)
        route_status["updated_at"] = datetime.now().isoformat()


def _reconcile_liveness(lock, status, alive_phases, runner_key, label):
    """A cached phase that claims a living process must be verified against
    that process at every read. The 2026-08-13 incident: the annotator died
    at SAM3 load after ui_ready was cached, and this API reported a dead port
    as "Combined annotator ready" for hours while the UI polled it. Flip to
    error (written back under the lock) the moment the process is gone; a
    dead process cannot come back, so the flip is safe and idempotent.

    Ownership: route_status is SHARED by three launch modes (step4test,
    step4loop, editmasks), so the phase setter stamps runner_key/runner_label
    and this check verifies the process that actually owns the current phase;
    runner_key/label arguments are only the fallback for pre-stamp states.

    Races: the runner poll happens outside the status lock (poll_status takes
    the runner's own lock; holding both invites ordering deadlocks), so the
    final write re-checks phase+updated_at and yields to any transition that
    landed in between (a finishing relaunch, a deliberate Stop)."""
    with lock:
        phase = status.get("phase")
        seen_at = status.get("updated_at")
        owner = status.get("runner_key") or runner_key
        owner_label = status.get("runner_label") or label
    if phase not in alive_phases:
        with lock:
            return dict(status)
    if runner.is_running(owner):
        with lock:
            return dict(status)
    proc = runner.poll_status(owner)
    if proc.get("running"):
        with lock:
            return dict(status)
    exit_code = proc.get("exit_code")
    detail = (f"exited (code {exit_code})" if exit_code is not None
              else "is not running (never started or already cleaned up)")
    with lock:
        if (status.get("phase"), status.get("updated_at")) != (phase, seen_at):
            return dict(status)
        status.update(
            phase="error", ui_port=None, died=True,
            error=f"{owner_label} process {detail}",
            message="The process died. See the log tail below, then relaunch.",
            log_tail=list(proc.get("log_tail") or [])[-12:],
        )
        status["updated_at"] = datetime.now().isoformat()
        return dict(status)


def _route_mark_stopped_if_owner(runner_key, message):
    """Deliberate Stop/Reset must never read as a crash, but only the mode
    that OWNS the current ui_ready may rewrite it: step4test's Stop killing
    its own keys must not claim an Edit Masks or Refine session stopped."""
    with route_lock:
        is_owner = (route_status.get("phase") == "ui_ready"
                    and (route_status.get("runner_key") or "step4test") == runner_key)
    if is_owner:
        _route_set(phase="stopped", ui_port=None, error=None, message=message)


def _sam3_mark_stopped(message):
    """Deliberate-Stop guard for the step-5 surface, mirroring
    _route_mark_stopped_if_owner: without this, stopping or completing step 5
    leaves review_ready cached and the liveness check reports the kill as a
    crash."""
    with sam3_lock:
        alive = sam3_status.get("phase") in ("review_ready", "segmenting",
                                             "configuring",
                                             "waiting_for_sub_app")
    if alive:
        _sam3_set(phase="stopped", running=False, error=None, message=message)


def _reset_transient_status(message):
    """Project open/quit invalidates every cached claim about running
    processes: without this, project B's tile inherits project A's ui_ready
    and port (2026-08-14 critic gap). Deliberate lifecycle event, so plain
    idle, never a crash error."""
    _route_set(phase="idle", ui_port=None, error=None, message=message)
    _sam3_set(phase="idle", running=False, error=None, message="")


def _vic_end_step(step, status, notes=""):
    """Emit a VICARIUS process_end event linked to the step's process_start.

    No-op when vicarius is disabled or the step was never started via the
    orchestrator (e.g. stale state).
    """
    if current_project is None:
        return
    s = str(step)
    st = current_project["steps"].get(s, {})
    event_id = st.pop("_vic_event_id", None)
    started = st.get("started_at")
    duration = 0.0
    if started:
        try:
            t0 = datetime.fromisoformat(started).timestamp()
            duration = max(0.0, datetime.now().timestamp() - t0)
        except Exception:
            duration = 0.0
    outputs = list(st.get("outputs", {}).values())
    step_name = f"step{step}_{pm.STEP_DIRS.get(s, '').split('_', 1)[-1] or 'stage'}"
    vic.process_end(step_name, event_id, status, duration,
                    outputs=outputs, notes=notes)


def _sam3_drive(port, input_dir, export_dir, categories, batch_size):
    """Background driver: wait for sub-app health, configure, process, poll.

    Runs after _run_step5 launches the sub-app. Pushes progress into
    `sam3_status` which the frontend polls via /api/step/5/sam3_status.
    """
    try:
        _sam3_set(phase="waiting_for_sub_app", running=True, error=None,
                  message="Waiting for SAM3 sub-app (loading models — can take minutes)...")
        t0 = time.time()
        healthy = False
        while time.time() - t0 < 600:  # 10 min ceiling for model loading
            try:
                with urlopen(f"http://localhost:{port}/api/status", timeout=3) as r:
                    if r.status == 200:
                        healthy = True
                        break
            except Exception:
                pass
            time.sleep(2)
        if not healthy:
            _sam3_set(phase="error", running=False,
                      error="SAM3 sub-app did not become healthy within 10 minutes")
            return

        _sam3_set(phase="configuring", message="Loading sam_click_prompts and building queue...")
        body = json.dumps({
            "input_dir": input_dir,
            "export_dir": export_dir,
            "categories": categories,
            "review_batch_size": batch_size,
        }).encode()
        req = Request(f"http://localhost:{port}/api/configure", data=body,
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=120) as r:
            cfg_data = json.loads(r.read())
        if cfg_data.get("error"):
            _sam3_set(phase="error", running=False,
                      error=f"configure failed: {cfg_data['error']}")
            return

        total = int(cfg_data.get("to_process") or 0)
        already = int(cfg_data.get("already_processed") or 0)
        _sam3_set(total=total,
                  message=f"{total} to segment, {already} already done. Starting...")

        # If everything is already done, skip the process call.
        if cfg_data.get("phase") == "review" or total == 0:
            _sam3_set(phase="review_ready", running=False,
                      message=f"Already segmented. Ready for review ({cfg_data.get('review_count', 0)} frames).")
            return

        # Kick off background processing on the sub-app.
        _sam3_set(phase="segmenting", message="SAM3 running...")
        req = Request(f"http://localhost:{port}/api/process", data=b"{}",
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=15):
            pass

        # Poll progress until done.
        while True:
            try:
                with urlopen(f"http://localhost:{port}/api/process_status", timeout=10) as r:
                    d = json.loads(r.read())
                _sam3_set(
                    processed=int(d.get("processed") or 0),
                    total=int(d.get("total") or 0),
                    running=bool(d.get("running", False)),
                )
                if d.get("done") or (not d.get("running") and d.get("processed", 0) >= d.get("total", 0) > 0):
                    _sam3_set(phase="review_ready", running=False,
                              message="Segmentation complete. Ready for review.")
                    return
            except Exception:
                # Transient errors are fine — the sub-app may be busy with GPU work.
                pass
            time.sleep(3)
    except Exception as e:
        _sam3_set(phase="error", running=False, error=str(e))


def _expertids_paths():
    """Effective paths for the native Add-Expert-IDs blueprint: the active
    project's step-5 export dir + the pipeline's review/library config. Read live
    each request so it tracks the currently-loaded project."""
    # Prefer the 4.test combined-annotator export (step4test replaces Steps 4+5)
    # when that project has a segmentations/ tree; else fall back to the legacy
    # Step-5 dir so older Step-5-annotated projects still round-trip.
    export_dir = ""
    if current_project is not None:
        d4 = _step4test_export_dir()
        # Prefer 4.test only when it actually HOLDS segmentation data (an empty /
        # aborted-routing step4test/segmentations/ dir must not shadow a real
        # Step-5 export). Matches importer._iter_segmentation_files' access pattern.
        if d4 and glob.glob(os.path.join(d4, "segmentations", "*", "segmentations.json")):
            export_dir = d4
        else:
            export_dir = current_project["steps"].get("5", {}).get("dir", "") or ""
    # Projects root for multi-project CSV routing (resolve project_id -> export
    # dir). The blueprint falls back to importer.default_projects_root() if
    # absent, but provide it explicitly + the site-codes table for full names.
    _projects_root = os.path.join(os.path.dirname(REPO_DIR), "inprocess")
    _site_codes = os.path.join(REPO_DIR, "supporting_data", "site_codes.csv")
    return {
        "export_dir": export_dir,
        "review_dir": PATHS.get("review_dir", ""),
        "library_dir": PATHS.get("expert_library_dir", "") or "",
        "review_repo_url": PATHS.get("review_repo_url", ""),
        "master_codes": PATHS.get("master_codes_csv", ""),
        "overlap_thresh": float(PATHS.get("review_overlap_thresh", 0.5) or 0.5),
        "git_push": os.environ.get("TCRMP_REVIEW_GIT_PUSH", "1") == "1",
        "projects_root": _projects_root if os.path.isdir(_projects_root) else "",
        "site_codes": _site_codes if os.path.isfile(_site_codes) else "",
    }


def _matrix_paths():
    """Effective paths for the native Label Coverage Matrix blueprint. The
    builder rescans this `inprocess` projects root on every /matrix/api/data
    call, so a purged project simply stops contributing. Same inprocess root
    that _expertids_paths() computes for its projects_root."""
    return {
        "inprocess_root": os.path.join(os.path.dirname(REPO_DIR), "inprocess"),
    }


def _labels_paths():
    """Effective paths for the native Label Manager blueprint. The canonical
    species vocabulary + all-points CSV come straight from the orchestrator's
    resolved pipeline.yaml PATHS (so they track config), with the recode output
    dir and the supporting_data duplicate mirror anchored to the repo."""
    return {
        "master_codes_csv": PATHS.get("master_codes_csv", ""),
        "duplicate_master_codes": PATHS.get("duplicate_master_codes", ""),
        "all_points_csv": PATHS.get("all_points_csv", ""),
        "recode_output_dir": os.path.join(REPO_DIR, "scripts", "TCRMPcvr_recodeSpecies", "output"),
        "pipeline_yaml": CONFIG_PATH,
    }


# Minimal standalone page served on every non-API path while the module lock
# is active (see lock_status.py). Inline CSS + inline SVG only: while locked,
# nothing but /static is served, so the page must carry everything itself.
_LOCK_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reef Point Seg (locked)</title>
<style>
  html, body { margin: 0; height: 100%; }
  body { background: #10151c; color: #dfe7ef;
         font-family: system-ui, -apple-system, sans-serif;
         display: flex; align-items: center; justify-content: center; }
  .vic-lock-page { text-align: center; padding: 40px; max-width: 520px; }
  .vic-lock-page svg { display: block; margin: 0 auto 18px; }
  .vic-lock-page h1 { font-size: 22px; font-weight: 600; margin: 0 0 12px; }
  .vic-lock-page p { font-size: 15px; line-height: 1.6; margin: 0; color: #b8c4d0; }
  .vic-lock-page a { color: #7fb3e8; }
  .vic-lock-footer { margin-top: 28px; font-size: 12px; color: #5a6673; }
</style>
</head>
<body>
<div class="vic-lock-page">
  <svg xmlns="http://www.w3.org/2000/svg" width="56" height="56"
       viewBox="0 0 24 24" fill="none" stroke="#8fa4b8" stroke-width="1.6"
       stroke-linecap="round" stroke-linejoin="round" role="img"
       aria-label="Locked">
    <rect x="4" y="10.5" width="16" height="10" rx="2"></rect>
    <path d="M8 10.5V7a4 4 0 0 1 8 0v3.5"></path>
    <circle cx="12" cy="15.5" r="1.4" fill="#8fa4b8" stroke="none"></circle>
  </svg>
  <h1>Reef Point Seg</h1>
  <p>Under development. Please come back later.
     <a href="mailto:lauren.olinger@uvi.edu">Email Lauren</a> for questions.</p>
  <div class="vic-lock-footer">VICARIUS module lock</div>
</div>
</body>
</html>"""


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )

    # ── Module-lock gate (VICARIUS platform contract) ──────────────────
    # Registered before the blueprints mount so every route in the app,
    # blueprint or not, is refused while the platform registry marks this
    # module locked (lock_status.is_locked(), read fresh per request).
    # Because before_request runs ahead of any view, index()'s ?project_dir=
    # auto-open can never execute while locked. /static stays served so the
    # lock page itself can render assets.
    @app.before_request
    def _module_lock_gate():
        if not lock_status.is_locked():
            return None
        if request.path.startswith("/static/"):
            return None
        if request.path == "/" and request.method == "GET":
            return render_template_string(_LOCK_PAGE_HTML)
        if request.path.startswith(("/api/", "/labels", "/expertids", "/matrix")):
            return (
                jsonify({"error": "locked", "message": lock_status.LOCK_MESSAGE}),
                423,
            )
        # Safe default: anything unrecognized gets the lock page too.
        return render_template_string(_LOCK_PAGE_HTML)

    # Add Expert IDs — mounted natively (its routes live under /expertids, its UI
    # is included into the step-6 panel). No subprocess, no iframe.
    app.register_blueprint(
        _make_expertids_bp(_expertids_paths,
                           log_fn=lambda m: vic.note(f"[expertids] {m}")),
        url_prefix="/expertids",
    )

    # Label Coverage Matrix — mounted natively as its own full page under
    # /matrix (cross-project image x label grid, live from inprocess/).
    app.register_blueprint(
        _make_matrix_bp(_matrix_paths,
                        log_fn=lambda m: vic.note(f"[matrix] {m}")),
        url_prefix="/matrix",
    )

    # Label Manager — mounted natively as its own full page under /labels
    # (canonical species vocabulary editor + remap composer + manual recode).
    app.register_blueprint(
        _make_labels_bp(_labels_paths,
                        log_fn=lambda m: vic.note(f"[labels] {m}")),
        url_prefix="/labels",
    )

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    # Path prefix that any auto-open project_dir must live under. This
    # prevents a crafted ?project_dir= URL param from pointing at arbitrary
    # filesystem locations (e.g. /etc/passwd). Derived from the configured
    # projects_dir (pipeline.yaml paths.projects_dir), falling back to the
    # module-standard inprocess/ sibling of the repo.
    _AUTOOPEN_ALLOWED_ROOT = (
        PATHS.get("projects_dir")
        or os.path.join(os.path.dirname(REPO_DIR), "inprocess")
    ).rstrip("/") + "/"

    @app.route("/")
    def index():
        global current_project
        from orchestrator_config import PATHS

        # Optional auto-open: /?project_dir=<abs-path>[&purpose=<text>]
        raw = request.args.get("project_dir", "")
        if raw:
            try:
                candidate = os.path.abspath(unquote(raw))
                # Guardrails: must live under the module's inprocess/ tree
                # and contain a project.json to be considered valid.
                if (
                    candidate.startswith(_AUTOOPEN_ALLOWED_ROOT)
                    and os.path.isfile(os.path.join(candidate, "project.json"))
                ):
                    # Idempotent: don't reload if the same project is already open.
                    already = (
                        current_project is not None
                        and current_project.get("project_dir") == candidate
                    )
                    if not already:
                        current_project = pm.load_project(candidate)
                        vic.note(
                            f"[orch] project '{current_project['name']}' "
                            f"opened via URL param"
                        )
                        raw_purpose = request.args.get("purpose", "")
                        if raw_purpose:
                            purpose = unquote(raw_purpose).strip()
                            if purpose:
                                vic.note(f"[orch] auto-open purpose: {purpose}")
            except Exception:
                # Never block page render on auto-open failure.
                pass

        # When a project is already loaded (either via the ?project_dir= auto-open
        # above, or via a prior POST /api/project/open from the VICARIUS UI), tell
        # the template to skip the landing/intro screen and render the project view
        # directly. The intro block is still emitted but pre-hidden so the
        # "Switch Project" button can re-reveal it without a page reload.
        return render_template(
            "index.html",
            repo_root=REPO_DIR,
            projects_dir=PATHS.get("projects_dir", os.path.join(REPO_DIR, "projects")),
            supporting_data_dir=PATHS.get("supporting_data_dir", ""),
            all_points_csv=PATHS.get("all_points_csv", ""),
            master_codes_csv=PATHS.get("master_codes_csv", ""),
            project_loaded=(current_project is not None),
            project_state=current_project,
        )

    # ------------------------------------------------------------------
    # Project endpoints
    # ------------------------------------------------------------------
    # NOTE: there is deliberately no /api/project/create route. Projects are
    # minted as run_* dirs by the VICARIUS launcher (vicarius_ui_os
    # module_views.api_module_new_run) and arrive here via /api/project/open
    # or the guarded GET /?project_dir=... auto-open. The old in-orchestrator
    # create path (removed 2026-08-14) used an incompatible YY-MM-DD_{name}
    # naming scheme and had no remaining caller.
    @app.route("/api/project/open", methods=["POST"])
    def project_open():
        global current_project
        data = request.get_json(force=True)
        project_dir = data.get("project_dir", "").strip()
        if not project_dir:
            return jsonify({"error": "Project directory is required"}), 400
        try:
            switching = (
                current_project is not None
                and os.path.abspath(current_project.get("project_dir", ""))
                != os.path.abspath(project_dir)
            )
            current_project = pm.load_project(project_dir)
            if switching:
                # A reopened SAME project keeps its live annotator state; a
                # DIFFERENT project must not inherit the old one's ui_ready.
                _reset_transient_status("Project switched.")
            vic.note(f"[orch] project '{current_project['name']}' reopened")
            return jsonify({"success": True, "state": current_project})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/state")
    def project_state():
        if current_project is None:
            return jsonify({"loaded": False})
        return jsonify({"loaded": True, "state": current_project})

    @app.route("/api/project/config/<int:step>", methods=["PUT"])
    def project_config(step):
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        data = request.get_json(force=True)
        s = str(step)
        if s not in current_project["steps"]:
            return jsonify({"error": f"Invalid step {step}"}), 400
        current_project["steps"][s]["config"].update(data)
        pm.save_project(current_project)
        return jsonify({"success": True})

    @app.route("/api/project/quit", methods=["POST"])
    def project_quit():
        global current_project
        runner.kill_all()
        _reset_transient_status("Project closed.")
        if current_project:
            vic.note(f"[orch] project '{current_project['name']}' closed")
            pm.save_project(current_project)
        current_project = None
        return jsonify({"success": True})

    # ------------------------------------------------------------------
    # Step execution endpoints
    # ------------------------------------------------------------------
    @app.route("/api/step/<int:step>/run", methods=["POST"])
    def step_run(step):
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400

        s = str(step)
        st = current_project["steps"].get(s)
        if not st:
            return jsonify({"error": f"Invalid step {step}"}), 400
        if st["status"] == "locked":
            return jsonify({"error": f"Step {step} is locked"}), 400

        cfg = st["config"]
        step_dir = st["dir"]
        os.makedirs(step_dir, exist_ok=True)

        st["status"] = "running"
        st["started_at"] = datetime.now().isoformat()

        # VICARIUS process_start — stash event_id so step completion can link to it.
        step_name = f"step{step}_{pm.STEP_DIRS.get(s, '').split('_', 1)[-1] or 'stage'}"
        purpose = cfg.get("purpose") or f"{st.get('name', 'step')}" + (
            f" (project '{current_project.get('name', '')}')"
        )
        event_id = vic.process_start(step_name, purpose,
                                     inputs=[str(v) for v in cfg.values() if isinstance(v, str)],
                                     notes="")
        if event_id is not None:
            st["_vic_event_id"] = event_id
        pm.save_project(current_project)

        try:
            if step == 1:
                result = _run_step1(step, cfg, step_dir)
            elif step == 2:
                result = _run_step2(step, cfg, step_dir)
            elif step == 3:
                result = _run_step3(step, cfg, step_dir)
            elif step == 4:
                result = _run_step4(step, cfg, step_dir)
            elif step == 5:
                result = _run_step5(step, cfg, step_dir)
            elif step == 6:
                result = _run_step6(step, cfg, step_dir)
            elif step == 7:
                result = _run_step7(step, cfg, step_dir)
            elif step == 8:
                result = _run_step8(step, cfg, step_dir)
            else:
                return jsonify({"error": "Invalid step"}), 400

            if "error" in result:
                st["status"] = "error"
                _vic_end_step(step, "failed", notes=str(result.get("error", ""))[:500])
                pm.save_project(current_project)
                return jsonify(result), 500

            return jsonify({"success": True, **result})
        except Exception as e:
            st["status"] = "error"
            _vic_end_step(step, "failed", notes=str(e)[:500])
            pm.save_project(current_project)
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    # Add Expert IDs is mounted natively via the _expertids blueprint registered
    # above (routes under /expertids, UI included into the step-6 panel). The old
    # subprocess-launch + status endpoints were retired in favor of that.

    def _run_step1(step, cfg, step_dir):
        py = PYTHON_PATHS.get(1, "python3")
        if not os.path.isfile(py):
            py = "python3"
        cmd = [py, ENTRY_POINTS[1], cfg.get("input_dir", ""), step_dir]
        return runner.run_cli_stage(step, cmd, cwd=WORKING_DIRS[1])

    def _run_step2(step, cfg, step_dir):
        port = find_free_port(cfg.get("port", 5055))
        py = PYTHON_PATHS.get(2, "python3")
        if not os.path.isfile(py):
            py = "python3"
        ap = cfg.get("all_points", "")
        mc = cfg.get("master_codes", "")
        cmd = [py, ENTRY_POINTS[2], "--port", str(port), "--all-points", ap, "--master-codes", mc]
        result = runner.run_flask_stage(step, cmd, port, cwd=WORKING_DIRS[2])
        current_project["steps"]["2"]["config"]["port"] = port
        pm.save_project(current_project)
        return result

    def _run_step3(step, cfg, step_dir):
        py = PYTHON_PATHS.get(3, "python3")
        if not os.path.isfile(py):
            py = "python3"
        species = [s.strip() for s in cfg.get("target_species", "").split(",") if s.strip()]
        cmd = [
            py, ENTRY_POINTS[3],
            "--species", *species,
            "--target", str(cfg.get("target_instances", 1000)),
            "--all-points", cfg.get("all_points", ""),
            "--master-codes", cfg.get("master_codes", ""),
        ]
        # Year bounds now flow from the UI form through to the CLI.
        if cfg.get("min_year") is not None:
            cmd += ["--min-year", str(cfg.get("min_year"))]
        if cfg.get("max_year") is not None:
            cmd += ["--max-year", str(cfg.get("max_year"))]

        env_extra = {"TCRMP_OUTPUT_DIR": step_dir}
        return runner.run_cli_stage(step, cmd, cwd=WORKING_DIRS[3], env_extra=env_extra)

    def _get_target_species_str():
        """Get comma-separated target species from step 3 config.

        No hardcoded fallback: target labels are chosen by the user in the
        Step 3 picker (nothing is selected by default). The UI blocks running
        Step 3 with an empty selection, so an empty string here means the user
        ran without picking labels rather than a default set to silently inject.
        """
        s3 = current_project["steps"].get("3", {}).get("config", {})
        return s3.get("target_species", "")

    def _get_remap_log_path():
        """Find the remap_log.json from step 2 output."""
        s2_dir = current_project["steps"].get("2", {}).get("dir", "")
        remap = os.path.join(s2_dir, "remap_log.json")
        if os.path.isfile(remap):
            return remap
        # Fall back to the source remap log configured in step 2
        src = current_project["steps"].get("2", {}).get("config", {}).get("remap_log_source", "")
        if src and os.path.isfile(src):
            return src
        return ""

    def _resolve_path(step_num, config_key, output_key=None):
        """Resolve a path from step config, falling back to step outputs."""
        s = str(step_num)
        step = current_project["steps"].get(s, {})
        val = step.get("config", {}).get(config_key, "")
        if not val and output_key:
            val = step.get("outputs", {}).get(output_key, "")
        return val or ""

    def _run_step4(step, cfg, step_dir):
        port = find_free_port(cfg.get("port", 5065))
        py = PYTHON_PATHS.get(4, "python3")
        if not os.path.isfile(py):
            py = "python3"
        cmd = [py, ENTRY_POINTS[4], "--port", str(port)]

        # Resolve selected_frames from step 3 (config or outputs)
        selected_frames = (
            cfg.get("selected_frames", "")
            or _resolve_path(3, "selected_frames", "selected_frames")
        )
        # Resolve all_points from step 2 (config or outputs)
        all_points = (
            cfg.get("all_points", "")
            or _resolve_path(2, "all_points", "all_points_recoded")
        )

        # Orchestrator URL the sub-app can call back to (e.g. to head-start SAM3).
        # We're inside a Flask request context so `request.host` is safe.
        host_port = request.host.split(":")[-1] if ":" in request.host else "80"
        orchestrator_url = f"http://localhost:{host_port}"

        # Pass ALL project paths so the sub-app reflects current settings
        env_extra = {
            "TCRMP_SELECTED_FRAMES": selected_frames,
            "TCRMP_EXPORT_DIR": step_dir,
            "TCRMP_CLIP_DIR": cfg.get("clip_dir", ""),
            "TCRMP_CPC_DIR": PATHS.get("cpc_all_dir", "") or os.path.join(REPO_DIR, "supporting_data", "cpc_all"),
            "TCRMP_TARGET_SPECIES": _get_target_species_str(),
            "TCRMP_REMAP_LOG": _get_remap_log_path(),
            "TCRMP_ALL_POINTS": all_points,
            "TCRMP_ORCHESTRATOR_URL": orchestrator_url,
            "TCRMP_AUTO_START_SAM3": "1" if cfg.get("auto_start_sam3") else "",
            # These used to be user-selectable on the sub-app startup form.
            # Orchestrator-driven auto-boot bypasses that form, so we relay the
            # values here for the sub-app to pre-populate before calling configure.
            "TCRMP_REFERENCE_MODE": "1" if cfg.get("reference_mode") else "",
            "TCRMP_SHUFFLE": "1" if cfg.get("shuffle") else "",
            "TCRMP_REVIEW_BATCH_SIZE": str(cfg.get("review_batch_size") or "10"),
        }

        # Remove empty values so sub-app uses its own defaults
        env_extra = {k: v for k, v in env_extra.items() if v}

        result = runner.run_flask_stage(step, cmd, port, cwd=WORKING_DIRS[4], env_extra=env_extra)
        current_project["steps"]["4"]["config"]["port"] = port
        pm.save_project(current_project)
        return result

    def _run_step5(step, cfg, step_dir):
        global sam3_thread
        port = find_free_port(cfg.get("port", 5070))
        py = PYTHON_PATHS.get(5, "python3")
        if not os.path.isfile(py):
            py = "python3"
        cmd = [py, ENTRY_POINTS[5], "--port", str(port)]

        # Resolve input_dir: step 4's output dir
        input_dir = cfg.get("input_dir", "")
        if not input_dir:
            s4_dir = current_project["steps"].get("4", {}).get("dir", "")
            if s4_dir:
                input_dir = s4_dir

        # Pass ALL project paths + SAM3 settings so sub-app reflects current config
        env_extra = {
            "TCRMP_INPUT_DIR": input_dir,
            "TCRMP_EXPORT_DIR": step_dir,
            # Canonical project identity (project.json id + human name), so review
            # items + library records are tagged by project instead of being
            # path-derived. Used by the multi-project review site + CSV routing.
            "TCRMP_PROJECT_ID": current_project.get("id", "") if current_project else "",
            "TCRMP_PROJECT_NAME": current_project.get("name", "") if current_project else "",
            "TCRMP_TARGET_SPECIES": _get_target_species_str(),
            "TCRMP_SAM3_DEVICE_TRACKER": cfg.get("sam3_device_tracker", ""),
            "TCRMP_SAM3_DEVICE_EXEMPLAR": cfg.get("sam3_device_exemplar", ""),
            "TCRMP_CONFIDENCE_THRESHOLD": str(cfg.get("confidence_threshold", "")),
            "TCRMP_MIN_MASK_AREA_PX": str(cfg.get("min_mask_area_px", "")),
            "TCRMP_MERGE_DISTANCE_PX": str(cfg.get("merge_distance_px", "")),
            "TCRMP_OVERLAP_STRATEGY": cfg.get("overlap_strategy", ""),
            # Expert-review export targets (REVIEW-flagged masks).
            "TCRMP_REVIEW_DIR": PATHS.get("review_dir", ""),
            "TCRMP_REVIEW_REPO_URL": PATHS.get("review_repo_url", ""),
            "TCRMP_EXPERT_LIBRARY_DIR": PATHS.get("expert_library_dir", ""),
            "TCRMP_REVIEW_CONTACTS": cfg.get("review_contacts", ""),
        }

        # Remove empty values so sub-app uses its own defaults
        env_extra = {k: v for k, v in env_extra.items() if v}

        # Recoded species dictionary (Step 2 output) so RECODED names (e.g.
        # Orbicella, not the obsolete Montastraea annularis) carry through into
        # the step-5 labels + the expert-review codes.json + the library. Without
        # this, step 5 falls back to the canonical pre-recode supporting_data
        # dictionary and leaks un-recoded names to reviewers. Falls back to the
        # canonical dict only when Step 2 hasn't produced a recoded one.
        _s2 = current_project["steps"].get("2", {}) if current_project else {}
        _mc = (_s2.get("outputs", {}).get("master_codes_recoded")
               or _s2.get("outputs", {}).get("master_codes"))
        if not (_mc and os.path.isfile(_mc)):
            _s2dir = _s2.get("dir", "")
            for _cand in ("master_codes_recoded.csv", "master_codes.csv"):
                _p = os.path.join(_s2dir, _cand) if _s2dir else ""
                if _p and os.path.isfile(_p):
                    _mc = _p
                    break
        if _mc and os.path.isfile(_mc):
            env_extra["TCRMP_MASTER_CODES"] = _mc

        result = runner.run_flask_stage(step, cmd, port, cwd=WORKING_DIRS[5], env_extra=env_extra)
        current_project["steps"]["5"]["config"]["port"] = port
        pm.save_project(current_project)

        # Kick off the orchestrator-side driver so SAM3 configures and starts
        # segmentation without the user touching the sub-app UI.
        categories = cfg.get("categories") or ["Target species only"]
        # Review batch size is user-selectable in the Step 5 panel; "all" is a
        # sentinel meaning "no batching" (the sub-app slices files[offset:+N],
        # so a huge N effectively returns everything).
        raw_batch = cfg.get("review_batch_size", "10")
        if isinstance(raw_batch, str) and raw_batch.strip().lower() == "all":
            batch_size = 999999
        else:
            try:
                batch_size = int(raw_batch)
            except (TypeError, ValueError):
                batch_size = 10
        _sam3_set(phase="launching", processed=0, total=0, running=True, error=None,
                  message="Launching SAM3 sub-app...")
        sam3_thread = threading.Thread(
            target=_sam3_drive,
            args=(port, input_dir, step_dir, categories, batch_size),
            daemon=True,
        )
        sam3_thread.start()
        return result

    def _routed_dir():
        """The 4.test-owned dir that placePoints routes detections into and the
        combined annotator then consumes (sam_click_prompts.json lives here)."""
        return os.path.join(_step4test_export_dir(), "routed_input")

    def _routed_prompts_exist(routed_dir):
        """True when a prior routing pass already wrote the flat prompts file."""
        return os.path.isfile(os.path.join(routed_dir, "ids", "sam_click_prompts.json"))

    def _route_drive(pp_port, selected_frames, routed_dir, species_filter, project_ctx):
        """Background driver: drive placePoints headlessly to produce
        sam_click_prompts.json, then launch the combined annotator.

        Clone of _sam3_drive but for the routing (CPC/OCR detection) pass:
        wait for health, configure, loop process until done, export_all, kill
        the placePoints stage (frees the GPU), then open the combined UI. Pushes
        progress into `route_status` which the frontend polls via
        /api/step/step4test/route_status.
        """
        def _abort_if_cancelled():
            """True (and cleans up) when the operator cancelled this pass.

            Checked between every step of the route. Killing placePoints frees the
            GPU immediately; the detections it already flushed stay on disk, so the
            next Open resumes from there instead of re-OCRing everything."""
            if not route_cancel.is_set():
                return False
            runner.kill("step4test_route")
            _route_set(phase="cancelled", ui_port=None, error=None,
                       message="Routing cancelled. Reopening resumes where it stopped.")
            return True

        try:
            _route_set(message="Waiting for routing sub-app (placePoints)...")
            t0 = time.time()
            healthy = False
            while time.time() - t0 < 600:  # 10 min ceiling
                if _abort_if_cancelled():
                    return
                try:
                    with urlopen(f"http://localhost:{pp_port}/api/status", timeout=3) as r:
                        if r.status == 200:
                            healthy = True
                            break
                except Exception:
                    pass
                time.sleep(2)
            if _abort_if_cancelled():
                return
            if not healthy:
                _route_set(phase="error",
                           error="Routing sub-app did not become healthy within 10 minutes")
                return

            _route_set(phase="routing_configure",
                       message="Loading selected frames and splitting CPC/OCR...")
            body = json.dumps({
                "selected_frames": selected_frames,
                "export_dir": routed_dir,
                "species_filter": species_filter or [],
                "review_batch_size": 999999,
                "reference_mode": False,
                "shuffle": False,
            }).encode()
            req = Request(f"http://localhost:{pp_port}/api/configure", data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=300) as r:
                cfg_data = json.loads(r.read())
            if cfg_data.get("error"):
                _route_set(phase="error", error=f"configure failed: {cfg_data['error']}")
                return

            # Routing accountability (Task 2): placePoints' configure response
            # reports per-frame drops (dropped[]) plus cpc_missing/image_missing
            # counts. Surface them as human warnings + the raw dropped list so
            # the 4.test overlay/panel can show "N of M frames skipped" without
            # auto-recovery (descoped). Absent fields default to empty/0 so this
            # stays resilient if the placePoints side hasn't landed the report yet.
            dropped = cfg_data.get("dropped") or []
            cpc_missing = int(cfg_data.get("cpc_missing") or 0)
            image_missing = int(cfg_data.get("image_missing") or 0)
            warnings = []
            n_dropped = len(dropped) if isinstance(dropped, list) else 0
            if n_dropped or cpc_missing or image_missing:
                bits = []
                if cpc_missing:
                    bits.append(f"{cpc_missing} missing CPC points")
                if image_missing:
                    bits.append(f"{image_missing} missing images")
                detail = f" ({', '.join(bits)})" if bits else ""
                warnings.append(
                    f"{n_dropped or (cpc_missing + image_missing)} frame(s) "
                    f"skipped during routing{detail}; see routing_report.json."
                )
            _route_set(warnings=warnings,
                       dropped=dropped if isinstance(dropped, list) else [])

            total = int(cfg_data.get("ocr_remaining") or 0)
            _route_set(phase="routing_ocr", processed=0, total=total,
                       message=f"Routing chosen images... 0/{total}")

            # Loop process (one OCR frame per call) until done. CPC-only frames
            # need no OCR, so total may be 0 and this loop exits immediately.
            while True:
                # The long leg of the pass (minutes to tens of minutes), so this is
                # where a cancel almost always lands.
                if _abort_if_cancelled():
                    return
                req = Request(f"http://localhost:{pp_port}/api/process", data=b"{}",
                              headers={"Content-Type": "application/json"}, method="POST")
                try:
                    with urlopen(req, timeout=120) as r:
                        d = json.loads(r.read())
                except Exception:
                    # Transient error (GPU busy) - retry after a beat.
                    time.sleep(2)
                    continue
                if d.get("error"):
                    _route_set(phase="error", error=f"process failed: {d['error']}")
                    return
                proc = int(d.get("processed") or 0)
                tot = int(d.get("total") or total)
                _route_set(processed=proc, total=tot,
                           message=f"Routing chosen images... {proc}/{tot}")
                if d.get("done"):
                    break

            if _abort_if_cancelled():
                return

            _route_set(phase="exporting",
                       message="Writing sam_click_prompts.json for all routed frames...")
            req = Request(f"http://localhost:{pp_port}/api/export_all", data=b"{}",
                          headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=600) as r:
                exp = json.loads(r.read())
            if exp.get("error"):
                _route_set(phase="error", error=f"export_all failed: {exp['error']}")
                return

            # Flatten the per-year routed prompts into the flat layout the
            # combined annotator now reads.
            fstats = _flatten_routed_input(routed_dir)
            _write_routed_lores_marker(
                routed_dir, bool((step4test_cfg or {}).get("lores_mode", True)))
            _route_set(message=(f"Flattened routed input: {fstats['merged']} frames, "
                                f"{fstats['year_dirs_removed']} year folder(s) collapsed."))

            # Free the GPU before the combined annotator (SAM3) opens - routing
            # and segmentation must never contend for the device.
            runner.kill("step4test_route")

            # Last gate before the launch. The operator closed the launch window,
            # so do NOT spring an annotator on them now - and above all do not run
            # _launch_step4test_ui, which kills every running annotator first. That
            # is exactly how an abandoned pass killed a live annotator out from
            # under the UI (2026-07-14). The routed prompts are written and cached,
            # so the next Open is instant.
            if route_cancel.is_set():
                _route_set(phase="cancelled", ui_port=None, error=None,
                           message="Routing finished, but the launch window was closed. "
                                   "Click Open to start the annotator.")
                return

            # This pass routed project_ctx's frames into project_ctx's
            # routed_input, but a route can outlive the project that started it
            # (the operator can open another project mid-route). _launch_step4test_ui
            # reads the GLOBAL current_project, so launching now would bind the
            # annotator to a DIFFERENT project's routed_input - one this pass never
            # wrote, so empty or half-written. Stop here and let the operator start
            # a route for the project they actually have open.
            cur_id = (current_project or {}).get("id")
            ctx_id = (project_ctx or {}).get("id")
            if cur_id != ctx_id:
                _route_set(phase="error",
                           error=(f"Project changed during routing ({ctx_id} -> "
                                  f"{cur_id}). The routed prompts for {ctx_id} are "
                                  f"written, but no annotator was launched. Reopen "
                                  f"that project, or click Open to route {cur_id}."))
                return

            _route_set(phase="route_ready",
                       message="Routing complete. Launching combined annotator...")

            # Open the combined window only after routing + placePoints kill.
            ui_result = _launch_step4test_ui()
            if isinstance(ui_result, dict) and ui_result.get("error"):
                _route_set(phase="error", error=ui_result["error"])
                return
        except Exception as e:
            _route_set(phase="error", error=str(e))

    def _annotator_reusable(routed_dir):
        """True if a combined annotator is already running, healthy, and bound to
        the CURRENT project + routed_dir, so we can open its existing window
        without a kill + relaunch.

        Reuse is only safe when the running process is the one launched for this
        project and this input dir: the annotator freezes its export_dir at first
        /api/configure and never rebinds, so a process launched for a different
        project would misroute every export. Any mismatch (or an unhealthy /
        vanished process) returns False, so the caller launches fresh."""
        if not step4test_state.get("running"):
            return False
        port = runner.ports.get("step4test") or step4test_state.get("port")
        if not port:
            return False
        pid = current_project.get("id") if current_project else None
        if step4test_state.get("project_id") != pid:
            return False
        if step4test_state.get("input_dir") != routed_dir:
            return False
        return runner.health_check(port)

    def _run_step4test(force_reroute=False):
        """Two-phase launch of the combined annotator (TCRMPclip_combinedAnnotate)
        as a full replacement for Step 4.

        Phase 1 (routing): drive placePoints headlessly over Step 3's chosen
        frames to produce sam_click_prompts.json in routed_input/, streaming
        progress via route_status. Phase 2: launch the combined annotator pointed
        at routed_input/ (done by _route_drive's tail, or directly on reuse).

        If routed_input/ already holds prompts and not force_reroute, skip
        routing and launch the UI instantly.
        """
        global route_thread

        # A routing pass already in flight OWNS the annotator launch: its tail
        # (_route_drive) kills every annotator and relaunches one when it
        # finishes. Starting a second launch path here races it - the older
        # thread would kill the annotator this call just launched, orphaning the
        # UI's wait on a port that never comes back up. Attach the UI to the
        # running pass instead, so it polls route_status and opens when that
        # pass lands. (Closing the launch popup abandons the frontend poll but
        # does NOT stop the thread, so an in-flight pass can outlive its click.)
        if route_thread is not None and route_thread.is_alive():
            with route_lock:
                phase = route_status.get("phase") or "routing_ocr"
            return {"phase": phase, "in_flight": True}

        routed_dir = _routed_dir()
        os.makedirs(routed_dir, exist_ok=True)

        # Resolve Step 3's selected frames (config or outputs).
        selected_frames = _resolve_path(3, "selected_frames", "selected_frames")
        if not selected_frames or not os.path.isfile(selected_frames):
            return {"error": "Run Step 3 (Choose Images) first - no selected_frames found."}

        # Reuse only a FRESH cache: if the Step 3 selection changed since the
        # last route, the cached prompts are stale and would leak the prior
        # run's frames into segmentation. Clear them and re-route. Otherwise
        # open instantly.
        want_lores = bool((step4test_cfg or {}).get("lores_mode", True))
        if not force_reroute and _routed_prompts_exist(routed_dir):
            cached_lores = _routed_lores_mode(routed_dir)
            lores_changed = cached_lores is not None and cached_lores != want_lores
            if _routed_prompts_stale(routed_dir, selected_frames) or lores_changed:
                cleared = _clear_routed_input(routed_dir)
                _reason = ("Low-res setting changed since last route; "
                           if lores_changed
                           else "Step 3 selection changed since last route; ")
                _route_set(phase="routing", processed=0, total=0, error=None,
                           message=(f"{_reason}cleared {cleared} stale route(s), "
                                    f"re-routing..."))
                # fall through to the routing path below (do NOT return)
            else:
                # Prompts are fresh. If a combined annotator is ALREADY running,
                # healthy, and bound to THIS project + routed_dir, reuse it as-is
                # instead of killing + relaunching. A relaunch reloads SAM3 (~1
                # min) and drifts the annotator's port, which pushes the browser's
                # window.open() past its user-activation window so the popup gets
                # silently blocked (the "Open does nothing" bug). Reuse keeps the
                # open instant, so the popup fires inside the click's activation.
                if _annotator_reusable(routed_dir):
                    reuse_port = runner.ports.get("step4test") or step4test_state.get("port")
                    _route_set(phase="ui_ready", ui_port=reuse_port,
                               runner_key="step4test",
                               runner_label="Combined annotator",
                               message="Combined annotator already running; reusing it.")
                    return {"phase": "ui_ready", "ui_port": reuse_port}
                _route_set(phase="route_ready", processed=0, total=0, error=None,
                           message="Reusing routed images. Launching combined annotator...")
                return _launch_step4test_ui()

        # Build the Step-3 routing env exactly like _run_step4, but exporting
        # into routed_dir (placePoints writes prompts under {routed_dir}/{year}/ids/).
        s4cfg = current_project["steps"].get("4", {}).get("config", {})
        all_points = _resolve_path(2, "all_points", "all_points_recoded")
        # Routing clip source: the 4.test panel's clip-dir wins, else the (now
        # archived) Step-4 saved clip_dir, else the LOCAL full-set clip dir.
        # A stale/missing saved path (e.g. an old Dropbox path) is skipped, so
        # routing always reads from the same supporting_data tree as Step 3.
        _clip_dir = _resolve_routing_clip_dir(step4test_cfg, s4cfg)
        routing_env = {
            "TCRMP_SELECTED_FRAMES": selected_frames,
            "TCRMP_EXPORT_DIR": routed_dir,
            "TCRMP_CLIP_DIR": _clip_dir,
            # Low-res delivery (step-4 panel "Low res only", default on): route
            # every >1920px frame from its lores twin with points scaled to match.
            "TCRMP_LORES_MODE": "1" if (step4test_cfg or {}).get("lores_mode", True) else "",
            "TCRMP_CPC_DIR": PATHS.get("cpc_all_dir", "") or os.path.join(REPO_DIR, "supporting_data", "cpc_all"),
            "TCRMP_TARGET_SPECIES": _get_target_species_str(),
            "TCRMP_REMAP_LOG": _get_remap_log_path(),
            "TCRMP_ALL_POINTS": all_points,
        }
        routing_env = {k: v for k, v in routing_env.items() if v}

        # Launch placePoints headless on a free port and drive it.
        py = PYTHON_PATHS.get(4, "python3")
        if not os.path.isfile(py):
            py = "python3"
        pp_entry = ENTRY_POINTS[4]
        pp_dir = WORKING_DIRS[4]
        pp_port = find_free_port(5066)
        _route_set(phase="launching", processed=0, total=0, ui_port=None, error=None,
                   warnings=[], dropped=[],
                   message="Launching routing sub-app (placePoints)...")
        cmd = [py, pp_entry, "--port", str(pp_port)]

        # No live driver thread got us here (the in-flight guard above returns
        # first), so any routing stage still holding this slot is STRANDED - a
        # placePoints left by a pass whose driver died, or one inherited across an
        # orchestrator restart. It will never finish or launch anything. Reclaim the
        # slot instead of letting run_flask_stage refuse with "already running",
        # which would fail every Open click until someone killed it by hand.
        runner.kill("step4test_route")

        result = runner.run_flask_stage("step4test_route", cmd, pp_port,
                                        cwd=pp_dir, env_extra=routing_env)
        if isinstance(result, dict) and result.get("error"):
            _route_set(phase="error", error=result["error"])
            return result

        # A fresh pass starts uncancelled (a previous pass may have left the flag
        # set when its window was closed).
        route_cancel.clear()

        species_list = [s.strip() for s in _get_target_species_str().split(",") if s.strip()]
        route_thread = threading.Thread(
            target=_route_drive,
            args=(pp_port, selected_frames, routed_dir, species_list, current_project),
            daemon=True,
        )
        route_thread.start()
        return {"phase": "launching"}

    def _kill_all_annotators():
        """Kill EVERY running annotator-family app before launching a fresh one,
        so the new process binds to the CURRENT project's export dir.

        The annotator is one long-lived process whose session['export_dir'] is
        frozen at its first /api/configure and never rebinds. If a stale one
        keeps running, every export lands in ITS original project instead of the
        active one. Covers (a) the stages this orchestrator tracks (step4test,
        step4loop share the combinedAnnotate entry; editmasks is a separate
        entry script) AND (b) zombies from a prior orchestrator session (not in
        runner.processes) by matching both entry scripts and freeing the shared
        annotator port range. No lsof on this box; fuser only."""
        import subprocess as _sp
        try:
            runner.kill("step4test")
        except Exception:
            pass
        try:
            runner.kill("editmasks")
        except Exception:
            pass
        entry_matches = (
            os.path.join("TCRMPclip_combinedAnnotate", "src", "app.py"),
            os.path.join("TCRMPclip_editMasks", "src", "app.py"),
        )
        for entry_match in entry_matches:
            try:
                _sp.run(["pkill", "-f", entry_match], check=False,
                        stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, timeout=10)
            except Exception:
                pass
        for _p in range(5080, 5090):
            try:
                _sp.run(["fuser", "-k", f"{_p}/tcp"], check=False,
                        stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, timeout=10)
            except Exception:
                pass
        time.sleep(0.5)  # let the killed process release its port

    def _resolve_master_codes():
        """Recoded master codes CSV path for the active project - same
        resolution used by both annotator launch paths (step4test, step4loop)
        and the step4loop seeder call: step 2's recoded output if present on
        disk, else the plain master_codes.csv, else the on-disk fallback
        filenames under the step-2 dir. Returns '' when nothing resolves."""
        _s2 = current_project["steps"].get("2", {})
        _mc = (_s2.get("outputs", {}).get("master_codes_recoded")
               or _s2.get("outputs", {}).get("master_codes"))
        if not (_mc and os.path.isfile(_mc)):
            _s2dir = _s2.get("dir", "")
            for _cand in ("master_codes_recoded.csv", "master_codes.csv"):
                _p = os.path.join(_s2dir, _cand) if _s2dir else ""
                if _p and os.path.isfile(_p):
                    _mc = _p
                    break
        return _mc if (_mc and os.path.isfile(_mc)) else ""

    def _launch_step4test_ui(mode="step4test", cfg=None):
        """Launch the combined annotator (TCRMPclip_combinedAnnotate) pointed at
        routed_input/ (the routed-prompts dir, NOT step 4's output dir).

        Inherits step 5's SAM3 settings + recoded master codes; stays in
        MANUAL_ANNOTATE mode with provenance source step4test. The routed points
        become the read-only reference layer in the UI.

        mode="step4loop" (the Refine active-learning loop) shares every path
        below EXCEPT: the cfg source (step4loop_cfg instead of step4test_cfg),
        the runner key ("step4loop"), the state dict (step4loop_state), and two
        env overrides that put the annotator into resume mode against the
        segmentations the seeder already wrote (TCRMP_SESSION_MODE=resume,
        TCRMP_PROVENANCE_SOURCE=step4loop) instead of the fresh-configure
        MANUAL_ANNOTATE flow step4test uses.

        mode="editmasks" launches a DIFFERENT app entirely: the standalone
        TCRMPclip_editMasks tool (not combinedAnnotate), against the current
        project's step4test export. Like step4loop it boots straight into the
        review queue via /api/resume (TCRMP_SESSION_MODE=edit,
        TCRMP_PROVENANCE_SOURCE=edit) rather than fresh-configure, but there is
        no seeding pass first: it opens on whatever masks Step 4 already wrote.
        Runner key "editmasks", state dict editmasks_state, own preferred port
        5085 (editMasks' own default PORT in config.py).
        """
        runner_key = {"step4loop": "step4loop", "editmasks": "editmasks"}.get(mode, "step4test")
        state_dict = {"step4loop": step4loop_state,
                      "editmasks": editmasks_state}.get(mode, step4test_state)

        # One annotator at a time: kill any already-running annotator (any of
        # combinedAnnotate/editMasks) so this launch binds to the current
        # project's export dir (not a stale one).
        _kill_all_annotators()

        # Hardcoded entry point + cwd (config dicts are int-keyed; this tile has
        # no integer key, so we build paths from REPO_DIR directly). editmasks
        # runs a different app (TCRMPclip_editMasks) than step4test/step4loop
        # (TCRMPclip_combinedAnnotate).
        if mode == "editmasks":
            entry = os.path.join(
                REPO_DIR, "scripts", "TCRMPclip_editMasks", "src", "app.py")
            cwd = os.path.join(REPO_DIR, "scripts", "TCRMPclip_editMasks")
            preferred_port = 5085
            app_label = "edit-masks app"
        else:
            entry = os.path.join(
                REPO_DIR, "scripts", "TCRMPclip_combinedAnnotate", "src", "app.py")
            cwd = os.path.join(REPO_DIR, "scripts", "TCRMPclip_combinedAnnotate")
            preferred_port = 5080
            app_label = "combined annotator"
        py = PYTHON_PATHS.get(5, "python3")  # same unified env as step 5
        if not os.path.isfile(py):
            py = "python3"
        if not os.path.isfile(entry):
            return {"error": f"{app_label} not found at {entry}"}

        port = find_free_port(preferred_port)

        # Input = the routed-prompts dir produced by the routing pass.
        input_dir = _routed_dir()

        # Output dir = sibling of the step-5 dir, under the project dir so
        # /api/fs/open's guardrails allow opening it.
        project_dir = current_project["project_dir"]
        out_dir = _step4test_export_dir()
        os.makedirs(out_dir, exist_ok=True)

        # Inherit step 5's SAM3 settings verbatim, but let the step4test panel
        # override per-launch. `step4test_cfg` carries the #panel-step4test inputs
        # (collected client-side and POSTed with /run|/reroute); each key falls
        # back to the inherited Step-5 saved config so a 4.test-only flow still
        # works even if Step 5 was never opened. mode="step4loop" reads from
        # step4loop_cfg instead (or an explicit `cfg` override, if given).
        # mode="editmasks" has no panel config of its own (Edit Masks is a
        # straight launch, no settings to collect); it falls back to the
        # inherited Step-5 config like a bare cfg={} would.
        s5cfg = current_project["steps"].get("5", {}).get("config", {})
        default_cfg = step4loop_cfg if mode == "step4loop" else step4test_cfg
        t4 = cfg if cfg is not None else (default_cfg or {})

        def _pick(key, default=""):
            """step4test panel value, else inherited Step-5 value, else default."""
            v = t4.get(key)
            if v is None or v == "":
                v = s5cfg.get(key)
            return v if (v is not None and v != "") else default

        # SETTINGS ENV CONTRACT (config.py reads these; the orchestrated
        # /api/configure is called with an empty body, so env defaults must apply).
        # Target-only + read-only-reference default ON ("1"); batch size default 10.
        target_only = "1" if t4.get("target_species_only", True) else "0"
        reference_default = "1" if t4.get("reference_default", True) else "0"
        raw_batch = _pick("review_batch_size", "10")
        batch_size = str(raw_batch).strip() or "10"

        # Expert-review export targets for a 4.test-only flow (Task 5): inherit
        # Step-5 config, else fall back to PATHS + the canonical repo url so REVIEW
        # export works even when Step 5 was never opened.
        review_dir = _pick("review_dir") or PATHS.get("review_dir", "")
        review_repo = (t4.get("review_repo_url") or s5cfg.get("review_repo_url")
                       or PATHS.get("review_repo_url", ""))
        expert_lib = (t4.get("expert_library_dir") or s5cfg.get("expert_library_dir")
                      or PATHS.get("expert_library_dir", ""))
        review_contacts = _pick("review_contacts")

        env_extra = {
            "TCRMP_INPUT_DIR": input_dir,
            "TCRMP_EXPORT_DIR": out_dir,
            "TCRMP_PROJECT_ID": current_project.get("id", "") if current_project else "",
            "TCRMP_PROJECT_NAME": current_project.get("name", "") if current_project else "",
            "TCRMP_TARGET_SPECIES": _get_target_species_str(),
            # Combined Step-4+5 settings (panel-overridable, Step-5-inherited).
            "TCRMP_TARGET_SPECIES_ONLY": target_only,
            "TCRMP_REFERENCE_DEFAULT": reference_default,
            "TCRMP_REVIEW_BATCH_SIZE": batch_size,
            "TCRMP_SAM3_DEVICE_TRACKER": _pick("sam3_device_tracker"),
            "TCRMP_SAM3_DEVICE_EXEMPLAR": _pick("sam3_device_exemplar"),
            "TCRMP_CONFIDENCE_THRESHOLD": str(_pick("confidence_threshold")),
            "TCRMP_MIN_MASK_AREA_PX": str(_pick("min_mask_area_px")),
            "TCRMP_MERGE_DISTANCE_PX": str(_pick("merge_distance_px")),
            "TCRMP_OVERLAP_STRATEGY": _pick("overlap_strategy"),
            # NEW env overrides plumbed through to combinedAnnotate config.py.
            "TCRMP_THIN_MASK_RATIO": str(_pick("thin_mask_ratio")),
            "TCRMP_POLYGON_SIMPLIFY_EPSILON": str(_pick("polygon_simplify_epsilon")),
            # Symlink-vs-copy of exported images (panel control; always sent so an
            # explicit "0" survives the empty-value sweep below).
            "TCRMP_SYMLINK_IMAGES": "1" if t4.get("symlink_images", True) else "0",
            # Expert-review export targets (REVIEW-flagged masks).
            "TCRMP_REVIEW_DIR": review_dir,
            "TCRMP_REVIEW_REPO_URL": review_repo,
            "TCRMP_EXPERT_LIBRARY_DIR": expert_lib,
            "TCRMP_REVIEW_CONTACTS": review_contacts,
        }
        # Target-only + reference default are an explicit "1"/"0" contract; keep
        # them even when "0". The rest drop out when blank so config.py defaults win.
        _always = {"TCRMP_TARGET_SPECIES_ONLY", "TCRMP_REFERENCE_DEFAULT",
                   "TCRMP_SYMLINK_IMAGES"}
        env_extra = {k: v for k, v in env_extra.items() if v or k in _always}

        # Combined-annotator mode + provenance source. These are NOT filtered
        # by the empty-value sweep above (they are always set).
        if mode == "step4loop":
            # Refine loop: the seeder already wrote segmentations.json (Task 3),
            # so the annotator must boot in resume mode (/api/resume, which reads
            # only segmentations.json) rather than fresh-configure MANUAL_ANNOTATE.
            env_extra["TCRMP_SESSION_MODE"] = "resume"
            env_extra["TCRMP_PROVENANCE_SOURCE"] = "step4loop"
        elif mode == "editmasks":
            # Edit Masks: no seeding pass, just open on the export's existing
            # segmentations.json. editMasks' own config.py already defaults
            # SESSION_MODE/PROVENANCE_SOURCE to "edit"; set them explicitly here
            # so the orchestrated launch never depends on that default drifting.
            env_extra["TCRMP_SESSION_MODE"] = "edit"
            env_extra["TCRMP_PROVENANCE_SOURCE"] = "edit"
        else:
            env_extra["TCRMP_MANUAL_ANNOTATE"] = "1"
            env_extra["TCRMP_PROVENANCE_SOURCE"] = "step4test"

        # Recoded master codes - same resolution as step 5.
        _mc = _resolve_master_codes()
        if _mc:
            env_extra["TCRMP_MASTER_CODES"] = _mc

        cmd = [py, entry, "--port", str(port)]
        result = runner.run_flask_stage(runner_key, cmd, port, cwd=cwd, env_extra=env_extra)
        if isinstance(result, dict) and result.get("error"):
            return result
        state_dict.update(running=True, port=port,
                          started_at=datetime.now().isoformat(), out_dir=out_dir,
                          project_id=(current_project.get("id") if current_project else None),
                          input_dir=input_dir)
        ready_message = "Edit Masks ready." if mode == "editmasks" else "Combined annotator ready."
        # Stamp which process owns this ui_ready: route_status is shared by all
        # three launch modes and the liveness check must poll the right one.
        owner_label = {"step4loop": "Refine annotator",
                       "editmasks": "Edit Masks app"}.get(mode, "Combined annotator")
        _route_set(phase="ui_ready", ui_port=port,
                   runner_key=runner_key, runner_label=owner_label,
                   message=ready_message)
        return result

    def _run_step6(step, cfg, step_dir):
        """Run dataset split + YOLO segmentation training via oceankind_CV.

        step_dir will receive:
            dataset/   train/valid/test split + data.yaml + test.yaml
            runs/      ultralytics training runs, one subdir per --name
        """
        # Input: the 4.test combined-annotator export dir (all_images/, all_labels/,
        # data.yaml). 4.test fully replaces Steps 4+5, so train always reads there.
        src_dir = _step4test_export_dir()
        if not src_dir or not os.path.isdir(src_dir):
            return {"error": f"Step 4 (Place + Segment) output not found at {src_dir}. Run Step 4 first."}
        # Quick sanity: a meaningful training payload needs the exported images.
        if not os.path.isdir(os.path.join(src_dir, "all_images")):
            return {"error": (
                f"Step 4 dir {src_dir} is missing all_images/. "
                "Export at least one batch from the Place + Segment UI to generate training data."
            )}

        # Resolve a training run name.
        run_name = (cfg.get("run_name") or "").strip()
        if not run_name:
            run_name = f"{current_project['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            current_project["steps"]["6"]["config"]["run_name"] = run_name

        model = cfg.get("model") or "yolo11m-seg.pt"
        epochs = str(cfg.get("epochs", 500))
        imgsz = str(cfg.get("imgsz", 512))

        # Fine-tune source: an explicit absolute weights path (e.g. a previous
        # run's best.pt, or the promoted champion) overrides the catalog model
        # name above. Preflight so a stale/typo'd path fails here with a clear
        # message instead of surfacing as an opaque ultralytics traceback.
        model_path = (cfg.get("model_path") or "").strip()
        if model_path:
            if not os.path.isfile(model_path):
                return {"error": f"Base weights not found: {model_path}"}
            model = model_path

        # Pre-flight: ultralytics refuses batch<1 (auto) under DDP. Catch it
        # here so the user sees a helpful error instead of the subprocess
        # traceback.
        device_preview = (cfg.get("device") or "").strip()
        batch_preview = cfg.get("batch", -1)
        try:
            batch_int = int(batch_preview)
        except (TypeError, ValueError):
            batch_int = -1
        if "," in device_preview and batch_int < 1:
            gpu_count = len([d for d in device_preview.split(",") if d.strip()])
            suggested = max(2, 8 * gpu_count)
            return {"error": (
                f"Multi-GPU training (device='{device_preview}') needs an explicit "
                f"batch size — ultralytics does not support auto-batch (batch=-1) "
                f"under DDP. Set Batch to a multiple of {gpu_count} in the Step 6 "
                f"panel (e.g. {suggested}) and re-run."
            )}

        # Invoke our bash driver in "all" mode: split then train.
        cmd = [
            ENTRY_POINTS[6], "all",
            src_dir, step_dir, run_name,
            epochs, imgsz, model,
        ]

        # Forward everything the user might have tweaked via env vars. The
        # shell driver only appends flags when the corresponding var is set,
        # so missing keys here just mean "use wrapper defaults".
        def _e(key, cfg_key, default=None):
            v = cfg.get(cfg_key, default)
            return ("", str(v)) if v is None else (key, str(v))

        # Pinned-split (frozen by-transect holdout) knobs. Unlike the rest of
        # _e()'s list below (omitted when unset, so run_step6.sh falls back to
        # its own default), these four are ALWAYS forwarded with an explicit
        # default so the Refine loop's seeder (which requires split_manifest.json
        # to exist) and pinned_split.py agree on the same defaults even when the
        # Step 6 panel never touched them.
        env_pin_split = {
            "TCRMP_STEP6_PIN_SPLIT": str(cfg.get("pin_split", "1")),
            "TCRMP_STEP6_VAL_TRANSECTS": str(cfg.get("val_transects", "5")),
            "TCRMP_STEP6_TEST_TRANSECTS": str(cfg.get("test_transects", "6")),
            "TCRMP_STEP6_HOLDOUT_MODE": str(cfg.get("holdout_mode", "transect")),
        }

        env_extra_raw = dict(_e(k, ck) for k, ck in [
            ("TCRMP_STEP6_VALID", "valid_ratio"),
            ("TCRMP_STEP6_TEST", "test_ratio"),
            ("TCRMP_STEP6_MIN_SAMPLES", "min_samples"),
            # Training plumbing
            ("TCRMP_STEP6_BATCH", "batch"),
            ("TCRMP_STEP6_PATIENCE", "patience"),
            ("TCRMP_STEP6_OPTIMIZER", "optimizer"),
            ("TCRMP_STEP6_SEED", "seed"),
            ("TCRMP_STEP6_COS_LR", "cos_lr"),
            ("TCRMP_STEP6_CLOSE_MOSAIC", "close_mosaic"),
            # LR
            ("TCRMP_STEP6_LR0", "lr0"),
            ("TCRMP_STEP6_LRF", "lrf"),
            ("TCRMP_STEP6_MOMENTUM", "momentum"),
            ("TCRMP_STEP6_WEIGHT_DECAY", "weight_decay"),
            ("TCRMP_STEP6_WARMUP_EPOCHS", "warmup_epochs"),
            # Loss
            ("TCRMP_STEP6_BOX", "box"),
            ("TCRMP_STEP6_CLS", "cls"),
            ("TCRMP_STEP6_DFL", "dfl"),
            ("TCRMP_STEP6_LABEL_SMOOTHING", "label_smoothing"),
            # Color
            ("TCRMP_STEP6_HSV_H", "hsv_h"),
            ("TCRMP_STEP6_HSV_S", "hsv_s"),
            ("TCRMP_STEP6_HSV_V", "hsv_v"),
            ("TCRMP_STEP6_BGR", "bgr"),
            # Geometric
            ("TCRMP_STEP6_DEGREES", "degrees"),
            ("TCRMP_STEP6_TRANSLATE", "translate"),
            ("TCRMP_STEP6_SCALE", "scale"),
            ("TCRMP_STEP6_SHEAR", "shear"),
            ("TCRMP_STEP6_PERSPECTIVE", "perspective"),
            ("TCRMP_STEP6_FLIPUD", "flipud"),
            ("TCRMP_STEP6_FLIPLR", "fliplr"),
            # Mixing
            ("TCRMP_STEP6_MOSAIC", "mosaic"),
            ("TCRMP_STEP6_MIXUP", "mixup"),
            ("TCRMP_STEP6_CUTMIX", "cutmix"),
            ("TCRMP_STEP6_COPY_PASTE", "copy_paste"),
            ("TCRMP_STEP6_COPY_PASTE_MODE", "copy_paste_mode"),
            ("TCRMP_STEP6_ERASING", "erasing"),
            ("TCRMP_STEP6_AUTO_AUGMENT", "auto_augment"),
        ])
        env_extra = {k: v for k, v in env_extra_raw.items() if k and v not in ("", "None")}
        env_extra.update(env_pin_split)

        # Also forward the fine-tune freeze depth, when set: TCRMP_STEP6_FREEZE
        # (int; number of leading layers to freeze). Unset means no freeze
        # (train_wrapper.py's --freeze default is None, i.e. train everything).
        freeze = cfg.get("freeze")
        if freeze not in (None, ""):
            env_extra["TCRMP_STEP6_FREEZE"] = str(freeze)

        # Class-inclusion filter: if the user picked a subset of classes in the
        # orchestrator, forward them as a CSV env var. Empty / None / "all" means
        # "train on every class present in step 5".
        include_classes = cfg.get("include_classes")

        # Guard: strip any unnamed classes from step 5's data.yaml so the
        # training payload never carries them, even if the user manually
        # re-added one via the UI or from a stale saved project. Matches
        # _step6_is_unnamed_class (empty / "(unnamed)" case-insensitive).
        try:
            import yaml as _y
            _src_yaml = os.path.join(src_dir, "data.yaml")
            if os.path.isfile(_src_yaml):
                with open(_src_yaml) as f:
                    _d = _y.safe_load(f) or {}
                _raw_names = _d.get("names") or {}
                if isinstance(_raw_names, dict):
                    _names = {int(k): v for k, v in _raw_names.items()}
                elif isinstance(_raw_names, list):
                    _names = {i: v for i, v in enumerate(_raw_names)}
                else:
                    _names = {}
                _unnamed_ids = {cid for cid, nm in _names.items()
                                if nm is None or not str(nm).strip() or
                                re.match(r"^\(unnamed\)$", str(nm).strip(),
                                         flags=re.IGNORECASE)}
                if _unnamed_ids:
                    if isinstance(include_classes, (list, tuple)):
                        include_classes = [int(x) for x in include_classes
                                           if int(x) not in _unnamed_ids]
                    else:
                        # "All classes present in the Step 4 (4.test) export"
                        # -> narrow to the explicit named subset so unnamed
                        # classes are stripped from the split.
                        include_classes = [cid for cid in sorted(_names.keys())
                                           if cid not in _unnamed_ids]
                    if not include_classes:
                        return {"error": (
                            "All classes resolved to unnamed / empty labels. "
                            "Re-annotate in Step 4 (Place + Segment) with proper class names in "
                            f"{_src_yaml}."
                        )}
        except Exception as _exc:
            # Non-fatal: fall back to what the UI sent if the 4.test data.yaml is
            # unreadable. (Distinct name from the _e() env-builder helper above.)
            pass

        if isinstance(include_classes, (list, tuple)) and len(include_classes) > 0:
            csv = ",".join(str(int(x)) for x in include_classes)
            env_extra["TCRMP_STEP6_INCLUDE_CLASSES"] = csv

        # Device selection — forwarded as ultralytics `device=`. A comma-separated
        # value (e.g. "0,1") triggers DDP across those GPUs. Blank means "wrapper
        # default" (cuda:0).
        device = (cfg.get("device") or "").strip()
        if device:
            env_extra["TCRMP_STEP6_DEVICE"] = device

        current_project["steps"]["6"]["config"].update({
            "run_name": run_name,
            "model": model,
            "model_path": model_path or None,
            "freeze": freeze if freeze not in (None, "") else None,
            "epochs": int(epochs),
            "imgsz": int(imgsz),
            "include_classes": include_classes if isinstance(include_classes, list) else None,
            "device": device or None,
        })
        pm.save_project(current_project)
        return runner.run_cli_stage(step, cmd, cwd=WORKING_DIRS[6], env_extra=env_extra)

    def _run_step7(step, cfg, step_dir):
        """Run ultralytics val on a trained run + emit metrics JSON + PDF report."""
        run_dir = (cfg.get("run_dir") or "").strip()
        if not run_dir or not os.path.isdir(run_dir):
            return {"error": (
                "Select a training run to evaluate. Expected a path like "
                f"{current_project['steps']['6']['dir']}/runs/<run_name>/"
            )}
        # Dataset is whatever the split step wrote.
        step6_dir = current_project["steps"].get("6", {}).get("dir", "")
        dataset_dir = os.path.join(step6_dir, "dataset")
        if not os.path.isdir(dataset_dir):
            return {"error": f"No dataset at {dataset_dir} — has step 6 run?"}

        py = PYTHON_PATHS.get(7)
        if not py or not os.path.isfile(py):
            return {"error": "Step 7 env not found — run TCRMPtrain_oceankindCV/setup_env.sh"}

        # Place evaluation outputs in a dedicated subdir tagged with the run name
        # so multiple evaluations don't clobber each other.
        run_name = os.path.basename(run_dir.rstrip("/"))
        out_dir = os.path.join(step_dir, f"eval_{run_name}")
        os.makedirs(out_dir, exist_ok=True)

        # --------------------------------------------------------------
        # imgsz handling: if the user left it blank/0, auto-read the value
        # the run was trained at from its args.yaml. If they supplied one
        # that differs from the training imgsz by more than 2×, surface a
        # non-blocking warning in the response (eval still runs).
        # --------------------------------------------------------------
        trained_imgsz = None
        args_path = os.path.join(run_dir, "args.yaml")
        if os.path.isfile(args_path):
            try:
                import yaml as _y
                with open(args_path) as f:
                    _args = _y.safe_load(f) or {}
                ti = _args.get("imgsz")
                if isinstance(ti, (int, float)) and ti > 0:
                    trained_imgsz = int(ti)
            except Exception:
                trained_imgsz = None

        raw_imgsz = cfg.get("imgsz")
        try:
            user_imgsz = int(raw_imgsz) if raw_imgsz not in (None, "", 0, "0") else 0
        except (TypeError, ValueError):
            user_imgsz = 0

        warnings = []
        if user_imgsz <= 0:
            # Auto-fill from the run's args.yaml; last-resort fallback 1024.
            chosen_imgsz = trained_imgsz if trained_imgsz else 1024
            warnings.append(
                f"imgsz was blank; auto-filled from run args.yaml -> {chosen_imgsz}"
                if trained_imgsz else
                "imgsz was blank and run has no args.yaml; falling back to 1024"
            )
        else:
            chosen_imgsz = user_imgsz
            if trained_imgsz and (
                chosen_imgsz > 2 * trained_imgsz or trained_imgsz > 2 * chosen_imgsz
            ):
                warnings.append(
                    f"imgsz={chosen_imgsz} differs from training imgsz={trained_imgsz} "
                    f"by more than 2×. Metrics may be distorted — consider matching the "
                    f"training value."
                )

        cmd = [
            py, ENTRY_POINTS[7],
            "--run_dir", run_dir,
            "--dataset_dir", dataset_dir,
            "--out_dir", out_dir,
            "--split", str(cfg.get("split", "test")),
            "--imgsz", str(chosen_imgsz),
            "--batch", str(cfg.get("batch", -1)),
            "--conf", str(cfg.get("conf_threshold", 0.25)),
            "--iou", str(cfg.get("iou_threshold", 0.6)),
            "--preview_count", str(cfg.get("preview_count", 8)),
        ]
        pdf_export_dir = (cfg.get("pdf_export_dir") or "").strip()
        if pdf_export_dir:
            cmd += ["--pdf_export_dir", pdf_export_dir]

        # Active-learning rounds ledger: every eval appends/updates a row in
        # {step6_dir}/rounds.csv (keyed by run_name, so re-evaluating a run
        # updates its row rather than duplicating it) and gates this round
        # against the current champion.json. rounds.csv + champion.json both
        # live directly under step6_dir (a sibling of dataset/), not under this
        # eval's own out_dir, so every round's eval shares one ledger.
        cmd += ["--rounds_dir", step6_dir]

        # Persist the evaluation-output dir so the panel knows where to look.
        current_project["steps"]["7"]["config"]["run_dir"] = run_dir
        current_project["steps"]["7"]["config"]["imgsz"] = chosen_imgsz
        current_project["steps"]["7"]["outputs"]["out_dir"] = out_dir
        current_project["steps"]["7"]["outputs"]["report_md"] = os.path.join(out_dir, "report.md")
        current_project["steps"]["7"]["outputs"]["report_pdf"] = os.path.join(out_dir, "report.pdf")
        current_project["steps"]["7"]["outputs"]["metrics_json"] = os.path.join(out_dir, "metrics.json")
        current_project["steps"]["7"]["outputs"]["rounds_csv"] = os.path.join(step6_dir, "rounds.csv")
        current_project["steps"]["7"]["outputs"]["champion_json"] = os.path.join(step6_dir, "champion.json")
        pm.save_project(current_project)
        result = runner.run_cli_stage(step, cmd, cwd=WORKING_DIRS[7])
        # Attach any warnings so the UI can surface them (non-blocking).
        if warnings and isinstance(result, dict):
            existing = result.get("warnings") or []
            if isinstance(existing, list):
                result["warnings"] = existing + warnings
            else:
                result["warnings"] = warnings
        return result

    def _run_step8(step, cfg, step_dir):
        """Run model inference on a user-selected source and save overlays + manifest."""
        run_dir = (cfg.get("run_dir") or "").strip()
        if not run_dir or not os.path.isdir(run_dir):
            return {"error": "Select a training run whose weights to use."}

        py = PYTHON_PATHS.get(8)
        if not py or not os.path.isfile(py):
            return {"error": "Step 8 env not found — run TCRMPtrain_oceankindCV/setup_env.sh"}

        # User-supplied run name — sanitize and stamp into the output folder.
        import re as _re
        run_name = (cfg.get("run_name") or "").strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name:
            safe = _re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name)
            out_leaf = f"{safe}_{timestamp}"
        else:
            safe = ""
            out_leaf = f"infer_{timestamp}"
        out_dir = os.path.join(step_dir, out_leaf)
        os.makedirs(out_dir, exist_ok=True)

        source_type = cfg.get("source_type", "unused")
        cmd = [
            py, ENTRY_POINTS[8],
            "--run_dir", run_dir,
            "--out_dir", out_dir,
            "--source_type", source_type,
            "--conf", str(cfg.get("conf_threshold", 0.25)),
            "--iou", str(cfg.get("iou_threshold", 0.7)),
            "--imgsz", str(cfg.get("imgsz", 512)),
            "--sample_count", str(cfg.get("sample_count", 100)),
            "--sample_mode", str(cfg.get("sample_mode", "random")),
            "--mask_alpha", str(cfg.get("mask_alpha", 0.45)),
            "--run_name", run_name or out_leaf,
            "--project_name", current_project.get("name", ""),
        ]
        if cfg.get("draw_boxes"):
            cmd.append("--draw_boxes")
        if source_type == "dir":
            source_dir = (cfg.get("source_dir") or "").strip()
            if not source_dir:
                return {"error": "source_type=dir requires source_dir"}
            cmd += ["--source_dir", source_dir]
        elif source_type in ("unused", "full"):
            # Pass the step 3 selected_frames.csv so we can exclude those for "unused".
            selected_frames = (
                current_project["steps"].get("3", {}).get("outputs", {}).get("selected_frames")
                or current_project["steps"].get("3", {}).get("config", {}).get("selected_frames")
                or ""
            )
            # Resolve the clip dir the same way routing does: a stale saved
            # Step-4 clip_dir (e.g. an old Dropbox path) falls back to the local
            # full set, so visualization reads the same tree as the rest.
            clip_dir = _resolve_routing_clip_dir(
                {}, current_project["steps"].get("4", {}).get("config", {}))
            if selected_frames:
                cmd += ["--selected_frames", selected_frames]
            if clip_dir:
                cmd += ["--clip_dir", clip_dir]
        if cfg.get("save_overlays"):
            cmd.append("--save_overlays")
        if cfg.get("save_crops"):
            cmd.append("--save_crops")
        if cfg.get("include_pts_variant"):
            cmd.append("--include_pts_variant")
        # Always persist per-detection polygons to predictions.json. The Refine
        # loop (step4loop) seeder reads this file to turn model detections into
        # pending review masks, so every inference run must produce it, not just
        # ones the user happens to flag for it.
        cmd.append("--save_predictions")

        current_project["steps"]["8"]["config"]["run_dir"] = run_dir
        current_project["steps"]["8"]["outputs"]["out_dir"] = out_dir
        current_project["steps"]["8"]["outputs"]["manifest"] = os.path.join(out_dir, "manifest.json")
        pm.save_project(current_project)
        return runner.run_cli_stage(step, cmd, cwd=WORKING_DIRS[8])

    # ------------------------------------------------------------------
    # Link existing outputs (skip running a step)
    # ------------------------------------------------------------------
    @app.route("/api/step/1/link", methods=["POST"])
    def step1_link():
        """Link existing all_points + master_codes to skip Step 1."""
        global current_project
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400

        data = request.get_json(force=True)
        ap_path = data.get("all_points", "").strip()
        mc_path = data.get("master_codes", "").strip()

        if not ap_path or not os.path.isfile(ap_path):
            return jsonify({"error": f"all_points file not found: {ap_path}"}), 400

        # master_codes: try to auto-detect from same directory if not given
        if not mc_path:
            mc_path = os.path.join(os.path.dirname(ap_path), "master_codes.csv")
        if not os.path.isfile(mc_path):
            return jsonify({"error": f"master_codes file not found: {mc_path}"}), 400

        step1 = current_project["steps"]["1"]
        step_dir = step1["dir"]
        os.makedirs(step_dir, exist_ok=True)

        # Symlink or copy into the project step dir
        import shutil
        for src, name in [(ap_path, os.path.basename(ap_path)), (mc_path, os.path.basename(mc_path))]:
            dst = os.path.join(step_dir, name)
            if not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copy2(src, dst)

        step1["outputs"]["all_points"] = ap_path
        step1["outputs"]["master_codes"] = mc_path

        # Count rows for feedback
        rows = None
        try:
            with open(ap_path) as f:
                rows = sum(1 for _ in f) - 1  # minus header
        except Exception:
            pass

        pm.complete_step(current_project, 1)

        return jsonify({
            "success": True,
            "rows": rows,
            "state": current_project,
        })

    # ------------------------------------------------------------------
    # Step status / control
    # ------------------------------------------------------------------
    @app.route("/api/step/<int:step>/status")
    def step_status(step):
        status = runner.poll_status(step)
        # For Flask stages, add health check
        port = runner.ports.get(step)
        if port:
            status["healthy"] = runner.health_check(port)
            status["port"] = port

        # Auto-detect CLI completion
        if current_project:
            s = str(step)
            st = current_project["steps"].get(s, {})
            if st.get("status") == "running" and not status["running"]:
                if status["exit_code"] == 0:
                    _vic_end_step(step, "success")
                    pm.complete_step(current_project, step)
                    status["completed"] = True
                else:
                    st["status"] = "error"
                    _vic_end_step(step, "failed",
                                  notes=f"exit_code={status.get('exit_code')}")
                    pm.save_project(current_project)

        return jsonify(status)

    @app.route("/api/step/<int:step>/log")
    def step_log(step):
        offset = request.args.get("offset", 0, type=int)
        lines, new_offset, dropped = runner.get_log(step, offset)
        return jsonify({
            "lines": lines,
            "offset": new_offset,
            "dropped": dropped,
        })

    @app.route("/api/step/<int:step>/done", methods=["POST"])
    def step_done(step):
        """User marks an interactive (Flask) step as complete."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        # Kill the sub-app
        runner.kill(step)
        if step == 5:
            _sam3_mark_stopped("Step 5 marked done.")
        _vic_end_step(step, "success", notes="marked done via UI")
        pm.complete_step(current_project, step)
        return jsonify({"success": True, "state": current_project})

    @app.route("/api/step/<int:step>/stop", methods=["POST"])
    def step_stop(step):
        runner.kill(step)
        if step == 5:
            _sam3_mark_stopped("Step 5 stopped.")
        if current_project:
            s = str(step)
            st = current_project["steps"].get(s, {})
            if st.get("status") == "running":
                st["status"] = "pending"
                _vic_end_step(step, "cancelled", notes="stopped via UI")
                pm.save_project(current_project)
        return jsonify({"success": True})

    @app.route("/api/step/<int:step>/reset", methods=["POST"])
    def step_reset(step):
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        # Kill any running processes for this and later steps
        last = int(pm.STEP_KEYS[-1])
        for s in range(step, last + 1):
            runner.kill(s)
        if step <= 5:
            _sam3_mark_stopped("Step reset.")
        pm.reset_step(current_project, step)
        return jsonify({"success": True, "state": current_project})

    # ------------------------------------------------------------------
    # Remap endpoints
    # ------------------------------------------------------------------
    @app.route("/api/remap/browse")
    def remap_browse():
        extra = []
        if current_project:
            extra.append(current_project["project_dir"])
        logs = remap_loader.browse_remap_logs(REPO_DIR, extra_dirs=extra)
        return jsonify({"logs": logs})

    @app.route("/api/remap/load", methods=["POST"])
    def remap_load():
        data = request.get_json(force=True)
        path = data.get("path", "")
        if not os.path.isfile(path):
            return jsonify({"error": "File not found"}), 404
        try:
            log = remap_loader.load_remap_log(path)
            return jsonify({"success": True, "remap_log": log})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/remap/apply", methods=["POST"])
    def remap_apply():
        """Auto-apply remaps without launching interactive UI."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400

        data = request.get_json(force=True)
        remap_log_path = data.get("remap_log_path", "")
        if not os.path.isfile(remap_log_path):
            return jsonify({"error": "Remap log file not found"}), 404

        step2 = current_project["steps"]["2"]
        ap = step2["config"].get("all_points", "")
        mc = step2["config"].get("master_codes", "")
        if not ap or not mc:
            return jsonify({"error": "Step 1 outputs not linked (all_points / master_codes)"}), 400

        output_dir = step2["dir"]
        try:
            step2["status"] = "running"
            pm.save_project(current_project)

            result = remap_loader.apply_remaps(ap, mc, remap_log_path, output_dir)

            step2["config"]["remap_log_source"] = remap_log_path
            pm.complete_step(current_project, 2)

            return jsonify({"success": True, **result, "state": current_project})
        except Exception as e:
            step2["status"] = "error"
            pm.save_project(current_project)
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------
    @app.route("/api/fs/open", methods=["POST"])
    def fs_open():
        """Open a directory in the desktop file manager via xdg-open.

        Accepts { path } and only opens paths inside the current project or
        the repo — we do not want random server-side path access.
        """
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        data = request.get_json(force=True) or {}
        path = os.path.abspath(data.get("path", "").strip())
        if not path or not os.path.isdir(path):
            return jsonify({"error": f"not a directory: {path}"}), 400
        project_dir = os.path.abspath(current_project["project_dir"])
        repo_dir = os.path.abspath(REPO_DIR)
        if not (path.startswith(project_dir + os.sep) or path == project_dir
                or path.startswith(repo_dir + os.sep) or path == repo_dir):
            return jsonify({"error": "path outside project/repo"}), 403
        try:
            import subprocess
            subprocess.Popen(
                ["xdg-open", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return jsonify({"success": True, "path": path})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    # ------------------------------------------------------------------
    # Step 3 helpers — label picker feed
    # ------------------------------------------------------------------
    @app.route("/api/step/3/target_labels")
    def step3_target_labels():
        """Return the label universe for the Step-3 picker.

        This feed is AUTO-LOADED by the Step 3 panel when the user enters the
        step (there is no manual "load labels" button); it is the sole way
        labels reach Step 3. The frontend groups the returned labels by their
        `category` string (empty/missing category is bucketed under
        "Uncategorized" client-side) and re-sorts within each category, so the
        server-side ordering below is harmless.

        Source of truth resolution (so the correct labels load straight from
        Step 2 when a recode has happened):
          1. Step 2's `master_codes_recoded.csv` if it exists (post-recode).
          2. Step 2's configured `master_codes`.
          3. Step 1's `master_codes.csv` (outputs, else config) otherwise.
        Also returns the excluded-code set from Step 2's remap log so the UI
        can mark or hide them. Row counts per label come from the current
        all_points CSV (step 2 recoded, else step 1) for sorting.

        Response shape (the client depends on these keys):
          labels[], excludes[], master_codes_path, all_points_path,
          remap_log_path. Every label always carries a clean `category`
          string (NaN is normalized away via .fillna("")). A missing master
          codes file returns {"error": ...}, 404; an unreadable one returns
          {"error": ...}, 500 rather than a raw stack trace.
        """
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        s1 = current_project["steps"]["1"]
        s2 = current_project["steps"]["2"]
        # Master codes: prefer step 2 output, fall back to step 1.
        mc_path = (
            s2.get("outputs", {}).get("master_codes_recoded")
            or s2.get("config", {}).get("master_codes")
            or s1.get("outputs", {}).get("master_codes")
            or s1.get("config", {}).get("master_codes", "")
        )
        if not mc_path or not os.path.isfile(mc_path):
            return jsonify({"error": f"master_codes not found: {mc_path}"}), 404
        # All points: prefer recoded.
        ap_path = (
            s2.get("outputs", {}).get("all_points_recoded")
            or s2.get("config", {}).get("all_points")
            or s1.get("outputs", {}).get("all_points", "")
        )
        # Excludes: look in the remap log that was applied.
        excludes: list[str] = []
        remap_log = s2.get("config", {}).get("remap_log_source", "")
        if remap_log and os.path.isfile(remap_log):
            try:
                with open(remap_log) as f:
                    excludes = list(json.load(f).get("excludes", []) or [])
            except Exception:
                excludes = []
        try:
            import pandas as _pd
            mc = _pd.read_csv(mc_path)
            mc = mc.fillna("")
            counts: dict = {}
            frame_counts: dict = {}
            if ap_path and os.path.isfile(ap_path):
                try:
                    ap = _pd.read_csv(
                        ap_path,
                        usecols=["species_code", "date", "site", "transect", "frame"],
                    )
                    counts = ap["species_code"].value_counts().to_dict()
                    ap["__fid"] = (
                        ap["date"].astype(str) + "|" + ap["site"].astype(str) + "|"
                        + ap["transect"].astype(str) + "|" + ap["frame"].astype(str)
                    )
                    frame_counts = (
                        ap.groupby("species_code")["__fid"].nunique().to_dict()
                    )
                except Exception:
                    counts = {}
                    frame_counts = {}
            labels = []
            for _, row in mc.iterrows():
                code = str(row.get("code", ""))
                if not code:
                    continue
                labels.append({
                    "code": code,
                    "name": str(row.get("name", "")),
                    "category": str(row.get("category", "")),
                    "count": int(counts.get(code, 0)),
                    "frames": int(frame_counts.get(code, 0)),
                    "excluded": code in excludes,
                })
            labels.sort(key=lambda r: (r["frames"], r["count"]), reverse=True)
            return jsonify({
                "labels": labels,
                "excludes": excludes,
                "master_codes_path": mc_path,
                "all_points_path": ap_path,
                "remap_log_path": remap_log,
            })
        except Exception as exc:
            return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500

    # ------------------------------------------------------------------
    # Step 7/8: listings, reports, downloads
    # ------------------------------------------------------------------
    def _finetune_model_path():
        """Base weights for the fine-tune preset: the promoted champion's
        best.pt if one exists, else the newest run's best.pt (by mtime), else
        '' (no runs yet - the preset then falls back to the catalog model)."""
        if current_project is None:
            return ""
        step6_dir = current_project["steps"].get("6", {}).get("dir", "")
        champion = _load_champion_json(step6_dir)
        if champion:
            run_dir = champion.get("run_dir", "")
            best = os.path.join(run_dir, "weights", "best.pt")
            if os.path.isfile(best):
                return best
        runs_dir = os.path.join(step6_dir, "runs")
        newest_best, newest_mtime = "", -1
        if os.path.isdir(runs_dir):
            for nm in os.listdir(runs_dir):
                best = os.path.join(runs_dir, nm, "weights", "best.pt")
                if os.path.isfile(best):
                    mtime = os.path.getmtime(best)
                    if mtime > newest_mtime:
                        newest_mtime, newest_best = mtime, best
        return newest_best

    # Built-in Step 6 presets (Task 8): unlike the file-based presets above,
    # these two are synthesized here because "finetune" needs a model_path
    # resolved live against the current project's champion/newest run.
    _BUILTIN_PRESET_IDS = ("finetune", "retrain")

    def _builtin_preset(preset_id):
        """Params dict for a built-in preset id, or None if not a builtin."""
        if preset_id == "finetune":
            return {
                "name": "Fine-tune preset (fast loop)",
                "description": (
                    "Continues from your best weights. 10x lower learning rate "
                    "protects what the model already knows; freezing the backbone "
                    "prevents overfitting on a small correction batch. Never uses "
                    "'resume' (that is only for crashed runs)."
                ),
                "params": {
                    "model_path": _finetune_model_path(),
                    "epochs": 150,
                    "patience": 20,
                    "lr0": 0.001,
                    "freeze": 10,
                    "pin_split": "1",
                },
            }
        if preset_id == "retrain":
            return {
                "name": "Full retrain preset (recommended)",
                "description": (
                    "Retrains from the COCO-pretrained base on ALL accumulated "
                    "labels (original + every Refine round). Most robust against "
                    "forgetting; preferred for the model you keep."
                ),
                "params": {
                    "model": "yolo11m-seg.pt",
                    "pin_split": "1",
                },
            }
        return None

    @app.route("/api/step/6/presets")
    def step6_list_presets():
        """List Step 6 training presets: the two built-in presets (finetune,
        retrain - Task 8) plus any file-based ones shipped in
        pipeline_orchestrator/presets/.

        Also picks a *default* preset for the UI to auto-apply when the form
        has never been populated. Resolution order:
          1. any preset whose top-level or _meta section carries `default: true`
          2. otherwise, the alphabetical-first preset on disk
        Built-in presets are never auto-applied as the default (opt-in only,
        via their own preset buttons).
        """
        out = []
        for pid in _BUILTIN_PRESET_IDS:
            b = _builtin_preset(pid)
            out.append({
                "id": pid,
                "name": b["name"],
                "description": b["description"],
                "default": False,
                "builtin": True,
            })

        preset_dir = PRESET_DIRS.get("train", "")
        default_id = None
        if os.path.isdir(preset_dir):
            import yaml as _y
            for fname in sorted(os.listdir(preset_dir)):
                if not fname.endswith(".yaml") and not fname.endswith(".yml"):
                    continue
                fpath = os.path.join(preset_dir, fname)
                try:
                    with open(fpath) as f:
                        data = _y.safe_load(f) or {}
                except Exception as e:
                    out.append({"id": fname, "error": f"parse error: {e}"})
                    continue
                meta = data.get("_meta") or {}
                is_default = bool(data.get("default") or meta.get("default"))
                if is_default and default_id is None:
                    default_id = fname
                out.append({
                    "id": fname,
                    "name": meta.get("name") or fname,
                    "description": meta.get("description") or "",
                    "default": is_default,
                })
            # Fallback: no preset flagged default -> alphabetical-first valid one.
            if default_id is None:
                for p in out:
                    if "error" not in p and not p.get("builtin"):
                        default_id = p["id"]
                        break
        return jsonify({"presets": out, "dir": preset_dir, "default_id": default_id})

    @app.route("/api/step/6/presets/<path:preset_id>")
    def step6_get_preset(preset_id):
        """Return the parsed params dict for one preset. preset_id is the
        filename for file-based presets, or "finetune"/"retrain" for the
        built-in ones (Task 8)."""
        builtin = _builtin_preset(preset_id)
        if builtin is not None:
            return jsonify({
                "id": preset_id,
                "name": builtin["name"],
                "description": builtin["description"],
                "params": builtin["params"],
            })
        # Reject any path traversal attempts — only allow plain filenames in the dir.
        safe_name = os.path.basename(preset_id)
        if safe_name != preset_id:
            return jsonify({"error": "invalid preset id"}), 400
        preset_dir = PRESET_DIRS.get("train", "")
        fpath = os.path.join(preset_dir, safe_name)
        if not os.path.isfile(fpath):
            return jsonify({"error": f"preset not found: {safe_name}"}), 404
        import yaml as _y
        try:
            with open(fpath) as f:
                data = _y.safe_load(f) or {}
        except Exception as e:
            return jsonify({"error": f"parse error: {e}"}), 500
        meta = data.get("_meta") or {}
        params = data.get("params") or {}
        return jsonify({
            "id": safe_name,
            "name": meta.get("name") or safe_name,
            "description": meta.get("description") or "",
            "params": params,
        })

    # ------------------------------------------------------------------
    # Step 7: preset loader (mirrors step 6 preset endpoints) + run_info
    # ------------------------------------------------------------------
    @app.route("/api/step/7/presets")
    def step7_list_presets():
        """List Step 7 evaluation presets shipped in pipeline_orchestrator/eval_presets/."""
        preset_dir = PRESET_DIRS.get("eval", "")
        out = []
        if os.path.isdir(preset_dir):
            import yaml as _y
            for fname in sorted(os.listdir(preset_dir)):
                if not fname.endswith(".yaml") and not fname.endswith(".yml"):
                    continue
                fpath = os.path.join(preset_dir, fname)
                try:
                    with open(fpath) as f:
                        data = _y.safe_load(f) or {}
                except Exception as e:
                    out.append({"id": fname, "error": f"parse error: {e}"})
                    continue
                meta = data.get("_meta") or {}
                out.append({
                    "id": fname,
                    "name": meta.get("name") or fname,
                    "description": meta.get("description") or "",
                })
        return jsonify({"presets": out, "dir": preset_dir})

    @app.route("/api/step/7/presets/<path:preset_id>")
    def step7_get_preset(preset_id):
        """Return the parsed params dict for one Step 7 preset. preset_id is the filename."""
        safe_name = os.path.basename(preset_id)
        if safe_name != preset_id:
            return jsonify({"error": "invalid preset id"}), 400
        preset_dir = PRESET_DIRS.get("eval", "")
        fpath = os.path.join(preset_dir, safe_name)
        if not os.path.isfile(fpath):
            return jsonify({"error": f"preset not found: {safe_name}"}), 404
        import yaml as _y
        try:
            with open(fpath) as f:
                data = _y.safe_load(f) or {}
        except Exception as e:
            return jsonify({"error": f"parse error: {e}"}), 500
        meta = data.get("_meta") or {}
        params = data.get("params") or {}
        return jsonify({
            "id": safe_name,
            "name": meta.get("name") or safe_name,
            "description": meta.get("description") or "",
            "params": params,
        })

    @app.route("/api/step/7/run_info")
    def step7_run_info():
        """Peek at a training run's args.yaml so the Step 7 Quick-start banner
        can auto-fill `imgsz` (among a few other useful fields). Returns the
        subset of fields the UI needs; missing file -> 200 with empty values."""
        run_dir = (request.args.get("path") or "").strip()
        if not run_dir:
            return jsonify({"error": "missing ?path=<run_dir>"}), 400
        if not os.path.isdir(run_dir):
            return jsonify({"error": f"not a directory: {run_dir}"}), 404
        args_path = os.path.join(run_dir, "args.yaml")
        info = {"model": None, "imgsz": None, "epochs": None}
        if os.path.isfile(args_path):
            import yaml as _y
            try:
                with open(args_path) as f:
                    data = _y.safe_load(f) or {}
            except Exception as e:
                return jsonify({"error": f"parse error: {e}", **info}), 500
            info["model"] = data.get("model")
            info["imgsz"] = data.get("imgsz")
            info["epochs"] = data.get("epochs")
        return jsonify({"run_dir": run_dir, "args_yaml_found": os.path.isfile(args_path), **info})

    # Helper: should this class name be dropped from training?
    # Drops empty strings and the literal "(unnamed)" (case-insensitive, with
    # any surrounding whitespace). Used both by /list_classes and the
    # Start-Training guard.
    def _step6_is_unnamed_class(name):
        if name is None:
            return True
        s = str(name).strip()
        if not s:
            return True
        return re.match(r"^\(unnamed\)$", s, flags=re.IGNORECASE) is not None

    @app.route("/api/step/6/list_classes")
    def step6_list_classes():
        """Enumerate classes present in the Step 4 (4.test) export so the user can
        pick which ones to include in training. Returns class id, name (from
        data.yaml), and per-class label-instance count from all_labels/.

        Unnamed/empty-name classes (including the literal "(unnamed)") are
        filtered out — they have leaked into prior training runs and the user
        wants them suppressed at source.

        Fallback: if Step 4 hasn't exported yet (no data.yaml), synthesize the
        class list from step 3's target_species so the panel is still usable.
        """
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400

        src_dir = _step4test_export_dir()
        yaml_path = os.path.join(src_dir, "data.yaml") if src_dir else ""
        src_ready = bool(src_dir and os.path.isfile(yaml_path))

        names = {}
        if src_ready:
            try:
                import yaml as _y
                with open(yaml_path) as f:
                    data = _y.safe_load(f) or {}
                raw = data.get("names") or {}
                if isinstance(raw, dict):
                    names = {int(k): v for k, v in raw.items()}
                elif isinstance(raw, list):
                    names = {i: v for i, v in enumerate(raw)}
            except Exception as e:
                return jsonify({"error": f"Could not parse {yaml_path}: {e}"}), 500

        counts = {}
        image_counts = {}
        if src_ready:
            label_dir = os.path.join(src_dir, "all_labels")
            if os.path.isdir(label_dir):
                for fname in os.listdir(label_dir):
                    fpath = os.path.join(label_dir, fname)
                    seen_in_file = set()
                    try:
                        with open(fpath) as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    cid = int(line.split()[0])
                                except ValueError:
                                    continue
                                counts[cid] = counts.get(cid, 0) + 1
                                seen_in_file.add(cid)
                    except OSError:
                        continue
                    for cid in seen_in_file:
                        image_counts[cid] = image_counts.get(cid, 0) + 1

        # If step 5 never ran, fall back to step 3's configured target_species
        # so the panel can at least preview the expected class roster. No
        # counts available in this mode.
        source = "step4"
        if not names and not counts:
            s3_cfg = current_project["steps"].get("3", {}).get("config", {}) or {}
            species_raw = s3_cfg.get("target_species") or ""
            species = [s.strip() for s in species_raw.split(",") if s.strip()]
            if species:
                names = {i: sp for i, sp in enumerate(species)}
                source = "step3_fallback"
            else:
                return jsonify({
                    "error": ("Step 4 output not found and step 3 has no "
                              "target_species — run Step 3 or Step 4 first."),
                }), 400

        all_ids = sorted(set(names.keys()) | set(counts.keys()))
        classes = []
        dropped_unnamed = []
        for cid in all_ids:
            nm = names.get(cid)
            if _step6_is_unnamed_class(nm):
                dropped_unnamed.append(cid)
                continue
            classes.append({
                "id": cid,
                "name": str(nm).strip(),
                "instance_count": counts.get(cid, 0),
                "image_count": image_counts.get(cid, 0),
            })
        saved = current_project["steps"].get("6", {}).get("config", {}).get("include_classes")
        return jsonify({
            "classes": classes,
            "src_dir": src_dir,
            "include_classes": saved,
            "source": source,
            "dropped_unnamed_ids": dropped_unnamed,
        })

    @app.route("/api/step/6/list_runs")
    def step6_list_runs():
        """List training runs that step 7/8 can pick from."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        step6_dir = current_project["steps"].get("6", {}).get("dir", "")
        runs_dir = os.path.join(step6_dir, "runs")
        # Champion (Task 9): when {step6_dir}/champion.json exists, the step-8
        # run picker pre-selects its run_dir and each matching run entry below
        # carries is_champion=true (so the eval panel can badge it too).
        champion = _load_champion_json(step6_dir)
        champion_run_dir = champion.get("run_dir") if champion else None
        if not os.path.isdir(runs_dir):
            return jsonify({"runs": [], "champion_run_dir": champion_run_dir})
        out = []
        for nm in sorted(os.listdir(runs_dir)):
            rdir = os.path.join(runs_dir, nm)
            if not os.path.isdir(rdir):
                continue
            weights_dir = os.path.join(rdir, "weights")
            best = os.path.join(weights_dir, "best.pt")
            last = os.path.join(weights_dir, "last.pt")
            has_best = os.path.isfile(best)
            has_last = os.path.isfile(last)
            # Pull a small subset of training args for the dropdown tooltip.
            args_yaml = os.path.join(rdir, "args.yaml")
            model = epochs = None
            if os.path.isfile(args_yaml):
                try:
                    import yaml as _y
                    with open(args_yaml) as f:
                        a = _y.safe_load(f) or {}
                    model = a.get("model")
                    epochs = a.get("epochs")
                except Exception:
                    pass
            out.append({
                "name": nm,
                "path": rdir,
                "has_best": has_best,
                "has_last": has_last,
                "model": model,
                "epochs": epochs,
                "is_champion": bool(champion_run_dir) and os.path.abspath(rdir) == os.path.abspath(champion_run_dir),
            })
        return jsonify({"runs": out, "runs_dir": runs_dir, "champion_run_dir": champion_run_dir})

    # Rounds ledger numeric columns (Task 9): cast on the way out of rounds.csv
    # so the frontend never has to parseFloat/parseInt csv strings itself.
    _ROUNDS_INT_COLS = ("round", "n_train", "n_valid", "n_test")
    _ROUNDS_FLOAT_COLS = ("map50_M", "map50_95_M", "recall_M", "precision_M",
                          "map50_B", "map50_95_B")

    def _cast_round_row(row):
        out = dict(row)
        for k in _ROUNDS_INT_COLS:
            v = out.get(k)
            try:
                out[k] = int(v) if v not in (None, "") else None
            except (TypeError, ValueError):
                out[k] = None
        for k in _ROUNDS_FLOAT_COLS:
            v = out.get(k)
            try:
                out[k] = float(v) if v not in (None, "") else None
            except (TypeError, ValueError):
                out[k] = None
        return out

    @app.route("/api/step/7/rounds")
    def step7_rounds():
        """Rounds ledger for the active learning loop (Task 9): every row of
        {step6_dir}/rounds.csv (numeric columns cast) plus the current
        champion, so the eval panel's Rounds table and the champion badge in
        the inference run picker can both be driven from one call."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        step6_dir = current_project["steps"].get("6", {}).get("dir", "")
        rounds_csv_path = os.path.join(step6_dir, "rounds.csv")
        rows = []
        if os.path.isfile(rounds_csv_path):
            with open(rounds_csv_path, newline="") as f:
                rows = [_cast_round_row(r) for r in csv.DictReader(f)]
        champion = _load_champion_json(step6_dir)
        return jsonify({"rounds": rows, "champion": champion})

    @app.route("/api/step/7/promote", methods=["POST"])
    def step7_promote():
        """Promote a trained run to champion (Task 9): writes
        {step6_dir}/champion.json so the fine-tune preset and the step 8 run
        picker both default to this run. Validates the run has weights AND an
        eval row in rounds.csv before writing (a run nobody has evaluated yet
        has no basis for being "the best")."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        body = request.get_json(force=True, silent=True) or {}
        run_dir = (body.get("run_dir") or "").strip()
        if not run_dir or not os.path.isdir(run_dir):
            return jsonify({"error": f"run_dir not found: {run_dir}"}), 400
        run_dir = os.path.abspath(run_dir)
        best = os.path.join(run_dir, "weights", "best.pt")
        if not os.path.isfile(best):
            return jsonify({"error": f"Run has no weights/best.pt: {run_dir}"}), 400

        step6_dir = current_project["steps"].get("6", {}).get("dir", "")
        rounds_csv_path = os.path.join(step6_dir, "rounds.csv")
        run_name = os.path.basename(run_dir.rstrip("/"))
        row = None
        if os.path.isfile(rounds_csv_path):
            with open(rounds_csv_path, newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("run_name") == run_name or r.get("run_dir") == run_dir:
                        row = r
                        break
        if row is None:
            return jsonify({"error": (
                f"No evaluation on record for {run_name}. Run Step 7 (Evaluate) "
                "on this run before promoting it."
            )}), 400

        try:
            map50_95_m = float(row.get("map50_95_M"))
        except (TypeError, ValueError):
            map50_95_m = None

        champion = {
            "run_dir": run_dir,
            "promoted_at": _ast_now(),
            "map50_95_M": map50_95_m,
            "note": str(body.get("note") or ""),
        }
        os.makedirs(step6_dir, exist_ok=True)
        champion_path = os.path.join(step6_dir, "champion.json")
        tmp_path = champion_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(champion, f, indent=2)
        os.replace(tmp_path, champion_path)
        vic.note(f"[orch] promoted {run_name} to champion (mask mAP50-95={map50_95_m})")
        return jsonify({"success": True, "champion": champion})

    @app.route("/api/step/7/report")
    def step7_report():
        """Return the latest evaluation report (markdown + metrics JSON + pdf path)."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        out_dir = current_project["steps"].get("7", {}).get("outputs", {}).get("out_dir", "")
        if not out_dir or not os.path.isdir(out_dir):
            return jsonify({"ready": False, "message": "No evaluation has been run yet."})

        md_path = os.path.join(out_dir, "report.md")
        json_path = os.path.join(out_dir, "metrics.json")
        pdf_path = os.path.join(out_dir, "report.pdf")

        md = None
        metrics = None
        if os.path.isfile(md_path):
            with open(md_path) as f:
                md = f.read()
        if os.path.isfile(json_path):
            with open(json_path) as f:
                try:
                    metrics = json.load(f)
                except Exception:
                    metrics = None

        return jsonify({
            "ready": bool(md),
            "markdown": md,
            "metrics": metrics,
            "pdf_exists": os.path.isfile(pdf_path),
            "pdf_path": pdf_path if os.path.isfile(pdf_path) else None,
            "out_dir": out_dir,
        })

    @app.route("/api/step/7/pdf")
    def step7_pdf():
        """Serve the most recent evaluation PDF for download."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        pdf = current_project["steps"].get("7", {}).get("outputs", {}).get("report_pdf", "")
        if not pdf or not os.path.isfile(pdf):
            return jsonify({"error": "No PDF available"}), 404
        from flask import send_file
        return send_file(pdf, as_attachment=True, download_name=os.path.basename(pdf))

    @app.route("/api/step/7/copy_pdf", methods=["POST"])
    def step7_copy_pdf():
        """Copy the evaluation PDF to a user-provided folder."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        data = request.get_json(force=True) or {}
        dst_dir = (data.get("dst_dir") or "").strip()
        if not dst_dir:
            return jsonify({"error": "dst_dir is required"}), 400
        pdf = current_project["steps"].get("7", {}).get("outputs", {}).get("report_pdf", "")
        if not pdf or not os.path.isfile(pdf):
            return jsonify({"error": "No PDF to copy yet"}), 404
        try:
            os.makedirs(dst_dir, exist_ok=True)
            import shutil
            dst = os.path.join(dst_dir, os.path.basename(pdf))
            shutil.copy2(pdf, dst)
            return jsonify({"ok": True, "dst": dst})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/step/8/manifest")
    def step8_manifest():
        """Return the inference gallery manifest."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        manifest_path = current_project["steps"].get("8", {}).get("outputs", {}).get("manifest", "")
        if not manifest_path or not os.path.isfile(manifest_path):
            return jsonify({"ready": False})
        with open(manifest_path) as f:
            data = json.load(f)
        data["ready"] = True
        return jsonify(data)

    @app.route("/api/step/8/image")
    def step8_image():
        """Stream a preview image from the latest inference run."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        out_dir = current_project["steps"].get("8", {}).get("outputs", {}).get("out_dir", "")
        if not out_dir:
            return jsonify({"error": "no inference run"}), 404
        sub = request.args.get("path", "")
        # Confine to out_dir to keep this safe.
        full = os.path.realpath(os.path.join(out_dir, sub))
        if not full.startswith(os.path.realpath(out_dir)) or not os.path.isfile(full):
            return jsonify({"error": "not found"}), 404
        from flask import send_file
        return send_file(full)

    @app.route("/api/step/8/verify", methods=["POST"])
    def step8_verify():
        """Record a yes/no verdict on an inference result (quick review UI)."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        data = request.get_json(force=True) or {}
        filename = data.get("filename", "")
        verdict = data.get("verdict", "")  # yes | no | skip
        notes = data.get("notes", "")
        out_dir = current_project["steps"].get("8", {}).get("outputs", {}).get("out_dir", "")
        if not out_dir:
            return jsonify({"error": "no inference run"}), 400
        if verdict not in ("yes", "no", "skip"):
            return jsonify({"error": "verdict must be yes|no|skip"}), 400
        log_path = os.path.join(out_dir, "verifications.csv")
        is_new = not os.path.isfile(log_path)
        with open(log_path, "a", newline="") as f:
            import csv as _csv
            w = _csv.writer(f)
            if is_new:
                w.writerow(["filename", "verdict", "notes", "recorded_at"])
            w.writerow([filename, verdict, notes, datetime.now().isoformat()])
        return jsonify({"ok": True, "log_path": log_path})

    # ------------------------------------------------------------------
    # SAM3 driver status (orchestrator-driven)
    # ------------------------------------------------------------------
    @app.route("/api/step/5/sam3_status")
    def step5_sam3_status():
        # These phases all claim a live step-5 sub-app process. The driver
        # thread reports its own failures, but it cannot speak for the sub-app
        # dying after review_ready (or mid-drive between its HTTP polls).
        # waiting_for_sub_app is included: the process exists from launch, and
        # dying DURING model load was the original 2026-08-13 incident window.
        return jsonify(_reconcile_liveness(
            sam3_lock, sam3_status,
            ("review_ready", "segmenting", "configuring",
             "waiting_for_sub_app"), 5,
            "Step 5 segmentation sub-app"))

    @app.route("/api/step/5/kick", methods=["POST"])
    def step5_kick():
        """Head-start SAM3. Called by step 4 sub-app after each batch export
        (when auto_start_sam3 is on) and by any other orchestrator-side flow
        that wants to nudge segmentation forward.

        - If step 5 sub-app is not running: launch it + start the SAM3 driver.
        - If it's running and the driver is idle/done: re-run configure+process
          so newly exported frames get picked up.
        - If the driver is still working: no-op (the sub-app's queue is rebuilt
          on the next cycle; new frames will be picked up then).
        """
        global sam3_thread
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400

        s5 = current_project["steps"].get("5")
        if not s5:
            return jsonify({"error": "Step 5 not in project"}), 400

        # Case 1: sub-app not running — do a full launch (same as Start button).
        if not runner.is_running(5):
            cfg = s5["config"]
            step_dir = s5["dir"]
            os.makedirs(step_dir, exist_ok=True)
            s5["status"] = "running"
            s5["started_at"] = datetime.now().isoformat()
            pm.save_project(current_project)
            try:
                result = _run_step5(5, cfg, step_dir)
                if "error" in result:
                    s5["status"] = "error"
                    pm.save_project(current_project)
                    return jsonify(result), 500
                return jsonify({"launched": True, **result})
            except Exception as e:
                s5["status"] = "error"
                pm.save_project(current_project)
                return jsonify({"error": str(e)}), 500

        # Case 2: sub-app running, driver thread still active — nothing to do.
        if sam3_thread is not None and sam3_thread.is_alive():
            return jsonify({"status": "already_driving"})

        # Case 3: sub-app running, driver idle — re-kick with fresh configure.
        port = runner.ports.get(5)
        if not port:
            return jsonify({"error": "No port recorded for running step 5"}), 500
        cfg = s5["config"]
        step_dir = s5["dir"]
        input_dir = cfg.get("input_dir") or current_project["steps"].get("4", {}).get("dir", "")
        categories = cfg.get("categories") or ["Target species only"]
        batch_size = int(cfg.get("review_batch_size") or 10)
        _sam3_set(phase="re_kicking", running=True, error=None,
                  message="New frames exported — re-configuring SAM3 queue...")
        sam3_thread = threading.Thread(
            target=_sam3_drive,
            args=(port, input_dir, step_dir, categories, batch_size),
            daemon=True,
        )
        sam3_thread.start()
        return jsonify({"status": "re_kicked"})

    # ------------------------------------------------------------------
    # step4test - combined annotator (place + SAM3 segment). Non-chain tile,
    # launched on its own port; NOT a STEP_KEYS project step. The int-keyed
    # /api/step/<int:step>/* routes 404 a string segment, so these are
    # dedicated string routes.
    # ------------------------------------------------------------------
    @app.route("/api/step/step4test/run", methods=["POST"])
    def step4test_run():
        global step4test_cfg
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        # Capture the #panel-step4test settings (Combined Step-4+5 config). The
        # body may be empty (legacy callers) - then the launch inherits Step-5.
        body = request.get_json(silent=True) or {}
        if isinstance(body.get("config"), dict):
            step4test_cfg = body["config"]
        try:
            result = _run_step4test(force_reroute=False)
            if isinstance(result, dict) and "error" in result:
                # run_flask_stage reports "Step <key> is already running", and the
                # ANNOTATOR and the ROUTING sub-app are different keys:
                #   step4test       -> the annotator process is up (possibly still
                #                      loading SAM3). Not a failure: hand the UI its
                #                      port and let it wait for health.
                #   step4test_route -> a ROUTING pass is in flight. That is NOT a
                #                      ready UI, and runner.ports["step4test"] here
                #                      is a STALE port from an earlier launch whose
                #                      process is long dead.
                # Matching the bare substring "already running" conflated the two and
                # answered ui_ready on that stale port, so the browser polled a dead
                # port for its whole deadline and reported "Combined Annotator did not
                # become ready". _run_step4test's in-flight guard now returns a routing
                # phase before this branch is ever reached; the narrowed match keeps a
                # routing-key error from ever being read as a live UI again.
                err = str(result.get("error", "")).lower()
                if "step4test is already running" in err:
                    # The error itself proves the annotator process is alive
                    # (run_flask_stage only raises it when poll() is None), so this
                    # port belongs to a running process - no health gate needed, and
                    # a mid-SAM3-load annotator still gets waited on by the UI.
                    port = runner.ports.get("step4test") or step4test_state.get("port")
                    if port:
                        return jsonify({"success": True, "already_running": True,
                                        "phase": "ui_ready", "ui_port": port})
                return jsonify(result), 500
            # Routing kicked (poll route_status) or instant reuse (ui_ready).
            with route_lock:
                phase = route_status.get("phase")
                ui_port = route_status.get("ui_port")
            payload = {"success": True, "phase": phase}
            if phase == "ui_ready" and ui_port:
                payload["ui_port"] = ui_port
            return jsonify(payload)
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    @app.route("/api/step/step4test/route_status")
    def step4test_route_status():
        # The routing phases are owned by the driver THREAD; if that thread
        # died without writing a phase (hard crash), the status would stick at
        # a routing phase forever with nothing checking it. route_thread.ident
        # filters the created-but-not-yet-started window.
        with route_lock:
            _phase = route_status.get("phase")
        if (_phase in ("routing", "routing_configure", "routing_ocr",
                       "exporting")
                and route_thread is not None
                and route_thread.ident is not None
                and not route_thread.is_alive()):
            _route_set(phase="error", ui_port=None, died=True,
                       error="Routing driver thread died mid-pass.",
                       message="Routing stopped unexpectedly. Reopen to "
                               "resume where it left off.")
        # ui_ready is the only phase claiming a live annotator process; the
        # setter stamps which mode's process owns it.
        return jsonify(_reconcile_liveness(
            route_lock, route_status, ("ui_ready",), "step4test",
            "Combined annotator"))

    @app.route("/api/step/step4test/cancel", methods=["POST"])
    def step4test_cancel():
        """Cancel an in-flight routing pass. Called when the operator closes the
        launch window, and by Stop.

        The UI is the only way this pipeline is ever launched, so a closed window
        must mean the work stops - not a routing sub-app left holding the GPU and
        later killing a live annotator to launch one nobody asked for.

        Cheap by construction: placePoints flushes detections every 50 frames and
        skips frames it already has, so reopening resumes rather than restarting.
        """
        route_cancel.set()
        # Kill the routing sub-app now rather than waiting for the driver thread to
        # notice: it is the thing holding the GPU.
        runner.kill("step4test_route")
        with route_lock:
            phase = route_status.get("phase")
        # Only rewrite the phase if a pass was actually underway - never clobber a
        # ui_ready/idle state.
        if phase in ("launching", "routing", "routing_configure", "routing_ocr",
                     "exporting", "route_ready"):
            _route_set(phase="cancelled", ui_port=None, error=None,
                       message="Routing cancelled. Reopening resumes where it stopped.")
        return jsonify({"success": True, "cancelled": True})

    @app.route("/api/step/step4test/status")
    def step4test_status():
        # runner keys by hashable step id; we use the string "step4test".
        status = runner.poll_status("step4test")
        port = runner.ports.get("step4test") or step4test_state.get("port")
        if port:
            status["healthy"] = runner.health_check(port)
            status["port"] = port
        return jsonify(status)

    @app.route("/api/step/step4test/log")
    def step4test_log():
        offset = request.args.get("offset", 0, type=int)
        lines, new_offset, dropped = runner.get_log("step4test", offset)
        return jsonify({"lines": lines, "offset": new_offset, "dropped": dropped})

    @app.route("/api/step/step4test/stop", methods=["POST"])
    def step4test_stop():
        """Stop everything this tile owns: the annotator AND any routing pass.

        Stopping used to kill only the annotator, leaving a routing pass running
        invisibly - which would then kill a later annotator to launch its own.
        Stop means stop.
        """
        route_cancel.set()
        runner.kill("step4test_route")
        runner.kill("step4test")
        with route_lock:
            phase = route_status.get("phase")
        if phase in ("launching", "routing", "routing_configure", "routing_ocr",
                     "exporting", "route_ready"):
            _route_set(phase="cancelled", ui_port=None, error=None,
                       message="Stopped. Reopening resumes where routing stopped.")
        else:
            # A deliberate Stop must not read as a crash: the liveness check
            # flips a lingering ui_ready over a dead process to a red
            # "process exited" error, so rewrite the phase first - but only
            # when step4test owns it (Edit Masks / Refine keep running).
            _route_mark_stopped_if_owner("step4test", "Annotator stopped.")
        step4test_state.update(running=False, project_id=None, input_dir=None)
        return jsonify({"success": True})

    @app.route("/api/step/step4test/folder", methods=["POST"])
    def step4test_folder():
        """Open the combined annotator's output dir in the file manager.

        Same guardrail as /api/fs/open: only opens paths inside the current
        project or the repo.
        """
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        out_dir = step4test_state.get("out_dir")
        if not out_dir:
            # Not launched yet - fall back to the canonical export dir.
            out_dir = _step4test_export_dir()
        path = os.path.abspath(out_dir)
        if not os.path.isdir(path):
            return jsonify({"error": f"not a directory: {path}"}), 400
        project_dir = os.path.abspath(current_project["project_dir"])
        repo_dir = os.path.abspath(REPO_DIR)
        if not (path.startswith(project_dir + os.sep) or path == project_dir
                or path.startswith(repo_dir + os.sep) or path == repo_dir):
            return jsonify({"error": "path outside project/repo"}), 403
        try:
            import subprocess
            subprocess.Popen(
                ["xdg-open", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return jsonify({"success": True, "path": path})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/step/step4test/reset", methods=["POST"])
    def step4test_reset():
        """Reset the combined annotator (4.test): stop its runners and delete the
        whole step4test_combinedAnnotate dir (routed_input + all_images/all_labels/
        segmentations) so the next Open re-routes from Step 3 and starts fresh.

        Same project-dir guardrail as /folder: only a path inside the current
        project is ever removed.
        """
        global step4test_cfg
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        # Kill both runners (the headless routing pass + the combined UI) so
        # nothing is mid-write while we remove the tree.
        runner.kill("step4test")
        runner.kill("step4test_route")
        project_dir = os.path.abspath(current_project["project_dir"])
        out_dir = os.path.abspath(_step4test_export_dir())
        # Guardrail: the target must live strictly under the project dir.
        if not out_dir.startswith(project_dir + os.sep):
            return jsonify({"error": "path outside project"}), 403
        try:
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        # Reset transient state so the panel reflects a clean slate.
        step4test_state.update(running=False, port=None,
                               started_at=None, out_dir=None,
                               project_id=None, input_dir=None)
        step4test_cfg = {}
        _route_set(phase="idle", processed=0, total=0, ui_port=None, error=None,
                   warnings=[], dropped=[], message="")
        vic.note("[orch] step4test reset - removed step4test_combinedAnnotate")
        return jsonify({"success": True, "state": current_project})

    # ------------------------------------------------------------------
    # ---- step4loop (Refine loop) string routes ----
    #
    # Active-learning review loop: Step 8 inference writes predictions.json;
    # this tile seeds those detections into the SAME step4test export
    # (segmentations.json) as pending model masks via seed_from_predictions.py,
    # then opens the combined annotator in resume mode so the reviewer accepts/
    # rejects/edits them like any other mask. Non-chain string routes for the
    # same reason step4test's are: /api/step/<int:step>/* can't hold "step4loop".
    # ------------------------------------------------------------------

    def _loop_rounds_dir():
        return os.path.join(_step4test_export_dir(), "loop_rounds")

    @app.route("/api/step/step4loop/inference_runs")
    def step4loop_inference_runs():
        """List step8_inference/*/ dirs that contain a predictions.json, newest
        first, so the Refine panel can offer a run to seed from."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        step8_dir = current_project["steps"].get("8", {}).get("dir", "")
        runs = []
        if step8_dir and os.path.isdir(step8_dir):
            for nm in os.listdir(step8_dir):
                rdir = os.path.join(step8_dir, nm)
                pred_path = os.path.join(rdir, "predictions.json")
                if not os.path.isfile(pred_path):
                    continue
                n_items = 0
                generated_at = None
                try:
                    with open(pred_path) as f:
                        data = json.load(f)
                    n_items = len(data.get("items") or [])
                    generated_at = data.get("generated_at")
                except Exception:
                    pass
                runs.append({
                    "dir": rdir,
                    "name": nm,
                    "n_items": n_items,
                    "generated_at": generated_at,
                    "_mtime": os.path.getmtime(pred_path),
                })
        runs.sort(key=lambda r: r["_mtime"], reverse=True)
        for r in runs:
            r.pop("_mtime", None)
        return jsonify({"runs": runs})

    @app.route("/api/step/step4loop/run", methods=["POST"])
    def step4loop_run():
        """Seed a chosen inference run's predictions.json into the step4test
        export as pending review masks, then open the combined annotator in
        resume mode against exactly those (plus any prior) segmentations.

        Fail-closed on the frozen split (D8): the seeder requires
        {step6_dir}/dataset/split_manifest.json so it never seeds a frame the
        training split pinned to valid/test. Train a model (Step 6) before
        this route will do anything.
        """
        global step4loop_cfg
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400

        body = request.get_json(silent=True) or {}
        cfg = body.get("config") if isinstance(body.get("config"), dict) else {}
        step4loop_cfg = cfg

        predictions_dir = (cfg.get("predictions_dir") or "").strip()
        if not predictions_dir or not os.path.isdir(predictions_dir):
            return jsonify({"success": False,
                            "error": f"predictions_dir not found: {predictions_dir}"}), 400
        predictions_dir = os.path.abspath(predictions_dir)
        step8_dir = os.path.abspath(current_project["steps"].get("8", {}).get("dir", "") or "")
        if step8_dir and not (predictions_dir == step8_dir
                              or predictions_dir.startswith(step8_dir + os.sep)):
            return jsonify({"success": False,
                            "error": "predictions_dir outside this project's step 8 output"}), 400
        predictions_path = os.path.join(predictions_dir, "predictions.json")
        if not os.path.isfile(predictions_path):
            return jsonify({"success": False,
                            "error": f"No predictions.json in {predictions_dir}. Re-run Step 8 (Inference)."}), 400

        # Fail-closed (D8): the seeder must never seed a frame the training
        # split already pinned to valid/test, so the split has to exist first.
        step6_dir = current_project["steps"].get("6", {}).get("dir", "")
        split_manifest = os.path.join(step6_dir, "dataset", "split_manifest.json") if step6_dir else ""
        if not split_manifest or not os.path.isfile(split_manifest):
            return jsonify({"success": False, "error": (
                "No frozen split yet: train a model (Step 5) before Refine so "
                "the holdout is pinned."
            )}), 400

        codes_csv = _resolve_master_codes()
        py = PYTHON_PATHS.get(5, "python3")
        if not os.path.isfile(py):
            py = "python3"
        seeder = os.path.join(REPO_DIR, "scripts", "TCRMPclip_combinedAnnotate",
                              "src", "seed_from_predictions.py")
        if not os.path.isfile(seeder):
            return jsonify({"success": False, "error": f"seeder script not found: {seeder}"}), 500

        export_dir = _step4test_export_dir()
        os.makedirs(export_dir, exist_ok=True)

        conf_min = cfg.get("conf_min", 0.25)
        max_frames = cfg.get("max_frames", 0)
        skip_empty = bool(cfg.get("skip_empty", False))

        cmd = [
            py, seeder,
            "--predictions", predictions_path,
            "--export_dir", export_dir,
            "--codes_csv", codes_csv,
            "--split_manifest", split_manifest,
            "--conf_min", str(conf_min),
            "--max_frames", str(max_frames),
        ]
        if skip_empty:
            cmd.append("--skip_empty")
        lores_manifest = os.path.join(REPO_DIR, "supporting_data", "TCRMP_clip_lores",
                                      "lores_manifest.csv")
        if os.path.isfile(lores_manifest):
            cmd += ["--lores_manifest", lores_manifest]

        import subprocess as _sp
        try:
            proc = _sp.run(cmd, capture_output=True, text=True, timeout=300)
        except _sp.TimeoutExpired:
            return jsonify({"success": False, "error": "Seeder timed out after 300s"}), 500
        except Exception as exc:
            return jsonify({"success": False, "error": f"Seeder failed to launch: {exc}"}), 500

        if proc.returncode != 0:
            return jsonify({"success": False, "error": (
                f"Seeder exited {proc.returncode}: "
                f"{(proc.stderr or proc.stdout or '').strip()[-2000:]}"
            )}), 500

        # The seeder prints exactly one JSON line (its summary) to stdout.
        summary = None
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                summary = json.loads(line)
            except Exception:
                continue
            break
        if summary is None:
            return jsonify({"success": False, "error": (
                f"Seeder produced no parseable summary. stdout tail: {proc.stdout[-2000:]}"
            )}), 500

        vic.note(
            f"[orch] step4loop seeded frames={summary.get('seeded_frames')} "
            f"masks={summary.get('seeded_masks')} "
            f"skipped_holdout={summary.get('skipped_holdout')} "
            f"skipped_existing={summary.get('skipped_existing')} "
            f"skipped_below_conf={summary.get('skipped_below_conf')}"
        )

        if not summary.get("seeded_frames"):
            return jsonify({"success": False, "seeded": summary, "error": (
                "Nothing new to review (all frames already in the labelset, or "
                "all remaining frames are on a held-out transect)."
            )}), 400

        # Preflight for the resume-mode launch: /api/resume reads only
        # segmentations.json (never sam_click_prompts.json), so the only thing
        # to assert here is that the seeder actually wrote it.
        seg_path = os.path.join(export_dir, "segmentations", "segmentations.json")
        if not os.path.isfile(seg_path) or os.path.getsize(seg_path) == 0:
            return jsonify({"success": False, "error": (
                f"Seeder reported {summary.get('seeded_frames')} seeded frames but "
                f"{seg_path} is missing or empty."
            )}), 500

        try:
            result = _launch_step4test_ui(mode="step4loop", cfg=cfg)
        except Exception as e:
            return jsonify({"success": False, "error": str(e),
                            "traceback": traceback.format_exc()}), 500
        if isinstance(result, dict) and result.get("error"):
            if "already running" in str(result.get("error", "")).lower():
                port = runner.ports.get("step4loop") or step4loop_state.get("port")
                return jsonify({"success": True, "seeded": summary, "already_running": True,
                                "phase": "ui_ready", "ui_port": port})
            return jsonify({"success": False, "error": result["error"]}), 500

        port = runner.ports.get("step4loop") or step4loop_state.get("port")
        return jsonify({"success": True, "seeded": summary,
                        "phase": "ui_ready", "ui_port": port})

    @app.route("/api/step/step4loop/status")
    def step4loop_status():
        status = runner.poll_status("step4loop")
        port = runner.ports.get("step4loop") or step4loop_state.get("port")
        if port:
            status["healthy"] = runner.health_check(port)
            status["port"] = port
        return jsonify(status)

    @app.route("/api/step/step4loop/log")
    def step4loop_log():
        offset = request.args.get("offset", 0, type=int)
        lines, new_offset, dropped = runner.get_log("step4loop", offset)
        return jsonify({"lines": lines, "offset": new_offset, "dropped": dropped})

    @app.route("/api/step/step4loop/stop", methods=["POST"])
    def step4loop_stop():
        runner.kill("step4loop")
        _route_mark_stopped_if_owner("step4loop", "Refine annotator stopped.")
        step4loop_state.update(running=False, project_id=None, input_dir=None)
        return jsonify({"success": True})

    @app.route("/api/step/step4loop/folder", methods=["POST"])
    def step4loop_folder():
        """Open loop_rounds/ (or the export dir if it doesn't exist yet) in the
        file manager. Same guardrail as step4test's /folder."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        path = os.path.abspath(_loop_rounds_dir())
        if not os.path.isdir(path):
            path = os.path.abspath(_step4test_export_dir())
        if not os.path.isdir(path):
            return jsonify({"error": f"not a directory: {path}"}), 400
        project_dir = os.path.abspath(current_project["project_dir"])
        repo_dir = os.path.abspath(REPO_DIR)
        if not (path.startswith(project_dir + os.sep) or path == project_dir
                or path.startswith(repo_dir + os.sep) or path == repo_dir):
            return jsonify({"error": "path outside project/repo"}), 403
        try:
            import subprocess
            subprocess.Popen(
                ["xdg-open", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return jsonify({"success": True, "path": path})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/step/step4loop/reset", methods=["POST"])
    def step4loop_reset():
        """Surgically undo unreviewed seeding, never rmtree.

        For every loop_rounds/round_*.json manifest: for each seeded frame that
        is still unexported, drop the PENDING model-sourced masks this round
        added (matched by id); if that empties the frame's mask list AND it was
        never reviewed, remove the seg entry entirely (plus its routed_input/
        raw/ symlink, only if it really is a symlink - never touch a real
        file). Frames already exported, or carrying any accepted/manual mask,
        are left completely alone and reported back as `kept` so the user can
        see what reset intentionally did not touch. Round manifests that were
        fully consumed by this pass are deleted; segmentations.json is written
        atomically (tmp + os.replace).
        """
        global step4loop_cfg
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        runner.kill("step4loop")
        _route_mark_stopped_if_owner("step4loop", "Refine annotator stopped.")

        export_dir = os.path.abspath(_step4test_export_dir())
        project_dir = os.path.abspath(current_project["project_dir"])
        if not export_dir.startswith(project_dir + os.sep):
            return jsonify({"error": "path outside project"}), 403

        rounds_dir = os.path.join(export_dir, "loop_rounds")
        seg_path = os.path.join(export_dir, "segmentations", "segmentations.json")

        dropped_masks = 0
        removed_frames = []
        kept_frames = []
        consumed_manifests = []

        seg_dict = {}
        if os.path.isfile(seg_path):
            try:
                with open(seg_path) as f:
                    seg_dict = json.load(f)
            except Exception as exc:
                return jsonify({"error": f"could not read segmentations.json: {exc}"}), 500

        round_manifests = sorted(glob.glob(os.path.join(rounds_dir, "round_*.json"))) \
            if os.path.isdir(rounds_dir) else []

        for manifest_path in round_manifests:
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
            except Exception:
                continue
            seeded = manifest.get("seeded") or {}
            manifest_fully_consumed = True

            for filename, mask_ids in seeded.items():
                seg = seg_dict.get(filename)
                if seg is None:
                    continue  # already removed by an earlier pass/manifest
                if seg.get("exported"):
                    kept_frames.append(filename)
                    manifest_fully_consumed = False
                    continue

                mask_id_set = set(mask_ids or [])
                before = seg.get("masks") or []
                keep_masks = []
                for m in before:
                    is_seeded_pending_model = (
                        m.get("id") in mask_id_set
                        and m.get("source_type") == "model"
                        and m.get("status") == "pending"
                    )
                    if is_seeded_pending_model:
                        dropped_masks += 1
                    else:
                        keep_masks.append(m)
                seg["masks"] = keep_masks

                if keep_masks or seg.get("reviewed"):
                    # Either a real (accepted/manual/other) mask remains, or a
                    # human already reviewed this frame - never silently delete
                    # review work. Leave the (now-trimmed) entry in place.
                    kept_frames.append(filename)
                    manifest_fully_consumed = False
                    continue

                # No masks left, never reviewed, and this frame was created by
                # seeding: remove the seg entry and its raw/ symlink.
                del seg_dict[filename]
                removed_frames.append(filename)
                link_path = os.path.join(export_dir, "routed_input", "raw", filename)
                if os.path.islink(link_path):
                    try:
                        os.unlink(link_path)
                    except OSError:
                        pass

            if manifest_fully_consumed:
                consumed_manifests.append(manifest_path)

        if round_manifests:
            os.makedirs(os.path.dirname(seg_path), exist_ok=True)
            import tempfile as _tempfile
            fd, tmp_path = _tempfile.mkstemp(
                dir=os.path.dirname(seg_path), suffix=".tmp")
            os.close(fd)
            with open(tmp_path, "w") as f:
                json.dump(seg_dict, f, indent=2)
            os.replace(tmp_path, seg_path)

            for mp in consumed_manifests:
                try:
                    os.remove(mp)
                except OSError:
                    pass

        step4loop_state.update(running=False, port=None,
                               started_at=None, out_dir=None,
                               project_id=None, input_dir=None)
        step4loop_cfg = {}
        vic.note(
            f"[orch] step4loop reset - dropped {dropped_masks} pending model "
            f"mask(s), removed {len(removed_frames)} frame(s), "
            f"kept {len(kept_frames)} frame(s), "
            f"consumed {len(consumed_manifests)} round manifest(s)"
        )
        return jsonify({
            "success": True,
            "dropped_masks": dropped_masks,
            "removed_frames": sorted(set(removed_frames)),
            "kept_frames": sorted(set(kept_frames)),
            "consumed_manifests": [os.path.basename(m) for m in consumed_manifests],
        })

    # ------------------------------------------------------------------
    # ---- editmasks (standalone Edit Masks) string routes ----
    #
    # Opens TCRMPclip_editMasks against the CURRENT project's step4test export
    # so a reviewer can relabel/fix masks Step 4 already wrote, without
    # re-running Step 4. Unlike step4loop there is no seeding pass: /run just
    # preflights that the export has masks to edit, then launches. Non-chain
    # string routes for the same reason step4test's/step4loop's are:
    # /api/step/<int:step>/* can't hold "editmasks".
    # ------------------------------------------------------------------

    @app.route("/api/step/editmasks/run", methods=["POST"])
    def editmasks_run():
        """Preflight + launch. The export must already have masks to edit
        (segmentations.json from a prior Step 4 session) - editmasks never
        creates a fresh export the way step4test/step4loop can."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400

        export_dir = _step4test_export_dir()
        seg_path = os.path.join(export_dir, "segmentations", "segmentations.json")
        if not os.path.isfile(seg_path):
            return jsonify({"success": False, "error": (
                "No masks to edit yet: run Step 4 first."
            )}), 400

        try:
            result = _launch_step4test_ui(mode="editmasks")
        except Exception as e:
            return jsonify({"success": False, "error": str(e),
                            "traceback": traceback.format_exc()}), 500
        if isinstance(result, dict) and result.get("error"):
            if "already running" in str(result.get("error", "")).lower():
                port = runner.ports.get("editmasks") or editmasks_state.get("port")
                return jsonify({"success": True, "already_running": True,
                                "phase": "ui_ready", "ui_port": port})
            return jsonify({"success": False, "error": result["error"]}), 500

        port = runner.ports.get("editmasks") or editmasks_state.get("port")
        return jsonify({"success": True, "phase": "ui_ready", "ui_port": port})

    @app.route("/api/step/editmasks/status")
    def editmasks_status():
        status = runner.poll_status("editmasks")
        port = runner.ports.get("editmasks") or editmasks_state.get("port")
        if port:
            status["healthy"] = runner.health_check(port)
            status["port"] = port
        return jsonify(status)

    @app.route("/api/step/editmasks/log")
    def editmasks_log():
        offset = request.args.get("offset", 0, type=int)
        lines, new_offset, dropped = runner.get_log("editmasks", offset)
        return jsonify({"lines": lines, "offset": new_offset, "dropped": dropped})

    @app.route("/api/step/editmasks/stop", methods=["POST"])
    def editmasks_stop():
        runner.kill("editmasks")
        _route_mark_stopped_if_owner("editmasks", "Edit Masks stopped.")
        editmasks_state.update(running=False, project_id=None, input_dir=None)
        return jsonify({"success": True})

    @app.route("/api/step/editmasks/folder", methods=["POST"])
    def editmasks_folder():
        """Open the step4test export dir (the masks Edit Masks operates on) in
        the file manager. Same guardrail as step4test's/step4loop's /folder."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        path = os.path.abspath(_step4test_export_dir())
        if not os.path.isdir(path):
            return jsonify({"error": f"not a directory: {path}"}), 400
        project_dir = os.path.abspath(current_project["project_dir"])
        repo_dir = os.path.abspath(REPO_DIR)
        if not (path.startswith(project_dir + os.sep) or path == project_dir
                or path.startswith(repo_dir + os.sep) or path == repo_dir):
            return jsonify({"error": "path outside project/repo"}), 403
        try:
            import subprocess
            subprocess.Popen(
                ["xdg-open", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return jsonify({"success": True, "path": path})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/step/editmasks/reset", methods=["POST"])
    def editmasks_reset():
        """Stop the Edit Masks service and clear its transient launch state.

        Unlike step4test's/step4loop's reset, this NEVER touches the export
        dir or segmentations.json: editmasks only ever opens the export Step 4
        (and Refine) already built, it never owns that data, so "reset" here
        means only "stop the service and forget the port" - the masks
        themselves are untouched.
        """
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        runner.kill("editmasks")
        _route_mark_stopped_if_owner("editmasks", "Edit Masks stopped.")
        editmasks_state.update(running=False, port=None,
                               started_at=None, out_dir=None,
                               project_id=None, input_dir=None)
        vic.note("[orch] editmasks reset - stopped service, export left untouched")
        return jsonify({"success": True, "state": current_project})

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    return app

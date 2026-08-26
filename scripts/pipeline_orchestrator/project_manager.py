"""
Project manager — create, load, save, auto-link, and reset pipeline projects.
"""

import os
import json
import glob
import shutil
import copy
from datetime import datetime

from orchestrator_config import REPO_DIR, DEFAULT_STEP_CONFIGS

STEP_DIRS = {
    "1": "step1_makeAllPoints",
    "2": "step2_recodeSpecies",
    "3": "step3_chooseImages",
    "4": "step4_routeChosenImages",
    "5": "step5_segmentImages",
    "6": "step6_trainModel",
    "7": "step7_evaluateModel",
    "8": "step8_inference",
    # The 4.test combined annotator (replaces Steps 4+5) writes its YOLO export
    # and segmentations/ here. Resolvable via resolve_step_dir(step="step4test").
    # NOT a numbered chain step (absent from STEP_KEYS), so it is not auto-created
    # at project setup; 4.test creates it on first routing/export.
    "step4test": "step4test_combinedAnnotate",
}

STEP_NAMES = {
    "1": "Make All Points",
    "2": "Recode Species",
    "3": "Choose Images",
    "4": "Place Points",
    "5": "Segment Images (SAM3)",
    "6": "Train Model",
    "7": "Evaluate Model",
    "8": "Model Inference",
}

# Ordered list of step keys (used in multiple places to avoid hardcoding 5/6 bumps).
STEP_KEYS = ["1", "2", "3", "4", "5", "6", "7", "8"]


def find_projects(projects_root):
    """[(project_id, project_dir)] for every <projects_root>/*/project.json.

    project_id is the project.json 'id' (falls back to the dir name). Used to
    resolve a returned expert CSV's project back to its on-disk step dirs, and
    to enumerate projects for the multi-project review-site dropdown."""
    out = []
    try:
        entries = sorted(os.listdir(projects_root))
    except (FileNotFoundError, NotADirectoryError):
        return out
    for entry in entries:
        pj = os.path.join(projects_root, entry, 'project.json')
        if not os.path.isfile(pj):
            continue
        try:
            with open(pj) as f:
                pid = (json.load(f).get('id') or entry)
        except Exception:
            pid = entry
        out.append((pid, os.path.join(projects_root, entry)))
    return out


def resolve_step_dir(project_id, projects_root, step="5"):
    """Map a canonical project_id to its <project_dir>/<step dir>, or '' if no
    project under projects_root has that id."""
    if not project_id:
        return ""
    for pid, pdir in find_projects(projects_root):
        if pid == project_id:
            return os.path.join(pdir, STEP_DIRS.get(str(step), ""))
    return ""


def create_project(name, base_dir):
    """Create a new project directory with project.json and step subdirs.

    One folder per project: the folder is named ``YY-MM-DD_{name}`` with NO
    random hash suffix, so a project has a single, stable, predictable home.
    If a folder of that name already exists, raise — the caller should Open the
    existing project instead of minting a duplicate (this is what produced the
    pile of ``{name}_{date}_{hash}`` folders and made exports land in the wrong
    one)."""
    name = name.strip().replace(" ", "_")
    if not name:
        raise ValueError("Project name cannot be empty")

    base_dir = os.path.abspath(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    date_str = datetime.now().strftime("%y-%m-%d")
    dir_name = f"{date_str}_{name}"
    project_dir = os.path.join(base_dir, dir_name)
    if os.path.exists(project_dir):
        raise ValueError(
            f"A project folder already exists: {dir_name}. "
            f"Open it instead of creating a new one.")
    os.makedirs(project_dir)

    # Create step subdirs (numbered chain steps only; step4test_combinedAnnotate
    # is created lazily by the 4.test routing/export, not at project setup).
    for s in STEP_KEYS:
        os.makedirs(os.path.join(project_dir, STEP_DIRS[s]), exist_ok=True)

    now = datetime.now().isoformat()
    state = {
        "id": dir_name,
        "name": name,
        "description": "",
        "created_at": now,
        "updated_at": now,
        "project_dir": project_dir,
        "repo_dir": REPO_DIR,
        "current_step": 1,
        "steps": {},
    }

    for s in STEP_KEYS:
        state["steps"][s] = {
            "status": "pending" if s == "1" else "locked",
            "name": STEP_NAMES[s],
            "dir": os.path.join(project_dir, STEP_DIRS[s]),
            "config": copy.deepcopy(DEFAULT_STEP_CONFIGS.get(s, {})),
            "outputs": {},
        }

    save_project(state)
    return state


def load_project(project_dir):
    """Load project.json from a project directory and validate."""
    project_dir = os.path.abspath(project_dir)
    pj = os.path.join(project_dir, "project.json")
    if not os.path.isfile(pj):
        raise FileNotFoundError(f"No project.json in {project_dir}")

    with open(pj) as f:
        state = json.load(f)

    state["project_dir"] = project_dir

    # Back-compat: older projects (pre-step-6) don't have a steps.6 entry.
    # Graft one on so the UI + orchestrator logic can find it.
    for s in STEP_KEYS:
        if s not in state.get("steps", {}):
            state.setdefault("steps", {})[s] = {
                "status": "locked",
                "name": STEP_NAMES[s],
                "dir": os.path.join(project_dir, STEP_DIRS[s]),
                "config": copy.deepcopy(DEFAULT_STEP_CONFIGS.get(s, {})),
                "outputs": {},
            }

    # Validate completed steps still have outputs
    for s in STEP_KEYS:
        step = state["steps"].get(s, {})
        if step.get("status") == "completed":
            step_dir = os.path.join(project_dir, STEP_DIRS[s])
            if not os.path.isdir(step_dir) or not os.listdir(step_dir):
                step["status"] = "pending"

    # Unlock any step whose predecessor is completed. This catches the case
    # where a step was added later (e.g. step 6 grafted onto a pre-6 project):
    # if step 5 is completed, step 6 should be pending, not stuck locked.
    for i, s in enumerate(STEP_KEYS):
        if i == 0:
            continue
        prev = STEP_KEYS[i - 1]
        if (state["steps"][prev]["status"] == "completed"
                and state["steps"][s]["status"] == "locked"):
            state["steps"][s]["status"] = "pending"

    # Step 8 (inference) only requires a trained model from step 6; evaluating
    # (step 7) is optional. If step 6 is done, unlock step 8 regardless of 7.
    if (state["steps"].get("6", {}).get("status") == "completed"
            and state["steps"].get("8", {}).get("status") == "locked"):
        state["steps"]["8"]["status"] = "pending"

    # 4.test promotion bridge: the combined annotator (non-chain "step4test")
    # replaces Steps 4 + 5, so those chain slots never complete and would leave
    # Step 6 (train) permanently locked. Once Step 3 is done, unlock Step 6
    # directly so the promoted flow can reach it; _run_step6's preflight is the
    # real gate on the 4.test export contents.
    if (state["steps"].get("3", {}).get("status") == "completed"
            and state["steps"].get("6", {}).get("status") == "locked"):
        state["steps"]["6"]["status"] = "pending"

    # Reconcile current_step
    for s in STEP_KEYS:
        if state["steps"][s]["status"] not in ("completed",):
            state["current_step"] = int(s)
            break
    else:
        state["current_step"] = int(STEP_KEYS[-1])

    save_project(state)
    return state


def save_project(state):
    """Atomically save project.json."""
    state["updated_at"] = datetime.now().isoformat()
    pj = os.path.join(state["project_dir"], "project.json")
    tmp = pj + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, pj)


def auto_link_outputs(state, completed_step):
    """After step N completes, wire its outputs into step N+1 config."""
    cs = str(completed_step)
    ns = str(completed_step + 1)
    if ns not in state["steps"]:
        return

    step_dir = os.path.join(state["project_dir"], STEP_DIRS[cs])
    next_step = state["steps"][ns]

    if cs == "1":
        # Find all_points CSV (may have date suffix)
        ap_files = sorted(glob.glob(os.path.join(step_dir, "all_points*.csv")))
        # Match both the legacy `master_codes*.csv` name and the VICARIUS
        # metadata-library name `tcrmp_species_codes*.csv`. Either may land
        # in step_dir depending on whether step 1 ran end-to-end or was
        # linked via /api/step/1/link.
        mc_files = sorted(glob.glob(os.path.join(step_dir, "master_codes*.csv")))
        if not mc_files:
            mc_files = sorted(glob.glob(os.path.join(step_dir, "tcrmp_species_codes*.csv")))
        # `step1_link` writes the resolved master_codes path directly into
        # step1.outputs before calling complete_step → auto_link_outputs.
        # Prefer that explicit value when the on-disk glob misses.
        mc_from_outputs = state["steps"]["1"].get("outputs", {}).get("master_codes")
        if ap_files:
            next_step["config"]["all_points"] = ap_files[-1]
            state["steps"][cs]["outputs"]["all_points"] = ap_files[-1]
        mc_path = mc_files[-1] if mc_files else mc_from_outputs
        if mc_path:
            next_step["config"]["master_codes"] = mc_path
            state["steps"][cs]["outputs"]["master_codes"] = mc_path

    elif cs == "2":
        ap_files = sorted(glob.glob(os.path.join(step_dir, "all_points*.csv")))
        mc_files = sorted(glob.glob(os.path.join(step_dir, "master_codes*.csv")))
        if ap_files:
            next_step["config"]["all_points"] = ap_files[-1]
            state["steps"][cs]["outputs"]["all_points_recoded"] = ap_files[-1]
        if mc_files:
            next_step["config"]["master_codes"] = mc_files[-1]
            state["steps"][cs]["outputs"]["master_codes_recoded"] = mc_files[-1]

    elif cs == "3":
        sf = os.path.join(step_dir, "selected_frames.csv")
        if os.path.isfile(sf):
            next_step["config"]["selected_frames"] = sf
            state["steps"][cs]["outputs"]["selected_frames"] = sf
        # Also pass all_points for species enrichment
        ap = state["steps"]["2"].get("outputs", {}).get("all_points_recoded")
        if ap:
            next_step["config"]["all_points"] = ap
        # 4.test replaces Steps 4+5 (non-chain "step4test"), so completing Step 3
        # should make train (Step 6) reachable directly. _run_step6 preflights the
        # 4.test export, so unlocking here cannot train on missing data.
        s6 = state["steps"].get("6")
        if s6 is not None and s6.get("status") == "locked":
            s6["status"] = "pending"

    elif cs == "4":
        next_step["config"]["input_dir"] = step_dir
        state["steps"][cs]["outputs"]["export_dir"] = step_dir

    elif cs == "5":
        # Step 5's export dir contains all_images/, all_labels/, data.yaml —
        # which is exactly what step 6 needs as input.
        next_step["config"]["input_dir"] = step_dir
        state["steps"][cs]["outputs"]["export_dir"] = step_dir

    elif cs == "6":
        # Step 6's runs/ dir holds every training run; both step 7 (evaluate)
        # and step 8 (inference) need it. Step 8 doesn't require step 7 to
        # have run — you can run inference without evaluating — so unlock
        # both here.
        runs_dir = os.path.join(step_dir, "runs")
        state["steps"][cs]["outputs"]["runs_dir"] = runs_dir
        next_step["config"]["runs_dir"] = runs_dir
        s8 = state["steps"].get("8")
        if s8 is not None:
            s8.setdefault("config", {})["runs_dir"] = runs_dir
            if s8.get("status") == "locked":
                s8["status"] = "pending"

    elif cs == "7":
        # Step 7 doesn't strictly feed step 8 — step 8 picks a model from
        # step 6's runs/ directly — but we still make sure the path is wired.
        next_step["config"]["runs_dir"] = (
            state["steps"].get("6", {}).get("outputs", {}).get("runs_dir", "")
        )

    # Unlock next step
    if next_step["status"] == "locked":
        next_step["status"] = "pending"
    state["current_step"] = completed_step + 1

    save_project(state)


def complete_step(state, step_num):
    """Mark a step as completed, auto-link outputs, unlock next."""
    s = str(step_num)
    state["steps"][s]["status"] = "completed"
    state["steps"][s]["completed_at"] = datetime.now().isoformat()
    auto_link_outputs(state, step_num)
    save_project(state)


def reset_step(state, step_num):
    """Clear step outputs and lock it + all subsequent steps."""
    last = int(STEP_KEYS[-1])
    for s in range(step_num, last + 1):
        ss = str(s)
        if ss not in state["steps"]:
            continue
        step_dir = os.path.join(state["project_dir"], STEP_DIRS[ss])
        # Clear directory contents but keep the dir
        if os.path.isdir(step_dir):
            for item in os.listdir(step_dir):
                path = os.path.join(step_dir, item)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        state["steps"][ss]["status"] = "locked" if s > step_num else "pending"
        state["steps"][ss]["outputs"] = {}
        state["steps"][ss].pop("completed_at", None)
        state["steps"][ss].pop("started_at", None)

    state["current_step"] = step_num
    save_project(state)

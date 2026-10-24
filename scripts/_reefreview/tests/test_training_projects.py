#!/usr/bin/env python
"""Shared minting helper for training projects (_reefreview/training_projects.py).

No-pytest house harness. Nothing here launches a service, binds a port or
copies imagery: every path is a temp fixture removed in the finally block.

Covers:
  composite source  run dir naming, the relative step4test symlink, the
                    project.json contract the orchestrator and the matrix
                    already key on, README
  bbox source       project_type bbox_training, the training_source block, and
                    the rule that no imagery and no symlink ever enters the
                    run dir (non-TCRMP imagery stays where it is staged)
  collisions        a second mint on the same day takes the _2 suffix
  guards            bad initials, unknown source kind, missing source

Run with:
  env/bin/python scripts/_reefreview/tests/test_training_projects.py
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(os.path.dirname(_HERE))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _reefreview import training_projects as TP  # noqa: E402

_fail = 0


def check(cond, msg):
    global _fail
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _fail += 1


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


AST = timezone(timedelta(hours=-4))
TODAY = datetime.now(AST).strftime("%y-%m-%d")

root = tempfile.mkdtemp(prefix="training_projects_test_")
try:
    # A composite export, and a staged detection dataset outside inprocess.
    comp = os.path.join(root, "inprocess", "_composite", "comp_A")
    write(os.path.join(comp, "data.yaml"), "names:\n  0: PA\n")
    write(os.path.join(comp, "all_images", "SITE-20260101-T101.jpeg"), "x")
    inprocess = os.path.join(root, "inprocess")
    bbox = os.path.join(root, "supporting_data", "bbox_datasets", "reefset")
    write(os.path.join(bbox, "images", "img_000.jpg"), "x")
    write(os.path.join(bbox, "labels", "img_000.txt"), "0 0.5 0.5 0.2 0.3\n")

    # ---- composite source ------------------------------------------------
    print("\n-- composite source --")
    res = TP.mint_training_project(inprocess, "composite", "comp_A", comp, "lo",
                                   extra={"source_projects": [{"project_id": "26-01-01_LO",
                                                               "n_masks": 4}]})
    run_dir = res["run_dir"]
    check(os.path.basename(run_dir) == f"run_{TODAY}_LO_train_comp_A",
          f"run dir is run_YY-MM-DD_INITIALS_train_name with upper-cased initials (got {os.path.basename(run_dir)})")
    check(res["project_id"] == f"{TODAY}_LO_train_comp_A",
          f"project_id drops the run_ prefix (got {res['project_id']})")
    check(res["display_name"] == "TRAIN_comp_A", f"display name (got {res['display_name']})")
    link = os.path.join(run_dir, "step4test_combinedAnnotate")
    check(os.path.islink(link), "step4test_combinedAnnotate is a symlink into the training set")
    check(os.readlink(link) == os.path.join("..", "_composite", "comp_A"),
          f"the symlink is relative so the pair survives a move (got {os.readlink(link)})")
    check(os.path.realpath(link) == os.path.realpath(comp), "the symlink resolves to the composite")
    state = json.load(open(os.path.join(run_dir, "project.json")))
    check(state["project_type"] == "composite_training",
          f"project_type marks it a training project (got {state.get('project_type')})")
    check(state["current_step"] == 6, "the project opens on Train")
    check(state["id"] == res["project_id"] and state["name"] == "TRAIN_comp_A",
          "id and name match what was returned")
    check(state["initials"] == "LO" and state["project_dir"] == run_dir,
          "initials and project dir are recorded")
    check(state["repo_dir"] == os.path.dirname(_SCRIPTS),
          f"repo_dir is the github_repo root, as project_manager writes it (got {state['repo_dir']})")
    check(sorted(state["steps"]) == ["1", "2", "3", "4"],
          f"only steps 1 to 4 are written: 5 to 8 are grafted on with the orchestrator's "
          f"own defaults (got {sorted(state['steps'])})")
    check(all(s["status"] == "skipped" for s in state["steps"].values()),
          "every written step is skipped, not pending")
    check(all(os.path.isdir(s["dir"]) for s in state["steps"].values()),
          "each skipped step still has its folder, so the sidebar has somewhere to point")
    check(state["steps"]["3"]["name"] == "Choose Images",
          f"step names are the pipeline's own (got {state['steps']['3']['name']})")
    check(state["composite"]["name"] == "comp_A"
          and state["composite"]["path"] == os.path.join("inprocess", "_composite", "comp_A"),
          f"the composite block keeps the contract keys (got {state.get('composite')})")
    check(state["composite"]["source_projects"] == [{"project_id": "26-01-01_LO", "n_masks": 4}],
          "the source projects passed in are recorded")
    check("training_source" not in state,
          "a composite project keeps the shape the orchestrator and the matrix already read")
    check(len(state["access_log"]) == 1 and state["access_log"][0]["initials"] == "LO"
          and state["access_log"][0]["action"] == "create",
          f"the mint is one create entry in the access log (got {state.get('access_log')})")
    # Wording is byte-identical to _matrix/composite.py::create_training_project,
    # so the later delegation changes no file on disk.
    check(state["access_log"][0]["purpose"] == "training project for the training set comp_A",
          f"the access-log purpose matches the composite module's wording "
          f"(got {state['access_log'][0]['purpose']!r})")
    check(state["steps"]["1"]["note"] == "from training set comp_A",
          f"so does the skipped-step note (got {state['steps']['1']['note']!r})")
    readme = open(os.path.join(run_dir, "README.md")).read()
    check("comp_A" in readme and "step4test_combinedAnnotate" in readme,
          "the README names the training set and where its data lives")
    check("—" not in readme and "–" not in readme, "no em dashes or en dashes in the README")

    # ---- collision -------------------------------------------------------
    print("\n-- collision --")
    res2 = TP.mint_training_project(inprocess, "composite", "comp_A", comp, "LO")
    check(os.path.basename(res2["run_dir"]) == f"run_{TODAY}_LO_train_comp_A_2",
          f"a second mint the same day takes the _2 suffix (got {os.path.basename(res2['run_dir'])})")
    # The mkdir IS the claim: a name that appeared between the check and the
    # create is simply skipped, never a FileExistsError out of the mint.
    os.makedirs(os.path.join(inprocess, f"run_{TODAY}_LO_train_comp_A_3"))
    res3 = TP.mint_training_project(inprocess, "composite", "comp_A", comp, "LO")
    check(os.path.basename(res3["run_dir"]) == f"run_{TODAY}_LO_train_comp_A_4",
          f"a dir that already exists is skipped, not crashed on (got {os.path.basename(res3['run_dir'])})")
    check(issubclass(TP.RunDirExhausted, Exception),
          "RunDirExhausted is exported so a route can map it to its own status")
    # A full day raises RunDirExhausted rather than a bare ValueError.
    full = os.path.join(root, "full")
    os.makedirs(full)
    base = f"run_{TODAY}_ZZ_train_comp_A"
    for n in range(1, 100):
        os.makedirs(os.path.join(full, base if n == 1 else f"{base}_{n}"))
    try:
        TP.mint_training_project(full, "composite", "comp_A", comp, "zz")
        check(False, "a full day of run dirs raises RunDirExhausted")
    except TP.RunDirExhausted as e:
        check("archive one" in str(e), f"a full day of run dirs raises RunDirExhausted ({e})")

    # ---- bbox source -----------------------------------------------------
    print("\n-- bbox source --")
    bres = TP.mint_training_project(inprocess, "bbox", "reefset", bbox, "LO")
    brun = bres["run_dir"]
    check(os.path.basename(brun) == f"run_{TODAY}_LO_train_reefset",
          f"the same run dir shape (got {os.path.basename(brun)})")
    bstate = json.load(open(os.path.join(brun, "project.json")))
    check(bstate["project_type"] == "bbox_training",
          f"a detection dataset mints its own project type (got {bstate.get('project_type')})")
    check(bstate["training_source"] == {"kind": "bbox", "name": "reefset",
                                        "path": os.path.abspath(bbox),
                                        "task_default": "detect"},
          f"the training source block records the staged path and the default task "
          f"(got {bstate.get('training_source')})")
    check("composite" not in bstate, "a bbox project carries no composite block")
    check(not os.path.exists(os.path.join(brun, "step4test_combinedAnnotate")),
          "no step4test symlink: Step 6 builds dataset_detect from the staged source")
    entries = sorted(os.listdir(brun))
    check(entries == ["README.md", "project.json", "step1_makeAllPoints",
                      "step2_recodeSpecies", "step3_chooseImages", "step4_routeChosenImages"],
          f"the run dir holds no imagery at all, and no temp file survived the "
          f"atomic project.json write (got {entries})")
    check(bstate["access_log"][0]["purpose"] == "training project for the detection dataset reefset"
          and bstate["steps"]["1"]["note"] == "from detection dataset reefset",
          f"a bbox project says what it trains on in the same shape "
          f"(got {bstate['access_log'][0]['purpose']!r})")
    check(all(os.listdir(os.path.join(brun, d)) == [] for d in entries
              if d.startswith("step")),
          "every skipped step folder is empty: non-TCRMP imagery never enters a run dir")
    check(bstate["current_step"] == 6 and bstate["name"] == "TRAIN_reefset",
          "a bbox project opens on Train too")
    breadme = open(os.path.join(brun, "README.md")).read()
    check("reefset" in breadme and os.path.abspath(bbox) in breadme,
          "the README records where the staged dataset lives")

    # ---- shared predicate -------------------------------------------------
    print("\n-- shared predicate --")
    check(TP.is_training_project(state) and TP.is_training_project(bstate),
          "both kinds read as training projects")
    check(not TP.is_training_project({"name": "ordinary"}) and not TP.is_training_project(None),
          "an ordinary project, and no project at all, are not training projects")
    check(set(TP.TRAINING_PROJECT_TYPES) == {"composite_training", "bbox_training"},
          f"the shared list of training project types (got {TP.TRAINING_PROJECT_TYPES})")

    # ---- guards ----------------------------------------------------------
    print("\n-- guards --")
    for bad in ("", "12345", "toolong"):
        try:
            TP.mint_training_project(inprocess, "composite", "comp_A", comp, bad)
            check(False, f"initials {bad!r} are refused")
        except ValueError:
            check(True, f"initials {bad!r} are refused")
    try:
        TP.mint_training_project(inprocess, "video", "comp_A", comp, "LO")
        check(False, "an unknown source kind is refused")
    except ValueError as e:
        check("kind" in str(e).lower(), f"an unknown source kind is refused ({e})")
    try:
        TP.mint_training_project(inprocess, "bbox", "gone", os.path.join(root, "gone"), "LO")
        check(False, "a missing source is refused")
    except FileNotFoundError as e:
        check(True, f"a missing source is refused ({e})")
    before = set(os.listdir(inprocess))
    try:
        TP.mint_training_project(inprocess, "bbox", "gone", os.path.join(root, "gone"), "LO")
    except FileNotFoundError:
        pass
    check(set(os.listdir(inprocess)) == before, "a refused mint leaves no half-built run dir")
finally:
    shutil.rmtree(root, ignore_errors=True)

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)

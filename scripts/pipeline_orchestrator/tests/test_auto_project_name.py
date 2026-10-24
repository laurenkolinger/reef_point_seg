"""Auto project names (2026-08-26): for launcher-created projects flagged
``name_auto``, project.json ``name`` is rebuilt on every save as
``{INITIALS}_{SPECIES}_{N}img_{YY-MM-DD}`` (project_manager.auto_project_name
is the single source of truth), and GET /api/project/name-preview lets the
orchestrator UI preview the name for a candidate species pick without
mutating state. Hand-named legacy projects are never touched.
Run with github_repo/env/bin/python.
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

import os, sys, json, shutil, tempfile, subprocess
import datetime as _dt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

# Standing rule for orchestrator route suites: neuter subprocess.run for the
# whole module so nothing here can reach pkill/fuser and kill a live annotator,
# even though name-preview itself launches nothing.
class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""
subprocess.run = lambda *a, **k: _FakeCompleted()

# Deterministic lock state for every request except the explicit gate check.
os.environ["VICARIUS_LOCK_BYPASS"] = "1"

import project_manager as pm
import app as appmod

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1

CREATED = "2026-08-26T10:11:12.345678"
DATE = "26-08-26"
today = _dt.datetime.now().strftime("%y-%m-%d")

base = tempfile.mkdtemp(prefix="auto_name_")


def write_selected_frames(step3_dir, n, trailing_blank=False):
    os.makedirs(step3_dir, exist_ok=True)
    path = os.path.join(step3_dir, "selected_frames.csv")
    lines = ["frame_id,basename,species_present"]
    lines += [f"id{i},img{i},PA" for i in range(n)]
    body = "\n".join(lines) + "\n"
    if trailing_blank:
        body += "\n"
    with open(path, "w") as f:
        f.write(body)
    return path


def make_launcher_stub(run_id, name, initials="LO", name_auto=True, extra=None):
    """Mimic the project.json the VICARIUS launcher writes for a new run."""
    run_dir = os.path.join(base, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    stub = {
        "id": run_id,
        "name": name,
        "created_at": CREATED,
        "updated_at": CREATED,
        "project_dir": run_dir,
        "repo_dir": "/nonexistent/github_repo",
        "current_step": 1,
        "steps": {},
        "access_log": [{"ts": "2026-08-26T10:11:12-04:00", "initials": initials,
                        "action": "create", "purpose": "test"}],
    }
    if name_auto:
        stub["initials"] = initials
        stub["name_auto"] = True
    if extra:
        stub.update(extra)
    with open(os.path.join(run_dir, "project.json"), "w") as f:
        json.dump(stub, f, indent=2)
    return run_dir


def read_pj(run_dir):
    with open(os.path.join(run_dir, "project.json")) as f:
        return json.load(f)


try:
    # ---------------------------------------------------------------- unit: auto_project_name
    print("auto_project_name")
    check(pm.AUTO_NAME_MAX_SPECIES == 3, "AUTO_NAME_MAX_SPECIES is 3")
    check(pm.auto_project_name("LO", [], None, CREATED) == f"LO_{DATE}",
          "fresh project: LO_26-08-26")
    check(pm.auto_project_name("lo", [], None, CREATED) == f"LO_{DATE}",
          "initials uppercased")
    check(pm.auto_project_name("LO", ["PA", "SS"], None, CREATED) == f"LO_PA-SS_{DATE}",
          "species only: LO_PA-SS_26-08-26")
    check(pm.auto_project_name("LO", ["PA", "SS"], 714, CREATED) == f"LO_PA-SS_714img_{DATE}",
          "species + images: LO_PA-SS_714img_26-08-26")
    check(pm.auto_project_name("LO", [], 0, CREATED) == f"LO_0img_{DATE}",
          "zero images still shows 0img (file present, header only)")
    check(pm.auto_project_name("LO", ["PA", "SS", "MC", "OA", "DS"], None, CREATED)
          == f"LO_PA-SS-MC+2_{DATE}", ">3 species collapses to PA-SS-MC+2")
    check(pm.auto_project_name("LO", ["PA", "SS", "MC"], None, CREATED)
          == f"LO_PA-SS-MC_{DATE}", "exactly 3 species shown in full")
    check(pm.auto_project_name("LO", ["PA", "PA", "SS", " "], None, CREATED)
          == f"LO_PA-SS_{DATE}", "species iterable deduped + blanks dropped")
    check(pm.auto_project_name("l o/x", ["P A", "S$S"], None, CREATED)
          == f"L_O_X_P_A-S_S_{DATE}", "unsafe characters become _ in every part")
    check(pm.auto_project_name("LO", ["A.B+C-D"], None, CREATED) == f"LO_A.B+C-D_{DATE}",
          "dot, plus, and dash survive sanitization")
    check(pm.auto_project_name("LO", [], None, _dt.datetime(2025, 1, 2, 3, 4)) == "LO_25-01-02",
          "datetime created_at formats YY-MM-DD")
    check(pm.auto_project_name("LO", [], None, "not a date") == f"LO_{today}",
          "garbage created_at falls back to today")
    check(pm.auto_project_name("LO", [], None, None) == f"LO_{today}",
          "None created_at falls back to today")
    check(pm.auto_project_name("LO", [], "12", CREATED) == f"LO_12img_{DATE}",
          "numeric-string n_images coerced to int")

    # ---------------------------------------------------------------- unit: species_from_config
    print("species_from_config")
    check(pm.species_from_config("PA, SS,,PA , MC") == ["PA", "SS", "MC"],
          "split, strip, drop empties, dedupe, order preserved")
    check(pm.species_from_config("") == [], "empty string -> []")
    check(pm.species_from_config(None) == [], "None -> []")
    check(pm.species_from_config("  PA  ") == ["PA"], "single padded code")
    check(pm.species_from_config(["SS", "PA", "SS"]) == ["SS", "PA"], "list input accepted")

    # ---------------------------------------------------------------- unit: count_selected_frames
    print("count_selected_frames")
    s3 = os.path.join(base, "unit_step3")
    os.makedirs(s3)
    check(pm.count_selected_frames(s3) is None, "missing csv -> None")
    check(pm.count_selected_frames("") is None, "empty dir arg -> None")
    check(pm.count_selected_frames(None) is None, "None dir arg -> None")
    write_selected_frames(s3, 0)
    check(pm.count_selected_frames(s3) == 0, "header-only csv -> 0")
    write_selected_frames(s3, 5, trailing_blank=True)
    check(pm.count_selected_frames(s3) == 5, "5 rows + trailing blank line -> 5")
    with open(os.path.join(s3, "selected_frames.csv"), "a") as f:
        f.write("\n   \n")
    check(pm.count_selected_frames(s3) == 5, "extra whitespace-only lines ignored")
    bad = os.path.join(base, "unit_step3_bad")
    os.makedirs(os.path.join(bad, "selected_frames.csv"))  # a dir, not a file
    check(pm.count_selected_frames(bad) is None, "selected_frames.csv that is a dir -> None")
    if os.geteuid() != 0:
        unreadable = os.path.join(base, "unit_step3_unreadable")
        p = write_selected_frames(unreadable, 3)
        os.chmod(p, 0)
        check(pm.count_selected_frames(unreadable) is None, "unreadable csv -> None")
        os.chmod(p, 0o644)
    else:
        print("  SKIP unreadable csv check (running as root)")

    # ---------------------------------------------------------------- unit: refresh_auto_name
    print("refresh_auto_name")
    legacy = {"name": "my hand name", "created_at": CREATED, "initials": "LO",
              "steps": {"3": {"config": {"target_species": "PA"}}}}
    check(pm.refresh_auto_name(legacy) is False and legacy["name"] == "my hand name",
          "name_auto absent -> untouched, returns False")
    legacy["name_auto"] = False
    check(pm.refresh_auto_name(legacy) is False and legacy["name"] == "my hand name",
          "name_auto false -> untouched")

    no_ini = {"name": "x", "name_auto": True, "created_at": CREATED, "steps": {}}
    check(pm.refresh_auto_name(no_ini) is False and no_ini["name"] == "x",
          "name_auto with no initials anywhere -> no-op")

    fb = {"name": "x", "name_auto": True, "created_at": CREATED, "steps": {},
          "access_log": [{"action": "open_project", "initials": "ZZ"},
                         {"action": "create", "initials": "ab"},
                         {"action": "create", "initials": "CD"}]}
    check(pm.refresh_auto_name(fb) is True and fb["name"] == f"AB_{DATE}",
          "initials fall back to the FIRST access_log create entry (uppercased)")
    check(pm.refresh_auto_name(fb) is False, "second call with no change returns False")
    check(pm.initials_for_state({"initials": "lo"}) == "LO", "initials_for_state prefers project.json initials")

    # ---------------------------------------------------------------- save_project round trip
    print("save_project / load_project round trip")
    run_dir = make_launcher_stub(f"{DATE}_LO", f"LO_{DATE}")
    st = pm.load_project(run_dir)
    check(st.get("initials") == "LO" and st.get("name_auto") is True,
          "load_project preserves initials + name_auto")
    check(st["name"] == f"LO_{DATE}", f"fresh auto project keeps LO_{DATE} (got {st['name']!r})")
    st["steps"]["3"]["config"]["target_species"] = "PA, SS"
    pm.save_project(st)
    check(st["name"] == f"LO_PA-SS_{DATE}", "labels saved, Step 3 not run -> LO_PA-SS_26-08-26")
    check(read_pj(run_dir)["name"] == f"LO_PA-SS_{DATE}", "name persisted to project.json")
    write_selected_frames(st["steps"]["3"]["dir"], 714)
    pm.complete_step(st, 3)
    check(st["name"] == f"LO_PA-SS_714img_{DATE}", "complete_step(3) -> LO_PA-SS_714img_26-08-26")
    check(read_pj(run_dir)["name"] == f"LO_PA-SS_714img_{DATE}", "img count persisted")
    check(not os.path.exists(os.path.join(run_dir, "project.json.tmp")),
          "save stays atomic (no .tmp left behind)")
    pm.reset_step(st, 3)
    check(st["name"] == f"LO_PA-SS_{DATE}", "reset_step(3) removes the img part")
    check(read_pj(run_dir)["name"] == f"LO_PA-SS_{DATE}", "reset persisted")
    # steps.3.dir missing -> derived from project_dir
    st_nodir = read_pj(run_dir)
    st_nodir["steps"]["3"].pop("dir", None)
    write_selected_frames(os.path.join(run_dir, pm.STEP_DIRS["3"]), 3)
    # A csv next to a NON-completed Step 3 (reset left it pending) lends no count.
    check(pm.refresh_auto_name(st_nodir) is False and st_nodir["name"] == f"LO_PA-SS_{DATE}",
          "csv present but Step 3 not completed -> no img part")
    st_nodir["steps"]["3"]["status"] = "completed"
    check(pm.refresh_auto_name(st_nodir) is True and st_nodir["name"] == f"LO_PA-SS_3img_{DATE}",
          "missing steps.3.dir derives the Step 3 dir from project_dir")
    check(pm.n_images_for_state({"steps": {"3": {"status": "running", "dir": st["steps"]["3"]["dir"]}}}) is None,
          "running Step 3 -> n_images_for_state None")
    # Persist a completed Step 3 with the 3-row csv for the route checks below.
    pm.complete_step(st, 3)
    check(st["name"] == f"LO_PA-SS_3img_{DATE}", "re-completed Step 3 recounts -> 3img")

    # legacy hand-named project through the same save path
    legacy_dir = make_launcher_stub("legacy_hand", "my_old_project", name_auto=False)
    lst = pm.load_project(legacy_dir)
    lst["steps"]["3"]["config"]["target_species"] = "PA"
    write_selected_frames(lst["steps"]["3"]["dir"], 9)
    pm.complete_step(lst, 3)
    check(read_pj(legacy_dir)["name"] == "my_old_project", "legacy project name untouched by saves")
    check("name_auto" not in read_pj(legacy_dir), "legacy project never gains name_auto")

    # ---------------------------------------------------------------- routes
    print("/api/project/name-preview + config PUT")
    client = appmod.create_app().test_client()
    appmod.current_project = None
    r = client.get("/api/project/name-preview?species=PA")
    check(r.status_code == 400 and "error" in (r.get_json() or {}),
          f"no project loaded -> 400 JSON error (got {r.status_code})")

    appmod.current_project = pm.load_project(legacy_dir)
    r = client.get("/api/project/name-preview?species=PA,SS")
    j = r.get_json() or {}
    check(r.status_code == 200 and j == {"auto": False, "name": "my_old_project"},
          f"legacy project -> auto false + current name (got {j})")

    appmod.current_project = pm.load_project(run_dir)
    cur = appmod.current_project
    name_before = cur["name"]
    updated_before = cur["updated_at"]
    pj_mtime_before = os.path.getmtime(os.path.join(run_dir, "project.json"))
    check(name_before == f"LO_PA-SS_3img_{DATE}", f"loaded auto project name is current (got {name_before!r})")
    r = client.get("/api/project/name-preview?species=" + "MC%2CPA%2CSS%2COA%2CDS")
    j = r.get_json() or {}
    check(r.status_code == 200 and j.get("auto") is True, "auto project -> auto true")
    check(j.get("name") == f"LO_MC-PA-SS+2_{DATE}",
          f"preview of a DIFFERENT pick omits the old selection's img count (got {j.get('name')!r})")
    r = client.get("/api/project/name-preview?species=PA,SS")
    check((r.get_json() or {}).get("name") == f"LO_PA-SS_3img_{DATE}",
          f"preview of the saved pick keeps the img count (got {(r.get_json() or {}).get('name')!r})")
    r = client.get("/api/project/name-preview?species=SS,PA")
    check((r.get_json() or {}).get("name") == f"LO_SS-PA_3img_{DATE}",
          "saved-pick match is order-insensitive")
    check(cur["name"] == name_before and cur["updated_at"] == updated_before,
          "preview does not mutate current_project")
    check(os.path.getmtime(os.path.join(run_dir, "project.json")) == pj_mtime_before,
          "preview does not save project.json")
    r = client.get("/api/project/name-preview")
    j = r.get_json() or {}
    check(j.get("name") == f"LO_{DATE}", f"no species query -> species + img parts omitted (got {j.get('name')!r})")
    r = client.get("/api/project/name-preview?species=")
    check((r.get_json() or {}).get("name") == f"LO_{DATE}", "empty species query -> species + img omitted")

    # config PUT rebuilds + returns the name
    r = client.put("/api/project/config/3", json={"target_species": "SS"})
    j = r.get_json() or {}
    check(r.status_code == 200 and j.get("success") is True, "config PUT still succeeds")
    check(j.get("name") == f"LO_SS_3img_{DATE}", f"config PUT response carries the rebuilt name (got {j.get('name')!r})")
    check(read_pj(run_dir)["name"] == f"LO_SS_3img_{DATE}", "config PUT persisted the rebuilt name")

    # lock gate: the route sits behind the module-lock before_request
    locked_reg = os.path.join(base, "locked.json")
    with open(locked_reg, "w") as f:
        json.dump({"schema_version": 1,
                   "locks": {"reef_point_seg": {"locked": True, "by": "LO",
                                                "ts": "2026-08-26 12:00 AST"}}}, f)
    os.environ.pop("VICARIUS_LOCK_BYPASS", None)
    os.environ["VICARIUS_LOCKS_PATH"] = locked_reg
    try:
        r = client.get("/api/project/name-preview?species=PA")
        check(r.status_code == 423 and (r.get_json() or {}).get("error") == "locked",
              f"locked module -> name-preview refuses 423 (got {r.status_code})")
    finally:
        os.environ["VICARIUS_LOCK_BYPASS"] = "1"
        os.environ.pop("VICARIUS_LOCKS_PATH", None)
    r = client.get("/api/project/name-preview?species=PA")
    check(r.status_code == 200, "unlocked again -> 200")

    appmod.current_project = None
finally:
    shutil.rmtree(base, ignore_errors=True)

check(not os.path.exists(base), "temp dir removed")
print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)

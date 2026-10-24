#!/usr/bin/env python
"""POST /boxes/api/create-project, the handoff that turns a staged box dataset
into a pipeline project that starts at Train and answers the URL that opens
it in the VEGA main window.

House harness, no pytest. subprocess.run and subprocess.Popen are replaced at
MODULE level before the blueprint is imported, so nothing here can start a
job. Nothing binds a port; the Flask test client drives the app in-process.

Run with:
  env/bin/python scripts/_boxes/tests/test_boxes_create_project.py

Covers:
  MINT       a staged dataset mints a run_* project under the inprocess root
             with project_type "bbox_training", training_source.kind "bbox",
             task_default "detect", steps 1 to 4 skipped and current_step 6,
             and the answer carries reused=false and the main-window URL.
  REUSE      the next press for the same dataset answers that project with
             reused=true and mints nothing; reuse=false mints a second; the
             listing route names both, newest first.
  NO COPY    no imagery and no symlink enters the run dir: the dataset stays
             where it is staged, so non-TCRMP frames never reach a project
             folder, the mask registry, the coverage matrix or STATS.
  VALIDATION unknown dataset, a half-staged folder, a dataset with no labels,
             a path instead of a name, bad initials and a non-boolean reuse
             are all 400 and mint nothing.
  409        a day with every run dir taken answers 409, the same shape the
             VICARIUS launcher answers at its own ceiling.
  PAGE       Train in VEGA is section 2, the conversion is folded into a
             collapsed advanced block, and the page script drives the route.
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
import os
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


class _FakeProc:
    def __init__(self, cmd, cwd=None, env=None, **_kw):
        self.cmd = list(cmd)
        self.pid = 2 ** 22 + 9999
        self.returncode = None
        self.stdout = io.StringIO("")

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -15


def _blocked_run(*_a, **_k):
    raise RuntimeError("subprocess.run blocked in tests")


_REAL_RUN = subprocess.run
_REAL_POPEN = subprocess.Popen
subprocess.run = _blocked_run
subprocess.Popen = _FakeProc

# The retired Train + Infer mount, assembled from parts so that the grep
# sweep proving this package independent of it ("/" + "models" under
# scripts/_boxes) finds nothing, this file included.
_RETIRED_MOUNT = "/" + "models"

from flask import Flask  # noqa: E402

from _boxes import blueprint as bpmod  # noqa: E402

_fail = 0


def check(cond, msg):
    global _fail
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _fail += 1


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def _stage(bbox_root, name="coral_intel", pairs=3, labels=True, manifest=True,
           data_yaml=True):
    d = os.path.join(bbox_root, name)
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    if labels:
        os.makedirs(os.path.join(d, "labels"), exist_ok=True)
    for i in range(pairs):
        _write(os.path.join(d, "images", f"img{i}.jpg"), "x")
        if labels:
            _write(os.path.join(d, "labels", f"img{i}.txt"), "0 0.5 0.5 0.2 0.2\n")
    _write(os.path.join(d, "classes.txt"), "Alpha\n")
    if data_yaml:
        _write(os.path.join(d, "data.yaml"),
               "path: %s\ntrain: images\nval: images\nnc: 1\nnames:\n  0: Alpha\n" % d)
    if manifest:
        with open(os.path.join(d, "ingest_manifest.json"), "w") as fh:
            json.dump({"n_images": pairs, "n_boxes": pairs, "classes": ["Alpha"]}, fh)
    return d


def _run_dirs(inprocess):
    return sorted(d for d in os.listdir(inprocess) if d.startswith("run_"))


def main():
    bbox_root = tempfile.mkdtemp(prefix="boxes_project_bbox_")
    seg_root = tempfile.mkdtemp(prefix="boxes_project_seg_")
    scripts_dir = tempfile.mkdtemp(prefix="boxes_project_scripts_")
    inprocess = tempfile.mkdtemp(prefix="boxes_project_inprocess_")
    src = tempfile.mkdtemp(prefix="boxes_project_src_")
    try:
        dataset = _stage(bbox_root)
        # A half-staged folder (no manifest) and a labelless one, beside it.
        _stage(bbox_root, "halfway", manifest=False)
        _stage(bbox_root, "no_labels", labels=False)
        _write(os.path.join(scripts_dir, "_boxes", "ingest_coral_intel.py"), "")
        py = os.path.join(scripts_dir, "python")
        _write(py, "")

        app = Flask("boxes_project_test")
        app.register_blueprint(
            bpmod.make_blueprint(
                lambda: {"source_path": src, "bbox_root": bbox_root,
                         "seg_root": seg_root, "scripts_dir": scripts_dir,
                         "python": py, "inprocess_root": inprocess},
                log_fn=lambda m: None),
            url_prefix="/boxes")
        c = app.test_client()

        # ---- mint ---------------------------------------------------------
        r = c.post("/boxes/api/create-project", json={"dataset": "coral_intel", "by": "lo"})
        body = r.get_json() or {}
        check(r.status_code == 200, f"a staged dataset mints a project (got {r.status_code} {body})")
        run_dir = body.get("run_dir") or ""
        check(os.path.isdir(run_dir), f"the run dir exists (got {run_dir!r})")
        check(os.path.dirname(run_dir) == os.path.abspath(inprocess),
              f"the run dir is minted directly under the inprocess root (got {run_dir!r})")
        base = os.path.basename(run_dir)
        check(base.startswith("run_") and "_LO_train_coral_intel" in base,
              f"run dir is run_YY-MM-DD_INITIALS_train_<dataset> (got {base})")
        check(body.get("project_id") and body.get("display_name") == "TRAIN_coral_intel",
              f"the response names the project (got {body})")
        check(body.get("reused") is False, f"the first press reports a fresh mint (got {body})")
        check(body.get("open_url") == "/?project_dir=" + quote(run_dir, safe=""),
              f"the response carries the main-window URL (got {body.get('open_url')!r})")

        with open(os.path.join(run_dir, "project.json")) as fh:
            meta = json.load(fh)
        check(meta.get("project_type") == "bbox_training",
              f"project.json marks the project type (got {meta.get('project_type')!r})")
        ts = meta.get("training_source") or {}
        check(ts.get("kind") == "bbox", f"the source kind (got {ts})")
        check(ts.get("name") == "coral_intel", f"the source name (got {ts})")
        check(ts.get("path") == os.path.abspath(dataset),
              f"the source path is where the dataset is staged (got {ts.get('path')!r})")
        check(ts.get("task_default") == "detect",
              f"a box dataset defaults to the detector (got {ts.get('task_default')!r})")
        check(meta.get("current_step") == 6, f"opens on Train (got {meta.get('current_step')!r})")
        check(meta.get("initials") == "LO", f"initials uppercased (got {meta.get('initials')!r})")
        for s in ("1", "2", "3", "4"):
            st = (meta.get("steps") or {}).get(s) or {}
            check(st.get("status") == "skipped", f"step {s} is skipped (got {st.get('status')!r})")
            check("coral_intel" in str(st.get("note") or ""),
                  f"step {s} note names the dataset (got {st.get('note')!r})")
        check(os.path.isfile(os.path.join(run_dir, "README.md")), "the run dir carries a README")
        readme = open(os.path.join(run_dir, "README.md")).read()
        check("coral_intel" in readme, "the README names the dataset")

        # ---- nothing is copied ---------------------------------------------
        entries = sorted(os.listdir(run_dir))
        check("step4test_combinedAnnotate" not in entries,
              f"no export link is minted for a box dataset (got {entries})")
        planted = []
        for dirpath, _dirnames, filenames in os.walk(run_dir):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in (".jpg", ".jpeg", ".png"):
                    planted.append(os.path.join(dirpath, fn))
        check(not planted, f"no imagery enters the run dir (got {planted})")
        links = [os.path.join(dp, n) for dp, dns, fns in os.walk(run_dir)
                 for n in list(dns) + list(fns)
                 if os.path.islink(os.path.join(dp, n))]
        check(not links, f"no symlink enters the run dir either (got {links})")

        # ---- reuse ------------------------------------------------------------
        before = _run_dirs(inprocess)
        r = c.post("/boxes/api/create-project", json={"dataset": "coral_intel", "by": "LO"})
        body2 = r.get_json() or {}
        check(r.status_code == 200 and body2.get("reused") is True,
              f"a second press answers the existing project (got {r.status_code} {body2})")
        check(body2.get("run_dir") == run_dir and body2.get("run_dir_name") == base
              and body2.get("display_name") == "TRAIN_coral_intel",
              f"the reused answer names the first project (got {body2})")
        check(body2.get("open_url") == body.get("open_url"),
              "the reused answer carries the same main-window URL")
        check(_run_dirs(inprocess) == before, "reusing mints nothing")
        r = c.post("/boxes/api/create-project", json={"dataset": "coral_intel", "by": ""})
        check(r.status_code == 200 and (r.get_json() or {}).get("reused") is True,
              f"reusing needs no initials (got {r.status_code} {r.get_json()})")
        check(_run_dirs(inprocess) == before, "reusing without initials mints nothing either")

        listed = c.get("/boxes/api/training-projects?dataset=coral_intel").get_json()
        check(listed.get("ok") is True and [p["run_dir_name"] for p in listed["projects"]] == [base],
              f"the listing names the one project (got {listed})")
        check(listed["projects"][0]["open_url"] == body.get("open_url"),
              "the listing carries the same main-window URL as the handoff")

        # ---- validation -----------------------------------------------------
        # reuse is false throughout, so the initials cases reach the initials
        # check instead of being answered with the project made above.
        for payload, why in (
            ({"dataset": "nope", "by": "LO"}, "an unknown dataset"),
            ({"dataset": "halfway", "by": "LO"}, "a half-staged folder"),
            ({"dataset": "no_labels", "by": "LO"}, "a dataset with no labels"),
            ({"dataset": "", "by": "LO"}, "a blank dataset name"),
            ({"dataset": dataset, "by": "LO"}, "a full path instead of a name"),
            ({"dataset": "../coral_intel", "by": "LO"}, "a traversing name"),
            ({"dataset": "coral_intel", "by": ""}, "blank initials"),
            ({"dataset": "coral_intel", "by": "12345"}, "numeric initials"),
            ({"dataset": "coral_intel", "by": "TOOLONG"}, "over-long initials"),
            ({"dataset": "coral_intel", "by": "LO", "reuse": "yes"}, "a non-boolean reuse"),
            ({"dataset": "coral_intel", "by": "LO", "reuse": None}, "a null reuse"),
        ):
            payload = dict(payload)
            payload.setdefault("reuse", False)
            before = set(os.listdir(inprocess))
            r = c.post("/boxes/api/create-project", json=payload)
            check(r.status_code == 400, f"{why} is refused (got {r.status_code} {r.get_json()})")
            check((r.get_json() or {}).get("error"), f"{why} says why")
            check(set(os.listdir(inprocess)) == before, f"{why} minted nothing")
        r = c.post("/boxes/api/create-project", data="not json",
                   content_type="application/json")
        check(r.status_code == 400, f"a body that is not JSON is refused (got {r.status_code})")

        # ---- 409 when the day is full ---------------------------------------
        real_mint = bpmod.mint_training_project

        def _exhausted(*_a, **_kw):
            raise bpmod.RunDirExhausted("no free run folder today; archive one and retry")

        bpmod.mint_training_project = _exhausted
        try:
            r = c.post("/boxes/api/create-project",
                       json={"dataset": "coral_intel", "by": "LO", "reuse": False})
            check(r.status_code == 409,
                  f"a full day of run dirs answers 409 (got {r.status_code} {r.get_json()})")
            check("archive one" in str((r.get_json() or {}).get("error") or ""),
                  f"the 409 says the way out (got {r.get_json()})")
        finally:
            bpmod.mint_training_project = real_mint

        # ---- a second project for the same dataset, when asked for -----------
        r = c.post("/boxes/api/create-project",
                   json={"dataset": "coral_intel", "by": "LO", "reuse": False})
        body3 = r.get_json() or {}
        check(r.status_code == 200 and body3.get("reused") is False,
              f"reuse false mints a second project (got {r.status_code} {body3})")
        check(os.path.basename(body3.get("run_dir") or "") == base + "_2",
              f"a same-day repeat suffixes _2 (got {body3.get('run_dir')!r})")
        check(body3.get("open_url") == "/?project_dir=" + quote(body3.get("run_dir") or "", safe=""),
              "the second project carries its own main-window URL")
        listed = c.get("/boxes/api/training-projects?dataset=coral_intel").get_json()
        check([p["run_dir_name"] for p in listed["projects"]] == [base + "_2", base],
              f"the listing names both, newest first (got {listed})")
        r = c.post("/boxes/api/create-project", json={"dataset": "coral_intel"})
        check((r.get_json() or {}).get("run_dir_name") == base + "_2",
              "a press with reuse on now answers the newest project")

        # ---- no inprocess root configured -----------------------------------
        app2 = Flask("boxes_project_noroot_test")
        app2.register_blueprint(
            bpmod.make_blueprint(
                lambda: {"source_path": src, "bbox_root": bbox_root,
                         "seg_root": seg_root, "scripts_dir": scripts_dir,
                         "python": py},
                name="boxes_noroot", log_fn=lambda m: None),
            url_prefix="/boxes")
        c2 = app2.test_client()
        r = c2.post("/boxes/api/create-project", json={"dataset": "coral_intel", "by": "LO"})
        check(r.status_code == 500,
              f"a host with no projects folder configured answers 500 (got {r.status_code})")
        r = c2.get("/boxes/api/training-projects?dataset=coral_intel")
        check(r.status_code == 200 and r.get_json()["projects"] == [],
              "a host with no projects folder lists no projects rather than failing")

        # ---- page ------------------------------------------------------------
        html = c.get("/boxes/").get_data(as_text=True)
        check('id="bx-sec-project"' in html, "the page carries the Train in VEGA section")
        check("2. Train in VEGA" in html, "the section is headed Train in VEGA")
        check(html.index("bx-sec-datasets") < html.index("bx-sec-project")
              < html.index("bx-sec-convert"),
              "Train in VEGA is section 2, above the advanced block")
        check('id="bx-adv"' in html, "the conversion sits in an advanced block")
        adv_at = html.index('id="bx-adv"')
        check(html.rfind("<details", 0, adv_at) > html.index("bx-sec-project"),
              "the advanced block is a details element below the Train in VEGA section")
        adv_tag = html[html.rfind("<details", 0, adv_at):html.index(">", adv_at) + 1]
        check("open" not in adv_tag.replace("open in", ""),
              f"the advanced block is collapsed by default (got {adv_tag!r})")
        for marker in ("bx-sec-convert", "bx-sec-progress", "bx-sec-outputs"):
            check(html.index(marker) > adv_at,
                  f"{marker} sits inside the advanced block")
        check("Advanced: convert boxes to masks" in html,
              "the advanced summary says what is inside it")
        for cid in ("bx-project-dataset", "bx-project-by", "bx-project-existing",
                    "bx-project-open", "bx-project-new", "bx-project-link",
                    "bx-project-result"):
            check(f'id="{cid}"' in html, f"the page has the {cid} control")
        # One cache-bust string, carried by both assets. The exact value is
        # pinned by the suite that owns the page (test_boxes_routes.py); here
        # it only has to be there and to be the same on the stylesheet and the
        # script, so a bump does not fail this suite for no reason.
        busts = [part.split('"')[0] for part in html.split("/static/boxes.")[1:]]
        busts = [b.split("?v=")[1] for b in busts if "?v=" in b]
        check(len(busts) == 2 and busts[0] == busts[1] and busts[0],
              f"both assets carry one fresh cache-bust string (got {busts})")
        check(_RETIRED_MOUNT not in html, "the page has no reference to the retired mount")
        check("—" not in html and "–" not in html, "page copy has no em or en dashes")

        js = c.get("/boxes/static/boxes.js").get_data(as_text=True)
        check("'/api/create-project'" in js, "the page script calls the create-project route")
        check("in VEGA; it starts at Train Model" in js,
              "the success message says where the project opens")
        check("window.open(url, MAIN_WINDOW_NAME)" in js,
              "the page script opens the project in the one VEGA window")
        check("Type your initials" in js,
              "the page script asks for initials before a fresh mint")
        check("—" not in js and "–" not in js, "page script copy has no em or en dashes")
    finally:
        subprocess.Popen = _REAL_POPEN
        subprocess.run = _REAL_RUN
        for d in (bbox_root, seg_root, scripts_dir, inprocess, src):
            shutil.rmtree(d, ignore_errors=True)
    check(not any(os.path.exists(d) for d in (bbox_root, seg_root, scripts_dir,
                                              inprocess, src)),
          "temp fixtures removed")

    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()

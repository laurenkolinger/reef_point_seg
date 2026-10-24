"""Model cards from the Train step: model_card_hook.write_train_card and
infer_tags, the step 6 completion paths that call them, and the
GET /api/step/6/tags route the Train form reads.

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_model_card_hook.py

Covers:
  CARD     a finished fake run gets best.pt.card.md with refreshed_by "VEGA Train",
           the weights checksum, and the person's subject and data source with the
           creator's initials in the history; a missing best.pt returns None and
           logs one line; a bad tag is one logged line and does not lose the card.
  STEP     a step that exited nonzero writes no card; a card failure never fails
           the completion and is one line in the step log.
  TAGS     infer_tags on a TCRMP-named coral source and on a fish source; the
           route answers the same two tags, and 400 with a sentence for a source
           outside the allowed roots or no project.
  FORM     the two Train form rows exist with the tooltips given, and the
           config keys they send do not collide with the recipe subject row.

Nothing here reaches a real process: subprocess.run and subprocess.Popen are
stubbed at module level, as every orchestrator suite does.
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

import contextlib
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("VICARIUS_LOCK_BYPASS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
sys.path.insert(0, _ORCH)

_RESULTS = []


class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


# Standing rule: no orchestrator test may reach a real subprocess.
_REAL_RUN = subprocess.run
subprocess.run = lambda *a, **k: _FakeCompleted()


def _blocked_popen(*_a, **_k):
    raise RuntimeError("subprocess.Popen blocked in tests")


_REAL_POPEN = subprocess.Popen
subprocess.Popen = _blocked_popen

import app as appmod  # noqa: E402
import model_card_hook as hook  # noqa: E402
import project_manager as pm  # noqa: E402

WEIGHTS_BYTES = b"fake-weights-for-the-card-test"
RUN_NAME = "run_one"
TCRMP_IMAGES = ["TCRMP20140926_clip_BWR_T106_frame_000001.jpg",
                "TCRMP20150411_clip_BWR_T105_frame_000002.jpg"]
FISH_IMAGES = ["fish_0001.jpg", "fish_0002.jpg"]
SUBJECT_TOOLTIP = "Pick what this model learns to find. VEGA writes it on the model's card so VISION can list the model under it; leave it Not set when neither fits."
SOURCE_TOOLTIP = "Pick TCRMP (Territorial Coral Reef Monitoring Program) when the training images come from its surveys, or Other for imagery from elsewhere. VEGA fills this from the training set's file names when it can."


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(fn):
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _make_source(folder, names_yaml, images):
    """A training source: data.yaml and an all_images folder of empty files."""
    _write(os.path.join(folder, "data.yaml"), "path: .\ntrain: all_images\nval: all_images\n" + names_yaml)
    for name in images:
        _write(os.path.join(folder, "all_images", name), "x")


CORAL_NAMES = "names:\n  0: AA\n  1: AC\n"
FISH_NAMES = ("names:\n  0: EPST_Dark\n  1: HASO_Bar\n"
              "names_full:\n  0: Epinephelus striatus\n  1: Haemulon sciurus\n")


def _make_run(step_dir, source_dir, with_weights=True):
    """A finished run folder under step_dir/runs/RUN_NAME. Returns the run dir."""
    run_dir = os.path.join(step_dir, "runs", RUN_NAME)
    _write(os.path.join(run_dir, "args.yaml"),
           f"task: detect\nmodel: yolo11l.pt\nepochs: 5\nimgsz: 512\ndata: {source_dir}/data.yaml\n")
    if with_weights:
        _write(os.path.join(run_dir, "weights", "best.pt"), WEIGHTS_BYTES.decode())
    return run_dir


@contextlib.contextmanager
def _scratch():
    tmp = tempfile.mkdtemp(prefix="t8_model_card_hook_")
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _card_of(run_dir):
    _, build = hook.load_card_modules()
    card = sys.modules["card"]
    weights = Path(run_dir) / "weights" / "best.pt"
    return card.read_card(card.card_path_for(weights))["vicarius"]


def test_write_train_card_writes_the_card_and_person_tags():
    with _scratch() as tmp:
        source = os.path.join(tmp, "source")
        _make_source(source, FISH_NAMES, FISH_IMAGES)
        run_dir = _make_run(os.path.join(tmp, "step6_trainModel"), source)
        lines = []
        path = hook.write_train_card(run_dir, {"card_subject": "fish", "card_data_source": "other"},
                                     "lo", log=lines.append)
        check(path is not None and path.name == "best.pt.card.md", f"card path: {path}")
        check(path.parent == Path(run_dir) / "weights", "the card sits beside best.pt")
        check(not lines, f"no failure line expected: {lines}")
        block = _card_of(run_dir)
        check(block["refreshed_by"] == "VEGA Train", f"refreshed_by: {block['refreshed_by']}")
        check(block["person"].get("subject") == "fish" and block["person"].get("data_source") == "other",
              f"person: {block['person']}")
        check([row["by"] for row in block["history"]] == ["LO", "LO"], f"history: {block['history']}")
        want = hashlib.sha256(WEIGHTS_BYTES).hexdigest()
        check(block["record"]["weights"]["sha256"] == want, f"sha256: {block['record']['weights']}")
        check(block["record"]["weights"]["bytes"] == len(WEIGHTS_BYTES), "bytes recorded")


def test_each_subject_choice_reaches_the_card_correctly():
    for choice, want in (("fish", "fish"), ("benthic", "benthic"), ("", None)):
        with _scratch() as tmp:
            source = os.path.join(tmp, "source")
            _make_source(source, FISH_NAMES, FISH_IMAGES)
            run_dir = _make_run(os.path.join(tmp, "step6_trainModel"), source)
            lines = []
            path = hook.write_train_card(run_dir, {"card_subject": choice, "card_data_source": "other"},
                                         "LO", log=lines.append)
            check(path is not None and not lines, f"{choice!r}: card written without a failure line: {lines}")
            person = _card_of(run_dir)["person"]
            if want is None:
                check("subject" not in person, f"Not set writes no subject: {person}")
                state = sys.modules["card"].card_state(sys.modules["card"].read_card(path))
                check(state == sys.modules["card"].STATE_NEEDS_PERSON, f"the card needs a person: {state}")
            else:
                check(person.get("subject") == want, f"{choice!r} stored as {want!r}: {person}")


def test_recipe_subjects_prefill_the_model_subject():
    import json
    js = open(os.path.join(_ORCH, "static", "orchestrator.js"), encoding="utf-8").read()
    start = js.index("const STEP6_CARD_SUBJECT_FROM_RECIPE")
    end = js.index("/**\n * Fill the two model card rows")
    script = js[start:end] + """
const cases = [["coral","benthic"],["algae","benthic"],["fish","fish"],["other",""],["",""],["nonsense",""]];
const out = cases.map(([r]) => step6CardSubjectFor(null, r));
out.push(step6CardSubjectFor("fish", "coral"), step6CardSubjectFor("benthic", "fish"), step6CardSubjectFor("", "algae"));
console.log(JSON.stringify(out));
"""
    # node only evaluates a pure function from the page script; no app process is spawned.
    subprocess.Popen = _REAL_POPEN
    try:
        proc = _REAL_RUN(["node", "-e", script], capture_output=True, text=True, timeout=30)
    finally:
        subprocess.Popen = _blocked_popen
    check(proc.returncode == 0, proc.stderr)
    got = json.loads(proc.stdout)
    check(got == ["benthic", "benthic", "fish", "", "", "", "fish", "benthic", "benthic"], f"prefill: {got}")


def _run_card_tag_script(body):
    """Run the page's restore and prefill functions in node against fake rows.

    body is node code that runs after the functions and stubs are defined and
    prints one JSON value. Returns that value.
    """
    import json
    js = open(os.path.join(_ORCH, "static", "orchestrator.js"), encoding="utf-8").read()
    start = js.index("function restoreStep6CardTag")
    end = js.index("let _s6CardTagsReq = 0;") + len("let _s6CardTagsReq = 0;")
    script = """
const rows = {};
function makeRow(values) {
  return {value: '', dataset: {}, options: values.map(v => ({value: v}))};
}
rows['s6-card-subject'] = makeRow(['', 'fish', 'benthic']);
rows['s6-card-data-source'] = makeRow(['', 'tcrmp', 'other']);
rows['s6-source'] = {value: ''};
global.document = {getElementById: id => rows[id] || null};
let recipe = 'coral';
function step6SubjectValue() { return recipe; }
let answer = null;
async function api() { if (answer instanceof Error) throw answer; return answer; }
""" + js[start:end] + "\n" + body
    subprocess.Popen = _REAL_POPEN
    try:
        proc = _REAL_RUN(["node", "-e", script], capture_output=True, text=True, timeout=30)
    finally:
        subprocess.Popen = _blocked_popen
    check(proc.returncode == 0, proc.stderr)
    return json.loads(proc.stdout)


def test_a_refused_tags_route_still_prefills_the_subject_from_the_recipe():
    got = _run_card_tag_script("""
(async () => {
  const out = [];
  for (const [r, ans] of [['coral', {error: 'Place + Segment has no output yet'}], ['fish', {error: 'x'}],
                          ['other', {error: 'x'}], ['coral', new Error('network')]]) {
    recipe = r; answer = ans; rows['s6-card-subject'].value = ''; delete rows['s6-card-subject'].dataset.picked;
    await refreshStep6CardTags();
    out.push(rows['s6-card-subject'].value);
  }
  // The tags route wins when it answers; a person's pick is never overwritten.
  recipe = 'coral'; answer = {subject: 'fish', data_source: 'tcrmp'};
  rows['s6-card-subject'].value = ''; delete rows['s6-card-subject'].dataset.picked;
  await refreshStep6CardTags();
  out.push(rows['s6-card-subject'].value, rows['s6-card-data-source'].value);
  rows['s6-card-subject'].value = 'fish'; rows['s6-card-subject'].dataset.picked = '1';
  answer = {error: 'x'}; recipe = 'coral';
  await refreshStep6CardTags();
  out.push(rows['s6-card-subject'].value);
  console.log(JSON.stringify(out));
})();
""")
    check(got == ["benthic", "fish", "", "benthic", "fish", "tcrmp", "fish"], f"refused-route prefill: {got}")


def test_a_saved_not_set_is_restored_as_not_set_and_not_refilled():
    got = _run_card_tag_script("""
(async () => {
  const out = [];
  const subj = rows['s6-card-subject'];
  recipe = 'coral'; answer = {error: 'x'};
  // Saved Not set that a person picked: stays Not set after the prefill runs.
  restoreStep6CardTag('s6-card-subject', '', true);
  await refreshStep6CardTags();
  out.push([subj.value, !!subj.dataset.picked]);
  // Saved value that was only prefilled: restored, still follows the recipe.
  restoreStep6CardTag('s6-card-subject', 'benthic', false);
  recipe = 'fish';
  await refreshStep6CardTags();
  out.push([subj.value, !!subj.dataset.picked]);
  // A project with no saved value at all prefills.
  restoreStep6CardTag('s6-card-subject', undefined, undefined);
  await refreshStep6CardTags();
  out.push([subj.value, !!subj.dataset.picked]);
  // A project saved before the flag existed: a value counts as a pick.
  restoreStep6CardTag('s6-card-subject', 'fish', undefined);
  out.push([subj.value, !!subj.dataset.picked]);
  // A value the row does not offer is dropped, not kept.
  restoreStep6CardTag('s6-card-subject', 'banana', true);
  out.push([subj.value, !!subj.dataset.picked]);
  console.log(JSON.stringify(out));
})();
""")
    check(got == [["", True], ["fish", False], ["fish", False], ["fish", True], ["", False]],
          f"restore: {got}")


def test_missing_weights_returns_none_and_logs_one_line():
    with _scratch() as tmp:
        source = os.path.join(tmp, "source")
        _make_source(source, FISH_NAMES, FISH_IMAGES)
        run_dir = _make_run(os.path.join(tmp, "step6_trainModel"), source, with_weights=False)
        lines = []
        check(hook.write_train_card(run_dir, {}, "LO", log=lines.append) is None, "no card without best.pt")
        check(len(lines) == 1 and "best.pt" in lines[0], f"one line naming best.pt: {lines}")
        check(not list(Path(run_dir).rglob("*.card.md")), "no card file written")


def test_a_bad_tag_is_one_line_and_keeps_the_card():
    with _scratch() as tmp:
        source = os.path.join(tmp, "source")
        _make_source(source, FISH_NAMES, FISH_IMAGES)
        run_dir = _make_run(os.path.join(tmp, "step6_trainModel"), source)
        lines = []
        path = hook.write_train_card(run_dir, {"card_subject": "banana", "card_data_source": "tcrmp"},
                                     "LO", log=lines.append)
        check(path is not None, "the card is still written")
        check(len(lines) == 1 and "subject" in lines[0], f"one line naming the tag: {lines}")
        block = _card_of(run_dir)
        check("subject" not in block["person"] and block["person"].get("data_source") == "tcrmp",
              f"person: {block['person']}")


def test_missing_initials_do_not_write_tags_but_keep_the_card():
    with _scratch() as tmp:
        source = os.path.join(tmp, "source")
        _make_source(source, FISH_NAMES, FISH_IMAGES)
        run_dir = _make_run(os.path.join(tmp, "step6_trainModel"), source)
        lines = []
        path = hook.write_train_card(run_dir, {"card_subject": "fish"}, "", log=lines.append)
        check(path is not None and len(lines) == 1, f"card kept, one line: {path} {lines}")
        check(_card_of(run_dir)["person"] == {}, "no person tag without initials")


def test_a_missing_card_code_folder_is_one_line():
    saved = os.environ.get("VICARIUS_ROOT")
    os.environ["VICARIUS_ROOT"] = "/nonexistent/vicarius_root"
    try:
        with _scratch() as tmp:
            run_dir = _make_run(os.path.join(tmp, "step6_trainModel"), tmp)
            lines = []
            check(hook.write_train_card(run_dir, {}, "LO", log=lines.append) is None, "no card")
            check(len(lines) == 1 and "/nonexistent/vicarius_root" in lines[0], f"line: {lines}")
            check(hook.infer_tags({}, tmp) == {"subject": None, "data_source": None}, "tags stay empty")
    finally:
        if saved is None:
            os.environ.pop("VICARIUS_ROOT", None)
        else:
            os.environ["VICARIUS_ROOT"] = saved


def test_infer_tags_tcrmp_coral_and_fish_sources():
    with _scratch() as tmp:
        coral = os.path.join(tmp, "coral")
        _make_source(coral, CORAL_NAMES, TCRMP_IMAGES)
        fish = os.path.join(tmp, "fish")
        _make_source(fish, FISH_NAMES, FISH_IMAGES)
        check(hook.infer_tags({}, coral) == {"subject": "benthic", "data_source": "tcrmp"},
              f"coral: {hook.infer_tags({}, coral)}")
        check(hook.infer_tags({}, fish) == {"subject": "fish", "data_source": "other"},
              f"fish: {hook.infer_tags({}, fish)}")


def test_infer_tags_survives_hostile_and_missing_sources():
    with _scratch() as tmp:
        empty = {"subject": None, "data_source": None}
        check(hook.infer_tags({}, os.path.join(tmp, "nope")) == empty, "a missing folder gives empty tags")
        broken = os.path.join(tmp, "broken")
        _write(os.path.join(broken, "data.yaml"), "names: [unclosed\n  - :::\n")
        check(hook.infer_tags({}, broken) == empty, "a malformed data.yaml gives empty tags")
        odd = os.path.join(tmp, "odd")
        _make_source(odd, CORAL_NAMES, ["=cmd|' /C calc'!A0.jpg", "name with spaces.jpg"])
        check(hook.infer_tags({}, odd)["data_source"] == "other", "odd file names are not TCRMP")


def _make_owner(tmp, source):
    """An owner project record on disk whose step 6 has a finished fake run."""
    step_dir = os.path.join(tmp, "step6_trainModel")
    _make_run(step_dir, source)
    steps = {}
    for key in pm.STEP_KEYS:
        d = os.path.join(tmp, pm.STEP_DIRS[key])
        os.makedirs(d, exist_ok=True)
        steps[key] = {"status": "pending", "name": pm.STEP_NAMES[key], "dir": d, "config": {}, "outputs": {}}
    steps["6"]["dir"] = step_dir
    steps["6"]["status"] = "running"
    steps["6"]["config"] = {"run_name": RUN_NAME, "card_subject": "fish", "card_data_source": "other"}
    state = {"id": os.path.basename(tmp), "name": "t", "project_dir": tmp, "created_at": "x",
             "updated_at": "x", "current_step": 6, "initials": "LO", "steps": steps, "access_log": []}
    pm.save_project(state)
    return state


def test_nonzero_exit_writes_no_card_and_zero_exit_does():
    with _scratch() as tmp:
        source = os.path.join(tmp, "source")
        _make_source(source, FISH_NAMES, FISH_IMAGES)
        state = _make_owner(tmp, source)
        weights_card = Path(tmp) / "step6_trainModel" / "runs" / RUN_NAME / "weights" / "best.pt.card.md"
        launch = {"project_dir": tmp}
        check(appmod._record_step_for_owner(6, launch, False, 1) == "", "the failure is recorded")
        check(not weights_card.exists(), "a run that exited nonzero writes no card")
        state = pm.load_project(tmp)
        state["steps"]["6"]["status"] = "running"
        pm.save_project(state)
        check(appmod._record_step_for_owner(6, launch, True, 0) == "", "the completion is recorded")
        check(weights_card.is_file(), "a run that exited zero gets its card")
        block = _card_of(weights_card.parent.parent)
        check(block["person"].get("subject") == "fish", f"person: {block['person']}")
        check(all(row["by"] == "LO" for row in block["history"]), "the creator's initials sign the tags")


def test_a_card_failure_never_fails_the_step():
    with _scratch() as tmp:
        source = os.path.join(tmp, "source")
        _make_source(source, FISH_NAMES, FISH_IMAGES)
        _make_owner(tmp, source)
        saved = hook.write_train_card
        seen = []
        saved_log = appmod._log_train_card_line

        def boom(*_a, **_k):
            raise RuntimeError("card exploded")

        hook.write_train_card = boom
        appmod._log_train_card_line = lambda step, line: seen.append((step, line))
        try:
            check(appmod._record_step_for_owner(6, {"project_dir": tmp}, True, 0) == "",
                  "the completion is still recorded")
        finally:
            hook.write_train_card = saved
            appmod._log_train_card_line = saved_log
        check(len(seen) == 1 and "card exploded" in seen[0][1], f"one line: {seen}")
        check(pm.load_project(tmp)["steps"]["6"]["status"] != "error", "the step is not marked as an error")


def test_a_log_line_that_raises_cannot_escape_the_step():
    with _scratch() as tmp:
        source = os.path.join(tmp, "source")
        _make_source(source, FISH_NAMES, FISH_IMAGES)
        _make_owner(tmp, source)
        saved, saved_log = hook.write_train_card, appmod._log_train_card_line

        def boom(*_a, **_k):
            raise RuntimeError("card exploded")

        def log_boom(*_a, **_k):
            raise OSError("log exploded")

        hook.write_train_card = boom
        appmod._log_train_card_line = log_boom
        try:
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                check(appmod._record_step_for_owner(6, {"project_dir": tmp}, True, 0) == "",
                      "the completion is still recorded")
            check("log exploded" in buf.getvalue(), "the double failure is printed once")
        finally:
            hook.write_train_card, appmod._log_train_card_line = saved, saved_log


def test_the_card_line_reaches_the_runner_log():
    appmod.runner.logs[6] = __import__("collections").deque(maxlen=10)
    before = appmod.runner.log_totals.get(6, 0)
    appmod._log_train_card_line(6, "[model_card] test line")
    check(list(appmod.runner.logs[6]) == ["[model_card] test line"], "line in the step log")
    check(appmod.runner.log_totals[6] == before + 1, "the running total follows the line")
    appmod.runner.logs.pop(6, None)


@contextlib.contextmanager
def _route_client(tmp, with_project=True):
    """A client with the standalone root pointed at tmp/_standalone."""
    standalone = os.path.join(tmp, "_standalone")
    os.makedirs(standalone, exist_ok=True)
    saved = appmod._standalone_root
    appmod._standalone_root = lambda: standalone
    appmod.current_project = None
    if with_project:
        steps = {}
        proj = os.path.join(tmp, "proj")
        for key in pm.STEP_KEYS:
            d = os.path.join(proj, pm.STEP_DIRS[key])
            os.makedirs(d, exist_ok=True)
            steps[key] = {"status": "pending", "name": pm.STEP_NAMES[key], "dir": d, "config": {}, "outputs": {}}
        appmod.current_project = {"id": "p", "name": "t", "project_dir": proj, "created_at": "x",
                                  "updated_at": "x", "current_step": 6, "steps": steps}
    try:
        yield appmod.create_app().test_client(), standalone
    finally:
        appmod._standalone_root = saved
        appmod.current_project = None


def test_step6_tags_route_gives_the_same_tags():
    with _scratch() as tmp, _route_client(tmp) as (client, standalone):
        fish = os.path.join(standalone, "fish_session")
        _make_source(fish, FISH_NAMES, FISH_IMAGES)
        coral = os.path.join(standalone, "coral_session")
        _make_source(coral, CORAL_NAMES, TCRMP_IMAGES)
        got = client.get("/api/step/6/tags", query_string={"source": fish}).get_json()
        check(got["ok"] and got["subject"] == "fish" and got["data_source"] == "other", f"fish: {got}")
        got = client.get("/api/step/6/tags", query_string={"source": coral}).get_json()
        check(got["subject"] == "benthic" and got["data_source"] == "tcrmp", f"coral: {got}")


def test_step6_tags_route_refuses_a_source_outside_the_roots():
    with _scratch() as tmp, _route_client(tmp) as (client, _standalone):
        outside = os.path.join(tmp, "outside_source")
        _make_source(outside, FISH_NAMES, FISH_IMAGES)
        resp = client.get("/api/step/6/tags", query_string={"source": outside})
        check(resp.status_code == 400, f"status {resp.status_code}")
        error = resp.get_json()["error"]
        check(len(error.split()) > 3 and error.strip(), f"a sentence: {error!r}")
        resp = client.get("/api/step/6/tags", query_string={"source": "/etc"})
        check(resp.status_code == 400, "a system folder is refused")


def test_step6_tags_route_without_a_project_is_400():
    with _scratch() as tmp, _route_client(tmp, with_project=False) as (client, _standalone):
        resp = client.get("/api/step/6/tags")
        check(resp.status_code == 400 and "No project" in resp.get_json()["error"], "no project is a 400")


def test_form_rows_carry_the_tooltips_and_do_not_reuse_the_recipe_ids():
    html = open(os.path.join(_ORCH, "templates", "index.html"), encoding="utf-8").read()
    js = open(os.path.join(_ORCH, "static", "orchestrator.js"), encoding="utf-8").read()
    # Each tooltip is one template variable that feeds both the glyph and the select.
    check(html.count(SUBJECT_TOOLTIP) == 1, "the subject tooltip text is in the template once")
    check(html.count(SOURCE_TOOLTIP) == 1, "the training data tooltip text is in the template once")
    row = html[html.index('id="s6-card-subject"') - 600: html.index('id="s6-classes-card"')]
    check(row.count('title="{{ t }}"') == 2 and row.count("{{ q(t,") == 2, "glyph and select both carry the tooltip")
    check(html.count('id="s6-card-subject"') == 1 and html.count('id="s6-card-data-source"') == 1, "one of each id")
    check(html.count('id="s6-subject"') == 1, "the recipe subject row is untouched")
    data_at = html.index('id="s6-section-data"')
    classes_at = html.index('id="s6-classes-card"')
    check(data_at < html.index('id="s6-card-subject"') < classes_at, "the rows sit in the Training data section")
    sel = html[html.index('id="s6-card-subject"'): html.index('</select>', html.index('id="s6-card-subject"'))]
    check([o.split('"')[1] for o in sel.split("<option ")[1:]] == ["", "fish", "benthic"], "three choices: Not set, Fish, Benthic")
    check("Other" not in sel and "general" not in sel, "no Other or general choice")
    check("cfg.card_subject" in js and "cfg.card_data_source" in js, "the form sends both tags")
    check("s6.card_subject" in js and "s6.card_data_source" in js, "the form restores both tags")
    check("Imagery source</label>" in html and "Training data</label>" not in row, "the source row is labelled Imagery source")
    check(row.count("cost-free") == 2, "both rows carry the free chip, so the selects align with the rows above")
    check("/api/step/6/tags" in js, "the form pre-fills from the tags route")
    for text in (html[html.index('id="s6-card-subject"') - 800: html.index('id="s6-classes-card"')],):
        check("—" not in text and "–" not in text, "no em or en dash in the new markup")


if __name__ == "__main__":
    for case in (
        test_write_train_card_writes_the_card_and_person_tags,
        test_each_subject_choice_reaches_the_card_correctly,
        test_recipe_subjects_prefill_the_model_subject,
        test_a_refused_tags_route_still_prefills_the_subject_from_the_recipe,
        test_a_saved_not_set_is_restored_as_not_set_and_not_refilled,
        test_missing_weights_returns_none_and_logs_one_line,
        test_a_bad_tag_is_one_line_and_keeps_the_card,
        test_missing_initials_do_not_write_tags_but_keep_the_card,
        test_a_missing_card_code_folder_is_one_line,
        test_infer_tags_tcrmp_coral_and_fish_sources,
        test_infer_tags_survives_hostile_and_missing_sources,
        test_nonzero_exit_writes_no_card_and_zero_exit_does,
        test_a_card_failure_never_fails_the_step,
        test_a_log_line_that_raises_cannot_escape_the_step,
        test_the_card_line_reaches_the_runner_log,
        test_step6_tags_route_gives_the_same_tags,
        test_step6_tags_route_refuses_a_source_outside_the_roots,
        test_step6_tags_route_without_a_project_is_400,
        test_form_rows_carry_the_tooltips_and_do_not_reuse_the_recipe_ids,
    ):
        run(case)

    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)

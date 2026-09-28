"""
A step's state line answers the reader, not the poll: a Stop the server
refused stays on screen until the reader acts again or the run's state
changes, and a step that finished for another project says what is true of
the open project and keeps the one way of clearing its mark reachable.

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_status_note_hold.py

Every test drives the SHIPPED scripts/pipeline_orchestrator/static/orchestrator.js
inside the Node stand-in that test_rail_family.py builds (imported from there,
never copied), against fetch stubs answering exactly as app.py's step_stop,
_survive_restart and _finish_for_owner do. Nothing here reaches the network
or a port.

Covers:
  HOLD      the red refusal from a Stop the server could not make survives
            the next poll beats, which carry the amber sentence about the
            same process; a second refused Stop keeps it; the poll's own
            sentence is back once the reader presses Run again.
  RELEASE   a Stop the server made, a run that ends for real, a run that
            fails, and a reset each take the held line off the screen.
  FOREIGN   a step finished for another project names that project, says
            nothing is running here, keeps Stop pressable with a title that
            says it only clears the mark, and gives Stop its live title back
            with the next run; the restart case shares the title.
  HOSTILE   a sentence the server sends is shown as text, never as markup.
"""

import json
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_rail_family as rail  # noqa: E402  (the Node stand-in and its stubs)

_RESULTS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(fn):
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:  # noqa: BLE001 - a suite reports, never crashes
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


# app.py's sentences, filled in as the live routes fill them.
_REFUSAL = ("Train Model (step 5) is still running as process 4242, which outlived a VEGA "
            "restart, so the orchestrator cannot stop it. Stop process 4242 yourself, then "
            "press Stop again.")
_ORPHAN = ("Train Model (step 5) is still running as process 4242, which outlived a VEGA "
           "restart, so the orchestrator cannot stop it. Its output goes on being written "
           "to /x/step6_trainModel.")
_FOREIGN = ("Train Model (step 5) was started by project 'coral_intel', not the one open "
            "here, so VEGA recorded the result there and left this project exactly as it was.")
_RESTART = ("VEGA was restarted while Train Model (step 5) was running, so it no longer owns "
            "that process and cannot say how it ended. Nothing was recorded. Check "
            "/x/step6_trainModel, then press Stop to clear the mark or run the step again.")
_HOSTILE = "<img src=x onerror=alert(1)> was started by project '<b>x</b>'."

# The Stop button as templates/index.html ships it: hidden, with its live title.
_LIVE_STOP_TITLE = ("Stop the training run that is live now. VICARIUS ends the trainer "
                    "process and keeps every checkpoint already written.")
_STOP_BUTTON = """
const stopBtn = document.getElementById('s6-stop-btn');
stopBtn.style.display = 'none';
stopBtn.title = %s;
""" % json.dumps(_LIVE_STOP_TITLE)


def _server(stop_status, stop_body, status_body):
    """A fetch stub answering Stop, the status poll, its log read, and every
    other route (the config PUT and run POST that Run makes) with success.

    Parameters:
        stop_status: HTTP status for POST /api/step/6/stop.
        stop_body: its JSON body, as JavaScript source.
        status_body: the poll payload, as JavaScript source; may be reassigned
            from a scenario through `statusBody`.
    Returns:
        the scenario JavaScript that installs the stub, a tick() that fires
        one beat of the shipped poller, and panelRead().
    """
    return rail._SETTLE + rail._PANEL_READER + _STOP_BUTTON + """
let statusBody = %s;
const calls = [];
fetch = (url, opts) => {
  calls.push(((opts && opts.method) || 'GET') + ' ' + url);
  const answer = (status, body) => Promise.resolve({ status: status, json: () => Promise.resolve(body) });
  if (url.indexOf('/stop') >= 0) return answer(%d, %s);
  if (url.indexOf('/status') >= 0) return answer(200, statusBody);
  if (url.indexOf('/log') >= 0) return answer(200, { lines: [], offset: 0, dropped: 0 });
  if (url.indexOf('/reset') >= 0) return answer(200, { success: true, state: state });
  return answer(200, { success: true, warnings: [] });
};
let tickFn = null;
const tick = async () => {
  if (!tickFn) { startPolling(6); tickFn = timers[timers.length - 1][0]; }
  await tickFn();
  await settle();
};
const stopRead = () => ({ display: stopBtn.style.display, title: stopBtn.title });
""" % (status_body, stop_status, stop_body)


_ORPHAN_POLL = "{ running: true, exit_code: null, orphan_pid: 4242, warnings: [%s] }" % json.dumps(_ORPHAN)
_REFUSING_STOP = ("{ success: false, stopped: false, orphan_pid: 4242, error: %s }"
                  % json.dumps(_REFUSAL))


def _drive(scenario):
    return rail._drive(rail._HONEST_STATE + scenario)


# ── HOLD ────────────────────────────────────────────────────────────
def test_the_refusal_outlives_the_next_poll_beats():
    """The poll repeats the amber orphan sentence every two seconds. It used
    to write it over the red refusal on the very next beat, so the
    instruction 'Stop process 4242 yourself, then press Stop again' was gone
    before a student could read it."""
    out = _drive(_server(409, _REFUSING_STOP, _ORPHAN_POLL) + """
(async () => {
  // populateConfigs shows Stop for a step the project records as running.
  setStepRunControls(6, true);
  await tick();
  const before = panelRead(6).note;
  await stopStep(6); await settle();
  const refused = panelRead(6).note;
  await tick(); await tick(); await tick();
  return { before: before, refused: refused, after: panelRead(6), stop: stopRead() };
})();""")
    check(_ORPHAN in out["before"], f"the poll's own sentence shows first: {out['before']}")
    check(_REFUSAL in out["refused"], f"the refusal replaces it on the press: {out['refused']}")
    check(_REFUSAL in out["after"]["note"],
          f"three poll beats later the refusal is still on screen: {out['after']['note']}")
    check(_ORPHAN not in out["after"]["note"],
          f"the poll's sentence has not written over it: {out['after']['note']}")
    check("var(--error)" in out["after"]["note"], "it still reads as a refusal")
    check(out["after"]["mark"] == "running", "the tile stays running, because nothing stopped")
    check(out["after"]["polling"], "the poller keeps going")
    check(out["stop"]["display"] == "", f"Stop stays pressable: {out['stop']}")


def test_a_second_refused_stop_keeps_the_refusal():
    out = _drive(_server(409, _REFUSING_STOP, _ORPHAN_POLL) + """
(async () => {
  await tick(); await stopStep(6); await settle(); await tick();
  await stopStep(6); await settle(); await tick(); await tick();
  return panelRead(6).note;
})();""")
    check(_REFUSAL in out and _ORPHAN not in out, f"the refusal holds through a second press: {out}")


def test_pressing_run_again_gives_the_poll_its_voice_back():
    """Run is the reader's next action: it clears the line and the hold, so
    the poll's sentence about the process it starts shows again."""
    out = _drive(_server(409, _REFUSING_STOP, _ORPHAN_POLL) + """
(async () => {
  await tick(); await stopStep(6); await settle();
  await runStep(6); await settle();
  const cleared = panelRead(6).note;
  await tick();
  return { cleared: cleared, after: panelRead(6).note };
})();""")
    check(out["cleared"] == "", f"Run takes the refusal off the screen: {out['cleared']!r}")
    check(_ORPHAN in out["after"], f"the next poll beat speaks again: {out['after']}")


# ── RELEASE ─────────────────────────────────────────────────────────
def test_a_stop_the_server_then_makes_clears_the_held_line():
    out = _drive(_server(200, "{ success: true, stopped: true, message: 'Stopped.' }",
                         _ORPHAN_POLL) + """
(async () => {
  showStepStatusNote(6, 'error', 'a refusal from before', true);
  await stopStep(6); await settle();
  return panelRead(6);
})();""")
    check(out["note"] == "" and out["noteShown"] == "none", f"the line is cleared: {out}")
    check(out["mark"] == "pending", f"the tile is not done: {out['mark']!r}")


def test_a_run_that_ends_for_real_takes_the_held_line_with_it():
    """A refusal is about a process; once the runner reports that process
    finished, the refusal must not sit beside the success badge."""
    out = _drive(_server(409, _REFUSING_STOP, _ORPHAN_POLL) + """
(async () => {
  await tick(); await stopStep(6); await settle();
  statusBody = { running: false, exit_code: 0 };
  await tick();
  return panelRead(6);
})();""")
    check(out["note"] == "", f"the held line is gone: {out['note']!r}")
    check("Step completed successfully" in out["result"], f"the success shows alone: {out['result']}")
    check(not out["polling"], "the poller stops with the run")


def test_a_run_that_fails_takes_the_held_line_with_it():
    out = _drive(_server(409, _REFUSING_STOP, _ORPHAN_POLL) + """
(async () => {
  await tick(); await stopStep(6); await settle();
  statusBody = { running: false, exit_code: 1 };
  await tick();
  return panelRead(6);
})();""")
    check(out["note"] == "", f"the held line is gone: {out['note']!r}")
    check("exit code 1" in out["result"], f"the failure shows alone: {out['result']}")
    check(out["mark"] == "error", f"the tile reads as failed: {out['mark']!r}")


def test_a_restart_the_server_reports_replaces_the_held_line():
    """The orphan dies: the poll answers untracked with the restart sentence,
    which is newer truth than the refusal and replaces it."""
    out = _drive(_server(409, _REFUSING_STOP, _ORPHAN_POLL) + """
(async () => {
  await tick(); await stopStep(6); await settle();
  statusBody = { running: false, exit_code: null, untracked: true, warnings: [%s] };
  await tick();
  return panelRead(6);
})();""" % json.dumps(_RESTART))
    check(_RESTART in out["note"] and _REFUSAL not in out["note"],
          f"the restart sentence stands alone: {out['note']}")


def test_a_reset_takes_the_state_line_off_the_screen():
    out = _drive(_server(409, _REFUSING_STOP, _ORPHAN_POLL) + """
(async () => {
  await tick(); await stopStep(6); await settle();
  await doResetStep(6); await settle();
  return panelRead(6);
})();""")
    check(out["note"] == "" and out["noteShown"] == "none", f"a reset clears the line: {out}")


# ── FOREIGN ─────────────────────────────────────────────────────────
_FOREIGN_POLL = ("{ running: false, exit_code: 0, completed: false, warnings: [%s] }"
                 % json.dumps(_FOREIGN))


def test_another_projects_finish_says_what_is_true_here_and_keeps_stop_reachable():
    """The rail read 'Train Model running' beside a panel with nothing to
    stop. The panel now says nothing runs here, names the project the result
    went to, and keeps Stop pressable, because the Stop route is the one
    thing that clears a mark no process backs; its title says exactly that."""
    out = _drive(_server(200, "{ success: true, stopped: false }", _FOREIGN_POLL) + """
(async () => { await tick(); return { panel: panelRead(6), stop: stopRead(), calls: calls }; })();""")
    panel, stop = out["panel"], out["stop"]
    check(_FOREIGN in panel["note"], f"the project the result went to is named: {panel['note']}")
    check("Nothing is running for this project." in panel["note"],
          f"what is true here is said: {panel['note']}")
    check("Press Stop to clear the running mark, or run the step again." in panel["note"],
          f"both ways out are offered: {panel['note']}")
    check(panel["result"] == "", "no success is claimed")
    check(panel["mark"] == "running", "the project's own record is left as it is")
    check(stop["display"] == "", f"Stop is pressable: {stop}")
    check(stop["title"].startswith("Clear the running mark"),
          f"Stop says it only clears the mark: {stop['title']}")
    check("VEGA only sets the step back to pending" in stop["title"], stop["title"])
    check(panel["runDisabled"] is False, "Run stays open as the other way out")
    check(not any("/api/project/state" in c for c in out["calls"]), "nothing is reloaded")


def test_stop_gets_its_live_title_back_with_the_next_run():
    out = _drive(_server(200, "{ success: true, stopped: false }", _FOREIGN_POLL) + """
(async () => {
  await tick();
  const stale = stopRead().title;
  await runStep(6); await settle();
  return { stale: stale, live: stopRead() };
})();""")
    check(out["stale"].startswith("Clear the running mark"), out["stale"])
    check(out["live"]["title"] == _LIVE_STOP_TITLE, f"the live title is restored: {out['live']}")
    check(out["live"]["display"] == "", "Stop shows for the live run")


def test_a_stop_pressed_over_a_stale_mark_clears_it():
    """The server answers a Stop over a stale mark with success and 'Nothing
    was running'; the tile goes to pending and the line clears."""
    out = _drive(_server(200, "{ success: true, stopped: false, message: 'Nothing was running for Train Model (step 5).' }",
                         _FOREIGN_POLL) + """
(async () => { await tick(); await stopStep(6); await settle(); return { panel: panelRead(6), stop: stopRead() }; })();""")
    check(out["panel"]["mark"] == "pending", f"the mark is cleared: {out['panel']['mark']!r}")
    check(out["panel"]["note"] == "", f"the line is cleared: {out['panel']['note']!r}")
    check(out["stop"]["display"] == "none", "Stop is hidden again")


def test_the_restart_case_shares_the_stale_mark_title():
    out = _drive(_server(200, "{ success: true }",
                         "{ running: false, exit_code: null, untracked: true, warnings: [%s] }"
                         % json.dumps(_RESTART)) + """
(async () => { await tick(); return stopRead(); })();""")
    check(out["display"] == "" and out["title"].startswith("Clear the running mark"), str(out))


def test_a_server_that_sends_no_sentence_still_names_both_facts():
    out = _drive(_server(200, "{ success: true }",
                         "{ running: false, exit_code: 0, completed: false }") + """
(async () => { await tick(); return panelRead(6).note; })();""")
    check("not recorded against the open project" in out, f"the fallback sentence shows: {out}")
    check("Nothing is running for this project." in out, f"and what is true here: {out}")


# ── HOSTILE ─────────────────────────────────────────────────────────
def test_a_hostile_sentence_is_shown_as_text():
    out = _drive(_server(409, "{ success: false, error: %s }" % json.dumps(_HOSTILE),
                         "{ running: false, exit_code: 0, completed: false, warnings: [%s] }"
                         % json.dumps(_HOSTILE)) + """
(async () => { await tick(); const poll = panelRead(6).note; await stopStep(6); await settle();
  return { poll: poll, stop: panelRead(6).note }; })();""")
    for key in ("poll", "stop"):
        check("<img" not in out[key] and "&lt;img" in out[key],
              f"the {key} sentence is escaped: {out[key]}")
        check("<b>" not in out[key], f"no markup survives in the {key} line: {out[key]}")


def test_the_new_copy_obeys_the_house_rules():
    js = open(rail._JS, encoding="utf-8").read()
    for name in ("STEP_NOTHING_RUNNING_HERE_TEXT", "STALE_MARK_STOP_TITLE"):
        start = js.index(f"const {name} = ")
        text = js[start:js.index(";", start)]
        for banned in ("—", "–", " land", "UVI", "You "):
            check(banned not in text, f"{name} carries {banned!r}")
    check("const heldStepNotes = {}" in js, "the hold is a named, documented map")


if __name__ == "__main__":
    print("HOLD:")
    run(test_the_refusal_outlives_the_next_poll_beats)
    run(test_a_second_refused_stop_keeps_the_refusal)
    run(test_pressing_run_again_gives_the_poll_its_voice_back)
    print("RELEASE:")
    run(test_a_stop_the_server_then_makes_clears_the_held_line)
    run(test_a_run_that_ends_for_real_takes_the_held_line_with_it)
    run(test_a_run_that_fails_takes_the_held_line_with_it)
    run(test_a_restart_the_server_reports_replaces_the_held_line)
    run(test_a_reset_takes_the_state_line_off_the_screen)
    print("FOREIGN:")
    run(test_another_projects_finish_says_what_is_true_here_and_keeps_stop_reachable)
    run(test_stop_gets_its_live_title_back_with_the_next_run)
    run(test_a_stop_pressed_over_a_stale_mark_clears_it)
    run(test_the_restart_case_shares_the_stale_mark_title)
    run(test_a_server_that_sends_no_sentence_still_names_both_facts)
    print("HOSTILE:")
    run(test_a_hostile_sentence_is_shown_as_text)
    run(test_the_new_copy_obeys_the_house_rules)

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print(f"\n{passed}/{total} passed")
    for name, ok, err in _RESULTS:
        if not ok:
            print(f"\n--- {name} ---\n{err}")
    sys.exit(0 if passed == total else 1)

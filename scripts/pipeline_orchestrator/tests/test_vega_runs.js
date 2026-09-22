#!/usr/bin/env node
/* Pure-logic tests for the All trained models registry
   (static/orchestrator_runs.js, window.VegaRuns). No browser, no DOM
   library: the render functions take data and return HTML strings, and the
   DOM glue is driven through a minimal fake element. Node 18.

   Run:  node scripts/pipeline_orchestrator/tests/test_vega_runs.js

   Rewritten 2026-09-16 for the registry rebuild, after Lauren read the card
   and wrote "wtf is this" of its broken run names and then ruled the whole
   platform hard to read. Simplified the same day, after she ruled that
   comparing model runs is not what this card is for and asked instead for the
   training data each run used, as something she can filter by. The suite pins
   what the rebuild and that simplification owe:

     smoke       - module loads, every documented export is present
     names       - a name is never broken mid-token: a stamped name splits
                   only at its stamp, an unstamped one is one span, and both
                   halves keep one ink on a line that cannot wrap
     states      - the process-state family: champion, did not finish, not
                   scored, scored, and the plain sentence each one carries
     trained     - the data subset each run trained on, in its own column
                   and its own filter, named by the route's own field or by
                   the open project's own record, never by the folder that
                   holds the run, and read as "not recorded" where neither
                   names one
     comparison  - nothing about comparability reaches the page: no chip, no
                   count, no dimmed score, while the payload keeps its fields
     groups      - detectors and segmenters in their own groups, the panel's
                   own family first, the metric named once per group
     project     - the open project's runs marked on the row and in the meta
                   line, and offered as a source filter
     summary     - the runs waiting on a person first, and the champion's
                   mAP50-95 labelled as mAP50-95 and never as the mAP50 the
                   rows show
     filters     - search, state, source and training data compose, the sort
                   heads reorder, Clear filters appears only while a filter is
                   narrowing
     actions     - an action that cannot work is disabled with its reason on
                   the focusable wrapper, never on the button
     stripes     - every other record carries the striping class
     scope       - the champion record's scope is rendered honestly
     adversarial - null data, runs not an array, string scores, a rejecting
                   fetch, a body carrying an error, unknown task, hostile
                   names, a click on a disabled button, a click outside any
                   action, storage that throws
     css         - the stylesheet's legibility floor and its narrow layout

   Tests run one after another on a promise chain, so an async test (load,
   init) finishes before the next one starts and the PASS/FAIL line for
   each name is printed in order. */
"use strict";
const fs = require("fs");
const path = require("path");
const V = require(path.join(__dirname, "..", "static", "orchestrator_runs.js"));

let pass = 0, fail = 0;
function ok(cond, msg) { if (cond) { pass++; } else { fail++; console.error("  FAIL:", msg); } }
function eq(a, b, msg) {
  ok(JSON.stringify(a) === JSON.stringify(b), `${msg} (got ${JSON.stringify(a)}, want ${JSON.stringify(b)})`);
}
let chain = Promise.resolve();
function section(title) { chain = chain.then(() => console.log(title)); }
function test(name, fn) {
  chain = chain.then(() => {
    const before = fail;
    const done = () => console.log(`  ${fail === before ? "PASS" : "FAIL"} ${name}`);
    return Promise.resolve().then(fn).then(done, e => {
      fail++;
      console.error("  FAIL:", name, "threw", e && e.stack ? e.stack : e);
      done();
    });
  });
}
function count(haystack, needle) { return haystack.split(needle).length - 1; }

// ── fixtures ─────────────────────────────────────────────────────────
const OPEN_PROJECT = "/inprocess/run_26-09-02_LO_train_coral_intel";
const HOSTILE_NAME = "<script>alert(\"x\")</script>'&";
const HOSTILE_PROJECT = "<b>evil</b>";
const CHAMPION = {
  run_dir: "/inprocess/_models/fish_photo_28feb25_paper/runs/run3_heavy_augment",
  run_name: "run3_heavy_augment",
  promoted_at: "2026-09-15T05:04:02-04:00",
  // The champion record's score is a mAP50-95, which is the whole point of
  // the headline test: the run's own row shows its mAP50, 0.901.
  score: 0.826,
  family: "box",
  scope: "module",
};
function run(overrides) {
  return Object.assign({
    run_dir: "/inprocess/run_x/step6_trainModel/runs/RUN",
    run_name: "RUN",
    origin: "project",
    project_name: "run_x",
    task: "detect",
    family: "box",
    map50: 0.4,
    map50_95: 0.2,
    evaluated_at: "2026-09-03T15:40:00-04:00",
    has_weights: true,
    champion: false,
    imgsz: 640,
    split_fingerprint: "abc123",
    score_source: "ledger",
    comparable_to_champion: true,
    comparability_reason: "",
  }, overrides);
}
const LONG_STAMPED = "TRAIN_fish_A3_finish_lowhue_20260916_091917";
const LONG_PLAIN = "run4_weighted_augmented";
const RUNS = [
  // the champion, a lab detector
  run({ run_dir: CHAMPION.run_dir, run_name: CHAMPION.run_name, origin: "lab",
        project_name: "fish_photo_28feb25_paper", map50: 0.901, map50_95: 0.826, champion: true,
        evaluated_at: "2026-09-15T05:02:21-04:00", trained_on: "fish_photo_28feb25", trained_on_kind: "bbox" }),
  // the open project's own detector, scored but on a split nobody recorded;
  // its record names no data set, so the open project's own record names it
  run({ run_dir: OPEN_PROJECT + "/step6_trainModel/runs/TRAIN_coral_intel_20260902_224223",
        run_name: "TRAIN_coral_intel_20260902_224223", project_name: "run_26-09-02_LO_train_coral_intel",
        map50: 0.623, map50_95: 0.535, imgsz: 512, split_fingerprint: "",
        comparable_to_champion: false, comparability_reason: "split not recorded" }),
  // another project's detector, scored from an evaluation folder
  run({ run_dir: "/inprocess/run_26-09-16_LO_fish/step6_trainModel/runs/" + LONG_STAMPED,
        run_name: LONG_STAMPED, project_name: "run_26-09-16_LO_fish_ablation_backend",
        map50: 0.9, map50_95: 0.827, score_source: "evaluation",
        evaluated_at: "2026-09-16T13:35:25-04:00", trained_on: "fish_photo_28feb25", trained_on_kind: "bbox" }),
  // a lab detector with a plain name and no stamp
  run({ run_dir: "/inprocess/_models/fish_photo_28feb25_paper/runs/" + LONG_PLAIN,
        run_name: LONG_PLAIN, origin: "lab", project_name: "fish_photo_28feb25_paper",
        map50: 0.89, map50_95: 0.814, evaluated_at: "2026-09-15T05:02:50-04:00",
        trained_on: "fish_photo_28feb25", trained_on_kind: "bbox" }),
  // a detector whose training never finished, in a project whose record
  // names no data set: the one row that reads "not recorded"
  run({ run_dir: "/inprocess/run_26-09-01_LO_aborted/step6_trainModel/runs/TRAIN_aborted_20260901_080000",
        run_name: "TRAIN_aborted_20260901_080000", project_name: "run_26-09-01_LO_aborted",
        map50: null, map50_95: null, evaluated_at: null, has_weights: false, imgsz: null,
        split_fingerprint: "", score_source: "", comparable_to_champion: false }),
  // a segmenter: another metric family entirely
  run({ run_dir: "/inprocess/run_18aprilpm_PA/step6_trainModel/runs/18aprilpm_PA_20260422_135837",
        run_name: "18aprilpm_PA_20260422_135837", project_name: "run_18aprilpm_PA_20260418_431d1b",
        task: "seg", family: "mask", map50: 0.601, map50_95: 0.503, imgsz: 1024,
        evaluated_at: "2026-07-14T13:14:12-04:00", split_fingerprint: "def456",
        trained_on: "18aprilpm_PA", trained_on_kind: "export",
        comparable_to_champion: false, comparability_reason: "different metric family" }),
  // a segmenter nobody has scored
  run({ run_dir: "/inprocess/run_18aprilpm_PA/step6_trainModel/runs/18aprilpm_PA_20260422_143501",
        run_name: "18aprilpm_PA_20260422_143501", project_name: "run_18aprilpm_PA_20260418_431d1b",
        task: "seg", family: "mask", map50: null, map50_95: null, imgsz: 1024, evaluated_at: null,
        trained_on: "18aprilpm_PA", trained_on_kind: "export",
        score_source: "", comparable_to_champion: false, comparability_reason: "different metric family" }),
  // a hostile lab name and a hostile data set name, an imported run
  run({ run_dir: "/inprocess/_models/" + HOSTILE_NAME + "/runs/" + HOSTILE_NAME, run_name: HOSTILE_NAME,
        origin: "lab", project_name: HOSTILE_PROJECT, origin_kind: "imported",
        trained_on: HOSTILE_PROJECT, trained_on_kind: "composite",
        imported_from: "/home/bizon/Desktop/ai/oceankind_CV", imported_at: "2026-09-15T04:00:00-04:00",
        map50: 0.5, map50_95: 0.3, evaluated_at: "2026-09-01T10:00:00-04:00", split_fingerprint: "",
        comparable_to_champion: false, comparability_reason: "split not recorded" }),
];
const DATA = { ok: true, runs: RUNS, champion: CHAMPION, champion_scope: "module" };
const VIEW_ALL = { search: "", state: "all", source: "all", trained: "all", sort: { key: "state", dir: "asc" } };
/** The filter value for one training data set, as the card writes it. */
function trainedSet(name) { return "set:" + name; }
/** The open project's own training source, as the state route names it. */
const OPEN_SOURCE = { name: "coral_intel", kind: "bbox" };
const DETECT_PANEL = { projectDir: OPEN_PROJECT, projectSource: OPEN_SOURCE, panelTask: "detect", taskPinned: true, view: VIEW_ALL };

/** A minimal stand-in for the card's root element: records dispatched
 *  events and listeners, and hands back one fake element per data-role. */
function fakeRoot() {
  const body = { innerHTML: "", querySelector() { return null; } };
  return {
    dispatched: [], listeners: {}, body,
    dispatchEvent(ev) { this.dispatched.push(ev); return true; },
    addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); },
    querySelector(sel) { return sel.indexOf("body") >= 0 ? body : null; },
  };
}
/** A fake action button as handleClick sees it through event.target.closest(). */
function fakeButton(attrs, disabled) {
  const el = { disabled: !!disabled, getAttribute(k) { return Object.prototype.hasOwnProperty.call(attrs, k) ? attrs[k] : null; } };
  el.closest = sel => {
    if (sel === "[data-sort]") return el.getAttribute("data-sort") === null ? null : el;
    if (sel === "[data-action]") return el.getAttribute("data-action") === null ? null : el;
    return el;
  };
  return el;
}
/** Records of a rendered table, one chunk per run (each holds its main row and its meta row). */
function dataRows(html) { return html.split('<tbody class="vega-runs-record').slice(1); }
/** The chunk of a rendered table belonging to one run name. */
function rowOf(html, name) { return dataRows(html).find(r => r.indexOf(name) >= 0) || ""; }
/** Render the whole board with the default detector-panel options. */
function board(overrides) { return V.render(DATA, Object.assign({}, DETECT_PANEL, overrides || {})); }

// ── smoke ────────────────────────────────────────────────────────────
section("SMOKE:");
test("test_exports_present", () => {
   ["render", "renderRows", "renderRow", "renderGroupHead", "renderFilters", "renderSummaryLine",
   "renderEmpty", "renderError", "renderLoading", "renderNoMatch", "renderNameParts",
   "trainedOn", "trainedTip", "trainedValue", "projectSourceOf",
   "formatAst", "taskLabel", "familyLabel", "strictFamilyLabel",
   "normaliseRun", "normaliseChampion", "normaliseView", "isOpenProjectRun",
   "runState", "sourceKind", "matchesFilters", "sortRows", "describeRuns", "groupByFamily",
   "countRuns", "finetuneRefusal", "finetuneOffer", "crossFamily", "esc", "load", "init",
   "initStep6Runs", "handleClick",
   "handleFilter", "dispatchAction"].forEach(name =>
    ok(typeof V[name] === "function", `VegaRuns.${name} is a function`));
  eq(V.RUNS_URL, "/api/step/6/all_runs", "RUNS_URL matches the wave-2 contract");
  eq(V.PROJECT_STATE_URL, "/api/project/state", "the card reads the open project from the state route");
  eq(V.EVENTS, { evaluate: "vega:evaluate-run", promote: "vega:promote-run", finetune: "vega:finetune-from" },
     "EVENTS carry the three host-wired names");
  eq(V.STORE_PREFIX, "vega.runs.", "every remembered choice sits under the one prefix the registry spec asks for");
});

// ── names ────────────────────────────────────────────────────────────
section("NAMES:");
test("test_a_name_without_a_stamp_is_never_split", () => {
  // The fault Lauren pasted: run4_weighted_augmented rendered as "run4_wei"
  // in one ink and "ghted_augmented" in another, wrapped onto two lines.
  const html = V.renderNameParts(LONG_PLAIN);
  eq(count(html, "<span"), 1, "an unstamped name is one span, so no token can be cut in half");
  ok(html.indexOf(LONG_PLAIN) >= 0, "the whole name is in that span");
  ["run3_heavy_augment", "run6_1024px_augment", "preset_smoke_1ep", "dual_gpu_640_batch32"].forEach(name => {
    eq(count(V.renderNameParts(name), "<span"), 1, `${name} renders as one span`);
  });
});

test("test_a_stamped_name_splits_only_at_its_stamp", () => {
  const html = V.renderNameParts(LONG_STAMPED);
  ok(html.indexOf('<span class="vega-runs-name-head">TRAIN_fish_A3_finish_lowhue</span>') >= 0,
     `the head stops at the stamp boundary: ${html}`);
  ok(html.indexOf('<span class="vega-runs-name-tail">_20260916_091917</span>') >= 0,
     "the stamp is its own span and never shortens");
  ok(html.indexOf("TRAIN_fish_A3_finish_lowhue_20260916_091917".slice(0, 30)) >= 0
     || html.replace(/<[^>]+>/g, "") === LONG_STAMPED, "the two spans together are the whole name");
  eq(V.renderNameParts("TRAIN_x-20260916-091917").indexOf('<span class="vega-runs-name-tail">-20260916-091917</span>') >= 0,
     true, "the dashed spelling of a stamp splits the same way");
});

test("test_a_name_that_is_only_a_stamp_is_not_split", () => {
  eq(count(V.renderNameParts("20260916_091917"), "<span"), 1,
     "a name with nothing before its stamp keeps one span rather than an empty head");
});

test("test_the_name_carries_the_full_name_and_its_folder_in_the_tooltip", () => {
  const row = rowOf(board(), LONG_STAMPED);
  ok(row.indexOf('title="' + LONG_STAMPED + " in /inprocess/run_26-09-16_LO_fish") >= 0,
     `a shortened name is readable in full from its tooltip: ${row.slice(0, 400)}`);
});

// ── states ───────────────────────────────────────────────────────────
section("STATES:");
test("test_every_state_comes_from_the_process_state_family", () => {
  eq(Object.keys(V.STATES).sort(), ["champion", "done", "empty", "failed"],
     "four states, no ninth role invented and no comparability state left");
  eq(V.STATES.champion.word, "champion", "the promoted run is the verified role");
  eq(V.STATES.failed.word, "did not finish", "failed");
  eq(V.STATES.empty.word, "not scored", "empty");
  eq(V.STATES.done.word, "scored", "done by machine");
  ok(Object.keys(V.STATES).every(k => V.STATES[k].word.indexOf("comparable") < 0),
     "no state is named after a comparison");
});

test("test_a_run_is_put_in_the_right_state", () => {
  const champ = V.normaliseChampion(CHAMPION, "module");
  const state = raw => V.runState(V.normaliseRun(raw), champ);
  eq(state(RUNS[0]), "champion", "the promoted run");
  eq(state(RUNS[1]), "done", "a scored run whose split nobody recorded is still a scored run");
  eq(state(RUNS[3]), "done", "and so is any other scored run");
  eq(state(RUNS[4]), "failed", "no weights file");
  eq(state(RUNS[6]), "empty", "weights but no score");
  eq(state(RUNS[5]), "done", "a segmenter under a detector champion is scored, not flagged");
});

// Lauren, 2026-09-16: "i dont really care about comparing model runs ... in
// some applications simple is better." The comparability apparatus comes off
// the screen; the payload keeps every field it ever sent.
test("test_nothing_about_comparability_reaches_the_page", () => {
  const html = board();
  ["not comparable", "comparable", "split fingerprint", "held-out images from the champion",
   "ranks", "ranked"].forEach(phrase => {
    ok(html.indexOf(phrase) < 0, `the page says nothing about ${phrase}: ${html.indexOf(phrase)}`);
  });
  ok(html.indexOf("is-attention") < 0, "and no row or count carries the not-comparable ink");
  ok(html.indexOf("is-dim") < 0, "a score is never dimmed for a comparison the card no longer makes");
  eq(typeof V.comparabilityReason, "undefined", "the comparability sentences are gone from the card");
  eq(typeof V.comparabilityShort, "undefined", "and so is the chip suffix that named them");
});

test("test_the_payload_keeps_its_comparability_fields_untouched", () => {
  const run1 = V.normaliseRun(RUNS[1]);
  eq(run1.comparable_to_champion, false, "the route's verdict is still read from the payload");
  eq(run1.comparability_reason, "split not recorded", "and so is its reason, word for word");
  eq(V.normaliseRun(RUNS[3]).comparable_to_champion, true, "a comparable run reads true, as it always did");
  eq(V.normaliseRun({}).comparable_to_champion, null, "and a payload that says nothing reads null");
});

test("test_the_state_chip_is_the_first_data_column", () => {
  const row = rowOf(board(), "TRAIN_aborted_20260901_080000");
  const firstCell = row.slice(row.indexOf('<td role="cell">'), row.indexOf("</td>"));
  ok(firstCell.indexOf("vega-runs-chip is-failed") >= 0,
     `the registry spec puts the state chip in the first data column: ${firstCell}`);
  ok(firstCell.indexOf("did not finish") >= 0, "and it carries the state word");
});

// ── the data each run trained on ─────────────────────────────────────
// Lauren asked for one thing in place of the comparison apparatus: "maybe
// just make a table that includes the training data subset (so we can filter
// by those)". The subset is the staged box dataset, the training set or the
// project export a run learned from. Its name comes from the route's own
// field, or from the open project's own record for the runs that project
// trained, and never from the folder that holds the run: a folder name is
// not a data set, and the reviewer of 2026-09-16 found the old fallback
// asserting one for folders that hold none. A record naming none reads "not
// recorded".
section("TRAINED:");
test("test_every_row_names_the_subset_it_trained_on", () => {
  const html = board();
  ok(html.indexOf(">Trained on<") >= 0, "the column has its own heading in sentence case");
  eq(count(html, 'class="vega-runs-trained'), RUNS.length, "and every record fills the cell");
  const lab = rowOf(html, LONG_PLAIN);
  ok(lab.indexOf(">fish_photo_28feb25<") >= 0, `a lab run names the staged box dataset it trained on: ${lab}`);
  ok(lab.indexOf("VEGA trained this run on the staged box dataset fish_photo_28feb25.") >= 0,
     `with the whole sentence in the cell's tooltip: ${lab}`);
  const cell = lab.slice(lab.indexOf('class="vega-runs-trained'), lab.indexOf("</span>", lab.indexOf('class="vega-runs-trained')));
  ok(cell.indexOf("fish_photo_28feb25_paper") < 0, `and never the lab folder that holds the run: ${cell}`);
  const other = rowOf(html, LONG_STAMPED);
  ok(other.indexOf(">fish_photo_28feb25<") >= 0, "another project's run trained on the same dataset says the same name");
  ok(other.indexOf("fish_ablation_backend") < 0 || other.indexOf(">fish_photo_28feb25<") >= 0,
     "so the lab and the project read as one data set, which is what the filter groups by");
  const seg = rowOf(html, "18aprilpm_PA_20260422_135837");
  ok(seg.indexOf(">18aprilpm_PA<") >= 0, "an ordinary project's run names the project whose export it trained on");
  ok(seg.indexOf("VEGA trained this run on the outlines the project 18aprilpm_PA exported from its own images.") >= 0,
     `and says so: ${seg}`);
});

test("test_the_name_comes_from_the_route_never_from_the_folder", () => {
  eq(V.trainedOn(V.normaliseRun(run({ trained_on: "coral_intel", trained_on_kind: "bbox", project_name: "run_x" })), null, false),
     { name: "coral_intel", kind: "bbox" }, "a trained_on field from the route is the name shown, with its kind");
  eq(V.trainedOn(V.normaliseRun(run({ project_name: "fish_lab", origin: "lab" })), null, false),
     { name: "", kind: "" }, "a lab folder name is never offered as the data set");
  eq(V.trainedOn(V.normaliseRun(run({ project_name: "run_26-09-16_LO_fish_ablation_backend" })), null, false),
     { name: "", kind: "" }, "nor is a project folder name, whatever its stamp");
  eq(V.trainedOn(V.normaliseRun(run({ trained_on: "x", trained_on_kind: "nonsense" })), null, false),
     { name: "x", kind: "" }, "a kind the card does not know reads as no kind, and the name still shows");
  eq(V.normaliseRun(run({ trained_on: 42, trained_on_kind: 7 })).trained_on, "42",
     "a value that is not a string is coerced, never thrown on");
  eq(typeof V.trainedDisplayName, "undefined", "the stamp-stripping display name is gone with the folder fallback");
});

test("test_the_open_projects_own_record_names_its_runs", () => {
  // The state route carries training_source_name for the open project; a run
  // that project trained learned from that source, and the route's own row
  // may not say so yet.
  const html = board();
  const mine = rowOf(html, "TRAIN_coral_intel_20260902_224223");
  ok(mine.indexOf(">coral_intel<") >= 0, `the open project's run names the project's own source: ${mine}`);
  ok(mine.indexOf("VEGA trained this run on the staged box dataset coral_intel.") >= 0, "with the box dataset sentence");
  const theirs = rowOf(html, "TRAIN_aborted_20260901_080000");
  ok(theirs.indexOf(">not recorded<") >= 0, "another project's run gets nothing from the open project's record");
  const noSource = board({ projectSource: null });
  ok(rowOf(noSource, "TRAIN_coral_intel_20260902_224223").indexOf(">not recorded<") >= 0,
     "and with no source on the state route the open project's run reads not recorded rather than guessing");
  eq(V.trainedOn(V.normaliseRun(run({ trained_on: "tutorial_boxes", trained_on_kind: "bbox" })), OPEN_SOURCE, true),
     { name: "tutorial_boxes", kind: "bbox" }, "the route's own field still wins over the project record");
  eq(V.projectSourceOf({ loaded: true, state: { project_dir: OPEN_PROJECT, training_source_name: "coral_intel", training_source_label: "box dataset" } }),
     { name: "coral_intel", kind: "bbox" }, "a box dataset label reads as the bbox kind");
  eq(V.projectSourceOf({ loaded: true, state: { training_source_name: "qs20260902_pa_ss", training_source_label: "training set" } }),
     { name: "qs20260902_pa_ss", kind: "composite" }, "a training set label reads as the composite kind");
  eq(V.projectSourceOf({ loaded: true, state: { training_source_name: "", training_source_label: "" } }),
     { name: "", kind: "" }, "an ordinary project names no source");
  eq(V.projectSourceOf({ loaded: false }), { name: "", kind: "" }, "and no open project names none");
  eq(V.projectSourceOf(null), { name: "", kind: "" }, "nor does a state route that answered nothing");
});

test("test_a_run_whose_record_names_no_data_reads_not_recorded", () => {
  const html = V.render({ ok: true, runs: [run({ run_name: "orphan", project_name: "run_orphan" })], champion: null },
                        DETECT_PANEL);
  ok(html.indexOf(">not recorded<") >= 0, "the cell says so rather than showing a blank or the folder");
  ok(html.indexOf("run_orphan") < 0 || html.indexOf(">run_orphan<") < 0, "the folder name is not drawn as a data set");
  // The sentence reaches the page escaped, apostrophe and all, which is what
  // the cell's tooltip shows.
  ok(html.indexOf("VEGA found no training data named in this run&#39;s own record.") >= 0,
     `and the tooltip says where VEGA looked: ${html}`);
  ok(html.indexOf('value="none"') >= 0, "the filter offers those runs as their own choice");
  const table = board().split('data-role="board"')[1];
  eq(count(table, ">not recorded<"), 1, "on the fixture board only the run whose record names none reads so");
});

test("test_each_kind_of_training_data_has_its_own_sentence", () => {
  eq(V.trainedTip({ name: "coral_intel_subset", kind: "bbox" }),
     "VEGA trained this run on the staged box dataset coral_intel_subset.", "a staged box dataset");
  eq(V.trainedTip({ name: "qs20260902_pa_ss", kind: "composite" }),
     "VEGA trained this run on the training set qs20260902_pa_ss.", "a training set");
  eq(V.trainedTip({ name: "18aprilpm_PA", kind: "export" }),
     "VEGA trained this run on the outlines the project 18aprilpm_PA exported from its own images.", "a project export");
  eq(V.trainedTip({ name: "something", kind: "" }), "VEGA trained this run on something.", "a name of no stated kind");
  eq(V.trainedTip({ name: "", kind: "" }), "VEGA found no training data named in this run's own record.", "no name");
});

test("test_the_trained_on_filter_names_the_subset_and_composes", () => {
  const champ = V.normaliseChampion(CHAMPION, "module");
  const rows = V.describeRuns(RUNS, champ, OPEN_PROJECT, OPEN_SOURCE);
  const kept = v => rows.filter(r => V.matchesFilters(r, v)).map(r => r.run.run_name);
  const fish = trainedSet("fish_photo_28feb25");
  eq(kept({ search: "", state: "all", source: "all", trained: fish }),
     [CHAMPION.run_name, LONG_STAMPED, LONG_PLAIN], "one data set keeps the lab run and the project run trained on it");
  eq(kept({ search: "run4", state: "all", source: "all", trained: fish }), [LONG_PLAIN],
     "and it narrows further with the other filters rather than replacing them");
  eq(kept({ search: "", state: "all", source: "mine", trained: fish }), [],
     "filters that cannot both be true keep nothing, rather than ignoring one");
  eq(kept({ search: "", state: "all", source: "all", trained: trainedSet("coral_intel") }),
     ["TRAIN_coral_intel_20260902_224223"], "the open project's own source is a choice too");
  const html = board();
  const select = html.split('id="s6-runs-trained"')[1].split("</select>")[0];
  ok(select.indexOf('value="' + fish + '"') >= 0, "with an option per data set the payload holds");
  ok(select.indexOf(">fish_photo_28feb25<") >= 0, "whose text is the subset name itself, so the control says what narrows the table");
  ok(select.indexOf("fish_photo_28feb25_paper") < 0 && select.indexOf("fish_ablation_backend") < 0
     && select.indexOf("run_26-09") < 0, `and never a folder name: ${select}`);
  ok(select.indexOf(">Every data set<") >= 0, "and its widest setting first");
  ok(select.indexOf('value="' + trainedSet("run_x") + '"') < 0,
     "a data set no run came from is never offered, so no choice can empty the table");
  const options = select.split("<option").slice(1);
  ok(/value="none"/.test(options[options.length - 1]), `the runs whose record names none come last: ${select}`);
});

test("test_not_recorded_stays_last_even_beside_a_remembered_choice", () => {
  const gone = trainedSet("zzz_a_set_that_was_deleted");
  const html = board({ view: Object.assign({}, VIEW_ALL, { trained: gone }) });
  const select = html.split('id="s6-runs-trained"')[1].split("</select>")[0];
  const options = select.split("<option").slice(1);
  ok(/value="none"/.test(options[options.length - 1]),
     `a remembered name that sorts after every other still sits before not recorded: ${select}`);
  ok(select.indexOf('value="' + gone + '" selected') >= 0, "and the remembered choice is the one selected");
});

test("test_a_remembered_data_set_the_payload_no_longer_holds_is_still_shown", () => {
  const gone = trainedSet("a_set_that_was_deleted");
  const view = Object.assign({}, VIEW_ALL, { trained: gone });
  eq(V.normaliseView(view).trained, gone, "a name is data, so the remembered value is kept by its shape");
  const html = board({ view: view });
  ok(html.indexOf('value="' + gone + '" selected') >= 0,
     "the control shows what is really filtering the table");
  ok(html.indexOf("No run matches these filters.") >= 0, "the board says why it is empty");
  ok(html.indexOf("Clear filters") >= 0, "with the way out on screen");
  eq(V.normaliseView({ trained: "nonsense" }).trained, "all",
     "a remembered value in no shape the card writes widens rather than hiding every row");
});

test("test_a_hostile_data_set_name_is_escaped_in_the_cell_and_the_filter", () => {
  const html = board();
  ok(html.indexOf('value="set:<b>evil</b>"') < 0, "the option value is escaped");
  ok(html.indexOf("set:&lt;b&gt;evil&lt;/b&gt;") >= 0, "into its escaped form");
  ok(html.indexOf("<b>evil</b>") < 0, "and no data set name reaches the page as markup");
  ok(html.indexOf("VEGA trained this run on the training set &lt;b&gt;evil&lt;/b&gt;.") >= 0,
     "the tooltip sentence carries it escaped too");
});

// ── groups ───────────────────────────────────────────────────────────
section("GROUPS:");
test("test_detectors_and_segmenters_are_grouped_apart", () => {
  const html = board();
  const det = html.indexOf("Models that draw boxes, scored on how well they find things: box mAP50");
  const seg = html.indexOf("Models that draw outlines, scored on how well they find things: outline mAP50");
  ok(det > 0 && seg > 0, "both families have their own heading");
  ok(det < seg, "the panel's own family opens the table");
  eq(count(html, 'class="vega-runs-group"'), 2, "one heading per family present, and no empty group");
  ok(html.indexOf("6 runs") > 0 && html.indexOf("2 runs") > 0,
     "each heading counts the runs the filters kept for it");
});

test("test_the_group_heading_carries_the_metric_so_the_rows_do_not", () => {
  const row = rowOf(board(), LONG_PLAIN);
  ok(row.indexOf("vega-runs-score") >= 0, "the row shows its number");
  ok(row.indexOf(">box mAP50<") < 0, "and never repeats the metric name the group already states");
});

test("test_the_segmenter_family_opens_the_table_on_a_segmentation_panel", () => {
  const html = board({ panelTask: "segment", taskPinned: false });
  ok(html.indexOf("Models that draw outlines, scored on how well they find things: outline mAP50") < html.indexOf("Models that draw boxes, scored on how well they find things: box mAP50"),
     "the family the Train panel is set to comes first");
});

test("test_a_run_whose_family_cannot_be_read_still_gets_a_group", () => {
  const odd = { ok: true, runs: [run({ run_name: "mystery", task: "", family: "" })], champion: null };
  const html = V.render(odd, DETECT_PANEL);
  ok(html.indexOf("Runs whose family VEGA cannot read") >= 0, "an unreadable family is stated, never guessed");
});

// ── the open project ─────────────────────────────────────────────────
section("PROJECT:");
test("test_the_open_projects_own_runs_are_marked", () => {
  const html = board();
  const mine = rowOf(html, "TRAIN_coral_intel_20260902_224223");
  ok(mine.indexOf("is-mine-record") >= 0, "the record says the open project trained it");
  ok(mine.indexOf(">this project<") >= 0, "and the meta line says so in words");
  const theirs = rowOf(html, LONG_PLAIN);
  ok(theirs.indexOf("is-mine-record") < 0, "another lab's run is not marked");
  ok(theirs.indexOf(">fish_photo_28feb25<") >= 0,
     "and the Trained on column names the data set it came from");
  const meta = theirs.slice(theirs.indexOf("vega-runs-meta-line"));
  ok(meta.indexOf("Model lab") < 0 && meta.indexOf("fish_photo_28feb25_paper") < 0,
     `so the meta line no longer repeats it: ${meta}`);
});

test("test_a_run_is_matched_to_the_project_by_containment_not_by_name", () => {
  const inside = V.normaliseRun(RUNS[1]);
  ok(V.isOpenProjectRun(inside, OPEN_PROJECT), "a run folder under the project folder belongs to it");
  ok(V.isOpenProjectRun(inside, OPEN_PROJECT + "/"), "a trailing slash on the project folder changes nothing");
  ok(!V.isOpenProjectRun(inside, "/inprocess/run_26-09-02_LO_train_coral"),
     "a sibling folder whose name merely prefixes this one is not the same project");
  ok(!V.isOpenProjectRun(inside, ""), "with no project open nothing is marked");
});

test("test_the_source_filter_offers_only_the_sources_the_payload_holds", () => {
  const html = board();
  ok(html.indexOf('value="mine"') >= 0 && html.indexOf('value="other"') >= 0 && html.indexOf('value="lab"') >= 0,
     "all three sources are on offer here");
  const labsOnly = { ok: true, runs: [RUNS[3]], champion: CHAMPION };
  const only = V.render(labsOnly, DETECT_PANEL);
  ok(only.indexOf('value="lab"') >= 0 && only.indexOf('value="mine"') < 0,
     "a source with no run is never offered, so no choice can empty the table");
});

// ── summary ──────────────────────────────────────────────────────────
section("SUMMARY:");
test("test_the_headline_labels_the_champion_score_as_map50_95", () => {
  // The fault: the summary printed the champion's mAP50-95 under the label
  // the rows use for mAP50, so the champion read 0.826 in the summary and
  // 0.901 on its own row.
  const champ = V.normaliseChampion(CHAMPION, "module");
  const line = V.renderSummaryLine(V.countRuns(V.describeRuns(RUNS, champ, OPEN_PROJECT)), RUNS.length, champ);
  ok(line.indexOf("box mAP50-95") >= 0, `the summary names the strict metric: ${line}`);
  ok(!/box mAP50 <b>/.test(line), "and never the metric the rows show");
  ok(line.indexOf("<b>0.826</b>") >= 0, "the number itself is unchanged");
  const row = rowOf(board(), "run3_heavy_augment");
  ok(row.indexOf("0.901") >= 0, "while the champion's own row still shows its mAP50");
  eq(V.strictFamilyLabel("mask", ""), "outline mAP50-95", "a segmenter champion is labelled in its own family");
  eq(V.familyLabel("box", ""), "box mAP50", "the row label is untouched");
});

test("test_the_runs_waiting_on_a_person_come_first", () => {
  const champ = V.normaliseChampion(CHAMPION, "module");
  const counted = V.countRuns(V.describeRuns(RUNS, champ, OPEN_PROJECT));
  eq(counted.attention, undefined, "no count is kept for a comparison the card no longer makes");
  eq(counted.failed, 1, "one run never finished");
  eq(counted.empty, 1, "one run is unscored");
  const line = V.renderSummaryLine(counted, RUNS.length, champ);
  const empty = line.indexOf("not scored");
  const failed = line.indexOf("did not finish");
  const total = line.indexOf(">runs<") >= 0 ? line.indexOf(">runs<") : line.indexOf(" runs");
  ok(empty > 0 && empty < failed && failed < total,
     `the unscored runs lead, then the ones that stopped, then the totals: ${line}`);
  ok(line.indexOf("is-failed") >= 0, "and the count that names a fault keeps that state's ink");
  ok(line.indexOf("not comparable") < 0, "the comparison count is gone from the line");
});

// The counts are held apart on screen by the flex gap, which a screen
// reader cannot hear: read out of the live region they ran together as
// "3 not comparable5 did not finish". Clipped punctuation separates them.
test("test_the_summary_reads_as_separate_facts_out_of_the_live_region", () => {
  const champ = V.normaliseChampion(CHAMPION, "module");
  const counted = V.countRuns(V.describeRuns(RUNS, champ, OPEN_PROJECT));
  const line = V.renderSummaryLine(counted, RUNS.length, champ);
  const spoken = line.replace(/<[^>]+>/g, "").replace(/\s+/g, " ");
  ok(/not scored, /.test(spoken), `a comma separates the counts: ${spoken}`);
  ok(/projects\. Best so far: /.test(spoken), `a full stop opens the best-so-far sentence: ${spoken}`);
  ok(!/(?:finish|scored|runs|labs|projects)\d/.test(spoken),
     `no count runs into the phrase before it: ${spoken}`);
  ok(line.indexOf('class="vega-runs-sr-only"') >= 0,
     "and the punctuation is clipped out of sight rather than drawn between the counts");
});

test("test_the_summary_says_how_many_rows_a_filter_is_hiding", () => {
  const champ = V.normaliseChampion(CHAMPION, "module");
  const counted = V.countRuns(V.describeRuns(RUNS, champ, OPEN_PROJECT));
  ok(V.renderSummaryLine(counted, RUNS.length, champ).indexOf("Showing") < 0,
     "with nothing filtered the line does not say it");
  ok(V.renderSummaryLine(counted, 3, champ).indexOf("Showing") >= 0, "with rows hidden it does");
});

test("test_no_champion_and_a_missing_champion_are_both_stated", () => {
  const none = V.renderSummaryLine({ total: 1, failed: 0, empty: 0, labs: 0, projects: 1 }, 1, null);
  ok(none.indexOf("No run has been promoted yet.") >= 0, "an unpromoted module says so");
  const gone = V.normaliseChampion(Object.assign({}, CHAMPION, { missing: true }), "module");
  const line = V.renderSummaryLine({ total: 1, failed: 0, empty: 0, labs: 0, projects: 1 }, 1, gone);
  ok(line.indexOf("no longer on disk") >= 0, "a champion whose folder has gone says so");
  ok(line.indexOf("0.826") < 0, "and shows no score, because the record it came from cannot be read");
});

// ── scope ────────────────────────────────────────────────────────────
section("SCOPE:");
test("test_the_champion_scope_is_rendered_honestly", () => {
  const moduleChamp = V.normaliseChampion(CHAMPION, "");
  eq(moduleChamp.scope, "module", "the record's own scope is read");
  const counted = { total: 1, failed: 0, empty: 0, labs: 1, projects: 0 };
  ok(V.renderSummaryLine(counted, 1, moduleChamp).indexOf("Model Inference uses") >= 0,
     "the module record is named as the model Model Inference uses unless another is picked");
  const projectChamp = V.normaliseChampion(Object.assign({}, CHAMPION, { scope: "project" }), "");
  ok(V.renderSummaryLine(counted, 1, projectChamp).indexOf("measures its later runs against") >= 0,
     "the project record is named as the score later runs are measured against, never as the model VEGA starts from");
  const noScope = V.normaliseChampion(Object.assign({}, CHAMPION, { scope: undefined }), undefined);
  eq(noScope.scope, "", "a payload that names no scope leaves it blank");
  const line = V.renderSummaryLine(counted, 1, noScope);
  ok(line.indexOf("every project in VEGA") < 0 && line.indexOf("evaluation baseline") < 0,
     "and the card then claims nothing about scope");
  eq(V.normaliseChampion(CHAMPION, "project").scope, "module",
     "the record's own scope wins over the payload's");
});

// ── filters and sorting ──────────────────────────────────────────────
section("FILTERS:");
test("test_search_state_and_source_compose", () => {
  const champ = V.normaliseChampion(CHAMPION, "module");
  const rows = V.describeRuns(RUNS, champ, OPEN_PROJECT);
  const kept = v => rows.filter(r => V.matchesFilters(r, v)).map(r => r.run.run_name);
  eq(kept(VIEW_ALL).length, RUNS.length, "the widest view keeps every run");
  eq(kept({ search: "coral", state: "all", source: "all", trained: "all" }), ["TRAIN_coral_intel_20260902_224223"],
     "search matches part of a run name");
  eq(kept({ search: "CORAL", state: "all", source: "all", trained: "all" }), ["TRAIN_coral_intel_20260902_224223"],
     "and ignores case");
  eq(kept({ search: "", state: "failed", source: "all", trained: "all" }), ["TRAIN_aborted_20260901_080000"],
     "the state filter keeps one state");
  eq(kept({ search: "", state: "all", source: "mine", trained: "all" }), ["TRAIN_coral_intel_20260902_224223"],
     "the source filter finds the open project's own runs");
  eq(kept({ search: "fish", state: "all", source: "lab", trained: "all" }), [],
     "the filters compose rather than replacing one another");
});

test("test_a_filter_that_matches_nothing_says_so_instead_of_showing_an_empty_table", () => {
  const html = board({ view: { search: "nothing-matches-this", state: "all", source: "all", trained: "all", sort: VIEW_ALL.sort } });
  ok(html.indexOf("No run matches these filters.") >= 0, "the board says why it is empty");
  ok(html.indexOf("<table") < 0, "and draws no headings over nothing");
  ok(html.indexOf("Clear filters") >= 0, "with the way out still on screen");
});

test("test_clear_filters_appears_only_while_a_filter_is_narrowing_the_table", () => {
  const present = { mine: true, other: true, lab: true, trained: [trainedSet("fish_photo_28feb25_paper")] };
  const wide = V.renderFilters(V.normaliseView(VIEW_ALL), present);
  ok(/vega-runs-clear" data-action="clear-filters" hidden/.test(wide),
     `with nothing filtered the button is hidden: ${wide.slice(-400)}`);
  const narrow = V.renderFilters(V.normaliseView({ search: "fish" }), present);
  ok(!/clear-filters" hidden/.test(narrow), "with a filter on it shows");
  const byData = V.renderFilters(V.normaliseView({ trained: trainedSet("fish_photo_28feb25_paper") }), present);
  ok(!/clear-filters" hidden/.test(byData), "and the training-data filter counts as one, like the other three");
});

test("test_every_filter_control_has_a_visible_label_bound_to_it", () => {
  const present = { mine: true, other: true, lab: true, trained: [trainedSet("fish_photo_28feb25_paper")] };
  const html = V.renderFilters(V.normaliseView(VIEW_ALL), present);
  ["s6-runs-search", "s6-runs-state", "s6-runs-source", "s6-runs-trained"].forEach(id => {
    ok(html.indexOf('for="' + id + '"') >= 0, `${id} has a label bound with for=`);
    ok(html.indexOf('id="' + id + '"') >= 0, `${id} exists to be bound to`);
  });
  eq(count(html, 'class="vega-runs-tip"'), 4, "each filter carries its own question glyph");
  // One owner per explanation: with a title on the control as well as on the
  // glyph, the same sentence opened from two hit areas a few pixels apart.
  eq(count(html, 'title="'), 5,
     "four glyph tooltips and the Clear filters button, and none on the controls themselves");
  ok(!/<input[^>]*title=/.test(html), `the search box carries no second copy: ${html}`);
  ok(!/<select[^>]*title=/.test(html), "and neither select does");
});

test("test_the_columns_sort_and_each_one_opens_the_way_it_is_read", () => {
  const champ = V.normaliseChampion(CHAMPION, "module");
  const rows = V.describeRuns(RUNS, champ, OPEN_PROJECT).filter(r => r.run.family === "box");
  const names = sort => V.sortRows(rows, sort).map(r => r.run.run_name);
  const byState = names({ key: "state", dir: "asc" });
  eq(byState[0], "run3_heavy_augment", "the champion heads its group, because every other row is read beside it");
  eq(byState[1], "TRAIN_aborted_20260901_080000",
     "then the row waiting on a person, before every settled one");
  ok(byState.indexOf(LONG_STAMPED) < byState.indexOf(LONG_PLAIN)
     && byState.indexOf(LONG_PLAIN) < byState.indexOf("TRAIN_coral_intel_20260902_224223"),
     `and the scored rows fall best first inside their state: ${byState.join(", ")}`);
  const byScore = names({ key: "score", dir: "desc" });
  eq(byScore[0], "run3_heavy_augment", "the score column opens best first");
  eq(byScore[byScore.length - 1], "TRAIN_aborted_20260901_080000", "a run with no score sorts last");
  eq(names({ key: "score", dir: "asc" })[0], "TRAIN_aborted_20260901_080000", "and the second click reverses it");
  eq(names({ key: "name", dir: "asc" })[0], HOSTILE_NAME, "the name column runs A to Z");
});

test("test_the_sortable_heads_are_sentence_case_and_say_which_one_is_active", () => {
  const html = board();
  ["State", "Run", "Trained on", "Score", "Actions"].forEach(head => {
    ok(html.indexOf(">" + head + "<") >= 0, `the ${head} heading is present in sentence case`);
  });
  ok(html.indexOf('aria-sort="ascending"') >= 0, "the active column says how it is sorted");
  eq(count(html, 'aria-sort="none"'), 2, "the other two sortable columns say they are not");
  ok(html.indexOf('data-sort="state"') >= 0 && html.indexOf('data-sort="score"') >= 0,
     "and each carries the key the click handler reads");
});

test("test_a_damaged_remembered_view_falls_back_rather_than_hiding_every_row", () => {
  const v = V.normaliseView({ state: "nonsense", source: "nonsense", trained: "nonsense", sort: { key: "nonsense", dir: "sideways" } });
  eq(v.state, "all", "an unknown state widens rather than hiding everything");
  eq(v.source, "all", "and so does an unknown source");
  eq(v.trained, "all", "and a training-data value in no shape the card writes");
  eq(v.sort, { key: "state", dir: "asc" }, "an unknown sort key falls back to the default order");
  eq(V.normaliseView(null).search, "", "no remembered view at all is the default view");
});

// ── actions ──────────────────────────────────────────────────────────
section("ACTIONS:");
test("test_an_action_that_cannot_work_is_disabled_with_its_reason_on_the_wrapper", () => {
  const row = rowOf(board(), "TRAIN_aborted_20260901_080000");
  eq(count(row, "<button"), 3, "the row still offers all three actions");
  eq(count(row, " disabled>"), 3, "a run with no weights can do none of them");
  eq(count(row, 'class="vega-runs-action" tabindex="0" title="'), 3,
     "each reason rides on a focusable wrapper, which is what a tooltip lookup resolves");
  ok(!/<button[^>]*disabled[^>]*title=/.test(row) && !/title="[^"]*"[^>]*disabled/.test(row),
     `a disabled button carries no title of its own: ${row}`);
  ok(row.indexOf("no weights file") >= 0, "and the reason says what is missing");
});

// Only a project that pins its Training task refuses a cross-family
// hand-off. The 2026-09-16 review found two faults here: the refusal read
// "This run trained a outlines and this project trains a boxes", because the
// sentence took its nouns from the plurals that name what a model predicts;
// and every cross-family row was disabled whether or not the task could
// move, which took away the host's one-click hand-off (onVegaFinetuneFrom
// switches the control itself) on every segmentation project.
test("test_fine_tune_is_refused_where_the_training_task_is_pinned", () => {
  const seg = V.normaliseRun(RUNS[5]);
  const det = V.normaliseRun(RUNS[3]);
  const pinned = V.finetuneRefusal(seg, "detect", true);
  ok(pinned.indexOf("This run trained a segmenter and this project trains a detector") >= 0,
     `a pinned detector panel refuses a segmenter in a grammatical sentence: ${pinned}`);
  ok(pinned.indexOf("Pick a detector run instead.") >= 0, "and names what to pick instead");
  ok(!/\ba (?:outlines|boxes)\b/.test(pinned), `no plural noun follows the singular article: ${pinned}`);
  eq(V.finetuneRefusal(det, "detect", true), "", "a run of the panel's own family is offered");
  eq(V.finetuneRefusal(V.normaliseRun(run({ task: "", family: "" })), "detect", true), "",
     "a run whose family VEGA cannot read is left alone: an unknown family is not evidence of a mismatch");
  const row = rowOf(board(), "18aprilpm_PA_20260422_135837");
  ok(/data-action="finetune"[^>]*disabled/.test(row), "and the segmenter's row shows it disabled");
  ok(row.indexOf("Pick a detector run instead.") >= 0, "with the reason on its wrapper");
  const own = rowOf(board(), LONG_PLAIN);
  ok(!/data-action="finetune"[^>]*disabled/.test(own), "while a detector row keeps Continue from");
});

test("test_fine_tune_stays_live_across_the_families_when_the_task_can_move", () => {
  const seg = V.normaliseRun(RUNS[5]);
  eq(V.finetuneRefusal(seg, "detect", false), "",
     "an unpinned panel refuses nothing: the host moves the Training task and hands the weights over");
  const offer = V.finetuneOffer(seg, "detect", false);
  eq(offer.disabled, false, "so the button stays live");
  ok(offer.tip.indexOf("sets Training task to Segmentation") >= 0,
     `and its sentence says what VICARIUS moves, in the control's own words: ${offer.tip}`);
  const row = rowOf(board({ taskPinned: false }), "18aprilpm_PA_20260422_135837");
  ok(!/data-action="finetune"[^>]*disabled/.test(row), "the rendered row keeps Continue from");
  ok(row.indexOf("sets Training task to Segmentation") >= 0, "with the hand-off sentence on the button itself");
  const aborted = rowOf(board({ taskPinned: false }), "TRAIN_aborted_20260901_080000");
  ok(/data-action="finetune"[^>]*disabled/.test(aborted),
     "while a run with no weights file is still refused, whatever the task is set to");
});

test("test_promote_is_offered_only_where_it_can_work", () => {
  const html = board();
  ok(rowOf(html, "run3_heavy_augment").indexOf('data-action="promote"') < 0,
     "the champion is never offered promotion to what it already is");
  ok(/data-action="promote"[^>]*disabled/.test(rowOf(html, "18aprilpm_PA_20260422_143501")),
     "an unscored run cannot be promoted");
  ok(rowOf(html, "18aprilpm_PA_20260422_143501").indexOf("Evaluate this run first.") >= 0,
     "and the reason says what to do first");
  const fromFolder = rowOf(html, LONG_STAMPED);
  ok(/data-action="promote"[^>]*disabled/.test(fromFolder),
     "a score read from an evaluation folder is not a ledger row, so Promote waits");
  ok(fromFolder.indexOf("score it again to make it the best so far") >= 0, "the meta line names the action that lifts the block");
  ok(!/data-action="promote"[^>]*disabled/.test(rowOf(html, LONG_PLAIN)), "a ledger-scored run can be promoted");
});

test("test_an_action_carries_the_run_identity_the_host_wires_on", () => {
  const row = rowOf(board(), LONG_PLAIN);
  ok(row.indexOf('data-run-dir="/inprocess/_models/fish_photo_28feb25_paper/runs/' + LONG_PLAIN + '"') >= 0,
     "every button names the run folder");
  ok(row.indexOf('data-run-name="' + LONG_PLAIN + '"') >= 0, "and the run name");
  ok(row.indexOf('data-task="detect"') >= 0, "and the task the host's family pass reads");
});

test("test_promote_names_the_run_in_the_hosts_confirm_dialog", () => {
  const root = fakeRoot();
  root.querySelector = sel => (sel.indexOf("body") >= 0 ? root.body : null);
  const messages = [];
  const el = { set textContent(v) { messages.push(v); }, get textContent() { return messages[messages.length - 1]; } };
  global.document = { getElementById: id => (id === "confirm-msg" ? el : null) };
  try {
    // The card must know the run before it can name it, so the payload is
    // loaded first through the same render path the page uses.
    return V.load(root, { fetchJson: url => Promise.resolve(url === V.RUNS_URL ? DATA : { loaded: false }) })
      .then(() => {
        V.dispatchAction(root, "promote", { run_dir: RUNS[3].run_dir, run_name: LONG_PLAIN, task: "detect" });
        eq(root.dispatched.length, 1, "the host still gets its event");
        eq(root.dispatched[0].detail.run_name, LONG_PLAIN, "carrying the run name");
        eq(root.dispatched[0].detail.map50, 0.89, "and the score, so a host dialog need not read the table back");
        ok(messages.length === 1 && messages[0].indexOf(LONG_PLAIN) >= 0,
           `the confirm sentence names the run: ${messages[0]}`);
        ok(messages[0].indexOf("box mAP50 0.890") >= 0, "and its score in the metric its row shows");
        ok(messages[0].indexOf("fish_photo_28feb25_paper") >= 0, "and where it was trained");
      })
      .then(() => { delete global.document; }, e => { delete global.document; throw e; });
  } catch (e) {
    delete global.document;
    throw e;
  }
});

// ── stripes and chrome ───────────────────────────────────────────────
section("STRIPES:");
test("test_every_other_record_is_striped", () => {
  const html = board();
  const rows = dataRows(html);
  eq(rows.length, RUNS.length, "one record per run");
  // Striping restarts inside each group, so the class is counted per group
  // rather than across the table: six detectors and two segmenters here.
  eq(count(html, "is-alt"), 4, "three of the six detectors' neighbours and one of the two segmenters' are washed");
  const detectors = html.slice(html.indexOf("draw boxes"), html.indexOf("draw outlines"));
  const flags = dataRows(detectors).map(r => r.indexOf("is-alt") >= 0);
  eq(flags, [false, true, false, true, false, true], "the wash alternates from the first record of the group");
});

test("test_the_champion_record_is_marked_and_never_striped_over", () => {
  const row = rowOf(board(), "run3_heavy_augment");
  ok(row.indexOf("is-champion-record") >= 0, "the champion row is its own wash");
  ok(row.indexOf("vega-runs-chip is-champion") >= 0, "and carries the champion chip");
});

test("test_the_table_keeps_its_semantics_for_the_stacked_layout", () => {
  const html = board();
  ok(html.indexOf('<table class="vega-runs-table"') >= 0, "the table is still a table element");
  ok(/<table[^>]*role="table"/.test(html), "and declares role=table, which display:block would otherwise strip");
  eq(count(html, 'role="rowgroup"'), 1 + 2 + RUNS.length, "the head, the two group headings and every record");
  eq(count(html, 'role="row"'), 1 + 2 + RUNS.length * 2, "a heading row, a row per group, and two rows per record");
  eq(count(html, 'role="columnheader"'), 5 + 2, "the five column headings plus one per group heading");
  eq(count(html, 'role="cell"'), RUNS.length * 6, "five cells on the main row and one on the meta row, per record");
});

// ── adversarial ──────────────────────────────────────────────────────
section("ADVERSARIAL:");
test("test_junk_payloads_render_an_error_rather_than_throwing", () => {
  [null, undefined, "runs", 7, [], { runs: "nope" }].forEach(bad => {
    const html = V.render(bad, DETECT_PANEL);
    ok(html.indexOf("vega-runs-error") >= 0, `a payload of ${JSON.stringify(bad)} renders the error line`);
  });
  ok(V.render({ error: "disk is full" }, DETECT_PANEL).indexOf("disk is full") >= 0, "a server error is shown as sent");
  ok(V.render({ ok: true, runs: [] }, DETECT_PANEL).indexOf("No trained models found") >= 0, "an empty list says so");
});

test("test_a_junk_run_never_hides_the_rest", () => {
  const html = V.render({ ok: true, runs: [null, "nope", 7, run({ run_name: "good" })], champion: null }, DETECT_PANEL);
  eq(dataRows(html).length, 4, "every entry still renders as a record");
  ok(html.indexOf("(unnamed run)") >= 0, "a run with no name says so rather than rendering a blank");
  ok(html.indexOf("good") >= 0, "and the readable run is still there");
});

test("test_a_score_that_is_not_a_number_reads_as_missing", () => {
  eq(V.normaliseRun({ map50: "0.9" }).map50, null, "a score sent as text is not parsed");
  eq(V.normaliseRun({ map50: 1.4 }).map50, null, "a score outside 0 to 1 is a ledger fault, not a number");
  eq(V.normaliseRun({ map50: NaN }).map50, null, "NaN is not a score");
  eq(V.normaliseRun({ imgsz: 0 }).imgsz, null, "an image size of zero is not a size");
  eq(V.normaliseRun({ imgsz: 640.5 }).imgsz, null, "and neither is a fraction of a pixel");
});

test("test_an_unknown_task_is_stated_rather_than_guessed", () => {
  eq(V.taskLabel("classify"), { word: "unknown task", code: "classify" }, "the code is kept beside the plain word");
  eq(V.normaliseRun({ task: "classify" }).family, "", "and no metric family is invented for it");
});

test("test_hostile_names_are_escaped_everywhere_they_appear", () => {
  const html = board();
  ok(html.indexOf("<script>alert") < 0, "no run name reaches the page as markup");
  ok(html.indexOf("&lt;script&gt;") >= 0, "it is escaped instead");
  ok(html.indexOf("<b>evil</b>") < 0, "and neither does a lab name");
  ok(V.esc(null) === "" && V.esc(undefined) === "", "null and undefined escape to nothing");
});

test("test_an_imported_run_says_VEGA_did_not_train_it", () => {
  const row = rowOf(board(), "&lt;script&gt;");
  ok(row.indexOf("vega-runs-chip is-imported") >= 0, "the imported marker is on the row");
  ok(row.indexOf("VEGA did not train this model") >= 0, "and says so in the marker's tooltip");
  ok(row.indexOf("oceankind_CV") >= 0, "naming where it was copied from");
  const bare = V.render({ ok: true, runs: [run({ origin_kind: "imported", imported_from: "", run_name: "bare" })], champion: null }, DETECT_PANEL);
  ok(bare.indexOf("a folder its provenance file does not name") >= 0,
     "an imported run with no source says that, rather than showing a blank");
});

test("test_a_timestamp_is_AST_whatever_the_browser_thinks", () => {
  eq(V.formatAst("2026-09-15T05:04:02-04:00"), "2026-09-15 05:04 AST", "an offset stamp is read at its offset");
  eq(V.formatAst("2026-09-15T09:04:02Z"), "2026-09-15 05:04 AST", "a UTC stamp is shifted");
  eq(V.formatAst("2026-09-15 05:04"), "2026-09-15 05:04 AST", "a naive stamp is AST by house rule");
  eq(V.formatAst("not a date"), "", "junk yields nothing rather than an invalid date");
  eq(V.formatAst("2026-13-40T99:99:99-04:00"), "", "and so does an impossible one");
  eq(V.formatAst(null), "", "as does a missing one");
});

test("test_a_click_on_a_disabled_button_or_outside_any_control_does_nothing", () => {
  const root = fakeRoot();
  V.handleClick(root, { target: fakeButton({ "data-action": "promote", "data-run-dir": "/x" }, true) });
  eq(root.dispatched.length, 0, "a disabled action dispatches nothing");
  V.handleClick(root, { target: { closest: () => null } });
  eq(root.dispatched.length, 0, "a click outside any control dispatches nothing");
  V.handleClick(root, { target: fakeButton({ "data-action": "nonsense" }, false) });
  eq(root.dispatched.length, 0, "an unknown action dispatches nothing");
  V.handleClick(root, {});
  V.handleClick(root, { target: {} });
  eq(root.dispatched.length, 0, "and neither does a malformed event");
  let threw = false;
  try { V.dispatchAction(root, "nonsense", {}); } catch (e) { threw = e instanceof RangeError; }
  ok(threw, "asking for an action that does not exist is a programming error and says so");
});

test("test_a_filter_event_from_something_that_is_not_a_filter_is_ignored", () => {
  const root = fakeRoot();
  V.handleFilter(root, { target: { getAttribute: () => null, value: "x" } });
  V.handleFilter(root, {});
  V.handleFilter(root, { target: {} });
  ok(true, "no throw from an event that names no filter");
});

test("test_storage_that_refuses_never_breaks_the_card", () => {
  const original = global.localStorage;
  global.localStorage = {
    getItem() { throw new Error("site data is off"); },
    setItem() { throw new Error("site data is off"); },
  };
  try {
    const root = fakeRoot();
    return V.init(root, { fetchJson: url => Promise.resolve(url === V.RUNS_URL ? DATA : { loaded: false }) })
      .then(() => {
        ok(root.body.innerHTML.indexOf("draw boxes") >= 0, "the card renders with no remembered view at all");
      })
      .then(() => { global.localStorage = original; }, e => { global.localStorage = original; throw e; });
  } catch (e) {
    global.localStorage = original;
    throw e;
  }
});

// ── the DOM glue ─────────────────────────────────────────────────────
section("GLUE:");
test("test_load_renders_the_payload_and_marks_the_open_project", () => {
  const root = fakeRoot();
  const asked = [];
  return V.load(root, {
    fetchJson: url => {
      asked.push(url);
      return Promise.resolve(url === V.RUNS_URL ? DATA : { loaded: true, state: {
        project_dir: OPEN_PROJECT, training_source_name: "coral_intel", training_source_label: "box dataset" } });
    },
  }).then(() => {
    eq(asked.sort(), ["/api/project/state", "/api/step/6/all_runs"], "both routes are read");
    ok(root.body.innerHTML.indexOf("is-mine-record") >= 0, "and the open project's own run is marked");
    ok(rowOf(root.body.innerHTML, "TRAIN_coral_intel_20260902_224223").indexOf(">coral_intel<") >= 0,
       "and named after the source the state route records for the open project");
  });
});

test("test_a_project_state_route_that_fails_is_not_an_error", () => {
  const root = fakeRoot();
  return V.load(root, {
    fetchJson: url => (url === V.RUNS_URL ? Promise.resolve(DATA) : Promise.reject(new Error("no project route"))),
  }).then(() => {
    ok(root.body.innerHTML.indexOf("draw boxes") >= 0, "the table still renders");
    ok(root.body.innerHTML.indexOf("is-mine-record") < 0, "with nothing marked as the open project's");
  });
});

test("test_a_failing_run_route_shows_its_message", () => {
  const root = fakeRoot();
  return V.load(root, { fetchJson: () => Promise.reject(new Error("HTTP 500 from /api/step/6/all_runs")) })
    .then(() => {
      ok(root.body.innerHTML.indexOf("HTTP 500") >= 0, "the server's words reach the card");
      ok(root.body.innerHTML.indexOf("vega-runs-error") >= 0, "in the error line");
    });
});

test("test_a_fetch_that_throws_before_it_returns_a_promise_is_caught", () => {
  const root = fakeRoot();
  return V.load(root, { fetchJson: () => { throw new Error("no network at all"); } }).then(() => {
    ok(root.body.innerHTML.indexOf("no network at all") >= 0, "a synchronous throw is an error line, not a crash");
  });
});

test("test_a_stale_answer_never_overwrites_a_fresh_one", () => {
  const root = fakeRoot();
  let resolveSlow;
  const slow = new Promise(r => { resolveSlow = r; });
  const first = V.load(root, { fetchJson: url => (url === V.RUNS_URL ? slow : Promise.resolve({ loaded: false })) });
  const second = V.load(root, { fetchJson: url => Promise.resolve(url === V.RUNS_URL ? DATA : { loaded: false }) });
  return second.then(() => {
    resolveSlow({ ok: true, runs: [run({ run_name: "STALE" })], champion: null });
    return first;
  }).then(() => {
    ok(root.body.innerHTML.indexOf("STALE") < 0, "the older answer is dropped");
    ok(root.body.innerHTML.indexOf("run3_heavy_augment") >= 0, "and the newer list stands");
  });
});

test("test_init_binds_its_listeners_once", () => {
  const root = fakeRoot();
  const fetchJson = url => Promise.resolve(url === V.RUNS_URL ? DATA : { loaded: false });
  return V.init(root, { fetchJson }).then(() => V.init(root)).then(() => {
    eq(Object.keys(root.listeners).sort(), ["change", "click", "input"], "click, input and change are wired");
    eq(root.listeners.click.length, 1, "and the click handler is bound exactly once across two calls");
  });
});

test("test_a_card_root_that_is_not_there_says_which_element_is_missing", () => {
  let message = "";
  try { V.load(null); } catch (e) { message = e.message; }
  ok(message.indexOf("s6-runs-card") >= 0, `the error names the contract id: ${message}`);
  message = "";
  try { V.load({ querySelector: () => null }); } catch (e) { message = e.message; }
  ok(message.indexOf('data-role="body"') >= 0, "or the body element it renders into");
});

test("test_initStep6Runs_survives_a_page_without_the_card", () => {
  const errors = [];
  const originalError = console.error;
  console.error = msg => errors.push(String(msg));
  let result;
  try {
    result = V.initStep6Runs();
  } finally {
    console.error = originalError;
  }
  ok(result && typeof result.then === "function", "without a document it returns a promise instead of throwing");
  ok(errors.length === 1 && errors[0].indexOf("s6-runs-card") >= 0, "the missing card is logged with its id");
  global.document = { getElementById: () => null };
  const more = [];
  console.error = msg => more.push(String(msg));
  let second;
  try {
    second = V.initStep6Runs();
  } finally {
    console.error = originalError;
    delete global.document;
  }
  ok(second && typeof second.then === "function", "with a document but no card it still returns a promise");
});

// ── the stylesheet ───────────────────────────────────────────────────
section("CSS:");
const RUNS_CSS = fs.readFileSync(path.join(__dirname, "..", "static", "orchestrator_runs.css"), "utf8");
/** The declarations of one selector, or "" when the stylesheet has no such rule. */
function declsOf(selector) {
  const m = new RegExp(selector.replace(/[.[\]()*+?^$|\\]/g, "\\$&") + "\\s*\\{([^}]*)\\}").exec(RUNS_CSS);
  return m ? m[1] : "";
}
/** The body of the stylesheet's @container at-rule, brace-matched. */
function containerBlock() {
  const start = RUNS_CSS.indexOf("@container");
  const open = RUNS_CSS.indexOf("{", start);
  let depth = 0;
  for (let i = open; i < RUNS_CSS.length; i += 1) {
    if (RUNS_CSS[i] === "{") depth += 1;
    else if (RUNS_CSS[i] === "}") { depth -= 1; if (depth === 0) return RUNS_CSS.slice(open + 1, i); }
  }
  return "";
}

test("test_no_user_visible_text_sits_below_the_legibility_floor", () => {
  const sizes = [];
  const re = /font(?:-size)?:\s*(?:[^;{}]*?\b)?(\d+)px/g;
  let m;
  while ((m = re.exec(RUNS_CSS)) !== null) sizes.push(Number(m[1]));
  ok(sizes.length > 8, `the stylesheet really does set type: ${sizes.join(", ")}`);
  const tooSmall = sizes.filter(n => n < 12);
  eq(tooSmall, [], "version 2 of the design puts no user-visible text below 12 px");
  ok(/\.vega-runs-table\s*\{[^}]*13px/.test(RUNS_CSS), "table cells are 13 px");
  ok(/\.vega-runs-meta-line\s*\{[^}]*font-size:\s*13px/.test(RUNS_CSS), "so are the meta facts");
  ok(/\.vega-runs-chip\s*\{[^}]*12px/.test(RUNS_CSS), "chips are 12 px");
  ok(/\.vega-runs-trained\s*\{[^}]*font-size:\s*13px/.test(RUNS_CSS),
     "the data a run trained on is a table cell and reads at the cell size");
  ok(/\.vega-runs-trained\s*\{[^}]*text-overflow:\s*ellipsis/.test(RUNS_CSS),
     "and shortens only on real overflow, with the whole name in its tooltip");
  ok(/\.vega-runs \.btn-sm\s*\{[^}]*font-size:\s*12px/.test(RUNS_CSS),
     "and the row buttons are raised to 12 px from the shared block's 11");
});

test("test_hint_prose_never_carries_a_hardcoded_grey", () => {
  const body = RUNS_CSS.replace(/\/\*[\s\S]*?\*\//g, "");
  eq(body.match(/#[0-9a-fA-F]{3,8}\b/g), null, "every colour is a token or an rgba wash of one");
  ok(body.indexOf("var(--text-dim,") > 0, "secondary ink reads the token that holds 4.5:1");
  ok(body.indexOf("var(--vr-text-dim)") > 0, "and every hint is painted through its alias");
  ok(body.indexOf("#666") < 0, "the grey that measured 3.03:1 is nowhere in this card");
});

test("test_the_rows_stripe_and_wash_on_hover", () => {
  // Mixed from the tokens rather than written as a literal rgba of their
  // values, so a host that moves a token moves the wash with it.
  const wash = /background:\s*color-mix\(in srgb, var\(--vr-[a-z-]+\) \d+%, transparent\)/;
  ok(wash.test(/\.vega-runs-record\.is-alt\s*\{([^}]*)\}/.exec(RUNS_CSS)?.[1] || ""),
     "every other record carries a visible wash, the version 2 row-legibility rule");
  ok(wash.test(/\.vega-runs-record:hover\s*\{([^}]*)\}/.exec(RUNS_CSS)?.[1] || ""),
     "and a row washes under the pointer");
  ok(wash.test(/\.vega-runs-record\.is-champion-record\s*\{([^}]*)\}/.exec(RUNS_CSS)?.[1] || ""),
     "the champion has its own");
  ok(/\.vega-runs-record\.is-mine-record td:first-child\s*\{[^}]*box-shadow/.test(RUNS_CSS),
     "and the open project's own runs wear an edge rather than a word in the name column");
});

test("test_the_five_columns_are_pinned_around_the_run_name", () => {
  const width = sel => Number((new RegExp(sel + "\\s*\\{[^}]*width:\\s*(\\d+)px").exec(RUNS_CSS) || [])[1]);
  const pinned = ["\\.vega-runs-col-state", "\\.vega-runs-col-trained", "\\.vega-runs-col-score",
                  "\\.vega-runs-col-actions"].map(width);
  pinned.forEach((w, i) => ok(w > 0, `pinned column ${i} declares a width: ${w}`));
  ok(pinned[1] >= 200, `the Trained on column holds a data set name of ordinary length whole: ${pinned[1]} px`);
  const floor = Number(/min-width:\s*(\d+)px/.exec(/\.vega-runs-table\s*\{([^}]*)\}/.exec(RUNS_CSS)[1])[1]);
  const run = floor - pinned.reduce((a, b) => a + b, 0);
  ok(run >= 200, `the run name keeps ${run} px at the table's floor, which no pinned column may eat`);
  ok(!/\.vega-runs-col-run\s*\{/.test(RUNS_CSS),
     "and the run column is never pinned itself: it is the one that absorbs the squeeze");
});

test("test_the_name_can_never_wrap_or_break_a_token", () => {
  ok(/\.vega-runs-name\s*\{[^}]*white-space:\s*nowrap/.test(RUNS_CSS),
     "the missing rule that let a split name wrap onto two lines is now declared");
  ok(/\.vega-runs-name-tail\s*\{[^}]*flex:\s*none/.test(RUNS_CSS), "the stamp never shrinks");
  ok(/\.vega-runs-name-head\s*\{[^}]*text-overflow:\s*ellipsis/.test(RUNS_CSS), "only the head shortens");
  ok(!/\.vega-runs-name-head\s*\{[^}]*color:/.test(RUNS_CSS),
     "and both halves keep one ink, so a shortened name never reads as two fragments");
});

test("test_the_card_has_panel_chrome_and_fills_its_width", () => {
  const card = declsOf(".vega-runs");
  ok(card.indexOf("border: 1px solid var(--vr-border)") >= 0, "the card sits inside a --border panel");
  ok(card.indexOf("border-radius: var(--vr-radius-lg)") >= 0, "with the panel radius");
  ok(/\.vega-runs-table\s*\{[^}]*width:\s*100%/.test(RUNS_CSS), "and the table fills it rather than sitting in a column");
});

test("test_the_narrow_card_stacks_instead_of_cutting_a_column", () => {
  const block = containerBlock();
  ok(block.length > 0, "the stylesheet declares the narrow layout");
  const at = RUNS_CSS.match(/@container\s+vega-runs\s*\(max-width:\s*(\d+)px\)/);
  ok(at, "the query names the container and a pixel threshold");
  const minWidth = /min-width:\s*(\d+)px/.exec(declsOf(".vega-runs-table"));
  ok(minWidth, "the wide table still declares its minimum width");
  if (at && minWidth) {
    eq(Number(at[1]), Number(minWidth[1]) - 1,
       "the stack begins exactly one pixel below the width the four columns need");
  }
  ok(/\.vega-runs-table\s*\{[^}]*display:\s*block/.test(block), "the table becomes a block stack");
  ok(/\.vega-runs-table\s*\{[^}]*min-width:\s*0/.test(block), "the stacked table drops the wide floor");
  ok(/\.vega-runs-main\s*\{[^}]*display:\s*flex/.test(block), "the main row becomes a flex line the cells can wrap inside");
  ok(/\.vega-runs-actions\s*\{[^}]*flex-wrap:\s*wrap/.test(block), "the action buttons wrap rather than being cut");
  ok(/\.vega-runs-chip\s*\{[^}]*max-width:\s*100%/.test(block), "a state chip gives way to a card narrower than its cap");
  ok(/\.vega-runs-trained\s*\{[^}]*max-width/.test(block), "the training data name's cap keys off the card's width");
  ok(/td:nth-child\(5\)/.test(block), "and all five cells are placed in the stack, the actions last");
  ok(/\.vega-runs-group-cell\s*\{[^}]*display:\s*block/.test(block), "and a group heading stacks with the records");
  ok(RUNS_CSS.indexOf("@media") < 0,
     "no rule in this card keys off the viewport: the VEGA window is resized inside the VICARIUS desktop");
  ok(block.indexOf("thead") >= 0, "the column headings are handled: they no longer sit over the stacked cells");
  ok(!/display:\s*none/.test(/\.vega-runs-table thead\s*\{([^}]*)\}/.exec(block)?.[1] || ""),
     "the headings are hidden from the eye, not removed from the accessibility tree");
});

test("test_every_focusable_thing_shows_a_focus_ring", () => {
  ["\\.vega-runs \\.btn:focus-visible", "\\.vega-runs-tip:focus-visible", "\\.vega-runs-chip:focus-visible",
   "\\.vega-runs-action:focus-visible", "\\.vega-runs-th:focus-visible",
   "\\.vega-runs-score\\[tabindex\\]:focus-visible"].forEach(sel => {
    ok(new RegExp(sel).test(RUNS_CSS), `${sel.replace(/\\/g, "")} carries the ring`);
  });
  ok(/outline:\s*2px solid var\(--vr-pink\)/.test(RUNS_CSS), "the ring is the design's 2 px pink");
  ok(/\.vega-runs \.btn-sm\s*\{[^}]*min-height:\s*28px/.test(RUNS_CSS), "and every hit target clears 28 px");
});

chain.then(() => {
  console.log(`vega_runs: ${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
});

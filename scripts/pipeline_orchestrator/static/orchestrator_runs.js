/* orchestrator_runs.js: the All trained models registry (VEGA), on the
   Evaluate Model panel since 2026-09-21 (it rendered on the Train panel
   before that date; see spec 2026-09-21-vega-starting-model-and-training-
   data-design.md section 4).

   Every trained run the module can see, as a registry table built to
   docs/superpowers/specs/2026-09-02-matrix-window-design.md (states, filters,
   board, handoffs) and to version 2 of the workflow page design (type scale
   and the legibility floor: cells 13 px, row names Space Mono 700 13 px,
   chips 12 px, visible row striping and a hover wash).

   Shape: an IIFE exposing window.VegaRuns. The render functions are pure
   (data in, HTML string out) so Node 18 can test them with no browser; the
   DOM glue (load, init, handleClick) is separate and takes the root element
   and an injectable fetch. The card fetches its own data from
   GET /api/step/6/all_runs and GET /api/project/state, and never calls the
   backend for an action: each row button dispatches a CustomEvent on the card
   root (vega:evaluate-run, vega:promote-run, vega:finetune-from) that the host
   page wires. The detail carries the run's identity and its score so the host
   and the confirm dialog can name what is about to change.

   Every string from the server (run names and project names come from folder
   names on disk) passes through esc() before it reaches the page.

   States: the process-state family of the registry spec, mapped onto four of
   its eight roles. A run with no weights file failed; a run with weights and
   no score is empty (nothing recorded about its quality); a scored run is done
   by machine; the promoted champion is verified. The summary line states the
   runs waiting on a person first, which is what somebody opens this card to
   find.

   Comparability came off the screen on 2026-09-16, by Lauren's ruling that
   comparing model runs is not what this card is for and that simple is better
   here. The route still sends comparable_to_champion and
   comparability_reason, and normaliseRun still reads them, so the payload and
   every route are untouched; the card draws nothing from them. What it draws
   instead is the data each run trained on.

   Trained on: the data subset a run learned from (a staged box dataset such
   as coral_intel_subset, a training set, or the outlines a project exported
   from its own images), as its own column and its own filter. The name is
   the route's own trained_on field, with trained_on_kind saying which of the
   three it is; for a run the open project trained, it is the project's own
   record instead (training_source_name and training_source_label from the
   state route), because that project trains from that one source. It is
   never the folder that holds the run: a lab or project folder is not a
   data set, and naming one asserted a data set for folders that hold none.
   A record naming neither reads "not recorded".

   Families: a detector's boxes and a segmenter's outlines are scored in
   different metric families, so the table groups them under their own
   headings rather than interleaving them, and the family name rides in the
   group heading instead of on every row.

   Names: a run name is never broken mid-token. A name that ends in a run
   stamp (_YYYYMMDD_HHMMSS or -YYYYMMDD-HHMMSS) splits at that boundary so the
   stamp always shows and only the head shortens on real overflow; a name
   without a stamp is one span that shortens at its end and only when it does
   not fit. Both halves keep one ink, so a shortened name never reads as two
   fragments, and the full name is in the row's tooltip.

   Width: the card has one layout per width, chosen by the stylesheet's
   container query on .vega-runs-wrap. Wide enough for the four columns, the
   record is a table row; narrower, it stacks so no column is cut at the
   card's edge. The rendered markup declares role="table" and its row, cell
   and header roles so the stacked layout keeps the table's semantics.

   Disabled actions: one element owns the reason an action is unavailable.
   A disabled button dispatches no pointer events, so the reason is written
   on the focusable wrapper around it and never on the button as well, or
   the button's own title would be the one a tooltip lookup resolves.

   Imported runs: a row may carry origin_kind "imported" with imported_from
   and imported_at, which the card shows as a marker in the meta line so a
   model somebody else trained never reads as one VEGA trained. All three
   are per-run facts: the route reads them from that run's own
   provenance.json (imported_from, imported_at), or decides origin_kind from
   the run's name appearing in the lab's imported_runs list, and never from
   the lab's flag alone. lab.json carries origin_kind once for the whole lab,
   so a run trained into an imported lab afterwards would inherit a claim its
   own record does not make. On 2026-09-17 the route sends none of the three
   yet, so the marker draws for no row; it stays because the fix is the
   route's (a per-run read of provenance.json, owned with app.py), the
   contract is this paragraph, and a row without the fields renders exactly
   as if the marker did not exist.

   Champion scope: the champion record says which of the two records it is.
   "module" is inprocess/_models/champion.json, the model Model Inference
   uses unless another is picked and the default choice under Continue from
   a VEGA model; "project" is one project's own evaluation baseline
   (gate_baseline). The summary says which in plain words and never calls one
   the other. A payload with no scope says nothing about scope at all. */
(function () {
  "use strict";

  var RUNS_URL = "/api/step/6/all_runs";
  var PROJECT_STATE_URL = "/api/project/state";
  var EVENTS = {
    evaluate: "vega:evaluate-run",
    promote: "vega:promote-run",
    finetune: "vega:finetune-from"
  };
  var ACTION_REFRESH = "refresh";
  var ACTION_CLEAR = "clear-filters";
  var BODY_SELECTOR = '[data-role="body"]';
  var BOARD_SELECTOR = '[data-role="board"]';
  var SUMMARY_SELECTOR = '[data-role="summary"]';
  var CLEAR_SELECTOR = '[data-action="clear-filters"]';
  var CARD_ID = "s6-runs-card";
  var TITLE_ID = "s6-runs-title";
  var TASK_SELECT_ID = "s6-task";
  var CONFIRM_MSG_ID = "confirm-msg";
  var AST_OFFSET_MINUTES = -240;
  var MS_PER_MINUTE = 60000;
  var SCORE_DECIMALS = 3;
  var COLUMN_COUNT = 5;
  var ORIGIN_LAB = "lab";
  var ORIGIN_PROJECT = "project";
  /* This run was copied in from training somebody ran outside this module.
     The fact belongs to the run, not to its lab: the importer writes
     imported_from and imported_at into each imported run's provenance.json
     and lists those run folders in the lab's imported_runs, so the route
     derives origin_kind from one of those two per-run sources and never from
     the lab's flag alone. A marked run's score is then never read as VEGA's
     own work, and a run trained into the lab later stays unmarked. */
  var ORIGIN_KIND_IMPORTED = "imported";
  /* Where the server read a run's score. "ledger" is a row in the run's
     rounds ledger; "evaluation" is the run's newest evaluation folder, read
     because no ledger row names the run; "" is no score at all. Only a
     ledger score can be promoted. */
  var SCORE_SOURCE_LEDGER = "ledger";
  var SCORE_SOURCE_EVALUATION = "evaluation";
  var SCORE_SOURCES = [SCORE_SOURCE_LEDGER, SCORE_SOURCE_EVALUATION];
  /* The two champion records the module keeps, named by the route. */
  var SCOPE_MODULE = "module";
  var SCOPE_PROJECT = "project";
  var TASK_WORDS = { seg: "outlines", detect: "boxes" };
  var TASK_TO_FAMILY = { seg: "mask", detect: "box" };
  var FAMILY_WORDS = { mask: "outline", box: "box" };
  /* What a model of each family is called in a sentence. FAMILY_WORDS names
     what a model predicts ("outline mAP50"); these name the model itself
     ("this run trained a segmenter"), which takes an article. The host page
     says the same thing with the same two words (STEP6_TASK_NOUNS in
     orchestrator.js), so the card and the panel agree. */
  var FAMILY_MODEL_NOUN = { mask: "segmenter", box: "detector" };
  /* The Training task control's own option words, so a sentence that asks
     for a change names the choice exactly as the control spells it. */
  var FAMILY_TASK_OPTION = { mask: "Segmentation", box: "Detector" };
  // 2026-09-17 (LO): named for what the model does, not for what the code calls
  // it. Lauren walked the tutorial and flagged this row of words as jargon.
  var FAMILY_GROUP_NAME = { box: "Models that draw boxes", mask: "Models that draw outlines" };
  var FAMILY_ORDER = ["box", "mask", ""];
  /* The Train panel's own task control, whose value decides which family the
     card lists first and whether a run's weights can start the next run. */
  var PANEL_TASK_DETECT = "detect";
  var PANEL_TASK_SEGMENT = "segment";
  var PANEL_TASK_FAMILY = { detect: "box", segment: "mask" };
  var UNKNOWN_TASK_WORD = "unknown task";
  var UNNAMED_RUN = "(unnamed run)";
  var UNGROUPED_NAME = "Runs whose family VEGA cannot read";
  /* A run folder's trailing stamp, the only boundary a long name is ever
     shortened at. Both spellings are in use on disk. */
  var NAME_STAMP_PATTERN = /^(.+?)([_-]\d{8}[_-]\d{6})$/;
  /* The three kinds of training data a run can have learned from, as the
     route's trained_on_kind names them: a staged box dataset (the Bounding
     Boxes page stages these under supporting_data/bbox_datasets), a training
     set built by the Coverage Matrix, or the outlines an ordinary project
     exported from its own images. A kind outside these three reads as no
     kind, and the cell then names the data set with no noun before it. */
  var TRAINED_KIND_BBOX = "bbox";
  var TRAINED_KIND_COMPOSITE = "composite";
  var TRAINED_KIND_EXPORT = "export";
  var TRAINED_KINDS = [TRAINED_KIND_BBOX, TRAINED_KIND_COMPOSITE, TRAINED_KIND_EXPORT];
  /* The state route's own words for the open project's source
     (training_source_label), mapped onto the kinds above. */
  var STATE_SOURCE_LABEL_KIND = { "box dataset": TRAINED_KIND_BBOX, "training set": TRAINED_KIND_COMPOSITE };
  var STAMP_PATTERN = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?\s*(Z|[+-]\d{2}:?\d{2})?$/;
  var MAX_MONTH = 12;
  var MAX_DAY = 31;
  var MAX_HOUR = 23;
  var MAX_MINUTE = 59;
  var MAX_SECOND = 60;

  /* Every remembered choice on this card, per browser, under the one prefix
     the registry spec asks for. */
  var STORE_PREFIX = "vega.runs.";
  var STORE_SEARCH = STORE_PREFIX + "search";
  var STORE_STATE = STORE_PREFIX + "state";
  var STORE_SOURCE = STORE_PREFIX + "source";
  var STORE_TRAINED = STORE_PREFIX + "trainedOn";
  var STORE_SORT = STORE_PREFIX + "sort";
  var STORE_DIR = STORE_PREFIX + "dir";

  var FILTER_ALL = "all";
  /* The Trained on filter's option values. A data set's own name cannot be
     used as the value: a folder called "all" would then read as every data
     set. So a named set carries a prefix no bare value has, and the runs whose
     record names no data carry their own value. */
  var TRAINED_PREFIX = "set:";
  var TRAINED_NONE = "none";
  var SOURCE_MINE = "mine";
  var SOURCE_OTHER = "other";
  var SOURCE_LAB = "lab";
  var SORT_STATE = "state";
  var SORT_NAME = "name";
  var SORT_SCORE = "score";
  var SORT_KEYS = [SORT_STATE, SORT_NAME, SORT_SCORE];
  var DIR_ASC = "asc";
  var DIR_DESC = "desc";
  /* Which way each column runs on its first click, so one click gives what
     the column is read for: the state order from the champion down through
     the runs needing a decision, names from A to Z, and scores best first. */
  var NATURAL_DIR = {};
  NATURAL_DIR[SORT_STATE] = DIR_ASC;
  NATURAL_DIR[SORT_NAME] = DIR_ASC;
  NATURAL_DIR[SORT_SCORE] = DIR_DESC;

  var SEARCH_ID = "s6-runs-search";
  var STATE_ID = "s6-runs-state";
  var SOURCE_ID = "s6-runs-source";
  var TRAINED_ID = "s6-runs-trained";

  /* The four states a trained run can be in, from the process-state family of
     the registry spec, each with the role that carries its colour and its
     place in the attention-first order. The champion heads its group: it is
     the run the others are read beside. Then the two states waiting on a
     person, a run nobody has scored and a run whose training stopped, and
     last the settled rows. */
  var STATE_CHAMPION = "champion";
  var STATE_FAILED = "failed";
  var STATE_EMPTY = "empty";
  var STATE_DONE = "done";
  var STATES = {};
  STATES[STATE_CHAMPION] = { word: "champion", cls: "is-champion", order: 0 };
  STATES[STATE_EMPTY] = { word: "not scored", cls: "is-empty", order: 1 };
  STATES[STATE_FAILED] = { word: "did not finish", cls: "is-failed", order: 2 };
  STATES[STATE_DONE] = { word: "scored", cls: "is-done", order: 3 };
  var STATE_FILTER_ORDER = [STATE_EMPTY, STATE_FAILED, STATE_DONE, STATE_CHAMPION];

  var TEXT = {
    loading: "Loading runs",
    empty: "No trained models found, in this project or anywhere else.",
    noMatch: "No run matches these filters.",
    malformed: "The run list from the server was not in the expected shape.",
    noChampion: "No run has been promoted yet.",
    noScore: "no score",
    /* Two different absences. notEvaluated is a run the server found no
       score for at all; evaluateToPromote is a run whose score the server
       read from an evaluation folder because the rounds ledger has no row
       naming it. The second says what to do rather than what is missing,
       because it is the on-screen reason a dimmed Promote gives, and it says
       re-evaluate because the stamp printed beside it on the same line says
       when the run was last evaluated. */
    notEvaluated: "not scored yet",
    evaluateToPromote: "score it again to make it the best so far",
    sizeUnknown: "size unknown",
    evaluatedLabel: "evaluated",
    strictLabel: "mAP50-95",
    championPrefix: "Best so far: ",
    promotedLabel: "promoted",
    scopeModule: "the model Model Inference uses unless another is picked",
    scopeProject: "the score this project measures its later runs against",
    championMissing: "The chosen run's folder is no longer on disk, so VEGA cannot read the model it learned.",
    scoreSource: "VICARIUS read this {label} from the run's rounds ledger",
    scoreSourceEvaluation: "VICARIUS read this {label} from the run's newest evaluation folder, because no rounds-ledger row names this run.",
    labWord: "Trained outside a project",
    projectWord: "Project",
    mine: "this project",
    imported: "imported",
    headState: "State",
    headRun: "Run",
    headTrained: "Trained on",
    headScore: "Score",
    headActions: "Actions",
    notRecorded: "not recorded",
    groupScoredIn: ", scored on how well they find things: ",
    runsWord: "runs",
    runWord: "run",
    showing: "Showing ",
    ofWord: " of ",
    evaluate: "Evaluate",
    promote: "Promote",
    finetune: "Continue from",
    clearFilters: "Clear filters",
    labelSearch: "Find a run",
    labelState: "State",
    labelSource: "Source",
    labelTrained: "Trained on",
    searchPlaceholder: "part of a run name",
    optionAllStates: "Every state",
    optionAllSources: "Every source",
    optionAllTrained: "Every data set",
    optionMine: "This project",
    optionOther: "Other projects",
    optionLab: "Model labs",
    /* The confirm dialog belongs to the host page, which writes one fixed
       sentence for every row. The card fills the run's own facts into it
       right after the event is dispatched, so the only gate on a module-wide
       champion change names what is about to change. Decimals keep this
       string out of the tooltip set, whose sentence rules are checked. */
    promoteConfirm: "Make {name} the champion: {score}, {origin}. It becomes the default model for Model Inference and the first choice under Continue from a VEGA model.",
    promoteConfirmPlain: "Make {name} the champion: {origin}. It becomes the default model for Model Inference and the first choice under Continue from a VEGA model."
  };

  var TIPS = {
    evaluate: "Score this run on its held-out images. VICARIUS opens the Evaluate Model step with this run selected.",
    evaluateNoWeights: "VICARIUS cannot score this run because it has no weights file.",
    promote: "Make this run the champion. VICARIUS preselects it for Model Inference, offers it first under Continue from a VEGA model, and records the choice in the run's rounds ledger.",
    promoteNoScore: "Evaluate this run first. VICARIUS promotes only a run with a score in its rounds ledger.",
    promoteNotInLedger: "Evaluate this run again from the Evaluate Model step. VICARIUS promotes only a run whose score is in the rounds ledger beside it.",
    promoteNoWeights: "VICARIUS cannot promote this run because it has no weights file.",
    finetune: "Continue training from this run's weights. VICARIUS opens Train Model with this run chosen under Continue from a VEGA model.",
    finetuneNoWeights: "VICARIUS cannot continue from this run because it has no weights file.",
    finetuneCrossFamilyPinned: "This run trained a {run} and this project trains a {panel}, so its weights cannot start the next run. Pick a {panel} run instead.",
    finetuneAcrossFamily: "Continue training from this run's weights. VICARIUS sets Training task to {runTask} to match this run, then opens Train Model with this run chosen under Continue from a VEGA model.",
    champion: "The promoted run, called the champion: the model Model Inference uses unless another is picked, and the first choice under Continue from a VEGA model. Pressing Promote on a different run makes that one the champion instead.",
    stateFailed: "Training did not finish: this run has no weights file under its folder, so it was interrupted or its folder was moved. Train it again.",
    stateEmpty: "Nothing is recorded about how good this run is. Press Evaluate to score it on its held-out images.",
    stateDone: "VICARIUS scored this run on its held-out images and recorded its mAP50.",
    mine: "The open project trained this run, so its weights and its split are the ones beside the other steps here.",
    /* What a run's training data was, one sentence per kind: the noun says
       what sort of thing the name is, and the recorded sentence is for a name
       whose kind the record does not state. */
    trainedOnBoxDataset: "VEGA trained this run on the staged box dataset {name}.",
    trainedOnTrainingSet: "VEGA trained this run on the training set {name}.",
    trainedOnExport: "VEGA trained this run on the outlines the project {name} exported from its own images.",
    trainedOnRecorded: "VEGA trained this run on {name}.",
    trainedOnUnknown: "VEGA found no training data named in this run's own record.",
    /* One sentence each, with the folder last: a source path carrying a dot
       would otherwise read as a second sentence. */
    importedWhenAndWhere: "VEGA did not train this model. VICARIUS copied it in on {when} from {from}.",
    importedWhere: "VEGA did not train this model. VICARIUS copied it in from {from}.",
    importedUnknownSource: "VEGA did not train this run: VICARIUS copied it into the model lab from a folder its provenance file does not name.",
    search: "Type part of a run name to shorten the table to the runs whose name contains it. VICARIUS remembers what is typed here for the next visit.",
    stateFilter: "Choose one state to see only the runs in it. VICARIUS keeps every other run in the table when this is set to every state.",
    sourceFilter: "Choose where the runs came from: the open project, another project, or a model lab trained outside any project.",
    trainedFilter: "Choose one data set to see only the runs trained on it. VICARIUS lists every staged box dataset, training set and project export the runs in the table learned from.",
    clearFilters: "Put every filter back to its widest setting and show every run again.",
    sortState: "Sort the runs by state, so the promoted champion and then the runs needing a decision come first.",
    sortName: "Sort the runs by name, from A to Z.",
    sortScore: "Sort the runs by score, best first.",
    groupBox: "Runs that draw a box around each organism. VICARIUS scores them in box mAP50.",
    groupMask: "Runs that draw an outline around each organism. VICARIUS scores them in outline mAP50.",
    groupUnknown: "VICARIUS cannot read what these runs predict from their folders, so it names no metric for them."
  };

  /* Punctuation a screen reader hears and the eye never sees: the summary
     line's flex gap separates its counts on screen and nothing separates
     them in the live region's text. Absolutely positioned, so neither span
     takes part in the flex layout. */
  var SPOKEN_COMMA = '<span class="vega-runs-sr-only">, </span>';
  var SPOKEN_STOP = '<span class="vega-runs-sr-only">. </span>';

  var stateByRoot = new WeakMap();

  /**
   * HTML-escape a value for text or attribute positions.
   * @param {*} value anything; null and undefined become an empty string
   * @returns {string} the escaped text
   */
  function esc(value) {
    if (value === null || value === undefined) return "";
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  /**
   * Coerce a value to a string, with null and undefined as an empty string.
   * @param {*} value
   * @returns {string}
   */
  function str(value) {
    return value === null || value === undefined ? "" : String(value);
  }

  /**
   * A finite number or null. Strings are never numbers here: a score that
   * arrives as text is treated as missing rather than parsed.
   * @param {*} value
   * @returns {number|null}
   */
  function num(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }

  /**
   * A mean average precision in the closed range 0 to 1, or null: a value
   * outside that range is a ledger fault and is treated as missing.
   * @param {*} value
   * @returns {number|null}
   */
  function score(value) {
    var n = num(value);
    return n !== null && n >= 0 && n <= 1 ? n : null;
  }

  /**
   * A positive whole number (an image size in pixels), or null.
   * @param {*} value
   * @returns {number|null}
   */
  function positiveInt(value) {
    var n = num(value);
    return n !== null && n > 0 && Math.floor(n) === n ? n : null;
  }

  /**
   * Format a timestamp as Atlantic Standard Time, 24 hour:
   * "YYYY-MM-DD HH:MM AST". A stamp with an offset or Z is shifted to UTC
   * minus four; a naive stamp (no zone) is AST by house rule and its digits
   * are read as written, so the result never depends on the browser's zone.
   * @param {*} iso an ISO 8601 date-time, with or without a zone
   * @returns {string} the AST stamp, or "" when the input is not a stamp
   */
  function formatAst(iso) {
    if (typeof iso !== "string") return "";
    var m = STAMP_PATTERN.exec(iso.trim());
    if (!m) return "";
    var parts = [Number(m[1]), Number(m[2]), Number(m[3]), Number(m[4]), Number(m[5]), Number(m[6] || 0)];
    if (parts[1] < 1 || parts[1] > MAX_MONTH || parts[2] < 1 || parts[2] > MAX_DAY
        || parts[3] > MAX_HOUR || parts[4] > MAX_MINUTE || parts[5] > MAX_SECOND) return "";
    if (m[7]) {
      var offsetMinutes = 0;
      if (m[7] !== "Z") {
        var sign = m[7].charAt(0) === "-" ? -1 : 1;
        var digits = m[7].slice(1).replace(":", "");
        offsetMinutes = sign * (Number(digits.slice(0, 2)) * 60 + Number(digits.slice(2)));
      }
      var utcMs = Date.UTC(parts[0], parts[1] - 1, parts[2], parts[3], parts[4], parts[5]) - offsetMinutes * MS_PER_MINUTE;
      var shifted = new Date(utcMs + AST_OFFSET_MINUTES * MS_PER_MINUTE);
      parts = [shifted.getUTCFullYear(), shifted.getUTCMonth() + 1, shifted.getUTCDate(), shifted.getUTCHours(), shifted.getUTCMinutes()];
    }
    var pad = function (n) { return (n < 10 ? "0" : "") + n; };
    return parts[0] + "-" + pad(parts[1]) + "-" + pad(parts[2]) + " " + pad(parts[3]) + ":" + pad(parts[4]) + " AST";
  }

  /**
   * The plain word for what a task predicts, with its code kept beside it.
   * @param {*} task "seg" or "detect"; anything else reads as unknown task
   * @returns {{word: string, code: string}}
   */
  function taskLabel(task) {
    var code = str(task);
    return { word: TASK_WORDS[code] || UNKNOWN_TASK_WORD, code: code };
  }

  /**
   * The metric family of a run, from its own family field when the server
   * sent one, else from its task.
   * @param {*} family "mask" or "box"
   * @param {*} task "seg" or "detect"
   * @returns {string} "mask", "box", or "" when neither is readable
   */
  function familyOf(family, task) {
    return FAMILY_WORDS[str(family)] ? str(family) : (TASK_TO_FAMILY[str(task)] || "");
  }

  /**
   * The mAP50 label for a metric family: "outline mAP50" or "box mAP50",
   * falling back to the task when the family is blank, else bare "mAP50".
   * @param {*} family "mask" or "box"
   * @param {*} task "seg" or "detect", used when family is blank
   * @returns {string}
   */
  function familyLabel(family, task) {
    var fam = familyOf(family, task);
    return fam ? FAMILY_WORDS[fam] + " mAP50" : "mAP50";
  }

  /**
   * The mAP50-95 label for a metric family: "box mAP50-95" for a detector,
   * "outline mAP50-95" for a segmenter. The champion record's score is a
   * mAP50-95, so the summary line names it with this and never with the
   * mAP50 label the rows use, which is what made the champion read 0.826 in
   * the summary and 0.901 on its own row.
   * @param {*} family "mask" or "box"
   * @param {*} task "seg" or "detect", used when family is blank
   * @returns {string}
   */
  function strictFamilyLabel(family, task) {
    var fam = familyOf(family, task);
    return (fam ? FAMILY_WORDS[fam] + " " : "") + TEXT.strictLabel;
  }

  /**
   * Validate one raw run record from the server into the shape the
   * renderers rely on. Junk (null, a string, missing keys) yields an empty
   * run rather than an exception so one bad folder never hides the rest.
   * @param {*} raw one entry of the all_runs payload
   * @returns {object} the normalised run
   */
  function normaliseRun(raw) {
    var r = raw && typeof raw === "object" ? raw : {};
    var task = str(r.task);
    return {
      run_dir: str(r.run_dir),
      run_name: str(r.run_name),
      origin: r.origin === ORIGIN_LAB ? ORIGIN_LAB : ORIGIN_PROJECT,
      origin_kind: str(r.origin_kind) === ORIGIN_KIND_IMPORTED ? ORIGIN_KIND_IMPORTED : "",
      imported_from: str(r.imported_from),
      imported_at: str(r.imported_at),
      project_name: str(r.project_name),
      trained_on: str(r.trained_on),
      trained_on_kind: TRAINED_KINDS.indexOf(str(r.trained_on_kind)) >= 0 ? str(r.trained_on_kind) : "",
      task: task,
      family: familyOf(r.family, task),
      map50: score(r.map50),
      map50_95: score(r.map50_95),
      evaluated_at: str(r.evaluated_at),
      has_weights: Boolean(r.has_weights),
      champion: Boolean(r.champion),
      imgsz: positiveInt(r.imgsz),
      split_fingerprint: str(r.split_fingerprint),
      score_source: SCORE_SOURCES.indexOf(str(r.score_source)) >= 0 ? str(r.score_source) : "",
      /* Read and never drawn. The route sends both fields and keeps sending
         them; the card stopped showing anything about comparability on
         2026-09-16 and reads them only so the normalised run stays the
         payload's own shape for whatever reads it next. */
      comparability_reason: str(r.comparability_reason),
      comparable_to_champion: r.comparable_to_champion === true ? true
        : (r.comparable_to_champion === false ? false : null)
    };
  }

  /**
   * A training source in the shape trainedOn reads, from whatever a caller
   * passed: a name outside the three kinds keeps its name and loses its kind,
   * and anything that is not an object with a name is no source at all.
   * @param {*} raw a {name, kind} or anything else
   * @returns {{name: string, kind: string}|null}
   */
  function normaliseSource(raw) {
    if (!raw || typeof raw !== "object" || !str(raw.name)) return null;
    var kind = str(raw.kind);
    return { name: str(raw.name), kind: TRAINED_KINDS.indexOf(kind) >= 0 ? kind : "" };
  }

  /**
   * Validate the champion record, or null when there is none. The scope says
   * which of the module's two champion records this is and is carried
   * through unchanged: a payload that names no scope leaves it blank and the
   * summary then claims nothing about scope.
   * @param {*} raw the payload's champion field
   * @param {*} payloadScope the payload's own champion_scope, used when the
   *   record itself carries none
   * @returns {object|null}
   */
  function normaliseChampion(raw, payloadScope) {
    if (!raw || typeof raw !== "object" || !str(raw.run_dir)) return null;
    var scope = str(raw.scope) || str(payloadScope);
    return {
      run_dir: str(raw.run_dir),
      run_name: str(raw.run_name),
      promoted_at: str(raw.promoted_at),
      score: score(raw.score),
      family: FAMILY_WORDS[str(raw.family)] ? str(raw.family) : "",
      scope: (scope === SCOPE_MODULE || scope === SCOPE_PROJECT) ? scope : "",
      missing: raw.missing === true
    };
  }

  /**
   * Whether a normalised run is the champion: the server's own flag (which
   * compares real paths, so a symlinked run_dir still counts) or a run_dir
   * matching the champion record.
   * @param {object} run a normalised run
   * @param {object|null} champion the normalised champion
   * @returns {boolean}
   */
  function isChampion(run, champion) {
    if (run.champion) return true;
    return Boolean(champion) && run.run_dir !== "" && run.run_dir === champion.run_dir;
  }

  /**
   * Whether a run was trained by the project open in this window, decided on
   * the run folder sitting under the project folder. A blank project folder
   * (no project open, or the state route unavailable) marks nothing.
   * @param {object} run a normalised run
   * @param {string} projectDir the open project's folder
   * @returns {boolean}
   */
  function isOpenProjectRun(run, projectDir) {
    var dir = str(projectDir).replace(/\/+$/, "");
    if (!dir || !run.run_dir) return false;
    return run.run_dir === dir || run.run_dir.indexOf(dir + "/") === 0;
  }

  /**
   * The state a run is in, as one of the four keys of STATES. Precedence
   * runs from the fault a person must fix to the settled record: a run with
   * no weights failed whatever else is true of it; the promoted champion wins
   * over every other state; a run nobody has scored is empty; anything else
   * is done.
   * @param {object} run a normalised run
   * @param {object|null} champion the normalised champion
   * @returns {string} a STATES key
   */
  function runState(run, champion) {
    if (!run.has_weights) return STATE_FAILED;
    if (isChampion(run, champion)) return STATE_CHAMPION;
    if (run.map50 === null) return STATE_EMPTY;
    return STATE_DONE;
  }

  /**
   * The sentence a state chip carries.
   * @param {string} state a STATES key
   * @returns {string}
   */
  function stateTip(state) {
    if (state === STATE_CHAMPION) return TIPS.champion;
    if (state === STATE_FAILED) return TIPS.stateFailed;
    if (state === STATE_EMPTY) return TIPS.stateEmpty;
    return TIPS.stateDone;
  }

  /**
   * Which of the three sources a run came from, for the source filter and
   * the meta line.
   * @param {object} run a normalised run
   * @param {boolean} mine whether the open project trained it
   * @returns {string} SOURCE_MINE, SOURCE_OTHER or SOURCE_LAB
   */
  function sourceKind(run, mine) {
    if (run.origin === ORIGIN_LAB) return SOURCE_LAB;
    return mine ? SOURCE_MINE : SOURCE_OTHER;
  }

  /**
   * A score for the screen: three decimals, never NaN.
   * @param {number|null} value
   * @returns {string}
   */
  function formatScore(value) {
    return value === null ? "" : value.toFixed(SCORE_DECIMALS);
  }

  /**
   * "run" or "runs" for a count.
   * @param {number} n
   * @returns {string}
   */
  function runWord(n) {
    return n === 1 ? TEXT.runWord : TEXT.runsWord;
  }

  /**
   * Why an imported run's numbers are not VEGA's own work, as one sentence
   * for the imported chip's tooltip: where the run was copied from and when,
   * as far as its provenance file says.
   * @param {object} run a normalised run whose origin_kind is imported
   * @returns {string} the sentence
   */
  function importedTip(run) {
    if (!run.imported_from) return TIPS.importedUnknownSource;
    var when = formatAst(run.imported_at);
    if (!when) return TIPS.importedWhere.replace("{from}", run.imported_from);
    return TIPS.importedWhenAndWhere.replace("{when}", when).replace("{from}", run.imported_from);
  }

  /**
   * A focusable chip with its explanation.
   * @param {string} cls the modifier class (is-champion, is-failed, is-empty, is-done, is-imported, is-mine)
   * @param {string} text the chip's word
   * @param {string} tip the tooltip sentence
   * @returns {string} HTML
   */
  function chip(cls, text, tip) {
    return '<span class="vega-runs-chip ' + cls + '" tabindex="0" title="' + esc(tip) + '">' + esc(text) + "</span>";
  }

  /**
   * One row action button carrying the run identity in data attributes.
   *
   * Exactly one element owns the reason an action is unavailable. A disabled
   * button dispatches no pointer events, so the browser resolves a tooltip
   * from the first element up the tree that carries a title; measured on the
   * live page at 946 and 1280 px on 2026-09-15, the hit-tested element at the
   * centre of a disabled Promote is the button itself. A title left there
   * would therefore be the one resolved and the wrapper's would never be
   * read, so a disabled button carries none and its focusable wrapper carries
   * the reason instead. The cost is one keyboard stop per disabled action,
   * the same stop the state chips take, and it is what puts the reason within
   * reach of a screen reader, which cannot focus the disabled button at all.
   * @param {string} action evaluate, promote or finetune
   * @param {object} run a normalised run
   * @param {string} label the button text
   * @param {string} tip the tooltip sentence
   * @param {boolean} disabled whether the action is unavailable
   * @returns {string} HTML
   */
  function actionButton(action, run, label, tip, disabled) {
    var title = disabled ? "" : ' title="' + esc(tip) + '"';
    var button = '<button type="button" class="btn btn-outline btn-sm" data-action="' + action + '"'
      + ' data-run-dir="' + esc(run.run_dir) + '" data-run-name="' + esc(run.run_name) + '" data-task="' + esc(run.task) + '"'
      + title + (disabled ? " disabled" : "") + ">" + esc(label) + "</button>";
    if (!disabled) return button;
    return '<span class="vega-runs-action" tabindex="0" title="' + esc(tip) + '">' + button + "</span>";
  }

  /**
   * A run name in one or two spans that never break a token. A name ending
   * in a run stamp splits at that boundary, so the stamp always shows: it
   * moves to a second line when the two do not fit the Run column, and the
   * head shortens at its end only when it alone is wider than the column.
   * Any other name is one span that shortens at its end and only when it
   * does not fit. Both spans carry the row's ink.
   * @param {string} name the run name
   * @returns {string} HTML
   */
  function renderNameParts(name) {
    var m = NAME_STAMP_PATTERN.exec(name);
    if (!m) return '<span class="vega-runs-name-head">' + esc(name) + "</span>";
    return '<span class="vega-runs-name-head">' + esc(m[1]) + "</span>"
      + '<span class="vega-runs-name-tail">' + esc(m[2]) + "</span>";
  }

  /**
   * The State cell: one chip naming the run's state, with what the state
   * means in its tooltip.
   * @param {string} state a STATES key
   * @returns {string} HTML
   */
  function renderStateCell(state) {
    var spec = STATES[state];
    return '<div class="vega-runs-line1">' + chip(spec.cls, spec.word, stateTip(state)) + "</div>";
  }

  /**
   * The data subset a run trained on, as the run's own record names it.
   *
   * The route's own trained_on field wins wherever it sends one, with its
   * kind. Otherwise a run the open project trained takes the project's own
   * source, which the state route names (training_source_name): a training
   * project trains from that one source, so every run under it learned from
   * it. Nothing else is a source of the name: the lab or project folder that
   * holds a run is not a data set, and the cell says "not recorded" rather
   * than naming one.
   * @param {object} run a normalised run
   * @param {{name: string, kind: string}|null} projectSource the open
   *   project's own source from projectSourceOf, or null when none is known
   * @param {boolean} mine whether the open project trained this run
   * @returns {{name: string, kind: string}} the data set's name and kind, both
   *   "" when the record names none
   */
  function trainedOn(run, projectSource, mine) {
    if (run.trained_on) return { name: run.trained_on, kind: run.trained_on_kind };
    if (mine && projectSource && projectSource.name) return { name: projectSource.name, kind: projectSource.kind };
    return { name: "", kind: "" };
  }

  /**
   * The open project's own training source from a /api/project/state answer:
   * the name of the staged box dataset or training set it was minted to train
   * from, and its kind read from the route's own label words. An ordinary
   * project, a closed project or an answer not in that shape name none.
   * @param {*} info the parsed answer
   * @returns {{name: string, kind: string}}
   */
  function projectSourceOf(info) {
    if (!info || typeof info !== "object" || info.loaded !== true) return { name: "", kind: "" };
    var state = info.state && typeof info.state === "object" ? info.state : {};
    var name = str(state.training_source_name);
    if (!name) return { name: "", kind: "" };
    return { name: name, kind: STATE_SOURCE_LABEL_KIND[str(state.training_source_label)] || "" };
  }

  /**
   * The filter value for one training data set, which is never the bare name:
   * a data set called "all" would otherwise read as every data set.
   * @param {string} name the data set's name, or "" when none is recorded
   * @returns {string} the option value
   */
  function trainedValue(name) {
    return name ? TRAINED_PREFIX + name : TRAINED_NONE;
  }

  /**
   * The name a filter value stands for, for the option's own text.
   * @param {string} value an option value from trainedValue
   * @returns {string} the data set's name, or the words for no record
   */
  function trainedLabel(value) {
    return value === TRAINED_NONE ? TEXT.notRecorded : str(value).slice(TRAINED_PREFIX.length);
  }

  /**
   * The order of the Trained on filter's options: data set names A to Z,
   * and the runs whose record names none last, whatever else is offered.
   * @param {string} a an option value from trainedValue
   * @param {string} b another
   * @returns {number} negative when a comes first
   */
  function compareTrained(a, b) {
    if (a === TRAINED_NONE) return b === TRAINED_NONE ? 0 : 1;
    if (b === TRAINED_NONE) return -1;
    return trainedLabel(a).localeCompare(trainedLabel(b));
  }

  /**
   * The sentence the Trained on cell carries: what the run trained on, with
   * the noun for its kind.
   * @param {{name: string, kind: string}} where the answer from trainedOn
   * @returns {string}
   */
  function trainedTip(where) {
    if (!where.name) return TIPS.trainedOnUnknown;
    if (where.kind === TRAINED_KIND_BBOX) return TIPS.trainedOnBoxDataset.replace("{name}", where.name);
    if (where.kind === TRAINED_KIND_COMPOSITE) return TIPS.trainedOnTrainingSet.replace("{name}", where.name);
    if (where.kind === TRAINED_KIND_EXPORT) return TIPS.trainedOnExport.replace("{name}", where.name);
    return TIPS.trainedOnRecorded.replace("{name}", where.name);
  }

  /**
   * The Trained on cell: the data set's name whole, with the sentence naming
   * it and its kind in the tooltip. The name shortens at its end only when it
   * really is wider than the column, and the tooltip then still carries all
   * of it.
   * @param {{name: string, kind: string}} where the answer from trainedOn
   * @returns {string} HTML
   */
  function renderTrainedCell(where) {
    var tip = ' tabindex="0" title="' + esc(trainedTip(where)) + '"';
    if (!where.name) {
      return '<div class="vega-runs-line1"><span class="vega-runs-trained is-none"' + tip + ">"
        + esc(TEXT.notRecorded) + "</span></div>";
    }
    return '<div class="vega-runs-line1"><span class="vega-runs-trained"' + tip + ">"
      + esc(where.name) + "</span></div>";
  }

  /**
   * The Run cell: the run name alone, so the column's whole width is the
   * name's. The full name and the run folder ride in its tooltip, so a
   * shortened name is always readable in full; which project or model lab
   * the run came from is the Trained on column beside it.
   * @param {object} run a normalised run
   * @returns {string} HTML
   */
  function renderNameCell(run) {
    var name = run.run_name || UNNAMED_RUN;
    return '<div class="vega-runs-line1"><span class="vega-runs-name" title="' + esc(name + " in " + run.run_dir) + '">'
      + renderNameParts(name) + "</span></div>";
  }

  /**
   * What the meta line says about this run's evaluation, in reading order:
   * when it was evaluated, and what a reader must do to promote it. A run
   * whose score the server read from an evaluation folder says "re-evaluate
   * to enable Promote", because the missing ledger row is what Promote waits
   * for and the run has an evaluation on disk already, which the stamp on
   * the same line names; a run with no score at all says "not evaluated
   * yet"; a run with a ledger row says neither. The champion is never told
   * to enable a Promote button its own row does not offer.
   * @param {object} run a normalised run
   * @param {boolean} champ whether this record is the champion
   * @returns {Array<string>} nought to two facts, each its own span
   */
  function evaluationFacts(run, champ) {
    var stamp = formatAst(run.evaluated_at);
    var facts = [];
    if (stamp) facts.push(TEXT.evaluatedLabel + " " + stamp);
    if (run.score_source === SCORE_SOURCE_EVALUATION) {
      if (!champ) facts.push(TEXT.evaluateToPromote);
    } else if (!stamp) {
      facts.push(TEXT.notEvaluated);
    }
    return facts;
  }

  /**
   * The meta line under a record, spanning every column: the markers a run
   * carries, the training image size, its strict score, and when it was
   * evaluated. What the run predicts lives in the group heading above it, and
   * what it trained on is its own column, so neither is repeated here. Facts
   * are separated by the line's own column gap and never by a border: a
   * bordered fact that wraps begins the next line with a bar standing on its
   * own.
   * @param {object} run a normalised run
   * @param {boolean} champ whether this record is the champion
   * @param {boolean} mine whether the open project trained this run
   * @returns {string} HTML
   */
  function renderMetaLine(run, champ, mine) {
    var size = run.imgsz === null ? TEXT.sizeUnknown : run.imgsz + " px";
    var facts = [];
    if (run.map50_95 !== null) facts.push(TEXT.strictLabel + " " + formatScore(run.map50_95));
    /* The numbers first and the instruction last, so the line that wraps on
       the busiest records is the one asking for an action and never a fact. */
    facts = facts.concat(evaluationFacts(run, champ));
    /* A run the open project trained is marked, so a student can find their
       own work without reading the Trained on column. */
    var markers = mine ? chip("is-mine", TEXT.mine, TIPS.mine) : "";
    if (run.origin_kind === ORIGIN_KIND_IMPORTED) markers += chip("is-imported", TEXT.imported, importedTip(run));
    return '<div class="vega-runs-meta-line">'
      + markers
      + '<span class="vega-runs-fact">' + esc(size) + "</span>"
      + facts.map(function (fact) {
        return '<span class="vega-runs-fact">' + esc(fact) + "</span>";
      }).join("")
      + "</div>";
  }

  /**
   * The Score cell: the mAP50 value alone. The metric family is named once in
   * the group heading above, so it is not repeated on every row; where the
   * number came from rides in the value's tooltip.
   * @param {object} run a normalised run
   * @returns {string} HTML
   */
  function renderScoreCell(run) {
    if (run.map50 === null) {
      return '<div class="vega-runs-line1"><span class="vega-runs-score is-none">' + TEXT.noScore + "</span></div>";
    }
    var stamp = formatAst(run.evaluated_at);
    var label = familyLabel(run.family, run.task);
    /* A score read from an evaluation folder says so instead of naming the
       ledger, and carries no stamp of its own: the meta line beside it
       already shows when the run was evaluated. */
    var source = run.score_source === SCORE_SOURCE_EVALUATION
      ? TEXT.scoreSourceEvaluation.replace("{label}", label)
      : TEXT.scoreSource.replace("{label}", label) + (stamp ? ", " + TEXT.evaluatedLabel + " " + stamp : "") + ".";
    return '<div class="vega-runs-line1">'
      + '<span class="vega-runs-score" tabindex="0" title="' + esc(source) + '">' + formatScore(run.map50) + "</span>"
      + "</div>";
  }

  /**
   * Whether this run trains the other metric family from the Train panel's.
   * A run whose family VEGA cannot read is never a mismatch: an unknown
   * family is not evidence of one.
   * @param {object} run a normalised run
   * @param {string} panelTask the Train panel's task, "segment" or "detect"
   * @returns {boolean}
   */
  function crossFamily(run, panelTask) {
    var wanted = PANEL_TASK_FAMILY[panelTask];
    return Boolean(wanted && run.family && run.family !== wanted);
  }

  /**
   * Why this run's weights cannot start the next training run, or "" when
   * they can. Only a project that pins its Training task refuses: a box
   * dataset trains a detector and nothing else, so a segmenter's weights
   * have nowhere to go. Where the task is free the panel follows the run
   * instead of refusing it, which is the hand-off the host page performs
   * (onVegaFinetuneFrom in orchestrator.js switches the control, then
   * selects the run under Continue from a VEGA model), so this returns ""
   * and the button stays live.
   * @param {object} run a normalised run
   * @param {string} panelTask the Train panel's task, "segment" or "detect"
   * @param {boolean} pinned whether the panel's task cannot be changed
   * @returns {string} the sentence, or ""
   */
  function finetuneRefusal(run, panelTask, pinned) {
    if (!pinned || !crossFamily(run, panelTask)) return "";
    return TIPS.finetuneCrossFamilyPinned
      .replace("{run}", FAMILY_MODEL_NOUN[run.family])
      .replace(/\{panel\}/g, FAMILY_MODEL_NOUN[PANEL_TASK_FAMILY[panelTask]]);
  }

  /**
   * What Continue from says on one row: the refusal when the panel's task is
   * pinned to the other family, the hand-off sentence when the task is free
   * and the families differ (VICARIUS moves the control itself), and the
   * plain offer otherwise.
   * @param {object} run a normalised run
   * @param {string} panelTask the Train panel's task, "segment" or "detect"
   * @param {boolean} pinned whether the panel's task cannot be changed
   * @returns {{tip: string, disabled: boolean}} the sentence and whether the
   *   button is unavailable
   */
  function finetuneOffer(run, panelTask, pinned) {
    if (!run.has_weights) return { tip: TIPS.finetuneNoWeights, disabled: true };
    var refusal = finetuneRefusal(run, panelTask, pinned);
    if (refusal) return { tip: refusal, disabled: true };
    if (crossFamily(run, panelTask)) {
      return {
        tip: TIPS.finetuneAcrossFamily.replace("{runTask}", FAMILY_TASK_OPTION[run.family]),
        disabled: false
      };
    }
    return { tip: TIPS.finetune, disabled: false };
  }

  /**
   * The Actions cell: Evaluate, Promote (absent on the champion row) and
   * Continue from, each disabled with its own reason when it cannot work.
   * @param {object} run a normalised run
   * @param {boolean} champ whether this row is the champion
   * @param {object} opts {panelTask, taskPinned}
   * @returns {string} HTML
   */
  function renderActionsCell(run, champ, opts) {
    var weights = run.has_weights;
    var html = '<div class="vega-runs-actions">'
      + actionButton("evaluate", run, TEXT.evaluate, weights ? TIPS.evaluate : TIPS.evaluateNoWeights, !weights);
    if (!champ) {
      /* Promotion is a ledger act, so a score the server read from an
         evaluation folder is shown but never promotable until the run is
         evaluated from the Evaluate Model step and earns its ledger row. */
      var fromFolder = run.score_source === SCORE_SOURCE_EVALUATION;
      var tip = TIPS.promote;
      if (!weights) tip = TIPS.promoteNoWeights;
      else if (run.map50 === null) tip = TIPS.promoteNoScore;
      else if (fromFolder) tip = TIPS.promoteNotInLedger;
      html += actionButton("promote", run, TEXT.promote, tip, !weights || run.map50 === null || fromFolder);
    }
    var offer = finetuneOffer(run, str(opts && opts.panelTask), Boolean(opts && opts.taskPinned));
    html += actionButton("finetune", run, TEXT.finetune, offer.tip, offer.disabled);
    return html + "</div>";
  }

  /**
   * One record for a run: a tbody holding the main row (state, name, score,
   * actions) and a meta row spanning every column. Every table part declares
   * its ARIA role, because the narrow card lays the record out as blocks and
   * a display change otherwise strips the table semantics. The striping class
   * is written here rather than left to nth-child, because the group headings
   * between records would otherwise shift every other stripe.
   * @param {*} rawRun one entry of the payload, or an already normalised run
   * @param {object|null} champion the normalised champion
   * @param {object} [opts] {projectDir, projectSource, panelTask, taskPinned, alt}
   * @returns {string} HTML for a tbody element
   */
  function renderRow(rawRun, champion, opts) {
    var options = opts || {};
    var run = normaliseRun(rawRun);
    var champ = isChampion(run, champion);
    var mine = isOpenProjectRun(run, options.projectDir);
    var where = trainedOn(run, options.projectSource || null, mine);
    var state = runState(run, champion);
    var classes = "vega-runs-record" + (champ ? " is-champion-record" : "")
      + (mine ? " is-mine-record" : "") + (options.alt ? " is-alt" : "");
    return '<tbody class="' + classes + '" role="rowgroup" data-state="' + esc(state) + '">'
      + '<tr class="vega-runs-main" role="row">'
      + '<td role="cell">' + renderStateCell(state) + "</td>"
      + '<td role="cell">' + renderNameCell(run) + "</td>"
      + '<td role="cell">' + renderTrainedCell(where) + "</td>"
      + '<td role="cell">' + renderScoreCell(run) + "</td>"
      + '<td role="cell">' + renderActionsCell(run, champ, options) + "</td>"
      + "</tr>"
      + '<tr class="vega-runs-meta" role="row"><td role="cell" colspan="' + COLUMN_COUNT + '">'
      + renderMetaLine(run, champ, mine) + "</td></tr>"
      + "</tbody>";
  }

  /**
   * Every record of one family group, striped so the eye can track a row
   * across its columns.
   * @param {Array<object>} runs normalised runs, already ordered
   * @param {object|null} champion the normalised champion
   * @param {object} opts {projectDir, projectSource, panelTask, taskPinned}
   * @returns {string} HTML
   */
  function renderRows(runs, champion, opts) {
    return runs.map(function (r, i) {
      var options = Object.assign({}, opts || {}, { alt: i % 2 === 1 });
      return renderRow(r, champion, options);
    }).join("");
  }

  /**
   * The heading row that opens a family group, naming what its runs predict,
   * the metric they are scored in and how many there are.
   * @param {string} family "box", "mask" or ""
   * @param {number} shown how many runs of this family the filters kept
   * @returns {string} HTML for a tbody element
   */
  function renderGroupHead(family, shown) {
    var name = FAMILY_GROUP_NAME[family] || UNGROUPED_NAME;
    var tip = family === "box" ? TIPS.groupBox : (family === "mask" ? TIPS.groupMask : TIPS.groupUnknown);
    var metric = family ? TEXT.groupScoredIn + familyLabel(family, "") : "";
    return '<tbody class="vega-runs-group" role="rowgroup"><tr role="row">'
      + '<th class="vega-runs-group-cell" colspan="' + COLUMN_COUNT + '" scope="colgroup" role="columnheader">'
      + '<span class="vega-runs-group-name" tabindex="0" title="' + esc(tip) + '">' + esc(name + metric) + "</span>"
      + '<span class="vega-runs-group-count">' + shown + " " + runWord(shown) + "</span>"
      + "</th></tr></tbody>";
  }

  /**
   * One sortable column heading.
   * @param {string} key the sort key, or "" for a column that cannot sort
   * @param {string} label the heading text
   * @param {string} tip the tooltip sentence, or ""
   * @param {object} sort the current {key, dir}
   * @param {string} cls an extra class for the heading cell, or ""
   * @returns {string} HTML
   */
  function renderHead(key, label, tip, sort, cls) {
    var active = sort.key === key;
    var head = '<th scope="col" role="columnheader"' + (cls ? ' class="' + cls + '"' : "")
      + (key ? ' aria-sort="' + (active ? (sort.dir === DIR_ASC ? "ascending" : "descending") : "none") + '"' : "") + ">";
    if (!key) return head + esc(label) + "</th>";
    return head + '<button type="button" class="vega-runs-th' + (active ? " is-sorted" : "") + '"'
      + ' data-sort="' + esc(key) + '" data-dir="' + esc(active ? sort.dir : "") + '"'
      + ' title="' + esc(tip) + '">' + esc(label) + '<span class="vega-runs-arrow" aria-hidden="true"></span></button></th>';
  }

  /**
   * The table: one heading row, then one heading plus its records per family
   * group. Groups with no run left after filtering are left out entirely.
   * @param {Array<object>} groups [{family, runs}] in display order
   * @param {object|null} champion the normalised champion
   * @param {object} opts {projectDir, projectSource, panelTask, taskPinned, sort}
   * @returns {string} HTML
   */
  function renderTable(groups, champion, opts) {
    var sort = opts.sort;
    var body = groups.map(function (g) {
      return renderGroupHead(g.family, g.runs.length) + renderRows(g.runs, champion, opts);
    }).join("");
    return '<div class="vega-runs-wrap"><table class="vega-runs-table" aria-labelledby="' + TITLE_ID + '" role="table">'
      + '<colgroup><col class="vega-runs-col-state"><col class="vega-runs-col-run"><col class="vega-runs-col-trained">'
      + '<col class="vega-runs-col-score"><col class="vega-runs-col-actions"></colgroup>'
      + '<thead role="rowgroup"><tr role="row">'
      + renderHead(SORT_STATE, TEXT.headState, TIPS.sortState, sort, "")
      + renderHead(SORT_NAME, TEXT.headRun, TIPS.sortName, sort, "")
      + renderHead("", TEXT.headTrained, "", sort, "")
      + renderHead(SORT_SCORE, TEXT.headScore, TIPS.sortScore, sort, "")
      + renderHead("", TEXT.headActions, "", sort, "")
      + "</tr></thead>" + body + "</table></div>";
  }

  /**
   * Whether a run passes the current filters. Every filter narrows the table
   * further: a run shows only when it passes all four.
   * @param {object} row {run, state, mine, source, trained}
   * @param {object} view {search, state, source, trained}
   * @returns {boolean}
   */
  function matchesFilters(row, view) {
    var q = str(view.search).trim().toLowerCase();
    if (q && row.run.run_name.toLowerCase().indexOf(q) < 0) return false;
    if (view.state !== FILTER_ALL && row.state !== view.state) return false;
    if (view.source !== FILTER_ALL && row.source !== view.source) return false;
    if (view.trained !== FILTER_ALL && row.trained !== view.trained) return false;
    return true;
  }

  /**
   * Order the rows of one group by the chosen column. Ties fall back to the
   * score, best first, and then to the name, so the order never depends on
   * the order the server happened to read the folders in.
   * @param {Array<object>} rows [{run, state, ...}]
   * @param {object} sort {key, dir}
   * @returns {Array<object>} a new, ordered array
   */
  function sortRows(rows, sort) {
    var byScore = function (a, b) {
      var x = a.run.map50 === null ? -1 : a.run.map50;
      var y = b.run.map50 === null ? -1 : b.run.map50;
      return y - x;
    };
    return rows.slice().sort(function (a, b) {
      var first = 0;
      if (sort.key === SORT_NAME) {
        first = a.run.run_name.localeCompare(b.run.run_name) * (sort.dir === DIR_DESC ? -1 : 1);
      } else if (sort.key === SORT_SCORE) {
        /* byScore already runs best first, which is the descending
           direction; ascending reverses it. A run with no score counts as
           the lowest score of all, so it closes the descending order and
           opens the ascending one. */
        first = byScore(a, b) * (sort.dir === DIR_ASC ? -1 : 1);
      } else {
        first = (STATES[a.state].order - STATES[b.state].order) * (sort.dir === DIR_DESC ? -1 : 1);
      }
      if (first !== 0) return first;
      var tie = byScore(a, b);
      return tie !== 0 ? tie : a.run.run_name.localeCompare(b.run.run_name);
    });
  }

  /**
   * Every run of the payload with the facts the board needs, before any
   * filter is applied.
   * @param {Array} runs the payload's runs
   * @param {object|null} champion the normalised champion
   * @param {string} projectDir the open project's folder
   * @param {{name: string, kind: string}|null} projectSource the open
   *   project's own training source, or null when none is known
   * @returns {Array<object>} [{run, state, mine, source, trained}]
   */
  function describeRuns(runs, champion, projectDir, projectSource) {
    return runs.map(function (raw) {
      var run = normaliseRun(raw);
      var mine = isOpenProjectRun(run, projectDir);
      return {
        run: run,
        state: runState(run, champion),
        mine: mine,
        source: sourceKind(run, mine),
        trained: trainedValue(trainedOn(run, projectSource || null, mine).name)
      };
    });
  }

  /**
   * The family groups in display order: the family the Train panel is set to
   * first, then the champion's, then the rest, so a student's own family
   * opens the table.
   * @param {Array<object>} rows described runs that passed the filters
   * @param {object|null} champion the normalised champion
   * @param {string} panelTask the Train panel's task
   * @param {object} sort {key, dir}
   * @returns {Array<object>} [{family, runs}] with no empty group
   */
  function groupByFamily(rows, champion, panelTask, sort) {
    var order = [];
    var wanted = PANEL_TASK_FAMILY[str(panelTask)];
    if (wanted) order.push(wanted);
    if (champion && champion.family && order.indexOf(champion.family) < 0) order.push(champion.family);
    FAMILY_ORDER.forEach(function (f) { if (order.indexOf(f) < 0) order.push(f); });
    return order.map(function (family) {
      var mine = rows.filter(function (r) { return r.run.family === family; });
      return { family: family, runs: sortRows(mine, sort).map(function (r) { return r.run; }) };
    }).filter(function (g) { return g.runs.length > 0; });
  }

  /**
   * The counts the summary line states, the runs waiting on a person first.
   * @param {Array<object>} rows every described run, unfiltered
   * @returns {object} {total, failed, empty, labs, projects}
   */
  function countRuns(rows) {
    var counted = { total: rows.length, failed: 0, empty: 0, labs: 0, projects: 0 };
    var labNames = {};
    var projectNames = {};
    rows.forEach(function (r) {
      if (r.state === STATE_FAILED) counted.failed += 1;
      if (r.state === STATE_EMPTY) counted.empty += 1;
      if (r.run.origin === ORIGIN_LAB) labNames[r.run.project_name] = 1;
      else projectNames[r.run.project_name] = 1;
    });
    counted.labs = Object.keys(labNames).length;
    counted.projects = Object.keys(projectNames).length;
    return counted;
  }

  /**
   * One count phrase of the summary line, with the number in the row ink.
   * @param {number} n the count
   * @param {string} words what the count is of
   * @param {string} cls an extra class, or ""
   * @returns {string} HTML
   */
  function countPhrase(n, words, cls) {
    return '<span class="vega-runs-count' + (cls ? " " + cls : "") + '"><b>' + n + "</b> " + esc(words) + "</span>";
  }

  /**
   * What the champion clause says, including which of the module's two
   * champion records this is. The score is a mAP50-95 and is labelled as
   * one, never as the mAP50 the rows show.
   * @param {object|null} champ the normalised champion
   * @returns {string} HTML
   */
  function championClause(champ) {
    if (!champ) return '<span class="vega-runs-champion">' + esc(TEXT.noChampion) + "</span>";
    var stamp = formatAst(champ.promoted_at);
    var showScore = champ.score !== null && !champ.missing;
    var scope = "";
    if (champ.scope === SCOPE_MODULE) scope = ", " + TEXT.scopeModule;
    else if (champ.scope === SCOPE_PROJECT) scope = ", " + TEXT.scopeProject;
    return '<span class="vega-runs-champion">' + esc(TEXT.championPrefix)
      + "<b>" + esc(champ.run_name || UNNAMED_RUN) + "</b>"
      + (showScore ? ', <span class="vega-runs-nowrap">' + esc(strictFamilyLabel(champ.family, "")) + " <b>" + formatScore(champ.score) + "</b></span>" : "")
      + (stamp ? ", " + esc(TEXT.promotedLabel) + " " + esc(stamp) : "")
      + esc(scope) + "."
      + (champ.missing ? " " + esc(TEXT.championMissing) : "") + "</span>";
  }

  /**
   * The summary line: the runs waiting on a person first (the unscored ones,
   * then the ones whose training stopped), then what the table holds, then
   * the champion. When a filter is on, it also says how many runs are being
   * shown of how many.
   *
   * The counts are separated on screen by the line's own column gap, which a
   * screen reader cannot hear: read out of the live region the phrases ran
   * together as "3 not scored5 did not finish". So each join carries a
   * punctuation mark in a span that is clipped out of sight, a comma between
   * the counts and a full stop before the champion sentence.
   * @param {object} counted the counts from countRuns
   * @param {number} shown how many runs the filters kept
   * @param {object|null} champ the normalised champion
   * @returns {string} HTML
   */
  function renderSummaryLine(counted, shown, champ) {
    var parts = [];
    if (counted.empty) parts.push(countPhrase(counted.empty, "not scored", ""));
    if (counted.failed) parts.push(countPhrase(counted.failed, "did not finish", "is-failed"));
    parts.push(countPhrase(counted.total, runWord(counted.total), ""));
    parts.push(countPhrase(counted.labs, counted.labs === 1 ? "model lab" : "model labs", ""));
    parts.push(countPhrase(counted.projects, counted.projects === 1 ? "project" : "projects", ""));
    if (shown !== counted.total) {
      parts.push('<span class="vega-runs-count">' + esc(TEXT.showing) + "<b>" + shown + "</b>"
        + esc(TEXT.ofWord) + "<b>" + counted.total + "</b></span>");
    }
    return '<div class="vega-runs-summary" aria-live="polite" data-role="summary">'
      + parts.join(SPOKEN_COMMA) + SPOKEN_STOP + championClause(champ) + "</div>";
  }

  /**
   * One filter row: a visible label bound to its control, a "?" glyph
   * carrying the explanation, and the control.
   * @param {string} id the control's id
   * @param {string} label the visible label
   * @param {string} tip the glyph's sentence
   * @param {string} control the control's HTML
   * @returns {string} HTML
   */
  function renderFilterRow(id, label, tip, control) {
    return '<div class="vega-runs-filter">'
      + '<div class="vega-runs-filter-head"><label class="vega-runs-label" for="' + esc(id) + '">' + esc(label) + "</label>"
      + '<button type="button" class="vega-runs-tip" aria-label="About ' + esc(label.toLowerCase()) + '" title="' + esc(tip) + '"><span aria-hidden="true">?</span></button>'
      + "</div>" + control + "</div>";
  }

  /**
   * One option of a filter select.
   * @param {string} value the option's value
   * @param {string} label the option's text
   * @param {string} current the filter's current value
   * @returns {string} HTML
   */
  function option(value, label, current) {
    return '<option value="' + esc(value) + '"' + (value === current ? " selected" : "") + ">" + esc(label) + "</option>";
  }

  /**
   * Every option of the Trained on filter: the data sets the runs came from,
   * in name order, with the runs whose record names none last. A remembered
   * choice the payload no longer holds is offered too, so the control shows
   * what is really filtering the table rather than reading as "every data
   * set" over an empty board.
   * @param {string} current the filter's current value
   * @param {Array<string>} values the values the payload holds
   * @returns {string} HTML
   */
  function trainedOptions(current, values) {
    var offered = (values || []).slice();
    if (current !== FILTER_ALL && offered.indexOf(current) < 0) offered.push(current);
    offered.sort(compareTrained);
    return offered.map(function (value) {
      return option(value, trainedLabel(value), current);
    }).join("");
  }

  /**
   * The filter block: search, state, source, training data, and a Clear
   * filters button that shows only while a filter is narrowing the table.
   *
   * Each explanation has one owner, the "?" glyph beside the field's visible
   * label (design section 7). The controls themselves carry no title: with
   * one on the control as well, the same sentence opened twice from two
   * hit areas a few pixels apart.
   * @param {object} view {search, state, source, trained}
   * @param {object} present what the payload holds, {mine, other, lab, trained}
   * @returns {string} HTML
   */
  function renderFilters(view, present) {
    var search = '<input type="search" class="vega-runs-input" id="' + SEARCH_ID + '" data-filter="search"'
      + ' value="' + esc(view.search) + '" placeholder="' + esc(TEXT.searchPlaceholder) + '">';
    var states = STATE_FILTER_ORDER.map(function (key) {
      return option(key, STATES[key].word, view.state);
    }).join("");
    var stateSelect = '<select class="vega-runs-select" id="' + STATE_ID + '" data-filter="state">'
      + option(FILTER_ALL, TEXT.optionAllStates, view.state) + states + "</select>";
    var sources = option(FILTER_ALL, TEXT.optionAllSources, view.source)
      + (present.mine ? option(SOURCE_MINE, TEXT.optionMine, view.source) : "")
      + (present.other ? option(SOURCE_OTHER, TEXT.optionOther, view.source) : "")
      + (present.lab ? option(SOURCE_LAB, TEXT.optionLab, view.source) : "");
    var sourceSelect = '<select class="vega-runs-select" id="' + SOURCE_ID + '" data-filter="source">'
      + sources + "</select>";
    var trainedSelect = '<select class="vega-runs-select" id="' + TRAINED_ID + '" data-filter="trained">'
      + option(FILTER_ALL, TEXT.optionAllTrained, view.trained)
      + trainedOptions(view.trained, present.trained) + "</select>";
    var active = filtersActive(view);
    return '<div class="vega-runs-filters">'
      + renderFilterRow(SEARCH_ID, TEXT.labelSearch, TIPS.search, search)
      + renderFilterRow(STATE_ID, TEXT.labelState, TIPS.stateFilter, stateSelect)
      + renderFilterRow(SOURCE_ID, TEXT.labelSource, TIPS.sourceFilter, sourceSelect)
      + renderFilterRow(TRAINED_ID, TEXT.labelTrained, TIPS.trainedFilter, trainedSelect)
      + '<button type="button" class="btn btn-outline btn-sm vega-runs-clear" data-action="' + ACTION_CLEAR + '"'
      + (active ? "" : " hidden") + ' title="' + esc(TIPS.clearFilters) + '">' + esc(TEXT.clearFilters) + "</button>"
      + "</div>";
  }

  /**
   * Whether any filter is narrowing the table, which is when Clear filters is
   * on screen. One owner, because the filter block and the partial redraw both
   * ask it and a fifth filter added to one and not the other would leave the
   * button behind.
   * @param {object} view the current view
   * @returns {boolean}
   */
  function filtersActive(view) {
    return Boolean(str(view.search).trim()) || view.state !== FILTER_ALL
      || view.source !== FILTER_ALL || view.trained !== FILTER_ALL;
  }

  /**
   * The empty state: one sentence.
   * @returns {string} HTML
   */
  function renderEmpty() {
    return '<div class="vega-runs-line vega-runs-dim">' + esc(TEXT.empty) + "</div>";
  }

  /**
   * The loading state: one line.
   * @returns {string} HTML
   */
  function renderLoading() {
    return '<div class="vega-runs-line vega-runs-dim">' + esc(TEXT.loading) + "</div>";
  }

  /**
   * The error state: the server's text, escaped.
   * @param {*} message the error text
   * @returns {string} HTML
   */
  function renderError(message) {
    return '<div class="vega-runs-line vega-runs-error">' + esc(message) + "</div>";
  }

  /**
   * The line shown in place of the table when the filters match no run.
   * @returns {string} HTML
   */
  function renderNoMatch() {
    return '<div class="vega-runs-line vega-runs-dim">' + esc(TEXT.noMatch) + "</div>";
  }

  /**
   * One view's defaults, so a missing or damaged remembered value never
   * leaves the board unsorted or half filtered.
   * @returns {object} {search, state, source, trained, sort: {key, dir}}
   */
  function defaultView() {
    return {
      search: "", state: FILTER_ALL, source: FILTER_ALL, trained: FILTER_ALL,
      sort: { key: SORT_STATE, dir: DIR_ASC }
    };
  }

  /**
   * A view object with every field validated: an unknown state, source or
   * sort key falls back to the default rather than hiding every row.
   * @param {*} raw anything shaped like a view
   * @returns {object} the validated view
   */
  function normaliseView(raw) {
    var v = raw && typeof raw === "object" ? raw : {};
    var sort = v.sort && typeof v.sort === "object" ? v.sort : {};
    var stateOk = v.state === FILTER_ALL || Object.prototype.hasOwnProperty.call(STATES, str(v.state));
    var sources = [FILTER_ALL, SOURCE_MINE, SOURCE_OTHER, SOURCE_LAB];
    /* A data set's name is data, not a fixed vocabulary, so the remembered
       value is validated by its shape rather than against a list: every
       real value carries the prefix trainedValue writes. A name that no run
       carries any more is kept, offered in the control and shown as a table
       with no matching run, which is honest about what is filtering it. */
    var trained = str(v.trained);
    var trainedOk = trained === FILTER_ALL || trained === TRAINED_NONE
      || trained.indexOf(TRAINED_PREFIX) === 0;
    return {
      search: str(v.search),
      state: stateOk ? str(v.state) : FILTER_ALL,
      source: sources.indexOf(str(v.source)) >= 0 ? str(v.source) : FILTER_ALL,
      trained: trainedOk ? trained : FILTER_ALL,
      sort: {
        key: SORT_KEYS.indexOf(str(sort.key)) >= 0 ? str(sort.key) : SORT_STATE,
        dir: sort.dir === DIR_DESC ? DIR_DESC : DIR_ASC
      }
    };
  }

  /**
   * The whole board: the filter block and the grouped table, or one line
   * when no run matches.
   * @param {Array} runs the payload's runs
   * @param {object|null} champ the normalised champion
   * @param {object} opts {projectDir, projectSource, panelTask, taskPinned, view}
   * @returns {object} {board, shown, counted, present}: the board's HTML, how
   *   many runs the filters kept, the counts the summary states, and which
   *   sources the payload holds
   */
  function renderBoard(runs, champ, opts) {
    var view = opts.view;
    var described = describeRuns(runs, champ, opts.projectDir, opts.projectSource);
    var kept = described.filter(function (r) { return matchesFilters(r, view); });
    var groups = groupByFamily(kept, champ, opts.panelTask, view.sort);
    var board = groups.length
      ? renderTable(groups, champ, { projectDir: opts.projectDir, projectSource: opts.projectSource,
                                     panelTask: opts.panelTask, taskPinned: opts.taskPinned, sort: view.sort })
      : renderNoMatch();
    return { board: board, shown: kept.length, counted: countRuns(described), present: sourcesPresent(described) };
  }

  /**
   * What the payload holds, so no filter offers an option that would empty the
   * table: which of the three sources, and every training data set in name
   * order with the runs whose record names none last.
   * @param {Array<object>} described the described runs
   * @returns {object} {mine, other, lab, trained}
   */
  function sourcesPresent(described) {
    var present = { mine: false, other: false, lab: false, trained: [] };
    var seen = {};
    described.forEach(function (r) {
      present[r.source] = true;
      if (!Object.prototype.hasOwnProperty.call(seen, r.trained)) {
        seen[r.trained] = true;
        present.trained.push(r.trained);
      }
    });
    present.trained.sort(compareTrained);
    return present;
  }

  /**
   * Render the whole body from an all_runs payload: the summary line, the
   * filters and the grouped table, the empty sentence, or the error state
   * for a payload that carries an error or is not in the contract's shape.
   * @param {*} data the parsed JSON of GET /api/step/6/all_runs
   * @param {object} [opts] {projectDir, projectSource, panelTask, taskPinned,
   *   view}; projectSource is {name, kind} from projectSourceOf, or absent
   * @returns {string} HTML for the card body
   */
  function render(data, opts) {
    if (!data || typeof data !== "object" || Array.isArray(data)) return renderError(TEXT.malformed);
    if (data.error) return renderError(data.error);
    if (!Array.isArray(data.runs)) return renderError(TEXT.malformed);
    if (!data.runs.length) return renderEmpty();
    var options = opts || {};
    var view = normaliseView(options.view);
    var champ = normaliseChampion(data.champion, data.champion_scope);
    var built = renderBoard(data.runs, champ, {
      projectDir: str(options.projectDir),
      projectSource: normaliseSource(options.projectSource),
      panelTask: str(options.panelTask),
      taskPinned: Boolean(options.taskPinned),
      view: view
    });
    return renderSummaryLine(built.counted, built.shown, champ)
      + renderFilters(view, built.present)
      + '<div class="vega-runs-board" data-role="board">' + built.board + "</div>";
  }

  /**
   * The default fetch: same-origin JSON, with the server's error text as
   * the rejection message on a non-2xx response.
   * @param {string} url
   * @returns {Promise<object>}
   */
  function defaultFetchJson(url) {
    return fetch(url, { credentials: "same-origin" }).then(function (resp) {
      return resp.json().catch(function () { return {}; }).then(function (body) {
        if (!resp.ok) throw new Error(body && body.error ? body.error : "HTTP " + resp.status + " from " + url);
        return body;
      });
    });
  }

  /**
   * The body element of a card root, or a TypeError naming what is missing.
   * @param {Element} root the card root
   * @returns {Element}
   */
  function bodyOf(root) {
    if (!root || typeof root.querySelector !== "function") {
      throw new TypeError("VegaRuns needs the card root element (#" + CARD_ID + "), got " + String(root));
    }
    var body = root.querySelector(BODY_SELECTOR);
    if (!body) throw new TypeError("VegaRuns: the card root has no " + BODY_SELECTOR + " element to render into");
    return body;
  }

  /**
   * Read one remembered value, or "" when this browser refuses storage (a
   * private window, or site data turned off).
   * @param {string} key the storage key
   * @returns {string}
   */
  function readStore(key) {
    try {
      if (typeof localStorage === "undefined" || !localStorage) return "";
      return str(localStorage.getItem(key));
    } catch (err) {
      return "";
    }
  }

  /**
   * Write one remembered value, ignoring a browser that refuses storage.
   * @param {string} key the storage key
   * @param {string} value the value to keep
   */
  function writeStore(key, value) {
    try {
      if (typeof localStorage === "undefined" || !localStorage) return;
      localStorage.setItem(key, str(value));
    } catch (err) {
      return;
    }
  }

  /**
   * The view this browser last used, validated.
   * @returns {object} the view
   */
  function loadView() {
    return normaliseView({
      search: readStore(STORE_SEARCH),
      state: readStore(STORE_STATE) || FILTER_ALL,
      source: readStore(STORE_SOURCE) || FILTER_ALL,
      trained: readStore(STORE_TRAINED) || FILTER_ALL,
      sort: { key: readStore(STORE_SORT) || SORT_STATE, dir: readStore(STORE_DIR) || DIR_ASC }
    });
  }

  /**
   * Remember the view for the next visit.
   * @param {object} view the current view
   */
  function saveView(view) {
    writeStore(STORE_SEARCH, view.search);
    writeStore(STORE_STATE, view.state);
    writeStore(STORE_SOURCE, view.source);
    writeStore(STORE_TRAINED, view.trained);
    writeStore(STORE_SORT, view.sort.key);
    writeStore(STORE_DIR, view.sort.dir);
  }

  /**
   * Per-root state: the stored fetch, the last payload, the current view and
   * whether the listeners are bound.
   * @param {Element} root
   * @returns {object}
   */
  function stateFor(root) {
    var state = stateByRoot.get(root);
    if (!state) {
      state = { fetchJson: null, bound: false, seq: 0, data: null, projectDir: "", projectSource: null, view: loadView() };
      stateByRoot.set(root, state);
    }
    return state;
  }

  /**
   * What the Train panel is currently set to train, read from its own task
   * control so the card can refuse a hand-off the panel cannot accept. An
   * absent control leaves both facts blank and nothing is refused.
   * @returns {{panelTask: string, taskPinned: boolean}}
   */
  function panelTaskState() {
    if (typeof document === "undefined" || !document.getElementById) return { panelTask: "", taskPinned: false };
    var el = document.getElementById(TASK_SELECT_ID);
    if (!el) return { panelTask: "", taskPinned: false };
    var value = str(el.value) === PANEL_TASK_DETECT ? PANEL_TASK_DETECT : PANEL_TASK_SEGMENT;
    return { panelTask: value, taskPinned: Boolean(el.disabled) };
  }

  /**
   * Draw the card body from the state's payload and view.
   * @param {Element} root the card root element
   */
  function draw(root) {
    var state = stateFor(root);
    var panel = panelTaskState();
    bodyOf(root).innerHTML = render(state.data, {
      projectDir: state.projectDir,
      projectSource: state.projectSource,
      panelTask: panel.panelTask,
      taskPinned: panel.taskPinned,
      view: state.view
    });
  }

  /**
   * Redraw only the parts a filter or a sort changes: the summary counts and
   * the board. The filter controls keep their own elements, so the cursor
   * stays where the person is typing.
   * @param {Element} root the card root element
   */
  function redrawBoard(root) {
    var state = stateFor(root);
    var data = state.data;
    if (!data || !Array.isArray(data.runs) || !data.runs.length) { draw(root); return; }
    var panel = panelTaskState();
    var champ = normaliseChampion(data.champion, data.champion_scope);
    var built = renderBoard(data.runs, champ, {
      projectDir: state.projectDir,
      projectSource: state.projectSource,
      panelTask: panel.panelTask,
      taskPinned: panel.taskPinned,
      view: state.view
    });
    var body = bodyOf(root);
    var board = body.querySelector(BOARD_SELECTOR);
    if (board) board.innerHTML = built.board;
    var summary = body.querySelector(SUMMARY_SELECTOR);
    if (summary) summary.outerHTML = renderSummaryLine(built.counted, built.shown, champ);
    var clear = body.querySelector(CLEAR_SELECTOR);
    if (clear) clear.hidden = !filtersActive(state.view);
  }

  /**
   * Fetch the run list and the open project's folder, then render the card.
   * The fetch is injectable through opts.fetchJson and remembered for later
   * reloads. Each call takes a sequence number, and a response that arrives
   * after a newer call started is dropped so a stale list never overwrites a
   * fresh one. A project-state route that fails is not an error: the card
   * simply marks no row as the open project's and names no source of its own.
   * @param {Element} root the card root element
   * @param {{fetchJson?: function}} [opts]
   * @returns {Promise<void>} resolves once the body is rendered, never rejects
   */
  function load(root, opts) {
    var body = bodyOf(root);
    var state = stateFor(root);
    if (opts && typeof opts.fetchJson === "function") state.fetchJson = opts.fetchJson;
    var fetchJson = state.fetchJson || defaultFetchJson;
    var seq = ++state.seq;
    body.innerHTML = renderLoading();
    var runs;
    try {
      runs = Promise.resolve(fetchJson(RUNS_URL));
    } catch (err) {
      runs = Promise.reject(err);
    }
    var project;
    try {
      project = Promise.resolve(fetchJson(PROJECT_STATE_URL)).catch(function () { return null; });
    } catch (err) {
      project = Promise.resolve(null);
    }
    return runs.then(function (data) {
      return project.then(function (info) {
        if (seq !== state.seq) return;
        state.data = data;
        state.projectDir = projectDirOf(info);
        state.projectSource = normaliseSource(projectSourceOf(info));
        draw(root);
      });
    }, function (err) {
      if (seq === state.seq) body.innerHTML = renderError(err && err.message ? err.message : String(err));
    });
  }

  /**
   * The open project's folder from a /api/project/state answer, or "" when
   * no project is open or the answer is not in that shape.
   * @param {*} info the parsed answer
   * @returns {string}
   */
  function projectDirOf(info) {
    if (!info || typeof info !== "object" || info.loaded !== true) return "";
    var state = info.state;
    return state && typeof state === "object" ? str(state.project_dir) : "";
  }

  /**
   * Build the host-facing event for a row action.
   * @param {string} name the event name
   * @param {object} detail the run the action concerns
   * @returns {Event|object} a CustomEvent, or a plain object where CustomEvent is absent (Node)
   */
  function makeEvent(name, detail) {
    if (typeof CustomEvent === "function") return new CustomEvent(name, { bubbles: true, detail: detail });
    return { type: name, detail: detail, bubbles: true };
  }

  /**
   * The run behind one action button, as far as the last payload knows it,
   * so the confirm sentence can name its score and where it was trained.
   * @param {Element} root the card root element
   * @param {string} runDir the run folder from the button
   * @returns {object|null} the normalised run, or null when the payload has none
   */
  function runByDir(root, runDir) {
    var data = stateFor(root).data;
    var rows = data && Array.isArray(data.runs) ? data.runs : [];
    for (var i = 0; i < rows.length; i += 1) {
      var run = normaliseRun(rows[i]);
      if (run.run_dir && run.run_dir === runDir) return run;
    }
    return null;
  }

  /**
   * Name the run inside the host's promote confirm dialog.
   *
   * The dialog belongs to the host page, which writes one fixed sentence for
   * every row ("Make this run the champion?"). A student who mis-clicked one
   * row up cannot tell from it what is about to change, and the promotion
   * changes the default weights for every project. The host sets that text
   * synchronously while handling the event, so the card refines it for its
   * own row immediately afterwards. When the host page starts naming the run
   * itself this function can go.
   * @param {object|null} run the normalised run being promoted
   */
  function describePromoteConfirm(run) {
    if (!run || typeof document === "undefined" || !document.getElementById) return;
    var el = document.getElementById(CONFIRM_MSG_ID);
    if (!el) return;
    var origin = (run.origin === ORIGIN_LAB ? TEXT.labWord.toLowerCase() : TEXT.projectWord.toLowerCase())
      + " " + (run.project_name || run.run_dir);
    var name = run.run_name || UNNAMED_RUN;
    if (run.map50 === null) {
      el.textContent = TEXT.promoteConfirmPlain.replace("{name}", name).replace("{origin}", origin);
      return;
    }
    el.textContent = TEXT.promoteConfirm
      .replace("{name}", name)
      .replace("{score}", familyLabel(run.family, run.task) + " " + formatScore(run.map50))
      .replace("{origin}", origin);
  }

  /**
   * Dispatch one row action on the card root for the host to handle. The
   * detail carries the run's identity and its score, so a host that names
   * the run in a dialog has the facts without reading the table back.
   * @param {Element} root the card root element
   * @param {string} action evaluate, promote or finetune
   * @param {{run_dir: *, run_name: *, task: *}} run the run the action concerns
   */
  function dispatchAction(root, action, run) {
    var name = EVENTS[action];
    if (!name) throw new RangeError("VegaRuns: unknown row action " + String(action));
    var r = run || {};
    var known = typeof root.querySelector === "function" ? runByDir(root, str(r.run_dir)) : null;
    root.dispatchEvent(makeEvent(name, {
      run_dir: str(r.run_dir),
      run_name: str(r.run_name),
      task: str(r.task),
      family: known ? known.family : "",
      map50: known ? known.map50 : null,
      origin: known ? known.origin : "",
      project_name: known ? known.project_name : ""
    }));
    if (action === "promote") describePromoteConfirm(known);
  }

  /**
   * Apply a click on a sortable column heading: the same column reverses,
   * a different column starts in its own natural direction.
   * @param {Element} root the card root element
   * @param {string} key the sort key clicked
   */
  function applySort(root, key) {
    var state = stateFor(root);
    if (SORT_KEYS.indexOf(key) < 0) return;
    if (state.view.sort.key === key) {
      state.view.sort.dir = state.view.sort.dir === DIR_ASC ? DIR_DESC : DIR_ASC;
    } else {
      state.view.sort = { key: key, dir: NATURAL_DIR[key] };
    }
    saveView(state.view);
    redrawBoard(root);
  }

  /**
   * The delegated click handler: Refresh reloads, Clear filters widens every
   * filter, a column heading sorts, a row action dispatches its event, and
   * anything else (a disabled button, a click outside any control) is
   * ignored.
   * @param {Element} root the card root element
   * @param {Event} ev the click event
   */
  function handleClick(root, ev) {
    var target = ev && ev.target;
    if (!target || typeof target.closest !== "function") return;
    var head = target.closest("[data-sort]");
    if (head) {
      applySort(root, head.getAttribute("data-sort"));
      return;
    }
    var btn = target.closest("[data-action]");
    if (!btn || btn.disabled) return;
    var action = btn.getAttribute("data-action");
    if (action === ACTION_REFRESH) {
      load(root);
      return;
    }
    if (action === ACTION_CLEAR) {
      clearFilters(root);
      return;
    }
    if (!EVENTS[action]) return;
    dispatchAction(root, action, {
      run_dir: btn.getAttribute("data-run-dir"),
      run_name: btn.getAttribute("data-run-name"),
      task: btn.getAttribute("data-task")
    });
  }

  /**
   * Put every filter back to its widest setting and redraw.
   * @param {Element} root the card root element
   */
  function clearFilters(root) {
    var state = stateFor(root);
    var fresh = defaultView();
    state.view.search = fresh.search;
    state.view.state = fresh.state;
    state.view.source = fresh.source;
    state.view.trained = fresh.trained;
    saveView(state.view);
    draw(root);
  }

  /**
   * The delegated filter handler: read the control that changed into the
   * view, remember it, and redraw the board around the live controls.
   * @param {Element} root the card root element
   * @param {Event} ev the input or change event
   */
  function handleFilter(root, ev) {
    var target = ev && ev.target;
    if (!target || typeof target.getAttribute !== "function") return;
    var field = target.getAttribute("data-filter");
    if (!field) return;
    var state = stateFor(root);
    state.view[field] = str(target.value);
    state.view = normaliseView(state.view);
    saveView(state.view);
    redrawBoard(root);
  }

  /**
   * Bind the handlers once and load the runs.
   * @param {Element} root the card root element
   * @param {{fetchJson?: function}} [opts]
   * @returns {Promise<void>}
   */
  function init(root, opts) {
    bodyOf(root);
    var state = stateFor(root);
    if (!state.bound) {
      root.addEventListener("click", function (ev) { handleClick(root, ev); });
      root.addEventListener("input", function (ev) { handleFilter(root, ev); });
      root.addEventListener("change", function (ev) { handleFilter(root, ev); });
      state.bound = true;
    }
    return load(root, opts);
  }

  /**
   * The plan's entry point for the host page: initialise the card at its
   * contract id. Call it when the Evaluate Model panel opens the "All
   * trained models" disclosure that holds this card; a later call reloads.
   * A missing document or card is logged and yields a resolved promise, so
   * the host's panel switch never breaks on a page without the card.
   * @returns {Promise<void>}
   */
  function initStep6Runs() {
    var card = typeof document === "undefined" ? null : document.getElementById(CARD_ID);
    if (!card) {
      console.error("VegaRuns: no #" + CARD_ID + " on this page, the runs card was not initialised");
      return Promise.resolve();
    }
    return init(card);
  }

  var VegaRuns = {
    RUNS_URL: RUNS_URL,
    PROJECT_STATE_URL: PROJECT_STATE_URL,
    EVENTS: EVENTS,
    ACTION_REFRESH: ACTION_REFRESH,
    ACTION_CLEAR: ACTION_CLEAR,
    STORE_PREFIX: STORE_PREFIX,
    STATES: STATES,
    esc: esc,
    formatAst: formatAst,
    taskLabel: taskLabel,
    familyLabel: familyLabel,
    strictFamilyLabel: strictFamilyLabel,
    normaliseRun: normaliseRun,
    normaliseChampion: normaliseChampion,
    normaliseView: normaliseView,
    isOpenProjectRun: isOpenProjectRun,
    runState: runState,
    sourceKind: sourceKind,
    matchesFilters: matchesFilters,
    sortRows: sortRows,
    describeRuns: describeRuns,
    groupByFamily: groupByFamily,
    countRuns: countRuns,
    finetuneRefusal: finetuneRefusal,
    finetuneOffer: finetuneOffer,
    crossFamily: crossFamily,
    trainedOn: trainedOn,
    trainedTip: trainedTip,
    trainedValue: trainedValue,
    projectSourceOf: projectSourceOf,
    renderNameParts: renderNameParts,
    renderRow: renderRow,
    renderRows: renderRows,
    renderGroupHead: renderGroupHead,
    renderFilters: renderFilters,
    renderSummaryLine: renderSummaryLine,
    renderEmpty: renderEmpty,
    renderLoading: renderLoading,
    renderError: renderError,
    renderNoMatch: renderNoMatch,
    render: render,
    load: load,
    init: init,
    initStep6Runs: initStep6Runs,
    handleClick: handleClick,
    handleFilter: handleFilter,
    dispatchAction: dispatchAction
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = VegaRuns;
  }
  if (typeof window !== "undefined") {
    window.VegaRuns = VegaRuns;
    window.initStep6Runs = initStep6Runs;
  }
})();

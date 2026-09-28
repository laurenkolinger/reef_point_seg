/* VEGA evaluation report card for the Step 6 (internal step 7) panel of the
   pipeline window. Reads GET /api/step/7/report and renders it: a heading
   with the run name and evaluation time, the top-line metrics by metric
   family (a detector scores boxes, a segmenter outlines and boxes), the
   per-species table (sortable, sort kept per browser), the figure strip, the
   prediction samples and the report text behind a disclosure.

   Pure render functions (data in, HTML string out) sit apart from the DOM
   glue so Node 18 can test them without a browser
   (tests/test_vega_report.js). Every string that arrives from the server is
   escaped before it reaches the page: report text, file names and class
   names come from files on disk and model class maps and are data, never
   markup. Exposes window.VegaReport and window.loadStep7ReportCard; the
   fragment it fills is templates/_step7_report.html and the stylesheet is
   static/orchestrator_report.css. */
(function (root, factory) {
  "use strict";
  const api = factory(root);
  if (typeof module === "object" && module !== null && module.exports) { module.exports = api; }
  if (root) { root.VegaReport = api; root.loadStep7ReportCard = api.loadStep7ReportCard; }
})(typeof window !== "undefined" ? window : null, function (root) {
  "use strict";

  // Routes from the wave-2 API contract (docs/superpowers/plans/2026-09-04-vega-training-foldin.md).
  const REPORT_URL = "/api/step/7/report";
  const IMAGE_URL = "/api/step/7/image";

  // Hooks into templates/_step7_report.html.
  const CARD_ID = "s7-report-card";
  const BODY_SELECTOR = ".vega-report-body";
  const STATUS_SELECTOR = ".vega-report-status";
  const SPECIES_SELECTOR = ".vega-report-species";
  const SORT_BUTTON_SELECTOR = ".vega-report-sort";

  // Sorting of the per-species table; the state is kept per browser.
  const SORT_STORAGE_KEY = "vega.report.sort";
  const SORT_KEYS = Object.freeze(["name", "instances", "precision", "recall", "map50", "map5095"]);
  const SORT_DIRS = Object.freeze(["asc", "desc"]);
  const DEFAULT_SORT = Object.freeze({ key: "map50", dir: "desc" });
  const ARIA_SORT = Object.freeze({ asc: "ascending", desc: "descending" });
  const TEXT_SORT_KEY = "name";

  // Metric families and the columns each one shows.
  const FAMILY_MASK = "mask";
  const FAMILY_BOX = "box";
  const FAMILY_COLUMNS = Object.freeze({
    mask: Object.freeze([{ prefix: "mask", label: "Outlines" }, { prefix: "box", label: "Boxes" }]),
    box: Object.freeze([{ prefix: "box", label: "Boxes" }]),
  });
  const TOP_LINE_ROWS = Object.freeze([
    { label: "mAP50", suffix: "map50" },
    { label: "mAP50-95", suffix: "map50_95" },
    { label: "Precision", suffix: "precision" },
    { label: "Recall", suffix: "recall" },
  ]);
  const ROW_SUFFIXES = Object.freeze({ precision: "precision", recall: "recall", map50: "map50", map5095: "map50_95" });
  const COLUMN_LABELS = Object.freeze({
    name: "Species", instances: "Instances", precision: "Precision", recall: "Recall", map50: "mAP50", map5095: "mAP50-95",
  });
  const SCORE_DECIMALS = 3;
  const SCORE_COLUMN_COUNT = 4;

  // Figures in their fixed order; a null url is simply not rendered.
  const FIGURES = Object.freeze([
    { key: "box_pr", caption: "Precision-recall curve, boxes" },
    { key: "box_f1", caption: "F1 against confidence, boxes" },
    { key: "mask_pr", caption: "Precision-recall curve, outlines" },
    { key: "mask_f1", caption: "F1 against confidence, outlines" },
    { key: "confusion_norm", caption: "Confusion matrix, normalized" },
    { key: "confusion_abs", caption: "Confusion matrix, counts" },
    { key: "results", caption: "Training curves by epoch" },
  ]);

  // Markdown: an image is kept only when its source is a file directly under previews/.
  const PREVIEW_REL_RE = /^previews\/([^/\\]+)$/;
  const HEADING_RE = /^(#{1,6})\s+(.*)$/;
  const BULLET_RE = /^[-*]\s+(.*)$/;
  const RULE_RE = /^(?:-{3,}|\*{3,})$/;
  const CONTINUATION_RE = /^\s{2,}\S/;
  const TABLE_SEPARATOR_CELL_RE = /^:?-+:?$/;
  // Table cells are split on the pipes the report did not escape; \| is one character of
  // cell text, the way a markdown reader outside this page reads it too.
  const UNESCAPED_PIPE_RE = /(?<!\\)\|/;
  const ESCAPED_PIPE_RE = /\\\|/g;
  // Code spans and finished image tags are parked under an index wrapped in NUL characters
  // while the other inline rules run, then restored; a NUL in the text itself is stripped
  // first so it can never collide with a token.
  const CODE_TOKEN = String.fromCharCode(0);
  const CODE_TOKEN_RE = new RegExp(CODE_TOKEN + "(\\d+)" + CODE_TOKEN, "g");
  const NUL_RE = new RegExp(CODE_TOKEN, "g");
  // Bounded captures: an unbounded [^\]]* backtracks quadratically on a run of brackets.
  const MARKDOWN_TEXT_MAX = 500;
  const MARKDOWN_TARGET_MAX = 2000;
  const IMAGE_RE = new RegExp("!\\[([^\\]]{0," + MARKDOWN_TEXT_MAX + "})\\]\\(([^)]{1," + MARKDOWN_TARGET_MAX + "})\\)", "g");
  const LINK_RE = new RegExp("\\[([^\\]]{1," + MARKDOWN_TEXT_MAX + "})\\]\\(([^)]{1," + MARKDOWN_TARGET_MAX + "})\\)", "g");
  const HEADING_LEVEL_MAX = 3;
  const HEADING_TAG_OFFSET = 3;

  // Species names: "Genus-species" class names resolve to the binomial in full.
  const BINOMIAL_RE = /^[A-Z][a-z]+(?:-[a-z]+){1,2}$/;
  // A resolved full name that is a binomial on its own ("Acropora palmata"), the shape a
  // names table or the code registry writes for a species with no qualifiers. The three
  // branches below match evaluate_run.class_display_html, so the card, the report text and
  // the PDF render one name one way.
  const SPECIES_NAME_RE = /^[A-Z][a-z]+(?: [a-z]+){1,2}$/;
  // A resolved full name that opens with a binomial and then adds qualifiers after a
  // comma, the shape a staged names table writes ("Epinephelus striatus, Nassau grouper,
  // dark phase"): group 1 is the species and is italic, group 2 is not a species name.
  const BINOMIAL_PREFIX_RE = /^([A-Z][a-z]+(?: [a-z]+){1,2})(,\s[\s\S]*)$/;
  const ISO_AST_RE = /^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})/;
  const TIMEZONE_LABEL = "AST";
  // Only a same-origin path is ever placed in a src or href.
  const SAFE_URL_RE = /^\/(?![/\\])\S*$/;

  // Copy on the card.
  const EMPTY_TEXT = "No evaluation has been run for this training run yet.";
  const LOADING_TEXT = "Loading the evaluation report.";
  const NO_PDF_TEXT = "No PDF was built for this evaluation.";
  const DETECTOR_TEXT = "This run is a detector: it predicts boxes, not outlines.";
  const NO_SPECIES_TEXT = "No per-species scores were recorded for this evaluation.";
  const NO_SCORE_TEXT = "no score";
  const ABSENT_TEXT = "not present in this split";
  const UNSCORED_TEXT = "no scores recorded";
  const PDF_LABEL = "Download PDF";
  const FULL_TEXT_LABEL = "Full report text";
  const SECTION_TOP_LINE = "Top-line metrics";
  const SECTION_SPECIES = "Per species";
  const SECTION_FIGURES = "Figures";
  const SECTION_PREVIEWS = "Prediction samples";
  const FETCH_FAILED_PREFIX = "Could not load the evaluation report: ";

  // Tooltips (platform rule: one or two complete sentences, the first names who does what).
  const TOOLTIPS = Object.freeze({
    pdf: "VICARIUS sends report.pdf from this evaluation's folder to your browser's downloads.",
    figure: "Open the figure at full size in a new browser tab.",
    fullText: "VICARIUS shows report.md from the evaluation folder, rendered from its markdown.",
    columns: Object.freeze({
      name: "Species names the class the model was trained on, with its class code beside it. Press to sort by name; press again to reverse the order.",
      instances: "Instances counts the labeled objects of this species in the evaluated split. Press to sort by count; press again to reverse the order.",
      precision: "Precision for this species is the share of its predictions that matched a labeled object. Press to sort by precision; press again to reverse the order.",
      recall: "Recall for this species is the share of its labeled objects the model found. Press to sort by recall; press again to reverse the order.",
      map50: "mAP50 for this species is its average precision when a prediction counts once it overlaps the labeled object by half. Press to sort by mAP50; press again to reverse the order.",
      map5095: "mAP50-95 for this species averages its precision over overlap thresholds from 0.5 to 0.95, so tight fits score higher. Press to sort by mAP50-95; press again to reverse the order.",
    }),
  });

  // Per-body state for the sort handler: rows and the current sort.
  const STATE = new WeakMap();
  const WIRED = new WeakSet();

  // ── small pure helpers ─────────────────────────────────────────────

  /**
   * True when value is a plain object (not null, not an array).
   * @param {*} value
   * @returns {boolean}
   */
  function isObject(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
  }

  /**
   * Escape the five HTML-significant characters so any server string is text, never markup.
   * @param {*} value null and undefined become the empty string; anything else is stringified.
   * @returns {string}
   */
  function escapeHtml(value) {
    const text = value === null || value === undefined ? "" : String(value);
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  /**
   * Reverse escapeHtml for a fragment lifted out of already-escaped text (an image source).
   * @param {string} text
   * @returns {string}
   */
  function unescapeHtml(text) {
    return String(text)
      .replace(/&quot;/g, "\"")
      .replace(/&#39;/g, "'")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&amp;/g, "&");
  }

  /**
   * True when url is a same-origin path: starts with one slash, no scheme, no whitespace.
   * @param {*} url
   * @returns {boolean}
   */
  function isSafeUrl(url) {
    return typeof url === "string" && SAFE_URL_RE.test(url);
  }

  /**
   * A finite number as itself, anything else as null.
   * @param {*} value
   * @returns {number|null}
   */
  function numberOrNull(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }

  /**
   * A preview count: numeric strings are accepted, anything unreadable counts as 0.
   * @param {*} value
   * @returns {number}
   */
  function countOrZero(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : 0;
  }

  /**
   * A count and its noun, with the noun in the singular only for exactly one:
   * plural(1, "instance") is "1 instance" and plural(2, "detection") is
   * "2 detections". Nouns whose plural is not the noun plus an s do not go
   * through this helper.
   * @param {number} count the number to state
   * @param {string} noun the singular noun
   * @returns {string} the count, a space, and the noun in the right number
   */
  function plural(count, noun) {
    return count + " " + noun + (count === 1 ? "" : "s");
  }

  /**
   * Format a score with three decimals, or null when it is not a finite number.
   * @param {*} value
   * @returns {string|null}
   */
  function formatScore(value) {
    const n = numberOrNull(value);
    return n === null ? null : n.toFixed(SCORE_DECIMALS);
  }

  /**
   * Render the evaluation timestamp as "YYYY-MM-DD HH:MM AST". The route sends ISO with
   * the AST offset; the date and minutes are lifted straight from the text so no browser
   * timezone conversion happens. Unparseable text is shown as it is; a non-string is empty.
   * @param {*} value
   * @returns {string}
   */
  function formatEvaluatedAt(value) {
    if (typeof value !== "string") return "";
    const text = value.trim();
    const m = ISO_AST_RE.exec(text);
    return m ? m[1] + " " + m[2] + " " + TIMEZONE_LABEL : text;
  }

  /**
   * Last path segment of a folder path, ignoring trailing slashes.
   * @param {*} value
   * @returns {string}
   */
  function basename(value) {
    if (typeof value !== "string") return "";
    const parts = value.replace(/[\\/]+$/, "").split(/[\\/]/);
    return parts[parts.length - 1] || "";
  }

  /**
   * The report route for a run folder; a blank folder asks the route for its default
   * (the open project's last evaluation).
   * @param {string} runDir absolute run folder, or blank
   * @returns {string}
   */
  function reportUrl(runDir) {
    const dir = typeof runDir === "string" ? runDir.trim() : "";
    return dir ? REPORT_URL + "?run_dir=" + encodeURIComponent(dir) : REPORT_URL;
  }

  /**
   * The image route for a preview referenced from report.md, or null when the source is
   * not a file directly under previews/ (parent segments, nested folders, absolute and
   * external sources are all refused).
   * @param {string} runDir absolute run folder
   * @param {string} src relative source as written in the markdown
   * @returns {string|null}
   */
  function previewImageUrl(runDir, src) {
    if (typeof runDir !== "string" || runDir.trim() === "" || typeof src !== "string") return null;
    const m = PREVIEW_REL_RE.exec(src);
    if (!m || m[1] === "." || m[1] === "..") return null;
    return IMAGE_URL + "?run_dir=" + encodeURIComponent(runDir) + "&rel=" + encodeURIComponent(src);
  }

  /**
   * Resolve a class name to the species shown on the page. labelNames may be keyed by the
   * class token (a code table the host passed) or by the class id (metrics.class_full_names,
   * which the evaluation writes from the staged dataset's names table); the token key wins,
   * then the class id, and either keeps the token as the chip. A "Genus-species" class name
   * becomes the binomial with no chip (the name is the class); anything else stays as it is,
   * because a code that resolves to nothing is shown unchanged.
   * @param {*} className
   * @param {Object<string,string>|null} labelNames optional full-name map, keyed by class token or class id
   * @param {string|number|null} [classId] this class's id, for a map keyed by class id
   * @returns {{name: string, code: string|null, resolved: boolean}}
   */
  function speciesFromClassName(className, labelNames, classId) {
    const raw = className === null || className === undefined ? "" : String(className);
    if (isObject(labelNames)) {
      const keys = [raw];
      if (classId !== null && classId !== undefined) keys.push(String(classId));
      for (let i = 0; i < keys.length; i += 1) {
        if (!Object.prototype.hasOwnProperty.call(labelNames, keys[i])) continue;
        const mapped = labelNames[keys[i]];
        if (typeof mapped === "string" && mapped.trim()) return { name: mapped.trim(), code: raw, resolved: true };
      }
    }
    if (BINOMIAL_RE.test(raw)) return { name: raw.replace(/-/g, " "), code: null, resolved: true };
    return { name: raw, code: null, resolved: false };
  }

  /**
   * Choose the metric family: the declared metric_family when valid, otherwise mask when
   * any mask score is a number, otherwise box.
   * @param {*} metrics the metrics object from the route
   * @returns {string} "mask" or "box"
   */
  function metricFamily(metrics) {
    const m = isObject(metrics) ? metrics : {};
    if (m.metric_family === FAMILY_MASK || m.metric_family === FAMILY_BOX) return m.metric_family;
    const overall = isObject(m.overall) ? m.overall : {};
    const hasMask = TOP_LINE_ROWS.some(row => numberOrNull(overall[FAMILY_MASK + "_" + row.suffix]) !== null);
    return hasMask ? FAMILY_MASK : FAMILY_BOX;
  }

  // ── sorting ────────────────────────────────────────────────────────

  /**
   * A valid sort state, or a copy of the default when the input is not one.
   * @param {*} state
   * @returns {{key: string, dir: string}}
   */
  function normaliseSort(state) {
    if (isObject(state) && SORT_KEYS.includes(state.key) && SORT_DIRS.includes(state.dir)) {
      return { key: state.key, dir: state.dir };
    }
    return { key: DEFAULT_SORT.key, dir: DEFAULT_SORT.dir };
  }

  /**
   * The sort state after pressing a column: the same column flips direction, a new
   * numeric column starts descending, the name column starts ascending, an unknown key
   * leaves the state alone.
   * @param {{key: string, dir: string}} current
   * @param {string} key
   * @returns {{key: string, dir: string}}
   */
  function nextSort(current, key) {
    const state = normaliseSort(current);
    if (!SORT_KEYS.includes(key)) return state;
    if (key === state.key) return { key: key, dir: state.dir === "asc" ? "desc" : "asc" };
    return { key: key, dir: key === TEXT_SORT_KEY ? "asc" : "desc" };
  }

  /**
   * Read the saved sort state from a storage object; every failure yields the default.
   * @param {Storage|null} storage localStorage or a stand-in with getItem
   * @returns {{key: string, dir: string}}
   */
  function readSortState(storage) {
    try {
      if (!storage || typeof storage.getItem !== "function") return normaliseSort(null);
      const raw = storage.getItem(SORT_STORAGE_KEY);
      return normaliseSort(typeof raw === "string" ? JSON.parse(raw) : null);
    } catch (err) {
      return normaliseSort(null);
    }
  }

  /**
   * Save the sort state; a missing or throwing storage is ignored.
   * @param {Storage|null} storage
   * @param {{key: string, dir: string}} state
   */
  function writeSortState(storage, state) {
    try {
      if (storage && typeof storage.setItem === "function") storage.setItem(SORT_STORAGE_KEY, JSON.stringify(normaliseSort(state)));
    } catch (err) {
      // A blocked storage keeps the sort for this page view only.
    }
  }

  /**
   * Sort group of a row: 0 scored, 1 unscored (instances but no scores), 2 absent.
   * @param {Object} row
   * @returns {number}
   */
  function rowRank(row) {
    if (row.absent) return 2;
    return row.unscored ? 1 : 0;
  }

  /**
   * Compare two rows on one column: scored species first, then unscored, then absent;
   * within a group a missing score sorts after every number, then the value in the
   * requested direction, then input order.
   * @param {{row: Object, index: number}} a
   * @param {{row: Object, index: number}} b
   * @param {string} key
   * @param {number} sign 1 ascending, -1 descending
   * @returns {number}
   */
  function compareRows(a, b, key, sign) {
    const ra = rowRank(a.row);
    const rb = rowRank(b.row);
    if (ra !== rb) return ra - rb;
    if (key === TEXT_SORT_KEY) {
      const an = String(a.row.name || "").toLowerCase();
      const bn = String(b.row.name || "").toLowerCase();
      if (an < bn) return -sign;
      if (an > bn) return sign;
      return a.index - b.index;
    }
    const av = numberOrNull(a.row[key]);
    const bv = numberOrNull(b.row[key]);
    if ((av === null) !== (bv === null)) return av === null ? 1 : -1;
    if (av !== null && av !== bv) return av < bv ? -sign : sign;
    return a.index - b.index;
  }

  /**
   * A new array of rows sorted by key and direction; the sort is stable and never
   * mutates its input.
   * @param {Array<Object>} rows from buildRows
   * @param {string} key one of SORT_KEYS
   * @param {string} dir "asc" or "desc"
   * @returns {Array<Object>}
   */
  function sortRows(rows, key, dir) {
    const list = Array.isArray(rows) ? rows.filter(isObject) : [];
    const sort = normaliseSort({ key: key, dir: dir });
    const sign = sort.dir === "asc" ? 1 : -1;
    return list
      .map((row, index) => ({ row: row, index: index }))
      .sort((a, b) => compareRows(a, b, sort.key, sign))
      .map(entry => entry.row);
  }

  // ── rows ───────────────────────────────────────────────────────────

  /**
   * Class id per class token from a metrics class map (class id to token), as strings, so a
   * name map keyed by class id can be reached from a row keyed by its token. The first id
   * wins when two ids carry the same token. A class map of any other shape gives no ids.
   * @param {*} classNames metrics.class_names, a mapping keyed by class id or a list in class-id order
   * @returns {Object<string,string>} class token to class id
   */
  function classIdsByName(classNames) {
    const out = {};
    /** Record one token's id, keeping the first id when a token repeats. */
    function put(token, id) {
      const key = String(token);
      if (!Object.prototype.hasOwnProperty.call(out, key)) out[key] = String(id);
    }
    if (Array.isArray(classNames)) { classNames.forEach(function (token, index) { put(token, index); }); return out; }
    if (isObject(classNames)) { Object.keys(classNames).forEach(function (id) { put(classNames[id], id); }); }
    return out;
  }

  /**
   * One row per class: every name in per_class, per_class_instances and absent_classes,
   * in first-seen order, with the family's scores, or the absent flag (not in this split),
   * or the unscored flag (instances but no scores). Each row carries its class id from
   * metrics.class_names, or null when that map does not name the token.
   * @param {*} metrics the metrics object from the route
   * @param {string} family "mask" or "box"
   * @param {Object<string,string>|null} labelNames optional full-name map, keyed by class token or class id
   * @returns {Array<Object>}
   */
  function buildRows(metrics, family, labelNames) {
    const m = isObject(metrics) ? metrics : {};
    const idByName = classIdsByName(m.class_names);
    const perClass = isObject(m.per_class) ? m.per_class : {};
    const counts = isObject(m.per_class_instances) ? m.per_class_instances : {};
    const absent = Array.isArray(m.absent_classes) ? m.absent_classes.map(String) : [];
    const prefix = family === FAMILY_MASK ? FAMILY_MASK : FAMILY_BOX;
    const seen = new Set();
    const names = [];
    /** Add a class name once, keeping first-seen order. */
    function add(name) {
      const key = String(name);
      if (!seen.has(key)) { seen.add(key); names.push(key); }
    }
    Object.keys(perClass).forEach(add);
    Object.keys(counts).forEach(add);
    absent.forEach(add);
    return names.map(function (className) {
      const scores = isObject(perClass[className]) ? perClass[className] : null;
      const instances = numberOrNull(counts[className]);
      // Declared absent, or unscored with no instances: not in this split. Unscored with
      // instances: the evaluation recorded no scores for it (an older metrics file).
      const isAbsent = absent.includes(className) || (scores === null && !(instances > 0));
      const isUnscored = scores === null && !isAbsent;
      const classId = Object.prototype.hasOwnProperty.call(idByName, className) ? idByName[className] : null;
      const species = speciesFromClassName(className, labelNames, classId);
      /** The family's score for one column, or null when the class has no scores. */
      function pick(suffix) {
        return isAbsent || isUnscored ? null : numberOrNull(scores[prefix + "_" + suffix]);
      }
      return {
        className: className,
        classId: classId,
        name: species.name,
        code: species.code,
        resolved: species.resolved,
        instances: instances,
        precision: pick(ROW_SUFFIXES.precision),
        recall: pick(ROW_SUFFIXES.recall),
        map50: pick(ROW_SUFFIXES.map50),
        map5095: pick(ROW_SUFFIXES.map5095),
        absent: isAbsent,
        unscored: isUnscored,
      };
    });
  }

  // ── render pieces ──────────────────────────────────────────────────

  /**
   * A section heading inside the card (the card itself is an h3).
   * @param {string} text plain copy, already safe
   * @returns {string}
   */
  function sectionHeading(text) {
    return '<h4 class="vega-report-h">' + text + "</h4>";
  }

  /**
   * A numeric table cell: the formatted score, or the placeholder word when missing.
   * @param {*} value
   * @returns {string}
   */
  function scoreCell(value) {
    const text = formatScore(value);
    return text === null
      ? '<td class="num vega-report-noscore">' + NO_SCORE_TEXT + "</td>"
      : '<td class="num">' + text + "</td>";
  }

  /**
   * The heading row: run name, "evaluated YYYY-MM-DD HH:MM AST", and the PDF link or the
   * sentence saying no PDF was built.
   * @param {Object} data the route payload
   * @returns {string}
   */
  function renderHeading(data) {
    const runName = (typeof data.run_name === "string" && data.run_name.trim()) || basename(data.run_dir);
    const when = formatEvaluatedAt(data.evaluated_at);
    const pdf = isSafeUrl(data.pdf_url)
      ? '<a class="btn btn-magenta btn-sm vega-report-pdf" href="' + escapeHtml(data.pdf_url) + '" title="' + escapeHtml(TOOLTIPS.pdf) + '">' + PDF_LABEL + "</a>"
      : '<span class="vega-report-nopdf">' + NO_PDF_TEXT + "</span>";
    return '<div class="vega-report-run">'
      + '<span class="vega-report-runname">' + escapeHtml(runName) + "</span>"
      + (when ? '<span class="vega-report-when">evaluated ' + escapeHtml(when) + "</span>" : "")
      + '<span class="vega-report-spacer"></span>' + pdf
      + "</div>";
  }

  /**
   * The top-line table from metrics.overall: one value column per family column, and for a
   * detector the one sentence saying it predicts boxes.
   * @param {Object} metrics
   * @param {string} family "mask" or "box"
   * @returns {string}
   */
  function renderTopLine(metrics, family) {
    const overall = isObject(metrics.overall) ? metrics.overall : {};
    const columns = FAMILY_COLUMNS[family] || FAMILY_COLUMNS[FAMILY_BOX];
    const head = "<tr><th>Metric</th>" + columns.map(c => '<th class="num">' + c.label + "</th>").join("") + "</tr>";
    const rows = TOP_LINE_ROWS.map(function (row) {
      return "<tr><td>" + row.label + "</td>" + columns.map(c => scoreCell(overall[c.prefix + "_" + row.suffix])).join("") + "</tr>";
    }).join("");
    const note = family === FAMILY_BOX ? '<div class="vega-report-line">' + DETECTOR_TEXT + "</div>" : "";
    return sectionHeading(SECTION_TOP_LINE)
      + '<div class="vega-report-scroll"><table class="vega-report-table vega-report-topline"><thead>' + head
      + "</thead><tbody>" + rows + "</tbody></table></div>" + note;
  }

  /**
   * A resolved species name as escaped HTML, in the three branches evaluate_run renders in
   * the report text and the PDF: italic throughout when the whole name is a binomial; the
   * binomial alone in italic when a staged name adds qualifiers after a comma, because
   * "Nassau grouper, dark phase" is not part of the species name; upright body text when
   * the name is not a species at all ("grouper, species not identified"). The qualifiers
   * and a name that is not a species carry the plain class, so they read in the same 13px
   * body face as the binomial instead of inheriting the table's monospace.
   * @param {string} name the resolved full name
   * @returns {string}
   */
  function speciesNameHtml(name) {
    const text = name === null || name === undefined ? "" : String(name);
    const open = '<em class="vega-report-binomial">';
    const plainOpen = '<span class="vega-report-plain">';
    const parts = BINOMIAL_PREFIX_RE.exec(text);
    if (parts) return open + escapeHtml(parts[1]) + "</em>" + plainOpen + escapeHtml(parts[2]) + "</span>";
    if (SPECIES_NAME_RE.test(text)) return open + escapeHtml(text) + "</em>";
    return plainOpen + escapeHtml(text) + "</span>";
  }

  /**
   * One species row: full name in italic with the code chip (or the plain class name when
   * it resolves to nothing), the instance count, then the four scores, or the absent or
   * unscored sentence.
   * @param {Object} row from buildRows
   * @returns {string}
   */
  function renderSpeciesRow(row) {
    const name = row.resolved
      ? speciesNameHtml(row.name)
        + (row.code ? ' <span class="vega-report-chip">' + escapeHtml(row.code) + "</span>" : "")
      : '<span class="vega-report-plain">' + escapeHtml(row.name) + "</span>";
    const instances = row.instances === null ? '<td class="num"></td>' : '<td class="num">' + row.instances + "</td>";
    if (row.absent) {
      return '<tr class="vega-report-absent"><td>' + name + "</td>" + instances
        + '<td colspan="' + SCORE_COLUMN_COUNT + '" class="vega-report-absent-text">' + ABSENT_TEXT + "</td></tr>";
    }
    if (row.unscored) {
      return '<tr class="vega-report-unscored"><td>' + name + "</td>" + instances
        + '<td colspan="' + SCORE_COLUMN_COUNT + '" class="vega-report-unscored-text">' + UNSCORED_TEXT + "</td></tr>";
    }
    return "<tr><td>" + name + "</td>" + instances
      + scoreCell(row.precision) + scoreCell(row.recall) + scoreCell(row.map50) + scoreCell(row.map5095) + "</tr>";
  }

  /**
   * The per-species table sorted by the given state, with a sort button and its tooltip on
   * every column header and aria-sort on the active one.
   * @param {Array<Object>} rows from buildRows
   * @param {{key: string, dir: string}} sort
   * @returns {string}
   */
  function renderPerClassTable(rows, sort) {
    const state = normaliseSort(sort);
    const sorted = sortRows(rows, state.key, state.dir);
    if (!sorted.length) return '<div class="vega-report-line">' + NO_SPECIES_TEXT + "</div>";
    const head = "<tr>" + SORT_KEYS.map(function (key) {
      const active = key === state.key;
      return '<th scope="col"' + (key === TEXT_SORT_KEY ? "" : ' class="num"')
        + (active ? ' aria-sort="' + ARIA_SORT[state.dir] + '"' : "")
        + '><button type="button" class="vega-report-sort" data-sort-key="' + key + '"'
        + (active ? ' data-dir="' + state.dir + '"' : "")
        + ' title="' + escapeHtml(TOOLTIPS.columns[key]) + '">' + COLUMN_LABELS[key] + "</button></th>";
    }).join("") + "</tr>";
    return '<div class="vega-report-scroll"><table class="vega-report-table vega-report-perclass"><thead>' + head
      + "</thead><tbody>" + sorted.map(renderSpeciesRow).join("") + "</tbody></table></div>";
  }

  /**
   * The per-species section: heading plus the container the sort handler re-renders into.
   * @param {Array<Object>} rows
   * @param {{key: string, dir: string}} sort
   * @returns {string}
   */
  function renderPerClassSection(rows, sort) {
    return sectionHeading(SECTION_SPECIES) + '<div class="vega-report-species">' + renderPerClassTable(rows, sort) + "</div>";
  }

  /**
   * The figure strip: every figure with a same-origin url, in the fixed order, as an image
   * with a caption; nothing for a null url and no strip when nothing is left.
   * @param {*} figures the figures object from the route
   * @returns {string}
   */
  function renderFigures(figures) {
    if (!isObject(figures)) return "";
    const items = FIGURES.filter(f => isSafeUrl(figures[f.key]));
    if (!items.length) return "";
    return sectionHeading(SECTION_FIGURES) + '<div class="vega-report-figs">' + items.map(function (f) {
      const url = escapeHtml(figures[f.key]);
      return '<figure class="vega-report-fig"><a href="' + url + '" target="_blank" rel="noopener" title="' + escapeHtml(TOOLTIPS.figure) + '">'
        + '<img src="' + url + '" alt="' + f.caption + '" loading="lazy"></a><figcaption>' + f.caption + "</figcaption></figure>";
    }).join("") + "</div>";
  }

  /**
   * The prediction samples: one image per preview entry with a same-origin url, captioned
   * with the file name and the two counts.
   * @param {*} previews the previews array from the route
   * @returns {string}
   */
  function renderPreviews(previews) {
    if (!Array.isArray(previews)) return "";
    const items = previews.filter(p => isObject(p) && isSafeUrl(p.url));
    if (!items.length) return "";
    return sectionHeading(SECTION_PREVIEWS) + '<div class="vega-report-previews">' + items.map(function (p) {
      const file = escapeHtml(p.filename);
      return '<figure class="vega-report-preview"><img src="' + escapeHtml(p.url) + '" alt="' + file + '" loading="lazy">'
        + '<figcaption><span class="vega-report-file">' + file + "</span>: the human labeler drew "
        + plural(countOrZero(p.n_gt), "instance") + "; the model produced "
        + plural(countOrZero(p.n_pred), "detection") + "</figcaption></figure>";
    }).join("") + "</div>";
  }

  /**
   * The report text behind a closed disclosure; nothing when the text is missing or blank.
   * @param {*} md report_md from the route
   * @param {function(string): (string|null)} imageUrlFor resolver for markdown image sources
   * @returns {string}
   */
  function renderMarkdownSection(md, imageUrlFor) {
    if (typeof md !== "string" || md.trim() === "") return "";
    return '<details class="vega-report-md"><summary title="' + escapeHtml(TOOLTIPS.fullText) + '">' + FULL_TEXT_LABEL + "</summary>"
      + '<div class="vega-report-mdbody">' + markdownToHtml(md, imageUrlFor) + "</div></details>";
  }

  /**
   * The empty state: one plain sentence.
   * @returns {string}
   */
  function renderEmpty() {
    return '<div class="vega-report-empty">' + EMPTY_TEXT + "</div>";
  }

  /**
   * The loading line shown while the route answers.
   * @returns {string}
   */
  function renderLoading() {
    return '<div class="vega-report-loading">' + LOADING_TEXT + "</div>";
  }

  /**
   * The error state: the server's text, escaped.
   * @param {string} text
   * @returns {string}
   */
  function renderError(text) {
    return '<div class="vega-report-error">' + escapeHtml(text) + "</div>";
  }

  /**
   * The error field as text: a string as it is, anything else serialised.
   * @param {*} error
   * @returns {string}
   */
  function errorText(error) {
    if (typeof error === "string") return error;
    try { return JSON.stringify(error); } catch (err) { return String(error); }
  }

  /**
   * True when the payload carries a non-blank error field.
   * @param {*} data
   * @returns {boolean}
   */
  function hasError(data) {
    return isObject(data) && data.error !== undefined && data.error !== null && data.error !== "";
  }

  /**
   * True when the payload carries an error to show as the server's text: any error unless
   * the route also says ready false, which is the empty state (no evaluation yet).
   * @param {*} data
   * @returns {boolean}
   */
  function isErrorState(data) {
    return hasError(data) && data.ready !== false;
  }

  /**
   * True when there is no evaluation to show: no payload, a ready or evaluated flag set to
   * false, or a body with no metrics, no report text and no previews.
   * @param {*} data
   * @returns {boolean}
   */
  function isEmptyState(data) {
    if (!isObject(data)) return true;
    if (data.ready === false || data.evaluated === false) return true;
    const hasMetrics = isObject(data.metrics);
    const hasText = typeof data.report_md === "string" && data.report_md.trim() !== "";
    const hasPreviews = Array.isArray(data.previews) && data.previews.length > 0;
    return !hasMetrics && !hasText && !hasPreviews;
  }

  /**
   * Everything render and the sort handler share: whether metrics exist, the family, the
   * rows and the image resolver for the report text. The two name sources are merged, not
   * chosen between: the orchestrator passes the label registry's coral codes on every open
   * of the Evaluate panel, and a staged dataset's fish tokens are deliberately not in that
   * table, so a fish class takes its name from metrics.class_full_names (class id to full
   * name) while a coral class keeps the name the host supplied.
   * @param {Object} data the route payload
   * @param {Object} opts render options (labelNames, imageUrlFor)
   * @returns {{hasMetrics: boolean, family: string, rows: Array<Object>, imageUrlFor: function}}
   */
  function prepare(data, opts) {
    const hasMetrics = isObject(data.metrics);
    const family = metricFamily(data.metrics);
    const fromMetrics = hasMetrics && isObject(data.metrics.class_full_names) ? data.metrics.class_full_names : null;
    const hostNames = isObject(opts.labelNames) ? opts.labelNames : null;
    // The merged map is built on a null prototype: a class id or a code written as
    // __proto__ then becomes an ordinary key instead of reaching Object.prototype's setter.
    const labelNames = hostNames && fromMetrics
      ? Object.assign(Object.create(null), fromMetrics, hostNames)
      : (hostNames || fromMetrics);
    const rows = hasMetrics ? buildRows(data.metrics, family, labelNames) : [];
    const runDir = typeof data.run_dir === "string" ? data.run_dir : "";
    const imageUrlFor = typeof opts.imageUrlFor === "function" ? opts.imageUrlFor : (src => previewImageUrl(runDir, src));
    return { hasMetrics: hasMetrics, family: family, rows: rows, imageUrlFor: imageUrlFor };
  }

  /**
   * Render the whole card body for one route payload. Pure: data in, HTML string out.
   * @param {*} data the parsed JSON from GET /api/step/7/report (or an error body)
   * @param {{sort?: Object, labelNames?: Object, imageUrlFor?: function}} [options]
   * @returns {string}
   */
  function render(data, options) {
    const opts = isObject(options) ? options : {};
    if (isErrorState(data)) return renderError(errorText(data.error));
    if (isEmptyState(data)) return renderEmpty();
    const prepared = prepare(data, opts);
    const parts = [renderHeading(data)];
    if (prepared.hasMetrics) {
      parts.push(renderTopLine(data.metrics, prepared.family));
      parts.push(renderPerClassSection(prepared.rows, opts.sort));
    }
    parts.push(renderFigures(data.figures));
    parts.push(renderPreviews(data.previews));
    parts.push(renderMarkdownSection(data.report_md, prepared.imageUrlFor));
    return parts.filter(Boolean).join("\n");
  }

  // ── markdown ───────────────────────────────────────────────────────

  /**
   * Park a finished HTML fragment under a token so the later inline rules cannot touch it.
   * @param {Array<string>} parked
   * @param {string} html
   * @returns {string} the token
   */
  function park(parked, html) {
    parked.push(html);
    return CODE_TOKEN + (parked.length - 1) + CODE_TOKEN;
  }

  /**
   * Replace every code span in escaped text with a parked code element.
   * @param {string} text
   * @param {Array<string>} parked
   * @returns {string}
   */
  function parkCodeSpans(text, parked) {
    return text.replace(/`([^`]+)`/g, (m, code) => park(parked, "<code>" + code + "</code>"));
  }

  /**
   * Put every parked fragment back in place of its token.
   * @param {string} text
   * @param {Array<string>} parked
   * @returns {string}
   */
  function restoreParked(text, parked) {
    return text.replace(CODE_TOKEN_RE, (m, i) => parked[Number(i)]);
  }

  /**
   * The inline rules after code spans are parked: images (a finished tag is parked too, so
   * emphasis markers in its alt or source never corrupt it; a dropped image leaves its alt
   * text), links as text, bold, italic.
   * @param {string} text escaped text with code spans parked
   * @param {function(string): (string|null)} imageUrlFor
   * @param {Array<string>} parked
   * @returns {string}
   */
  function applyInlineRules(text, imageUrlFor, parked) {
    let s = text.replace(IMAGE_RE, function (m, alt, src) {
      const plainAlt = alt.replace(CODE_TOKEN_RE, (t, i) => parked[Number(i)].replace(/<\/?code>/g, ""));
      const url = imageUrlFor(unescapeHtml(src.trim()));
      if (isSafeUrl(url)) {
        return park(parked, '<img class="vega-report-mdimg" src="' + escapeHtml(url) + '" alt="' + plainAlt + '" loading="lazy">');
      }
      return plainAlt;
    });
    s = s.replace(LINK_RE, "$1");
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
    s = s.replace(/(^|[^\w])_([^_]+)_(?=[^\w]|$)/g, "$1<em>$2</em>");
    return s;
  }

  /**
   * Inline rules over one escaped block of text.
   * @param {string} text
   * @param {function(string): (string|null)} imageUrlFor
   * @returns {string}
   */
  function renderInline(text, imageUrlFor) {
    const parked = [];
    return restoreParked(applyInlineRules(parkCodeSpans(text, parked), imageUrlFor, parked), parked);
  }

  /**
   * Split one escaped pipe-table line into trimmed cells. A pipe the report escaped as \|
   * belongs to the cell text (evaluate_run escapes one inside a species name), so the line
   * is split on the unescaped pipes only and the escape is dropped from each cell.
   * @param {string} line starts and ends with a pipe
   * @returns {Array<string>}
   */
  function splitPipeRow(line) {
    return line.slice(1, -1).split(UNESCAPED_PIPE_RE).map(c => c.replace(ESCAPED_PIPE_RE, "|").trim());
  }

  /**
   * A pipe table: code spans are parked before a line is split, so a pipe inside inline
   * code never splits a cell; the first line is the header when a separator line follows
   * it; a row with more cells than the header folds the surplus into its last column (the
   * report writes an unescaped pipe inside its "Good range" cells), a shorter row is padded.
   * @param {Array<string>} lines escaped table lines
   * @param {function(string): (string|null)} imageUrlFor
   * @returns {string}
   */
  function renderTable(lines, imageUrlFor) {
    const parsed = lines.map(function (line) {
      const parked = [];
      return { cells: splitPipeRow(parkCodeSpans(line, parked)), parked: parked };
    });
    const hasHeader = parsed.length >= 2 && parsed[1].cells.every(c => TABLE_SEPARATOR_CELL_RE.test(c));
    const header = hasHeader ? parsed[0] : null;
    const body = hasHeader ? parsed.slice(2) : parsed;
    const width = header ? header.cells.length : Math.max.apply(null, parsed.map(r => r.cells.length));
    /** Fit a row to the header width: fold surplus cells into the last one, pad short rows. */
    function fit(cells) {
      if (cells.length > width) return cells.slice(0, width - 1).concat([cells.slice(width - 1).join(" | ")]);
      return cells.concat(new Array(width - cells.length).fill(""));
    }
    /** One table row with the given cell tag. */
    function rowHtml(row, tag) {
      return "<tr>" + fit(row.cells).map(function (cell) {
        return "<" + tag + ">" + restoreParked(applyInlineRules(cell, imageUrlFor, row.parked), row.parked) + "</" + tag + ">";
      }).join("") + "</tr>";
    }
    return "<table>"
      + (header ? "<thead>" + rowHtml(header, "th") + "</thead>" : "")
      + "<tbody>" + body.map(r => rowHtml(r, "td")).join("") + "</tbody></table>";
  }

  /**
   * A small safe markdown renderer. The whole text is escaped first, then block rules run:
   * headings (# to h4, ## to h5, ### and deeper to h6, under the card's h3), paragraphs,
   * bullet lists with indented continuation lines, pipe tables, horizontal rules; inline
   * rules run inside each block. Links render as text; an image is kept only when
   * imageUrlFor returns a same-origin url for its source, otherwise its alt text is shown.
   * @param {string} md the report markdown
   * @param {function(string): (string|null)} imageUrlFor maps a decoded image source to a url or null
   * @returns {string} HTML
   */
  function markdownToHtml(md, imageUrlFor) {
    if (typeof md !== "string") throw new TypeError("markdownToHtml: md must be a string");
    if (typeof imageUrlFor !== "function") throw new TypeError("markdownToHtml: imageUrlFor must be a function");
    const lines = escapeHtml(md).replace(NUL_RE, "").replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let paragraph = [];
    let listItems = null;
    let tableLines = null;
    /** Close an open paragraph. */
    function flushParagraph() {
      if (paragraph.length) { out.push("<p>" + renderInline(paragraph.join(" "), imageUrlFor) + "</p>"); paragraph = []; }
    }
    /** Close an open list. */
    function flushList() {
      if (listItems) { out.push("<ul>" + listItems.map(item => "<li>" + renderInline(item, imageUrlFor) + "</li>").join("") + "</ul>"); listItems = null; }
    }
    /** Close an open table. */
    function flushTable() {
      if (tableLines) { out.push(renderTable(tableLines, imageUrlFor)); tableLines = null; }
    }
    /** Close every open block. */
    function flushAll() { flushParagraph(); flushList(); flushTable(); }
    lines.forEach(function (raw) {
      const line = raw.replace(/\s+$/, "");
      const trimmed = line.trim();
      if (trimmed === "") { flushAll(); return; }
      if (trimmed.startsWith("|") && trimmed.endsWith("|") && trimmed.length > 1) {
        flushParagraph(); flushList();
        (tableLines = tableLines || []).push(trimmed);
        return;
      }
      flushTable();
      const heading = HEADING_RE.exec(trimmed);
      if (heading) {
        flushAll();
        const level = Math.min(heading[1].length, HEADING_LEVEL_MAX);
        const tag = "h" + (level + HEADING_TAG_OFFSET);
        out.push("<" + tag + ">" + renderInline(heading[2], imageUrlFor) + "</" + tag + ">");
        return;
      }
      if (RULE_RE.test(trimmed)) { flushAll(); out.push("<hr>"); return; }
      const bullet = BULLET_RE.exec(trimmed);
      if (bullet) { flushParagraph(); (listItems = listItems || []).push(bullet[1]); return; }
      if (listItems && CONTINUATION_RE.test(line)) { listItems[listItems.length - 1] += " " + trimmed; return; }
      flushList();
      paragraph.push(trimmed);
    });
    flushAll();
    return out.join("\n");
  }

  // ── DOM glue ───────────────────────────────────────────────────────

  /**
   * The element the card body is written into: the card's .vega-report-body, or the root itself.
   * @param {Element} rootEl
   * @returns {Element}
   */
  function bodyElement(rootEl) {
    const inner = typeof rootEl.querySelector === "function" ? rootEl.querySelector(BODY_SELECTOR) : null;
    return inner || rootEl;
  }

  /**
   * Write the one-line status (loading, empty, error) and the card body. A root with a
   * status line shows the sentence there and keeps the body for a rendered card; a root
   * without one is filled directly with whichever is present.
   * @param {Element} rootEl
   * @param {string} statusHtml
   * @param {string} bodyHtml
   */
  function paint(rootEl, statusHtml, bodyHtml) {
    const status = typeof rootEl.querySelector === "function" ? rootEl.querySelector(STATUS_SELECTOR) : null;
    const body = bodyElement(rootEl);
    if (status && status !== body) {
      status.innerHTML = statusHtml;
      body.innerHTML = bodyHtml;
    } else {
      body.innerHTML = statusHtml + bodyHtml;
    }
  }

  /**
   * The browser's localStorage when reachable, otherwise null.
   * @returns {Storage|null}
   */
  function defaultStorage() {
    try { return root && root.localStorage ? root.localStorage : null; } catch (err) { return null; }
  }

  /**
   * Fetch a JSON route and return its body; a non-JSON or non-object reply, or a failing
   * status with no error text, becomes an error body naming the HTTP status.
   * @param {string} url
   * @returns {Promise<Object>}
   */
  async function defaultFetchJson(url) {
    if (!root || typeof root.fetch !== "function") throw new Error("fetch is not available in this environment");
    const resp = await root.fetch(url, { headers: { Accept: "application/json" } });
    let body;
    try { body = await resp.json(); } catch (err) { body = { error: "The evaluation report route returned a non-JSON reply (HTTP " + resp.status + ")." }; }
    if (!isObject(body)) body = { error: "The evaluation report route returned an unexpected reply (HTTP " + resp.status + ")." };
    if (!resp.ok && !hasError(body)) body.error = "The evaluation report route failed with HTTP " + resp.status + ".";
    return body;
  }

  /**
   * Attach the delegated click handler for the sort buttons once per body element.
   * @param {Element} body
   * @param {Storage|null} storage
   */
  function wireSorting(body, storage) {
    if (WIRED.has(body) || typeof body.addEventListener !== "function") return;
    WIRED.add(body);
    body.addEventListener("click", function (event) {
      const target = event && event.target;
      const button = target && typeof target.closest === "function" ? target.closest(SORT_BUTTON_SELECTOR) : null;
      if (!button) return;
      const state = STATE.get(body);
      if (!state) return;
      state.sort = nextSort(state.sort, button.getAttribute("data-sort-key"));
      writeSortState(storage, state.sort);
      const container = typeof body.querySelector === "function" ? body.querySelector(SPECIES_SELECTOR) : null;
      if (!container) return;
      container.innerHTML = renderPerClassTable(state.rows, state.sort);
      // The re-render destroyed the button that was pressed, so keyboard focus would fall
      // to the document; put it back on the same column in the fresh table.
      const again = typeof container.querySelector === "function"
        ? container.querySelector(SORT_BUTTON_SELECTOR + '[data-sort-key="' + state.sort.key + '"]') : null;
      if (again && typeof again.focus === "function") again.focus();
    });
  }

  /**
   * Load the evaluation for a run folder into the card: show the loading line, fetch the
   * report route through the injected fetch, render, remember the rows for sorting and wire
   * the sort buttons. Each call takes a ticket on the body's state, and a response that
   * arrives after a newer call took over is discarded. A rejected fetch becomes the error
   * state; nothing throws past here except invalid arguments.
   * @param {Element} rootEl the card (templates/_step7_report.html) or any element to fill
   * @param {string} runDir absolute run folder; blank asks the route for its default
   * @param {{fetchJson?: function, storage?: (Storage|null), labelNames?: Object, imageUrlFor?: function}} [options]
   * @returns {Promise<Object>} the payload that was fetched, rendered or not
   */
  async function load(rootEl, runDir, options) {
    if (!rootEl || typeof rootEl !== "object") throw new TypeError("VegaReport.load: rootEl must be an element");
    if (runDir !== undefined && runDir !== null && typeof runDir !== "string") throw new TypeError("VegaReport.load: runDir must be a string");
    const opts = isObject(options) ? options : {};
    const fetchJson = typeof opts.fetchJson === "function" ? opts.fetchJson : defaultFetchJson;
    const storage = opts.storage === undefined ? defaultStorage() : opts.storage;
    const body = bodyElement(rootEl);
    const state = STATE.get(body) || { rows: [], sort: normaliseSort(null), sequence: 0 };
    state.sequence += 1;
    const ticket = state.sequence;
    STATE.set(body, state);
    paint(rootEl, renderLoading(), "");
    let data;
    try {
      data = await fetchJson(reportUrl(runDir || ""));
    } catch (err) {
      data = { error: FETCH_FAILED_PREFIX + (err && err.message ? err.message : String(err)) };
    }
    if (state.sequence !== ticket) return data;
    const oneLine = isErrorState(data) || isEmptyState(data);
    state.sort = readSortState(storage);
    const html = render(data, { sort: state.sort, labelNames: opts.labelNames, imageUrlFor: opts.imageUrlFor });
    paint(rootEl, oneLine ? html : "", oneLine ? "" : html);
    state.rows = oneLine ? [] : prepare(data, opts).rows;
    wireSorting(body, storage);
    return data;
  }

  /**
   * The host page's entry point: load the run's evaluation into the card with id
   * s7-report-card. Throws when the fragment is not on the page.
   * @param {string} runDir absolute run folder; blank asks the route for its default
   * @param {Object} [options] passed through to load (labelNames, fetchJson)
   * @returns {Promise<Object>}
   */
  function loadStep7ReportCard(runDir, options) {
    if (!root || !root.document || typeof root.document.getElementById !== "function") {
      throw new Error("loadStep7ReportCard needs a browser document");
    }
    const card = root.document.getElementById(CARD_ID);
    if (!card) throw new Error("loadStep7ReportCard: no element with id " + CARD_ID + "; include templates/_step7_report.html in the panel");
    return load(card, typeof runDir === "string" ? runDir : "", options);
  }

  return {
    REPORT_URL: REPORT_URL,
    IMAGE_URL: IMAGE_URL,
    CARD_ID: CARD_ID,
    SORT_STORAGE_KEY: SORT_STORAGE_KEY,
    DEFAULT_SORT: DEFAULT_SORT,
    TOOLTIPS: TOOLTIPS,
    escapeHtml: escapeHtml,
    isSafeUrl: isSafeUrl,
    formatScore: formatScore,
    formatEvaluatedAt: formatEvaluatedAt,
    reportUrl: reportUrl,
    previewImageUrl: previewImageUrl,
    speciesFromClassName: speciesFromClassName,
    metricFamily: metricFamily,
    buildRows: buildRows,
    sortRows: sortRows,
    nextSort: nextSort,
    readSortState: readSortState,
    writeSortState: writeSortState,
    renderPerClassTable: renderPerClassTable,
    markdownToHtml: markdownToHtml,
    render: render,
    load: load,
    loadStep7ReportCard: loadStep7ReportCard,
  };
});

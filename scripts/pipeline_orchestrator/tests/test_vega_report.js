#!/usr/bin/env node
/* Pure-logic tests for the VEGA evaluation report card
   (scripts/pipeline_orchestrator/static/orchestrator_report.js). The module
   is an IIFE that also exports through module.exports, so it loads in Node 18
   with no browser. Run:
     /usr/bin/node scripts/pipeline_orchestrator/tests/test_vega_report.js

   Three layers, the house convention:
     smoke       - module loads, every documented export is present
     unit        - the nine behaviours named in the T11 brief
     adversarial - hostile strings, wrong types, huge input, a storage that
                   throws, a fetch that rejects, unsafe URLs, a run folder
                   with no evaluation. */
"use strict";
const path = require("path");
const R = require(path.join(__dirname, "..", "static", "orchestrator_report.js"));

let pass = 0, fail = 0;
function ok(cond, msg) { if (cond) { pass++; } else { fail++; console.error("  FAIL:", msg); } }
function eq(a, b, msg) {
  ok(JSON.stringify(a) === JSON.stringify(b), `${msg} (got ${JSON.stringify(a)}, want ${JSON.stringify(b)})`);
}
function has(html, needle, msg) { ok(String(html).includes(needle), `${msg}: expected ${JSON.stringify(needle)}`); }
function lacks(html, needle, msg) { ok(!String(html).includes(needle), `${msg}: must not contain ${JSON.stringify(needle)}`); }
function count(html, needle) { return String(html).split(needle).length - 1; }

// ── fixtures ─────────────────────────────────────────────────────────
const RUN_DIR = "/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/inprocess/_models/lab_a/runs/run_a";
const RUN_DIR_ENC = encodeURIComponent(RUN_DIR);
function imageUrl(rel) { return "/api/step/7/image?run_dir=" + RUN_DIR_ENC + "&rel=" + encodeURIComponent(rel); }

function detectorFixture() {
  return {
    ok: true,
    run_name: "run_a",
    run_dir: RUN_DIR,
    out_dir: RUN_DIR + "/eval",
    evaluated_at: "2026-09-03T20:49:29-04:00",
    report_md: [
      "# Evaluation report for run `run_a`",
      "",
      "A paragraph.",
      "",
      "- **Acropora-palmata_881.jpg** drew 1 instance; the model produced 2 detections.",
      "  ![Acropora-palmata_881.jpg](previews/Acropora-palmata_881_sbs.jpg)",
      "",
    ].join("\n"),
    metrics: {
      overall: {
        box_map50: 0.2088, box_map50_95: 0.1769, box_precision: 0.3346, box_recall: 0.2512,
        mask_map50: null, mask_map50_95: null, mask_precision: null, mask_recall: null,
      },
      per_class: {
        "Acropora-palmata": { box_precision: 0.7727, box_recall: 0.5862, box_map50: 0.5459, box_map50_95: 0.4382 },
        "Acropora-cervicornis": { box_precision: 0.6053, box_recall: 0.4423, box_map50: 0.4083, box_map50_95: 0.2606 },
        "Porites-astreoides": { box_precision: 0.5, box_recall: 0.5, box_map50: 0.4083, box_map50_95: 0.3 },
      },
      per_class_instances: { "Acropora-palmata": 35, "Acropora-cervicornis": 25, "Porites-astreoides": 25, "Agaricia-undata": 0 },
      absent_classes: ["Agaricia-undata"],
      metric_family: "box",
      class_names: { "0": "Acropora-cervicornis", "1": "Acropora-palmata", "2": "Porites-astreoides", "3": "Agaricia-undata" },
    },
    previews: [
      { filename: "Acropora-palmata_881.jpg", url: imageUrl("previews/Acropora-palmata_881_sbs.jpg"), n_gt: 1, n_pred: 2 },
    ],
    figures: {
      box_pr: imageUrl("val_run/BoxPR_curve.png"),
      box_f1: imageUrl("val_run/BoxF1_curve.png"),
      mask_pr: null,
      mask_f1: null,
      confusion_norm: imageUrl("val_run/confusion_matrix_normalized.png"),
      confusion_abs: imageUrl("val_run/confusion_matrix.png"),
      results: null,
    },
    pdf_url: "/api/step/7/pdf?run_dir=" + RUN_DIR_ENC,
  };
}

function segmenterFixture() {
  const d = detectorFixture();
  d.run_name = "run_seg";
  d.metrics = {
    overall: {
      box_map50: 0.671, box_map50_95: 0.569, box_precision: 0.664, box_recall: 0.619,
      mask_map50: 0.671, mask_map50_95: 0.545, mask_precision: 0.664, mask_recall: 0.619,
    },
    per_class: {
      PA: { box_precision: 0.664, box_recall: 0.619, box_map50: 0.671, box_map50_95: 0.569,
            mask_precision: 0.664, mask_recall: 0.619, mask_map50: 0.671, mask_map50_95: 0.545 },
      None: { box_precision: 0.1, box_recall: 0.1, box_map50: 0.1, box_map50_95: 0.1,
              mask_precision: 0.1, mask_recall: 0.1, mask_map50: 0.1, mask_map50_95: 0.1 },
    },
    per_class_instances: { PA: 120, None: 3 },
    absent_classes: [],
    metric_family: "mask",
    class_names: { "0": "PA", "1": "None" },
  };
  d.figures.mask_pr = imageUrl("val_run/MaskPR_curve.png");
  d.figures.mask_f1 = imageUrl("val_run/MaskF1_curve.png");
  d.figures.results = imageUrl("results.png");
  d.pdf_url = null;
  return d;
}

/* A detector over fish classes: tokens no species code table and no binomial rule
   resolves, with the full names the staged dataset supplied carried in metrics
   under class_full_names, keyed by class id exactly as metrics.json writes them. */
const FISH_TOKEN = "EPST_Dark";
const FISH_NAME = "Epinephelus striatus, Nassau grouper, dark phase";
const FISH_TOKEN_2 = "KYSE_Normal";
const FISH_NAME_2 = "Kyphosus sectatrix, Bermuda chub";
// How the card must render FISH_NAME: the binomial italic, the qualifiers upright in the
// same body face, never inheriting the table's monospace.
const FISH_NAME_HTML = '<em class="vega-report-binomial">Epinephelus striatus</em>'
  + '<span class="vega-report-plain">, Nassau grouper, dark phase</span>';
// The label vocabulary orchestrator.js fetches once and passes on every open of the
// Evaluate panel: coral codes from the label registry, which name no fish token.
const CORAL_VOCABULARY = { AAGA: "Agaricia agaricites", PAST: "Porites astreoides", SSID: "Siderastrea siderea" };
function fishFixture() {
  const d = detectorFixture();
  d.run_name = "run_fish";
  d.metrics = {
    overall: {
      box_map50: 0.904, box_map50_95: 0.71, box_precision: 0.88, box_recall: 0.85,
      mask_map50: null, mask_map50_95: null, mask_precision: null, mask_recall: null,
    },
    per_class: {
      [FISH_TOKEN]: { box_precision: 0.88, box_recall: 0.85, box_map50: 0.904, box_map50_95: 0.71 },
      [FISH_TOKEN_2]: { box_precision: 0.5, box_recall: 0.4, box_map50: 0.45, box_map50_95: 0.3 },
    },
    per_class_instances: { [FISH_TOKEN]: 120, [FISH_TOKEN_2]: 40 },
    absent_classes: [],
    metric_family: "box",
    class_names: { "0": FISH_TOKEN, "1": FISH_TOKEN_2 },
    class_full_names: { "0": FISH_NAME, "1": FISH_NAME_2 },
  };
  return d;
}

function fakeCard() {
  /* The species container doubles as a tiny DOM: re-rendering replaces innerHTML, and
     querySelector finds a sort button by its data-sort-key so the focus restore is visible. */
  const species = {
    innerHTML: "",
    focused: [],
    querySelector(sel) {
      const m = /^\.vega-report-sort\[data-sort-key="([^"]+)"\]$/.exec(sel);
      if (!m || !this.innerHTML.includes('data-sort-key="' + m[1] + '"')) return null;
      const holder = this;
      return { key: m[1], focus() { holder.focused.push(this.key); } };
    },
  };
  const status = { innerHTML: "" };
  const body = {
    innerHTML: "",
    handlers: {},
    addEventListener(type, fn) { this.handlers[type] = fn; },
    querySelector(sel) { return sel === ".vega-report-species" ? species : null; },
  };
  return { body, species, status, querySelector(sel) {
    if (sel === ".vega-report-body") return body;
    if (sel === ".vega-report-status") return status;
    return null;
  } };
}
function fakeStorage(initial) {
  const store = Object.assign({}, initial || {});
  return {
    store,
    getItem(k) { return Object.prototype.hasOwnProperty.call(store, k) ? store[k] : null; },
    setItem(k, v) { store[k] = String(v); },
  };
}
function throwingStorage() {
  return { getItem() { throw new Error("storage blocked"); }, setItem() { throw new Error("storage blocked"); } };
}
function clickOn(key) {
  return { target: { closest(sel) {
    return sel === ".vega-report-sort" ? { getAttribute(n) { return n === "data-sort-key" ? key : null; } } : null;
  } } };
}
/* Strip every quoted attribute value so an escaped payload inside alt="..."
   cannot masquerade as an attribute of its own. */
function withoutAttributeValues(html) { return String(html).replace(/="[^"]*"/g, '=""'); }
/* Only the tags, with their attribute values stripped: escaped text content never counts. */
function tagsOnly(html) { return (withoutAttributeValues(html).match(/<[^>]*>/g) || []).join(""); }

// ── SMOKE ────────────────────────────────────────────────────────────
["render", "markdownToHtml", "load", "loadStep7ReportCard", "escapeHtml", "previewImageUrl",
 "reportUrl", "speciesFromClassName", "buildRows", "sortRows", "nextSort", "readSortState",
 "writeSortState", "formatEvaluatedAt", "formatScore", "metricFamily", "isSafeUrl",
 "renderPerClassTable"]
  .forEach(function (n) { ok(typeof R[n] === "function", `export ${n} is a function`); });
eq(R.DEFAULT_SORT, { key: "map50", dir: "desc" }, "smoke: default sort is mAP50 descending");
ok(R.SORT_STORAGE_KEY === "vega.report.sort", "smoke: storage key constant");
ok(R.REPORT_URL === "/api/step/7/report" && R.IMAGE_URL === "/api/step/7/image", "smoke: route constants");
ok(R.TOOLTIPS && typeof R.TOOLTIPS.columns === "object", "smoke: tooltip strings are exported for the prose test");

// ── UNIT: test_markdown_escapes_html_before_rendering ────────────────
(function test_markdown_escapes_html_before_rendering() {
  const md = [
    "# Title <script>alert(1)</script>",
    "",
    "Para <b>bold</b> & more",
    "",
    '![x" onerror="alert(1)](previews/a.jpg)',
    "",
    "| H<i> | v |",
    "|---|---|",
    "| <u>c</u> | 1 |",
  ].join("\n");
  const html = R.markdownToHtml(md, src => R.previewImageUrl(RUN_DIR, src));
  lacks(html, "<script", "escape: no script tag survives");
  has(html, "&lt;script&gt;alert(1)&lt;/script&gt;", "escape: script shown as text");
  lacks(html, "<b>", "escape: raw bold tag does not survive");
  has(html, "&lt;b&gt;bold&lt;/b&gt; &amp; more", "escape: ampersand and tags escaped in paragraph");
  lacks(html, "<u>", "escape: raw tag inside a table cell does not survive");
  lacks(html, "<i>", "escape: raw tag inside a table header does not survive");
  const want = '<img class="vega-report-mdimg" src="' + R.escapeHtml(R.previewImageUrl(RUN_DIR, "previews/a.jpg")) +
    '" alt="x&quot; onerror=&quot;alert(1)" loading="lazy">';
  has(html, want, "escape: image alt with a quote and onerror is emitted as one escaped attribute");
  lacks(tagsOnly(html), "onerror", "escape: onerror never becomes an attribute");
  ok(/^[^<]*<h4/.test(html), "escape: output starts with the heading, no stray markup before it");
})();

// ── UNIT: test_markdown_renders_tables_lists_headings ────────────────
(function test_markdown_renders_tables_lists_headings() {
  const md = [
    "# H1 here",
    "## H2 here",
    "### H3 here",
    "#### H4 here",
    "",
    "Para one",
    "continues here.",
    "",
    "- item **bold** and *ital* and `code_with_underscores`",
    "- item two",
    "  indented continuation",
    "* star item",
    "",
    "| Metric | Box | Good range |",
    "|---|---|---|",
    "| mAP50 | 0.209 | <0.35 poor | 0.35-0.55 weak |",
    "| mAP50-95 | 0.177 | <0.2 poor |",
    "| *Genus species*, a \\| b (`GS`) | 4 | fine |",
    "",
    "---",
    "",
    "_Weights_: `x` and snake_case_name stays and epochs_requested=20 · epochs_completed=20",
  ].join("\n");
  const html = R.markdownToHtml(md, () => null);
  has(html, '<h4>H1 here</h4>', "headings: # becomes h4 (card is an h3)");
  has(html, '<h5>H2 here</h5>', "headings: ## becomes h5");
  has(html, '<h6>H3 here</h6>', "headings: ### becomes h6");
  has(html, '<h6>H4 here</h6>', "headings: deeper levels clamp to h6");
  has(html, "<p>Para one continues here.</p>", "paragraph: consecutive lines join into one paragraph");
  ok(count(html, "<ul>") === 1, "list: one ul for the whole run of bullets");
  ok(count(html, "<li>") === 3, "list: three items (dash, dash, star)");
  has(html, "<strong>bold</strong>", "inline: bold");
  has(html, "<em>ital</em>", "inline: italic with asterisks");
  has(html, "<code>code_with_underscores</code>", "inline: code kept verbatim, underscores untouched");
  has(html, "<li>item two indented continuation</li>", "list: an indented line continues the previous item");
  ok(count(html, "<table") === 1, "table: one table");
  has(html, "<th>Metric</th><th>Box</th><th>Good range</th>", "table: header row from the first line");
  has(html, "<td>mAP50</td><td>0.209</td><td>&lt;0.35 poor | 0.35-0.55 weak</td>", "table: surplus cells fold into the last column");
  has(html, "<td>mAP50-95</td><td>0.177</td><td>&lt;0.2 poor</td>", "table: a row with the right cell count is untouched");
  has(html, "<td><em>Genus species</em>, a | b (<code>GS</code>)</td><td>4</td><td>fine</td>",
    "table: a vertical bar the report escaped stays inside its cell and the row keeps its columns");
  lacks(html, "|---|", "table: separator row is not rendered");
  has(html, "<hr>", "rule: --- becomes hr");
  has(html, "<em>Weights</em>", "inline: underscore italic at a word boundary");
  has(html, "snake_case_name", "inline: underscores inside a word are left alone");
  has(html, "epochs_requested=20 · epochs_completed=20", "inline: underscores in key=value pairs are left alone");
  lacks(html, "<em>requested", "inline: no false italic across words");
  eq(R.markdownToHtml("", () => null), "", "empty markdown renders to an empty string");
  eq(R.markdownToHtml("\n\n  \n", () => null), "", "blank-only markdown renders to an empty string");
})();

// ── UNIT: test_image_rule_rewrites_previews_and_drops_others ─────────
(function test_image_rule_rewrites_previews_and_drops_others() {
  const md = [
    "![ok](previews/a b.jpg)",
    "",
    "![amp](previews/a&b.jpg)",
    "",
    "![up](previews/../secret.png)",
    "",
    "![abs](/etc/passwd)",
    "",
    "![http](https://evil.example/x.png)",
    "",
    "![nested](previews/sub/x.jpg)",
    "",
    "![dot](previews/..)",
    "",
    "![bare](x.jpg)",
    "",
    "[a link](https://example.com) and ![](previews/noalt.jpg)",
  ].join("\n");
  const seen = [];
  const html = R.markdownToHtml(md, src => { seen.push(src); return R.previewImageUrl(RUN_DIR, src); });
  eq(count(html, "<img"), 3, "images: only the three previews/ sources become img tags");
  has(html, "rel=previews%2Fa%20b.jpg", "images: a space in the filename is percent-encoded");
  has(html, "rel=previews%2Fa%26b.jpg", "images: an ampersand in the filename is percent-encoded");
  ok(seen.includes("previews/a&b.jpg"), "images: the resolver receives the decoded source");
  has(html, "&amp;rel=", "images: the ampersand between query parameters is escaped in the attribute");
  has(html, "run_dir=" + R.escapeHtml(RUN_DIR_ENC), "images: the run folder rides in the query string");
  ["up", "abs", "http", "nested", "dot", "bare"].forEach(function (alt) {
    has(html, "<p>" + alt + "</p>", "images: dropped image shows its alt text as plain text: " + alt);
  });
  lacks(html, "secret.png", "images: a parent-directory source is dropped");
  lacks(html, "evil.example", "images: an external source is dropped");
  lacks(html, "/etc/passwd", "images: an absolute source is dropped");
  lacks(html, "<a ", "links: rendered as text, never anchors");
  has(html, "a link", "links: the link text survives");
  lacks(html, "example.com", "links: the target is dropped");
  eq(R.previewImageUrl(RUN_DIR, "previews/x.jpg"), imageUrl("previews/x.jpg"), "previewImageUrl: exact route shape");
  eq(R.previewImageUrl(RUN_DIR, "previews/../x.jpg"), null, "previewImageUrl: parent segment refused");
  eq(R.previewImageUrl(RUN_DIR, "previews\\x.jpg"), null, "previewImageUrl: backslash refused");
  eq(R.previewImageUrl(RUN_DIR, "PREVIEWS/x.jpg"), null, "previewImageUrl: folder name is case-sensitive");
  eq(R.previewImageUrl("", "previews/x.jpg"), null, "previewImageUrl: no run folder, no url");
  eq(R.previewImageUrl(RUN_DIR, 42), null, "previewImageUrl: non-string source refused");
  const emphasis = R.markdownToHtml("![*a*](previews/*a*.jpg) and *b* and ![_u_](previews/_u_.jpg)", src => R.previewImageUrl(RUN_DIR, src));
  has(emphasis, '<img class="vega-report-mdimg" src="' + R.escapeHtml(R.previewImageUrl(RUN_DIR, "previews/*a*.jpg")) + '" alt="*a*" loading="lazy">', "images: asterisks in alt and source never become emphasis inside the tag");
  has(emphasis, '<img class="vega-report-mdimg" src="' + R.escapeHtml(R.previewImageUrl(RUN_DIR, "previews/_u_.jpg")) + '" alt="_u_" loading="lazy">', "images: underscores in alt and source never become emphasis inside the tag");
  has(emphasis, "<em>b</em>", "images: emphasis outside the tag still applies");
  eq(count(emphasis, "<em>"), 1, "images: exactly one em, outside the tags");
  const codeAlt = R.markdownToHtml("![see `x`](previews/c.jpg)", src => R.previewImageUrl(RUN_DIR, src));
  has(codeAlt, 'alt="see x"', "images: a code span in the alt text is kept as plain text");
  const resolverRejects = R.markdownToHtml("![x](previews/a.jpg)", () => "javascript:alert(1)");
  lacks(resolverRejects, "<img", "images: a resolver returning an unsafe url yields no img");
  has(resolverRejects, "<p>x</p>", "images: unsafe resolver result falls back to the alt text");
})();

// ── UNIT: test_per_class_table_shows_full_name_and_chip_and_absent_rows
(function test_per_class_table_shows_full_name_and_chip_and_absent_rows() {
  const html = R.render(detectorFixture(), {});
  has(html, '<em class="vega-report-binomial">Acropora palmata</em>', "species: binomial rendered in full, italic");
  lacks(html, '<span class="vega-report-chip">Acropora-palmata</span>', "species: a class name that is itself a binomial gets no chip");
  has(html, '<td><em class="vega-report-binomial">Acropora palmata</em></td>', "species: the binomial stands alone in its cell");
  lacks(html, 'vega-report-chip', "species: no chip anywhere when no code resolved through a name map");
  has(html, "not present in this split", "absent: the absent species carries the sentence instead of numbers");
  has(html, '<tr class="vega-report-absent">', "absent: row is marked");
  has(html, '<em class="vega-report-binomial">Agaricia undata</em>', "absent: absent species still shows its full name");
  const rowOrder = ["Acropora palmata", "Acropora cervicornis", "Porites astreoides", "Agaricia undata"]
    .map(n => html.indexOf(n));
  ok(rowOrder.every((p, i) => p >= 0 && (i === 0 || p > rowOrder[i - 1])),
    "default order: mAP50 descending, ties in input order, absent last: " + JSON.stringify(rowOrder));
  has(html, '<td class="num">35</td>', "instances: count shown");
  has(html, '<td class="num">0.773</td>', "precision: three decimals");
  has(html, '<td class="num">0.586</td>', "recall: three decimals");
  has(html, '<td class="num">0.546</td>', "mAP50: three decimals");
  has(html, '<td class="num">0.438</td>', "mAP50-95: three decimals");
  lacks(html, "N/A", "no N/A anywhere in the card");
  ok(count(html, 'class="vega-report-sort"') === 6, "six sortable column headers");
  has(html, 'aria-sort="descending"', "the active column carries aria-sort");
  eq(count(html, "aria-sort="), 1, "only the active column carries aria-sort");
  Object.keys(R.TOOLTIPS.columns).forEach(function (k) {
    has(html, 'title="' + R.escapeHtml(R.TOOLTIPS.columns[k]) + '"', "column header tooltip present: " + k);
  });

  // Segmenter with a code-to-name map: PA resolves through the map, None stays as it is.
  const seg = R.render(segmenterFixture(), { labelNames: { PA: "Porites astreoides" } });
  has(seg, '<em class="vega-report-binomial">Porites astreoides</em> <span class="vega-report-chip">PA</span>', "species: code resolved through the injected name map, chip beside it");
  has(seg, '<span class="vega-report-plain">None</span>', "species: an unresolvable class stays as plain text");
  lacks(seg, '<span class="vega-report-chip">None</span>', "species: an unresolvable class gets no chip");
  const segNoMap = R.render(segmenterFixture(), {});
  has(segNoMap, '<span class="vega-report-plain">PA</span>', "species: without a map a code stays as it is (never a chip alone)");
  has(segNoMap, '<td class="num">0.545</td>', "segmenter per-class numbers come from the mask family");

  // Pure helpers.
  eq(R.speciesFromClassName("Acropora-palmata", null), { name: "Acropora palmata", code: null, resolved: true }, "speciesFromClassName: hyphenated binomial");
  eq(R.speciesFromClassName("Madracis-decactis-x", null), { name: "Madracis decactis x", code: null, resolved: true }, "speciesFromClassName: three-part name");
  eq(R.speciesFromClassName("PA", { PA: "Porites astreoides" }), { name: "Porites astreoides", code: "PA", resolved: true }, "speciesFromClassName: map wins");
  eq(R.speciesFromClassName("PA", {}), { name: "PA", code: null, resolved: false }, "speciesFromClassName: unmapped code stays");
  eq(R.speciesFromClassName("None", null), { name: "None", code: null, resolved: false }, "speciesFromClassName: None stays");
  eq(R.speciesFromClassName("", null), { name: "", code: null, resolved: false }, "speciesFromClassName: empty");
  eq(R.speciesFromClassName(7, null), { name: "7", code: null, resolved: false }, "speciesFromClassName: number coerced");
  const rows = R.buildRows(detectorFixture().metrics, "box", null);
  eq(rows.map(r => r.className), ["Acropora-palmata", "Acropora-cervicornis", "Porites-astreoides", "Agaricia-undata"], "buildRows: per_class order, then absent names");
  eq(rows[3].absent, true, "buildRows: absent flag");
  eq(rows[3].instances, 0, "buildRows: absent instances from the count map");
  eq(rows[0].map50, 0.5459, "buildRows: box family reads box_map50");
  // Unscored (instances, no scores) is not absent: an older metrics file dropped species.
  const older = detectorFixture().metrics;
  older.per_class_instances["Siderastrea-stellata"] = 4;
  older.per_class_instances["Meandrina-danae"] = 0;
  older.absent_classes = [];
  const olderRows = R.buildRows(older, "box", null);
  const byName = Object.fromEntries(olderRows.map(r => [r.className, r]));
  eq([byName["Siderastrea-stellata"].absent, byName["Siderastrea-stellata"].unscored], [false, true], "buildRows: instances without scores is unscored");
  eq([byName["Meandrina-danae"].absent, byName["Meandrina-danae"].unscored], [true, false], "buildRows: zero instances without scores is absent");
  eq([byName["Agaricia-undata"].absent, byName["Agaricia-undata"].unscored], [true, false], "buildRows: zero instances stays absent");
  const olderHtml = R.renderPerClassTable(olderRows, R.DEFAULT_SORT);
  has(olderHtml, '<tr class="vega-report-unscored"><td><em class="vega-report-binomial">Siderastrea stellata</em></td><td class="num">4</td><td colspan="4" class="vega-report-unscored-text">no scores recorded</td></tr>', "unscored row: count shown, no scores recorded");
  ok(olderHtml.indexOf("Siderastrea stellata") < olderHtml.indexOf("Meandrina danae"), "unscored rows sort before absent rows");
  ok(olderHtml.indexOf("Porites astreoides") < olderHtml.indexOf("Siderastrea stellata"), "scored rows sort before unscored rows");
  const declared = detectorFixture().metrics;
  declared.absent_classes = ["Acropora-palmata"];
  const declaredRow = R.buildRows(declared, "box", null).find(r => r.className === "Acropora-palmata");
  eq([declaredRow.absent, declaredRow.map50], [true, null], "buildRows: a declared absent class shows no numbers even when scores exist");
})();

// ── UNIT: test_family_aware_top_line ─────────────────────────────────

(function test_class_full_names_name_the_species_when_the_host_passes_no_map() {
  const html = R.render(fishFixture(), {});
  has(html, FISH_NAME_HTML + ' <span class="vega-report-chip">' + FISH_TOKEN + "</span>",
    "class_full_names: the species is italic, its qualifiers are upright body text, and the token is the chip");
  has(html, '<em class="vega-report-binomial">Kyphosus sectatrix</em>'
    + '<span class="vega-report-plain">, Bermuda chub</span>',
    "class_full_names: the second fish class is named the same way");
  has(R.render(detectorFixture(), {}), '<em class="vega-report-binomial">Acropora palmata</em></td>',
    "class_full_names: a name that is a binomial on its own is still italic in full");
  lacks(html, '<span class="vega-report-plain">' + FISH_TOKEN + "</span>",
    "class_full_names: the bare token is never shown alone once the dataset names it");

  // The host's map wins for a token it names; the metrics names fill in every other class.
  const hosted = R.render(fishFixture(), { labelNames: { [FISH_TOKEN]: "A name the page supplied" } });
  has(hosted, "A name the page supplied", "labelNames: the host's map wins for a token it names");
  lacks(hosted, "Nassau grouper", "labelNames: the metrics name is dropped for a token the host names");
  has(hosted, '<em class="vega-report-binomial">Kyphosus sectatrix</em>',
    "labelNames: a class the host does not name still takes its name from the metrics file");

  // A metrics file without the key reads exactly as it did before it existed.
  const older = fishFixture();
  delete older.metrics.class_full_names;
  const olderHtml = R.render(older, {});
  has(olderHtml, '<span class="vega-report-plain">' + FISH_TOKEN + "</span>",
    "class_full_names: without the key a token that resolves to nothing stays as it is");
  lacks(olderHtml, "vega-report-chip", "class_full_names: no chip without a name to put beside it");

  // Pure helpers: a lookup keyed by class id resolves a token the map does not carry.
  eq(R.speciesFromClassName(FISH_TOKEN, { 3: "Nassau grouper" }, 3),
    { name: "Nassau grouper", code: FISH_TOKEN, resolved: true }, "speciesFromClassName: class id key resolves");
  eq(R.speciesFromClassName(FISH_TOKEN, { 3: "Nassau grouper" }, "3"),
    { name: "Nassau grouper", code: FISH_TOKEN, resolved: true }, "speciesFromClassName: the class id may arrive as text");
  eq(R.speciesFromClassName(FISH_TOKEN, { 3: "Nassau grouper" }, 4),
    { name: FISH_TOKEN, code: null, resolved: false }, "speciesFromClassName: another class id resolves nothing");
  eq(R.speciesFromClassName(FISH_TOKEN, { 3: "Nassau grouper" }, null),
    { name: FISH_TOKEN, code: null, resolved: false }, "speciesFromClassName: without a class id the token stays");
  eq(R.speciesFromClassName("PA", { PA: "By token", 0: "By class id" }, 0),
    { name: "By token", code: "PA", resolved: true }, "speciesFromClassName: the token key wins over the class id key");
  eq(R.speciesFromClassName("Acropora-palmata", { 9: "Something else" }, 9),
    { name: "Something else", code: "Acropora-palmata", resolved: true }, "speciesFromClassName: a named class id beats the binomial rule");
  eq(R.speciesFromClassName("Acropora-palmata", { 3: "Not this one" }, 9),
    { name: "Acropora palmata", code: null, resolved: true }, "speciesFromClassName: an unnamed class id leaves the binomial rule alone");
  eq(R.speciesFromClassName(FISH_TOKEN, { 3: "   " }, 3),
    { name: FISH_TOKEN, code: null, resolved: false }, "speciesFromClassName: a blank name resolves nothing");
  eq(R.speciesFromClassName(FISH_TOKEN, { 3: 7 }, 3),
    { name: FISH_TOKEN, code: null, resolved: false }, "speciesFromClassName: a name that is not text resolves nothing");

  // buildRows carries the class id from the metrics class map, so the id-keyed lookup reaches every row.
  const rows = R.buildRows(fishFixture().metrics, "box", fishFixture().metrics.class_full_names);
  eq(rows.map(r => [r.classId, r.name]), [["0", FISH_NAME], ["1", FISH_NAME_2]], "buildRows: class id and full name per row");
  const coral = R.buildRows(detectorFixture().metrics, "box", null);
  eq(coral.map(r => r.classId), ["1", "0", "2", "3"], "buildRows: class ids follow the metrics class map, not the row order");
  const unmapped = R.buildRows({ per_class: { Wrasse: { box_map50: 0.5 } }, class_names: { 0: "Other" } }, "box", { 0: "Named" });
  eq([unmapped[0].classId, unmapped[0].name], [null, "Wrasse"], "buildRows: a token the class map does not carry has no class id");
  const listed = R.buildRows({ per_class: { B: { box_map50: 0.5 } }, class_names: ["A", "B"] }, "box", { 1: "Second class" });
  eq([listed[0].classId, listed[0].name], ["1", "Second class"], "buildRows: a class map written as a list gives each token its index");
  const dup = R.buildRows({ per_class: { A: { box_map50: 0.5 } }, class_names: { 0: "A", 1: "A" } }, "box", { 1: "Second" });
  eq(dup[0].classId, "0", "buildRows: when two ids share a token the first one wins");
  const hostile = R.buildRows({ per_class: { A: { box_map50: 0.5 } }, class_names: "not a map" }, "box", { 0: "Never" });
  eq([hostile[0].classId, hostile[0].name], [null, "A"], "buildRows: a class map of the wrong type is ignored, not thrown on");
  const proto = R.buildRows({ per_class: { __proto__: { box_map50: 0.5 }, A: { box_map50: 0.4 } },
    class_names: { 0: "__proto__", 1: "A" } }, "box", { 0: "Never", 1: "Second" });
  ok(Array.isArray(proto) && proto.some(r => r.className === "A" && r.name === "Second"),
    "buildRows: a class token named __proto__ neither throws nor renames another class");

  // A name that arrives from the metrics file is data, so it is escaped before it is shown.
  const scripted = fishFixture();
  scripted.metrics.class_full_names = { 0: "<script>alert(1)</script>", 1: FISH_NAME_2 };
  const scriptedHtml = R.render(scripted, {});
  lacks(tagsOnly(scriptedHtml), "<script>", "class_full_names: a name carrying markup is escaped, never rendered as a tag");
  has(scriptedHtml, "&lt;script&gt;", "class_full_names: the escaped name is still shown");
})();

// ── UNIT: the case the live page takes, with the host's coral vocabulary passed ──

(function test_class_full_names_survive_the_hosts_coral_vocabulary() {
  // orchestrator.js passes the label registry's coral codes on every open of the Evaluate
  // panel, so this, not the empty-options case, is what a fish run renders through.
  const html = R.render(fishFixture(), { labelNames: CORAL_VOCABULARY });
  has(html, FISH_NAME_HTML + ' <span class="vega-report-chip">' + FISH_TOKEN + "</span>",
    "coral vocabulary: a fish class still reads its full name with the token as the chip");
  has(html, '<em class="vega-report-binomial">Kyphosus sectatrix</em>'
    + '<span class="vega-report-plain">, Bermuda chub</span>',
    "coral vocabulary: the second fish class is named too");
  lacks(html, '<span class="vega-report-plain">' + FISH_TOKEN + "</span>",
    "coral vocabulary: the fish token never stands alone once the dataset names it");

  // A coral run is unchanged: the host's map names the coral codes exactly as it did.
  const coralRun = detectorFixture();
  coralRun.metrics.class_names = { 0: "PAST" };
  coralRun.metrics.per_class = { PAST: { box_precision: 0.5, box_recall: 0.5, box_map50: 0.5, box_map50_95: 0.4 } };
  coralRun.metrics.per_class_instances = { PAST: 9 };
  coralRun.metrics.absent_classes = [];
  delete coralRun.metrics.class_full_names;
  has(R.render(coralRun, { labelNames: CORAL_VOCABULARY }),
    '<em class="vega-report-binomial">Porites astreoides</em> <span class="vega-report-chip">PAST</span>',
    "coral vocabulary: a coral code reads the same way it did before the metrics names existed");

  // A name that is not a species reads upright, the way evaluate_run renders it in the PDF.
  const unidentified = fishFixture();
  unidentified.metrics.class_full_names = { 0: "grouper, species not identified", 1: FISH_NAME_2 };
  const unidentifiedHtml = R.render(unidentified, { labelNames: CORAL_VOCABULARY });
  has(unidentifiedHtml, '<span class="vega-report-plain">grouper, species not identified</span>'
    + ' <span class="vega-report-chip">' + FISH_TOKEN + "</span>",
    "a name that is not a species is upright body text with the token as the chip");
  lacks(unidentifiedHtml, '<em class="vega-report-binomial">grouper',
    "a name that is not a species is never italicised as a binomial");
})();

(function test_family_aware_top_line() {
  const det = R.render(detectorFixture(), {});
  has(det, "<th>Metric</th><th class=\"num\">Boxes</th></tr>", "detector: box column only");
  lacks(det, "Outlines", "detector: no outline column");
  has(det, "This run is a detector: it predicts boxes, not outlines.", "detector: one sentence says it predicts boxes");
  has(det, "<td>mAP50</td><td class=\"num\">0.209</td>", "detector: mAP50 row");
  has(det, "<td>mAP50-95</td><td class=\"num\">0.177</td>", "detector: mAP50-95 row");
  has(det, "<td>Precision</td><td class=\"num\">0.335</td>", "detector: precision row");
  has(det, "<td>Recall</td><td class=\"num\">0.251</td>", "detector: recall row");
  lacks(det, "N/A", "detector: never a column of N/A");
  lacks(det, "no score", "detector: no placeholder cells when every box value is present");

  const seg = R.render(segmenterFixture(), {});
  has(seg, "<th>Metric</th><th class=\"num\">Outlines</th><th class=\"num\">Boxes</th></tr>", "segmenter: outline and box columns");
  lacks(seg, "This run is a detector", "segmenter: no detector sentence");
  has(seg, "<td>mAP50-95</td><td class=\"num\">0.545</td><td class=\"num\">0.569</td>", "segmenter: outline then box values");

  // Family from the numbers when metric_family is missing.
  eq(R.metricFamily({ overall: { mask_map50: 0.5 } }), "mask", "metricFamily: mask when a mask score exists");
  eq(R.metricFamily({ overall: { mask_map50: null, box_map50: 0.5 } }), "box", "metricFamily: box when mask scores are null");
  eq(R.metricFamily({ metric_family: "box", overall: { mask_map50: 0.5 } }), "box", "metricFamily: declared family wins");
  eq(R.metricFamily({ metric_family: "nonsense", overall: { mask_map50: 0.5 } }), "mask", "metricFamily: unknown declared family falls back to the numbers");
  eq(R.metricFamily(null), "box", "metricFamily: no metrics defaults to box");

  // A segmenter with one missing outline value shows a placeholder cell, not N/A.
  const partial = segmenterFixture();
  partial.metrics.overall.mask_recall = null;
  const p = R.render(partial, {});
  has(p, '<td>Recall</td><td class="num vega-report-noscore">no score</td><td class="num">0.619</td>', "partial: a single missing value shows the placeholder word");
  lacks(p, "N/A", "partial: still no N/A");
})();

// ── UNIT: test_figures_with_null_urls_are_omitted ────────────────────
(function test_figures_with_null_urls_are_omitted() {
  const det = R.render(detectorFixture(), {});
  eq(count(det, '<figure class="vega-report-fig">'), 4, "detector: four figures (two curves, two matrices)");
  has(det, "Precision-recall curve, boxes", "detector: box PR caption");
  has(det, "F1 against confidence, boxes", "detector: box F1 caption");
  has(det, "Confusion matrix, normalized", "detector: normalized matrix caption");
  has(det, "Confusion matrix, counts", "detector: count matrix caption");
  lacks(det, "outlines</figcaption>", "detector: no outline curve captions");
  lacks(det, "Training curves", "detector: results figure omitted when null");
  const seg = R.render(segmenterFixture(), {});
  eq(count(seg, '<figure class="vega-report-fig">'), 7, "segmenter: all seven figures when every url is present");
  has(seg, "Training curves by epoch", "segmenter: results figure captioned");
  const order = ["Precision-recall curve, boxes", "F1 against confidence, boxes", "Precision-recall curve, outlines",
    "F1 against confidence, outlines", "Confusion matrix, normalized", "Confusion matrix, counts", "Training curves by epoch"]
    .map(c => seg.indexOf(c));
  ok(order.every((p, i) => i === 0 || p > order[i - 1]), "segmenter: figures in the fixed order");
  const hostile = detectorFixture();
  hostile.figures = { box_pr: "javascript:alert(1)", box_f1: "//evil.example/x.png", mask_pr: "https://evil.example/y.png",
    mask_f1: 12, confusion_norm: imageUrl("val_run/confusion_matrix_normalized.png"), confusion_abs: "", results: undefined };
  const h = R.render(hostile, {});
  eq(count(h, '<figure class="vega-report-fig">'), 1, "hostile: only the same-origin path renders");
  lacks(h, "javascript:", "hostile: javascript url dropped");
  lacks(h, "evil.example", "hostile: external urls dropped");
  const none = detectorFixture();
  none.figures = null;
  lacks(R.render(none, {}), "vega-report-figs", "no figures object: the strip is omitted entirely");
  const empty = detectorFixture();
  empty.figures = {};
  lacks(R.render(empty, {}), "vega-report-figs", "empty figures object: the strip is omitted entirely");
})();

// ── UNIT: test_sort_by_column_is_stable ──────────────────────────────
(function test_sort_by_column_is_stable() {
  const rows = [
    { className: "b", name: "Bb", instances: 5, precision: 0.2, recall: 0.9, map50: 0.4, map5095: 0.1, absent: false },
    { className: "a", name: "Aa", instances: 5, precision: 0.2, recall: 0.1, map50: 0.4, map5095: 0.3, absent: false },
    { className: "z", name: "Zz", instances: 0, precision: null, recall: null, map50: null, map5095: null, absent: true },
    { className: "c", name: "Cc", instances: 7, precision: 0.9, recall: 0.5, map50: 0.9, map5095: 0.2, absent: false },
    { className: "d", name: "Dd", instances: 1, precision: null, recall: 0.5, map50: 0.4, map5095: null, absent: false },
  ];
  const names = list => list.map(r => r.className);
  eq(names(R.sortRows(rows, "map50", "desc")), ["c", "b", "a", "d", "z"], "desc: ties keep input order, absent last");
  eq(names(R.sortRows(rows, "map50", "asc")), ["b", "a", "d", "c", "z"], "asc: ties keep input order, absent still last");
  eq(names(R.sortRows(rows, "instances", "desc")), ["c", "b", "a", "d", "z"], "instances desc");
  eq(names(R.sortRows(rows, "precision", "asc")), ["b", "a", "c", "d", "z"], "asc: a missing score sorts after every number, absent after that");
  eq(names(R.sortRows(rows, "name", "asc")), ["a", "b", "c", "d", "z"], "name asc alphabetical, absent last");
  eq(names(R.sortRows(rows, "name", "desc")), ["d", "c", "b", "a", "z"], "name desc, absent last");
  eq(names(R.sortRows(rows, "bogus", "desc")), names(R.sortRows(rows, "map50", "desc")), "unknown key falls back to the default sort");
  eq(names(rows), ["b", "a", "z", "c", "d"], "sortRows never mutates its input");
  eq(R.sortRows([], "map50", "desc"), [], "empty rows");
  eq(R.sortRows(null, "map50", "desc"), [], "null rows tolerated");

  eq(R.nextSort({ key: "map50", dir: "desc" }, "map50"), { key: "map50", dir: "asc" }, "nextSort: same key flips direction");
  eq(R.nextSort({ key: "map50", dir: "asc" }, "recall"), { key: "recall", dir: "desc" }, "nextSort: numeric column starts descending");
  eq(R.nextSort({ key: "map50", dir: "desc" }, "name"), { key: "name", dir: "asc" }, "nextSort: name column starts ascending");
  eq(R.nextSort({ key: "map50", dir: "desc" }, "bogus"), { key: "map50", dir: "desc" }, "nextSort: unknown key leaves the state alone");

  eq(R.readSortState(fakeStorage({ "vega.report.sort": JSON.stringify({ key: "recall", dir: "asc" }) })), { key: "recall", dir: "asc" }, "readSortState: saved state");
  eq(R.readSortState(fakeStorage({ "vega.report.sort": "{not json" })), R.DEFAULT_SORT, "readSortState: corrupt json falls back");
  eq(R.readSortState(fakeStorage({ "vega.report.sort": JSON.stringify({ key: "evil", dir: "sideways" }) })), R.DEFAULT_SORT, "readSortState: invalid values fall back");
  eq(R.readSortState(fakeStorage({})), R.DEFAULT_SORT, "readSortState: nothing saved");
  eq(R.readSortState(null), R.DEFAULT_SORT, "readSortState: no storage");
  eq(R.readSortState(throwingStorage()), R.DEFAULT_SORT, "readSortState: a throwing storage is swallowed");
  const st = fakeStorage({});
  R.writeSortState(st, { key: "name", dir: "asc" });
  eq(JSON.parse(st.store["vega.report.sort"]), { key: "name", dir: "asc" }, "writeSortState: saved as json");
  ok((function () { try { R.writeSortState(throwingStorage(), R.DEFAULT_SORT); return true; } catch (e) { return false; } })(), "writeSortState: a throwing storage is swallowed");
  ok((function () { try { R.writeSortState(null, R.DEFAULT_SORT); return true; } catch (e) { return false; } })(), "writeSortState: no storage is fine");

  const table = R.renderPerClassTable(rows, { key: "name", dir: "asc" });
  ok(table.indexOf("Aa") < table.indexOf("Bb") && table.indexOf("Bb") < table.indexOf("Cc"), "renderPerClassTable honours the sort passed in");
  has(table, 'aria-sort="ascending"', "renderPerClassTable marks the active column ascending");
  const rendered = R.render(detectorFixture(), { sort: { key: "name", dir: "asc" } });
  ok(rendered.indexOf("Acropora cervicornis") < rendered.indexOf("Acropora palmata"), "render honours options.sort");
})();

// ── UNIT: test_empty_and_error_states ────────────────────────────────
(function test_empty_and_error_states() {
  const EMPTY = "No evaluation has been run for this training run yet.";
  has(R.render(null, {}), EMPTY, "empty: null data");
  has(R.render(undefined, {}), EMPTY, "empty: undefined data");
  has(R.render({ error: "no evaluation folder for run_x", ready: false }, {}), EMPTY, "empty: the route says ready false");
  has(R.render({ error: "run folder not found: /x", _status: 404 }, {}), "run folder not found: /x", "error: a 404 without ready false shows the server text");
  has(R.render({ error: "" , ready: false }, {}), EMPTY, "empty: blank error with ready false");
  has(R.render({ error: "" }, {}), EMPTY, "empty: a blank error alone is the empty state, never an empty red line");
  lacks(R.render({ error: "" }, {}), "vega-report-error", "empty: no error element for a blank error");
  const blankError = Object.assign(detectorFixture(), { error: "" });
  lacks(R.render(blankError, {}), "vega-report-error", "a blank error on a good body is not an error");
  has(R.render(blankError, {}), '<span class="vega-report-runname">run_a</span>', "a blank error on a good body renders the card");
  has(R.render({ ok: true, run_name: "x", run_dir: RUN_DIR }, {}), EMPTY, "empty: ok body with no metrics and no report");
  has(R.render({ ready: false }, {}), EMPTY, "empty: legacy ready false");
  has(R.render({ ok: true, evaluated: false, run_name: "x" }, {}), EMPTY, "empty: evaluated false");
  has(R.render("a string", {}), EMPTY, "empty: a non-object body");
  eq(count(R.render(null, {}), "<"), 2, "empty: exactly one element (open and close), one plain sentence");
  const err = R.render({ error: "Evaluation folder unreadable: <script>alert(1)</script>" }, {});
  has(err, 'class="vega-report-error"', "error: marked as an error");
  has(err, "Evaluation folder unreadable: &lt;script&gt;alert(1)&lt;/script&gt;", "error: server text shown, escaped");
  lacks(err, "<script", "error: no script tag");
  has(R.render({ error: "HTTP 500", _status: 500 }, {}), "HTTP 500", "error: a non-404 status with an error is an error");
  has(R.render({ error: { nested: true } }, {}), "vega-report-error", "error: a non-string error still renders as an error");
  has(R.render({ ok: true, run_name: "x", run_dir: RUN_DIR, report_md: "# Only text" }, {}), "Full report text", "report text alone is not the empty state");
  lacks(R.render({ ok: true, run_name: "x", run_dir: RUN_DIR, report_md: "# Only text" }, {}), "vega-report-perclass", "report text alone: no per-class table rendered");
})();

// ── UNIT: test_pdf_link_only_when_present ────────────────────────────
(function test_pdf_link_only_when_present() {
  const det = R.render(detectorFixture(), {});
  has(det, '<a class="btn btn-magenta btn-sm vega-report-pdf" href="/api/step/7/pdf?run_dir=' + R.escapeHtml(RUN_DIR_ENC) + '" title="' + R.escapeHtml(R.TOOLTIPS.pdf) + '">Download PDF</a>', "pdf: link built from pdf_url with its tooltip");
  lacks(det, "No PDF was built", "pdf: no absence sentence when the link exists");
  const seg = R.render(segmenterFixture(), {});
  lacks(seg, "vega-report-pdf", "pdf: no link when pdf_url is null");
  has(seg, '<span class="vega-report-nopdf">No PDF was built for this evaluation.</span>', "pdf: one sentence when absent");
  const bad = detectorFixture();
  bad.pdf_url = "javascript:alert(1)";
  lacks(R.render(bad, {}), "vega-report-pdf", "pdf: an unsafe url is treated as absent");
  has(R.render(bad, {}), "No PDF was built", "pdf: unsafe url shows the absence sentence");
  const missing = detectorFixture();
  delete missing.pdf_url;
  has(R.render(missing, {}), "No PDF was built", "pdf: missing key is absent");
})();

// ── UNIT: heading ────────────────────────────────────────────────────
(function test_heading_shows_run_name_and_ast_timestamp() {
  const det = R.render(detectorFixture(), {});
  has(det, '<span class="vega-report-runname">run_a</span>', "heading: run name");
  has(det, '<span class="vega-report-when">evaluated 2026-09-03 20:49 AST</span>', "heading: evaluated timestamp in AST, 24 hour");
  eq(R.formatEvaluatedAt("2026-09-03T20:49:29-04:00"), "2026-09-03 20:49 AST", "formatEvaluatedAt: ISO with offset");
  eq(R.formatEvaluatedAt("2026-09-03T20:49:29.432679"), "2026-09-03 20:49 AST", "formatEvaluatedAt: naive ISO");
  eq(R.formatEvaluatedAt("2026-09-03 20:49"), "2026-09-03 20:49 AST", "formatEvaluatedAt: space separated");
  eq(R.formatEvaluatedAt("yesterday-ish"), "yesterday-ish", "formatEvaluatedAt: unparseable text shown as it is");
  eq(R.formatEvaluatedAt(null), "", "formatEvaluatedAt: null is empty");
  eq(R.formatEvaluatedAt(1725400000), "", "formatEvaluatedAt: a number is empty, never a guessed date");
  const noWhen = detectorFixture();
  delete noWhen.evaluated_at;
  lacks(R.render(noWhen, {}), "vega-report-when", "heading: no evaluated phrase when the timestamp is missing");
  const noName = detectorFixture();
  noName.run_name = "";
  has(R.render(noName, {}), '<span class="vega-report-runname">run_a</span>', "heading: falls back to the run folder basename");
  eq(R.formatScore(0.20883), "0.209", "formatScore: three decimals");
  eq(R.formatScore(1), "1.000", "formatScore: integer");
  eq(R.formatScore("0.5"), null, "formatScore: a string is not a score");
  eq(R.formatScore(NaN), null, "formatScore: NaN");
  eq(R.formatScore(Infinity), null, "formatScore: Infinity");
})();

// ── UNIT: previews ───────────────────────────────────────────────────
(function test_previews_render_with_captions() {
  const det = R.render(detectorFixture(), {});
  eq(count(det, '<figure class="vega-report-preview">'), 1, "previews: one figure per entry");
  has(det, '<img src="' + R.escapeHtml(imageUrl("previews/Acropora-palmata_881_sbs.jpg")) + '" alt="Acropora-palmata_881.jpg" loading="lazy">', "previews: img from url");
  has(det, '<span class="vega-report-file">Acropora-palmata_881.jpg</span>: the human labeler drew 1 instance; the model produced 2 detections', "previews: caption sentence");
  const many = detectorFixture();
  many.previews = [
    { filename: "<b>x</b>.jpg", url: imageUrl("previews/x.jpg"), n_gt: "3", n_pred: null },
    { filename: "bad.jpg", url: "https://evil.example/bad.jpg", n_gt: 1, n_pred: 1 },
    { filename: "nourl.jpg", n_gt: 1, n_pred: 1 },
    "not an object",
  ];
  const m = R.render(many, {});
  eq(count(m, '<figure class="vega-report-preview">'), 1, "previews: entries without a safe url are dropped");
  has(m, "&lt;b&gt;x&lt;/b&gt;.jpg", "previews: filename escaped");
  has(m, "drew 3 instances; the model produced 0 detections", "previews: numeric strings coerced, a missing count reads as 0");
  const one = detectorFixture();
  one.previews = [{ filename: "one.jpg", url: imageUrl("previews/one.jpg"), n_gt: 1, n_pred: 1 }];
  has(R.render(one, {}), "drew 1 instance; the model produced 1 detection", "previews: a count of one takes the singular noun");
  const zero = detectorFixture();
  zero.previews = [{ filename: "zero.jpg", url: imageUrl("previews/zero.jpg"), n_gt: 0, n_pred: 0 }];
  has(R.render(zero, {}), "drew 0 instances; the model produced 0 detections", "previews: a count of zero takes the plural noun");
  lacks(R.render(one, {}), "(s)", "previews: the caption never falls back to the (s) construct");
  const none = detectorFixture();
  none.previews = "nope";
  lacks(R.render(none, {}), "vega-report-previews", "previews: a non-array omits the section");
})();

// ── UNIT: markdown section ───────────────────────────────────────────
(function test_markdown_section_is_collapsed_and_rewrites_previews() {
  const det = R.render(detectorFixture(), {});
  has(det, '<details class="vega-report-md">', "md: details element");
  lacks(det, "<details class=\"vega-report-md\" open", "md: collapsed by default");
  has(det, ">Full report text</summary>", "md: disclosure label");
  has(det, 'title="' + R.escapeHtml(R.TOOLTIPS.fullText) + '"', "md: summary tooltip");
  has(det, '<img class="vega-report-mdimg" src="' + R.escapeHtml(imageUrl("previews/Acropora-palmata_881_sbs.jpg")), "md: preview image inside the report rewritten to the image route");
  lacks(det, "![", "md: no literal image syntax left on the page");
  has(det, "<code>run_a</code>", "md: inline code in the heading");
  const custom = R.render(detectorFixture(), { imageUrlFor: () => "/custom/path.jpg" });
  has(custom, 'src="/custom/path.jpg"', "md: options.imageUrlFor overrides the route builder");
  const noMd = detectorFixture();
  noMd.report_md = "";
  lacks(R.render(noMd, {}), "vega-report-md", "md: empty report text omits the disclosure");
  const wrongType = detectorFixture();
  wrongType.report_md = { not: "a string" };
  lacks(R.render(wrongType, {}), "vega-report-md", "md: a non-string report text omits the disclosure");
})();

// ── ADVERSARIAL ──────────────────────────────────────────────────────
(function test_hostile_strings_and_wrong_types() {
  const d = detectorFixture();
  d.run_name = '<img src=x onerror=alert(1)>';
  d.run_dir = RUN_DIR;
  d.metrics.per_class['<script>alert(2)</script>'] = { box_map50: 0.1 };
  d.metrics.per_class_instances['<script>alert(2)</script>'] = 2;
  const html = R.render(d, {});
  lacks(html, "<img src=x", "hostile run name escaped");
  has(html, "&lt;img src=x onerror=alert(1)&gt;", "hostile run name shown as text");
  lacks(html, "<script", "hostile class name escaped");
  lacks(tagsOnly(html), "onerror", "no onerror attribute in any tag");

  const wrong = { ok: true, run_name: "w", run_dir: RUN_DIR, evaluated_at: 5,
    metrics: { overall: "nope", per_class: ["a", "b"], per_class_instances: 7, absent_classes: "x", metric_family: 3 },
    previews: { not: "array" }, figures: "none", report_md: 9, pdf_url: 4 };
  const w = R.render(wrong, {});
  has(w, '<span class="vega-report-runname">w</span>', "wrong types: still renders the heading");
  lacks(w, "vega-report-fig\"", "wrong types: no figures");
  lacks(w, "vega-report-preview\"", "wrong types: no previews");
  has(w, "No PDF was built", "wrong types: pdf treated as absent");
  ok(!/undefined|NaN|\[object Object\]/.test(w), "wrong types: no undefined, NaN or [object Object] leaks into the page");

  const nulls = { ok: true, run_name: "n", run_dir: RUN_DIR, metrics: null, previews: null, figures: null, report_md: null, pdf_url: null };
  has(R.render(nulls, {}), "No evaluation has been run", "all-null body is the empty state");

  ok((function () { try { R.markdownToHtml(null, () => null); return false; } catch (e) { return e instanceof TypeError; } })(), "markdownToHtml validates md");
  ok((function () { try { R.markdownToHtml("x", null); return false; } catch (e) { return e instanceof TypeError; } })(), "markdownToHtml validates the resolver");
  eq(R.escapeHtml("<&>\"'"), "&lt;&amp;&gt;&quot;&#39;", "escapeHtml covers the five characters");
  eq(R.escapeHtml(null), "", "escapeHtml: null is empty");
  eq(R.escapeHtml(0), "0", "escapeHtml: zero is the digit");
  eq(R.isSafeUrl("/api/step/7/image?x=1"), true, "isSafeUrl: same-origin path");
  eq(R.isSafeUrl("//evil"), false, "isSafeUrl: protocol-relative refused");
  eq(R.isSafeUrl("/\\evil"), false, "isSafeUrl: backslash after the slash refused");
  eq(R.isSafeUrl("javascript:alert(1)"), false, "isSafeUrl: scheme refused");
  eq(R.isSafeUrl("/api/x y"), false, "isSafeUrl: whitespace refused");
  eq(R.isSafeUrl(""), false, "isSafeUrl: empty refused");
  eq(R.isSafeUrl(null), false, "isSafeUrl: null refused");
})();

(function test_huge_report_renders_in_bounded_time() {
  const lines = [];
  for (let i = 0; i < 20000; i++) {
    if (i % 4 === 0) lines.push("| a" + i + " | b | c |");
    else if (i % 4 === 1) lines.push("- bullet **" + i + "** with `code` and ![p](previews/" + i + ".jpg)");
    else if (i % 4 === 2) lines.push("## Heading " + i);
    else lines.push("paragraph " + i + " *em* text");
  }
  const md = lines.join("\n");
  const t0 = Date.now();
  const html = R.markdownToHtml(md, src => R.previewImageUrl(RUN_DIR, src));
  const ms = Date.now() - t0;
  ok(ms < 3000, "huge: 20000 lines render in under 3 s (took " + ms + " ms)");
  ok(html.length > md.length, "huge: output produced");
  eq(count(html, "<img"), 5000, "huge: every preview image rewritten");
  const big = detectorFixture();
  big.metrics.per_class = {};
  big.metrics.per_class_instances = {};
  for (let i = 0; i < 600; i++) {
    big.metrics.per_class["Genus-species" + i] = { box_map50: (i % 7) / 10, box_map50_95: 0.1, box_precision: 0.2, box_recall: 0.3 };
    big.metrics.per_class_instances["Genus-species" + i] = i;
  }
  const t1 = Date.now();
  const rendered = R.render(big, {});
  ok(Date.now() - t1 < 2000, "huge: 600 species render quickly");
  eq(count(rendered, "<tr>") + count(rendered, '<tr class="vega-report-absent">'), 600 + 1 + 6, "huge: 600 species rows, the absent row, two header rows and four top-line rows");
})();

(function test_pathological_markdown_renders_in_bounded_time() {
  const t0 = Date.now();
  const html = R.markdownToHtml("[".repeat(100000) + "\n\n" + "![".repeat(50000) + "\n\n" + "](".repeat(50000), () => null);
  const ms = Date.now() - t0;
  ok(ms < 1500, "pathological: 100000 brackets render in under 1.5 s (took " + ms + " ms)");
  ok(html.length >= 100000, "pathological: output produced");
  const t1 = Date.now();
  R.markdownToHtml("![" + "a".repeat(100000) + "](x) [" + "b".repeat(100000) + "](y)", () => null);
  ok(Date.now() - t1 < 1500, "pathological: a 100000-character alt or link text is bounded");
})();

(function test_nul_in_report_text_is_stripped() {
  const nul = String.fromCharCode(0);
  const html = R.markdownToHtml("a" + nul + "0" + nul + "b and `code` here", () => null);
  lacks(html, "undefined", "nul: never prints undefined");
  lacks(html, nul, "nul: stripped from the output");
  has(html, "<p>a0b and <code>code</code> here</p>", "nul: the surrounding text and the code span survive");
})();

(function test_code_span_with_pipe_inside_a_table_cell() {
  const md = ["| Code | Value |", "|---|---|", "| `x|y` | 1 |", "| plain | `a|b|c` |"].join("\n");
  const html = R.markdownToHtml(md, () => null);
  has(html, "<tr><td><code>x|y</code></td><td>1</td></tr>", "table: a pipe inside inline code does not split a non-last cell");
  has(html, "<tr><td>plain</td><td><code>a|b|c</code></td></tr>", "table: pipes inside code in the last column");
  eq(count(html, "<td>"), 4, "table: exactly four cells");
})();

async function test_stale_load_is_discarded() {
  const card = fakeCard();
  let releaseSlow;
  const slowReply = new Promise(resolve => { releaseSlow = resolve; });
  const slow = R.load(card, RUN_DIR + "_old", { fetchJson: () => slowReply, storage: null });
  const fast = await R.load(card, RUN_DIR, { fetchJson: async () => detectorFixture(), storage: null });
  const stale = detectorFixture();
  stale.run_name = "stale_run";
  releaseSlow(stale);
  const slowData = await slow;
  eq(slowData.run_name, "stale_run", "stale load: still resolves with its own payload");
  has(card.body.innerHTML, "run_a", "stale load: the newer render stays on the page");
  lacks(card.body.innerHTML, "stale_run", "stale load: the older response never overwrites it");
  eq(card.status.innerHTML, "", "stale load: the status line stays clear");
  ok(fast && fast.ok === true, "stale load: the newer load resolved normally");
}

async function test_load_wires_fetch_states_and_sorting() {
  const EMPTY = "No evaluation has been run for this training run yet.";
  let card = fakeCard();
  const seen = [];
  const storage = fakeStorage({});
  const data = await R.load(card, RUN_DIR, { fetchJson: async url => { seen.push(url); return detectorFixture(); }, storage });
  eq(seen, ["/api/step/7/report?run_dir=" + RUN_DIR_ENC], "load: fetches the report route keyed by run folder");
  has(card.body.innerHTML, '<span class="vega-report-runname">run_a</span>', "load: rendered into the body");
  ok(data && data.ok === true, "load: resolves with the data");
  ok(typeof card.body.handlers.click === "function", "load: a click handler is attached once");

  // Click a column header: the table re-renders sorted and the state is saved.
  card.body.handlers.click(clickOn("name"));
  ok(card.species.innerHTML.indexOf("Acropora cervicornis") < card.species.innerHTML.indexOf("Acropora palmata"), "sort click: name ascending re-renders the table");
  eq(card.species.focused, ["name"], "sort click: keyboard focus returns to the pressed column button");
  eq(JSON.parse(storage.store["vega.report.sort"]), { key: "name", dir: "asc" }, "sort click: state saved to storage");
  card.body.handlers.click(clickOn("name"));
  eq(JSON.parse(storage.store["vega.report.sort"]), { key: "name", dir: "desc" }, "sort click: second press flips the direction");
  eq(card.species.focused, ["name", "name"], "sort click: focus returns on every press");
  card.body.handlers.click({ target: { closest() { return null; } } });
  eq(JSON.parse(storage.store["vega.report.sort"]), { key: "name", dir: "desc" }, "sort click: a click elsewhere changes nothing");

  // The saved state is honoured on the next load, and a second load does not attach a second handler.
  let attached = 0;
  card.body.addEventListener = function (type, fn) { attached += 1; this.handlers[type] = fn; };
  await R.load(card, RUN_DIR, { fetchJson: async () => detectorFixture(), storage });
  ok(card.body.innerHTML.indexOf("Porites astreoides") < card.body.innerHTML.indexOf("Acropora palmata"), "load: saved name-descending sort applied");
  eq(attached, 0, "load: the handler is attached only once per body");

  // Empty runDir: no query string (the route falls back to the project's last evaluation).
  card = fakeCard();
  const urls = [];
  await R.load(card, "", { fetchJson: async url => { urls.push(url); return { error: "none", ready: false }; }, storage: null });
  eq(urls, ["/api/step/7/report"], "load: blank run folder fetches without a query string");
  has(card.status.innerHTML, EMPTY, "load: ready false renders the empty sentence in the status line");
  eq(card.body.innerHTML, "", "load: the body is cleared in the empty state");

  // A rejecting fetch becomes the error state with the message, never an exception.
  card = fakeCard();
  await R.load(card, RUN_DIR, { fetchJson: async () => { throw new Error("net down"); }, storage: throwingStorage() });
  has(card.status.innerHTML, 'class="vega-report-error"', "load: rejected fetch is an error state in the status line");
  has(card.status.innerHTML, "net down", "load: the message is shown");
  eq(card.body.innerHTML, "", "load: the body is cleared in the error state");

  // A throwing storage never breaks the page, including on a sort click.
  card = fakeCard();
  await R.load(card, RUN_DIR, { fetchJson: async () => detectorFixture(), storage: throwingStorage() });
  ok((function () { try { card.body.handlers.click(clickOn("recall")); return true; } catch (e) { return false; } })(), "sort click with a throwing storage still re-renders");
  has(card.species.innerHTML, 'aria-sort="descending"', "sort click: recall descending applied in memory");

  // Input validation.
  for (const bad of [null, "string", 42]) {
    let threw = false;
    try { await R.load(bad, RUN_DIR, { fetchJson: async () => ({}) }); } catch (e) { threw = e instanceof TypeError; }
    ok(threw, "load: rejects a non-element root: " + JSON.stringify(bad));
  }
  let threwRun = false;
  try { await R.load(fakeCard(), 42, { fetchJson: async () => ({}) }); } catch (e) { threwRun = e instanceof TypeError; }
  ok(threwRun, "load: rejects a non-string run folder");

  // A root that is itself the body (no .vega-report-body inside) is filled directly.
  const bare = { innerHTML: "", addEventListener() {}, querySelector() { return null; } };
  await R.load(bare, RUN_DIR, { fetchJson: async () => detectorFixture(), storage: null });
  has(bare.innerHTML, "run_a", "load: a root without an inner body is filled itself");
  const bareEmpty = { innerHTML: "", addEventListener() {}, querySelector() { return null; } };
  await R.load(bareEmpty, RUN_DIR, { fetchJson: async () => ({ ready: false }), storage: null });
  has(bareEmpty.innerHTML, EMPTY, "load: a root without a status line shows the one-line state itself");

  // Label names reach the species column.
  card = fakeCard();
  await R.load(card, RUN_DIR, { fetchJson: async () => segmenterFixture(), storage: null, labelNames: { PA: "Porites astreoides" } });
  has(card.body.innerHTML, "Porites astreoides</em> <span class=\"vega-report-chip\">PA</span>", "load: labelNames option resolves codes");

  // The loading line is written before the fetch settles.
  card = fakeCard();
  let seenLoading = "";
  await R.load(card, RUN_DIR, { fetchJson: async () => { seenLoading = card.status.innerHTML; return detectorFixture(); }, storage: null });
  has(seenLoading, "Loading the evaluation report.", "load: loading line shown in the status line while fetching");
  eq(card.status.innerHTML, "", "load: status line cleared once the card renders");

  // Focus follows whichever column was pressed, and a container that cannot find the
  // button (an older browser, a stubbed node) still re-renders without throwing.
  const focusCard = fakeCard();
  await R.load(focusCard, RUN_DIR, { fetchJson: async () => detectorFixture(), storage: null });
  focusCard.body.handlers.click(clickOn("recall"));
  focusCard.body.handlers.click(clickOn("instances"));
  eq(focusCard.species.focused, ["recall", "instances"], "sort click: focus follows the column that was pressed");
  const noFocus = fakeCard();
  noFocus.species.querySelector = () => null;
  await R.load(noFocus, RUN_DIR, { fetchJson: async () => detectorFixture(), storage: null });
  ok((function () { try { noFocus.body.handlers.click(clickOn("recall")); return true; } catch (e) { return false; } })(), "sort click: a container that cannot find the button still re-renders");
  has(noFocus.species.innerHTML, 'aria-sort="descending"', "sort click: the table still re-rendered without a focus target");

  eq(R.reportUrl(RUN_DIR), "/api/step/7/report?run_dir=" + RUN_DIR_ENC, "reportUrl: keyed");
  eq(R.reportUrl(""), "/api/step/7/report", "reportUrl: blank");
  eq(R.reportUrl("  "), "/api/step/7/report", "reportUrl: whitespace is blank");
}

(async function main() {
  try {
    await test_load_wires_fetch_states_and_sorting();
    await test_stale_load_is_discarded();
  } catch (e) {
    fail++;
    console.error("  FAIL: load suite threw:", e && e.stack ? e.stack : e);
  }
  console.log(`test_vega_report: ${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
})();

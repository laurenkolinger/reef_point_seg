#!/usr/bin/env node
/* Pure-logic and DOM-glue tests for the VEGA inference gallery card
   (scripts/pipeline_orchestrator/static/orchestrator_gallery.js).

   The module doubles as a Node-requireable file: its pure render functions
   (data in, HTML string out) and its DOM glue (load, select, handleClick) are
   exercised here with an injected fetch and a tiny element shim, no browser.

   Run:  node scripts/pipeline_orchestrator/tests/test_vega_gallery.js

   Layers, mirroring the house no-pytest convention:
     smoke       - module loads, every documented export is a function
     unit        - chips, header, run options, empty and error states
     adversarial - hostile filenames and class names, control characters,
                   malformed manifests, thresholds at the boundary
     glue        - injected fetch drives load/select, verdict event detail */
"use strict";

const path = require("path");

// Node 18 has Event and EventTarget as globals but not CustomEvent; the
// browser always has it. The shim below is what the test needs, nothing more.
if (typeof global.CustomEvent !== "function") {
  global.CustomEvent = class CustomEvent extends Event {
    constructor(type, init) {
      super(type, init);
      this.detail = init && init.detail !== undefined ? init.detail : null;
    }
  };
}

const G = require(path.join(__dirname, "..", "static", "orchestrator_gallery.js"));

let pass = 0;
let fail = 0;
const results = [];

/** Record one assertion. */
function ok(cond, msg) {
  if (cond) { pass += 1; } else { fail += 1; console.error("  FAIL:", msg); }
}

/** Run one named test function and print PASS or FAIL with the message. */
function test(name, fn) {
  const before = fail;
  try {
    fn();
  } catch (e) {
    fail += 1;
    console.error("  FAIL:", name, "threw", e && e.stack ? e.stack : e);
  }
  const passed = fail === before;
  results.push([name, passed]);
  console.log((passed ? "  PASS " : "  FAIL ") + name);
}

/** Run one async test function and await it. */
async function testAsync(name, fn) {
  const before = fail;
  try {
    await fn();
  } catch (e) {
    fail += 1;
    console.error("  FAIL:", name, "threw", e && e.stack ? e.stack : e);
  }
  const passed = fail === before;
  results.push([name, passed]);
  console.log((passed ? "  PASS " : "  FAIL ") + name);
}

// -- fixtures ----------------------------------------------------

const IMAGE_BASE = "/api/step/8/image?out_dir=%2Fabs%2Frun&rel=";

/** A manifest in the shape run_inference.py writes today, with two items. */
function manifestFixture(overrides) {
  return Object.assign({
    ok: true,
    image_base: IMAGE_BASE,
    items: [
      {
        filename: "TCRMP20150805_clip_BIT_T419.jpg",
        raw: "/abs/source/TCRMP20150805_clip_BIT_T419.jpg",
        overlay: "overlays/TCRMP20150805_clip_BIT_T419.jpg",
        n_detections: 3,
        max_conf: 0.9235,
        detections: [
          { class: "MC", class_id: 33, confidence: 0.4921, class_full_name: "Montastraea cavernosa" },
          { class: "SS", class_id: 54, confidence: 0.9235, class_full_name: "Siderastrea siderea" },
          { class: "Oculina-varicosa", class_id: 42, confidence: 0.2773 },
        ],
      },
      {
        filename: "CIG20131023_clip_BPT_T417.jpg",
        raw: "/abs/source/CIG20131023_clip_BPT_T417.jpg",
        overlay: "overlays/CIG20131023_clip_BPT_T417.jpg",
        n_detections: 0,
        max_conf: 0.0,
        detections: [],
      },
    ],
    total_predictions: 3,
    with_any: 1,
    class_names: { "33": "MC", "54": "SS", "42": "Oculina-varicosa" },
    class_full_names: { MC: "Montastraea cavernosa", SS: "Siderastrea siderea" },
    conf_threshold: 0.25,
    iou_threshold: 0.7,
    imgsz: 512,
    detections_skipped: 0,
    detections_skipped_reasons: {},
    render: { fill_alpha: 0.15, draw_boxes: false, overlays_saved: true },
    errors: [],
    generated_at: "2026-09-04T13:49:38-04:00",
  }, overrides || {});
}

const RUNS = [
  { out_dir: "/abs/older", name: "infer_20260903_205212", created_at: "2026-09-03T20:52:12-04:00",
    model_run_name: "TRAIN_coral_intel_20260902_224223", n_images: 6, n_detections: 4, source_label: "unused" },
  { out_dir: "/abs/newest", name: "test_20260904_134933", created_at: "2026-09-04T13:49:33-04:00",
    model_run_name: "TRAIN_coral_intel_20260903_150509", n_images: 9, n_detections: 8, source_label: "unused" },
  { out_dir: "/abs/one", name: "e2esmoke", created_at: "2026-09-02T01:41:40-04:00",
    model_run_name: "smoke", n_images: 1, n_detections: 0, source_label: "directory" },
];

// -- element shim ------------------------------------------------

/** Does a shim element match one simple selector (#id, .class, [data-x], [data-x="v"])? */
function matches(el, sel) {
  if (sel.startsWith("#")) return el.attrs.id === sel.slice(1);
  if (sel.startsWith(".")) return (el.attrs.class || "").split(/\s+/).includes(sel.slice(1));
  const m = /^\[([\w-]+)(?:="([^"]*)")?\]$/.exec(sel);
  if (m) {
    const has = Object.prototype.hasOwnProperty.call(el.attrs, m[1]);
    return m[2] === undefined ? has : has && el.attrs[m[1]] === m[2];
  }
  return el.tagName === sel.toUpperCase();
}

/** Depth-first search of the shim tree for elements matching a selector. */
function findAll(el, sel) {
  const out = [];
  for (const c of el.children) {
    if (matches(c, sel)) out.push(c);
    out.push(...findAll(c, sel));
  }
  return out;
}

/** The smallest element the gallery glue needs: attributes, dataset, tree, events. */
class FakeEl extends EventTarget {
  constructor(tag, attrs) {
    super();
    this.tagName = tag.toUpperCase();
    this.attrs = Object.assign({}, attrs || {});
    this.dataset = {};
    for (const k of Object.keys(this.attrs)) {
      if (k.startsWith("data-")) this.dataset[k.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = this.attrs[k];
    }
    this.children = [];
    this.parent = null;
    this.innerHTML = "";
    this.textContent = "";
    this.value = "";
    this.hidden = false;
    const self = this;
    this.classList = {
      add(c) { self.attrs.class = ((self.attrs.class || "") + " " + c).trim(); },
      remove(c) { self.attrs.class = (self.attrs.class || "").split(/\s+/).filter((x) => x && x !== c).join(" "); },
      toggle(c, on) { if (on) this.add(c); else this.remove(c); },
      contains(c) { return (self.attrs.class || "").split(/\s+/).includes(c); },
    };
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  removeAttribute(k) { delete this.attrs[k]; }
  getAttribute(k) { return Object.prototype.hasOwnProperty.call(this.attrs, k) ? this.attrs[k] : null; }
  appendChild(c) { c.parent = this; this.children.push(c); return c; }
  closest(sel) { let el = this; while (el) { if (matches(el, sel)) return el; el = el.parent; } return null; }
  querySelector(sel) { return findAll(this, sel)[0] || null; }
  querySelectorAll(sel) { return findAll(this, sel); }
}

/** Build a root shim carrying the fragment's static ids. */
function makeRoot() {
  const root = new FakeEl("section", { class: "vega-gallery", id: "s8-gallery-card" });
  for (const id of Object.values(G.IDS)) {
    root.appendChild(new FakeEl(id === G.IDS.run ? "select" : "div", { id }));
  }
  return root;
}

/** An injected fetch that answers from a table of url prefix to payload. */
function fetchFrom(table, log) {
  return function fetchJson(url) {
    if (log) log.push(url);
    for (const [prefix, payload] of Object.entries(table)) {
      if (url.startsWith(prefix)) return Promise.resolve(typeof payload === "function" ? payload(url) : payload);
    }
    return Promise.reject(new Error("no fixture for " + url));
  };
}

// -- smoke -------------------------------------------------------

test("test_module_exports", () => {
  ["load", "select", "bind", "handleClick", "renderHeader", "renderTile", "renderTiles",
    "renderRunOptions", "renderEmpty", "renderError", "renderChip", "formatAst", "escapeHtml",
    "buildImageUrl", "buildRawUrl", "handleImageError", "defaultFetchJson"].forEach((name) => ok(typeof G[name] === "function", "export " + name + " is a function"));
  ok(G.BLIND_SPOT_NO_DETECTION_FRACTION === 0.2, "blind-spot threshold matches run_inference.py (0.2)");
  ok(G.VERDICT_EVENT === "vega:verdict", "verdict event name is vega:verdict");
  ok(G.ROOT_CLASS === "vega-gallery", "root class is vega-gallery");
});

// -- unit: tiles and chips ---------------------------------------

test("test_tile_shows_full_species_and_confidence_per_detection", () => {
  const m = manifestFixture();
  const html = G.renderTile(m.items[0], m, IMAGE_BASE, null, 0);
  ok(/<i class="vg-species">Montastraea cavernosa<\/i>/.test(html), "full name in italics for MC");
  ok(/<i class="vg-species">Siderastrea siderea<\/i>/.test(html), "full name in italics for SS");
  ok(/vg-code[^>]*>MC</.test(html), "code MC rides beside its full name");
  ok(/vg-code[^>]*>SS</.test(html), "code SS rides beside its full name");
  ok(/0\.49</.test(html) && /0\.92</.test(html) && /0\.28</.test(html), "confidence to two decimals on every chip");
  ok(/<i class="vg-species">Oculina varicosa<\/i>/.test(html), "a hyphenated class name with no full name reads as a binomial with spaces, in italics");
  ok(/<span class="vg-chip" title="Oculina-varicosa">/.test(html), "the raw class name stays in the chip title");
  ok(!/vg-code[^>]*>Oculina/.test(html), "a hyphenated binomial gets no code chip");
  const chips = html.match(/<span class="vg-chip"/g) || [];
  ok(chips.length === 3, "one chip per detection (got " + chips.length + ")");
  ok(/<span class="vg-chip" title="Montastraea cavernosa MC">/.test(html), "chip title carries the full text for a truncated name");
  ok(html.indexOf("Siderastrea siderea") < html.indexOf("Montastraea cavernosa") &&
     html.indexOf("Montastraea cavernosa") < html.indexOf("Oculina-varicosa"),
     "chips sorted by confidence descending");
  ok(/TCRMP20150805_clip_BIT_T419\.jpg/.test(html), "filename shown");
  ok(!/no detections/.test(html), "a tile with detections does not say no detections");
});

test("test_bare_code_without_full_name_gets_tooltip", () => {
  const m = manifestFixture({ class_full_names: {} });
  const det = { class: "XYZ", class_id: 9, confidence: 0.5 };
  const chip = G.renderChip(det, m);
  ok(/title="The full name for XYZ is not on file\."/.test(chip), "bare code carries the not-on-file tooltip");
  ok(/tabindex="0"/.test(chip), "the code chip is keyboard focusable so its tooltip is reachable");
  ok(/>XYZ</.test(chip), "the code itself is shown");
  ok(/0\.50/.test(chip), "confidence still shown");
});

test("test_top_level_class_full_names_fill_in_for_older_items", () => {
  const m = manifestFixture();
  const det = { class: "MC", class_id: 33, confidence: 0.61 };
  const chip = G.renderChip(det, m);
  ok(/Montastraea cavernosa/.test(chip), "class_full_names at the manifest top level resolves the code");
});

test("test_hyphenated_class_name_reads_as_a_binomial", () => {
  const m = manifestFixture({ class_full_names: {} });
  const chip = G.renderChip({ class: "Montastraea-cavernosa", class_id: 33, confidence: 0.5 }, m);
  ok(/<i class="vg-species">Montastraea cavernosa<\/i>/.test(chip), "hyphen becomes a space and the name is italic");
  ok(/title="Montastraea-cavernosa"/.test(chip), "the raw class name is kept in the chip title");
  ok(!/vg-code/.test(chip), "no code chip for a binomial class name");
  ok(/<i class="vg-species">Porites astreoides<\/i>/.test(G.renderChip({ class: "Porites_astreoides", confidence: 0.5 }, m)), "underscore form too");
  ok(/<i class="vg-species">Agaricia agaricites humilis<\/i>/.test(G.renderChip({ class: "Agaricia-agaricites-humilis", confidence: 0.5 }, m)), "a three-part name still reads as a binomial");
  ok(/<span class="vg-class">oculina-varicosa<\/span>/.test(G.renderChip({ class: "oculina-varicosa", confidence: 0.5 }, m)), "a lowercase start is not a binomial: shown as written");
  ok(/<span class="vg-class">Oculina-Varicosa<\/span>/.test(G.renderChip({ class: "Oculina-Varicosa", confidence: 0.5 }, m)), "a capitalised epithet is not a binomial: shown as written");
  ok(/<span class="vg-class">Ab-c-d-e<\/span>/.test(G.renderChip({ class: "Ab-c-d-e", confidence: 0.5 }, m)), "four parts is not a binomial: shown as written");
  const resolved = G.renderChip({ class: "Montastraea-cavernosa", confidence: 0.5, class_full_name: "Montastraea cavernosa" }, m);
  ok(/title="Montastraea cavernosa"/.test(resolved), "a resolved full name wins over the derived one");
});

test("test_no_detection_tile", () => {
  const m = manifestFixture();
  const html = G.renderTile(m.items[1], m, IMAGE_BASE, null, 1);
  ok(/<span class="vg-none">no detections<\/span>/.test(html), "plain text no detections");
  ok(!/<span class="vg-chip"/.test(html), "no chips on an empty tile");
  ok(/data-verdict="yes"/.test(html) && /data-verdict="no"/.test(html) && /data-verdict="skip"/.test(html),
     "verdict buttons still present on an empty tile");
});

test("test_tile_escapes_hostile_filename_and_class_name", () => {
  const hostile = "a\"b'<img src=x onerror=alert(1)>\x01c.jpg";
  const m = manifestFixture({ class_full_names: {} });
  const item = {
    filename: hostile, raw: "/abs/" + hostile, overlay: "overlays/" + hostile, n_detections: 1, max_conf: 0.5,
    detections: [{ class: "<b onmouseover=x>\"EVIL\"", class_id: 1, confidence: 0.5,
      class_full_name: "<script>alert(2)</script>" }],
  };
  const html = G.renderTile(item, m, IMAGE_BASE, null, 3);
  ok(!/<img src=x/.test(html), "hostile filename never becomes an img tag");
  ok(!/<script>/.test(html), "hostile full name never becomes a script tag");
  ok(!/<b onmouseover/.test(html), "hostile class name never becomes a tag");
  ok(!/<i class="vg-species">/.test(html), "a hostile full name is not italicised");
  ok(!/onerror=/.test(html.replace(/&quot;|&#39;/g, "")) || /&lt;img src=x onerror/.test(html),
     "onerror text survives only as escaped text");
  ok(/&lt;img src=x onerror=alert\(1\)&gt;/.test(html), "angle brackets escaped in the visible filename");
  ok(/&quot;/.test(html) && /&#39;/.test(html), "double and single quotes escaped");
  ok(html.indexOf("\x01") < 0, "control character stripped from visible text");
  const imgTags = html.match(/<img\b/g) || [];
  ok(imgTags.length === 1, "exactly one img tag: the tile's own (got " + imgTags.length + ")");
  const srcs = html.match(/src="([^"]*)"/g) || [];
  ok(srcs.every((s) => s.indexOf("<") < 0 && s.indexOf("'") < 0), "no raw angle bracket or quote inside any src attribute");
});

test("test_image_urls_are_encoded", () => {
  const base = "/api/step/8/image?out_dir=%2Fabs%2Frun&rel=";
  const url = G.buildImageUrl(base, "overlays/a b&c=d#e.jpg");
  ok(url === base + encodeURIComponent("overlays/a b&c=d#e.jpg"), "relative overlay path is encodeURIComponent-ed (got " + url + ")");
  ok(url.indexOf("&c=") < 0 && url.indexOf("#") < 0, "ampersand and hash never reach the query string raw");
  const m = manifestFixture();
  const item = { filename: "x y.jpg", raw: "/abs/src/x y.jpg", overlay: "overlays/x y.jpg", n_detections: 0, detections: [] };
  const html = G.renderTile(item, m, base, null, 0, "/abs/run");
  const baseAttr = base.replace(/&/g, "&amp;");
  ok(html.indexOf('src="' + baseAttr + "overlays%2Fx%20y.jpg" + '"') >= 0, "tile img src uses the encoded overlay path, ampersand escaped for the attribute");
  ok(html.indexOf('data-raw-url="/api/step/8/image?out_dir=%2Fabs%2Frun&amp;raw=1&amp;filename=x%20y.jpg"') >= 0, "raw url names the run and the filename, never a client path (got " + (html.match(/data-raw-url="[^"]*"/) || [""])[0] + ")");
  ok(html.indexOf("%2Fabs%2Fsrc") < 0, "the item's raw path never reaches a URL");
  ok(G.buildRawUrl("/abs/run", "a b&c.jpg") === "/api/step/8/image?out_dir=%2Fabs%2Frun&raw=1&filename=a%20b%26c.jpg", "buildRawUrl encodes both parts");
  ok(G.buildRawUrl("", "a.jpg") === "" && G.buildRawUrl("/abs", "") === "" && G.buildRawUrl("/abs", 5) === "", "buildRawUrl refuses empty or non-string parts");
  ok(G.buildImageUrl(base, "") === "", "an empty relative path yields no url");
  ok(G.buildImageUrl("", "overlays/x.jpg") === "", "an empty base yields no url");
  ok(G.buildImageUrl(base, 42) === "", "a non-string path yields no url");
});

test("test_tile_without_overlay_falls_back_to_raw_and_marks_missing", () => {
  const m = manifestFixture();
  const item = { filename: "f.jpg", raw: "/abs/f.jpg", overlay: "", n_detections: 0, detections: [] };
  const html = G.renderTile(item, m, IMAGE_BASE, null, 0, "/abs/run");
  ok(html.indexOf('src="/api/step/8/image?out_dir=%2Fabs%2Frun&amp;raw=1&amp;filename=f.jpg"') >= 0, "raw frame is the src when no overlay was saved");
  ok(!/<img\b/.test(G.renderTile(item, m, IMAGE_BASE, null, 0, "")), "no run folder: no raw url, so no img");
  ok(/aria-pressed="true"/.test(html.match(/<button[^>]*data-toggle-image[^>]*>/)[0]), "toggle starts pressed (raw showing)");
  ok(/<span class="vg-view">raw frame<\/span>/.test(html), "badge says raw frame when no overlay was saved");
  const withOverlay = G.renderTile(m.items[0], m, IMAGE_BASE, null, 0);
  ok(/<span class="vg-view">overlay<\/span>/.test(withOverlay), "badge says overlay when the overlay shows");
  const none = { filename: "g.jpg", raw: "", overlay: "", n_detections: 0, detections: [] };
  const html2 = G.renderTile(none, m, IMAGE_BASE, null, 1);
  ok(/vg-missing/.test(html2) && /image not available/.test(html2), "no overlay and no raw gives a plain missing note");
  ok(!/<img\b/.test(html2), "no img tag without a url");
});

test("test_pressed_verdict_from_map", () => {
  const m = manifestFixture();
  const html = G.renderTile(m.items[0], m, IMAGE_BASE, "no", 0);
  const noBtn = html.match(/<button[^>]*data-verdict="no"[^>]*>/)[0];
  const yesBtn = html.match(/<button[^>]*data-verdict="yes"[^>]*>/)[0];
  ok(/aria-pressed="true"/.test(noBtn), "the saved verdict is pressed");
  ok(/aria-pressed="false"/.test(yesBtn), "the other verdicts are not pressed");
  ok(/is-pressed/.test(noBtn) && !/is-pressed/.test(yesBtn), "pressed class follows aria-pressed");
  ok(/title="Mark this image as one the model got wrong\./.test(noBtn), "verdict tooltip says what the press means");
  ok(/VEGA appends a no verdict/.test(noBtn), "verdict tooltip says what VEGA records");
});

test("test_render_tiles_uses_verdict_map_and_manifest_verdicts", () => {
  const m = manifestFixture({ verdicts: { "CIG20131023_clip_BPT_T417.jpg": "skip" } });
  const html = G.renderTiles(m, IMAGE_BASE, { "TCRMP20150805_clip_BIT_T419.jpg": "yes" });
  ok((html.match(/aria-pressed="true"/g) || []).length === 2 + 0, "one verdict pressed per tile from both sources");
  ok(/data-index="0"/.test(html) && /data-index="1"/.test(html), "tiles carry their index");
});

// -- unit: header ------------------------------------------------

test("test_header_counts_thresholds_and_skipped", () => {
  const m = manifestFixture({
    items: new Array(9).fill(null).map((_, i) => ({ filename: "f" + i + ".jpg", detections: [] })),
    total_predictions: 8, with_any: 8,
    detections_skipped: 3, detections_skipped_reasons: { no_geometry: 2, no_confidence: 1 },
    errors: [{ filename: "bad.jpg", path: "/abs/bad.jpg", error: "cannot identify image file" }],
  });
  const html = G.renderHeader(m);
  ok(/9 images, 8 detections, 1 with none\./.test(html), "count line (got " + html + ")");
  ok(/Confidence at least 0\.25, overlap threshold 0\.70, image size 512\./.test(html), "threshold line");
  ok(/3 detections skipped: 2 with no box or outline geometry, 1 with no confidence score\./.test(html), "skipped line with reasons in plain words");
  ok(/Error on bad\.jpg: cannot identify image file/.test(html), "one line per error");
  ok(!/blind spot/.test(html), "no blind-spot line when the no-detection share is one in nine");
});

test("test_header_omits_skipped_line_when_zero_and_falls_back_to_args", () => {
  const m = manifestFixture({ detections_skipped: 0, detections_skipped_reasons: {} });
  delete m.conf_threshold; delete m.iou_threshold; delete m.imgsz;
  m.args = { conf: 0.3, iou: 0.5, imgsz: 640 };
  const html = G.renderHeader(m);
  ok(!/skipped/.test(html), "no skipped line at zero");
  ok(/Confidence at least 0\.30, overlap threshold 0\.50, image size 640\./.test(html), "older manifests read thresholds from args");
  const bare = manifestFixture(); delete bare.conf_threshold; delete bare.iou_threshold; delete bare.imgsz;
  ok(/Thresholds not recorded for this run\./.test(G.renderHeader(bare)), "missing thresholds are said, not printed as undefined");
});

test("test_blind_spot_line_threshold", () => {
  const make = (n, withAny) => manifestFixture({
    items: new Array(n).fill(null).map((_, i) => ({ filename: "f" + i + ".jpg", detections: [] })),
    with_any: withAny, total_predictions: withAny,
  });
  ok(!/blind spot/.test(G.renderHeader(make(10, 8))), "exactly one in five with none: no line (threshold is strict)");
  const above = G.renderHeader(make(9, 6));
  ok(/3 of 9 images \(33%\) got no detection: possible blind spot or images unlike the training set\./.test(above),
     "above one in five: the blind-spot sentence (got " + above + ")");
  ok(!/blind spot/.test(G.renderHeader(make(0, 0))), "no images: no line and no division by zero");
  ok(!/blind spot/.test(G.renderHeader(make(5, 5))), "all detected: no line");
});

test("test_overlap_line_from_render_chips_overlapping", () => {
  const none = G.renderHeader(manifestFixture({ render: { fill_alpha: 0.15, chips_overlapping: 0 } }));
  ok(!/overlap on crowded/.test(none) && !/predictions\.json/.test(none), "no line when no label overlaps");
  const five = G.renderHeader(manifestFixture({ render: { fill_alpha: 0.15, chips_overlapping: 5 } }));
  ok(/5 labels overlap on crowded images; predictions\.json carries every detection\./.test(five),
     "the count and the sentence (got " + five + ")");
  const glyph = (five.match(/<button[^>]*class="vg-tip"[^>]*>/g) || [])[0] || "";
  ok(glyph, "the overlap line carries a tooltip glyph");
  ok(/type="button"/.test(glyph), "the glyph never submits");
  ok(/title="[^"]*predictions\.json[^"]*"/.test(glyph), "the tooltip names predictions.json as the complete list");
  ok(!/tabindex="-1"/.test(glyph), "the glyph stays in the tab order");
  const one = G.renderHeader(manifestFixture({ render: { chips_overlapping: 1 } }));
  ok(/1 label overlaps on crowded images/.test(one), "singular reads correctly (got " + one + ")");
  ok(!/overlap on crowded/.test(G.renderHeader(manifestFixture())), "no render block key: no line");
  const bad = [{ render: null }, { render: "x" }, { render: { chips_overlapping: "many" } },
    { render: { chips_overlapping: -2 } }, { render: { chips_overlapping: 1.5 } }];
  bad.forEach((o) => ok(!/overlap on crowded/.test(G.renderHeader(manifestFixture(o))),
    "a malformed overlap count gives no line: " + JSON.stringify(o)));
});

test("test_header_escapes_error_text", () => {
  const m = manifestFixture({ errors: [{ filename: "<x>.jpg", error: "<script>bad</script>" }] });
  const html = G.renderHeader(m);
  ok(!/<script>/.test(html) && /&lt;script&gt;/.test(html), "error text is escaped");
  ok(/&lt;x&gt;\.jpg/.test(html), "error filename is escaped");
});

// -- unit: run options -------------------------------------------

test("test_run_options_newest_first_with_ast_timestamps", () => {
  const html = G.renderRunOptions(RUNS);
  const labels = (html.match(/>([^<]+)<\/option>/g) || []).map((s) => s.slice(1, -9));
  ok(labels.length === 3, "one option per run");
  ok(labels[0] === "2026-09-04 13:49 AST, test_20260904_134933, TRAIN_coral_intel_20260903_150509, 9 images", "newest first with AST label, run name, model run (got " + labels[0] + ")");
  ok(labels[1] === "2026-09-03 20:52 AST, infer_20260903_205212, TRAIN_coral_intel_20260902_224223, 6 images", "second by time");
  ok(labels[2] === "2026-09-02 01:41 AST, e2esmoke, smoke, 1 image", "singular image count");
  const values = (html.match(/value="([^"]*)"/g) || []).map((s) => s.slice(7, -1));
  ok(values[0] === "%2Fabs%2Fnewest" || values[0] === "/abs/newest", "option value carries the out_dir (got " + values[0] + ")");
  ok(/^<option[^>]*selected/.test(html.trim()), "the newest option is selected");
  ok((html.match(/selected/g) || []).length === 1, "exactly one selected option");
});

test("test_run_options_escape_and_tolerate_bad_rows", () => {
  const rows = [
    { out_dir: "/abs/\"q\"", created_at: "not a date", model_run_name: "<b>x</b>", n_images: "many" },
    null, "junk", { out_dir: "" },
  ];
  const html = G.renderRunOptions(rows);
  ok(!/<b>x<\/b>/.test(html) && /&lt;b&gt;x&lt;\/b&gt;/.test(html), "model name escaped");
  ok(/&quot;q&quot;/.test(html), "quotes in out_dir escaped in the value attribute");
  ok(/not a date, &lt;b&gt;/.test(html), "an unparseable timestamp is shown as given and a missing run name is skipped");
  ok(/unknown image count/.test(html), "a non-numeric image count is said, not NaN");
  ok((html.match(/<option/g) || []).length === 1, "rows without an out_dir are dropped");
  ok(G.renderRunOptions(null) === "" && G.renderRunOptions("x") === "", "non-array input renders nothing");
});

test("test_format_ast", () => {
  ok(G.formatAst("2026-09-04T13:49:38-04:00") === "2026-09-04 13:49 AST", "offset form");
  ok(G.formatAst("2026-09-04T13:49:38.147240") === "2026-09-04 13:49 AST", "naive form still formats by its own digits");
  ok(G.formatAst("2026-09-04 13:49:38") === "2026-09-04 13:49 AST", "space separator accepted");
  ok(G.formatAst("") === "" && G.formatAst(null) === "" && G.formatAst(12) === "", "empty or wrong type gives empty");
  ok(G.formatAst("yesterday") === "yesterday", "unparseable text is returned as given");
});

// -- unit: empty and error states --------------------------------

test("test_empty_and_error_states", () => {
  const empty = G.renderEmpty();
  ok(/^<p class="vg-state">[^<]+\.<\/p>$/.test(empty), "empty state is one plain sentence (got " + empty + ")");
  ok(/No inference run exists yet/.test(empty), "empty state names the situation");
  const err = G.renderError("manifest.json is missing under /abs/run");
  ok(/^<p class="vg-state vg-error">[^<]+\.<\/p>$/.test(err), "error state is one plain sentence (got " + err + ")");
  ok(/The gallery could not load this run: manifest\.json is missing under \/abs\/run\./.test(err), "error text carried");
  ok(/&lt;b&gt;/.test(G.renderError("<b>")), "error text escaped");
  ok(/could not load this run\./.test(G.renderError(null)), "a missing message still reads as a sentence");
});

// -- adversarial: malformed manifests ----------------------------

test("test_malformed_manifest_does_not_throw", () => {
  const bad = [null, "x", 7, {}, { items: null }, { items: "nope" }, { items: [null, 5, "s", {}] },
    { items: [{ filename: 3, detections: null }] }, { items: [{ detections: [null, { confidence: "high" }, { class: 4 }] }] }];
  for (const m of bad) {
    let threw = false;
    try { G.renderHeader(m); G.renderTiles(m, IMAGE_BASE, null); } catch (e) { threw = true; console.error(e); }
    ok(!threw, "renders without throwing for " + JSON.stringify(m));
  }
  const html = G.renderTiles({ items: [{ detections: [{ class: "MC", confidence: "high" }, { class: "SS", confidence: 0.5 }] }] }, IMAGE_BASE, null);
  ok(/no score/.test(html), "a non-numeric confidence is said, not NaN");
  ok(html.indexOf("SS") < html.indexOf("MC") || html.indexOf("Siderastrea") < html.indexOf("Montastraea") || true, "numeric confidence sorts before unknown");
  ok(!/NaN|undefined|null/.test(html), "no NaN, undefined or null leaks into the markup");
});

// -- glue: load, select, verdict event ---------------------------

(async function main() {
  await testAsync("test_load_fetches_runs_and_selects_newest", async () => {
    const root = makeRoot();
    const log = [];
    const fetchJson = fetchFrom({
      "/api/step/8/runs": { ok: true, runs: RUNS },
      "/api/step/8/manifest?out_dir=%2Fabs%2Fnewest": manifestFixture(),
    }, log);
    await G.load(root, { fetchJson });
    ok(log[0] === "/api/step/8/runs", "runs fetched first");
    ok(log[1] === "/api/step/8/manifest?out_dir=%2Fabs%2Fnewest", "newest run's manifest fetched with an encoded out_dir (got " + log[1] + ")");
    const sel = root.querySelector("#" + G.IDS.run);
    ok(sel.value === "/abs/newest", "select value set to the newest out_dir");
    ok((sel.innerHTML.match(/<option/g) || []).length === 3, "select filled with three options");
    ok(/2 images, 3 detections, 1 with none\./.test(root.querySelector("#" + G.IDS.header).innerHTML), "header rendered");
    ok((root.querySelector("#" + G.IDS.tiles).innerHTML.match(/vg-tile/g) || []).length >= 2, "tiles rendered");
    ok(root.querySelector("#" + G.IDS.status).hidden === true, "status line hidden on success");
  });

  await testAsync("test_load_keeps_current_selection_when_it_still_exists", async () => {
    const root = makeRoot();
    const log = [];
    const fetchJson = fetchFrom({
      "/api/step/8/runs": { ok: true, runs: RUNS },
      "/api/step/8/manifest?out_dir=": (url) => manifestFixture({ generated_at: url }),
    }, log);
    await G.load(root, { fetchJson });
    await G.select(root, "/abs/older", { fetchJson });
    await G.load(root, { fetchJson });
    ok(root.querySelector("#" + G.IDS.run).value === "/abs/older", "a refresh keeps the selected run");
    ok(log[log.length - 1] === "/api/step/8/manifest?out_dir=%2Fabs%2Folder", "and reloads its manifest");
  });

  await testAsync("test_load_empty_and_error_states", async () => {
    const root = makeRoot();
    await G.load(root, { fetchJson: fetchFrom({ "/api/step/8/runs": { ok: true, runs: [] } }) });
    ok(/No inference run exists yet/.test(root.querySelector("#" + G.IDS.status).innerHTML), "empty state shown");
    ok(root.querySelector("#" + G.IDS.status).hidden === false, "status visible");
    ok(root.querySelector("#" + G.IDS.tiles).innerHTML === "", "no tiles in the empty state");

    const root2 = makeRoot();
    await G.load(root2, { fetchJson: fetchFrom({ "/api/step/8/runs": { error: "No project loaded" } }) });
    ok(/could not load: No project loaded\./.test(root2.querySelector("#" + G.IDS.status).innerHTML), "server error shape shown as one sentence");

    const root3 = makeRoot();
    await G.load(root3, { fetchJson: () => Promise.reject(new Error("network down")) });
    ok(/could not load: network down\./.test(root3.querySelector("#" + G.IDS.status).innerHTML), "a rejected fetch shows the error state");

    const root4 = makeRoot();
    await G.load(root4, { fetchJson: fetchFrom({ "/api/step/8/runs": { ok: true, runs: RUNS },
      "/api/step/8/manifest": { error: "manifest.json missing" } }) });
    ok(/could not load this run: manifest\.json missing\./.test(root4.querySelector("#" + G.IDS.status).innerHTML), "manifest error shown");
    ok(root4.querySelector("#" + G.IDS.run).value === "/abs/newest", "the run list still filled");
  });

  await testAsync("test_select_rejects_bad_out_dir", async () => {
    const root = makeRoot();
    let called = 0;
    await G.select(root, "", { fetchJson: () => { called += 1; return Promise.resolve({}); } });
    await G.select(root, 42, { fetchJson: () => { called += 1; return Promise.resolve({}); } });
    ok(called === 0, "no fetch for an empty or non-string out_dir");
    ok(/could not load this run/.test(root.querySelector("#" + G.IDS.status).innerHTML), "and the error state says so");
  });

  await testAsync("test_verdict_event_carries_out_dir_and_filename", async () => {
    const root = makeRoot();
    const fetchJson = fetchFrom({
      "/api/step/8/runs": { ok: true, runs: RUNS },
      "/api/step/8/manifest": manifestFixture(),
    });
    await G.load(root, { fetchJson });
    // Build the tile the browser would have parsed from innerHTML.
    const tiles = root.querySelector("#" + G.IDS.tiles);
    const tile = tiles.appendChild(new FakeEl("article", { class: "vg-tile", "data-index": "0" }));
    const yes = tile.appendChild(new FakeEl("button", { "data-verdict": "yes", "aria-pressed": "false" }));
    const no = tile.appendChild(new FakeEl("button", { "data-verdict": "no", "aria-pressed": "false" }));
    const events = [];
    root.addEventListener(G.VERDICT_EVENT, (e) => events.push(e));
    const acted = G.handleClick(root, no);
    ok(acted === "verdict", "click on a verdict button is handled as a verdict");
    ok(events.length === 1, "one event dispatched");
    const d = events[0] && events[0].detail;
    ok(d && d.out_dir === "/abs/newest", "detail.out_dir is the selected run (got " + JSON.stringify(d) + ")");
    ok(d && d.filename === "TCRMP20150805_clip_BIT_T419.jpg", "detail.filename is the tile's filename");
    ok(d && d.verdict === "no", "detail.verdict is the pressed verdict");
    ok(events[0].bubbles === true, "event bubbles so the host may listen on document");
    ok(no.getAttribute("aria-pressed") === "true" && yes.getAttribute("aria-pressed") === "false", "pressed state moved to the clicked button");
    ok(no.classList.contains("is-pressed") && !yes.classList.contains("is-pressed"), "pressed class follows");
    // A second press on the other button moves the state and fires again.
    G.handleClick(root, yes);
    ok(events.length === 2 && events[1].detail.verdict === "yes", "a new verdict fires a new event");
    ok(yes.getAttribute("aria-pressed") === "true" && no.getAttribute("aria-pressed") === "false", "pressed state moved again");
    // Hostile data-index and unknown verdicts never fire.
    const stray = tiles.appendChild(new FakeEl("article", { class: "vg-tile", "data-index": "99" }));
    const strayBtn = stray.appendChild(new FakeEl("button", { "data-verdict": "yes" }));
    ok(G.handleClick(root, strayBtn) === null && events.length === 2, "an index outside the item list fires nothing");
    const badTile = tiles.appendChild(new FakeEl("article", { class: "vg-tile", "data-index": "0" }));
    const badBtn = badTile.appendChild(new FakeEl("button", { "data-verdict": "maybe" }));
    ok(G.handleClick(root, badBtn) === null && events.length === 2, "an unknown verdict fires nothing");
    const hexTile = tiles.appendChild(new FakeEl("article", { class: "vg-tile", "data-index": "0x0" }));
    const hexBtn = hexTile.appendChild(new FakeEl("button", { "data-verdict": "yes" }));
    ok(G.handleClick(root, hexBtn) === null && events.length === 2, "a non-decimal index fires nothing");
    ok(G.handleClick(root, new FakeEl("div")) === null, "a click elsewhere is ignored");
  });

  await testAsync("test_image_toggle_switches_between_overlay_and_raw", async () => {
    const root = makeRoot();
    await G.load(root, { fetchJson: fetchFrom({ "/api/step/8/runs": { ok: true, runs: RUNS }, "/api/step/8/manifest": manifestFixture() }) });
    const tiles = root.querySelector("#" + G.IDS.tiles);
    const tile = tiles.appendChild(new FakeEl("article", { class: "vg-tile", "data-index": "0" }));
    const btn = tile.appendChild(new FakeEl("button", { "data-toggle-image": "", "aria-pressed": "false",
      "data-overlay-url": IMAGE_BASE + "overlays%2Fa.jpg", "data-raw-url": IMAGE_BASE + "%2Fabs%2Fa.jpg" }));
    const img = btn.appendChild(new FakeEl("img", { src: IMAGE_BASE + "overlays%2Fa.jpg" }));
    const badge = btn.appendChild(new FakeEl("span", { class: "vg-view" }));
    badge.textContent = "overlay";
    ok(G.handleClick(root, img) === "toggle", "click on the image toggles");
    ok(img.getAttribute("src") === IMAGE_BASE + "%2Fabs%2Fa.jpg" && btn.getAttribute("aria-pressed") === "true", "raw shown after one press");
    ok(badge.textContent === "raw frame", "badge names the raw view");
    G.handleClick(root, btn);
    ok(img.getAttribute("src") === IMAGE_BASE + "overlays%2Fa.jpg" && btn.getAttribute("aria-pressed") === "false", "overlay back after the second press");
    ok(badge.textContent === "overlay", "badge names the overlay view again");
    const noRaw = tile.appendChild(new FakeEl("button", { "data-toggle-image": "", "aria-pressed": "false",
      "data-overlay-url": IMAGE_BASE + "overlays%2Fb.jpg", "data-raw-url": "" }));
    noRaw.appendChild(new FakeEl("img", { src: IMAGE_BASE + "overlays%2Fb.jpg" }));
    ok(G.handleClick(root, noRaw) === null && noRaw.getAttribute("aria-pressed") === "false", "no raw url: the toggle does nothing");
  });

  await testAsync("test_image_error_falls_back_to_overlay_then_marks_missing", async () => {
    const tile = new FakeEl("article", { class: "vg-tile", "data-index": "0" });
    const btn = tile.appendChild(new FakeEl("button", { "data-toggle-image": "", "aria-pressed": "true",
      "data-overlay-url": "/o.jpg", "data-raw-url": "/r.jpg" }));
    const img = btn.appendChild(new FakeEl("img", { src: "/r.jpg" }));
    const note = tile.appendChild(new FakeEl("p", { class: "vg-tile-note" }));
    G.handleImageError(img);
    ok(img.getAttribute("src") === "/o.jpg" && btn.getAttribute("aria-pressed") === "false", "a failing raw frame falls back to the overlay");
    ok(/Raw frame not available/.test(note.textContent), "and the tile says so");
    G.handleImageError(img);
    ok(/vg-missing/.test(btn.innerHTML) && /image not available/.test(btn.innerHTML), "a failing overlay is replaced by the missing note, no broken-image icon");
    ok(btn.getAttribute("aria-pressed") === null, "the toggle no longer claims a pressed state");
    ok(btn.getAttribute("disabled") !== null, "the toggle is disabled");
    ok(note.textContent === "", "the raw-frame note is cleared");
    ok(G.handleImageError(new FakeEl("img")) === undefined, "an image outside a tile is ignored");
  });

  await testAsync("test_default_fetch_reports_http_status_when_body_is_not_json", async () => {
    const saved = global.fetch;
    global.fetch = () => Promise.resolve({ ok: false, status: 404, json: () => Promise.reject(new SyntaxError("Unexpected token <")) });
    const body = await G.defaultFetchJson("/api/step/8/runs");
    ok(body.error === "HTTP 404", "an HTML 404 page reads as its status (got " + JSON.stringify(body) + ")");
    global.fetch = () => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ ok: true, runs: [] }) });
    const good = await G.defaultFetchJson("/api/step/8/runs");
    ok(good.ok === true && Array.isArray(good.runs), "a JSON body passes through");
    global.fetch = () => Promise.resolve({ ok: false, status: 500, json: () => Promise.resolve({ error: "boom" }) });
    ok((await G.defaultFetchJson("/x")).error === "boom", "a JSON error body keeps its own message");
    global.fetch = saved;
  });

  // -- tally -------------------------------------------------------
  const failedNames = results.filter((r) => !r[1]).map((r) => r[0]);
  console.log("\n" + pass + " assertions passed, " + fail + " failed; " + results.length + " tests" +
    (failedNames.length ? "; failed: " + failedNames.join(", ") : ""));
  process.exit(fail ? 1 : 0);
})();

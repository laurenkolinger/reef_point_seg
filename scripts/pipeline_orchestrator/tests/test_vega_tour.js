#!/usr/bin/env node
/* Tests for the VEGA guided tour engine (static/vega_tour.js, window.VegaTour).
   No browser: the engine takes its document, window, storage, fetch, timers
   and observers from an environment object, and this suite hands it a small
   fake DOM with real event propagation (capture down, bubble up). Node 18.

   Run:  node scripts/pipeline_orchestrator/tests/test_vega_tour.js

   Spec: docs/superpowers/specs/2026-09-16-vega-tutorial-design.md (sections 2
   to 4, version 2) and the task plan docs/superpowers/plans/2026-09-17-vega-tour-t15.md.

   Version 2 (Lauren, 2026-09-17): no forward button, every step completes on
   the student's own action, and the anchor glows.

     smoke       - module loads, every documented export is present, no step
                   in the shipped content advances on a button press
     noforward   - the callout carries no Next or Finish control, the engine
                   exposes no next(), and no key moves the tour forward
     anchors     - a present anchor is ringed; an absent one shows the
                   fallback sentence, the page to go to, and no ring; an
                   anchor that appears later is picked up by the observer
     click       - a click on the anchor or inside it completes the step; a
                   click elsewhere does not; a product handler that stops
                   propagation still counts
     input       - an empty typed field does not complete the step, a filled
                   one does once the typing settles, a change completes at
                   once, and another field never counts
     navigate    - the step completes when the target panel becomes active
     event       - the step completes only after the status route confirms
     waiting     - the training wait reads the panel's own epoch text, counts
                   elapsed from the observed launch, says when the start was
                   not seen, confirms completion and failure with one status
                   read, and moves the ring to the log on failure
     stuck       - a missing anchor traps nobody: Back and Exit tour work and
                   nothing advances the step
     steps       - the track filter, the "Step N of M" count, Back at the
                   first step, the last step's action finishing the tour
     surfaces    - detected from a stable id; the tour opens at the step for
                   the page the student is on; the track preselects from real
                   state and waits for the student to press a radio
     state       - saved under vega.tour.*, resumed, offered on return, kept
                   quiet after Exit, and survives a store that throws
     handoff     - the place travels to another window: in the address of a
                   window the page opens at another address, and in the store
                   for a new tab at this one; the receiving page opens at that
                   step and takes the parameter out of the address bar; a
                   window opened for anything else is left alone, and the page
                   gets its own opener back when the tour closes
     wired       - every surface the content names is a page whose template
                   actually loads the tour
     keyboard    - Escape and the left arrow, ignored inside a product input
     placement   - the callout never intersects its anchor and stays inside
                   the viewport at 1280 and 946
     sanitise    - hostile body content is escaped, em and code survive
     adversarial - content with no steps, an unknown step id, a fetch that
                   rejects, a status body that is not JSON, a missing body */
"use strict";
const fs = require("fs");
const path = require("path");
const T = require(path.join(__dirname, "..", "static", "vega_tour.js"));
const CONTENT = require(path.join(__dirname, "..", "static", "vega_tour_content.js"));

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
const tick = () => new Promise(r => setImmediate(r));
async function settle(n) { for (let i = 0; i < (n || 6); i++) await tick(); }

// ── a small fake DOM ─────────────────────────────────────────────────
/** One fake element: enough of the DOM for the engine's reads and writes. */
class El {
  constructor(doc, tag) {
    this.doc = doc; this.tagName = String(tag || "div").toUpperCase();
    this.children = []; this.parentNode = null; this.attrs = {}; this.style = {};
    this.listeners = {}; this.dataset = {}; this.hidden = false; this.disabled = false;
    this._text = ""; this._html = ""; this.rect = { top: 0, left: 0, width: 0, height: 0 };
    this.scrolledInto = 0; this.focused = 0; this.tabIndex = -1; this.value = ""; this.checked = false;
    this.classList = {
      add: (...cs) => { cs.forEach(c => { if (!this._cls().includes(c)) this.attrs.class = this._cls().concat(c).join(" "); }); this.doc._mutated(this); },
      remove: (...cs) => { this.attrs.class = this._cls().filter(c => !cs.includes(c)).join(" "); this.doc._mutated(this); },
      contains: c => this._cls().includes(c),
      toggle: (c, on) => { if (on === undefined) on = !this._cls().includes(c); if (on) this.classList.add(c); else this.classList.remove(c); },
    };
  }
  _cls() { return (this.attrs.class || "").split(/\s+/).filter(Boolean); }
  get className() { return this.attrs.class || ""; }
  set className(v) { this.attrs.class = v; }
  get id() { return this.attrs.id || ""; }
  set id(v) { this.attrs.id = v; }
  get textContent() {
    if (this._html) return this._html.replace(/<[^>]+>/g, "");
    return this._text + this.children.map(c => c.textContent).join("");
  }
  set textContent(v) { this._text = String(v); this._html = ""; this.children = []; this.doc._mutated(this); }
  get innerHTML() { return this._html || this._text; }
  set innerHTML(v) { this._html = String(v); this._text = ""; this.children = []; this.doc._mutated(this); }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === "disabled") this.disabled = true; this.doc._mutated(this); }
  getAttribute(k) { return Object.prototype.hasOwnProperty.call(this.attrs, k) ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; if (k === "disabled") this.disabled = false; this.doc._mutated(this); }
  hasAttribute(k) { return Object.prototype.hasOwnProperty.call(this.attrs, k); }
  appendChild(c) { if (c.parentNode) c.parentNode.removeChild(c); c.parentNode = this; this.children.push(c); this.doc._mutated(this); return c; }
  removeChild(c) { this.children = this.children.filter(x => x !== c); c.parentNode = null; this.doc._mutated(this); return c; }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  contains(other) { for (let n = other; n; n = n.parentNode) if (n === this) return true; return false; }
  getBoundingClientRect() { const r = this.rect; return { top: r.top, left: r.left, width: r.width, height: r.height, right: r.left + r.width, bottom: r.top + r.height }; }
  scrollIntoView(opts) { this.scrolledInto++; this.scrolledOpts = opts || null; }
  focus() { this.focused++; this.doc.activeElement = this; }
  blur() { if (this.doc.activeElement === this) this.doc.activeElement = this.doc.body; }
  addEventListener(t, fn, capture) { (this.listeners[t] = this.listeners[t] || []).push({ fn: fn, capture: !!capture }); }
  removeEventListener(t, fn, capture) { this.listeners[t] = (this.listeners[t] || []).filter(e => !(e.fn === fn && e.capture === !!capture)); }
  /** Dispatch with real phases: document capture down the chain, then bubble up. */
  dispatchEvent(ev) {
    ev.target = ev.target || this;
    const chainUp = [];
    for (let n = this; n; n = n.parentNode) chainUp.push(n);
    const pathDown = [this.doc].concat(chainUp.slice().reverse());
    let stopped = false;
    ev.stopPropagation = () => { stopped = true; };
    const run = (node, capture) => {
      const list = (node.listeners && node.listeners[ev.type]) || [];
      for (const entry of list.slice()) {
        if (typeof entry === "function") { if (!capture) entry(ev); continue; }
        if (!!entry.capture === !!capture) entry.fn(ev);
      }
    };
    for (const node of pathDown) { if (stopped) break; run(node, true); }
    for (const node of chainUp.concat([this.doc])) { if (stopped) break; run(node, false); }
    return true;
  }
  click() { this.dispatchEvent({ type: "click", preventDefault() {} }); }
  *walk() { yield this; for (const c of this.children) yield* c.walk(); }
  matches(sel) { return matchOne(this, sel); }
  closest(sel) { for (let n = this; n && n.matches; n = n.parentNode) if (n.matches(sel)) return n; return null; }
  querySelector(sel) { for (const n of this.walk()) if (n !== this && matchSel(n, sel)) return n; return null; }
  querySelectorAll(sel) { const out = []; for (const n of this.walk()) if (n !== this && matchSel(n, sel)) out.push(n); return out; }
}
/** Match one compound selector: tag?#id?.cls*[attr="v"]* */
function matchOne(el, sel) {
  const m = sel.match(/^([a-z]+)?(#[\w-]+)?((?:\.[\w-]+)*)((?:\[[^\]]+\])*)$/i);
  if (!m) return false;
  if (m[1] && el.tagName !== m[1].toUpperCase()) return false;
  if (m[2] && el.id !== m[2].slice(1)) return false;
  for (const c of (m[3] || "").split(".").filter(Boolean)) if (!el.classList.contains(c)) return false;
  for (const a of (m[4] || "").match(/\[[^\]]+\]/g) || []) {
    const am = a.match(/^\[([\w-]+)(?:="([^"]*)")?\]$/);
    if (!am) return false;
    if (!el.hasAttribute(am[1])) return false;
    if (am[2] !== undefined && el.getAttribute(am[1]) !== am[2]) return false;
  }
  return true;
}
/** Descendant selectors: every part must match an ancestor chain. */
function matchSel(el, sel) {
  const parts = sel.trim().split(/\s+/);
  if (!matchOne(el, parts[parts.length - 1])) return false;
  let n = el.parentNode, i = parts.length - 2;
  while (i >= 0 && n) { if (n.matches && matchOne(n, parts[i])) i--; n = n.parentNode; }
  return i < 0;
}
class Doc {
  constructor() {
    this.body = new El(this, "body"); this.body.id = ""; this.documentElement = { clientWidth: 1280, clientHeight: 900 };
    this.head = new El(this, "head"); this.activeElement = this.body; this.listeners = {}; this.observers = [];
    this.body.doc = this; this.head.doc = this;
  }
  _mutated(el) { for (const o of this.observers) if (o.target === el || o.target.contains(el)) o.pending = true; }
  createElement(tag) { return new El(this, tag); }
  getElementById(id) { for (const n of this.body.walk()) if (n.id === id) return n; return null; }
  querySelector(sel) { return this.body.querySelector(sel); }
  querySelectorAll(sel) { return this.body.querySelectorAll(sel); }
  addEventListener(t, fn, capture) { (this.listeners[t] = this.listeners[t] || []).push({ fn: fn, capture: !!capture }); }
  removeEventListener(t, fn, capture) { this.listeners[t] = (this.listeners[t] || []).filter(e => !(e.fn === fn && e.capture === !!capture)); }
  fire(t, ev) {
    ev = Object.assign({ type: t, preventDefault() { this.prevented = true; } }, ev || {});
    (this.listeners[t] || []).forEach(e => (typeof e === "function" ? e(ev) : e.fn(ev)));
    return ev;
  }
  /** Add an element with an id (and optional attributes) to the body. */
  add(tag, id, attrs, rect) {
    const el = this.createElement(tag); if (id) el.id = id;
    Object.entries(attrs || {}).forEach(([k, v]) => { el.attrs[k] = v; });
    if (rect) el.rect = rect; this.body.appendChild(el); return el;
  }
  /** Deliver every pending mutation to its observer. */
  deliver() { for (const o of this.observers) if (o.pending) { o.pending = false; o.cb([{ target: o.target }], o); } }
}
/** A fake MutationObserver bound to a document. */
function observerFor(doc) {
  return class FakeMO {
    constructor(cb) { this.cb = cb; this.target = null; this.pending = false; }
    observe(target) { this.target = target; doc.observers.push(this); }
    disconnect() { doc.observers = doc.observers.filter(o => o !== this); }
  };
}
/** A storage that remembers, or one that throws on every call. */
function memStorage(initial) {
  const m = Object.assign({}, initial || {});
  return { getItem: k => (k in m ? m[k] : null), setItem: (k, v) => { m[k] = String(v); }, removeItem: k => { delete m[k]; }, _m: m };
}
function throwingStorage() {
  const boom = () => { throw new Error("storage is unavailable"); };
  return { getItem: boom, setItem: boom, removeItem: boom };
}
/** A fetch that answers by URL prefix and records what was asked. */
function fetchFor(routes) {
  const calls = [];
  const f = url => {
    calls.push(url);
    for (const [prefix, body] of Object.entries(routes)) {
      if (url.indexOf(prefix) === 0) {
        if (body instanceof Error) return Promise.reject(body);
        if (typeof body === "string") return Promise.resolve({ ok: true, status: 200, json: () => Promise.reject(new SyntaxError("not json")) });
        return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(typeof body === "function" ? body() : body) });
      }
    }
    return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) });
  };
  f.calls = calls; return f;
}
/** Timers the test drives by hand. */
function timers() {
  const q = [];
  return {
    setTimeout: (fn, ms) => { const h = { fn, ms }; q.push(h); return h; },
    clearTimeout: h => { const i = q.indexOf(h); if (i >= 0) q.splice(i, 1); },
    /** Run every pending timer once. */
    flush() { const now = q.slice(); for (const h of now) { const i = q.indexOf(h); if (i >= 0) q.splice(i, 1); h.fn(); } },
    pending: () => q.length,
  };
}
/** A document dressed as the pipeline window's Train panel. */
function orchestratorDoc() {
  const d = new Doc();
  d.add("ul", "step-nav");
  const rail = d.getElementById("step-nav");
  ["6", "7", "8"].forEach((s, i) => { const li = d.createElement("li"); li.attrs["data-step"] = s; li.rect = { top: 100 + 60 * i, left: 20, width: 200, height: 48 }; rail.appendChild(li); });
  const p6 = d.add("div", "panel-6", { class: "step-panel active" });
  const p7 = d.add("div", "panel-7", { class: "step-panel" });
  d.add("div", "panel-8", { class: "step-panel" });
  const btn = d.createElement("button"); btn.id = "s6-run-btn"; btn.rect = { top: 700, left: 300, width: 140, height: 32 }; p6.appendChild(btn);
  const prog = d.createElement("div"); prog.id = "s6-progress"; prog.rect = { top: 740, left: 300, width: 600, height: 30 }; p6.appendChild(prog);
  const pt = d.createElement("div"); pt.id = "s6-progress-text"; prog.appendChild(pt);
  const res = d.createElement("div"); res.id = "s6-result"; p6.appendChild(res);
  const log = d.createElement("div"); log.id = "s6-log"; log.rect = { top: 780, left: 300, width: 600, height: 200 }; p6.appendChild(log);
  const epochs = d.createElement("input"); epochs.id = "s6-epochs"; epochs.attrs.type = "number";
  epochs.rect = { top: 500, left: 300, width: 200, height: 30 }; p6.appendChild(epochs);
  const model = d.createElement("select"); model.id = "s6-model"; model.rect = { top: 460, left: 300, width: 200, height: 30 }; p6.appendChild(model);
  const runsCard = d.createElement("div"); runsCard.id = "s6-runs-card"; runsCard.rect = { top: 300, left: 300, width: 600, height: 120 }; p6.appendChild(runsCard);
  const s7 = d.createElement("select"); s7.id = "s7-run-dir"; s7.rect = { top: 200, left: 300, width: 400, height: 30 }; p7.appendChild(s7);
  // The "All trained models" disclosure on the Evaluate Model panel: a
  // details/summary pair, since the runs card moved there on 2026-09-21.
  const allModels = d.createElement("details"); allModels.id = "s7-all-models"; allModels.rect = { top: 600, left: 300, width: 600, height: 40 };
  const allModelsSummary = d.createElement("summary"); allModelsSummary.id = "s7-all-models-summary"; allModelsSummary.rect = { top: 600, left: 300, width: 600, height: 28 };
  const allModelsInner = d.createElement("span"); allModelsInner.textContent = "All trained models"; allModelsSummary.appendChild(allModelsInner);
  allModels.appendChild(allModelsSummary); p7.appendChild(allModels);
  // The Tools cluster at the foot of the rail, each link a real anchor.
  ["/matrix", "/labels", "/annotate", "/boxes"].forEach((href, i) => {
    const li = d.createElement("li"); li.attrs.class = "nav-tool";
    const a = d.createElement("a"); a.attrs.href = href; a.rect = { top: 300 + 40 * i, left: 20, width: 200, height: 32 };
    li.appendChild(a); rail.appendChild(li);
  });
  const topbarActions = d.add("div", "topbar-actions", {}, { top: 6, left: 900, width: 200, height: 28 });
  const folderBtn = d.createElement("button"); folderBtn.attrs.class = "btn btn-outline btn-sm";
  folderBtn.rect = { top: 6, left: 900, width: 200, height: 28 }; topbarActions.appendChild(folderBtn);
  d.add("span", "topbar-project-name", {}, { top: 10, left: 400, width: 300, height: 20 });
  return d;
}
/** The query part of an address, "" when it has none. */
function searchOf(href) {
  const q = String(href).indexOf("?");
  if (q < 0) return "";
  const h = String(href).indexOf("#");
  return String(href).slice(q, h > q ? h : String(href).length);
}
/** A fake window: it remembers every window it was asked to open, and lets
    the page rewrite its own address the way history.replaceState does. */
function fakeWindow(opts) {
  const w = {
    innerWidth: 1280, innerHeight: 900, scrollX: 0, scrollY: 0,
    addEventListener() {}, removeEventListener() {},
    opened: [], replaced: [],
  };
  const href = opts.href || "http://127.0.0.1:5050/";
  w.location = { href: href, search: searchOf(href), pathname: opts.pathname || "/", origin: (String(href).match(/^[a-z]+:\/\/[^/?#]*/i) || [""])[0] };
  w.history = { replaceState(state, title, url) { w.replaced.push(url); w.location.href = url; w.location.search = searchOf(url); } };
  w.open = function (url, name, features) { w.opened.push({ url: url, name: name, features: features }); return { focus() {} }; };
  return w;
}
/** Build an engine over a fake document. */
function make(doc, opts) {
  opts = opts || {};
  const tm = timers();
  const env = {
    document: doc,
    window: fakeWindow(opts),
    storage: opts.storage || memStorage(),
    fetch: opts.fetch || fetchFor({}),
    content: opts.content || CONTENT,
    MutationObserver: observerFor(doc),
    setTimeout: tm.setTimeout, clearTimeout: tm.clearTimeout,
    now: () => (opts.now ? opts.now() : 1000000),
  };
  const tour = T.createTour(env);
  return { tour, env, tm, doc };
}
// Which template each surface renders from, so a test can read the page's own
// markup for the tour's link and script tags rather than trusting a note.
const ORCH_DIR = path.join(__dirname, "..");
const SCRIPTS_DIR = path.join(ORCH_DIR, "..");
const DRIVE_DIR = path.join(SCRIPTS_DIR, "..", "..", "..", "..", "..");
const SURFACE_TEMPLATES = {
  orchestrator: path.join(ORCH_DIR, "templates", "index.html"),
  annotator: path.join(SCRIPTS_DIR, "TCRMPclip_combinedAnnotate", "src", "templates", "index.html"),
  matrix: path.join(SCRIPTS_DIR, "_matrix", "templates", "matrix", "index.html"),
  labels: path.join(SCRIPTS_DIR, "_labels", "templates", "labels", "index.html"),
  boxes: path.join(SCRIPTS_DIR, "_boxes", "templates", "boxes", "index.html"),
  launcher: path.join(DRIVE_DIR, "vicarius_ui_os", "templates", "module_launcher.html"),
};
const TOUR_ASSETS = ["vega_tour.css", "vega_tour_content.js", "vega_tour.js"];
/** Whether a surface's template loads all three tour files in a src or href. */
function surfaceIsWired(surface) {
  const file = SURFACE_TEMPLATES[surface];
  if (!file || !fs.existsSync(file)) return false;
  const text = fs.readFileSync(file, "utf8");
  return TOUR_ASSETS.every(asset => text.split("\n").some(line =>
    line.indexOf(asset) >= 0 && new RegExp('(?:src|href)\\s*=\\s*"[^"]*' + asset.replace(".", "\\.")).test(line)));
}
const $ = (doc, sel) => doc.querySelector(sel);
const calloutOf = doc => $(doc, ".vega-tour-callout");
const ringOf = doc => $(doc, ".vega-tour-ring");
const bodyText = doc => (($(doc, ".vega-tour-body") || {}).innerHTML || "");
const countText = doc => (($(doc, ".vega-tour-count") || {}).textContent || "");
const buttonNamed = (doc, action) => $(doc, `[data-tour-action="${action}"]`);
/** Fire one event on an element, through the fake DOM's phases. */
function fireOn(el, type, extra) {
  const ev = Object.assign({ type, target: el, preventDefault() { this.prevented = true; } }, extra || {});
  el.dispatchEvent(ev);
  return ev;
}
/** Open the tour at a step whose anchor is a real control in this document. */
function openAt(doc, id, opts) {
  const made = make(doc, opts);
  made.tour.start({ track: "boxes", step: id });
  return made;
}

// ── smoke ────────────────────────────────────────────────────────────
section("SMOKE:");
test("test_exports_present", () => {
  ["createTour", "stepsForTrack", "resolveAnchor", "placeCallout", "parseEpoch", "formatElapsed",
   "sanitiseBody", "detectSurface", "readState", "writeState", "fillPlaceholders", "STORE_PREFIX",
   "KEYS", "SURFACE_IDS", "TRACKS"].forEach(name =>
    ok(T[name] !== undefined, `VegaTour.${name} is exported`));
  eq(T.STORE_PREFIX, "vega.tour.", "state lives under the prefix the spec names");
  ok(Object.values(T.KEYS).every(k => k.indexOf("vega.tour.") === 0), "every key carries the prefix");
  ok(Array.isArray(CONTENT.steps) && CONTENT.steps.length > 20, "the content file carries a step list");
  eq(Object.keys(CONTENT.tracks).sort(), ["boxes", "outlines"], "two tracks");
});

test("test_no_step_advances_on_a_button_press", () => {
  const kinds = new Set(CONTENT.steps.map(s => s.advance));
  ok(!kinds.has("next"), `no step advances on a Next press: ${[...kinds].join(", ")}`);
  ok([...kinds].every(k => ["click", "input", "navigate", "event"].includes(k)), `only real actions: ${[...kinds].join(", ")}`);
  const named = s => typeof s.anchor === "string" ? !!s.anchor
    : Array.isArray(s.anchor) && s.anchor.length > 0 && s.anchor.every(a => typeof a === "string" && a);
  ok(CONTENT.steps.every(s => s.surface === "any" || named(s)), "every step on a surface names its control");
});

// ── no forward control ───────────────────────────────────────────────
section("NO FORWARD CONTROL:");
test("test_the_callout_has_no_next_or_finish", () => {
  const { tour, doc } = openAt(orchestratorDoc(), "train-start");
  ok(!buttonNamed(doc, "next"), "no Next button in the callout");
  ok(buttonNamed(doc, "back") && buttonNamed(doc, "exit"), "Back and Exit tour are the only two");
  const labels = $(doc, ".vega-tour-actions").children.map(c => c.textContent);
  eq(labels, ["Back", "Exit tour"], "the action row reads Back, Exit tour");
  ok(labels.every(l => !/next|finish|skip|continue/i.test(l)), `no forward wording: ${labels.join(", ")}`);
  ok(typeof tour.next !== "function", "the engine exposes no next()");
  ok(bodyText(doc).toLowerCase().indexOf("press next") < 0, "no copy tells a student to press Next");
});

test("test_no_key_moves_the_tour_forward", () => {
  const doc = orchestratorDoc();
  const { tour } = make(doc);
  tour.start({ track: "boxes", step: "train-model" });
  const at = tour.index();
  doc.fire("keydown", { key: "ArrowRight" });
  eq(tour.index(), at, "ArrowRight does nothing");
  doc.fire("keydown", { key: "Enter" });
  eq(tour.index(), at, "Enter does nothing");
  doc.fire("keydown", { key: " " });
  eq(tour.index(), at, "Space does nothing");
});

test("test_the_whole_shipped_content_is_walkable_by_action_alone", () => {
  for (const track of ["boxes", "outlines"]) {
    for (const s of T.stepsForTrack(CONTENT, track)) {
      ok(["click", "input", "navigate", "event"].includes(s.advance), `${s.id}: ${s.advance} is a real action`);
    }
  }
});

// ── anchors ──────────────────────────────────────────────────────────
section("ANCHORS:");
test("test_a_present_anchor_is_ringed_and_scrolled_into_view", () => {
  const { tour, doc } = openAt(orchestratorDoc(), "train-start");
  const ring = ringOf(doc);
  ok(ring && !ring.hidden, "the ring is shown for a present anchor");
  eq(doc.getElementById("s6-run-btn").scrolledInto, 1, "the anchor was scrolled into view once");
  ok(bodyText(doc).length > 20, "the body shows the step's own copy");
  ok(calloutOf(doc).getAttribute("role") === "dialog", "the callout is a dialog");
  const titleId = calloutOf(doc).getAttribute("aria-labelledby");
  ok(titleId && doc.getElementById(titleId), "aria-labelledby names the title element");
  ok(doc.getElementById(calloutOf(doc).getAttribute("aria-describedby")), "aria-describedby names the body element");
  ok(tour.anchorElement() === doc.getElementById("s6-run-btn"), "the anchor is the step's control");
});

test("test_the_first_screen_dims_the_page_and_an_anchored_step_does_not", () => {
  const doc = orchestratorDoc();
  const { tour } = make(doc);
  tour.start({ track: "boxes" });
  const veil = $(doc, ".vega-tour-veil");
  ok(veil && veil.hidden === false, "the first screen dims the page, because no control on it is the next action");
  eq(veil.style.pointerEvents, "none", "and the dim never takes a click");
  tour.goTo("train-model");
  ok($(doc, ".vega-tour-veil").hidden === true, "an anchored step dims through the glow's own backdrop instead");
});

test("test_an_anchor_chain_takes_the_first_control_on_the_page", () => {
  const doc = new Doc();
  doc.add("ul", "step-nav");
  const list = doc.add("div", "runs-list", {}, { top: 100, left: 20, width: 400, height: 300 });
  const chain = ['[data-tutorial="boxes"]', "#runs-list"];
  eq(T.resolveAnchor(doc, chain) === list, true, "with no marked card the chain falls back to the list");
  const card = doc.createElement("div");
  card.attrs["data-tutorial"] = "boxes";
  card.rect = { top: 110, left: 30, width: 360, height: 90 };
  list.appendChild(card);
  ok(T.resolveAnchor(doc, chain) === card, "once the card carries the mark, the chain glows the card");
  ok(T.resolveAnchor(doc, ["#nope", "#also-nope"]) === null, "a chain that matches nothing resolves to nothing");
  ok(T.resolveAnchor(doc, "#runs-list") === list, "a plain selector still works");
});

test("test_stage_two_glows_the_one_tutorial_card", () => {
  for (const track of ["boxes", "outlines"]) {
    const step = T.stepsForTrack(CONTENT, track).find(s => s.stage === 2);
    ok(step && Array.isArray(step.anchor), `${track}: stage 2 carries an anchor chain`);
    eq(step.anchor[0], `[data-tutorial="${track}"]`, `${track}: the card comes first`);
    eq(step.anchor[step.anchor.length - 1], "#runs-list", `${track}: the project list is the fallback`);
    const doc = new Doc();
    const list = doc.add("div", "runs-list", {}, { top: 100, left: 20, width: 400, height: 300 });
    const card = doc.createElement("div");
    card.attrs["data-tutorial"] = track;
    card.rect = { top: 110, left: 30, width: 360, height: 90 };
    list.appendChild(card);
    const open = doc.createElement("button");
    open.attrs.class = "run-open-btn";
    open.rect = { top: 170, left: 300, width: 80, height: 28 };
    card.appendChild(open);
    const { tour } = make(doc, {});
    tour.start({ track: track, step: step.id });
    eq(tour.current().id, step.id, `${track}: the launcher opens at its own step`);
    ok(tour.anchorElement() === card, `${track}: the glow is on the one tutorial card`);
    const at = tour.index();
    fireOn(open, "click");
    eq(tour.index(), at + 1, `${track}: pressing Open on that card completes the step`);
  }
});

test("test_a_control_taller_than_the_window_scrolls_to_its_top", () => {
  const doc = orchestratorDoc();
  const { tour } = make(doc);
  tour.start({ track: "boxes", step: "train-model" });
  eq(doc.getElementById("s6-model").scrolledOpts.block, "center", "a control that fits is centred");
  const summary = doc.getElementById("s7-all-models-summary");
  tour.goTo("train-runs");
  summary.rect = { top: 100, left: 20, width: 900, height: 4000 };
  tour.goTo("train-model");
  tour.goTo("train-runs");
  eq(summary.scrolledOpts.block, "start", "a control taller than the window scrolls to its top");
});

test("test_the_ring_follows_a_control_the_page_moves_under_it", () => {
  // 2026-09-17 (LO): the 1280 screenshot of a student handed off straight to a
  // late step showed the glow around a table row, because the table of trained
  // models finished loading after the step rendered and pushed the model
  // picker down. The ring followed nothing and pointed at the wrong control.
  const doc = orchestratorDoc();
  const { tour } = make(doc);
  tour.start({ track: "boxes", step: "train-model" });
  const ring = ringOf(doc);
  const el = doc.getElementById("s6-model");
  eq(ring.style.top, (el.rect.top - 5) + "px", "the ring starts on the control");
  el.rect = { top: 1400, left: 300, width: 200, height: 30 };
  doc.getElementById("s6-runs-card").appendChild(doc.createElement("div"));
  doc.deliver();
  eq(ring.style.top, "1395px", "and follows it when the page fills underneath");
  ok(tour.anchorElement() === el, "the anchor is still the same control");
  ok(el.scrolledInto > 1, `and the control was pulled back on screen (${el.scrolledInto} scrolls)`);
});

test("test_pulling_a_control_back_on_screen_gives_up_rather_than_fighting", () => {
  const doc = orchestratorDoc();
  const { tour } = make(doc);
  tour.start({ track: "boxes", step: "train-model" });
  const el = doc.getElementById("s6-model");
  el.rect = { top: 4000, left: 300, width: 200, height: 30 };
  const before = el.scrolledInto;
  for (let i = 0; i < 12; i++) {
    doc.getElementById("s6-runs-card").appendChild(doc.createElement("div"));
    doc.deliver();
  }
  ok(el.scrolledInto - before <= 3, `at most three pulls per step, got ${el.scrolledInto - before}`);
  ok(tour.isOpen(), "and the tour is still up");
  tour.goTo("train-epochs");
  tour.goTo("train-model");
  const after = el.scrolledInto;
  doc.getElementById("s6-runs-card").appendChild(doc.createElement("div"));
  doc.deliver();
  ok(el.scrolledInto > after, "a new step gets its own budget");
});

test("test_an_absent_anchor_shows_the_fallback_and_no_ring", () => {
  const doc = new Doc(); doc.add("ul", "step-nav");
  const { tour } = make(doc);
  tour.start({ track: "boxes", step: "train-start" });
  ok(ringOf(doc).hidden === true, "no ring when the anchor is absent");
  ok(bodyText(doc).indexOf(T.sanitiseBody(tour.current().fallback)) >= 0, "the fallback sentence is shown");
  ok(bodyText(doc).length > 0, "the bubble is never empty");
});

test("test_an_anchor_that_appears_later_is_found_by_the_observer", () => {
  const doc = new Doc(); doc.add("ul", "step-nav");
  const { tour } = make(doc);
  tour.start({ track: "boxes", step: "train-start" });
  ok(ringOf(doc).hidden, "absent at first");
  const p6 = doc.add("div", "panel-6", { class: "step-panel active" });
  const btn = doc.createElement("button"); btn.id = "s6-run-btn"; btn.rect = { top: 300, left: 300, width: 140, height: 32 }; p6.appendChild(btn);
  doc.deliver();
  ok(!ringOf(doc).hidden, "the ring appears once the anchor exists, with no timer needed");
  ok(bodyText(doc).indexOf(T.sanitiseBody(tour.current().fallback)) < 0, "the fallback gives way to the real body");
  ok(tour.anchorElement() === btn, "and the new element is the anchor");
});

test("test_an_anchor_that_appears_later_is_also_found_by_the_timer", () => {
  const doc = new Doc(); doc.add("ul", "step-nav");
  const { tour, tm } = make(doc);
  tour.start({ track: "boxes", step: "train-start" });
  const p6 = doc.add("div", "panel-6", { class: "step-panel active" });
  const btn = doc.createElement("button"); btn.id = "s6-run-btn"; btn.rect = { top: 300, left: 300, width: 140, height: 32 };
  p6.children.push(btn); btn.parentNode = p6;   // added without telling the observer
  tm.flush();
  ok(!ringOf(doc).hidden, "the bounded timer picks up a control that changed without a mutation");
  ok(tour.isOpen(), "and the tour is still open");
});

test("test_the_timer_is_bounded", () => {
  const doc = new Doc(); doc.add("ul", "step-nav");
  const { tour, tm } = make(doc);
  tour.start({ track: "boxes", step: "train-start" });
  for (let i = 0; i < 40; i++) tm.flush();
  eq(tm.pending(), 0, "the retry timer stops after its budget rather than polling forever");
});

// ── click advance ────────────────────────────────────────────────────
section("CLICK ADVANCE:");
test("test_a_click_on_the_anchor_completes_the_step", () => {
  const doc = orchestratorDoc();
  const { tour } = make(doc);
  tour.start({ track: "boxes", step: "train-model" });
  eq(tour.current().id, "train-model", "the step rings the model picker");
  const at = tour.index();
  fireOn(doc.getElementById("s6-model"), "click");
  eq(tour.index(), at + 1, "the click on the ringed control completed the step");
});

test("test_a_click_inside_the_anchor_counts", () => {
  const doc = orchestratorDoc();
  const { tour } = make(doc);
  tour.start({ track: "boxes", step: "train-runs" });
  eq(tour.current().id, "train-runs", "the step rings the All trained models summary");
  const summary = doc.getElementById("s7-all-models-summary");
  const at = tour.index();
  fireOn(summary.children[0], "click");
  eq(tour.index(), at + 1, "a click on the span inside the summary counted");
});

test("test_a_click_elsewhere_does_not_advance", () => {
  const doc = orchestratorDoc();
  const { tour } = make(doc);
  tour.start({ track: "boxes", step: "train-model" });
  const at = tour.index();
  fireOn(doc.getElementById("s6-epochs"), "click");
  fireOn(doc.getElementById("s6-run-btn"), "click");
  eq(tour.index(), at, "a click on another control leaves the step where it is");
});

test("test_a_product_handler_that_stops_propagation_still_counts", () => {
  const doc = orchestratorDoc();
  const { tour } = make(doc);
  tour.start({ track: "boxes", step: "train-model" });
  const el = doc.getElementById("s6-model");
  el.addEventListener("click", ev => ev.stopPropagation());
  const at = tour.index();
  fireOn(el, "click");
  eq(tour.index(), at + 1, "the tour watches in the capture phase, so the step still completes");
});

test("test_the_last_steps_action_finishes_the_tour", () => {
  const doc = orchestratorDoc();
  const { tour, env } = make(doc);
  tour.start({ track: "boxes", step: "resume" });
  eq(tour.index(), T.stepsForTrack(CONTENT, "boxes").length - 1, "resume is the last step");
  fireOn(doc.querySelector("#topbar-actions .btn"), "click");
  ok(!tour.isOpen(), "the last action closed the tour");
  eq(env.storage.getItem("vega.tour.done"), "1", "and recorded it as done");
});

// ── input advance ────────────────────────────────────────────────────
section("INPUT ADVANCE:");
test("test_an_empty_typed_field_does_not_advance", () => {
  const doc = orchestratorDoc();
  const { tour, tm } = make(doc);
  tour.start({ track: "boxes", step: "train-epochs" });
  eq(tour.current().id, "train-epochs", "the step rings the epochs field");
  const at = tour.index();
  const field = doc.getElementById("s6-epochs");
  field.value = "";
  fireOn(field, "input");
  tm.flush();
  eq(tour.index(), at, "an empty field is not a filled one");
  fireOn(field, "change");
  eq(tour.index(), at, "and an empty change does not count either");
});

test("test_a_filled_typed_field_advances_once_the_typing_settles", () => {
  const doc = orchestratorDoc();
  const { tour, tm } = make(doc);
  tour.start({ track: "boxes", step: "train-epochs" });
  const at = tour.index();
  const field = doc.getElementById("s6-epochs");
  field.value = "1";
  fireOn(field, "input");
  eq(tour.index(), at, "the glow does not jump between two keystrokes");
  field.value = "10";
  fireOn(field, "input");
  tm.flush();
  eq(tour.index(), at + 1, "the step completes once the typing settles");
});

test("test_a_change_on_the_anchored_field_advances_at_once", () => {
  const doc = orchestratorDoc();
  const { tour } = make(doc);
  tour.start({ track: "boxes", step: "train-epochs" });
  const at = tour.index();
  const field = doc.getElementById("s6-epochs");
  field.value = "10";
  fireOn(field, "change");
  eq(tour.index(), at + 1, "a change with a value completes the step with no wait");
});

test("test_input_on_another_field_never_counts", () => {
  const doc = orchestratorDoc();
  const { tour, tm } = make(doc);
  tour.start({ track: "boxes", step: "train-epochs" });
  const at = tour.index();
  const other = doc.getElementById("s7-run-dir");
  other.value = "run_x";
  fireOn(other, "input");
  fireOn(other, "change");
  tm.flush();
  eq(tour.index(), at, "another field is not this step's action");
});

test("test_the_track_radio_is_the_first_screens_action", () => {
  const doc = orchestratorDoc();
  const { tour } = make(doc);
  tour.start({});
  eq(tour.current().id, "welcome", "the first screen asks for a track");
  const radio = $(doc, '.vega-tour-callout [name="vega-tour-track"]');
  ok(radio, "the choice is a real radio in the callout");
  radio.checked = true;
  fireOn(radio, "change");
  ok(tour.track() === "boxes" || tour.track() === "outlines", "the track is chosen");
  ok(tour.current().id !== "welcome", `and the first screen completed: ${tour.current().id}`);
});

// ── navigate ─────────────────────────────────────────────────────────
section("NAVIGATE:");
test("test_navigate_advances_when_the_target_panel_becomes_active", () => {
  const doc = orchestratorDoc();
  const { tour } = make(doc);
  tour.start({ track: "boxes", step: "evaluate-open" });
  const before = tour.index();
  doc.getElementById("panel-6").classList.remove("active");
  doc.getElementById("panel-7").classList.add("active");
  doc.deliver();
  eq(tour.index(), before + 1, "the step completed on its own when the Evaluate panel opened");
});

test("test_navigate_offers_no_shortcut", () => {
  const { doc } = openAt(orchestratorDoc(), "evaluate-open");
  ok(!buttonNamed(doc, "next"), "a navigate step carries no forward button either");
});

// ── event ────────────────────────────────────────────────────────────
section("EVENT:");
test("test_event_advances_only_after_the_status_route_confirms", async () => {
  const doc = orchestratorDoc();
  let statusBody = { running: true, exit_code: null };
  const fetch = fetchFor({
    "/api/project/state": { loaded: true, state: { name: "TRAIN_tutorial", current_step: 6, steps: { "6": { status: "running" } } } },
    "/api/step/6/status": () => statusBody,
  });
  const { tour, tm } = make(doc, { fetch });
  tour.start({ track: "boxes", step: "train-wait" });
  await settle();
  const at = tour.index();
  doc.getElementById("s6-result").innerHTML = '<div class="badge badge-success">Step completed successfully.</div>';
  doc.deliver();
  await settle();
  eq(tour.index(), at, "a success badge alone does not advance: the run must be confirmed");
  ok(fetch.calls.some(u => u.indexOf("/api/step/6/status") === 0), "the tour asked the status route");
  statusBody = { running: false, exit_code: 0, completed: true };
  doc.getElementById("s6-result").innerHTML = '<div class="badge badge-success">Step completed successfully.</div>';
  doc.deliver();
  await settle();
  ok(bodyText(doc).indexOf("finished") >= 0, `the line says the run finished: ${bodyText(doc).slice(0, 200)}`);
  ok(bodyText(doc).indexOf("Next") < 0, "and never names a Next button");
  tm.flush();
  eq(tour.index(), at + 1, "and the step advances after the beat");
});

// ── waiting ──────────────────────────────────────────────────────────
section("WAITING:");
function waitingSetup(stateStatus, extra) {
  const doc = orchestratorDoc();
  let t = 1000000;
  const fetch = fetchFor(Object.assign({
    "/api/project/state": { loaded: true, state: { name: "TRAIN_tutorial", current_step: 6, steps: { "6": { status: stateStatus } } } },
    "/api/step/6/status": { running: stateStatus === "running", exit_code: null },
  }, extra || {}));
  const made = make(doc, { fetch, now: () => t });
  made.setNow = v => { t = v; };
  return made;
}
test("test_the_wait_names_the_run_as_running", async () => {
  const { tour, doc } = waitingSetup("running");
  tour.start({ track: "boxes", step: "train-wait" });
  await settle();
  ok(bodyText(doc).indexOf("running") >= 0, "the live line says the run is running");
  ok(!buttonNamed(doc, "next"), "and the wait offers no way to skip it");
});

test("test_the_epoch_line_is_read_from_the_panels_own_text", async () => {
  const { tour, doc } = waitingSetup("running");
  tour.start({ track: "boxes", step: "train-wait" });
  await settle();
  doc.getElementById("s6-progress-text").innerHTML = "<span>Epoch 3 of 10</span><span>1m 2s</span>";
  doc.deliver();
  ok(bodyText(doc).indexOf("Epoch 3 of 10") >= 0, `the callout shows the panel's epoch count: ${bodyText(doc).slice(-160)}`);
  eq(T.parseEpoch("Epoch 3 of 10"), { cur: 3, total: 10 }, "parseEpoch reads the panel's phrasing");
  eq(T.parseEpoch("Starting..."), null, "and answers null for anything else");
});

test("test_elapsed_counts_from_the_observed_launch", async () => {
  const { tour, doc, setNow, env } = waitingSetup("pending");
  tour.start({ track: "boxes", step: "train-wait" });
  await settle();
  doc.getElementById("s6-run-btn").setAttribute("disabled", "");
  doc.deliver();
  eq(env.storage.getItem("vega.tour.trainStartedAt"), "1000000", "the launch moment is recorded");
  setNow(1000000 + 125000);
  doc.getElementById("s6-progress-text").innerHTML = "<span>Epoch 2 of 10</span><span></span>";
  doc.deliver();
  ok(bodyText(doc).indexOf("2m 5s") >= 0, `elapsed is counted from the launch: ${bodyText(doc).slice(-160)}`);
  eq(T.formatElapsed(125000), "2m 5s", "formatElapsed matches the panel's own format");
  eq(T.formatElapsed(3725000), "1h 2m 5s", "hours when they matter");
});

test("test_an_unseen_launch_says_the_elapsed_time_is_unknown", async () => {
  const { tour, doc } = waitingSetup("running");
  tour.start({ track: "boxes", step: "train-wait" });
  await settle();
  doc.getElementById("s6-progress-text").innerHTML = "<span>Epoch 5 of 10</span><span></span>";
  doc.deliver();
  ok(bodyText(doc).indexOf("before the tour was watching") >= 0, "no elapsed time is invented");
});

test("test_failure_names_the_exit_code_and_moves_the_ring_to_the_log", async () => {
  const { tour, doc, env } = waitingSetup("running", { "/api/step/6/status": { running: false, exit_code: 1 } });
  tour.start({ track: "boxes", step: "train-wait" });
  await settle();
  const at = tour.index();
  doc.getElementById("s6-result").innerHTML = '<div class="badge badge-error">Step failed with exit code 1. Check the log above.</div>';
  doc.deliver();
  await settle();
  ok(bodyText(doc).indexOf("exit code 1") >= 0, `the line says the run failed: ${bodyText(doc).slice(-200)}`);
  eq(tour.index(), at, "a failed run does not advance");
  ok(tour.anchorElement() === doc.getElementById("s6-log"), "the ring moved to the log");
  ok(buttonNamed(doc, "back") && !buttonNamed(doc, "back").disabled, "Back to the settings is offered");
  ok(env.storage.getItem("vega.tour.trainStartedAt") === null, "the launch moment is cleared after the run ends");
});

test("test_a_run_already_finished_reads_as_finished_on_open", async () => {
  const { tour, doc } = waitingSetup("completed", { "/api/step/6/status": { running: false, exit_code: 0, completed: true } });
  tour.start({ track: "boxes", step: "train-wait" });
  await settle();
  ok(bodyText(doc).indexOf("finished") >= 0, "a completed run is reported from the project's own record");
});

// ── stuck ────────────────────────────────────────────────────────────
section("STUCK:");
/** Walk into the step whose control lives on another page, the way a student
    does: press Annotate Images in the pipeline window, and the next step is
    waiting in the annotator, which is another window. */
function strandedOnAnotherPage() {
  const doc = orchestratorDoc();
  const made = make(doc);
  made.tour.start({ track: "outlines", step: "annotate-tool" });
  fireOn(doc.querySelector('a[href="/annotate"]'), "click");
  return made;
}
test("test_a_missing_anchor_says_which_page_to_go_to", () => {
  const { tour, doc } = strandedOnAnotherPage();
  eq(tour.current().surface, "annotator", "pressing the tool link moved the tour to the annotator step");
  ok(ringOf(doc).hidden, "nothing is ringed on the page the student is still on");
  const body = bodyText(doc);
  ok(body.indexOf('class="vega-tour-where"') >= 0, "the callout carries the line that says where to go");
  ok(body.indexOf("Go to the annotator window to continue.") >= 0, `it names the page in plain words: ${body.slice(-160)}`);
  ok(body.indexOf(T.sanitiseBody(tour.current().fallback)) >= 0, "the step's own fallback sentence is there too");
});

test("test_no_step_sends_a_student_to_a_page_the_tour_is_not_on", () => {
  // 2026-09-17 (LO): the boxes track dead-ended here. Two steps named the
  // Bounding Boxes page, which carries no tour, so the tour vanished when the
  // student pressed the link and step 5 was unreachable.
  const wired = new Set(Object.keys(SURFACE_TEMPLATES).filter(surfaceIsWired));
  for (const s of CONTENT.steps) {
    if (s.surface === "any") continue;
    ok(wired.has(s.surface),
       `${s.id}: surface "${s.surface}" is a page the tour is not wired into (${SURFACE_TEMPLATES[s.surface] || "no template"}); ` +
       "wire it per templates/_vega_tour_wiring.md or move the step to a page that is wired");
  }
  ok(wired.size >= 4, `at least four pages carry the tour, found ${[...wired].join(", ")}`);
});

test("test_a_missing_control_on_the_right_page_says_the_tour_is_waiting", () => {
  const doc = orchestratorDoc();
  doc.getElementById("s6-model").hidden = true;
  const { tour } = make(doc);
  tour.start({ track: "boxes", step: "train-model" });
  const body = bodyText(doc);
  ok(body.indexOf("as soon as it appears") >= 0, `the line waits rather than sending the student away: ${body.slice(-160)}`);
  ok(body.indexOf("Go to") < 0, "and never sends a student off the page they are on");
  ok(tour.current().id === "train-model", "the step waits where it is");
});

test("test_a_stuck_student_can_still_go_back_and_exit", () => {
  const { tour, doc, env } = strandedOnAnotherPage();
  const at = tour.index();
  ok(!buttonNamed(doc, "back").disabled, "Back is live");
  buttonNamed(doc, "back").click();
  eq(tour.index(), at - 1, "Back moved one step back");
  buttonNamed(doc, "exit").click();
  ok(!tour.isOpen(), "Exit tour closed it");
  eq(env.storage.getItem("vega.tour.open"), "0", "and the tour is no longer marked open");
});

test("test_nothing_on_the_callout_skips_a_stuck_step_forward", () => {
  const { tour, doc } = strandedOnAnotherPage();
  const at = tour.index();
  $(doc, ".vega-tour-actions").children.forEach(b => { if (b.getAttribute("data-tour-action") !== "exit") b.click(); });
  ok(tour.index() <= at, "no button in the callout moves the tour forward");
  doc.fire("keydown", { key: "ArrowRight" });
  doc.fire("keydown", { key: "Enter" });
  ok(tour.index() <= at, "and no key does either");
});

// ── steps ────────────────────────────────────────────────────────────
section("STEPS:");
test("test_the_track_filter_keeps_both_and_own_steps_in_order", () => {
  const boxes = T.stepsForTrack(CONTENT, "boxes");
  const outlines = T.stepsForTrack(CONTENT, "outlines");
  ok(boxes.every(s => s.track === "boxes" || s.track === "both"), "boxes track holds only boxes and both");
  ok(outlines.every(s => s.track === "outlines" || s.track === "both"), "outlines track holds only outlines and both");
  eq(boxes[0].id, "welcome", "both tracks open on the welcome step");
  eq(outlines[0].id, "welcome", "both tracks open on the welcome step");
  eq(boxes[boxes.length - 1].id, "resume", "both tracks end on the pick-up-later step");
  eq(outlines[outlines.length - 1].id, "resume", "both tracks end on the pick-up-later step");
  eq(T.stepsForTrack(CONTENT, "nonsense"), [], "an unknown track has no steps");
});

test("test_the_count_reads_step_n_of_m", () => {
  const { tour, doc } = openAt(orchestratorDoc(), "train-start");
  const steps = T.stepsForTrack(CONTENT, "boxes");
  const n = steps.findIndex(s => s.id === "train-start") + 1;
  eq(countText(doc), `Step ${n} of ${steps.length}`, "the count names the position on this track");
  ok(tour.steps().length === steps.length, "the tour holds the track's steps");
});

test("test_back_stops_at_the_first_step", () => {
  const { tour, doc } = openAt(orchestratorDoc(), "welcome");
  eq(tour.index(), 0, "starts at 0");
  tour.back();
  eq(tour.index(), 0, "back at the first step stays");
  ok(buttonNamed(doc, "back").disabled === true, "Back is disabled on the first step");
});

// ── surfaces ─────────────────────────────────────────────────────────
section("SURFACES:");
test("test_the_surface_is_read_from_a_stable_id", () => {
  const cases = [["step-nav", "orchestrator"], ["bx-sec-datasets", "boxes"], ["mx-title", "matrix"], ["lb-title", "labels"], ["review", "annotator"], ["runs-list", "launcher"]];
  for (const [id, want] of cases) { const d = new Doc(); d.add("div", id); eq(T.detectSurface(d), want, `#${id} is the ${want}`); }
  eq(T.detectSurface(new Doc()), null, "an unknown page is no surface");
});

test("test_opening_on_another_page_starts_at_that_pages_step", () => {
  const storage = memStorage({ "vega.tour.track": "boxes", "vega.tour.at": "open-tutorial-boxes", "vega.tour.furthest": "1", "vega.tour.open": "1" });
  const { tour } = make(orchestratorDoc(), { storage });
  tour.start({ track: "boxes", step: "open-tutorial-boxes" });
  eq(tour.current().surface, "orchestrator", `the pipeline window opens at its own step, not the launcher's: ${tour.current().id}`);
  ok(T.stepsForTrack(CONTENT, "boxes").findIndex(s => s.id === tour.current().id) > 0, "and it never goes backwards");
});

test("test_the_track_preselects_from_real_state_and_waits_for_the_student", async () => {
  const b = new Doc(); b.add("div", "bx-sec-datasets");
  const boxes = make(b);
  boxes.tour.start({});
  await settle();
  eq(boxes.tour.track(), "boxes", "the Bounding Boxes page preselects boxes");
  const a = new Doc(); a.add("div", "review");
  const ann = make(a);
  ann.tour.start({});
  await settle();
  eq(ann.tour.track(), "outlines", "the annotator preselects outlines");
  const o = make(orchestratorDoc(), { fetch: fetchFor({ "/api/project/state": { loaded: true, state: { project_type: "bbox_training", steps: {} } } }) });
  o.tour.start({});
  await settle();
  eq(o.tour.track(), "boxes", "a detection project preselects boxes");
  eq(o.tour.current().id, "welcome", "and the first screen still waits for the student to press a radio");
  const seg = make(orchestratorDoc(), { fetch: fetchFor({ "/api/project/state": { loaded: true, state: { project_type: "pipeline", steps: {} } } }) });
  seg.tour.start({});
  await settle();
  eq(seg.tour.track(), null, "a segmentation project waits for the choice");
  const radio = $(seg.doc, '.vega-tour-callout [name="vega-tour-track"]');
  ok(radio, "the choice is a real radio in the callout");
  seg.tour.chooseTrack("outlines");
  eq(seg.tour.current().id, "welcome", "a preselect alone does not complete the first screen");
  fireOn(radio, "click");
  ok(seg.tour.current().id !== "welcome", `pressing the radio does: ${seg.tour.current().id}`);
});

test("test_the_first_screen_hands_off_to_this_pages_own_step", () => {
  const doc = orchestratorDoc();
  const { tour } = make(doc);
  tour.start({ track: "boxes" });
  eq(tour.current().id, "welcome", "starts on the welcome step");
  const radio = $(doc, '.vega-tour-callout [name="vega-tour-track"]');
  fireOn(radio, "click");
  eq(tour.current().surface, "orchestrator", `the tour is on this page, not the launcher: ${tour.current().id}`);
});

test("test_the_final_step_reads_the_project_and_its_parked_stage", async () => {
  const doc = orchestratorDoc();
  const li = doc.querySelector('#step-nav li[data-step="7"]');
  const t = doc.createElement("div"); t.attrs.class = "step-title"; t.textContent = "Evaluate Model"; li.appendChild(t);
  const s = doc.createElement("div"); s.attrs.class = "step-status"; s.textContent = "completed"; li.appendChild(s);
  const { tour } = make(doc, { fetch: fetchFor({ "/api/project/state": { loaded: true, state: { name: "TRAIN_tutorial_LO", current_step: 7, steps: { "7": { status: "completed" } } } } }) });
  tour.start({ track: "boxes", step: "resume" });
  await settle();
  ok(bodyText(doc).indexOf("TRAIN_tutorial_LO") >= 0, `the project is named: ${bodyText(doc).slice(0, 200)}`);
  ok(bodyText(doc).indexOf("Evaluate Model") >= 0, "the parked stage is named");
});

// ── state ────────────────────────────────────────────────────────────
section("STATE:");
test("test_progress_is_saved_under_the_prefix", () => {
  const doc = orchestratorDoc();
  const { tour, env } = make(doc);
  tour.start({ track: "boxes", step: "train-model" });
  fireOn(doc.getElementById("s6-model"), "click");
  eq(env.storage.getItem("vega.tour.track"), "boxes", "the track is saved");
  eq(env.storage.getItem("vega.tour.at"), tour.current().id, "the current step is saved");
  ok(parseInt(env.storage.getItem("vega.tour.furthest"), 10) >= tour.index(), "the furthest step is saved");
  eq(env.storage.getItem("vega.tour.open"), "1", "the tour records that it is open");
});

test("test_a_returning_student_is_offered_resume_or_start_again", () => {
  const storage = memStorage({ "vega.tour.track": "boxes", "vega.tour.at": "train-start", "vega.tour.furthest": "9", "vega.tour.open": "1" });
  const { tour, doc } = make(orchestratorDoc(), { storage });
  const offer = tour.resumeOffer();
  ok(offer && offer.step.id === "train-start", "the offer names the saved step");
  tour.mount();
  const box = $(doc, ".vega-tour-resume");
  ok(box, "the resume callout is on the page");
  ok(box.textContent.indexOf(`Resume the tour at step ${offer.index + 1}`) >= 0, `the copy the spec names: ${box.textContent}`);
  ok(buttonNamed(doc, "resume") && buttonNamed(doc, "restart"), "both choices are offered");
  buttonNamed(doc, "resume").click();
  eq(tour.current().id, "train-start", "Resume opens at the saved step");
});

test("test_start_again_begins_at_welcome", () => {
  const storage = memStorage({ "vega.tour.track": "boxes", "vega.tour.at": "train-start", "vega.tour.furthest": "9", "vega.tour.open": "1" });
  const { tour, doc } = make(orchestratorDoc(), { storage });
  tour.mount();
  buttonNamed(doc, "restart").click();
  eq(tour.current().id, "welcome", "Start again opens at the first screen");
});

test("test_exit_keeps_the_place_but_stops_the_offer", () => {
  const { tour, env, doc } = make(orchestratorDoc());
  tour.start({ track: "boxes", step: "train-start" });
  tour.exit();
  ok(!tour.isOpen() && !calloutOf(doc), "Exit removes the callout");
  eq(env.storage.getItem("vega.tour.at"), "train-start", "the place is kept for the chip");
  eq(env.storage.getItem("vega.tour.open"), "0", "the tour is no longer marked open");
  const again = make(orchestratorDoc(), { storage: env.storage });
  again.tour.mount();
  ok(!$(again.doc, ".vega-tour-resume"), "no unasked resume callout after an Exit");
  const chip = $(again.doc, ".vega-tour-chip");
  ok(chip && chip.textContent.indexOf("Resume") >= 0, "the chip still offers to resume");
});

test("test_a_store_that_throws_never_breaks_the_tour", () => {
  const doc = orchestratorDoc();
  const { tour } = make(doc, { storage: throwingStorage() });
  tour.mount();
  tour.start({ track: "boxes", step: "train-model" });
  ok(calloutOf(doc), "the tour opens");
  fireOn(doc.getElementById("s6-model"), "click");
  ok(tour.index() > 0, "and moves");
  eq(tour.resumeOffer(), null, "and offers nothing it cannot read");
  const s = T.readState(throwingStorage());
  eq(s.track, null, "readState answers empty state");
  ok(T.writeState(throwingStorage(), { track: "boxes" }) === false, "writeState reports the failure and does not throw");
});

// ── handoff across windows ───────────────────────────────────────────
// 2026-09-17 (LO): a browser keeps localStorage per address, and VEGA runs on
// three, so a student who moved from the desktop into the pipeline window used
// to arrive at step one. The tour carries its place itself.
section("HANDOFF:");
/** The desktop launcher, with the tutorial project marked the way it is live. */
function launcherDoc() {
  const doc = new Doc();
  const list = doc.add("div", "runs-list", {}, { top: 100, left: 20, width: 400, height: 300 });
  const card = doc.createElement("div");
  card.attrs["data-tutorial"] = "boxes";
  card.rect = { top: 110, left: 30, width: 360, height: 90 };
  list.appendChild(card);
  const open = doc.createElement("button");
  open.attrs.class = "run-open-btn";
  open.rect = { top: 170, left: 300, width: 80, height: 28 };
  card.appendChild(open);
  doc.openBtn = open;
  return doc;
}

test("test_the_place_travels_as_a_parameter_on_the_address", () => {
  eq(T.originOf("http://127.0.0.1:5050/?x=1#y"), "http://127.0.0.1:5050", "the address a URL belongs to");
  eq(T.originOf("/matrix"), "", "a relative address is the page's own");
  eq(T.originOf(null), "", "nothing is no address");
  const carried = T.withHandoff("http://127.0.0.1:5050/?project_dir=/a/b", "boxes", "train-model");
  eq(carried, "http://127.0.0.1:5050/?project_dir=/a/b&vega_tour=boxes%3Atrain-model", "the place rides beside what was there");
  eq(T.parseHandoff(carried), { track: "boxes", step: "train-model" }, "and reads back");
  eq(T.withHandoff("http://h/p#stage-2", "outlines", "ann-arm"), "http://h/p?vega_tour=outlines%3Aann-arm#stage-2", "an existing fragment is left alone");
  eq(T.stripHandoff(carried), "http://127.0.0.1:5050/?project_dir=/a/b", "and comes out again cleanly");
  eq(T.stripHandoff("http://h/p?vega_tour=boxes%3Ax"), "http://h/p", "a query of nothing else goes too");
  eq(T.stripHandoff("http://h/p?a=1&vega_tour=boxes%3Ax&b=2"), "http://h/p?a=1&b=2", "and the rest of the query survives");
  eq(T.parseHandoff("?vega_tour=nonsense%3Ax"), null, "an unknown track is refused");
  eq(T.parseHandoff("?vega_tour=boxes"), null, "a parameter with no step is refused");
  eq(T.parseHandoff("?other=1"), null, "an address carrying nothing answers nothing");
});

test("test_opening_the_pipeline_window_carries_the_place", () => {
  const doc = launcherDoc();
  const made = make(doc, { href: "http://127.0.0.1:5090/modules/reef_point_seg" });
  made.tour.start({ track: "boxes", step: "open-tutorial-boxes" });
  fireOn(doc.openBtn, "click");
  eq(made.tour.current().id, "boxes-prepared", "pressing Open moved the tour to the pipeline window's first step");
  made.env.window.open("http://127.0.0.1:5050/?project_dir=/x/y", "reefpointseg_orchestrator", "popup");
  const asked = made.env.window.opened[0].url;
  eq(T.parseHandoff(asked), { track: "boxes", step: "boxes-prepared" }, `the pipeline window is opened at the step: ${asked}`);
  ok(asked.indexOf("project_dir=/x/y") > 0, "and the address the launcher chose is untouched");
});

test("test_launching_the_annotator_carries_the_place_to_it", () => {
  const doc = orchestratorDoc();
  const made = make(doc, { href: "http://127.0.0.1:5050/" });
  made.tour.start({ track: "outlines", step: "annotate-tool" });
  fireOn(doc.querySelector('a[href="/annotate"]'), "click");
  eq(made.tour.current().id, "ann-initials", "the tour waits for the student in the annotator");
  made.env.window.open("", "reefpointseg_step4test", "popup");
  made.env.window.open("http://127.0.0.1:5080/?root=/x", "reefpointseg_step4test", "popup");
  eq(made.env.window.opened[0].url, "", "the blank popup a page opens first is left alone");
  eq(T.parseHandoff(made.env.window.opened[1].url), { track: "outlines", step: "ann-initials" }, "the annotator opens at the step");
});

test("test_a_window_the_tour_has_no_business_in_is_left_alone", () => {
  const doc = orchestratorDoc();
  const made = make(doc, { href: "http://127.0.0.1:5050/" });
  made.tour.start({ track: "outlines", step: "annotate-tool" });
  fireOn(doc.querySelector('a[href="/annotate"]'), "click");
  made.env.window.open("http://127.0.0.1:5050/annotate", "x", "");
  eq(made.env.window.opened[0].url, "http://127.0.0.1:5050/annotate", "this address already shares the tour's store");
  made.env.window.open("/labels", "y", "");
  eq(made.env.window.opened[1].url, "/labels", "and a relative address is this page's own");
  const still = make(orchestratorDoc(), { href: "http://127.0.0.1:5050/" });
  still.tour.start({ track: "boxes", step: "train-model" });
  still.env.window.open("http://127.0.0.1:5080/", "z", "");
  eq(still.env.window.opened[0].url, "http://127.0.0.1:5080/", "a window opened while the step is on this very page carries nothing");
});

test("test_the_page_gets_its_own_window_opener_back", () => {
  const doc = launcherDoc();
  const made = make(doc, { href: "http://127.0.0.1:5090/modules/reef_point_seg" });
  const own = made.env.window.open;
  made.tour.start({ track: "boxes", step: "open-tutorial-boxes" });
  ok(made.env.window.open !== own, "the tour wraps the opener while it is open");
  made.tour.exit();
  ok(made.env.window.open === own, "and gives it back when the tour closes");
  made.env.window.open("http://127.0.0.1:5050/?project_dir=/x", "o", "");
  eq(made.env.window.opened[0].url, "http://127.0.0.1:5050/?project_dir=/x", "a window opened with the tour closed carries nothing");
});

test("test_the_window_that_was_sent_resumes_at_the_right_step", () => {
  const doc = orchestratorDoc();
  const made = make(doc, { href: "http://127.0.0.1:5050/?project_dir=/x&vega_tour=boxes%3Atrain-model" });
  made.tour.mount();
  ok(made.tour.isOpen(), "the tour opened itself on the window it was sent to");
  eq(made.tour.current().id, "train-model", "at the step it was sent to, not at step one");
  eq(made.tour.track(), "boxes", "on the track it was sent on");
  ok(made.tour.index() > 0, `and not at the first step: ${made.tour.index()}`);
  ok(made.env.window.location.href.indexOf("vega_tour") < 0, `the parameter is out of the address bar: ${made.env.window.location.href}`);
  ok(made.env.window.location.href.indexOf("project_dir=/x") > 0, "and the rest of the address is untouched");
  ok(!$(doc, ".vega-tour-resume"), "no resume box: the student was sent here a moment ago");
  eq(made.env.storage.getItem("vega.tour.at"), "train-model", "and this browser now knows the place too");
});

test("test_a_parameter_that_names_another_pages_step_starts_at_this_pages_own", () => {
  const doc = orchestratorDoc();
  const made = make(doc, { href: "http://127.0.0.1:5050/?vega_tour=outlines%3Aann-place" });
  made.tour.mount();
  ok(made.tour.isOpen(), "the tour still opens");
  eq(made.tour.current().surface, "orchestrator", `on this page's own step: ${made.tour.current().id}`);
  eq(made.tour.track(), "outlines", "on the track it was sent on");
});

test("test_a_hostile_parameter_is_ignored", () => {
  const doc = orchestratorDoc();
  const made = make(doc, { href: "http://127.0.0.1:5050/?vega_tour=%3Cscript%3E" });
  made.tour.mount();
  ok(!made.tour.isOpen(), "nothing opens for a parameter that names no track");
  ok($(doc, ".vega-tour-chip"), "and the Tutorial button is still there");
});

test("test_a_new_tab_at_this_address_opens_where_the_student_was_sent", () => {
  const doc = orchestratorDoc();
  const first = make(doc, { href: "http://127.0.0.1:5050/" });
  first.tour.start({ track: "boxes", step: "tools-matrix" });
  fireOn(doc.querySelector('a[href="/matrix"]'), "click");
  eq(first.tour.current().id, "matrix-grid", "the tour moved to the Coverage Matrix step");
  eq(first.env.storage.getItem("vega.tour.moving"), "1", "and marked the move for the new tab");
  const md = new Doc();
  md.add("div", "mx-title");
  md.add("button", "mx-rescan", {}, { top: 100, left: 40, width: 120, height: 30 });
  const second = make(md, { storage: first.env.storage, href: "http://127.0.0.1:5050/matrix/" });
  second.tour.mount();
  ok(second.tour.isOpen(), "the tour opened on the new tab");
  eq(second.tour.current().id, "matrix-grid", "at the step the student was sent to");
  ok(!$(md, ".vega-tour-resume"), "and never asks a question the student has already answered");
  eq(second.env.storage.getItem("vega.tour.moving"), null, "the marker is spent, so a later visit is offered the resume box");
});

test("test_a_step_that_opens_nothing_leaves_no_marker", () => {
  const doc = orchestratorDoc();
  const made = make(doc, { href: "http://127.0.0.1:5050/" });
  made.tour.start({ track: "boxes", step: "train-model" });
  fireOn(doc.getElementById("s6-model"), "click");
  eq(made.env.storage.getItem("vega.tour.moving"), null, "an ordinary step marks no move");
});

test("test_the_steps_that_open_a_window_say_so", () => {
  const withHandoff = CONTENT.steps.filter(s => s.handoff);
  ok(withHandoff.length >= 3, `the steps that open a window declare it: ${withHandoff.map(s => s.id).join(", ")}`);
  for (const s of withHandoff) {
    const body = s.body.toLowerCase();
    ok(body.indexOf("opens") >= 0 && (body.indexOf("tab") >= 0 || body.indexOf("window") >= 0),
       `${s.id}: the body says what opens: ${s.body}`);
  }
});

// ── keyboard ─────────────────────────────────────────────────────────
section("KEYBOARD:");
test("test_escape_exits_and_the_left_arrow_goes_back", () => {
  const doc = orchestratorDoc();
  const { tour } = make(doc);
  const prior = doc.add("button", "somewhere-else");
  prior.focus();
  tour.start({ track: "boxes", step: "train-model" });
  ok(doc.activeElement === calloutOf(doc), "focus moved to the callout on open");
  const at = tour.index();
  doc.fire("keydown", { key: "ArrowLeft" });
  eq(tour.index(), at - 1, "ArrowLeft goes back");
  doc.fire("keydown", { key: "Escape" });
  ok(!tour.isOpen(), "Escape exits");
  ok(doc.activeElement === prior, "focus went back to where it was");
});

test("test_keys_inside_a_product_input_are_left_alone", () => {
  const doc = orchestratorDoc();
  const { tour } = make(doc);
  tour.start({ track: "boxes", step: "train-model" });
  const input = doc.getElementById("s6-epochs");
  input.focus();
  const at = tour.index();
  const ev = doc.fire("keydown", { key: "ArrowLeft" });
  eq(tour.index(), at, "a key typed into a product field does not move the tour");
  ok(!ev.prevented, "and the key reaches the field");
});

test("test_the_product_stays_usable_under_the_tour", () => {
  const { doc } = openAt(orchestratorDoc(), "train-start");
  const ring = ringOf(doc);
  eq(ring.style.pointerEvents, "none", "the ring and its glow never take a click");
  ok(!calloutOf(doc).hasAttribute("aria-modal"), "the dialog is not modal");
});

// ── placement ────────────────────────────────────────────────────────
section("PLACEMENT:");
function intersects(a, b) { return !(a.left + a.width <= b.left || b.left + b.width <= a.left || a.top + a.height <= b.top || b.top + b.height <= a.top); }
function inside(r, vp) { return r.left >= 0 && r.top >= 0 && r.left + r.width <= vp.width && r.top + r.height <= vp.height; }
test("test_the_callout_never_covers_its_anchor_and_stays_on_screen", () => {
  const size = { width: 340, height: 200 };
  const anchors = [
    { top: 100, left: 100, width: 200, height: 40 },
    { top: 700, left: 300, width: 140, height: 32 },
    { top: 10, left: 1000, width: 260, height: 30 },
    { top: 850, left: 900, width: 360, height: 40 },
    { top: 400, left: 0, width: 232, height: 48 },
    { top: 60, left: 230, width: 1040, height: 700 },
  ];
  for (const vp of [{ width: 1280, height: 900 }, { width: 946, height: 700 }]) {
    for (const a of anchors) {
      const p = T.placeCallout(a, size, vp);
      const r = { top: p.top, left: p.left, width: size.width, height: size.height };
      ok(inside(r, vp), `inside the ${vp.width} viewport for anchor ${JSON.stringify(a)}: ${JSON.stringify(p)}`);
      if (p.side !== "corner") ok(!intersects(r, a), `no overlap with anchor ${JSON.stringify(a)} at ${vp.width}: ${JSON.stringify(p)}`);
    }
  }
});

test("test_a_huge_anchor_sends_the_callout_to_the_corner", () => {
  const p = T.placeCallout({ top: 0, left: 0, width: 1280, height: 900 }, { width: 340, height: 200 }, { width: 1280, height: 900 });
  eq(p.side, "corner", "no side has room, so the corner is used");
});

test("test_the_side_with_room_is_chosen", () => {
  const p = T.placeCallout({ top: 100, left: 100, width: 200, height: 40 }, { width: 340, height: 200 }, { width: 1280, height: 900 });
  eq(p.side, "right", "room to the right is used first");
  const q = T.placeCallout({ top: 100, left: 900, width: 300, height: 40 }, { width: 340, height: 200 }, { width: 1280, height: 900 });
  eq(q.side, "left", "then the left");
});

// ── sanitise ─────────────────────────────────────────────────────────
section("SANITISE:");
test("test_hostile_body_content_is_escaped_and_em_code_survive", () => {
  const out = T.sanitiseBody('<script>alert("x")</script> <em>Acropora palmata</em> <code>PA</code> <b onclick="x">bold</b> a & b');
  ok(out.indexOf("<script>") < 0 && out.indexOf("&lt;script&gt;") >= 0, "script is escaped");
  ok(out.indexOf("<em>Acropora palmata</em>") >= 0, "em survives");
  ok(out.indexOf("<code>PA</code>") >= 0, "code survives");
  ok(out.indexOf("<b") < 0 && out.indexOf("onclick") < 0 || out.indexOf("&lt;b") >= 0, "other tags are escaped");
  ok(out.indexOf("a &amp; b") >= 0, "bare ampersands are escaped");
  eq(T.sanitiseBody(null), "", "null is an empty string");
  eq(T.sanitiseBody(12), "12", "a number is its text");
});

test("test_placeholders_fill_from_context_or_fall_back", () => {
  eq(T.fillPlaceholders("Project {project_name} is parked at {stage_name}.", { project_name: "TRAIN_x", stage_name: "Evaluate Model" }),
     "Project TRAIN_x is parked at Evaluate Model.", "placeholders are filled");
  eq(T.fillPlaceholders("Project {project_name}.", {}), null, "a missing value answers null so the caller uses the fallback");
  eq(T.fillPlaceholders("No holes.", null), "No holes.", "text without holes passes through");
});

// ── adversarial ──────────────────────────────────────────────────────
section("ADVERSARIAL:");
test("test_content_with_no_steps_and_unknown_step_ids", () => {
  const empty = make(orchestratorDoc(), { content: { tracks: { boxes: {}, outlines: {} }, steps: [] } });
  empty.tour.start({ track: "boxes" });
  ok(!empty.tour.isOpen(), "nothing to show, nothing opens, no throw");
  const { tour } = make(orchestratorDoc());
  tour.start({ track: "boxes", step: "no-such-step" });
  eq(tour.current().id, "welcome", "an unknown step id falls back to the first step");
});

test("test_a_rejecting_fetch_and_a_non_json_status_keep_the_wait_honest", async () => {
  const doc = orchestratorDoc();
  const { tour } = make(doc, { fetch: fetchFor({ "/api/project/state": new Error("offline"), "/api/step/6/status": "not json" }) });
  tour.start({ track: "boxes", step: "train-wait" });
  await settle();
  ok(tour.isOpen(), "the tour stays open");
  const at = tour.index();
  doc.getElementById("s6-result").innerHTML = '<div class="badge badge-success">done</div>';
  doc.deliver();
  await settle();
  eq(tour.index(), at, "an unreadable status never counts as a finished run");
  ok(bodyText(doc).indexOf("finished") < 0, "and the line never claims it");
});

test("test_a_step_with_no_body_and_a_click_on_a_missing_anchor", () => {
  const doc = orchestratorDoc();
  const content = { tracks: { boxes: {}, outlines: {} }, steps: [
    { id: "welcome", stage: 1, track: "both", surface: "any", anchor: null, title: "T", body: "B.", advance: "input", fallback: "F." },
    { id: "x", stage: 2, track: "both", surface: "orchestrator", anchor: "#nope", title: "X", body: "", advance: "click" },
  ] };
  const { tour } = make(doc, { content });
  tour.start({ track: "boxes", step: "x" });
  ok(bodyText(doc).length > 0, "a step with no body and no fallback still shows a sentence");
  const at = tour.index();
  fireOn(doc.getElementById("s6-run-btn"), "click");
  eq(tour.index(), at, "a click with no anchor resolved never advances");
  ok(tour.isOpen(), "and nothing throws");
});

test("test_a_hostile_anchor_selector_never_throws", () => {
  const doc = orchestratorDoc();
  const content = { tracks: { boxes: {}, outlines: {} }, steps: [
    { id: "welcome", stage: 1, track: "both", surface: "any", anchor: null, title: "T", body: "B.", advance: "input", fallback: "F." },
    { id: "bad", stage: 2, track: "both", surface: "orchestrator", anchor: "((((", title: "X", body: "B.", advance: "click", fallback: "F." },
  ] };
  const { tour } = make(doc, { content });
  tour.start({ track: "boxes", step: "bad" });
  ok(tour.isOpen() && tour.anchorElement() === null, "an unparseable selector resolves to nothing and the tour stays up");
  eq(T.resolveAnchor(doc, "((("), null, "resolveAnchor swallows the error");
});

chain.then(() => {
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
});

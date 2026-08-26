#!/usr/bin/env node
/* SMOKE: boot the REAL viewer.js browser branch against a tiny DOM shim + a
   fixture codes.json/review_manifest.json, with no real browser (no jsdom).

   Proves the C1 name gate opens on load, cards render GROUPED BY PROJECT with
   sticky group headers + pending counts, tentative IDs (C5) show reviewer only
   (confidence UI removed), the project/site filter bar (Phase 4 / C4) exists,
   and export is blocked until a name is set. This exercises the same file the
   GitHub-Pages site loads (viewer.js), guarding the HTML/JS element-id
   contract that the pure-logic test cannot see.

   Run: node scripts/_reefreview/tests/test_viewer_dom.js */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

let pass = 0, fail = 0;
function ok(c, m) { if (c) pass++; else { fail++; console.error("  FAIL:", m); } }

// ── parse the static element ids from index.html so the shim DOM has them ──
const VIEWDIR = path.join(__dirname, "..", "viewer");
const html = fs.readFileSync(path.join(VIEWDIR, "index.html"), "utf8");
const STATIC_IDS = Array.from(new Set(
  (html.match(/id=["']([\w-]+)["']/g) || []).map(s => s.replace(/id=["']|["']/g, ""))
));

// ── static HTML/CSS contract (element ids + revamp copy rules) ───────
["counts", "returnAlert", "emailTargets", "whoami", "changeName", "instructions",
 "filterbar", "cards", "progress", "exportBtn", "nameGate", "reviewerName",
 "nameErr", "nameOk", "exportModal", "csvName", "contactList", "mailtoLink",
 "closeModal"].forEach(id => ok(STATIC_IDS.indexOf(id) >= 0, "index.html keeps required id #" + id));
ok(!/<details[^>]*\bopen\b/.test(html), "instructions <details> collapsed by default (no open attr)");
ok(/<summary[^>]*>How to review<\/summary>/.test(html), "instructions summary reads 'How to review'");
ok(!/confiden/i.test(html), "no confidence copy left in index.html");
ok(html.indexOf("—") < 0 && !/&mdash;/.test(html), "no em dashes in index.html");
const css = fs.readFileSync(path.join(VIEWDIR, "viewer.css"), "utf8");
ok(!/\.conf\b|confiden/.test(css), "no confidence styles left in viewer.css");
ok(/\.grouphead\b/.test(css), "group header styles present in viewer.css");

// ── minimal DOM shim ─────────────────────────────────────────────────
function makeEl(tag) {
  return {
    tagName: (tag || "div").toUpperCase(), _children: [], _listeners: {},
    className: "", textContent: "", innerHTML: "", title: "", hidden: false,
    style: {}, dataset: {}, value: "", disabled: false, type: "",
    appendChild(c) { this._children.push(c); return c; },
    addEventListener(ev, fn) { (this._listeners[ev] = this._listeners[ev] || []).push(fn); },
    removeEventListener() {},
    setAttribute(k, v) { if (k === "id") this.id = v; this[k] = v; },
    getAttribute(k) { return this[k]; },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    classList: {
      _s: {}, add(c) { this._s[c] = 1; }, remove(c) { delete this._s[c]; },
      toggle(c, on) { if (on === undefined) on = !this._s[c]; if (on) this._s[c] = 1; else delete this._s[c]; },
      contains(c) { return !!this._s[c]; },
    },
    focus() {}, click() {},
    get firstChild() { return this._children[0] || null; },
    get children() { return this._children; },
  };
}
const REG = {};
STATIC_IDS.forEach(id => { REG[id] = makeEl("div"); });
// the things boot() reads .value/.hidden on:
["reviewerName"].forEach(id => { REG[id] = REG[id] || makeEl("input"); REG[id].value = ""; });

const cardsHost = REG["cards"];
cardsHost.querySelector = function (sel) {
  // support .card[data-uid="X"] lookups used by bulkApply (none triggered here)
  return null;
};
// class-exact matching so ".card" and ".grouphead" select their own elements
cardsHost.querySelectorAll = function (sel) {
  const cls = String(sel || "").replace(/^\./, "").split("[")[0];
  return cardsHost._children.filter(c => (" " + (c.className || "") + " ").indexOf(" " + cls + " ") >= 0);
};

const documentShim = {
  readyState: "complete",
  getElementById(id) { return REG[id] || null; },
  createElement(tag) { return makeEl(tag); },
  addEventListener() {},
  querySelector() { return null; },
  querySelectorAll() { return []; },
};

// fixture data the fetch shim returns
const CODES = {
  generated_at: "x", codes: [{ code: "MCAV", name: "Montastraea cavernosa" },
    { code: "OFRA", name: "Orbicella franksi" }],
  groups: [{ group: "Coral", codes: ["MCAV", "OFRA"] }],
  confidence: { high: { definition: "hi" }, low: { definition: "lo" } },
  default_confidence: "high", idk: { code: "IDK", label: "I don't know" },
  something_else: { code: "OTHER_PICK", label: "Something else" },
  sites: { GRP: "Great Pond", BWR: "Brewers Bay" },
  candidate_codes: ["OFRA"], operator: "op@reef.org",
};
const MANIFEST = {
  generated_at: "x", contacts: ["ops@x.org"], reviewer_names: [], count: 2,
  items: [
    { uid: "u1", project_id: "p1", project_name: "Reef One", site: "GRP", site_full: "Great Pond",
      year: 2020, transect: 1, frame: 1, featured_codes: ["MCAV"], candidate_codes: [],
      reviews: [{ reviewer: "Bob", code: "OFRA", confidence: "high" }] },
    { uid: "u2", project_id: "p2", project_name: "Reef Two", site: "BWR", site_full: "Brewers Bay",
      year: 2021, transect: 2, frame: 2, featured_codes: ["MCAV"], candidate_codes: [], reviews: [] },
  ],
};
function fetchShim(url) {
  const body = url.indexOf("codes.json") >= 0 ? CODES : MANIFEST;
  return Promise.resolve({ ok: true, json: () => Promise.resolve(body) });
}

const store = {};
const localStorageShim = {
  getItem(k) { return k in store ? store[k] : null; },
  setItem(k, v) { store[k] = String(v); }, removeItem(k) { delete store[k]; },
};

// Image/Blob/URL/alert/confirm stubs the card builder + export touch
function ImageShim() { this.onload = null; this.onerror = null; Object.defineProperty(this, "src", { set() {}, get() { return ""; } }); }
const sandbox = {
  document: documentShim, window: {}, self: {},
  fetch: fetchShim, localStorage: localStorageShim,
  Image: ImageShim, Option: function (t, v) { return { text: t, value: v }; },
  Blob: function () {}, URL: { createObjectURL: () => "blob:x", revokeObjectURL() {} },
  alert: () => {}, confirm: () => true, setTimeout: (fn) => { try { fn(); } catch (e) {} return 0; },
  console, encodeURIComponent,
};
sandbox.window = sandbox; sandbox.self = sandbox;

// ── run the real viewer.js in the shimmed context ────────────────────
const code = fs.readFileSync(path.join(VIEWDIR, "viewer.js"), "utf8");
vm.createContext(sandbox);
try {
  vm.runInContext(code, sandbox, { filename: "viewer.js" });
} catch (e) {
  fail++; console.error("  FAIL: viewer.js threw on boot:", e.message);
}

// boot() is sync-dispatched (readyState=complete) but fetch resolves on a
// microtask; flush microtasks before asserting.
setImmediate(function () {
  // C1: name gate opened on load (no stored name)
  ok(REG.nameGate && REG.nameGate.hidden === false, "C1: name gate is shown on load when no reviewer stored");

  // cards rendered (one per item)
  const cards = cardsHost._children.filter(c => c.className === "card");
  ok(cards.length === 2, "two cards rendered (got " + cards.length + ")");

  // grouped render: header, its cards, next header, its cards (label order)
  const kinds = cardsHost._children.map(k => (k.className || "").split(" ")[0]);
  ok(JSON.stringify(kinds) === JSON.stringify(["grouphead", "card", "grouphead", "card"]),
     "grouped render order: header, card, header, card (got " + kinds.join(",") + ")");
  const heads = cardsHost._children.filter(c => c.className === "grouphead");
  ok(heads.length === 2, "one group header per project");
  const headNames = heads.map(h => (h._children[0] || {}).textContent);
  ok(JSON.stringify(headNames) === JSON.stringify(["Reef One", "Reef Two"]),
     "group headers show project names in label order (got " + headNames.join(",") + ")");
  const headPending = heads.map(h => (h._pending || {}).textContent);
  ok(JSON.stringify(headPending) === JSON.stringify(["1 pending", "1 pending"]),
     "group headers show pending counts (got " + headPending.join(",") + ")");
  ok(heads.every(h => (h.title || "").length > 0), "group headers carry a tooltip");

  // confidence UI removed: no Confidence label / High / Low anywhere on a card;
  // tentative chips show the reviewer only (fixture review carries a legacy
  // confidence value the viewer must ignore)
  function collectText(n, out) {
    out.push(n.textContent || "");
    (n._children || []).forEach(c => collectText(c, out));
    return out;
  }
  const cardText = cards.map(c => collectText(c, []).join(" ")).join(" ");
  ok(cardText.indexOf("Confidence") < 0 && !/\bHigh\b/.test(cardText) && !/\bLow\b/.test(cardText),
     "no confidence UI on cards");
  ok(cardText.indexOf("Bob") >= 0, "C5: tentative chip shows the reviewer");
  ok(!/\bhigh\b/.test(cardText), "C5: tentative chip omits the legacy confidence value");

  // C2: header email targets painted with operator + contacts + lauren
  const targets = REG.emailTargets.textContent || "";
  ok(targets.indexOf("op@reef.org") >= 0, "C2: operator email shown in header");
  ok(targets.indexOf("lauren.olinger@uvi.edu") >= 0, "C2: lauren email always shown in header");

  // filter bar built (Phase 4 + C4): project + site + bulk selects exist
  const fb = REG.filterbar._children;
  ok(fb.length >= 3, "filter bar populated (project + site + bulk) controls");

  // export is blocked until a name is set: clicking export with no name re-opens
  // the gate rather than throwing.
  store["tcrmp_reviewer_name_v1"] = "";   // ensure unset
  const exportListeners = (REG.exportBtn._listeners.click || []);
  ok(exportListeners.length === 1, "export button wired exactly once");
  let threw = false;
  try { exportListeners[0](); } catch (e) { threw = true; }
  ok(!threw, "C1: export with no name does not throw");
  ok(REG.nameGate.hidden === false, "C1: export with no name re-opens the name gate (blocks labeling)");

  // accept a name -> gate closes, whoami pill set
  REG.reviewerName.value = "Jane Doe";
  const okListeners = REG.nameOk._listeners.click || [];
  ok(okListeners.length === 1, "name-OK button wired");
  okListeners[0]();
  ok(REG.nameGate.hidden === true, "C1: entering a name closes the gate");
  ok((REG.whoami.textContent || "").indexOf("Jane Doe") >= 0, "C1: whoami pill shows the reviewer");
  ok(store["tcrmp_reviewer_name_v1"] === "Jane Doe", "C1: reviewer persisted to localStorage");

  console.log(`viewer_dom: ${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
});

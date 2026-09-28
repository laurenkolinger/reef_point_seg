/* VEGA guided tour engine.

   A guided overlay that runs on top of the real VEGA surfaces. It never
   simulates the product and never clicks for the student: it glows the real
   control the student must use next, says what it does, and waits.

   Version 2 (Lauren, 2026-09-17): there is no forward button. A step
   completes only when the student does the real thing on the live page, one
   of four kinds: "click" (a click on the anchor or inside it), "input" (the
   anchored field filled or changed), "navigate" (the target surface or panel
   reached), "event" (a run starting or finishing). Back and Exit tour remain;
   neither moves the tour forward.

   Version 3 (Lauren, 2026-09-17, walking the tour): the tour keeps its place
   when the student moves to another window. A browser keeps localStorage per
   address, and VEGA runs on three (the VICARIUS desktop on 5090, the pipeline
   window and its tools on 5050, the annotator on 5080 and up), so a student
   who crossed from one to another used to arrive at step one. The tour is the
   thing that tells the student to open the next window, so it owns the
   handoff, by two carriers with one contract:
     - another address: while the tour is open and the step it is waiting on
       belongs to another page, the tour wraps the page's own window opener and
       puts "vega_tour=<track>:<step id>" in the address it opens. The tour on
       the receiving page reads that on load, resumes exactly there, and takes
       the parameter out of the address bar. Nothing else about the opened
       window changes, and the wrap is removed when the tour closes.
     - the same address in a new tab: the tour writes vega.tour.moving beside
       the place it already saves, and the tour on the new tab opens at that
       place instead of offering to resume. The flag is read once and cleared.
   A student who opens a window some other way still gets what was already
   there: the saved place, and the first step at or after it for the page.

   Spec: docs/superpowers/specs/2026-09-16-vega-tutorial-design.md. Design
   authority for type, tokens and copy: docs/superpowers/specs/
   2026-09-02-workflow-page-design.md version 2. Plan:
   docs/superpowers/plans/2026-09-17-vega-tour-t15.md (decisions DT1 to DT11).

   Content is data in static/vega_tour_content.js (window.VegaTourContent).
   The stylesheet is static/vega_tour.css, scoped under .vega-tour.

   In a browser this file creates window.VegaTour, the tour bound to the real
   document, and mounts it (a Tutorial chip, and a resume offer when the tour
   was open the last time this window closed). Under node it exports
   createTour and the pure helpers so the engine can be tested without a
   browser; every DOM, storage, network, timer and observer access goes
   through the environment object createTour is given.

   Rules the engine keeps:
   - the product is never disabled: the ring and its backdrop take no
     pointer events, the dialog is not modal, keys typed into a product
     field are the field's;
   - no state is fabricated: a run is called finished or failed only after
     the project's own record or the step status route says so;
   - no polling loop of its own: the wait for a run watches the panel
     elements the existing two-second poll already writes;
   - a missing anchor never breaks the tour and never shows an empty bubble:
     the callout says which page the step happens on, Back and Exit tour keep
     working, and nothing skips the step forward. */
(function (root, factory) {
  'use strict';
  var api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root && root.document) {
    var tour = api.createTour({
      document: root.document,
      window: root,
      storage: api.safeStorage(root),
      fetch: typeof root.fetch === 'function' ? root.fetch.bind(root) : null,
      content: null,
      MutationObserver: root.MutationObserver,
      ResizeObserver: root.ResizeObserver,
      setTimeout: root.setTimeout.bind(root),
      clearTimeout: root.clearTimeout.bind(root),
      now: function () { return Date.now(); }
    });
    tour.helpers = api;
    root.VegaTour = tour;
    if (root.document.readyState === 'loading') {
      root.document.addEventListener('DOMContentLoaded', function () { tour.mount(); });
    } else {
      tour.mount();
    }
  }
})(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  var STORE_PREFIX = 'vega.tour.';
  var KEYS = {
    track: STORE_PREFIX + 'track',
    at: STORE_PREFIX + 'at',
    furthest: STORE_PREFIX + 'furthest',
    open: STORE_PREFIX + 'open',
    done: STORE_PREFIX + 'done',
    trainStartedAt: STORE_PREFIX + 'trainStartedAt',
    // Set when a step that opens another tab at the same address completes, so
    // the tour on the new tab opens where the student was sent rather than
    // asking whether to resume. Read once, then cleared.
    moving: STORE_PREFIX + 'moving'
  };
  // The name of the query parameter that carries the place to another address.
  var HANDOFF_PARAM = 'vega_tour';
  var TRACKS = ['boxes', 'outlines'];
  // A stable id on each page the tour visits, and the surface it names.
  var SURFACE_IDS = [
    ['step-nav', 'orchestrator'],
    ['bx-sec-datasets', 'boxes'],
    ['mx-title', 'matrix'],
    ['lb-title', 'labels'],
    ['review', 'annotator'],
    ['runs-list', 'launcher']
  ];
  var STATE_URL = '/api/project/state';
  var CALLOUT_WIDTH = 340;
  var CALLOUT_DEFAULT_HEIGHT = 200;
  var GAP = 12;
  var MARGIN = 12;
  var RING_PAD = 5;
  var RETRY_MS = 500;
  var RETRY_MAX = 20;
  // How many times one step may pull its control back on screen after the page
  // pushed it off, so the tour can never fight a student who scrolled away.
  var FOLLOW_SCROLL_MAX = 3;
  var BEAT_MS = 2000;
  // A typed field advances once the typing settles, so the glow does not jump
  // to the next control between two keystrokes; a change event advances at once.
  var INPUT_SETTLE_MS = 700;
  var TYPED_TYPES = ['text', 'number', 'search', 'tel', 'url', 'email', 'password'];
  var STEP_NAMES = { 6: 'Training', 7: 'The evaluation', 8: 'Inference' };
  var EMPTY_BODY_TEXT = 'The control for this step is not on this surface yet.';
  // Where a step happens, in the words a student would use, for the line the
  // callout adds when the anchor is absent.
  var SURFACE_NAMES = {
    launcher: 'the VEGA page of the VICARIUS desktop',
    orchestrator: 'the pipeline window',
    boxes: 'the Bounding Boxes page',
    matrix: 'the Coverage Matrix',
    labels: 'the Label Manager',
    annotator: 'the annotator window'
  };
  var WAITING_HERE_TEXT = 'VEGA glows the control as soon as it appears.';
  var CHIP_TITLE = 'Open the guided tutorial, which glows the real control to use next and waits for the student to use it. VEGA remembers the track and the step in this browser.';
  var CHIP_RESUME_TITLE = 'Open the guided tutorial at the step it was left on. VEGA remembers the track and the step in this browser.';

  // -- pure helpers --------------------------------------------------------

  /**
   * The steps of one track, in content order.
   * @param {object} content the content module ({steps})
   * @param {string} track "boxes" or "outlines"
   * @returns {object[]} steps whose track is the given one or "both"
   */
  function stepsForTrack(content, track) {
    var steps = content && Array.isArray(content.steps) ? content.steps : [];
    if (TRACKS.indexOf(track) < 0) return [];
    return steps.filter(function (s) { return s && (s.track === track || s.track === 'both'); });
  }

  /** Whether an element is on screen enough to ring: not hidden, with a size. */
  function isVisible(el) {
    if (!el || el.hidden === true) return false;
    var r = el.getBoundingClientRect ? el.getBoundingClientRect() : null;
    return !!(r && r.width > 0 && r.height > 0);
  }

  /**
   * The element an anchor names, when it is on the page and visible. An
   * array is a chain: the selectors are tried in order and the first match
   * wins, so a step can glow one card when the page marks it and the list
   * that holds it when it does not.
   * @param {Document} doc
   * @param {string|string[]} selector
   * @returns {Element|null}
   */
  function resolveAnchor(doc, selector) {
    if (!doc || !selector) return null;
    var list = Array.isArray(selector) ? selector : [selector];
    for (var i = 0; i < list.length; i++) {
      if (!list[i] || typeof list[i] !== 'string') continue;
      var el = null;
      try { el = doc.querySelector(list[i]); } catch (e) { el = null; }
      if (isVisible(el)) return el;
    }
    return null;
  }

  /** Whether two rectangles ({top,left,width,height}) overlap. */
  function intersects(a, b) {
    return !(a.left + a.width <= b.left || b.left + b.width <= a.left ||
             a.top + a.height <= b.top || b.top + b.height <= a.top);
  }

  /** Whether a rectangle sits inside the viewport with the margin kept. */
  function fits(r, vp) {
    return r.left >= MARGIN && r.top >= MARGIN &&
           r.left + r.width <= vp.width - MARGIN && r.top + r.height <= vp.height - MARGIN;
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(v, hi)); }

  /**
   * Where the callout goes: the side of the anchor with room, right, left,
   * below then above, never intersecting the anchor; the bottom right corner
   * when no side has room.
   * @param {object} a the anchor rect {top,left,width,height}
   * @param {object} size the callout size {width,height}
   * @param {object} vp the viewport {width,height}
   * @returns {{top:number,left:number,side:string}}
   */
  function placeCallout(a, size, vp) {
    var w = size.width, h = size.height;
    var maxLeft = vp.width - w - MARGIN, maxTop = vp.height - h - MARGIN;
    var candidates = [
      { side: 'right', left: a.left + a.width + GAP, top: clamp(a.top, MARGIN, maxTop) },
      { side: 'left', left: a.left - GAP - w, top: clamp(a.top, MARGIN, maxTop) },
      { side: 'below', left: clamp(a.left, MARGIN, maxLeft), top: a.top + a.height + GAP },
      { side: 'above', left: clamp(a.left, MARGIN, maxLeft), top: a.top - GAP - h }
    ];
    for (var i = 0; i < candidates.length; i++) {
      var c = candidates[i];
      var r = { top: c.top, left: c.left, width: w, height: h };
      if (fits(r, vp) && !intersects(r, a)) return c;
    }
    return { side: 'corner', left: Math.max(MARGIN, maxLeft), top: Math.max(MARGIN, maxTop) };
  }

  /** Where a callout with no anchor goes: the centre for the first screen, else the corner. */
  function placeFree(size, vp, centre) {
    if (centre) {
      return { side: 'centre', left: Math.max(MARGIN, (vp.width - size.width) / 2), top: Math.max(MARGIN, (vp.height - size.height) / 2) };
    }
    return { side: 'corner', left: Math.max(MARGIN, vp.width - size.width - MARGIN), top: Math.max(MARGIN, vp.height - size.height - MARGIN) };
  }

  /**
   * The epoch count from the Train panel's own progress text ("Epoch 3 of 10").
   * @param {string} text
   * @returns {{cur:number,total:number}|null}
   */
  function parseEpoch(text) {
    var m = String(text || '').match(/Epoch\s+(\d+)\s+of\s+(\d+)/);
    return m ? { cur: parseInt(m[1], 10), total: parseInt(m[2], 10) } : null;
  }

  /** Milliseconds as the panel writes them: "1h 2m 5s", "2m 5s", "12s". */
  function formatElapsed(ms) {
    var s = Math.max(0, Math.floor((Number(ms) || 0) / 1000));
    var m = Math.floor(s / 60), h = Math.floor(m / 60);
    if (h > 0) return h + 'h ' + (m % 60) + 'm ' + (s % 60) + 's';
    if (m > 0) return m + 'm ' + (s % 60) + 's';
    return s + 's';
  }

  function escapeHtml(text) {
    return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  /**
   * Body copy made safe: everything escaped, then only <em> and <code> let
   * back through, because a species binomial is italic and a file name is a chip.
   * @param {*} html
   * @returns {string}
   */
  function sanitiseBody(html) {
    if (html === null || html === undefined) return '';
    return escapeHtml(html)
      .replace(/&lt;(\/?)(em|code)&gt;/g, '<$1$2>');
  }

  /** Text stripped of markup, for reading the panel's own lines. */
  function stripTags(html) { return String(html || '').replace(/<[^>]+>/g, ''); }

  /**
   * Which VEGA surface a document is, read from one stable id per page.
   * @param {Document} doc
   * @returns {string|null}
   */
  function detectSurface(doc) {
    if (!doc || !doc.getElementById) return null;
    for (var i = 0; i < SURFACE_IDS.length; i++) {
      var el = null;
      try { el = doc.getElementById(SURFACE_IDS[i][0]); } catch (e) { el = null; }
      if (el) return SURFACE_IDS[i][1];
    }
    return null;
  }

  /** The saved state, every read guarded; an unreadable store reads as empty. */
  function readState(storage) {
    var out = { track: null, at: null, furthest: -1, open: false, done: false, trainStartedAt: null, moving: false };
    if (!storage) return out;
    function get(k) { try { return storage.getItem(k); } catch (e) { return null; } }
    var track = get(KEYS.track);
    out.track = TRACKS.indexOf(track) >= 0 ? track : null;
    out.at = get(KEYS.at) || null;
    var f = parseInt(get(KEYS.furthest), 10);
    out.furthest = isNaN(f) ? -1 : f;
    out.open = get(KEYS.open) === '1';
    out.done = get(KEYS.done) === '1';
    out.moving = get(KEYS.moving) === '1';
    var t = parseInt(get(KEYS.trainStartedAt), 10);
    out.trainStartedAt = isNaN(t) ? null : t;
    return out;
  }

  /**
   * Write the given fields; a null value removes the key.
   * @returns {boolean} false when the store refused, never throws
   */
  function writeState(storage, patch) {
    if (!storage || !patch) return false;
    var ok = true;
    Object.keys(patch).forEach(function (name) {
      var key = KEYS[name];
      if (!key) return;
      try {
        if (patch[name] === null || patch[name] === undefined) storage.removeItem(key);
        else storage.setItem(key, String(patch[name]));
      } catch (e) { ok = false; }
    });
    return ok;
  }

  /** A storage that never throws to its caller, or null when there is none. */
  function safeStorage(win) {
    try { return win.localStorage || null; } catch (e) { return null; }
  }

  // -- carrying the place to another window --------------------------------

  /**
   * The address an absolute URL belongs to, lowercased ("http://127.0.0.1:5050"),
   * or an empty string for a relative one, which is always the page's own.
   * @param {string} url
   * @returns {string}
   */
  function originOf(url) {
    var m = String(url === null || url === undefined ? '' : url)
      .match(/^([a-z][a-z0-9+.-]*:)\/\/([^\/?#]*)/i);
    return m ? (m[1] + '//' + m[2]).toLowerCase() : '';
  }

  /**
   * The same URL with any handoff parameter taken out, so a place is never
   * carried twice and the address bar is left as the page's own.
   * @param {string} url
   * @returns {string}
   */
  function stripHandoff(url) {
    var s = String(url === null || url === undefined ? '' : url);
    var re = new RegExp('([?&])' + HANDOFF_PARAM + '=[^&#]*(&)?', 'g');
    var out = s.replace(re, function (all, lead, trail) { return trail ? lead : ''; });
    return out.replace(/\?(#|$)/, '$1');
  }

  /**
   * The same URL carrying the tour's place, as a query parameter so an
   * existing fragment is left alone.
   * @param {string} url
   * @param {string} track "boxes" or "outlines"
   * @param {string} stepId the step to resume at
   * @returns {string}
   */
  function withHandoff(url, track, stepId) {
    var base = stripHandoff(url);
    var hash = '';
    var h = base.indexOf('#');
    if (h >= 0) { hash = base.slice(h); base = base.slice(0, h); }
    var sep = base.indexOf('?') >= 0 ? '&' : '?';
    return base + sep + HANDOFF_PARAM + '=' + encodeURIComponent(track + ':' + stepId) + hash;
  }

  /**
   * The place an address carries, or null when it carries none or one that is
   * not a track and a step id.
   * @param {string} search the query part of an address ("?a=1&vega_tour=...")
   * @returns {{track:string, step:string}|null}
   */
  function parseHandoff(search) {
    var m = String(search === null || search === undefined ? '' : search)
      .match(new RegExp('[?&]' + HANDOFF_PARAM + '=([^&#]+)'));
    if (!m) return null;
    var raw;
    try { raw = decodeURIComponent(m[1]); } catch (e) { return null; }
    var bits = raw.split(':');
    if (bits.length !== 2) return null;
    if (TRACKS.indexOf(bits[0]) < 0) return null;
    if (!/^[a-z0-9-]+$/.test(bits[1])) return null;
    return { track: bits[0], step: bits[1] };
  }

  /**
   * Fill {name} holes from a context; null when any hole has no value, so
   * the caller shows the fallback rather than a sentence with a hole in it.
   * @param {string} text
   * @param {object|null} ctx
   * @returns {string|null}
   */
  function fillPlaceholders(text, ctx) {
    var s = String(text || '');
    if (s.indexOf('{') < 0) return s;
    var missing = false;
    var out = s.replace(/\{([a-z_]+)\}/g, function (all, name) {
      if (ctx && ctx[name] !== undefined && ctx[name] !== null && String(ctx[name]).trim()) return escapeHtml(ctx[name]);
      missing = true;
      return all;
    });
    return missing ? null : out;
  }

  // -- the tour -------------------------------------------------------------

  /**
   * Build a tour bound to an environment.
   * @param {object} env document, window, storage, fetch, content,
   *   MutationObserver, ResizeObserver, setTimeout, clearTimeout, now
   * @returns {object} the tour
   */
  function createTour(env) {
    if (!env || !env.document) throw new Error('createTour needs an environment with a document');
    var doc = env.document;
    var win = env.window || {};
    var storage = env.storage || null;
    var setTimer = env.setTimeout || function (fn) { fn(); return null; };
    var clearTimer = env.clearTimeout || function () {};
    var now = env.now || function () { return Date.now(); };
    var MO = env.MutationObserver || null;
    var RO = env.ResizeObserver || null;

    var track = null;
    var steps = [];
    var idx = 0;
    var open = false;
    var els = null;            // { root, ring, callout, count, title, body, choice, actions, back, exit }
    var anchorEl = null;
    var prevFocus = null;
    var cleanups = [];         // watcher teardown functions for the current step
    var retryHandle = null;
    var retriesLeft = 0;
    var followScrolls = 0;    // re-scrolls spent on the step showing now
    var live = null;           // { state, epoch, elapsedText, exitCode } for an event step
    var ctx = null;            // { project_name, stage_name } from the open project
    var ctxRequested = false;
    var chip = null;
    var resumeBox = null;
    var keyHandler = null;
    var placeHandler = null;
    var launchObserver = null;
    var stateFetchSeq = 0;
    var nativeOpen = null;     // the page's own window opener, while wrapped

    function content() { return env.content || (win && win.VegaTourContent) || null; }
    function surface() { return detectSurface(doc); }
    function viewport() {
      var de = doc.documentElement || {};
      return { width: de.clientWidth || win.innerWidth || 1280, height: de.clientHeight || win.innerHeight || 900 };
    }

    /** GET a JSON route; null on any failure (no route, network, non-JSON). */
    function getJson(url) {
      if (typeof env.fetch !== 'function') return Promise.resolve(null);
      var p;
      try { p = env.fetch(url, { credentials: 'same-origin' }); } catch (e) { return Promise.resolve(null); }
      return Promise.resolve(p).then(function (resp) {
        if (!resp || !resp.ok || typeof resp.json !== 'function') return null;
        return resp.json().then(function (j) { return j && typeof j === 'object' ? j : null; }, function () { return null; });
      }, function () { return null; });
    }

    function firstStep() {
      var c = content();
      var all = c && Array.isArray(c.steps) ? c.steps : [];
      for (var i = 0; i < all.length; i++) if (all[i] && all[i].surface === 'any') return all[i];
      return all[0] || null;
    }

    function computeSteps() {
      if (track) { steps = stepsForTrack(content(), track); return; }
      var w = firstStep();
      steps = w ? [w] : [];
    }

    function current() { return steps[idx] || null; }

    function save(extra) {
      var cur = current();
      var patch = { track: track, at: cur ? cur.id : null, open: open ? '1' : '0' };
      var st = readState(storage);
      patch.furthest = Math.max(st.furthest, idx);
      if (extra) Object.keys(extra).forEach(function (k) { patch[k] = extra[k]; });
      writeState(storage, patch);
    }

    // -- the callout ---------------------------------------------------------

    function make(tag, cls, text) {
      var el = doc.createElement(tag);
      if (cls) el.className = cls;
      if (text !== undefined) el.textContent = text;
      return el;
    }

    function button(action, label, primary) {
      var b = make('button', 'vega-tour-btn' + (primary ? ' is-primary' : ''), label);
      b.setAttribute('type', 'button');
      b.setAttribute('data-tour-action', action);
      return b;
    }

    function build() {
      if (els) return;
      var rootEl = make('div', 'vega-tour');
      // The first screen has no control on the page to glow, so it dims the
      // page instead: the choice in the callout is the only thing to act on.
      var veil = make('div', 'vega-tour-veil');
      veil.setAttribute('aria-hidden', 'true');
      veil.style.pointerEvents = 'none';
      veil.hidden = true;
      var ring = make('div', 'vega-tour-ring');
      ring.setAttribute('aria-hidden', 'true');
      ring.style.pointerEvents = 'none';
      ring.hidden = true;
      var callout = make('div', 'vega-tour-callout');
      callout.setAttribute('role', 'dialog');
      callout.setAttribute('aria-labelledby', 'vega-tour-title');
      callout.setAttribute('aria-describedby', 'vega-tour-body');
      callout.tabIndex = -1;
      callout.setAttribute('tabindex', '-1');
      var count = make('div', 'vega-tour-count');
      var title = make('h2', 'vega-tour-title');
      title.id = 'vega-tour-title';
      title.setAttribute('id', 'vega-tour-title');
      var body = make('div', 'vega-tour-body');
      body.id = 'vega-tour-body';
      body.setAttribute('id', 'vega-tour-body');
      var choice = make('div', 'vega-tour-choice');
      choice.hidden = true;
      // Back and Exit tour only. There is no forward control of any kind:
      // the step's own action on the real page is the only way on.
      var actions = make('div', 'vega-tour-actions');
      var back = button('back', 'Back', false);
      var exit = button('exit', 'Exit tour', false);
      exit.className += ' vega-tour-exit';
      actions.appendChild(back); actions.appendChild(exit);
      callout.appendChild(count); callout.appendChild(title); callout.appendChild(body);
      callout.appendChild(choice); callout.appendChild(actions);
      rootEl.appendChild(veil); rootEl.appendChild(ring); rootEl.appendChild(callout);
      back.addEventListener('click', function () { tour.back(); });
      exit.addEventListener('click', function () { tour.exit(); });
      doc.body.appendChild(rootEl);
      els = { root: rootEl, veil: veil, ring: ring, callout: callout, count: count, title: title, body: body, choice: choice, actions: actions, back: back, exit: exit };
    }

    function teardownCallout() {
      if (!els) return;
      els.root.remove();
      els = null;
    }

    function clearWatchers() {
      cleanups.forEach(function (fn) { try { fn(); } catch (e) { /* a watcher that is already gone */ } });
      cleanups = [];
      if (retryHandle) { clearTimer(retryHandle); retryHandle = null; }
      live = null;
    }

    /** Whether an element holds a value worth calling filled in. */
    function hasValue(el) {
      if (!el) return false;
      var tag = String(el.tagName || '').toLowerCase();
      var type = String(el.getAttribute ? (el.getAttribute('type') || '') : '').toLowerCase();
      if (tag === 'textarea' || (tag === 'input' && TYPED_TYPES.indexOf(type || 'text') >= 0)) {
        return String(el.value === undefined ? '' : el.value).trim() !== '';
      }
      // A select, a checkbox, a radio or a slider: the change is the action.
      return true;
    }

    function renderChoice() {
      var c = content();
      els.choice.innerHTML = '';
      var cur = current();
      if (!cur || cur.surface !== 'any' || !c || !c.tracks) { els.choice.hidden = true; return; }
      els.choice.hidden = false;
      TRACKS.forEach(function (name) {
        var info = c.tracks[name] || {};
        var label = make('label', 'vega-tour-opt');
        var input = make('input');
        input.setAttribute('type', 'radio');
        input.setAttribute('name', 'vega-tour-track');
        input.setAttribute('value', name);
        input.setAttribute('title', info.blurb || '');
        if (track === name) { input.checked = true; input.setAttribute('checked', ''); }
        // Both, so a radio VEGA preselected still has an action: pressing the
        // ticked one fires no change, and the click is the student's act.
        input.addEventListener('change', function () { tour.chooseTrack(name, true); });
        input.addEventListener('click', function () { tour.chooseTrack(name, true); });
        var text = make('span', 'vega-tour-opt-name', info.name || name);
        var blurb = make('span', 'vega-tour-opt-blurb', info.blurb || '');
        label.appendChild(input); label.appendChild(text); label.appendChild(blurb);
        els.choice.appendChild(label);
      });
    }

    /** The live line under the body while a run is watched. */
    function liveLine() {
      if (!live) return '';
      var name = STEP_NAMES[live.step] || 'The run';
      var cls = 'vega-tour-live';
      var text;
      if (live.state === 'failed') {
        text = name + ' failed with exit code ' + live.exitCode + '. The trainer\'s last lines are in the log, ringed here, and the run folder keeps everything written before the failure. Back returns to the settings.';
        cls += ' is-failed';
      } else if (live.state === 'finished') {
        text = name + ' has finished' + (live.epoch ? ' after ' + live.epoch.total + ' epochs' : '') + '. The tour moves on in a moment.';
        cls += ' is-done';
      } else if (live.state === 'running') {
        text = name + ' is running.';
        if (live.epoch) text += ' Epoch ' + live.epoch.cur + ' of ' + live.epoch.total + '.';
        if (live.elapsedText) text += ' ' + live.elapsedText + ' elapsed.';
        else text += ' The elapsed time is not known because the run started before the tour was watching.';
        cls += ' is-running';
      } else {
        return '';
      }
      return '<p class="' + cls + '" role="status" aria-live="polite">' + escapeHtml(text) + '</p>';
    }

    /**
     * The line the callout adds when the anchor is not on screen, so a
     * student with no forward button always knows where to go next.
     * @param {object} cur the current step
     * @returns {string} an HTML paragraph, or an empty string
     */
    function whereLine(cur) {
      if (!cur || !cur.anchor || anchorEl) return '';
      var here = surface();
      var text = (cur.surface !== here && SURFACE_NAMES[cur.surface])
        ? 'Go to ' + SURFACE_NAMES[cur.surface] + ' to continue.'
        : WAITING_HERE_TEXT;
      return '<p class="vega-tour-where">' + escapeHtml(text) + '</p>';
    }

    function renderBody(cur, anchored) {
      var text;
      if (cur.surface === 'any' || anchored) {
        var filled = fillPlaceholders(cur.body || '', ctx);
        text = filled === null ? (cur.fallback || '') : filled;
      } else {
        text = cur.fallback || '';
      }
      if (!String(text).trim()) text = cur.body || EMPTY_BODY_TEXT;
      if (!String(text).trim()) text = EMPTY_BODY_TEXT;
      els.body.innerHTML = sanitiseBody(text) + whereLine(cur) + liveLine();
    }

    /** Back is the only button beside Exit tour, and it stops at the first step. */
    function renderButtons() {
      els.back.disabled = idx <= 0;
      if (els.back.disabled) els.back.setAttribute('disabled', ''); else els.back.removeAttribute('disabled');
    }

    function render() {
      var cur = current();
      if (!cur || !els) return;
      clearWatchers();
      followScrolls = 0;
      var here = surface();
      var onSurface = cur.surface === 'any' || cur.surface === here;
      anchorEl = onSurface && cur.anchor ? resolveAnchor(doc, cur.anchor) : null;
      els.count.textContent = track ? 'Step ' + (idx + 1) + ' of ' + steps.length : 'Step ' + (idx + 1);
      els.title.textContent = cur.title || '';
      renderBody(cur, !!anchorEl);
      renderChoice();
      renderButtons();
      els.veil.hidden = cur.surface !== 'any';
      if (anchorEl) {
        // A control taller than the window (a whole project list) scrolls to
        // its top, where the newest rows are, rather than to its middle.
        if (anchorEl.scrollIntoView) {
          var ar = anchorEl.getBoundingClientRect ? anchorEl.getBoundingClientRect() : null;
          var taller = !!(ar && ar.height > viewport().height);
          anchorEl.scrollIntoView({ block: taller ? 'start' : 'center', inline: 'nearest' });
        }
        els.ring.hidden = false;
        followAnchor();
      } else {
        els.ring.hidden = true;
        if (onSurface && cur.anchor) watchForAnchor();
      }
      if (cur.surface === 'orchestrator' && cur.id === 'resume' && !ctx) requestContext();
      place();
      installAdvance(cur);
    }

    /** Position the ring on the anchor and the callout beside it. */
    function place() {
      if (!els) return;
      var vp = viewport();
      var r = els.callout.getBoundingClientRect ? els.callout.getBoundingClientRect() : { width: 0, height: 0 };
      var size = { width: Math.min(CALLOUT_WIDTH, vp.width - 2 * MARGIN), height: r.height || CALLOUT_DEFAULT_HEIGHT };
      var pos;
      if (anchorEl) {
        var a = anchorEl.getBoundingClientRect();
        var ar = { top: a.top - RING_PAD, left: a.left - RING_PAD, width: a.width + 2 * RING_PAD, height: a.height + 2 * RING_PAD };
        els.ring.style.top = ar.top + 'px';
        els.ring.style.left = ar.left + 'px';
        els.ring.style.width = ar.width + 'px';
        els.ring.style.height = ar.height + 'px';
        pos = placeCallout(ar, size, vp);
      } else {
        var cur = current();
        pos = placeFree(size, vp, !!(cur && cur.surface === 'any'));
      }
      els.callout.style.top = pos.top + 'px';
      els.callout.style.left = pos.left + 'px';
      els.callout.style.width = size.width + 'px';
      els.callout.setAttribute('data-side', pos.side);
    }

    /**
     * Keep the ring on its control while the page fills under it. A panel that
     * finishes loading after the step rendered (the table of trained models,
     * a gallery) moves the control down, and a ring left where the control
     * used to be points a student at the wrong thing. Found 2026-09-17 in the
     * screenshot of a student handed off straight to a late step.
     *
     * Only the ring and the callout move, so this can never loop: the watcher
     * listens for elements added or removed, and repositioning writes nothing
     * but style.
     */
    function followAnchor() {
      if (!anchorEl) return;
      var moved = function () { keepInView(); place(); };
      observe(doc.body, { subtree: true, childList: true }, moved);
      if (!RO) return;
      var ro = new RO(moved);
      try { ro.observe(anchorEl); } catch (e) { return; }
      cleanups.push(function () { ro.disconnect(); });
    }

    /**
     * Bring the control back on screen when the page pushed it off. Only a
     * change the page made calls this, never the student's own scrolling, and
     * it gives up after a few tries so it can never fight a student who has
     * deliberately scrolled away to read something.
     */
    function keepInView() {
      if (!anchorEl || !anchorEl.getBoundingClientRect || followScrolls >= FOLLOW_SCROLL_MAX) return;
      var r = anchorEl.getBoundingClientRect();
      var vp = viewport();
      var offScreen = r.bottom <= 0 || r.top >= vp.height || r.right <= 0 || r.left >= vp.width;
      if (!offScreen) return;
      followScrolls += 1;
      if (anchorEl.scrollIntoView) anchorEl.scrollIntoView({ block: 'center', inline: 'nearest' });
    }

    /**
     * Wait for an anchor that is not on the page yet: a short bounded timer
     * for a control that appears without a DOM change, and an observer that
     * waits as long as it takes, because with no forward button a student
     * whose panel opens a minute later must still be picked up.
     */
    function watchForAnchor() {
      retriesLeft = RETRY_MAX;
      var found = function () {
        var cur = current();
        if (!open || !els || !cur || !cur.anchor) return false;
        if (!resolveAnchor(doc, cur.anchor)) return false;
        render();
        return true;
      };
      var tickRetry = function () {
        retryHandle = null;
        if (found()) return;
        if (--retriesLeft > 0) retryHandle = setTimer(tickRetry, RETRY_MS);
      };
      retryHandle = setTimer(tickRetry, RETRY_MS);
      observe(doc.body, { subtree: true, attributes: true, childList: true }, function () { found(); });
    }

    // -- context for the last step ------------------------------------------

    function stageNameFor(state) {
      var key = state && state.current_step !== undefined ? String(state.current_step) : '';
      if (!key) return null;
      var li = null;
      try { li = doc.querySelector('#step-nav li[data-step="' + key + '"]'); } catch (e) { li = null; }
      if (!li) return null;
      var t = li.querySelector('.step-title');
      var s = li.querySelector('.step-status');
      var name = t ? String(t.textContent).trim() : '';
      if (!name) return null;
      var status = s ? String(s.textContent).trim() : '';
      return status ? name + ' (' + status + ')' : name;
    }

    function requestContext() {
      if (ctxRequested) return;
      ctxRequested = true;
      var seq = ++stateFetchSeq;
      getJson(STATE_URL).then(function (j) {
        if (seq !== stateFetchSeq) return;
        var st = j && j.state ? j.state : null;
        if (!st) return;
        ctx = { project_name: st.name || st.id || '', stage_name: stageNameFor(st) };
        var cur = current();
        if (open && els && cur && cur.id === 'resume') render();
      });
    }

    // -- advance watchers ----------------------------------------------------

    function observe(target, options, cb) {
      if (!MO || !target) return null;
      var o = new MO(cb);
      try { o.observe(target, options); } catch (e) { return null; }
      cleanups.push(function () { o.disconnect(); });
      return o;
    }

    /**
     * Watch the one real action that completes this step. Every kind is
     * watched on the live page; none of them is a button of the tour's own.
     * @param {object} cur the current step
     */
    function installAdvance(cur) {
      if (cur.advance === 'navigate') installNavigate(cur);
      else if (cur.advance === 'event') installEvent(cur);
      else if (cur.advance === 'click') installClick(cur);
      else if (cur.advance === 'input') installInput(cur);
    }

    /** True when the event happened on the anchor or inside it. */
    function withinAnchor(target) {
      if (!anchorEl || !target) return false;
      return anchorEl === target || !!(anchorEl.contains && anchorEl.contains(target));
    }

    /**
     * A click step: the student clicks the ringed control, or anything inside
     * it. The listener captures, so a product handler that stops propagation
     * still counts as the action.
     */
    function installClick(cur) {
      if (!anchorEl) return;
      var onClick = function (ev) {
        if (!open || current() !== cur) return;
        if (withinAnchor(ev && ev.target)) advance();
      };
      doc.addEventListener('click', onClick, true);
      cleanups.push(function () { doc.removeEventListener('click', onClick, true); });
    }

    /**
     * An input step: the student fills or changes the ringed field. A typed
     * field must hold a value and settles first, so the glow does not move
     * between two keystrokes; a change event (a select, a radio, a slider
     * released) completes the step at once.
     */
    function installInput(cur) {
      if (!anchorEl) return;
      var settleHandle = null;
      var clearSettle = function () { if (settleHandle) { clearTimer(settleHandle); settleHandle = null; } };
      var act = function (ev, immediate) {
        if (!open || current() !== cur) return;
        var target = ev && ev.target;
        if (!withinAnchor(target)) return;
        if (!hasValue(target)) { clearSettle(); return; }
        if (immediate) { clearSettle(); advance(); return; }
        clearSettle();
        settleHandle = setTimer(function () {
          settleHandle = null;
          if (open && current() === cur && hasValue(target)) advance();
        }, INPUT_SETTLE_MS);
      };
      var onInput = function (ev) { act(ev, false); };
      var onChange = function (ev) { act(ev, true); };
      doc.addEventListener('input', onInput, true);
      doc.addEventListener('change', onChange, true);
      cleanups.push(function () {
        clearSettle();
        doc.removeEventListener('input', onInput, true);
        doc.removeEventListener('change', onChange, true);
      });
    }

    function installNavigate(cur) {
      var target = String(cur.target || '');
      if (!target) return;
      if (target.indexOf('surface:') === 0) {
        if (surface() === target.slice(8)) advance();
        return;
      }
      var check = function () {
        var el = null;
        try { el = doc.querySelector(target); } catch (e) { el = null; }
        if (el) advance();
      };
      check();
      observe(doc.body, { subtree: true, attributes: true, childList: true }, check);
    }

    /**
     * Complete the current step: the next one, or the end of the tour. A step
     * that opens another tab at this same address leaves the moving marker, so
     * the tour on that tab opens where the student was sent.
     */
    function advance() {
      if (!open) return;
      var from = current();
      if (idx >= steps.length - 1) { finish(); return; }
      idx += 1;
      save({ moving: from && from.handoff ? '1' : null });
      render();
    }

    /** The tour is done: the place is kept, the resume offer is not. */
    function finish() {
      save({ done: '1', open: '0' });
      writeState(storage, { open: '0' });
      close();
    }

    /**
     * The first step at or after an index that belongs to the surface the
     * student is on. The tour meets the student where they are when it is
     * opened or resumed on another page; it never moves forward on its own
     * while a step is showing.
     * @param {number} from the index to search from
     * @returns {number} that index, or the one found
     */
    function surfaceEntry(from) {
      var here = surface();
      var start = Math.max(from, 0);
      if (!here || !steps.length) return start;
      for (var i = start; i < steps.length; i++) {
        if (steps[i] && steps[i].surface === here) return i;
      }
      return start;
    }

    /** The step number and mode ("start" or "finish") an event step waits on. */
    function eventSpec(cur) {
      var m = String(cur.event || '').match(/^step:([678])(?::(start))?$/);
      if (!m) return null;
      return { step: parseInt(m[1], 10), mode: m[2] === 'start' ? 'start' : 'finish' };
    }

    function stepEl(n, suffix) { return doc.getElementById('s' + n + '-' + suffix); }

    /** The epoch and elapsed text from the panel's progress line, if any. */
    function readProgress(n) {
      var el = stepEl(n, 'progress-text');
      if (!el) return { epoch: null, panelElapsed: '' };
      var html = String(el.innerHTML || '');
      var parts = html.split(/<\/span>/);
      var epoch = parseEpoch(stripTags(parts[0] || html));
      var panelElapsed = stripTags(parts[1] || '').trim();
      return { epoch: epoch, panelElapsed: panelElapsed };
    }

    function elapsedText(panelElapsed) {
      var st = readState(storage);
      if (st.trainStartedAt) return formatElapsed(now() - st.trainStartedAt);
      return panelElapsed || '';
    }

    function installEvent(cur) {
      var spec = eventSpec(cur);
      if (!spec) return;
      var n = spec.step;
      var confirming = false;
      var settled = false;
      var seq = ++stateFetchSeq;

      var showRunning = function () {
        if (settled) return;
        var p = readProgress(n);
        live = { step: n, state: 'running', epoch: p.epoch, elapsedText: elapsedText(p.panelElapsed) };
        renderBody(cur, !!anchorEl);
        place();
      };
      var showFinished = function (epoch) {
        settled = true;
        live = { step: n, state: 'finished', epoch: epoch || readProgress(n).epoch };
        writeState(storage, { trainStartedAt: null });
        renderBody(cur, !!anchorEl);
        place();
        var handle = setTimer(function () { advance(); }, BEAT_MS);
        cleanups.push(function () { clearTimer(handle); });
      };
      var showFailed = function (code) {
        settled = true;
        live = { step: n, state: 'failed', exitCode: code };
        writeState(storage, { trainStartedAt: null });
        var log = stepEl(n, 'log');
        if (log && isVisible(log)) anchorEl = log;
        renderBody(cur, !!anchorEl);
        place();
      };
      var confirm = function () {
        if (confirming || settled) return;
        confirming = true;
        getJson('/api/step/' + n + '/status').then(function (j) {
          confirming = false;
          if (!j || settled || current() !== cur) return;
          if (j.completed === true || j.exit_code === 0) showFinished(null);
          else if (typeof j.exit_code === 'number' && j.exit_code !== 0) showFailed(j.exit_code);
        });
      };

      // What the project's own record says right now.
      getJson(STATE_URL).then(function (j) {
        if (seq !== stateFetchSeq || current() !== cur || !open) return;
        var st = j && j.state && j.state.steps ? j.state.steps[String(n)] : null;
        var status = st && st.status ? String(st.status) : '';
        if (spec.mode === 'start') {
          if (status === 'running') advance();
          return;
        }
        if (status === 'running') showRunning();
        else if (status === 'completed') showFinished(null);
        else if (status === 'error') confirm();
      });

      var panel = doc.getElementById('panel-' + n) || doc.body;
      observe(panel, { subtree: true, attributes: true, childList: true, characterData: true }, function () {
        if (!open || current() !== cur) return;
        var btn = stepEl(n, 'run-btn');
        if (spec.mode === 'start') {
          if (btn && btn.disabled) advance();
          return;
        }
        var res = stepEl(n, 'result');
        var html = res ? String(res.innerHTML || '') : '';
        if (html.indexOf('badge-success') >= 0 || html.indexOf('badge-error') >= 0) { confirm(); return; }
        if (btn && btn.disabled) showRunning();
        else if (live && live.state === 'running') showRunning();
      });
    }

    // -- carrying the place to another window --------------------------------

    /** The address this page is served from, or an empty string when unknown. */
    function hereOrigin() {
      var loc = win && win.location ? win.location : null;
      if (!loc) return '';
      if (loc.origin) return String(loc.origin).toLowerCase();
      return originOf(loc.href || '');
    }

    /**
     * The address to open, carrying the tour's place when the tour is waiting
     * for the student on a page at another address. Anything else is returned
     * untouched: the tour never edits an address it has no business in.
     * @param {*} url the address the page asked to open
     * @returns {*} the same value, or the address with the place added
     */
    function handoffUrl(url) {
      if (!open || !track || typeof url !== 'string' || !url) return url;
      var cur = current();
      if (!cur) return url;
      // Only while the step is on another page: a window opened for anything
      // else (a dashboard, a folder) is none of the tour's business.
      if (cur.surface === 'any' || cur.surface === surface()) return url;
      var there = originOf(url);
      if (!there || there === hereOrigin()) return url;
      return withHandoff(url, track, cur.id);
    }

    /**
     * Wrap the page's own window opener while the tour is open, so a window
     * the student opens from here arrives carrying the tour's place. The
     * opener's arguments and its return value are the page's own.
     */
    function installOpenWatch() {
      if (nativeOpen || !win || typeof win.open !== 'function') return;
      var original = win.open;
      var wrapped = function (url, name, features) {
        var out = url;
        try { out = handoffUrl(url); } catch (e) { out = url; }
        return original.call(win, out, name, features);
      };
      wrapped.vegaTourWrapped = true;
      try { win.open = wrapped; nativeOpen = original; } catch (e) { nativeOpen = null; }
    }

    /** Put the page's own opener back, and only ever ours. */
    function removeOpenWatch() {
      if (!nativeOpen) return;
      try { if (win.open && win.open.vegaTourWrapped) win.open = nativeOpen; } catch (e) { /* gone with the page */ }
      nativeOpen = null;
    }

    /** The place this page's address carries, or null. */
    function urlHandoff() {
      var loc = win && win.location ? win.location : null;
      if (!loc) return null;
      var search = loc.search !== undefined && loc.search !== null ? loc.search : '';
      var found = parseHandoff(search);
      if (found) return found;
      return parseHandoff(loc.href || '');
    }

    /** Take the handoff parameter out of the address bar, leaving the rest. */
    function clearUrlHandoff() {
      var hist = win && win.history ? win.history : null;
      var loc = win && win.location ? win.location : null;
      if (!hist || typeof hist.replaceState !== 'function' || !loc || !loc.href) return;
      var cleaned = stripHandoff(loc.href);
      if (cleaned === loc.href) return;
      try { hist.replaceState(null, '', cleaned); } catch (e) { /* a page that forbids it */ }
    }

    /** Record the moment the Train button locks, whichever step is open. */
    function installLaunchWatch() {
      if (launchObserver || !MO || surface() !== 'orchestrator') return;
      var btn = doc.getElementById('s6-run-btn');
      if (!btn) return;
      var wasDisabled = !!btn.disabled;
      launchObserver = new MO(function () {
        var nowDisabled = !!btn.disabled;
        if (nowDisabled && !wasDisabled) {
          var st = readState(storage);
          if (!st.trainStartedAt) writeState(storage, { trainStartedAt: now() });
        } else if (!nowDisabled && wasDisabled) {
          // The run ended while the tour was somewhere else; the moment is spent.
        }
        wasDisabled = nowDisabled;
      });
      try { launchObserver.observe(btn, { attributes: true }); } catch (e) { launchObserver = null; }
    }

    function removeLaunchWatch() {
      if (launchObserver) { launchObserver.disconnect(); launchObserver = null; }
    }

    // -- keyboard and focus --------------------------------------------------

    function insideCallout(el) { return !!(els && el && els.root.contains && els.root.contains(el)); }

    /**
     * Escape exits and the left arrow goes back. There is no forward key,
     * because there is no forward control: the page's own action is the only
     * way on. Keys count only on the callout or the bare page, never inside
     * a product field.
     */
    function onKey(ev) {
      if (!open || !els) return;
      var active = doc.activeElement;
      var onBody = !active || active === doc.body;
      if (!onBody && !insideCallout(active)) return;
      var key = ev.key;
      if (key === 'Escape') { ev.preventDefault(); tour.exit(); return; }
      if (key === 'ArrowLeft') { ev.preventDefault(); tour.back(); }
    }

    function installGlobal() {
      if (!keyHandler) {
        keyHandler = onKey;
        doc.addEventListener('keydown', keyHandler);
      }
      if (!placeHandler && win.addEventListener) {
        placeHandler = function () { place(); };
        win.addEventListener('scroll', placeHandler, true);
        win.addEventListener('resize', placeHandler);
      }
      installLaunchWatch();
      installOpenWatch();
    }

    function removeGlobal() {
      if (keyHandler) { doc.removeEventListener('keydown', keyHandler); keyHandler = null; }
      if (placeHandler && win.removeEventListener) {
        win.removeEventListener('scroll', placeHandler, true);
        win.removeEventListener('resize', placeHandler);
        placeHandler = null;
      }
      removeLaunchWatch();
      removeOpenWatch();
    }

    function close() {
      clearWatchers();
      removeGlobal();
      teardownCallout();
      open = false;
      anchorEl = null;
      if (prevFocus && typeof prevFocus.focus === 'function') { try { prevFocus.focus(); } catch (e) { /* gone */ } }
      prevFocus = null;
      refreshChip();
    }

    // -- the chip and the resume offer --------------------------------------

    function refreshChip() {
      if (!chip) return;
      var st = readState(storage);
      var resumable = !!(st.at && !st.done && st.track);
      chip.textContent = resumable ? 'Resume tutorial' : 'Tutorial';
      chip.setAttribute('title', resumable ? CHIP_RESUME_TITLE : CHIP_TITLE);
      chip.hidden = open;
    }

    function buildChip() {
      if (chip) return;
      chip = make('button', 'vega-tour-chip');
      chip.setAttribute('type', 'button');
      chip.addEventListener('click', function () {
        var st = readState(storage);
        if (resumeBox) removeResumeBox();
        if (st.at && !st.done && st.track) tour.start({ track: st.track, step: st.at });
        else tour.start({});
      });
      doc.body.appendChild(chip);
      refreshChip();
    }

    function removeResumeBox() {
      if (resumeBox) { resumeBox.remove(); resumeBox = null; }
    }

    function buildResumeBox(offer) {
      removeResumeBox();
      var box = make('div', 'vega-tour vega-tour-resume');
      var card = make('div', 'vega-tour-callout');
      card.setAttribute('role', 'dialog');
      card.setAttribute('aria-labelledby', 'vega-tour-resume-title');
      card.tabIndex = -1;
      var count = make('div', 'vega-tour-count', 'Step ' + (offer.index + 1) + ' of ' + offer.total);
      var title = make('h2', 'vega-tour-title', 'Resume the tour at step ' + (offer.index + 1));
      title.setAttribute('id', 'vega-tour-resume-title');
      var body = make('div', 'vega-tour-body');
      body.innerHTML = sanitiseBody('The tutorial was left at "' + offer.step.title + '" on the ' + offer.track + ' track. Resume picks up there; Start again returns to the first screen.');
      var actions = make('div', 'vega-tour-actions');
      var resume = button('resume', 'Resume at step ' + (offer.index + 1), true);
      var restart = button('restart', 'Start again', false);
      var dismiss = button('dismiss', 'Not now', false);
      dismiss.className += ' vega-tour-exit';
      resume.addEventListener('click', function () { removeResumeBox(); tour.start({ track: offer.track, step: offer.step.id }); });
      restart.addEventListener('click', function () { removeResumeBox(); writeState(storage, { at: null, furthest: null, done: null }); tour.start({ track: offer.track, step: 'welcome' }); });
      dismiss.addEventListener('click', function () { removeResumeBox(); writeState(storage, { open: '0' }); });
      actions.appendChild(resume); actions.appendChild(restart); actions.appendChild(dismiss);
      card.appendChild(count); card.appendChild(title); card.appendChild(body); card.appendChild(actions);
      box.appendChild(card);
      doc.body.appendChild(box);
      var vp = viewport();
      var pos = placeFree({ width: Math.min(CALLOUT_WIDTH, vp.width - 2 * MARGIN), height: CALLOUT_DEFAULT_HEIGHT }, vp, true);
      card.style.top = pos.top + 'px';
      card.style.left = pos.left + 'px';
      card.style.width = Math.min(CALLOUT_WIDTH, vp.width - 2 * MARGIN) + 'px';
      resumeBox = box;
    }

    function onStorage(ev) {
      if (!ev || ev.key !== KEYS.at || !open) return;
      var id = ev.newValue;
      if (!id || (current() && current().id === id)) return;
      for (var i = 0; i < steps.length; i++) if (steps[i].id === id) { idx = i; render(); return; }
    }

    // -- the public tour -----------------------------------------------------

    var tour = {
      /**
       * Put the Tutorial chip on the page, pick up a place the tour handed to
       * this window, and otherwise offer to resume when the tour was open.
       * A handoff outranks the offer: the student was sent here by the tour a
       * moment ago and has already answered that question.
       */
      mount: function () {
        var sent = urlHandoff();
        if (!surface() && !readState(storage).at && !sent) return;
        buildChip();
        if (win.addEventListener) win.addEventListener('storage', onStorage);
        if (sent) {
          clearUrlHandoff();
          writeState(storage, { moving: null });
          this.start({ track: sent.track, step: sent.step });
          return;
        }
        var st = readState(storage);
        if (st.moving && st.track && st.at && !st.done && !open) {
          writeState(storage, { moving: null });
          this.start({ track: st.track, step: st.at });
          return;
        }
        var offer = this.resumeOffer();
        if (offer && st.open && !open) buildResumeBox(offer);
      },

      /**
       * Open the tour. Options: track ("boxes" or "outlines"), step (an id),
       * surface (override detection). With no track the first screen asks,
       * preselected from real state where the page tells.
       */
      start: function (opts) {
        opts = opts || {};
        if (open) close();
        var st = readState(storage);
        track = TRACKS.indexOf(opts.track) >= 0 ? opts.track : null;
        if (!track && !opts.step && st.track && st.at) track = st.track;
        var here = surface();
        if (!track) {
          if (here === 'boxes') track = 'boxes';
          else if (here === 'annotator') track = 'outlines';
        }
        computeSteps();
        if (!steps.length) return;
        idx = 0;
        if (opts.step) {
          for (var i = 0; i < steps.length; i++) if (steps[i].id === opts.step) { idx = i; break; }
        }
        // Opening or resuming on another page: start at this page's own step
        // rather than one whose control is in a window that is not open.
        var atStep = steps[idx];
        if (atStep && atStep.surface !== 'any' && atStep.surface !== surface()) idx = surfaceEntry(idx);
        open = true;
        prevFocus = doc.activeElement && doc.activeElement !== doc.body ? doc.activeElement : null;
        build();
        installGlobal();
        writeState(storage, { done: null });
        save();
        render();
        refreshChip();
        if (els && els.callout.focus) els.callout.focus();
        if (!track && here === 'orchestrator') preselectFromProject();
      },

      /**
       * Choose the track on the first screen. The track radio is that step's
       * real control: a student pressing one completes the step, and the tour
       * moves to the first step this page can show for the track. VEGA also
       * calls this with no second argument to preselect the track from what
       * the page already shows, which ticks the radio and waits.
       * @param {string} name "boxes" or "outlines"
       * @param {boolean} [byStudent] true when a student pressed the radio
       */
      chooseTrack: function (name, byStudent) {
        if (TRACKS.indexOf(name) < 0) return;
        var cur = current();
        var wasChoosing = !!(cur && cur.surface === 'any');
        if (track === name && !wasChoosing) return;
        track = name;
        computeSteps();
        idx = 0;
        for (var i = 0; i < steps.length; i++) if (cur && steps[i].id === cur.id) { idx = i; break; }
        if (open && wasChoosing && byStudent && steps.length > 1) {
          var target = surfaceEntry(1);
          idx = target > idx ? target : idx + 1;
        }
        save();
        if (open && els) render();
      },

      back: function () {
        if (!open || idx <= 0) return;
        idx -= 1;
        save();
        render();
      },

      goTo: function (id) {
        for (var i = 0; i < steps.length; i++) if (steps[i].id === id) { idx = i; save(); if (open) render(); return true; }
        return false;
      },

      exit: function () {
        if (!open) return;
        save();
        writeState(storage, { open: '0' });
        close();
      },

      /** The saved place, when there is one to resume. */
      resumeOffer: function () {
        var st = readState(storage);
        if (!st.track || !st.at || st.done) return null;
        var list = stepsForTrack(content(), st.track);
        for (var i = 0; i < list.length; i++) {
          if (list[i].id === st.at) return { step: list[i], index: i, total: list.length, track: st.track };
        }
        return null;
      },

      current: current,
      index: function () { return idx; },
      isOpen: function () { return open; },
      track: function () { return track; },
      anchorElement: function () { return anchorEl; },
      steps: function () { return steps.slice(); },
      state: function () { return readState(storage); },
      destroy: function () { if (open) close(); removeResumeBox(); if (chip) { chip.remove(); chip = null; } if (win.removeEventListener) win.removeEventListener('storage', onStorage); }
    };

    /** On the pipeline window, a detection project preselects the boxes track. */
    function preselectFromProject() {
      var seq = ++stateFetchSeq;
      getJson(STATE_URL).then(function (j) {
        if (seq !== stateFetchSeq || !open || track) return;
        var st = j && j.state ? j.state : null;
        if (!st) return;
        var src = st.training_source || {};
        var kind = String(src.kind || '').toLowerCase();
        var ptype = String(st.project_type || '').toLowerCase();
        if (ptype === 'bbox_training' || kind === 'bbox') tour.chooseTrack('boxes');
        else if (ptype === 'composite_training') tour.chooseTrack('outlines');
      });
    }

    return tour;
  }

  return {
    createTour: createTour,
    stepsForTrack: stepsForTrack,
    resolveAnchor: resolveAnchor,
    placeCallout: placeCallout,
    placeFree: placeFree,
    parseEpoch: parseEpoch,
    formatElapsed: formatElapsed,
    sanitiseBody: sanitiseBody,
    detectSurface: detectSurface,
    readState: readState,
    writeState: writeState,
    safeStorage: safeStorage,
    fillPlaceholders: fillPlaceholders,
    originOf: originOf,
    stripHandoff: stripHandoff,
    withHandoff: withHandoff,
    parseHandoff: parseHandoff,
    STORE_PREFIX: STORE_PREFIX,
    HANDOFF_PARAM: HANDOFF_PARAM,
    KEYS: KEYS,
    SURFACE_IDS: SURFACE_IDS,
    TRACKS: TRACKS
  };
});

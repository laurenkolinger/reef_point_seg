/* Coverage Matrix - STATS view. Fetches /matrix/api/stats and draws inline
   SVG charts (instances per species, stacked histograms by year and by site,
   images reviewed per year and site), a reviewers table, the composite
   export list + build form, and the registry hygiene line.
   Species colors are assigned once per page from a validated 8-slot palette
   and follow the species, never its rank; the tail folds into "Other".
   Exposes window.MatrixStats = { activate, refresh, showStage }. No emoji,
   no em dashes. */
(function () {
  'use strict';

  var BASE = (window.MATRIX_BASE || '').replace(/\/$/, '');
  var LS_BY = 'mx.by';
  var LS_STAGE = 'mx.stage';

  // ── DOM handles ────────────────────────────────────────
  var el = {
    view:       document.getElementById('mx-view-stats'),
    head:       document.getElementById('mx-stats-head'),
    status:     document.getElementById('mx-stats-status'),
    refresh:    document.getElementById('mx-stats-refresh'),
    error:      document.getElementById('mx-stats-error'),
    kpis:       document.getElementById('mx-stats-kpis'),
    species:    document.getElementById('mx-stats-species'),
    spAll:      document.getElementById('mx-stats-species-all'),
    spNone:     document.getElementById('mx-stats-species-none'),
    legend:     document.getElementById('mx-stats-legend'),
    chSpecies:  document.getElementById('mx-chart-species'),
    chYear:     document.getElementById('mx-chart-year'),
    chSite:     document.getElementById('mx-chart-site'),
    chImgYear:  document.getElementById('mx-chart-images-year'),
    chImgSite:  document.getElementById('mx-chart-images-site'),
    chExcluded: document.getElementById('mx-stats-excluded'),
    reviewers:  document.getElementById('mx-stats-reviewers'),
    composites: document.getElementById('mx-composites'),
    compName:   document.getElementById('mx-composite-name'),
    by:         document.getElementById('mx-stats-by'),
    compLabels: document.getElementById('mx-composite-labels'),
    compAll:    document.getElementById('mx-composite-labels-all'),
    compNone:   document.getElementById('mx-composite-labels-none'),
    compProjects: document.getElementById('mx-composite-projects'),
    compProjAll:  document.getElementById('mx-composite-projects-all'),
    compProjNone: document.getElementById('mx-composite-projects-none'),
    compRebuild: document.getElementById('mx-composite-rebuild'),
    compBuild:  document.getElementById('mx-composite-build'),
    compResult: document.getElementById('mx-composite-result'),
    hygiene:    document.getElementById('mx-hygiene'),
    purge:      document.getElementById('mx-purge'),
    purgeResult: document.getElementById('mx-purge-result'),
    // Training Sets stage rail (Task 6, 2026-09-03).
    trainView:  document.getElementById('mx-view-training'),
    stage1:     document.getElementById('mx-stage-1'),
    stage2:     document.getElementById('mx-stage-2'),
    stage3:     document.getElementById('mx-stage-3'),
    pane1:      document.getElementById('mx-pane-1'),
    pane2:      document.getElementById('mx-pane-2'),
    pane3:      document.getElementById('mx-pane-3'),
    trainBy:    document.getElementById('mx-train-by'),
    trainProjects: document.getElementById('mx-train-projects')
  };
  if (!el.view) return;

  // Species hues (design spec 9.2): only the first three slots are
  // spec-pinned and contrast-validated against the matrix panel surface
  // #11111a -- Porites astreoides #3987e5 (blue), Siderastrea siderea
  // #d95926 (orange), Orbicella franksii #199e70 (teal), each at least
  // 4.8:1 on --surface. Slots 4-8 extend the palette for a fourth-plus
  // species (amber #e0a23a, violet #8b6fc0, light blue #3aa0e0, orchid
  // #B159CF, slate #64809c) and are neither spec-pinned nor contrast-
  // validated; anything past slot 8 folds into "Other" (OTHER_COLOR).
  var SLOTS = ['#3987e5', '#d95926', '#199e70', '#e0a23a', '#8b6fc0', '#3aa0e0', '#B159CF', '#64809c'];
  var OTHER_COLOR = '#b0a7ab';   // the folded tail; matches --text-dim
  // Grid lines and the zero baseline carry class="grid"/class="axis"
  // (matrix.css: .mx-svg .grid / .mx-svg .axis, stroke: var(--border-dim)
  // / var(--border)), never a hardcoded stroke hex.

  var S = {
    data: null, loading: false, loaded: false,
    selected: {},   // species code -> bool (in the charts)
    slot: {},       // species code -> slot index (sticky for the page)
    compSel: {},    // label code -> bool (training set build form)
    composites: null,
    projects: null,      // /api/data project list: {project_id, name}
    projectsById: {},    // project_id -> that entry, for name lookups
    compProjSel: {},     // project_id -> bool ("Projects" build-form checklist)
    labelNames: {},      // label code -> full name (/api/data label_names, Task 2)
    stage: '1',           // Training Sets rail: the shown stage, '1'|'2'|'3'
    stageActivated: false // showStage(remembered) runs once, on first activate()
  };

  // ── Helpers ────────────────────────────────────────────
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function mk(tag, attrs, text) {
    var n = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === 'class') n.className = attrs[k];
        else n.setAttribute(k, attrs[k]);
      });
    }
    if (text != null) n.textContent = text;
    return n;
  }
  // The only way this view builds a button, input or select: a tooltip is required.
  function control(tag, attrs, text) {
    if (!attrs || !attrs.title) throw new Error('control without tooltip: ' + tag);
    return mk(tag, attrs, text);
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function fmtN(n) { return (Number(n) || 0).toLocaleString(); }
  function orUnrecorded(v) { return v ? String(v) : 'unrecorded'; }
  // Provenance filter (Task 6 /matrix/api/composite contract): the radio
  // value maps to the source_types list posted with the build, null meaning
  // no filter (every source_type, including raw model predictions).
  var SOURCE_TYPE_MAP = {
    all: null,
    hand: ['manual_box', 'manual_click', 'merged', 'edit'],
    hand_model: ['manual_box', 'manual_click', 'merged', 'edit', 'model']
  };
  function selectedSourceType() {
    var checked = document.querySelector('input[name="mx-composite-source"]:checked');
    var key = checked ? checked.value : 'all';
    return key in SOURCE_TYPE_MAP ? SOURCE_TYPE_MAP[key] : null;
  }
  function getBy() { return (el.by.value || '').trim().toUpperCase().slice(0, 8); }
  function getTrainBy() { return el.trainBy ? (el.trainBy.value || '').trim().toUpperCase().slice(0, 8) : ''; }
  // The initials VICARIUS passes as ?initials=XX when it opens this page:
  // trimmed, upper-cased, two to four letters, else "" (ignored). Only ever
  // assigned to a field's .value, never inserted as markup.
  function paramInitials() {
    try {
      var raw = new URLSearchParams(window.location.search).getAll('initials');
      var v = String(raw.length === 1 ? raw[0] : '').trim().toUpperCase();
      return /^[A-Z]{2,4}$/.test(v) ? v : '';
    } catch (e) { return ''; }
  }
  // Rewrites a field's value to upper case as it is typed and keeps the caret.
  function upperAsTyped(input) {
    var upper = input.value.toUpperCase();
    if (upper === input.value) return;
    var at = input.selectionStart;
    input.value = upper;
    try { input.setSelectionRange(at, at); } catch (e) { /* not a text field */ }
  }
  // Seed order: the passed initials, then the remembered ones, then blank.
  // Seeding never writes mx.by; only a typed change does (saveBy).
  function loadBy() {
    var v = paramInitials();
    if (!v) {
      try { v = localStorage.getItem(LS_BY) || ''; } catch (e) { /* storage blocked */ }
    }
    if (v) { el.by.value = v; if (el.trainBy) el.trainBy.value = v; }
  }
  // Shared across the drawer's #mx-by, the build form's #mx-stats-by and the
  // create-project form's #mx-train-by: whichever field changes mirrors into
  // the other two and into localStorage, so one set of initials covers the
  // whole tab.
  function saveBy() {
    try { localStorage.setItem(LS_BY, getBy()); } catch (e) { /* storage blocked */ }
    var twin = document.getElementById('mx-by');
    if (twin && twin !== el.by) twin.value = getBy();
    if (el.trainBy && el.trainBy !== el.by) el.trainBy.value = getBy();
  }
  function saveTrainBy() {
    if (!el.trainBy) return;
    var v = getTrainBy();
    el.trainBy.value = v;
    try { localStorage.setItem(LS_BY, v); } catch (e) { /* storage blocked */ }
    el.by.value = v;
    var twin = document.getElementById('mx-by');
    if (twin) twin.value = v;
  }
  function showError(msg) { el.error.hidden = false; el.error.textContent = msg; }
  function hideError() { el.error.hidden = true; el.error.textContent = ''; }

  function getJson(path) {
    return fetch(BASE + path, { headers: { 'Accept': 'application/json' } }).then(parse);
  }
  function postJson(path, body) {
    return fetch(BASE + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(body || {})
    }).then(parse);
  }
  function parse(r) {
    return r.text().then(function (t) {
      var j = null;
      try { j = t ? JSON.parse(t) : null; } catch (e) { j = null; }
      if (!r.ok || (j && j.error)) {
        var err = new Error((j && j.error) ||
          (r.status === 404 ? 'route not available on this server (404); restart the orchestrator to pick up the stats routes'
                            : 'HTTP ' + r.status));
        // Error bodies carry the path as 'path' (open-folder, purge) or as
        // 'out_dir' (a 409 on a composite name collision); show either.
        err.path = (j && (j.path || j.out_dir)) || '';
        err.status = r.status;
        throw err;
      }
      return j || {};
    });
  }
  function copyText(text, into) {
    function done(ok) { into.textContent = ok ? 'Copied: ' + text : 'Copy failed; select the path and copy it by hand'; }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { done(true); }, fallback);
    } else fallback();
    function fallback() {
      var ta = mk('textarea', null, text);
      ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select();
      var ok = false;
      try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
      document.body.removeChild(ta);
      done(ok);
    }
  }

  // Error line for the build result: the same .result line the success path
  // uses, with the --error modifier (design spec section 5.4); the message,
  // then the path (when the server named one) with a Copy path button so
  // the reader can reach it by hand.
  function resultError(msg, path) {
    clear(el.compResult);
    el.compResult.className = 'result result-error';
    el.compResult.appendChild(mk('span', { 'class': 'ok', 'aria-hidden': 'true' }, 'x'));
    el.compResult.appendChild(document.createTextNode(' ' + msg));
    if (path) {
      el.compResult.appendChild(mk('span', { 'class': 'id' }, path));
      var act = mk('span', { 'class': 'result-act' });
      var note = mk('span', { 'class': 'mx-cell-dim' });
      // Copy path: the label says everything, so mk(), not control().
      var cp = mk('button', { type: 'button', 'class': 'btn btn-mini' }, 'Copy path');
      cp.addEventListener('click', function () { copyText(path, note); });
      act.appendChild(cp); act.appendChild(note);
      el.compResult.appendChild(act);
    }
  }

  // ── Load ───────────────────────────────────────────────
  function activate() {
    if (!S.loaded && !S.loading) load();
    else renderCharts();
    if (S.composites === null) loadComposites();
    if (S.projects === null) loadProjects();
    // The rail remembers which stage was open; that only applies once, on
    // the first activation, so a click made after loading is never undone
    // by the tab losing and regaining focus (setView() calls activate()
    // every time the Training Sets tab is shown, not only the first time).
    if (!S.stageActivated) {
      S.stageActivated = true;
      var remembered = null;
      try { remembered = localStorage.getItem(LS_STAGE); } catch (e) { remembered = null; }
      showStage(remembered || '1');
    }
  }

  function load() {
    if (S.loading) return;
    S.loading = true;
    el.refresh.disabled = true;
    el.head.textContent = 'Computing statistics...';
    hideError();
    getJson('/api/stats?status=' + encodeURIComponent(el.status.value || 'accepted'))
      .then(function (d) {
        S.data = d; S.loaded = true;
        renderAll();
      })
      .catch(function (err) {
        el.head.textContent = 'Statistics unavailable';
        showError('Could not load the statistics: ' + err.message);
      })
      .then(function () { S.loading = false; el.refresh.disabled = false; });
  }

  // ── Color slots (sticky per page) ──────────────────────
  function assignSlots(species) {
    var used = {};
    Object.keys(S.slot).forEach(function (c) { used[S.slot[c]] = true; });
    var ranked = species.slice().sort(function (a, b) { return (b.n_masks || 0) - (a.n_masks || 0); });
    ranked.forEach(function (sp) {
      if (S.slot[sp.code] !== undefined) return;
      for (var i = 0; i < SLOTS.length; i++) {
        if (!used[i]) { used[i] = true; S.slot[sp.code] = i; return; }
      }
    });
  }
  function colorOf(code) {
    return S.slot[code] !== undefined ? SLOTS[S.slot[code]] : OTHER_COLOR;
  }
  function isOther(code) { return S.slot[code] === undefined; }

  // 'YYYY-MM-DD HH:MM AST' from the server stamp (ISO with offset, or the
  // registry's space-separated form); the raw ISO string is the fallback.
  function fmtAst(stamp) {
    var m = /^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})/.exec(String(stamp || ''));
    return m ? m[1] + ' ' + m[2] + ' AST' : (stamp || '');
  }

  // ── Render ─────────────────────────────────────────────
  function renderAll() {
    var d = S.data;
    el.head.textContent = 'Computed ' + fmtAst(d.generated_at) + ', ' +
      (d.status === 'all' ? 'every mask status' : 'accepted masks only');
    var species = d.species || [];
    assignSlots(species);
    species.forEach(function (sp) { if (!(sp.code in S.selected)) S.selected[sp.code] = true; });
    species.forEach(function (sp) { if (!(sp.code in S.compSel)) S.compSel[sp.code] = true; });
    renderKpis();
    renderSpeciesControls();
    renderCharts();
    renderReviewers();
    renderExcluded();
    renderCompLabels();
    renderHygiene();
  }

  // Totals row (design spec 9.2): five plain figures, images/labels/masks/
  // projects/reviewers recorded, mono number + Inter caption. No per-item
  // tooltip; the row's explanation lives on the "Totals ?" glyph in the
  // template (Task 4).
  function renderKpis() {
    clear(el.kpis);
    var t = (S.data && S.data.totals) || {};
    [[t.images, 'images'], [t.labels, 'labels'], [t.masks, 'masks'],
     [t.projects, 'projects'], [t.reviewers, 'reviewers recorded']
    ].forEach(function (row) {
      var tot = mk('span', { 'class': 'tot' });
      tot.appendChild(mk('b', null, fmtN(row[0])));
      tot.appendChild(document.createTextNode(' ' + row[1]));
      el.kpis.appendChild(tot);
    });
  }

  // Species shown row: checkbox, swatch, italic full name, code chip, count.
  // No per-option tooltip (the row already says everything; the "?" glyph
  // beside "Species shown" carries the explanation).
  function renderSpeciesControls() {
    clear(el.species);
    var species = (S.data && S.data.species) || [];
    species.forEach(function (sp) {
      var wrap = mk('label', { 'class': 'opt' });
      var cb = mk('input', { type: 'checkbox', value: sp.code, name: 'species' });
      cb.checked = S.selected[sp.code] !== false;
      cb.addEventListener('change', function () { S.selected[sp.code] = cb.checked; renderCharts(); });
      var sw = mk('span', { 'class': 'sw', 'aria-hidden': 'true' });
      sw.style.background = colorOf(sp.code);
      wrap.appendChild(cb);
      wrap.appendChild(sw);
      var name = speciesFullName(sp.code);
      if (name) wrap.appendChild(mk('span', { 'class': 'name' }, name));
      wrap.appendChild(mk('span', { 'class': 'id code' }, sp.code));
      wrap.appendChild(mk('span', { 'class': 'count' }, fmtN(sp.n_masks)));
      el.species.appendChild(wrap);
    });
  }

  function setAllSpecies(on) {
    Object.keys(S.selected).forEach(function (c) { S.selected[c] = on; });
    Array.prototype.forEach.call(el.species.querySelectorAll('input[type="checkbox"]'), function (cb) { cb.checked = on; });
    renderCharts();
  }

  function selectedSpecies() {
    return ((S.data && S.data.species) || []).filter(function (sp) { return S.selected[sp.code] !== false; });
  }

  // Series for the stacked charts: slotted species in slot order, then Other.
  function seriesFor(selected) {
    var slotted = selected.filter(function (sp) { return !isOther(sp.code); })
      .sort(function (a, b) { return S.slot[a.code] - S.slot[b.code]; })
      .map(function (sp) { return { key: sp.code, codes: [sp.code], color: colorOf(sp.code) }; });
    var others = selected.filter(function (sp) { return isOther(sp.code); }).map(function (sp) { return sp.code; });
    if (others.length) slotted.push({ key: 'Other', codes: others, color: OTHER_COLOR });
    return slotted;
  }

  // Folded into the Species shown row's swatches (design spec 9.2): the
  // tick row is also the color key, so the separate legend box is gone.
  // #mx-stats-legend stays empty and hidden (kept as a call site in case a
  // future chart needs a legend the species row cannot express).
  function renderLegend(series) {
    clear(el.legend);
  }

  function renderCharts() {
    if (!S.data) return;
    var d = S.data;
    var sel = selectedSpecies();
    var series = seriesFor(sel);
    renderLegend(series);
    speciesBars(el.chSpecies, sel);
    stacked(el.chYear, categoriesFor(d.years, d.by_year), d.by_year || {}, series, 'Year');
    // Site order is shared by both site charts (design spec 9.2: "masks
    // total descending, code ascending"), computed once here from every
    // species regardless of the current tick state so the row order never
    // reshuffles as species are toggled.
    var siteCats = rankedSiteCats(categoriesFor(d.sites, d.by_site), d.by_site || {});
    stacked(el.chSite, siteCats, d.by_site || {}, series, 'Site');
    single(el.chImgYear, Object.keys(d.images_by_year || {}).sort(unknownLast), d.images_by_year || {}, 'Year', 'Images');
    single(el.chImgSite, siteCats, d.images_by_site || {}, 'Site', 'Images');
  }
  function numericThenText(a, b) {
    var na = Number(a), nb = Number(b);
    if (isFinite(na) && isFinite(nb)) return na - nb;
    return a < b ? -1 : (a > b ? 1 : 0);
  }
  // Numbers first, then text, with the 'unknown' bucket always last.
  function unknownLast(a, b) {
    if (a === 'unknown' && b !== 'unknown') return 1;
    if (b === 'unknown' && a !== 'unknown') return -1;
    return numericThenText(a, b);
  }
  // Stacked-chart categories: the parsed years/sites the server lists plus
  // every key the per-species buckets carry (the 'unknown' bucket for masks
  // whose year or site could not be parsed), so the columns sum to the KPI.
  function categoriesFor(known, byCode) {
    var seen = {};
    (known || []).forEach(function (k) { seen[String(k)] = true; });
    Object.keys(byCode || {}).forEach(function (code) {
      Object.keys(byCode[code] || {}).forEach(function (c) { seen[String(c)] = true; });
    });
    return Object.keys(seen).sort(unknownLast);
  }

  // ── Name resolution (Task 2: S.data.label_names / S.data.site_names) ──
  // Empty string, never the code, when unresolved: callers decide the
  // fallback (a bare code chip for species, a .term span for sites).
  function speciesFullName(code) { return (S.data && S.data.label_names && S.data.label_names[code]) || ''; }
  function siteName(code) { return (S.data && S.data.site_names && S.data.site_names[code]) || ''; }
  // Species row label (charts + tables): italic full name plus a code chip,
  // or the chip alone when the registry has no name for the code.
  function speciesLabelHtml(code) {
    var name = speciesFullName(code);
    return (name ? '<span class="name">' + esc(name) + '</span> ' : '') + '<span class="id code">' + esc(code) + '</span>';
  }
  // Site row label: full site name plus a code chip; a code the site table
  // cannot resolve (MRS today) stays a bare code, marked as a focusable
  // .term with the "stands alone" tooltip (design spec 9.2). The 'unknown'
  // bucket (no site could be parsed) is neither: plain text, no chip.
  function siteLabelHtml(code) {
    if (code === 'unknown') return esc(code);
    var name = siteName(code);
    if (name) return '<span class="site">' + esc(name) + '</span> <span class="id code">' + esc(code) + '</span>';
    return '<span class="id term" tabindex="0" title="' + esc(code) + ' matches no site in the site table, so the code stands alone.">' + esc(code) + '</span>';
  }
  // Site display order shared by Masks by site and Images reviewed per
  // site (design spec 9.2: "masks total descending, code ascending"),
  // summed over every species in byCode regardless of the current tick
  // state, so the row order never reshuffles as species are toggled.
  function rankedSiteCats(cats, byCode) {
    var totals = {};
    cats.forEach(function (c) {
      totals[c] = Object.keys(byCode).reduce(function (sum, code) {
        return sum + (((byCode[code] || {})[c]) || 0);
      }, 0);
    });
    return cats.slice().sort(function (a, b) {
      if (a === 'unknown' && b !== 'unknown') return 1;
      if (b === 'unknown' && a !== 'unknown') return -1;
      var diff = (totals[b] || 0) - (totals[a] || 0);
      return diff !== 0 ? diff : numericThenText(a, b);
    });
  }

  // ── SVG chart primitives ───────────────────────────────
  // Counts: never a fractional tick.
  function niceStep(max, ticks) {
    if (max <= 0) return 1;
    var raw = max / ticks;
    var pow = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var f = raw / pow;
    var nice = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
    return Math.max(1, nice * pow);
  }
  // Every chart text element: a plain <text>, no class or fill of its own.
  // matrix.css's `.mx-svg text` rule alone gives every one a fixed 10px
  // Space Mono at --text-dim (Task 9 fix round 1: a class+currentColor
  // fill set the color but not the size, so text inherited the 13px body
  // size; the token now comes from the element-selector rule instead).
  function svgText(x, y, text, anchor, extra) {
    return '<text x="' + x + '" y="' + y + '"' +
      (anchor ? ' text-anchor="' + anchor + '"' : '') + (extra || '') + '>' + esc(text) + '</text>';
  }
  function setBody(fig, svg, table) {
    var body = fig.querySelector('.mx-chart-body');
    if (!body) return;
    clear(body);
    // A plain wrapper (no class): matrix.css styles the <svg class="mx-svg">
    // markup it holds directly, so an unstyled "mx-chart-svg" wrapper class
    // carried no rule of its own; dropped 2026-09-03 (M8, dead-CSS sweep).
    var holder = mk('div', null);
    holder.innerHTML = svg;
    body.appendChild(holder);
    if (table) {
      var det = mk('details', { 'class': 'tbl' });
      // No tooltip: "Table view" already says what the disclosure does
      // (tooltip rules 3).
      det.appendChild(mk('summary', null, 'Table view'));
      var twrap = mk('div', { 'class': 'twrap' });
      twrap.appendChild(table);
      det.appendChild(twrap);
      body.appendChild(det);
    }
  }
  // head/row cells: a plain string, or {html: '...'} for a name+chip cell
  // (species/site rows and the year/site tables' species column heads).
  function setCell(node, v) {
    if (v && typeof v === 'object' && 'html' in v) node.innerHTML = v.html;
    else node.textContent = v;
  }
  function tableOf(head, rows) {
    var t = mk('table', { 'class': 't' });
    var tr = mk('tr');
    head.forEach(function (h, i) { var th = mk('th', { 'class': i ? 'num' : '' }); setCell(th, h); tr.appendChild(th); });
    t.appendChild(mk('thead')).appendChild(tr);
    var tb = mk('tbody');
    rows.forEach(function (r) {
      var row = mk('tr');
      r.forEach(function (c, i) { var td = mk('td', { 'class': i ? 'num' : '' }); setCell(td, c); row.appendChild(td); });
      tb.appendChild(row);
    });
    t.appendChild(tb);
    return t;
  }
  function empty(fig, msg) {
    setBody(fig, '<div class="mx-chart-empty">' + esc(msg) + '</div>', null);
  }

  // Horizontal chart primitive (design spec 9.2 / mockups/stats): a
  // div.hbar holding an HTML label column (div.ylabs, one div.yl per row,
  // positioned to its bar's centre) beside the SVG. x positions and bar
  // widths are percentages of the SVG's own width; y positions and text
  // are pixels, so the bars scale with the column while text stays a
  // fixed size. rows: [{ labelHtml, total, segs: [{value, color|dim,
  // title}] }]; a dim segment (the plain "images" bars) is drawn at 55%
  // opacity of --text-dim via class="mx-cell-dim" fill="currentColor"
  // fill-opacity="0.55", never a species color, and carries no title.
  function hbar(fig, rows, colClass, rowH, barH, ariaLabel, tableHead, tableRows) {
    var body = fig.querySelector('.mx-chart-body');
    if (!body) return;
    clear(body);
    var n = rows.length;
    if (!n) { body.appendChild(mk('div', { 'class': 'mx-chart-empty' }, 'Nothing to chart.')); return; }
    var top = 20;
    var H = top + n * rowH + 4;
    var max = Math.max.apply(null, rows.map(function (r) { return r.total || 0; }).concat([1]));
    var step = niceStep(max, 4), top_ = Math.ceil(max / step) * step;
    var pv = function (v) { return (top_ > 0 ? (v / top_ * 100) : 0).toFixed(2) + '%'; };

    var labs = '<div class="ylabs ' + colClass + '" aria-hidden="true">';
    rows.forEach(function (r, i) {
      var cy = top + i * rowH + rowH / 2;
      labs += '<span class="yl" style="top:' + cy + 'px">' + r.labelHtml + '</span>';
    });
    labs += '</div>';

    var svg = '<svg class="mx-svg" width="100%" height="' + H + '" role="img" aria-label="' + esc(ariaLabel) + '"><g>';
    for (var g = 0; g <= top_; g += step) {
      svg += svgText(pv(g), '10', fmtN(g), 'middle');
      svg += '<line class="' + (g === 0 ? 'axis' : 'grid') + '" x1="' + pv(g) + '" x2="' + pv(g) + '" y1="' + (top - 5) + '" y2="' + (top + n * rowH) + '"/>';
    }
    rows.forEach(function (r, i) {
      var y = top + i * rowH + (rowH - barH) / 2;
      var acc = 0;
      (r.segs || []).forEach(function (sg) {
        svg += '<rect class="mx-mark' + (sg.dim ? ' mx-cell-dim' : '') + '" x="' + pv(acc) + '" width="' + pv(sg.value) + '" y="' + y + '" height="' + barH + '"' +
          (sg.dim ? ' fill="currentColor" fill-opacity="0.55"' : ' fill="' + sg.color + '"') +
          (sg.title ? ' title="' + esc(sg.title) + '"' : '') + '/>';
        acc += sg.value;
      });
      if (r.total > 0) svg += svgText(pv(r.total), (y + barH - 2), fmtN(r.total), null, ' dx="6"');
    });
    svg += '</g></svg>';

    var wrap = mk('div', { 'class': 'hbar' });
    wrap.innerHTML = labs + svg;
    body.appendChild(wrap);

    if (tableHead && tableRows) {
      var det = mk('details', { 'class': 'tbl' });
      // No tooltip: "Table view" already says what the disclosure does
      // (tooltip rules 3).
      det.appendChild(mk('summary', null, 'Table view'));
      var twrap = mk('div', { 'class': 'twrap' });
      twrap.appendChild(tableOf(tableHead, tableRows));
      det.appendChild(twrap);
      body.appendChild(det);
    }
  }

  // Vertical stacked columns (Masks by year): left-margin ticks, plain
  // <rect> segments (no rounded caps, matching the mockup), percent x /
  // pixel y. Geometry matches mockups/stats/index.html exactly (H=182,
  // 42px left margin, 88% plot width, 62%-of-band bar width).
  function columnsVertical(fig, cats, byCode, series, totals, xLabel, tableHead) {
    var n = cats.length;
    var H = 182, base = 158, top = 18;
    var max = Math.max.apply(null, totals.concat([1]));
    var step = niceStep(max, 4), top_ = Math.ceil(max / step) * step;
    var plotH = base - top;
    var sy = function (v) { return base - (top_ > 0 ? (v / top_) * plotH : 0); };
    var band = n > 0 ? 88 / n : 88, barW = band * 0.62;
    var maxIdx = totals.indexOf(max);
    var svg = '<svg class="mx-svg" width="100%" height="' + H + '" role="img" aria-label="Masks by ' + esc(xLabel.toLowerCase()) + ', stacked by species">';
    for (var g = 0; g <= top_; g += step) svg += svgText('36', sy(g) + 3.5, fmtN(g), 'end');
    svg += '<g transform="translate(42 0)">';
    for (var g2 = step; g2 <= top_; g2 += step) {
      svg += '<line class="grid" x1="0" x2="88.0%" y1="' + sy(g2) + '" y2="' + sy(g2) + '"/>';
    }
    svg += '<line class="axis" x1="0" x2="88.0%" y1="' + sy(0) + '" y2="' + sy(0) + '"/>';
    cats.forEach(function (c, i) {
      var x = i * band + (band - barW) / 2;
      var acc = 0;
      series.forEach(function (s) {
        var v = s.codes.reduce(function (a2, code) { return a2 + (((byCode[code] || {})[c]) || 0); }, 0);
        if (v <= 0) return;
        var y1 = sy(acc + v), y0 = sy(acc);
        var subject = s.key === 'Other' ? 'Other' : (speciesFullName(s.codes[0]) || s.key);
        var when = c === 'unknown' ? 'an unknown year' : c;
        var title = subject + ' has ' + fmtN(v) + ' of the ' + fmtN(totals[i]) + ' masks from ' + when + '.';
        svg += '<rect class="mx-mark" x="' + x.toFixed(2) + '%" width="' + barW.toFixed(2) + '%" y="' + y1 + '" height="' + (y0 - y1) + '" fill="' + s.color + '" title="' + esc(title) + '"/>';
        acc += v;
      });
      if (i === maxIdx && totals[i] > 0) svg += svgText((x + barW / 2).toFixed(2) + '%', sy(totals[i]) - 4, fmtN(totals[i]), 'middle');
      svg += svgText((x + barW / 2).toFixed(2) + '%', sy(0) + 15, c, 'middle');
    });
    svg += '</g></svg>';
    var trows = cats.map(function (c, i) {
      return [c].concat(series.map(function (s) {
        return fmtN(s.codes.reduce(function (a2, code) { return a2 + (((byCode[code] || {})[c]) || 0); }, 0));
      })).concat([fmtN(totals[i])]);
    });
    setBody(fig, svg, tableOf(tableHead, trows));
  }

  // Vertical single-series columns (Images reviewed per year): same
  // geometry as columnsVertical, plain dim (55% opacity) bars, no title.
  function columnsSingle(fig, cats, map, xLabel, yLabel) {
    var n = cats.length;
    var H = 182, base = 158, top = 18;
    var vals = cats.map(function (c) { return Number(map[c]) || 0; });
    var max = Math.max.apply(null, vals.concat([1]));
    var step = niceStep(max, 4), top_ = Math.ceil(max / step) * step;
    var plotH = base - top;
    var sy = function (v) { return base - (top_ > 0 ? (v / top_) * plotH : 0); };
    var band = n > 0 ? 88 / n : 88, barW = band * 0.62;
    var maxIdx = vals.indexOf(max);
    var svg = '<svg class="mx-svg" width="100%" height="' + H + '" role="img" aria-label="' + esc(yLabel) + ' reviewed per ' + esc(xLabel.toLowerCase()) + '">';
    for (var g = 0; g <= top_; g += step) svg += svgText('36', sy(g) + 3.5, fmtN(g), 'end');
    svg += '<g transform="translate(42 0)">';
    for (var g2 = step; g2 <= top_; g2 += step) {
      svg += '<line class="grid" x1="0" x2="88.0%" y1="' + sy(g2) + '" y2="' + sy(g2) + '"/>';
    }
    svg += '<line class="axis" x1="0" x2="88.0%" y1="' + sy(0) + '" y2="' + sy(0) + '"/>';
    cats.forEach(function (c, i) {
      var x = i * band + (band - barW) / 2;
      if (vals[i] > 0) {
        svg += '<rect class="mx-mark mx-cell-dim" fill="currentColor" fill-opacity="0.55" x="' + x.toFixed(2) + '%" width="' + barW.toFixed(2) + '%" y="' + sy(vals[i]) + '" height="' + (sy(0) - sy(vals[i])) + '"/>';
      }
      if (i === maxIdx && max > 0) svg += svgText((x + barW / 2).toFixed(2) + '%', sy(max) - 4, fmtN(max), 'middle');
      svg += svgText((x + barW / 2).toFixed(2) + '%', sy(0) + 15, c, 'middle');
    });
    svg += '</g></svg>';
    setBody(fig, svg, tableOf([xLabel, yLabel], cats.map(function (c, i) { return [c, fmtN(vals[i])]; })));
  }

  // Masks per species: one horizontal bar per species, its own color, a
  // one-sentence tooltip naming its images and projects.
  function speciesBars(fig, sel) {
    if (!sel.length) { empty(fig, 'No species selected.'); return; }
    var rows = sel.slice().sort(function (a, b) { return (b.n_masks || 0) - (a.n_masks || 0); });
    var hrows = rows.map(function (r) {
      var proj = r.n_projects === 1 ? '1 project' : fmtN(r.n_projects) + ' projects';
      var title = (speciesFullName(r.code) || r.code) + ' has ' + fmtN(r.n_masks) + ' masks on ' + fmtN(r.n_images) + ' images from ' + proj + '.';
      return { labelHtml: speciesLabelHtml(r.code), total: r.n_masks || 0, segs: [{ value: r.n_masks || 0, color: colorOf(r.code), title: title }] };
    });
    var trows = rows.map(function (r) { return [{ html: speciesLabelHtml(r.code) }, fmtN(r.n_masks), fmtN(r.n_images), fmtN(r.n_projects)]; });
    hbar(fig, hrows, 'w-sp', 24, 14, 'Masks per species', ['Species', 'Masks', 'Images', 'Projects'], trows);
  }

  // Stacked chart: one column (Year) or horizontal bar (Site) per
  // category, one segment per series. Site routes through hbar(); Year
  // through columnsVertical().
  function stacked(fig, cats, byCode, series, xLabel) {
    if (!cats.length || !series.length) { empty(fig, cats.length ? 'No species selected.' : 'No ' + xLabel.toLowerCase() + ' could be parsed from the image names.'); return; }
    var totals = cats.map(function (c) {
      return series.reduce(function (acc, s) { return acc + s.codes.reduce(function (a2, code) { return a2 + (((byCode[code] || {})[c]) || 0); }, 0); }, 0);
    });
    var head = [xLabel].concat(series.map(function (s) { return s.key === 'Other' ? 'Other' : { html: speciesLabelHtml(s.codes[0]) }; })).concat(['Total']);
    if (xLabel === 'Site') {
      var hrows = cats.map(function (c, i) {
        var segs = [];
        series.forEach(function (s) {
          var v = s.codes.reduce(function (a2, code) { return a2 + (((byCode[code] || {})[c]) || 0); }, 0);
          if (v <= 0) return;
          var subject = s.key === 'Other' ? 'Other' : (speciesFullName(s.codes[0]) || s.key);
          var where = c === 'unknown' ? 'an unknown site' : (siteName(c) || c);
          segs.push({ value: v, color: s.color, title: subject + ' has ' + fmtN(v) + ' of the ' + fmtN(totals[i]) + ' masks at ' + where + '.' });
        });
        return { labelHtml: siteLabelHtml(c), total: totals[i], segs: segs };
      });
      var trows = cats.map(function (c, i) {
        return [{ html: siteLabelHtml(c) }].concat(series.map(function (s) {
          return fmtN(s.codes.reduce(function (a2, code) { return a2 + (((byCode[code] || {})[c]) || 0); }, 0));
        })).concat([fmtN(totals[i])]);
      });
      hbar(fig, hrows, 'w-site', 18, 11, 'Masks by site, stacked by species', head, trows);
      return;
    }
    columnsVertical(fig, cats, byCode, series, totals, xLabel, head);
  }

  // Single-series chart: Images reviewed per year (vertical columns) or
  // per site (horizontal bars, hbar, same row order as Masks by site).
  // Plain dim bars, no title (design spec 9.2: "plain image bars carry none").
  function single(fig, cats, map, xLabel, yLabel) {
    if (!cats.length) { empty(fig, 'No ' + xLabel.toLowerCase() + ' could be parsed from the image names.'); return; }
    if (xLabel === 'Site') {
      var rows = cats.map(function (c) {
        var v = Number(map[c]) || 0;
        return { labelHtml: siteLabelHtml(c), total: v, segs: v > 0 ? [{ value: v, dim: true }] : [] };
      });
      var trows = cats.map(function (c) { return [{ html: siteLabelHtml(c) }, fmtN(Number(map[c]) || 0)]; });
      hbar(fig, rows, 'w-site', 18, 11, yLabel + ' reviewed per ' + xLabel.toLowerCase(), [xLabel, yLabel], trows);
      return;
    }
    columnsSingle(fig, cats, map, xLabel, yLabel);
  }

  // Bar tooltips (showTip/hideTip, removed): every colored mark now
  // carries a plain `title` (a one-sentence tooltip), read by the shared
  // bubble (vicarius_tooltips.js) exactly like every other control's
  // tooltip on the page. No per-view mousemove listener is needed.

  // ── Reviewers table ────────────────────────────────────
  function renderReviewers() {
    clear(el.reviewers);
    var rows = (S.data && S.data.reviewers) || [];
    var total = rows.reduce(function (a, r) { return a + (r.n_masks || 0); }, 0);
    if (!rows.length) { el.reviewers.appendChild(mk('div', { 'class': 'mx-cell-dim' }, 'No masks counted.')); return; }
    var twrap = mk('div', { 'class': 'twrap' });
    twrap.appendChild(tableOf(['Reviewer', 'Masks', 'Share'], rows.map(function (r) {
      return [orUnrecorded(r.initials), fmtN(r.n_masks), total ? Math.round(100 * (r.n_masks || 0) / total) + '%' : '0%'];
    })));
    el.reviewers.appendChild(twrap);
  }

  // ── Left out of these counts ───────────────────────────
  // Masks the totals and charts above never count: a blank species, or a
  // project id the matrix cannot find on disk (design spec 9.2).
  function renderExcluded() {
    if (!el.chExcluded) return;
    var body = el.chExcluded.querySelector('.mx-chart-body');
    if (!body) return;
    clear(body);
    var ex = (S.data && S.data.excluded) || {};
    var unlabeled = Number(ex.unlabeled_rows) || 0;
    var unknown = Number(ex.unknown_project_rows) || 0;
    var ids = (ex.unknown_project_ids || []).join(', ');
    var dl = mk('dl', { 'class': 'excl' });
    dl.appendChild(mk('dt', null, fmtN(unlabeled)));
    dl.appendChild(mk('dd', null, 'masks with no label'));
    dl.appendChild(mk('dt', null, fmtN(unknown)));
    var dd2 = mk('dd', null, 'masks from a project the matrix cannot find');
    if (ids) {
      dd2.appendChild(document.createTextNode(', '));
      dd2.appendChild(mk('span', { 'class': 'id' }, ids));
    }
    dl.appendChild(dd2);
    body.appendChild(dl);
  }

  // ── Projects ("Limit to projects" build-form control, and the name
  //    lookup used to render each composite's "built from" line) ───────
  function loadProjects() {
    getJson('/api/data')
      .then(function (j) {
        S.projects = (j && j.projects) || [];
        S.labelNames = (j && j.label_names) || {};
        S.projectsById = {};
        S.projects.forEach(function (p) { S.projectsById[p.project_id] = p; });
        S.projects.forEach(function (p) {
          if (!(p.project_id in S.compProjSel)) S.compProjSel[p.project_id] = true;
        });
        renderCompProjects();
        renderCompLabels();
        if (S.composites !== null) { renderComposites(); renderTrainingProjects(); }   // refresh names/labels once looked up
      })
      .catch(function (err) {
        S.projects = [];
        S.projectsById = {};
        clear(el.compProjects);
        el.compProjects.appendChild(mk('div', { 'class': 'mx-cell-dim' }, 'Project list unavailable: ' + err.message));
      });
  }
  function renderCompProjects() {
    clear(el.compProjects);
    var projects = S.projects || [];
    if (!projects.length) {
      el.compProjects.appendChild(mk('div', { 'class': 'mx-cell-dim' }, 'No project available yet.'));
      return;
    }
    // A checkbox naming the project needs no tooltip (the label already
    // says everything); the group's own meaning is on the "?" glyph beside
    // the Projects heading in the template.
    projects.forEach(function (p) {
      var name = p.name || p.project_id;
      var wrap = mk('label', { 'class': 'opt' });
      var cb = mk('input', { type: 'checkbox', value: p.project_id });
      cb.checked = S.compProjSel[p.project_id] !== false;
      cb.addEventListener('change', function () { S.compProjSel[p.project_id] = cb.checked; });
      wrap.appendChild(cb); wrap.appendChild(mk('span', { 'class': 'id' }, name));
      el.compProjects.appendChild(wrap);
    });
  }
  function setAllCompProjects(on) {
    Object.keys(S.compProjSel).forEach(function (c) { S.compProjSel[c] = on; });
    Array.prototype.forEach.call(el.compProjects.querySelectorAll('input[type="checkbox"]'), function (cb) { cb.checked = on; });
  }
  // null means "every project" (the pinned /matrix/api/composite contract);
  // a narrower list is sent only when at least one project is unticked.
  function selectedProjectIds() {
    var all = Object.keys(S.compProjSel);
    if (!all.length) return null;
    var chosen = all.filter(function (c) { return S.compProjSel[c] !== false; });
    return (chosen.length === all.length) ? null : chosen;
  }
  function projectName(pid) {
    var p = S.projectsById[pid];
    return (p && p.name) || pid;
  }

  // ── Composites ─────────────────────────────────────────
  function loadComposites() {
    getJson('/api/composites')
      .then(function (j) {
        S.composites = j.composites || [];
        renderRail();
        renderComposites();
        renderTrainingProjects();
      })
      .catch(function (err) {
        S.composites = [];
        clear(el.composites);
        el.composites.appendChild(mk('div', { 'class': 'mx-cell-dim' }, 'Training set list unavailable: ' + err.message));
        renderRail();
        clear(el.trainProjects);
        el.trainProjects.appendChild(mk('div', { 'class': 'mx-cell-dim' }, 'Training set list unavailable: ' + err.message));
      });
  }
  // label code -> full name from /api/data label_names (Task 2), falling
  // back to the bare code when the registry has no name for it.
  function labelFullName(code) { return (S.labelNames && S.labelNames[code]) || code; }
  function labelFullNames(codes) { return (codes || []).map(labelFullName).join(', '); }
  // A run of text and <b> spans, built without string-templating markup.
  function metaItem(parts) {
    var span = mk('span', null);
    parts.forEach(function (p) {
      if (p && typeof p === 'object' && 'b' in p) span.appendChild(mk('b', null, p.b));
      else span.appendChild(document.createTextNode(String(p)));
    });
    return span;
  }
  // 'YYYY-MM-DD HH:MM:SS' AST (the /api/composites created_at shape) with
  // the seconds dropped; anything else that does not match passes through.
  function createdAtNoSeconds(s) {
    var m = /^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}):\d{2}$/.exec(String(s || ''));
    return m ? m[1] : String(s || '');
  }
  // The row-meta line shared by a composite row: images, masks, labels,
  // built-by, and a source-projects disclosure (or the pre-Task-I fallback
  // note when a set was built before source_projects existed).
  function compositeMeta(c) {
    var meta = mk('span', { 'class': 'row-meta' });
    meta.appendChild(metaItem([{ b: fmtN(c.n_images) }, ' images']));
    meta.appendChild(metaItem([{ b: fmtN(c.n_masks) }, ' masks']));
    meta.appendChild(metaItem(['labels ', { b: labelFullNames(c.labels) }]));
    meta.appendChild(metaItem(['built ', { b: createdAtNoSeconds(c.created_at) }, ' by ', { b: c.by || 'unrecorded' }]));
    var sp = c.source_projects;
    if (sp && sp.length) {
      var det = mk('details', { 'class': 'src' });
      det.appendChild(mk('summary', null, fmtN(sp.length) + ' source projects'));
      var ul = mk('ul');
      sp.forEach(function (s) {
        var li = mk('li');
        li.appendChild(mk('b', null, projectName(s.project_id)));
        li.appendChild(document.createTextNode(' ' + fmtN(s.n_masks)));
        ul.appendChild(li);
      });
      det.appendChild(ul);
      meta.appendChild(det);
    } else {
      meta.appendChild(mk('span', { 'class': 'mx-cell-dim' }, 'sources recorded at next build'));
    }
    return meta;
  }
  // The shared initials field is looser than this route needs (maxlength 8,
  // digits allowed, used by Build training set and the drawer's Use these
  // masks too); the live route enforces exactly 1-4 letters
  // (blueprint.py _INITIALS_RE) and folds an out-of-range value into the
  // training project's folder name, so this checks it here rather than
  // letting a value that works everywhere else on the tab fail only on this
  // action, discovered server-side.
  var TRAIN_INITIALS_RE = /^[A-Za-z]{1,4}$/;
  function createTrainingProject(c, btn, note) {
    var by = getTrainBy();
    if (!by) { el.trainBy.focus(); note.textContent = 'Enter your initials first; the training project records who created it.'; return; }
    if (!TRAIN_INITIALS_RE.test(by)) {
      el.trainBy.focus();
      note.textContent = 'Training projects need 1 to 4 letter initials (they go into the folder name)';
      return;
    }
    saveTrainBy();
    btn.disabled = true;
    note.textContent = 'Creating the training project...';
    postJson('/api/composite/create-project', { name: c.name, by: by })
      .then(function (j) {
        note.textContent = 'Created ' + (j.display_name || j.project_id || '') + '. It is listed in stage 3.';
        loadComposites();
      })
      .catch(function (err) {
        note.textContent = 'Could not create the training project: ' + err.message;
        btn.disabled = false;
      });
  }
  function renderComposites() {
    clear(el.composites);
    var cs = S.composites || [];
    if (!cs.length) { el.composites.appendChild(mk('div', { 'class': 'mx-cell-dim' }, 'No training set built yet.')); return; }
    cs.forEach(function (c) {
      var row = mk('li', { 'class': 'row' });
      row.appendChild(mk('span', { 'class': 'row-name' }, c.name || ''));

      // Copy path: the label says everything, so this is mk(), not
      // control() (no tooltip, per the tooltip rules' rule 3).
      var act1 = mk('span', { 'class': 'row-act' });
      var note1 = mk('span', { 'class': 'mx-cell-dim' });
      var copyBtn = mk('button', { type: 'button', 'class': 'btn btn-mini' }, 'Copy path');
      copyBtn.addEventListener('click', function () { copyText(c.path || '', note1); });
      act1.appendChild(copyBtn); act1.appendChild(note1);
      row.appendChild(act1);

      row.appendChild(compositeMeta(c));

      var linked = c.linked_projects || [];
      var state = mk('span', { 'class': linked.length ? 'row-state' : 'row-state wait' });
      if (linked.length) {
        state.appendChild(mk('span', { 'class': 'ok', 'aria-hidden': 'true' }, String.fromCharCode(0x2713)));
        state.appendChild(document.createTextNode(' Training project '));
        state.appendChild(mk('span', { 'class': 'id' }, linked.join(', ')));
      } else {
        state.textContent = 'No training project yet';
      }
      row.appendChild(state);

      var act2 = mk('span', { 'class': 'row-act' });
      var note2 = mk('span', { 'class': 'mx-cell-dim' });
      if (linked.length) {
        var again = control('button', {
          type: 'button', 'class': 'btn',
          title: 'Create a second training project from this set, for example to train with different settings.'
        }, 'Create another');
        again.addEventListener('click', function () { createTrainingProject(c, again, note2); });
        act2.appendChild(again);
      } else {
        var mainBtn = control('button', {
          type: 'button', 'class': 'btn btn-primary',
          title: 'VICARIUS creates the project from this set and lists it in stage 3.'
        }, 'Create training project');
        mainBtn.addEventListener('click', function () { createTrainingProject(c, mainBtn, note2); });
        act2.appendChild(mainBtn);
      }
      act2.appendChild(note2);
      row.appendChild(act2);

      el.composites.appendChild(row);
    });
  }

  // ── Stage rail (Task 6, 2026-09-03) ────────────────────
  var STAGE_BTNS = { '1': el.stage1, '2': el.stage2, '3': el.stage3 };
  var STAGE_PANES = { '1': el.pane1, '2': el.pane2, '3': el.pane3 };
  function showStage(n) {
    n = String(n);
    if (!STAGE_BTNS[n]) n = '1';
    Object.keys(STAGE_BTNS).forEach(function (i) {
      var active = i === n;
      if (STAGE_BTNS[i]) STAGE_BTNS[i].setAttribute('aria-selected', active ? 'true' : 'false');
      if (STAGE_PANES[i]) STAGE_PANES[i].hidden = !active;
    });
    S.stage = n;
    try { localStorage.setItem(LS_STAGE, n); } catch (e) { /* storage blocked */ }
  }
  // Rail counts and node state, from S.composites: how many training sets
  // exist, how many still need a training project, how many training
  // projects there are in total to train.
  function renderRail() {
    var cs = S.composites || [];
    var nSets = cs.length;
    var nNoProject = cs.filter(function (c) { return !(c.linked_projects || []).length; }).length;
    var nProjects = cs.reduce(function (a, c) { return a + (c.linked_projects || []).length; }, 0);
    var withProject = nSets - nNoProject;

    var meta1 = nSets === 1 ? '1 set built' : fmtN(nSets) + ' sets built';
    var meta2 = nSets === 0 ? 'no sets yet'
      : nNoProject === 0 ? 'all sets have a project'
      : nNoProject === 1 ? '1 set needs a project'
      : fmtN(nNoProject) + ' sets need a project';
    var meta3 = nProjects === 0 ? 'no project yet'
      : nProjects === 1 ? '1 project to train'
      : fmtN(nProjects) + ' projects to train';

    var m1 = el.stage1 && el.stage1.querySelector('.stage-meta');
    var m2 = el.stage2 && el.stage2.querySelector('.stage-meta');
    var m3 = el.stage3 && el.stage3.querySelector('.stage-meta');
    if (m1) m1.textContent = meta1;
    if (m2) m2.textContent = meta2;
    if (m3) m3.textContent = meta3;

    var node1 = el.stage1 && el.stage1.querySelector('.node');
    var node2 = el.stage2 && el.stage2.querySelector('.node');
    if (node1) node1.classList.toggle('done', nSets > 0);
    if (node2) {
      node2.classList.toggle('done', nSets > 0 && nNoProject === 0);
      node2.classList.toggle('part', withProject > 0 && nNoProject > 0);
    }
  }
  // The orchestrator's own origin: BASE is the matrix mount prefix (e.g.
  // "/matrix"); its index route lives at BASE with that suffix stripped.
  // Matches matrix_cell.js's orchestratorOrigin() exactly.
  function orchestratorOrigin() { return BASE.replace(/\/matrix\/?$/, ''); }
  // Reuses the drawer's Open in pipeline contract exactly: same confirm
  // wording, same window.open target name.
  function openInPipeline(dir) {
    if (!window.confirm('Switches the pipeline window to this project and closes any open annotator.')) return;
    var url = orchestratorOrigin() + '/?project_dir=' + encodeURIComponent(dir);
    window.open(url, 'reefpointseg_orchestrator');
  }
  // Stage 3: one row per (training set, linked training project); a set
  // with no project yet gets a single waiting row that points back to
  // stage 2.
  function renderTrainingProjects() {
    clear(el.trainProjects);
    var cs = S.composites || [];
    if (!cs.length) { el.trainProjects.appendChild(mk('div', { 'class': 'mx-cell-dim' }, 'No training set built yet.')); return; }
    var any = false;
    cs.forEach(function (c) {
      var linked = c.linked_projects || [];
      var dirs = c.linked_project_dirs || [];
      if (!linked.length) {
        any = true;
        var row = mk('li', { 'class': 'row' });
        row.appendChild(mk('span', { 'class': 'row-name' }, c.name || ''));
        row.appendChild(mk('span', { 'class': 'row-state wait' }, 'Training set, no project yet'));
        var act = mk('span', { 'class': 'row-act' });
        act.appendChild(mk('button', { type: 'button', 'class': 'btn btn-mini', 'data-go': '2' }, 'Go to stage 2'));
        row.appendChild(act);
        el.trainProjects.appendChild(row);
        return;
      }
      linked.forEach(function (name, i) {
        any = true;
        var dir = dirs[i] || '';
        var row = mk('li', { 'class': 'row' });
        row.appendChild(mk('span', { 'class': 'row-name' }, name));
        var act = mk('span', { 'class': 'row-act' });
        var openBtn = control('button', {
          type: 'button', 'class': 'btn btn-primary',
          title: 'Training starts only when the Train step is run there.'
        }, 'Open in pipeline');
        openBtn.addEventListener('click', function () { openInPipeline(dir); });
        act.appendChild(openBtn);
        row.appendChild(act);
        var meta = mk('span', { 'class': 'row-meta' });
        meta.appendChild(metaItem(['from ', { b: c.name || '' }]));
        meta.appendChild(metaItem([{ b: fmtN(c.n_images) }, ' images']));
        meta.appendChild(metaItem([{ b: fmtN(c.n_masks) }, ' masks']));
        meta.appendChild(metaItem(['labels ', { b: labelFullNames(c.labels) }]));
        meta.appendChild(metaItem(['opens at ', { b: 'Train' }]));
        row.appendChild(meta);
        el.trainProjects.appendChild(row);
      });
    });
    if (!any) el.trainProjects.appendChild(mk('div', { 'class': 'mx-cell-dim' }, 'No training set built yet.'));
  }

  function renderCompLabels() {
    clear(el.compLabels);
    var species = (S.data && S.data.species) || [];
    // With no species under the current filter a build would write an empty
    // training set (nc: 0, negatives only) that Step 6 then lists as a source.
    el.compBuild.disabled = !species.length;
    if (!species.length) {
      el.compLabels.appendChild(mk('div', { 'class': 'mx-cell-dim' }, 'No label to export under the current status filter.'));
      el.compResult.textContent = 'Nothing to build: no masks match the current status filter, so no training set can be exported.';
    } else if ((el.compResult.textContent || '').indexOf('Nothing to build') === 0) {
      el.compResult.textContent = '';
    }
    // A checkbox naming a full species and a count needs no tooltip (the
    // label already says everything), so this goes through mk(), not
    // control(): a name, a code chip and a count, one option per line.
    species.forEach(function (sp) {
      var wrap = mk('label', { 'class': 'opt' });
      var cb = mk('input', { type: 'checkbox', value: sp.code });
      cb.checked = S.compSel[sp.code] !== false;
      cb.addEventListener('change', function () { S.compSel[sp.code] = cb.checked; });
      wrap.appendChild(cb);
      wrap.appendChild(mk('span', { 'class': 'name' }, labelFullName(sp.code)));
      wrap.appendChild(mk('span', { 'class': 'id code' }, sp.code));
      wrap.appendChild(mk('span', { 'class': 'count' }, fmtN(sp.n_masks)));
      el.compLabels.appendChild(wrap);
    });
  }
  function setAllCompLabels(on) {
    Object.keys(S.compSel).forEach(function (c) { S.compSel[c] = on; });
    Array.prototype.forEach.call(el.compLabels.querySelectorAll('input[type="checkbox"]'), function (cb) { cb.checked = on; });
  }

  function buildComposite() {
    var by = getBy();
    if (!by) { el.by.focus(); el.compResult.textContent = 'Enter your initials first; the training set records who built it.'; return; }
    saveBy();
    var species = (S.data && S.data.species) || [];
    if (!S.data) { el.compResult.textContent = 'Statistics have not loaded yet; wait for them before building a training set.'; return; }
    if (!species.length) { el.compResult.textContent = 'Nothing to build: no masks match the current status filter, so no training set can be exported.'; return; }
    var all = Object.keys(S.compSel);
    var chosen = all.filter(function (c) { return S.compSel[c] !== false; });
    if (!chosen.length) { el.compResult.textContent = 'Tick at least one label.'; return; }
    var name = (el.compName.value || '').trim();
    if (name && !/^[A-Za-z0-9._-]+$/.test(name)) { el.compResult.textContent = 'Name may use letters, digits, dot, underscore and dash only.'; return; }
    el.compBuild.disabled = true;
    // A prior attempt may have left the error modifier on; reset the result
    // line to plain 'result' before this attempt's progress/success content.
    el.compResult.className = 'result';
    clear(el.compResult);
    el.compResult.textContent = 'Building the training set...';
    postJson('/api/composite', {
      name: name || null,
      labels: (chosen.length === all.length) ? null : chosen,
      by: by,
      source_types: selectedSourceType(),
      project_ids: selectedProjectIds(),
      rebuild: !!(el.compRebuild && el.compRebuild.checked)
    })
      .then(function (j) {
        clear(el.compResult);
        el.compResult.appendChild(mk('span', { 'class': 'ok', 'aria-hidden': 'true' }, String.fromCharCode(0x2713)));
        el.compResult.appendChild(document.createTextNode(' Built ' + (j.name || name || '') + ': ' +
          fmtN(j.n_images) + ' images, ' + fmtN(j.n_masks) + ' masks, '));
        el.compResult.appendChild(mk('span', {
          'class': 'term', tabindex: '0',
          title: 'Negatives are images in the set with no mask for these labels. The model learns from them what the labels do not look like.'
        }, fmtN(j.n_negatives) + ' negatives'));
        if (j.skipped_review_only) {
          el.compResult.appendChild(mk('div', { 'class': 'mx-cell-dim' },
            fmtN(j.skipped_review_only) + ' image(s) skipped (pending review only).'));
        }
        var act = mk('span', { 'class': 'result-act' });
        var note = mk('span', { 'class': 'mx-cell-dim' });
        // Copy path: the label says everything, so mk(), not control().
        var cp = mk('button', { type: 'button', 'class': 'btn btn-mini' }, 'Copy path');
        cp.addEventListener('click', function () { copyText(j.out_dir || '', note); });
        var op = control('button', { type: 'button', 'class': 'btn btn-mini', title: 'Open the training set folder in the file manager on the orchestrator machine.' }, 'Open folder');
        op.addEventListener('click', function () {
          note.textContent = 'Opening...';
          // The blueprint accepts {path} for a directory directly under
          // inprocess/_composite (and {project_id} for a run folder).
          postJson('/api/open-folder', { path: j.out_dir || '' })
            .then(function (r) { note.textContent = 'Opened ' + (r.path || j.out_dir || ''); })
            .catch(function (err) {
              var p = err.path || j.out_dir || '';
              note.textContent = 'Could not open: ' + err.message + (p ? ' (' + p + ')' : '') +
                (p ? '; use Copy path and open it by hand' : '');
            });
        });
        // Go to stage 2: the label says everything, so mk(), not control().
        var go = mk('button', { type: 'button', 'class': 'btn btn-mini', 'data-go': '2' }, 'Go to stage 2');
        act.appendChild(cp); act.appendChild(op); act.appendChild(go); act.appendChild(note);
        el.compResult.appendChild(act);
        loadComposites();
      })
      .catch(function (err) {
        // A 409 (name already taken) names the existing folder as out_dir;
        // parse() folds it into err.path so it shows with Copy path.
        resultError('Could not build the training set: ' + err.message +
          (err.status === 409 ? '. Pick another name or leave it blank for a stamped one.' : ''), err.path);
      })
      .then(function () { el.compBuild.disabled = !((S.data && S.data.species) || []).length; });
  }

  // ── Registry hygiene ───────────────────────────────────
  function renderHygiene() {
    clear(el.hygiene);
    var ex = (S.data && S.data.excluded) || {};
    var d = S.data || {};
    var residue = ex.test_residue_rows || 0;
    var unknown = ex.unknown_project_rows || 0;
    // The server may predate these counters; absent means none.
    var unkYear = Number(d.unknown_year_masks) || 0;
    var unkSite = Number(d.unknown_site_masks) || 0;
    var unlabeled = Number(ex.unlabeled_rows) || 0;
    el.hygiene.appendChild(mk('div', null, 'Leftover rows from automated tests: ' + fmtN(residue) + '. ' +
      'Rows from a project the matrix cannot find: ' + fmtN(unknown) +
      ((ex.unknown_project_ids || []).length ? ' (' + ex.unknown_project_ids.join(', ') + ')' : '') + '.'));
    el.hygiene.appendChild(mk('div', null, 'Masks whose year could not be parsed from the image name (charted as unknown): ' + fmtN(unkYear) +
      '. Masks whose site could not be parsed: ' + fmtN(unkSite) + '. ' +
      'Rows with a blank species (not counted anywhere): ' + fmtN(unlabeled) + '.'));
    el.purge.disabled = residue === 0;
  }
  function purge() {
    var ex = (S.data && S.data.excluded) || {};
    var n = ex.test_residue_rows || 0;
    if (!window.confirm('Remove ' + n + ' test-fixture row(s) from the mask registry? A backup of manifest.csv is written first.')) return;
    el.purge.disabled = true;
    el.purgeResult.textContent = 'Purging...';
    postJson('/api/registry/purge-test-rows', { confirm: true })
      .then(function (j) {
        el.purgeResult.textContent = 'Removed ' + fmtN(j.removed) + ' row(s). Backup: ' + (j.backup || 'none written');
        S.loaded = false;
        load();
      })
      .catch(function (err) {
        el.purgeResult.textContent = 'Purge failed: ' + err.message + (err.path ? ' (' + err.path + ')' : '');
        el.purge.disabled = false;
      });
  }

  // ── Wiring ─────────────────────────────────────────────
  el.status.addEventListener('change', function () { S.loaded = false; load(); });
  el.refresh.addEventListener('click', function () { S.loaded = false; load(); loadComposites(); loadProjects(); });
  el.spAll.addEventListener('click', function () { setAllSpecies(true); });
  el.spNone.addEventListener('click', function () { setAllSpecies(false); });
  el.compAll.addEventListener('click', function () { setAllCompLabels(true); });
  el.compNone.addEventListener('click', function () { setAllCompLabels(false); });
  el.compProjAll.addEventListener('click', function () { setAllCompProjects(true); });
  el.compProjNone.addEventListener('click', function () { setAllCompProjects(false); });
  el.compBuild.addEventListener('click', buildComposite);
  el.purge.addEventListener('click', purge);
  el.by.addEventListener('input', function () { upperAsTyped(el.by); });
  el.by.addEventListener('change', saveBy);
  if (el.trainBy) {
    el.trainBy.addEventListener('input', function () { upperAsTyped(el.trainBy); });
    el.trainBy.addEventListener('change', saveTrainBy);
  }
  // Stage rail: click a stage button to show it; click any [data-go] button
  // (built into a result line or a row, data-go="2") to jump there too.
  Object.keys(STAGE_BTNS).forEach(function (i) {
    if (STAGE_BTNS[i]) STAGE_BTNS[i].addEventListener('click', function () { showStage(i); });
  });
  if (el.trainView) {
    el.trainView.addEventListener('click', function (e) {
      var btn = e.target.closest && e.target.closest('[data-go]');
      if (btn) showStage(btn.getAttribute('data-go'));
    });
  }
  var resizeTimer = null;
  window.addEventListener('resize', function () {
    if (el.view.hidden || !S.data) return;
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(renderCharts, 150);
  });
  loadBy();

  window.MatrixStats = {
    activate: activate,
    refresh: function () { S.loaded = false; load(); loadComposites(); loadProjects(); },
    showStage: showStage
  };
})();

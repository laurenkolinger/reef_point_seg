/* Coverage Matrix - fetch /matrix/api/data, render the image x label grid,
   re-render on filter change. The grid is the hero; everything else serves
   reading the gaps at a glance. No emoji, no em dashes. */
(function () {
  'use strict';

  // ── Basename parsing + name lookups ────────────────────
  // A reviewed image's stem reads TCRMP<YYYYMMDD>_clip_<SITE>_T<transect>.
  var STEM_RE = /^TCRMP(\d{8})_clip_([A-Za-z0-9]+)_T(\d+)/;
  function labelName(code) { return (DATA && DATA.label_names && DATA.label_names[code]) || code; }
  function siteName(code) { return (DATA && DATA.site_names && DATA.site_names[code]) || code; }
  function whereFromStem(stem) {
    var m = STEM_RE.exec(stem || '');
    if (!m) return null;
    var d = m[1];
    return { site: m[2], siteName: siteName(m[2]), transect: 'T' + m[3],
             date: d.slice(0, 4) + '-' + d.slice(4, 6) + '-' + d.slice(6, 8) };
  }
  window.MatrixNames = { label: labelName, site: siteName, where: whereFromStem };

  var BASE = (window.MATRIX_BASE || '').replace(/\/$/, '');

  // ── DOM handles ────────────────────────────────────────
  var elHeadline = document.getElementById('mx-headline');
  var elSearch   = document.getElementById('mx-search');
  var elOutcome  = document.getElementById('mx-outcome');
  var elLabels   = document.getElementById('mx-labels');
  var elLblAll   = document.getElementById('mx-labels-all');
  var elLblNone  = document.getElementById('mx-labels-none');
  var elGridWrap = document.getElementById('mx-grid-wrap');
  var elGrid     = document.getElementById('mx-grid');
  var elThead    = document.getElementById('mx-thead');
  var elTbody    = document.getElementById('mx-tbody');
  var elEmpty    = document.getElementById('mx-empty');
  var elPop      = document.getElementById('mx-pop');
  var elGenerated = document.getElementById('mx-generated');
  var elRescan   = document.getElementById('mx-rescan');
  var elProject  = document.getElementById('mx-project');
  var elProjectNote = document.getElementById('mx-project-note');
  var elClearFilters = document.getElementById('mx-clear-filters');
  var elCopyCsv  = document.getElementById('mx-copy-csv');

  // ── State ──────────────────────────────────────────────
  var DATA = null;            // the matrix contract
  var visibleLabels = {};     // label -> bool (column shown)
  var loading = false;        // a fetch is in flight (guards re-entry)
  var lastLoadAt = 0;         // ms epoch of the last completed load (throttle)
  var projectFilter = 'all';  // 'all' or a project_id to isolate
  var PROJECT_STATS = {};     // project_id -> {pid, name, touches, unique}
  var projectNoteTimer = null; // pending restore for a temporary #mx-project-note announcement (Copy as CSV)
  var pendingNote = null;      // { html, hidden } captured once per announcement burst; a retrigger before it fires reuses it instead of snapshotting the announcement itself

  var OUTCOME_RANK = {
    found_expert: 5, found_manual: 4, found_ai: 3, found_model: 2,
    pending_expert: 1, not_found: 0
  };
  var OUTCOME_WORDS = {
    found_expert: 'expert verified',
    found_manual: 'found by hand',
    found_ai: 'found, AI assisted',
    found_model: 'found by model',
    pending_expert: 'awaiting expert review',
    not_found: 'reviewed, not found'
  };
  var SOURCE_WORDS = {
    step4test: 'step 4, Place + Segment',
    step5: 'step 5, Segment Images',
    derived: 'derived from step 5, Segment Images',
    model: 'model prediction'
  };

  // ── Helpers ────────────────────────────────────────────
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  // Clipboard write with an execCommand fallback for a context without the
  // async Clipboard API. matrix_cell.js and matrix_stats.js each carry the
  // same pattern; this page has no shared module system (three separate
  // closures loaded as plain scripts), so matrix.js keeps its own copy.
  function copyText(text, onDone) {
    function fallback() {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      var ok = false;
      try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
      document.body.removeChild(ta);
      onDone(ok);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { onDone(true); }, fallback);
    } else {
      fallback();
    }
  }

  // Quote a CSV field only when it needs it (a comma, quote or line break).
  function csvField(v) {
    var s = String(v == null ? '' : v);
    if (/["\r\n,]/.test(s)) s = '"' + s.replace(/"/g, '""') + '"';
    return s;
  }

  function cellAt(image, label) {
    var byLabel = DATA.cells[image];
    if (!byLabel) return null;
    return byLabel[label] || null;
  }

  // ── Per-project provenance (for the "isolate a project" control) ──
  // The set of project_ids that contributed a source to this cell.
  function cellSourceIds(cell) {
    var ids = {};
    if (cell && cell.sources) {
      for (var i = 0; i < cell.sources.length; i++) {
        ids[cell.sources[i].project_id] = true;
      }
    }
    return ids;
  }
  function cellHasProject(cell, pid) {
    return !!cell && cellSourceIds(cell)[pid] === true;
  }
  // A cell is unique to pid when pid is its ONLY source: archiving pid removes it.
  function cellUniqueTo(cell, pid) {
    var ids = Object.keys(cellSourceIds(cell));
    return ids.length === 1 && ids[0] === pid;
  }
  function labelTouchedByProject(label, pid) {
    for (var i = 0; i < DATA.images.length; i++) {
      if (cellHasProject(cellAt(DATA.images[i], label), pid)) return true;
    }
    return false;
  }
  function rowTouchedByProject(image, labels, pid) {
    for (var i = 0; i < labels.length; i++) {
      if (cellHasProject(cellAt(image, labels[i]), pid)) return true;
    }
    return false;
  }
  // Count, per project, cells it touches and cells unique to it.
  function computeProjectStats() {
    var stats = {};
    (DATA.projects || []).forEach(function (p) {
      stats[p.project_id] = {
        pid: p.project_id, name: p.name || p.project_id, run_dir: p.run_dir || '',
        touches: 0, unique: 0, partial: !!p.partially_registered
      };
    });
    Object.keys(DATA.cells).forEach(function (img) {
      var byLabel = DATA.cells[img];
      Object.keys(byLabel).forEach(function (lbl) {
        var ids = Object.keys(cellSourceIds(byLabel[lbl]));
        var only = ids.length === 1 ? ids[0] : null;
        ids.forEach(function (pid) {
          if (!stats[pid]) stats[pid] = { pid: pid, name: pid, touches: 0, unique: 0 };
          stats[pid].touches++;
          if (only === pid) stats[pid].unique++;
        });
      });
    });
    return stats;
  }

  // The footer stamp reads the server's ISO timestamp (with fractional
  // seconds and a UTC offset) straight from /api/data; this trims it to the
  // AST 24 h form the design spec asks for ("2026-09-03 14:06 AST"), the
  // same shape matrix_cell.js and matrix_stats.js each return from their
  // own copy of this helper (M3, 2026-09-03 whole-batch review).
  function fmtAst(stamp) {
    var m = /^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})/.exec(String(stamp || ''));
    return m ? m[1] + ' ' + m[2] + ' AST' : (stamp || '');
  }

  function fmtSource(s) { return SOURCE_WORDS[s] || (s || 'unknown source'); }
  function fmtAt(at) { return at ? at : 'an unrecorded time'; }
  function fmtReviewer(r) { return r ? r : 'unrecorded'; }
  // A mask uid reads SITE-DATE-Tnnn-xNyM-hash; the tail is the distinctive part.
  function shortUid(uid) {
    var parts = String(uid).split('-');
    return parts.length >= 4 ? parts.slice(-3).join('-') : String(uid);
  }

  // ── Fetch ──────────────────────────────────────────────
  // The server rescans the inprocess/ tree on every call, so a re-fetch is all
  // it takes to drop an archived project (and its uniquely-contributed cells)
  // out of the grid. Guarded against re-entry so overlapping triggers (the
  // Rescan button plus the auto-refresh-on-focus) never fire two fetches.
  function load() {
    if (loading) return;
    loading = true;
    if (elRescan) { elRescan.disabled = true; elRescan.classList.add('is-busy'); }
    elHeadline.textContent = 'Loading the coverage matrix...';
    fetch(BASE + '/api/data', { headers: { 'Accept': 'application/json' } })
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (data) {
        if (data && data.error) throw new Error(data.error);
        DATA = data;
        boot();
      })
      .catch(function (err) {
        elHeadline.textContent = 'Could not load the matrix: ' + err.message;
        showEmpty(false);
      })
      .then(function () {
        loading = false;
        lastLoadAt = Date.now();
        if (elRescan) { elRescan.disabled = false; elRescan.classList.remove('is-busy'); }
      });
  }

  function showEmpty(isEmpty) {
    elEmpty.hidden = !isEmpty;
    elGrid.hidden = isEmpty;
  }

  function boot() {
    // mount-aware generated stamp
    if (DATA.generated_at) {
      elGenerated.textContent = 'Rescanned ' + fmtAst(DATA.generated_at);
    }

    if (!DATA.projects || DATA.projects.length === 0) {
      elHeadline.textContent = '';
      elLabels.innerHTML = '';
      showEmpty(true);
      return;
    }
    showEmpty(false);

    // Keep the reader's column show/hide choices across a rescan; a label that
    // did not exist before defaults to visible. (First load: prev is empty, so
    // every label starts visible.)
    var prev = visibleLabels || {};
    visibleLabels = {};
    DATA.labels.forEach(function (lbl) {
      visibleLabels[lbl] = (lbl in prev) ? prev[lbl] : true;
    });

    buildHeadline();
    buildLabelControls();
    buildProjectControl();
    render();
    applyHash();
  }

  // Deep links, applied once after the first load: #cell=<image>|<label>
  // opens that cell's drawer, #stats opens the STATS tab.
  var hashApplied = false;
  function applyHash() {
    if (hashApplied) return;
    hashApplied = true;
    var h = '';
    try { h = decodeURIComponent((location.hash || '').replace(/^#/, '')); } catch (e) { h = ''; }
    if (h === 'stats') { setView('stats', false); return; }
    if (h === 'training') { setView('training', false); return; }
    if (h === 'training-2' || h === 'training-3') { setView('training', false); window.MatrixStats.showStage(h.slice(-1)); return; }
    var m = /^cell=(.+)\|([^|]+)$/.exec(h);
    if (m && window.MatrixCell && cellAt(m[1], m[2])) {
      // The drawer lives inside #mx-view-matrix; select that view first so a
      // Stats or Training tab remembered from localStorage cannot hide it.
      setView('matrix', false);
      window.MatrixCell.open(m[1], m[2]);
    }
  }

  // ── Project isolate control ────────────────────────────
  function buildProjectControl() {
    if (!elProject) return;
    PROJECT_STATS = computeProjectStats();
    var order = Object.keys(PROJECT_STATS).map(function (k) { return PROJECT_STATS[k]; });
    // Most-contributing first; empty (0-cell) projects sink to the bottom so it
    // is obvious which projects actually put anything in the grid.
    order.sort(function (a, b) {
      if (b.touches !== a.touches) return b.touches - a.touches;
      return a.name < b.name ? -1 : (a.name > b.name ? 1 : 0);
    });
    var prev = projectFilter;
    elProject.innerHTML = '';
    var optAll = document.createElement('option');
    optAll.value = 'all';
    optAll.textContent = 'All projects';
    elProject.appendChild(optAll);
    // Auto-named projects share a display name until their labels or image
    // counts diverge (three LO_26-08-26 on 2026-08-26), so a name that occurs
    // more than once carries its run folder to keep the options tellable apart.
    var nameCount = {};
    order.forEach(function (s) { nameCount[s.name] = (nameCount[s.name] || 0) + 1; });
    order.forEach(function (s) {
      var o = document.createElement('option');
      o.value = s.pid;
      var label = (nameCount[s.name] > 1 && s.run_dir) ? s.name + ' [' + s.run_dir + ']' : s.name;
      // A project not yet fully in the canonical mask registry (Task 4's
      // partially_registered flag) is marked so its cell counts here are
      // read as a lower bound, not a final total.
      var suffix = s.partial ? ' (partial)' : '';
      o.textContent = label + suffix + ' (' + s.touches + (s.touches === 1 ? ' cell' : ' cells') + ')';
      var titleParts = [s.run_dir ? 'Run folder: ' + s.run_dir : s.name];
      if (s.partial) {
        titleParts.push('Some of this project\'s reviewed cells are not yet backed by the canonical registry; ' +
          'counts may grow after its next export.');
      }
      o.title = titleParts.join(' ');
      elProject.appendChild(o);
    });
    // Keep the prior selection across a rescan if that project still exists.
    if (prev !== 'all' && PROJECT_STATS[prev]) {
      elProject.value = prev;
      projectFilter = prev;
    } else {
      elProject.value = 'all';
      projectFilter = 'all';
    }
  }

  function updateProjectNote(pid) {
    if (!elProjectNote) return;
    // A pending Copy as CSV announcement (announceProjectNote) would
    // otherwise restore stale content over whatever this call writes next;
    // a fresh project-note update always wins.
    if (projectNoteTimer) { clearTimeout(projectNoteTimer); projectNoteTimer = null; }
    pendingNote = null;
    if (pid === 'all' || !PROJECT_STATS[pid]) {
      elProjectNote.hidden = true;
      elProjectNote.innerHTML = '';
      return;
    }
    var s = PROJECT_STATS[pid];
    var shared = s.touches - s.unique;
    elProjectNote.hidden = false;
    if (s.touches === 0) {
      elProjectNote.innerHTML =
        'Isolating <b>' + esc(s.name) + '</b>: it contributes <b>no cells</b> to the matrix, ' +
        'so archiving it changes nothing here.';
    } else {
      elProjectNote.innerHTML =
        'Isolating <b>' + esc(s.name) + '</b>: <b>' + s.touches + '</b> cells, ' +
        '<b>' + s.unique + '</b> unique, <b>' + shared + '</b> shared.';
    }
  }

  // Copy as CSV announces into the same live region a project isolate uses,
  // then restores whatever was there after 3 seconds (never leaves the
  // reader's prior note clobbered by a confirmation line). A second
  // announcement inside the 3-second window must not snapshot the first
  // announcement as "prior" -- that would restore the announcement text
  // forever instead of the real content -- so the true prior state is
  // captured into pendingNote only the first time (when no timer is
  // pending); a retrigger just reuses that record and resets the timer.
  function announceProjectNote(msg) {
    if (!elProjectNote) return;
    if (projectNoteTimer) {
      clearTimeout(projectNoteTimer);
    } else {
      pendingNote = { html: elProjectNote.innerHTML, hidden: elProjectNote.hidden };
    }
    elProjectNote.hidden = false;
    elProjectNote.textContent = msg;
    projectNoteTimer = setTimeout(function () {
      elProjectNote.innerHTML = pendingNote.html;
      elProjectNote.hidden = pendingNote.hidden;
      projectNoteTimer = null;
      pendingNote = null;
    }, 3000);
  }

  // ── Summary headline ───────────────────────────────────
  function buildHeadline() {
    var s = DATA.stats;
    // found_expert counts as found; pending_expert does NOT (still awaiting
    // an expert ID), though it stays in the reviewed denominator.
    var found = (s.by_outcome.found_expert || 0) + s.by_outcome.found_manual +
                s.by_outcome.found_ai + s.by_outcome.found_model;
    var pct = s.n_reviewed_cells > 0
      ? Math.round((found / s.n_reviewed_cells) * 100)
      : 0;
    // #mx-attention sits first (usability rule: attention before detail,
    // design spec 2026-09-02-matrix-window-design.md section 3.1);
    // attentionCount() fills it and hides it at zero right after.
    elHeadline.innerHTML =
      '<span id="mx-attention" hidden></span>' +
      '<b>' + s.n_images + '</b> images x <b>' + s.n_labels + '</b> labels - ' +
      '<b>' + s.n_projects + '</b> projects - ' +
      '<b>' + pct + '%</b> of reviewed cells found';
    attentionCount();
  }

  // Cells still awaiting an expert ID: the one attention role this grid can
  // compute without fetching /api/preferences (stale choices stay a
  // drawer-only concern, per the ledger ruling). Recomputed on every
  // buildHeadline() call, so it updates on every load().
  function attentionCount() {
    var el = document.getElementById('mx-attention');
    var n = 0;
    Object.keys(DATA.cells).forEach(function (img) {
      var byLabel = DATA.cells[img];
      Object.keys(byLabel).forEach(function (lbl) {
        if (byLabel[lbl] && byLabel[lbl].outcome === 'pending_expert') n++;
      });
    });
    if (el) {
      el.hidden = n === 0;
      el.textContent = n === 1 ? '1 needs attention' : n + ' need attention';
    }
    return n;
  }

  // ── Label column show/hide controls ────────────────────
  function buildLabelControls() {
    elLabels.innerHTML = '';
    DATA.labels.forEach(function (lbl) {
      var id = 'mxlbl-' + lbl.replace(/[^A-Za-z0-9_-]/g, '_');
      var wrap = document.createElement('label');
      wrap.className = 'opt';
      var cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.id = id;
      // The label already names the column and its code (mockup: no title on
      // a checkbox whose own row says everything); an empty title suppresses
      // the browser's own tooltip without tripping the "control gets a
      // title" sweep (test_js_tooltips scans for a .title assignment here).
      cb.title = '';
      cb.checked = visibleLabels[lbl] !== false;
      cb.value = lbl;
      cb.addEventListener('change', function () {
        visibleLabels[lbl] = cb.checked;
        render();
      });
      var name = document.createElement('span');
      name.className = 'name';
      name.textContent = labelName(lbl);
      var code = document.createElement('span');
      code.className = 'id code';
      code.textContent = lbl;
      wrap.appendChild(cb);
      wrap.appendChild(name);
      wrap.appendChild(code);
      elLabels.appendChild(wrap);
    });
  }

  function setAllLabels(on) {
    DATA.labels.forEach(function (lbl) { visibleLabels[lbl] = on; });
    Array.prototype.forEach.call(
      elLabels.querySelectorAll('input[type="checkbox"]'),
      function (cb) { cb.checked = on; }
    );
    render();
  }

  // ── Filtering predicates ───────────────────────────────
  function activeLabels() {
    return DATA.labels.filter(function (lbl) { return visibleLabels[lbl]; });
  }

  function rowMatchesSearch(image, q) {
    if (!q) return true;
    return image.toLowerCase().indexOf(q) !== -1;
  }

  // 'needs_attention' is not a stored outcome; it stands for pending_expert,
  // the one attention role this grid can compute (see attentionCount above).
  function outcomeToMatch(outcome) {
    return outcome === 'needs_attention' ? 'pending_expert' : outcome;
  }

  // For the outcome filter we keep a row visible only if it has at least one
  // visible cell matching the selected outcome (in a visible column).
  function rowMatchesOutcome(image, labels, outcome) {
    if (outcome === 'all') return true;
    var want = outcomeToMatch(outcome);
    for (var i = 0; i < labels.length; i++) {
      var cell = cellAt(image, labels[i]);
      if (cell && cell.outcome === want) return true;
    }
    return false;
  }

  // ── Coverage ribbon math ───────────────────────────────
  // reviewed = any populated cell; here "% reviewed" = reviewed / shown-labels
  // (for a row) or reviewed / shown-images (for a column).
  function rowReviewedCount(image, labels) {
    var n = 0;
    for (var i = 0; i < labels.length; i++) {
      if (cellAt(image, labels[i])) n++;
    }
    return n;
  }

  function colReviewedCount(label, images) {
    var n = 0;
    for (var i = 0; i < images.length; i++) {
      if (cellAt(images[i], label)) n++;
    }
    return n;
  }

  // ── Render ─────────────────────────────────────────────
  function render() {
    if (!DATA) return;
    var q = (elSearch.value || '').trim().toLowerCase();
    var outcome = elOutcome.value;
    var pid = projectFilter;
    var labels = activeLabels();

    // Isolating a project narrows the columns to labels it actually touches.
    if (pid !== 'all') {
      labels = labels.filter(function (lbl) { return labelTouchedByProject(lbl, pid); });
    }

    // Which image rows survive the search + outcome (+ project) filter.
    var images = DATA.images.filter(function (img) {
      if (!rowMatchesSearch(img, q)) return false;
      if (!rowMatchesOutcome(img, labels, outcome)) return false;
      if (pid !== 'all' && !rowTouchedByProject(img, labels, pid)) return false;
      return true;
    });

    updateProjectNote(pid);
    renderHead(labels, images);
    renderBody(labels, images, outcomeToMatch(outcome), pid);
    updateClearFilters(q, outcome, pid);
  }

  // Clear filters shows only while a filter is doing something: search
  // text, a non-default outcome or project, or a hidden label column.
  function updateClearFilters(q, outcome, pid) {
    if (!elClearFilters) return;
    var hiddenLabel = DATA.labels.some(function (lbl) { return visibleLabels[lbl] === false; });
    var active = !!q || outcome !== 'all' || pid !== 'all' || hiddenLabel;
    elClearFilters.hidden = !active;
  }

  // Reset every Matrix-tab filter to its default and re-render.
  function clearFilters() {
    elSearch.value = '';
    elOutcome.value = 'all';
    if (elProject) elProject.value = 'all';
    projectFilter = 'all';
    setAllLabels(true);
  }

  // Copy the grid exactly as it is currently shown: one row per visible
  // image, one column per visible label, with the row and column selection
  // (search, outcome, project and label-column filters) read straight from
  // the rendered table. The outcome per cell, though, is always the true
  // one from DATA.cells via cellAt() -- never td.dataset.outcome, which the
  // renderer only sets on a cell that also currently passes the outcome or
  // project filter; reading that would blank out a reviewed cell of a
  // different outcome, indistinguishable from a cell never reviewed.
  function copyCsv() {
    if (!DATA || !elThead.rows[0]) return;
    var headCells = elThead.rows[0].children;
    var labels = [];
    for (var i = 1; i < headCells.length; i++) {
      labels.push(headCells[i].dataset.label);
    }
    var lines = ['image,' + labels.map(function (lbl) { return csvField(labelName(lbl)); }).join(',')];
    var n = elTbody.rows.length;
    Array.prototype.forEach.call(elTbody.rows, function (tr) {
      var image = tr.dataset.image;
      var row = [csvField(image)];
      labels.forEach(function (lbl) {
        var cell = cellAt(image, lbl);
        row.push(csvField(cell ? (OUTCOME_WORDS[cell.outcome] || cell.outcome) : ''));
      });
      lines.push(row.join(','));
    });
    copyText(lines.join('\n'), function (ok) {
      announceProjectNote(ok ? 'Copied ' + n + ' rows'
        : 'Could not copy to the clipboard; select and copy the CSV text by hand.');
    });
  }

  function renderHead(labels, images) {
    var thead = document.createElement('tr');

    // The corner spans two facts (the image name, then its reviewed share)
    // that share one physical column with each data row's single th.mx-
    // rowhead; two head words inside one flex wrapper keep the thead/tbody
    // column count equal (copyCsv and the crosshair both index columns by
    // position) while still labeling both parts, per the design spec's
    // "Column heads Image and Reviewed" rule.
    var corner = document.createElement('th');
    corner.className = 'mx-corner';
    corner.scope = 'col';
    var cornerInner = document.createElement('div');
    cornerInner.className = 'corner-inner';
    var cornerImage = document.createElement('span');
    cornerImage.textContent = 'Image';
    var cornerReviewed = document.createElement('span');
    cornerReviewed.textContent = 'Reviewed';
    cornerInner.appendChild(cornerImage);
    cornerInner.appendChild(cornerReviewed);
    corner.appendChild(cornerInner);
    thead.appendChild(corner);

    labels.forEach(function (lbl) {
      var th = document.createElement('th');
      th.className = 'mx-colhead';
      th.scope = 'col';
      th.dataset.label = lbl;

      var reviewed = colReviewedCount(lbl, images);
      var pct = images.length ? Math.round((reviewed / images.length) * 100) : 0;
      var name = labelName(lbl);
      th.title = 'Reviewers have looked for ' + name + ' (' + lbl + ') on ' + reviewed +
                 ' of ' + images.length + ' images (' + pct + '%); the bar shows that share.';

      var stack = document.createElement('div');
      stack.className = 'lbl-stack';

      var nameSpan = document.createElement('span');
      nameSpan.className = 'lbl-name';
      nameSpan.textContent = name;

      var code = document.createElement('span');
      code.className = 'id code';
      code.textContent = lbl;

      var bar = document.createElement('span');
      bar.className = 'col-bar';
      bar.setAttribute('aria-hidden', 'true');
      var fill = document.createElement('span');
      fill.className = 'col-fill';
      fill.style.height = pct + '%';
      bar.appendChild(fill);

      stack.appendChild(nameSpan);
      stack.appendChild(code);
      stack.appendChild(bar);
      th.appendChild(stack);
      thead.appendChild(th);
    });

    elThead.innerHTML = '';
    elThead.appendChild(thead);
  }

  function renderBody(labels, images, outcome, pid) {
    var frag = document.createDocumentFragment();

    images.forEach(function (image) {
      var tr = document.createElement('tr');
      tr.dataset.image = image;

      // sticky row label + per-row coverage ribbon
      var th = document.createElement('th');
      th.className = 'mx-rowhead';
      th.scope = 'row';
      th.dataset.image = image;

      var reviewed = rowReviewedCount(image, labels);
      var pct = labels.length ? Math.round((reviewed / labels.length) * 100) : 0;

      var inner = document.createElement('div');
      inner.className = 'row-inner';

      var name = document.createElement('div');
      name.className = 'row-name';
      var where = whereFromStem(image);
      name.title = where
        ? 'The full image name is ' + image + ': ' + where.siteName + ' (' + where.site +
          '), transect ' + where.transect + ', ' + where.date + '.'
        : 'The full image name is ' + image + '.';
      var nameSpan = document.createElement('span');
      nameSpan.textContent = image;
      name.appendChild(nameSpan);

      var ribbon = document.createElement('div');
      ribbon.className = 'row-ribbon';
      ribbon.setAttribute('aria-hidden', 'true');
      var rfill = document.createElement('div');
      rfill.className = 'row-ribbon-fill';
      rfill.style.width = pct + '%';
      ribbon.appendChild(rfill);

      var pctEl = document.createElement('div');
      pctEl.className = 'row-pct';
      pctEl.textContent = pct + '%';
      pctEl.title = 'Reviewers have looked at ' + reviewed + ' of the ' + labels.length +
                    ' shown labels on this image.';

      inner.appendChild(name);
      inner.appendChild(ribbon);
      inner.appendChild(pctEl);
      th.appendChild(inner);
      tr.appendChild(th);

      // data cells
      labels.forEach(function (lbl) {
        var td = document.createElement('td');
        td.className = 'cell';
        td.dataset.image = image;
        td.dataset.label = lbl;

        var cell = cellAt(image, lbl);
        var sq = document.createElement('span');
        sq.className = 'sq';
        td.appendChild(sq);

        if (cell) {
          // outcome + project filters dim non-matching reviewed cells to the
          // not-reviewed look; only a passing cell is interactive.
          var passesOutcome = (outcome === 'all' || cell.outcome === outcome);
          var passesProject = (pid === 'all' || cellHasProject(cell, pid));
          if (passesOutcome && passesProject) {
            td.dataset.outcome = cell.outcome;
            td.classList.add('is-reviewed');
            td.tabIndex = 0;
            td.setAttribute('role', 'gridcell');
            var cellText = labelName(lbl) + ' (' + lbl + ') on ' + image + ': ' +
                           (OUTCOME_WORDS[cell.outcome] || cell.outcome);
            td.setAttribute('aria-label', cellText);
            td.title = cellText;
            // When isolating a project, ring the cells unique to it (removed on
            // archive) and dim the ones it shares with another active project.
            if (pid !== 'all') {
              td.classList.add(cellUniqueTo(cell, pid) ? 'is-unique' : 'is-shared');
            }
          }
        }
        tr.appendChild(td);
      });

      frag.appendChild(tr);
    });

    elTbody.innerHTML = '';
    elTbody.appendChild(frag);
  }

  // ── Crosshair + provenance popover (event delegation) ──
  function clearCross() {
    Array.prototype.forEach.call(
      elGrid.querySelectorAll('.row-on-cross, .col-on-cross, .cross-here'),
      function (n) { n.classList.remove('row-on-cross', 'col-on-cross', 'cross-here'); }
    );
  }

  function applyCross(td) {
    clearCross();
    var image = td.dataset.image;
    var colIndex = Array.prototype.indexOf.call(td.parentNode.children, td);

    // row highlight
    Array.prototype.forEach.call(td.parentNode.children, function (c) {
      c.classList.add('row-on-cross');
    });
    // column highlight (same index in every row + header)
    var headCells = elThead.rows[0] ? elThead.rows[0].children : [];
    if (headCells[colIndex]) headCells[colIndex].classList.add('col-on-cross');
    Array.prototype.forEach.call(elTbody.rows, function (row) {
      if (row.children[colIndex]) row.children[colIndex].classList.add('col-on-cross');
    });
    td.classList.add('cross-here');
  }

  function popoverHtml(image, label, cell) {
    var html = '<div class="pop-head">' + esc(image) + ' x ' + esc(labelName(label)) + '</div>';
    (cell.sources || []).forEach(function (s) {
      html += '<div class="pop-src"><div class="pop-line">' +
        'reviewed in <b>' + esc(s.name || s.project_id || 'a project') + '</b> ' +
        'via <b>' + esc(fmtSource(s.source)) + '</b> ' +
        'by <b>' + esc(fmtReviewer(s.reviewer)) + '</b> ' +
        'on <b>' + esc(fmtAt(s.at)) + '</b>' +
        ' - ' + esc(OUTCOME_WORDS[s.outcome] || s.outcome) +
        (s.uid ? ' - mask ' + esc(shortUid(s.uid)) + (s.status ? ' ' + esc(s.status) : '') : '') +
        '</div></div>';
    });
    html += '<div class="pop-hint">click, Enter or Space opens the cell drawer</div>';
    return html;
  }

  function showPop(td) {
    // The drawer is a fixed-position panel above the popover in z-order
    // (z-index 60 vs 50); positionPop only clamps against window.innerWidth,
    // so on a cell still visible to the left of an open drawer the popover
    // could render partly hidden underneath it. Simplest fix: the popover
    // never shows while the drawer is open.
    if (window.MatrixCell && window.MatrixCell.isOpen()) { hidePop(); return; }
    var cell = cellAt(td.dataset.image, td.dataset.label);
    if (!cell) { hidePop(); return; }
    elPop.innerHTML = popoverHtml(td.dataset.image, td.dataset.label, cell);
    elPop.hidden = false;
    positionPop(td);
  }

  function positionPop(td) {
    var r = td.getBoundingClientRect();
    var pad = 8;
    // measure after un-hiding
    var pw = elPop.offsetWidth, ph = elPop.offsetHeight;
    var left = r.right + pad;
    var top = r.top;
    if (left + pw > window.innerWidth - pad) left = r.left - pw - pad;
    if (left < pad) left = pad;
    if (top + ph > window.innerHeight - pad) top = window.innerHeight - ph - pad;
    if (top < pad) top = pad;
    elPop.style.left = left + 'px';
    elPop.style.top = top + 'px';
  }

  function hidePop() {
    elPop.hidden = true;
    elPop.innerHTML = '';
  }

  function onCellEnter(td) {
    if (!td.classList.contains('is-reviewed')) {
      // hover a not-reviewed empty cell shows nothing
      clearCross();
      hidePop();
      return;
    }
    applyCross(td);
    showPop(td);
  }

  // mouse
  elTbody.addEventListener('mouseover', function (e) {
    var td = e.target.closest && e.target.closest('td.cell');
    if (td) onCellEnter(td);
  });
  elGridWrap.addEventListener('mouseleave', function () {
    clearCross();
    hidePop();
  });
  // keyboard: crosshair + popover follow focus
  elTbody.addEventListener('focusin', function (e) {
    var td = e.target.closest && e.target.closest('td.cell');
    if (td) onCellEnter(td);
  });
  elTbody.addEventListener('focusout', function () {
    hidePop();
  });

  // Activation: click, Enter or Space on a reviewed cell opens the drawer
  // (matrix_cell.js). Not-reviewed cells stay inert.
  function openCell(td) {
    if (!td.classList.contains('is-reviewed')) return;
    hidePop();
    // The cell is passed so the drawer can hand focus back to it on close.
    if (window.MatrixCell) window.MatrixCell.open(td.dataset.image, td.dataset.label, td);
  }
  elTbody.addEventListener('click', function (e) {
    var td = e.target.closest && e.target.closest('td.cell');
    if (td) openCell(td);
  });
  elTbody.addEventListener('keydown', function (e) {
    var td = e.target.closest && e.target.closest('td.cell');
    if (!td) return;
    if (e.key !== 'Enter' && e.key !== ' ' && e.key !== 'Spacebar') {
      // Arrow keys walk the grid (registry window design section 3, rule 6):
      // Tab alone would step through every reviewed cell of 726 rows before
      // reaching the row below. Only reviewed cells take focus, so a step in
      // a direction keeps going until it finds one or runs out of grid.
      var step = ARROW_STEPS[e.key];
      if (!step) return;
      var next = neighbourCell(td, step[0], step[1]);
      if (!next) return;
      e.preventDefault();
      next.focus();
      return;
    }
    e.preventDefault();
    openCell(td);
  });

  // key -> [row step, column step]
  var ARROW_STEPS = {
    ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1]
  };

  // The nearest focusable cell from td in one direction, or null at the edge.
  function neighbourCell(td, dRow, dCol) {
    var rows = elTbody.rows;
    var r = td.parentNode.sectionRowIndex;
    var c = td.cellIndex;
    for (var i = 0; i < 400; i++) {
      r += dRow;
      c += dCol;
      if (r < 0 || r >= rows.length) return null;
      var cells = rows[r].cells;
      if (c < 0 || c >= cells.length) return null;
      var cand = cells[c];
      if (cand && cand.classList.contains('cell') && cand.tabIndex === 0) return cand;
      if (dRow === 0 && dCol === 0) return null;
    }
    return null;
  }

  // ── Controls wiring ────────────────────────────────────
  var searchTimer = null;
  elSearch.addEventListener('input', function () {
    if (searchTimer) clearTimeout(searchTimer);
    searchTimer = setTimeout(render, 120);
  });
  elOutcome.addEventListener('change', render);
  if (elProject) elProject.addEventListener('change', function () {
    projectFilter = elProject.value;
    render();
  });
  elLblAll.addEventListener('click', function () { setAllLabels(true); });
  elLblNone.addEventListener('click', function () { setAllLabels(false); });
  if (elClearFilters) elClearFilters.addEventListener('click', clearFilters);
  if (elCopyCsv) elCopyCsv.addEventListener('click', copyCsv);

  // Manual rescan: rebuild the grid from the current inprocess/ tree on demand.
  if (elRescan) elRescan.addEventListener('click', function () { load(); });

  // Auto-rescan when the reader returns to this tab or window, so a project
  // archived (or annotated) elsewhere shows up without a manual click. Only
  // fires once data has loaded, never while a fetch is in flight, and at most
  // once per ~1.5s so a focus+visibility pair does not double-fetch.
  function maybeAutoRefresh() {
    if (DATA && !loading && (Date.now() - lastLoadAt > 1500)) load();
  }
  document.addEventListener('visibilitychange', function () {
    if (document.visibilityState === 'visible') maybeAutoRefresh();
  });
  window.addEventListener('focus', maybeAutoRefresh);

  // ── View tabs (MATRIX / STATS / TRAINING SETS), remembered in localStorage ─
  var elTabMatrix    = document.getElementById('mx-tab-matrix');
  var elTabStats     = document.getElementById('mx-tab-stats');
  var elTabTraining  = document.getElementById('mx-tab-training');
  var elViewMatrix   = document.getElementById('mx-view-matrix');
  var elViewStats    = document.getElementById('mx-view-stats');
  var elViewTraining = document.getElementById('mx-view-training');
  var LS_TAB = 'mx.tab';
  var TABS = {
    matrix:   { tab: elTabMatrix,   view: elViewMatrix },
    stats:    { tab: elTabStats,    view: elViewStats },
    training: { tab: elTabTraining, view: elViewTraining }
  };
  function setView(which, remember) {
    if (!TABS[which]) which = 'matrix';
    Object.keys(TABS).forEach(function (key) {
      var t = TABS[key];
      var active = key === which;
      if (t.view) t.view.hidden = !active;
      if (t.tab) {
        t.tab.classList.toggle('is-active', active);
        t.tab.setAttribute('aria-selected', active ? 'true' : 'false');
      }
    });
    if (which === 'stats' || which === 'training') {
      hidePop();
      clearCross();
      // Composite exports (Training Sets) and the reviewer/chart KPIs
      // (Stats) both read the same /matrix/api/stats fetch, so either tab
      // triggers the same load; matrix_stats.js renders into whichever
      // view's elements exist on the page regardless of which tab called it.
      if (window.MatrixStats) window.MatrixStats.activate();
    }
    if (remember !== false) {
      try { localStorage.setItem(LS_TAB, which); } catch (e) { /* storage blocked */ }
    }
  }
  if (elTabMatrix)   elTabMatrix.addEventListener('click', function () { setView('matrix'); });
  if (elTabStats)    elTabStats.addEventListener('click', function () { setView('stats'); });
  if (elTabTraining) elTabTraining.addEventListener('click', function () { setView('training'); });
  (function restoreTab() {
    var saved = null;
    try { saved = localStorage.getItem(LS_TAB); } catch (e) { saved = null; }
    if (saved === 'stats' || saved === 'training') setView(saved, false);
  })();

  // ── Go ─────────────────────────────────────────────────
  load();
})();

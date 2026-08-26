/* Coverage Matrix - fetch /matrix/api/data, render the image x label grid,
   re-render on filter change. The grid is the hero; everything else serves
   reading the gaps at a glance. No emoji, no em dashes. */
(function () {
  'use strict';

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

  // ── State ──────────────────────────────────────────────
  var DATA = null;            // the matrix contract
  var visibleLabels = {};     // label -> bool (column shown)
  var loading = false;        // a fetch is in flight (guards re-entry)
  var lastLoadAt = 0;         // ms epoch of the last completed load (throttle)
  var projectFilter = 'all';  // 'all' or a project_id to isolate
  var PROJECT_STATS = {};     // project_id -> {pid, name, touches, unique}

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
    step4test: '4.test combined annotator',
    step5: 'Step 5 review',
    derived: 'derived from step 5',
    model: 'model prediction'
  };

  // ── Helpers ────────────────────────────────────────────
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
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
      stats[p.project_id] = { pid: p.project_id, name: p.name || p.project_id, touches: 0, unique: 0 };
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

  function fmtSource(s) { return SOURCE_WORDS[s] || (s || 'unknown source'); }
  function fmtAt(at) { return at ? at : 'an unrecorded time'; }
  function fmtReviewer(r) { return r ? r : 'an unnamed reviewer'; }

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
      elGenerated.textContent = 'Rescanned ' + DATA.generated_at + ' (AST)';
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
    order.forEach(function (s) {
      var o = document.createElement('option');
      o.value = s.pid;
      o.textContent = s.name + ' (' + s.touches + (s.touches === 1 ? ' cell' : ' cells') + ')';
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
        'Isolating <b>' + esc(s.name) + '</b>: <b>' + s.touches + '</b> cells - ' +
        '<b>' + s.unique + '</b> unique (ringed; these leave the matrix if you archive it) - ' +
        '<b>' + shared + '</b> shared (dimmed; another active project keeps them).';
    }
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
    elHeadline.innerHTML =
      '<b>' + s.n_images + '</b> images x <b>' + s.n_labels + '</b> labels - ' +
      '<b>' + s.n_projects + '</b> projects - ' +
      '<b>' + pct + '%</b> of reviewed cells found';
  }

  // ── Label column show/hide controls ────────────────────
  function buildLabelControls() {
    elLabels.innerHTML = '';
    DATA.labels.forEach(function (lbl) {
      var id = 'mxlbl-' + lbl.replace(/[^A-Za-z0-9_-]/g, '_');
      var wrap = document.createElement('label');
      wrap.setAttribute('title', 'Show or hide the ' + lbl + ' column');
      var cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.id = id;
      cb.checked = visibleLabels[lbl] !== false;
      cb.value = lbl;
      cb.addEventListener('change', function () {
        visibleLabels[lbl] = cb.checked;
        render();
      });
      var txt = document.createElement('span');
      txt.textContent = lbl;
      wrap.appendChild(cb);
      wrap.appendChild(txt);
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

  // For the outcome filter we keep a row visible only if it has at least one
  // visible cell matching the selected outcome (in a visible column).
  function rowMatchesOutcome(image, labels, outcome) {
    if (outcome === 'all') return true;
    for (var i = 0; i < labels.length; i++) {
      var cell = cellAt(image, labels[i]);
      if (cell && cell.outcome === outcome) return true;
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
    renderBody(labels, images, outcome, pid);
  }

  function renderHead(labels, images) {
    var thead = document.createElement('tr');

    var corner = document.createElement('th');
    corner.className = 'mx-corner';
    corner.scope = 'col';
    corner.textContent = 'image \\ label';
    thead.appendChild(corner);

    labels.forEach(function (lbl) {
      var th = document.createElement('th');
      th.className = 'mx-colhead';
      th.scope = 'col';
      th.dataset.label = lbl;

      var reviewed = colReviewedCount(lbl, images);
      var pct = images.length ? Math.round((reviewed / images.length) * 100) : 0;

      var code = document.createElement('span');
      code.className = 'col-code';
      code.textContent = lbl;
      code.title = lbl + ' - reviewed in ' + reviewed + ' of ' + images.length +
                   ' shown images (' + pct + '%)';

      var ribbon = document.createElement('div');
      ribbon.className = 'col-ribbon';
      ribbon.setAttribute('aria-hidden', 'true');
      var fill = document.createElement('div');
      fill.className = 'col-ribbon-fill';
      fill.style.height = pct + '%';
      ribbon.appendChild(fill);

      th.appendChild(code);
      th.appendChild(ribbon);
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
      name.title = image;
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
      pctEl.title = 'reviewed in ' + reviewed + ' of ' + labels.length + ' shown labels';

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
            td.setAttribute('aria-label',
              image + ' x ' + lbl + ': ' + (OUTCOME_WORDS[cell.outcome] || cell.outcome));
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
    var html = '<div class="pop-head">' + esc(image) + ' x ' + esc(label) + '</div>';
    (cell.sources || []).forEach(function (s) {
      html += '<div class="pop-src"><div class="pop-line">' +
        'reviewed in <b>' + esc(s.name || s.project_id || 'a project') + '</b> ' +
        'via <b>' + esc(fmtSource(s.source)) + '</b> ' +
        'by <b>' + esc(fmtReviewer(s.reviewer)) + '</b> ' +
        'on <b>' + esc(fmtAt(s.at)) + '</b>' +
        ' - ' + esc(OUTCOME_WORDS[s.outcome] || s.outcome) +
        '</div></div>';
    });
    return html;
  }

  function showPop(td) {
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

  // ── Go ─────────────────────────────────────────────────
  load();
})();

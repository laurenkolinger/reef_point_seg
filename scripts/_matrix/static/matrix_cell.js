/* Coverage Matrix - cell drawer. Opens when a reviewed cell is clicked (or
   activated with Enter/Space), fetches /matrix/api/cell, draws the image with
   that label's masks on a canvas, lists the projects that annotated it, and
   lets the reader choose the canonical masks for composite exports.
   Exposes window.MatrixCell = { open, close, isOpen }. No emoji, no em dashes. */
(function () {
  'use strict';

  var BASE = (window.MATRIX_BASE || '').replace(/\/$/, '');
  var LS_BY = 'mx.by';

  // ── DOM handles ────────────────────────────────────────
  var el = {
    drawer:   document.getElementById('mx-cell'),
    title:    document.getElementById('mx-cell-title'),
    where:    document.getElementById('mx-cell-where'),
    species:  document.getElementById('mx-cell-species'),
    label:    document.getElementById('mx-cell-label'),
    outcome:  document.getElementById('mx-cell-outcome'),
    close:    document.getElementById('mx-cell-close'),
    by:       document.getElementById('mx-by'),
    status:   document.getElementById('mx-cell-status'),
    error:    document.getElementById('mx-cell-error'),
    opacity:  document.getElementById('mx-cell-opacity'),
    others:   document.getElementById('mx-cell-others'),
    wrap:     document.getElementById('mx-cell-canvas-wrap'),
    canvas:   document.getElementById('mx-cell-canvas'),
    tip:      document.getElementById('mx-cell-tip'),
    tabs:     document.getElementById('mx-cell-tabs'),
    masks:    document.getElementById('mx-cell-masks'),
    trainSets: document.getElementById('mx-cell-composites'),
    prefer:   document.getElementById('mx-cell-prefer'),
    clear:    document.getElementById('mx-cell-clear'),
    pref:     document.getElementById('mx-cell-pref'),
    projects: document.getElementById('mx-cell-projects')
  };
  if (!el.drawer || !el.canvas) return;
  var ctx = el.canvas.getContext('2d');

  // ── State ──────────────────────────────────────────────
  var S = {
    image: null, label: null, data: null,
    projectIdx: 0,          // index into data.projects shown on the canvas
    img: null,              // loaded HTMLImageElement
    imgUrl: '',
    scale: 1, ox: 0, oy: 0, // fit transform: canvas = image * scale + offset
    hover: null,            // hovered mask object
    checked: {},            // uid -> bool for the label's masks (current project)
    opacity: 0.45,
    showOthers: false,
    reqSeq: 0,              // guards stale fetch responses
    returnTo: null          // element that activated the drawer; focus goes back to it on close
  };

  var OUTCOME_WORDS = {
    found_expert: 'expert verified',
    found_manual: 'found by hand',
    found_ai: 'found, AI assisted',
    found_model: 'found by model',
    pending_expert: 'awaiting expert review',
    not_found: 'reviewed, not found'
  };
  // Outcome -> the drawer's shared color-modifier classes (.lchip.c-<outcome>,
  // the same suffixes the template's legend and #mx-cell-label already carry
  // per the 2026-09-03 stage-rail redesign; matrix.css binds them).
  var OUTCOME_CLASS = {
    found_expert: 'c-expert', found_manual: 'c-manual',
    found_ai: 'c-ai', found_model: 'c-model', pending_expert: 'c-pending', not_found: 'c-not-found'
  };
  // Mask paint: the label's masks strong, other labels faint.
  var PAINT = {
    label_accepted: '#E20074',
    label_other: '#e0a23a',
    other: '#b0a7ab',
    hover: '#ffffff',
    checked: '#4bbf73'
  };

  // ── Helpers ────────────────────────────────────────────
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
  // Every visible control carries a tooltip; this is the only way the drawer
  // builds a button, input or select, and it refuses one without a title.
  function control(tag, attrs, text) {
    if (!attrs || !attrs.title) throw new Error('control without tooltip: ' + tag);
    return mk(tag, attrs, text);
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  // A mask uid reads SITE-DATE-Tnnn-xNyM-hash; the tail is the distinctive part.
  function short(uid) {
    if (!uid) return '(no uid)';
    var parts = String(uid).split('-');
    return parts.length >= 4 ? parts.slice(-3).join('-') : String(uid);
  }
  function orUnrecorded(v) { return v ? String(v) : 'unrecorded'; }
  // Every server timestamp is AST already; this only reformats the ISO
  // string for display (slice off seconds/fraction, append the zone
  // label) with a plain regex, never a Date-object UTC conversion.
  function fmtAst(stamp) {
    var m = /^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})/.exec(String(stamp || ''));
    return m ? m[1] + ' ' + m[2] + ' AST' : (stamp || '');
  }
  function fmtArea(a) {
    var n = Number(a);
    if (!isFinite(n) || n <= 0) return 'area unknown';
    return Math.round(n).toLocaleString() + ' px';
  }
  // Full names for the codes the drawer shows (design spec 6: never a bare
  // code when a name resolves). window.MatrixNames is defined by matrix.js,
  // which loads after this file, so these are only ever called from inside
  // a render function (once data is on screen), never at parse time; a
  // missing MatrixNames (matrix.js not yet loaded, or an older cached page)
  // falls back to the bare code or a null "where".
  function labelNameFor(code) { return window.MatrixNames ? window.MatrixNames.label(code) : code; }
  function whereForImage(stem) { return window.MatrixNames ? window.MatrixNames.where(stem) : null; }
  // A row-meta value with its own <b>; e.g. metaItem('folder ', run_dir).
  function metaItem(prefix, value) {
    var s = mk('span', null, prefix);
    s.appendChild(mk('b', null, value));
    return s;
  }
  // Bold area figure for a masks-list row-meta line; unlike fmtArea() (used
  // by the canvas hover tip) this never carries its own "px" suffix, so the
  // caller can bold the number and keep "px" as plain text beside it.
  function areaCount(a) {
    var n = Number(a);
    if (!isFinite(n) || n <= 0) return null;
    return Math.round(n).toLocaleString();
  }
  function getBy() {
    var v = (el.by.value || '').trim().toUpperCase().slice(0, 8);
    return v;
  }
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
    var passed = paramInitials();
    if (passed) { el.by.value = passed; return; }
    try { var v = localStorage.getItem(LS_BY); if (v) el.by.value = v; } catch (e) { /* storage blocked */ }
  }
  function saveBy() {
    try { localStorage.setItem(LS_BY, getBy()); } catch (e) { /* storage blocked */ }
    // The stats view and the training-project stage share the same
    // initials; mirror by setting .value directly (never dispatch a change
    // event, which would re-trigger each field's own change handler).
    var twin = document.getElementById('mx-stats-by');
    if (twin && twin !== el.by) twin.value = getBy();
    var trainTwin = document.getElementById('mx-train-by');
    if (trainTwin && trainTwin !== el.by) trainTwin.value = getBy();
  }
  function setStatus(msg, isErr) {
    el.status.textContent = msg || '';
    el.status.classList.toggle('is-err', !!isErr);
  }
  function showError(msg, path) {
    clear(el.error);
    el.error.hidden = false;
    el.error.appendChild(mk('div', { 'class': 'mx-cell-error-msg' }, msg));
    if (path) {
      var row = mk('div', { 'class': 'mx-cell-error-path' });
      row.appendChild(mk('code', null, path));
      // Copy path needs no tooltip (its label already says what it does,
      // design spec 7.3); it is built with mk(), never control(), which
      // would refuse a titleless button.
      row.appendChild(mk('button', {
        type: 'button', 'class': 'btn btn-mini'
      }, 'Copy path')).addEventListener('click', function () { copyText(path); });
      el.error.appendChild(row);
    }
  }
  function hideError() { el.error.hidden = true; clear(el.error); }

  function copyText(text) {
    function done(ok) { setStatus(ok ? 'Copied: ' + text : 'Copy failed; select the path and copy it by hand', !ok); }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { done(true); }, function () { fallback(); });
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

  function postJson(path, body) {
    return fetch(BASE + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(body || {})
    }).then(function (r) {
      return r.text().then(function (t) {
        var j = null;
        try { j = t ? JSON.parse(t) : null; } catch (e) { j = null; }
        if (!r.ok || (j && j.error)) {
          var err = new Error((j && j.error) || ('HTTP ' + r.status));
          err.path = j && j.path;
          err.status = r.status;
          throw err;
        }
        return j || {};
      });
    });
  }

  // ── Open / close ───────────────────────────────────────
  function open(image, label, opener) {
    S.image = image; S.label = label; S.data = null; S.projectIdx = 0;
    S.hover = null; S.checked = {};
    // Remember what activated the drawer (the grid cell, or whatever held
    // focus) so close() can hand focus back to it; a hidden drawer never
    // counts as the opener.
    var active = document.activeElement;
    if (opener && opener.focus) S.returnTo = opener;
    else if (!isOpen() && active && active !== document.body && !el.drawer.contains(active)) S.returnTo = active;
    el.drawer.hidden = false;
    document.body.classList.add('mx-cell-open');
    resetPrefer();
    el.title.textContent = image;
    el.where.hidden = true;
    el.where.textContent = '';
    el.where.removeAttribute('title');
    el.species.textContent = '';
    el.label.textContent = label;
    el.label.className = 'lchip';
    el.outcome.textContent = '';
    clear(el.tabs); clear(el.masks); clear(el.projects); clear(el.pref);
    if (el.trainSets) clear(el.trainSets);
    hideError();
    setStatus('Loading ' + image + ' x ' + label + '...');
    ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
    // Focus moves into the dialog so a keyboard reader reaches its controls
    // next, instead of the rest of the grid.
    try { el.close.focus(); } catch (e) { /* not focusable in this context */ }
    var seq = ++S.reqSeq;
    fetch(BASE + '/api/cell?image=' + encodeURIComponent(image) +
          '&label=' + encodeURIComponent(label), { headers: { 'Accept': 'application/json' } })
      .then(function (r) {
        return r.text().then(function (t) {
          var j = null;
          try { j = t ? JSON.parse(t) : null; } catch (e) { j = null; }
          if (!r.ok || !j || j.error) {
            throw new Error((j && j.error) ||
              (r.status === 404 ? 'The cell route is not available on this server (404). Restart the orchestrator to pick up the drill-down routes.'
                                : 'HTTP ' + r.status));
          }
          return j;
        });
      })
      .then(function (data) {
        if (seq !== S.reqSeq) return;
        S.data = data;
        setStatus('');
        renderAll();
      })
      .catch(function (err) {
        if (seq !== S.reqSeq) return;
        setStatus('Could not load this cell: ' + err.message, true);
      });
  }

  function close() {
    var wasOpen = isOpen();
    el.drawer.hidden = true;
    document.body.classList.remove('mx-cell-open');
    S.reqSeq++;
    S.data = null; S.img = null;
    hideTip();
    // Hand focus back to the cell (or control) that opened the drawer.
    var back = S.returnTo;
    S.returnTo = null;
    if (wasOpen && back && back.focus && document.contains(back)) {
      try { back.focus(); } catch (e) { /* element no longer focusable */ }
    }
  }
  function isOpen() { return !el.drawer.hidden; }
  // Neutral state of the 'Use these masks' button: disabled, generic label.
  // Called before any early return so a cell without a project never keeps
  // the enabled button and label text of the previously opened cell.
  function resetPrefer() {
    el.prefer.disabled = true;
    el.prefer.textContent = 'Use these masks';
  }

  // ── Render ─────────────────────────────────────────────
  function currentProject() {
    var ps = (S.data && S.data.projects) || [];
    return ps[S.projectIdx] || null;
  }
  function labelMasks(p) {
    return ((p && p.masks) || []).filter(function (m) { return m.is_label; });
  }
  function otherMasks(p) {
    return ((p && p.masks) || []).filter(function (m) { return !m.is_label; });
  }

  function renderAll() {
    var d = S.data;
    var img = d.image || S.image;
    var lbl = d.label || S.label;
    var lblName = labelNameFor(lbl);
    el.title.textContent = img;
    var where = whereForImage(img);
    if (where) {
      el.where.hidden = false;
      el.where.textContent = where.siteName + ', transect ' + where.transect + ', ' + where.date;
      el.where.title = 'VICARIUS reads the site, transect and date from the image name; ' +
        where.site + ' is the code for ' + where.siteName + '.';
    } else {
      el.where.hidden = true;
      el.where.textContent = '';
      el.where.removeAttribute('title');
    }
    el.species.textContent = lblName;
    el.label.textContent = lbl;
    el.label.className = 'lchip ' + (OUTCOME_CLASS[d.outcome] || '');
    el.outcome.textContent = OUTCOME_WORDS[d.outcome] || d.outcome || '';
    el.prefer.title = 'Use the ticked masks of the shown project as the masks for ' + lblName +
      ' on this image. Training sets built later take them.';
    el.clear.title = 'Clear the saved choice for ' + lblName +
      ' on this image; training sets then take the strongest project.';
    // Prefer the project the preference points at, else the first.
    var ps = d.projects || [];
    S.projectIdx = 0;
    if (d.preference && d.preference.project_id) {
      ps.forEach(function (p, i) { if (p.project_id === d.preference.project_id) S.projectIdx = i; });
    }
    renderTabs();
    renderProjects();
    renderTrainSets();
    selectProject(S.projectIdx);
  }

  // "In training sets": which composites already draw on this label's masks.
  // Per-cell membership cannot be derived from the /api/composites listing
  // alone (composite_manifest.csv is not served to the browser), so this
  // renders only from an optional "composites" field on the /api/cell
  // payload itself; when the backend does not yet carry it, nothing shows
  // (never a guessed match against the composites list).
  function renderTrainSets() {
    if (!el.trainSets) return;
    clear(el.trainSets);
    var names = S.data && S.data.composites;
    if (!names || !names.length) return;
    el.trainSets.textContent = 'Feeds training sets: ' + names.join(', ');
  }

  function renderTabs() {
    clear(el.tabs);
    var ps = (S.data && S.data.projects) || [];
    if (!ps.length) {
      el.tabs.appendChild(mk('div', { 'class': 'mx-cell-none' }, 'No project holds this image.'));
      return;
    }
    var lblName = labelNameFor((S.data && S.data.label) || S.label);
    ps.forEach(function (p, i) {
      var b = control('button', {
        type: 'button', role: 'tab', 'class': 'mx-cell-tab' + (i === S.projectIdx ? ' is-active' : ''),
        'aria-selected': i === S.projectIdx ? 'true' : 'false',
        title: 'Show the masks ' + (p.name || p.project_id) + ' holds for this image (' +
               (OUTCOME_WORDS[p.outcome_for_label] || p.outcome_for_label || 'no outcome') + ', ' +
               labelMasks(p).length + ' masks of ' + lblName + ').'
      }, p.name || p.project_id);
      b.addEventListener('click', function () { selectProject(i); });
      el.tabs.appendChild(b);
    });
  }

  function selectProject(i) {
    S.projectIdx = i;
    S.hover = null;
    Array.prototype.forEach.call(el.tabs.querySelectorAll('.mx-cell-tab'), function (b, j) {
      b.classList.toggle('is-active', j === i);
      b.setAttribute('aria-selected', j === i ? 'true' : 'false');
    });
    var p = currentProject();
    S.checked = {};
    var pref = S.data && S.data.preference;
    labelMasks(p).forEach(function (m) {
      // Default ticks: the saved choice when it points at this project (all
      // when uids is empty), otherwise every accepted mask.
      if (pref && pref.project_id === p.project_id) {
        S.checked[m.uid] = !pref.uids || !pref.uids.length || pref.uids.indexOf(m.uid) !== -1;
      } else {
        S.checked[m.uid] = m.status === 'accepted';
      }
    });
    renderMasks();
    renderPref();
    loadImage(p);
  }

  function renderMasks() {
    clear(el.masks);
    resetPrefer();
    var p = currentProject();
    var ms = labelMasks(p);
    if (!p) return;
    var lblName = labelNameFor((S.data && S.data.label) || S.label);
    if (!ms.length) {
      el.masks.appendChild(mk('div', { 'class': 'mx-cell-none' },
        'This project holds no mask of ' + lblName + ' on this image.'));
    }
    ms.forEach(function (m, i) {
      // The row is a plain div, not a label: a label cannot wrap another
      // label without giving the checkbox two labelling relationships (and
      // invalid HTML besides). The checkbox gets its own id so the inner
      // <label for> around the uid text is its one and only label. A
      // delegated click listener on the div forwards a click anywhere else
      // in the row to the checkbox; a click on the checkbox itself, or
      // inside the inner label, is left entirely to native browser
      // handling so the checkbox never toggles twice.
      var row = mk('div', { 'class': 'mx-cell-mask', 'data-uid': m.uid || '' });
      var cbId = 'mk-' + i;
      // Tick behaviour is already explained by the section's own "i" glyph
      // ("Masks for this label"); a per-row repeat of the same sentence, and
      // a per-row repeat of the status word already visible beside the
      // checkbox, would say nothing new (tooltip rules 3), so this is
      // mk(), not control().
      var cb = mk('input', { type: 'checkbox', id: cbId });
      cb.checked = !!S.checked[m.uid];
      if (m.status !== 'accepted') {
        cb.disabled = true;
      }
      cb.addEventListener('change', function () { S.checked[m.uid] = cb.checked; draw(); });
      row.addEventListener('click', function (e) {
        if (e.target === cb || e.target.closest('label')) return;
        if (!cb.disabled) cb.click();
      });
      row.appendChild(cb);
      var sw = mk('span', { 'class': 'mx-cell-sw' });
      sw.style.background = maskColor(m, false);
      row.appendChild(sw);
      row.appendChild(mk('label', { 'for': cbId, title: 'The full mask uid is ' + (m.uid || '(none)') + '.' }, short(m.uid)));
      var stCls = m.status === 'accepted' ? 'st-accepted' : (m.status === 'rejected' ? 'st-rejected' : 'st-pending');
      row.appendChild(mk('span', { 'class': 'st ' + stCls }, m.status || 'unlabeled'));
      var meta = mk('span', { 'class': 'row-meta' });
      meta.appendChild(metaItem('source ', m.source_type || 'unknown source'));
      meta.appendChild(metaItem('reviewer ', orUnrecorded(m.reviewer)));
      var areaSpan = mk('span', null);
      var areaN = areaCount(m.area);
      var atText = m.at ? fmtAst(m.at) : '';
      if (areaN) {
        areaSpan.appendChild(mk('b', null, areaN));
        areaSpan.appendChild(document.createTextNode(' px' + (atText ? ', ' + atText : '')));
      } else {
        areaSpan.appendChild(mk('b', null, 'area unknown'));
        if (atText) areaSpan.appendChild(document.createTextNode(', ' + atText));
      }
      meta.appendChild(areaSpan);
      row.appendChild(meta);
      row.addEventListener('mouseenter', function () { S.hover = m; draw(); });
      row.addEventListener('mouseleave', function () { S.hover = null; draw(); });
      el.masks.appendChild(row);
    });
    // The button text stays fixed ("Use these masks", set once in
    // resetPrefer); the label name lives in the tooltip only (renderAll),
    // so it never grows past the 440px drawer.
    el.prefer.disabled = !ms.some(function (m) { return m.status === 'accepted'; });
  }

  function renderPref() {
    clear(el.pref);
    var pref = S.data && S.data.preference;
    if (!pref) {
      // Plain dim state line, not the green .result block: nothing has been
      // saved, so this is not a finished state (I3, 2026-09-03 whole-batch
      // review). The saved-choice branch below restores .result.
      el.pref.className = 'row-state';
      el.pref.appendChild(mk('span', { 'class': 'mx-cell-dim' },
        'No choice saved: training sets take the strongest project for this cell.'));
      el.clear.disabled = true;
      el.clear.classList.remove('is-prominent');
      return;
    }
    el.pref.className = 'result';
    el.clear.disabled = false;
    var pname = pref.project_id;
    var found = false;
    ((S.data && S.data.projects) || []).forEach(function (p) {
      if (p.project_id === pref.project_id) { pname = p.name || p.project_id; found = true; }
    });
    // A saved choice that points at a project no longer on disk (the server
    // may flag it as stale, or no listed project matches) is not applied by
    // composites; say so and make Clear choice the prominent action.
    var stale = !!pref.stale || !found;
    el.clear.classList.toggle('is-prominent', stale);
    var n = (pref.uids && pref.uids.length) ? pref.uids.length + ' chosen masks' : 'all accepted masks';
    el.pref.appendChild(mk('span', { 'class': 'ok', 'aria-hidden': 'true' }, '\u2713'));
    var wrap = mk('span', { 'class': 'wrap' });
    wrap.appendChild(document.createTextNode('Saved: ' + n + ' from '));
    wrap.appendChild(mk('b', null, pname));
    wrap.appendChild(document.createTextNode(', by ' + orUnrecorded(pref.chosen_by) +
      (pref.at ? ', ' + fmtAst(pref.at) : '') + (pref.note ? '. ' + pref.note : '')));
    el.pref.appendChild(wrap);
    if (stale) {
      el.pref.appendChild(mk('div', { 'class': 'mx-cell-warn' },
        'This project no longer exists; training sets fall back to the strongest project. Clear the choice or pick masks from a listed project.'));
    }
  }

  function renderProjects() {
    clear(el.projects);
    var d = S.data;
    var ps = d.projects || [];
    var bySrc = {};
    (d.sources || []).forEach(function (s) { (bySrc[s.project_id] = bySrc[s.project_id] || []).push(s); });
    // Projects that hold the image in segmentations.json, then the ones the
    // ledger names but that carry no segmentation entry for it (still
    // openable by project_id).
    var seen = {};
    ps.forEach(function (p) { seen[p.project_id] = true; });
    var ledgerOnly = [];
    (d.sources || []).forEach(function (s) {
      if (!s.project_id || seen[s.project_id]) return;
      seen[s.project_id] = true;
      var out = null;
      (bySrc[s.project_id] || []).forEach(function (x) { if (x.outcome) out = out || x.outcome; });
      ledgerOnly.push({ project_id: s.project_id, name: s.name || s.project_id,
                        outcome_for_label: out, ledger_only: true });
    });
    var all = ps.concat(ledgerOnly);
    if (!all.length) {
      el.projects.appendChild(mk('div', { 'class': 'mx-cell-none' }, 'No linked project.'));
      return;
    }
    all.forEach(function (p) {
      var row = mk('div', { 'class': 'row' });
      row.appendChild(mk('span', { 'class': 'row-name' }, p.name || p.project_id));
      row.appendChild(mk('span', { 'class': 'lchip ' + (OUTCOME_CLASS[p.outcome_for_label] || '') },
        OUTCOME_WORDS[p.outcome_for_label] || p.outcome_for_label || 'no outcome'));
      var meta = mk('span', { 'class': 'row-meta' });
      if (p.ledger_only) {
        meta.appendChild(mk('span', null,
          'Named by the label ledger only: no segmentation data for this image in this project.'));
      } else {
        meta.appendChild(metaItem('folder ', p.run_dir || '?'));
        meta.appendChild(metaItem('opened by ', orUnrecorded(p.opened_by)));
        meta.appendChild(metaItem('created by ', orUnrecorded(p.initials)));
      }
      var srcs = bySrc[p.project_id] || [];
      var det = mk('details', { 'class': 'src' });
      det.appendChild(mk('summary', null, srcs.length + ' review entries'));
      var ul = mk('ul', null);
      if (srcs.length) {
        srcs.forEach(function (s) {
          ul.appendChild(mk('li', null,
            (OUTCOME_WORDS[s.outcome] || s.outcome || '') + ' via ' + (s.source || 'unknown source') +
            ' by ' + orUnrecorded(s.reviewer) + ' on ' + (s.at ? fmtAst(s.at) : 'an unrecorded time') +
            (s.uid ? ', mask ' + short(s.uid) + (s.status ? ' ' + s.status : '') : '')));
        });
      } else {
        ul.appendChild(mk('li', null, 'no ledger entry for this label'));
      }
      det.appendChild(ul);
      meta.appendChild(det);
      row.appendChild(meta);
      // Copy path needs no tooltip (its label already says what it does,
      // design spec 7.3); it is built with mk(), never control(), which
      // would refuse a titleless button.
      var acts = mk('div', { 'class': 'row-act' });
      var openB = control('button', {
        type: 'button', 'class': 'btn',
        title: 'Open this project\'s run folder in the file manager on the machine running the orchestrator.'
      }, 'Open project folder');
      openB.addEventListener('click', function () { openFolder(p, openB); });
      var copyB = mk('button', { type: 'button', 'class': 'btn' }, 'Copy path');
      copyB.addEventListener('click', function () { copyText(p.project_dir || p.run_dir || p.project_id); });
      // A ledger-only entry carries no project_dir (the matrix only knows
      // its id from the provenance ledger, never resolved to a run folder),
      // so there is no path to route the orchestrator to.
      var pipeB = null;
      if (!p.ledger_only && p.project_dir) {
        // No tooltip: the confirm() in openInPipeline() below already tells
        // the person it switches the pipeline window and closes any open
        // annotator, at the moment that matters (tooltip rules 3).
        pipeB = mk('button', { type: 'button', 'class': 'btn' }, 'Open in pipeline');
        pipeB.addEventListener('click', function () { openInPipeline(p); });
      }
      // Handoffs sit in the fixed order of the registry window design
      // (section 5): the next step in the flow first, then Copy path, then
      // Open folder. The buttons are built above in reading order.
      if (pipeB) acts.appendChild(pipeB);
      acts.appendChild(copyB);
      acts.appendChild(openB);
      row.appendChild(acts);
      el.projects.appendChild(row);
    });
  }

  // ── Actions ────────────────────────────────────────────
  function openFolder(p, btn) {
    hideError();
    btn.disabled = true;
    setStatus('Opening ' + (p.name || p.project_id) + '...');
    postJson('/api/open-folder', { project_id: p.project_id })
      .then(function (j) {
        setStatus('Opened ' + (j.path || p.project_dir || ''));
      })
      .catch(function (err) {
        setStatus('', false);
        showError('Could not open the folder: ' + err.message, err.path || p.project_dir || '');
      })
      .then(function () { btn.disabled = false; });
  }

  // The orchestrator serves this matrix page as a blueprint mounted under
  // BASE (e.g. "/matrix"); its own index route lives at BASE with that
  // suffix stripped, same origin, same Flask app. Auto-open contract:
  // GET /?project_dir=<abs run dir> (see pipeline_orchestrator/app.py).
  function orchestratorOrigin() {
    return BASE.replace(/\/matrix\/?$/, '');
  }
  function openInPipeline(p) {
    hideError();
    var dir = p.project_dir || '';
    if (!dir) { setStatus('No project folder path is known for this project.', true); return; }
    if (!window.confirm('Switches the pipeline window to this project and closes any open annotator.')) return;
    var url = orchestratorOrigin() + '/?project_dir=' + encodeURIComponent(dir);
    // A fixed window name reuses the already-open pipeline tab (matching
    // the orchestrator's own reefpointseg_* popup naming) instead of
    // spawning a new one on every click.
    window.open(url, 'reefpointseg_orchestrator');
  }

  function prefer() {
    var by = getBy();
    if (!by) { el.by.focus(); setStatus('Enter your initials first; the choice records who made it.', true); return; }
    saveBy();
    var p = currentProject();
    if (!p) { setStatus('No project holds this image, so there are no masks to choose.', true); return; }
    var ms = labelMasks(p);
    var chosen = ms.filter(function (m) { return S.checked[m.uid] && m.status === 'accepted'; })
                   .map(function (m) { return m.uid; });
    var accepted = ms.filter(function (m) { return m.status === 'accepted'; }).length;
    if (!chosen.length) { setStatus('Tick at least one accepted mask.', true); return; }
    hideError();
    el.prefer.disabled = true;
    postJson('/api/prefer', {
      image: S.data.image || S.image, label: S.data.label || S.label,
      project_id: p.project_id,
      uids: chosen.length === accepted ? [] : chosen,
      by: by, note: ''
    })
      .then(function (j) {
        S.data.preference = j.preference || null;
        setStatus('Choice saved for ' + (S.data.label || S.label) + ' on ' + (S.data.image || S.image));
        renderPref();
        draw();
      })
      .catch(function (err) { showError('Could not save the choice: ' + err.message, err.path); })
      .then(function () { el.prefer.disabled = false; });
  }

  function clearPref() {
    hideError();
    el.clear.disabled = true;
    postJson('/api/prefer/clear', { image: S.data.image || S.image, label: S.data.label || S.label })
      .then(function () {
        S.data.preference = null;
        setStatus('Choice cleared');
        selectProject(S.projectIdx);
      })
      .catch(function (err) { showError('Could not clear the choice: ' + err.message, err.path); el.clear.disabled = false; });
  }

  // ── Canvas viewer ──────────────────────────────────────
  function imageUrl(p) {
    var u = (p && p.image_url) || '';
    if (!u) return '';
    if (u.indexOf('/img/') === 0) return BASE + u;
    return u;
  }

  function loadImage(p) {
    S.img = null;
    ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
    if (!p) return;
    var url = imageUrl(p);
    if (!url) { setStatus('This project has no image path for ' + (S.data.image || S.image), true); return; }
    var seq = S.reqSeq;
    var im = new Image();
    im.onload = function () {
      if (seq !== S.reqSeq || currentProject() !== p) return;
      S.img = im; S.imgUrl = url;
      draw();
    };
    im.onerror = function () {
      if (seq !== S.reqSeq) return;
      setStatus('The image could not be loaded from ' + url, true);
      draw();
    };
    im.src = url;
  }

  function fit(p) {
    var iw = (S.img && S.img.naturalWidth) || (p && p.image_width) || 1;
    var ih = (S.img && S.img.naturalHeight) || (p && p.image_height) || 1;
    var cw = Math.max(240, (el.wrap.clientWidth || 480));
    var ch = Math.max(120, Math.min(640, Math.round(cw * ih / iw)));
    if (el.canvas.width !== cw) el.canvas.width = cw;
    if (el.canvas.height !== ch) el.canvas.height = ch;
    S.scale = Math.min(cw / iw, ch / ih);
    S.ox = (cw - iw * S.scale) / 2;
    S.oy = (ch - ih * S.scale) / 2;
  }

  function maskColor(m, hovered) {
    if (hovered) return PAINT.hover;
    if (!m.is_label) return PAINT.other;
    if (S.checked[m.uid]) return PAINT.checked;
    return m.status === 'accepted' ? PAINT.label_accepted : PAINT.label_other;
  }

  function tracePolys(polys) {
    ctx.beginPath();
    for (var i = 0; i < polys.length; i++) {
      var poly = polys[i];
      if (!poly || poly.length < 6) continue;
      ctx.moveTo(poly[0], poly[1]);
      for (var j = 2; j < poly.length; j += 2) ctx.lineTo(poly[j], poly[j + 1]);
      ctx.closePath();
    }
  }

  function draw() {
    var p = currentProject();
    fit(p);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, el.canvas.width, el.canvas.height);
    ctx.save();
    ctx.translate(S.ox, S.oy);
    ctx.scale(S.scale, S.scale);
    if (S.img) ctx.drawImage(S.img, 0, 0);
    if (!p) { ctx.restore(); return; }
    var order = [];
    if (S.showOthers) order = order.concat(otherMasks(p));
    order = order.concat(labelMasks(p));
    for (var i = 0; i < order.length; i++) {
      var m = order[i];
      var polys = m.polygon_px || [];
      if (!polys.length) continue;
      var hovered = S.hover === m;
      var faint = !m.is_label;
      tracePolys(polys);
      ctx.globalAlpha = faint ? S.opacity * 0.35 : S.opacity;
      ctx.fillStyle = maskColor(m, false);
      ctx.fill();
      ctx.globalAlpha = faint ? 0.5 : 1.0;
      ctx.strokeStyle = maskColor(m, hovered);
      ctx.lineWidth = (hovered ? 3 : (faint ? 1 : 1.5)) / S.scale;
      ctx.stroke();
    }
    ctx.globalAlpha = 1.0;
    ctx.restore();
  }

  // Point-in-polygon (ray casting) in image coordinates; a point inside any
  // ring of a mask counts as inside that mask.
  function inPoly(poly, x, y) {
    var inside = false;
    for (var i = 0, j = poly.length - 2; i < poly.length; j = i, i += 2) {
      var xi = poly[i], yi = poly[i + 1], xj = poly[j], yj = poly[j + 1];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-9) + xi) inside = !inside;
    }
    return inside;
  }
  function hitTest(cx, cy) {
    var p = currentProject();
    if (!p) return null;
    var x = (cx - S.ox) / S.scale, y = (cy - S.oy) / S.scale;
    var cands = labelMasks(p).concat(S.showOthers ? otherMasks(p) : []);
    var best = null;
    for (var i = 0; i < cands.length; i++) {
      var m = cands[i];
      var polys = m.polygon_px || [];
      for (var k = 0; k < polys.length; k++) {
        if (polys[k] && polys[k].length >= 6 && inPoly(polys[k], x, y)) {
          // Smallest area wins so a nested mask is reachable.
          if (!best || (Number(m.area) || Infinity) < (Number(best.area) || Infinity)) best = m;
          break;
        }
      }
    }
    return best;
  }

  function showTip(m, cx, cy) {
    clear(el.tip);
    el.tip.appendChild(mk('b', null, short(m.uid)));
    el.tip.appendChild(mk('div', null, (m.species || 'unlabeled') + ' ' + (m.status || 'unlabeled')));
    el.tip.appendChild(mk('div', null, 'via ' + (m.source_type || 'unknown source')));
    el.tip.appendChild(mk('div', null, 'by ' + orUnrecorded(m.reviewer)));
    el.tip.appendChild(mk('div', null, fmtArea(m.area)));
    el.tip.hidden = false;
    var pad = 10;
    var left = cx + pad, top = cy + pad;
    if (left + el.tip.offsetWidth > el.wrap.clientWidth - 4) left = cx - el.tip.offsetWidth - pad;
    if (top + el.tip.offsetHeight > el.wrap.clientHeight - 4) top = cy - el.tip.offsetHeight - pad;
    el.tip.style.left = Math.max(0, left) + 'px';
    el.tip.style.top = Math.max(0, top) + 'px';
  }
  function hideTip() { el.tip.hidden = true; clear(el.tip); }

  el.canvas.addEventListener('mousemove', function (e) {
    if (!S.data) return;
    var r = el.canvas.getBoundingClientRect();
    var cx = (e.clientX - r.left) * (el.canvas.width / r.width);
    var cy = (e.clientY - r.top) * (el.canvas.height / r.height);
    var m = hitTest(cx, cy);
    if (m !== S.hover) { S.hover = m; draw(); }
    if (m) showTip(m, e.clientX - r.left, e.clientY - r.top); else hideTip();
  });
  el.canvas.addEventListener('mouseleave', function () {
    if (S.hover) { S.hover = null; draw(); }
    hideTip();
  });

  // ── Wiring ─────────────────────────────────────────────
  el.close.addEventListener('click', close);
  el.by.addEventListener('input', function () { upperAsTyped(el.by); });
  el.by.addEventListener('change', saveBy);
  el.opacity.addEventListener('input', function () {
    S.opacity = Math.max(0, Math.min(1, Number(el.opacity.value) / 100));
    draw();
  });
  el.others.addEventListener('change', function () {
    S.showOthers = !!el.others.checked;
    S.hover = null;
    draw();
  });
  el.prefer.addEventListener('click', prefer);
  el.clear.addEventListener('click', clearPref);
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && isOpen()) { e.preventDefault(); close(); }
  });
  var resizeTimer = null;
  window.addEventListener('resize', function () {
    if (!isOpen() || !S.data) return;
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(draw, 120);
  });
  loadBy();
  S.opacity = Math.max(0, Math.min(1, Number(el.opacity.value) / 100));

  window.MatrixCell = { open: open, close: close, isOpen: isOpen };
})();

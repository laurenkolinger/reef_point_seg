/* Label Manager - a curator's bench over the species-code vocabulary.
   Three sections: Vocabulary (add new, edit name/category inline, codes locked),
   Remaps (show the active log, compose a new timestamped one), and Locations +
   Recode (read-only paths, one guarded re-run). Thin transport over the
   /labels/api/* routes; the safety posture lives in labels_io on the server.
   No emoji, no em dashes. */
(function () {
  'use strict';

  var BASE = (window.LABELS_BASE || '').replace(/\/$/, '');

  // ── small helpers ──────────────────────────────────────
  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function show(node, on) {
    if (node) node.hidden = !on;
  }
  function clear(node) { while (node && node.firstChild) node.removeChild(node.firstChild); }

  function api(path, opts) {
    return fetch(BASE + path, opts).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        return { status: r.status, ok: r.ok, data: data };
      });
    });
  }
  function postJSON(path, body) {
    return api(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {})
    });
  }

  function baseName(p) {
    if (!p) return '';
    var parts = String(p).split('/');
    return parts[parts.length - 1] || p;
  }

  // ── toasts ─────────────────────────────────────────────
  function toast(kind, title, body) {
    var wrap = $('lb-toasts');
    var t = el('div', 'lb-toast is-' + kind);
    t.appendChild(el('div', 't-title', title));
    if (body) t.appendChild(el('div', 't-body', body));
    wrap.appendChild(t);
    setTimeout(function () {
      if (t.parentNode) t.parentNode.removeChild(t);
    }, kind === 'error' ? 9000 : 6500);
  }
  function showError(node, msg) {
    if (!node) return;
    node.textContent = msg || '';
    show(node, !!msg);
  }

  // ── state ──────────────────────────────────────────────
  var VOCAB = { rows: [], categories: [] };
  var COMPOSE = { remaps: [], excludes: [] };  // [{old,new}] / ["CODE"]
  var LOCATIONS = null;
  var nextComposeId = 1;

  // ====================================================================
  // Section 1: Vocabulary
  // ====================================================================
  function loadVocabulary() {
    api('/api/vocabulary').then(function (res) {
      if (!res.ok) {
        toast('error', 'Could not load vocabulary', res.data.error || ('HTTP ' + res.status));
        return;
      }
      VOCAB.rows = res.data.rows || [];
      VOCAB.categories = res.data.categories || [];
      renderPaths(res.data);
      renderCategoryFilter();
      renderCategoryDatalist();
      renderVocabTable();
    });
  }

  function renderPaths(data) {
    var box = $('vocab-paths');
    clear(box);
    function row(label, val) {
      var r = el('div', 'lb-path-row');
      r.appendChild(el('span', null, label + ' '));
      r.appendChild(el('b', null, val || '(not set)'));
      box.appendChild(r);
    }
    row('Canonical', data.master_codes_csv);
    if (data.duplicate_master_codes) row('Mirror', data.duplicate_master_codes);
  }

  function renderCategoryFilter() {
    var sel = $('vocab-cat');
    var current = sel.value;
    clear(sel);
    sel.appendChild(new Option('All categories', ''));
    VOCAB.categories.forEach(function (c) { sel.appendChild(new Option(c, c)); });
    sel.value = current || '';
  }

  function renderCategoryDatalist() {
    var dl = $('cat-list');
    clear(dl);
    VOCAB.categories.forEach(function (c) {
      var o = document.createElement('option');
      o.value = c;
      dl.appendChild(o);
    });
  }

  function filteredVocab() {
    var q = ($('vocab-search').value || '').trim().toLowerCase();
    var cat = $('vocab-cat').value || '';
    return VOCAB.rows.filter(function (r) {
      if (cat && (r.category || '') !== cat) return false;
      if (q) {
        var hay = ((r.code || '') + ' ' + (r.name || '') + ' ' + (r.category || '')).toLowerCase();
        if (hay.indexOf(q) === -1) return false;
      }
      return true;
    });
  }

  function renderVocabTable() {
    var rows = filteredVocab();
    var tbody = $('vocab-tbody');
    clear(tbody);
    show($('vocab-table'), rows.length > 0);
    show($('vocab-empty'), rows.length === 0);

    rows.forEach(function (r) {
      tbody.appendChild(makeVocabRow(r));
    });

    var total = VOCAB.rows.length;
    var shown = rows.length;
    $('vocab-count').textContent =
      shown === total
        ? (total + ' label' + (total === 1 ? '' : 's'))
        : (shown + ' of ' + total + ' labels shown');
  }

  function makeVocabRow(r) {
    var tr = el('tr');
    tr.dataset.code = r.code;

    // Code (locked)
    var tdCode = el('td');
    var codeWrap = el('span', 'lb-code');
    codeWrap.appendChild(el('span', null, r.code));
    var lock = el('span', 'lb-lock', 'locked');
    lock.title = 'Codes are referenced across every project and dataset, so a '
      + 'code cannot be changed or deleted. Only its name and category are editable.';
    codeWrap.appendChild(lock);
    tdCode.appendChild(codeWrap);
    tr.appendChild(tdCode);

    // Category
    var tdCat = el('td');
    var tag = el('span', 'lb-cat-tag', r.category || '');
    tdCat.appendChild(tag);
    tr.appendChild(tdCat);

    // Name
    tr.appendChild(el('td', null, r.name || ''));

    // Edit
    var tdEdit = el('td', 'col-edit');
    var editBtn = el('button', 'lb-btn lb-btn-ghost', 'Edit');
    editBtn.type = 'button';
    editBtn.title = 'Edit the name and category of ' + r.code;
    editBtn.addEventListener('click', function () { startEdit(tr, r); });
    tdEdit.appendChild(editBtn);
    tr.appendChild(tdEdit);

    return tr;
  }

  function startEdit(tr, r) {
    tr.classList.add('lb-row-editing');
    clear(tr);
    tr.dataset.code = r.code;

    // locked code (unchanged)
    var tdCode = el('td');
    var codeWrap = el('span', 'lb-code');
    codeWrap.appendChild(el('span', null, r.code));
    var lock = el('span', 'lb-lock', 'locked');
    lock.title = 'This code is immutable. Editing changes its name and category only.';
    codeWrap.appendChild(lock);
    tdCode.appendChild(codeWrap);
    tr.appendChild(tdCode);

    // category input
    var tdCat = el('td');
    var catIn = el('input');
    catIn.type = 'text';
    catIn.value = r.category || '';
    catIn.setAttribute('list', 'cat-list');
    catIn.title = 'Category (required, non-empty)';
    tdCat.appendChild(catIn);
    tr.appendChild(tdCat);

    // name input
    var tdName = el('td');
    var nameIn = el('input');
    nameIn.type = 'text';
    nameIn.value = r.name || '';
    nameIn.title = 'Name (required, non-empty)';
    tdName.appendChild(nameIn);
    tr.appendChild(tdName);

    // actions
    var tdAct = el('td', 'col-edit');
    var actWrap = el('div', 'lb-row-actions');
    var saveBtn = el('button', 'lb-btn lb-btn-accent', 'Save');
    saveBtn.type = 'button';
    saveBtn.title = 'Back up the canonical file, write the change, and mirror the duplicate';
    var cancelBtn = el('button', 'lb-btn', 'Cancel');
    cancelBtn.type = 'button';
    cancelBtn.title = 'Discard this edit';
    actWrap.appendChild(saveBtn);
    actWrap.appendChild(cancelBtn);
    tdAct.appendChild(actWrap);
    tr.appendChild(tdAct);

    // confirm note + error row beneath
    var noteTr = el('tr', 'lb-row-confirm');
    var noteTd = el('td');
    noteTd.colSpan = 4;
    var note = el('p', 'note',
      'Saving backs up the canonical file and mirrors the duplicate so the two '
      + 'stay byte-identical.');
    noteTd.appendChild(note);
    var errP = el('p', 'lb-inline-error');
    errP.setAttribute('role', 'alert');
    errP.hidden = true;
    noteTd.appendChild(errP);
    noteTr.appendChild(noteTd);
    tr.parentNode.insertBefore(noteTr, tr.nextSibling);

    cancelBtn.addEventListener('click', function () { loadVocabulary(); });
    saveBtn.addEventListener('click', function () {
      showError(errP, '');
      saveBtn.disabled = true;
      postJSON('/api/label', {
        code: r.code,
        category: catIn.value.trim(),
        name: nameIn.value.trim(),
        is_new: false
      }).then(function (res) {
        saveBtn.disabled = false;
        if (!res.ok) { showError(errP, res.data.error || ('HTTP ' + res.status)); return; }
        toast('ok', 'Saved ' + r.code,
          'Backup written: ' + baseName(res.data.backup_path)
          + (res.data.mirrored ? '. Duplicate mirrored.' : '.'));
        loadVocabulary();
      });
    });
    nameIn.focus();
  }

  // Add-label form
  function openAddForm() {
    showError($('add-error'), '');
    $('add-code').value = '';
    $('add-category').value = '';
    $('add-name').value = '';
    show($('vocab-addform'), true);
    $('add-code').focus();
  }
  function closeAddForm() {
    show($('vocab-addform'), false);
    showError($('add-error'), '');
  }
  function submitAddForm(ev) {
    ev.preventDefault();
    showError($('add-error'), '');
    var btn = $('add-save');
    btn.disabled = true;
    postJSON('/api/label', {
      code: $('add-code').value.trim(),
      category: $('add-category').value.trim(),
      name: $('add-name').value.trim(),
      is_new: true
    }).then(function (res) {
      btn.disabled = false;
      if (!res.ok) { showError($('add-error'), res.data.error || ('HTTP ' + res.status)); return; }
      toast('ok', 'Added ' + (res.data.row ? res.data.row.code : ''),
        'Backup written: ' + baseName(res.data.backup_path)
        + (res.data.mirrored ? '. Duplicate mirrored.' : '.'));
      closeAddForm();
      loadVocabulary();
    });
  }

  // ====================================================================
  // Section 2: Remaps
  // ====================================================================
  function loadRemaps() {
    api('/api/remaps').then(function (res) {
      if (!res.ok) {
        toast('error', 'Could not load remaps', res.data.error || ('HTTP ' + res.status));
        return;
      }
      renderActiveRemap(res.data.active);
      renderRemapHistory(res.data.logs || [], res.data.active);
    });
  }

  function remapPair(entry) {
    // tolerate {old,new} or {from,to} shapes
    var oldC = entry.old != null ? entry.old : (entry.from != null ? entry.from : '');
    var newC = entry.new != null ? entry.new : (entry.to != null ? entry.to : '');
    return { old: oldC, new: newC };
  }

  function renderActiveRemap(active) {
    var meta = $('remap-active-meta');
    clear(meta);
    var table = $('remap-active-table');
    var tbody = $('remap-active-tbody');
    var emptyP = $('remap-active-empty');
    var exWrap = $('remap-active-excludes');
    clear(tbody);

    if (!active) {
      show(table, false);
      show(exWrap, false);
      show(emptyP, true);
      return;
    }
    show(emptyP, false);

    var stamp = active.timestamp || '';
    var name = active.name || '';
    var note = active.source_note || '';
    var m = el('div');
    m.appendChild(el('span', null, 'File: '));
    m.appendChild(el('b', null, name));
    if (stamp) { m.appendChild(el('span', null, '  Stamp: ')); m.appendChild(el('b', null, stamp)); }
    meta.appendChild(m);
    if (note) {
      var nm = el('div');
      nm.appendChild(el('span', null, 'Note: '));
      nm.appendChild(el('b', null, note));
      meta.appendChild(nm);
    }

    var remaps = (active.remaps || []).map(remapPair);
    show(table, remaps.length > 0);
    remaps.forEach(function (p) {
      var tr = el('tr');
      tr.appendChild(el('td', null, p.old));
      tr.appendChild(el('td', 'lb-arrow', 'to'));
      tr.appendChild(el('td', null, p.new));
      tbody.appendChild(tr);
    });

    var excludes = active.excludes || [];
    show(exWrap, excludes.length > 0);
    if (excludes.length) {
      var chips = $('remap-active-exchips');
      clear(chips);
      excludes.forEach(function (c) {
        var chip = el('span', 'lb-chip', c);
        chips.appendChild(chip);
      });
    }
  }

  function renderRemapHistory(logs, active) {
    var list = $('remap-history');
    clear(list);
    if (!logs.length) {
      list.appendChild(el('li', null, 'No remap logs yet.'));
      return;
    }
    var activeName = active ? (active.name || '') : '';
    logs.forEach(function (lg) {
      var li = el('li');
      if (lg.name === activeName) li.className = 'is-active';
      var nameSpan = el('span', 'h-name', lg.name);
      var meta = el('span', null,
        (lg.n_remaps || 0) + ' remap' + ((lg.n_remaps === 1) ? '' : 's')
        + (lg.name === activeName ? ' (active)' : ''));
      li.appendChild(nameSpan);
      li.appendChild(meta);
      list.appendChild(li);
    });
  }

  // compose builder
  function renderComposeRows() {
    var wrap = $('compose-rows');
    clear(wrap);
    if (!COMPOSE.remaps.length && !COMPOSE.excludes.length) {
      wrap.appendChild(el('p', 'lb-card-note',
        'No rows yet. Add a remap (old to new) or an exclude.'));
      return;
    }
    COMPOSE.remaps.forEach(function (r) { wrap.appendChild(makeRemapRow(r)); });
    COMPOSE.excludes.forEach(function (x) { wrap.appendChild(makeExcludeRow(x)); });
  }

  function makeRemapRow(r) {
    var row = el('div', 'lb-compose-row is-remap');
    var oldIn = el('input');
    oldIn.type = 'text';
    oldIn.placeholder = 'OLD';
    oldIn.value = r.old || '';
    oldIn.title = 'The existing code to collapse';
    oldIn.addEventListener('input', function () { r.old = oldIn.value.trim().toUpperCase(); oldIn.value = r.old; });
    var arrow = el('span', 'row-arrow', 'to');
    var newIn = el('input');
    newIn.type = 'text';
    newIn.placeholder = 'NEW';
    newIn.value = r.new || '';
    newIn.title = 'The code it becomes';
    newIn.addEventListener('input', function () { r.new = newIn.value.trim().toUpperCase(); newIn.value = r.new; });
    var del = el('button', 'lb-btn lb-btn-ghost', 'remove');
    del.type = 'button';
    del.title = 'Remove this remap row';
    del.addEventListener('click', function () {
      COMPOSE.remaps = COMPOSE.remaps.filter(function (x) { return x !== r; });
      renderComposeRows();
    });
    row.appendChild(oldIn);
    row.appendChild(arrow);
    row.appendChild(newIn);
    row.appendChild(del);
    return row;
  }

  function makeExcludeRow(x) {
    var row = el('div', 'lb-compose-row is-exclude');
    row.appendChild(el('span', 'row-tag', 'exclude'));
    var input = el('input');
    input.type = 'text';
    input.placeholder = 'CODE';
    input.value = x.code || '';
    input.title = 'A code to drop from the recode entirely';
    input.addEventListener('input', function () { x.code = input.value.trim().toUpperCase(); input.value = x.code; });
    var del = el('button', 'lb-btn lb-btn-ghost', 'remove');
    del.type = 'button';
    del.title = 'Remove this exclude row';
    del.addEventListener('click', function () {
      COMPOSE.excludes = COMPOSE.excludes.filter(function (e) { return e !== x; });
      renderComposeRows();
    });
    row.appendChild(input);
    row.appendChild(del);
    return row;
  }

  function addComposeRemap() {
    COMPOSE.remaps.push({ id: nextComposeId++, old: '', new: '' });
    renderComposeRows();
  }
  function addComposeExclude() {
    COMPOSE.excludes.push({ id: nextComposeId++, code: '' });
    renderComposeRows();
  }

  function saveComposed() {
    showError($('compose-error'), '');
    var remaps = COMPOSE.remaps
      .map(function (r) { return { old: (r.old || '').trim(), new: (r.new || '').trim() }; })
      .filter(function (r) { return r.old || r.new; });
    var excludes = COMPOSE.excludes
      .map(function (x) { return (x.code || '').trim(); })
      .filter(function (c) { return c; });

    // local validation before the round trip
    for (var i = 0; i < remaps.length; i++) {
      if (!remaps[i].old || !remaps[i].new) {
        showError($('compose-error'), 'Each remap row needs both an old code and a new code.');
        return;
      }
    }
    if (!remaps.length && !excludes.length) {
      showError($('compose-error'), 'Add at least one remap or exclude before saving.');
      return;
    }

    var btn = $('compose-save');
    btn.disabled = true;
    postJSON('/api/remap', {
      remaps: remaps,
      excludes: excludes,
      source_note: ($('compose-note').value || '').trim()
    }).then(function (res) {
      btn.disabled = false;
      if (!res.ok) { showError($('compose-error'), res.data.error || ('HTTP ' + res.status)); return; }
      toast('ok', 'New remap log written', baseName(res.data.path));
      COMPOSE.remaps = [];
      COMPOSE.excludes = [];
      $('compose-note').value = '';
      renderComposeRows();
      loadRemaps();
      loadLocations();
    });
  }

  // ====================================================================
  // Section 3: Locations + recode
  // ====================================================================
  function loadLocations() {
    api('/api/locations').then(function (res) {
      if (!res.ok) {
        toast('error', 'Could not load locations', res.data.error || ('HTTP ' + res.status));
        return;
      }
      LOCATIONS = res.data;
      renderLocations(res.data);
    });
  }

  function renderLocations(data) {
    var dl = $('loc-list');
    clear(dl);
    var loc = data.locations || {};
    var exists = data.exists || {};
    var rows = [
      ['Master codes', loc.master_codes_csv, 'master_codes_csv'],
      ['Duplicate (mirror)', loc.duplicate_master_codes, 'duplicate_master_codes'],
      ['All points', loc.all_points_csv, 'all_points_csv'],
      ['Recode output dir', loc.recode_output_dir, 'recode_output_dir']
    ];
    rows.forEach(function (r) {
      dl.appendChild(el('dt', null, r[0]));
      var dd = el('dd');
      dd.appendChild(document.createTextNode(r[1] || '(not set)'));
      if (r[1] && Object.prototype.hasOwnProperty.call(exists, r[2])) {
        var flag = el('span', 'lb-flag ' + (exists[r[2]] ? 'lb-flag-ok' : 'lb-flag-missing'),
          exists[r[2]] ? 'present' : 'missing');
        dd.appendChild(flag);
      }
      dl.appendChild(dd);
    });

    // latest remap log
    dl.appendChild(el('dt', null, 'Latest remap log'));
    var ddLog = el('dd');
    var latest = data.latest_remap_log;
    ddLog.textContent = latest ? latest.name : '(none yet)';
    dl.appendChild(ddLog);

    // tune the recode note + button availability
    var note = $('recode-note');
    if (!latest) {
      note.textContent = 'No remap log to recode from yet. Save one in the Remaps '
        + 'section first, then re-run.';
      $('recode-run').disabled = true;
    } else {
      note.textContent = 'Regenerates the recoded master codes from the canonical '
        + 'inputs and ' + latest.name + '. The prior recode outputs are backed up first.';
      $('recode-run').disabled = false;
    }
  }

  // confirm modal
  var modalConfirmFn = null;
  function openModal(body, onConfirm) {
    $('modal-body').textContent = body;
    modalConfirmFn = onConfirm;
    show($('lb-modal'), true);
    $('modal-confirm').focus();
  }
  function closeModal() {
    show($('lb-modal'), false);
    modalConfirmFn = null;
  }

  function askRecode() {
    var latest = LOCATIONS && LOCATIONS.latest_remap_log;
    var name = latest ? latest.name : 'the latest remap log';
    openModal(
      'This regenerates the recoded master codes that new projects seed from, '
      + 'using ' + name + '. The prior recode outputs are backed up first. '
      + 'It does not touch the canonical vocabulary. Continue?',
      runRecode
    );
  }

  function runRecode() {
    closeModal();
    showError($('recode-error'), '');
    show($('recode-result'), false);
    var btn = $('recode-run');
    btn.disabled = true;
    var prevLabel = btn.textContent;
    btn.textContent = 'Recoding...';
    postJSON('/api/recode', { confirm: true }).then(function (res) {
      btn.disabled = false;
      btn.textContent = prevLabel;
      if (!res.ok) {
        var msg = res.data.error || ('HTTP ' + res.status);
        showError($('recode-error'), msg);
        toast('error', 'Recode failed', msg);
        return;
      }
      toast('ok', 'Recode complete',
        'From ' + baseName(res.data.remap_log_path)
        + (res.data.backup_dir ? '. Prior outputs backed up.' : '.'));
      var out = $('recode-result');
      out.textContent = JSON.stringify({
        remap_log: baseName(res.data.remap_log_path),
        outputs: res.data.outputs || {},
        backup_dir: res.data.backup_dir || null,
        summary: res.data.summary || {}
      }, null, 2);
      show(out, true);
      loadLocations();
    });
  }

  // ====================================================================
  // wire up
  // ====================================================================
  function init() {
    $('lb-generated').textContent = 'Loaded ' + new Date().toLocaleString();

    // vocabulary
    $('vocab-search').addEventListener('input', renderVocabTable);
    $('vocab-cat').addEventListener('change', renderVocabTable);
    $('vocab-add').addEventListener('click', openAddForm);
    $('add-cancel').addEventListener('click', closeAddForm);
    $('vocab-addform').addEventListener('submit', submitAddForm);

    // remaps
    $('compose-add-remap').addEventListener('click', addComposeRemap);
    $('compose-add-exclude').addEventListener('click', addComposeExclude);
    $('compose-save').addEventListener('click', saveComposed);

    // recode
    $('recode-run').addEventListener('click', askRecode);
    $('modal-cancel').addEventListener('click', closeModal);
    $('modal-confirm').addEventListener('click', function () {
      if (modalConfirmFn) modalConfirmFn();
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && !$('lb-modal').hidden) closeModal();
    });

    renderComposeRows();
    loadVocabulary();
    loadRemaps();
    loadLocations();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

/* Label Manager - a curator's bench over the species-code vocabulary.
   Three sections: Vocabulary (add new, edit name/category inline, codes locked),
   Remaps (show the active log, compose a new timestamped one), and Locations +
   Recode (read-only paths, one guarded re-run). Thin transport over the
   /labels/api/* routes; the safety posture lives in labels_io on the server.

   The vocabulary table sorts on its column heads, filters as you type, and
   remembers the search, the category and the sort per browser under the
   vega.labels. prefix. It has two empty lines, chosen on the vocabulary's
   total rather than the filtered rows: no labels at all, or no label that
   matches the filters (a remembered search can open the page onto the
   second). Names are the primary cell and codes ride beside them as chips
   (design spec section 6). No emoji, no em dashes. */
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

  // House format for every timestamp: 24 hour, no locale surprises. The box
  // runs on AST, so the local clock is already the reported zone.
  function stamp24(d) {
    var t = d || new Date();
    var p = function (n) { return (n < 10 ? '0' : '') + n; };
    return t.getFullYear() + '-' + p(t.getMonth() + 1) + '-' + p(t.getDate()) +
           ' ' + p(t.getHours()) + ':' + p(t.getMinutes());
  }

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

  // ── remembered choices ─────────────────────────────────
  // Every key sits under one prefix so a browser's storage can be read or
  // cleared per window (registry window design, section 1). Storage can be
  // absent or refused (a private window, a blocked site), so both helpers
  // swallow the failure and the page runs without a memory.
  var PREF_PREFIX = 'vega.labels.';

  // The remembered value for a key, or null when there is none or storage
  // is unavailable.
  function readPref(key) {
    try {
      return window.localStorage.getItem(PREF_PREFIX + key);
    } catch (e) {
      return null;
    }
  }

  // Remember a value for a key; an empty value forgets the key.
  function writePref(key, value) {
    try {
      if (value == null || value === '') window.localStorage.removeItem(PREF_PREFIX + key);
      else window.localStorage.setItem(PREF_PREFIX + key, String(value));
    } catch (e) {
      // The choice is simply not remembered this time.
    }
  }

  // ── names and codes ────────────────────────────────────
  // Categories whose names are never a taxon: a coral condition, a substrate,
  // a survey artefact. Any other category (Coral, Sponge, Gorgonian,
  // Macroalgae, Cyanobacteria, Zoanthid, Calcareous, or one typed later) may
  // hold a binomial, and then the name's shape decides.
  var NON_TAXON_CATEGORIES = ['coral condition', 'non-living', 'other',
                              'other, living', 'turf', 'dca'];

  // Common-name heads this vocabulary uses, in both positions. "Ball sponge",
  // "Sea fan", "Bleached coral" and "Coral spp." are names of a kind of thing,
  // not binomials, so they stay upright; "Agaricia agaricites" and "Agaricia
  // spp." are taxa and read in italics (design spec section 6).
  var COMMON_SECOND = ['sponge', 'coral', 'algae', 'fan', 'plume', 'rod',
                       'whip', 'tape', 'line', 'plague', 'cyanobacteria',
                       'disease', 'band', 'complex', 'community', 'shadow'];
  var COMMON_FIRST = ['coral', 'juvenile', 'branching', 'sea', 'other',
                      'encrusting', 'erect', 'dead', 'diseased', 'bleached',
                      'crustose', 'filamentous', 'epilithic', 'stony',
                      'transect', 'weighted', 'white', 'black', 'yellow',
                      'dark', 'ball', 'barrel', 'boring', 'rope', 'tube'];

  // True when a name is a species binomial or a genus with "spp." after it,
  // in a category that can hold a taxon. Returns false for a one-word name,
  // a name of three words or more, any name whose first or second word is
  // one of the common-name heads above, and every name in a non-taxon
  // category whatever its shape.
  function isScientific(name, category) {
    var cat = String(category || '').trim().toLowerCase();
    if (NON_TAXON_CATEGORIES.indexOf(cat) !== -1) return false;
    var parts = String(name || '').trim().split(/\s+/);
    if (parts.length !== 2) return false;
    var first = parts[0];
    var second = parts[1];
    if (!/^[A-Z][a-z]{2,}$/.test(first)) return false;
    if (COMMON_FIRST.indexOf(first.toLowerCase()) !== -1) return false;
    if (/^(spp\.|sp\.)$/.test(second)) return true;
    if (!/^[a-z]{3,}$/.test(second)) return false;
    return COMMON_SECOND.indexOf(second) === -1;
  }

  // The name cell: a binomial in italics with "spp." left upright beside it.
  function nameNode(name, category) {
    var wrap = el('span', 'lb-name');
    if (!isScientific(name, category)) {
      wrap.textContent = name || '';
      return wrap;
    }
    var parts = String(name).trim().split(/\s+/);
    if (/^(spp\.|sp\.)$/.test(parts[1])) {
      wrap.appendChild(el('i', 'sci', parts[0]));
      wrap.appendChild(document.createTextNode(' ' + parts[1]));
    } else {
      wrap.appendChild(el('i', 'sci', name));
    }
    return wrap;
  }

  // A code as a chip: Space Mono on a hairline ring, beside the name it
  // stands for. The lock on codes is explained once, on the Code column head.
  function codeChip(code) {
    return el('span', 'lb-code', code || '');
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

  // The sort of the vocabulary table: a column and a direction, or nothing
  // for the file order (the order of the canonical CSV).
  var SORT_COLUMNS = ['name', 'code', 'category'];
  var SORT = { col: '', dir: '' };

  // The category remembered from the last visit, applied once the category
  // list has loaded (the select has no options to pick from before then).
  var pendingCategory = readPref('category') || '';

  // ====================================================================
  // Section 1: Vocabulary
  // ====================================================================
  function loadVocabulary(then) {
    api('/api/vocabulary').then(function (res) {
      if (!res.ok) {
        toast('error', 'Could not load vocabulary', res.data.error || ('HTTP ' + res.status));
        if (then) then();
        return;
      }
      VOCAB.rows = res.data.rows || [];
      VOCAB.categories = res.data.categories || [];
      renderCategoryFilter();
      renderCategoryDatalist();
      renderVocabTable();
      if (then) then();
    });
  }

  // The vocabulary row for a code, or null when the vocabulary has none.
  function labelRow(code) {
    if (!code) return null;
    for (var i = 0; i < VOCAB.rows.length; i++) {
      if (VOCAB.rows[i].code === code) return VOCAB.rows[i];
    }
    return null;
  }

  function renderCategoryFilter() {
    var sel = $('vocab-cat');
    var wanted = sel.value || pendingCategory;
    pendingCategory = '';
    clear(sel);
    sel.appendChild(new Option('All categories', ''));
    VOCAB.categories.forEach(function (c) { sel.appendChild(new Option(c, c)); });
    sel.value = wanted;
    // A remembered category that no longer exists is forgotten rather than
    // left to filter every row away.
    if (sel.value !== wanted) {
      sel.value = '';
      writePref('category', '');
    }
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

  function filtersActive() {
    return !!(($('vocab-search').value || '').trim()) || !!$('vocab-cat').value;
  }

  function onSearchInput() {
    writePref('search', ($('vocab-search').value || '').trim());
    renderVocabTable();
  }

  function onCategoryChange() {
    writePref('category', $('vocab-cat').value || '');
    renderVocabTable();
  }

  // Clear filters clears the search and the category and forgets both. The
  // sort stays: it is an order, not a filter.
  function clearVocabFilters() {
    $('vocab-search').value = '';
    $('vocab-cat').value = '';
    writePref('search', '');
    writePref('category', '');
    renderVocabTable();
    $('vocab-search').focus();
  }

  // Text comparison that ignores case and accents, so "agaricia" and
  // "Agaricia" sort together.
  function compareText(a, b) {
    return String(a || '').localeCompare(String(b || ''), undefined, { sensitivity: 'base' });
  }

  // A sorted copy of the rows under the current sort; the file order when
  // no column is chosen. Ties on name or category fall back to the code.
  function sortedRows(rows) {
    if (!SORT.col) return rows.slice();
    var col = SORT.col;
    var sign = SORT.dir === 'desc' ? -1 : 1;
    return rows.slice().sort(function (a, b) {
      var c = compareText(a[col], b[col]);
      if (c === 0 && col !== 'code') c = compareText(a.code, b.code);
      return c * sign;
    });
  }

  // Restore the remembered sort ("name:asc") when it names a real column.
  function restoreSort() {
    var parts = (readPref('sort') || '').split(':');
    if (SORT_COLUMNS.indexOf(parts[0]) !== -1 && (parts[1] === 'asc' || parts[1] === 'desc')) {
      SORT = { col: parts[0], dir: parts[1] };
    }
  }

  // One press sorts a column ascending, a second reverses it, a third
  // returns to the file order.
  function cycleSort(col) {
    if (SORT.col !== col) SORT = { col: col, dir: 'asc' };
    else if (SORT.dir === 'asc') SORT = { col: col, dir: 'desc' };
    else SORT = { col: '', dir: '' };
    writePref('sort', SORT.col ? SORT.col + ':' + SORT.dir : '');
    renderVocabTable();
  }

  // Mark the active head with aria-sort; the stylesheet draws the arrow.
  function paintSortHeads() {
    var heads = document.querySelectorAll('#vocab-table th[data-sort]');
    Array.prototype.forEach.call(heads, function (th) {
      if (th.dataset.sort === SORT.col) {
        th.setAttribute('aria-sort', SORT.dir === 'asc' ? 'ascending' : 'descending');
      } else {
        th.removeAttribute('aria-sort');
      }
    });
  }

  // Optional control: hidden when absent rather than throwing.
  function setHidden(id, on) {
    var node = $(id);
    if (node) node.hidden = on;
  }

  function renderVocabTable() {
    var rows = sortedRows(filteredVocab());
    setHidden('vocab-clear', !filtersActive());
    paintSortHeads();
    var tbody = $('vocab-tbody');
    clear(tbody);
    var total = VOCAB.rows.length;
    var shown = rows.length;
    show($('vocab-table'), shown > 0);
    // Two empty lines, never both. The mint line is for a vocabulary with
    // no labels; the filter line says what hid the rows and how to get them
    // back, and carries the total so the count line and it agree.
    show($('vocab-empty'), total === 0);
    show($('vocab-nomatch'), total > 0 && shown === 0);
    // Guarded: an orchestrator that compiled an older template at startup
    // serves this script fresh, and the render must not throw on it.
    var totalNode = $('vocab-nomatch-total');
    if (totalNode) totalNode.textContent = String(total);

    rows.forEach(function (r) {
      tbody.appendChild(makeVocabRow(r));
    });

    $('vocab-count').textContent =
      shown === total
        ? (total + ' label' + (total === 1 ? '' : 's'))
        : (shown + ' of ' + total + ' labels shown');
  }

  function makeVocabRow(r) {
    var tr = el('tr');
    tr.dataset.code = r.code;

    // Name, the primary cell.
    var tdName = el('td', 'col-name');
    tdName.appendChild(nameNode(r.name, r.category));
    tr.appendChild(tdName);

    // Code, as a chip. The lock is stated once, on the Code column head.
    var tdCode = el('td', 'col-code');
    tdCode.appendChild(codeChip(r.code));
    tr.appendChild(tdCode);

    // Category
    var tdCat = el('td', 'col-category');
    tdCat.appendChild(el('span', 'lb-cat-tag', r.category || ''));
    tr.appendChild(tdCat);

    // Edit
    var tdEdit = el('td', 'col-edit');
    var editBtn = el('button', 'btn btn-mini', 'Edit');
    editBtn.type = 'button';
    // The full name with its code, so the tooltip never shows a bare code.
    editBtn.title = 'Edit the name and category of '
      + (r.name ? r.name + ' (' + r.code + ')' : r.code) + '.';
    editBtn.addEventListener('click', function () { startEdit(tr, r); });
    tdEdit.appendChild(editBtn);
    tr.appendChild(tdEdit);

    return tr;
  }

  function startEdit(tr, r) {
    tr.classList.add('lb-row-editing');
    clear(tr);
    tr.dataset.code = r.code;

    // name input, first because the name is the primary cell
    var tdName = el('td', 'col-name');
    var nameIn = el('input');
    nameIn.type = 'text';
    nameIn.value = r.name || '';
    nameIn.setAttribute('aria-label', 'Name for ' + r.code);
    nameIn.title = 'Write the full name this code stands for.';
    tdName.appendChild(nameIn);
    // The save error sits under the name input, inside this row. A separate
    // row would take an nth-child slot and flip the stripes beneath it.
    var errLine = el('div', 'lb-inline-error');
    errLine.setAttribute('role', 'alert');
    errLine.hidden = true;
    tdName.appendChild(errLine);
    tr.appendChild(tdName);

    // The code itself never changes; only the name and category do.
    var tdCode = el('td', 'col-code');
    tdCode.appendChild(codeChip(r.code));
    tr.appendChild(tdCode);

    // category input
    var tdCat = el('td', 'col-category');
    var catIn = el('input');
    catIn.type = 'text';
    catIn.value = r.category || '';
    catIn.setAttribute('list', 'cat-list');
    catIn.setAttribute('aria-label', 'Category for ' + r.code);
    catIn.title = 'Reuse one of the categories already in the list, or type a new one.';
    tdCat.appendChild(catIn);
    tr.appendChild(tdCat);

    // actions
    var tdAct = el('td', 'col-edit');
    var actWrap = el('div', 'lb-row-actions');
    var saveBtn = el('button', 'btn btn-mini', 'Save');
    saveBtn.type = 'button';
    saveBtn.title = 'VICARIUS backs up the canonical file, writes the change, and mirrors the copy every project reads.';
    var cancelBtn = el('button', 'btn btn-mini', 'Cancel');
    cancelBtn.type = 'button';
    actWrap.appendChild(saveBtn);
    actWrap.appendChild(cancelBtn);
    tdAct.appendChild(actWrap);
    tr.appendChild(tdAct);

    cancelBtn.addEventListener('click', function () { loadVocabulary(); });
    saveBtn.addEventListener('click', function () {
      showError(errLine, '');
      saveBtn.disabled = true;
      postJSON('/api/label', {
        code: r.code,
        category: catIn.value.trim(),
        name: nameIn.value.trim(),
        is_new: false
      }).then(function (res) {
        saveBtn.disabled = false;
        if (!res.ok) { showError(errLine, res.data.error || ('HTTP ' + res.status)); return; }
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
    // The whole action row goes, not just its button, so the form takes the
    // row's place rather than floating under an empty row's padding.
    setHidden('vocab-actions', true);
    $('add-code').focus();
  }
  function closeAddForm() {
    show($('vocab-addform'), false);
    setHidden('vocab-actions', false);
    if ($('vocab-add')) $('vocab-add').focus();
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

  // A remap log row names its codes old_code and new_code (labels_io writes
  // the log; see its remaps[] entries, which also carry action, new_name,
  // new_category and points_affected). {old,new} and {from,to} are tolerated
  // so a hand-written log still renders.
  function remapPair(entry) {
    var oldC = entry.old_code != null ? entry.old_code
      : (entry.old != null ? entry.old : (entry.from != null ? entry.from : ''));
    var newC = entry.new_code != null ? entry.new_code
      : (entry.new != null ? entry.new : (entry.to != null ? entry.to : ''));
    return { old: oldC, new: newC, newName: entry.new_name || '', action: entry.action || '',
             points: entry.points_affected };
  }

  // A code in a sentence: the full name with the code after it in
  // parentheses when the vocabulary (or the log) can supply one, or the bare
  // code when nothing resolves (a code merged away and gone from the file).
  function describeCode(code, name) {
    var row = labelRow(code);
    var full = name || (row ? row.name : '');
    return full ? (full + ' (' + code + ')') : String(code || '');
  }

  // One table cell: the full name when the vocabulary (or the log) can
  // resolve it, then the code as a chip.
  function codeCell(code, name) {
    var td = el('td');
    var row = labelRow(code);
    var full = name || (row ? row.name : '');
    if (full) td.appendChild(nameNode(full, row ? row.category : ''));
    td.appendChild(codeChip(code));
    return td;
  }

  function renderActiveRemap(active) {
    var meta = $('remap-active-meta');
    clear(meta);
    var table = $('remap-active-table');
    var tbody = $('remap-active-tbody');
    var emptyLine = $('remap-active-empty');
    var exWrap = $('remap-active-excludes');
    clear(tbody);

    if (!active) {
      show(table, false);
      show(exWrap, false);
      show(emptyLine, true);
      return;
    }
    show(emptyLine, false);

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
      // The row's tooltip names every code in full. A rename keeps its code,
      // so the vocabulary already holds the new name and the sentence says
      // which name the code now carries.
      if (p.action === 'merge') {
        tr.title = (p.points != null
          ? 'Merge: the ' + p.points + ' points recorded as ' + describeCode(p.old, '')
            + ' become ' + describeCode(p.new, p.newName) + '.'
          : 'Merge: every point recorded as ' + describeCode(p.old, '')
            + ' becomes ' + describeCode(p.new, p.newName) + '.');
      } else if (p.action) {
        var nowRow = labelRow(p.new);
        var nowName = p.newName || (nowRow ? nowRow.name : '');
        tr.title = 'Rename: the code ' + p.new + ' stays'
          + (nowName ? ' and its name is now ' + nowName : ' and takes a new name')
          + (p.points != null ? ' (' + p.points + ' points).' : '.');
      }
      tr.appendChild(codeCell(p.old, ''));
      tr.appendChild(el('td', 'lb-arrow', 'to'));
      tr.appendChild(codeCell(p.new, p.newName));
      tbody.appendChild(tr);
    });

    var excludes = active.excludes || [];
    show(exWrap, excludes.length > 0);
    if (excludes.length) {
      var list = $('remap-active-exchips');
      clear(list);
      excludes.forEach(function (c) {
        var item = el('span', 'lb-excl');
        var row = labelRow(c);
        if (row) {
          item.appendChild(nameNode(row.name, row.category));
        } else {
          item.title = 'The vocabulary has no label with this code, so the recode has nothing to drop for it.';
        }
        item.appendChild(codeChip(c));
        list.appendChild(item);
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
      wrap.appendChild(el('div', 'lb-hint',
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
    oldIn.setAttribute('aria-label', 'Old code');
    oldIn.title = 'Type the code this remap collapses.';
    oldIn.addEventListener('input', function () { r.old = oldIn.value.trim().toUpperCase(); oldIn.value = r.old; });
    var arrow = el('span', 'row-arrow', 'to');
    var newIn = el('input');
    newIn.type = 'text';
    newIn.placeholder = 'NEW';
    newIn.value = r.new || '';
    newIn.setAttribute('aria-label', 'New code');
    newIn.title = 'Type the code the old one becomes.';
    newIn.addEventListener('input', function () { r.new = newIn.value.trim().toUpperCase(); newIn.value = r.new; });
    var del = el('button', 'btn btn-mini', 'Remove');
    del.type = 'button';
    del.title = 'Take this remap out of the draft.';
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
    input.setAttribute('aria-label', 'Code to exclude');
    input.title = 'Type a code the recode should drop instead of rewrite.';
    input.addEventListener('input', function () { x.code = input.value.trim().toUpperCase(); input.value = x.code; });
    var del = el('button', 'btn btn-mini', 'Remove');
    del.type = 'button';
    del.title = 'Take this exclude out of the draft.';
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
      note.textContent = 'No remap log yet, so there is nothing to recode from.';
      $('recode-run').disabled = true;
    } else {
      note.textContent = 'From ' + latest.name;
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
    $('lb-generated').textContent = 'Loaded ' + stamp24() + ' (AST)';

    // vocabulary: remembered search and sort first, then the handlers
    restoreSort();
    $('vocab-search').value = readPref('search') || '';
    $('vocab-search').addEventListener('input', onSearchInput);
    $('vocab-cat').addEventListener('change', onCategoryChange);
    if ($('vocab-clear')) $('vocab-clear').addEventListener('click', clearVocabFilters);
    var sortButtons = document.querySelectorAll('#vocab-table th[data-sort] .lb-sort');
    Array.prototype.forEach.call(sortButtons, function (btn) {
      btn.addEventListener('click', function () { cycleSort(btn.parentNode.dataset.sort); });
    });
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
    loadVocabulary(function () {
      loadRemaps();
      loadLocations();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

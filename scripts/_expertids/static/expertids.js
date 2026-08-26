/* Add-Expert-IDs UI behavior. Self-contained (IIFE -> window.ExpertIDs) so it can
   live alongside a host app's own scripts. All requests go to window.EXPERTIDS_BASE
   (e.g. "/expertids"); the path fields shown in the form are sent with each action,
   so there is no server-side session state to keep in sync. */
(function () {
  'use strict';
  var BASE = window.EXPERTIDS_BASE || '';
  var populated = false;

  var $ = function (id) { return document.getElementById(id); };
  var val = function (id) { var e = $(id); return e ? e.value.trim() : ''; };
  var libVal = function () { var v = val('cfg-library'); return v.indexOf('(') === 0 ? '' : v; };
  function _esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
      return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]; });
  }

  // ── status + counts ───────────────────────────────────────────────
  async function statusRefresh(populateFields) {
    var qs = '';
    if (!populateFields) {
      qs = '?' + new URLSearchParams({ review_dir: val('cfg-review'), lib: libVal() }).toString();
    }
    var d;
    try { d = await (await fetch(BASE + '/api/status' + qs)).json(); }
    catch (e) { return; }
    if ($('s-pending')) $('s-pending').textContent = d.pending_review;
    if ($('s-lib')) $('s-lib').textContent = d.library_total;
    if ($('s-expert')) $('s-expert').textContent = d.library_expert;
    if ($('hist')) $('hist').innerHTML = (d.history || []).map(function (h) {
      return '<div class="row">' + h.at + ' - +' + (h.reviews_added || 0) + ' tentative, '
        + (h.auto_tentative || 0) + ' via overlap, ' + ((h.unrouted || []).length) + ' unrouted, pushed '
        + h.pushed + '</div>';
    }).join('') || '<div class="row">No imports yet.</div>';
    if (populateFields) {
      if ($('cfg-review')) $('cfg-review').value = d.review_dir || '';
      if ($('cfg-export')) $('cfg-export').value = d.export_dir || '';
      if ($('cfg-library')) $('cfg-library').value = d.library_dir || '(module default)';
      if ($('review-site-url')) $('review-site-url').value = d.review_site_url || '';
    }
  }

  // ── Step 3: import CSV (rolling) ──────────────────────────────────
  async function sendFile(file) {
    if ($('err')) $('err').textContent = '';
    var fd = new FormData();
    fd.append('csv', file);
    if (val('cfg-reviewer')) fd.append('reviewer', val('cfg-reviewer'));
    fd.append('export_dir', val('cfg-export'));
    fd.append('review_dir', val('cfg-review'));
    fd.append('library_dir', libVal());
    var d;
    try { d = await (await fetch(BASE + '/api/import', { method: 'POST', body: fd })).json(); }
    catch (e) { d = { error: e.message }; }
    if (!d.ok) { if ($('err')) $('err').textContent = d.error || 'Import failed'; return; }
    showResult(d);
    statusRefresh(false);
    loadCatalog();
    loadPendingByProject();
    loadConsensus();
    loadEmailRequests();
  }

  function showResult(d) {
    var res = $('result'); if (!res) return;
    res.classList.add('show', 'ok');
    var nProj = (d.projects || []).length;
    $('result-head').innerHTML = '<strong>Imported ' + d.rows + ' row(s) across ' + nProj + ' project(s).</strong>'
      + (d.pushed ? ' Tentative IDs pushed to the review site.' : ' (push skipped or unavailable; committed locally.)');
    var cells = [
      ['tentative reviews added', d.reviews_added], ['auto-tentative (>50% overlap)', d.auto_tentative],
      ['unknown codes (kept)', d.unknown_code], ['unsafe UIDs (skipped)', d.unsafe_uid],
      ['unrouted (reported)', (d.unrouted || []).length], ['still pending', d.pending_total],
    ];
    $('result-grid').innerHTML = cells.map(function (c) {
      return '<div class="cell"><b>' + (c[1] == null ? 0 : c[1]) + '</b><br>' + c[0] + '</div>'; }).join('');
    var ur = $('result-unrouted');
    if (ur) {
      if (d.unrouted && d.unrouted.length) {
        ur.innerHTML = '<b class="warn">Unrouted UIDs (no project resolved; fix the project_id / open the project, then re-drop):</b> '
          + d.unrouted.map(_esc).join(', ');
      } else { ur.innerHTML = ''; }
    }
  }

  // ── pending by project (CONTRACTS §2) ─────────────────────────────
  async function loadPendingByProject() {
    var host = $('pbp'); if (!host) return;
    var qs = '?' + new URLSearchParams({ review_dir: val('cfg-review') }).toString();
    var d;
    try { d = await (await fetch(BASE + '/api/pending_by_project' + qs)).json(); }
    catch (e) { return; }
    var bar = $('pbp-actions');
    if (!d.ok || !(d.projects || []).length) {
      host.innerHTML = '<div class="row">No masks pending review.</div>';
      if (bar) bar.style.display = 'none';
      return;
    }
    if (bar) bar.style.display = '';
    host.innerHTML = d.projects.map(function (b) {
      var pid = b.project_id || '';
      var name = b.project_name || b.project_id || '(unknown project)';
      var attrs = ' data-pid="' + _esc(pid) + '" data-name="' + _esc(name) + '" data-count="' + b.count + '"';
      return '<div class="row pbp-row">'
        + '<input type="checkbox" class="pbp-check"' + attrs + (pid ? '' : ' disabled')
        + ' title="Select this project for the Delete selected button below.">'
        + '<b>' + b.count + '</b><span class="pbp-name">' + _esc(name) + '</span>'
        + (pid ? '<button class="btn ghost sm2 pbp-del"' + attrs + ' onclick="ExpertIDs.deleteProject(this)"'
          + ' title="Remove this project\'s pending masks from the public review site. Accepted IDs and the expert-ID library are unaffected.">Delete from site</button>' : '')
        + '</div>';
    }).join('');
  }

  // ── batch delete projects from the review site ────────────────────
  function _confirmProjectDelete(list) {
    var lines = list.map(function (p) {
      return '- ' + p.name + ' (' + p.count + ' item' + (p.count === 1 ? '' : 's') + ')';
    });
    return window.confirm(
      'Delete pending masks from the public review site for:\n\n' + lines.join('\n')
      + '\n\nThis removes the pending masks from the public review site. '
      + 'Accepted IDs and the expert-ID library are unaffected.');
  }

  async function _deleteProjects(list) {
    var note = $('pbp-note');
    var okCount = 0, removed = 0, errs = [];
    for (var i = 0; i < list.length; i++) {
      var p = list[i];
      if (note) { note.textContent = 'Deleting ' + p.name + '...'; note.className = 'pbp-note'; }
      var d;
      try {
        d = await (await fetch(BASE + '/api/delete_project', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ project_id: p.pid, review_dir: val('cfg-review') })
        })).json();
      } catch (e) { d = { error: e.message }; }
      if (d.ok) { okCount++; removed += (d.removed || 0); }
      else { errs.push(p.name + ': ' + (d.error || 'delete failed')); }
    }
    if (note) {
      note.textContent = errs.length
        ? ('Removed ' + removed + ' item(s); failed: ' + errs.join('; '))
        : ('Removed ' + removed + ' item(s) across ' + okCount + ' project(s).');
      note.className = errs.length ? 'pbp-note warn' : 'pbp-note good';
    }
    statusRefresh(false);
    loadCatalog();
    loadPendingByProject();
    loadConsensus();
    loadEmailRequests();
  }

  function deleteProject(btn) {
    var p = { pid: btn.getAttribute('data-pid') || '',
              name: btn.getAttribute('data-name') || '',
              count: parseInt(btn.getAttribute('data-count') || '0', 10) };
    if (!p.pid) return;
    if (!_confirmProjectDelete([p])) return;
    _deleteProjects([p]);
  }

  function deleteSelectedProjects() {
    var note = $('pbp-note');
    var list = [];
    document.querySelectorAll('#pbp .pbp-check:checked').forEach(function (c) {
      if (c.getAttribute('data-pid')) {
        list.push({ pid: c.getAttribute('data-pid'), name: c.getAttribute('data-name') || '',
                    count: parseInt(c.getAttribute('data-count') || '0', 10) });
      }
    });
    if (!list.length) {
      if (note) { note.textContent = 'Check at least one project first.'; note.className = 'pbp-note warn'; }
      return;
    }
    if (!_confirmProjectDelete(list)) return;
    _deleteProjects(list);
  }

  // ── C2/C3: operator setup + per-project email requests ────────────
  async function loadOperatorSetup() {
    var qs = '?' + new URLSearchParams({ review_dir: val('cfg-review') }).toString();
    var d;
    try { d = await (await fetch(BASE + '/api/operator_setup' + qs)).json(); }
    catch (e) { return; }
    if (!d.ok) return;
    if ($('cfg-op-email')) $('cfg-op-email').value = d.email || '';
    if ($('cfg-op-codes')) $('cfg-op-codes').value = (d.candidate_codes || []).join(', ');
  }

  async function saveOperatorSetup() {
    var note = $('op-note'); if (note) { note.textContent = 'Saving…'; note.className = 'op-note'; }
    var body = { review_dir: val('cfg-review'), email: val('cfg-op-email'), candidate_codes: val('cfg-op-codes') };
    var d;
    try {
      d = await (await fetch(BASE + '/api/operator_setup', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
      })).json();
    } catch (e) { d = { error: e.message }; }
    if (note) {
      if (d.ok) { note.textContent = 'Saved.'; note.className = 'op-note good'; }
      else { note.textContent = d.error || 'Save failed'; note.className = 'op-note warn'; }
    }
    loadEmailRequests();
  }

  function _emailParts(text) {
    var lines = String(text || '').split('\n');
    var subject = (lines[0] || '').replace(/^Subject:\s*/i, '');
    var body = lines.slice(1).join('\n').replace(/^\n+/, '');
    return { subject: subject, body: body };
  }

  async function loadEmailRequests() {
    var host = $('email-requests'); if (!host) return;
    var qs = '?' + new URLSearchParams({ review_dir: val('cfg-review') }).toString();
    var d;
    try { d = await (await fetch(BASE + '/api/email_requests' + qs)).json(); }
    catch (e) { return; }
    if (!d.ok || !(d.requests || []).length) {
      host.innerHTML = '<div class="row">No projects with masks awaiting review.</div>'; return;
    }
    host.innerHTML = d.requests.map(function (r, i) {
      var parts = _emailParts(r.text);
      var mailto = 'mailto:?subject=' + encodeURIComponent(parts.subject)
        + '&body=' + encodeURIComponent(parts.body);
      return '<div class="emreq"><div class="emreq-head"><b>' + _esc(r.project_name || r.project_id) + '</b>'
        + ' <span class="muted">(' + r.count + ' mask' + (r.count === 1 ? '' : 's') + ')</span>'
        + '<a class="btn ghost sm2 emreq-mailto" href="' + _esc(mailto) + '"'
        + ' title="Open a pre-filled email draft (subject and body) in your mail app; add the expert\'s address and send.">Email draft</a>'
        + '<button class="btn ghost sm2" onclick="ExpertIDs.copyEmailReq(' + i + ')"'
        + ' title="Copy the full request text (subject and body) to the clipboard.">Copy</button></div>'
        + '<textarea class="emreq-text" id="emreq-' + i + '" readonly rows="13">' + _esc(r.text) + '</textarea></div>';
    }).join('');
  }

  async function copyEmailReq(i) {
    var el = $('emreq-' + i); if (!el) return;
    try { await navigator.clipboard.writeText(el.value); }
    catch (e) { el.select(); document.execCommand('copy'); }
    el.classList.add('copied'); setTimeout(function () { el.classList.remove('copied'); }, 900);
  }

  // ── Step 1: publish (preview, then confirm push) ──────────────────
  function syncBusy(on, label) { var b = $('sync-btn'); if (b) { b.disabled = on; if (label) b.textContent = label; } }
  function syncCell(n, l) { return '<div class="cell"><b>' + n + '</b><span>' + l + '</span></div>'; }
  function renderSync(summaryHtml, actionsHtml) {
    if ($('sync-summary')) $('sync-summary').innerHTML = summaryHtml;
    if ($('sync-actions')) $('sync-actions').innerHTML = actionsHtml;
    if ($('sync-panel')) $('sync-panel').classList.add('show');
  }
  function closeSync() { if ($('sync-panel')) $('sync-panel').classList.remove('show'); }

  function _reviewBody() {
    return { method: 'POST', headers: { 'Content-Type': 'application/json' },
             body: JSON.stringify({ review_dir: val('cfg-review') }) };
  }

  async function previewSite() {
    syncBusy(true, 'Checking live site…');
    var d;
    try { d = await (await fetch(BASE + '/api/site_preview', _reviewBody())).json(); }
    catch (e) { d = { error: e.message }; }
    syncBusy(false, 'Check & update review site');
    if (!d.ok) {
      renderSync('<div class="sync-note warn">Could not check the site: ' + _esc(d.error || 'unknown error') + '</div>',
                 '<button class="btn ghost" onclick="ExpertIDs.closeSync()">Close</button>');
      return;
    }
    var nums = '<div class="nums">'
      + syncCell(d.additions.length, 'new to review')
      + syncCell(d.completions.length, 'completed / removed')
      + syncCell(d.pending_total, 'pending after push')
      + '</div>';
    var note = '';
    if (!d.remote_known)
      note = '<div class="sync-note warn">Couldn’t read the live site’s current state (offline, or it has never been pushed). Pushing will publish the ' + d.pending_total + ' item(s) currently queued.</div>';
    else if (!d.has_changes)
      note = '<div class="sync-note good">Live site is already up to date; nothing to push.</div>';
    var actions = (d.has_changes || !d.remote_known)
      ? '<button class="btn" onclick="ExpertIDs.pushSite()">Push to GitHub</button><button class="btn ghost" onclick="ExpertIDs.closeSync()">Cancel</button>'
      : '<button class="btn ghost" onclick="ExpertIDs.closeSync()">Close</button>';
    renderSync(note + nums, actions);
  }

  async function pushSite() {
    renderSync('<div class="sync-note">Pushing to GitHub…</div>', '');
    var d;
    try { d = await (await fetch(BASE + '/api/site_push', _reviewBody())).json(); }
    catch (e) { d = { error: e.message }; }
    if (!d.ok) {
      renderSync('<div class="sync-note warn">Push failed: ' + _esc(d.error || 'unknown error') + '</div>',
                 '<button class="btn ghost" onclick="ExpertIDs.closeSync()">Close</button>');
      return;
    }
    var msg = d.pushed
      ? '<div class="sync-note good">Pushed. Live site now has ' + d.pending_total + ' pending item(s).</div>'
      : '<div class="sync-note warn">Committed locally, but the push did not go through (check gh auth / network). The live site is unchanged; try again.</div>';
    renderSync(msg, '<button class="btn ghost" onclick="ExpertIDs.closeSync()">Close</button>');
    statusRefresh(false);
  }

  // ── Step 2: review-site link ──────────────────────────────────────
  function openReviewSite() {
    var url = val('review-site-url');
    if (url) window.open(url, '_blank', 'noopener');
  }
  async function copyReviewLink() {
    var url = val('review-site-url');
    if (!url) return;
    try { await navigator.clipboard.writeText(url); }
    catch (e) { var el = $('review-site-url'); if (el) { el.select(); document.execCommand('copy'); } }
    var note = $('copy-note');
    if (note) { note.classList.add('show'); setTimeout(function () { note.classList.remove('show'); }, 1800); }
  }

  // ── mask catalog ──────────────────────────────────────────────────
  function _opt(sel, vals) {
    var keep = sel.value || '';
    sel.innerHTML = '<option value="">all</option>' + vals.map(function (v) {
      return '<option' + (v === keep ? ' selected' : '') + '>' + v + '</option>'; }).join('');
  }

  async function loadCatalog() {
    var lib = libVal();
    var params = { code: val('f-code'), mode: val('f-mode'), site: val('f-site'),
                   year: val('f-year'), q: val('f-q'), limit: 200 };
    if (lib) params.lib = lib;
    var d;
    try { d = await (await fetch(BASE + '/api/catalog?' + new URLSearchParams(params).toString())).json(); }
    catch (e) { if ($('cat-meta')) $('cat-meta').textContent = 'Could not load catalog: ' + e.message; return; }
    _opt($('f-code'), d.facets.code || []); _opt($('f-mode'), d.facets.mode || []);
    _opt($('f-site'), d.facets.site || []); _opt($('f-year'), d.facets.year || []);
    if ($('cat-meta')) $('cat-meta').textContent = d.count + ' of ' + d.total + ' masks shown'
      + (d.total > d.count ? ' (refine filters to see more)' : '');
    var host = $('tiles'); if (!host) return; host.innerHTML = '';
    if (!d.tiles.length) {
      host.innerHTML = '<div class="cat-empty">No masks in the library yet; export a review batch or import an expert CSV first.</div>';
      return;
    }
    var libQ = lib ? ('?lib=' + encodeURIComponent(lib)) : '';
    d.tiles.forEach(function (t) {
      var tile = document.createElement('div'); tile.className = 'tile';
      var area = (t.area != null) ? (Number(t.area).toLocaleString() + ' px') : '-';
      var ctx = [t.site || '?', (t.transect !== '' ? 'T' + t.transect : ''), (t.year || '')].filter(Boolean).join(' · ');
      var uidE = encodeURIComponent(t.uid);
      var pic = t.has_image
        ? '<div class="pic"><img class="base" src="' + BASE + '/catalog/' + uidE + '/image.jpg' + libQ + '" alt="">'
          + (t.has_overlay ? '<img class="ov" src="' + BASE + '/catalog/' + uidE + '/overlay.png' + libQ + '" style="opacity:0.45">' : '')
          + '</div>'
        : '<div class="pic"><span style="color:#667;font-size:11px">no image</span></div>';
      tile.innerHTML = pic
        + '<div class="meta"><span class="mode ' + _esc(t.mode) + '">' + _esc(t.mode || '?') + '</span>'
        + '<span class="code">' + _esc(t.code || '?') + '</span>'
        + (t.name ? '<div>' + _esc(t.name) + '</div>' : '')
        + '<div class="row2">' + _esc(ctx) + '</div>'
        + '<div class="row2">area ' + area + (t.labeler ? ' · ' + _esc(t.labeler) : '') + '</div></div>'
        + (t.has_overlay ? '<div class="op"><span>mask</span><input type="range" min="0" max="100" value="45"><span>opacity</span></div>' : '');
      if (t.has_overlay) {
        var ov = tile.querySelector('img.ov'), rng = tile.querySelector('input[type=range]');
        rng.addEventListener('input', function () { ov.style.opacity = rng.value / 100; });
      }
      host.appendChild(tile);
    });
  }

  // ── C6: consensus builder ─────────────────────────────────────────
  function _statusChip(s) {
    var label = { consensus: 'consensus', conflict: 'conflict', single: 'single', none: 'no reviews' }[s] || s;
    return '<span class="chip ' + _esc(s) + '">' + _esc(label) + '</span>';
  }

  function _consensusRow(r, rev) {
    var reviews = r.reviews || [];
    var byRev = {};
    reviews.forEach(function (rv) { byRev[rv.reviewer] = rv; });
    var reviewDir = encodeURIComponent(val('cfg-review'));
    var uidE = encodeURIComponent(r.uid);
    var thumbBase = BASE + '/api/consensus_asset/' + uidE;
    var rd = reviewDir ? ('?review_dir=' + reviewDir) : '';
    var thumb = '<div class="cons-thumb"><img src="' + thumbBase + '/crop' + rd + '" alt="" '
      + 'onerror="this.style.display=\'none\'"></div>';
    var cells = rev.map(function (name) {
      var rv = byRev[name];
      if (!rv) return '<td class="empty">-</td>';
      return '<td><span class="rv-code">' + _esc(rv.code || '?') + '</span></td>';
    }).join('');
    // Final-code selector defaults to the suggested (consensus) code.
    var opts = '<option value="">- pick -</option>';
    var seen = {};
    (r.codes || []).concat(r.suggested ? [r.suggested] : []).forEach(function (c) {
      if (!c || seen[c]) return; seen[c] = 1;
      opts += '<option value="' + _esc(c) + '"' + (c === r.suggested ? ' selected' : '') + '>' + _esc(c) + '</option>';
    });
    var ctx = [_esc(r.project_name || r.project_id), _esc(r.site || ''), _esc(r.frame || '')].filter(Boolean).join(' · ');
    var canAccept = r.status !== 'none' && r.status !== 'conflict';
    return '<tr data-uid="' + _esc(r.uid) + '">'
      + '<td class="cons-id">' + thumb + '<div class="cons-meta">' + _statusChip(r.status)
      + '<div class="cons-ctx">' + ctx + '</div>'
      + (r.name ? '<div class="cons-name">' + _esc(r.name) + '</div>' : '') + '</div></td>'
      + cells
      + '<td class="cons-accept">'
      + '<select class="cons-final">' + opts + '</select>'
      + '<button class="btn sm2" onclick="ExpertIDs.accept(this)"'
      + (canAccept ? '' : ' title="Resolve the conflict (or wait for a review) before accepting"')
      + '>Accept</button></td>'
      + '</tr>';
  }

  async function loadConsensus() {
    var host = $('cons-wrap'); if (!host) return;
    if ($('cons-err')) $('cons-err').textContent = '';
    var lib = libVal();
    var params = { review_dir: val('cfg-review') }; if (lib) params.lib = lib;
    var d;
    try { d = await (await fetch(BASE + '/api/consensus?' + new URLSearchParams(params).toString())).json(); }
    catch (e) { if ($('cons-err')) $('cons-err').textContent = 'Could not load: ' + e.message; return; }
    if (!d.ok) { if ($('cons-err')) $('cons-err').textContent = d.error || 'Failed to load consensus'; return; }
    var c = d.counts || {};
    if ($('cons-summary')) $('cons-summary').textContent =
      d.total + ' pending: ' + (c.consensus || 0) + ' consensus, ' + (c.conflict || 0) + ' conflict, '
      + (c.single || 0) + ' single, ' + (c.none || 0) + ' no reviews';
    if (!d.rows.length) { host.innerHTML = '<div class="cons-empty">No masks pending consensus.</div>'; return; }
    // Distinct reviewer columns across all rows (excluding the synthetic overlap reviewer last).
    var revSet = {}, rev = [];
    d.rows.forEach(function (r) { (r.reviews || []).forEach(function (rv) {
      if (!revSet[rv.reviewer]) { revSet[rv.reviewer] = 1; rev.push(rv.reviewer); } }); });
    rev.sort(function (a, b) { return (a === 'overlap') - (b === 'overlap') || a.localeCompare(b); });
    var head = '<tr><th>mask</th>' + rev.map(function (n) { return '<th>' + _esc(n) + '</th>'; }).join('')
      + '<th>final</th></tr>';
    host.innerHTML = '<table class="cons-table"><thead>' + head + '</thead><tbody>'
      + d.rows.map(function (r) { return _consensusRow(r, rev); }).join('') + '</tbody></table>';
  }

  async function accept(btn) {
    var tr = btn.closest('tr'); if (!tr) return;
    var uid = tr.getAttribute('data-uid');
    var sel = tr.querySelector('select.cons-final');
    var code = sel ? sel.value : '';
    if (!code) { if ($('cons-err')) $('cons-err').textContent = 'Pick a final code before accepting.'; return; }
    btn.disabled = true; btn.textContent = 'Accepting…';
    // Send the SAME paths the panel is showing (stateless contract): the
    // consensus table was read from cfg-library, so accept must write there,
    // never the provider default. Blank values fall back server-side.
    var body = { uid: uid, code: code, review_dir: val('cfg-review'), basis: 'operator',
                 library_dir: libVal(), export_dir: val('cfg-export') };
    var d;
    try {
      d = await (await fetch(BASE + '/api/accept', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
      })).json();
    } catch (e) { d = { error: e.message }; }
    if (!d.ok) {
      btn.disabled = false; btn.textContent = 'Accept';
      if ($('cons-err')) $('cons-err').textContent = d.error || 'Accept failed';
      return;
    }
    tr.classList.add('accepted');
    btn.textContent = 'Accepted';
    statusRefresh(false);
    loadCatalog();
    loadPendingByProject();
    loadEmailRequests();
    setTimeout(loadConsensus, 400);
  }

  // ── wiring ────────────────────────────────────────────────────────
  function init() {
    var drop = $('drop'), fileInput = $('file');
    if (drop && fileInput) {
      drop.addEventListener('click', function () { fileInput.click(); });
      fileInput.addEventListener('change', function (e) { if (e.target.files[0]) sendFile(e.target.files[0]); });
      ['dragenter', 'dragover'].forEach(function (ev) {
        drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add('over'); }); });
      ['dragleave', 'drop'].forEach(function (ev) {
        drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove('over'); }); });
      drop.addEventListener('drop', function (e) { var f = e.dataTransfer.files[0]; if (f) sendFile(f); });
    }
    ['f-code', 'f-mode', 'f-site', 'f-year'].forEach(function (id) {
      var e = $(id); if (e) e.addEventListener('change', loadCatalog); });
    var fq = $('f-q');
    if (fq) fq.addEventListener('keydown', function (e) { if (e.key === 'Enter') loadCatalog(); });
  }

  // Called when the panel becomes visible (host hook). First time: populate the
  // path fields + counts from defaults; thereafter just refresh counts + catalog.
  async function onShow() {
    await statusRefresh(!populated);
    populated = true;
    loadCatalog();
    loadPendingByProject();
    loadOperatorSetup();
    loadEmailRequests();
    loadConsensus();
  }

  window.ExpertIDs = {
    onShow: onShow, previewSite: previewSite, pushSite: pushSite, closeSync: closeSync,
    copyReviewLink: copyReviewLink, openReviewSite: openReviewSite, loadCatalog: loadCatalog,
    loadPendingByProject: loadPendingByProject, loadConsensus: loadConsensus, accept: accept,
    saveOperatorSetup: saveOperatorSetup, loadEmailRequests: loadEmailRequests, copyEmailReq: copyEmailReq,
    deleteProject: deleteProject, deleteSelectedProjects: deleteSelectedProjects,
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();

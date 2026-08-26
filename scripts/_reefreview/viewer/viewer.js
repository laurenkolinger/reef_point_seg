/* TCRMP Coral expert-review viewer - static, no backend.
   Reads review_manifest.json + codes.json (same dir), renders one card per
   flagged mask, persists answers in localStorage, exports a CSV whose rows
   carry the reviewer's name (CONTRACTS §4,§7,§8 + C1-C5).

   This file is BOTH the browser viewer AND a node-requireable logic module:
   the pure helpers (CSV row builder, site-name map, project/site filter
   predicate, bulk-apply selection, grouped render order via sortItems /
   groupByProject) are exported via module.exports when loaded
   under node (no `document`); the DOM bootstrap runs only in a browser. Mirror
   of the placePoints pp_core.js pattern so the logic is unit-testable with
   `node tests/test_viewer_core.js` and the page still loads via one <script>. */
(function (root) {
  "use strict";

  // ── PURE LOGIC (browser + node) ───────────────────────────────────
  // No DOM, no localStorage - exported for unit tests.

  var CSV_HEADER = ["uid", "code", "confidence", "reviewer", "project_id"];
  var CSV_HUMAN = ["project_name", "site", "frame"];   // optional, ignored on import

  // Full site name for an item: codes.json.sites[site_code] wins, then the
  // export-stamped site_full, then the raw site_code, then "?". (C4 / CONTRACTS §7)
  function siteName(item, sites) {
    item = item || {};
    sites = sites || {};
    var code = (item.site_code || item.site || "");
    var key = String(code).toUpperCase();
    if (sites[key]) return sites[key];
    if (sites[code]) return sites[code];
    if (item.site_full) return item.site_full;
    return code || "?";
  }

  // Phase-4 project label: project_name, falling back to project_id. (C/Phase 4)
  function projectLabel(item) {
    item = item || {};
    return item.project_name || item.project_id || "";
  }
  // Stable project KEY used for filtering + the filename suffix.
  function projectKey(item) {
    item = item || {};
    return item.project_id || item.project_name || "";
  }

  // Distinct projects across the manifest, ordered by label then key. Each:
  // {key, label}. (Phase-4 dropdown source)
  function distinctProjects(items) {
    var seen = {}, out = [];
    (items || []).forEach(function (it) {
      var key = projectKey(it);
      if (key === "" || seen[key]) return;
      seen[key] = 1;
      out.push({ key: key, label: projectLabel(it) || key });
    });
    out.sort(function (a, b) {
      var la = (a.label || "").toLowerCase(), lb = (b.label || "").toLowerCase();
      return la < lb ? -1 : la > lb ? 1 : (a.key < b.key ? -1 : a.key > b.key ? 1 : 0);
    });
    return out;
  }

  // Distinct sites across the manifest, ordered by full name. Each:
  // {code, name}. (C4 site filter source)
  function distinctSites(items, sites) {
    var seen = {}, out = [];
    (items || []).forEach(function (it) {
      var code = (it.site_code || it.site || "");
      var key = String(code);
      if (key === "" || seen[key]) return;
      seen[key] = 1;
      out.push({ code: code, name: siteName(it, sites) });
    });
    out.sort(function (a, b) {
      var na = (a.name || "").toLowerCase(), nb = (b.name || "").toLowerCase();
      return na < nb ? -1 : na > nb ? 1 : 0;
    });
    return out;
  }

  // Does an item pass the active project + site filters? Empty filter = match
  // all. (Phase-4 + C4 filter predicate)
  function matchesFilter(item, filter) {
    item = item || {};
    filter = filter || {};
    if (filter.project && projectKey(item) !== filter.project) return false;
    if (filter.site) {
      var code = String(item.site_code || item.site || "");
      if (code !== String(filter.site)) return false;
    }
    return true;
  }

  // Is this answer a submitted ID (counts as "answered" for export/progress)?
  function isAnswered(a) { return !!(a && a.answered && !a.skipped); }
  // Still-unanswered = not yet a submitted ID and not explicitly skipped. These
  // are the cards a BULK-APPLY touches. (C4 bulk-apply)
  function isUnanswered(a) { return !(a && (a.answered || a.skipped)); }

  // UIDs the bulk-apply control would assign to: every filtered card that is
  // still unanswered (skips already-answered + skipped). (C4)
  function bulkApplyTargets(items, answers, filter) {
    answers = answers || {};
    var out = [];
    (items || []).forEach(function (it) {
      if (!matchesFilter(it, filter)) return;
      if (isUnanswered(answers[it.uid])) out.push(it.uid);
    });
    return out;
  }

  // Per-project pending (still-unanswered) counts within the current answers.
  // Returns {projectKey: count} plus a "" bucket total. (Phase-4 per-project pending)
  function pendingByProject(items, answers) {
    answers = answers || {};
    var out = {};
    (items || []).forEach(function (it) {
      if (!isUnanswered(answers[it.uid])) return;
      var key = projectKey(it);
      out[key] = (out[key] || 0) + 1;
    });
    return out;
  }

  // Render-order sort: project label asc (items with no project fields LAST),
  // then site code, then frame (missing frames last), then uid. Non-mutating.
  function frameOrd(item) {
    var f = item && item.frame;
    if (f == null || f === "") return Infinity;
    var n = Number(f);
    return isNaN(n) ? Infinity : n;
  }
  function sortItems(items) {
    return (items || []).slice().sort(function (a, b) {
      var ka = projectKey(a), kb = projectKey(b);
      if ((ka === "") !== (kb === "")) return ka === "" ? 1 : -1;
      var la = (projectLabel(a) || ka).toLowerCase(), lb = (projectLabel(b) || kb).toLowerCase();
      if (la !== lb) return la < lb ? -1 : 1;
      if (ka !== kb) return ka < kb ? -1 : 1;   // duplicate labels split by key
      var sa = String((a && (a.site_code || a.site)) || "");
      var sb = String((b && (b.site_code || b.site)) || "");
      if (sa !== sb) return sa < sb ? -1 : 1;
      var fa = frameOrd(a), fb = frameOrd(b);
      if (fa !== fb) return fa < fb ? -1 : 1;
      var ua = String((a && a.uid) || ""), ub = String((b && b.uid) || "");
      return ua < ub ? -1 : ua > ub ? 1 : 0;
    });
  }

  // Grouped render order: sortItems, then bucket by project key. Items with no
  // project fields collapse into one trailing "Unknown project" group. Each
  // group: {key, label, items}.
  var UNKNOWN_PROJECT = "Unknown project";
  function groupByProject(items) {
    var byKey = {}, out = [];
    sortItems(items).forEach(function (it) {
      var key = projectKey(it);
      var g = byKey[key];
      if (!g) {
        g = { key: key, label: key === "" ? UNKNOWN_PROJECT : (projectLabel(it) || key), items: [] };
        byKey[key] = g;
        out.push(g);
      }
      g.items.push(it);
    });
    return out;
  }

  // One CSV cell, RFC-4180 quoting.
  function csvCell(v) {
    v = String(v == null ? "" : v);
    return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  }

  // Build the export rows (array-of-arrays incl. header) for the answered cards
  // that pass the active filter. `reviewer` is stamped on EVERY row (C1/§8); a
  // project filter exports only that project's answered rows (§4).
  function buildCsvRows(items, answers, reviewer, filter, opts) {
    answers = answers || {};
    opts = opts || {};
    var human = opts.human !== false;   // include optional human cols by default
    var header = human ? CSV_HEADER.concat(CSV_HUMAN) : CSV_HEADER.slice();
    var rows = [header];
    (items || []).forEach(function (it) {
      var a = answers[it.uid];
      if (!isAnswered(a)) return;
      if (!matchesFilter(it, filter)) return;
      // confidence column retired: always blank, header unchanged for the
      // importer; legacy stored conf values are ignored
      var row = [it.uid, a.code, "", reviewer || "", it.project_id || ""];
      if (human) row = row.concat([it.project_name || "", it.site || "", it.frame == null ? "" : it.frame]);
      rows.push(row);
    });
    return rows;
  }

  // Serialize rows to CSV text.
  function rowsToCsv(rows) {
    return (rows || []).map(function (r) {
      return r.map(csvCell).join(",");
    }).join("\n");
  }

  // Safe filename token (lower, alnum + dash/underscore).
  function slug(s) {
    return String(s == null ? "" : s).trim().toLowerCase()
      .replace(/[^a-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "") || "x";
  }
  // Download filename: tcrmp_expert_ids_<reviewer>_<project|all>.csv (§4)
  function csvFilename(reviewer, projectKeyOrEmpty) {
    var who = slug(reviewer);
    var proj = projectKeyOrEmpty ? slug(projectKeyOrEmpty) : "all";
    return "tcrmp_expert_ids_" + who + "_" + proj + ".csv";
  }

  // Operator/recipient emails for the email-back instruction (C2). Operator
  // email may live in codes.json (contacts/operator) when present; lauren is
  // always appended; de-duped, order preserved.
  var LAUREN = "lauren.olinger@uvi.edu";
  function recipientList(codes, manifest) {
    codes = codes || {};
    manifest = manifest || {};
    var out = [];
    function add(v) {
      if (!v) return;
      if (typeof v === "string") v = [v];
      (v || []).forEach(function (c) {
        c = (c == null ? "" : String(c)).trim();
        if (c && out.indexOf(c) < 0) out.push(c);
      });
    }
    add(codes.operator);
    add(codes.contacts);
    add(manifest.contacts);
    add(LAUREN);
    return out;
  }

  var CORE = {
    CSV_HEADER: CSV_HEADER, CSV_HUMAN: CSV_HUMAN, LAUREN: LAUREN,
    UNKNOWN_PROJECT: UNKNOWN_PROJECT,
    siteName: siteName, projectLabel: projectLabel, projectKey: projectKey,
    distinctProjects: distinctProjects, distinctSites: distinctSites,
    sortItems: sortItems, groupByProject: groupByProject,
    matchesFilter: matchesFilter, isAnswered: isAnswered, isUnanswered: isUnanswered,
    bulkApplyTargets: bulkApplyTargets, pendingByProject: pendingByProject,
    csvCell: csvCell, buildCsvRows: buildCsvRows, rowsToCsv: rowsToCsv,
    slug: slug, csvFilename: csvFilename, recipientList: recipientList,
  };

  // node: export and stop (no DOM to wire). browser: expose + boot below.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = CORE;
    return;
  }
  root.ReefViewerCore = CORE;
  if (typeof document === "undefined") return;   // non-browser host, no boot

  // ── BROWSER VIEWER ────────────────────────────────────────────────
  var LS_ANS = "tcrmp_review_answers_v1";   // {uid:{code,skipped,answered}}; legacy conf values tolerated + ignored
  var LS_NAME = "tcrmp_reviewer_name_v1";    // C1: persisted reviewer identity
  var FILL = "#22d3ee";                      // mask highlight (matches the baked outline)
  var DEFAULT_OPACITY = 0.45;

  var CODES = null, MANIFEST = null, BYCODE = {};
  var answers = load(LS_ANS) || {};
  var REVIEWER = "";
  var FILTER = { project: "", site: "" };

  function load(k) { try { return JSON.parse(localStorage.getItem(k)); } catch (e) { return null; } }
  function save(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} }
  function loadStr(k) { try { return localStorage.getItem(k) || ""; } catch (e) { return ""; } }
  function saveStr(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  function el(tag, cls, txt) { var e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; }

  function getJSON(url) {
    return fetch(url, { cache: "no-store" }).then(function (r) {
      if (!r.ok) throw new Error(url + " -> " + r.status); return r.json();
    });
  }

  function defaultAnswer() {
    return { code: CODES.idk.code, conf: "", skipped: false, answered: false };
  }
  function ansFor(uid) {
    if (!answers[uid]) answers[uid] = defaultAnswer();
    return answers[uid];
  }
  function persist() { save(LS_ANS, answers); refreshCounts(); }

  // ── card ──────────────────────────────────────────────────────────
  function buildCard(item) {
    var a = ansFor(item.uid);
    var card = el("div", "card");
    card.dataset.uid = item.uid;
    card.dataset.project = CORE.projectKey(item);
    card.dataset.site = String(item.site_code || item.site || "");

    // context line - full site name (C4), project, transect/frame
    var ctx = el("div", "ctx");
    var site = el("span", "site", CORE.siteName(item, CODES.sites) + (item.year ? "  ·  " + item.year : ""));
    var meta = el("span", null,
      "T" + (item.transect != null ? item.transect : "?") +
      (item.frame != null ? ("/" + String(item.frame).padStart(2, "0")) : "") );
    var badge = el("span", "badge", "needs ID");
    ctx.appendChild(site); ctx.appendChild(meta); ctx.appendChild(badge);
    card.appendChild(ctx);

    var proj = CORE.projectLabel(item);
    if (proj) {
      var pj = el("div", "projline");
      pj.appendChild(el("span", "pjlabel", "Project"));
      pj.appendChild(el("span", "pjname", proj));
      card.appendChild(pj);
    }

    // C5 tentative IDs from other reviewers
    var tentative = (item.reviews || []).filter(function (r) {
      return r && (r.code || "").trim() && r.code !== CODES.idk.code;
    });
    if (tentative.length) {
      var tbox = el("div", "tentative");
      tbox.appendChild(el("span", "tlabel", "Tentative IDs (other reviewers, not yet accepted)"));
      var tlist = el("div", "tlist");
      tentative.forEach(function (r) {
        var chip = el("span", "tchip");
        chip.appendChild(el("span", "tcode", r.code));
        chip.appendChild(el("span", "twho", r.reviewer || "anon"));
        chip.title = "Tentative ID by " + (r.reviewer || "an anonymous reviewer") +
          ". Not accepted. Provide your own independent ID.";
        tlist.appendChild(chip);
      });
      tbox.appendChild(tlist);
      card.appendChild(tbox);
    }

    // viewer tools
    var tools = el("div", "vtools");
    tools.appendChild(el("span", null, "mask"));
    var slider = el("input"); slider.type = "range"; slider.min = 0; slider.max = 100;
    slider.value = Math.round(DEFAULT_OPACITY * 100);
    slider.title = "Mask fill opacity";
    tools.appendChild(slider);
    var fullBtn = el("button", null, "Show whole frame");
    fullBtn.title = "Toggle between the close-up and the whole frame.";
    tools.appendChild(fullBtn);
    card.appendChild(tools);

    // viewer surface
    var view = el("div", "viewer");
    var wrap = el("div", "imgwrap");
    var canvas = el("canvas");
    wrap.appendChild(canvas);
    view.appendChild(wrap);
    var fullImg = el("img", "full"); fullImg.style.display = "none"; fullImg.alt = "full frame";
    if (item.full) fullImg.src = item.full;
    view.appendChild(fullImg);
    card.appendChild(view);

    // composited closeup
    var cropImg = new Image(), maskImg = new Image(), loaded = { crop: false, mask: false };
    function redraw() {
      if (!loaded.crop) return;
      var w = cropImg.naturalWidth, h = cropImg.naturalHeight;
      canvas.width = w; canvas.height = h;
      var ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(cropImg, 0, 0, w, h);
      if (loaded.mask) {
        var off = document.createElement("canvas"); off.width = w; off.height = h;
        var octx = off.getContext("2d");
        octx.drawImage(maskImg, 0, 0, w, h);
        octx.globalCompositeOperation = "source-in";
        octx.fillStyle = FILL; octx.fillRect(0, 0, w, h);
        ctx.globalAlpha = slider.value / 100;
        ctx.drawImage(off, 0, 0);
        ctx.globalAlpha = 1;
      }
    }
    cropImg.onload = function () { loaded.crop = true; redraw(); };
    maskImg.onload = function () { loaded.mask = true; redraw(); };
    cropImg.onerror = function () { wrap.appendChild(el("div", "empty", "image unavailable")); };
    maskImg.onerror = function () { slider.disabled = true; slider.title = "No mask overlay for this item"; };
    if (item.crop) cropImg.src = item.crop;
    if (item.mask) { maskImg.src = item.mask; }
    else { slider.disabled = true; slider.title = "No mask overlay for this item"; }
    slider.addEventListener("input", redraw);
    fullBtn.addEventListener("click", function () {
      var showing = fullImg.style.display !== "none";
      fullImg.style.display = showing ? "none" : "block";
      wrap.style.display = showing ? "" : "none";
      fullBtn.classList.toggle("on", !showing);
      fullBtn.textContent = showing ? "Show whole frame" : "Show close-up";
    });

    // ── labeling form ──
    var form = el("div", "form");
    var opts = el("div", "opts");

    function setCode(code, answered) {
      a.code = code; a.skipped = false; a.answered = answered !== false;
      syncUI(); persist();
    }

    // featured (target) codes first
    (item.featured_codes || []).forEach(function (code) {
      var info = BYCODE[code] || { name: "" };
      var o = el("div", "opt");
      o.dataset.code = code;
      o.appendChild(el("span", "code", code));
      if (info.name) { o.appendChild(el("span", "nm", info.name)); }
      o.title = (info.name || code) + (info.category ? " (" + info.category + ")" : "");
      o.addEventListener("click", function () { setCode(code, true); });
      opts.appendChild(o);
    });
    // C3 candidate codes (operator-surfaced quick picks), shown after featured
    // and de-duplicated against featured.
    var featuredSet = {};
    (item.featured_codes || []).forEach(function (c) { featuredSet[c] = 1; });
    var cands = (item.candidate_codes && item.candidate_codes.length
                 ? item.candidate_codes
                 : (CODES.candidate_codes || []));
    cands.forEach(function (code) {
      if (!code || featuredSet[code]) return;
      var info = BYCODE[code] || { name: "" };
      var o = el("div", "opt cand");
      o.dataset.code = code;
      o.appendChild(el("span", "code", code));
      if (info.name) { o.appendChild(el("span", "nm", info.name)); }
      o.title = "Candidate code: " + (info.name || code) +
        (info.category ? " (" + info.category + ")" : "");
      o.addEventListener("click", function () { setCode(code, true); });
      opts.appendChild(o);
    });
    // I don't know
    var idk = el("div", "opt idk"); idk.dataset.code = CODES.idk.code;
    idk.appendChild(el("span", "code", CODES.idk.label));
    idk.title = "Use when you cannot determine the ID from the image.";
    idk.addEventListener("click", function () { setCode(CODES.idk.code, true); });
    opts.appendChild(idk);
    // Something else
    var elseOpt = el("div", "opt else"); elseOpt.dataset.code = "__else__";
    elseOpt.appendChild(el("span", "code", CODES.something_else.label));
    elseOpt.title = "Pick any other code from the full list, grouped by category.";
    opts.appendChild(elseOpt);
    form.appendChild(el("div", "lbl", "Identification"));
    form.appendChild(opts);

    // nested "something else" picker
    var elseBox = el("div", "elsebox");
    var grpSel = el("select"); grpSel.title = "Group (coral, algae, sponge, and more).";
    grpSel.appendChild(new Option("- group -", ""));
    CODES.groups.forEach(function (g) { grpSel.appendChild(new Option(g.group, g.group)); });
    var codeSel = el("select"); codeSel.title = "Code within the chosen group."; codeSel.disabled = true;
    codeSel.appendChild(new Option("- code -", ""));
    var chosen = el("span", "chosen", "");
    elseBox.appendChild(grpSel); elseBox.appendChild(codeSel); elseBox.appendChild(chosen);
    form.appendChild(elseBox);

    elseOpt.addEventListener("click", function () { elseBox.classList.add("show"); });
    grpSel.addEventListener("change", function () {
      codeSel.innerHTML = ""; codeSel.appendChild(new Option("- code -", ""));
      var g = CODES.groups.filter(function (x) { return x.group === grpSel.value; })[0];
      if (g) { g.codes.forEach(function (c) {
        var info = BYCODE[c] || {}; codeSel.appendChild(new Option(c + " - " + (info.name || ""), c));
      }); }
      codeSel.disabled = !g;
    });
    codeSel.addEventListener("change", function () {
      if (codeSel.value) { setCode(codeSel.value, true);
        chosen.textContent = "✓ " + codeSel.value + " - " + ((BYCODE[codeSel.value] || {}).name || ""); }
    });

    // skip (confidence UI removed; CSV keeps a blank confidence column)
    var row = el("div", "row");
    var skip = el("button", "skipbtn", "Skip");
    skip.title = "Not sure. Exclude this mask from your results.";
    skip.addEventListener("click", function () {
      a.skipped = !a.skipped; a.answered = false; syncUI(); persist();
    });
    row.appendChild(skip);
    form.appendChild(row);
    card.appendChild(form);

    function syncUI() {
      Array.prototype.forEach.call(opts.children, function (o) {
        var c = o.dataset.code;
        var isElse = c === "__else__";
        var selByCode = !isElse && a.code === c && !a.skipped;
        var pickedInList = (item.featured_codes || []).indexOf(a.code) >= 0 ||
          cands.indexOf(a.code) >= 0;
        var selByElse = isElse && a.answered && !a.skipped &&
          !pickedInList && a.code !== CODES.idk.code;
        o.classList.toggle("sel", selByCode || selByElse);
      });
      skip.classList.toggle("sel", !!a.skipped);
      card.classList.toggle("skip", !!a.skipped);
      var done = a.answered && !a.skipped;
      card.classList.toggle("done", done);
      badge.textContent = a.skipped ? "skipped" : (done ? (a.code) : "needs ID");
      badge.classList.toggle("ok", done);
    }
    card._sync = syncUI;
    syncUI();
    return card;
  }

  // ── project group headers (grouped card render) ──────────────────
  function buildGroupHeader(group) {
    var head = el("div", "grouphead");
    head.dataset.project = group.key;
    head.title = "Project group. The count shows still-unanswered cards in this project.";
    head.appendChild(el("span", "gname", group.label));
    var pend = el("span", "gpending");
    head._pending = pend;
    head.appendChild(pend);
    return head;
  }
  function refreshGroupPending() {
    if (!MANIFEST) return;
    var host = document.getElementById("cards");
    if (!host || !host.querySelectorAll) return;
    var pending = CORE.pendingByProject(MANIFEST.items, answers);
    Array.prototype.forEach.call(host.querySelectorAll(".grouphead"), function (head) {
      if (head._pending)
        head._pending.textContent = (pending[head.dataset.project] || 0) + " pending";
    });
  }
  // Sticky group headers sit below the sticky topbar; measure its height into
  // a CSS var so the offset tracks header wrapping at narrow widths.
  function syncTopbarOffset() {
    if (typeof document.querySelector !== "function") return;
    var tb = document.querySelector(".topbar");
    var rootEl = document.documentElement;
    if (!tb || !rootEl || !rootEl.style || typeof rootEl.style.setProperty !== "function") return;
    rootEl.style.setProperty("--topbar-h", (tb.offsetHeight || 0) + "px");
  }

  // ── filters + bulk apply (Phase 4 + C4) ───────────────────────────
  function applyFilter() {
    var host = document.getElementById("cards");
    var shown = 0, shownByProject = {};
    Array.prototype.forEach.call(host.querySelectorAll(".card"), function (card) {
      // dataset.project / dataset.site hold the resolved filter keys per card
      // (set in buildCard from CORE.projectKey + site_code); compare directly so
      // show/hide uses the same predicate semantics as CORE.matchesFilter.
      var ok = (!FILTER.project || card.dataset.project === FILTER.project) &&
               (!FILTER.site || card.dataset.site === String(FILTER.site));
      card.style.display = ok ? "" : "none";
      if (ok) {
        shown++;
        shownByProject[card.dataset.project] = (shownByProject[card.dataset.project] || 0) + 1;
      }
    });
    // group headers hide with a non-matching project filter, and when the site
    // filter leaves the group with no visible cards
    Array.prototype.forEach.call(host.querySelectorAll(".grouphead"), function (head) {
      var ok = (!FILTER.project || head.dataset.project === FILTER.project) &&
               (shownByProject[head.dataset.project] || 0) > 0;
      head.style.display = ok ? "" : "none";
    });
    var info = document.getElementById("filterInfo");
    if (info) info.textContent = shown + " of " + MANIFEST.items.length + " shown";
    refreshCounts();
  }

  function bulkApply(code) {
    if (!code) return;
    var targets = CORE.bulkApplyTargets(MANIFEST.items, answers, FILTER);
    if (!targets.length) {
      alert("No still-unanswered cards in the current view to apply to.");
      return;
    }
    var name = (BYCODE[code] || {}).name || "";
    if (!confirm("Apply code " + code + (name ? " (" + name + ")" : "") +
                 " to all " + targets.length + " still-unanswered card(s) in the current view?\n\n" +
                 "Already-answered and skipped cards are NOT changed.")) return;
    targets.forEach(function (uid) {
      var a = ansFor(uid);
      a.code = code; a.skipped = false; a.answered = true;
    });
    save(LS_ANS, answers);
    // re-sync any visible cards we touched
    var host = document.getElementById("cards");
    targets.forEach(function (uid) {
      var card = host.querySelector('.card[data-uid="' + cssEsc(uid) + '"]');
      if (card && card._sync) card._sync();
    });
    refreshCounts();
  }
  function cssEsc(s) {
    return String(s).replace(/["\\]/g, "\\$&");
  }

  function buildControls() {
    var bar = document.getElementById("filterbar");
    if (!bar) return;
    bar.innerHTML = "";

    // project dropdown (Phase 4)
    var projects = CORE.distinctProjects(MANIFEST.items);
    var pending = CORE.pendingByProject(MANIFEST.items, answers);
    var projWrap = el("label", "fctl");
    projWrap.appendChild(el("span", "fcap", "Project"));
    var projSel = el("select"); projSel.id = "projectFilter";
    projSel.title = "Show only one project's cards. Counts show still-unanswered cards.";
    var totalPending = MANIFEST.items.filter(function (it) {
      return CORE.isUnanswered(answers[it.uid]);
    }).length;
    projSel.appendChild(new Option("All projects (" + totalPending + " pending)", ""));
    projects.forEach(function (p) {
      projSel.appendChild(new Option(p.label + " (" + (pending[p.key] || 0) + " pending)", p.key));
    });
    projSel.value = FILTER.project;
    projSel.addEventListener("change", function () { FILTER.project = projSel.value; applyFilter(); });
    projWrap.appendChild(projSel);
    bar.appendChild(projWrap);

    // site dropdown (C4)
    var sites = CORE.distinctSites(MANIFEST.items, CODES.sites);
    var siteWrap = el("label", "fctl");
    siteWrap.appendChild(el("span", "fcap", "Site"));
    var siteSel = el("select"); siteSel.id = "siteFilter";
    siteSel.title = "Show only one site's cards (full site name).";
    siteSel.appendChild(new Option("All sites", ""));
    sites.forEach(function (s) {
      siteSel.appendChild(new Option(s.name + (s.code ? "  (" + s.code + ")" : ""), s.code));
    });
    siteSel.value = FILTER.site;
    siteSel.addEventListener("change", function () { FILTER.site = siteSel.value; applyFilter(); });
    siteWrap.appendChild(siteSel);
    bar.appendChild(siteWrap);

    // bulk apply (C4) - code dropdown + apply button
    var bulkWrap = el("label", "fctl bulk");
    bulkWrap.appendChild(el("span", "fcap", "Bulk-apply to view"));
    var bulkSel = el("select"); bulkSel.id = "bulkCode";
    bulkSel.title = "Assign one code to every still-unanswered card in the current view " +
      "(e.g. one species ubiquitous at a site). Answered/skipped cards are untouched.";
    bulkSel.appendChild(new Option("- pick a code -", ""));
    var addOpt = function (code) {
      var info = BYCODE[code] || {};
      bulkSel.appendChild(new Option(code + (info.name ? " - " + info.name : ""), code));
    };
    var featured = MANIFEST.featured_codes || [];
    (CODES.candidate_codes || []).concat(featured).forEach(function (c) {
      if (c) addOpt(c);
    });
    (CODES.codes || []).forEach(function (c) { if (c.code) addOpt(c.code); });
    bulkWrap.appendChild(bulkSel);
    var bulkBtn = el("button", "bulkbtn", "Apply to all in view");
    bulkBtn.title = "Apply the chosen code to all still-unanswered cards currently shown.";
    bulkBtn.addEventListener("click", function () { bulkApply(bulkSel.value); });
    bulkWrap.appendChild(bulkBtn);
    bar.appendChild(bulkWrap);

    var info = el("span", "finfo"); info.id = "filterInfo";
    bar.appendChild(info);
  }

  // ── counts / progress ───────────────────────────────────────────
  function tally() {
    var items = MANIFEST.items.filter(function (it) { return CORE.matchesFilter(filterItem(it), FILTER); });
    var total = items.length, answered = 0, skipped = 0;
    items.forEach(function (it) {
      var a = answers[it.uid]; if (!a) return;
      if (a.skipped) skipped++; else if (a.answered) answered++;
    });
    return { total: total, answered: answered, skipped: skipped };
  }
  // The manifest items already carry site/site_code/project - filterItem is a
  // light shim keeping matchesFilter pure-input shaped.
  function filterItem(it) {
    return { project_id: it.project_id, project_name: it.project_name,
             site: it.site, site_code: it.site_code };
  }
  function refreshCounts() {
    var t = tally();
    var scope = (FILTER.project || FILTER.site) ? " in view" : "";
    document.getElementById("counts").innerHTML =
      "<strong>" + t.total + "</strong> masks to review" + scope;
    document.getElementById("progress").textContent =
      t.answered + " identified · " + t.skipped + " skipped · " +
      (t.total - t.answered - t.skipped) + " remaining" + scope;
    refreshGroupPending();
  }

  // ── CSV export ────────────────────────────────────────────────────
  function exportCSV() {
    if (!REVIEWER) { openNameGate(); return; }   // C1: no labeling/export without a name
    var rows = CORE.buildCsvRows(MANIFEST.items, answers, REVIEWER, FILTER);
    if (rows.length === 1) {
      alert("No IDs to export yet" + (FILTER.project || FILTER.site ? " in the current view" : "") +
            ". Pick a code on at least one card (Skip and untouched cards are not exported).");
      return;
    }
    var csv = CORE.rowsToCsv(rows);
    var name = CORE.csvFilename(REVIEWER, FILTER.project);
    var blob = new Blob([csv], { type: "text/csv" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a"); a.href = url; a.download = name; a.click();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1500);
    showModal(name, rows.length - 1);
  }

  function showModal(csvName, n) {
    var contacts = CORE.recipientList(CODES, MANIFEST);
    document.getElementById("csvName").textContent = csvName;
    var list = document.getElementById("contactList"); list.innerHTML = "";
    var link = document.getElementById("mailtoLink");
    var subject = encodeURIComponent("TCRMP reef expert IDs - " + n + " labels (" + REVIEWER + ")");
    var body = encodeURIComponent(
      "Hi,\n\nAttached is my expert ID CSV (" + csvName + ") with " + n +
      " identifications from the TCRMP reef review.\nReviewer: " + REVIEWER +
      "\n\n(Please remember to attach the downloaded file.)\n");
    if (contacts.length) {
      contacts.forEach(function (c) { list.appendChild(el("li", null, c)); });
      link.href = "mailto:" + contacts.join(",") + "?subject=" + subject + "&body=" + body;
      link.style.display = "";
    } else {
      list.appendChild(el("li", null,
        "(no recipient was configured, please send the CSV to whoever asked you to review)"));
      link.style.display = "none";
    }
    document.getElementById("exportModal").hidden = false;
  }

  // ── reviewer name gate (C1) ───────────────────────────────────────
  function setReviewer(name) {
    REVIEWER = (name || "").trim();
    saveStr(LS_NAME, REVIEWER);
    var pill = document.getElementById("whoami");
    if (pill) pill.textContent = REVIEWER ? ("Reviewer: " + REVIEWER) : "";
  }
  function openNameGate() {
    var gate = document.getElementById("nameGate");
    if (!gate) return;
    var input = document.getElementById("reviewerName");
    input.value = REVIEWER || "";
    gate.hidden = false;
    setTimeout(function () { try { input.focus(); } catch (e) {} }, 0);
  }
  function tryAcceptName() {
    var input = document.getElementById("reviewerName");
    var v = (input.value || "").trim();
    var errEl = document.getElementById("nameErr");
    if (!v) {
      if (errEl) errEl.textContent = "Please enter your name to begin. It is recorded with every ID you submit.";
      try { input.focus(); } catch (e) {}
      return;
    }
    setReviewer(v);
    document.getElementById("nameGate").hidden = true;
  }

  // ── boot ──────────────────────────────────────────────────────────
  function boot() {
    REVIEWER = loadStr(LS_NAME) || "";
    document.getElementById("exportBtn").addEventListener("click", exportCSV);
    document.getElementById("closeModal").addEventListener("click", function () {
      document.getElementById("exportModal").hidden = true;
    });
    var nameOk = document.getElementById("nameOk");
    var nameInput = document.getElementById("reviewerName");
    if (nameOk) nameOk.addEventListener("click", tryAcceptName);
    if (nameInput) nameInput.addEventListener("keydown", function (e) {
      if (e.key === "Enter") { e.preventDefault(); tryAcceptName(); }
    });
    var changeName = document.getElementById("changeName");
    if (changeName) changeName.addEventListener("click", function (e) {
      e.preventDefault(); openNameGate();
    });

    // codes.json is required (the labeling form needs it). The manifest may be
    // missing/404 (e.g. a freshly-deployed empty review repo) - treat that as
    // "nothing to review" rather than a hard error.
    getJSON("codes.json").then(function (codesData) {
      CODES = codesData;
      CODES.sites = CODES.sites || {};
      (CODES.codes || []).forEach(function (c) { BYCODE[c.code] = c; });
      return getJSON("review_manifest.json").catch(function () {
        return { contacts: [], items: [] };
      });
    }).then(function (manifest) {
      MANIFEST = manifest || { contacts: [], items: [] };
      // C2 email recipients into the header instruction.
      paintHeaderEmails();
      setReviewer(REVIEWER);   // refresh pill
      var host = document.getElementById("cards");
      host.innerHTML = "";
      if (!MANIFEST.items || !MANIFEST.items.length) {
        host.appendChild(el("div", "empty",
          "Nothing to review right now. All masks have been identified. Thank you!"));
        refreshCounts();
        if (!REVIEWER) openNameGate();
        return;
      }
      buildControls();
      // grouped render: one sticky header per project, cards sorted within
      CORE.groupByProject(MANIFEST.items).forEach(function (group) {
        host.appendChild(buildGroupHeader(group));
        group.items.forEach(function (it) { host.appendChild(buildCard(it)); });
      });
      syncTopbarOffset();
      if (typeof root.addEventListener === "function")
        root.addEventListener("resize", syncTopbarOffset);
      applyFilter();
      refreshCounts();
      if (!REVIEWER) openNameGate();   // C1: require a name before any labeling
    }).catch(function (e) {
      document.getElementById("cards").appendChild(
        el("div", "empty", "Could not load review data: " + e.message));
    });
  }

  function paintHeaderEmails() {
    var holder = document.getElementById("emailTargets");
    if (!holder) return;
    var contacts = CORE.recipientList(CODES, MANIFEST);
    holder.textContent = contacts.length ? contacts.join(", ")
      : "the operator who sent you this link, and " + CORE.LAUREN;
  }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", boot);
  else boot();
})(typeof self !== "undefined" ? self : this);

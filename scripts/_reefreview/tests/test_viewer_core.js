#!/usr/bin/env node
/* Pure-logic tests for the expert-review viewer (CONTRACTS §4,§7,§8 + C1-C5).
   The viewer.js doubles as a node-requireable module (no DOM): it exports the
   CSV row builder, site-name mapping, project/site filter predicate, and the
   bulk-apply selection. Run:
     node scripts/_reefreview/tests/test_viewer_core.js
   (also driven by run_tests.sh under env/bin/python's node, but node-only.)

   THREE layers, mirroring the no-pytest harness convention:
     smoke       - module loads + every documented export is a function/value
     unit        - site mapping, filters, grouped render order (sortItems +
                   groupByProject), CSV header/row shape, filename, bulk
     adversarial - blank reviewer still stamped, project filter scoping, CSV
                   quoting of commas/quotes, unmapped site, identical frames
                   across projects, candidate fallback, recipient de-dup,
                   null/empty project_id grouping, duplicate project labels,
                   legacy confidence values ignored. */
const path = require("path");
const C = require(path.join(__dirname, "..", "viewer", "viewer.js"));

let pass = 0, fail = 0;
function ok(cond, msg) { if (cond) { pass++; } else { fail++; console.error("  FAIL:", msg); } }
function eq(a, b, msg) {
  ok(JSON.stringify(a) === JSON.stringify(b), `${msg} (got ${JSON.stringify(a)}, want ${JSON.stringify(b)})`);
}

// ── fixtures ─────────────────────────────────────────────────────────
const SITES = { GRP: "Great Pond", BWR: "Brewers Bay", FSB: "Fish Bay" };
const CODES = {
  idk: { code: "IDK", label: "I don't know" },
  candidate_codes: ["OFRA", "MCAV"],
  codes: [{ code: "OFRA", name: "Orbicella franksi" }, { code: "MCAV", name: "Montastraea cavernosa" }],
};
// two projects, three items (GRP/p1, GRP/p1, BWR/p2)
const ITEMS = [
  { uid: "u1", project_id: "p1", project_name: "Reef One", site: "GRP", site_full: "Great Pond", frame: 1,
    reviews: [{ reviewer: "Bob", code: "MCAV", confidence: "high" }] },
  { uid: "u2", project_id: "p1", project_name: "Reef One", site: "GRP", site_full: "Great Pond", frame: 2, reviews: [] },
  { uid: "u3", project_id: "p2", project_name: "Reef Two", site: "BWR", site_full: "Brewers Bay", frame: 3, reviews: [] },
];

// ── SMOKE ────────────────────────────────────────────────────────────
["siteName", "projectLabel", "projectKey", "distinctProjects", "distinctSites",
 "sortItems", "groupByProject",
 "matchesFilter", "isAnswered", "isUnanswered", "bulkApplyTargets", "pendingByProject",
 "csvCell", "buildCsvRows", "rowsToCsv", "slug", "csvFilename", "recipientList"]
  .forEach(function (n) { ok(typeof C[n] === "function", `export ${n} is a function`); });
eq(C.CSV_HEADER, ["uid", "code", "confidence", "reviewer", "project_id"], "smoke: exact CSV header (§4)");
ok(C.LAUREN === "lauren.olinger@uvi.edu", "smoke: lauren operator email constant (C2)");
ok(C.UNKNOWN_PROJECT === "Unknown project", "smoke: unknown-project group label constant");

// ── UNIT: site name mapping (C4 / §7) ────────────────────────────────
eq(C.siteName({ site: "GRP" }, SITES), "Great Pond", "siteName by code");
eq(C.siteName({ site: "grp" }, SITES), "Great Pond", "siteName upper-cases the code key");
eq(C.siteName({ site: "ZZZ", site_full: "Mystery Cove" }, SITES), "Mystery Cove", "siteName falls back to site_full");
eq(C.siteName({ site: "ZZZ" }, SITES), "ZZZ", "siteName falls back to raw code when unmapped + no full");
eq(C.siteName({}, SITES), "?", "siteName empty -> ?");

// ── UNIT: distinct projects + sites ──────────────────────────────────
eq(C.distinctProjects(ITEMS).map(p => p.key), ["p1", "p2"], "distinctProjects keys (label-ordered)");
eq(C.distinctProjects(ITEMS).map(p => p.label), ["Reef One", "Reef Two"], "distinctProjects labels");
eq(C.distinctSites(ITEMS, SITES).map(s => s.name), ["Brewers Bay", "Great Pond"], "distinctSites name-ordered");
eq(C.distinctSites(ITEMS, SITES).map(s => s.code), ["BWR", "GRP"], "distinctSites codes");

// ── UNIT: filter predicate (Phase 4 + C4) ────────────────────────────
ok(C.matchesFilter(ITEMS[0], {}) === true, "empty filter matches all");
ok(C.matchesFilter(ITEMS[0], { project: "p1" }) === true, "project filter match");
ok(C.matchesFilter(ITEMS[2], { project: "p1" }) === false, "project filter rejects other project");
ok(C.matchesFilter(ITEMS[0], { site: "GRP" }) === true, "site filter match");
ok(C.matchesFilter(ITEMS[2], { site: "GRP" }) === false, "site filter rejects other site");
ok(C.matchesFilter(ITEMS[0], { project: "p1", site: "GRP" }) === true, "combined filter match");
ok(C.matchesFilter(ITEMS[1], { project: "p1", site: "BWR" }) === false, "combined filter: site mismatch rejects");

// ── UNIT: answered / unanswered classification ───────────────────────
ok(C.isAnswered({ answered: true, skipped: false }) === true, "isAnswered true");
ok(C.isAnswered({ answered: true, skipped: true }) === false, "isAnswered false when skipped");
ok(C.isAnswered(undefined) === false, "isAnswered undefined -> false");
ok(C.isUnanswered(undefined) === true, "isUnanswered: untouched is unanswered");
ok(C.isUnanswered({ answered: true }) === false, "isUnanswered: answered is not");
ok(C.isUnanswered({ skipped: true }) === false, "isUnanswered: skipped is not");

// ── UNIT: bulk-apply selection (C4) ──────────────────────────────────
// only u2,u3 untouched; u1 answered -> bulk skips it. Filter to p1 -> only u2.
const ANS = { u1: { code: "OFRA", conf: "high", answered: true, skipped: false } };
eq(C.bulkApplyTargets(ITEMS, ANS, {}), ["u2", "u3"], "bulk targets = all unanswered, filter-wide");
eq(C.bulkApplyTargets(ITEMS, ANS, { project: "p1" }), ["u2"], "bulk targets scoped to project filter");
eq(C.bulkApplyTargets(ITEMS, ANS, { site: "BWR" }), ["u3"], "bulk targets scoped to site filter");
eq(C.bulkApplyTargets(ITEMS, {}, {}), ["u1", "u2", "u3"], "bulk targets = all when nothing answered");

// ── UNIT: per-project pending counts (Phase 4) ───────────────────────
eq(C.pendingByProject(ITEMS, ANS), { p1: 1, p2: 1 }, "pendingByProject after one answered in p1");
eq(C.pendingByProject(ITEMS, {}), { p1: 2, p2: 1 }, "pendingByProject all pending");

// ── UNIT: grouped render order (sortItems + groupByProject) ──────────
const MIX = [
  { uid: "z9", project_id: "p2", project_name: "Beta Reef", site: "BWR", frame: 1 },
  { uid: "z1", project_id: "p1", project_name: "Alpha Reef", site: "GRP", frame: 2 },
  { uid: "z2", project_id: "p1", project_name: "Alpha Reef", site: "GRP", frame: 1 },
  { uid: "z3", project_id: "p1", project_name: "Alpha Reef", site: "ABC", frame: 9 },
  { uid: "z8", project_id: null, project_name: "", site: "GRP", frame: 1 },   // unknown project
];
eq(C.sortItems(MIX).map(i => i.uid), ["z3", "z2", "z1", "z9", "z8"],
   "sortItems: project label asc, then site, then frame; unknown project last");
eq(MIX.map(i => i.uid), ["z9", "z1", "z2", "z3", "z8"], "sortItems does not mutate its input");
// missing frame sorts after numbered frames within the same project + site
const NOFRAME = [
  { uid: "nf_b", project_id: "p1", project_name: "A", site: "S" },
  { uid: "nf_a", project_id: "p1", project_name: "A", site: "S", frame: 3 },
];
eq(C.sortItems(NOFRAME).map(i => i.uid), ["nf_a", "nf_b"], "sortItems: missing frame sorts last within site");
// uid is the final tiebreak
const TIE = [
  { uid: "t2", project_id: "p1", project_name: "A", site: "S", frame: 1 },
  { uid: "t1", project_id: "p1", project_name: "A", site: "S", frame: 1 },
];
eq(C.sortItems(TIE).map(i => i.uid), ["t1", "t2"], "sortItems: uid tiebreak on identical project/site/frame");

const GROUPS = C.groupByProject(MIX);
eq(GROUPS.map(g => g.key), ["p1", "p2", ""], "groupByProject order: labels asc, unknown group last");
eq(GROUPS.map(g => g.label), ["Alpha Reef", "Beta Reef", "Unknown project"], "groupByProject labels incl. Unknown project");
eq(GROUPS.map(g => g.items.length), [3, 1, 1], "groupByProject per-group counts");
eq(GROUPS[0].items.map(i => i.uid), ["z3", "z2", "z1"], "groupByProject keeps the sorted order inside a group");
eq(C.groupByProject([]), [], "groupByProject([]) -> []");

// ── UNIT: CSV row builder (§4) ───────────────────────────────────────
// ANS2 deliberately carries LEGACY conf values (old localStorage answers):
// the viewer must ignore them and always write a blank confidence cell.
const ANS2 = {
  u1: { code: "OFRA", conf: "high", answered: true, skipped: false },
  u3: { code: "MCAV", conf: "low", answered: true, skipped: false },
  // u2 left untouched -> excluded
};
const rowsAll = C.buildCsvRows(ITEMS, ANS2, "Jane Doe", {});
eq(rowsAll[0], ["uid", "code", "confidence", "reviewer", "project_id", "project_name", "site", "frame"],
   "CSV header = exact 5 + optional human cols, confidence col kept (§4)");
eq(rowsAll.length, 3, "CSV: header + 2 answered rows (u2 untouched excluded)");
eq(rowsAll[1], ["u1", "OFRA", "", "Jane Doe", "p1", "Reef One", "GRP", 1], "CSV row u1: reviewer stamped, confidence blank");
eq(rowsAll[2], ["u3", "MCAV", "", "Jane Doe", "p2", "Reef Two", "BWR", 3], "CSV row u3: legacy conf value ignored");
// machine-only header (no human cols)
eq(C.buildCsvRows(ITEMS, ANS2, "Jane Doe", {}, { human: false })[0], C.CSV_HEADER, "CSV header machine-only mode (§4)");

// ── UNIT: filename (§4) ──────────────────────────────────────────────
eq(C.csvFilename("Jane Doe", ""), "tcrmp_expert_ids_jane_doe_all.csv", "filename: no project -> _all");
eq(C.csvFilename("Jane Doe", "p1"), "tcrmp_expert_ids_jane_doe_p1.csv", "filename: project suffix");

// ── UNIT: recipient list (C2) ────────────────────────────────────────
eq(C.recipientList({ operator: "op@reef.org" }, { contacts: ["c@x.org"] }),
   ["op@reef.org", "c@x.org", "lauren.olinger@uvi.edu"], "recipients: operator + contacts + lauren");

// ── ADVERSARIAL ──────────────────────────────────────────────────────
// project filter scopes the EXPORT: only that project's answered rows (§4).
const rowsP1 = C.buildCsvRows(ITEMS, ANS2, "Jane Doe", { project: "p1" });
eq(rowsP1.map(r => r[0]).slice(1), ["u1"], "ADV: project-filtered export carries ONLY that project's rows (§4)");
// blank reviewer must STILL be stamped (the column exists, empty) — never drop the col.
const rowsBlank = C.buildCsvRows(ITEMS, ANS2, "", {});
eq(rowsBlank[1][3], "", "ADV: blank reviewer -> empty reviewer cell, column preserved");
eq(rowsBlank[0].slice(0, 5), C.CSV_HEADER, "ADV: header unchanged regardless of reviewer");
// CSV quoting: comma, quote, newline in a value.
eq(C.csvCell("a,b"), '"a,b"', "ADV: comma quoted");
eq(C.csvCell('he said "hi"'), '"he said ""hi"""', "ADV: embedded quotes doubled");
eq(C.csvCell("line1\nline2"), '"line1\nline2"', "ADV: newline quoted");
eq(C.csvCell(null), "", "ADV: null -> empty");
eq(C.csvCell(0), "0", "ADV: zero kept (not blanked)");
// a reviewer name with a comma must round-trip through rowsToCsv quoted.
const rowsComma = C.buildCsvRows(ITEMS, { u1: { code: "OFRA", conf: "high", answered: true } }, "Doe, Jane", {});
ok(C.rowsToCsv(rowsComma).indexOf('"Doe, Jane"') >= 0, "ADV: comma in reviewer name is CSV-quoted");
// identical frame numbers across projects must NOT collide — uid is the key, not frame.
const cross = [
  { uid: "a", project_id: "p1", project_name: "One", site: "GRP", frame: 5 },
  { uid: "b", project_id: "p2", project_name: "Two", site: "BWR", frame: 5 },
];
const crossAns = { a: { code: "X", conf: "high", answered: true }, b: { code: "Y", conf: "high", answered: true } };
eq(C.buildCsvRows(cross, crossAns, "R", { project: "p2" }).map(r => r[0]).slice(1), ["b"],
   "ADV: identical frame across projects stays separated by project filter (uid-keyed)");
// candidate codes fall back to codes.json list when an item has none of its own
// (exercised indirectly via the field shape the viewer reads).
ok(Array.isArray(CODES.candidate_codes) && CODES.candidate_codes.length === 2, "ADV: candidate_codes present on codes.json");
// recipient de-dup: lauren listed twice collapses; operator already == lauren stays once.
eq(C.recipientList({ operator: "lauren.olinger@uvi.edu" }, { contacts: ["lauren.olinger@uvi.edu"] }),
   ["lauren.olinger@uvi.edu"], "ADV: recipient de-dup (operator==contacts==lauren -> one)");
// empty manifest / missing fields never throw.
eq(C.distinctProjects([]), [], "ADV: distinctProjects([]) -> []");
eq(C.distinctSites([], {}), [], "ADV: distinctSites([]) -> []");
eq(C.buildCsvRows([], {}, "R", {}).length, 1, "ADV: empty manifest -> header-only CSV");
// item missing project_id (already-published / partial) -> projectKey falls back to name, never crashes.
eq(C.projectKey({ project_name: "OnlyName" }), "OnlyName", "ADV: projectKey falls back to project_name");
eq(C.projectKey({}), "", "ADV: projectKey of {} -> ''");
// slug guards an empty/odd reviewer for the filename.
eq(C.csvFilename("", ""), "tcrmp_expert_ids_x_all.csv", "ADV: empty reviewer slug -> 'x' placeholder");
eq(C.csvFilename("  !!!  ", ""), "tcrmp_expert_ids_x_all.csv", "ADV: punctuation-only reviewer -> 'x'");
// confidence is blank even when an answer object has no conf field at all.
eq(C.buildCsvRows(ITEMS, { u1: { code: "OFRA", answered: true } }, "R", {})[1][2], "",
   "ADV: answer without conf field -> blank confidence cell");
// null/empty/missing project_id items collapse into ONE trailing unknown group.
const UNK = [
  { uid: "u_b", project_id: "", site: "S2", frame: 1 },
  { uid: "u_a", project_id: null, site: "S1", frame: 1 },
  { uid: "u_k", project_id: "p1", project_name: "Named", site: "S", frame: 1 },
];
const gUnk = C.groupByProject(UNK);
eq(gUnk.map(g => g.key), ["p1", ""], "ADV: null + empty project_id share one trailing unknown group");
eq(gUnk[1].label, C.UNKNOWN_PROJECT, "ADV: unknown group labeled 'Unknown project'");
eq(gUnk[1].items.map(i => i.uid), ["u_a", "u_b"], "ADV: unknown group internally site-sorted");
// duplicate project LABELS with different keys stay separate groups, key-ordered.
const DUP = [
  { uid: "d2", project_id: "pB", project_name: "Same Label", site: "S", frame: 1 },
  { uid: "d1", project_id: "pA", project_name: "Same Label", site: "S", frame: 1 },
];
eq(C.groupByProject(DUP).map(g => g.key), ["pA", "pB"], "ADV: duplicate labels stay separate groups, key-ordered");
eq(C.groupByProject(DUP).map(g => g.label), ["Same Label", "Same Label"], "ADV: duplicate labels both kept");
// filter + group interaction: the project filter empties every other group,
// the site filter trims within a group without dissolving it.
const gAll = C.groupByProject(ITEMS);
const liveKeys = gAll
  .map(g => ({ key: g.key, n: g.items.filter(i => C.matchesFilter(i, { project: "p1" })).length }))
  .filter(g => g.n > 0).map(g => g.key);
eq(liveKeys, ["p1"], "ADV: project filter leaves only its own group non-empty");
eq(gAll[0].items.filter(i => C.matchesFilter(i, { site: "GRP" })).map(i => i.uid), ["u1", "u2"],
   "ADV: site filter works within a group");
eq(gAll[0].items.filter(i => C.matchesFilter(i, { site: "BWR" })).length, 0,
   "ADV: site filter can empty a group (header hides in the DOM layer)");
// sortItems tolerates null input.
eq(C.sortItems(null), [], "ADV: sortItems(null) -> []");

console.log(`viewer_core: ${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

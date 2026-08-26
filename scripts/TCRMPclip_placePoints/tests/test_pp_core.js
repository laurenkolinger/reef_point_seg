#!/usr/bin/env node
// Unit + adversarial tests for src/static/pp_core.js — the pure label / sort /
// pick-radius helpers used by the Place Points UI. Run: node tests/test_pp_core.js
const C = require('../src/static/pp_core.js');

let pass = 0, fail = 0;
function ok(cond, msg) { if (cond) { pass++; } else { fail++; console.error('  FAIL:', msg); } }
function eq(a, b, msg) { ok(JSON.stringify(a) === JSON.stringify(b), `${msg} (got ${JSON.stringify(a)}, want ${JSON.stringify(b)})`); }

// ── colLabel: spreadsheet/bijective base-26 ──
eq(C.colLabel(1), 'A', 'colLabel 1');
eq(C.colLabel(26), 'Z', 'colLabel 26');
eq(C.colLabel(27), 'AA', 'colLabel 27');
eq(C.colLabel(28), 'AB', 'colLabel 28');
eq(C.colLabel(52), 'AZ', 'colLabel 52');
eq(C.colLabel(53), 'BA', 'colLabel 53 -> BA (the case the user named)');
eq(C.colLabel(54), 'BB', 'colLabel 54 -> BB');
eq(C.colLabel(55), 'BC', 'colLabel 55 -> BC');
eq(C.colLabel(702), 'ZZ', 'colLabel 702 -> ZZ');
eq(C.colLabel(703), 'AAA', 'colLabel 703 -> AAA (past two letters)');
eq(C.colLabel(18278), 'ZZZ', 'colLabel 18278 -> ZZZ');

// ── nextLabel: first unused, infinite supply ──
eq(C.nextLabel([]), 'A', 'nextLabel empty -> A');
eq(C.nextLabel(['A', 'B']), 'C', 'nextLabel A,B -> C');
eq(C.nextLabel(new Set(['A', 'C'])), 'B', 'nextLabel gap (Set) -> B');
eq(C.nextLabel(['B', 'A', 'D', 'C']), 'E', 'nextLabel unordered A-D -> E');
// all single letters used -> rolls to AA, then BA past AZ
const A_Z = []; for (let i = 1; i <= 26; i++) A_Z.push(C.colLabel(i));
eq(C.nextLabel(A_Z), 'AA', 'nextLabel A..Z used -> AA');
const A_AZ = []; for (let i = 1; i <= 52; i++) A_AZ.push(C.colLabel(i));
eq(C.nextLabel(A_AZ), 'BA', 'nextLabel A..AZ used -> BA (the exact ask)');
// Adversarial: 1000 used labels still returns the 1001st fast (no hang)
const big = []; for (let i = 1; i <= 1000; i++) big.push(C.colLabel(i));
eq(C.nextLabel(big), C.colLabel(1001), 'nextLabel 1000 used -> #1001');
// Adversarial: junk/unknown labels in the set never match the sequence -> 'A'
eq(C.nextLabel(['', '??', '123', 'aa']), 'A', 'nextLabel junk labels -> A (case-sensitive, uppercase only)');

// ── labelOrdinal: inverse of colLabel (0 for non-conforming) ──
eq(C.labelOrdinal('A'), 1, 'labelOrdinal A -> 1');
eq(C.labelOrdinal('Z'), 26, 'labelOrdinal Z -> 26');
eq(C.labelOrdinal('AA'), 27, 'labelOrdinal AA -> 27');
eq(C.labelOrdinal('BA'), 53, 'labelOrdinal BA -> 53');
eq(C.labelOrdinal('ZZ'), 702, 'labelOrdinal ZZ -> 702');
eq(C.labelOrdinal(''), 0, 'labelOrdinal empty -> 0');
eq(C.labelOrdinal('a'), 0, 'labelOrdinal lowercase -> 0');
eq(C.labelOrdinal('A1'), 0, 'labelOrdinal with digit -> 0');
// labelOrdinal is the exact inverse of colLabel across the range
let ordInv = true; for (let i = 1; i <= 2000; i++) if (C.labelOrdinal(C.colLabel(i)) !== i) ordInv = false;
ok(ordInv, 'labelOrdinal(colLabel(n)) === n for 1..2000');

// ── nextLabelMonotonic: AFTER the max used label (freed letters NOT reused) ──
eq(C.nextLabelMonotonic([]), 'A', 'monotonic empty -> A');
eq(C.nextLabelMonotonic(['A', 'B', 'C']), 'D', 'monotonic A,B,C -> D');
eq(C.nextLabelMonotonic(['A', 'B', 'C', 'D']), 'E', 'monotonic A..D -> E');
// the headline case: D deleted (A,B,C,E present) -> next is F, NOT D
eq(C.nextLabelMonotonic(['A', 'B', 'C', 'E']), 'F', 'monotonic past max ignores freed D -> F');
eq(C.nextLabelMonotonic(['A', 'Z']), 'AA', 'monotonic max Z -> AA (rollover)');
eq(C.nextLabelMonotonic(new Set(['B', 'A', 'D', 'C'])), 'E', 'monotonic unordered Set -> E');
eq(C.nextLabelMonotonic(['AZ', 'B']), 'BA', 'monotonic max AZ -> BA');
// junk labels never raise the max; only A,B count -> C
eq(C.nextLabelMonotonic(['', '??', 'a', 'A', 'B']), 'C', 'monotonic junk ignored -> C');

// ── compareLabelsDesc: DESCENDING alphabetical ──
eq(C.compareLabelsDesc('A', 'A'), 0, 'compare equal -> 0');
ok(C.compareLabelsDesc('A', 'B') > 0, 'A after B in desc');
ok(C.compareLabelsDesc('B', 'A') < 0, 'B before A in desc');

// ── compareLabelsAsc / panelSortIndices: natural label-ORDINAL order ──
eq(C.compareLabelsAsc('A', 'A'), 0, 'compareAsc equal -> 0');
ok(C.compareLabelsAsc('A', 'B') < 0, 'A before B in asc');
ok(C.compareLabelsAsc('B', 'A') > 0, 'B after A in asc');
// panelSortIndices sorts by label ORDINAL (A,B,..,Z,AA,..) not raw string, so
// the monotonic next label 'AA' lands at the BOTTOM (after Z), never between A
// and B. This is the "new labels append to the bottom" guarantee.
const pts = [{ label: 'A' }, { label: 'Z' }, { label: 'AA' }, { label: 'B' }];
eq(C.panelSortIndices(pts, () => false).map(i => pts[i].label), ['A', 'B', 'Z', 'AA'], 'panel ORDINAL order (A, B, Z, AA — AA last)');
// hidden predicate removes a point
const idxH = C.panelSortIndices(pts, p => p.label === 'Z');
ok(idxH.every(i => pts[i].label !== 'Z'), 'hidden Z removed from panel');
eq(idxH.map(i => pts[i].label), ['A', 'B', 'AA'], 'panel ordinal order after hide (AA still last)');
// after 26 points, the next monotonic label 'AA' must sort LAST, not 2nd
const pts27 = []; for (let i = 1; i <= 26; i++) pts27.push({ label: C.colLabel(i) }); pts27.push({ label: 'AA' });
const ord27 = C.panelSortIndices(pts27, () => false).map(i => pts27[i].label);
eq(ord27[ord27.length - 1], 'AA', 'panel: AA appended after Z lands LAST');
eq(ord27.slice(0, 3), ['A', 'B', 'C'], 'panel: single letters stay alphabetical');
// non-conforming labels (ordinal 0 — e.g. lowercase) sort ahead of real
// A..Z/AA.. labels; among themselves they fall back to a string tiebreak.
const ptsJunk = [{ label: 'B' }, { label: 'b' }, { label: 'A' }];
eq(C.panelSortIndices(ptsJunk, () => false).map(i => ptsJunk[i].label), ['b', 'A', 'B'], 'panel: ordinal-0 junk first, then ordinal order');
// stable on empty + all-hidden
eq(C.panelSortIndices([], () => false), [], 'panel empty -> []');
eq(C.panelSortIndices(pts, () => true), [], 'panel all-hidden -> []');

// ── quickKeyAction: which action a quick-add number/REVIEW key tap fires ──
// The tap is overloaded: it can arm/switch the sticky quick-add label OR relabel
// the currently selected point. An ARMED sticky label takes priority over the
// selection so quick-add DROP mode switches labels for the NEXT click instead of
// reaching back to relabel the point you just dropped (which is auto-selected).
eq(C.quickKeyAction(null, -1, 0), 'set', 'qk: nothing armed/selected -> arm sticky');
eq(C.quickKeyAction(null, 3, 0), 'relabel', 'qk: point selected, no sticky armed -> relabel it');
eq(C.quickKeyAction(0, -1, 1), 'set', 'qk: sticky 0 armed, tap 1 -> switch sticky');
eq(C.quickKeyAction(0, -1, 0), 'clear', 'qk: tap the armed sticky again -> clear it');
// the headline bug: dropping a run of 1s auto-selects each placed point; tapping
// 2 must SWITCH the sticky label for the NEXT click, NEVER relabel the previous.
eq(C.quickKeyAction(0, 7, 1), 'set', 'qk: sticky armed + auto-selected point, tap different -> switch sticky (NOT relabel)');
eq(C.quickKeyAction(0, 7, 0), 'clear', 'qk: sticky armed + auto-selected point, tap same -> clear (NOT relabel)');

// ── pickRadiusImg: constant screen px regardless of zoom ──
eq(C.pickRadiusImg(1), 45, 'radius @1x -> 45 img px (= 45 screen px)');
eq(C.pickRadiusImg(2), 22.5, 'radius @2x -> 22.5 img px (still 45 screen px)');
eq(C.pickRadiusImg(0.5), 90, 'radius @0.5x -> 90 img px (still 45 screen px)');
eq(C.pickRadiusImg(1, 60), 60, 'radius custom screenPx');
ok(C.SELECT_RADIUS_PX === 45, 'SELECT_RADIUS_PX constant = 45');

console.log(`pp_core: ${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

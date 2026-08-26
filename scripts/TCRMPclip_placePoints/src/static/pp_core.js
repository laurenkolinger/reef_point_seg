// pp_core.js — pure helpers for TCRMP Place Points.
// Loaded in the browser (attached to window.PPCore) AND requirable in Node
// unit tests (module.exports). NO DOM / app-state access — keep it pure so it
// stays unit-testable. The inline app code in templates/index.html calls these.
(function (root, factory) {
  var api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.PPCore = api;
})(typeof self !== 'undefined' ? self : this, function () {

  // Spreadsheet-style bijective base-26 label: 1->A, 26->Z, 27->AA, 28->AB,
  // 52->AZ, 53->BA ... unbounded. n must be an integer >= 1.
  function colLabel(n) {
    var s = '';
    n = Math.floor(n);
    while (n > 0) {
      var r = (n - 1) % 26;
      s = String.fromCharCode(65 + r) + s;
      n = Math.floor((n - 1) / 26);
    }
    return s;
  }

  // First spreadsheet label not already taken by an existing point. `used` is
  // any iterable of label strings (Array or Set). Always terminates: there is
  // an infinite supply of labels, so the first gap is found quickly.
  function nextLabel(used) {
    var taken = (used instanceof Set) ? used : new Set(used || []);
    var n = 1;
    // Safety cap far beyond any real frame (26^4 = 456976) so a pathological
    // `used` set can never hang the UI; colLabel stays correct past it anyway.
    while (n < 500000) {
      var l = colLabel(n);
      if (!taken.has(l)) return l;
      n++;
    }
    return colLabel(n);
  }

  // Bijective base-26 ordinal of a spreadsheet label: inverse of colLabel.
  // 'A'->1, 'Z'->26, 'AA'->27, 'BA'->53 ... Non-conforming strings (lowercase,
  // empty, with digits/symbols) return 0 so they never raise the running max.
  function labelOrdinal(label) {
    var s = String(label);
    if (!/^[A-Z]+$/.test(s)) return 0;
    var n = 0;
    for (var i = 0; i < s.length; i++) {
      n = n * 26 + (s.charCodeAt(i) - 64);   // 'A'==65 -> 1
    }
    return n;
  }

  // MONOTONIC next label: the label AFTER the current maximum used label, so
  // freed letters (e.g. a deleted 'D') are never reused — new points always
  // append past the highest existing label, keeping them at the bottom of the
  // alphabetical pane. Empty `used` starts at 'A'. Multi-letter rollover is
  // inherited from colLabel (…Z -> AA -> AB …).
  function nextLabelMonotonic(used) {
    var iter = (used instanceof Set) ? Array.from(used) : (used || []);
    var maxOrd = 0;
    for (var i = 0; i < iter.length; i++) {
      var o = labelOrdinal(iter[i]);
      if (o > maxOrd) maxOrd = o;
    }
    return colLabel(maxOrd + 1);
  }

  // Compare two label strings for DESCENDING alphabetical order (Z..A, then
  // multi-letter). Uppercased so case never flips the order.
  function compareLabelsDesc(a, b) {
    a = String(a).toUpperCase();
    b = String(b).toUpperCase();
    if (a === b) return 0;
    return a < b ? 1 : -1;
  }

  // Compare two label strings for ASCENDING alphabetical order (A..Z, then
  // multi-letter). Uppercased so case never flips the order.
  function compareLabelsAsc(a, b) {
    a = String(a).toUpperCase();
    b = String(b).toUpperCase();
    if (a === b) return 0;
    return a < b ? -1 : 1;
  }

  // Indices of `points` to show in the Labels panel: hidden ones removed, the
  // rest sorted by bijective base-26 ORDINAL of `.label` (natural order
  // A,B,…,Z,AA,AB,…) — NOT raw string order. This keeps single letters
  // alphabetical AND, because nextLabelMonotonic hands out the label past the
  // current max ordinal, new points always sort LAST (e.g. 'AA' lands after
  // 'Z', never between 'A' and 'B' as plain string sort would put it).
  // Non-conforming labels (ordinal 0: lowercase/empty/with digits) fall back to
  // a string tiebreak so they stay stable rather than collapsing together.
  // `isHidden(pt, i)` is an app-supplied predicate (skipped species,
  // reference-only points, ...).
  function panelSortIndices(points, isHidden) {
    isHidden = isHidden || function () { return false; };
    var idx = [];
    for (var i = 0; i < points.length; i++) {
      if (!isHidden(points[i], i)) idx.push(i);
    }
    idx.sort(function (a, b) {
      var oa = labelOrdinal(points[a].label);
      var ob = labelOrdinal(points[b].label);
      if (oa !== ob) return oa - ob;
      return compareLabelsAsc(points[a].label, points[b].label);
    });
    return idx;
  }

  // Decide what a quick-add number/REVIEW key tap does. The tap is overloaded —
  // it can arm/switch the sticky quick-add label OR relabel the selected point —
  // and getting the priority wrong caused a "relabels the wrong point" bug:
  //   * armedIdx !== null  -> a STICKY label is armed (quick-add DROP mode). A
  //        tap SWITCHES the sticky label ('set'), or CLEARS it ('clear') when you
  //        tap the one already armed. It must NEVER relabel: each point you drop
  //        is auto-selected, so relabeling here changed the PREVIOUS point instead
  //        of arming the label for the next click.
  //   * armedIdx === null  -> not in quick-add. With a point selected the tap
  //        relabels it ('relabel'); with nothing selected it arms the sticky
  //        label ('set').
  // Returns 'set' | 'clear' | 'relabel'. `armedIdx` is numKeyDown (null when no
  // sticky label is armed); `selectedPoint` is -1 when nothing is selected.
  function quickKeyAction(armedIdx, selectedPoint, idx) {
    if (armedIdx !== null && armedIdx !== undefined) {
      return armedIdx === idx ? 'clear' : 'set';
    }
    return (selectedPoint != null && selectedPoint >= 0) ? 'relabel' : 'set';
  }

  // Image-space pick radius for selecting a point — a CONSTANT screen-pixel
  // grab distance regardless of zoom (screenPx / scale). A generous default so
  // a click "near enough" to a point still selects it.
  var SELECT_RADIUS_PX = 45;
  function pickRadiusImg(scale, screenPx) {
    var px = (screenPx == null) ? SELECT_RADIUS_PX : screenPx;
    return px / scale;
  }

  return {
    colLabel: colLabel,
    nextLabel: nextLabel,
    nextLabelMonotonic: nextLabelMonotonic,
    labelOrdinal: labelOrdinal,
    compareLabelsDesc: compareLabelsDesc,
    compareLabelsAsc: compareLabelsAsc,
    panelSortIndices: panelSortIndices,
    quickKeyAction: quickKeyAction,
    pickRadiusImg: pickRadiusImg,
    SELECT_RADIUS_PX: SELECT_RADIUS_PX,
  };
});

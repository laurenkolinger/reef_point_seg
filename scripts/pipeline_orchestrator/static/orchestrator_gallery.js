/* VEGA inference gallery card (Step 7: Model Inference results block).
 *
 * Reads GET /api/step/8/runs and GET /api/step/8/manifest?out_dir=<abs> and
 * renders a run picker, a header of counts and thresholds, and one tile per
 * image with the overlay, the filename, one chip per detection (full species
 * name, code beside it, confidence to two decimals) and the yes/no/skip
 * verdict buttons. A verdict press dispatches a bubbling CustomEvent named
 * "vega:verdict" with detail {out_dir, filename, verdict} on the card root;
 * the host page posts it.
 *
 * Pure render functions (data in, HTML string out) are kept apart from the
 * DOM glue so Node 18 can test them. Every string that comes from a manifest
 * is HTML-escaped and every URL part is encodeURIComponent-ed: filenames come
 * from disk and manifest content is data, never markup.
 *
 * Exposed as window.VegaGallery in the browser and as module.exports in Node.
 */
(function (globalScope) {
  'use strict';

  // ---------------------------------------------------------------- constants

  /** Root class the stylesheet is scoped under; also the fragment's class. */
  const ROOT_CLASS = 'vega-gallery';
  /** Element ids the fragment declares and the glue fills. */
  const IDS = Object.freeze({
    run: 's8-gallery-run',
    refresh: 's8-gallery-refresh',
    status: 's8-gallery-status',
    header: 's8-gallery-header',
    tiles: 's8-gallery-tiles',
  });
  const RUNS_URL = '/api/step/8/runs';
  const MANIFEST_URL = '/api/step/8/manifest';
  const IMAGE_URL = '/api/step/8/image';
  /** Event the card dispatches on its root when a verdict button is pressed. */
  const VERDICT_EVENT = 'vega:verdict';
  /** Share of images with no detection above which the blind-spot sentence
   *  appears; the same constant as BLIND_SPOT_NO_DETECTION_FRACTION in
   *  scripts/TCRMPtrain_oceankindCV/src/run_inference.py, strict greater-than. */
  const BLIND_SPOT_NO_DETECTION_FRACTION = 0.2;
  const CONFIDENCE_DECIMALS = 2;
  const THRESHOLD_DECIMALS = 2;
  const PERCENT = 100;
  /** A class name that is a bare species code: capitals and digits, short. */
  const BARE_CODE_PATTERN = /^[A-Z0-9]{1,8}$/;
  /** A full name that reads as a species binomial: Genus species, with up to one more epithet. */
  const BINOMIAL_PATTERN = /^[A-Z][a-z]+(?: [a-z-]+){1,2}$/;
  /** A class name written as a binomial with separators (Acropora-cervicornis, Porites_astreoides):
   *  the rule run_inference.py applies to the overlays (BINOMIAL_TOKEN_RE and BINOMIAL_SEPARATOR_RE). */
  const BINOMIAL_TOKEN_PATTERN = /^[A-Z][a-z]+(?:[ _-][a-z]+){1,2}$/;
  const BINOMIAL_SEPARATOR_PATTERN = /[_-]/g;
  /** A tile index as written in data-index: decimal digits only. */
  const INDEX_PATTERN = /^\d+$/;
  /** ISO timestamp prefix: date, a T or space, hours and minutes. */
  const ISO_PREFIX_PATTERN = /^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})/;
  /** C0 control characters and DEL never reach the screen. */
  const CONTROL_CHARS = /[\x00-\x1f\x7f]/g;
  const VERDICTS = Object.freeze(['yes', 'no', 'skip']);
  const VERDICT_LABELS = Object.freeze({ yes: 'Yes', no: 'No', skip: 'Skip' });
  const VERDICT_TOOLTIPS = Object.freeze({
    yes: 'Mark this image as one the model got right. VEGA appends a yes verdict for this image to verifications.csv in the run folder and leaves every mask as it is.',
    no: 'Mark this image as one the model got wrong. VEGA appends a no verdict for this image to verifications.csv in the run folder and leaves every mask as it is.',
    skip: 'Mark this image as one you cannot judge either way. VEGA records a skip verdict, so the image stays unjudged.',
  });
  const TOGGLE_IMAGE_TOOLTIP = 'Switch between the overlay with the model\'s labelled detections and the raw frame.';
  /** The file that lists every detection whatever the overlay could draw. */
  const PREDICTIONS_FILENAME = 'predictions.json';
  const OVERLAP_TOOLTIP = 'VEGA writes every detection to predictions.json in the run folder, including the ones whose label the overlay could not place clear of another.';
  const CODE_ONLY_TOOLTIP_PREFIX = 'The full name for ';
  const CODE_ONLY_TOOLTIP_SUFFIX = ' is not on file.';
  /** Plain words for the skipped-detection reason keys run_inference.py writes. */
  const SKIP_REASON_TEXT = Object.freeze({
    no_geometry: 'with no box or outline geometry',
    no_confidence: 'with no confidence score',
  });
  const EMPTY_TEXT = 'No inference run exists yet, so the gallery has nothing to show.';
  const LOAD_ERROR_PREFIX = 'The gallery could not load: ';
  const RUN_ERROR_PREFIX = 'The gallery could not load this run: ';
  const NO_DETECTIONS_TEXT = 'no detections';
  const IMAGE_MISSING_TEXT = 'image not available';
  const RAW_MISSING_TEXT = 'Raw frame not available; showing the overlay.';
  const CONFIDENCE_UNKNOWN_TEXT = 'no score';
  const UNKNOWN_COUNT_TEXT = 'unknown image count';
  const SEL_VERDICT = '[data-verdict]';
  const SEL_TOGGLE = '[data-toggle-image]';
  const SEL_TILE = '[data-index]';
  const SEL_IMG = 'img';
  const SEL_NOTE = '.vg-tile-note';
  const SEL_VIEW = '.vg-view';
  /** Badge text on the image well for each view. */
  const VIEW_TEXT = Object.freeze({ overlay: 'overlay', raw: 'raw frame' });

  /** Per-root state: selected run, manifest, image base, verdicts, fetch. */
  const STATE = new WeakMap();

  // ---------------------------------------------------------------- text helpers

  /**
   * Escape a value for HTML text or attribute context and strip control characters.
   * @param {*} value anything; non-strings are stringified, null and undefined become ''
   * @returns {string} safe text
   */
  function escapeHtml(value) {
    if (value === null || value === undefined) return '';
    return String(value)
      .replace(CONTROL_CHARS, '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /**
   * Format an ISO timestamp as "YYYY-MM-DD HH:MM AST" by its own digits.
   * The manifest and the run list already carry Atlantic Standard Time, so the
   * digits are used as written and never passed through the browser's zone.
   * @param {*} iso an ISO string; anything else gives ''
   * @returns {string} the AST form, or the input unchanged when it has no ISO prefix
   */
  function formatAst(iso) {
    if (typeof iso !== 'string' || !iso) return '';
    const m = ISO_PREFIX_PATTERN.exec(iso);
    return m ? m[1] + ' ' + m[2] + ' AST' : iso;
  }

  /**
   * "1 image" or "9 images".
   * @param {number} n count
   * @param {string} noun singular noun
   * @returns {string}
   */
  function plural(n, noun) {
    return n + ' ' + noun + (n === 1 ? '' : 's');
  }

  /**
   * A finite number or null.
   * @param {*} value
   * @returns {number|null}
   */
  function finiteOrNull(value) {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  /**
   * A non-negative integer or null.
   * @param {*} value
   * @returns {number|null}
   */
  function countOrNull(value) {
    return Number.isInteger(value) && value >= 0 ? value : null;
  }

  /**
   * The manifest's items as an array of objects, dropping anything else.
   * @param {*} manifest
   * @returns {Array<Object>}
   */
  function itemsOf(manifest) {
    const items = manifest && Array.isArray(manifest.items) ? manifest.items : [];
    return items.filter((it) => it && typeof it === 'object');
  }

  /**
   * Build an image URL from the manifest's image_base and a path from the manifest.
   * @param {string} imageBase "/api/step/8/image?out_dir=...&rel=" as the route returns it
   * @param {string} relPath overlay path relative to the run, or the raw source path
   * @returns {string} the URL, or '' when either part is unusable
   */
  function buildImageUrl(imageBase, relPath) {
    if (typeof imageBase !== 'string' || !imageBase) return '';
    if (typeof relPath !== 'string' || !relPath) return '';
    return imageBase + encodeURIComponent(relPath);
  }

  /**
   * The raw-frame URL for one item. The route resolves the frame from the run's
   * own item list by filename, so no client path is ever sent.
   * @param {string} outDir absolute output folder of the run
   * @param {string} filename the item's filename as the run's results list it
   * @returns {string} the URL, or '' when either part is unusable
   */
  function buildRawUrl(outDir, filename) {
    if (typeof outDir !== 'string' || !outDir) return '';
    if (typeof filename !== 'string' || !filename) return '';
    return IMAGE_URL + '?out_dir=' + encodeURIComponent(outDir) + '&raw=1&filename=' + encodeURIComponent(filename);
  }

  /**
   * The image_base to use for a run: the route's own, else one built from out_dir.
   * @param {Object} manifest
   * @param {string} outDir
   * @returns {string}
   */
  function imageBaseOf(manifest, outDir) {
    if (manifest && typeof manifest.image_base === 'string' && manifest.image_base) return manifest.image_base;
    return IMAGE_URL + '?out_dir=' + encodeURIComponent(outDir) + '&rel=';
  }

  // ---------------------------------------------------------------- pure render: chips

  /**
   * Resolve what a detection chip should say.
   * @param {Object} det one manifest detection {class, class_id, confidence, class_full_name?}
   * @param {Object} manifest the manifest, for its top-level class_full_names map
   * @returns {{className: string, fullName: string, isCode: boolean, confidence: number|null, derived: boolean}}
   *   derived is true when the full name was read out of a hyphenated or underscored class name
   */
  function describeDetection(det, manifest) {
    const d = det && typeof det === 'object' ? det : {};
    const className = typeof d.class === 'string' ? d.class : (d.class === undefined || d.class === null ? '' : String(d.class));
    let fullName = typeof d.class_full_name === 'string' ? d.class_full_name : '';
    const map = manifest && manifest.class_full_names && typeof manifest.class_full_names === 'object' ? manifest.class_full_names : null;
    if (!fullName && map && typeof map[className] === 'string') fullName = map[className];
    let derived = false;
    if (!fullName && BINOMIAL_TOKEN_PATTERN.test(className)) {
      fullName = className.replace(BINOMIAL_SEPARATOR_PATTERN, ' ');
      derived = true;
    }
    return {
      className: className,
      fullName: fullName,
      isCode: BARE_CODE_PATTERN.test(className),
      confidence: finiteOrNull(d.confidence),
      derived: derived,
    };
  }

  /**
   * Render one detection chip: the species in full with its code beside it and
   * the confidence to two decimals. A bare code with no full name on file is
   * shown as the code with a tooltip saying so; a class name written as a
   * binomial with separators reads with spaces, in italics, with the raw class
   * name in the chip's title; anything else is shown as written.
   * @param {Object} det one manifest detection
   * @param {Object} manifest the manifest
   * @returns {string} HTML
   */
  function renderChip(det, manifest) {
    const info = describeDetection(det, manifest);
    const conf = info.confidence === null
      ? '<span class="vg-conf">' + CONFIDENCE_UNKNOWN_TEXT + '</span>'
      : '<span class="vg-conf">' + info.confidence.toFixed(CONFIDENCE_DECIMALS) + '</span>';
    let name;
    let plainName;
    if (info.fullName) {
      const nameTag = BINOMIAL_PATTERN.test(info.fullName) ? 'i' : 'span';
      name = '<' + nameTag + ' class="vg-species">' + escapeHtml(info.fullName) + '</' + nameTag + '>';
      plainName = info.derived ? info.className : info.fullName;
      if (info.isCode) {
        name += ' <span class="vg-code">' + escapeHtml(info.className) + '</span>';
        plainName += ' ' + info.className;
      }
    } else if (info.isCode) {
      name = '<span class="vg-code vg-code-only" tabindex="0" title="'
        + escapeHtml(CODE_ONLY_TOOLTIP_PREFIX + info.className + CODE_ONLY_TOOLTIP_SUFFIX) + '">'
        + escapeHtml(info.className) + '</span>';
      plainName = info.className;
    } else {
      plainName = info.className || 'unnamed class';
      name = '<span class="vg-class">' + escapeHtml(plainName) + '</span>';
    }
    return '<span class="vg-chip" title="' + escapeHtml(plainName) + '">' + name + ' ' + conf + '</span>';
  }

  /**
   * Detections of one item sorted by confidence descending, unknown confidences last.
   * @param {Object} item one manifest item
   * @returns {Array<Object>}
   */
  function sortedDetections(item) {
    const dets = Array.isArray(item.detections) ? item.detections.filter((d) => d && typeof d === 'object') : [];
    return dets.slice().sort((a, b) => {
      const ca = finiteOrNull(a.confidence);
      const cb = finiteOrNull(b.confidence);
      if (ca === null && cb === null) return 0;
      if (ca === null) return 1;
      if (cb === null) return -1;
      return cb - ca;
    });
  }

  // ---------------------------------------------------------------- pure render: tiles

  /**
   * Render the three verdict buttons for one tile.
   * @param {string|null} verdict the saved verdict to show pressed, or null
   * @returns {string} HTML
   */
  function renderVerdictButtons(verdict) {
    return '<div class="vg-verdicts" role="group" aria-label="Verdict">'
      + VERDICTS.map((v) => {
        const pressed = verdict === v;
        return '<button type="button" class="vg-verdict' + (pressed ? ' is-pressed' : '') + '" data-verdict="' + v
          + '" aria-pressed="' + (pressed ? 'true' : 'false') + '" title="' + escapeHtml(VERDICT_TOOLTIPS[v]) + '">'
          + VERDICT_LABELS[v] + '</button>';
      }).join('')
      + '</div>';
  }

  /**
   * The note shown in place of an image that cannot be served.
   * @returns {string} HTML
   */
  function renderMissingImage() {
    return '<div class="vg-missing">' + IMAGE_MISSING_TEXT + '</div>';
  }

  /**
   * Render the image well: the overlay when saved, else the raw frame, else a note.
   * The raw frame is offered only when the item records a source path.
   * @param {Object} item one manifest item
   * @param {string} imageBase the run's image_base
   * @param {string} filename the item's filename, escaped here for the alt text
   * @param {string} outDir absolute output folder of the run, for the raw-frame URL
   * @returns {string} HTML
   */
  function renderImageWell(item, imageBase, filename, outDir) {
    const overlayUrl = buildImageUrl(imageBase, item.overlay);
    const rawUrl = typeof item.raw === 'string' && item.raw ? buildRawUrl(outDir, filename) : '';
    const startRaw = !overlayUrl && !!rawUrl;
    const src = overlayUrl || rawUrl;
    if (!src) return renderMissingImage();
    return '<button type="button" class="vg-image" data-toggle-image aria-pressed="' + (startRaw ? 'true' : 'false')
      + '" data-overlay-url="' + escapeHtml(overlayUrl) + '" data-raw-url="' + escapeHtml(rawUrl)
      + '" title="' + escapeHtml(TOGGLE_IMAGE_TOOLTIP) + '">'
      + '<img src="' + escapeHtml(src) + '" alt="' + (startRaw ? 'Raw frame ' : 'Overlay of ') + escapeHtml(filename) + '" loading="lazy">'
      + '<span class="vg-view">' + (startRaw ? VIEW_TEXT.raw : VIEW_TEXT.overlay) + '</span>'
      + '</button>';
  }

  /**
   * Render one gallery tile.
   * @param {Object} item one manifest item {filename, raw, overlay, detections}
   * @param {Object} manifest the manifest (for class_full_names)
   * @param {string} imageBase the run's image_base
   * @param {string|null} verdict saved verdict for this image, or null
   * @param {number} index the item's position in manifest.items
   * @param {string} [outDir] absolute output folder of the run, for the raw-frame URL
   * @returns {string} HTML
   */
  function renderTile(item, manifest, imageBase, verdict, index, outDir) {
    const it = item && typeof item === 'object' ? item : {};
    const filename = typeof it.filename === 'string' ? it.filename : '';
    const dets = sortedDetections(it);
    const chips = dets.length
      ? dets.map((d) => renderChip(d, manifest)).join('')
      : '<span class="vg-none">' + NO_DETECTIONS_TEXT + '</span>';
    return '<article class="vg-tile" data-index="' + (countOrNull(index) === null ? '' : index) + '">'
      + renderImageWell(it, imageBase, filename, typeof outDir === 'string' ? outDir : '')
      + '<div class="vg-name">' + escapeHtml(filename || 'unnamed image') + '</div>'
      + '<div class="vg-chips">' + chips + '</div>'
      + '<p class="vg-tile-note"></p>'
      + renderVerdictButtons(VERDICTS.indexOf(verdict) >= 0 ? verdict : null)
      + '</article>';
  }

  /**
   * Saved verdicts for a run: the manifest's own map first, the host's map on top.
   * @param {Object} manifest
   * @param {Object|null} verdicts host-supplied {filename: verdict}
   * @returns {Object} {filename: verdict}
   */
  function mergedVerdicts(manifest, verdicts) {
    const out = {};
    const fromManifest = manifest && manifest.verdicts && typeof manifest.verdicts === 'object' ? manifest.verdicts : {};
    const fromHost = verdicts && typeof verdicts === 'object' ? verdicts : {};
    [fromManifest, fromHost].forEach((src) => {
      Object.keys(src).forEach((k) => { if (VERDICTS.indexOf(src[k]) >= 0) out[k] = src[k]; });
    });
    return out;
  }

  /**
   * Render every tile of a manifest.
   * @param {Object} manifest
   * @param {string} imageBase
   * @param {Object|null} verdicts host-supplied {filename: verdict}
   * @param {string} [outDir] absolute output folder of the run, for the raw-frame URLs
   * @returns {string} HTML
   */
  function renderTiles(manifest, imageBase, verdicts, outDir) {
    const saved = mergedVerdicts(manifest, verdicts);
    return itemsOf(manifest)
      .map((it, i) => renderTile(it, manifest, imageBase, saved[it.filename] || null, i, outDir))
      .join('');
  }

  // ---------------------------------------------------------------- pure render: header

  /**
   * The threshold line: from the manifest top level, else from its args block.
   * @param {Object} m
   * @returns {string} plain sentence
   */
  function thresholdLine(m) {
    const args = m.args && typeof m.args === 'object' ? m.args : {};
    const conf = finiteOrNull(m.conf_threshold) !== null ? m.conf_threshold : finiteOrNull(args.conf);
    const iou = finiteOrNull(m.iou_threshold) !== null ? m.iou_threshold : finiteOrNull(args.iou);
    const imgsz = countOrNull(m.imgsz) !== null ? m.imgsz : countOrNull(args.imgsz);
    if (conf === null || iou === null || imgsz === null) return 'Thresholds not recorded for this run.';
    return 'Confidence at least ' + conf.toFixed(THRESHOLD_DECIMALS) + ', overlap threshold '
      + iou.toFixed(THRESHOLD_DECIMALS) + ', image size ' + imgsz + '.';
  }

  /**
   * A keyboard-focusable tooltip glyph carrying one explanation.
   * @param {string} text the tooltip sentence
   * @returns {string} HTML
   */
  function renderTipGlyph(text) {
    return '<button type="button" class="vg-tip" title="' + escapeHtml(text) + '"><span>?</span></button>';
  }

  /**
   * The overlap line: how many labels the renderer could not place clear of an
   * earlier one on a crowded image, or '' when none did. The run's results carry
   * no per-image breakdown, so the count is stated for the run as a whole.
   * @param {Object} m the run's results
   * @returns {string} plain sentence, or ''
   */
  function overlapLine(m) {
    const render = m.render && typeof m.render === 'object' ? m.render : {};
    const n = countOrNull(render.chips_overlapping);
    if (!n) return '';
    return plural(n, 'label') + (n === 1 ? ' overlaps' : ' overlap') + ' on crowded images; '
      + PREDICTIONS_FILENAME + ' carries every detection.';
  }

  /**
   * The skipped-detection line, or '' when nothing was skipped.
   * @param {Object} m
   * @returns {string}
   */
  function skippedLine(m) {
    const skipped = countOrNull(m.detections_skipped);
    if (!skipped) return '';
    const reasons = m.detections_skipped_reasons && typeof m.detections_skipped_reasons === 'object' ? m.detections_skipped_reasons : {};
    const parts = Object.keys(reasons)
      .filter((k) => countOrNull(reasons[k]))
      .map((k) => reasons[k] + ' ' + (SKIP_REASON_TEXT[k] || 'with reason ' + escapeHtml(k)));
    return plural(skipped, 'detection') + ' skipped' + (parts.length ? ': ' + parts.join(', ') : '') + '.';
  }

  /**
   * The blind-spot sentence when the no-detection share is above the threshold, else ''.
   * @param {number} nImages
   * @param {number} withAny
   * @returns {string}
   */
  function blindSpotLine(nImages, withAny) {
    const none = nImages - withAny;
    if (!nImages || none <= 0 || none / nImages <= BLIND_SPOT_NO_DETECTION_FRACTION) return '';
    return none + ' of ' + plural(nImages, 'image') + ' (' + Math.round(none / nImages * PERCENT)
      + '%) got no detection: possible blind spot or images unlike the training set.';
  }

  /**
   * Render the header block: counts, blind spot, thresholds, skipped, errors.
   * @param {Object} manifest
   * @returns {string} HTML, one <p> per line
   */
  function renderHeader(manifest) {
    const m = manifest && typeof manifest === 'object' ? manifest : {};
    const items = itemsOf(m);
    const nImages = items.length;
    const totalFromItems = items.reduce((n, it) => n + (Array.isArray(it.detections) ? it.detections.length : 0), 0);
    const total = countOrNull(m.total_predictions) !== null ? m.total_predictions : totalFromItems;
    const withAnyFromItems = items.filter((it) => Array.isArray(it.detections) && it.detections.length).length;
    const withAny = Math.min(nImages, countOrNull(m.with_any) !== null ? m.with_any : withAnyFromItems);
    const lines = [];
    lines.push('<p class="vg-counts">' + plural(nImages, 'image') + ', ' + plural(total, 'detection') + ', ' + (nImages - withAny) + ' with none.</p>');
    const blind = blindSpotLine(nImages, withAny);
    if (blind) lines.push('<p class="vg-blind">' + blind + '</p>');
    lines.push('<p>' + thresholdLine(m) + '</p>');
    const skipped = skippedLine(m);
    if (skipped) lines.push('<p>' + skipped + '</p>');
    const overlap = overlapLine(m);
    if (overlap) lines.push('<p>' + overlap + ' ' + renderTipGlyph(OVERLAP_TOOLTIP) + '</p>');
    const errors = Array.isArray(m.errors) ? m.errors.filter((e) => e && typeof e === 'object') : [];
    errors.forEach((e) => {
      lines.push('<p class="vg-error">Error on ' + escapeHtml(e.filename || e.path || 'an image') + ': ' + escapeHtml(e.error || 'no message') + '.</p>');
    });
    return lines.join('');
  }

  // ---------------------------------------------------------------- pure render: runs and states

  /**
   * Sort runs newest first by created_at; rows without an out_dir are dropped.
   * @param {*} runs
   * @returns {Array<Object>}
   */
  function orderedRuns(runs) {
    if (!Array.isArray(runs)) return [];
    return runs
      .filter((r) => r && typeof r === 'object' && typeof r.out_dir === 'string' && r.out_dir)
      .map((r, i) => ({ r: r, i: i }))
      .sort((a, b) => {
        const ta = typeof a.r.created_at === 'string' ? a.r.created_at : '';
        const tb = typeof b.r.created_at === 'string' ? b.r.created_at : '';
        if (ta === tb) return a.i - b.i;
        return ta < tb ? 1 : -1;
      })
      .map((x) => x.r);
  }

  /**
   * The label of one run option: "YYYY-MM-DD HH:MM AST, run name, model run, N images".
   * @param {Object} run
   * @returns {string} plain text (escaped by the caller)
   */
  function runLabel(run) {
    const when = formatAst(run.created_at) || 'time unknown';
    const name = typeof run.name === 'string' ? run.name : '';
    const model = typeof run.model_run_name === 'string' && run.model_run_name ? run.model_run_name : 'model unknown';
    const n = countOrNull(run.n_images);
    return [when, name, model, n === null ? UNKNOWN_COUNT_TEXT : plural(n, 'image')].filter(Boolean).join(', ');
  }

  /**
   * Render the run picker's options, newest first, the first one selected.
   * @param {Array<Object>} runs rows from /api/step/8/runs
   * @param {string} [selected] out_dir to select instead of the newest
   * @returns {string} HTML options
   */
  function renderRunOptions(runs, selected) {
    const ordered = orderedRuns(runs);
    const chosen = ordered.some((r) => r.out_dir === selected) ? selected : (ordered.length ? ordered[0].out_dir : '');
    return ordered.map((r) => '<option value="' + escapeHtml(r.out_dir) + '"' + (r.out_dir === chosen ? ' selected' : '') + '>'
      + escapeHtml(runLabel(r)) + '</option>').join('');
  }

  /**
   * The empty state: one plain sentence.
   * @returns {string} HTML
   */
  function renderEmpty() {
    return '<p class="vg-state">' + EMPTY_TEXT + '</p>';
  }

  /**
   * The run error state: one plain sentence carrying the message.
   * @param {*} message the error text, escaped
   * @returns {string} HTML
   */
  function renderError(message) {
    const text = typeof message === 'string' && message ? message : '';
    const sentence = text ? RUN_ERROR_PREFIX + text : RUN_ERROR_PREFIX.replace(/: $/, '');
    return '<p class="vg-state vg-error">' + escapeHtml(sentence.replace(/\.+$/, '')) + '.</p>';
  }

  // ---------------------------------------------------------------- DOM glue

  /**
   * Default JSON fetch: the parsed body, with an error field added on a non-2xx status.
   * @param {string} url
   * @returns {Promise<Object>}
   */
  function defaultFetchJson(url) {
    return fetch(url, { headers: { Accept: 'application/json' } }).then((resp) => resp.json().catch(() => ({})).then((data) => {
      const body = data && typeof data === 'object' ? data : {};
      if (!resp.ok && !body.error) body.error = 'HTTP ' + resp.status;
      return body;
    }));
  }

  /**
   * The state record for a root, created on first use.
   * @param {Element} rootEl
   * @param {Object} [opts] {fetchJson, verdicts}
   * @returns {Object}
   */
  function stateOf(rootEl, opts) {
    if (!rootEl || typeof rootEl.querySelector !== 'function') {
      throw new TypeError('VegaGallery: rootEl must be the gallery card element');
    }
    let s = STATE.get(rootEl);
    if (!s) {
      s = { outDir: '', manifest: null, imageBase: '', verdicts: {}, fetchJson: defaultFetchJson, bound: false };
      STATE.set(rootEl, s);
    }
    if (opts && typeof opts.fetchJson === 'function') s.fetchJson = opts.fetchJson;
    if (opts && opts.verdicts && typeof opts.verdicts === 'object') s.verdicts = opts.verdicts;
    return s;
  }

  /**
   * One of the fragment's elements by its IDS key.
   * @param {Element} rootEl the card root
   * @param {string} key a key of IDS
   * @returns {Element|null}
   */
  function part(rootEl, key) {
    return rootEl.querySelector('#' + IDS[key]);
  }

  /**
   * Show a state sentence in the status line, or hide the line.
   * @param {Element} rootEl the card root
   * @param {string} html the rendered state, or '' to hide the line
   * @returns {void}
   */
  function setStatus(rootEl, html) {
    const el = part(rootEl, 'status');
    if (!el) return;
    el.innerHTML = html;
    el.hidden = !html;
  }

  /**
   * Clear the header and the tiles.
   * @param {Element} rootEl the card root
   * @returns {void}
   */
  function clearBody(rootEl) {
    const header = part(rootEl, 'header');
    const tiles = part(rootEl, 'tiles');
    if (header) header.innerHTML = '';
    if (tiles) tiles.innerHTML = '';
  }

  /**
   * Error text from a fetch result or a thrown error.
   * @param {*} dataOrError
   * @returns {string}
   */
  function messageOf(dataOrError) {
    if (dataOrError && typeof dataOrError.error === 'string') return dataOrError.error;
    if (dataOrError && typeof dataOrError.message === 'string') return dataOrError.message;
    return 'no reason given';
  }

  /**
   * Load one run's manifest into the card.
   * @param {Element} rootEl the card root
   * @param {string} outDir absolute output folder of the run
   * @param {Object} [opts] {fetchJson, verdicts}
   * @returns {Promise<void>}
   */
  function select(rootEl, outDir, opts) {
    const s = stateOf(rootEl, opts);
    bind(rootEl);
    if (typeof outDir !== 'string' || !outDir) {
      clearBody(rootEl);
      setStatus(rootEl, renderError('no run folder was given'));
      return Promise.resolve();
    }
    const runSelect = part(rootEl, 'run');
    if (runSelect) runSelect.value = outDir;
    return s.fetchJson(MANIFEST_URL + '?out_dir=' + encodeURIComponent(outDir))
      .then((data) => {
        if (!data || typeof data !== 'object' || data.error) {
          clearBody(rootEl);
          setStatus(rootEl, renderError(messageOf(data)));
          return;
        }
        s.outDir = outDir;
        s.manifest = data;
        s.imageBase = imageBaseOf(data, outDir);
        setStatus(rootEl, '');
        const header = part(rootEl, 'header');
        const tiles = part(rootEl, 'tiles');
        if (header) header.innerHTML = renderHeader(data);
        if (tiles) tiles.innerHTML = renderTiles(data, s.imageBase, s.verdicts, s.outDir);
      })
      .catch((err) => {
        clearBody(rootEl);
        setStatus(rootEl, renderError(messageOf(err)));
      });
  }

  /**
   * Fetch the run list, fill the picker, and load the selected run (the
   * current selection when it still exists, else the newest).
   * @param {Element} rootEl the card root
   * @param {Object} [opts] {fetchJson, verdicts}
   * @returns {Promise<void>}
   */
  function load(rootEl, opts) {
    const s = stateOf(rootEl, opts);
    bind(rootEl);
    const runSelect = part(rootEl, 'run');
    const current = runSelect && typeof runSelect.value === 'string' ? runSelect.value : '';
    return s.fetchJson(RUNS_URL)
      .then((data) => {
        if (!data || typeof data !== 'object' || data.error) {
          clearBody(rootEl);
          setStatus(rootEl, '<p class="vg-state vg-error">' + escapeHtml(LOAD_ERROR_PREFIX + messageOf(data)) + '.</p>');
          return undefined;
        }
        const runs = orderedRuns(data.runs);
        if (!runs.length) {
          if (runSelect) runSelect.innerHTML = '';
          clearBody(rootEl);
          setStatus(rootEl, renderEmpty());
          return undefined;
        }
        const chosen = runs.some((r) => r.out_dir === current) ? current : runs[0].out_dir;
        if (runSelect) {
          runSelect.innerHTML = renderRunOptions(runs, chosen);
          runSelect.value = chosen;
        }
        return select(rootEl, chosen);
      })
      .catch((err) => {
        clearBody(rootEl);
        setStatus(rootEl, '<p class="vg-state vg-error">' + escapeHtml(LOAD_ERROR_PREFIX + messageOf(err)) + '.</p>');
      });
  }

  /**
   * Move the pressed state inside one tile to the given verdict.
   * @param {Element} tileEl the tile element
   * @param {string} verdict one of VERDICTS
   * @returns {void}
   */
  function setPressed(tileEl, verdict) {
    const buttons = tileEl.querySelectorAll(SEL_VERDICT);
    for (let i = 0; i < buttons.length; i += 1) {
      const b = buttons[i];
      const on = (b.dataset ? b.dataset.verdict : b.getAttribute('data-verdict')) === verdict;
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
      b.classList.toggle('is-pressed', on);
    }
  }

  /**
   * Swap the tile image between the overlay and the raw frame.
   * @param {Element} toggleEl the image button
   * @returns {boolean} true when a swap happened
   */
  function toggleImage(toggleEl) {
    const img = toggleEl.querySelector(SEL_IMG);
    if (!img) return false;
    const showingRaw = toggleEl.getAttribute('aria-pressed') === 'true';
    const overlayUrl = toggleEl.getAttribute('data-overlay-url') || '';
    const rawUrl = toggleEl.getAttribute('data-raw-url') || '';
    const next = showingRaw ? overlayUrl : rawUrl;
    if (!next) return false;
    img.setAttribute('src', next);
    toggleEl.setAttribute('aria-pressed', showingRaw ? 'false' : 'true');
    const badge = toggleEl.querySelector(SEL_VIEW);
    if (badge) badge.textContent = showingRaw ? VIEW_TEXT.overlay : VIEW_TEXT.raw;
    return true;
  }

  /**
   * Handle one click inside the card: a verdict press or an image toggle.
   * Exposed so a test can drive it without a browser event.
   * @param {Element} rootEl the card root
   * @param {Element} target the clicked element
   * @returns {string|null} 'verdict', 'toggle', or null when nothing applied
   */
  function handleClick(rootEl, target) {
    if (!target || typeof target.closest !== 'function') return null;
    const s = STATE.get(rootEl);
    const verdictBtn = target.closest(SEL_VERDICT);
    if (verdictBtn) {
      const verdict = verdictBtn.dataset ? verdictBtn.dataset.verdict : verdictBtn.getAttribute('data-verdict');
      const tile = verdictBtn.closest(SEL_TILE);
      if (!s || !s.manifest || !tile || VERDICTS.indexOf(verdict) < 0) return null;
      const rawIndex = String(tile.dataset ? tile.dataset.index : tile.getAttribute('data-index'));
      if (!INDEX_PATTERN.test(rawIndex)) return null;
      const item = itemsOf(s.manifest)[Number(rawIndex)];
      if (!item || typeof item.filename !== 'string') return null;
      setPressed(tile, verdict);
      s.verdicts[item.filename] = verdict;
      rootEl.dispatchEvent(new CustomEvent(VERDICT_EVENT, {
        bubbles: true,
        detail: { out_dir: s.outDir, filename: item.filename, verdict: verdict },
      }));
      return 'verdict';
    }
    const toggle = target.closest(SEL_TOGGLE);
    if (toggle) return toggleImage(toggle) ? 'toggle' : null;
    return null;
  }

  /**
   * When a raw frame fails to load, fall back to the overlay and say so on the tile;
   * when the overlay itself fails, replace the image with the missing note so no
   * broken-image icon shows, and retire the toggle.
   * @param {Element} imgEl the failing image
   * @returns {void}
   */
  function handleImageError(imgEl) {
    const toggle = imgEl && typeof imgEl.closest === 'function' ? imgEl.closest(SEL_TOGGLE) : null;
    const tile = toggle ? toggle.closest(SEL_TILE) : null;
    if (!toggle || !tile) return;
    const note = tile.querySelector(SEL_NOTE);
    if (toggle.getAttribute('aria-pressed') === 'true' && toggle.getAttribute('data-overlay-url')) {
      toggleImage(toggle);
      if (note) note.textContent = RAW_MISSING_TEXT;
      return;
    }
    toggle.innerHTML = renderMissingImage();
    toggle.removeAttribute('aria-pressed');
    toggle.setAttribute('disabled', '');
    if (note) note.textContent = '';
  }

  /**
   * Attach the card's listeners once: click delegation, run change, refresh, image errors.
   * @param {Element} rootEl the card root
   * @returns {void}
   */
  function bind(rootEl) {
    const s = stateOf(rootEl);
    if (s.bound || typeof rootEl.addEventListener !== 'function') return;
    s.bound = true;
    rootEl.addEventListener('click', (e) => {
      const refresh = e.target && typeof e.target.closest === 'function' ? e.target.closest('#' + IDS.refresh) : null;
      if (refresh) { load(rootEl); return; }
      handleClick(rootEl, e.target);
    });
    rootEl.addEventListener('change', (e) => {
      if (e.target && e.target.id === IDS.run) select(rootEl, e.target.value);
    });
    rootEl.addEventListener('error', (e) => {
      if (e.target && String(e.target.tagName).toLowerCase() === SEL_IMG) handleImageError(e.target);
    }, true);
  }

  // ---------------------------------------------------------------- exports

  const VegaGallery = Object.freeze({
    load: load,
    select: select,
    bind: bind,
    handleClick: handleClick,
    renderHeader: renderHeader,
    renderTile: renderTile,
    renderTiles: renderTiles,
    renderChip: renderChip,
    renderRunOptions: renderRunOptions,
    renderEmpty: renderEmpty,
    renderError: renderError,
    formatAst: formatAst,
    escapeHtml: escapeHtml,
    buildImageUrl: buildImageUrl,
    buildRawUrl: buildRawUrl,
    handleImageError: handleImageError,
    defaultFetchJson: defaultFetchJson,
    IDS: IDS,
    ROOT_CLASS: ROOT_CLASS,
    VERDICT_EVENT: VERDICT_EVENT,
    VERDICTS: VERDICTS,
    BLIND_SPOT_NO_DETECTION_FRACTION: BLIND_SPOT_NO_DETECTION_FRACTION,
  });

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = VegaGallery;
  }
  if (globalScope) {
    globalScope.VegaGallery = VegaGallery;
  }
})(typeof window !== 'undefined' ? window : null);

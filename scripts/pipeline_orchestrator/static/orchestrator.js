/* Pipeline Orchestrator, SPA logic */

// Internal step key -> the number the sidebar shows. Chain slots 4 and 5 were
// archived when the combined place + segment annotator (route key step4test)
// took over both, so every step after it displays one lower than its internal
// id. Mirrors STEP_DISPLAY in app.py; every user-facing string reads from here.
const STEP_DISPLAY = {
    1: '1', 2: '2', 3: '3', step4test: '4', 6: '5', 7: '6', 8: '7'
};

// The rail keys that are numbered stages, in rail order. The two tool tiles
// (Refine, Edit Masks) carry a glyph instead of a number, so they are not
// stages and are not numbered.
const STAGE_CHAIN = ['1', '2', '3', 'step4test', '6', '7', '8'];

// This project's own stage numbers, {rail key: '1'}, rebuilt by
// applyStageNumbers() every time the rail changes. A detection project's rail
// holds three stages, so its Train panel is stage 1 there and stage 5 in a
// project that annotates its own images. Null until a project opens, when the
// archived-chain map above is the whole truth.
let STAGE_DISPLAY = null;

// The number to print for a step: this project's own stage number once the
// rail has been numbered for it, else the archived-chain map, else the raw key
// so an unmapped step never renders "undefined". The typeof guard keeps the
// function runnable on its own, which is how test_orchestrator_ui.py lifts it
// out of this file.
function stepDisplayNum(step) {
    const stages = (typeof STAGE_DISPLAY === 'undefined') ? null : STAGE_DISPLAY;
    if (stages && stages[String(step)]) return stages[String(step)];
    return STEP_DISPLAY[step] || String(step);
}

let state = null;       // Current project state
let activeStep = 1;     // Currently viewed step panel
let pollTimers = {};    // step -> interval ID for polling
let stepStartTimes = {};// step -> Date when started
let elapsedTimers = {}; // step -> interval ID for elapsed display
let serviceWindows = {};// step -> window reference

// ── API helpers ─────────────────────────────────────────────────────────────

// The orchestrator's own death used to render as a frozen but healthy-looking
// page: every poller swallows its fetch error individually. Track consecutive
// failures across ALL api() calls and raise one page-level banner; any
// success clears it.
let _apiConsecutiveFails = 0;

function _setServerDownBanner(down) {
    let el = document.getElementById('orch-down-banner');
    if (!el) {
        el = document.createElement('div');
        el.id = 'orch-down-banner';
        el.style.cssText = 'display:none;position:fixed;top:0;left:0;right:0;'
            + 'z-index:9999;background:var(--error);color:#fff;text-align:center;'
            + 'padding:8px 16px;font-size:13px;font-weight:600';
        el.textContent = 'Orchestrator is not responding. Restart VEGA from the '
            + 'VICARIUS launcher; this page recovers on its own '
            + 'once the server is back.';
        el.title = 'Shown after several consecutive failed requests to the '
            + 'orchestrator on port 5050. Clears automatically when a request '
            + 'succeeds again.';
        document.body.appendChild(el);
    }
    el.style.display = down ? 'block' : 'none';
}

async function api(url, opts = {}) {
    try {
        const resp = await fetch(url, {
            headers: { 'Content-Type': 'application/json' },
            ...opts,
        });
        const data = await resp.json();
        // Module lock (VICARIUS): the server refuses with 423 while locked.
        if (resp.status === 423) data._locked = true;
        // The HTTP status rides on the parsed body so a loader can tell a 404
        // (nothing there yet, which the report card shows as its empty state)
        // from a failure.
        if (data && typeof data === 'object' && !Array.isArray(data)) data._status = resp.status;
        _apiConsecutiveFails = 0;
        _setServerDownBanner(false);
        return data;
    } catch (e) {
        _apiConsecutiveFails += 1;
        if (_apiConsecutiveFails >= 3) _setServerDownBanner(true);
        throw e;
    }
}

function post(url, body) {
    return api(url, { method: 'POST', body: JSON.stringify(body) });
}

function put(url, body) {
    return api(url, { method: 'PUT', body: JSON.stringify(body) });
}

// ── Time formatting ─────────────────────────────────────────────────────────

function formatElapsed(ms) {
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60);
    const h = Math.floor(m / 60);
    if (h > 0) return `${h}h ${m % 60}m ${s % 60}s`;
    if (m > 0) return `${m}m ${s % 60}s`;
    return `${s}s`;
}

function startElapsedTimer(step) {
    stepStartTimes[step] = Date.now();
    if (elapsedTimers[step]) clearInterval(elapsedTimers[step]);
    elapsedTimers[step] = setInterval(() => {
        const el = document.getElementById(`s${step}-elapsed`);
        if (el) el.textContent = formatElapsed(Date.now() - stepStartTimes[step]);
    }, 1000);
}

function stopElapsedTimer(step) {
    if (elapsedTimers[step]) {
        clearInterval(elapsedTimers[step]);
        delete elapsedTimers[step];
    }
}

// ── Project management ──────────────────────────────────────────────────────

// Projects are created and opened ONLY from the VICARIUS UI launcher; this
// orchestrator window never shows a local create/open/resume home. When it has
// no project (opened bare, stale, or after Save and Close), it bounces back to
// the VICARIUS UI instead.
function vicariusHomeUrl() {
    // Prefer the launcher URL the VICARIUS UI injected via ?home= when it opened
    // this window; fall back to the known local launcher.
    try {
        const h = new URLSearchParams(window.location.search).get('home');
        if (h) return h;
    } catch (e) {}
    return 'http://127.0.0.1:5090/modules/reef_point_seg';
}

function bounceToVicarius() {
    Object.values(pollTimers).forEach(clearInterval);
    Object.values(elapsedTimers).forEach(clearInterval);
    pollTimers = {};
    elapsedTimers = {};
    window.location.replace(vicariusHomeUrl());
}

// Module lock (VICARIUS): when the server starts refusing with 423, an
// already-open tab swaps the whole SPA for the same minimal message the
// server serves on / while locked. Same copy, same mailto, no bounce.
function buildLockPage() {
    return '<div class="vic-lock-page" style="min-height:100vh;display:flex;' +
        'align-items:center;justify-content:center;text-align:center;' +
        'background:#10151c;color:#dfe7ef;font-family:system-ui,sans-serif;">' +
        '<div style="max-width:520px;padding:40px;">' +
        '<svg xmlns="http://www.w3.org/2000/svg" width="56" height="56"' +
        ' viewBox="0 0 24 24" fill="none" stroke="#8fa4b8" stroke-width="1.6"' +
        ' stroke-linecap="round" stroke-linejoin="round"' +
        ' style="display:block;margin:0 auto 18px;">' +
        '<rect x="4" y="10.5" width="16" height="10" rx="2"></rect>' +
        '<path d="M8 10.5V7a4 4 0 0 1 8 0v3.5"></path></svg>' +
        '<h1 style="font-size:22px;font-weight:600;margin:0 0 12px;">Reef Point Seg</h1>' +
        '<p style="font-size:15px;line-height:1.6;margin:0;color:#b8c4d0;">' +
        'Under development. Please come back later. ' +
        '<a href="mailto:lauren.olinger@uvi.edu" style="color:#7fb3e8;">Email Lauren</a>' +
        ' for questions.</p>' +
        '<div style="margin-top:28px;font-size:12px;color:#5a6673;">VICARIUS module lock</div>' +
        '</div></div>';
}

// Topbar project name: "{name}  ({id})". Called from enterApp() and from every
// place `state` is replaced by a server response, so an orchestrator-side
// rename (auto project names are rebuilt on each save) reaches the topbar
// without a page reload.
const AUTO_NAME_TIP = 'Project name: initials, target labels, image count, date created. ' +
    'The labels join the name when Step 3 runs and the image count when Step 3 completes.';
const MANUAL_NAME_TIP = 'Project name as typed when the project was created.';

function renderProjectName() {
    const el = document.getElementById('topbar-project-name');
    if (!el || !state) return;
    el.textContent = (state.name || '') + '  (' + (state.id || '') + ')';
    el.title = state.name_auto ? AUTO_NAME_TIP : MANUAL_NAME_TIP;
    // A server refresh repaints the SAVED name. If the Step 3 picker holds an
    // unsaved selection, re-derive the preview so the topbar keeps matching
    // the chips instead of silently reverting.
    if (state.name_auto && step3PickerDirty()) previewProjectName();
}

// True when the rendered Step 3 label picker differs from the saved
// target_species (order-insensitive). False while the picker is not rendered,
// so an empty hidden store is never mistaken for "labels cleared".
function step3PickerDirty() {
    if (!state || !document.querySelector('.s3-label-cb')) return false;
    const speciesEl = document.getElementById('s3-species');
    const picked = speciesEl ? (speciesEl.value || '') : '';
    const s3 = (state.steps && state.steps['3'] && state.steps['3'].config) || {};
    const norm = s => String(s || '').split(',').map(x => x.trim()).filter(Boolean).sort().join(',');
    return norm(picked) !== norm(s3.target_species);
}

// Live preview of the auto project name while the Step 3 label picker changes.
// The server owns the name format (GET /api/project/name-preview); this only
// paints the reply into the topbar and never mutates state.name (the real
// rename happens server-side on the next config save). Debounced so a burst
// of checkbox toggles makes one request. Errors and non-auto projects are
// ignored, leaving the topbar as renderProjectName() drew it.
let _namePreviewTimer = null;
let _namePreviewSeq = 0;
function previewProjectName() {
    if (_namePreviewTimer) clearTimeout(_namePreviewTimer);
    _namePreviewTimer = setTimeout(async () => {
        _namePreviewTimer = null;
        if (!state || !state.name_auto) return;
        const speciesEl = document.getElementById('s3-species');
        const value = speciesEl ? (speciesEl.value || '').trim() : '';
        const seq = ++_namePreviewSeq;
        try {
            const data = await api('/api/project/name-preview?species=' + encodeURIComponent(value));
            if (seq !== _namePreviewSeq) return;  // a newer preview is in flight; never paint stale
            if (data && data.auto === true && typeof data.name === 'string' && data.name) {
                const el = document.getElementById('topbar-project-name');
                if (el) el.textContent = data.name + '  (' + (state.id || '') + ')';
            }
        } catch (e) { /* ignore: preview is cosmetic */ }
    }, 250);
}

// ── Training projects ───────────────────────────────────────────────────────
// A project minted from a prepared training source has no annotation work of
// its own, and the server refuses every annotation route on it. Two kinds
// arrive here and both behave identically:
//
//   composite_training  minted from a training set on the coverage matrix. Its
//                       step4test export is a symlink to the set.
//   bbox_training       minted from a staged box dataset on the Bounding Boxes
//                       page. Step 6 builds its split from that dataset.
//
// The server marks both with state.is_training_project and hands over the
// plain-language state.training_source_label / training_source_name, so this
// one code path serves both without the browser knowing the project types.
//
// Reflect it in the sidebar by disabling those tiles the same way the Expert
// Review I/O tile is disabled (renaming data-step to data-step-disabled drops
// them out of getNavOrder()/switchStep()), and name the source on the Train
// panel.
const COMPOSITE_DISABLED_TILES = ['1', '2', '3', 'step4test', 'step4loop', 'editmasks'];
const COMPOSITE_TILE_TOOLTIP =
    'Not applicable: this project trains from a training set or box dataset';
// The project types the flag above stands for. Kept for the fallback below, so
// an older server that has not yet grown the flag still disables the tiles.
const TRAINING_PROJECT_TYPES = ['composite_training', 'bbox_training'];

/**
 * True when a project state trains from a prepared source of either kind.
 * Takes the state rather than reading the module-level one so the rail rules
 * below can be exercised without a page.
 * @param {Object|null} st project state as /api/project/state returns it
 * @returns {boolean}
 */
function isTrainingProjectState(st) {
    if (!st) return false;
    if (typeof st.is_training_project === 'boolean') return st.is_training_project;
    return TRAINING_PROJECT_TYPES.indexOf(st.project_type) >= 0;
}

function isCompositeTrainingProject() {
    return isTrainingProjectState(state);
}

// ── Project family (decision D2) ────────────────────────────────────────────
// Every project belongs to one of two families, and the rail shows the stages
// that family actually has:
//
//   detection      trains a model that draws a box around each object. A
//                  project declares it by carrying a training_source of kind
//                  "bbox": a staged box dataset holds boxes and no outlines,
//                  so the four annotation stages have nothing to do in it.
//   segmentation   trains a model that outlines each object. Every other
//                  project, including one minted from a training set.
//
// The family is a fact about the project, not about one run: the Train panel's
// task control can be moved for a single run on an ordinary project, and that
// never changes which stages the project has.
const FAMILY_DETECTION = 'detection';
const FAMILY_SEGMENTATION = 'segmentation';
// The training_source spellings that mean detection. kind is what the minter
// writes; task_default is read too so a source that grows another kind name
// still resolves by the task it asks for.
const DETECTION_SOURCE_KINDS = ['bbox'];
const DETECTION_TASK_WORDS = ['detect', 'detection', 'bbox'];

/**
 * The family a project belongs to.
 * @param {Object|null} st project state as /api/project/state returns it
 * @returns {string} FAMILY_DETECTION or FAMILY_SEGMENTATION
 */
function projectFamily(st) {
    const src = (st && st.training_source) || {};
    const kind = String(src.kind || '').trim().toLowerCase();
    const task = String(src.task_default || '').trim().toLowerCase();
    if (DETECTION_SOURCE_KINDS.indexOf(kind) >= 0) return FAMILY_DETECTION;
    if (DETECTION_TASK_WORDS.indexOf(task) >= 0) return FAMILY_DETECTION;
    return FAMILY_SEGMENTATION;
}

/**
 * Which rail tiles this project hides and which it shows as already done
 * elsewhere.
 *
 * A detection project hides the annotation stages: they are not part of its
 * family and it can never run them. A segmentation project minted from a
 * training set keeps them on the rail, because the family has them and the
 * work was done in the projects the set was built from; their caption says
 * skipped. An ordinary project hides and skips nothing.
 *
 * @param {Object|null} st project state as /api/project/state returns it
 * @returns {{hidden: Array<string>, skipped: Array<string>}} rail keys
 */
function railPlan(st) {
    if (!isTrainingProjectState(st)) return { hidden: [], skipped: [] };
    if (projectFamily(st) === FAMILY_DETECTION) {
        return { hidden: COMPOSITE_DISABLED_TILES.slice(), skipped: [] };
    }
    return { hidden: [], skipped: COMPOSITE_DISABLED_TILES.slice() };
}

// What this project's Train panel opens on: 'detect' only when project.json's
// training_source says so (a box dataset holds no outlines to learn from),
// 'segment' for every other project. Segmentation is the default everywhere.
function defaultStep6Task(st) {
    const src = (st && st.training_source) || {};
    const want = String(src.task_default || '').trim().toLowerCase();
    return (want === 'detect' || want === 'detection' || want === 'bbox') ? 'detect' : 'segment';
}

// True for a project minted from a staged box dataset. Its Train panel has
// nothing to choose in two of its cards: the training source is the dataset
// itself (the server ignores cfg.source_dir on that path), and the class
// picker reads a YOLO-seg export's data.yaml + all_labels/, which a box
// dataset has neither of (/api/step/6/list_classes answers 400 there, naming
// steps this project can never run). Both cards are hidden and replaced by one
// line naming the dataset. Composite and ordinary projects keep them as they
// are.
function isBoxDatasetProject() {
    return isCompositeTrainingProject() && state.training_source_label === 'box dataset';
}

// Tooltip on the Training task control while it is pinned to Detector.
const BOX_DATASET_TASK_TOOLTIP = 'A box dataset holds boxes and no outlines, so only a detector can train on it. VICARIUS pins the task to Detector while a box dataset is the training source.';
// The visible line under the pinned select, which otherwise describes a
// change nobody can make. First sentence of the tooltip above, verbatim.
const BOX_DATASET_TASK_NOTE = 'A box dataset holds boxes and no outlines, so only a detector can train on it.';

// True when the option selected in the training-source picker is a staged
// box dataset (kind "bbox" in /api/composites); false for the project's own
// export and for a composite (kind "composite").
function step6SelectedSourceIsBoxDataset() {
    const sel = document.getElementById('s6-source');
    if (!sel || !sel.value) return false;
    const opt = sel.options[sel.selectedIndex];
    return !!(opt && opt.dataset && opt.dataset.kind === 'bbox');
}

// Show or hide one element by id; a missing element is ignored.
function setElementDisplay(id, display) {
    const el = document.getElementById(id);
    if (el) el.style.display = display;
}

// Pin the Training task to Detector (and say why on its tooltip), or hand the
// control back with the tooltip it had.
function pinStep6Task(pinned) {
    const taskEl = document.getElementById('s6-task');
    const note = document.getElementById('s6-task-note');
    if (!taskEl) return;
    if (pinned) {
        // Stash the task the user had BEFORE forcing Detector, the same way
        // the tooltip and the note below are stashed. Without it, browsing the
        // training-source picker into a box dataset and back out left a
        // segmentation project in Detector mode and it trained the wrong
        // family with nothing on screen saying so.
        if (!('freeTask' in taskEl.dataset)) taskEl.dataset.freeTask = taskEl.value || '';
        // Always through the change handler, even when the value is already
        // Detector: the starting model has to follow the task. A project that
        // opens on Detector with segmentation weights still selected (its
        // saved model, or the markup default) would otherwise keep them, and
        // the launch would be refused for a pairing nobody chose.
        taskEl.value = 'detect';
        onStep6TaskChange();
        taskEl.disabled = true;
        if (!taskEl.dataset.freeTooltip) taskEl.dataset.freeTooltip = taskEl.title;
        taskEl.title = BOX_DATASET_TASK_TOOLTIP;
        // The line under the select describes a change nobody can make while
        // the task is pinned; it says why it is pinned instead.
        if (note) {
            if (!note.dataset.freeText) note.dataset.freeText = note.textContent.trim();
            note.textContent = BOX_DATASET_TASK_NOTE;
        }
        // Which of the trained runs can be continued from here follows the
        // pin: while the task is pinned, a run of the other task cannot start
        // this one.
        applyStep6FinetuneFamily();
        return;
    }
    taskEl.disabled = false;
    if (taskEl.dataset.freeTooltip) { taskEl.title = taskEl.dataset.freeTooltip; delete taskEl.dataset.freeTooltip; }
    if (note && note.dataset.freeText) { note.textContent = note.dataset.freeText; delete note.dataset.freeText; }
    if ('freeTask' in taskEl.dataset) {
        const free = taskEl.dataset.freeTask;
        delete taskEl.dataset.freeTask;
        // The base-model catalog follows the task, so the value goes back
        // through onStep6TaskChange rather than being assigned on its own.
        if (free && taskEl.value !== free) { taskEl.value = free; onStep6TaskChange(); }
    }
    applyStep6FinetuneFamily();
}

// Box-dataset mode, applied whenever the training data is boxes: for a project
// minted from a staged box dataset (its source card is hidden too, since the
// dataset is fixed) and for any project whose training-source pick is a
// staged box dataset. The class picker is hidden (every class in the dataset
// trains, and /api/step/6/list_classes answers 400 for boxes), the two split
// controls the detection splitter never reads are hidden (min samples per
// class and the pinned transect split) while the validation and test
// fractions stay because the server forwards them to it, and the task is
// pinned to Detector. Everything is put back when a segmentation source is
// picked again. Re-run whenever the source selection settles.
function applyStep6BoxDatasetMode() {
    const projectIsBox = isBoxDatasetProject();
    const boxes = projectIsBox || step6SelectedSourceIsBoxDataset();
    const BOX_ONLY_HIDDEN_IDS = ['s6-classes-card', 's6-min-samples-group', 's6-pin-split-group'];
    setElementDisplay('s6-source-card', projectIsBox ? 'none' : '');
    BOX_ONLY_HIDDEN_IDS.forEach(id => setElementDisplay(id, boxes ? 'none' : ''));
    pinStep6Task(boxes);
    const note = document.getElementById('s6-bbox-source-note');
    if (note) {
        if (projectIsBox) {
            const name = state.training_source_name || '';
            // 2026-09-17 (LO): says what the folder holds, not what the code calls it.
            note.innerHTML = 'Learning from <strong>' + escapeHtml(name)
                + '</strong>, a folder of pictures with a box already drawn round every organism. '
                + 'Every kind of organism in it is learned.';
        }
        note.style.display = projectIsBox ? 'block' : 'none';
    }
}

// The caption under a stage the project does not run.
const TILE_SKIPPED_STATUS = 'skipped';

/**
 * Find a rail tile by its key, whether it is still in the flow (data-step) or
 * has been taken out of it (data-step-disabled).
 * @param {number|string} key the rail key
 * @returns {Element|null}
 */
function railTile(key) {
    return document.querySelector(`.step-nav li[data-step="${key}"]`)
        || document.querySelector(`.step-nav li[data-step-disabled="${key}"]`);
}

/**
 * Take one tile out of the flow. The nav order reads data-step, so renaming
 * the attribute drops the tile from getNavOrder(), from Prev and Next, and
 * from switchStep, without deleting anything from the markup.
 * @param {string} key the rail key
 * @param {string} tooltip the reason, shown on the tile
 * @returns {Element|null} the tile, or null when this page has none
 */
function disableRailTile(key, tooltip) {
    const li = railTile(key);
    if (!li) return null;
    if (li.hasAttribute('data-step')) {
        li.removeAttribute('data-step');
        li.setAttribute('data-step-disabled', key);
    }
    li.classList.add('step-disabled');
    // A tile nobody can open is never "you are here". Two tiles used to read
    // as the current stage at once because this highlight was left behind.
    li.classList.remove('active');
    li.removeAttribute('onclick');
    li.onclick = null;
    li.style.cursor = 'default';
    li.title = tooltip;
    return li;
}

/**
 * Apply this project's rail plan: hide the stages its family does not have,
 * and mark the stages it has but does not run.
 */
function applyRailFamily() {
    const plan = railPlan(state);
    plan.hidden.forEach(key => {
        const li = disableRailTile(key, COMPOSITE_TILE_TOOLTIP);
        if (li) li.style.display = 'none';
    });
    plan.skipped.forEach(key => {
        const li = disableRailTile(key, COMPOSITE_TILE_TOOLTIP);
        if (!li) return;
        li.style.display = '';
        const statusEl = li.querySelector('.step-status');
        if (statusEl) {
            statusEl.textContent = TILE_SKIPPED_STATUS;
            statusEl.title = COMPOSITE_TILE_TOOLTIP;
        }
    });
    hideEmptyRailSections();
}

/**
 * Hide a rail section heading whose every tile is hidden, so a detection
 * project never shows a heading with nothing under it.
 */
function hideEmptyRailSections() {
    const nav = document.getElementById('step-nav');
    if (!nav || !nav.children) return;
    let heading = null;
    let shown = 0;
    const settle = () => { if (heading) heading.style.display = shown ? '' : 'none'; };
    Array.from(nav.children).forEach(li => {
        const isHeading = li.classList && li.classList.contains('nav-section');
        if (isHeading) {
            settle();
            heading = li;
            shown = 0;
            return;
        }
        if (!li.style || li.style.display !== 'none') shown += 1;
    });
    settle();
}

// ── Stage numbering ─────────────────────────────────────────────────────────
// The rail is the only place a student counts stages, so the number in a panel
// heading, the number on a rail node and the number inside a sentence all come
// from one pass over the rail this project actually shows. A detection project
// shows three stages and its Model Inference panel is stage 3, not stage 7.

// The number a finished stage draws over its node, written by updateSidebar().
const TILE_DONE_MARK = '\u2713';
// The span that carries a stage's number inside its panel heading: the window
// ships one per panel ("Step <span class="stage-num">3</span>: Model
// Inference"), and a window that spells the number into the heading words has
// it wrapped into a span of the same class, marked with the inline class.
const STAGE_NUM_CLASS = 'stage-num';
const STAGE_NUM_INLINE_CLASS = 'stage-num-inline';
// Each panel's number answers to its own id, "step<key>-stage-num", stamped on
// that panel's span the first time it is numbered. No id is ever shared
// between panels: an id moved onto the open panel's own span would be found
// again when the next stage opens, and that panel's heading would then be
// written with another stage's number and contradict the rail.
const STAGE_NUM_ID_PREFIX = 'step';
const STAGE_NUM_ID_SUFFIX = '-stage-num';
// A heading that still spells its stage out, "Step 7: Model Inference".
// Group 1 is the word it uses, group 2 the number, group 3 the title.
const STAGE_HEADING_RE = /^\s*(Step|Stage)\s+(\d+)\s*:\s*(.+)$/;

/**
 * The stage numbers this project's rail shows, counted in rail order over the
 * stages the project has. A project that hides none of them is numbered
 * exactly as STEP_DISPLAY numbers it.
 * @param {Object|null} st project state as /api/project/state returns it
 * @returns {Object} {rail key: display number as a string}
 */
function stageNumbersFor(st) {
    const hidden = railPlan(st).hidden;
    const numbers = {};
    let shown = 0;
    STAGE_CHAIN.forEach(key => {
        if (hidden.indexOf(key) >= 0) return;
        shown += 1;
        numbers[key] = String(shown);
    });
    return numbers;
}

/**
 * Number this project's stages and draw the number everywhere it is shown:
 * the rail node and the panel heading, each panel written only with its own
 * number. Runs before updateSidebar(), which repaints a node from data-num.
 */
function applyStageNumbers() {
    STAGE_DISPLAY = stageNumbersFor(state);
    STAGE_CHAIN.forEach(key => {
        const num = STAGE_DISPLAY[key];
        if (!num) return;
        const li = railTile(key);
        const circle = li ? li.querySelector('.step-circle') : null;
        if (circle) {
            circle.dataset.num = num;
            if (String(circle.textContent).trim() !== TILE_DONE_MARK) circle.textContent = num;
        }
        renumberPanelHeading(key, num);
    });
}

/**
 * The id of the span that carries one stage's number, inside that stage's own
 * panel and nowhere else.
 * @param {number|string} key the rail key
 * @returns {string} "step8-stage-num" and the like
 */
function stageNumberId(key) {
    return `${STAGE_NUM_ID_PREFIX}${key}${STAGE_NUM_ID_SUFFIX}`;
}

/**
 * The span that carries one stage's number: the one already stamped with this
 * panel's id, else the one this panel ships, else null.
 * @param {number|string} key the rail key
 * @returns {Element|null}
 */
function stageNumberSlot(key) {
    const own = document.getElementById(stageNumberId(key));
    if (own) return own;
    const panel = document.getElementById(`panel-${key}`);
    return panel ? panel.querySelector(`.${STAGE_NUM_CLASS}`) : null;
}

/**
 * Wrap the number a plain heading spells out ("Step 7: Model Inference") in a
 * span, so a window that ships no stage span still has one to write into. A
 * heading that holds markup of its own is left exactly as it is, because
 * rewriting it would throw that markup away.
 * @param {number|string} key the rail key
 * @returns {Element|null} the span, or null when this heading has no number
 */
function wrapHeadingStageNumber(key) {
    const panel = document.getElementById(`panel-${key}`);
    const heading = panel ? panel.querySelector('h2') : null;
    if (!heading) return null;
    if (heading.children && heading.children.length) return null;
    const parts = String(heading.textContent || '').match(STAGE_HEADING_RE);
    if (!parts) return null;
    const slot = document.createElement('span');
    // class stage-num alone, which is what the shipped headings carry: it sets
    // the number's colour and tabular figures and no type of its own, so the
    // wrapped number reads exactly like a heading number the window ships.
    slot.className = `${STAGE_NUM_CLASS} ${STAGE_NUM_INLINE_CLASS}`;
    // Built as elements rather than as markup: the words came off the page and
    // go back onto it unescaped and unparsed.
    heading.textContent = '';
    heading.appendChild(document.createTextNode(`${parts[1]} `));
    heading.appendChild(slot);
    heading.appendChild(document.createTextNode(`: ${parts[3]}`));
    return slot;
}

/**
 * Put a stage's number in its own panel heading and stamp that span with the
 * panel's id, so every later pass writes the same stage's number back into the
 * same heading and no panel can be renumbered by another panel opening.
 * @param {number|string} key the rail key
 * @param {string} num the stage's number
 */
function renumberPanelHeading(key, num) {
    const slot = stageNumberSlot(key) || wrapHeadingStageNumber(key);
    if (!slot) return;
    slot.textContent = num;
    if (!slot.id) slot.id = stageNumberId(key);
}

// The route this project's work takes, named beside the wordmark. The markup
// carries the segmentation route through the annotation stages; a project
// minted from a prepared source starts further along, so it says so.
const PIPELINE_LINES = {
    detection: {
        text: 'Box dataset -> YOLO11 detector',
        tip: 'The route this project takes: VEGA splits the staged box dataset it was '
            + 'made from, then trains and runs a YOLO11 detector, which draws a box '
            + 'around each object it finds.',
    },
    segmentation: {
        text: 'Training set -> YOLO11 segmenter',
        tip: 'The route this project takes: VEGA splits the training set it was made '
            + 'from, then trains and runs a YOLO11 segmentation model, which outlines '
            + 'each colony it finds.',
    },
};

/**
 * Paint the route line for a project that starts from a prepared source. An
 * ordinary project keeps the full route the markup names.
 */
function renderPipelineLine() {
    const el = document.getElementById('topbar-pipeline');
    if (!el || !state || !isCompositeTrainingProject()) return;
    const line = PIPELINE_LINES[projectFamily(state)];
    if (!line) return;
    el.textContent = line.text;
    el.title = line.tip;
}

function applyCompositeTrainingMode() {
    if (!isCompositeTrainingProject()) return;
    applyStep6BoxDatasetMode();
    applyRailFamily();
    renderPipelineLine();
    const banner = document.getElementById('s6-composite-banner');
    // A box-dataset project already names its source on the line the mode
    // above fills, so the banner would say it a second time.
    if (banner && !isBoxDatasetProject()) {
        const c = state.composite || {};
        const label = state.training_source_label || 'training set';
        const name = state.training_source_name || c.name || '';
        // A composite also names the projects its masks came from. Each entry
        // carries the source project's id and, when that project is still on
        // disk, its display name. Show the name when there is one.
        const sources = (c.source_projects || [])
            .map(s => (s && (s.name || s.project_id)) || '')
            .filter(Boolean);
        let text = 'Training from ' + escapeHtml(label) + ' ' + escapeHtml(name);
        if (state.composite) {
            text += ', built from: ' + escapeHtml(sources.join(', ') || 'no recorded project');
        }
        banner.innerHTML = text;
        banner.style.display = 'block';
    }
}

function enterApp() {
    document.getElementById('setup-screen').style.display = 'none';
    document.getElementById('app-screen').style.display = 'flex';
    document.getElementById('topbar-actions').style.display = 'flex';
    renderProjectName();

    // Fresh project context: let the step-6 default preset auto-apply
    // again for this project, and wipe any run-name-edited flag from a
    // previous session so the field reverts to blank.
    window._step6PresetAutoApplied = false;
    const nameEl = document.getElementById('s6-run-name');
    if (nameEl) delete nameEl.dataset.userEdited;

    wireVegaCards();
    populateConfigs();
    // The rail runs BEFORE the sidebar pass so the tiles this project does not
    // have are already out of the flow when the first one is painted as the
    // current stage, and before the first switchStep so the nav order it reads
    // already excludes them.
    applyCompositeTrainingMode();
    // The stage numbers follow the rail the pass above settled, and are set
    // before updateSidebar() repaints each node from data-num.
    applyStageNumbers();
    updateSidebar();
    switchStep(openingStep(state));

    // A routing pass runs server-side and outlives this page, so a reload must
    // still show it. Paint it now and keep it current for as long as it is
    // alive, and refresh the non-chain tile captions on the same tick. A
    // training project runs none of those tools, so it polls nothing.
    if (!isCompositeTrainingProject()) {
        pollToolTiles();
        if (!window._step4testRouteTimer) {
            window._step4testRouteTimer = setInterval(pollToolTiles, 2500);
        }
    }
}

// One tick for everything the non-chain tools surface outside their panels:
// the routing banner on the Place + Segment panel and the live status caption
// on each tool tile.
//
// A tick makes several blocking health checks, so a slow one would otherwise
// stack on the next timer fire and pile up in-flight requests. The guard skips
// a tick while the previous one is still resolving.
let _toolTilesTickBusy = false;

async function pollToolTiles() {
    if (_toolTilesTickBusy) return;
    _toolTilesTickBusy = true;
    try {
        // One route_status read per tick, shared by the banner and the tile
        // caption instead of each fetching its own copy.
        let rs = null;
        try { rs = await api('/api/step/step4test/route_status'); } catch (e) { rs = null; }
        await renderStep4testRouteBanner(rs);
        await refreshToolTileStatuses(rs);
    } finally {
        _toolTilesTickBusy = false;
    }
}

// ── Sidebar ─────────────────────────────────────────────────────────────────

// Live caption for the three non-chain tool tiles. updateSidebar() only knows
// the numeric chain slots, so without this the tool tiles carried a frozen
// caption while their service was running, open, or dead.
const TOOL_TILE_IDLE = {
    step4test: 'combined annotator',
    step4loop: 'model-assist review',
    editmasks: 'relabel + fix',
};

function setToolTileStatus(key, text) {
    const el = document.getElementById(`${key}-status-text`);
    if (el && el.textContent !== text) el.textContent = text;
}

// A crashed sub-app, read off a /status payload that has no error field.
// StageRunner.poll_status returns {running, exit_code, log_tail} (plus healthy
// and port when a port is known), so a crash is "not running any more, and it
// left a nonzero exit code". A process that never launched reports exit_code
// null, which is idle rather than error.
function toolExitedBadly(s) {
    return !!s && s.running === false
        && s.exit_code !== null && s.exit_code !== undefined && s.exit_code !== 0;
}

async function refreshToolTileStatuses(routeStatus) {
    // Place + Segment also has a routing phase that runs before the annotator
    // exists, so its caption reads route_status first and /status second. The
    // tick hands the shared route_status in; a direct caller passes nothing and
    // gets its own read.
    let rs = routeStatus;
    if (rs === undefined) {
        try { rs = await api('/api/step/step4test/route_status'); } catch (e) { rs = null; }
    }
    let s4 = null;
    try { s4 = await api('/api/step/step4test/status'); } catch (e) { s4 = null; }
    let caption = TOOL_TILE_IDLE.step4test;
    if (rs && rs.phase === 'error') caption = 'error';
    else if (rs && ROUTE_LIVE_PHASES.includes(rs.phase) && rs.phase !== 'route_ready') caption = 'routing';
    else if (s4 && s4.healthy) caption = 'open';
    else if (s4 && s4.running) caption = 'running';
    setToolTileStatus('step4test', caption);

    for (const key of ['step4loop', 'editmasks']) {
        let s = null;
        try { s = await api(`/api/step/${key}/status`); } catch (e) { s = null; }
        let text = TOOL_TILE_IDLE[key];
        if (toolExitedBadly(s)) text = 'error';
        else if (s && s.healthy) text = 'open';
        else if (s && s.running) text = 'running';
        setToolTileStatus(key, text);
    }
}

function updateSidebar() {
    if (!state) return;
    // Cover every numeric chain slot 1..8. The slot 4 and 5 tiles were archived
    // when the combined annotator replaced them, so their <li> is absent and
    // skipped; 7 (Evaluate) and 8 (Inference) must be covered so their
    // status/checkmark update (they sit below train and display as 6/7).
    //
    // A tile the rail has taken out of the flow is looked up by its
    // data-step-disabled spelling too, so this pass still resets it. While it
    // was looked up by data-step alone, a stale "active" highlight stayed on a
    // skipped tile for the life of the window.
    for (let s = 1; s <= 8; s++) {
        const li = railTile(s);
        if (!li) continue;
        const st = state.steps[s];
        const status = (st && st.status) ? st.status : 'locked';
        const disabled = li.hasAttribute('data-step-disabled');

        li.className = disabled ? 'step-disabled' : '';
        if (!disabled) {
            li.classList.add(status);
            if (s === activeStep) li.classList.add('active');
        }

        const circle = li.querySelector('.step-circle');
        if (circle) {
            // Display number may differ from the internal step id (data-step);
            // see STEP_DISPLAY at the top of this file. data-num on the circle
            // carries the display number; fall back to the id.
            circle.textContent = (!disabled && status === 'completed')
                ? '\u2713' : (circle.dataset.num || String(s));
        }

        // The caption under a disabled tile was written by the rail and says
        // the stage is skipped; the live status would overwrite it with a
        // "locked" that is never going to change.
        const statusText = document.getElementById(`step${s}-status-text`);
        if (statusText && !disabled) statusText.textContent = status;
    }
}

// Chain slots 4 and 5 were archived into the combined annotator's tile, so a
// saved current_step of 4 or 5 opens the tile that owns their work instead of
// a panel that no longer exists.
const ARCHIVED_STEP_OWNER = { 4: 'step4test', 5: 'step4test' };
// The stage a project minted from a prepared source starts on.
const TRAIN_STEP = 6;

/**
 * The rail key a request opens.
 * @param {number|string} step the requested step
 * @returns {number|string} the key whose panel is shown
 */
function resolveStepKey(step) {
    const owner = ARCHIVED_STEP_OWNER[String(step)];
    return owner === undefined ? step : owner;
}

/**
 * The stage a project opens on: the one the server recorded, with an archived
 * slot resolved to the tile that owns it, floored at Train for a project that
 * trains from a prepared source (its earlier stages are not on its rail). A
 * student who left on Evaluate or Inference comes back to it.
 * @param {Object|null} st project state as /api/project/state returns it
 * @returns {number|string} the rail key to open
 */
function openingStep(st) {
    const key = resolveStepKey((st && st.current_step) || 1);
    if (!isTrainingProjectState(st)) return key;
    if (typeof key !== 'number' || key < TRAIN_STEP) return TRAIN_STEP;
    return key;
}

function switchStep(step) {
    let key = resolveStepKey(step);
    let panel = document.getElementById(`panel-${key}`);
    if (!panel) {
        // Blanking the whole content area is never the right answer to a step
        // that reaches no panel; the panel on screen stays where it is.
        console.warn(`switchStep: no panel for step ${key}; the open panel is kept`);
        return;
    }
    // A stage this project does not have is not in the nav order, so open the
    // first stage it does have rather than a panel the rail cannot get back to.
    const order = getNavOrder();
    if (order.length && order.indexOf(String(key)) < 0) {
        const first = navStepArg(order[0]);
        const firstPanel = document.getElementById(`panel-${first}`);
        if (firstPanel) {
            console.warn(`switchStep: step ${key} is not part of this project; opening ${first}`);
            key = first;
            panel = firstPanel;
        }
    }
    step = key;
    activeStep = key;
    document.querySelectorAll('.step-panel').forEach(p => p.classList.remove('active'));
    panel.classList.add('active');
    updateSidebar();
    // updateSidebar() only manages numeric steps; toggle the non-chain Expert
    // Review I/O tile's active state here and refresh it on open.
    const eli = document.querySelector('.step-nav li[data-step="expertids"]');
    if (eli) eli.classList.toggle('active', step === 'expertids');
    if (step === 'expertids' && window.ExpertIDs && window.ExpertIDs.onShow) window.ExpertIDs.onShow();
    // step4test (combined annotator) is another non-chain tile; toggle its
    // active state here too (updateSidebar only manages numeric steps).
    const tli = document.querySelector('.step-nav li[data-step="step4test"]');
    if (tli) tli.classList.toggle('active', step === 'step4test');
    // step4loop (Refine loop) is a third non-chain tile; same pattern.
    const lli = document.querySelector('.step-nav li[data-step="step4loop"]');
    if (lli) lli.classList.toggle('active', step === 'step4loop');
    // editmasks (Edit Masks) is a fourth non-chain tile; same pattern.
    const emli = document.querySelector('.step-nav li[data-step="editmasks"]');
    if (emli) emli.classList.toggle('active', step === 'editmasks');

    // Ensure the active panel always has working Prev/Next buttons.
    renderStepNavFooters();

    // Each tab switch refreshes project state from the server so the sidebar
    // shows the live "completed / running / error" after background work ends.
    // Also re-loads the per-step viewer so the panel is never stale.
    (async () => {
        try {
            const data = await api('/api/project/state');
            if (data.loaded && data.state) {
                state = data.state;
                renderProjectName();
                updateSidebar();
            }
        } catch (e) { /* ignore */ }
        if (step === 3) {
            // Re-fetch the label universe every time Step 3 is entered so the
            // picker always reflects the CURRENT Step 2 recode (master_codes_recoded).
            // A cached pre-recode load (Step 1's master codes) must never linger here.
            loadStep3Labels(true);
        }
        if (step === 6) {
            // Step 6 panel expects: preset auto-loaded if first time, class
            // list always freshly read from step 5 (or step 3 fallback), and
            // the run-name field blank until the user types. The Trained runs
            // card moved to the Evaluate panel (2026-09-21); openStep7Panel()
            // and its own disclosure toggle load it now, not this panel.
            const nameEl = document.getElementById('s6-run-name');
            if (nameEl && !nameEl.dataset.userEdited) nameEl.value = '';
            refreshStep6Presets({ autoApply: true });
            refreshStep6Classes();
            refreshStep6ContinueOptions();
            refreshStep6Sources();
            refreshStep6DataSummary();
        }
        if (step === 7) await openStep7Panel();
        if (step === 8) refreshStep8GalleryCard();
        if (step === 'step4loop') refreshStep4loopRuns();
    })();
}

// ── Step Prev/Next navigation (computed from #step-nav order) ───────────────
// The nav order is the single source of truth so the non-chain panels
// (s4review, expertids) participate too. data-step values are a mix of numbers
// ("1".."8") and strings ("s4review", "expertids"); we keep them as strings.

function getNavOrder() {
    const lis = document.querySelectorAll('#step-nav li[data-step]');
    return Array.from(lis).map(li => li.getAttribute('data-step'));
}

// Normalize a switchStep argument: numeric steps must be passed as Numbers
// (panels are #panel-1 etc and switchStep does numeric comparisons), the
// non-chain panels stay strings.
function navStepArg(ds) {
    return /^\d+$/.test(ds) ? Number(ds) : ds;
}

function navTitleFor(ds) {
    const li = document.querySelector(`#step-nav li[data-step="${ds}"]`);
    if (!li) return ds;
    const t = li.querySelector('.step-title');
    return t ? t.textContent.trim() : ds;
}

// (Re)build the Prev/Next buttons inside every .step-nav-footer so the user can
// move through the flow without the sidebar. Idempotent, so it is safe to call often.
function renderStepNavFooters() {
    const order = getNavOrder();
    order.forEach((ds, idx) => {
        const footer = document.getElementById(`nav-footer-${ds}`);
        if (!footer) return;
        const prevDs = idx > 0 ? order[idx - 1] : null;
        let nextDs = idx < order.length - 1 ? order[idx + 1] : null;
        let nextLabel = null;
        // step4loop (Refine loop) sits near the end of the nav order, but its
        // Next closes the active-learning loop by jumping back to Train rather
        // than dead-ending; the label names the loop, not just the tile.
        if (ds === 'step4loop') {
            nextDs = '6';
            nextLabel = `Loop to Train (${stepDisplayNum(6)})`;
        }
        const isNumeric = /^\d+$/.test(ds);

        // LEFT zone: Prev (always outline).
        const prevBtn = prevDs
            ? `<button class="btn btn-outline btn-sm" onclick='switchStep(${JSON.stringify(navStepArg(prevDs))})' title="Go to ${escAttr(navTitleFor(prevDs))}">&larr; Prev: ${escHtmlNav(navTitleFor(prevDs))}</button>`
            : `<button class="btn btn-outline btn-sm" disabled>&larr; Prev</button>`;

        // CENTER zone: Reset Step (numeric panels only) sits directly next to
        // Save and Close. Reset opens a styled confirm modal before clearing.
        let resetBtn = '';
        if (isNumeric) {
            resetBtn = `<button class="btn btn-outline btn-sm" onclick='openResetConfirm(${Number(ds)})' title="Clear this step's outputs and re-lock the steps after it. Asks for confirmation first; cannot be undone.">Reset Step</button>`;
        } else if (ds === 'step4test') {
            // 4.test is a non-chain tile with its own reset route; clearing it
            // removes routed_input + the YOLO export so the next Open re-routes.
            resetBtn = `<button class="btn btn-outline btn-sm" onclick='openResetConfirm("step4test")' title="Delete this tool's routed images and annotations so you can re-route and redo. Cannot be undone.">Reset Step</button>`;
        } else if (ds === 'step4loop') {
            // step4loop has its own surgical reset route (drops only unreviewed
            // pending model masks; accepted/exported work is kept).
            resetBtn = `<button class="btn btn-outline btn-sm" onclick='openResetConfirm("step4loop")' title="Remove model proposals you have not reviewed yet. Accepted masks and exported work are kept. Cannot be undone.">Reset Step</button>`;
        } else if (ds === 'editmasks') {
            // editmasks has its own reset route, but it never touches the
            // export - it only stops the service. Label reflects that.
            resetBtn = `<button class="btn btn-outline btn-sm" onclick='openResetConfirm("editmasks")' title="Stop the Edit Masks service. This never deletes or changes any mask; your export is untouched.">Reset Step</button>`;
        }
        const saveCloseBtn = `<button class="btn btn-outline btn-sm" onclick="saveAndClose()" title="Save the project as-is and close this window. Stops running sub-tools; you can reopen and resume later.">Save and Close</button>`;

        // RIGHT zone: Next. Neutral by default, pink once this numeric step is
        // completed; non-numeric panels keep Next neutral.
        const completed = isNumeric
            && typeof state !== 'undefined' && state && state.steps
            && state.steps[String(ds)]
            && state.steps[String(ds)].status === 'completed';
        const nextClass = completed ? 'btn btn-magenta btn-sm' : 'btn btn-outline btn-sm';
        const nextText = nextLabel || (nextDs ? `Next: ${escHtmlNav(navTitleFor(nextDs))}` : '');
        const nextBtn = nextDs
            ? `<button class="${nextClass}" onclick='switchStep(${JSON.stringify(navStepArg(nextDs))})' title="Go to ${escAttr(navTitleFor(nextDs))}">${nextText} &rarr;</button>`
            : `<button class="${nextClass}" disabled>Next &rarr;</button>`;

        footer.innerHTML =
            `<span class="nav-zone-left">${prevBtn}</span>` +
            `<span class="nav-zone-center">${resetBtn}${saveCloseBtn}</span>` +
            `<span class="nav-zone-right">${nextBtn}</span>`;
    });
}

// Local escapers (escapeHtml/escapeAttr are defined later in the file; these
// keep the footer renderer self-contained and hoisting-safe).
function escHtmlNav(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function escAttr(s) {
    return escHtmlNav(s).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// ── Step completion signal ("action button turns black, Next turns pink") ───
// step -> id of that step's primary action button.
const STEP_ACTION_BTN = {
    1:'s1-fetch-btn', 2:'s2-auto-btn', 3:'s3-run-btn',
    6:'s6-run-btn', 7:'s7-run-btn', 8:'s8-run-btn'
};

// On a step's primary success: flip its action button to the black "completed"
// state and re-render the footers so this step's Next lights up pink.
function markStepActionDone(step) {
    const btn = document.getElementById(STEP_ACTION_BTN[step]);
    if (btn) { btn.classList.remove('btn-magenta'); btn.classList.add('btn-done'); }
    renderStepNavFooters();
}

// ── Open-UI loading overlay (sub-app spin-up feedback) ─────────────────────

function showOpenUiOverlay(label) {
    const ov = document.getElementById('open-ui-overlay');
    const modal = document.getElementById('open-ui-modal');
    const msg = document.getElementById('open-ui-msg');
    const sub = document.getElementById('open-ui-sub');
    if (!ov || !modal) return;
    modal.classList.remove('error');
    if (msg) msg.innerHTML = `Starting ${escHtmlNav(label || '')} UI, waiting for it to come up&hellip;`;
    if (sub) sub.textContent = 'This can take a moment while models load.';
    ov.classList.add('visible');
    ov.setAttribute('aria-hidden', 'false');
}

function setOpenUiError(text, detail) {
    const modal = document.getElementById('open-ui-modal');
    const msg = document.getElementById('open-ui-msg');
    const sub = document.getElementById('open-ui-sub');
    if (!modal) return;
    modal.classList.add('error');
    if (msg) msg.textContent = text || 'The UI did not come up in time.';
    if (sub) sub.textContent = detail
        || 'Give it another moment, then try Open again.';
}

function hideOpenUiOverlay() {
    const ov = document.getElementById('open-ui-overlay');
    if (!ov) return;
    ov.classList.remove('visible');
    ov.setAttribute('aria-hidden', 'true');
}

// ── Populate configs from project state ─────────────────────────────────────

function populateConfigs() {
    if (!state) return;

    // Step 1
    const s1 = state.steps['1'].config;
    setVal('s1-input-dir', s1.input_dir || '');

    // Step 2
    const s2 = state.steps['2'].config;
    setVal('s2-remap-path', s2.remap_log_source || '');

    // Step 3
    const s3 = state.steps['3'].config;
    // Default to EMPTY so a brand-new project starts with NOTHING selected
    // (no checked rows, no chips). A saved project still restores whatever
    // target_species the user previously picked. The old build seeded every
    // project with a hardcoded 7-species default; treat that exact legacy
    // string as "not a deliberate choice" so projects created before this
    // change also open with nothing checked.
    const LEGACY_S3_DEFAULT = 'OFRA, PA, OA, OFAV, AL, MC, AA';
    let savedS3 = (s3.target_species || '').trim();
    if (savedS3 === LEGACY_S3_DEFAULT) savedS3 = '';
    setVal('s3-species', savedS3);
    setVal('s3-target', s3.target_instances || 1000);
    setVal('s3-min-year', s3.min_year || 2014);
    setVal('s3-max-year', s3.max_year || 2025);
    // Site-balance controls: default ON / 0.25, matching select_images.py's
    // own argparse defaults (site_balance true unless --no-site-balance).
    const s3siteBal = document.getElementById('s3-site-balance');
    if (s3siteBal) s3siteBal.checked = s3.site_balance !== false;
    setVal('s3-max-site-frac', s3.max_site_frac != null ? s3.max_site_frac : 0.25);
    // Auto-hydrate the label picker if master_codes_recoded.csv exists.
    // loadStep3Labels pre-checks rows matching the hidden s3-species CSV.
    loadStep3Labels(false);

    // Chain slots 4 and 5 have no panel any more (the Place + Segment tile owns
    // both), so nothing here hydrates them. Their saved config still travels
    // with the project and still feeds the step4test launcher server-side.

    // Step 6: training params (core + advanced)
    const s6 = (state.steps['6'] && state.steps['6'].config) || {};
    const s6set = (id, v, fallback) => setVal(id, (v !== undefined && v !== null) ? v : fallback);
    // Preset marker: seed the select's dataset from the stored config BEFORE
    // refreshStep6Presets() runs below, so a plain Run (no preset re-import
    // this session) still carries the existing marker forward via
    // collectConfig(6) instead of blanking it out.
    const s6presetSel = document.getElementById('s6-preset-select');
    if (s6presetSel) s6presetSel.dataset.appliedPresetId = s6.preset_id || '';
    // Core
    // run_name is ALWAYS blank by default; blank signifies "new run".
    // Only surface what the user typed for this session (non-empty strings
    // only); never rehydrate the auto-stamped name from prior state.
    setVal('s6-run-name', (typeof s6.run_name === 'string' && s6.run_name) ? s6.run_name : '');
    s6set('s6-epochs', s6.epochs, 500);
    s6set('s6-imgsz', s6.imgsz, 512);
    s6set('s6-batch', s6.batch, -1);
    s6set('s6-patience', s6.patience, 50);
    s6set('s6-valid', s6.valid_ratio, 0.2);
    s6set('s6-test', s6.test_ratio, 0.1);
    s6set('s6-min-samples', s6.min_samples, 10);
    // Training task, restored before the starting model so the model picker
    // is already showing the right catalog when the saved pick is applied. A
    // project with nothing saved opens on its own default: segmentation
    // everywhere except a project made from a box dataset.
    const taskEl = document.getElementById('s6-task');
    if (taskEl) {
        const savedTask = String(s6.task || '').trim().toLowerCase();
        // A box-dataset project can only train a detector, whatever an older
        // saved config says; the mode below pins the control as well.
        taskEl.value = isBoxDatasetProject() ? 'detect'
            : (savedTask === 'detect' || savedTask === 'segment') ? savedTask : defaultStep6Task(state);
        applyStep6TaskToModelPicker(taskEl.value);
    }
    // Subject: the other half of what chooses a training recipe. A saved value
    // wins and is marked as such, so the subject this project's staged dataset
    // recorded (applied by refreshStep6Presets below) only ever fills in for a
    // project that has none of its own.
    const subjectEl = document.getElementById('s6-subject');
    if (subjectEl) {
        const savedSubject = String(s6.subject || '').trim().toLowerCase();
        if (STEP6_PANEL_SUBJECTS.indexOf(savedSubject) >= 0) {
            subjectEl.value = savedSubject;
            subjectEl.dataset.subjectSource = 'saved';
        } else {
            subjectEl.value = STEP6_SUBJECT_DEFAULT;
            delete subjectEl.dataset.subjectSource;
        }
    }
    // The starting model and the continue-from pick are independent controls
    // now (2026-09-21): a saved run always restores its starting model, and
    // separately restores a continue-from pick when it saved one, rather than
    // the old either/or where a saved model_path hid which starting model the
    // form would fall back to if the pick were ever cleared.
    const modelEl = document.getElementById('s6-model');
    if (modelEl && s6.model) modelEl.value = s6.model;
    if (s6.model_path) applyStep6ModelPath(s6.model_path);
    s6set('s6-freeze', s6.freeze, '');
    const pinSplitEl = document.getElementById('s6-pin-split');
    if (pinSplitEl) pinSplitEl.checked = s6.pin_split !== '0' && s6.pin_split !== 0 && s6.pin_split !== false;
    if (s6.optimizer) { const sel = document.getElementById('s6-optimizer'); if (sel) sel.value = s6.optimizer; }
    if (s6.copy_paste_mode) {
        const sel = document.getElementById('s6-copy-paste-mode');
        if (sel) sel.value = s6.copy_paste_mode;
    }
    const cosEl = document.getElementById('s6-cos-lr');
    if (cosEl) cosEl.checked = !!s6.cos_lr;
    // Color
    s6set('s6-hsv-h', s6.hsv_h, 0.2);
    s6set('s6-hsv-s', s6.hsv_s, 0.3);
    s6set('s6-hsv-v', s6.hsv_v, 0.3);
    s6set('s6-bgr', s6.bgr, 0.0);
    // Geometric
    s6set('s6-degrees', s6.degrees, 0);
    s6set('s6-translate', s6.translate, 0);
    s6set('s6-scale', s6.scale, 0.2);
    s6set('s6-shear', s6.shear, 0);
    s6set('s6-perspective', s6.perspective, 0);
    s6set('s6-flipud', s6.flipud, 0.5);
    s6set('s6-fliplr', s6.fliplr, 0.5);
    // Mixing
    s6set('s6-mosaic', s6.mosaic, 0);
    s6set('s6-mixup', s6.mixup, 0);
    s6set('s6-cutmix', s6.cutmix, 0);
    s6set('s6-copy-paste', s6.copy_paste, 0);
    s6set('s6-close-mosaic', s6.close_mosaic, 10);
    // LR / loss
    s6set('s6-lr0', s6.lr0, 0.01);
    s6set('s6-lrf', s6.lrf, 0.01);
    s6set('s6-momentum', s6.momentum, 0.937);
    s6set('s6-weight-decay', s6.weight_decay, 0.0005);
    s6set('s6-warmup-epochs', s6.warmup_epochs, 3.0);
    s6set('s6-box', s6.box, 7.5);
    s6set('s6-cls', s6.cls, 0.5);
    s6set('s6-dfl', s6.dfl, 1.5);
    s6set('s6-label-smoothing', s6.label_smoothing, 0);
    // Model card tags: a saved pick wins; a project with none gets the tags
    // inferred from the training source (refreshStep6CardTags, below).
    restoreStep6CardTag('s6-card-subject', s6.card_subject, s6.card_subject_picked);
    restoreStep6CardTag('s6-card-data-source', s6.card_data_source, s6.card_data_source_picked);
    // Device / DDP
    applyStepDevice(6, s6.device, TRAIN_DEVICE_DEFAULT);

    // Step 7: evaluation
    const s7 = (state.steps['7'] && state.steps['7'].config) || {};
    // Blank means "the size the run trained at"; prefillStep7ImageSize()
    // fills it from the selected run when the Evaluate panel opens.
    setVal('s7-imgsz', s7.imgsz != null ? s7.imgsz : '');
    // The fallbacks come from EVAL_DEFAULTS, which mirrors config/pipeline.yaml:
    // a project whose step 7 block names no confidence must still open on the
    // measurement cut-off, never on a literal repeated here.
    setVal('s7-conf', s7.conf_threshold != null ? s7.conf_threshold : EVAL_DEFAULTS.conf_threshold);
    setVal('s7-iou', s7.iou_threshold != null ? s7.iou_threshold : EVAL_DEFAULTS.iou_threshold);
    setVal('s7-preview-count', s7.preview_count != null ? s7.preview_count : EVAL_DEFAULTS.preview_count);
    setVal('s7-pdf-export-dir', s7.pdf_export_dir || '');
    if (s7.split) { const sel = document.getElementById('s7-split'); if (sel) sel.value = s7.split; }
    showStep7ConfUpgrade(state.eval_conf_upgraded_from);
    // Every Evaluate setting this project saved that is not the recommended
    // one, so the panel opens saying which of its numbers differ.
    renderEvalOffRecommended(state.eval_settings_off_recommended);

    // Step 8: inference
    const s8 = (state.steps['8'] && state.steps['8'].config) || {};
    setVal('s8-run-name', s8.run_name || '');
    setVal('s8-imgsz', s8.imgsz != null ? s8.imgsz : 512);
    setVal('s8-source-dir', s8.source_dir || '');
    setVal('s8-sample-count', s8.sample_count != null ? s8.sample_count : 100);
    setVal('s8-conf', s8.conf_threshold != null ? s8.conf_threshold : 0.25);
    setVal('s8-iou', s8.iou_threshold != null ? s8.iou_threshold : 0.7);
    setVal('s8-mask-alpha', s8.mask_alpha != null ? s8.mask_alpha : step8FillAlphaDefault());
    applyStepDevice(8, s8.device, INFER_DEVICE_DEFAULT);
    if (s8.sample_mode) { const sel = document.getElementById('s8-sample-mode'); if (sel) sel.value = s8.sample_mode; }
    const srcType = s8.source_type || 'unused';
    const radio = document.querySelector(`input[name="s8-src"][value="${srcType}"]`);
    if (radio) radio.checked = true;
    toggleStep8SourceFields();
    const so = document.getElementById('s8-save-overlays');
    if (so) so.checked = s8.save_overlays !== false;
    const pv = document.getElementById('s8-include-pts');
    if (pv) pv.checked = !!s8.include_pts_variant;
    const db = document.getElementById('s8-draw-boxes');
    if (db) db.checked = !!s8.draw_boxes;

    // Populate run dropdowns (async)
    refreshRunList();
    // Populate step-6 preset list (always, since presets are static YAML files).
    // autoApply=true fills in the form from the default preset on first load
    // when the user has not yet manually imported one.
    refreshStep6Presets({ autoApply: true });
    // Populate step-7 eval-preset list too.
    refreshStep7Presets();
    // Populate step-6 class picker. Backend falls back to step 3
    // target_species if step 5 hasn't produced data.yaml yet.
    refreshStep6Classes();
    refreshStep6CardTags();
    // Populate "Continue from a VEGA model" (champion, this project's runs,
    // other projects' runs, and runs trained outside any project).
    refreshStep6ContinueOptions();
    // Populate the Step 6 training-source select (composite exports).
    refreshStep6Sources(s6.source_dir || '');
    // Populate the Rounds ledger table (Task 9).
    refreshStep7Rounds();

    // The report and gallery cards load when their panels open (switchStep)
    // and again when their runs finish (startPolling), keyed by the selected run.

    // Restore running services. Slots 4 and 5 are skipped: their panels are
    // gone, so a stale "running" flag on an old project has no widget to drive
    // and must not start a poller that paints nothing.
    for (const s of [1, 2, 3, 6, 7, 8]) {
        if (state.steps[s] && state.steps[s].status === 'running') {
            startPolling(s);
            setStepRunControls(s, true);
            if (s === 2) {
                const svc = document.getElementById(`s${s}-service`);
                if (svc) svc.style.display = 'block';
                startElapsedTimer(s);
            }
        }
    }
}

function setVal(id, val) {
    const el = document.getElementById(id);
    if (el) el.value = val;
}

// Defaults the Evaluate and Inference panels fall back to when a numeric
// field is blank or not a number; they mirror the step 7 and step 8 seeds in
// config/pipeline.yaml. A substituted value is echoed into the field so what
// is sent is what is shown.
const EVAL_DEFAULTS = Object.freeze({ conf_threshold: 0.001, iou_threshold: 0.6, preview_count: 8 });
const INFER_DEFAULTS = Object.freeze({ imgsz: 512, sample_count: 100, conf_threshold: 0.25, iou_threshold: 0.7 });

/**
 * Read a numeric field, keeping a deliberate 0 and substituting the fallback
 * only when the field is blank or not a finite number. A substituted value is
 * written back into the field.
 * @param {string} id the input's element id
 * @param {number} fallback the value used when the field holds no number
 * @param {{integer?: boolean}} [opts] integer: parse as a whole number
 * @returns {number} the value that will be sent
 */
function readNumberField(id, fallback, opts) {
    const el = document.getElementById(id);
    const raw = el ? String(el.value).trim() : '';
    const parsed = (opts && opts.integer) ? parseInt(raw, 10) : parseFloat(raw);
    if (raw !== '' && Number.isFinite(parsed)) return parsed;
    if (el) el.value = String(fallback);
    return fallback;
}

/**
 * Read a numeric field that may be left blank on purpose: '' when blank, else
 * the whole number typed. Nothing is substituted client-side.
 * @param {string} id the input's element id
 * @returns {number|string} the number, or '' for a blank field
 */
function readOptionalNumberField(id) {
    const el = document.getElementById(id);
    const raw = el ? String(el.value).trim() : '';
    if (raw === '') return '';
    const parsed = parseInt(raw, 10);
    return Number.isFinite(parsed) ? parsed : '';
}

// Fill opacity the Inference panel starts at: a detector's boxes cover much
// of a frame, so a detector project fills lightly.
const FILL_ALPHA_DEFAULT_SEGMENT = 0.45;
const FILL_ALPHA_DEFAULT_DETECT = 0.15;

// The task this project trains: its saved Train-step task, else its default.
function projectTrainingTask() {
    const cfg = (state && state.steps && state.steps['6'] && state.steps['6'].config) || {};
    const saved = String(cfg.task || '').trim().toLowerCase();
    if (saved === 'detect' || saved === 'segment') return saved;
    return defaultStep6Task(state);
}

/**
 * The fill opacity the Inference panel uses when its field is blank.
 * @returns {number} 0.15 for a detector project, else 0.45
 */
function step8FillAlphaDefault() {
    return projectTrainingTask() === 'detect' ? FILL_ALPHA_DEFAULT_DETECT : FILL_ALPHA_DEFAULT_SEGMENT;
}

// ── Collect config from form before running ─────────────────────────────────

function collectConfig(step) {
    const cfg = {};
    if (step === 1) {
        cfg.input_dir = document.getElementById('s1-input-dir').value.trim();
    } else if (step === 3) {
        // Rebuild target_species from the checked label-picker rows so the
        // hidden store is always in sync before we POST the config.
        syncStep3SpeciesField();
        cfg.target_species = document.getElementById('s3-species').value.trim();
        cfg.target_instances = parseInt(document.getElementById('s3-target').value) || 1000;
        cfg.min_year = parseInt(document.getElementById('s3-min-year').value) || 2014;
        cfg.max_year = parseInt(document.getElementById('s3-max-year').value) || 2025;
        const siteBalEl = document.getElementById('s3-site-balance');
        cfg.site_balance = siteBalEl ? siteBalEl.checked : true;
        const maxFracEl = document.getElementById('s3-max-site-frac');
        const maxFracVal = maxFracEl ? parseFloat(maxFracEl.value) : NaN;
        // The field's min/max attributes are cosmetic (no checkValidity call),
        // so clamp here too. This is a cheap client-side guard only; the
        // authoritative check is server-side in _run_step3 (_resolve_max_site_frac),
        // which also catches a value posted straight to the API.
        cfg.max_site_frac = isNaN(maxFracVal) ? 0.25 : Math.min(1.0, Math.max(0.05, maxFracVal));
    } else if (step === 6) {
        const $ = id => document.getElementById(id);
        const f = id => parseFloat($(id).value);
        const i = id => parseInt($(id).value);
        cfg.run_name = $('s6-run-name').value.trim();
        // Training source: '' = this project's Step 4 export; else the
        // absolute path of a composite export (validated server-side, D5).
        cfg.source_dir = $('s6-source') ? ($('s6-source').value || '') : '';
        // What to train: 'segment' (the default) or 'detect'. The server
        // normalizes it, follows it for the base-weights default, and forwards
        // it as TCRMP_STEP6_TASK. Every other knob below is task-agnostic.
        cfg.task = $('s6-task') ? ($('s6-task').value || 'segment') : 'segment';
        // What this run is training on: coral, fish, algae or other. With the
        // task above it is what chooses a training recipe, and it is saved so
        // the panel opens on the same pair next time.
        cfg.subject = step6SubjectValue();
        // The two finding tags VEGA writes on the model's card (subject and
        // training data source); blank when nobody picked and none was inferred.
        cfg.card_subject = ($('s6-card-subject') ? $('s6-card-subject').value : '') || '';
        cfg.card_data_source = ($('s6-card-data-source') ? $('s6-card-data-source').value : '') || '';
        // Whether a person chose each tag (Not set included), so a reload keeps
        // a saved Not set and a prefilled value keeps following the recipe.
        cfg.card_subject_picked = !!($('s6-card-subject') && $('s6-card-subject').dataset.picked);
        cfg.card_data_source_picked = !!($('s6-card-data-source') && $('s6-card-data-source').dataset.picked);
        // The starting model is always a base checkpoint now (2026-09-21):
        // #s6-model holds only catalog names. A pick under "Continue from a
        // VEGA model" (#s6-continue-from) carries the run's absolute best.pt
        // path and, when present, overrides model server-side (_run_step6).
        cfg.model = $('s6-model').value;
        const continueSel = $('s6-continue-from');
        const continueVal = continueSel ? continueSel.value.trim() : '';
        if (continueVal) cfg.model_path = continueVal;
        const freezeVal = $('s6-freeze').value;
        cfg.freeze = (freezeVal === '' || freezeVal === null) ? null : (isNaN(i('s6-freeze')) ? null : i('s6-freeze'));
        cfg.pin_split = $('s6-pin-split').checked ? '1' : '0';
        cfg.epochs = i('s6-epochs') || 500;
        cfg.imgsz = i('s6-imgsz') || 512;
        cfg.batch = isNaN(i('s6-batch')) ? -1 : i('s6-batch');
        cfg.patience = i('s6-patience') || 50;
        cfg.valid_ratio = isNaN(f('s6-valid')) ? 0.2 : f('s6-valid');
        cfg.test_ratio = isNaN(f('s6-test')) ? 0.1 : f('s6-test');
        cfg.min_samples = i('s6-min-samples') || 10;
        cfg.optimizer = $('s6-optimizer').value;
        cfg.cos_lr = $('s6-cos-lr').checked;
        cfg.close_mosaic = i('s6-close-mosaic');
        // Color
        cfg.hsv_h = isNaN(f('s6-hsv-h')) ? 0 : f('s6-hsv-h');
        cfg.hsv_s = isNaN(f('s6-hsv-s')) ? 0 : f('s6-hsv-s');
        cfg.hsv_v = isNaN(f('s6-hsv-v')) ? 0 : f('s6-hsv-v');
        cfg.bgr = isNaN(f('s6-bgr')) ? 0 : f('s6-bgr');
        // Geometric
        cfg.degrees = isNaN(f('s6-degrees')) ? 0 : f('s6-degrees');
        cfg.translate = isNaN(f('s6-translate')) ? 0 : f('s6-translate');
        cfg.scale = isNaN(f('s6-scale')) ? 0 : f('s6-scale');
        cfg.shear = isNaN(f('s6-shear')) ? 0 : f('s6-shear');
        cfg.perspective = isNaN(f('s6-perspective')) ? 0 : f('s6-perspective');
        cfg.flipud = isNaN(f('s6-flipud')) ? 0 : f('s6-flipud');
        cfg.fliplr = isNaN(f('s6-fliplr')) ? 0 : f('s6-fliplr');
        // Mixing
        cfg.mosaic = isNaN(f('s6-mosaic')) ? 0 : f('s6-mosaic');
        cfg.mixup = isNaN(f('s6-mixup')) ? 0 : f('s6-mixup');
        cfg.cutmix = isNaN(f('s6-cutmix')) ? 0 : f('s6-cutmix');
        cfg.copy_paste = isNaN(f('s6-copy-paste')) ? 0 : f('s6-copy-paste');
        cfg.copy_paste_mode = $('s6-copy-paste-mode').value;
        // LR / loss
        cfg.lr0 = isNaN(f('s6-lr0')) ? 0.01 : f('s6-lr0');
        cfg.lrf = isNaN(f('s6-lrf')) ? 0.01 : f('s6-lrf');
        cfg.momentum = isNaN(f('s6-momentum')) ? 0.937 : f('s6-momentum');
        cfg.weight_decay = isNaN(f('s6-weight-decay')) ? 0.0005 : f('s6-weight-decay');
        cfg.warmup_epochs = isNaN(f('s6-warmup-epochs')) ? 3.0 : f('s6-warmup-epochs');
        cfg.box = isNaN(f('s6-box')) ? 7.5 : f('s6-box');
        cfg.cls = isNaN(f('s6-cls')) ? 0.5 : f('s6-cls');
        cfg.dfl = isNaN(f('s6-dfl')) ? 1.5 : f('s6-dfl');
        cfg.label_smoothing = isNaN(f('s6-label-smoothing')) ? 0 : f('s6-label-smoothing');
        // Class-inclusion filter: null means "train on every class in step 5".
        cfg.include_classes = collectStep6IncludedClasses();
        // The id space those ids belong to: the training source the picker was
        // last loaded from ('' = this project's Step 4 export). The server
        // carries the selection over by name when the training source differs.
        const clsHost = $('s6-classes-list');
        cfg.include_classes_source = (clsHost && clsHost.dataset.source) ? clsHost.dataset.source : '';
        // Device / DDP selection: string forwarded as ultralytics `device=`.
        cfg.device = collectStepDevice(6);
        // Preset marker (D-preset-marker, 2026-09-02): carries forward whichever
        // preset was last applied to this form, seeded from the stored config on
        // load (populateConfigs) and overwritten by importStep6Preset on a fresh
        // apply. Saved alongside every other Step 6 field so a later panel open
        // can tell a deliberately-tuned config apart from an untouched one and
        // skip re-applying the default preset over it.
        const presetSel = $('s6-preset-select');
        cfg.preset_id = (presetSel && presetSel.dataset.appliedPresetId) || '';
    } else if (step === 7) {
        const $ = id => document.getElementById(id);
        cfg.run_dir = $('s7-run-dir').value;
        cfg.split = $('s7-split').value;
        // Blank means "the size the run trained at": the route fills it from
        // the run's args.yaml, so the field is never coerced client-side.
        cfg.imgsz = readOptionalNumberField('s7-imgsz');
        cfg.conf_threshold = readNumberField('s7-conf', EVAL_DEFAULTS.conf_threshold);
        cfg.iou_threshold = readNumberField('s7-iou', EVAL_DEFAULTS.iou_threshold);
        cfg.preview_count = readNumberField('s7-preview-count', EVAL_DEFAULTS.preview_count, { integer: true });
        cfg.pdf_export_dir = $('s7-pdf-export-dir').value.trim();
    } else if (step === 8) {
        const $ = id => document.getElementById(id);
        cfg.run_dir = $('s8-run-dir').value;
        cfg.run_name = $('s8-run-name').value.trim();
        cfg.imgsz = readNumberField('s8-imgsz', INFER_DEFAULTS.imgsz, { integer: true });
        const srcRadio = document.querySelector('input[name="s8-src"]:checked');
        cfg.source_type = srcRadio ? srcRadio.value : 'unused';
        cfg.source_dir = $('s8-source-dir').value.trim();
        cfg.sample_count = readNumberField('s8-sample-count', INFER_DEFAULTS.sample_count, { integer: true });
        cfg.sample_mode = $('s8-sample-mode').value;
        cfg.include_pts_variant = $('s8-include-pts').checked;
        cfg.conf_threshold = readNumberField('s8-conf', INFER_DEFAULTS.conf_threshold);
        cfg.iou_threshold = readNumberField('s8-iou', INFER_DEFAULTS.iou_threshold);
        cfg.mask_alpha = readNumberField('s8-mask-alpha', step8FillAlphaDefault());
        cfg.draw_boxes = $('s8-draw-boxes').checked;
        cfg.save_overlays = $('s8-save-overlays').checked;
        // Which GPU (or the CPU) runs the model; forwarded as --device.
        cfg.device = collectStepDevice(8);
    }
    return cfg;
}

// ── Progress indicators ─────────────────────────────────────────────────────

function showProgress(step, text, pct) {
    const container = document.getElementById(`s${step}-progress`);
    if (!container) return;
    container.style.display = 'block';
    const fill = document.getElementById(`s${step}-progress-fill`);
    const textEl = document.getElementById(`s${step}-progress-text`);
    if (pct === null) {
        // Indeterminate
        fill.className = 'progress-fill indeterminate';
        fill.style.width = '30%';
    } else {
        fill.className = 'progress-fill';
        fill.style.width = Math.min(100, pct) + '%';
    }
    if (textEl) {
        const elapsed = stepStartTimes[step] ? formatElapsed(Date.now() - stepStartTimes[step]) : '';
        textEl.innerHTML = `<span>${text}</span><span>${elapsed}</span>`;
    }
}

function hideProgress(step) {
    const container = document.getElementById(`s${step}-progress`);
    if (container) container.style.display = 'none';
}

function parseProgressFromLog(step, lines) {
    // Try to extract progress hints from log output
    for (let i = lines.length - 1; i >= 0; i--) {
        const line = lines[i];

        // Step 1: file parsing progress "Processing file 50 of 839"
        let m = line.match(/(\d+)\s*(?:of|\/)\s*(\d+)\s*(?:files?|sheets?)/i);
        if (m) {
            const cur = parseInt(m[1]), total = parseInt(m[2]);
            showProgress(step, `Processing ${cur} of ${total} files`, (cur / total) * 100);
            return;
        }

        // Step 1: year progress
        m = line.match(/(?:Year|Processing)\s*(20\d{2})/i);
        if (m) {
            showProgress(step, `Processing year ${m[1]}...`, null);
            return;
        }

        // Step 3: selection progress. select_images.py prints
        // `Selected {n:,} frames`, so the count can carry thousands separators.
        m = line.match(/Selected\s+([\d,]+)\s+frames?/i);
        if (m) {
            const n = parseInt(m[1].replace(/,/g, ''), 10);
            showProgress(step, `${n.toLocaleString()} frames selected`, null);
            return;
        }

        // Step 6: ultralytics prints the epoch column first ("12/500  5.6G ...").
        if (step === 6) {
            m = line.match(/^\s*(\d+)\/(\d+)\s+[\d.]+G\b/);
            if (m) {
                const cur = parseInt(m[1], 10), total = parseInt(m[2], 10);
                showProgress(step, `Epoch ${cur} of ${total}`, total ? (cur / total) * 100 : null);
                return;
            }
        }

        // Generic progress percentage
        m = line.match(/(\d+(?:\.\d+)?)\s*%/);
        if (m) {
            showProgress(step, `${m[0]} complete`, parseFloat(m[1]));
            return;
        }

        // Step 1: "Saving" or "Writing"
        if (/(?:saving|writing|output)/i.test(line)) {
            showProgress(step, 'Writing output files...', 95);
            return;
        }
    }
}

// ── Run a step ──────────────────────────────────────────────────────────────

async function runStep(step) {
    if (!state) return;

    // Save config first
    const cfg = collectConfig(step);
    // Step 3 needs at least one target label (nothing is selected by default).
    if (step === 3 && (!cfg.target_species || !cfg.target_species.trim())) {
        alert('Select at least one target label before running image selection.');
        return;
    }
    const saved = await put(`/api/project/config/${step}`, cfg);
    Object.assign(state.steps[step].config, cfg);
    // The config PUT returns the project's current name (auto names are rebuilt
    // on every save), so the topbar tracks the rename right away.
    if (saved && typeof saved.name === 'string' && saved.name) {
        state.name = saved.name;
        renderProjectName();
    }

    renderStepWarnings(step, []);
    // A state line about the last run (a restart VEGA could not account for, a
    // result that belonged to another project) is about that run and not this
    // one, so the new launch takes it off the screen.
    showStepStatusNote(step, 'warning', []);
    const data = await post(`/api/step/${step}/run`, {});
    if (data._locked) {
        return alert(data.message ||
            'Under development. Please come back later. Email Lauren for questions.');
    }
    if (data.error) return alert('Error: ' + data.error);
    // The run went ahead; anything the route wants read alongside it (an
    // auto-filled image size, a size mismatch) shows under the result line.
    renderStepWarnings(step, data.warnings);
    // Step 6: classes ticked against one source but absent from the training
    // source are dropped server-side; say which ones.
    if (step === 6 && Array.isArray(data.dropped_classes) && data.dropped_classes.length) {
        alert('Classes absent from the training source were dropped: ' + data.dropped_classes.join(', '));
    }

    state.steps[step].status = 'running';
    updateSidebar();
    setStepRunControls(step, true);

    // Show log area
    const logEl = document.getElementById(`s${step}-log`);
    if (logEl) {
        logEl.style.display = 'block';
        logEl.innerHTML = '';
    }

    // Show progress
    stepStartTimes[step] = Date.now();
    showProgress(step, 'Starting...', null);

    // For Flask stages, show service panel + elapsed
    if ([2, 4, 5].includes(step)) {
        const svc = document.getElementById(`s${step}-service`);
        if (svc) svc.style.display = 'block';
        startElapsedTimer(step);
    } else {
        startElapsedTimer(step);
    }

    startPolling(step);
}

// ── Link existing Step 1 output ─────────────────────────────────────────────

async function linkExistingStep1() {
    const ap = document.getElementById('s1-existing-ap').value.trim();
    const mc = document.getElementById('s1-existing-mc').value.trim();
    if (!ap) return alert('Please enter a path to an existing all_points.csv');

    const data = await post('/api/step/1/link', { all_points: ap, master_codes: mc });
    if (data.error) return alert('Error: ' + data.error);

    if (data.state) state = data.state;
    renderProjectName();
    updateSidebar();
    populateConfigs();
    showResult(1, true, `Linked existing output: ${data.rows ? data.rows.toLocaleString() + ' rows' : 'OK'}. Ready for Step 2.`);
    markStepActionDone(1);
}

// ── Polling ─────────────────────────────────────────────────────────────────

// The SAM3 progress widgets lived on the archived Place Points and Segment
// panels, so nothing on this page paints the step 5 SAM3 status any more. That
// route and its background driver stay on the server for the step-4 sub-app
// bridge; the client just stopped polling them forever to fill deleted nodes.

// The refusal status: the server answers with it when something is still
// going on that it will not act over. The Stop route uses it for a process
// that outlived a VEGA restart.
const HTTP_CONFLICT = 409;

// What the panel says when the server reports a step it no longer owns but
// sends no sentence of its own, which only a server older than this window
// does. It states what is known and the two ways out, and claims nothing.
const STEP_UNTRACKED_TEXT = 'VEGA no longer owns this step\'s process and cannot say how it '
    + 'ended, so nothing was recorded. Press Stop to clear the mark, or run the step again.';

// The same for a result the server refused to credit to the open project
// without naming the project it belongs to.
const STEP_NOT_THIS_PROJECT_TEXT = 'This result was not recorded against the open project, '
    + 'because the work was started somewhere else. Open the project that ran the step to see it.';

// What Stop says when the server refuses and sends no sentence of its own.
const STOP_REFUSED_TEXT = 'VEGA could not stop this step. Stop the process yourself, then '
    + 'press Stop again.';

// What follows the sentence naming the project a finished step belonged to:
// what is true of the open project now, and the two ways to make its tile
// agree with that. The running mark is the project's own record, which only
// the Stop route or a new run rewrites, so nothing here guesses at it.
const STEP_NOTHING_RUNNING_HERE_TEXT = 'Nothing is running for this project. Press Stop to '
    + 'clear the running mark, or run the step again.';

// The Stop button's explanation while it is kept pressable over a mark that
// no live process backs; its ordinary title comes back with the next live run.
const STALE_MARK_STOP_TITLE = 'Clear the running mark this step still carries. Nothing is '
    + 'running for this project, so VEGA only sets the step back to pending and keeps every '
    + 'file it wrote.';

// Steps whose state line answers something the reader pressed (a Stop the
// server refused) and must outlive the poll: the poll repeats its own
// sentence about the same run every two seconds, and used to write it over
// the refusal before anyone could read it. A hold stands until the reader's
// next action on the step (Stop or the run button pressed again, a reset) or
// until the run's state changes (it ends, is stopped, or is launched again).
const heldStepNotes = {};

function startPolling(step) {
    if (pollTimers[step]) clearInterval(pollTimers[step]);

    let logOffset = 0;

    pollTimers[step] = setInterval(async () => {
        const status = await api(`/api/step/${step}/status`);

        // Update log: server caps its in-memory buffer, but returns a
        // `total` offset + `dropped` count so we keep advancing even when
        // older lines roll off. Client also caps its DOM so long runs don't
        // bog down the page.
        const logData = await api(`/api/step/${step}/log?offset=${logOffset}`);
        if (logData.lines && logData.lines.length > 0) {
            const logEl = document.getElementById(`s${step}-log`);
            if (logEl) {
                logEl.style.display = 'block';
                // DOM-side rolling window. If a single response carries more
                // lines than we want to keep, clip to the tail before
                // rendering so we don't build DOM we're about to drop.
                const MAX_DOM_LINES = 2500;
                let incoming = logData.lines;
                let predrop = 0;
                if (incoming.length > MAX_DOM_LINES) {
                    predrop = incoming.length - MAX_DOM_LINES;
                    incoming = incoming.slice(predrop);
                }
                const serverDropped = logData.dropped || 0;
                const totalDropped = serverDropped + predrop;
                if (totalDropped > 0) {
                    const dropDiv = document.createElement('div');
                    dropDiv.className = 'log-line log-line-dropped';
                    dropDiv.textContent = `... (${totalDropped} earlier lines rolled off)`;
                    logEl.appendChild(dropDiv);
                }
                for (const line of incoming) {
                    const div = document.createElement('div');
                    div.className = 'log-line';
                    div.textContent = line;
                    logEl.appendChild(div);
                }
                // Final trim in case we were already near the cap.
                while (logEl.childElementCount > MAX_DOM_LINES) {
                    logEl.removeChild(logEl.firstChild);
                }
                logEl.scrollTop = logEl.scrollHeight;
                parseProgressFromLog(step, incoming);
            }
            logOffset = logData.offset;
        } else if (typeof logData.offset === 'number' && logData.offset > logOffset) {
            // No new lines this tick but server advanced the total; keep in sync.
            logOffset = logData.offset;
        }

        // Flask stage health
        if (status.port) {
            const healthDot = document.getElementById(`s${step}-health`);
            const healthText = document.getElementById(`s${step}-health-text`);

            if (status.healthy) {
                if (healthDot) healthDot.className = 'health-dot healthy';
                if (healthText) healthText.textContent = 'Service running on port ' + status.port;
                // (Live launch path is openStepService; no per-step open-window
                // button exists, so nothing to enable here.)
            } else if (status.running) {
                if (healthDot) healthDot.className = 'health-dot checking';
                if (healthText) healthText.textContent = 'Starting up (loading models)...';
            } else {
                if (healthDot) healthDot.className = 'health-dot unhealthy';
                if (healthText) healthText.textContent = 'Service stopped';
            }
        }

        // What the server said about this poll, in its own words: a process
        // that outlived a VEGA restart, a restart it cannot account for, a
        // result that belongs to another project. Every one of these used to be
        // read off the wire and thrown away.
        const notes = Array.isArray(status.warnings)
            ? status.warnings.filter(w => typeof w === 'string' && w.trim())
            : [];

        // CLI completion
        if (status.running) {
            // A process the runner no longer owns keeps the tile running and
            // keeps this poll going; the sentence says it is out of VEGA's
            // hands and where its output is still being written. A line the
            // reader's own Stop press put there stays until they act again.
            if (notes.length && !heldStepNotes[step]) showStepStatusNote(step, 'warning', notes);
        } else {
            // The run's state changed, so whatever was held about it is over.
            releaseStepStatusNote(step);
            clearInterval(pollTimers[step]);
            delete pollTimers[step];
            stopElapsedTimer(step);
            setStepRunControls(step, false);

            if (status.untracked) {
                // The runner holds no process for a step this project still has
                // marked running, which is what a VEGA restart leaves behind.
                // Whether that process finished or died cannot be read from
                // here, so the mark is left exactly as it is and the reader is
                // told what is known and what to do next.
                hideProgress(step);
                showStepStatusNote(step, 'warning',
                    notes.length ? notes : [STEP_UNTRACKED_TEXT]);
                keepStopReachable(step);
            } else if (status.completed === false) {
                // The server answered that this project did not do this work:
                // another project started the step, or the evaluation scored a
                // run this project does not own. A zero exit code is still
                // there, so nothing but "completed" can tell the two apart.
                // The tile's running mark is the project's own record, left
                // as it was; the line says nothing runs here and Stop stays
                // pressable, because pressing it is how that mark is cleared.
                hideProgress(step);
                showStepStatusNote(step, 'warning',
                    (notes.length ? notes : [STEP_NOT_THIS_PROJECT_TEXT])
                        .concat([STEP_NOTHING_RUNNING_HERE_TEXT]));
                keepStopReachable(step);
            } else if (status.completed || status.exit_code === 0) {
                // A line about the run (a Stop it refused, a restart) is over
                // with the run; it must not sit beside the success badge.
                showStepStatusNote(step, 'warning', []);
                const proj = await api('/api/project/state');
                if (proj.state) {
                    state = proj.state;
                    renderProjectName();
                    updateSidebar();
                    populateConfigs();
                }
                showProgress(step, 'Complete!', 100);
                setTimeout(() => hideProgress(step), 3000);
                if (step === 3 && status.site_summary) {
                    renderStep3SiteSummary(step, status.site_summary);
                } else {
                    showResult(step, true, 'Step completed successfully.');
                }
                markStepActionDone(step);
                // Step-specific post-completion hooks: each card reloads for
                // the run that just finished. The runs card lives inside the
                // closed-by-default "All trained models" disclosure on the
                // Evaluate panel now, so a Train Model finish only reloads it
                // when a reader already has that disclosure open; otherwise
                // opening it later (openStep7Panel, or its own toggle) is
                // what loads the run that just finished.
                if (step === 6 && step7AllModelsOpen()) openStep6RunsCard();
                if (step === 7) loadStep7ReportForSelectedRun();
                if (step === 8) refreshStep8GalleryCard(newestStep8OutDir());
            } else if (status.exit_code !== null && status.exit_code !== 0) {
                showStepStatusNote(step, 'warning', []);
                state.steps[step].status = 'error';
                updateSidebar();
                hideProgress(step);
                showResult(step, false, `Step failed with exit code ${status.exit_code}. Check the log above.`);
            }
        }
    }, 2000);
}

function showResult(step, success, message) {
    const el = document.getElementById(`s${step}-result`);
    if (!el) return;
    el.innerHTML = `<div class="badge ${success ? 'badge-success' : 'badge-error'}" style="margin-top:12px">${message}</div>`;
}

// The two voices a step's state line speaks in: amber for something the reader
// has to know about a run that is over, red for an action VEGA refused to take.
const STEP_NOTE_COLORS = { warning: 'var(--warning)', error: 'var(--error)' };

/**
 * The element that carries a step's state line, made on first use.
 *
 * It is a sibling of the panel's warnings container rather than a child of it,
 * because that container is hidden until a run route fills it and is emptied
 * by the next launch; a state line about the run that just ended has to
 * outlive both.
 * @param {number|string} step the step key
 * @returns {Element|null} null when the panel carries neither anchor
 */
function stepStatusNoteEl(step) {
    const id = `s${step}-status-note`;
    const found = document.getElementById(id);
    if (found) return found;
    const anchor = document.getElementById(`s${step}-warnings`)
        || document.getElementById(`s${step}-result`);
    if (!anchor) return null;
    const el = document.createElement('div');
    el.id = id;
    if (anchor.parentNode) anchor.parentNode.insertBefore(el, anchor.nextSibling);
    else anchor.appendChild(el);
    return el;
}

/**
 * Show what the server said about a step, in the server's own words.
 *
 * The status poll and the Stop route both answer with plain sentences that are
 * the whole truth of a press: a training that outlived a VEGA restart, a
 * result that belongs to another project, a stop the orchestrator could not
 * make. Each sentence is a state line at the reading size (.hint is 13 px
 * Inter, line height 1.5) in the colour its severity asks for, never a badge,
 * because a badge is a 12 px chip and these are explanations.
 *
 * @param {number|string} step the step key the sentences are about
 * @param {string} tone 'warning' or 'error'
 * @param {Array<string>|string} sentences what the server said; an empty list
 *   clears the line
 * @param {boolean} [hold=false] true when the line answers the reader's own
 *   press and must not be replaced by the poll until they act again or the
 *   run's state changes
 * @returns {void}
 */
function showStepStatusNote(step, tone, sentences, hold) {
    const list = (Array.isArray(sentences) ? sentences : [sentences])
        .filter(s => typeof s === 'string' && s.trim());
    // An empty list clears the line and with it any hold; a held line is
    // recorded before the flicker guard so a repeat of the same text still
    // holds.
    if (list.length && hold) heldStepNotes[step] = true;
    else if (!list.length) releaseStepStatusNote(step);
    const el = stepStatusNoteEl(step);
    if (!el) return;
    const color = STEP_NOTE_COLORS[tone] || STEP_NOTE_COLORS.warning;
    const html = list.map(s =>
        `<div class="hint" style="color:${color};margin-top:8px">${escHtmlNav(s)}</div>`).join('');
    // The poll repeats every two seconds and says the same thing each time;
    // rewriting the node would flicker the line for no new fact.
    if (el.innerHTML === html) return;
    el.innerHTML = html;
    el.style.display = list.length ? 'block' : 'none';
    // Two states never stand together: a line saying nothing was recorded here,
    // beside a badge saying the step completed, is the contradiction this fixes.
    if (list.length) {
        const resultEl = document.getElementById(`s${step}-result`);
        if (resultEl && resultEl !== el) resultEl.innerHTML = '';
    }
}

/**
 * Let the poll write a step's state line again.
 *
 * The hold a refused Stop put on the line ends here: on the reader's next
 * action on the step and whenever the run's state changes.
 * @param {number|string} step the step key
 * @returns {void}
 */
function releaseStepStatusNote(step) {
    delete heldStepNotes[step];
}

/**
 * Keep Stop pressable on a step whose running mark no live process backs.
 *
 * setStepRunControls pairs the two buttons: a step that is not running hides
 * Stop and enables Start. A step left marked running by a restart, or by a
 * run another project owned, needs both, because the reader is told to press
 * Stop to clear the mark or to run the step again. While the button stands
 * for that, its title says so; setStepRunControls gives the live title back
 * with the next run.
 * @param {number|string} step the step key
 * @returns {void}
 */
function keepStopReachable(step) {
    const stopBtn = document.getElementById(`s${step}-stop-btn`);
    if (!stopBtn) return;
    stopBtn.style.display = '';
    if (stopBtn.dataset.liveTitle === undefined) stopBtn.dataset.liveTitle = stopBtn.title || '';
    stopBtn.title = STALE_MARK_STOP_TITLE;
}

// Step 3 completion readout: select_images.py writes output/selection_summary.json
// (schema: per_site, per_species, cap_frac, cap_ok, cap_satisfiable, n_selected,
// shortfalls) alongside the .txt summary; the status poll attaches it to the
// completion payload as `site_summary`. Renders a Sites table + cap line +
// per-species shortfall reasons in place of the bare "Step completed
// successfully." text. Plain divs/table, no library.
const STEP3_SHORTFALL_REASON_TEXT = {
    'cap-limited': 'capped by the max share per site',
    'allocation-limited': 'the per-site / per-year spread could not place enough frames',
    'pool exhausted': 'no more eligible frames exist for this label',
};

function renderStep3SiteSummary(step, summary) {
    const el = document.getElementById(`s${step}-result`);
    if (!el) return;
    const total = summary.n_selected || 0;
    const perSite = summary.per_site || {};
    const siteRows = Object.keys(perSite)
        .map(site => [site, perSite[site]])
        .sort((a, b) => b[1] - a[1]);
    const rowsHtml = siteRows.map(([site, n]) => {
        const share = total ? (n / total * 100) : 0;
        return `<tr>
            <td>${escHtmlNav(site)}</td>
            <td style="text-align:right">${n.toLocaleString()}</td>
            <td style="text-align:right">${share.toFixed(1)}%</td>
        </tr>`;
    }).join('');
    const capPct = ((summary.cap_frac != null ? summary.cap_frac : 0.25) * 100).toFixed(0);
    let capBadge;
    if (summary.cap_frac >= 1.0) {
        capBadge = '';
    } else if (summary.cap_ok) {
        capBadge = `<div class="badge badge-success" style="margin-top:8px"
            title="No site holds more than the ${capPct}% share you set in Max share per site.">Site cap satisfied (&le;${capPct}% per site)</div>`;
    } else if (summary.cap_satisfiable === false) {
        capBadge = `<div class="badge badge-warning" style="margin-top:8px"
            title="Too few sites carry these labels for the ${capPct}% cap to hold on every one at once; every site still gets at least one frame (the inclusion floor), so the cap could not be met exactly. The selection used the tightest cap the data allowed instead.">Cap not satisfiable with so few sites (tightest even split used)</div>`;
    } else {
        capBadge = `<div class="badge badge-warning" style="margin-top:8px"
            title="At least one site exceeded the ${capPct}% share you set in Max share per site.">Site cap exceeded (target &le;${capPct}% per site)</div>`;
    }
    const shortfalls = summary.shortfalls || [];
    // Each shortfall is a hint line: 13 px, --text-dim, and focusable, so its
    // reason opens from the keyboard as well as on hover (spec sections 3, 7).
    const shortfallHtml = shortfalls.length ? `
        <div style="margin-top:8px">
            ${shortfalls.map(s => {
                const reasonText = STEP3_SHORTFALL_REASON_TEXT[s.reason] || s.reason;
                // The species reads as a species: a binomial in italic, the
                // way every other name the window renders does.
                return `<div class="hint" tabindex="0"
                    title="${escAttr(s.species)} reached ${s.achieved} of its ${s.target}-instance target: ${escAttr(reasonText)}.">${vegaSpeciesNameHtml(s.species)}: ${s.achieved}/${s.target} (${escHtmlNav(reasonText)})</div>`;
            }).join('')}
        </div>` : '';
    // No-re-screening line (2026-09-27 VEGA annotator tidy-up plan): present
    // only on a run made with --prior-screening (both keys set, even when a
    // count is 0), so a legacy run or one made before this feature shipped
    // shows nothing extra here (no filler status for a flag that never ran).
    const screeningHtml = step3ScreeningLineHtml(summary);
    el.innerHTML = `
        <div style="margin-top:12px" title="Per-site frame counts from this run's selection_summary.json (${total.toLocaleString()} frames selected total).">
            <table class="remap-table">
                <thead><tr>
                    <th>Site</th>
                    <th style="text-align:right">Frames</th>
                    <th style="text-align:right">Share</th>
                </tr></thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        </div>
        ${capBadge}
        ${shortfallHtml}
        ${screeningHtml}`;
}

/**
 * The no-re-screening line for the Step 3 completion panel (2026-09-27 VEGA
 * annotator tidy-up plan): how many eligible images this run skipped because
 * every target label was already screened for them in another project, and
 * how many of the CHOSEN images carry a partial screening (some labels
 * already covered) and so show another project's masks as locked previews
 * in the annotator.
 *
 * @param {object} summary - this run's selection_summary.json.
 * @returns {string} one hint-line <div>, or '' when Step 3 ran without
 *   --prior-screening (both keys absent) or found nothing to report (both
 *   counts are 0).
 */
function step3ScreeningLineHtml(summary) {
    if (summary.skipped_fully_screened == null && summary.partly_screened == null) return '';
    const skipped = summary.skipped_fully_screened || 0;
    const partly = (summary.partly_screened && summary.partly_screened.count) || 0;
    if (skipped === 0 && partly === 0) return '';
    const clauses = [];
    if (skipped > 0) {
        clauses.push(`Skipped ${skipped.toLocaleString()} image${skipped === 1 ? '' : 's'} `
            + `already screened for every target label in other projects`);
    }
    if (partly > 0) {
        // "results", not "outlines": a screened label can mean another
        // project found and drew a mask, or looked and found nothing there
        // (not_found) - the second case has no outline to preview, only the
        // fact that it was already checked, so the annotator shows whichever
        // is true rather than promising a mask that may not exist.
        clauses.push(`${partly.toLocaleString()} chosen image${partly === 1 ? '' : 's'} `
            + `${partly === 1 ? 'was' : 'were'} screened for some labels in other `
            + `projects and ${partly === 1 ? 'shows' : 'show'} those results in the `
            + `annotator`);
    }
    const text = clauses.join('; ') + '.';
    return `<div class="hint" tabindex="0" style="margin-top:8px"
        title="VICARIUS checked the label coverage matrix before selecting frames, so nobody is asked to redo a screening another project already recorded.">${escHtmlNav(text)}</div>`;
}

// ── Open service step (single-button flow: run-if-needed -> open tab) ──────

// One button per service step (2, 4, 5): if not running yet, start it. Show a
// loading overlay while we poll status for up to ~10s until the port is healthy
// (or at least running). Only window.open() once it's actually up, never a
// dead tab. If it never comes up, surface an error in the overlay instead.
const OPEN_UI_LABELS = { 2: 'Mapping Editor', step4test: 'Annotator', step4loop: 'Refine Review', editmasks: 'Edit Masks' };

async function openStepService(step) {
    if (!state) return;
    const st = state.steps[step];
    const status = st && st.status;

    showOpenUiOverlay(OPEN_UI_LABELS[step] || `Step ${stepDisplayNum(step)}`);

    // If not running yet (and not already completed), start it.
    if (status !== 'running') {
        try {
            await runStep(step);
        } catch (e) {
            setOpenUiError('Failed to start the service: ' + (e && e.message || e));
            return;
        }
    }

    // Poll status for up to ~10 seconds until healthy or at least running.
    const deadline = Date.now() + 10000;
    let port = null;
    let healthy = false;
    while (Date.now() < deadline) {
        let s;
        try { s = await api(`/api/step/${step}/status`); } catch (e) { s = null; }
        if (s && s.port) port = s.port;
        if (s && s.healthy) { healthy = true; break; }
        // "running but not yet healthy" keeps us waiting until the deadline so
        // we don't open a tab before the sub-app's HTTP server is actually up.
        await new Promise(r => setTimeout(r, 400));
    }

    // Fall back to the config-stored port if status didn't surface one yet.
    if (!port) {
        port = state.steps[step] && state.steps[step].config && state.steps[step].config.port;
    }

    if (!port) {
        setOpenUiError('Service port not set yet; the UI did not come up in time.');
        return;
    }
    if (!healthy) {
        // Port known but health never confirmed within the window. Don't open a
        // potentially-dead tab; let the user retry once it's warmed up.
        setOpenUiError(`The ${OPEN_UI_LABELS[step] || 'service'} UI is still starting (port ${port}) and did not respond in ~10s.`);
        return;
    }

    hideOpenUiOverlay();
    const url = `http://localhost:${port}/`;
    // Open every sub-tool UI in a real new WINDOW, never a tab, so this main
    // orchestrator window is always there to return to (Lauren: never a tab).
    // One named popup per step lets several sub-tools coexist without clobbering
    // each other. No 'noopener': we keep the handle so markDone()/saveAndClose()
    // can close the window on cleanup (guard for null if a popup blocker withholds it).
    const win = window.open(url, `reefpointseg_step${step}`,
        'popup,width=1500,height=980,scrollbars=yes,resizable=yes');
    if (win) { serviceWindows[step] = win; try { win.focus(); } catch (e) {} }
}

// step4test (combined annotator) - non-chain tile with its own launch/status.
// 4.test now fully replaces Step 4: /run routes Step 3's chosen images in the
// background (driving placePoints headless) then launches the combined UI. We
// POST /run, then either open immediately (reuse -> ui_ready) or poll
// route_status, showing routing progress in the open-UI overlay.
// Collect the #panel-step4test config block (Combined Step-4+5 settings). Sent
// with /run so _launch_step4test_ui can set the ENV CONTRACT vars,
// each falling back to the inherited Step-5 saved config. All ids are optional
// (panel may predate this build), so every read is null-guarded.
function collectStep4testConfig() {
    const cfg = {};
    const chk = (id, dflt) => { const el = document.getElementById(id); return el ? el.checked : dflt; };
    const val = (id) => { const el = document.getElementById(id); return el ? el.value : undefined; };
    const num = (id) => { const v = val(id); const n = parseFloat(v); return (v === undefined || isNaN(n)) ? undefined : n; };
    cfg.target_species_only = chk('s4test-target-only', true);
    cfg.reference_default = chk('s4test-reference', true);
    const b = val('s4test-batch-size'); if (b !== undefined && b !== '') cfg.review_batch_size = b;
    const dev1 = val('s4test-tracker-device'); if (dev1) cfg.sam3_device_tracker = dev1.trim();
    const dev2 = val('s4test-exemplar-device'); if (dev2) cfg.sam3_device_exemplar = dev2.trim();
    const conf = num('s4test-confidence'); if (conf !== undefined) cfg.confidence_threshold = conf;
    const minA = num('s4test-min-area'); if (minA !== undefined) cfg.min_mask_area_px = minA;
    const mrg = num('s4test-merge-dist'); if (mrg !== undefined) cfg.merge_distance_px = mrg;
    const ov = val('s4test-overlap'); if (ov) cfg.overlap_strategy = ov;
    const thin = num('s4test-thin-ratio'); if (thin !== undefined) cfg.thin_mask_ratio = thin;
    const eps = num('s4test-simplify'); if (eps !== undefined) cfg.polygon_simplify_epsilon = eps;
    const cd = val('s4test-clip-dir'); if (cd && cd.trim()) cfg.clip_dir = cd.trim();
    cfg.symlink_images = chk('s4test-symlink', true);
    cfg.lores_mode = chk('s4test-lores', true);
    return cfg;
}

// Every launch popup is drawn from ONE skeleton, so the id contract
// paintStep4testBoot depends on (boot-title, boot-msg, boot-bar, boot-hint)
// is written in exactly one place instead of three near-copies that could
// drift apart.
// Palette matches the canonical VICARIUS tokens (D4, 2026-09-02): magenta
// #E20074 progress fill, #11111a surface card, #0a0a0f page ground, #F5EDF0
// text, #b0a7ab dim text, #1d1d26 track/border-dim, #2a2a38 border - the same
// values templates/annotate.html carries, since this skeleton draws the
// identical boot overlay for every other launched sub-window.
function bootSkeletonHtml({ title, msg, hint } = {}) {
    return '<div style="text-align:center;max-width:460px;padding:24px;background:#11111a;'
      + 'border:1px solid #2a2a38;border-radius:3px">'
      + '<div style="font-size:15px" id="boot-title">' + escHtmlNav(title || '') + '</div>'
      + '<div style="margin-top:10px;font-size:12px;color:#b0a7ab;line-height:1.5" id="boot-msg">'
      + escHtmlNav(msg || '') + '</div>'
      + '<div style="margin-top:14px;height:6px;border-radius:2px;background:#1d1d26;overflow:hidden">'
      + '<div id="boot-bar" style="height:100%;width:0%;background:#E20074;transition:width .3s"></div></div>'
      + '<div style="margin-top:14px;font-size:11px;color:#b0a7ab;line-height:1.5" id="boot-hint">'
      + escHtmlNav(hint || '') + '</div></div>';
}

const BOOT_BODY_CSS = 'margin:0;font-family:system-ui,-apple-system,sans-serif;'
    + 'background:#0a0a0f;color:#F5EDF0;display:flex;align-items:center;justify-content:center;height:100vh';

// Is a launch popup's document ours to paint? window.open('', name) on a tool
// window that is ALREADY open hands back the running tool rather than a blank
// page, and that window is cross-origin (it serves on its own port), so the
// title write throws and every later paint silently no-ops. Without this flag a
// failing relaunch showed the operator nothing at all, anywhere. Keyed the same
// way as serviceWindows, and rewritten by each placeholder on every launch.
const bootPaintable = { step4test: false, step4loop: false, editmasks: false };

// Where a launch failure gets reported, per flow.
const ANNOTATOR_UI = {
    step4test: {
        btnId: 's4test-run-btn', errId: 's4test-open-error',
        title: 'Could not start the Combined Annotator',
        retry: () => openStep4test(),
    },
    step4loop: {
        btnId: 's4loop-open-btn', errId: 's4loop-open-error',
        title: 'Could not start the Refine Review',
        retry: () => openStep4loop(),
    },
    editmasks: {
        btnId: 's4em-open-btn', errId: 's4em-open-error',
        title: 'Could not start Edit Masks',
        retry: () => openEditMasks(),
    },
};

// One decision point for every annotator launch failure: paint it into the popup
// while that popup is ours to write, otherwise put it inline beside the Open
// button. Either way the operator gets exactly one message, and never zero.
function annotatorLaunchFailed(key, win, msg, tip, paintable) {
    const ui = ANNOTATOR_UI[key];
    if (!ui) return;
    // Prefer the flag the caller snapshotted when it opened its popup. The shared
    // map is only the fallback: a second launch of the same flow rewrites it while
    // this one is still in flight, and this failure belongs to THIS popup.
    const canPaint = typeof paintable === 'boolean' ? paintable : bootPaintable[key];
    if (canPaint && win && !win.closed) {
        paintStep4testBoot(win, { title: ui.title, msg, error: true, retry: ui.retry });
        return;
    }
    setAnnotatorOpenError(ui.btnId, ui.errId, msg, tip);
}

async function openStep4test() {
    clearAnnotatorOpenError('s4test-open-error');
    // Open the annotator popup NOW - synchronously, inside this click handler -
    // so the browser ties it to the user gesture. Routing + the SAM3 load can
    // take ~a minute; deferring window.open() until after that await chain (as
    // this used to) means the popup blocker silently kills it and "Open" appears
    // to do nothing. The popup shows a loading page until we navigate it to the
    // annotator once it is healthy, or we close it on failure.
    const win = openStep4testPlaceholder();
    // Snapshot the paintability of THIS popup now: clicking Open twice rewrites
    // the shared flag while the first launch is still in flight.
    const paintable = bootPaintable['step4test'];
    if (!win) {
        annotatorLaunchFailed('step4test', null,
            'Your browser blocked the annotator window. Allow pop-ups for this page, then click Open Combined Annotator again.',
            'The browser refused the new window, so there is nowhere to show the start. Allow pop-ups for this page and click Open again.',
            paintable);
        return;
    }
    serviceWindows['step4test'] = win;
    let r = null;
    try {
        r = await post('/api/step/step4test/run', { config: collectStep4testConfig() });
    } catch (e) {
        // Keep the popup open on failure: it is the only loader now, so closing it
        // would leave the operator with no message at all.
        annotatorLaunchFailed('step4test', win, 'Failed to start: ' + (e && e.message || e), null, paintable);
        return;
    }
    if (r && r.error && !/already running/i.test(r.error)) {
        annotatorLaunchFailed('step4test', win, 'Failed to start: ' + r.error, null, paintable);
        return;
    }
    await finishStep4testOpen(r, win);
}

// Open the annotator popup immediately (same user gesture as the click) so the
// popup blocker lets it through, then paint a loading message into it. We
// navigate THIS window to the service once it is healthy - never a second
// window.open() from inside an async callback, which the browser would block.
function openStep4testPlaceholder() {
    let win;
    bootPaintable['step4test'] = false;
    try {
        win = window.open('', 'reefpointseg_step4test',
            'popup,width=1500,height=980,scrollbars=yes,resizable=yes');
    } catch (e) { win = null; }
    if (!win) return null;
    try {
        win.document.title = 'Combined Annotator - starting';
        win.document.body.style.cssText = BOOT_BODY_CSS;
        // Live region, not a frozen sentence. This window is the one the operator
        // actually watches, so routing progress has to show HERE - it used to paint
        // "up to a minute" once and never update, so a 20-minute pass looked hung
        // while the real progress bar ticked away in the main tab behind it.
        win.document.body.innerHTML = bootSkeletonHtml({
            title: 'Starting the Combined Annotator',
            msg: 'Routing the chosen images. This runs about one image per second, so a large '
               + 'selection can take several minutes. The tool loads here automatically when it is ready.',
            hint: 'Closing this window cancels the start. Reopening resumes where it stopped.',
        });
        // The document accepted our writes, so this popup is ours to paint.
        bootPaintable['step4test'] = true;
    } catch (e) { bootPaintable['step4test'] = false; }
    try { win.focus(); } catch (e) {}
    return win;
}

// Paint launch progress into the POPUP the operator is actually looking at.
// This is the ONE loader for all three annotator launches (Combined Annotator,
// Refine Review, Edit Masks): they no longer raise the SPA modal, so successes,
// progress AND failures all read here. Silently no-ops once the popup has
// navigated to the tool (cross-document access throws) or was closed.
// `error: true` paints the bar red; `retry` wires a Retry link in the popup.
function paintStep4testBoot(win, { title, msg, pct, error, retry } = {}) {
    if (!win || win.closed) return;
    try {
        const d = win.document;
        if (title) { const t = d.getElementById('boot-title'); if (t) t.textContent = title; }
        if (msg) { const m = d.getElementById('boot-msg'); if (m) m.textContent = msg; }
        if (typeof pct === 'number') {
            const b = d.getElementById('boot-bar');
            if (b) b.style.width = Math.max(0, Math.min(100, pct)) + '%';
        }
        const bar = d.getElementById('boot-bar');
        if (bar) bar.style.background = error ? '#FF4D6D' : '#E20074';
        // A failure replaces the whole progress story. The hint underneath talks
        // about a start that is no longer running ("Reopening resumes where it
        // stopped"), so it goes away rather than sitting under the error.
        const hint = d.getElementById('boot-hint');
        if (error && hint) hint.textContent = '';
        let r = d.getElementById('boot-retry');
        if (error && retry) {
            if (!r) {
                r = d.createElement('a'); r.id = 'boot-retry'; r.href = '#';
                r.textContent = 'Retry';
                r.style.cssText = 'display:inline-block;margin-top:14px;color:#F5EDF0;border:1px solid #2a2a38;border-radius:2px;padding:6px 14px;text-decoration:none';
                r.title = 'Try opening the annotator again.';
                (hint ? hint.parentNode : d.body).appendChild(r);
            }
            r.onclick = (e) => { e.preventDefault(); retry(); };
            r.style.display = 'inline-block';
        } else if (r) {
            r.style.display = 'none';
        }
    } catch (e) {}
}

// The one failure the popup cannot report is "there is no popup": a blocked
// window, or one closed while the tool was still starting. Those get an inline
// note beside the Open button instead, so the operator still sees exactly one
// message about the launch.
function setAnnotatorOpenError(btnId, errId, text, tip) {
    const btn = document.getElementById(btnId);
    let el = document.getElementById(errId);
    if (!el) {
        if (!btn) return;
        el = document.createElement('div');
        el.id = errId;
        el.className = 'badge badge-error';
        el.style.cssText = 'margin-top:10px;display:block;white-space:normal';
        (btn.parentNode || document.body).appendChild(el);
    }
    // Tooltip on every call, not only on creation: the same note is reused for
    // several failures, and a stale tooltip would describe the wrong one.
    el.title = tip || 'The launch could not finish. Read the message, fix it, then click Open again.';
    el.style.display = 'block';
    el.textContent = text;
}

function clearAnnotatorOpenError(errId) {
    const el = document.getElementById(errId);
    if (!el) return;
    el.style.display = 'none';
    el.textContent = '';
}

// The operator closed the launch window. The UI is the only way in, so that has to
// mean the work stops: cancel the routing pass, free the GPU, and make sure no
// annotator gets launched (and no LIVE annotator gets killed) by a pass nobody is
// watching any more.
async function cancelStep4testRoute() {
    try { await post('/api/step/step4test/cancel', {}); } catch (e) {}
}

const ROUTE_LIVE_PHASES = ['launching', 'routing', 'routing_configure',
                           'routing_ocr', 'exporting', 'route_ready'];

// Nothing runs invisibly. A routing pass lives in a background thread on the
// server, so it survives closing the launch window AND a full page reload - and
// used to leave no trace anywhere in the UI while it held the GPU. Surface it on
// the tile, with a Stop, for as long as it is alive.
async function renderStep4testRouteBanner(routeStatus) {
    const el = document.getElementById('s4test-route-banner');
    if (!el) return;
    // pollToolTiles() already read route_status for this tick and hands it in,
    // so the banner and the tile caption share one request. Every other caller
    // (Stop Routing, the launch tails) passes nothing and gets its own read.
    let rs = routeStatus;
    if (rs === undefined) {
        try { rs = await api('/api/step/step4test/route_status'); } catch (e) { rs = null; }
    }

    // The banner has two personalities: routing progress (bar, GPU hint, Stop
    // Routing) and crashed-process alarm (red border, error text only). Every
    // pass sets ALL mode-specific elements so a state flip never leaves
    // leftovers from the other personality.
    const dot = document.getElementById('s4test-route-dot');
    const barwrap = document.getElementById('s4test-route-barwrap');
    const hint = document.getElementById('s4test-route-hint');
    const actions = document.getElementById('s4test-route-actions');

    // A crashed annotator (server-side liveness check flips ui_ready to
    // error + died) must be visible on the tile, not only inside the launch
    // overlay: without this the tile looks idle while the port is dead.
    if (rs && rs.phase === 'error' && rs.died) {
        el.style.display = 'block';
        el.style.borderColor = 'var(--error, #c0392b)';
        const dtxt = document.getElementById('s4test-route-text');
        if (dtxt) dtxt.textContent = (rs.error || 'The annotator process died.')
            + ' Use Open to relaunch.';
        if (dot) dot.className = 'health-dot unhealthy';
        if (barwrap) barwrap.style.display = 'none';
        if (hint) hint.style.display = 'none';
        if (actions) actions.style.display = 'none';
        return;
    }
    el.style.borderColor = 'var(--border)';
    if (dot) dot.className = 'health-dot checking';
    if (barwrap) barwrap.style.display = '';
    if (hint) hint.style.display = '';
    if (actions) actions.style.display = '';
    if (!rs || !ROUTE_LIVE_PHASES.includes(rs.phase)) { el.style.display = 'none'; return; }

    const p = rs.processed || 0, t = rs.total || 0;
    const pct = t > 0 ? Math.round((p / t) * 100) : 0;
    el.style.display = 'block';
    const txt = document.getElementById('s4test-route-text');
    if (txt) {
        txt.textContent = t
            ? `Routing chosen images… ${p}/${t} (${pct}%), about ${Math.max(1, Math.round((t - p) / 60))} min left`
            : (rs.message || 'Routing chosen images…');
    }
    const bar = document.getElementById('s4test-route-bar');
    if (bar) bar.style.width = pct + '%';
}

async function stopStep4testRoute() {
    await cancelStep4testRoute();
    await renderStep4testRouteBanner();
}

function closeStep4testWin(win) {
    try { if (win) win.close(); } catch (e) {}
    if (serviceWindows['step4test'] === win) delete serviceWindows['step4test'];
}

// Shared tail for openStep4test: given the /run state, either open the combined
// UI now (ui_ready) or poll route_status until routing completes (or errors).
async function finishStep4testOpen(r, win) {
    const svc = document.getElementById('s4test-service');
    if (svc) svc.style.display = 'block';

    // ui_ready means the annotator is already up (reuse): open right away.
    if (r && r.phase === 'ui_ready') {
        await openStep4testWindow(r.ui_port || null, win);
        return;
    }

    // Routing in progress: poll route_status and paint it into the popup, which
    // is the one loader for this launch (the SPA modal is not raised any more).
    while (true) {
        // Closed window == cancel. Not just "stop polling": the routing thread would
        // otherwise keep the GPU and, on finishing, kill whatever annotator is up to
        // launch one nobody is waiting for.
        if (win && win.closed) {
            await cancelStep4testRoute();
            closeStep4testWin(win);
            renderStep4testRouteBanner();
            return;
        }
        let rs; try { rs = await api('/api/step/step4test/route_status'); } catch (e) { rs = null; }
        if (rs) {
            if (rs.phase === 'error') {
                // When the annotator process itself died, the API attaches the
                // last log lines; show the tail end (where the traceback is)
                // so the cause is visible without hunting for the service log.
                const tail = (rs.log_tail || []).slice(-3).join('  ');
                annotatorLaunchFailed('step4test', win,
                    (rs.error || 'Routing failed.') + (tail ? '  ' + tail : ''));
                renderStep4testRouteBanner();
                return;
            }
            if (rs.phase === 'cancelled') {
                closeStep4testWin(win);
                renderStep4testRouteBanner();
                return;
            }
            if (rs.phase === 'ui_ready') {
                if (rs.warnings && rs.warnings.length) {
                    window._step4testWarnings = rs.warnings;
                }
                await openStep4testWindow(rs.ui_port || null, win);
                return;
            }
            const processed = rs.processed || 0, total = rs.total || 0;
            const pct = total > 0 ? Math.round((processed / total) * 100) : 0;
            // The popup is the window she is actually watching, so progress goes
            // there and nowhere else.
            let progressMsg = total
                ? `About ${Math.max(1, Math.round((total - processed) / 60))} min left at ~1 image/sec. `
                  + 'The annotator opens here automatically when routing finishes.'
                : 'Detecting reference points from your Step 3 selections.';
            if (rs.warnings && rs.warnings.length) progressMsg += '  ' + rs.warnings.join('  ');
            paintStep4testBoot(win, {
                title: total ? `Routing chosen images… ${processed}/${total}` : 'Routing chosen images…',
                msg: progressMsg,
                pct,
            });
        }
        await new Promise(res => setTimeout(res, 1500));
    }
}

// Wait for the combined sub-app to report healthy, then point the already-open
// popup (opened synchronously on the click) at it. We NAVIGATE the existing
// window rather than calling window.open() here: a window.open() from inside this
// async callback runs outside the click's user activation, so the browser blocks
// it, which was the "Open does nothing" bug.
async function openStep4testWindow(uiPort, win) {
    // Wait on the PROCESS, not on a fixed clock. A cold SAM3 load can take well
    // over a minute, and a fixed 120s deadline gave up 4 seconds before a healthy
    // annotator finished loading (2026-07-14). So: keep waiting while the annotator
    // process is alive, up to the same 10-minute ceiling the server gives SAM3, and
    // fail fast when NOTHING is running behind the port instead of burning it.
    const ceiling = Date.now() + 600000;
    let port = uiPort || null, healthy = false, waited = 0, deadProbes = 0;
    while (Date.now() < ceiling) {
        // Closed mid-launch == stop launching. Leaving a SAM3 process loading for a
        // window that no longer exists is exactly the kind of invisible work that
        // has no way to be seen or stopped from the UI.
        if (win && win.closed) {
            try { await post('/api/step/step4test/stop', {}); } catch (e) {}
            closeStep4testWin(win);
            renderStep4testRouteBanner();
            return;
        }
        let s; try { s = await api('/api/step/step4test/status'); } catch (e) { s = null; }
        if (s && s.port) port = s.port;
        if (s && s.healthy) { healthy = true; break; }
        paintStep4testBoot(win, {
            title: 'Loading the SAM3 model…',
            msg: 'Routing is done. A cold model load takes a few minutes; the annotator opens here by itself.',
            pct: 100,
        });
        // No process behind the port. Allow ~10s of probes for a just-launched one
        // to appear, then stop: waiting out the ceiling on a port that nobody is
        // ever going to serve just makes the operator stare at a spinner.
        if (s && !s.running) { deadProbes += 1; if (deadProbes >= 20) break; }
        else { deadProbes = 0; }
        waited += 1;
        await new Promise(r => setTimeout(r, 500));
    }
    if (!port) {
        annotatorLaunchFailed('step4test', win, 'Service port not set yet.');
        return;
    }
    if (!healthy) {
        annotatorLaunchFailed('step4test', win, deadProbes >= 20
            ? `No Combined Annotator process is running on port ${port}. The launch did not start, or it exited. Check the service log.`
            : `Combined Annotator did not become ready (port ${port}). Check the service log.`);
        return;
    }

    // Surface any routing-skip warnings in the panel (non-fatal): they tell the
    // operator that N of M frames were dropped (see routing_report.json).
    renderStep4testWarnings(window._step4testWarnings);
    const url = `http://localhost:${port}/?orchestrated=1`;
    if (win && !win.closed) {
        // Navigate the popup we opened on the click. No blocker risk here.
        try { win.location.href = url; } catch (e) {}
        try { win.focus(); } catch (e) {}
        serviceWindows['step4test'] = win;
        return;
    }
    // Popup was closed during the wait: best-effort reopen (may be blocked, since
    // we are outside the original gesture). Surface a manual link if it fails.
    const w2 = window.open(url, 'reefpointseg_step4test',
        'popup,width=1500,height=980,scrollbars=yes,resizable=yes');
    if (w2) { serviceWindows['step4test'] = w2; try { w2.focus(); } catch (e) {} }
    else {
        annotatorLaunchFailed('step4test', null,
            'The annotator is ready but the browser blocked its window. Open it directly at ' + url,
            'The annotator is up. Allow pop-ups for this page, or paste the address into a new window.');
    }
}

// Render (or clear) the routing-skip warnings note inside the s4test service
// panel. Non-fatal: routing reports drops, it never silently loses frames.
function renderStep4testWarnings(warnings) {
    const svc = document.getElementById('s4test-service');
    if (!svc) return;
    let note = document.getElementById('s4test-route-warnings');
    if (!warnings || !warnings.length) {
        if (note) note.style.display = 'none';
        return;
    }
    if (!note) {
        note = document.createElement('div');
        note.id = 's4test-route-warnings';
        note.className = 'badge badge-error';
        note.style.cssText = 'margin-top:10px;display:block;white-space:normal';
        note.title = 'Frames skipped during routing. See routing_report.json in the output folder.';
        svc.appendChild(note);
    }
    note.style.display = 'block';
    note.textContent = warnings.join('  ');
}

async function stopStep4test() {
    await post('/api/step/step4test/stop', {});
    if (serviceWindows['step4test']) {
        try { serviceWindows['step4test'].close(); } catch (e) {}
        delete serviceWindows['step4test'];
    }
    const svc = document.getElementById('s4test-service');
    if (svc) svc.style.display = 'none';
}

async function openStep4testFolder() {
    try {
        const resp = await post('/api/step/step4test/folder', {});
        if (resp && resp.error) alert(folderFailText(resp));
    } catch (e) { alert('Open folder failed: ' + e); }
}

// ── Refine loop (step4loop): active-learning review loop ──────────────────
// Non-chain tile, same popup-safe launch pattern as step4test (see above), but
// simpler: /api/step/step4loop/run seeds the chosen batch's predictions
// SYNCHRONOUSLY (the seeder subprocess runs to completion inside the request)
// and always comes back with phase:"ui_ready" (or a 4xx error); there is no
// route_status poll step here, unlike step4test's routing pass.

async function refreshStep4loopRuns() {
    const sel = document.getElementById('s4loop-run-dir');
    const openBtn = document.getElementById('s4loop-open-btn');
    if (!sel) return;
    try {
        const data = await api('/api/step/step4loop/inference_runs');
        const runs = data.runs || [];
        const prev = sel.value;
        sel.innerHTML = '';
        if (!runs.length) {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = '(no inference batches yet, run Inference (7) first)';
            sel.appendChild(opt);
            if (openBtn) {
                openBtn.disabled = true;
                openBtn.title = 'Run Inference (7) first';
            }
            return;
        }
        for (const r of runs) {
            const opt = document.createElement('option');
            opt.value = r.dir;
            opt.textContent = `${r.name}: ${r.n_items} item(s)${r.generated_at ? ' · ' + r.generated_at : ''}`;
            sel.appendChild(opt);
        }
        if (prev) sel.value = prev;
        if (openBtn) {
            openBtn.disabled = false;
            openBtn.title = 'Seed the chosen batch\'s predictions as pending masks, then open the combined annotator in resume mode on just those frames.';
        }
    } catch (e) {
        console.warn('refreshStep4loopRuns failed:', e);
    }
}

function collectStep4loopConfig() {
    const cfg = {};
    const sel = document.getElementById('s4loop-run-dir');
    cfg.predictions_dir = sel ? sel.value : '';
    const conf = parseFloat(document.getElementById('s4loop-conf').value);
    cfg.conf_min = isNaN(conf) ? 0.25 : conf;
    const max = parseInt(document.getElementById('s4loop-max').value);
    cfg.max_frames = isNaN(max) ? 0 : max;
    const skipEl = document.getElementById('s4loop-skip-empty');
    cfg.skip_empty = skipEl ? skipEl.checked : false;
    return cfg;
}

async function openStep4loop() {
    const sel = document.getElementById('s4loop-run-dir');
    if (!sel || !sel.value) {
        alert('Pick an inference batch to review first.');
        return;
    }
    clearAnnotatorOpenError('s4loop-open-error');
    // Open the popup synchronously, inside the click handler, same reason as
    // step4test (see openStep4testPlaceholder); the popup blocker only allows
    // window.open() tied to the original user gesture. That popup is also the
    // one loader for this launch: no SPA modal is raised alongside it.
    const win = openStep4loopPlaceholder();
    // Snapshot the paintability of THIS popup now: clicking Open twice rewrites
    // the shared flag while the first launch is still in flight.
    const paintable = bootPaintable['step4loop'];
    if (!win) {
        annotatorLaunchFailed('step4loop', null,
            'Your browser blocked the review window. Allow pop-ups for this page, then click Seed + Open Review again.',
            'The browser refused the new window, so there is nowhere to show the start. Allow pop-ups for this page and click Seed + Open Review again.',
            paintable);
        return;
    }
    serviceWindows['step4loop'] = win;
    const svc = document.getElementById('s4loop-service');
    const stopBtn = document.getElementById('s4loop-stop-btn');
    // Was the row already up before this click? /run answers success:false for
    // ordinary outcomes too ("Nothing new to review"), and a Refine Review from
    // an EARLIER launch can still be running with its own window open. Hiding the
    // row then would take Stop away from a live service, so this launch only ever
    // undoes what it revealed itself.
    const rowWasHidden = !svc || svc.style.display === 'none';
    if (svc) svc.style.display = 'block';
    if (stopBtn) stopBtn.style.display = 'inline-block';
    const next = document.getElementById('s4loop-next-card');
    if (next) next.style.display = 'none';
    const resultEl = document.getElementById('s4loop-seed-result');
    if (resultEl) resultEl.innerHTML = '';

    let r = null;
    try {
        r = await post('/api/step/step4loop/run', { config: collectStep4loopConfig() });
    } catch (e) {
        // Popup stays open on failure: it is the only loader, so closing it would
        // leave the operator with no message at all.
        annotatorLaunchFailed('step4loop', win, 'Failed to start: ' + (e && e.message || e), null, paintable);
        return;
    }
    if (!r || r.success !== true) {
        const msg = (r && r.error) || 'Seeding failed.';
        annotatorLaunchFailed('step4loop', win, msg, null, paintable);
        // The server says nothing launched, so the service row and its Stop
        // button would be offering to stop a process that does not exist. The
        // !port / !healthy paths below keep them, since there a half-started
        // process really is out there.
        if (rowWasHidden) {
            if (svc) svc.style.display = 'none';
            if (stopBtn) stopBtn.style.display = 'none';
        }
        if (resultEl) resultEl.innerHTML = `<div class="badge badge-error" style="margin-top:8px">${escHtmlNav(msg)}</div>`;
        return;
    }
    renderStep4loopSeedSummary(r.seeded);
    await finishStep4loopOpen(r, win);
}

function openStep4loopPlaceholder() {
    let win;
    bootPaintable['step4loop'] = false;
    try {
        win = window.open('', 'reefpointseg_step4loop',
            'popup,width=1500,height=980,scrollbars=yes,resizable=yes');
    } catch (e) { win = null; }
    if (!win) return null;
    try {
        win.document.title = 'Refine Review - starting';
        win.document.body.style.cssText = BOOT_BODY_CSS;
        // Same boot-* skeleton as the Combined Annotator popup, so
        // paintStep4testBoot can drive this window as the single loader
        // (progress, failure text and the Retry link all read here).
        // The hint says "once seeding finishes" because seeding runs inside the
        // /run POST: a close during that window is not observed until the POST
        // returns, and only then does the stop go out.
        win.document.body.innerHTML = bootSkeletonHtml({
            title: 'Starting the Refine Review',
            msg: 'Seeding model predictions and starting the review tool. This can take up to a minute. '
               + 'The tool loads here automatically when it is ready.',
            hint: 'Closing this window stops the review start once seeding finishes. Masks already seeded stay saved.',
        });
        // The document accepted our writes, so this popup is ours to paint.
        bootPaintable['step4loop'] = true;
    } catch (e) { bootPaintable['step4loop'] = false; }
    try { win.focus(); } catch (e) {}
    return win;
}

function closeStep4loopWin(win) {
    try { if (win) win.close(); } catch (e) {}
    if (serviceWindows['step4loop'] === win) delete serviceWindows['step4loop'];
    const svc = document.getElementById('s4loop-service');
    if (svc) svc.style.display = 'none';
}

// Every count the seeder returns for something it did not turn into a review
// mask, in the order the summary states them: the count's key in the seeder's
// summary, what that count counts (frames are dropped whole, detections one at
// a time), the words that follow the number, and, where there is something to
// do about it, the sentence that says what. skipped_holdout is normal rather
// than an error (those frames sit on the frozen T5/T6 transects the loop never
// seeds), while skipped_box_only and skipped_no_geometry are the two the round
// used to drop in silence, leaving a green "0 masks" with no reason on it.
const STEP4LOOP_SKIP_REASONS = [
    { key: 'skipped_holdout', unit: 'frame', text: 'on held-out transects (skipped, expected)' },
    { key: 'skipped_existing', unit: 'frame', text: 'already in the labelset' },
    {
        key: 'skipped_below_conf',
        unit: 'detection',
        text: 'below the confidence threshold',
        fix: 'Lower the confidence and seed again to see the detections this round cut off.',
    },
    {
        key: 'skipped_box_only',
        unit: 'detection',
        text: 'that carried a box and no outline',
        fix: 'Run Inference again with a segmentation model, then start the round: a detector draws a box and a review mask is built from an outline.',
    },
    {
        key: 'skipped_no_geometry',
        unit: 'detection',
        text: 'that carried neither an outline nor a box',
        fix: 'Read the inference run behind this round before trusting its counts: these prediction records hold no shape at all.',
    },
];
// What the summary says when a round seeded nothing and the seeder reported no
// reason for it.
const STEP4LOOP_NOTHING_TO_SEED =
    ' The inference run this round read holds no prediction to review.';
// The glyph beside the drop count opens the breakdown on hover and on keyboard
// focus, so the reasons are one sentence in a tooltip rather than a paragraph
// on screen (spec sections 6 and 7).
const STEP4LOOP_SKIP_GLYPH_LABEL = 'About what this round left out';

/**
 * A count and its noun, singular where the count is one.
 * @param {number} count how many
 * @param {string} noun the singular noun
 * @returns {string} "1 frame", "12 frames"
 */
function countedNoun(count, noun) {
    return `${count} ${noun}${count === 1 ? '' : 's'}`;
}

/**
 * A list of phrases as one clause: "a", "a and b", "a, b and c".
 * @param {Array<string>} parts the phrases, already in the order to read them
 * @returns {string}
 */
function joinPhrases(parts) {
    if (parts.length <= 1) return parts[0] || '';
    return `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}`;
}

/**
 * Every drop this round counted, read off the seeder's summary in the order
 * STEP4LOOP_SKIP_REASONS states them.
 * @param {Object} summary the seeder's summary block
 * @returns {Array<Object>} {count, unit, text, fix} for each drop with a count
 */
function step4loopSkips(summary) {
    return STEP4LOOP_SKIP_REASONS
        .map(reason => Object.assign({ count: summary[reason.key] || 0 }, reason))
        .filter(reason => reason.count > 0);
}

/**
 * What this round left out, as one clause of counts by unit: frames are left
 * out whole and detections one at a time, so the two are never added together.
 * @param {Array<Object>} skips from step4loopSkips
 * @returns {string} "11 frames and 36 detections", '' when nothing was dropped
 */
function step4loopSkipTotals(skips) {
    const totals = ['frame', 'detection'].map(unit => {
        const count = skips.filter(s => s.unit === unit)
            .reduce((sum, s) => sum + s.count, 0);
        return count ? countedNoun(count, unit) : '';
    }).filter(Boolean);
    return joinPhrases(totals);
}

/**
 * The breakdown behind the drop count: every reason with its count, then the
 * one sentence that says what to do about the largest drop that can be acted
 * on, which on a round seeded from a detector is almost always the detections
 * that carried a box and no outline.
 * @param {Array<Object>} skips from step4loopSkips
 * @returns {string} the tooltip text
 */
function step4loopSkipTooltip(skips) {
    const clauses = skips.map(s => `${countedNoun(s.count, s.unit)} ${s.text}`);
    const actionable = skips.filter(s => s.fix).slice().sort((a, b) => b.count - a.count);
    const fix = actionable.length ? actionable[0].fix : '';
    return `VEGA left these out of the round: ${joinPhrases(clauses)}.${fix ? ` ${fix}` : ''}`;
}

/**
 * Render the seeder's summary as one line: what the round seeded, what it left
 * out, and a glyph carrying the reasons. A round that seeded no mask reads as
 * a warning rather than a success.
 * @param {Object|null} summary the seeder's summary block from /run
 */
function renderStep4loopSeedSummary(summary) {
    const el = document.getElementById('s4loop-seed-result');
    if (!el || !summary) return;
    const seeded = summary.seeded_frames || 0;
    const masks = summary.seeded_masks || 0;
    const skips = step4loopSkips(summary);
    const totals = step4loopSkipTotals(skips);
    const glyph = skips.length
        ? ` <button type="button" class="tip" aria-label="${escAttr(STEP4LOOP_SKIP_GLYPH_LABEL)}"`
          + ` title="${escAttr(step4loopSkipTooltip(skips))}"><span aria-hidden="true">i</span></button>`
        : '';
    const badge = masks ? 'badge-success' : 'badge-warning';
    // 13 px, the floor for text that explains anything (workflow page design,
    // section 3), and one line: the reasons live on the glyph.
    el.innerHTML =
        `<div class="badge ${badge}" style="margin-top:8px;display:block;`
        + `font-size:13px;line-height:1.7;white-space:normal">`
        + `Seeded ${countedNoun(seeded, 'frame')}, ${countedNoun(masks, 'mask')} for review.`
        + (totals ? ` Left out: ${totals}.` : '')
        + ((!masks && !skips.length) ? STEP4LOOP_NOTHING_TO_SEED : '')
        + `${glyph}</div>`;
}

// Shared tail for openStep4loop: /run already seeded synchronously and
// returned phase:"ui_ready" with ui_port, so just wait for /status to report
// healthy, then navigate the already-open popup, same technique as
// openStep4testWindow (never window.open() from inside an async callback).
async function finishStep4loopOpen(r, win) {
    // Same process-based wait as openStep4testWindow: a cold SAM3 load outlives any
    // fixed deadline, so wait while the process is alive (10-min ceiling) and bail
    // early only when nothing is running behind the port.
    const ceiling = Date.now() + 600000;
    let port = (r && r.ui_port) || null, healthy = false, waited = 0, deadProbes = 0;
    while (Date.now() < ceiling) {
        if (win && win.closed) {
            // Closing the window means stop, which is what the popup hint promised:
            // a cold SAM3 load would otherwise keep running with nobody waiting
            // for it, holding the GPU with no way to see it from the UI.
            try { await post('/api/step/step4loop/stop', {}); } catch (e) {}
            closeStep4loopWin(win);
            onStep4loopWindowClosed();
            return;
        }
        let s; try { s = await api('/api/step/step4loop/status'); } catch (e) { s = null; }
        if (s && s.port) port = s.port;
        if (s && s.healthy) { healthy = true; break; }
        if (s && !s.running) { deadProbes += 1; if (deadProbes >= 20) break; }
        else { deadProbes = 0; }
        waited += 1;
        if (waited >= 3) {
            paintStep4testBoot(win, {
                title: 'Loading the SAM3 model…',
                msg: 'Seeding is done. A cold model load takes a few minutes; the review tool opens here by itself.',
                pct: 100,
            });
        }
        await new Promise(res => setTimeout(res, 500));
    }
    if (!port) {
        annotatorLaunchFailed('step4loop', win, 'Service port not set yet.');
        return;
    }
    if (!healthy) {
        annotatorLaunchFailed('step4loop', win, deadProbes >= 20
            ? `No Refine Review process is running on port ${port}. The launch did not start, or it exited. Check the service log.`
            : `Refine Review did not become ready (port ${port}). Check the service log.`);
        return;
    }

    const url = `http://localhost:${port}/?orchestrated=1`;
    if (win && !win.closed) {
        try { win.location.href = url; } catch (e) {}
        try { win.focus(); } catch (e) {}
        serviceWindows['step4loop'] = win;
        watchStep4loopWindow(win);
        return;
    }
    const w2 = window.open(url, 'reefpointseg_step4loop',
        'popup,width=1500,height=980,scrollbars=yes,resizable=yes');
    if (w2) { serviceWindows['step4loop'] = w2; try { w2.focus(); } catch (e) {} watchStep4loopWindow(w2); }
    else {
        annotatorLaunchFailed('step4loop', null,
            'The review tool is ready but the browser blocked its window. Open it directly at ' + url,
            'The review tool is up. Allow pop-ups for this page, or paste the address into a new window.');
    }
}

// Poll for the popup closing so the after-export card can appear the moment
// the reviewer finishes a session (annotator export closes its own window).
function watchStep4loopWindow(win) {
    if (window._step4loopWatchTimer) clearInterval(window._step4loopWatchTimer);
    window._step4loopWatchTimer = setInterval(() => {
        if (!win || win.closed) {
            clearInterval(window._step4loopWatchTimer);
            window._step4loopWatchTimer = null;
            if (serviceWindows['step4loop'] === win) delete serviceWindows['step4loop'];
            onStep4loopWindowClosed();
        }
    }, 1000);
}

// The annotator window closed (export finished or the user just closed it):
// show the after-export card so the reviewer can jump straight into retraining.
function onStep4loopWindowClosed() {
    const svc = document.getElementById('s4loop-service');
    if (svc) svc.style.display = 'none';
    const stopBtn = document.getElementById('s4loop-stop-btn');
    if (stopBtn) stopBtn.style.display = 'none';
    const next = document.getElementById('s4loop-next-card');
    if (next) next.style.display = 'block';
}

async function stopStep4loop() {
    await post('/api/step/step4loop/stop', {});
    if (window._step4loopWatchTimer) { clearInterval(window._step4loopWatchTimer); window._step4loopWatchTimer = null; }
    if (serviceWindows['step4loop']) {
        try { serviceWindows['step4loop'].close(); } catch (e) {}
        delete serviceWindows['step4loop'];
    }
    const svc = document.getElementById('s4loop-service');
    if (svc) svc.style.display = 'none';
    const stopBtn = document.getElementById('s4loop-stop-btn');
    if (stopBtn) stopBtn.style.display = 'none';
}

async function openStep4loopFolder() {
    try {
        const resp = await post('/api/step/step4loop/folder', {});
        if (resp && resp.error) alert(folderFailText(resp));
    } catch (e) { alert('Open folder failed: ' + e); }
}

// ── Edit Masks (editmasks): standalone relabel/fix tool ────────────────────
// Non-chain tile, same popup-safe launch pattern as step4loop (see above), but
// simpler still: /api/step/editmasks/run has no config to collect and no
// seeding pass - it preflights that the export has masks, then launches
// directly and always comes back with phase:"ui_ready" (or a 4xx error).

async function openEditMasks() {
    clearAnnotatorOpenError('s4em-open-error');
    // Open the popup synchronously, inside the click handler, same reason as
    // step4loop/step4test (the popup blocker only allows window.open() tied
    // to the original user gesture). That popup is also the one loader for this
    // launch: no SPA modal is raised alongside it.
    const win = openEditMasksPlaceholder();
    // Snapshot the paintability of THIS popup now: clicking Open twice rewrites
    // the shared flag while the first launch is still in flight.
    const paintable = bootPaintable['editmasks'];
    if (!win) {
        annotatorLaunchFailed('editmasks', null,
            'Your browser blocked the Edit Masks window. Allow pop-ups for this page, then click Open Edit Masks again.',
            'The browser refused the new window, so there is nowhere to show the start. Allow pop-ups for this page and click Open Edit Masks again.',
            paintable);
        return;
    }
    serviceWindows['editmasks'] = win;
    const svc = document.getElementById('s4em-service');
    const stopBtn = document.getElementById('s4em-stop-btn');
    // Same guard as the Refine Review: only ever hide a row this launch revealed,
    // never one an earlier, still-running Edit Masks session put up.
    const rowWasHidden = !svc || svc.style.display === 'none';
    if (svc) svc.style.display = 'block';
    if (stopBtn) stopBtn.style.display = 'inline-block';

    let r = null;
    try {
        r = await post('/api/step/editmasks/run', {});
    } catch (e) {
        // Popup stays open on failure: it is the only loader, so closing it would
        // leave the operator with no message at all.
        annotatorLaunchFailed('editmasks', win, 'Failed to start: ' + (e && e.message || e), null, paintable);
        return;
    }
    if (!r || r.success !== true) {
        annotatorLaunchFailed('editmasks', win, (r && r.error) || 'Failed to start Edit Masks.', null, paintable);
        // The preflight refused, so nothing launched: the service row and its
        // Stop button would be offering to stop a process that does not exist.
        // The !port / !healthy paths below keep them, since there a half-started
        // process really is out there.
        if (rowWasHidden) {
            if (svc) svc.style.display = 'none';
            if (stopBtn) stopBtn.style.display = 'none';
        }
        return;
    }
    await finishEditMasksOpen(r, win);
}

function openEditMasksPlaceholder() {
    let win;
    bootPaintable['editmasks'] = false;
    try {
        win = window.open('', 'reefpointseg_editmasks',
            'popup,width=1500,height=980,scrollbars=yes,resizable=yes');
    } catch (e) { win = null; }
    if (!win) return null;
    try {
        win.document.title = 'Edit Masks - starting';
        win.document.body.style.cssText = BOOT_BODY_CSS;
        // Same boot-* skeleton as the Combined Annotator popup, so
        // paintStep4testBoot can drive this window as the single loader
        // (progress, failure text and the Retry link all read here).
        win.document.body.innerHTML = bootSkeletonHtml({
            title: 'Starting the Edit Masks tool',
            msg: 'This can take up to a minute. The tool loads here automatically when it is ready.',
            hint: 'Closing this window cancels the start. Masks already saved are not changed.',
        });
        // The document accepted our writes, so this popup is ours to paint.
        bootPaintable['editmasks'] = true;
    } catch (e) { bootPaintable['editmasks'] = false; }
    try { win.focus(); } catch (e) {}
    return win;
}

function closeEditMasksWin(win) {
    try { if (win) win.close(); } catch (e) {}
    if (serviceWindows['editmasks'] === win) delete serviceWindows['editmasks'];
    const svc = document.getElementById('s4em-service');
    if (svc) svc.style.display = 'none';
}

// Shared tail for openEditMasks: /run already preflighted and launched, and
// returned phase:"ui_ready" with ui_port, so just wait for /status to report
// healthy, then navigate the already-open popup (never window.open() from
// inside an async callback).
async function finishEditMasksOpen(r, win) {
    // Same process-based wait as openStep4testWindow: a cold SAM3 load outlives any
    // fixed deadline, so wait while the process is alive (10-min ceiling) and bail
    // early only when nothing is running behind the port.
    const ceiling = Date.now() + 600000;
    let port = (r && r.ui_port) || null, healthy = false, waited = 0, deadProbes = 0;
    while (Date.now() < ceiling) {
        if (win && win.closed) {
            // Closing the window means stop, which is what the popup hint promised:
            // a cold SAM3 load would otherwise keep running with nobody waiting
            // for it, holding the GPU with no way to see it from the UI.
            try { await post('/api/step/editmasks/stop', {}); } catch (e) {}
            closeEditMasksWin(win);
            onEditMasksWindowClosed();
            return;
        }
        let s; try { s = await api('/api/step/editmasks/status'); } catch (e) { s = null; }
        if (s && s.port) port = s.port;
        if (s && s.healthy) { healthy = true; break; }
        if (s && !s.running) { deadProbes += 1; if (deadProbes >= 20) break; }
        else { deadProbes = 0; }
        waited += 1;
        if (waited >= 3) {
            paintStep4testBoot(win, {
                title: 'Loading the SAM3 model…',
                msg: 'A cold model load takes a few minutes; Edit Masks opens here by itself.',
                pct: 100,
            });
        }
        await new Promise(res => setTimeout(res, 500));
    }
    if (!port) {
        annotatorLaunchFailed('editmasks', win, 'Service port not set yet.');
        return;
    }
    if (!healthy) {
        annotatorLaunchFailed('editmasks', win, deadProbes >= 20
            ? `No Edit Masks process is running on port ${port}. The launch did not start, or it exited. Check the service log.`
            : `Edit Masks did not become ready (port ${port}). Check the service log.`);
        return;
    }

    const url = `http://localhost:${port}/?orchestrated=1`;
    if (win && !win.closed) {
        try { win.location.href = url; } catch (e) {}
        try { win.focus(); } catch (e) {}
        serviceWindows['editmasks'] = win;
        watchEditMasksWindow(win);
        return;
    }
    const w2 = window.open(url, 'reefpointseg_editmasks',
        'popup,width=1500,height=980,scrollbars=yes,resizable=yes');
    if (w2) { serviceWindows['editmasks'] = w2; try { w2.focus(); } catch (e) {} watchEditMasksWindow(w2); }
    else {
        annotatorLaunchFailed('editmasks', null,
            'Edit Masks is ready but the browser blocked its window. Open it directly at ' + url,
            'Edit Masks is up. Allow pop-ups for this page, or paste the address into a new window.');
    }
}

// Poll for the popup closing so the service status row hides the moment the
// reviewer finishes a session (editMasks export closes its own window).
function watchEditMasksWindow(win) {
    if (window._editMasksWatchTimer) clearInterval(window._editMasksWatchTimer);
    window._editMasksWatchTimer = setInterval(() => {
        if (!win || win.closed) {
            clearInterval(window._editMasksWatchTimer);
            window._editMasksWatchTimer = null;
            if (serviceWindows['editmasks'] === win) delete serviceWindows['editmasks'];
            onEditMasksWindowClosed();
        }
    }, 1000);
}

function onEditMasksWindowClosed() {
    const svc = document.getElementById('s4em-service');
    if (svc) svc.style.display = 'none';
    const stopBtn = document.getElementById('s4em-stop-btn');
    if (stopBtn) stopBtn.style.display = 'none';
}

async function stopEditMasks() {
    await post('/api/step/editmasks/stop', {});
    if (window._editMasksWatchTimer) { clearInterval(window._editMasksWatchTimer); window._editMasksWatchTimer = null; }
    if (serviceWindows['editmasks']) {
        try { serviceWindows['editmasks'].close(); } catch (e) {}
        delete serviceWindows['editmasks'];
    }
    const svc = document.getElementById('s4em-service');
    if (svc) svc.style.display = 'none';
    const stopBtn = document.getElementById('s4em-stop-btn');
    if (stopBtn) stopBtn.style.display = 'none';
}

async function openEditMasksFolder() {
    try {
        const resp = await post('/api/step/editmasks/folder', {});
        if (resp && resp.error) alert(folderFailText(resp));
    } catch (e) { alert('Open folder failed: ' + e); }
}

// After-export card buttons (Refine panel): jump to the Train panel already
// set up the way each button promises. 'base' clears any pick under Continue
// from a VEGA model, so the run starts from the starting model chosen above
// it; 'champion' picks the champion for the training task on this panel, if
// one has been promoted for it, or says there is none to continue from.
const NO_CHAMPION_RESULT_TEXT =
    'No champion has been promoted yet. Evaluate a run and press Promote first.';

/**
 * Jump to the Train panel from the Refine after-export card's "Train a new
 * model" or "Continue from the champion" buttons.
 * @param {'base'|'champion'} kind which button was pressed
 * @returns {Promise<void>|undefined}
 */
async function gotoStep6Start(kind) {
    switchStep(6);
    const sel = document.getElementById('s6-continue-from');
    if (kind !== 'champion') {
        if (sel) sel.value = '';
        onStep6ContinueChange();
        return;
    }
    // The champion pick needs a fresh read of who the champion is and which
    // task it trained; switchStep(6) also starts one, but its own refresh
    // runs in an unawaited background pass, so this panel would otherwise
    // read a not-yet-loaded select.
    await refreshStep6ContinueOptions();
    const wantTask = step6TaskValue() === 'detect' ? 'detect' : 'seg';
    const value = sel ? (sel.dataset.championValue || '') : '';
    const task = sel ? (sel.dataset.championTask || '') : '';
    if (value && task === wantTask) {
        applyStep6ModelPath(value);
    } else {
        showResult(6, false, NO_CHAMPION_RESULT_TEXT);
    }
}

// Add Expert IDs is now a native panel (the _expertids blueprint + expertids.js).
// switchStep('expertids') calls window.ExpertIDs.onShow() to refresh it; there is
// no longer any subprocess/iframe wiring here.

// ── Open folder helpers (xdg-open on server) ──────────────────────────────

// Every open-folder error body carries the resolved path, so the reader can
// still get to the folder by hand when the file manager refuses to open.
function folderFailText(resp) {
    let msg = 'Open folder failed: ' + (resp.error || 'unknown error');
    if (resp.path) msg += '\n\nPath: ' + resp.path;
    return msg;
}

async function openStepFolder(step) {
    if (!state) return;
    const dir = state.steps && state.steps[String(step)] && state.steps[String(step)].dir;
    if (!dir) {
        alert(`Step ${stepDisplayNum(step)} has no output folder yet. `
              + `Run the step once and VEGA makes the folder.`);
        return;
    }
    try {
        const resp = await post('/api/fs/open', { path: dir });
        if (resp && resp.error) alert(folderFailText(resp));
    } catch (e) {
        alert('Open folder failed: ' + e);
    }
}

async function openProjectFolder() {
    if (!state) return;
    const dir = state.project_dir || state.dir;
    if (!dir) {
        alert('No project folder path available in state.');
        return;
    }
    try {
        const resp = await post('/api/fs/open', { path: dir });
        if (resp && resp.error) alert(folderFailText(resp));
    } catch (e) {
        alert('Open folder failed: ' + e);
    }
}

// ── Step control ────────────────────────────────────────────────────────────

/**
 * Stop a step's run, and report what the server actually did.
 *
 * The Stop route answers three ways: it held the process and killed it,
 * nothing was running and the stale mark was cleared, or it refuses with
 * HTTP_CONFLICT because a process outlived a VEGA restart and the runner has
 * no handle on it. The refusal used to be read off the wire and discarded: the
 * tile was cleared and the reader told the run had stopped while the trainer
 * went on writing into the run folder. A refusal now leaves the tile, the
 * poller and the buttons exactly as they are and says what is still running.
 * @param {number|string} step the step key
 * @returns {Promise<void>}
 */
async function stopStep(step) {
    // The press is the reader's next action, so any earlier held line is over.
    releaseStepStatusNote(step);
    const data = (await post(`/api/step/${step}/stop`, {})) || {};
    if (data._status === HTTP_CONFLICT || data.error) {
        // The refusal names the process and says what to do; it holds against
        // the poll's own sentence about that process until the reader acts.
        showStepStatusNote(step, 'error', data.error || STOP_REFUSED_TEXT, true);
        return;
    }
    showStepStatusNote(step, 'warning', []);
    if (pollTimers[step]) {
        clearInterval(pollTimers[step]);
        delete pollTimers[step];
    }
    stopElapsedTimer(step);
    hideProgress(step);
    setStepRunControls(step, false);
    state.steps[step].status = 'pending';
    updateSidebar();
    const svc = document.getElementById(`s${step}-service`);
    if (svc) svc.style.display = 'none';
}

async function markDone(step) {
    const data = await post(`/api/step/${step}/done`, {});
    if (data.error) return alert('Error: ' + data.error);
    if (data.state) state = data.state;
    renderProjectName();
    if (pollTimers[step]) {
        clearInterval(pollTimers[step]);
        delete pollTimers[step];
    }
    stopElapsedTimer(step);
    hideProgress(step);
    const svc = document.getElementById(`s${step}-service`);
    if (svc) svc.style.display = 'none';
    // Close the service window
    if (serviceWindows[step]) {
        try { serviceWindows[step].close(); } catch(e) {}
        delete serviceWindows[step];
    }
    updateSidebar();
    populateConfigs();
    showResult(step, true, 'Step completed.');
    markStepActionDone(step);
}

// Per-step "this removes XYZ" copy, shown in the reset-confirm modal so the user
// knows exactly what a reset clears before they confirm.
const RESET_DESCRIPTIONS = {
    1: "This removes the linked or parsed all_points.csv and master_codes.csv and re-locks Steps 2 through 7.",
    2: "This removes the recoded outputs (all_points_recoded.csv, master_codes_recoded.csv, remap_log.json) and re-locks Steps 3 through 7.",
    3: "This removes selected_frames.csv and the route files, and re-locks Steps 4 through 7.",
    6: "This removes the dataset split and training runs produced by this step and re-locks Steps 6 and 7.",
    7: "This removes the evaluation report and metrics produced by this step.",
    8: "This removes the inference run outputs and gallery produced by this step.",
    step4test: "This removes this tool's routed_input and its annotations (all_images, all_labels, segmentations). The next Open re-routes Step 3's chosen images from scratch.",
    step4loop: "Removes model proposals you have not reviewed yet (pending cyan masks seeded by Refine). Accepted masks, your manual masks, and everything already exported are kept.",
    editmasks: "Stops the Edit Masks service. This never deletes or changes any mask; your Step 4 export is left exactly as it was.",
};

// The actual reset network/DOM work. Called only after the confirm modal is
// accepted (openResetConfirm wires this to #confirm-ok-btn).
async function doResetStep(step, confirmExported) {
    const body = confirmExported === undefined ? {} : { confirm_exported: confirmExported };
    const data = await post(`/api/step/${step}/reset`, body);
    // Exported frames: the server names the count and waits for the same count
    // back, so the same confirm modal opens once more with the count in it.
    if (data.confirm_token !== undefined && confirmExported === undefined) {
        return openResetConfirm(step, data);
    }
    if (data.error) return alert('Error: ' + data.error);
    if (data.state) state = data.state;
    renderProjectName();
    updateSidebar();
    const logEl = document.getElementById(`s${step}-log`);
    if (logEl) { logEl.innerHTML = ''; logEl.style.display = 'none'; }
    const resultEl = document.getElementById(`s${step}-result`);
    if (resultEl) resultEl.innerHTML = '';
    // A state line is about the run the reset just cleared.
    showStepStatusNote(step, 'warning', []);
    const svc = document.getElementById(`s${step}-service`);
    if (svc) svc.style.display = 'none';
    // step4test uses s4test-* ids (not sstep4test-*), so clear those explicitly
    // and close any open combined-annotator window the reset just orphaned.
    if (step === 'step4test') {
        const t4svc = document.getElementById('s4test-service');
        if (t4svc) t4svc.style.display = 'none';
        const t4log = document.getElementById('s4test-log');
        if (t4log) { t4log.innerHTML = ''; t4log.style.display = 'none'; }
        if (serviceWindows['step4test']) {
            try { serviceWindows['step4test'].close(); } catch (e) {}
            delete serviceWindows['step4test'];
        }
    }
    // step4loop uses s4loop-* ids (not sstep4loop-*), same reason as step4test
    // above; also hide the after-export card and close its window if open.
    if (step === 'step4loop') {
        const l4svc = document.getElementById('s4loop-service');
        if (l4svc) l4svc.style.display = 'none';
        const l4log = document.getElementById('s4loop-log');
        if (l4log) { l4log.innerHTML = ''; l4log.style.display = 'none'; }
        const l4result = document.getElementById('s4loop-seed-result');
        if (l4result) l4result.innerHTML = '';
        const l4next = document.getElementById('s4loop-next-card');
        if (l4next) l4next.style.display = 'none';
        const l4stop = document.getElementById('s4loop-stop-btn');
        if (l4stop) l4stop.style.display = 'none';
        if (serviceWindows['step4loop']) {
            try { serviceWindows['step4loop'].close(); } catch (e) {}
            delete serviceWindows['step4loop'];
        }
    }
    // editmasks uses s4em-* ids (not sstep4mask-*), same reason as step4test/
    // step4loop above; also close its window if open. Reset never touches the
    // export here - see editmasks_reset() server-side.
    if (step === 'editmasks') {
        const emsvc = document.getElementById('s4em-service');
        if (emsvc) emsvc.style.display = 'none';
        const emlog = document.getElementById('s4em-log');
        if (emlog) { emlog.innerHTML = ''; emlog.style.display = 'none'; }
        const emstop = document.getElementById('s4em-stop-btn');
        if (emstop) emstop.style.display = 'none';
        if (serviceWindows['editmasks']) {
            try { serviceWindows['editmasks'].close(); } catch (e) {}
            delete serviceWindows['editmasks'];
        }
    }
    hideProgress(step);
}

// Open the styled reset-confirm modal for a step, describing what will be
// cleared and wiring the Confirm button to run the reset.
function openResetConfirm(step, exportedInfo) {
    const titleEl = document.getElementById('confirm-title');
    if (titleEl) titleEl.textContent = 'Reset this step?';
    const msgEl = document.getElementById('confirm-msg');
    if (msgEl) {
        // editmasks reset is stop-only (never touches the export), so the
        // "cannot be undone" warning would contradict its own description.
        const destructive = step !== 'editmasks';
        msgEl.textContent = (RESET_DESCRIPTIONS[step] || 'This clears this step\'s outputs.') +
            (destructive ? ' This cannot be undone.' : '');
        if (exportedInfo) {
            msgEl.textContent += exportedInfo.annotated_count === null
                ? ' The annotation record could not be read, so reviewed or exported frames cannot be counted.'
                : ` ${exportedInfo.annotated_count} frames are already reviewed or exported and will be deleted.`;
        }
    }
    const confirmToken = exportedInfo ? exportedInfo.confirm_token : undefined;
    const okBtn = document.getElementById('confirm-ok-btn');
    if (okBtn) { okBtn.textContent = 'Confirm Reset'; okBtn.onclick = () => { closeConfirmModal(); doResetStep(step, confirmToken); }; }
    const ov = document.getElementById('confirm-overlay');
    if (ov) ov.classList.add('visible');
}

function closeConfirmModal() {
    const ov = document.getElementById('confirm-overlay');
    if (ov) ov.classList.remove('visible');
}

// Keyboard: Escape closes whichever dialog is up, Enter confirms the reset /
// promote modal while it is open. Nothing else is bound, so typing in a field
// still behaves exactly as it did.
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeConfirmModal();
        hideOpenUiOverlay();
        return;
    }
    if (e.key !== 'Enter') return;
    const ov = document.getElementById('confirm-overlay');
    if (!ov || !ov.classList.contains('visible')) return;
    const okBtn = document.getElementById('confirm-ok-btn');
    if (!okBtn) return;
    e.preventDefault();
    okBtn.click();
});

// Save the project and close this window. Stops running sub-tools server-side
// (/api/project/quit calls pm.save_project + runner.kill_all) and returns to
// the setup screen; window.close() succeeds when this was a script-opened window.
async function saveAndClose() {
    try { Object.values(serviceWindows).forEach(w => { try { w.close(); } catch(e){} }); } catch(e){}
    serviceWindows = {};
    await post('/api/project/quit', {});
    state = null;
    // This window was opened by the VICARIUS UI; close it to return there. If the
    // browser refuses (tab not script-opened), bounce so the user never reaches
    // a local home screen.
    try { window.close(); } catch(e) {}
    bounceToVicarius();
}

// ── Remap browsing (Step 2) ─────────────────────────────────────────────────

async function browseRemaps() {
    const data = await api('/api/remap/browse');
    if (!data.logs || data.logs.length === 0) {
        alert('No remap logs found.');
        return;
    }

    const listEl = document.getElementById('s2-remap-list');
    listEl.style.display = 'block';

    let html = '<table class="remap-table"><tr><th>Timestamp</th><th>Remaps</th><th>Excludes</th><th>Preview</th><th></th></tr>';
    for (const log of data.logs) {
        const preview = (log.remaps_summary || []).slice(0, 3).join(', ');
        html += `<tr>
            <td>${log.timestamp}</td>
            <td>${log.n_remaps}</td>
            <td>${log.n_excludes}</td>
            <td style="color:var(--text-dim);font-size:12px">${preview}</td>
            <td><button class="btn btn-outline btn-sm" onclick="selectRemap('${log.path.replace(/'/g, "\\'")}')">Select</button></td>
        </tr>`;
    }
    html += '</table>';
    listEl.innerHTML = html;
}

async function selectRemap(path) {
    document.getElementById('s2-remap-path').value = path;
    document.getElementById('s2-remap-list').style.display = 'none';

    const data = await post('/api/remap/load', { path });
    if (data.error) return alert('Error: ' + data.error);

    const log = data.remap_log;
    const previewEl = document.getElementById('s2-remap-preview');
    previewEl.style.display = 'block';

    let html = `<p style="margin-bottom:8px;color:var(--text-dim)">${log.remaps.length} remaps, ${(log.excludes||[]).length} excludes</p>`;
    html += '<table class="remap-table"><tr><th>Old Code</th><th>New Code</th><th>Label Name</th><th>Action</th><th>Points Affected</th></tr>';
    for (const rm of (log.remaps || []).slice(0, 20)) {
        html += `<tr>
            <td>${rm.old_code}</td>
            <td>${rm.new_code}</td>
            <td>${rm.new_name || ''}</td>
            <td><span class="badge ${rm.action === 'merge' ? 'badge-warning' : 'badge-info'}">${rm.action || ''}</span></td>
            <td>${(rm.points_affected || 0).toLocaleString()}</td>
        </tr>`;
    }
    if (log.remaps.length > 20) html += `<tr><td colspan="5" style="color:var(--text-dim)">... and ${log.remaps.length - 20} more</td></tr>`;
    html += '</table>';
    if (log.excludes && log.excludes.length > 0) {
        html += `<p style="margin-top:8px;color:var(--text-dim);font-size:12px">Excluded from ranking: ${log.excludes.join(', ')}</p>`;
    }
    previewEl.innerHTML = html;
}

async function autoApplyRemap() {
    const path = document.getElementById('s2-remap-path').value.trim();
    if (!path) return alert('Select a remap log file first (use Browse or paste a path).');

    showProgress(2, 'Applying remaps to ~2M point observations...', null);
    stepStartTimes[2] = Date.now();

    const data = await post('/api/remap/apply', { remap_log_path: path });

    hideProgress(2);
    if (data.error) return alert('Error: ' + data.error);

    if (data.state) state = data.state;
    renderProjectName();
    updateSidebar();
    populateConfigs();
    showResult(2, true,
        `Auto-applied ${data.remaps_applied} remaps to ${data.points_processed.toLocaleString()} points. ` +
        `${data.excludes} codes excluded. Output files: ${(data.output_files || []).join(', ')}`);

    markStepActionDone(2);
}

// ── Initialize ──────────────────────────────────────────────────────────────

// ── Step 7 / 8 helpers (training-run picker, report viewer, gallery) ───────

// ── Step 6 preset loader (bulk-fill form from YAML recipe) ─────────────────

// Mapping: preset param key -> function that writes the value into the form.
// Kept separate from populateConfigs() because presets intentionally omit
// keys the user shouldn't inherit from a preset (run_name, include_classes).
const STEP6_PRESET_APPLIERS = {
    model:          v => { const el = document.getElementById('s6-model'); if (el) el.value = v; },
    model_path:     v => applyStep6ModelPath(v),
    freeze:         v => setVal('s6-freeze', v),
    pin_split:      v => { const el = document.getElementById('s6-pin-split'); if (el) el.checked = !!(v && v !== '0'); },
    epochs:         v => setVal('s6-epochs', v),
    imgsz:          v => setVal('s6-imgsz', v),
    batch:          v => setVal('s6-batch', v),
    patience:       v => setVal('s6-patience', v),
    valid_ratio:    v => setVal('s6-valid', v),
    test_ratio:     v => setVal('s6-test', v),
    min_samples:    v => setVal('s6-min-samples', v),
    device:         v => applyStepDevice(6, v, TRAIN_DEVICE_DEFAULT),
    hsv_h:          v => setVal('s6-hsv-h', v),
    hsv_s:          v => setVal('s6-hsv-s', v),
    hsv_v:          v => setVal('s6-hsv-v', v),
    bgr:            v => setVal('s6-bgr', v),
    degrees:        v => setVal('s6-degrees', v),
    translate:      v => setVal('s6-translate', v),
    scale:          v => setVal('s6-scale', v),
    shear:          v => setVal('s6-shear', v),
    perspective:    v => setVal('s6-perspective', v),
    flipud:         v => setVal('s6-flipud', v),
    fliplr:         v => setVal('s6-fliplr', v),
    mosaic:         v => setVal('s6-mosaic', v),
    mixup:          v => setVal('s6-mixup', v),
    cutmix:         v => setVal('s6-cutmix', v),
    copy_paste:     v => setVal('s6-copy-paste', v),
    copy_paste_mode:v => { const el = document.getElementById('s6-copy-paste-mode'); if (el) el.value = v; },
    close_mosaic:   v => setVal('s6-close-mosaic', v),
    optimizer:      v => { const el = document.getElementById('s6-optimizer'); if (el) el.value = v; },
    lr0:            v => setVal('s6-lr0', v),
    lrf:            v => setVal('s6-lrf', v),
    momentum:       v => setVal('s6-momentum', v),
    weight_decay:   v => setVal('s6-weight-decay', v),
    warmup_epochs:  v => setVal('s6-warmup-epochs', v),
    cos_lr:         v => { const el = document.getElementById('s6-cos-lr'); if (el) el.checked = !!v; },
    box:            v => setVal('s6-box', v),
    cls:            v => setVal('s6-cls', v),
    dfl:            v => setVal('s6-dfl', v),
    label_smoothing:v => setVal('s6-label-smoothing', v),
};

// ── Step 6 training recipe selection: task + subject ───────────────────────
// Two controls decide which recipe the Load preset list offers: Training task
// (#s6-task) and Subject (#s6-subject). Every recipe in config/presets/
// declares the pair it was written for, /api/step/6/presets reports those
// declarations and chooses nothing, and the matching happens here so a change
// to either control re-filters the list at once.
//
// No file name, file time or folder order takes any part in it. The rule this
// replaced on 2026-09-15 picked whichever name sorted first among the recipes
// carrying a flag, so a coral recipe added beside the fish one would have
// taken over box-dataset training with nothing on the panel saying so.

// The subject a recipe declares when it suits every subject.
const STEP6_SUBJECT_ANY = 'any';
// Every value the Subject control may hold. 'other' names a subject no recipe
// is written for, so it matches the 'any' recipes and nothing else. Same list,
// same order, as the <option> values in templates/index.html.
const STEP6_PANEL_SUBJECTS = ['coral', 'fish', 'algae', 'other'];
// Where the Subject control starts when neither the saved config nor the
// project's staged dataset says otherwise.
const STEP6_SUBJECT_DEFAULT = 'coral';
// What each task is called in the panel's own sentences.
const STEP6_TASK_WORDS = { segment: 'segmentation', detect: 'detector' };

// What /api/step/6/presets last reported, so both controls can re-filter the
// list without another request. null until the first answer arrives: an empty
// list and "not loaded yet" are different states and must not read alike.
let _step6Recipes = null;
// The sentence that answer carried about the project's dataset subject, shown
// with the selection line. '' when there was nothing to say.
let _step6DatasetNote = '';

/**
 * The training task on the form, normalized to what the engine takes.
 * @returns {string} 'detect' or 'segment'
 */
function step6TaskValue() {
    const el = document.getElementById('s6-task');
    return (el && el.value === 'detect') ? 'detect' : 'segment';
}

/**
 * The subject on the form, normalized to the control's vocabulary.
 * @returns {string} one of STEP6_PANEL_SUBJECTS
 */
function step6SubjectValue() {
    const el = document.getElementById('s6-subject');
    const want = el ? String(el.value || '').trim().toLowerCase() : '';
    return STEP6_PANEL_SUBJECTS.indexOf(want) >= 0 ? want : STEP6_SUBJECT_DEFAULT;
}

/**
 * Sort the listed recipes against one training task and one subject.
 * @param {Array} recipes the entries /api/step/6/presets returned
 * @param {string} task 'detect' or 'segment'
 * @param {string} subject one of STEP6_PANEL_SUBJECTS
 * @returns {{matches: Array, unclassified: Array, hidden: Array, builtins: Array}}
 *   matches are the recipes written for exactly this pair (a recipe declaring
 *   subject 'any' matches every subject); unclassified are the recipes that
 *   declared neither key, or something outside the vocabulary, or would not
 *   parse, which are listed and never selected; hidden are the recipes written
 *   for another task or subject; builtins is always empty now (2026-09-21):
 *   the two preset shortcuts it used to hold were removed from
 *   /api/step/6/presets along with the buttons that applied them, and the
 *   bucket is kept only so a recipe carrying an unexpected builtin flag still
 *   sorts somewhere rather than raising.
 */
function step6SortRecipes(recipes, task, subject) {
    const sorted = { matches: [], unclassified: [], hidden: [], builtins: [] };
    for (const recipe of (recipes || [])) {
        if (recipe.builtin) { sorted.builtins.push(recipe); continue; }
        if (!recipe.classified) { sorted.unclassified.push(recipe); continue; }
        const subjectFits = (recipe.subject === subject || recipe.subject === STEP6_SUBJECT_ANY);
        if (recipe.task === task && subjectFits) sorted.matches.push(recipe);
        else sorted.hidden.push(recipe);
    }
    return sorted;
}

/**
 * Names joined as a reader would say them: "A", "A" and "B", "A", "B" and "C".
 * @param {Array<string>} names already quoted recipe names
 * @returns {string}
 */
function step6JoinNames(names) {
    if (names.length <= 1) return names[0] || '';
    return names.slice(0, -1).join(', ') + ' and ' + names[names.length - 1];
}

/**
 * Find one recipe among everything the picker knows, by its id.
 * Every list is searched, because the recipe whose values are in the form may
 * have been written for another task or subject, may be unclassified, or may
 * be one of the two built-in shortcuts. Order is not consulted: the id is an
 * exact match or there is no answer.
 * @param {object} sorted what step6SortRecipes returned
 * @param {string} id the recipe id to look for, '' for none
 * @returns {object|null} the entry, or null when no list holds that id
 */
function step6FindRecipe(sorted, id) {
    if (!id) return null;
    const lists = [sorted.matches, sorted.unclassified, sorted.hidden, sorted.builtins];
    for (const list of lists) {
        for (const recipe of (list || [])) {
            if (recipe.id === id) return recipe;
        }
    }
    return null;
}

/**
 * The sentence naming the recipe whose values are in the form now.
 * The panel must always be able to answer which recipe is in this form, and a
 * recipe imported by hand stays in the form after the two controls move off it,
 * so every state that applied nothing itself still has to name what is there.
 * @param {object} sorted what step6SortRecipes returned
 * @param {string} appliedId the recipe whose values are in the form, '' for none
 * @returns {string} one complete sentence, '' when no recipe filled the form
 */
function step6HeldRecipeSentence(sorted, appliedId) {
    const id = String(appliedId || '').trim();
    if (!id) return '';
    const recipe = step6FindRecipe(sorted, id);
    if (!recipe) {
        return `This form holds the values of "${id}", which config/presets/ no longer lists.`;
    }
    return `This form holds the values of "${recipe.name || recipe.id}", `
        + 'which VICARIUS applied when that recipe was chosen.';
}

/**
 * The line under the recipe picker: which recipes the two controls match, what
 * VICARIUS did about it, and which recipe filled the form. Pure, so it can be
 * read back in a test.
 * @param {string} task 'detect' or 'segment'
 * @param {string} subject the Subject control's value
 * @param {object} sorted what step6SortRecipes returned
 * @param {string} appliedId the recipe whose values are in the form, '' for none
 * @param {string} datasetNote a sentence about the project's dataset, or ''
 * @returns {string} one to five complete sentences
 */
function step6SelectionMessage(task, subject, sorted, appliedId, datasetNote) {
    const lead = `Training task ${STEP6_TASK_WORDS[task] || task} and subject ${subject}`;
    const names = sorted.matches.map(r => `"${r.name || r.id}"`);
    // Named once, because all three branches below have to answer for it: the
    // form can hold a recipe's values while the controls match none of them.
    const held = step6HeldRecipeSentence(sorted, appliedId);
    let text;
    if (names.length === 1) {
        const holdsTheMatch = Boolean(appliedId) && appliedId === sorted.matches[0].id;
        text = `${lead} match one recipe, ${names[0]}. `
            + (holdsTheMatch
                ? 'VICARIUS filled this form from it.'
                : 'Press Import to fill this form from it.');
        if (!holdsTheMatch && held) text += ` ${held}`;
    } else if (names.length > 1) {
        text = `${lead} match ${names.length} recipes, ${step6JoinNames(names)}. `
            + 'VICARIUS applied none of them, because more than one recipe claims the same '
            + 'task and subject: pick one and press Import.';
        if (held) text += ` ${held}`;
    } else if (held) {
        // The form is not on its own defaults here, so the sentence that says
        // it is must not be reached: an imported recipe survives the controls
        // moving off it.
        text = `${lead} match no recipe in config/presets/, so VICARIUS applied none. ${held}`;
    } else {
        text = `${lead} match no recipe in config/presets/. `
            + 'VICARIUS applied none, so this form keeps its own defaults.';
    }
    // Counted in words rather than with a bracketed plural: this line is the
    // one place the panel explains itself, so it reads like a sentence.
    if (sorted.hidden.length === 1) {
        text += ' 1 other recipe in that folder is written for another task or subject '
            + 'and is not listed.';
    } else if (sorted.hidden.length) {
        text += ` ${sorted.hidden.length} other recipes in that folder are written for `
            + 'another task or subject and are not listed.';
    }
    if (sorted.unclassified.length === 1) {
        text += ' 1 recipe declares no task or subject and is listed as unclassified; '
            + 'VICARIUS never selects it on its own.';
    } else if (sorted.unclassified.length) {
        text += ` ${sorted.unclassified.length} recipes declare no task or subject and are `
            + 'listed as unclassified; VICARIUS never selects one of those on its own.';
    }
    if (datasetNote) text += ` ${datasetNote}`;
    return text;
}

/**
 * The dropdown label for one recipe.
 * @param {object} recipe one entry from /api/step/6/presets
 * @returns {string}
 */
function step6RecipeOptionLabel(recipe) {
    if (recipe.error) return `${recipe.id} [error: ${recipe.error}]`;
    const name = recipe.name || recipe.id;
    return (recipe.builtin || recipe.classified) ? name : `${name} (unclassified)`;
}

/**
 * The line the picker shows while one recipe is selected: its own description,
 * plus why an unclassified recipe is never selected on its own.
 * @param {object} recipe one entry from /api/step/6/presets
 * @returns {string}
 */
function step6RecipeOptionDescription(recipe) {
    const parts = [];
    if (recipe.description) parts.push(recipe.description);
    if (recipe.warning) parts.push(recipe.warning);
    if (recipe.error) parts.push(recipe.error);
    return parts.join(' ');
}

/** Write the description of whichever recipe is selected now. */
function step6ShowPresetDescription() {
    const sel = document.getElementById('s6-preset-select');
    const desc = document.getElementById('s6-preset-desc');
    if (!sel || !desc) return;
    const opt = sel.options[sel.selectedIndex];
    desc.textContent = opt ? (opt.dataset.description || '') : '';
}

/** Write the selection line, which says which recipes matched and why. */
function setStep6SelectionNote(text) {
    const el = document.getElementById('s6-preset-selection');
    if (el) el.textContent = text;
}

/**
 * Rebuild the recipe dropdown for the task and subject on the form now, write
 * the selection line, and hand back what matched. Applies nothing: the caller
 * decides, and only ever on a single match.
 * @returns {object} what step6SortRecipes returned, all four lists empty when
 *   the recipe folder has not been read yet
 */
function renderStep6RecipeOptions() {
    const empty = { matches: [], unclassified: [], hidden: [], builtins: [] };
    const sel = document.getElementById('s6-preset-select');
    if (!sel || _step6Recipes === null) return empty;
    const task = step6TaskValue();
    const subject = step6SubjectValue();
    const sorted = step6SortRecipes(_step6Recipes, task, subject);
    const prev = sel.value;
    const listed = sorted.builtins.concat(sorted.matches, sorted.unclassified);
    sel.innerHTML = '<option value="">(none selected)</option>';
    for (const recipe of listed) {
        const opt = document.createElement('option');
        opt.value = recipe.id;
        opt.textContent = step6RecipeOptionLabel(recipe);
        opt.dataset.description = step6RecipeOptionDescription(recipe);
        sel.appendChild(opt);
    }
    // A pick the operator already made survives while the filter still lists
    // it. Otherwise the single match is offered, and nothing at all is offered
    // when the two controls match none or more than one: a tie is never broken
    // here.
    const ids = listed.map(recipe => recipe.id);
    if (prev && ids.indexOf(prev) >= 0) sel.value = prev;
    else if (sorted.matches.length === 1) sel.value = sorted.matches[0].id;
    else sel.value = '';
    sel.onchange = step6ShowPresetDescription;
    step6ShowPresetDescription();
    setStep6SelectionNote(step6SelectionMessage(
        task, subject, sorted, sel.dataset.appliedPresetId || '', _step6DatasetNote));
    return sorted;
}

/**
 * Start the Subject control on the subject the project's staged box dataset
 * recorded for itself. The STARTING value only: a subject restored from the
 * saved config and a subject the operator picked both win, and this never
 * selects a recipe, since the two controls do that.
 * @param {string} subject what /api/step/6/presets reported, '' for none
 */
function applyStep6DatasetSubject(subject) {
    const el = document.getElementById('s6-subject');
    const want = String(subject || '').trim().toLowerCase();
    if (!el || !want) return;
    if (el.dataset.subjectSource) return;
    if (STEP6_PANEL_SUBJECTS.indexOf(want) < 0) return;
    el.value = want;
    el.dataset.subjectSource = 'dataset';
}

/** The Subject control changed: re-filter the recipe list against it. */
function onStep6SubjectChange() {
    const el = document.getElementById('s6-subject');
    if (el) el.dataset.subjectSource = 'operator';
    renderStep6RecipeOptions();
    refreshStep6CardTags();
}

async function refreshStep6Presets(opts) {
    opts = opts || {};
    const sel = document.getElementById('s6-preset-select');
    const desc = document.getElementById('s6-preset-desc');
    if (!sel) return;
    try {
        const data = await api('/api/step/6/presets');
        _step6Recipes = data.presets || [];
        _step6DatasetNote = data.dataset_subject_warning || '';
        applyStep6DatasetSubject(data.dataset_subject);
        // Read before the render, which offers the single match.
        const prev = sel.value;
        const sorted = renderStep6RecipeOptions();

        // Auto-apply on first panel load (or when the caller asks for it),
        // and ONLY when exactly one recipe is written for the task and
        // subject on the form. Two matching recipes apply nothing: the
        // selection line names both and the operator picks.
        //
        // Skipped as well when the user has already selected a recipe this
        // session, and when the project's stored Step 6 config carries a
        // preset_id marker (D-preset-marker, 2026-09-02): a project saved
        // with one was deliberately configured (possibly from a recipe that
        // was then hand-tuned), and those values must not be clobbered by
        // re-applying anything on a fresh page load. A manual Import still
        // overwrites the form either way.
        const storedPresetId = (state && state.steps && state.steps['6'] &&
            state.steps['6'].config && state.steps['6'].config.preset_id) || '';
        if (opts.autoApply && sorted.matches.length === 1 && !window._step6PresetAutoApplied
                && !prev && !storedPresetId) {
            sel.value = sorted.matches[0].id;
            step6ShowPresetDescription();
            try {
                await importStep6Preset({ auto: true });
                window._step6PresetAutoApplied = true;
            } catch (e) {
                console.warn('auto-apply recipe failed:', e);
            }
        } else if (storedPresetId && !prev) {
            // The panel must always be able to say which recipe is in the
            // form. A project saved from one opens on its own values, so name
            // the recipe they came from rather than leaving the line blank.
            const known = (_step6Recipes || []).find(r => r.id === storedPresetId);
            const name = (known && (known.name || known.id)) || storedPresetId;
            setStep6PresetAppliedNote(`This project's saved training settings came from `
                + `"${name}". VICARIUS kept them and applied no recipe over them.`);
        }
    } catch (e) {
        console.warn('refreshStep6Presets failed:', e);
        if (desc) desc.textContent = `Error loading presets: ${e}`;
    }
}

// Apply a preset's params object to the Step 6 form, then reconcile the
// starting model against the current training task. A preset's `model` (or
// `model_path`) key can set weights that do not match the project's task
// (D1, 2026-09-02: the default tuned_v1_regularized preset auto-applies
// yolo11m-seg.pt on first panel load, stranding a detect-task box-dataset
// project with weights the server refuses). onStep6TaskChange() is called
// LAST, after every params[key] has already written into the form, so it
// re-derives the starting model from the CURRENT task and the value the loop
// just set - never the other way round.
function applyStep6PresetParams(params) {
    let applied = 0, skipped = [];
    for (const [key, val] of Object.entries(params || {})) {
        const fn = STEP6_PRESET_APPLIERS[key];
        if (fn) { fn(val); applied++; }
        else { skipped.push(key); }
    }
    onStep6TaskChange();
    return { applied, skipped };
}

// The line under the preset picker that names the preset whose values are in
// the form now. It stays until another preset is applied, so the working
// defaults are always attributable to a file.
function setStep6PresetAppliedNote(text) {
    const el = document.getElementById('s6-preset-applied');
    if (el) el.textContent = text;
}

async function importStep6Preset(opts) {
    opts = opts || {};
    const sel = document.getElementById('s6-preset-select');
    if (!sel || !sel.value) {
        if (!opts.auto) alert('Pick a preset first.');
        return;
    }
    try {
        const data = await api(`/api/step/6/presets/${encodeURIComponent(sel.value)}`);
        if (data.error) {
            if (!opts.auto) alert(`Preset error: ${data.error}`);
            return;
        }
        const params = data.params || {};
        const { applied, skipped } = applyStep6PresetParams(params);
        // Mark this form as seeded from sel.value (auto or manual): collectConfig(6)
        // reads this to save preset_id, which in turn stops a future auto-apply
        // from clobbering whatever the operator does with the form from here.
        sel.dataset.appliedPresetId = sel.value;
        const prefix = opts.auto
            ? `VICARIUS applied the one recipe written for this run, "${data.name}"`
            : `Imported "${data.name}"`;
        setStep6PresetAppliedNote(`${prefix}: ${applied} field(s) updated`
            + (skipped.length ? ` (ignored: ${skipped.join(', ')})` : '')
            + '. Its values are the working defaults of this form.');
        // The selection line reads the marker just set, so it can now say that
        // this form is filled from the recipe rather than offering to fill it.
        renderStep6RecipeOptions();
    } catch (e) {
        if (!opts.auto) alert(`Failed to import preset: ${e}`);
    }
}

// ── Step 7 preset loader (mirrors Step 6 pattern) ──────────────────────────

// Mapping: preset param key -> function that writes the value into the Step 7 form.
const STEP7_PRESET_APPLIERS = {
    split:          v => { const el = document.getElementById('s7-split'); if (el) el.value = v; },
    imgsz:          v => setVal('s7-imgsz', v),
    conf_threshold: v => setVal('s7-conf', v),
    iou_threshold:  v => setVal('s7-iou', v),
    preview_count:  v => setVal('s7-preview-count', v),
    pdf_export_dir: v => setVal('s7-pdf-export-dir', v || ''),
};

async function refreshStep7Presets() {
    const sel = document.getElementById('s7-preset-select');
    const desc = document.getElementById('s7-preset-desc');
    if (!sel) return;
    try {
        const data = await api('/api/step/7/presets');
        const presets = data.presets || [];
        const prev = sel.value;
        sel.innerHTML = '<option value="">(none selected)</option>';
        for (const p of presets) {
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = p.error ? `${p.id} [error: ${p.error}]` : p.name;
            opt.dataset.description = p.description || '';
            sel.appendChild(opt);
        }
        if (prev) sel.value = prev;
        sel.onchange = () => {
            const opt = sel.options[sel.selectedIndex];
            if (desc) desc.textContent = opt ? (opt.dataset.description || '') : '';
        };
        sel.onchange();
    } catch (e) {
        console.warn('refreshStep7Presets failed:', e);
        if (desc) desc.textContent = `Error loading presets: ${e}`;
    }
}

async function importStep7Preset() {
    const sel = document.getElementById('s7-preset-select');
    const desc = document.getElementById('s7-preset-desc');
    if (!sel || !sel.value) {
        alert('Pick a preset first.');
        return;
    }
    try {
        const data = await api(`/api/step/7/presets/${encodeURIComponent(sel.value)}`);
        if (data.error) {
            alert(`Preset error: ${data.error}`);
            return;
        }
        const params = data.params || {};
        let applied = 0, skipped = [];
        for (const [key, val] of Object.entries(params)) {
            const fn = STEP7_PRESET_APPLIERS[key];
            if (fn) { fn(val); applied++; }
            else { skipped.push(key); }
        }
        if (desc) {
            const msg = `Imported "${data.name}": ${applied} field(s) updated` +
                (skipped.length ? ` (ignored: ${skipped.join(', ')})` : '');
            desc.textContent = msg;
            desc.style.color = 'var(--magenta-light)';
            setTimeout(() => { desc.style.color = 'var(--text-dim)'; }, 4000);
        }
    } catch (e) {
        alert(`Failed to import preset: ${e}`);
    }
}

// ── Step 7 "Use recommended defaults" quick-start ──────────────────────────
// Fills the form with the defaults the user should use ~95% of the time.
// Auto-reads imgsz from the currently selected run's args.yaml; falls back
// to 1024 if no run is selected or args.yaml is missing.
/**
 * Say, under the Evaluate confidence field, that a confidence saved before
 * the measurement default was fixed has been replaced. Called on every
 * project load; a project that saved nothing retired hides the line.
 * @param {number|null|undefined} replaced the confidence VEGA replaced
 */
function showStep7ConfUpgrade(replaced) {
    const el = document.getElementById('s7-conf-upgraded');
    if (!el) return;
    if (replaced == null) {
        el.style.display = 'none';
        el.textContent = '';
        el.removeAttribute('title');
        return;
    }
    el.textContent = `This project was saved with confidence ${replaced}, and VEGA set it to `
        + `${EVAL_DEFAULTS.conf_threshold} when the project opened.`;
    el.title = `VEGA replaced the saved confidence because ${replaced} leaves the `
        + `low-confidence tail out of average precision, so the score reads lower than the `
        + `weights earn. An evaluation already in the ledger keeps the number it was `
        + `measured with.`;
    // The reason opens on keyboard focus as well as on hover, the way the
    // off-recommendation notes beside the same fields do.
    el.tabIndex = 0;
    el.style.display = '';
}

// Each Evaluate setting VEGA recommends a value for, and the field that holds
// it. The keys are the ones the server reports in
// project state (eval_settings_off_recommended) and in
// GET /api/step/7/run_info (off_recommended).
const EVAL_SETTING_FIELDS = {
    split: 's7-split',
    conf_threshold: 's7-conf',
    iou_threshold: 's7-iou',
    preview_count: 's7-preview-count',
};
// What each setting is called in the sentence beside its field.
const EVAL_SETTING_NAMES = {
    split: 'split',
    conf_threshold: 'confidence',
    iou_threshold: 'overlap',
    preview_count: 'preview count',
};
// Why the difference matters, one sentence per setting.
const EVAL_SETTING_NOTES = {
    split: 'The ledger records which images each score was measured on, so a run scored on another split is not ranked against the rest.',
    conf_threshold: 'Average precision is the area under the whole precision-recall curve, so a higher cut-off leaves its low-confidence tail out and the score reads lower than the weights earn.',
    iou_threshold: 'The overlap decides how closely a prediction must sit on a labelled object to count, so two runs measured at different overlaps are not one measurement.',
    preview_count: 'The preview count decides how many sample overlays the report carries and changes no score.',
};
// The id of the note beside a field is the field's id plus this.
const EVAL_OFF_NOTE_SUFFIX = '-off-recommended';

/**
 * The note element beside one Evaluate field, made on first use when the page
 * does not ship one, so the sentence sits with the field it is about.
 * @param {string} key an EVAL_SETTING_FIELDS key
 * @returns {Element|null} the note, or null when this page has no such field
 */
function evalOffNoteElement(key) {
    const fieldId = EVAL_SETTING_FIELDS[key];
    if (!fieldId) return null;
    const noteId = fieldId + EVAL_OFF_NOTE_SUFFIX;
    const existing = document.getElementById(noteId);
    if (existing) return existing;
    const field = document.getElementById(fieldId);
    if (!field || !field.parentNode) return null;
    const note = document.createElement('div');
    note.id = noteId;
    note.className = 'hint';
    note.style.color = 'var(--warning)';
    note.style.display = 'none';
    // The line is one sentence and its reason is a tooltip, which the house
    // bubble opens on hover and on keyboard focus (spec sections 6 and 7), so
    // the line is reachable from the keyboard like the legend items are.
    note.tabIndex = 0;
    field.parentNode.appendChild(note);
    return note;
}

/**
 * Say, beside each Evaluate field, that this project opens on a value VEGA
 * does not recommend. A project saved with an overlap of 0.7 says so next to
 * the field whose help recommends 0.6, because the number decides which other
 * runs its score may be ranked against.
 * @param {Object|null} off {key: {saved, recommended}} for every differing
 *   setting; {} when every saved setting matches, and null or undefined when
 *   the server stated nothing, which leaves the notes as they are
 */
function renderEvalOffRecommended(off) {
    if (!off || typeof off !== 'object') return;
    Object.keys(EVAL_SETTING_FIELDS).forEach(key => {
        const note = evalOffNoteElement(key);
        if (!note) return;
        const stated = off[key];
        if (!stated) {
            note.textContent = '';
            note.removeAttribute('title');
            note.style.display = 'none';
            return;
        }
        const name = EVAL_SETTING_NAMES[key];
        note.textContent = `This project is saved with ${name} ${stated.saved}, `
            + `and VEGA recommends ${stated.recommended}.`;
        note.title = EVAL_SETTING_NOTES[key];
        note.style.display = '';
    });
}

/**
 * The measurement settings the server recommends for the Evaluate panel,
 * read from config/pipeline.yaml by GET /api/step/7/run_info so the button,
 * the stored default and the launch fallback cannot disagree.
 * @param {string} runDir the selected training run, or '' when none is picked
 * @returns {Promise<object>} the server's recommended block, or EVAL_DEFAULTS
 *   when no run is picked yet or the call fails
 */
async function step7RecommendedDefaults(runDir) {
    if (!runDir) return EVAL_DEFAULTS;
    try {
        const info = await api(`/api/step/7/run_info?path=${encodeURIComponent(runDir)}`);
        const rec = (info && info.recommended) || null;
        return (rec && rec.conf_threshold != null) ? rec : EVAL_DEFAULTS;
    } catch (e) {
        console.warn('step7RecommendedDefaults fell back to the frozen defaults:', e);
        return EVAL_DEFAULTS;
    }
}

async function applyStep7Defaults() {
    const status = document.getElementById('s7-defaults-status');
    const runSel = document.getElementById('s7-run-dir');
    const runDir = runSel ? (runSel.value || '') : '';
    const rec = await step7RecommendedDefaults(runDir);
    const splitSel = document.getElementById('s7-split');
    if (splitSel) splitSel.value = rec.split || 'test';
    setVal('s7-conf', rec.conf_threshold);
    setVal('s7-iou', rec.iou_threshold);
    setVal('s7-preview-count', rec.preview_count);
    // The image size follows the run again, even after a hand edit.
    const imgszEl = document.getElementById('s7-imgsz');
    if (imgszEl) delete imgszEl.dataset.userEdited;
    const imgszMsg = await prefillStep7ImageSize(runDir);
    if (status) {
        status.textContent = `Applied recommended defaults (split=${splitSel ? splitSel.value : 'test'}, `
            + `conf=${rec.conf_threshold}, iou=${rec.iou_threshold}, preview=${rec.preview_count}; `
            + `image size: ${imgszMsg}).`;
        status.style.color = 'var(--magenta-light)';
        setTimeout(() => { status.style.color = 'var(--text-dim)'; }, 5000);
    }
}

// The sentence under the Evaluate image-size field.
const IMGSZ_SOURCE_NONE = 'Pick a training run to read the image size it trained at.';
const IMGSZ_SOURCE_MISSING = 'The run records no image size; leave the field blank and VICARIUS fills it when the evaluation starts.';

/**
 * Prefill the Evaluate image size from the selected run's args.yaml and say
 * where the value came from. A value typed during this session is kept.
 * @param {string} runDir the selected run folder, or '' for none
 * @returns {Promise<string>} the sentence shown beside the field
 */
async function prefillStep7ImageSize(runDir) {
    const imgszEl = document.getElementById('s7-imgsz');
    const sourceEl = document.getElementById('s7-imgsz-source');
    let message = IMGSZ_SOURCE_NONE;
    if (runDir) {
        try {
            const info = await api(`/api/step/7/run_info?path=${encodeURIComponent(runDir)}`);
            // The same payload carries every saved setting that is off the
            // recommendation, so the notes beside the fields are refreshed
            // here rather than through a second read of the same route.
            renderEvalOffRecommended(info && info.off_recommended);
            const trained = info && Number.isFinite(Number(info.imgsz)) ? Number(info.imgsz) : 0;
            if (trained > 0) {
                if (imgszEl && !imgszEl.dataset.userEdited) imgszEl.value = trained;
                message = `From the run's args.yaml: ${trained}`;
            } else {
                if (imgszEl && !imgszEl.dataset.userEdited) imgszEl.value = '';
                message = IMGSZ_SOURCE_MISSING;
            }
        } catch (e) {
            message = `Could not read the run's args.yaml (${e}); VICARIUS fills the size when the evaluation starts.`;
        }
    }
    if (sourceEl) sourceEl.textContent = message;
    return message;
}

// Fires when a different training run is picked in the Evaluate panel, and
// when the panel opens: the image size follows the run and the report card
// shows that run's evaluation.
async function onStep7RunChange() {
    const runSel = document.getElementById('s7-run-dir');
    const runDir = runSel ? (runSel.value || '') : '';
    await prefillStep7ImageSize(runDir);
    await loadStep7ReportForSelectedRun();
}

// ── Device selector, shared by the Train (6) and Inference (8) panels ─────
// Each panel carries s<step>-device-preset (a select whose last option is the
// custom sentinel) and s<step>-device-custom (a free-form string shown only
// for that sentinel). The value is forwarded as ultralytics `device=`.
const DEVICE_CUSTOM_SENTINEL = '__custom__';
const TRAIN_DEVICE_DEFAULT = '0,1';
const INFER_DEVICE_DEFAULT = '0';

function stepDeviceElements(step) {
    return {
        sel: document.getElementById(`s${step}-device-preset`),
        custom: document.getElementById(`s${step}-device-custom`),
    };
}

/**
 * Show the custom device field only while the preset is the custom sentinel.
 * @param {number} step 6 or 8
 */
function onStepDeviceChange(step) {
    const { sel, custom } = stepDeviceElements(step);
    if (!sel || !custom) return;
    custom.style.display = (sel.value === DEVICE_CUSTOM_SENTINEL) ? '' : 'none';
}

/**
 * The device string a panel sends: the preset, or the trimmed custom string.
 * @param {number} step 6 or 8
 * @returns {string|null} null when nothing is chosen
 */
function collectStepDevice(step) {
    const { sel, custom } = stepDeviceElements(step);
    if (!sel) return null;
    if (sel.value === DEVICE_CUSTOM_SENTINEL) {
        const v = (custom && custom.value || '').trim();
        return v || null;
    }
    return sel.value || null;
}

/**
 * Restore a saved device into a panel's selector: a preset when it is one of
 * the select's own options, the custom sentinel plus the string otherwise,
 * and the panel's default when nothing was saved.
 * @param {number} step 6 or 8
 * @param {string|null|undefined} saved the device from the saved config
 * @param {string} fallback the panel's default device
 */
function applyStepDevice(step, saved, fallback) {
    const { sel, custom } = stepDeviceElements(step);
    if (!sel) return;
    const presets = new Set(Array.from(sel.options).map(o => o.value).filter(v => v !== DEVICE_CUSTOM_SENTINEL));
    const want = saved || fallback;
    if (presets.has(want)) { sel.value = want; if (custom) custom.value = ''; }
    else { sel.value = DEVICE_CUSTOM_SENTINEL; if (custom) custom.value = want; }
    onStepDeviceChange(step);
}

// ── Step 6 "Continue from a VEGA model" (2026-09-21) ────────────────────────
// The advanced disclosure that replaced the old "Continue from a previous
// run" optgroup inside the starting-model select: #s6-continue-from lists
// the champion, this project's own runs, other projects' runs and runs
// trained outside any project, each an absolute best.pt path. Built from two
// routes: /api/step/6/list_runs for who the champion is and which task it
// trained, and /api/step/6/all_runs (the Trained runs card's own route,
// module-wide rather than this project's runs/ alone) for every candidate
// run's name, task, family and score.

// The buckets #s6-continue-from lists, in order, and the optgroup each fills.
const STEP6_CONTINUE_GROUPS = [
    { key: 'champion', id: 's6-continue-champion', label: 'Champion' },
    { key: 'mine', id: 's6-continue-mine', label: "This project's runs" },
    { key: 'other', id: 's6-continue-other', label: 'Other projects' },
    { key: 'lab', id: 's6-continue-lab', label: 'Trained outside a project' },
];
// A run's metric family, named the way its Continue-from option text reads
// it ("outline mAP50-95", "box mAP50-95"); local to this file rather than
// shared with orchestrator_runs.js so the two stay independently testable.
const STEP6_CONTINUE_FAMILY_WORDS = { mask: 'outline', box: 'box' };
const STEP6_CONTINUE_TASK_TO_FAMILY = { seg: 'mask', detect: 'box' };
const STEP6_CONTINUE_SCORE_DECIMALS = 3;
const STEP6_CONTINUE_UNNAMED_RUN = '(unnamed run)';

/**
 * Whether a run folder sits under the project open in this window, decided
 * the same way the Trained runs card decides it (its own isOpenProjectRun):
 * on the run folder sitting under the project folder, never on a name match.
 * @param {string} runDir a run's absolute folder
 * @returns {boolean}
 */
function step6RunIsOpenProject(runDir) {
    const dir = String((state && state.project_dir) || '').replace(/\/+$/, '');
    const rd = String(runDir || '').replace(/\/+$/, '');
    if (!dir || !rd) return false;
    return rd === dir || rd.indexOf(dir + '/') === 0;
}

/**
 * The mAP50-95 label for a run's metric family, falling back to its task when
 * the family is blank, else bare "mAP50-95".
 * @param {string} family "mask" or "box"
 * @param {string} task "seg" or "detect", used when family is blank
 * @returns {string}
 */
function step6ContinueFamilyLabel(family, task) {
    const fam = STEP6_CONTINUE_FAMILY_WORDS[family] ? family : (STEP6_CONTINUE_TASK_TO_FAMILY[task] || '');
    return (STEP6_CONTINUE_FAMILY_WORDS[fam] ? STEP6_CONTINUE_FAMILY_WORDS[fam] + ' ' : '') + 'mAP50-95';
}

/**
 * The text of one Continue-from option: the run's name, and its score when it
 * has one.
 * @param {object} row a normalised /api/step/6/all_runs entry (or the
 *   minimal champion stand-in refreshStep6ContinueOptions builds when the
 *   champion is missing from that list)
 * @returns {string}
 */
function step6ContinueOptionText(row) {
    const name = (row.run_name && String(row.run_name).trim()) || STEP6_CONTINUE_UNNAMED_RUN;
    const score = (typeof row.map50_95 === 'number' && Number.isFinite(row.map50_95)) ? row.map50_95 : null;
    if (score === null) return `${name}: not scored`;
    return `${name}: ${step6ContinueFamilyLabel(row.family, row.task)} ${score.toFixed(STEP6_CONTINUE_SCORE_DECIMALS)}`;
}

/**
 * One <option> for a run under "Continue from a VEGA model": its absolute
 * best.pt path as the value, its name and score as the text, and the task it
 * trained stamped on it so applyStep6ContinueTaskFilter() can hide a run of
 * the other task.
 * @param {object} row a normalised /api/step/6/all_runs entry
 * @returns {Element}
 */
function step6ContinueOption(row) {
    const opt = document.createElement('option');
    opt.value = String(row.run_dir).replace(/\/$/, '') + WEIGHTS_REL;
    opt.textContent = step6ContinueOptionText(row);
    opt.dataset.task = row.task === 'detect' ? 'detect' : 'seg';
    opt.dataset.runName = (row.run_name && String(row.run_name).trim()) || STEP6_CONTINUE_UNNAMED_RUN;
    return opt;
}

/**
 * A one-off <option> for a "Continue from" value that is not (or not yet) in
 * the fetched run list: a champion or hand-off path applyStep6ModelPath is
 * asked to select before refreshStep6ContinueOptions() has loaded, or a value
 * a rebuild carries forward that the fresh server answer does not happen to
 * name. Stamped with the training task showing now, the way every fetched
 * option is, so the task filter treats it like any other.
 * @param {string} v the run's absolute best.pt path
 * @returns {Element}
 */
function step6ContinueOneOffOption(v) {
    const opt = document.createElement('option');
    opt.value = v;
    const parts = String(v).split('/');
    opt.textContent = parts.slice(-3).join('/');
    const taskEl = document.getElementById('s6-task');
    opt.dataset.task = (taskEl && taskEl.value === 'detect') ? 'detect' : 'seg';
    opt.dataset.runName = parts.length >= 3 ? parts[parts.length - 3] : v;
    return opt;
}

/**
 * Every <option> refreshStep6ContinueOptions built inside #s6-continue-from,
 * whether inside one of its four optgroups or added loose by
 * applyStep6ModelPath. Walked by the presence of dataset.task (every real
 * option carries one; the leading "None" option and the optgroup wrappers
 * never do) rather than by tag name, so the walk needs nothing browser-only.
 * @param {Element} sel #s6-continue-from
 * @returns {Array<Element>}
 */
function step6ContinueOptionElements(sel) {
    const out = [];
    const visit = (node) => {
        for (const child of Array.from((node && node.children) || [])) {
            if (child.dataset && child.dataset.task) out.push(child);
            else visit(child);
        }
    };
    visit(sel);
    return out;
}

// The fixed lead of the "Continue from a VEGA model" disclosure summary; only
// the count after it changes.
const STEP6_CONTINUE_SUMMARY_PREFIX = 'Continue from a VEGA model (advanced): ';

/**
 * The disclosure summary text for a run count the task filter is offering.
 * @param {number} shown how many "Continue from" options match the task now
 * @returns {string}
 */
function step6ContinueSummaryText(shown) {
    if (shown === 0) return STEP6_CONTINUE_SUMMARY_PREFIX + 'no runs to choose from yet';
    if (shown === 1) return STEP6_CONTINUE_SUMMARY_PREFIX + '1 run to choose from';
    return STEP6_CONTINUE_SUMMARY_PREFIX + `${shown} runs to choose from`;
}

/**
 * Hide and disable every "Continue from" option whose task does not match the
 * training task now, write the visible count into the disclosure summary,
 * and clear a selection the filter just hid (through onStep6ContinueChange,
 * so the freeze/learning-rate note and the starting-model select follow). The
 * summary is left as the template wrote it until refreshStep6ContinueOptions()
 * has completed at least once (marked by #s6-continue-from's own
 * dataset.loaded): before that, "0 offered" would misreport an unread count
 * as a known, empty one.
 */
function applyStep6ContinueTaskFilter() {
    const sel = document.getElementById('s6-continue-from');
    if (!sel) return;
    const wantTask = step6TaskValue() === 'detect' ? 'detect' : 'seg';
    let shown = 0;
    let hidSelected = false;
    for (const opt of step6ContinueOptionElements(sel)) {
        const match = opt.dataset.task === wantTask;
        opt.hidden = !match;
        opt.disabled = !match;
        if (match) shown += 1;
        else if (sel.value && opt.value === sel.value) hidSelected = true;
    }
    if (sel.dataset.loaded === '1') {
        const summary = document.getElementById('s6-continue-summary');
        if (summary) summary.textContent = step6ContinueSummaryText(shown);
    }
    if (hidSelected) {
        sel.value = '';
        onStep6ContinueChange();
    }
}

// Request token for #s6-continue-from's rebuild. switchStep(6) fires a
// refresh in the background after an awaited /api/project/state, and
// gotoStep6Start('champion') and onVegaFinetuneFrom each fire one of their
// own; two of these can be in flight together, and the slower one used to win
// by finishing last, restoring whatever value it saw when IT started and
// wiping a champion or hand-off pick made in between (found by the parity
// reviewer, 2026-09-22). Every refresh bumps this counter and checks it again
// after its fetch: a call whose token no longer matches the latest one
// discards its whole answer rather than touch the select at all, since a
// newer refresh (or a direct hand-off through applyStep6ModelPath, which
// bumps the same counter) already owns it. This is the same pattern as
// _s6DataSummaryReq.
let _s6ContinueReq = 0;

/**
 * Fill "Continue from a VEGA model" with the champion, this project's own
 * runs, other projects' runs and runs trained outside any project, then
 * apply the task filter so only the runs matching the training task now are
 * pickable. Replaces refreshStep6ModelRunOptions(): the group that used to
 * live inside the starting-model select now has its own advanced disclosure,
 * built from the module-wide run list (/api/step/6/all_runs) rather than
 * just this project's own runs, which still supplies who the champion is.
 * @returns {Promise<void>}
 */
async function refreshStep6ContinueOptions() {
    const sel = document.getElementById('s6-continue-from');
    if (!sel) return;
    const token = ++_s6ContinueReq;
    try {
        const [runsData, allData] = await Promise.all([
            api('/api/step/6/list_runs'),
            api('/api/step/6/all_runs'),
        ]);
        // A newer refresh, or a hand-off through applyStep6ModelPath, started
        // while this fetch was in flight and now owns the select; this
        // slower answer is discarded whole; not even its option list is
        // applied, so it can never overwrite a fresher rebuild either.
        if (token !== _s6ContinueReq) return;

        const rows = (allData.runs || []).filter(r => r && r.run_dir && r.has_weights);
        const champDir = String(runsData.champion_run_dir || '').replace(/\/$/, '');
        // The value to carry across the rebuild, read as late as possible
        // (never captured before the fetch above): a hand-off made while
        // this fetch was in flight, and still current per the check just
        // above, is what survives, not whatever was selected when this call
        // started.
        const currentValue = sel.value;

        // Rebuild from scratch: every child but the leading "None" option is
        // ours to replace, including a one-off option a saved model_path may
        // have added directly onto the select before this ever loaded.
        while (sel.children.length > 1) sel.removeChild(sel.children[sel.children.length - 1]);

        let championRow = null;
        const mine = [];
        const other = [];
        const lab = [];
        for (const row of rows) {
            const dir = String(row.run_dir).replace(/\/$/, '');
            if (champDir && dir === champDir) { championRow = row; continue; }
            if (row.origin === 'lab') { lab.push(row); continue; }
            if (step6RunIsOpenProject(row.run_dir)) mine.push(row);
            else other.push(row);
        }
        // A promoted champion missing from all_runs (its folder just vanished,
        // or has_weights is briefly false while a fresh run writes its
        // weights) still gets an option: the launch preflight is what refuses
        // a genuinely missing file, and this panel must not silently drop the
        // model Model Inference is using.
        if (champDir && !championRow) {
            championRow = {
                run_dir: champDir,
                run_name: champDir.split('/').filter(Boolean).pop() || champDir,
                task: runsData.champion_task || '', family: '', map50_95: null,
            };
        }
        const buckets = { champion: championRow ? [championRow] : [], mine: mine, other: other, lab: lab };
        for (const group of STEP6_CONTINUE_GROUPS) {
            const list = buckets[group.key];
            if (!list.length) continue;
            const optgroup = document.createElement('optgroup');
            optgroup.id = group.id;
            optgroup.label = group.label;
            for (const row of list) optgroup.appendChild(step6ContinueOption(row));
            sel.appendChild(optgroup);
        }
        // Cached on the select itself so gotoStep6Start('champion') can ask
        // "is there a champion, and does it match the task on the form now"
        // without walking the option tree a second time.
        sel.dataset.championValue = championRow
            ? (String(championRow.run_dir).replace(/\/$/, '') + WEIGHTS_REL) : '';
        sel.dataset.championTask = championRow ? (championRow.task === 'detect' ? 'detect' : 'seg') : '';

        // The value carries forward whether or not the fresh list happens to
        // name it: a one-off option is added when it does not, the same way
        // applyStep6ModelPath would, rather than blanking a real pick because
        // this particular fetch did not include it. The task filter below is
        // the only place a value is ever cleared, and only when its own task
        // does not match.
        if (currentValue) {
            const stillThere = step6ContinueOptionElements(sel).some(o => o.value === currentValue);
            if (!stillThere) sel.appendChild(step6ContinueOneOffOption(currentValue));
            sel.value = currentValue;
        }
        // Marks the count as known so the task filter's summary line may
        // report it; left unset on a failed fetch below, where the count is
        // still genuinely unread.
        sel.dataset.loaded = '1';
        applyStep6ContinueTaskFilter();
    } catch (e) {
        console.warn('refreshStep6ContinueOptions failed:', e);
    }
}

// Step 6 training source (2026-08-26): '' = this project's Step 4 export,
// else the path of a composite export under inprocess/_composite/ (built on
// the coverage matrix). Re-read every time the panel opens so a composite
// just built shows up without a reload; the prior pick survives when its
// dir still exists. `want` (saved config) wins over the current selection.
async function refreshStep6Sources(want) {
    // A box-dataset project trains on its own dataset and the card is hidden,
    // so there is nothing to fill and no reason to ask the server.
    if (isBoxDatasetProject()) return;
    const sel = document.getElementById('s6-source');
    if (!sel) return;
    const note = document.getElementById('s6-source-note');
    const prev = (want !== undefined && want !== null) ? String(want) : sel.value;
    try {
        const data = await api('/api/composites');
        const list = data.composites || [];
        sel.innerHTML = '';
        const own = document.createElement('option');
        own.value = '';
        own.textContent = "This project's Step 4 export";
        own.title = "The all_images/ + all_labels/ + data.yaml written by this project's Step 4 annotator.";
        sel.appendChild(own);
        for (const c of list) {
            const opt = document.createElement('option');
            opt.value = c.path;
            // Three kinds: a staged box dataset, an Annotate Images session
            // (decision record 0006), or a training set from the coverage matrix.
            opt.dataset.kind = c.kind === 'bbox' ? 'bbox' : (c.kind === 'standalone' ? 'standalone' : 'composite');
            opt.textContent = step6SourceOptionText(c);
            opt.title = step6SourceOptionTitle(c);
            sel.appendChild(opt);
        }
        if (prev && Array.from(sel.options).some(o => o.value === prev)) {
            sel.value = prev;
        } else if (prev) {
            // A saved composite that no longer exists must not silently
            // retarget training; say so and fall back to the project export.
            sel.value = '';
            if (note) note.textContent = `Saved training source is gone (${prev}); using this project's Step 4 export.`;
            applyStep6BoxDatasetMode();
            syncStep6ClassesToSource(sel);
            return;
        }
        if (note) {
            const matrixLink = '<a href="/matrix#training" target="_blank" rel="noopener"'
                + ' title="Open the coverage matrix Training Sets tab in a new tab to assemble a'
                + ' composite export across projects.">the coverage matrix</a>';
            const nSession = list.filter(c => c.kind === 'standalone').length;
            const nBox = list.filter(c => c.kind === 'bbox').length;
            const nComposite = list.length - nSession - nBox;
            const parts = [];
            if (nComposite) parts.push(`${escHtmlNav(String(nComposite))} training set(s) under ${escHtmlNav(data.root || 'inprocess/_composite/')}`);
            if (nSession) parts.push(`${escHtmlNav(String(nSession))} Annotate Images session(s)`);
            if (nBox) parts.push(`${escHtmlNav(String(nBox))} staged box dataset(s)`);
            note.innerHTML = list.length
                ? parts.join(', ') + '.'
                : `No training sets yet. Build one on ${matrixLink}.`;
        }
        applyStep6BoxDatasetMode();
        syncStep6ClassesToSource(sel);
    } catch (e) {
        console.warn('refreshStep6Sources failed:', e);
    }
}

// How many class names a training-source option names before it counts the
// rest. Every name is in the option's tooltip. A staged 59-class box dataset
// once wrote 984 characters into one option and stretched the select to
// 7225 px inside a 946 px window.
const STEP6_SOURCE_LABELS_SHOWN = 3;

/**
 * The option text for one training-source row from /api/composites: a
 * composite shows its mask count; a staged box dataset says so, since it has
 * no masks (its n_masks is null and must not be printed). At most
 * STEP6_SOURCE_LABELS_SHOWN class names are named and the rest are counted,
 * so one dataset with many classes cannot stretch the select past the window;
 * step6SourceOptionTitle carries the full list.
 * @param {object} c one row: name, kind, n_images, n_masks, labels
 * @returns {string}
 */
function step6SourceOptionText(c) {
    const names = (c.labels && c.labels.length) ? c.labels : [];
    const shown = names.slice(0, STEP6_SOURCE_LABELS_SHOWN);
    const rest = names.length - shown.length;
    const labels = shown.length
        ? ` (${shown.join(', ')}${rest > 0 ? `, and ${rest} more` : ''})`
        : '';
    const count = c.kind === 'bbox' ? 'box dataset' : `${c.n_masks} masks`;
    const origin = c.kind === 'standalone' ? ' (Annotate Images session)' : '';
    return `${c.name}${origin}: ${c.n_images} images, ${count}${labels}`;
}

/**
 * The tooltip for one training-source option: where the source is, when it
 * was built, and every class name in it, since the option text names only the
 * first few.
 * @param {object} c one row: path, created_at, labels
 * @returns {string}
 */
function step6SourceOptionTitle(c) {
    const names = (c.labels && c.labels.length) ? c.labels : [];
    return `${c.path}${c.created_at ? ' | built ' + c.created_at : ''}`
        + (names.length ? ` | ${names.length} class(es): ${names.join(', ')}` : '');
}

// The class picker must show the ids of the source that will train (H2,
// 2026-08-26): once the source select settles, re-fetch the picker whenever
// it was loaded from a different source (or never loaded).
function syncStep6ClassesToSource(sel) {
    const host = document.getElementById('s6-classes-list');
    if (!host || !sel) return;
    if ((host.dataset.source || '') !== (sel.value || '')) refreshStep6Classes();
}

// ── Step 6 training task ────────────────────────────────────────────────────
// The starting-model catalog follows the task, because the two are not
// interchangeable: ultralytics reads the head from the checkpoint and ignores
// the task it is handed, so segmentation weights with task=detect would train
// the wrong model. The server refuses that pairing outright; showing only the
// matching catalog is what keeps anyone from reaching it by accident.
const STEP6_SEG_OPTGROUPS = ['s6-model-seg-optgroup', 's6-model-seg26-optgroup'];
const STEP6_DETECT_OPTGROUP = 's6-model-detect-optgroup';

/**
 * Show only the starting-model catalog for the training task now: the two
 * segmentation optgroups (YOLO11 and YOLO26) or the one detection optgroup.
 * "Continue from a VEGA model" is a separate disclosure now and follows the
 * task through its own filter, applyStep6ContinueTaskFilter().
 * @param {'detect'|'segment'} task
 */
function applyStep6TaskToModelPicker(task) {
    const detect = task === 'detect';
    STEP6_SEG_OPTGROUPS.forEach(id => {
        const g = document.getElementById(id);
        if (g) g.hidden = detect;
    });
    const dg = document.getElementById(STEP6_DETECT_OPTGROUP);
    if (dg) dg.hidden = !detect;
}

// Same size, other task: yolo11m-seg.pt <-> yolo11m.pt. Only the two stock
// YOLO11 catalogs convert; a YOLO26 pick is left alone, and the server
// refuses it if the pairing is genuinely wrong.
function step6SiblingWeights(value, task) {
    const v = String(value || '');
    const m = v.match(/^yolo11([nsmlx])(-seg)?\.pt$/);
    if (!m) return '';
    return task === 'detect' ? `yolo11${m[1]}.pt` : `yolo11${m[1]}-seg.pt`;
}

/**
 * Whether a starting-model catalog value trains the given task, read from the
 * value's own name rather than the DOM: every segmentation checkpoint in
 * #s6-model, YOLO11 or YOLO26, ends in "-seg.pt"; every detection checkpoint
 * does not.
 * @param {string} value a #s6-model option value
 * @param {'detect'|'segment'} task
 * @returns {boolean}
 */
function step6ModelValueMatchesTask(value, task) {
    const v = String(value || '');
    if (!v) return true; // nothing chosen is never a mismatch
    const isSegValue = /-seg\.pt$/.test(v);
    return task === 'detect' ? !isSegValue : isSegValue;
}

function onStep6TaskChange() {
    const taskEl = document.getElementById('s6-task');
    const sel = document.getElementById('s6-model');
    if (!taskEl) return;
    const task = taskEl.value === 'detect' ? 'detect' : 'segment';
    applyStep6TaskToModelPicker(task);
    if (sel) {
        const sibling = step6SiblingWeights(sel.value, task);
        // Carry the chosen SIZE across the task change; a YOLO26 pick has no
        // sibling and falls through to the catalog fallback below.
        if (sibling && Array.from(sel.options).some(o => o.value === sibling)) {
            sel.value = sibling;
        }
        // A value that now belongs to the hidden catalog must never stay
        // selected: the form would show one thing and the launch would carry
        // another. Fall back to the recommended catalog weights, which is
        // what the field documents as its default.
        if (sel.value && !step6ModelValueMatchesTask(sel.value, task)) {
            const fallback = task === 'detect' ? 'yolo11m.pt' : 'yolo11m-seg.pt';
            if (Array.from(sel.options).some(o => o.value === fallback)) sel.value = fallback;
        }
    }
    // The task is half of what chooses a training recipe, so the recipe list
    // and the line under it follow it immediately.
    renderStep6RecipeOptions();
    // A run under Continue from a VEGA model that trained the other task is
    // no longer offered; clear a pick the filter just hid.
    applyStep6ContinueTaskFilter();
}

// Freeze layers and learning rate VEGA sets while a run is chosen under
// Continue from a VEGA model: a low freeze depth (roughly the backbone of a
// YOLO11 model) and a low learning rate so a small correction batch still
// adapts what it needs to without erasing what the chosen run already learned.
const STEP6_CONTINUE_FREEZE = 10;
const STEP6_CONTINUE_LR0 = 0.001;
const STEP6_CONTINUE_REPLACED_HINT_PREFIX = 'Replaced: this run continues from ';
const STEP6_CONTINUE_REPLACED_HINT_SUFFIX =
    '. Clear the choice under Continue from a VEGA model to start from a base model.';

/**
 * Follow a "Continue from a VEGA model" pick: set Freeze layers and the
 * learning rate to VEGA's continuation values and disable the starting-model
 * select while a value is chosen, or restore the operator's own numbers when
 * the choice is cleared. The previous values are stashed on #s6-freeze's and
 * #s6-lr0's own dataset, so a page that opens with a saved model_path still
 * restores the operator's numbers if they later clear the pick, and a second
 * pick made without clearing first does not overwrite the stash with VEGA's
 * own continuation values.
 */
function onStep6ContinueChange() {
    const sel = document.getElementById('s6-continue-from');
    const freezeEl = document.getElementById('s6-freeze');
    const lr0El = document.getElementById('s6-lr0');
    const modelEl = document.getElementById('s6-model');
    const note = document.getElementById('s6-model-note');
    const value = sel ? String(sel.value || '') : '';
    if (value) {
        if (freezeEl && !('freeValue' in freezeEl.dataset)) freezeEl.dataset.freeValue = freezeEl.value;
        if (lr0El && !('freeValue' in lr0El.dataset)) lr0El.dataset.freeValue = lr0El.value;
        if (freezeEl) freezeEl.value = String(STEP6_CONTINUE_FREEZE);
        if (lr0El) lr0El.value = String(STEP6_CONTINUE_LR0);
        if (modelEl) modelEl.disabled = true;
        if (note) {
            if (!('freeText' in note.dataset)) note.dataset.freeText = note.textContent;
            const opt = step6ContinueOptionElements(sel).find(o => o.value === value);
            const runName = (opt && opt.dataset && opt.dataset.runName) || value;
            note.textContent = STEP6_CONTINUE_REPLACED_HINT_PREFIX + runName + STEP6_CONTINUE_REPLACED_HINT_SUFFIX;
        }
        return;
    }
    if (freezeEl && 'freeValue' in freezeEl.dataset) { freezeEl.value = freezeEl.dataset.freeValue; delete freezeEl.dataset.freeValue; }
    if (lr0El && 'freeValue' in lr0El.dataset) { lr0El.value = lr0El.dataset.freeValue; delete lr0El.dataset.freeValue; }
    if (modelEl) modelEl.disabled = false;
    if (note && 'freeText' in note.dataset) { note.textContent = note.dataset.freeText; delete note.dataset.freeText; }
}

/**
 * Choose a run's weights under "Continue from a VEGA model": select it in
 * #s6-continue-from, adding a one-off option first when the run is not
 * already listed there (a champion or hand-off path applied before
 * refreshStep6ContinueOptions() has loaded), open the disclosure, and run
 * onStep6ContinueChange() so the freeze/learning-rate note and the disabled
 * starting-model select follow the pick. Bumps the same request token
 * refreshStep6ContinueOptions() checks, so a rebuild already in flight when
 * this hand-off arrives finds itself stale on completion and discards its
 * whole answer instead of overwriting the pick a moment later.
 * @param {string} v the run's absolute best.pt path
 */
function applyStep6ModelPath(v) {
    if (!v) return;
    const sel = document.getElementById('s6-continue-from');
    if (!sel) return;
    _s6ContinueReq += 1;
    const exists = step6ContinueOptionElements(sel).some(o => o.value === v);
    if (!exists) sel.appendChild(step6ContinueOneOffOption(v));
    sel.value = v;
    const details = document.getElementById('s6-continue-details');
    if (details) details.open = true;
    onStep6ContinueChange();
}

// ── Step 6 class-inclusion picker ──────────────────────────────────────────

// Request token: a slow answer for an earlier source must never overwrite
// the picker loaded for the source selected now.
let _s6ClassesReq = 0;

async function refreshStep6Classes() {
    // A box-dataset project has no class picker: its classes come from the
    // dataset's own classes.txt and every one of them trains. The card is
    // hidden, and /api/step/6/list_classes would answer 400 pointing at steps
    // this project can never run.
    if (isBoxDatasetProject()) return;
    // A staged box dataset picked as the training source has no class picker
    // either; applyStep6BoxDatasetMode() hides the card.
    if (step6SelectedSourceIsBoxDataset()) return;
    const host = document.getElementById('s6-classes-list');
    if (!host) return;
    // H2 (2026-08-26): the picker follows #s6-source so the ids it shows are
    // the ids the trainer keeps. '' = this project's Step 4 export.
    const srcSel = document.getElementById('s6-source');
    const src = srcSel ? (srcSel.value || '') : '';
    host.dataset.source = src;
    const token = ++_s6ClassesReq;
    const from = src ? 'the selected training source' : "this project's Step 4 export";
    host.innerHTML = `<span style="color:var(--text-dim)">Loading classes from ${from}...</span>`;
    try {
        const url = '/api/step/6/list_classes' + (src ? `?source_dir=${encodeURIComponent(src)}` : '');
        // The label registry is read beside the classes so every code that
        // resolves is shown as the species it names. vegaLabelNames() answers
        // null when the registry is unavailable, and each code then stays as
        // it is rather than the picker waiting on it.
        const [data, names] = await Promise.all([api(url), vegaLabelNames()]);
        if (token !== _s6ClassesReq) return;
        if (data.error) {
            host.innerHTML = `<span style="color:var(--magenta-light)">Error: ${data.error}</span>`;
            return;
        }
        renderStep6Classes(data.classes || [], data.include_classes, names);
        const gone = data.saved_classes_dropped || [];
        if (gone.length) {
            const n = document.createElement('div');
            n.style.cssText = 'color:var(--text-dim);font-size:11px;margin-top:4px';
            n.textContent = 'Saved classes absent from this source (left unticked): '
                + gone.map(code => step6ClassLabelText(code, names)).join(', ');
            host.appendChild(n);
        }
    } catch (e) {
        if (token !== _s6ClassesReq) return;
        host.innerHTML = `<span style="color:var(--magenta-light)">Error loading classes: ${e}</span>`;
    }
}

// A resolved full name that is a species binomial on its own, and one that
// opens with a binomial before a comma ("Epinephelus striatus, Nassau grouper,
// dark phase"). Both match the evaluation report card, so a species reads the
// same way on the Train panel, on the report and in the PDF.
const SPECIES_BINOMIAL_RE = /^[A-Z][a-z]+(?: [a-z]+){1,2}$/;
const SPECIES_BINOMIAL_PREFIX_RE = /^([A-Z][a-z]+(?: [a-z]+){1,2})(,\s[\s\S]*)$/;

/**
 * The full name a class code stands for, or '' when nothing resolves it. A
 * registry row whose name is the code again resolves to nothing new.
 * @param {string} code the class name as the training source spells it
 * @param {Object<string,string>|null} names code -> full name, from the label registry
 * @returns {string} the full name, or ''
 */
function step6ResolveClassName(code, names) {
    const raw = String(code == null ? '' : code).trim();
    if (!raw || !names || typeof names !== 'object') return '';
    const hit = Object.prototype.hasOwnProperty.call(names, raw) ? names[raw] : '';
    const full = (typeof hit === 'string') ? hit.trim() : '';
    return (full && full.toLowerCase() !== raw.toLowerCase()) ? full : '';
}

/**
 * A resolved name as escaped HTML: italic throughout when the whole name is a
 * binomial, the binomial alone in italic when a staged name adds qualifiers
 * after a comma, upright when the name is not a species at all.
 * @param {string} name the resolved full name
 * @returns {string} HTML
 */
function vegaSpeciesNameHtml(name) {
    const text = String(name == null ? '' : name);
    const parts = SPECIES_BINOMIAL_PREFIX_RE.exec(text);
    if (parts) {
        return `<em class="vega-report-binomial">${escapeHtml(parts[1])}</em>`
            + `<span class="vega-report-plain">${escapeHtml(parts[2])}</span>`;
    }
    if (SPECIES_BINOMIAL_RE.test(text)) return `<em class="vega-report-binomial">${escapeHtml(text)}</em>`;
    return `<span class="vega-report-plain">${escapeHtml(text)}</span>`;
}

/**
 * One class as the picker shows it: the full name with the code beside it as a
 * chip. A code the registry cannot resolve stays exactly as it is.
 * @param {string} code the class name as the training source spells it
 * @param {Object<string,string>|null} names code -> full name
 * @returns {string} HTML
 */
function step6ClassLabelHtml(code, names) {
    const raw = String(code == null ? '' : code).trim();
    const full = step6ResolveClassName(raw, names);
    if (!full) return `<span class="vega-report-plain">${escapeHtml(raw)}</span>`;
    return vegaSpeciesNameHtml(full) + ` <span class="vega-report-chip">${escapeHtml(raw)}</span>`;
}

/**
 * The same name as plain text, for a tooltip or a note.
 * @param {string} code the class name as the training source spells it
 * @param {Object<string,string>|null} names code -> full name
 * @returns {string} "Porites astreoides (PA)", or "PA" when nothing resolves it
 */
function step6ClassLabelText(code, names) {
    const raw = String(code == null ? '' : code).trim();
    const full = step6ResolveClassName(raw, names);
    return full ? `${full} (${raw})` : raw;
}

function renderStep6Classes(classes, saved, names) {
    const host = document.getElementById('s6-classes-list');
    if (!host) return;
    // Belt-and-suspenders: also drop any name that is empty or literally
    // "(unnamed)" (case-insensitive) in case an older payload sneaks in.
    const UNNAMED_RE = /^\(unnamed\)$/i;
    const filtered = (classes || []).filter(c => {
        const nm = (c && c.name != null) ? String(c.name).trim() : '';
        return nm && !UNNAMED_RE.test(nm);
    });
    if (!filtered.length) {
        host.innerHTML = '<span style="color:var(--text-dim)">No usable classes found. Run Step 4 (or set Step 3 target labels) so class names are present.</span>';
        updateStep6ClassesCount();
        return;
    }
    // saved === null|undefined -> include every class (default).
    // saved is an array of IDs -> include only those.
    const savedSet = Array.isArray(saved) ? new Set(saved.map(Number)) : null;
    const rows = filtered.map(c => {
        const nm = String(c.name).trim();
        const checked = (savedSet === null || savedSet.has(c.id)) ? 'checked' : '';
        const countTxt = `${c.instance_count || 0} instances · ${c.image_count || 0} images`;
        const tip = `Tick this to train on ${step6ClassLabelText(nm, names)}, id ${c.id} in the `
            + 'selected training source. VEGA strips every class left unticked out of the '
            + 'labels before the split is built.';
        return `<label style="display:flex;align-items:center;gap:8px;padding:3px 0;cursor:pointer">
            <input type="checkbox" class="s6-class-cb" data-class-id="${c.id}" data-class-name="${escapeAttr(nm)}" ${checked} title="${escapeAttr(tip)}">
            <span style="min-width:2.5em;color:var(--text-dim);font-variant-numeric:tabular-nums">#${c.id}</span>
            <span style="flex:1">${step6ClassLabelHtml(nm, names)}</span>
            <span style="color:var(--text-dim);font-size:11px">${countTxt}</span>
        </label>`;
    }).join('');
    host.innerHTML = rows;
    updateStep6ClassesCount();
}

function selectAllStep6Classes(on) {
    document.querySelectorAll('.s6-class-cb').forEach(cb => { cb.checked = !!on; });
    updateStep6ClassesCount();
    refreshStep6DataSummary();
}

// Returns null (meaning "all classes") if the picker hasn't been loaded yet or
// every available class is checked. Returns [] if the user has unchecked
// everything (we treat that as "train nothing"; the caller errors out).
// Drops any class whose name is empty or matches "(unnamed)" (case-insensitive)
// as a last-line-of-defense so the training payload is never polluted.
function collectStep6IncludedClasses() {
    const cbs = document.querySelectorAll('.s6-class-cb');
    if (!cbs.length) return null;
    const UNNAMED_RE = /^\(unnamed\)$/i;
    let total = 0;
    const selected = [];
    cbs.forEach(cb => {
        const nm = (cb.dataset.className || '').trim();
        if (!nm || UNNAMED_RE.test(nm)) return;  // skip unnamed entirely
        total++;
        if (cb.checked) selected.push(parseInt(cb.dataset.classId));
    });
    if (selected.length === total) return null;
    return selected;
}

/**
 * Write "N of M ticked" into the class picker's heading count, read from the
 * checkboxes on the page now.
 */
function updateStep6ClassesCount() {
    const el = document.getElementById('s6-classes-count');
    if (!el) return;
    const cbs = document.querySelectorAll('.s6-class-cb');
    let ticked = 0;
    cbs.forEach(cb => { if (cb.checked) ticked += 1; });
    el.textContent = `${ticked} of ${cbs.length} ticked`;
}

// ── Step 6 training data summary (2026-09-21) ───────────────────────────────
// The Pictures and Split lines under Training data: what /api/step/6/source_
// summary answers for the controls on the form now, rendered exactly as the
// server wrote it. The route is read-only and builds nothing itself, so this
// is safe to call on every change to the controls it depends on.

// Debounce window for the two split-fraction inputs, which fire 'input' on
// every keystroke; one fetch per pause in typing, not one per digit.
const STEP6_DATA_SUMMARY_DEBOUNCE_MS = 300;
// The other controls the summary depends on, which re-read it on 'change'
// with no debounce: a select or a checkbox fires 'change' once per pick.
const STEP6_DATA_SUMMARY_TRIGGER_IDS = ['s6-source', 's6-pin-split', 's6-valid', 's6-test', 's6-task'];
let _s6DataSummaryTimer = null;
// Request token: a slow answer for an earlier query must never overwrite the
// Pictures and Split lines a newer query already wrote.
let _s6DataSummaryReq = 0;

/**
 * The query string refreshStep6DataSummary sends, built from the controls
 * /api/step/6/source_summary reads: the training source, the holdout-split
 * checkbox and its two fractions, the training task, and the ticked class ids
 * (omitted entirely when every class is ticked, which the route reads the
 * same way collectConfig(6) does: every class included).
 * @returns {string} a URLSearchParams string, no leading "?"
 */
function step6DataSummaryQuery() {
    const $ = id => document.getElementById(id);
    const params = new URLSearchParams();
    const sourceEl = $('s6-source');
    params.set('source_dir', sourceEl ? (sourceEl.value || '') : '');
    const pinEl = $('s6-pin-split');
    params.set('pin_split', (pinEl && pinEl.checked) ? '1' : '0');
    const validEl = $('s6-valid');
    const validVal = validEl ? parseFloat(validEl.value) : NaN;
    params.set('valid_ratio', String(Number.isFinite(validVal) ? validVal : 0.2));
    const testEl = $('s6-test');
    const testVal = testEl ? parseFloat(testEl.value) : NaN;
    params.set('test_ratio', String(Number.isFinite(testVal) ? testVal : 0.1));
    params.set('task', step6TaskValue());
    const classes = collectStep6IncludedClasses();
    if (Array.isArray(classes)) params.set('include_classes', classes.join(','));
    return params.toString();
}

/**
 * Put a saved model card tag into its select and mark it as a person's pick,
 * so a later inference from the training source never overwrites it.
 * @param {string} id  the select's element id
 * @param {*} saved    the saved value, '' (Not set) or undefined when none was saved
 * @param {*} picked    the saved pick flag; undefined for a project saved before
 *                      the flag existed, where a non-blank value counts as a pick
 * @returns {void}
 */
function restoreStep6CardTag(id, saved, picked) {
    const el = document.getElementById(id);
    if (!el) return;
    const value = String(saved || '').trim().toLowerCase();
    const known = value === '' || Array.from(el.options).some(o => o.value === value);
    const wasPicked = (picked === undefined || picked === null) ? value !== '' : !!picked;
    el.value = known ? value : '';
    if (known && wasPicked) el.dataset.picked = '1'; else delete el.dataset.picked;
}

// The model card subject each recipe subject suggests; Other suggests nothing.
const STEP6_CARD_SUBJECT_FROM_RECIPE = { coral: 'benthic', algae: 'benthic', fish: 'fish', other: '' };

/**
 * The model card subject to prefill: what the training source says when it
 * says anything, else what the recipe subject suggests, else '' (Not set).
 * @param {string|null} inferred  the tags route's subject, or null
 * @param {string} recipe         the recipe subject (coral, fish, algae, other)
 * @returns {string} 'fish', 'benthic' or ''
 */
function step6CardSubjectFor(inferred, recipe) {
    if (inferred === 'fish' || inferred === 'benthic') return inferred;
    return STEP6_CARD_SUBJECT_FROM_RECIPE[String(recipe || '').trim().toLowerCase()] || '';
}

/**
 * Fill the two model card rows (subject, training data) from what the
 * training source says about itself, for each row nobody has picked yet.
 * A row the server could not infer stays blank. Guarded by a request token
 * so a slow answer for a source that is no longer selected is dropped.
 * @returns {Promise<void>}
 */
async function refreshStep6CardTags() {
    const subjectEl = document.getElementById('s6-card-subject');
    const sourceEl = document.getElementById('s6-card-data-source');
    if (!subjectEl && !sourceEl) return;
    const srcSel = document.getElementById('s6-source');
    const src = srcSel ? (srcSel.value || '') : '';
    const token = ++_s6CardTagsReq;
    try {
        const data = await api('/api/step/6/tags' + (src ? `?source=${encodeURIComponent(src)}` : ''));
        if (token !== _s6CardTagsReq || !data) return;
        // A refused answer (no export yet, a source outside the roots) infers
        // nothing, but the recipe subject still prefills an unpicked row.
        const answered = !data.error;
        if (subjectEl && !subjectEl.dataset.picked) {
            subjectEl.value = step6CardSubjectFor(answered ? data.subject : null, step6SubjectValue());
        }
        if (answered && sourceEl && data.data_source && !sourceEl.dataset.picked) sourceEl.value = data.data_source;
    } catch (e) {
        // The rows stay as they are; the person can still pick both.
        const el = document.getElementById('s6-card-subject');
        if (el && !el.dataset.picked) el.value = step6CardSubjectFor(null, step6SubjectValue());
    }
}
let _s6CardTagsReq = 0;

/**
 * Re-read the training data sentences: how many pictures and labels the
 * source, classes and split controls describe now, and how the split divides
 * them. Both sentences come whole from the server and are written exactly as
 * it sent them; this formats nothing itself. On an error response (a source
 * outside the allowed roots, a missing folder, no project loaded), the
 * server's own error sentence goes into both lines instead. Guarded by a
 * request token so a slow answer for a query that is no longer current can
 * never overwrite what a newer one already wrote.
 * @returns {Promise<void>}
 */
async function refreshStep6DataSummary() {
    const picEl = document.getElementById('s6-data-pictures');
    const splitEl = document.getElementById('s6-data-split');
    if (!picEl && !splitEl) return;
    const token = ++_s6DataSummaryReq;
    const query = step6DataSummaryQuery();
    try {
        const data = await api(`/api/step/6/source_summary?${query}`);
        if (token !== _s6DataSummaryReq) return;
        if (data && data.error) {
            if (picEl) picEl.textContent = data.error;
            if (splitEl) splitEl.textContent = data.error;
            return;
        }
        if (picEl) picEl.textContent = (data && data.pictures_sentence) || '';
        if (splitEl) splitEl.textContent = (data && data.split && data.split.sentence) || '';
    } catch (e) {
        if (token !== _s6DataSummaryReq) return;
        const msg = `Could not read the training data: ${e}`;
        if (picEl) picEl.textContent = msg;
        if (splitEl) splitEl.textContent = msg;
    }
}

/**
 * Debounced entry point for the two fraction inputs' 'input' event, so a run
 * of keystrokes sends one query instead of one per digit.
 */
function scheduleStep6DataSummary() {
    if (_s6DataSummaryTimer) clearTimeout(_s6DataSummaryTimer);
    _s6DataSummaryTimer = setTimeout(() => {
        _s6DataSummaryTimer = null;
        refreshStep6DataSummary();
    }, STEP6_DATA_SUMMARY_DEBOUNCE_MS);
}

async function refreshRunList() {
    try {
        const data = await api('/api/step/6/list_runs');
        const runs = data.runs || [];
        const championRunDir = data.champion_run_dir || null;
        const populate = (selId) => {
            const sel = document.getElementById(selId);
            if (!sel) return;
            const prev = sel.value;
            sel.innerHTML = '';
            if (!runs.length) {
                const opt = document.createElement('option');
                opt.value = '';
                opt.textContent = `(no training runs yet, finish Train Model (step ${stepDisplayNum(6)}) first)`;
                sel.appendChild(opt);
            }
            for (const r of runs) {
                const opt = document.createElement('option');
                opt.value = r.path;
                let label = r.name;
                if (r.model || r.epochs) label += `: ${r.model || '?'} · ${r.epochs || '?'} epochs`;
                if (!r.has_best && r.has_last) label += ' [last.pt only]';
                if (r.is_champion) label += ' [champion]';
                opt.textContent = label;
                sel.appendChild(opt);
            }
            // A run picked from the runs card may live outside this project's
            // runs/ folder (a lab run, or another project's): keep it
            // selectable. A run inside runs/ that is no longer listed is gone
            // and is not re-added.
            const runsRoot = String(data.runs_dir || '').replace(/\/$/, '') + '/';
            if (prev && !(data.runs_dir && prev.startsWith(runsRoot))) {
                ensureSelectOption(sel, prev, foreignRunLabel(prev));
            }
            if (prev) sel.value = prev;
            // Pre-populate from project state when nothing is selected yet.
            const savedS7 = state && state.steps['7'] && state.steps['7'].config && state.steps['7'].config.run_dir;
            const savedS8 = state && state.steps['8'] && state.steps['8'].config && state.steps['8'].config.run_dir;
            if (selId === 's7-run-dir' && savedS7 && !prev) sel.value = savedS7;
            // Step 8 (Task 9): prefer the promoted champion over a stale saved
            // config value, so promoting a new champion actually changes what
            // Inference defaults to next time the panel loads.
            if (selId === 's8-run-dir') {
                if (championRunDir) sel.value = championRunDir;
                else if (savedS8 && !prev) sel.value = savedS8;
            }
        };
        populate('s7-run-dir');
        populate('s8-run-dir');
        const badge = document.getElementById('s8-champion-badge');
        if (badge) badge.style.display = championRunDir ? 'inline-block' : 'none';
    } catch (e) {
        console.warn('refreshRunList failed:', e);
    }
}

// ── Step 7 Rounds ledger + Promote (Task 9) ─────────────────────────────────

const _ROUNDS_GATE_TOOLTIPS = {
    pass: 'Within tolerance versus the baseline (current champion, or best prior round).',
    fail: 'mAP50-95 in the row\'s own family regressed more than the tolerance (0.005) versus the baseline; do not promote.',
    flag: 'Recall or a per-class AP regressed more than tolerance versus the baseline; review before promoting.',
    unpinned: 'This run\'s split was not a frozen transect holdout, so the comparison is informational only, not a clean gate.',
    incomparable: 'VICARIUS could not compare this round with the baseline: the two were scored in different metric families, on different held-out images, or at different confidence cut-offs. Judge the run on its own numbers.',
};
const _ROUNDS_GATE_CLASSES = {
    pass: 'badge-success', fail: 'badge-error', flag: 'badge-warning',
    unpinned: 'badge-info', incomparable: 'badge-info',
};

// The gate each line of a Gates cell names, short enough to read down a
// column. Its full name is in GATE_LABELS, which the line's tooltip states.
const GATE_CELL_NAMES = {
    gate_map: 'mAP',
    gate_class: 'per class',
    gate_recall: 'recall',
};
// Each verdict word as the words the cell carries. The class codes a
// per-class flag names are counted rather than printed: a run can flag
// seventeen of them, and the names belong in the tooltip.
const GATE_CELL_WORDS = {
    pass: 'passed',
    fail: 'failed',
    flag: 'flagged',
    unpinned: 'split not pinned',
    incomparable: 'nothing to compare',
};
// What a gate the ledger never wrote reads as in a cell.
const GATE_CELL_NO_VERDICT = 'no verdict';
// How a Gates chip lays its words out: wrapping allowed, the lines balanced,
// line height loosened so a two-line chip stays readable.
const GATE_CHIP_STYLE = 'white-space:normal;text-wrap:balance;line-height:1.4;max-width:100%';

/**
 * One gate as a line of the Gates column: the gate named, its verdict in plain
 * words, and how many classes dropped where classes dropped. The ledger's own
 * token ("flag:PA,SS", "unpinned") never reaches the screen; the tooltip
 * states the verdict as a sentence, names the classes in full and says what
 * the verdict means.
 * @param {string} column one of GATE_ORDER
 * @param {string} verdict the cell the server sent, '' when none was written
 * @param {Object<string,string>|null} names code -> full name
 * @returns {string} HTML
 */
function _roundsGateBadge(column, verdict, names) {
    const parts = gateVerdictParts(verdict);
    const flagged = parts.word === 'flag' && parts.codes.length;
    let said = GATE_CELL_NO_VERDICT;
    if (flagged) said = `${parts.codes.length} flagged`;
    else if (verdict) said = GATE_CELL_WORDS[parts.word] || String(verdict);
    const tip = [gateVerdictText(column, verdict, names), _ROUNDS_GATE_TOOLTIPS[parts.word] || '']
        .filter(Boolean).join(' ');
    const cls = _ROUNDS_GATE_CLASSES[parts.word] || 'badge-info';
    // The chip wraps at a space when the column is narrower than its words,
    // balanced so the two lines are about equal, and is never cut mid-word:
    // the table has to hold its Promote column at 946 px without a sideways
    // scroll, and the Gates column is the one that gives.
    return `<span class="badge ${cls}" style="${GATE_CHIP_STYLE}" title="${escAttr(tip)}">`
        + `${escHtmlNav(GATE_CELL_NAMES[column] || column)}: ${escHtmlNav(said)}</span>`;
}

/**
 * A row's three gates, one line each so the column stays as narrow as its
 * longest verdict.
 * @param {object} row a rounds row
 * @param {Object<string,string>|null} names code -> full name
 * @returns {string} HTML
 */
function roundsGatesHtml(row, names) {
    const lines = GATE_ORDER
        .map(column => _roundsGateBadge(column, String((row && row[column]) || '').trim(), names))
        .join('');
    return `<div style="display:flex;flex-direction:column;gap:3px;align-items:flex-start;min-width:${ROUNDS_GATES_MIN_WIDTH}">`
        + `${lines}</div>`;
}

// Placeholder for a cell with no value. Plain text, never a dash entity: the
// house copy rule bans em and en dashes in user-facing strings, entities
// included.
const ROUNDS_NO_VALUE = 'n/a';

// The ledger columns each metric family reads (rounds.csv, evaluate_run.py).
const ROUNDS_FAMILY_COLUMNS = {
    mask: { map50_95: 'map50_95_M', recall: 'recall_M' },
    box: { map50_95: 'map50_95_B', recall: 'recall_B' },
};
const ROUNDS_FAMILY_TIPS = { mask: 'Scored on outlines (masks).', box: 'Scored on boxes.' };

/**
 * A ledger row's metric family: the recorded one, else box when only box
 * scores exist (a detector evaluated before the column was recorded).
 * @param {object} r a rounds row or the champion record
 * @returns {string} 'mask' or 'box'
 */
function roundFamily(r) {
    if (r.metric_family === 'box' || r.metric_family === 'mask') return r.metric_family;
    return (r.map50_95_M == null && r.map50_95_B != null) ? 'box' : 'mask';
}

/**
 * One score from a row, read from its own family's column.
 * @param {object} r a rounds row or the champion record
 * @param {string} family 'mask' or 'box'
 * @param {string} key 'map50_95' or 'recall'
 * @returns {number|null|undefined}
 */
function roundScore(r, family, key) {
    return r[ROUNDS_FAMILY_COLUMNS[family][key]];
}

// The widest the Base column grows: a share of the window rather than a
// fixed width, so the column yields at 946 px and the Promote column stays in
// view. A base that is a stock weight file ("yolo11m.pt") shows in full; one
// that is a path shows its last three parts, cut with an ellipsis, and carries
// the whole path on its tooltip.
// The narrowest the mAP50-95 column's content goes (the cell adds 16 px of
// padding): the header "mAP50-95" measures 70.6 px in Inter 700 13 px, and
// under this floor it broke at the hyphen into "mAP50-" and "95". Carried by
// a block inside the cell, because Chrome's automatic table layout honours a
// floor on the content and not reliably one on the cell.
const ROUNDS_MAP_MIN_WIDTH = '71px';
// The narrowest a run name's block goes: the widest part VEGA mints into a
// run name, the date "20260902_", measures 71.7 px in Inter 13 px. Above this
// floor a name wraps only at its underscores; a part wider than the floor is
// the one case that breaks inside a part.
const ROUNDS_RUN_MIN_WIDTH = '72px';
// The Promote button's side padding, 4 px tighter than the small button's
// own: the 8 px it saves is what lets a 1280 px window hold every chip and
// every run name on one line.
const ROUNDS_PROMOTE_PADDING = '10px';
// The narrowest the Gates column's content goes (the cell adds 16 px of
// padding): room for a chip's widest word group ("per class:", 75.8 px in
// Space Mono 12 px, plus the chip's own 18 px) so a chip wraps between its
// words and never inside one. At 946 px this leaves the Run column its floor
// and the Promote column in view without a sideways scroll; the three "split
// not pinned" chips of an unpinned round take three lines there, every other
// chip two.
const ROUNDS_GATES_MIN_WIDTH = '96px';

/**
 * A run name for the Run column: escaped, with a break opportunity after
 * every underscore so a long name wraps between its parts at the narrow
 * window instead of widening the table. The block that holds it also allows
 * a break inside a part, which only happens when one part alone is wider
 * than ROUNDS_RUN_MIN_WIDTH.
 * @param {string|null|undefined} runName the ledger's run_name
 * @returns {string} HTML
 */
function roundsRunNameHtml(runName) {
    return escHtmlNav(runName || '').replace(/_/g, '_<wbr>');
}

/**
 * A score cell: the value to three decimals (or n/a) with the family chip
 * under it. The chip takes its size from the stylesheet, so it reads at the
 * table's chip size like every other chip. It sits under the number at every
 * window width rather than beside it, so the two score columns stay the same
 * narrow width everywhere and the Gates and Run columns get the room.
 * @param {number|null|undefined} value
 * @param {string} family 'mask' or 'box'
 * @returns {string} HTML
 */
function roundScoreCell(value, family) {
    const text = value != null ? Number(value).toFixed(3) : ROUNDS_NO_VALUE;
    return `${text}<br><span class="badge badge-info" style="margin-top:2px" title="${escAttr(ROUNDS_FAMILY_TIPS[family])}">${family}</span>`;
}

function _roundsDelta(cur, prevMap) {
    if (cur == null || prevMap == null) return ROUNDS_NO_VALUE;
    const d = cur - prevMap;
    const sign = d >= 0 ? '+' : '';
    return `${sign}${d.toFixed(3)}`;
}

// Two confidences count as the same measurement within this much, which
// absorbs a CSV round trip and nothing else. Mirrors CONF_MATCH_TOLERANCE in
// TCRMPtrain_oceankindCV/src/evaluate_run.py.
const ROUNDS_CONF_TOLERANCE = 1e-9;

/**
 * Whether two rows may be ranked against each other. mAP is the area under
 * the whole precision-recall curve, so the same weights score lower when the
 * confidence cut-off is raised: a delta across two cut-offs would credit the
 * measurement change to the model. Two rows that BOTH record nothing are
 * compared as they were before the column existed.
 * @param {number|null} prevConf the earlier row's recorded confidence
 * @param {number|null} conf this row's recorded confidence
 * @returns {boolean}
 */
function roundsComparable(prevConf, conf) {
    if (prevConf == null && conf == null) return true;
    if (prevConf == null || conf == null) return false;
    return Math.abs(Number(prevConf) - Number(conf)) <= ROUNDS_CONF_TOLERANCE;
}

/**
 * The sentence behind a score cell: what that row was measured with.
 * @param {object} r a rounds row
 * @returns {string}
 */
function roundsMeasuredAt(r) {
    if (r.conf == null) {
        return 'This row records nothing about what it was measured with: it was written before '
            + 'VEGA kept the confidence, overlap and image size of a score.';
    }
    const iou = r.iou != null ? r.iou : 'not recorded';
    const imgsz = r.imgsz != null ? r.imgsz : 'not recorded';
    return `Measured at confidence ${r.conf}, overlap ${iou}, image size ${imgsz}.`;
}

async function refreshStep7Rounds() {
    const tbody = document.getElementById('s7-rounds-tbody');
    const note = document.getElementById('s7-rounds-champion-note');
    if (!tbody) return;
    try {
        const data = await api('/api/step/7/rounds');
        const rows = data.rounds || [];
        const champion = data.champion || null;
        // The gates name the classes that dropped, and a code never stands
        // alone where the registry resolves it. Null when the Label Manager
        // is unavailable, which leaves every code as it is.
        const labelNames = await vegaLabelNames();
        if (note) {
            const champFamily = champion ? roundFamily(champion) : 'mask';
            const champScore = champion ? roundScore(champion, champFamily, 'map50_95') : null;
            note.textContent = champion
                ? `Champion: ${champion.run_dir ? champion.run_dir.split('/').pop() : '(unknown)'} `
                  + `(mAP50-95 ${champScore != null ? Number(champScore).toFixed(3) : '?'}, ${champFamily})`
                : 'No champion promoted yet.';
        }
        if (!rows.length) {
            tbody.innerHTML = '<tr><td colspan="8" style="padding:10px 8px;color:var(--text-dim)">No evaluated rounds yet. Run Evaluation above to add one.</td></tr>';
            return;
        }
        // The delta compares a row with the previous row of the same family;
        // a mask score against a box score would be a number with no meaning.
        const prevByFamily = {};
        tbody.innerHTML = rows.map(r => {
            const family = roundFamily(r);
            const map = roundScore(r, family, 'map50_95');
            const recall = roundScore(r, family, 'recall');
            const prev = prevByFamily[family] || null;
            const comparable = prev ? roundsComparable(prev.conf, r.conf) : false;
            const delta = (prev && comparable) ? _roundsDelta(map, prev.map) : ROUNDS_NO_VALUE;
            const deltaTip = (prev && !comparable)
                ? `No delta: this row was measured at confidence ${r.conf != null ? r.conf : 'an unrecorded value'} `
                  + `and the previous ${family} row at ${prev.conf != null ? prev.conf : 'an unrecorded value'}.`
                : 'Change in mAP50-95 against the previous row of the same family.';
            if (map != null) prevByFamily[family] = { map: map, conf: r.conf };
            const isChamp = champion && champion.run_dir === r.run_dir;
            const gates = roundsGatesHtml(r, labelNames);
            const promoteBtn = isChamp
                ? '<span class="badge badge-info" title="This run is the current champion.">champion</span>'
                : `<button class="btn btn-outline btn-sm" style="padding-left:${ROUNDS_PROMOTE_PADDING};padding-right:${ROUNDS_PROMOTE_PADDING}" onclick='confirmPromoteRun(${JSON.stringify(r.run_dir)})' title="Make this run the champion. It becomes the default weights for Inference and the first pick to continue training from.">Promote</button>`;
            // What a run started from may be a full path to another run's
            // best.pt: show its file name and keep the whole path on hover so
            // the column cannot widen the table past the card.
            const base = String(r.base_model || '');
            // The one token that identifies the base: the run folder when the base is
            // another run's weights (<run>/weights/best.pt), else the weights file.
            const baseParts = base.split('/').filter(Boolean);
            const baseShort = baseParts.length >= 3 && baseParts[baseParts.length - 2] === 'weights'
                ? baseParts[baseParts.length - 3] : (baseParts[baseParts.length - 1] || base);
            return `<tr>
                <td style="padding:6px 8px;border-bottom:1px solid var(--border)">${escHtmlNav(r.round != null ? r.round : '')}</td>
                <td style="padding:6px 8px;border-bottom:1px solid var(--border)" title="${escAttr(r.run_name || '')}"><span style="display:inline-block;min-width:${ROUNDS_RUN_MIN_WIDTH};overflow-wrap:anywhere">${roundsRunNameHtml(r.run_name)}</span></td>
                <td style="padding:6px 8px;border-bottom:1px solid var(--border)" title="${escAttr(base)}"><span style="display:inline-block;overflow-wrap:anywhere">${escHtmlNav(baseShort)}</span></td>
                <td style="padding:6px 8px;border-bottom:1px solid var(--border);text-align:right" title="${escAttr(roundsMeasuredAt(r))}"><span style="display:inline-block;min-width:${ROUNDS_MAP_MIN_WIDTH}">${roundScoreCell(map, family)}</span></td>
                <td style="padding:6px 8px;border-bottom:1px solid var(--border);text-align:right" title="${escAttr(deltaTip)}">${delta}</td>
                <td style="padding:6px 8px;border-bottom:1px solid var(--border);text-align:right">${roundScoreCell(recall, family)}</td>
                <td style="padding:6px 8px;border-bottom:1px solid var(--border)">${gates}</td>
                <td style="padding:6px 8px;border-bottom:1px solid var(--border)">${promoteBtn}</td>
            </tr>`;
        }).join('');
    } catch (e) {
        console.warn('refreshStep7Rounds failed:', e);
        tbody.innerHTML = `<tr><td colspan="8" style="padding:10px 8px;color:var(--error)">Error loading rounds: ${escHtmlNav(String(e))}</td></tr>`;
    }
}

// The three gates a ledger row records, in the order the server states them,
// each with the plain words for it. Mirrors GATE_COLUMNS and GATE_LABELS in
// app.py, which the refusal reads from.
const GATE_ORDER = ['gate_map', 'gate_class', 'gate_recall'];
const GATE_LABELS = {
    gate_map: 'the mean average precision gate',
    gate_class: 'the per-class gate',
    gate_recall: 'the recall gate',
};
// A verdict cell is a word, and the per-class gate's word may be followed by
// ':' and the label codes that dropped ("flag:PA,SS"), which is the shape
// app.py's GATE_COLUMNS block documents and evaluate_run.py writes. Group 1 is
// the word, group 2 the codes.
const GATE_VERDICT_RE = /^([A-Za-z_]+)(?::(.*))?$/;
// Each verdict word as the words a reader sees. A word this list does not hold
// is stated as the server wrote it rather than guessed at.
const GATE_VERDICT_WORDS = {
    pass: 'passed',
    fail: 'failed',
    flag: 'flagged a class',
    unpinned: 'could not run, because the split was not pinned',
    incomparable: 'found nothing to compare this run against',
};
// What a gate that was never written reads as. A ledger that recorded nothing
// has stated no failure, so it is not a verdict.
const GATE_NO_VERDICT = 'recorded no verdict';

/**
 * One gate's verdict as the sentence fragment a reader sees, with every label
 * code resolved to its full name and the code beside it as a chip.
 * @param {string} column one of GATE_ORDER
 * @param {string} verdict the cell the server sent, '' when none was written
 * @param {Object<string,string>|null} names code -> full name, null when the
 *   label registry is unavailable, which leaves a code as it is
 * @returns {string} HTML
 */
function gateVerdictHtml(column, verdict, names) {
    const label = GATE_LABELS[column];
    if (!verdict) return `${label} ${GATE_NO_VERDICT}`;
    const parts = gateVerdictParts(verdict);
    if (parts.word === 'flag' && parts.codes.length) {
        const named = joinPhrases(parts.codes.map(code => step6ClassLabelHtml(code, names)));
        return `${label} flagged ${named}`;
    }
    const said = GATE_VERDICT_WORDS[parts.word];
    return said ? `${label} ${said}` : `${label} says ${escHtmlNav(verdict)}`;
}

/**
 * A gate cell as the server wrote it, split into its verdict word and the
 * class codes it names. A code the ledger repeats (a class that dropped on two
 * metrics) is kept once, because it is one class either way.
 * @param {string} verdict the cell the server sent
 * @returns {{word: string, codes: Array<string>}} word '' when nothing was
 *   written or the cell is not a verdict
 */
function gateVerdictParts(verdict) {
    const parts = GATE_VERDICT_RE.exec(String(verdict == null ? '' : verdict).trim());
    if (!parts) return { word: '', codes: [] };
    const codes = [];
    (parts[2] ? parts[2].split(',') : []).forEach(raw => {
        const code = raw.trim();
        if (code && codes.indexOf(code) < 0) codes.push(code);
    });
    return { word: parts[1].toLowerCase(), codes: codes };
}

// How many classes a per-class tooltip names in full before it gives the rest
// as a count. Four binomials fill the bubble's 36 characters about four lines
// deep, which is as much as a tooltip carries; a detector can flag seventeen.
const GATE_NAMED_CLASS_MAX = 4;

/**
 * The classes a per-class gate flagged, as plain text for a tooltip: each name
 * in full with its code beside it, the rest as a count.
 * @param {Array<string>} codes the class codes the gate named
 * @param {Object<string,string>|null} names code -> full name
 * @returns {string} plain text
 */
function gateFlaggedClassesText(codes, names) {
    const named = codes.slice(0, GATE_NAMED_CLASS_MAX)
        .map(code => step6ClassLabelText(code, names));
    const rest = codes.length - named.length;
    const list = joinPhrases(named);
    return rest > 0 ? `${list} and ${countedNoun(rest, 'other')}` : list;
}

/**
 * One gate's verdict as the plain sentence its tooltip carries. The same facts
 * gateVerdictHtml paints into the promote dialog, with no markup, so a cell
 * and a dialog cannot tell a reader two different things.
 * @param {string} column one of GATE_ORDER
 * @param {string} verdict the cell the server sent, '' when none was written
 * @param {Object<string,string>|null} names code -> full name
 * @returns {string} plain text, one sentence
 */
function gateVerdictText(column, verdict, names) {
    const label = GATE_LABELS[column];
    const opening = `${label.charAt(0).toUpperCase()}${label.slice(1)}`;
    if (!verdict) return `${opening} ${GATE_NO_VERDICT}.`;
    const parts = gateVerdictParts(verdict);
    if (parts.word === 'flag' && parts.codes.length) {
        return `${opening} flagged ${gateFlaggedClassesText(parts.codes, names)}.`;
    }
    const said = GATE_VERDICT_WORDS[parts.word];
    return said ? `${opening} ${said}.` : `${opening} says ${verdict}.`;
}

/**
 * Every gate verdict as one line, so a refusal states all three rather than
 * the one that stopped the promotion.
 * @param {Object|null} gate {gate column: verdict} as the server sends it
 * @param {Object<string,string>|null} names code -> full name
 * @returns {Array<string>} one HTML fragment per gate, in GATE_ORDER
 */
function gateVerdictLines(gate, names) {
    return GATE_ORDER.map(column =>
        gateVerdictHtml(column, String((gate && gate[column]) || '').trim(), names));
}

// The run the gate dialog is open for, kept on the message element so the
// label names, which arrive after the dialog opens, are only ever painted back
// into the dialog they were fetched for.
const GATE_DIALOG_RUN_KEY = 'gateRun';

/**
 * Paint the gate dialog's body: one sentence for what promoting does, then one
 * line per gate verdict.
 * @param {string} runDir the run folder being promoted
 * @param {Object|null} gate the verdicts the refusal carried
 * @param {Object<string,string>|null} names code -> full name
 */
function paintPromoteGateDialog(runDir, gate, names) {
    const msgEl = document.getElementById('confirm-msg');
    if (!msgEl) return;
    const runName = String(runDir || '').split('/').filter(Boolean).pop() || 'this run';
    // Each verdict is a sentence of its own: the gate labels are written to sit
    // inside one of the server's clauses, so the line that carries one alone
    // opens in upper case and closes with a full stop.
    const lines = gateVerdictLines(gate, names)
        .map(line => `<li style="margin:2px 0">${line.charAt(0).toUpperCase()}${line.slice(1)}.</li>`)
        .join('');
    msgEl.dataset[GATE_DIALOG_RUN_KEY] = String(runDir);
    msgEl.innerHTML =
        `<div>Promoting ${escHtmlNav(runName)} makes its weights the default for `
        + `Inference and the first pick to continue training from.</div>`
        + `<ul style="margin:8px 0 0;padding:0;list-style:none">${lines}</ul>`;
}

/**
 * Ask whether to promote a run the server refused on its own gates, stating
 * each verdict. Confirming re-sends the promotion with the operator's
 * confirmation on it.
 * @param {string} runDir the run folder being promoted
 * @param {Object|null} gate the verdicts the refusal carried
 */
function confirmPromoteOverGate(runDir, gate) {
    const titleEl = document.getElementById('confirm-title');
    if (titleEl) titleEl.textContent = 'Promote over the gate?';
    paintPromoteGateDialog(runDir, gate, null);
    const okBtn = document.getElementById('confirm-ok-btn');
    if (okBtn) {
        okBtn.textContent = 'Promote Anyway';
        okBtn.onclick = () => { closeConfirmModal(); promoteRun(runDir, true); };
    }
    const ov = document.getElementById('confirm-overlay');
    if (ov) ov.classList.add('visible');
    // The label registry answers after the dialog is already on screen, so the
    // codes are painted first and the full names replace them when they
    // arrive. A dialog that has moved on to another run is left alone.
    vegaLabelNames().then(names => {
        if (!names) return;
        const msgEl = document.getElementById('confirm-msg');
        if (!msgEl || msgEl.dataset[GATE_DIALOG_RUN_KEY] !== String(runDir)) return;
        paintPromoteGateDialog(runDir, gate, names);
    });
}

// Reuses the standard reset-confirm overlay's modal chrome (styled confirm,
// not window.confirm()) for the Promote action's "are you sure" step.
function confirmPromoteRun(runDir) {
    const titleEl = document.getElementById('confirm-title');
    if (titleEl) titleEl.textContent = 'Promote to champion?';
    const msgEl = document.getElementById('confirm-msg');
    if (msgEl) {
        delete msgEl.dataset[GATE_DIALOG_RUN_KEY];
        msgEl.textContent = 'Make this run the champion? It becomes the default weights for Inference and the first pick to continue training from.';
    }
    const okBtn = document.getElementById('confirm-ok-btn');
    if (okBtn) { okBtn.textContent = 'Confirm Promote'; okBtn.onclick = () => { closeConfirmModal(); promoteRun(runDir); }; }
    const ov = document.getElementById('confirm-overlay');
    if (ov) ov.classList.add('visible');
}

/**
 * Promote a run to champion. The server refuses a run whose own gates recorded
 * a failure, a flag or an incomparable comparison, and answers with every
 * verdict; that refusal opens the confirmation dialog rather than an error,
 * and confirming sends the same promotion back with confirm_gate on it.
 * @param {string} runDir the run folder to promote
 * @param {boolean} [confirmGate] true when the operator has already confirmed
 */
async function promoteRun(runDir, confirmGate) {
    try {
        const body = { run_dir: runDir };
        if (confirmGate) body.confirm_gate = true;
        const data = await post('/api/step/7/promote', body);
        if (data && data.needs_confirmation && !confirmGate) {
            confirmPromoteOverGate(runDir, data.gate);
            return;
        }
        if (data.error) { alert('Promote failed: ' + data.error); return; }
        await refreshStep7Rounds();
        await refreshRunList();
        await refreshStep6ContinueOptions();
        openStep6RunsCard();
    } catch (e) {
        alert('Promote failed: ' + e);
    }
}

// ── VEGA cards: runs (Train), report (Evaluate), gallery (Inference) ──────
// Each card is a var-free fragment plus its own script (orchestrator_runs.js,
// orchestrator_report.js, orchestrator_gallery.js). The host owns the wiring:
// when each card loads, and what its row actions and verdicts do.
const RUNS_CARD_ID = 's6-runs-card';
const GALLERY_CARD_ID = 's8-gallery-card';
const GALLERY_RUN_PICKER_ID = 's8-gallery-run';
const VOCABULARY_URL = '/labels/api/vocabulary';
const WEIGHTS_REL = '/weights/best.pt';
// The runs card reports a run's task as "seg" or "detect"; the task select says "segment".
const RUN_TASK_TO_PANEL_TASK = { seg: 'segment', segment: 'segment', detect: 'detect' };
const RUN_WARNING_TOOLTIP = 'VICARIUS started the run and reports this so the result is read with it in mind.';
// What each task is called in a sentence.
const STEP6_TASK_NOUNS = { segment: 'segmenter', detect: 'detector' };
// A run's weights carry the head of the task they trained, so continuing a
// detector from segmentation weights (or the reverse) trains the wrong model.
// The launch preflight reads only the weights file name, and every run's file
// is called best.pt, so a cross-family hand-off has to be refused here.
const FINETUNE_CROSS_FAMILY_TEXT =
    'This run trained a {run} and this project trains a {panel}, so its weights cannot start this run. '
    + 'Pick a {panel} run to continue from.';
const FINETUNE_CROSS_FAMILY_TIP =
    'Continuing from this run is unavailable here because it trained a {run} while this project trains a {panel}. '
    + 'VEGA refuses the pair because a checkpoint carries the head of the task it trained.';
// The runs card puts the reason for an action it disables on a focusable
// wrapper, because a disabled button is the element the pointer hits and a
// title left on it is never read. The pass below follows that contract.
const RUNS_ACTION_WRAPPER_CLASS = 'vega-runs-action';
const FINETUNE_ACTION_SELECTOR = '[data-action="finetune"][data-task]';

/**
 * The task the Train panel will launch: the control's value, or the project's
 * own default when the control is not on the page.
 * @returns {string} 'detect' or 'segment'
 */
function step6PanelTask() {
    const taskEl = document.getElementById('s6-task');
    if (!taskEl) return projectTrainingTask();
    return taskEl.value === 'detect' ? 'detect' : 'segment';
}

/**
 * True when the Train panel's task cannot follow a run's task, which is what
 * a pinned control means (a box dataset trains a detector and nothing else).
 * @returns {boolean}
 */
function step6TaskIsPinned() {
    const taskEl = document.getElementById('s6-task');
    return !!(taskEl && taskEl.disabled);
}

/**
 * Why a run cannot be continued from this panel, or '' when it can. A run
 * whose task the card did not report is left alone: an unknown task is not
 * evidence of a mismatch.
 * @param {string} runTask the run's task as the runs card reports it ('seg' or 'detect')
 * @param {string} panelTask the panel's task ('segment' or 'detect')
 * @param {boolean} pinned whether the panel's task is pinned
 * @param {string} template the sentence to fill, with {run} and {panel}
 * @returns {string} the filled sentence, or ''
 */
function finetuneRefusal(runTask, panelTask, pinned, template) {
    const mapped = RUN_TASK_TO_PANEL_TASK[String(runTask || '').trim().toLowerCase()] || '';
    if (!pinned || !mapped || mapped === panelTask) return '';
    return String(template)
        .replace(/\{run\}/g, STEP6_TASK_NOUNS[mapped] || mapped)
        .replace(/\{panel\}/g, STEP6_TASK_NOUNS[panelTask] || panelTask);
}

/**
 * Disable one runs-card Continue-from button and put the reason where it can
 * be read.
 * @param {Element} btn the button
 * @param {string} reason the sentence saying why
 */
function disableFinetuneButton(btn, reason) {
    btn.dataset.familyDisabled = '1';
    if (!('freeTip' in btn.dataset)) btn.dataset.freeTip = btn.title || '';
    btn.disabled = true;
    // Both spellings: the property is what a browser reads back, the attribute
    // is what it renders, and the reason belongs on the wrapper instead.
    btn.title = '';
    btn.removeAttribute('title');
    const parent = btn.parentNode;
    if (parent && parent.classList && parent.classList.contains(RUNS_ACTION_WRAPPER_CLASS)) {
        parent.title = reason;
        return;
    }
    if (!parent || typeof document.createElement !== 'function') return;
    const wrap = document.createElement('span');
    wrap.className = RUNS_ACTION_WRAPPER_CLASS;
    wrap.tabIndex = 0;
    wrap.title = reason;
    parent.insertBefore(wrap, btn);
    wrap.appendChild(btn);
}

/**
 * Hand one runs-card Continue-from button back after its reason is gone (the
 * training source moved back to a segmentation export, so the task is free
 * again).
 * @param {Element} btn the button
 */
function restoreFinetuneButton(btn) {
    delete btn.dataset.familyDisabled;
    btn.disabled = false;
    if ('freeTip' in btn.dataset) {
        if (btn.dataset.freeTip) btn.title = btn.dataset.freeTip;
        delete btn.dataset.freeTip;
    }
    const parent = btn.parentNode;
    if (parent && parent.classList && parent.classList.contains(RUNS_ACTION_WRAPPER_CLASS)
            && parent.parentNode) {
        parent.parentNode.insertBefore(btn, parent);
        if (typeof parent.remove === 'function') parent.remove();
        else parent.parentNode.removeChild(parent);
    }
}

/**
 * Walk the Trained runs card and disable Continue from on every run this
 * panel cannot continue from, with the reason on its wrapper. Idempotent: a
 * button the card itself disabled (no weights) is left alone, and a button
 * this pass disabled is handed back when its reason is gone.
 */
function applyStep6FinetuneFamily() {
    const card = document.getElementById(RUNS_CARD_ID);
    if (!card || typeof card.querySelectorAll !== 'function') return;
    const panelTask = step6PanelTask();
    const pinned = step6TaskIsPinned();
    Array.from(card.querySelectorAll(FINETUNE_ACTION_SELECTOR)).forEach(btn => {
        const reason = finetuneRefusal(btn.getAttribute('data-task'), panelTask, pinned,
                                       FINETUNE_CROSS_FAMILY_TIP);
        const ours = btn.dataset.familyDisabled === '1';
        if (reason && !btn.disabled) { disableFinetuneButton(btn, reason); return; }
        if (!reason && ours) restoreFinetuneButton(btn);
    });
}

/**
 * Re-apply the family pass whenever the runs card re-renders itself, which its
 * own Refresh button does; every Continue-from button comes back enabled.
 */
function watchStep6RunsCard() {
    const card = document.getElementById(RUNS_CARD_ID);
    if (!card || card.dataset.familyWatched === '1') return;
    if (typeof MutationObserver !== 'function') return;
    card.dataset.familyWatched = '1';
    new MutationObserver(() => applyStep6FinetuneFamily())
        .observe(card, { childList: true, subtree: true });
}

/**
 * Render the warnings a run route returned into the panel's warnings
 * container (s<step>-warnings); an empty list hides it.
 * @param {number|string} step the step key
 * @param {Array<string>|undefined} warnings the route's warnings array
 */
function renderStepWarnings(step, warnings) {
    const host = document.getElementById(`s${step}-warnings`);
    if (!host) return;
    const list = Array.isArray(warnings) ? warnings.filter(w => typeof w === 'string' && w.trim()) : [];
    if (!list.length) { host.innerHTML = ''; host.style.display = 'none'; return; }
    host.innerHTML = list.map(w =>
        `<div class="badge badge-warning" style="display:block;white-space:normal;margin-top:8px" title="${escapeAttr(RUN_WARNING_TOOLTIP)}">${escapeHtml(w)}</div>`
    ).join('');
    host.style.display = 'block';
}

/**
 * Reflect a live run on a panel's buttons: Start is disabled and Stop shown
 * while the run is live, the reverse when it ends. Only a panel that carries
 * a Stop button (s<step>-stop-btn) is affected.
 * @param {number|string} step the step key
 * @param {boolean} running whether the step's run is live
 */
function setStepRunControls(step, running) {
    const stopBtn = document.getElementById(`s${step}-stop-btn`);
    if (!stopBtn) return;
    stopBtn.style.display = running ? '' : 'none';
    // A live run gets the button's own title back after keepStopReachable
    // lent it the stale-mark one.
    if (running && stopBtn.dataset.liveTitle !== undefined) {
        stopBtn.title = stopBtn.dataset.liveTitle;
        delete stopBtn.dataset.liveTitle;
    }
    const runBtn = document.getElementById(`s${step}-run-btn`);
    if (runBtn) runBtn.disabled = !!running;
}

/**
 * Make sure a select can hold a value: append a one-off option when none of
 * its options carries it.
 * @param {HTMLSelectElement} sel
 * @param {string} value
 * @param {string} label the text of the appended option
 */
function ensureSelectOption(sel, value, label) {
    if (!sel || !value) return;
    if (Array.from(sel.options).some(o => o.value === value)) return;
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = label;
    sel.appendChild(opt);
}

// Label for a run folder outside this project's runs/: its own name, marked.
function foreignRunLabel(runDir) {
    return `${String(runDir).replace(/\/$/, '').split('/').pop()} (outside this project)`;
}

/**
 * Initialise or reload the cross-project runs card in the Train panel.
 * @returns {Promise<void>|undefined}
 */
function openStep6RunsCard() {
    if (typeof window.initStep6Runs !== 'function') {
        console.error('orchestrator_runs.js did not load; the runs card cannot open');
        return undefined;
    }
    watchStep6RunsCard();
    const loading = window.initStep6Runs();
    return (loading && typeof loading.then === 'function')
        ? loading.then(() => { applyStep6FinetuneFamily(); })
        : loading;
}

/**
 * Attach the host's listeners to the cards once: the runs card's row actions
 * and the gallery's verdicts.
 */
function wireVegaCards() {
    const runsCard = document.getElementById(RUNS_CARD_ID);
    if (runsCard && !runsCard.dataset.hostWired) {
        runsCard.dataset.hostWired = '1';
        runsCard.addEventListener('vega:evaluate-run', e => onVegaEvaluateRun(e.detail || {}));
        runsCard.addEventListener('vega:promote-run', e => onVegaPromoteRun(e.detail || {}));
        runsCard.addEventListener('vega:finetune-from', e => onVegaFinetuneFrom(e.detail || {}));
    }
    const gallery = document.getElementById(GALLERY_CARD_ID);
    if (gallery && !gallery.dataset.hostWired) {
        gallery.dataset.hostWired = '1';
        gallery.addEventListener('vega:verdict', e => onVegaVerdict(e.detail || {}));
    }
}

/**
 * Evaluate a run from the runs card: select it in the Evaluate panel, then
 * open that panel, whose loader keeps the selection, prefills the image size
 * and shows the run's report.
 * @param {{run_dir: string}} detail the card's event detail
 */
function onVegaEvaluateRun(detail) {
    const runDir = String(detail.run_dir || '');
    if (!runDir) return;
    const sel = document.getElementById('s7-run-dir');
    if (sel) {
        ensureSelectOption(sel, runDir, foreignRunLabel(runDir));
        sel.value = runDir;
    }
    switchStep(7);
}

/**
 * Promote a run from the runs card through the same confirm dialog the
 * Rounds table uses; promoteRun reloads the card on success.
 * @param {{run_dir: string}} detail the card's event detail
 */
function onVegaPromoteRun(detail) {
    const runDir = String(detail.run_dir || '');
    if (runDir) confirmPromoteRun(runDir);
}

/**
 * Continue training from a run picked on the Trained runs card: align the
 * Training task with the run's task (unless the task is pinned), then choose
 * that run's best.pt under Continue from a VEGA model. The card lives on the
 * Evaluate panel now, so this also switches to the Train panel, which is
 * where the pick and its refusal message are both shown.
 * @param {{run_dir: string, task: string}} detail the card's event detail
 */
function onVegaFinetuneFrom(detail) {
    const runDir = String(detail.run_dir || '').replace(/\/$/, '');
    if (!runDir) return;
    const taskEl = document.getElementById('s6-task');
    // Refuse the mix outright when the panel's task cannot follow the run's.
    // Without this the hand-off stamped segmentation weights as a detector run
    // and the launch preflight, which judges by file name, let it through.
    const refusal = finetuneRefusal(detail.task, step6PanelTask(), step6TaskIsPinned(),
                                    FINETUNE_CROSS_FAMILY_TEXT);
    if (refusal) {
        showResult(6, false, escapeHtml(refusal));
        const el = document.getElementById('s6-result');
        if (el && typeof el.scrollIntoView === 'function') el.scrollIntoView({ block: 'center' });
        return;
    }
    const task = RUN_TASK_TO_PANEL_TASK[String(detail.task || '').toLowerCase()];
    if (taskEl && task && !taskEl.disabled && taskEl.value !== task) {
        taskEl.value = task;
        onStep6TaskChange();
    }
    switchStep(6);
    applyStep6ModelPath(runDir + WEIGHTS_REL);
    const continueSel = document.getElementById('s6-continue-from');
    if (continueSel && typeof continueSel.scrollIntoView === 'function') continueSel.scrollIntoView({ block: 'center' });
}

/**
 * Record a gallery verdict against the run it belongs to.
 * @param {{out_dir: string, filename: string, verdict: string}} detail the card's event detail
 */
async function onVegaVerdict(detail) {
    try {
        const data = await post('/api/step/8/verify', {
            out_dir: detail.out_dir, filename: detail.filename, verdict: detail.verdict,
        });
        if (data.error) alert('Verdict not recorded: ' + data.error);
    } catch (e) {
        alert('Verdict not recorded: ' + e.message);
    }
}

// The label vocabulary as {code: full name}, fetched once per page; null when
// the Label Manager route is unavailable, in which case codes stay as they are.
let _vegaLabelNamesPromise = null;
function vegaLabelNames() {
    if (!_vegaLabelNamesPromise) {
        _vegaLabelNamesPromise = api(VOCABULARY_URL).then(data => {
            if (!data || data.error || !Array.isArray(data.rows)) return null;
            const names = {};
            for (const row of data.rows) {
                if (row && row.code && row.name) names[String(row.code)] = String(row.name);
            }
            return names;
        }).catch(() => null);
    }
    return _vegaLabelNamesPromise;
}

/**
 * Load the report card for the run selected in the Evaluate panel ('' asks
 * the route for the project's last evaluation).
 * @returns {Promise<void>}
 */
async function loadStep7ReportForSelectedRun() {
    if (typeof window.loadStep7ReportCard !== 'function') {
        console.error('orchestrator_report.js did not load; the report card cannot load');
        return;
    }
    const sel = document.getElementById('s7-run-dir');
    const runDir = sel ? (sel.value || '') : '';
    const labelNames = await vegaLabelNames();
    await window.loadStep7ReportCard(runDir, { fetchJson: api, labelNames: labelNames });
}

/**
 * True when the "All trained models" disclosure on the Evaluate panel is
 * open, so a caller can decide whether the runs card inside it needs a fresh
 * load: the card lives on this panel now (moved from Train, 2026-09-21) and
 * loads only while its disclosure is open, never into a hidden panel.
 * @returns {boolean}
 */
function step7AllModelsOpen() {
    const details = document.getElementById('s7-all-models');
    return !!(details && details.open);
}

/**
 * Load the Trained runs card the first time "All trained models" is opened,
 * and every later time it is opened again, since a run may have finished,
 * been evaluated or been promoted while the disclosure was closed. Wired
 * once per page.
 */
function wireStep7AllModelsToggle() {
    const details = document.getElementById('s7-all-models');
    if (!details || details.dataset.wired === '1') return;
    details.dataset.wired = '1';
    details.addEventListener('toggle', () => {
        if (details.open) openStep6RunsCard();
    });
}

/**
 * Open the Evaluate panel: rebuild the run list, follow the selected run
 * (image size and report), refresh the Rounds table, and load the Trained
 * runs card inside "All trained models" when that disclosure is already open
 * (its own toggle listener loads it the rest of the time).
 */
async function openStep7Panel() {
    await refreshRunList();
    await onStep7RunChange();
    refreshStep7Rounds();
    wireStep7AllModelsToggle();
    if (step7AllModelsOpen()) openStep6RunsCard();
}

/**
 * Copy the selected run's evaluation PDF into a folder the user names.
 */
async function copyStep7Pdf() {
    const dst = prompt('Copy the evaluation PDF to which folder?', '');
    if (!dst) return;
    const sel = document.getElementById('s7-run-dir');
    const data = await post('/api/step/7/copy_pdf', { dst_dir: dst, run_dir: sel ? (sel.value || '') : '' });
    if (data.error) { alert('Copy failed: ' + data.error); return; }
    alert('PDF copied to ' + data.dst);
}

// The output folder of the newest inference run this project recorded.
function newestStep8OutDir() {
    const outputs = (state && state.steps && state.steps['8'] && state.steps['8'].outputs) || {};
    return typeof outputs.out_dir === 'string' ? outputs.out_dir : '';
}

/**
 * Load the gallery card: its run picker and the selected run's tiles. When an
 * output folder is given (the run that just finished), that run is shown.
 * @param {string} [preferOutDir] the inference output folder to select
 * @returns {Promise<void>|undefined}
 */
function refreshStep8GalleryCard(preferOutDir) {
    const card = document.getElementById(GALLERY_CARD_ID);
    if (!card || !window.VegaGallery || typeof window.VegaGallery.load !== 'function') {
        console.error('orchestrator_gallery.js did not load; the gallery card cannot load');
        return undefined;
    }
    return window.VegaGallery.load(card).then(() => {
        const picker = document.getElementById(GALLERY_RUN_PICKER_ID);
        if (preferOutDir && picker && picker.value !== preferOutDir
                && Array.from(picker.options).some(o => o.value === preferOutDir)) {
            picker.value = preferOutDir;
            return window.VegaGallery.select(card, preferOutDir);
        }
        return undefined;
    });
}

function toggleStep8SourceFields() {
    const srcRadio = document.querySelector('input[name="s8-src"]:checked');
    const src = srcRadio ? srcRadio.value : 'unused';
    document.getElementById('s8-dir-wrap').style.display = (src === 'dir') ? '' : 'none';
    document.getElementById('s8-sample-wrap').style.display = (src === 'dir' || src === 'full') ? 'none' : 'flex';
}
document.addEventListener('change', (e) => {
    if (e.target && e.target.name === 's8-src') toggleStep8SourceFields();
    // Step 6 box-dataset mode and class picker follow the training source
    // (H2, 2026-08-26; box-dataset sources 2026-09-05).
    if (e.target && e.target.id === 's6-source') { applyStep6BoxDatasetMode(); refreshStep6Classes(); }
    if (e.target && e.target.id === 's6-source') refreshStep6CardTags();
    // A person's pick on a model card row is kept when the source changes.
    if (e.target && (e.target.id === 's6-card-subject' || e.target.id === 's6-card-data-source')) {
        e.target.dataset.picked = '1';
    }
    // Training data summary (2026-09-21): a ticked class updates the picker's
    // heading count and re-reads the summary; every other control the
    // summary depends on re-reads it on its own 'change'.
    if (e.target && e.target.classList && e.target.classList.contains('s6-class-cb')) {
        updateStep6ClassesCount();
        refreshStep6DataSummary();
    } else if (e.target && STEP6_DATA_SUMMARY_TRIGGER_IDS.includes(e.target.id)) {
        refreshStep6DataSummary();
    }
});

// Fields the page fills on its own until the user types into them: the Step 6
// run name (blank until typed) and the Evaluate image size (prefilled from the
// run). A typed value is kept across panel re-entries; clearing the field
// hands it back to the page.
const USER_EDITED_FIELDS = ['s6-run-name', 's7-imgsz'];
document.addEventListener('input', (e) => {
    if (e.target && USER_EDITED_FIELDS.includes(e.target.id)) {
        if (e.target.value && e.target.value.length > 0) {
            e.target.dataset.userEdited = '1';
        } else {
            delete e.target.dataset.userEdited;
        }
    }
    // The two split fractions re-read the training data summary live while
    // typing, debounced so a run of keystrokes sends one query.
    if (e.target && (e.target.id === 's6-valid' || e.target.id === 's6-test')) scheduleStep6DataSummary();
});

// ── Step 3 label picker ───────────────────────────────────────────────────
// Data for the picker lives here so filter/toggle operations don't need to
// round-trip to the server. _loaded is the raw server list; _excludedSet is
// the set of codes flagged excluded in step 2's remap log.
const step3LabelState = {
    loaded: [],
    excludedSet: new Set(),
    loadedOnce: false,
};

async function loadStep3Labels(force) {
    const host = document.getElementById('s3-label-picker');
    const status = document.getElementById('s3-label-status');
    if (!host) return;
    if (step3LabelState.loadedOnce && !force) {
        // Still render (to re-apply current config's target_species checks).
        renderStep3Labels();
        return;
    }
    host.innerHTML = '<span style="color:var(--text-dim)">Loading labels from Step 2...</span>';
    try {
        const data = await api('/api/step/3/target_labels');
        if (data.error) {
            host.innerHTML = `<span style="color:var(--magenta-light)">Error: ${data.error}</span>`;
            return;
        }
        step3LabelState.loaded = data.labels || [];
        step3LabelState.excludedSet = new Set((data.excludes || []).map(String));
        step3LabelState.loadedOnce = true;
        if (status) {
            const src = data.master_codes_path ? ` (source: ${data.master_codes_path})` : '';
            status.textContent = `${step3LabelState.loaded.length} labels loaded${src}.`;
        }
        renderStep3Labels();
    } catch (e) {
        host.innerHTML = `<span style="color:var(--magenta-light)">Error loading labels: ${e}</span>`;
    }
}

// Category display order, top to bottom. Matched case-insensitively against the
// trimmed label.category string. Anything not listed sorts alphabetically AFTER
// these; empty/missing category becomes 'Uncategorized', placed dead last.
const CATEGORY_ORDER = [
    'Coral', 'Macroalgae', 'Sponge', 'Gorgonian', 'Turf', 'Cyanobacteria',
    'Calcareous', 'Other, Living', 'Coral Condition', 'Non-living'
];
const UNCATEGORIZED = 'Uncategorized';

// True if a label should never be shown (excluded by step 2 or by server flag).
// Excluded labels are ALWAYS hidden; there is no hide-excluded toggle anymore.
function step3IsExcluded(L) {
    return !!L.excluded || step3LabelState.excludedSet.has(String(L.code));
}

// Resolve a code -> full display name from the loaded list (for chip labels).
function step3NameForCode(code) {
    const c = String(code);
    for (const L of step3LabelState.loaded) {
        if (String(L.code) === c) return L.name || '';
    }
    return '';
}

// Group the (non-excluded) loaded labels by category, returning an ordered
// array of { category, labels } where labels are sorted points DESC, frames
// DESC, code ASC, and categories follow CATEGORY_ORDER then alpha then
// Uncategorized last.
function step3GroupedLabels() {
    const groups = new Map(); // displayCategory -> [labels]
    for (const L of step3LabelState.loaded) {
        if (step3IsExcluded(L)) continue; // excluded labels are always hidden
        const raw = (L.category == null ? '' : String(L.category)).trim();
        const cat = raw || UNCATEGORIZED;
        if (!groups.has(cat)) groups.set(cat, []);
        groups.get(cat).push(L);
    }
    // Within each group: points DESC, frames DESC, code ASC.
    for (const arr of groups.values()) {
        arr.sort((a, b) => {
            const pa = Number(a.count || 0), pb = Number(b.count || 0);
            if (pb !== pa) return pb - pa;
            const fa = Number(a.frames || 0), fb = Number(b.frames || 0);
            if (fb !== fa) return fb - fa;
            return String(a.code).localeCompare(String(b.code));
        });
    }
    // Order the categories.
    const orderIndex = {};
    CATEGORY_ORDER.forEach((c, i) => { orderIndex[c.toLowerCase()] = i; });
    const cats = Array.from(groups.keys());
    cats.sort((a, b) => {
        // Uncategorized always last.
        if (a === UNCATEGORIZED && b === UNCATEGORIZED) return 0;
        if (a === UNCATEGORIZED) return 1;
        if (b === UNCATEGORIZED) return -1;
        const ia = orderIndex[a.toLowerCase()];
        const ib = orderIndex[b.toLowerCase()];
        const aKnown = ia !== undefined, bKnown = ib !== undefined;
        if (aKnown && bKnown) return ia - ib;
        if (aKnown) return -1;          // known categories before unknown
        if (bKnown) return 1;
        return a.localeCompare(b);      // unknown categories: alphabetical
    });
    return cats.map(c => ({ category: c, labels: groups.get(c) }));
}

function renderStep3Labels() {
    const host = document.getElementById('s3-label-picker');
    if (!host) return;
    const filterRaw = (document.getElementById('s3-label-filter') || {}).value || '';
    const filter = filterRaw.trim().toLowerCase();

    // Pre-check set comes from the current hidden s3-species CSV (restores a
    // saved project's prior selection on return).
    const speciesEl = document.getElementById('s3-species');
    const preChecked = new Set(
        ((speciesEl && speciesEl.value) || '')
            .split(',').map(s => s.trim()).filter(Boolean)
    );

    const grouped = step3GroupedLabels();
    const blocks = [];
    for (const { category, labels } of grouped) {
        // Apply the text filter within the group.
        const visible = filter
            ? labels.filter(L => {
                const hay = (`${L.code} ${L.name || ''} ${category}`).toLowerCase();
                return hay.includes(filter);
            })
            : labels;
        // When filtering, hide groups with zero matches entirely.
        if (filter && visible.length === 0) continue;

        // Filtering auto-expands matching groups; no filter collapses all.
        const expanded = !!filter;
        const selCount = visible.reduce(
            (n, L) => n + (preChecked.has(String(L.code)) ? 1 : 0), 0);

        const rows = visible.map(L => {
            const checked = preChecked.has(String(L.code)) ? 'checked' : '';
            const frames = Number(L.frames || 0);
            const points = Number(L.count || 0);
            const tip = `${L.code} - ${L.name || ''} (${category}) · ` +
                        `${points.toLocaleString()} points across ${frames.toLocaleString()} frames`;
            const countLabel = frames > 0
                ? `${frames.toLocaleString()} fr / ${points.toLocaleString()} pt`
                : `${points.toLocaleString()} pt`;
            return (
                `<label class="label-picker-row" title="${escapeAttr(tip)}">
                    <input type="checkbox" class="s3-label-cb" value="${escapeAttr(String(L.code))}" data-category="${escapeAttr(category)}" data-name="${escapeAttr(L.name || '')}" ${checked}>
                    <span class="code">${escapeHtml(String(L.code))}</span>
                    <span class="name">${escapeHtml(L.name || '')}</span>
                    <span class="count">${escapeHtml(countLabel)}</span>
                </label>`
            );
        }).join('');

        blocks.push(
            `<div class="label-cat-group" data-category="${escapeAttr(category)}">
                <button type="button" class="label-cat-header" aria-expanded="${expanded ? 'true' : 'false'}" onclick="toggleStep3Group(this)">
                    <span class="caret">${expanded ? '&#9660;' : '&#9654;'}</span>
                    <span class="label-cat-name">${escapeHtml(category)}</span>
                    <span class="label-cat-meta"><span class="sel-count">${selCount}</span> of <span class="tot-count">${visible.length}</span> selected</span>
                </button>
                <div class="label-cat-body"${expanded ? '' : ' hidden'}>
                    ${rows}
                </div>
            </div>`
        );
    }

    if (!blocks.length) {
        host.innerHTML = filter
            ? '<span style="color:var(--text-dim)">No labels match the current filter.</span>'
            : '<span style="color:var(--text-dim)">No labels available.</span>';
    } else {
        host.innerHTML = blocks.join('');
    }

    // Wire each checkbox: sync hidden store, repaint chips, update its group meta.
    host.querySelectorAll('.s3-label-cb').forEach(cb => {
        cb.addEventListener('change', () => {
            syncStep3SpeciesField();
            repaintStep3Chips();
            const grp = cb.closest('.label-cat-group');
            if (grp) updateStep3GroupMeta(grp);
            previewProjectName();
        });
    });

    syncStep3SpeciesField();
    repaintStep3Chips();
    // Re-entering Step 3 re-renders the picker; if it still holds an unsaved
    // selection, restore the preview the last state refresh painted over.
    if (step3PickerDirty()) previewProjectName();
}

// Recompute and write the "<sel> of <tot> selected" header meta for one group.
function updateStep3GroupMeta(grp) {
    if (!grp) return;
    const cbs = grp.querySelectorAll('.s3-label-cb');
    let sel = 0;
    cbs.forEach(cb => { if (cb.checked) sel++; });
    const selEl = grp.querySelector('.sel-count');
    const totEl = grp.querySelector('.tot-count');
    if (selEl) selEl.textContent = String(sel);
    if (totEl) totEl.textContent = String(cbs.length);
}

function updateAllStep3GroupMeta() {
    const host = document.getElementById('s3-label-picker');
    if (!host) return;
    host.querySelectorAll('.label-cat-group').forEach(updateStep3GroupMeta);
}

// Repaint the selection chips (#s3-label-chips) from the set of currently
// selected codes: the union of checked visible boxes AND any codes already in
// the hidden s3-species field that are currently filtered out of the DOM (same
// preservation logic syncStep3SpeciesField uses). Resolves names from the
// loaded list for the chip label; falls back to the bare code if unknown.
function repaintStep3Chips() {
    const chipsEl = document.getElementById('s3-label-chips');
    const countEl = document.getElementById('s3-chip-count');
    const speciesEl = document.getElementById('s3-species');
    const codes = ((speciesEl && speciesEl.value) || '')
        .split(',').map(s => s.trim()).filter(Boolean);
    if (countEl) countEl.textContent = String(codes.length);
    if (!chipsEl) return;
    if (!codes.length) {
        chipsEl.innerHTML = '<span class="muted">None selected yet. Pick labels above.</span>';
        return;
    }
    chipsEl.innerHTML = codes.map(code => {
        const name = step3NameForCode(code);
        const nameSpan = name ? `<span class="chip-name">${escapeHtml(name)}</span>` : '';
        return `<span class="label-chip" data-code="${escapeAttr(code)}"><span class="chip-code">${escapeHtml(code)}</span>${nameSpan}<button type="button" class="chip-x" title="Remove ${escapeAttr(code)}" onclick="removeStep3Chip('${escapeAttr(code).replace(/'/g, "\\'")}')">&times;</button></span>`;
    }).join('');
}

// Source of truth feeding collectConfig(3). Merges checked visible boxes with
// any previously-selected codes that are currently filtered out of the DOM so
// filtering never silently drops a user's selection.
function syncStep3SpeciesField() {
    const host = document.getElementById('s3-label-picker');
    const speciesEl = document.getElementById('s3-species');
    if (!host || !speciesEl) return;
    const existing = (speciesEl.value || '')
        .split(',').map(s => s.trim()).filter(Boolean);
    const visibleCodes = new Set();
    const checkedVisible = [];
    host.querySelectorAll('.s3-label-cb').forEach(cb => {
        visibleCodes.add(cb.value);
        if (cb.checked) checkedVisible.push(cb.value);
    });
    const kept = existing.filter(c => !visibleCodes.has(c));
    const merged = [...new Set([...kept, ...checkedVisible])];
    speciesEl.value = merged.join(', ');
}

// Toggle one group's collapsed/expanded state (body hidden + aria + caret glyph).
function toggleStep3Group(headerEl) {
    if (!headerEl) return;
    const grp = headerEl.closest('.label-cat-group');
    if (!grp) return;
    const body = grp.querySelector('.label-cat-body');
    const caret = headerEl.querySelector('.caret');
    const open = headerEl.getAttribute('aria-expanded') === 'true';
    const next = !open;
    headerEl.setAttribute('aria-expanded', next ? 'true' : 'false');
    if (body) body.hidden = !next;
    if (caret) caret.innerHTML = next ? '&#9660;' : '&#9654;';
}

// Expand (open=true) or collapse (false) every group.
function toggleAllStep3Groups(open) {
    const host = document.getElementById('s3-label-picker');
    if (!host) return;
    host.querySelectorAll('.label-cat-group').forEach(grp => {
        const header = grp.querySelector('.label-cat-header');
        const body = grp.querySelector('.label-cat-body');
        const caret = grp.querySelector('.caret');
        if (header) header.setAttribute('aria-expanded', open ? 'true' : 'false');
        if (body) body.hidden = !open;
        if (caret) caret.innerHTML = open ? '&#9660;' : '&#9654;';
    });
}

// Check/uncheck every visible row, then sync + repaint chips + refresh meta.
// HTML now only calls this with false (Clear selection), but keep it general.
function selectAllStep3Labels(on) {
    const host = document.getElementById('s3-label-picker');
    if (!host) return;
    host.querySelectorAll('.s3-label-cb').forEach(cb => { cb.checked = !!on; });
    syncStep3SpeciesField();
    repaintStep3Chips();
    updateAllStep3GroupMeta();
    previewProjectName();
}

// Remove a single selected code (chip X). Uncheck its box if present in the DOM,
// strip it from the hidden s3-species field, then repaint everything.
function removeStep3Chip(code) {
    const host = document.getElementById('s3-label-picker');
    const speciesEl = document.getElementById('s3-species');
    if (host) {
        host.querySelectorAll('.s3-label-cb').forEach(cb => {
            if (cb.value === code) cb.checked = false;
        });
    }
    if (speciesEl) {
        const remaining = (speciesEl.value || '')
            .split(',').map(s => s.trim()).filter(Boolean)
            .filter(c => c !== code);
        speciesEl.value = remaining.join(', ');
    }
    syncStep3SpeciesField();
    repaintStep3Chips();
    updateAllStep3GroupMeta();
    previewProjectName();
}

// Wire the filter input once. Script executes after the DOM is parsed (it's at
// the bottom of <body>). The hide-excluded toggle is gone, so it is NOT wired.
(function wireStep3Picker() {
    const filterEl = document.getElementById('s3-label-filter');
    if (filterEl) filterEl.addEventListener('input', () => renderStep3Labels());
})();

function escapeHtml(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function escapeAttr(s) {
    return escapeHtml(s).replace(/"/g, '&quot;');
}

(async function init() {
    // The VICARIUS UI hands this window a project via ?project_dir= auto-open (or
    // a prior POST /api/project/open), which pre-renders #app-screen and sets
    // data-project-loaded="true". In that case go straight to the 8-step view.
    const preloaded = document.body.dataset.projectLoaded === 'true';
    const data = await api('/api/project/state');
    if (data._locked) {
        // Module locked (VICARIUS): show the lock message here, never bounce.
        document.body.innerHTML = buildLockPage();
        return;
    }
    if (preloaded && data.loaded && data.state) {
        state = data.state;
        enterApp();
        return;
    }
    // No project was handed to this window (opened bare, or after Save and
    // Close). Projects are created/opened only in the VICARIUS UI, so bounce
    // there rather than ever showing a local home/resume screen.
    bounceToVicarius();
})();

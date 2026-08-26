/* Pipeline Orchestrator — SPA Logic */

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
        el.textContent = 'Orchestrator is not responding. Restart Reef Point '
            + 'Seg from the VICARIUS launcher; this page recovers on its own '
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
    return 'http://127.0.0.1:5077/modules/reef_point_seg';
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

// Legacy callers route through showSetup(); it now bounces to the VICARIUS UI
// rather than revealing any local create/open screen.
function showSetup() {
    bounceToVicarius();
}

function enterApp() {
    document.getElementById('setup-screen').style.display = 'none';
    document.getElementById('app-screen').style.display = 'flex';
    document.getElementById('topbar-actions').style.display = 'flex';
    document.getElementById('topbar-project-name').textContent =
        state.name + '  (' + state.id + ')';

    // Fresh project context — let the step-6 default preset auto-apply
    // again for this project, and wipe any run-name-edited flag from a
    // previous session so the field reverts to blank.
    window._step6PresetAutoApplied = false;
    const nameEl = document.getElementById('s6-run-name');
    if (nameEl) delete nameEl.dataset.userEdited;

    populateConfigs();
    updateSidebar();
    // current_step is an int chain slot; slots 4 and 5 are now occupied by the
    // combined annotator (non-chain tile data-step="step4test"), whose legacy
    // panels are archived from the sidebar. Land on it rather than an orphaned,
    // nav-less legacy panel.
    let cs = state.current_step || 1;
    if (cs === 4 || cs === 5) cs = 'step4test';
    switchStep(cs);

    // A routing pass runs server-side and outlives this page, so a reload must
    // still show it. Paint it now and keep it current for as long as it is alive.
    renderStep4testRouteBanner();
    if (!window._step4testRouteTimer) {
        window._step4testRouteTimer = setInterval(renderStep4testRouteBanner, 2500);
    }
}

// ── Sidebar ─────────────────────────────────────────────────────────────────

function updateSidebar() {
    if (!state) return;
    // Cover every numeric chain slot 1..8. Slots 4 & 5 tiles are archived from
    // the sidebar (4.test replaces them) so their <li> is absent and skipped;
    // 7 (Evaluate) and 8 (Inference) must be covered so their status/checkmark
    // update (they sit below train and display as 6/7).
    for (let s = 1; s <= 8; s++) {
        const li = document.querySelector(`.step-nav li[data-step="${s}"]`);
        if (!li) continue;
        const st = state.steps[s];
        const status = st ? st.status : 'locked';

        li.className = '';
        li.classList.add(status);
        if (s === activeStep) li.classList.add('active');

        const circle = li.querySelector('.step-circle');
        if (status === 'completed') {
            circle.textContent = '\u2713';
        } else {
            // Display number may differ from the internal step id (data-step):
            // 4.test is shown as "4" and train (id 6) is shown as "5". data-num
            // on the circle carries the display number; fall back to the id.
            circle.textContent = circle.dataset.num || s;
        }

        const statusText = document.getElementById(`step${s}-status-text`);
        if (statusText) statusText.textContent = status;
    }
}

function switchStep(step) {
    activeStep = step;
    document.querySelectorAll('.step-panel').forEach(p => p.classList.remove('active'));
    const panel = document.getElementById(`panel-${step}`);
    if (panel) panel.classList.add('active');
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
            // the run-name field blank until the user types.
            const nameEl = document.getElementById('s6-run-name');
            if (nameEl && !nameEl.dataset.userEdited) nameEl.value = '';
            refreshStep6Presets({ autoApply: true });
            refreshStep6Classes();
            refreshStep6ModelRunOptions();
        }
        if (step === 7) { loadStep7Report(); refreshStep7Rounds(); }
        if (step === 8) refreshStep8Gallery();
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
// move through the flow without the sidebar. Idempotent — safe to call often.
function renderStepNavFooters() {
    const order = getNavOrder();
    order.forEach((ds, idx) => {
        const footer = document.getElementById(`nav-footer-${ds}`);
        if (!footer) return;
        const prevDs = idx > 0 ? order[idx - 1] : null;
        let nextDs = idx < order.length - 1 ? order[idx + 1] : null;
        let nextLabel = null;
        // step4test (the combined annotator) will replace Place Points + Segment,
        // so its Next jumps straight to Train Model rather than the next tile.
        if (ds === 'step4test') nextDs = '6';
        // step4loop (Refine loop) sits last in the nav order, but its Next closes
        // the active-learning loop by jumping back to Train (step 6) rather than
        // dead-ending: "Loop to Train (5)" names the loop, not just the tile.
        if (ds === 'step4loop') { nextDs = '6'; nextLabel = 'Loop to Train (5)'; }
        const isNumeric = /^\d+$/.test(ds);

        // LEFT zone — Prev (always outline).
        const prevBtn = prevDs
            ? `<button class="btn btn-outline btn-sm" onclick='switchStep(${JSON.stringify(navStepArg(prevDs))})' title="Go to ${escAttr(navTitleFor(prevDs))}">&larr; Prev: ${escHtmlNav(navTitleFor(prevDs))}</button>`
            : `<button class="btn btn-outline btn-sm" disabled>&larr; Prev</button>`;

        // CENTER zone — Reset Step (numeric panels only) sits directly next to
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

        // RIGHT zone — Next. Neutral by default, pink once this numeric step is
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
    1:'s1-fetch-btn', 2:'s2-auto-btn', 3:'s3-run-btn', 4:'s4-run-btn',
    5:'s5-run-btn', 6:'s6-run-btn', 7:'s7-run-btn', 8:'s8-run-btn'
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
    // Auto-hydrate the label picker if master_codes_recoded.csv exists.
    // loadStep3Labels pre-checks rows matching the hidden s3-species CSV.
    loadStep3Labels(false);

    // Step 4
    const s4 = state.steps['4'].config;
    setVal('s4-port', s4.port || 5065);
    setVal('s4-clip-dir', s4.clip_dir || '');
    if (s4.target_species_only !== undefined)
        document.getElementById('s4-target-only').checked = !!s4.target_species_only;
    const autoSam3El = document.getElementById('s4-auto-sam3');
    if (autoSam3El) autoSam3El.checked = s4.auto_start_sam3 !== false; // default on
    const refEl = document.getElementById('s4-reference');
    if (refEl) refEl.checked = !!s4.reference_mode;
    const shufEl = document.getElementById('s4-shuffle');
    if (shufEl) shufEl.checked = !!s4.shuffle;
    const batchEl = document.getElementById('s4-batch-size');
    if (batchEl) batchEl.value = s4.review_batch_size || '10';

    // Step 5
    const s5 = state.steps['5'].config;
    setVal('s5-port', s5.port || 5070);
    setVal('s5-tracker-device', s5.sam3_device_tracker || 'cuda:1');
    setVal('s5-exemplar-device', s5.sam3_device_exemplar || 'cuda:0');
    setVal('s5-confidence', s5.confidence_threshold || 0.5);
    setVal('s5-min-area', s5.min_mask_area_px || 500);
    setVal('s5-merge-dist', s5.merge_distance_px || 30);
    setVal('s5-thin-ratio', s5.thin_mask_ratio || 0.10);
    setVal('s5-simplify', s5.polygon_simplify_epsilon || 0.001);
    if (s5.overlap_strategy)
        document.getElementById('s5-overlap').value = s5.overlap_strategy;
    if (s5.symlink_images !== undefined)
        document.getElementById('s5-symlink').checked = !!s5.symlink_images;
    const s5batch = document.getElementById('s5-batch-size');
    if (s5batch) s5batch.value = s5.review_batch_size || '10';

    // Step 6 — training params (core + advanced)
    const s6 = (state.steps['6'] && state.steps['6'].config) || {};
    const s6set = (id, v, fallback) => setVal(id, (v !== undefined && v !== null) ? v : fallback);
    // Core
    // run_name is ALWAYS blank by default — blank signifies "new run".
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
    if (s6.model_path) { applyStep6ModelPath(s6.model_path); }
    else if (s6.model) { const sel = document.getElementById('s6-model'); if (sel) sel.value = s6.model; }
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
    // Device / DDP
    applyStep6Device(s6.device);

    // Step 7 — evaluation
    const s7 = (state.steps['7'] && state.steps['7'].config) || {};
    setVal('s7-imgsz', s7.imgsz != null ? s7.imgsz : 512);
    setVal('s7-conf', s7.conf_threshold != null ? s7.conf_threshold : 0.25);
    setVal('s7-iou', s7.iou_threshold != null ? s7.iou_threshold : 0.6);
    setVal('s7-preview-count', s7.preview_count != null ? s7.preview_count : 8);
    setVal('s7-pdf-export-dir', s7.pdf_export_dir || '');
    if (s7.split) { const sel = document.getElementById('s7-split'); if (sel) sel.value = s7.split; }

    // Step 8 — inference
    const s8 = (state.steps['8'] && state.steps['8'].config) || {};
    setVal('s8-run-name', s8.run_name || '');
    setVal('s8-imgsz', s8.imgsz != null ? s8.imgsz : 512);
    setVal('s8-source-dir', s8.source_dir || '');
    setVal('s8-sample-count', s8.sample_count != null ? s8.sample_count : 100);
    setVal('s8-conf', s8.conf_threshold != null ? s8.conf_threshold : 0.25);
    setVal('s8-iou', s8.iou_threshold != null ? s8.iou_threshold : 0.7);
    setVal('s8-mask-alpha', s8.mask_alpha != null ? s8.mask_alpha : 0.45);
    if (s8.sample_mode) { const sel = document.getElementById('s8-sample-mode'); if (sel) sel.value = s8.sample_mode; }
    const srcType = s8.source_type || 'unused';
    const radio = document.querySelector(`input[name="s8-src"][value="${srcType}"]`);
    if (radio) radio.checked = true;
    toggleStep8SourceFields();
    const so = document.getElementById('s8-save-overlays');
    if (so) so.checked = s8.save_overlays !== false;
    const sc = document.getElementById('s8-save-crops');
    if (sc) sc.checked = !!s8.save_crops;
    const pv = document.getElementById('s8-include-pts');
    if (pv) pv.checked = !!s8.include_pts_variant;
    const db = document.getElementById('s8-draw-boxes');
    if (db) db.checked = !!s8.draw_boxes;

    // Populate run dropdowns (async)
    refreshRunList();
    // Populate step-6 preset list (always — presets are static YAML files).
    // autoApply=true fills in the form from the default preset on first load
    // when the user has not yet manually imported one.
    refreshStep6Presets({ autoApply: true });
    // Populate step-7 eval-preset list too.
    refreshStep7Presets();
    // Populate step-6 class picker. Backend falls back to step 3
    // target_species if step 5 hasn't produced data.yaml yet.
    refreshStep6Classes();
    // Populate the "Continue from a previous run" model options + champion.
    refreshStep6ModelRunOptions();
    // Populate the Rounds ledger table (Task 9).
    refreshStep7Rounds();

    // If an evaluation has already been run, auto-load the report.
    if (state.steps['7'] && state.steps['7'].outputs && state.steps['7'].outputs.report_md) {
        loadStep7Report();
    }
    if (state.steps['8'] && state.steps['8'].outputs && state.steps['8'].outputs.manifest) {
        refreshStep8Gallery();
    }

    // Restore running services (cover all numeric slots; 4/5 simply have no
    // running service in the promoted flow, and 7/8 can be long-running).
    for (let s = 1; s <= 8; s++) {
        if (state.steps[s] && state.steps[s].status === 'running') {
            startPolling(s);
            if ([2, 4, 5].includes(s)) {
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
    } else if (step === 4) {
        cfg.port = parseInt(document.getElementById('s4-port').value) || 5065;
        cfg.clip_dir = document.getElementById('s4-clip-dir').value.trim();
        cfg.target_species_only = document.getElementById('s4-target-only').checked;
        const autoSam3El = document.getElementById('s4-auto-sam3');
        cfg.auto_start_sam3 = autoSam3El ? autoSam3El.checked : true;
        const refEl = document.getElementById('s4-reference');
        cfg.reference_mode = refEl ? refEl.checked : false;
        const shufEl = document.getElementById('s4-shuffle');
        cfg.shuffle = shufEl ? shufEl.checked : false;
        const batchEl = document.getElementById('s4-batch-size');
        cfg.review_batch_size = batchEl ? batchEl.value : '10';
    } else if (step === 6) {
        const $ = id => document.getElementById(id);
        const f = id => parseFloat($(id).value);
        const i = id => parseInt($(id).value);
        cfg.run_name = $('s6-run-name').value.trim();
        // "Continue from a previous run" options carry an absolute .pt path as
        // their <option value>; everything else is a catalog name like
        // yolo11m-seg.pt. Route accordingly: model_path overrides model
        // server-side (_run_step6), so only one of the two should be meaningful.
        const modelSel = $('s6-model').value;
        if (modelSel && /^\//.test(modelSel) && modelSel.endsWith('.pt')) {
            cfg.model_path = modelSel;
            cfg.model = 'yolo11m-seg.pt';  // ignored server-side once model_path is set
        } else {
            cfg.model = modelSel;
        }
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
        // Class-inclusion filter — null means "train on every class in step 5".
        cfg.include_classes = collectStep6IncludedClasses();
        // Device / DDP selection — string forwarded as ultralytics `device=`.
        cfg.device = collectStep6Device();
    } else if (step === 5) {
        cfg.port = parseInt(document.getElementById('s5-port').value) || 5070;
        cfg.sam3_device_tracker = document.getElementById('s5-tracker-device').value.trim();
        cfg.sam3_device_exemplar = document.getElementById('s5-exemplar-device').value.trim();
        cfg.confidence_threshold = parseFloat(document.getElementById('s5-confidence').value) || 0.5;
        cfg.min_mask_area_px = parseInt(document.getElementById('s5-min-area').value) || 500;
        cfg.merge_distance_px = parseInt(document.getElementById('s5-merge-dist').value) || 30;
        cfg.overlap_strategy = document.getElementById('s5-overlap').value;
        cfg.thin_mask_ratio = parseFloat(document.getElementById('s5-thin-ratio').value) || 0.10;
        cfg.polygon_simplify_epsilon = parseFloat(document.getElementById('s5-simplify').value) || 0.001;
        cfg.symlink_images = document.getElementById('s5-symlink').checked;
        const s5batch = document.getElementById('s5-batch-size');
        cfg.review_batch_size = s5batch ? s5batch.value : '10';
    } else if (step === 7) {
        const $ = id => document.getElementById(id);
        cfg.run_dir = $('s7-run-dir').value;
        cfg.split = $('s7-split').value;
        cfg.imgsz = parseInt($('s7-imgsz').value) || 512;
        cfg.conf_threshold = parseFloat($('s7-conf').value) || 0.25;
        cfg.iou_threshold = parseFloat($('s7-iou').value) || 0.6;
        cfg.preview_count = parseInt($('s7-preview-count').value) || 8;
        cfg.pdf_export_dir = $('s7-pdf-export-dir').value.trim();
    } else if (step === 8) {
        const $ = id => document.getElementById(id);
        cfg.run_dir = $('s8-run-dir').value;
        cfg.run_name = $('s8-run-name').value.trim();
        cfg.imgsz = parseInt($('s8-imgsz').value) || 512;
        const srcRadio = document.querySelector('input[name="s8-src"]:checked');
        cfg.source_type = srcRadio ? srcRadio.value : 'unused';
        cfg.source_dir = $('s8-source-dir').value.trim();
        cfg.sample_count = parseInt($('s8-sample-count').value) || 100;
        cfg.sample_mode = $('s8-sample-mode').value;
        cfg.include_pts_variant = $('s8-include-pts').checked;
        cfg.conf_threshold = parseFloat($('s8-conf').value) || 0.25;
        cfg.iou_threshold = parseFloat($('s8-iou').value) || 0.7;
        cfg.mask_alpha = parseFloat($('s8-mask-alpha').value);
        if (isNaN(cfg.mask_alpha)) cfg.mask_alpha = 0.45;
        cfg.draw_boxes = $('s8-draw-boxes').checked;
        cfg.save_overlays = $('s8-save-overlays').checked;
        cfg.save_crops = $('s8-save-crops').checked;
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

        // Step 3: species progress
        m = line.match(/(\d+)\s*frames?\s*selected/i);
        if (m) {
            showProgress(step, `${parseInt(m[1]).toLocaleString()} frames selected`, null);
            return;
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
    await put(`/api/project/config/${step}`, cfg);
    Object.assign(state.steps[step].config, cfg);

    const data = await post(`/api/step/${step}/run`, {});
    if (data._locked) {
        return alert(data.message ||
            'Under development. Please come back later. Email Lauren for questions.');
    }
    if (data.error) return alert('Error: ' + data.error);

    state.steps[step].status = 'running';
    updateSidebar();

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
    updateSidebar();
    populateConfigs();
    showResult(1, true, `Linked existing output: ${data.rows ? data.rows.toLocaleString() + ' rows' : 'OK'}. Ready for Step 2.`);
    markStepActionDone(1);
}

// ── Polling ─────────────────────────────────────────────────────────────────

// SAM3 driver status polling (step 5 only) — runs independently so it stays
// alive even if the sub-app takes minutes to load models.
let sam3PollTimer = null;

function startSam3Polling() {
    if (sam3PollTimer) clearInterval(sam3PollTimer);
    sam3PollTimer = setInterval(updateSam3Panel, 2000);
    updateSam3Panel();
}

function stopSam3Polling() {
    if (sam3PollTimer) { clearInterval(sam3PollTimer); sam3PollTimer = null; }
}

// Update a "batches: X done / Y left" readout from per-frame SAM3 progress.
// `prefix` is 's4' or 's5'. We translate the frame counts the orchestrator
// already reports into batches using the panel's configured review batch size.
// TODO: the orchestrator's sam3_status reports frames, not review batches —
// when a dedicated batch-count field is added to the status payload, read it
// directly instead of dividing frames by the configured batch size.
function updateBatchReadout(prefix, processed, total) {
    const doneEl = document.getElementById(`${prefix}-batch-done`);
    const leftEl = document.getElementById(`${prefix}-batch-left`);
    if (!doneEl || !leftEl) return;
    const sel = document.getElementById(`${prefix}-batch-size`);
    let size = sel ? parseInt(sel.value, 10) : NaN;
    if (!Number.isFinite(size) || size <= 0) size = 1; // 'all' or unset -> per-frame
    const totalBatches = total > 0 ? Math.ceil(total / size) : 0;
    // When every frame is processed, every batch is done — including the final
    // partial batch that floor() would otherwise round down (e.g. 25/25 frames
    // at size 10 is 3 of 3, not 2). Mid-run, never let the done count exceed
    // the total batch count.
    let doneBatches;
    if (total > 0 && processed >= total) {
        doneBatches = totalBatches;
    } else {
        doneBatches = processed > 0 ? Math.floor(processed / size) : 0;
        doneBatches = Math.min(doneBatches, totalBatches);
    }
    const left = Math.max(0, totalBatches - doneBatches);
    doneEl.textContent = String(doneBatches);
    leftEl.textContent = String(left);
}

async function updateSam3MiniPanel() {
    const phaseEl = document.getElementById('s4-sam3-phase');
    if (!phaseEl) return;
    let s;
    try { s = await api('/api/step/5/sam3_status'); }
    catch (e) { return; }
    const msgEl = document.getElementById('s4-sam3-message');
    const fillEl = document.getElementById('s4-sam3-fill');
    const countsEl = document.getElementById('s4-sam3-counts');
    const errEl = document.getElementById('s4-sam3-err');
    phaseEl.textContent = s.phase || 'idle';
    if (msgEl) msgEl.textContent = s.message || '';
    const processed = s.processed || 0, total = s.total || 0;
    const pct = total > 0 ? Math.round((processed / total) * 100) : 0;
    if (fillEl) fillEl.style.width = pct + '%';
    if (countsEl) countsEl.textContent = total ? `${processed}/${total} (${pct}%)` : '';
    updateBatchReadout('s4', processed, total);
    if (errEl) {
        if (s.error) { errEl.textContent = 'Error: ' + s.error; errEl.style.display = 'block'; }
        else errEl.style.display = 'none';
    }
}

// Keep the Step 4 widget fresh even when the user is on the OCR UI tab.
let sam3MiniTimer = null;
function startSam3MiniPoll() {
    if (sam3MiniTimer) return;
    updateSam3MiniPanel();
    sam3MiniTimer = setInterval(updateSam3MiniPanel, 2500);
}

async function updateSam3Panel() {
    let s;
    try { s = await api('/api/step/5/sam3_status'); }
    catch (e) { return; }

    const phaseEl = document.getElementById('s5-sam3-phase');
    const msgEl = document.getElementById('s5-sam3-message');
    const fillEl = document.getElementById('s5-sam3-fill');
    const countsEl = document.getElementById('s5-sam3-counts');
    const errEl = document.getElementById('s5-sam3-error');

    if (phaseEl) phaseEl.textContent = s.phase || 'idle';
    if (msgEl) msgEl.textContent = s.message || '';
    const processed = s.processed || 0, total = s.total || 0;
    const pct = total > 0 ? Math.round((processed / total) * 100) : 0;
    if (fillEl) fillEl.style.width = pct + '%';
    if (countsEl) countsEl.textContent = `${processed} / ${total}` + (total ? ` (${pct}%)` : '');
    updateBatchReadout('s5', processed, total);

    if (s.error) {
        if (errEl) { errEl.textContent = 'Error: ' + s.error; errEl.style.display = 'block'; }
    } else if (errEl) {
        errEl.style.display = 'none';
    }

    if (s.phase === 'review_ready' || s.phase === 'error') {
        stopSam3Polling();
    }
}

function startPolling(step) {
    if (pollTimers[step]) clearInterval(pollTimers[step]);

    let logOffset = 0;

    if (step === 5) startSam3Polling();

    pollTimers[step] = setInterval(async () => {
        const status = await api(`/api/step/${step}/status`);

        // Update log — server caps its in-memory buffer, but returns a
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
            // No new lines this tick but server advanced the total — keep in sync.
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

        // CLI completion
        if (!status.running) {
            clearInterval(pollTimers[step]);
            delete pollTimers[step];
            stopElapsedTimer(step);

            if (status.completed || status.exit_code === 0) {
                const proj = await api('/api/project/state');
                if (proj.state) {
                    state = proj.state;
                    updateSidebar();
                    populateConfigs();
                }
                showProgress(step, 'Complete!', 100);
                setTimeout(() => hideProgress(step), 3000);
                showResult(step, true, 'Step completed successfully.');
                markStepActionDone(step);
                // Step-specific post-completion hooks
                if (step === 7) loadStep7Report();
                if (step === 8) refreshStep8Gallery();
            } else if (status.exit_code !== null && status.exit_code !== 0) {
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

// ── Open service step (single-button flow: run-if-needed -> open tab) ──────

// One button per service step (2, 4, 5): if not running yet, start it. Show a
// loading overlay while we poll status for up to ~10s until the port is healthy
// (or at least running). Only window.open() once it's actually up — never open a
// dead tab. If it never comes up, surface an error in the overlay instead.
const OPEN_UI_LABELS = { 2: 'Recode', 4: 'Place Points', 5: 'SAM3 Review', step4test: 'Combined Annotator', step4loop: 'Refine Review', editmasks: 'Edit Masks' };

async function openStepService(step) {
    if (!state) return;
    const st = state.steps[step];
    const status = st && st.status;

    showOpenUiOverlay(OPEN_UI_LABELS[step] || `Step ${step}`);

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
        setOpenUiError('Service port not set yet — the UI did not come up in time.');
        return;
    }
    if (!healthy) {
        // Port known but health never confirmed within the window. Don't open a
        // potentially-dead tab; let the user retry once it's warmed up.
        setOpenUiError(`The ${OPEN_UI_LABELS[step] || 'service'} UI is still starting (port ${port}) and did not respond in ~10s.`);
        return;
    }

    // Step 5 (SAM3 Review) is additionally gated on segmentation progress: the
    // review UI is only meaningful once SAM3 has produced at least one frame
    // (or the queue finished). Port-health alone is not enough for this step.
    if (step === 5) {
        let sam3 = null;
        try { sam3 = await api('/api/step/5/sam3_status'); } catch (e) { sam3 = null; }
        const processed = (sam3 && sam3.processed) || 0;
        const phase = sam3 && sam3.phase;
        const readyForReview = processed >= 1 || phase === 'review_ready';
        if (!readyForReview) {
            setOpenUiError('SAM3 has not segmented any frames yet — the Review UI opens once the first frame is ready.');
            return;
        }
    }

    hideOpenUiOverlay();
    const url = `http://localhost:${port}/`;
    // Open every sub-tool UI in a real new WINDOW, never a tab, so this main
    // orchestrator window is always there to return to (Lauren: never a tab).
    // One named popup per step lets several sub-tools coexist without clobbering
    // each other. No 'noopener' — we keep the handle so markDone()/saveAndClose()
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

async function openStep4test() {
    showOpenUiOverlay(OPEN_UI_LABELS['step4test'] || 'Combined Annotator');
    // Open the annotator popup NOW - synchronously, inside this click handler -
    // so the browser ties it to the user gesture. Routing + the SAM3 load can
    // take ~a minute; deferring window.open() until after that await chain (as
    // this used to) means the popup blocker silently kills it and "Open" appears
    // to do nothing. The popup shows a loading page until we navigate it to the
    // annotator once it is healthy, or we close it on failure.
    const win = openStep4testPlaceholder();
    if (!win) {
        setOpenUiError('Your browser blocked the annotator popup. Allow pop-ups for this page, then click Open Combined Annotator again.');
        return;
    }
    serviceWindows['step4test'] = win;
    let r = null;
    try {
        r = await post('/api/step/step4test/run', { config: collectStep4testConfig() });
    } catch (e) { setOpenUiError('Failed to start: ' + (e && e.message || e)); closeStep4testWin(win); return; }
    if (r && r.error && !/already running/i.test(r.error)) {
        setOpenUiError('Failed to start: ' + r.error); closeStep4testWin(win); return;
    }
    await finishStep4testOpen(r, win);
}

// Open the annotator popup immediately (same user gesture as the click) so the
// popup blocker lets it through, then paint a loading message into it. We
// navigate THIS window to the service once it is healthy - never a second
// window.open() from inside an async callback, which the browser would block.
function openStep4testPlaceholder() {
    let win;
    try {
        win = window.open('', 'reefpointseg_step4test',
            'popup,width=1500,height=980,scrollbars=yes,resizable=yes');
    } catch (e) { win = null; }
    if (!win) return null;
    try {
        win.document.title = 'Combined Annotator - starting';
        win.document.body.style.cssText = 'margin:0;font-family:system-ui,-apple-system,sans-serif;'
            + 'background:#12141a;color:#e8e8ea;display:flex;align-items:center;justify-content:center;height:100vh';
        // Live region, not a frozen sentence. This window is the one the operator
        // actually watches, so routing progress has to land HERE - it used to paint
        // "up to a minute" once and never update, so a 20-minute pass looked hung
        // while the real progress bar ticked away in the main tab behind it.
        win.document.body.innerHTML =
            '<div style="text-align:center;max-width:460px;padding:0 24px">'
          + '<div style="font-size:15px" id="boot-title">Starting the Combined Annotator&hellip;</div>'
          + '<div style="margin-top:10px;font-size:12px;color:#9aa0ae;line-height:1.5" id="boot-msg">'
          + 'Routing the chosen images. This runs about one image per second, so a large '
          + 'selection can take several minutes. The tool loads here automatically when it is ready.</div>'
          + '<div style="margin-top:14px;height:6px;border-radius:3px;background:#252833;overflow:hidden">'
          + '<div id="boot-bar" style="height:100%;width:0%;background:#c14bd6;transition:width .3s"></div></div>'
          + '<div style="margin-top:14px;font-size:11px;color:#6b7280;line-height:1.5" id="boot-hint">'
          + 'Closing this window cancels the routing pass. Reopening resumes where it stopped.</div></div>';
    } catch (e) {}
    try { win.focus(); } catch (e) {}
    return win;
}

// Paint routing progress into the POPUP the operator is actually looking at.
// Silently no-ops once the popup has navigated to the annotator (cross-document
// access throws) or was closed.
function paintStep4testBoot(win, { title, msg, pct } = {}) {
    if (!win || win.closed) return;
    try {
        if (title) { const t = win.document.getElementById('boot-title'); if (t) t.textContent = title; }
        if (msg) { const m = win.document.getElementById('boot-msg'); if (m) m.textContent = msg; }
        if (typeof pct === 'number') {
            const b = win.document.getElementById('boot-bar');
            if (b) b.style.width = Math.max(0, Math.min(100, pct)) + '%';
        }
    } catch (e) {}
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
async function renderStep4testRouteBanner() {
    const el = document.getElementById('s4test-route-banner');
    if (!el) return;
    let rs; try { rs = await api('/api/step/step4test/route_status'); } catch (e) { rs = null; }

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
            ? `Routing chosen images… ${p}/${t} (${pct}%) — about ${Math.max(1, Math.round((t - p) / 60))} min left`
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

    // Routing in progress: poll route_status, updating the overlay AND the popup.
    const msg = document.getElementById('open-ui-msg');
    const sub = document.getElementById('open-ui-sub');
    while (true) {
        // Closed window == cancel. Not just "stop polling": the routing thread would
        // otherwise keep the GPU and, on finishing, kill whatever annotator is up to
        // launch one nobody is waiting for.
        if (win && win.closed) {
            await cancelStep4testRoute();
            closeStep4testWin(win);
            hideOpenUiOverlay();
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
                setOpenUiError(rs.error || 'Routing failed.', tail || null);
                closeStep4testWin(win);
                return;
            }
            if (rs.phase === 'cancelled') {
                closeStep4testWin(win);
                hideOpenUiOverlay();
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
            if (msg) {
                msg.innerHTML = 'Routing chosen images&hellip; ' + processed + '/' + total
                    + '<div style="margin-top:10px;height:6px;border-radius:3px;background:var(--border);overflow:hidden">'
                    + '<div style="height:100%;width:' + pct + '%;background:var(--accent);transition:width .3s"></div></div>';
            }
            if (sub) {
                let subText = rs.message || 'Detecting reference points from Step 3 selections.';
                if (rs.warnings && rs.warnings.length) {
                    subText += '  ' + rs.warnings.join('  ');
                }
                sub.textContent = subText;
            }
            // Mirror it into the popup, which is the window she is actually watching.
            paintStep4testBoot(win, {
                title: total ? `Routing chosen images… ${processed}/${total}` : 'Routing chosen images…',
                msg: total
                    ? `About ${Math.max(1, Math.round((total - processed) / 60))} min left at ~1 image/sec. `
                      + 'The annotator opens here automatically when routing finishes.'
                    : 'Detecting reference points from your Step 3 selections.',
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
// it — that was the "Open does nothing" bug.
async function openStep4testWindow(uiPort, win) {
    // Wait on the PROCESS, not on a fixed clock. A cold SAM3 load can take well
    // over a minute, and a fixed 120s deadline gave up 4 seconds before a healthy
    // annotator finished loading (2026-07-14). So: keep waiting while the annotator
    // process is alive, up to the same 10-minute ceiling the server gives SAM3, and
    // fail fast when NOTHING is running behind the port instead of burning it.
    const ceiling = Date.now() + 600000;
    const msg = document.getElementById('open-ui-msg');
    let port = uiPort || null, healthy = false, waited = 0, deadProbes = 0;
    while (Date.now() < ceiling) {
        // Closed mid-launch == stop launching. Leaving a SAM3 process loading for a
        // window that no longer exists is exactly the kind of invisible work that
        // has no way to be seen or stopped from the UI.
        if (win && win.closed) {
            try { await post('/api/step/step4test/stop', {}); } catch (e) {}
            closeStep4testWin(win);
            hideOpenUiOverlay();
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
        if (msg && waited >= 3) {
            msg.innerHTML = 'Starting the Combined Annotator and loading the SAM3 model&hellip;'
                + '<div style="margin-top:8px;font-size:12px;color:var(--text-dim)">'
                + 'A cold model load can take a few minutes. The window opens automatically.</div>';
        }
        await new Promise(r => setTimeout(r, 500));
    }
    if (!port) { setOpenUiError('Service port not set yet.'); closeStep4testWin(win); return; }
    if (!healthy) {
        setOpenUiError(deadProbes >= 20
            ? `No Combined Annotator process is running on port ${port} — the launch did not start or it exited. Check the service log.`
            : `Combined Annotator did not become ready (port ${port}). Check the service log.`);
        closeStep4testWin(win); return;
    }

    hideOpenUiOverlay();
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
    else { setOpenUiError('The annotator is ready but the popup was blocked. Open it directly at ' + url); }
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
        if (resp && resp.error) alert('Open folder failed: ' + resp.error);
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
    showOpenUiOverlay(OPEN_UI_LABELS['step4loop'] || 'Refine Review');
    // Open the popup synchronously, inside the click handler, same reason as
    // step4test (see openStep4testPlaceholder); the popup blocker only allows
    // window.open() tied to the original user gesture.
    const win = openStep4loopPlaceholder();
    if (!win) {
        setOpenUiError('Your browser blocked the review popup. Allow pop-ups for this page, then click Seed + Open Review again.');
        return;
    }
    serviceWindows['step4loop'] = win;
    const svc = document.getElementById('s4loop-service');
    if (svc) svc.style.display = 'block';
    const stopBtn = document.getElementById('s4loop-stop-btn');
    if (stopBtn) stopBtn.style.display = 'inline-block';
    const next = document.getElementById('s4loop-next-card');
    if (next) next.style.display = 'none';
    const resultEl = document.getElementById('s4loop-seed-result');
    if (resultEl) resultEl.innerHTML = '';

    let r = null;
    try {
        r = await post('/api/step/step4loop/run', { config: collectStep4loopConfig() });
    } catch (e) {
        setOpenUiError('Failed to start: ' + (e && e.message || e));
        closeStep4loopWin(win);
        return;
    }
    if (!r || r.success !== true) {
        const msg = (r && r.error) || 'Seeding failed.';
        setOpenUiError(msg);
        if (resultEl) resultEl.innerHTML = `<div class="badge badge-error" style="margin-top:8px">${escHtmlNav(msg)}</div>`;
        closeStep4loopWin(win);
        return;
    }
    renderStep4loopSeedSummary(r.seeded);
    await finishStep4loopOpen(r, win);
}

function openStep4loopPlaceholder() {
    let win;
    try {
        win = window.open('', 'reefpointseg_step4loop',
            'popup,width=1500,height=980,scrollbars=yes,resizable=yes');
    } catch (e) { win = null; }
    if (!win) return null;
    try {
        win.document.title = 'Refine Review - starting';
        win.document.body.style.cssText = 'margin:0;font-family:system-ui,-apple-system,sans-serif;'
            + 'background:#12141a;color:#e8e8ea;display:flex;align-items:center;justify-content:center;height:100vh';
        win.document.body.innerHTML =
            '<div style="text-align:center;max-width:420px;padding:0 24px">'
          + '<div style="font-size:15px">Seeding model predictions and starting the review UI&hellip;</div>'
          + '<div style="margin-top:10px;font-size:12px;color:#9aa0ae;line-height:1.5">'
          + 'This can take up to a minute; the tool loads here automatically when it is ready.</div></div>';
    } catch (e) {}
    try { win.focus(); } catch (e) {}
    return win;
}

function closeStep4loopWin(win) {
    try { if (win) win.close(); } catch (e) {}
    if (serviceWindows['step4loop'] === win) delete serviceWindows['step4loop'];
    const svc = document.getElementById('s4loop-service');
    if (svc) svc.style.display = 'none';
}

// Render the seeder's summary counts, including skipped_holdout (normal, not
// an error: those are frames on the frozen T5/T6 transects the loop never
// seeds, so they never show up for review here).
function renderStep4loopSeedSummary(summary) {
    const el = document.getElementById('s4loop-seed-result');
    if (!el || !summary) return;
    const seeded = summary.seeded_frames || 0;
    const masks = summary.seeded_masks || 0;
    const holdout = summary.skipped_holdout || 0;
    const existing = summary.skipped_existing || 0;
    const belowConf = summary.skipped_below_conf || 0;
    el.innerHTML =
        `<div class="badge badge-success" style="margin-top:8px;display:block;white-space:normal">` +
        `Seeded ${seeded} frame(s), ${masks} mask(s) for review.` +
        (holdout ? ` <span title="Frames on the held-out test/validation transects (T5/T6), reserved for the frozen evaluation gate; the loop never lets model predictions touch them, so they are not reviewable here.">${holdout} on held-out transects (skipped, expected)</span>.` : '') +
        (existing ? ` ${existing} already in the labelset.` : '') +
        (belowConf ? ` ${belowConf} below the confidence threshold.` : '') +
        `</div>`;
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
    const msg = document.getElementById('open-ui-msg');
    let port = (r && r.ui_port) || null, healthy = false, waited = 0, deadProbes = 0;
    while (Date.now() < ceiling) {
        if (win && win.closed) { closeStep4loopWin(win); hideOpenUiOverlay(); onStep4loopWindowClosed(); return; }
        let s; try { s = await api('/api/step/step4loop/status'); } catch (e) { s = null; }
        if (s && s.port) port = s.port;
        if (s && s.healthy) { healthy = true; break; }
        if (s && !s.running) { deadProbes += 1; if (deadProbes >= 20) break; }
        else { deadProbes = 0; }
        waited += 1;
        if (msg && waited >= 3) {
            msg.innerHTML = 'Starting the Refine review UI&hellip;'
                + '<div style="margin-top:8px;font-size:12px;color:var(--text-dim)">'
                + 'A cold model load can take a few minutes. The window opens automatically.</div>';
        }
        await new Promise(res => setTimeout(res, 500));
    }
    if (!port) { setOpenUiError('Service port not set yet.'); closeStep4loopWin(win); return; }
    if (!healthy) {
        setOpenUiError(deadProbes >= 20
            ? `No Refine Review process is running on port ${port} — the launch did not start or it exited. Check the service log.`
            : `Refine Review did not become ready (port ${port}). Check the service log.`);
        closeStep4loopWin(win); return;
    }

    hideOpenUiOverlay();
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
    else { setOpenUiError('The review UI is ready but the popup was blocked. Open it directly at ' + url); }
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
        if (resp && resp.error) alert('Open folder failed: ' + resp.error);
    } catch (e) { alert('Open folder failed: ' + e); }
}

// ── Edit Masks (editmasks): standalone relabel/fix tool ────────────────────
// Non-chain tile, same popup-safe launch pattern as step4loop (see above), but
// simpler still: /api/step/editmasks/run has no config to collect and no
// seeding pass - it preflights that the export has masks, then launches
// directly and always comes back with phase:"ui_ready" (or a 4xx error).

async function openEditMasks() {
    showOpenUiOverlay(OPEN_UI_LABELS['editmasks'] || 'Edit Masks');
    // Open the popup synchronously, inside the click handler, same reason as
    // step4loop/step4test (the popup blocker only allows window.open() tied
    // to the original user gesture).
    const win = openEditMasksPlaceholder();
    if (!win) {
        setOpenUiError('Your browser blocked the Edit Masks popup. Allow pop-ups for this page, then click Open Edit Masks again.');
        return;
    }
    serviceWindows['editmasks'] = win;
    const svc = document.getElementById('s4em-service');
    if (svc) svc.style.display = 'block';
    const stopBtn = document.getElementById('s4em-stop-btn');
    if (stopBtn) stopBtn.style.display = 'inline-block';

    let r = null;
    try {
        r = await post('/api/step/editmasks/run', {});
    } catch (e) {
        setOpenUiError('Failed to start: ' + (e && e.message || e));
        closeEditMasksWin(win);
        return;
    }
    if (!r || r.success !== true) {
        const msg = (r && r.error) || 'Failed to start Edit Masks.';
        setOpenUiError(msg);
        closeEditMasksWin(win);
        return;
    }
    await finishEditMasksOpen(r, win);
}

function openEditMasksPlaceholder() {
    let win;
    try {
        win = window.open('', 'reefpointseg_editmasks',
            'popup,width=1500,height=980,scrollbars=yes,resizable=yes');
    } catch (e) { win = null; }
    if (!win) return null;
    try {
        win.document.title = 'Edit Masks - starting';
        win.document.body.style.cssText = 'margin:0;font-family:system-ui,-apple-system,sans-serif;'
            + 'background:#12141a;color:#e8e8ea;display:flex;align-items:center;justify-content:center;height:100vh';
        win.document.body.innerHTML =
            '<div style="text-align:center;max-width:420px;padding:0 24px">'
          + '<div style="font-size:15px">Starting the Edit Masks tool&hellip;</div>'
          + '<div style="margin-top:10px;font-size:12px;color:#9aa0ae;line-height:1.5">'
          + 'This can take up to a minute; the tool loads here automatically when it is ready.</div></div>';
    } catch (e) {}
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
    const msg = document.getElementById('open-ui-msg');
    let port = (r && r.ui_port) || null, healthy = false, waited = 0, deadProbes = 0;
    while (Date.now() < ceiling) {
        if (win && win.closed) { closeEditMasksWin(win); hideOpenUiOverlay(); onEditMasksWindowClosed(); return; }
        let s; try { s = await api('/api/step/editmasks/status'); } catch (e) { s = null; }
        if (s && s.port) port = s.port;
        if (s && s.healthy) { healthy = true; break; }
        if (s && !s.running) { deadProbes += 1; if (deadProbes >= 20) break; }
        else { deadProbes = 0; }
        waited += 1;
        if (msg && waited >= 3) {
            msg.innerHTML = 'Starting the Edit Masks tool&hellip;'
                + '<div style="margin-top:8px;font-size:12px;color:var(--text-dim)">'
                + 'A cold model load can take a few minutes. The window opens automatically.</div>';
        }
        await new Promise(res => setTimeout(res, 500));
    }
    if (!port) { setOpenUiError('Service port not set yet.'); closeEditMasksWin(win); return; }
    if (!healthy) {
        setOpenUiError(deadProbes >= 20
            ? `No Edit Masks process is running on port ${port} — the launch did not start or it exited. Check the service log.`
            : `Edit Masks did not become ready (port ${port}). Check the service log.`);
        closeEditMasksWin(win); return;
    }

    hideOpenUiOverlay();
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
    else { setOpenUiError('Edit Masks is ready but the popup was blocked. Open it directly at ' + url); }
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
        if (resp && resp.error) alert('Open folder failed: ' + resp.error);
    } catch (e) { alert('Open folder failed: ' + e); }
}

// After-export card buttons: jump to Step 6 and apply the matching preset.
function gotoStep6Preset(presetId) {
    switchStep(6);
    applyStep6BuiltinPreset(presetId);
}

// Add Expert IDs is now a native panel (the _expertids blueprint + expertids.js).
// switchStep('expertids') calls window.ExpertIDs.onShow() to refresh it; there is
// no longer any subprocess/iframe wiring here.

// ── Open folder helpers (xdg-open on server) ──────────────────────────────

async function openStepFolder(step) {
    if (!state) return;
    const dir = state.steps && state.steps[String(step)] && state.steps[String(step)].dir;
    if (!dir) {
        alert(`Step ${step} has no output folder yet.`);
        return;
    }
    try {
        const resp = await post('/api/fs/open', { path: dir });
        if (resp && resp.error) alert('Open folder failed: ' + resp.error);
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
        if (resp && resp.error) alert('Open folder failed: ' + resp.error);
    } catch (e) {
        alert('Open folder failed: ' + e);
    }
}

// ── Step control ────────────────────────────────────────────────────────────

async function stopStep(step) {
    await post(`/api/step/${step}/stop`, {});
    if (pollTimers[step]) {
        clearInterval(pollTimers[step]);
        delete pollTimers[step];
    }
    if (step === 5) stopSam3Polling();
    stopElapsedTimer(step);
    hideProgress(step);
    state.steps[step].status = 'pending';
    updateSidebar();
    const svc = document.getElementById(`s${step}-service`);
    if (svc) svc.style.display = 'none';
}

async function markDone(step) {
    const data = await post(`/api/step/${step}/done`, {});
    if (data.error) return alert('Error: ' + data.error);
    if (data.state) state = data.state;
    if (pollTimers[step]) {
        clearInterval(pollTimers[step]);
        delete pollTimers[step];
    }
    if (step === 5) stopSam3Polling();
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
    4: "This clears placed-point exports for this step and re-locks Steps 5 through 8.",
    5: "This removes the SAM3 masks and YOLO export for this step and re-locks Steps 6 through 8.",
    6: "This removes the dataset split and training runs produced by this step and re-locks Steps 6 and 7.",
    7: "This removes the evaluation report and metrics produced by this step.",
    8: "This removes the inference run outputs and gallery produced by this step.",
    step4test: "This removes this tool's routed_input and its annotations (all_images, all_labels, segmentations). The next Open re-routes Step 3's chosen images from scratch.",
    step4loop: "Removes model proposals you have not reviewed yet (pending cyan masks seeded by Refine). Accepted masks, your manual masks, and everything already exported are kept.",
    editmasks: "Stops the Edit Masks service. This never deletes or changes any mask; your Step 4 export is left exactly as it was.",
};

// The actual reset network/DOM work. Called only after the confirm modal is
// accepted (openResetConfirm wires this to #confirm-ok-btn).
async function doResetStep(step) {
    const data = await post(`/api/step/${step}/reset`, {});
    if (data.error) return alert('Error: ' + data.error);
    if (data.state) state = data.state;
    updateSidebar();
    const logEl = document.getElementById(`s${step}-log`);
    if (logEl) { logEl.innerHTML = ''; logEl.style.display = 'none'; }
    const resultEl = document.getElementById(`s${step}-result`);
    if (resultEl) resultEl.innerHTML = '';
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
function openResetConfirm(step) {
    const titleEl = document.getElementById('confirm-title');
    if (titleEl) titleEl.textContent = 'Reset this step?';
    const msgEl = document.getElementById('confirm-msg');
    if (msgEl) {
        // editmasks reset is stop-only (never touches the export), so the
        // "cannot be undone" warning would contradict its own description.
        const destructive = step !== 'editmasks';
        msgEl.textContent = (RESET_DESCRIPTIONS[step] || 'This clears this step\'s outputs.') +
            (destructive ? ' This cannot be undone.' : '');
    }
    const okBtn = document.getElementById('confirm-ok-btn');
    if (okBtn) { okBtn.textContent = 'Confirm Reset'; okBtn.onclick = () => { closeConfirmModal(); doResetStep(step); }; }
    const ov = document.getElementById('confirm-overlay');
    if (ov) ov.classList.add('visible');
}

function closeConfirmModal() {
    const ov = document.getElementById('confirm-overlay');
    if (ov) ov.classList.remove('visible');
}

// Save the project and close this window. Stops running sub-tools server-side
// (/api/project/quit calls pm.save_project + runner.kill_all) and returns to
// the setup screen; window.close() succeeds when this was a script-opened window.
async function saveAndClose() {
    try { Object.values(serviceWindows).forEach(w => { try { w.close(); } catch(e){} }); } catch(e){}
    serviceWindows = {};
    await post('/api/project/quit', {});
    state = null;
    // This window was opened by the VICARIUS UI; close it to return there. If the
    // browser refuses (tab not script-opened), bounce so the user never lands on
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
    device:         v => applyStep6Device(v),
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

async function refreshStep6Presets(opts) {
    opts = opts || {};
    const sel = document.getElementById('s6-preset-select');
    const desc = document.getElementById('s6-preset-desc');
    if (!sel) return;
    try {
        const data = await api('/api/step/6/presets');
        const presets = data.presets || [];
        const prev = sel.value;
        sel.innerHTML = '<option value="">(none selected)</option>';
        for (const p of presets) {
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = p.error ? `${p.id} [error: ${p.error}]` : p.name;
            opt.dataset.description = p.description || '';
            if (p.default) opt.dataset.isDefault = '1';
            sel.appendChild(opt);
        }
        if (prev) sel.value = prev;
        sel.onchange = () => {
            const opt = sel.options[sel.selectedIndex];
            if (desc) desc.textContent = opt ? (opt.dataset.description || '') : '';
        };
        sel.onchange();

        // Auto-apply the default preset on first panel load (or when caller
        // explicitly asks for autoApply), unless the user has already
        // imported or selected one. Selection rule: server's default_id,
        // which honors `default: true` in the preset YAML and falls back to
        // alphabetical-first.
        const defaultId = data.default_id || null;
        if (opts.autoApply && defaultId && !window._step6PresetAutoApplied && !prev) {
            sel.value = defaultId;
            sel.onchange();
            try {
                await importStep6Preset({ auto: true });
                window._step6PresetAutoApplied = true;
            } catch (e) {
                console.warn('auto-apply preset failed:', e);
            }
        }
    } catch (e) {
        console.warn('refreshStep6Presets failed:', e);
        if (desc) desc.textContent = `Error loading presets: ${e}`;
    }
}

async function importStep6Preset(opts) {
    opts = opts || {};
    const sel = document.getElementById('s6-preset-select');
    const desc = document.getElementById('s6-preset-desc');
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
        let applied = 0, skipped = [];
        for (const [key, val] of Object.entries(params)) {
            const fn = STEP6_PRESET_APPLIERS[key];
            if (fn) { fn(val); applied++; }
            else { skipped.push(key); }
        }
        if (desc) {
            const prefix = opts.auto ? `Auto-loaded default preset "${data.name}"` : `Imported "${data.name}"`;
            const msg = `${prefix} — ${applied} field(s) updated` +
                (skipped.length ? ` (ignored: ${skipped.join(', ')})` : '');
            desc.textContent = msg;
            desc.style.color = 'var(--magenta-light)';
            setTimeout(() => { desc.style.color = 'var(--text-dim)'; }, 4000);
        }
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
            const msg = `Imported "${data.name}" — ${applied} field(s) updated` +
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
async function applyStep7Defaults() {
    const status = document.getElementById('s7-defaults-status');
    const runSel = document.getElementById('s7-run-dir');
    const splitSel = document.getElementById('s7-split');
    if (splitSel) splitSel.value = 'test';
    setVal('s7-conf', 0.25);
    setVal('s7-iou', 0.6);
    setVal('s7-preview-count', 8);

    let imgszMsg = 'fallback 1024';
    let chosenImgsz = 1024;
    const runPath = runSel && runSel.value;
    if (runPath) {
        try {
            const info = await api(`/api/step/7/run_info?path=${encodeURIComponent(runPath)}`);
            if (info && info.imgsz) {
                chosenImgsz = info.imgsz;
                imgszMsg = `auto-read from run args.yaml -> ${chosenImgsz}`;
            } else {
                imgszMsg = 'run has no args.yaml; fallback 1024';
            }
        } catch (e) {
            imgszMsg = `run_info error: ${e}; fallback 1024`;
        }
    } else {
        imgszMsg = 'no run selected; fallback 1024 — re-click after picking a run';
    }
    setVal('s7-imgsz', chosenImgsz);

    if (status) {
        status.textContent = `Applied recommended defaults (split=test, conf=0.25, iou=0.6, preview=8, imgsz: ${imgszMsg}).`;
        status.style.color = 'var(--magenta-light)';
        setTimeout(() => { status.style.color = 'var(--text-dim)'; }, 5000);
    }
}

// Fires when the user picks a different training run in the Step 7 dropdown.
// If the imgsz field still holds the default 1024, auto-sync it to the run's
// training imgsz so metrics stay honest without the user having to remember.
async function onStep7RunChange() {
    const runSel = document.getElementById('s7-run-dir');
    if (!runSel || !runSel.value) return;
    try {
        const info = await api(`/api/step/7/run_info?path=${encodeURIComponent(runSel.value)}`);
        if (info && info.imgsz) {
            const imgszEl = document.getElementById('s7-imgsz');
            if (imgszEl) {
                const cur = parseInt(imgszEl.value);
                // Only auto-update if field is empty or at the default — never clobber
                // a value the user deliberately entered.
                if (!cur || cur === 1024 || cur === 512) {
                    imgszEl.value = info.imgsz;
                }
            }
        }
    } catch (e) {
        /* non-fatal — user can still set imgsz manually */
    }
}

// ── Step 6 device / DDP selector ───────────────────────────────────────────

function onStep6DeviceChange() {
    const sel = document.getElementById('s6-device-preset');
    const custom = document.getElementById('s6-device-custom');
    if (!sel || !custom) return;
    custom.style.display = (sel.value === '__custom__') ? '' : 'none';
}

function collectStep6Device() {
    const sel = document.getElementById('s6-device-preset');
    const custom = document.getElementById('s6-device-custom');
    if (!sel) return null;
    if (sel.value === '__custom__') {
        const v = (custom && custom.value || '').trim();
        return v || null;
    }
    return sel.value || null;
}

function applyStep6Device(saved) {
    const sel = document.getElementById('s6-device-preset');
    const custom = document.getElementById('s6-device-custom');
    if (!sel) return;
    const presets = new Set(['0', '1', '0,1', 'cpu']);
    // DDP across both GPUs is the default for new / unseeded projects.
    if (!saved) { sel.value = '0,1'; if (custom) custom.value = ''; }
    else if (presets.has(saved)) { sel.value = saved; if (custom) custom.value = ''; }
    else { sel.value = '__custom__'; if (custom) custom.value = saved; }
    onStep6DeviceChange();
}

// ── Step 6 "Continue from a previous run" model options (Task 8) ───────────
// Populates the s6-model optgroup with one option per Step 5 run that has
// weights/best.pt, plus a leading "Champion (promoted best)" option when
// {step6_dir}/champion.json exists. Reuses /api/step/6/list_runs (same call
// refreshRunList makes for the s7/s8 pickers) so this never needs its own route.
async function refreshStep6ModelRunOptions() {
    const optgroup = document.getElementById('s6-model-runs-optgroup');
    const modelSel = document.getElementById('s6-model');
    if (!optgroup || !modelSel) return;
    try {
        const data = await api('/api/step/6/list_runs');
        const runs = (data.runs || []).filter(r => r.has_best);
        const prevValue = modelSel.value;
        optgroup.innerHTML = '';
        if (data.champion_run_dir) {
            const champBest = data.champion_run_dir.replace(/\/$/, '') + '/weights/best.pt';
            const opt = document.createElement('option');
            opt.value = champBest;
            opt.textContent = 'Champion (promoted best)';
            optgroup.appendChild(opt);
        }
        for (const r of runs) {
            const best = r.path.replace(/\/$/, '') + '/weights/best.pt';
            const opt = document.createElement('option');
            opt.value = best;
            let label = r.name;
            if (r.model || r.epochs) label += `: ${r.model || '?'} · ${r.epochs || '?'} epochs`;
            if (r.is_champion) label += ' [champion]';
            opt.textContent = label;
            optgroup.appendChild(opt);
        }
        // Restore the previous selection if it still resolves to a valid option
        // (covers the case where model_path was applied before runs loaded).
        if (prevValue) {
            const stillValid = Array.from(modelSel.options).some(o => o.value === prevValue);
            if (stillValid) modelSel.value = prevValue;
        }
    } catch (e) {
        console.warn('refreshStep6ModelRunOptions failed:', e);
    }
}

// Preset applier for model_path: if the path isn't already an option (the
// run-options optgroup hasn't loaded, or this is a champion path not yet
// resolved into the list), insert a one-off option so the select always shows
// something meaningful rather than silently falling back to the first entry.
function applyStep6ModelPath(v) {
    if (!v) return;
    const sel = document.getElementById('s6-model');
    if (!sel) return;
    const exists = Array.from(sel.options).some(o => o.value === v);
    if (!exists) {
        const optgroup = document.getElementById('s6-model-runs-optgroup');
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v.split('/').slice(-3).join('/');
        (optgroup || sel).appendChild(opt);
    }
    sel.value = v;
}

// Apply one of the two built-in Step 6 presets (finetune / retrain) straight
// from the server (/api/step/6/presets/<id>), same route the after-export
// card on the Refine panel uses via gotoStep6Preset(). Reuses the same
// STEP6_PRESET_APPLIERS map as the file-based preset importer.
async function applyStep6BuiltinPreset(presetId) {
    const desc = document.getElementById('s6-preset-desc');
    try {
        // Model-run options must be loaded first so a champion/previous-run
        // model_path resolves to a real <option> rather than a synthesized one.
        await refreshStep6ModelRunOptions();
        const data = await api(`/api/step/6/presets/${encodeURIComponent(presetId)}`);
        if (data.error) {
            alert(`Preset error: ${data.error}`);
            return;
        }
        const params = data.params || {};
        let applied = 0, skipped = [];
        for (const [key, val] of Object.entries(params)) {
            const fn = STEP6_PRESET_APPLIERS[key];
            if (fn) { fn(val); applied++; }
            else { skipped.push(key); }
        }
        if (desc) {
            const msg = `Applied "${data.name}": ${applied} field(s) updated` +
                (skipped.length ? ` (ignored: ${skipped.join(', ')})` : '');
            desc.textContent = msg;
            desc.style.color = 'var(--magenta-light)';
            setTimeout(() => { desc.style.color = 'var(--text-dim)'; }, 4000);
        }
    } catch (e) {
        alert(`Failed to apply preset: ${e}`);
    }
}

// ── Step 6 class-inclusion picker ──────────────────────────────────────────

async function refreshStep6Classes() {
    const host = document.getElementById('s6-classes-list');
    if (!host) return;
    host.innerHTML = '<span style="color:var(--text-dim)">Loading classes from Step 4 output...</span>';
    try {
        const data = await api('/api/step/6/list_classes');
        if (data.error) {
            host.innerHTML = `<span style="color:var(--magenta-light)">Error: ${data.error}</span>`;
            return;
        }
        renderStep6Classes(data.classes || [], data.include_classes);
    } catch (e) {
        host.innerHTML = `<span style="color:var(--magenta-light)">Error loading classes: ${e}</span>`;
    }
}

function renderStep6Classes(classes, saved) {
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
        return;
    }
    // saved === null|undefined -> include every class (default).
    // saved is an array of IDs -> include only those.
    const savedSet = Array.isArray(saved) ? new Set(saved.map(Number)) : null;
    const rows = filtered.map(c => {
        const nm = String(c.name).trim();
        const checked = (savedSet === null || savedSet.has(c.id)) ? 'checked' : '';
        const countTxt = `${c.instance_count || 0} instances · ${c.image_count || 0} images`;
        return `<label style="display:flex;align-items:center;gap:8px;padding:3px 0;cursor:pointer">
            <input type="checkbox" class="s6-class-cb" data-class-id="${c.id}" data-class-name="${nm}" ${checked}>
            <span style="min-width:2.5em;color:var(--text-dim);font-variant-numeric:tabular-nums">#${c.id}</span>
            <span style="flex:1;font-weight:500">${nm}</span>
            <span style="color:var(--text-dim);font-size:11px">${countTxt}</span>
        </label>`;
    }).join('');
    host.innerHTML = rows;
}

function selectAllStep6Classes(on) {
    document.querySelectorAll('.s6-class-cb').forEach(cb => { cb.checked = !!on; });
}

// Returns null (meaning "all classes") if the picker hasn't been loaded yet or
// every available class is checked. Returns [] if the user has unchecked
// everything (we treat that as "train nothing" — caller will error out).
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
                opt.textContent = '(no training runs yet — finish Step 5 first)';
                sel.appendChild(opt);
                return;
            }
            for (const r of runs) {
                const opt = document.createElement('option');
                opt.value = r.path;
                let label = r.name;
                if (r.model || r.epochs) label += ` — ${r.model || '?'} · ${r.epochs || '?'} epochs`;
                if (!r.has_best && r.has_last) label += ' [last.pt only]';
                if (r.is_champion) label += ' [champion]';
                opt.textContent = label;
                sel.appendChild(opt);
            }
            if (prev) sel.value = prev;
            // Pre-populate from project state if empty
            const savedS7 = state && state.steps['7'] && state.steps['7'].config && state.steps['7'].config.run_dir;
            const savedS8 = state && state.steps['8'] && state.steps['8'].config && state.steps['8'].config.run_dir;
            if (selId === 's7-run-dir' && savedS7) sel.value = savedS7;
            // Step 8 (Task 9): prefer the promoted champion over a stale saved
            // config value, so promoting a new champion actually changes what
            // Inference defaults to next time the panel loads.
            if (selId === 's8-run-dir') {
                if (championRunDir) sel.value = championRunDir;
                else if (savedS8) sel.value = savedS8;
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
    fail: 'Mask mAP50-95 regressed more than the tolerance (0.005) versus the baseline; do not promote.',
    flag: 'Recall or a per-class AP regressed more than tolerance versus the baseline; review before promoting.',
    unpinned: 'This run\'s split was not a frozen transect holdout, so the comparison is informational only, not a clean gate.',
};

function _roundsGateBadge(value) {
    if (!value) return '';
    const base = String(value).split(':')[0];
    const cls = base === 'pass' ? 'badge-success'
        : base === 'fail' ? 'badge-error'
        : base === 'flag' ? 'badge-warning'
        : 'badge-info';
    const tip = _ROUNDS_GATE_TOOLTIPS[base] || '';
    return `<span class="badge ${cls}" style="margin-right:4px" title="${escAttr(tip)}">${escHtmlNav(value)}</span>`;
}

function _roundsDelta(cur, prevMap) {
    if (cur == null || prevMap == null) return '&mdash;';
    const d = cur - prevMap;
    const sign = d >= 0 ? '+' : '';
    return `${sign}${d.toFixed(3)}`;
}

async function refreshStep7Rounds() {
    const tbody = document.getElementById('s7-rounds-tbody');
    const note = document.getElementById('s7-rounds-champion-note');
    if (!tbody) return;
    try {
        const data = await api('/api/step/7/rounds');
        const rows = data.rounds || [];
        const champion = data.champion || null;
        if (note) {
            note.textContent = champion
                ? `Champion: ${champion.run_dir ? champion.run_dir.split('/').pop() : '(unknown)'} (mAP50-95 ${champion.map50_95_M != null ? champion.map50_95_M.toFixed(3) : '?'})`
                : 'No champion promoted yet.';
        }
        if (!rows.length) {
            tbody.innerHTML = '<tr><td colspan="8" style="padding:10px 8px;color:var(--text-dim)">No evaluated rounds yet. Run Evaluation above to add one.</td></tr>';
            return;
        }
        let prevMap = null;
        tbody.innerHTML = rows.map(r => {
            const delta = _roundsDelta(r.map50_95_M, prevMap);
            prevMap = r.map50_95_M != null ? r.map50_95_M : prevMap;
            const isChamp = champion && champion.run_dir === r.run_dir;
            const gates = _roundsGateBadge(r.gate_map) + _roundsGateBadge(r.gate_class) + _roundsGateBadge(r.gate_recall);
            const promoteBtn = isChamp
                ? '<span class="badge badge-info" title="This run is the current champion.">champion</span>'
                : `<button class="btn btn-outline btn-sm" onclick='confirmPromoteRun(${JSON.stringify(r.run_dir)})' title="Make this run the champion. It becomes the default weights for Inference and for fine-tuning.">Promote</button>`;
            return `<tr>
                <td style="padding:6px 8px;border-bottom:1px solid var(--border)">${escHtmlNav(r.round != null ? r.round : '')}</td>
                <td style="padding:6px 8px;border-bottom:1px solid var(--border)">${escHtmlNav(r.run_name || '')}</td>
                <td style="padding:6px 8px;border-bottom:1px solid var(--border)">${escHtmlNav(r.base_model || '')}</td>
                <td style="padding:6px 8px;border-bottom:1px solid var(--border);text-align:right">${r.map50_95_M != null ? r.map50_95_M.toFixed(3) : '&mdash;'}</td>
                <td style="padding:6px 8px;border-bottom:1px solid var(--border);text-align:right">${delta}</td>
                <td style="padding:6px 8px;border-bottom:1px solid var(--border);text-align:right">${r.recall_M != null ? r.recall_M.toFixed(3) : '&mdash;'}</td>
                <td style="padding:6px 8px;border-bottom:1px solid var(--border)">${gates}</td>
                <td style="padding:6px 8px;border-bottom:1px solid var(--border)">${promoteBtn}</td>
            </tr>`;
        }).join('');
    } catch (e) {
        console.warn('refreshStep7Rounds failed:', e);
        tbody.innerHTML = `<tr><td colspan="8" style="padding:10px 8px;color:var(--error)">Error loading rounds: ${escHtmlNav(String(e))}</td></tr>`;
    }
}

// Reuses the standard reset-confirm overlay's modal chrome (styled confirm,
// not window.confirm()) for the Promote action's "are you sure" step.
function confirmPromoteRun(runDir) {
    const titleEl = document.getElementById('confirm-title');
    if (titleEl) titleEl.textContent = 'Promote to champion?';
    const msgEl = document.getElementById('confirm-msg');
    if (msgEl) {
        msgEl.textContent = 'Make this run the champion? It becomes the default weights for Inference and for fine-tuning.';
    }
    const okBtn = document.getElementById('confirm-ok-btn');
    if (okBtn) { okBtn.textContent = 'Confirm Promote'; okBtn.onclick = () => { closeConfirmModal(); promoteRun(runDir); }; }
    const ov = document.getElementById('confirm-overlay');
    if (ov) ov.classList.add('visible');
}

async function promoteRun(runDir) {
    try {
        const data = await post('/api/step/7/promote', { run_dir: runDir });
        if (data.error) { alert('Promote failed: ' + data.error); return; }
        await refreshStep7Rounds();
        await refreshRunList();
        await refreshStep6ModelRunOptions();
    } catch (e) {
        alert('Promote failed: ' + e);
    }
}

// Minimal markdown -> HTML. Handles headers, bold, italic, inline code, lists,
// tables (pipe syntax), and horizontal rules. Intentionally small — we control
// the upstream markdown, so we don't need a full parser.
function mdToHtml(md) {
    const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const inl = s => esc(s)
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>');
    const lines = md.split('\n');
    const out = [];
    let inList = false;
    let inTable = false;
    let tableRows = [];
    const flushList = () => { if (inList) { out.push('</ul>'); inList = false; } };
    const flushTable = () => {
        if (!inTable) return;
        if (tableRows.length < 2) { inTable = false; tableRows = []; return; }
        out.push('<table style="border-collapse:collapse;margin:6px 0;font-size:12px">');
        tableRows.forEach((row, idx) => {
            const cells = row.slice(1, row.length - 1).split('|').map(c => c.trim());
            if (idx === 1) return; // separator row
            const tag = idx === 0 ? 'th' : 'td';
            out.push('<tr>' + cells.map(c =>
                `<${tag} style="padding:4px 8px;border:1px solid #2a2a4e">${inl(c)}</${tag}>`).join('') + '</tr>');
        });
        out.push('</table>');
        inTable = false; tableRows = [];
    };
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmed = line.trim();
        if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
            flushList();
            inTable = true;
            tableRows.push(trimmed);
            continue;
        } else if (inTable) { flushTable(); }
        if (/^#{1,6}\s/.test(trimmed)) {
            flushList();
            const lvl = trimmed.match(/^#+/)[0].length;
            out.push(`<h${lvl} style="margin:10px 0 4px;color:var(--magenta-light)">${inl(trimmed.replace(/^#+\s*/, ''))}</h${lvl}>`);
        } else if (trimmed.startsWith('- ')) {
            if (!inList) { out.push('<ul style="margin:4px 0 4px 20px">'); inList = true; }
            out.push('<li>' + inl(trimmed.slice(2)) + '</li>');
        } else if (trimmed === '' ) {
            flushList();
            out.push('<div style="height:6px"></div>');
        } else if (trimmed.startsWith('---')) {
            flushList();
            out.push('<hr style="border:none;border-top:1px solid #333;margin:8px 0">');
        } else {
            flushList();
            out.push('<p style="margin:4px 0">' + inl(trimmed) + '</p>');
        }
    }
    flushList(); flushTable();
    return out.join('\n');
}

async function loadStep7Report() {
    try {
        const data = await api('/api/step/7/report');
        if (!data.ready) {
            document.getElementById('s7-report-block').style.display = 'none';
            return;
        }
        document.getElementById('s7-report-block').style.display = '';
        document.getElementById('s7-report').innerHTML = mdToHtml(data.markdown || '');
        const dl = document.getElementById('s7-pdf-download');
        if (dl) dl.style.display = data.pdf_exists ? '' : 'none';
    } catch (e) {
        console.warn('loadStep7Report failed:', e);
    }
}

async function copyStep7Pdf() {
    const dst = prompt('Copy PDF to which folder?', '');
    if (!dst) return;
    const data = await post('/api/step/7/copy_pdf', { dst_dir: dst });
    if (data.error) { alert('Copy failed: ' + data.error); return; }
    alert('PDF copied to ' + data.dst);
}

function toggleStep8SourceFields() {
    const srcRadio = document.querySelector('input[name="s8-src"]:checked');
    const src = srcRadio ? srcRadio.value : 'unused';
    document.getElementById('s8-dir-wrap').style.display = (src === 'dir') ? '' : 'none';
    document.getElementById('s8-sample-wrap').style.display = (src === 'dir' || src === 'full') ? 'none' : 'flex';
}
document.addEventListener('change', (e) => {
    if (e.target && e.target.name === 's8-src') toggleStep8SourceFields();
});

// Track whether the user has typed into the Step 6 run-name field so that
// subsequent step-6 panel re-entries don't stomp their input. Blank stays
// blank until the user types a first character.
document.addEventListener('input', (e) => {
    if (e.target && e.target.id === 's6-run-name') {
        if (e.target.value && e.target.value.length > 0) {
            e.target.dataset.userEdited = '1';
        } else {
            delete e.target.dataset.userEdited;
        }
    }
});

async function refreshStep8Gallery() {
    try {
        const data = await api('/api/step/8/manifest');
        if (!data.ready) {
            document.getElementById('s8-gallery-block').style.display = 'none';
            return;
        }
        const block = document.getElementById('s8-gallery-block');
        const gallery = document.getElementById('s8-gallery');
        const counts = document.getElementById('s8-counts');
        const blind = document.getElementById('s8-blind-spots');
        block.style.display = '';
        const items = data.items || [];
        counts.textContent = `${items.length} images, ` +
            `${data.total_predictions || 0} predictions ` +
            `(${data.with_any || 0} with ≥1 detection, ${(items.length - (data.with_any || 0))} with none)`;

        if (data.blind_spots) {
            blind.style.display = '';
            blind.innerHTML = '<strong style="color:var(--magenta-light)">Blind-spot signal:</strong> ' + data.blind_spots;
        } else {
            blind.style.display = 'none';
        }

        gallery.innerHTML = '';
        items.forEach((it, idx) => {
            const tile = document.createElement('div');
            tile.style.cssText = 'background:var(--surface-light);border-radius:3px;padding:6px;font-size:11px';
            const imgSrc = `/api/step/8/image?path=${encodeURIComponent(it.overlay || it.raw)}`;
            tile.innerHTML = `
                <img src="${imgSrc}" style="width:100%;height:150px;object-fit:cover;border-radius:2px;background:#000">
                <div style="margin-top:4px;font-family:monospace;color:#4ecca3;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${it.filename || ''}</div>
                <div style="color:var(--text-dim);font-size:10px">${it.n_detections || 0} det, max conf ${it.max_conf ? it.max_conf.toFixed(2) : '-'}</div>
                <div style="display:flex;gap:4px;margin-top:4px">
                    <button class="btn btn-sm" style="flex:1;padding:2px;background:#1a5c32" onclick="verifyStep8('${encodeURIComponent(it.filename)}','yes', this)">YES</button>
                    <button class="btn btn-sm" style="flex:1;padding:2px;background:#5a1a2a" onclick="verifyStep8('${encodeURIComponent(it.filename)}','no', this)">NO</button>
                    <button class="btn btn-sm" style="flex:1;padding:2px" onclick="verifyStep8('${encodeURIComponent(it.filename)}','skip', this)">?</button>
                </div>
            `;
            gallery.appendChild(tile);
        });
    } catch (e) {
        console.warn('refreshStep8Gallery failed:', e);
    }
}

async function verifyStep8(filenameEnc, verdict, btn) {
    const filename = decodeURIComponent(filenameEnc);
    try {
        await post('/api/step/8/verify', { filename, verdict });
        if (btn) { btn.style.opacity = 0.4; btn.disabled = true; }
    } catch (e) {
        alert('Verify failed: ' + e.message);
    }
}

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
// Excluded labels are ALWAYS hidden — there is no hide-excluded toggle anymore.
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
        });
    });

    syncStep3SpeciesField();
    repaintStep3Chips();
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
// selected codes — the union of checked visible boxes AND any codes already in
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

// Fire up the background SAM3 mini-poll on page load.
setTimeout(startSam3MiniPoll, 500);

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

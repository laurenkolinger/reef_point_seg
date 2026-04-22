/* Pipeline Orchestrator — SPA Logic */

let state = null;       // Current project state
let activeStep = 1;     // Currently viewed step panel
let pollTimers = {};    // step -> interval ID for polling
let stepStartTimes = {};// step -> Date when started
let elapsedTimers = {}; // step -> interval ID for elapsed display
let serviceWindows = {};// step -> window reference

// ── API helpers ─────────────────────────────────────────────────────────────

async function api(url, opts = {}) {
    const resp = await fetch(url, {
        headers: { 'Content-Type': 'application/json' },
        ...opts,
    });
    return resp.json();
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

async function createProject() {
    const name = document.getElementById('new-name').value.trim();
    const base = document.getElementById('new-basedir').value.trim();
    if (!name || !base) return alert('Please fill in project name and base directory.');

    const data = await post('/api/project/create', { name, base_dir: base });
    if (data.error) return alert('Error: ' + data.error);
    state = data.state;
    enterApp();
}

async function openProject() {
    const dir = document.getElementById('open-dir').value.trim();
    if (!dir) return alert('Please enter a project directory path.');

    const data = await post('/api/project/open', { project_dir: dir });
    if (data.error) return alert('Error: ' + data.error);
    state = data.state;
    enterApp();
}

function showSetup() {
    Object.values(pollTimers).forEach(clearInterval);
    Object.values(elapsedTimers).forEach(clearInterval);
    pollTimers = {};
    elapsedTimers = {};
    document.getElementById('setup-screen').style.display = '';
    document.getElementById('app-screen').style.display = 'none';
    document.getElementById('topbar-actions').style.display = 'none';
    document.getElementById('topbar-project-name').textContent = '';
}

async function quitProject() {
    if (!confirm('Quit project? All running services will be stopped.')) return;
    // Close any open service windows
    Object.values(serviceWindows).forEach(w => { try { w.close(); } catch(e) {} });
    serviceWindows = {};
    await post('/api/project/quit', {});
    state = null;
    showSetup();
}

function resumeLoaded() {
    if (!state) return;
    document.getElementById('resume-card').style.display = 'none';
    enterApp();
}

async function discardLoaded() {
    if (!confirm('Discard loaded project and start fresh? Any running sub-apps will be stopped.')) return;
    await post('/api/project/quit', {});
    state = null;
    document.getElementById('resume-card').style.display = 'none';
    document.getElementById('resume-name').textContent = '';
}

function enterApp() {
    document.getElementById('setup-screen').style.display = 'none';
    document.getElementById('app-screen').style.display = 'flex';
    document.getElementById('topbar-actions').style.display = 'flex';
    document.getElementById('topbar-project-name').textContent =
        state.name + '  (' + state.id + ')';

    populateConfigs();
    updateSidebar();
    switchStep(state.current_step || 1);
}

// ── Sidebar ─────────────────────────────────────────────────────────────────

function updateSidebar() {
    if (!state) return;
    for (let s = 1; s <= 6; s++) {
        const li = document.querySelector(`.step-nav li[data-step="${s}"]`);
        const st = state.steps[s];
        const status = st ? st.status : 'locked';

        li.className = '';
        li.classList.add(status);
        if (s === activeStep) li.classList.add('active');

        const circle = li.querySelector('.step-circle');
        if (status === 'completed') {
            circle.textContent = '\u2713';
        } else {
            circle.textContent = s;
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
        if (step === 7) loadStep7Report();
        if (step === 8) refreshStep8Gallery();
    })();
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
    setVal('s3-species', s3.target_species || 'OFRA, PA, OA, OFAV, AL, MC, AA');
    setVal('s3-target', s3.target_instances || 1000);
    setVal('s3-min-year', s3.min_year || 2014);
    setVal('s3-max-year', s3.max_year || 2025);
    setVal('s3-category', s3.category_filter || 'Coral');
    if (s3.skip_image_check) document.getElementById('s3-skip-check').checked = true;

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
    s6set('s6-run-name', s6.run_name, '');
    s6set('s6-epochs', s6.epochs, 500);
    s6set('s6-imgsz', s6.imgsz, 512);
    s6set('s6-batch', s6.batch, -1);
    s6set('s6-patience', s6.patience, 50);
    s6set('s6-valid', s6.valid_ratio, 0.2);
    s6set('s6-test', s6.test_ratio, 0.1);
    s6set('s6-min-samples', s6.min_samples, 10);
    if (s6.model) { const sel = document.getElementById('s6-model'); if (sel) sel.value = s6.model; }
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
    refreshStep6Presets();
    // Populate step-7 eval-preset list too.
    refreshStep7Presets();
    // Populate step-6 class picker if step 5 has run.
    if (state.steps['5'] && state.steps['5'].status === 'completed') {
        refreshStep6Classes();
    }

    // If an evaluation has already been run, auto-load the report.
    if (state.steps['7'] && state.steps['7'].outputs && state.steps['7'].outputs.report_md) {
        loadStep7Report();
    }
    if (state.steps['8'] && state.steps['8'].outputs && state.steps['8'].outputs.manifest) {
        refreshStep8Gallery();
    }

    // Restore running services
    for (let s = 1; s <= 6; s++) {
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
        cfg.target_species = document.getElementById('s3-species').value.trim();
        cfg.target_instances = parseInt(document.getElementById('s3-target').value) || 1000;
        cfg.min_year = parseInt(document.getElementById('s3-min-year').value) || 2014;
        cfg.max_year = parseInt(document.getElementById('s3-max-year').value) || 2025;
        cfg.category_filter = document.getElementById('s3-category').value.trim() || 'Coral';
        cfg.skip_image_check = document.getElementById('s3-skip-check').checked;
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
        cfg.model = $('s6-model').value;
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
    await put(`/api/project/config/${step}`, cfg);
    Object.assign(state.steps[step].config, cfg);

    const data = await post(`/api/step/${step}/run`, {});
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
    const openBtn = document.getElementById('s5-open-window');

    if (phaseEl) phaseEl.textContent = s.phase || 'idle';
    if (msgEl) msgEl.textContent = s.message || '';
    const processed = s.processed || 0, total = s.total || 0;
    const pct = total > 0 ? Math.round((processed / total) * 100) : 0;
    if (fillEl) fillEl.style.width = pct + '%';
    if (countsEl) countsEl.textContent = `${processed} / ${total}` + (total ? ` (${pct}%)` : '');

    if (s.error) {
        if (errEl) { errEl.textContent = 'Error: ' + s.error; errEl.style.display = 'block'; }
    } else if (errEl) {
        errEl.style.display = 'none';
    }

    // Gate the Review UI button: only enable once at least one frame has been
    // segmented, OR the queue had nothing to process (already done).
    if (openBtn) {
        const readyForReview = processed >= 1 || s.phase === 'review_ready';
        openBtn.disabled = !readyForReview;
        if (s.phase === 'review_ready') {
            openBtn.textContent = 'Open Review UI (segmentation complete)';
        } else if (readyForReview) {
            openBtn.textContent = `Open Review UI (${processed} frames ready — SAM3 still running)`;
        } else {
            openBtn.textContent = 'Open Review UI (waiting for first segmentation)';
        }
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

        // Update log
        const logData = await api(`/api/step/${step}/log?offset=${logOffset}`);
        if (logData.lines && logData.lines.length > 0) {
            const logEl = document.getElementById(`s${step}-log`);
            if (logEl) {
                logEl.style.display = 'block';
                for (const line of logData.lines) {
                    const div = document.createElement('div');
                    div.className = 'log-line';
                    div.textContent = line;
                    logEl.appendChild(div);
                }
                logEl.scrollTop = logEl.scrollHeight;
            }
            logOffset = logData.offset;

            // Parse log for progress hints
            parseProgressFromLog(step, logData.lines);
        }

        // Flask stage health
        if (status.port) {
            const healthDot = document.getElementById(`s${step}-health`);
            const healthText = document.getElementById(`s${step}-health-text`);

            if (status.healthy) {
                if (healthDot) healthDot.className = 'health-dot healthy';
                if (healthText) healthText.textContent = 'Service running on port ' + status.port;
                // Enable the open window button — except for step 5, which is
                // gated on SAM3 progress by updateSam3Panel().
                if (step !== 5) {
                    const btn = document.getElementById(`s${step}-open-window`);
                    if (btn) btn.disabled = false;
                }
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

// ── Open service in new window ──────────────────────────────────────────────

function openServiceWindow(step) {
    const port = state.steps[step] && state.steps[step].config && state.steps[step].config.port;
    if (!port) return alert('Service port not set. Launch the service first.');

    const url = `http://localhost:${port}`;
    const name = `tcrmp_step${step}`;

    // Try to open a proper new window (not a tab)
    const w = window.open(url, name, 'width=1400,height=900,menubar=no,toolbar=yes,location=yes,status=yes,scrollbars=yes,resizable=yes');
    if (w) {
        serviceWindows[step] = w;
        w.focus();
    } else {
        // Popup blocked, fall back to new tab
        window.open(url, '_blank');
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
    // Auto-advance to next step
    if (step < 5) {
        setTimeout(() => switchStep(step + 1), 500);
    }
}

async function resetStep(step) {
    if (!confirm(`Reset step ${step}? This will clear all outputs for this step and lock subsequent steps.`)) return;
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
    hideProgress(step);
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
    html += '<table class="remap-table"><tr><th>Old Code</th><th>New Code</th><th>Species Name</th><th>Action</th><th>Points Affected</th></tr>';
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

    // Auto-advance to step 3
    setTimeout(() => switchStep(3), 800);
}

// ── Initialize ──────────────────────────────────────────────────────────────

// ── Step 7 / 8 helpers (training-run picker, report viewer, gallery) ───────

// ── Step 6 preset loader (bulk-fill form from YAML recipe) ─────────────────

// Mapping: preset param key -> function that writes the value into the form.
// Kept separate from populateConfigs() because presets intentionally omit
// keys the user shouldn't inherit from a preset (run_name, include_classes).
const STEP6_PRESET_APPLIERS = {
    model:          v => { const el = document.getElementById('s6-model'); if (el) el.value = v; },
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

async function refreshStep6Presets() {
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
            sel.appendChild(opt);
        }
        if (prev) sel.value = prev;
        sel.onchange = () => {
            const opt = sel.options[sel.selectedIndex];
            if (desc) desc.textContent = opt ? (opt.dataset.description || '') : '';
        };
        sel.onchange();
    } catch (e) {
        console.warn('refreshStep6Presets failed:', e);
        if (desc) desc.textContent = `Error loading presets: ${e}`;
    }
}

async function importStep6Preset() {
    const sel = document.getElementById('s6-preset-select');
    const desc = document.getElementById('s6-preset-desc');
    if (!sel || !sel.value) {
        alert('Pick a preset first.');
        return;
    }
    try {
        const data = await api(`/api/step/6/presets/${encodeURIComponent(sel.value)}`);
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
    if (!saved) { sel.value = '0'; if (custom) custom.value = ''; }
    else if (presets.has(saved)) { sel.value = saved; if (custom) custom.value = ''; }
    else { sel.value = '__custom__'; if (custom) custom.value = saved; }
    onStep6DeviceChange();
}

// ── Step 6 class-inclusion picker ──────────────────────────────────────────

async function refreshStep6Classes() {
    const host = document.getElementById('s6-classes-list');
    if (!host) return;
    host.innerHTML = '<span style="color:var(--text-dim)">Loading classes from Step 5 output...</span>';
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
    if (!classes.length) {
        host.innerHTML = '<span style="color:var(--text-dim)">No classes found in Step 5 output. Run Step 5 first.</span>';
        return;
    }
    // saved === null|undefined -> include every class (default).
    // saved is an array of IDs -> include only those.
    const savedSet = Array.isArray(saved) ? new Set(saved.map(Number)) : null;
    const rows = classes.map(c => {
        const nm = c.name ? c.name : '(unnamed)';
        const checked = (savedSet === null || savedSet.has(c.id)) ? 'checked' : '';
        const countTxt = `${c.instance_count} instances · ${c.image_count} images`;
        return `<label style="display:flex;align-items:center;gap:8px;padding:3px 0;cursor:pointer">
            <input type="checkbox" class="s6-class-cb" data-class-id="${c.id}" ${checked}>
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
function collectStep6IncludedClasses() {
    const cbs = document.querySelectorAll('.s6-class-cb');
    if (!cbs.length) return null;
    const total = cbs.length;
    const selected = [];
    cbs.forEach(cb => { if (cb.checked) selected.push(parseInt(cb.dataset.classId)); });
    if (selected.length === total) return null;
    return selected;
}

async function refreshRunList() {
    try {
        const data = await api('/api/step/6/list_runs');
        const runs = data.runs || [];
        const populate = (selId) => {
            const sel = document.getElementById(selId);
            if (!sel) return;
            const prev = sel.value;
            sel.innerHTML = '';
            if (!runs.length) {
                const opt = document.createElement('option');
                opt.value = '';
                opt.textContent = '(no training runs yet — finish Step 6 first)';
                sel.appendChild(opt);
                return;
            }
            for (const r of runs) {
                const opt = document.createElement('option');
                opt.value = r.path;
                let label = r.name;
                if (r.model || r.epochs) label += ` — ${r.model || '?'} · ${r.epochs || '?'} epochs`;
                if (!r.has_best && r.has_last) label += ' [last.pt only]';
                opt.textContent = label;
                sel.appendChild(opt);
            }
            if (prev) sel.value = prev;
            // Pre-populate from project state if empty
            const savedS7 = state && state.steps['7'] && state.steps['7'].config && state.steps['7'].config.run_dir;
            const savedS8 = state && state.steps['8'] && state.steps['8'].config && state.steps['8'].config.run_dir;
            if (selId === 's7-run-dir' && savedS7) sel.value = savedS7;
            if (selId === 's8-run-dir' && savedS8) sel.value = savedS8;
        };
        populate('s7-run-dir');
        populate('s8-run-dir');
    } catch (e) {
        console.warn('refreshRunList failed:', e);
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

// Fire up the background SAM3 mini-poll on page load.
setTimeout(startSam3MiniPoll, 500);

(async function init() {
    // Always show the landing screen on page load. If the server still has a
    // project in memory (e.g. from a prior browser session), offer a Resume
    // banner rather than silently entering the app.
    const data = await api('/api/project/state');
    if (data.loaded && data.state) {
        state = data.state;
        const nameEl = document.getElementById('resume-name');
        if (nameEl) nameEl.textContent = state.name + '  (' + state.id + ')';
        const card = document.getElementById('resume-card');
        if (card) card.style.display = '';
    }
})();

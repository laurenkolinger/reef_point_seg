# Wiring the VEGA guided tour into each page

Written 2026-09-17 by LO for task T15, updated the same day for version 2 of the tour (no forward
button, a glowing anchor, every step completing on a real action) and again for version 3 (no step
on a page the tour is not on, the tour's place carried across windows, plain words first). The tour is three static files
served by the pipeline orchestrator on port 5050 from `scripts/pipeline_orchestrator/static/`:
`vega_tour.css` (the stylesheet), `vega_tour_content.js` (the step list as data) and `vega_tour.js`
(the engine). This note lists the exact lines each page must carry and the ids each page owes the
tour. The controller adds them; the tour task does not edit templates.

Wired so far: the pipeline window, the Coverage Matrix, the Label Manager, the annotator and the
desktop launcher. Not wired: the Bounding Boxes page, which Lauren has protected; its three lines
are recorded in the TODO log as T20260917-145750-3 and nothing on that page is edited until she
says so.

**No step lives on an unwired page (2026-09-17, LO).** Version 2 of the content put two steps on the
Bounding Boxes page, and a student who pressed the link arrived to find the tour gone and step 5
unreachable. Version 3 has no step on that page at all: one step in the pipeline window says where a
folder of boxed pictures is prepared and that the page opens in its own tab, what such a folder
holds is taught on the Train panel that names it, and the difference between a box and an outline is
one sentence on the step that already talks about tracing. Two tests read the templates listed in
this note for the tour's own link and script tags and fail on any step whose surface is not among
them: `test_every_surface_is_a_surface_the_tour_is_wired_into` in
`tests/test_vega_tour_content.py` and `test_no_step_sends_a_student_to_a_page_the_tour_is_not_on` in
`tests/test_vega_tour.js`. Adding the three lines to the Bounding Boxes page is what makes a step on
it legal again; nothing else needs to change.

Order matters on every page: the stylesheet in the head, then the content script, then the engine,
both scripts at the end of the body so the page's own ids exist when the engine mounts. The engine
puts a "Tutorial" button at the bottom right of every wired page and offers to resume when the
tour was open the last time that window closed. Nothing else on the page changes.

Cache-bust pins follow each page's own convention (the orchestrator's `asset_version`, the matrix's
`v`, and the hand-written `20260917-tour` on the labels page, the annotator and the launcher). The
Bounding Boxes page takes the same hand-written pin when its lines are added.

## The pipeline window: `scripts/pipeline_orchestrator/templates/index.html`

In the head, after line 75 (`orchestrator_gallery.css`):

    <link rel="stylesheet" href="/static/vega_tour.css?v={{ asset_version }}">

At the end of the body, after line 2182 (`orchestrator_gallery.js`):

    <script src="/static/vega_tour_content.js?v={{ asset_version }}"></script>
    <script src="/static/vega_tour.js?v={{ asset_version }}"></script>

Anchors on this page resolve to ids and classes that already exist: the rail tiles
(`li[data-step="6"]`, `"7"`, `"8"`), the Tools links (`a[href="/matrix"]`, `/labels`, `/annotate`,
`/boxes`), the Train settings (`#s6-source`, `#s6-model`, `#s6-epochs`, `#s6-imgsz`, `#s6-run-name`,
`#s6-run-btn`, `#s6-progress`), the runs card glyph (`#s6-runs-card .vega-runs-tip`), the Evaluate
controls (`#s7-run-dir`, `#s7-split`, `#s7-run-btn`, `#s7-pdf-export-dir`) and the two glyphs that
explain the numbers (`#s7-report-card .tip`, `#s7-rounds-block .tip`), the Inference controls
(`#s8-run-dir`, `input[name="s8-src"][value="dir"]`, `input[name="s8-src"][value="unused"]`,
`#s8-source-dir`, `#s8-conf`, `#s8-run-btn`, `#s8-gallery-refresh`), and the top bar's Open Project
Folder button (`#topbar-actions .btn`). Every one of them is checked against this template by
`tests/test_vega_tour_content.py`, so an id renamed here fails that suite rather than the student.

## The Coverage Matrix: `scripts/_matrix/templates/matrix/index.html`

In the head, after the `vicarius_workflow.css` link (line 14):

    <link rel="stylesheet" href="{{ shared }}/vega_tour.css?v={{ v }}">

At the end of the body, after the page's own script tags:

    <script src="{{ shared }}/vega_tour_content.js?v={{ v }}"></script>
    <script src="{{ shared }}/vega_tour.js?v={{ v }}"></script>

`{{ shared }}` is the orchestrator's app-level static base the blueprint already passes for the
shared stylesheet.

## The Label Manager: `scripts/_labels/templates/labels/index.html`

In the head, after the `vicarius_workflow.css` link (line 20):

    <link rel="stylesheet" href="{{ url_for('static', filename='vega_tour.css') }}?v=20260917-tour">

At the end of the body, after the page's own script tags:

    <script src="{{ url_for('static', filename='vega_tour_content.js') }}?v=20260917-tour"></script>
    <script src="{{ url_for('static', filename='vega_tour.js') }}?v=20260917-tour"></script>

## The Bounding Boxes page: `scripts/_boxes/templates/boxes/index.html`

NOT WIRED. This page is protected (Lauren's approval of 2026-09-16) and nothing on it is edited
until she says so; the request is recorded in the TODO log as T20260917-145750-3. The three lines
here are the only change it needs, and none of its own markup moves. Until they are added, no step
lives on this page. The boxes track's one step about it (`boxes-prepared`) stays in the pipeline
window, glows the tool link `a[href="/boxes"]`, says the page opens in its own tab, and the tour
waits in the pipeline window with the next step ready for when the student comes back. In the head,
after the `boxes.css` link (line 11):

    <link rel="stylesheet" href="{{ url_for('static', filename='vega_tour.css') }}?v=20260917-tour">

At the end of the body, after `boxes.js`:

    <script src="{{ url_for('static', filename='vega_tour_content.js') }}?v=20260917-tour"></script>
    <script src="{{ url_for('static', filename='vega_tour.js') }}?v=20260917-tour"></script>

## The annotator: `scripts/TCRMPclip_combinedAnnotate/src/templates/index.html`

The annotator is its own Flask app on another port, so it links the orchestrator's static folder by
absolute URL. `orchestrator_url` is already passed to the template (`_orchestrator_url()` in
`src/app.py`, from `TCRMP_ORCHESTRATOR_URL`); it is empty for a standalone session started outside
the orchestrator, in which case the lines are skipped and that session has no tour. In the head,
after the fonts link (line 95):

    {% if orchestrator_url %}<link rel="stylesheet" href="{{ orchestrator_url }}/static/vega_tour.css?v=20260917-tour">{% endif %}

At the end of the body, after the page's own scripts:

    {% if orchestrator_url %}
    <script src="{{ orchestrator_url }}/static/vega_tour_content.js?v=20260917-tour"></script>
    <script src="{{ orchestrator_url }}/static/vega_tour.js?v=20260917-tour"></script>
    {% endif %}

The annotator's steps glow `#reviewer-input`, `#quickadd-bar`, `#canvas-wrap`, `#btn-mode-refine`,
`#mask-size`, `#mask-list`, `#btn-next-frame` and `#btn-save-close`. `id="btn-next-frame"` was added
to the "Next (or finish)" button (line 1172) for this, which is what lets the frame step glow the
button a student presses rather than the counter beside it. `#btn-save-close` is inside the export
overlay and therefore absent until the last frame of a batch; while it is absent the callout shows
that step's fallback sentence and keeps watching for the control, and Back and Exit tour still work.

## The desktop launcher: `vicarius_ui_os/templates/module_launcher.html`

The VICARIUS desktop is on port 5090 and the orchestrator on 5050, so the launcher links by
absolute URL from the module's declared launch address. Only the reef_point_seg launcher needs the
tour, so the lines are guarded on the module name. In the `content` block, before the
`<section class="launcher-page"` line:

    {% if module.name == 'reef_point_seg' %}
    <link rel="stylesheet" href="{{ module.ui.launch.url }}/static/vega_tour.css?v=20260917-tour">
    {% endif %}

At the end of the same block, after the page's own inline script:

    {% if module.name == 'reef_point_seg' %}
    <script src="{{ module.ui.launch.url }}/static/vega_tour_content.js?v=20260917-tour"></script>
    <script src="{{ module.ui.launch.url }}/static/vega_tour.js?v=20260917-tour"></script>
    {% endif %}

The files load only while the orchestrator is running; when it is stopped the browser drops the
requests and the launcher page is unchanged.

The launcher owes the tour one hook: `data-tutorial="boxes"` or `data-tutorial="outlines"` on the
card of the matching TUTORIAL project, written by the `tutorialAttr()` helper beside
`reviewerChip()`. Stage 2 of each track declares an anchor chain,
`['[data-tutorial="<track>"]', '#runs-list']`: the glow goes to that one card when the attribute is
there and to the whole project list when it is not, so a student is guided either way.

The helper must be fed the name a student reads, not the folder name. The card renderer calls
`tutorialAttr(run.name)`, and `run.name` is the run folder
(`run_26-09-17_LO_train_tutorial_boxes`), which does not begin with TUTORIAL; the display name is
`run.project_name` ("TUTORIAL boxes (detection)"). Until that call passes
`run.project_name || run.name`, no card carries the attribute and every stage 2 glows the whole
list.

## State across windows

Browsers keep localStorage per origin. The pipeline window, the matrix, the labels page and the
Bounding Boxes page share the 5050 origin and therefore one tour state; the launcher (5090) and the
annotator (5080 and up) each keep their own. Until 2026-09-17 that meant a student who pressed Open
on the launcher arrived in the pipeline window at step 1, with the tour's place left behind on the
other origin. Lauren hit it walking the tour: "I need the tutorial to better link across windows."

The tour now carries its own place, by two carriers with one contract, and needs nothing from any
template:

- **Another origin.** While the tour is open and the step it is waiting on belongs to another page,
  the engine wraps that page's own `window.open` and puts `vega_tour=<track>:<step id>` in the
  address it opens. Nothing else about the call changes (the name, the features and the returned
  window are the page's own), an empty or relative address is left untouched, and the wrap is
  removed when the tour closes. This covers both moves a student makes: the launcher's
  `openOrchWindow()` opening 5050, and `orchestrator.js` opening the annotator on 5080 (its blank
  placeholder popup, opened with an empty address, is left alone).
- **The same origin, a new tab.** A step that declares `handoff` writes `vega.tour.moving` beside
  the place the tour already saves. The tour on the new tab reads it once, clears it, and opens at
  that place instead of asking whether to resume. This covers the Tools links, which are
  `target="_blank"` on 5050.

On arrival the engine reads the parameter, resumes exactly there, and takes the parameter out of the
address bar with `history.replaceState`, so the address a student sees is the page's own. A step id
that belongs to another page still resolves to the first step at or after it that belongs to the
page the student is on, which is what already happened. A student who opens a window some other way
gets exactly what was there before: the saved place, and that same first-step-for-this-page rule.

Not covered: the standalone Annotate Images page (`templates/annotate.html`) carries no tour, so a
student who reaches the annotator through it rather than from the pipeline window arrives with no
place to pick up and presses the Tutorial button, which opens the tour at the annotator's own first
step. Wiring that page is three lines like every other and is not part of this task's ownership.

## Registering the suites

Two new suites for `scripts/run_all_tests.sh`, in the `SUITES` array beside the other pipeline
orchestrator entries:

    "js pipeline_orchestrator/tests/test_vega_tour.js"
    "py pipeline_orchestrator/tests/test_vega_tour_content.py"

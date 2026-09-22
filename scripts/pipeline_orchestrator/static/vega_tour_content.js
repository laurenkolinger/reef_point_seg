/* VEGA guided tour: the step list as data.

   Spec: docs/superpowers/specs/2026-09-16-vega-tutorial-design.md, section 2
   (the step shape and the advance kinds), section 6 (the ten stages every
   track reaches). The engine is static/vega_tour.js; this file holds copy
   only, so a sentence can change without touching behaviour.

   Version 2 (Lauren, 2026-09-17): there is no Next button. Every step
   completes on the student's own action on the real control, and every body
   ends by naming that action, because the sentence is the only way forward.
   Bodies are bite size: one idea, at most three short sentences, so a floating
   callout never carries a paragraph.

   Version 3 (Lauren, 2026-09-17, walking the tour): three faults fixed.
     1. No step lives on a page the tour is not wired into. The two steps that
        used to sit on the Bounding Boxes page are gone: one step on the
        pipeline window now says plainly where a folder of boxed pictures is
        prepared and that the page opens in its own tab, what such a folder
        holds is taught on the Train panel that names it, and the difference
        between a box and an outline is one sentence on the step that already
        talks about tracing. tests/test_vega_tour_content.py fails loudly if a
        step ever names a surface whose template carries no tour script tag.
     2. A step whose action opens another window carries "handoff", and its
        body says where it is going before it goes. The engine hands its place
        to the new window, so the student never arrives at step one again.
     3. Plain words first. A student is never shown a word the tour has not
        taught: no staged dataset, no cards, no rail, no class list, no
        holdout, no split, no champion, no composite standing alone. Where a
        word is on screen (Fine-tune, mAP50, test, survey frames, Mask size,
        champion) the plain words come first and the screen word after.

   Step shape:
     id        a slug, unique
     stage     1 to 10, the stage from spec section 6 this step belongs to
     track     "boxes", "outlines" or "both"
     surface   "launcher" (the VEGA page of the VICARIUS desktop),
               "orchestrator" (the pipeline window), "matrix", "labels" or
               "annotator", or "any" (the first screen). Every one of these is
               a page the tour is actually wired into: a step may never name a
               page whose template does not load vega_tour.js, because the
               student would walk there and the tour would vanish.
     anchor    a CSS selector for the real control, by id or attribute,
               never by text; an array of selectors is tried in order and the
               first one on the page wins, for a control that is only there
               once another part of VICARIUS has marked it; null only on the
               first screen, whose control is the track choice the callout
               itself draws
     title     Syne 17 px, sentence case, no full stop
     body      Inter 14 px; one idea, at most three short sentences, at most
               40 words; only <em> (a species binomial) and <code> (a file or
               folder name) markup; the last sentence names the action;
               {project_name} and {stage_name} are filled from the open
               project at render time and the fallback is shown when they
               cannot be
     advance   how the step completes, always a real action on the live page:
               "click" (the student clicks the anchor or something inside it),
               "input" (the student fills or changes the anchored field),
               "navigate" (the student reaches the target surface or panel),
               "event" (a training, evaluation or inference run starts or
               finishes). The first screen is the one "input" step with no
               anchor: the engine completes it when a track is chosen.
     event     event steps only: "step:6:start" (the run on step 6 starts),
               "step:6", "step:7", "step:8" (that step's run finishes)
     target    navigate steps only: "surface:<name>" for another page, or a
               selector that matches once the student has arrived
     handoff   the surface this step's action opens in another window, on the
               steps that open one. The engine carries the tour's place to
               that window: in the address of a window it opens on another
               address (the desktop is 5090, the pipeline window 5050, the
               annotator 5080), and in this browser's own store for a new tab
               at the same address. A step that declares it says so in its
               body, so a new window is never a surprise.
     fallback  the sentence shown when the anchor is absent (the panel is
               not open yet, or the step belongs to another page). The engine
               adds the page to go to, so a student is never trapped.

   Copy rules (design spec section 7, and Lauren's jargon ruling of
   2026-09-17): complete sentences; the first names who does what; never opens
   with "You"; the last sentence is the action the student takes; what the
   system does is a sentence with VEGA as its subject; a thing is named for
   what it is rather than for what the code calls it; a term of art is
   explained the first time a student meets it and only once per track; a code
   is expanded the first time it appears; no emoji, no em or en dash, never the
   word "land", never "UVI".

   The two tutorial projects are minted under the module's inprocess folder.
   A student reads their display names on the launcher's project list:
   "TUTORIAL boxes (detection)" and "TUTORIAL outlines (segmentation)". Stage 2
   of each track sends the student to open the one for the track, and glows
   that one entry through the "data-tutorial" attribute the launcher puts on
   it, falling back to the whole project list when the attribute is not there.
   The copy names it in words rather than pinning its exact name, so a rename
   strands nobody. */
(function (root, factory) {
  'use strict';
  var content = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = content;
  if (root) root.VegaTourContent = content;
})(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  var tracks = {
    boxes: {
      name: 'Boxes round each organism',
      blurb: 'Start from pictures whose boxes are already drawn, and train a model to box new ones.'
    },
    outlines: {
      name: 'Outlines traced round each organism',
      blurb: 'Trace outlines in VEGA, then train a model to trace them on new pictures.'
    }
  };

  var steps = [
    // -- Stage 1: what VEGA is, and the track ----------------------------
    {
      id: 'welcome', stage: 1, track: 'both', surface: 'any', anchor: null,
      title: 'Welcome to VEGA',
      body: 'VEGA teaches a computer to find organisms in reef pictures. VEGA traces outlines itself, but never draws boxes, so boxes arrive already drawn on the pictures. Choose which kind to learn.',
      advance: 'input',
      fallback: 'Choose one of the two kinds to begin the tutorial.'
    },

    // -- Stage 2: open the tutorial project ------------------------------
    // The desktop is one address and the pipeline window another, so these two
    // steps carry the handoff: the engine puts the tour's place in the address
    // the Open button opens, and the tour picks up there rather than at step 1.
    {
      id: 'open-tutorial-boxes', stage: 2, track: 'boxes', surface: 'launcher',
      anchor: ['[data-tutorial="boxes"]', '#runs-list'],
      title: 'Open the tutorial project',
      body: 'A project is one folder holding a set of pictures and everything VEGA makes from them. Opening one opens the pipeline window, where the rest of the tutorial runs. Press Open beside TUTORIAL boxes.',
      advance: 'click', handoff: 'orchestrator',
      fallback: 'The two TUTORIAL projects are in the project list on the VEGA page of the VICARIUS desktop.'
    },
    {
      id: 'open-tutorial-outlines', stage: 2, track: 'outlines', surface: 'launcher',
      anchor: ['[data-tutorial="outlines"]', '#runs-list'],
      title: 'Open the tutorial project',
      body: 'A project is one folder holding a set of pictures and everything VEGA makes from them. Opening one opens the pipeline window, where the rest of the tutorial runs. Press Open beside TUTORIAL outlines.',
      advance: 'click', handoff: 'orchestrator',
      fallback: 'The two TUTORIAL projects are in the project list on the VEGA page of the VICARIUS desktop.'
    },

    // -- Stage 3: data preparation, boxes ---------------------------------
    // One step, on a page the tour is on. The Bounding Boxes page carries no
    // tour, so the tour stays in the pipeline window while the student looks,
    // and the next step is waiting when they come back. What a folder of boxed
    // pictures holds is taught on the Train panel that names it (see
    // train-source-boxes), and the box against outline idea is one sentence on
    // boxes-annotator-note.
    {
      id: 'boxes-prepared', stage: 3, track: 'boxes', surface: 'orchestrator', anchor: 'a[href="/boxes"]',
      title: 'Where the boxes came from',
      body: 'Each tutorial picture already has a box drawn round every organism, saved in a short text file beside it. The Bounding Boxes page, where such folders are prepared, opens in its own tab. Press Bounding Boxes to look.',
      advance: 'click',
      fallback: 'Bounding Boxes is one of the tools under the stages in the pipeline window.'
    },

    // -- Stage 3: data preparation, outlines ------------------------------
    {
      id: 'annotate-tool', stage: 3, track: 'outlines', surface: 'orchestrator', anchor: 'a[href="/annotate"]',
      title: 'Where outlines are traced',
      body: 'The tutorial project already holds outlines someone traced and checked. Annotate Images opens the tracing window, called the annotator, in its own tab. Press Annotate Images.',
      advance: 'click',
      fallback: 'Annotate Images is one of the tools under the stages in the pipeline window.'
    },

    // -- Stage 4: the annotator itself (outlines) --------------------------
    {
      id: 'ann-initials', stage: 4, track: 'outlines', surface: 'annotator', anchor: '#reviewer-input',
      title: 'Sign your work',
      body: 'VEGA records these initials on every outline accepted from now on, and accepts none while the field is empty. Type one to four letters.',
      advance: 'input',
      fallback: 'The initials field sits in the bar across the top of the annotator.'
    },
    {
      id: 'ann-arm', stage: 4, track: 'outlines', surface: 'annotator', anchor: '#quickadd-bar',
      title: 'Choose what to name it',
      body: 'A label is the name an outline carries, such as <em>Porites astreoides</em>, whose survey code is PA. Press a label button, or tap its number key.',
      advance: 'click',
      fallback: 'The label buttons run across the top of the annotator review screen.'
    },
    {
      id: 'ann-place', stage: 4, track: 'outlines', surface: 'annotator', anchor: '#canvas-wrap',
      title: 'Place a point',
      body: 'One click is enough: VEGA asks a model to trace the outline round whatever was clicked. VEGA calls each finished outline a mask. Click once on an organism.',
      advance: 'click',
      fallback: 'The picture fills the middle of the annotator review screen.'
    },
    {
      id: 'ann-refine', stage: 4, track: 'outlines', surface: 'annotator', anchor: '#btn-mode-refine',
      title: 'Correct an outline',
      body: 'Refine fixes an outline that came out wrong: a left click adds the region clicked, a right click takes one away. Press Refine with an outline selected.',
      advance: 'click',
      fallback: 'The edit modes sit on the annotator toolbar under the top bar.'
    },
    {
      id: 'ann-size', stage: 4, track: 'outlines', surface: 'annotator', anchor: '#mask-size',
      title: 'Ask for more or less',
      body: 'Mask size decides how much of an organism the next outline covers, from one branch to a whole colony. Drag Mask size.',
      advance: 'input',
      fallback: 'The Mask size slider sits on the annotator toolbar.'
    },
    {
      id: 'ann-list', stage: 4, track: 'outlines', surface: 'annotator', anchor: '#mask-list',
      title: 'Accept or reject',
      body: 'Accepted outlines are what the model learns from, so a wrong one is worth rejecting. Click a row, then press Enter to accept it or X to reject it.',
      advance: 'click',
      fallback: 'The list of outlines is the panel at the right of the annotator review screen.'
    },
    {
      id: 'ann-next', stage: 4, track: 'outlines', surface: 'annotator', anchor: '#btn-next-frame',
      title: 'Move through the pictures',
      body: 'A batch is the pictures opened together, and the counter says which one is open. VEGA saves them all when the batch closes. Press Next (or finish).',
      advance: 'click',
      fallback: 'The picture buttons and the counter sit in the bar across the top of the annotator.'
    },
    {
      id: 'ann-export', stage: 4, track: 'outlines', surface: 'annotator', anchor: '#btn-save-close',
      title: 'Export the batch',
      body: 'Next on the last picture offers the export, which writes one short text file of outlines per picture for the trainer to read. Press Save and close.',
      advance: 'click',
      fallback: 'The export dialog opens from the last picture of the batch, and Save and close returns to the pipeline window.'
    },

    // -- Stage 4: the annotator, seen from a box project -------------------
    // This step carries the sentence that used to send a boxes student to the
    // Bounding Boxes page to watch a conversion they will never run.
    {
      id: 'boxes-annotator-note', stage: 4, track: 'boxes', surface: 'orchestrator', anchor: 'a[href="/annotate"]',
      title: 'A box is not an outline',
      body: 'A box says where an organism is; an outline says which pixels it covers. A box project skips the tracing stages, which is why so few are listed. Press Annotate Images to try tracing anyway.',
      advance: 'click',
      fallback: 'Annotate Images sits with the other tools under the stages in the pipeline window.'
    },

    // -- Stage 5: the Coverage Matrix and the Label Manager ---------------
    {
      id: 'tools-matrix', stage: 5, track: 'both', surface: 'orchestrator', anchor: 'a[href="/matrix"]',
      title: 'Open the Coverage Matrix',
      body: 'The Coverage Matrix is one grid of every picture ever surveyed against every label, across all projects. It opens in its own tab. Press Coverage Matrix.',
      advance: 'click', handoff: 'matrix',
      fallback: 'Coverage Matrix is one of the tools under the stages in the pipeline window.'
    },
    {
      id: 'matrix-grid', stage: 5, track: 'both', surface: 'matrix', anchor: '#mx-rescan',
      title: 'Read the grid',
      body: 'Each row is one picture, each column is one label, and an empty square means nobody has looked at that pair yet. Press Rescan to read every project again.',
      advance: 'click',
      fallback: 'The grid is the first tab of the Coverage Matrix.'
    },
    {
      id: 'matrix-legend', stage: 5, track: 'both', surface: 'matrix', anchor: '#mx-legend',
      title: 'What the colours mean',
      body: 'Five colours cover what happened to a picture and label pair, and only accepted outlines ever reach training. Click a colour key to read what it means.',
      advance: 'click',
      fallback: 'The colour key sits under the Coverage Matrix headline.'
    },
    {
      id: 'matrix-training-tab', stage: 5, track: 'both', surface: 'matrix', anchor: '#mx-tab-training',
      title: 'Where training sets come from',
      body: 'A training set is the folder of pictures and outlines a model learns from, gathered from the outlines reviewers accepted. Press Training Sets.',
      advance: 'navigate', target: '#mx-tab-training[aria-selected="true"]',
      fallback: 'Training Sets is the third tab of the Coverage Matrix.'
    },
    {
      id: 'tools-labels', stage: 5, track: 'both', surface: 'orchestrator', anchor: 'a[href="/labels"]',
      title: 'Open the Label Manager',
      body: 'The Label Manager holds the one list of names every project draws its labels from, so a short code always has a full name. It opens in its own tab. Press Label Manager.',
      advance: 'click', handoff: 'labels',
      fallback: 'Label Manager is one of the tools under the stages in the pipeline window.'
    },
    {
      id: 'labels-search', stage: 5, track: 'both', surface: 'labels', anchor: '#vocab-search',
      title: 'Where labels come from',
      body: 'Each row is one label: its short survey code, its category and its full name. Type a code such as PA for <em>Porites astreoides</em>.',
      advance: 'input',
      fallback: 'The list of names is the first section of the Label Manager.'
    },

    // -- Stage 6: Train ----------------------------------------------------
    {
      id: 'train-tile', stage: 6, track: 'both', surface: 'orchestrator', anchor: 'li[data-step="6"]',
      title: 'Go to Train Model',
      body: 'Train Model divides the pictures three ways: most to learn from, a few to say when to stop, and a few kept back to score the result. Press Train Model.',
      advance: 'navigate', target: '#panel-6.active',
      fallback: 'Train Model is the first stage a box project shows and the fifth on an outline project.'
    },
    {
      id: 'train-source', stage: 6, track: 'outlines', surface: 'orchestrator', anchor: '#s6-source',
      title: 'Check the training pictures',
      body: 'The list holds this project\'s own outlines, every training set built on the Coverage Matrix, and every folder of boxed pictures. Press the list and keep the tutorial set.',
      advance: 'click',
      fallback: 'The first choice on the Train Model panel names the pictures to train on.'
    },
    {
      id: 'train-source-boxes', stage: 6, track: 'boxes', surface: 'orchestrator', anchor: '#s6-source',
      title: 'What the box folder holds',
      body: 'The chosen training data is a folder of pictures, a short text file of boxes beside each one, and the list of names those boxes use. Press the list and keep the tutorial folder.',
      advance: 'click',
      fallback: 'The first choice on the Train Model panel names the pictures to train on.'
    },
    {
      id: 'train-model', stage: 6, track: 'both', surface: 'orchestrator', anchor: '#s6-model',
      title: 'Choose the model size',
      body: 'A bigger model learns more from a big set of pictures and takes far longer, which this small tutorial does not need. Press the model list and choose YOLO11 nano.',
      advance: 'click',
      fallback: 'The model picker is on the Train Model panel under Starting model.'
    },
    {
      id: 'train-epochs', stage: 6, track: 'both', surface: 'orchestrator', anchor: '#s6-epochs',
      title: 'Set the epochs',
      body: 'An epoch is one pass over every training picture, and ten passes over the tutorial set take a minute or two. Type 10.',
      advance: 'input',
      fallback: 'Epochs is a number field on the Train Model panel.'
    },
    {
      id: 'train-imgsz', stage: 6, track: 'boxes', surface: 'orchestrator', anchor: '#s6-imgsz',
      title: 'Set the picture size',
      body: 'The trainer reads every picture at this size, and the tutorial pictures are 256 pixels square. Type 256.',
      advance: 'input',
      fallback: 'Image size is a number field on the Train Model panel.'
    },
    {
      id: 'train-name', stage: 6, track: 'both', surface: 'orchestrator', anchor: '#s6-run-name',
      title: 'Name the run',
      body: 'VEGA appends the date and time, so two runs never share a folder. Type tutorial and your initials.',
      advance: 'input',
      fallback: 'The run name field is on the Train Model panel.'
    },
    {
      id: 'train-start', stage: 6, track: 'both', surface: 'orchestrator', anchor: '#s6-run-btn',
      title: 'Start training',
      body: 'VEGA saves these settings and starts the trainer, which keeps running even if the window closes. Press Start Training.',
      advance: 'event', event: 'step:6:start',
      fallback: 'Start Training is the primary button at the foot of the Train Model settings.'
    },

    // -- Stage 7: waiting for the run ---------------------------------------
    {
      id: 'train-wait', stage: 7, track: 'both', surface: 'orchestrator', anchor: '#s6-progress',
      title: 'Wait and watch',
      body: 'Each pass, the trainer scores how wrong its guesses are, a number the log calls the loss, and adjusts itself. The pictures kept back are never shown to it. Wait and watch.',
      advance: 'event', event: 'step:6',
      fallback: 'The progress bar and the log fill on the Train Model panel while a run is live.'
    },

    // -- Stage 8: Evaluate -------------------------------------------------
    {
      id: 'evaluate-open', stage: 8, track: 'both', surface: 'orchestrator', anchor: 'li[data-step="7"]',
      title: 'Go to Evaluate Model',
      body: 'Evaluate Model scores a finished model on the pictures kept back from it and writes a report. Press Evaluate Model.',
      advance: 'navigate', target: '#panel-7.active',
      fallback: 'Evaluate Model is the stage after Train Model in the pipeline window.'
    },
    // train-runs moved here from the Train stage on 2026-09-21 (LO): the
    // table of every trained model now renders as a closed disclosure on
    // the Evaluate Model panel rather than on Train Model (spec
    // 2026-09-21-vega-starting-model-and-training-data-design.md section 4),
    // so the tour meets it here instead.
    {
      id: 'train-runs', stage: 8, track: 'both', surface: 'orchestrator', anchor: '#s7-all-models-summary',
      title: 'Every model so far',
      body: 'Every model trained so far, box models and outline models apart, is listed behind All trained models on the Evaluate Model panel. Press All trained models to open it.',
      advance: 'click',
      fallback: 'All trained models sits near the foot of the Evaluate Model panel.'
    },
    {
      id: 'evaluate-run', stage: 8, track: 'both', surface: 'orchestrator', anchor: '#s7-run-dir',
      title: 'Choose the training',
      body: 'VEGA lists every finished training, and uses each one\'s best pass rather than its last. Press the run list and choose the training just finished.',
      advance: 'click',
      fallback: 'The training run picker is the first choice on the Evaluate Model panel.'
    },
    {
      id: 'evaluate-split', stage: 8, track: 'both', surface: 'orchestrator', anchor: '#s7-split',
      title: 'Score on pictures kept back',
      body: 'The test pictures were kept back from both learning and stopping, so this score says how the model does on pictures it has never seen. Press the split list and keep test.',
      advance: 'click',
      fallback: 'The split picker is on the Evaluate Model panel beside the run picker.'
    },
    {
      id: 'evaluate-start', stage: 8, track: 'both', surface: 'orchestrator', anchor: '#s7-run-btn',
      title: 'Run the evaluation',
      body: 'VEGA scores every kept-back picture with that model\'s best pass and fills the report. Press Run Evaluation.',
      advance: 'event', event: 'step:7',
      fallback: 'Run Evaluation is the primary button on the Evaluate Model panel.'
    },
    {
      id: 'evaluate-numbers', stage: 8, track: 'both', surface: 'orchestrator', anchor: '#s7-report-card .tip',
      title: 'Read the numbers',
      body: 'Precision is the share of the model\'s finds that were right. Recall is the share of the real organisms the model found. Press the i on the report for the rest.',
      advance: 'click',
      fallback: 'The report fills at the foot of the Evaluate Model panel once an evaluation has run.'
    },
    {
      id: 'evaluate-map', stage: 8, track: 'both', surface: 'orchestrator', anchor: '#s7-rounds-block .tip',
      title: 'What mAP50 means',
      body: 'One overall score, written mAP50, counts a find as right when it covers at least half the real organism. The stricter score, mAP50-95, asks for more overlap. Press the i on Rounds.',
      advance: 'click',
      fallback: 'The Rounds table sits at the foot of the Evaluate Model panel.'
    },
    {
      id: 'evaluate-files', stage: 8, track: 'both', surface: 'orchestrator', anchor: '#s7-pdf-export-dir',
      title: 'Where the report lives',
      body: 'VEGA writes the numbers as <code>metrics.json</code>, the reading copy and <code>report.pdf</code>, into the project\'s <code>step7_evaluateModel</code> folder. Type a folder path to copy the PDF elsewhere.',
      advance: 'input',
      fallback: 'The evaluation files sit in the project\'s step7_evaluateModel folder, one subfolder per run.'
    },

    // -- Stage 9: Model Inference -------------------------------------------
    {
      id: 'infer-open', stage: 9, track: 'both', surface: 'orchestrator', anchor: 'li[data-step="8"]',
      title: 'Go to Model Inference',
      body: 'Model Inference runs the trained model over pictures it has never seen, which is where its blind spots show. Press Model Inference.',
      advance: 'navigate', target: '#panel-8.active',
      fallback: 'Model Inference is the last stage in the pipeline window.'
    },
    {
      id: 'infer-model', stage: 9, track: 'both', surface: 'orchestrator', anchor: '#s8-run-dir',
      title: 'Choose the model',
      body: 'VEGA picks the project\'s best model when it has one, and uses that training\'s best pass either way. Press the model list and choose the training just finished.',
      advance: 'click',
      fallback: 'The model run picker is on the Model Inference panel.'
    },
    {
      id: 'infer-source-boxes', stage: 9, track: 'boxes', surface: 'orchestrator', anchor: 'input[name="s8-src"][value="dir"]',
      title: 'Choose the pictures',
      body: 'The tutorial pictures were seen in training and would flatter the model. Press Directory to run over a folder it has never seen.',
      advance: 'click',
      fallback: 'The source choice is on the Model Inference panel under Source.'
    },
    {
      id: 'infer-dir-boxes', stage: 9, track: 'boxes', surface: 'orchestrator', anchor: '#s8-source-dir',
      title: 'Name the folder',
      body: 'VEGA reads every picture directly inside the folder and none of its subfolders. Type the path of a folder of reef photographs.',
      advance: 'input',
      fallback: 'The source directory field appears on the Model Inference panel once Directory is chosen.'
    },
    {
      id: 'infer-source-outlines', stage: 9, track: 'outlines', surface: 'orchestrator', anchor: 'input[name="s8-src"][value="unused"]',
      title: 'Choose the pictures',
      body: 'Unused survey frames takes pictures from the archive this project never chose, so the model meets ones it has never seen. Press Unused survey frames.',
      advance: 'click',
      fallback: 'The source choice is on the Model Inference panel under Source.'
    },
    {
      id: 'infer-conf', stage: 9, track: 'both', surface: 'orchestrator', anchor: '#s8-conf',
      title: 'Set the confidence',
      body: 'Confidence is how sure the model is of a find, from 0 to 1, and anything under the value typed is thrown away. Type 0.25.',
      advance: 'input',
      fallback: 'Confidence is a number field on the Model Inference panel.'
    },
    {
      id: 'infer-start', stage: 9, track: 'both', surface: 'orchestrator', anchor: '#s8-run-btn',
      title: 'Run the model',
      body: 'VEGA saves a copy of each picture with the finds drawn on it, plus a list of what it found. Press Run Inference.',
      advance: 'event', event: 'step:8',
      fallback: 'Run Inference is the primary button on the Model Inference panel.'
    },
    {
      id: 'infer-gallery', stage: 9, track: 'both', surface: 'orchestrator', anchor: '#s8-gallery-refresh',
      title: 'Read the gallery',
      body: 'Each picture shows what the model found, and the small counts beside it say how many of each species. Press Refresh to read the batch again.',
      advance: 'click',
      fallback: 'The gallery fills at the foot of the Model Inference panel once a batch has run.'
    },

    // -- Stage 10: iterating, and coming back ------------------------------
    {
      id: 'iterate-boxes', stage: 10, track: 'boxes', surface: 'orchestrator', anchor: 'li[data-step="6"]',
      title: 'Train a better box model',
      body: 'A second round trains for more passes, on a bigger model, or onward from this one with Fine-tune. Press Train Model to start one.',
      advance: 'navigate', target: '#panel-6.active',
      fallback: 'The next round of a box project starts on Train Model, with Fine-tune on the table of trained models.'
    },
    {
      id: 'iterate-outlines', stage: 10, track: 'outlines', surface: 'orchestrator', anchor: 'li[data-step="6"]',
      title: 'Improve the outline model',
      body: 'A refine round puts the model\'s own guesses into the annotator to be corrected, and those corrections join the next training set. Press Train Model to start one.',
      advance: 'navigate', target: '#panel-6.active',
      fallback: 'The next round of an outline project starts on Train Model, after a refine round in the annotator.'
    },
    {
      id: 'iterate-holdout', stage: 10, track: 'both', surface: 'orchestrator', anchor: 'li[data-step="7"]',
      title: 'Keep the kept-back pictures out',
      body: 'A model that already saw a picture finds its organisms again, so a score on it would measure memory rather than skill. Press Evaluate Model to see the pictures every round is scored on.',
      advance: 'navigate', target: '#panel-7.active',
      fallback: 'The pictures kept back are the test ones every evaluation scores on.'
    },
    {
      id: 'resume', stage: 10, track: 'both', surface: 'orchestrator', anchor: '#topbar-actions .btn',
      title: 'Pick this up any time',
      body: 'The project {project_name} is parked at {stage_name}, and Open beside its name on the VEGA page returns to it. Press Open Project Folder to see everything it holds.',
      advance: 'click',
      fallback: 'Every project is parked where it was left: press Open beside its name on the VEGA page to come back.'
    }
  ];

  return { version: '2026-09-17-v3', tracks: tracks, steps: steps };
});

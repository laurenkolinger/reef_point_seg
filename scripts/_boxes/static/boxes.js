/* Bounding Boxes - external box datasets.

   Two sections read first: the staged dataset cards, and Train in VEGA, the
   one press that hands a staged dataset to the VEGA main window as a
   training project. The box-to-mask conversion, its progress and the
   converted datasets are folded into the collapsed "Advanced" details block
   below them.

   Every card says how its own dataset is staged again: the Coral Intel card
   holds the ingest action row, which this script moves into it, and a
   dataset written by the command-line stager carries the line that names
   that tool. Refresh reads the staged folder again, which is what a dataset
   staged from a terminal needs.

   Training never happens on this page. The VEGA main window is the one
   training surface (Lauren, 2026-09-04): Train in VEGA asks the server for
   the dataset's training project (minted now, or reused when one exists)
   and opens the answered URL in the main window, which honours
   ?project_dir=<absolute run dir> and renders that project on Train Model.
   Same contract, same fixed window name as the coverage matrix's "open in
   pipeline".

   Every control the page creates carries a title tooltip, same as the
   markup. No emoji, no em dashes. */
(function () {
  'use strict';

  var BASE = (window.BOXES_BASE || '').replace(/\/$/, '');
  var POLL_MS = 2500;
  // The window name the coverage matrix and the orchestrator's own popups
  // use for the main window, so every open reuses the one VEGA tab instead
  // of spawning another.
  var MAIN_WINDOW_NAME = 'reefpointseg_orchestrator';
  var INITIALS_RE = /^[A-Za-z]{1,4}$/;
  // The stager behind a staged dataset, shown by name: a code never stands
  // alone on a VICARIUS surface. The server answers the token, this maps it.
  var SOURCE_LABELS = {
    coral_intel: 'Coral Intel',
    yolo: 'YOLO',
    subset: 'Class-count subset'
  };
  // The three splits a dataset can pin an image to, in the order the card
  // shows them. Mirrors blueprint.py SPLIT_NAMES.
  var SPLIT_KEYS = ['train', 'valid', 'test'];
  // One phrase for a fact the dataset does not carry, in every cell of every
  // card: terms, a citation, a names table, a background count, split pins.
  // Three phrases were here before ("unknown", "none", "none recorded") and
  // read as three different states when the state is one, that the dataset
  // records nothing for this. A count of zero is a different thing and is
  // still shown as 0, so nothing is lost. UNREADABLE stays its own phrase:
  // a sidecar file that is there and cannot be read is a state the operator
  // has to act on before training, not an absent fact.
  var NOT_RECORDED = 'none recorded';
  var UNREADABLE = 'unreadable';
  // The stager token the server answers for the Coral Intel dataset, which is
  // the one dataset restaged by a button on this page rather than from a
  // terminal. Mirrors blueprint.list_datasets.
  var CORAL_INTEL_KIND = 'coral_intel';
  // The command-line stager that writes every other dataset, named where it
  // is so a card says how it is staged again.
  var YOLO_STAGER = 'scripts/_boxes/ingest_yolo_boxes.py';
  // The stager token the server answers for a dataset written by narrowing
  // another staged dataset to the classes that carry enough boxes, and the
  // tool that writes one. Mirrors subset_by_class_count.SOURCE_KIND.
  var SUBSET_KIND = 'subset';
  var SUBSET_TOOL = 'scripts/_boxes/subset_by_class_count.py';

  var elFlash = document.getElementById('bx-flash');
  var elDatasets = document.getElementById('bx-datasets');
  var elDatasetsSection = document.getElementById('bx-sec-datasets');
  var elIngestRow = document.getElementById('bx-ingest-row');
  var elListRow = document.getElementById('bx-list-row');
  var elRefresh = document.getElementById('bx-refresh');
  var elSource = document.getElementById('bx-source');
  var elIngest = document.getElementById('bx-ingest');
  var elProjectDataset = document.getElementById('bx-project-dataset');
  var elProjectBy = document.getElementById('bx-project-by');
  var elProjectExisting = document.getElementById('bx-project-existing');
  var elProjectResult = document.getElementById('bx-project-result');
  var elProjectLink = document.getElementById('bx-project-link');
  var elProjectNew = document.getElementById('bx-project-new');
  var elProjectOpen = document.getElementById('bx-project-open');
  var elConvertDataset = document.getElementById('bx-convert-dataset');
  var elConvertLimit = document.getElementById('bx-convert-limit');
  var elConvertName = document.getElementById('bx-convert-name');
  var elKeepDegenerate = document.getElementById('bx-keep-degenerate');
  var elConvert = document.getElementById('bx-convert');
  var elStatus = document.getElementById('bx-status');
  var elStop = document.getElementById('bx-stop');
  var elLog = document.getElementById('bx-log');
  var elOutputsBody = document.getElementById('bx-outputs-body');

  var DATASETS = [];
  // Training projects already minted for the dataset chosen in section 2,
  // newest first, as the server last listed them.
  var EXISTING = [];
  var lastPhase = '';
  // In-flight guard: a status read that has not answered yet must not have a
  // second one stacked on top of it every 2.5 seconds.
  var polling = false;

  // ── plumbing ───────────────────────────────────────────
  function api(path, opts) {
    return fetch(BASE + path, opts).catch(function () {
      throw new Error('Could not reach the orchestrator. Check that it is still running, then reload.');
    }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (body) {
        if (!r.ok) throw new Error(body.error || ('request failed: ' + r.status));
        return body;
      });
    });
  }

  function post(path, body) {
    return api(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {})
    });
  }

  function flash(msg, ok) {
    if (!elFlash) return;
    if (!msg) { elFlash.hidden = true; elFlash.textContent = ''; return; }
    elFlash.textContent = msg;
    elFlash.classList.toggle('is-ok', !!ok);
    elFlash.hidden = false;
  }

  function el(tag, attrs, text) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // The value of one fact. A string is the whole value; a list is a value
  // whose own words are captions rather than data (the split pins read
  // "5300 train / 1514 valid / 758 test"), and a piece marked dim is set in
  // the caption colour at the caption size so the numbers still read as the
  // value.
  function statValue(value) {
    var node = el('span', { class: 'bx-stat-n' });
    if (!Array.isArray(value)) {
      node.textContent = String(value);
      return node;
    }
    value.forEach(function (piece) {
      node.appendChild(el('span', piece.dim ? { class: 'bx-stat-sub' } : {},
        String(piece.text)));
    });
    return node;
  }

  // One fact on a dataset card: a count, or a short phrase such as a stager
  // name or a per-split triple. The tooltip covers the whole cell, so the
  // value and its label answer the same question. span asks for more than one
  // of the card's grid columns, which a value longer than about nine
  // characters needs: at one column the stager's name breaks after Coral,
  // which drops the label out of line with the rest of the row. SPAN_TWO
  // holds two columns; SPAN_ROW holds the whole row, which the split pins
  // need now that each count carries the name of its split. The two spanning
  // facts are placed last so the columns still pack from the left.
  var SPAN_TWO = 'grid-column: span 2';
  var SPAN_ROW = 'grid-column: 1 / -1';
  function stat(node, value, key, tip, span) {
    var box = el('div', { class: 'bx-stat', title: tip });
    // A cell whose value is the "nothing recorded" phrase takes two columns
    // unless the caller asked for more: the phrase is thirteen characters and
    // at one column it breaks after "none", which drops the caption below it
    // out of line with the rest of the row.
    var rule = span || (isAbsent(value) ? SPAN_TWO : '');
    if (rule) box.setAttribute('style', rule);
    box.appendChild(statValue(value));
    box.appendChild(el('span', { class: 'bx-stat-k' }, key));
    node.appendChild(box);
  }

  // A fact the dataset does not carry, as a cell value: the one phrase for
  // that, at caption size and in the caption colour. It is not a value, and
  // at the value size the phrase breaks across two lines and pushes the
  // cell's own caption out of line with the rest of the row.
  function absent() {
    return [{ text: NOT_RECORDED, dim: true }];
  }

  // True when a cell value is that phrase rather than data the dataset
  // carries, which decides how wide the cell has to be.
  function isAbsent(value) {
    return Array.isArray(value) && value.length === 1 && value[0].dim === true;
  }

  // The stager that wrote a dataset, by name. A token this page does not know
  // is shown as it came rather than dropped, so a dataset staged by a later
  // tool still says where it came from; a manifest that names no stager reads
  // as nothing recorded, the same phrase every other absent fact carries.
  function sourceLabel(kind) {
    var key = String(kind || '');
    if (!key) return absent();
    return SOURCE_LABELS[key] || key;
  }

  // Every tooltip below is true of the dataset the card is showing. The two
  // stagers record different facts in different ways, so a line whose
  // explanation depends on which one wrote the dataset branches on the
  // manifest's source kind rather than claiming one story for both.
  var SOURCE_TIP = 'VICARIUS shows which tool wrote this dataset: the Coral Intel stager, the generic YOLO box stager, or the subset tool that narrows another staged dataset to the classes carrying at least so many boxes.';
  var SOURCE_TIP_UNKNOWN = "This dataset's manifest names no stager, so VICARIUS cannot say which one wrote it.";
  // Coral Intel derives its class map from the file names; the generic YOLO
  // stager takes it from the dataset's own data.yaml and checks every label
  // line against it. Neither sentence is true of the other dataset.
  var CLASSES_TIP = {
    coral_intel: 'Classes counts the species names derived from the filenames and verified against every annotation file.',
    yolo: 'Classes counts the class codes this dataset declares in its data.yaml, which VICARIUS checks against every annotation file before staging.'
  };
  var CLASSES_TIP_OTHER = 'Classes counts the class codes this dataset declares, as the stager that wrote it recorded them.';
  var WHOLE_FRAME_TIP = 'Whole frame counts the boxes that cover the whole picture, so they say only that the species is somewhere in it.';
  var BACKGROUND_TIP = 'Background counts the staged images whose label file holds no box, so they show the detector what a frame with nothing to find looks like.';
  var SPLIT_PINS_TIP = 'VICARIUS shows how many images the dataset pins to each split, in the order train, valid, test, so a training project and an imported model score on the same held-out images.';
  var SPLIT_PINS_TIP_NONE = 'VICARIUS finds no split pins in this dataset, so a training project built from it draws its own train, valid and test split, and two projects on this dataset hold out different images.';
  var SPLIT_PINS_TIP_UNREADABLE = 'VICARIUS could not read the split pins in this dataset and cannot say which images are held out. Check split_pins.json in the dataset folder before training, because a pins file the split builder cannot read stops a training project.';
  // How a card says the dataset is written again. The Coral Intel card carries
  // the ingest button itself (the page script moves the action row into it),
  // so only a dataset written from a terminal needs the line. A subset is
  // written by neither stager: running the YOLO stager over a subset folder
  // would stage it as a plain box dataset and reproduce neither the
  // class-count rule nor class_map.csv, so a subset card names its own tool.
  var RESTAGE_TIP = 'VICARIUS staged this dataset from the command line: the stager at ' + YOLO_STAGER + ' writes it again over the folder recorded in ingest_manifest.json inside the dataset folder. Press Refresh once it finishes.';
  var SUBSET_RESTAGE_TIP = 'VICARIUS wrote this dataset from the command line: the tool at ' + SUBSET_TOOL + ' reads the parent dataset and the class-count minimum that ingest_manifest.json inside the dataset folder records. Remove this folder before running it, because that tool never writes into a folder that is already there.';

  function restageText(d) {
    if (String(d.source_kind || '') !== SUBSET_KIND) {
      return 'Staged again by running ' + YOLO_STAGER + ' over the same folder.';
    }
    return 'Made again by running ' + SUBSET_TOOL + ' over the parent dataset named in ingest_manifest.json.';
  }

  function restageTip(d) {
    return String(d.source_kind || '') === SUBSET_KIND ? SUBSET_RESTAGE_TIP : RESTAGE_TIP;
  }

  function sourceTip(kind) {
    return String(kind || '') ? SOURCE_TIP : SOURCE_TIP + ' ' + SOURCE_TIP_UNKNOWN;
  }

  function classesTip(kind) {
    return CLASSES_TIP[String(kind || '')] || CLASSES_TIP_OTHER;
  }

  // The labels_localized folder is the Coral Intel stager's second labels
  // folder, with the whole-frame boxes left out. The generic stager writes no
  // such folder, so only a dataset that carries one is promised one.
  function wholeFrameTip(hasLocalized) {
    return WHOLE_FRAME_TIP + (hasLocalized
      ? ' VICARIUS drops them from the labels_localized folder inside the dataset and skips them when converting to masks.'
      : ' VICARIUS skips them when converting to masks.');
  }

  // A count the stager never wrote down, as against a count of zero: zero is
  // a fact and is shown as one, and only a missing count reads none recorded.
  function countOrNotRecorded(value) {
    return (value === null || value === undefined) ? absent() : Number(value);
  }

  function backgroundTip(value) {
    return BACKGROUND_TIP + ((value === null || value === undefined)
      ? ' The stager that wrote this dataset counted none of them, so VICARIUS records no count here; the images may still be there.'
      : ' VICARIUS stages them beside the annotated images, and they add nothing to the box count.');
  }

  // The per-split pin counts, each number carrying the name of its split, so
  // the cell says which images train and which are held out rather than
  // leaving the order to be guessed. None recorded when the dataset pins
  // nothing and VEGA splits it fresh; unreadable when a pins file is there
  // that the server could not read, which the split builder refuses too, so
  // the operator has to fix it before training.
  function splitPins(d) {
    if (d.split_pins_unreadable) return UNREADABLE;
    if (!d.split_counts) return absent();
    var pieces = [];
    SPLIT_KEYS.forEach(function (k, i) {
      if (i) pieces.push({ text: ' / ' });
      pieces.push({ text: String(Number(d.split_counts[k] || 0)) });
      pieces.push({ text: ' ' + k, dim: true });
    });
    return pieces;
  }

  function splitPinsTip(d) {
    if (d.split_pins_unreadable) return SPLIT_PINS_TIP_UNREADABLE;
    return d.split_counts ? SPLIT_PINS_TIP : SPLIT_PINS_TIP_NONE;
  }

  function datasetOption(d) {
    return el('option', { value: d.name, title: d.path },
      d.name + '  (' + d.n_images + ' images, ' + d.n_boxes + ' boxes, '
      + d.n_classes + ' classes)');
  }

  // ── 1 datasets ─────────────────────────────────────────
  function datasetCard(d) {
    var card = el('div', { class: 'bx-ds' });
    card.appendChild(el('div', { class: 'bx-ds-name', title: d.path }, d.name));

    var stats = el('div', { class: 'bx-ds-stats' });
    stat(stats, d.n_images, 'images', 'Images counts the records staged in this dataset.');
    stat(stats, d.n_boxes, 'boxes', 'Boxes counts the bounding boxes across every staged record.');
    stat(stats, d.n_classes, 'classes', classesTip(d.source_kind));
    stat(stats, d.n_degenerate, 'whole frame', wholeFrameTip(d.has_localized));
    stat(stats, countOrNotRecorded(d.n_background), 'background',
      backgroundTip(d.n_background));
    stat(stats, d.n_names ? d.n_names : absent(), 'names table',
      'VICARIUS counts the species names the dataset carries for its class codes, which the evaluation report and the inference overlay show beside each code.');
    stat(stats, sourceLabel(d.source_kind), 'source', sourceTip(d.source_kind), SPAN_TWO);
    stat(stats, splitPins(d), 'split pins', splitPinsTip(d), SPAN_ROW);
    card.appendChild(stats);

    // Terms and citation are the dataset's own or nothing at all. The server
    // answers an empty string both for a manifest that carries no text and
    // for one carrying a stager's own words for nothing recorded, so an empty
    // string is the single signal here: no card claims terms for a dataset
    // whose own text says there are none, and none borrows another dataset's
    // paper.
    var cite = el('div', { class: 'bx-ds-cite' });
    var terms = String(d.license_note || '');
    var citation = String(d.citation || '');
    var termsTip = terms
      ? 'VICARIUS stages this dataset under these terms and keeps them beside the data as LICENSE.md.'
      : 'VICARIUS shows no license terms recorded for this dataset. Record them by staging it again with the license argument.';
    var citeTip = citation
      ? 'Cite this publication whenever the annotations are used.'
      : 'VICARIUS shows no citation recorded for this dataset. Record one by staging it again with the citation argument.';
    cite.appendChild(el('span', { class: 'bx-stat-k', title: termsTip }, 'Terms'));
    cite.appendChild(el('p', { title: termsTip }, terms || NOT_RECORDED));
    cite.appendChild(el('span', { class: 'bx-stat-k', title: citeTip }, 'Cite'));
    cite.appendChild(el('p', { title: citeTip }, citation || NOT_RECORDED));
    if (d.ingested_at) {
      cite.appendChild(el('p', {
        title: 'Staged is when the last ingest wrote this dataset, in Atlantic Standard Time.'
      }, 'Staged ' + d.ingested_at + (d.limit ? ' (limited to ' + d.limit + ' records)' : '')));
    }
    // How this dataset is written again. The Coral Intel card gets the action
    // row itself; every other dataset was written from a terminal, so the
    // card names the tool that writes it, which is not the same tool for a
    // subset as for a staged dataset the YOLO stager wrote.
    if (String(d.source_kind || '') !== CORAL_INTEL_KIND) {
      cite.appendChild(el('p', { class: 'bx-ds-restage', title: restageTip(d) },
        restageText(d)));
    }
    card.appendChild(cite);
    return card;
  }

  // The Coral Intel action row sits inside the Coral Intel card, so that card
  // carries the one action that restages it; while that dataset is not staged
  // the row stays in the section, above the Refresh row, where it is the only
  // way to stage it at all. The row is moved, never rebuilt, so its button
  // keeps the click handler wired at boot.
  function placeIngestRow(coralCard) {
    if (!elIngestRow) return;
    if (coralCard) {
      coralCard.appendChild(elIngestRow);
      elIngestRow.classList.add('is-in-card');
      return;
    }
    elIngestRow.classList.remove('is-in-card');
    if (elDatasetsSection && elListRow) {
      elDatasetsSection.insertBefore(elIngestRow, elListRow);
    }
  }

  function renderDatasets(body) {
    DATASETS = body.datasets || [];
    var coralCard = null;
    elDatasets.innerHTML = '';
    if (!DATASETS.length) {
      elDatasets.appendChild(el('p', { class: 'bx-ds-empty' },
        'Nothing staged yet. Run the ingest below to copy the Coral Intel dataset into this module.'));
    } else {
      DATASETS.forEach(function (d) {
        var card = datasetCard(d);
        if (String(d.source_kind || '') === CORAL_INTEL_KIND) coralCard = card;
        elDatasets.appendChild(card);
      });
    }
    placeIngestRow(coralCard);

    // The folder name only, never the path: the path runs through a personal
    // Dropbox tree that has no business in page text or in a tooltip, so the
    // server answers the name and never sends the path at all.
    var src = (body.source || {});
    var srcName = String(src.name || '');
    elSource.textContent = src.present
      ? 'Source folder found: ' + srcName
      : 'Source folder is not on this machine';
    elSource.setAttribute('title', srcName
      ? 'VICARIUS reads the distribution folder named ' + srcName
        + '. Point TCRMP_CORAL_INTEL_SOURCE at it to read it from somewhere else.'
      : 'No source folder is configured for this host. Point TCRMP_CORAL_INTEL_SOURCE at the distribution.');
    if (src.present) {
      elIngest.disabled = false;
      elIngest.setAttribute('title',
        'Copy the Coral Intel images and annotations into supporting_data/bbox_datasets/coral_intel, derive and verify the class names from the filenames, and write the class map, the license and the ingest manifest. Safe to run again: it copies only what changed.');
    } else {
      elIngest.disabled = true;
      elIngest.setAttribute('title',
        'The Coral Intel distribution folder is not on this machine where this host expects it. '
        + 'Put it there, or point TCRMP_CORAL_INTEL_SOURCE at it, then reload.');
    }

    var prev = elConvertDataset.value;
    elConvertDataset.innerHTML = '';
    if (!DATASETS.length) {
      elConvertDataset.appendChild(el('option', { value: '' }, 'No staged datasets yet'));
      elConvert.disabled = true;
    } else {
      DATASETS.forEach(function (d) {
        elConvertDataset.appendChild(el('option', { value: d.path, title: d.path },
          d.name + '  (' + d.n_images + ' images, ' + d.n_boxes + ' boxes, '
          + d.n_classes + ' classes)'));
      });
      if (prev) elConvertDataset.value = prev;
      if (!elConvertDataset.value) elConvertDataset.value = DATASETS[0].path;
      // The status poll can answer before this list does and disable the
      // button against an empty list; re-enable it here rather than on the
      // next poll tick.
      elConvert.disabled = lastPhase === 'running';
    }
    syncProjectDatasets();
  }

  // Answers whether the read settled, so the Refresh press can report a count
  // rather than overwriting the message a failed read just put on screen.
  function loadDatasets() {
    return api('/api/datasets').then(function (body) {
      renderDatasets(body);
      return true;
    }).catch(function (e) { flash(e.message); return false; });
  }

  // The staged datasets are read at boot and when a job here finishes. A
  // dataset staged from a terminal happens while neither is true, so this is
  // the press that reads the folder again without reloading the page.
  function refreshDatasets() {
    flash('');
    if (elRefresh) elRefresh.disabled = true;
    loadDatasets().then(function (ok) {
      if (elRefresh) elRefresh.disabled = false;
      if (!ok) return;
      flash(DATASETS.length === 1
        ? '1 staged dataset read.'
        : DATASETS.length + ' staged datasets read.', true);
    });
  }

  function startIngest() {
    flash('');
    post('/api/ingest', {}).then(function (body) {
      flash('Ingest started. Staging into ' + body.dest, true);
      poll();
    }).catch(function (e) { flash(e.message); });
  }

  // ── 2 train in VEGA ────────────────────────────────────
  function chosenProjectDataset() {
    var name = elProjectDataset ? elProjectDataset.value : '';
    return DATASETS.filter(function (d) { return d.name === name; })[0] || null;
  }

  function syncProjectDatasets() {
    if (!elProjectDataset) return;
    var prev = elProjectDataset.value;
    elProjectDataset.innerHTML = '';
    if (!DATASETS.length) {
      elProjectDataset.appendChild(el('option', { value: '' }, 'No staged datasets yet'));
    } else {
      DATASETS.forEach(function (d) { elProjectDataset.appendChild(datasetOption(d)); });
      if (prev) elProjectDataset.value = prev;
      if (!elProjectDataset.value) elProjectDataset.value = DATASETS[0].name;
    }
    loadTrainingProjects();
  }

  function loadTrainingProjects() {
    var d = chosenProjectDataset();
    if (!d) { EXISTING = []; renderExisting(''); return Promise.resolve(); }
    return api('/api/training-projects?dataset=' + encodeURIComponent(d.name))
      .then(function (body) {
        // An answer for a dataset that is no longer chosen is stale.
        var now = chosenProjectDataset();
        if (!now || body.dataset !== now.name) return;
        EXISTING = body.projects || [];
        renderExisting('');
      })
      .catch(function (e) { EXISTING = []; renderExisting(e.message); });
  }

  function renderExisting(problem) {
    if (!elProjectExisting) return;
    var d = chosenProjectDataset();
    var text, tip;
    if (!d) {
      text = 'Stage a box dataset above first.';
      tip = 'Nothing is staged, so there is no dataset to hand to VEGA yet.';
    } else if (problem) {
      text = 'Could not list the training projects for ' + d.name + ': ' + problem;
      tip = 'VICARIUS answered this when the page asked which projects train on ' + d.name
        + ': ' + problem;
    } else if (!EXISTING.length) {
      text = 'No training project yet for ' + d.name + '. Train in VEGA makes one and opens it.';
      tip = 'VICARIUS found no project folder that trains on ' + d.name
        + ', so the press mints run_<date>_<initials>_train_' + d.name
        + ' in the projects folder and opens it.';
    } else {
      var p = EXISTING[0];
      var made = p.created_at ? ' (made ' + p.created_at + (p.initials ? ' by ' + p.initials : '') + ')' : '';
      text = d.name + ' already has '
        + (EXISTING.length === 1 ? 'a training project' : EXISTING.length + ' training projects')
        + '. Train in VEGA opens ' + p.run_dir_name + made + '.';
      tip = 'VICARIUS found ' + EXISTING.length + ' project folder'
        + (EXISTING.length === 1 ? '' : 's') + ' that train on ' + d.name
        + ' and opens the newest, so every training run of this dataset shares one split and '
        + 'one rounds ledger. Make a second project only for a fresh split.';
    }
    elProjectExisting.textContent = text;
    elProjectExisting.setAttribute('title', tip);
    if (elProjectNew) elProjectNew.hidden = !EXISTING.length;
    syncProjectButtons();
  }

  function syncProjectButtons() {
    if (!elProjectOpen) return;
    var d = chosenProjectDataset();
    elProjectOpen.disabled = !d;
    if (elProjectNew) elProjectNew.disabled = !d;
    var tip;
    if (!d) {
      tip = 'Stage a box dataset first: there is nothing to hand to VEGA yet.';
    } else if (EXISTING.length) {
      tip = 'Open ' + EXISTING[0].run_dir_name + ' in the VEGA main window on Train Model, where the '
        + 'full training form trains a detector on ' + d.name + '. VEGA switches its window to '
        + 'that project; nothing new is written.';
    } else {
      tip = 'Make a training project for ' + d.name + ' and open it in the VEGA main window on '
        + 'Train Model, where the full training form trains a detector. Steps 1 to 4 are skipped '
        + 'because there is nothing to annotate, and the images stay where they are staged.';
    }
    elProjectOpen.setAttribute('title', tip);
  }

  function openInVega(url) {
    // A fixed window name reuses the already-open VEGA tab. A browser that
    // blocks the window still has the link beside the result line.
    var win = window.open(url, MAIN_WINDOW_NAME);
    if (!win) {
      flash('The browser blocked the VEGA window. Use the "open it" link beside the result.');
    }
  }

  function trainInVega(reuse) {
    flash('');
    var d = chosenProjectDataset();
    if (!d || !elProjectBy) return;
    var by = String(elProjectBy.value || '').trim();
    var willMint = !reuse || !EXISTING.length;
    if (willMint && !INITIALS_RE.test(by)) {
      flash('Type your initials (1 to 4 letters) so the new project folder says who made it.');
      elProjectBy.focus();
      return;
    }
    var target = willMint ? 'a new training project for ' + d.name : EXISTING[0].run_dir_name;
    if (!window.confirm('VEGA switches its main window to ' + target
                        + '. A project open there is set aside, not changed. Continue?')) return;
    elProjectOpen.disabled = true;
    if (elProjectNew) elProjectNew.disabled = true;
    post('/api/create-project', { dataset: d.name, by: by, reuse: reuse })
      .then(function (body) {
        var msg = (body.reused ? 'Opening ' : 'Made ') + body.display_name
          + ' (' + body.run_dir_name + ') in VEGA; it starts at Train Model.';
        flash(msg, true);
        if (elProjectResult) {
          elProjectResult.textContent = msg;
          elProjectResult.setAttribute('title',
            'Project folder ' + body.run_dir_name + ' records the path of ' + body.dataset
            + ' and holds no imagery of its own; Train Model builds its own train, validation '
            + 'and test split from that path.');
        }
        if (elProjectLink) {
          elProjectLink.setAttribute('href', body.open_url);
          elProjectLink.hidden = false;
        }
        openInVega(body.open_url);
        return loadTrainingProjects();
      })
      .catch(function (e) { flash(e.message); })
      .then(function () { syncProjectButtons(); });
  }

  // ── convert (advanced) ─────────────────────────────────
  function startConvert() {
    flash('');
    post('/api/convert', {
      dataset: elConvertDataset.value,
      limit: parseInt(elConvertLimit.value, 10),
      name: elConvertName.value,
      keep_degenerate: !!elKeepDegenerate.checked
    }).then(function (body) {
      flash('Converting ' + body.limit + ' images. Output goes to ' + body.out_dir, true);
      poll();
    }).catch(function (e) { flash(e.message); });
  }

  // ── progress (advanced) ────────────────────────────────
  function renderStatus(st) {
    var phase = st.phase || 'idle';
    var text;
    if (phase === 'running') {
      text = (st.kind || 'job') + ' running' + (st.name ? ': ' + st.name : '')
             + (st.started_at ? ' since ' + st.started_at : '');
    } else if (phase === 'error') {
      text = st.message || ('failed with exit code ' + st.exit_code);
    } else if (phase === 'done') {
      text = (st.message || 'finished') + (st.name ? ': ' + st.name : '');
    } else {
      text = st.message || 'Idle';
    }
    elStatus.textContent = text;
    elStatus.className = 'bx-status'
      + (phase === 'running' ? ' is-running' : phase === 'error' ? ' is-error'
         : phase === 'done' ? ' is-done' : '');
    elStop.disabled = phase !== 'running';
    elConvert.disabled = phase === 'running' || !DATASETS.length;
    elIngest.disabled = phase === 'running' || elIngest.disabled;
    if (st.log_tail && st.log_tail.length) elLog.textContent = st.log_tail.join('\n');
    if (phase !== lastPhase && (phase === 'done' || phase === 'error')) {
      // A finished ingest changes the dataset cards and what can be handed
      // to VEGA; a finished conversion changes the outputs table.
      loadDatasets();
      loadOutputs();
    }
    lastPhase = phase;
  }

  function poll() {
    if (polling) return;
    polling = true;
    api('/api/status').then(renderStatus).catch(function () { /* transient */ })
      .then(function () { polling = false; });
  }

  function stopJob() {
    post('/api/stop').then(function () { flash('Stopped.', true); poll(); })
      .catch(function (e) { flash(e.message); });
  }

  // ── converted datasets (advanced) ──────────────────────
  function loadOutputs() {
    return api('/api/seg_outputs').then(function (body) {
      var outputs = body.outputs || [];
      elOutputsBody.innerHTML = '';
      if (!outputs.length) {
        var tr = el('tr');
        tr.appendChild(el('td', { colspan: '4', class: 'bx-empty' },
          'No conversions yet. Convert a staged dataset above.'));
        elOutputsBody.appendChild(tr);
        return;
      }
      outputs.forEach(function (o) {
        var row = el('tr');
        row.appendChild(el('td', { title: o.path }, o.name));
        row.appendChild(el('td', {}, String(o.n_labels)));
        row.appendChild(el('td', {}, String(o.n_polygons)));
        row.appendChild(el('td', {}, o.built_at || ''));
        elOutputsBody.appendChild(row);
      });
    }).catch(function (e) { flash(e.message); });
  }

  // ── wiring ─────────────────────────────────────────────
  elIngest.addEventListener('click', startIngest);
  if (elRefresh) elRefresh.addEventListener('click', refreshDatasets);
  if (elProjectDataset) elProjectDataset.addEventListener('change', loadTrainingProjects);
  if (elProjectOpen) elProjectOpen.addEventListener('click', function () { trainInVega(true); });
  if (elProjectNew) elProjectNew.addEventListener('click', function () { trainInVega(false); });
  elConvert.addEventListener('click', startConvert);
  elStop.addEventListener('click', stopJob);

  // The initials VICARIUS passes as ?initials=XX when it opens this page:
  // trimmed, upper-cased, two to four letters, else "" (ignored). Only ever
  // assigned to .value, never inserted as markup.
  function passedInitials() {
    try {
      var raw = new URLSearchParams(window.location.search).getAll('initials');
      var v = String(raw.length === 1 ? raw[0] : '').trim().toUpperCase();
      return /^[A-Z]{2,4}$/.test(v) ? v : '';
    } catch (e) { return ''; }
  }
  // Rewrites the initials value to upper case as it is typed, keeping the caret.
  function upperAsTyped(input) {
    var upper = input.value.toUpperCase();
    if (upper === input.value) return;
    var at = input.selectionStart;
    input.value = upper;
    try { input.setSelectionRange(at, at); } catch (e) { /* not a text field */ }
  }
  if (elProjectBy) {
    elProjectBy.value = passedInitials();
    elProjectBy.addEventListener('input', function () { upperAsTyped(elProjectBy); });
  }

  loadDatasets();
  loadOutputs();
  poll();
  setInterval(function () { poll(); }, POLL_MS);
})();

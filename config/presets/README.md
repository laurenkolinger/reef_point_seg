# Step 6 training presets

Each `.yaml` file in this directory is a named bundle of Step 6 form values
that can be loaded into the orchestrator with one click (**Load preset**
dropdown in the Step 6 panel).

## File format

```yaml
_meta:
  name: "Short human-readable name"
  description: "One-line summary shown in the dropdown."
  task: detect                  # required: detect or segment
  subject: coral                # required: coral, fish, algae or any

params:
  # Keys match the orchestrator's internal cfg keys (same shape as
  # project.json's steps['6'].config block). Any key listed here
  # overwrites the current form field on import. Any key NOT listed is
  # left alone, so presets can be layered or treated as base configs.
  epochs: 1000
  patience: 150
  ...
```

Add comments freely: the loader ignores them.

A key with no applier in `STEP6_PRESET_APPLIERS`
(`scripts/pipeline_orchestrator/static/orchestrator.js`) is dropped in silence
at import time and named in the "ignored" part of the applied-preset line, so
every key a preset ships must have a form field behind it.

## Which recipe the Train panel offers

The Train panel carries two controls that together decide which recipe is on
offer: **Training task** (Segmentation or Detector) and **Subject** (coral,
fish, algae or other). `GET /api/step/6/presets`
(`scripts/pipeline_orchestrator/app.py`, `step6_list_presets`) reports what
every file declares and chooses nothing; the panel does the matching, in the
browser, so a change to either control re-filters the list at once.

A recipe matches when its `_meta.task` equals the task control and its
`_meta.subject` equals the subject control. A recipe declaring
`subject: any` matches whatever the subject control says. The control's own
`other` option names a subject no recipe is written for, so it matches the
`any` recipes and nothing else.

What the panel does with the matches:

1. **Exactly one recipe matches.** It is selected in the dropdown, and it is
   applied to a project whose Step 6 settings have never been saved. The
   panel names it and says which task and subject chose it.
2. **More than one matches.** Nothing is applied. The panel lists the
   candidates by name and says that more than one recipe claims the same task
   and subject, so the operator picks one and presses Import. There is
   no tie-break at all: not by file name, not by file time, not by size, not
   by folder order.
3. **None matches.** Nothing is applied, the panel says so, and the form keeps
   the Train panel's own defaults.

A recipe that declares neither key, or a value outside the two vocabularies,
is **unclassified**: the dropdown still lists it so it can be imported by
hand, marks it as unclassified, and never selects it on its own. A file that
will not parse is listed the same way, with its parse error.

`task` accepts `detect` or `segment` (the spellings the trainer takes:
`detection`, `bbox`, `seg` and `segmentation` are read as well). `subject`
accepts `coral`, `fish`, `algae` or `any`. Both are read with the surrounding
spaces removed and in lower case, and a value that is not a string is treated
as if the key were absent.

The two built-in presets (**Full retrain** and **Fine-tune**) were removed on
2026-09-21: a run now trains from a base model by default with no preset
needed to get there, and continuing from an earlier VEGA run's weights is the
"Continue from a VEGA model (advanced)" disclosure on the Train panel instead
of a preset button. See
[docs/adr/0009-a-run-starts-from-a-base-model-and-continuing-from-a-vega-model-is-the-advanced-choice.md](../../docs/adr/0009-a-run-starts-from-a-base-model-and-continuing-from-a-vega-model-is-the-advanced-choice.md).

Today the four shipped recipes declare:

| File | task | subject |
|---|---|---|
| `coral_heavy_augment_640.yaml` | detect | coral |
| `fish_heavy_augment_640.yaml` | detect | fish |
| `tuned_v1_regularized.yaml` | segment | coral |
| `yolo26l_heavy_seg_aug.yaml` | segment | coral |

So Detector plus coral applies the Coral Intel detector, Detector plus fish
applies the fish detector, and Segmentation plus coral applies neither of the
two coral segmentation recipes: it lists both and waits for a choice, which is
the deliberate behaviour of rule 2.

## Where the Subject control starts

A staged box dataset may record what it is about, as `subject` in its
`ingest_manifest.json` (`coral`, `fish` or `algae`). When a project trains
from one, the panel opens its Subject control on that value. It does nothing
else: it never overrides a value the operator set, and it never selects a
recipe on its own. `scripts/_boxes/ingest_yolo_boxes.py --subject` writes the
key when a dataset is staged, and a subset made by
`scripts/_boxes/subset_by_class_count.py` inherits it from its parent.

## Shipping a new preset

1. Copy an existing preset and rename it (`.yaml` extension required).
2. Edit the `_meta.name` and `_meta.description` so it is identifiable.
3. Declare `_meta.task` and `_meta.subject`. Without both, the recipe is
   listed as unclassified and is never offered.
4. Set `params` to the values you want the preset to impose.
5. Check what else declares the same task and subject. Two recipes claiming
   one pair means neither is applied by itself, which is safe but is probably
   not what you wanted: either narrow one of them or accept that the operator
   picks between them every time.
6. Hit **Refresh presets** in the Step 6 panel (no server restart needed).

The file name plays no part in any of this. It is only the preset id the
backend lists and the id `/api/step/6/presets/<id>` takes, extension included:
`fish_heavy_augment_640.yaml`, not `fish_heavy_augment_640`.

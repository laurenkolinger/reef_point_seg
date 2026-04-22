# Step 7 evaluation presets

Each `.yaml` file in this directory is a named bundle of Step 7 form
values that can be loaded into the orchestrator with one click
(**Load preset** dropdown at the top of the Step 7 panel).

## File format

```yaml
_meta:
  name: "Short human-readable name"
  description: "One-line summary shown under the dropdown."

params:
  # Keys match the orchestrator's internal cfg keys for Step 7
  # (same shape as project.json's steps['7'].config block). Any key
  # listed here overwrites the current form field on import. Any key
  # NOT listed is left alone — so presets can be layered.
  split: test
  imgsz: 1024
  conf_threshold: 0.25
  iou_threshold: 0.6
  preview_count: 8
  pdf_export_dir: ""
```

Add YAML comments freely — the loader ignores them.

## Shipping a new preset

1. Copy `standard.yaml` and rename it (`.yaml` extension required).
2. Edit the `_meta.name` / `_meta.description` so it's identifiable.
3. Set `params` to the values you want the preset to impose.
4. Hit **Refresh** in the Step 7 preset loader (no server restart needed).

The file name (minus `.yaml`) is what the backend uses as the preset ID.

## Relation to Quick-start

The **Use recommended defaults** button in the Step 7 panel is a
shortcut that fills the form with `split=test`, `conf=0.25`, `iou=0.6`,
`preview_count=8`, and auto-detects `imgsz` from the chosen run's
`args.yaml`. Presets here serve a different purpose — saving/swapping
full configuration bundles (e.g. per-experiment variants).

# Step 6 training presets

Each `.yaml` file in this directory is a named bundle of Step 6 form values
that can be loaded into the orchestrator with one click (**Load preset**
dropdown in the Step 6 panel).

## File format

```yaml
_meta:
  name: "Short human-readable name"
  description: "One-line summary shown in the dropdown."

params:
  # Keys match the orchestrator's internal cfg keys (same shape as
  # project.json's steps['6'].config block). Any key listed here
  # overwrites the current form field on import. Any key NOT listed is
  # left alone — so presets can be layered or treated as base configs.
  epochs: 1000
  patience: 150
  ...
```

Add comments freely — the loader ignores them.

## Shipping a new preset

1. Copy an existing preset and rename it (`.yaml` extension required).
2. Edit the `_meta.name` / `_meta.description` so it's identifiable.
3. Set `params` to the values you want the preset to impose.
4. Hit **Refresh presets** in the Step 6 panel (no server restart needed).

The file name (minus `.yaml`) is what the backend uses as the preset ID.

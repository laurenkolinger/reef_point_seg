# Coral Intel replication study: params files

The settings for the Coral Intel replication (Cordero et al. 2026, Table 2) and its three challengers, in VEGA's preset format (`_meta` plus `params`). Nothing in this study's code carries a setting; change a value here and the next run uses it.

| File | What it sets |
|---|---|
| `coral_intel_A0_replicate.yaml` | Train Model: the Coral Intel replicate, YOLOv8x at 640 px |
| `coral_intel_B1_yolo11x.yaml` | Train Model: A0 with the YOLO11x backbone |
| `coral_intel_B2_imgsz1024.yaml` | Train Model: A0 at 1024 px |
| `coral_intel_B3_offline_only.yaml` | Train Model: A0 with online augmentation off |
| `evaluate_test_split.yaml` | Evaluate Model: the scoring protocol for all four |

The offline augmentation that builds the training dataset has its own params file, `config/augment_presets/coral_intel_table2.yaml`.

Read by `scripts/_ablation/coral_intel_vega_driver.py`, which saves each recipe as a project's Train settings and presses Start through VEGA's web API. A challenger may differ from A0 only in the keys its `_meta.changes_from_replicate` lists; the driver refuses any other difference.

These files sit outside `config/presets/` because that folder's rules require each recipe to declare a task and a subject and every key to have a Train panel field. Four more recipes declaring detect and coral would stop `coral_heavy_augment_640.yaml` being applied on its own, and `task`, `seed`, `erasing` and `auto_augment` have no panel field. VEGA forwards all of them from a project's saved Train settings.

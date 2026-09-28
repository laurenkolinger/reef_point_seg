# One training surface: the VEGA main window

Status: accepted
Source: Lauren Olinger's ruling of 2026-09-04 ("I do not want Train and Infer. This should all be done in the VEGA main window"), recorded in `/mnt/rip/vicarius_drive/docs/superpowers/plans/2026-09-04-vega-training-foldin.md`; TODO T20260903-154500 ("if Lauren later wants the runs table and champion pick, fold them into the Train step").
Last updated: 2026-09-04 (LO)

Until 2026-09-04 a model could be trained from three places: the pipeline orchestrator's Train step (the VEGA main window), the project-free Train + Infer page at `/models`, and the Bounding Boxes page's own trainer, which was a remote control for `/models`. The three did not agree: the second forwarded no training task, no image size and four of about fifty knobs, and it seeded the detection split on a user-typed lab name while the first seeded it on the dataset name, so two scores shown side by side had been measured on different held-out images. We decided that training, evaluation and inference happen in the VEGA main window only. Train + Infer is retired; the four things only it offered (the cross-project runs table, the champion, training from a staged box dataset, and fine-tuning from the newest weights) move into the Train, Evaluate and Inference steps as cards; the Bounding Boxes page keeps its two real purposes (staging an external box dataset and converting boxes to outlines) and hands training off by minting a training project that opens in the main window.

## Considered options

- Keep Train + Infer and bring it up to the main window's control set. Rejected: two surfaces for one job drift again, and every knob, tooltip and test would exist twice.
- Give the Bounding Boxes page its own trainer so it no longer depends on `/models`. Rejected: a third training surface, which is the opposite of the ruling.
- One surface, the main window, with the shared run library (`scripts/_modellab/labruns.py`) kept as library code both the window and the Bounding Boxes handoff import. Chosen.

## Consequences

- A training run is always a project's run: it lives under `inprocess/run_*/step6_trainModel/`, is launched through `POST /api/step/6/run` with the full control set and the training task pinned, and appears in the runs card with every other run the module can see. Lab runs already on disk under `inprocess/_models/` stay readable through the same card.
- The `/models` page, its template, script and stylesheet are removed; the routes that only it called go with it. `labruns.py` survives as the shared library and is imported by the orchestrator directly.
- The Bounding Boxes page keeps its admin-mode hide from TODO T20260903-154500; the Train + Infer half of that entry is superseded by this record.

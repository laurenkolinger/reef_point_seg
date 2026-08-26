# pipeline_orchestrator changelog

- 2026-07-03 (LO): step-3 "Lores only" checkbox sets TCRMP_LORES_ONLY for Choose Images.
- 2026-07-03 (LO): "Low res only" checkbox moved to the Step-4 (combined annotator) panel, checked by default -> TCRMP_LORES_MODE in the routing env. Removed the Step-3 "Lores only" selection filter + checkbox (it skipped naming-mismatched frames). Route reuse now records + compares lores_mode (.lores_mode marker) so toggling it forces a re-route.

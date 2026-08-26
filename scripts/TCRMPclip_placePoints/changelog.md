# TCRMPclip_placePoints — changelog

- 2026-07-03 (LO): export_batch scales point coords by the selected_frames 'scale' column for lores frames so prompts land in lores image space.
- 2026-07-03 (LO): lores delivery moved to routing. export_batch(lores_mode) delivers every >1920px frame from its deterministic lores twin (path mirror off the resolved raw path, NOT a reconstructed name -> never skips), scaling points by the twins actual/orig long-edge ratio; already-small frames + missing-twin frames pass through as-is. Replaces the selected_frames is_lores/manifest approach.

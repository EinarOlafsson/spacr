# omezarr-p4 (0aj P4) English source handoff

Translations are dropped until everything else is done (maintainer, 2026-11-23
15:10). This file and `tr_omezarr-p4_sources.json` list the English sources
only. No catalog, translation pin or reviewed translation record is changed.

P4 (no masks/ working folder: the normalised GPU batches are built in memory
from fov.ome.zarr, plotting runs included; the batch plan and the run's small
records go to `fov.ome.zarr/.spacr_mask_run`, removed at the end of a run that
keeps no intermediates): no new runtime templates. The one new console line
(`plotting: ... (in memory)`) is log text, like the archive line it replaces,
so it stays English without tr().

Changed settings help: `keep_intermediate`.

Changed docstrings: `spacr.image_store._mask_sidecar_folder`,
`spacr.io._mask_batches_stay_in_memory`, `spacr.io.concatenate_and_normalize`
(returns), `spacr._mask_batches._plan_file`. Added (module-level, not in
`__all__`): `spacr.image_store.MASK_RUN_RECORDS`,
`_project_keeps_masks_in_store`, `_mask_run_dir`, `_remove_mask_run_records`;
`spacr._mask_batches._plot_first_batch`; `spacr.plot._plot_array_channels`.

The exact help text and sha256 are in the JSON (regenerated with
`0aj_scratch/tools/tr_new.py` against 2ffa5815eb).

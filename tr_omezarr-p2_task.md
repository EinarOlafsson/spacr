# omezarr-p2 (0aj P2) English source handoff

Translations are dropped until everything else is done (maintainer, 2026-11-23
15:10). This file and `tr_omezarr-p2_sources.json` list the English sources
only. No catalog, translation pin or reviewed translation record is changed.

P2 (object masks written straight into fov.ome.zarr labels, no
masks/<object>_mask_stack): no new runtime templates (the shared mask reader
and writer in `spacr/image_store.py` print nothing).

Changed settings help: `masks`, `keep_intermediate`, `mask_image_format`.

Changed public docstrings: `spacr.image_store.PlateWriter.write_labels` (new
`keep_others`, `unchanged`). Added (module-level, not in `__all__`):
`spacr.image_store._mask_label_name`, `_mask_names`, `_mask_folder_exists`,
`_mask_exists`, `_read_mask`, `_mask_header`, `_mask_valid`, `_mask_stat`,
`_write_mask`, `_prepare_mask_folder`, `_remove_mask_folder`, `_mask_folders`,
`_mask_digest`, `_mask_sidecar_folder`, `_consume_working_masks`.

The exact help texts and sha256 are in the JSON (regenerated with
`0aj_scratch/tools/tr_new.py` against 4747deb472).

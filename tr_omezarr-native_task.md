# omezarr-native (0aj) English source handoff

Translations are dropped until everything else is done (maintainer, 2026-11-23
15:10). This file and `tr_omezarr-native_sources.json` list the English sources
only. No catalog, translation pin or reviewed translation record is changed.

New runtime templates (all wrapped in `tr()`, English fallback works):

- `spacr/io.py`: four notes when a Mask run cannot be one plate store and is
  merged to `merged/*.npy` instead (not a field name; two files for one field;
  several plates; time or depth axes).
- `spacr/measure.py`: "[measure] Reading {count} fields from the merged store
  {path}."; the one older-layout note; the crop summary line.
- `spacr/core.py`: the end-of-merge channel check line and its failure message.
- `spacr/ports.py`: the merged-store shape problem and its fix line (chaining readiness).
- `spacr/image_store.py`: the refusal when a plate store would take a second
  plate.

Changed settings help: `mask_image_format`, `keep_intermediate`,
`crop_image_format`.

Changed public docstrings: `spacr.image_store.export_merged`, `PlateWriter`
(new `store_path`, `shards`, `attributes` parameters), `PlateWriter.write_field`
(`record`), `spacr.crop_store` module, `cut_crops` (`all_labels`), `class_dir`,
`committed_rows`, `spacr.crops.read_merged_plane_layout`,
`spacr.utils.cleanup_pipeline_folders`, `spacr.utils.measure_test_mode`. Added: `spacr.image_store.verify_store`,
`spacr.crop_store.read_crop_labels` (module-level, not in `__all__`). The exact
texts and sha256 are in the JSON.

# omezarr-p3 (0aj P3: t and z axes) English source handoff

Translations are dropped until everything else is done (maintainer, 2026-11-23
15:10). This file and `tr_omezarr-p3_sources.json` list the new English
sources only; every one is wrapped in `tr()` and the English fallback works.
No catalog, translation pin or reviewed translation record is changed.

- `spacr/io.py`: the note when a field's channels have different z slices
  (their maximum projection is stored), and the two refusals when the time
  points of a time-lapse field differ in channels or in size/type.

Settings help changed: `mask_image_format`, `keep_intermediate`,
`keep_original_images`, `save_original_images`, `z_stack`.

Retired (no longer emitted; the catalog entries are now stale): "Spatial
OME-Zarr export cannot represent timepoints. Timelapse processing will use
the retained merged NPY files."

The exact texts and sha256 are in the JSON (generated with
`0aj_scratch/tools/tr_new.py` against 4747deb472).

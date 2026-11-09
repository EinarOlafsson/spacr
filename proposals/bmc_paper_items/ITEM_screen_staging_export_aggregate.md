# Item spec: screen staging, one settings file per plate for prep + masks + measure, per-plate parquet export, and the DMSO-referenced well aggregate

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. This is "NEW-4" in `R/REPRODUCIBILITY.md` §6.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`. Source inventory: `R/SPACR_PROVENANCE.md` (port P13).

## Why
BBBC021, BBBC022 and JUMP CPJUMP1 were run through drivers. The drivers call `spacr.core.preprocess_generate_masks` twice (preprocess pass, then mask pass) and `spacr.measure.measure_crop`, and add custom steps for import, export and aggregation. As a result:
- each saved `gen_mask_settings.csv` holds only one of the two passes;
- import (renaming files into cellvoyager names), the sqlite → parquet export and the well aggregate have no settings at all;
- measure settings were not saved for every JUMP unit.

## Reference implementation
- **`R/bbbc021/spacr/bbbc021_spacr.py`:**
  - `cmd_import`: symlinks the raw files under cellvoyager names;
  - mask stages: `preprocess_generate_masks` with `preprocess=True, masks=False`, then `preprocess=False, masks=True`;
  - measure: `get_measure_crop_settings` + `measure_crop`;
  - `export_plate`: sqlite → per-plate parquet; n_cells, cells_without_nucleus, cells_multi_nucleus;
  - `cmd_aggregate`: wells_mean, wells_median, DMSO 1st/99th percentiles;
  - `_timed` / `Sampler` (runtime), `cmd_snapshot` (code pin).
- **`R/jump/spacr/jump_spacr.py`:** the same calls, plus `mirror`, `claim`/`publish`, `park`, `gpu_guard`. Its aggregate also writes `wells_heterogeneity.parquet` (per-well SD and 10–90% spread).
- **`R/bbbc022/spacr/bbbc022_spacr.py`:** same as BBBC021, with the 5-channel choices (nucleus w1, cell w4 + w1).
- **Settings already saved:** `R/bbbc021/settings/<plate>/`, `R/jump/settings/cpjump1/<unit>/` and `R/jump/settings/work_wd4tb/<unit>/`, `R/bbbc022/settings/<plate>/`.

## What to build
1. **One settings file runs the whole plate.**
   - `spacr-run mask` with `preprocess=True, masks=True` must equal the two-pass result. Verify; if it does not, fix the module so that it does.
   - A new module entry `spacr-run screen` (or a `phases` list on Make Masks) runs import → prep → masks → measure → export → aggregate from one CSV.
2. **Import:** settings `import_naming` ('cellvoyager'), `import_regex` (source pattern), `import_mode` ('symlink' | 'copy'), `import_map` (an optional CSV of source → target names). This replaces `cmd_import` and records the mapping in `<src>/settings/import_map.csv`.
3. **Export:** `export_parquet` (bool), `export_parquet_dir`. One parquet per plate with the `export_plate` columns and the per-plate count summary (n_cells, cells_without_nucleus, cells_multi_nucleus) in the image table (item 814).
4. **Aggregate:**
   - `aggregate_levels` ('well_mean, well_median');
   - `aggregate_control` (e.g. 'DMSO');
   - `aggregate_control_percentiles` ('1,99');
   - `aggregate_heterogeneity` (bool: per-well SD and 10–90% spread, as JUMP).
   - Writes `wells_mean.parquet`, `wells_median.parquet`, `wells_heterogeneity.parquet`.
5. **Timing:** `record_timing` (bool) writes the per-stage CPU-s/GPU-s JSON that `_timed` wrote.

## Validation data and acceptance
Acceptance: identical output, or within the tolerance stated per check, to the reference outputs at the paths below.
1. **BBBC021 plate Week10_40111:** a one-file settings run (after re-import) gives masks with identical label counts, matched IoU at least the GPU self-diff floor, and parquet features within rtol 1e-3 (GPU) of the of-record outputs.
   - The parquet files are under `/mnt/wd4tb/bmc_reanalysis/bbbc021_spacr/work/`; the tables were moved to `/nas_mnt/data/bmc_reanalysis/bbbc021_archive/`.
   - `wells_mean`/`wells_median` are identical when computed from the same parquet.
   - Then all 55 plates: the total cell count equals the sum over the of-record per-plate parquet files in the same folders.
2. **JUMP unit BR00117010-r0102:** identical masks/parquet/aggregate against `/mnt/wd4tb/jump/spacr/` outputs.
3. **BBBC022 plate 20585:** same criteria (`R/bbbc022/`).
4. **Tests (CPU):**
   - one-pass equals two-pass on a synthetic plate;
   - import map round trip;
   - aggregate with DMSO percentiles on a toy table.

## Effort
- **Effort:** M.

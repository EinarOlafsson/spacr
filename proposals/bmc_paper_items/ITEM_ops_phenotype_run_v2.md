# Item spec: OPS phenotype run from one settings file, writing merged/, standard measurements.db tables, owned cells and crops

- **Lifecycle:** future (alpha), behind "Show alpha features" under `ALPHA_FEATURES[752]`, like the rest of the phenotype path.
- **Number:** the build part of items 813 (OPS module v2), 812 (standard layout), 809 (sharded crops), 810 (bulk SQLite) and 811 (join indexes). Record it under 813 or take the next free number; check `features/feature_list/` and the claims files first.
- **Rule:** a `spacr-run ops --settings <csv>` run (or Import settings + Run in the GUI) must give output identical to the reference on the validation data. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Source inventory:** `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/SPACR_PROVENANCE.md` (port P1).

## Why
The OPS phenotype run on the Funk et al. 2022 data (plate A and later plates) was produced by a custom driver, `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/scripts/funk_phenotype.py`, on frozen spaCR 8147e421e5 from the unmerged branch `ops-funk-phenotype-run`.
- Ownership (A1b), flat-field correction, the failed-field drop and the edge exclusion were applied afterwards by pilot scripts.
- origin/nightly already has the phases `ph_flatfield, ph_segment, ph_measure, ph_masks, merge, aggregate` in `spacr.ops_engine.run_ops` (items 852, 853, 857, 860–862). These phases write only per-well parquet and mask npz files.
- They do not write `merged/`, the standard measurement tables, `png_list` or crops.

## Reference implementation (read these first)
- **Driver:** `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/scripts/funk_phenotype.py`. Archived copy: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_phenotype/code_archive/scripts/`; frozen source: `code_archive/src_frozen.tar.gz`.
  - `raw_index_for_plate` (via `ops_engine._index_tiles`);
  - `project_field` (z max projection through `_ph_read_field`; plates with a second acquisition are registered by phase correlation on the centre 1,024 px, then `_shift_plane`, falling back to the well median; shifts logged to `registration.jsonl`);
  - `stage_task`, `staged_files` (a 4-thread stager 48 fields ahead);
  - `measure_task` (`_ph_measure_field` on GPU, CPU fallback on OOM; a provisional link through `ops_merge.refine_field_transform`, `field_to_well`, `merge_field`);
  - `crop_cells`, `write_png_crops` (16-bit PNG r = AF750, g = GFP, b = DAPI, raw after 2026-11-05), `well_crop_targets` (crops per guide);
  - `spacr_sqlite_store` (`utils._append_to_measurements_db`, `filepaths_to_database`, `build_png_channels`, `_crop_center`, `_generate_names`);
  - `export_well`, `run_well`, `cmd_run`, `aggregate_plate`.
- **Settings used** (lines ~500–503): `ops_ph_channels` = DAPI,GFP,A594,AF750; `ops_ph_cell_channel` = AF750; `ops_ph_downsample` = 2; `ops_ph_batch_size` = 32; `ops_ph_texture` = True; nucleus and cell models cpsam; `ops_merge_threshold` = 4.0. Launcher: `launch_gpu_v6.sh`.
- **Post-run pilot chain** (already ported as 852/853/860/861; reuse it, do not re-port):
  - `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/scripts/merge_a1b.py` (A1b ownership);
  - `mask_pass.py`, `flat_correct.py`, `remeasure.py` (flat-field);
  - `fix_registration.py` (failed fields);
  - `aggregate_final.py`.
- **Crop re-cut:** `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/ops_gene_classes/scripts/recut_crops.py`, `convert_flat.py` (tar shards + index).
- **Layout spec:** `features/feature_list/813_ops_module_v2.txt` ("merged/", "measurements/", "data/ crops", "PHENOTYPE ILLUMINATION"); `812_*.txt`; `809_*.txt`; `810_*.txt`; `811_*.txt`.
- **Ownership design:** `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/ops_overlap/REPORT.md` §9. Cross-field ownership by masks is a separate spec (`ITEM_ops_cross_field_ownership.md`, not yet published); call it when it lands, but do not build it here.

## What to build (all in `spacr/ops_engine.py` and `spacr/ops_settings.py`, alpha)
1. **New phase `ph_merged`**, after `ph_masks` and before `merge`. It runs only when named in `phases`. For each placed field it writes `<ops_merged_root>/merged/phenotype/<plate>/<well>/<site>.npz` (`np.savez_compressed`) containing:
   - `channels`: the max-projected phenotype channels in `ops_ph_channels` order, uint16 as read, then flat-field divided when `ops_merged_flatfield` = True, using the `ph_flatfield` maps. Formula (I − b)/F + b, as `ops_flatfield='exact'` does (853), stored as float32, or rounded back to uint16 when `ops_merged_dtype='uint16'`;
   - `nuclei`, `cells`, `cytoplasm` label images at full phenotype resolution (`_ph_upscale` of the downsampled masks, exactly as `ph_masks` writes them);
   - `owned`: a uint8 per-label flag array (1 = owned by this field under `ops_ownership`, 0 = margin copy, 2 = touches the field edge);
   - `meta`: a JSON string with plate, well, site, the field-to-well transform (`_ph_transforms` columns) and the spaCR commit.
   - The phase also writes a layout manifest `merged/LAYOUT.json` (`{"layout": "ops_merged_v1", "channels": [...], "masks": [...], "dtype": ...}`).
   - Optional SBS stacks under `merged/sbs/...` (setting `ops_merged_save_sbs`, default False here) as specified in 813.
2. **New phase `ph_tables`**, after `aggregate`. It writes the standard spaCR tables into `<dst_root>/measurements/measurements.db`:
   - `cell`, `nucleus`, `cytoplasm` (one row per owned object, columns = `_ph_feature_columns` with the compartment prefix the Measure module uses);
   - `image` (item 814 shape, one row per field);
   - `png_list` (when crops are written).
   - Each row carries `prcfo` (`plate_row_col_field_object`), `plateID`, `rowID`, `columnID`, `fieldID`, `object_label`, `object_id` (the OPS link id), `sgRNA`, `gene`, `genotype_rule`, `mitotic`, `owned`.
   - Bulk loading follows 810 (one transaction per frame, `executemany`, PRAGMAs as in commit 8147e421e5). Indexes follow 811 (after the load).
   - When the row count exceeds `ops_tables_parquet_threshold` (default 5,000,000), the wide feature tables go to per-well Parquet under `measurements/<table>/<plate>_<well>.parquet`, and the db keeps the index columns. This matches 813 §5.
3. **New phase `ph_crops`**: raw crops of owned, linked cells (no per-crop stretch, item 772), into the sharded store of item 809 (`data/<category>/<plate>/<well>/shard-NNNN.tar` + `.index.parquet`, with at most `ops_max_files_per_folder` files per folder). The sampling rule matches `well_crop_targets`:
   - `ops_crop_per_guide` (int, None = all);
   - `ops_crop_size` (int px);
   - `ops_crop_channels`;
   - `ops_crop_flatfield` (bool, default True);
   - `ops_crop_seed`.
4. **Second-acquisition registration** for plates whose channels came from two acquisitions (in the Funk data, plates B and C):
   - `ops_ph_secondary_channels` (list; the channels in the second acquisition);
   - `ops_ph_secondary_registration` (`none` | `phase_correlation`, default `none`);
   - `ops_ph_secondary_window` (1024);
   - `ops_ph_secondary_fallback` (`well_median`).
   - It must log the per-field shifts into `ops_phenotype` (columns `secondary_dy`, `secondary_dx`, `secondary_fallback`).
5. **Destination on the NAS:** `ops_merged_root` (str, default = `dst_root`). merged/ and crops can then go to `/nas_mnt/data/...` while the working database stays on local NVMe.
   - Writes are atomic (`.part` then `os.replace`), as `ph_masks` does.
   - Resumable: a field whose npz exists and whose options marker (`_ph_options_marker`) matches is skipped.
6. **GPU measurement switch** `ops_ph_measure_gpu` (bool, default False). The reference used GPU measure; GPU and CPU measure differ slightly on some columns (the per-column comparison is in the reference run folder), and skew/kurtosis are fixed by item 788.
7. **One settings file of record.** `spacr-run ops --settings` with `phases = stitch,phenotype,objects,decode,ph_flatfield,ph_segment,ph_measure,ph_masks,ph_merged,merge,aggregate,ph_tables,ph_crops` runs a whole plate. Settings: `ops_ownership='a1b'`, `ops_flatfield='analytic'`, `ops_drop_failed_fields=True`, `ops_drop_nonfinite_rows=True`, `ops_ph_funk_features=True`, `ops_feature_qc=True`, `ops_genotype_table` = the 842 decode output.

## Settings (new; add to the defaults, types, OPS list, catalogue and `docs/setting_consumers.json`)
`ops_merged_root`, `ops_merged_flatfield` (True), `ops_merged_dtype` ('float32' | 'uint16', default 'uint16'), `ops_merged_save_sbs` (False), `ops_tables_parquet_threshold` (5000000), `ops_crop_per_guide` (None), `ops_crop_size` (96), `ops_crop_channels` (None = all), `ops_crop_flatfield` (True), `ops_crop_seed` (0), `ops_max_files_per_folder` (10000), `ops_ph_secondary_channels` (None), `ops_ph_secondary_registration` ('none'), `ops_ph_secondary_window` (1024), `ops_ph_secondary_fallback` ('well_median'), `ops_ph_measure_gpu` (False).

## Validation data and acceptance
Acceptance: identical output, or within the tolerance stated per check, to the reference outputs at the NAS paths below.
1. **Field level.** Plate A, field A1/514 (the pilot field; its 406 feature columns are known to be bit-identical across code versions).
   - Same masks: label count exact; matched-label IoU at least the GPU self-diff floor. Measure the floor first by running `_ph_segment_field` twice on the same field.
   - Same 406 feature columns: rtol 1e-5 on CPU measure.
   - Reference: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_phenotype/work/plateA/A1/` (`phenotype_cells_A1.parquet`, `phenotype_masks/site-514.npz`).
2. **Well A1, plate A.** Run from settings, then compare:
   - owned-cell set, link `object_id` and A1b flags identical to `merge_a1b.py` outputs (the 852 equivalence check: zero mismatches);
   - flat-field-corrected features bit-identical to `flat_correct.py` (853);
   - crops for 200 sampled cells pixel-identical to `recut_crops.py` / `convert_flat.py` output.
3. **Plate A funnel.** The per-well counts (segmented, whole and owned, linked, with a guide after the 861 drop) must equal `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/csv/funnel_per_well.csv`, integer for integer.
4. **Tables.** `measurements.db` `cell` row count = owned cells; joins on `prcfo` give the same guide/gene tables as `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/settings/runs/all_cells.json` (float32-identical).
5. **Second-acquisition registration (plate B).** The shifts in `ops_phenotype` equal `registration.jsonl` of the reference run (integer-exact).
6. **Tests (CPU, synthetic):**
   - merged npz layout and manifest;
   - Measure and the other merged/ readers detect `ops_merged_v1`;
   - no object owned twice;
   - folder caps;
   - resume skips finished fields;
   - the 100 M-row reader on synthetic data (813).

## Effort
- **Effort:** L (about 3–5 working days, mostly I/O plumbing; the science is already ported).

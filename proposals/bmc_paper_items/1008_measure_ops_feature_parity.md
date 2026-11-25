# Item spec: Measure/OPS per-cell feature parity, the Otsu colocalisation family and the final-run family list

- **Lifecycle:** future (alpha), behind "Show alpha features". With every new setting at its default, Measure and OPS output stay byte for byte as today.
- **Number:** 1008 (assigned by master 2026-11-21; 1003-1007 are taken). Schema number 954 is reserved; ask the coordinator before writing a schema row.
- **Paper-critical:** yes. The final Funk runs need it before the pilot, together with item 1007.
- **Code basis:** train 7 final source `dfe6633b5b`: `spacr/measure.py` (`PROFILE_FEATURE_FAMILIES`, `_intensity_measurements`, `_morphological_measurements`, `_add_profile_frames`, `_add_funk_frames`, `_calculate_correlation_object_level`, `_calculate_radial_distribution`, `_create_dataframe`, `_analyze_cytoskeleton`, `_spatial_measurements`), `spacr/ops_sample.py` (`_measurement_settings`, `funk_object_features`, `measure_phenotype_field`), `spacr/ops_settings.py` (`OPS_TO_MEASURE_KEYS`, `ops_ph_*`), `spacr/ops_engine.py` ph_measure task (`measure_phenotype_field` call near line 5288 only).
- **Builds on:** item 934 (every OPS column is a Measure column with the same value; ratchet `tests/test_measure_ops_features_934.py`), item 941 (cost note on the slow families).
- **Maintainer decisions 2026-11-21 (~10:45, 11:00):**
  1. Measure and OPS offer the SAME per-cell measurements, computed by the SAME `spacr.measure` code.
  2. Final-run features: the B2 set without granularity and Gabor, plus Funk 2022 and Zernike.
  3. Colocalisation for the OPS run: only FAST measurements, covering Brieflow's colocalisation set, plus at most a little more.
  4. OPS exposes the old-spaCR blocks B2 used: legacy_radial, cytoskeleton, spatial, coloc_block.
  5. Every family in the final preset is first tested for information value (gain in clustering known pairs and complexes; separate design card). The preset is therefore a plain explicit family list: a family that fails is dropped by editing the list, with no code change.
- **Lab rule:** spaCR development runs on the WS (2026-11-21, 11:10). This spec was written at Home; the build goes to a WS session or Codex.

## 1. Parity audit (state at dfe6633b5b)
Costs are CPU seconds per 100,000 cells, measured on the B2 run (300 Funk plate A well A1 fields, 374,432 cells, 1,248 cells per field, 1 thread per worker; `/mnt/wd4tb/scratch_fm_1116/b2/feat/timing.jsonl`). Column counts are for 4 channels and the OPS compartments cell, nucleus, cytoplasm (`family_map_A.json`).

| Family / block | Measure | OPS ph_measure | Computed by | Cols | CPU-s / 100k cells |
|---|---|---|---|---|---|
| morphology (regionprops) | yes | yes | `_morphological_measurements` | in base | in base |
| intensity (extended regionprops, percentiles) | yes | yes | `_intensity_measurements` / `_extended_regionprops_table` | in base | in base |
| GLCM homogeneity (`homogeneity`, `ops_ph_texture`) | yes | yes | `_calculate_homogeneity` | in base | in base |
| blur | yes | yes | `_estimate_blur` | in base | in base |
| distance to nucleus (`distance_gaussian_sigma`) | yes | yes | `_measure_intensity_distance` | in base | in base |
| Funk mass displacement, pair correlation, form factor | yes, every mask incl. cytoplasm (`_add_funk_frames`) | nucleus and cell only (`funk_object_features`) | `_funk_object_columns`, `_form_factor` | in base (+11 cytoplasm in Measure) | in base |
| Zernike, degree 8 (`zernike`) | yes | yes (cell, nucleus) | `_calculate_zernike` | 50 | in base |
| **base total** (all rows above) | | | | 475 | 7,893 |
| edge_intensity | yes | yes | `measure_edge_intensity` | 60 | 316 |
| intensity_moments | yes | yes | `measure_intensity_moments` | 88 | 1,591 |
| radial_distribution (4 rings) | yes | yes | `measure_radial_distribution` | 144 | 681 |
| shape_extended | yes | yes | `measure_shape_extended` | 34 | 825 |
| texture_haralick | yes | yes | `measure_texture` | 156 | 1,214 |
| texture_pftas | yes | yes | `measure_texture` | 648 | 1,405 |
| texture_lbp | yes | yes | `_lbp_frames` | 336 | 2,845 |
| texture_gabor | yes | yes | `_gabor_frames` | 288 | 9,394 |
| periphery (profile family) | yes | yes | `measure_periphery` | 96 | 859 |
| outside (profile family) | yes | yes | `measure_outside` | 99 | 819 |
| colocalisation | yes | yes | `measure_colocalisation` | 108 | 6,892 |
| colocalisation_extra | yes | yes | `measure_colocalisation_extra(costes=False)` | 78 | 5,488 |
| colocalisation_costes | yes | yes | `measure_costes` | 36 | 12,730 |
| neighbours | yes | yes | `measure_neighbours` / `_spatial_measurements` | 10 | 984 |
| granularity | yes | yes | `_granularity_frames` | not in B2 | 58,900 (item 941 note) |
| coloc_block (Pearson, Manders m1/m2, overlap) | yes (`calculate_correlation`) | only as a side effect of the `colocalisation` family | `_calculate_correlation_object_level` | 72 | 922 |
| legacy_radial (6 bins, `radial_dist`) | yes, as `nucleus_rad_dist_*` | no | `_calculate_radial_distribution`, `_create_dataframe` | 24 | 197 |
| cytoskeleton | **no per-cell path** (only whole-image `_image_fibres`) | no | `_analyze_cytoskeleton` | 4 (GFP, AF750) | 422 |
| spatial (`spatial_measurements`) | yes | yes, through `neighbours` | `_spatial_measurements` | 5 | 209 |
| legacy periphery / outside intensity (`periphery=True, outside=True` in `measure_crop`) | yes | no | `_periphery_intensity`, `_outside_intensity` | not in B2 | not measured |
| object distances, bystanders, vacuole counts | yes | no | `_morphological_measurements` options | not in B2 | not measured |
| organelle / pathogen summaries per cell | yes | yes | `_summarize_organelles_per_parent` | not in B2 | not measured |

Findings:
- `spatial` is not a missing block. The 5 B2 `cell_spatial_*` columns equal the `neighbours` columns value for value on the B2 fields (checked on site 1001: all 5 duplicates). OPS reaches it through `neighbours`; no new code.
- `coloc_block` is computed by the same function in both apps, but OPS can only switch it on together with the slow `colocalisation` family (`_measurement_settings`: `calculate_correlation = 'colocalisation' in families`).
- `legacy_radial` is Measure only. Its Measure naming is `nucleus_rad_dist_channel_<c>_bin_<b>` (the nucleus as the object, inside its cell); B2's prototype named the same values `cell_rad_dist_<ch>_bin_<b>`.
- `cytoskeleton` has no per-cell path in either app. B2 used a prototype crop loop over `_analyze_cytoskeleton`.
- Funk features differ by one compartment: Measure writes them for the cytoplasm, OPS does not.
- `_create_dataframe` fills a DataFrame cell by cell (`df.loc`); at 1,248 cells x 24 columns it is the slow part of legacy_radial on the Measure path.

## 2. Brieflow colocalisation set and the spaCR mapping
Source: Brieflow (Di Bernardo et al., Nat Commun 2026, doi 10.1038/s41467-026-73643-7), code at `/mnt/firecuda2/brieflow_run/brieflow`. Our reruns (`run_A1..B2/config/config.yml`) use `phenotype.cp_method: cp_emulator`: `workflow/lib/phenotype/extract_phenotype_cp_emulator.py` with `correlation_features_multichannel` of `workflow/lib/external/cp_emulator.py`, applied to nucleus, cell and cytoplasm, every channel.

Per compartment and channel pair (`combinations` order; slope over `permutations`), on the raw object pixels:

| Brieflow column | Definition (cp_emulator) | spaCR today | Equal? | 1008 |
|---|---|---|---|---|
| `correlation_{a}_{b}` | Pearson, `np.corrcoef` | Funk `correlation_<a>_<b>` (`_funk_object_columns`); `Pearson_correlation` of coloc_block | yes | reuse Funk (and coloc_block) |
| `lstsq_slope_{a}_{b}`, `lstsq_slope_{b}_{a}` | least-squares slope of `b` on `a` (`np.linalg.lstsq([a, 1], b)`), both directions | none | missing | new |
| `overlap_{a}_{b}` | `sum(AB)/sqrt(sum(A^2) sum(B^2))`, no threshold | `manders_overlap_coefficient` on background-subtracted values (median + 3 x 1.4826 MAD) | no | new (raw) |
| `K_{a}_{b}`, `K_{b}_{a}` | `sum(AB)/sum(A^2)`, `sum(AB)/sum(B^2)`, no threshold | `k1`, `k2` in `colocalisation` (slow), background-subtracted | no | new (raw) |
| `manders_{a}_{b}`, `manders_{b}_{a}` | `sum(A[B > otsu(B)])/sum(A)` and mirror; mahotas Otsu per object | `manders_m1/m2` (MAD threshold, subtracted); `costes_m1/m2` (slow) | no | new (Otsu) |
| `rwc_{a}_{b}`, `rwc_{b}_{a}` | Singan 2011: dense ranks, `R = max rank`, `w = (R - |rA - rB|)/R`, `sum(A w [B > otsu(B)])/sum(A)` | `rwc1/rwc2` in `colocalisation` (slow), MAD threshold, subtracted | no | new (Otsu) |

Brieflow writes 10 columns per pair and compartment: 6 pairs x 10 x 3 compartments = **180 columns**. It returns NaN for all 7 colocalisation values when either channel sums to 0 in the object.

Chosen set for the final run (fast only):
- `colocalisation_otsu` (new profile family, below): 9 columns per pair, 162 columns. Together with the Funk Pearson columns it reproduces all 180 Brieflow colocalisation columns. Expected cost below 1,000 CPU-s per 100k cells (vectorised per-label sums, one sort per channel and compartment; measure on the WS).
- `coloc_block` (cheap extra, 922 CPU-s per 100k cells, in B2): spaCR's MAD-background Manders m1/m2 and overlap, 54 non-duplicate columns (its Pearson repeats Funk's; 18 duplicate columns).
- Left out as slow: `colocalisation` (6,892), `colocalisation_extra` (5,488; also carries the per-channel puncta count and area fraction), `colocalisation_costes` (12,730).

## 3. What to build
### 3.1 One registry of per-cell blocks (`spacr/measure.py`)
- New tuple `MEASURE_BLOCK_FAMILIES = ('coloc_block', 'legacy_radial', 'cytoskeleton')`: names accepted by `profile_feature_families` alongside `PROFILE_FEATURE_FAMILIES`. `'profiling'` and `'all'` do NOT include them; nothing changes for existing settings.
- Add `'colocalisation_otsu'` to `PROFILE_FEATURE_FAMILIES` (after `colocalisation_costes`), NOT to `PROFILE_FEATURE_DEFAULT`. Note: `'all'` then includes it (+162 columns, cheap); the 941 note and any `'all'` column-count test are updated.
- `_intensity_measurements` reads the blocks from the family list: `coloc_block` sets `calculate_correlation` on, `legacy_radial` sets `radial_dist` on, `cytoskeleton` runs the new per-cell cytoskeleton block. The old settings (`calculate_correlation`, `radial_dist`) keep working as aliases; if both are given the block is computed once.
- `_measurement_settings` (ops_sample) keeps deriving `calculate_correlation` from `'colocalisation'`, so existing OPS runs with `colocalisation` keep their columns; `coloc_block` is the new, separate switch.
- `_create_dataframe` is rewritten vectorised (one DataFrame from dicts); same columns, same values, same `cell_id`.

### 3.2 New family `colocalisation_otsu` (`measure_colocalisation_otsu(labels, image)`)
- Returns `label` and, per pair `i < j`, `channel_<i>_channel_<j>_<stat>` for `lstsq_slope_ij` (slope of j on i), `lstsq_slope_ji`, `raw_overlap`, `raw_k1`, `raw_k2`, `otsu_m1`, `otsu_m2`, `otsu_rwc1`, `otsu_rwc2`, with Brieflow's definitions above on the raw pixel values.
- Otsu threshold per object and channel equal to `mahotas.thresholding.otsu` on integer-valued data: the observed value `v` that maximises the between-class variance of `{x <= v}` vs `{x > v}`, first maximum wins; 0 when the object has one distinct value. Works on any real values (non-integer images: the same rule on the observed values).
- Dense ranks within the object per channel, `R` = the larger of the two maxima (reuse `_sorted_by_label` / `_within_label_ranks` if they give dense ranks; otherwise a new helper).
- spaCR value rules: 0.0 (not NaN) where Brieflow gives NaN (a channel sums to 0, or zero variance for the slope), because `filter_dataframe_features` drops any column with a NaN. Documented in the docstring and in the mapping table of the docs.
- Vectorised with `np.bincount` over the label index; no Python loop per object except where unavoidable.

### 3.3 Per-cell cytoskeleton block (`measure_cytoskeleton(labels, image, channels)`)
- Per object and listed channel: `channel_<c>_skeleton_length`, `channel_<c>_skeleton_branch_points`, by `_analyze_cytoskeleton` on the object's bounding box grown by 24 px (clipped to the field), other labels zeroed. 24 px exceeds the reach of `threshold_local(block_size=35)` (Gaussian, sigma 5.67, truncate 4), so values equal a whole-field `_analyze_cytoskeleton(image, labels == k)` call exactly. (B2's prototype used the unpadded box; values differ only at the box border.)
- Cell compartment only (`cell_` prefix), as B2.
- Channels from the new settings `cytoskeleton_channels` (Measure) / `ops_ph_cytoskeleton_channels` (OPS), mapped in `OPS_TO_MEASURE_KEYS`: channel indices, or names resolved against `channel_names`; `None` with `cytoskeleton` in the family list means every channel.

### 3.4 OPS wiring (`spacr/ops_sample.py`, `spacr/ops_settings.py`, `spacr/ops_engine.py` ph_measure only)
- `measure_phenotype_field` gains `cytoskeleton_channels=None` and passes `profile_features` (with the blocks) through `_measurement_settings` to the same `_intensity_measurements` call.
- Funk features move to `_add_funk_frames` (set `funk_object_features` in the settings dict) so OPS writes them for the cytoplasm too, as Measure does (+11 columns when `funk_features=True`). The OPS names stay as today (`<prefix>_<ch>_mass_displacement`, `<prefix>_correlation_<a>_<b>`, `<prefix>_form_factor`); the 934 ratchet guards this. This changes `funk_features=True` OPS output by adding the 11 cytoplasm columns; nothing else moves.
- `ops_engine.py`: only the ph_measure task dict and the `measure_phenotype_field` call (around lines 5264-5300 and the options builder near 5553 / 5576 / 5820) pass `ops_ph_cytoskeleton_channels`. The 1007 lane owns the PlateWriter call sites in `ops_engine.py`, `ome_zarr.py` and `image_store.py`; do not touch them.
- `OPS_TYPES['ops_ph_profile_features']` already takes a list; `ops_ph_cytoskeleton_channels: (str, list, type(None))`.

### 3.5 Final-run family list (explicit, editable)
No named preset string: the final runs pass the list itself, so the information test (decision 5) can drop a family by deleting a name.
```
ops_ph_texture: True
ops_ph_funk_features: True
ops_ph_zernike: True
ops_ph_profile_features: [edge_intensity, intensity_moments, radial_distribution, shape_extended,
                          texture_haralick, texture_pftas, texture_lbp, periphery, outside, neighbours,
                          colocalisation_otsu, coloc_block, legacy_radial, cytoskeleton]
ops_ph_profile_optional_families: None
ops_ph_cytoskeleton_channels: GFP,AF750
ops_ph_granularity_subsample: 1.0   (unused: granularity is not in the list)
```
- = B2 - granularity - Gabor - the slow colocalisation families (`colocalisation`, `colocalisation_extra`, `colocalisation_costes`) - `spatial` (duplicate of `neighbours`) + `colocalisation_otsu` + Funk cytoplasm. Funk and Zernike were already in B2's base.
- Measure equivalent: `homogeneity: True`, `funk_object_features: True`, `zernike: True`, `profile_features: <the same list>`, `cytoskeleton_channels: [1, 3]` (or the names with `channel_names`).
- A constant `FINAL_RUN_FAMILIES_1008` in `spacr/ops_settings.py` holds the list for the docs and the test only; no code path reads it implicitly.
- Expected columns about 2,640 (B2 2,761 - 288 Gabor - 222 slow colocalisation - 5 spatial + 162 Otsu + 11 Funk cytoplasm + 0 granularity; legacy_radial naming per 3.1).
- Expected cost, from the B2 per-family costs: about 21,900 CPU-s per 100k cells + `colocalisation_otsu` (< 1,000) + cytoskeleton padding = about 23,000 CPU-s per 100k cells, about 290 CPU-s per field of 1,248 cells (B2: 699; Ready set: 142; 'all': about 2,950). About 1,280 fields per well, so about 103 CPU-h per well, about 10.3 h at 10 workers. To be measured on the WS (sites 357 and 1078, 1 worker; peak RAM).

### 3.6 Alpha status
- New settings `cytoskeleton_channels`, `ops_ph_cytoskeleton_channels` and the family names `coloc_block`, `legacy_radial`, `cytoskeleton`, `colocalisation_otsu` are listed under `ALPHA_FEATURES[1008]` in `spacr/settings.py`; the GUI shows the new setting rows only with "Show alpha features". Defaults leave every output unchanged.

## 4. Tests (CPU, synthetic; `tests/test_measure_ops_parity_1008.py`)
- **P1 parity, both directions:** on the 934 fixture field (`_merged_field`) with the final family list, every OPS column is a Measure column with the identical value (extend `_ops_against_measure`), AND every Measure per-cell column of the cell, nucleus and cytoplasm frames from the listed families is an OPS column (reverse ratchet; Measure-only legacy periphery/outside, distances, bystanders are an explicit allow-list).
- **P2 the same cell:** for one chosen cell, the OPS row and the Measure row (database read back from `measure_crop` on a tmp folder) agree on every shared column, `array_equal` with NaN equal.
- **P3 Brieflow equivalence:** `measure_colocalisation_otsu` equals a copy of cp_emulator's `measure_colocalization` + `lstsq_slope_all_multichannel` (vendored as test reference in the test file, uint16 objects) to 1e-12, per pair and compartment; NaN cases give 0.0.
- **P4 Otsu:** the per-object threshold equals `mahotas.thresholding.otsu` on 200 random uint16 objects (skip without mahotas); a constant object gives 0.
- **P5 cytoskeleton:** the padded-crop values equal whole-field `_analyze_cytoskeleton(image, labels == k, c)` per object; names and channel resolution by index and name; a wrong name stops with a clear tr() message.
- **P6 blocks:** `coloc_block` alone gives the `calculate_correlation` columns and nothing else; `legacy_radial` gives the `radial_dist` columns; old settings and family names together compute each block once; `'profiling'` and `'all'` exclude the blocks.
- **P7 defaults unchanged:** default Measure and default OPS output equal dfe6633b5b's (column list and values) on the fixture; `funk_features=True` OPS output only gains the 11 cytoplasm Funk columns.
- **P8 `_create_dataframe`:** the vectorised version equals the old one on the fixture (columns, values, `cell_id`).
- **P9 final list:** `FINAL_RUN_FAMILIES_1008` is accepted by `profile_feature_families`; deleting any one name still runs (no family depends on another); the column count is recorded.
- **P10 settings:** `ops_ph_cytoskeleton_channels` maps to `cytoskeleton_channels` in `OPS_TO_MEASURE_KEYS`; update `test_every_ops_measurement_setting_maps_to_a_measure_setting` (934) and `tests/test_settings_categories.py`.
- Run as well: `tests/test_measure_ops_features_934.py`, the measure and ops_sample tests, the ratchets group (`tools/gate.sh ratchets` or its current equivalent), the docstring-correctness and arrival-contract tests. CPU only, `pytest -n 2`.

## 5. Arrival contract, strings, catalogue
- New documented callables `measure_colocalisation_otsu`, `measure_cytoskeleton` (and any public Otsu helper) ship with their arrival contract in the same commit; locale records via `codex_inbox/tr_1008_task.md` + sources in `/mnt/firecuda2/codex/scratch-docs/codex_inbox/`.
- All user-facing strings through `tr()`; no `#` comments in `spacr/`.
- Setting registries for the new key: `spacr/settings.py` (defaults, types, descriptions, Measurements category), `spacr/ops_settings.py` (`OPS_DEFAULTS`, `OPS_TYPES`, `OPS_TO_MEASURE_KEYS`), `spacr/qt/help_api_index.py`, `spacr/qt/screens/settings_flow_index.py`, `spacr/qt/screens/setting_api_targets.py`, `docs/setting_consumers.json` (rebuild with `tools/build_setting_consumer_map.py`), `en.py` labels and descriptions through the inbox.
- Schema: none expected (columns go to existing per-object tables). If a schema row turns out to be needed, number 954 is reserved; ask the coordinator first.

## 6. Build split (for the WS session or Codex)
- **Part A (design, Claude WS lane):** 3.1 registry + `_measurement_settings` + 3.5 final list constant + tests P1, P2, P6, P7, P9. Files: `spacr/measure.py` (registry functions and `_intensity_measurements` block switches only), `spacr/ops_sample.py`, `spacr/ops_settings.py`, `tests/test_measure_ops_parity_1008.py`.
- **Part B (bounded, Codex):** 3.2 `measure_colocalisation_otsu` + tests P3, P4. File: a new section of `spacr/measure.py` after `measure_costes` (new function and private helpers only) + `tests/test_colocalisation_otsu_1008.py`.
- **Part C (bounded, Codex):** 3.3 `measure_cytoskeleton` + 3.1 `_create_dataframe` vectorisation + tests P5, P8. File: `spacr/measure.py` (new function after `_analyze_cytoskeleton`; `_create_dataframe` body) + `tests/test_cytoskeleton_radial_1008.py`.
- **Part D (bounded, Codex, after A):** new setting rows in every registry of section 5, `ops_engine.py` ph_measure pass-through, P10, inbox task. Files: `spacr/settings.py`, `spacr/ops_engine.py` (ph_measure only), the qt index files, `docs/setting_consumers.json`, `tests/test_settings_categories.py`.
- Order: B and C in parallel; A wires them in; D last. Timing (3.5) on the WS after A-D.

## Open points for the maintainer
- `'all'` gains `colocalisation_otsu` (+162 cheap columns). Alternative: keep it out of `'all'`. Recommended: include.
- OPS `funk_features=True` gains the cytoplasm Funk columns (parity with Measure). Existing OPS tables keep their columns; reruns add 11.
- The final list keeps `coloc_block` as the cheap extra; its Pearson columns duplicate Funk's.

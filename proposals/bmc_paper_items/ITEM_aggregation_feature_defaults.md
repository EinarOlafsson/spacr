# Item spec: a default aggregation for every measured feature when per-cell tables are merged to parent, well, guide or gene level

- **Lifecycle:** future (alpha), behind "Show alpha features".
  - New default methods apply only when the new `aggregation_defaults='per_feature'` is chosen (alpha).
  - With today's settings, every table is byte-identical.
- **Number:** 937 (reserved 2026-11-13).
  - 79 (DONE: `merge_tables.AGGREGATION_RULES`, child → parent roll-up);
  - 41 (working-table merge rules), 130;
  - `ITEM_ops_gene_level_aggregation.md` (gene = median over cells);
  - `ITEM_screen_staging_export_aggregate.md` (well aggregate);
  - `ITEM_measure_ops_features_general.md` (this batch: new Measure features);
  - 928 and 929 (Ready, built on median gene profiles).
- **Overlap with existing specs:**
  - Item 79 already defines per-kind rules, but **only for the child → parent roll-up**, and its rules predate the profiling families.
  - `ITEM_ops_gene_level_aggregation.md` fixes the gene-level statistic (median) for the OPS profile path and nothing else.
  - Neither covers well, guide or gene defaults per feature. This is filed as a new item that extends 79's registry. If the coordinator prefers, the "roll-up fixes" part (Part B) can go to item 79 as an addendum.
- **Rule:**
  - Default output is byte-identical to today at every level.
  - With `per_feature`, every numeric column resolves to an explicit rule; nothing falls through to the default unnoticed.
  - The OPS profile path stays median unless the maintainer decides otherwise, because the Ready proofs of 928 and 929 rest on it.
  - Every new documented callable ships with its arrival contract and nine-locale records in the same commit.
- **Code basis:** origin/nightly (46f259ea23).
- **Paper-critical:** no for OPS profiles, which are unchanged. Yes for any per-well summary table in the paper that includes counts, fractions or flags.

## Problem and goal
When spaCR merges per-cell tables into one row per well, guide or gene, it uses one statistic for every column: mean, or median. That is wrong for several kinds of feature:
- **counts** should add up (or be averaged per cell), not be medianed;
- a **yes/no flag** medianed becomes 0 or 1 instead of a fraction;
- an **angle** averaged arithmetically is wrong where it wraps around (179° and −179° average to 0°, not 180°);
- **identifiers and coordinates** should not be averaged at all.

The new measurement families (profile families, Funk features) added many columns that no rule names.

**Goal:**
- one registry that gives every feature a sensible default per aggregation level, with the reason;
- every aggregation path consults it;
- a run report lists which method each column got.

## Current state (verified on origin/nightly)

### Aggregation code paths
| Level | Function | Method today | Per-column? |
|---|---|---|---|
| child → parent (nucleus, pathogen, cytoplasm onto cell) | `merge_tables.roll_up` (`merge_tables.py:542`) → `aggregation_plan` (l. 232) → `aggregation_for` (l. 206) | ordered regex rules `AGGREGATION_RULES` (l. 75; first match wins). `DEFAULT_AGGREGATION = MEAN` (l. 97); text takes FIRST; overrides via `aggregation_overrides` (l. 529) | yes (regex) |
| same, many databases | `plate_measurements.py` (reuses the rules; `default_aggregated_columns`, l. 417, lists fall-through columns) | as above | yes |
| same, older path | `io._read_and_join_tables` (`io.py:4841`; aggregation at about l. 5031) | `'mean'` for every numeric child column, `'first'` otherwise, plus `count_<entity>` | **no**; disagrees with `roll_up` |
| cell → well | `screen_staging.aggregate_wells` (`screen_staging.py:475`) | `wells_mean` and `wells_median` tables, each one method for all columns; only `cell_`, `nucleus_`, `cytoplasm_` and `pathogen_` columns (`FEATURE_PREFIXES`, l. 77), others dropped; heterogeneity std and p90–p10 | no |
| cell → well (embeddings) | `embedding_profiles.aggregate_embedding_wells` (l. 473) → `aggregate_wells` (`embedding_aggregate='well_median,well_mean'`) | median or mean | no |
| cell → well (QC) | `assay_report.py:910-922` | counts summed, listed columns medianed | hand list |
| cell → guide → gene (OPS) | `gene_measurement_compare.aggregate_guides` (l. 1293) and `aggregate_genes` (l. 1326), called from `ops_engine._ph_aggregate` (l. 4340) and `ops_tvn.py` | `.median()` for every feature. `n_cells`, `n_guides` and `mitotic_cells` are summed; `mitotic_fraction` and `mitotic_index` are ratios. Features are pre-filtered by `ops_engine._ph_feature_columns` (l. 4070), which drops names containing `centroid`, `bbox`, `coords`, `orientation` or `label` | no |
| profiles | `ops_tvn` recipe (l. 1153-1163) | median of PCs per guide or gene | no (PCs, not features) |

- `feature_dict.py` has `FEATURE_FAMILIES` (l. 117) and `parse_column`, which give the family per column, with **no aggregation field**.
- `measurement_schema.py` holds stamp columns only.

### What `AGGREGATION_RULES` does with the new and existing features
Checked by evaluating the rules on origin/nightly against the column names; DEFAULT means it fell through to mean.

| Column (example) | Roll-up today | Problem |
|---|---|---|
| `*_puncta_area_fraction` | **sum** (matches `area`) | a fraction summed; wrong |
| `n_pathogens`, `n_vacuoles` | DEFAULT mean | counts; should be summed |
| `neighbors_within_<r>`, `touching_neighbors` | DEFAULT mean | per-object counts; acceptable at roll-up, but not named |
| `orientation` | mean (named) | angle; arithmetic mean wrong |
| `is_bystander`, `is_distal`, `mitotic`, `touches_field_edge`, `owned` | DEFAULT mean | gives a fraction (acceptable) but unnamed |
| `form_factor`, `compactness`, `percent_touching` | DEFAULT mean | fine, but unnamed |
| `frac_at_d_<k>`, `radial_cv_<k>` | DEFAULT mean | fine, but unnamed |
| `haralick_*`, `pftas_*`, `lbp_*`, `granularity_*`, `gabor_*` (via `_mean`) | DEFAULT mean / mean | fine, but unnamed |
| `spearman`, `icq`, `k1`, `k2`, `rwc1`, `rwc2`, `nmi`, `costes_m1/m2`, `manders_*`, Funk `correlation_<a>_<b>` | DEFAULT mean | correlation coefficients; unnamed |
| `cv_intensity`, `gini_intensity`, `mode_intensity`, `mass_displacement` | DEFAULT mean | unnamed |
| `euler_number` | sum (matches `number`) | **correct by luck**: Euler number is additive over disjoint objects |
| `max_radius` | max | fine |
| `integrated_edge` | sum | fine at roll-up |
| `area_bbox` | sum | questionable: the bbox of a union is not the sum of bboxes |
| `centroid_*`, `bbox*` | mean | fine at roll-up (child centre); must be dropped at well level and above |

**At well, guide and gene level, every one of these gets the same median (or mean), and the rules are never consulted.**

## Proposed design

### Part A: level-aware default registry
- New module data in `spacr/merge_tables.py`: `SUMMARY_AGGREGATION_RULES`, an ordered `(regex, method, reason)` table for cell → well, guide or gene. `AGGREGATION_RULES` stays the child → parent table.
- `aggregation_for(column, *, level='parent' | 'summary', numeric, family=None)` consults the right table. When a column matches no rule:
  - the family from `feature_dict.parse_column` is used;
  - failing that, the level's default applies, and the column is listed as `unruled` in the plan.
- New public function `summary_aggregation_plan(frame, *, overrides=None) -> DataFrame`: columns `column`, `method`, `rule`, `reason`, `family`.

### Proposed summary defaults (cell → well, guide, gene) with reasons
| Kind | Columns (patterns) | Default | Reason |
|---|---|---|---|
| Identifiers and provenance | `*_label`, `object_label`, `cell_id`, `prcf*`, `plateID`, `rowID`, `columnID`, `fieldID`, `path_name`, stamp columns (`measurement_units`, `n_z`, `voxel_size_*`) | **drop**, or **first** for stamps, with an error if a stamp is not constant within the group | an ID has no mean. A non-constant stamp means mixed units in one row |
| Position | `centroid*`, `bbox*`, `coords*`, `distance_to_field_edge` | **drop** | position in a field is an acquisition artefact, not a phenotype |
| Intensity, location | `*_mean_intensity`, `*_median_intensity`, `*_min_intensity`, `*_max_intensity`, `*_integrated_intensity`, `*_mode_intensity`, `percentile_*`, `*_edge` (mean/min/max/integrated), `outside_*`, `periphery_*`, `mean_radius`/`median_radius` | **median** | intensity distributions are skewed, with bright debris and mitotic cells; the median is robust, and it is the standard in image-based profiling (Brieflow, CellProfiler/pycytominer) |
| Intensity, spread and shape of the distribution | `std_intensity`, `iqr`, `range`, `mad`, `cv_intensity`, `gini_intensity`, `skew_*`, `kurtosis_*`, `entropy`, `radial_cv_*` | **median** | same; these are per-cell statistics to summarise, not to pool |
| Size and shape | `area*`, `perimeter`, `*_axis_length`, `equivalent_diameter*`, `feret*`, `max_radius`, `volume*`, `euler_number`, `hu_*`, `weighted_hu_*`, `zernike_*`, `mass_displacement` | **median** | per-cell values; robust |
| Fractions and ratios bounded in [0, 1] | `eccentricity`, `solidity`, `extent`, `form_factor`, `compactness`, `*_fraction`, `frac_at_d_*`, `mean_frac_*`, `percent_touching`, `manders_*`, `costes_m1/m2`, `rwc1/2`, `k1/k2`, `overlap_coefficient`, `*_overlap_fraction` | **mean** | the mean of per-cell fractions is the expected fraction per cell; bounded values carry no heavy outliers. A median of bimodal fractions (many 0s) hides the effect |
| Flags (bool or 0/1) | `is_*`, `mitotic`, `touches_field_edge`, `owned`, `monolayer_ok`, `infected` | **mean**, written as `<col>_fraction` | the share of cells with the flag. The median gives 0 or 1 |
| Counts per cell | `n_pathogens`, `n_vacuoles`, `pathogen_count`, `puncta_count`, `maxima_count`, `*_count`, `skeleton_branch_points` | **mean** (per-cell average), **plus `<col>_sum`** (total) | the maintainer asked for sum. A sum alone scales with the number of cells in the well, so wells with more cells look different. Writing both keeps the total and gives a per-cell figure comparable between wells. `n_cells` is always written |
| Neighbourhood counts | `neighbors_within_<r>`, `touching_neighbors` | **mean** (no sum) | each neighbour pair is counted once from each side, so a sum double-counts; the mean is local density |
| Distances | `nearest_neighbor_distance`, `second_neighbor_distance`, `distance_to_*`, `centre_to_*`, `relative_radial_position` | **median** | skewed; robust |
| Correlation coefficients | `Pearson_correlation`, `correlation_<a>_<b>`, `spearman`, `icq`, `nmi`, `M1_*`, `M2_*` | **median** | a Fisher-z mean is the textbook choice but breaks at r = ±1, which small objects often give; the median needs no transform. `fisher_z_mean` is offered as an override |
| Texture | `haralick_*`, `pftas_*`, `lbp_*`, `gabor_*`, `granularity_*`, `homogeneity_*`, `blur` | **median** | per-cell statistics; robust |
| Angles | `orientation` (regionprops, radians in [−π/2, π/2]; axial, period π) | **axial circular mean**: θ̄ = ½·atan2(mean sin 2θ, mean cos 2θ), plus `<col>_coherence` = resultant length of 2θ | an axis at +89° and one at −89° are 2° apart; an arithmetic mean gives 0°. The coherence says whether there is a common direction at all |
| Embeddings | `emb_*` | **median** (unchanged) | as `embedding_aggregate` today |
| Anything unruled | | **median**, listed as `unruled` | a robust default. The run report shows the list, so a new feature without a rule is visible |

### Part B: roll-up (child → parent) corrections to item 79's rules
- `*_fraction` (including `puncta_area_fraction`) → **mean**. A new rule placed before the `area` rule.
- `n_pathogens`, `n_vacuoles`, `n_<child>` → **sum** (extend the count rule with `^n_` and `(^|_)n_[a-z]+$`).
- Flags `is_*` → **mean** (named; result unchanged).
- `orientation` → **axial circular mean** (new method `circmean_axial` in `EXPLICIT_AGGREGATIONS`).
- `area_bbox` → **max** (open question 3).
- Named rules for every profile family listed above, so `default_aggregated_columns` is empty on a profiling run.

These change roll-up results **only** under `aggregation_defaults='per_feature'`. The legacy table is kept verbatim as `AGGREGATION_RULES_V1`.

### Part C: consumers
- `screen_staging.aggregate_wells` gets `levels` value `'default'` (alongside `'mean'` and `'median'`). It writes `wells_default` from the registry, and keeps non-prefixed count and flag columns (`n_pathogens`, `is_bystander`), which it drops today.
- `gene_measurement_compare.aggregate_guides` and `aggregate_genes` get `method='median' | 'per_feature'`, default `'median'`.
- `io._read_and_join_tables` calls `merge_tables.roll_up`, so the two roll-up paths agree.
  - This changes its output only under `per_feature`; the legacy branch keeps `'mean'`.
  - A test pins the legacy output.
- `assay_report.py:910-922` takes its methods from the registry; the output is unchanged for its listed columns.
- Each run writes `aggregation_plan.csv` beside the aggregated tables.

### Settings (alpha)
| Key | Type | Default | Meaning |
|---|---|---|---|
| `aggregation_defaults` | str | `'legacy'` | `'per_feature'` uses the registry at every level |
| `aggregation_overrides` | dict | {} | column regex → method; wins over the registry |
| `aggregation_drop_positions` | bool | True | under `per_feature`, drop IDs and positions at summary level |
| `aggregation_count_sum` | bool | True | also write `<col>_sum` for counts |
| `aggregation_report` | bool | True | write `aggregation_plan.csv` |
| `ops_aggregate_method` | str | `'median'` | OPS guide and gene profiles: `'median'` (today; the 928/929 proofs) or `'per_feature'` |

### Public callables
- `spacr.merge_tables.summary_aggregation_plan`
- `spacr.merge_tables.aggregation_for` (extended with `level`)
- `spacr.merge_tables.circmean_axial(values) -> (angle, coherence)`
- `spacr.merge_tables.SUMMARY_AGGREGATION_RULES` (documented module data)

### GUI
- Measure analysis and the merge editor (item 79's typed custom merge editor) gain `aggregation_defaults` and a read-only "Aggregation plan" table (column, method, reason). It is editable through `aggregation_overrides`.
- The OPS panel gains `ops_aggregate_method` under alpha.

## Acceptance tests (CPU; `CUDA_VISIBLE_DEVICES=''`; at most 4 test workers)
1. **Legacy identity.** With default settings these are byte-identical to today on the bundled example plate and on a synthetic OPS plate:
   - `roll_up`, `aggregate_wells` (mean and median), `aggregate_guides`, `aggregate_genes`, `_read_and_join_tables`;
   - OPS `ops_guides_<plate>` and `ops_genes_<plate>`.
2. **928/929 unaffected.** With `ops_aggregate_method='median'`, the 928 and 929 proof (`spacr_stage5_proof/stage5_proof_result.json`) reproduces: 80/80 partitions, gates ≤ 1.8e-15.
3. **Every feature has a rule.** Measure runs on a synthetic plate with every family on (`profile_features='all'`, `funk_object_features`, `radial_dist`, `spatial_measurements`, `bystander_measurements`, Zernike, neighbours, vacuoles), followed by the OPS phenotype path on a synthetic field. With `per_feature`:
   - the plan has **no** `unruled` numeric column;
   - as a ratchet, a new unruled column fails the test and names the column.
4. **Method correctness on planted data.**
   - Flags: 30 % true gives `_fraction` 0.30.
   - Counts: per-cell 0, 1, 2 gives mean 1 and sum 3.
   - Angles: orientations ±(π/2 − 0.01) give a mean of ±π/2 (mod π) and coherence ≈ 1. Uniform orientations give coherence ≈ 0.
   - Fractions: mean equals the expected value.
   - Positions are absent from summary tables.
   - A non-constant stamp raises.
   - `puncta_area_fraction` rolled up is a mean, not a sum.
5. **Roll-up paths agree.** Under `per_feature`, `io._read_and_join_tables` and `merge_tables.roll_up` give identical parent tables.
6. **Effect report** (not a gate; the development deck). On the bundled host-pathogen example and on plate A A1, list per column the relative change of the well value between `legacy` and `per_feature`, and the share of columns that changed. For hit calling, report the change in the number of wells beyond 3 MAD from the controls, with the shuffled-label control.
7. **Time** (marked slow). Aggregation time per plate for `legacy` vs `per_feature` on a 384-well synthetic plate with 400 columns. Pre-set: `per_feature` ≤ 1.5 × legacy. Report the CPU seconds.

## Translations / catalog obligations
- `summary_aggregation_plan`, `circmean_axial`, the extended `aggregation_for` and `SUMMARY_AGGREGATION_RULES` (including the `reason` strings, which appear in the GUI plan table) ship in the same commit with:
  - their arrival contracts;
  - nine-locale records;
  - the regenerated help index, API catalogs, setting consumer map and settings flow.
- All new tooltips and GUI labels get nine-locale records.
- Warm the catalogs first.

## Dependencies on items 928–932
- **928 / 929:** OPS profiles keep the median by default (`ops_aggregate_method='median'`), so their Ready status is untouched. Switching OPS to `per_feature` would need a new proof under the READINESS rule.
- **931:** the settings-schema version increments; `aggregation_defaults` gets a row if its default is ever flipped.
- 930 and 932: none.
- Also `ITEM_measure_ops_features_general.md` (this batch): its new columns (`channel_pair_pearson`, Funk mass displacement) must be in the registry in the same commit, or test 3 fails.

## Alpha status
Alpha. `per_feature` and all new keys sit under one `ALPHA_FEATURES` entry. Legacy is the default.

## Maintainer decisions (2026-11-13, AskUserQuestion in the master session)
Counts: per-cell mean as the main column plus a _sum total; correlations: median; area_bbox: max; OPS gene profiles stay median for every feature (Brieflow); per_feature becomes the default for new settings files once tests 3-6 pass, existing files stay legacy through the settings-schema version.

## Open questions for the maintainer (answered above)
1. Counts: you asked for **sum**. The spec writes both the per-cell mean and the `_sum` total, and uses the mean as the main column, because a sum scales with cells per well. Is mean as the main column acceptable, or should the sum be the main column?
2. Correlations: median (proposed), or Fisher-z mean with r clipped to ±0.999?
3. `area_bbox` at roll-up: max, or drop?
4. Should OPS gene profiles ever use `per_feature`? It changes the profile matrix, and would need a new proof against Brieflow under the fair-comparison rule. Brieflow uses the median for every feature.
5. Should the default flip to `per_feature` for new settings files once tests 3–6 pass, with existing settings files kept on legacy through the schema version?

# Item spec: plaque figure mode: absolute and relative plaque size (to the panel's largest value, to the well extent, to the reference well, within-well spread) and agreement with the sizes papers report

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Related:
  - `ITEM_plaque_measured_vs_reported.md`: public `segment_wells` / `resolve_plaque_model`, `measured_by_condition`, `compare_reported` (Spearman ρ on log2 ratios to the reference condition, permutation control), `cond_match`, well merge. **This item extends it and must be built on top of it:** it adds scales and an agreement table beside the log2-to-reference effect, and reuses its segmentation entry, condition matcher, condition labels and reference-condition setting. Nothing in that spec is redefined here. The "relative to the reference well" scale below is the same log2 ratio that `compare_reported` scores; this item only adds R² and the scatter for it, from the same rows.
  - 843–849 (plaque figure mode, open, alpha): 844 (reported values table: metric, statistic, unit, value per condition, panel) is the input of every scale here; 846 (per-figure effect table) gains the extra scale columns.
  - 500 (well diameter, scale and time) and 501 (calibrated growth): the only route to mm²; not used by default, because published plaque figures rarely state a well diameter or a scale.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.

## Why
Papers report plaque readouts in different units (counts, mm², arbitrary units, pixels, percent of a control), on the one representative well shown in the figure or on many wells. Comparing plaque sizes measured from a published figure with the values the paper reports therefore needs more than raw values. The analysis is not yet in spaCR; it compares measured and reported values on four scales:
- **absolute:** plaques per well, published count vs spaCR count, only for readouts whose unit is a count (only counts share a unit across papers);
- **relative to the largest value in the same figure panel:** published and spaCR value each divided by the panel maximum;
- **relative to well extent:** spaCR mean plaque area / well box area (px²), then divided by the panel maximum, against the published value relative to its panel maximum (area readouts);
- **relative to the reference well:** log2 (condition / reference condition), ratios floored at 0.02.

For area readouts it also reports the largest plaque area in the well divided by the mean plaque area (a within-well size spread). `ITEM_plaque_measured_vs_reported.md` covers only the reference-well scale, and scores it with Spearman ρ, not R².

## Reference implementation
- **`comp.py`** → `cond.csv` (one row per condition: figure, pmcid, panel, metric, unit, strain, treatment, is_ref, reported, n_wells, spacr_n, spacr_mean_area_px2, spacr_max_area_px2, well_box_px2, well_box_px):
  - published values: the manual (gold) annotations' quantification blocks with metric in plaque_area / plaque_number / relative_plaque_size;
  - spaCR values: one stored segmentation JSON per well box (`n`, `areas` px², `bbox` of the well box, `cleared_frac`, model plaque v8);
  - conditions matched with `score_lib.cond_match` on the manual box labels; several boxes for one condition → mean over boxes; no plaques → area 0.
- **`build` (figure and statistics step):**
  - spaCR value = count for plaque_number, mean plaque area otherwise; `spacr_well` = value / `well_box_px2` (area readouts only);
  - grouped by (pmcid, panel): `rep_rel`, `sp_rel`, `sp_wellrel` = value / group maximum (NaN when the maximum is 0);
  - reference-well scale from `results_measured_vs_reported.csv`, rows with `labels == 'gold'`, log2 of `reported_ratio` / `measured_ratio` floored at 0.02;
  - R² = squared Pearson r (`scipy.stats.pearsonr`), p, n, figures per scale; NaN when n < 3;
  - per-condition table with "spaCR largest/mean".
- **Outputs:** `cond.csv`, a scale-statistics table (scale, R², p, n, figures), a per-condition table of published and spaCR values with their relative forms and "largest/mean", one four-panel scatter figure.

## What to build
In `spacr.plaque_papers` (alpha), run from the plaque figure-mode settings file after `measured_by_condition` (from `ITEM_plaque_measured_vs_reported.md`):
1. **Per-well size readouts** (added to the per-well table of figure mode and to the plaque module's per-well table, so they also exist outside figure mode):
   - `plaque_count`, `plaque_mean_area_px2`, `plaque_median_area_px2`, `plaque_max_area_px2` (0 when no plaques), `well_extent_px2` (well box width × height) and `well_extent_px` (mean of width and height);
   - `plaque_area_rel_well_extent` = mean area / well extent;
   - within-well spread: `plaque_max_over_mean` and, per plaque, `area_rel_well_max` (each plaque / the largest plaque in the same well) in the per-plaque table;
   - `plaque_area_mm2` only when `plaque_well_diameter_mm` (item 500) is set; otherwise absent, and the report says why (no scale).
2. **Per-condition table** `conditions_measured_vs_reported.csv` with the `cond.csv` columns, built from the 844 reported values and the matched wells:
   - `figure_condition_aggregate` ('mean' | 'median'; default 'mean'): over boxes sharing a condition;
   - `figure_size_stat` ('mean' | 'median'; default 'mean'): which per-well area is "plaque size".
3. **Scales**, `figure_scales` (list; default all four):
   - 'absolute': only readouts whose reported unit is in `figure_absolute_units` (default ['count']);
   - 'rel_panel_max': value / maximum of its (figure, panel) over conditions with both values;
   - 'rel_well_extent': area readouts only; spaCR area / well extent, then / panel maximum; the published side is rel_panel_max;
   - 'rel_reference': log2 (condition / reference condition), floor `figure_ratio_floor` (0.02), reference from `figure_reference_condition` (shared with the measured-vs-reported item; same rows as `compare_reported`).
4. **Agreement table** `agreement_by_scale.csv`: per scale, R² (squared Pearson r), p, n, figures, plus Spearman ρ, Lin's concordance and the median log2 (spaCR / published) bias; NaN when n < 3. Overall and per readout (number, area, relative size).
5. **Controls** (new, not in the reference; the reference values stay identical):
   - within-panel permutation null for every relative scale: spaCR values permuted among the conditions of each panel, `figure_scale_permutations` (1000), `figure_scale_seed` (0) → null R² median and 95th percentile. Normalising both sides by the panel maximum puts one point per panel at (1, 1) for free, and this null measures that inflation;
   - the same agreement with each panel's maximum-defining conditions left out (`figure_scale_drop_panel_max`, reported beside, not instead);
   - identity line on every panel.
6. **Figure:** four scatter panels (absolute, rel. panel max, rel. well extent, log2 vs reference), colour by readout (teal number, blue area, navy relative size; ~60 % opacity), identity dashed, R² and n in the corner, null band from item 5 (spaCR palette, plain theme).

## Validation data and acceptance
- **Data:** the gold figure set with manual condition labels and published quantification panels, the stored per-well segmentations (plaque v8), and `results_measured_vs_reported.csv`.
- **Acceptance:** identical output to the reference outputs:
  - from the stored segmentations: `conditions_measured_vs_reported.csv` identical to the reference `cond.csv` (same rows; floats ≤ 1e-12);
  - `agreement_by_scale.csv` R², p, n and figures per scale identical to the reference scale-statistics table;
  - the per-condition relative values and "largest/mean" identical to the reference per-condition table;
  - the rel_reference rows identical to the gold rows `compare_reported` produces (same rows, same log2 values), so the two items cannot drift apart;
  - permutation null: identical on rerun with the same seed.
- **End to end:** masks from `segment_wells` on nightly with the same model; per-well areas rtol 1e-3, counts exact (the tolerance of the measured-vs-reported item). When a newer plaque model replaces v8, rerun both items and keep the new reference with its commit and model hash.
- **Synthetic test** (unit test, CPU): planted per-condition sizes with a lab-specific unit scale and one well per condition: absolute R² low, rel_panel_max and rel_reference recover the planted agreement; with spaCR values shuffled within panels the agreement falls to the null band.

## Literature
- *Toxoplasma* plaque assay: Roos et al. 1994 (Methods Cell Biol).
- Plaque quantification tools: ViralPlaque (Cacciabue et al. 2019, MethodsX); Plaque Size Tool (Trofimova & Jaworski 2021, BMC Microbiol).
- Agreement beyond correlation: Bland & Altman 1986 (Lancet); concordance correlation (Lin 1989, Biometrics).
- Segmentation: Cellpose-SAM (Pachitariu et al. 2025, bioRxiv).

## Effort
- **Effort:** S–M (after `ITEM_plaque_measured_vs_reported.md`; mostly table logic and one figure).

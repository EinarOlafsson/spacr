# Item spec: guide-efficacy (JACKS-style) gene profile, blended with the all-cell gene median (X04)

- **Lifecycle:** future (alpha), behind "Show alpha features" like every new feature.
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** the spaCR settings-file run must reproduce the reference outputs identically (same rows, same values to float32 precision; the method has no random step). Each documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** no (supplement-only strategy).

## Reference implementation
`/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/strategies_xfield/scripts/xf_methods.py`: `efficacy_profiles`, `x04_jacks(mode="blend")`, `s02_transform`, `gene_index`. The row export (the profile rows whose cosine is the arm's similarity) is in `scripts/export_rows.py`. It builds on the contrastive weighting (S02, `ITEM_ops_contrastive_weighting.md`) and the all-cell gene-level aggregation (`ITEM_ops_gene_level_aggregation.md`).

## What it does
The method runs after S02, in the S02-weighted space. Gene rows Z are the all-cell gene medians; guide rows Zg are the guide medians.
1. Per gene, fit the rank-1 guide model of JACKS (Allen et al. 2019 Genome Res, doi:10.1101/gr.238923.118), with the features in the role of JACKS' screens: x_g = e_g m + noise, noise variance proportional to 1/n_cells(g).
   - Alternate m = sum_g w_g e_g x_g / sum_g w_g e_g^2 (w_g = n_cells) and e_g = clip(<x_g, m> / <m, m>, 0, 1).
   - After each update, rescale so that max e_g = 1.
   - Initialise e to 1. Run at most 20 iterations, and stop when |de| < 1e-6.
2. The efficacy-weighted gene profile is M = sum_g w_g e_g x_g / sum_g w_g e_g^2. NT rows (no guide group) keep Z.
3. Output gene row = Z/|Z| + M/|M|. The similarity is the cosine of these rows.

The method uses no reference labels.

## spaCR home
`spacr.gene_measurement_compare.guide_efficacy_profiles(genes, guides, n_cells)`, called after `contrastive_weighting`.

## Settings
`ops_guide_efficacy` (null | 'jacks_blend'), `ops_guide_efficacy_iters` (20), `ops_guide_efficacy_emax` (1.0)

## Validation data
- Inputs: `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/external/X_bf_spacrcells.parquet` and `_guides.parquet` (via strategies_lit `sl_common.base_rows("BF", "full")`).
- Reference output rows: `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/strategies_xfield/modules/eval/arms/X04_jacks_blend.npz` (names, rows). Their cosine reproduces the scored arm exactly.
- Reference scores: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/strategies_xfield/tables/full_SELECT_arms.csv` and `full_REPORT_arms.csv` (row `X04_jacks_blend`).

## Acceptance
Output must be identical to the reference outputs above:
- rows equal to `X04_jacks_blend.npz` (float32);
- retrieval scores equal to the `X04_jacks_blend` rows of `full_SELECT_arms.csv` and `full_REPORT_arms.csv` (both halves);
- the efficacy vector e identical per guide.

## Tests (CPU)
- **Planted efficacy.** On synthetic guides x_g = e_g m with known e in [0, 1] and max e = 1, the fitted e matches the planted e within 1e-6.
- **NT rows.** NT rows keep Z unchanged.
- **Refusal.** Like S02, the function refuses when fewer than 2 S02 directions are kept.
- **Stopping.** The fit stops at `ops_guide_efficacy_iters` or when |de| < 1e-6, whichever comes first.

## Effort
S.

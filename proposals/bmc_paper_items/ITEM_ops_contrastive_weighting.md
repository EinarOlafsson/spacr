# Item spec: contrastive generalised-eigen Wiener weighting (S02) after aggregation

- **Lifecycle:** future (alpha), behind "Show alpha features" like every new feature.
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Porting map: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/PORTING.md`.
- **Rule:** the spaCR settings-file run must reproduce the reference outputs identically (same rows, same values to float32 precision, same random streams: every seed is fixed in the reference code). Each documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** no (supplement-only strategy).

## Reference implementation
`/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/strategies_lit/scripts/strategies.py` `_geneig`, `shrink_cov`, `sl_common.Embedder`; applied by `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/s02_arms.py`.

## What it does
Solve Sigma_genes v = lambda Sigma_NT,LW v (Ledoit–Wolf covariance, Ledoit & Wolf 2004, of centred NT guide medians / mean guides per gene) on the embedded gene rows; weight directions by max((lambda-1)/lambda, 0); guides in the same space. This is a contrastive generalised eigenproblem in the spirit of contrastive PCA (Abid et al. 2018).

MUST refuse when fewer than 2 directions have lambda > 1: on eb_shrink-ed profiles every weight can be 0, and the tie rule then scores AP = 1. Do not stack with 2b shrinkage.

## spaCR home
`spacr.gene_measurement_compare.contrastive_weighting(genes, guides, nt)`

## Settings
`ops_profile_weighting` (null | 'contrastive_wiener')

## Validation data
`/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/external/W_bf_S02.parquet` (+ `_guides`, `.json`), `W_bestnoshr_S02.parquet`.

## Acceptance
Identical output to the reference weighted rows at `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/external/`; the refusal test on the shrunk profile.

## Effort
S.

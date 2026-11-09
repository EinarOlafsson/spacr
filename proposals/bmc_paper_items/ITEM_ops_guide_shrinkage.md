# Item spec: guide-reliability empirical-Bayes shrinkage of gene profiles (upgrade 2b)

- **Lifecycle:** future (alpha), behind "Show alpha features" like every new feature.
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Porting map: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/PORTING.md`.
- **Rule:** the spaCR settings-file run must reproduce the reference outputs identically (same rows, same values to float32 precision, same random streams: every seed is fixed in the reference code). Each documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** no (supplement-only strategy).

## Reference implementation
`/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/ic_lib.py` `eb_shrink`; used by `build_profiles.py` and `spacr_best.py`.

## What it does
Per dimension, guide noise = NT between-guide variance a_d + (pi/2) v_d / n (median of n cells); each gene's per-guide noise pooled between that prior (prior_df = 4) and its own guide spread (K - 1 df); prior signal variance by moments; shrink factor tau2 / (tau2 + V / K) applied to the gene's mean guide profile (toward NT). Standard empirical-Bayes variance moderation (Smyth 2004) and James–Stein-type shrinkage.

## spaCR home
`spacr.gene_measurement_compare.shrink_gene_profiles`

## Settings
`ops_profile_shrinkage` ('eb_nt' | null), `ops_profile_shrinkage_prior_df` (4)

## Validation data
`/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/profiles/guides_base.parquet` -> `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/profiles/genes_shrink.parquet`, `shrink_factor.parquet`, `shrink_params.json`.

## Acceptance
Identical output to the reference at `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/profiles/`: gene table and shrink factors identical; `tests_retrieval.py` check 4 as a spaCR test.

## Effort
S.

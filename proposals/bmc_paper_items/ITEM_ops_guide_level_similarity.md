# Item spec: guide-level gene-gene similarity with calibrated pair p values (upgrade 3b)

- **Lifecycle:** future (alpha), behind "Show alpha features" like every new feature.
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Porting map: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/PORTING.md`.
- **Rule:** the spaCR settings-file run must reproduce the reference outputs identically (same rows, same values to float32 precision, same random streams: every seed is fixed in the reference code). Each documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** no (supplement-only strategy).

## Reference implementation
`/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/ic_lib.py` `unit_mean_guides`; `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/guide_pair_pvalues.py`; `evaluate.py` 'G_' arms.

## What it does
s(A,B) = mean of the k_A x k_B guide cosines in the embedded guide space; for single-guide NT rows normalise the mean vectors (the 'G_' arms do; un-normalised sums let NT rows crowd neighbour lists). p from 200,000 random-guide-set pairs of the same shape; NT-guide null reported; calibration on NT pseudo-genes.

## spaCR home
`spacr.gene_measurement_compare.guide_level_similarity`

## Settings
`ops_profile_similarity` ('guide_level'), `ops_guide_null_size` (200000)

## Validation data
`/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/guide_pairs/` (summary.json, significant_pairs_q05.csv.gz, calibration).

## Acceptance
Identical output to the reference at `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/guide_pairs/`: same pair statistics, p values and calibration summary.

## Effort
S–M.

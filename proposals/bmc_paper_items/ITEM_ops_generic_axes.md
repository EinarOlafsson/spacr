# Item spec: fitness / generic-axis removal (upgrade 2d)

- **Lifecycle:** future (alpha), behind "Show alpha features" like every new feature.
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Porting map: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/PORTING.md`.
- **Rule:** the spaCR settings-file run must reproduce the reference outputs identically (same rows, same values to float32 precision, same random streams: every seed is fixed in the reference code). Each documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** no (supplement-only strategy).

## Reference implementation
`/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/evaluate.py` `generic_removal`.

## What it does
OLS across genes of each profile column on [log cells per guide, G1 fraction shift, M fraction shift] (targeting genes fit, NT rows get the same correction); project out the first singular vector of the lowest-decile (fitness) genes if |rho(score, log cells per guide)| > 0.30.

## spaCR home
`spacr.gene_measurement_compare.remove_generic_axes`

## Settings
`ops_profile_generic_axes` (list), `ops_profile_essential_axis_min_rho` (0.30)

## Validation data
`/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/metrics/C2d_generic_info.json` (variance removed, rho) and `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/arms/C2d_generic.npz`.

## Acceptance
Identical output to the reference at those two paths: residual matrix (before embedding) and the info JSON identical.

## Effort
S.

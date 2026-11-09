# Item spec: responder fraction, responder profiles, sliced Wasserstein and energy distance (upgrade 3a)

- **Lifecycle:** future (alpha), behind "Show alpha features" like every new feature.
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Porting map: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/PORTING.md`.
- **Rule:** the spaCR settings-file run must reproduce the reference outputs identically (same rows, same values to float32 precision, same random streams: every seed is fixed in the reference code). Each documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** no (supplement-only strategy).

## Reference implementation
`/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/build_profiles.py` (3a block).

## What it does
Per gene vs NT cells: 5-fold cross-fitted mean-difference discriminant in TVN space, threshold = held-out NT 95th percentile, responder fraction = (flagged - 0.05) / 0.95, responder shift = mean(flagged gene cells) - mean(flagged NT cells); energy distance to NT (300 vs 2,000 cells; Székely & Rizzo 2013); sliced-Wasserstein embedding (64 directions x 32 quantiles, 300 cells; Bonneel et al. 2015). The responder fraction is also a per-gene penetrance output.

## spaCR home
`spacr.gene_measurement_compare.responder_profiles`, `distribution_distances`

## Settings
`ops_profile_similarity` ('responder' | 'sliced_wasserstein'), `ops_profile_responder_alpha` (0.05)

## Validation data
`/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/profiles/responder_*.parquet`, `sw_quantiles.npy`, `build_profiles.json` (parameters).

## Acceptance
Identical output to the reference at `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/profiles/`: identical tables with the seeds in build_profiles.py (default_rng(0) stream order).

## Effort
M.

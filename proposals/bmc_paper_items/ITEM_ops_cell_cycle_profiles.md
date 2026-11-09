# Item spec: cell-cycle-resolved gene profiles (upgrade 2c)

- **Lifecycle:** future (alpha), behind "Show alpha features" like every new feature.
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Porting map: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/PORTING.md`.
- **Rule:** the spaCR settings-file run must reproduce the reference outputs identically (same rows, same values to float32 precision, same random streams: every seed is fixed in the reference code). Each documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.

## Reference implementation
`/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/build_cells.py` (phase call: G1 / S-G2 split at +0.30 log2 above the well's NT G1 mode of flat-field-corrected integrated nuclear DAPI; M = mitotic call), `build_profiles.py` (per-well NT-phase centring, per-phase guide medians, per-phase eb_shrink), `phase_trained.py` (M from the item-856 classifier).

## What it does
Per-phase profiles concatenated [G1 | S-G2 | M]; each phase alone as an option.

## spaCR home
`ops_engine._ph_aggregate`: a `phase` column; `aggregate_guides(..., group='phase')`.

## Settings
`ops_profile_cell_cycle` (null | 'concat' | 'G1' | 'SG2' | 'M'), `ops_profile_g1_sg2_split_log2` (0.30), `ops_mitotic_source` ('rule' | 'classifier')

## Validation data
`/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/cells/tvn_<well>.parquet` (phase, dna_log2_rel) and `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/profiles/guides_phase*.parquet`, `genes_phase*.parquet`.

## Acceptance
Identical output to the reference outputs at those paths: phase calls and per-phase guide/gene tables identical.

## Effort
M.

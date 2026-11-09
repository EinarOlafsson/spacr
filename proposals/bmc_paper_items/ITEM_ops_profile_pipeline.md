# Item spec: one settings-driven profile pipeline: stored cell tables -> profiles -> metrics

- **Lifecycle:** future (alpha), behind "Show alpha features" like every new feature.
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Porting map: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/PORTING.md`.
- **Rule:** the spaCR settings-file run must reproduce the reference outputs identically (same rows, same values to float32 precision, same random streams: every seed is fixed in the reference code). Each documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.

## Reference implementation
`/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/final_pipeline.py`; settings `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/settings/FINAL.json`; runtime per plate in `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/final/<run>/runtime.json`.

## What it does
Recipe (lof / impute / pca / agg / nt) -> TVN (typical variation normalisation, whitening epsilon) -> optional shrinkage, active filter, contrastive weighting -> evaluation. Re-applicable to any set of plates from the stored per-plate cell tables (`cfg.work(P)/cells/raw_<well>.parquet`) without image steps.

## spaCR home
`spacr.ops_engine.run_ops(settings, phases=('profile', 'evaluate'))`

## Settings
The FINAL.json keys: `recipe`, `tvn_whiten_epsilon`, `shrinkage`, `active_filter`, `weighting`, `similarity`, `min_cells`. Pin `ops_tvn_whiten_epsilon` explicitly in every settings file of record.

## Validation data
`/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/final/A_check/` (the plate A reproduction run) and the matching arm files in `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/`.

## Acceptance
Identical output to the reference outputs at `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/final/A_check/`: the spaCR run on plate A gives the same gene table and metrics as `final_pipeline.py --plates A` (float32 precision).

## Effort
M.

# Item spec: gene profiles from frozen single-cell embeddings, and fusion with measurements

- **Lifecycle:** future (alpha), behind "Show alpha features" like every new feature.
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Porting map: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/PORTING.md`.
- **Rule:** the spaCR settings-file run must reproduce the reference outputs identically (same rows, same values to float32 precision, same random streams: every seed is fixed in the reference code). Each documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** no (supplement-only strategy).

## Reference implementation
`/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/aggregate_ogc.py` (memory-bounded, ops_gene_classes export layout), `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/aggregate_embeddings.py` (generic per-well parquet layout).

## What it does
Join index (well, site, label) to the cell table; interphase only; TVN (per-well centre-scale -> PCA 99 % on 100k -> tvn_on_controls) or robust z vs NT -> PCA; guide median -> gene median; the measurement profile on exactly the same cells; fusion = per-block standardise + whitened PCA 99 % + 1/sqrt(dims), concatenated.

## spaCR home
`spacr.embeddings` consumers + `ops_engine` profile phase.

## Settings
`ops_profile_space` ('embedding:<tag>' | 'fusion'), `ops_fusion_blocks`

## Validation data
`/mnt/wd4tb/ops_gene_classes/embeddings/plateA/<model>/` -> `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/external/X_emb_ogc_<model>_<slice>_<norm>.parquet`, `X_meas_ogc_*`, `X_fusion_ogc_*`.

## Acceptance
Identical output to the reference gene tables at `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/external/`.

## Effort
M.

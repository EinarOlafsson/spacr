# Item spec: gene profile = median over all of a gene's cells, NT rows per barcode

- **Lifecycle:** future (alpha), behind "Show alpha features" like every new feature.
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Porting map: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/PORTING.md`.
- **Rule:** the spaCR settings-file run must reproduce the reference outputs identically (same rows, same values to float32 precision, same random streams: every seed is fixed in the reference code). Each documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.

## Reference implementation
`/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/recipe_ablation.py` (`--agg gene --nt barcode`).

## What it does
Gene profile = median over all cells of the gene (rather than the median of guide medians); non-targeting (NT) rows aggregated per barcode. Attribution: this aggregation follows Brieflow (Di Bernardo et al., Nat Commun 2026, doi:10.1038/s41467-026-73643-7; cheeseman-lab/brieflow, MIT), `lib.aggregate.aggregate` by `gene_symbol_0`.

## spaCR home
`spacr.gene_measurement_compare.aggregate_genes(..., level='cells')`

## Settings
`ops_aggregate_level` ('guides' | 'cells'), `ops_nt_rows` ('pseudo' | 'barcode')

## Validation data
`/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/external/X_bfr_only_agg.parquet`, `X_bfr_best_eps3.parquet` (+ `_guides`).

## Acceptance
Identical output to the reference gene tables at `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/external/` from the stored plate A cell tables; credit Brieflow in the docstring and release notes.

## Effort
S.

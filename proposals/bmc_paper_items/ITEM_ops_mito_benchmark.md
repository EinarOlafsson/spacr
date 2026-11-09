# Item spec: MitoCarta sub-module benchmark against an external published module table

- **Lifecycle:** future (alpha), behind "Show alpha features" like every new feature.
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Porting map: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/PORTING.md`.
- **Rule:** the spaCR settings-file run must reproduce the reference outputs identically (same rows, same values to float32 precision, same random streams: every seed is fixed in the reference code). Each documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** no (supplement-only strategy).

## Reference implementation
`/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/mito_compare.py`; inputs in `/nas_mnt/data/refdata_archive/` (MitoCarta3.0, Rath et al. 2021; the external module tables are published supplementary data).

## What it does
MitoCarta enrichment of a cluster partition (genes, purity; own and common gene sets); retrieval among MitoCarta genes by sub-compartment (OXPHOS, mitoribosome, ATP synthase, membrane organisation); co-membership of the sub-modules of an external published module table in the consensus atlas vs fitness-matched sets; the behaviour of named external clusters listed in a settings file.

## spaCR home
`spacr.ops_benchmarks.mitocarta_benchmark`

## Settings
`ops_eval_mitocarta` (path), `ops_eval_external_modules` (path)

## Validation data
`/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/tables/mito_*.csv`.

## Acceptance
Identical output to the reference tables at `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/tables/mito_*.csv`.

## Effort
S.

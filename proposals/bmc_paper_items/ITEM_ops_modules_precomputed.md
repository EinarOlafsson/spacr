# Item spec: module atlas on precomputed (weighted) embeddings

- **Lifecycle:** future (alpha), behind "Show alpha features" like every new feature.
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Porting map: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/PORTING.md`.
- **Rule:** the spaCR settings-file run must reproduce the reference outputs identically (same rows, same values to float32 precision, same random streams: every seed is fixed in the reference code). Each documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.

## Reference implementation
`/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/atlas_arm.py` (wraps the pilot's consensus.py + module_atlas.py; `tables/precomputed.json` switches off the consensus' standardise + PCA; consensus runs in-process because forked pools deadlocked).

## What it does
Same gates and controls as the pilot's module atlas (consensus clustering over repeated runs); for weighted profiles (contrastive weighting) the profile columns are used as they are.

## spaCR home
New `spacr.ops_modules`.

## Settings
`ops_modules_runs` (100), `ops_modules_precomputed_embedding` (bool)

## Validation data
`/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/atlas/<arm>/` (consensus partitions, module_summary.json).

## Acceptance
Identical output to the reference at `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/atlas/<arm>/`: identical consensus partitions (seeds as in consensus.py) and module summaries.

## Effort
L.

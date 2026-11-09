# Item spec: versioned reference sets, SELECT / REPORT halves and the 2022 time split

- **Lifecycle:** future (alpha), behind "Show alpha features" like every new feature.
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Porting map: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/PORTING.md`.
- **Rule:** the spaCR settings-file run must reproduce the reference outputs identically (same rows, same values to float32 precision, same random streams: every seed is fixed in the reference code). Each documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.

## Reference implementation
`/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/ic_lib.py` `_ref_pairs`, `corum_halves`, `_pair_half`; `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/prep_archive_refs.py`.

## What it does
Loaders keyed by name and version: CORUM (the published Brieflow benchmark file), KEGG, STRING v12 >= 0.4 / 0.7, BioGRID physical, published Brieflow STRING pairs; halves 'REF@SELECT' / 'REF@REPORT' (CORUM by a seeded permutation of complex names, SPLIT_SEED 20261107, REPORT pairs that also occur in a SELECT complex dropped; pair references by md5('a|b') parity); archived 2022 references (STRING v11.5, BioGRID 4.4.214, CORUM 3.0 via the Wayback snapshot) and 'new after 2022' sets diffed on stable ids (ENSP, Entrez; both genes present in the old release).

## spaCR home
New `spacr.ops_reference` module.

## Settings
`ops_reference_dir`, `ops_eval_references`, `ops_eval_split`

## Validation data
`/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/refs/*.parquet` and `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/refs/versions.json`; archives and md5s in `/nas_mnt/data/refdata_archive/` (MD5SUMS.txt, PROVENANCE files).

## Acceptance
Identical output to the reference at `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/refs/`: pair counts and pair sets identical to `versions.json` and the parquet files for every reference and half.

## Effort
S–M.

# Item spec: gene-gene retrieval mAP with multi-label references (copairs-style)

- **Lifecycle:** future (alpha), behind "Show alpha features" like every new feature.
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Porting map: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/PORTING.md`.
- **Rule:** the spaCR settings-file run must reproduce the reference outputs identically (same rows, same values to float32 precision, same random streams: every seed is fixed in the reference code). Each documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.

## Reference implementation
`/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/ic_lib.py`: `retrieval_ap`, `_ap_rows`, `_null`, `summarize_ap`, `boot_ci`; checks in `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/tests_retrieval.py`.

## What it does
For every gene, rank all other annotated genes by profile similarity (float32, ties rank the positive first, as copairs / spaCR `_profile_average_precision`) and compute the average precision of its reference partners. Positives come from a boolean gene x gene matrix (multi-label: complexes overlap, STRING is a graph), which `_profile_average_precision` cannot express. Per-gene p from spaCR's `_null_average_precision` (copairs null; one cached null per (positives, list length), seed = md5 of the shape mod 8096, 5,000 draws), BH via `multiple_testing.adjust_p_values`; mean AP, normalised mAP (`_expected_ap`), fraction q < 0.05, bootstrap CI over genes (B = 1000, seed 0).

References: copairs (github.com/cytomining/copairs); Benjamini & Hochberg 1995 (BH adjustment).

## spaCR home
`spacr.sp_stats._pair_average_precision(sims, positive_matrix, candidates)` + `pair_map(...)`

## Settings
`ops_eval_references` (list), `ops_eval_null_size` (5000), `ops_eval_split` ('select' | 'report' | 'all')

## Validation data
`/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/arms/C0_base.npz` (names, pcs) + `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/per_gene/C0_base.csv.gz` (per-gene ap, n_pos, n_total, nap, p, q for CORUM, STRING>=0.7, STRING>=0.4, KEGG, BioGRID).

## Acceptance
Identical output to the reference at `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/per_gene/C0_base.csv.gz`: per-gene `ap`, `n_pos`, `n_total`, `p`, `q` identical for every reference; `tests_retrieval.py` check 1 (AP identical to `_profile_average_precision` on single-label data) passes as a spaCR test.

## Effort
S–M.

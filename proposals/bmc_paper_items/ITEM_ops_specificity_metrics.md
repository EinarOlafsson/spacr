# Item spec: specificity metrics for gene profiles

- **Lifecycle:** future (alpha), behind "Show alpha features" like every new feature.
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Porting map: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/PORTING.md`.
- **Rule:** the spaCR settings-file run must reproduce the reference outputs identically (same rows, same values to float32 precision, same random streams: every seed is fixed in the reference code). Each documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.

## Reference implementation
`/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/ic_lib.py` `retrieval_ap_fitmatched`, `complex_identification`, `heldout_complex`, `fitness_bins`, `fitness_matched_positives`, `pair_auprc`; `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_plateA_full/improved/scripts/evaluate.py` `time_split`, `extra_metrics`.

## What it does
Retrieval against fitness-matched non-partners (10 per partner, same cell-count decile); CORUM complex identification top-1 / top-5 (own complex leave-one-out vs every complex without the gene; shuffled-label control); held-out CORUM members (5 folds within each complex, vs 10 fitness-matched decoys; AUROC / AP, CI over complexes); time split (top 500 / 2,000 pairs unknown in 2022, confirmed later; fitness-matched random control; binomial p); pair AUPRC with a gene bootstrap.

## spaCR home
`spacr.sp_stats` evaluation helpers.

## Settings
`ops_eval_heldout_folds` (5), `ops_eval_decoys` (10), `ops_eval_time_split` (bool)

## Validation data
`/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/metrics/<arm>.csv` and `<arm>.extra.csv` for every arm.

## Acceptance
Identical output to the reference metric rows at `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/metrics/C0_base.csv` (+ `.extra.csv`) and `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/metrics/X_bf_spacrcells.csv` (+ `.extra.csv`).

## Effort
M.

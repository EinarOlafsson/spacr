# Item spec: a public profile-retrieval entry: replicate mAP, compound–gene matching and plate effect

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`. Source inventory: `R/SPACR_PROVENANCE.md` (port P15).

## Why
Replicate retrieval (mAP per perturbation), compound–gene target matching and a plate-effect measure on JUMP CPJUMP1 profiles (Chandrasekaran et al. 2024) are computed by `jump_compare.py`, which calls only private `spacr.sp_stats` functions. Compound–gene matching and the plate effect are custom code. There is no public entry or settings file.

## Reference implementation
- **`R/jump/scripts/jump_compare.py`:**
  - `activity`, using `sp_stats._normalize_profiles`, `_select_profile_features`, `_profile_map`, `_profile_average_precision`, `_percent_replicating`;
  - `compound_consensus`, `gene_consensus`, `match`;
  - `plate_effect` (same-plate mAP; plate share of the top-10-PC variance).
- An independent copairs implementation (`R/jump/cp_baseline/code/cp_benchmark.py`) is kept as a cross-check, not ported.

## What to build
A Measure-module analysis entry `profiling_retrieval` (alpha), with public functions:
- `sp_stats.replicate_retrieval`;
- `sp_stats.target_matching`;
- `sp_stats.plate_effect`.

Settings:
- `retrieval_profiles` (well-level parquet);
- `retrieval_perturbation_key` ('Metadata_JCP2022');
- `retrieval_type_key` ('Metadata_perturbation_type');
- `retrieval_negcon` ('DMSO' / 'non-targeting' / 'BFP' per type);
- `retrieval_normalisation` ('mad_robustize'; 'tvn' via item 863);
- `retrieval_feature_select` (as `_select_profile_features`);
- `retrieval_null_size`;
- `retrieval_seed`;
- `retrieval_matching` ('compound_gene');
- `retrieval_target_map` (path);
- `retrieval_plate_effect` (bool).

Outputs in `results/profiling_retrieval/`: `activity.csv`, `summary.csv`, `matching.csv`, `plate_effect.csv`.

## Validation data and acceptance
- **Data:** the published CellProfiler CPJUMP1 profiles used by `jump_compare.py` (`R/jump/source_data`, `R/jump/cp_baseline`).
- **Acceptance:** identical output to the reference outputs written by `jump_compare.py` under `R/jump/`:
  - identical mAP per perturbation and identical summary, including the shuffled controls;
  - identical matching mAP and its shuffled control;
  - identical plate-effect numbers (same-plate mAP and plate share per perturbation type).
- Later spaCR-feature runs are then produced through this entry.

## Effort
- **Effort:** M.

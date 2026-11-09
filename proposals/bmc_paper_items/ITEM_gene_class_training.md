# Item spec: gene-class (multi-label) classifier training with grouped splits, decoy heads and shuffled controls

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. This is "NEW-1" in `R/REPRODUCIBILITY.md` §6.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`. Source inventory: `R/SPACR_PROVENANCE.md` (port P10).

## Why
Gene-class training (predicting multi-label gene classes from cell images or gene-level features) is produced by custom scripts.
- The logistic-regression baseline on gene features exists in several versions (v1, v3 unrepaired, v3 repaired); this item reproduces each from settings so that the version of record is unambiguous.
- The image-model arms and an architecture benchmark on BBBC021 run through `gc_train.py`, which uses only `spacr.utils.TorchModel` and `CustomCellClassifier` from a frozen spaCR.
- Port this before the planned GPU pilot, so that the pilot run is a spaCR run.

## Reference implementation
`R/ops_gene_classes/scripts/`:
- **`build_classes.py`:** class construction (`kinase_classes`, `corum_classes`, `kegg_classes`, `goslim_classes`, `funk_classes`, `filter_kegg`, the merge rule; KEGG disease/organismal maps dropped) and `iterative_stratification` (gene-grouped multi-label folds; Sechidis et al. 2011). Writes `tables_v3/` and `tables_v3_gated/`.
- **`baseline_lr.py::main`:** logistic regression on gene features, per-class AUROC, shuffled-class control. Runs: `runs/v3_lr_funkscores_d300`, `runs/v3_lr_funkscores_shuffled_d300`, `runs/v3gated_*`.
- **`eval_ops.py`:** `fast_auc`, `fast_ap`, `boot_ci`, the gene-wise decoy null with 300 decoy heads, discovery (a) known members with a new phenotype / (b) predicted new functions at BH q < 0.05.
- **`gc_train.py`:**
  - `ops_setup`, `ops_sampler` (balanced crop sampling per class/gene), `bbbc_setup`;
  - `run_fold`, `transform_spec` / `check_transform` (the saved input transform; item 772), `DevicePrep`;
  - decoy heads, admission/yield.
- **Other scripts:** `make_pilot.py`, `export_pilot.py`, `discovery_crops.py`, `cluster_classes.py`.
- **Benchmark drivers:** `R/ops_gene_classes/benchmark_archs/code/` (`launch_bench.sh`, `run_224.sh`, `eval_bbbc.py`, `build_bbbc021_subset.py`, `make_configs.py`, `pick_winner.py`, `spacr_cnn_fixed.py`, the patched spaCR CNN).
- **Design:** `R/ops_gene_classes/DESIGN.md` (§12 for the final rules).

## What to build
1. **`spacr.gene_classes` (new module):**
   - `build_gene_classes(sources, settings)` reproduces `build_classes.py` from versioned reference files: CORUM, KEGG, GO-slim, kinase list, the published Funk et al. 2022 clusters; md5 recorded.
   - `grouped_multilabel_folds(...)` = `iterative_stratification`, with a seed.
2. **Classifier training** in the train/classify module (`spacr.deep_spacr` / `spacr.ml`).
   - New settings:
     - `train_label_source` ('annotation' | 'gene_classes');
     - `train_gene_classes` (path to the class table);
     - `train_split` ('grouped_multilabel');
     - `train_split_group` ('gene');
     - `train_folds` (5);
     - `train_decoy_heads` (300);
     - `train_shuffle_control` (bool);
     - `train_balanced_sampling` ('class_gene');
     - `train_seed`;
     - `train_arch` (the arms in `make_configs.py`: resnet18, convnext_tiny, dinov2_frozen, maxvit_t, spacr_cnn);
     - `train_input_size` (224).
   - The input transform is saved with the checkpoint and checked at inference (item 772, already on nightly).
   - Feature-table models (the LR baseline): `ml_model='logistic_regression'` on a gene-level table, with `train_label_source='gene_classes'`.
3. **Evaluation report** (`results/gene_classes/`):
   - `per_class_auroc.csv` (with bootstrap CI);
   - `summary.json` (median AUROC by source: CORUM / KEGG / GO-slim / kinase; shuffled; decoy-head null);
   - `discovery_a.csv`, `discovery_b.csv` (BH q against the decoy null).
4. **Fix the spaCR CNN** as in `spacr_cnn_fixed.py`, with a test. This can be its own small item.

## Validation data and acceptance
Acceptance: identical output, or within the stated tolerance, to the reference outputs under `R/ops_gene_classes/`.
1. **Class tables:** `build_gene_classes` output identical to `R/ops_gene_classes/tables_v3/` and `tables_v3_gated/`, row for row. The folds are identical.
2. **LR baseline:** a settings run on plate A gene features reproduces exactly (CPU, deterministic):
   - the per-class AUROCs and summary of `runs/v3_lr_funkscores_d300` and of its shuffled control `runs/v3_lr_funkscores_shuffled_d300`;
   - the `discovery_b` table of the same run.
   - Also reproduce the v1 run `runs/lr_funkscores_all`.
3. **BBBC021 benchmark:** for each arm, a settings run on the `benchmark_archs` subset with the same seed matches `runs/bbbc021_*/eval/summary.json`.
   - GPU training is not bit-deterministic, so accept within the seed-to-seed spread measured over 5 seeds of the reference.
   - The data loading, splits, transform and evaluation code must be identical (same batches for a fixed seed on CPU).
4. **CNN fix:** the fixed arm reproduces the reference fixed-arm accuracy (`benchmark_archs` run summary) within the seed spread.
5. **Tests (CPU):** fold disjointness by gene; decoy-null calibration on random labels; the transform check refuses a mismatch.

## Effort
- **Effort:** L.

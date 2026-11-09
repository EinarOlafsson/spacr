# Item spec: spaCR model architectures: channel-adaptive CNN/ViT, well set-transformer, MAE/DINO pre-training, and the spaCR CNN fix

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`. Source inventory: `R/SPACR_PROVENANCE.md` (port P17).

## Why
spaCR should ship three architectures that work with any stain panel: a channel-adaptive CNN, a channel-adaptive ViT, and a well-level set transformer, plus self-supervised pre-training (MAE, He et al. 2022; DINO, Caron et al. 2021) and a missing-stain evaluation. The models live outside the repo, in `spacr_arch.py`, with no spaCR import. The shipped spaCR CNN (`CustomCellClassifier`) has a defect that a local patch (`spacr_cnn_fixed.py`) fixes.

## Reference implementation
- **`R/spacr_architectures/code/spacr_arch.py`:** `SpacrChannelCNN`, `SpacrChannelViT`, `SpacrWellTransformer`, `ChannelDINO`, `ChannelMAE`, `penetrant_fraction` (25 unit tests).
- **Training:** `train_well.py` (well-level set training on ResNet-18 crop embeddings with TVN-like normalisation), `train_cell.py`, `pretrain_cell.py`, `summarize_cpu.py` → `runs_cpu/summary_table.csv`. Runs: `runs_cpu/bbbc021_NSCB_r18_*`.
- **Design:** `R/spacr_architectures/DESIGN.md`, `PLAN.md`.
- **CNN fix:** `R/ops_gene_classes/benchmark_archs/code/spacr_cnn_fixed.py` against `spacr.utils.CustomCellClassifier`.

## What to build
1. **Models:** add `SpacrChannelCNN`, `SpacrChannelViT` and `SpacrWellTransformer` to spaCR's model registry (`spacr.utils` / `deep_spacr` `choose_model`), selectable through `train_arch` ('spacr_channel_cnn', 'spacr_channel_vit', 'spacr_well_transformer').
2. **Well-level training** from embedding tables:
   - `train_level` ('cell' | 'well_set');
   - `train_set_max_cells`;
   - `train_set_input` (an embedding parquet from `ITEM_crop_embedding_well_profiles.md`);
   - `train_set_normalisation` (as in `train_well.py`);
   - `train_protocol` ('NSC' | 'NSCB', the folds from the MoA evaluation entry);
   - `train_shuffle_control`;
   - `train_seed`.
3. **Self-supervised pre-training:** `pretrain_method` ('mae' | 'dino'), `pretrain_epochs`, `pretrain_mask_ratio`; few-shot `train_labels_per_class`.
4. **Missing-stain evaluation:** `eval_channel_subsets` (a list of channel subsets, scored per model).
5. **CNN fix:** replace `CustomCellClassifier` behaviour with the fixed version, with a regression test on the benchmark subset.
6. Report `results/model_benchmark/summary.csv` with one row per arm and the controls.

## Validation data and acceptance
- **Well transformer:** with the same input embeddings (`runs_cpu/bbbc021_NSCB_r18_*` inputs) and seeds, the CPU well-transformer run gives identical output to the reference at `R/spacr_architectures/runs_cpu/summary_table.csv` (main arm and every ablation/control row).
- **Model unit tests:** port the 25 tests of `spacr_arch.py`.
- **Missing-stain and self-supervised arms:** not yet run, so run them through spaCR first; those runs then are the reference.
- **CNN fix:** the regression test asserts the fixed arm reaches the accuracy of the reference fixed arm in the `R/ops_gene_classes/benchmark_archs/` run summary within the seed spread, and exceeds the as-shipped arm.

## Effort
- **Effort:** L.

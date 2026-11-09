# Item spec: raw crop shards from merged fields, pretrained embeddings, well-level embedding profiles and UMAP

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Related: 809 (sharded crop store), 772 (raw crops, transform in the model; on nightly).
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`. Source inventory: `R/SPACR_PROVENANCE.md` (port P16).

## Why
Pretrained-backbone embeddings of spaCR crops, their well-level aggregate and a UMAP of wells were made as follows:
- `make_crops.py` and `embed_crops.py` are drivers that call spaCR internals;
- the aggregate and UMAP are custom.

The `R/bbbc021/cv/README` lists six gaps to close:
- per-crop stretch;
- mask edits not written back;
- illumination on crop windows;
- OpenPhenom loading;
- ChAda-ViT padding;
- no ImageNet normalisation.

## Reference implementation
- **`R/bbbc021/cv/make_crops.py`:**
  - reads through `spacr.crops.open_merged_field` and `MergedField.label_index` / `read_window` / `read_mask_window`;
  - illumination via `spacr.illumination.load_illumination_model().field_for().gain_stack` and `_offset_stack`;
  - writes h5 shards with a 300-cells-per-well cap and mask bit flags.
- **`R/bbbc021/cv/embed_crops.py`:** `spacr.embeddings.EmbeddingSpec`, `CHANNEL_PER_CHANNEL`, `_backbone_encoder`, `_estimate_channel_scale`, `_embed_plate`, `_weights_on_disk`.
- **`R/bbbc021/cv/aggregate_emb.py`:** well median/mean of embeddings. Scoring: `score_emb.sh` → `supervised_moa.py` (see `ITEM_profiling_moa_evaluation.md`); `build_table.py`.
- **UMAP:** `/nas_mnt/data/sync/papers/BMCspacr/figures/scripts/fig5_cv.py` (`umap.UMAP`, cosine, 15 neighbours, seed 0, on resnet18 `wells_median`, metadata via `run_and_score.attach_metadata`; McInnes et al. 2018) → `/nas_mnt/data/sync/papers/BMCspacr/figures/source_data/Fig5D_resnet18_well_umap.csv`.

## What to build
1. **Crop export in `measure_crop`** (809): `crop_store` ('png' | 'h5_shards' | 'tar_shards'), `crop_cap_per_well` (300), `crop_mask_flags` (bool), `crop_illumination` ('none' | 'gain_offset', the saved illumination model), `crop_seed`. Crops are raw (772).
2. **Embeddings module run from settings:**
   - `embedding_backbone` ('resnet18', …);
   - `embedding_channel_mode` ('per_channel');
   - `embedding_channel_scale` ('estimate', as `_estimate_channel_scale`);
   - `embedding_input_normalisation` ('none' as in the reference | 'imagenet');
   - `embedding_weights` (path or name);
   - `embedding_out`.
3. **Well aggregate of embeddings:** `embedding_aggregate` ('well_median, well_mean'), writing `wells_median.parquet` like the feature aggregate (`ITEM_screen_staging_export_aggregate.md`).
4. **UMAP export:** `embedding_umap` (bool), `umap_metric` ('cosine'), `umap_neighbors` (15), `umap_seed` (0), writing the well UMAP CSV.

## Validation data and acceptance
- **Data:** BBBC021 merged fields (`/mnt/wd4tb/bmc_reanalysis/bbbc021_spacr/work/<plate>/merged/`) and the cv outputs in `R/bbbc021/cv/`.
- **Acceptance** (identical, or within the stated tolerance, to the reference outputs in `R/bbbc021/cv/`):
  - crop shards identical (the same cells for the same seed, pixel-identical);
  - embeddings identical on CPU (allclose 1e-6 on GPU);
  - the well median table identical;
  - logistic NSC/NSCB accuracy through the MoA evaluation entry equal to `R/bbbc021/cv/results/cv_summary.md`;
  - UMAP coordinates identical to the UMAP CSV above with seed 0, same umap-learn version (record the version).

## Effort
- **Effort:** M.

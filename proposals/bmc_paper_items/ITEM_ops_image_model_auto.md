# Item spec: automatic per-screen OPS image-model training (spacr_ops_transformer default; gene-held-out splits, QC gate, weights and embeddings saved)

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs on the validation data: identical gene split, held-out cells, cell lists and QC tables (CPU-deterministic parts bit-identical; GPU training within the reference's seed-0 vs seed-1 spread). Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Related:** `ITEM_gene_class_training.md` (grouped splits, saved transform, item 772), `ITEM_ops_embedding_profiles.md`, `ITEM_crop_embedding_well_profiles.md`.

## Why
spaCR should give every optical pooled screen the best, most generalizable image features it can learn from that screen: a channel-aware model that treats each fluorescence channel as its own information domain, takes any number of channels and classes, and is trained automatically from one settings file. That needs:
1. a gene-held-out split that can never leak evaluation genes into training or tuning;
2. self-supervised pre-training on every crop of the screen (no labels), then a gene-identity fine-tune;
3. an equal, recorded tuning budget chosen on held-out training genes only;
4. an automatic quality-control gate, with a refusal to pass on embeddings from a model that fails it;
5. saved weights for every run, the saved input transform, and per-cell embeddings for downstream profiles.

## Behaviour
1. **Inputs.** Per-cell crops written by the phenotype run (raw, unnormalised), the cell → guide → gene table, and the per-plate × channel scale (99th percentile of cell pixels). Channel names come from the screen's channel settings and map to learned stain embeddings through a stain vocabulary with aliases (unknown stains get spare slots), so any channel count runs with the same code.
2. **Gene split (written once, seeded, saved as `split.json`).**
   - Non-targeting guides → NT pseudo-genes; about half are evaluation NT, the rest training-side (a few inner held-out).
   - Training genes: a settings-chosen gene set; 10 % of them are inner held-out (tuning and early stopping only).
   - Every other gene is an evaluation gene: never loaded in training, never used for tuning; embedded once at the end.
   - Defaults from the reference: seed 20261110; ≤ 500 cells per gene; 5 held-out cells per training class (50 for NT).
3. **Transform** (saved with every checkpoint and re-checked at inference, item 772): fixed plate × channel scale → clip [0, 1] → (x − 0.5) / 0.25; native crop size (96 px, ViT patch 12) unless the optional resolution pilot adopts 224 (patch 32). Augmentation on the device: dihedral, shift ± 4 px, per-channel gain U(0.9, 1.1) ("standard"), or ± 8 px / U(0.8, 1.2) ("strong").
4. **Pre-training (default on).** ChannelDINO self-distillation of the channel ViT-S (per-channel patch tokens + stain embedding + 2-D sin-cos position, attention across all channels' tokens): two global views, hierarchical channel sampling (channel dropout, DNA always kept) on the student, EMA teacher, 4,096 prototypes, teacher temperature 0.04, EMA 0.996 → 1, AdamW lr 5e-4 × batch / 256, weight decay 0.05, 2,000 warm-up steps, batch 256 with automatic micro-batching. All crops of the screen, no labels; reports state that evaluation genes' cells are seen without labels. Length = a GPU-time budget (throughput measured early fixes the step count; capped in epochs). Teacher encoder saved every 2,000 steps; snapshots every 10,000; a held-out-gene monitor every 5,000.
5. **Fine-tune (gene identity).** Encoder + dropout 0.1 + linear head over training genes + one pooled NT class. Cross-entropy with label smoothing 0.1, AdamW weight decay 1e-4, cosine schedule with 3 % warm-up, batch 64, class-uniform then cell-uniform sampling, 8,000 steps, gradient clipping 3.0. Every 1,000 steps: held-out-cell loss / accuracy and the held-out-gene AUROC (50 cells per inner held-out gene + 200 NT; fold-z-scored embeddings, 3-fold cross-fitted nearest-centroid cosine, macro one-vs-rest AUROC). The best checkpoint by that AUROC and the last checkpoint are both saved.
6. **Tuning (equal budget, recorded).** 4 settings in two stages: lr {low, high} with standard augmentation and channel dropout on; then, at the better lr, (strong augmentation, channel dropout on) and (standard augmentation, channel dropout off). Selection by the held-out-gene AUROC only. Defaults: pre-trained ViT lr {1e-4, 3e-4}; models trained from scratch {3e-4, 1e-3}.
7. **Controls.** A seed-1 replicate of the selected setting, and a model trained on cell-shuffled labels with the same budget.
8. **Correctness.** Re-embedding the held-out-gene cells from the saved checkpoint reproduces the saved embeddings bit for bit; held-out accuracy ≥ 5 × the shuffled control (recorded).
9. **Inference.** ONE model (the selected run) embeds every evaluation-gene cell, every evaluation-NT cell and any probe-label cells; output `emb_f16.npy`, `cells.parquet` and a manifest with the checkpoint md5 and transform.
10. **QC gate.** Six checks with fixed pass rules: convergence (held-out loss change < 2 % in the last 20 % of steps); held-out learning on evaluation genes (5 seeded groups of 133 genes + evaluation NT, 50 cells each; beats the shuffled control and chance in 5/5 groups and is ≥ frozen ImageNet); a mitotic linear probe ≥ 90 % when labels exist; known-phenotype positive controls vs an NT-guide null (≥ 80 % above the null 95th percentile); batch leakage (NT-vs-NT AUC ≤ the measurement features' + 0.02; well prediction ≤ measurements); seed stability (gene–gene similarity r ≥ 0.5 between seeds). Embeddings from a failing model are written with `qc_pass = false` and refused by downstream profile phases unless `ops_img_allow_failed_qc = True` (recorded).
11. **Channel attribution** (models with a channel attention pool): per gene, AUC(all stains) − AUC(one stain left out) through the model's own channel-subset path (paired bootstrap CI), plus the shift in channel attention vs NT; the dominant stain per gene with a phenotype; concordance with the reference hits' feature channels.
12. **Optional baselines in the same run** (`ops_img_baselines`): OpenPhenom-S/16 frozen (non-commercial licence; channel-agnostic, the screen's stains as channels, resized to 256), frozen ImageNet ResNet-50, fine-tuned ResNet-50, measurements; all on the same cells and QC.

## What to build
- `spacr.ops_image_model` (new, alpha):
  - `gene_split(cells, settings)`, `heldout_cells(...)`, `hog_auroc(Z, y, seed)`;
  - `build_encoder(name, stains, settings)` for `spacr_ops_transformer` (default: channel ViT-S + ChannelDINO pre-training), `spacr_channel_vit` (no pre-training), `spacr_channel_cnn`, `resnet50` (ImageNet init), `openphenom` (frozen, through the existing OpenPhenom loader in `spacr.embeddings`);
  - `pretrain_dino(...)`, `finetune_gene_identity(...)`, `tune(...)` (the 4-setting rule), `controls(...)`, `check_reproduction(...)`;
  - `embed_cells(...)`, `qc_gate(...)` (the six checks as functions), `channel_attribution(...)`.
- The channel models (channel ViT, ChannelDINO, channel-adaptive CNN, stain vocabulary, hierarchical channel sampling) as plain `torch.nn` modules in a new `spacr/channel_models.py`, with their unit tests (any channel count; missing channels have no influence; a channel subset equals masking; DINO / MAE losses backpropagate).
- A settings-file phase `ops_image_model` after the crop phase; GUI under the OPS module (alpha).

## Settings (alpha)
`ops_img_model` ('spacr_ops_transformer' | 'spacr_channel_vit' | 'spacr_channel_cnn' | 'resnet50'), `ops_img_stains`, `ops_img_crop_size` (96), `ops_img_resolution_pilot` (False), `ops_img_pretrain` (True), `ops_img_pretrain_hours` (9), `ops_img_pretrain_max_epochs` (10), `ops_img_pretrain_exclude_eval` (False; True = strict pre-training on genotyped non-evaluation cells only, the leakage control), `ops_img_split_seed` (20261110), `ops_img_heldout_gene_frac` (0.1), `ops_img_eval_nt_frac` (0.5), `ops_img_cells_per_gene` (500), `ops_img_train_genes` (path or 'auto'), `ops_img_steps` (8000), `ops_img_batch` (64), `ops_img_lr_grid` (per model default), `ops_img_label_smoothing` (0.1), `ops_img_weight_decay` (1e-4), `ops_img_hcs` (True), `ops_img_vram_gb` (6), `ops_img_controls` (True), `ops_img_qc_controls` (positive-control gene list), `ops_img_qc_mitotic_labels` (optional), `ops_img_allow_failed_qc` (False), `ops_img_baselines` ([]) (adds 'openphenom' and 'openphenom_pca_centerscale': PCA fitted on training-side NT cells, then a per-well StandardScaler on that well's training-side NT cells, as the OpenPhenom model card recommends), `ops_img_out`.

## Outputs
`results/ops_image_model/`: `split.json`; `runs/<setting>/` (`config.json`, `transform.json`, `train_log.jsonl`, `model.pt`, `last.pt`, `result.json`, held-out-gene embeddings); `pretrain/` (`encoder.pt`, snapshots, log); `selection.json`; `embeddings/` (`cells.parquet`, `emb_f16.npy`, manifest); `qc/qc_summary.json` + tables; `attribution/`; one bars + jitter QC figure (models × checks, controls as rows).

## Validation and acceptance
1. **Split and cells:** on the reference OPS plate, the split, held-out cells, evaluation cell list and the QC / attribution cell lists are identical to the reference run's.
2. **Training:** for each model, the settings run's best held-out-gene AUROC lies within the reference seed-0 / seed-1 spread; the reproduction check passes bit-identically on its own checkpoint.
3. **QC:** `qc_gate` on the reference embeddings reproduces the reference QC summary and tables exactly (CPU, fixed seeds).
4. **Tests (CPU):** no evaluation gene in any training or tuning set; the transform check refuses a mismatch; the channel-model unit tests; the QC refusal path; a 4-step smoke run of each model.

## Effort
L.

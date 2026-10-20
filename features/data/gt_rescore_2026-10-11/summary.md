# GT re-score 2026-10-11: PV and plaque models on the new curated ground truth

This covers steps 2 and 3, scoring. Step 1, the inventory, is in `../README.md` and `../manifest.json`.

Everything here is reproducible from the files in this folder:
- `per_field.csv`: one row per model per field.
- `rescore.ipynb`: rebuilds every table from `per_field.csv`.
- `code/`: the scripts.
- `preds/`: cached predictions as compressed npz, keyed by image pixel sha256.

Nothing was published, pushed or written outside `scores/`. The GT sources were only read.

## What was run

**Checkpoints.** All are listed in `weights.json`.
- Every zoo model was resolved by `spacr.model_zoo._ensure_model_file` and its sha256 matched the zoo row:
  - PV v1–v5: cpsam_v2_toxo_r2, _r5, _r6, _r7 and _r8
  - plaque v1–v4: cpsam_plaque_r3, _r5, _r5_geldoc and _r6
- Stock `cpsam` (1,233,587,898 bytes) and `cpsam_v2` (1,233,586,851 bytes) were loaded by full path; both sizes match.
- No bare Cellpose model names were used.

**Inference.** Each family ran as one GPU job through `tools/gpu_turn.sh` and `run_capped.sh 24G`.
- **Plaque**: Plaque mode's own pair, `spacr.submodules._plaque_cellpose_model` and `spacr.plaque.segment_plaque_image(..., get_analyze_plaque_settings({}))`. Settings: diameter 30, flow 0.4, CP_prob 0.
- **PV**: the Model Zoo path, `spacr.model_zoo.benchmark` → `config_for` → `model_compare.segment_with_cellpose`, with the `ModelConfig` defaults. Settings: diameter 30, flow 0.4, cellprob 0, normalize, min_size 15.
- **Image matching**: every image is the exact array whose pixel sha256 the manifest recorded; `infer.read_image` verifies this.
  - pmc_staged: the masks were drawn on the upscaled `img` array inside the seg.npy bundle. The `.png` beside each bundle is the small original, and using it would have misaligned every mask.
  - Round-3 PV fields that use the r7 mask are paired with the r7 raw image, not with the 8-bit bundle rendering.

**Metrics.** Per field, then pooled by summing TP, FP and FN over the fields of each cell.
- F1 at IoU 0.5 and 0.75, using `cellpose.metrics.average_precision` with Cellpose's optimal one-to-one matching.
- Precision and recall at IoU 0.5.
- AJI (Kumar 2017):
  - `aji_pooled` = ΣC / ΣU over fields
  - `aji_mean` = the mean of the per-field AJI
  - For checking, the per-field `aji_cellpose` from `cellpose.metrics.aggregated_jaccard_index` is also stored in `per_field.csv`.
- Counts:
  - `count_mae` = mean |n_pred − n_gt| per well or field
  - `count_bias` = mean (n_pred − n_gt)
- Reviewed-empty fields are scored, so everything predicted on them is a false positive.
- 95% CIs come from a percentile bootstrap over fields (2,000 resamples). Comparisons between two models use a paired bootstrap on the same fields.

## Ground-truth set

| | PV | plaque |
|---|---|---|
| records scored (all sources) | 771 | 810 |
| distinct fields, deduplicated pool | 655 | 562 |
| reviewed-empty fields in pool | 113 | 67 |

**Rules**
- Curated manifest records only.
- **Included**, by maintainer decision on 2026-10-11: the 98 round-3 PV fields with no ledger row. The maintainer paged through them. They are labelled `review_basis = paged_no_ledger_row`, and **39 of the 98 are empty masks**, so they score as reviewed-empty.
- **Excluded**:
  - the 3 plaque literature fields marked both done and Discard (with their frozen copies)
  - all 11 Gel Doc set-aside wells: 6 GRA8/Plate1 duplicates of ReimagedPlate1, and 5 Dia_Plate18
  - 57 stale frozen copies in the 263-set, where the live curated mask differs
  - 1 frozen copy in the 50-set of a field the maintainer later discarded

**Mask choice**
- Round-3 PV: the later `toxoplasma_pv_r7` mask is used for 335 of the 373 ledger-done fields. The other 38, and the 98 paged fields, use the round-3 seg.npy mask. The choice is in `gt_mask_note`.
- Frozen copies identical to a live curated mask (`frozen_state = duplicate`) are scored in their own source rows but not pooled. Copies with no live curated counterpart (`frozen_only`) are pooled: 44 plaque round-4 review fields, the 20 PV plate-test fields and 6 PV compare50 fields.

**Pool**
- The pool has one record per physical field (`identity_stem`). The source priority is:
  - PV: r7 recuration > anchors > round3 > literature > plate test20 > compare50
  - plaque: literature > gel_doc (unseeded) > pmc_staged > model_review held_out > model_review > compare263
- A pooled field counts as seen by a model if any of its copies was seen.

## (a) Strict hold-out per model per source

For each model, the fields it trained or validated on are dropped (`seen_in_training_or_validation`), so **n differs between models** and rows are not directly comparable across models.

The `n_no_group` and `f1_50_no_group` columns are the sensitivity check: they also drop fields that share a plate, acquisition or paper with that model's training data (`same_group_in_training_or_validation`).

### PV

**anchor11**

| model    |   n_total |   n |   n_empty |   f1_50 |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   n_no_group |   f1_50_no_group |
|:---------|----------:|----:|----------:|--------:|--------:|------------:|---------:|-------------:|-----------:|-------------:|-----------------:|
| pv_v1    |        11 |  11 |         1 |   0.858 |   0.747 |       0.872 |    0.845 |        0.714 |      0.739 |            0 |                  |
| pv_v2    |        11 |   0 |         0 |         |         |             |          |              |            |            0 |                  |
| pv_v3    |        11 |  11 |         1 |   0.855 |   0.735 |       0.838 |    0.873 |        0.764 |      0.764 |            0 |                  |
| pv_v4    |        11 |  11 |         1 |   0.851 |   0.743 |       0.843 |    0.86  |        0.754 |      0.738 |            0 |                  |
| pv_v5    |        11 |   8 |         1 |   0.789 |   0.692 |       0.714 |    0.881 |        0.661 |      0.526 |            0 |                  |
| cpsam    |        11 |  11 |         1 |   0.723 |   0.599 |       0.714 |    0.731 |        0.3   |      0.348 |           11 |            0.723 |
| cpsam_v2 |        11 |  11 |         1 |   0.766 |   0.644 |       0.75  |    0.781 |        0.386 |      0.446 |           11 |            0.766 |

**compare50_test**

| model    |   n_total |   n |   n_empty |   f1_50 |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   n_no_group |   f1_50_no_group |
|:---------|----------:|----:|----------:|--------:|--------:|------------:|---------:|-------------:|-----------:|-------------:|-----------------:|
| pv_v1    |        49 |  27 |         3 |   0.752 |   0.701 |       0.804 |    0.706 |        0.549 |      0.465 |           13 |            0.178 |
| pv_v2    |        49 |  10 |         1 |   0.203 |   0.051 |       0.583 |    0.123 |        0.128 |      0.109 |            8 |            0.203 |
| pv_v3    |        49 |  11 |         1 |   0.468 |   0.348 |       0.63  |    0.372 |        0.367 |      0.272 |            8 |            0.263 |
| pv_v4    |        49 |  11 |         1 |   0.504 |   0.366 |       0.663 |    0.406 |        0.334 |      0.335 |            8 |            0.373 |
| pv_v5    |        49 |  49 |         4 |   0.796 |   0.712 |       0.756 |    0.841 |        0.554 |      0.5   |           42 |            0.791 |
| cpsam    |        49 |  49 |         4 |   0.565 |   0.475 |       0.472 |    0.705 |        0.126 |      0.21  |           49 |            0.565 |
| cpsam_v2 |        49 |  49 |         4 |   0.561 |   0.471 |       0.465 |    0.704 |        0.125 |      0.223 |           49 |            0.561 |

**literature_holdout**

| model    |   n_total |   n |   n_empty |   f1_50 |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   n_no_group |   f1_50_no_group |
|:---------|----------:|----:|----------:|--------:|--------:|------------:|---------:|-------------:|-----------:|-------------:|-----------------:|
| pv_v1    |        39 |  39 |         1 |   0.314 |   0.179 |       0.391 |    0.263 |        0.155 |      0.261 |           39 |            0.314 |
| pv_v2    |        39 |  39 |         1 |   0.397 |   0.188 |       0.541 |    0.313 |        0.15  |      0.265 |           39 |            0.397 |
| pv_v3    |        39 |  39 |         1 |   0.392 |   0.173 |       0.534 |    0.309 |        0.193 |      0.321 |           39 |            0.392 |
| pv_v4    |        39 |  39 |         1 |   0.464 |   0.218 |       0.561 |    0.396 |        0.196 |      0.311 |           39 |            0.464 |
| pv_v5    |        39 |  10 |         1 |   0.351 |   0.223 |       0.344 |    0.359 |        0.122 |      0.237 |            7 |            0.31  |
| cpsam    |        39 |  39 |         1 |   0.593 |   0.543 |       0.51  |    0.707 |        0.174 |      0.357 |           39 |            0.593 |
| cpsam_v2 |        39 |  39 |         1 |   0.643 |   0.635 |       0.537 |    0.803 |        0.179 |      0.392 |           39 |            0.643 |

**r7_dataset_recuration**

| model    |   n_total |   n |   n_empty |   f1_50 |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   n_no_group |   f1_50_no_group |
|:---------|----------:|----:|----------:|--------:|--------:|------------:|---------:|-------------:|-----------:|-------------:|-----------------:|
| pv_v1    |       181 | 108 |         2 |   0.778 |   0.715 |       0.847 |    0.719 |        0.485 |      0.457 |           84 |            0.62  |
| pv_v2    |       181 |  60 |         2 |   0.657 |   0.602 |       0.736 |    0.593 |        0.44  |      0.515 |           60 |            0.657 |
| pv_v3    |       181 |  64 |         2 |   0.732 |   0.677 |       0.705 |    0.761 |        0.536 |      0.584 |           60 |            0.686 |
| pv_v4    |       181 |   4 |         0 |   0.848 |   0.736 |       0.847 |    0.85  |        0.731 |      0.697 |            0 |                  |
| pv_v5    |       181 |  36 |         1 |   0.82  |   0.746 |       0.773 |    0.874 |        0.611 |      0.609 |           29 |            0.819 |
| cpsam    |       181 | 181 |         3 |   0.629 |   0.517 |       0.576 |    0.693 |        0.184 |      0.337 |          181 |            0.629 |
| cpsam_v2 |       181 | 181 |         3 |   0.621 |   0.514 |       0.555 |    0.705 |        0.19  |      0.347 |          181 |            0.621 |

**r7_plate_test20**

| model    |   n_total |   n |   n_empty |   f1_50 |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   n_no_group |   f1_50_no_group |
|:---------|----------:|----:|----------:|--------:|--------:|------------:|---------:|-------------:|-----------:|-------------:|-----------------:|
| pv_v1    |        20 |  20 |         0 |   0.613 |   0.565 |       0.968 |    0.448 |        0.228 |      0.33  |           20 |            0.613 |
| pv_v2    |        20 |  20 |         0 |   0.719 |   0.628 |       0.967 |    0.572 |        0.461 |      0.654 |           20 |            0.719 |
| pv_v3    |        20 |  20 |         0 |   0.899 |   0.894 |       0.982 |    0.829 |        0.584 |      0.784 |           20 |            0.899 |
| pv_v4    |        20 |  20 |         0 |   0.748 |   0.646 |       0.908 |    0.636 |        0.775 |      0.812 |            0 |                  |
| pv_v5    |        20 |  20 |         0 |   0.689 |   0.625 |       0.845 |    0.582 |        0.736 |      0.648 |            0 |                  |
| cpsam    |        20 |  20 |         0 |   0.577 |   0.445 |       0.706 |    0.488 |        0.569 |      0.546 |           20 |            0.577 |
| cpsam_v2 |        20 |  20 |         0 |   0.429 |   0.326 |       0.47  |    0.394 |        0.396 |      0.416 |           20 |            0.429 |

**round3_queue**

| model    |   n_total |   n |   n_empty |   f1_50 |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   n_no_group |   f1_50_no_group |
|:---------|----------:|----:|----------:|--------:|--------:|------------:|---------:|-------------:|-----------:|-------------:|-----------------:|
| pv_v1    |       471 | 346 |       102 |   0.853 |   0.795 |       0.905 |    0.806 |        0.711 |      0.561 |          200 |            0.796 |
| pv_v2    |       471 | 136 |        39 |   0.787 |   0.709 |       0.784 |    0.791 |        0.575 |      0.264 |            0 |                  |
| pv_v3    |       471 | 136 |        39 |   0.804 |   0.734 |       0.853 |    0.761 |        0.615 |      0.272 |            0 |                  |
| pv_v4    |       471 | 136 |        39 |   0.807 |   0.697 |       0.773 |    0.844 |        0.681 |      0.359 |            0 |                  |
| pv_v5    |       471 | 431 |       107 |   0.734 |   0.639 |       0.639 |    0.86  |        0.585 |      0.45  |          171 |            0.74  |
| cpsam    |       471 | 471 |       107 |   0.403 |   0.35  |       0.298 |    0.622 |        0.047 |      0.075 |          471 |            0.403 |
| cpsam_v2 |       471 | 471 |       107 |   0.401 |   0.34  |       0.291 |    0.646 |        0.051 |      0.087 |          471 |            0.401 |

**POOLED (deduplicated)**

| model    |   n_total |   n |   n_empty |   f1_50 |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   n_no_group |   f1_50_no_group |
|:---------|----------:|----:|----------:|--------:|--------:|------------:|---------:|-------------:|-----------:|-------------:|-----------------:|
| pv_v1    |       655 | 486 |       108 |   0.793 |   0.73  |       0.86  |    0.735 |        0.558 |      0.48  |          329 |            0.63  |
| pv_v2    |       655 | 258 |        42 |   0.713 |   0.622 |       0.766 |    0.667 |        0.451 |      0.365 |          120 |            0.582 |
| pv_v3    |       655 | 269 |        43 |   0.75  |   0.669 |       0.8   |    0.707 |        0.528 |      0.426 |          120 |            0.64  |
| pv_v4    |       655 | 209 |        41 |   0.761 |   0.636 |       0.769 |    0.754 |        0.582 |      0.423 |           40 |            0.463 |
| pv_v5    |       655 | 481 |       111 |   0.733 |   0.636 |       0.646 |    0.847 |        0.585 |      0.458 |          184 |            0.73  |
| cpsam    |       655 | 655 |       113 |   0.46  |   0.386 |       0.359 |    0.64  |        0.071 |      0.177 |          655 |            0.46  |
| cpsam_v2 |       655 | 655 |       113 |   0.455 |   0.38  |       0.347 |    0.66  |        0.074 |      0.187 |          655 |            0.455 |


### Plaque

**compare263_test**

| model     |   n_total |   n |   n_empty |   f1_50 |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   n_no_group |   f1_50_no_group |   count_mae |   count_bias |
|:----------|----------:|----:|----------:|--------:|--------:|------------:|---------:|-------------:|-----------:|-------------:|-----------------:|------------:|-------------:|
| plaque_v1 |       203 | 168 |        11 |   0.797 |   0.649 |       0.857 |    0.745 |        0.676 |      0.575 |          145 |            0.781 |      12.167 |       -6.095 |
| plaque_v2 |       203 |  74 |         7 |   0.615 |   0.511 |       0.652 |    0.581 |        0.554 |      0.389 |           66 |            0.564 |      19.027 |       -3.838 |
| plaque_v3 |       203 | 100 |         1 |   0.834 |   0.683 |       0.831 |    0.838 |        0.657 |      0.666 |           76 |            0.812 |       7.32  |        0.4   |
| plaque_v4 |       203 | 203 |        24 |   0.827 |   0.639 |       0.831 |    0.822 |        0.685 |      0.609 |          174 |            0.823 |       6.182 |       -0.468 |
| cpsam     |       203 | 203 |        24 |   0.246 |   0.176 |       0.172 |    0.433 |        0.277 |      0.326 |          203 |            0.246 |     102.128 |       68.089 |
| cpsam_v2  |       203 | 203 |        24 |   0.335 |   0.244 |       0.295 |    0.387 |        0.238 |      0.283 |          203 |            0.335 |      50.675 |       13.897 |

**gel_doc**

| model     |   n_total |   n |   n_empty |   f1_50 |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   n_no_group |   f1_50_no_group |   count_mae |   count_bias |
|:----------|----------:|----:|----------:|--------:|--------:|------------:|---------:|-------------:|-----------:|-------------:|-----------------:|------------:|-------------:|
| plaque_v1 |        89 |  89 |         2 |   0.151 |   0.082 |       0.823 |    0.083 |        0.108 |      0.082 |           89 |            0.151 |      26.562 |      -26.517 |
| plaque_v2 |        89 |  89 |         2 |   0.174 |   0.104 |       0.803 |    0.098 |        0.131 |      0.11  |           89 |            0.174 |      25.933 |      -25.91  |
| plaque_v3 |        89 |  18 |         0 |   0.642 |   0.299 |       0.697 |    0.595 |        0.501 |      0.433 |           18 |            0.642 |       5.222 |       -4.111 |
| plaque_v4 |        89 |  12 |         0 |   0.454 |   0.076 |       0.541 |    0.391 |        0.388 |      0.23  |           12 |            0.454 |       8.25  |       -7.75  |
| cpsam     |        89 |  89 |         2 |   0.001 |   0.001 |       0.021 |    0.001 |        0.002 |      0.01  |           89 |            0.001 |      28.449 |      -28.404 |
| cpsam_v2  |        89 |  89 |         2 |   0     |   0     |       0     |    0     |        0     |      0     |           89 |            0     |      28.27  |      -27.708 |

**gel_doc_model_review**

| model     |   n_total |   n |   n_empty |   f1_50 |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   n_no_group |   f1_50_no_group |   count_mae |   count_bias |
|:----------|----------:|----:|----------:|--------:|--------:|------------:|---------:|-------------:|-----------:|-------------:|-----------------:|------------:|-------------:|
| plaque_v1 |       126 | 126 |         4 |   0.091 |   0.052 |       0.834 |    0.048 |        0.073 |      0.039 |          126 |            0.091 |      20.516 |      -20.468 |
| plaque_v2 |       126 | 126 |         4 |   0.117 |   0.071 |       0.896 |    0.063 |        0.098 |      0.069 |          126 |            0.117 |      20.19  |      -20.19  |
| plaque_v3 |       126 |  55 |         0 |   0.927 |   0.924 |       0.998 |    0.866 |        0.864 |      0.787 |           31 |            0.922 |       2.564 |       -2.564 |
| plaque_v4 |       126 |  21 |         0 |   0.837 |   0.7   |       0.884 |    0.794 |        0.693 |      0.687 |            0 |                  |       3.476 |       -1.857 |
| cpsam     |       126 | 126 |         4 |   0.001 |   0     |       0.016 |    0.001 |        0.002 |      0.001 |          126 |            0.001 |      20.905 |      -20.698 |
| cpsam_v2  |       126 | 126 |         4 |   0     |   0     |       0     |    0     |        0     |      0     |          126 |            0     |      20.698 |      -20.302 |

**gel_doc_model_review_heldout**

| model     |   n_total |   n |   n_empty |   f1_50 |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   n_no_group |   f1_50_no_group |   count_mae |   count_bias |
|:----------|----------:|----:|----------:|--------:|--------:|------------:|---------:|-------------:|-----------:|-------------:|-----------------:|------------:|-------------:|
| plaque_v1 |        30 |  30 |         0 |   0.218 |   0.152 |       0.913 |    0.124 |        0.152 |      0.134 |           30 |            0.218 |      26.8   |      -26.8   |
| plaque_v2 |        30 |  30 |         0 |   0.266 |   0.176 |       0.973 |    0.154 |        0.166 |      0.147 |           30 |            0.266 |      26.1   |      -26.1   |
| plaque_v3 |        30 |  30 |         0 |   0.921 |   0.92  |       1     |    0.854 |        0.868 |      0.811 |           30 |            0.921 |       4.533 |       -4.533 |
| plaque_v4 |        30 |  24 |         0 |   0.801 |   0.681 |       0.909 |    0.716 |        0.651 |      0.504 |           24 |            0.801 |       6.708 |       -6.625 |
| cpsam     |        30 |  30 |         0 |   0     |   0     |       0     |    0     |        0     |      0     |           30 |            0     |      30.033 |      -30.033 |
| cpsam_v2  |        30 |  30 |         0 |   0     |   0     |       0     |    0     |        0     |      0     |           30 |            0     |      28.933 |      -28.867 |

**literature_holdout**

| model     |   n_total |   n |   n_empty |   f1_50 |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   n_no_group |   f1_50_no_group |   count_mae |   count_bias |
|:----------|----------:|----:|----------:|--------:|--------:|------------:|---------:|-------------:|-----------:|-------------:|-----------------:|------------:|-------------:|
| plaque_v1 |        20 |  20 |         1 |   0.558 |   0.43  |       0.491 |    0.645 |        0.427 |      0.373 |           20 |            0.558 |       20.35 |        11.85 |
| plaque_v2 |        20 |  20 |         1 |   0.602 |   0.535 |       0.494 |    0.768 |        0.503 |      0.452 |           20 |            0.602 |       24.95 |        21.05 |
| plaque_v3 |        20 |  20 |         1 |   0.58  |   0.514 |       0.561 |    0.601 |        0.387 |      0.337 |           20 |            0.58  |       15.15 |         2.75 |
| plaque_v4 |        20 |  20 |         1 |   0.573 |   0.492 |       0.514 |    0.647 |        0.443 |      0.362 |           20 |            0.573 |       19.9  |         9.9  |
| cpsam     |        20 |  20 |         1 |   0.182 |   0.162 |       0.164 |    0.205 |        0.136 |      0.155 |           20 |            0.182 |       53.6  |         9.6  |
| cpsam_v2  |        20 |  20 |         1 |   0.144 |   0.123 |       0.127 |    0.167 |        0.115 |      0.117 |           20 |            0.144 |       56.05 |        11.95 |

**pmc_staged**

| model     |   n_total |   n |   n_empty |   f1_50 |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   n_no_group |   f1_50_no_group |   count_mae |   count_bias |
|:----------|----------:|----:|----------:|--------:|--------:|------------:|---------:|-------------:|-----------:|-------------:|-----------------:|------------:|-------------:|
| plaque_v1 |       342 | 289 |        48 |   0.802 |   0.674 |       0.856 |    0.754 |        0.692 |      0.609 |          233 |            0.787 |      12.855 |       -5.298 |
| plaque_v2 |       342 | 105 |        36 |   0.599 |   0.442 |       0.483 |    0.786 |        0.542 |      0.455 |           65 |            0.448 |      26     |       19.238 |
| plaque_v3 |       342 |  94 |        29 |   0.638 |   0.447 |       0.53  |    0.8   |        0.566 |      0.467 |           32 |            0.189 |      22.798 |       17.819 |
| plaque_v4 |       342 | 220 |        50 |   0.767 |   0.556 |       0.72  |    0.82  |        0.671 |      0.542 |          177 |            0.763 |      11.691 |        5.673 |
| cpsam     |       342 | 342 |        61 |   0.494 |   0.361 |       0.475 |    0.515 |        0.476 |      0.36  |          342 |            0.494 |      40.389 |        3.711 |
| cpsam_v2  |       342 | 342 |        61 |   0.497 |   0.352 |       0.525 |    0.472 |        0.428 |      0.32  |          342 |            0.497 |      35.374 |       -4.468 |

**POOLED (deduplicated)**

| model     |   n_total |   n |   n_empty |   f1_50 |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   n_no_group |   f1_50_no_group |   count_mae |   count_bias |
|:----------|----------:|----:|----------:|--------:|--------:|------------:|---------:|-------------:|-----------:|-------------:|-----------------:|------------:|-------------:|
| plaque_v1 |       562 | 498 |        51 |   0.707 |   0.581 |       0.83  |    0.616 |        0.657 |      0.427 |          442 |            0.678 |      16.627 |      -10.321 |
| plaque_v2 |       562 | 282 |        39 |   0.455 |   0.342 |       0.5   |    0.417 |        0.514 |      0.242 |          242 |            0.34  |      24.975 |       -4.812 |
| plaque_v3 |       562 | 232 |        30 |   0.732 |   0.569 |       0.676 |    0.798 |        0.628 |      0.585 |          146 |            0.674 |      13.009 |        6.5   |
| plaque_v4 |       562 | 329 |        54 |   0.765 |   0.56  |       0.734 |    0.798 |        0.662 |      0.548 |          265 |            0.76  |      10.751 |        3.644 |
| cpsam     |       562 | 562 |        67 |   0.295 |   0.212 |       0.242 |    0.378 |        0.293 |      0.239 |          562 |            0.295 |      64.854 |       23.125 |
| cpsam_v2  |       562 | 562 |        67 |   0.346 |   0.241 |       0.353 |    0.34  |        0.272 |      0.207 |          562 |            0.346 |      42.42  |       -1.541 |


## (b) Common subset (identical fields for every model)

### PV

The primary subset is the pooled fields unseen by **all of v1–v5**: n = 163.

PV v5 is marked as leaked: it trained or validated on 29 of the 39 literature hold-out fields and on 3 of the 11 anchors. It is scored only on fields it never saw. The secondary subset, **unseen by v1–v4**, adds the 29 literature fields v5 saw. In that subset, v5 is unseen on 163 of its 192 fields, so v5's row there is not a clean hold-out.

**common**: n = 163 fields; by source {'pv_round3_queue': 130, 'pv_r7_plate_test20': 20, 'pv_literature_holdout': 10, 'pv_compare50_test': 3}

| model    |   n |   n_empty |   gt_objects |   f1_50 |   f1_50_ci_lo |   f1_50_ci_hi |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |
|:---------|----:|----------:|-------------:|--------:|--------------:|--------------:|--------:|------------:|---------:|-------------:|-----------:|
| pv_v1    | 163 |        40 |         4221 |   0.783 |         0.71  |         0.83  |   0.718 |       0.88  |    0.706 |        0.545 |      0.264 |
| pv_v2    | 163 |        40 |         4221 |   0.761 |         0.691 |         0.806 |   0.679 |       0.794 |    0.731 |        0.517 |      0.32  |
| pv_v3    | 163 |        40 |         4221 |   0.79  |         0.726 |         0.835 |   0.722 |       0.858 |    0.733 |        0.585 |      0.358 |
| pv_v4    | 163 |        40 |         4221 |   0.781 |         0.725 |         0.817 |   0.67  |       0.776 |    0.787 |        0.648 |      0.435 |
| pv_v5    | 163 |        40 |         4221 |   0.633 |         0.544 |         0.695 |   0.527 |       0.523 |    0.802 |        0.465 |      0.305 |
| cpsam    | 163 |        40 |         4221 |   0.377 |         0.288 |         0.455 |   0.307 |       0.276 |    0.593 |        0.032 |      0.134 |
| cpsam_v2 | 163 |        40 |         4221 |   0.323 |         0.234 |         0.405 |   0.261 |       0.219 |    0.61  |        0.032 |      0.126 |

**common_excl_v5**: n = 192 fields; by source {'pv_round3_queue': 130, 'pv_literature_holdout': 39, 'pv_r7_plate_test20': 20, 'pv_compare50_test': 3}

| model    |   n |   n_empty |   gt_objects |   f1_50 |   f1_50_ci_lo |   f1_50_ci_hi |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |
|:---------|----:|----------:|-------------:|--------:|--------------:|--------------:|--------:|------------:|---------:|-------------:|-----------:|
| pv_v1    | 192 |        40 |         4818 |   0.731 |         0.645 |         0.789 |   0.656 |       0.83  |    0.654 |        0.497 |      0.27  |
| pv_v2    | 192 |        40 |         4818 |   0.726 |         0.655 |         0.777 |   0.628 |       0.773 |    0.683 |        0.469 |      0.316 |
| pv_v3    | 192 |        40 |         4818 |   0.751 |         0.683 |         0.802 |   0.664 |       0.833 |    0.683 |        0.514 |      0.356 |
| pv_v4    | 192 |        40 |         4818 |   0.749 |         0.691 |         0.792 |   0.622 |       0.758 |    0.741 |        0.565 |      0.413 |
| pv_v5    | 192 |        40 |         4818 |   0.65  |         0.573 |         0.708 |   0.539 |       0.549 |    0.796 |        0.463 |      0.359 |
| cpsam    | 192 |        40 |         4818 |   0.401 |         0.318 |         0.47  |   0.333 |       0.299 |    0.611 |        0.034 |      0.167 |
| cpsam_v2 | 192 |        40 |         4818 |   0.353 |         0.269 |         0.426 |   0.297 |       0.244 |    0.64  |        0.034 |      0.168 |


**Sensitivity to the 98 paged round-3 fields.** They make up most of the PV common subset: 98 of its 130 round-3 fields, and 39 of its 40 reviewed-empty fields.

**Pooled strict hold-out WITHOUT the 98 paged fields**

| model    | source                |   n_total |   n |   n_empty |   f1_50 |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   n_no_group |   f1_50_no_group |
|:---------|:----------------------|----------:|----:|----------:|--------:|--------:|------------:|---------:|-------------:|-----------:|-------------:|-----------------:|
| pv_v1    | POOLED (deduplicated) |       557 | 388 |        69 |   0.787 |   0.725 |       0.855 |    0.729 |        0.542 |      0.519 |          291 |            0.647 |
| pv_v2    | POOLED (deduplicated) |       557 | 160 |         3 |   0.666 |   0.556 |       0.764 |    0.59  |        0.399 |      0.381 |          120 |            0.582 |
| pv_v3    | POOLED (deduplicated) |       557 | 171 |         4 |   0.712 |   0.631 |       0.77  |    0.662 |        0.493 |      0.46  |          120 |            0.64  |
| pv_v4    | POOLED (deduplicated) |       557 | 111 |         2 |   0.731 |   0.588 |       0.781 |    0.687 |        0.541 |      0.434 |           40 |            0.463 |
| pv_v5    | POOLED (deduplicated) |       557 | 383 |        72 |   0.758 |   0.66  |       0.686 |    0.846 |        0.62  |      0.507 |          142 |            0.737 |
| cpsam    | POOLED (deduplicated) |       557 | 557 |        74 |   0.475 |   0.398 |       0.375 |    0.645 |        0.084 |      0.201 |          557 |            0.475 |
| cpsam_v2 | POOLED (deduplicated) |       557 | 557 |        74 |   0.473 |   0.395 |       0.367 |    0.665 |        0.087 |      0.211 |          557 |            0.473 |

**common_without_paged**: n = 65; by source {'pv_round3_queue': 32, 'pv_r7_plate_test20': 20, 'pv_literature_holdout': 10, 'pv_compare50_test': 3}

| model    |   n |   n_empty |   gt_objects |   f1_50 |   f1_50_ci_lo |   f1_50_ci_hi |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |
|:---------|----:|----------:|-------------:|--------:|--------------:|--------------:|--------:|------------:|---------:|-------------:|-----------:|
| pv_v1    |  65 |         1 |         1942 |   0.732 |         0.578 |         0.8   |   0.665 |       0.87  |    0.632 |        0.428 |      0.269 |
| pv_v2    |  65 |         1 |         1942 |   0.742 |         0.62  |         0.798 |   0.637 |       0.832 |    0.669 |        0.444 |      0.316 |
| pv_v3    |  65 |         1 |         1942 |   0.754 |         0.649 |         0.816 |   0.704 |       0.868 |    0.667 |        0.528 |      0.384 |
| pv_v4    |  65 |         1 |         1942 |   0.757 |         0.653 |         0.804 |   0.635 |       0.807 |    0.714 |        0.628 |      0.465 |
| pv_v5    |  65 |         1 |         1942 |   0.665 |         0.54  |         0.732 |   0.541 |       0.603 |    0.742 |        0.563 |      0.38  |
| cpsam    |  65 |         1 |         1942 |   0.427 |         0.308 |         0.522 |   0.341 |       0.33  |    0.606 |        0.059 |      0.276 |
| cpsam_v2 |  65 |         1 |         1942 |   0.323 |         0.194 |         0.447 |   0.253 |       0.22  |    0.604 |        0.055 |      0.245 |

**common_excl_v5_without_paged**: n = 94; by source {'pv_literature_holdout': 39, 'pv_round3_queue': 32, 'pv_r7_plate_test20': 20, 'pv_compare50_test': 3}

| model    |   n |   n_empty |   gt_objects |   f1_50 |   f1_50_ci_lo |   f1_50_ci_hi |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |
|:---------|----:|----------:|-------------:|--------:|--------------:|--------------:|--------:|------------:|---------:|-------------:|-----------:|
| pv_v1    |  94 |         1 |         2539 |   0.641 |         0.48  |         0.732 |   0.554 |       0.768 |    0.551 |        0.378 |      0.276 |
| pv_v2    |  94 |         1 |         2539 |   0.673 |         0.542 |         0.746 |   0.542 |       0.779 |    0.593 |        0.382 |      0.31  |
| pv_v3    |  94 |         1 |         2539 |   0.683 |         0.571 |         0.753 |   0.591 |       0.812 |    0.588 |        0.431 |      0.372 |
| pv_v4    |  94 |         1 |         2539 |   0.698 |         0.586 |         0.761 |   0.545 |       0.762 |    0.644 |        0.504 |      0.418 |
| pv_v5    |  94 |         1 |         2539 |   0.695 |         0.6   |         0.749 |   0.566 |       0.652 |    0.743 |        0.541 |      0.458 |
| cpsam    |  94 |         1 |         2539 |   0.469 |         0.362 |         0.552 |   0.389 |       0.371 |    0.638 |        0.064 |      0.297 |
| cpsam_v2 |  94 |         1 |         2539 |   0.384 |         0.267 |         0.498 |   0.325 |       0.27  |    0.662 |        0.061 |      0.289 |

**The 98 paged fields alone (pooled, strict hold-out)**

| model    | source                |   n_total |   n |   n_empty |   f1_50 |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   n_no_group |   f1_50_no_group |
|:---------|:----------------------|----------:|----:|----------:|--------:|--------:|------------:|---------:|-------------:|-----------:|-------------:|-----------------:|
| pv_v1    | POOLED (deduplicated) |        98 |  98 |        39 |   0.824 |   0.76  |       0.887 |    0.769 |        0.698 |      0.259 |           38 |            0.395 |
| pv_v2    | POOLED (deduplicated) |        98 |  98 |        39 |   0.776 |   0.71  |       0.768 |    0.785 |        0.614 |      0.325 |            0 |                  |
| pv_v3    | POOLED (deduplicated) |        98 |  98 |        39 |   0.818 |   0.736 |       0.85  |    0.789 |        0.66  |      0.331 |            0 |                  |
| pv_v4    | POOLED (deduplicated) |        98 |  98 |        39 |   0.799 |   0.696 |       0.755 |    0.849 |        0.675 |      0.405 |            0 |                  |
| pv_v5    | POOLED (deduplicated) |        98 |  98 |        39 |   0.611 |   0.517 |       0.476 |    0.854 |        0.387 |      0.246 |           42 |            0.656 |
| cpsam    | POOLED (deduplicated) |        98 |  98 |        39 |   0.341 |   0.283 |       0.242 |    0.581 |        0.021 |      0.032 |           98 |            0.341 |
| cpsam_v2 | POOLED (deduplicated) |        98 |  98 |        39 |   0.323 |   0.268 |       0.219 |    0.616 |        0.023 |      0.037 |           98 |            0.323 |


### Plaque

The common subset is the pooled fields unseen by all of v1–v4: n = 114.

21 + 12 of these come from the Gel Doc model_review truth, which was **seeded with plaque v3**, and part of that truth is v3 output accepted unchanged. The second table drops those fields.

**common**: n = 114 fields; by source {'plaque_pmc_staged': 48, 'plaque_gel_doc_model_review': 21, 'plaque_literature_holdout': 20, 'plaque_gel_doc': 12, 'plaque_gel_doc_model_review_heldout': 12, 'plaque_compare263_test': 1}

| model     |   n |   n_empty |   gt_objects |   f1_50 |   f1_50_ci_lo |   f1_50_ci_hi |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   count_mae |   count_bias |
|:----------|----:|----------:|-------------:|--------:|--------------:|--------------:|--------:|------------:|---------:|-------------:|-----------:|------------:|-------------:|
| plaque_v1 | 114 |        28 |         2867 |   0.402 |         0.3   |         0.504 |   0.324 |       0.41  |    0.395 |        0.392 |      0.187 |      24.632 |       -0.912 |
| plaque_v2 | 114 |        28 |         2867 |   0.36  |         0.249 |         0.472 |   0.279 |       0.3   |    0.45  |        0.446 |      0.202 |      35.009 |       12.518 |
| plaque_v3 | 114 |        28 |         2867 |   0.548 |         0.449 |         0.645 |   0.448 |       0.448 |    0.707 |        0.374 |      0.481 |      21.518 |       14.535 |
| plaque_v4 | 114 |        28 |         2867 |   0.534 |         0.44  |         0.625 |   0.397 |       0.441 |    0.675 |        0.417 |      0.384 |      21.237 |       13.289 |
| cpsam     | 114 |        28 |         2867 |   0.09  |         0.044 |         0.168 |   0.067 |       0.059 |    0.185 |        0.122 |      0.096 |      88.184 |       52.956 |
| cpsam_v2  | 114 |        28 |         2867 |   0.106 |         0.056 |         0.171 |   0.076 |       0.079 |    0.161 |        0.114 |      0.081 |      61.596 |       26.316 |

**common_without_v3_seeded**: n = 81 fields; by source {'plaque_pmc_staged': 48, 'plaque_literature_holdout': 20, 'plaque_gel_doc': 12, 'plaque_compare263_test': 1}

| model     |   n |   n_empty |   gt_objects |   f1_50 |   f1_50_ci_lo |   f1_50_ci_hi |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   count_mae |   count_bias |
|:----------|----:|----------:|-------------:|--------:|--------------:|--------------:|--------:|------------:|---------:|-------------:|-----------:|------------:|-------------:|
| plaque_v1 |  81 |        28 |         2030 |   0.472 |         0.359 |         0.586 |   0.382 |       0.41  |    0.555 |        0.394 |      0.258 |      24.481 |        8.877 |
| plaque_v2 |  81 |        28 |         2030 |   0.406 |         0.291 |         0.524 |   0.316 |       0.3   |    0.632 |        0.448 |      0.266 |      39.136 |       27.753 |
| plaque_v3 |  81 |        28 |         2030 |   0.444 |         0.333 |         0.553 |   0.317 |       0.341 |    0.637 |        0.371 |      0.289 |      29     |       21.741 |
| plaque_v4 |  81 |        28 |         2030 |   0.459 |         0.349 |         0.563 |   0.319 |       0.356 |    0.646 |        0.416 |      0.273 |      27.691 |       20.457 |
| cpsam     |  81 |        28 |         2030 |   0.097 |         0.048 |         0.196 |   0.072 |       0.06  |    0.261 |        0.136 |      0.135 |     114.173 |       84.444 |
| cpsam_v2  |  81 |        28 |         2030 |   0.118 |         0.064 |         0.197 |   0.085 |       0.079 |    0.228 |        0.117 |      0.114 |      76.827 |       46.877 |


**Pooled strict hold-out without the v3-seeded Gel Doc model_review truth**

| model     | source                |   n_total |   n |   n_empty |   f1_50 |   f1_75 |   precision |   recall |   aji_pooled |   aji_mean |   n_no_group |   f1_50_no_group |   count_mae |   count_bias |
|:----------|:----------------------|----------:|----:|----------:|--------:|--------:|------------:|---------:|-------------:|-----------:|-------------:|-----------------:|------------:|-------------:|
| plaque_v1 | POOLED (deduplicated) |       495 | 431 |        51 |   0.74  |   0.608 |       0.83  |    0.667 |        0.658 |      0.494 |          375 |            0.716 |      15.724 |       -8.443 |
| plaque_v2 | POOLED (deduplicated) |       495 | 215 |        39 |   0.505 |   0.381 |       0.5   |    0.511 |        0.518 |      0.309 |          175 |            0.394 |      25.786 |        0.66  |
| plaque_v3 | POOLED (deduplicated) |       495 | 165 |        30 |   0.699 |   0.507 |       0.628 |    0.787 |        0.627 |      0.497 |          103 |            0.63  |      16.885 |       10.545 |
| plaque_v4 | POOLED (deduplicated) |       495 | 296 |        54 |   0.762 |   0.552 |       0.727 |    0.801 |        0.662 |      0.535 |          253 |            0.759 |      11.348 |        4.53  |
| cpsam     | POOLED (deduplicated) |       495 | 495 |        67 |   0.303 |   0.218 |       0.242 |    0.405 |        0.309 |      0.273 |          495 |            0.303 |      70.681 |       29.202 |
| cpsam_v2  | POOLED (deduplicated) |       495 | 495 |        67 |   0.359 |   0.25  |       0.354 |    0.364 |        0.275 |      0.236 |          495 |            0.359 |      45.257 |        1.152 |


## (c) Plaque count per well (predicted minus GT)

The `count_mae` and `count_bias` columns in the plaque tables above give these per source and pooled.

**Common subset, n = 114**

| model | MAE | bias |
|---|---|---|
| v4 | 21.2 | +13.3 |
| v3 | 21.5 | +14.5 |
| v1 | 24.6 | −0.9 |
| v2 | 35.0 | +12.5 |
| cpsam_v2 | 61.6 | +26.3 |
| cpsam | 88.2 | +53.0 |

**Pooled strict hold-out**

| model | n | MAE | bias |
|---|---|---|---|
| v4 | 329 | 10.8 | +3.6 |
| v3 | 232 | 13.0 | +6.5 |
| v1 | 498 | 16.6 | −10.3 |
| v2 | 282 | 25.0 | −4.8 |

**Gel Doc wells (unseeded queue)**
- v1, v2 and both stock models under-count by about 26–28 plaques per well (bias ≈ −MAE), meaning they find almost nothing.
- v3 (n = 18 unseen) has MAE 5.2 and bias −4.1.
- v4 (n = 12 unseen) has MAE 8.3 and bias −7.8.

**Plaque literature hold-out (n = 20, unseen by every model)**
- Every model over-counts:
  - v1: bias +11.9
  - v2: +21.1
  - v3: +2.8
  - v4: +9.9
- MAE runs from 15 to 25.

## (d) Best model per view compared with the current defaults

### PV (default `toxoplasma_pv_v4`)

**Common subset, unseen by v1–v5 (n = 163)**
- F1@0.5:
  - v3: 0.790 (CI 0.726–0.835)
  - v1: 0.783
  - v4: 0.781 (CI 0.725–0.817)
  - v2: 0.761
  - v5: 0.633
  - stock cpsam: 0.377
  - stock cpsam_v2: 0.323
- v3 − v4 = +0.009 (95% CI -0.016 to +0.033). **This is a tie.**
- At F1@0.75, v3 leads (0.722 against 0.670).
- On pooled AJI, v4 leads (0.648 against 0.585).
- v4 is the only fine-tuned model with near-zero count bias (+0.4).

**Common subset, unseen by v1–v4 (n = 192)**
- F1@0.5: v3 0.751 and v4 0.749, difference +0.001 (95% CI -0.024 to +0.024). A tie.

**Without the 98 paged fields (n = 65)**
- F1@0.5: v4 0.757, v3 0.754, v2 0.742 and v1 0.732, with overlapping CIs.

**Pooled strict hold-out**
- v1 has the highest F1@0.5: 0.793 on n = 486, against v4's 0.761 on n = 209.
- This is not a like-for-like comparison. v1 is scored on many more round-3 fields, which every model finds easy, so this view does not rank the models.

**PV plate-test20 (Replication09182026, all unseen at field level)**
- v3 F1 0.899 against v4 0.748; paired difference +0.151 (CI +0.080 to +0.221).
- This reproduces the zoo note that v4 under-performs on that plate. For v4 and v5 these fields share a plate with their training data.

**PV literature hold-out (39 fields, v1–v4 unseen)**
- **Stock models beat every fine-tuned model**: cpsam_v2 0.643 and cpsam 0.593.
- The best fine-tuned model is v4 at 0.464; v3 scores 0.392 and v1 0.314.
- cpsam_v2 − v4 = +0.179 (CI −0.007 to +0.347).
- On the 10 literature fields v5 never saw:
  - v5 0.351
  - v4 0.327
  - cpsam_v2 0.386
  - cpsam 0.353

**Verdict:** no PV model is distinguishable from the default v4 on the common subsets. The data give no reason to change the default.
- v3 ties v4 on F1@0.5, is ahead at IoU 0.75, and is clearly ahead on the Replication09182026 plate.
- v4 is ahead on AJI and count bias.
- v5 is clearly worse on the common subset. Its precision is 0.523, driven by false positives on reviewed-empty fields.

### Plaque (default `toxoplasma_plaque_v4`)

**Common subset, unseen by v1–v4 (n = 114)**
- F1@0.5:
  - v3: 0.548 (CI 0.449–0.645)
  - v4: 0.534 (CI 0.440–0.625)
  - v1: 0.402
  - v2: 0.360
  - stock models: ≤ 0.106
- v3 − v4 = +0.015 (95% CI -0.012 to +0.045). A tie.
- v1 − v4 = −0.131 (CI −0.218 to −0.051).
- v4 has the lowest count MAE.

**Without the v3-seeded Gel Doc truth (n = 81)**
- F1@0.5: v1 0.472, v4 0.459, v3 0.444, v2 0.406.
- Paired differences: v1 − v4 = +0.013 (CI −0.057 to +0.091) and v3 − v4 = −0.014 (CI −0.038 to +0.013).
- **No model differs from v4.**

**Pooled strict hold-out**
- v4 is best on F1@0.5 (0.765, n = 329) and on AJI (0.662).
- v3 scores 0.732 (n = 232) and v1 0.707 (n = 498).

**Gel Doc (unseeded queue)**
- Only v3 and v4 work; v1, v2 and both stock models score F1 ≤ 0.18.
- On unseen wells: v3 0.642 (n = 18) and v4 0.454 (n = 12). The n is small and the wells differ between the two models.

**Verdict:** the default **plaque v4 is best or tied-best in every view.** v3 ties it on the common subset, where part of the truth was seeded by v3.

## (e) Caveats

1. **Small n.** The fully clean pools are small:
   - plaque literature: 20 fields
   - PV literature: 39 fields, of which only 10 are unseen by v5
   - PV common subset without the paged fields: 65
   - plaque common subset without the v3-seeded truth: 81

   Most differences between the leading fine-tuned models fall inside the bootstrap CIs.
2. **The paged round-3 PV fields** (98, no ledger row) are included by the maintainer's decision. 39 of them are empty masks. They dominate the PV common subset, and they penalise high-recall, low-precision models (v5, the stock models). The tables with and without them are both given above.
3. **v3-seeded Gel Doc truth.**
   - The model_review batches were drafted by plaque v3, and 57 review fields were accepted as drawn (Keep-only).
   - v3's high scores there (F1 ≈ 0.92) are biased in its favour.
   - The unseeded `gel_doc_2026-10-05` queue is preferred in the pool, and a common-subset table without the seeded truth is given.
   - The pmc_staged drafts came from the plaque round-3 lineage (`cpsam_seg_r3`), so they may favour v1-like output.
4. **Reconstructed splits.**
   - PV v1 and plaque v1 splits were rebuilt from code and seeds, not from saved lists.
   - The 98 paged fields had no manifest record, so their seen flags were computed here with `build_manifest.py`'s own matcher, by stem. They were in no model's training split.
   - Round-3 bundle images have no file sha, so they match by name only, as in step 1.
5. **Leakage beyond the field level.**
   - Strict hold-out removes only exact field matches.
   - The `*_no_group` columns show that fields sharing a plate, acquisition or paper with training data often score higher. Example: PV v1 on compare50 scores 0.752 overall but 0.178 on fields with no shared acquisition, n = 13.
   - PV v2–v4 have no plate-level hold-out among the round-3 fields, and v4 and v5 have none on plate-test20.
6. **PV v5 is leaked** on the literature hold-out (29 of 39) and the anchors (3 of 11). It is scored only where unseen. Even there, v5 shares papers and acquisitions with most fields; see the `no_group` columns.
7. **Frozen copies.**
   - The 263- and 50-field comparison sets are not reproduced as such.
   - Stale copies were dropped, and duplicates of live fields were scored in their own rows but not pooled. Their source rows are therefore subsets, and not comparable with the numbers of 10-09 and 10-10.
8. **pmc_staged** may still be under curation; its ledger was last written 2026-10-06. The manifest snapshot is from 2026-10-11.
9. **Pooled AJI vs mean AJI.** Pooled AJI is dominated by fields with large or many objects; mean AJI weights every field equally. The two can rank models differently. Example: plaque common subset, where v2 has the highest pooled AJI but a low mean AJI.
10. **Defaults settings only.** Every model ran at diameter 30 and the other defaults listed above. A model trained at a different scale may do better with tuned settings.

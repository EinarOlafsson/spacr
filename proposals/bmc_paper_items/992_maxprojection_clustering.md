# Item 992: max-projection clustering over feature views (OPS clustering, alpha)

- **Number:** 992. Checked free on 2026-11-19 against `features/feature_list` (highest 988 on origin/nightly), `proposals/` (no numbered spec at 992) and `/mnt/firecuda2/codex/scratch-docs/status/*CLAIMS*` (no 99x claims). 990 and 991 are taken (branches `claude-990-cellcycle-1119`, `codex-tr-991`). The master confirms the reservation.
- **Status:** SPEC ONLY; design card SIGNED OFF 2026-11-19 (maintainer, options 1a–8a). Design card: `BMCspacr/reanalysis/experiment_designs/20261119_maxprojection_clustering.md`. Nothing is implemented and nothing has run.
- **Requested by the maintainer (2026-11-19):** some phenotypes are broad and need many features, others are thin and show only on the few columns that carry signal. Do a broad pass, narrow down in different directions by removing different columns, save the structure at each level, then project the strongest pairs and clusters across levels, "like a maximum intensity projection".
- **Readiness:** Under development. Feature workflow: script proof on planted data (card §4, §8) → this item → spaCR reproduces the script proof identically → production run at the step-4 freeze. No paper number before that run.
- **Alpha:** every setting below ships behind "Show alpha features" and is off by default. With `ops_maxproj_enabled` False, every existing output is byte-identical to nightly.

Abbreviations: OPS optical pooled screen; NT non-targeting; TVN typical variation normalisation; PCA principal component analysis; PHATE potential of heat-diffusion for affinity-based transition embedding; BH Benjamini–Hochberg; max-T maximum-statistic permutation adjustment (Westfall–Young); CORUM comprehensive resource of mammalian protein complexes; STRING protein-interaction database; SNF similarity network fusion; COSA clustering objects on subsets of attributes; k-NN k nearest neighbours.

## 1. What it does

A new alpha phase `cluster_maxproj` of the OPS profile path (next to `cluster_consensus` and `cluster_bench_test` in `spacr/ops_engine.py`'s phase list):

1. **Views.** Read a views file (`ops_maxproj_views`, JSON: name, block, column list; written by the user or by the script that builds views from a selection half). Inside each view: standardise the view's columns, embed (standardise + PCA to `ops_maxproj_pca_variance`), then whiten in the PCA space by the NT gene covariance (gene-level TVN; a half has only about 124 NT rows for about 2,700 columns, so whitening must follow the PCA), then the cosine similarity of every gene pair.
2. **Per-view calibration.** Guide-level label permutation (guides shuffled between targeting genes, guide count per gene kept, NT guides fixed), gene rows rebuilt by the cell-weighted guide mean. The empirical null of each view's pair similarities, pooled per bin of the two genes' cell counts (`ops_maxproj_bins`), turns every similarity into a normal score c_v(i, j). Seeds `ops_maxproj_seed + p`.
3. **Max projection.** M(i, j) = max over views of c_v(i, j); the argmax view is saved per pair.
4. **Max-statistic null.** `ops_maxproj_null_permutations` fresh joint permutations (one shuffle for all views), the same calibration tables, running max kept; pair p-value from the pooled max null within the bin; BH q over pairs. Seeds `ops_maxproj_seed + 100000 + p`.
5. **Graph and Leiden.** Each gene's `ops_maxproj_knn` strongest partners by −log10 p, edges kept below `ops_maxproj_edge_p` (each gene keeps at least its best edge), or PHATE on the projected matrix (`ops_maxproj_graph` `'phate'`). Leiden over `ops_maxproj_resolutions` and `ops_maxproj_leiden_seeds`, plus the matched count (928's `round(n × 209 / 2,647)` or `ops_maxproj_k`). Optional guide-bootstrap consensus on the projected graph (`ops_maxproj_consensus`).
6. **Cluster labels by view.** Per cluster: the share of within-cluster edges by argmax view and view block, and the label broad / thin / mixed (card §2.5).

## 2. Reuse (no duplicated helpers)

Everything below exists on nightly and is called, not copied:

| Need | Existing code |
|---|---|
| Embedding per view | `spacr.ops_reference.embed_gene_profiles` |
| NT whitening per view | `spacr.ops_tvn.tvn_on_controls` (on gene rows, NT rows as controls, one batch) |
| Rebuilding gene rows from guides | the shift of `spacr.ops_clusters.guide_bootstrap_profiles`. A small refactor splits it into `_guide_groups(...)` (members per gene) and `_guide_shift(rows, groups, drawn, ...)`, which the bootstrap and the new permutation draw both call. Bootstrap output stays bit-identical (tested). |
| Control rows | `spacr.ops_clusters._is_control_row` |
| PHATE on a precomputed matrix | `spacr.ops_clusters.fused_phate_graph` (item 948) |
| k-NN graph | `spacr.ops_clusters.gene_graph` (`kind='knn'`). Its edge loop moves into `_knn_graph(similarity, k)`, which `gene_graph` and the new phase both call. `gene_graph`'s output is unchanged (tested). |
| Leiden, matched count | `leiden_partitions`, `match_resolution`, `match_count`, `target_cluster_count` |
| Consensus on the projected graph | `guide_bootstrap_consensus` gets an optional `graph_of(rows, seed)` hook (default: today's PHATE graph, so 928 is unchanged); the phase passes the projected-graph builder |
| Multiple testing | `spacr.multiple_testing.adjust_p_values` (`fdr_bh`) |
| Scoring | `score_partition` / `score_partitions` (Brieflow's scorer) and `cluster_bench_test_phase` (929) read the new partition file like any other |
| SNF comparator over the same views | `snf_fuse` |
| Tables | `spacr.tabular.write_table` |

New code is only the views reader, the binned calibration, the max-null loop and the view labelling, in a new module `spacr/ops_maxproj.py` (one public entry `cluster_maxproj_phase(settings, genes, guides, folder, report=None)` plus `calibrate_view`, `max_projection`, `max_null_pvalues`, `label_clusters_by_view`). Each new documented callable ships with its arrival contract in the same commit.

## 3. Settings (all alpha, `ops_maxproj_*`, defaults off)

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `ops_maxproj_enabled` | bool | False | run the phase |
| `ops_maxproj_rows` | path or None | None | gene table in named measurement columns (`gene`, `n_cells`, columns); None: the profile phase's per-column gene table |
| `ops_maxproj_guides` | path or None | None | guide table (`guide`, `gene`, `n_cells`, same columns) |
| `ops_maxproj_views` | path | None (required when enabled) | views JSON; its md5 is written to the summary |
| `ops_maxproj_pca_variance` | float | 0.99 | per-view PCA variance kept |
| `ops_maxproj_bins` | int | 3 | cell-count quantile bins per gene (bins² pair bins) |
| `ops_maxproj_calibration_permutations` | int | 50 | permutations of the per-view null |
| `ops_maxproj_null_permutations` | int | 200 | permutations of the max null |
| `ops_maxproj_seed` | int | 20261119 | base seed of every permutation |
| `ops_maxproj_graph` | str | 'knn' | 'knn' or 'phate' |
| `ops_maxproj_knn` | int | 10 | partners per gene |
| `ops_maxproj_edge_p` | float | 0.05 | edges kept below this max-null p |
| `ops_maxproj_resolutions` | list or None | None | Leiden grid; None: 20 log-spaced values from 0.05 to 50 |
| `ops_maxproj_leiden_seeds` | list | [42, 1, 2] | Leiden seeds |
| `ops_maxproj_k` | int or None | None | matched count; None: `target_cluster_count` |
| `ops_maxproj_consensus` | bool | False | guide-bootstrap consensus on the projected graph (B from `ops_consensus_bootstraps`) |
| `ops_maxproj_save_views` | bool | True | save every view's calibrated pair scores (the structure at each level) |
| `ops_maxproj_workers` | int | 1 | processes; results do not depend on it |

The settings appear in the settings schema, the help text and the settings panel (alpha group). **Translations:** not in this item's commits. At implementation the new labels and help strings go to Codex as `/mnt/firecuda2/codex/scratch-docs/codex_inbox/tr_992_task.md` (sources with SHA-256, 9 locales), as for earlier trains.

## 4. Outputs (`results/ops_maxproj/`)

`views.csv` (view, block, column count, md5), `calibration_bins.parquet`, `pairs_projected.parquet` (gene_a, gene_b, M, view, p, q; upper triangle), `per_view_scores.npz` (when `ops_maxproj_save_views`), `leiden_grid.csv` (resolution, seed, clusters), `maxproj_partition.csv` (gene, cluster, label broad/thin/mixed, targeting), `cluster_views.csv` (cluster × view share), `maxproj_summary.json` (every setting, views md5, input md5s, spaCR version, V and V_eff, timings), and one figure (cluster share by view block; spacr-graphs style, both themes).

## 5. Reproducibility

- Every random draw is seeded from `ops_maxproj_seed` and the permutation index, in a fixed order, independent of `ops_maxproj_workers`.
- Rerunning the saved settings file (`maxproj_summary.json` → settings) on the same inputs gives **byte-identical** output files (tables written with fixed float format and row order; figure metadata without timestamps). This is a test.

## 6. Tests (`tests/test_ops_maxproj.py`, CPU only, `CUDA_VISIBLE_DEVICES=''`, `pytest -n 4` at most)

1. Planted toy (≈ 200 genes × 3 guides, 60 columns): a thin group on 4 columns and a broad group on 40 columns; max projection recovers both; whole-profile only (one view) misses the thin group.
2. Null calibration: on a toy with no structure, the share of pairs with p < 0.05 is within [0.03, 0.07] over 20 seeds; the naive max (no max-T) is inflated.
3. One view equals the whole-profile calibrated graph exactly.
4. Views given in a different order give the same M and p (argmax ties to the lower index are documented).
5. Worker count 1 vs 3: identical outputs.
6. Settings-file rerun: byte-identical files.
7. Refactors: `guide_bootstrap_profiles` and `gene_graph` outputs bit-identical before and after; 928 consensus partitions unchanged on its existing test data.
8. Off-switch: with `ops_maxproj_enabled` False, the OPS path's outputs are byte-identical to nightly.
9. Ratchets before hand-over: raw-reader, nested-function, documented-callable and hash-comment ratchets.

## 7. Acceptance (from the card)

- spaCR reproduces the script planted proof exactly (identical p matrices to 1e-12, identical partitions at fixed seeds).
- Then the card's planted criterion (§8) holds through spaCR, and only then the real step-4 run.
- Comparator rows (Brieflow rerun with Brieflow code plus Di Bernardo et al., Nat Commun 2026, doi 10.1038/s41467-026-73643-7, published values; COSA; SNF over the same views) are produced by the card's scripts, not by this item, except SNF, which uses `snf_fuse`.

## 8. Out of scope

Choosing the views (Test 3 and the card's scripts write the views file); COSA (an R comparator, not shipped in spaCR); making this a default (maintainer decision after the real run).

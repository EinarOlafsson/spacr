# 1012 external max-projection translation task

Branch: `ws-offload-codex-work-1012`; base: `299179d9530ac618d82f11cc823f3ad8abd1c001`.

**Scientific limitation: the fixed matched partition recovered 0% of the
260 planted complexes, in every view/strength stratum including 8x; shuffled
recovery was also 0%. Passing code gates do not establish successful full-size
planted recovery. Maintainer review is required before landing.**

A separate one-CPU/2-GB scheduled audit verified the exact prescribed shifted
values for all 260 complexes and 2,080 unique genes, all eight members covered,
all direction seeds, untouched off-view values, preserved missingness and
unchanged inputs. It added no fit, scoring arm or parameter adjustment.

The owner approved the exact 11 additive arrival-hook lines at 10:19 on 2026-11-22.
Historical 871/911 and 992 arrival records remain immutable. The new source adapter
composes after the existing 1014 adapter. Merge the additive train 11/12 hooks;
the owner explicitly requested no rebase.

Home owns sv, de, es, zh_CN, pt, hi, ko, is and fr. Exact English source bodies,
hashes, locations and admission categories are in `tr_1012_sources.json`.
This branch authors no translation records, generated API catalogs or pins.

## Behaviour and source admission

The opt-in gene caller adds independent within-view column permutations, coverage
masks, no-control embedding, float32 packed pair storage and blocked scoring.
The existing guide-null OPS outputs remain byte-identical against frozen source,
for both similarities and distinct seeds. The existing numerical embedding,
KNN graph, Leiden and enrichment/recovery owners are reused. No settings or
schema changes are included. See `proposals/bmc_paper_items/1012_maxproj_external_caller.md`.

The canonical API inventory retains 14,961 bodies. Nine bodies change, comprising
85 blocks, with no additions or removals. The remaining changed bodies are
private/nested helpers and tests. The 25 argument-validation expressions follow
the existing English numeric-API pattern and have no runtime UI admission.
There are no new GUI controls or runtime `tr` calls. Native source consumers
pass with zero new production imports; all 2,367 protected assertions remain intact.

## Validation

| Group | Console result |
| --- | --- |
| focused core and legacy identity | 75 passed, 20 warnings in 70.28s (0:01:10) |
| source-only arrival contracts | 3 passed in 7.46s |
| full items | 1599 passed, 10 skipped, 153 warnings, 960 subtests passed in 780.76s (0:13:00) |
| full settings | 630 passed, 2 skipped, 369 subtests passed in 128.84s (0:02:08) |
| full ratchets | 4093 passed, 76 skipped, 3 xfailed, 71 warnings in 3106.29s (0:51:46) |

The full code-rule group retains all six mandatory ratchets and the strict
source/native/raw-reader checks. Only the exact 30 standing Home-owned catalog
selectors are deferred; their node IDs are in the receipt. Existing tests that
require a fresh docs build retain their original skip conditions. No docs build
or full pytest suite was run. Runs use scheduler P2, nice 19 and CUDA hidden;
pytest uses at most two workers. Each full benchmark uses three scoring workers.

The 76 ratchet skips comprise 73 fresh-doc-build cases, one opt-in real-document
preflight, one obsolete TSNE-bound check and one missing-PyInstaller check.
The three existing expected failures are registered translation incompatibilities.
JUnit counts encode expected failures as skipped; console counts keep them separate.

## Fixed full-size acceptance

All three prescribed arms use 8,140 genes, 50 calibration draws and 200 joint-null
draws, cosine similarity, seed 20261119, 0.99 variance, three bins, KNN 10, edge
p=0.05, the original 20-resolution/three-seed Leiden grid and existing matched-count
search. All parameters remain fixed. No numeric acceptance cutoff was supplied;
the receipt reports actual metrics without an invented threshold or tuning.

The original inputs have 651 columns and 13 views. Planting uses independent
per-view column copies (653 references) because `n_holes` belongs to three views.
All 260 prescribed complexes are applied simultaneously at their supplied
1x/2x/4x/8x strengths. Only covered members and finite measurements change.
The no-plant arm applies no effects. Niche labels are scored only in the
12-view no-place arm. Direct pairs, EC groups and generated complexes are scored
through the existing 992 owners, with a shuffled-label control. All generated
complexes are evaluation controls and select no parameter.

Only the eight approved input files are read, and their hashes are unchanged
before and after every arm. Manifest provenance paths are not followed. The
receipt binds the exact production source hashes, input hashes, arrays, parameters,
timings and measured memory. Aggregate full-grid and stratified metrics are
committed under `reports/1012_external_maxproj/`; input tables and per-gene
partitions remain outside the repository.

| Arm | Views | Graph edges | Matched clusters | Scoring seconds | Total seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_plant | 13 | 80759 | 642 | 1954.6 | 3492.7 |
| planted | 13 | 80693 | 646 | 3024.8 | 4136.6 |
| no_place | 12 | 80023 | 642 | 2989.4 | 3826.6 |

## Exact changed source locations

- `spacr/ops_maxproj.py:39` — `_packed_indices` (private/nested).
- `spacr/ops_maxproj.py:44` — `_pair_values` (private/nested).
- `spacr/ops_maxproj.py:49` — `_pair_row` (private/nested).
- `spacr/ops_maxproj.py:58` — `_projected_strength_rows` (private/nested).
- `spacr/ops_maxproj.py:80` — `_pair_rows` (private/nested).
- `spacr/ops_maxproj.py:89` — `_covered_pairs` (private/nested).
- `spacr/ops_maxproj.py:101` — `_coverage_pairs` (private/nested).
- `spacr/ops_maxproj.py:106` — `_joint_coverage` (private/nested).
- `spacr/ops_maxproj.py:115` — `_view_arrays` (private/nested).
- `spacr/ops_maxproj.py:134` — `_column_rows` (private/nested).
- `spacr/ops_maxproj.py:143` — `_iter_view_rows` (private/nested).
- `spacr/ops_maxproj.py:159` — `_projection` (private/nested).
- `spacr/ops_maxproj.py:167` — `_projection.scores` (private/nested).
- `spacr/ops_maxproj.py:234` — `pair_bins` (canonical API).
- `spacr/ops_maxproj.py:258` — `view_similarity` (canonical API).
- `spacr/ops_maxproj.py:352` — `_packed_grid_indices` (private/nested).
- `spacr/ops_maxproj.py:382` — `_counts` (private/nested).
- `spacr/ops_maxproj.py:405` — `calibrate_view` (canonical API).
- `spacr/ops_maxproj.py:438` — `max_projection` (canonical API).
- `spacr/ops_maxproj.py:468` — `max_null_pvalues` (canonical API).
- `spacr/ops_maxproj.py:488` — `projected_graph` (canonical API).
- `spacr/ops_maxproj.py:538` — `label_clusters_by_view` (canonical API).
- `spacr/ops_maxproj.py:662` — `_view_job` (private/nested).
- `spacr/ops_maxproj.py:678` — `_max_job` (private/nested).
- `spacr/ops_maxproj.py:704` — `maxproj_scores` (canonical API).
- `spacr/ops_reference.py:581` — `embed_gene_profiles` (canonical API).
- `spacr/ops_reference.py:608` — `_GeneEmbedding.__init__` (private/nested).
- `spacr/ops_clusters.py:1304` — `_knn_rows` (private/nested).
- `spacr/ops_clusters.py:1322` — `_knn_graph` (private/nested).
- `tests/test_ops_maxproj_external_1012.py:1` — `<module>` (test).
- `tests/test_ops_maxproj_external_1012.py:22` — `_one_blas_thread` (test).
- `tests/test_ops_maxproj_external_1012.py:31` — `legacy` (test).
- `tests/test_ops_maxproj_external_1012.py:52` — `_ops_inputs` (test).
- `tests/test_ops_maxproj_external_1012.py:65` — `_assert_result_bytes` (test).
- `tests/test_ops_maxproj_external_1012.py:80` — `test_default_ops_outputs_are_byte_identical_to_frozen_original_code` (test).
- `tests/test_ops_maxproj_external_1012.py:91` — `test_column_null_shuffles_columns_independently_within_covered_genes` (test).
- `tests/test_ops_maxproj_external_1012.py:106` — `_external` (test).
- `tests/test_ops_maxproj_external_1012.py:113` — `test_coverage_excludes_unmeasured_pairs_from_each_view_and_every_null` (test).
- `tests/test_ops_maxproj_external_1012.py:141` — `test_float32_packed_scores_agree_with_float64_square_results` (test).
- `tests/test_ops_maxproj_external_1012.py:158` — `test_packed_pipeline_does_not_allocate_global_upper_triangle_indices` (test).
- `tests/test_ops_maxproj_external_1012.py:162` — `test_packed_pipeline_does_not_allocate_global_upper_triangle_indices.refused` (test).
- `tests/test_ops_maxproj_external_1012.py:173` — `test_packed_projection_streams_views_and_selects_the_first_eligible_tie` (test).
- `tests/test_ops_maxproj_external_1012.py:178` — `test_packed_projection_streams_views_and_selects_the_first_eligible_tie.refused` (test).
- `tests/test_ops_maxproj_external_1012.py:188` — `test_optional_missingness_cutoff_reuses_embedding_and_preserves_ops_default` (test).
- `tests/test_ops_maxproj_external_1012.py:206` — `test_shrunk_similarity_without_controls_uses_all_covered_rows` (test).
- `tests/test_ops_maxproj_external_1012.py:219` — `test_column_null_calibrates_unstructured_gene_profiles` (test).
- `tests/test_ops_maxproj_external_1012.py:236` — `test_column_null_recovers_small_planted_complexes` (test).
- `tests/test_ops_maxproj_external_1012.py:256` — `test_column_draws_keep_identical_bytes_across_worker_counts` (test).
- `tests/test_ops_maxproj_external_1012.py:271` — `legacy_knn` (test).
- `tests/test_ops_maxproj_external_1012.py:283` — `_graph_bytes` (test).
- `tests/test_ops_maxproj_external_1012.py:289` — `test_default_dense_graph_and_labels_match_the_original_owners` (test).
- `tests/test_ops_maxproj_external_1012.py:319` — `test_packed_graph_matches_dense_without_global_pair_or_square_helpers` (test).
- `tests/test_ops_maxproj_external_1012.py:329` — `test_packed_graph_matches_dense_without_global_pair_or_square_helpers.no_global_matrix` (test).
- `tests/test_ops_maxproj_external_1012.py:339` — `test_graph_and_view_labels_exclude_pairs_without_covered_support` (test).
- `tests/test_ops_maxproj_packed_lookup_1012.py:1` — `<module>` (test).
- `tests/test_ops_maxproj_packed_lookup_1012.py:11` — `test_packed_grid_indices_match_searchsorted_at_every_boundary_and_neighbour` (test).
- `tests/test_ops_maxproj_packed_lookup_1012.py:24` — `_square_values` (test).
- `tests/test_ops_maxproj_packed_lookup_1012.py:40` — `test_packed_histogram_counts_are_exactly_the_frozen_legacy_counts` (test).
- `tests/test_ops_maxproj_packed_lookup_1012.py:49` — `test_packed_calibration_scores_match_frozen_legacy_bytes` (test).
- `tests/test_ops_maxproj_sources_1012.py:1` — `<module>` (test).
- `tests/test_ops_maxproj_sources_1012.py:20` — `_external_api_sources` (test).
- `tests/test_ops_maxproj_sources_1012.py:41` — `_current_external_maxproj_api` (test).
- `tests/test_ops_maxproj_sources_1012.py:51` — `_historical_external_arrivals` (test).
- `tests/test_ops_maxproj_sources_1012.py:60` — `test_external_adapter_changes_only_its_exact_symbols_and_is_idempotent` (test).
- `tests/test_ops_maxproj_sources_1012.py:74` — `test_external_adapter_composes_with_existing_never_calibration_revisions` (test).
- `tests/test_ops_maxproj_sources_1012.py:86` — `test_external_adapter_refuses_unbound_third_bodies` (test).
- `spacr/ops_maxproj.py:281` — `view_similarity`: 'max-projection dtype must be float32 or float64' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:283` — `view_similarity`: 'max-projection similarity must be cosine or shrunk' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:285` — `view_similarity`: 'max-projection block_size must be a positive integer' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:288` — `view_similarity`: 'view coverage must have one boolean per gene' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:290` — `view_similarity`: 'view controls must have one boolean per gene' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:511` — `projected_graph`: 'packed graph probabilities require n_genes of at least two' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:514` — `projected_graph`: 'packed graph probabilities must contain every unordered gene pair' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:518` — `projected_graph`: 'graph probabilities must be a square matrix or a packed vector' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:522` — `projected_graph`: 'graph validity must have the same layout as its probabilities' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:747` — `maxproj_scores`: 'max-projection null_mode must be guides or columns' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:749` — `maxproj_scores`: 'max-projection calibration, null and bins must be positive' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:752` — `maxproj_scores`: 'max-projection dtype must be float32 or float64' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:757` — `maxproj_scores`: 'max-projection needs at least two gene rows and matching names' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:759` — `maxproj_scores`: 'max-projection needs at least one view' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:763` — `maxproj_scores`: 'max-projection views must contain valid measurement-column indices' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:767` — `maxproj_scores`: 'max-projection coverage must have shape genes by views' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:770` — `maxproj_scores`: 'max-projection control must be prefix, None or a boolean mask' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:777` — `maxproj_scores`: 'max-projection controls must have one boolean per gene' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:782` — `maxproj_scores`: 'the guide null needs guide rows, gene names and cell counts' (API exception, runtime not admitted).
- `spacr/ops_maxproj.py:792` — `maxproj_scores`: 'max-projection cell counts must have one value per gene' (API exception, runtime not admitted).
- `spacr/ops_reference.py:613` — `_GeneEmbedding.__init__`: 'max_missing must be greater than zero and at most one' (API exception, runtime not admitted).
- `spacr/ops_clusters.py:1306` — `_knn_rows`: 'streamed graph neighbours must be positive and below the row count' (API exception, runtime not admitted).
- `spacr/ops_clusters.py:1310` — `_knn_rows`: 'streamed graph has more rows than declared' (API exception, runtime not admitted).
- `spacr/ops_clusters.py:1313` — `_knn_rows`: 'each streamed graph row must contain one value per vertex' (API exception, runtime not admitted).
- `spacr/ops_clusters.py:1318` — `_knn_rows`: 'streamed graph has fewer rows than declared' (API exception, runtime not admitted).

Machine-local full logs and proofs: `/home/carruthers/Documents/repo/spacr-codex-recovery-20261121/1012/`.

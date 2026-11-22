# 1012 translation source checkpoint

**Not FINAL: the planted and no-place full acceptance runs are pending.**

Branch: `ws-offload-codex-work-1012`; base: `299179d9530ac618d82f11cc823f3ad8abd1c001`.
Home owns sv, de, es, zh_CN, pt, hi, ko, is and fr. Exact English bodies, source
hashes and admission categories are in `tr_1012_sources.json`. No translation
records, generated API catalogs or pins are authored here.

The opt-in external caller adds column-wise null permutations, no-control
embedding, coverage and packed float32 scoring. Existing guide-null outputs
remain byte-identical. Shared embedding, graph and scoring owners are reused.
No settings or schema change. See the short API/null/memory specification in
`proposals/bmc_paper_items/1012_maxproj_external_caller.md`.

The canonical inventory retains 14,961 bodies: nine revisions and 85 blocks,
with no additions/removals. There are 66 changed/new documentation bodies
including private, nested and test scopes, plus 25 validation expressions.
Those exceptions follow the existing English numeric-API pattern; they have
no runtime UI admission. No GUI control or runtime `tr` call is added.

The exact 11 additive hooks in three protected consumers were explicitly
approved at 10:19 on 2026-11-22 (proposal SHA256
`23a618e19174c638865461fa696b0d64e316f1c2df8443f64ef8a7ed7ef7791a`).
All 2,367 assertions remain intact. Historical 871/911 and 992 records remain
immutable. The source adapter composes after the existing 1014 adapter;
merge the additive train 11/12 calls. The owner requested no rebase.

All code gates below passed; native source consumers also passed with zero
new production imports. The full ratchets defer only the exact 30 standing
Home-owned catalog selectors, retaining all mandatory strict source checks.
Existing fresh-doc-build skips remain unchanged. No full suite or docs build.

| Group | Console result |
| --- | --- |
| focused core | 75 passed, 20 warnings in 70.28s (0:01:10) |
| source contracts | 3 passed in 7.46s |
| full items | 1599 passed, 10 skipped, 153 warnings, 960 subtests passed in 780.76s (0:13:00) |
| full settings | 630 passed, 2 skipped, 369 subtests passed in 128.84s (0:02:08) |
| full ratchets | 4093 passed, 76 skipped, 3 xfailed, 71 warnings in 3106.29s (0:51:46) |

The 76 skips comprise 73 fresh-doc-build cases, one opt-in real-document
preflight, one obsolete TSNE-bound check and one missing-PyInstaller check.
The three existing expected failures are registered translation incompatibilities.
JUnit represents expected failures as skipped cases; console counts above keep them separate.

The no-plant 50/200 acceptance is complete: 8,140 genes, 80,759 edges,
642 matched clusters against the fixed target 643. Direct pair TP real/shuffled
is 26/6; EC-enriched clusters are 5.6/2.5%; unapplied complex recovery is 0/0%.
All eight input hashes and the three production source hashes are unchanged.
No numeric acceptance cutoff was supplied and no parameter is tuned.
The final handoff will replace this checkpoint and include all three receipts.

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

Machine-local proofs and logs: `/home/carruthers/Documents/repo/spacr-codex-recovery-20261121/1012/`.

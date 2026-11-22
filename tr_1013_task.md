# Item 1013 English source handoff

Home owns the sv, de, es, zh_CN, pt, hi, ko, is and fr translation records, catalogs and pins.
No translation, generated catalog or translation pin is authored in this branch.

The default neighbour discount is 0.25. Schema 959 preserves effective 1.0 for every older settings era, including 957 and 958. The requested 956 was already below the live schema 958.
The recipe bleed switch remains alpha and False. Its separate paper acceptance is still required before any default change.

Canonical public API: one addition and five revisions, 64 translatable blocks, exact before/current bodies and hashes in tr_1013_sources.json. The shipped public_docstrings and translatable_blocks owners supplied these records; no docs build ran.
The exact frozen consumer patch was approved at 13:26 (32401fd1...8f87). All 2367 original assertion expressions, historical arrival files and pins remain unchanged.
The one-line runtime extractor owner admission was approved at 13:34 (8e6878e9...8d42). The shipped canonical_sources owner recovered the notice, two method labels and actual setting labels/tooltips.

Runtime notice sources:
- cache replay without nucleus positions [a67d8be7d8ba45d05e64bdcabe4b44954e484655f1901141a9e39084f4003d42]
- ops_read_neighbour_discount ({discount}) does not act under {method}; reads are not discounted [cefcfba98821b346c9ff84168740e6721bfd67619641275164e28c07ee94ef95]
- rules off [3124e092c2d4f02ae53f5992a7412d6368a25aed59c7d3da9877add921c3d6f4]

Settings sources:
- setting_labels ops.ops_prefix_bleed_in_recipe: Prefix bleed in recipe
- setting_labels ops.ops_read_neighbour_discount: Read neighbour discount
- setting_labels ops_prefix_bleed_in_recipe: Ops prefix bleed in recipe
- setting_labels ops_read_neighbour_discount: Ops read neighbour discount
- setting_tooltips ops_prefix_bleed_in_recipe: Under ops_prefix_quality_method 'recipe_simple', ignore a minority barcode that has the most reads in a touching cell when ops_read_neighbour_discount is below 1. Tied top barcodes remain. This can rescue calls blocked by neighbour bleed. Default False. Alpha.
- setting_tooltips ops_read_neighbour_discount: Weighs down a minority guide when the same guide has the most reads in a neighbouring cell, with ties kept. Neighbours are nuclei whose discs come within ops_footprint, or touching cells in the spacer-prefix decode. 1 is off and 0 ignores flagged reads. The spacer-prefix decode uses the discount with 'weighted_majority'; under 'recipe_simple', ops_prefix_bleed_in_recipe enables ignoring flagged minority barcodes. A notice reports paths where the default cannot act. Default 0.25.

The discount labels are existing context; reuse current translations. The new alpha label has both generic and OPS display keys. Tooltip bodies above are the exact runtime strings after the type prefix is stripped.

All new or changed documented scopes and message expressions, including private and test prose:
- spacr/ops_sbs.py:1243 neighbour_bleed_flags [canonical API]
- spacr/ops_sbs.py:1291 _discount_neighbour_bleed [private/nested]
- spacr/ops_sbs.py:1358 vote_mapped_reads [canonical API]
- spacr/ops_decode.py:128 resolve_settings [canonical API]
- spacr/ops_decode.py:194 _neighbour_discount_context [private/nested]
- spacr/ops_decode.py:220 _neighbour_discount_state [private/nested]
- spacr/ops_decode.py:261 decode_rule [canonical API]
- spacr/ops_decode.py:2621 _rule_with_neighbours [private/nested]
- spacr/ops_prefix.py:758 _minority_ignored [private/nested]
- spacr/ops_prefix.py:921 _barcode_groups [private/nested]
- spacr/ops_prefix.py:1234 prefix_cell_calls [canonical API]
- spacr/ops_prefix.py:1324 _prefix_cell_calls [private/nested]
- spacr/ops_prefix.py:2024 prefix_settings [canonical API]
- spacr/ops_prefix.py:2203 _check_weight_settings [private/nested]
- spacr/ops_prefix.py:2402 _weight_plan [private/nested]
- spacr/ops_prefix.py:2429 _setting_counts [private/nested]
- spacr/ops_prefix.py:2477 _weight_arguments [private/nested]
- spacr/ops_prefix.py:2557 _site_geometry [private/nested]
- spacr/settings.py:2698 _fold_older_defaults [private/nested]
- tests/test_ops_neighbour_discount_default_1013.py:1 <module> [test]
- tests/test_ops_neighbour_discount_default_1013.py:17 legacy [test]
- tests/test_ops_neighbour_discount_default_1013.py:31 _grouped_votes [test]
- tests/test_ops_neighbour_discount_default_1013.py:54 test_full_barcode_votes_equal_the_frozen_owner_for_all_four_discounts [test]
- tests/test_ops_neighbour_discount_default_1013.py:68 test_prefix_group_arrays_equal_the_frozen_owner_with_the_shared_flags [test]
- tests/test_ops_neighbour_discount_default_1013.py:83 test_shared_flags_preserve_ties_directed_edges_and_large_cell_identifiers [test]
- tests/test_ops_neighbour_discount_default_1013.py:98 test_both_adapters_call_the_same_shared_flag_owner [test]
- tests/test_ops_neighbour_discount_default_1013.py:103 test_both_adapters_call_the_same_shared_flag_owner.trace [test]
- tests/test_ops_neighbour_discount_default_1013.py:115 test_shared_flags_handle_empty_groups_and_missing_neighbours [test]
- tests/test_ops_neighbour_discount_default_1013.py:126 test_shared_flags_reject_inconsistent_group_or_pair_shapes [test]
- tests/test_ops_neighbour_discount_default_1013.py:132 test_new_default_preserves_every_old_settings_era_without_rewriting_files [test]
- tests/test_ops_neighbour_discount_default_1013.py:155 test_rules_off_skip_only_the_new_default_and_log_once [test]
- tests/test_ops_neighbour_discount_default_1013.py:172 test_positionless_cache_default_matches_discount_off_and_logs_once [test]
- tests/test_ops_neighbour_discount_default_1013.py:190 test_prefix_default_no_op_preserves_all_decode_files [test]
- tests/test_ops_neighbour_discount_default_1013.py:203 test_application_flags_are_published_to_the_existing_run_journal [test]
- tests/test_ops_neighbour_discount_default_1013.py:217 _recipe_bleed_reads [test]
- tests/test_ops_neighbour_discount_default_1013.py:225 _recipe_neighbours [test]
- tests/test_ops_neighbour_discount_default_1013.py:232 test_opt_in_recipe_bleed_uses_the_existing_minority_filter [test]
- tests/test_ops_neighbour_discount_default_1013.py:248 test_recipe_switch_off_and_discount_off_preserve_all_cell_values [test]
- tests/test_ops_neighbour_discount_default_1013.py:260 test_recipe_bleed_never_removes_a_tied_top_barcode [test]
- tests/test_ops_neighbour_discount_default_1013.py:272 test_recipe_call_counts_and_control_labels_use_the_same_selected_reads [test]
- tests/test_ops_neighbour_discount_default_1013.py:293 test_opt_in_recipe_bleed_requires_actual_neighbour_information [test]
- tests/test_ops_neighbour_discount_default_1013.py:300 test_prefix_run_exercises_recipe_flags_and_records_the_opt_in [test]
- tests/test_ops_neighbour_discount_default_1013.py:307 test_prefix_run_exercises_recipe_flags_and_records_the_opt_in.trace [test]
- tests/test_ops_neighbour_discount_default_1013.py:322 test_combinatorial_early_return_records_default_as_unapplied_once [test]
- tests/test_ops_neighbour_discount_default_1013.py:329 test_combinatorial_early_return_records_default_as_unapplied_once.decode [test]
- tests/test_ops_neighbour_discount_default_1013.py:348 test_runtime_extractor_admits_the_notice_and_alpha_control [test]
- tests/test_ops_neighbour_discount_default_1013.py:367 test_generic_resolver_keeps_unrelated_mappings_exact [test]
- tests/test_ops_neighbour_discount_default_1013.py:377 test_ops_resolver_shares_one_notice_across_well_copies [test]
- tests/test_ops_neighbour_equivalence_1013.py:1 <module> [test]
- tests/test_ops_neighbour_equivalence_1013.py:24 _exact [test]
- tests/test_ops_neighbour_equivalence_1013.py:45 _digest [test]
- tests/test_ops_neighbour_equivalence_1013.py:50 _arrays [test]
- tests/test_ops_neighbour_equivalence_1013.py:68 _owner [test]
- tests/test_ops_neighbour_equivalence_1013.py:74 _funk_equivalence [test]
- tests/test_ops_neighbour_equivalence_1013.py:101 _prefix_equivalence [test]
- tests/test_ops_neighbour_equivalence_1013.py:132 _write_receipt [test]
- tests/test_ops_neighbour_equivalence_1013.py:143 test_builtin_a1_core_equivalence [test]
- tests/test_ops_neighbour_equivalence_1013.py:173 test_external_funk_and_periscope_core_equivalence [test]
- tests/test_ops_neighbour_sources_1013.py:1 <module> [test]
- tests/test_ops_neighbour_sources_1013.py:22 _neighbour_api_sources [test]
- tests/test_ops_neighbour_sources_1013.py:51 _neighbour_predecessor_source [test]
- tests/test_ops_neighbour_sources_1013.py:73 _current_neighbour_api [test]
- tests/test_ops_neighbour_sources_1013.py:84 _neighbour_birth_arrivals [test]
- tests/test_ops_neighbour_sources_1013.py:93 _validated_neighbour_arrivals [test]
- tests/test_ops_neighbour_sources_1013.py:107 _historical_neighbour_arrivals [test]
- tests/test_ops_neighbour_sources_1013.py:116 test_neighbour_adapter_changes_only_bound_revisions_after_existing_adapter [test]
- tests/test_ops_neighbour_sources_1013.py:134 test_neighbour_adapter_refuses_unbound_bodies_and_birth_collisions [test]
- tests/test_ops_neighbour_sources_1013.py:147 test_new_callable_validator_retains_unrelated_entries_and_checks_parameters [test]
- tests/test_ops_neighbour_sources_1013.py:164 test_predecessor_bridge_rejects_current_source_and_signature_drift [test]
- tests/test_ops_read_weights_988.py:148 test_the_settings_are_alpha_off_and_checked [test]
- tests/test_ops_read_weights_988.py:245 test_the_decode_rule_carries_the_weighting_only_when_on [test]
- spacr/ops_sbs.py:1262 neighbour_bleed_flags: 'neighbour bleed needs one cell, barcode and read count per group' [API exception, runtime not admitted]
- spacr/ops_sbs.py:1270 neighbour_bleed_flags: 'neighbour bleed pairs must have shape pairs by two' [API exception, runtime not admitted]
- spacr/ops_decode.py:245 _neighbour_discount_state: 'rules off' [runtime tr]
- spacr/ops_decode.py:246 _neighbour_discount_state: 'cache replay without nucleus positions' [runtime tr]
- spacr/ops_decode.py:247 _neighbour_discount_state: 'ops_read_neighbour_discount ({discount}) does not act under {method}; reads are not discounted' [runtime tr]
- spacr/ops_prefix.py:779 _minority_ignored: 'ops_prefix_bleed_in_recipe needs touching-cell information' [API exception, runtime not admitted]
- spacr/ops_prefix.py:2582 _site_geometry: 'read weighting, recipe bleed and the multi-guide table need cell labels (ops_prefix_masks)' [API exception, runtime not admitted]
- spacr/ops_decode.py:246 _neighbour_discount_state: cache replay without nucleus positions [ui]
- spacr/ops_decode.py:247 _neighbour_discount_state: ops_read_neighbour_discount ({discount}) does not act under {method}; reads are not discounted [ui]
- spacr/ops_decode.py:245 _neighbour_discount_state: rules off [ui]
- spacr/ops_settings.py:556 ops_prefix_bleed_in_recipe: Prefix bleed in recipe [setting_labels]
- spacr/ops_settings.py:242 ops_read_neighbour_discount: Read neighbour discount [setting_labels]
- spacr/ops_settings.py:556 ops_prefix_bleed_in_recipe: Ops prefix bleed in recipe [setting_labels]
- spacr/ops_settings.py:242 ops_read_neighbour_discount: Ops read neighbour discount [setting_labels]
- spacr/ops_settings.py:3468 ops_prefix_bleed_in_recipe: Under ops_prefix_quality_method 'recipe_simple', ignore a minority barcode that has the most reads in a touching cell when ops_read_neighbour_discount is below 1. Tied top barcodes remain. This can rescue calls blocked by neighbour bleed. Default False. Alpha. [setting_tooltips]
- spacr/ops_settings.py:3459 ops_read_neighbour_discount: Weighs down a minority guide when the same guide has the most reads in a neighbouring cell, with ties kept. Neighbours are nuclei whose discs come within ops_footprint, or touching cells in the spacer-prefix decode. 1 is off and 0 ignores flagged reads. The spacer-prefix decode uses the discount with 'weighted_majority'; under 'recipe_simple', ops_prefix_bleed_in_recipe enables ignoring flagged minority barcodes. A notice reports paths where the default cannot act. Default 0.25. [setting_tooltips]

Equivalence evidence: 16 built-in A1 field/discount combinations plus 120 approved external field/site/discount combinations passed exact vote and call equality against frozen nightly flag owners.
External inputs are the supplied 988 replay caches, not the unavailable item-942 caches. Every package file retained its SHA-256 hash. No embedded Home source path was followed.
Funk uses the existing nucleus-disc neighbours from the supplied centroids; PERISCOPE uses the existing touching-cell owner and log candidates. PERISCOPE distance weighting is off in both implementations because nucleus masks/per-read nucleus gaps were not supplied; quality weighting is on. These comparisons establish shared-core equivalence under the stated setup, not a new paper acceptance result.
Counts describe cache replay. The four-field generation count and replay count are separate measurements; no image-pipeline identity claim is made.
Full group results and exact source fingerprints are recorded in reports/1013_neighbour_bleed/acceptance.json.

Home catalog resolution (ANSWER14:37): the existing arrival set names exactly ops_prefix_bleed_in_recipe until the train13 rebuild. The two runtime inventory checks pass after this approved addition. The unchanged stale sv discount-tooltip review is expected, cleared by the train 13 pin. Home removes the arrival key after rebuilding. Standing30deferrals are unchanged; no translations, catalogs or pins were edited.

Completed validation:

- final focused CUDA hidden: 188 passed in 28.97s
- final focused CUDA visible, declared 2 GB VRAM: 188 passed in 24.65s
- built-in A1 equivalence and combinatorial regression: 2 passed in 55.29s
- external 20-field and 10-site equivalence: 1 passed in 336.90s (0:05:36)
- full items: 1631 passed, 10 skipped, 153 warnings, 960 subtests passed in 467.33s (0:07:47)
- full OPS: 1516 passed, 31 skipped, 209 warnings, 102 subtests passed in 364.88s (0:06:04)
- full settings: 630 passed, 2 skipped, 369 subtests passed in 63.38s (0:01:03)
- full source ratchets before catalog arrival: 3 failed, 4090 passed, 76 skipped, 3 xfailed, 69 warnings in 1134.57s (0:18:54)
- authorized catalog arrival rerun; stale review expected: 1 failed, 2 passed in 43.12s

The source-continuity record verifies that the resolver guard added after the numerical run leaves every numerical owner unchanged. The final source file hashes and all initial attempts are recorded in the acceptance receipt.

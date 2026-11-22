# Item 1012fix English source handoff

Home owns the nine-language records, catalogs and pins. This branch changes no translation records or pins.

Complex recovery uses the existing Fisher owner with an enrichment tail and Bonferroni correction over all nonempty clusters. The default two-sided enrichment behavior is preserved. Set members are distinct scored genes.

One public API docstring is revised: spacr.ops_clusters.complex_metrics. The exact before/current bodies and hashes are in tr_1012fix_sources.json. The historical arrival remains unchanged. Claude approved the exact eight-line consumer adapter patch at 15:16 on 2026-11-22.

The plant survives into PCA; the matched graph loses it. Graph construction is unchanged. Primary acceptance FAILS: the no-plant p < 0.05 share is 187/7280 (2.56868%), below the required 3–7%. No tuning or arm rerun was performed. The specified mAP bootstrap and Spearman criteria pass; pooled mAP means are not strictly monotone.

No new runtime message, setting, default, schema or caption is introduced.

Changed English source locations:

- spacr/ops_clusters.py:208 _fisher_p [production]
- spacr/ops_clusters.py:1382 complex_metrics [production]
- tests/test_ops_leiden_sweep_916.py:161 test_complex_metrics_count_recovered_report_complexes [test]
- tests/test_ops_complex_recovery_chance_1012fix.py:1 <module> [test]
- tests/test_ops_complex_recovery_chance_1012fix.py:8 _recovery [test]
- tests/test_ops_complex_recovery_chance_1012fix.py:15 test_existing_fisher_tail_remains_two_sided_by_default [test]
- tests/test_ops_complex_recovery_chance_1012fix.py:22 test_two_members_recover_four_and_eight_member_sets_when_chance_is_small [test]
- tests/test_ops_complex_recovery_chance_1012fix.py:32 test_large_clusters_and_single_cluster_partitions_do_not_supply_recovery_evidence [test]
- tests/test_ops_complex_recovery_chance_1012fix.py:39 test_searching_more_clusters_changes_the_family_correction [test]
- tests/test_ops_complex_recovery_chance_1012fix.py:48 test_repeated_members_do_not_inflate_set_size_or_support [test]
- tests/test_ops_complex_recovery_chance_1012fix.py:55 test_shuffled_sets_have_recovery_near_the_nominal_five_percent [test]
- tests/test_ops_complex_recovery_sources_1012fix.py:1 <module> [test]
- tests/test_ops_complex_recovery_sources_1012fix.py:14 _complex_recovery_api_source [test]
- tests/test_ops_complex_recovery_sources_1012fix.py:31 _current_complex_recovery_api [test]
- tests/test_ops_complex_recovery_sources_1012fix.py:41 test_recovery_adapter_preserves_all_other_symbols_and_the_input [test]
- tests/test_ops_complex_recovery_sources_1012fix.py:51 test_recovery_adapter_rejects_an_unbound_third_body [test]
- tests/test_ops_complex_recovery_sources_1012fix.py:59 test_recovery_adapter_composes_with_the_external_projection_revision [test]

The shipped translatable_blocks owner supplied the exact public API blocks/layout in the paired JSON; the native source consumers all passed without importing production modules. No docs build ran.

# Item 1008 Part D translation handover

Branch: `ws-offload-codex-work-1008d`.

Parent: FINAL Part A `7b903471b71212ec8096e90b33864fa696d9aaaa` (including finalized B and C).

Canonical extraction found no new or changed public API source. Runtime source changes below are admitted by the actual clean source extractor. Home owns the nine-language records and catalog rebuild. The accompanying `tr_1008d_sources.json` carries exact source bodies, hashes and locations.

Locales: sv, de, es, zh_CN, pt, hi, ko, is, fr.

## Runtime sources

`categories/The image channels used for cell skeleton length and branch points. Select cytoskeleton in Feature families to add these measurements.`; SHA256 `24ce6526a1933240aeb81e668946c4532f56162b1fcdb88085372c27598f317e`.

`spacr/qt/screens/settings_model.py:3121`.

The image channels used for cell skeleton length and branch points. Select cytoskeleton in Feature families to add these measurements.

`setting_labels/cytoskeleton_channels`; SHA256 `fe5915df97195fd01a1ca905a0a626ab5ee2fa213dc0596c3892b1445d78ead6`.

`spacr/settings.py:2144`.

Cytoskeleton channels

`setting_labels/ops.ops_ph_cytoskeleton_channels`; SHA256 `3cba927bf68b474a9a0084eb0eeb92b6d007eced90925c9498f7384e3cdf9499`.

`spacr/ops_settings.py:163`.

Ph cytoskeleton channels

`setting_labels/ops_ph_cytoskeleton_channels`; SHA256 `f2b2fb761c980a4338ff0edb3262c72004a900f762cd2227863ace1f6224b4d4`.

`spacr/ops_settings.py:163`.

Ops ph cytoskeleton channels

`setting_tooltips/cytoskeleton_channels`; SHA256 `7f27ecbd8096f99447f6aa8575025ba52f3589f499f8d32fe0bf48abb177aa99`.

`spacr/settings.py:5881`.

Channels for the cell-only cytoskeleton block in profile_features. Use image channel indices or names from channel_names. None measures every image channel when cytoskeleton is selected; otherwise this setting has no effect. Default None.

`setting_tooltips/ops_ph_cytoskeleton_channels`; SHA256 `585b5402d503223d321db0bba404c6ede392a0ce7685723d65d4844d49d283cc`.

`spacr/ops_settings.py:1907`.

Image channel indices or names from ops_ph_channels for the cell-only cytoskeleton block in ops_ph_profile_features. None measures every channel when cytoskeleton is selected; otherwise this setting has no effect. Default None. Alpha.

`setting_tooltips/ops_ph_profile_features`; SHA256 `d1b9fe4863e293af74b92ae316d985a33309dddb93579911518fa4ea050f1a63`.

`spacr/ops_settings.py:1905`.

Add the same per-object measurements as Measure to the nucleus, cell and cytoplasm. 'profiling' selects fast profiling families; ops_ph_profile_optional_families adds slow families. 'all' includes every profiling family, including colocalisation_otsu, but excludes measurement blocks. Lists also accept coloc_block (Pearson, Manders and overlap), legacy_radial (six radial bins) and cytoskeleton (cell-only skeleton measurements using ops_ph_cytoskeleton_channels). Empty measures exactly as before. Default None. Alpha.

`setting_tooltips/profile_features`; SHA256 `b5f59ee410eb6f4f00a890128f478cfd4969d125bafb72a170000eb06a27220c`.

`spacr/settings.py:5880`.

Add per-object measurements to 2-D masks. 'profiling' selects the fast profiling families; profile_optional_families adds slow Gabor, colocalisation and granularity families. 'all' includes every profiling family, including colocalisation_otsu, but excludes measurement blocks. Lists or comma-separated names also accept coloc_block (Pearson, Manders and overlap), legacy_radial (six radial bins) and cytoskeleton (cell-only skeleton measurements using cytoskeleton_channels). Empty adds nothing. Default None.

`ui/Cytoskeleton measurements`; SHA256 `43219d83f3c66ef730ca36393ca8943adc983d008fef4669f89bcd1b333cfd8b`.

`spacr/ops_settings.py:1283`, `spacr/qt/screens/settings_model.py:2362`.

Cytoskeleton measurements

`ui/The image channels used for cell skeleton length and branch points. Select cytoskeleton in Feature families to add these measurements.`; SHA256 `24ce6526a1933240aeb81e668946c4532f56162b1fcdb88085372c27598f317e`.

`spacr/qt/screens/settings_model.py:3121`.

The image channels used for cell skeleton length and branch points. Select cytoskeleton in Feature families to add these measurements.

## Private and test documentation

These changed bodies need no public API records; exact bodies are in the companion JSON.

- `spacr/ops_engine.py:5322` `_ph_measure_field`: private/tool documentation; no changed canonical API.
- `spacr/settings.py:9929` `_alpha_names`: private/tool documentation; no changed canonical API.
- `tests/test_alpha_deck_registry.py:19` `test_the_published_alpha_page_covers_the_current_registry`: test documentation.
- `tests/test_measure_ops_features_934.py:391` `test_every_ops_measurement_setting_maps_to_a_measure_setting`: test documentation.
- `tests/test_measure_ops_features_934.py:519` `test_cytoskeleton_channel_selection_reaches_the_measurement_worker`: test documentation.
- `tests/test_measure_ops_features_934.py:542` `test_cytoskeleton_channels_survive_saved_cli_settings`: test documentation.
- `tests/test_measure_ops_features_934.py:554` `test_cytoskeleton_families_and_settings_are_scoped_to_item_1008`: test documentation.
- `tests/test_measure_ops_features_934.py:571` `test_historical_phenotype_parts_accept_default_channels_and_refuse_changed_selection`: test documentation.
- `tests/test_setting_api_links_point_at_the_consumer.py:109` `test_declared_ops_setting_is_found_before_english_catalog_rebuild`: test documentation.
- `tests/test_the_settings_flow_analyser.py:432` `test_ops_source_help_fills_cold_catalog_gaps_without_replacing_existing_text`: test documentation.
- `tools/build_setting_consumer_map.py:38` `setting_keys`: private/tool documentation; no changed canonical API.
- `tools/settings_flow.py:725` `_tooltips`: private/tool documentation; no changed canonical API.
- `tools/settings_flow.py:763` `_literal_dict`: private/tool documentation; no changed canonical API.

## Scope and validation

P10 exercises channel selections through the real phenotype worker and saved CLI settings for both modules, including None, index lists, name lists and a name string. Historical phenotype parts accept the new default None and refuse a different explicit selection. New setting rows remain alpha gated; typed family names are registered as families, keeping the existing text selectors and real dropdown registry distinct.

WORK_TASKS v86 approves the one-key Measure layout arrival, the dedicated OPS heading, the alpha deck arrival, the bounded cold-source analyser extensions and paired real metadata regeneration. The strict published alpha page awaits the master rebuild at landing. WORK_TASKS v87 explicitly accepts the inherited flow-page failure: untouched FINAL A has 13,621 links versus a 9,700 limit; D has 13,635. Item 1005 in train 8 owns the measured ceiling update. No ceiling was raised and no Sphinx build was run. The corrected live Qt gate, P10 and all six required code rules passed 211 checks in 132.08 seconds. Actual full-group results and any exact deferred failures appear in FINAL; a failure is never counted as passing.

## Final code checks and deferred records

Full items: 1,303 passed plus 959 subtests, three CUDA skips (214.54 s). Full OPS: 1,283 passed plus 102 subtests, 28 skips (312.67 s). The initial settings run had 613 passed and one missing two-claim arrival. WORK_TASKS v89 approved the exact named arrival without changing the 1005 census or any variants; the full rerun passed 614 plus 369 subtests, with two skips (65.73 s). The privately exercised proposal also passed against the actual defaults.

Raw ratchets: 4,069 passed, three failed, 76 skipped and three built-in expected failures (721.43 s), with exactly the 30 catalog-dependent nodes deferred under v74. The three actual failures remain failures: `tests/qt/test_external_i18n_catalogs.py::test_runtime_source_inventory_is_complete_before_optional_module_imports` and `::test_runtime_source_inventory_is_stable_after_runctx_import` see only the missing Home English catalog entries for `cytoskeleton_channels`, `ops_ph_cytoskeleton_channels` and inherited `cytoskeleton_object_support`; `::test_empty_flowchart_caption_translates_without_changing_python_none` sees stale Swedish reviewed records for `profile_features` and `ops_ph_profile_features`. Canonical source admission is separately verified; records and catalogs are Home work.

The strict published-alpha-page check is deferred by v86. The inherited settings-flow limit is deferred by v87 at 13,635 links; untouched FINAL A fails the same test at 13,621. No pin or ceiling was relaxed. All five metadata artifacts match the native final-source generation. The last consumer-map refresh changes only 41 source line numbers by two; its other JSON values and the remaining four artifacts are unchanged.

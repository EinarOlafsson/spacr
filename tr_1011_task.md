# Item1011 L2 source handover

Source branch: `ws-offload-codex-work-1011b`; base: `9b3dbe01976dcc21becabb0e51e2c24c24d60728`.

This lane wires the existing shared plan into Mask Generation and OPS. Parent combines this evidence with the L3 tooltip/label handover. Home owns locale records and catalog rebuilds for sv, de, es, zh_CN, pt, hi, ko, is and fr.

Canonical extraction used `canonical_sources()` and `public_docstrings()` without building catalogs. Exact sources, file hashes and private/test documentation inventory are in `tr_1011_sources.json` at the branch root (the private extraction copy is `tr_1011b_sources.json`).

New or changed admitted sources in the L2 delta against the stated base: 0 API blocks; 0 runtime literals.

## Source files

- `spacr/core.py`: SHA256 `47d9edb81da6bbad085860d5e38e14a8fdc897726284f743d05c440f640181bc`.
- `spacr/io.py`: SHA256 `6f8f32858e780c77ab8a4f3a266f5079e141bd2f8a924077eb44a54aade61833`.
- `spacr/object.py`: SHA256 `ccdf1c77f25a88a2526589192e7e9f62da6fb50cf683df969e98a9a33fcb1716`.
- `spacr/ops_engine.py`: SHA256 `ce28c614356b54cdedc0ff90a9291ced52105b883712495ee7b20eb986ce354a`.
- `tests/test_mask_batches_vram_auto_1011_wiring.py`: SHA256 `1862979973876fcf6945a0ff429122069843719e8fc1691ca34d5c639735b835`.

## Existing runtime source moved during input factoring

spacr/ops_engine.py:4670

Canonical ID: none; unchanged literal is not admitted by the existing runtime scan and needs no locale record.

SHA256: `15dfbc983a90d0f5f9d02f3f73f6e51a79df0ee9bf1b79920d4151508de6c7fe`; admitted: False. Existing translations remain the same.

ops_ph_channels {channels} must name {nuclear} and the cell channel {cell}

## Private, nested and test documentation

No records are needed for the following docstrings.

- `_prepare_mask_batch_plan` — `spacr/core.py:288`; no records: private or nested callable; SHA256 `9994ab8f5cd4904660bfe8f9bc0ae153f61da89c2ac0b2ed824aa3f8069866db`.
- `_concatenate_channel` — `spacr/io.py:1665`; no records: private or nested callable; SHA256 `be997b660db027c87967da5c768c8620c9c286a0fb924e1ccfc65ad6e462ae9c`.
- `_eval_tile_batch` — `spacr/object.py:1429`; no records: private or nested callable; SHA256 `1222b94536cb4d6bce0f2ad9ab89dbab72519512f31743d74b29d3aa5d7ad07c`.
- `_mask_model_session` — `spacr/object.py:1726`; no records: private or nested callable; SHA256 `7ad762584683d6957bda1d88e46d164d7deb9f73e87ad3d1863c61f21cc29a63`.
- `_session_cached_mask_model` — `spacr/object.py:1746`; no records: private or nested callable; SHA256 `054752fe529348dfecf6c08560be5262ad179f3ac1d5e2c47538c070b1f0ea3a`.
- `_release_calibration_model` — `spacr/object.py:1761`; no records: private or nested callable; SHA256 `5a87af8a0d03a7984b9ead19d90f4ce788a12579794e476ef7408aa59c34bf04`.
- `_mask_model_setup` — `spacr/object.py:1776`; no records: private or nested callable; SHA256 `61558907b09dc6d55877785d10ad4f471194545dc4e87576ac8630f71b9e3a5a`.
- `_mask_eval_options` — `spacr/object.py:1823`; no records: private or nested callable; SHA256 `3e1fe2e725f2e2d005442ceecdbeacf42d9c851e8d603735fd95046fb6605712`.
- `_organelle_channel_position` — `spacr/object.py:1835`; no records: private or nested callable; SHA256 `c395485dfe52b1612cee90245407b4a677002c1b92bd857803a50dcc817973cc`.
- `_organelle_model_setup` — `spacr/object.py:1845`; no records: private or nested callable; SHA256 `31af96be6e4ffe9c4a1709d986102f850256b461748cc3622e71e0d66d4c192c`.
- `_organelle_input_batch` — `spacr/object.py:1863`; no records: private or nested callable; SHA256 `885bb2094858ef1739c4b54c4d0c90be01bb56a4e1f6ee3ff00d562706a1011b`.
- `_organelle_eval_options` — `spacr/object.py:1874`; no records: private or nested callable; SHA256 `ed104a12d37d63558612b9979647c79c867ec5d0116b9fd50a5e26b7d55ff818`.
- `_mask_calibration_setup` — `spacr/object.py:1882`; no records: private or nested callable; SHA256 `0befa0f6da280e4c123a354d2b26814efcd6b411f4c14d7960e4c1ec105f01e3`.
- `_restore_mask_calibration` — `spacr/object.py:1911`; no records: private or nested callable; SHA256 `1ec1cced4d33c59dc76b6e15a6b2faa2d07ce8e0d1f64a908c24df16c9696990`.
- `_mask_probe_eval` — `spacr/object.py:1930`; no records: private or nested callable; SHA256 `1767f2d7561ecdc4e9730846b1473a197df6f25e7b8bada30320adeb50b991a1`.
- `_mask_batch_configuration` — `spacr/object.py:1936`; no records: private or nested callable; SHA256 `6c920fb6a78be5ed2e9dfb4e66ce5348598cb7eb521c37e9b43e267f8ef5021d`.
- `_validate_mask_batch_draw` — `spacr/object.py:1951`; no records: private or nested callable; SHA256 `906d671400ca570ae9958bb2e87f11e5f0b3d77c3e447fd5dd01eca6536cbc26`.
- `_record_mask_batch` — `spacr/object.py:1968`; no records: private or nested callable; SHA256 `2253a697c210389c5d5028f0020071cb824b818401dff9e9299d3dd1d0d6734d`.
- `_ops_calibration_context` — `spacr/ops_engine.py:2220`; no records: private or nested callable; SHA256 `9fcd83f42fb507aaf4a7157ac2cb481e00e4b0bfffe0291100280b1801a0e304`.
- `_ops_prefetched_load` — `spacr/ops_engine.py:2233`; no records: private or nested callable; SHA256 `a4b5bcd1be8496a14c5da94d6921702622733caf67c2a44d83cdb5a214ab93a4`.
- `_ops_journal_plan` — `spacr/ops_engine.py:2244`; no records: private or nested callable; SHA256 `e7fd5d5f09aafc3acf2776b26af5c3ff0ea75efb1b30907b073dc209f3cbde65`.
- `_ops_batch_fields` — `spacr/ops_engine.py:2268`; no records: private or nested callable; SHA256 `61c18ef45472a8ed6103162ed38f394271354286107134b8fc9342bbae792a90`.
- `_ops_array_batches` — `spacr/ops_engine.py:2306`; no records: private or nested callable; SHA256 `fe8f85c8d7c051f63e5cf4f26879202e21c4a6ec9b4821aaf764cdba3d4a1335`.
- `_ph_segmentation_inputs` — `spacr/ops_engine.py:4662`; no records: private or nested callable; SHA256 `0a87c1251d65361ff70ccd7b7379737842deec95633f85676533d74932f71aab`.
- `_ph_model_pair` — `spacr/ops_engine.py:5588`; no records: private or nested callable; SHA256 `d88270c112af37a3c437900f46ceafa827a8f07407a9e9707bd5532676fb7efc`.
- `_ph_probe_input` — `spacr/ops_engine.py:5599`; no records: private or nested callable; SHA256 `d3608963898353c9886698933e133f8900977067f0042ca89bb2f6794dadfd90`.
- `_mask_project` — `tests/test_mask_batches_vram_auto_1011_wiring.py:18`; no records: test callable; SHA256 `751ca768a07dc99c147568e7f7b93a393834f88ebb67b0dc7773fd210619bb19`.
- `_direct_eval` — `tests/test_mask_batches_vram_auto_1011_wiring.py:34`; no records: test callable; SHA256 `45f6568b976b1692137dcae09d50916c1625e4cede53fe65c0073769002796b7`.
- `_direct_eval.evaluate` — `tests/test_mask_batches_vram_auto_1011_wiring.py:38`; no records: test callable; SHA256 `fc65aad8a724c4cdec7de931c6aa0c356cd3f6747f700620bb25b4c232e3e78e`.
- `test_mask_calibrates_before_planning_with_resident_bytes_and_reuses_model` — `tests/test_mask_batches_vram_auto_1011_wiring.py:48`; no records: test callable; SHA256 `885f0f1b2d855faf39fa03f8735f832ceafa25e87bdddb1542fc6f021ebd1deb`.
- `test_mask_saved_gpu_plan_restores_calibration_without_new_probes` — `tests/test_mask_batches_vram_auto_1011_wiring.py:81`; no records: test callable; SHA256 `9752f368f9ea82535242dd28eae0188d636abe9805fabecd13a2a51655695d1c`.
- `test_model_session_restores_nested_and_exceptional_cache` — `tests/test_mask_batches_vram_auto_1011_wiring.py:99`; no records: test callable; SHA256 `f0e57cc62c4bade674aea18f4b7dddcc779eedda26b7342d220f8a37d9850cd9`.
- `test_coordinator_releases_calibrated_weights_before_subprocess_workers` — `tests/test_mask_batches_vram_auto_1011_wiring.py:111`; no records: test callable; SHA256 `27478e45bac4a95b6bb3d21f2d5c10c6d5c908c73f75f7785f885fe755a52f23`.
- `test_ops_adapter_even_groups_journal_and_resume_keep_saved_calibration` — `tests/test_mask_batches_vram_auto_1011_wiring.py:124`; no records: test callable; SHA256 `dd0527cc700b776b6f4719e86e0f11aea2223f912a4450954a632bf5c862106f`.
- `test_ops_adapter_even_groups_journal_and_resume_keep_saved_calibration.load` — `tests/test_mask_batches_vram_auto_1011_wiring.py:135`; no records: test callable; SHA256 `5470a71ce7101cbb53c85ac0e76e52558714639febb85c942a40e63df0e636f9`.
- `test_ops_adapter_even_groups_journal_and_resume_keep_saved_calibration.prepare` — `tests/test_mask_batches_vram_auto_1011_wiring.py:140`; no records: test callable; SHA256 `8fac4580195abc7af9262347144f872705b61dd64a15ea83ca14660625112f6f`.
- `test_mask_batch_journal_refuses_a_changed_draw` — `tests/test_mask_batches_vram_auto_1011_wiring.py:173`; no records: test callable; SHA256 `1f133990cc28b1929d9ade91f34796af3de6e739218c9a0887ce9bb87c3f1e6c`.
- `test_legacy_ops_selectors_keep_original_groups_without_calibration` — `tests/test_mask_batches_vram_auto_1011_wiring.py:190`; no records: test callable; SHA256 `5320319371f171e40dfd3932988a07407a25b87799d483acbc9350f3c70b5b94`.
- `test_real_phenotype_stager_matches_seeded_segmentation_order` — `tests/test_mask_batches_vram_auto_1011_wiring.py:204`; no records: test callable; SHA256 `5d408cecac26d3588995ae1beff1b273f43a284278ad225800aec214b1f11fa8`.
- `test_real_phenotype_stager_matches_seeded_segmentation_order.Stager` — `tests/test_mask_batches_vram_auto_1011_wiring.py:218`; no records: test callable; SHA256 `2fdb129350fd80be051c5abb0e2b5c2b5b59e404aa4cf56b0153f90b1681251d`.
- `test_real_phenotype_stager_matches_seeded_segmentation_order.Stager.__init__` — `tests/test_mask_batches_vram_auto_1011_wiring.py:224`; no records: test callable; SHA256 `d2a3699104f6484705095e8c76af0c6e697a9d6e7c990ba5304d5f7cf6db45a9`.
- `test_real_phenotype_stager_matches_seeded_segmentation_order.Stager.load` — `tests/test_mask_batches_vram_auto_1011_wiring.py:228`; no records: test callable; SHA256 `ad265fdea752dcdf8b3d78c7de32005c021075b1bfff9225857f062fe0d7ccb1`.
- `test_real_phenotype_stager_matches_seeded_segmentation_order.Stager.release` — `tests/test_mask_batches_vram_auto_1011_wiring.py:233`; no records: test callable; SHA256 `cbcbaf3cae350560907c30334c111c7794a8b88cc289092d038626f7463210a9`.
- `test_real_phenotype_stager_matches_seeded_segmentation_order.Stager.close` — `tests/test_mask_batches_vram_auto_1011_wiring.py:236`; no records: test callable; SHA256 `65cec6b71d3e8bdcef252b3add0886c6d1c90623c55193e316002431d563f8c0`.
- `test_cpu_mask_plan_never_evaluates_a_probe_and_caps_at_fifty` — `tests/test_mask_batches_vram_auto_1011_wiring.py:261`; no records: test callable; SHA256 `5269a60444c07bb658ffe1b6c28d1bc5f6138f7da99ba4d5fb01f2415eaf94fd`.
- `test_concurrent_mask_runs_reuse_models_only_within_their_own_context` — `tests/test_mask_batches_vram_auto_1011_wiring.py:275`; no records: test callable; SHA256 `8902a0b304018c26a96b15eaee4937837bd3c7eb5c16047fd5b6c2b005b634fa`.
- `test_concurrent_mask_runs_reuse_models_only_within_their_own_context.run` — `tests/test_mask_batches_vram_auto_1011_wiring.py:283`; no records: test callable; SHA256 `069661c8b0a0ac2451b27eb77c3f6127c3c0f1b3e6a65214f77dfd374fc49d8f`.
- `test_completed_ops_run_refuses_a_changed_draw_before_loading_or_probing` — `tests/test_mask_batches_vram_auto_1011_wiring.py:302`; no records: test callable; SHA256 `72a1daa16689631041d2867d32afeea16e26978d56a0a3876c92c1cb147e987b`.
- `test_completed_mask_run_refuses_changed_seed_before_loading_or_probing` — `tests/test_mask_batches_vram_auto_1011_wiring.py:328`; no records: test callable; SHA256 `7741690a3df2493026c9cc88ebd9555d901537f0e7cbb167086095a85a47b400`.
- `test_stitched_windows_keep_acquisition_order_with_randomize_enabled` — `tests/test_mask_batches_vram_auto_1011_wiring.py:344`; no records: test callable; SHA256 `55a71637fdc70c955454531eec058965c27001f6f71db998126086777d6de81b`.
- `test_stitched_windows_keep_acquisition_order_with_randomize_enabled.batches` — `tests/test_mask_batches_vram_auto_1011_wiring.py:359`; no records: test callable; SHA256 `bc135cda0b3b7e206e7f06f42d906d2695f2d9c64ea663196b13e90c1aa6dd4f`.
- `test_real_spatial_windows_keep_acquisition_order_with_randomize_enabled` — `tests/test_mask_batches_vram_auto_1011_wiring.py:373`; no records: test callable; SHA256 `0d5a3f258e36859a17f32f228f0a082a7a50e05faf59f8e9f2cc741eaba2f586`.
- `test_real_spatial_windows_keep_acquisition_order_with_randomize_enabled.batches` — `tests/test_mask_batches_vram_auto_1011_wiring.py:386`; no records: test callable; SHA256 `f8dad581aa00ba018dfd31c7cc9407e9681ced7aaf4b7176c7780f4a00523083`.
- `test_sequential_ops_sites_use_their_own_resident_calibration_and_resume` — `tests/test_mask_batches_vram_auto_1011_wiring.py:397`; no records: test callable; SHA256 `083dd219c2d3635a23c775c17ec5cc1ed1146ee0cb3ef396aecd488c8d11144c`.
- `test_sequential_ops_sites_use_their_own_resident_calibration_and_resume.prepare` — `tests/test_mask_batches_vram_auto_1011_wiring.py:410`; no records: test callable; SHA256 `95df4958d1f0877d26db347ba4124114ab9ee2fc2356be86709ce790e210bfb0`.
- `test_ops_journal_replay_without_external_plan_restores_matching_site` — `tests/test_mask_batches_vram_auto_1011_wiring.py:441`; no records: test callable; SHA256 `9dc68a36f4c17e76b49e9136babe46356aa110f8bce95c8e124b28443ca142ab`.
- `test_ops_journal_records_the_actual_explicit_site_tile_override` — `tests/test_mask_batches_vram_auto_1011_wiring.py:465`; no records: test callable; SHA256 `48f258d7d3a7a0514b60630fbbb81bd1adfd518ce5f711ee4e21cfb55a080c7c`.
- `test_real_mask_io_uses_repeatable_seeded_order_and_acquisition_control` — `tests/test_mask_batches_vram_auto_1011_wiring.py:479`; no records: test callable; SHA256 `d4ea74df907fc66d49d0c45672f08b9fb9ecb92c362be6e4c665d3bed03b8f03`.
- `test_calibration_probes_keep_exact_tiles_when_runtime_controller_would_shrink` — `tests/test_mask_batches_vram_auto_1011_wiring.py:493`; no records: test callable; SHA256 `566cbdffbef6bb3d6dfc25b1de8bbff8e77362b0bdbb5d80a09a71436c351339`.
- `test_calibration_probes_keep_exact_tiles_when_runtime_controller_would_shrink.fit` — `tests/test_mask_batches_vram_auto_1011_wiring.py:499`; no records: test callable; SHA256 `ee129288a69ce4934f2186f21e4ecca2165ab246903c7fff5753922e9c2e5eca`.
- `test_second_probe_oom_reaches_the_existing_calibrator_without_smaller_retry` — `tests/test_mask_batches_vram_auto_1011_wiring.py:521`; no records: test callable; SHA256 `8f3e742fd04fd304707fba07236ec8973ed742cd2a14805e7d0f28b222202430`.
- `test_second_probe_oom_reaches_the_existing_calibrator_without_smaller_retry.evaluate` — `tests/test_mask_batches_vram_auto_1011_wiring.py:531`; no records: test callable; SHA256 `77ffd9596e0bf0e7ce72ec3347fe790a96f75b8849f5fc8868ec71091de89087`.
- `test_journal_only_multi_group_membership_and_replay_ignore_changed_ram` — `tests/test_mask_batches_vram_auto_1011_wiring.py:556`; no records: test callable; SHA256 `8629b7f8df212162c7ad7f7bed69cf725b0d4e0a9ee84fb9b70bc906ce934d49`.
- `test_temporary_shared_plan_cleans_up_empty_and_missing_fields` — `tests/test_mask_batches_vram_auto_1011_wiring.py:582`; no records: test callable; SHA256 `b9899112a5cc489f45b6b8a97e18d49462bd7d3a2a9bf8b69cf8763986a45a10`.
- `test_temporary_shared_plan_cleans_up_empty_and_missing_fields.temporary` — `tests/test_mask_batches_vram_auto_1011_wiring.py:590`; no records: test callable; SHA256 `0da60c02b2949989e782866164d88b179614a57b0c7110fb4caf6f33f2b96060`.
- `test_temporary_shared_plan_is_removed_on_a_segmentation_exception` — `tests/test_mask_batches_vram_auto_1011_wiring.py:614`; no records: test callable; SHA256 `27d787ed3e2d76e83c6927e1640076a6773b618a401e21b5e61adb2b188f7fb6`.
- `test_temporary_shared_plan_is_removed_on_a_segmentation_exception.temporary` — `tests/test_mask_batches_vram_auto_1011_wiring.py:622`; no records: test callable; SHA256 `18a1cb7e9c1f757f95d80dca9f70185f5ab7d5eff604e4b4f7dfd177d14178c4`.
- `test_classical_only_mask_caller_never_constructs_or_probes_cellpose` — `tests/test_mask_batches_vram_auto_1011_wiring.py:642`; no records: test callable; SHA256 `3dc7dec768cbb31e3d04271b0bb7206c9b1c982618b27815f4472d2a7e5949ed`.
- `test_organelle_cellpose_uses_real_preparation_options_and_the_same_loaded_model` — `tests/test_mask_batches_vram_auto_1011_wiring.py:664`; no records: test callable; SHA256 `88cf2bd685c222e6b5f98ea7c1a4a788b5a6e42b8d755ab60f04f0360b899a5a`.
- `test_organelle_cellpose_uses_real_preparation_options_and_the_same_loaded_model.choose` — `tests/test_mask_batches_vram_auto_1011_wiring.py:679`; no records: test callable; SHA256 `837abb7e2bf7e9bffb5deed0d5cb294aafd34902103a0c1f7fdce07e65f65977`.
- `test_organelle_cellpose_uses_real_preparation_options_and_the_same_loaded_model.preprocess` — `tests/test_mask_batches_vram_auto_1011_wiring.py:684`; no records: test callable; SHA256 `0836906204423b93f5d3778282a9e64ccc0c59f7abdd363d6048a40b544e2c2a`.
- `test_unet_only_mask_plan_skips_tile_probes_and_loads_only_at_actual_segmentation` — `tests/test_mask_batches_vram_auto_1011_wiring.py:713`; no records: test callable; SHA256 `4858169de307370ab2d5855cfd448a6e7f3c84f1e8928c4b14af9bd42ee8a7ce`.
- `test_unet_only_mask_plan_skips_tile_probes_and_loads_only_at_actual_segmentation.load` — `tests/test_mask_batches_vram_auto_1011_wiring.py:725`; no records: test callable; SHA256 `c7052038bc49848baa637513a80a45d19dd94a09429b92bda78f01fc3cf2e23b`.
- `test_unet_only_mask_plan_skips_tile_probes_and_loads_only_at_actual_segmentation.segment` — `tests/test_mask_batches_vram_auto_1011_wiring.py:730`; no records: test callable; SHA256 `6f425ac78662f1261cdf9a2a71a58f66f6d8b6d5fd078555b393b0725ed70901`.
- `test_real_phenotype_preserves_legacy_limit_before_model_residency` — `tests/test_mask_batches_vram_auto_1011_wiring.py:750`; no records: test callable; SHA256 `560d2a2a9dd1ab5ba69385e69175d2c8be80d042b5eacae563d03ff1c3b62884`.
- `test_real_phenotype_preserves_legacy_limit_before_model_residency.load_model` — `tests/test_mask_batches_vram_auto_1011_wiring.py:766`; no records: test callable; SHA256 `3e714f666d67f0ad116840dccca580ddab544afd63d4034202fe3e8643c77690`.
- `test_real_phenotype_preserves_legacy_limit_before_model_residency.limit` — `tests/test_mask_batches_vram_auto_1011_wiring.py:772`; no records: test callable; SHA256 `b01320925d5ed26f5c4629fcfe4be10db7f4f018a086f96952daec06383b41a2`.
- `test_legacy_mask_session_keeps_original_uncached_model_construction` — `tests/test_mask_batches_vram_auto_1011_wiring.py:805`; no records: test callable; SHA256 `40342559b513b6451e1ff088b7a8a45faa9bd7c46332fe5ac0748a88e36a95f0`.
- `test_mask_tile_auto_session_reuses_model_when_field_count_is_explicit` — `tests/test_mask_batches_vram_auto_1011_wiring.py:822`; no records: test callable; SHA256 `6ebed37d10bf1765219c8937fada1f6afb2e6778dc2dadf659f65c10ce923f86`.
- `test_published_mask_settings_replay_preserves_selectors_and_resolved_groups` — `tests/test_mask_batches_vram_auto_1011_wiring.py:836`; no records: test callable; SHA256 `00dcb4380e3a99ecc526ce7f3c68be6865c82572d9fb9d52b734b3a0d9fe9800`.
- `test_second_session_does_not_capture_an_in_progress_constructor_cache` — `tests/test_mask_batches_vram_auto_1011_wiring.py:887`; no records: test callable; SHA256 `e9bd562cab394d083b07c6d1135af8dec8041bbe79fa3d1a16d6d76289620e7e`.
- `test_second_session_does_not_capture_an_in_progress_constructor_cache.build` — `tests/test_mask_batches_vram_auto_1011_wiring.py:896`; no records: test callable; SHA256 `0f1e0fc18311fed684b944a7b48c04f36bbcc3eb8589cd58e754331578a3d851`.
- `test_second_session_does_not_capture_an_in_progress_constructor_cache.first` — `tests/test_mask_batches_vram_auto_1011_wiring.py:902`; no records: test callable; SHA256 `bbf80a1f214a28008799d26e9cb8e3c7d052ac8787edaa178df6cf24a9f5ea6a`.
- `test_second_session_does_not_capture_an_in_progress_constructor_cache.second` — `tests/test_mask_batches_vram_auto_1011_wiring.py:908`; no records: test callable; SHA256 `adceefee342c9ea6d3bf1dd8dfd4c6e4d9440c64f8a9d342bb59abcd295492e8`.
- `test_legacy_coordinator_release_does_not_clear_another_run_cache` — `tests/test_mask_batches_vram_auto_1011_wiring.py:928`; no records: test callable; SHA256 `1fd474a7d4110a92a11a6bb6b5e4e09584c9333bfb87a7b8fca3162e698b1c24`.
- `test_legacy_coordinator_release_does_not_clear_another_run_cache.held_handoff` — `tests/test_mask_batches_vram_auto_1011_wiring.py:940`; no records: test callable; SHA256 `bc81a381c5ebed2fd07e4fd928bb021238554cdc733c42933eac25c6adfa3f38`.
- `test_legacy_coordinator_release_does_not_clear_another_run_cache.ordinary_auto` — `tests/test_mask_batches_vram_auto_1011_wiring.py:947`; no records: test callable; SHA256 `691b1f08d8707b378a6e3a0e04b250ca90c45bf36248813ef35ffafccc1c7b5d`.
- `test_legacy_coordinator_release_does_not_clear_another_run_cache.legacy_parallel` — `tests/test_mask_batches_vram_auto_1011_wiring.py:955`; no records: test callable; SHA256 `87cfa34c3d71e9b455967980b57589f02460862be9100a39f38d97ee0567ee05`.

## Inherited L1 console source outside the L2 delta

`spacr/_mask_batches.py:607–627`, `_calibration_line`, called at line 814, constructs the Mask batches / no CUDA device / VRAM budget / measured console text as plain f-strings. It is **runtime-not-admitted**: no `tr()` wrapper or canonical ID. This is an inherited L1 owner follow-up, not merely missing Home locale records. L2 does not edit this owner file. Exact raw function source and SHA256 are in `tr_1011_sources.json` under `inherited_l1_runtime_not_admitted`.

Raw function source SHA256: `d0686827af079bb1b0d7a00d9769f5d9245c49e2fc13778eb156206d7c7f71a9`.

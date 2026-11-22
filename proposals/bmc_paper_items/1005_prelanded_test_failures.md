# 1005: test failures that predate train 6

Five tests fail on nightly 04ec46d370 and on train 6b 20e35b6846. Each was
run serially (`-p no:xdist`, `CUDA_VISIBLE_DEVICES=''`,
`QT_QPA_PLATFORM=offscreen`) through the WS scheduler. Branch
`ws-offload-ws-1005`, from 20e35b6846. Decisions were taken without the
maintainer (asleep), as the recommended option, and are recorded here.

## 1. tests/qt/test_measure_settings_regrouped.py::test_the_top_level_reads_in_run_order

- Failure: `At index 4 diff: 'Vacuoles α' != 'Lineage Trees α'`. Measure
  had an extra top-level heading.
- Breaking commit: 60c70e6145 (items 898/910/911, 2026-11-10). It added the
  `Vacuoles α` category between Features and Confluency α in
  `_APP_CATEGORIES["measure"]` but did not add it to Measure's parent map.
- Verdict: the code is wrong. Item 595 puts the assays under Features, and
  60c70e6145 put Vacuoles α among them.
- Fix (a6b44acfeb): `"Vacuoles α"` added to the Features group of
  `_APP_CATEGORY_PARENTS["measure"]` in `spacr/qt/screens/settings_model.py`.
  `test_features_keeps_its_rows_and_nests_the_assays` now lists it first.
- Proof: the whole of `tests/qt/test_measure_settings_regrouped.py` passes
  (6 tests). The flat order in `tests/test_settings_categories.py` is
  unchanged and passes in the settings group.

## 2. tests/test_the_settings_flow_analyser.py::test_the_page_stays_inside_the_docs_job_timeout

- Failure: 13,578 cross-references against a ceiling of 9,700.
- Breaking commit: the analyser (`tools/settings_flow.py`) has not changed
  since eb6473bcf5. Counts from archived trees of path-scoped commits:
  - f9f98663b4 had 9,417 and 5256172175 had 9,461.
  - Item 971, a47d67189f, took it to 10,947. `spacr.ops_live` passes the OPS
    settings to `spacr.ops_engine.run_ops`, so every OPS setting section
    links it.
  - Item 973, 6cfa1cf89d, added another 661 on its own branch.
  - Together they reach 11,914 at 209e564475, which reached nightly in
    a7f4863f72.
  - The later OPS readers (with `spacr.ops_prefix.prefix_settings` linked
    760 times) took it to 13,578.

  This is real settings flow. It is not a duplicate-link bug, so trimming
  the page was not needed.
- Verdict: the ratchet number was stale.
- Measurement: a whole-site build of da3fb0940a, which is 20e35b6846 plus
  the 1005 test fixes, with the same 13,578 count.
  - Run after the docs job's Make Masks reference step and both
    `--sources-only` i18n steps.
  - Command: `sphinx-build -W -E -b html --keep-going`.
  - Result: 17:50.71 wall clock, 6,628,416 KB peak RSS, 0 warnings, build
    succeeded, at a load average of 33 to 41.
  - Logs are in `/media/carruthers/mnt2/spacr/1005_logs/docsbuild/`.
- Fix (76c4d9d7bd): the ceiling is raised to 13,900 (322 of headroom), with
  the measurement written into the docstring.
- Limits of the measurement:
  - It used `SPACR_DOCS_API_LANGUAGE=english`. The docs job uses `all`,
    which needs the assembled API translation memory, and that was not
    fetched.
  - No baseline build of the 9,417 page was made at the same load.
  - Proof: `tests/test_the_settings_flow_analyser.py` passes (38 tests).

## 3. tests/test_n702_pipeline_settings_pin_guard.py::test_legacy_settings_without_model_metadata_never_call_pin_validator

- Failure: the saved JSON has one more item, `{'spacr_settings_version': 948}`.
- Breaking commit: 9902032562 (item 931, 2026-11-13). `save_settings` now
  stamps every saved settings file with the schema version.
- Verdict: the test is wrong (stale).
- Fix (1c428a4051): the test now expects exactly
  `{SETTINGS_VERSION_KEY: SETTINGS_SCHEMA_VERSION, **original}`. It still
  proves that the pin validator is never called and that the CSV has no
  `model_metadata` row.
- Proof: the whole file passes.

## 4. tests/test_an_old_settings_file_keeps_its_values_through_spacr_run.py::test_a_dry_run_report_names_what_moved

- Failure: the dry run exits 2 with `paired_data: Regression needs at least
  one score table and one count table.`
- Breaking commit: 52bb83cf69 (issue 193, 2026-11-18). A regression
  pre-flight now refuses a run that has no score or count table.
- Verdict: the test is wrong. Its file held only `src` and `min_cell_count`.
- Fix (da3fb0940a): the test writes a one-row score CSV (with `pred`) and a
  count CSV, and names them as `score_data` and `count_data` lists (the
  declared type). The rename assertion is unchanged.
- Proof: the whole file passes (15 tests), and so does
  `tests/test_issue_193_regression_inputs.py`.

## 5. tests/test_cell_cycle_phase_three_ways.py::test_the_real_plate1_nuclei_fit_a_g1_and_a_g2_peak

This test was added to 1005 by master.

- Failure: on nightly 04ec46d370 (run read-only in its worktree) and on
  20e35b6846, the test raises `ValueError: The nucleus table has no
  nucleus_channel_0_integrated_intensity column`.
- Cause: missing data. The WS cache
  `~/.cache/spacr/example_data/plate1/measurements/measurements.db` was
  measured on 2026-09-25, a day before the test was added (d1be6c1d92,
  item 535). Its nucleus table has `nucleus_channel_0_mean_intensity`
  but no integrated intensity. The test skipped only when no database
  existed at all.
- Verdict: the test is wrong, because it does not check the data it needs.
  The code is correct.
- Fix (c2310c8481): the test reads the nucleus columns first. If the
  column is missing, it skips with a reason that names the database path
  and the column. The "not downloaded" skip now names the path too.
- Proof: the whole file gives 30 passed and 1 skipped, with the reason
  naming the path.

## Gate

Final gate (`gate.sh ratchets settings items`, one sched job, 4 threads,
16 GB), run on c2310c8481. The settings-flow test is not in these groups.

| group | result |
|---|---|
| ratchets | 4094 passed, 76 skipped, 11 xfailed |
| settings | 605 passed, 2 skipped |
| items | 1 failed, 1243 passed, 3 skipped |

The one red in items is the known unrelated
`tests/qt/test_ops_settings_layout_969.py::test_plate_layout_and_controls_sit_in_ops_input`
(the train 6 'Advanced' category, fixed in train 6c). It was left as it is.

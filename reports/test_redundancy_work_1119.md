# Test redundancy review — 19 November 2026

Reviewed `origin/nightly` at `4ef66cb340`. This report proposes changes; no test or application code was changed.

The strongest removals are two duplicated files: **14 collected test cases and 393 source lines** can be removed while retaining their checks in existing files. A third recommendation removes one repeated capability check after moving its restoration assertion into the test that changes the capability. The other recommendations consolidate test definitions while retaining every scenario; they do not claim a runtime reduction.

The scan examined 5,373 eligible test modules and 60,299 test functions. It excluded issue-specific tests, files identifying themselves as ratchets, and the named policy guards. AST comparison ignored function names, docstrings and source positions, while retaining fixtures, assertions and decorators. It found 26 groups with identical function bodies and 93 groups within a file that differed only in literal values. These are candidate counts, not counts of removable tests. The recommendations below were checked against their surrounding fixtures and helpers. Collection of the seven modules used for the removal and subprocess recommendations succeeded with 200 cases; the full suite was not run and runtime savings were not measured.

## 1. Remove the duplicate graph-save module

`tests/test_a_graph_save_leaves_the_field_its_rate.py` and `tests/qt/test_a_graph_save_leaves_the_field_its_rate.py` are byte-for-byte identical, including their fixtures, constants and Qt marker. Keep the copy under `tests/qt/`, which receives the Qt directory fixtures and cleanup. Remove the root-level copy after checking that CI invokes the retained path. Preserve all timing checks in the retained file.

| Remove | Retain |
| --- | --- |
| `tests/test_a_graph_save_leaves_the_field_its_rate.py::test_the_field_keeps_its_rate_while_the_gate_graph_is_saved` | `tests/qt/test_a_graph_save_leaves_the_field_its_rate.py::test_the_field_keeps_its_rate_while_the_gate_graph_is_saved` |
| `tests/test_a_graph_save_leaves_the_field_its_rate.py::test_a_paced_field_whose_frame_lands_after_its_tick_keeps_its_rate` | `tests/qt/test_a_graph_save_leaves_the_field_its_rate.py::test_a_paced_field_whose_frame_lands_after_its_tick_keeps_its_rate` |
| `tests/test_a_graph_save_leaves_the_field_its_rate.py::test_saves_share_one_warm_process_no_slower_than_drawing_here` | `tests/qt/test_a_graph_save_leaves_the_field_its_rate.py::test_saves_share_one_warm_process_no_slower_than_drawing_here` |
| `tests/test_a_graph_save_leaves_the_field_its_rate.py::test_a_pickled_snapshot_writes_the_files_a_copied_figure_wrote` | `tests/qt/test_a_graph_save_leaves_the_field_its_rate.py::test_a_pickled_snapshot_writes_the_files_a_copied_figure_wrote` |
| `tests/test_a_graph_save_leaves_the_field_its_rate.py::test_save_graph_unpickles_nothing_on_the_gui_thread` | `tests/qt/test_a_graph_save_leaves_the_field_its_rate.py::test_save_graph_unpickles_nothing_on_the_gui_thread` |

**Saving:** 5 collected cases, 5 function definitions and one 289-line file. This removes a second execution of the rendering and pacing scenarios. Wall-clock savings depend on the CI lane and renderer startup.

## 2. Remove the superseded measurement-name module

The helpers and five test bodies in `tests/test_final.py` are duplicated in `tests/test_measure_crop_names_and_defaults.py`. The two remaining scenarios are already the two explicitly named rows of the replacement parameterized test. Both files use the same parent fixtures. Keep the descriptive module, including its two additional error-message tests, and remove `tests/test_final.py`.

| Remove | Retain |
| --- | --- |
| `tests/test_final.py::test_arrayscan_names_write_and_read_without_a_time_component` | `tests/test_measure_crop_names_and_defaults.py::test_arrayscan_names_write_and_read_without_a_time_component` |
| `tests/test_final.py::test_parser_accepts_new_and_existing_none_names[plate01_A01_3_None.npy]` | `tests/test_measure_crop_names_and_defaults.py::test_parser_accepts_new_and_existing_none_names[plate01_A01_3_None.npy]` |
| `tests/test_final.py::test_parser_accepts_new_and_existing_none_names[plate01_A01_3.npy]` | `tests/test_measure_crop_names_and_defaults.py::test_parser_accepts_new_and_existing_none_names[plate01_A01_3.npy]` |
| `tests/test_final.py::test_stem_writer_keeps_the_empty_and_timed_forms` | `tests/test_measure_crop_names_and_defaults.py::test_stem_writer_keeps_the_empty_and_timed_forms` |
| `tests/test_final.py::test_unparsable_field_name_fails_once_and_names_the_file` | `tests/test_measure_crop_names_and_defaults.py::test_unparsable_field_name_fails_once_and_names_the_file` |
| `tests/test_final.py::test_measure_crop_fills_its_own_defaults` | `tests/test_measure_crop_names_and_defaults.py::test_measure_crop_measures_a_stack_with_or_without_the_none_name[fills-its-own-defaults]` |
| `tests/test_final.py::test_old_none_named_stack_still_measures` | `tests/test_measure_crop_names_and_defaults.py::test_measure_crop_measures_a_stack_with_or_without_the_none_name[old-none-named-stack]` |
| `tests/test_final.py::test_merged_src_is_read_as_its_plate_folder[/merged]` | `tests/test_measure_crop_names_and_defaults.py::test_merged_src_is_read_as_its_plate_folder[/merged]` |
| `tests/test_final.py::test_merged_src_is_read_as_its_plate_folder[/merged/]` | `tests/test_measure_crop_names_and_defaults.py::test_merged_src_is_read_as_its_plate_folder[/merged/]` |

**Saving:** 9 collected cases, 7 function definitions and one 104-line file. Five removed cases invoke the measurement pipeline, including the invalid-name case. The replacement retains the same inputs and assertions.

## 3. Merge the capability-restoration check into the mutation test

These two tests have identical executable bodies and use the same autouse fixture:

- `tests/test_cov_r8_measure_probes.py::TestTheExpandLabelsProbe::test_the_probe_matches_the_installed_signature`
- `tests/test_cov_r8_measure_probes.py::TestTheExpandLabelsProbe::test_the_probe_is_restored_afterwards`

Keep the first as the normal-install check. Move the restoration assertion into `tests/test_cov_r8_measure_probes.py::TestTheExpandLabelsProbe::test_a_signature_that_cannot_be_read_answers_no`, immediately after its `monkeypatch.undo()` and `_reimport_measure()` cleanup, then remove the standalone `test_the_probe_is_restored_afterwards`.

The current standalone test does not establish that it ran after the mutation test, especially under distributed or randomized execution. An assertion in the mutation test checks the intended restoration directly.

**Saving:** 1 collected case and 1 function definition; 3 cases become 2. Retain the module-state cleanup fixture. Net source-line savings are small because the restoration assertion moves rather than disappears.

## 4. Parameterize the common table-screen contracts

The following failures are checked with identical bodies, but the `screen` fixtures construct three different classes. They are distinct coverage and must remain three cases.

- `tests/qt/test_cov_w2_1_feature_explorer_screen.py::test_a_failed_read_is_reported_on_the_source_line`
- `tests/qt/test_cov_w2_1_graph_builder_screen.py::test_a_failed_read_is_reported_on_the_source_line`
- `tests/qt/test_cov_w2_1_tabulate.py::test_a_failed_read_is_reported_on_the_source_line`

Use one parameterized contract over `FeatureExplorerScreen`, `GraphBuilderScreen` and `TabulateScreen`, retaining a private `LinkedSelection`, `threaded=False` and Qt cleanup for each case. Apply the same factory to the second repeated contract:

- `tests/qt/test_cov_w2_1_feature_explorer_screen.py::test_picking_a_table_with_no_file_loaded_does_nothing`
- `tests/qt/test_cov_w2_1_graph_builder_screen.py::test_picking_a_table_with_no_file_loaded_does_nothing`
- `tests/qt/test_cov_w2_1_tabulate.py::test_picking_a_table_with_no_file_loaded_does_nothing`

**Saving:** 6 definitions become 2; all 6 collected cases remain. Runtime savings are not expected. Keep explicit case IDs for the three screen classes and retain the existing screen-specific load tests.

## 5. Parameterize the CLI lightweight-command checks

- `tests/test_cli.py::test_help_pulls_no_gui_or_torch`
- `tests/test_cli.py::test_list_pulls_no_gui_or_torch`

These use the same subprocess script and assertion structure, differing in the command, expected output token and failure label. Parameterize `(--help, spacr-run)` and `(--list, measure)` with explicit `help` and `list` IDs. Keep one fresh interpreter per case so an earlier import cannot hide an unwanted dependency.

**Saving:** 2 definitions become 1; both collected cases and both subprocesses remain. No runtime saving is claimed. The import-only test remains separate because it checks a different entry path.

## 6. Parameterize window-width boundary examples

- `tests/qt/test_the_window_opens_at_the_measured_width.py::TestItGrowsAndNeverShrinks::test_a_wider_requirement_widens_the_window`
- `tests/qt/test_the_window_opens_at_the_measured_width.py::TestItGrowsAndNeverShrinks::test_a_narrower_requirement_is_ignored`
- `tests/qt/test_the_window_opens_at_the_measured_width.py::TestItGrowsAndNeverShrinks::test_the_same_width_is_not_a_resize`
- `tests/qt/test_the_window_opens_at_the_measured_width.py::TestItNeverOpensPastTheEdge::test_it_is_clamped_to_the_display`

Parameterize the requested width, display dimensions, expected change flag and resulting width. Preserve the four named scenarios: wider, narrower, unchanged and display-clamped. The height-preservation and unavailable-policy tests remain separate.

**Saving:** 4 definitions become 1; all 4 cases remain. This consolidates four copies of the same setup and two assertions, with no expected runtime reduction.

## 7. Share the shutdown subprocess wrapper while retaining each exit path

- `tests/qt/test_quitting_during_a_slow_job.py::test_a_script_that_ends_during_a_job_runner_job_exits_cleanly`
- `tests/qt/test_quitting_during_a_slow_job.py::test_quitting_the_event_loop_during_a_slow_job_exits_cleanly`
- `tests/qt/test_quitting_during_a_slow_job.py::test_a_qthread_owned_by_a_widget_does_not_abort_the_exit`

Each body calls `_run(script, tmp_path)` and then `_assert_clean(out)`. Parameterize only that wrapper over the three complete scripts, with explicit IDs for interpreter exit, event-loop quit and widget-owned QThread exit. Keep their script-specific assertions. Keep the longer-than-budget and Home shutdown tests separate.

**Saving:** 3 definitions become 1; all 3 cases and all 3 subprocesses remain. Combining the scripts into one interpreter would lose independent shutdown coverage and is not proposed.

## 8. Share the focus-refusal contract across backdrop classes

- `tests/qt/test_ambient.py::test_a_focus_event_that_reaches_the_backdrop_is_refused`
- `tests/qt/test_dna_rain.py::test_a_focus_event_that_reaches_the_rain_is_refused`

The bodies are identical, but `make_widget` resolves to different implementations. A shared contract can accept a widget factory and keep one case for each class. Preserve the existing animation-preference setup and thread cleanup from both modules; deleting either scenario would remove coverage of a different widget.

**Saving:** 2 definitions become 1; both cases remain. This is a lower-priority maintenance cleanup because adapting the fixtures may offset the source-line reduction.

## Execution and savings limits

The two file removals account for all **14 immediately removable cases and 393 source lines** above. The capability-test merge would bring the case reduction to **15**, after its restoration assertion has been moved. The five parameterization recommendations remove **11 function definitions** while preserving all **17 cases** they represent. These totals do not double-count the removals.

Before implementing a removal, run the retained module and its affected fixture scope with the normal resource limits, and check collection paths used by CI. The graph timing cases need the project timing-test policy. This audit used source inspection and selected collection only; it did not execute rendering benchmarks, measurement runs or the full test suite.

# N47 Measure headings widget ownership, 2026-10-07

`tests/qt/test_features_button_opens_the_measure_table.py::test_measures_settings_have_the_measure_modules_headings` built a second Measure `SettingsWidgets` model without a Qt parent. Its `build_sections()` method materializes hundreds of editor widgets. The test already owns the real `MeasureInputsScreen` through `qtbot`; the single-line repair gives that temporary model the screen as parent. The headings, order, and one-placement assertions are unchanged.

All measurements used `/home/olafsson/anaconda3/envs/spacr/bin/python`, `tools/run_capped.sh 4G`, hidden CUDA, Qt offscreen, the existing passive serial journal, and the same selected test order. There was no forced garbage collection or memory-ceiling change.

| Boundary | Before fix | After fix |
|---|---:|---:|
| Two-node probe: headings test, then `test_field_fade.py::test_the_field_owner_outlives_its_native_painter` | 2 passed / 2.87 s | 2 passed / 2.61 s |
| After headings teardown | 1,224 widgets / 185 top levels; RSS 679.9 MiB | 1,225 widgets / 36 top levels; RSS 679.4 MiB |
| After field-fade sentinel teardown | 449 widgets / 167 top levels; RSS 679.9 MiB | 2 widgets / 1 top level; RSS 679.5 MiB |
| Exact seven preceding files plus sentinel | 164 passed / 21.86 s | 164 passed / 19.60 s |
| Before field-fade setup in that cohort | 447 widgets / 166 top levels; RSS 1,608.7 MiB | 0 widgets / 0 top levels; RSS 1,604.4 MiB |

The transient 1,225-widget snapshot immediately after the headings test is deferred Qt cleanup; the following ordinary file boundaries show those children are no longer retained. The paired RSS remains about 1.6 GiB because importing and allocating the same dependencies and widget tree can leave allocator pages resident even after ownership is fixed. This proves widget lifetime improvement, not a 4 MiB sustained-memory saving or a cure for the hosted 2,071-second field-fade outlier. It does not replace uninterrupted serial acceptance. The preceding cohort's before journal is already archived in `features/data/47_qt_style_hotspot_2026-10-07/` by commit `3066129a913b83225cc85f8749e06c78adc69ecd`.

`manifest.json` pins source and artifact bytes. The compressed journals contain every file boundary; `seven-predecessors-after.log.gz` holds the complete passing pytest output.

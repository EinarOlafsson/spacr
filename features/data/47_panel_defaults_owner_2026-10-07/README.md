# N47 module-defaults test control ownership, 2026-10-07

`tests/qt/test_the_panel_keeps_module_defaults.py::_built` built every registered module's settings controls without a Qt parent. Its all-apps case can materialize many unparented controls in one test. Every caller already has `qtbot`; the helper now registers a QWidget owner with it and passes that owner to `SettingsWidgets`. The default values, offered options and duplicate-option assertions are unchanged. The separate `_widget_for` case already registered its individual widget and is untouched.

An exact two-node probe selected `test_the_canned_table_really_does_override_something` followed by the unrelated field-fade sentinel. Both cases passed before and after with the existing passive journal, 4 GiB cap, Qt offscreen and hidden CUDA.

| Boundary | Before | After |
|---|---:|---:|
| After all-apps test teardown | 1,018 widgets / 384 top levels; RSS 624.2 MiB | 930 widgets / 33 top levels; RSS 629.0 MiB |
| After field-fade sentinel teardown | 1,020 widgets / 385 top levels; RSS 624.3 MiB | 2 widgets / 1 top level; RSS 629.0 MiB |

The complete affected test file plus the sentinel then passed 153/153 in 8.01 s under the same cap; the file ended with 0 widgets / 0 top levels and the sentinel ended with 2 / 1. The full-file run reported one busy JobRunner originating in its UMAP parameter case; this ownership repair does not establish that runner's cause or fix it. The paired RSS is 4.7 MiB higher after the change, so this is a demonstrated widget-ownership repair, not a measured RSS saving. N47 uninterrupted serial acceptance remains open. `manifest.json` pins source and artifact hashes.

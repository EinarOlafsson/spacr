# N47 settings-tree test ownership, 2026-10-07

`tests/qt/test_the_settings_tree_is_drawn.py::_model_tree` returned built Mask, UMAP, and Regression sections from temporary `SettingsWidgets` models with no Qt parent. The heading and translation tests called it repeatedly; their rendered controls survived the tests as top-level widgets while automatic cyclic GC was disabled by the normal GUI policy. `_model_tree` now registers one owner QWidget with `qtbot` and passes it as parent to each temporary model. The model and all heading, ordering, placement, and translation assertions are unchanged.

A matched 4 GiB CPU-only run selected the entire settings-tree file followed by `test_field_fade.py::test_the_field_owner_outlives_its_native_painter`, under the existing passive serial journal, hidden CUDA and Qt offscreen. No forced GC, production change or ceiling adjustment occurred.

| Boundary | Before | After |
|---|---:|---:|
| Selected tests | 18 passed / 8.92 s | 18 passed / 7.05 s |
| After settings-tree teardown | 6,489 live widgets / 2,626 top levels; RSS 714.2 MiB | 498 widgets / 11 top levels; RSS 625.0 MiB |
| After field-fade sentinel teardown | 5,818 widgets / 2,550 top levels; RSS 714.2 MiB | 2 widgets / 1 top level; RSS 625.0 MiB |

The 498-widget first boundary precedes ordinary deferred deletion; the next file boundary establishes they do not persist. RSS in this bounded process is 89.2 MiB lower after the same file order. This is not a whole-Qt memory reduction claim and does not establish the cause of the hosted multi-minute field-fade delay. N47 uninterrupted serial acceptance remains open. `manifest.json` pins the exact source and journal hashes.

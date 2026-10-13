# N47 UMAP row-exclusion test ownership, 2026-10-07

Five tests in `tests/qt/test_umap_row_exclusion_settings.py` built full UMAP settings models without Qt parents. All five already used `qtbot` to drive actual controls or wait for async source reads. One private helper now registers a QWidget owner with qtbot and passes it to each `SettingsWidgets` model. Every section, database value, filter and async behavior assertion remains unchanged.

The complete file followed by one unrelated field-fade sentinel passed 17/17 before and after under the existing 4 GiB cap, Qt offscreen, hidden CUDA and passive serial journal.

| Boundary | Before | After |
|---|---:|---:|
| After UMAP exclusion file teardown | 180 widgets / 20 top levels; RSS 566.3 MiB | 26 widgets / 3 top levels; RSS 569.0 MiB |
| After field-fade sentinel teardown | 156 widgets / 18 top levels; RSS 566.3 MiB | 2 widgets / 1 top level; RSS 569.0 MiB |

The paired RSS is 2.7 MiB higher after the fix; this is a widget ownership result, not a measured RSS saving. No production code, forced collection or memory ceiling changed. N47 uninterrupted serial acceptance remains open. `manifest.json` pins source and artifact hashes.

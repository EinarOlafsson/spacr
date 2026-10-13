# N47 database working-set UMAP test ownership, 2026-10-07

The `umap_panel(qtbot, qt_theme_applied)` fixture in `tests/qt/test_the_databases_are_a_working_set.py` built an unparented full UMAP settings panel for three cross-database control tests. It now registers a QWidget owner with qtbot and passes it to `SettingsWidgets`. The real GateEditor screen tests and all source, colour-by, collision and data assertions are unchanged.

The complete file followed by one unrelated field-fade sentinel passed 17/17 before and after under the same 4 GiB cap, hidden CUDA, Qt offscreen and passive serial journal.

| Boundary | Before | After |
|---|---:|---:|
| After working-set file teardown | 760 widgets / 86 top levels; RSS 1,581.9 MiB | 498 widgets / 11 top levels; RSS 1,578.0 MiB |
| After field-fade sentinel teardown | 80 widgets / 10 top levels; RSS 1,594.7 MiB | 2 widgets / 1 top level; RSS 1,590.3 MiB |

The first post-fix snapshot precedes deferred deletion; the next boundary shows the temporary panel no longer persists. The bounded RSS difference is 4.4 MiB at sentinel end and is not a whole-suite saving claim. No production code, forced collection or memory ceiling changed. N47 uninterrupted serial acceptance remains open. `manifest.json` pins source and artifact hashes.

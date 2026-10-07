# N47 UMAP reducer test ownership, 2026-10-07

`tests/qt/test_umap_reducer_settings.py::_model` built a full UMAP `SettingsWidgets` panel without a Qt parent while registering only its reducer control with `qtbot`. It now registers one owner QWidget with qtbot and passes it to the model; the reducer and all other controls share that owner. Every reducer, metric, disabled-setting and collected-setting assertion is unchanged.

The whole file followed by one unrelated field-fade sentinel passed 5/5 before and after under the same 4 GiB cap, hidden CUDA, Qt offscreen and existing passive serial journal.

| Boundary | Before | After |
|---|---:|---:|
| After UMAP file teardown | 279 widgets / 78 top levels; RSS 540.1 MiB | 203 widgets / 11 top levels; RSS 539.7 MiB |
| After field-fade sentinel teardown | 106 widgets / 13 top levels; RSS 540.1 MiB | 2 widgets / 1 top level; RSS 539.7 MiB |

The initial run warned about two busy JobRunners; the after run did not. These are timing-sensitive warnings, so this pair alone does not establish a worker fix. The first post-fix widget snapshot precedes deferred Qt cleanup; the following boundary shows no retained settings tree. The bounded RSS difference is 0.4 MiB and is not a full-suite saving claim. No production code, forced collection or memory ceiling changed. N47 uninterrupted serial acceptance remains open. `manifest.json` pins source and artifact hashes.

# N47 Regression panel test ownership, 2026-10-07

The `panel(qtbot)` fixture in `tests/qt/test_regression_panel_is_one_page.py` built every Regression setting without using qtbot or a Qt parent. One test built a second unparented copy. The fixture now creates and registers a QWidget owner and passes it to `SettingsWidgets`; the direct test reuses that existing panel. All settings, section and CSV picker assertions remain unchanged.

The complete file followed by one unrelated field-fade sentinel passed 35/35 both before and after under the unchanged 4 GiB cap, Qt offscreen and hidden CUDA. The existing passive serial journal measured:

| Boundary | Before | After |
|---|---:|---:|
| After Regression file teardown | 1,568 widgets / 228 top levels; RSS 549.2 MiB | 387 widgets / 19 top levels; RSS 545.4 MiB |
| After field-fade sentinel teardown | 1,227 widgets / 136 top levels; RSS 549.2 MiB | 2 widgets / 1 top level; RSS 545.4 MiB |

The first post-fix boundary includes deferred Qt deletion; the following ordinary file boundary shows the model controls no longer persist. The bounded RSS difference is 3.8 MiB and is not an aggregate or full-suite saving claim. No production code, forced collection or memory ceiling changed. N47 uninterrupted serial acceptance remains open. `manifest.json` pins source and artifact hashes.

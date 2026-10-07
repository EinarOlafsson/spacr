# Qt ownership of two settings-form tests (2026-10-07)

The host-pathogen buildability test and Qt plugin settings test each called
`SettingsWidgets(...).build_sections()` without a Qt parent. The returned
controls were inspected, but the tests did not mount or own the forms. This
receipt covers only those two test files and a following field-fade sentinel.

Measurements used Python 3.12.13, PySide6 6.11.2, hidden CUDA, offscreen Qt,
and an enforced 4 GiB cap. The existing passive per-test observer sampled
live widgets at the next test's setup; it did not run GC or process Qt events.
The `*-repeated-*` runs temporarily parameterized each of the two existing
test functions three times in an isolated worktree, retaining their original
assertions. That diagnostic parameterization was removed before commit.
Both matched seven-test sequences passed.

| Boundary after prior test | Before | After |
| --- | ---: | ---: |
| First plugin setup, after three host-pathogen builds | 18 live widgets | 0 |
| Third plugin setup, after earlier plugin builds | 4 | 0 |
| Following field-fade sentinel setup | 8 | 0 |

The 18 live widgets included nine settings `_ScalarEdit` controls; the eight
at the sentinel included `_ScalarEdit`, QLineEdit, Toggle and QDoubleSpinBox
controls. In the after run every recorded setup had zero live and tracked
widget wrappers. The owner is a `qtbot`-registered `QWidget` passed as the
existing optional `SettingsWidgets(parent=...)` argument. All original
settings, label, tooltip and API documentation assertions remain unchanged.

This is a demonstrated Qt ownership improvement, not a measured process-RSS
reduction. The first form build increased RSS from roughly 526 to 570 MiB in
both matched runs, then the bounded sequence plateaued. An import-only
neighbor ended near 528 MiB; the extra allocation follows actual form
construction. Retained allocator pages and imports prevent attributing the
remaining RSS to live Qt controls from these samples.

After removing the temporary repetition, both complete test files and the
same field-fade sentinel passed 5/5. Ruff and `git diff --check` passed.
`manifest.json` identifies the before and after source blobs and hashes all
compressed raw logs and observer journals.

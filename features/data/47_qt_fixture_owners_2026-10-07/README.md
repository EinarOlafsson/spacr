# Standalone Qt test widget ownership, 2026-10-07

Three pre-existing test constructions created Qt controls without a pytest-qt owner: the Compare panel fixture and the classify/measure `SettingsWidgets.build_sections()` paths. The latter can also create an unparented `QUndoStack`. The test-only repair registers the comparison panel with `qtbot` and passes a `qtbot`-owned QWidget to each standalone settings model. No application source, result assertion, test guard, timeout, or GC policy changed.

The exact four-node comparison (including the existing field-fade sentinel) passes before 4/4 and after 4/4. The after-change two-file plus sentinel cohort passes 37/37 under CUDA-hidden, offscreen 4 GiB. Raw logs and source/test hashes are recorded in `receipt.json`. This removes a test-owned object lifetime risk; it is not a claim that the prior native puncta SIGSEGV was caused by these fixtures.

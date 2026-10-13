# Notes from `spacr/qt/app_catalog.py`

Prose lifted out of `spacr/qt/app_catalog.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [LazyScreenFactory.__call__](#lazyscreenfactory__call__) (1 entry)
- [register_declared](#register_declared) (1 entry)
- [Module level](#module-level) (3 entries)

## LazyScreenFactory.__call__

### lines 180-182

```python
"""Import the screen class if needed and build one.
```

`inspect` is imported here rather than at the top of the file: it costs a dozen modules of its own, and this module is read while the splash screen is up to avoid exactly that kind of bill.

## register_declared

### lines 245-248

```python
from .app import APPS, register_app
```

Imported here rather than at the top: `app` reads this table while it is itself being imported, so a module-level import would be a cycle. By the time anything CALLS this, `register_app` is defined — that is what the ordering note in `spacr.qt` is about.

## Module level

### lines 362-367

```python
name='QC',
```

"QC", NOT "QC Dashboard". Asked for on 2026-08-31 as part of making ONE QC module: Layer Viewer and Control Charts folded in as buttons, and the module that hosts them is now just QC. The key is unchanged -- it is in saved sessions, run records and settings files, and renaming a display name must not rename anything that has been written to disk.

### lines 390-397

```python
translations=('QC',) * 9,
```

ALL NINE IDENTICAL, and that is a rule rather than laziness. "QC" is declared in `tools/build_i18n_catalogs.py::_IDENTITY_TEXT` alongside PNG and RGB -- text that must stay byte-identical in every language because it is an identifier, not a word. Translating it to 质控 and Gæðaeftirlit, which is what the old "QC Dashboard" names did, breaks that rule; the test `test_standalone_technical_identity_values_remain_exact_in_every_language` is what caught it.

### lines 875-880

```python
intro=(
```

UNDER 500 CHARACTERS, MEASURED. The runtime catalog's translator returns a long row unchanged rather than failing, and the audit then calls it "exact English". Of the six longest intros in this file, `dose_response` at 495 translates and this one at 544 did not -- the only difference being length. Keep it near the shorter of those two.

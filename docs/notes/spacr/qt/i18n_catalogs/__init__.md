# Notes from `spacr/qt/i18n_catalogs/__init__.py`

Prose lifted out of `spacr/qt/i18n_catalogs/__init__.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_module](#_module) (1 entry)
- [setting_label](#setting_label) (1 entry)

## _module

### lines 38-39

```python
if exc.name == name:
```

A missing optional catalog is a safe English fallback.  Do not hide an import failure *inside* a catalog module, which is a real defect.

## setting_label

### lines 106-111

```python
default_limit = 4
```

Slots above the four default organelles reuse the primary slot's reviewed translation.  Materialising all 26 otherwise copies the same 53 labels and tooltips 22 extra times into every language catalog.  Accept the alias only when both its key and its exact generated English label match; edited prose must still fall back to English instead of displaying a stale translation.

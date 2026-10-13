# Notes from `spacr/qt/i18n_module_summaries.py`

Prose lifted out of `spacr/qt/i18n_module_summaries.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [module_summary](#module_summary) (1 entry)

## Module level

### lines 23-27

```python
REVIEWED_SOURCE_HASHES = {
```

Hand-reviewed prose is still source-coupled data.  These hashes bind each reviewed row to the exact English summary it was reviewed against, just as the generated external catalogs do.  A changed app description therefore falls through to the current hashed external catalog (or safe English), instead of silently displaying an obsolete but fluent translation.

### lines 41-47

```python
"foreign": "9df3c545054e3fadfdf3fa1e193c30fd139ede278791bb8d2c3540687d2697aa",
```

REBOUND 2026-09-06. The English gained "or adopting masks made elsewhere" when the import screen landed, and the nine reviewed rows were rewritten with it -- all nine name that clause -- but this hash was not. A stale hash here does not show an obsolete translation, it shows NO reviewed translation: the row falls through to the generated catalog. So nine current, correct summaries were being suppressed by the mechanism meant to suppress obsolete ones.

## module_summary

### lines 97-99

```python
return _exact_translation(str(english), code) or str(english)
```

Plugins may ship exact translations in their manifest.  Do not apply conservative term substitution to a scientific paragraph: either the plugin supplies the whole sentence or it stays canonical English.

# Notes from `spacr/qt/widgets/metadata_table.py`

Prose lifted out of `spacr/qt/widgets/metadata_table.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [MetadataTablePanel.__init__](#metadatatablepanel__init__) (1 entry)
- [MetadataTablePanel._on_item_changed](#metadatatablepanel_on_item_changed) (1 entry)
- [MetadataTablePanel._recompute_canonical](#metadatatablepanel_recompute_canonical) (1 entry)

## Module level

### line 42  _(unsure)_

```python
_EDITABLE = {"plate", "well", "field", "channel", "time"}
```

Columns the user may edit; "original" and "canonical" are read-only.

## MetadataTablePanel.__init__

### line 86, trailing  _(unsure)_

```python
self._guard = False
```

re-entrancy guard while we rewrite cells

## MetadataTablePanel._on_item_changed

### line 147

```python
if key in _INT_COLS:
```

Coerce integer columns; revert bad input to 1.

## MetadataTablePanel._recompute_canonical

### line 165  _(unsure)_

```python
if not well.startswith(plate + "_") and "_" not in well:
```

Keep the plate prefix on the well token, matching convert_to_yokogawa.

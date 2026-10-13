# Notes from `spacr/plate_qc.py`

Prose lifted out of `spacr/plate_qc.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [_quote_ident](#_quote_ident) (1 entry)
- [_prc_parts](#_prc_parts) (1 entry)
- [plate_layout](#plate_layout) (1 entry)
- [_ring_profile](#_ring_profile) (1 entry)

## Module level

### lines 248-250  _(unsure)_

```python
_ROW_NUMERIC_RE = re.compile(r"^\s*(?:row)?[\s_-]*r?[\s_-]*(\d+)\s*$", re.IGNORECASE)
```

Well / row / column labels

## _quote_ident

### lines 392-394  _(unsure)_

```python
def _quote_ident(name: str) -> str:
```

Read-only database access (mirrors spacr.agreement / the Database Browser)

## _prc_parts

### lines 562-564

```python
best: Optional[Tuple[int, int]] = None
```

Candidate (row, column) token offsets, most likely first:

1,2 -> plateID_rowID_columnID[_fieldID]   (what spacr.io writes) 2,3 -> a plate name containing one underscore

## plate_layout

### lines 868-870

```python
empty = _empty_layout()
```

Everything was filtered away. There is no grid to infer — a 2x3 "plate" invented from an empty extent would be a lie with a shape.

## _ring_profile

### lines 1507-1508

```python
break
```

This ring is part of the core; comparing it against itself would be circular, so the profile stops here.

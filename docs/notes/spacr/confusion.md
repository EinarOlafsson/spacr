# Notes from `spacr/confusion.py`

Prose lifted out of `spacr/confusion.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [split_by_confidence](#split_by_confidence) (1 entry)
- [rank_confusions](#rank_confusions) (1 entry)

## split_by_confidence

### lines 304-306

```python
low = low.iloc[np.argsort(
```

NaN sorts last under argsort, which is what we want: a row with no confidence is not "the least confident", it is unknown, and it belongs after the rows that actually sat on the boundary.

## rank_confusions

### lines 395-396  _(unsure)_

```python
total_errors = float(sum(
```

A non-square matrix has no diagonal to trace; "off-diagonal" then means every cell whose row and column names differ.

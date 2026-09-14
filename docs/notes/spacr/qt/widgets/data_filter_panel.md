# Notes from `spacr/qt/widgets/data_filter_panel.py`

Prose lifted out of `spacr/qt/widgets/data_filter_panel.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [classify_columns](#classify_columns) (1 entry)
- [_classify_columns_uncached](#_classify_columns_uncached) (1 entry)
- [_ClauseRow.__init__](#_clauserow__init__) (1 entry)
- [_ClauseRow.clause](#_clauserowclause) (1 entry)
- [_RangeRow.__init__](#_rangerow__init__) (2 entries)
- [DataFilterPanel.__init__](#datafilterpanel__init__) (2 entries)
- [DataFilterPanel.state](#datafilterpanelstate) (2 entries)

## classify_columns

### lines 126-127

```python
pass
```

Not weak-referenceable. Skip the cache rather than risk an id-reuse false hit.

## _classify_columns_uncached

### lines 143-144  _(unsure)_

```python
distinct = series.nunique(dropna=True)
```

A numeric column with a handful of values (a class label, a plate number) is far more useful as ticks than as a range.

## _ClauseRow.__init__

### lines 182-184

```python
apply_close_mark(drop, tooltip=f"Stop filtering on {column}")
```

THE APPLICATION'S CLOSE MARK. The glyph, its square and its two colours come from the theme, which is also what keeps the target from shrinking when the mark grows. See `theme.apply_close_mark`.

## _ClauseRow.clause

### lines 191-193

```python
"""Return the filter clause this row describes.
```

THE CONTRACT EVERY ROW IS HELD TO. A subclass that forgets it fails loudly here rather than filtering on nothing, which reads as a filter that silently matches everything.

## _RangeRow.__init__

### lines 226-227

```python
hi = lo + 1.0
```

A constant column would give a spinbox with no travel; widen it so the control is still usable rather than inert.

### lines 236-238

```python
box.setRange(lo - abs(lo) - 1e6, hi + abs(hi) + 1e6)
```

The bounds are widened well past the data so a user can type a cut-off outside the observed range — which is exactly what you do when you want "everything above what this plate happens to show".

## DataFilterPanel.__init__

### lines 369-370

```python
self._link = link if link is not None else linked_selection()
```

Injectable so a test can drive a private instance rather than the process-wide one, which every other open view is also listening to.

### lines 409-410

```python
self._debounce = QTimer(self)
```

One shared debounce: a burst of edits across several clauses still costs one re-filter.

## DataFilterPanel.state

### lines 492-495

```python
def state(self) -> dict:
```

saving a filter set

Gates have had Save/Load since the beginning and filters had not, so a filter set -- which is as much of an analysis decision as a gate -- had to be rebuilt by hand every session.

### lines 504-507

```python
rows = [row.state() for row in self._rows.values()
```

`_rows` is a dict and dicts keep insertion order, which IS the order the user added the filters in and the order they read on screen. Restoring in the same order matters for a category filter whose box list is built from what earlier filters left.

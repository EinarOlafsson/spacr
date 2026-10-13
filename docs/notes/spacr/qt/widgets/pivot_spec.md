# Notes from `spacr/qt/widgets/pivot_spec.py`

Prose lifted out of `spacr/qt/widgets/pivot_spec.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [pivot](#pivot) (6 entries)

## Module level

### lines 75-79

```python
from .graph_spec import MISSING_LEVEL, _level_series, _sort_key
```

Both imported rather than re-derived. `_level_series` is how a NaN key becomes a visible level instead of a dropped row, and `_sort_key` is why plate 2 sorts before plate 10 — and the pivot's row order and the chart's facet order have to be the same order, or pivoting by plate and then plotting by plate would disagree about which plate is first.

## pivot

### lines 674-675

```python
_require_axis_column(frame, key)
```

Before the label frame is built, not after: reading a missing key out of the frame is what turns a stale spec into a bare KeyError.

### lines 694-695  _(unsure)_

```python
row_levels = row_levels[:len(row_levels) - 1]
```

Trim rows, not columns: a column removed loses a whole series, a row removed loses one group, and the table is read down the page.

### lines 712-714

```python
notices.append(
```

Text dropped on the values well. Every statistic would come out blank and every n zero, which reads as "no data" rather than as "you cannot average a gene name" — so it is said instead.

### lines 724-725  _(unsure)_

```python
work["__all"] = 0
```

No keys at all: one cell, the whole frame. `groupby` on a constant is the same computation with the same code path below.

### lines 733-736

```python
columns: Dict[Tuple[str, str], np.ndarray] = {}
```

One numpy column per (value, agg), positionally aligned to the group index. Reindexed onto it rather than trusted to match, and read positionally rather than through `.loc` — a label lookup per group per value turns a 50 000-group pivot into a visible pause.

### lines 741-748

```python
if QUANTILE in spec.aggs:
```

NO Series BRANCH. `SeriesGroupBy.agg` returns a DataFrame for a LIST of function names however short the list is, and `wanted` is never empty: `PivotSpec.__post_init__` starts its agg list with `n` and appends the rest, so `n` survives even `aggs=()` and `with_aggs(())`, and `n` maps to pandas' `count`.

Checked exhaustively -- all 256 subsets of the eight aggregations -- and `agg` returned a Series for none of them.

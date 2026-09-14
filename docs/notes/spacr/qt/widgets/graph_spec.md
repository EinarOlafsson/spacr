# Notes from `spacr/qt/widgets/graph_spec.py`

Prose lifted out of `spacr/qt/widgets/graph_spec.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [GraphSpec.__post_init__](#graphspec__post_init__) (1 entry)
- [_sort_key](#_sort_key) (2 entries)
- [facet_grid.axis](#facet_gridaxis) (1 entry)
- [facet_grid](#facet_grid) (3 entries)
- [prepare_data](#prepare_data) (2 entries)

## GraphSpec.__post_init__

### lines 321-322  _(unsure)_

```python
object.__setattr__(self, channel,
```

"" and None both mean "empty zone"; normalising here is what lets `if spec.x:` be the whole test everywhere else.

## _sort_key

### lines 538-539

```python
if value == value:
```

NaN has no order, and one in a sort key makes the whole sort arbitrary rather than wrong in one place. Treat it as text.

### line 544, trailing  _(unsure)_

```python
if index % 2:
```

split() alternates text, digits

## facet_grid.axis

### lines 688-691

```python
return (None,), 0
```

A facet column with no levels at all — everything filtered out, or an all-NaN column. One panel, drawn empty. A zero-column grid is not a figure matplotlib (or anyone) can draw, and "your filter matches nothing" is an answer worth rendering.

## facet_grid

### lines 702-703

```python
while len(row_levels) * len(col_levels) > max_panels:
```

Trim the *columns* axis first when the product is too big: a grid is read down the page, so losing a column costs less than losing a row.

### lines 710-715

```python
break
```

NEITHER AXIS CAN LOSE ANOTHER LEVEL. At one row and one column the product is 1, so the loop is only still running if the ceiling is below 1 -- which no caller in spaCR passes, but `max_panels` is a documented keyword and the cost of being wrong here is not a wrong picture, it is an infinite loop and a frozen window with nothing in the log.

### lines 720-722

```python
row_live = bool(spec.facet_row) and row_levels != (None,)
```

Only split on an axis that actually has levels: an axis that degenerated to `(None,)` above is a single panel, and matching rows against a level of ``None`` would put every one of them nowhere.

## prepare_data

### lines 1013-1017

```python
per_point_encoding = bool(spec.size) or (
```

Above the budget. Binning keeps every row, so it is preferred — but it can only draw what a raster can carry: a density, optionally shaded by the mean of a continuous colour column. A categorical colour or a size channel needs one mark per row, and for those the only honest option left is a sample the chart admits to.

### lines 1027-1029

```python
picked = np.sort(np.random.default_rng(spec.seed).choice(
```

Positional, seeded, and sorted back into the frame's own order — not `DataFrame.sample`, whose result has to be re-sorted by *index*, which is not the row order for a frame that arrived from a filter or a join.

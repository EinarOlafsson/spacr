# Notes from `spacr/qt/widgets/sortable_table.py`

Prose lifted out of `spacr/qt/widgets/sortable_table.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [is_missing](#is_missing) (1 entry)
- [_numeric_candidates](#_numeric_candidates) (2 entries)
- [_SortableMixin.__lt__](#_sortablemixin__lt__) (3 entries)
- [SortableTableItem.setData](#sortabletableitemsetdata) (1 entry)
- [_SortState.__init__](#_sortstate__init__) (1 entry)
- [_SortState.suspend_for_fill](#_sortstatesuspend_for_fill) (1 entry)
- [restore_natural_order](#restore_natural_order) (1 entry)
- [install_sorting](#install_sorting) (3 entries)

## Module level

### lines 67-70

```python
_SERIAL = 0
```

The natural order is the order the cells were created in. A serial handed out at construction records it without a second pass over the table, and a table is filled row by row, so serials increase down a column whichever way the loop that fills it is nested.

### lines 73-75

```python
_RESTORING = False
```

Set only while this module is putting a table back into its natural order. Sorting is synchronous and on the GUI thread, so a module-level flag is read by exactly the comparisons of the sort that set it.

## is_missing

### lines 99-100  _(unsure)_

```python
return False
```

isna on an array-like returns an array; a cell holding one is not missing.

## _numeric_candidates

### lines 135-136

```python
without_separators = _THOUSANDS.sub("", text)
```

1,234,567 -- a separator only between digits, never a decimal comma, which would turn "1,5" into fifteen thousand.

### lines 144-145

```python
head, sep, tail = text.rpartition(" ")
```

"3.2 s", "12 MB", "0.4 µm" -- the unit is a word after a space, not a letter glued to digits, which is what an identifier looks like.

## _SortableMixin.__lt__

### lines 231-233

```python
return NotImplemented
```

Declined, not guessed. ``sorted`` believes every answer it is given, so a cell that guessed would produce an order nobody could account for; NotImplemented lets Python raise.

### lines 239-241

```python
return mine_missing if self._descending() else theirs_missing
```

Missing goes last in BOTH directions, so it has to be the largest when Qt is ordering by "<" and the smallest when Qt is ordering by ">".

### lines 247-248

```python
return mine is not None
```

A number and a word in one column: numbers first, always the same way round, so the order never depends on which cell Qt asked.

## SortableTableItem.setData

### lines 289-290

```python
self._sort_serial = serial
```

The row keeps the place it was built in: an edit is not a new row, and the natural order must not shuffle under one.

## _SortState.__init__

### lines 446-452

```python
self._resume_timer = QTimer(self)
```

A static ``QTimer.singleShot`` stores a queued call to the Python bound method.  A short-lived table can disappear before that call is delivered; PySide then tries to resolve a slot on the already torn-down ``_SortState`` and reports ``Slot '_SortState::' not found`` from the event loop.  An owned timer is disconnected and destroyed with this state, so no callback can outlive the view it is meant to sort.

## _SortState.suspend_for_fill

### line 457  _(unsure)_

```python
@Slot(QModelIndex, int, int)
```

keeping a fill from scrambling the rows

## restore_natural_order

### lines 604-606

```python
model.sort(-1, Qt.AscendingOrder)
```

A proxy holds its source's order and hands it back when the sort column goes away. Asked through the model rather than through ``sortByColumn``, which declines a negative column.

## install_sorting

### lines 628-629

```python
getattr(view, _STATE_ATTR).stamp_initial_order()
```

Idempotent: a second call would wire the model signals twice and run every fill guard twice with it.

### lines 640-641  _(unsure)_

```python
header.setSortIndicatorClearable(True)
```

The third click. Without it Qt flips between two orders forever and the order the table was built in is unreachable.

### lines 643-644

```python
header.setSortIndicator(-1, Qt.AscendingOrder)
```

A fresh view already points its indicator at column 0, so the first click there would read as a flip and give ascending.

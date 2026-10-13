# Notes from `spacr/qt/widgets/column_aligned_row.py`

Prose lifted out of `spacr/qt/widgets/column_aligned_row.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ColumnAlignedRow.__init__](#columnalignedrow__init__) (3 entries)
- [ColumnAlignedRow.setGeometry](#columnalignedrowsetgeometry) (1 entry)
- [align_row_to_columns](#align_row_to_columns) (2 entries)

## ColumnAlignedRow.__init__

### lines 92-93

```python
"""Lay a row out against a header's column widths.
```

`parent` installs this as the widget's layout, which is why the caller has to have removed the previous one first.

### lines 115-120

```python
header.sectionResized.connect(self._restate)
```

THE ONLY THREE THINGS THAT MOVE A COLUMN. A drag on a section edge (`sectionResized`), a reorder if one is ever enabled (`sectionMoved`), and the header itself being re-laid-out when the table changes width (`geometriesChanged`). None of them resizes the strip, so without these the buttons would stay where the last strip resize left them.

### lines 124-128

```python
header.viewport().installEventFilter(self)
```

And the table MOVING under the strip, which changes where the columns are without changing their widths -- a section opening above it does exactly that. The viewport is watched rather than the table because it is the widget the column positions are measured from.

## ColumnAlignedRow.setGeometry

### lines 251-253

```python
right = rect.x()
```

Where the aligned run ends, so the un-aligned buttons start clear of it. Starts at the strip's own left edge for the case where no column could be read at all.

## align_row_to_columns

### lines 346-348

```python
while existing.count():
```

EMPTIED FIRST, THEN DELETED. Deleting a layout that still holds its items deletes the buttons with it, which would take the Download row away rather than align it.

### lines 354-356

```python
shiboken6.delete(existing)
```

`shiboken6.delete`, not `deleteLater`: the new layout is installed on the next line, and Qt refuses to install one while the widget still has a layout -- which a deferred deletion leaves it with.

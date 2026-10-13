# Notes from `spacr/qt/widgets/plate_map_picker.py`

Prose lifted out of `spacr/qt/widgets/plate_map_picker.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_Header.__init__](#_header__init__) (2 entries)
- [_Well.__init__](#_well__init__) (1 entry)
- [_Well._picker](#_well_picker) (1 entry)
- [_Well.mouseReleaseEvent](#_wellmousereleaseevent) (1 entry)
- [PlateMapPicker.__init__](#platemappicker__init__) (1 entry)
- [PlateMapPicker.set_layout_size](#platemappickerset_layout_size) (5 entries)
- [PlateMapPicker.begin_drag](#platemappickerbegin_drag) (2 entries)
- [PlateMapPicker._read](#platemappicker_read) (1 entry)

## _Header.__init__

### lines 152-153

```python
self.setFixedSize(well_side(), well_side())
```

The hint. The floor and the ceiling are in the sheet below, for the reason `_locked_square` gives.

### lines 155-160

```python
self.setStyleSheet("QLabel { border-width: 0px; %s }"
```

`border-width` IS STATED, and only the width. `_locked_square` pins padding, margin and the min/max box, but Qt adds the BORDER back on top of all four -- so an application sheet carrying `QLabel { border: 5px solid ... }` grew this header to 32 x 32 in a grid pitched at 22, and the letters drifted off their rows. The colour is left alone so the theme still paints the rim it wants.

## _Well.__init__

### lines 182-183  _(unsure)_

```python
self.setFixedSize(well_side(), well_side())
```

The hint only: `_paint` states the floor and the ceiling in the sheet, which is the half of this that survives being polished.

## _Well._picker

### lines 189-194

```python
def _picker(self):
```

the drag

THE PRESSED BUTTON RECEIVES EVERY MOVE, because Qt grabs the mouse on a press -- so a sibling's `enterEvent` never fires while a drag is in progress, and the well under the pointer has to be found by asking the grid rather than by waiting to be told.

## _Well.mouseReleaseEvent

### lines 246-248

```python
dragged = (picker is not None and hasattr(picker, "finish_drag")
```

BEFORE `super()`, which is what emits `clicked` and toggles the button: a drag that has already painted the rectangle must not then have its anchor flipped a second time by the click.

## PlateMapPicker.__init__

### line 310

```python
row = QHBoxLayout()
```

THE THREE BUTTONS THE ASK NAMED, bottom right and in that order.

## PlateMapPicker.set_layout_size

### lines 341-343

```python
while self._grid.count():
```

EVERY ITEM IN THIS GRID IS A WIDGET -- it is built with `addWidget` alone, and the minimum sizes and stretches below add no items of their own -- so each one taken out is a label or a well to drop.

### lines 349-351

```python
for index in range(self._grid.rowCount()):
```

A GRID KEEPS ITS ROW AND COLUMN COUNT when its items are taken out, so the stretch that absorbed the spare space on the previous layout would sit in the middle of a smaller one.

### lines 368-370

```python
self._grid.addWidget(well, row, column, Qt.AlignCenter)
```

CENTRED, LIKE ITS LABEL. Both share the cell, so a column number is over its column and a row letter beside its row however wide the cell has had to grow for the text in it.

### lines 374-375  _(unsure)_

```python
self._grid.setColumnMinimumWidth(0, well_side())
```

The corner the labels meet in holds nothing, and is a cell of the plate all the same.

### lines 378-382

```python
self._grid.setRowStretch(rows + 1, 1)
```

WHERE THE SPARE SPACE GOES: past the last well, into an empty row and column that hold nothing. The holder fills the scroll area, and a grid with nowhere to put the extra width shares it out among the cells -- which is exactly what pulls the numbers off their columns and the letters off their rows as the window grows.

## PlateMapPicker.begin_drag

### lines 412-417

```python
def begin_drag(self, row: int, column: int, modifiers=None) -> None:
```

the drag

`select_region` SELECTS A RECTANGLE WITHOUT A HUMAN, which is all it can do on its own: press, move and release are what reach it from a mouse, and a picker that is only ever driven through the method below has the gesture implemented and unreachable.

### lines 432-434

```python
self._before = self.selection()
```

WHAT TO GO BACK TO ON EVERY PREVIEW. A drag redraws from the state at the PRESS rather than from the last frame, so growing and then shrinking the rectangle leaves nothing behind.

## PlateMapPicker._read

### lines 563-564

```python
return set()
```

A FIELD THAT WILL NOT PARSE OPENS EMPTY rather than refusing to open: the picker is how a user fixes a value they typed wrong.

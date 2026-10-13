# Notes from `spacr/qt/screens/experiment_design.py`

Prose lifted out of `spacr/qt/screens/experiment_design.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [_Well.__init__](#_well__init__) (1 entry)
- [_Well.lock_square](#_welllock_square) (1 entry)
- [_Well.mouseMoveEvent](#_wellmousemoveevent) (1 entry)
- [_design_qss](#_design_qss) (1 entry)
- [ExperimentDesignScreen.__init__](#experimentdesignscreen__init__) (2 entries)
- [ExperimentDesignScreen._build](#experimentdesignscreen_build) (4 entries)
- [ExperimentDesignScreen.begin_well_drag](#experimentdesignscreenbegin_well_drag) (2 entries)
- [ExperimentDesignScreen._select_wells](#experimentdesignscreen_select_wells) (1 entry)
- [ExperimentDesignScreen._draw_plate](#experimentdesignscreen_draw_plate) (7 entries)

## Module level

### lines 47-54

```python
from ..widgets.plate_map_picker import _Header, _locked_square, well_side
```

THE OTHER PLATE'S GEOMETRY, IMPORTED RATHER THAN REPEATED. `well_side` is the side both plates are pitched at -- a FUNCTION, because the side follows the font scale and a constant read at import would freeze it at whatever the scale was when this module loaded -- `_locked_square` states that side in the one language that survives being polished under the application stylesheet, and `_Header` is the row letter and the column number locked to a cell of it. All three were written for the picker and are not specific to it; a second copy here is what let the two plates drift.

### lines 68-72

```python
_ROW = declared_app(APP_KEY)
```

The row this screen puts in the registry is declared in

`spacr.qt.app_catalog`, which is what lets the app be registered without importing this module -- the launch reads the table, not the screen. These read the same row back rather than restating it, so the name, the blurb and the nine translations have one spelling and no second copy to drift from.

### lines 351-353

```python
register_widget_qss("ExperimentDesign", _design_qss, replace=True)
```

`replace=True`: this module is reachable both through the screens package and by direct import, and a second import must refresh the block rather than raise. Same posture as the power screen.

## _Well.__init__

### lines 176-177  _(unsure)_

```python
self.setFixedSize(well_side(), well_side())
```

The hint. The floor and the ceiling are in the sheet below, which is the half of this that survives being polished.

## _Well.lock_square

### line 199  _(unsure)_

```python
self.setStyleSheet(_well_sheet(rim))
```

Setting a sheet repolishes the widget on its own.

## _Well.mouseMoveEvent

### lines 231-232

```python
"""Extend the drag-select to the well under the pointer.
```

THE PRESSED WIDGET KEEPS THE GRAB, so the well under the pointer has to be found by asking rather than by waiting to be entered.

## _design_qss

### lines 284-287

```python
return f"""
```

THE RIM WIDTHS BELOW COME FROM THE TABLE, not from a number typed here: a well states its own square in content-box pixels, and Qt adds the border back on, so a border this sheet draws wider than the well allowed for is a well two pixels bigger than its neighbours.

## ExperimentDesignScreen.__init__

### line 379

```python
self._well_anchor = None
```

194 A: the drag's state. `None` anchor means no gesture is live.

### lines 387-389

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## ExperimentDesignScreen._build

### lines 493-498

```python
self._plate_grid.setRowStretch(_SLACK, 1)
```

THE SLACK GOES OUTSIDE THE PLATE, NOT BETWEEN ITS WELLS. The grid is inside a resizable scroll area, so the panel grows with the window; without somewhere for that extra space to go, a QGridLayout hands it to the cells and the map spreads. One trailing row and column take all of it, which keeps every well the same distance from its neighbour at every window size.

### lines 506-508

```python
scroll.viewport().setAutoFillBackground(False)
```

The viewport auto-fills with the WINDOW colour, which no page opacity can reach. The plate map covers most of it, but the strip beside a short plate is the same slab the settings column was.

### lines 518-519

```python
self._findings_layout.setContentsMargins(SPACING["sm"], SPACING["xs"],
```

Room for the panel's own border, now that the findings sit on a surface rather than straight on the window.

### lines 530-531

```python
self._set_conditions([
```

A default worth starting from rather than an empty table: both controls present, enough replicates to mean something.

## ExperimentDesignScreen.begin_well_drag

### lines 661-670

```python
def begin_well_drag(self, row: int, column: int, modifiers=None) -> None:
```

194 A

"the plate map should be made up of squares that are clickable nad the user should be able to drag and select."

PRESS ANCHORS, MOVE PREVIEWS, RELEASE COMMITS -- and the preview is visible while dragging, because a selection you cannot see until you let go is one you have to undo to correct. Ctrl adds a second rectangle; a plain drag replaces, which is what every other grid in this application does.

### lines 677-678

```python
self._wells_before = set(self.selected_wells())
```

REDRAWN FROM THE STATE AT THE PRESS, not from the last frame, so growing and then shrinking the rectangle leaves nothing behind.

## ExperimentDesignScreen._select_wells

### lines 735-737

```python
label.lock_square()
```

Repaints AND re-squares: the selection is drawn as a rim, and a rim is part of the widget's size. See `_Well.lock_square`.

## ExperimentDesignScreen._draw_plate

### lines 741-747

```python
"""Rebuild the plate map, one square well per position.
```

WHAT THE USER CHOSE OUTLIVES THE REDRAW. Every well on this plate is destroyed and rebuilt on every `refresh`, and `refresh` runs on ONE KEYSTROKE in the plate name, on a nudge of the seed spinner and on every edit of the condition table -- so a selection read off the widgets alone was wiped by typing, not by anything the user did to the selection. Carried across as coordinates, which is the one form of it that survives the widgets being thrown away.

### lines 768-770

```python
chosen = {(row, column) for row, column in chosen
```

A SMALLER PLATE DROPS WHAT IS NO LONGER ON IT rather than keeping a coordinate that names nothing: 384 down to 96 has to forget H13, or a later switch back would resurrect a well the user cannot see.

### lines 778-783

```python
for column in range(1, columns + 1):
```

THE HEADERS ARE CELLS OF THE PLATE, not captions beside it, and `_Header` is the same one the picker's plate uses: locked to the well's square, so the header row is exactly one cell tall and the header column exactly one cell wide however tall the theme's font makes a label. Left to size themselves they answer to a blanket QLabel rule like anything else.

### lines 793-801

```python
label = _Well(row, column, self._plate_panel)
```

SQUARE, AND FIXED TO ITS NEIGHBOURS (194). This was a

`QLabel` with `setMinimumSize(22, 18)` and no maximum, so every well stretched with the window -- measured at 900, 1500 and 1900 px wide, one went 65 -> 111 -> 141 px across while staying 18 tall. A plate map is a picture of a physical object, and the point of the picture is that its proportions are the object's: at 141 x 18 it is not a plate any more, and "the elements drift appart" is what a reader sees.

### lines 803-807

```python
label.setProperty("wellName",
```

EVERY WELL KNOWS ITS NAME, assigned or not. A name is a

COORDINATE -- `letters_from_row_index(row)` and the column and it was being set only on the assigned branch, so a user who selected an empty block got a selection that could not say which wells it held.

### lines 827-831

```python
label.setProperty(
```

SET BEFORE THE SQUARE IS LOCKED, not restored afterwards:

the selection is drawn as a rim, a rim is part of the widget's size, and stating it here settles the well at its final size in one pass rather than resizing a plate's worth of wells the moment the selection is put back on them.

### lines 835-837

```python
label.lock_square()
```

The role, the edge mark and the selection are what decide the rim, so the square is settled once they are on the widget.

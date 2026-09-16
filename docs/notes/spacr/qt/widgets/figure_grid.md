# Notes from `spacr/qt/widgets/figure_grid.py`

Prose lifted out of `spacr/qt/widgets/figure_grid.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [reflow_shape](#reflow_shape) (2 entries)
- [axis_layout](#axis_layout) (1 entry)
- [SearchFigureGrid.__init__](#searchfiguregrid__init__) (1 entry)
- [SearchFigureGrid.relayout](#searchfiguregridrelayout) (1 entry)
- [SearchFigureGrid._make_label](#searchfiguregrid_make_label) (1 entry)

## reflow_shape

### lines 95-97

```python
fitting = max(1, (width + spacing) // (min_cell + spacing))
```

The most columns the width allows at all, ignoring how many figures there are. One column is always offered: a container too narrow for a readable cell still has to show something.

### lines 110-111  _(unsure)_

```python
overflow = used_h > height and height > 0
```

Prefer the shape that comes closest to filling the height without overflowing it; among those, the larger cell.

## axis_layout

### lines 153-154  _(unsure)_

```python
return [], [], [(0, i) for i in range(len(coordinates))]
```

No axes to speak of: arrival order, one long row. The widget reflows it; there is nothing meaningful to say about position.

## SearchFigureGrid.__init__

### lines 266-268

```python
make_transparent(self._page)
```

An anonymous QWidget inherits the blanket `QWidget { background: bg }` rule and paints the window colour as a solid rectangle over whatever is behind it. See INVARIANTS 1 and 3.

## SearchFigureGrid.relayout

### lines 393-395

```python
cell_w = max(
```

The axes decide the columns; the container decides how wide each one is. A search space is not free to be reshaped to fit a window -- moving a cell would change what it claims.

## SearchFigureGrid._make_label

### lines 427-429

```python
label.setText(cell.caption or "no figure")
```

A figure that failed to render is a missing result, not a missing widget: the cell stays so the grid keeps its shape and says which configuration produced nothing.

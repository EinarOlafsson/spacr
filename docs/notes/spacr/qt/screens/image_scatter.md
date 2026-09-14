# Notes from `spacr/qt/screens/image_scatter.py`

Prose lifted out of `spacr/qt/screens/image_scatter.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [load_scatter_frame](#load_scatter_frame) (2 entries)
- [Module level](#module-level) (2 entries)
- [ScatterCanvas._project](#scattercanvas_project) (2 entries)
- [ScatterCanvas.paintEvent](#scattercanvaspaintevent) (1 entry)
- [ImageScatterScreen.__init__](#imagescatterscreen__init__) (2 entries)
- [ImageScatterScreen.set_frame](#imagescatterscreenset_frame) (1 entry)
- [ImageScatterScreen._apply_linked_selection](#imagescatterscreen_apply_linked_selection) (1 entry)

## load_scatter_frame

### lines 86-88

```python
def load_scatter_frame(db_path: str, table: str,
```

Loading — off the GUI thread, and with no widget in sight

### lines 113-116

```python
return with_object_type(frame, table)
```

The frame does not know what it is; this function does. Without the stamp a point in the nucleus table and a point in the pathogen table publish the same key when they share a label, and clicking one opens whichever crop the table happened to list first.

## Module level

### lines 172-173

```python
register_widget_qss("ImageScatter", _image_scatter_qss, replace=True)
```

`replace=True`: reachable through the screens package and by direct import, and a second import must refresh the block rather than raise.

### lines 945-954

NO REGISTRY ROW. The scatter is reached as a button on Image UMAP's masthead -- :data:`spacr.qt.screens.image_umap.FOLDED_APPS` -- which builds it through :func:`make_image_scatter_screen` and then points it at the measurements database the UMAP screen is already reading. That seeding is what makes the fold a superset of the tile: a standalone tile opened on an empty path and made the user find the same file again.

The strings above are kept because they are this module's public description -- the fold button's name and sentence are asserted against them, and the i18n catalogs carry the translations.

## ScatterCanvas._project

### lines 286-287

```python
sx = width / (x1 - x0) if x1 > x0 else 0.0
```

A constant column would divide by zero; centre it instead of collapsing every point onto the left edge.

### line 291  _(unsure)_

```python
self._py[good] = (pad + height - (ys - y0) * sy if sy
```

Screen y grows downward; data y grows upward.

## ScatterCanvas.paintEvent

### line 367  _(unsure)_

```python
LOG.exception("Could not paint the image scatter")
```

A paintEvent that raises takes the window with it.

## ImageScatterScreen.__init__

### lines 487-488

```python
self._hover_timer = QTimer(self)
```

Debounce: the cursor crossing a cluster must not queue a decode per point it passed over. Only what it rests on is worth an image.

### lines 496-497  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

## ImageScatterScreen.set_frame

### lines 699-700

```python
for column in ("png_path", "sample", "path"):
```

A frame read straight from a crop table already carries the path; use it rather than going back to the database.

## ImageScatterScreen._apply_linked_selection

### lines 876-878

```python
self.canvas.set_selected(
```

Matched by specificity rather than by equality: a view that states no object type still has to light up for one that does, and the other way round. See `spacr.selection.match_keys`.

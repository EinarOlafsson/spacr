# Notes from `spacr/qt/layer_viewer.py`

Prose lifted out of `spacr/qt/layer_viewer.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_layer_viewer_qss](#_layer_viewer_qss) (1 entry)
- [stack_from_paths](#stack_from_paths) (2 entries)
- [CanvasTool](#canvastool) (1 entry)
- [LayerCanvas.paintEvent](#layercanvaspaintevent) (1 entry)
- [LayerCanvas.mouseReleaseEvent](#layercanvasmousereleaseevent) (1 entry)
- [LayerListWidget._on_layers_changed](#layerlistwidget_on_layers_changed) (1 entry)
- [LayerListWidget._on_rows_moved](#layerlistwidget_on_rows_moved) (1 entry)
- [LayerViewer.__init__](#layerviewer__init__) (1 entry)
- [LayerViewer._build](#layerviewer_build) (1 entry)
- [LayerViewer._on_activated](#layerviewer_on_activated) (1 entry)
- [Module level](#module-level) (2 entries)

## _layer_viewer_qss

### lines 105-107

```python
def _layer_viewer_qss(palette: Dict[str, Any], opacity) -> str:
```

Styling, through the seam rather than through theme.py

## stack_from_paths

### lines 160-162  _(unsure)_

```python
def stack_from_paths(image_path=None, labels_path=None, *,
```

Loading, reusing the preview stack's readers

### lines 189-190  _(unsure)_

```python
from .widgets.timelapse_preview import frame_channel
```

`frame_channel` owns the channels-first/last heuristic; asking it for channel 0 tells us which axis it decided was channels.

## CanvasTool

### lines 207-209  _(unsure)_

```python
class CanvasTool:
```

Tools — what takes the canvas's mouse away from picking

## LayerCanvas.paintEvent

### lines 454-455

```python
LOG.exception("Could not paint the layer canvas")
```

A paint handler that raises takes the window with it, and the traceback never reaches the console panel.

## LayerCanvas.mouseReleaseEvent

### lines 569-570

```python
canvas = self._ensure_canvas()
```

Offered to the tool AFTER the pan check, so releasing a shift-drag pan never reads as the end of a brush stroke.

## LayerListWidget._on_layers_changed

### lines 642-646

```python
"""Rebuild the rows for a stack that has changed."""
```

Not while this list is the one making the change. `refresh()` clears every row, and the handler that started it is still holding the QListWidgetItem it was called with — using it after the rebuild is a "C++ object already deleted" RuntimeError raised inside a Qt signal, where no `except` in this file can reach it.

## LayerListWidget._on_rows_moved

### line 714  _(unsure)_

```python
for position, name in enumerate(reversed(names)):
```

The list is top-first; the model is bottom-first.

## LayerViewer.__init__

### lines 754-755  _(unsure)_

```python
from .dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

## LayerViewer._build

### lines 812-814

```python
self.remove_button = close_mark_button(
```

THE APPLICATION'S CLOSE MARK, not a flat text button that happens to hold an X. Red under the pointer is the whole point on the one control in this row that destroys something.

## LayerViewer._on_activated

### lines 1049-1051

```python
if key is None or not has_object_opener(DEFAULT_OPEN_KIND):
```

Asked rather than caught: with nothing registered to show crops, a double click should do nothing visible, not raise NoObjectOpener out of a mouse handler.

## Module level

### lines 1145-1149

```python
_ROW = declared_app(LAYER_VIEWER_APP_KEY)
```

The row this screen puts in the registry is declared in

`spacr.qt.app_catalog`, which is what lets the app be registered without importing this module -- the launch reads the table, not the screen. These read the same row back rather than restating it, so the name, the blurb and the nine translations have one spelling and no second copy to drift from.

### lines 1173-1176

```python
("spacr.qt.screens.lineage", "register"),
```

Image Scatter used to ride in here. It is folded onto Image UMAP now a button on that masthead, opened already pointed at the same measurements database -- so it has no row to register and nothing to ride in on. `spacr.qt.screens.image_scatter` is imported by its host.

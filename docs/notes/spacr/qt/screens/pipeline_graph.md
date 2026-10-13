# Notes from `spacr/qt/screens/pipeline_graph.py`

Prose lifted out of `spacr/qt/screens/pipeline_graph.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [layout_rects](#layout_rects) (1 entry)
- [GraphCanvas._paint_nodes](#graphcanvas_paint_nodes) (1 entry)
- [PipelineGraphScreen.__init__](#pipelinegraphscreen__init__) (1 entry)
- [PipelineGraphScreen._build_ui](#pipelinegraphscreen_build_ui) (1 entry)
- [PipelineGraphScreen._on_graph_ready](#pipelinegraphscreen_on_graph_ready) (1 entry)

## Module level

### lines 88-92

```python
_ROW = declared_app(APP_KEY)
```

The row this screen puts in the registry is declared in

`spacr.qt.app_catalog`, which is what lets the app be registered without importing this module -- the launch reads the table, not the screen. These read the same row back rather than restating it, so the name, the blurb and the nine translations have one spelling and no second copy to drift from.

### lines 152-154

```python
register_widget_qss("PipelineGraphCanvasArea", _graph_qss, replace=True)
```

``replace=True`` because this module owns the name: a reimport (a test that reloads it, a plugin that pulls it in twice) must re-register the same block rather than raise on the duplicate and leave the screen unstyled.

## layout_rects

### lines 158-160  _(unsure)_

```python
def layout_rects(graph: PipelineGraph) -> Dict[str, QRect]:
```

Layout — a pure function, so the arrangement is testable without pixels

## GraphCanvas._paint_nodes

### lines 336-340

```python
continue
```

A layer naming an artifact the graph does not hold. Nothing

`build_graph` produces looks like this -- it builds the layers FROM the nodes -- but a graph handed in by a caller can, and a KeyError raised inside paintEvent is a window that will not redraw rather than one box missing.

## PipelineGraphScreen.__init__

### lines 412-413  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

## PipelineGraphScreen._build_ui

### lines 511-512  _(unsure)_

```python
mark_surface(self._details)
```

The canvas half has `PipelineGraphCanvasArea` for a surface; the detail pane is the other half and is a page surface too.

## PipelineGraphScreen._on_graph_ready

### lines 548-551

```python
self._set_verdict("The graph could not be built.", problem=True)
```

The job delivered nothing. Saying so is the point: a screen that silently keeps the PREVIOUS project's graph on it after a failed read is showing one project's provenance under another project's name.

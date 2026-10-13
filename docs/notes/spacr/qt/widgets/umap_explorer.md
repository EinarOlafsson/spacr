# Notes from `spacr/qt/widgets/umap_explorer.py`

Prose lifted out of `spacr/qt/widgets/umap_explorer.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_ScaledPreview.__init__](#_scaledpreview__init__) (1 entry)
- [ImageUmapExplorer.__init__](#imageumapexplorer__init__) (2 entries)
- [ImageUmapExplorer._build_ui._OwnedTimerFigureCanvas._spacr_draw](#imageumapexplorer_build_ui_ownedtimerfigurecanvas_spacr_draw) (1 entry)
- [ImageUmapExplorer._build_ui](#imageumapexplorer_build_ui) (5 entries)
- [ImageUmapExplorer.open_display_settings](#imageumapexploreropen_display_settings) (1 entry)
- [ImageUmapExplorer.apply_display](#imageumapexplorerapply_display) (1 entry)
- [ImageUmapExplorer._build_point_identity](#imageumapexplorer_build_point_identity) (2 entries)
- [ImageUmapExplorer._draw_embedding](#imageumapexplorer_draw_embedding) (1 entry)
- [ImageUmapExplorer._recompute_visible_points](#imageumapexplorer_recompute_visible_points) (1 entry)
- [ImageUmapExplorer._apply_point_alpha](#imageumapexplorer_apply_point_alpha) (1 entry)
- [ImageUmapExplorer._recompute_linked_points](#imageumapexplorer_recompute_linked_points) (2 entries)
- [ImageUmapExplorer.closeEvent](#imageumapexplorercloseevent) (2 entries)

## _ScaledPreview.__init__

### lines 319-321

```python
follow_device_ratio(self, self._rescale)
```

Dragging the window onto a denser screen changes how many real pixels this label has without changing its size, so no resize arrives and the crop would stay at the old density.

## ImageUmapExplorer.__init__

### lines 415-416  _(unsure)_

```python
self.link_selection("umap")
```

After the UI: both hooks repaint, and a filter can already be set by the time this screen opens.

### lines 418-420

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## ImageUmapExplorer._build_ui._OwnedTimerFigureCanvas._spacr_draw

### lines 471-472  _(unsure)_

```python
return
```

Qt may be closing the parent hierarchy in this same event-loop turn. There is nothing left to repaint.

## ImageUmapExplorer._build_ui

### lines 489-495

```python
self._body_splitter.setHandleWidth(SPACING["sm"])
```

A HAIRLINE THAT CAN STILL BE HIT. Trading width between the chart and the sidebar is this widget's main gesture -- a projection of a few thousand crops is unreadable at panel size -- and the theme paints every splitter handle 1px, which is 1px of paint and about 5px of grab. The handle keeps the 1px line the rest of the app uses and gets a real grab area around it, the same trade the console panel's divider makes.

### lines 534-545

```python
self._apply_selected = ElidingPushButton("Label lasso selection", self)
```

THESE TWO SET THE SIDEBAR'S FLOOR, AND THROUGH IT THE DIVIDER'S.

A plain QPushButton's size hint is its whole label, and its horizontal policy treats that as a hard minimum -- so "Propagate automatic clusters" pinned the sidebar at 198 px. Measured: the chart/sidebar divider moved on a 1400 px window and was STUCK at 1000 px and below, because the sidebar was already as narrow as its widest word allowed. It was never the image preview.

An eliding button shortens its own label instead, so the sidebar gives way and the divider moves at every window size. The full text stays reachable: eliding sets the tooltip to it.

### lines 556-559

```python
self._display_btn = QPushButton("Display settings…", self)
```

Every display setting in one window, live and not-live together, as asked. The propagate callback is the same seam the Mask live preview uses, so a value tuned here lands in the settings panel and is saved with the run rather than living only in this widget.

### lines 576-578

```python
try:
```

The line inside the grab area, so a wider handle does not become a wider bar -- and an accent line on hover, so the divider answers before it is dragged.

### lines 594-596

```python
handle = self._body_splitter.handle(1)
```

THE ONLY THING THAT SAYS THE DIVIDER IS THERE before it is found. A 1px line with no hover text is indistinguishable from the edge of the chart.

## ImageUmapExplorer.open_display_settings

### lines 643-645

```python
getter = getattr(self, "_settings_getter", None)
```

The not-live half is not held by this widget -- it belongs to the run -- so seed it from the settings panel when there is one, or the dialog opens showing zeros for settings that have values.

## ImageUmapExplorer.apply_display

### lines 698-699  _(unsure)_

```python
self._draw_embedding()
```

Redrawn from `self._embedding`, which nothing above touched, so every point keeps its coordinates and its neighbours.

## ImageUmapExplorer._build_point_identity

### lines 760-764

```python
identity = identity.iloc[:, :0]
```

A payload that names only *some* of its points is worse than one that names none, in both directions: half a lasso gets published as the whole of it, and a filter tested against a column of blanks dims every point as though it had matched nothing. Refuse the lot.

### line 775, trailing  _(unsure)_

```python
return
```

nothing to key on, and nothing to filter with

## ImageUmapExplorer._draw_embedding

### lines 833-835

```python
self._linked_artist = self._axes.scatter(
```

Selections made elsewhere get their own ring, in the accent colour rather than the foreground one, so "what I lassoed" and "what the table is showing me" stay tellable apart at a glance.

## ImageUmapExplorer._recompute_visible_points

### line 874

```python
def _recompute_visible_points(self) -> None:
```

the shared filter: dim, never remove

## ImageUmapExplorer._apply_point_alpha

### lines 930-935

```python
self._scatter._alpha = None
```

`Artist.set_alpha` short-circuits on `alpha != self._alpha`, which raises "the truth value of an array is ambiguous" when the artist is currently holding a per-point array and a scalar is being set (matplotlib 3.10). Array→array and scalar→scalar are fine; only this direction needs the array dropped first, and there is no public call that does it.

## ImageUmapExplorer._recompute_linked_points

### line 953

```python
def _recompute_linked_points(self,
```

the shared selection: highlight, never hide

### lines 967-971

```python
self._linked_points = np.flatnonzero(
```

`match_keys`, not `Index.isin`: the table publishes `..._f1_cell1` now that a reader states which table it read, while these points are keyed off a `prcfo` that states nothing. Exact equality highlighted nothing at all, which on a UMAP is indistinguishable from the user having lassoed empty space.

## ImageUmapExplorer.closeEvent

### line 1188  _(unsure)_

```python
pass
```

The process-wide link's C++ side is gone (interpreter teardown).

### lines 1195-1196  _(unsure)_

```python
if self._lasso is not None:
```

FigureCanvasQTAgg implements draw_idle with a zero-delay Qt timer. Cancel that pending draw before Qt deletes the C++ canvas.

# Notes from `spacr/qt/widgets/graph_builder.py`

Prose lifted out of `spacr/qt/widgets/graph_builder.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_orientation](#_orientation) (1 entry)
- [ColumnWell](#columnwell) (1 entry)
- [_DraggableList.mimeData](#_draggablelistmimedata) (1 entry)
- [DropZone.__init__](#dropzone__init__) (1 entry)
- [DropZone.set_column](#dropzoneset_column) (1 entry)
- [_canvas_class.OwnedTimerFigureCanvas.__init__](#_canvas_classownedtimerfigurecanvas__init__) (1 entry)
- [_canvas_class.OwnedTimerFigureCanvas.draw_idle](#_canvas_classownedtimerfigurecanvasdraw_idle) (1 entry)
- [GraphCanvas.__init__](#graphcanvas__init__) (1 entry)
- [GraphCanvas._build_ui](#graphcanvas_build_ui) (1 entry)
- [GraphCanvas.set_frame](#graphcanvasset_frame) (1 entry)
- [GraphCanvas.render_now](#graphcanvasrender_now) (3 entries)
- [GraphCanvas._draw_panel_marks](#graphcanvas_draw_panel_marks) (1 entry)
- [GraphCanvas._draw_points.update](#graphcanvas_draw_pointsupdate) (1 entry)
- [GraphCanvas._draw_bar](#graphcanvas_draw_bar) (1 entry)
- [GraphCanvas._draw_mean_bar](#graphcanvas_draw_mean_bar) (1 entry)
- [GraphCanvas._draw_jitter](#graphcanvas_draw_jitter) (1 entry)
- [GraphCanvas._label_panel](#graphcanvas_label_panel) (1 entry)
- [GraphCanvas.on_linked_selection_changed](#graphcanvason_linked_selection_changed) (1 entry)
- [GraphCanvas._drag_patch_style](#graphcanvas_drag_patch_style) (1 entry)
- [GraphCanvas._on_release](#graphcanvas_on_release) (1 entry)
- [GraphBuilderPanel.__init__](#graphbuilderpanel__init__) (1 entry)
- [_graph_builder_qss](#_graph_builder_qss) (1 entry)

## _orientation

### lines 185-188

```python
modern = True
```

AN UNPARSEABLE VERSION MEANS MODERN. Guessing old on a new matplotlib brings back the per-panel, per-render warning this function exists to silence; guessing modern on an old one is a TypeError the caller sees immediately. Fail toward the loud one.

## ColumnWell

### lines 195-197  _(unsure)_

```python
class ColumnWell(QWidget):
```

The well of columns, and the six zones

## _DraggableList.mimeData

### lines 318-319

```python
payload.setText(names[0])
```

A plain-text copy as well, so dropping a column into a text field elsewhere pastes its name rather than nothing.

## DropZone.__init__

### line 372

```python
apply_close_mark(
```

THE APPLICATION'S CLOSE MARK -- see `theme.apply_close_mark`.

## DropZone.set_column

### line 404  _(unsure)_

```python
self.style().unpolish(self)
```

Qt does not restyle on a property change by itself.

## _canvas_class.OwnedTimerFigureCanvas.__init__

### lines 576-577

```python
figure.patch.set_alpha(0.0)
```

Whatever is below is the surface now. Leaving the patch opaque would paint the old rectangle straight back over it.

## _canvas_class.OwnedTimerFigureCanvas.draw_idle

### lines 600-602

```python
self._draw_pending = False
```

A queued host redraw may arrive after Qt has destroyed the canvas-owned timer during window teardown. There is no live surface left to update, so discard the pending draw.

## GraphCanvas.__init__

### lines 678-679  _(unsure)_

```python
self._visible: Optional[pd.DataFrame] = None
```

What the last render produced — kept so brushing, highlighting and the tests can ask what is on screen without re-deriving it.

## GraphCanvas._build_ui

### lines 709-712

```python
self._figure = Figure(figsize=(7.5, 5.0))
```

No `facecolor` and no inline `background:` — the canvas paints the page panel in its own `paintEvent` and its figure patch is transparent, so either of those would put an opaque rectangle back over it and stop the page-opacity slider reaching the chart.

## GraphCanvas.set_frame

### lines 743-745

```python
self._keyed = False
```

No object key columns: the chart still draws, it just cannot join the linked selection. Said out loud in the notice rather than silently publishing nothing.

## GraphCanvas.render_now

### lines 857-859

```python
self._figure.patch.set_alpha(0.0)
```

`clear()` restores the rc facecolor AND its alpha, so the transparency the canvas set has to be re-asserted or the first redraw paints the opaque rectangle straight back over the panel.

### lines 885-886

```python
self._brush_grid = (grid if data.frame is self._visible
```

A second grid over the *unsampled* rows, so a brush selects every row inside the rectangle rather than only the ones drawn.

### lines 890-891  _(unsure)_

```python
scale_source = data.frame
```

Limits from `data.frame` (post-filter) or from the whole table, depending on RESCALE_ON_FILTER -- see the class attribute.

## GraphCanvas._draw_panel_marks

### lines 1095-1097

```python
return None
```

A KIND THE CHAIN DOES NOT KNOW. The return value is an updater for a cheap highlight repaint, so None is the honest answer for a kind that drew nothing.

## GraphCanvas._draw_points.update

### lines 1175-1177

```python
base.set_alpha(self.POINT_ALPHA)
```

The configured opacity, not a literal: this runs on every selection change, so a hard-coded value here quietly undoes the user's setting the first time anything is highlighted.

## GraphCanvas._draw_bar

### lines 1281-1285

```python
other = spec.y if column == spec.x else spec.x
```

A MEAN BAR WHEN THERE IS SOMETHING TO AVERAGE (204). The other channel is numeric only when the user put a measurement there; with one categorical column alone this is a COUNT bar, and a count has nothing to be spread about -- an error bar on it would be a statement about a number that is exact.

## GraphCanvas._draw_mean_bar

### lines 1332-1335

```python
ax.errorbar(range(len(levels)), heights, yerr=errors,
```

`np.nan` in `yerr` draws nothing for that bar, which is the right answer for a level with one observation -- a zero-length whisker would say "no variation measured" where the truth is "not measurable".

## GraphCanvas._draw_jitter

### lines 1378-1381

```python
colour = palette["fg"] if over_bars else self._series_colour(0)
```

OVER A BAR, THE POINTS MUST READ AS POINTS. On their own they are the whole plot and can be solid; on top of a bar they are an annotation of it, so they lighten and shrink rather than competing with the shape underneath.

## GraphCanvas._label_panel

### lines 1507-1512

```python
y_label = "count" if counts_on_y else (y_column or "")
```

A BAR IS ONLY A COUNT WHEN THERE IS NOTHING TO AVERAGE (204). With a numeric channel the bar is a MEAN, and if it carries a whisker the label has to say which one -- SD and SEM differ by sqrt(n), fifty-five-fold at n=3000, so a reader who assumes the wrong one reads a real effect as noise or the reverse. This label is drawn AFTER `_draw_bar`, so setting it there was not enough.

## GraphCanvas.on_linked_selection_changed

### lines 1588-1591

```python
self.render_now()
```

An aggregate's highlight is a recomputed reduction, not a re-styled artist, so it costs a redraw. Point marks do not: two artists move, which is what keeps a lasso in another view from re-rendering fifty thousand marks here.

## GraphCanvas._drag_patch_style

### lines 1660-1663

```python
def _drag_patch_style(self) -> dict:
```

The preview shape is a HOOK because the shape being previewed is not always a rectangle. The Gate Editor draws ovals with the same gesture, and a rectangular preview for an elliptical gate tells the user the wrong thing about what they are about to make.

## GraphCanvas._on_release

### lines 1712-1713  _(unsure)_

```python
self.clear_linked_selection()
```

A click, not a drag: back to the resting state, which is a different thing from an empty selection.

## GraphBuilderPanel.__init__

### lines 1845-1847

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## _graph_builder_qss

### lines 1938-1940

```python
def _graph_builder_qss(palette, opacity) -> str:
```

Styling, through the seam rather than by editing theme.py

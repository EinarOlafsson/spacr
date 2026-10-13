# Notes from `spacr/qt/widgets/pca_view.py`

Prose lifted out of `spacr/qt/widgets/pca_view.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [FeaturePicker.__init__](#featurepicker__init__) (1 entry)
- [ScreePlot.__init__](#screeplot__init__) (2 entries)
- [ScreePlot.render_now](#screeplotrender_now) (1 entry)
- [PCAScoresCanvas.__init__](#pcascorescanvas__init__) (1 entry)
- [PCAScoresCanvas.render_now](#pcascorescanvasrender_now) (1 entry)
- [PCAScoresCanvas._draw_arrows](#pcascorescanvas_draw_arrows) (2 entries)
- [PCAPanel.__init__](#pcapanel__init__) (1 entry)
- [PCAPanel.recompute](#pcapanelrecompute) (2 entries)
- [PCAPanel.recompute._fit](#pcapanelrecompute_fit) (2 entries)
- [PCAPanel._apply_view](#pcapanel_apply_view) (1 entry)

## Module level

### lines 68-71

```python
from .graph_builder import (GraphCanvas, _canvas_class, _page_surface_axes,
```

`_canvas_class` is the owned-timer FigureCanvas fix — a matplotlib canvas whose deferred draw cannot fire after Qt has deleted it, which is a segfault on close. Imported rather than copied: two copies of a crash fix is one copy too many, and the scree plot needs the same protection the scores plot has.

### lines 1050-1057

```python
register_widget_qss("PCA", _pca_qss, replace=True)
```

Registered at import of this module, which happens when the screen module is imported — and the row that does that lives in ``app.py``'s ``_SELF_REGISTERING_APPS``, whose loop runs while ``app.py`` itself is being imported. That is before ``launch()`` calls ``stylesheet()``, which is the deadline: a block registered after the stylesheet is built is missing from the one the application was actually given. `spacr.qt.widgets.__init__` imports `graph_builder` eagerly for exactly this reason; this module needs no such entry only because its screen is imported earlier still.

## FeaturePicker.__init__

### lines 176-178

```python
self._checked: set = set()
```

Per instance, not per class: a set on the class would be shared by every picker in the process, so opening a second PCA screen would silently retick the first one's features.

## ScreePlot.__init__

### lines 334-335  _(unsure)_

```python
self._figure = Figure(figsize=(3.4, 2.4))
```

No `facecolor` and no inline `background:` — the canvas paints the page panel in its own `paintEvent` under a transparent figure patch.

### lines 337-339

```python
self._canvas = _canvas_class()(self._figure, panel=False)
```

`panel=False`: the scree plot sits inside `PCAShelf`, which is already a page surface, and a second panel under the figure would read 0.49 at a requested 30 % -- a shade the slider cannot reach.

## ScreePlot.render_now

### line 363  _(unsure)_

```python
self._figure.patch.set_alpha(0.0)
```

`clear()` restores the rc facecolor and its alpha with it.

## PCAScoresCanvas.__init__

### lines 451-452  _(unsure)_

```python
"""Build the scores plot and link it to the shared selection.
```

Before super().__init__: the base constructor wires a debounce timer to self.render_now, which is this class's override and reads these.

## PCAScoresCanvas.render_now

### lines 531-533

```python
LOG.debug("could not draw the loading arrows", exc_info=True)
```

A DECORATION MUST NEVER TAKE THE CHART WITH IT. The scores are the plot; the loading arrows are an overlay on top of them, so a failure here costs the arrows and nothing else.

## PCAScoresCanvas._draw_arrows

### lines 554-556

```python
first = axes[min(axes)]
```

One scale for every panel, from the first one: faceted panels share their axes, and an arrow that changed length between panels would make two panels of the same PCA look like two different PCAs.

### lines 575-590

```python
for i in picked:
```

THE FINITE CHECK IS FOR set_result's CALLERS, not for

`pca()`. A result from `pca()` cannot carry a non-finite correlation -- it ends that block with `np.clip(np.nan_to_num(correlations), -1, 1)`, and 4,000 adversarial frames (zero-variance columns, collinear pairs, 1e12 and 1e-12 magnitudes, NaN and infinite entries) never produced one.

But `set_result` is public and takes any PCAResult, and the dataclass validates nothing. An arrow to a NaN is a line to nowhere on a plot the reader takes at face value, so the feature loses its arrow and the rest keep theirs.

It carried a `no cover` pragma claiming it was unreachable. It was covered all along, by test_a_feature_whose_correlation_is_not_a_number_gets_no_arrow.

## PCAPanel.__init__

### lines 773-775

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## PCAPanel.recompute

### lines 856-857

```python
self._jobs.cancel()
```

A superseded fit must not paint over the one the user asked for: dragging the components spin box starts one per value.

### lines 879-882

```python
return None if self._threaded else self._result
```

Unthreaded, `submit` has already run the fit and `_on_fit_done`, so `_result` is this call's result. Threaded, it is whatever was on screen before, and returning that would be a lie about which spec it came from.

## PCAPanel.recompute._fit

### lines 867-871

```python
LOG.info("PCA failed", exc_info=True)
```

ANYTHING THAT IS NOT A PCAError. That one is the expected refusal and carries its own explanation; this is a fault inside the decomposition, and it runs on a worker where an escaping exception has nowhere to go. The 'PCA failed:' prefix is what tells the two apart.

### lines 874-875

```python
return {"result": result, "scores": result.scores_frame(frame)}
```

`scores_frame` is another pass over the table; it belongs on this side of the boundary with the fit, not on the GUI thread.

## PCAPanel._apply_view

### line 957  _(unsure)_

```python
self.canvas.set_biplot(self._biplot.isChecked(),
```

Arrows first with render off, then the spec — one redraw per action.

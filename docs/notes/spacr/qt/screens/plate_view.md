# Notes from `spacr/qt/screens/plate_view.py`

Prose lifted out of `spacr/qt/screens/plate_view.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_cmap_lut](#_cmap_lut) (1 entry)
- [PlateGridWidget.__init__](#plategridwidget__init__) (1 entry)
- [PlateGridWidget.paintEvent](#plategridwidgetpaintevent) (4 entries)
- [PlateViewScreen.__init__](#plateviewscreen__init__) (3 entries)
- [PlateViewScreen.closeEvent](#plateviewscreencloseevent) (1 entry)
- [PlateViewScreen._build_ui](#plateviewscreen_build_ui) (4 entries)
- [PlateViewScreen.recompute](#plateviewscreenrecompute) (1 entry)
- [PlateViewScreen.recompute._job](#plateviewscreenrecompute_job) (1 entry)
- [PlateViewScreen._draw_plate](#plateviewscreen_draw_plate) (1 entry)
- [PlateViewScreen._run_job](#plateviewscreen_run_job) (1 entry)

## _cmap_lut

### lines 141-142  _(unsure)_

```python
level = int(round(255 * t))
```

Greyscale fallback — a plate is still readable without matplotlib, which is better than a screen that won't paint.

## PlateGridWidget.__init__

### lines 191-192

```python
make_transparent(self)
```

The panel is drawn in `paintEvent`; the widget must not also paint the blanket opaque window fill under it. See `paint_panel`.

## PlateGridWidget.paintEvent

### lines 349-351

```python
paint_panel(painter, self, role="surface", inset=0.5)
```

A rounded panel at the page opacity. `fillRect(..., surface)` is opaque hex, which is why the "choose a database, a table, and a measurement, then press Render" state read as a bare dark area.

### line 367  _(unsure)_

```python
palette = active_palette()
```

Column numbers along the top.

### line 374  _(unsure)_

```python
for r in range(1, self._n_rows + 1):
```

Row letters down the side.

### line 387  _(unsure)_

```python
painter.setPen(empty_pen)
```

Blank, and visibly so: an absent well is not a zero.

## PlateViewScreen.__init__

### lines 466-470

```python
self._recompute_timer = QTimer(self)
```

One aggregation per gesture rather than one per spin-box tick — see `_on_view_changed`. Unthreaded, there is no timer at all and an option change recomputes on the spot, which is the same rule `_run_job` already follows: `threaded=False` means "behave exactly as this did before any of it moved off the GUI thread".

### lines 484-487

```python
self.link_selection("plate_view")
```

Join the shared population. The mixin carries the rules this screen used to spell out itself: bound methods rather than lambdas (the link is process-wide and outlives every screen), a flag-guarded disconnect, and echo suppression.

### lines 489-491

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## PlateViewScreen.closeEvent

### line 516  _(unsure)_

```python
pass
```

The singleton is gone during interpreter teardown.

## PlateViewScreen._build_ui

### line 544  _(unsure)_

```python
src_row = QHBoxLayout()
```

── Source row ────────────────────────────────────────────────

### line 564  _(unsure)_

```python
pick_row = QHBoxLayout()
```

── Selection row ─────────────────────────────────────────────

### line 629  _(unsure)_

```python
split = QSplitter(Qt.Horizontal, self)
```

── Heatmap | report ──────────────────────────────────────────

### line 656  _(unsure)_

```python
self._well_label = QLabel("Click a well to see what is behind it.", self)
```

── Well readout ──────────────────────────────────────────────

## PlateViewScreen.recompute

### lines 974-986

```python
note = ""
```

Honour the shared Local Data Filter, so narrowing the population in one view narrows it here too. Applied to the frame already in memory rather than at the query, because the frame is cached across renders and a filter change must not cost a re-read of the database.

It degrades to the unfiltered frame rather than refusing to draw: a filter carried over from another table can name columns this one does not have, and an empty heatmap is a worse answer than a complete one -- PROVIDED the view says which it is showing, which is what `_filter_note` puts on the status line.

`is_empty` and `describe()` are read HERE, on the GUI thread, so the worker is handed a frame and a note and never looks at the link.

## PlateViewScreen.recompute._job

### lines 1014-1019

```python
return {"error": exc}
```

Carried back as data rather than raised. An aggregation that refuses -- "no such column", "this plate is not in the frame" -- is a sentence for the status line, and it was written by this screen before the work moved to a thread. Letting it out as a worker error would relabel it "Plate view failed: …" and leave the old grid on screen.

## PlateViewScreen._draw_plate

### lines 1065-1067

```python
+ getattr(self, "_filter_note", ""))
```

A filtered heatmap that does not say it is filtered is how an edge-effect verdict gets read as covering the whole plate when it covers a third of it.

## PlateViewScreen._run_job

### lines 1211-1213

```python
self._jobs.append((thread, worker))
```

Strong references: PySide6 will not keep the worker alive through the started→run connection alone, and a QThread garbage-collected while still running takes the process down with it.

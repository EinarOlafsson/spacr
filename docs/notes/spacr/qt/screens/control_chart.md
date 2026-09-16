# Notes from `spacr/qt/screens/control_chart.py`

Prose lifted out of `spacr/qt/screens/control_chart.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [ControlChartCanvas.__init__](#controlchartcanvas__init__) (1 entry)
- [ControlChartCanvas.render_now](#controlchartcanvasrender_now) (4 entries)
- [ControlChartScreen.__init__](#controlchartscreen__init__) (4 entries)
- [ControlChartScreen._build_controls](#controlchartscreen_build_controls) (2 entries)
- [ControlChartScreen._refill_pickers](#controlchartscreen_refill_pickers) (1 entry)
- [ControlChartScreen.spec](#controlchartscreenspec) (1 entry)
- [ControlChartScreen.closeEvent](#controlchartscreencloseevent) (1 entry)

## Module level

### lines 81-82

```python
register_widget_qss("ControlChart", _control_chart_qss, replace=True)
```

`replace=True`: reachable through the screens package and by direct import, and a second import must refresh the block rather than raise.

### lines 84-87

```python
from ..widgets.graph_builder import (_canvas_class, _page_surface_axes,
```

`_canvas_class` is the owned-timer FigureCanvas fix: matplotlib schedules its idle draw on a static QTimer that is not owned by the canvas and can fire after Qt has deleted it, which is a segfault on close. Imported from the one place that has it rather than copied.

### lines 864-868

```python
_ROW = declared_app(APP_KEY)
```

The row this screen puts in the registry is declared in

`spacr.qt.app_catalog`, which is what lets the app be registered without importing this module -- the launch reads the table, not the screen. These read the same row back rather than restating it, so the name, the blurb and the nine translations have one spelling and no second copy to drift from.

## ControlChartCanvas.__init__

### lines 168-170

```python
self.figure = Figure(figsize=(8.0, 4.2))
```

No `facecolor` and no inline `background:` -- the canvas paints the page panel in its own `paintEvent` under a transparent figure patch, and either of those would put the opaque rectangle back.

## ControlChartCanvas.render_now

### line 192  _(unsure)_

```python
self.figure.patch.set_alpha(0.0)
```

`clear()` restores the rc facecolor and its alpha with it.

### lines 219-222

```python
if not result.degenerate:
```

The zones, outermost first so the inner ones sit on top. Three bands rather than two lines because rules 5 and 6 are statements about the 2- and 1-sigma bands, and a reader cannot check them against a chart that only draws the 3-sigma limit.

### lines 235-238

```python
if result.baseline.size and int(result.baseline.max()) < len(result) - 1:
```

Where Phase I ends. The limits are a statement about the points to the left of this line and a test of the points to the right, and a chart that does not show the boundary invites reading the baseline as evidence for itself.

### lines 251-253

```python
series = categorical_colours()
```

One overplotted marker per rule, so a plate that trips three rules carries three marks and the legend says which. Colour-coded by rule number through the fixed eight-hue series — eight rules, eight hues.

## ControlChartScreen.__init__

### lines 378-379

```python
lower.setObjectName(OUTPUT_OBJECT)
```

Named, so it is a panel rather than scaffolding the container sweep tags transparent -- see `_control_chart_qss`.

### lines 382-383  _(unsure)_

```python
lower_layout.setContentsMargins(SPACING["sm"], SPACING["sm"],
```

Room for the column's own rounded surface around the report and the violations table, which show it through.

### lines 415-416  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

### lines 419-421

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## ControlChartScreen._build_controls

### lines 431-438

```python
from ..preferences import scaled_px
```

SCALED, NOT A DEVICE-PIXEL CONSTANT. This cap exists to stop the settings column eating the figure beside it, and 330 px is the right answer at 100 %% -- and only there. The glyphs inside it double at 200 %% and the box did not, which is the same defect instruction 350 already fixed on UsageBar's fixed 48 px caption column. Measured on Control Charts: the column's own sizeHint wants 586 px at 100 %%, 707 at 125 %% and 1107 at 200 %%, against a cap that stayed 330 in all three.

### lines 442-443

```python
form.setContentsMargins(SPACING["sm"], SPACING["sm"],
```

Room for the panel's own rounded surface: the column sits ON a page surface now rather than straight on the window.

## ControlChartScreen._refill_pickers

### lines 568-572

```python
values = [name for name in columns
```

The classifier offers *continuous* columns, and a control that never moved is not continuous — which is exactly the table a user opens this screen to find out about. Falling back to every numeric column keeps the degenerate case reachable; falling back to every column would offer the plate id as a measurement.

## ControlChartScreen.spec

### lines 674-675  _(unsure)_

```python
levels = tuple(x for x in (positive, negative) if x)
```

Z' does not chart one control's level, so an empty tick list is not a missing answer here — the two named controls are.

## ControlChartScreen.closeEvent

### lines 847-849

```python
"""Stop background work and unlink before going away.
```

Abandon in-flight work rather than let it outlive the screen: Qt aborts the process if a running QThread is destroyed, and a worker delivering into a closed widget is a use-after-free.

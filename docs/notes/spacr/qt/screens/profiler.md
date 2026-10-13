# Notes from `spacr/qt/screens/profiler.py`

Prose lifted out of `spacr/qt/screens/profiler.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [curve_points](#curve_points) (2 entries)
- [ProfilerScreen.__init__](#profilerscreen__init__) (2 entries)
- [ProfilerScreen._build_ui](#profilerscreen_build_ui) (1 entry)
- [ProfilerScreen._on_model_ready](#profilerscreen_on_model_ready) (2 entries)
- [ProfilerScreen.variable](#profilerscreenvariable) (1 entry)

## Module level

### lines 84-88

```python
_ROW = declared_app(APP_KEY)
```

The row this screen puts in the registry is declared in

`spacr.qt.app_catalog`, which is what lets the app be registered without importing this module -- the launch reads the table, not the screen. These read the same row back rather than restating it, so the name, the blurb and the nine translations have one spelling and no second copy to drift from.

### lines 136-137

```python
register_widget_qss("ProfilerPlot", _profiler_qss, replace=True)
```

``replace=True`` because this module owns the name: a reimport must re-register the same block rather than raise and leave the screen unstyled.

## curve_points

### lines 141-143  _(unsure)_

```python
def curve_points(curve: Optional[Profile], width: int, height: int, *,
```

The curve, as a pure function

### lines 170-171

```python
y_low, y_high = y_low - 0.5, y_high + 0.5
```

A flat curve is a real answer ("this input does nothing"), and it must be drawn along the middle rather than divided by zero.

## ProfilerScreen.__init__

### lines 327-328  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

### lines 331-333

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## ProfilerScreen._build_ui

### lines 401-402  _(unsure)_

```python
mark_surface(self._inputs)
```

The plot half of this splitter has `ProfilerPlot` for a surface; the input tree is the other half and had none.

## ProfilerScreen._on_model_ready

### lines 516-520

```python
from ...profiler import FittedLinear
```

A live fitted object carries its own link and applies it inside predict(); the combo would be a control that changes nothing, which is worse than no control. It is only meaningful for a model rebuilt from a coefficient table, where the link is genuinely unknown and has to be supplied.

### lines 544-546

```python
movable = [name for name in design.columns
```

Two different nothings, and the difference is the fix: a model with only an intercept was never going to be profilable, while a design whose columns never vary is a design problem the user can solve.

## ProfilerScreen.variable

### line 658

```python
def variable(self) -> str:
```

profiling

# Notes from `spacr/qt/screens/dose_response.py`

Prose lifted out of `spacr/qt/screens/dose_response.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_format](#_format) (1 entry)
- [DoseResponseScreen.__init__](#doseresponsescreen__init__) (4 entries)
- [DoseResponseScreen._on_fitted](#doseresponsescreen_on_fitted) (2 entries)
- [DoseResponseScreen._on_row_selected](#doseresponsescreen_on_row_selected) (1 entry)
- [DoseResponseScreen._draw](#doseresponsescreen_draw) (2 entries)
- [DoseResponseScreen.closeEvent](#doseresponsescreencloseevent) (1 entry)
- [Module level](#module-level) (1 entry)

## _format

### lines 119-124

```python
def _format(value) -> str:
```

SECTION NOTE, 2026-09-03: the sections were restructured to Core / Data / Tools / Assays, and SECTION_DESIGN / SECTION_EXPLORE / SECTION_RESULTS are still declared but are no longer in SECTION_ORDER. Every screen below now files under Data. The docstrings keep their original reasoning because it still says what each screen IS -- and they are published, translated API prose, so editing them invalidates reviewed translations in nine languages.

## DoseResponseScreen.__init__

### lines 266-267  _(unsure)_

```python
self._figure = Figure(figsize=(6.5, 4.6))
```

No `facecolor`: the canvas paints the page panel in its own

`paintEvent` under a transparent figure patch.

### lines 293-295

```python
mark_surface(self.table, self.report)
```

The two halves of the side splitter are the page on this screen; the curve canvas beside them paints its own panel in `paintEvent`, and these two had nothing.

### lines 304-305  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

### lines 308-310

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## DoseResponseScreen._on_fitted

### lines 481-482  _(unsure)_

```python
text = text[:NOTE_WIDTH].rstrip() + "…"
```

The refusal messages are paragraphs by design; the grid shows the first sentence and the tooltip has all of it.

### lines 488-489  _(unsure)_

```python
item.setData(Qt.UserRole, row)
```

Which fit this row is, so a sorted table still draws the curve the user clicked.

## DoseResponseScreen._on_row_selected

### lines 510-511  _(unsure)_

```python
fit = None if item is None else item.data(Qt.UserRole)
```

The fit index the row was built from, not the row number: the table sorts, and the top row is not always the first curve.

## DoseResponseScreen._draw

### line 540  _(unsure)_

```python
self._figure.patch.set_alpha(0.0)
```

`clear()` restores the rc facecolor and its alpha with it.

### lines 579-583

```python
limits = axes.get_xlim()
```

The axis belongs to the measurements. An interval on a poorly determined midpoint can span twenty decades, and letting it set the limits would shrink the actual data to a single pixel — so the range is taken before the marker is drawn and put back afterwards.

## DoseResponseScreen.closeEvent

### lines 643-644

```python
"""Stop background work and unlink before going away.
```

Abandon an in-flight fit rather than let it outlive the screen: Qt aborts the process if a running QThread is destroyed.

## Module level

### lines 661-665

```python
_ROW = declared_app(APP_KEY)
```

The row this screen puts in the registry is declared in

`spacr.qt.app_catalog`, which is what lets the app be registered without importing this module -- the launch reads the table, not the screen. These read the same row back rather than restating it, so the name, the blurb and the nine translations have one spelling and no second copy to drift from.

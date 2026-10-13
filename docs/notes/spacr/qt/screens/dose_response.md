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

## 2026-09-19 — the Curve picker

Hand-written, like the engine's note of the same date. The screen offers
four-parameter and five-parameter and chooses neither on the user's behalf,
which is the decision item 387 recorded under DELIBERATELY NOT DONE
("Offer it, do not default to it").

Picking the better-fitting model per group would have been easy and wrong:
the table would then hold EC50s from two different models in one column,
sorted against each other as though they were comparable. So the picker is
one choice for the whole fit, and the five-parameter report says in its own
caveats whether the fifth parameter earned itself on that series.

Hormesis has no control at all. It is not a mode a user selects — it is a
diagnosis the engine reaches when the low-dose end of a series departs from
control against the trend, and the screen shows it exactly where a refusal
already appears: in the Note cell, its tooltip, and the report pane.


---

# Notes from `spacr/qt/screens/dose_response.py`

Prose lifted out of `spacr/qt/screens/dose_response.py` by `tools/extract_source_notes.py`.
Ordinary comments move here; tool directives and published attribute documentation stay in the module. The path mirrors the source path, which is how its reasons are found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_fit_with_plates](#_fit_with_plates) (2 entries)
- [DoseResponseScreen.__init__](#doseresponsescreen__init__) (4 entries)

## _fit_with_plates

### lines 187-189

```python
single = _alone(fitted_frame, second_dose)
```

THE CURVES ARE THE FIRST COMPOUND ALONE when a second one is named. Combination wells are not a dose series of either agent, and fitting them into one would describe neither; the synergy surface is where they count.

### lines 196-199

```python
host = fit_frame(_alone(frame, second_dose),
```

THE HOST READOUT IS FITTED RAW, on the table as loaded. Plate normalisation scales the RESPONSE by that response's own controls; a host readout has different controls, or none, and an EC50 does not need them -- it is a concentration, not a percentage.

## DoseResponseScreen.__init__

### lines 493-496

```python
self.fit_button = QPushButton("Fit curve", self)
```

"FIT CURVE", NOT "FIT". The bare word is also zoom-to-fit in the ortho view, comparison grid and layer viewer, and a reviewed translation is keyed by its English source, so one string could never be translated right for both meanings.

### lines 505-508

```python
plates = QHBoxLayout()
```

THE PLATES ROW. Every caption on it already exists elsewhere in the application, so it adds no string a translator has not seen. Both column pickers start at "(none)", which keeps the default fit exactly the raw-response fit it always was.

### lines 534-538

```python
hosts = QHBoxLayout()
```

THE HOST READOUT. A second column from the same wells; with it, each group's host EC50 over its response EC50 is the selectivity index the number that decides whether an anti-parasitic compound is worth anything, because killing the parasite at 1 uM means nothing if the host monolayer dies at 1.2.

### lines 553-555

```python
combos = QHBoxLayout()
```

THE SECOND COMPOUND. Naming its dose column turns the table into a checkerboard: the curves use the first compound alone, and the combination wells are scored against Bliss or Loewe.

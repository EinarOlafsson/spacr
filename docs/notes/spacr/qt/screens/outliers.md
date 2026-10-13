# Notes from `spacr/qt/screens/outliers.py`

Prose lifted out of `spacr/qt/screens/outliers.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_outliers_qss](#_outliers_qss) (1 entry)
- [Module level](#module-level) (2 entries)
- [OutliersScreen.__init__](#outliersscreen__init__) (2 entries)
- [OutliersScreen._build_controls](#outliersscreen_build_controls) (2 entries)
- [OutliersScreen.load_path](#outliersscreenload_path) (2 entries)
- [OutliersScreen._fill_object_table](#outliersscreen_fill_object_table) (1 entry)
- [OutliersScreen.closeEvent](#outliersscreencloseevent) (1 entry)
- [_cell](#_cell) (1 entry)

## _outliers_qss

### lines 68-73

```python
def _outliers_qss(palette: dict, opacity=None) -> str:
```

SECTION NOTE, 2026-09-03: the sections were restructured to Core / Data / Tools / Assays, and SECTION_DESIGN / SECTION_EXPLORE / SECTION_RESULTS are still declared but are no longer in SECTION_ORDER. Every screen below now files under Data. The docstrings keep their original reasoning because it still says what each screen IS -- and they are published, translated API prose, so editing them invalidates reviewed translations in nine languages.

## Module level

### lines 93-94

```python
register_widget_qss("Outliers", _outliers_qss, replace=True)
```

`replace=True`: reachable through the screens package and by direct import, and a second import must refresh the block rather than raise.

### lines 679-683

```python
_ROW = declared_app(APP_KEY)
```

The row this screen puts in the registry is declared in

`spacr.qt.app_catalog`, which is what lets the app be registered without importing this module -- the launch reads the table, not the screen. These read the same row back rather than restating it, so the name, the blurb and the nine translations have one spelling and no second copy to drift from.

## OutliersScreen.__init__

### lines 248-249  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

### lines 252-254

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## OutliersScreen._build_controls

### lines 276-283

```python
from ..preferences import scaled_px
```

SCALED, NOT A DEVICE-PIXEL CONSTANT. This cap exists to stop the settings column eating the figure beside it, and 360 px is the right answer at 100 %% -- and only there. The glyphs inside it double at 200 %% and the box did not, which is the same defect instruction 350 already fixed on UsageBar's fixed 48 px caption column. Measured on Control Charts: the column's own sizeHint wants 586 px at 100 %%, 707 at 125 %% and 1107 at 200 %%, against a cap that stayed 330 in all three.

### lines 287-288

```python
layout.setContentsMargins(SPACING["sm"], SPACING["sm"],
```

Room for the panel's own rounded surface: the column sits ON a page surface now rather than straight on the window.

## OutliersScreen.load_path

### lines 456-457

```python
self._report_failure(
```

Same seam as a failed job, so a host that only listens for

`failed` hears about an unreadable file too.

### lines 469-470  _(unsure)_

```python
self._jobs.cancel()
```

A second load supersedes the first, so switching table twice does not deliver the frames in whatever order the reads happen to finish.

## OutliersScreen._fill_object_table

### lines 570-571

```python
order = shown[names["score"]].sort_values(
```

Sort by score, NaN last: an unscorable object is not the most interesting row on the screen, but it must not disappear either.

## OutliersScreen.closeEvent

### lines 634-636

```python
"""Stop background work and unlink before going away.
```

Abandon an in-flight read or scan rather than let it outlive the screen: Qt aborts the process if a running QThread is destroyed, and a worker that delivers into a closed widget is a use-after-free.

## _cell

### line 668, trailing  _(unsure)_

```python
if value != value:
```

NaN, without importing math

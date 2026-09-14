# Notes from `spacr/qt/screens/qc_dashboard.py`

Prose lifted out of `spacr/qt/screens/qc_dashboard.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [QCDashboardScreen.__init__](#qcdashboardscreen__init__) (3 entries)
- [QCDashboardScreen._build](#qcdashboardscreen_build) (4 entries)
- [QCDashboardScreen.refresh](#qcdashboardscreenrefresh) (1 entry)
- [_build_layer_viewer](#_build_layer_viewer) (1 entry)

## Module level

### lines 50-54

```python
_ROW = declared_app(APP_KEY)
```

The row this screen puts in the registry is declared in

`spacr.qt.app_catalog`, which is what lets the app be registered without importing this module -- the launch reads the table, not the screen. These read the same row back rather than restating it, so the name, the blurb and the nine translations have one spelling and no second copy to drift from.

### lines 150-151

```python
register_widget_qss("QCDashboard", _dashboard_qss, replace=True)
```

`replace=True`: reachable both through the screens package and by direct import, and a second import must refresh the block rather than raise.

## QCDashboardScreen.__init__

### lines 188-194

```python
self.app_key = "qc_dashboard"
```

ITS OWN REGISTRY KEY. Screens that build themselves rather than being the generic `AppScreen` had no `app_key`, and `install_folds_on` dispatches on exactly that -- so this screen could declare folds (it does, below) and never be handed them. Every other consumer of `app_key` reads it the same way the generic screen sets it, so naming it here is the screen answering a question it always could.

### lines 197-202

```python
self._jobs = JobRunner(self, threaded=threaded, app_key=APP_KEY,
```

`user_visible=False`: this runner never runs anything. It reads verdicts that are already on disk, and it now also takes the folder check and the fingerprint that used to sit inline in `refresh` -- so it fires on every visit, including the ones where nothing has changed. Visible, each of those would flash "QC - running" on Home for a read the user never started.

### lines 216-217  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

## QCDashboardScreen._build

### lines 266-268

```python
self._cards_layout.setContentsMargins(SPACING["sm"], SPACING["sm"],
```

Room for the panel's own border: the cards sit ON a surface now rather than directly on the window, and zero margins would put the first heading through the hairline.

### lines 275-277

```python
self._cards_panel.setAutoFillBackground(False)
```

setWidget() enables autoFillBackground on its child. This panel already owns a QSS surface, so leaving that flag on paints the same 30% fill twice (0.7 * 0.7 = 0.49 transmission).

### lines 280-284

```python
scroll.viewport().setAutoFillBackground(False)
```

A QScrollArea's viewport auto-fills by default, and what it fills with is the WINDOW colour -- not a surface -- so no page-opacity setting can reach it and the card column reads as an opaque slab over the animated backdrop. Same call the settings column and the sidebar make.

### lines 286-290

```python
try:
```

...and tag it, because autoFillBackground(False) does NOT stop a STYLESHEET background: QSS paints through QStyle regardless of that flag, so the blanket `QWidget { background-color: bg }` still reaches the viewport. `make_transparent` tags a scroll area's viewport along with it, which is the whole reason it takes one.

## QCDashboardScreen.refresh

### lines 406-407  _(unsure)_

```python
self._read_started = True
```

Assumed started, then corrected by `_on_read` -- which has already run by the time `submit` returns when the screen reads inline.

## _build_layer_viewer

### lines 591-593

```python
from .map_barcodes import build_registered_screen
```

IMPORTED HERE. This module used `build_registered_screen` without importing it, so every folded module it hosts raised NameError the moment its button was pressed.

# Notes from `spacr/qt/screens/graph_builder.py`

Prose lifted out of `spacr/qt/screens/graph_builder.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [GraphBuilderScreen.__init__](#graphbuilderscreen__init__) (5 entries)
- [GraphBuilderScreen.load_path](#graphbuilderscreenload_path) (2 entries)
- [GraphBuilderScreen._on_frame_loaded](#graphbuilderscreen_on_frame_loaded) (1 entry)
- [GraphBuilderScreen.closeEvent](#graphbuilderscreencloseevent) (1 entry)
- [Module level](#module-level) (1 entry)
- [_build_plate_view](#_build_plate_view) (1 entry)

## GraphBuilderScreen.__init__

### lines 179-185

```python
self.app_key = "graph_builder"
```

ITS OWN REGISTRY KEY. Screens that build themselves rather than being the generic `AppScreen` had no `app_key`, and `install_folds_on` dispatches on exactly that -- so this screen could declare folds (it does, below) and never be handed them. Every other consumer of `app_key` reads it the same way the generic screen sets it, so naming it here is the screen answering a question it always could.

### lines 189-191

```python
self._jobs = JobRunner(self, threaded=threaded, app_key="graph_builder")
```

Every table read goes through here, so it never runs on the GUI thread and always shows up in the run registry (and so in the background-activity spinner).

### lines 243-250

```python
from ..preferences import scaled_px
```

SCALED, NOT A DEVICE-PIXEL CONSTANT. This cap exists to stop the settings column eating the figure beside it, and 320 px is the right answer at 100 %% -- and only there. The glyphs inside it double at 200 %% and the box did not, which is the same defect instruction 350 already fixed on UsageBar's fixed 48 px caption column. Measured on Control Charts: the column's own sizeHint wants 586 px at 100 %%, 707 at 125 %% and 1107 at 200 %%, against a cap that stayed 330 in all three.

### lines 259-260  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

### lines 263-265

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## GraphBuilderScreen.load_path

### lines 315-318

```python
self._jobs.cancel()
```

A second load supersedes the first. Without this, switching table twice in quick succession delivers the frames in whatever order the reads happen to finish, and the picker ends up disagreeing with the panel below it.

### lines 320-322

```python
self._source.setText(
```

The table can only be named here when the caller already knew it the picker, or a drop that asked. Otherwise the worker is the first thing that can find out, so the label says it a moment later.

## GraphBuilderScreen._on_frame_loaded

### lines 362-365

```python
self._table_picker.blockSignals(True)
```

Blocked, because `addItems` moves the current index and

`currentTextChanged` is wired to `_on_table_picked` -- unblocked this populates the picker by starting another load of the table it has just loaded.

## GraphBuilderScreen.closeEvent

### lines 451-454

```python
"""Stop background work and unlink before going away.
```

Abandon an in-flight read rather than let it outlive the screen: Qt aborts the process if a running QThread is destroyed, and a worker that delivers into a closed widget is a use-after-free.

## Module level

### lines 469-473

```python
_ROW = declared_app(APP_KEY)
```

The row this screen puts in the registry is declared in

`spacr.qt.app_catalog`, which is what lets the app be registered without importing this module -- the launch reads the table, not the screen. These read the same row back rather than restating it, so the name, the blurb and the nine translations have one spelling and no second copy to drift from.

## _build_plate_view

### lines 539-542

```python
from .map_barcodes import build_registered_screen
```

IMPORTED HERE, like `install_fold_strip` below. This module was calling `build_registered_screen` without importing it at all, so both folded modules raised NameError the moment their button was pressed -- reported from the Measure console.

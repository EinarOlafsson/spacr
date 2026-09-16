# Notes from `spacr/qt/widgets/feature_dictionary.py`

Prose lifted out of `spacr/qt/widgets/feature_dictionary.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_doc_html](#_doc_html) (1 entry)
- [FeatureDictionaryPanel.set_query](#featuredictionarypanelset_query) (1 entry)
- [FeatureDictionaryPanel.show_column](#featuredictionarypanelshow_column) (1 entry)
- [FeatureDictionaryPanel._select_key](#featuredictionarypanel_select_key) (1 entry)
- [FeatureDictionaryPanel._on_row_changed](#featuredictionarypanel_on_row_changed) (1 entry)
- [make_screen](#make_screen) (1 entry)
- [register](#register) (1 entry)
- [_find_menu](#_find_menu) (1 entry)
- [install_help_action](#install_help_action) (2 entries)
- [column_name_at](#column_name_at) (1 entry)
- [Module level](#module-level) (1 entry)
- [FeatureHelpFilter.eventFilter](#featurehelpfiltereventfilter) (5 entries)

## _doc_html

### lines 197-201

```python
if (entry is not None and entry.object_type and doc.family != "meta"
```

Only for a genuine per-object-table feature. `cell_before_filtration` carries a `cell_` prefix and lives in pivoted_counts, and an `organelle_summary_*` column lives in `<parent>_organelle_summary` — for either of them "the cell table" would be a confident lie, and each says where it really lives in its own note.

## FeatureDictionaryPanel.set_query

### lines 348-350

```python
self._column = None
```

A search is a new question.  In particular, do not leave a concrete column supplied by ``show_column`` pinned while the result list is replaced underneath it.

## FeatureDictionaryPanel.show_column

### lines 374-375

```python
self._concept.setCurrentIndex(0)
```

A column name is a specific question; the filters would only narrow it away.

## FeatureDictionaryPanel._select_key

### lines 408-409

```python
doc = doc_for(key)
```

A concrete column always resolves even when the free-text search would not have surfaced its feature; show it anyway.

## FeatureDictionaryPanel._on_row_changed

### lines 463-464  _(unsure)_

```python
entry = None
```

Once the user moves off the column they asked about, stop pinning the detail pane to it.

## make_screen

### lines 569-571  _(unsure)_

```python
def make_screen(host=None) -> QWidget:
```

hook 1 — the app registry and the theme, through their seams

## register

### lines 624-625

```python
LOG.exception("Could not register the Feature Dictionary app")
```

A registry that cannot take one more app is not a reason for the GUI to refuse to start; the Help menu route still works.

## _find_menu

### lines 637-639  _(unsure)_

```python
def _find_menu(window: QMainWindow, title: str) -> Optional[QMenu]:
```

hook 2 — Help menu + "What is this?" on every results table

## install_help_action

### lines 683-684

```python
from ..menus import set_menu_role
```

Explicit, so Qt cannot relocate it on macOS by liking its text. See spacr.qt.menus for what that costs when it happens.

### lines 692-693

```python
before = None
```

Above the separator that precedes "Check for updates…", so it sits with the other "explain something" entries rather than with the tools.

## column_name_at

### lines 728-731

```python
viewport = view.viewport()
```

QAbstractItemView.indexAt consumes *viewport* coordinates. Context-menu events delivered to the view itself use view coordinates, which include the row-header/frame offset and can therefore select a neighbouring column near a boundary.

## Module level

### lines 819-826

```python
try:
```

RESOLVED ONCE, NOT PER EVENT. `_still_alive` is called twice for every event in the application -- 323,014 times while a single Regression screen is built -- because `FeatureHelpFilter` is installed on the QApplication and has to check liveness BEFORE it may touch `event.type()`. With the import inside the function that is 323,014 executions of an import statement: 625 ms per screen build, against 33 ms resolved once. Measured, not assumed. The fallback stays a module-level None so the "cannot ask the question" branch below behaves exactly as it did.

## FeatureHelpFilter.eventFilter

### lines 859-886

```python
"""Watch the widgets this filter is installed on.
```

THIS RUNS FOR EVERY EVENT IN THE APPLICATION, and it segfaulted.

Captured by the crash dump on 2026-08-19, milliseconds after a regression closed [success] while that run's figure widgets were being torn down:

Fatal Python error: Segmentation fault

Current thread (most recent call first): feature_dictionary.py, line 776 in eventFilter app.py, line 3110 in launch

The frame EXISTS at the `def` line, so PySide had already built both wrappers and the crash is on the first bytecode -- `event.type()`, reading a QEvent whose C++ half was already freed. An application-wide filter is handed every event in the process, including ones whose receiver is being destroyed at that instant.

THE FILTER STAYS APPLICATION-WIDE. Installing it on the item views instead was tried and REVERTED: the feature deliberately answers a right-click on an arbitrary child widget INSIDE a cell by walking back to the table behind it, and a per-view install cannot see those events. Three tests name that case.

So this is a seatbelt, and its limit is worth stating: `isValid` reports a wrapper whose deletion shiboken was TOLD about. It turns that case from a segfault into a no-op. A C++ object freed without shiboken being told still dereferences, and the fix for that is wherever the object is being freed, not here.

### line 897, trailing  _(unsure)_

```python
except (RuntimeError, ReferenceError):
```

died between the two checks

### lines 906-908

```python
view = obj if isinstance(obj, QAbstractItemView) else obj.parent()
```

An ignored event from the vertical header propagates to the table.  Its second delivery must remain a row-header gesture, rather than being reinterpreted as a click on column zero.

### lines 923-924

```python
return False
```

Somebody else's grid: not a measurements table, so an item about spaCR features would be noise on it.

### lines 936-937

```python
LOG.debug("Feature help context menu failed", exc_info=True)
```

A context menu is help, not function: never let it take a right-click (or the app) down with it.

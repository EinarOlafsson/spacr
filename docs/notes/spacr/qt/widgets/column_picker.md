# Notes from `spacr/qt/widgets/column_picker.py`

Prose lifted out of `spacr/qt/widgets/column_picker.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [resolve_db_path](#resolve_db_path) (1 entry)
- [SchemaReader.__init__](#schemareader__init__) (1 entry)
- [open_reader](#open_reader) (1 entry)
- [near_miss](#near_miss) (1 entry)
- [read_table](#read_table) (1 entry)
- [Module level](#module-level) (5 entries)
- [ColumnPickerDialog.__init__](#columnpickerdialog__init__) (1 entry)
- [ColumnPickerDialog._build_ui](#columnpickerdialog_build_ui) (2 entries)
- [ColumnPickerDialog._apply_schema](#columnpickerdialog_apply_schema) (1 entry)
- [ColumnPickerDialog._on_table_changed](#columnpickerdialog_on_table_changed) (1 entry)
- [ColumnPickerDialog._evaluate](#columnpickerdialog_evaluate) (1 entry)
- [ColumnPickerDialog.chosen_columns](#columnpickerdialogchosen_columns) (1 entry)
- [ColumnPickerDialog.select_columns](#columnpickerdialogselect_columns) (1 entry)
- [ColumnPickerDialog.accept](#columnpickerdialogaccept) (1 entry)
- [ColumnPickerButton](#columnpickerbutton) (1 entry)
- [set_field_values](#set_field_values) (1 entry)
- [_replace_layout_widget](#_replace_layout_widget) (1 entry)
- [attach_column_picker](#attach_column_picker) (1 entry)

## resolve_db_path

### lines 131-133  _(unsure)_

```python
def resolve_db_path(path: str) -> str:
```

Path resolution + read-only schema access

## SchemaReader.__init__

### lines 195-200

```python
self._log_lock = threading.Lock()
```

`executed` is appended to on whichever thread runs the query and read from the GUI thread by `ColumnPickerDialog.executed_sql`. list.append is atomic under the GIL, but the *snapshot* the assertion hook hands out must not be taken mid-append, or a test that asserts "opening cost no COUNT(*)" could be reading a list that is one element ahead of the statement it is about.

## open_reader

### lines 350-352

```python
return None, f"Cannot open {os.path.basename(resolved)}: {exc}"
```

"unable to open database file" — a permission problem, a stale network mount. Saying "not a database" here would send the user looking for the wrong fault.

## near_miss

### lines 468-470

```python
lower = text.lower()
```

get_close_matches compares whole strings, so a long shared prefix with a short tail ('annotate' vs 'annotate_pass_two_of_the_second_batch') scores below the cutoff. Those are near-misses too.

## read_table

### lines 479-481

```python
def read_table(reader: Optional[SchemaReader],
```

Worker-safe reads — everything opening the dialog costs, off the GUI thread

## Module level

### line 552, trailing  _(unsure)_

```python
ACTION_USE = "use"
```

the name is an existing column

### line 553, trailing  _(unsure)_

```python
ACTION_CREATE = "create"
```

the name is new and safe

### line 554, trailing  _(unsure)_

```python
ACTION_CONFIRM = "confirm"
```

the name is new but looks like a typo

### line 555, trailing

```python
ACTION_INVALID = "invalid"
```

SQLite would refuse the name

### line 556, trailing  _(unsure)_

```python
ACTION_UNCHECKED = "unchecked"
```

no database to check against

## ColumnPickerDialog.__init__

### lines 653-655

```python
self._jobs.submit(
```

Unthreaded, `submit` calls its job inline and `_apply_schema` has run by the time this returns — which is the whole of the default mode's contract.

## ColumnPickerDialog._build_ui

### lines 679-680

```python
self._source.setText("Reading the schema…" if self._threaded
```

Threaded, nothing has been opened yet and saying "No database open." would be a lie the user reads for the whole of the load.

### lines 719-720  _(unsure)_

```python
self._column_tree.itemSelectionChanged.connect(self._evaluate)
```

The selection, not the name box, is the answer in this mode, so the verdict has to follow it.

## ColumnPickerDialog._apply_schema

### lines 807-812

```python
blocked = self._tables.blockSignals(True)
```

The columns for `wanted` are already in `payload` — the worker read them in the same trip. Letting `setCurrentRow` fire `currentTextChanged` would send `_on_table_changed` off to read them a second time, and in threaded mode that is a second job racing the one that just landed. Select it quietly and paint from what we were handed.

## ColumnPickerDialog._on_table_changed

### lines 827-828

```python
self._jobs.cancel()
```

Supersede: clicking down the table list must not leave the tree showing whichever read happened to finish last.

## ColumnPickerDialog._evaluate

### lines 959-962

```python
picked = self._selected_column_names()
```

Multi-select: once more than one row is highlighted the name box is no longer the answer, so judging the name would report on one column out of several. Every selected row is an existing column by construction, so the verdict is simply "use them".

## ColumnPickerDialog.chosen_columns

### lines 1116-1117

```python
picked = picked + [typed]
```

Only one row selected: the tree filled the name box from it, so `typed` IS that row and appending it would not add anything.

## ColumnPickerDialog.select_columns

### lines 1135-1136

```python
tree.setCurrentItem(last, 0, QItemSelectionModel.NoUpdate)
```

setCurrentItem would clear the selection we just made; the current item only matters for the name box and Count non-null.

## ColumnPickerDialog.accept

### lines 1209-1210

```python
"""Take the chosen columns and close."""
```

Belt and braces: the button is already disabled in these states, but Enter in the name box would otherwise bypass it.

## ColumnPickerButton

### lines 1230-1232  _(unsure)_

```python
class ColumnPickerButton(QToolButton):
```

The button + the one-line attachment

## set_field_values

### lines 1461-1462

```python
ok = set_field_text(field, wanted[0], append=False)
```

Replacing a single-valued field with several names would lose all but one silently; keep them all, in the field's own style.

## _replace_layout_widget

### lines 1558-1561

```python
trailing = []
```

Python QLayout subclasses do not expose Qt's protected replaceAt(), so replaceWidget() cannot update them. Rebuild only the suffix around the occupied slot; retaining the QLayoutItems preserves their widgets and their reading order instead of appending the replacement at the end.

## attach_column_picker

### lines 1640-1645

```python
return button
```

Visibility is deliberately left alone. Qt's own reparent-into-layout path handles it: QWidget::setParent clears WA_WState_ExplicitShowHide when it hides an already-shown widget, so the field reappears with the wrapper, and QLayout::addChildWidget queues a show for a wrapper added under a parent that is visible right now. Forcing visibility here would set the explicit-show flag and break a collapsed Section's expand.

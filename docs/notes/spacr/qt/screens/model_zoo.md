# Notes from `spacr/qt/screens/model_zoo.py`

Prose lifted out of `spacr/qt/screens/model_zoo.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [ModelZooScreen.__init__](#modelzooscreen__init__) (3 entries)
- [ModelZooScreen._build_ui](#modelzooscreen_build_ui) (5 entries)
- [ModelZooScreen.set_entries](#modelzooscreenset_entries) (3 entries)
- [ModelZooScreen._apply_download](#modelzooscreen_apply_download) (1 entry)
- [ModelZooScreen.set_segment_fn](#modelzooscreenset_segment_fn) (1 entry)
- [ModelZooScreen._apply_benchmark](#modelzooscreen_apply_benchmark) (1 entry)
- [ModelZooScreen._run_job](#modelzooscreen_run_job) (1 entry)

## Module level

### lines 196-197

```python
register_widget_qss(ZOO_QSS_NAME, _model_zoo_qss, replace=True)
```

`replace=True`: reachable through the screens package and by direct import, and a second import must refresh the block rather than raise.

## ModelZooScreen.__init__

### lines 324-325  _(unsure)_

```python
self._jobs: List[tuple] = []
```

Strong references to in-flight (QThread, worker) pairs: a QThread garbage-collected while still running takes the process down.

### lines 331-334

```python
ensure_widget_qss_applied(ZOO_QSS_NAME, root=self)
```

`app.py` imports this module inside the branch that builds the screen, long after the launch stylesheet was generated, so the block registered above is not in the sheet that is live and every container opens bare. See `ensure_widget_qss_applied`.

### lines 349-351

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## ModelZooScreen._build_ui

### line 394  _(unsure)_

```python
self._table = QTableWidget(0, len(_ZOO_HEADERS), self)
```

── the listing ───────────────────────────────────────────────

### lines 397-398

```python
self._table.setObjectName(TABLE_NAME)
```

The listing IS the container here — nothing is under it — so it keeps a surface rather than showing the page through.

### line 412  _(unsure)_

```python
self._detail = QPlainTextEdit(self)
```

── provenance card ───────────────────────────────────────────

### line 463

```python
test = QGroupBox("Test on fields", self)
```

── benchmark ─────────────────────────────────────────────────

### lines 512-515

```python
self._preview.setObjectName(PREVIEW_NAME)
```

No inline stylesheet: it used to be

`background: {active_palette()["bg"]}`, raw hex and the window colour, opaque by construction. The panel is a rule now, reached by this name.

## ModelZooScreen.set_entries

### lines 579-580  _(unsure)_

```python
item = _cell(str(text),
```

The size column is printed in whichever unit reads best, so it sorts on the byte count behind it.

### lines 584-585  _(unsure)_

```python
item.setData(Qt.UserRole, r)
```

Which model the row stands for. The table sorts, so a row number names nothing outside the moment it was read.

### lines 588-590

```python
item.setForeground(_brush(active_palette()["warning"]))
```

Never blank, and never quiet: 'unknown' in the warning colour, because a model with no provenance is the one you are most likely to misapply.

## ModelZooScreen._apply_download

### lines 793-795

```python
listing = [e for e in self._entries
```

The catalogue row this came from is replaced by the local file, not listed beside it: one model, one row, and the row now points at bytes that are here.

## ModelZooScreen.set_segment_fn

### line 856

```python
def set_segment_fn(self, fn: Optional[Callable]) -> None:
```

benchmark

## ModelZooScreen._apply_benchmark

### lines 1003-1004

```python
self._summary.setText(
```

The field set is named in the summary on purpose: this number is only meaningful next to another number from the same three fields.

## ModelZooScreen._run_job

### lines 1173-1175

```python
self._jobs.append((thread, worker))
```

Strong references: PySide6 will not keep the worker alive through the started→run connection alone, and a QThread garbage-collected while still running takes the process down with it.

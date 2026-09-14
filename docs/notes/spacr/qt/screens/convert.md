# Notes from `spacr/qt/screens/convert.py`

Prose lifted out of `spacr/qt/screens/convert.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [PlanTableModel.data](#plantablemodeldata) (1 entry)
- [ConvertScreen._build_ui](#convertscreen_build_ui) (5 entries)
- [ConvertScreen._run_job](#convertscreen_run_job) (1 entry)

## PlanTableModel.data

### lines 202-203

```python
return os.path.basename(str(value))
```

The full path is the tooltip; the cell shows enough to recognise the file without a 200-pixel column of prefix.

## ConvertScreen._build_ui

### line 293  _(unsure)_

```python
src_row = QHBoxLayout()
```

── Source row ────────────────────────────────────────────────

### line 337  _(unsure)_

```python
dst_row = QHBoxLayout()
```

── Destination row ───────────────────────────────────────────

### line 358  _(unsure)_

```python
self._model = PlanTableModel(self)
```

── Preview table ─────────────────────────────────────────────

### lines 362-363  _(unsure)_

```python
install_sorting(self._table)
```

After setModel: the helper puts a sorting proxy over the plan model, which replaces the view's model and its selection model.

### line 373  _(unsure)_

```python
self._summary = QPlainTextEdit(self)
```

── Summary + progress + status ───────────────────────────────

## ConvertScreen._run_job

### lines 720-722

```python
self._jobs.append((thread, worker))
```

Strong references: PySide6 will not keep the worker alive through the started→run connection alone, and a QThread garbage-collected while still running takes the process down with it.

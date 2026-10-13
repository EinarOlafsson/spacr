# Notes from `spacr/qt/screens/foreign.py`

Prose lifted out of `spacr/qt/screens/foreign.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ForeignScreen._build_ui](#foreignscreen_build_ui) (5 entries)
- [ForeignScreen.preview](#foreignscreenpreview) (1 entry)
- [ForeignScreen._run_job](#foreignscreen_run_job) (1 entry)

## ForeignScreen._build_ui

### lines 375-378

```python
"""Lay out the input rows over the mapping table and the report."""
```

ITS OWN REGISTRY KEY -- what `install_folds_on` dispatches on. A screen that builds itself has no `app_key` unless it says so, and without it this screen could declare folds and never be handed them.

### lines 386-390

```python
header = ModuleHeader(
```

A ModuleHeader RATHER THAN A BARE LABEL, for the same reason

Database Browser grew one: it draws the same `DisplayHeading` this built by hand, and its `add_trailing` is where the fold strip hangs. Without a masthead, Format Converter and External Masks have nowhere to appear.

### line 455  _(unsure)_

```python
table_row = QHBoxLayout()
```

── Measurement table ─────────────────────────────────────────

### lines 495-496  _(unsure)_

```python
install_sorting(self._table)
```

After setModel: the helper puts a sorting proxy over the mapping model, which replaces the view's model and its selection model.

### line 515  _(unsure)_

```python
dst_row = QHBoxLayout()
```

── Destination + actions ─────────────────────────────────────

## ForeignScreen.preview

### lines 819-822

```python
"Add at least one mask folder. A spaCR project without "
```

TWO SENTENCES, NOT ONE EM-DASH CLAUSE. The Hindi checkpoint was the only one of the nine that would not take this line, and the repo's remedy for that is to shorten and split rather than to accept an English row.

## ForeignScreen._run_job

### lines 1090-1093

```python
self._jobs.append((thread, worker))
```

Strong references: PySide6 will not keep the worker alive through the started→run connection alone, and a QThread garbage-collected while still running takes the process down with it. The worker is deliberately never deleteLater'd — see bridge.make_thread.

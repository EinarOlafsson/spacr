# Notes from `spacr/qt/screens/queue.py`

Prose lifted out of `spacr/qt/screens/queue.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_QueueRunner.run](#_queuerunnerrun) (1 entry)
- [QueueScreen](#queuescreen) (1 entry)
- [QueueScreen.__init__](#queuescreen__init__) (1 entry)
- [QueueScreen._build_ui](#queuescreen_build_ui) (2 entries)
- [QueueScreen._refresh_elapsed_only](#queuescreen_refresh_elapsed_only) (1 entry)

## _QueueRunner.run

### lines 92-101

```python
"""Run each queued plate in turn until the queue empties or is stopped.
```

EVERY EMIT GOES THROUGH `emit_safely`. A queue run outlives the screen that started it -- closing the window mid-run leaves this thread emitting at a destroyed C++ object, which raises `RuntimeError: Internal C++ object already deleted` out of a QThread::run override, and an exception out of a virtual override aborts the process rather than failing the run.

The database updates stay unguarded on purpose: a queue item that finished must be recorded as finished whether or not anyone is watching, and sqlite does not care that the window closed.

## QueueScreen

### lines 165-166  _(unsure)_

```python
queue_size_changed = Signal(int)
```

Emitted whenever the queue changes size (add / remove / clear). MainWindow can use this to update the Home-tile badge count.

## QueueScreen.__init__

### line 185  _(unsure)_

```python
self._tick = QTimer(self)
```

Poll for elapsed-time updates while the runner is going

## QueueScreen._build_ui

### line 215  _(unsure)_

```python
bar = QHBoxLayout()
```

Toolbar

### lines 233-234  _(unsure)_

```python
self._table = QTableWidget(self)
```

`_btn_add` isn't wired here — MainWindow connects it to the active app screen's settings snapshot. See wire_add_current.

## QueueScreen._refresh_elapsed_only

### lines 422-423

```python
"""Tick the elapsed column of the running plates, and only that column.
```

Only touch the elapsed column so we don't churn the whole table (and lose selection state) every second.

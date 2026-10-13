# Notes from `spacr/qt/screens/report.py`

Prose lifted out of `spacr/qt/screens/report.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ReportScreen.__init__](#reportscreen__init__) (1 entry)
- [ReportScreen._build_ui](#reportscreen_build_ui) (3 entries)
- [ReportScreen._run_job](#reportscreen_run_job) (2 entries)

## ReportScreen.__init__

### lines 126-127  _(unsure)_

```python
self._jobs: List[tuple] = []
```

Ownership list for in-flight (QThread, worker) pairs — a QThread collected while still running takes the process down with it.

## ReportScreen._build_ui

### line 169  _(unsure)_

```python
src_row = QHBoxLayout()
```

── Source row ────────────────────────────────────────────────

### line 185  _(unsure)_

```python
self._verdict = QLabel("", self)
```

── Overall verdict ───────────────────────────────────────────

### line 197  _(unsure)_

```python
opts = QHBoxLayout()
```

── Options ───────────────────────────────────────────────────

## ReportScreen._run_job

### lines 504-506

```python
self._jobs.append((thread, worker))
```

Strong references: PySide6 will not keep the worker alive through the started→run connection alone, and a QThread garbage-collected while still running takes the process down with it.

### lines 512-524

```python
thread.finished.connect(self._on_thread_finished)
```

A BOUND METHOD, not a closure — the rule `make_thread` states and then relies on for its own `handle.retire`. `thread.finished` is emitted in the worker thread, so PySide6 queues the call to the RECEIVER's thread; with a closure the receiver is the QThread itself, and `make_thread` connects `thread.finished -> thread.deleteLater` FIRST. Slots run in connection order, so the DeferredDelete for the QThread is posted ahead of the closure's metacall — and Qt discards queued events for a destroyed receiver. The job was then never retired, `active_jobs()` never returned to zero, and every `waitUntil(active_jobs() == 0)` sat there until it timed out with the QThread's C++ half already gone. Binding to the widget makes the widget the receiver, so the call survives the thread it is reporting on.

# Notes from `spacr/qt/screens/batch.py`

Prose lifted out of `spacr/qt/screens/batch.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [BatchScreen.__init__](#batchscreen__init__) (3 entries)
- [BatchScreen._build_ui](#batchscreen_build_ui) (4 entries)
- [BatchScreen.run](#batchscreenrun) (1 entry)
- [BatchScreen._relay_progress](#batchscreen_relay_progress) (1 entry)
- [BatchScreen._on_progress](#batchscreen_on_progress) (1 entry)
- [BatchScreen._on_queue_settled](#batchscreen_on_queue_settled) (1 entry)
- [BatchScreen._refresh_table](#batchscreen_refresh_table) (2 entries)
- [BatchScreen._load_log](#batchscreen_load_log) (1 entry)

## BatchScreen.__init__

### lines 146-149

```python
self._progress_relayed.connect(
```

This signal is emitted by ``run_queue`` on its worker thread. Auto connections between two signals owned by the same QWidget can be treated as direct by PySide6 even when ``emit`` occurs elsewhere, so spell out the GUI-thread hop.

### lines 162-163  _(unsure)_

```python
self._tick = QTimer(self)
```

Elapsed time for the running job. Cheap, and only ever touches one cell, so it does not churn the table or the selection.

### lines 168-170

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## BatchScreen._build_ui

### line 203  _(unsure)_

```python
edit = QHBoxLayout()
```

── Job editor ────────────────────────────────────────────────

### line 245  _(unsure)_

```python
bar = QHBoxLayout()
```

── Queue toolbar ─────────────────────────────────────────────

### line 271  _(unsure)_

```python
run_row = QHBoxLayout()
```

── Run controls ──────────────────────────────────────────────

### line 304  _(unsure)_

```python
split = QSplitter(Qt.Vertical, self)
```

── Table + panes ─────────────────────────────────────────────

## BatchScreen.run

### lines 594-596

```python
self._jobs.append((thread, worker))
```

Strong references: PySide6 will not keep the worker alive through the started→run connection alone, and a QThread garbage-collected while still running takes the whole process down with it.

## BatchScreen._relay_progress

### line 622

```python
def _relay_progress(self, progress: "bt.Progress") -> None:
```

progress, on the worker thread

## BatchScreen._on_progress

### lines 656-658

```python
selected = self.selected_job()
```

Keep the log pane live: the selected job's file is being written right now, and a pane that only updates on selection would show the previous job's output for the next seven hours.

## BatchScreen._on_queue_settled

### line 663

```python
def _on_queue_settled(self, ok: bool) -> None:
```

completion, back on the GUI thread

## BatchScreen._refresh_table

### lines 818-819

```python
key = (job.elapsed_s if col == COLUMNS.index("Time")
```

A duration reads as "2m 05s" and sorts on the seconds behind it, or "2m" would land under "9s".

### lines 834-835  _(unsure)_

```python
item.setData(Qt.UserRole, job.id)
```

The row's identity, so a sorted table still knows which job the user picked.

## BatchScreen._load_log

### lines 887-889

```python
text = text.replace("\r\n", "\n").replace("\r", "\n")
```

Match text mode's universal-newline decoding. In particular, QPlainTextEdit returns LF, so an unchanged Windows log must not look different and reset the scroll position each tick.

# Notes from `spacr/qt/verbose_logger.py`

Prose lifted out of `spacr/qt/verbose_logger.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [log_dir](#log_dir) (1 entry)
- [_ensure_file_handler](#_ensure_file_handler) (4 entries)
- [_ConsoleRelay._deliver](#_consolerelay_deliver) (4 entries)
- [_ConsoleForwarder.emit](#_consoleforwarderemit) (2 entries)
- [apply_console_levels](#apply_console_levels) (2 entries)
- [apply_verbose_logging](#apply_verbose_logging) (2 entries)

## log_dir

### lines 153-155

```python
def log_dir() -> Path:
```

Log file — always on, so a crash/hang can be diagnosed after the fact

## _ensure_file_handler

### lines 189-203

```python
if _file_handler not in logging.getLogger(_SINK_LOGGER).handlers:
```

ALREADY BUILT IS NOT ALREADY ATTACHED. This returned here unconditionally, and the console sink does not: `_ensure_handler` continues past its own `is None` block and re-adds itself when it is missing from the sink logger. So the two functions promised the same thing -- "attach a handler at the package root, idempotent" and only one of them kept it if anything had detached the handler since.

Nothing in the application detaches it, which is why this never showed in a run; a test that restores `spacr`'s handler list does, and `test_it_attaches_at_the_package_root` then failed depending on which sibling ran first (seeds 3-7 of eight, and not 1, 2 or 8).

Re-attaching costs one list membership test and makes the pair consistent.

### lines 215-216

```python
return None                                                       # type: ignore[return-value]
```

If we can't open the file, don't crash — just skip file logging so the app still runs.

### lines 218-221

```python
from ..logging_util import _CompactTraceFormat
```

COMPACT FOR THE TRACE, ordinary for everything else. See

`_CompactTraceFormat`: on a trace line the level is always DEBUG, the logger is always `spacr.trace`, and the date is the same on every line of one run -- so the prefix was longer than the message it introduced.

### line 234  _(unsure)_

```python
logger.setLevel(min(logger.level or logging.INFO, logging.INFO))
```

Ensure records propagate to the root logger's format if any.

## _ConsoleRelay._deliver

### lines 274-278

```python
if console_write_in_progress():
```

Records produced *by* a console write are dropped, not queued: see :data:`_DELIVERY_STATE`. Re-entering the widget mid-``setPlainText`` destroys its QTextDocument's frames twice and takes the process down. Dropping them loses nothing a user could act on — they are the trace of the console drawing the previous line.

### lines 284-286

```python
if not isValid(target):
```

A weak reference only detects Python collection. PySide can keep the wrapper alive after Qt has deleted the underlying C++ QWidget; calling a method on that zombie can segfault rather than raise RuntimeError.

### lines 293-295

```python
try:
```

The latch is raised by the *innermost* writer — ConsolePanel's own append methods — and only checked here. Raising it around this call instead would make the panel refuse the very delivery it was handed.

### line 299

```python
pass
```

Never let a logging failure escape into the app.

## _ConsoleForwarder.emit

### lines 323-326

```python
"""Forward one record to the console, unless that would recurse.
```

Cut the feedback loop as early as possible: a record emitted while this thread is inside a console write is a record *about* that write. Formatting and emitting it would run more spaCR code, which under the function-trace profile hook produces more records still.

### line 347

```python
pass
```

Never let a logging failure escape into the app.

## apply_console_levels

### line 462

```python
handler.setLevel(logging.DEBUG)
```

The handler's own threshold would veto the filter before it ran.

### lines 464-465  _(unsure)_

```python
lowest = min(wanted) if wanted else logging.CRITICAL
```

The loggers feeding it carry thresholds too; open them to the lowest level any sink wants. The file filters still decide what is written.

## apply_verbose_logging

### lines 500-504

```python
if on:
```

Keep the interpreter-wide profiler an explicit developer tool. Even a filtered profile hook runs for every Python call in every thread, so it cannot be part of an always-on GUI preference. Decorated entry points, button presses, and ordinary DEBUG records still provide the useful verbose trail without imposing that process-wide cost.

### lines 506-512

```python
logging.getLogger("cellpose").setLevel(logging.INFO)
```

Nudge cellpose's own logger to INFO so its "loaded model X" breadcrumbs come through. We deliberately DO NOT touch torch/PIL/matplotlib: torch's built-in handler writes to a stream that pytest captures + closes, and dialling that logger up produces spurious "I/O operation on closed file" noise. Users can raise those loggers manually if they need to.

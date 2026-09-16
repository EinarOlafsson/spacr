# Notes from `spacr/logging_util.py`

Prose lifted out of `spacr/logging_util.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (6 entries)
- [setup_logging](#setup_logging) (4 entries)
- [apply_level_policy](#apply_level_policy) (2 entries)
- [get_logger](#get_logger) (1 entry)
- [_trace_profile](#_trace_profile) (1 entry)
- [_trace_one_event](#_trace_one_event) (2 entries)
- [time_module](#time_module) (1 entry)

## Module level

### line 54, trailing

```python
MAX_BYTES = 5 * 1024 * 1024
```

5 MB per file

### line 55, trailing

```python
BACKUP_COUNT = 3
```

→ up to ~20 MB total

### lines 107-112

```python
"fontTools",
```

fontTools.subset logs about FORTY lines for every figure saved -- each glyph name and glyph ID, twice, for MATH then GSUB then glyf, followed by a line per font table. A regression run saves a dozen figures, so thousands of lines of glyph inventory bury the run's own output and the user cannot see what happened. "matplotlib" does not cover this: fontTools is a separate top-level logger that matplotlib calls into.

### line 222  _(unsure)_

```python
_INITIALISED: bool = False
```

Module-level bookkeeping — set once by setup_logging().

### lines 229-233

```python
_TRACE_ROOT = os.path.realpath(os.path.dirname(__file__)) + os.sep
```

Function-level DEBUG tracing is opt-in.  A profile hook is used instead of decorating thousands of functions: it also covers private helpers, class methods and functions imported after the preference is enabled.  The hook filters by filename before touching logging, so third-party calls are only a couple of string comparisons and normal (non-debug) operation pays nothing.

### lines 669-689

```python
import functools
```

Timing utilities — for benchmarking

Two shapes users can adopt as they need:

``@timed`` decorator — wraps one function; on every call, logs the elapsed wall-clock at INFO. Skips logs faster than SPACR_TIME_THRESHOLD_MS (default 5 ms) so tight inner loops don't drown the log.

:class:`Timer` context manager — same idea for arbitrary blocks::

with Timer("cellpose batch"):

model.eval(...)     # "cellpose batch took 2.34s"

:func:`time_module` — one call to wrap every public function on a module with ``@timed``. Use it during ad-hoc profiling; don't leave it on in production code.

All three no-op cheaply when :func:`disable_timing` has been called (default: enabled, since the threshold already filters noise).

## setup_logging

### line 362, trailing  _(unsure)_

```python
root.setLevel(logging.DEBUG)
```

let each handler cap its own view

### lines 363-364  _(unsure)_

```python
logging.getLogger("spacr").setLevel(level)
```

spacr.* explicitly follows the requested level so enable_debug is the only way records below `level` reach the handlers.

### lines 375-377

```python
sys.stderr.write(
```

Logging is diagnostic infrastructure; a read-only home directory must not prevent analysis from starting. Keep failures visible on stderr when the requested file cannot be opened.

### lines 383-385

```python
file_h.setLevel(logging.DEBUG)
```

The handler passes everything and the filter decides, so the set of enabled levels can be changed at runtime without rebuilding the handler underneath whatever thread is logging.

## apply_level_policy

### lines 475-476

```python
lowest = min(files) if files else logging.CRITICAL
```

spacr.* carries its own threshold, which would veto the switches before any handler filter ran. Open it to the lowest level asked for.

### line 483, trailing  _(unsure)_

```python
except Exception:
```

Qt is optional; the CLI has no console panel.

## get_logger

### lines 498-500  _(unsure)_

```python
def get_logger(name: str) -> logging.Logger:
```

Convenience API for modules and interactive sessions

## _trace_profile

### lines 573-583

```python
try:
```

AT INTERPRETER SHUTDOWN the hook is still installed while everything it depends on is being torn down -- this module's globals are set to None, and so are `logging`'s own, so even `logging.getLogger` fails from inside a finaliser (`_removeHandlerRef` is one). Guarding this module alone was not enough; the whole body is guarded, because a tracing aid must never alter, or comment on, the code it observes.

A bare `except` is right here and almost nowhere else: there is no caller to report to -- Python prints "Exception ignored in" and carries on -- and the only alternative is noise in every process that ever enabled verbose logging.

## _trace_one_event

### lines 596-600

```python
if not logging.getLogger("spacr.trace").isEnabledFor(logging.DEBUG):
```

BEFORE ANY OF THE WORK BELOW. `realpath` is a syscall per event, and this hook runs on every call and every return in the process -- so a trace that is not going to be written must cost as little as possible to decide against. Asking the logger first turns the whole hook into one dictionary lookup while verbose is off.

### line 619

```python
pass
```

A tracing aid must never alter the code it observes.

## time_module

### line 876  _(unsure)_

```python
if getattr(obj, "__module__", None) != module.__name__:
```

Only wrap functions actually defined IN the module

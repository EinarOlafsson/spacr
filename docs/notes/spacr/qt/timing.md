# Notes from `spacr/qt/timing.py`

Prose lifted out of `spacr/qt/timing.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_stall_duration_ms](#_stall_duration_ms) (1 entry)
- [_the_spacr_frame](#_the_spacr_frame) (1 entry)
- [_ImportTimer.find_spec](#_importtimerfind_spec) (1 entry)
- [watch_interactive._InteractivePaintProbe.event_loop_started](#watch_interactive_interactivepaintprobeevent_loop_started) (2 entries)
- [watch_interactive._InteractivePaintProbe._settle](#watch_interactive_interactivepaintprobe_settle) (2 entries)
- [_peak_rss_mb](#_peak_rss_mb) (2 entries)
- [snapshot](#snapshot) (1 entry)

## _stall_duration_ms

### lines 138-140

```python
try:
```

Compatibility for an in-memory diagnostic row made by an older caller.  Production watchdog rows always carry both timestamps and the release artifact validator rejects a row without them.

## _the_spacr_frame

### line 279  _(unsure)_

```python
return ""
```

The environment is merely NAMED spacr.

## _ImportTimer.find_spec

### lines 340-342

```python
self._pending = (fullname, started, caller)
```

Let the real finders answer; we only time how long that takes and how long the module then takes to execute, which the next call into this finder for a submodule will nest under.

## watch_interactive._InteractivePaintProbe.event_loop_started

### lines 600-606

```python
"""Discard paints that arrived before the loop began.
```

A paint may have been delivered by show() before exec().  It is evidence about the widget, but not evidence for this contract: readiness begins only after the application event loop has actually dispatched a callback.  Discard every pre-loop paint before forcing another one, or an already-painted control plus the settle timer below could report a false ready state without a post-exec paint ever being observed.

### lines 616-622

```python
QWidget.update(self.root)
```

QTableWidget and QListWidget overload ``update`` with a

QModelIndex argument. Calling the bound method with no arguments therefore raises TypeError even though the QWidget repaint overload is the one this probe needs. Invoke QWidget's implementation explicitly for every subclass so readiness instrumentation cannot break the very table/list screens it is measuring.

## watch_interactive._InteractivePaintProbe._settle

### lines 678-684

```python
if not root_usable or not painted_usable:
```

A transparent container legitimately receives no paint event of its own: Home is exactly such a widget under an ambient theme. A descendant control's completed paint is stronger evidence than forcing the root to repaint for the benchmark, and it is the state the contract asks for -- a control the user can see and operate.  Keep root_painted as factual diagnostic evidence, but do not invent a root paint where Qt optimised one away.

### lines 728-729

```python
continue
```

Instrumentation may never make navigation fail.  The benchmark controller records its own errors separately.

## _peak_rss_mb

### line 841  _(unsure)_

```python
return value / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0)
```

macOS reports bytes; Linux and the supported BSD runners report KiB.

### lines 848-850

```python
value = getattr(info, "peak_wset", None)
```

Windows exposes the process peak as peak_wset.  On a platform without that field, current RSS is explicitly the best available fallback rather than a fabricated peak.

## snapshot

### lines 931-932  _(unsure)_

```python
import platform
```

Keep disabled timing stdlib-light: platform performs several imports, and snapshots exist only for an explicitly enabled diagnostic run.

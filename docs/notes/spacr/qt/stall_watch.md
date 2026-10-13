# Notes from `spacr/qt/stall_watch.py`

Prose lifted out of `spacr/qt/stall_watch.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [watch_this_application](#watch_this_application) (3 entries)
- [watch_this_application.watch](#watch_this_applicationwatch) (3 entries)

## watch_this_application

### lines 114-122

```python
previous = getattr(app, "_spacr_stall_timer", None)
```

KEPT ON THE APPLICATION as well as parented to it: a local would be collected the moment this function returns, and a collected QTimer stops, which would leave the watcher reporting one endless stall.

THE PREVIOUS ONE IS STOPPED FIRST, because this attribute is the only handle on it and assigning over it used to leave the old timer parented to the application and still firing every 100 ms with nothing left to read its heartbeat. Two installs meant two immortal timers; a test suite that installs one per test meant a pile.

### lines 134-140

```python
stopping = threading.Event()
```

A SAMPLER THAT CANNOT BE SWITCHED OFF HAS NO BUSINESS INSIDE A TWO-THOUSAND-TEST SESSION. The watcher walks the GUI thread's live frames four times a second, which is a debugging liberty that is only safe while somebody is watching. Left running after the test that installed it, it kept walking frames the main thread was pushing and popping, and the interpreter segfaulted -- reproduced 3 times in 5, and 0 in 6 once the sampling was neutered.

### lines 256-260

```python
thread.stop = stopping.set
```

ATTACHED RATHER THAN RETURNED SEPARATELY, so the return stays a plain `threading.Thread` and this function keeps the signature and the documented return it already had. A new module-scope class or a changed docstring would move the published API surface, and the ten API catalogs pin every docstring by hash.

## watch_this_application.watch

### lines 160-162

```python
stopping.wait(POLL_SECONDS)
```

`wait` rather than `sleep` so a stop is acted on immediately instead of after the poll interval. POLL_SECONDS is still read per iteration: the tests monkeypatch it.

### lines 168-171

```python
if samples:
```

THE STALL IS OVER, so its summary is due now rather than when the next one starts. Waiting for the next one loses the last stall of every session, which is the only stall a process that wedges and dies ever has.

### lines 176-179

```python
frame = sys._current_frames().get(main_thread.ident)
```

SAME STALL, ANOTHER SAMPLE. The first crossing writes the full stack; every later one only counts a frame, so a thirteen-second freeze is one readable report rather than fifty identical ones.

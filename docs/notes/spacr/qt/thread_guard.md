# Notes from `spacr/qt/thread_guard.py`

Prose lifted out of `spacr/qt/thread_guard.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## install

### lines 126-143

```python
real_object_init = QObject.__init__
```

AND WHERE A QObject IS BORN, because the timer warning names neither the object nor the code that made it, and the Python-level wrappers above never fired for the crash being chased: the start was in C++.

A QObject constructed on a worker LIVES on that worker. Every later touch from the GUI thread is then illegal -- Qt says so about the one case it can detect (a timer) and says nothing about the rest, which is how the process comes to segfault somewhere entirely unrelated, in an event filter one time and inside pandas' CSV parser the next.

Constructing one off-thread is not always wrong, so this REPORTS and never refuses, and it stops after a handful so a legitimate producer cannot flood the log.

MEASURED before leaving it on: 200,000 QObject() cost 0.407 s bare and 0.468 s guarded -- 0.31 us each, 1.15x. A window is tens of thousands of objects, so the whole of a launch pays single-digit milliseconds for a diagnostic that names a crash nothing else has.

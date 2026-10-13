# Notes from `spacr/qt/hidpi.py`

Prose lifted out of `spacr/qt/hidpi.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## follow_device_ratio

### lines 270-280

```python
if not isinstance(widget, QObject):
```

ASKED BEFORE BUILT, not caught after. `_RatioWatcher` is a QObject parented to `widget`, so constructing one with something that is not a QObject fails INSIDE PySide6's C++ layer -- and the exception unwinds leaving a half-built native object whose teardown is not safe. Catching it looked sufficient and was not: under `coverage run` the tracing perturbs that teardown enough to turn it into a segmentation fault, reproducible two times out of two, and never once without coverage.

That is a crash in a measurement, not in the product, and it is exactly the kind that gets written off as "the coverage tool is flaky". The object it needs is one that was never constructed.

### lines 287-288

```python
return None
```

Still caught: a real QWidget can refuse an event filter after its C++ half is gone, which `isinstance` cannot see.

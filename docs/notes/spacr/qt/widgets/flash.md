# Notes from `spacr/qt/widgets/flash.py`

Prose lifted out of `spacr/qt/widgets/flash.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Flash._end

### lines 72-75

```python
"""Clear the highlight and repaint, if the widget still exists.
```

The widget may have been destroyed between the trigger and the timeout -- a screen torn down mid-flash is ordinary, not an error. shiboken raises RuntimeError on a deleted C++ object, and there is nothing to repaint by then either way.

# Notes from `spacr/figures/headless.py`

Prose lifted out of `spacr/figures/headless.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [application](#application) (1 entry)
- [render_offscreen](#render_offscreen) (1 entry)

## application

### line 75, trailing  _(unsure)_

```python
if app is None:
```

ready() usually made it

## render_offscreen

### lines 121-124

```python
app.processEvents()
```

PROCESS EVENTS BEFORE EXPORTING. Offscreen, `resize` posts a layout that nothing has delivered yet, so an export taken straight after it photographs the widget's startup geometry -- which is how a figure comes out with its axes in the wrong place.

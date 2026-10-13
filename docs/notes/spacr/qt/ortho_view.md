# Notes from `spacr/qt/ortho_view.py`

Prose lifted out of `spacr/qt/ortho_view.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [OrthoPanel.paintEvent](#orthopanelpaintevent) (1 entry)
- [OrthoView._build_sliders](#orthoview_build_sliders) (1 entry)

## OrthoPanel.paintEvent

### line 179  _(unsure)_

```python
LOG.exception("Could not paint the %s panel", self._name)
```

A paint handler that raises takes the window with it.

## OrthoView._build_sliders

### lines 346-348

```python
self._drain(item.layout())
```

The rows are nested layouts; draining the widgets and then the layout keeps `set_stack` from leaving an empty row behind every time it is called.

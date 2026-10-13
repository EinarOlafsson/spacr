# Notes from `spacr/qt/widgets/colour_picker.py`

Prose lifted out of `spacr/qt/widgets/colour_picker.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## pick_colour

### lines 62-63

```python
start = QColor("#ffffff")
```

A stored preference can hold anything, including "auto" or "none", and QColorDialog on an invalid colour opens on transparent black.

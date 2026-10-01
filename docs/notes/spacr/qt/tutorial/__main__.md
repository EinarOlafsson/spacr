# Notes from `spacr/qt/tutorial/__main__.py`

Prose lifted out of `spacr/qt/tutorial/__main__.py` by `tools/extract_source_notes.py`.
Ordinary comments move here; tool directives and published attribute documentation stay in the module. The path mirrors the source path, which is how its reasons are found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## main

### lines 51-53

```python
with _open_sans_is_the_default():
```

291: the tutorial films the application, which draws its figures in Open Sans; `render_tutorial` builds MainWindow itself rather than going through `spacr.qt.run`, so it holds the same default here.

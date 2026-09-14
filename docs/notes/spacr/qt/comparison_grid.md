# Notes from `spacr/qt/comparison_grid.py`

Prose lifted out of `spacr/qt/comparison_grid.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## ComparisonGrid.__init__

### lines 231-234

```python
self._canvas_link = CanvasLink()
```

NOT `_link`: that name belongs to LinkedView, which keeps the process-wide selection bus in it. Shadowing it here would leave this grid publishing selections into its own canvas link and hearing nothing from the rest of the app.

# Notes from `spacr/qt/widgets/collapsible_splitter.py`

Prose lifted out of `spacr/qt/widgets/collapsible_splitter.py` by `tools/extract_source_notes.py`.
Ordinary comments move here; tool directives and published attribute documentation stay in the module. The path mirrors the source path, which is how its reasons are found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## _PaneHandle._paint_arrow

### lines 553-554

```python
tab = tab.lighter(165) if tab.lightness() < 128 else tab.darker(108)
```

Dark GREY, not the near-black the surfaces carry: the tab has to read as a control sitting on the page rather than a hole in it.

# Notes from `spacr/qt/roi_tool.py`

Prose lifted out of `spacr/qt/roi_tool.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## RoiPanel._show

### lines 578-579

```python
style = self.status.style()
```

An objectName change only takes effect on a re-polish; without this the warning colour arrives one message late.

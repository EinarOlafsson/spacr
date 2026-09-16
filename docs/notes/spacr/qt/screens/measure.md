# Notes from `spacr/qt/screens/measure.py`

Prose lifted out of `spacr/qt/screens/measure.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## install_folds

### lines 157-159

```python
mark_fold_sources(screen)
```

The icons the folded modules brought with them go on their settings as well as on their buttons. After the strip: a mark that cannot be drawn must not cost the buttons.

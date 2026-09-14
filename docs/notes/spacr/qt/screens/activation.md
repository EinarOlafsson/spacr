# Notes from `spacr/qt/screens/activation.py`

Prose lifted out of `spacr/qt/screens/activation.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## ExplainNavigator._on_train_requested

### lines 171-173

```python
key = APP_KEY
```

No fold on this host -- the page is in a window of its own. Ask the window for the module by the name the registry knows it under rather than by the one that reaches nothing.

# Notes from `spacr/qt/widgets/model_share_dialog.py`

Prose lifted out of `spacr/qt/widgets/model_share_dialog.py` by `tools/extract_source_notes.py`.
Ordinary comments move here; tool directives and published attribute documentation stay in the module. The path mirrors the source path, which is how its reasons are found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## ShareDialog.__init__

### lines 59-60  _(unsure)_

```python
self._train_dir = ""
```

Optional training data. A model whose data came with it can be retrained and checked; one without it can only be believed.

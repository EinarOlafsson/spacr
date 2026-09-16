# Notes from `spacr/qt/widgets/external_mask_inputs.py`

Prose lifted out of `spacr/qt/widgets/external_mask_inputs.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## ExternalMaskInputWidget._rebuild

### lines 224-225  _(unsure)_

```python
source_item.setData(Qt.UserRole, row)
```

Which group the row was built from. The table sorts, so the third row is not the third group after a header click.

# Notes from `spacr/ops_sample.py`

Prose lifted out of `spacr/ops_sample.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## barcode_rows

### lines 265-269

```python
entry = mapped[index]
```

`correct_to_library` returns None for a read that is equally close to two library barcodes -- "AMBIGUITY IS DISCARDED, NOT GUESSED". That None must survive into the table as an empty cell rather than the string "None", which would look like a guide called None and join against nothing.

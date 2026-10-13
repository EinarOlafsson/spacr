# Notes from `spacr/methods_export.py`

Prose lifted out of `spacr/methods_export.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## digest_numbers

### lines 180-181  _(unsure)_

```python
found.add(float(token))
```

_NUMBER accepts a strict subset of Python's float syntax, so a fully matched token is always convertible.

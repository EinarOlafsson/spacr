# Notes from `spacr/intensity_rescale.py`

Prose lifted out of `spacr/intensity_rescale.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [fallback_record](#fallback_record) (1 entry)
- [resolve_record](#resolve_record) (1 entry)

## fallback_record

### line 176, trailing  _(unsure)_

```python
comparable = True
```

fixed by definition, even without a pre-pass

## resolve_record

### line 226  _(unsure)_

```python
return fallback_record(data, filename, settings)
```

Missing plate or pixels replaced by brighter ones after the scan.

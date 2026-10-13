# Notes from `spacr/mask_io.py`

Prose lifted out of `spacr/mask_io.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## save_mask

### line 86  _(unsure)_

```python
if p.suffix.lower() in (".tif", ".tiff", ".npy"):
```

If path already has a recognised suffix, that wins over `fmt`.

### lines 93-96

```python
write_tiff = import_module(".tiff_io", __package__).write_tiff
```

Resolve through sys.modules rather than the package attribute. Python leaves ``spacr.tiff_io`` attached to the parent package after its module-cache entry is removed, which could otherwise make an optional dependency look available after it vanished.

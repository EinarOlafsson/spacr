# Notes from `spacr/ops_settings.py`

Prose lifted out of `spacr/ops_settings.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Module level

### lines 28-32

```python
"dst_root": (str, type(None)),
```

where the data is

`src` IS NOT HERE ON PURPOSE. It is already declared as (str, list) by another module, and `register_defaults` refuses a redeclaration rightly, because two modules disagreeing about what a shared key may hold is a bug, not a preference. OPS uses the shared meaning.

### line 88  _(unsure)_

```python
"save_qc": bool,
```

what it draws for you to check

### lines 96-100

```python
"ops_gpu": bool,
```

`ops_gpu` and not `gpu`: `gpu` is already declared by Image UMAP, where it means "use the RAPIDS cuML reducer". Two modules disagreeing about what one key means is the bug `register_defaults` refuses `src` to prevent, and a prefix costs nothing.

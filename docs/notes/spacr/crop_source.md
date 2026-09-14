# Notes from `spacr/crop_source.py`

Prose lifted out of `spacr/crop_source.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [normalise_extension](#normalise_extension) (2 entries)
- [validate](#validate) (1 entry)

## normalise_extension

### lines 166-168

```python
def normalise_extension(file_type: Any) -> str:
```

Pre-generated: two filters that used to be one confused setting

### lines 183-184

```python
if "_" in text:
```

Tolerated because it is what every old settings CSV holds: the old value was `<object>_png`, whose extension is the part after the underscore.

## validate

### lines 492-497

```python
if not settings.get("image_size"):
```

ONE COLUMN OR TWO, because there are two ways a database says where an object is and both are in use. One column NAMES THE OBJECT `cell_id` -- and the mask plane supplies its extent; two give a centroid's row and column, and the box is cut around it. Demanding two refused spaCR's own derived value, which is the single identifier column `stream_dataset.coordinate_column` produces.

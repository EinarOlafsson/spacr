# Notes from `spacr/annotation_dataset.py`

Prose lifted out of `spacr/annotation_dataset.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [filter_selection](#filter_selection) (2 entries)
- [png_list_frame](#png_list_frame) (1 entry)
- [generate_annotation_dataset](#generate_annotation_dataset) (4 entries)
- [generate_annotation_dataset._write](#generate_annotation_dataset_write) (1 entry)

## filter_selection

### lines 138-139

```python
if minimum:
```

0 IS NOT A MINIMUM. The Measure panel writes 0 for an unset bound, and honouring it as one would filter nothing while looking filtered.

### lines 157-158

```python
frame = frame.head(int(cap))
```

Head of the existing order, not a sample: a set that differs between two runs of the same settings cannot be compared with anything.

## png_list_frame

### lines 196-198

```python
out["fieldID"] = out["fieldID"].astype(str)
```

`prcfo` is the join key every other measurement table carries, so a streamed set can be joined to the measurements the same way a measured one can.

## generate_annotation_dataset

### lines 309-312

```python
table = str(settings.get("table") or "")
```

THE TABLE NAME IS CLAIMED FIRST, and the crop folder is named after it. `png_list` gets `data`, `png_list_2` gets `data_2` -- so a folder on disk says which table describes it. Two independent counters would drift the first time either was deleted.

### lines 329-330

```python
objects = read_objects_from_database(database, object_type)
```

THE COORDINATE COLUMNS. They are all the database stores, which is why this route can only ever produce a bounding box.

### line 381  _(unsure)_

```python
bounding_box=(True if source == "database"
```

FORCED for the database route, which has no mask to cut to.

### lines 391-393

```python
report["table"] = ""
```

The reserved table stays, empty, rather than being dropped: it is the record that this name is spoken for, and dropping it would let a later run reuse a name whose folder is already on disk.

## generate_annotation_dataset._write

### lines 364-372

```python
from .measure import _save_object_crop
```

`measure_crop`'S OWN WRITER, not a second one.

The annotation viewer shows pictures, so a set written as .npy is not an annotation set -- but that is the smaller reason. The bigger one is that instruction 338 asks for a streamed set and a measured set to be the SAME IMAGES, and two writers cannot be relied on to narrow to 8-bit, pad a two-channel crop, or resize identically. One writer makes that a property of the code rather than a coincidence to be re-tested after every change to either.

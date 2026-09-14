# Notes from `spacr/stream_dataset.py`

Prose lifted out of `spacr/stream_dataset.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [selection_from_objects](#selection_from_objects) (1 entry)
- [selection_from_arrays](#selection_from_arrays) (1 entry)
- [build_selection](#build_selection) (1 entry)
- [crop_name](#crop_name) (1 entry)
- [_field_stem](#_field_stem) (1 entry)
- [_stack_for](#_stack_for) (2 entries)
- [_well_spelling](#_well_spelling) (1 entry)
- [stream](#stream) (5 entries)

## Module level

### lines 30-33

```python
"array": ("object_array", "channel_arrays", "bounding_box"),
```

`mask_array` was here and is retired (357-Q4): it claimed to name the labelled plane and this module takes that from `object_array` for BOTH methods, so the 'array' route's own documented input was never read. Greying a setting that does not exist greys nothing.

### lines 43-46

```python
"organelle": "organelle_id",
```

THE ORGANELLE ROLES TOO. The column is named after the object type in every case, so the four organelle planes a measure run can write follow the same rule -- and a user who segmented one and cannot stream it has an object type spaCR measured and will not show.

## selection_from_objects

### lines 187-188  _(unsure)_

```python
answer, why = ask(tried=tried, object_array=object_array)
```

The user knows where the coordinates live and the program does not, so asking is strictly more useful than reporting.

## selection_from_arrays

### line 259, trailing  _(unsure)_

```python
if int(label) == 0:
```

background is not an object

## build_selection

### lines 304-310

```python
from .tabular import write_table
```

WRITTEN BEFORE ANY IMAGE IS. A training set decided at run time and never recorded cannot be re-made or audited.

Through `tabular.write_table`, not `to_csv`: the selection names wells and plates, and those columns have three spellings in this codebase. A record written in whichever one the source frame happened to use is a record the next reader has to guess at.

## crop_name

### lines 317-319  _(unsure)_

```python
def crop_name(field_stem: str, object_id, *, crop_mode: str = "cell",
```

The crop, named the way measure_crop names one

## _field_stem

### lines 376-378  _(unsure)_

```python
def _field_stem(row) -> str:
```

The pass that writes the dataset

## _stack_for

### lines 407-409

```python
for name in sorted(os.listdir(str(merged_folder))):
```

A field written as `plate_A01_1_0.npy` has a trailing token the stem does not; match on the prefix as a second pass rather than a first, so an exact name always wins.

### lines 413-421

```python
well = _well_spelling(wanted)
```

AND THE WELL SPELLING, which is the difference between the two routes.

A selection built from the merged arrays records the file it came from, so it never reaches here. One built from the DATABASE has only the parsed identifiers, and those come back as `r1`/`c1` while the file is named `plate1_A01_1_1.npy` -- so neither test above matches and every object in every field was reported missing. The database route wrote nothing at all, which is what instruction 338's parity test found the first time it ran.

## _well_spelling

### lines 452-453  _(unsure)_

```python
return ""
```

Beyond Z a plate uses AA, AB … and this is not the place to invent that convention; say so by returning nothing.

## stream

### lines 517-521

```python
frame["_stem"] = [
```

THE TABLE ALREADY KNOWS WHICH FILE EACH OBJECT CAME FROM when it was built from the .npy stacks, and that beats rebuilding the name from the parsed parts: a merged file is named `plate1_A01_1_0.npy` while the parts come back as r1/c1, so a rebuilt stem matches NOTHING. The first attempt at this missed every field for exactly that reason.

### lines 531-533

```python
found_stem = (os.path.splitext(os.path.basename(path))[0]
```

The stem AS THE FILE SPELLS IT, which is what the crops are named after. `_stack_for` resolves a database-built stem onto the file's own well spelling, and that resolution must reach the names too.

### lines 537-539

```python
report["missing"] += int(len(here))
```

COUNTED, NOT SKIPPED SILENTLY. A dataset short by a field is a dataset trained on a different screen from the one the table describes, and nothing else would say so.

### lines 554-561

```python
from .crops import object_label
```

THROUGH THE ONE PARSER. The coordinate method takes its object id from the object table's own column, and `png_list` spells it `cell_id = 'o2'` -- so `int(float(...))` raised on every row of the crop table and each object was counted as missing and dropped in silence. The array method never hit it, because a label scanned off a mask plane is already an integer, so the two methods produced different datasets from the same screen.

### lines 573-578

```python
name = crop_name(found_stem, label, crop_mode=crop_mode)
```

NAMED FROM THE FILE THAT WAS FOUND, not from the stem that went looking for it. The two spell the same field differently `plate1_A01_1_1` from the arrays, `plate1_r1_c1_1` from the database -- so naming from the stem gave the two routes different names for identical pictures, and a set built one way could not be matched against a set built the other.

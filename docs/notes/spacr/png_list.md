# Notes from `spacr/png_list.py`

Prose lifted out of `spacr/png_list.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## crop_rows_from_png_list

### lines 135-137

```python
alternatives = [
```

A png_list written for one crop mode carries only that mode's id column. Resolve the MODE with its column, because carrying nucleus labels forward as cells silently cuts a different object.

### lines 153-155

```python
labels = df['object_label'].map(_object_id_int)
```

Not a png_list at all: a frame that already came off the object table (crop_rows_from_object_table) carries the integer label directly. Looking up a column that is not there would drop every row.

### line 162, trailing  _(unsure)_

```python
pass
```

the frame already names its merged array

### lines 164-165

```python
fields = _merged_field_paths(db_path, effective_object_type)
```

png_list records where a crop was written, never which merged array produced it; the object table is the only place that link exists.

### lines 172-183

```python
df['object_type'] = object_type
```

THE OBJECT THE CALLER ASKED FOR, not the column the labels came from. These are two different questions and answering both with `effective_object_type` broke the montage: `crops` reads this column PER ROW to choose the mask plane a crop is cut by (`_row_get(row, "object_type", ...)`), so a nucleus request whose png_list carries only `cell_id` came back saying "cell" and was cut from the cell plane. Choosing an object type then changed nothing on screen.

The labels stay whatever column exists -- that is what the fallback above is for, and it is the honest answer to "which objects" when the png_list was written for one crop mode. The PLANE is the user's choice.

### lines 185-189

```python
df['object_label_type'] = effective_object_type
```

WHERE THE LABELS CAME FROM, recorded rather than folded into the line above. The two are different questions and one column cannot answer both: `object_type` is an INSTRUCTION the crop cutter obeys, and this is PROVENANCE. They differ exactly when a png_list written for one crop mode is read for another, which is the case worth being able to see.

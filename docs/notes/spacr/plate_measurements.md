# Notes from `spacr/plate_measurements.py`

Prose lifted out of `spacr/plate_measurements.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [_rows](#_rows) (1 entry)
- [merge_plate_databases](#merge_plate_databases) (9 entries)

## Module level

### lines 66-70

```python
from .merge_tables import (IDENTITY, OBJECT_COLUMN, OBJECT_TABLES, PNG_TABLE,
```

The structural constants are imported by value; AGGREGATION_RULES and DEFAULT_AGGREGATION are read through `mt.` at CALL time, deliberately. They are the maintainer's living decision about what each measurement means, and a module that snapshots them at import is a module that can disagree with them.

### lines 449-462

```python
IDENTIFIER_CARRIED = "first"
```

Instruction 154 C: a "no rule matched" bucket is not one bucket

The panel used to report 85 columns as "matched no aggregation rule and would take the default (mean)", and two of them were `file_name` and `path_name`. A mean of a path is not merely unhelpful, it is impossible and it is not what happens: `aggregation_plan` asks the DTYPE first, so a text column takes `TEXT_AGGREGATION` (first) whatever its name. The list was right about which columns no NAME rule matched and wrong about what would be done to them.

So the bucket is split by kind before it is reported, and the text half gets the treatment instruction 79 asks for rather than a silent `first`.

## _rows

### lines 342-343  _(unsure)_

```python
pairs.append((entry.get("plate") or "",
```

The input table's own row shape, so a caller can pass

`PairedFileTableWidget.get_value()` straight through.

## merge_plate_databases

### lines 736-739

```python
raise MergeRefused(
```

A nucleus carries its parent in `cell_id` and a pathogen carries its own identity in `object_label`. Anchoring on a many-per-cell table would join one to the other and match a cell id against a pathogen label -- a join on a coincidence, which returns rows.

### lines 762-766

```python
frame = read_merged(paths, table, plan=plan, columns=columns,
```

`on_collision` is deliberately left at 'refuse'. 'qualify' rewrites plate1 to `<label>-plate1`, which makes the keys unique by hiding the screen inside the plate id and stops it being analysable (instruction 122). Two screens sharing a plate id do not reach this branch at all once `screens=` is passed.

### lines 770-771

```python
dropped = tuple(frame.attrs.get("dropped_columns", ()))
```

What the read cost, from the read itself: `columns='union'` drops nothing, and the plan's own list would say otherwise.

### lines 785-787

```python
note = (f"carries no {link}, so its rows cannot be matched to a "
```

Measured without a parent mask: the roll-up is not empty, it is UNDEFINED. Named and skipped, exactly as `merge_tables` does one unlinkable table must not cost the user the others.

### lines 800-802

```python
skip = set(keys) | {"prcf", "prcfo"}
```

One row per cell already: aggregating it is not wrong so much as meaningless, and it would put the table's own measurements through the sum/mean rules meant for a GROUP of children.

### lines 810-811

```python
plan_for_table = aggregation_plan(frame, overrides=policy.overrides,
```

What the panel shows and what actually happens, from the same function, so they cannot disagree.

### lines 814-817

```python
ambiguous = ambiguous_identifiers(frame, keys,
```

REFUSED, NOT PICKED. A text identifier that differs across the children being combined is a genuine ambiguity (instruction 79 item 2): `first` would put one of two file names on the cell and the merged row would claim a provenance the data cannot support.

### lines 837-839

```python
mt._align_keys(merged, rolled, on)
```

One dtype policy for join keys, not two: `_align_keys` is where spaCR decided that a plate called `1` read as an integer from one table and a string from another is the same plate.

### lines 859-860

```python
merged.attrs["anchor"] = anchor_table
```

Carried on the frame so they cannot be separated from the data they describe -- the same reason `read_merged` does it.

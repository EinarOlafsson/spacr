# Notes from `spacr/foreign.py`

Prose lifted out of `spacr/foreign.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ColumnMap.resolve](#columnmapresolve) (2 entries)
- [ImportPlan.with_column_maps](#importplanwith_column_maps) (1 entry)
- [_resolve_columns](#_resolve_columns) (1 entry)
- [plan_import](#plan_import) (4 entries)
- [_db_table_names](#_db_table_names) (1 entry)
- [release_canonical_copy](#release_canonical_copy) (8 entries)
- [_check_destination](#_check_destination) (1 entry)
- [_replace_table_atomically](#_replace_table_atomically) (1 entry)
- [run_import](#run_import) (9 entries)
- [import_project](#import_project) (1 entry)

## ColumnMap.resolve

### lines 576-577

```python
return None, (f'unknown transform {self.transform!r}; expected '
```

Includes '/0', whose literal_factor is None because dividing by zero is not a unit conversion.

### line 623  _(unsure)_

```python
return float(scale ** -power), ''
```

px = um / (um per px)

## ImportPlan.with_column_maps

### lines 1503-1505

```python
status = {r.source: r.status for r in resolved}
```

"Unmapped" covers both ways a column ends up undecided: no row in the map file at all, and a row whose target was left blank. They are the same thing to the user, so they are one list.

## _resolve_columns

### line 1622

```python
if target in RESERVED_COLUMNS:
```

reserved key columns: never, under any setting

## plan_import

### lines 1811-1813

```python
if z_handling == cv.Z_KEEP and any(m.z > 1 for m in image_plan.mappings):
```

A merged array is (H, W, C): one plane per channel, no z axis. Keeping every plane would produce N files per channel with nothing to merge them into, so it is refused here rather than half-way through.

### lines 1873-1874  _(unsure)_

```python
hit = None
```

A mask tree that has no plate/well folders of its own still has to reach the images' single plate and well.

### lines 1949-1952

```python
frame = read_measurements(measurements,
```

3. their table reset_index so row positions are 0..n-1: every count below is positional, and a table read back from SQL with a non-unique index would otherwise silently multiply rows on the .loc select.

### lines 2056-2059

```python
proposed = column_maps is None
```

5. the columns

Built last, and through with_column_maps(), so that the mapping a GUI re-resolves on every edit goes down exactly the same code path as the one plan_import produces. One resolver, one set of conflicts.

## _db_table_names

### lines 2414-2416

```python
def _db_table_names(connection: 'sqlite3.Connection') -> Set[str]:
```

Never writing over what is already there

## release_canonical_copy

### line 2758, trailing  _(unsure)_

```python
return 0
```

no import ever touched this table

### line 2759  _(unsure)_

```python
written = resume.importer_written_columns(connection, object_type) or set()
```

Read before the un-claim below removes the provenance it comes from.

### lines 2790-2791

```python
connection.isolation_level = None
```

One transaction for the delete and the un-claim: a claim that outlives the rows it was about is the failure this replaces.

### lines 2798-2803

```python
cursor.execute(
```

The same WHERE clause the two counts above were taken with, applied to the table itself. No row identity is named: see :data:`_RELEASE_ALIAS` for what naming one cost. ``DELETE FROM t AS alias`` has been SQLite since 3.25 and this module already needs 3.35 for the DROP COLUMN below.

### lines 2808-2814

```python
if removed != held:
```

``held`` rows matched the importer clause and ``orphans`` of them — zero, or the raise above fired — failed the twin check, so exactly ``held`` rows should have gone. Anything else means the delete did not select what the checks inspected, which is the failure this whole function exists to make impossible. Refuse loudly and roll the delete back rather than report a number that is not what happened.

### lines 2824-2828

```python
if (FOREIGN_COLUMNS_TABLE in names
```

``"table"`` is checked for rather than assumed: SQLite reads a double-quoted name matching no column as a string literal, so a provenance table without it would silently match nothing instead of failing — the same trap that once made :func:`_importer_owns` answer False for its own table.

### lines 2851-2858

```python
for column in [c for c in _db_columns(connection, object_type)
```

Their measurement columns are left behind by the delete, empty, and would otherwise collide with the same columns in the view below — ``cell.foreign_areashape_area`` (all NULL) forcing theirs to be aliased ``foreign_foreign_areashape_area``. Dropped only when provably empty, one at a time, and never fatally: ALTER TABLE … DROP COLUMN needs SQLite 3.35 and refuses a column an index names, and a tidier schema is not worth failing a release that has already happened.

### lines 2872-2873

```python
_write_view(db_path, object_type)
```

Outside the transaction: the view is a convenience, and a failure to build one must not roll back a release that already succeeded.

## _check_destination

### lines 2923-2929

```python
copied = release_canonical_copy(db_path, object_type,
```

spaCR's own measurements are about to fill ``<object>``, and a *previous* import may have left its convenience copy in there — this branch used to return before ever looking. The copy has to go, and whether it can go losslessly is asked here, before a single file is written, rather than after the conversion has run for minutes. Asked through the function that does the removal, so the answer cannot differ from it.

## _replace_table_atomically

### line 2954, trailing  _(unsure)_

```python
connection.isolation_level = None
```

so BEGIN below is really ours

## run_import

### lines 3320-3321

```python
mode, destination_notes = _check_destination(db_path, plan, object_type,
```

Asked and answered before a single file is written: a destination that cannot take this import must not be left holding half of it.

### line 3343  _(unsure)_

```python
_step(1, 'converting images')
```

1. their images -> Yokogawa TIFFs, via spacr.convert

### lines 3380-3392

```python
_step(3, 'writing masks')
```

3. their masks

``mask_type``, not ``object_type``: an import declares one mask class per folder but has exactly **one** measured object type — the one the table was joined against, ``plan.join.object_type``, bound above and already used to decide ``mode`` against the destination. A loop variable named ``object_type`` rebinds it to the *last* mask class, and everything downstream then aims at the wrong table: their cell measurements were written to ``foreign_nucleus`` and to the canonical ``nucleus`` table, joined by a ``nucleus_with_foreign`` view that paired their cell 1 with spaCR's nucleus 1, while ``foreign_import`` went on recording ``canonical_table = 'cell'``. Python has no block scope, so the only guard is the name.

### lines 3397-3399

```python
for stem in usable:
```

Iterating the field's own masks rather than the declared classes: a stem only reaches ``masks.fields`` when it has every class, so there is no "missing" case here to guess at.

### line 3408  _(unsure)_

```python
_step(4, 'merging arrays')
```

4. merged arrays, with spaCR's own merger

### lines 3430-3432

```python
_replace_table_atomically(connection, table, frame)
```

replace, never append: a second run of the same import must not leave two generations of the same rows behind. This table is the importer's own, which is the only reason replacing it is safe.

### lines 3438-3448

```python
allowed, held = _may_write_canonical(connection, object_type)
```

A project built purely by import has no spaCR measurements of its own, so the same rows are copied into the canonical table to make it readable by every tool that reads one.

The question was already answered before any file was written, and it is asked again here against the database as it is now: minutes of image conversion separate the two, and a measure_crop running alongside would have created that table in between. The check that decides whether a table may be dropped has to be the one taken at the moment it is dropped.

### line 3471  _(unsure)_

```python
if measure:
```

6. spaCR's own measurements, optional and separate

### lines 3474-3479

```python
release_canonical_copy(db_path, object_type)
```

An earlier import's convenience copy is superseded by what is about to be measured, and ``measure_crop`` appends. Released here, at the last moment before the write, for the same reason the canonical write above re-asks its question: minutes of image conversion separate this point from the check in ``_check_destination``, which established that it *can* be done.

## import_project

### lines 3603-3606

```python
for note in _check_destination(
```

What the destination already holds is part of the plan a user reads, so it is checked here too and not only inside run_import — a preview that does not mention the table it will refuse to touch is not a preview of this import.

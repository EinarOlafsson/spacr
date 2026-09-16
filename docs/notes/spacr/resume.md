# Notes from `spacr/resume.py`

Prose lifted out of `spacr/resume.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (5 entries)
- [_descr_itemsize](#_descr_itemsize) (1 entry)
- [validate_merged_field](#validate_merged_field) (2 entries)
- [completed_fields_in_merged](#completed_fields_in_merged) (2 entries)
- [_foreign_name_column](#_foreign_name_column) (1 entry)
- [completed_fields_in_db](#completed_fields_in_db) (3 entries)
- [clear_field_rows](#clear_field_rows) (2 entries)
- [_fields_matching](#_fields_matching) (1 entry)
- [supersede_imported_copies](#supersede_imported_copies) (3 entries)
- [SettingsComparison](#settingscomparison) (1 entry)
- [plan_resume](#plan_resume) (1 entry)
- [plan_measure_resume](#plan_measure_resume) (7 entries)

## Module level

### line 260, trailing  _(unsure)_

```python
'cell', 'cytoplasm',
```

_PARENT_OBJECT_TABLES

### line 261, trailing  _(unsure)_

```python
'nucleus', 'pathogen', *ORGANELLE_ROLES,
```

_CHILD_OBJECT_TABLES

### line 264, trailing  _(unsure)_

```python
'png_list',
```

filepaths_to_database

### line 265, trailing  _(unsure)_

```python
'intensity_rescale',
```

measure provenance upsert

### line 333  _(unsure)_

```python
REASON_DONE = 'done'
```

Reason codes recorded against a field in ResumeState.reasons.

## _descr_itemsize

### lines 479-481  _(unsure)_

```python
def _descr_itemsize(descr: Any) -> Optional[int]:
```

.npy validation — the difference between "the file is there" and "it finished"

## validate_merged_field

### lines 606-607  _(unsure)_

```python
return False, REASON_UNREADABLE
```

Zero bytes is caught above; anything else here is a header that does not parse — a partial write, or not an array at all.

### lines 612-613

```python
return False, REASON_UNREADABLE
```

Unverifiable dtype. Cannot prove the file is complete, so treat it as pending: re-measuring is safe, skipping garbage is not.

## completed_fields_in_merged

### line 701  _(unsure)_

```python
verdicts: Dict[str, Tuple[bool, str]] = {}
```

Pass 1: structural validity and plane counts.

### lines 715-718

```python
if min_planes is None and len(planes) >= 3:
```

Pass 2: modal plane count, only when the caller gave no explicit floor and there is enough of a population to have a mode. A merged folder is written by one loop with one channel layout, so a field with fewer planes than its neighbours did not finish.

## _foreign_name_column

### lines 770-799

```python
def _foreign_name_column(conn: sqlite3.Connection) -> Optional[str]:
```

Owned by name is not owned in fact: which *rows* the measure stage wrote

``cell`` is on the allow-list above, and in a project built by ``foreign.run_import`` the rows in it are the import's: the importer copies its own frame into the canonical table when nothing of anyone else's is there, so that a purely-imported project is readable by every spaCR tool. Nothing about the *table* tells the two apart, which is why a table-scoped ownership claim was tried here and backed out — it took the whole table with it, including fields the import never covered and rows ``measure_crop`` wrote afterwards.

The signal that is already in the tree, written by every version of the importer that has ever existed, is ``foreign_columns``: one row per column per table, naming exactly what that importer put there. The table-scoped question ``foreign._importer_owns`` asks of it —"are this table's columns a subset of what the importer recorded?"— has a row-scoped twin, and that is what is used below:

a row is the importer's when every column it is non-NULL in is one the importer wrote into this table.

A ``measure_crop`` row in a canonical table always carries at least its own area and intensity columns, which the importer never wrote and never recorded, so it fails that test; an imported row carries only metadata and ``foreign_``-prefixed measurements, which it passes. The test is evaluated per row in SQL, so a table can be half one and half the other — which, until the append itself is fixed, is exactly what a project that was imported and then measured contains.

## completed_fields_in_db

### lines 1140-1145

```python
clause = measure_rows_clause(conn, table)
```

Rows a foreign import copied in are not evidence that measure ran. In a purely-imported project ``cell`` is the only field table and holds a row for every field, so counting them made ``measure_crop`` report the whole plate done, run nothing at all, and present the collaborator's numbers as spaCR's own output.

### lines 1161-1170

```python
continue
```

An import filled this table and measure has never written a row into it, so it is not one of "the tables this run writes" and must not make every field look partial. It rejoins the moment measure puts a row in it.

``_has_rows`` is the difference between "theirs" and

"empty". An emptied table — one whose copy has been released back to spaCR — is an ordinary measure table with nothing in it yet, and every field must read as not-measured in it, exactly as for any other empty table.

### line 1196  _(unsure)_

```python
all_keys: Set[Tuple[str, ...]] = set()
```

No candidate list: answer in terms of the rows themselves.

## clear_field_rows

### lines 1291-1292  _(unsure)_

```python
plans = []
```

Pre-flight and deletes share one write transaction. No other writer can replace a checked table between validation and use.

### lines 1308-1314

```python
clause = measure_rows_clause(conn, table)
```

...and only the rows the measure stage itself wrote. The allow-list answers "could measure have written this table?"; this answers "did it write this row?", which in a project built by ``foreign.run_import`` is a different question — the rows in ``cell`` are the import's, under spaCR's own metadata columns, and clearing a pending field used to take them with it.

## _fields_matching

### lines 1328-1330  _(unsure)_

```python
def _fields_matching(conn: sqlite3.Connection, table: str,
```

The importer's convenience copy, when measure is about to supersede it

## supersede_imported_copies

### line 1457, trailing  _(unsure)_

```python
continue
```

no import ever wrote here

### lines 1495-1499

```python
from .foreign import release_canonical_copy
```

Imported here, not at module scope: ``spacr.foreign`` pulls in pandas, numpy and ``spacr.convert``, and this module is consulted at the top of ``measure_crop`` precisely so that a question answered by reading a sqlite table costs nothing. A project with no import never reaches this line.

### lines 1502-1506

```python
notes.append(
```

Everything, not ImportError alone: a module that fails to initialise raises whatever its own top level raised. Refuse, never delete blind — without the importer's verification that every row has a twin in foreign_<object> there is no way to know the removal is lossless.

## SettingsComparison

### lines 1557-1559  _(unsure)_

```python
@dataclass(frozen=True)
```

Settings compatibility — a resume across different settings is not a resume

## plan_resume

### line 1930  _(unsure)_

```python
ordered: List[str] = []
```

Preserve order, drop duplicates.

## plan_measure_resume

### lines 2096-2098

```python
recorded = read_recorded_settings(db_path)
```

1. Settings guard. A recorded-settings table that does not exist yet means nothing has been measured, so there is nothing to be incompatible with.

### line 2103  _(unsure)_

```python
rejected: Dict[str, str] = {}
```

2. Which fields are physically usable.

### line 2108  _(unsure)_

```python
partial: Dict[str, str] = {}
```

3. Which fields the database already has, in full.

### line 2115, trailing  _(unsure)_

```python
reasons.update(rejected)
```

truncated / empty / too-few-planes

### line 2116, trailing  _(unsure)_

```python
reasons.update(partial)
```

rows in some tables but not all

### lines 2121-2129

```python
cleared = 0
```

4. Delete-before-insert. Every field about to be re-measured has any rows it already left behind removed first, in one transaction per field. Without this the resume silently doubles objects.

Only the fields that actually have rows are cleared. On a typical resume that is one field — the one that was mid-flight when the run died — and issuing a DELETE for the other ninety-nine would mean ninety-nine full scans of a million-row table to delete nothing. `require_all=False` is the "has rows anywhere" query.

### lines 2133-2137

```python
if tables and state.pending:
```

3a. A foreign import's convenience copy in a canonical table is superseded the moment spaCR measures the same fields into it. Released here — before the deletes and long before the first insert — because measure appends, and a table holding both populations makes every per-well count the sum of two.

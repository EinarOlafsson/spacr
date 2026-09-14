# Notes from `spacr/database_schema.py`

Prose lifted out of `spacr/database_schema.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [_rename_legacy_columns](#_rename_legacy_columns) (1 entry)
- [migrate_database](#migrate_database) (1 entry)
- [ensure_database_schema](#ensure_database_schema) (1 entry)

## Module level

### lines 47-48  _(unsure)_

```python
SPACR_APPLICATION_ID = int.from_bytes(b"SPCR", "big")
```

``SPCR`` as a four-byte big-endian integer.  SQLite reserves

``application_id`` for applications to identify their file format.

### line 51  _(unsure)_

```python
CURRENT_SCHEMA_VERSION = 1
```

Version 1 establishes canonical metadata/feature column spellings.

### lines 124-138

```python
DB_COLUMN_RENAMES = _schema.LEGACY_COLUMN_NAMES
```

Legacy column spellings and their canonical spaCR names.  These are *aliases*, not copies.  This module used to define its own narrower, case-sensitive rename map and its own ``canonical_column_name`` alongside ``spacr.schema``'s wider case-insensitive pair, and which one a caller got depended on whether it had imported ``spacr.schema`` or ``spacr.utils`` (which re-exports from here).  The two disagreed on 11 aliases and on case, so a database column named ``Row`` was canonicalised on one path and left alone on the other -- and a half-canonicalised database produces a join that quietly returns the wrong rows long before it produces an error.  There is now one definition, in ``spacr.schema``; see its ``canonical_column_name`` docstring for what widened and why.

``spacr.schema`` is standard-library-only at module scope, so importing it here preserves this module's promise that a measurement worker can import it without pulling in pandas or any optional analysis dependency.

## _rename_legacy_columns

### lines 173-188

```python
others = {name.lower() for name in columns if name != old}
```

SQLite compares identifiers case-insensitively, so the

"target already exists, keep both" test has to fold case too.  A table holding `row` and `RowID` is a table that already has the canonical column -- with the old case-sensitive `new in columns` test this loop asked SQLite to rename `row` to `rowID`, SQLite answered "duplicate column name: RowID", and the OperationalError rolled back the whole migration.  A user whose database had that pair could not open it at all.

`old` is excluded from the comparison because it is the column being renamed: without that, a pure respelling (`RowID` -> `rowID`, one column, same identifier as far as SQLite is concerned) would look like a collision with itself and never happen, leaving pandas readers with a frame that has no `rowID` column on a database that does.

## migrate_database

### lines 421-427

```python
if given.startswith("~"):
```

A TILDE THAT NOBODY EXPANDED IS THE COMMONEST WAY TO REACH HERE, and `FileNotFoundError: ~/x/measurements.db` is a message that reads as "your database is missing" when the database is fine and the PATH was never resolved. GitHub issue #108 is exactly this, from a macOS user whose settings carried `~`. This function's contract stays strict see the docstring, and `ensure_database_schema` is where expansion belongs -- but it can at least name the real problem.

## ensure_database_schema

### lines 488-497

```python
db_path = os.path.abspath(os.path.expanduser(os.fspath(db_path)))
```

EXPANDED HERE, AND ONLY HERE. This is the function every reader calls to make a database usable, so it is the boundary a user-supplied path crosses -- and `os.path.abspath(os.path.expanduser(os.fspath(path)))` is already the idiom in `annotation.py` and `artifacts.py`. `database_schema` was the outlier, and GitHub issue #108 is what that cost: a macOS user whose settings carried `~` got FileNotFoundError from four frames down.

`migrate_database` keeps its strict contract deliberately: it is the low-level operation, its docstring promises no expansion, and a caller that has already resolved a path should not have it resolved twice.

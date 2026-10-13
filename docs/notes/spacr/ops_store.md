# Notes from `spacr/ops_store.py`

Prose lifted out of `spacr/ops_store.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [write_table](#write_table) (2 entries)
- [read_table](#read_table) (1 entry)

## write_table

### lines 110-121

```python
from .tabular import write_database
```

THROUGH `spacr.tabular.write_database`, the funnel, rather than a bare `to_sql`. `test_the_raw_reader_count_only_goes_down` holds the number of writes and reads outside it to a ceiling that only falls, and the reason is that a reader or writer that does not canonicalise DOES NOT FAIL -- it returns a number.

`canonicalise=False` here, for the same reason `read_table` below passes it: these are OPS tables whose column names ARE the storage contract, and the parquet sidecar written a few lines down spells them the same way. Renaming on the way in would make the two disagree about a table neither had changed, and the row-count check cannot see a renamed column.

### lines 140-143

```python
_remove_stale(path)
```

A CACHE THAT CANNOT BE WRITTEN IS NOT A FAILED RUN. The authority is already on disk and complete; losing the sidecar costs speed. Removing a stale one matters more than creating a new one, because a stale cache is the only way this design can hand back wrong data.

## read_table

### lines 206-219

```python
from .tabular import _read_query
```

THROUGH `spacr.tabular`, not a direct pandas SQL read. Its own `_read_query` docstring describes this caller exactly: somebody who already holds a connection, for whom `read_table` would reopen the database, and who therefore reached for pandas directly -- which is "how a frame with un-canonicalised column names gets into the package: it does not fail, it returns a number".

`canonicalise=False`, and this is the one place that is right. These are OPS tables written by this module from `PlateObject.row()` and the sampling rows, so their names ARE the storage contract. Canonicalising renames `object_id` to `objectID`, which the parquet cache sitting beside the database still spells the old way -- the two would then disagree about a table neither had changed, and the disagreement check above compares ROW COUNTS, so it would not notice.

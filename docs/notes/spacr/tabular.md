# Notes from `spacr/tabular.py`

Prose lifted out of `spacr/tabular.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## _connect

### lines 323-324  _(unsure)_

```python
from .database_concurrency import connect
```

Keep read-only table loads on the same URI escaping, busy-timeout, query-only and connection policy as every other database reader.

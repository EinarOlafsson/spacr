# Notes from `spacr/regression_failure.py`

Prose lifted out of `spacr/regression_failure.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## describe_failure

### lines 130-132

```python
try:
```

WHAT IT COST ON THE WAY (instruction 160). A failure that ran out of memory looks identical to one that did not, unless the readings taken per stage are beside it.

### line 147

```python
return f"THE REGRESSION FAILED: {type(error).__name__}: {error}\n"
```

The reporter must never replace the failure it is reporting.

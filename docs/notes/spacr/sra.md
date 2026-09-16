# Notes from `spacr/sra.py`

Prose lifted out of `spacr/sra.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [runs_for](#runs_for) (1 entry)
- [fetch_reads](#fetch_reads) (2 entries)

## runs_for

### lines 151-153

```python
files.append(RunFile(
```

The portal returns a bare host/path. HTTPS rather than FTP:

FTP is blocked on many institutional networks and is the reason a "download failed" here would be unexplainable.

## fetch_reads

### lines 195-197

```python
part = target.with_suffix(target.suffix + ".part")
```

A PARTIAL FILE IS WORSE THAN NO FILE: it looks like a finished download to everything that lists the folder. Written beside the target and moved only on success.

### lines 222-223

```python
if wanted_lines is None and pending:
```

The tail, only when the whole file was asked for -- a truncated request must not end on half a record.

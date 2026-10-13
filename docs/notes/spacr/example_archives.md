# Notes from `spacr/example_archives.py`

Prose lifted out of `spacr/example_archives.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [explain_download_failure](#explain_download_failure) (2 entries)
- [extract_example_archive](#extract_example_archive) (1 entry)
- [expand_measure_arrays](#expand_measure_arrays) (2 entries)
- [make_the_example_paths_absolute](#make_the_example_paths_absolute) (4 entries)

## Module level

### lines 58-59  _(unsure)_

```python
DATASET_REPO  = "einarolafsson/toxo_mito"
```

Match the classic Tk GUI's demo endpoints so users see the same dataset here they'd have seen in the Tk build.

## explain_download_failure

### lines 256-260

```python
if isinstance(exc, OSError) and "Truncated download" in str(exc):
```

The truncation check comes first: `IOError` IS `OSError`, and the builtin ConnectionError below is an OSError subclass, so ordering these the other way round would let a half-finished transfer be reported as "check your internet connection" — true but useless, because the connection was fine right up to the point it was not.

### lines 268-274

```python
network_errors: tuple = (ConnectionError, TimeoutError, socket.gaierror)
```

requests is an install-time dependency of huggingface_hub, but the import is kept local so a broken environment reports the missing package above rather than dying here. The builtins are in the tuple too: `requests.exceptions.ConnectionError` descends from OSError, not from the builtin ConnectionError, and a DNS failure raised by anything other than requests (urllib, socket, huggingface_hub's own client) arrives as one of these instead.

## extract_example_archive

### lines 430-431

```python
for member in members:
```

No filter available: refuse anything that leaves the tree rather than trusting the archive.

## expand_measure_arrays

### lines 478-479  _(unsure)_

```python
key = "image" if "image" in bundle else bundle.files[0]
```

Written by the publisher under `image`; the first key is the fallback so a hand-made archive still loads.

### lines 484-485

```python
LOG.warning("could not unpack %s", archive, exc_info=True)
```

One bad archive must not cost the other fifteen. It is left on disk, so what failed is visible rather than merely absent.

## make_the_example_paths_absolute

### lines 514-516

```python
for database in (root / "measurements" / "measurements.db",
```

WHEREVER THE DATABASE IS. spaCR keeps it at `measurements/measurements.db` inside a plate; the published archive used to carry it at the top. Both are checked so an already-unpacked older copy is still repaired.

### lines 522-529

```python
connection = connect(database)
```

THE HOUSE CONNECT, for its busy timeout. This ran without one for as long as it lived in `spacr/qt/hf_download.py`, where the concurrency audit does not look. Moving it here put it in scope and the audit caught it immediately: an untimed connection raises "database is locked" the instant a Measure writer holds the file, rather than waiting for it -- and this runs right after an example unpacks, which is exactly when something else may be opening the same database.

### lines 539-541

```python
cursor = connection.execute(
```

Only the values that look like OUR relative paths. A column holding prose is untouched, and one already absolute is skipped by the same test.

### lines 549-551

```python
continue
```

A column that cannot be updated -- a generated one, or a type that will not concatenate -- is not a reason to abandon the other forty.

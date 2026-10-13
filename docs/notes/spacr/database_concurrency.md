# Notes from `spacr/database_concurrency.py`

Prose lifted out of `spacr/database_concurrency.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [transaction](#transaction) (1 entry)
- [_filesystem_type_via_psutil](#_filesystem_type_via_psutil) (2 entries)
- [filesystem_type](#filesystem_type) (1 entry)
- [enable_wal_where_safe](#enable_wal_where_safe) (1 entry)
- [run_concurrency_probe](#run_concurrency_probe) (1 entry)
- [_discard_scratch](#_discard_scratch) (2 entries)
- [_run_probe](#_run_probe) (2 entries)

## Module level

### lines 166-175

```python
MINIMUM_ATTEMPT_BUSY_TIMEOUT_MS = 25
```

Smallest per-attempt busy timeout worth asking SQLite for.

SQLite's default busy handler sleeps down a fixed ladder 1, 2, 5, 10, 15, 20, 25, 25, 25, 50, 50, 100 ms -- and clamps the final sleep to whatever is left of the budget. A 1 ms budget therefore buys one 1 ms sleep and a single re-try of the lock: BEGIN gives up while the holder is still inside its commit, and the caller burns a whole retry attempt on a lock that was about to be released. 25 ms is where the ladder reaches its steady step, so it is the smallest budget in which a contended writer can realistically hand the lock over.

## transaction

### lines 252-257

```python
total_busy_timeout = (
```

sqlite's busy_timeout applies to *each* BEGIN. Without dividing the caller's budget, eight retries on a 30-second connection can block for four minutes. Share that budget across attempts -- clamped by _attempt_busy_timeout_ms, because a plain division handed a 50 ms connection asking for 40 attempts 1 ms per BEGIN -- then restore the connection's own value before executing the transaction body.

## _filesystem_type_via_psutil

### lines 328-335

```python
best: Optional[tuple] = None
```

NO WALK-UP BEFORE MATCHING. A mount point either is a prefix of this path or it is not, and that is true whether or not the leaf exists yet a measurement.db about to be created on a share is still on the share. Walking up to the nearest EXISTING ancestor first sent a path under a share that had no file yet all the way to "/", which matches the root mount and reports the local disk. That is the one wrong answer that matters here: the root is usually apfs, apfs is on WAL_SAFE_FILESYSTEMS, and the result would be WAL enabled on a network share.

### lines 340-341  _(unsure)_

```python
return None
```

Advisory only: a platform that refuses to enumerate mounts leaves the answer unknown, which wal_is_safe_here already treats as unsafe.

## filesystem_type

### lines 368-377

```python
return _filesystem_type_via_psutil(target)
```

NOT LINUX. Until this branch existed the answer here was None on every macOS and Windows machine, and `wal_is_safe_here` turns None into False -- so every Mac ran without WAL even on local APFS, and, worse, `doctor` could not tell a user on an SMB share that they WERE on one. Issue 115 is exactly that reporter: Apple Silicon, a measurement.db on an SMB server, and nothing in spaCR able to name the filesystem in its own diagnosis.

psutil is already a declared dependency and reports fstype on every platform spaCR supports, so this needs no new requirement.

## enable_wal_where_safe

### lines 470-473

```python
return None
```

A refusal here is informative, not fatal: SQLite declines WAL on storage that cannot hold it, which is exactly the outcome the allowlist is guessing at. Staying on DELETE is the shipped behaviour, so the run continues as it always did.

## run_concurrency_probe

### lines 724-735

```python
try:
```

A PROBE THAT NEVER RAN LEAVES NOTHING BEHIND. Everything from here to the metrics is inside one handler, because a failure anywhere in it happens AFTER the scratch database has been created and the cleanup used to be the last statement of the function. The commonest is a journal mode `connect` refuses -- 'MEMORY', 'TRUNCATE' -- which left an empty scratch database in the system temp directory for good, and at an explicit path left a file that makes the NEXT run on it fail with FileExistsError against a database the user never got a probe out of.

The deliberate survivor is the STALLED one: a worker that outlives the join deadline is a normal return, guarded by `not alive` at the end, and its database is worth keeping to look at.

## _discard_scratch

### line 757  _(unsure)_

```python
for suffix in ("", "-wal", "-shm"):
```

An explicit path, with the sidecars WAL leaves beside it.

### line 763, trailing  _(unsure)_

```python
except Exception:
```

below OSError: a path the OS rejects outright

## _run_probe

### lines 867-868

```python
finished.set()
```

Release reader loops even if a writer stalled, then give all workers one final bounded chance to close their thread-owned connection.

### lines 873-876

```python
survivors = []
```

``is_alive`` can change between the post-join snapshot above and this final check. Keep only workers that are still alive now, so a thread that exits in that small window is neither reported as stalled nor used to preserve an otherwise disposable scratch database.

# Notes from `spacr/errors.py`

Prose lifted out of `spacr/errors.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [SpacrError](#spacrerror) (1 entry)
- [Failure](#failure) (1 entry)
- [RunLedger.record_failure](#runledgerrecord_failure) (1 entry)
- [RunLedger.item](#runledgeritem) (2 entries)
- [RunLedger.finalize](#runledgerfinalize) (1 entry)
- [read_run_status](#read_run_status) (2 entries)

## SpacrError

### lines 120-122

```python
class SpacrError(Exception):
```

Exception types

## Failure

### lines 178-180

```python
@dataclass(frozen=True)
```

Failure record

## RunLedger.record_failure

### lines 359-361

```python
self._log.debug('[%s] traceback for %s:\n%s',
```

DEBUG, not ERROR: forty failures would otherwise dump forty tracebacks over the console. The full text is kept on the Failure and persisted by stamp(), which is the durable record.

## RunLedger.item

### line 399

```python
raise
```

Setup is wrong for every item — surviving is not an option.

### line 402  _(unsure)_

```python
raise
```

Operator intent, not a data problem.

## RunLedger.finalize

### lines 519-521

```python
self._log.error('[%s] RUN INCOMPLETE — %d of %d items failed '
```

One-line log record, full block on stdout: logging the whole block too rendered the summary twice in a plain terminal session (logging's last-resort handler writes to stderr).

## read_run_status

### lines 712-713

```python
return []
```

Never stamped. The artifact predates stamping, or was written by a code path that does not stamp yet.

### lines 740-742

```python
raise RunStatusUnreadable(
```

A sidecar half-written by an interrupted run is the same species of evidence as a locked database, and gets the same answer: unknown, never "complete".

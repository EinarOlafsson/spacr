# Notes from `spacr/runctx.py`

Prose lifted out of `spacr/runctx.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [new_run_id](#new_run_id) (1 entry)
- [read_run_log](#read_run_log) (1 entry)
- [_RunLogHandler.emit](#_runloghandleremit) (1 entry)
- [SeedReport](#seedreport) (1 entry)
- [resolve_seed](#resolve_seed) (1 entry)
- [seed_everything](#seed_everything) (3 entries)
- [_SkippedType](#_skippedtype) (1 entry)
- [ErrorPolicy.attempts_for](#errorpolicyattempts_for) (1 entry)
- [_performance_logging_preference](#_performance_logging_preference) (1 entry)
- [_register_settings](#_register_settings) (1 entry)

## new_run_id

### lines 203-205  _(unsure)_

```python
def new_run_id() -> str:
```

S7 — the run id, and getting it onto every log line

## read_run_log

### lines 407-408  _(unsure)_

```python
continue
```

A run killed mid-write leaves a half line. Everything before it is still perfectly good evidence.

## _RunLogHandler.emit

### lines 520-521

```python
self.handleError(record)
```

A full disk must not take the run down with it: the run log is evidence, not the result.

## SeedReport

### lines 536-538  _(unsure)_

```python
@dataclass(frozen=True)
```

S5 — one seed, and an honest account of where it does not reach

## resolve_seed

### lines 648-650

```python
return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
```

A word rather than a number: hash it, so `random_seed:

"plate3-rerun"` is a usable, reproducible seed rather than a crash or a silent fall back to 42.

## seed_everything

### lines 690-691

```python
narrow = int(value) % (2 ** 32)
```

Python's random and NumPy's legacy seeder want different ranges; 2**32 is the narrower of the two, so normalise once.

### line 726

```python
LOG.debug("could not seed CUDA: %s", exc)
```

A driver mismatch must not stop a CPU run from being seeded.

### lines 741-742  _(unsure)_

```python
seeded.append("cellpose(via numpy+torch)")
```

Nothing to call — recorded so the report does not read as though Cellpose was overlooked.

## _SkippedType

### lines 855-857  _(unsure)_

```python
class _SkippedType:
```

S9 — on_error: stop | skip | retry

## ErrorPolicy.attempts_for

### lines 1136-1137

```python
return
```

The body never ran, or `break`/`continue` skipped the

`with`. Nothing to judge; leave the loop alone.

## _performance_logging_preference

### lines 1448-1449  _(unsure)_

```python
return None
```

``None`` asks the sampler to resolve the environment itself and preserve the fact that the environment, not a preference, won.

## _register_settings

### lines 1748-1749

```python
LOG.debug("run-control settings not registered: %s", exc)
```

Another module already declared one of these. Say so once rather than take the import of spacr.runctx down with it.

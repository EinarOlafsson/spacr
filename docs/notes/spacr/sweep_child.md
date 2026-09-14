# Notes from `spacr/sweep_child.py`

Prose lifted out of `spacr/sweep_child.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [main](#main) (3 entries)

## Module level

### lines 17-21

```python
try:
```

Volunteer for the OOM killer, before importing anything large. Same reason as spacr.parameter_sweep.be_polite: left alone the kernel scores by resident size and kills the biggest process on the box, which during a sweep is the user's editor and not this child. Repeated here because a contained trial is exec'd into a fresh interpreter and never runs be_polite.

### line 25, trailing  _(unsure)_

```python
except OSError:
```

not Linux, or not permitted

## main

### lines 73-75

```python
"_resource_worker": _worker_stamp(
```

The parent removes this private transport field before writing the sweep table. PID plus creation time lets the run sampler attach the trial name to samples it took while this short-lived child existed.

### lines 85-86

```python
_pin_threads()
```

Belt as well as braces: the environment above is read at import, this resizes the pool that already exists.

### lines 97-102

```python
controls = payload.get("controls") or {}
```

The caller's own control ALIASES, on top of the canonical positive_control_* columns. The sweep screen puts `positive_rank` in its table, and that column is built from this mapping -- so a contained trial that did not compute it would leave the one column the run is judged on blank, which looks exactly like a control that was never recovered.

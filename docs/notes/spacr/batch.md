# Notes from `spacr/batch.py`

Prose lifted out of `spacr/batch.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [inprocess_runner](#inprocess_runner) (1 entry)
- [_status_snapshot](#_status_snapshot) (1 entry)
- [Module level](#module-level) (1 entry)
- [_safe](#_safe) (1 entry)
- [run_queue.persist](#run_queuepersist) (1 entry)
- [run_queue](#run_queue) (2 entries)
- [_blocking_dependency](#_blocking_dependency) (1 entry)

## inprocess_runner

### line 1375, trailing  _(unsure)_

```python
argv = job_command(job, settings_path)[3:]
```

drop python -m spacr.cli

## _status_snapshot

### line 1423, trailing

```python
except Exception:
```

a locked or corrupt artifact must not stop the queue

## Module level

### line 1488

```python
_EXC_LINE = re.compile(r'^(?P<type>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Interrupt))'
```

failure classification

## _safe

### line 1568, trailing

```python
except Exception:
```

a GUI callback must never kill an overnight run

## run_queue.persist

### line 1661, trailing

```python
except Exception as exc:
```

a persistence problem must not end the night

## run_queue

### line 1683, trailing  _(unsure)_

```python
continue
```

a resumed queue: already settled

### line 1751, trailing

```python
except Exception as exc:
```

the runner itself broke; that is a job failure

## _blocking_dependency

### line 1854, trailing

```python
continue
```

validate_queue already reported this as an error

# Notes from `spacr/projects.py`

Prose lifted out of `spacr/projects.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [producing_modules](#producing_modules) (1 entry)
- [module_states](#module_states) (1 entry)
- [looks_like_project](#looks_like_project) (2 entries)
- [discover._walk](#discover_walk) (1 entry)
- [ProjectSummary.stage_label](#projectsummarystage_label) (1 entry)
- [_stale_of](#_stale_of) (1 entry)
- [scan](#scan) (6 entries)
- [browse](#browse) (1 entry)

## producing_modules

### lines 148-150  _(unsure)_

```python
def producing_modules() -> Tuple[str, ...]:
```

The pipeline order, read off the port graph

## module_states

### lines 383-385

```python
alive = [row for row in rows if row.exists]
```

The registry watched this run finish. Whether its outputs are still on disk is a different question, and the one PARTIAL is for.

## looks_like_project

### lines 430-431  _(unsure)_

```python
try:
```

No output anywhere: is there raw data waiting to have something run on it? The mask pipeline's own input declaration answers that.

### lines 434-435  _(unsure)_

```python
except _ports.UnknownModule:
```

'mask' is declared by spaCR itself; a build that dropped it, or a plugin registry that replaced PORTS, is what this guards against.

## discover._walk

### line 494, trailing  _(unsure)_

```python
except OSError:
```

a mount that went away

## ProjectSummary.stage_label

### line 623, trailing  _(unsure)_

```python
return self.stage
```

a summary with no states

## _stale_of

### line 713, trailing  _(unsure)_

```python
except Exception as exc:
```

corrupt row

## scan

### line 778, trailing  _(unsure)_

```python
except _dm.DataManagerError as exc:
```

raced delete

### lines 783-787

```python
LOG.debug("cannot measure %s: %s", project, exc)
```

The walk reconciles against the registry, so a corrupt or half-written ``artifacts.db`` breaks the measurement rather than the folder. The project is still there and still worth listing: its bytes go unmeasured, with the reason recorded, instead of the whole row -- or the whole browse -- vanishing.

### lines 792-794

```python
try:
```

The registry first: it is the authority on what ran, and

`module_states` takes it into account rather than guessing from files alone wherever it has an answer.

### lines 798-799  _(unsure)_

```python
LOG.debug("cannot open the registry for %s: %s", project, exc)
```

Opening it is as fallible as reading it, and a browser that raises here shows no list at all.

### line 806, trailing  _(unsure)_

```python
except Exception as exc:
```

locked db

### lines 824-826

```python
last_ns = max((state.newest_ns for state in states), default=0)
```

Nothing recorded. The outputs themselves still carry a date, and it is a weaker claim reported as a weaker claim rather than dressed up as a run record.

## browse

### line 875, trailing  _(unsure)_

```python
except Exception:
```

caller's bug

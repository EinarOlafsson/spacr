# Notes from `spacr/workspace.py`

Prose lifted out of `spacr/workspace.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [register](#register) (1 entry)
- [_state_of](#_state_of) (1 entry)
- [section_states](#section_states) (1 entry)
- [save](#save) (1 entry)
- [restore](#restore) (1 entry)

## register

### lines 123-125

```python
def register(name: str, provider: Callable[[], Any]) -> None:
```

The registry — how GUI state reaches a journal that cannot import Qt

## _state_of

### lines 186-188

```python
legacy = getattr(source, "plot_state", None)
```

The regression panel already had this pair before the workspace existed. Taking it as-is keeps ONE state model for the volcano rather than a second one that has to be remembered alongside it.

## section_states

### lines 214-217

```python
if source is None:
```

NOT OPEN IS NOT A PROBLEM. Every screen registers the same panels and most screens build none of them -- a measure screen has no volcano. Reporting those would bury the one section that genuinely failed under a dozen that were simply not there.

## save

### lines 511-514

```python
if mode == "copy" and not record.get("carry") and size > limit_bytes:
```

The limit bounds `copy`; a file a section asked to CARRY is carried whatever its size, because the section is asserting it exists nowhere else and a silently dropped figure is worse than a large run folder.

## restore

### lines 699-701

```python
if applied is False:
```

`False` is an ANSWER, not a failure: a panel with no table yet cannot take a plot state and says so. It is reported as skipped because from the user's side nothing was put back either way.

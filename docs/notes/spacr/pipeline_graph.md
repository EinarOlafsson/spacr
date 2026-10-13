# Notes from `spacr/pipeline_graph.py`

Prose lifted out of `spacr/pipeline_graph.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_layer](#_layer) (1 entry)
- [build_graph](#build_graph) (1 entry)
- [_staleness](#_staleness) (1 entry)
- [format_graph](#format_graph) (1 entry)

## _layer

### lines 423-424  _(unsure)_

```python
base = max(depth.values(), default=-1) + 1
```

Everything left is in (or downstream of) a cycle. Park it one column past the deepest thing that resolved.

## build_graph

### line 485, trailing

```python
except OSError as exc:
```

rare, but it is a race

## _staleness

### line 576, trailing  _(unsure)_

```python
except Exception:
```

the registry raced us

## format_graph

### line 679, trailing  _(unsure)_

```python
if node is None:
```

layers normally come from nodes

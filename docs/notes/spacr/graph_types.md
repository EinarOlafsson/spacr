# Notes from `spacr/graph_types.py`

Prose lifted out of `spacr/graph_types.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [shape_of](#shape_of) (1 entry)
- [default_for](#default_for) (2 entries)

## Module level

### lines 55-57

```python
"continuous_continuous": ("scatter",),
```

NO BAR AND NO BOX HERE. Both need groups to summarise, and forming groups out of a continuous x means binning it -- which is a different graph of different data, not this one drawn another way.

### lines 59-60

```python
"ordered_continuous": ("line", "scatter", "jitter"),
```

A LINE NEEDS AN ORDER. Through unordered categories it is a row of markers joined for no reason, which is why it is here and not above.

## shape_of

### lines 157-159

```python
try:
```

ORDERED IS A PROPERTY OF THE VALUES, not of the dtype. An x that is already sorted and unique is a series; one that is neither is a cloud, and joining a cloud with a line is 178 A's bug.

## default_for

### line 204, trailing  _(unsure)_

```python
fallback = DEFAULTS[shape]
```

KeyError for a bad shape

### lines 210-211

```python
return fallback
```

No Qt, no stored preferences, or a preference file that cannot be read: a figure still has to be drawn.

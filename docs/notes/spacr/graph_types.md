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


---

# Notes from `spacr/graph_types.py`

Prose lifted out of `spacr/graph_types.py` by `tools/extract_source_notes.py`.
Ordinary comments move here; tool directives and published attribute documentation stay in the module. The path mirrors the source path, which is how its reasons are found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [chosen_for](#chosen_for) (1 entry)

## Module level

### lines 25-34

```python
_READ_THE_PREFERENCE_STORE = ContextVar(
```

WHETHER `chosen_for` MAY ASK THE QT PREFERENCE STORE. Reading it imports PySide6.QtCore. `spacr.validate._known_setting_keys` calls every settings default with `{}` only to learn which KEYS exist, and since 293 three of those defaults ask this module for a graph type. That sweep has no use for the user's choice. It runs when a batch queue is validated, which must not import Qt (tests/test_batch.py). So the sweep turns this off, and `chosen_for` answers "nothing was chosen", which every default already falls back to. A ContextVar rather than a module flag, so another thread asking at the same moment still reads the real preference. Everything else, including a headless pipeline run, reads it exactly as before.

## chosen_for

### line 277

```python
if not _READ_THE_PREFERENCE_STORE.get():
```

Checked BEFORE the import, because the import is the cost being avoided.

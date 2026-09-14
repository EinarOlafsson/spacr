# Notes from `spacr/qt/widgets/__init__.py`

Prose lifted out of `spacr/qt/widgets/__init__.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [__getattr__](#__getattr__) (1 entry)

## Module level

### lines 31-33

```python
from .hover_tooltip import HoverTooltip
```

Kept eager because it is a cheap, public widget. QSS registration is no longer a reason to import a heavy widget at launch: the screen host scopes any late registered block to the screen before its first paint.

## __getattr__

### line 75, trailing  _(unsure)_

```python
globals()[name] = value
```

cached; this runs once per name

# Notes from `spacr/qt/screens/curate.py`

Prose lifted out of `spacr/qt/screens/curate.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [CurateScreen.__init__](#curatescreen__init__) (2 entries)
- [register](#register) (1 entry)

## CurateScreen.__init__

### lines 61-62  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

### lines 65-67

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## register

### lines 274-281

```python
key, APP_NAME, APP_DESCRIPTION, section or SECTION_TOOLS,
```

TOOLS, not Core. Curate fixes a mask by hand; Core is the pipeline you run, and a section that lists everything sorts nothing.

It asked for SECTION_MODELS until 2026-09-03, which is still

DEFINED and still described but was dropped from SECTION_ORDER when Home was restructured to Core / Data / Tools / Assays. So every call to this function raised "app 'curate' has unknown section 'Segmentation models'" and the screen could not register at all.

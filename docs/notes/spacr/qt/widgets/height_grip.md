# Notes from `spacr/qt/widgets/height_grip.py`

Prose lifted out of `spacr/qt/widgets/height_grip.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [HeightGrip.__init__](#heightgrip__init__) (1 entry)
- [HeightGrip._follow_the_font](#heightgrip_follow_the_font) (1 entry)

## HeightGrip.__init__

### lines 85-87

```python
self.setFocusPolicy(Qt.StrongFocus)
```

THE HANDLE IS A TAB STOP. Everything this widget offers is otherwise reachable only with a pointer, and a nested container a keyboard user cannot enlarge is the one that stays too small to read.

## HeightGrip._follow_the_font

### lines 139-140

```python
base = (int(round(self.target_height() / max(was, 0.01)))
```

IN BASE PX, taken BEFORE `rescale` moves the clamp: converting after would divide a height by the new scale and keep the old pixels.

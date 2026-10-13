# Notes from `spacr/qt/widgets/foldable.py`

Prose lifted out of `spacr/qt/widgets/foldable.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Folder.__init__](#folder__init__) (1 entry)
- [Folder.set_shut](#folderset_shut) (1 entry)
- [Folder._refresh_tooltip](#folder_refresh_tooltip) (1 entry)

## Folder.__init__

### lines 102-105

```python
heading.setProperty("i18nSkipText", True)
```

A HEADING IS COMPOSED, NOT WRITTEN. The line reads as an arrow, the panel name and sometimes an alert, so asking the catalog for the finished line asks for a key that cannot exist. Keep the generic language pass off it and rebuild it from the translated parts.

## Folder.set_shut

### lines 136-138

```python
self._alert = ""
```

ARRIVING IS SEEING. An alert that survived the unfold would keep claiming there is something to look at after the user has looked at it.

## Folder._refresh_tooltip

### lines 166-168

```python
"""Write the heading's hover text.
```

THE NAME LOOKS CLICKABLE BEFORE IT IS CLICKED. A gesture nobody knows about is not a feature, and the pointer is the only hint a heading can carry without a second widget beside it.

# Notes from `spacr/qt/widgets/drawer.py`

Prose lifted out of `spacr/qt/widgets/drawer.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [EdgeDrawer.__init__](#edgedrawer__init__) (2 entries)
- [EdgeDrawer.set_enabled](#edgedrawerset_enabled) (1 entry)

## EdgeDrawer.__init__

### lines 109-111

```python
self.move(-self._width, 0)
```

Start fully off-screen to the left. Not hidden: a hidden widget reports no geometry, and the tutorial overlay (which highlights the sidebar) needs a rectangle to point at.

### lines 130-132

```python
self._trigger = _EdgeTrigger(host, self)
```

The hot strip. A separate zero-chrome child so the drawer's own geometry can stay off-screen while something on-screen still receives the hover.

## EdgeDrawer.set_enabled

### line 160  _(unsure)_

```python
def set_enabled(self, enabled: bool) -> None:
```

the preference: locked, hidden, or the reveal


---

# Notes from `spacr/qt/widgets/drawer.py`

Prose lifted out of `spacr/qt/widgets/drawer.py` by `tools/extract_source_notes.py`.
Ordinary comments move here; tool directives and published attribute documentation stay in the module. The path mirrors the source path, which is how its reasons are found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## EdgeDrawer.eventFilter

### lines 332-334

```python
host = getattr(self, "_host", None)
```

getattr: the drawer and its host are a reference cycle, so the collector can clear this wrapper before the host's destructor reaches the filter.

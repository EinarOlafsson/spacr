# Notes from `spacr/qt/screens/lineage.py`

Prose lifted out of `spacr/qt/screens/lineage.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [LineageScreen.__init__](#lineagescreen__init__) (1 entry)
- [LineageScreen._build](#lineagescreen_build) (1 entry)
- [LineageScreen._fill_orphans](#lineagescreen_fill_orphans) (1 entry)
- [Module level](#module-level) (1 entry)

## LineageScreen.__init__

### lines 104-105  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

## LineageScreen._build

### lines 192-193  _(unsure)_

```python
mark_surface(self.tree, self.orphan_list)
```

Both halves of the splitter sit straight on the page; the splitter itself is scaffolding and paints nothing.

## LineageScreen._fill_orphans

### lines 295-298

```python
key = lin.node_key(row, str(row.get("table") or "") or None)
```

`orphans` stamps each row with the table it came from, so the key an unattached child publishes says which child it is — the same identity the tree uses, rather than one that names every object with that label in the field.

## Module level

### lines 519-523

```python
_ROW = declared_app(APP_KEY)
```

The row this screen puts in the registry is declared in

`spacr.qt.app_catalog`, which is what lets the app be registered without importing this module -- the launch reads the table, not the screen. These read the same row back rather than restating it, so the name, the blurb and the nine translations have one spelling and no second copy to drift from.

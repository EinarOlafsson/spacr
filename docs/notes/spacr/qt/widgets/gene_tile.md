# Notes from `spacr/qt/widgets/gene_tile.py`

Prose lifted out of `spacr/qt/widgets/gene_tile.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [GeneTilePanel.__init__](#genetilepanel__init__) (1 entry)
- [GeneTilePanel.show_feature](#genetilepanelshow_feature) (1 entry)

## GeneTilePanel.__init__

### lines 86-87

```python
self._view.setProperty("i18nSkipText", True)
```

A gene id must survive translation intact: TGGT1_239740 is not a phrase, and a catalog that "translated" it would be renaming a gene.

## GeneTilePanel.show_feature

### lines 143-146

```python
LOG.exception("gene tile: could not reach the results frame")
```

A BROKEN HOST, NOT A BROKEN TILE. The provider belongs to whatever screen owns this panel; if it raises, the tile still draws from no frame rather than letting the host's failure out through a click on a plot point.

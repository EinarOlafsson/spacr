# Notes from `spacr/qt/widgets/annotation_umap_tab.py`

Prose lifted out of `spacr/qt/widgets/annotation_umap_tab.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [PurityScatter.set_embedding](#purityscatterset_embedding) (1 entry)
- [AnnotationUmapTab.__init__](#annotationumaptab__init__) (2 entries)
- [AnnotationUmapTab.run](#annotationumaptabrun) (1 entry)
- [AnnotationUmapTab._score](#annotationumaptab_score) (2 entries)

## PurityScatter.set_embedding

### lines 128-130

```python
self.set_status(f"Purity is not drawn as a colour: {exc}")
```

Every cell shares one purity, or none of them has a finite one. The scatter is still worth showing; what is not true is that the colour means anything, so it is said rather than implied.

## AnnotationUmapTab.__init__

### lines 180-182

```python
from ...cell_montage import PICKING_MODES
```

THE ONES THIS CHECK CAN SPEAK ABOUT. `rank` is absent on purpose: it takes the top-scoring cells in the well, so its cells sitting near the positive controls restates how it chose them.

### lines 193-196

```python
self.body = QSplitter(Qt.Horizontal)
```

The plot and the table are two views of ONE result, so they sit side by side behind a divider the user owns: the picture says where the cells landed, the table says by how much, and reading one against the other is the whole job.

## AnnotationUmapTab.run

### lines 257-259

```python
self.refuse(
```

SHOWN INSTEAD OF THE PLOT, not beside it. A picture drawn under a warning that it means nothing is still a picture somebody will screenshot.

## AnnotationUmapTab._score

### lines 343-347

```python
groups, level = self._control_groups(control_rows)
```

HOLD THE WELLS APART WHERE THE FRAME NAMES THEM. Sibling control cells on both sides of the split would separate because they came from the same well, and the held-out silhouette would report that as biology. A frame that cannot name a well is split per object, which the result records rather than hiding.

### lines 361-363

```python
self.refuse(
```

REFUSED, WITH THE NUMBERS. This is the guard that matters: a search that separates only the half it was tuned on has found the split, not the biology.

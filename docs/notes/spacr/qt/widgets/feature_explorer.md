# Notes from `spacr/qt/widgets/feature_explorer.py`

Prose lifted out of `spacr/qt/widgets/feature_explorer.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## FeatureExplorerPanel.__init__

### line 115

```python
self._statistic.setToolTip("\n\n".join(
```

The blind spots, on screen rather than in a manual.

### lines 157-159

```python
mark_surface(self.table)
```

`FeatureExplorerPanel` is transparent scaffolding by design

(see the GraphBuilder block), so the ranking table is the page here; the distribution canvas beside it paints its own panel.

### lines 167-169

```python
self._figure = Figure(figsize=(5.0, 6.0))
```

No `facecolor`: the canvas paints the page panel in its own

`paintEvent` under a transparent figure patch, so a solid one here would put the opaque rectangle straight back.

### lines 187-189

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

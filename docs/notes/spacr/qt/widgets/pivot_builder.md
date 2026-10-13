# Notes from `spacr/qt/widgets/pivot_builder.py`

Prose lifted out of `spacr/qt/widgets/pivot_builder.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [DropWell.__init__](#dropwell__init__) (1 entry)
- [PivotTable.set_result](#pivottableset_result) (1 entry)
- [PivotPanel.__init__](#pivotpanel__init__) (3 entries)
- [PivotPanel.recompute](#pivotpanelrecompute) (1 entry)
- [Module level](#module-level) (1 entry)

## DropWell.__init__

### line 143

```python
apply_close_mark(clear, tooltip=f"Empty the {AXIS_LABELS[axis]} well")
```

THE APPLICATION'S CLOSE MARK -- see `theme.apply_close_mark`.

## PivotTable.set_result

### lines 368-370

```python
self._truncated = result.n_cells
```

Build the shape but not the contents: a QTableWidget with a million items takes minutes to construct, for a table nobody is going to scroll to the end of.

## PivotPanel.__init__

### lines 532-534

```python
box.setEnabled(False)
```

n is not a choice. Every cell carries it, and a table where the user could turn it off is a table where a mean over four objects looks like a mean over four thousand.

### lines 575-576  _(unsure)_

```python
mark_surface(self.table)
```

The shelf half of the splitter has `PivotShelf` for a surface; the grid is the other half and sits straight on the page.

### lines 592-594

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## PivotPanel.recompute

### lines 686-689

```python
LOG.info("the pivot failed", exc_info=True)
```

ANYTHING THAT IS NOT A PivotError. That one is the expected refusal and carries its own explanation; this is a fault inside the pivot, and the "could not build that table" wrapper is what tells the two apart on screen.

## Module level

### lines 800-807

```python
register_widget_qss("Pivot", _pivot_qss, replace=True)
```

Registered at import of this module, which happens when the screen module is imported — and the row that does that lives in ``app.py``'s ``_SELF_REGISTERING_APPS``, whose loop runs while ``app.py`` itself is being imported. That is before ``launch()`` calls ``stylesheet()``, which is the deadline: a block registered after the stylesheet is built is missing from the one the application was actually given. `spacr.qt.widgets.__init__` imports `graph_builder` eagerly for exactly this reason; this module needs no such entry only because its screen is imported earlier still.

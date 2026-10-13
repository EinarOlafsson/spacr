# Notes from `spacr/qt/widgets/object_grid_binding.py`

Prose lifted out of `spacr/qt/widgets/object_grid_binding.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ObjectGridBinding.__init__](#objectgridbinding__init__) (1 entry)
- [ObjectGridBinding.owned_keys](#objectgridbindingowned_keys) (1 entry)
- [ObjectGridBinding.seed](#objectgridbindingseed) (1 entry)
- [ObjectGridBinding._show_what_the_panel_holds](#objectgridbinding_show_what_the_panel_holds) (1 entry)
- [ObjectGridBinding.write_through](#objectgridbindingwrite_through) (1 entry)

## ObjectGridBinding.__init__

### lines 70-76

```python
self._busy = False
```

REENTRANCY. Writing a value into a widget makes that widget emit, and a screen that reseeds the grid on every widget change would then rebuild the table under the cursor that is still in a cell. NO VALUE depends on this: `write_through` reads the grid once, before the first widget moves, so a reseed halfway cannot drop an edit. What the guard buys is that the table does not visibly rebuild, and the cell being typed into keeps its focus.

## ObjectGridBinding.owned_keys

### lines 93-97

```python
owned = set()
```

READ FROM THE GRID, NOT FROM THE SETTINGS. The grid draws only the questions every object asks, and only the organelle slots the count asks for. Claiming a key it does not show would hide that setting from the form as well, and it would then be reachable from nowhere at all -- which is worse than either place on its own.

## ObjectGridBinding.seed

### lines 120-122

```python
self.follow_the_form()
```

AFTER THE TABLE EXISTS, not before. `follow_the_form` connects the widgets behind the cells the grid is SHOWING, and before the first seed it is showing none.

## ObjectGridBinding._show_what_the_panel_holds

### lines 193-198

```python
value = current[key]
```

SPLIT BY THE TABLE'S OWN RULE. `cell_mask_dim` divides into

`cell` and `mask_dim`, but `organelleb_min_area` has to divide at the longest object prefix rather than the first underscore. Asking `to_table` for a one-key dict gets exactly the split the grid itself was built with, instead of a second rule beside it that could disagree.

## ObjectGridBinding.write_through

### lines 225-228

```python
before = self._panel.collect()
```

READ BOTH SIDES FIRST. Writing a widget makes it emit, which a screen may answer by reseeding the grid; taking the grid's answers as a snapshot here means the rest of the write proceeds from what the user actually typed rather than from a table reloaded halfway.

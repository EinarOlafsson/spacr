# Notes from `spacr/qt/widgets/formula_editor.py`

Prose lifted out of `spacr/qt/widgets/formula_editor.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [FormulaPanel.__init__](#formulapanel__init__) (3 entries)
- [FormulaDialog.__init__](#formuladialog__init__) (2 entries)

## FormulaPanel.__init__

### lines 154-159

```python
self._status.setSizePolicy(QSizePolicy.Preferred,
```

A WRAPPED LABEL NEEDS (Preferred, Minimum): with Qt's default Preferred height a parent is free to hand it less than its heightForWidth. This is the house rule `prerun._label` documents. NECESSARY BUT NOT SUFFICIENT HERE -- 350's sweep still reports this label clipped at 2.0x, because the container above it does not grow either. See 350; the remaining fix is the dialog's layout, not this.

### lines 188-193

```python
self._help.setSizePolicy(QSizePolicy.Preferred,
```

A WRAPPED LABEL NEEDS (Preferred, Minimum): with Qt's default Preferred height a parent is free to hand it less than its heightForWidth. This is the house rule `prerun._label` documents. NECESSARY BUT NOT SUFFICIENT HERE -- 350's sweep still reports this label clipped at 2.0x, because the container above it does not grow either. See 350; the remaining fix is the dialog's layout, not this.

### lines 206-208

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## FormulaDialog.__init__

### lines 487-493

```python
self.resize(scaled_px(560), scaled_px(460))
```

SIZED IN SCALED PIXELS, NOT RAW ONES. A dialog size set from

Python does not grow when the stylesheet's font size does, so at the 200%% font scale the prose inside this window wrapped to more height than the window had and the last line was cut off. The size-policy fix on the label was necessary and not sufficient: a policy stops a parent handing a label less than it asks for, but it cannot make a window grow that has no room to give.

### lines 495-497

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

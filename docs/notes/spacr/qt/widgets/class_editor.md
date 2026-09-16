# Notes from `spacr/qt/widgets/class_editor.py`

Prose lifted out of `spacr/qt/widgets/class_editor.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [ClassChip.__init__](#classchip__init__) (1 entry)
- [ClassEditorWidget.__init__](#classeditorwidget__init__) (5 entries)
- [ClassEditorWidget.set_frame](#classeditorwidgetset_frame) (1 entry)
- [ClassEditorWidget.set_value](#classeditorwidgetset_value) (1 entry)
- [ClassEditorWidget.populate_from_column](#classeditorwidgetpopulate_from_column) (1 entry)
- [ClassEditorWidget._rebuild](#classeditorwidget_rebuild) (1 entry)
- [ClassEditorWidget._on_item_changed](#classeditorwidget_on_item_changed) (1 entry)
- [ClassEditorWidget._say](#classeditorwidget_say) (1 entry)

## Module level

### lines 23-27

```python
import pandas as pd
```

PANDAS IS NOT NEEDED TO RUN THIS FILE. Both mentions are annotations, and `from __future__ import annotations` above makes those strings but the plain import still ran, and it cost 0.365 s of a 1.5 s main window, because the Home page reaches this module through the settings model. Nothing here calls pandas; nothing here should import it.

## ClassChip.__init__

### line 137

```python
apply_close_mark(self._close,
```

THE APPLICATION'S CLOSE MARK -- see `theme.apply_close_mark`.

## ClassEditorWidget.__init__

### lines 202-207

```python
self.column.setEditable(True)
```

EDITABLE, because the combo is filled from a LOADED TABLE and there is not always one. With no frame the list came back empty, the "Add values" button was disabled, and a non-editable empty combo left no way at all to name a column -- so no class could be added and the module could not be configured. Typing a name is the fallback; the SQL button below is the answer when a database is there to ask.

### lines 217-222

```python
entry = QHBoxLayout()
```

TWO FIELDS, SIDE BY SIDE -- the gesture the maintainer asked for: "2 fields next to each other with class then value". Typing a class and its value and pressing Enter in either field adds one chip, so the whole interaction is two words and a keystroke, and it is the SAME in metadata mode and annotation mode. Only the columns the picker above offers differ between the two bases.

### line 245  _(unsure)_

```python
self.chips_host = QWidget(self)
```

The bubbles themselves.

### lines 253-255

```python
self.table = QTreeWidget(self)
```

The table stays, hidden, as the accessible/edit-a-name surface and because every existing test and integration reads `self.table`. Removing it would be a second change riding on this one.

### lines 292-294

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## ClassEditorWidget.set_frame

### lines 309-312

```python
if current:
```

A NAME TYPED OR PICKED STAYS PUT. `set_frame` runs again whenever the basis changes or a table is attached, and clearing the combo used to throw away a column the user had already named -- silently, because an empty combo looks the same as one nobody has touched.

## ClassEditorWidget.set_value

### lines 397-399

```python
for name in value:
```

The old shape: names with nothing saying what they select. Shown as named rows with no value, so the user can see what has to be filled in rather than finding the table empty.

## ClassEditorWidget.populate_from_column

### lines 449-450

```python
self._rules.append(ClassRule(name=f"{column}={_key(value)}",
```

`_key` for the label too, so a float 1.0 reads as "1" -- the name is what the user sees in every report afterwards.

## ClassEditorWidget._rebuild

### lines 544-546

```python
item.setFlags(item.flags() | Qt.ItemIsEditable)
```

Only the NAME is editable. The value and its column are facts about the table, and letting them be typed over would produce a class that selects nothing with no sign of why.

## ClassEditorWidget._on_item_changed

### lines 569-570

```python
self.table.blockSignals(True)
```

A class with no name cannot be trained on or reported, so the old one is put back rather than accepted and failing later.

## ClassEditorWidget._say

### lines 613-617

```python
self._hint.setProperty("_spacr_i18n_text_template", None)
```

Two things would otherwise rewrite this line on a language pass: the source a fixed sentence leaves behind, which would come back over this one, and the general label walk, which translates known words wherever it finds them. A fixed sentence set later still retranslates -- its template is consulted before the opt-out.

# Notes from `spacr/qt/widgets/screen_data_picker.py`

Prose lifted out of `spacr/qt/widgets/screen_data_picker.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## ScreenDataPicker.__init__

### lines 71-72

```python
from ...screen_data import published_archives
```

One lookup for the whole dialog. None means "could not tell", which is deliberately not the same as "nothing is published".

### lines 83-84

```python
self._list.setSelectionMode(QListWidget.MultiSelection)
```

A tick and a highlight say the same thing, which is what makes the selection legible at a glance rather than only on close inspection.

### lines 88-90

```python
continue
```

Filtered rather than greyed: a Feature download that listed eight rows and refused four of them would be four chances to start a 30 GB transfer by mistake.

### lines 98-100

```python
item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
```

Disabled, because ticking it could only fail. Left visible so the set still reads as eight pieces with one not ready, rather than as a set that never had it.

### lines 104-106

```python
item.setToolTip("Already downloaded. Tick it to fetch it "
```

ALREADY ON DISK. Left selectable rather than disabled: a re-download is how a truncated or edited copy gets repaired, and a row that cannot be ticked gives no way to do that.

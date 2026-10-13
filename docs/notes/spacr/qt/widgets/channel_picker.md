# Notes from `spacr/qt/widgets/channel_picker.py`

Prose lifted out of `spacr/qt/widgets/channel_picker.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ChannelPicker.__init__](#channelpicker__init__) (1 entry)
- [ChannelPicker._on_toggled](#channelpicker_on_toggled) (2 entries)
- [ChannelPicker](#channelpicker) (1 entry)

## ChannelPicker.__init__

### lines 93-94  _(unsure)_

```python
box = Toggle(LABELS[name], self)
```

`Toggle`, which subclasses the plain control: same behaviour, and the look every other boolean in spaCR has.

## ChannelPicker._on_toggled

### lines 115-117

```python
box = self.sender()
```

PUT THE LAST ONE BACK rather than let the picture go blank. Blocked so this correction does not re-enter and does not announce a value the user never chose.

### lines 119-121

```python
if isinstance(box, Toggle):
```

`Toggle`, which is what the boxes above are. Checked at all because `sender()` is typed as QObject and this runs from a signal.

## ChannelPicker

### lines 143-144  _(unsure)_

```python
text = value
```

The panel reads editors through duck-typed accessors; these are the two it looks for, so this widget drops into `_editor` without a special case.

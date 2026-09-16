# Notes from `spacr/qt/widgets/channel_mapping.py`

Prose lifted out of `spacr/qt/widgets/channel_mapping.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## ChannelMappingWidget.__init__

### lines 83-86

```python
label.setToolTip(tip + ". “—” leaves this colour empty.")
```

The whole help, on the name (instruction 113). The spin box used to carry a longer variant of this text, so hovering the field the user was about to type in covered it with a tooltip they had already read on the label beside it.

### line 93, trailing  _(unsure)_

```python
box.setSpecialValueText("—")
```

shown when the value is _EMPTY

### lines 99-104

```python
try:
```

A plain QWidget used as a layout container inherits the blanket `QWidget { background-color: bg }` rule and paints the window colour over whatever is behind it (INVARIANTS §1/§3). This widget registers no QSS of its own precisely so there is no new rule to forget to add to theme.WIDGET_QSS_MODULES -- the children are styled by the existing QSpinBox/QLabel rules, and the container paints nothing.

### lines 109-111

```python
pass
```

Decoration must never be load-bearing (INVARIANTS §10): if the theme cannot be reached the field still works, it just sits on the window colour.

### lines 115-117

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

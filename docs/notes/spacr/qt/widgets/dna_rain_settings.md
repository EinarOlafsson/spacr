# Notes from `spacr/qt/widgets/dna_rain_settings.py`

Prose lifted out of `spacr/qt/widgets/dna_rain_settings.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [DnaRainSettingsPopover.__init__](#dnarainsettingspopover__init__) (2 entries)
- [DnaRainSettingsPopover._position_near](#dnarainsettingspopover_position_near) (1 entry)
- [DnaSettingsButton._on_toggled](#dnasettingsbutton_on_toggled) (1 entry)

## DnaRainSettingsPopover.__init__

### lines 70-73

```python
"""Build the popover holding the DNA rain settings bar.
```

Parented, but still a window: a Qt.Popup with a parent is destroyed with it and lands on the parent's screen, which is what a per-screen popover wants. It is never laid out inside the parent.

### lines 99-101

```python
layout.addWidget(bar)
```

addWidget does the reparenting. An explicit setParent() first would mark the bar hidden and it would stay blank inside a shown popover.

## DnaRainSettingsPopover._position_near

### line 163, trailing  _(unsure)_

```python
except RuntimeError:
```

anchor's C++ side is gone

## DnaSettingsButton._on_toggled

### lines 273-274

```python
self.setChecked(False)
```

The click that closed the popover reached us as well. Stay closed rather than flickering straight back open.

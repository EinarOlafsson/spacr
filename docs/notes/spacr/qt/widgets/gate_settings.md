# Notes from `spacr/qt/widgets/gate_settings.py`

Prose lifted out of `spacr/qt/widgets/gate_settings.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [GateEditorSettings.__post_init__](#gateeditorsettings__post_init__) (1 entry)
- [GateSettingsDialog._xd_tab](#gatesettingsdialog_xd_tab) (2 entries)
- [GateSettingsDialog._three_d_tab](#gatesettingsdialog_three_d_tab) (1 entry)

## GateEditorSettings.__post_init__

### lines 214-216

```python
object.__setattr__(self, "gate_mode", "3D")
```

A settings file written while xD was a third mode. It meant

"project, and give me a Z" -- xD produced three components precisely so the 3D view had one -- so that is what it becomes.

## GateSettingsDialog._xd_tab

### lines 584-585  _(unsure)_

```python
column.addWidget(QLabel(
```

Stated, not an empty box: "no channels in this table" and

"channels exist and none are ticked" are different answers.

### lines 621-625

```python
self._n_neighbors = QSpinBox(page)
```

GREYED, NOT REMOVED, when another method is chosen -- INVARIANTS 6, and the same rule the merged Classify module follows. A control that vanishes teaches the user nothing about why; a greyed one says "this belongs to a method you are not using", and the value they set survives switching away and back.

## GateSettingsDialog._three_d_tab

### lines 834-845

```python
pending = QLabel(
```

THE VOLUME ITSELF IS NOT BUILT YET (instruction 52), and until it is these four controls turn nothing. They are still SHOWN -- the values are saved, reloaded and carried in the settings, so hiding them would lose a user's 3D setup silently the first time they opened this tab. What changes is that they no longer promise behaviour the application does not have.

This is instruction 52's own prescription, quoted: "Until this instruction lands, the 3D group should either be hidden or carry a visible 'not yet'." A control that turns nothing is a promise the app does not keep, which is the defect the whole phantom-settings sweep of instruction 77 was about.

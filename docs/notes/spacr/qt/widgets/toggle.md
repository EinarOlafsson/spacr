# Notes from `spacr/qt/widgets/toggle.py`

Prose lifted out of `spacr/qt/widgets/toggle.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Toggle.__init__](#toggle__init__) (1 entry)
- [Toggle.paintEvent](#togglepaintevent) (3 entries)

## Toggle.__init__

### lines 23-25

```python
self._track_x = 2
```

Approximately 75% of the original 40 x 22 px switch. Leave two physical pixels before the track: a track starting at x=0 clips half of its antialiased 1.5 px outline.

## Toggle.paintEvent

### lines 56-57

```python
def paintEvent(self, event):
```

Custom paint — QCheckBox default indicator is hidden via QSS

(we override paintEvent so we don't render it at all).

### line 63  _(unsure)_

```python
checked = self.isChecked()
```

Track

### line 89  _(unsure)_

```python
if self.text():
```

Label

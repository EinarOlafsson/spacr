# Notes from `spacr/qt/widgets/percentile_pair.py`

Prose lifted out of `spacr/qt/widgets/percentile_pair.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [PercentilePair.__init__](#percentilepair__init__) (2 entries)
- [PercentilePair._field](#percentilepair_field) (1 entry)
- [PercentilePair._on_low](#percentilepair_on_low) (1 entry)
- [PercentilePair.set_value](#percentilepairset_value) (1 entry)
- [PercentilePair.text](#percentilepairtext) (1 entry)

## PercentilePair.__init__

### lines 101-104

```python
self._low.setMaximum(self._high.value())
```

THE ORDER IS ENFORCED BY THE CONTROLS, not checked on the way out. A window whose low end is above its high end has no meaning, and a panel that lets one be entered has to decide later what the user meant.

### lines 109-112

```python
from ..screens.settings_model import retarget_field_tooltips
```

HOVER HELP BELONGS TO THE SETTING'S NAME, never to the box you type in. Built here on the field, it is moved onto the label as the last step, so every panel in the application explains itself the same way.

## PercentilePair._field

### lines 121-123

```python
spin.setSingleStep(0.5)
```

A STEP THAT MATCHES THE NUMBERS. Left at Qt's default of 1.0 a wheel tick on 99.5 lands on 100.5, which the range then clamps so the control appears to ignore the gesture.

## PercentilePair._on_low

### lines 130-131  _(unsure)_

```python
def _on_low(self, value: float) -> None:
```

keeping the two ends in order

## PercentilePair.set_value

### lines 169-171

```python
self._low.setMaximum(100.0)
```

THE BOUNDS ARE OPENED BEFORE THE VALUES ARE SET. Each field's range is pinned to the other's value, so setting a whole new window in place clamps whichever end moves first.

## PercentilePair.text

### lines 190-192

```python
def text(self) -> str:
```

The picture dialog reads unfamiliar editors through `text()`/`setText`, so the pair answers those too rather than needing a special case in every reader.

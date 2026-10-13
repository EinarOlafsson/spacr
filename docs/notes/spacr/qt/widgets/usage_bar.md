# Notes from `spacr/qt/widgets/usage_bar.py`

Prose lifted out of `spacr/qt/widgets/usage_bar.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [UsageBar.__init__](#usagebar__init__) (1 entry)
- [UsageBar._size_the_columns](#usagebar_size_the_columns) (2 entries)
- [UsageBar.set_value](#usagebarset_value) (1 entry)

## UsageBar.__init__

### lines 48-75

```python
self.setStyleSheet(
```

Two rules, and the second is the load-bearing one.

The ROW must paint nothing, or the blanket

`QWidget { background-color: bg }` fills it with the WINDOW colour inside the System card.

So must the bar's TRACK, and that was the part missing. The application sheet gives `QProgressBar#UsageBar` a `surface_alt` fill *at page opacity*, and the card it sits in is already `surface_alt` at page opacity — so the track laid a second copy of the same translucent grey over the first and read as a band the slider could not thin: measured, at a requested 30 % the card passed 0.70 of the backdrop and the track only 0.49.

One of the four bars escaped that, and only by accident: the CPU bar sits in a wrapper carrying an unqualified `background: transparent`, and in Qt a sheet set on an ANCESTOR beats the application sheet irrespective of selector specificity, so the wrapper's rule reached the bar and cancelled the fill. RAM, GPU and VRAM go straight into the card body, whose sheet is qualified (`QWidget#CardBody`) and never reached theirs. Saying it here is what makes all four behave the same wherever they are put.

The selector is name-agnostic on purpose: `set_value` renames the bar to UsageBarWarn / UsageBarError past 75 / 90 %, and the only QProgressBar under this row is that one bar. `::chunk` is a separate sub-control, so the filled part keeps its accent / warning / error colour — only the empty track goes away.

## UsageBar._size_the_columns

### lines 102-104

```python
def _size_the_columns(self) -> None:
```

The two fixed widths, measured rather than assumed

### lines 128-130

```python
if getattr(self, "_pct", None) is None:
```

`setStyleSheet` in `__init__` delivers a StyleChange to this row BEFORE either label exists, so the hook below can arrive during construction. Nothing to size yet is not an error.

## UsageBar.set_value

### line 181  _(unsure)_

```python
self._bar.style().unpolish(self._bar)
```

Force restyle since QSS keys on objectName.

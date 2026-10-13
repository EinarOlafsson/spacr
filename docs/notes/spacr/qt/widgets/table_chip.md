# Notes from `spacr/qt/widgets/table_chip.py`

Prose lifted out of `spacr/qt/widgets/table_chip.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_chip_qss](#_chip_qss) (1 entry)
- [TableChip.__init__](#tablechip__init__) (3 entries)

## _chip_qss

### lines 20-27

```python
ink = palette["fg"]
```

The theme's own text colour: white on the dark themes, as asked, and dark on the light ones without a second rule. It was the WINDOW colour before, which is black on dark -- black on a blue chip.

Contrast maths would pick black here (a mid blue is bright enough that black scores higher), so it is deliberately not used: the ask was white on dark, and following the theme's text colour is what keeps that true in every theme rather than only this one.

## TableChip.__init__

### lines 79-81

```python
self._close = close_mark_button(
```

THE APPLICATION'S CLOSE MARK, not a chip-shaped one. Its glyph, its size and its two colours come from the theme; this chip only says what pressing it removes. See `theme.close_mark_button`.

### lines 86-88

```python
self._close.setVisible(removable)
```

The last table has no x: a gate editor with no table is a screen with nothing on it, and the user's next move would be to load the same table again.

### lines 92-101

```python
self.ensurePolished()
```

THE MARK IS MEASURED, NOT GUESSED. The chip has to hold the name AND whatever box the close mark takes at the user's Zoom, or a larger mark would crop the name it belongs to.

A widget inherits the application's QSS font only when Qt polishes it.  Measuring an unpolished chip therefore uses the platform default font, which can be narrower than the font drawn after ``show()`` (Ubuntu's fallback is one example).  Resolve the style first so this minimum describes the text the user will actually see, not the construction-time fallback.

# Notes from `spacr/qt/widgets/dna_rain.py`

Prose lifted out of `spacr/qt/widgets/dna_rain.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [derive_head_color](#derive_head_color) (2 entries)
- [DnaRainEngine.__init__](#dnarainengine__init__) (1 entry)
- [DnaRainEngine._roll](#dnarainengine_roll) (2 entries)
- [DnaRainEngine.advance](#dnarainengineadvance) (2 entries)
- [DnaRainWidget.__init__](#dnarainwidget__init__) (4 entries)
- [DnaRainWidget._column_color](#dnarainwidget_column_color) (1 entry)
- [DnaRainWidget._backdrop_origin](#dnarainwidget_backdrop_origin) (1 entry)
- [DnaRainWidget._coalesce](#dnarainwidget_coalesce) (1 entry)
- [DnaRainWidget._render_strip](#dnarainwidget_render_strip) (1 entry)
- [DnaRainWidget.paintEvent](#dnarainwidgetpaintevent) (3 entries)
- [_effective_theme](#_effective_theme) (1 entry)
- [DnaRainSettingsBar.__init__](#dnarainsettingsbar__init__) (1 entry)
- [DnaRainSettingsBar.pick_color](#dnarainsettingsbarpick_color) (1 entry)
- [install_dna_rain](#install_dna_rain) (2 entries)
- [_place_beside](#_place_beside) (1 entry)

## derive_head_color

### lines 285-288

```python
target = max((0.0, 1.0), key=gain)
```

A near-white trail over a mid-grey background has nowhere to go within one step: up hits the ceiling, down lands on the background. Give up on the step and take whichever end of the lightness axis is furthest from *both*.

### line 292, trailing  _(unsure)_

```python
if hue < 0.0:
```

achromatic: QColor reports hue -1

## DnaRainEngine.__init__

### lines 458-459  _(unsure)_

```python
self._hue_rng = random.Random(
```

Its own stream, offset from the seed so it is neither the same sequence nor correlated with it, and still reproducible.

## DnaRainEngine._roll

### lines 587-589

```python
head = rng.uniform(-(length + self.n_rows), float(self.n_rows))
```

Spread the initial heads over a whole life cycle so the field looks like it has been running, and so no two columns share a start time.

### lines 593-594  _(unsure)_

```python
run = min(HIGHLIGHT_RUN_CELLS, length)
```

The highlighted run, clamped so it fits however short the string is.

## DnaRainEngine.advance

### lines 671-674

```python
if new_y == old_y and not respawned:
```

Dirty on PIXEL movement, not cell movement. The old test — "did the integer row change" — is what made slow columns step: they were skipped for five frames out of six and then redrawn a whole glyph lower.

### lines 679-681

```python
cell = max(1, self.cell_size)
```

The span still has to be expressed in rows, so widen it by a cell on each side to cover a strip that now straddles a boundary.

## DnaRainWidget.__init__

### lines 783-784  _(unsure)_

```python
self._color = _as_color(color, QColor(DEFAULT_COLOR))
```

The shipped teal, NOT the theme accent: the accent is the Run button and the AI toggle, and a backdrop in it read as chrome.

### line 790

```python
self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
```

Never in front of, never in the way of, the real content.

### lines 806-811

```python
self._hue_pens: dict = {}
```

Per-hue pen sets for random-colour mode, quantised to whole degrees. Strips are already cached per column, so this is only ever hit on a respawn or a restyle — but a restyle re-renders every column at once, and 120 pen sets built one after another is the sort of thing that shows up in a frame budget of half a millisecond.

### lines 813-814  _(unsure)_

```python
self._strips: List[Optional[QPixmap]] = []
```

Pre-rendered opaque strip per column, keyed by

(column generation, styling generation).

## DnaRainWidget._column_color

### lines 899-901

```python
return random_hue_color(self._color, _hue_bucket(column.hue) / 360.0)
```

Quantised exactly as the pen cache quantises it, so this reports the colour that is on screen rather than one a fraction of a degree away from it.

## DnaRainWidget._backdrop_origin

### lines 973-974

```python
window = self.window()
```

`window()` is the widget itself when it has no parent, never

None, so a parentless rain centres the image on itself.

## DnaRainWidget._coalesce

### lines 1229-1232

```python
over = 0
```

Every column is exactly one cell wide now. The spaCR splice used to make its column wider than its stride, because the word was drawn horizontally out of a single cell and bled over its neighbours; five one-letter cells cannot.

## DnaRainWidget._render_strip

### lines 1333-1334

```python
continue
```

The word is wider than a cell, so it is drawn live at full canvas width instead of baked into this strip.

## DnaRainWidget.paintEvent

### lines 1374-1380

```python
touched = set()
```

Pass 1: clear only the rectangles Qt asked for. Everything else in the backing store is still valid.

One fillRect per region rectangle beats clearing per column around each string: at 120 columns the per-call overhead of 240 small fills costs about 2 ms more than the ~0.1 ms of pixels they save. Measured, twice, in both directions.

### lines 1385-1387

```python
first = max(0, rect.left() // cell)
```

No left-hand margin needed. This used to reach extra columns leftward because a spaCR splice over there drew across into this one; the word is now confined to its own column.

### lines 1393-1394  _(unsure)_

```python
order = sorted(touched)
```

Pass 2: blit each touched string. Qt has already clipped the painter to the region, so nothing outside it is written.

## _effective_theme

### lines 1401-1407

```python
def _effective_theme() -> str:
```

There is no second pass. The spaCR splice used to need one: the word lived in a single cell, was drawn horizontally, and had to clear a backing rectangle that ran across its neighbouring columns — so it had to come last, after every strip was down. Now it is five ordinary one-letter cells inside its own column's strip, so it is rendered by the loop above like any other glyph, cached like any other glyph, and cannot overdraw anything.

## DnaRainSettingsBar.__init__

### lines 1474-1477

```python
self.setObjectName("DnaRainBar")
```

The rain is painted behind this bar, and the global

``QWidget { background }`` rule does not reach a widget that has its own stylesheet — so give the bar an opaque surface of its own or its labels land on falling glyphs.

## DnaRainSettingsBar.pick_color

### lines 1665-1666

```python
chosen = pick_colour(self, self._color, "DNA rain colour")
```

Qt's own dialog, never the platform one -- see :mod:`spacr.qt.widgets.colour_picker`.

## install_dna_rain

### lines 1811-1812

```python
bar = DnaRainSettingsBar(None, theme=kwargs.get("theme"), vertical=True)
```

No parent: the popover adopts it. Parenting it to `host` first would flash a settings bar across the screen for one event loop.

### lines 1815-1817

```python
from .dna_rain_settings import DnaSettingsButton
```

Imported here rather than at module scope: the popover module imports this one for the bar, so a top-level import either way round is a cycle.

## _place_beside

### line 1873  _(unsure)_

```python
layout.addWidget(button)
```

Not a box layout — better beside nothing than not at all.

# Notes from `spacr/qt/widgets/tile.py`

Prose lifted out of `spacr/qt/widgets/tile.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_TileButton](#_tilebutton) (1 entry)
- [HTile](#htile) (1 entry)
- [HTile.__init__](#htile__init__) (10 entries)
- [HTile.required_width](#htilerequired_width) (2 entries)

## _TileButton

### lines 42-44  _(unsure)_

```python
class _TileButton(QPushButton):
```

Classic square Tile (kept for backwards compatibility)

## HTile

### lines 162-164  _(unsure)_

```python
class HTile(QPushButton):
```

Horizontal Tile — the new minimalist home-screen card

## HTile.__init__

### lines 200-201  _(unsure)_

```python
self._name_lbl = None
```

Set before any layout work: Qt can ask for sizeHint() while the widget is still being built, and the hint reads this attribute.

### lines 204-207

```python
from ..preferences import scaled_px
```

Icon size and all icon-adjacent geometry track the user's font-size preference so the tile grows with the text and nothing clips when the font is bumped up. ``icon_size`` is the base (100 %) side length. (No hover zoom — the icon/text stay a fixed size on hover.)

### lines 213-215

```python
self.setAccessibleName(text)
```

Accessibility: screen readers announce the app name + one-line description as the button's role. Tooltip stays for sighted hover; the accessible bits are what NVDA / VoiceOver read.

### lines 223-225

```python
self.setMinimumHeight(scaled_px(72))
```

Height tracks the font scale (scaled_px) but keeps the original proportions — the earlier icon-driven height made the tiles too tall. Width is handled by the caller (also via scaled_px).

### lines 227-228  _(unsure)_

```python
self.setToolTip(f"{text} — {description}" if description else text)
```

Tooltip leads with the NAME even when there's a description, so a tile too narrow for its label is still identifiable on hover.

### lines 231-232  _(unsure)_

```python
layout = QHBoxLayout(self)
```

Two-line label stack next to the icon. Left padding (scaled) leaves room for the QIcon the button paints on the left edge.

### lines 242-247

```python
name_lbl = ElidingLabel(text)
```

The name is an ElidingLabel, not a plain QLabel: a plain one silently clips ("Annotator Agreeme") when the tile is narrower than the name, which is unreadable and unclickable. This one shortens with an ellipsis and moves the full name into the tooltip — and :meth:`sizeHint` below makes sure the tile is usually wide enough that it never has to.

### lines 250-252

```python
name_lbl.setMinimumWidth(0)
```

Don't clip — the tile stretches to accommodate the label when longer app names appear. Explicit minimum width so short names still look proportionate.

### line 259  _(unsure)_

```python
text_col.addStretch(1)
```

Description shown BELOW the name (two-line tile).

### lines 268-269

```python
text_col.addStretch(1)
```

Name-only tile: vertically centre the label so it sits in the middle rather than pinned to the top-left.

## HTile.required_width

### lines 276-284

```python
def required_width(self) -> int:
```

geometry

HTile draws its name in a CHILD QLabel, not in the button's own text. QPushButton.sizeHint()/minimumSizeHint() only measure the button's own text + icon, so without these overrides the label's width requirement never reaches the layout: every tile reported the same ~92 px hint no matter how long the app name was, callers that did `max(floor, tile.sizeHint().width())` always got the floor, and anything longer than the floor left over got clipped.

### line 295  _(unsure)_

```python
return QPushButton.sizeHint(self).width()
```

Asked mid-construction — fall back to the plain button hint.

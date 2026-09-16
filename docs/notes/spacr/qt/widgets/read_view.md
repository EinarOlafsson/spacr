# Notes from `spacr/qt/widgets/read_view.py`

Prose lifted out of `spacr/qt/widgets/read_view.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [ReadView.__init__](#readview__init__) (1 entry)
- [ReadView.changeEvent](#readviewchangeevent) (1 entry)
- [ReadView.refresh_colours](#readviewrefresh_colours) (1 entry)

## Module level

### lines 52-69

```python
_GOLDEN_ANGLE = 137.50776405003785
```

What it costs, measured

150-base reads with three matched spans each, offscreen, a 900x700 view on the maintainer's box. The column that matters is the last one: the number of reads whose character ownership had to be worked out does not depend on how many reads there are, because only the rows on screen are ever asked for.

reads    set_reads   first paint   resolved   scroll to end   resolved 1 000      0.3 ms       16.5 ms         38          8.9 ms         76 10 000      1.1 ms       23.2 ms         38          9.1 ms         76 50 000      5.9 ms       76.7 ms         38          9.6 ms         76

The first paint is the one number that grows with the file, and it is Qt laying the rows out on the posted event after the model reset rather than anything here formatting text. At fifty thousand reads it is 77 ms once, against the 400 ms the preview responsiveness guard next door treats as a freeze, and it does not come back when the user scrolls.

### lines 72-98

```python
_GOLDEN_ANGLE = 137.50776405003785
```

The colour rule

One mechanism, no table of literals, and no ceiling on how many barcode types it can serve. Colour number `i` is the theme's own `accent` role put through `spaceout_palette`, the published re-hue whose documented job is to move a role in hue while preserving the luminance that makes it readable.

WHY THAT IS SAFE RATHER THAN MERELY PRETTY. `CONTRAST_RULES` holds `accent` at 4.5:1 against every page surface of every theme, and WCAG contrast is a function of relative luminance alone. `spaceout_palette` leaves the luminance where it found it -- measured drift below 0.003 across all four palettes -- so every hue it hands back clears what `accent` clears. Measured worst case over the first eight colours, against every page surface:

dark 5.55:1   light 4.55:1   cell 7.78:1   glass 5.68:1

and against `accent_soft`, which is what a selected row is painted with:

dark 4.52:1   light 4.69:1   cell 8.75:1   glass 6.86:1

THE STEP IS THE GOLDEN ANGLE, not `360 / count`. Even spacing would make a type's colour depend on how many types there are, so finding a fourth barcode would recolour the three already on screen and the user would have to re-read the legend. The golden angle spreads any prefix of the sequence about as well as even spacing does while leaving colour number three the same colour it was before colour number four existed.

## ReadView.__init__

### lines 654-656

```python
self._list.setUniformItemSizes(True)
```

Uniform sizes is the setting that makes the view ask the delegate for one size instead of one per read, and wrapping off with elide off is what stops a long read becoming two visual rows.

## ReadView.changeEvent

### lines 754-755  _(unsure)_

```python
if not hasattr(self, "_delegate"):
```

Qt can deliver a change event while the base class is still constructing, before anything below exists to refresh.

## ReadView.refresh_colours

### lines 776-780

```python
self._list.doItemsLayout()
```

A new text size means a new row height and a new row width, and the view has cached the old ones because it was told the sizes are uniform. Laying the items out again asks for them afresh, which resetting the view would also do at the cost of throwing away the user's scroll position and selection.

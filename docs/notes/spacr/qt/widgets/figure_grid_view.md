# Notes from `spacr/qt/widgets/figure_grid_view.py`

Prose lifted out of `spacr/qt/widgets/figure_grid_view.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_SectionHeader.__init__](#_sectionheader__init__) (1 entry)
- [_SectionHeader.mouseReleaseEvent](#_sectionheadermousereleaseevent) (1 entry)
- [_FigureCell.__init__](#_figurecell__init__) (6 entries)
- [_FigureCell.fit_to](#_figurecellfit_to) (1 entry)
- [_FigureCell.mousePressEvent](#_figurecellmousepressevent) (1 entry)
- [FigureGridView.set_live_tiles](#figuregridviewset_live_tiles) (1 entry)
- [FigureGridView.set_pinned](#figuregridviewset_pinned) (2 entries)
- [FigureGridView._discard](#figuregridview_discard) (1 entry)
- [FigureGridView.clear](#figuregridviewclear) (3 entries)
- [FigureGridView.workspace_state](#figuregridviewworkspace_state) (1 entry)
- [FigureGridView.apply_workspace_state](#figuregridviewapply_workspace_state) (1 entry)
- [FigureGridView._is_raised](#figuregridview_is_raised) (1 entry)
- [FigureGridView.toggle_section](#figuregridviewtoggle_section) (1 entry)
- [FigureGridView._relayout](#figuregridview_relayout) (4 entries)
- [FigureGridView._lay_out_live_section](#figuregridview_lay_out_live_section) (1 entry)

## _SectionHeader.__init__

### lines 214-216

```python
self._chevron.setProperty("i18nSkipText", True)
```

A run heading is generated text -- a timestamp, or a trial's name in whatever language was active when the run started. A later whole-window language switch must not try to reinterpret it.

## _SectionHeader.mouseReleaseEvent

### lines 263-264

```python
"""Fold or unfold this run's section on a click inside the bar.
```

Release rather than press, so dragging off the bar cancels -- what every other clickable in the app does.

## _FigureCell.__init__

### lines 333-335

```python
self.setContextMenuPolicy(Qt.CustomContextMenu)
```

"all gigures should be editable by right clicking" -- a tile is a figure, so the gesture has to work here too and not only on the one figure that happens to be open.

### lines 344-347

```python
self.setAutoFillBackground(False)
```

A TILE DOES NOT PAINT ITS OWN GROUND. Reported as "on the grid (all figures) the graphs still have a black background": the figures are transparent and the frame behind them was not, so every tile was a slab. The frame stays for its border; only its fill goes.

### lines 352-357

```python
from ..theme import font_px
```

IMPORTED HERE, NOT INSIDE `if letter:`. It was, and the caption block below uses it too -- so a panel with a title and NO letter raised UnboundLocalError and took the grid down with it. A letter is optional and a caption is optional, which makes the pairing "captioned but unlettered" an ordinary panel rather than an edge case.

### lines 361-363

```python
tag = QLabel(letter.upper())
```

UPPER-CASE PANEL LETTER, top left, bold -- asked for by name: "i asked you to make the all figures pannel publication style (with each panel having an uppercase letter) and be on a grid".

### lines 375-376  _(unsure)_

```python
self._image.setMinimumHeight(80)
```

NOT setScaledContents: that is exactly the stretch this replaces. The pixmap is scaled with KeepAspectRatio when the cell is sized.

### lines 378-379  _(unsure)_

```python
follow_device_ratio(self._image, self._refit)
```

A grid dragged onto a denser screen keeps its cell widths, so no relayout arrives to refit the figures -- this is what does.

## _FigureCell.fit_to

### lines 410-412

```python
self._image.setFixedHeight(logical_size(scaled).height())
```

The height the cell reserves is what the picture OCCUPIES, not how many pixels it was drawn with -- those differ by the ratio, and a cell sized in device pixels is twice as tall as its picture.

## _FigureCell.mousePressEvent

### lines 430-431

```python
"""Open the figure on a left click; leave a right click to the menu.
```

A right-click opens the menu; it must not ALSO open the figure, or every attempt to restyle a tile navigates away from the grid first.

## FigureGridView.set_live_tiles

### lines 583-586

```python
for cell in previous:
```

BEFORE the relayout, not after: `_discard` is what takes a tile off the body, and a tile still parented to the body when the layout runs paints itself at its old geometry for the rest of the event-loop turn. That ordering is the fix, not the discarding.

## FigureGridView.set_pinned

### lines 641-644

```python
cell.menu_requested.connect(
```

"all gigures should be editable by right clicking" -- and this one is the only tile on the grid that is a real, live figure, so a right-click that did nothing here would be the gesture failing on the one tile where it can do the most.

### lines 647-648  _(unsure)_

```python
self._live = [cell] + others
```

FIRST, whatever else is on the section. "always first" is what the name promises and what the caller relies on to find it.

## FigureGridView._discard

### line 671  _(unsure)_

```python
pass
```

Already torn down by Qt -- the screen closed under us.

## FigureGridView.clear

### lines 686-692

```python
if widget is not None and id(widget) not in live:
```

The live tiles survive a clear: they are not the figures being replaced, and a run that streams new ones must not make the interactive graphs disappear. Compared by identity through a set of ids rather than `in self._live` -- `in` on a list of QWidgets goes through `__eq__`, which Qt does not define for widgets, so it degrades to identity anyway but at O(n) per tile on a grid that can hold a few hundred.

### lines 695-701

```python
seen = set(map(id, doomed))
```

THE LAYOUT IS NOT THE WHOLE GRID. A cell belonging to a FOLDED run is deliberately left out of the layout by `_relayout` (so the next run flows up under the folded heading instead of into a hole), which means walking the layout alone never reaches it -- it stays a child of the body while `_cells` is emptied out from under it, and the only reference to it is gone. Nothing on screen, but it is still there, and a sweep that folds its runs away leaks one per figure.

### lines 709-714

```python
self._headers = []
```

The headings went out with the rest of the layout, so the list must go too -- otherwise _relayout reaches through a wrapper whose C object has already been torn down and raises RuntimeError. The COLLAPSED SET deliberately survives: clearing is how the grid is rebuilt after every run, and a fold that came undone on each rebuild is the unusable sweep this exists to prevent.

## FigureGridView.workspace_state

### line 764

```python
def workspace_state(self) -> dict:
```

instruction 180: what the grid contributes to a saved run

## FigureGridView.apply_workspace_state

### lines 796-798

```python
self.set_target_cell_width(int(width))
```

Through the setter: it clamps and relayouts, and a raw

`_target` would leave the grid drawn at the old width until something else happened to trigger a relayout.

## FigureGridView._is_raised

### line 856, trailing  _(unsure)_

```python
return False
```

header rebuilt between click and query

## FigureGridView.toggle_section

### lines 899-900  _(unsure)_

```python
QTimer.singleShot(0, lambda: self._scroll_section_to_top(key))
```

After layout, not during: the geometry this scroll needs does not exist until the cells just shown have been placed.

## FigureGridView._relayout

### lines 906-914

```python
for header in self._headers:
```

THE PREVIOUS HEADINGS ARE DESTROYED, NOT MERELY UNPARENTED FROM THE LAYOUT. `takeAt` removes the layout item and leaves the widget a visible child of the body at its old geometry, so every relayout and a window resize is a relayout -- used to leave another copy of every run heading painted on the grid. Measured before the fix: three relayouts of a two-run grid left six headings. The pinned tile had the same bug for the same reason -- see :meth:`set_pinned`, which now shares this one's cleanup rather than re-deriving it a third time.

### lines 925-936

```python
heading_at = {}
```

A HEADING PER RUN, INCLUDING THE FIRST AND ONLY ONE.

It used to appear only from the second run onwards, on the argument that the lettering restarting is what needs explaining and one run never restarts. That argument was about the LABEL and this control is also the fold: with one run there was no header, so there was nothing to click, and the maintainer reported the figures as "still not colapsable into runs" while the folding worked perfectly from the second run on.

A heading over a single run costs one row and answers "which run is this" -- which the grid could not previously say at all.

### lines 957-960

```python
cell.setVisible(False)
```

Left out of the layout AND hidden. Out of the layout so the next run flows up under the folded heading instead of into a hole; hidden because a widget removed from a layout keeps painting itself where it last was.

### lines 965-966

```python
if column + span > columns:
```

A wide figure that will not fit in what is left of this row starts the next one, rather than being squeezed.

## FigureGridView._lay_out_live_section

### lines 1014-1016

```python
cell.setVisible(False)
```

Out of the layout AND hidden, for the reason the folded figures are: a widget merely removed from a layout goes on painting itself where it last was.

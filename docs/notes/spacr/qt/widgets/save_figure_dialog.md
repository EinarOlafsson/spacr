# Notes from `spacr/qt/widgets/save_figure_dialog.py`

Prose lifted out of `spacr/qt/widgets/save_figure_dialog.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [style_for_file](#style_for_file) (3 entries)
- [SaveFigureDialog.__init__](#savefiguredialog__init__) (7 entries)
- [SaveFigureDialog._colour_box](#savefiguredialog_colour_box) (1 entry)
- [SaveFigureDialog._resolve_choice](#savefiguredialog_resolve_choice) (2 entries)
- [SaveFigureDialog._show_the_page](#savefiguredialog_show_the_page) (1 entry)
- [SaveFigureDialog._refresh_fast_plot](#savefiguredialog_refresh_fast_plot) (2 entries)
- [SaveFigureDialog.save](#savefiguredialogsave) (1 entry)

## style_for_file

### lines 262-263

```python
if font_scale and font_scale > 0:
```

Apply the scale to existing artists because rcParams only affect text created after the parameter change.

### lines 286-288

```python
axes.tick_params(labelcolor=text_ink, which="both")
```

THE NUMBERS BESIDE THE TICKS ARE TEXT and the little dashes are lines, so the two halves of `tick_params` follow two different controls. One call with both would tie them together again.

### lines 300-301  _(unsure)_

```python
if grid:
```

Matplotlib enables the grid when line properties accompany

``grid(False)``. Supply styling arguments only for the enabled case.

## SaveFigureDialog.__init__

### line 374  _(unsure)_

```python
self.background = self._colour_box(
```

the four that are the file's

### lines 399-401

```python
self.ink = self._colour_box(
```

NAMED `ink` IN THE CODE, "text colour" ON THE ROW. The attribute is what every caller and test already reaches for; the label is the word the maintainer used and the word the row beside it uses.

### lines 408-412

```python
if not self._fast:
```

A MATPLOTLIB FIGURE HAS NO LIVE STYLE MENU. Its page can be reduced from a wide on-screen figure to a journal column here, but without an export-only scale the existing labels remain full size and crowd out the axes. Fast plots already own one font-size setting on their right-click menu, so they must not get a second answer here.

### lines 426-434

```python
try:
```

THE SHAPE OF THE FIGURE, as a choice rather than a number. A ratio is a number, and a reader deciding how a figure sits on a page is choosing a shape.

THE SAME VOCABULARY THE GRAPH'S OWN MENU USES, read from the one table, so a figure shaped from the menu and one shaped here are shaped by the same names. "Lock axis scales" -- one y unit drawn as n x units -- is a statement about the DATA and lives on that menu under Axes; it is not what "save it as a square" means.

### lines 461-464

```python
self.dpi = QSpinBox()
```

A RASTER EXPORT IS WHAT A RESOLUTION IS FOR. It follows the FORMAT and not the kind of plot: a journal asking for 300 dpi is asking about the PNG, and greying the one control that decides how big that file really is takes the answer away.

### lines 483-487

```python
for box in (self.width, self.height):
```

The page a pyqtgraph plot writes onto is set in millimetres on its OWN right-click menu (`set_export_size`), and it is read by all three of its export paths. Offering a second answer in inches here would give the user two controls for one quantity and no way to tell which won -- so these SHOW it instead.

### lines 511-513

```python
self._trouble = QLabel()
```

WHERE A REFUSAL IS SAID OUT LOUD. A preview or a save that fails writes its reason here; an empty label is hidden, so the dialog gains a line only when there is something to read.

## SaveFigureDialog._colour_box

### lines 549-551

```python
box.currentIndexChanged.connect(
```

THE CHOOSER RUNS FIRST. Connected before the refresh so a chosen colour is already in the combo by the time the preview is rebuilt; the other order previews the sentinel and then the colour.

## SaveFigureDialog._resolve_choice

### lines 571-575

```python
lowered = name.lower()
```

CASE-INSENSITIVELY, because QColor.name() answers in lower case and the shipped entries are written upper. An exact match misses "white" for #ffffff and inserts a second, visually identical row -- once for every time the user picks a colour the list already had.

### line 583  _(unsure)_

```python
index = box.count() - 1
```

Before the chooser, so the chooser stays last.

## SaveFigureDialog._show_the_page

### lines 684-686

```python
blocked = box.blockSignals(True)
```

Written with the handler blocked: these boxes only REPORT the plot's page, and letting them re-enter the refresh that is about to run renders the preview twice per keystroke.

## SaveFigureDialog._refresh_fast_plot

### line 858  _(unsure)_

```python
self._preview = pixmap
```

The pixmap IS the preview; there is no figure object behind it.

### lines 865-867

```python
self._say(f"{PREVIEW_FAILED} {failure}" if self._preview is not None
```

KEEP THE FIGURE. The holder is NOT cleared: what is in it is the last drawing that worked, and the note says exactly that so it cannot be read as the answer to the settings just made.

## SaveFigureDialog.save

### lines 935-940

```python
target.savefig(chosen, dpi=int(self.dpi.value()),
```

NOT `plot.save_figure`, and deliberately. Every decision that writer makes -- the format, the DPI, the page colour -- the user has just made in this dialog and is looking at in the preview. Overriding any of them here would write something other than what was previewed, which is the one thing this window promises not to do.

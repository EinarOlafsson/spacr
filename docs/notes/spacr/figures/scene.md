# Notes from `spacr/figures/scene.py`

Prose lifted out of `spacr/figures/scene.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [pyqtgraph_ready](#pyqtgraph_ready) (2 entries)
- [_Look.__init__](#_look__init__) (1 entry)
- [_Look.paint](#_lookpaint) (1 entry)
- [_dash](#_dash) (1 entry)
- [_add_rectangles](#_add_rectangles) (3 entries)
- [_in_data_coordinates](#_in_data_coordinates) (2 entries)
- [_plain_text](#_plain_text) (2 entries)
- [_plain_text._script](#_plain_text_script) (1 entry)
- [_add_image](#_add_image) (2 entries)
- [_configure_axes](#_configure_axes) (1 entry)
- [_carry_ticks](#_carry_ticks) (1 entry)
- [_translate_axes](#_translate_axes) (1 entry)
- [build_scene](#build_scene) (2 entries)

## Module level

### line 34

```python
"Spine", "XAxis", "YAxis", "XTick", "YTick", "AxesSubplot", "Axes",
```

Chrome, translated as axis CONFIGURATION rather than as items.

## pyqtgraph_ready

### lines 184-197

```python
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
```

THE IMPORT ORDER IS LOAD-BEARING AND THE FAILURE IS BRUTAL. pyqtgraph picks its Qt binding on first import by trying PyQt5, PyQt6, PySide2, PySide6 in that order -- and PyQt6 is installed in this environment, so a bare `import pyqtgraph` binds to PyQt6 and loads ITS libQt6Core. PySide6 6.11 then cannot load at all:

libpyside6.abi3.so.6.11: undefined symbol:

ZN9QtPrivate9sizedFreeEPvm, version Qt_6

Measured 2026-08-18, and it is not hypothetical -- it took the whole QC suite back to matplotlib on the first run through this path. Importing PySide6 FIRST leaves it in `sys.modules` where pyqtgraph finds it, and PYQTGRAPH_QT_LIB says so out loud for anything that imports pyqtgraph before this function is ever called.

### lines 213-243

```python
if threading.current_thread() is not threading.main_thread():
```

NOT ON A WORKER THREAD, WHEN THERE IS A GUI TO BE A WORKER OF.

`build_scene` makes a `pg.GraphicsLayoutWidget`, and a QWidget must be constructed on the GUI thread. Built on a worker it LIVES there, and every later touch -- including Qt destroying it -- is undefined. Qt reports the one case it can detect ("QBasicTimer::start: Timers cannot be started from another thread") and says nothing about the rest.

Traced 2026-08-19 from a crash dump, after the process had segfaulted twice in places that had nothing to do with it: once inside an application-wide event filter, once inside pandas' CSV parser. The construction guard named the real one:

WidgetGroup was CONSTRUCTED on 'Dummy-2' bridge.py run  ->  perform_regression >  _run_guide_permutation_analysis  ->  write_diagnostic_suite >  plot_inference_diagnostics  ->  write_figure  ->  render_figure >  build_scene

i.e. the QC suite of every regression, on the run's own worker thread.

Answered HERE because this function already exists to say whether a scene can be built "here and now", and its callers already know what to do with a no: `render_figure` returns None and the caller writes the matplotlib page instead. The figure is still produced; only the renderer changes, which is the trade this module already makes for a missing pyqtgraph.

A HEADLESS RUN IS NOT AFFECTED. With no GUI, the run IS the main thread and this is true; the check only fires for a worker under a live application, which is exactly the dangerous case.

## _Look.__init__

### lines 317-323

```python
self.scale = max(float(dpi), 1.0) / _POINTS_PER_INCH
```

POINTS ARE NOT PIXELS, AND THE FIGURE'S OWN DPI IS THE EXCHANGE RATE. Every size matplotlib carries -- a line width, a marker diameter, a font size -- is in POINTS; a pyqtgraph scene is in the pixels the widget is that many inches wide at. Passing the number through unchanged draws a 3 pt marker 3 px across, which at this suite's 140 dpi is 1.9 times too small, and a panel of dust where the panel had points.

## _Look.paint

### lines 353-355

```python
if record:
```

Recorded rather than changed: a palette chosen against a dark ground can be illegible on paper, and the honest answer is to NAME it rather than substitute a colour the user did not choose.

## _dash

### line 373  _(unsure)_

```python
try:
```

matplotlib also carries (offset, (on, off, ...)) tuples.

## _add_rectangles

### lines 619-626

```python
if rectangle.get_data_transform() is not axes.transData:
```

A PATCH'S `get_transform` IS NOT THE DATA TRANSFORM, and testing it against `ax.transData` silently dropped EVERY bar in the suite the p-value histogram, the VIF bars, the response distribution and the design conditioning all came out as empty axes with correct ranges, which is the most convincing kind of wrong figure there is. `Patch.get_transform` is `get_patch_transform() + the artist's`, so it is a composite and is never that object; `get_data_transform` is the artist's, and is.

### lines 628-630

```python
corner = _in_data_coordinates(_FractionPoint(left, bottom), axes)
```

A backdrop drawn in axes fraction -- `_skip_box` draws one behind the reason a panel is missing. Converted, because a skipped tile with no backdrop stops looking skipped.

### lines 638-645

```python
left, right = _clamp(left, left + width, *sorted(axes.get_xlim()))
```

CLAMPED TO THE VIEW. A ViewBox does not clip its children, and matplotlib does clip a patch, so a `barh` whose bars start at x = 0 on an axis that starts above zero drew them straight out through the left spine and across the page -- seen on `vif`. Clamping the GEOMETRY rather than clipping the ITEM is deliberate: an annotation deliberately placed below the axes (`predictor_correlation` puts its caption at y = -0.34) must still be drawn, and a clip on the ViewBox would take that with it.

## _in_data_coordinates

### lines 739-744

```python
try:
```

USE THE ARTIST'S ACTUAL TRANSFORM when it has one.  This is required for ``Annotation(textcoords='offset points')``: its position is the offset (for example ``(-5, 8)``), while the transform also carries the data anchor.  Treating those two numbers as axes fractions silently puts the guide label off-canvas.  Matplotlib resolves an Annotation's offset transform on draw, so give an otherwise-undrawn figure that one chance.

### lines 757-759

```python
(left, right), (bottom, top) = axes.get_xlim(), axes.get_ylim()
```

Last-resort compatibility for the small adapters used by rectangle translation: ``_FractionPoint`` deliberately has no transform and its position is defined in axes fractions.

## _plain_text

### lines 840-845

```python
for _ in range(6):
```

THE BRACES THAT ARE STILL DOING WORK ARE THE ONES AFTER \sqrt, ^ AND _. Everything else is a group left behind by a removed \mathrm, and it has to go before \sqrt can be read -- its argument is not brace-free until then. Stripping the lot in one pass instead ate the sqrt's own braces and, separately, let a greedy subscript run swallow the "(p)" after `_{10}`. Both measured.

### lines 876-880

```python
understood = not any(character in body for character in "\\{}^_")
```

`_` IS ON THIS LIST AND `\lambda_{GC}` IS WHY. A subscript whose characters have no Unicode form comes back with its `_` intact, and "lambda underscore GC" printed on an axis is a label the panel did not write. A subscript that DID convert leaves no `_` behind, so the honest ones cost nothing.

## _plain_text._script

### lines 865-867

```python
return converted if all(
```

A run with one character outside the table is left ALONE and the leftover ^ or _ then fails the check below, rather than coming out half-raised.

## _add_image

### lines 968-975

```python
image = pg.ImageItem(array.T)
```

pyqtgraph indexes an image as [x, y]; matplotlib's array is

[row, column]. Transposing is the whole conversion -- the way UP is already carried, because `imshow` states it in its extent (a default `origin='upper'` returns bottom > top) and `_configure_axes` inverts the view for it. Flipping the array as WELL, which is the obvious-looking thing to do, turns a correlation matrix's diagonal into its anti-diagonal -- measured on `predictor_correlation`, where the identity cells came out at (0,2), (1,1), (2,0).

### lines 984-989

```python
return 1
```

THE COLOUR MAP IS NOT A LIST OF MARKS, so it is deliberately NOT fed to the legibility check. A diverging map's midpoint is pale BY DESIGN RdBu_r at r = 0 is near white -- and reporting it would fire the warning on every correlation panel ever written, which is the "a warning that fires on every figure is a warning nobody reads" failure the floor was chosen to avoid. A ramp is read against its own bar, not against the page.

## _configure_axes

### lines 1028-1032

```python
try:
```

THE TYPE SIZE IS THE PANEL'S, NOT A DEFAULT. The house style pins ticks at 6 pt and labels at 7 pt precisely so a page of twenty panels reads as one figure; a renderer that substitutes pyqtgraph's own defaults undoes that silently and the two libraries stop agreeing about the thing they most obviously should.

## _carry_ticks

### lines 1103-1107

```python
continue
```

A LOG AXIS WRITES ITS TICKS IN MATHTEXT, and carrying those across raw put `$\mathdefault{10^{-1}}$` down the side of the design-spectrum panel. pyqtgraph's own log axis writes correct labels, so the honest move is to leave them to it rather than to print a formatter's source code on the figure.

## _translate_axes

### lines 1220-1225

```python
if (isinstance(artist, Annotation)
```

Annotation positions can live in an OFFSET coordinate system: guide volcano labels use ``textcoords='offset points'``.  A plain TextItem has no anchor point plus offset transform, and treating (-5, 8) as axes fractions puts the label off-canvas while still claiming a complete translation.  Decline the scene so write_figure retains the exact matplotlib page.

## build_scene

### lines 1391-1396

```python
offset = 1 if title is not None and title.get_text() else 0
```

THE SUPTITLE TAKES ROW 0 AND EVERYTHING ELSE MOVES DOWN. pyqtgraph's `addLabel(row=-1)` is not "the row above"; it is an invalid row, and Qt says `QGraphicsGridLayout::addItem: invalid row/column: -1` and drops the label -- so the three multi-panel diagnostic sheets lost their titles silently, which for a page called "Screen design diagnostics" is the one line a reader needs.

### lines 1403-1409

```python
widget.ci.layout.setColumnFixedWidth(column, COLORBAR_PX)
```

A KEY IS NOT A PANEL. A GraphicsLayout gives every column the same width, so a colour bar added as a second column took HALF the page -- measured on `predictor_correlation`, where a 3x3 matrix came out smaller than its own legend. The width is fixed here, on the layout, rather than on the item: a maximum width on the PlotItem shrinks the item inside a cell that stays half the page and leaves it floating in the middle of the white.

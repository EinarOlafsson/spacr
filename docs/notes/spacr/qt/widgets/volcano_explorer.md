# Notes from `spacr/qt/widgets/volcano_explorer.py`

Prose lifted out of `spacr/qt/widgets/volcano_explorer.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_OptionalNumbers.__init__](#_optionalnumbers__init__) (1 entry)
- [_OptionalNumbers.setValue](#_optionalnumberssetvalue) (1 entry)
- [VolcanoExplorer.__init__](#volcanoexplorer__init__) (3 entries)
- [VolcanoExplorer.build_style_menu.changed](#volcanoexplorerbuild_style_menuchanged) (1 entry)
- [VolcanoExplorer._style_choices](#volcanoexplorer_style_choices) (1 entry)
- [VolcanoExplorer.merge_annotation_file](#volcanoexplorermerge_annotation_file) (2 entries)
- [VolcanoExplorer._build_controls](#volcanoexplorer_build_controls) (1 entry)
- [VolcanoExplorer._register](#volcanoexplorer_register) (1 entry)
- [VolcanoExplorer._combo](#volcanoexplorer_combo) (1 entry)
- [VolcanoExplorer._repopulate_column_menus](#volcanoexplorer_repopulate_column_menus) (1 entry)
- [VolcanoExplorer._pull_style_from_controls](#volcanoexplorer_pull_style_from_controls) (1 entry)
- [VolcanoExplorer.refresh](#volcanoexplorerrefresh) (1 entry)
- [VolcanoExplorer._error_ink](#volcanoexplorer_error_ink) (1 entry)
- [VolcanoExplorer._show_problems](#volcanoexplorer_show_problems) (1 entry)
- [VolcanoExplorer.nearest_point](#volcanoexplorernearest_point) (1 entry)
- [VolcanoExplorer.export](#volcanoexplorerexport) (1 entry)

## _OptionalNumbers.__init__

### line 161  _(unsure)_

```python
self._auto.toggled.connect(self._auto_toggled)
```

Connected LAST, so building the widget is not a change to report.

## _OptionalNumbers.setValue

### lines 212-214

```python
self._auto.setChecked(True)
```

A value this control cannot express -- a pair where one number was wanted -- reads as automatic rather than as a number nobody typed.

## VolcanoExplorer.__init__

### lines 378-384

```python
self._canvas.setContextMenuPolicy(Qt.CustomContextMenu)
```

INSTRUCTION 108: RIGHT-CLICK THE FIGURE ITSELF. Every control this explorer offers is in the side panel, which is the right home for them -- but 108 is about reaching a figure's own style FROM the figure, and a matplotlib canvas had no menu at all. The entries are built from `dataclasses.fields(VolcanoStyle)` by the same two functions the pyqtgraph plots use, so a style that gains a field gains a menu entry here without anyone remembering to add one.

### lines 388-390

```python
self._problem_line = QLabel("", self)
```

THE EXPLANATION GOES UNDER THE PLOT. Not over it and not instead of it: a message that replaces the figure takes away the one thing the reader was looking at, over a single mistyped field.

### lines 416-418

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## VolcanoExplorer.build_style_menu.changed

### lines 482-485

```python
def changed(_name=None, _value=None):
```

ONE REDRAW PER CHANGE, and through `set_style` rather than

`refresh`, so the side panel's controls follow the menu. Two ways to change one setting that disagree about what it now is would be worse than having only one of them.

## VolcanoExplorer._style_choices

### lines 521-524

```python
if data is None and widget.itemText(index) != _NONE_ROW:
```

A "— none —" row carries `None` as its data, and `None` IS the value there -- it is how a colour-by column is taken back off -- so it stays on the offered list. Any other row with no data falls back to what it says.

## VolcanoExplorer.merge_annotation_file

### lines 588-589  _(unsure)_

```python
key = max(shared, key=lambda c: self._results[c].astype(str).isin(
```

The column that actually matches the most rows, not the first one that happens to share a name.

### lines 601-604

```python
merged = self._results.copy()
```

Join on the string form of the key, so 'TGGT1_225160' matches whatever dtype each side happened to be read as -- a numeric-looking guide id read as int on one side and str on the other would otherwise match nothing and silently annotate every row with NaN.

## VolcanoExplorer._build_controls

### lines 703-705

```python
("localizations", self._multi(
```

SEVERAL AT ONCE. Ticking two compartments asks one question about both, which is why this is a list of tick boxes and not a drop-down.

## VolcanoExplorer._register

### lines 815-819

```python
label.setWordWrap(True)
```

A CAPTION IS A SENTENCE HERE, not a word: "Multiplier applied to the rule above (the quantile itself when the method is 'quantile')" is one of them. Unwrapped, a form layout gives the name every pixel it asks for and pushes the field it names off the side of the panel, which is a control the user cannot reach.

## VolcanoExplorer._combo

### lines 827-828  _(unsure)_

```python
def _combo(self, options, caption: str, *, labels=None) -> QComboBox:
```

Each factory stores the caption on the widget so _group can label it without a parallel table that can fall out of step.

## VolcanoExplorer._repopulate_column_menus

### lines 992-995

```python
try:
```

pandas 3 normalizes a ``None`` column label to ``nan``. Keep the menu contract stable: an unnamed column is shown as "None" and, unlike the em-dash sentinel above, falls back to that visible text in ``_style_choices``.

## VolcanoExplorer._pull_style_from_controls

### lines 1060-1061

```python
continue
```

Shown, not edited: reading a label back would write its printed form over the value it was printed from.

## VolcanoExplorer.refresh

### lines 1122-1125

```python
self._figure.clear()
```

Nothing drew, not even the defaults: there has never been a good style to fall back to. An empty frame rather than an error over the plot -- the reasons are on the line under it, and they are the same reasons either way.

## VolcanoExplorer._error_ink

### line 1178

```python
@staticmethod
```

broken settings

## VolcanoExplorer._show_problems

### line 1236

```python
if section is not None and not section.is_expanded():
```

A RED LABEL INSIDE A CLOSED SECTION IS A RED LABEL NOBODY SEES.

## VolcanoExplorer.nearest_point

### lines 1311-1312

```python
return index if distance[index] <= 0.05 else None
```

Ignore clicks on empty space: 5% of the diagonal is about the radius of a marker at default size.

## VolcanoExplorer.export

### lines 1381-1388

```python
from .figure_settings import save_figure_as
```

SVG IS OFFERED HERE AND THE ONE WRITER CANNOT KEEP IT. The renderer's `save_path` goes through `spacr.plot.save_figure`, which writes PNG and PDF and CORRECTS the file's extension to whichever it wrote -- so an .svg asked for here was written as a PDF beside it and this handed back the .svg path, naming a file that had never been created. `save_figure_as` is the writer that already answers for the vector formats, print rule included, and it reports the path it actually wrote.

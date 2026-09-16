# Notes from `spacr/qt/widgets/measurement_compare_dialog.py`

Prose lifted out of `spacr/qt/widgets/measurement_compare_dialog.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_WellChoice.__init__](#_wellchoice__init__) (2 entries)
- [MeasurementComparePanel](#measurementcomparepanel) (1 entry)
- [MeasurementComparePanel.__init__](#measurementcomparepanel__init__) (13 entries)
- [MeasurementComparePanel.chosen_wells](#measurementcomparepanelchosen_wells) (1 entry)
- [MeasurementComparePanel.join_the_tables](#measurementcomparepaneljoin_the_tables) (3 entries)
- [MeasurementComparePanel._finish_join](#measurementcomparepanel_finish_join) (1 entry)
- [MeasurementComparePanel._on_scope](#measurementcomparepanel_on_scope) (1 entry)
- [MeasurementComparePanel._heading](#measurementcomparepanel_heading) (2 entries)
- [MeasurementComparePanel._join_the_dependent_variable](#measurementcomparepanel_join_the_dependent_variable) (1 entry)
- [MeasurementComparePanel._draw](#measurementcomparepanel_draw) (1 entry)
- [MeasurementComparePanel._live_plot](#measurementcomparepanel_live_plot) (1 entry)
- [MeasurementComparePanel._report](#measurementcomparepanel_report) (1 entry)
- [MeasurementCompareDialog.refresh](#measurementcomparedialogrefresh) (1 entry)

## _WellChoice.__init__

### lines 58-62

```python
box = Toggle(str(well), self)
```

`Toggle`, not a bare check box: it subclasses one, so nothing about the behaviour changes, and it is what every other boolean in spaCR looks like. A test greps the Qt package for the bare constructor precisely to stop the two drifting -- which is why this comment does not write it out.

### lines 64-66

```python
box.setChecked(chosen is None or str(well) in chosen)
```

EVERYTHING ON BY DEFAULT. `None` means "all of them", and a panel that opened with nothing ticked would read as "nothing is being compared", which is not what was happening.

## MeasurementComparePanel

### lines 98-103

```python
class MeasurementComparePanel(QWidget):
```

THE ARGUMENTS ARE DOCUMENTED ON `__init__`, ONCE. They were listed here too, as a NumPy ``Parameters`` section, and AutoAPI runs with ``class_content='both'``: the class docstring and ``__init__``'s are concatenated before Napoleon sees them, the section became a field list, and ``__init__``'s opening prose then ended it mid-way -- "Field list ends without a blank line", which `sphinx-build -W` makes fatal.

## MeasurementComparePanel.__init__

### lines 144-149

```python
from ..job_runner import JobRunner
```

THE JOIN RUNS OFF THE GUI THREAD. It reads every object table out of every attached database and joins them onto the crop rows: measured at 3.2 s for one plate's 553 objects, so a four-plate screen of 60,000 is minutes with the window frozen solid reported as "pressing join the measurements table in the cell tab in the regression module makes spacr unresponsive".

### lines 156-159

```python
self._chosen_wells: Optional[set] = None
```

THE WELLS THE USER LEFT IN. `None` means "all of them", which is not the same as the full list: a well that appears after a re-run should be included, and a stored full list would silently exclude it.

### lines 178-180

```python
for name in self._numeric_columns():
```

OFFERED FROM THE DATA, never typed: the same rule every other chooser in spaCR follows, and the reason `object_array` stopped being a text box.

### lines 186-189

```python
self.operator = QComboBox()
```

THE SECOND MEASUREMENT AND THE OPERATOR (179 B). "one mes minus, plus, multiplied by or devided by another mes" -- and the combined column is named for the expression, so the table, the legend and the settings file all say the same thing.

### line 207, trailing  _(unsure)_

```python
self.level.setCurrentIndex(1)
```

well: the unit the screen randomises

### lines 219-232

```python
self.spread = QComboBox()
```

WHAT THE WHISKER MEANS, WHEN THERE IS ONE. "for the cell table graphs if bar is chosen the user should be able to choose SD, Var, or SEM error bars."

THEY ARE NOT INTERCHANGEABLE: SD describes the cells, SEM the confidence in their mean, and at n=3000 they differ by a factor of fifty-five -- a reader who assumes the wrong one reads a real effect as noise or noise as a real effect. So the caption under the plot names the one that was drawn, not only this box.

ABSENT, NOT INERT, for a graph type that has no error bar (106): a box already draws its quartiles and a jitter draws every point, and a control that cannot change either is a promise that was kept once.

### lines 247-252

```python
row.addWidget(self._heading("show"))
```

B2, asked for 2026-08-20: "it should be possible to show only one class". A FILTER ON THE DRAW, not on the build -- the statistics below still describe the whole comparison, because a test computed on one of two groups is not a comparison at all and a panel that quietly re-ran it on the visible half would be reporting a different question than the one on screen.

### lines 254-258

```python
from ...well_scope import SCOPES
```

THE THREE POPULATIONS (instruction 205). Beside the class filter rather than merged into it: "which objects are on the plot" and "which of the classes on it to draw" are different questions, and one box answering both would have a list whose entries mean two different things.

### lines 291-295

```python
second_row = QHBoxLayout()
```

187 B

THE CONTRAST IS A SEPARATE ROW because it is a separate decision. The row above chooses WHAT is measured; this one chooses WHAT IT IS HELD AGAINST, and the same cells under three contrasts give three different p-values.

### lines 306-308

```python
self.controls = QLineEdit()
```

THE CONTROLS, resolved through `spacr.control_names` (184) -- so a gene, a guide, a prefixed name and a bare one all work, and this panel does not grow a fifth opinion about what a control is.

### lines 319-326

```python
self.wells_button = QPushButton("wells…")
```

AND WHICH OF THE GENE'S WELLS COUNT. "i whould be able to choose which wells to include from the gene annotation" -- a well that failed for an unrelated reason should not have to poison the contrast.

A CHECKLIST RATHER THAN 185's PLATE MAP: an annotation's wells span plates, and a plate map can only show one plate at a time. The NAMES are the same either way, which is the part that had to agree.

### lines 335-340

```python
self._join_row = QHBoxLayout()
```

187 A

THE JOIN IS OFFERED, NOT SILENTLY SKIPPED. `png_list` holds the crop path and the classification score; every morphological measurement is in the object tables beside it. Offering a short list of measurements with no reason for its shortness is what this is against.

### lines 355-358

```python
from PySide6.QtWidgets import QCheckBox
```

TWO MORE BOXES (instruction 213 A). `png_list` and the dependent variable are already the substance of the analysis and neither could be reached from this control -- which offered the four object tables and no reason for stopping there.

## MeasurementComparePanel.chosen_wells

### lines 498-499

```python
return [w for w in self.wells_on_offer() if w in self._chosen_wells]
```

INTERSECTED WITH WHAT IS THERE NOW, so a choice made before a re-run cannot name a well the new montage does not have.

## MeasurementComparePanel.join_the_tables

### line 565

```python
png_list = bool(self.join_png_list.isChecked())
```

READ ON THE GUI THREAD, because a worker may not touch a widget.

### lines 569-571

```python
self.join_button.setEnabled(True)
```

THE BUTTON BECOMES THE CANCEL. A join of a four-plate screen is minutes of reading, and a run that can only be waited out is the freeze this moved off the GUI thread to avoid, one step removed.

### lines 589-601

```python
self._joining = False
```

PUT BOTH BACK. `_joining` and the Cancel label were set

BEFORE the submit, because the submit is the part that takes minutes. Left set after a refusal the panel is stuck: the button offers to cancel a job that is not running, and `_joining` makes every later press return early.

This used to be marked `no cover - JobRunner always returns

True today`, and that reason was wrong. `JobRunner.submit` returns False whenever it is UNTHREADED and the work or the completion callback raises. Only this panel's runner, built threaded, cannot reach it -- so the guard covers a real contract and is driven in tests/qt/test_a_refused_join_puts_the_button_back.py.

## MeasurementComparePanel._finish_join

### lines 653-654

```python
self.set_data(wide, self._groups)
```

THE GROUPS SURVIVE because `join_measurements` keeps the index, and `set_data` re-reads them from the same values.

## MeasurementComparePanel._on_scope

### line 663, trailing  _(unsure)_

```python
self._selected_wells = None
```

derived again from the guides

## MeasurementComparePanel._heading

### lines 711-714

```python
if getattr(self, "_heading_filter", None) is None:
```

HELD ON `self`, not made and dropped. Qt keeps a bare pointer to an event filter, so one that only this call referenced is collected as soon as the call returns -- after which the heading stops responding and nothing says why.

### lines 722-723

```python
LOG.debug("could not install the hover tooltip", exc_info=True)
```

The plain Qt tooltip still shows; only the stay-open behaviour is lost, which is a worse tooltip rather than none.

## MeasurementComparePanel._join_the_dependent_variable

### lines 757-758

```python
return wide, f"the dependent variable did not join: {exc}"
```

REPORTED, NOT SWALLOWED. A route that matches nothing is a failure rather than an empty answer.

## MeasurementComparePanel._draw

### lines 887-898

```python
self._canvas = self._live_plot(showing)
```

PYQTGRAPH, NOT MATPLOTLIB. Asked for four times: "I WANT ALL

FIGURES TO BE IN pyqtgraph NOT matplotlib ... i also want to be able to right click on the graph and change the graph typem to whatever is possibel with the underlying data ... and i want to be able to save the figure intp afolder as stats, data, ong figure and pdf figure".

THE THREE ARE ONE ASK. A plot that HOLDS ITS DATA can be redrawn as any kind the data supports and can write that data and its statistics beside the picture; a rendered Figure in a layout can do neither, which is why the retyping and the folder save had to be bolted on beside it and kept drifting out of reach.

## MeasurementComparePanel._live_plot

### lines 927-929

```python
background=REST)
```

"the rest" is the population the selected genes are being compared AGAINST, so it is grey: the ink goes on the claim, not on what the claim is measured against.

## MeasurementComparePanel._report

### lines 956-957

```python
one_sided = self.nothing_to_compare_against()
```

FIRST AMONG THE REASONS, because it is the one the reader can act on and the one that explains an empty-looking graph.

## MeasurementCompareDialog.refresh

### lines 1060-1061

```python
def refresh(self, *args):
```

The window forwards what the button and the tests ask of it, rather than reimplementing any of it.

# Notes from `spacr/qt/widgets/sweep_panel.py`

Prose lifted out of `spacr/qt/widgets/sweep_panel.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [sweep_inputs](#sweep_inputs) (1 entry)
- [SweepPanel.__init__](#sweeppanel__init__) (5 entries)
- [SweepPanel.start](#sweeppanelstart) (1 entry)
- [SweepPanel.rows](#sweeppanelrows) (1 entry)
- [SweepPanel.selected_gene](#sweeppanelselected_gene) (1 entry)
- [SweepPanel.figure](#sweeppanelfigure) (3 entries)
- [SweepPanel.show_picture](#sweeppanelshow_picture) (2 entries)

## sweep_inputs

### lines 75-76  _(unsure)_

```python
offered = normalise_plate_ids(pd.DataFrame(scores).copy())
```

The scores live in the run's own score CSVs, not in the measurement tables -- and THEY say `pplate1` where the databases say `plate1`.

## SweepPanel.__init__

### lines 154-157

```python
self._level_label = QLabel("rank by")
```

THE HELP GOES ON A NAME, so there has to be one. This combo had no label at all, which is why its tooltip sat on the field: `retarget_field_tooltips` pairs a field with a sibling label and correctly leaves a field that has none alone (113).

### lines 171-174

```python
self._picture_label = QLabel("picture")
```

WHICH PICTURE. The heatmap answers "what moved"; it cannot answer "is this gene just over-represented", "what KIND of thing does it move" or "do its own guides agree", and those are the questions that decide whether a hit is worth following up.

### lines 188-194

```python
leave_out = QHBoxLayout()
```

WHAT TO LEAVE OUT, on its own row. Asked for 2026-08-19: "there should be the option to remove columns befor the sweep and remove specific genes or guides and to remove over represented guides". Three separate controls because they are three different judgements -- a column you do not trust, a gene you already know about, and a guide whose breadth is doing the work its biology is being credited with.

### lines 252-257

```python
self.show_button = QPushButton("Show picture")
```

SHOW THE PICTURE. The chooser and the ten views existed with no way to look at any of them -- `figure()` was reachable only through Save, so the answer to "i ran a measurement sweep how do i see the graphs?" was "you write them to disk and open them yourself". A picture nobody can look at is the same defect as a setter nobody calls.

### lines 285-295

```python
from ..screens.settings_model import retarget_field_tooltips
```

HOVER HELP BELONGS ON THE SETTING'S NAME (instruction 113): a tooltip on the control fires while the user is using it, which is the one moment they did not ask for it.

AT THE END OF `__init__`, which is the whole contract -- this call was at the end of `_refill`, so the panel's help moved only once a sweep had been loaded, and a user reading the controls BEFORE running anything (which is when a control's help is worth having) got the tooltip over the field they were typing into. The cross-screen audit found it as two QLineEdits on this panel and nowhere else, because every other screen calls it where it says to.

## SweepPanel.start

### lines 323-325

```python
scores = None
```

THE WORKER TOUCHES NO WIDGET. It returns the result object and the GUI thread does the rest -- which is the rule a regression broke today by building Qt widgets on its own worker.

## SweepPanel.rows

### lines 389-391

```python
bar = 0.15 if (self.hide_circular.isChecked()
```

A CIRCULARITY BAR THE RESULT CANNOT HONOUR IS REFUSED, not silently applied to a column of NaN -- which returns nothing and looks like an answer.

## SweepPanel.selected_gene

### line 454, trailing  _(unsure)_

```python
item = self.table.item(sorted(rows)[0], 1)
```

the gene column

## SweepPanel.figure

### lines 498-500

```python
chosen = str(self.level.currentData() or "gene")
```

The picture follows the level the panel is showing, so a "both" sweep does not draw a gene and its own guides as if they were independent agreement.

### lines 519-523

```python
gene = self.selected_gene()
```

THE ONE VIEW THAT NEEDS A SUBJECT. The row the user selected in the table is the gene they are asking about; with nothing selected the strongest survivor is the honest default, and it is named in the title so nobody mistakes it for a choice they made.

### lines 536-539

```python
return plot_guide_concordance(self._result, path=path,
```

NOT given `level`: this picture IS the guide comparison, and passing the panel's gene default would leave it nothing to compare. It says so by drawing nothing when the sweep was run at gene level.

## SweepPanel.show_picture

### lines 568-569

```python
self.status.setText(
```

NOTHING TO DRAW IS AN ANSWER, and it is said here rather than by opening an empty window -- which reads as a broken button.

### lines 586-587  _(unsure)_

```python
self._pictures.append(dialog)
```

KEPT, or Python collects the dialog the moment this returns and the window vanishes as it appears.

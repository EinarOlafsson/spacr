# Notes from `spacr/qt/widgets/refit_dialog.py`

Prose lifted out of `spacr/qt/widgets/refit_dialog.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [RefitDialog.__init__](#refitdialog__init__) (5 entries)
- [RefitDialog._refresh](#refitdialog_refresh) (2 entries)

## RefitDialog.__init__

### lines 38-40

```python
from ...regression_spec import (REGRESSION_SETTINGS_USED,
```

From the spec, not from ml: importing ml here would pull torch, cv2 and IPython onto the GUI thread the moment the user opens this dialog. The tables are the same objects; ml re-exports them.

### lines 52-57

```python
intro = QLabel(
```

WRAPPED, like `_notice` below it. Without this the sentence is one line and the dialog is either too wide for its content or clipping the end of it -- measured at 828 px of room for 875 px of text at font scale 1, and 1656 for 1752 at scale 2. It is prose, so it wraps; eliding would lose the half that says the current run is safe, which is the reassurance it exists to give.

### lines 67-69

```python
self._type.addItem("as before", None)
```

"As before" first, because changing ONLY the correction is a real request -- comparing thirteen corrections on one fit is what the results-folder rule was written for.

### lines 78-85

```python
from ...settings import REGRESSION_LEVELS
```

LEVEL, BECAUSE IT IS WHAT TURNS A BLUP INTO AN ESTIMATE. A mixed fit makes the guide a RANDOM effect, so its guide rows are shrunken predictions with no p value and the guide volcano has nothing to draw. The question that follows -- "how do I get a p value per guide" -- is answered by re-fitting at guide level with a fixed-effect model, and until now the dialog could change the model but not the level, so the answer was out of reach from the panel where the question arises.

### lines 115-117

```python
self._fdr_alpha = QDoubleSpinBox()
```

THE SIGNIFICANCE LEVEL, which is not the penalty weight below it despite `alpha` being the name of both in the settings. This one cuts the hit list; that one changes the model.

## RefitDialog._refresh

### lines 189-191

```python
self._alpha.setEnabled("alpha" in self._used.get(chosen, ()))
```

A penalty weight on an unpenalised model is not ignored, it is REFUSED -- so the box is disabled rather than left to produce a number the run will reject.

### lines 204-214

```python
lines.append(
```

THE OLD FALLBACK SAID THE OPPOSITE OF THE TRUTH, and said it only here: "Nothing to change ... would repeat the run you are looking at" was reachable ONLY when `where` was None, because a resolved destination always adds a line of its own. So the one case where the dialog could not work out where the output would go was the one case where it promised nothing would happen with the button still enabled to start a real fit.

destination() returns None when no count-data path resolves or the results folder cannot be created, so there is nothing to describe and nothing safe to start. Refuse, and say why.

# Notes from `spacr/qt/screens/classifier_evaluation.py`

Prose lifted out of `spacr/qt/screens/classifier_evaluation.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [ClassifierEvaluationScreen.__init__](#classifierevaluationscreen__init__) (2 entries)
- [ClassifierEvaluationScreen._build_confusion_page](#classifierevaluationscreen_build_confusion_page) (2 entries)
- [ClassifierEvaluationScreen._error_column](#classifierevaluationscreen_error_column) (1 entry)

## Module level

### lines 61-69

```python
from ..bridge import make_thread
```

`spacr.classifier_evaluation` is NOT imported here, and the two functions it owns are imported inside the worker bodies that call them instead. It reads `sklearn.metrics` at its top, which reads `scipy.sparse`, which is the whole of both libraries and the better part of a second -- and this screen module is one of `theme.WIDGET_QSS_MODULES`, so every launch imported it to collect a stylesheet block, whether or not anybody ever opened Classifier Evaluation. Both call sites are already inside a background worker, so the import is paid off the GUI thread by the user who asked for the scan.

### lines 190-191

```python
register_widget_qss(TABS_NAME, _tabs_qss, replace=True)
```

``replace=True``: this module owns the name, so a reimport re-registers rather than raising and leaving the tabs unstyled.

## ClassifierEvaluationScreen.__init__

### lines 239-240  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

### lines 243-245

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## ClassifierEvaluationScreen._build_confusion_page

### lines 356-358  _(unsure)_

```python
def _build_confusion_page(self) -> QWidget:
```

C8 — the confusion matrix as a set of live queries

### lines 390-393

```python
self._threshold = QDoubleSpinBox(inspector)
```

Where "sure" starts is a property of the assay, not of arithmetic — see `spacr.confusion.confidence_threshold`. Exposed rather than baked in, because the person reading the crops is the one who can tell whether 0.75 is where their model stops guessing.

## ClassifierEvaluationScreen._error_column

### lines 455-456  _(unsure)_

```python
return box, listing, button
```

`heading` is kept on the layout only for the caller's convenience; the layout itself is what gets added, so nothing here is orphaned.

# Notes from `spacr/qt/widgets/annotation_strategy_panel.py`

Prose lifted out of `spacr/qt/widgets/annotation_strategy_panel.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [AnnotationStrategyPanel.__init__](#annotationstrategypanel__init__) (6 entries)
- [AnnotationStrategyPanel._on_strategy_changed](#annotationstrategypanel_on_strategy_changed) (2 entries)
- [AnnotationStrategyPanel.reason](#annotationstrategypanelreason) (1 entry)
- [AnnotationStrategyPanel._refresh_controls](#annotationstrategypanel_refresh_controls) (1 entry)

## AnnotationStrategyPanel.__init__

### lines 122-124

```python
self._result = None
```

EVERY PIECE OF STATE A CONTROL READS EXISTS BEFORE A SIGNAL IS CONNECTED, which is this package's rule for a widget whose handlers fire during construction.

### lines 132-135

```python
controls_host = QWidget()
```

THE CONTROLS SCROLL, THE REPORT DOES NOT. This panel is a tab in the left half of the figures splitter, which is often 500 px tall; without a scroll area the two forms are squashed until their rows overlap and every value on screen is unreadable.

### lines 358-362

```python
for box in (self._menu, self._split, self._leakage, self._model,
```

THE PANEL SITS IN THE LEFT HALF OF THE FIGURES SPLITTER, which starts at 780 px and floors at 520. The prose in these choosers is what a combo measures itself by, so left alone the widget's minimum width forces the whole regression screen wider. The words live in the entries and the tooltips; the boxes elide.

### lines 373-375

```python
for prose in (self._about, self._status):
```

A word-wrapped label asks for the width its longest paragraph wants. These two are paragraphs, so they say how narrow they can be and wrap instead.

### lines 386-393

```python
self._filled: Dict[str, bool] = {}
```

A FIELD THAT IS THE REASON HAS TO CLEAR THE REASON. Naming the control wells is what makes the anchor strategy runnable, and a button that stayed grey until something else happened would read as a button that does not work.

ON THE EMPTINESS, NOT ON EVERY KEYSTROKE: re-checking asks the montage for its object rows, and a montage over a plate is not a thing to concatenate once per character typed.

### lines 402-405

```python
from ..screens.settings_model import retarget_field_tooltips
```

HOVER HELP BELONGS ON A SETTING'S NAME, not on the field the user is about to type into. Run BEFORE the first greying, because the greying puts its reason on the field and marks it so this pass leaves that reason where it can be read.

## AnnotationStrategyPanel._on_strategy_changed

### lines 484-485  _(unsure)_

```python
label.setToolTip(self._row_help.get(key, ""))
```

The help goes back where the hover-help pass put it: on the name, not on the field.

### lines 493-494

```python
widget.setProperty(DISABLED_REASON_TOOLTIP, True)
```

ON THE FIELD AS WELL AS THE NAME, and marked so the hover-help pass does not move a disabled control's reason off it.

## AnnotationStrategyPanel.reason

### lines 542-544

```python
LOG.debug("could not pre-flight the strategy", exc_info=True)
```

The execution path repeats this validation and reports a specific error. A preflight exception must not disable an otherwise valid strategy because of an unusual column dtype.

## AnnotationStrategyPanel._refresh_controls

### lines 561-564

```python
if reason and self._result is None and not self._running:
```

A TAB THAT CANNOT BE FILLED SAYS WHY. The panel is present from the moment the Cells tab is built, so before a montage has loaded the reason is the only thing on it worth reading -- and it must not overwrite the report of a run that has already happened.

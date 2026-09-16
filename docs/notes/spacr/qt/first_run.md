# Notes from `spacr/qt/first_run.py`

Prose lifted out of `spacr/qt/first_run.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_TourOverlay.__init__](#_touroverlay__init__) (4 entries)
- [_TourOverlay.paintEvent](#_touroverlaypaintevent) (4 entries)

## _TourOverlay.__init__

### line 233  _(unsure)_

```python
self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
```

Full-window frameless overlay

### line 239  _(unsure)_

```python
self._card = QWidget(self)
```

Step card

### lines 258-259

```python
self._step_lbl = QLabel(tr("Step {n} / {total}", n=1,
```

THE STEP COUNTER IS COMPOSED FROM A TEMPLATE, so the catalog is asked for a key that exists rather than for the numbers baked in.

### line 285  _(unsure)_

```python
btn_row = QWidget()
```

Buttons

## _TourOverlay.paintEvent

### line 323  _(unsure)_

```python
p.fillRect(self.rect(), QColor(0, 0, 0, 170))
```

Dim overlay

### line 326  _(unsure)_

```python
highlight_fn = self._steps[self._idx].highlight
```

Cut a hole around the highlighted widget, if any

### line 334  _(unsure)_

```python
p.setBrush(Qt.transparent)
```

Draw a bright ring around it

### lines 340-341  _(unsure)_

```python
p.setCompositionMode(
```

Clear the dimming inside the ring so users see the widget in its natural colour.

# Notes from `spacr/qt/widgets/training_monitor.py`

Prose lifted out of `spacr/qt/widgets/training_monitor.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [TrainingMonitor](#trainingmonitor) (1 entry)
- [TrainingMonitor.append](#trainingmonitorappend) (2 entries)

## TrainingMonitor

### lines 28-43

```python
class TrainingMonitor(QWidget):
```

NO ``Attributes`` SECTION, AND THAT IS THE FIX RATHER THAN A STYLE CHOICE. AutoAPI runs with ``class_content='both'``, so this docstring and ``__init__``'s are concatenated before Napoleon sees them, and a trailing ``Attributes`` section swallows whatever follows it: Sphinx emitted `.. attribute:: Build the panel that follows a training run's losses and metrics.` and then bare `.. attribute::` directives carrying `:type: param parent: ...`. That is an ERROR -- "1 argument(s) required, 0 supplied" -- and `sphinx-build -W` fails the docs job on it.

TWO WEAKER FIXES WERE TRIED AND MEASURED, both on the real build: a closing paragraph after the section (the ERROR survived), and making ``__init__`` NumPy-sectioned so the two agreed on a style (it became TWO errors). The section itself is the problem, so the two attributes are prose. They are still documented; they are simply not a Napoleon section in a class whose docstring is about to have another one glued to it.

## TrainingMonitor.append

### line 123  _(unsure)_

```python
continue
```

Omitting the point represents a non-finite epoch as a gap.

### line 130  _(unsure)_

```python
curve.setData(xs, ys)
```

Updating the existing item preserves the view and legend.

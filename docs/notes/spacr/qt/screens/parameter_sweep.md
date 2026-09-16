# Notes from `spacr/qt/screens/parameter_sweep.py`

Prose lifted out of `spacr/qt/screens/parameter_sweep.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_make_screen.ParameterSweepScreen.__init__](#_make_screenparametersweepscreen__init__) (7 entries)
- [_make_screen.ParameterSweepScreen.apply_settings](#_make_screenparametersweepscreenapply_settings) (2 entries)
- [_make_screen.ParameterSweepScreen.start](#_make_screenparametersweepscreenstart) (1 entry)
- [_make_screen.ParameterSweepScreen._on_row_activated](#_make_screenparametersweepscreen_on_row_activated) (2 entries)
- [_make_screen.ParameterSweepScreen._trial_figures_ready](#_make_screenparametersweepscreen_trial_figures_ready) (1 entry)
- [_make_screen.ParameterSweepScreen._show](#_make_screenparametersweepscreen_show) (1 entry)
- [_make_screen](#_make_screen) (1 entry)
- [Module level](#module-level) (1 entry)
- [build_parameter_sweep_card](#build_parameter_sweep_card) (1 entry)

## _make_screen.ParameterSweepScreen.__init__

### lines 87-88  _(unsure)_

```python
"""Build the sweep screen. ``host`` is the window, not a Qt parent."""
```

`host` is the main window the registry passes for navigation, not a Qt parent.

### lines 134-135  _(unsure)_

```python
include.setChecked(key not in (
```

Default on for the axes the comparison is actually about; the filtration cutoffs are usually pinned by the user.

### lines 179-184

```python
from ...parameter_sweep import (
```

WHETHER THE KERNEL CAP IS ACTUALLY THERE (114 point 1). The sweep took this user's desktop down seven times, and the only thing that has ever held is the kernel enforcing a ceiling so a screen that let them believe a cap existed when it does not would be sending them back into exactly that. Red when there is none, because it changes what they should do next.

### lines 256-263

```python
self.table.setSelectionBehavior(QTableWidget.SelectRows)
```

CLICK A ROW TO GET THAT REGRESSION BACK.

A sweep row carries every setting its trial was given, so it is enough to reproduce the trial exactly. Running it here rather than opening the saved page matters: these come back as live matplotlib Figures, so they land in the figure queue below and can be restyled -- thresholds, colours, legend, axis limits which is the whole reason for looking at a condition again.

### lines 292-297

```python
from ..widgets.regression_results import RegressionResultsPanel
```

THE WHOLE SET OF GRAPHS FOR THE CLICKED ROW.

Re-running was the expensive half and it already worked, but showing one figure at a time in a queue means the user still cannot put a run's residual plot beside its volcano -- which is the comparison that decides whether a configuration is any good.

### line 306, trailing  _(unsure)_

```python
self.figures.hide()
```

shown only when a re-run makes figures

### lines 312-317

```python
from ..dnd import install_for
```

Every screen that reads a path takes a drop, and a sweep reads more paths than anything else in spaCR -- a plate per pair. The policy is SweepInputsDropHandler, which sorts each CSV into the score or the count list from its header. install_for never raises: a Qt build without drag-and-drop loses the convenience, not the screen.

## _make_screen.ParameterSweepScreen.apply_settings

### lines 381-383

```python
self.destination.setText(os.path.join(str(source), "sweep"))
```

Beside the data rather than inside it: a sweep writes thousands of folders and they should not be mixed in with the user's inputs.

### lines 392-393  _(unsure)_

```python
editor.setText(f"{text}, {editor.text()}")
```

Keep the user's value in the swept range, so their own condition is one of the trials that gets run.

## _make_screen.ParameterSweepScreen.start

### line 494

```python
self._runner.submit(job, self._sweep_finished)
```

Bound method, so the handler runs on the GUI thread.

## _make_screen.ParameterSweepScreen._on_row_activated

### lines 537-538

```python
key_item = self.table.item(row_index, 0)
```

The table may be sorted, so trust the trial_id in the row rather than the table's row number.

### lines 562-566

```python
folder = record.get("folder")
```

A SAVED RUN IS INSTANT; A RE-FIT IS A MINUTE.

The trial wrote its results when the sweep ran, so prefer them. Re-fitting to see something already on disk is a minute of waiting for an identical answer.

## _make_screen.ParameterSweepScreen._trial_figures_ready

### lines 602-603

```python
output = payload.get("output") or {}
```

The re-fit wrote its results too, so the full panel can show them exactly as it would for a saved trial.

## _make_screen.ParameterSweepScreen._show

### lines 626-631

```python
"""Put the results in the table, useful columns first.
```

The columns worth reading first, when they exist. The rest are still in the CSV; this is a view, not a filter. Settings first, then WHAT WENT IN, then what came out. A hit count means little without the size of the design it came from: two trials differing only by a filtration cutoff can fit completely different data.

## _make_screen

### lines 658-674

```python
from ..theme import clear_container_surfaces, make_transparent
```

THE BLACK BOX BEHIND THE SWEEP, reported 2026-09-04: the regression module's parameter sweep had a black box background where it should have been transparent. (Paraphrased rather than quoted: the report misspelt "parameter", and the suite keeps that spelling out of the package so the back-compat alias stays the only place it appears which is why this comment cannot name the test that does it either.)

Measured, the screen rendered over magenta: the screen came back `#000000` and so did its `QSplitter`. Both are plain `QWidget`s with no QSS rule of their own, so both take the blanket `QWidget { background-color: bg }` -- which is the window colour, not a surface, so no value of the page-opacity preference could ever reach them and the panel sat as a slab over the animated backdrop.

`clear_container_surfaces` tags what is UNDER a root, splitters by type; the root itself needs `make_transparent`, which is why both are here. 135 containers were tagged on this screen.

## Module level

### lines 681-686

```python
SWEEP_TOGGLE_TEXT = "Parameter sweep"
```

NO REGISTRY ROW. The sweep is reached as the Regression screen's sweep card :func:`build_parameter_sweep_card`, built from the same :func:`_make_screen` factory the tile used. The card is the superset of the two: it seeds its axes from the regression form beside it, which a standalone tile has nothing to seed from. The strings above are kept because they are this module's public description and the i18n catalogs carry them.

## build_parameter_sweep_card

### lines 714-725

```python
holder = _lazy_sweep_panel(host)
```

BUILT WHEN IT IS FIRST SHOWN, not when the Regression screen is built.

The panel is a whole second screen: it carries its own results panel, and that panel builds ELEVEN pyqtgraph plots. Measured on the Regression screen, those eleven were 0.32 s of a 0.88 s construction -- a third of the cost of opening the module, paid by every user, for a card that starts collapsed behind a toggle and that most runs never open.

Nothing is switched off: the card, the toggle and the panel are all exactly as they were, and the first time the card is opened the panel is there. This is the optimisation the laptop item asks for rather than the feature removal it calls the fallback.

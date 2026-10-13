# Notes from `spacr/qt/screens/train_compare.py`

Prose lifted out of `spacr/qt/screens/train_compare.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [panel_canvas_class.PanelCanvas.__init__](#panel_canvas_classpanelcanvas__init__) (1 entry)
- [TrainCompareScreen.__init__](#traincomparescreen__init__) (2 entries)
- [TrainCompareScreen._build_ui](#traincomparescreen_build_ui) (3 entries)
- [TrainCompareScreen._apply_runs](#traincomparescreen_apply_runs) (1 entry)
- [TrainCompareScreen.root](#traincomparescreenroot) (1 entry)
- [TrainCompareScreen._clear_plot](#traincomparescreen_clear_plot) (1 entry)
- [TrainCompareScreen._style_axes](#traincomparescreen_style_axes) (1 entry)
- [TrainCompareScreen._fill_diff](#traincomparescreen_fill_diff) (1 entry)
- [TrainCompareScreen._run_job](#traincomparescreen_run_job) (1 entry)

## panel_canvas_class.PanelCanvas.__init__

### lines 173-174

```python
figure.patch.set_alpha(0.0)
```

The panel below is the surface now. Leaving the patch opaque would paint the old rectangle straight back over it.

## TrainCompareScreen.__init__

### lines 249-250  _(unsure)_

```python
self._jobs: List[tuple] = []
```

Ownership list for in-flight (QThread, worker) pairs — a QThread collected while still running takes the process down with it.

### lines 262-264

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## TrainCompareScreen._build_ui

### line 288  _(unsure)_

```python
src_row = QHBoxLayout()
```

── Source row ────────────────────────────────────────────────

### line 306  _(unsure)_

```python
split = QSplitter(Qt.Horizontal, self)
```

── Runs | plot + diff ────────────────────────────────────────

### lines 355-356

```python
from matplotlib.figure import Figure
```

Matplotlib canvas — created here so the same figure is reused for every overlay rather than leaking one per click.

## TrainCompareScreen._apply_runs

### lines 485-488

```python
self._clear_plot()
```

Clear the old tree's curves before any visible state starts naming the new tree. If Qt rejects the redraw because its canvas has been deleted, no new run state has been installed and no old curve mapping survives behind it.

## TrainCompareScreen.root

### line 557  _(unsure)_

```python
def root(self) -> str:
```

introspection helpers (used by tests and by callers)

## TrainCompareScreen._clear_plot

### lines 739-741

```python
self._figure.patch.set_alpha(0.0)
```

`clear()` restores the rc facecolor AND its alpha, so the transparency `PanelCanvas` set has to be re-asserted or the first redraw paints the opaque rectangle straight back.

## TrainCompareScreen._style_axes

### lines 767-769

```python
ax.patch.set_facecolor(pal["surface_alt"])
```

The axes keep a fill — the plotting area is meant to read as a panel within the panel — but at the page opacity, so the slider reaches the plot too rather than stopping at its frame.

## TrainCompareScreen._fill_diff

### line 817

```python
self._diff_summary.setText(
```

An empty table reads as a failure; say it in words instead.

## TrainCompareScreen._run_job

### lines 963-967

```python
worker.finished.connect(self._on_job_settled)
```

A bound QWidget receiver gives Qt enough thread affinity information to queue this callback onto the GUI thread. Connecting to a plain closure can execute it in the worker thread, where the calls below that mutate labels, tables and Matplotlib canvases are undefined behaviour.

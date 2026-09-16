# Notes from `spacr/figures/fast_render.py`

Prose lifted out of `spacr/figures/fast_render.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [renderer_for](#renderer_for) (1 entry)
- [_pyqtgraph_ready](#_pyqtgraph_ready) (2 entries)
- [build_fast_plot](#build_fast_plot) (3 entries)
- [render_panel](#render_panel) (1 entry)
- [_render_with_matplotlib](#_render_with_matplotlib) (1 entry)
- [write_panels](#write_panels) (1 entry)

## renderer_for

### lines 113-131

```python
def renderer_for(key: str, force: Optional[str] = None) -> tuple:
```

THERE IS NO DETECTION HERE, AND THAT IS THE DESIGN. Two attempts at asking "is the GUI up?" were built and both were wrong, so the question is not asked at all: a scene is rendered when a caller HANDS ONE IN, and otherwise the page that has always been written is written.

"a QApplication exists" is false. Measured 2026-08-18: matplotlib's QtAgg backend -- the DEFAULT backend in this environment -- calls `_create_qApp` from inside `plt.figure()` and constructs a `QApplication(["matplotlib"])`. So the first matplotlib panel of a headless run created one, and every panel after it saw a live QApplication and switched renderer. One run, seven figures, two libraries: a worse disagreement than the one this module removes. "`spacr.qt.widgets.fast_plots` is in sys.modules" is also false. A test that does nothing but check the seven classes exist puts it there, and so does any import of the widget package. Module presence is not evidence that a plot was ever built, let alone that one is on screen.

The unambiguous fact is the widget itself. `render_panel(..., plot=widget)` renders that widget; nothing else can be mistaken for it.

## _pyqtgraph_ready

### lines 173-179

```python
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
```

PySide6 IS NAMED BEFORE pyqtgraph IS IMPORTED, AND THE ORDER IS LOAD-BEARING. pyqtgraph binds to the first of PyQt5, PyQt6, PySide2, PySide6 that imports, and PyQt6 is installed in this environment -- so `import pyqtgraph` first loads PyQt6's libQt6Core and PySide6 6.11 then cannot load at all: "libpyside6.abi3.so.6.11: undefined symbol: _ZN9QtPrivate9sizedFreeEPvm". Measured 2026-08-18 on the generated-figure path, where it silently sent a whole QC suite back to matplotlib.

### lines 193-195

```python
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
```

MEASURED, not assumed: with this set, `ImageExporter` writes a PNG and `QPdfWriter` + `scene().render()` writes a real vector PDF on a machine with no display at all.

## build_fast_plot

### lines 203-205  _(unsure)_

```python
def build_fast_plot(key: str, frame, *, alpha: float = 0.05):
```

Feeding a scene from a coefficient table

### lines 249-253

```python
rows = tested(frame)
```

THE FAMILY, NOT THE AXIS. The house-style panel histograms the TESTED coefficients; handing this one the whole table would put the intercept and the nuisance terms into a picture whose caption says "the tested coefficients", which is the two renderers disagreeing about what the figure is OF.

### lines 269-270  _(unsure)_

```python
from ..guide_concordance import guide_support
```

FAST_PANELS is exhaustive; this is the former

``elif key == "agreement":`` arm after the six keys above.

## render_panel

### lines 346-348

```python
chosen, why = "pyqtgraph", ""
```

A live widget IS the pyqtgraph answer. Being handed one and then drawing matplotlib because no QApplication was detected would be absurd: the widget could not exist without one.

## _render_with_matplotlib

### lines 427-430

```python
plt.close(figure)
```

`build_panel` goes through `plt.figure`, so pyplot holds a reference and `clf()` would clear the figure without releasing it. Seven panels a run, leaked, is how a long session runs out of memory drawing pictures nobody is looking at.

## write_panels

### lines 455-458

```python
chosen = renderer
```

ONE RENDERER FOR THE WHOLE SET, DECIDED ONCE. Asking per panel is not the same question asked seven times -- see the comment above :func:`renderer_for`, where an earlier per-panel rule drew one run's first figure in matplotlib and its other six in pyqtgraph.

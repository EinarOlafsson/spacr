# Notes from `spacr/qt/screens/align.py`

Prose lifted out of `spacr/qt/screens/align.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [TileLayoutWidget.__init__](#tilelayoutwidget__init__) (1 entry)
- [TileLayoutWidget.paintEvent](#tilelayoutwidgetpaintevent) (2 entries)
- [AlignScreen.__init__](#alignscreen__init__) (1 entry)
- [AlignScreen._build_ui](#alignscreen_build_ui) (2 entries)
- [AlignScreen.active_jobs](#alignscreenactive_jobs) (1 entry)
- [AlignScreen._run_job](#alignscreen_run_job) (2 entries)

## Module level

### lines 66-74

```python
_PAD = 10
```

NOTHING IS FOLDED ONTO THIS MASTHEAD ANY MORE. Optical pooled screening was: OPS is stitching, which is this screen's job, so it was reached from here. It is reached from MASK GENERATION instead, as asked on 2026-09-09 "i think the OPS button should be in Mask generation instead of align" and it is a switch in that screen's actions row beside Live rather than an icon on a masthead. Its declaration, its name, its sentence and its maturity moved with it: see `spacr.qt.screens.mask.PAGE_FOLDS` and `mask.FOLD_FALLBACK`. This screen is consequently no longer a fold host, which is why it is no longer in `fold_strip.FOLD_HOST_MODULES`.

## TileLayoutWidget.__init__

### lines 122-125

```python
make_transparent(self)
```

The panel is drawn in `paintEvent`, so the widget itself must not also paint the blanket window fill underneath it — that fill is opaque and would swallow the backdrop before the translucent panel ever composited over it.

## TileLayoutWidget.paintEvent

### lines 170-173

```python
paint_panel(painter, self, role="surface", inset=0.5)
```

A rounded panel at the page opacity, not `fillRect(..., surface)`: the hex `active_palette` returns carries no alpha, so the empty state used to be the one flat black rectangle on an otherwise see-through page.

### lines 194-199

```python
painter.fillRect(rect, QBrush(QColor(palette["surface"]),
```

Hatch as well as colour: a colour-vision-impaired reader must still be able to count the fallbacks. The hatch is drawn in the *surface* colour, not the warning colour — warning-on-warning is the same colour twice and paints nothing at all, which is exactly as much help to that reader as leaving the hatch out.

## AlignScreen.__init__

### lines 264-266

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## AlignScreen._build_ui

### line 295  _(unsure)_

```python
src_row = QHBoxLayout()
```

── Source row ────────────────────────────────────────────────

### line 410  _(unsure)_

```python
split = QSplitter(Qt.Horizontal, self)
```

── Layout | report ───────────────────────────────────────────

## AlignScreen.active_jobs

### lines 516-521

```python
self._retire_finished_jobs()
```

A queued ``QThread.finished`` signal can outlive its sender's C object: ``make_thread`` schedules deferred thread deletion before this screen installs its retirement slot. In that ordering ``sender()`` intermittently returned None and the dead tuple stayed here for ever. Polling is also a safe recovery path for a queued retirement event: a finished QThread may release its last references now.

## AlignScreen._run_job

### lines 780-782

```python
self._jobs.append((thread, worker))
```

Strong references: PySide6 will not keep the worker alive through the started→run connection alone, and a QThread garbage-collected while still running takes the process down with it.

### lines 788-791

```python
thread.finished.connect(self._retire_finished_jobs)
```

A context-free lambda may run in the emitting worker thread. Use a bound QObject slot so Qt queues retirement onto this widget's GUI thread; otherwise active_jobs() can race the thread's final signal under a loaded full suite.

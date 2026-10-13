# Notes from `spacr/qt/screens/power.py`

Prose lifted out of `spacr/qt/screens/power.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_power_qss](#_power_qss) (1 entry)
- [Module level](#module-level) (3 entries)
- [run_power_sweep](#run_power_sweep) (3 entries)
- [PowerCurveView.__init__](#powercurveview__init__) (1 entry)
- [PowerCurveView.paintEvent](#powercurveviewpaintevent) (2 entries)
- [PowerScreen.__init__](#powerscreen__init__) (1 entry)
- [PowerScreen._build_form](#powerscreen_build_form) (3 entries)
- [PowerScreen._build_output](#powerscreen_build_output) (1 entry)
- [PowerScreen.run](#powerscreenrun) (2 entries)
- [register](#register) (1 entry)

## _power_qss

### lines 185-190

```python
def _power_qss(palette: dict, opacity: Optional[float] = None) -> str:
```

SECTION NOTE, 2026-09-03: the sections were restructured to Core / Data / Tools / Assays, and SECTION_DESIGN / SECTION_EXPLORE / SECTION_RESULTS are still declared but are no longer in SECTION_ORDER. Every screen below now files under Data. The docstrings keep their original reasoning because it still says what each screen IS -- and they are published, translated API prose, so editing them invalidates reviewed translations in nine languages.

## Module level

### lines 227-229

```python
register_widget_qss("PowerDesign", _power_qss, replace=True)
```

`replace=True` and at import: this module is reachable from two paths (the screens package and a direct import), and a second import must refresh the block rather than raise. Same posture as `pivot_builder`.

### lines 1422-1426

```python
_ROW = declared_app(APP_KEY)
```

The row this screen puts in the registry is declared in

`spacr.qt.app_catalog`, which is what lets the app be registered without importing this module -- the launch reads the table, not the screen. These read the same row back rather than restating it, so the name, the blurb and the nine translations have one spelling and no second copy to drift from.

### lines 1611-1614

```python
register_settings()
```

``defaults_module`` means importing this lazily loaded owner establishes its defaults.  Without this call, asking for Power settings after another test imported the screen returned a different inventory from asking in a fresh process.

## run_power_sweep

### lines 233-235  _(unsure)_

```python
def run_power_sweep(payload: Dict[str, Any]) -> Dict[str, Any]:
```

The job — a plain function, so it is testable without a QThread

### lines 300-302

```python
results: Dict[str, Any] = {"cancelled": False, "spec": spec}
```

The spec travels WITH the result. Rendering against whatever is on the form when the sweep lands would label a three-minute run with a design the user edited while waiting for it.

### lines 304-312

```python
with warnings.catch_warnings(record=True) as caught:
```

`always`, not the default `once`: the abundance-clipping warning is emitted from one source line, so the default filter would report the first simulated screen that clipped and silently swallow the other twenty-six — turning "most of this sweep clipped" into "one did".

`catch_warnings` swaps the process-wide filter list, which is not thread-safe. It is used anyway because only one sweep runs at a time and the window is the sweep itself; the cost of getting it wrong is a mis-counted clip warning, not a wrong power.

## PowerCurveView.__init__

### lines 388-389

```python
make_transparent(self)
```

The panel drawn in `paintEvent` is the surface; the widget itself must not also paint the blanket window fill underneath it.

## PowerCurveView.paintEvent

### lines 449-452

```python
paint_panel(painter, self, role="surface", inset=0.5)
```

A rounded translucent panel, not `fillRect(rect, surface)`: that hex carries no alpha, so the fill was opaque by construction whatever the page-opacity preference said, and the two curve views were the only flat rectangles on a page of panels.

### lines 473-474

```python
painter.setPen(QPen(QColor(palette["border_soft"]), 1, Qt.DotLine))
```

y grid at 0 / 0.5 / 1, because a power curve is read against those three and nothing else.

## PowerScreen.__init__

### lines 682-684

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## PowerScreen._build_form

### lines 751-754

```python
self._effect = self._float_box(0.05, 50.0, defaults.effect_fold,
```

The floor is below 1 on purpose. A spin box that refuses the keystroke teaches nothing; one that accepts a protective effect and then says why the model cannot score it teaches the thing worth knowing — see DesignSpec.validate.

### lines 778-782

```python
self._held_note = QLabel("")
```

Everything the simulator needs that the form does not ask for is printed rather than left implicit: a power analysis defended in a methods section needs every number that went into it, and a parameter that only exists in a dataclass default is a parameter nobody knows they accepted.

### lines 815-821

```python
self._setting_fields = {
```

This is a hand-built form, but these are still ordinary registered settings.  Carry the same semantic identity as SettingsWidgets so the post-construction tooltip pass can use the source-hashed SETTING_TOOLTIPS catalog and a later language switch can rebuild the help.  Without these properties retarget_field_tooltips merely moved the authored English string from editor to label; all nine locale translations existed in the catalog and none could reach this form.

## PowerScreen._build_output

### lines 905-907

```python
mark_surface(self._caveats, self._table)
```

The caveat list and the sample-size table are the two regions of this column with nothing behind them; the two curve views between them paint their own panel in `paintEvent`.

## PowerScreen.run

### lines 1091-1096

```python
thread, worker = make_thread(run_power_sweep, payload,
```

journal=False: the sweep reads no user data and writes no files, so there is no artefact for a reproducibility manifest to describe — and a housekeeping job that blocks shutdown is what `RunRegistry.cancel_all` documents as the way to hang a headless run. The record that matters (seed, backend, every parameter) is the DesignSpec, which is on screen.

### lines 1102-1104

```python
worker.line_ready.connect(self._on_worker_line)
```

Bound QWidget methods: Qt queues these back onto the GUI thread. A closure would run on PipelineWorker's thread and must never touch a label or a table.

## register

### lines 1602-1606

```python
register_settings()
```

The declared row names this module as its defaults owner, but a caller has already imported the module in order to call this function.  The generic resolver therefore cannot rely on a later import side effect to install the defaults.  Keep the app row and its settings atomic, as the pre-declaration registration path did.

# Notes from `spacr/qt/screens/model_compare.py`

Prose lifted out of `spacr/qt/screens/model_compare.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [ModelCompareScreen.__init__](#modelcomparescreen__init__) (3 entries)
- [ModelCompareScreen._build_ui](#modelcomparescreen_build_ui) (4 entries)
- [ModelCompareScreen._build_preview](#modelcomparescreen_build_preview) (1 entry)
- [ModelCompareScreen._prepare_table](#modelcomparescreen_prepare_table) (1 entry)
- [ModelCompareScreen._fill_param_table](#modelcomparescreen_fill_param_table) (1 entry)
- [ModelCompareScreen._run_job](#modelcomparescreen_run_job) (1 entry)
- [to_display_gray](#to_display_gray) (1 entry)

## Module level

### lines 234-235

```python
register_widget_qss(MODEL_PANEL_NAME, _model_panel_qss, replace=True)
```

``replace=True``: this module owns the name, and a reimport must re-register rather than raise and leave the panels unstyled.

## ModelCompareScreen.__init__

### lines 390-392

```python
self._jobs: List[tuple] = []
```

Ownership list for in-flight (QThread, worker) pairs — a QThread collected while still running takes the process down with it. Same idiom as AgreementScreen._jobs.

### lines 397-401

```python
ensure_widget_qss_applied(MODEL_PANEL_NAME, root=self)
```

`app.py` imports this module inside the branch that builds the screen, which is long after the launch stylesheet was generated — so the block registered above is not in the sheet that is live and the panels open bare. That is why the fix measured correct in a test and was still black in the running app.

### lines 412-414

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## ModelCompareScreen._build_ui

### line 441  _(unsure)_

```python
src_row = QHBoxLayout()
```

── source row ────────────────────────────────────────────────

### line 490  _(unsure)_

```python
outer.addWidget(QLabel("Parameters that reached each model", self))
```

── resolved parameters ───────────────────────────────────────

### line 499  _(unsure)_

```python
outer.addWidget(QLabel("Per-field comparison", self))
```

── per-field metrics ─────────────────────────────────────────

### line 518  _(unsure)_

```python
preview = QSplitter(Qt.Horizontal, self)
```

── side-by-side masks ────────────────────────────────────────

## ModelCompareScreen._build_preview

### lines 550-553

```python
canvas.setObjectName(PREVIEW_NAME)
```

No inline stylesheet: it used to be

`background: {active_palette()["bg"]}`, raw hex and the window colour, which is opaque by construction. The panel is a rule now (see `_model_panel_qss`), reached by this name.

## ModelCompareScreen._prepare_table

### lines 562-566

```python
table.setObjectName(RESULT_TABLE_NAME)
```

These two tables ARE the containers on this half of the page — nothing else is under them — so they keep a surface where a table sitting on a panel would show it through. The name is what the registered block reaches, and what outranks the transparent tag `clear_container_surfaces` puts on every scroll area.

## ModelCompareScreen._fill_param_table

### lines 836-837

```python
label = "model (requested)" if key == "model" else key
```

'model' appears in both halves — what was asked for and what will load. Two rows with the same label would read as a contradiction.

## ModelCompareScreen._run_job

### lines 994-996

```python
worker.finished.connect(self._on_job_settled)
```

Bound QWidget method: Qt queues this back onto the GUI thread. A closure is invoked directly on PipelineWorker's thread and must never update labels/tables.

## to_display_gray

### lines 1117-1119  _(unsure)_

```python
def to_display_gray(image: Optional[np.ndarray], shape) -> np.ndarray:
```

drawing helpers — plain numpy, so they are testable without a widget

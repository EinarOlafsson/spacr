# Notes from `spacr/qt/screens/hyperparam.py`

Prose lifted out of `spacr/qt/screens/hyperparam.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [parse_values](#parse_values) (1 entry)
- [build_panel_figure](#build_panel_figure) (4 entries)
- [_search_figure_dir](#_search_figure_dir) (1 entry)
- [_SearchWorker](#_searchworker) (2 entries)
- [_SearchWorker.__init__](#_searchworker__init__) (1 entry)
- [_SearchWorker._emit_trial](#_searchworker_emit_trial) (1 entry)
- [_SearchWorker.run](#_searchworkerrun) (1 entry)
- [HyperparamPanel.__init__](#hyperparampanel__init__) (1 entry)
- [HyperparamPanel._build_ui](#hyperparampanel_build_ui) (11 entries)
- [HyperparamPanel._on_adaptive_toggled](#hyperparampanel_on_adaptive_toggled) (1 entry)
- [HyperparamPanel.current_space](#hyperparampanelcurrent_space) (1 entry)
- [HyperparamPanel.request_gpu_enabled](#hyperparampanelrequest_gpu_enabled) (1 entry)
- [HyperparamPanel.run_search](#hyperparampanelrun_search) (3 entries)
- [HyperparamPanel.stop_search](#hyperparampanelstop_search) (1 entry)
- [HyperparamPanel._on_worker_finished](#hyperparampanel_on_worker_finished) (1 entry)
- [HyperparamPanel._on_trial_ready](#hyperparampanel_on_trial_ready) (1 entry)
- [HyperparamPanel._set_row](#hyperparampanel_set_row) (1 entry)
- [UmapSearchSettingsDialog](#umapsearchsettingsdialog) (1 entry)
- [UmapSearchSettingsDialog.__init__](#umapsearchsettingsdialog__init__) (6 entries)
- [UmapSearchSettingsDialog._build_umap_tabs](#umapsearchsettingsdialog_build_umap_tabs) (1 entry)
- [UmapSearchSettingsDialog.propagate_settings](#umapsearchsettingsdialogpropagate_settings) (1 entry)
- [_complete_metrics_when_opened.show_popup](#_complete_metrics_when_openedshow_popup) (1 entry)
- [WalkAxesDialog._style_surfaces](#walkaxesdialog_style_surfaces) (1 entry)
- [WalkAxesDialog._update_cost](#walkaxesdialog_update_cost) (1 entry)

## Module level

### lines 58-60

```python
from ...figures.style import figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

### lines 120-125

```python
"classify_merged": (
```

THE MERGED SCREEN, which lost the search entirely when it took over from both classifiers -- its panel was None and there is no other door to the cross-validated search. The four Torch knobs come first because `classifier_family` defaults to the image classifier; the two gradient-boosting knobs follow, and a search that names one the selected family does not take simply leaves it at its setting.

### lines 140-142

```python
"activation": (
```

Activation sweeps the settings that change the MAP, not the model: which method, which layer it hooks, how much the input is smoothed, and the window / step counts the perturbation and path-integral methods take.

## parse_values

### lines 160-162  _(unsure)_

```python
def parse_values(text: str, kind: str, name: str) -> List[Any]:
```

Pure helpers — no Qt, unit-testable without a display

## build_panel_figure

### lines 356-358

```python
attributed = [t for t in ranked
```

Attribution sweeps first: the maps ARE the deliverable, and the four scores go in every title so the panel shows the criteria disagreeing rather than hiding it behind one ranking.

### lines 365-368

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 404-407

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 434-437

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

## _search_figure_dir

### line 565  _(unsure)_

```python
pass
```

An unwritable or missing src is not a reason to lose the run.

## _SearchWorker

### lines 576-577

```python
trial_ready = Signal(object, int, int, str)
```

(Trial, completed, total, png_path). The path is rendered HERE, on the worker thread, and is "" when the trial has no embedding.

### line 579, trailing  _(unsure)_

```python
search_done = Signal(object, str)
```

(SearchResult or None, error)

## _SearchWorker.__init__

### lines 601-603

```python
self.result: Optional[SearchResult] = None
```

QThread.finished is the lifecycle boundary the panel must wait for. The result signal is emitted just before QThread.run() returns, so it is too early to drop the final Python reference to this object.

## _SearchWorker._emit_trial

### lines 623-624

```python
LOG.debug("could not render trial %s", trial.index, exc_info=True)
```

A figure is decoration; a search that dies because a plot failed would lose hours of real work. INVARIANTS 10.

## _SearchWorker.run

### lines 648-649

```python
self.search_done.emit(self.result, self.error)
```

Kept for callers that use the private worker directly. The panel deliberately consumes the stored payload from ``finished``.

## HyperparamPanel.__init__

### lines 781-783

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## HyperparamPanel._build_ui

### lines 827-839

```python
edit.addItems(UMAP_METRICS)
```

THE SHORT LIST NOW, THE FULL ONE WHEN IT IS OPENED.

`umap_metrics()` reads the metric names off the INSTALLED umap-learn, and touching that package makes numba compile pynndescent: 9.4 s of a 9.6 s screen construction, measured, spent so a dropdown nobody has clicked can be complete.

Nothing is lost. The static names go in immediately so the control is usable at once, and the first time the list is opened it is completed from the installed library -- by which point a user choosing a metric is about to run UMAP and needs it loaded anyway. Anyone who never opens it never pays, which is most of the people opening this screen.

### lines 859-861

```python
run_grid = QGridLayout()
```

run controls. A grid keeps the settings dialog usable at normal laptop widths; the old single horizontal row forced the popup wider than the screen and made the first tab/title appear to overlap.

### lines 876-878

```python
self._criterion.currentTextChanged.connect(
```

The header follows the criterion, so switching from trustworthiness to multi_objective re-labels the column the user is about to read rather than leaving it saying "score".

### lines 891-895

```python
self._adaptive = Toggle("Walk")
```

"Walk", not "Adaptive 2x2". One name for one idea, shared with the Gate Editor's clustering search -- a directional search through hyperparameter space is the same thing in both modules and was called two things. "2x2" was also a description of today's two-parameter special case rather than of the design.

### lines 934-936

```python
self._n_folds_label.setVisible(False)
```

Neither app cross-validates: UMAP fits one embedding per trial and Activation attributes an already-trained model, so a fold count would be a control that does nothing.

### lines 992-994

```python
self._multi_objective_controls = QWidget(self)
```

Multi-objective UMAP controls. These stay visible in the Search tab so the composite is never a hidden formula, but are editable only while that criterion is selected.

### line 1074  _(unsure)_

```python
adaptive_row = QHBoxLayout()
```

adaptive UMAP controls. Blank means the documented API default.

### lines 1146-1147  _(unsure)_

```python
root.removeWidget(self._settings_panel)
```

Match Measure Live: keep the card focused on results and put the complete control set behind a settings button in a separate window.

### lines 1206-1208

```python
self._table.setColumnHidden(self.COLUMNS.index("fold sd"), True)
```

Hidden rather than filled with a placeholder: a column of NA invites the reader to wonder what went wrong, when nothing did -- this app simply has no folds.

### lines 1227-1230

```python
self._preview_stack = QWidget(self)
```

UMAP owns an interactive coordinate viewer. The other search apps still use the generic static figure panel: attribution images and classifier score curves are figures, while a 3-D UMAP is a map the user needs to spin, recolour and cluster.

### lines 1262-1263

```python
self._preview = QLabel("", self._preview_stack)
```

Compatibility attribute for integrations that looked for the old label. It is not shown and never receives a static UMAP.

## HyperparamPanel._on_adaptive_toggled

### lines 1426-1427  _(unsure)_

```python
controls.setEnabled(True)
```

Keep labels and their API dots active even when adaptive search is off; only the value fields are unavailable.

## HyperparamPanel.current_space

### lines 1601-1604

```python
for name, spec in self._walk_axes.items():
```

A Walk needs a starting POINT, so each chosen axis enters the space as exactly one value. Parameters that are not axes keep whatever the fields hold, which is how a fixed metric or a fixed n_components travels with the walk.

## HyperparamPanel.request_gpu_enabled

### line 1672

```python
self._gpu_enabled = False
```

Not ready, so the toggle must not stay down claiming it is.

## HyperparamPanel.run_search

### lines 1831-1832  _(unsure)_

```python
try:
```

Axes from the SPACE, set before the first figure lands, so the grid does not rearrange itself as results arrive.

### lines 1840-1844

```python
walked = list(request.walk_parameters
```

Named for what it does now: a Walk over however many axes the user chose, not the two the first version could search. No axes chosen is the engine's own default pair, not every name in the space -- `metric` sits in the starting centre without being walked.

### lines 1859-1865

```python
worker.finished.connect(self._on_worker_finished)
```

NOT worker.deleteLater. `finished` is emitted from inside the worker thread, so scheduling the object's deletion there hands C++ a second owner for an object Python already owns, and the two race — see the measured account in spacr.qt.bridge.make_thread. The relay below is a bound method, so the connection keeps `self` alive rather than a lambda closure Qt cannot introspect, and the worker is freed when the panel drops its reference on the GUI thread.

## HyperparamPanel.stop_search

### lines 1879-1880  _(unsure)_

```python
from ..button_roles import set_button_busy
```

A stop request is asynchronous: keep the pressed negative button solid red and prevent repeat requests until the worker exits.

## HyperparamPanel._on_worker_finished

### lines 1917-1918

```python
if worker is None or worker is not self._worker:
```

A stale completion must never re-enable controls belonging to a newer run. This is defensive now that starts are serialized at ``finished``.

## HyperparamPanel._on_trial_ready

### lines 1959-1964

```python
if trial.score is not None:
```

Running maximum. A Walk scores every NEIGHBOUR of its current centre, including the ones it then rejects, so the score column in arrival order legitimately goes up and down -- which reads as "the walk is not converging" when it is. Measured on a landscape with a known optimum: arrival order wandered while best-so-far climbed 0.663 -> 0.705 monotonically. This column shows the climb.

## HyperparamPanel._set_row

### lines 2120-2121  _(unsure)_

```python
item = _NumericTableItem(text, float("inf"))
```

Ranked failures have no rank and belong after every successful trial in the default ascending view.

## UmapSearchSettingsDialog

### lines 2487-2489  _(unsure)_

```python
class UmapSearchSettingsDialog(QDialog):
```

Tabbed settings window — mirrors Measure Live's CropSettingsDialog

## UmapSearchSettingsDialog.__init__

### lines 2523-2525

```python
self._search_page = QWidget(self)
```

A group box cannot safely be the tab page itself: its title notch and frame share the tab pane's top-left origin. Put it on an ordinary padded page instead, matching Measure Live's settings dialogs.

### lines 2547-2550

```python
panel._run_btn.hide()
```

The popup has one action row at its foot. The copies embedded in the Search group were left over from when the panel itself was the whole window and produced two Runs and two Propagates plus an unnecessary middle Stop.

### lines 2559-2560  _(unsure)_

```python
close_button.setIcon(QIcon())
```

Some platform themes put a red X on the standard Close button. The semantic red outline/text already communicates the action.

### lines 2582-2589

```python
set_a_sheeted_widgets_own_rule(
```

Only this popup: every settings surface is the theme's black canvas; editable/value fields alone are lifted to dark gray. Do not alter the application palette. NOT `self.setStyleSheet`. A dialog is a window and therefore a sheet root, so a plain assignment here replaces the theme this dialog is carrying instead of adding to it -- measured as `#000000` text on the dark theme for as long as the dialog stays open, and the rule below lost at the next theme change.

### lines 2708-2710

```python
white = QColor(fg)
```

Some platform styles apply disabled/placeholder opacity after QSS. Pin every text palette role to white inside this dialog so labels and field text remain white even when an adaptive control is inactive.

### lines 2744-2745  _(unsure)_

```python
install_api_tooltips(self, panel.app_key, search_tooltips)
```

No API link dots, matching the Live Preview panel and the main settings form -- both of which dropped them for the same reason.

## UmapSearchSettingsDialog._build_umap_tabs

### lines 2803-2806

```python
for key, widget in list(self._module_model._widgets.items()):
```

SettingsWidgets materializes every UMAP category with ``self`` as the initial parent. Remove anything not represented above entirely: merely hiding a compound control can leave a child eligible for a transient paint at (0, 0).

## UmapSearchSettingsDialog.propagate_settings

### lines 2826-2827  _(unsure)_

```python
for key, _label, kind in APP_PARAMS[self._panel.app_key]:
```

One-value search fields are valid module settings. A selected result is more authoritative and therefore wins.

## _complete_metrics_when_opened.show_popup

### lines 2895-2896

```python
self.showPopup = lambda: original(self)
```

Once, whatever happened: a list that cannot be completed must not try again on every click.

## WalkAxesDialog._style_surfaces

### lines 3057-3060

```python
set_a_sheeted_widgets_own_rule(self, f"""
```

See `UmapSearchSettingsDialog` above: a plain assignment on a window replaces the sheet it carries. This docstring's own "after the application stylesheet has been composed" is exactly the moment that matters.

## WalkAxesDialog._update_cost

### lines 3108-3109

```python
self._cost.setText(
```

A half-typed starting value. Say nothing rather than a number that is wrong; the OK button validates properly.

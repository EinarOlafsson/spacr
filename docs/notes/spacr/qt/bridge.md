# Notes from `spacr/qt/bridge.py`

Prose lifted out of `spacr/qt/bridge.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (4 entries)
- [_StreamRedirector.__init__](#_streamredirector__init__) (1 entry)
- [_StreamRedirector.write](#_streamredirectorwrite) (1 entry)
- [checkpoint](#checkpoint) (1 entry)
- [RunRegistry.unregister](#runregistryunregister) (1 entry)
- [wait_for_parked_threads](#wait_for_parked_threads) (1 entry)
- [prune_parked_threads](#prune_parked_threads) (1 entry)
- [drain_thread](#drain_thread) (2 entries)
- [PipelineWorker](#pipelineworker) (1 entry)
- [PipelineWorker.run](#pipelineworkerrun) (12 entries)
- [PipelineWorker.run._mark_emitted](#pipelineworkerrun_mark_emitted) (1 entry)
- [PipelineWorker.run._capture_show](#pipelineworkerrun_capture_show) (3 entries)
- [PipelineWorker.run._publish_figure](#pipelineworkerrun_publish_figure) (2 entries)
- [_tag](#_tag) (1 entry)
- [_say_what_is_wrong_with_the_settings.run](#_say_what_is_wrong_with_the_settingsrun) (1 entry)
- [resolve_pipeline_entry](#resolve_pipeline_entry) (6 entries)
- [make_thread](#make_thread) (5 entries)

## Module level

### lines 39-41

```python
from ..figures.style import figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

### lines 466-468  _(unsure)_

```python
_PROGRESS_RE = re.compile(r"\bProgress:\s*(\d+)\s*/\s*(\d+)")
```

Which jobs are running right now

### lines 476-481

```python
WORKER_SETTING_KEYS = (
```

Settings that directly control process/thread pools in shipped pipelines. Each is capped to the budget remaining after older active runs. A run with N workers consumes N-1 slots because its own QThread is already one of the concurrently executing units; this gives the requested sequence: second = total - first + 1, then each later run subtracts the extra workers reserved by every older run.

### lines 807-809  _(unsure)_

```python
_PARKED_THREADS: List[tuple] = []
```

Draining a QThread without ever terminating it

## _StreamRedirector.__init__

### lines 77-80

```python
self._lock = threading.Lock()
```

The worker thread writes via print(); the idle-flush pump thread calls idle_flush() concurrently. Both mutate _buf, so every access is guarded — an unlocked race corrupted the buffer and could crash the interpreter.

## _StreamRedirector.write

### line 107

```python
for chunk in emits:
```

Emit OUTSIDE the lock so a slow slot can't block the writer.

## checkpoint

### lines 444-445

```python
cancellation_checkpoint()
```

Cancellation is checked before waiting on Pause. request_cancel() also releases the gate, so a paused worker cannot be stranded during shutdown.

## RunRegistry.unregister

### lines 720-722

```python
handle.setParent(None)
```

Hand ownership back to Python. Left parented, the handle

(and the worker it references) would live until the registry did, i.e. until the process exited.

## wait_for_parked_threads

### line 846  _(unsure)_

```python
continue
```

Wrapper already gone, which only happens after it finished.

## prune_parked_threads

### lines 883-884  _(unsure)_

```python
pass
```

The C++ QThread is already gone, which can only happen after it finished — nothing left to hold on to.

## drain_thread

### line 1003

```python
return True
```

Internal C++ object already deleted — it cannot still be running.

### lines 1006-1007

```python
thread.quit()
```

quit() is documented thread-safe and posts to the thread's OWN event loop, unlike a queued connection to this GUI-affine object.

## PipelineWorker

### line 1058, trailing  _(unsure)_

```python
figure_ready = Signal(object, str)
```

(figure, prerendered_png_path or "")

## PipelineWorker.run

### lines 1146-1149

```python
if self.cancel_token.cancelled:
```

Stop can be clicked in the event-loop tick immediately after Run, before this slot begins. Do not import matplotlib, hash inputs, or open a journal for work that never started; acknowledge it and let the DirectConnection to QThread.quit retire the thread immediately.

### lines 1172-1180

```python
capture_show = None
```

NOTE: there used to be a background "idle-flush pump" daemon thread here that emitted line_ready periodically. It was removed — emitting a Qt signal from a non-Qt-affinity Python thread delivered as a DirectConnection and mutated console widgets off the GUI thread, aborting the process. The real cause of "pressed Run, nothing happens" was a garbage-collected worker (see AppScreen._on_run keeping self._worker), not missing flushes; the redirector's 1024-char chunk-cap already surfaces long newline-less bursts.

### lines 1182-1183

```python
capture_show = None
```

Intercept matplotlib show() so figures land in the UI instead of a blocking Tk window. `plt.show` gets restored in `finally`.

### lines 1189-1196

```python
if matplotlib.get_backend().lower() != "agg":
```

force=True, and the difference is the whole bug: force=False is a NO-OP once a backend is active, and by the time a run starts `qtagg` is. Every plt.figure() on this worker then carried a FigureCanvasQTAgg owned by the worker thread. `app.launch` sets Agg before any figure exists, which is the real fix; this stays as the guard for the paths that do not come through launch the CLI, a test, a script -- and is a no-op when it is already Agg.

### lines 1219-1223

```python
preexisting_figures = tuple(_registered_figures())
```

pyplot's registry is process-global.  A figure that was already open before this worker began belongs to its caller (or to a different screen), not to this run.  Hold the objects, rather than only their figure numbers: pyplot reuses a closed number and the replacement must still be eligible for this run.

### lines 1364-1368

```python
if self._settings.get("hash_inputs", False):
```

Only announce the hashing when it is actually going to happen. Hashing is the slow part of opening a run, which is why the line exists at all; printing it with hashing off names a pause that is not there and claims a record that was not made.

### lines 1394-1398

```python
with installed_token(self.cancel_token), responsive_gui():
```

INSTRUCTION 126. A pure-Python worker starves the GUI thread of the interpreter lock and the backdrop drops to 42 ms a frame, which is the reported lag; asking for the lock more often brings it back to 17.7. Scoped to the run and restored after, so a headless `spacr-run` in the same interpreter pays nothing.

### lines 1403-1405

```python
if payload is not None:
```

Hand the answer back rather than making the screen find it. Emitted before `finished` so a listener has the data by the time the run is announced as over.

### lines 1421-1424

```python
ok = exc.code in (None, 0)
```

``sys.exit()`` and ``sys.exit(0)`` are successful early exits. Any other code is a failure and must reach the same error path as an exception; treating ``sys.exit(1)`` as success made CLI-style pipeline failures appear as a green, completed GUI run.

### lines 1441-1443

```python
tb = traceback.format_exc()
```

KeyboardInterrupt and cancellation-style BaseExceptions must leave a failed, inspectable manifest instead of a forever "running" record.

### lines 1469-1471

```python
try:
```

THE SINK COMES DOWN WITH THE RUN. A finished run is not still publishing, and a sink left installed holds `worker` alive and emits into a dead signal on the next run.

### lines 1482-1484

```python
self.gate.resume()
```

Release the gate before announcing completion: a worker left paused would otherwise strand anything that later waits on it, and the job is over either way.

## PipelineWorker.run._mark_emitted

### lines 1247-1248  _(unsure)_

```python
pass
```

A figure that refuses an attribute still gets its tile; it only loses the cross-route half of the guard.

## PipelineWorker.run._capture_show

### lines 1252-1259

```python
"""Emit each new figure once, rendering it HERE on the worker thread.
```

Emit ordinary figures only once. Figures explicitly marked

``_spacr_live_update`` are re-rendered and emitted in place; this is how the training monitor refreshes without filling the gallery with one snapshot per epoch. Render each figure to a PNG HERE, in the worker thread (Agg savefig touches no Qt) — the expensive part — so the GUI thread only does a cheap file-move + pixmap load and never hangs while figures stream in.

### lines 1271-1273

```python
if id(fig) in preexisting_ids:
```

Holding the baseline objects for the run prevents their

IDs from being reused, so this lookup stays both exact and constant-time even after a long interactive session.

### lines 1276-1279

```python
with figure_style(theme_target()):
```

The creation site owns the artists' house style; a context opened here cannot retroactively restyle them. Keep render-time Matplotlib work scoped to the same target without changing process-wide rcParams.

## PipelineWorker.run._publish_figure

### lines 1304-1318

```python
def _publish_figure(fig, path=""):
```

AND A SINK FOR FIGURES PYPLOT NEVER SEES.

`_capture_show` walks `plt.get_fignums()`, so it can only ever emit figures that are IN pyplot's registry and only when somebody calls `show`. A module that builds a bare `matplotlib.figure.Figure` and writes it with savefig -- which is the correct thing for a library to do, and what `spacr.regression_qc` does for its whole ~19-panel report satisfies neither condition. Every one of those panels was on disk and none of them was in the application, reported 2026-08-18 as "several graphs are saved but I cannot see them".

Same rendering path as above: the PNG is written HERE, on the worker thread, so the GUI thread only moves a file and loads a pixmap.

### lines 1320-1324

```python
"""Publish one figure, unless it has already been shown.
```

A picture that has already been shown is not a second picture because it was also saved. `save the sheet, then plot_plates(verbose=True) shows it` is a real sequence in `generate_ml_scores`, and without this the gallery held two tiles for one file.

## _tag

### lines 1494-1496  _(unsure)_

```python
def _tag(app_key: str, fn: Optional[Callable]) -> Optional[Callable]:
```

Dispatch: app_key -> function to run

## _say_what_is_wrong_with_the_settings.run

### lines 1551-1557

```python
settings = coerce_expected_types(settings, app_key)
```

RESTORE THE DECLARED TYPES FIRST, and hand the pipeline the restored dict. A number typed into a GUI field arrives as text, so `cell_diameter='60.0'` was reported as an error the user had to fix by hand -- for a well-formed value -- and then crashed the run inside Cellpose on `diameter > 0`. Converting once, here, fixes both, and does it for every setting rather than for the ones that have already bitten.

## resolve_pipeline_entry

### lines 1612-1619

```python
from spacr.illumination import prepare_illumination_correction
```

NAMED HERE BECAUSE THE ROW IS GONE. Its entry point used to arrive through the registry's APP_META, which unregistering pops -- so folding the module into Measure's settings took `spacr-run illumination` and the Run button with it. The correction is applied before any intensity feature is computed, which is why the settings belong on the measure run; estimating and inspecting the field on its own is a separate act and still has to work.

### lines 1623-1625

```python
from spacr.classify import classify
```

One entry point over both families. It calls deep_spacr or generate_ml_scores unchanged, so a run here and a run through either original module are the same run.

### lines 1629-1635

```python
from spacr.deep_spacr import deep_spacr
```

deep_spacr, not train_test_model. The Classify screen builds its panel from deep_spacr_defaults, so it SHOWS generate_training_ dataset, apply_model_to_dataset, n_top_examples and tar_path every one of which train_test_model ignores. Running the training stage alone meant those switches were settable and silently did nothing. Tk (gui_utils.run_function_gui) and validate.APP_FUNCTIONS both map classify -> deep_spacr; Qt was the odd one out.

### lines 1669-1671

```python
from spacr.ops_engine import run_ops
```

Imported HERE and not at module scope: this function is called while a screen is being built.

### lines 1686-1695

```python
if app_key == "barcode_qc":
```

THE FOLDED MODULES, whose row used to carry their entry point.

Each of these three declared `entry=` on a `register_app` call. Folding the module into a host screen deletes that row, and the registered-entry seam below is the only other place the string lived -- so without these branches the Run button on a folded module's page resolves to None and does nothing, while `spacr-run barcode_qc` goes on working, which is the worst of both. The pipeline functions are unchanged and are the same ones the CLI runs; only where the Run button finds them moves.

### lines 1705-1710

```python
from .app import registered_entry
```

Apps that registered their own entry point. The chain above is the built-in table; this is the seam a module registered through `spacr.qt.app.register_app(..., entry="mod:func")` reaches, so a new pipeline app is one registration call rather than a branch here plus seven other files. Consulted before plugins because a built-in registration is not a contribution.

## make_thread

### lines 1811-1828

```python
if capture_figures and "matplotlib.pyplot" not in sys.modules:
```

THE FIRST pyplot IMPORT HAPPENS ON THIS THREAD, NOT ON THE WORKER.

`PipelineWorker.run` imports matplotlib.pyplot, so on the first job a module that large is imported from the worker while the GUI thread may be collecting garbage -- and that combination segfaults the process. A segfault is not one failed test: it takes the whole pytest shard, and every coverage measurement with it, which is how it was found.

`make_thread` runs on the caller's thread, so importing here puts that first import where it is safe. Temporarily suspend cyclic GC as well: pyplot's many allocations can otherwise start a collection in the middle of the import. If that collection releases an old PySide wrapper, Qt re-enters Python while matplotlib's module graph is only half initialized; settings-search followed by the UMAP dialog used to reproduce that native crash reliably. Every job after the first is a dict lookup, and the worker's own `matplotlib.use("Agg", force=True)` still wins -- importing pyplot does not choose a backend.

### lines 1836-1837

```python
LOG.debug("matplotlib.pyplot could not be pre-imported",
```

A build with no matplotlib still starts jobs; the worker's own import is what would fail, and it already handles that.

### lines 1850-1851

```python
worker.user_visible = bool(user_visible)
```

Set on the worker rather than passed to its constructor, so a PipelineWorker built anywhere else keeps the visible default.

### lines 1855-1857

```python
worker.finished.connect(thread.quit, Qt.DirectConnection)
```

quit() is explicitly thread-safe. A DirectConnection matters during shutdown: if the GUI thread is blocked in QThread.wait(), a queued call to the GUI-affine QThread object can never run and the join deadlocks.

### lines 1864-1874

```python
thread.finished.connect(handle.retire)
```

NOTE the absence of `handle.deleteLater` here. The handle is parented to the registry, so C++ already owns it; adding a deferred delete on top is the second owner, which is precisely the mistake documented above for the worker. `RunRegistry.unregister` reparents it to nothing instead, and Python frees it when the last reference goes — on the thread that holds it.

The slot is a bound method of a GUI-thread QObject, not a closure: `thread.finished` is delivered across a thread boundary, and a closure would both capture the handle and run with the emitting thread's affinity.

# The map

Where things live and why. Counts are in `FACTS.md`, generated — this file
carries the shape, which changes far more slowly.

## Two halves

**The pipeline** (`spacr/*.py`) is plain Python and has no Qt. It can be
driven from a script, a notebook or the CLI, and the GUI is one caller
among several. Keep it that way: a `from PySide6 import ...` in a pipeline
module makes the package unimportable on a cluster.

**The GUI** (`spacr/qt/`) is PySide6 and calls into the pipeline. It never
reimplements analysis. Where a preview appears to duplicate pipeline
logic, it is a bug — the Mask live preview once *subtracted* a background
where the pipeline *thresholds* it, and the preview was lying about what a
run would produce.

## The pipeline

| Module | What it owns |
|---|---|
| `core.py`, `io.py` | The pipeline entry points and image ingest/normalisation |
| `measure.py`, `object.py` | Per-object features; the object table |
| `deep_spacr.py`, `ml.py` | Torch classifiers; classical ML |
| `sequencing.py` | Reads → barcodes → gRNA counts |
| `timelapse.py` | Tracking and track-based relabelling |
| `settings.py` | `get_*_settings()` — the defaults every module starts from |
| `settings_spec.py` | Types for the GUI. Imports **nothing** — deliberately: reaching it through `gui_utils` cost 770 ms of Tk dependencies on the GUI thread |
| `run_journal.py` | Per-run manifest: version, git hash, settings, input hashes, timings |
| `schema.py` | Table and key definitions, incl. `OBJECT_KEY_COLUMNS` |
| `updater.py` | Version check and the upgrade command (see INVARIANTS §12) |

## The GUI

```
spacr/qt/
  app.py              MainWindow, the APPS registry, navigation
  theme.py            palettes, the stylesheet, WIDGET_QSS_MODULES
  bridge.py           RunHandle, RunRegistry, PipelineWorker, threads
  preferences.py      the preferences store AND the dialog
  shutdown.py         graceful/force quit, the 5-minute re-prompt
  settings_search.py  the Ctrl+F strip; owns SettingsSearchPane
  screens/            one module per screen
    app_screen.py     the generic screen every module gets
    settings_model.py settings → widgets → sections
  widgets/            reusable pieces and the live previews
```

### How a module screen exists

1. A row in `APPS` in `app.py`, or a `register()` called through
   `SELF_REGISTERING_MODULES` / `_SELF_REGISTERING_APPS`.
2. `resolve_default_settings(app_key)` in `settings_model.py` produces its
   settings dict.
3. `_APP_CATEGORY_SPECS` lays those keys out in named groups. A key no
   layout claims lands in "Additional Settings" — the bucket the layouts
   exist to keep empty.
4. `AppScreen` builds the form, the console, the preview card and the run
   controls.

A module with no dedicated screen still gets all of this. That is the
point of `AppScreen`.

### Registration seams

`register_app`, `register_defaults`, `register_widget_qss`,
`SELF_REGISTERING_MODULES`, `_SELF_REGISTERING_APPS`. They exist so a new
screen does not require editing five tables — but **each is a place a new
module can be half-registered**, which is exactly how a screen ends up
built, tested and unreachable. Four features spent weeks in that state.

If you add a screen, check all of: it appears in `APPS`; its QSS module is
in `WIDGET_QSS_MODULES` (INVARIANTS §1); it is filed in a Home band
(`CATS_STAGE5`) rather than falling into the fallback; it has an icon.

### Running work

Everything long goes through `bridge.py`. `RunHandle` is one in-flight
job — pause gate, progress, elapsed, `request_cancel`. `RunRegistry`
owns them and `cancel_all` is what `closeEvent` uses. `user_visible=False`
marks housekeeping so Home's banner ignores it.

### The live previews

`widgets/live_preview.py` (Mask) is the newest and is the pattern to
follow. `measure_preview`, `motility_preview`, `timelapse_preview` and
`timelapse_movie` are siblings.

A preview must produce **the image a real run produces**. Where it does
its own preprocessing, that code has to match the pipeline's — cite the
pipeline function in a comment so the next person can check.

## Tests

```
tests/            pipeline tests, no Qt
tests/qt/         GUI tests; need QT_QPA_PLATFORM=offscreen
tests/qt/conftest.py   the isolation fixtures — read before adding one
```

Markers split CI into six suites. `tests/qt/` is by far the largest and
slowest.

# Notes from `spacr/qt/widgets/motility_preview.py`

Prose lifted out of `spacr/qt/widgets/motility_preview.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Calibration](#calibration) (1 entry)
- [resolve_merged_dir](#resolve_merged_dir) (1 entry)
- [Module level](#module-level) (1 entry)
- [smooth_and_filter_tracks](#smooth_and_filter_tracks) (1 entry)
- [render_motility_figure](#render_motility_figure) (4 entries)
- [MotilityRequest](#motilityrequest) (1 entry)
- [_MotilityWorker](#_motilityworker) (1 entry)
- [MotilityPreviewPanel](#motilitypreviewpanel) (1 entry)
- [MotilityPreviewPanel.__init__](#motilitypreviewpanel__init__) (7 entries)
- [MotilityPreviewPanel._build_ui](#motilitypreviewpanel_build_ui) (6 entries)
- [MotilityPreviewPanel._install_plate](#motilitypreviewpanel_install_plate) (1 entry)
- [MotilityPreviewPanel.run_preview](#motilitypreviewpanelrun_preview) (3 entries)
- [MotilityPreviewPanel._on_worker_done](#motilitypreviewpanel_on_worker_done) (1 entry)
- [MotilityPreviewPanel._on_metric_changed](#motilitypreviewpanel_on_metric_changed) (1 entry)
- [MotilityPreviewPanel._selected_group_key](#motilitypreviewpanel_selected_group_key) (1 entry)
- [MotilityPreviewPanel._refresh_plane_layout](#motilitypreviewpanel_refresh_plane_layout) (1 entry)
- [MotilityPreviewPanel.closeEvent](#motilitypreviewpanelcloseevent) (2 entries)

## Calibration

### lines 77-79

```python
@dataclass(frozen=True)
```

Units — stated, never assumed

## resolve_merged_dir

### lines 130-132  _(unsure)_

```python
def resolve_merged_dir(path) -> str:
```

Reading merged arrays — lazily, and only a few frames of them

## Module level

### lines 253-255  _(unsure)_

```python
TRACK_KEYS = ["plateID", "wellID", "fieldID", "cellID"]
```

Metrics — cheap, recomputed live from the cached point table

## smooth_and_filter_tracks

### lines 281-282  _(unsure)_

```python
x = g["x"].to_numpy(dtype=float, copy=True)
```

Interpolation edits these arrays; pandas 3 may expose the group columns as read-only views.

## render_motility_figure

### lines 437-439  _(unsure)_

```python
def render_motility_figure(points, tracks, calibration: Calibration,
```

Plot — matplotlib Agg into an RGB array (no Qt backend needed)

### line 473  _(unsure)_

```python
ax_tracks.set_title("Tracks (from origin)", fontsize=8)
```

1 — origin-centred tracks

### line 496  _(unsure)_

```python
ax_len.set_title("Track length", fontsize=8)
```

2 — track-length distribution with the cutoff marked

### line 510  _(unsure)_

```python
ax_vel.set_title("Velocity by infection state", fontsize=8)
```

3 — velocity + straightness, split by infection, unit stated

## MotilityRequest

### lines 545-547

```python
@dataclass
```

Worker — the expensive half only

## _MotilityWorker

### line 574, trailing  _(unsure)_

```python
finished_result = Signal(object, str)
```

(DataFrame or None, error)

## MotilityPreviewPanel

### line 671, trailing

```python
preview_ready = Signal(object)
```

MotilitySummary, or None on failure

## MotilityPreviewPanel.__init__

### lines 691-695

```python
self._jobs = JobRunner(self, threaded=threaded,
```

Scanning a plate lists every file in `merged/` and parses each name: thousands of entries on a 384-well plate, and not GUI-thread work. `threaded=False` runs each job inline, emitting the same signals in the same order, so a test can drive this panel synchronously without the behaviour diverging.

### lines 698-704

```python
self._plane_jobs = JobRunner(self, threaded=threaded,
```

A SECOND runner, and the separation is load-bearing twice over. `_loads_in_flight` reports `self._jobs`' pending work as "a plate is still being scanned", and a plane-count read is not a plate scan; and `user_visible=False` keeps `spacr.qt.widgets.home` -- which filters run banners on exactly that flag -- from flashing "motility preview - running" for a read the user never started. Nothing user-started is ever submitted here.

### line 725, trailing

```python
self._points = None
```

cached — the expensive half

### lines 731-732

```python
self._retired_worker: Optional[_MotilityWorker] = None
```

A worker whose result has landed but whose QThread may still be unwinding. Held until ``finished`` so it is never collected mid-run.

### lines 734-735  _(unsure)_

```python
self._run_token = 0
```

Bumped whenever the pass in flight is superseded (a new plate, an explicit cancel); a stale result is dropped. See LivePreviewContract.

### lines 739-740

```python
self._sampler = ImageSetSampler(DEFAULT_MAX_SETS)
```

Bounded, reproducible sample of the plate's time series — the dropdown never lists them all. See ImageSetSampler.

### lines 744-746

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## MotilityPreviewPanel._build_ui

### lines 758-759  _(unsure)_

```python
pick = QHBoxLayout()
```

FOV and channel dropdowns sit immediately LEFT of the Choose control; all three wear the flat "Live toggle" look.

### lines 776-777  _(unsure)_

```python
self._group_box = self._fov_box
```

Kept under its historical name for the integrations and tests that already drive it.

### line 795  _(unsure)_

```python
self._tracked_object = QComboBox(self)
```

array layout (changing one re-reads the merged arrays)

### line 825  _(unsure)_

```python
self._min_len = QSpinBox(self)
```

metrics (live — recomputed from the cached point table)

### lines 905-906  _(unsure)_

```python
self._tracked_plane.valueChanged.connect(
```

One plane, two surfaces: the settings spinner and the flat dropdown in the pick row stay in step.

### line 913  _(unsure)_

```python
self._cancel_btn = QPushButton(PREVIEW_CANCEL_TEXT, self)
```

Same control, same place, same words as the Mask live preview.

## MotilityPreviewPanel._install_plate

### line 1121

```python
self._refresh_plane_layout()
```

Off the GUI thread: reading the plane count opens a merged array.

## MotilityPreviewPanel.run_preview

### lines 1300-1301  _(unsure)_

```python
worker.preview_token_value = self.preview_token()
```

The generation this pass belongs to. Cancelling bumps the panel's token, and a result whose token no longer matches is dropped.

### lines 1304-1305

```python
worker.finished_result.connect(self._on_worker_done)
```

Bound method, not a closure — a plain callable would be invoked on the worker thread and every widget touch below would be off-thread.

### lines 1307-1311

```python
worker.finished.connect(self._on_worker_finished)
```

NOT worker.deleteLater — that hands Qt a second owner for an object Python already holds, and the two race (the measured account is in spacr.qt.bridge.make_thread). The Mask preview was fixed away from this pattern; keeping it here left a running QThread owned by nobody when the user closed the screen mid-read.

## MotilityPreviewPanel._on_worker_done

### lines 1353-1355

```python
if self._worker is not None:
```

The pass is over as far as the panel is concerned, so a new one may start; the reference is kept until ``QThread.finished`` because a QThread collected while it is still unwinding aborts the process.

## MotilityPreviewPanel._on_metric_changed

### line 1375

```python
def _on_metric_changed(self, *_):
```

metrics (GUI thread — cheap)

## MotilityPreviewPanel._selected_group_key

### line 1459

```python
def _selected_group_key(self):
```

plane layout (read off the GUI thread, applied on it)

## MotilityPreviewPanel._refresh_plane_layout

### lines 1499-1500  _(unsure)_

```python
self._apply_plane_layout(0)
```

No plate, or a group with no files. The old code arrived at the same answer through `except StopIteration` -- 0 planes.

## MotilityPreviewPanel.closeEvent

### lines 1677-1678

```python
self.shutdown()
```

Cancel the scan before waiting on the motility worker: leaving the screen mid-scan must not leave a QThread behind either.

### lines 1680-1681

```python
for worker in (self._worker, getattr(self, "_retired_worker", None)):
```

Both of them: the pass in flight, and the one whose result has landed while its thread was still unwinding.

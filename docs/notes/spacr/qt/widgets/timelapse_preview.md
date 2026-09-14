# Notes from `spacr/qt/widgets/timelapse_preview.py`

Prose lifted out of `spacr/qt/widgets/timelapse_preview.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [FrameSequence.open](#framesequenceopen) (1 entry)
- [segment_frame](#segment_frame) (2 entries)
- [backend_available](#backend_available) (2 entries)
- [_tracks_from_features](#_tracks_from_features) (1 entry)
- [track_colour](#track_colour) (1 entry)
- [render_frame](#render_frame) (1 entry)
- [_TimelapseWorker](#_timelapseworker) (1 entry)
- [open_sequence_payload](#open_sequence_payload) (2 entries)
- [TimelapsePreviewPanel](#timelapsepreviewpanel) (1 entry)
- [TimelapsePreviewPanel.__init__](#timelapsepreviewpanel__init__) (8 entries)
- [TimelapsePreviewPanel._build_ui](#timelapsepreviewpanel_build_ui) (8 entries)
- [TimelapsePreviewPanel._dropped_path](#timelapsepreviewpanel_dropped_path) (1 entry)
- [TimelapsePreviewPanel._on_job_failed](#timelapsepreviewpanel_on_job_failed) (1 entry)
- [TimelapsePreviewPanel._on_sequence_loaded](#timelapsepreviewpanel_on_sequence_loaded) (1 entry)
- [TimelapsePreviewPanel.load_sequence](#timelapsepreviewpanelload_sequence) (1 entry)
- [TimelapsePreviewPanel._frame_channel_count](#timelapsepreviewpanel_frame_channel_count) (2 entries)
- [TimelapsePreviewPanel._on_fov_changed](#timelapsepreviewpanel_on_fov_changed) (1 entry)
- [TimelapsePreviewPanel.load_masks](#timelapsepreviewpanelload_masks) (1 entry)
- [TimelapsePreviewPanel._start](#timelapsepreviewpanel_start) (5 entries)
- [TimelapsePreviewPanel._on_worker_done](#timelapsepreviewpanel_on_worker_done) (1 entry)
- [TimelapsePreviewPanel._movie_source_paths](#timelapsepreviewpanel_movie_source_paths) (1 entry)
- [TimelapsePreviewPanel._refresh_movie_targets](#timelapsepreviewpanel_refresh_movie_targets) (1 entry)
- [TimelapsePreviewPanel._push_to_movie](#timelapsepreviewpanel_push_to_movie) (1 entry)
- [TimelapsePreviewPanel._pick_masks](#timelapsepreviewpanel_pick_masks) (1 entry)
- [TimelapsePreviewPanel.closeEvent](#timelapsepreviewpanelcloseevent) (2 entries)
- [build_timelapse_preview_card](#build_timelapse_preview_card) (1 entry)

## Module level

### lines 78-80

```python
from .live_preview import (
```

Reuse the Mask live preview's rendering + canvas primitives wholesale so the two panels behave identically from the user's side: the same zoom/pan pair, the same percentile stretch, the same boundary drawing.

### lines 114-116

```python
_LIVE_FRAME_SEQUENCES: "weakref.WeakSet[FrameSequence]" = weakref.WeakSet()
```

These weak sets let the process-wide memory policy discover already-loaded preview caches without importing this comparatively heavy module and without keeping a closed preview alive.

## FrameSequence.open

### lines 264-268

```python
if n_pages > 1:
```

A time series can be one page per frame (page-addressable, the cheapest read) or a single page holding a 3-D array, which is what tifffile writes for a plain (T, H, W) save. The second form is not page-addressable, so it is memory-mapped instead — still lazy, just at the OS page level rather than the TIFF page level.

## segment_frame

### lines 420-422

```python
def segment_frame(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
```

Segmentation (expensive — cached by the panel)

### lines 433-435

```python
model = preview_cellpose_model(str(params.get("model", "cpsam")))
```

ONE constructor for every live view — see

`preview_contract.preview_cellpose_model` for why `model_type=` may never appear here. The Mask preview calls the same helper.

## backend_available

### lines 478-480  _(unsure)_

```python
def backend_available(mode: str) -> Tuple[bool, str]:
```

Linking (cheap — re-run on every tracking-setting change)

### line 497, trailing  _(unsure)_

```python
if pkg is None:
```

iou is pure numpy + scipy, always available

## _tracks_from_features

### lines 512-517

```python
return tracks_df.merge(features[cols], on=["frame", "original_label"],
```

many_to_one: ``features`` comes from regionprops, so it holds exactly one row per (frame, label); the track table may name one label twice in a frame when two tracks claim it at a merge/split event, which is why the left side is not constrained. A duplicated label on the features side would invent extra track rows with fabricated centroids. Same contract as the identical join in timelapse._track_by_iou's caller.

## track_colour

### lines 803-805  _(unsure)_

```python
def track_colour(track_id: int) -> Tuple[int, int, int]:
```

Overlay rendering (pure numpy — unit-testable without a display)

## render_frame

### lines 869-874

```python
_needed = {"x", "y", "frame", "track_id"}
```

Every column the block below actually touches, not just the two it used to name. `frame` and `track_id` are indexed and grouped by three lines down, so a tracks frame carrying x/y under a different id column -- `particle`, which is what trackpy returns before `_link_trackpy` renames it -- raised a KeyError from inside a renderer rather than being skipped like any other unusable input.

## _TimelapseWorker

### line 1056, trailing  _(unsure)_

```python
finished_result = Signal(object, str)
```

(result dict or None, error)

## open_sequence_payload

### lines 1148-1151

```python
out["siblings"] = sibling_sources(
```

`seq.kind` already records the layout: `open` builds "files" from a directory listing and every other kind from a single file. Reading it back is free, where `target.is_dir()` is one more stat on a path that has just been opened.

### lines 1157-1164

```python
out["siblings"] = [target]
```

AND THE FIELD ITSELF IS STILL AN ANSWER. `siblings=None` means "nobody listed", which sends `_refresh_source_selectors` off to list the folder ITSELF -- on the GUI thread, on the very path whose listing has just failed here. If that failure was a sleeping /nas_mnt share, that retry is the twenty-second freeze. One entry is the same thing `sibling_sources` returns when it cannot read the parent, and it keeps the FOV dropdown honest: it lists what is known to be there.

## TimelapsePreviewPanel

### line 1188, trailing

```python
preview_ready = Signal(object)
```

TrackStats, or None on failure

## TimelapsePreviewPanel.__init__

### lines 1199-1203

```python
self._jobs = JobRunner(self, threaded=threaded,
```

Opening a sequence reads a TIFF header or memory-maps a stack, and then lists every sibling field of view. On a plate that is not GUI -thread work. `threaded=False` runs each job inline, emitting the same signals in the same order, so a test can drive this panel synchronously without the behaviour diverging.

### lines 1206-1209

```python
self._movie_jobs = JobRunner(
```

Additional fields are deliberately serialized through their own runner.  One Cellpose field at a time keeps the GUI responsive without multiplying model/GPU memory by the Fields setting, and a cap change can cancel this queue without disturbing a source open.

### lines 1212-1214

```python
self._jobs.job_failed.connect(self._on_job_failed)
```

A worker that raises never reaches its `on_done`, so a "Opening …" placeholder written before `submit()` would stay on screen for the life of the panel. This is the other half of `_set_transient_status`.

### lines 1243-1244

```python
self._retired_worker: Optional[_TimelapseWorker] = None
```

A worker whose result has landed but whose QThread may still be unwinding. Held until ``finished`` so it is never collected mid-run.

### lines 1246-1248

```python
self._run_token = 0
```

Bumped whenever the pass in flight is superseded (a new sequence, an explicit cancel). A worker's result is only adopted while the token it carries still matches — see LivePreviewContract.

### lines 1255-1256  _(unsure)_

```python
self._loading_fov = False
```

Guards the FOV dropdown against re-entering itself while the sequence it just asked for is being opened.

### lines 1258-1259

```python
self._sampler = ImageSetSampler(DEFAULT_MAX_SETS)
```

Bounded, reproducible sample of the folder's sequences — the dropdown never lists a whole plate. See ImageSetSampler.

### lines 1269-1271

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## TimelapsePreviewPanel._build_ui

### lines 1283-1284  _(unsure)_

```python
pick = QHBoxLayout()
```

FOV and channel dropdowns sit immediately LEFT of the Choose control; all of them wear the flat "Live toggle" look.

### lines 1322-1324

```python
self._model_box = QComboBox(self)
```

segmentation settings (changing one invalidates the mask cache) Read from the Cellpose API — see `spacr.settings.cellpose_model_menu`.

### line 1365  _(unsure)_

```python
self._mode_box = QComboBox(self)
```

tracking settings (changing one re-links only)

### line 1436

```python
for w in (self._displacement, self._memory, self._iou):
```

Re-link (never re-segment) whenever a *linking* knob moves.

### lines 1441-1442  _(unsure)_

```python
self._min_len.valueChanged.connect(self._on_scoring_changed)
```

The minimum length only decides what counts as a fragment, so it re-scores the existing tracks — no linking, no segmentation.

### line 1444

```python
self._tail.valueChanged.connect(lambda *_: self._refresh_canvases())
```

Pure display knobs never touch masks or tracks.

### lines 1447-1449

```python
self._channel.valueChanged.connect(self._sync_channel_combo_from_spin)
```

One channel, two surfaces: the settings spinner and the flat dropdown in the pick row are kept in step so the frame the user looks at is always the frame Cellpose is handed.

### line 1455  _(unsure)_

```python
self._cancel_btn = QPushButton(PREVIEW_CANCEL_TEXT, self)
```

Same control, same place, same words as the Mask live preview.

## TimelapsePreviewPanel._dropped_path

### lines 1539-1569

```python
if (p.suffix.lower() in FRAME_SUFFIXES
```

The suffix is a pure-string test, so it is free; ask the filesystem only when it does not already decide. And ask it through the cache, never with `p.is_dir()`: this runs from dragEnterEvent/dragMoveEvent/dropEvent on the GUI thread, on a path the user dragged in, and dragMoveEvent fires on every mouse-move. Measured 2026-09-04, a stat under /nas_mnt (autofs, share asleep) had not returned after twenty seconds -- one hover over the panel with a network folder held would freeze the whole window with no traceback.

THE DEFAULT IS THE NAME, because the accept/reject answer is owed NOW and a probe queued this instant cannot have finished. `path_probe.isdir` returns the cached answer once there is one and this guess until then:

no extension  -> almost certainly a folder -> accept. This is the field of view on the plate share, and accepting it wrongly only costs a "Load failed" sentence in `self._status`, because the open happens on the JobRunner worker inside `load_sequence_async`. some other extension -> a file this panel cannot read -> refuse, exactly as the old `p.is_dir()` did for `notes.txt`. Refusing on the name alone is what keeps the "not allowed" drag cursor honest instead of accepting every document and reporting the mistake afterwards.

A folder that really does have a dot in its name is refused for the first hover only: asking queues the probe, and `dragMoveEvent` fires again on the next mouse-move, by which time the cache has the real answer. The drag itself is the retry, so there is no signal to subscribe to here.

## TimelapsePreviewPanel._on_job_failed

### lines 1654-1655  _(unsure)_

```python
self._transient_status = None
```

The label's C++ half went with the panel while the worker was still unwinding. Nothing to tell anyone.

## TimelapsePreviewPanel._on_sequence_loaded

### lines 1691-1692

```python
self._sampler.enumerate_paths(
```

Adopt before installing, so `_refresh_source_selectors` finds the listing cached rather than walking the plate again.

## TimelapsePreviewPanel.load_sequence

### lines 1712-1713

```python
self._load_token += 1
```

This install is authoritative, so anything already on its way is superseded here rather than allowed to land on top of it later.

## TimelapsePreviewPanel._frame_channel_count

### lines 1759-1760

```python
return 1
```

A plain 2-D frame still has one channel; reporting zero would leave the dropdown empty and looking broken.

### line 1762  _(unsure)_

```python
if frame.shape[-1] <= 8 and frame.shape[0] > 8:
```

Same channel-axis heuristic ``frame_channel`` applies.

## TimelapsePreviewPanel._on_fov_changed

### lines 1876-1877

```python
self.load_sequence_async(path, list_siblings=False)
```

The path came out of the sampler, so the folder is already listed; re-listing would rediscover what is in hand.

## TimelapsePreviewPanel.load_masks

### lines 1951-1952

```python
self._mask_load_token += 1
```

A synchronous open supersedes anything the picker started, or the in-flight job would install its own masks over these on arrival.

## TimelapsePreviewPanel._start

### lines 2179-2180  _(unsure)_

```python
if blocked and blocked != self.PREVIEW_SOURCE_HINT:
```

A missing tracking backend is a *result* as well as a refusal: the panel's listeners are told the preview produced nothing.

### lines 2210-2212

```python
note = "Reading the label images, then linking…"
```

It says what it is about to do. Loaded label images are read, not segmented, and claiming otherwise made a fast pass look like a Cellpose run that had hung.

### lines 2219-2220  _(unsure)_

```python
worker.preview_token_value = self.preview_token()
```

The generation this pass belongs to. Cancelling bumps the panel's token, and a result whose token no longer matches is dropped.

### lines 2223-2225

```python
worker.finished_result.connect(self._on_worker_done)
```

Bound method, not a closure: PySide6 delivers a plain-callable connection on the *worker* thread, which would put every widget touch below on the wrong thread.

### lines 2227-2232

```python
worker.finished.connect(self._on_worker_finished)
```

NOT worker.deleteLater — that hands Qt a second owner for an object Python already holds, and the two race (the measured account is in spacr.qt.bridge.make_thread). The Mask preview was fixed away from this pattern; keeping it here left a running QThread owned by nobody when the user closed the screen mid-pass, because the result slot dropped the panel's own reference before the thread had exited.

## TimelapsePreviewPanel._on_worker_done

### lines 2277-2279

```python
if self._worker is not None:
```

The pass is over as far as the panel is concerned, so a new one may start; the reference is kept until ``QThread.finished`` because a QThread collected while it is still unwinding aborts the process.

## TimelapsePreviewPanel._movie_source_paths

### lines 2405-2406  _(unsure)_

```python
return [current] + [path for path in siblings if path != current]
```

Preserve sibling_sources' deterministic order, but the field the user chose is unconditionally first.

## TimelapsePreviewPanel._refresh_movie_targets

### lines 2471-2474

```python
self._cancel_pending_movie_field()
```

Jobs are serialized, and build_movie_field checks the QThread's interruption flag between frames. Lowering the cap therefore cancels the one surplus field instead of letting a whole queue segment and then throwing its arrays away.

## TimelapsePreviewPanel._push_to_movie

### lines 2585-2587

```python
images = self._masks
```

A truthful, immediately available placeholder. It is replaced by raw source frames on the movie worker before being counted as a current/ready entry.

## TimelapsePreviewPanel._pick_masks

### lines 2716-2717  _(unsure)_

```python
self.load_masks_async(path)
```

Async, like `_pick_sequence`: the open lists the folder, and the folder is whatever the user just pointed at.

## TimelapsePreviewPanel.closeEvent

### lines 2728-2729

```python
self.shutdown()
```

Cancel the load before waiting on the segmentation worker: leaving the screen mid-open must not leave a QThread behind either.

### lines 2731-2732

```python
for worker in (self._worker, getattr(self, "_retired_worker", None)):
```

Both of them: the pass in flight, and the one whose result has landed while its thread was still unwinding.

## build_timelapse_preview_card

### lines 2787-2790

```python
movie = TimelapseMoviePanel(card)
```

The movie sits under the tuning controls, not beside them: it is what you look at after a pass to find out WHY the numbers came out the way they did, and a track break is found by scrubbing frames rather than by reading a fragmentation figure.

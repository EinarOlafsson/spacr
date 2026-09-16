# Notes from `spacr/qt/tutorial/engine.py`

Prose lifted out of `spacr/qt/tutorial/engine.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Step](#step) (1 entry)
- [Recorder](#recorder) (1 entry)
- [Recorder.refresh_base](#recorderrefresh_base) (2 entries)
- [Recorder.snap](#recordersnap) (1 entry)
- [RenderResult](#renderresult) (1 entry)
- [Director._run_capture](#director_run_capture) (6 entries)
- [Director._resolve_target](#director_resolve_target) (2 entries)
- [Director._animate_cursor](#director_animate_cursor) (1 entry)
- [Director._concat_audio](#director_concat_audio) (1 entry)
- [Director._make_silence](#director_make_silence) (1 entry)
- [Director.render](#directorrender) (1 entry)
- [render_tutorial](#render_tutorial) (1 entry)

## Step

### lines 45-47

```python
@dataclass
```

Step — the atomic unit of a tutorial

## Recorder

### lines 227-229  _(unsure)_

```python
class Recorder:
```

Recorder — captures the MainWindow at FRAME_RATE

## Recorder.refresh_base

### lines 272-283

```python
pm = self.window.grab()
```

A VIDEO FRAME IS MEASURED IN FILE PIXELS, not screen ones, so this is the one picture in the application that does NOT go through `hidpi.scaled_for`: the recording must come out the same size on every machine.

`grab()` does come back carrying the window's device pixel ratio, and a pixmap that says it is dense draws at a fraction of its size on the plain canvas below -- a quarter-size picture in the corner of every frame on a retina display. Declaring the grab plain before scaling keeps the extra pixels a HiDPI window really has (they make the downscale sharper) and drops the claim that would halve the frame.

### lines 290-291  _(unsure)_

```python
if pm.size().width() != self.size[0] or pm.size().height() != self.size[1]:
```

If the grab is smaller than the target frame (e.g. window smaller than VIDEO_SIZE), centre it on a black canvas

## Recorder.snap

### line 322  _(unsure)_

```python
pm = QPixmap(self._base_frame)
```

Overlay painters mutate their pixmap, so copy the clean keyframe.

## RenderResult

### lines 339-341  _(unsure)_

```python
@dataclass
```

Director — orchestrates the whole render

## Director._run_capture

### line 431  _(unsure)_

```python
self.window.resize(VIDEO_SIZE[0], VIDEO_SIZE[1])
```

Force window to VIDEO_SIZE for consistent frames

### line 441  _(unsure)_

```python
self._recorder.cursor_pos = (
```

Start cursor at bottom-right (out of the way)

### line 451  _(unsure)_

```python
target_pos = self._resolve_target(step)
```

Cursor animation (if target set)

### line 465  _(unsure)_

```python
if step.action is not None:
```

Fire action (if any)

### lines 469-470  _(unsure)_

```python
for _ in range(3):
```

Let queued layout/paint work settle, then capture one new visual keyframe for the changed UI state.

### line 477  _(unsure)_

```python
for _ in range(budget - move_frames):
```

Fill remaining frames, letting the UI catch up between grabs

## Director._resolve_target

### lines 546-547  _(unsure)_

```python
sx, sy = self._scale_factors()
```

Scale window coords to VIDEO_SIZE (they should already match since we resized, but be defensive)

### lines 551-553

```python
LOG.warning("tutorial step target did not resolve (%r): %s",
```

Never let a stale target abort a render — but never let it pass unnoticed either. A silent None here is exactly how a tutorial ends up pointing at nothing.

## Director._animate_cursor

### line 603  _(unsure)_

```python
eased = 0.5 * (1 - math.cos(math.pi * t))
```

Smooth ease-in-out

## Director._concat_audio

### lines 619-620

```python
lines: List[str] = []
```

Use ffmpeg's concat demuxer. Build a list file with a silent

WAV inserted after each step for hold_ms.

## Director._make_silence

### line 640

```python
"""Write a silent audio file.
```

Match Piper's sample rate (22050) + mono + 16-bit

## Director.render

### lines 731-733

```python
try:
```

Cleanup scratch dir. A render that succeeded must not fail because its temp dir could not be removed — but say so, or the frames quietly pile up in /tmp for the rest of the session.

## render_tutorial

### lines 772-774

```python
if app_key not in AVAILABLE_TUTORIALS:
```

Validate before booting: MainWindow takes ~10 s to construct, and a typo should not cost that before it is reported. build_steps stays the authority — this only front-runs it.

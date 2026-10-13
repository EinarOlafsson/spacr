# Notes from `spacr/zstack.py`

Prose lifted out of `spacr/zstack.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [project](#project) (1 entry)
- [_plane_iou](#_plane_iou) (1 entry)
- [stitch_planes](#stitch_planes) (1 entry)
- [flag_truncated_z](#flag_truncated_z) (1 entry)
- [volume_stats](#volume_stats) (1 entry)
- [segment_3d](#segment_3d) (2 entries)
- [plan_from_settings](#plan_from_settings) (2 entries)
- [estimate_peak_bytes](#estimate_peak_bytes) (3 entries)
- [detect_axes](#detect_axes) (2 entries)
- [resolve_axis_order](#resolve_axis_order) (1 entry)
- [TStackSpec.__post_init__](#tstackspec__post_init__) (1 entry)
- [TStackSpec.to_z_spec](#tstackspecto_z_spec) (1 entry)
- [segment_4d](#segment_4d) (1 entry)
- [_displacement_scale](#_displacement_scale) (1 entry)
- [_centroid_matches](#_centroid_matches) (1 entry)
- [track_4d](#track_4d) (5 entries)
- [plan_4d_from_settings](#plan_4d_from_settings) (3 entries)
- [estimate_peak_bytes_4d](#estimate_peak_bytes_4d) (1 entry)

## Module level

### line 98  _(unsure)_

```python
"AXIS_ORDER_TZYX", "AXIS_ORDER_ZTYX", "AXIS_ORDER_TYX", "AXIS_ORDERS",
```

4D (Beta): t on top of z

### lines 1400-1494

```python
AXIS_ORDER_TZYX = "TZYX"
```

4D (Beta): the time axis on top of the z axis

Time-plus-z handling for the 4D (Beta) settings: x, y, z, t.

This is the *t* half of spaCR's volumetric support and it sits directly on top of the z half above. Everything z-shaped -- ZStackSpec, segment_3d, stitch_planes, resolve_anisotropy, flag_truncated_z, volume_stats -- is delegated to rather than re-derived. Like the z half it is free of Cellpose and of any tracker library, so the 4-D logic can be tested against synthetic label volumes on a CPU in milliseconds: every entry point takes a plain numpy array plus a caller-supplied ``segment_fn``.

Five things drive the design.

The axis order is the crux, and it cannot be guessed. ``(T, Z, Y, X)`` and ``(Z, T, Y, X)`` are both written by real microscopes and a 4-D shape does not say which one you have: ``(10, 21, 512, 512)`` is either ten timepoints of twenty-one planes or twenty-one timepoints of ten planes, and nothing in the array distinguishes them. Getting it wrong does not crash -- it links objects *across z* and calls the result a track, which produces smooth, plausible, entirely fictional trajectories. :func:`detect_axes` therefore returns ``None`` for the ambiguous case and never picks a side; the order must come from the user, from ``t_axis_order``, or from an explicit ``n_t``/``n_z`` that settles it. (spaCR's own ingest already gets this wrong: ``io.py``'s 4-D TIFF branch hard-codes ``t_dim, z_dim, y_dim, x_dim = images.shape`` with no check.)

A tracker that cannot do 3-D must not be handed a volume. Silently projecting z away and linking the projection would give a table that looks exactly like a real one. :func:`track_4d` refuses, names the backend, and says whether the limit is the library's or spaCR's adapter's see :data:`TRACK_BACKENDS`. Projection is available, but only when the caller asks for it by name (``project_for_tracking``), and it then says in ``notes`` what it destroyed.

Anisotropy applies to linking, not just to segmentation. A displacement gate expressed in pixels means something different along z: at ``dz/dxy = 5`` a two-plane move is a ten-pixel move. The distance-based backends therefore scale the z component of every displacement by the anisotropy before comparing it with the gate, and refuse to run without one (:func:`~spacr.zstack.resolve_anisotropy` raises rather than assuming 1.0). ``max_displacement_px`` is measured in **xy pixels** with z so scaled; ``max_displacement_um`` is measured in **micrometres** and needs a voxel size. The overlap-based backend has no distance in it at all, so anisotropy genuinely does not enter -- exactly as in :data:`~spacr.zstack.MODE_STITCH`.

The tracks table keeps its existing columns. ``frame`` / ``track_id`` / ``original_label`` / ``x`` / ``y`` are emitted in that order with the same meanings ``timelapse._relabelled_stack_to_tracks_df`` already gives them, so the track visualiser and the motility assay need no change. ``z`` and the volume columns are *additional*. A stack with no z axis gets ``area_px2`` and a volumetric one gets ``volume_voxels``; the two are never written into the same column, because a px^2 area and a voxel count are different quantities (the point :data:`spacr.zstack.VOLUME_STATS_UNITS` exists to make).

Truncation now has two directions. An object touching the first or last z plane is cut off in z, exactly as ``seg_qc`` treats an object touching the xy field edge; a track present in the first or last *timepoint* is cut off in t -- it began before the movie did or was still going when it stopped, so its lifetime, its displacement and its division count are all lower bounds. :func:`volume_tracks` flags both, in separate columns, because they are different defects.

Memory

A 4-D acquisition is ``n_t * n_z`` fields. :func:`iter_volumes` yields **views into the input, one ``(Z, Y, X)`` timepoint at a time, and never materialises the 4-D intensity array; :func:`segment_4d` holds exactly one volume plus whatever the segmenter transiently needs (see :func:`spacr.zstack.estimate_peak_bytes`). What it *does* have to hold is the label array for every timepoint, because linking across t cannot start until the last timepoint is segmented; those are int32, so a 41-timepoint, 21-plane, 2048x2048 acquisition costs ~14 GB of labels against ~350 MB for the one live float32 volume. :func:`estimate_peak_bytes_4d` gives the number.

Scope, stated plainly

This reaches exactly as far as the z half above does, which is to say the library is real and the pipeline cannot feed it. ``spacr.io`` MIPs z away while it organises raw files -- for a 4-D TIFF at ``io.py:5051`` and for a LIF at ``io.py:5009`` -- so by the time a timelapse batch reaches segmentation it is ``(frames, Y, X, C)`` and the z axis no longer exists. On top of that, none of spaCR's five tracker adapters accepts a 4-D array: ``btrack`` and the ``trackastra``/``ultrack`` adapters raise on ``ndim != 3``, and the trackpy/iou feature table raises out of skimage. :func:`plan_4d_from_settings` therefore returns ``None`` whenever ``t_stack`` is off -- the default -- so not one line of the t half executes in an ordinary run, and when it is on without a real 4-D array the callers raise :class:`TAxisNotPresentError` naming the cause. spaCR will not project a volume, link the projection, and call the result a 4-D track.

## project

### lines 823-825

```python
if vol.shape[0] == 1:
```

best_focus: keep the sharpest single plane rather than blending planes, which is what you want when only one plane is actually in focus and a MIP would drag every out-of-focus plane's haze into the result.

## _plane_iou

### lines 905-906

```python
overlap = np.zeros((prev_ids.size, cur_ids.size), dtype=np.int64)
```

searchsorted rather than a dict lookup per pixel: a field can carry hundreds of objects and this runs once per plane pair.

## stitch_planes

### lines 967-969

```python
first = np.asarray(planes[0])
```

Plane 0: every object starts a new 3-D object. Relabelling goes through a lookup table so the cost is one pass over the plane, not one pass per label.

## flag_truncated_z

### lines 1033-1035  _(unsure)_

```python
def flag_truncated_z(labels) -> np.ndarray:
```

Truncation at the ends of the stack

## volume_stats

### lines 1140-1141

```python
z_min = np.full(n_labels + 1, np.inf)
```

Seeded with the infinities rather than NaN: np.minimum propagates NaN, so a NaN seed would leave every extent undefined.

## segment_3d

### lines 1241-1242  _(unsure)_

```python
if n_z == 1:
```

A single plane is 2-D. Not a degenerate volume, not a 1-plane stitch the ordinary path, returning an ordinary 2-D mask.

### line 1276, trailing  _(unsure)_

```python
else:
```

MODE_VOLUMETRIC

## plan_from_settings

### lines 1349-1351

```python
resample_to_isotropic=False,
```

Cellpose does its own z rescaling under do_3D, so the pipeline hands it `anisotropy` rather than pre-stretching the volume. Direct API callers with a segmenter that does not can set this on the spec.

### lines 1355-1356

```python
if spec.mode == MODE_VOLUMETRIC:
```

Fail here rather than after the model has been loaded and the first field read: the answer cannot change later in the run.

## estimate_peak_bytes

### line 1389  _(unsure)_

```python
planes = int(volume_shape[0]) if len(volume_shape) else 1
```

The volume plus one plane; z is gone immediately.

### line 1393  _(unsure)_

```python
return volume_bytes + n_voxels * 4 + n_voxels * 8
```

Volume + per-plane labels + the stitched int64 output.

### line 1395  _(unsure)_

```python
iso = int(volume_bytes * max(anisotropy, 1.0))
```

Volumetric: the isotropic copy dominates, plus a 3-component flow field.

## detect_axes

### line 1843  _(unsure)_

```python
candidates = [(a0, a1), (a1, a0)]
```

Candidate readings: (t_axis, z_axis).

### line 1874  _(unsure)_

```python
if strict:
```

Both readings survive: the hint did not discriminate.

## resolve_axis_order

### line 1947  _(unsure)_

```python
leading = [i for i in kept[:2]]
```

One given, the other is whichever leading axis is left.

## TStackSpec.__post_init__

### lines 2106-2108

```python
self.to_z_spec()
```

Validated by ZStackSpec, which owns these three; constructing it here means an impossible z_mode/projection/anisotropy is refused when the spec is built and not after the first field has been read.

## TStackSpec.to_z_spec

### line 2138, trailing  _(unsure)_

```python
z_axis=0,
```

iter_volumes always yields z-first volumes

## segment_4d

### line 2363  _(unsure)_

```python
result = ZStackResult(
```

A flat time series: one 2-D call per frame, no z code at all.

## _displacement_scale

### lines 2536-2537  _(unsure)_

```python
return np.array([float(anisotropy), 1.0, 1.0][3 - ndim:], dtype=float)
```

xy-pixel space: x and y are already in pixels, z is `anisotropy` pixels per plane. This is the whole reason anisotropy matters to tracking.

## _centroid_matches

### lines 2565-2566

```python
big = float(max_distance) * 1e6 + 1.0
```

Gate first so the solver never prefers a long link just to complete a permutation, then drop anything that is still over the gate.

## track_4d

### lines 2745-2750

```python
raise TrackerIsTwoDError(
```

Deliberately not unlocked by project_for_tracking. This module does not drive this backend at all -- it lives in spacr.timelapse and takes (T, Y, X) -- so projecting here and then linking with a DIFFERENT linker would report one tracker's answer under another tracker's name, which is a worse lie than the one this refusal prevents.

### lines 2765-2767

```python
labels = np.stack([project_labels(frame) for frame in labels], axis=0)
```

A real choice with a real cost, and only ever made explicitly: link the projection rather than the volume. Faster and less sensitive to a wrong anisotropy, at the price of fusing anything stacked in z.

### lines 2781-2783

```python
tracked = stitch_planes(labels, iou_threshold=threshold)
```

Linking across t by overlap is the same algorithm as linking across z by overlap, so this is zstack.stitch_planes applied along axis 0 which here is time, explicitly, never inferred.

### lines 2813-2815

```python
aniso_used = resolve_anisotropy(aniso_value, spec.voxel_size_um)
```

Required, not defaulted: at dz/dxy = 5 a two-plane move is a ten-pixel move, and treating it as two lets objects five times too far apart link.

### lines 2861-2864

```python
raise TStackError(
```

Reached when one of the three adapter-only backends is asked for on a flat (T, Y, X) stack. That is a perfectly reasonable thing to want it is just not this module's job, and pretending otherwise would mean quietly substituting a built-in linker for the one named.

## plan_4d_from_settings

### lines 3118-3121

```python
if z_axis is not None:
```

A flat time series: no z axis is claimed, so there is no order to refuse and nothing to disambiguate. This is the common case -- an ordinary 2-D movie -- and it is the only spelling that makes it expressible from the settings.

### lines 3142-3145

```python
for label, given, implied in (("t_axis", t_axis, name.index("T")),
```

An explicit t_axis/z_axis alongside the order must agree with it. Letting the order silently win would mean a user who set both, and got one of them wrong, is segmenting a differently-transposed array than they think -- which is the failure this setting exists to stop.

### lines 3192-3193

```python
if spec.z_axis is not None and spec.z_mode == MODE_VOLUMETRIC:
```

Fail here rather than after the model has been loaded and the first timepoint read: none of these answers can change later in the run.

## estimate_peak_bytes_4d

### line 3274, trailing  _(unsure)_

```python
label_voxels = int(np.prod(volume_shape[1:3]))
```

z is gone

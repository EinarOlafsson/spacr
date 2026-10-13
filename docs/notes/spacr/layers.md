# Notes from `spacr/layers.py`

Prose lifted out of `spacr/layers.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [colormap](#colormap) (1 entry)
- [Blending.apply](#blendingapply) (1 entry)
- [_default_axes](#_default_axes) (1 entry)
- [_extent_of](#_extent_of) (1 entry)
- [Canvas.covering](#canvascovering) (1 entry)
- [OrthoViews.covering](#orthoviewscovering) (1 entry)
- [FieldKey](#fieldkey) (1 entry)
- [Layer._check_opacity](#layer_check_opacity) (1 entry)
- [Layer.name](#layername) (1 entry)
- [LabelsLayer.brush_index](#labelslayerbrush_index) (1 entry)
- [PointsLayer.__init__](#pointslayer__init__) (1 entry)
- [PointsLayer._draw](#pointslayer_draw) (1 entry)
- [ShapesLayer._inside_polygon](#shapeslayer_inside_polygon) (1 entry)

## Module level

### lines 72-79

```python
__all__ = [
```

The object identity every table in measurements.db already agrees on. A labels layer that invented its own key scheme would be a fifth island; the whole point of `linked_selection` is that there are no more of those. Imported inside the methods that use them, not here. `spacr.selection` imports pandas at module scope, and every one of the four uses below is at CALL time -- so a module-scope import here bought nothing and put ~200 ms of pandas on the Qt startup path, which is instruction 55. Measured: `spacr.qt.app` -> `spacr.layers` -> `spacr.selection` -> pandas.

## colormap

### line 351, trailing  _(unsure)_

```python
try:
```

depends on the installed matplotlib

## Blending.apply

### line 490, trailing  _(unsure)_

```python
else:
```

translucent and opaque share the source-over arithmetic

## _default_axes

### lines 495-497

```python
def _default_axes(ndim: int) -> Tuple[str, ...]:
```

Spacing — the one thing that must not be guessed

## _extent_of

### lines 722-724  _(unsure)_

```python
def _extent_of(source: Any) -> Tuple[Dict[str, Tuple[float, float]], str]:
```

Canvas — a window onto the world

## Canvas.covering

### line 899  _(unsure)_

```python
origin = (origins[0] + 0.5 * step[0], origins[1] + 0.5 * step[1])
```

Pixel centres, not corners: half a step in from the extent edge.

## OrthoViews.covering

### lines 1111-1112  _(unsure)_

```python
scale = spans[column_axis] / width
```

One scale for all three panels. Anything else is the squashed side view this class exists to prevent.

## FieldKey

### lines 1578-1580  _(unsure)_

```python
@dataclass(frozen=True)
```

Object identity for a labels layer

## Layer._check_opacity

### lines 1794-1795

```python
return min(1.0, max(0.0, v))
```

Clamped rather than refused: a slider that overshoots by a float rounding error should not raise in a paint handler.

## Layer.name

### line 1820  _(unsure)_

```python
self._stack.rename(self, text)
```

Go through the stack so uniqueness is still enforced.

## LabelsLayer.brush_index

### line 2564  _(unsure)_

```python
def brush_index(self, world: Mapping[str, float], *,
```

editing (the seam the brush item builds on)

## PointsLayer.__init__

### lines 2710-2712

```python
self._default_size = (float(size) if np.isscalar(size)
```

Kept apart from `_size` so that the layer's declared size survives an empty layer: `_size` is empty until the first point exists, and a counting layer starts empty by definition.

## PointsLayer._draw

### line 2978  _(unsure)_

```python
return rgb, coverage
```

This layer does not live in the plane being drawn.

## ShapesLayer._inside_polygon

### lines 3310-3311

```python
edge_c = vc[i] + (r - vr[i]) * (vc[j] - vc[i]) / (dr if dr else 1.0)
```

`dr == 0` only where `crosses` is False, so the guarded divide never contributes; guarding it keeps numpy from warning.

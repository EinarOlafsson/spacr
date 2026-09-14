# Notes from `spacr/qt/widgets/gate_spec.py`

Prose lifted out of `spacr/qt/widgets/gate_spec.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [points_in_polygon](#points_in_polygon) (1 entry)
- [Gate.kind](#gatekind) (1 entry)
- [Gate.columns](#gatecolumns) (1 entry)
- [Gate.describe](#gatedescribe) (1 entry)
- [Gate.to_dict](#gateto_dict) (1 entry)
- [Gate.translated](#gatetranslated) (1 entry)
- [Gate.handles](#gatehandles) (1 entry)
- [ThresholdGate.centre](#thresholdgatecentre) (1 entry)
- [ThresholdGate.with_handle](#thresholdgatewith_handle) (1 entry)
- [ThresholdGate.scaled](#thresholdgatescaled) (1 entry)
- [RectGate.with_handle](#rectgatewith_handle) (1 entry)
- [PolygonGate.__post_init__](#polygongate__post_init__) (2 entries)
- [EllipseGate.mask](#ellipsegatemask) (1 entry)
- [EllipseGate.with_handle](#ellipsegatewith_handle) (1 entry)
- [EllipseGate.scaled](#ellipsegatescaled) (1 entry)
- [ClusterError](#clustererror) (1 entry)
- [_convex_hull._half](#_convex_hull_half) (1 entry)
- [Module level](#module-level) (1 entry)
- [CylinderGate.mask](#cylindergatemask) (1 entry)
- [CompositeGate.mask_with](#compositegatemask_with) (1 entry)
- [wand_select](#wand_select) (3 entries)
- [_cluster_matrix](#_cluster_matrix) (1 entry)
- [cluster_walk_candidates](#cluster_walk_candidates) (1 entry)
- [_fit_labels](#_fit_labels) (1 entry)
- [cluster_gates](#cluster_gates) (2 entries)
- [GateClause](#gateclause) (1 entry)
- [GateSet.__post_init__](#gateset__post_init__) (1 entry)
- [GateSet.add](#gatesetadd) (1 entry)
- [GateSet._mask_of](#gateset_mask_of) (1 entry)

## points_in_polygon

### lines 237-238

```python
straddles = (yi > y) != (yj > y)
```

A horizontal edge never straddles the ray, so the division it would divide by zero for is masked out before it is used.

## Gate.kind

### line 305, trailing  _(unsure)_

```python
def kind(self) -> str:
```

overridden

## Gate.columns

### line 318, trailing  _(unsure)_

```python
def columns(self) -> Tuple[str, ...]:
```

overridden

## Gate.describe

### line 374, trailing  _(unsure)_

```python
def describe(self) -> str:
```

overridden

## Gate.to_dict

### line 385, trailing  _(unsure)_

```python
def to_dict(self) -> Dict[str, Any]:
```

overridden

## Gate.translated

### lines 416-426

```python
def translated(self, dx: float, dy: float) -> "Gate":
```

editing after the fact

A gate you cannot adjust is a gate you redraw from scratch, which is the single biggest gap in this editor. Both operations return a NEW gate rather than mutating: these are frozen dataclasses, the GateSet holds them by name, and an in-place edit would change a gate that something else is already holding a reference to.

Both are defined on the base so a caller can move ANY gate without knowing which kind it has -- which is what the canvas drag handler needs, since the user just clicks a shape.

## Gate.handles

### lines 463-469

```python
def handles(self, view: "View") -> Tuple["Handle", ...]:
```

anchor points

Resizing is "pull a corner or a side", so every kind has to be able to say where its corners and sides ARE, and what it becomes when one is dragged. Both live here rather than in the canvas because they are geometry -- no axes, no pixels, no Qt -- and because a canvas that special-cased four gate kinds inside a mouse handler is how the drag code became unreadable the first time.

## ThresholdGate.centre

### lines 642-644

```python
return None, None
```

Open-ended, so there is no middle. Reported rather than invented: a made-up centre would send the first resize somewhere arbitrary.

## ThresholdGate.with_handle

### lines 685-687

```python
low, high = high, low
```

Dragged past the other bound. Swapping beats refusing: the user's intent is unambiguous and a gate that will not invert feels stuck at exactly the moment they are trying to fix it.

## ThresholdGate.scaled

### lines 705-707

```python
return self
```

Nothing to scale about, and nothing sensible to do. Returned unchanged rather than raising: the user dragged, and a half-open gate simply has no width to grow.

## RectGate.with_handle

### lines 957-959

```python
values[lo], values[hi] = b, a
```

Pulled through the opposite side. The user has turned the rectangle inside out, which they clearly meant; keeping it a rectangle is the only correction needed.

## PolygonGate.__post_init__

### lines 1024-1025

```python
points = points[:-1]
```

A closing vertex is accepted and dropped: the polygon closes itself, and keeping the duplicate would make an edge of length 0.

### lines 1032-1035

```python
xs = np.array([p[0] for p in points], dtype=float)
```

The shoelace area. Zero means the vertices are collinear (or all in one place), which is a *line*: it would select nothing, and a gate that selects nothing because of a slipped click is worth catching at the click rather than three screens later.

## EllipseGate.mask

### lines 1245-1247

```python
dx = (x - self.x_centre) / self.x_radius
```

Normalised radius: <= 1 is inside. Written this way rather than as a distance so the two axes keep their own scales -- the whole point of an ellipse over a circle on a two-measurement scatter.

## EllipseGate.with_handle

### lines 1329-1331

```python
return self
```

A zero radius is not an ellipse and EllipseGate refuses one. Handing back the gate unchanged makes the handle stop at the centre instead of the drag raising into a mouse handler.

## EllipseGate.scaled

### line 1346  _(unsure)_

```python
return replace(self, x_radius=self.x_radius * f,
```

Grow in place: the centre is fixed and only the radii change.

## ClusterError

### lines 1379-1393

```python
class ClusterError(GateError):
```

Density clustering

DBSCAN, not k-means: a scatter of cells has dense populations of unequal size sitting in sparse debris, which is exactly the shape DBSCAN was made for and exactly the shape k-means is bad at. It also does not need to be told how many populations there are, which is the number a user opening this dialog does not yet know.

Clusters become REAL GATES rather than a separate kind of selection. A cluster is then editable, nestable, serialisable and usable as a DataFilter clause -- everything a hand-drawn gate can do -- because it IS one. A parallel "cluster selection" concept would have needed all of that rebuilt beside it.

## _convex_hull._half

### line 1426  _(unsure)_

```python
cross = ((x2 - x1) * (point[1] - y1)
```

Cross product of the last edge with the candidate edge.

## Module level

### lines 1688-1690

```python
_GATE_CLASSES[BOX] = BoxGate
```

Registered after the class rather than in the literal above: BoxGate is defined further down the file, beside the volume it belongs to, and a forward reference in the dict would be a NameError at import.

## CylinderGate.mask

### line 1786  _(unsure)_

```python
return np.zeros(len(frame), dtype=bool)
```

A zero radius is an empty gate, not a division by zero.

## CompositeGate.mask_with

### lines 2238-2240

```python
out = masks[0].copy()
```

subtract: the FIRST operand minus every other. Order matters, and it is the order the user listed them in -- A minus B is not B minus A, and a set that sorted its operands would silently change which.

## wand_select

### lines 2353-2355

```python
sx, sy = _unit_scale(xs[finite]), _unit_scale(ys[finite])
```

Map each axis onto 0..1 across the DATA, not the view: a gate is a statement about measurements, and scaling by the visible window would make the same click give a different gate at a different zoom.

### lines 2372-2374

```python
seed = int(np.argmin(from_click))
```

The seed is the nearest object to the click, NOT the click itself: the user points at a cloud, and a click landing in a gap between two of its objects must still start inside the cloud.

### line 2388  _(unsure)_

```python
remaining = np.flatnonzero(~selected)
```

Distance from every unselected candidate to the newest selections.

## _cluster_matrix

### lines 2489-2492

```python
flat = [column for column, sd in zip((x_column, y_column), spread)
```

A constant axis is refused rather than worked around. Every cluster on it is a straight line, every hull is collinear and has no area, and the honest result would be an empty list -- which reads as "clustering is broken" rather than "this measurement is the same for every object".

## cluster_walk_candidates

### lines 2586-2588

```python
try:
```

Silhouette is defined on the CLUSTERED points only. Including noise as if it were one more cluster would reward runs that discard the awkward objects, which is the opposite of useful.

## _fit_labels

### lines 2651-2654

```python
return HDBSCAN(min_cluster_size=max(2, int(min_samples)), copy=False,
```

`eps` becomes the floor below which HDBSCAN stops splitting, so the control keeps the meaning it has for DBSCAN -- larger merges. Zero (its own default) would make the setting inert, which is the defect this whole change is about.

## cluster_gates

### lines 2706-2707  _(unsure)_

```python
found.sort(key=lambda lab: int((labels == lab).sum()), reverse=True)
```

Largest first, so the populations that matter are drawn and named before the specks.

### lines 2714-2715

```python
continue
```

A collinear cluster has no area. Skipped rather than widened into a fake polygon, which would select rows outside it.

## GateClause

### lines 2757-2759  _(unsure)_

```python
@dataclass(frozen=True)
```

The clause every linked view honours

## GateSet.__post_init__

### lines 2895-2897

```python
"""Re-add every incoming gate through :meth:`add`.
```

Re-added one at a time through `add`, so a set built from a list — or read back from a file — gets the same parent and cycle checks a set built by clicking does.

## GateSet.add

### lines 2930-2931

```python
self.gates = before
```

Put the set back exactly as it was. Re-drawing a gate into a cycle must not also delete the gate it was replacing.

## GateSet._mask_of

### lines 3109-3111

```python
lookup[operand] = self._mask_chain(
```

Each operand carries its OWN ancestors, because a gate drawn inside another means the pair, and combining it as though it were the shape alone would include rows its parent excluded.

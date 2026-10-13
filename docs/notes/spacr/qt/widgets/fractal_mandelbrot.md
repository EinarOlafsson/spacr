# Notes from `spacr/qt/widgets/fractal_mandelbrot.py`

Prose lifted out of `spacr/qt/widgets/fractal_mandelbrot.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [steering_from_one_number](#steering_from_one_number) (3 entries)
- [ReferenceOrbit.__init__](#referenceorbit__init__) (1 entry)
- [ReferenceOrbit._build](#referenceorbit_build) (2 entries)
- [perturbation_escape_map](#perturbation_escape_map) (1 entry)
- [structure_mask](#structure_mask) (1 entry)
- [plan_guided_step](#plan_guided_step) (6 entries)
- [a_more_interesting_anchor](#a_more_interesting_anchor) (5 entries)
- [best_reference_in_view](#best_reference_in_view) (4 entries)

## Module level

### lines 87-90

```python
"gpu_fp64": False,
```

FP64 OFF. The double-precision shader needs GLSL 400, which many drivers lack or emulate at a cost far larger than the precision is worth: perturbation is what buys the depth, not the shader's float width.

### lines 92-102

```python
"path": "tour",
```

THE TOUR, ON BY DEFAULT since 2026-09-10 at the maintainer's decision. 327(3) asked for "say 20 regions on the image that the camera will automatically smoothely float towards", and an opt-in nobody opens is not that.

IT IS NOT "guided" AND CANNOT SHAKE THE WAY THAT DID. Guided SEARCHES while it draws, and moving the camera on a survey is what was reported as jumping. The tour visits coordinates found once and written down, eases out of one and into the next with zero velocity at both ends, and stops steering entirely once the view is narrower than the region's own measured half-width. "fixed" remains one dropdown away.

### lines 104-105  _(unsure)_

```python
"steering_strength": 0.09,
```

Kept so a guided path can still be asked for, at the values the original uses for it.

## steering_from_one_number

### line 274

```python
interval = 3.0 - 2.6 * amount
```

Restless steers about every 0.4 decades; calm about every 3.

### lines 276-277

```python
strength = 0.02 + 0.16 * amount
```

And reaches less far when it is calm, so a rare move is also a small one rather than a lurch after a long wait.

### lines 279-281

```python
duration = min(6.0, 0.45 * interval * seconds)
```

HALF THE INTERVAL AT MOST, which is the rule that removes the jerkiness: however short the interval gets, the move finishes with time to spare before the next is planned.

## ReferenceOrbit.__init__

### lines 370-376

```python
self.packed = np.zeros((2, self.max_iter + 1, 4), dtype=np.float32)
```

TWO ROWS, SIX FLOATS. Row 0 holds the high and middle words of each component and row 1 the low ones, because a texel has four channels and Z needs three floats apiece to be worth carrying.

Two floats reproduce Z to 2.2e-16 -- about 15.7 decades, which is where the picture turned to mush. Three reach roughly 2^-72, and the depth follows.

## ReferenceOrbit._build

### lines 395-400

```python
re_hi = np.float32(float(real))
```

HIGH AND LOW, because one float32 cannot hold Z at this depth. The shader adds the pair back; the residual is what a single float would have thrown away. EACH WORD IS THE REMAINDER OF THE ONE BEFORE IT, which is what makes the sum more accurate than any single float: the error of the pair becomes the value of the third.

### lines 415-417

```python
self.escaped_at = n + 1
```

A REFERENCE THAT ESCAPES IS NOT A REFERENCE. Every pixel perturbs around it, so the rest is zeroed rather than left holding numbers that mean nothing.

## perturbation_escape_map

### lines 480-493

```python
def perturbation_escape_map(orbit, width, height, scale, max_iter,
```

The guided path

WITHOUT THIS THE DIVE ALWAYS ENDS IN THE SAME PLACE. A fixed path descends to one Misiurewicz point for ever: correct, and the same picture every time. Guided steering looks around every so often, picks a nearby point on the boundary that has structure worth arriving at, and eases the camera onto it -- so the descent keeps finding new things instead of drilling one shaft.

THE LOOK-AROUND IS CHEAP ON PURPOSE. It renders a 96x54 escape map, which is 5,184 points against the two million a frame draws, and it runs on a worker thread. It is a decision about where to go, not a picture.

## structure_mask

### lines 593-594  _(unsure)_

```python
threshold = float(np.quantile(gradient[gradient > 0.0], 0.90))
```

The steepest tenth: enough candidates to choose between, few enough that they are all genuinely on a filament.

## plan_guided_step

### lines 682-683

```python
eligible = edge & (radius >= 0.025) & (radius <= max(0.34, 2.2 * strength))
```

NOT THE POINT ALREADY UNDER THE CAMERA, and not the far corners: the first is where it is going anyway and the second is a lurch.

### line 689, trailing  _(unsure)_

```python
phase = step_index * 2.399963229728653
```

the golden angle

### lines 692-700

```python
point_angle = np.arctan2(screen_y, screen_x)
```

THE HEADING IS A CONSTRAINT, NOT A PREFERENCE. Scoring every boundary point and merely penalising the distant ones lets the most structured point in the frame win whatever direction the step is supposed to be exploring -- measured, six consecutive steps chose two targets between them, which is a fixed path wearing a guided path's settings.

Restricting the candidates to an arc around this step's own heading makes each step go somewhere it has not been, and the golden angle walks that arc around the frame without ever repeating a heading.

### lines 708-717

```python
spread = 2.0 * math.pi / 3.0
```

AN ARC PER STEP, not the whole circle. Spread over 360 degrees the candidate set is nearly the same however the phase is rotated, so the best-scoring point is the same every time -- which is a fixed path wearing a guided path's settings. Measured: six consecutive steps chose one target.

A third of a circle around this step's own heading gives each step somewhere different to look while keeping the move a STEER: the golden angle then walks that window around the frame without ever repeating a heading.

### lines 720-721

```python
want_x = strength * math.cos(angle)
```

Within the arc the candidates fan out, so the choice is still made between real alternatives rather than one point being scored.

### lines 732-734

```python
penalty = math.sqrt(float(masked[row, col])) / max(0.04, strength)
```

A CLOSER TARGET IS WORTH SOMETHING TOO. Left unpenalised the search would keep choosing the most interesting point in the frame, which is a jump rather than a steer.

## a_more_interesting_anchor

### lines 929-930

```python
pick = np.linspace(0, len(rows) - 1, int(candidates)).astype(int)
```

Evenly through the list rather than the first N, which would all come from the top of the frame.

### lines 936-937

```python
if math.hypot(float(grid_x[row, col]),
```

NOT THE VERY EDGE OF THE FRAME: a point there is half outside the survey, so its neighbourhood is scored on missing data.

### lines 952-954

```python
deep_escaped, deep_iterations = perturbation_escape_map(
```

A HUNDREDTH OF THE SCALE: far enough in that anything shallow has been passed through, near enough that the reference orbit is still accurate there.

### lines 960-961  _(unsure)_

```python
if share < 0.02 or share > 0.98:
```

A frame that entirely escapes or entirely does not is one colour, however busy its surface looked.

### lines 967-968  _(unsure)_

```python
return shortlist[0][1], shortlist[0][2]
```

Nothing survived the deeper look: the surface best is still a better guess than the middle of the frame.

## best_reference_in_view

### lines 1025-1035

```python
bounded = ~escaped
```

IN THE SET, AND ON ITS EDGE. Both halves matter and they pull opposite ways.

In the set, because a reference that escapes is not one. On the edge, because the interior is where the picture stops changing: taking the point furthest INSIDE was tried and gives a bounded reference whose neighbourhood is solid colour two decades down -- measured, detail 304 at the surface and 0.0 at depth two.

A boundary point is bounded and has an escaping neighbour, which is exactly the pair of conditions.

### lines 1039-1041

```python
best_row, best_col, best_score = -1, -1, -1.0
```

Among the boundary points, the one whose neighbourhood varies most: that is where the structure is densest and so where a descent keeps finding something.

### lines 1051-1052  _(unsure)_

```python
row, col = np.unravel_index(int(np.argmax(bounded.astype(np.int8))),
```

No edge in view: somewhere inside, which at least keeps the reference valid until the camera is moved again.

### lines 1056-1057  _(unsure)_

```python
row, col = np.unravel_index(int(np.argmax(iterations)),
```

Nothing of the set at all: the longest-lived point is the nearest thing to it this view contains.

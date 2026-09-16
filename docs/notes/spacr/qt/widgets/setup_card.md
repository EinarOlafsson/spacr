# Notes from `spacr/qt/widgets/setup_card.py`

Prose lifted out of `spacr/qt/widgets/setup_card.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [SetupCard.__init__](#setupcard__init__) (3 entries)
- [SetupCard.event](#setupcardevent) (1 entry)
- [SetupCard.paintEvent](#setupcardpaintevent) (1 entry)
- [SetupCard.perimeter_position](#setupcardperimeter_position) (6 entries)
- [SetupCard._tick](#setupcard_tick) (6 entries)
- [SetupCard._paint](#setupcard_paint) (5 entries)
- [SetupCard.accent_span](#setupcardaccent_span) (1 entry)
- [SetupCard.accent_alpha](#setupcardaccent_alpha) (1 entry)
- [SetupCard.ink_at](#setupcardink_at) (1 entry)
- [SetupCard.spaceout_hue](#setupcardspaceout_hue) (1 entry)
- [SetupCard._paint_accent](#setupcard_paint_accent) (4 entries)

## SetupCard.__init__

### lines 126-131

```python
self._arc = int(arc) if arc is not None else self._preferred_arc()
```

THE THREE THAT ARE A MATTER OF TASTE come from the preference store unless the caller names one. How long the run is, how hard it chases, and whether it is centred on the pointer or trails behind it are all things to look at and decide about, so they are settings rather than constants -- and a caller that wants a particular look for a particular card can still say so.

### line 156, trailing

```python
self._timer.setInterval(16)
```

~60fps

### lines 158-159

```python
self.setMouseTracking(True)
```

MOUSE TRACKING, or `mouseMoveEvent` only fires while a button is held -- which is never, on a card the user is only reading.

## SetupCard.event

### lines 196-197  _(unsure)_

```python
"""Handle the events Qt gives no named handler for.
```

A hover move arrives even when the widget has no mouse grab, which is the ordinary case here.

## SetupCard.paintEvent

### lines 232-233

```python
pass
```

Decoration is never load-bearing: an unpainted card is still a card with working controls on it.

## SetupCard.perimeter_position

### lines 272-273  _(unsure)_

```python
span = max(abs(dx) / half_w, abs(dy) / half_h)
```

Scale the ray until it touches whichever pair of sides it reaches first. The larger of the two normalised components decides.

### lines 277-278  _(unsure)_

```python
x = min(max(x, 0.0), width)
```

Rounding can leave it a hair outside; the run below assumes it is on the boundary.

### line 283, trailing  _(unsure)_

```python
if y <= 1e-6:
```

top edge, left to right

### line 285, trailing  _(unsure)_

```python
elif x >= width - 1e-6:
```

right edge, down

### line 287, trailing  _(unsure)_

```python
elif y >= height - 1e-6:
```

bottom edge, right to left

### line 289, trailing  _(unsure)_

```python
else:
```

left edge, up

## SetupCard._tick

### lines 517-518  _(unsure)_

```python
self._phase += self._timer.interval() / 1000.0
```

The animation clock advances whatever else happens, so a pulse keeps its rhythm through a circuit and across a slide change.

### lines 524-527

```python
if abs(self._laps) < 0.031:
```

A LAP ENDS EXACTLY, not approximately: floating error across thirty-odd frames would otherwise leave the accent a little further round after every circuit, and after ten slides it would be somewhere the pointer never put it.

### lines 529-533

```python
self._at = round(self._at + self._laps, 6)
```

THE REMAINDER IS TRAVELLED, not undone: `_laps` is what is still owed, so the last partial step ADDS it. Subtracting left the accent 2% short of home on every circuit, which after ten slides is a fifth of the way round from where the pointer last put it.

### lines 537-539

```python
self._aim_at_the_cursor()
```

WHERE THE POINTER IS NOW, read fresh every frame. See

`_aim_at_the_cursor`: events cannot carry a pointer that is outside the window, and this accent is meant to follow one.

### lines 544-548

```python
return
```

ARRIVED AND NOTHING MOVED. The timer keeps running it is what notices the cursor moving again -- but a repaint of a card that has not changed is sixty needless composites a second over a live backdrop. A pulsing or spectral rim DOES change, so it paints.

### line 552, trailing  _(unsure)_

```python
self._at += gap * self.ease()
```

ease, not jump: water

## SetupCard._paint

### lines 567-568

```python
self._frame = {}
```

THE FRAME OPENS HERE and everything the run is drawn with is read inside it. See :meth:`_held`.

### line 575, trailing  _(unsure)_

```python
body.setAlpha(216)
```

translucent: the blobs show through

### lines 578-584

```python
painter.drawRoundedRect(QRectF(self.rect()),
```

THE BODY COVERS THE WHOLE CARD, not the inset the rim is stroked on. Inset by a pixel it left a gap to whatever sits behind -- a hairline along the straight edges, but half again as wide across the diagonal of each corner, where the drifting backdrop showed through as a blue crescent on every rounded part. The rim still strokes the inset rect, so its width has somewhere to sit.

### lines 588-593

```python
edge = QColor(palette.get("border", palette["fg"]))
```

THE RESTING RIM IS DARK GREY, not a faint white. It was the foreground ink at alpha 38, which on a dark theme is a pale line round the card and competes with the accent travelling along it -- the lit part should be the only bright part. `border` is the palette's own dark grey, and on a light theme it is the grey that reads against white.

### lines 600-615

```python
self._paint_accent(painter, QColor(palette["accent"]), rect)
```

THE ACCENT, a run of rim centred on `position`, FADING AT

BOTH ENDS.

DRAWN AS SEGMENTS OF THE ROUNDED PATH ITSELF rather than as the four hand-built corner paths below. A continuous position cannot be expressed as one of four corners, and `QPainterPathStroker` would give the outline of the stroke rather than a segment of it -- so the segments come from `QPainterPath.pointAtPercent` along the whole rim, which is the one thing Qt measures in arc length for us.

SEGMENT BY SEGMENT, because a QPen carries ONE colour: a run that fades has to be many short strokes, each with its own alpha and its own width. Twenty-four of them is below the threshold at which the joins are visible and well inside the frame budget at 60 fps.

## SetupCard.accent_span

### lines 657-659

```python
rim = QPainterPath()
```

The reference rim is a constant of the radius, so building and measuring it belongs once per length rather than once per frame.

## SetupCard.accent_alpha

### lines 692-693  _(unsure)_

```python
return max(0.0, min(1.0, ramp ** (1.0 + self.FADE)))
```

Squared, so the fall is gentle near the peak and quick at the ends the shape a wake has.

## SetupCard.ink_at

### lines 762-763

```python
spectral.setHsvF(hue, min(1.0, accent.saturationF() + 0.25),
```

SATURATION AND VALUE FROM THE ACCENT, so a rainbow on a pale theme is not the same searing colour as one on a dark theme.

## SetupCard.spaceout_hue

### lines 779-783

```python
drift = self._held("drift", read)
```

ONE DRIFT FOR THE WHOLE RUN. It is a function of the animation clock, which does not advance while a frame is being drawn, so asking per segment returned the same number at the cost of a call -- and a run whose ends had drifted differently from each other would be a fault, not a feature.

## SetupCard._paint_accent

### lines 810-813

```python
pulse = self.beat()
```

THE PULSE IS ONE VALUE FOR THE FRAME. It is read off the animation clock, which does not advance while the frame is being drawn, so asking per segment gave the same answer every time and a run that pulsed along its own length would be a fault.

### lines 815-820

```python
run_px = max(1.0, span * rim_px)
```

ONE STEP PER `STEP_PX` OF RIM, not a fixed count. At 24 segments a run this long was 23 px a step: the alpha moved in visible jumps and every corner was cut into four straight chords, which is the "chunky" of the 2026-08-22 report. The count now follows the length being drawn, so it stays smooth on a card of any size and costs nothing on a small one.

### lines 832-835

```python
middle = (alpha + previous_alpha) / 2.0
```

THE MIDPOINT ALPHA, so a segment is the shade of the rim it covers rather than of the end it stops at -- which is what leaves a visible edge between one segment and the next at the faint end.

### lines 839-845

```python
pen = QPen(ink, 1.2 + 2.2 * middle, Qt.SolidLine,
```

THE WIDTH TAPERS WITH THE ALPHA. A constant-width stroke fading to nothing still shows its full thickness where it is faint, which reads as a smear; a taper reads as a wake.

ROUND CAPS AND JOINS: a round cap on a 5 px segment overlaps its neighbour by half a width, so the joins fill instead of leaving the pale notch a flat cap leaves.

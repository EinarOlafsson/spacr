# Notes from `spacr/qt/widgets/ambient.py`

Prose lifted out of `spacr/qt/widgets/ambient.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (23 entries)
- [screen_pixels](#screen_pixels) (1 entry)
- [_as_color](#_as_color) (1 entry)
- [_theme_background](#_theme_background) (1 entry)
- [AmbientEngine.__init__](#ambientengine__init__) (1 entry)
- [BlobsEngine._configure](#blobsengine_configure) (3 entries)
- [AuroraEngine._configure](#auroraengine_configure) (2 entries)
- [AuroraEngine._resize](#auroraengine_resize) (1 entry)
- [AuroraEngine.curtain_color](#auroraenginecurtain_color) (1 entry)
- [AuroraEngine.ray_lengths](#auroraengineray_lengths) (1 entry)
- [AuroraEngine._tile](#auroraengine_tile) (7 entries)
- [AuroraEngine._mask](#auroraengine_mask) (2 entries)
- [AuroraEngine._paint_field](#auroraengine_paint_field) (3 entries)
- [RippleEngine._configure](#rippleengine_configure) (1 entry)
- [RippleEngine.geometry](#rippleenginegeometry) (1 entry)
- [DriftEngine._configure](#driftengine_configure) (1 entry)
- [DriftEngine.geometry](#driftenginegeometry) (1 entry)
- [DriftEngine._pen](#driftengine_pen) (1 entry)
- [DriftEngine.paint](#driftenginepaint) (1 entry)
- [BokehEngine._configure](#bokehengine_configure) (1 entry)
- [BokehEngine._paint_field](#bokehengine_paint_field) (1 entry)
- [CellsEngine._paint_field](#cellsengine_paint_field) (3 entries)
- [FractalEngine._configure](#fractalengine_configure) (1 entry)
- [FractalEngine.advance](#fractalengineadvance) (1 entry)
- [FractalEngine.view](#fractalengineview) (2 entries)
- [FractalEngine.buds](#fractalenginebuds) (3 entries)
- [FractalEngine._paint_field](#fractalengine_paint_field) (6 entries)
- [_FrameProducer._run](#_frameproducer_run) (1 entry)
- [AmbientWidget.__init__](#ambientwidget__init__) (7 entries)
- [AmbientWidget._rebuild_engine](#ambientwidget_rebuild_engine) (1 entry)
- [AmbientWidget.hideEvent](#ambientwidgethideevent) (1 entry)
- [AmbientWidget.eventFilter](#ambientwidgeteventfilter) (2 entries)
- [AmbientWidget._on_tick](#ambientwidget_on_tick) (2 entries)
- [AmbientWidget._backdrop_origin](#ambientwidget_backdrop_origin) (1 entry)
- [AmbientWidget.paintEvent](#ambientwidgetpaintevent) (2 entries)
- [_the_heavy_import_lock_is_free](#_the_heavy_import_lock_is_free) (1 entry)
- [_the_backdrop_wants_a_retry](#_the_backdrop_wants_a_retry) (1 entry)
- [_the_spaceout_fractal](#_the_spaceout_fractal) (3 entries)
- [install_ambient](#install_ambient) (1 entry)

## Module level

### lines 278-280

```python
SPACEOUT_THEME: "Fractals",
```

Never shown: nothing builds a menu row from a name that is not in `AMBIENT_THEMES`. Present so that a log line, a traceback or a test naming this engine reads like the other six.

### lines 339-340  _(unsure)_

```python
("#A3A3A3", "#CFCFCF", "#707070"),
```

Neutral greys, not slate: a set named Monochrome that carries a blue cast is a set that lies about what it is.

### lines 350-352

```python
"borealis": PaletteSpec(
```

Not invented: these are the emission lines the sky actually radiates, converted to sRGB. Ordered by the role the aurora engine gives them — main, high, fringe, blend — see AURORA_RAMP.

### lines 361-366

```python
"rainbow": PaletteSpec(
```

The spaceout palette. Seven stops right around the wheel and back to where they started, because the fractal reads it as a RING: the escape bands cycle through it repeatedly across one frame, so the last colour sits next to the first and a set that did not close would show a seam at every band boundary. Offered by no theme a menu lists — see `_THEME_PALETTES`.

### lines 373-374  _(unsure)_

```python
"fluor": PaletteSpec(
```

Also not invented: the three filter cubes on essentially every fluorescence scope, at the wavelength the eyepiece sees.

### lines 413-415

```python
SPACEOUT_THEME: (SPACEOUT_PALETTE,),
```

One palette, and it is offered by nothing else: `palettes_for` is what the Preferences palette picker is filled from, and it is only ever asked about a theme from `AMBIENT_THEMES`.

### lines 660-691

```python
RESOLUTION_RANGE = (0.25, 2.0)
```

The user controls: resolution, blur, speed, size and density

All of them are *multipliers on what the theme already does*, never absolute pixels or seconds. 1.0 is the shipped animation, exactly — every engine is written so that multiplying by 1.0 is the identity, and the tests assert the default frame is byte-for-byte the frame from before these existed. A multiplier is also the only formulation that means the same thing in every theme: "twice as big" is meaningful for a blob radius, a curtain, a ripple wavelength and a 2 px star, where "40 px" is meaningful for none of them.

Resolution and blur used to be ONE control, and that was the bug this pair replaces. Blur was implemented *as* the buffer resolution — softer meant shading fewer pixels and stretching them further — so "sharper" and "less blocky" were the same slider, and a sharp *soft* backdrop could not be asked for at all. They are two different questions and they now have two different answers:

resolution  how many pixels the scene is shaded into. Decides how much of the geometry survives: where the aurora's lower edge falls between two pixels, how wide each ray of its comb is. blur        how much of what was shaded is then thrown away, by an area average over the finished buffer. Decides how soft it looks, and nothing else.

The order matters and is the whole point. Shading at a low resolution *point-samples* the geometry: the fold's position is quantised, the ray comb aliases against the buffer grid, and no amount of subsequent blurring puts back what was never computed. Shading high and averaging down *prefilters* it: the same softness, with every edge still where the model put it. That is the difference between "soft" and "blocky", and it is why a picture can now be both sharp and soft.

### line 1500, trailing  _(unsure)_

```python
BLOB_LARGE_RADIUS = (0.22, 0.46)
```

fraction of the short edge

### line 1507, trailing  _(unsure)_

```python
BLOB_DRIFT_PERIOD = (24.0, 70.0)
```

seconds

### line 1511, trailing  _(unsure)_

```python
BLOB_PULSE_PERIOD = (7.0, 19.0)
```

seconds

### lines 1647-1700

```python
AURORA_CURTAINS = 3
```

aurora

What the real thing does, and what each part of it costs here.

An aurora is not a band of colour sliding across the sky. It is a *folded sheet seen edge-on*: charged particles spiral down the geomagnetic field lines and light the atmosphere along them, so the sheet is made of vertical rays, and the arc is where that sheet crosses the sky. Four things follow, and all four are what makes it recognisable:

1. *Vertical ray striations.* The rays are the defining feature. A smooth band with no rays reads as a gradient, which is what this theme was before. 2. *The folds travel ALONG the arc.* The sheet ripples the way a ribbon held at one end does — the fold pattern propagates lengthwise while the arc itself stays where it is. It does not slide sideways as a body. Every wave below is written ``sin(2*pi*(u - v*t)/lambda)``: a phase that depends on position and travels at ``v``, which is a travelling wave and nothing else. 3. *Several frequencies at once.* One slow, long, deep fold with faster small ripples riding on it. A single sine is a flag, not an aurora. 4. *Brightness surges* running along the arc on their own schedule, faster than the folds and on a different wavelength, so the two are visibly independent phenomena rather than one driving the other.

Then the vertical colour structure, which is pure atomic physics and the cheapest realism available. 557.7 nm atomic oxygen (green) through the body, 630.0 nm atomic oxygen (red) at the top where the air is thin enough for that slow transition to survive the wait, and 427.8 nm ionised nitrogen (blue-violet) along the bottom. The lower edge is *sharp* — it is where the particles finally run out of altitude — and the top is diffuse.

The single most useful thing to know about that colour structure is that it is a function of ALTITUDE, not of position within the curtain. The ramp is therefore anchored to the frame and not to the folded edge, and everything falls out of it for free:

where the fold dips low, the sheet's edge reaches into the violet, and where it rises, it does not — which is exactly what photographs show; the curtain looks *taller* where the fold dips, because its top is at a fixed altitude and only its bottom moved. Real rays behave that way for the same reason, and it costs nothing to reproduce; one affine brush transform per curtain is correct, so the curtain is one filled path and not a strip of separately anchored pieces. Anchoring per piece was tried first: it puts a step in the ramp at every seam, and at these alphas the step is ~13/255 — visible as vertical banding.

How it is drawn, and why. Measured at 1920x1080 in the 256 px buffer: one ``drawImage`` per ray costs ~5 us of Python-to-Qt overhead, so 150 rays is 0.8 ms — more than this theme is allowed in total. Instead each curtain is ONE filled path (the folded sheet) painted with a *tiled texture brush* whose tile is the ray comb crossed with the colour ramp, and then filled a second time with a small per-frame image holding the surge. The ray count costs nothing because the rays are the brush rather than the geometry.

### line 1774, trailing  _(unsure)_

```python
AURORA_DRIFT_PERIOD = (30.0, 90.0)
```

seconds

### line 1775, trailing  _(unsure)_

```python
AURORA_HUE_PERIOD = (18.0, 46.0)
```

seconds per colour cross-fade cycle

### line 2519, trailing  _(unsure)_

```python
RIPPLE_PERIOD = (14.0, 26.0)
```

seconds for a ring to cross its reach

### line 2520, trailing  _(unsure)_

```python
RIPPLE_REACH = (0.55, 0.95)
```

fraction of half the canvas diagonal

### line 2649, trailing  _(unsure)_

```python
DRIFT_AREA_PER_PARTICLE = 9500
```

pixels of canvas per particle

### line 2683, trailing  _(unsure)_

```python
DRIFT_HALO_SPREAD = 1.6
```

extra diameter per unit of blur above 1

### line 2684, trailing  _(unsure)_

```python
DRIFT_HALO_ALPHA = 0.34
```

halo alpha as a share of the dot's own

### lines 2968-2985

```python
BOKEH_COUNT = 11
```

bokeh

What an epifluorescence field looks like off the focal plane, which is a state every user of this app has spent hours staring at. A point source out of focus does not become a Gaussian smudge: it becomes an image of the aperture — a disc, brighter at its rim than in its middle, with a hard-ish edge. That inversion is the whole reason bokeh is recognisable and is the whole reason this is not "blobs with different numbers": a blob is a Gaussian, brightest in the centre and gone by the edge.

Two things follow and both are cheap:

1. *Focus varies per disc.* A field has depth, so some sources are nearly in focus (small, tight, bright rim) and some are far out (large, flat, faint). One radial gradient expresses both — see :meth:`_stops`. 2. *They overlap and add.* Additive compositing over a dark page is literally correct here rather than merely convenient: two out-of-focus emitters really do sum.

### lines 3168-3180

```python
CELL_COUNT = 9
```

cells

The other thing this app's users look at all day. A cell in a widefield image is three concentric statements, not one: a soft cytoplasmic body, a slightly brighter membrane where the edge is seen nearly edge-on, and a distinctly brighter nucleus sitting off-centre. Draw those three and the shape reads as a cell at any size; draw only the first and it is a blob.

They are ellipses rather than circles, they are not all pointing the same way, and they turn as they drift — slowly, because this sits behind a settings form. The rotation is a painter transform per cell (nine of them a frame, in a 480x270 buffer), which measures free next to the two gradient fills each one already costs.

### lines 3363-3365

```python
FRACTAL_BUFFER_EDGE = 448
```

fractal

The spaceout backdrop. Not one of :data:`AMBIENT_THEMES` and not reachable from Preferences — see :data:`SPACEOUT_THEME`.

### lines 3508-3522

```python
FRACTAL_STATE_OCTAVES: Tuple[Tuple[float, float], ...] = (
```

the fractal is alive: states, a beat, buds and a vortex

Everything below is what separates a living pattern from a screensaver, and the distinction is a real one: a loop that always does the same thing at the same rate is learnable in about a minute, and once it is learned it stops being looked at. So the engine has STATES it moves between — deep and flat, busy and quiet, fast and slow, crowded and empty — on its own, gradually, and in an order that does not repeat.

ALL OF IT IS A PURE FUNCTION OF (SEED, TIME). Not an aesthetic preference: `shade` runs on `_FrameProducer`, a test renders the same clock twice and compares byte for byte, and `set_time` carries the clock across a theme change. A state machine that stepped itself per frame would break all three. The wander below is therefore hashed noise indexed by a cell number rather than a random walk, and the beat's phase is an integral in closed form rather than an accumulator.

### lines 3740-3744

```python
FRACTAL_FRAME_SHARE = 0.09
```

the guard

`WORK_BUDGET` bounds what the USER can ask for, in multiples of what the theme costs at its own defaults. It cannot answer the other question — what this machine can afford — because it knows nothing about the machine. This does: it measures the shading pass and trims the engine until it fits.

## screen_pixels

### lines 655-656  _(unsure)_

```python
return pixels if pixels >= BUFFER_MIN_EDGE ** 2 else BUFFER_MAX_PIXELS
```

A screen smaller than the fallback is still a real ceiling; a nonsensical one (a headless plugin reporting nothing) is not.

## _as_color

### lines 826-831

```python
def _as_color(value: Union[QColor, str, None], fallback: QColor) -> QColor:
```

Small colour helpers

Deliberately local rather than imported from ``dna_rain``: this module is installed on *every* module screen, and it should not drag the sequencing backdrop in behind it just to reuse fifteen lines of arithmetic.

## _theme_background

### lines 942-944

```python
return QColor(page_colour("dark"))
```

`page_colour` is imported at module scope, so this arm cannot itself fail on an import the way a second `from ..theme import` could — and a backdrop must never raise on its way to a screen.

## AmbientEngine.__init__

### line 1037  _(unsure)_

```python
self.blur = _clamp(blur, *BLUR_RANGE)
```

Before ``_configure``: a subclass may size something from them.

## BlobsEngine._configure

### lines 1558-1560

```python
"""Roll this theme's constants from the seed.
```

Seed the blobs on a jittered 5x3 grid rather than uniformly at random: with only fourteen of them, uniform sampling reliably leaves one corner empty and clumps three in the middle.

### lines 1570-1573

```python
for i in range(_pool_size(BLOB_COUNT)):
```

The pool is rolled for the top of the density range and then a prefix of it is painted. Extending the loop is the one way to add them that leaves the first BLOB_COUNT draws bit-identical: the RNG is consumed in order, so blob 3 gets the numbers it always got.

### lines 1592-1594

```python
color=i,
```

Straight round-robin, not a random pick: with three colours and fourteen blobs a random assignment leaves a palette colour missing about one run in fifty.

## AuroraEngine._configure

### lines 2015-2017

```python
base = (AURORA_BASE[i % len(AURORA_BASE)]
```

Each extra tier of three sits a little lower than the last, or a dense aurora would be three curtains painted on top of each other inside one jitter's width rather than a deeper one.

### lines 2029-2030  _(unsure)_

```python
fold_phase=tuple(rng.uniform(0.0, 2 * math.pi)
```

Every wave gets its own phase, or the three curtains fold in lockstep and the depth illusion collapses.

## AuroraEngine._resize

### line 2045  _(unsure)_

```python
"""Re-lay the bands for a new widget size."""
```

Both caches are keyed on pixel sizes derived from it.

## AuroraEngine.curtain_color

### lines 2181-2183

```python
wander = colors[1 + curtain.color % (len(colors) - 1)] \
```

Never index 0: a curtain whose wander target is its own body colour does not shimmer at all, which is what happened to the third one on every three-colour palette.

## AuroraEngine.ray_lengths

### lines 2228-2232

```python
unit = 0.5 * (1.0 + math.sin(angle))
```

Quantise the UNIT and then map, not the mapped value. The other way round quantises [low, high] against a 0..1 grid, so only the top of the range survives -- measured, it gave four levels spanning 0.80..0.98 out of an intended 0.55..0.98, and the breathing was a fifth of the depth it should have been.

## AuroraEngine._tile

### lines 2258-2259

```python
self._tiles = {}
```

Only a long run of resizes can get here. Start again rather than grow without bound.

### lines 2271-2272  _(unsure)_

```python
gradient = QLinearGradient(0.0, float(ramp_bottom), 0.0,
```

Ramp position 0 is the curtain's lower edge, which is the *bottom* of the ramp rows, so the gradient runs upward through the image.

### lines 2279-2281

```python
inner.setCompositionMode(QPainter.CompositionMode_DestinationIn)
```

... then cut the ray comb out of it. DestinationIn keeps the colour and replaces the alpha, which is one pass over 1 920 pixels, done about three dozen times in the life of the widget.

### lines 2295-2302

```python
for (centre, half, _strength), length in zip(AURORA_TILE_RAYS,
```

Each ray cut to its own length. Done AFTER the comb, still in DestinationIn, so it takes alpha away from one ray's band without touching its neighbours or the sheet between them.

Cut from the TOP: an aurora ray is anchored at the lower edge and reaches upward, so a ray that is breathing shortens away from its tip. Shortening from the bottom would lift it off the curtain's edge and look like it is floating.

### lines 2311-2312  _(unsure)_

```python
kept = int(round((ramp_bottom - ramp_top) * length))
```

`ramp_bottom` is the curtain's lower edge and `ramp_top` its tip, so the kept part runs upward from the bottom.

### lines 2315-2317

```python
feather = max(1, int(round(
```

The tip fades over a fixed share of the FULL ray length, so a short ray and a long one taper alike rather than the short one being all taper.

### lines 2322-2330

```python
fade = QLinearGradient(0.0, float(max(0, cut_bottom - feather)),
```

FEATHERED, not a hard rectangle. A square cut gives a shortened ray a flat tip, which is both wrong -- a real ray fades out at the top -- and measurable: it sharpened the curtain's upper edge until the lower-edge-to-upper-edge contrast fell from 2.5x to 2.1x and `test_the_lower_edge_is_sharp_and_the_top_is_diffuse` caught it. The asymmetry between the two edges is as recognisable as the colour, so it is not something to trade away for a cheaper fill.

## AuroraEngine._mask

### line 2355, trailing  _(unsure)_

```python
edge = pad / band
```

the curtain's lower edge

### line 2361

```python
fade = QLinearGradient(0.0, float(height), 0.0, 0.0)
```

Stops run bottom-to-top, so 0.0 is the bottom of the image.

## AuroraEngine._paint_field

### lines 2442-2444

```python
painter.setRenderHint(QPainter.Antialiasing, True)
```

The fold is a near-horizontal edge in a buffer that is about to be stretched sevenfold. Without antialiasing it upscales as a visible staircase; with it, it costs about 0.03 ms.

### line 2471  _(unsure)_

```python
brush.setTransform(QTransform.fromTranslate(
```

Translation only — see AURORA_TILE_RAMP for why that matters.

### lines 2477-2482

```python
left, right = columns[0][0], columns[-1][0]
```

The surge, over its own shorter path: it is transparent above AURORA_PULSE_HEIGHT and the rest of the sheet is not worth compositing nothing onto. It is a texture brush rather than a blit so that the path clips it — a surge that spilled past the sheet's lower edge would soften the one edge that has to stay hard.

## RippleEngine._configure

### lines 2558-2562

```python
"""Roll this theme's constants from the seed.
```

The first three are the shipped anchors. The rest exist for the density control and are placed on a jittered ring around the middle: reusing the same three with a wider jitter puts two sources close enough that their rings arrive together, which reads as one source with a doubled amplitude rather than as two.

## RippleEngine.geometry

### lines 2607-2609

```python
reach = source.reach * half_diagonal * self.size
```

The size setting is the ripple's *wavelength*: the rings of one source are evenly spaced across its reach, so stretching the reach stretches the spacing between them by the same factor.

## DriftEngine._configure

### lines 2772-2776

```python
for i in range(DRIFT_POOL):
```

The draw order here is load-bearing and is the reason this reads oddly. The shipped pool and the shipped twinkle came off this RNG in this order, and the frame the tests hold this engine to is the one those exact numbers produce. Everything density and direction added is drawn *after* both, so the shipped starfield is untouched.

## DriftEngine.geometry

### line 2893

```python
y = particle.y + sign * particle.speed * t
```

Wrapped: the field never runs out.

## DriftEngine._pen

### lines 2926-2927  _(unsure)_

```python
pen.setWidthF(self.halo_size(layer) if halo
```

A round-capped pen makes drawPoints draw filled circles, which is how a batch of dots gets drawn in one call.

## DriftEngine.paint

### lines 2957-2958

```python
if self.blur > 0.0:
```

The halo goes down first, so the crisp core sits on top of it rather than being washed out by it.

## BokehEngine._configure

### lines 3075-3077

```python
near = 1.0 - (radius - lo) / max(1e-6, hi - lo)
```

Focus falls with size. An aperture image is large exactly because it is far out of focus, so a big disc with a knife edge on it is not a defocused anything — it is a ring.

## BokehEngine._paint_field

### lines 3140-3143

```python
"""Draw one frame's field of shapes.
```

A bokeh disc has an edge — that is what makes it a disc and not a blob — and an edge in a buffer that is about to be stretched fourfold upscales as a staircase without this. Same trade the aurora makes for its fold, and the same ~0.03 ms.

## CellsEngine._paint_field

### line 3308  _(unsure)_

```python
"""Draw one frame's field of shapes.
```

The membrane is an edge; see BokehEngine._paint_field.

### lines 3325-3329

```python
body = QRadialGradient(0.0, 0.0, major)
```

Body plus membrane: one gradient, because the membrane is a brightening of the body's own falloff and not a stroked outline. A stroked one is line work — see the module docstring for what that costs — and it also looks drawn rather than imaged.

### lines 3338-3340

```python
painter.save()
```

The ellipse is the circle the gradient was built for, squashed on one axis — so the gradient squashes with it and the membrane stays on the edge all the way round.

## FractalEngine._configure

### lines 3860-3862

```python
"""Roll this theme's constants from the seed.
```

Independent phases and a direction, so two backdrops built with different seeds are not in lockstep — the same reason every other engine rolls its periods rather than sharing one clock.

## FractalEngine.advance

### lines 3959-3961

```python
over = max(spent) / max(0.001, self.frame_budget())
```

The worst of the batch, not the mean: the guard is answering

"can this machine keep up", and one long pass a second is a dropped frame however cheap the others were.

## FractalEngine.view

### lines 4020-4021

```python
breath *= 1.0 - FRACTAL_BEAT_ZOOM * self.beat()
```

The beat is a squeeze on the whole form, so it pulses rather than drifting evenly.

### lines 4023-4024  _(unsure)_

```python
span = (_lerp((FRACTAL_SPAN, FRACTAL_SPAN_DEEP), deep) * breath
```

The size control is a *zoom*: bigger elements means fewer units of the plane across the canvas, so it divides the span.

## FractalEngine.buds

### lines 4050-4053

```python
beta_re, beta_im = self.fixed_point()
```

A BUD IS A BUBBLE OF THE SAME VORTEX. It carries its parent's constant and looks at the same fixed point, so what is inside it is the parent's own field at the bud's scale rather than a second picture that happens to be nearby.

### lines 4069-4071

```python
heading = 2.0 * math.pi * _hash01(self._salt, 400 + slot, number)
```

It LEAVES the rim: at birth the centre is on the parent's own rim and it drifts outward from there, so the bud is seen to separate rather than to appear somewhere else.

### lines 4076-4077  _(unsure)_

```python
bulge = math.sin(math.pi * age) ** 0.65
```

And it swells and goes: nothing at the moment of budding, widest in the middle of its life, nothing again at the end.

## FractalEngine._paint_field

### lines 4325-4326  _(unsure)_

```python
np.subtract(soft_top, magnitude, out=share)
```

A FRACTION of an iteration, not a yes/no — see

FRACTAL_SOFT_LIMIT.

### lines 4337-4343

```python
np.clip(zr, -reach, reach, out=zr)
```

An escaped orbit doubles its exponent every iteration and would reach infinity in six, and then NaN on the first `zr2 - zi2` that subtracts one infinity from another — which would poison the clip above and take the pixel with it. Bounding z is two passes a frame and is the price of a smooth field; the alternative, counting escapes as integers, is free and is what the visible banding of a fractal is made of.

### lines 4347-4349

```python
interior = survived >= np.float32(iterations - 0.5)
```

Inside the set: never escaped, so every iteration counted. Taken before the wrap, because after it the interior is just another point on the ring.

### lines 4354-4356

```python
survived += bands
```

The tunnel: rings spaced on the log of the radius, travelling down it. Added in index units, so it moves the colour without touching what the field is.

### lines 4361-4363

```python
painter.drawImage(0, 0, QImage(frame.data, width, height,
```

The QImage borrows `frame`'s memory rather than copying it, so the array has to outlive the blit — it does, by being a local here, and `drawImage` is synchronous.

### lines 4367-4368

```python
self._spent.append((time.perf_counter() - started) * 1000.0)
```

What it cost, for `advance` to act on. Measured rather than modelled: the guard is answering a question about the machine.

## _FrameProducer._run

### lines 4624-4637

```python
self._stop.wait(remaining if remaining > 0 else 0.0)
```

Sleeping on the stop event rather than time.sleep is what makes stop() return immediately instead of at the end of the beat.

A pass that overran the beat gets no sleep at all, which is how this thread degrades: it keeps shading as fast as it can and the GUI thread repeats whatever the last finished frame was. That cannot make the machine worse, and the reason is arithmetic rather than good intentions: the no-sleep branch is only reached when one pass already took longer than the interval, so the rate is 1/pass and therefore *below* the cap by construction. Driven and counted at 1080p on ``cells``, cap 24: 23.5 passes a second idle, 15.4 with one Python worker, 0.7 with three. The total shading work is what the GUI thread used to do, on a thread nobody is waiting for.

## AmbientWidget.__init__

### lines 4727-4731

```python
self._engine_lock = threading.RLock()
```

the shading thread

First, before anything that could reach a setter: every one of them takes the lock, so a construction order that reached one early would raise AttributeError rather than race.

### line 4753  _(unsure)_

```python
theme, palette = dressed(theme, palette)
```

What is asked for, then what is actually worn — see `dressed`.

### lines 4758-4760

```python
asked = (blur, speed, size, resolution, density, direction)
```

Unset means "whatever the user asked for in Preferences", so a screen built after a settings change comes up already correct instead of waiting for the next apply_ambient_preferences().

### lines 4776-4779

```python
self._background_explicit = background is not None
```

Remember whether the caller *chose* the colour. If they did, a later application palette change is theirs to react to; if they did not, this widget follows the theme itself rather than leaving a black rectangle on a white page.

### line 4784

```python
self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
```

Never in front of, never in the way of, the real content.

### lines 4807-4808

```python
self._timer = QTimer(self)
```

One timer for the life of the widget. Switching theme swaps the engine underneath it and never creates a second one.

### line 4448

```python
self._watched: Optional[weakref.ReferenceType] = None
```

WEAK, NOT A PLAIN REFERENCE. The window owns this widget's wrapper, or is this widget when it has no parent, so a strong `_watched` made a reference cycle and the pair could be freed only by Python's cycle collector. The collector clears each wrapper's `__dict__` before the C++ objects are destroyed; the window's destructor then hides its children and notifies its filters, and both reached a widget with no `_watched` and no `_timer`. CI run 34989909231 (2a84d1d60) failed on exactly that, once per widget, in whichever test the collector happened to run. With a weak reference no cycle forms, and reference counting frees the window while every attribute is still there. Pinned by `tests/qt/test_a_backdrop_outlives_nothing_it_watches.py`.

## AmbientWidget._rebuild_engine

### lines 4947-4948

```python
self._last_frame = None
```

The old engine's last frame was drawn by the old engine; keeping it would blit the theme the user just switched away from.

## AmbientWidget.hideEvent

### line 4904

```python
if getattr(self, "_timer", None) is not None:
```

A window torn down after the collector emptied this wrapper still hides it. The weak watch removes the cycle this widget made, but any other cycle through the widget reopens the same path, so with no `_timer` there is nothing to stop and the event is simply passed on.

## AmbientWidget.eventFilter

### lines 5285-5287

```python
self._follow_screen()
```

The window was dragged, possibly onto another display — see

`_follow_screen`. A move that stays on one screen finds the ceiling unchanged and returns without touching the engine.

### line 4911

```python
ref = getattr(self, "_watched", None)
```

`getattr`, for the same teardown as `hideEvent`: the window's destructor notifies this filter after the collector cleared the wrapper. With no watch there is nothing to pause or resume. The same guard FigureQueue got for its `_view`.

## AmbientWidget._on_tick

### lines 5317-5345

```python
dt = self._clock.restart() / 1000.0
```

NO POPUP HOLD HERE, AND THE REASON IS A MEASUREMENT (385).

This tick used to return early while a menu or a tooltip was up, borrowed from the fix for the dock flicker. That fix's own diagnosis names its subject exactly: "a popup composited over the NATIVE GL backdrop". This widget has no GL surface and no native window -- it is a QWidget painting with QPainter -- so the mechanism it was defending against is not one this widget has.

And the burst it was counting is not the popup's. Widget repaints over 1.2 s with an ambient backdrop running behind forty labels and twelve buttons, offscreen:

menu open,  hold on      2      (the animation is stopped) menu open,  hold off  1,592 NO menu,    hold on   1,590 NO menu,    hold off  1,590

The last two lines are the finding. A moving backdrop repaints everything above it whether or not a popup is on screen, so holding for the popup was not removing a burst the popup caused it was stopping the animation, which stops the burst that was there the whole time.

The cost of that was the whole of 385: opening Preferences or the Help menu froze the theme, and Preferences is where the theme's own controls live, so a user changing the speed could not see the change they were making. The GL path in `fractal_travel` keeps its hold, because that is where the flicker was reported.

### lines 5349-5357

```python
advance_spaceout_drift(step)
```

THE PALETTE'S DRIFT RIDES THIS TICK. It is process state in

`theme` and something has to move it; a wall clock read inside `palette_for` would make the palette a different value on two calls in the same frame, and `palette_for` is on the path of every stylesheet build and every widget that paints. A backdrop already painting frames is the honest driver — and a user who turned the animation off asked for zero frames and gets a still palette to go with them, which is the same bargain the rest of this widget makes. Costs a comparison on an ordinary start.

## AmbientWidget._backdrop_origin

### line 5417, trailing

```python
window = self.window()
```

never None; the widget itself if top level

## AmbientWidget.paintEvent

### lines 5457-5459

```python
painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
```

Clip BEFORE the base fill, so the flat page colour is rounded too. Anti-aliased, because a setMask() region would give the corners hard stair-steps against the card's smooth rim.

### lines 5474-5499

```python
whole = event.rect().contains(rect)
```

A PARTIAL REPAINT MUST NOT ADVANCE THE ANIMATION, and this is the line that decides it. Qt clips this paint to `event.rect()`, so a repaint asked for by something on top of the backdrop -- a hovered settings category writing the hint strip, a scrollbar appearing, a console line -- redraws a BAND and leaves every pixel outside it holding the frame that was blitted last time. Taking the newest frame for that band paints it one animation step ahead of its own surroundings, and on a 3840x2160 screen with the blobs backdrop that is a rectangle of mismatched backdrop appearing and vanishing the "random flickers", and the flicker under the strips and the settings categories.

MEASURED on the mask screen at font_scale 2, offscreen, counting every paint this widget received: three seconds of moving the pointer across eight category headers gave 368 backdrop repaints, 329 of them partial, and 36 of those partial ones swapped in a newer frame -- twelve torn bands a second. Six expand/collapse clicks gave four more. Idle, all 37 repaints were full-widget and every swap was legitimate.

`latest()` PEEKS at the producer's slot rather than draining it, so a frame passed over here is still there for the next full repaint; the timer's own repaint is always full-widget. The cost of this rule is that a band redrawn between two ticks shows backdrop that is up to one frame old, which is exactly what the pixels beside it are showing.

## _the_heavy_import_lock_is_free

### lines 5567-5577

```python
if "spacr.qt.widgets.fractal_travel" not in sys.modules:
```

NOT IMPORTED MEANS NOT LOCKED, and asking is what used to cost.

`from .fractal_travel import _heavy_import_lock` IMPORTS the module if nothing has yet, and that module pulls numba: measured at 0.44 s in a cold interpreter, spent on the GUI thread inside something documented as "the cheap half" of the pair.

It is also unnecessary. The lock lives in that module, so nothing can be holding it while the module has never been imported -- the only code that takes it is code that had to import it first. Answering from sys.modules is exact here, not an approximation.

## _the_backdrop_wants_a_retry

### lines 5612-5613  _(unsure)_

```python
return False
```

No class to compare against means no backdrop module, which means nothing could have raised it.

## _the_spaceout_fractal

### lines 5635-5639

```python
_retire_fractals_on(host)
```

ONE BACKDROP PER HOST. `install_ambient` is called again whenever a screen is rebuilt, and the previous fractal was left parented and RUNNING: four live canvases, four vispy timers and four render threads were on screen at once, which is what filled the console with "Internal C++ object already deleted" and ended in a core dump.

### lines 5652-5655

```python
RuntimeControls(speed=values["speed"], dream=values["dream"],
```

EVERY SAVED CONTROL, not most of them. The three pointer settings were collected, stored and never passed, so Mouse gravity could not be turned off: `RuntimeControls` defaults it to on, and nothing here ever said otherwise.

### lines 5673-5678

```python
raise
```

NOT A FAILURE, and so not something to log an exception for or to answer with `None`. `None` means "this launch has no fractal" and the caller then installs the ordinary ambient engine instead which under spaceout is the wrong artwork, kept for good, because a heavy import happened to be running at the moment a screen was opened. Raising says "not yet"; see `_the_backdrop_wants_a_retry`.

## install_ambient

### lines 5753-5758

```python
replacement = _the_spaceout_fractal(host)
```

SPACEOUT DRAWS THE OTHER FRACTAL (instruction 260). Hooked HERE rather than at the call sites because there are three of them -- the module screens, the Home screen and the setup slides -- and hooking one left Home showing the old Julia set, which is what the maintainer saw. `dressed()` below would otherwise swap the theme to SPACEOUT_THEME, which IS the old artwork.

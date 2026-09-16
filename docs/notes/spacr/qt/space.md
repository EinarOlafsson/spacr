# Notes from `spacr/qt/space.py`

Prose lifted out of `spacr/qt/space.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (20 entries)
- [sample_star_fluxes](#sample_star_fluxes) (1 entry)
- [starfield](#starfield) (2 entries)
- [galaxy](#galaxy) (8 entries)
- [sun](#sun) (4 entries)
- [_nebula](#_nebula) (1 entry)
- [_compress_highlights](#_compress_highlights) (1 entry)
- [_enforce_legibility](#_enforce_legibility) (1 entry)
- [render](#render) (2 entries)
- [to_qimage](#to_qimage) (1 entry)
- [screen_size](#screen_size) (1 entry)
- [download_nasa_background](#download_nasa_background) (2 entries)

## Module level

### lines 260-262  _(unsure)_

```python
_BB_T = np.array(
```

Stellar colour — blackbody locus, sRGB

### line 271, trailing

```python
[255, 137, 18],
```

2000 K — deep orange M

### line 272, trailing

```python
[255, 180, 107],
```

3000 K — orange K/M

### line 273, trailing

```python
[255, 209, 163],
```

4000 K — warm white K

### line 274, trailing

```python
[255, 228, 206],
```

5000 K — G

### line 275, trailing

```python
[255, 244, 242],
```

6000 K — sun-white

### line 276, trailing

```python
[245, 243, 255],
```

7000 K — F

### line 277, trailing

```python
[227, 233, 255],
```

8000 K — A

### line 278, trailing

```python
[201, 215, 255],
```

10000 K — hot A

### line 279, trailing

```python
[191, 207, 255],
```

12000 K — B

### line 280, trailing

```python
[175, 195, 255],
```

20000 K — hot B

### line 281, trailing

```python
[168, 189, 255],
```

40000 K — O

### line 288, trailing  _(unsure)_

```python
[15000., 33000.],
```

O/B

### line 289, trailing  _(unsure)_

```python
[7500., 10000.],
```

A

### line 290, trailing  _(unsure)_

```python
[6000., 7500.],
```

F

### line 291, trailing  _(unsure)_

```python
[5200., 6000.],
```

G

### line 292, trailing  _(unsure)_

```python
[3700., 5200.],
```

K

### line 293, trailing  _(unsure)_

```python
[2400., 3700.],
```

M

### lines 363-365

```python
STAR_CORE_HDR = 1.0
```

HDR star pixels above this level are point-source cores. They are restored after the final whole-frame legibility solve: dozens of isolated pixels do not move a text-window mean, while dimming them turns the sky uniformly grey.

### lines 1177-1179  _(unsure)_

```python
NASA_IMAGES = (
```

Optional real imagery — NASA / ESA public domain

## sample_star_fluxes

### line 345

```python
u = np.clip(u, np.finfo(np.float64).tiny, 1.0)
```

u == 0 would divide to infinity; nextafter keeps it finite.

## starfield

### lines 464-466

```python
scale = max(0.6, min(width, height) / 1400.0)
```

Brightness -> visual size. Real point sources all have the same PSF; what makes a bright star look bigger is the wings clipping above threshold, which a gentle power law imitates cheaply.

### lines 469-470  _(unsure)_

```python
amp = 8.0 * (flux / FLUX_SATURATION) ** 0.8
```

HDR amplitude spanning ~80x from the faintest star to a saturated core, so the tone map has something to clip to white.

## galaxy

### line 529  _(unsure)_

```python
arm_theta = np.log(r + 1e-4) / pitch
```

Spiral phase: distance (in angle) to the nearest arm ridge.

### line 533, trailing  _(unsure)_

```python
d = np.abs(phase - half)
```

0 at the ridge, `half` between arms

### line 535  _(unsure)_

```python
width_arm = 0.34 + 0.30 * np.exp(-r * 1.4)
```

Arms narrow at large radius, and fade out with the disc.

### lines 541-542  _(unsure)_

```python
d_dust = np.abs((phase - half * 0.55))
```

Dust lanes trail the arm ridge on the inner edge — a second, phase-shifted ridge that *removes* light.

### line 547  _(unsure)_

```python
bulge = np.exp(-(r / 0.19) ** 0.72).astype(np.float32)
```

Warm core bulge, Sérsic-ish, falling off into the disc.

### line 550  _(unsure)_

```python
knots = _value_noise(sh, sw, rng, octaves=4, base=6)
```

Clumpy HII knots along the arms.

### lines 557-558  _(unsure)_

```python
arm_col = np.array([0.44, 0.66, 1.00], dtype=np.float32)
```

Young blue arms, a cooler blue haze between them, and a warm old-population core — the colour gradient every spiral has.

### line 562, trailing  _(unsure)_

```python
knot_col = np.array([1.00, 0.52, 0.66], dtype=np.float32)
```

HII pink

## sun

### line 602  _(unsure)_

```python
mu = np.sqrt(np.clip(1.0 - (r / R) ** 2, 0.0, 1.0)).astype(np.float32)
```

µ = cos(angle from disc centre as seen from the star's centre).

### line 608

```python
gran = _value_noise(sh, sw, rng, octaves=3, base=10)
```

Granulation: convective cells, ±9 % on the photosphere only.

### lines 612-613  _(unsure)_

```python
outside = np.maximum(r - R, 0.0)
```

Corona — smooth exponential falloff outside the limb, plus a few radial streamers so it does not read as a plain glow.

### lines 617-618  _(unsure)_

```python
streamers = (1.0
```

Two harmonics at low amplitude. One strong harmonic gives the corona symmetric "ears"; two weak ones read as structure.

## _nebula

### lines 634-636  _(unsure)_

```python
def _nebula(width: int, height: int, seed: int) -> np.ndarray:
```

Nebula haze — ties the composition together

## _compress_highlights

### lines 867-869

```python
bent = foot + span * (1.0 - np.exp(-np.maximum(luma - foot, 0.0) / span))
```

Exponential shoulder: identity below `foot`, asymptotic to

`ceiling`, C1 at the join (both value and slope match), so a smooth gradient crossing it gains no Mach band.

## _enforce_legibility

### lines 932-934

```python
dimmed[mask] = arr[mask]
```

Restoring point sources adds a tiny amount back to the measured window. Re-solve the non-core pixels until the *composited* output satisfies the same guarantee; normally this takes one additional iteration.

## render

### lines 1013-1022

```python
small = _area_downsample(hdr, 8)
```

Mild bloom so bright things bleed the way a lens does. The downsample has to *average* — point-sampling a 1 px star into a 1/8-scale buffer and box-blurring it paints a visible 70 px square, which is how the first cut of this looked. Two blur passes then turn the box kernel into a tent so no hard edge survives the upsample.

Bloom counts as a smooth layer: it is a blur, it has no detail finer than ~100 px by construction, and a star's bloom really does cover a text window even though the star itself does not.

### lines 1028-1029  _(unsure)_

```python
exposure = tone_exposure(_luma(hdr))
```

Solved on the composed frame, before any ceiling, so the sky anchor sees exactly what it always saw.

## to_qimage

### line 1053, trailing  _(unsure)_

```python
return img.copy()
```

detach from the numpy buffer

## screen_size

### lines 1166-1170

```python
return (_clampi(w, MIN_BACKGROUND[0], MAX_DIM[0]),
```

Never smaller than MIN_BACKGROUND. The QSS centres the image without repeating it, so a background narrower than the window letterboxes into hard-edged bands of flat colour. That cannot happen when the screen is bigger than the window, but a virtual/offscreen display can report almost anything.

## download_nasa_background

### lines 1287-1289

```python
from PySide6.QtGui import QImage
```

Only accept it if Qt can actually decode it — a captive-portal HTML error page is 2 kB of "valid" bytes that would otherwise be installed as the wallpaper.

### lines 1295-1300

```python
from . import imagery
```

This file becomes the Space wallpaper *directly* — the stylesheet points at it, nothing renders it per screen — so it is the one path by which an unbounded picture could still get behind the app's text. A solar flare frame is exactly that. Solve it here or refuse it; the procedural sky is the fallback and it is bounded.

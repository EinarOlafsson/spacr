"""
Procedural deep-space imagery for the Space theme.

Why generate instead of ship a JPEG
-----------------------------------
Two things rule out bundling a photograph:

* a multi-megabyte image per aspect ratio bloats the wheel, and
* a fixed 1920x1080 asset is *soft* on a 5K display — the one place a
  space background actually needs to look good.

So the background is synthesised with numpy at the display's native
pixel size and cached to the user's config directory, keyed by
``(width, height, variant, seed, CACHE_VERSION)``. It is computed once
per screen size and reloaded from disk on every later launch.

Everything here is **seeded and deterministic**: the same
``(width, height, variant, seed)`` always produces byte-identical
pixels, which is what lets the test suite assert on the output.

Three generators, composed
--------------------------
``starfield``
    A magnitude distribution drawn from the Euclidean number-count law
    (``N(>F) ∝ F^-3/2``) — many faint stars, very few bright ones — with
    stellar colour sampled from a blackbody locus spanning hot blue O/B
    through white A/F/G to cool red K/M, and 4-way diffraction spikes on
    only the handful of brightest stars.

``galaxy``
    A two-arm logarithmic spiral (``r = a·e^(bθ)``) rendered as an
    analytic field, with dust lanes trailing the arms, a warm Sérsic-ish
    core bulge and an exponential disc falloff, projected at an
    inclination.

``sun``
    A star disc with classic linear limb darkening
    (``I(µ) = 1 − u(1 − µ)``), value-noise granulation on the
    photosphere, and a corona that falls off smoothly with radial
    streamers.

They are *composed* rather than offered as three separate wallpapers
because the user asked for "a galaxy stars a sun" — one sky containing
all three reads as a photograph of space; three separate wallpapers read
as three clip-art assets. ``variant`` re-weights the composition
(which element is the subject) rather than switching elements on and
off, so no variant ever loses the stars.

Legibility is solved here too, not only for the photographs
-----------------------------------------------------------
:mod:`spacr.qt.imagery` has always run every photographic master
through :func:`spacr.qt.imagery.solve_dim` against
:func:`spacr.qt.imagery.exposure_target` — the brightest a bare window
background may be before white text on it drops under WCAG AA. The
generated sky never went through it, and it showed: measured at
1440x900, the brightest text-line-sized region of the ``galaxy`` sky was
0.4879 against a 0.0586 limit (8.3x over), ``sun`` 0.8201 (14.0x) and
``stars`` 0.5041 (8.6x). Bare white text over that measured 1.20-1.96:1
where 4.5:1 is required — the "AI" toggle in the title bar sat on the
sun's halo. :func:`render` now solves it, and :func:`legibility` reports
the same measured dict :func:`spacr.qt.imagery.legibility` returns for a
photograph.

Reconciling the limit with the 40th-percentile anchor
-----------------------------------------------------
:data:`TARGET_SKY_PERCENTILE` exposes the frame so that *empty sky* —
not the mean — lands on :data:`TARGET_SKY_LUMA`, deliberately, so that a
sun stays blown out instead of dragging the whole frame to black. Taken
as a statement about one global exposure that is now flatly incompatible
with the limit, and both of the obvious ways to force it are ruined
pictures. Re-solving the exposure, or dimming the finished frame with
:func:`spacr.qt.imagery.solve_dim`, needs a factor of 0.064-0.108; both
take the sky's own 40th percentile from 0.00091 to 0.00000, i.e. every
faint star and the whole nebula go to pure black, and the peak pixel
falls from 236-252 to 67-90. Measured, rendered and looked at: a dark
grey smudge.

The reconciliation is that the two rules are about different things.
The exposure anchor is about a *pixel with nothing in it*; the WCAG
limit is about the mean over a **region the size of a line of text**
(:data:`spacr.qt.imagery.TEXT_WINDOW`), which is the point
:mod:`spacr.qt.imagery` already makes about photographs — every
photograph has a white pixel somewhere and it is a bright *patch* that
makes a caption unreadable. A 3 px star inside a 202x48 px text window
moves that window's mean by a quarter of one per cent; the whole
starfield measured on its own comes to 0.0007-0.0067, i.e. 1-11 % of the
limit. A 207 px sun disc fills the window completely and cannot.

So the exposure anchor is kept exactly as it was — the sky background is
unchanged, to the byte — and the limit is enforced where it is actually
violated, by compressing the highlights of the **smooth** layers only
(:func:`_compress_highlights`, applied to nebula + galaxy + sun + bloom,
never to the starfield). The sun therefore does get noticeably darker,
which is the accepted cost; the galaxy is barely touched because it was
never the offender; the stars keep white cores and diffraction spikes.
:func:`_enforce_legibility` then measures the finished frame with the
same :func:`spacr.qt.imagery.brightest_window` used on the photographs
and applies whatever residual dim is left, so the guarantee is a
measurement and not an argument.

Real imagery
------------
:func:`download_nasa_background` optionally fetches a public-domain
NASA/ESA image instead. NASA media is public domain (see
https://www.nasa.gov/nasa-brand-center/images-and-media/) and the
credit line is recorded so the UI can display it. Nothing here touches
the network at import time or from any code path other than that one
function, so an offline machine silently gets the procedural sky.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Bumped whenever the generators change output, so old cached PNGs are
#: not reused for a different-looking sky.
#:
#: v2: the sky is exposure-solved against the palette's bare-text limit.
#: Without this bump every existing user keeps the old, 8-14x too bright
#: sky forever — the cache is keyed by size, variant and seed, all of
#: which are unchanged, so nothing else would ever invalidate it.
CACHE_VERSION = 2

VARIANTS = ("galaxy", "sun", "stars")
DEFAULT_VARIANT = "galaxy"

#: Any int works; this is just a pleasing sky.
DEFAULT_SEED = 20250726

#: Never generate larger than this even on an 8K panel — beyond 4K the
#: extra pixels cost seconds and buy nothing behind a UI.
MAX_DIM = (3840, 2400)

#: Smallest sensible background. Keeps a test asking for 4x4 from
#: dividing by zero.
MIN_DIM = (16, 16)

#: Floor for a background chosen from the *screen* size. The stylesheet
#: centres the image without repeating it, so anything smaller than the
#: window letterboxes into hard-edged bands.
MIN_BACKGROUND = (1920, 1200)

#: Environment override for the cache location (used by the tests, and
#: handy for a read-only home directory).
ENV_CACHE_DIR = "SPACR_SPACE_CACHE"


# ---------------------------------------------------------------------------
# Small numpy helpers
# ---------------------------------------------------------------------------

def _clampi(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def _bilinear_upsample(src: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize ``src`` (h, w, c) float32 to (height, width, c), bilinearly.

    Written by hand rather than pulled from PIL/scipy so the result is
    bit-reproducible across versions of those libraries — the cache and
    the determinism test both depend on that.
    """
    sh, sw = src.shape[:2]
    if (sh, sw) == (height, width):
        return src

    def _axis(n_out: int, n_in: int):
        if n_in == 1:
            return (np.zeros(n_out, dtype=np.intp),
                    np.zeros(n_out, dtype=np.intp),
                    np.zeros(n_out, dtype=np.float32))
        pos = (np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5
        pos = np.clip(pos, 0.0, n_in - 1.0)
        i0 = np.floor(pos).astype(np.intp)
        i0 = np.clip(i0, 0, n_in - 2)
        frac = (pos - i0).astype(np.float32)
        return i0, i0 + 1, frac

    y0, y1, fy = _axis(height, sh)
    x0, x1, fx = _axis(width, sw)

    rows = (src[y0] * (1.0 - fy)[:, None, None]
            + src[y1] * fy[:, None, None])
    out = (rows[:, x0] * (1.0 - fx)[None, :, None]
           + rows[:, x1] * fx[None, :, None])
    return out.astype(np.float32, copy=False)


def _value_noise(height: int, width: int, rng, octaves: int = 4,
                 base: int = 4) -> np.ndarray:
    """Sum-of-octaves value noise in [0, 1], (height, width) float32."""
    out = np.zeros((height, width), dtype=np.float32)
    amp = 1.0
    total = 0.0
    for octave in range(octaves):
        cells = base * (2 ** octave)
        grid = rng.random((max(2, cells), max(2, cells))).astype(np.float32)
        out += amp * _bilinear_upsample(grid[:, :, None], width, height)[:, :, 0]
        total += amp
        amp *= 0.5
    return out / max(total, 1e-6)


def _area_downsample(img: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool ``img`` (h, w, c) by an integer ``factor``.

    Energy-preserving, unlike point sampling — a single bright pixel
    survives as a dim block instead of either vanishing or being
    replicated.
    """
    factor = max(1, int(factor))
    h, w = img.shape[:2]
    nh = max(1, h // factor)
    nw = max(1, w // factor)
    cropped = img[:nh * factor, :nw * factor]
    return cropped.reshape(nh, factor, nw, factor, img.shape[2]
                           ).mean(axis=(1, 3)).astype(np.float32, copy=False)


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur via a cumulative sum. ``img`` is (h, w, c)."""
    if radius < 1:
        return img
    out = img
    for axis in (0, 1):
        n = out.shape[axis]
        r = min(radius, max(1, n // 2))
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        padded = np.pad(out, pad, mode="edge")
        cs = np.cumsum(padded, axis=axis, dtype=np.float32)
        zero = np.zeros_like(np.take(cs, [0], axis=axis))
        cs = np.concatenate([zero, cs], axis=axis)
        hi = np.take(cs, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
        lo = np.take(cs, np.arange(0, n), axis=axis)
        out = (hi - lo) / float(2 * r + 1)
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Stellar colour — blackbody locus, sRGB
# ---------------------------------------------------------------------------

#: Temperature (K) -> sRGB of a blackbody normalised to peak channel.
#: Anchors from the standard Planckian-locus tables; interpolated
#: linearly in log(T) in between.
_BB_T = np.array(
    [2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 20000, 40000],
    dtype=np.float64)
_BB_RGB = np.array([
    [255, 137, 18],    # 2000 K — deep orange M
    [255, 180, 107],   # 3000 K — orange K/M
    [255, 209, 163],   # 4000 K — warm white K
    [255, 228, 206],   # 5000 K — G
    [255, 244, 242],   # 6000 K — sun-white
    [245, 243, 255],   # 7000 K — F
    [227, 233, 255],   # 8000 K — A
    [201, 215, 255],   # 10000 K — hot A
    [191, 207, 255],   # 12000 K — B
    [175, 195, 255],   # 20000 K — hot B
    [168, 189, 255],   # 40000 K — O
], dtype=np.float64)

#: Rough naked-eye spectral-class mix. Not the IMF (which is almost all
#: M dwarfs, none of them visible) — this is what the sky looks like.
_CLASS_WEIGHTS = np.array([0.10, 0.22, 0.19, 0.14, 0.20, 0.15])
_CLASS_TRANGE = np.array([
    [15000., 33000.],   # O/B
    [7500., 10000.],    # A
    [6000., 7500.],     # F
    [5200., 6000.],     # G
    [3700., 5200.],     # K
    [2400., 3700.],     # M
])


def star_colors(temps: np.ndarray) -> np.ndarray:
    """Map blackbody temperatures (K) to sRGB floats in [0, 1]."""
    logt = np.log(np.clip(temps, _BB_T[0], _BB_T[-1]))
    ref = np.log(_BB_T)
    out = np.empty((temps.shape[0], 3), dtype=np.float32)
    for c in range(3):
        out[:, c] = np.interp(logt, ref, _BB_RGB[:, c]) / 255.0
    return out


def sample_star_temperatures(rng, n: int) -> np.ndarray:
    """Draw ``n`` stellar temperatures from the naked-eye class mix."""
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    cls = rng.choice(len(_CLASS_WEIGHTS), size=n,
                     p=_CLASS_WEIGHTS / _CLASS_WEIGHTS.sum())
    lo = _CLASS_TRANGE[cls, 0]
    hi = _CLASS_TRANGE[cls, 1]
    return lo + rng.random(n) * (hi - lo)


# ---------------------------------------------------------------------------
# Magnitude distribution
# ---------------------------------------------------------------------------

#: Slope of the cumulative number counts. 1.5 is the Euclidean value
#: for sources spread uniformly through space: N(>F) ∝ F^-1.5. Inverse
#: transform sampling of that CDF gives F = F_min · u^(-1/1.5).
COUNT_SLOPE = 1.5

#: Flux at which a star saturates to a white core (in units of F_min).
FLUX_SATURATION = 240.0


def sample_star_fluxes(rng, n: int) -> np.ndarray:
    """Draw ``n`` relative fluxes (>= 1.0) from the Euclidean count law.

    The tail is unbounded in principle; it is clipped at
    :data:`FLUX_SATURATION` so one absurd draw cannot whiteout the sky.

    The resulting distribution is emphatically *not* uniform: by
    construction the fraction brighter than ``k·F_min`` is ``k^-1.5``,
    i.e. ~65 % of stars sit in the faintest factor-of-two bin while
    only ~4 % are 8x brighter than the limit.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    u = rng.random(n)
    # u == 0 would divide to infinity; nextafter keeps it finite.
    u = np.clip(u, np.finfo(np.float64).tiny, 1.0)
    flux = u ** (-1.0 / COUNT_SLOPE)
    return np.minimum(flux, FLUX_SATURATION)


# ---------------------------------------------------------------------------
# Starfield
# ---------------------------------------------------------------------------

#: Stars per megapixel. Dense enough to read as sky, sparse enough that
#: it does not turn into luminance noise.
STAR_DENSITY = 2600.0

#: How many stars get diffraction spikes. Only the very brightest — a
#: spike on every star reads as a filter, not as a telescope.
SPIKE_COUNT = 14


def _splat(width: int, height: int, xs, ys, fluxes, colors,
           radii) -> np.ndarray:
    """Accumulate Gaussian point sources into an (h, w, 3) float32 buffer.

    Splats are built as one flat (index, weight) list and reduced with
    three ``np.bincount`` calls, which is an order of magnitude cheaper
    at 4K than looping ``np.add.at`` per kernel tap.
    """
    buf = np.zeros((height * width, 3), dtype=np.float32)
    if xs.size == 0:
        return buf.reshape(height, width, 3)

    max_r = int(np.ceil(radii.max()))
    max_r = _clampi(max_r, 1, 24)
    taps = np.arange(-max_r, max_r + 1)

    xi = np.floor(xs).astype(np.intp)
    yi = np.floor(ys).astype(np.intp)
    fx = (xs - xi).astype(np.float32)
    fy = (ys - yi).astype(np.float32)
    sigma = np.maximum(radii, 0.55).astype(np.float32)
    inv2s2 = (1.0 / (2.0 * sigma * sigma)).astype(np.float32)

    idx_chunks = []
    w_chunks = []
    for dy in taps:
        oy = yi + dy
        inside_y = (oy >= 0) & (oy < height)
        if not inside_y.any():
            continue
        gy = (dy - fy) ** 2
        for dx in taps:
            ox = xi + dx
            inside = inside_y & (ox >= 0) & (ox < width)
            if not inside.any():
                continue
            d2 = gy + (dx - fx) ** 2
            w = np.exp(-d2 * inv2s2).astype(np.float32)
            w *= fluxes.astype(np.float32)
            keep = inside & (w > 1e-4)
            if not keep.any():
                continue
            idx_chunks.append((oy[keep] * width + ox[keep]).astype(np.intp))
            w_chunks.append((w[keep], keep))

    if not idx_chunks:
        return buf.reshape(height, width, 3)

    flat = np.concatenate(idx_chunks)
    n_px = height * width
    for c in range(3):
        weights = np.concatenate(
            [w * colors[keep, c] for (w, keep) in w_chunks])
        buf[:, c] = np.bincount(flat, weights=weights,
                                minlength=n_px).astype(np.float32)
    return buf.reshape(height, width, 3)


def _diffraction_spikes(width: int, height: int, xs, ys, fluxes,
                        colors) -> np.ndarray:
    """Draw 4-way spikes for the given (already selected) bright stars."""
    buf = np.zeros((height, width, 3), dtype=np.float32)
    if xs.size == 0:
        return buf
    span = max(6, int(0.035 * min(width, height)))
    t = np.arange(1, span + 1, dtype=np.float32)
    falloff = np.exp(-t / (span * 0.34)).astype(np.float32)
    for sx, sy, flux, col in zip(xs, ys, fluxes, colors):
        amp = float(flux) ** 0.5 * 0.16
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            px = np.rint(sx + dx * t).astype(np.intp)
            py = np.rint(sy + dy * t).astype(np.intp)
            ok = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            if not ok.any():
                continue
            w = falloff[ok] * amp
            np.add.at(buf, (py[ok], px[ok]), w[:, None] * col[None, :])
    return buf


def starfield(width: int, height: int, seed: int = DEFAULT_SEED,
              density: float = STAR_DENSITY,
              spike_count: int = SPIKE_COUNT) -> np.ndarray:
    """Render a starfield as an (height, width, 3) float32 HDR buffer."""
    rng = np.random.default_rng(seed)
    megapixels = (width * height) / 1.0e6
    n = int(max(24, round(density * megapixels)))

    xs = rng.random(n) * width
    ys = rng.random(n) * height
    flux = sample_star_fluxes(rng, n)
    temps = sample_star_temperatures(rng, n)
    colors = star_colors(temps)

    # Brightness -> visual size. Real point sources all have the same
    # PSF; what makes a bright star look bigger is the wings clipping
    # above threshold, which a gentle power law imitates cheaply.
    scale = max(0.6, min(width, height) / 1400.0)
    radii = (0.62 + 0.42 * flux ** 0.32) * scale
    # HDR amplitude spanning ~80x from the faintest star to a saturated
    # core, so the tone map has something to clip to white.
    amp = 8.0 * (flux / FLUX_SATURATION) ** 0.8

    buf = _splat(width, height, xs, ys, amp, colors, radii)

    if spike_count > 0 and n > spike_count:
        top = np.argsort(flux)[-spike_count:]
        buf += _diffraction_spikes(width, height, xs[top], ys[top],
                                   amp[top], colors[top])
    return buf


# ---------------------------------------------------------------------------
# Galaxy
# ---------------------------------------------------------------------------

#: Downsample factor for the smooth (galaxy / sun / nebula) components.
#: They contain no detail finer than a few pixels, so computing them at
#: a third of the resolution and upsampling is visually identical and
#: ~9x cheaper.
SMOOTH_SCALE = 3


def galaxy(width: int, height: int, seed: int = DEFAULT_SEED,
           center: Tuple[float, float] = (0.30, 0.34),
           radius_frac: float = 0.38,
           arms: int = 2,
           inclination: float = 0.62,
           position_angle: float = -0.55,
           pitch: float = 0.30) -> np.ndarray:
    """Render a logarithmic-spiral galaxy, (height, width, 3) float32.

    :param center: galaxy centre as a fraction of (width, height).
    :param radius_frac: disc scale radius as a fraction of the short edge.
    :param arms: number of spiral arms.
    :param inclination: 1.0 = face on, 0.0 = edge on.
    :param position_angle: rotation of the disc, radians.
    :param pitch: ``b`` in ``r = a·e^(bθ)``; smaller = more tightly wound.
    """
    rng = np.random.default_rng(seed ^ 0x9E3779B9)
    sw = max(8, width // SMOOTH_SCALE)
    sh = max(8, height // SMOOTH_SCALE)

    cx = center[0] * sw
    cy = center[1] * sh
    scale = radius_frac * min(sw, sh)

    yy = (np.arange(sh, dtype=np.float32) - cy)[:, None]
    xx = (np.arange(sw, dtype=np.float32) - cx)[None, :]

    ca, sa = np.cos(position_angle), np.sin(position_angle)
    xr = xx * ca + yy * sa
    yr = (-xx * sa + yy * ca) / max(inclination, 0.05)

    r = np.sqrt(xr * xr + yr * yr) / scale + 1e-4
    theta = np.arctan2(yr, xr)

    # Spiral phase: distance (in angle) to the nearest arm ridge.
    arm_theta = np.log(r + 1e-4) / pitch
    phase = (theta - arm_theta) % (2.0 * np.pi / arms)
    half = np.pi / arms
    d = np.abs(phase - half)          # 0 at the ridge, `half` between arms

    # Arms narrow at large radius, and fade out with the disc.
    width_arm = 0.34 + 0.30 * np.exp(-r * 1.4)
    ridge = np.exp(-(d / width_arm) ** 2).astype(np.float32)
    disc = np.exp(-r * 1.85).astype(np.float32)
    arm = ridge * disc

    # Dust lanes trail the arm ridge on the inner edge — a second,
    # phase-shifted ridge that *removes* light.
    d_dust = np.abs((phase - half * 0.55))
    dust = np.exp(-(d_dust / (width_arm * 0.42)) ** 2).astype(np.float32)
    dust *= np.exp(-r * 1.5).astype(np.float32)

    # Warm core bulge, Sérsic-ish, falling off into the disc.
    bulge = np.exp(-(r / 0.19) ** 0.72).astype(np.float32)

    # Clumpy HII knots along the arms.
    knots = _value_noise(sh, sw, rng, octaves=4, base=6)
    arm = arm * (0.62 + 0.85 * knots)

    smooth_disc = disc * 0.16

    out = np.zeros((sh, sw, 3), dtype=np.float32)
    # Young blue arms, a cooler blue haze between them, and a warm
    # old-population core — the colour gradient every spiral has.
    arm_col = np.array([0.44, 0.66, 1.00], dtype=np.float32)
    haze_col = np.array([0.36, 0.48, 0.95], dtype=np.float32)
    core_col = np.array([1.00, 0.78, 0.44], dtype=np.float32)
    knot_col = np.array([1.00, 0.52, 0.66], dtype=np.float32)   # HII pink
    out += arm[:, :, None] * arm_col[None, None, :] * 1.05
    out += (arm * np.clip(knots - 0.62, 0.0, 1.0) * 1.6
            )[:, :, None] * knot_col[None, None, :]
    out += smooth_disc[:, :, None] * haze_col[None, None, :]
    out += bulge[:, :, None] * core_col[None, None, :] * 1.55

    out *= (1.0 - 0.72 * dust)[:, :, None]
    out = _box_blur(out, radius=1)
    return _bilinear_upsample(out, width, height)


# ---------------------------------------------------------------------------
# Sun
# ---------------------------------------------------------------------------

#: Linear limb-darkening coefficient. 0.6 is the standard visual-band
#: value for a solar-type photosphere.
LIMB_DARKENING_U = 0.6


def sun(width: int, height: int, seed: int = DEFAULT_SEED,
        center: Tuple[float, float] = (0.80, 0.74),
        radius_frac: float = 0.085,
        temperature: float = 5800.0,
        corona_scale: float = 1.9) -> np.ndarray:
    """Render a star with limb darkening, granulation and a corona."""
    rng = np.random.default_rng(seed ^ 0x85EBCA6B)
    sw = max(8, width // SMOOTH_SCALE)
    sh = max(8, height // SMOOTH_SCALE)

    cx = center[0] * sw
    cy = center[1] * sh
    R = max(1.5, radius_frac * min(sw, sh))

    yy = (np.arange(sh, dtype=np.float32) - cy)[:, None]
    xx = (np.arange(sw, dtype=np.float32) - cx)[None, :]
    r = np.sqrt(xx * xx + yy * yy)

    inside = r < R
    # µ = cos(angle from disc centre as seen from the star's centre).
    mu = np.sqrt(np.clip(1.0 - (r / R) ** 2, 0.0, 1.0)).astype(np.float32)
    disc = np.where(inside,
                    (1.0 - LIMB_DARKENING_U * (1.0 - mu)), 0.0
                    ).astype(np.float32)

    # Granulation: convective cells, ±9 % on the photosphere only.
    gran = _value_noise(sh, sw, rng, octaves=3, base=10)
    disc = disc * (0.91 + 0.18 * gran)

    # Corona — smooth exponential falloff outside the limb, plus a few
    # radial streamers so it does not read as a plain glow.
    outside = np.maximum(r - R, 0.0)
    corona = np.exp(-outside / (R * corona_scale)).astype(np.float32)
    ang = np.arctan2(yy, xx).astype(np.float32)
    # Two harmonics at low amplitude. One strong harmonic gives the
    # corona symmetric "ears"; two weak ones read as structure.
    streamers = (1.0
                 + 0.16 * np.cos(ang * 7.0 + 0.7)
                 + 0.10 * np.cos(ang * 13.0 - 1.9))
    streamers = 1.0 + (streamers - 1.0) * np.exp(-outside / (R * 2.2))
    corona = corona * streamers.astype(np.float32)
    corona = np.where(inside, 0.0, corona).astype(np.float32)

    col = star_colors(np.array([temperature]))[0]
    out = np.zeros((sh, sw, 3), dtype=np.float32)
    out += disc[:, :, None] * col[None, None, :] * 60.0
    out += corona[:, :, None] * col[None, None, :] * 1.1
    out = _box_blur(out, radius=1)
    return _bilinear_upsample(out, width, height)


# ---------------------------------------------------------------------------
# Nebula haze — ties the composition together
# ---------------------------------------------------------------------------

def _nebula(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed ^ 0xC2B2AE35)
    sw = max(8, width // (SMOOTH_SCALE * 2))
    sh = max(8, height // (SMOOTH_SCALE * 2))
    n = _value_noise(sh, sw, rng, octaves=5, base=3)
    n = np.clip((n - 0.42) / 0.58, 0.0, 1.0) ** 1.8
    col = np.array([0.30, 0.36, 0.72], dtype=np.float32)
    out = n[:, :, None] * col[None, None, :] * 0.16
    return _bilinear_upsample(out, width, height)


# ---------------------------------------------------------------------------
# Composition + tone mapping
# ---------------------------------------------------------------------------

#: Per-variant weights: (galaxy, sun, star density multiplier, nebula).
_VARIANT_MIX = {
    "galaxy": dict(galaxy=1.0, sun=0.55, stars=1.0, nebula=1.0,
                   galaxy_radius=0.38, sun_radius=0.055),
    "sun":    dict(galaxy=0.22, sun=1.0, stars=1.05, nebula=0.55,
                   galaxy_radius=0.15, sun_radius=0.115),
    "stars":  dict(galaxy=0.18, sun=0.30, stars=1.55, nebula=1.25,
                   galaxy_radius=0.26, sun_radius=0.035),
}

#: Target luminance of the *sky background* — the 40th percentile, i.e.
#: a pixel with nothing in it. Anchoring the exposure there rather than
#: on the mean is what lets a big bright sun stay white-hot without
#: dragging the rest of the frame to black: a few per cent of the
#: pixels being a star must not re-expose the other 95 %.
#:
#: This still does exactly that, and it is still the *only* thing that
#: sets how dark the empty sky is — :func:`_compress_highlights` runs
#: strictly above :data:`HIGHLIGHT_KNEE` of the ceiling and cannot reach
#: down here. What it no longer implies is that the sun stays blown out
#: over an area the size of a line of text; see the module docstring for
#: why those are separable and what happens if you try to satisfy the
#: WCAG limit by moving this number instead.
TARGET_SKY_PERCENTILE = 40.0
TARGET_SKY_LUMA = 0.013

#: Hard ceiling on the *mean*. Text and panels sit on top of this
#: image, so a bright wallpaper is exactly how a themed app becomes
#: unreadable. If the sky anchor lands above this, the exposure is
#: re-solved against the mean instead.
MAX_MEAN_LUMA = 0.075


#: Fraction of the highlight ceiling below which :func:`_compress_highlights`
#: is the identity. Above it the curve bends; below it nothing moves, so
#: the sky anchor, the nebula and the galaxy's arms — all of which live
#: two orders of magnitude below the ceiling — come out untouched.
#:
#: 0.55 rather than something lower because the alternative was measured:
#: a power-law highlight gamma, which starts compressing at the knee and
#: keeps compressing all the way up, preserves the *sun's* limb but takes
#: the galaxy down with it (its brightest text window fell to 0.41x the
#: limit against 0.69x here, and the arms visibly washed out). The galaxy
#: was never what broke the limit and should not pay for the sun.
HIGHLIGHT_KNEE = 0.55


def _luma(img: np.ndarray) -> np.ndarray:
    """Relative-luminance channel of an (h, w, 3) buffer."""
    return (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1]
            + 0.0722 * img[:, :, 2])


def _tone_stat(mapped: np.ndarray, percentile: Optional[float]) -> float:
    if percentile is None:
        return float(mapped.mean())
    return float(np.percentile(mapped, percentile))


def _solve_exposure(luma: np.ndarray, target: float,
                    percentile: Optional[float]) -> float:
    """Bisect log-exposure so the tone-mapped statistic hits ``target``.

    Bisection on exposure rather than a post-hoc linear dim: dimming
    *after* the filmic curve crushes the star cores to grey, which is
    the difference between "a night sky" and "a dark grey rectangle".
    Scaling the exposure before the curve keeps the bright end pinned at
    white and darkens only the midtones.
    """
    sample = luma[::4, ::4]
    if sample.size == 0 or float(sample.max()) <= 0.0:
        return 1.0
    lo, hi = 1e-4, 1e5
    for _ in range(44):
        mid = float(np.sqrt(lo * hi))
        stat = _tone_stat(1.0 - np.exp(-sample * mid), percentile)
        if stat < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def tone_exposure(luma: np.ndarray) -> float:
    """The exposure :func:`_tone_map` would use for this frame.

    Split out of :func:`_tone_map` because :func:`render` needs the
    number itself: the highlight ceiling is a luminance in *HDR* units,
    and converting the palette's limit — which is a luminance in the
    finished, tone-mapped image — back into HDR units is exactly
    inverting this curve at this exposure.
    """
    exposure = _solve_exposure(luma, TARGET_SKY_LUMA, TARGET_SKY_PERCENTILE)
    sample = luma[::4, ::4]
    if sample.size and float((1.0 - np.exp(-sample * exposure)).mean()) > MAX_MEAN_LUMA:
        exposure = _solve_exposure(luma, MAX_MEAN_LUMA, None)
    return exposure


def _apply_tone_curve(hdr: np.ndarray, exposure: float) -> np.ndarray:
    out = 1.0 - np.exp(-hdr * exposure)
    np.clip(out, 0.0, 1.0, out=out)
    return out.astype(np.float32, copy=False)


def _tone_map(hdr: np.ndarray) -> np.ndarray:
    """Filmic compression to [0, 1] at an auto-solved exposure."""
    return _apply_tone_curve(hdr, tone_exposure(_luma(hdr)))


# ---------------------------------------------------------------------------
# The legibility solve
# ---------------------------------------------------------------------------

def exposure_target() -> float:
    """Luminance the brightest text-line-sized region is aimed at.

    The Space palette's own limit from
    :func:`spacr.qt.theme.max_background_luma`, backed off by
    :data:`spacr.qt.imagery.SAFETY_MARGIN` — the identical number the
    photographic masters are solved to, fetched from the identical
    function, so the two pipelines cannot drift apart.

    :mod:`spacr.qt.imagery` imports *this* module at module scope, so
    the import has to be deferred to call time. By then ``space`` is
    fully loaded whichever of the two the caller reached first.
    """
    from . import imagery
    return imagery.exposure_target("space")


def highlight_ceiling(exposure: float, target: Optional[float] = None
                      ) -> float:
    """HDR luminance that tone-maps to ``target``'s encoded value.

    :func:`exposure_target` is a *linear-light* relative luminance of the
    finished image. ``_tone_map``'s output is written straight to 8-bit
    without a gamma encode, so it is an **sRGB signal value**, and the
    two are a transfer function apart — the 0.0586 limit is a mid-dark
    grey around ``#444444``, not a 6 % signal. Encode first, then invert
    ``1 - exp(-x·E)``.

    :returns: ``inf`` when there is nothing to solve for — a palette
        that admits no wallpaper at all (:func:`spacr.qt.theme.max_background_luma`
        is *negative* for the light theme) or a zero exposure. Callers
        read that as "no ceiling", never as "clamp everything to zero".
        A palette that admits a white wallpaper lands on a ceiling far
        above anything the generators emit, which comes to the same
        thing without a second branch to leave untested.
    """
    from . import imagery
    target = exposure_target() if target is None else target
    if target <= 0.0 or exposure <= 0.0:
        return float("inf")
    headroom = max(1e-12, 1.0 - imagery.srgb_encode(target))
    return float(-np.log(headroom) / exposure)


def _compress_highlights(smooth: np.ndarray, ceiling: float,
                         knee: float = HIGHLIGHT_KNEE) -> np.ndarray:
    """Bend ``smooth``'s luminance so no pixel of it exceeds ``ceiling``.

    Pointwise, monotone and hue-preserving: every pixel is scaled by the
    ratio its own luminance is compressed by, so nothing in the frame
    changes colour and — this is the whole reason it is pointwise —
    nothing gains a halo. Two spatial alternatives were built and looked
    at first: a local gain solved from the sliding window mean rings the
    sun's limb, and taking the max envelope of that gain to kill the ring
    stamps a visible dark *square* around the sun, the shape of its own
    structuring element.

    Only the smooth layers are handed to this. The starfield is added
    afterwards and keeps its saturated white cores, because a point
    source does not move a text-window mean (measured: 0.0007-0.0067
    for the whole starfield, against a 0.0586 limit).

    Modifies ``smooth`` in place and returns it — at 3840x2400 a copy is
    another 110 MB for no gain.
    """
    if not np.isfinite(ceiling) or ceiling <= 0.0:
        return smooth
    luma = _luma(smooth)
    if float(luma.max()) <= ceiling:
        return smooth
    foot = ceiling * knee
    span = ceiling - foot
    # Exponential shoulder: identity below `foot`, asymptotic to
    # `ceiling`, C1 at the join (both value and slope match), so a
    # smooth gradient crossing it gains no Mach band.
    bent = foot + span * (1.0 - np.exp(-np.maximum(luma - foot, 0.0) / span))
    scale = np.where(luma <= foot, np.float32(1.0),
                     bent / np.maximum(luma, 1e-9)).astype(np.float32)
    smooth *= scale[:, :, None]
    return smooth


def _measure_probe(arr: np.ndarray, long_edge: int = 480) -> np.ndarray:
    """Box-averaged thumbnail of a uint8 frame, for measurement only.

    Same trick, and the same 480 px, as :func:`spacr.qt.imagery._probe`:
    every number measured off this is a mean over a region hundreds of
    pixels across, and a box average answers those to several decimals
    without building a 221 MB float array at 4K.
    """
    factor = max(1, int(max(arr.shape[:2]) // max(1, long_edge)))
    if factor <= 1:
        return arr
    small = _area_downsample(arr.astype(np.float32), factor)
    return np.clip(small + 0.5, 0, 255).astype(np.uint8)


def _enforce_legibility(arr: np.ndarray,
                        target: Optional[float] = None) -> np.ndarray:
    """Final measured guarantee: dim ``arr`` until it is under ``target``.

    :func:`_compress_highlights` bounds the smooth layers by
    construction and in practice lands the finished frame at 0.69-0.78x
    the limit with nothing left to do, so this normally resolves to a
    factor of exactly 1.0 and returns ``arr`` untouched. It is here
    because "by construction" is an argument and this is a measurement:
    the starfield, the bloom and the vignette all land on the frame
    after the ceiling is chosen, and an unusual size or seed is allowed
    to put them somewhere the argument did not anticipate.

    It is the same measure-and-solve pair the photographic masters get —
    :func:`spacr.qt.imagery.brightest_window` into
    :func:`spacr.qt.imagery.solve_dim` — run over the same
    :data:`spacr.qt.imagery.TEXT_WINDOW`.
    """
    from . import imagery
    target = exposure_target() if target is None else target
    if target <= 0.0:
        return arr
    measured, _ = imagery.brightest_window(_measure_probe(arr))
    factor = imagery.solve_dim(measured, target)
    if factor >= 1.0:
        return arr
    return imagery.dim(arr, factor)


def legibility(variant: str = DEFAULT_VARIANT, width: int = 0,
               height: int = 0, seed: int = DEFAULT_SEED) -> dict:
    """Measure how readable the generated sky actually is.

    The same dict, measured the same way over the same region, that
    :func:`spacr.qt.imagery.legibility` returns for a photographic
    master — so "is the wallpaper legible" is one question with one
    answer shape whether the pixels were generated or photographed.

    ``width``/``height`` default to :func:`screen_size`.
    """
    from . import imagery
    if width <= 0 or height <= 0:
        width, height = screen_size()
    arr = render(width, height, variant=variant, seed=seed)
    return imagery.legibility_of(_measure_probe(arr), "space",
                                 key=f"space:{variant}")


def _vignette(width: int, height: int) -> np.ndarray:
    yy = (np.linspace(-1.0, 1.0, height, dtype=np.float32))[:, None]
    xx = (np.linspace(-1.0, 1.0, width, dtype=np.float32))[None, :]
    r2 = xx * xx + yy * yy
    return (1.0 - 0.30 * np.clip(r2 / 2.0, 0.0, 1.0)).astype(np.float32)


def render(width: int, height: int, variant: str = DEFAULT_VARIANT,
           seed: int = DEFAULT_SEED, legible: bool = True) -> np.ndarray:
    """Render the composed sky as an (height, width, 3) uint8 array.

    Deterministic: identical arguments always give identical bytes.

    :param legible: when true (always, outside the tests) the finished
        frame is bounded by the Space palette's bare-text limit — see
        :func:`_compress_highlights` and :func:`_enforce_legibility`.
        ``False`` renders the unbounded sky, which exists so the test
        suite can measure what the bound is worth rather than assert
        that it was called.

    The starfield is kept in its own buffer to the very end. That is not
    tidiness: it is the one layer the highlight ceiling must not touch,
    and keeping it separate is what lets a star core stay at 255 while
    the sun beside it comes down by two and a half stops.
    """
    width = _clampi(width, MIN_DIM[0], MAX_DIM[0])
    height = _clampi(height, MIN_DIM[1], MAX_DIM[1])
    mix = _VARIANT_MIX.get(variant, _VARIANT_MIX[DEFAULT_VARIANT])

    smooth = np.zeros((height, width, 3), dtype=np.float32)
    smooth += _nebula(width, height, seed) * mix["nebula"]
    smooth += galaxy(width, height, seed=seed,
                     radius_frac=mix["galaxy_radius"]) * mix["galaxy"]
    smooth += sun(width, height, seed=seed,
                  radius_frac=mix["sun_radius"]) * mix["sun"]
    stars = starfield(width, height, seed=seed,
                      density=STAR_DENSITY * mix["stars"])
    hdr = smooth + stars

    # Mild bloom so bright things bleed the way a lens does. The
    # downsample has to *average* — point-sampling a 1 px star into a
    # 1/8-scale buffer and box-blurring it paints a visible 70 px
    # square, which is how the first cut of this looked. Two blur
    # passes then turn the box kernel into a tent so no hard edge
    # survives the upsample.
    #
    # Bloom counts as a smooth layer: it is a blur, it has no detail
    # finer than ~100 px by construction, and a star's bloom really does
    # cover a text window even though the star itself does not.
    small = _area_downsample(hdr, 8)
    small = _box_blur(_box_blur(small, 3), 3)
    smooth += _bilinear_upsample(small, width, height) * 0.85
    np.add(smooth, stars, out=hdr)

    # Solved on the composed frame, before any ceiling, so the sky
    # anchor sees exactly what it always saw.
    exposure = tone_exposure(_luma(hdr))
    if legible:
        _compress_highlights(smooth, highlight_ceiling(exposure))
        np.add(smooth, stars, out=hdr)

    ldr = _apply_tone_curve(hdr, exposure)
    ldr *= _vignette(width, height)[:, :, None]
    arr = np.clip(ldr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return _enforce_legibility(arr) if legible else arr


def to_qimage(arr: np.ndarray):
    """Convert an (h, w, 3) uint8 array to a detached ``QImage``."""
    from PySide6.QtGui import QImage
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    return img.copy()          # detach from the numpy buffer


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def cache_dir() -> Path:
    """Directory holding generated backgrounds.

    Honours ``$SPACR_SPACE_CACHE`` so tests (and read-only homes) can
    redirect it; otherwise ``~/.spacr/backgrounds``, matching where the
    verbose logger already writes.
    """
    override = os.environ.get(ENV_CACHE_DIR)
    root = Path(override) if override else Path.home() / ".spacr" / "backgrounds"
    return root


def cache_name(width: int, height: int, variant: str, seed: int) -> str:
    return f"space-{variant}-{width}x{height}-s{seed}-v{CACHE_VERSION}.png"


def _load_cached(path: Path, width: int, height: int) -> bool:
    """True when ``path`` holds a usable PNG of the requested size.

    A truncated or garbage file (interrupted write, half-synced home
    directory) must regenerate, not raise.
    """
    try:
        if not path.is_file() or path.stat().st_size < 128:
            return False
        from PySide6.QtGui import QImage
        probe = QImage()
        if not probe.load(str(path)):
            return False
        return probe.width() == width and probe.height() == height
    except Exception:
        return False


def background_path(width: int, height: int,
                    variant: str = DEFAULT_VARIANT,
                    seed: int = DEFAULT_SEED,
                    regenerate: bool = False) -> Optional[Path]:
    """Return the on-disk path of the background, generating if needed.

    Returns ``None`` (never raises) when the background cannot be
    produced — a read-only home directory, no PNG writer, anything. The
    Space theme falls back to a flat gradient in that case, so a failure
    here costs some prettiness and nothing else.
    """
    width = _clampi(width, MIN_DIM[0], MAX_DIM[0])
    height = _clampi(height, MIN_DIM[1], MAX_DIM[1])
    try:
        directory = cache_dir()
        path = directory / cache_name(width, height, variant, seed)
        if not regenerate and _load_cached(path, width, height):
            return path
        directory.mkdir(parents=True, exist_ok=True)
        arr = render(width, height, variant=variant, seed=seed)
        img = to_qimage(arr)
        tmp = path.with_suffix(".png.part")
        if not img.save(str(tmp), "PNG"):
            return None
        os.replace(tmp, path)
        return path
    except Exception:
        return None


def clear_cache() -> int:
    """Delete every cached background. Returns the number removed."""
    removed = 0
    try:
        for p in cache_dir().glob("space-*.png"):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    except Exception:
        pass
    return removed


def _gui_app():
    """The running ``QGuiApplication``, or ``None``.

    A one-line indirection so tests can simulate "no app" / "Qt threw"
    without monkeypatching ``QGuiApplication.instance`` process-wide,
    which breaks every later test that needs a real application.
    """
    from PySide6.QtGui import QGuiApplication
    return QGuiApplication.instance()


def screen_size(default: Tuple[int, int] = (2560, 1440)) -> Tuple[int, int]:
    """Native pixel size of the primary screen, or ``default`` headless."""
    try:
        app = _gui_app()
        if app is None:
            return default
        screen = app.primaryScreen()
        if screen is None:
            return default
        geo = screen.geometry()
        ratio = float(screen.devicePixelRatio() or 1.0)
        w = int(round(geo.width() * ratio))
        h = int(round(geo.height() * ratio))
        if w < MIN_DIM[0] or h < MIN_DIM[1]:
            return default
        # Never smaller than MIN_BACKGROUND. The QSS centres the image
        # without repeating it, so a background narrower than the window
        # letterboxes into hard-edged bands of flat colour. That cannot
        # happen when the screen is bigger than the window, but a
        # virtual/offscreen display can report almost anything.
        return (_clampi(w, MIN_BACKGROUND[0], MAX_DIM[0]),
                _clampi(h, MIN_BACKGROUND[1], MAX_DIM[1]))
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Optional real imagery — NASA / ESA public domain
# ---------------------------------------------------------------------------

#: NASA still images, audio, and video are generally not copyrighted and
#: may be used for any purpose; see
#: https://www.nasa.gov/nasa-brand-center/images-and-media/
#: ESA/Webb and ESA/Hubble release under CC BY 4.0 with the same credit
#: requirement. Either way the credit line has to reach the user, which
#: is why every entry carries one and why it is persisted next to the
#: downloaded file.
NASA_IMAGES = (
    {
        "key": "carina",
        "title": "Cosmic Cliffs, Carina Nebula (NIRCam)",
        "url": "https://www.nasa.gov/wp-content/uploads/2023/03/"
               "main_image_star-forming_region_carina_nircam_final-5mb.jpg",
        "credit": "NASA, ESA, CSA, and STScI",
        "source": "https://www.nasa.gov/webbfirstimages",
    },
    {
        "key": "deep_field",
        "title": "Webb's First Deep Field (SMACS 0723)",
        "url": "https://www.nasa.gov/wp-content/uploads/2023/03/"
               "main_image_deep_field_smacs0723-5mb.jpg",
        "credit": "NASA, ESA, CSA, and STScI",
        "source": "https://www.nasa.gov/webbfirstimages",
    },
    {
        "key": "sun_flare",
        "title": "Solar flare, Solar Dynamics Observatory",
        "url": "https://www.nasa.gov/wp-content/uploads/2023/03/"
               "sdo-flare-20170906.jpg",
        "credit": "NASA/SDO",
        "source": "https://sdo.gsfc.nasa.gov/",
    },
)

#: Written next to a downloaded image so the attribution survives a
#: restart and can be shown in Preferences.
CREDITS_FILE = "credits.json"


def imagery_dir() -> Path:
    return cache_dir() / "nasa"


def read_credits() -> Optional[dict]:
    """Return the recorded attribution for the downloaded image, if any."""
    try:
        path = imagery_dir() / CREDITS_FILE
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("file"):
            if (imagery_dir() / str(data["file"])).is_file():
                return data
        return None
    except Exception:
        return None


def downloaded_background() -> Optional[Path]:
    """Path of the downloaded NASA image, or ``None`` if there isn't one."""
    data = read_credits()
    if not data:
        return None
    path = imagery_dir() / str(data["file"])
    return path if path.is_file() else None


def _urlopen_bytes(url: str, timeout: float) -> bytes:
    """Read ``url`` into memory. Imported lazily — nothing in this module
    touches :mod:`urllib` unless a download is actually requested."""
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def download_nasa_background(key: str = "carina", timeout: float = 20.0,
                             opener=None) -> Optional[dict]:
    """Download a public-domain NASA image and record its credit.

    **Never called at import time, and never from a test.** The Space
    theme works without it; this is strictly an upgrade the user opts
    into from Preferences.

    :param key: which entry of :data:`NASA_IMAGES` to fetch.
    :param timeout: socket timeout in seconds.
    :param opener: injection point for tests — a callable taking
        ``(url, timeout)`` and returning bytes. Defaults to
        ``urllib.request.urlopen``.
    :returns: the credit dict on success, ``None`` on any failure
        (offline, 404, unwritable cache). Callers must treat ``None`` as
        "keep using the procedural sky", not as an error.
    """
    entry = next((e for e in NASA_IMAGES if e["key"] == key), None)
    if entry is None:
        return None
    try:
        fetch = opener or _urlopen_bytes
        blob = fetch(entry["url"], timeout)
        if not blob or len(blob) < 1024:
            return None
        directory = imagery_dir()
        directory.mkdir(parents=True, exist_ok=True)
        fname = f"{entry['key']}.jpg"
        tmp = directory / (fname + ".part")
        tmp.write_bytes(blob)
        # Only accept it if Qt can actually decode it — a captive-portal
        # HTML error page is 2 kB of "valid" bytes that would otherwise
        # be installed as the wallpaper.
        from PySide6.QtGui import QImage
        probe = QImage()
        if not probe.load(str(tmp)):
            tmp.unlink(missing_ok=True)
            return None
        # This file becomes the Space wallpaper *directly* — the
        # stylesheet points at it, nothing renders it per screen — so it
        # is the one path by which an unbounded picture could still get
        # behind the app's text. A solar flare frame is exactly that.
        # Solve it here or refuse it; the procedural sky is the fallback
        # and it is bounded.
        from . import imagery
        if not imagery.solve_image_file(tmp, "space"):
            tmp.unlink(missing_ok=True)
            return None
        os.replace(tmp, directory / fname)
        record = {
            "file": fname,
            "title": entry["title"],
            "credit": entry["credit"],
            "source": entry["source"],
            "width": probe.width(),
            "height": probe.height(),
        }
        (directory / CREDITS_FILE).write_text(
            json.dumps(record, indent=2), encoding="utf-8")
        return record
    except Exception:
        return None


def attribution_text() -> str:
    """One-line credit for the imagery currently in use."""
    data = read_credits()
    if not data:
        return ("Procedural sky — generated locally, no download. "
                "Stars, a spiral galaxy and a star with a corona.")
    return f"{data.get('title', 'NASA image')} — {data.get('credit', 'NASA')}"

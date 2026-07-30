"""
Photographic backgrounds for the image-backed themes.

Why this exists next to :mod:`spacr.qt.space`
---------------------------------------------
:mod:`spacr.qt.space` *generates* a sky with numpy. This module does the
same job for pixels that already exist: three photographs the user
supplied — two of their own micrographs and one deep-field astronomy
frame — turned into wallpapers for the Space and Cell themes.

It reuses ``space``'s cache directory, its size clamps and its
"never raise, fall back to something" contract, so from the outside a
photo background and a generated one behave identically. What is new is
the part that photographs need and procedural pixels do not: a decode
budget, a crop policy, and a measured legibility check.

The performance limit is decoded memory, not file size
------------------------------------------------------
``space_1.jpeg`` is 10.2 MB on disk and **281 MB decoded** as
10000x7020 RGBA. Re-decoding that on a window resize would be brutal and
holding it resident is worse. So:

* the masters shipped in ``spacr/resources/themes`` are already cropped
  and capped at :data:`MASTER_CAP` (3840x2400 — :data:`spacr.qt.space.MAX_DIM`),
  so the largest thing ever decoded at runtime is ~27 MB, not 281 MB;
* each screen size is rendered **once** and cached as JPEG under
  ``~/.spacr/backgrounds``; and
* :func:`decode_count` counts every master decode, so the claim "no
  master is touched during a resize or a repaint" is a number the test
  suite checks rather than a promise in a docstring.

The crop policy
---------------
:data:`MASTERS` records, per image, the sub-rectangle of the original
that is usable and the vertical focus for the aspect crop. Two of the
three are trimmed for a reason:

* ``cell_2.png`` carries a burned-in "5 um" scale bar and label in the
  bottom right (measured at x 0.825-0.949, y 0.925-0.953 of the frame).
  A wallpaper is not a figure, and a stray scale bar reads as an
  artefact, so its ``source_crop`` cuts above it. Because the shipped
  master is built from that crop, **no** runtime crop can bring the bar
  back.
* ``space_1.jpeg`` and ``cell.png`` carry no burned-in annotation
  (checked: zero 32x32 blocks more than 35 % saturated achromatic
  white), so they are used whole.

Legibility is solved, not eyeballed
-----------------------------------
These are not a procedural sky of point stars — ``cell.png``'s cyan core
fills most of the frame and measures 0.236 in the brightest
text-line-sized window, which is far too bright to put white-on-nothing
text over. So each master is dimmed at build time by a factor solved
from the palette: :func:`spacr.qt.theme.max_background_luma` returns the
brightest a bare window background may be before some role that gets
painted straight onto it (``fg``, ``fg_muted``, ``accent``, ``fg_dim``,
the status hues) drops below its WCAG minimum, and
:func:`solve_dim` scales the image in linear light until the brightest
:data:`TEXT_WINDOW`-sized region lands on that number less
:data:`SAFETY_MARGIN`.

The same solve runs again in :func:`render`, where it is a no-op on the
shipped masters and the whole guarantee for an image the user dropped
into ``~/.spacr/themes`` themselves.

:mod:`spacr.qt.space` now uses the same three functions —
:func:`exposure_target`, :func:`brightest_window` and :func:`solve_dim`
— on its generated sky, which for a long time was the one wallpaper in
the app that had never been measured against the rule the photographs
were held to. It cannot simply call :func:`solve_dim` on the finished
frame (that lands the sky on a solid black rectangle; the numbers are in
that module's docstring), so it applies the ceiling where the frame
actually breaks the rule and then measures the result here.

Scrimmed surfaces need no such treatment: :func:`spacr.qt.theme.contrast_report`
already judges every panel against a **pure white** background, which is
the worst case any photograph can present.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .space import (
    MAX_DIM, MIN_DIM, _clampi, cache_dir, screen_size,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Bumped whenever the rendering below changes output, so a cache written
#: by an older spaCR is not reused for a different-looking wallpaper.
CACHE_VERSION = 1

#: Aspect the shipped masters are cropped to. 16:10 is the widest shape
#: that still covers every display in :data:`spacr.qt.space.MAX_DIM`
#: without ever having to *upscale*: a 16:9 screen crops a band out of
#: it, a 4:3 screen crops columns, and both are centre crops of an
#: already annotation-free image.
MASTER_ASPECT = 1.6

#: Largest master shipped. Same ceiling as the generated sky, for the
#: same reason: past 4K the extra pixels cost megabytes and buy nothing
#: behind a UI.
MASTER_CAP = MAX_DIM

#: Region the legibility solve is run over, as (height, width) fractions
#: of the frame: one line of body text, roughly 540x48 px at 3840x2400.
#: Fractions rather than pixels because the metric then does not move
#: when the same wallpaper is rendered for a different screen.
#:
#: A *pixel* metric would be meaningless (every photograph has a white
#: pixel somewhere) and a whole-frame mean would be far too lenient —
#: it is a bright patch the size of a caption that makes a caption
#: unreadable.
TEXT_WINDOW = (0.02, 0.14)

#: Fraction of the palette's hard limit the exposure actually aims at.
#: Solving straight onto the limit leaves nothing for the three lossy
#: steps that follow the solve — JPEG encoding, the Lanczos resample to
#: the user's screen size, and rounding the measured region to a
#: 24-bit colour — and the first cut of this landed a wallpaper at
#: 2.99:1 against a 3.0:1 rule for exactly that reason. 10 % is about
#: two hundred times the observed drift and costs a barely perceptible
#: half-stop of exposure.
SAFETY_MARGIN = 0.90

#: Encoder settings for both the shipped masters and the per-screen
#: cache. JPEG, not PNG: these are photographs, and PNG is 4-6x larger
#: for pixels no one can tell apart. 4:2:0 chroma at q90 is
#: indistinguishable from lossless at 1:1 on this material — it was
#: compared crop by crop — and roughly a third smaller than 4:4:4.
JPEG_QUALITY = 90
JPEG_SUBSAMPLING = 2

#: Where the shipped masters live.
RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resources" / "themes"

#: A user can drop their own image here under any master's filename and
#: it wins over the bundled one — the cheapest possible "use my own
#: wallpaper" feature, and the escape hatch if they dislike these three.
USER_DIR_NAME = "themes"

#: Environment override for where masters are read from. Used by the
#: tests; also lets a packager relocate the assets.
ENV_MASTER_DIR = "SPACR_THEME_IMAGES"


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
# ``source``      original filename, for :func:`build_masters`
# ``file``        the shipped, cropped, dimmed master
# ``theme``       which theme's palette this wallpaper is judged against
# ``source_crop`` (x0, y0, x1, y1) fractions kept from the original —
#                 this is where burned-in annotation is removed
# ``focus``       vertical centre of the aspect crop, as a fraction
# ``title``       shown in Preferences
# ``annotation``  measured bounds of burned-in annotation in the
#                 original, when it has any. ``source_crop`` must not
#                 intersect it; the test suite checks that arithmetic
#                 rather than trusting the comment.
MASTERS: Dict[str, dict] = {
    "microtubules": {
        "source": "cell.png",
        "file": "microtubules.jpg",
        "theme": "cell",
        "source_crop": (0.0, 0.0, 1.0, 1.0),
        "focus": 0.46,
        "title": "Microtubule network (confocal)",
    },
    "filopodia": {
        "source": "cell_2.png",
        "file": "filopodia.jpg",
        "theme": "cell",
        # Bottom 10 % dropped: that is where the burned-in "5 um" scale
        # bar sits. The 3 % trim on the other three edges removes the
        # frame's darker border rows.
        "source_crop": (0.03, 0.025, 0.97, 0.90),
        "focus": 0.50,
        "title": "Filopodia (confocal)",
        "annotation": (0.8252, 0.9249, 0.9488, 0.9528),
    },
    "deep_field": {
        "source": "space_1.jpeg",
        "file": "deep_field.jpg",
        "theme": "space",
        "source_crop": (0.0, 0.0, 1.0, 1.0),
        "focus": 0.50,
        "title": "Galaxy deep field (photograph)",
    },
}

#: Variants of the Cell theme.
CELL_VARIANTS = ("microtubules", "filopodia")
DEFAULT_CELL_VARIANT = "microtubules"

#: Photographic variants offered by the *Space* theme, alongside the
#: procedural ones in :data:`spacr.qt.space.VARIANTS`. Kept out of that
#: tuple because those three index :data:`spacr.qt.space._VARIANT_MIX`
#: and this one has no mix — it is a photograph.
SPACE_PHOTO_VARIANTS = ("deep_field",)


def theme_for(key: str) -> Optional[str]:
    """Which theme's palette ``key`` is judged against, or ``None``."""
    entry = MASTERS.get(key)
    return entry["theme"] if entry else None


def title_for(key: str) -> str:
    """Human-readable name of a wallpaper, for Preferences."""
    entry = MASTERS.get(key)
    return entry["title"] if entry else str(key)


# ---------------------------------------------------------------------------
# Decode accounting — the performance claim, made assertable
# ---------------------------------------------------------------------------

_DECODES = 0


def decode_count() -> int:
    """How many times a master has been decoded this process.

    Every master decode goes through :func:`_open_master`, which bumps
    this. A resize or a repaint must not move it: the window paints a
    cached, screen-sized JPEG that Qt loaded once when the stylesheet was
    applied. :func:`reset_decode_count` exists so a test can measure a
    span rather than an absolute.
    """
    return _DECODES


def reset_decode_count() -> None:
    """Zero the decode counter."""
    global _DECODES
    _DECODES = 0


# ---------------------------------------------------------------------------
# Locating masters
# ---------------------------------------------------------------------------

def user_dir() -> Path:
    """Directory where a user may drop replacement masters."""
    return cache_dir().parent / USER_DIR_NAME


def master_dirs() -> Tuple[Path, ...]:
    """Directories searched for a master, most specific first."""
    override = os.environ.get(ENV_MASTER_DIR)
    if override:
        return (Path(override),)
    return (user_dir(), RESOURCE_DIR)


def master_path(key: str) -> Optional[Path]:
    """Absolute path of ``key``'s master image, or ``None``.

    ``None`` is a normal outcome, not an error: a source build with the
    assets stripped, or an unknown key. Callers fall back to the
    procedural sky or to the flat gradient.
    """
    entry = MASTERS.get(key)
    if entry is None:
        return None
    try:
        for directory in master_dirs():
            candidate = directory / entry["file"]
            if candidate.is_file():
                return candidate
    except Exception:
        return None
    return None


def available_keys() -> Tuple[str, ...]:
    """Keys whose master is actually present on this machine."""
    return tuple(k for k in MASTERS if master_path(k) is not None)


# ---------------------------------------------------------------------------
# Colour maths — WCAG luminance over numpy arrays
# ---------------------------------------------------------------------------

def _srgb_to_linear_lut() -> np.ndarray:
    c = np.arange(256, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92,
                    ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


_TO_LINEAR = _srgb_to_linear_lut()


def _linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return np.where(value <= 0.0031308, value * 12.92,
                    1.055 * np.power(value, 1.0 / 2.4) - 0.055)


def srgb_encode(linear: float) -> float:
    """Encode one linear-light value as an sRGB signal value in [0, 1].

    Public because :mod:`spacr.qt.space` needs it, and because the
    distinction it carries is the one that is easiest to get wrong here:
    every limit in this module — :func:`exposure_target`,
    :func:`spacr.qt.theme.max_background_luma` — is a *linear* relative
    luminance, while the sky generator's tone map emits an sRGB signal
    value. Space's 0.0586 limit is ``#444444``, not a 6 % signal; the two
    readings are a factor of 4.6 apart, which is the difference between
    a dimmed sun and a black rectangle.
    """
    return float(_linear_to_srgb(np.asarray(float(linear),
                                            dtype=np.float64)))


def linear_rgb(arr: np.ndarray) -> np.ndarray:
    """Map a uint8 (h, w, 3) image to linear-light floats in [0, 1]."""
    return _TO_LINEAR[arr]


def luminance_map(arr: np.ndarray) -> np.ndarray:
    """WCAG relative luminance of every pixel of a uint8 RGB array."""
    lin = linear_rgb(arr)
    return (0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1]
            + 0.0722 * lin[:, :, 2])


def _window_means(values: np.ndarray, box_h: int, box_w: int) -> np.ndarray:
    """Mean of ``values`` over every ``box_h`` x ``box_w`` window.

    Summed-area table, so the cost is one pass regardless of the window
    size — a sliding mean written as a loop over a 480x300 window is
    minutes, this is milliseconds.
    """
    integral = np.cumsum(np.cumsum(
        np.pad(values, ((1, 0), (1, 0))), axis=0), axis=1)
    total = (integral[box_h:, box_w:] - integral[:-box_h, box_w:]
             - integral[box_h:, :-box_w] + integral[:-box_h, :-box_w])
    return total / float(box_h * box_w)


def brightest_window(arr: np.ndarray,
                     window: Tuple[float, float] = TEXT_WINDOW
                     ) -> Tuple[float, str]:
    """Brightest ``window``-sized region of a uint8 RGB image.

    :returns: ``(luminance, "#rrggbb")`` — the region's mean relative
        luminance and its mean colour, which is what
        :func:`spacr.qt.theme.image_contrast_report` should be handed as
        the thing text has to be readable over.

    Averaging is done in *linear light*, so the returned colour's own
    luminance is exactly the returned number.
    """
    lin = linear_rgb(arr)
    luma = (0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1]
            + 0.0722 * lin[:, :, 2])
    h, w = luma.shape
    box_h = _clampi(round(h * window[0]), 1, h)
    box_w = _clampi(round(w * window[1]), 1, w)
    means = _window_means(luma, box_h, box_w)
    flat = int(np.argmax(means))
    row, col = divmod(flat, means.shape[1])
    value = float(means[row, col])
    patch = lin[row:row + box_h, col:col + box_w].reshape(-1, 3).mean(axis=0)
    rgb = np.clip(_linear_to_srgb(patch) * 255.0 + 0.5, 0, 255).astype(int)
    return value, "#%02x%02x%02x" % tuple(rgb)


def exposure_target(theme: str) -> float:
    """Luminance the brightest text-line-sized region is aimed at.

    The palette's hard WCAG limit, backed off by :data:`SAFETY_MARGIN`.
    """
    from .theme import max_background_luma
    return max(0.0, max_background_luma(theme)) * SAFETY_MARGIN


def solve_dim(measured: float, target: float) -> float:
    """Exposure factor taking ``measured`` luminance down to ``target``.

    Closed form rather than the bisection :mod:`spacr.qt.space` needs,
    because scaling linear light scales relative luminance by exactly
    the same factor — there is no tone curve in the way. Never brightens:
    an image already dark enough is left alone.
    """
    if measured <= 0.0:
        return 1.0
    return float(min(1.0, max(0.0, target) / measured))


def _dim_lut(factor: float) -> np.ndarray:
    """256-entry uint8 LUT applying ``factor`` in linear light.

    A LUT rather than float arithmetic on the image: a 3840x2400 float32
    RGB buffer is 88 MB and this is 256 bytes, for identical output.
    """
    return np.clip(_linear_to_srgb(_TO_LINEAR * factor) * 255.0 + 0.5,
                   0, 255).astype(np.uint8)


def dim(arr: np.ndarray, factor: float) -> np.ndarray:
    """Return ``arr`` darkened by ``factor`` in linear light."""
    if factor >= 1.0:
        return arr
    return _dim_lut(factor)[arr]


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def rects_overlap(a: Tuple[float, float, float, float],
                  b: Tuple[float, float, float, float]) -> bool:
    """True when two ``(x0, y0, x1, y1)`` rectangles share any area."""
    return (a[0] < b[2] and b[0] < a[2]
            and a[1] < b[3] and b[1] < a[3])


#: Block size and fill fraction used by :func:`solid_annotation_blocks`.
#: A scale bar is at least a couple of hundred pixels long and tens tall,
#: so it saturates several 32 px blocks; nothing in a fluorescence frame
#: or a deep field fills a third of one at peak brightness.
ANNOTATION_BLOCK = 32
ANNOTATION_FILL = 0.35


def solid_annotation_blocks(arr: np.ndarray) -> int:
    """Count blocks that look like burned-in annotation.

    Annotation — a scale bar, a label, a timestamp — is *solid,
    achromatic and at the frame's peak brightness*, and it covers whole
    blocks. Real content in these images does not: the brightest galaxy
    core in the deep field fills 15 % of a block, a drawn scale bar
    fills 63 %.

    The thresholds are relative to the image's own 99.99th percentile
    rather than absolute, because these masters are deliberately exposed
    differently — the microtubule frame peaks at 218, not 255, and an
    absolute "≥ 190 is white" test would go blind on it.

    :returns: number of blocks that look like annotation. Zero for every
        shipped master; the tests also check it is *non*-zero on the same
        masters with a bar drawn on, so a detector that has quietly
        stopped detecting cannot pass.
    """
    if arr.size == 0 or min(arr.shape[:2]) < ANNOTATION_BLOCK:
        return 0
    lo = arr.min(axis=2).astype(np.float32)
    hi = arr.max(axis=2).astype(np.float32)
    peak = float(np.percentile(hi, 99.99))
    if peak <= 0.0:
        return 0
    solid = ((lo >= 0.82 * peak) & ((hi - lo) <= 0.12 * peak)).astype(np.float32)
    block = ANNOTATION_BLOCK
    rows = solid.shape[0] // block
    cols = solid.shape[1] // block
    tiles = solid[:rows * block, :cols * block].reshape(
        rows, block, cols, block).mean(axis=(1, 3))
    return int((tiles > ANNOTATION_FILL).sum())


def cover_box(src_w: int, src_h: int, out_w: int, out_h: int,
              focus: float = 0.5) -> Tuple[int, int, int, int]:
    """Largest sub-rectangle of the source that has the output's aspect.

    "Cover", never "contain": the result always fills the target, so the
    stylesheet — which centres the image without repeating it — can
    never end up letterboxing it into bands of flat colour.

    :param focus: vertical centre of the crop as a fraction of the
        source height. Clamped so the box stays inside the frame.
    """
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    out_w = max(1, int(out_w))
    out_h = max(1, int(out_h))
    if src_w * out_h > out_w * src_h:
        box_h = src_h
        box_w = _clampi(round(src_h * out_w / out_h), 1, src_w)
    else:
        box_w = src_w
        box_h = _clampi(round(src_w * out_h / out_w), 1, src_h)
    left = _clampi(round(src_w * 0.5 - box_w / 2.0), 0, src_w - box_w)
    top = _clampi(round(src_h * focus - box_h / 2.0), 0, src_h - box_h)
    return (left, top, left + box_w, top + box_h)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def _open_master(path: Path, hint: Optional[Tuple[int, int]] = None):
    """Decode a master into a PIL RGB image. **The only decode site.**

    :param hint: target size. For a JPEG this is passed to
        ``Image.draft``, which lets libjpeg decode at 1/2, 1/4 or 1/8
        scale directly — the difference between holding 281 MB and
        holding 18 MB when a full-size original is used as the master.
    """
    global _DECODES
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None       # these are legitimately huge
    with Image.open(path) as handle:
        if hint is not None:
            try:
                handle.draft("RGB", (max(1, hint[0]), max(1, hint[1])))
            except Exception:
                pass                    # PNG has no draft mode; fine
        image = handle.convert("RGB")
    _DECODES += 1
    return image


# ---------------------------------------------------------------------------
# Rendering a background for one screen size
# ---------------------------------------------------------------------------

def _probe(image, long_edge: int = 480) -> np.ndarray:
    """A box-averaged thumbnail of ``image``, for measurement only.

    Every measurement in this module is a mean over a region hundreds of
    pixels across, and a box-averaged thumbnail answers those to several
    decimals for a fraction of the memory — the alternative is a 221 MB
    float array to learn one number.
    """
    from PIL import Image
    scale = max(1, int(max(image.size) // max(1, long_edge)))
    if scale > 1:
        image = image.resize((max(1, image.width // scale),
                              max(1, image.height // scale)), Image.BOX)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def render(key: str, width: int, height: int):
    """Render ``key``'s wallpaper at exactly ``width`` x ``height``.

    :returns: a PIL image, or ``None`` when the master is missing.

    The crop and the resample happen in a single ``Image.resize`` call
    with a ``box``, so no intermediate full-resolution crop is
    materialised.

    The dim solve runs here as well as in :func:`build_master`, and it
    is deliberately not redundant: on the shipped masters it resolves to
    a no-op because they are already at the limit, but a user who drops
    their own photograph into ``~/.spacr/themes`` gets the same
    guarantee without having to know it exists. Solving costs one pass
    over a 480 px thumbnail plus one 256-entry lookup table.
    """
    path = master_path(key)
    if path is None:
        return None
    from PIL import Image
    width = _clampi(width, MIN_DIM[0], MAX_DIM[0])
    height = _clampi(height, MIN_DIM[1], MAX_DIM[1])
    source = _open_master(path, hint=(width, height))
    box = cover_box(source.width, source.height, width, height)
    image = source.resize((width, height), Image.LANCZOS, box=box,
                          reducing_gap=2.0)
    source.close()
    del source

    measured, _ = brightest_window(_probe(image))
    factor = solve_dim(measured, exposure_target(MASTERS[key]["theme"]))
    if factor >= 1.0:
        return image
    arr = np.asarray(image, dtype=np.uint8)
    image.close()
    return Image.fromarray(dim(arr, factor))


# ---------------------------------------------------------------------------
# Disk cache — same directory and contract as the generated sky
# ---------------------------------------------------------------------------

def cache_name(key: str, width: int, height: int) -> str:
    """Return the versioned cache filename for a photographic background."""
    return f"photo-{key}-{width}x{height}-v{CACHE_VERSION}.jpg"


def _qt_can_read(path: Path, width: int, height: int) -> bool:
    """True when Qt decodes ``path`` at the expected size."""
    try:
        from PySide6.QtGui import QImage
        probe = QImage()
        if not probe.load(str(path)):
            return False
        return probe.width() == width and probe.height() == height
    except Exception:
        return False


def _load_cached(path: Path, width: int, height: int) -> bool:
    """True when ``path`` already holds a usable image of that size.

    A truncated or garbage file — interrupted write, half-synced home
    directory — must regenerate rather than raise, and a file of the
    wrong size must regenerate rather than be stretched by the
    stylesheet.
    """
    try:
        if not path.is_file() or path.stat().st_size < 128:
            return False
        return _qt_can_read(path, width, height)
    except Exception:
        return False


def _write(image, path: Path, width: int, height: int) -> Optional[Path]:
    """Write ``image`` to ``path`` atomically, as JPEG if Qt can read it.

    Falls back to PNG when the Qt build has no JPEG plugin: a wallpaper
    Qt cannot decode is worse than a wallpaper that is four times
    larger, and silently regenerating a JPEG on every launch because
    ``_load_cached`` can never validate it would be worse still.
    """
    tmp = path.with_suffix(path.suffix + ".part")
    image.save(tmp, "JPEG", quality=JPEG_QUALITY,
               subsampling=JPEG_SUBSAMPLING, optimize=True)
    if not _qt_can_read(tmp, width, height):
        image.save(tmp, "PNG")
        if not _qt_can_read(tmp, width, height):
            tmp.unlink(missing_ok=True)
            return None
    os.replace(tmp, path)
    return path


def background_path(key: str, width: int = 0, height: int = 0,
                    regenerate: bool = False) -> Optional[Path]:
    """On-disk path of ``key``'s wallpaper at this size, rendering if needed.

    Returns ``None`` — never raises — when it cannot be produced: no
    master, a read-only home directory, no image encoder. Callers treat
    that as "use the procedural sky" or "use the flat gradient", so a
    failure here costs some prettiness and nothing else.
    """
    try:
        if width <= 0 or height <= 0:
            # `screen_size` is what applies the MIN_BACKGROUND floor —
            # the stylesheet centres the image without repeating it, so
            # a background narrower than the window would letterbox into
            # bands of flat colour. An explicit size is honoured as
            # given, which is what makes this testable at small sizes.
            width, height = screen_size()
        width = _clampi(width, MIN_DIM[0], MAX_DIM[0])
        height = _clampi(height, MIN_DIM[1], MAX_DIM[1])
        directory = cache_dir()
        path = directory / cache_name(key, width, height)
        if not regenerate and _load_cached(path, width, height):
            return path
        image = render(key, width, height)
        if image is None:
            return None
        directory.mkdir(parents=True, exist_ok=True)
        return _write(image, path, width, height)
    except Exception:
        return None


def clear_cache() -> int:
    """Delete every cached photo background. Returns the number removed."""
    removed = 0
    try:
        for pattern in ("photo-*.jpg", "photo-*.png"):
            for entry in cache_dir().glob(pattern):
                try:
                    entry.unlink()
                    removed += 1
                except OSError:
                    pass
    except Exception:
        pass
    return removed


# ---------------------------------------------------------------------------
# Measured legibility
# ---------------------------------------------------------------------------

def master_array(key: str) -> Optional[np.ndarray]:
    """The shipped master as a uint8 (h, w, 3) array, or ``None``.

    Decoded at 1/8 scale where the format allows it and box-averaged the
    rest of the way — see :func:`_probe`.
    """
    path = master_path(key)
    if path is None:
        return None
    image = _open_master(path, hint=(MASTER_CAP[0] // 8, MASTER_CAP[1] // 8))
    arr = _probe(image)
    image.close()
    return arr


def legibility_of(arr: np.ndarray, theme: str,
                  key: str = "") -> dict:
    """Measure how readable a wallpaper's pixels actually are.

    Everything in the returned dict comes from the real image data:

    ``brightest``   luminance of the worst :data:`TEXT_WINDOW`-sized region
    ``color``       that region's mean colour, as ``#rrggbb``
    ``limit``       the most that region is allowed to be, from the palette
    ``target``      what the exposure aimed at — ``limit`` less the margin
    ``passes``      whether it is within the limit
    ``failures``    every WCAG rule the theme fails **over that region**

    Takes an array rather than a registry key so the *generated* sky can
    be held to the identical measurement — :func:`spacr.qt.space.legibility`
    is this function over a rendered frame. Until it was, the sky was the
    one background in the app that had never been measured, and it was
    8-14x over the limit.
    """
    from .theme import image_contrast_failures, max_background_luma
    value, color = brightest_window(arr)
    limit = max_background_luma(theme)
    return {
        "key": key,
        "theme": theme,
        "brightest": value,
        "color": color,
        "limit": limit,
        "target": exposure_target(theme),
        "passes": value <= limit,
        "failures": image_contrast_failures(theme, color),
    }


def legibility(key: str) -> Optional[dict]:
    """Measure how readable ``key``'s master actually is.

    See :func:`legibility_of` for the returned dict. ``None`` when the
    master is not installed.
    """
    arr = master_array(key)
    if arr is None:
        return None
    return legibility_of(arr, MASTERS[key]["theme"], key=key)


# ---------------------------------------------------------------------------
# Build-time: originals -> shipped masters
# ---------------------------------------------------------------------------

def solve_image_file(path, theme: str, fmt: str = "JPEG") -> bool:
    """Exposure-solve an image file *in place*. ``True`` when it is legible.

    For pixels that arrive at runtime rather than in the wheel — today
    that is the optional NASA/ESA download in
    :func:`spacr.qt.space.download_nasa_background`. :func:`render`
    cannot help there: that file is handed to the stylesheet directly,
    at whatever size it arrived, so the solve has to happen to the file.

    ``False`` — never an exception — when it cannot be read, solved or
    rewritten. A caller must then **refuse** the image rather than
    install it. A wallpaper that is not bounded, under a theme whose
    scrims are solved against the bound, is precisely the failure
    :data:`spacr.qt.theme.EXPOSURE_BOUNDED_THEMES` warns about: panels
    thinned to what a dark sky can carry, with a solar flare behind them.
    """
    try:
        from PIL import Image
        path = Path(path)
        Image.MAX_IMAGE_PIXELS = None       # NASA masters are huge
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        try:
            measured, _ = brightest_window(_probe(image))
            factor = solve_dim(measured, exposure_target(theme))
            if factor >= 1.0:
                return True
            arr = dim(np.asarray(image, dtype=np.uint8), factor)
        finally:
            image.close()
        Image.fromarray(arr).save(path, fmt, quality=JPEG_QUALITY,
                                  subsampling=JPEG_SUBSAMPLING, optimize=True)
        return True
    except Exception:
        return False


def build_master(key: str, src_dir, dst_dir) -> Optional[Path]:
    """Turn one original into the master that ships in the wheel.

    Crop away burned-in annotation, take the :data:`MASTER_ASPECT` crop
    around the subject, cap at :data:`MASTER_CAP`, dim until the
    brightest text-line-sized region satisfies the theme's palette, and
    write JPEG.

    Kept in the shipped module rather than a build script so the assets
    can be re-derived from the originals by anyone who has them, and so
    the crop rectangles live next to the code that documents why they
    are where they are.

    :returns: the written path, or ``None`` if the original is missing.
    """
    from PIL import Image
    entry = MASTERS[key]
    source = Path(src_dir) / entry["source"]
    if not source.is_file():
        return None

    image = _open_master(source, hint=(MASTER_CAP[0] * 2, MASTER_CAP[1] * 2))
    width, height = image.size
    x0, y0, x1, y1 = entry["source_crop"]
    image = image.crop((int(x0 * width), int(y0 * height),
                        int(x1 * width), int(y1 * height)))

    out_w = MASTER_CAP[0]
    out_h = int(round(MASTER_CAP[0] / MASTER_ASPECT))
    box = cover_box(image.width, image.height, out_w, out_h,
                    focus=entry["focus"])
    box_w, box_h = box[2] - box[0], box[3] - box[1]
    # Never upscale here. ``cell.png`` is only 2048 px wide, and
    # inventing pixels at build time would bake a soft image into the
    # wheel for every user including the ones whose screen is 1920.
    scale = min(1.0, out_w / box_w, out_h / box_h)
    image = image.resize((max(1, int(round(box_w * scale))),
                          max(1, int(round(box_h * scale)))),
                         Image.LANCZOS, box=box, reducing_gap=2.0)

    measured, _ = brightest_window(_probe(image))
    factor = solve_dim(measured, exposure_target(entry["theme"]))
    if factor < 1.0:
        image = Image.fromarray(dim(np.asarray(image, dtype=np.uint8),
                                    factor))

    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / entry["file"]
    image.save(out, "JPEG", quality=JPEG_QUALITY,
               subsampling=JPEG_SUBSAMPLING, optimize=True)
    return out


def build_masters(src_dir, dst_dir=None) -> Dict[str, Optional[Path]]:
    """Build every master in :data:`MASTERS`. See :func:`build_master`."""
    dst_dir = RESOURCE_DIR if dst_dir is None else dst_dir
    return {key: build_master(key, src_dir, dst_dir) for key in MASTERS}

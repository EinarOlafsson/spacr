"""
Central icon lookup for the spacr Qt GUI.

Two sources of icons, both made theme-aware here:

* **qtawesome glyphs** — vector, painted in whatever colour we ask for.
  Wrapped so callers stay decoupled from Font Awesome glyph names, and
  degrading to an empty ``QIcon()`` when qtawesome isn't installed so
  the UI still renders with text-only buttons.
* **bundled PNGs** in ``spacr/resources/icons`` — flat monochrome
  artwork baked at a fixed colour.

The bundled PNGs were *theme-blind*, and it showed: ``convert.png``
(Format Converter) shipped as solid black with an alpha mask, so on the
black home page it rendered as nothing at all. The other twenty-odd were
solid white, which is the same bug pointed the other way — invisible on
the light theme, nobody had noticed because nobody used it. A third
theme made this unavoidable, so :func:`themed_qimage` now re-inks every
bundled PNG for the active theme. **This is what makes the artwork
swappable**: seventeen icons were redrawn and reinstalled without a line
of change here, because nothing in this module knows what any of them
look like.

* the artwork's ink polarity is detected from its own alpha-weighted
  mean luminance, so black-on-transparent and white-on-transparent are
  both handled without a per-file table;
* the ink is remapped onto a band running from the theme's foreground
  down to a ``veil`` colour computed to clear :data:`MIN_ICON_CONTRAST`
  against the *hardest* surface the icon can land on, so internal shading
  survives instead of being flattened to a silhouette;
* genuinely polychrome artwork keeps its hue and is only re-levelled.

:func:`icon_contrast` exposes the measured ratio so the test suite can
assert visibility numerically rather than checking that a file exists.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple

from PySide6.QtGui import QIcon

from .theme import (
    contrast_ratio, effective_surface, palette_for, relative_luminance,
)

#: WCAG 1.4.11 (non-text contrast) minimum for a UI graphic.
MIN_ICON_CONTRAST = 3.0

#: Surfaces a bundled icon can end up sitting on. The veil is solved
#: against whichever of these is hardest for the theme's ink.
ICON_SURFACES = ("bg", "surface", "surface_alt", "surface_hi")

#: Max per-pixel chroma (max channel − min channel, 0-255) for artwork to
#: count as monochrome and be re-inked outright. Every bundled spaCR
#: icon measures ≤ 13; the flow-chart diagram measures 150 and is
#: correctly treated as polychrome.
CHROMA_MONO_MAX = 32.0

#: Fraction of the luminance range an icon's RGB must span before it is
#: treated as carrying shading rather than being a flat mask. Below
#: this, RGB is noise from the exporter — several bundled icons vary by
#: ~0.03 across their "solid white" fill, and stretching that across
#: the ink band paints visible banding into a flat glyph.
MIN_TONAL_RANGE = 0.12

#: Longest edge an icon is processed at. Icons are drawn into 16-52 px
#: slots; anything past 512 px is pure cost.
MAX_WORK_SIZE = 512

#: Where the bundled PNGs live.
RESOURCE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "resources", "icons"))


@lru_cache(maxsize=128)
def _try_qta():
    """Return the ``qtawesome`` module, or ``None`` when it isn't installed."""
    try:
        import qtawesome as qta
        return qta
    except Exception:
        return None


def active_theme() -> str:
    """The theme icons should be drawn for, resolved from preferences.

    Falls back to ``"dark"`` if preferences can't be read at all —
    icon lookup must never be the thing that stops the GUI booting.
    """
    try:
        from .preferences import resolve_effective_theme
        return resolve_effective_theme()
    except Exception:
        return "dark"


def _theme_palette(theme: Optional[str]) -> dict:
    return palette_for(theme or active_theme())


# ---------------------------------------------------------------------------
# qtawesome glyphs
# ---------------------------------------------------------------------------

def icon(name: str, color: Optional[str] = None, size: int = 16,
         theme: Optional[str] = None) -> QIcon:
    """Return a QIcon for the named glyph, or an empty QIcon fallback.

    `name` is a semantic key (e.g. "open", "run", "brush") mapped to a
    Font Awesome glyph. Unknown names fall back to a puzzle piece. The
    default fill follows the active theme rather than the dark palette,
    which is why a light-theme sidebar no longer draws pale-grey icons
    on white.
    """
    qta = _try_qta()
    if qta is None:
        return QIcon()
    glyph = _NAME_TO_GLYPH.get(name, "fa5s.puzzle-piece")
    fill = color or _theme_palette(theme)["fg_muted"]
    try:
        return qta.icon(glyph, color=fill)
    except Exception:
        return QIcon()


def accent_icon(name: str, theme: Optional[str] = None) -> QIcon:
    """Icon painted in the accent color (used for primary buttons)."""
    return icon(name, color=_theme_palette(theme)["accent"], theme=theme)


def contrast_icon(name: str, theme: Optional[str] = None) -> QIcon:
    """Icon painted for use inside a filled (PrimaryButton) button,
    where the button background IS the accent fill."""
    return icon(name, color=_theme_palette(theme)["button_accent_ink"],
                theme=theme)


# ---------------------------------------------------------------------------
# Bundled PNGs — re-inked per theme
# ---------------------------------------------------------------------------

def _blend(a: str, b: str, t: float) -> str:
    """Linear blend from colour ``a`` (t=0) to colour ``b`` (t=1)."""
    def ch(c):
        c = c.lstrip("#")
        return [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    ca, cb = ch(a), ch(b)
    return "#%02x%02x%02x" % tuple(
        int(round(x + (y - x) * t)) for x, y in zip(ca, cb))


def hardest_surface(theme: str) -> str:
    """The surface colour an icon has the least contrast against."""
    palette = _theme_palette(theme)
    ink = palette["fg"]
    return min((effective_surface(theme, role) for role in ICON_SURFACES),
               key=lambda s: contrast_ratio(ink, s))


@lru_cache(maxsize=16)
def veil_color(theme: str) -> str:
    """Dimmest ink allowed in a themed icon.

    Solved, not guessed: bisect the blend from the hardest surface
    toward the theme foreground until the contrast crosses
    :data:`MIN_ICON_CONTRAST`. That way the shadow end of an icon's
    tonal range is still a visible shape, and the answer tracks the
    palette instead of being a magic grey someone eyeballed once.
    """
    palette = _theme_palette(theme)
    ink = palette["fg"]
    surface = hardest_surface(theme)
    if contrast_ratio(surface, ink) < MIN_ICON_CONTRAST:
        return ink            # palette can't do better; use full ink
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = (lo + hi) / 2.0
        if contrast_ratio(_blend(surface, ink, mid), surface) < MIN_ICON_CONTRAST:
            lo = mid
        else:
            hi = mid
    return _blend(surface, ink, hi)


def _load_rgba(path: str):
    """Load a PNG as an (h, w, 4) float array, or ``None`` if unreadable.

    Downscaled to :data:`MAX_WORK_SIZE` first. Some bundled assets are
    enormous for icon artwork — ``logo_spacr.png`` is 3334x3334, which
    is 356 MB as a float64 RGBA array and about half a second to
    re-ink, for something drawn into a 52 px slot. 512 px is four times
    the largest slot at 2x device pixel ratio.
    """
    try:
        import numpy as np
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGBA")
            if max(im.size) > MAX_WORK_SIZE:
                # reducing_gap makes PIL box-reduce by an integer factor
                # first and only then resample — ~6x faster than going
                # straight to LANCZOS from 3334 px, same result.
                im.thumbnail((MAX_WORK_SIZE, MAX_WORK_SIZE),
                             Image.LANCZOS, reducing_gap=2.0)
            return np.asarray(im, dtype=np.float64)
    except Exception:
        return None


def _file_stamp(path: str):
    """``(path, mtime, size)`` — the cache key for a decoded icon.

    Some bundled assets are large for icon artwork (``activation.png``
    is 1024x1024, 1.5 MB on disk), and re-decoding one every time a
    sidebar rebuilds is pure waste. Keying on mtime+size means an
    artwork swap invalidates the entry on its own.
    """
    try:
        stat = os.stat(path)
        return (path, stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (path, 0, 0)


def _hex_to_array(color: str):
    import numpy as np
    text = color.lstrip("#")
    return np.array([int(text[i:i + 2], 16) for i in (0, 2, 4)],
                    dtype=np.float64)


def carries_tonal_structure(rgba) -> bool:
    """True when an icon's RGB channels carry shading worth preserving.

    Measured, not assumed. Every bundled spaCR icon is **a monochrome
    mask**: the alpha channel holds the entire shape and the RGB is a
    uniform fill (white for most of them, black for ``convert.png``).
    For those, the RGB carries no information at all and the right
    answer is to paint the alpha mask in the theme's ink.

    A couple of assets (``umap.png``, ``activation.png``) genuinely do
    put shading in RGB, and flattening those to a silhouette would
    destroy the picture. The discriminator is whether the visible-pixel
    luminance spans a meaningful fraction of the range — 2 % of
    variation is noise from whatever exported the file, not shading.
    """
    import numpy as np
    alpha = rgba[:, :, 3] / 255.0
    visible = alpha > 0.02
    if not visible.any():
        return False
    rgb = rgba[:, :, :3]
    lum = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1]
           + 0.0722 * rgb[:, :, 2])[visible] / 255.0
    lo, hi = np.percentile(lum, (2.0, 98.0))
    return float(hi - lo) >= MIN_TONAL_RANGE


def reink(rgba, theme: str):
    """Re-ink an RGBA array for ``theme``. Returns a uint8 RGBA array.

    The **alpha channel is the shape**, always. RGB is consulted only
    when :func:`carries_tonal_structure` says it holds real shading, so
    swapping in different artwork later cannot break this — a redrawn
    monochrome mask keeps working with no code change.

    :param rgba: (h, w, 4) float array, channels 0-255.
    :param theme: theme name.
    """
    import numpy as np

    palette = _theme_palette(theme)
    ink = palette["fg"]
    veil = veil_color(theme)

    alpha = rgba[:, :, 3] / 255.0
    rgb = rgba[:, :, :3]
    visible = alpha > 0.02
    out = rgba.copy()
    if not visible.any():
        return out.astype(np.uint8)

    ink_rgb, veil_rgb = _hex_to_array(ink), _hex_to_array(veil)
    lum = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1]
           + 0.0722 * rgb[:, :, 2])

    if not carries_tonal_structure(rgba):
        # Pure mask: paint it flat in the theme ink and let alpha —
        # including its antialiased edges — do all the shaping.
        new = np.broadcast_to(ink_rgb, rgb.shape).copy()
    else:
        weights = alpha[visible]
        mean_lum = float((lum[visible] * weights).sum()
                         / max(weights.sum(), 1e-9))
        # Which end of the tonal range is the drawing? A black glyph on
        # transparent and a white glyph on transparent are the same
        # picture with opposite polarity; guessing wrong inverts it.
        ink_is_bright = mean_lum > 127.5
        t = lum / 255.0 if ink_is_bright else 1.0 - lum / 255.0
        lo = float(np.percentile(t[visible], 2))
        hi = float(np.percentile(t[visible], 98))
        t = np.clip((t - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

        chroma = rgb.max(axis=2) - rgb.min(axis=2)
        polychrome = float(np.percentile(chroma[visible], 98)) > CHROMA_MONO_MAX
        if polychrome:
            # Keep the hue: scale each pixel toward the target luminance
            # rather than replacing its colour outright.
            ink_l = relative_luminance(ink)
            veil_l = relative_luminance(veil)
            target = (veil_l + (ink_l - veil_l) * t) * 255.0
            scale = target / np.maximum(lum, 1.0)
            new = np.clip(rgb * scale[:, :, None], 0.0, 255.0)
            # Where the source was pure black there is no hue to
            # preserve, so lift it to the neutral target instead of
            # multiplying zero.
            flat = lum < 1.0
            new[flat] = np.clip(target[flat], 0.0, 255.0)[:, None]
        else:
            new = veil_rgb[None, None, :] + \
                (ink_rgb - veil_rgb)[None, None, :] * t[:, :, None]

    out[:, :, :3] = np.where(visible[:, :, None], new, rgb)
    return np.clip(out, 0, 255).astype(np.uint8)



#: Where re-inked icons are kept between launches. Honours
#: ``$SPACR_ICON_CACHE`` so tests and read-only homes can redirect it,
#: matching what `spacr.qt.space.cache_dir` does for backgrounds.
ENV_ICON_CACHE = "SPACR_ICON_CACHE"

#: Bumped when the re-inking maths changes. A cached icon from an older
#: formula is WRONG rather than merely stale, and a version in the name is
#: cheaper than trying to detect that.
ICON_CACHE_VERSION = 1


def icon_cache_dir() -> Path:
    """Directory holding re-inked icons, one PNG per (file, theme)."""
    override = os.environ.get(ENV_ICON_CACHE)
    if override:
        return Path(override)
    return Path.home() / ".spacr" / "icons"


def _cache_path(stamp, theme: str) -> Path:
    """Cache filename for one (file, mtime, size) at one theme.

    The stamp carries mtime and size, so an edited or replaced icon gets a
    different name and the old entry is simply never read again. No
    invalidation logic, and none to get wrong.
    """
    key = f"{stamp[0]}|{stamp[1]}|{stamp[2]}|{theme}|v{ICON_CACHE_VERSION}"
    digest = hashlib.sha1(key.encode("utf-8", "replace")).hexdigest()[:20]
    return icon_cache_dir() / f"{Path(stamp[0]).stem}-{digest}.png"


def _read_cached_icon(path: Path):
    """The cached re-inked RGBA at ``path``, or None.

    Any failure returns None and the caller re-renders. A cache is an
    optimisation, and one that can break icon loading is a liability --
    INVARIANTS 10.
    """
    try:
        if not path.is_file():
            return None
        import numpy as np
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGBA"), dtype=np.uint8)
    except Exception:
        return None


def _write_cached_icon(path: Path, array) -> None:
    """Store a re-inked icon, atomically. Failure is silent by design."""
    try:
        from PIL import Image
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a half-written PNG left by a crash or a full
        # disk would be read as a corrupt icon on every later launch.
        tmp = path.with_suffix(".part")
        # `format=` explicitly: PIL infers it from the extension, and the
        # temp name ends in `.part`, which it does not recognise. Without
        # this every write raised and the silent `except` below swallowed
        # it -- a cache that logged nothing and stored nothing.
        Image.fromarray(array, "RGBA").save(tmp, format="PNG", optimize=False)
        os.replace(tmp, path)
    except Exception:
        pass


@lru_cache(maxsize=192)
def _themed_array(stamp, theme: str):
    """Re-inked RGBA for one (file, theme).

    Cached twice over. The `lru_cache` covers repeats within one run -- the
    home grid asks for 160 icons that are only 50 distinct files -- and the
    PNG on disk covers repeats ACROSS runs, which the lru_cache cannot.

    That second cache is worth having: re-inking 50 icons cold was measured
    at 2.8 s of a 4.8 s startup, because the source art is large
    (`logo_spacr.png` is 3334x3334) and every launch was paying full decode
    plus LANCZOS downscale plus re-ink. Reading the finished PNGs back is
    19x faster and the files are 42x smaller than the arrays -- 0.5 MB for
    20 icons -- and the round trip is lossless, which is asserted by
    `tests/qt/test_icon_cache.py`.
    """
    path = _cache_path(stamp, theme)
    cached = _read_cached_icon(path)
    if cached is not None:
        return cached
    rgba = _load_rgba(stamp[0])
    if rgba is None:
        return None
    inked = reink(rgba, theme)
    if inked is not None:
        _write_cached_icon(path, inked)
    return inked


def themed_array(path: str, theme: Optional[str] = None):
    """Re-inked ``(h, w, 4)`` uint8 array for ``path``, or ``None``."""
    return _themed_array(_file_stamp(path), theme or active_theme())


def themed_qimage(path: str, theme: Optional[str] = None):
    """Return the bundled PNG at ``path``, re-inked for ``theme``.

    ``None`` when the file can't be read. Returns a ``QImage``, which
    (unlike ``QPixmap``) needs no running QGuiApplication, so this is
    safe to call from a headless test.
    """
    import numpy as np
    from PySide6.QtGui import QImage

    arr = themed_array(path, theme)
    if arr is None:
        return None
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888)
    return img.copy()          # detach from the numpy buffer


def themed_pixmap(path: str, theme: Optional[str] = None):
    """:func:`themed_qimage` as a ``QPixmap``, or ``None``."""
    from PySide6.QtGui import QPixmap
    img = themed_qimage(path, theme)
    if img is None:
        return None
    pix = QPixmap.fromImage(img)
    return None if pix.isNull() else pix


def icon_ink_color(path: str, theme: Optional[str] = None) -> Optional[str]:
    """Alpha-weighted mean colour of the re-inked artwork, as hex.

    This is the colour the eye integrates when the icon is small, which
    is what makes it the right thing to measure contrast on.
    """
    import numpy as np

    themed = themed_array(path, theme)
    if themed is None:
        return None
    arr = themed.astype(np.float64)
    alpha = arr[:, :, 3] / 255.0
    total = float(alpha.sum())
    if total <= 0.0:
        return None
    mean = [(arr[:, :, c] * alpha).sum() / total for c in range(3)]
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in mean)


def icon_contrast(path: str, theme: Optional[str] = None) -> float:
    """Worst-case contrast of a themed icon against the theme's surfaces.

    ``0.0`` when the file can't be read or is fully transparent.
    """
    theme = theme or active_theme()
    ink = icon_ink_color(path, theme)
    if ink is None:
        return 0.0
    return min(contrast_ratio(ink, effective_surface(theme, role))
               for role in ICON_SURFACES)


def bundled_icon_paths() -> Tuple[str, ...]:
    """Every bundled PNG, sorted. Used by the theme-visibility test."""
    try:
        names = sorted(n for n in os.listdir(RESOURCE_DIR)
                       if n.lower().endswith(".png"))
    except OSError:
        return ()
    return tuple(os.path.join(RESOURCE_DIR, n) for n in names)


def bundled_icon_path(key: str, override: Optional[str] = None
                      ) -> Optional[str]:
    """Resolve an app key to its bundled PNG, or ``None``.

    :param override: explicit filename to try first. The key → filename
        table lives in :mod:`spacr.qt.app` next to the app registry it
        describes; this module only knows how to *render* what it's
        pointed at.
    """
    candidates = [override] if override else []
    candidates += [f"{key}.png", f"{key.replace('_', ' ')}.png"]
    for candidate in candidates:
        path = os.path.join(RESOURCE_DIR, candidate)
        if os.path.isfile(path):
            return path
    return None


def app_icon(key: str, override: Optional[str] = None,
             theme: Optional[str] = None) -> QIcon:
    """Icon for an app key: the bundled PNG re-inked for the theme,
    falling back to the themed qtawesome glyph."""
    path = bundled_icon_path(key, override)
    if path is not None:
        pix = themed_pixmap(path, theme)
        if pix is not None:
            return QIcon(pix)
    return icon(key, theme=theme)


# Semantic name → Font Awesome glyph. Keep names short + generic so
# callers don't have to think about the icon library.
_NAME_TO_GLYPH = {
    # File / source
    "open":            "fa5s.folder-open",
    "folder":          "fa5s.folder",
    "file":            "fa5s.file",
    "save":            "fa5s.save",
    "import":          "fa5s.file-import",
    "export":          "fa5s.file-export",
    "report":          "fa5s.file-alt",
    "invasion":        "fa5s.sign-in-alt",
    # Navigation
    "prev":            "fa5s.chevron-left",
    "next":            "fa5s.chevron-right",
    "up":              "fa5s.chevron-up",
    "down":            "fa5s.chevron-down",
    "home":            "fa5s.home",
    "skip":            "fa5s.forward",
    # Editing
    "brush":           "fa5s.paint-brush",
    "erase":           "fa5s.eraser",
    "erase_object":    "fa5s.trash-alt",
    "wand":            "fa5s.magic",
    "wand_add":        "fa5s.plus-circle",
    "wand_erase":      "fa5s.minus-circle",
    "zoom":            "fa5s.search-plus",
    "zoom_reset":      "fa5s.compress-arrows-alt",
    "undo":            "fa5s.undo",
    "redo":            "fa5s.redo",
    "fill":            "fa5s.fill",
    "invert":          "fa5s.adjust",
    "relabel":         "fa5s.tags",
    "remove":          "fa5s.filter",
    "clear":           "fa5s.times-circle",
    # Actions
    "run":             "fa5s.play",
    "stop":            "fa5s.stop",
    "settings":        "fa5s.cog",
    "info":            "fa5s.info-circle",
    "check":           "fa5s.check",
    "warning":         "fa5s.exclamation-triangle",
    "chart":           "fa5s.chart-bar",
    "tag":             "fa5s.tag",
    "search":          "fa5s.search",
    # App keys mirrored from app.py for the sidebar / tiles.
    "mask":            "fa5s.mask",
    "measure":         "fa5s.ruler",
    "annotate":        "fa5s.tag",
    "make_masks":      "fa5s.paint-brush",
    # Distinct from every other module's glyph on purpose -- two identical
    # icons in the sidebar is a worse affordance than a missing one. Checked
    # by tests/test_classify_merged.py.
    "classify_merged": "fa5s.sitemap",
    "classify":        "fa5s.layer-group",
    "umap":            "fa5s.project-diagram",
    "ml_analyze":      "fa5s.chart-line",
    "regression":      "fa5s.wave-square",
    "recruitment":     "fa5s.crosshairs",
    "activation":      "fa5s.bolt",
    "run_history":     "fa5s.history",
    "distributed_jobs": "fa5s.cloud-upload-alt",
    "classifier_evaluation": "fa5s.clipboard-check",
    "analyze_plaques": "fa5s.microscope",
    "train_cellpose":  "fa5s.brain",
    "cellpose_masks":  "fa5s.shapes",
    "cellpose_all":    "fa5s.th",
    # One square divided into four by its own seams: tiles registered into
    # a single canvas. Align & Stitch renders this glyph rather than a
    # bundled PNG (spacr.qt.app._FORCE_GLYPH) because no bundled artwork
    # says "stitched mosaic".
    "align":           "fa5s.border-all",
    "map_barcodes":    "fa5s.barcode",
    "ai_console":      "fa5s.robot",
    # Stacked platters: the app is about what a project weighs on disk and
    # what of it can safely go. Without an entry here a new key falls back
    # to the shared puzzle piece, which is artwork every unfiled app draws
    # — indistinguishable tiles on Home.
    "data_manager":    "fa5s.hdd",
}

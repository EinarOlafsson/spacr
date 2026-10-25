"""Full-window loading cover and startup photomosaic animation.

The application shows this cover while constructing its window. With eager
preloading, :class:`spacr.qt.app._PipelinePreloader` imports pipeline modules
on a worker thread and reports progress through a GUI-thread timer. The
default on-demand policy ends the loading phase without importing those
modules. The host then closes the cover or starts its photomosaic animation,
according to the startup preferences.

Reported progress lights the three phases of :data:`STRAP_LINE` and fills
the thin rule below them. :meth:`LoadingScreen.show_mosaic` replaces the
plain logo with the photomosaic; :meth:`LoadingScreen.blow_away` animates
its removal. :func:`replay_startup_animation` runs the sequence again over
an existing window.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import weakref
from typing import Callable, Optional, Sequence

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRect, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import (QColor, QFont, QImage, QImageReader,
                           QPainter, QPainterPath, QPen, QPixmap, QTransform)
from PySide6.QtWidgets import QWidget

from ..hidpi import logical_size, scaled_for
from ..iconset import RESOURCE_DIR
from ..preferences import scaled_px

#: The one place this sentence lives. The home screen shows it beside the
#: logo too, so a change here changes both -- and it needs a translation row
#: like every other UI string (see tests/qt/test_i18n.py).
STRAP_PHASES = (
    "End-to-end microscopy",
    "single-cell image analysis",
    "genotype-to-phenotype mapping",
)

#: Rendered form, for callers that want the whole sentence.
STRAP_LINE = "  →  ".join(STRAP_PHASES)

def _role(name: str, fallback: str) -> str:
    """A palette role, or ``fallback`` if the theme module cannot be reached.

    The loading screen is the FIRST thing painted, sometimes before the
    theme has been resolved and always before anything else could report a
    problem. A palette lookup that raises here would replace the splash with
    a traceback, so every lookup carries the literal it replaced.
    """
    try:
        from ..theme import active_palette
        value = active_palette().get(name)
        return str(value) if value else fallback
    except Exception:
        return fallback


#: Alias kept for readability at the one call site that names a colour
#: rather than painting with it.
splash_role = _role


def _dark_role(name: str, fallback: str) -> str:
    """A role from the dark palette, or ``fallback`` if it cannot be read.

    For module constants. A constant is evaluated once, at import, so a
    read through :func:`_role` would keep whichever theme was in force at
    that moment; the dark palette is the same answer in every process.
    """
    try:
        from ..theme import palette_for
        value = palette_for("dark").get(name)
        return str(value) if value else fallback
    except Exception:
        return fallback


def _rgba(spec: str, fallback: "QColor") -> "QColor":
    """``rgba(r, g, b, a)`` or ``#rrggbb`` as a :class:`QColor`."""
    text = str(spec).strip()
    if text.lower().startswith("rgba(") and text.endswith(")"):
        parts = [p.strip() for p in text[5:-1].split(",")]
        try:
            values = [int(float(p)) for p in parts]
        except ValueError:
            return fallback
        if len(values) == 4:
            return QColor(*values)
        if len(values) == 3:
            return QColor(*values, 255)
        return fallback
    colour = QColor(text)
    return colour if colour.isValid() else fallback


def _role_color(name: str, fallback: str = "#000000") -> "QColor":
    """Resolve a splash palette role to a colour.

    :param name: the role.
    :param fallback: the colour to use when the role is unknown -- the
        splash is shown before the theme is loaded, so it cannot assume one.
    :returns: the colour.
    """
    return _rgba(_role(name, fallback), QColor(fallback))


def _role_brush(name: str) -> "QColor":
    """Resolve a splash palette role to a colour, defaulting to white.

    :param name: the role.
    :returns: the colour.
    """
    return _role_color(name, "#FFFFFF")


def _ink(alpha: int) -> "QColor":
    """The splash's text colour at ``alpha``.

    Kept for callers that want a weight the palette does not name. The
    paint path no longer uses it: `splash_ink` and `splash_ink_dim` are
    already flattened against the background, so a phase is drawn with an
    opaque colour and no alpha maths.
    """
    colour = QColor(_role("splash_ink", "#FFFFFF"))
    if not colour.isValid():
        colour = QColor(255, 255, 255)
    colour.setAlpha(max(0, min(255, int(alpha))))
    return colour


#: The full-window cover's background, taken from the theme's own window
#: background (`splash_bg`, derived in :func:`spacr.qt.theme.palette_for`).
#:
#: It used to be ``#003737``, sampled from the installer icon, and it read
#: as teal because it IS teal -- a very dark cyan-green at hue 180. That
#: made the first thing the application shows the one full-window surface
#: with a colour cast. It is now the window's own background: black on the
#: dark theme, and identical to the window that replaces it, so the handover
#: has nothing to flash.
#:
#: The name changed with the colour. `INSTALLER_GREEN` described neither.
SPLASH_BACKGROUND = _dark_role("splash_bg", "#000000")

#: Deprecated alias. It was never green after this change and was not
#: accurately named before it; kept only so an existing importer does not
#: break on upgrade.
INSTALLER_GREEN = SPLASH_BACKGROUND

#: The mark, in white, as it appears on the installer icon.
LOGO_FILE = "logo_spacr.png"

#: Where the startup photomosaic lives: the picture and its tile map, both
#: written by ``tools/build_splash_mosaic.py``.
_SPLASH_DIR = os.path.normpath(os.path.join(RESOURCE_DIR, "..", "splash"))

#: The tile map of the photomosaic, next to the picture it describes.
_MOSAIC_MAP_FILE = "photomosaic.json"

#: Set to ``1`` to start without the photomosaic and its animation. Tests,
#: tutorial captures and anything else that must not wait set it.
NO_SPLASH_ENV = "SPACR_NO_SPLASH"

#: The logo holds still this long after it first paints, and longer while
#: the window is still loading (item 695: 0-2 s).
_MOSAIC_MIN_HOLD_S = 2.0

#: The blow-away's length: the last tile has gone this long after it began
#: (item 695: 2-6 s).
_BLOW_AWAY_S = 4.0

#: Seconds into the blow-away the black container starts to fade and is
#: gone: the window shows from 3 s and is fully visible at 5 s.
_CONTAINER_FADE_S = (1.0, 3.0)

#: Seconds the wind front takes to cross the logo, and how much slower than
#: the 694 flight each tile flies, so the motion fills the four seconds.
_SWEEP_S = 2.4
_PACE = 1.8

#: The strap line under the logo fades out over this much of the blow-away.
_STRAP_FADE_S = 0.6

#: Under Reduce motion the Help replay holds the logo, then cross-fades to
#: the window over this long instead of blowing it away.
_PLAIN_FADE_S = 1.0

#: Frame interval of the blow-away, in milliseconds: 60 frames a second.
_MOSAIC_FRAME_MS = 16

#: The seed of the blow-away's randomness, so every start looks the same and
#: a test can reproduce a frame.
_MOSAIC_SEED = 694

#: Share of the screen's height the logo takes (item 695: half).
_MOSAIC_SCREEN_SHARE = 0.5

#: Most of the cover's width the logo may take, on a window narrower than
#: half its screen's height.
_MOSAIC_WIDTH_SHARE = 0.9


def _env_says_no_splash() -> bool:
    """Whether :data:`NO_SPLASH_ENV` asks for a start without the mosaic."""
    value = os.environ.get(NO_SPLASH_ENV, "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def mosaic_splash_wanted() -> bool:
    """Whether the loading cover shows the photomosaic.

    :returns: ``False`` when :data:`NO_SPLASH_ENV` has a nonempty value other than
        ``0``, ``false``, ``no`` or ``off`` (case-insensitive, with
        surrounding whitespace ignored); otherwise ``True``.
    """
    return not _env_says_no_splash()


def startup_animation_allowed() -> bool:
    """Whether the photomosaic may blow away when the window is ready.

    No when :data:`NO_SPLASH_ENV` is set, on the offscreen platform, while
    Preferences > Appearance > Accessibility > Reduce motion is on, or when
    Preferences > Appearance > Animation > Startup animation is off. An
    unreadable preference disables the animation.

    :returns: ``True`` when the blow-away should play.
    """
    if _env_says_no_splash():
        return False
    try:
        from PySide6.QtGui import QGuiApplication

        if QGuiApplication.platformName() == "offscreen":
            return False
        from ..preferences import get_reduce_motion, get_startup_animation

        return bool(get_startup_animation()) and not get_reduce_motion()
    except Exception:
        return False


def will_blow_away(screen) -> bool:
    """Whether ``screen`` will end with the blow-away rather than vanish.

    :param screen: the window's loading cover, or anything standing in for
        it.
    :returns: ``True`` for a :class:`LoadingScreen` showing the mosaic, not
        yet asked to skip, where :func:`startup_animation_allowed` says yes.
    """
    return (isinstance(screen, LoadingScreen) and screen.mosaic_shown()
            and not screen._skip_requested and startup_animation_allowed())


class MosaicMap:
    """The startup photomosaic's picture and which of its tiles hold the logo.

    :param image_path: the picture, ``grid * tile_px`` pixels square.
    :param grid: tiles per side.
    :param tile_px: pixels per tile in the picture.
    :param occupied: ``grid x grid`` booleans, row-major; ``True`` where the
        tile holds the picture and flies, ``False`` for the dim background.
    """

    def __init__(self, image_path: str, grid: int, tile_px: int,
                 occupied: "np.ndarray"):
        """Hold the map; see the class docstring for the parameters."""
        self.image_path = image_path
        self.grid = int(grid)
        self.tile_px = int(tile_px)
        self.occupied = np.asarray(occupied, dtype=bool).reshape(
            self.grid, self.grid)

    @classmethod
    def load(cls, directory: Optional[str] = None) -> Optional["MosaicMap"]:
        """Read the tile map and check its picture is there.

        :param directory: where the map and picture are; the packaged
            ``spacr/resources/splash`` by default.
        :returns: the map, or ``None`` when either file is missing or the map
            cannot be read -- the loading cover then keeps its plain look.
        """
        folder = directory or _SPLASH_DIR
        try:
            with open(os.path.join(folder, _MOSAIC_MAP_FILE),
                      encoding="utf-8") as handle:
                meta = json.load(handle)
            grid = int(meta["grid"])
            bits = np.unpackbits(np.frombuffer(
                bytes.fromhex(meta["occupied"]), dtype=np.uint8))
            occupied = bits[:grid * grid].astype(bool)
            image = os.path.join(folder, meta["image"])
            if occupied.size != grid * grid or not os.path.isfile(image):
                return None
            return cls(image, grid, int(meta["tile_px"]), occupied)
        except Exception:
            return None


def _smoothstep(values):
    """Hermite ease of ``values`` clipped to 0..1; works on arrays."""
    x = np.clip(values, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _container_opacity(t: float, plain: bool = False) -> float:
    """How opaque the black container is ``t`` seconds into the blow-away.

    :param t: seconds since the blow-away (or the plain cross-fade) began.
    :param plain: the Reduce motion cross-fade, which fades over
        :data:`_PLAIN_FADE_S` from its start.
    :returns: 1.0 before :data:`_CONTAINER_FADE_S` starts, falling
        monotonically (smoothstep) to 0.0 at its end.
    """
    if plain:
        return 1.0 - float(_smoothstep(t / _PLAIN_FADE_S))
    start, end = _CONTAINER_FADE_S
    return 1.0 - float(_smoothstep((t - start) / (end - start)))


class BlowAway:
    """The motion of the photomosaic's tiles as a wind front blows them away.

    Pure arithmetic on arrays, with no Qt: every tile's launch time and flight
    are drawn once from a seeded generator, and :meth:`frame` evaluates the
    whole flock at one moment, so the motion depends on elapsed time only and
    never on how many frames were drawn.

    The front sweeps across the mosaic along ``wind``. As it reaches a tile,
    the tile lifts towards the viewer, turns over like a card (one axis scaled
    through the cosine of its flip angle, with a skew and a little
    foreshortening for perspective), spins, and flies downwind with random
    turbulence and slight gravity while it fades. Background tiles do not
    fly; they leave with the front.

    :param occupied: ``grid x grid`` booleans; ``True`` tiles fly.
    :param origin: the mosaic's top-left corner, in widget coordinates.
    :param side: the mosaic's side, in the same units.
    :param seed: seed of the randomness.
    :param wind: direction the wind blows, ``(x, y)``; down-screen is ``+y``.
    :param start: seconds before the front reaches the first tile.
    :param sweep: seconds the front takes to cross the mosaic.
    :param pace: how many times slower than the 694 flight each tile flies
        (lift, flip, spin, travel and fade are stretched alike).
    :param end: when given, the last tile has gone by then: lives are cut
        to end there and :attr:`duration` is ``end``.
    """

    LIFT_S = 0.14

    def __init__(self, occupied, origin, side: float, *,
                 seed: int = _MOSAIC_SEED, wind=(1.0, 0.3),
                 start: float = 0.05, sweep: float = 1.1, pace: float = 1.0,
                 end: Optional[float] = None):
        """Draw every tile's launch and flight; see the class docstring."""
        cells = np.asarray(occupied, dtype=bool)
        grid = cells.shape[0]
        rng = np.random.default_rng(seed)
        self.grid = grid
        self.side = float(side)
        self.pitch = self.side / grid
        self.origin = (float(origin[0]), float(origin[1]))
        direction = np.asarray(wind, dtype=float)
        direction = direction / max(float(np.hypot(*direction)), 1e-9)
        self.wind = (float(direction[0]), float(direction[1]))
        self.start = float(start)
        self.sweep = float(sweep)
        self.pace = float(pace)
        rows, cols = np.mgrid[0:grid, 0:grid]
        rows = rows.ravel().astype(float)
        cols = cols.ravel().astype(float)
        span = abs(direction[0]) + abs(direction[1])
        along = ((cols + 0.5) / grid * direction[0]
                 + (rows + 0.5) / grid * direction[1])
        along = (along - min(0.0, direction[0]) - min(0.0, direction[1]))
        along = along / span
        ragged = 0.04 * np.sin(rows * 0.9 + rng.uniform(0.0, 2.0 * np.pi))
        jitter = rng.uniform(-0.06, 0.06, along.size) + ragged
        self.launch = self.start + self.sweep * np.clip(along + jitter,
                                                         0.0, 1.0)
        self.flyers = np.flatnonzero(cells.ravel())
        count = self.flyers.size
        self.centre_x = self.origin[0] + (cols[self.flyers] + 0.5) * self.pitch
        self.centre_y = self.origin[1] + (rows[self.flyers] + 0.5) * self.pitch
        self.fly_launch = self.launch[self.flyers]
        self.life = rng.uniform(0.5, 0.85, count) * self.pace
        self.speed = rng.uniform(0.25, 0.6, count) * self.side
        self.gust = rng.uniform(1.2, 2.2, count) * self.side
        self.lift = rng.uniform(0.05, 0.3, count) * self.side
        self.gravity = 0.6 * self.side
        self.turb_amp = rng.uniform(0.01, 0.035, count) * self.side
        self.turb_freq = rng.uniform(6.0, 12.0, count)
        self.turb_phase = rng.uniform(0.0, 2.0 * np.pi, count)
        self.flip_rate = (rng.uniform(1.2, 2.8, count) * 2.0 * np.pi
                          * rng.choice((-1.0, 1.0), count))
        self.spin_rate = np.radians(rng.uniform(-360.0, 360.0, count))
        if end is not None:
            self.life = np.minimum(self.life,
                                   float(end) - 0.02 - self.fly_launch)
            self.duration = float(end)
            return
        last = float(np.max(self.fly_launch + self.life)) if count else 0.0
        self.duration = max(last, self.start + self.sweep + 0.5) + 0.02

    def backdrop(self, t: float):
        """How opaque the cover's ground is behind and ahead of the front.

        :param t: seconds since the blow-away began.
        :returns: ``(front, behind, ahead)``: the front's position along the
            wind as a share of the mosaic (0 at its upwind edge, 1 at the
            downwind one), and the ground's opacity, 0..1, on either side.
        """
        front = (t - self.start) / self.sweep
        ahead = 1.0 - float(_smoothstep((t - self.start)
                                        / (self.sweep + 0.5)))
        behind = ahead * (1.0 - float(_smoothstep((t - self.start) / 0.45)))
        return front, behind, ahead

    def launched(self, t: float) -> "np.ndarray":
        """Indices of every tile, flying or not, the front has reached.

        :param t: seconds since the blow-away began.
        :returns: row-major tile indices.
        """
        return np.flatnonzero(self.launch <= t)

    def frame(self, t: float, bounds=None, floor: float = 0.02):
        """Where every flying tile is at ``t``.

        :param t: seconds since the blow-away began.
        :param bounds: ``(x0, y0, x1, y1)``; tiles whose centre has left it by
            more than two pitches are dropped. ``None`` keeps them all.
        :param floor: tiles fainter than this are dropped.
        :returns: ``(index, m11, m12, m21, m22, dx, dy, opacity)``: indices
            into the flying tiles and, for each, the affine matrix that places
            a tile drawn centred on the origin, and its opacity.
        """
        age = t - self.fly_launch
        live = (age >= 0.0) & (age < self.life)
        index = np.flatnonzero(live)
        lived = age[index]
        a = lived / self.pace
        wind_x, wind_y = self.wind
        travel = self.speed[index] * a + 0.5 * self.gust[index] * a * a
        sway = self.turb_amp[index] * (
            np.sin(self.turb_freq[index] * a + self.turb_phase[index])
            - np.sin(self.turb_phase[index]))
        dx = self.centre_x[index] + wind_x * travel - wind_y * sway
        dy = (self.centre_y[index] + wind_y * travel + wind_x * sway
              - self.lift[index] * a + 0.5 * self.gravity * a * a)
        flip = self.flip_rate[index] * np.maximum(a - 0.03, 0.0)
        spin = self.spin_rate[index] * a
        lift = 1.0 + 0.25 * _smoothstep(a / self.LIFT_S)
        cos_f, sin_f = np.cos(flip), np.sin(flip)
        sx = lift * cos_f
        sy = lift * (1.0 - 0.14 * np.abs(sin_f))
        skew = 0.22 * sin_f
        cos_s, sin_s = np.cos(spin), np.sin(spin)
        m11 = cos_s * sx
        m21 = (cos_s * skew - sin_s) * sy
        m12 = sin_s * sx
        m22 = (sin_s * skew + cos_s) * sy
        fade = 1.0 - _smoothstep((lived / self.life[index] - 0.35) / 0.65)
        opacity = fade * (0.72 + 0.28 * cos_f)
        keep = opacity > floor
        if bounds is not None:
            margin = 2.0 * self.pitch
            keep &= ((dx > bounds[0] - margin) & (dx < bounds[2] + margin)
                     & (dy > bounds[1] - margin) & (dy < bounds[3] + margin))
        return (index[keep], m11[keep], m12[keep], m21[keep], m22[keep],
                dx[keep], dy[keep], opacity[keep])



class LoadingScreen(QWidget):
    """Full-window loading cover with optional photomosaic animation.

    By default it shows the white mark beside the strap line. Once
    :meth:`show_mosaic` is called it shows the spaCR photomosaic above the
    strap line instead, and :meth:`blow_away` ends it by blowing the mosaic's
    tiles off the window to reveal the window underneath.

    :param total: how many steps will be reported. Zero or negative means
        "unknown", and the phases stay dim rather than dividing by it.
    :param parent: the window this fills. It is sized to the parent and
        resizes with it, rather than being a separate top-level -- a second
        window would earn its own taskbar entry and could be dragged off the
        thing it is supposed to be covering.
    """

    def __init__(self, total: int = 0, parent: Optional[QWidget] = None):
        """Build the splash that covers the window while it is being assembled.

        Opaque by construction: it covers a partly-built window, and any
        transparency would show the thing it exists to hide.

        :param total: how many steps the progress bar counts to; ``0`` shows no
            proportion.
        :param parent: the window to cover; its geometry is adopted.
        """
        super().__init__(parent)
        font = QFont(self.font())
        font.setWeight(QFont.Weight.Bold)
        self.setFont(font)
        self.setObjectName("LoadingScreen")
        self._total = max(0, int(total))
        self._done = 0
        self._logo: Optional[QPixmap] = None
        self._mosaic: Optional[MosaicMap] = None
        self._mosaic_pixmap: Optional[QPixmap] = None
        self._mosaic_key = None
        self._clock: Callable[[], float] = time.monotonic
        self._first_paint_at: Optional[float] = None
        self._state = "idle"
        self._skip_requested = False
        self._plain = False
        self._duration = 0.0
        self._on_finished: Optional[Callable[[], None]] = None
        self._model: Optional[BlowAway] = None
        self._began_at = 0.0
        self._tiles: list = []
        self._rest: Optional[QImage] = None
        self._rest_order = None
        self._rest_launch = None
        self._rest_cleared = 0
        self._rest_tile = 1
        self._rest_target = QRectF()
        self._anchor = QPointF()
        self._tick_timer: Optional[QTimer] = None
        self._underlay: Optional[QPixmap] = None
        self._focus_before: Optional[QWidget] = None
        self._floor = 0.02
        self._frame_ms: list = []
        self._drawn: list = []
        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        if parent is not None:
            self.setGeometry(parent.rect())
            parent.installEventFilter(self)
        self._load_logo()

    def set_total(self, total: int) -> None:
        """Set the denominator; repaints if it changed.

        :param total: number of steps the progress counts towards; converted
            to ``int`` and floored at 0, which means unknown and makes
            :meth:`fraction` report 0.0.
        """
        total = max(0, int(total))
        if total != self._total:
            self._total = total
            self.update()

    def advance(self, done: Optional[int] = None) -> None:
        """Report progress. Without an argument, counts one more step."""
        self._done = self._done + 1 if done is None else max(0, int(done))
        self.update()

    def fraction(self) -> float:
        """Completed share, 0.0 to 1.0. Zero when the total is unknown."""
        if self._total <= 0:
            return 0.0
        return min(1.0, self._done / float(self._total))

    def lit_phases(self) -> int:
        """How many of the three phases are lit at the current fraction.

        The last phase lights only at completion, so a user never sees the
        sentence finished while work is still running.
        """
        f = self.fraction()
        if f >= 1.0:
            return len(STRAP_PHASES)
        return min(len(STRAP_PHASES) - 1, int(f * len(STRAP_PHASES)))

    def _load_logo(self) -> None:
        """Load the splash logo, tolerating its absence.

        A missing logo must not stop the application starting: the screen still
        covers the window and still reports progress.
        """
        try:
            path = os.path.join(RESOURCE_DIR, LOGO_FILE)
            pix = QPixmap(path)
            if not pix.isNull():
                self._logo = pix
        except Exception:
            self._logo = None

    def show_mosaic(self, mosaic: Optional[MosaicMap] = None) -> bool:
        """Show the photomosaic instead of the mark.

        :param mosaic: the map to show; the packaged one by default.
        :returns: whether the mosaic is now shown. ``False`` when its files
            are missing, and the cover keeps the mark.
        """
        mosaic = mosaic if mosaic is not None else MosaicMap.load()
        if mosaic is None:
            return False
        self._mosaic = mosaic
        self._mosaic_pixmap = None
        self._mosaic_key = None
        self.update()
        return True

    def mosaic_shown(self) -> bool:
        """Whether the cover shows the photomosaic."""
        return self._mosaic is not None

    def playing(self) -> bool:
        """Whether the blow-away is waiting to start or under way."""
        return self._state in ("waiting", "holding", "playing")

    def _ratio(self) -> float:
        """Device pixels per logical pixel on this widget's screen."""
        from ..hidpi import device_ratio

        return max(1.0, float(device_ratio(self)))

    def _mosaic_layout(self):
        """Where the mosaic and the strap line go at the current size.

        The logo is half its screen's height, centred on the cover, with the
        strap line under it; a cover too small for that shrinks it to fit.
        It is decoded at the whole number of device pixels per tile at or
        just above that size, so its tiles can be cut out again exactly, and
        drawn scaled down to the size, which keeps it sharp.

        :returns: ``(x, y, side, tile_device_px, ratio)``: the mosaic's
            top-left corner and side in logical pixels, its tile pitch in
            device pixels, and the device pixel ratio.
        """
        mosaic = self._mosaic
        ratio = self._ratio()
        strap_room = scaled_px(15) + scaled_px(60)
        screen = self.screen()
        tall = screen.geometry().height() if screen is not None \
            else self.height()
        side = max(1.0, min(tall * _MOSAIC_SCREEN_SHARE,
                            self.width() * _MOSAIC_WIDTH_SHARE,
                            self.height() - 2.0 * strap_room))
        tile = math.ceil(side * ratio / mosaic.grid - 1e-9)
        tile = max(1, min(mosaic.tile_px, tile))
        return ((self.width() - side) / 2.0, (self.height() - side) / 2.0,
                side, tile, ratio)

    def _mosaic_picture(self, tile: int, ratio: float) -> Optional[QPixmap]:
        """The mosaic decoded at ``tile`` device pixels per tile, cached.

        Decoded straight to that size by the JPEG reader, which is faster than
        decoding the whole picture and scaling it, and lands every tile edge
        on a pixel. Its black is made transparent (:func:`_black_to_alpha`),
        so the still mosaic and the flying tiles are one picture.

        :param tile: device pixels per tile.
        :param ratio: device pixel ratio the picture is drawn at.
        :returns: the picture, carrying ``ratio``, or ``None`` if it cannot
            be read.
        """
        key = (tile, ratio)
        if self._mosaic_key == key:
            return self._mosaic_pixmap
        side = tile * self._mosaic.grid
        reader = QImageReader(self._mosaic.image_path)
        reader.setScaledSize(QSize(side, side))
        image = reader.read()
        if image.isNull():
            self._mosaic = None
            return None
        pixmap = QPixmap.fromImage(_black_to_alpha(image))
        pixmap.setDevicePixelRatio(ratio)
        self._mosaic_pixmap = pixmap
        self._mosaic_key = key
        return pixmap

    def eventFilter(self, watched, event) -> bool:
        """Keep covering the window as it resizes.

        A window that opens maximised is resized after the cover is put
        over it, and by then the window may have handed the cover to the
        blow-away and stopped sizing it.

        :param watched: the window this covers.
        :param event: the event; only a resize is acted on.
        :returns: ``False``, so the window still sees every event.
        """
        if event.type() == QEvent.Type.Resize and watched is self.parent():
            self.setGeometry(watched.rect())
        return False

    def resizeEvent(self, event) -> None:
        """Re-centre the message for the new size.

        :param event: the Qt resize event.
        """
        super().resizeEvent(event)
        self.update()

    def mousePressEvent(self, event) -> None:
        """A click on the cover asks to skip the blow-away.

        :param event: the Qt mouse event; accepted, so the click goes no
            further than the cover.
        """
        self.skip()
        event.accept()

    def _strap_metrics(self, painter: QPainter):
        """The strap line's words, their widths and the arrow's width.

        :param painter: the painter whose font measures them.
        :returns: ``(phases, widths, arrow_width, text_width)``.
        """
        font = QFont(self.font())
        font.setPixelSize(scaled_px(15))
        painter.setFont(font)
        metrics = painter.fontMetrics()
        phases = [_translate(p) for p in STRAP_PHASES]
        widths = [metrics.horizontalAdvance(p) for p in phases]
        arrow_w = metrics.horizontalAdvance("  →  ")
        text_w = sum(widths) + arrow_w * (len(phases) - 1)
        return phases, widths, arrow_w, text_w

    def _paint_strap_arrow(self, painter: QPainter, tx: float,
                           baseline: float, advance: float,
                           colour: QColor) -> None:
        """Draw the arrow between two strap phases as strokes, not a glyph.

        The strap font rarely carries U+2192 in bold. On Linux Qt fell back
        to Noto Sans Math Regular, a hairline arrow, and once the default
        scale became 90 % (7884c4a1f) the unlit arrow kept 4 legible pixels
        on the light splash where the phrases kept hundreds. Stroked at the
        bold face's stem width, the arrow reads the same in every font and
        on every platform, inside the advance ``"  →  "`` still reserves.
        The stem is a whole number of pixels and the shaft sits on the pixel
        grid, so it is drawn in full ink rather than spread over two
        half-tone rows at a small font scale.

        :param painter: the painter, with the strap font set.
        :param tx: left edge of the arrow's advance.
        :param baseline: the text baseline.
        :param advance: the width ``"  →  "`` takes in the strap font.
        :param colour: the arrow's ink.
        """
        metrics = painter.fontMetrics()
        size = float(painter.font().pixelSize() or metrics.height())
        length = float(max(5, round(metrics.horizontalAdvance("→") * 0.8)))
        stem = float(max(2, round(size * 0.13)))
        head = length * 0.38
        centre_x = tx + advance / 2.0
        centre_y = round(baseline - metrics.xHeight() / 2.0)
        if int(stem) % 2:
            centre_y += 0.5
        left = round(centre_x - length / 2.0)
        right = left + length
        path = QPainterPath(QPointF(left, centre_y))
        path.lineTo(QPointF(right, centre_y))
        path.moveTo(QPointF(right - head, centre_y - head * 0.8))
        path.lineTo(QPointF(right, centre_y))
        path.lineTo(QPointF(right - head, centre_y + head * 0.8))
        pen = QPen(colour, stem)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.save()
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)
        finally:
            painter.restore()

    def _paint_strap(self, painter: QPainter, tx: float, baseline: float,
                     strap) -> None:
        """Draw the strap line from ``tx`` on ``baseline``, and its rule.

        :param painter: the painter, with the strap font already set.
        :param tx: left edge of the line.
        :param baseline: the text baseline.
        :param strap: what :meth:`_strap_metrics` returned.
        """
        phases, widths, arrow_w, text_w = strap
        metrics = painter.fontMetrics()
        lit = self.lit_phases()
        rule_x = tx
        for index, phase in enumerate(phases):
            painter.setPen(_role_color(
                "splash_ink" if index < lit else "splash_ink_dim"))
            painter.drawText(int(tx), int(baseline), phase)
            tx += widths[index]
            if index < len(phases) - 1:
                self._paint_strap_arrow(
                    painter, tx, baseline, arrow_w, _role_color(
                        "splash_ink" if index + 1 < lit
                        else "splash_ink_dim"))
                tx += arrow_w

        rule_y = baseline + metrics.descent() + scaled_px(12)
        painter.setPen(Qt.NoPen)
        painter.setBrush(_role_brush("splash_track"))
        painter.drawRect(QRectF(rule_x, rule_y, text_w, 2.0))
        painter.setBrush(_role_brush("splash_fill"))
        painter.drawRect(QRectF(rule_x, rule_y, text_w * self.fraction(), 2.0))

    def paintEvent(self, event) -> None:
        """Draw the loading state, or one frame of the blow-away.

        :param event: the Qt paint event.
        """
        if self._first_paint_at is None:
            self._first_paint_at = self._clock()
            if self._state == "waiting":
                self._hold_then_begin()
        if self._state == "playing":
            self._paint_blow_away()
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.fillRect(self.rect(), QColor(_role_color("splash_bg")))
            if self._mosaic is not None and self._paint_mosaic(painter):
                return

            side = scaled_px(140)
            gap = scaled_px(28)
            strap = self._strap_metrics(painter)
            metrics = painter.fontMetrics()

            block_w = side + gap + strap[3]
            x = (self.width() - block_w) / 2.0
            y = self.height() / 2.0

            if self._logo is not None:
                scaled = scaled_for(self._logo, self, side)
                drawn = logical_size(scaled)
                painter.drawPixmap(
                    int(x), int(y - drawn.height() / 2.0), scaled)

            baseline = y + metrics.ascent() / 2.0 - metrics.descent() / 2.0
            self._paint_strap(painter, x + side + gap, baseline, strap)
        finally:
            painter.end()

    def _paint_mosaic(self, painter: QPainter) -> bool:
        """Draw the still mosaic with the strap line under it.

        :param painter: an active painter on this widget.
        :returns: ``False`` when the picture cannot be read, so the caller
            draws the mark instead.
        """
        x, y, side, tile, ratio = self._mosaic_layout()
        picture = self._mosaic_picture(tile, ratio)
        if picture is None:
            return False
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        target = QRectF(x, y, side, side)
        painter.drawPixmap(target, picture, QRectF(picture.rect()))
        self._paint_strap_under(painter, target)
        return True

    def _paint_strap_under(self, painter: QPainter, target: QRectF) -> None:
        """Draw the strap line centred under the logo at ``target``.

        :param painter: an active painter on this widget; its opacity is
            the strap line's.
        :param target: where the logo is drawn, in logical pixels.
        """
        strap = self._strap_metrics(painter)
        metrics = painter.fontMetrics()
        baseline = target.bottom() + scaled_px(28) + metrics.ascent()
        self._paint_strap(painter, (self.width() - strap[3]) / 2.0,
                          baseline, strap)

    def blow_away(self, on_finished: Optional[Callable[[], None]] = None,
                  *, force: bool = False, plain: bool = False) -> bool:
        """End the cover by blowing the mosaic's tiles off the window.

        The logo holds until two seconds after the cover first painted (at
        once when called later than that), then blows away over four
        seconds: the black container fades from one to three seconds in and
        the last tile has gone at four. The cover deletes itself then.

        :param on_finished: called once, when the cover has gone.
        :param force: play even where :func:`startup_animation_allowed` says
            no; for the Help replay, tests and previews.
        :param plain: hold the logo, then cross-fade it and the container
            away in one second instead of blowing it away (Reduce motion).
        :returns: ``True`` when the blow-away will play and the cover will
            take itself down; ``False`` when it will not -- no mosaic, the
            animation is off, or a click already asked to skip -- and the
            caller takes the cover down as before.
        """
        if self._mosaic is None or self._state != "idle":
            return False
        if self._skip_requested:
            return False
        if not (force or startup_animation_allowed()):
            return False
        self._on_finished = on_finished
        self._plain = bool(plain)
        self._take_the_keys()
        _PLAYING.add(self)
        self._state = "waiting"
        if self._first_paint_at is not None:
            self._hold_then_begin()
        else:
            self.update()
        return True

    def _hold_then_begin(self) -> None:
        """Begin once the mosaic has been up for the hold."""
        from .. import _after

        self._state = "holding"
        held = self._clock() - self._first_paint_at
        wait_ms = max(0, int(math.ceil((_MOSAIC_MIN_HOLD_S - held) * 1000.0)))
        _after(wait_ms, self, self._begin)

    def _begin(self) -> None:
        """Cut the mosaic into its tiles and start the frames.

        Called before the hold is over -- a timer that fired a moment early
        -- it waits out the rest of the hold instead.
        """
        if self._state != "holding":
            return
        if self._clock() - self._first_paint_at < _MOSAIC_MIN_HOLD_S:
            self._hold_then_begin()
            return
        x, y, side, tile, ratio = self._mosaic_layout()
        picture = self._mosaic_picture(tile, ratio)
        if picture is None:
            self._finish()
            return
        self._rest_target = QRectF(x, y, side, side)
        if self._plain:
            self._duration = _PLAIN_FADE_S
        else:
            self._cut_tiles(picture, x, y, side, tile)
        self.setAutoFillBackground(False)
        self._capture_underlay()
        self.setAttribute(Qt.WA_OpaquePaintEvent, self._underlay is not None)
        self.setAttribute(Qt.WA_StyledBackground, False)
        self._state = "playing"
        self._began_at = self._clock()
        timer = QTimer(self)
        timer.setTimerType(Qt.TimerType.PreciseTimer)
        timer.setInterval(_MOSAIC_FRAME_MS)
        timer.timeout.connect(self._tick)
        timer.start()
        self._tick_timer = timer
        self.update()

    def _capture_underlay(self) -> None:
        """Composite over a single native Mac frame during the short dissolve.

        A translucent child otherwise repaints the entire Retina window below
        it on every animation tick. Keep the cover opaque to Qt and perform
        the same blend locally; normal live painting resumes when it closes.
        """
        from PySide6.QtGui import QGuiApplication

        parent = self.parentWidget()
        if QGuiApplication.platformName() != "cocoa" or parent is None:
            return
        self.hide()
        try:
            picture = parent.grab()
            if not picture.isNull():
                self._underlay = picture
        finally:
            self.show()
            self.raise_()

    def _cut_tiles(self, picture: QPixmap, x: float, y: float, side: float,
                   tile: int) -> None:
        """Lay out the flight and cut the logo into the tiles that fly.

        :param picture: the logo, decoded at ``tile`` device pixels per tile.
        :param x: the logo's left edge, in logical pixels.
        :param y: its top edge.
        :param side: its side.
        :param tile: device pixels per tile in ``picture``.
        """
        grid = self._mosaic.grid
        occupied = self._mosaic.occupied
        if sys.platform == 'darwin' and grid >= 64 and grid % 2 == 0:
            occupied = np.asarray(occupied).reshape(grid // 2, 2,
                                                   grid // 2, 2).any(axis=(1, 3))
            grid //= 2
            tile *= 2
        pitch = side / grid
        model = BlowAway(occupied, (x, y), side, sweep=_SWEEP_S,
                         pace=_PACE, end=_BLOW_AWAY_S)
        rest = picture.toImage().convertToFormat(
            QImage.Format.Format_ARGB32_Premultiplied)
        rest.setDevicePixelRatio(1.0)
        tiles = []
        for flat in model.flyers.tolist():
            row, col = divmod(flat, grid)
            piece = QPixmap.fromImage(
                rest.copy(QRect(col * tile, row * tile, tile, tile)))
            piece.setDevicePixelRatio(tile / pitch)
            tiles.append(piece)
        self._model = model
        self._duration = model.duration
        self._tiles = tiles
        self._rest = rest
        self._rest_tile = tile
        self._rest_order = np.argsort(model.launch, kind="stable")
        self._rest_launch = model.launch[self._rest_order]
        self._rest_cleared = 0
        self._anchor = QPointF(-pitch / 2.0, -pitch / 2.0)

    def elapsed(self) -> float:
        """Seconds since the blow-away began; ``0.0`` before it has."""
        if self._state != "playing":
            return 0.0
        return self._clock() - self._began_at

    def container_opacity(self) -> float:
        """How opaque the black container behind the logo is now.

        :returns: 1.0 until a second into the blow-away, then falling
            monotonically to 0.0 three seconds in (over the one second of the
            plain cross-fade under Reduce motion); 0.0 once the cover is done.
        """
        if self._state == "done":
            return 0.0
        return _container_opacity(self.elapsed(), self._plain)

    def _tick(self) -> None:
        """Ask for the next frame, or finish when the last tile has gone."""
        if self.elapsed() >= self._duration:
            self._finish()
            return
        self.update()

    def _clear_launched(self, t: float) -> None:
        """Take every tile the front has reached out of the still picture.

        :param t: seconds since the blow-away began.
        """
        order = self._rest_order
        reached = int(np.searchsorted(self._rest_launch, t, side="right"))
        if reached <= self._rest_cleared:
            return
        grid = self._model.grid
        tile = self._rest_tile
        painter = QPainter(self._rest)
        try:
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            for flat in order[self._rest_cleared:reached].tolist():
                row, col = divmod(flat, grid)
                painter.fillRect(QRect(col * tile, row * tile, tile, tile),
                                 Qt.transparent)
        finally:
            painter.end()
        self._rest_cleared = reached

    def _paint_blow_away(self) -> None:
        """Draw one frame of the blow-away, or of the plain cross-fade."""
        began = time.perf_counter()
        t = self.elapsed()
        drawn = 0
        painter = QPainter(self)
        try:
            if self._underlay is not None:
                painter.drawPixmap(self.rect(), self._underlay)
            ground = _role_color("splash_bg")
            ground.setAlphaF(self.container_opacity())
            painter.fillRect(self.rect(), ground)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            fade = _PLAIN_FADE_S if self._plain else _STRAP_FADE_S
            strap = 1.0 - float(_smoothstep(t / fade))
            if strap > 0.0:
                painter.setOpacity(strap)
                self._paint_strap_under(painter, self._rest_target)
            if self._plain:
                picture = self._mosaic_pixmap
                painter.setOpacity(self.container_opacity())
                painter.drawPixmap(self._rest_target, picture,
                                   QRectF(picture.rect()))
            else:
                drawn = self._paint_tiles(painter, t)
        finally:
            painter.end()
        spent = (time.perf_counter() - began) * 1000.0
        self._frame_ms.append(spent)
        self._drawn.append(drawn)
        self._adapt(spent)

    def _paint_tiles(self, painter: QPainter, t: float) -> int:
        """Draw the tiles still at rest and every tile in flight at ``t``.

        :param painter: an active painter on this widget.
        :param t: seconds since the blow-away began.
        :returns: how many flying tiles were drawn.
        """
        model = self._model
        self._clear_launched(t)
        painter.setOpacity(1.0)
        painter.drawImage(self._rest_target, self._rest)
        index, m11, m12, m21, m22, dx, dy, alpha = model.frame(
            t, (0.0, 0.0, float(self.width()), float(self.height())),
            self._floor)
        tiles = self._tiles
        anchor = self._anchor
        matrix = QTransform()
        set_matrix = matrix.setMatrix
        set_transform = painter.setTransform
        set_opacity = painter.setOpacity
        draw = painter.drawPixmap
        for i, a, b, c, d, e, f, o in zip(
                index.tolist(), m11.tolist(), m12.tolist(), m21.tolist(),
                m22.tolist(), dx.tolist(), dy.tolist(), alpha.tolist()):
            set_matrix(a, b, 0.0, c, d, 0.0, e, f, 1.0)
            set_transform(matrix)
            set_opacity(o)
            draw(anchor, tiles[i])
        return int(index.size)

    def _adapt(self, spent_ms: float) -> None:
        """Drop the faintest tiles while frames are slow, and take them back.

        :param spent_ms: how long the last frame took to draw.
        """
        if spent_ms > 14.0:
            self._floor = min(0.5, self._floor + 0.08)
        elif spent_ms < 9.0:
            self._floor = max(0.02, self._floor - 0.04)

    def frame_stats(self) -> dict:
        """What drawing the blow-away cost, for measuring it.

        :returns: ``frames``, the mean and the slowest frame's paint time in
            milliseconds, and the most tiles drawn in one frame.
        """
        spent = self._frame_ms
        return {
            "frames": len(spent),
            "mean_ms": float(np.mean(spent)) if spent else 0.0,
            "max_ms": float(np.max(spent)) if spent else 0.0,
            "most_tiles": max(self._drawn) if self._drawn else 0,
        }

    def skip(self) -> None:
        """Skip to the end: take the cover down now, or as soon as allowed.

        Before :meth:`blow_away` it is remembered, and the cover then comes
        down without the animation when loading ends.
        """
        if self.playing():
            self._finish()
        elif self._state == "idle":
            self._skip_requested = True

    def keyPressEvent(self, event) -> None:
        """A key while the blow-away runs skips it.

        :param event: the Qt key event; accepted, so the key goes no further.
        """
        self.skip()
        event.accept()

    def _take_the_keys(self) -> None:
        """Hold the keyboard focus while the blow-away runs.

        Keys then reach :meth:`keyPressEvent` without a filter on the whole
        application, which would see every event the starting window makes.
        The widget that had the focus gets it back in :meth:`_finish`.
        """
        from PySide6.QtWidgets import QApplication

        self._focus_before = QApplication.focusWidget()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def _give_the_keys_back(self) -> None:
        """Return the keyboard focus to the widget that had it."""
        before, self._focus_before = self._focus_before, None
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.clearFocus()
        try:
            if before is not None and before.isVisible():
                before.setFocus(Qt.FocusReason.OtherFocusReason)
        except RuntimeError:
            pass

    def _finish(self) -> None:
        """Stop the frames, take the cover down and say so, once."""
        if self._state == "done":
            return
        self._state = "done"
        _PLAYING.discard(self)
        self._give_the_keys_back()
        if self._tick_timer is not None:
            self._tick_timer.stop()
            self._tick_timer.deleteLater()
            self._tick_timer = None
        self._tiles = []
        self._rest = None
        self._underlay = None
        self.hide()
        self.deleteLater()
        callback, self._on_finished = self._on_finished, None
        if callback is not None:
            callback()


def _black_to_alpha(image: QImage) -> QImage:
    """``image`` with its black made transparent, for drawing over anything.

    Each pixel's opacity becomes its brightest channel and its colour is kept
    as the premultiplied colour, so over black it composites to exactly the
    picture it was, and over the window the tiles keep their glow while their
    black ground lets the window through.

    :param image: the mosaic, any format.
    :returns: a premultiplied ARGB copy at a device pixel ratio of 1.
    """
    out = image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    out.setDevicePixelRatio(1.0)
    width, height = out.width(), out.height()
    pixels = np.frombuffer(out.bits(), dtype=np.uint8).reshape(
        height, out.bytesPerLine() // 4, 4)[:, :width]
    alpha = np.maximum(pixels[..., 0], pixels[..., 1])
    np.maximum(alpha, pixels[..., 2], out=alpha)
    pixels[..., 3] = alpha
    return out


#: The covers whose blow-away is waiting or under way.
_PLAYING: "weakref.WeakSet" = weakref.WeakSet()


def blow_away_running() -> bool:
    """Whether a blow-away is drawing its frames.

    Idle-time warming asks this and waits, so its slices do not take frames
    from the animation in the four seconds it runs. While the logo only
    holds, the window keeps loading underneath it.

    :returns: ``True`` while any cover that still exists is past its hold
        and drawing; a cover deleted mid-way, with its window, no longer
        counts.
    """
    from shiboken6 import isValid

    for screen in list(_PLAYING):
        if not isValid(screen):
            _PLAYING.discard(screen)
        elif screen._state == "playing":
            return True
    return False


def _translate(text: str) -> str:
    """Translate one phase, falling back to English.

    Imported lazily and defensively: this widget is on the launch path, and a
    loading screen that cannot render because the catalog failed to import
    would take the whole application with it.
    """
    try:
        from ..i18n import tr
        return tr(text)
    except Exception:
        return text


def strap_phrases() -> Sequence[str]:
    """The strap line's phases, translated, for the home screen to reuse.

    The home screen shows the same sentence beside the same logo, so it takes
    the words from here rather than repeating them -- one string, one place,
    one set of translation rows.
    """
    return tuple(_translate(p) for p in STRAP_PHASES)


def strap_line() -> str:
    """The whole strap line as one translated string."""
    return "  →  ".join(strap_phrases())


def replay_startup_animation(window: QWidget) -> Optional[LoadingScreen]:
    """Play the startup sequence again inside ``window``.

    Help > Show startup animation. The black container with the logo comes
    back over the window, holds for two seconds and blows away as it does
    at startup. It plays even when Preferences > Appearance > Animation >
    Startup animation is off; under Preferences > Appearance > Accessibility
    > Reduce motion the logo holds and then cross-fades away in one second.
    A click or a key skips it. The window's size, state and keyboard focus
    are left as they were.

    :param window: the window to play it in.
    :returns: the cover now playing, or ``None`` when the logo cannot be
        read or a cover is already over the window.
    """
    if any(not cover.isHidden() and cover._state != "done"
           for cover in window.findChildren(
               LoadingScreen, options=Qt.FindChildOption.FindDirectChildrenOnly)):
        return None
    cover = LoadingScreen(total=1, parent=window)
    if not cover.show_mosaic():
        cover.deleteLater()
        return None
    cover.advance(1)
    cover.setGeometry(window.rect())
    cover.raise_()
    cover.show()
    try:
        from ..preferences import get_reduce_motion

        plain = bool(get_reduce_motion())
    except Exception:
        plain = True
    cover.blow_away(force=True, plain=plain)
    return cover

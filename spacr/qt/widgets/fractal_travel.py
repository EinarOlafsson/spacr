"""The spaceout fractal: a GPU shader when there is one, Numba otherwise.

Ported from `fractal_travel.py` v2.1.0. Two renderers, not one engine with a
switch:

* **GPU** -- VisPy/gloo with a GLSL fragment shader, four spatial samples per
  physical pixel, and a detail loop that adapts from sampled GPU time.
* **CPU** -- a cheaper orbit-fold fractal in Numba, evaluated at animation
  rate with a four-position temporal 2x2 jitter and a rolling four-frame
  window. No keyframes and no crossfades: every displayed frame is new.

THREE THINGS THIS FILE DOES THAT THE SCRIPT DID NOT.

`vispy` is not installed in the shipped environment, so `backend='auto'`
resolves to CPU today and to GPU the day it is. Nothing else changes.

It is PySide6. The script was PyQt6, and importing that binding inside this
application would put two Qt bindings in one process, which does not raise --
it segfaults.

And it WINDS DOWN. A backdrop eating cores while a segmentation runs is the
opposite of what it is for, so `pause()` stops the render loop and leaves the
last frame on screen. See :meth:`CpuFractalWidget.pause`.
"""
from __future__ import annotations

import math
import re
import os
import sys
import time
import logging
from dataclasses import dataclass, replace
from typing import Final, Literal, Optional

import numpy as np

try:
    from numba import config as numba_config, njit, prange, set_num_threads
except Exception:                                            # pragma: no cover
    njit = None
    prange = range
    numba_config = None

    def set_num_threads(_n: int) -> None:
        return None


VERSION: Final[str] = "2.1.0"
Backend = Literal["auto", "gpu", "cpu"]
Quality = Literal["auto", "balanced", "high"]
BACKENDS: Final[tuple[str, ...]] = ("auto", "gpu", "cpu")
QUALITIES: Final[tuple[str, ...]] = ("auto", "balanced", "high")

#: Reference defaults shared by both renderers. `auto` picks the GPU when
#: vispy is importable and the CPU otherwise, which makes one set of numbers
#: serve both.
#: Which fractal. Two genuinely different families, not one with knobs:
#: `orbit` is the orbit-fold of `fractal_travel.py` v2.1.0, whose CPU path
#: antialiases by walking a sub-pixel grid ACROSS FOUR FRAMES; `cascade` is
#: the fold-inversion of v1.0.0, which takes all four samples INSIDE one
#: frame and is four times the work per pixel because of it.
#: THIS MODULE HAD NO LOGGER. Every `LOG` call added to it raised
#: NameError, and because those calls sit in the `except` blocks that report
#: failures, each one replaced a real error with a NameError about the
#: reporting -- so a widget that could not be built said nothing anyone
#: could act on.
LOG = logging.getLogger("spacr.qt.widgets.fractal_travel")

# ONE LIST, in the Qt-free module, because `preferences` needs it at
# import time and cannot import this one. Two copies is how a pattern
# came to be selectable in code and absent from the Preferences combo.
from ..fractal_defaults import (GPU_ONLY_PATTERNS,  # noqa: E402
                                PATTERNS)
PATTERN_LABELS: Final[dict] = {
    "orbit": "Orbit fold (temporal 2x2)",
    # A SECOND ENTRY, NOT A BACKEND SWITCH. Same map, but the CPU path
    # jitters four samples across four FRAMES -- averaging four different
    # animation times -- while this takes four samples of one instant.
    # They are not the same picture, so one setting drawing either would
    # mean the setting no longer says what appears. Instruction 327 (5).
    "orbit_gpu": "Orbit fold (sharp, GPU 2x2)",
    "cascade": "Fold-inversion cascade (spatial 2x2)",
    "space": "Space (star field flight)",
    "mandelbrot": "Mandelbrot (perturbation deep zoom)",
}
from ..fractal_defaults import (DEFAULT_PATTERN,  # noqa: E402
                               FALLBACK_PATTERN)

DEFAULT_BACKEND: Final[str] = "auto"
DEFAULT_QUALITY: Final[str] = "auto"
DEFAULT_SCALE: Final[float] = 1.0
DEFAULT_SPEED: Final[float] = 4.0
DEFAULT_DREAM: Final[float] = 1.5
DEFAULT_VARIABLE_SPEED: Final[bool] = False
#: The pointer pulls the pattern toward it, and a click shoves
#: it away. On by default: it is the thing that makes the
#: backdrop feel answerable rather than merely present.
DEFAULT_FOLLOW_POINTER: Final[bool] = True
#: How far the pointer reaches by default: the widget's short edge.
DEFAULT_POINTER_SIZE: Final[float] = 1.0
#: How hard it pulls by default.
DEFAULT_POINTER_STRENGTH: Final[float] = 1.0
#: The bounds `variable_speed` sweeps between. They bracket DEFAULT_SPEED so
#: turning it on changes the RANGE and not the average pace.
DEFAULT_SPEED_MIN: Final[float] = 2.0
DEFAULT_SPEED_MAX: Final[float] = 6.0
#: SECONDS FOR ONE FULL SWEEP, slow to fast and back. This is the "how
#: gradually" control: a larger number is a slower change, not a slower
#: fractal. Below about ten seconds it stops reading as drift and starts
#: reading as a pulse.
DEFAULT_SPEED_PERIOD: Final[float] = 41.0

#: Fallback sweep period, used when a caller supplies none.
_VARIABLE_SPEED_PERIOD: Final[float] = DEFAULT_SPEED_PERIOD


from .popup_state import a_popup_is_on_screen

class Pointer:
    """Where the pointer is, and whether it is pushing.

    SAMPLED, NEVER RECEIVED. The backdrop sits behind every control; a
    widget that accepted mouse events would eat the click meant for the
    button on top of it. So nothing here is a mouse handler -- the position
    is read from `QCursor.pos()` on the render tick, and the buttons from
    `QApplication.mouseButtons()`, both of which are global state that costs
    nothing and steals nothing.

    Coordinates come back in the -1..1 space the fractals already work in,
    with (0, 0) at the centre, so a kernel can use them without knowing
    anything about widgets.
    """

    __slots__ = ("x", "y", "pull", "push", "inside", "drag_x", "drag_y",
                 "_last_x", "_last_y", "_dragging")

    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        #: How strongly the pattern is drawn toward the pointer, 0..1.
        self.pull = 0.0
        #: How strongly it is pushed away. Negative pull, kept separate so a
        #: kernel can shape the two differently -- a shove is not a tug
        #: backwards.
        self.push = 0.0
        self.inside = False
        #: How far the pointer moved while held down, since the last frame,
        #: in the same -1..1 space. Consumed by the renderer and reset, so a
        #: frame that is dropped does not lose the movement -- it arrives
        #: with the next one instead.
        self.drag_x = 0.0
        self.drag_y = 0.0
        self._last_x = 0.0
        self._last_y = 0.0
        self._dragging = False

    def sample(self, widget, size: float = 1.0,
               strength: float = 1.0) -> "Pointer":
        """Read the pointer relative to ``widget``. Never raises.

        :param widget: visible Qt widget whose global rectangle defines the
            returned centred coordinates and inside/outside state. Coordinates
            use the short edge as their scale, so that axis maps to -1..1 and
            the long axis may extend beyond it.
        :param size: how far the effect reaches in short-edge-normalised
            coordinate units; 1.0 reaches the widget's short edge.
        :param strength: how hard it pulls, 0 to 2.
        """
        try:
            from PySide6.QtGui import QCursor
            from PySide6.QtWidgets import QApplication

            if widget is None or not widget.isVisible():
                self.inside = False
                return self
            local = widget.mapFromGlobal(QCursor.pos())
            width = max(1, widget.width())
            height = max(1, widget.height())
            denominator = float(min(width, height))
            self.x = (2.0 * local.x() - width) / denominator
            self.y = (height - 2.0 * local.y()) / denominator
            self.inside = (0 <= local.x() < width and 0 <= local.y() < height)

            buttons = QApplication.mouseButtons()
            from PySide6.QtCore import Qt

            left = bool(buttons & Qt.MouseButton.LeftButton)
            right = bool(buttons & Qt.MouseButton.RightButton)
            if not self.inside:
                # Off the widget it neither pulls nor pushes, rather than
                # pulling toward an edge it is nowhere near.
                self.pull = max(0.0, self.pull - 0.08)
                self.push = max(0.0, self.push - 0.15)
                return self
            # SIZE IS A REACH, not a hard edge. Beyond it the pull falls to
            # nothing smoothly, so a pointer crossing the boundary does not
            # snap the picture. `size` is in the same -1..1 space as the
            # coordinates, so 1.0 reaches the short edge of the widget.
            distance = math.hypot(self.x, self.y)
            reach = max(0.05, float(size))
            within = clamp(1.0 - (distance / reach), 0.0, 1.0)
            wanted_pull = 0.0 if (left or right) else within
            wanted_push = within if left else (0.6 * within if right else 0.0)
            # EASED, not switched. A pull that appears the instant the
            # pointer enters reads as a glitch; this reaches full strength
            # over about a second and lets go about as fast.
            self.pull += 0.06 * (wanted_pull * float(strength) - self.pull)
            self.push += 0.25 * (wanted_push * float(strength) - self.push)

            # DRAG THE VIEW. Asked for 2026-08-28: "best would be if the
            # user could drag the visual field with the mouse by clicking
            # and mooving the mouse."
            #
            # ACCUMULATED, not assigned: the renderer consumes this and
            # zeroes it, so a frame dropped under load does not lose the
            # movement -- it arrives with the next frame instead, and a
            # slow machine pans by the same total distance as a fast one.
            held = left or right
            if held and self._dragging:
                self.drag_x += self.x - self._last_x
                self.drag_y += self.y - self._last_y
            self._dragging = held
            self._last_x = self.x
            self._last_y = self.y
        except Exception:                                    # noqa: BLE001
            self.inside = False
        return self


def clamp(value: float, low: float, high: float) -> float:
    """Return ``value`` limited to the inclusive ``low``/``high`` range."""
    return low if value < low else high if value > high else value


@dataclass(frozen=True)
class Settings:
    """What the picture is made of. Every field is a Preferences row."""

    pattern: str = DEFAULT_PATTERN
    backend: str = DEFAULT_BACKEND
    quality: str = DEFAULT_QUALITY
    scale: float = DEFAULT_SCALE
    fps: int = 60
    cpu_threads: Optional[int] = None

    def validated(self) -> "Settings":
        """A copy with every field inside the range the renderers accept.

        Clamped rather than refused: this is a backdrop, and a preferences
        file with a silly number in it must not stop the application from
        drawing one.
        """
        return Settings(
            pattern=self.pattern if self.pattern in PATTERNS else DEFAULT_PATTERN,
            backend=self.backend if self.backend in BACKENDS else DEFAULT_BACKEND,
            quality=self.quality if self.quality in QUALITIES else DEFAULT_QUALITY,
            scale=clamp(float(self.scale), 0.25, 2.0),
            fps=int(clamp(float(self.fps), 15, 240)),
            cpu_threads=(None if self.cpu_threads is None
                         else max(1, int(self.cpu_threads))),
        )


@dataclass
class RuntimeControls:
    """What the user can move while it is running."""

    speed: float = DEFAULT_SPEED
    dream: float = DEFAULT_DREAM
    variable_speed: bool = DEFAULT_VARIABLE_SPEED
    #: Whether the pointer pulls the pattern about. Off leaves the kernels
    #: taking a zero, which costs nothing measurable.
    follow_pointer: bool = DEFAULT_FOLLOW_POINTER
    #: How far the pointer's pull reaches, in the -1..1 coordinate space.
    pointer_size: float = DEFAULT_POINTER_SIZE
    #: How hard it pulls. 0 is off; above 1 exaggerates.
    pointer_strength: float = DEFAULT_POINTER_STRENGTH
    #: A multiplier on the Mandelbrot descent, changed by Up and Down and
    #: by the wheel while the backdrop is running -- as in the source this
    #: pattern came from.
    zoom_rate: float = 1.0
    #: Bumped to ask the dive to start again from the surface. A COUNTER
    #: rather than a flag, so the canvas can tell "asked again" from "still
    #: asked" without anyone having to clear it -- and two changes made in
    #: quick succession are two restarts, not one that may be missed.
    restart_token: int = 0
    speed_min: float = DEFAULT_SPEED_MIN
    speed_max: float = DEFAULT_SPEED_MAX
    speed_period: float = DEFAULT_SPEED_PERIOD

    def speed_at(self, t: float) -> float:
        """The speed to use at ``t`` seconds.

        Constant `speed` unless `variable_speed` is on, in which case it
        sweeps between `speed_min` and `speed_max` -- named bounds rather
        than a hidden percentage, so what the travel will actually do is
        readable from the settings instead of inferred from watching it.
        `speed_period` is how long one full sweep takes, which is the "how
        gradually" control: a larger number is a slower CHANGE, not a slower
        fractal.

        The bounds are used in whichever order they are given: a min above a
        max is a swapped pair, not an empty range, and refusing to animate
        would be a worse answer than animating between the two numbers.
        """
        if not self.variable_speed:
            return max(0.05, self.speed)
        low = min(self.speed_min, self.speed_max)
        high = max(self.speed_min, self.speed_max)
        middle = 0.5 * (low + high)
        half = 0.5 * (high - low)
        # A period of zero would divide by zero; a tiny one is a strobe.
        period = max(1.0, float(self.speed_period or _VARIABLE_SPEED_PERIOD))
        swept = middle + half * math.sin(2.0 * math.pi * t / period)
        return max(0.05, swept)


@dataclass(frozen=True)
class HardwareProfile:
    """CPU capacity used to choose a conservative automatic render quality.

    :param logical_cpus: logical processors available to the application.
    """

    logical_cpus: int

    @staticmethod
    def detect() -> "HardwareProfile":
        return HardwareProfile(logical_cpus=max(1, os.cpu_count() or 1))


def resolved_quality(requested: str, backend: str,
                     hardware: HardwareProfile) -> str:
    """Resolve ``auto`` to a quality the selected backend can sustain.

    Explicit quality names pass through unchanged. GPU auto mode uses the
    balanced profile because GPU capacity is otherwise unknown; CPU auto mode
    uses high only when at least sixteen logical processors are available.

    :param requested: ``auto``, ``balanced``, ``high``, or a future explicit
        quality name.
    :param backend: resolved renderer backend, normally ``gpu`` or ``cpu``.
    :param hardware: detected logical-CPU capacity.
    :returns: the explicit quality name to apply.
    """
    if requested != "auto":
        return requested
    if backend == "gpu":
        # CONSERVATIVE ON A GPU TOO. A first impression that stutters is
        # worse than one that is merely plain, and this cannot know whether
        # the card is a workstation's or a laptop's -- "high" here would
        # have asked every machine ever made for four samples a pixel at
        # native resolution before anyone chose it.
        #
        # A GPU still gets more than a CPU: the per-backend budgets below
        # give it a wider frame and a higher frame-rate cap at the same
        # level, which is the headroom, without guessing at the hardware.
        return "balanced"
    # AND THE CPU IS ASKED FOR EVIDENCE FIRST. Sixteen cores is a machine
    # that can spare some; anything less gets the light profile.
    return "high" if hardware.logical_cpus >= 16 else "balanced"


def resolved_cpu_threads(settings: Settings,
                         hardware: HardwareProfile) -> int:
    """How many Numba workers to take, leaving the application some.

    Capped at 24 because beyond that the scheduling overhead grows for this
    image size, and capped below the machine's count because a backdrop that
    takes every core starves the run the user actually cares about.
    """
    numba_limit = hardware.logical_cpus
    if numba_config is not None:
        try:
            numba_limit = min(numba_limit, int(numba_config.NUMBA_NUM_THREADS))
        except Exception:                                    # noqa: BLE001
            pass
    available = max(1, min(hardware.logical_cpus, numba_limit, 24))
    if settings.cpu_threads is not None:
        return max(1, min(available, settings.cpu_threads))
    if available <= 2:
        return 1
    if available <= 6:
        return available - 1
    return max(2, min(available - 2, round(available * 0.78)))


def platform_can_do_opengl() -> bool:
    """Whether this Qt platform can host a GL canvas at all.

    CHECKED BEFORE THE GPU WIDGET IS BUILT, because getting it wrong does not
    raise. On the `offscreen` platform Qt prints "QOpenGLWidget is not
    supported on this platform" and the process DUMPS CORE -- which no
    `except` around the constructor can catch, so the fallback in
    `create_fractal_widget` would never run. Every test and every headless
    launch is that platform.
    """
    # SAFE MODE REFUSES GL OUTRIGHT. `safespacr` exists because the crash
    # log points here, and a safe start that still built a GL canvas would
    # be no safer than the ordinary one. Read from the environment because
    # the context can be created before any preference has been read.
    if os.environ.get("SPACR_NO_GL"):
        return False
    platform = str(os.environ.get("QT_QPA_PLATFORM", "")).strip().lower()
    if platform.startswith(("offscreen", "minimal", "vnc")):
        return False
    if platform in ("", "xcb", "wayland", "cocoa", "windows"):
        if platform in ("cocoa", "windows"):
            return True
        # AN EMPTY VALUE MEANS QT WILL CHOOSE, and what it chooses decides
        # whether DISPLAY is the right question. On X11 and Wayland it is:
        # no DISPLAY, no GL. On macOS and Windows it is not -- neither sets
        # DISPLAY, both always have a window server, and Qt picks cocoa or
        # windows without being told.
        #
        # THIS TEST USED TO ASK DISPLAY REGARDLESS, so on every Mac -- where
        # QT_QPA_PLATFORM is normally unset -- it answered no and the
        # spaceout fractal ran its Numba CPU renderer instead of its shader.
        # Measured on the reporting iMac: VisPy opens a context on that
        # machine reporting GL_RENDERER "AMD Radeon Pro 5300 OpenGL Engine",
        # so the card was there and working the whole time and the
        # environment heuristic was what said otherwise.
        if not platform and sys.platform in ("darwin", "win32"):
            return True
        return bool(os.environ.get("DISPLAY")
                    or os.environ.get("WAYLAND_DISPLAY"))
    return True


def gpu_is_available() -> bool:
    """Whether the GPU renderer can be built at all.

    Asked rather than assumed, and asked WITHOUT importing vispy into the
    application when the answer is no: `importlib.util.find_spec` looks the
    module up without executing it, so a missing vispy costs nothing and a
    present one is not initialised twice.

    A platform that cannot host a GL canvas counts as no GPU, because the
    alternative is a core dump rather than an exception.
    """
    import importlib.util

    if not platform_can_do_opengl():
        return False
    try:
        return importlib.util.find_spec("vispy") is not None
    except Exception:                                        # noqa: BLE001
        return False


def resolve_backend(requested: str) -> str:
    """Which renderer will actually run, given what is installed.

    :returns: ``'gpu'`` or ``'cpu'`` -- never ``'auto'``, because a caller
        showing the user which one they are on cannot show them "auto".
    """
    wanted = requested if requested in BACKENDS else DEFAULT_BACKEND
    if wanted == "cpu":
        return "cpu"
    if wanted == "gpu":
        return "gpu"
    return "gpu" if gpu_is_available() else "cpu"


# ===========================================================================
# CPU backend -- orbit-fold, evaluated at animation rate
# ===========================================================================

_FAST_PI: Final[float] = math.pi
_FAST_TWO_PI: Final[float] = 2.0 * math.pi

if njit is not None:

    @njit(inline="always", fastmath=True)
    def _fast_sin(value):
        """Bounded sine approximation. Good enough for a picture, and the
        kernel calls it a dozen times per pixel per iteration."""
        value -= math.floor((value + _FAST_PI) / _FAST_TWO_PI) * _FAST_TWO_PI
        result = (1.2732395447351627 * value
                  - 0.4052847345693511 * value * abs(value))
        return 0.225 * (result * abs(result) - result) + result

    @njit(inline="always", fastmath=True)
    def _fast_cos(value):
        return _fast_sin(value + 0.5 * _FAST_PI)

    @njit(inline="always", fastmath=True)
    def _orbit_sample(px, py, width, height, t, speed, dream, iterations,
                      pointer_x, pointer_y, pull, push):
        denominator = float(min(width, height))
        x = (2.0 * px - width) / denominator
        y = (height - 2.0 * py) / denominator

        # THE POINTER BENDS THE PLANE, it does not move the camera. Warping
        # the sample position pulls the STRUCTURE toward the cursor and
        # leaves the travel alone, so the fractal still goes where it was
        # going -- which a camera shove would not.
        if pull > 0.0 or push > 0.0:
            to_x = pointer_x - x
            to_y = pointer_y - y
            distance2 = to_x * to_x + to_y * to_y + 0.05
            # 1/r falloff: firm near the pointer, gone by the far corner.
            strength = (0.55 * pull - 0.95 * push) / distance2
            if strength > 0.9:
                strength = 0.9
            elif strength < -1.4:
                strength = -1.4
            x += strength * to_x
            y += strength * to_y

        rotation = (0.24 * _fast_sin(0.17 * t)
                    + 0.11 * _fast_sin(0.043 * t + 1.2))
        cs = _fast_cos(rotation)
        sn = _fast_sin(rotation)
        tx = cs * x - sn * y
        ty = sn * x + cs * y

        drift_x = dream * (0.10 * _fast_sin(0.071 * t)
                           + 0.04 * _fast_sin(0.019 * t + 1.3))
        drift_y = dream * (0.09 * _fast_cos(0.063 * t + 0.4)
                           + 0.04 * _fast_sin(0.023 * t + 2.1))
        stretch_x = math.exp(0.10 * dream * _fast_sin(0.041 * t))
        stretch_y = math.exp(0.09 * dream * _fast_cos(0.037 * t + 0.8))
        shear_x = 0.12 * dream * _fast_sin(0.052 * t + 0.6)
        shear_y = 0.07 * dream * _fast_cos(0.047 * t)

        old_x = tx
        tx = stretch_x * tx + shear_x * ty + drift_x
        ty = stretch_y * ty + shear_y * old_x + drift_y

        radius_squared = tx * tx + ty * ty + 1e-4
        inverse_radius = 1.0 / math.sqrt(radius_squared)
        radial_phase = 0.80 * _fast_sin(
            5.5 * math.log(radius_squared + 0.03) + 0.42 * t * speed)
        tx += 0.10 * dream * radial_phase * tx * inverse_radius
        ty += 0.10 * dream * radial_phase * ty * inverse_radius

        constant_x = (0.73 + 0.08 * _fast_sin(0.11 * t)
                      + 0.05 * _fast_sin(0.031 * t + 2.0))
        constant_y = (0.48 + 0.10 * _fast_cos(0.13 * t + 0.7)
                      + 0.04 * _fast_sin(0.037 * t))

        orbit_a = 0.0
        orbit_b = 0.0
        orbit_c = 0.0
        previous_radius = 1e9
        ox = tx
        oy = ty

        for iteration in range(iterations):
            ox = abs(ox)
            oy = abs(oy)
            if ox < oy:
                ox, oy = oy, ox
            ox = abs(ox - 0.45 * oy)
            current_radius = ox * ox + oy * oy + 0.055
            ox = ox / current_radius - constant_x
            oy = oy / current_radius - constant_y

            radius_change = abs(current_radius - previous_radius)
            previous_radius = current_radius
            orbit_a += 1.0 / (1.0 + 12.0 * abs(current_radius - 0.42))
            orbit_b += 1.0 / (1.0 + 9.0 * abs(ox - oy))
            orbit_c += 1.0 / (1.0 + 18.0 * radius_change)

            next_x = ox + 0.035 * dream * _fast_sin(
                1.7 * oy + 0.19 * t + iteration)
            next_y = oy + 0.035 * dream * _fast_cos(
                1.5 * ox - 0.17 * t - iteration)
            ox = next_x
            oy = next_y

        inverse_iterations = 1.0 / iterations
        orbit_a *= inverse_iterations
        orbit_b *= inverse_iterations
        orbit_c *= inverse_iterations

        # Three orbit traps drive the phase rather than one escape count,
        # which is what keeps the colour moving where a Mandelbrot would band.
        palette_phase = (5.2 * orbit_a + 3.7 * orbit_b + 2.3 * orbit_c
                         + 0.075 * t)
        red = 0.50 + 0.43 * _fast_cos(palette_phase + 0.15) + 0.12 * orbit_c
        green = 0.48 + 0.42 * _fast_cos(palette_phase + 2.25) + 0.11 * orbit_a
        blue = 0.50 + 0.45 * _fast_cos(palette_phase + 4.35) + 0.13 * orbit_b

        glow = max(0.0, min(1.0, 1.4 * orbit_a * orbit_b))
        red += 0.15 * glow
        green += 0.10 * glow
        blue += 0.24 * glow

        # The vignette is what lets controls sit on top and stay readable.
        screen_radius = math.sqrt(x * x + y * y)
        vignette = 1.0 - max(0.0, min(1.0, (screen_radius - 0.55) / 1.30))
        brightness = 0.78 + 0.22 * vignette

        red = max(0.0, min(1.0, red)) * brightness
        green = max(0.0, min(1.0, green)) * brightness
        blue = max(0.0, min(1.0, blue)) * brightness
        return int(255.0 * red), int(255.0 * green), int(255.0 * blue)

    @njit(cache=True, parallel=True, fastmath=True, nogil=True)
    def _render_into(output, t, speed, dream, iterations, jitter_x, jitter_y,
                     pointer_x, pointer_y, pull, push):
        height, width, _channels = output.shape
        for y in prange(height):
            for x in range(width):
                red, green, blue = _orbit_sample(
                    x + jitter_x, y + jitter_y, width, height,
                    t, speed, dream, iterations,
                    pointer_x, pointer_y, pull, push)
                output[y, x, 0] = red
                output[y, x, 1] = green
                output[y, x, 2] = blue

    @njit(cache=True, parallel=True, fastmath=True, nogil=True)
    def _blend_temporal(ring, output, newest):
        """Combine the four jitter phases with a short temporal weighting."""
        height, width, _channels = output.shape
        previous_1 = (newest - 1) % 4
        previous_2 = (newest - 2) % 4
        previous_3 = (newest - 3) % 4
        for y in prange(height):
            for x in range(width):
                for channel in range(3):
                    value = (0.62 * ring[newest, y, x, channel]
                             + 0.22 * ring[previous_1, y, x, channel]
                             + 0.10 * ring[previous_2, y, x, channel]
                             + 0.06 * ring[previous_3, y, x, channel])
                    output[y, x, channel] = int(value)

else:                                                        # pragma: no cover

    def _render_into(*_args, **_kwargs):
        raise RuntimeError("numba is required for the CPU fractal backend")

    def _blend_temporal(*_args, **_kwargs):
        raise RuntimeError("numba is required for the CPU fractal backend")


#: The four 2x2 sub-pixel positions, walked one per frame.
JITTERS: Final[tuple[tuple[float, float], ...]] = (
    (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
)


class OrbitEngine:
    """The four-frame temporal window, and nothing else.

    Holds no keyframes: the only state is the ring of the last four jitter
    phases, which is what the antialiasing needs and all it needs.

    :param thread_count: worker threads to render with. Clamped to at least
        one, so a caller that computed zero from an unavailable CPU count
        still renders.
    """

    def __init__(self, thread_count: int) -> None:
        self.thread_count = max(1, int(thread_count))
        self.width = 0
        self.height = 0
        self.ring: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None
        self.slot = 0
        self.frames = 0

    def _ensure_size(self, width: int, height: int) -> None:
        if width == self.width and height == self.height and self.ring is not None:
            return
        self.width = width
        self.height = height
        self.ring = np.empty((4, height, width, 3), dtype=np.uint8)
        self.output = np.empty((height, width, 3), dtype=np.uint8)
        self.slot = 0
        self.frames = 0

    def render(self, width: int, height: int, t: float, speed: float,
               dream: float, iterations: int, pointer_x: float = 0.0,
               pointer_y: float = 0.0, pull: float = 0.0,
               push: float = 0.0) -> np.ndarray:
        set_num_threads(self.thread_count)
        self._ensure_size(width, height)
        jitter_x, jitter_y = JITTERS[self.slot]
        _render_into(self.ring[self.slot], t, speed, dream, iterations,
                     jitter_x, jitter_y, pointer_x, pointer_y, pull, push)
        if self.frames == 0:
            # Fill the unused history with the first real sample, so the
            # opening frame is the picture rather than a fade up from black.
            for index in range(4):
                if index != self.slot:
                    self.ring[index, :, :, :] = self.ring[self.slot, :, :, :]
        _blend_temporal(self.ring, self.output, self.slot)
        self.slot = (self.slot + 1) % 4
        self.frames += 1
        return self.output.copy()


# ===========================================================================
# The CPU widget -- a render thread, and a paint that never computes
# ===========================================================================

def _quit_and_join_thread(thread) -> None:
    """Stop ``thread`` and do not return while its worker can still run.

    Five seconds is the normal shutdown budget.  The CPU renderer's first
    frame can spend longer compiling a Numba kernel, though, and destroying a
    live ``QThread`` is a process-fatal Qt error.  Once the soft deadline is
    missed there is no safe detached state: wait for that finite render to
    finish before allowing the native wrapper to be freed.
    """
    try:
        thread.quit()
        if not thread.wait(5000):
            LOG.warning("fractal renderer exceeded the shutdown deadline")
            thread.wait()
    except Exception:                                        # noqa: BLE001
        # Shutdown is also reached while Qt is tearing its own wrappers down.
        # A wrapper that is already gone has no live thread left to join.
        pass


def _join_on_destroy(widget, thread) -> None:
    """Quit and wait for ``thread`` when Qt frees ``widget``.

    The handler closes over the THREAD only. `destroyed` is emitted while the
    widget is being torn down, so anything that touched the widget from here
    would be reaching into a half-freed object -- which is a second crash on
    top of the one this prevents.
    """
    def _join(*_args):
        """Stop and join the render thread when the widget is destroyed."""
        try:
            _quit_and_join_thread(thread)
        except Exception:                                    # noqa: BLE001
            pass

    try:
        widget.destroyed.connect(_join)
    except Exception:                                        # noqa: BLE001
        pass


def _make_cpu_widget(settings: Settings, controls: RuntimeControls,
                     hardware: HardwareProfile):
    if njit is None:
        raise RuntimeError("numba is required for the CPU fractal backend")

    from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtWidgets import QApplication, QWidget

    quality = resolved_quality(settings.quality, "cpu", hardware)
    thread_count = resolved_cpu_threads(settings, hardware)
    cascade = settings.pattern == "cascade"

    # EACH PATTERN'S OWN BUDGET. The cascade evaluates four samples per pixel
    # inside one frame where the orbit evaluates one and averages across
    # frames, so it renders roughly a quarter of the pixels and holds a lower
    # cap to spend the same wall-clock. Sharing one set of numbers would make
    # one of them either wasteful or unusable.
    if settings.pattern == "space":
        from .fractal_space import SpaceEngine

        engine_factory = SpaceEngine
        # MOSTLY EMPTY SKY IS CHEAP. Each pixel walks six layers of a 3x3
        # neighbourhood and three object slots, and almost every cell misses
        # -- so it carries a wider frame and a higher cap than either fold.
        iterations = 0
        target_fps = max(15, min(settings.fps, 30 if quality == "balanced" else 26))
        base_pixels = 300_000.0 if quality == "balanced" else 460_000.0
    elif cascade:
        from .fractal_cascade import CascadeEngine

        engine_factory = CascadeEngine
        iterations = 4 if quality == "balanced" else 5
        target_fps = max(15, min(settings.fps, 24 if quality == "balanced" else 20))
        base_pixels = 115_000.0 if quality == "balanced" else 190_000.0
    else:
        engine_factory = OrbitEngine
        iterations = 5 if quality == "balanced" else 6
        # Capped at 30 because every displayed frame is newly evaluated --
        # there is no keyframe to re-project, so a higher cap only burns
        # cores.
        target_fps = max(15, min(settings.fps, 30))
        # A pixel COUNT, not a percentage: on a 4K display a percentage is an
        # unbounded promise, and this is a backdrop.
        base_pixels = 460_000.0 if quality == "balanced" else 680_000.0

    target_period = 1.0 / target_fps
    base_pixels *= settings.scale * settings.scale

    class _Worker(QObject):
        frame_ready = Signal(object, float)
        failed = Signal(str)

        def __init__(self) -> None:
            """Build the engine this thread will shade with."""
            super().__init__()
            self.engine = engine_factory(thread_count)

        @Slot(object)
        def render(self, request: object) -> None:
            """Shade one frame and hand it back, if anyone is still there.

            A FRAME CAN FINISH AFTER ITS WIDGET IS GONE. The shading runs on
            this thread while the GUI thread may be closing the window or
            swapping the pattern, and Qt deletes the worker's C++ side with
            it -- so `emit` raises "Signal source has been deleted". The
            except below then emitted the FAILURE signal, which raised the
            same way, and an exception escaping a slot on a QThread takes
            the process down: "Aborted (core dumped)".
            """
            try:
                started = time.perf_counter()
                frame = self.engine.render(
                    request["width"], request["height"], request["t"],
                    request["speed"], request["dream"], request["iterations"],
                    request.get("pointer_x", 0.0),
                    request.get("pointer_y", 0.0),
                    request.get("pull", 0.0), request.get("push", 0.0))
            except Exception as error:                       # noqa: BLE001
                self._say_something(self.failed,
                                    f"{type(error).__name__}: {error}")
                return
            self._say_something(self.frame_ready, frame,
                                time.perf_counter() - started)

        @staticmethod
        def _say_something(signal, *args) -> None:
            """Emit, unless the object that owns the signal has been freed.

            The last thing this thread does with a widget that is being
            destroyed, so it must never raise: nothing is listening, and an
            exception here ends the process rather than the frame.
            """
            try:
                signal.emit(*args)
            except RuntimeError:
                # "Signal source has been deleted" -- the widget went away
                # while this frame was being shaded. There is nobody to tell.
                pass
            except Exception:                                # noqa: BLE001
                LOG.debug("could not deliver a frame", exc_info=True)

    class CpuFractalWidget(QWidget):
        """The orbit-fold fractal, rendered off the GUI thread.

        `paintEvent` only blits: every fractal evaluation happens on the
        worker thread, so a slow frame makes the picture late and never makes
        the interface late.

        :param parent: parent widget.
        """

        backend_name: Final[str] = "cpu"
        render_requested = Signal(object)

        def __init__(self, parent=None) -> None:
            """Build the CPU canvas, painting its own background."""
            super().__init__(parent)
            self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
            self.setAutoFillBackground(False)

            # NOT PARENTED TO self. A QThread whose parent is deleted while it
            # runs prints "Destroyed while thread is still running" and takes
            # the process down; the backdrop is reparented and deleted with
            # its screen, so that is the ordinary path, not an edge case.
            self._thread = QThread()
            self._worker = _Worker()
            self._worker.moveToThread(self._thread)
            self.render_requested.connect(
                self._worker.render, Qt.ConnectionType.QueuedConnection)
            self._worker.frame_ready.connect(self._accept_frame)
            self._worker.failed.connect(self._on_failure)
            self._thread.finished.connect(self._worker.deleteLater)
            self._thread.start()
            # JOINED WHENEVER QT FREES THE WIDGET. `destroyed` fires during
            # destruction, so the handler must hold the THREAD and never
            # `self` -- reaching for a half-destroyed widget is its own crash.
            _join_on_destroy(self, self._thread)
            # QApplication can tear down its native widgets before Python
            # releases their wrappers.  In that order ``destroyed`` is too
            # late to protect an unparented QThread, so join at the earlier,
            # explicit application-shutdown boundary as well.  The closure
            # captures only the thread; keeping it on ``self`` merely lets an
            # explicit shutdown disconnect the now-unneeded application hook.
            self._app_quit_join = (
                lambda thread=self._thread: _quit_and_join_thread(thread))
            application = QApplication.instance()
            if application is not None:
                application.aboutToQuit.connect(self._app_quit_join)

            self._timer = QTimer(self)
            self._timer.setSingleShot(True)
            self._timer.setTimerType(Qt.TimerType.PreciseTimer)
            self._timer.timeout.connect(self._request_frame)

            self._busy = False
            self._stopped = False
            self._paused = False
            self._sim_time = 0.0
            self._adaptive_scale = 1.0
            self._render_ema: Optional[float] = None
            self._last_render_seconds: Optional[float] = None
            self._actual_fps = 0.0
            self._last_arrival = 0.0
            self._frames = 0
            self._render_size = (1, 1)
            self._image = None
            self._image_array = None
            self._error: Optional[str] = None
            #: Read on the GUI thread each tick. The widget never becomes a
            #: mouse target -- see `Pointer`.
            self._pointer = Pointer()
            # Integrated travel, so a speed change does not teleport the
            # camera. One per canvas: it is this canvas's own position on
            # the trajectory.
            self._depth_phase = DepthPhase()
            self._timer.start(30)

        # ---------------------------------------------------- winding down

        def pause(self) -> bool:
            """Stop rendering and leave the last frame on screen.

            Called when a RUN STARTS. A backdrop taking nineteen cores while
            a segmentation is queued is the opposite of what it is for, and
            stopping is better than thinning: a slower fractal still holds
            the threads.

            :returns: True when this call did the stopping, False when it was
                already paused -- so a caller can tell whether to resume.
            """
            if self._paused:
                return False
            self._paused = True
            self._timer.stop()
            return True

        def resume(self) -> bool:
            """Start rendering again from where the clock left off."""
            if not self._paused or self._stopped:
                return False
            self._paused = False
            self._timer.start(10)
            return True

        def is_paused(self) -> bool:
            """Whether the animation is currently held."""
            return self._paused

        def set_animating(self, on: bool) -> bool:
            """`AmbientWidget`'s verb for the same thing.

            The ambient backdrop this replaces is stopped and started with
            `set_animating`, and its callers -- the Home screen's teardown
            among them -- reach for that name. Answering to it makes this a
            drop-in rather than something every call site has to learn.
            """
            return self.resume() if on else self.pause()


        # ------------------------------------------------------ the frames

        def _target_size(self) -> tuple[int, int]:
            # RENDER SCALE, which was a setting nobody read. It is the
            # fraction of the window's own pixels to shade, so 1.0 is native
            # and anything less trades sharpness for speed -- the direct
            # answer to "how do i get the image sharper".
            """The pixel size to shade at, from the render scale and the window.

            RENDER SCALE is the fraction of the window's own pixels to shade: 1.0 is
            native and anything less trades sharpness for speed. It was a setting
            nobody read, and it is the direct answer to "how do I get the image
            sharper".
            """
            try:
                render_scale = float(_render_scale())
            except Exception:                                # noqa: BLE001
                render_scale = 1.0
            # THE ARITHMETIC LIVES IN `target_render_size`, so instruction
            # 327's "measure before you change anything" can be done
            # without a GL context.
            return target_render_size(
                self.width(), self.height(), self.devicePixelRatioF(),
                render_scale, base_pixels, self._adaptive_scale)

        @Slot()
        def _request_frame(self) -> None:
            """Ask the worker for the next frame, unless stopped or paused."""
            if self._stopped or self._paused:
                return
            if self._busy or not self.isVisible():
                self._timer.start(50)
                return
            width, height = self._target_size()
            self._render_size = (width, height)
            self._busy = True
            # SAMPLED ON THE GUI THREAD, sent to the worker as numbers.
            # QCursor and QApplication are not safe to touch from the render
            # thread, and by the time the frame is drawn the pointer has
            # moved anyway -- so the position that matters is the one when
            # the frame was ASKED for.
            pointer = self._pointer.sample(
                self, controls.pointer_size, controls.pointer_strength)
            self.render_requested.emit({
                "width": width, "height": height, "t": self._sim_time,
                "speed": controls.speed_at(self._sim_time),
                "dream": controls.dream, "iterations": iterations,
                "pointer_x": pointer.x, "pointer_y": pointer.y,
                "pull": pointer.pull if controls.follow_pointer else 0.0,
                "push": pointer.push if controls.follow_pointer else 0.0,
            })
            self._sim_time += target_period

        def _adapt_resolution(self) -> None:
            """Trade resolution for frame rate, from the measured render time.

            WAITS FOR TWELVE FRAMES and then only reconsiders every twenty-fourth,
            so the scale settles instead of oscillating on a single slow frame. The
            budget is 78% of the period rather than all of it, because Qt's own
            conversion and the rest of the application have to fit in the remainder.

            Both directions are damped and clamped -- down no further than 0.58, up
            no further than 1.35 -- so a stall cannot drive the picture to nothing
            and a fast machine cannot drive it past what the window can show.
            """
            if self._render_ema is None or self._frames < 12:
                return
            if self._frames % 24 != 0:
                return
            # About 22% of the period is left for Qt's conversion, the scale
            # and the rest of the application.
            budget = 0.78 * target_period
            ratio = budget / max(1e-6, self._render_ema)
            if ratio < 0.92:
                factor = max(0.82, math.sqrt(ratio) * 0.98)
                self._adaptive_scale = max(0.58, self._adaptive_scale * factor)
            elif ratio > 1.65:
                factor = min(1.055, ratio ** 0.16)
                self._adaptive_scale = min(1.35, self._adaptive_scale * factor)

        @Slot(object, float)
        def _accept_frame(self, frame, render_seconds: float) -> None:
            """Take a rendered frame, note how long it took, and repaint."""
            if self._stopped:
                return
            from PySide6.QtGui import QImage

            height, width, _channels = frame.shape
            # The array is kept alive alongside the QImage: QImage does not
            # copy, and a freed buffer paints garbage or crashes.
            self._image_array = frame
            self._image = QImage(frame.data, width, height, frame.strides[0],
                                 QImage.Format.Format_RGB888)
            self._last_render_seconds = max(1e-6, render_seconds)
            self._render_ema = (
                self._last_render_seconds if self._render_ema is None
                else 0.82 * self._render_ema + 0.18 * self._last_render_seconds)
            self._busy = False
            self._frames += 1

            now = time.perf_counter()
            if self._last_arrival > 0.0:
                rate = 1.0 / max(1e-5, now - self._last_arrival)
                self._actual_fps = (rate if self._actual_fps <= 0.0
                                    else 0.88 * self._actual_fps + 0.12 * rate)
            self._last_arrival = now
            self._adapt_resolution()
            self.update()
            if self._paused or self._stopped:
                return
            delay = max(0.0, target_period - self._last_render_seconds)
            self._timer.start(max(1, round(1000.0 * delay)))

        @Slot(str)
        def _on_failure(self, message: str) -> None:
            """Record the worker's error and stop treating a frame as pending."""
            self._error = message
            self._busy = False
            self.update()
            if not (self._paused or self._stopped):
                self._timer.start(750)

        def paintEvent(self, _event) -> None:
            """Draw the last frame, or the ground colour before there is one."""
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(5, 5, 10))
            if self._image is not None:
                painter.setRenderHint(
                    QPainter.RenderHint.SmoothPixmapTransform, True)
                painter.drawImage(self.rect(), self._image)
            painter.end()

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            if not (self._busy or self._stopped or self._paused):
                self._timer.start(10)

        def stats_text(self) -> str:
            width, height = self._render_size
            if self._paused:
                timing = "paused for a run"
            elif self._last_render_seconds is None:
                timing = "compiling"
            else:
                timing = f"{1000.0 * self._last_render_seconds:.1f} ms frame"
            error = "" if self._error is None else f"\n{self._error}"
            aa = "spatial 2x2" if cascade else "temporal 2x2"
            return (f"v{VERSION} · CPU/{quality} · {settings.pattern} · {aa}\n"
                    f"{width}×{height} · {self._actual_fps:.1f} fps · "
                    f"{thread_count} threads\n{timing}{error}")

        def shutdown(self) -> None:
            """Stop for good and join the thread. Safe to call twice."""
            if self._stopped:
                return
            self._stopped = True
            self._timer.stop()
            application = QApplication.instance()
            if application is not None:
                try:
                    application.aboutToQuit.disconnect(self._app_quit_join)
                except (RuntimeError, TypeError):
                    pass
            _quit_and_join_thread(self._thread)

        def closeEvent(self, event) -> None:
            self.shutdown()
            super().closeEvent(event)

    return CpuFractalWidget()


# ===========================================================================
# GPU backend -- the GLSL field, unchanged from the script
# ===========================================================================

VERTEX_SHADER: Final[str] = """
attribute vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER: Final[str] = """
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_speed;
uniform float u_dream;
uniform float u_palette_phase;
uniform float u_pointer_x;
uniform float u_pointer_y;
uniform float u_pull;
uniform float u_push;
uniform float u_tx;
uniform float u_ty;
uniform float u_rotation;
uniform float u_shear_x;
uniform float u_shear_y;
uniform float u_stretch_x;
uniform float u_stretch_y;
uniform int u_detail;

const float LN10 = 2.302585092994046;

vec3 palette(float x) {
    vec3 a = vec3(0.56, 0.50, 0.45);
    vec3 b = vec3(0.44, 0.46, 0.55);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.05, 0.37, 0.70)
        + vec3(0.17, 0.13, 0.09) * sin(0.1 * u_time);
    return a + b * cos(6.28318 * (c * x + d));
}

vec2 rotate2(vec2 p, float a) {
    float cs = cos(a);
    float sn = sin(a);
    return vec2(cs * p.x - sn * p.y, sn * p.x + cs * p.y);
}

float field(vec2 uv) {
    float t = u_time;
    float depth = t * u_speed / 12.0;
    float log_shift = depth * LN10;

    vec2 p = rotate2(uv, u_rotation);
    p = mat2(u_stretch_x, u_shear_x, u_shear_y, u_stretch_y) * p;
    p += vec2(u_tx, u_ty);

    float total = 0.0;
    float amplitude = 1.0;
    vec2 q = p;

    for (int i = 0; i < 10; ++i) {
        if (i >= u_detail) {
            break;
        }
        float radius = length(q) + 1e-5;
        float angle = atan(q.y, q.x);
        float log_radius = log(radius)
            + log_shift * (0.86 + 0.12 * sin(0.17 * t));
        float petals = sin(
            5.0 * angle + 2.8 * log_radius
            + 0.65 * sin(0.43 * t + 1.8 * q.x));
        float folds = cos(
            7.5 * angle - 2.3 * log_radius
            + 0.50 * cos(0.39 * t - 1.6 * q.y));
        float eyes = sin(
            3.0 * log_radius - 2.0 * angle + 0.7 * sin(t * 0.29));
        float bloom = sin(
            2.5 * q.x + 1.7 * q.y + 0.35 * t + 1.2 * petals);
        float lace = cos(
            4.0 * q.y - 1.5 * q.x - 0.28 * t + 1.3 * folds);
        total += amplitude * (
            0.38 * petals + 0.28 * folds + 0.20 * eyes
            + 0.10 * bloom + 0.04 * lace);
        vec2 warp = vec2(
            sin(1.7 * angle + 1.3 * log_radius + 0.17 * t + 0.8 * folds),
            cos(1.3 * angle - 1.5 * log_radius - 0.19 * t + 0.8 * petals));
        q = rotate2(
            q * 1.55 + 0.35 * u_dream * warp,
            0.42 + 0.08 * sin(0.07 * t));
        amplitude *= 0.58;
    }
    return total;
}


// THE POINTER BENDS THE PLANE, IT DOES NOT MOVE THE CAMERA.
//
// This used to translate the whole plane: `uv - target * pull` shifts
// EVERY pixel by the same amount, which is towing the viewport. Two
// things follow from that, and both were reported. The shift grows with
// the pointer's distance from centre, so near an edge the whole picture
// is dragged; and when the pointer leaves the widget the pull decays to
// zero, so the picture springs back -- "if the mouse is to close to the
// sides of the screen the camera snapps back".
//
// The CPU orbit fold never had either problem, and the user says so:
// "the orbit fold cpu effect is like a magnigying glass, which looks
// cool". This is that same warp, transliterated, so the two renderers
// bend the picture identically:
//
//   * the displacement is TOWARD the pointer and falls off as 1/r^2, so
//     it is firm under the cursor and gone by the far corner;
//   * distant pixels are left where they were, so there is no global
//     shift to spring back from;
//   * a click reverses it, pushing the structure away instead.
//
// The 0.05 floor keeps the divide finite at the pointer itself, and the
// clamps stop a pixel being thrown past it.
vec2 toward_pointer(vec2 uv) {
    vec2 target = vec2(u_pointer_x, u_pointer_y);
    vec2 to_pointer = target - uv;
    float distance2 = dot(to_pointer, to_pointer) + 0.05;
    float strength = (0.55 * u_pull - 0.95 * u_push) / distance2;
    strength = clamp(strength, -1.4, 0.9);
    return uv + strength * to_pointer;
}

vec3 render_sample(vec2 fragment_position) {
    float denominator = min(u_resolution.x, u_resolution.y);
    vec2 uv = (2.0 * fragment_position - u_resolution) / denominator;
    uv *= 1.10;
    uv = toward_pointer(uv);
    uv += 0.06 * u_dream * vec2(
        sin(0.23 * u_time + 1.1 * uv.y),
        cos(0.21 * u_time - 1.1 * uv.x));
    float value = field(uv);
    float glow = 0.5 + 0.5 * sin(
        1.3 * value + u_palette_phase + 0.11 * u_time);
    float rim = exp(-1.5 * dot(uv, uv));
    float palette_index = 0.20 * value + 0.22 * glow + 0.23 * rim
        + 0.07 * sin(0.11 * u_time);
    vec3 color = palette(palette_index);
    float neon = smoothstep(0.45, 0.95, 0.5 + 0.5 * sin(2.7 * value));
    color += vec3(0.25, 0.18, 0.34) * neon * (0.35 + 0.65 * rim);
    color = pow(max(color, vec3(0.0)), vec3(0.85));
    float vignette = 1.0 - smoothstep(0.55, 1.75, length(uv));
    color *= 0.72 + 0.28 * vignette;
    return clamp(color, 0.0, 1.0);
}

void main() {
    vec3 color = vec3(0.0);
    color += render_sample(gl_FragCoord.xy + vec2(-0.25, -0.25));
    color += render_sample(gl_FragCoord.xy + vec2( 0.25, -0.25));
    color += render_sample(gl_FragCoord.xy + vec2(-0.25,  0.25));
    color += render_sample(gl_FragCoord.xy + vec2( 0.25,  0.25));
    gl_FragColor = vec4(0.25 * color, 1.0);
}
"""


@dataclass(frozen=True)
class CameraState:
    """Where the GPU field is looking, at one instant."""

    t: float
    depth: float
    tx: float
    ty: float
    rotation: float
    shear_x: float
    shear_y: float
    stretch_x: float
    stretch_y: float
    palette_phase: float


def target_render_size(logical_width: int, logical_height: int,
                       device_scale: float, render_scale: float,
                       base_pixels: float = 0.0,
                       adaptive_scale: float = 1.0) -> tuple:
    """How many pixels to shade for a widget of this size.

    LIFTED OUT OF THE CANVAS so it can be measured. The backdrop asks
    which of four candidates makes fullscreen choppy while the backdrop
    is smooth, and says to answer with numbers before writing a fix --
    which is not possible while the arithmetic only exists inside a
    nested method on a class that needs a GL context.

    The rule itself is unchanged: shade ``render_scale`` squared of the
    widget's own physical pixels, never more than the widget has and
    never fewer than 180,000, keeping the aspect ratio and an even
    width and height.

    :returns: ``(width, height)`` in physical pixels.
    """
    logical_width = max(320, int(logical_width))
    logical_height = max(180, int(logical_height))
    device_scale = max(1.0, float(device_scale))
    physical_width = max(320, round(logical_width * device_scale))
    physical_height = max(180, round(logical_height * device_scale))
    aspect = physical_width / physical_height

    adaptive_scale = max(0.0, float(adaptive_scale))
    requested = float(base_pixels) * adaptive_scale ** 2
    native = float(physical_width) * float(physical_height)
    if render_scale > 0.0:
        # MULTIPLIED IN, NOT REPLACED. This branch used to overwrite
        # `requested`, which threw the adaptive scale away -- and since
        # `render_scale` defaults above zero, that was every launch.
        #
        # `_adapt_resolution` measures the render time, compares it with
        # the frame budget and computes a new scale between 0.58 and 1.35.
        # All of that ran, and none of it reached the renderer. Measured:
        # sweeping the adaptive scale across its whole range left the
        # shaded size at 1280x720 every time.
        #
        # That is the answer to instruction 327 (1). Fullscreen shades
        # 5.12x a panel's pixels at 2560x1440 and 11.53x at 4K -- the cost
        # IS the area -- but the machinery meant to compensate was
        # disconnected, so the frame rate fell instead of the resolution.
        #
        # The user's own `scale` still means what it says: 0.5 is half
        # native when frames are comfortable. The adaptive term only ever
        # takes it further down, or back up toward it.
        requested = native * render_scale * render_scale * adaptive_scale ** 2

    requested = max(180_000.0, min(native, requested))
    width = int(round(math.sqrt(requested * aspect)))
    height = int(round(width / aspect))
    width = min(physical_width, max(320, width))
    height = min(physical_height, max(180, height))
    width -= width % 2
    height -= height % 2
    return max(320, width), max(180, height)


class DepthPhase:
    """How far along the trajectory the camera is, as a number that only grows.

    SPEED MUST CHANGE THE RATE, NOT THE POSITION. Depth used to be
    ``t * speed``, so a scroll that doubled the speed doubled the depth
    in the same instant: measured at t=60s, speed 1 -> 2 moved the camera
    5.0 units, which is 3,600 frames of ordinary travel arriving in one.
    That is the jump reported as "it ruins the immersion when it jumps".

    Integrating instead -- ``phase += dt * speed`` -- makes a speed change
    continuous by construction. The camera is exactly where it was; only
    how fast it leaves matters.

    Kept as a small object rather than two floats on the widget because
    the invariant is worth naming: `value` never decreases.
    """

    __slots__ = ("value", "_last_t")

    def __init__(self) -> None:
        self.value = 0.0
        self._last_t: Optional[float] = None

    def advance(self, t: float, speed: float) -> float:
        """Move the phase to wall-clock ``t`` at ``speed``, and return it.

        A t that goes BACKWARDS -- a restart, a clock reset -- re-bases
        rather than rewinding: the phase is the distance travelled, and
        travel does not un-happen.
        """
        last = self._last_t
        self._last_t = t
        if last is None or t < last:
            return self.value
        self.value += (t - last) * max(0.0, float(speed))
        return self.value


class RegionTour:
    """Floats the camera between the coordinates worth looking at.

    Around twenty regions are chosen on the image and the camera floats
    automatically towards them.

    SMOOTHLY IS THE WHOLE REQUIREMENT, so the interpolation is a
    smoothstep rather than a straight line: it leaves one region and
    arrives at the next with zero velocity, which is what stops the
    arrival reading as a stop. A linear blend is continuous in position
    and not in velocity, and the eye sees the corner.

    DRIFT IS OFF THE MOMENT THE USER TAKES THE CAMERA. Dragging is a
    statement about where they want to be, and a tour that resumes over
    it is the application arguing. :meth:`take_over` stops it for good;
    :meth:`restart` is what Ctrl+R calls.

    :param regions: ``(name, x, y, half_width, score)`` rows, usually
        :data:`spacr.qt.widgets.fractal_regions.REGIONS`.
    :param dwell: seconds spent at a region before leaving.
    :param travel: seconds spent moving between two regions.
    """

    __slots__ = ("regions", "dwell", "travel", "_taken")

    def __init__(self, regions, dwell: float = 18.0,
                 travel: float = 9.0) -> None:
        self.regions = tuple(regions or ())
        self.dwell = max(0.1, float(dwell))
        self.travel = max(0.1, float(travel))
        self._taken = False

    @property
    def active(self) -> bool:
        """Whether the tour is still steering."""
        return bool(self.regions) and not self._taken

    def take_over(self) -> None:
        """The user moved the camera. The tour does not argue."""
        self._taken = True

    def restart(self) -> None:
        """Ctrl+R: hand the camera back to the tour."""
        self._taken = False

    def period(self) -> float:
        """Seconds for one full circuit of every region."""
        return len(self.regions) * (self.dwell + self.travel)

    def target_at(self, seconds: float) -> Optional[tuple]:
        """Where the camera should be heading at ``seconds``.

        ``None`` when the tour is not steering, so a caller can leave the
        camera exactly where the user put it rather than being handed a
        coordinate it has to ignore.
        """
        if not self.active:
            return None
        leg = self.dwell + self.travel
        total = len(self.regions) * leg
        # Modulo, so the tour is a loop and a long session does not run
        # off the end of the list.
        position = float(seconds) % total
        index = int(position // leg)
        into = position - index * leg
        here = self.regions[index]
        if into <= self.dwell:
            return float(here[1]), float(here[2])
        there = self.regions[(index + 1) % len(self.regions)]
        fraction = (into - self.dwell) / self.travel
        eased = fraction * fraction * (3.0 - 2.0 * fraction)
        return (float(here[1]) + (float(there[1]) - float(here[1])) * eased,
                float(here[2]) + (float(there[2]) - float(here[2])) * eased)


def default_region_tour(**kwargs) -> RegionTour:
    """A tour over the committed regions, or an empty one without them."""
    try:
        from .fractal_regions import REGIONS
    except Exception:                                    # noqa: BLE001
        LOG.debug("no fractal regions to tour", exc_info=True)
        REGIONS = ()
    return RegionTour(REGIONS, **kwargs)


def state_at_seconds(t: float, speed: float, dream: float,
                     depth_phase: Optional[float] = None) -> CameraState:
    """The camera at ``t``. Pure, so a test can assert it moves.

    :param depth_phase: the integrated distance travelled. When given it
        is what positions the camera along the trajectory, and ``speed``
        no longer does -- which is what stops a scroll teleporting it.
        ``None`` reproduces the old ``t * speed``, for callers that have
        no phase to keep.
    """
    travelled = t * speed if depth_phase is None else float(depth_phase)
    depth = travelled / 12.0
    tx = dream * (0.090 * math.sin(2.0 * math.pi * t / 47.0)
                  + 0.035 * math.sin(2.0 * math.pi * t / 131.0 + 1.1)
                  + 0.025 * math.cos(2.0 * math.pi * t / 307.0 + 0.6))
    ty = dream * (0.080 * math.cos(2.0 * math.pi * t / 53.0 + 0.5)
                  + 0.040 * math.sin(2.0 * math.pi * t / 149.0 + 0.2)
                  + 0.025 * math.sin(2.0 * math.pi * t / 283.0 + 1.8))
    rotation = (0.26 * math.sin(2.0 * math.pi * t / 59.0)
                + 0.11 * math.sin(2.0 * math.pi * t / 211.0 + 0.7)
                ) * (0.55 + 0.75 * dream)
    shear_x = 0.18 * dream * math.sin(2.0 * math.pi * t / 73.0 + 0.2)
    shear_y = 0.16 * dream * math.cos(2.0 * math.pi * t / 89.0 + 0.8)
    stretch_x = math.exp(0.17 * dream * math.sin(2.0 * math.pi * t / 97.0 + 0.2))
    stretch_y = math.exp(0.15 * dream * math.cos(2.0 * math.pi * t / 107.0 + 1.4))
    palette_phase = (0.38 * math.sin(2.0 * math.pi * t / 173.0)
                     + 0.22 * math.cos(2.0 * math.pi * t / 337.0 + 0.3))
    return CameraState(t=t, depth=depth, tx=tx, ty=ty, rotation=rotation,
                       shear_x=shear_x, shear_y=shear_y,
                       stretch_x=stretch_x, stretch_y=stretch_y,
                       palette_phase=palette_phase)


class GpuBackendError(RuntimeError):
    """The GPU renderer could not be built. Always caught by `auto`."""


class _HeavyImportInProgress(RuntimeError):
    """The heavy-import lock was busy, so no GL context was built yet.

    Deliberately NOT a :class:`GpuBackendError`, and deliberately not
    caught by the ``auto`` fallback: this machine's GPU is fine and the
    shaders would compile. Treating it as a GPU failure would answer a
    two-second wait with the CPU renderer -- the one that saturates twenty
    cores -- instead of the backdrop that was asked for.

    The caller is expected to come back on a timer. It is decoration:
    arriving a fraction of a second late costs nothing, and blocking the
    GUI thread to be punctual is what this exception exists to stop.
    """


#: How long :class:`GpuFractalWidget` will wait for the heavy-import lock
#: before giving up and raising :class:`_HeavyImportInProgress`.
#:
#: SHORT ON PURPOSE. The preloader holds the lock for a whole module
#: import -- 2.3 s for each of the two that pull torch -- and this
#: constructor runs on the GUI thread, so an unbounded wait is a freeze
#: the compositor offers to force-quit. A tenth of a second is under any
#: compositor's threshold and under the eye's, while still being long
#: enough to ride out the brief holds that are not an import at all.
_HEAVY_LOCK_WAIT: Final[float] = 0.1


def _heavy_import_lock():
    """The lock the module preloader holds while importing, or None.

    Imported lazily and defensively: this widget is also usable on its own,
    with no application around it, and a backdrop must not fail to build
    because the lock could not be found.
    """
    try:
        from ..app import HEAVY_IMPORT_LOCK

        return HEAVY_IMPORT_LOCK
    except Exception:                                        # noqa: BLE001
        return None


def _make_gpu_widget(settings: Settings, controls: RuntimeControls,
                     hardware: HardwareProfile):
    try:
        from PySide6.QtWidgets import QVBoxLayout, QWidget
        from vispy import app as vispy_app, gloo

        # PySIDE6, not pyqt6. Two Qt bindings in one process segfault, and
        # vispy will happily import the other one if asked.
        vispy_app.use_app("pyside6")
        from vispy.app import Canvas
    except Exception as error:                               # noqa: BLE001
        raise GpuBackendError(str(error)) from error

    quality = resolved_quality(settings.quality, "gpu", hardware)
    if settings.pattern == "mandelbrot":
        from .fractal_mandelbrot import FRAGMENT_SHADER as _FRAGMENT

        # ITERATIONS, NOT FOLD DEPTH. The adaptive loop turns `_detail` down
        # when a frame runs long, and here that is the iteration budget --
        # which is also what the zoom needs MORE of as it descends, so the
        # floor is high enough that a deep frame does not go solid.
        base_detail = 6
        detail_floor = 5
    elif settings.pattern == "space":
        from .fractal_space import FRAGMENT_SHADER as _FRAGMENT

        # THE SCENE HAS NO ITERATION COUNT. Its cost is six parallax star
        # layers and three object slots, all fixed, so the adaptive detail
        # loop has nothing to turn down. The numbers are equal so a frame
        # that runs long cannot make the picture change.
        base_detail = 4
        detail_floor = 4
    elif settings.pattern == "cascade":
        from .fractal_cascade import FRAGMENT_SHADER as _FRAGMENT

        base_detail = 5 if quality == "balanced" else 6
        detail_floor = 4
    elif settings.pattern == "orbit_gpu":
        from .fractal_orbit_gpu import FRAGMENT_SHADER as _FRAGMENT

        # THE ITERATION COUNT IS FIXED IN THE SHADER, so the adaptive
        # detail loop has nothing to turn down here -- equal numbers mean
        # a frame that runs long cannot change the picture. Resolution is
        # what gives way instead, through the adaptive render scale.
        base_detail = 4
        detail_floor = 4
    else:
        _FRAGMENT = FRAGMENT_SHADER
        base_detail = 6 if quality == "balanced" else 8
        detail_floor = 5

    # WHAT THIS SHADER ACTUALLY DECLARES, read out of its source once. The
    # patterns share one uniform update and not one uniform list, and vispy
    # warns per frame per unknown name rather than ignoring it.
    _DECLARED = frozenset(
        match.group(1) for match in
        re.finditer(r"uniform\s+\w+\s+(u_\w+)\s*;", _FRAGMENT))

    #: The saved Mandelbrot numbers, read once when the backdrop is built.
    #:
    #: THE RENDERER READ THE MODULE'S DEFAULTS. Every one of the twelve
    #: settings the panel offers was collected, stored and then ignored --
    #: "changing render scale changes nothing", and the same was true of all
    #: of them. The published defaults are the FALLBACK now, not the answer.
    def _mandel_setting(name, fallback=None):
        from .fractal_mandelbrot import DEFAULTS as _PUBLISHED

        try:
            from ..preferences import get_fractal_settings

            saved = get_fractal_settings()
        except Exception:                                    # noqa: BLE001
            saved = {}
        if name in saved and saved[name] is not None:
            return saved[name]
        if fallback is not None:
            return fallback
        return _PUBLISHED[name]

    class _Canvas(Canvas):
        def __init__(self) -> None:
            """Build the GL canvas, hidden until it is placed."""
            super().__init__(keys=None, size=(1200, 760), show=False)
            # THE SAME POINTER THE CPU PATH USES. It samples QCursor rather
            # than receiving events, so it needs nothing from the widget
            # except a rectangle to be relative to -- which is why one class
            # serves both backends.
            self._pointer = Pointer()
            # Integrated travel, so a speed change does not teleport the
            # camera. One per canvas: it is this canvas's own position on
            # the trajectory.
            self._depth_phase = DepthPhase()
            # THE REFERENCE ORBIT, for the Mandelbrot pattern only. Built on
            # a worker thread because iterating a few thousand points at 320
            # decimal digits takes seconds, and the backdrop has to keep
            # drawing while it happens -- until it arrives the shader has an
            # all-zero orbit, which renders as the flat interior colour
            # rather than as a stall.
            self._orbit = None
            self._orbit_thread = None
            if settings.pattern == "mandelbrot":
                self._start_the_reference_orbit()
            self._started = time.perf_counter()
            self._last_sample = 0.0
            self._render_ema: Optional[float] = None
            self._detail = base_detail
            self._paused = False
            #: Set once Qt has freed the C++ side. The timer checks it so a
            #: single late tick does not become an endless retry.
            self._dead = False
            self._program = gloo.Program(VERTEX_SHADER, _FRAGMENT)
            self._program["a_position"] = np.asarray(
                [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)],
                dtype=np.float32)
            if settings.pattern == "mandelbrot":
                # A PLACEHOLDER UNTIL THE REAL ORBIT ARRIVES, and AFTER the
                # program exists: this used to run before `self._program`
                # was assigned, so it raised AttributeError on every build.
                #
                # vispy warns once per DRAW for a uniform a linked program
                # has never been given, and the real orbit takes seconds to
                # iterate on its thread. One black texel costs nothing and
                # the shader reads it as an orbit at the origin, which draws
                # the interior colour -- what an unset sampler drew anyway,
                # without sixty warnings a second.
                try:
                    self._program["u_orbit"] = gloo.Texture2D(
                        np.zeros((1, 1, 4), dtype=np.float32),
                        interpolation="nearest",
                        wrapping="clamp_to_edge")
                except Exception:                            # noqa: BLE001
                    LOG.debug("could not seed the orbit texture",
                              exc_info=True)
            gloo.set_state(depth_test=False, blend=False)
            self._update_uniforms(0.0)
            self._timer = vispy_app.Timer(interval=1.0 / settings.fps,
                                          connect=self._on_timer, start=True)
            # STOPPED WHEN QT FREES THE WIDGET, not only when someone calls
            # shutdown. A backdrop is reparented and deleted with its screen,
            # which never runs closeEvent.
            self.native.destroyed.connect(self._on_native_destroyed)

        def _update_uniforms(self, elapsed: float) -> None:
            """Push this frame's camera and time into the shader.

            The size is floored at one pixel: a canvas mid-resize can report zero, and
            a zero dimension reaches the shader as a division by nothing.
            """
            width, height = self.physical_size
            width = max(1, int(width))
            height = max(1, int(height))
            speed = controls.speed_at(elapsed)
            # THE PHASE, not `elapsed * speed`. Scrolling changes how fast
            # the trajectory is travelled; it must not change WHERE on the
            # trajectory the camera is. See DepthPhase.
            phase = self._depth_phase.advance(elapsed, speed)
            state = state_at_seconds(elapsed, speed, controls.dream,
                                     depth_phase=phase)
            pointer_x, pointer_y, pull, push = self._pointer_state()
            # ONLY WHAT THIS SHADER DECLARES. The three patterns share this
            # update but not their uniforms -- space has no dream term, since
            # a star field has nothing to warp -- and vispy warns once per
            # frame for every value handed to a name it cannot find. GPU
            # space printed "Value provided for 'u_dream'" sixty times a
            # second, into the terminal AND the console.
            for name, value in (
                    ("u_resolution", (width, height)),
                    ("u_time", np.float32(elapsed)),
                    ("u_speed", np.float32(speed)),
                    ("u_dream", np.float32(controls.dream)),
                    ("u_palette_phase", np.float32(state.palette_phase)),
                    ("u_tx", np.float32(state.tx)),
                    ("u_ty", np.float32(state.ty)),
                    ("u_rotation", np.float32(state.rotation)),
                    ("u_shear_x", np.float32(state.shear_x)),
                    ("u_shear_y", np.float32(state.shear_y)),
                    ("u_stretch_x", np.float32(state.stretch_x)),
                    ("u_stretch_y", np.float32(state.stretch_y)),
                    ("u_detail", np.int32(self._detail)),
                    # THE POINTER REACHES THE GPU TOO. These were fed on the
                    # CPU path only, so the backdrop followed the mouse in
                    # one backend and ignored it in the other.
                    ("u_pointer_x", np.float32(pointer_x)),
                    ("u_pointer_y", np.float32(pointer_y)),
                    ("u_pull", np.float32(pull)),
                    ("u_push", np.float32(push)),
            ) + tuple(self._mandelbrot_uniforms(elapsed).items()):
                if name in _DECLARED:
                    self._program[name] = value
            self._upload_the_orbit_if_it_arrived()

        def _start_the_reference_orbit(self) -> None:
            """Iterate Z off the GUI thread and upload it when it is ready."""
            import threading

            from .fractal_mandelbrot import DEFAULTS, ReferenceOrbit

            def _work():
                try:
                    orbit = ReferenceOrbit(
                        max_iter=int(_mandel_setting("max_iterations")),
                        digits=int(_mandel_setting("precision_digits")))
                except Exception:                            # noqa: BLE001
                    LOG.exception("could not build the reference orbit")
                    return
                # HANDED OVER BY ASSIGNMENT, which is atomic, rather than
                # touched into the GL program from this thread: a GL call
                # off the thread that owns the context is undefined.
                self._orbit = orbit

            self._orbit_thread = threading.Thread(
                target=_work, name="spacr-mandelbrot-orbit", daemon=True)
            self._orbit_thread.start()

        def _mandelbrot_uniforms(self, elapsed: float) -> dict:
            """The zoom's own uniforms for this instant.

            :returns: ``{}`` for every other pattern, so the shared update
                can splice it in unconditionally.
            """
            if settings.pattern != "mandelbrot":
                return {}
            from .fractal_mandelbrot import (DEFAULTS, depth_after_restart,
                                             depth_decades,
                                             iteration_budget, scale_at)

            # THE DEPTH IS INTEGRATED, not recomputed from the elapsed
            # time: Up and Down change the rate, and a depth derived from
            # `elapsed * rate` would jump backwards the moment the rate was
            # lowered, because the whole flight so far would be re-scaled.
            now = time.perf_counter()
            previous = getattr(self, "_zoom_clock", None)
            self._zoom_clock = now
            if previous is not None:
                # SIGNED, and floored at the surface: Down past zero backs
                # out of the zoom, and there is nothing above the starting
                # scale to back out into.
                step = (now - previous) * controls.speed * controls.zoom_rate \
                    / max(0.1, float(_mandel_setting("seconds_per_decade")))
                self._depth = max(0.0,
                                  getattr(self, "_depth", 0.0) + step)
            # THE DIVE STARTS AGAIN RATHER THAN ENDING IN A BLACK FRAME.
            # The per-pixel offset is a float32 whatever the reference
            # orbit's precision, and past about forty-five decades the step
            # between neighbouring pixels underflows to zero -- one sample
            # of one point, filling the screen.
            # ASKED TO START AGAIN. The settings changed, so the dive goes
            # back to the surface rather than applying new numbers thirty
            # decades down where they have nothing recognisable to act on.
            token = getattr(controls, "restart_token", 0)
            if token != getattr(self, "_restart_token", None):
                self._restart_token = token
                self._depth = 0.0
                self._zoom_clock = None
                # BACK TO THE ANCHOR AS WELL. A restart that kept the course
                # would begin at the surface but already pointed thirty
                # decades of steering away from the centre.
                camera = getattr(self, "_camera", None)
                if camera is not None:
                    camera.restart()
                self._refine_due = None
                self._refined = None

                self._plan = None
                self._steer_step = 0
                self._next_steer = 0.0
            depth = depth_after_restart(
                getattr(self, "_depth", 0.0),
                float(_mandel_setting("max_depth", 34.0)))
            self._depth = depth
            orbit = self._orbit
            length = float(orbit.max_iter + 1) if orbit is not None else 1.0
            budget = iteration_budget(
                depth, int(_mandel_setting("base_iterations")),
                float(_mandel_setting("iterations_per_decade")),
                int(_mandel_setting("max_iterations")))
            if orbit is not None:
                budget = min(budget, orbit.max_iter)
            # A FAULT IN THE STEERING MUST NOT STOP THE PICTURE. This
            # returns the uniforms for the whole frame, and a NameError in
            # the course-plotting once left every one of them unset -- so
            # the pattern drew nothing at all, silently, which is a far
            # worse failure than a dive that goes straight down.
            try:
                centre = self._steer(depth, budget, orbit)
            except Exception:                                # noqa: BLE001
                LOG.exception("could not steer the dive")
                camera = getattr(self, "_camera", None)
                centre = camera.centre if camera is not None else (0.0, 0.0)
            return {
                "u_scale": np.float32(
                    scale_at(depth, float(_mandel_setting("initial_scale")))),
                "u_center_offset": (np.float32(centre[0]),
                                    np.float32(centre[1])),
                "u_depth": np.float32(depth),
                "u_orbit_length": np.float32(length),
                "u_max_iter": np.int32(max(1, int(budget))),
            }

        def _steer(self, depth: float, budget: int, orbit):
            """Where the dive is heading, in the reference orbit's frame.

            :returns: ``(offset_re, offset_im)`` for ``u_center_offset``.

            THE DECIDING IS IN `SteeringCamera`, which has no Qt in it and
            can be driven frame by frame in a test. This method is the part
            that cannot be: reading the settings, and running the search on
            a worker thread so a 96x54 escape map does not stall the frame.

            Every claim about how smooth the motion is used to come from a
            simulation written beside the code rather than from the code,
            because this logic lived inside a canvas that needs a GL
            context to exist. That is why three fixes in a row were wrong.
            """
            import threading

            from .fractal_mandelbrot import (SteeringCamera, plan_guided_step,
                                             scale_at)

            if orbit is None:
                return (0.0, 0.0)

            camera = getattr(self, "_camera", None)
            if camera is None:
                camera = SteeringCamera()
                self._camera = camera
            camera.configure(
                strength=float(_mandel_setting("steering_strength")),
                interval=float(_mandel_setting("steering_interval_decades")),
                duration=float(_mandel_setting("steering_duration")),
                seconds_per_decade=float(
                    _mandel_setting("seconds_per_decade")))

            # FIXED MEANS FIXED -- but not "aimed at the least interesting
            # place in the frame". The anchor is chosen ONCE, before
            # anything moves, and then never again: the dive is exactly as
            # steady as a fixed path because it IS one, and the survey
            # costs a fraction of a second on a worker thread while the
            # backdrop is already drawing.
            #
            # Continuous steering is what shook; choosing where to point
            # before the descent starts moves nothing.
            # DRAGGING WORKS ON EITHER PATH. It is the user moving the
            # camera, and refusing that on the steady path would mean the
            # only way to look somewhere else was to turn on the search
            # that shook.
            span = scale_at(depth, float(_mandel_setting("initial_scale")))
            pointer = getattr(self, "_pointer", None)
            if pointer is not None and (pointer.drag_x or pointer.drag_y):
                here = camera.drag(pointer.drag_x, pointer.drag_y, span,
                                   depth)
                pointer.drag_x = 0.0
                pointer.drag_y = 0.0
                # THE REFERENCE FOLLOWS THE CAMERA. Perturbation measures
                # every pixel as a small offset from ONE orbit, so a camera
                # that walks away from it takes the picture with it:
                # measured, a reference 0.3 away escapes at iteration six
                # and the detail in the dragged view falls to nothing.
                self._refine_due = 0.0
                return here

            # AND KEEPS FOLLOWING IT DOWN. Each refinement is picked out of
            # the current view, and the view shrinks -- so a reference
            # accurate to a pixel now is accurate to a hundredth of that two
            # decades on. Measured: refining every decade holds the picture
            # sharp to eleven, against one or two without.
            self._refine_the_reference(camera, orbit, budget, depth, span)

            if str(_mandel_setting("path", "fixed")) != "guided":
                # NO AUTOMATIC AIMING. Choosing a "more interesting" point
                # by surveying the surface was tried and made it worse: a
                # point on a busy edge at the starting scale was measured
                # going completely flat three decades in -- entirely interior at
                # one candidate, entirely exterior at another -- because
                # surface structure does not predict what survives a
                # descent. Only a genuinely special point does, and the
                # reference centre already is one.
                #
                # The camera is steered BY HAND instead: drag to move it,
                # and the arrow keys change the speed and the direction.
                return camera.centre
            if not camera.steering:
                return camera.centre

            plan = getattr(self, "_plan", None)
            if plan is not None and plan.get("done"):
                self._plan = None
                camera.aim_at(plan.get("target"), depth, span)
            elif plan is None and camera.wants_a_target(depth):
                slot = {"done": False, "target": None}
                self._plan = slot
                step = camera.step
                strength = camera.strength
                here = camera.centre

                def _look():
                    try:
                        found = plan_guided_step(
                            orbit, span, budget, strength=strength,
                            candidates=int(
                                _mandel_setting("candidate_count")),
                            step_index=step,
                            offset_re=here[0], offset_im=here[1])
                    except Exception:                        # noqa: BLE001
                        LOG.debug("could not plan a steering step",
                                  exc_info=True)
                        found = None
                    slot["target"] = None if found is None else found[:2]
                    slot["done"] = True

                threading.Thread(target=_look, daemon=True,
                                 name="spacr-mandelbrot-steer").start()

            return camera.advance(time.perf_counter())

        def _refine_the_reference(self, camera, orbit, budget, depth, span):
            """Move the reference back onto the boundary, now and then.

            Surveyed and rebuilt on a worker thread: the survey is a 96x54
            escape map and the rebuild iterates the orbit at full precision,
            neither of which belongs in a frame.
            """
            import threading

            from .fractal_mandelbrot import (REFINE_EVERY,
                                             best_reference_in_view,
                                             rebased_orbit)

            landed = getattr(self, "_refined", None)
            if landed is not None:
                self._refined = None
                centre, fresh = landed
                if fresh is not None:
                    self._orbit = fresh
                    # The camera is now sitting ON the new reference, so
                    # its offset starts again from nothing.
                    camera.centre = (0.0, 0.0)
                    camera.target = None
                self._refine_due = depth + REFINE_EVERY
                return

            if getattr(self, "_refine_thread_running", False):
                return
            due = getattr(self, "_refine_due", None)
            if due is None:
                self._refine_due = depth + REFINE_EVERY
                return
            if depth < due:
                return

            here = camera.centre
            digits = int(_mandel_setting("precision_digits"))
            ceiling = int(_mandel_setting("max_iterations"))
            self._refine_thread_running = True

            def _work():
                try:
                    offset = best_reference_in_view(
                        orbit, here[0], here[1], span, int(budget))
                    self._refined = rebased_orbit(
                        orbit, offset[0], offset[1], digits, ceiling)
                except Exception:                            # noqa: BLE001
                    LOG.debug("could not refine the reference",
                              exc_info=True)
                    self._refined = (None, None)
                finally:
                    self._refine_thread_running = False

            threading.Thread(target=_work, daemon=True,
                             name="spacr-mandelbrot-refine").start()

        def _upload_the_orbit_if_it_arrived(self) -> None:
            """Put the finished orbit into the shader's texture.

            ON THE GUI THREAD, from inside the draw, because that is where
            the GL context lives. Uploaded once: `_orbit_uploaded` is the
            orbit object itself, so a rebuilt one would be noticed.
            """
            orbit = self._orbit
            if orbit is None or getattr(self, "_orbit_uploaded", None) is orbit:
                return
            try:
                # CLAMPED AND NEAREST, said outright. The orbit is one row
                # of 2,201 texels -- not a power of two -- and a driver that
                # defaults to REPEAT wrapping can refuse a non-power-of-two
                # texture outright. Nearest because every texel is one
                # iteration of the reference orbit: interpolating between
                # two of them is a number that is not on the orbit at all.
                self._program["u_orbit"] = gloo.Texture2D(
                    orbit.packed, interpolation="nearest",
                    wrapping="clamp_to_edge", internalformat="rgba32f")
                self._orbit_uploaded = orbit
            except Exception:                                # noqa: BLE001
                LOG.exception("could not upload the reference orbit")
                # Marked done regardless, or a driver that refuses the
                # format would be asked again sixty times a second.
                self._orbit_uploaded = orbit

        def _pointer_state(self):
            """``(x, y, pull, push)`` for the shader, in -1..1 space.

            :returns: zeros when the pointer is not being followed, so a
                shader can multiply by them unconditionally.

            THE MANDELBROT IS DRAGGED, NOT ATTRACTED. Mouse interaction moves
            it only through drag input; the pointer's stationary position does
            not pull the view about.
            
            The other three patterns are fields that can be warped toward a
            point and look right doing it. A deep zoom is a camera: pulling
            its coordinates toward wherever the mouse happens to rest slides
            the picture continuously, which reads as the image drifting away
            from you rather than as anything you did.

            The pointer is still SAMPLED, because that is what accumulates
            the drag -- `_steer` consumes it. Only `pull` and `push`, which
            are the position-driven terms, are withheld.
            """
            if not controls.follow_pointer:
                return 0.0, 0.0, 0.0, 0.0
            try:
                pointer = self._pointer.sample(
                    self.native, controls.pointer_size,
                    controls.pointer_strength)
            except Exception:                                # noqa: BLE001
                # A backdrop that cannot find the mouse still draws.
                return 0.0, 0.0, 0.0, 0.0
            if settings.pattern == "mandelbrot":
                return pointer.x, pointer.y, 0.0, 0.0
            return pointer.x, pointer.y, pointer.pull, pointer.push

        def on_resize(self, _event) -> None:
            width, height = self.physical_size
            gloo.set_viewport(0, 0, max(1, int(width)), max(1, int(height)))

        def on_draw(self, _event) -> None:
            if self._dead:
                return
            benchmark = time.perf_counter() - self._last_sample >= 2.0
            started = time.perf_counter()
            try:
                self._program.draw("triangle_strip")
            except Exception:                                # noqa: BLE001
                # ONE COMPLAINT, NOT A STORM. vispy catches whatever a
                # DrawEvent handler raises, logs it as an ERROR, and RETRIES
                # -- doubling a repeat counter each time. A draw that cannot
                # succeed once cannot succeed at all, so the retries only
                # fill the terminal and, because these are logged at ERROR,
                # raise a "spaCR ERROR" panel per retry while a module runs.
                #
                # This machine's context is OpenGL ES; vispy compiles these
                # shaders as desktop GLSL 120, which ES rejects. Stopping is
                # the honest response: the backdrop cannot draw here.
                self._dead = True
                self.stop_timer()
                LOG.warning("the GPU backdrop cannot draw on this GL context "
                            "and has been stopped", exc_info=True)
                return
            if not benchmark:
                return
            try:
                gloo.gl.glFinish()
                duration = max(1e-5, time.perf_counter() - started)
                self._render_ema = (
                    duration if self._render_ema is None
                    else 0.75 * self._render_ema + 0.25 * duration)
                self._last_sample = time.perf_counter()
                target_period = 1.0 / settings.fps
                if self._render_ema > 1.05 * target_period:
                    self._detail = max(detail_floor, self._detail - 1)
                elif self._render_ema < 0.52 * target_period:
                    self._detail = min(base_detail + 1, self._detail + 1)
            except Exception:                                # noqa: BLE001
                self._last_sample = time.perf_counter()

        def _on_timer(self, _event) -> None:
            """Advance one frame, unless paused or already torn down."""
            if self._paused or self._dead:
                return
            # HOLD STILL UNDER A POPUP. A menu or a tooltip composited over
            # this native GL surface makes the widgets around it repaint,
            # which is the flicker of the dock and the header. The last
            # frame stays up; only the clock stops moving.
            if a_popup_is_on_screen():
                return
            try:
                self._update_uniforms(time.perf_counter() - self._started)
                self.update()
            except RuntimeError:
                # "Internal C++ object already deleted". vispy's Timer is not
                # a QTimer and is not destroyed with the widget, so it goes on
                # firing at a canvas Qt has freed -- and vispy's own handler
                # catches, logs and RETRIES, which is where the 2,4,8...4096
                # repeat storm comes from. Stop the timer at the first one.
                self._dead = True
                self.stop_timer()

        def stop_timer(self) -> None:
            """Stop the vispy timer. Safe to call twice, and after deletion."""
            try:
                self._timer.stop()
            except Exception:                                # noqa: BLE001
                pass

        def _on_native_destroyed(self, *_args) -> None:
            """Mark the canvas dead and stop its timer.

            The native window can go before Python does, and a timer that fires after
            it draws into an object that is not there.
            """
            self._dead = True
            self.stop_timer()

        def stats_text(self) -> str:
            width, height = self.physical_size
            if self._paused:
                timing = "paused for a run"
            elif self._render_ema is None:
                timing = "measuring"
            else:
                timing = f"{1000.0 * self._render_ema:.1f} ms GPU"
            return (f"v{VERSION} · GPU/{quality} · {settings.pattern} · "
                    f"spatial 2x2\n"
                    f"{int(width)}×{int(height)} · target {settings.fps} fps · "
                    f"detail {self._detail}\n{timing}")

    class GpuFractalWidget(QWidget):
        """The GLSL fractal. The GPU does the work; Qt only hosts it.

        :param parent: parent widget.
        """

        backend_name: Final[str] = "gpu"

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            # UNDER THE HEAVY-IMPORT LOCK. Creating a GL context while the
            # preloader is bringing torch (and therefore CUDA) up is exactly
            # the concurrent initialisation `_PipelinePreloader` used to stay
            # on the GUI thread to avoid. It is on a worker thread now, so
            # the two take turns instead.
            #
            # THE WAIT IS BOUNDED, and it has to be. This constructor runs on
            # the GUI thread, and the lock's other holder is an import that
            # takes seconds -- so `with lock:` here was a priority inversion:
            # a background task with no deadline holding up the one thread
            # that has one. Measured at 2,130 ms of blocked GUI thread for a
            # 2,000 ms hold, against ~130 ms of actual construction. That is
            # the whole of the difference the maintainer reported between
            # `spaceout` and `spacr` on opening a module: only spaceout builds
            # this widget, and only this widget takes the lock. The ordinary
            # ambient backdrop never does, which is why `spacr` opened the
            # same screen without the compositor offering to force-quit.
            #
            # `AppScreen._heavy_lock_is_free` cannot prevent it. That peek is
            # a check, not a reservation, and the preloader re-takes the lock
            # between two imports -- so landing in the gap is not rare, it is
            # the ordinary case for a click made while the preloader runs.
            lock = _heavy_import_lock()
            if lock is None:
                self._canvas = _Canvas()
            elif not lock.acquire(timeout=_HEAVY_LOCK_WAIT):
                raise _HeavyImportInProgress(
                    "the heavy-import lock is held; the backdrop will be "
                    "built when it frees")
            else:
                try:
                    self._canvas = _Canvas()
                finally:
                    lock.release()
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            layout.addWidget(self._canvas.native)

        def pause(self) -> bool:
            """Stop drawing while a run is on. The last frame stays up."""
            if self._canvas._paused:
                return False
            self._canvas._paused = True
            return True

        def resume(self) -> bool:
            if not self._canvas._paused:
                return False
            self._canvas._paused = False
            return True

        def is_paused(self) -> bool:
            return bool(self._canvas._paused)

        def set_animating(self, on: bool) -> bool:
            """`AmbientWidget`'s verb for the same thing.

            The ambient backdrop this replaces is stopped and started with
            `set_animating`, and its callers -- the Home screen's teardown
            among them -- reach for that name. Answering to it makes this a
            drop-in rather than something every call site has to learn.
            """
            return self.resume() if on else self.pause()


        def stats_text(self) -> str:
            return self._canvas.stats_text()

        def shutdown(self) -> None:
            """Stop for good. Safe to call twice and after Qt has freed it."""
            try:
                self._canvas._dead = True
                self._canvas.stop_timer()
                self._canvas.close()
            except Exception:                                # noqa: BLE001
                pass

        def closeEvent(self, event) -> None:
            self.shutdown()
            super().closeEvent(event)

    return GpuFractalWidget()


#: Every RuntimeControls a live backdrop is reading, so a key press can
#: reach the one on screen without the backdrop having to accept events.
#:
#: THE BACKDROP MUST NOT TAKE EVENTS. It sits behind every control, and a
#: widget that accepted the mouse would eat the click meant for the button
#: on top of it -- which is why the pointer is SAMPLED rather than received.
#: The same reasoning applies to the keyboard, so Up and Down are handled by
#: the window and applied here.
_LIVE_CONTROLS: list = []

#: What one press of Up or Down, or one notch of the wheel, multiplies the
#: zoom rate by. From the source this pattern came from.
ZOOM_STEP: Final[float] = 1.12

#: The range a nudge will move within.
MIN_ZOOM_RATE: Final[float] = 0.05
MAX_ZOOM_RATE: Final[float] = 20.0


def apply_saved_controls() -> int:
    """Push the saved settings into every running backdrop.

    :returns: how many were updated.

    THE BACKDROP KEEPS THE CONTROLS IT WAS BUILT WITH. Saving Preferences
    writes new values to the store, but an existing `RuntimeControls` object
    otherwise continues holding its old values. This function synchronises
    every live object so changes take effect immediately.

    Everything that can change while a backdrop is on screen is pushed here.
    What cannot -- the pattern, the backend, the shader -- still needs the
    backdrop rebuilding, which is what changing a screen does anyway.
    """
    try:
        from ..preferences import get_fractal_settings

        values = get_fractal_settings()
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not read the fractal settings", exc_info=True)
        return 0

    updated = 0
    for controls in list(_LIVE_CONTROLS):
        try:
            controls.speed = float(values["speed"])
            controls.dream = float(values["dream"])
            controls.variable_speed = bool(values["variable_speed"])
            controls.speed_min = float(values["speed_min"])
            controls.speed_max = float(values["speed_max"])
            controls.speed_period = float(values["speed_period"])
            controls.follow_pointer = bool(values["pointer_gravity"])
            controls.pointer_size = float(values["pointer_size"])
            controls.pointer_strength = float(values["pointer_strength"])
            controls.zoom_rate = float(values["zoom_rate"])
            updated += 1
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not update a live backdrop", exc_info=True)
    return updated


def restart_the_dive() -> None:
    """Send every running backdrop back to the surface.

    Called when the fractal settings change. A dive that resumed at the
    depth it had reached would apply the new numbers to a viewport thirty
    decades down, where a changed starting scale or iteration count has
    nothing recognisable to act on -- so the change looks as though it did
    nothing.
    """
    for controls in list(_LIVE_CONTROLS):
        try:
            controls.restart_token += 1
        except Exception:                                    # noqa: BLE001
            continue


def nudge_zoom_rate(steps: int) -> float:
    """Speed the descent up or slow it down.

    :param steps: how many notches; positive is faster.
    :returns: the resulting rate, or 0.0 when no backdrop is running.

    Clamped, unlike the settings fields: this is a key held down rather than
    a number somebody typed, so there is nothing to tell them about and a
    rate of 10^12 from leaning on an arrow key is not a request.
    """
    if not _LIVE_CONTROLS:
        return 0.0
    rate = 0.0
    for controls in list(_LIVE_CONTROLS):
        try:
            # SIGNED, AND THE KEY CHANGES THE VALUE, NOT THE MAGNITUDE.
            # Asked for 2026-08-28: "so i could go slow and fast forward and
            # back". Down means "less", all the way through zero into
            # backing out of the zoom; Up means "more".
            #
            # Stepping the magnitude multiplicatively and flipping the sign
            # at the floor OSCILLATES: every press at the floor swaps the
            # direction, so holding Down never gets anywhere. Which side of
            # zero the rate is on has to decide whether a step grows or
            # shrinks it.
            rate = float(controls.zoom_rate)
            for _ in range(abs(int(steps))):
                going_up = int(steps) > 0
                if (rate > 0) == going_up:
                    # Away from zero on the side it is already on.
                    rate = rate * ZOOM_STEP if rate > 0 else rate * ZOOM_STEP
                else:
                    # Toward zero, and through it when there is nowhere
                    # left to shrink to.
                    shrunk = rate / ZOOM_STEP
                    if abs(shrunk) < MIN_ZOOM_RATE:
                        rate = MIN_ZOOM_RATE if going_up else -MIN_ZOOM_RATE
                    else:
                        rate = shrunk
                magnitude = clamp(abs(rate), MIN_ZOOM_RATE, MAX_ZOOM_RATE)
                rate = magnitude if rate >= 0 else -magnitude
            controls.zoom_rate = rate
        except Exception:                                    # noqa: BLE001
            continue
    return rate


def _render_scale() -> float:
    """The saved render scale, or 1.0 -- native resolution.

    MODULE LEVEL, because both builders need it: it was defined inside the
    GPU one and read from the CPU one, which is a NameError at the moment a
    frame is sized.
    """
    try:
        from ..preferences import get_fractal_settings

        return float(get_fractal_settings().get("render_scale", 1.0))
    except Exception:                                        # noqa: BLE001
        return 1.0


def pattern_for_this_machine(pattern: str, backend: str = "auto") -> str:
    """The pattern that can actually be drawn here.

    :param pattern: what the user asked for.
    :param backend: the resolved backend, or ``"auto"``.
    :returns: ``pattern``, or the fallback when it cannot be drawn.

    TWO PATTERNS ARE GPU-ONLY. Mandelbrot needs a texture of the reference
    orbit and a shader to perturb around it, and `orbit_gpu` is a fragment
    shader with no numba twin -- the CPU orbit fold is a DIFFERENT picture,
    four samples across four frames rather than four of one instant, which
    is why the two are separate entries at all.

    So a machine with no usable GL context gets the orbit fold instead --
    it has a CPU renderer and is the cheapest of the ones that do -- rather
    than a backdrop that draws nothing.

    Silent, and deliberately: the backdrop is decoration, and a dialog
    explaining that a machine cannot run one of five ornaments is worth
    less than the interruption costs.
    """
    if str(pattern) not in GPU_ONLY_PATTERNS:
        return str(pattern)
    if str(backend) == "cpu":
        return FALLBACK_PATTERN
    if not platform_can_do_opengl() or not gpu_is_available():
        return FALLBACK_PATTERN
    return str(pattern)


def create_fractal_widget(settings: Optional[Settings] = None,
                          controls: Optional[RuntimeControls] = None,
                          hardware: Optional[HardwareProfile] = None):
    """Build the fractal backdrop, GPU when there is one.

    :returns: a QWidget carrying `backend_name`, `stats_text()`, `pause()`,
        `resume()` and `shutdown()`. Never raises for a missing GPU: an
        explicit ``backend='gpu'`` that cannot be built still falls back,
        because a backdrop is not worth refusing to start the application
        over.
    """
    settings = (settings or Settings()).validated()
    controls = controls or RuntimeControls()
    # THE PATTERN THIS MACHINE CAN ACTUALLY DRAW. Mandelbrot is GPU-only,
    # so a machine with no usable context gets the orbit fold rather than a
    # backdrop that draws nothing.
    settings = replace(
        settings,
        pattern=pattern_for_this_machine(settings.pattern, settings.backend))
    # Registered so a key press can reach it; the backdrop itself must not
    # accept events, or it would eat the clicks meant for the interface in
    # front of it.
    # BY IDENTITY, NOT BY VALUE. `RuntimeControls` is a dataclass, so `in`
    # compares field by field: a new backdrop whose settings happen to match
    # a previous one was never added, and the keys then drove a stale object
    # that no canvas reads. That is why Ctrl+R and Up and Down stopped
    # working after the first backdrop was replaced.
    #
    # The list is also trimmed here, because nothing else can: a backdrop
    # that has been destroyed leaves its controls behind, and updating a few
    # dead ones is harmless while letting the list grow without bound is not.
    if not any(existing is controls for existing in _LIVE_CONTROLS):
        _LIVE_CONTROLS.append(controls)
    del _LIVE_CONTROLS[:-8]
    hardware = hardware or HardwareProfile.detect()

    # `gpu_is_available` covers the explicit 'gpu' request as well: asking
    # for a renderer this platform would crash on is still a crash.
    if settings.backend in ("auto", "gpu") and gpu_is_available():
        try:
            return _make_gpu_widget(settings, controls, hardware)
        except _HeavyImportInProgress:
            # NOT A GPU FAILURE, so not the CPU renderer's cue. The context
            # was never attempted; the lock was busy. Falling through here
            # would trade a 0.3 s wait for the twenty-core fallback, and
            # would do it every time a module is opened during startup.
            raise
        except Exception:                                    # noqa: BLE001
            # SAID OUT LOUD. This swallowed every GPU failure without a
            # word, so a shader that would not compile looked exactly like a
            # machine with no GPU -- and the Mandelbrot pattern, which has
            # no CPU renderer, came out as the orbit fold with nothing
            # anywhere to say why.
            LOG.warning("the GPU backdrop could not be built; falling back "
                        "to the CPU renderer", exc_info=True)
    if settings.pattern == "mandelbrot":
        # AND THE CPU CANNOT DRAW THIS ONE. Handing it to the CPU builder
        # silently produced the orbit fold, because that is what its final
        # `else` does -- so the user chose Mandelbrot and got something
        # else with no indication anything had happened.
        LOG.warning("the Mandelbrot pattern needs the GPU renderer; "
                    "drawing %s instead", FALLBACK_PATTERN)
        settings = replace(settings, pattern=FALLBACK_PATTERN)
    return _make_cpu_widget(settings, controls, hardware)

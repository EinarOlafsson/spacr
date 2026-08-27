"""The spaceout fractal: a GPU shader when there is one, Numba otherwise.

Ported from the maintainer's `fractal_travel.py` v2.1.0. Two renderers,
not one engine with a switch:

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
import os
import time
from dataclasses import dataclass
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

#: The maintainer's own defaults, given as two command lines. `auto` picks
#: the GPU when vispy is importable and the CPU otherwise, which is what
#: makes one set of numbers serve both.
#: Which fractal. Two genuinely different families, not one with knobs:
#: `orbit` is the orbit-fold of `fractal_travel.py` v2.1.0, whose CPU path
#: antialiases by walking a sub-pixel grid ACROSS FOUR FRAMES; `cascade` is
#: the fold-inversion of v1.0.0, which takes all four samples INSIDE one
#: frame and is four times the work per pixel because of it.
PATTERNS: Final[tuple[str, ...]] = ("orbit", "cascade")
PATTERN_LABELS: Final[dict] = {
    "orbit": "Orbit fold (temporal 2x2)",
    "cascade": "Fold-inversion cascade (spatial 2x2)",
}
DEFAULT_PATTERN: Final[str] = "orbit"

DEFAULT_BACKEND: Final[str] = "auto"
DEFAULT_QUALITY: Final[str] = "auto"
DEFAULT_SCALE: Final[float] = 1.0
DEFAULT_SPEED: Final[float] = 4.0
DEFAULT_DREAM: Final[float] = 1.5
DEFAULT_VARIABLE_SPEED: Final[bool] = False

#: How far `variable_speed` swings the speed, and over how long. Slow enough
#: that it reads as drift rather than as a pulse.
_VARIABLE_SPEED_DEPTH: Final[float] = 0.45
_VARIABLE_SPEED_PERIOD: Final[float] = 41.0


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


@dataclass(frozen=True, slots=True)
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


@dataclass(slots=True)
class RuntimeControls:
    """What the user can move while it is running."""

    speed: float = DEFAULT_SPEED
    dream: float = DEFAULT_DREAM
    variable_speed: bool = DEFAULT_VARIABLE_SPEED

    def speed_at(self, t: float) -> float:
        """The speed to use at ``t`` seconds.

        Constant unless `variable_speed` is on, in which case it breathes
        around the set value rather than replacing it -- so the preference
        still means what it says.
        """
        if not self.variable_speed:
            return self.speed
        swing = 1.0 + _VARIABLE_SPEED_DEPTH * math.sin(
            2.0 * math.pi * t / _VARIABLE_SPEED_PERIOD)
        return max(0.05, self.speed * swing)


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    logical_cpus: int

    @staticmethod
    def detect() -> "HardwareProfile":
        return HardwareProfile(logical_cpus=max(1, os.cpu_count() or 1))


def resolved_quality(requested: str, backend: str,
                     hardware: HardwareProfile) -> str:
    if requested != "auto":
        return requested
    if backend == "gpu":
        return "balanced"
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
    platform = str(os.environ.get("QT_QPA_PLATFORM", "")).strip().lower()
    if platform.startswith(("offscreen", "minimal", "vnc")):
        return False
    if platform in ("", "xcb", "wayland", "cocoa", "windows"):
        # An empty value means Qt will choose, which it only does when there
        # is a display to choose for.
        return bool(os.environ.get("DISPLAY")
                    or os.environ.get("WAYLAND_DISPLAY")
                    or platform in ("cocoa", "windows"))
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
    def _orbit_sample(px, py, width, height, t, speed, dream, iterations):
        denominator = float(min(width, height))
        x = (2.0 * px - width) / denominator
        y = (height - 2.0 * py) / denominator

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
    def _render_into(output, t, speed, dream, iterations, jitter_x, jitter_y):
        height, width, _channels = output.shape
        for y in prange(height):
            for x in range(width):
                red, green, blue = _orbit_sample(
                    x + jitter_x, y + jitter_y, width, height,
                    t, speed, dream, iterations)
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
               dream: float, iterations: int) -> np.ndarray:
        set_num_threads(self.thread_count)
        self._ensure_size(width, height)
        jitter_x, jitter_y = JITTERS[self.slot]
        _render_into(self.ring[self.slot], t, speed, dream, iterations,
                     jitter_x, jitter_y)
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

def _make_cpu_widget(settings: Settings, controls: RuntimeControls,
                     hardware: HardwareProfile):
    if njit is None:
        raise RuntimeError("numba is required for the CPU fractal backend")

    from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtWidgets import QWidget

    quality = resolved_quality(settings.quality, "cpu", hardware)
    thread_count = resolved_cpu_threads(settings, hardware)
    cascade = settings.pattern == "cascade"

    # EACH PATTERN'S OWN BUDGET. The cascade evaluates four samples per pixel
    # inside one frame where the orbit evaluates one and averages across
    # frames, so it renders roughly a quarter of the pixels and holds a lower
    # cap to spend the same wall-clock. Sharing one set of numbers would make
    # one of them either wasteful or unusable.
    if cascade:
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
            super().__init__()
            self.engine = engine_factory(thread_count)

        @Slot(object)
        def render(self, request: object) -> None:
            try:
                started = time.perf_counter()
                frame = self.engine.render(
                    request["width"], request["height"], request["t"],
                    request["speed"], request["dream"], request["iterations"])
                self.frame_ready.emit(frame, time.perf_counter() - started)
            except Exception as error:                       # noqa: BLE001
                self.failed.emit(f"{type(error).__name__}: {error}")

    class CpuFractalWidget(QWidget):
        """The orbit-fold fractal, rendered off the GUI thread.

        `paintEvent` only blits: every fractal evaluation happens on the
        worker thread, so a slow frame makes the picture late and never makes
        the interface late.
        """

        backend_name: Final[str] = "cpu"
        render_requested = Signal(object)

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
            self.setAutoFillBackground(False)

            self._thread = QThread(self)
            self._worker = _Worker()
            self._worker.moveToThread(self._thread)
            self.render_requested.connect(
                self._worker.render, Qt.ConnectionType.QueuedConnection)
            self._worker.frame_ready.connect(self._accept_frame)
            self._worker.failed.connect(self._on_failure)
            self._thread.finished.connect(self._worker.deleteLater)
            self._thread.start()

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
            logical_width = max(320, self.width())
            logical_height = max(180, self.height())
            device_scale = max(1.0, float(self.devicePixelRatioF()))
            physical_width = max(320, round(logical_width * device_scale))
            physical_height = max(180, round(logical_height * device_scale))
            aspect = physical_width / physical_height
            requested = base_pixels * self._adaptive_scale ** 2
            requested = max(180_000.0, min(1_250_000.0, requested))
            width = int(round(math.sqrt(requested * aspect)))
            height = int(round(width / aspect))
            width = min(physical_width, max(320, width))
            height = min(physical_height, max(180, height))
            width -= width % 2
            height -= height % 2
            return max(320, width), max(180, height)

        @Slot()
        def _request_frame(self) -> None:
            if self._stopped or self._paused:
                return
            if self._busy or not self.isVisible():
                self._timer.start(50)
                return
            width, height = self._target_size()
            self._render_size = (width, height)
            self._busy = True
            self.render_requested.emit({
                "width": width, "height": height, "t": self._sim_time,
                "speed": controls.speed_at(self._sim_time),
                "dream": controls.dream, "iterations": iterations,
            })
            self._sim_time += target_period

        def _adapt_resolution(self) -> None:
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
            self._error = message
            self._busy = False
            self.update()
            if not (self._paused or self._stopped):
                self._timer.start(750)

        def paintEvent(self, _event) -> None:
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
            self._thread.quit()
            self._thread.wait(5000)

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

vec3 render_sample(vec2 fragment_position) {
    float denominator = min(u_resolution.x, u_resolution.y);
    vec2 uv = (2.0 * fragment_position - u_resolution) / denominator;
    uv *= 1.10;
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


@dataclass(frozen=True, slots=True)
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


def state_at_seconds(t: float, speed: float, dream: float) -> CameraState:
    """The camera at ``t``. Pure, so a test can assert it moves."""
    depth = t * speed / 12.0
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
    if settings.pattern == "cascade":
        from .fractal_cascade import FRAGMENT_SHADER as _FRAGMENT

        base_detail = 5 if quality == "balanced" else 6
        detail_floor = 4
    else:
        _FRAGMENT = FRAGMENT_SHADER
        base_detail = 6 if quality == "balanced" else 8
        detail_floor = 5

    class _Canvas(Canvas):
        def __init__(self) -> None:
            super().__init__(keys=None, size=(1200, 760), show=False)
            self._started = time.perf_counter()
            self._last_sample = 0.0
            self._render_ema: Optional[float] = None
            self._detail = base_detail
            self._paused = False
            self._program = gloo.Program(VERTEX_SHADER, _FRAGMENT)
            self._program["a_position"] = np.asarray(
                [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)],
                dtype=np.float32)
            gloo.set_state(depth_test=False, blend=False)
            self._update_uniforms(0.0)
            self._timer = vispy_app.Timer(interval=1.0 / settings.fps,
                                          connect=self._on_timer, start=True)

        def _update_uniforms(self, elapsed: float) -> None:
            width, height = self.physical_size
            width = max(1, int(width))
            height = max(1, int(height))
            speed = controls.speed_at(elapsed)
            state = state_at_seconds(elapsed, speed, controls.dream)
            self._program["u_resolution"] = (width, height)
            self._program["u_time"] = np.float32(elapsed)
            self._program["u_speed"] = np.float32(speed)
            self._program["u_dream"] = np.float32(controls.dream)
            self._program["u_palette_phase"] = np.float32(state.palette_phase)
            self._program["u_tx"] = np.float32(state.tx)
            self._program["u_ty"] = np.float32(state.ty)
            self._program["u_rotation"] = np.float32(state.rotation)
            self._program["u_shear_x"] = np.float32(state.shear_x)
            self._program["u_shear_y"] = np.float32(state.shear_y)
            self._program["u_stretch_x"] = np.float32(state.stretch_x)
            self._program["u_stretch_y"] = np.float32(state.stretch_y)
            self._program["u_detail"] = np.int32(self._detail)

        def on_resize(self, _event) -> None:
            width, height = self.physical_size
            gloo.set_viewport(0, 0, max(1, int(width)), max(1, int(height)))

        def on_draw(self, _event) -> None:
            benchmark = time.perf_counter() - self._last_sample >= 2.0
            started = time.perf_counter()
            self._program.draw("triangle_strip")
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
            if self._paused:
                return
            self._update_uniforms(time.perf_counter() - self._started)
            self.update()

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
        """The GLSL fractal. The GPU does the work; Qt only hosts it."""

        backend_name: Final[str] = "gpu"

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self._canvas = _Canvas()
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
            try:
                self._canvas._timer.stop()
                self._canvas.close()
            except Exception:                                # noqa: BLE001
                pass

        def closeEvent(self, event) -> None:
            self.shutdown()
            super().closeEvent(event)

    return GpuFractalWidget()


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
    hardware = hardware or HardwareProfile.detect()

    # `gpu_is_available` covers the explicit 'gpu' request as well: asking
    # for a renderer this platform would crash on is still a crash.
    if settings.backend in ("auto", "gpu") and gpu_is_available():
        try:
            return _make_gpu_widget(settings, controls, hardware)
        except Exception:                                    # noqa: BLE001
            pass
    return _make_cpu_widget(settings, controls, hardware)

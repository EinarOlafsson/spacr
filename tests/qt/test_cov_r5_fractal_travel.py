"""The backdrop's refusals, its kernels, and the canvas it cannot always have.

`fractal_travel` is two renderers wrapped in a great deal of declining to
fail: a pointer that answers even when the widget it was handed has been
freed, kernels that saturate rather than tear when the pointer is dragged
across them, a worker that must not shout at a widget Qt has already
deleted, and a GPU canvas that has to give up during CONSTRUCTION so the
factory can fall back.

Three techniques are used here rather than hoping the machine cooperates.

* The Numba kernels are driven through ``.py_func``. A jitted body is not
  Python at run time, so the arithmetic that decides what the picture looks
  like is otherwise never executed as source at all -- and the saturation
  rules inside `_orbit_sample` are the difference between a pointer that
  bends the plane and one that turns it inside out.

* The GPU canvas is built against a STAND-IN ``vispy``. `QT_QPA_PLATFORM=
  offscreen` prints "QOpenGLWidget is not supported on this platform" and
  then behaves in whatever way the local driver behaves, which is not
  something a test can assert against; the same reasoning as
  `test_cov_gpu_backdrop_does_not_storm`, one step further. Everything below
  ``gloo.Program`` is somebody else's code and everything above it is what
  this module does with it, so that is where the seam goes.

* The CPU widget is given a cheap engine before it is built, by replacing
  the module-level `OrbitEngine` the builder looks up. The frame loop is
  then real -- a real QThread, real signals, real timers -- while a frame
  costs a memset instead of half a second of shading.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget                          # noqa: E402

from spacr.qt import preferences as P                          # noqa: E402
from spacr.qt.widgets import fractal_travel as F               # noqa: E402


# =========================================================================
# Which renderer this machine gets
# =========================================================================


def test_an_unrecognised_platform_plugin_is_given_the_benefit_of_the_doubt(
        monkeypatch):
    """`eglfs` is a real GL platform; the list below it is not exhaustive."""
    monkeypatch.delenv("SPACR_NO_GL", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "eglfs")
    assert F.platform_can_do_opengl() is True
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    assert F.platform_can_do_opengl() is False


def test_a_gl_platform_reports_whether_vispy_is_importable(monkeypatch):
    """And a broken import system counts as no GPU, not as a crash."""
    import importlib.util

    pytest.importorskip("vispy")
    monkeypatch.setattr(F, "platform_can_do_opengl", lambda: True)
    assert F.gpu_is_available() is True

    def _broken(_name):
        raise ValueError("a meta path finder went wrong")

    monkeypatch.setattr(importlib.util, "find_spec", _broken)
    assert F.gpu_is_available() is False


def test_an_explicit_backend_is_taken_at_its_word(monkeypatch):
    """Only `auto` asks; `gpu` and `cpu` are answers already."""
    monkeypatch.setattr(F, "gpu_is_available", lambda: False)
    assert F.resolve_backend("cpu") == "cpu"
    assert F.resolve_backend("gpu") == "gpu"
    assert F.resolve_backend("auto") == "cpu"


# =========================================================================
# The Numba kernels, driven as Python
# =========================================================================


@pytest.fixture
def orbit_sample(monkeypatch):
    """The Python `_orbit_sample` was compiled from, and its helpers with it.

    The kernel calls `_fast_sin`/`_fast_cos` by module-global name, so
    patching the globals is what makes the WHOLE thing run as Python --
    otherwise the inner calls go back through the dispatcher and compile.
    """
    pytest.importorskip("numba")
    sample = F._orbit_sample.py_func
    monkeypatch.setattr(F, "_fast_sin", F._fast_sin.py_func)
    monkeypatch.setattr(F, "_fast_cos", F._fast_cos.py_func)
    monkeypatch.setattr(F, "_orbit_sample", sample)
    return sample


def test_the_fast_trig_stays_bounded_and_tracks_the_real_thing(monkeypatch):
    """A dozen calls per pixel per iteration is why it is approximated; a
    sine that drifted outside -1..1 would blow the colour out."""
    import math

    pytest.importorskip("numba")
    sine = F._fast_sin.py_func
    cosine = F._fast_cos.py_func
    monkeypatch.setattr(F, "_fast_sin", sine)
    for step in range(-60, 61):
        angle = step * 0.37
        assert -1.0001 <= sine(angle) <= 1.0001
        assert sine(angle) == pytest.approx(math.sin(angle), abs=0.002)
        assert cosine(angle) == pytest.approx(math.cos(angle), abs=0.002)


def test_the_orbit_kernel_answers_a_colour_for_every_pixel(orbit_sample):
    sample = orbit_sample
    seen = set()
    for py in range(0, 48, 6):
        for px in range(0, 64, 6):
            red, green, blue = sample(px, py, 64, 48, 3.0, 4.0, 1.5, 5,
                                      0.0, 0.0, 0.0, 0.0)
            assert 0 <= red <= 255 and 0 <= green <= 255 and 0 <= blue <= 255
            seen.add((red, green, blue))
    # The fold has to produce a picture, not a wash: a constant kernel would
    # give one colour for the whole grid.
    assert len(seen) > 8


def test_the_pointer_bends_the_plane_and_then_stops_bending_it(orbit_sample):
    """The 1/r falloff saturates on both sides, which is what keeps a
    pointer resting on a pixel from turning the neighbourhood inside out."""
    sample = orbit_sample
    px, py, width, height = 10, 10, 64, 48
    x = (2.0 * px - width) / min(width, height)
    y = (height - 2.0 * py) / min(width, height)
    near = (x + 0.1, y + 0.1)
    far = (x + 1.0, y + 1.0)

    def at(pointer, pull, push):
        return sample(px, py, width, height, 0.0, 4.0, 1.5, 5,
                      pointer[0], pointer[1], pull, push)

    untouched = at(near, 0.0, 0.0)
    # Well inside the reach, the pull moves the sample...
    assert at(near, 1.0, 0.0) != untouched
    # ...but not five times as far when asked five times as hard.
    assert at(near, 5.0, 0.0) == at(near, 1.0, 0.0)
    # The shove saturates the same way, on its own side of zero.
    assert at(near, 0.0, 1.0) != untouched
    assert at(near, 0.0, 5.0) == at(near, 0.0, 1.0)
    assert at(near, 0.0, 1.0) != at(near, 1.0, 0.0)
    # Far away and gentle it is NOT saturated: doubling the pull there does
    # change the answer, which is what makes the clamp above meaningful.
    assert at(far, 0.02, 0.0) != at(far, 0.01, 0.0)


def test_the_frame_kernel_lays_the_samples_out_at_the_jittered_positions(
        orbit_sample):
    """`_render_into` is the pixel loop and nothing else -- so every texel
    must equal the sample taken at that pixel plus the frame's jitter."""
    sample = orbit_sample
    render = F._render_into.py_func
    output = np.zeros((6, 8, 3), dtype=np.uint8)
    jitter_x, jitter_y = F.JITTERS[2]
    render(output, 1.5, 4.0, 1.5, 5, jitter_x, jitter_y, 0.0, 0.0, 0.0, 0.0)

    for y in range(6):
        for x in range(8):
            expected = sample(x + jitter_x, y + jitter_y, 8, 6,
                              1.5, 4.0, 1.5, 5, 0.0, 0.0, 0.0, 0.0)
            assert tuple(int(v) for v in output[y, x]) == expected
    assert output.std() > 0


def test_the_temporal_blend_weights_the_newest_frame_most(orbit_sample):
    """0.62/0.22/0.10/0.06 over the ring, indexed BACKWARDS from `newest`.

    Getting the ring order wrong would antialias against the future.
    """
    blend = F._blend_temporal.py_func
    ring = np.zeros((4, 2, 3, 3), dtype=np.uint8)
    ring[0] = 200        # newest
    ring[3] = 100        # newest - 1
    ring[2] = 40         # newest - 2
    ring[1] = 10         # newest - 3
    output = np.zeros((2, 3, 3), dtype=np.uint8)
    blend(ring, output, 0)

    expected = int(0.62 * 200 + 0.22 * 100 + 0.10 * 40 + 0.06 * 10)
    assert (output == expected).all()
    assert expected == 150


# =========================================================================
# The CPU widget
# =========================================================================


class _CheapEngine:
    """Answers a frame instantly, so the real loop can be driven in a test.

    Substituted for `OrbitEngine` before the widget is built: the builder
    reads the module global, so this is the seam that does not require
    reaching inside the widget afterwards and racing its thread.
    """

    def __init__(self, thread_count):
        self.thread_count = thread_count
        self.calls = []
        self.explode = None

    def render(self, width, height, t, speed, dream, iterations,
               pointer_x=0.0, pointer_y=0.0, pull=0.0, push=0.0):
        self.calls.append(dict(width=width, height=height, t=t, speed=speed,
                               dream=dream, iterations=iterations,
                               pointer_x=pointer_x, pointer_y=pointer_y,
                               pull=pull, push=push))
        if self.explode is not None:
            raise self.explode
        return np.full((height, width, 3), 7, dtype=np.uint8)


@pytest.fixture
def cpu_widget(qapp, monkeypatch):
    """A live `CpuFractalWidget` whose frames cost nothing."""
    pytest.importorskip("numba")
    monkeypatch.setattr(F, "OrbitEngine", _CheapEngine)
    widget = F._make_cpu_widget(F.Settings(pattern="orbit", backend="cpu"),
                                F.RuntimeControls(),
                                F.HardwareProfile(logical_cpus=4))
    yield widget
    widget.shutdown()
    widget.deleteLater()


def test_a_frame_that_finishes_after_its_widget_is_gone_says_nothing(
        cpu_widget, caplog):
    """`emit` on a freed sender raises RuntimeError, and the handler that
    then emitted the FAILURE signal raised the same way."""
    class _Signal:
        def __init__(self, error):
            self.error = error
            self.attempts = 0

        def emit(self, *args):
            self.attempts += 1
            if self.error is not None:
                raise self.error

    say = cpu_widget._worker._say_something
    live = _Signal(None)
    say(live, "a frame")
    assert live.attempts == 1

    freed = _Signal(RuntimeError("Signal source has been deleted"))
    say(freed, "a frame")                      # must not raise
    assert freed.attempts == 1

    with caplog.at_level("DEBUG", logger=F.LOG.name):
        odd = _Signal(TypeError("wrong argument types"))
        say(odd, "a frame")                    # must not raise either
    assert "could not deliver a frame" in caplog.text


_SAVED_CONTROLS = {
    "speed": 9.0, "dream": 2.0, "variable_speed": True,
    "speed_min": 1.0, "speed_max": 7.0, "speed_period": 30.0,
    "pointer_gravity": False, "pointer_size": 0.5,
    "pointer_strength": 1.5, "zoom_rate": 2.0,
}


class _DeafControls:
    """A `RuntimeControls` that has gone away underneath the registry."""

    @property
    def restart_token(self):
        raise AttributeError("restart_token")

    @property
    def zoom_rate(self):
        raise AttributeError("zoom_rate")

    def __setattr__(self, name, value):
        raise AttributeError(name)


@pytest.fixture
def live_controls(monkeypatch):
    """`_LIVE_CONTROLS` is module state shared by every backdrop."""
    monkeypatch.setattr(F, "_LIVE_CONTROLS", [])
    return F._LIVE_CONTROLS


def test_saved_settings_reach_a_running_backdrop_unless_they_cannot_be_read(
        live_controls, monkeypatch):
    """The backdrop keeps the controls it was built with, so saving
    Preferences has to push into them; a store that will not open must leave
    what is on screen alone rather than half-updating it."""
    controls = F.RuntimeControls(speed=4.0)
    live_controls.append(controls)
    monkeypatch.setattr(P, "get_fractal_settings",
                        lambda: dict(_SAVED_CONTROLS))
    assert F.apply_saved_controls() == 1
    assert controls.speed == 9.0
    assert controls.follow_pointer is False
    assert controls.zoom_rate == 2.0

    def _broken():
        raise RuntimeError("the settings store is not there")

    monkeypatch.setattr(P, "get_fractal_settings", _broken)
    assert F.apply_saved_controls() == 0
    assert controls.speed == 9.0


def test_one_dead_backdrop_does_not_stop_the_live_one_being_updated(
        live_controls, monkeypatch):
    """The list is only trimmed on build, so it can hold objects that have
    stopped answering."""
    controls = F.RuntimeControls(speed=4.0)
    live_controls.extend([_DeafControls(), controls])
    monkeypatch.setattr(P, "get_fractal_settings",
                        lambda: dict(_SAVED_CONTROLS))

    assert F.apply_saved_controls() == 1
    assert controls.speed == 9.0


def test_a_key_press_that_one_backdrop_refuses_still_reaches_the_rest(
        live_controls):
    good = F.RuntimeControls(zoom_rate=1.0)
    live_controls.extend([_DeafControls(), good])

    F.restart_the_dive()
    assert good.restart_token == 1

    rate = F.nudge_zoom_rate(1)
    assert good.zoom_rate == pytest.approx(F.ZOOM_STEP)
    assert rate == pytest.approx(F.ZOOM_STEP)


def test_the_render_scale_is_read_from_the_store_or_assumed_native(
        monkeypatch):
    monkeypatch.setattr(P, "get_fractal_settings",
                        lambda: {"render_scale": 0.5})
    assert F._render_scale() == 0.5

    def _broken():
        raise RuntimeError("the settings store is not there")

    monkeypatch.setattr(P, "get_fractal_settings", _broken)
    assert F._render_scale() == 1.0


# =========================================================================
# The GPU canvas, built against a stand-in vispy
# =========================================================================


#: What the Mandelbrot canvas reads out of the settings store.
#:
#: `max_depth` and `initial_scale` are DELIBERATELY absent: the first has a
#: fallback at the call site and the second has none, so between them the
#: three ways `_mandel_setting` can answer are all exercised by the ordinary
#: build rather than by a special case.
_MANDEL_SAVED = {
    # THE PATH, STATED RATHER THAN INHERITED. These tests exercise the FIXED
    # path -- the guided ones below say so out loud, "built on the fixed
    # path and switched afterwards" -- and until 2026-09-10 none of them set
    # it: they inherited whatever `DEFAULTS["path"]` happened to be. When
    # the published default became "tour" they started testing the tour
    # under names about the reference centre, and three of them failed for
    # a reason none of their docstrings mention.
    #
    # A test that names a mode in its title has to select that mode.
    "path": "fixed",
    "seconds_per_decade": 2.0,
    "base_iterations": 100,
    "iterations_per_decade": 10.0,
    "max_iterations": 64,
    "precision_digits": 30,
    "steering_strength": 0.09,
    "steering_interval_decades": 0.4,
    "steering_duration": 3.8,
    "candidate_count": 4,
}


class _StandInOrbit:
    """A `ReferenceOrbit` that costs nothing to build.

    The real one iterates a couple of thousand points at 320 decimal digits,
    which is seconds -- that is why the canvas builds it on a thread at all.
    """

    def __init__(self, max_iter=2200, digits=320, center=None):
        self.max_iter = int(max_iter)
        self.digits = int(digits)
        self.center = center
        self.packed = np.zeros((2, self.max_iter + 1, 4), dtype=np.float32)


class _FixedPointer:
    """A pointer at a known place, since the real one reads `QCursor.pos()`."""

    def __init__(self, x=0.3, y=-0.2, pull=0.7, push=0.4, error=None):
        self.x = x
        self.y = y
        self.pull = pull
        self.push = push
        self.drag_x = 0.0
        self.drag_y = 0.0
        self.error = error
        self.samples = 0

    def sample(self, _widget, _size=1.0, _strength=1.0):
        self.samples += 1
        if self.error is not None:
            raise self.error
        return self


@pytest.fixture
def stand_in_vispy(qapp, monkeypatch):
    """A `vispy` made entirely of Python, for the length of one test.

    Under `QT_QPA_PLATFORM=offscreen` Qt says "QOpenGLWidget is not supported
    on this platform" and then does whatever the local driver does, which is
    not something to assert against. Everything below `gloo.Program` is
    vispy's code; everything above it is this module's, and that is where
    the seam goes.
    """
    world = types.SimpleNamespace(
        programs=[], textures=[], timers=[], canvases=[], viewports=[],
        states=[], used_app=[], texture_error=None, finish_error=None)

    class _Program(dict):
        def __init__(self, vertex, fragment):
            super().__init__()
            self.vertex = vertex
            self.fragment = fragment
            self.draws = 0
            self.set_error = None
            self.draw_error = None
            world.programs.append(self)

        def __setitem__(self, name, value):
            if self.set_error is not None:
                raise self.set_error
            super().__setitem__(name, value)

        def draw(self, mode):
            self.draws += 1
            if self.draw_error is not None:
                raise self.draw_error

    class _Texture2D:
        def __init__(self, data, **options):
            if world.texture_error is not None:
                raise world.texture_error
            self.data = data
            self.options = options
            world.textures.append(self)

    class _Timer:
        def __init__(self, interval=None, connect=None, start=False):
            self.interval = interval
            self.tick = connect
            self.running = bool(start)
            self.stop_error = None
            world.timers.append(self)

        def stop(self):
            if self.stop_error is not None:
                raise self.stop_error
            self.running = False

    class _Canvas:
        def __init__(self, keys=None, size=(1, 1), show=False, **_options):
            self.native = QWidget()
            self.native.resize(*size)
            self.physical_size = size
            self.currents = 0
            self.updates = 0
            self.closed = False
            self.close_error = None
            world.canvases.append(self)

        def set_current(self):
            self.currents += 1

        def update(self):
            self.updates += 1

        def close(self):
            if self.close_error is not None:
                raise self.close_error
            self.closed = True

    def _finish():
        if world.finish_error is not None:
            raise world.finish_error

    gloo = types.SimpleNamespace(
        Program=_Program, Texture2D=_Texture2D,
        set_state=lambda **options: world.states.append(options),
        set_viewport=lambda *box: world.viewports.append(box),
        gl=types.SimpleNamespace(glFinish=_finish))
    app_module = types.ModuleType("vispy.app")
    app_module.Canvas = _Canvas
    app_module.Timer = _Timer
    app_module.use_app = world.used_app.append
    vispy = types.ModuleType("vispy")
    vispy.app = app_module
    vispy.gloo = gloo

    monkeypatch.setitem(sys.modules, "vispy", vispy)
    monkeypatch.setitem(sys.modules, "vispy.app", app_module)
    return world


@pytest.fixture
def gpu_backdrop(stand_in_vispy):
    """Build GPU backdrops and shut them all down afterwards."""
    built = []

    def build(pattern="orbit", controls=None, **fields):
        widget = F._make_gpu_widget(
            F.Settings(pattern=pattern, backend="gpu", **fields),
            controls if controls is not None else F.RuntimeControls(),
            F.HardwareProfile(logical_cpus=4))
        built.append(widget)
        return widget

    yield build
    for widget in built:
        widget.shutdown()
        widget.deleteLater()


def test_the_preloader_lock_is_found_or_the_backdrop_manages_without_it(
        monkeypatch):
    """Building a GL context while the preloader brings CUDA up is exactly
    the concurrent initialisation the lock exists to prevent -- but this
    widget is usable with no application around it at all."""
    from spacr.qt import app as qt_app

    assert F._heavy_import_lock() is qt_app.HEAVY_IMPORT_LOCK
    monkeypatch.delattr(qt_app, "HEAVY_IMPORT_LOCK")
    assert F._heavy_import_lock() is None


# =========================================================================
# The Mandelbrot dive, which is the GPU canvas's own machinery
# =========================================================================


def _until(predicate, seconds=5.0):
    """Wait for a worker thread to land. Returns whether it did."""
    import time as _time

    deadline = _time.perf_counter() + seconds
    while _time.perf_counter() < deadline:
        if predicate():
            return True
        _time.sleep(0.005)
    return bool(predicate())


@pytest.fixture
def mandel(gpu_backdrop, monkeypatch):
    """A Mandelbrot canvas with a settings store and a cheap reference."""
    from spacr.qt.widgets import fractal_mandelbrot as M

    monkeypatch.setattr(M, "ReferenceOrbit", _StandInOrbit)
    saved = dict(_MANDEL_SAVED)
    monkeypatch.setattr(P, "get_fractal_settings", lambda: dict(saved))

    def build(controls=None):
        canvas = gpu_backdrop("mandelbrot", controls=controls)._canvas
        if canvas._orbit_thread is not None:
            canvas._orbit_thread.join(10)
        return canvas

    return types.SimpleNamespace(build=build, saved=saved, module=M)


def test_the_saved_mandelbrot_numbers_are_used_and_the_rest_defaulted(
        mandel, monkeypatch):
    """Every one of the twelve settings the panel offers was collected,
    stored and then ignored -- the published defaults are the FALLBACK now,
    not the answer."""
    built = []

    class _Recording(_StandInOrbit):
        def __init__(self, **fields):
            super().__init__(**fields)
            built.append(self)

    monkeypatch.setattr(mandel.module, "ReferenceOrbit", _Recording)
    canvas = mandel.build()
    assert (built[0].max_iter, built[0].digits) == (64, 30)
    # `initial_scale` is not in the store and has no fallback at the call
    # site, so the published default is what answers.
    assert float(canvas._program["u_scale"]) == pytest.approx(1.25)

    def _broken():
        raise RuntimeError("the settings store is not there")

    monkeypatch.setattr(P, "get_fractal_settings", _broken)
    mandel.build()
    assert (built[1].max_iter, built[1].digits) == (2200, 320)


def test_a_reference_orbit_that_cannot_be_built_leaves_the_dive_drawing(
        mandel, monkeypatch, caplog):
    """Until it arrives the shader has an all-zero orbit, which renders as
    the flat interior colour rather than as a stall."""
    working = mandel.build()
    assert working._orbit is not None

    class _Refusing:
        def __init__(self, **_fields):
            raise ValueError("mpmath is not installed")

    monkeypatch.setattr(mandel.module, "ReferenceOrbit", _Refusing)
    with caplog.at_level("ERROR", logger=F.LOG.name):
        broken = mandel.build()
    assert broken._orbit is None
    assert "could not build the reference orbit" in caplog.text


def test_the_orbit_texture_is_seeded_and_a_refusal_is_only_logged(
        mandel, monkeypatch, stand_in_vispy, caplog):
    """vispy warns once per DRAW for a uniform a linked program has never
    been given, and the real orbit takes seconds to iterate."""
    class _Refusing:
        def __init__(self, **_fields):
            raise ValueError("no mpmath")

    monkeypatch.setattr(mandel.module, "ReferenceOrbit", _Refusing)
    seeded = mandel.build()
    assert seeded._program["u_orbit"].data.shape == (1, 1, 4)

    stand_in_vispy.texture_error = RuntimeError("no float textures here")
    with caplog.at_level("DEBUG", logger=F.LOG.name):
        bare = mandel.build()
    assert "u_orbit" not in bare._program
    assert "could not seed the orbit texture" in caplog.text


def test_the_finished_orbit_is_uploaded_exactly_once(mandel, monkeypatch,
                                                     stand_in_vispy, caplog):
    """`_orbit_uploaded` is the orbit OBJECT, so a rebuilt one is noticed and
    a driver that refuses the format is not asked sixty times a second."""
    class _Refusing:
        def __init__(self, **_fields):
            raise ValueError("no mpmath")

    monkeypatch.setattr(mandel.module, "ReferenceOrbit", _Refusing)
    canvas = mandel.build()
    before = len(stand_in_vispy.textures)
    canvas._update_uniforms(0.1)
    assert len(stand_in_vispy.textures) == before      # nothing to upload

    orbit = _StandInOrbit(max_iter=4, digits=20)
    canvas._orbit = orbit
    canvas._update_uniforms(0.2)
    assert canvas._program["u_orbit"].data is orbit.packed
    assert len(stand_in_vispy.textures) == before + 1
    canvas._update_uniforms(0.3)
    assert len(stand_in_vispy.textures) == before + 1  # and not again

    stand_in_vispy.texture_error = RuntimeError("rgba32f is not supported")
    canvas._orbit = _StandInOrbit(max_iter=4, digits=20)
    with caplog.at_level("ERROR", logger=F.LOG.name):
        canvas._update_uniforms(0.4)
        canvas._update_uniforms(0.5)
    assert caplog.text.count("could not upload the reference orbit") == 1


def test_the_reference_orbit_caps_the_iteration_budget(mandel):
    """Every pixel perturbs around it, so asking for more iterations than
    the reference has is asking about numbers that are not there."""
    canvas = mandel.build()
    canvas._orbit = None
    uncapped = canvas._mandelbrot_uniforms(0.0)
    assert float(uncapped["u_orbit_length"]) == 1.0
    assert int(uncapped["u_max_iter"]) == 100

    canvas._orbit = _StandInOrbit(max_iter=3, digits=20)
    capped = canvas._mandelbrot_uniforms(0.0)
    assert float(capped["u_orbit_length"]) == 4.0
    assert int(capped["u_max_iter"]) == 3


def test_a_fault_in_the_steering_still_returns_a_whole_frame(mandel,
                                                             monkeypatch,
                                                             caplog):
    """A NameError in the course-plotting once left every uniform unset, so
    the pattern drew nothing at all -- silently."""
    from spacr.qt.widgets.fractal_mandelbrot import SteeringCamera

    canvas = mandel.build()
    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)
    camera = SteeringCamera()
    camera.centre = (0.125, -0.25)
    canvas._camera = camera

    def _broken(*_args, **_kwargs):
        raise NameError("name 'plot_the_course' is not defined")

    monkeypatch.setattr(canvas, "_steer", _broken)
    with caplog.at_level("ERROR", logger=F.LOG.name):
        values = canvas._mandelbrot_uniforms(0.0)

    assert set(values) == {"u_scale", "u_center_offset", "u_depth",
                           "u_orbit_length", "u_max_iter"}
    assert tuple(float(v) for v in values["u_center_offset"]) == (0.125, -0.25)
    assert "could not steer the dive" in caplog.text


def test_the_camera_is_made_once_and_kept(mandel, monkeypatch):
    """It carries the course; rebuilding it every frame would restart the
    dive's aim sixty times a second."""
    class _Refusing:
        def __init__(self, **_fields):
            raise ValueError("no mpmath")

    monkeypatch.setattr(mandel.module, "ReferenceOrbit", _Refusing)
    canvas = mandel.build()
    # With no reference there is nothing to be pointed at, so there is no
    # camera either and the dive looks straight down.
    assert getattr(canvas, "_camera", None) is None
    nowhere = canvas._mandelbrot_uniforms(0.0)
    assert tuple(float(v) for v in nowhere["u_center_offset"]) == (0.0, 0.0)

    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)
    canvas._mandelbrot_uniforms(0.0)
    camera = canvas._camera
    assert camera is not None
    canvas._mandelbrot_uniforms(0.0)
    assert canvas._camera is camera


def test_dragging_moves_the_camera_and_pulls_the_reference_after_it(mandel):
    """Perturbation measures every pixel as an offset from ONE orbit, so a
    camera that walks away from it takes the picture with it."""
    canvas = mandel.build()
    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)
    pointer = _FixedPointer()
    canvas._pointer = pointer
    canvas._mandelbrot_uniforms(0.0)
    camera = canvas._camera
    camera.target = (0.5, 0.5)
    settled = camera.centre

    pointer.drag_x, pointer.drag_y = 0.5, -0.25
    dragged = canvas._mandelbrot_uniforms(0.0)
    assert camera.centre != settled
    assert tuple(float(v) for v in dragged["u_center_offset"]) == (
        pytest.approx(camera.centre[0]), pytest.approx(camera.centre[1]))
    # Consumed, so a dropped frame does not pan twice...
    assert (pointer.drag_x, pointer.drag_y) == (0.0, 0.0)
    # ...the reference is asked to follow at once...
    assert canvas._refine_due == 0.0
    # ...and the hand wins: the camera stops chasing where it was headed.
    assert camera.target is None


def test_the_fixed_path_stays_pointed_at_the_reference(mandel):
    """Surveying the surface for a "more interesting" point was tried and
    made it worse; only a genuinely special point survives a descent, and
    the reference centre already is one."""
    canvas = mandel.build()
    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)
    canvas._mandelbrot_uniforms(0.0)
    canvas._camera.centre = (0.375, -0.125)

    values = canvas._mandelbrot_uniforms(0.0)
    assert tuple(float(v) for v in values["u_center_offset"]) == (
        pytest.approx(0.375), pytest.approx(-0.125))
    assert canvas._camera.step == 0          # nothing was aimed at


def test_the_guided_path_looks_around_on_a_worker_thread(mandel, monkeypatch):
    """A 96x54 escape map does not belong in a frame, so the search runs off
    the GUI thread and the result is collected on a later one."""
    monkeypatch.setattr(mandel.module, "plan_guided_step",
                        lambda *_a, **_k: (0.25, -0.125, 9.0))
    # Built on the fixed path and switched afterwards, so the search below
    # is the first one and not a second reading of one the build started.
    canvas = mandel.build()
    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)
    mandel.saved["path"] = "guided"

    canvas._mandelbrot_uniforms(0.0)                 # starts the search
    assert canvas._plan is not None
    assert _until(lambda: canvas._plan["done"])
    span = canvas._plan["target"]
    assert span == (0.25, -0.125)

    canvas._mandelbrot_uniforms(0.0)                 # collects it
    assert canvas._plan is None
    assert canvas._camera.target is not None
    assert canvas._camera.step == 1
    aimed_at = canvas._camera.next_steer
    assert aimed_at > 0.0

    # Not due again yet, so the frame after that only advances the follow.
    canvas._mandelbrot_uniforms(0.0)
    assert canvas._plan is None
    assert canvas._camera.next_steer == aimed_at

    # And a strength of zero means DO NOT STEER: with no reach there is no
    # direction to look in, so every choice would be arbitrary.
    mandel.saved["steering_strength"] = 0.0
    canvas._camera.centre = (0.5, 0.25)
    held = canvas._mandelbrot_uniforms(0.0)
    assert tuple(float(v) for v in held["u_center_offset"]) == (
        pytest.approx(0.5), pytest.approx(0.25))


def test_a_search_that_finds_nothing_asks_again_sooner(mandel, monkeypatch,
                                                       caplog):
    """Rather than giving up on steering for the rest of the dive."""
    def _broken(*_args, **_kwargs):
        raise RuntimeError("the escape map could not be built")

    monkeypatch.setattr(mandel.module, "plan_guided_step", _broken)
    canvas = mandel.build()
    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)
    mandel.saved["path"] = "guided"

    with caplog.at_level("DEBUG", logger=F.LOG.name):
        canvas._mandelbrot_uniforms(0.0)
        assert _until(lambda: canvas._plan["done"])
        canvas._mandelbrot_uniforms(0.0)

    assert "could not plan a steering step" in caplog.text
    camera = canvas._camera
    assert camera.target is None
    assert camera.step == 0
    # 0.35 of the interval rather than a whole one.
    assert 0.0 < camera.next_steer < 0.4


def test_the_reference_follows_the_view_down(mandel, monkeypatch, caplog):
    """Each refinement is picked out of the CURRENT view, and the view
    shrinks: refining every decade holds the picture sharp to eleven,
    against one or two without."""
    module = mandel.module
    canvas = mandel.build()
    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)

    canvas._mandelbrot_uniforms(0.0)                 # schedules the first one
    due = canvas._refine_due
    assert due == pytest.approx(canvas._depth + module.REFINE_EVERY, abs=0.01)
    canvas._mandelbrot_uniforms(0.0)                 # not due yet
    assert canvas._refine_due == due

    fresh = _StandInOrbit(max_iter=8, digits=20)
    monkeypatch.setattr(module, "best_reference_in_view",
                        lambda *_a, **_k: (0.1, 0.2))
    monkeypatch.setattr(module, "rebased_orbit",
                        lambda *_a, **_k: ((0.1, 0.2), fresh))

    # A survey already running is not started twice.
    canvas._refine_due = 0.0
    canvas._refine_thread_running = True
    canvas._mandelbrot_uniforms(0.0)
    assert _until(lambda: canvas._refined is not None, seconds=0.3) is False

    canvas._refine_thread_running = False
    canvas._camera.centre = (0.3, 0.4)
    canvas._mandelbrot_uniforms(0.0)                 # starts the survey
    assert _until(lambda: canvas._refined is not None)

    canvas._mandelbrot_uniforms(0.0)                 # lands it
    assert canvas._orbit is fresh
    # The reference moved by (0.1, 0.2) and the camera's offset by the
    # opposite, so the picture is exactly where it was -- resetting the
    # offset to zero here was a jump of a whole survey offset every decade.
    assert canvas._camera.centre == (pytest.approx(0.2), pytest.approx(0.2))
    assert canvas._refine_due == pytest.approx(
        canvas._depth + module.REFINE_EVERY, abs=0.01)


def test_a_refinement_that_fails_keeps_the_reference_it_has(mandel,
                                                            monkeypatch,
                                                            caplog):
    """A poor reference draws noise, where an old one merely draws a view
    that is off centre."""
    module = mandel.module
    canvas = mandel.build()
    orbit = _StandInOrbit(max_iter=8, digits=20)
    canvas._orbit = orbit

    def _broken(*_args, **_kwargs):
        raise RuntimeError("the escape map could not be built")

    monkeypatch.setattr(module, "best_reference_in_view", _broken)
    canvas._mandelbrot_uniforms(0.0)                 # schedules
    canvas._refine_due = 0.0
    with caplog.at_level("DEBUG", logger=F.LOG.name):
        canvas._mandelbrot_uniforms(0.0)             # starts the survey
        assert _until(lambda: canvas._refined is not None)
        canvas._mandelbrot_uniforms(0.0)             # lands nothing

    assert "could not refine the reference" in caplog.text
    assert canvas._orbit is orbit
    assert canvas._refine_due == pytest.approx(
        canvas._depth + module.REFINE_EVERY, abs=0.01)

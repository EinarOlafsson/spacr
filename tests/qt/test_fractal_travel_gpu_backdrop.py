"""The fractal backdrop's GPU half, built against a stand-in vispy.

Under ``QT_QPA_PLATFORM=offscreen`` a real GL canvas behaves however the local
driver behaves, so the canvas is built against the pure-Python vispy harness
of ``tests.qt.test_cov_r5_fractal_travel``: everything below
``gloo.Program`` is vispy's code, everything above it is this module's.  What
is pinned: the canvas refuses at construction when vispy cannot import, each
pattern gets its own shader, uniforms and detail budget, a failed draw or a
freed canvas stops instead of storming, the detail tracks GPU time, the
widget pauses and closes cleanly, and the runtime controls steer the pointer
and the Mandelbrot dive.  One general test per behaviour family, each walking
a case table with pytest's ``subtests`` fixture.
"""
from __future__ import annotations

import sys
import time

import pytest

pytest.importorskip("PySide6")

from spacr.qt import preferences as P  # noqa: E402
from spacr.qt.widgets import fractal_travel as F  # noqa: E402
from tests.qt.test_cov_r5_fractal_travel import (  # noqa: E402,F401
    _MANDEL_SAVED, _FixedPointer, _StandInOrbit, gpu_backdrop, stand_in_vispy,
)

pytestmark = pytest.mark.qt


def _lines(canvas):
    return canvas.stats_text().splitlines()


@pytest.fixture
def mandelbrot_store(monkeypatch):
    """A Mandelbrot settings store and a reference orbit that costs nothing."""
    from spacr.qt.widgets import fractal_mandelbrot as M

    monkeypatch.setattr(M, "ReferenceOrbit", _StandInOrbit)
    monkeypatch.setattr(P, "get_fractal_settings", lambda: dict(_MANDEL_SAVED))
    return M


def test_the_gpu_canvas_is_built_per_pattern_or_refused(gpu_backdrop, stand_in_vispy, mandelbrot_store, monkeypatch, subtests):
    """No vispy is a GpuBackendError; each pattern gets its shader, uniforms and detail budget."""
    from spacr.qt.widgets import fractal_cascade, fractal_space

    with subtests.test(msg="vispy will not import"), monkeypatch.context() as m:
        m.setitem(sys.modules, "vispy", None)
        with pytest.raises(F.GpuBackendError):
            F._make_gpu_widget(F.Settings(backend="gpu"), F.RuntimeControls(),
                               F.HardwareProfile(logical_cpus=4))

    with subtests.test(msg="built with and without the preloader lock"), monkeypatch.context() as m:
        before = len(stand_in_vispy.canvases)
        gpu_backdrop("orbit")
        m.setattr(F, "_heavy_import_lock", lambda: None)
        gpu_backdrop("orbit")
        assert len(stand_in_vispy.canvases) == before + 2
        assert stand_in_vispy.used_app[-2:] == ["pyside6", "pyside6"]

    expected = {
        "orbit": (F.FRAGMENT_SHADER, 6, True),
        "cascade": (fractal_cascade.FRAGMENT_SHADER, 5, True),
        "space": (fractal_space.FRAGMENT_SHADER, 4, False),
        "mandelbrot": (mandelbrot_store.FRAGMENT_SHADER, 6, False),
    }
    shaders = set()
    for pattern, (shader, detail, has_dream) in expected.items():
        with subtests.test(msg=f"pattern {pattern}"):
            canvas = gpu_backdrop(pattern)._canvas
            shaders.add(canvas._program.fragment)
            assert canvas._program.fragment == shader
            assert canvas._detail == detail
            assert ("u_dream" in canvas._program) is has_dream
            # The deep zoom is a camera with its own uniforms; the fields share time and size.
            assert ("u_time" in canvas._program) is (pattern != "mandelbrot")
            if pattern != "mandelbrot":
                assert tuple(canvas._program["u_resolution"]) == (1200, 760)
            deep = set(canvas._mandelbrot_uniforms(1.0))
            assert deep == ({"u_scale", "u_center_offset", "u_depth", "u_orbit_length",
                             "u_max_iter"} if pattern == "mandelbrot" else set())
    with subtests.test(msg="four distinct shaders"):
        assert len(shaders) == 4


def test_the_gpu_canvas_draws_measures_and_stops_instead_of_storming(gpu_backdrop, stand_in_vispy, caplog, subtests):
    """Resizes move the viewport, failures and freed canvases stop the timer, detail tracks GPU time."""
    with subtests.test(msg="resize moves the viewport"):
        canvas = gpu_backdrop("orbit")._canvas
        canvas.physical_size = (640, 480)
        canvas.on_resize(None)
        assert stand_in_vispy.viewports[-1] == (0, 0, 640, 480)

    with subtests.test(msg="a failing draw stops the canvas once"):
        canvas = gpu_backdrop("orbit")._canvas
        canvas.on_draw(None)
        assert canvas._program.draws >= 1 and canvas._dead is False
        canvas._program.draw_error = RuntimeError("the GL context was lost")
        caplog.clear()
        with caplog.at_level("WARNING", logger=F.LOG.name):
            canvas.on_draw(None)
        assert (canvas._dead, canvas._timer.running) == (True, False)
        assert "stopped" in caplog.text and "backdrop" in caplog.text
        before = canvas._program.draws
        canvas.on_draw(None)
        assert canvas._program.draws == before

    with subtests.test(msg="detail follows the GPU time within bounds"):
        canvas = gpu_backdrop("orbit")._canvas

        def draw_with(smoothed):
            canvas._last_sample = 0.0
            canvas._render_ema = smoothed
            canvas.on_draw(None)
            return canvas._detail

        assert canvas._detail == 6
        assert [draw_with(s) for s in (0.1, 0.1, 0.016, 0.0, 0.0, 0.0)] == [5, 5, 5, 6, 7, 7]
        drawn, sampled_at = canvas._program.draws, canvas._last_sample
        canvas.on_draw(None)
        assert canvas._program.draws == drawn + 1
        assert canvas._last_sample == sampled_at

    with subtests.test(msg="glFinish fails, next benchmark still scheduled"):
        canvas = gpu_backdrop("orbit")._canvas
        canvas._last_sample = 0.0
        stand_in_vispy.finish_error = RuntimeError("glFinish is not available")
        try:
            canvas.on_draw(None)
        finally:
            stand_in_vispy.finish_error = None
        assert canvas._render_ema is None
        assert canvas._last_sample > 0.0

    with subtests.test(msg="timer ticks: paused, freed, dead"):
        canvas = gpu_backdrop("orbit")._canvas
        canvas._on_timer(None)
        assert canvas.updates == 1
        canvas._paused = True
        canvas._on_timer(None)
        assert canvas.updates == 1
        canvas._paused = False
        canvas._program.set_error = RuntimeError("Internal C++ object (Canvas) already deleted.")
        canvas._on_timer(None)
        assert (canvas._dead, canvas._timer.running) == (True, False)
        canvas._on_timer(None)
        assert canvas.updates == 1

    with subtests.test(msg="timer stopped twice and after deletion"):
        canvas = gpu_backdrop("orbit")._canvas
        canvas.stop_timer()
        assert canvas._timer.running is False
        canvas._timer.stop_error = RuntimeError("the timer is gone")
        canvas.stop_timer()

    with subtests.test(msg="Qt freeing the native widget kills the canvas"):
        canvas = gpu_backdrop("orbit")._canvas
        assert canvas._dead is False
        canvas.native.destroyed.emit()
        assert (canvas._dead, canvas._timer.running) == (True, False)

    with subtests.test(msg="the stats line"):
        canvas = gpu_backdrop("orbit")._canvas
        first, second, timing = _lines(canvas)
        assert first.startswith(f"v{F.VERSION} · GPU/balanced · orbit · spatial ")
        assert second.endswith(" · detail 6")
        assert timing == "measuring"
        canvas._render_ema = 0.0125
        assert _lines(canvas)[-1] == "12.5 ms GPU"
        canvas._paused = True
        assert _lines(canvas)[-1] == "paused for a run"


def test_the_gpu_widget_pauses_and_closes_cleanly(gpu_backdrop, subtests):
    """Pause/resume report change and closing stops the canvas, even one that is already gone."""
    with subtests.test(msg="winds down and starts again"):
        widget = gpu_backdrop("orbit")
        assert (widget.backend_name, widget.is_paused()) == ("gpu", False)
        assert [widget.pause(), widget.pause(), widget.is_paused()] == [True, False, True]
        assert widget.stats_text() == widget._canvas.stats_text()
        assert [widget.resume(), widget.resume()] == [True, False]
        assert [widget.set_animating(False), widget.set_animating(True)] == [True, True]

    with subtests.test(msg="closing stops the canvas"):
        widget = gpu_backdrop("orbit")
        canvas = widget._canvas
        widget.close()
        assert (canvas._dead, canvas.closed, canvas._timer.running) == (True, True, False)

    with subtests.test(msg="a canvas that will not close"):
        widget = gpu_backdrop("orbit")
        widget._canvas.close_error = RuntimeError("already deleted")
        widget.shutdown()
        assert widget._canvas._dead is True


def test_runtime_controls_steer_the_pointer_and_the_dive(gpu_backdrop, mandelbrot_store, subtests):
    """The pointer warps the folds but never the deep zoom; the dive integrates its depth and restarts."""
    with subtests.test(msg="pointer state per pattern and switch"):
        controls = F.RuntimeControls(follow_pointer=True)
        folded = gpu_backdrop("orbit", controls=controls)._canvas
        folded._pointer = _FixedPointer()
        assert folded._pointer_state() == (0.3, -0.2, 0.7, 0.4)
        diving = gpu_backdrop("mandelbrot", controls=controls)._canvas
        diving._pointer = _FixedPointer()
        assert diving._pointer_state() == (0.3, -0.2, 0.0, 0.0)
        controls.follow_pointer = False
        assert folded._pointer_state() == (0.0, 0.0, 0.0, 0.0)
        controls.follow_pointer = True
        folded._pointer = _FixedPointer(error=RuntimeError("no cursor here"))
        assert folded._pointer_state() == (0.0, 0.0, 0.0, 0.0)

    def dive(controls):
        canvas = gpu_backdrop("mandelbrot", controls=controls)._canvas
        if canvas._orbit_thread is not None:
            canvas._orbit_thread.join(10)
        return canvas

    with subtests.test(msg="the dive integrates its depth and backs out to the surface"):
        controls = F.RuntimeControls(speed=4.0, zoom_rate=1.0)
        canvas = dive(controls)
        canvas._depth = 5.0
        canvas._zoom_clock = time.perf_counter() - 1.0
        assert float(canvas._mandelbrot_uniforms(0.0)["u_depth"]) == pytest.approx(7.0, abs=0.05)
        controls.zoom_rate = -4.0
        canvas._zoom_clock = time.perf_counter() - 1.0
        assert float(canvas._mandelbrot_uniforms(0.0)["u_depth"]) == 0.0

    with subtests.test(msg="a restart resets depth, camera and plan"):
        controls = F.RuntimeControls()
        canvas = dive(controls)
        canvas._orbit = _StandInOrbit(max_iter=8, digits=20)
        canvas._mandelbrot_uniforms(0.0)
        camera = canvas._camera
        camera.centre = (0.25, -0.5)
        camera.step = 7
        canvas._depth = 6.0
        controls.restart_token += 1
        after = canvas._mandelbrot_uniforms(0.0)
        assert float(after["u_depth"]) == 0.0
        assert (camera.centre, camera.step, canvas._plan) == ((0.0, 0.0), 0, None)

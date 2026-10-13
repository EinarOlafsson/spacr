"""The fractal backdrop's tour, shaders, blends and platform check at edges.

Pinned here, each as what the user sees or gets:

* on the "tour" path the deep zoom builds one glide camera and keeps it; a
  drag takes the camera from the tour and Ctrl+R hands it back;
* a tour with nowhere to go leaves the camera target alone, and a missing
  region list is an empty tour rather than an error;
* a shader whose main has nested blocks is still rewritten to the N x N
  grid, and one whose main never closes is refused out loud;
* the weighted temporal blend weights the newest frame most, across a ring
  of any length; without numba it says numba is needed;
* Cocoa and Windows can host a GL canvas without a DISPLAY;
* a CPU backdrop built or shut down with no application still runs and
  still stops.
"""
from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path
import types

import numpy as np
import pytest

from spacr.qt.widgets import fractal_travel as F
from tests.qt.test_cov_r5_fractal_travel import (  # noqa: F401
    _FixedPointer,
    _no_mouse_button_is_held,
    _StandInOrbit,
    gpu_backdrop,
    mandel,
    stand_in_vispy,
)

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# The tour
# ---------------------------------------------------------------------------

def test_the_tour_path_builds_one_glide_camera_and_keeps_it(mandel):  # noqa: F811
    from spacr.qt.widgets.fractal_mandelbrot import _GlideCamera

    mandel.saved["path"] = "tour"
    canvas = mandel.build()
    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)

    values = canvas._mandelbrot_uniforms(0.0)
    glide = canvas._glide
    assert isinstance(glide, _GlideCamera)
    assert len(values["u_center_offset"]) == 2

    canvas._mandelbrot_uniforms(0.0)
    assert canvas._glide is glide


def test_a_drag_takes_the_camera_and_a_restart_gives_it_back(mandel):  # noqa: F811
    mandel.saved["path"] = "tour"
    controls = F.RuntimeControls()
    canvas = mandel.build(controls=controls)
    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)
    pointer = _FixedPointer()
    canvas._pointer = pointer
    canvas._mandelbrot_uniforms(0.0)
    glide = canvas._glide
    assert not glide.taken

    pointer.drag_x, pointer.drag_y = 0.5, -0.25
    canvas._mandelbrot_uniforms(0.0)
    assert glide.taken
    assert (pointer.drag_x, pointer.drag_y) == (0.0, 0.0)

    controls.restart_token += 1
    canvas._mandelbrot_uniforms(0.0)
    assert not glide.taken


def test_a_tour_with_nowhere_to_go_leaves_the_camera_alone():
    class Nowhere:
        regions = (("a", 0.0, 0.0, 0.5, 1.0),)
        active = True

        def target_at(self, _seconds):
            return None

    camera = types.SimpleNamespace(target=(0.25, 0.25))
    pilot = F._TourPilot(Nowhere())
    assert pilot.steer(camera, 3.0, 1.0) is False
    assert camera.target == (0.25, 0.25)


def test_a_missing_region_list_is_an_empty_tour(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.qt.widgets.fractal_regions", None)
    tour = F.default_region_tour()
    assert tour.regions == ()
    assert not tour.active
    assert tour.target_at(5.0) is None


# ---------------------------------------------------------------------------
# Shaders
# ---------------------------------------------------------------------------

_NESTED = """
vec3 shade(vec2 p) { return vec3(p, 0.0); }
void main() {
    vec3 total = vec3(0.0);
    if (true) {
        total += shade(gl_FragCoord.xy + vec2(0.25, 0.25));
    }
    gl_FragColor = vec4(total, 1.0);
}
"""


def test_a_main_with_nested_blocks_is_still_rewritten():
    out = F._supersampled_shader(_NESTED, 3)
    assert "sy < 3" in out and "sx < 3" in out
    assert "shade(gl_FragCoord.xy + offset)" in out
    assert "if (true)" not in out
    assert out.startswith("\nvec3 shade(vec2 p)")


def test_a_main_that_never_closes_is_refused():
    unclosed = "void main() {\n    x += shade(gl_FragCoord.xy + vec2(0.1));\n"
    with pytest.raises(ValueError, match="not a sample grid"):
        F._supersampled_shader(unclosed, 3)


# ---------------------------------------------------------------------------
# The weighted blend
# ---------------------------------------------------------------------------

def test_the_weighted_blend_counts_back_from_the_newest_frame():
    pytest.importorskip("numba")
    blend = F._blend_weighted.py_func
    ring = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    ring[1] = 200        # newest
    ring[0] = 100        # newest - 1
    ring[2] = 10         # newest - 2
    weights = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    output = np.zeros((2, 2, 3), dtype=np.uint8)
    blend(ring, output, 1, weights)
    assert (output == int(0.5 * 200 + 0.3 * 100 + 0.2 * 10)).all()


def test_without_numba_the_weighted_blend_says_so(monkeypatch):
    real_import = builtins.__import__

    def refuse(name, g=None, l=None, fromlist=(), level=0):  # noqa: E741
        if name == "numba" or name.startswith("numba."):
            raise ImportError("numba is not installed")
        return real_import(name, g, l, fromlist, level)

    import importlib.util

    import spacr.qt.widgets as package

    # A private copy under its own name: a worker thread left by an earlier
    # test can re-import the shared module, numba and all, between a
    # sys.modules purge and the import (seen once on a CI shard).
    path = Path(package.__file__).with_name("fractal_travel.py")
    spec = importlib.util.spec_from_file_location(
        "spacr.qt.widgets._fractal_travel_without_numba", path)
    scratch = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, scratch)
    monkeypatch.setattr(builtins, "__import__", refuse)
    monkeypatch.delitem(sys.modules, "numba", raising=False)
    spec.loader.exec_module(scratch)
    assert scratch.njit is None

    with pytest.raises(RuntimeError, match="numba is required"):
        scratch._blend_weighted(1, 2, 3, 4)


# ---------------------------------------------------------------------------
# The platform
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("platform", ["cocoa", "windows"])
def test_cocoa_and_windows_host_gl_without_a_display(monkeypatch, platform):
    monkeypatch.delenv("SPACR_NO_GL", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", platform)
    assert F.platform_can_do_opengl() is True


@pytest.mark.parametrize("system", ["darwin", "win32"])
def test_the_default_platform_on_a_mac_or_pc_hosts_gl(monkeypatch, system):
    monkeypatch.delenv("SPACR_NO_GL", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "")
    monkeypatch.setattr(F.sys, "platform", system)
    assert F.platform_can_do_opengl() is True


# ---------------------------------------------------------------------------
# A CPU backdrop with no application
# ---------------------------------------------------------------------------

def test_a_cpu_backdrop_without_an_application_still_runs_and_stops(
        qapp, monkeypatch):
    pytest.importorskip("numba")
    from PySide6.QtWidgets import QApplication

    from tests.qt.test_cov_r5_fractal_travel import _CheapEngine

    monkeypatch.setattr(F, "OrbitEngine", _CheapEngine)
    with monkeypatch.context() as patch:
        patch.setattr(QApplication, "instance", staticmethod(lambda: None))
        widget = F._make_cpu_widget(F.Settings(pattern="orbit",
                                               backend="cpu"),
                                    F.RuntimeControls(),
                                    F.HardwareProfile(logical_cpus=4))
    try:
        assert widget._thread.isRunning()
        with monkeypatch.context() as patch:
            patch.setattr(QApplication, "instance",
                          staticmethod(lambda: None))
            widget.shutdown()
        assert not widget._thread.isRunning()
    finally:
        widget.shutdown()
        widget.deleteLater()

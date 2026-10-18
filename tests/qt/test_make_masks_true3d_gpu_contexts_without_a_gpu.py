"""True 3D GPU contexts, checked on a computer with no GPU (N673).

The OpenGL and wgpu contexts the True 3D view draws with are made only where
a driver answers, so a CPU-only runner (CUDA_VISIBLE_DEVICES empty) skips the
real pixel tests in test_673_true3d_gpu.py and test_673_true3d_wgpu.py. Here
the Qt OpenGL classes and the spaCR renderers are replaced by fakes that
answer like a driver would, so every refusal a context makes, the frame it
hands back, and how it is freed are checked everywhere.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6 import QtGui

from spacr import layers
from spacr.qt import preferences
from spacr.qt.screens import make_masks as mm

from .test_673_napari_true3d import _wait_render, screen, view
from .test_673_true3d_wgpu import _FakeGPU, _chain, _restart

__all__ = ["screen", "view"]


class _FakeRenderer:
    """A volume renderer that answers like a working GPU without one."""

    made = []

    def __init__(self, *args, renderer="Fake GPU 9000", frame=None, close_error=None):
        self.args = args
        self.renderer = renderer
        self.frame = frame
        self.close_error = close_error
        self.calls = []
        self.closed = 0
        _FakeRenderer.made.append(self)

    def limits(self):
        return {"renderer": self.renderer, "max_3d": 2048}

    def render(self, layer_list, rays, **kwargs):
        self.calls.append((list(layer_list), rays, kwargs))
        size = kwargs.get("size") or (2, 2)
        if self.frame is not None:
            return self.frame, np.zeros(self.frame.shape[:2], np.int32)
        rgb = np.full(tuple(size) + (3,), 200, np.uint8)
        return rgb, np.ones(tuple(size), np.int32)

    def close(self):
        self.closed += 1
        if self.close_error:
            raise RuntimeError(self.close_error)


class _FakeFormat:
    def __init__(self, *args):
        pass


class _FakeContext:
    """QOpenGLContext as a driver answers it."""

    es = False
    current = True
    made = []

    def __init__(self):
        self.current_calls = 0
        self.done_calls = 0
        _FakeContext.made.append(self)

    def setFormat(self, _fmt):  # noqa: N802
        pass

    def create(self):
        return True

    def isOpenGLES(self):  # noqa: N802
        return type(self).es

    def format(self):
        return _FakeFormat()

    def makeCurrent(self, _surface):  # noqa: N802
        self.current_calls += 1
        return type(self).current

    def doneCurrent(self):  # noqa: N802
        self.done_calls += 1


class _FakeSurface:
    valid = True

    def setFormat(self, _fmt):  # noqa: N802
        pass

    def create(self):
        pass

    def isValid(self):  # noqa: N802
        return type(self).valid


def _fake_gl(monkeypatch, *, es=False, valid=True, current=True, **renderer):
    _FakeContext.es, _FakeContext.current, _FakeContext.made = es, current, []
    _FakeSurface.valid = valid
    _FakeRenderer.made = []
    monkeypatch.setattr(QtGui, "QOpenGLContext", _FakeContext)
    monkeypatch.setattr(QtGui, "QOffscreenSurface", _FakeSurface)
    monkeypatch.setattr(layers, "GLVolumeRenderer", lambda: _FakeRenderer(**renderer))
    monkeypatch.setattr(mm, "find_spec", lambda name: object())


def _fake_wgpu(monkeypatch, **renderer):
    _FakeRenderer.made = []
    asked = []
    monkeypatch.setattr(layers, "wgpu_adapter",
                        lambda backend, allow_software=True: asked.append(
                            (backend, allow_software)) or f"adapter:{backend}")
    monkeypatch.setattr(layers, "WGPUVolumeRenderer",
                        lambda adapter: _FakeRenderer(adapter, **renderer))
    monkeypatch.setattr(mm, "find_spec", lambda name: object())
    return asked


def _request():
    rays = layers.VolumeRays(origin=(0.0, 0.0, 0.0), across=(0.0, 0.0, 1.0),
                             down=(0.0, 1.0, 0.0), depth=(1.0, 0.0, 0.0), shape=(2, 3, 4))
    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    entry = ("Image", True, volume, True, 1.0, "gray", (0.0, 23.0), "mip", "matte")
    hidden = ("Hidden", True, volume, False, 1.0, "gray", (0.0, 23.0), "mip", "matte")
    return SimpleNamespace(rect=(10.0, 14.6, 5.0, 8.2), rays=rays, highlight=2,
                           voxel_size=(), lens=None, layers=(entry, hidden), cache={},
                           version=1, steps=(1, 1, 1))


# ---------------------------------------------------------------------------
# The OpenGL context
# ---------------------------------------------------------------------------

def test_the_gl_context_refuses_without_vispy(monkeypatch):
    monkeypatch.setattr(mm, "find_spec", lambda name: None if name == "vispy" else object())
    with pytest.raises(RuntimeError, match="VisPy is not installed"):
        mm._GLVolumeContext()


@pytest.mark.parametrize("setup, message", [
    (dict(es=True), "only OpenGL ES is available"),
    (dict(valid=False), "no offscreen OpenGL surface"),
    (dict(current=False), "could not be made current"),
    (dict(frame=np.zeros((2, 2, 3), np.uint8)), "OpenGL test frame came back empty"),
    (dict(frame=np.full((3, 2, 3), 9, np.uint8)), "OpenGL test frame came back empty"),
])
def test_the_gl_context_says_why_a_driver_cannot_draw(monkeypatch, setup, message):
    _fake_gl(monkeypatch, **setup)
    with pytest.raises(RuntimeError, match=message):
        mm._GLVolumeContext()


def test_auto_refuses_a_software_gl_but_a_chosen_opengl_accepts_it(monkeypatch):
    _fake_gl(monkeypatch, renderer="llvmpipe (LLVM 17.0.6, 256 bits)")
    with pytest.raises(RuntimeError, match=r"OpenGL is drawn in software \(llvmpipe"):
        mm._GLVolumeContext(allow_software=False)
    context = mm._GLVolumeContext(allow_software=True)
    assert context.backend == "opengl"
    assert context.name() == "llvmpipe (LLVM 17.0.6, 256 bits)"


def test_the_gl_context_draws_the_request_at_its_rect_size_and_releases(monkeypatch):
    _fake_gl(monkeypatch)
    context = mm._GLVolumeContext()
    gl = _FakeContext.made[-1]
    renderer = _FakeRenderer.made[-1]
    assert renderer.calls and renderer.calls[0][0][0].name == "probe"
    assert gl.current_calls == gl.done_calls == 1
    rgb, picked = context.render(_request())
    assert rgb.shape == (3, 5, 3) and picked.shape == (3, 5)
    layer_list, rays, kwargs = renderer.calls[-1]
    assert [layer.name for layer in layer_list] == ["Image"]
    assert kwargs == {"highlight": 2, "voxel_size": None, "lens": None, "size": (3, 5)}
    assert gl.current_calls == gl.done_calls == 2
    context.close()
    assert renderer.closed == 1 and gl.done_calls == 3


def test_a_gl_context_that_cannot_be_freed_only_logs_it(monkeypatch, caplog):
    _fake_gl(monkeypatch, close_error="context lost")
    context = mm._GLVolumeContext()
    with caplog.at_level(logging.DEBUG, logger=mm.LOG.name):
        context.close()
    assert any("could not be freed" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# The wgpu context
# ---------------------------------------------------------------------------

def test_the_wgpu_context_refuses_without_wgpu(monkeypatch):
    monkeypatch.setattr(mm, "find_spec", lambda name: None if name == "wgpu" else object())
    with pytest.raises(RuntimeError, match="wgpu is not installed"):
        mm._WGPUVolumeContext("vulkan")


def test_the_wgpu_context_asks_for_its_backend_and_draws_at_the_rect_size(monkeypatch):
    asked = _fake_wgpu(monkeypatch, renderer="Fake RTX")
    context = mm._WGPUVolumeContext("metal", allow_software=False)
    assert asked == [("metal", False)]
    renderer = _FakeRenderer.made[-1]
    assert renderer.args == ("adapter:metal",)
    assert context.backend == "metal" and context.name() == "Fake RTX"
    rgb, picked = context.render(_request())
    assert rgb.shape == (3, 5, 3)
    assert renderer.calls[-1][2]["size"] == (3, 5)
    context.close()
    assert renderer.closed == 1


def test_an_empty_wgpu_test_frame_frees_the_device_and_names_the_api(monkeypatch):
    _fake_wgpu(monkeypatch, frame=np.zeros((2, 2, 3), np.uint8))
    with pytest.raises(RuntimeError, match="the Direct3D 12 test frame came back empty"):
        mm._WGPUVolumeContext("d3d12")
    assert _FakeRenderer.made[-1].closed == 1


def test_a_wgpu_device_that_cannot_be_freed_only_logs_it(monkeypatch, caplog):
    _fake_wgpu(monkeypatch, close_error="device lost")
    context = mm._WGPUVolumeContext("vulkan")
    with caplog.at_level(logging.DEBUG, logger=mm.LOG.name):
        context.close()
    assert any("wgpu renderer could not be freed" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# The saved view settings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("saved, expected", [
    ({"backend": "vulkan", "scale": 40, "finish": "glossy"}, ("auto", 40, "glossy")),
    ({"backend": "cpu", "scale": "forty"}, ("cpu", None, "matte")),
    ({"backend": "warp", "scale": [3]}, ("auto", None, "matte")),
    ({"scale": 500}, ("auto", 100, "matte")),
    ({"scale": -3, "finish": "velvet"}, ("auto", 1, "matte")),
])
def test_saved_view_settings_are_checked_against_this_computer(monkeypatch, saved, expected):
    monkeypatch.setattr(preferences, "get_volume_view", lambda: dict(saved))
    monkeypatch.setattr(mm, "volume_render_backends", lambda: ("auto", "cpu"))
    assert mm._volume_view_settings() == expected


# ---------------------------------------------------------------------------
# The backend chain on the stage
# ---------------------------------------------------------------------------

def _choose(stage, key):
    """Pick ``key`` in the Render backend list, as on a computer that offers it."""
    box = stage._backend_box
    if box.findData(key) < 0:
        box.addItem(stage._backend_names()[key], key)
    box.setCurrentIndex(box.findData(key))


def test_a_chain_whose_backends_all_failed_before_stays_on_the_cpu_quietly(view, monkeypatch):
    stage = view
    _chain(monkeypatch)
    stage._backend_choice = "auto"
    stage._gl, stage._backend_note = None, ""
    stage._gpu_refused = {"vulkan", "opengl"}
    stage._open_gpu()
    assert stage._gl is None and _FakeGPU.made == []
    assert stage._backend_note == ""


def test_an_explicit_backend_that_fails_mid_frame_goes_to_the_cpu(view, qtbot, monkeypatch):
    stage = view
    _chain(monkeypatch)
    _restart(stage, qtbot)
    _choose(stage, "vulkan")
    _wait_render(stage, qtbot)
    assert stage._gl is not None and stage._gl.backend == "vulkan"
    stage._gl.failing = "device lost"
    assert stage._submit_render("preview")
    _wait_render(stage, qtbot)
    assert stage._gl is None and stage._render_state.backend == "cpu"
    assert "GPU rendering is unavailable (device lost)" in stage._backend_note
    assert _FakeGPU.made.count("opengl") == 0


def test_choosing_another_gpu_backend_frees_the_one_drawing(view, qtbot, monkeypatch):
    stage = view
    _chain(monkeypatch)
    _restart(stage, qtbot)
    first = stage._gl
    assert first.backend == "vulkan"
    closed = []
    first.close = lambda: closed.append(first.backend)
    _choose(stage, "opengl")
    _wait_render(stage, qtbot)
    assert closed == ["vulkan"]
    assert stage._gl is not first and stage._gl.backend == "opengl"
    assert stage._render_state.backend == "opengl"

"""N684: the backdrop widget, its producer and click ripples at their edges.

A producer with no canvas or no frame publishes nothing; a theme without
a graphics path ignores the backend choice; a stale or blocked click
ripple behaves as documented; a gravity burst can skip its wave; and the
software probe and NVML monitor fail closed.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QWidget

from spacr.qt import preferences as prefs
from spacr.qt.widgets import ambient


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "ci3w.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    return settings


class _Blank:
    """An engine with no frame to give and no graphics renderer."""

    def __init__(self):
        self.calls = []

    def shade(self, width, height):
        self.calls.append((width, height))
        return None


def _run_producer(engine, size, until):
    producer = ambient._FrameProducer(engine, threading.Lock(), 240, size)
    producer.start()
    try:
        deadline = time.monotonic() + 5
        while not until() and time.monotonic() < deadline:
            time.sleep(0.005)
    finally:
        producer.stop()
    assert not producer.is_alive()
    return producer


def test_a_producer_without_a_canvas_never_shades():
    engine = _Blank()
    started = time.monotonic()
    producer = _run_producer(engine, (0, 120), lambda: time.monotonic() - started > 0.05)
    assert engine.calls == []
    assert producer.frames_shaded == 0 and producer.latest() is None


def test_a_frame_the_engine_refuses_is_not_published():
    engine = _Blank()
    producer = _run_producer(engine, (40, 30), lambda: len(engine.calls) >= 3)
    assert len(engine.calls) >= 3 and set(engine.calls) == {(40, 30)}
    assert producer.frames_shaded == 0 and producer.latest() is None


@pytest.fixture
def no_producer(monkeypatch):
    monkeypatch.setattr(ambient.AmbientWidget, "_start_producer", lambda self: None)


def test_a_theme_without_graphics_ignores_the_backend_choice(qtbot, no_producer):
    prefs._set_flow_graphics_backend("gpu")
    widget = ambient.AmbientWidget(theme="fractal", seed=2)
    qtbot.addWidget(widget)
    assert not isinstance(widget._engine, ambient._GraphicsBackend)
    assert "_graphics_backend" not in vars(widget._engine)
    widget._set_graphics_backend("AUTO")
    assert widget._graphics_backend == "auto"
    widget._set_graphics_backend("vulkan")
    assert widget._graphics_backend == "cpu"
    assert "_graphics_backend" not in vars(widget._engine)
    before = widget._engine
    widget._rebuild_engine()
    assert widget._engine is not before
    assert not isinstance(widget._engine, ambient._GraphicsBackend)
    assert "_graphics_backend" not in vars(widget._engine)


def _field(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(320, 180)
    field = ambient.AmbientWidget(host, theme="data_art_impulse_lens", gravity_radius=0,
                                  popup_wave_frequency=0, blink_percent=0,
                                  density=1, seed=19)
    field.resize(host.size())
    host.show()
    field.show()
    field._timer.stop()
    return host, field


def test_a_superseded_click_candidate_sends_nothing(qtbot, no_producer):
    _host, field = _field(qtbot)
    stale = ((0.5, 0.5), ambient._CONTAINER_RIPPLES[0])
    field._click_candidate = ((0.2, 0.2), ambient._CONTAINER_RIPPLES[0])
    assert field._commit_click_ripple(stale) is False
    assert field._click_candidate is not None
    assert not field.engine._popup_waves


def test_a_click_ripple_waits_for_a_busy_engine(qtbot, no_producer):
    _host, field = _field(qtbot)
    assert field._should_run()
    candidate = ((0.25, 0.75), ambient._CONTAINER_RIPPLES[0])
    field._click_candidate = candidate
    held, done = threading.Event(), threading.Event()

    def shading():
        with field._engine_lock:
            held.set()
            done.wait(5)

    producer = threading.Thread(target=shading)
    producer.start()
    try:
        assert held.wait(5)
        assert field._commit_click_ripple(candidate) is True
        assert not field.engine._popup_waves
    finally:
        done.set()
        producer.join()
    assert field._click_candidate is None
    field._art_input._consume(field.engine)
    origin = field.engine._popup_waves[-1][1]
    assert origin == pytest.approx((0.25, 0.75))


def test_a_gravity_burst_can_skip_its_wave(qapp):
    engine = ambient.make_engine("data_art_impulse_lens", "spacr", "#191919", seed=5)
    engine.gravity_radius = 0.3
    engine._add_impulse((0.4, 0.6), ripple=False)
    engine._add_impulse((0.6, 0.4), ripple=True)
    engine.ripples_enabled = False
    engine._add_impulse((0.5, 0.5), ripple=True)
    assert len(engine._gravity_impulses) == 3
    rippled = list(engine._gravity_ripple_impulses.values())
    assert [impulse[1] for impulse in rippled] == [(0.6, 0.4)]


@pytest.mark.parametrize("error", [OSError("no interpreter"),
                                   InterruptedError("interrupted spawn"),
                                   subprocess.TimeoutExpired("python", 5)])
def test_a_probe_that_cannot_run_keeps_the_cpu(monkeypatch, error):
    import importlib.util

    def run(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(ambient, "_FLOW_GRAPHICS_PROBE", None)
    monkeypatch.delenv("SPACR_NO_GL", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(subprocess, "Popen", run)
    assert ambient._flow_graphics_preflight() is False
    assert ambient._FLOW_GRAPHICS_PROBE is False


def test_a_hardware_probe_answer_is_accepted(monkeypatch):
    import importlib.util

    monkeypatch.setattr(ambient.sys, "platform", "linux")

    seen = []
    real_popen = subprocess.Popen

    def run(command, **kwargs):
        """Exercise captured child output using Python only, without GL."""
        seen.append(command[-1])
        return real_popen([sys.executable, '-c', "print('hardware-gl-ok')"], **kwargs)

    monkeypatch.setattr(ambient, "_FLOW_GRAPHICS_PROBE", None)
    monkeypatch.delenv("SPACR_NO_GL", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(subprocess, "Popen", run)
    assert ambient._flow_graphics_preflight() is True
    assert "llvmpipe" in seen[0] and "create_context" in seen[0]


def test_a_failing_nvml_query_keeps_the_gpu_for_analysis(monkeypatch):
    events = []

    def lost(_index):
        raise RuntimeError("GPU fell off the bus")

    def shutdown():
        events.append("shutdown")
        raise RuntimeError("already shut")

    module = types.SimpleNamespace(nvmlInit=lambda: events.append("init"),
                                   nvmlDeviceGetCount=lambda: 1,
                                   nvmlDeviceGetHandleByIndex=lost,
                                   nvmlShutdown=shutdown)
    monkeypatch.setitem(sys.modules, "pynvml", module)
    assert ambient._flow_compute_idle() is False
    assert events == ["init", "shutdown"]


def test_policy_applies_without_a_running_application(monkeypatch, qapp):
    policy = ambient._AnimationResourcePolicy()
    held = []
    monkeypatch.setattr(policy, "_hold_fractals", lambda: held.append("hold"))
    monkeypatch.setattr(policy, "_release_fractals", lambda: held.append("release"))
    emitted = []
    policy.changed.connect(lambda: emitted.append(1))
    monkeypatch.setattr(ambient.QApplication, "instance", staticmethod(lambda: None))
    policy.gpu_busy = True
    policy._apply()
    policy.gpu_busy = False
    policy._apply()
    assert held == ["hold", "release"] and emitted == [1, 1]


def test_random_palette_aurora_tints_each_column_with_its_own_colour(qapp):
    import numpy as np

    engine = ambient.make_engine("aurora", "random", "#f2f2f2", seed=12)
    assert not engine.dark and engine._random_palette
    layer = engine._aurora_layers(480, 300)[0]
    ids = np.asarray(layer["color_ids"])
    texture, words = engine._aurora_texture(layer)
    assert (texture.width(), texture.height()) == (layer["width"],
                                                   ambient._AURORA_TEXTURE_ROWS)
    rgb = np.stack([(words >> shift) & 0xFF for shift in (16, 8, 0)], axis=-1)
    inverse = 255 - rgb.astype(np.float64)
    colours = np.array([[c.red(), c.green(), c.blue()] for c in engine.paint_colors],
                       dtype=np.float64)
    target = 255.0 - colours[ids]
    assert inverse.max() > 0, "light aurora must darken the page somewhere"
    scale = inverse.max(axis=-1) / np.maximum(target.max(axis=-1), 1.0)[None, :]
    assert np.abs(inverse - scale[..., None] * target[None, :, :]).max() <= 2.0
    assert len(set(ids.tolist())) > 1


def test_random_palette_points_release_a_renderer_and_stay_on_the_cpu(qapp):
    class Renderer:
        def __init__(self):
            self._owner = threading.get_ident()
            self.closed = False
            self.draws = []

        def _draw(self, *args):
            self.draws.append(args)

        def _close(self):
            self.closed = True

    engine = ambient.make_engine("data_art_impulse_lens", "random", "#191919", seed=3)
    engine.set_max_pixels(320 * 200)
    engine._graphics_backend = "gpu"
    box = {}

    def run():
        renderer = engine._point_graphics = Renderer()
        box["frame"] = engine.shade(320, 200)
        box["renderer"] = renderer

    thread = threading.Thread(target=run, name="spacr-ambient-shade")
    thread.start()
    thread.join()
    renderer = box["renderer"]
    assert renderer.closed and renderer.draws == []
    assert engine._point_graphics is None
    assert box["frame"] is not None and not box["frame"].isNull()


def _hold_lock(lock):
    held, done = threading.Event(), threading.Event()

    def shading():
        with lock:
            held.set()
            done.wait(5)

    thread = threading.Thread(target=shading)
    thread.start()
    assert held.wait(5)
    return thread, done


def test_a_panel_ripple_waits_for_a_busy_engine(qtbot, no_producer):
    from PySide6.QtCore import QRect

    _host, field = _field(qtbot)
    thread, done = _hold_lock(field._engine_lock)
    try:
        field._ripple_from_rect(QRect(40, 30, 120, 80))
        assert not field.engine._popup_waves
    finally:
        done.set()
        thread.join()
    field._art_input._consume(field.engine)
    assert len(field.engine._popup_waves) == 1
    assert isinstance(field.engine._popup_waves[0][1][0], tuple)


def test_a_dialog_owned_by_a_nested_child_still_sends_popup_waves(qtbot, no_producer,
                                                                  monkeypatch):
    from PySide6.QtWidgets import QDialog

    host, field = _field(qtbot)
    inner = QWidget(host)
    inner.setGeometry(10, 10, 100, 80)
    deeper = QWidget(inner)
    dialog = QDialog(deeper)
    dialog.setWindowFlag(ambient.Qt.Widget, True)
    dialog.setGeometry(20, 20, 60, 40)
    dialog.show()
    monkeypatch.setattr(ambient.QApplication, "activePopupWidget",
                        staticmethod(lambda: dialog))
    segments = field._popup_wave_origin_for_tick()
    assert segments
    assert field._popup_wave_popup_id == id(dialog)
    stranger = QDialog()
    qtbot.addWidget(stranger)
    stranger.show()
    monkeypatch.setattr(ambient.QApplication, "activePopupWidget",
                        staticmethod(lambda: stranger))
    assert field._popup_wave_origin_for_tick() is None


def test_feedback_for_a_widget_whose_object_is_gone_is_dropped(qapp):
    import shiboken6

    gone = QWidget()
    shiboken6.delete(gone)
    assert not shiboken6.isValid(gone)
    before = ambient._CONTAINER_RIPPLES[0]
    assert ambient.field_ripple_for_widget(gone, edge="left") is None
    assert ambient._CONTAINER_RIPPLES[0] == before + 1

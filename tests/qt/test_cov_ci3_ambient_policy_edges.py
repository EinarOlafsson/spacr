"""N684: Dynamic animation and the measured Automatic choice at their edges.

Covers the cost-table lookup, the renderer cache between idle checks, the
cheap "could graphics exist" test, GPU detection without and with PyTorch,
and the workload policy surviving a missing registry, a dead widget and a
fractal that refuses to pause or resume.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtGui import QColor, QImage, QPainter

from spacr.qt import preferences as prefs
from spacr.qt.widgets import ambient


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "ci3.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    return settings


def _on_producer(function):
    box = {}

    def run():
        try:
            box["value"] = function()
        except BaseException as error:
            box["error"] = error

    thread = threading.Thread(target=run, name="spacr-ambient-shade")
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def _engine(theme, **kwargs):
    return ambient.make_engine(theme, ambient.default_palette_for(theme), "#191919",
                               seed=7, **kwargs)


@pytest.mark.parametrize("theme,pixels,density,expected", [
    ("fractal", 2073600, 1.0, None),
    ("blobs", 2073600, 1.0, True),
    ("drift", 2073600, 1.0, False),
    ("aurora", 8294400, 0.1, True),
    ("data_art_impulse_lens", 0, 1.0, None),
    ("data_art_impulse_lens", 2073600, 0.0, None),
    ("data_art_impulse_lens", 320 * 200, 1.0, False),
    ("data_art_impulse_lens", 2073600, 0.25, True),
    ("data_art_impulse_lens", 8294400, 1.0, False),
    ("data_art_genetic_advection", 2073600, 0.1, False),
    ("data_art_genetic_advection", 8294400, 1.0, False),
    # 2026-10-11: 28c7b269e regenerated the table from the round-5 workstation
    # audit (features/data/684_animation_gpu_audit_r5_2026-10-11). The nearest
    # point-atlas row (2073600 px, density 1.0) went from cpu 1.913 / gpu 1.806 ms
    # (gpu 0.94x cpu, a near tie, CPU) to cpu 2.236 / gpu 1.984 ms (0.89x, under
    # the 0.9 _GRAPHICS_GAIN), so Automatic now measures a graphics win here.
    ("data_art_point_atlas", 1920 * 1200, 0.9, True),
    ("data_art_point_atlas", 3840 * 2400, 0.9, True)])
def test_the_nearest_measured_row_decides(theme, pixels, density, expected):
    assert ambient._measured_graphics_win(theme, pixels, density) is expected


def test_automatic_follows_the_table_before_the_unmeasured_rule(qapp):
    measured = _engine("data_art_genetic_advection", density=1.0)
    assert measured._graphics_unmeasured(3840, 2160, work=10 ** 6) is True
    assert measured._graphics_automatic(3840, 2160, work=10 ** 6) is False
    lens = _engine("data_art_impulse_lens", density=0.25)
    assert lens._graphics_unmeasured(1920, 1080, work=10 ** 6) is False
    assert lens._graphics_automatic(1920, 1080) is True


def test_unmeasured_data_art_wants_dense_4k_work(qapp):
    ribbon = _engine("data_art_point_atlas")
    assert ribbon._graphics_unmeasured(3840, 2160, work=100000) is True
    assert ribbon._graphics_unmeasured(3840, 2160, work=99999) is False
    assert ribbon._graphics_unmeasured(1920, 1080, work=10 ** 7) is False
    paper = _engine("data_art_tissue_facets")
    assert paper._graphics_unmeasured(3840, 2160) is True
    assert paper._graphics_unmeasured(3839, 2160) is False


class _Renderer:
    made = 0

    def __init__(self):
        _Renderer.made += 1
        self._owner = threading.get_ident()
        self.closed = False

    def _close(self):
        self.closed = True


def test_a_kept_renderer_is_reused_until_the_idle_check_expires(monkeypatch, qapp):
    checks = []
    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda: True)
    monkeypatch.setattr(ambient, "_flow_compute_idle", lambda: checks.append(1) or True)
    monkeypatch.setattr(ambient, "_FlowPointGraphics", _Renderer)
    _Renderer.made = 0
    engine = _engine("blobs")
    engine._graphics_backend = "gpu"

    def frames():
        first = engine._graphics_renderer(320, 200)
        assert engine._graphics_idle_until > ambient.time.monotonic()
        second = engine._graphics_renderer(320, 200)
        engine._graphics_idle_until = 0.0
        third = engine._graphics_renderer(320, 200)
        engine._release_point_graphics()
        return first, second, third

    first, second, third = _on_producer(frames)
    assert first is second is third
    assert _Renderer.made == 1 and len(checks) == 2
    assert first.closed and engine._point_graphics is None


def test_a_constructor_error_is_remembered_for_the_engine(monkeypatch, qapp):
    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda: True)
    monkeypatch.setattr(ambient, "_flow_compute_idle", lambda: True)
    attempts = []

    def broken():
        attempts.append(1)
        raise RuntimeError("no context")

    monkeypatch.setattr(ambient, "_FlowPointGraphics", broken)
    engine = _engine("aurora")
    engine._graphics_backend = "gpu"
    assert _on_producer(lambda: engine._graphics_renderer(320, 200)) is None
    assert engine._graphics_failed
    assert _on_producer(lambda: engine._graphics_renderer(320, 200)) is None
    assert attempts == [1]


def test_graphics_possible_needs_the_module_and_no_failed_probe(monkeypatch):
    import importlib.util

    looked = []
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: looked.append(name) or None)
    monkeypatch.setattr(ambient, "_FLOW_GRAPHICS_PROBE", False)
    assert ambient._flow_graphics_possible() is False
    assert looked == []
    monkeypatch.setattr(ambient, "_FLOW_GRAPHICS_PROBE", None)
    assert ambient._flow_graphics_possible() is False
    assert looked == ["moderngl"]
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(ambient, "_FLOW_GRAPHICS_PROBE", True)
    assert ambient._flow_graphics_possible() is True


def test_stratified_frames_at_their_edges(qapp):
    engine = _engine("drift")
    assert engine.shade(0, 200) is None and engine.shade(320, -1) is None
    canvas = QImage(4, 4, QImage.Format_RGB32)
    canvas.fill(QColor("#123456"))
    painter = QPainter(canvas)
    try:
        mode = painter.compositionMode()
        engine.blit(painter, None, 4, 4)
        engine.blit(painter, QImage(), 4, 4)
        engine.blit(painter, QImage(2, 2, QImage.Format_RGB32), 0, 4)
        assert painter.compositionMode() == mode
    finally:
        painter.end()
    assert canvas.pixelColor(0, 0) == QColor("#123456")
    engine.set_max_pixels(160 * 100)
    image = engine.shade(320, 200)
    assert (image.width(), image.height()) == engine.buffer_size(320, 200)
    assert image.width() < 320 and image.format() == QImage.Format_ARGB32_Premultiplied


def test_blinking_stars_get_their_own_white_passes(qapp):
    engine = _engine("drift", blink_percent=10.0, density=3)
    engine.set_time(3.0)
    passes = engine._dot_passes(1920, 1080)
    white = [(pen, points) for pen, points in passes if pen.color() == QColor("white")]
    assert white
    sizes = {pen.widthF() for pen, _ in white}
    assert sizes <= {engine.dot_size(layer) for layer in range(len(ambient.DRIFT_LAYERS))}
    blinking = sum(len(points) for _pen, points in white)
    assert blinking == len(engine._blinking_indices(len(engine.geometry(1920, 1080)))) > 0


class _Cuda:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        if isinstance(self.available, Exception):
            raise self.available
        return self.available


@pytest.mark.parametrize("cuda,mps,expected", [
    (True, None, True), (False, True, True), (False, False, False),
    (False, "absent", False), (RuntimeError("driver"), None, False)])
def test_gpu_presence_asks_an_imported_torch(monkeypatch, cuda, mps, expected):
    torch = types.SimpleNamespace(cuda=_Cuda(cuda))
    if mps != "absent":
        torch.backends = types.SimpleNamespace(mps=None if mps is None else _Cuda(mps))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "pynvml", None)
    assert ambient._analysis_gpu_present() is expected


@pytest.mark.parametrize("count,fails,expected", [(2, False, True), (0, False, False),
                                                   (1, True, False)])
def test_gpu_presence_without_torch_asks_nvml(monkeypatch, count, fails, expected):
    events = []

    def init():
        if fails:
            raise OSError("no driver")
        events.append("init")

    module = types.SimpleNamespace(nvmlInit=init, nvmlDeviceGetCount=lambda: count,
                                   nvmlShutdown=lambda: events.append("shutdown"))
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setitem(sys.modules, "pynvml", module)
    assert ambient._analysis_gpu_present() is expected
    assert events == ([] if fails else ["init", "shutdown"])


class _Registry(QObject):
    changed = Signal()

    def __init__(self, handles=()):
        super().__init__()
        self.handles = list(handles)
        self.connections = 0

    def active(self):
        return list(self.handles)


def test_policy_defaults_on_when_the_switch_cannot_be_read(monkeypatch, qapp):
    def broken():
        raise RuntimeError("settings unreadable")

    monkeypatch.setattr(prefs, "_dynamic_animation_enabled", broken)
    policy = ambient._AnimationResourcePolicy()
    assert policy.enabled is True
    assert not policy.cpu_busy and not policy.gpu_busy


def test_attach_follows_the_registry_once(monkeypatch, qapp):
    from spacr.qt import bridge

    fake = _Registry()
    asked = []
    monkeypatch.setattr(bridge, "registry", lambda: asked.append(1) or fake)
    monkeypatch.setattr(ambient, "_analysis_gpu_present", lambda: False)
    policy = ambient._AnimationResourcePolicy()
    policy.attach()
    calls = len(asked)
    policy.attach()
    assert len(asked) == calls
    fake.handles.append(types.SimpleNamespace(app_key="measure", user_visible=True))
    fake.changed.emit()
    assert policy.cpu_busy and policy.pauses_animation()


def test_a_missing_registry_leaves_the_policy_idle(monkeypatch, qapp, caplog):
    from spacr.qt import bridge

    def gone():
        raise ImportError("no bridge")

    monkeypatch.setattr(bridge, "registry", gone)
    policy = ambient._AnimationResourcePolicy()
    with caplog.at_level("DEBUG", logger=ambient.LOG.name):
        policy.attach()
    assert "could not follow the run registry" in caplog.text
    assert policy._attached is False
    assert policy._kinds() == set()
    assert not policy.blocks_gpu()


def test_applying_skips_a_widget_whose_object_is_gone(monkeypatch, qapp):
    class Dead:
        pass

    dead = Dead()
    widgets = [dead]

    def sync():
        raise RuntimeError("Internal C++ object already deleted")

    dead._sync_run_state = sync
    monkeypatch.setattr(ambient, "AmbientWidget", Dead)
    monkeypatch.setattr(type(qapp), "allWidgets", staticmethod(lambda: widgets))
    monkeypatch.setattr(ambient, "_live_spaceout_fractals", lambda: [])
    policy = ambient._AnimationResourcePolicy()
    emitted = []
    policy.changed.connect(lambda: emitted.append(1))
    policy.cpu_busy = True
    policy._apply()
    assert emitted == [1]


class _Fractal:
    def __init__(self, error):
        self.error = error
        self.backend_name = "gpu"

    def is_paused(self):
        return False

    def pause(self):
        raise self.error

    def resume(self):
        raise self.error


@pytest.mark.parametrize("error", [RuntimeError("deleted"), ValueError("refused")])
def test_fractals_that_refuse_are_not_remembered(monkeypatch, qapp, caplog, error):
    fractal = _Fractal(error)
    monkeypatch.setattr(ambient, "_live_spaceout_fractals", lambda: [fractal])
    policy = ambient._AnimationResourcePolicy()
    policy.gpu_busy = True
    with caplog.at_level("DEBUG", logger=ambient.LOG.name):
        policy._hold_fractals()
        assert policy._paused_fractals == []
        policy._paused_fractals = [fractal]
        policy._release_fractals()
    assert policy._paused_fractals == []
    logged = "could not pause a fractal" in caplog.text
    assert logged is isinstance(error, ValueError)
    assert ("could not resume a fractal" in caplog.text) is isinstance(error, ValueError)


def test_no_policy_exists_before_qt(monkeypatch):
    monkeypatch.setattr(ambient.QApplication, "instance", staticmethod(lambda: None))
    assert ambient._animation_resource_policy() is None
    assert ambient._RESOURCE_POLICY is None


@pytest.fixture
def refusals(monkeypatch, caplog):
    monkeypatch.setattr(ambient, "_GRAPHICS_REFUSALS_LOGGED", set())
    caplog.set_level("WARNING", logger=ambient.LOG.name)
    return caplog


def _warnings(caplog):
    return [record.getMessage() for record in caplog.records
            if record.levelname == "WARNING"]


def test_each_refusal_names_its_reason_and_is_logged_once(monkeypatch, refusals, qapp):
    monkeypatch.setattr(ambient, "_flow_compute_idle", lambda: True)
    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda: True)
    monkeypatch.setattr(ambient, "_FlowPointGraphics", _Renderer)
    engine = _engine("blobs")
    engine._graphics_backend = "gpu"

    def reason(width=320, height=200):
        assert engine._graphics_renderer(width, height) is None
        return engine._graphics_refusal

    assert _on_producer(lambda: reason(3841, 2160)) == (
        "canvas larger than 3840x2160 or big-endian host")
    assert _on_producer(lambda: reason(3841, 2160)) == (
        "canvas larger than 3840x2160 or big-endian host")
    monkeypatch.setattr(ambient, "_decorative_gpu_blocked", lambda: True)
    assert _on_producer(reason) == "Dynamic animation holds the GPU for analysis"
    monkeypatch.setattr(ambient, "_decorative_gpu_blocked", lambda: False)
    monkeypatch.setattr(ambient, "_flow_compute_idle", lambda: False)
    assert _on_producer(reason) == "another process is computing on the GPU"
    assert _warnings(refusals) == [
        "blobs animation stays on the CPU: canvas larger than 3840x2160 or big-endian host",
        "blobs animation stays on the CPU: Dynamic animation holds the GPU for analysis",
        "blobs animation stays on the CPU: another process is computing on the GPU"]
    engine._graphics_idle_until = 0.0
    monkeypatch.setattr(ambient, "_flow_compute_idle", lambda: True)

    def accept_then_cpu():
        made = engine._graphics_renderer(320, 200)
        accepted = engine._graphics_refusal
        engine._graphics_backend = "cpu"
        assert engine._graphics_renderer(320, 200) is None
        return made, accepted

    renderer, accepted = _on_producer(accept_then_cpu)
    assert isinstance(renderer, _Renderer) and accepted is None
    assert renderer.closed and engine._graphics_refusal is None


def test_automatic_choosing_the_cpu_is_recorded_but_not_logged(refusals, qapp):
    engine = _engine("drift")
    engine._graphics_backend = "auto"
    assert _on_producer(lambda: engine._graphics_renderer(1920, 1080)) is None
    assert engine._graphics_refusal == "automatic chose the CPU"
    assert _warnings(refusals) == []


def test_probe_constructor_and_draw_failures_say_what_failed(monkeypatch, refusals, qapp):
    monkeypatch.setattr(ambient, "_flow_compute_idle", lambda: True)
    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda: False)
    probed = _engine("aurora")
    probed._graphics_backend = "gpu"
    assert _on_producer(lambda: probed._graphics_renderer(320, 200)) is None
    assert probed._graphics_refusal == "no hardware OpenGL 4.3 context (probe failed)"
    assert probed._graphics_failed
    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda: True)

    def broken():
        raise OSError("EGL_BAD_DISPLAY")

    monkeypatch.setattr(ambient, "_FlowPointGraphics", broken)
    built = _engine("aurora")
    built._graphics_backend = "gpu"
    assert _on_producer(lambda: built._graphics_renderer(320, 200)) is None
    assert built._graphics_refusal == "context creation failed: OSError: EGL_BAD_DISPLAY"

    class Failing(_Renderer):
        def _draw_blobs(self, *args):
            raise KeyError("x" * 1000)

    monkeypatch.setattr(ambient, "_FlowPointGraphics", Failing)
    drawn = _engine("blobs")
    drawn._graphics_backend = "gpu"
    assert _on_producer(lambda: drawn._graphics_frame("_draw_blobs", 320, 200)) is None
    reason = drawn._graphics_refusal
    assert reason.startswith("_draw_blobs failed in _draw_blobs line ")
    assert ": KeyError: " in reason and len(reason) < 700
    assert drawn._graphics_failed and drawn._point_graphics is None
    assert len(_warnings(refusals)) == 3


def test_aurora_now_follows_its_measured_rows_not_the_4k_fallback(qapp):
    aurora = _engine("aurora", density=1.0)
    assert aurora._graphics_unmeasured(3840, 2160) is True
    assert aurora._graphics_unmeasured(1920, 1080) is False
    assert aurora._graphics_automatic(1920, 1080) is True
    assert ambient._measured_graphics_win("aurora", 1920 * 1080, 1.0) is True

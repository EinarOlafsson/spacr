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
    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda stop_event=None: True)
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
    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda stop_event=None: True)
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
    monkeypatch.setattr(ambient, "_live_ambient_widgets", lambda: widgets)
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
    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda stop_event=None: True)
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
    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda stop_event=None: False)
    probed = _engine("aurora")
    probed._graphics_backend = "gpu"
    assert _on_producer(lambda: probed._graphics_renderer(320, 200)) is None
    assert probed._graphics_refusal == ambient._graphics_probe_refusal()
    assert probed._graphics_failed
    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda stop_event=None: True)

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


def test_the_probe_refusal_names_the_platform_path(monkeypatch):
    """A Mac is told about its own 4.1/Radeon path, not an OpenGL 4.3 probe."""
    monkeypatch.setattr(ambient.sys, "platform", "linux")
    assert ambient._graphics_probe_refusal() == "no hardware OpenGL 4.3 context (probe failed)"
    monkeypatch.setattr(ambient.sys, "platform", "darwin")
    monkeypatch.setattr(ambient, "_mac_radeon_graphics", lambda: False)
    assert "Radeon devices only" in ambient._graphics_probe_refusal()
    monkeypatch.setattr(ambient, "_mac_radeon_graphics", lambda: True)
    assert ambient._graphics_probe_refusal() == "no hardware OpenGL 4.1 context (probe failed)"


@pytest.mark.parametrize('low_power,reduce_motion,missing_process,expected', [
    (True, False, False, True), (False, True, False, True),
    (False, False, False, False), (False, True, True, True),
])
def test_macos_motion_policy_queries_the_runtime_and_caches_the_answer(
        monkeypatch, low_power, reduce_motion, missing_process, expected):
    """An explicit Objective-C API shim checks policy and TTL, not native macOS behavior."""
    calls = []
    clock = [100.0]
    owners = {b'NSProcessInfo': 1, b'NSWorkspace': 2}
    selectors = {b'processInfo': 10, b'sharedWorkspace': 20,
                 b'isLowPowerModeEnabled': 30, b'accessibilityDisplayShouldReduceMotion': 40}

    def get_class(name):
        calls.append(name)
        return 0 if missing_process and name == b'NSProcessInfo' else owners[name]

    def selector(name):
        return selectors[name]

    objc = types.SimpleNamespace(objc_getClass=get_class, sel_registerName=selector,
                                 objc_msgSend=object())
    pointer, boolean = object(), object()

    def function_type(result, *arguments):
        def bind(address):
            assert address == 123
            if result is pointer:
                return lambda owner, message: owner * 100
            return lambda owner, message: low_power if message == 30 else reduce_motion
        return bind

    def load(name):
        assert name == '/usr/lib/libobjc.A.dylib'
        return objc

    ffi = types.SimpleNamespace(cdll=types.SimpleNamespace(LoadLibrary=load),
        c_void_p=pointer, c_bool=boolean, c_char_p=object(), CFUNCTYPE=function_type,
        cast=lambda value, type: types.SimpleNamespace(value=123))
    monkeypatch.setattr(ambient, 'sys', types.SimpleNamespace(platform='darwin'))
    monkeypatch.setattr(ambient, 'time', types.SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(ambient, '_MACOS_STILLNESS', [float('-inf'), False])
    monkeypatch.setitem(sys.modules, 'ctypes', ffi)
    assert ambient._macos_asks_for_stillness() is expected
    first = list(calls)
    assert first and first[0] == b'NSProcessInfo'
    assert (b'NSWorkspace' in first) is not low_power
    assert ambient._macos_asks_for_stillness() is expected
    assert calls == first
    clock[0] += ambient._MACOS_STILLNESS_TTL_S + 0.1
    assert ambient._macos_asks_for_stillness() is expected
    assert calls == first + first


def test_macos_motion_metadata_failure_is_not_a_claim_of_native_stillness(monkeypatch):
    """An unavailable Objective-C runtime returns an unknown/off answer and caches it."""
    def unavailable(name):
        raise OSError('runtime unavailable')

    monkeypatch.setattr(ambient, 'sys', types.SimpleNamespace(platform='darwin'))
    monkeypatch.setattr(ambient, '_MACOS_STILLNESS', [float('-inf'), False])
    monkeypatch.setitem(sys.modules, 'ctypes', types.SimpleNamespace(
        cdll=types.SimpleNamespace(LoadLibrary=unavailable)))
    assert ambient._macos_asks_for_stillness() is False
    assert ambient._MACOS_STILLNESS[1] is False


@pytest.mark.parametrize('version', [410, 430])
def test_gl_table_shader_adapter_changes_only_the_supported_access_protocol(version):
    """A recording context observes source adaptation without claiming GPU compilation."""
    source = ('#version 430\n'
        'layout(std430, binding=2) readonly buffer P { uint palette[]; };\n'
        'layout(std430, binding=3) readonly buffer C { vec4 columns[]; };\n'
        'layout(std430, binding=4) readonly buffer L { vec4 lookup[]; };\n'
        'void main() { uint p = palette[7]; vec4 c = columns[3]; vec4 l = lookup[2]; }')
    recorded = {}
    token = object()

    def program(**sources):
        recorded.update(sources)
        return token

    renderer = ambient._FlowPointGraphics.__new__(ambient._FlowPointGraphics)
    renderer._context = types.SimpleNamespace(version_code=version, program=program)
    assert renderer._make_program(vertex_shader=source) is token
    actual = recorded['vertex_shader']
    if version == 430:
        assert actual == source
    else:
        assert actual.startswith('#version 410') and 'buffer ' not in actual
        assert 'uniform usampler2D spacr_palette;' in actual
        for name, index, suffix in [('palette', 7, '.r'), ('columns', 3, ''), ('lookup', 2, '')]:
            assert f'texelFetch(spacr_{name}, ivec2(int({index}), 0), 0){suffix}' in actual
    assert source.startswith('#version 430')


@pytest.mark.parametrize('binding,components,dtype', [(2, 1, 'u4'), (3, 4, 'f4'), (4, 4, 'f4')])
def test_gl_41_table_transfers_preserve_bytes_reuse_size_and_retire_replaced_texture(
        binding, components, dtype):
    """Recording texture APIs verify unchanged CPU bytes and resource ownership only."""
    import numpy as np

    created, events = [], []

    class Texture:
        def __init__(self, size, count, kind):
            self.size = size
            self.components = count
            self.dtype = kind
            self.raw = None

        def write(self, raw):
            self.raw = raw

        def use(self, location):
            events.append(('use', self, location))

        def release(self):
            events.append(('release', self))

    def texture(size, count, *, dtype):
        value = Texture(size, count, dtype)
        created.append(value)
        return value

    renderer = ambient._FlowPointGraphics.__new__(ambient._FlowPointGraphics)
    renderer._resources = []
    renderer._context = types.SimpleNamespace(version_code=410, texture=texture)
    uniform = types.SimpleNamespace(value=None)
    name = {2: 'palette', 3: 'columns', 4: 'lookup'}[binding]
    program = {'spacr_' + name: uniform}
    raw = np.arange(2 * components, dtype=np.dtype(dtype)).tobytes()
    buffer = types.SimpleNamespace(read=lambda: raw)
    renderer._bind_storage(buffer, binding, program)
    first = created[0]
    assert first.raw == raw and first.size == (2, 1)
    assert (first.components, first.dtype) == (components, dtype)
    assert uniform.value == binding
    renderer._bind_storage(buffer, binding, program)
    assert created == [first]
    raw = np.arange(3 * components, dtype=np.dtype(dtype)).tobytes()
    renderer._bind_storage(buffer, binding, program)
    second = created[1]
    assert second.raw == raw and second.size == (3, 1)
    assert ('release', first) in events
    assert first not in renderer._resources and second in renderer._resources
    assert events[-1] == ('use', second, binding)


def test_gl_43_storage_binding_keeps_the_direct_buffer_protocol():
    """A capable context keeps native storage API dispatch rather than inventing textures."""
    calls = []
    renderer = ambient._FlowPointGraphics.__new__(ambient._FlowPointGraphics)
    renderer._context = types.SimpleNamespace(version_code=430)
    renderer._bind_storage(types.SimpleNamespace(bind_to_storage_buffer=calls.append), 3, {})
    assert calls == [3]
    assert '_storage_textures' not in vars(renderer)


def test_real_field_node_visits_follow_visibility_and_motion_preferences(qtbot):
    """Visible native selection reaches the real CPU field; leave/stillness clears it."""
    from PySide6.QtWidgets import QWidget

    prefs.set_ambient_enabled(True)
    prefs.set_reduce_motion(False)
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(320, 240)
    backdrop = ambient.AmbientWidget(host, theme='data_art_impulse_lens', seed=7)
    backdrop.setGeometry(host.rect())
    node = QWidget(host)
    node.setGeometry(80, 60, 100, 80)
    host.show()
    try:
        qtbot.waitUntil(lambda: backdrop._should_run())
        serial = backdrop._art_input._node_visit[0]
        ambient._field_visit_node(node)
        assert backdrop._art_input._node_visit == (serial + 1, (129 / 320, 99 / 240))
        qtbot.waitUntil(lambda: backdrop._engine._node_visit is not None)
        ambient._field_visit_node(node, active=False)
        assert backdrop._art_input._node_visit[1] is None
        qtbot.waitUntil(lambda: backdrop._engine._node_visit is None)
        prefs.set_reduce_motion(True)
        ambient._field_visit_node(node)
        assert backdrop._art_input._node_visit[1] is None
        prefs.set_reduce_motion(False)
        node.hide()
        ambient._field_visit_node(node)
        assert backdrop._art_input._node_visit[1] is None
        node.show()
        prefs.set_ambient_enabled(False)
        ambient._field_visit_node(node)
        assert backdrop._art_input._node_visit[1] is None
    finally:
        backdrop.stop()
        host.close()


def test_shutdown_prevents_new_kernel_warmup_without_starting_compilers(monkeypatch):
    """An owned shutdown event rejects warmup before invoking either compiler."""
    stop = threading.Event()
    stop.set()
    monkeypatch.setattr(ambient, '_COMPILER_STOP', stop)

    def unexpected():
        raise AssertionError('shutdown must not start a compiler')

    monkeypatch.setattr(ambient, '_ready_packed_scatter', unexpected)
    monkeypatch.setattr(ambient, '_SATIN_COMPILER', types.SimpleNamespace(ready=unexpected))
    assert ambient._start_kernel_warmup_before_first_paint() is False


def test_kernel_warmup_wait_joins_only_owned_other_threads(monkeypatch, tmp_path):
    """An already primed cache allows bounded joining without joining the current thread."""
    done = threading.Event()
    worker = threading.Thread(target=done.set)
    worker.start()
    monkeypatch.setattr(ambient, '_COMPILER_THREADS', {threading.current_thread(), worker})
    monkeypatch.setattr(ambient, '_priming_allowed', lambda: True)
    monkeypatch.setattr(ambient, '_numba_cache_directory', lambda: tmp_path)
    marker = ambient._primed_marker(tmp_path)
    marker.write_text('owned cache marker')
    expected = (ambient._PACKED_SCATTER is not None and ambient._SUBPIXEL_SCATTER is not None
                and (ambient._SATIN_COMPILER.kernel is not None or ambient._SATIN_COMPILER.failed))
    assert ambient._finish_kernel_warmup_before_first_paint(timeout_s=.25) is expected
    assert done.is_set() and not worker.is_alive()


@pytest.mark.parametrize('platform', ['linux', 'darwin'])
def test_native_graphics_name_refusal_does_not_start_a_probe(monkeypatch, platform):
    """Unsupported OS or unavailable metadata refuses the known-driver graphics path."""
    from spacr import accelerator

    def unavailable():
        raise OSError('GPU name unavailable')

    monkeypatch.setattr(ambient, 'sys', types.SimpleNamespace(platform=platform))
    monkeypatch.setattr(accelerator, '_metal_gpu_name', unavailable)
    assert ambient._mac_radeon_graphics() is False


def test_field_visit_expires_after_real_engine_time_and_rejects_malformed_column(qtbot):
    """Actual selected-node delivery expires through engine time without adding a wave."""
    from PySide6.QtWidgets import QWidget

    prefs.set_ambient_enabled(True)
    prefs.set_reduce_motion(False)
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(240, 180)
    backdrop = ambient.AmbientWidget(host, theme='data_art_impulse_lens', seed=7)
    backdrop.setGeometry(host.rect())
    node = QWidget(host)
    node.setGeometry(20, 20, 80, 60)
    host.show()
    try:
        qtbot.waitUntil(lambda: backdrop._should_run())
        ambient._field_visit_node(node)
        with backdrop._engine_lock:
            backdrop._art_input._consume(backdrop._engine)
            engine = backdrop._engine
            assert engine._node_visit is not None
            engine.advance(7.0)
            assert engine._field_center()[2] == 0.0 and engine._node_visit is None
            before = list(engine._column_waves)
            engine._add_column_wave((.2, .8, .5, .5, .1))
            assert list(engine._column_waves) == before
    finally:
        backdrop.stop()
        host.close()


def test_held_probe_capacity_clears_real_retired_widget_samples(qtbot):
    """A fresh walk replaces historical tile samples with current background observations."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    qtbot.addWidget(host)
    backdrop = ambient.AmbientWidget(host, theme='blobs', seed=7)
    backdrop.resize(80, 60)
    held = backdrop._held_rows
    for _ in range(ambient._HOLD_PROBES_MAX + 1):
        field = QWidget(host)
        field.resize(1, 1)
        field.setAttribute(Qt.WA_StyledBackground, True)
        field.setStyleSheet('background:#123456;')
        held._painted_tiles(field)
    assert len(held._probes) > ambient._HOLD_PROBES_MAX
    historical = set(held._probes)
    assert all(key[0] == 'tiles' for key in historical)
    covered = held._covered()
    assert covered == ambient.QRegion(ambient.QRect(0, 0, ambient._HOLD_GRID, ambient._HOLD_GRID))
    assert historical.isdisjoint(held._probes)
    assert all(key[0] == 'background' for key in held._probes)
    assert all(sample[1]() is not None for sample in held._probes.values())
    backdrop.stop()


def test_empty_and_sparse_custom_paint_probes_cover_only_actual_ink(qtbot):
    """A custom container's empty tiles stay live and its painted tiles are held."""
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QRegion
    from PySide6.QtWidgets import QWidget

    class Ink(QWidget):
        def paintEvent(self, event):
            painter = QPainter(self)
            painter.fillRect(QRect(32, 0, 16, 16), QColor('#ff0000'))
            painter.end()

    host = QWidget()
    qtbot.addWidget(host)
    backdrop = ambient.AmbientWidget(host, theme='blobs', seed=7)
    field = Ink(host)
    field.resize(80, 32)
    field.setAttribute(ambient.Qt.WA_NoSystemBackground, True)
    region = backdrop._held_rows._painted_tiles(field)
    assert region.contains(ambient.QPoint(36, 4))
    assert not region.contains(ambient.QPoint(4, 4))
    field.resize(0, 0)
    assert backdrop._held_rows._painted_tiles(field) == QRegion()
    backdrop.stop()


def test_field_dispatch_skips_noninteractive_backdrops_and_discarded_native_widgets(qtbot):
    """Mixed background themes receive only their own supported input protocols."""
    import shiboken6
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    qtbot.addWidget(host)
    host.resize(240, 180)
    classic = ambient.AmbientWidget(host, theme='blobs', seed=7)
    node = QWidget(host)
    ambient._field_visit_node(node)
    ambient._field_growing_regions_for_widget(node, (ambient.QRect(-10, -10, 4, 4),))
    assert classic._art_input is None
    popup = QWidget(host)
    shiboken6.delete(popup)
    ambient._field_ripple_for_popup(popup)
    ambient.field_ripple_for_widget(popup)
    classic.stop()


def test_wallpaper_held_region_and_fresh_region_paint_their_own_real_pixels(qtbot):
    """A wallpaper-backed field uses the complete painter fallback with exact region boundaries."""
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QPaintEvent, QRegion
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    qtbot.addWidget(host)
    host.resize(120, 100)
    wallpaper = QImage(120, 100, QImage.Format_RGB32)
    wallpaper.fill(QColor('#102030'))
    backdrop = ambient.AmbientWidget(host, theme='blobs', backdrop=wallpaper, seed=7)
    backdrop.resize(120, 100)
    old, fresh = QImage(120, 100, QImage.Format_RGB32), QImage(120, 100, QImage.Format_RGB32)
    old.fill(QColor('#d02030'))
    fresh.fill(QColor('#1040d0'))
    held = backdrop._held_rows
    held.frame = old
    held.region = QRegion(QRect(24, 24, 48, 48))
    backdrop._last_frame = fresh
    backdrop._timer.start()
    references = []
    for frame in (old, fresh):
        reference = QImage(120, 100, QImage.Format_ARGB32_Premultiplied)
        reference.fill(0)
        renderer = QPainter(reference)
        backdrop._paint_base(renderer, backdrop.rect())
        backdrop._engine.blit(renderer, frame, 120, 100)
        renderer.end()
        references.append(reference)
    expected = references[1].copy()
    renderer = QPainter(expected)
    renderer.setClipRegion(held.region)
    renderer.drawImage(0, 0, references[0])
    renderer.end()
    canvas = QImage(120, 100, QImage.Format_ARGB32_Premultiplied)
    canvas.fill(0)
    painter = QPainter(canvas)
    try:
        assert backdrop._paint_held(painter, QPaintEvent(backdrop.rect()), 120, 100)
    finally:
        painter.end()
        backdrop.stop()
    assert canvas == expected
    assert canvas.pixelColor(40, 40) == references[0].pixelColor(40, 40)
    assert canvas.pixelColor(100, 80) == references[1].pixelColor(100, 80)


def test_no_fresh_frame_cannot_claim_a_complete_held_paint(qtbot):
    """A held input alone is insufficient while the next field frame is unavailable."""
    from PySide6.QtGui import QPaintEvent, QRegion
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    qtbot.addWidget(host)
    backdrop = ambient.AmbientWidget(host, theme='blobs', seed=7)
    backdrop.resize(80, 60)
    held = backdrop._held_rows
    held.frame = QImage(80, 60, QImage.Format_RGB32)
    held.frame.fill(QColor('#123456'))
    held.region = QRegion(backdrop.rect())
    assert backdrop._last_frame is None
    backdrop._timer.start()
    canvas = QImage(80, 60, QImage.Format_ARGB32_Premultiplied)
    canvas.fill(0)
    painter = QPainter(canvas)
    try:
        assert not backdrop._paint_held(painter, QPaintEvent(backdrop.rect()), 80, 60)
    finally:
        painter.end()
        backdrop.stop()
    assert canvas.pixelColor(40, 30).alpha() == 0


def test_unknown_power_supply_entries_do_not_override_a_real_discharging_battery(tmp_path):
    """Only system battery and mains records take part in the local sysfs decision."""
    for name, values in [('unknown', {'type': 'USB_PD'}),
                         ('battery', {'type': 'Battery', 'status': 'Discharging'})]:
        folder = tmp_path / name
        folder.mkdir()
        for key, value in values.items():
            (folder / key).write_text(value)
    assert ambient._linux_on_battery(tmp_path) is True


def test_parentless_field_and_zero_sized_background_have_no_covered_area(qtbot):
    """Unembedded fields and empty layout containers cannot hold unrelated pixels."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget

    backdrop = ambient.AmbientWidget(theme='blobs', seed=7)
    qtbot.addWidget(backdrop)
    assert backdrop._held_rows._covered().isEmpty()
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.setAttribute(Qt.WA_StyledBackground, True)
    widget.resize(0, 0)
    assert backdrop._held_rows._paints_a_background(widget) is False
    backdrop.stop()


def test_held_classic_field_updates_only_live_area_and_can_be_fully_covered(qtbot, monkeypatch, request):
    """Real settling and explicit occluded-field capture drive exact tick dispatch regions.

    The card paints through ``autoFillBackground``, which Qt switches off under
    a style sheet with a background. The session ``qt_theme_applied`` fixture
    leaves spaCR's sheet on the shared QApplication for every later test in
    the worker, where this card would rightly cover nothing; the case runs
    without an application sheet and puts back the one it found.
    """
    from PySide6.QtWidgets import QApplication, QWidget

    app = QApplication.instance()
    found_sheet = app.styleSheet()
    request.addfinalizer(lambda: app.setStyleSheet(found_sheet))
    app.setStyleSheet('')
    prefs.set_ambient_enabled(True)
    prefs.set_reduce_motion(False)
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(240, 180)
    backdrop = ambient.AmbientWidget(host, theme='blobs', seed=7)
    backdrop.setGeometry(host.rect())
    card = QWidget(host)
    card.setGeometry(20, 20, 80, 60)
    card.setAutoFillBackground(True)
    host.show()
    updates = []
    original = backdrop.update

    def update(*args):
        updates.append(args)
        original(*args)

    monkeypatch.setattr(backdrop, 'update', update)
    try:
        qtbot.waitUntil(lambda: backdrop._held_rows.frame is not None, timeout=5000)
        held = backdrop._held_rows
        assert not held.live.isEmpty()
        updates.clear()
        backdrop._on_tick()
        assert len(updates) == 1 and updates[0] == (held.live,)
        from spacr.qt.gui_scale import _resize_actual
        card.move(0, 0)
        _resize_actual(card, host.size())
        assert card.geometry() == host.rect() == backdrop.rect()
        held.stir()
        qtbot.waitUntil(lambda: held.arming and held.live.isEmpty(), timeout=5000)
        backdrop.grab()
        assert held.frame is not None and held.live.isEmpty()
        updates.clear()
        backdrop._on_tick()
        assert updates == []
    finally:
        backdrop.stop()
        host.close()


def test_gl_table_adapter_rewrites_each_distinct_lookup_without_changing_indices():
    """Repeated same-table lookups retain their own addresses through source adaptation."""
    captured = {}
    renderer = ambient._FlowPointGraphics.__new__(ambient._FlowPointGraphics)
    renderer._context = types.SimpleNamespace(version_code=410,
        program=lambda **sources: captured.update(sources))
    renderer._make_program(fragment_shader=(
        '#version 430\nlayout(std430,binding=2) readonly buffer P { uint palette[]; };\n'
        'void main() { uint a = palette[3] + palette[9]; }'))
    source = captured['fragment_shader']
    assert 'ivec2(int(3), 0)' in source and 'ivec2(int(9), 0)' in source
    assert source.count('texelFetch(spacr_palette,') == 2


def test_unrecognized_power_supply_kind_is_ignored_before_battery_policy(tmp_path):
    """Vendor-specific supply records do not masquerade as mains power."""
    for name, values in [('unknown', {'type': 'VendorPrivate'}),
                         ('battery', {'type': 'Battery', 'status': 'Discharging'})]:
        folder = tmp_path / name
        folder.mkdir()
        for key, value in values.items():
            (folder / key).write_text(value)
    assert ambient._linux_on_battery(tmp_path) is True


def test_disallowed_external_priming_still_joins_an_owned_resident_worker(monkeypatch):
    """The no-helper policy bypasses disk-marker lookup while retiring a real worker."""
    done = threading.Event()
    worker = threading.Thread(target=done.set)
    worker.start()
    monkeypatch.setattr(ambient, '_COMPILER_THREADS', {worker})
    monkeypatch.setattr(ambient, '_priming_allowed', lambda: False)

    def unexpected():
        raise AssertionError('disabled external priming must not query its cache marker')

    monkeypatch.setattr(ambient, '_numba_cache_directory', unexpected)
    expected = (ambient._PACKED_SCATTER is not None and ambient._SUBPIXEL_SCATTER is not None
                and (ambient._SATIN_COMPILER.kernel is not None or ambient._SATIN_COMPILER.failed))
    assert ambient._finish_kernel_warmup_before_first_paint(timeout_s=.25) is expected
    assert done.is_set() and not worker.is_alive()


def test_native_line_separator_with_a_child_is_not_treated_as_a_hollow_ring(qtbot):
    """Line-shaped QFrames cover their own native paint rather than an invented ring."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QFrame, QWidget

    host = QWidget()
    qtbot.addWidget(host)
    backdrop = ambient.AmbientWidget(host, theme='blobs', seed=7)
    line = QFrame(host)
    line.setFrameShape(QFrame.HLine)
    line.setLineWidth(2)
    line.setAttribute(Qt.WA_NoSystemBackground, True)
    QWidget(line)
    assert line.frameWidth() > 0
    assert backdrop._held_rows._own_paint(line) is None
    backdrop.stop()


@pytest.mark.parametrize('route', ['popup', 'widget'])
def test_retired_native_owner_during_ripple_lookup_is_ignored(qtbot, monkeypatch, route):
    """A real owner retiring after lookup cannot send feedback through an invalid wrapper."""
    import shiboken6
    from PySide6.QtWidgets import QWidget

    owner = QWidget()
    qtbot.addWidget(owner)
    widget = QWidget(owner)
    qtbot.addWidget(widget)
    lookup = widget.parentWidget if route == 'popup' else widget.window

    def retired():
        previous = lookup()
        widget.setParent(None)
        shiboken6.delete(owner)
        assert not shiboken6.isValid(previous)
        return previous

    monkeypatch.setattr(widget, 'parentWidget' if route == 'popup' else 'window', retired)
    if route == 'popup':
        ambient._field_ripple_for_popup(widget)
    else:
        ambient.field_ripple_for_widget(widget)
    assert shiboken6.isValid(widget) and not shiboken6.isValid(owner)


def test_growing_field_regions_clip_outside_bounds_even_when_holding_is_disabled(qtbot):
    """Geometry still reaches the real field while optional row holding is switched off."""
    from PySide6.QtCore import QRect
    from PySide6.QtWidgets import QWidget

    prefs.set_ambient_enabled(True)
    prefs.set_reduce_motion(False)
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(240, 180)
    backdrop = ambient.AmbientWidget(host, theme='data_art_impulse_lens', seed=7)
    backdrop.setGeometry(host.rect())
    host.show()
    try:
        qtbot.waitUntil(lambda: backdrop._should_run())
        backdrop._held_rows.enabled = False
        inside = QRect(8, 8, 20, 30)
        outside = QRect(-100, -100, 10, 10)
        ambient._field_growing_regions_for_widget(host, (outside, inside), invalidate=True)
        width, height = backdrop.width(), backdrop.height()
        assert backdrop._art_input._growing_regions == ((8 / width, 8 / height,
                                                         28 / width, 38 / height),)
        assert backdrop._held_rows.frame is None
    finally:
        backdrop.stop()
        host.close()


def test_rewinding_the_animation_clock_defers_a_real_selected_node_visit(qtbot):
    """A replay rewind renders the wandering field until its recorded visit begins."""
    import math
    from PySide6.QtWidgets import QWidget

    prefs.set_ambient_enabled(True)
    prefs.set_reduce_motion(False)
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(320, 240)
    backdrop = ambient.AmbientWidget(host, theme='data_art_impulse_lens', seed=7)
    backdrop.setGeometry(host.rect())
    node = QWidget(host)
    node.setGeometry(280, 12, 24, 24)
    host.show()
    try:
        qtbot.waitUntil(lambda: backdrop._should_run() and node.isVisible())
        with backdrop._engine_lock:
            backdrop.set_time(10.0)
            ambient._field_visit_node(node)
            engine = backdrop._engine
            started, point = engine._node_visit
            assert started == 10.0
            backdrop.set_time(13.0)
            assert engine._field_center() == pytest.approx((*point, 1.0))
            backdrop.set_time(0.0)
            expected = (0.5 + 0.20 * math.sin(engine._anchors[0][0]),
                        0.5 + 0.18 * math.cos(engine._anchors[0][1]), 0.0)
            assert engine._field_center() == pytest.approx(expected)
            assert engine._node_visit == (started, point)
            frame = backdrop.grab().toImage()
            assert not frame.isNull()
            assert frame.deviceIndependentSize().toSize() == backdrop.size()
            assert all(math.isfinite(value) for value in engine._field_center())
    finally:
        backdrop.stop()
        host.close()

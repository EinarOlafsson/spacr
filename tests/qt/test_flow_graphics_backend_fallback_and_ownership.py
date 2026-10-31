"""N679/N684: optional GPU point renderer keeps a usable CPU path and owns its context."""
import threading
import time
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QImage

from spacr.qt import preferences
from spacr.qt.widgets import ambient


@pytest.fixture(autouse=True)
def no_policy(monkeypatch):
    monkeypatch.setattr(ambient, '_RESOURCE_POLICY', None)


def engine():
    return ambient.make_engine('data_art_impulse_lens', 'spacr', '#14171b',
                               seed=311, blur=0, resolution=1, density=.1)


def wait_for_frames(producer, count=2):
    deadline = time.monotonic() + 10
    while producer.frames_shaded < count and time.monotonic() < deadline:
        time.sleep(.01)
    assert producer.frames_shaded >= count


@pytest.mark.parametrize('mode', ['cpu', 'auto'])
def test_small_material_never_probes_or_constructs_graphics(mode, monkeypatch, qapp):
    candidate = engine()
    candidate._graphics_backend = mode
    expected = candidate.shade(320, 200).copy()

    def forbidden():
        pytest.fail('small/CPU material must not probe a graphics driver')

    monkeypatch.setattr(ambient, '_flow_graphics_preflight', forbidden)
    monkeypatch.setattr(ambient, '_flow_compute_idle', forbidden)
    producer = ambient._FrameProducer(candidate, threading.Lock(), 30, (320, 200))
    producer.start()
    try:
        wait_for_frames(producer)
        assert producer.latest() == expected
    finally:
        producer.stop()
    assert not producer.is_alive()
    assert candidate._point_graphics is None


@pytest.mark.parametrize('idle,probe', [(False, True), (True, False)])
def test_busy_or_unavailable_graphics_preserves_actual_cpu_pixels(
        idle, probe, monkeypatch, qapp):
    candidate = engine()
    expected = candidate.shade(320, 200).copy()
    candidate._graphics_backend = 'gpu'
    monkeypatch.setattr(ambient, '_flow_compute_idle', lambda: idle)
    monkeypatch.setattr(ambient, '_flow_graphics_preflight', lambda stop_event=None: probe)

    def forbidden():
        pytest.fail('refused graphics must not construct a context')

    monkeypatch.setattr(ambient, '_FlowPointGraphics', forbidden)
    producer = ambient._FrameProducer(candidate, threading.Lock(), 30, (320, 200))
    producer.start()
    try:
        wait_for_frames(producer)
        assert producer.latest() == expected
    finally:
        producer.stop()
    assert not producer.is_alive()
    assert candidate._point_graphics is None


def test_context_constructor_failure_preserves_cpu_animation(monkeypatch, qapp):
    candidate = engine()
    expected = candidate.shade(320, 200).copy()
    candidate._graphics_backend = 'gpu'
    monkeypatch.setattr(ambient, '_flow_compute_idle', lambda: True)
    monkeypatch.setattr(ambient, '_flow_graphics_preflight', lambda stop_event=None: True)
    attempts = []

    def broken():
        attempts.append(threading.get_ident())
        raise RuntimeError('unsupported test context')

    monkeypatch.setattr(ambient, '_FlowPointGraphics', broken)
    producer = ambient._FrameProducer(candidate, threading.Lock(), 30, (320, 200))
    producer.start()
    try:
        wait_for_frames(producer)
        assert producer.latest() == expected
    finally:
        producer.stop()
    assert len(attempts) == 1
    assert attempts[0] != threading.get_ident()
    assert candidate._graphics_failed
    assert not producer.is_alive()


def test_renderer_construction_draw_and_release_share_producer_thread(monkeypatch, qapp):
    events = []

    class Renderer:
        def __init__(self):
            self._owner = threading.get_ident()
            events.append(('create', self._owner))

        def _draw(self, width, height, *args):
            events.append(('draw', threading.get_ident()))
            image = QImage(width, height, QImage.Format_RGB32)
            image.fill(QColor('white'))
            return image

        _draw_subpixel = _draw

        def _close(self):
            events.append(('close', threading.get_ident()))

    monkeypatch.setattr(ambient, '_flow_compute_idle', lambda: True)
    monkeypatch.setattr(ambient, '_flow_graphics_preflight', lambda stop_event=None: True)
    monkeypatch.setattr(ambient, '_FlowPointGraphics', Renderer)
    candidate = engine()
    candidate._graphics_backend = 'gpu'
    producer = ambient._FrameProducer(candidate, threading.Lock(), 30, (320, 200))
    producer.start()
    try:
        wait_for_frames(producer)
        first = producer.latest().copy()
        wait_for_frames(producer, producer.frames_shaded + 2)
        assert first == producer.latest()
    finally:
        producer.stop()
    assert events[0][0] == 'create' and events[-1][0] == 'close'
    assert ('draw', events[0][1]) in events and not candidate._graphics_failed
    assert {owner for _, owner in events} == {events[0][1]}
    assert events[0][1] != threading.get_ident()
    assert candidate._point_graphics is None
    assert not producer.is_alive()


def _mock_compute_process_metadata(monkeypatch, metadata):
    from pathlib import Path

    def proc_path(location):
        prefix, proc, pid, leaf = str(location).split("/")
        assert prefix == "" and proc == "proc" and leaf in ("exe", "cmdline")
        entry = metadata.get(int(pid), {})

        def read(key):
            value = entry.get(key)
            if value is None:
                raise PermissionError(location)
            return value

        return SimpleNamespace(readlink=lambda: Path(read("exe")),
                               read_bytes=lambda: read("cmdline"))

    def portable_process(pid):
        entry = metadata.get(pid, {})

        def read(key):
            value = entry.get(key)
            if value is None:
                raise PermissionError(pid)
            return value

        return SimpleNamespace(exe=lambda: read("portable_exe"),
                               cmdline=lambda: read("portable_cmdline"))

    monkeypatch.setattr(ambient, "Path", proc_path)
    monkeypatch.setitem(__import__("sys").modules, "psutil",
                        SimpleNamespace(Process=portable_process, Error=OSError))


@pytest.fixture
def compute_query(monkeypatch):
    def query(pids, metadata):
        events = []
        module = SimpleNamespace(
            nvmlInit=lambda: events.append("init"),
            nvmlDeviceGetCount=lambda: 1,
            nvmlDeviceGetHandleByIndex=lambda index: index,
            nvmlDeviceGetComputeRunningProcesses=lambda handle: [
                SimpleNamespace(pid=pid) for pid in pids],
            nvmlShutdown=lambda: events.append("shutdown"))
        monkeypatch.setitem(__import__("sys").modules, "pynvml", module)
        _mock_compute_process_metadata(monkeypatch, metadata)
        idle = ambient._flow_compute_idle()
        assert events == ["init", "shutdown"]
        return idle

    return query


@pytest.mark.parametrize('processes', [[], [SimpleNamespace(pid=123)]])
def test_compute_process_monitor_refuses_busy_device_and_shuts_down(
        processes, monkeypatch):
    _mock_compute_process_metadata(monkeypatch, {})
    events = []
    module = SimpleNamespace(
        nvmlInit=lambda: events.append('init'),
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetComputeRunningProcesses=lambda handle: processes,
        nvmlShutdown=lambda: events.append('shutdown'))
    monkeypatch.setitem(__import__('sys').modules, 'pynvml', module)
    assert ambient._flow_compute_idle() is (not bool(processes))
    assert events == ['init', 'shutdown']


def test_compute_monitor_ignores_this_process_and_missing_nvml(monkeypatch):
    import os
    import sys

    module = SimpleNamespace(
        nvmlInit=lambda: None, nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetComputeRunningProcesses=lambda handle: [SimpleNamespace(pid=os.getpid())],
        nvmlShutdown=lambda: None)
    monkeypatch.setitem(sys.modules, 'pynvml', module)
    assert ambient._flow_compute_idle() is True

    def broken():
        raise RuntimeError('no driver')

    monkeypatch.setitem(sys.modules, 'pynvml', SimpleNamespace(nvmlInit=broken))
    assert ambient._flow_compute_idle() is True
    monkeypatch.setitem(sys.modules, 'pynvml', None)
    assert ambient._flow_compute_idle() is True


def test_missing_moderngl_refuses_without_a_child_process(monkeypatch):
    import importlib.util
    import subprocess

    monkeypatch.setattr(ambient, '_FLOW_GRAPHICS_PROBE', None)
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: None)
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('no child'))
    assert ambient._flow_graphics_preflight() is False


@pytest.mark.parametrize('platform,backend,required', [
    pytest.param('linux', 'egl', 430, id='linux-egl'),
    pytest.param('win32', None, 430, id='win32-None'),
    pytest.param('darwin', None, 410, id='darwin-None'),
])
def test_context_options_per_platform(platform, backend, required, monkeypatch):
    monkeypatch.setattr(ambient.sys, 'platform', platform)
    options = ambient._graphics_context_options()
    assert options.get('backend') == backend
    assert options['standalone'] and options['require'] == required


def test_policy_block_releases_and_refuses_graphics(monkeypatch, qapp):
    candidate = engine()
    candidate._graphics_backend = 'gpu'
    monkeypatch.setattr(ambient, '_decorative_gpu_blocked', lambda: True)
    monkeypatch.setattr(ambient, '_flow_graphics_preflight',
                        lambda stop_event=None: pytest.fail('blocked graphics must not probe'))
    released = []
    candidate._point_graphics = SimpleNamespace(
        _owner=threading.get_ident(), _close=lambda: released.append(True))
    monkeypatch.setattr(threading, 'current_thread',
                        lambda: SimpleNamespace(name='spacr-ambient-shade'))
    assert candidate._graphics_point_image(320, 200, [], [], [], None, False) is None
    assert released == [True] and candidate._point_graphics is None


def test_dynamic_off_skips_the_external_process_check(monkeypatch, qapp):
    candidate = engine()
    candidate._graphics_backend = 'gpu'
    monkeypatch.setattr(ambient, '_dynamic_animation_on', lambda: False)
    monkeypatch.setattr(ambient, '_flow_compute_idle', lambda: pytest.fail('policy off'))
    monkeypatch.setattr(ambient, '_flow_graphics_preflight', lambda stop_event=None: True)
    made = []

    class Renderer:
        def __init__(self):
            self._owner = threading.get_ident()
            made.append(self)

        def _draw(self, width, height, *args):
            return QImage(width, height, QImage.Format_RGB32)

        def _close(self):
            pass

    monkeypatch.setattr(ambient, '_FlowPointGraphics', Renderer)
    monkeypatch.setattr(threading, 'current_thread',
                        lambda: SimpleNamespace(name='spacr-ambient-shade'))
    assert candidate._graphics_point_image(320, 200, [], [], [], None, False) is not None
    assert made


def test_automatic_follows_the_measured_cost_per_theme_and_size(monkeypatch, qapp):
    """N684: Automatic picks the GPU only where the workstation audit measured it faster."""
    monkeypatch.setattr(ambient, '_flow_graphics_preflight', lambda stop_event=None: False)
    monkeypatch.setattr(threading, 'current_thread',
                        lambda: SimpleNamespace(name='spacr-ambient-shade'))

    def asks(theme, width, height, density):
        candidate = ambient.make_engine(theme, 'spacr', '#14171b', seed=311,
                                        density=density)
        candidate._graphics_backend = 'auto'
        candidate._graphics_point_image(width, height, [0] * 200000, [], [], None, False)
        return candidate._graphics_failed

    assert not asks('data_art_genetic_advection', 1920, 1080, 1.0)
    assert not asks('data_art_genetic_advection', 3840, 2160, 1.0)
    # Round-2 audit (684_animation_gpu_audit_r2_2026-10-10): point_atlas at 4K
    # full is now faster on the GPU (3.3 vs 5.1 ms) and impulse_lens at 4K full
    # faster on the CPU (5.0 vs 9.1 ms); the winners follow the receipt.
    assert asks('data_art_point_atlas', 3840, 2160, 1.0)
    # 1080p full is a near tie. Round 5 (684_animation_gpu_audit_r5_2026-10-11)
    # measured 2.24 CPU vs 1.98 GPU ms, just past the 0.9 gain, so Automatic
    # takes the GPU there (round 2: 1.91 vs 1.81, CPU).
    assert asks('data_art_point_atlas', 1920, 1080, 1.0)
    assert not asks('data_art_impulse_lens', 3840, 2160, 1.0)
    assert not asks('data_art_spaceout_field', 1920, 1080, 1.0)
    assert asks('data_art_spaceout_field', 3840, 2160, 1.0)


def test_unmeasured_themes_keep_their_documented_automatic_rule(monkeypatch, qapp):
    monkeypatch.setitem(ambient._GRAPHICS_COST, 'data_art_impulse_lens', ())
    candidate = engine()
    assert not candidate._graphics_automatic(1920, 1080, 200000)
    assert not candidate._graphics_automatic(3840, 2160, 1000)
    assert candidate._graphics_automatic(3840, 2160, 200000)


def test_failed_isolated_probe_is_cached_without_constructing_in_parent(monkeypatch):
    import subprocess
    import sys

    monkeypatch.setattr(ambient.sys, "platform", "linux")
    monkeypatch.delenv('SPACR_NO_GL', raising=False)
    calls = []
    real_popen = subprocess.Popen

    def failed(command, **kwargs):
        """Return a real CPU child's failed probe result without creating GL."""
        calls.append((command, kwargs))
        return real_popen([sys.executable, '-c', 'raise SystemExit(11)'], **kwargs)

    import importlib.util

    monkeypatch.setattr(ambient, '_FLOW_GRAPHICS_PROBE', None)
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: object())
    monkeypatch.setattr(subprocess, 'Popen', failed)
    assert ambient._flow_graphics_preflight() is False
    assert ambient._flow_graphics_preflight() is False
    assert len(calls) == 1
    assert calls[0][1] == {'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE, 'text': True}


def test_probe_deadline_reaps_a_real_cpu_child_and_caches_its_refusal(monkeypatch):
    """A probe that cannot finish within its existing deadline owns no orphan.

    moderngl is an optional dependency; without it the probe correctly never
    starts a child. Report it present, so the deadline path runs whether or not
    the test machine has it installed. The child never imports it.
    """
    import importlib.util
    import subprocess
    import sys

    monkeypatch.setattr(ambient.sys, 'platform', 'linux')
    monkeypatch.delenv('SPACR_NO_GL', raising=False)
    monkeypatch.setattr(ambient, '_FLOW_GRAPHICS_PROBE', None)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec',
                        lambda name, *args, **kwargs: object() if name == 'moderngl'
                        else real_find_spec(name, *args, **kwargs))
    real_popen = subprocess.Popen
    children = []

    def stalled_probe(command, **kwargs):
        """Exercise the timeout with Python sleeping, without invoking GL."""
        assert command[0] == sys.executable and 'import moderngl;' in command[-1]
        child = real_popen([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, 'Popen', stalled_probe)
    try:
        assert ambient._flow_graphics_preflight() is False
        assert len(children) == 1 and children[0].poll() is not None
        assert children[0].returncode != 0
        assert ambient._FLOW_GRAPHICS_PROBE is False
        assert ambient._flow_graphics_preflight() is False
        assert len(children) == 1
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()


def test_flow_renderer_preference_has_explicit_cpu_and_validates_storage(
        tmp_path, monkeypatch, qapp):
    store = QSettings(str(tmp_path / 'flow.ini'), QSettings.IniFormat)
    monkeypatch.setattr(preferences, '_settings', lambda: store)
    assert preferences._flow_graphics_backend() == 'gpu'
    preferences._set_flow_graphics_backend('auto')
    assert preferences._flow_graphics_backend() == 'auto'
    preferences._set_flow_graphics_backend('cpu')
    assert preferences._flow_graphics_backend() == 'cpu'
    preferences._set_flow_graphics_backend('invalid')
    assert preferences._flow_graphics_backend() == 'gpu'


def test_macos_never_creates_a_gl_context_to_probe(monkeypatch):
    """Three WindowServer watchdog kills on an Intel Iris Mac (macOS 26, 2026-10-19) each followed a spaCR
    start that probed for OpenGL 4.3, which macOS never offers. The probe is not run there at all."""
    import subprocess

    from spacr.qt.widgets import ambient

    monkeypatch.setattr(ambient.sys, "platform", "darwin")
    monkeypatch.setattr(ambient, "_FLOW_GRAPHICS_PROBE", None)
    monkeypatch.delenv("SPACR_NO_GL", raising=False)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("the GL probe child was started"))
    assert ambient._flow_graphics_preflight() is False


@pytest.mark.parametrize("executable", [
    "chrome", "chromium", "chromium-browser", "google-chrome",
    "google-chrome-stable", "chrome (deleted)",
])
def test_compute_monitor_excludes_browser_executables(compute_query, executable):
    assert compute_query([424242], {424242: {"exe": "/opt/browser/" + executable}}) is True


@pytest.mark.parametrize("arguments,idle", [
    (b"/opt/browser/helper\0--type=gpu-process\0", True),
    (b"/usr/bin/python\0--type=gpu-process-extra\0", False),
    (b"/usr/bin/python\0--file=chrome\0", False),
])
def test_compute_monitor_requires_the_exact_graphics_argument(
        compute_query, arguments, idle):
    assert compute_query([424242], {424242: {"exe": "/usr/bin/helper",
                                          "cmdline": arguments}}) is idle


@pytest.mark.parametrize("processes,metadata", [
    ([424242, 424243], {424242: {"exe": "/opt/browser/chrome"},
                       424243: {"exe": "/usr/bin/python", "cmdline": b"python\0train.py\0"}}),
    ([424242], {}),
    ([None], {}),
])
def test_compute_monitor_still_refuses_real_or_unknown_compute(
        compute_query, processes, metadata):
    assert compute_query(processes, metadata) is False


@pytest.mark.parametrize("metadata,idle", [
    ({"portable_exe": r"C:\Program Files\Google\Chrome\Application\chrome.exe"}, True),
    ({"portable_exe": r"C:\Program Files\Google\Chrome\Application\CHROME.EXE"}, True),
    ({"portable_exe": "/opt/browser/chromium (deleted)"}, True),
    ({"portable_exe": r"C:\Browser\helper.exe",
      "portable_cmdline": ["helper.exe", "--type=gpu-process"]}, True),
    ({"portable_cmdline": ["helper.exe", "--type=gpu-process"]}, True),
    ({"portable_exe": r"C:\Python\python.exe",
      "portable_cmdline": ["python.exe", "train.py"]}, False),
    ({"portable_exe": r"C:\Browser\chrome.exe.backup",
      "portable_cmdline": ["helper.exe", "--type=gpu-process-extra"]}, False),
    ({"portable_exe": r"C:\Python\python.exe",
      "portable_cmdline": ["python.exe", "--file=chrome"]}, False),
    ({}, False),
])
def test_compute_monitor_uses_portable_metadata_without_proc(
        compute_query, metadata, idle):
    assert compute_query([424242], {424242: metadata}) is idle


def test_portable_browser_does_not_hide_a_real_compute_process(compute_query):
    assert compute_query([424242, 424243], {
        424242: {"portable_exe": r"C:\Browser\chrome.exe"},
        424243: {"portable_exe": r"C:\Python\python.exe",
                 "portable_cmdline": ["python.exe", "train.py"]},
    }) is False

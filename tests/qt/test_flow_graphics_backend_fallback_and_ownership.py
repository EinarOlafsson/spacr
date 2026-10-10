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
    monkeypatch.setattr(ambient, '_flow_graphics_preflight', lambda: probe)

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
    monkeypatch.setattr(ambient, '_flow_graphics_preflight', lambda: True)
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

        def _close(self):
            events.append(('close', threading.get_ident()))

    monkeypatch.setattr(ambient, '_flow_compute_idle', lambda: True)
    monkeypatch.setattr(ambient, '_flow_graphics_preflight', lambda: True)
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
    assert {owner for _, owner in events} == {events[0][1]}
    assert events[0][1] != threading.get_ident()
    assert candidate._point_graphics is None
    assert not producer.is_alive()


@pytest.mark.parametrize('processes', [[], [SimpleNamespace(pid=123)]])
def test_compute_process_monitor_refuses_busy_device_and_shuts_down(
        processes, monkeypatch):
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
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('no child'))
    assert ambient._flow_graphics_preflight() is False


@pytest.mark.parametrize('platform,backend', [('linux', 'egl'), ('win32', None), ('darwin', None)])
def test_context_options_per_platform(platform, backend, monkeypatch):
    monkeypatch.setattr(ambient.sys, 'platform', platform)
    options = ambient._graphics_context_options()
    assert options.get('backend') == backend
    assert options['standalone'] and options['require'] == 430


def test_policy_block_releases_and_refuses_graphics(monkeypatch, qapp):
    candidate = engine()
    candidate._graphics_backend = 'gpu'
    monkeypatch.setattr(ambient, '_decorative_gpu_blocked', lambda: True)
    monkeypatch.setattr(ambient, '_flow_graphics_preflight',
                        lambda: pytest.fail('blocked graphics must not probe'))
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
    monkeypatch.setattr(ambient, '_flow_graphics_preflight', lambda: True)
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


def test_automatic_uses_gpu_only_for_dense_4k_frames(monkeypatch, qapp):
    candidate = engine()
    candidate._graphics_backend = 'auto'
    monkeypatch.setattr(ambient, '_flow_graphics_preflight', lambda: False)
    monkeypatch.setattr(threading, 'current_thread',
                        lambda: SimpleNamespace(name='spacr-ambient-shade'))
    assert candidate._graphics_point_image(1920, 1080, [0] * 200000, [], [], None, False) is None
    assert not candidate._graphics_failed
    candidate._graphics_point_image(3840, 2160, [0] * 200000, [], [], None, False)
    assert candidate._graphics_failed


def test_failed_isolated_probe_is_cached_without_constructing_in_parent(monkeypatch):
    import subprocess

    calls = []

    def failed(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=-11, stdout='')

    import importlib.util

    monkeypatch.setattr(ambient, '_FLOW_GRAPHICS_PROBE', None)
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: object())
    monkeypatch.setattr(subprocess, 'run', failed)
    assert ambient._flow_graphics_preflight() is False
    assert ambient._flow_graphics_preflight() is False
    assert len(calls) == 1
    assert calls[0][1]['timeout'] == 5


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

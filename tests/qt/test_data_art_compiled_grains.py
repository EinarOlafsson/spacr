"""Exact round-grain pixels with bounded background CPU compilation."""

import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QTimer

from spacr.qt.widgets import ambient


@pytest.fixture(autouse=True)
def isolated_compiler(monkeypatch):
    for thread in threading.enumerate():
        if thread.name == 'spacr-grain-compile':
            thread.join(10)
            assert not thread.is_alive()
    monkeypatch.setattr(ambient, '_PACKED_SCATTER', None)
    monkeypatch.setattr(ambient, '_PACKED_SCATTER_STARTED', False)
    monkeypatch.setattr(ambient, '_PACKED_SCATTER_FAILED', False)
    monkeypatch.setattr(ambient, '_PACKED_SCATTER_LOCK', threading.Lock())
    yield
    for thread in threading.enumerate():
        if thread.name == 'spacr-grain-compile':
            thread.join(10)
            assert not thread.is_alive()


@pytest.fixture(scope='module')
def compiled_kernel():
    from numba import njit

    kernel = njit(nogil=True, cache=False)(ambient._scatter_packed_grains)
    flat = np.zeros(1, dtype=np.uint32)
    coordinates = np.zeros(1, dtype=np.int32)
    intensity = np.zeros(1, dtype=np.uint8)
    lookup = np.arange(256, dtype=np.uint32)
    kernel(flat, coordinates, coordinates, intensity, lookup, lookup, lookup, 1, 1, True)
    assert kernel.nopython_signatures
    return kernel


def _engine(backdrop='#101418'):
    return ambient.make_engine('data_art_point_atlas', 'spacr', backdrop, seed=42,
                               blur=0, resolution=2, density=1)


def _pixels(image):
    return image.bits().tobytes()


@pytest.mark.parametrize('backdrop', ('#101418', '#fafafa'))
@pytest.mark.parametrize('colors', (('#ff0000', '#00ffff'), ('#010101', '#fefefe')))
@pytest.mark.parametrize('spread', (False, True))
def test_all_levels_duplicate_and_border_pixels_match_numpy(
        compiled_kernel, monkeypatch, backdrop, colors, spread):
    engine = _engine(backdrop)
    engine.set_colors(colors)
    x = np.tile(np.arange(-2, 19), 300)
    y = np.repeat(np.arange(-2, 13), 420)
    values = (np.arange(x.size) % 256) / 255
    monkeypatch.setattr(ambient, '_PACKED_SCATTER_FAILED', True)
    expected = _pixels(engine._point_material(17, 11, x, y, values, spread))
    monkeypatch.setattr(ambient, '_PACKED_SCATTER', compiled_kernel)
    actual = _pixels(engine._point_material(17, 11, x, y, values, spread))
    assert actual == expected


@pytest.mark.parametrize('dark', (False, True))
def test_packed_order_duplicates_match_ufunc_with_every_intensity(compiled_kernel, dark):
    rng = np.random.default_rng(82)
    px = np.tile(np.array([0, 1, 5, 6], dtype=np.int32), 256)
    py = np.tile(np.array([0, 4, 0, 4], dtype=np.int32), 256)
    intensity = np.repeat(np.arange(256, dtype=np.uint8), 4)
    lookup = rng.integers(0, 0xFFFFFF, 256, dtype=np.uint32) | np.uint32(0xFF000000)
    axis = lookup[np.rint(np.arange(256) * .68).astype(np.uint8)]
    diagonal = lookup[np.rint(np.arange(256) * .24).astype(np.uint8)]
    expected = np.full(35, lookup[0], dtype=np.uint32)
    combine = np.maximum.at if dark else np.minimum.at
    for sy, sx in ((0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)):
        valid = (px + sx >= 0) & (px + sx < 7) & (py + sy >= 0) & (py + sy < 5)
        palette = lookup if sy == sx == 0 else diagonal if sy and sx else axis
        combine(expected, (py[valid] + sy) * 7 + px[valid] + sx, palette[intensity[valid]])
    actual = np.full(35, lookup[0], dtype=np.uint32)
    compiled_kernel(actual, px, py, intensity, lookup, axis, diagonal, 7, 5, dark)
    np.testing.assert_array_equal(actual, expected)
    python = np.full(35, lookup[0], dtype=np.uint32)
    ambient._scatter_packed_grains(python, px, py, intensity, lookup, axis, diagonal,
                                  7, 5, dark)
    np.testing.assert_array_equal(python, expected)


@pytest.mark.parametrize('failure', ('import', 'compile', 'disabled'))
def test_unavailable_compiler_keeps_exact_numpy_once(monkeypatch, failure):
    starts, targets = [], []
    if failure == 'import':
        monkeypatch.setitem(sys.modules, 'numba', None)
    else:
        def njit(**kwargs):
            assert kwargs == {'nogil': True, 'cache': False}
            if failure == 'compile':
                raise RuntimeError('unavailable compiler')
            return lambda function: function
        monkeypatch.setitem(sys.modules, 'numba', SimpleNamespace(njit=njit))

    class ImmediateThread:
        def __init__(self, target, **kwargs):
            starts.append(kwargs)
            self.target = target

        def start(self):
            targets.append(self.target)

    monkeypatch.setattr(ambient, 'threading', SimpleNamespace(Thread=ImmediateThread))
    assert ambient._ready_packed_scatter() is None
    targets[0]()
    for _ in range(3):
        assert ambient._ready_packed_scatter() is None
    assert ambient._PACKED_SCATTER_FAILED and ambient._PACKED_SCATTER_STARTED
    assert starts == [{'name': 'spacr-grain-compile', 'daemon': True}]
    image = _engine()._point_material(3, 3, [1], [1], [.8], True)
    assert image.width() == image.height() == 3


def test_startup_failure_is_bounded_and_a_contended_lock_never_waits(monkeypatch):
    lock = ambient._PACKED_SCATTER_LOCK
    lock.acquire()
    try:
        assert ambient._ready_packed_scatter() is None
        assert not ambient._PACKED_SCATTER_STARTED
    finally:
        lock.release()
    starts = []

    class BrokenThread:
        def __init__(self, **kwargs):
            starts.append(kwargs)

        def start(self):
            raise RuntimeError('thread unavailable')

    monkeypatch.setattr(ambient, 'threading', SimpleNamespace(Thread=BrokenThread))
    assert ambient._ready_packed_scatter() is None
    assert ambient._ready_packed_scatter() is None
    assert len(starts) == 1 and ambient._PACKED_SCATTER_FAILED


def test_concurrent_warmup_start_is_not_duplicated(monkeypatch):
    released = []

    class RacedLock:
        def acquire(self, **kwargs):
            ambient._PACKED_SCATTER_STARTED = True
            return True

        def release(self):
            released.append(True)

    monkeypatch.setattr(ambient, '_PACKED_SCATTER_LOCK', RacedLock())
    assert ambient._ready_packed_scatter() is None
    assert released == [True]
    assert ambient._PACKED_SCATTER_STARTED and not ambient._PACKED_SCATTER_FAILED


def test_cold_compilation_uses_worker_and_gui_keeps_ticking(qtbot, monkeypatch):
    callers, beats = [], []
    original = ambient._warm_packed_scatter

    def record():
        callers.append(threading.current_thread().name)
        original()

    monkeypatch.setattr(ambient, '_warm_packed_scatter', record)
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: beats.append(time.perf_counter()))
    timer.start()
    try:
        assert ambient._ready_packed_scatter() is None
        qtbot.waitUntil(lambda: ambient._PACKED_SCATTER is not None, timeout=10000)
        assert callers == ['spacr-grain-compile']
        assert len(beats) >= 3
        assert ambient._PACKED_SCATTER.nopython_signatures
    finally:
        timer.stop()


def test_native_buffer_survives_hide_while_compiled_worker_finishes(
        qtbot, monkeypatch, compiled_kernel):
    monkeypatch.setattr(ambient, '_PACKED_SCATTER', compiled_kernel)

    class NativeWidget(ambient.AmbientWidget):
        def _follow_screen(self):
            with self._engine_lock:
                self._engine.set_max_pixels(self.width() * self.height())

    widget = NativeWidget(theme='data_art_point_atlas', palette='spacr',
                          background='#101418', fps=24, seed=42, blur=0, resolution=2)
    qtbot.addWidget(widget)
    widget.resize(3840, 2160)
    widget.show()
    qtbot.waitUntil(widget.shading_thread_alive)
    producer = widget._producer_box[0]
    entered, release = threading.Event(), threading.Event()

    def finishing(*args):
        entered.set()
        assert release.wait(5)
        compiled_kernel(*args)

    monkeypatch.setattr(ambient, '_PACKED_SCATTER', finishing)
    try:
        qtbot.waitUntil(entered.is_set, timeout=5000)
        widget.hide()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not producer.is_alive(), timeout=5000)
    frame = producer.latest()
    assert frame.width() == 3840 and frame.height() == 2160
    assert len(frame.bits()) == 3840 * 2160 * 4
    assert not widget.shading_thread_alive()

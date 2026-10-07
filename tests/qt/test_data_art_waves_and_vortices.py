"""Native satin fibres travel as waves and seeded flow structures evolve."""

import gc
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from spacr.qt.widgets import ambient


@pytest.fixture(scope='module')
def wave_kernel():
    from numba import njit

    kernel = njit(nogil=True, cache=False)(ambient._warp_satin_columns)
    source = np.zeros((1, 1), dtype=np.uint32)
    coordinates = np.zeros(1, dtype=np.int32)
    kernel(source, source.copy(), coordinates, coordinates, coordinates + 1, 0)
    assert kernel.nopython_signatures
    return kernel


def _compiler(kernel=None):
    compiler = ambient._SatinCompiler()
    compiler.kernel = kernel
    compiler.failed = kernel is None
    return compiler


def _engine(background='#101418', detail=2):
    engine = ambient._DataArtEngine(
        ambient.palette_colors('data_art_tissue_facets', 'spacr'), background,
        family='chromatin_ribbon', seed=42, resolution=detail, blur=0, density=1)
    engine.set_max_pixels(960 * 540)
    return engine


def _frame(engine, stamp):
    engine.set_time(stamp)
    image = engine.shade(960, 540)
    return image.bits().tobytes()


@pytest.mark.parametrize('background', ['#101418', '#fafafa'])
@pytest.mark.parametrize('detail', [1, 2])
def test_compiled_wave_matches_every_numpy_pixel_and_history_reset(
        wave_kernel, monkeypatch, background, detail):
    monkeypatch.setattr(ambient, '_SATIN_COMPILER', _compiler())
    reference = _engine(background, detail)
    reference.set_colors(['#ff7733', '#33ddff'])
    expected = {stamp: _frame(reference, stamp) for stamp in [0, 3, 14, 1.1]}
    monkeypatch.setattr(ambient, '_SATIN_COMPILER', _compiler(wave_kernel))
    actual = _engine(background, detail)
    actual.set_colors(['#ff7733', '#33ddff'])
    for stamp in [0, 3, 14, 1.1, 14, 0]:
        assert _frame(actual, stamp) == expected[stamp]
    assert actual.buffer_size(960, 540) == (960, 540)
    assert len(actual._material_cache) == 2


def _reference_shift(source, shifts, padding=0):
    result = np.zeros((source.shape[0] + 2 * padding, source.shape[1]), dtype=source.dtype)
    for column, shift in enumerate(shifts):
        for row in range(result.shape[0]):
            source_row = row - shift - padding
            if 0 <= source_row < source.shape[0]:
                result[row, column] = source[source_row, column]
    return result


@pytest.mark.parametrize('padding', [0, 16])
def test_native_copy_preserves_sparse_edges_and_padding(wave_kernel, padding):
    rng = np.random.default_rng(17)
    source = np.zeros((89, 97), dtype=np.uint32)
    source[17:68, 4:95] = rng.integers(0, 2**32, (51, 91), dtype=np.uint32)
    occupied = source != 0
    present = np.any(occupied, axis=0)
    tops = np.where(present, np.argmax(occupied, axis=0), 89).astype(np.int32)
    bottoms = np.where(present, 89 - np.argmax(occupied[::-1], axis=0), 0).astype(np.int32)
    for kernel in [ambient._warp_satin_columns, wave_kernel]:
        for reach in [12, 110, 3, 0, 24]:
            shifts = rng.integers(-reach, reach + 1, 97, dtype=np.int32)
            target = np.zeros((89 + 2 * padding, 97), dtype=np.uint32)
            kernel(source, target, shifts, tops, bottoms, padding)
            assert np.array_equal(target, _reference_shift(source, shifts, padding))
            target.fill(2**32 - 1)
            ambient._numpy_satin_columns(source, target, shifts, padding)
            assert np.array_equal(target, _reference_shift(source, shifts, padding))


def test_wave_deforms_columns_without_rerolling_or_interpolating_fibres(
        wave_kernel, monkeypatch):
    captured = []

    def capture(source, target, shifts, tops, bottoms, padding):
        wave_kernel(source, target, shifts, tops, bottoms, padding)
        captured.append((source.copy(), target.copy(), shifts.copy(), padding))

    monkeypatch.setattr(ambient, '_SATIN_COMPILER', _compiler(capture))
    engine = _engine()
    first = _frame(engine, 3)
    ribbons = engine._material_cache[('chromatin_native_folds', 960, 540, 2, 1, 1)]
    warps = engine._material_cache[('chromatin_native_waves', 960, 540, 2, 1, 1)]
    original = [warp[0].copy() for warp in warps]
    captured.clear()
    assert _frame(engine, 3.2) != first
    assert engine._material_cache[('chromatin_native_folds', 960, 540, 2, 1, 1)] is ribbons
    for source, warp in zip(original, warps):
        assert np.array_equal(source, warp[0])
        assert not warp[0].flags.owndata
    assert len(captured) > len(warps)
    assert any(np.ptp(shifts) > 1 for _, _, shifts, _ in captured)
    for source, target, shifts, padding in captured:
        rows = np.arange(target.shape[0])[:, None] - shifts[None, :] - padding
        inside = (rows >= 0) & (rows < source.shape[0])
        expected = source[np.clip(rows, 0, source.shape[0] - 1), np.arange(source.shape[1])]
        expected[~inside] = 0
        assert np.array_equal(target, expected)
    assert _frame(engine, 3) == first
    engine.set_density(.25)
    assert not engine._material_cache


@pytest.mark.parametrize('failure', ['import', 'compile', 'disabled'])
def test_wave_compile_failure_falls_back_once_without_retry(monkeypatch, failure):
    compiler = ambient._SatinCompiler()
    scheduled = []

    class DeferredThread:
        def __init__(self, target, name, daemon):
            assert name == 'spacr-satin-compile' and daemon
            self.target = target

        def start(self):
            scheduled.append(self.target)

    monkeypatch.setattr(ambient.threading, 'Thread', DeferredThread)
    if failure == 'import':
        monkeypatch.setitem(sys.modules, 'numba', None)
    else:
        def decorator(*args, **kwargs):
            if failure == 'compile':
                raise RuntimeError('compiler unavailable')
            return lambda function: function

        monkeypatch.setitem(sys.modules, 'numba', SimpleNamespace(njit=decorator))
    assert compiler.ready() is None
    assert len(scheduled) == 1
    scheduled[0]()
    assert compiler.failed
    for _ in range(3):
        assert compiler.ready() is None
    assert len(scheduled) == 1


def test_contended_or_failed_wave_startup_never_waits_or_retries(monkeypatch):
    compiler = ambient._SatinCompiler()
    compiler.lock.acquire()
    assert compiler.ready() is None
    assert not compiler.started
    compiler.lock.release()

    def reject_thread(**kwargs):
        raise RuntimeError('thread creation refused')

    monkeypatch.setattr(ambient.threading, 'Thread', reject_thread)
    assert compiler.ready() is None
    assert compiler.failed and compiler.started
    assert compiler.ready() is None


def test_wave_warmup_publishes_real_cpu_signature_away_from_gui():
    compiler = ambient._SatinCompiler()
    assert compiler.ready() is None
    workers = [thread for thread in threading.enumerate()
               if thread.name == 'spacr-satin-compile']
    assert workers
    for worker in workers:
        worker.join(10)
        assert not worker.is_alive()
    assert compiler.ready().nopython_signatures


def test_competing_wave_start_wins_between_eligibility_and_lock(monkeypatch):
    compiler = ambient._SatinCompiler()

    class RacingLock:
        def acquire(self, blocking):
            assert not blocking
            compiler.started = True
            return True

        def release(self):
            pass

    compiler.lock = RacingLock()
    monkeypatch.setattr(ambient.threading, 'Thread', lambda **kwargs: pytest.fail(
        'a competing startup must not schedule another compiler'))
    assert compiler.ready() is None


def _tasks():
    tasks = []
    for index in range(5):
        source = np.zeros((12, 9), dtype=np.uint32)
        source[2:10] = index + 5
        shifts = np.arange(9, dtype=np.int32) % 5 - 2
        tasks.append((source, np.zeros_like(source), shifts,
                      np.full(9, 2, dtype=np.int32), np.full(9, 10, dtype=np.int32), 0))
    return tuple(tasks)


def _check_tasks(tasks):
    for source, target, shifts, _, _, padding in tasks:
        assert np.array_equal(target, _reference_shift(source, shifts, padding))


def test_parallel_copy_is_exact_with_one_shared_queued_batch(wave_kernel, monkeypatch):
    compiler = _compiler(wave_kernel)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='spacr-satin-copy-test')
    submitted = []

    class TrackedPool:
        def submit(self, *args):
            submitted.append(args)
            return pool.submit(*args)

    compiler.pool = TrackedPool()
    entered, release = threading.Event(), threading.Event()
    original = ambient._copy_wave_batch

    def blocked_batch(kernel, tasks):
        if threading.current_thread().name.startswith('spacr-satin-copy-test'):
            assert all(isinstance(value, (np.ndarray, int)) for arguments in tasks
                       for value in arguments)
            entered.set()
            assert release.wait(5)
        original(kernel, tasks)

    monkeypatch.setattr(ambient, '_copy_wave_batch', blocked_batch)
    first, second = _tasks(), _tasks()
    errors = []

    def copy_first():
        try:
            compiler.copy_waves(wave_kernel, first)
        except Exception as error:
            errors.append(error)

    owner = threading.Thread(target=copy_first, name='wave-owner-test')
    owner.start()
    try:
        assert entered.wait(2)
        assert owner.is_alive()
        compiler.copy_waves(wave_kernel, second)
        _check_tasks(second)
        assert len(submitted) == 1
        assert owner.is_alive()
    finally:
        release.set()
        owner.join(5)
        pool.shutdown(wait=True)
    assert not owner.is_alive() and not errors
    _check_tasks(first)
    assert compiler.copy_gate.acquire(blocking=False)
    compiler.copy_gate.release()


def test_retired_parallel_executor_falls_back_to_exact_synchronous_copy(wave_kernel):
    compiler = _compiler(wave_kernel)
    compiler.pool = ThreadPoolExecutor(max_workers=1)
    compiler.pool.shutdown(wait=True)
    tasks = _tasks()
    compiler.copy_waves(wave_kernel, tasks)
    _check_tasks(tasks)
    single = tasks[:1]
    compiler.copy_waves(wave_kernel, single)
    _check_tasks(single)


def test_helper_exception_is_joined_and_releases_shared_copy_gate(monkeypatch):
    compiler = _compiler()
    compiler.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='failing-wave-copy')
    finished = threading.Event()

    def fail_background(*args):
        if threading.current_thread().name.startswith('failing-wave-copy'):
            finished.set()
            raise RuntimeError('injected CPU copy failure')

    monkeypatch.setattr(ambient, '_copy_wave_batch', fail_background)
    try:
        with pytest.raises(RuntimeError, match='CPU copy failure'):
            compiler.copy_waves(None, _tasks())
        assert finished.is_set()
        assert compiler.copy_gate.acquire(blocking=False)
        compiler.copy_gate.release()
    finally:
        compiler.pool.shutdown(wait=True)



def test_failed_runtime_copy_joins_helper_then_restores_exact_frame(monkeypatch):
    monkeypatch.setattr(ambient, '_SATIN_COMPILER', _compiler())
    expected = _frame(_engine(), 4)
    completed = threading.Event()

    def broken(source, target, *args):
        target.fill(0xffffffff)
        completed.set()
        raise RuntimeError('partial native layer')

    compiler = _compiler(broken)
    compiler.pool = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(ambient, '_SATIN_COMPILER', compiler)
    try:
        engine = _engine()
        assert _frame(engine, 4) == expected
        assert completed.is_set()
        assert compiler.failed and compiler.kernel is None
        assert compiler.copy_gate.acquire(blocking=False)
        compiler.copy_gate.release()
        assert _frame(engine, 4) == expected
    finally:
        compiler.pool.shutdown(wait=True)


def test_native_wave_cache_retains_geometry_not_frame_work(monkeypatch, wave_kernel):
    targets = []

    def capture(source, target, *args):
        wave_kernel(source, target, *args)
        targets.append(weakref.ref(target))

    monkeypatch.setattr(ambient, '_SATIN_COMPILER', _compiler(capture))
    engine = _engine()
    engine.set_max_pixels(3840 * 2160)
    image = engine.shade(3840, 2160)
    del image
    gc.collect()
    assert targets and all(reference() is None for reference in targets)
    warps = next(value for key, value in engine._material_cache.items()
                 if key[0] == 'chromatin_native_waves')
    owned = sum(array.nbytes for warp in warps for array in warp
                if isinstance(array, np.ndarray) and array.flags.owndata)
    assert owned < 512 * 1024
    assert all(not warp[0].flags.owndata for warp in warps)


@pytest.mark.parametrize('failure', [False, True])
def test_daemon_copy_queue_releases_completed_array_ownership(failure):
    worker = ambient._WaveCopyWorker()
    value = np.arange(31)
    reference = weakref.ref(value)

    def use(array):
        assert array.size == 31
        if failure:
            raise RuntimeError('failed CPU batch')

    future = worker.submit(use, value)
    if failure:
        with pytest.raises(RuntimeError, match='failed CPU batch'):
            future.result(timeout=5)
    else:
        future.result(timeout=5)
    worker._pending.join()
    del value, future
    gc.collect()
    assert reference() is None
    assert worker._pending.maxsize == 1
    assert worker._pending.unfinished_tasks == 0


def test_daemon_copy_queue_bounds_pending_work_and_honours_cancellation():
    from queue import Full

    worker = ambient._WaveCopyWorker()
    entered, release = threading.Event(), threading.Event()

    def blocking():
        entered.set()
        assert release.wait(5)

    first = worker.submit(blocking)
    assert entered.wait(2)
    second = worker.submit(lambda: pytest.fail('cancelled work must not run'))
    assert second.cancel()
    with pytest.raises(Full):
        worker.submit(lambda: None)
    release.set()
    first.result(timeout=5)
    worker._pending.join()
    assert second.cancelled()
    assert worker._pending.unfinished_tasks == 0


@pytest.mark.parametrize('background', ['#101418', '#fafafa'])
@pytest.mark.parametrize('dimensions', [(960, 540), (3840, 2160)])
def test_cropped_strip_composite_matches_complete_native_layers(
        wave_kernel, monkeypatch, background, dimensions):
    import math
    from types import MethodType

    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QImage

    monkeypatch.setattr(ambient, '_SATIN_COMPILER', _compiler(wave_kernel))
    width, height = dimensions
    actual = _engine(background)
    reference = _engine(background)
    for engine in (actual, reference):
        engine.set_max_pixels(width * height)
        engine.set_colors(['#ff7733', '#33ddff'])
        engine.set_size(1.1)
    initial = actual.shade(width, height)
    del initial

    def paint_complete(self, painter, w, h):
        key = ('chromatin_native_folds', w, h, self.resolution, self.size, self.density)
        ribbons = actual._material_cache[key]
        warps = actual._material_cache[('chromatin_native_waves', *key[1:])]
        for ribbon, warp in zip(ribbons, warps):
            centre, phase, _, origin_x, origin_y = ribbon
            source, along, _, _, padding = warp
            shifts = np.rint(h * .024 * (
                np.sin(math.tau * along * 1.1 - self.time * .55 + phase)
                + .35 * np.sin(math.tau * along * 2.6 + self.time * .31 + phase * .63)
            )).astype(np.int32)
            target = np.empty((source.shape[0] + 2 * padding, source.shape[1]),
                              dtype=np.uint32)
            ambient._numpy_satin_columns(source, target, shifts, padding)
            image = QImage(target.data, target.shape[1], target.shape[0],
                           target.strides[0], QImage.Format_ARGB32_Premultiplied)
            painter.drawImage(QPointF(origin_x, centre + origin_y - padding), image)

    reference._paint_chromatin_ribbon = MethodType(paint_complete, reference)
    for stamp in [0, 3, 14, 1.1]:
        actual.set_time(stamp)
        reference.set_time(stamp)
        observed = actual.shade(width, height)
        expected = reference.shade(width, height)
        assert observed.bits().tobytes() == expected.bits().tobytes()

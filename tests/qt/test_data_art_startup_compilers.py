"""Home's actual readiness boundary defers optional compiler imports."""

import builtins

import pytest

from spacr.qt.widgets import ambient


@pytest.fixture
def cold_compilers(monkeypatch):
    monkeypatch.setattr(ambient, '_PACKED_SCATTER', None)
    monkeypatch.setattr(ambient, '_PACKED_SCATTER_STARTED', False)
    monkeypatch.setattr(ambient, '_PACKED_SCATTER_FAILED', False)
    monkeypatch.setattr(ambient, '_COLORED_SCATTER', None)
    monkeypatch.setattr(ambient, '_COLORED_SCATTER_FAILED', False)
    compiler = ambient._SatinCompiler()
    monkeypatch.setattr(ambient, '_SATIN_COMPILER', compiler)
    threads = []

    class DeferredThread:
        def __init__(self, target, name, daemon):
            self.target = target
            self.name = name
            assert daemon

        def start(self):
            threads.append(self)

    monkeypatch.setattr(ambient.threading, 'Thread', DeferredThread)
    ambient._complete_ambient_startup()
    try:
        yield compiler, threads
    finally:
        ambient._complete_ambient_startup()


def test_closed_startup_preserves_one_attempt_until_actual_ready(cold_compilers):
    compiler, threads = cold_compilers
    ambient._begin_ambient_startup()
    for _ in range(5):
        assert ambient._ready_packed_scatter() is None
        assert compiler.ready() is None
    assert not ambient._PACKED_SCATTER_STARTED
    assert not compiler.started
    assert not threads
    ambient._complete_ambient_startup()
    assert ambient._ready_packed_scatter() is None
    assert compiler.ready() is None
    assert [thread.name for thread in threads] == ['spacr-grain-compile',
                                                  'spacr-satin-compile']
    for _ in range(5):
        ambient._ready_packed_scatter()
        compiler.ready()
    assert len(threads) == 2


def test_close_between_scheduling_and_worker_entry_defers_without_import_or_failure(
        cold_compilers, monkeypatch):
    compiler, threads = cold_compilers
    ambient._ready_packed_scatter()
    compiler.ready()
    ambient._begin_ambient_startup()
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in ('numba', 'scipy'):
            raise AssertionError('Compiler imported before readiness')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    for thread in tuple(threads):
        thread.target()
    assert not ambient._PACKED_SCATTER_STARTED
    assert not ambient._PACKED_SCATTER_FAILED
    assert not compiler.started
    assert not compiler.failed
    ambient._complete_ambient_startup()
    ambient._ready_packed_scatter()
    compiler.ready()
    assert len(threads) == 4


@pytest.mark.parametrize('theme', ['data_art_impulse_lens', 'data_art_point_atlas',
                                  'data_art_tissue_facets'])
def test_first_frames_while_blocked_equal_exact_numpy_fallback(
        theme, qapp, cold_compilers, monkeypatch):
    compiler, threads = cold_compilers
    engine = ambient.make_engine(theme, 'spacr', '#101418', seed=42,
                                 resolution=2)
    engine.set_max_pixels(960 * 540)
    engine.set_time(17)
    ambient._begin_ambient_startup()
    image = engine.shade(960, 540)
    blocked_pixels = image.bits().tobytes()
    assert not threads
    assert not compiler.started
    assert not ambient._PACKED_SCATTER_STARTED
    ambient._complete_ambient_startup()
    monkeypatch.setattr(ambient, '_PACKED_SCATTER_FAILED', True)
    compiler.failed = True
    image = engine.shade(960, 540)
    assert image.bits().tobytes() == blocked_pixels


def test_existing_kernels_are_reusable_without_new_imports(cold_compilers, monkeypatch):
    compiler, threads = cold_compilers
    kernel = object()
    monkeypatch.setattr(ambient, '_PACKED_SCATTER', kernel)
    compiler.kernel = kernel
    ambient._begin_ambient_startup()
    assert ambient._ready_packed_scatter() is kernel
    assert compiler.ready() is kernel
    assert not threads


def test_random_palette_starts_no_compiler_before_actual_readiness(qapp, cold_compilers):
    _, threads = cold_compilers
    ambient._begin_ambient_startup()
    engine = ambient.make_engine('data_art_impulse_lens', 'random', '#101418', seed=42)
    image = engine.shade(960, 540)
    assert not image.isNull()
    assert not threads
    assert not ambient._PACKED_SCATTER_STARTED
    assert ambient._ready_colored_scatter() is None
    ambient._complete_ambient_startup()
    assert ambient._ready_colored_scatter() is None
    assert [thread.name for thread in threads] == ['spacr-grain-compile']
    for _ in range(5):
        ambient._ready_colored_scatter()
    assert len(threads) == 1

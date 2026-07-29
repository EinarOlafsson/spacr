"""``measure_crop`` under a ``spawn`` multiprocessing pool.

Windows and macOS have defaulted to ``spawn`` since Python 3.8; Linux still
defaults to ``fork``, and Python 3.14 moves it off ``fork`` too. Every spaCR
developer machine is Linux, so the entire measure pool has only ever been
exercised under ``fork`` in CI -- the one start method that hides both classes
of bug this module guards:

* **Correctness.** A ``fork`` worker inherits the parent's imports, globals and
  open handles. A ``spawn`` worker inherits none of it: the callable and every
  argument must survive a pickle round trip and be reachable by qualified name
  from a cold interpreter. Turning ``_measure_crop_core`` into a closure, or
  passing it a live sqlite connection or a loaded model, breaks Windows and
  macOS while leaving Linux green.

* **Cost.** A ``fork`` worker is nearly free. A ``spawn`` worker is a fresh
  interpreter that re-imports the whole measure chain -- measured on a
  developer box at ~3.5 s and ~930 MB of RSS *each*, and it was 8 s and 1.5 GB
  before ``spacr.plot`` and umap/TensorFlow were taken off that path. With
  ``n_jobs`` defaulting to ``cpu_count - 4`` and no cap at the field count, a
  16-core Windows box reserved ~18 GB to measure a handful of fields, and the
  run presented as "prints 'using 12 cpu cores', then nothing happens".

Nothing here calls ``set_start_method(force=True)``: that would mutate the
whole interpreter for every test that runs afterwards. The pool takes an
explicit context, selected by :data:`spacr.measure.START_METHOD_ENV_VAR`, so a
Linux box can run the Windows/macOS path for real without changing its own
default.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import sqlite3
import sys

import numpy as np
import pytest

from spacr import measure as M


# ---------------------------------------------------------------------------
# a small real project: merged .npy fields, written the way the pipeline does
# ---------------------------------------------------------------------------

def _merged_field(size=96, n_cells=3):
    """One merged (Y, X, C) stack: 4 intensity channels then cell/nucleus/pathogen."""
    yy, xx = np.mgrid[:size, :size]
    centres = [(24, 24), (24, 72), (72, 24), (72, 72)][:n_cells]
    cell = np.zeros((size, size), np.uint16)
    nucleus = np.zeros((size, size), np.uint16)
    for i, (cy, cx) in enumerate(centres, start=1):
        cell[(yy - cy) ** 2 + (xx - cx) ** 2 <= 14 ** 2] = i
        nucleus[(yy - cy) ** 2 + (xx - cx) ** 2 <= 5 ** 2] = i
    pathogen = np.zeros((size, size), np.uint16)
    cy, cx = centres[0]
    pathogen[(yy - cy) ** 2 + (xx - cx) ** 2 <= 3 ** 2] = 1

    rng = np.random.default_rng(0)
    chans = []
    for _ in range(4):
        base = rng.integers(50, 200, size=(size, size)).astype(np.uint16)
        base[cell > 0] += 3000
        chans.append(base)
    return np.stack(chans + [cell, nucleus, pathogen], axis=-1).astype(np.uint16)


@pytest.fixture
def merged_project(tmp_path):
    """Two merged fields plus the ``measurements/`` folder the pipeline leaves."""
    merged = tmp_path / 'merged'
    merged.mkdir(parents=True)
    (tmp_path / 'measurements').mkdir(parents=True)
    data = _merged_field()
    for field in (1, 2):
        np.save(merged / f'plate1_A01_F{field:03d}.npy', data)
    return tmp_path


def _settings(merged_dir, **over):
    from spacr.settings import get_measure_crop_settings
    s = get_measure_crop_settings(settings={})
    s.update({
        'src': str(merged_dir),
        'channels': [0, 1, 2, 3],
        'cell_mask_dim': 4, 'nucleus_mask_dim': 5, 'pathogen_mask_dim': 6,
        'png_dims': [0, 1, 2], 'png_size': [32, 32],
        'save_measurements': True, 'save_png': True, 'save_arrays': False,
        'plot': False, 'verbose': False, 'timelapse': False,
        'crop_mode': ['cell'], 'normalize': [1, 99], 'normalize_by': 'png',
        'experiment': 'exp', 'test_mode': False, 'cytoplasm': True,
        # Two workers, two fields: enough for a real pool, cheap enough that
        # two cold interpreters is the whole bill.
        'n_jobs': 2,
    })
    s.update(over)
    return s


# ---------------------------------------------------------------------------
# the context and the pool size
# ---------------------------------------------------------------------------

def test_pool_context_defaults_to_the_multiprocessing_module(monkeypatch):
    """Unset means unchanged: ``mp.Pool``/``mp.Manager`` looked up as before."""
    monkeypatch.delenv(M.START_METHOD_ENV_VAR, raising=False)
    assert M._pool_context() is mp


def test_pool_context_honours_an_explicit_start_method(monkeypatch):
    monkeypatch.setenv(M.START_METHOD_ENV_VAR, 'spawn')
    ctx = M._pool_context()
    assert ctx is not mp
    assert ctx.get_start_method() == 'spawn'


def test_pool_context_tolerates_whitespace_and_case(monkeypatch):
    monkeypatch.setenv(M.START_METHOD_ENV_VAR, '  SPAWN ')
    assert M._pool_context().get_start_method() == 'spawn'


def test_an_unusable_start_method_falls_back_rather_than_raising(monkeypatch, capsys):
    """``fork`` on Windows, or a typo, must not take a measure run down."""
    monkeypatch.setenv(M.START_METHOD_ENV_VAR, 'no_such_method')
    assert M._pool_context() is mp
    assert 'not a multiprocessing start method' in capsys.readouterr().out


def test_fork_starts_exactly_what_was_asked_for():
    """A surplus fork worker is a page-table copy; capping it would be a
    behaviour change for no gain."""
    assert M.resolve_pool_size(12, 1, start_method='fork') == 12
    assert M.resolve_pool_size(12, 40, start_method='fork') == 12


@pytest.mark.parametrize('method', ['spawn', 'forkserver'])
def test_a_cold_start_pool_is_capped_at_the_field_count(method):
    """This is the Windows/macOS fix: 12 interpreters were booted, at ~930 MB
    each, to measure four fields."""
    assert M.resolve_pool_size(12, 4, start_method=method) == 4
    assert M.resolve_pool_size(2, 40, start_method=method) == 2


@pytest.mark.parametrize('method', ['fork', 'spawn'])
def test_no_fields_still_yields_a_legal_pool(method):
    """``Pool(0)`` raises ``ValueError``. A resumed folder with nothing left to
    measure must finish quietly, not crash."""
    assert M.resolve_pool_size(8, 0, start_method=method) >= 1


def test_resolve_pool_size_defaults_to_the_live_start_method():
    """Called without an explicit method it must agree with the interpreter."""
    expected = 8 if mp.get_start_method() == 'fork' else 3
    assert M.resolve_pool_size(8, 3) == expected


# ---------------------------------------------------------------------------
# what spawn requires of the worker entry point
# ---------------------------------------------------------------------------

def test_the_worker_is_importable_by_qualified_name():
    """A closure or a local function pickles under fork's inheritance and
    fails under spawn. ``<locals>`` in the qualname is the tell."""
    assert '<locals>' not in M._measure_crop_core.__qualname__
    assert M._measure_crop_core.__module__ == 'spacr.measure'
    assert pickle.loads(pickle.dumps(M._measure_crop_core)) is M._measure_crop_core


def test_everything_apply_async_sends_to_a_worker_pickles(merged_project):
    """The exact args tuple ``measure_crop`` builds, through a real round trip."""
    settings = _settings(merged_project / 'merged')
    args = (0, [], 'plate1_A01_F001.npy', settings)
    index, time_ls, name, restored = pickle.loads(pickle.dumps(args))
    assert (index, name) == (0, 'plate1_A01_F001.npy')
    assert restored['src'] == settings['src']
    assert restored['cell_mask_dim'] == 4


# ---------------------------------------------------------------------------
# the end-to-end run, in a real spawn pool
# ---------------------------------------------------------------------------

def test_measure_crop_completes_in_a_spawn_pool(merged_project, monkeypatch):
    """The deliverable: the Windows/macOS pool, run on Linux, writing real rows.

    Every worker here is a cold interpreter that has never seen spaCR -- no
    inherited ``sys.modules``, no inherited globals, no inherited file
    descriptors beyond the pool's own pipes.
    """
    monkeypatch.setenv(M.START_METHOD_ENV_VAR, 'spawn')
    before = mp.get_start_method()

    M.measure_crop(_settings(merged_project / 'merged'))

    # The run must not have changed the interpreter's own start method.
    assert mp.get_start_method() == before

    db = merged_project / 'measurements' / 'measurements.db'
    assert db.is_file(), 'the spawn pool wrote no measurements database'
    con = sqlite3.connect(db)
    try:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {'cell', 'nucleus', 'cytoplasm', 'png_list'} <= tables
        # Two fields x three cells, measured by two separate interpreters.
        assert con.execute('SELECT COUNT(*) FROM cell').fetchone()[0] == 6
        fields = {r[0] for r in con.execute('SELECT DISTINCT fieldID FROM cell')}
        assert len(fields) == 2, f'a field went missing: {fields}'
        status, attempted, succeeded, failed = con.execute(
            'SELECT status, n_attempted, n_succeeded, n_failed '
            'FROM run_status').fetchone()
        assert (status, attempted, succeeded, failed) == ('complete', 2, 2, 0)
    finally:
        con.close()

    pngs = list(merged_project.rglob('*.png'))
    assert len(pngs) == 6, f'expected one crop per cell, got {len(pngs)}'


def test_a_spawn_pool_is_not_oversized_for_the_work(merged_project, monkeypatch):
    """Asking for more workers than there are fields must not boot them.

    Left uncapped this is what made Measure unusable on a normal Windows or
    Mac: ~930 MB and a multi-second interpreter boot per idle worker.
    """
    monkeypatch.setenv(M.START_METHOD_ENV_VAR, 'spawn')
    sizes = []
    real_pool = mp.get_context('spawn').Pool

    class _CountingContext:
        """The spawn context, recording the size of every pool it builds."""

        def __init__(self):
            self._ctx = mp.get_context('spawn')

        def get_start_method(self):
            return 'spawn'

        def Manager(self):
            return self._ctx.Manager()

        def Pool(self, processes=None, *a, **kw):
            sizes.append(processes)
            return real_pool(processes, *a, **kw)

    monkeypatch.setattr(M, '_pool_context', _CountingContext)
    M.measure_crop(_settings(merged_project / 'merged', n_jobs=8))

    assert sizes == [2], f'asked for 8 workers to measure 2 fields: {sizes}'


# ---------------------------------------------------------------------------
# the import weight a spawn worker pays
# ---------------------------------------------------------------------------

def test_utils_does_not_import_umap_at_module_scope():
    """umap drags in numba, pynndescent and -- through ``parametric_umap`` --
    TensorFlow. None of it measures anything, and under spawn every worker
    paid for all of it."""
    import spacr.utils as U
    # Other tests legitimately exercise UMAP first.  Reset the proxy so this
    # assertion tests its pristine lazy state rather than global test order.
    U.umap.reset()
    assert isinstance(U.umap, U._LazyModule)
    assert U.umap.__dict__['_module'] is None, 'umap was imported eagerly'
    assert 'not yet imported' in repr(U.umap)


def test_the_lazy_umap_still_behaves_like_the_module():
    """The two call sites write ``umap.UMAP(...)``; that has to keep working."""
    import spacr.utils as U
    pytest.importorskip('umap')
    assert hasattr(U.umap, 'UMAP')
    assert U.umap.__dict__['_module'] is not None
    assert 'loaded' in repr(U.umap)
    assert 'UMAP' in dir(U.umap)


def _module_census(q):
    """Import exactly what a default measure worker imports, then report."""
    import sys
    import spacr.measure  # noqa: F401
    from spacr.utils import _merge_and_save_to_database  # noqa: F401
    q.put(sorted(sys.modules))


def test_a_cold_worker_does_not_import_umap_or_tensorflow():
    """Run the worker's import list in a real spawned interpreter.

    A unit test on ``spacr.utils`` cannot see this: by the time it runs the
    parent has already imported half the world. Only a cold process can say
    what a spawn worker actually pays for -- and on Windows and macOS every
    worker is a cold process.
    """
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    proc = ctx.Process(target=_module_census, args=(q,))
    proc.start()
    try:
        loaded = set(q.get(timeout=300))
    finally:
        proc.join(60)

    assert 'spacr.measure' in loaded, 'the worker never got as far as spacr'
    assert 'spacr.utils' in loaded
    tops = {m.split('.')[0] for m in loaded}
    assert 'umap' not in tops, 'umap is back on the measure worker path'
    assert 'tensorflow' not in tops, (
        'TensorFlow reached a measure worker -- it arrives through '
        'umap.parametric_umap, and spaCR bans it')
    # spacr.plot is reachable only behind settings['plot']; a default run must
    # not pay ~1.9 s and ~720 MB per worker to not draw anything.
    assert 'spacr.plot' not in loaded

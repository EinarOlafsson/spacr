"""``measure_crop`` honours the worker count it is given.

The old block threw the user's ``n_jobs`` away. It compared ``n_jobs`` with the
core count *before* checking for ``None`` (so leaving it blank -- which the
warning it printed explicitly recommended -- raised ``TypeError``), and then
ended with an unconditional ``settings['n_jobs'] = spacr_cores``, so on a
32-core machine ``n_jobs=1`` ran 28 workers. That is what made the concurrent
``CREATE TABLE`` race reachable from a test that had asked for a single worker.

These tests assert on the number actually handed to :class:`multiprocessing.Pool`,
not on the settings dict: the settings dict was already right, and the pool was
what was wrong.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from spacr import measure as M
from spacr.errors import ConfigurationError


# --------------------------------------------------------------------------
# the resolver on its own
# --------------------------------------------------------------------------

def test_explicit_n_jobs_is_honoured():
    assert M.resolve_n_jobs(1, cpu_count=32) == 1
    assert M.resolve_n_jobs(4, cpu_count=32) == 4
    assert M.resolve_n_jobs(32, cpu_count=32) == 32


def test_blank_n_jobs_leaves_headroom_and_never_raises():
    """``None`` used to raise before it was ever checked for."""
    assert M.resolve_n_jobs(None, cpu_count=32) == 32 - M.N_JOBS_HEADROOM
    assert M.resolve_n_jobs(None, cpu_count=8) == 8 - M.N_JOBS_HEADROOM


@pytest.mark.parametrize('cores', [1, 2, 3, 4, 5])
def test_a_small_machine_still_gets_at_least_one_worker(cores):
    assert M.resolve_n_jobs(None, cpu_count=cores) >= 1


def test_n_jobs_above_the_core_count_is_clamped_and_says_so(capsys):
    assert M.resolve_n_jobs(64, cpu_count=8) == 8
    out = capsys.readouterr().out
    assert 'exceeds the 8 available cores' in out


def test_zero_and_negative_n_jobs_are_refused():
    for bad in (0, -1, -8):
        with pytest.raises(ConfigurationError):
            M.resolve_n_jobs(bad, cpu_count=16)


def test_a_non_integer_n_jobs_is_refused():
    for bad in ('4', 4.5, True, [4]):
        with pytest.raises(ConfigurationError):
            M.resolve_n_jobs(bad, cpu_count=16)


def test_resolve_n_jobs_defaults_to_the_real_cpu_count(monkeypatch):
    monkeypatch.setattr(mp, 'cpu_count', lambda: 12)
    assert M.resolve_n_jobs(None) == 12 - M.N_JOBS_HEADROOM
    assert M.resolve_n_jobs(3) == 3
    assert M.resolve_n_jobs(99) == 12


# --------------------------------------------------------------------------
# what measure_crop actually hands to the pool
# --------------------------------------------------------------------------

class _NoResult:
    """What a pool that runs nothing hands back for a submitted field.

    ``apply_async`` used to return a bare ``None`` here, and the
    ``AttributeError`` from ``None.get()`` was quietly absorbed by
    measure_crop's per-field except. It is now the ``on_error`` boundary
    that decides, so the failure is made explicit and ``_run`` asks for
    ``on_error='skip'`` — these tests are about the worker count, not
    about surviving a field.
    """

    def get(self, timeout=None):
        raise RuntimeError('this pool runs nothing; there is no result')


class _RecordingPool:
    """Stand-in for ``mp.Pool`` that records its worker count and runs nothing."""

    created = []

    def __init__(self, processes=None, *args, **kwargs):
        type(self).created.append(processes)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def apply_async(self, *args, **kwargs):
        return _NoResult()

    def close(self):
        pass

    def join(self):
        pass

    def terminate(self):
        pass


def _one_field(tmp_path):
    """Write a single tiny merged field so ``measure_crop`` has work to plan."""
    src = tmp_path / 'merged'
    src.mkdir(parents=True)
    data = np.zeros((24, 24, 7), np.uint16)
    data[6:18, 6:18, 4] = 1
    data[9:15, 9:15, 5] = 1
    np.save(src / 'plate1_A01_f1.npy', data)
    return src


def _run(tmp_path, monkeypatch, n_jobs, cpu_count=None):
    src = _one_field(tmp_path)
    _RecordingPool.created = []
    monkeypatch.setattr(mp, 'Pool', _RecordingPool)
    # These assertions cover resolution of the requested worker count, not
    # the separate spawn/forkserver optimization that caps workers to files.
    # Python 3.14 changed Linux's default away from fork, so pin this seam.
    monkeypatch.setattr(mp, 'get_start_method', lambda: 'fork')
    if cpu_count is not None:
        monkeypatch.setattr(mp, 'cpu_count', lambda: cpu_count)
    from spacr.settings import get_measure_crop_settings
    settings = get_measure_crop_settings({
        'src': str(src), 'save_png': False, 'save_arrays': False,
        'plot': False, 'test_mode': False, 'channels': [0, 1, 2],
        'cell_mask_dim': 4, 'nucleus_mask_dim': 5, 'pathogen_mask_dim': None,
        'experiment': 'exp', 'normalize': False, 'normalize_by': 'png',
        'strict_errors': False,
        # The stub pool returns no result for any field, so every field
        # fails. on_error defaults to 'stop', which would abort before the
        # pool size could be observed; these tests are about the size.
        'on_error': 'skip',
    })
    # Set after the defaults so setdefault cannot overwrite a deliberate None.
    settings['n_jobs'] = n_jobs
    M.measure_crop(settings)
    return _RecordingPool.created


def test_measure_crop_builds_a_pool_of_exactly_one(tmp_path, monkeypatch):
    """``n_jobs=1`` means one worker, whatever the machine has."""
    created = _run(tmp_path, monkeypatch, n_jobs=1, cpu_count=32)
    assert created == [1]


def test_measure_crop_blank_n_jobs_reaches_the_pool(tmp_path, monkeypatch):
    """Blank does not raise, and resolves to ``cpu_count - headroom``."""
    created = _run(tmp_path, monkeypatch, n_jobs=None, cpu_count=32)
    assert created == [32 - M.N_JOBS_HEADROOM]


def test_measure_crop_clamps_an_oversized_request(tmp_path, monkeypatch):
    created = _run(tmp_path, monkeypatch, n_jobs=99, cpu_count=6)
    assert created == [6]


def test_measure_crop_progress_and_pool_agree(tmp_path, monkeypatch):
    """``print_progress`` and ``mp.Pool`` are told the same resolved number."""
    seen = []
    import spacr.utils as U
    real = U.print_progress
    monkeypatch.setattr(U, 'print_progress',
                        lambda *a, **k: seen.append(k.get('n_jobs', a[2] if len(a) > 2 else None)))
    created = _run(tmp_path, monkeypatch, n_jobs=2, cpu_count=16)
    assert created == [2]
    assert seen and set(seen) == {2}
    assert real is not None


def test_measure_crop_refuses_zero_workers(tmp_path, monkeypatch):
    with pytest.raises(ConfigurationError):
        _run(tmp_path, monkeypatch, n_jobs=0, cpu_count=16)

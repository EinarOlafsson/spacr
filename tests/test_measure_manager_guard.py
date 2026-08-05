"""A ``Manager()`` that will not start must say so, and say what to do.

``measure_crop`` shares its per-field timing list with the worker pool through
a :class:`multiprocessing.managers.SyncManager`. Starting one forks (on Linux,
the default) a server process, which writes its socket address back down a
pipe; the parent reads that address. When the server never gets that far the
parent's read raises a bare ``EOFError`` from
``multiprocessing/connection.py`` -- no message, no mention of spaCR, no
mention of the start method, and nothing to act on.

That is not hypothetical. It was reproduced 33 times, deterministically,
across 7 test modules in this suite: every traceback ran
``spacr/measure.py`` -> ``multiprocessing/managers.py:566 start`` ->
``multiprocessing/connection.py:383 EOFError``, and it appeared only once
enough test modules had been collected for the parent to be carrying live
threads. The crash is measured; the fork-with-threads *mechanism* is the
inference, which is why the diagnosis names it as the likely cause rather
than as fact -- and why it prints the thread census it is reasoning from.

The guard deliberately does not fall back to a serial run. A pool that
silently becomes one worker turns a loud failure into an overnight one.
"""

from __future__ import annotations

import multiprocessing as mp
import sqlite3

import numpy as np
import pytest

from spacr import measure as M
from spacr.errors import ConfigurationError


# ---------------------------------------------------------------------------
# the diagnosis itself
# ---------------------------------------------------------------------------

def test_a_fork_diagnosis_names_the_method_the_cause_and_the_remedy():
    """The three things the bare EOFError never said."""
    message = M._manager_start_diagnosis('fork', EOFError())

    # the start method in use
    assert "'fork'" in message
    # the likely cause, named as likely
    assert 'live threads' in message
    assert 'os.fork()' in message
    assert 'Most likely cause' in message
    # the concrete remedy
    assert 'spawn' in message
    assert M.START_METHOD_ENV_VAR in message
    assert f"export {M.START_METHOD_ENV_VAR}=spawn" in message
    # and the underlying error, so the traceback is not lost in translation
    assert 'EOFError' in message


def test_the_diagnosis_reports_the_thread_census_it_reasons_from(monkeypatch):
    """The count is measured at failure time, not asserted in prose."""
    class _FakeThread:
        def __init__(self, name):
            self.name = name

    fake = [_FakeThread(n) for n in ('MainThread', 'QThread-1', 'QThread-2')]
    monkeypatch.setattr(M.threading, 'enumerate', lambda: fake)

    n, names = M._thread_census()
    assert n == 3
    assert 'QThread-1' in names

    message = M._manager_start_diagnosis('fork', EOFError())
    assert 'live threads in this process: 3' in message
    assert 'QThread-2' in message
    # 3 live threads means 2 other threads whose locks the child inherits.
    assert '2 thread(s) held' in message


def test_a_long_thread_list_is_truncated_rather_than_dumped(monkeypatch):
    """A Qt process can carry dozens; an unreadable message is not a diagnosis."""
    class _FakeThread:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(
        M.threading, 'enumerate',
        lambda: [_FakeThread(f'T{i}') for i in range(20)])

    n, names = M._thread_census()
    assert n == 20
    assert '(+12 more)' in names
    assert 'T19' not in names


def test_a_spawn_diagnosis_does_not_blame_the_parents_threads():
    """Under spawn the child inherits no locks, so the fork story is wrong.

    Handing a spawn user the fork diagnosis would send them to change a setting
    that is already set, which is worse than saying nothing.
    """
    message = M._manager_start_diagnosis('spawn', OSError('no such file'))

    assert "'spawn'" in message
    assert 'inherits no locks' in message
    assert 'Most likely cause' not in message
    # what is actually left to check under spawn
    assert 'TMPDIR' in message
    assert M.START_METHOD_ENV_VAR in message


# ---------------------------------------------------------------------------
# _start_manager
# ---------------------------------------------------------------------------

class _FailingContext:
    """A pool context whose Manager cannot start, exactly as reported."""

    def __init__(self, exc=None, start_method='fork'):
        self._exc = exc or EOFError()
        self._start_method = start_method

    def get_start_method(self):
        return self._start_method

    def Manager(self):
        raise self._exc

    def Pool(self, *a, **kw):  # pragma: no cover - never reached
        raise AssertionError('the pool must not be built after Manager failed')


def test_start_manager_returns_a_working_manager_when_nothing_is_wrong():
    """The guard must be transparent on the path that already worked."""
    with M._start_manager(mp) as manager:
        shared = manager.list()
        shared.append(1.5)
        assert list(shared) == [1.5]


def test_a_failed_manager_raises_a_diagnosed_error_not_a_bare_eoferror():
    """The whole point: EOFError in, actionable spaCR error out."""
    original = EOFError()
    with pytest.raises(M.ManagerStartError) as excinfo:
        M._start_manager(_FailingContext(original))

    assert excinfo.value.__cause__ is original, (
        'the original traceback must survive as __cause__')
    message = str(excinfo.value)
    assert "'fork'" in message
    assert M.START_METHOD_ENV_VAR in message


def test_the_error_is_a_configuration_error_so_the_ledger_re_raises_it():
    """A Manager that will not start is not a per-field failure.

    ``RunLedger.item`` records ordinary exceptions and carries on; it re-raises
    :class:`~spacr.errors.ConfigurationError` because continuing past one
    produces garbage for every remaining item. No field can be measured without
    the shared timing list, so this belongs on that side of the line.
    """
    assert issubclass(M.ManagerStartError, ConfigurationError)


def test_a_ctx_that_cannot_report_its_start_method_still_gets_diagnosed():
    """A third-party or test double context need not implement everything."""
    class _Broken(_FailingContext):
        def get_start_method(self):
            raise AttributeError('no such thing')

    with pytest.raises(M.ManagerStartError) as excinfo:
        M._start_manager(_Broken())
    # Falls back to the interpreter's own default rather than losing the field.
    assert repr(mp.get_start_method()) in str(excinfo.value)


def test_a_keyboard_interrupt_during_the_handshake_is_not_relabelled():
    """Ctrl-C is a cancellation, not a misconfiguration.

    ``except Exception`` rather than ``except BaseException`` is load-bearing:
    telling a user who just pressed Ctrl-C to set SPACR_START_METHOD would be a
    lie in the traceback.
    """
    with pytest.raises(KeyboardInterrupt):
        M._start_manager(_FailingContext(KeyboardInterrupt()))


# ---------------------------------------------------------------------------
# the caller, not the callee: measure_crop itself
# ---------------------------------------------------------------------------

def _merged_field(size=64, n_cells=2):
    """One merged (Y, X, C) stack: 4 intensity channels then cell/nucleus/pathogen."""
    yy, xx = np.mgrid[:size, :size]
    centres = [(18, 18), (18, 46)][:n_cells]
    cell = np.zeros((size, size), np.uint16)
    nucleus = np.zeros((size, size), np.uint16)
    for i, (cy, cx) in enumerate(centres, start=1):
        cell[(yy - cy) ** 2 + (xx - cx) ** 2 <= 10 ** 2] = i
        nucleus[(yy - cy) ** 2 + (xx - cx) ** 2 <= 4 ** 2] = i
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
    merged = tmp_path / 'merged'
    merged.mkdir(parents=True)
    (tmp_path / 'measurements').mkdir(parents=True)
    np.save(merged / 'plate1_A01_F001.npy', _merged_field())
    return tmp_path


def _settings(merged_dir, **over):
    from spacr.settings import get_measure_crop_settings
    s = get_measure_crop_settings(settings={})
    s.update({
        'src': str(merged_dir),
        'channels': [0, 1, 2, 3],
        'cell_mask_dim': 4, 'nucleus_mask_dim': 5, 'pathogen_mask_dim': 6,
        'png_dims': [0, 1, 2], 'png_size': [32, 32],
        'save_measurements': True, 'save_png': False, 'save_arrays': False,
        'plot': False, 'verbose': False, 'timelapse': False,
        'crop_mode': ['cell'], 'normalize': [1, 99], 'normalize_by': 'png',
        'experiment': 'exp', 'test_mode': False, 'cytoplasm': True,
        'n_jobs': 1,
    })
    s.update(over)
    return s


def test_measure_crop_reports_a_failed_manager_instead_of_crashing_blind(
        merged_project, monkeypatch):
    """The path a user actually takes.

    Before the guard this call raised ``EOFError`` with an empty message, from
    a frame inside the standard library, in a run the user had left overnight.
    """
    monkeypatch.setattr(
        M, '_pool_context', lambda: _FailingContext(EOFError()))

    with pytest.raises(M.ManagerStartError) as excinfo:
        M.measure_crop(_settings(merged_project / 'merged'))

    message = str(excinfo.value)
    assert 'measure_crop' in message
    assert 'Nothing was measured.' in message
    assert "'fork'" in message
    assert f"export {M.START_METHOD_ENV_VAR}=spawn" in message


def test_the_guard_does_not_silently_degrade_to_a_serial_run(
        merged_project, monkeypatch):
    """No rows, loudly -- not a run that quietly takes all night.

    A fallback that measured the fields one at a time would look like success
    to every caller and to the database, so the only evidence of the problem
    would be the wall clock.
    """
    monkeypatch.setattr(
        M, '_pool_context', lambda: _FailingContext(EOFError()))

    with pytest.raises(M.ManagerStartError):
        M.measure_crop(_settings(merged_project / 'merged'))

    db = merged_project / 'measurements' / 'measurements.db'
    if db.is_file():
        con = sqlite3.connect(db)
        try:
            tables = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            # The settings table is written before the pool starts; object
            # tables are only written by workers, and no worker ran.
            assert 'cell' not in tables
        finally:
            con.close()


def test_measure_crop_still_measures_when_the_manager_is_fine(merged_project):
    """The guard must not have changed the working path.

    ``ctx.Manager()`` already returns a *started* manager, so entering it as a
    context manager is a no-op re-entry; routing it through ``_start_manager``
    has to leave that unchanged.
    """
    M.measure_crop(_settings(merged_project / 'merged'))

    db = merged_project / 'measurements' / 'measurements.db'
    assert db.is_file()
    con = sqlite3.connect(db)
    try:
        assert con.execute('SELECT COUNT(*) FROM cell').fetchone()[0] == 2
    finally:
        con.close()

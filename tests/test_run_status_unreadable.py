"""An artifact whose run status cannot be read is not a finished run.

:func:`spacr.errors.read_run_status` had one ``except sqlite3.Error: return
[]``, which folded three different states into one answer:

1. the artifact holds no ``run_status`` table --- never stamped, no
   information. ``[]``, and :func:`run_is_complete` says True. Correct and
   deliberate: stamping is opt-in.
2. the database is **locked** --- a writer still holds it. That is what a run
   killed mid-write leaves behind, and it used to read as (1).
3. the file is **truncated or not a database** --- what a ``kill -9`` during a
   commit leaves. Also read as (1).

So an interrupted run reported as complete, and ``assert_run_complete``, whose
entire job is to stop downstream code analysing a subset, passed. The three
are now separated: (1) stays ``[]``, (2) and (3) raise
:class:`spacr.errors.RunStatusUnreadable`, which is a
:class:`spacr.errors.DataIntegrityError`, and ``run_is_complete`` answers
False for them.

The databases are built by :func:`spacr.utils.filepaths_to_database` and
:func:`spacr.utils._merge_and_save_to_database` and stamped by the real
:class:`spacr.errors.RunLedger`.

CPU-only, offline, deterministic. The locked cases pass an explicit short
``timeout`` so the suite does not spend SQLite's 5 s default waiting for a lock
that is never going to be released.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.errors import (DataIntegrityError, RunLedger, RunStatusUnreadable,
                          assert_run_complete, read_run_status,
                          run_is_complete)


LOCK_TIMEOUT = 0.05


# ---------------------------------------------------------------------------
# a real measurements database, stamped by a real ledger
# ---------------------------------------------------------------------------

def measured_database(root, n_objects=3):
    """One field measured and cropped through the writers ``measure`` uses."""
    from spacr.utils import _merge_and_save_to_database, filepaths_to_database

    os.makedirs(os.path.join(root, 'measurements'), exist_ok=True)
    labels = list(range(1, n_objects + 1))
    for table in ('cell', 'nucleus'):
        morphology = pd.DataFrame({
            'label': labels,
            f'{table}_area': [100.0 + i for i in range(n_objects)]})
        intensity = pd.DataFrame({
            'label': labels,
            f'{table}_channel_0_mean_intensity': [5.0] * n_objects})
        if table == 'nucleus':
            morphology['cell_id'] = np.asarray(labels, dtype=float)
        _merge_and_save_to_database(morphology, intensity, table, root,
                                    'plate1_A01_1', 'exp', False)
    folder = os.path.join(root, 'data', 'cell_png')
    os.makedirs(folder, exist_ok=True)
    filepaths_to_database(
        [os.path.join(folder, f'plate1_A01_1_{i}.png') for i in labels],
        {'timelapse': False}, root, 'cell')
    return os.path.join(root, 'measurements', 'measurements.db')


def stamp_partial(db, name='measure_crop'):
    """Stamp the database the way a run that lost a field does."""
    ledger = RunLedger(name)
    ledger.record_success('plate1_A01_1')
    ledger.record_failure('plate1_A01_2', RuntimeError('disk went away'))
    ledger.stamp(db)
    return ledger


@pytest.fixture()
def partial_db(tmp_path):
    db = measured_database(str(tmp_path))
    stamp_partial(db)
    return db


class held_lock:
    """Hold an exclusive write lock, as the process still writing does."""

    def __init__(self, db):
        self.db = db

    def __enter__(self):
        self.conn = sqlite3.connect(self.db, timeout=LOCK_TIMEOUT,
                                    isolation_level=None)
        self.conn.execute('BEGIN EXCLUSIVE')
        return self

    def __exit__(self, *exc):
        self.conn.rollback()
        self.conn.close()
        return False


# ---------------------------------------------------------------------------
# 1. the contract that must not change
# ---------------------------------------------------------------------------

def test_a_readable_partial_stamp_still_reads_partial(partial_db):
    records = read_run_status(partial_db)
    assert len(records) == 1
    assert records[0]['status'] == 'partial'
    assert records[0]['n_failed'] == 1
    assert run_is_complete(partial_db) is False
    with pytest.raises(DataIntegrityError):
        assert_run_complete(partial_db)


def test_an_unstamped_database_still_reads_as_no_information(tmp_path):
    """The deliberate case: stamping is opt-in and predates most outputs."""
    db = measured_database(str(tmp_path))
    assert read_run_status(db) == []
    assert run_is_complete(db) is True
    assert_run_complete(db)                       # must not raise


def test_a_missing_file_still_reads_as_no_information(tmp_path):
    missing = tmp_path / 'never_written.db'
    assert read_run_status(missing) == []
    assert run_is_complete(missing) is True


# ---------------------------------------------------------------------------
# 2. the finding: locked
# ---------------------------------------------------------------------------

def test_a_locked_database_is_unreadable_not_complete(partial_db):
    with held_lock(partial_db):
        with pytest.raises(RunStatusUnreadable) as excinfo:
            read_run_status(partial_db, timeout=LOCK_TIMEOUT)
        assert str(partial_db) in str(excinfo.value)
        assert run_is_complete(partial_db, timeout=LOCK_TIMEOUT) is False


def test_assert_run_complete_refuses_a_locked_database(partial_db):
    with held_lock(partial_db):
        with pytest.raises(RunStatusUnreadable):
            assert_run_complete(partial_db, timeout=LOCK_TIMEOUT)


def test_a_locked_database_that_was_never_stamped_is_unreadable_too(tmp_path):
    """The lock, not the stamp, is the evidence.

    An unstamped database reads as complete precisely because nothing is
    known; a *locked* one is different, because something is known --- a
    writer is or was in the middle of it.
    """
    db = measured_database(str(tmp_path))
    with held_lock(db):
        with pytest.raises(RunStatusUnreadable):
            read_run_status(db, timeout=LOCK_TIMEOUT)
        assert run_is_complete(db, timeout=LOCK_TIMEOUT) is False


def test_the_lock_is_released_and_the_verdict_comes_back(partial_db):
    with held_lock(partial_db):
        assert run_is_complete(partial_db, timeout=LOCK_TIMEOUT) is False
    assert len(read_run_status(partial_db)) == 1
    assert run_is_complete(partial_db) is False


# ---------------------------------------------------------------------------
# 3. the finding: truncated / not a database
# ---------------------------------------------------------------------------

def test_a_truncated_database_is_unreadable_not_complete(partial_db, tmp_path):
    """What a kill during a commit leaves on disk."""
    broken = str(tmp_path / 'truncated.db')
    shutil.copy(partial_db, broken)
    with open(broken, 'r+b') as handle:
        handle.truncate(200)

    with pytest.raises(RunStatusUnreadable):
        read_run_status(broken)
    assert run_is_complete(broken) is False
    with pytest.raises(DataIntegrityError):
        assert_run_complete(broken)


def test_a_file_that_is_not_a_database_at_all_is_unreadable(tmp_path):
    impostor = tmp_path / 'measurements.db'
    impostor.write_text('this is not a database\n', encoding='utf-8')
    with pytest.raises(RunStatusUnreadable):
        read_run_status(impostor)
    assert run_is_complete(impostor) is False


# ---------------------------------------------------------------------------
# 4. the sidecar flavour gets the same treatment
# ---------------------------------------------------------------------------

def test_a_readable_sidecar_is_unaffected(tmp_path):
    csv = tmp_path / 'reads.csv'
    csv.write_text('a,b\n1,2\n', encoding='utf-8')
    stamp_partial(csv, name='sequencing')
    records = read_run_status(csv)
    assert len(records) == 1 and records[0]['n_failed'] == 1
    assert run_is_complete(csv) is False


def test_a_missing_sidecar_still_reads_as_no_information(tmp_path):
    csv = tmp_path / 'reads.csv'
    csv.write_text('a,b\n1,2\n', encoding='utf-8')
    assert read_run_status(csv) == []
    assert run_is_complete(csv) is True


def test_a_half_written_sidecar_is_unreadable_not_complete(tmp_path):
    csv = tmp_path / 'reads.csv'
    csv.write_text('a,b\n1,2\n', encoding='utf-8')
    stamp_partial(csv, name='sequencing')
    sidecar = tmp_path / 'reads.run_status.json'
    text = sidecar.read_text(encoding='utf-8')
    assert json.loads(text)                       # it was valid a moment ago
    sidecar.write_text(text[:len(text) // 2], encoding='utf-8')

    with pytest.raises(RunStatusUnreadable):
        read_run_status(csv)
    assert run_is_complete(csv) is False


# ---------------------------------------------------------------------------
# 5. the queue keeps running past an artifact it cannot read
# ---------------------------------------------------------------------------

def test_the_batch_queue_survives_an_unreadable_artifact(partial_db, tmp_path):
    """``spacr.batch`` wraps both reads; a raise must not stop the queue."""
    from spacr import batch

    settings = {'src': os.path.dirname(os.path.dirname(partial_db))}
    with held_lock(partial_db):
        before = batch._status_snapshot(settings)
        collected = batch._collect_run_status(settings, before)
    assert isinstance(before, dict)
    assert collected is None or isinstance(collected, dict)

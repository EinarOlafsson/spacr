"""Guards and fallbacks in ``spacr.utils`` that ordinary runs never reach.

The functions here sit under a whole pipeline: a settings file written beside a
finished run, the per-field INSERT into ``measurements.db``, the embedding a
UMAP screen is plotted from. Each carries a branch for something that is not
supposed to happen -- a numpy value JSON cannot hold, a duplicated measurement
column, a GPU array handed back where a numpy one was expected -- and every one
of those branches decides whether a run ends with a note or with a traceback.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import utils


# ---------------------------------------------------------------------------
# The deferred-module proxy
# ---------------------------------------------------------------------------

def test_a_deferred_module_says_whether_it_has_been_imported_yet():
    """Its repr is the only way to see whether the import has happened.

    The proxy exists so that importing ``spacr.utils`` does not drag in
    Cellpose and its model stack; a repr that triggered the import would
    defeat the whole point of it.
    """
    proxy = utils._DeferredModule('json')

    assert repr(proxy) == "<deferred module 'json' (not yet imported)>"
    assert proxy.dumps([1]) == '[1]'
    assert repr(proxy) == "<deferred module 'json' (loaded)>"


def test_writing_through_a_deferred_module_sets_the_real_module(monkeypatch):
    """An assignment imports the module and lands on it, not on the proxy."""
    import sys
    import types

    fake = types.ModuleType('spacr_deferred_target')
    monkeypatch.setitem(sys.modules, 'spacr_deferred_target', fake)
    proxy = utils._DeferredModule('spacr_deferred_target')

    proxy.answer = 42

    assert fake.answer == 42
    assert 'answer' not in proxy.__dict__
    assert proxy.answer == 42


def test_the_legacy_skimage_square_is_used_when_the_new_name_is_absent(
        monkeypatch):
    """Without ``footprint_rectangle`` the helper still returns a square.

    scikit-image 0.22-0.24 spells it ``square``; the compatibility branch is
    only defined on those installations, so reaching it here needs the new
    name taken away and the module re-imported. The element it produces has to
    be identical, or dilation quietly changes shape between installations.
    """
    import importlib

    from skimage import morphology

    monkeypatch.delattr(morphology, 'footprint_rectangle', raising=False)
    reloaded = importlib.reload(utils)
    try:
        footprint = np.asarray(reloaded._square_footprint(3))
    finally:
        monkeypatch.undo()
        importlib.reload(utils)

    assert footprint.shape == (3, 3)
    assert np.all(footprint)


def test_the_square_footprint_helper_returns_a_square():
    """Morphology calls need a square structuring element on either skimage.

    ``footprint_rectangle`` replaced ``square`` in scikit-image 0.25; both
    spellings have to produce the same element or dilation silently changes
    shape between installations.
    """
    footprint = utils._square_footprint(3)

    assert np.asarray(footprint).shape == (3, 3)
    assert np.all(np.asarray(footprint))


# ---------------------------------------------------------------------------
# The JSON settings sibling
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The measurement-database append
# ---------------------------------------------------------------------------

def test_a_nul_in_a_table_name_is_refused_before_it_reaches_sqlite():
    """NUL cannot occur in an SQLite identifier, and says so here."""
    with pytest.raises(ValueError, match='NUL'):
        utils._sqlite_identifier('cell\x00measurements')


def test_a_timedelta_is_stored_as_nanoseconds_like_pandas_does():
    """Matching ``to_sql`` matters: both writers reach the same table."""
    assert utils._sqlite_value(pd.Timedelta(1, unit='s')) == 1_000_000_000


def test_an_empty_frame_writes_nothing_at_all():
    """A field with no objects must not open a transaction on the shared db."""
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE cell (a INTEGER)')
    try:
        utils._insert_frame(conn, 'cell', pd.DataFrame({'a': []}))
        assert conn.execute('SELECT count(*) FROM cell').fetchone()[0] == 0
    finally:
        conn.close()


def test_duplicate_measurement_columns_are_refused_by_name():
    """Two columns of one name would silently write one and drop the other."""
    frame = pd.DataFrame([[1, 2]], columns=['a', 'a'])
    conn = sqlite3.connect(':memory:')
    try:
        with pytest.raises(ValueError, match='duplicate names'):
            utils._insert_frame(conn, 'cell', frame)
    finally:
        conn.close()


def test_an_append_error_that_is_not_a_schema_problem_is_re_raised(
        monkeypatch):
    """Only a missing table or a missing column is repaired, nothing else.

    Retrying a locked database four times inside the append would hide a real
    contention problem behind a delayed, confusing failure; the error the
    caller needs to see is the first one.
    """
    def locked(conn, table, frame):
        raise sqlite3.OperationalError('database is locked')

    monkeypatch.setattr(utils, '_insert_frame', locked)
    conn = sqlite3.connect(':memory:')
    try:
        with pytest.raises(sqlite3.OperationalError, match='locked'):
            utils._append_frame(conn, 'cell', pd.DataFrame({'a': [1]}))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Reduction and clustering
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Metadata merge
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Adjusting cell masks in parallel
# ---------------------------------------------------------------------------


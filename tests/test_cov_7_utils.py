"""Guards and fallbacks in ``spacr.utils`` that ordinary runs never reach.

The functions here sit under a whole pipeline: a settings file written beside a
finished run, the per-field INSERT into ``measurements.db``, the embedding a
UMAP screen is plotted from. Each carries a branch for something that is not
supposed to happen -- a numpy value JSON cannot hold, a duplicated measurement
column, a GPU array handed back where a numpy one was expected -- and every one
of those branches decides whether a run ends with a note or with a traceback.
"""

from __future__ import annotations

import builtins
import json
import os
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

def test_numpy_scalars_are_written_as_plain_json_numbers(tmp_path):
    """A numpy scalar in the settings must not become its repr.

    The JSON copy exists so a results folder can say exactly what produced it;
    ``"np.float64(0.5)"`` in place of ``0.5`` would make it unreadable by
    anything but a human.
    """
    path = str(tmp_path / 'settings.json')
    utils._save_settings_json({'thr': np.float64(0.5), 'n': np.int64(7)}, path)

    saved = json.loads(open(path, encoding='utf-8').read())
    assert saved == {'thr': 0.5, 'n': 7}


def test_a_settings_value_numpy_cannot_be_asked_about_falls_back_to_repr(
        tmp_path, monkeypatch):
    """Losing numpy mid-write costs the shape of one value, not the file.

    The JSON copy is written after a finished run. An import that fails there
    must leave the remaining settings readable rather than take the whole
    file down with it.
    """
    real_import = builtins.__import__

    def no_numpy(name, *args, **kwargs):
        if name == 'numpy':
            raise ImportError('numpy is gone')
        return real_import(name, *args, **kwargs)

    path = str(tmp_path / 'settings.json')
    monkeypatch.setattr(builtins, '__import__', no_numpy)
    try:
        utils._save_settings_json({'n': np.int64(7), 'src': 'here'}, path)
    finally:
        monkeypatch.undo()

    saved = json.loads(open(path, encoding='utf-8').read())
    assert saved['src'] == 'here'
    assert saved['n'] == repr(np.int64(7))


def test_a_settings_copy_that_cannot_be_written_is_a_note_not_a_failure(
        tmp_path, capsys):
    """A finished run is never lost over its settings sibling."""
    path = str(tmp_path / 'no_such_folder' / 'settings.json')

    utils._save_settings_json({'src': 'here'}, path)

    assert 'could not write' in capsys.readouterr().out
    assert not os.path.exists(path)


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

class _GpuArray:
    """Stands in for a cuML result, which is returned as a device array."""

    def __init__(self, values):
        self._values = np.asarray(values)

    def get(self):
        return self._values


class _SealedReducer:
    """A reducer that returns a device array and accepts no new attributes.

    Both are real cuML behaviours: the embedding comes back on the device, and
    some estimators are slotted, so the backend tag spaCR wants to record on
    the fitted reducer cannot be written.
    """

    __slots__ = ()

    def fit_transform(self, values):
        return _GpuArray(np.zeros((len(values), 2), float))


def _reduce(values, **kwargs):
    """``reduction_and_clustering`` with the positional arguments filled in."""
    return utils.reduction_and_clustering(
        values, 5, 0.1, 'euclidean', 0.5, 2, 'dbscan', **kwargs)


def test_a_device_embedding_is_brought_back_to_the_host(monkeypatch):
    """A GPU array reaches clustering as numpy, and a sealed reducer is fine.

    ``DBSCAN.fit`` cannot read a device array, and failing to record the
    backend on the reducer is cosmetic -- neither may end the run.
    """
    from spacr import gpu_reduce

    monkeypatch.setattr(gpu_reduce, 'make_reducer',
                        lambda name, prefer_gpu=False, **kwargs:
                        (_SealedReducer(), 'cpu'))
    values = np.random.default_rng(0).normal(size=(6, 3))

    embedding, labels, reducer = _reduce(values, reduction_method='umap')

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (6, 2)
    assert len(labels) == 6
    assert isinstance(reducer, _SealedReducer)


def test_a_perplexity_of_zero_is_refused_before_t_sne_runs():
    """t-SNE with a non-positive perplexity is a settings error, not a crash."""
    values = np.random.default_rng(0).normal(size=(6, 3))

    with pytest.raises(ValueError, match='perplexity'):
        _reduce(values, reduction_method='tsne',
                reducer_options={'perplexity': 0})


def test_a_perplexity_larger_than_the_data_is_lowered_and_reported(capsys):
    """A row limit can leave fewer rows than the saved perplexity.

    Failing after the data has been loaded would waste the whole run, so the
    largest valid neighbourhood is used instead -- and the adjustment is
    printed, because a plot made at a different perplexity than the settings
    file records is otherwise untraceable.
    """
    values = np.random.default_rng(0).normal(size=(5, 3))

    embedding, _labels, _reducer = _reduce(
        values, reduction_method='tsne', verbose=True,
        reducer_options={'perplexity': 30.0})

    assert embedding.shape == (5, 2)
    printed = capsys.readouterr().out
    assert 'Adjusted t-SNE perplexity from 30 to 4' in printed


def test_a_reducer_that_cannot_transform_says_which_setting_to_change():
    """t-SNE has no ``transform``, and the message names the way out."""
    values = np.random.default_rng(0).normal(size=(6, 3))

    with pytest.raises(ValueError, match='embedding_by_controls'):
        _reduce(values, reduction_method='tsne', mode='transform',
                model=object())


def test_a_device_embedding_from_a_saved_model_is_brought_back_too():
    """The transform path returns numpy whatever the fitted model returns."""
    class _Fitted:
        def transform(self, values):
            return _GpuArray(np.zeros((len(values), 2), float))

    values = np.random.default_rng(0).normal(size=(6, 3))

    embedding, labels, reducer = _reduce(values, mode='transform',
                                         model=_Fitted())

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (6, 2)
    assert len(labels) == 6
    assert isinstance(reducer, _Fitted)


# ---------------------------------------------------------------------------
# Metadata merge
# ---------------------------------------------------------------------------

def test_a_metadata_file_with_no_identifier_column_lists_what_it_has(tmp_path):
    """The identifier column is detected, and a miss names every candidate.

    Hard-coding ``Gene ID`` killed any other annotation table with a KeyError
    naming a column the user never claimed to have, after the whole regression
    had already run.
    """
    results = tmp_path / 'results.csv'
    metadata = tmp_path / 'metadata.csv'
    pd.DataFrame({'feature': ['C(gene)[T.TGGT1_225160_2]'],
                  'coefficient': [0.5]}).to_csv(results, index=False)
    pd.DataFrame({'description': ['a kinase'], 'notes': ['x']}).to_csv(
        metadata, index=False)

    with pytest.raises(ValueError, match='no column holding a gene'):
        utils.merge_regression_res_with_metadata(str(results), str(metadata))


# ---------------------------------------------------------------------------
# Adjusting cell masks in parallel
# ---------------------------------------------------------------------------

def _mask_set(root):
    """Two fields of parasite / cell / nuclei masks that merge cleanly."""
    folders = {}
    for kind in ('parasite', 'cell', 'nuclei'):
        folder = root / kind
        folder.mkdir()
        folders[kind] = str(folder)
    for index in range(2):
        cell = np.zeros((16, 16), np.uint16)
        cell[2:8, 2:8] = 1
        cell[2:8, 9:15] = 2
        nuclei = np.zeros((16, 16), np.uint16)
        nuclei[3:5, 3:5] = 1
        nuclei[3:5, 10:12] = 2
        parasite = np.zeros((16, 16), np.uint16)
        parasite[5:11, 5:12] = 1
        name = f'field{index}.npy'
        np.save(os.path.join(folders['cell'], name), cell)
        np.save(os.path.join(folders['nuclei'], name), nuclei)
        np.save(os.path.join(folders['parasite'], name), parasite)
    return folders


@pytest.mark.integration
def test_adjusting_cell_masks_across_a_pool_rewrites_every_field(tmp_path,
                                                                capsys):
    """The parallel path has to write the same masks the inline path does.

    ``adjust_cell_masks`` overwrites the cell masks in place, so a worker that
    returned without saving would leave a folder half-merged and the run would
    measure two different mask conventions in one table.
    """
    folders = _mask_set(tmp_path)
    before = [np.load(os.path.join(folders['cell'], f'field{i}.npy')).copy()
              for i in range(2)]

    utils.adjust_cell_masks(folders['parasite'], folders['cell'],
                            folders['nuclei'], n_jobs=2)

    after = [np.load(os.path.join(folders['cell'], f'field{i}.npy'))
             for i in range(2)]
    assert all(np.array_equal(after[0], plane) for plane in after)
    assert len(np.unique(after[0])) <= len(np.unique(before[0]))
    assert 'adjust_cell_masks' in capsys.readouterr().out

"""A store hands identifiers back as integers; they must never be features.

SQLite held ``nucleus_nucleus``, ``cell_id`` and the key columns as text, so
a numeric dtype never put them among the Classify features. The store reads
them as integers, so the shared role rule
(:func:`spacr.schema.is_provenance_column`) has to keep them out by name.
"""
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import measurement_store as ms
from spacr import schema

IDENTIFIERS = ('nucleus_nucleus', 'pathogen_pathogen', 'cell_cell', 'cell_id',
               'nucleus_cell_id', 'object_key', 'object_id', 'prcfo', 'prcf',
               'plateID', 'rowID', 'columnID', 'fieldID', 'timeID',
               'object_label', 'plate_key', 'well_key', 'field_key',
               'parent_cell_key', 'parent_nucleus_key')


def _nuclei(n=40, text_ids=False):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        'object_label': np.arange(1, n + 1), 'plateID': 'plate1',
        'rowID': 'r1', 'columnID': 'c1', 'fieldID': 'f1',
        'prcf': 'plate1_r1_c1_f1', 'file_name': 'plate1_r1c1_1',
        'nucleus_nucleus': np.arange(1, n + 1),
        'nucleus_cell_id': np.arange(1, n + 1) % 7,
        'nucleus_area': rng.uniform(50, 90, n),
        'nucleus_channel_0_mean_intensity': rng.uniform(1, 9, n),
        'nucleus_channel_1_mean_intensity': rng.uniform(1, 9, n),
    })
    frame['prcfo'] = frame['prcf'] + '_o' + frame['object_label'].astype(str)
    if text_ids:
        for column in ('nucleus_nucleus', 'nucleus_cell_id', 'object_label'):
            frame[column] = frame[column].astype(str)
    return frame


@pytest.mark.parametrize('name', IDENTIFIERS)
def test_identifier_and_key_names_are_provenance(name):
    assert schema.is_provenance_column(name)


@pytest.mark.parametrize('name', ('nucleus_area', 'cell_channel_1_mean_intensity',
                                  'pathogen_channel_2_std_intensity'))
def test_measurements_are_not_provenance(name):
    assert not schema.is_provenance_column(name)


def _features(frame):
    from spacr.utils import filter_dataframe_features
    _frame, features = filter_dataframe_features(
        frame, channel_of_interest=None, exclude=None,
        remove_low_variance_features=False,
        remove_highly_correlated_features=False)
    return features


def test_a_store_training_pull_has_no_identifier_among_its_features(tmp_path):
    from spacr.io import _read_db
    folder = tmp_path / 'store' / 'measurements'
    db = folder / 'measurements.db'
    ms.ensure_store(db)
    ms.write_rows(str(folder), 'nucleus', _nuclei())
    pulled = _read_db(str(db), ['nucleus'])[0]
    assert pd.api.types.is_integer_dtype(pulled['nucleus_nucleus'])

    legacy_dir = tmp_path / 'legacy' / 'measurements'
    legacy_dir.mkdir(parents=True)
    with sqlite3.connect(legacy_dir / 'measurements.db') as conn:
        _nuclei(text_ids=True).to_sql('nucleus', conn, index=False)
    legacy = _read_db(str(legacy_dir / 'measurements.db'), ['nucleus'])[0]

    from_store = _features(pulled)
    assert from_store == _features(legacy)
    assert not [c for c in from_store if schema.is_provenance_column(c)]
    assert 'nucleus_nucleus' not in from_store
    assert 'nucleus_area' in from_store
    ms.forget_sessions()


def test_unknown_numeric_paths_also_drop_identifiers():
    frame = _nuclei()
    frame['object_key'] = np.arange(len(frame))
    frame['parent_cell_key'] = np.arange(len(frame))
    kept = schema.model_feature_columns(frame, allow_unknown=True)
    assert not [c for c in kept if schema.is_provenance_column(c)]
    from spacr.active_learning import _feature_columns
    assert not set(_feature_columns(frame.columns)) & {
        'object_key', 'parent_cell_key', 'nucleus_nucleus'}

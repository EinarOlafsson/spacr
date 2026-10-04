"""Crop enrichment preserves physical object identity and read-only sources."""
import hashlib
import sqlite3

import pandas as pd
import pytest

from spacr.png_list import _attach_object_crop_paths



def _paths(result):
    """The joined paths with every missing value as ``None``.

    pandas 3 returns a missing string as NaN where pandas 2 kept ``None``;
    an unmatched row is what is checked, not which marker pandas uses.
    """
    import pandas as pd

    return [None if pd.isna(value) else value
            for value in result['png_path'].tolist()]

def _objects():
    """Same object labels reused across distinct fields and plates."""
    return pd.DataFrame({
        'plateID': ['P1', 'P1', 'P2', 'P1'],
        'rowID': ['r1'] * 4, 'columnID': ['c1'] * 4,
        'fieldID': [1, 2, 1, 1], 'object_label': [1, 1, 1, 2],
        'measurement': [10., 20., 30., 40.],
    }, index=[9, 3, 9, 1])


def _crops(frame):
    """Matching crop rows with the stored crop-mode identifier convention."""
    crops = frame.drop(columns=['object_label', 'measurement']).reset_index(drop=True)
    crops['cell_id'] = ['o' + str(value) for value in frame['object_label']]
    crops['png_path'] = ['first.png', 'second.png', 'third.png', 'fourth.png']
    return crops


def _database(tmp_path, crops):
    """A real SQLite file with an unrelated table that must remain unchanged."""
    path = tmp_path / 'measurements #1.db'
    with sqlite3.connect(path) as connection:
        crops.to_sql('png_list', connection, index=False)
        connection.execute('CREATE TABLE user_data (value TEXT)')
        connection.execute("INSERT INTO user_data VALUES ('untouched')")
    return path


def test_full_identity_order_indices_and_read_only_database(tmp_path, monkeypatch):
    frame = _objects()
    frame.attrs['source'] = 'keep'
    original = frame.copy(deep=True)
    db = _database(tmp_path, _crops(frame))
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    original_connect = sqlite3.connect
    calls = []

    def capture(*args, **kwargs):
        """Record the connection mode while retaining the real SQLite reader."""
        calls.append((args, kwargs))
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, 'connect', capture)
    result = _attach_object_crop_paths(db, frame, 'cell')
    assert result['png_path'].tolist() == ['first.png', 'second.png', 'third.png', 'fourth.png']
    pd.testing.assert_frame_equal(result.drop(columns='png_path'), original)
    pd.testing.assert_frame_equal(frame, original)
    assert result.attrs == original.attrs
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    assert len(calls) == 1
    assert calls[0][0][0].endswith('?mode=ro') and calls[0][1]['uri'] is True


def test_duplicates_same_path_collapse_conflicts_do_not_choose(tmp_path, caplog):
    frame = _objects()
    crops = _crops(frame)
    identical = crops.iloc[[0]].copy()
    conflict = crops.iloc[[1]].copy()
    conflict['png_path'] = 'another.png'
    db = _database(tmp_path, pd.concat([crops, identical, conflict], ignore_index=True))
    result = _attach_object_crop_paths(db, frame, 'cell')
    assert _paths(result) == ['first.png', None, 'third.png', 'fourth.png']
    assert len(result) == len(frame)
    assert 'conflicting paths' in caplog.text


def test_existing_nonempty_paths_win_but_blank_and_missing_are_filled(tmp_path):
    frame = _objects()
    crops = _crops(frame)
    frame['png_path'] = ['chosen.png', '', pd.NA, '   ']
    result = _attach_object_crop_paths(_database(tmp_path, crops), frame, 'cell')
    assert result['png_path'].tolist() == ['chosen.png', 'second.png', 'third.png', 'fourth.png']
    assert frame['png_path'].iloc[0] == 'chosen.png'
    assert pd.isna(frame['png_path'].iloc[2])


def test_time_aliases_numeric_types_and_missing_values(tmp_path):
    frame = _objects()
    frame.loc[:, 'fieldID'] = 1
    frame.loc[:, 'plateID'] = 'P1'
    frame.loc[:, 'object_label'] = 1
    frame['timeID'] = [1., 2., None, 4.]
    crops = _crops(frame).drop(columns='timeID')
    crops['time_id'] = ['t1', 't2', None, 't4']
    result = _attach_object_crop_paths(_database(tmp_path, crops), frame, 'cell')
    assert _paths(result) == ['first.png', 'second.png', None, 'fourth.png']


@pytest.mark.parametrize('which', ['objects', 'crops'])
def test_one_sided_time_never_matches_across_frames(tmp_path, which, caplog):
    frame = _objects()
    crops = _crops(frame)
    (frame if which == 'objects' else crops)['timeID'] = 't1'
    result = _attach_object_crop_paths(_database(tmp_path, crops), frame, 'cell')
    assert result is frame
    assert 'timepoint identities differ' in caplog.text


@pytest.mark.parametrize('which', ['objects', 'crops'])
def test_conflicting_time_aliases_do_not_attach(tmp_path, which, caplog):
    frame = _objects()
    frame['timeID'] = 1
    crops = _crops(frame)
    (frame if which == 'objects' else crops)['time_id'] = 't2'
    result = _attach_object_crop_paths(_database(tmp_path, crops), frame, 'cell')
    assert result is frame
    assert 'conflicting timepoint aliases' in caplog.text


def test_other_crop_modes_invalid_labels_and_missing_keys_do_not_match(tmp_path):
    frame = _objects()
    frame['object_label'] = [1, 1, 0, -1]
    crops = _crops(frame)
    crops['cell_id'] = [None, 'omulti', 'o0', 'o-1']
    crops['nucleus_id'] = 'o1'
    result = _attach_object_crop_paths(_database(tmp_path, crops), frame, 'cell')
    assert result is frame


@pytest.mark.parametrize('object_type', ['nucleus', 'pathogen', 'cytoplasm', 'organelle'])
def test_crop_mode_matches_its_own_object_table(tmp_path, object_type):
    frame = _objects()
    crops = _crops(frame).rename(columns={'cell_id': object_type + '_id'})
    result = _attach_object_crop_paths(_database(tmp_path, crops), frame, object_type)
    assert result['png_path'].tolist() == crops['png_path'].tolist()


@pytest.mark.parametrize('missing', ['plateID', 'rowID', 'columnID', 'fieldID', 'object_label'])
def test_incomplete_measurement_identity_is_not_guessed(tmp_path, missing):
    frame = _objects()
    db = _database(tmp_path, _crops(frame))
    frame = frame.drop(columns=missing)
    assert _attach_object_crop_paths(db, frame, 'cell') is frame


def test_missing_png_table_and_unknown_table_preserve_frame(tmp_path):
    frame = _objects()
    db = tmp_path / 'empty.db'
    sqlite3.connect(db).close()
    assert _attach_object_crop_paths(db, frame, 'cell') is frame
    assert _attach_object_crop_paths(db, frame, 'custom_join') is frame


def test_null_metadata_keys_do_not_match_each_other(tmp_path):
    frame = _objects()
    frame['fieldID'] = None
    db = _database(tmp_path, _crops(frame))
    assert _attach_object_crop_paths(db, frame, 'cell') is frame


def test_time_normalization_never_strips_a_non_numeric_identifier(tmp_path):
    frame = _objects()
    frame['timeID'] = ['t01', 'time1', 't3', None]
    crops = _crops(frame)
    crops['timeID'] = ['1', 'ime1', '003', None]
    result = _attach_object_crop_paths(_database(tmp_path, crops), frame, 'cell')
    assert _paths(result) == ['first.png', None, 'third.png', None]


def test_multiple_mode_ids_without_explicit_mode_never_guess_crop_role(tmp_path, caplog):
    frame = _objects()
    crops = _crops(frame)
    crops['nucleus_id'] = ['o7'] * len(crops)
    db = _database(tmp_path, crops)
    assert _attach_object_crop_paths(db, frame, 'cell') is frame
    assert 'ambiguous or conflicting crop modes' in caplog.text


def test_explicit_nucleus_crop_with_parent_cell_id_only_matches_nucleus(tmp_path):
    frame = _objects()
    crops = _crops(frame)
    crops['nucleus_id'] = crops['cell_id']
    crops['crop_mode'] = 'nucleus'
    db = _database(tmp_path, crops)
    assert _attach_object_crop_paths(db, frame, 'cell') is frame
    result = _attach_object_crop_paths(db, frame, 'nucleus')
    assert result['png_path'].tolist() == crops['png_path'].tolist()


@pytest.mark.parametrize('mode,other,matched', [
    ('cell', 'cell', True), ('cell', 'nucleus', False),
    ('unknown', 'cell', False), (None, None, False),
    ('', '', False), ('cell', None, False), (None, 'cell', False),
])
def test_explicit_crop_mode_aliases_must_agree(tmp_path, mode, other, matched):
    frame = _objects()
    crops = _crops(frame)
    crops['nucleus_id'] = 'o7'
    crops['crop_mode'] = mode
    crops['object_type'] = other
    result = _attach_object_crop_paths(_database(tmp_path, crops), frame, 'cell')
    if matched:
        assert result['png_path'].tolist() == crops['png_path'].tolist()
    else:
        assert result is frame

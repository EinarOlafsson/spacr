import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr._lineage_trees import _lineage_segments
from spacr.timelapse import _event_correct_divisions


def _events():
    return pd.DataFrame(columns=['track_id', 'frame', 'event'])


@pytest.mark.parametrize('dtype,missing,as_string', [
    ('object', None, False),
    ('object', pd.NA, False),
    ('Int64', pd.NA, False),
    ('object', None, True),
])
def test_empty_event_correction_preserves_exact_native_ancestry(
        tmp_path, dtype, missing, as_string):
    mother = 2**53 + 1
    child = mother + 2
    declared = str(mother) if as_string else mother
    tracks = pd.DataFrame({
        'track_id': [mother, mother, child, child],
        'frame': [0, 1, 2, 3], 'x': [10., 10., 11., 11.], 'y': [10.] * 4,
        'parent_track_id': pd.Series([missing, missing, declared, declared],
                                     dtype=dtype),
        'parent_track_id_source': ['trackastra'] * 4,
    })
    events = _events()
    original_frame = tracks.copy(deep=True)
    original_events = pickle.dumps(events)
    path = tmp_path / 'declared-tracks.csv'
    tracks.to_csv(path, index=False)
    stored = path.read_bytes()
    original = pickle.dumps(tracks)
    corrected, added = _event_correct_divisions(tracks, events)
    assert corrected.parent_track_id.tolist() == [0, 0, mother, mother]
    segments = _lineage_segments(corrected)
    daughter = segments.loc[segments.track_id.eq(child)].iloc[0]
    parent = segments.loc[segments.segment_id.eq(daughter.parent_segment_id)].iloc[0]
    assert parent.track_id == mother
    assert corrected.parent_track_id_source.tolist() == ['trackastra'] * 4
    assert added.empty
    pd.testing.assert_frame_equal(tracks, original_frame)
    assert pickle.dumps(tracks) == original
    assert pickle.dumps(events) == original_events
    assert path.read_bytes() == stored


@pytest.mark.parametrize('values,expected', [
    ([None, '', 'bad', '0'], [0, 0, 0, 0]),
    ([-1, '-2', 1.75, '2.5'], [-1, -2, 1, 2]),
    ([True, np.bool_(False), pd.NA, np.nan], [1, 0, 0, 0]),
    ([0, 0, 2**53 + 1, 2**53 + 1], [0, 0, 2**53 + 1, 2**53 + 1]),
])
def test_existing_scalar_coercion_policy_is_preserved(values, expected):
    tracks = pd.DataFrame({
        'track_id': [1] * len(values), 'frame': range(len(values)),
        'x': [10.] * len(values), 'y': [10.] * len(values),
        'parent_track_id': pd.Series(values, dtype=object),
    })
    original = pickle.dumps(tracks)
    fixed, added = _event_correct_divisions(tracks, _events())
    assert fixed.parent_track_id.tolist() == expected
    assert pd.api.types.is_integer_dtype(fixed.parent_track_id.dtype)
    assert added.empty
    assert pickle.dumps(tracks) == original


@pytest.mark.parametrize('value', [np.inf, -np.inf, 'Inf', '-Infinity'])
def test_nonfinite_stored_parent_retains_integer_conversion_refusal(value):
    tracks = pd.DataFrame({'track_id': [1], 'frame': [0], 'x': [10.],
                           'y': [10.], 'parent_track_id': [value]})
    original = pickle.dumps(tracks)
    with pytest.raises(pd.errors.IntCastingNaNError,
                       match='Cannot convert non-finite values'):
        _event_correct_divisions(tracks, _events())
    assert pickle.dumps(tracks) == original


@pytest.mark.parametrize('mother', [2**53 + 1, 2**63 - 1])
def test_native_csv_event_and_lineage_roundtrip_keeps_signed64_parent(tmp_path, mother):
    from spacr.timelapse import _event_field_inputs
    from spacr._lineage_trees import _lineage_trees_from_tracks
    from spacr.tabular import read_table, write_table

    child = mother + 2 if mother < 2**63 - 2 else mother - 2
    tracks = pd.DataFrame({
        'track_id': [mother, mother, child, child], 'frame': [0, 1, 2, 3],
        'x': [10., 10., 11., 11.], 'y': [10.] * 4,
        'parent_track_id': pd.Series([None, None, mother, mother], dtype=object),
        'parent_track_id_source': ['trackastra'] * 4,
    })
    path = tmp_path / 'native_tracks_cell_r1c1f1.csv'
    write_table(tracks, path, canonicalise=False)
    original = path.read_bytes()
    loaded, crops = _event_field_inputs(str(path), 30.)
    assert crops is None
    corrected, added = _event_correct_divisions(loaded, _events())
    assert corrected.loc[corrected.track_id.eq(child), 'parent_track_id'].tolist() == [mother, mother]
    assert added.empty
    corrected_path = tmp_path / 'corrected.csv'
    write_table(corrected, corrected_path, canonicalise=False)
    result = _lineage_trees_from_tracks(str(corrected_path), plot=False, max_distance=-1.)
    segments = result['segments']
    daughter = segments.loc[segments.track_id.eq(child)].iloc[0]
    parent_id = daughter.parent_segment_id
    assert segments.loc[segments.segment_id.eq(parent_id), 'track_id'].iat[0] == mother
    persisted = read_table(result['paths']['segments'], report=None)
    assert set(persisted.track_id) == {mother, child}
    newick = Path(result['paths']['newick']).read_text()
    assert f't{mother}_f0:' in newick
    assert f't{child}_f2:' in newick
    assert path.read_bytes() == original


@pytest.mark.parametrize('parents', [
    [None, '', 'bad', '0'],
    ['-1', '-2', '1.75', '2.5'],
    ['True', 'False', 'TRUE', 'false'],
    ['True', 'bad', 'False', ''],
])
def test_native_csv_parent_scalar_policy_matches_existing_reader(tmp_path, parents):
    from spacr.timelapse import _event_field_inputs
    from spacr.tabular import read_table, write_table

    tracks = pd.DataFrame({'track_id': [1] * len(parents), 'frame': range(len(parents)),
                           'x': [10.] * len(parents), 'y': [10.] * len(parents),
                           'parent_track_id': parents})
    path = tmp_path / 'tracks.csv'
    write_table(tracks, path, canonicalise=False)
    expected, _ = _event_correct_divisions(read_table(path, report=None), _events())
    loaded, _ = _event_field_inputs(str(path), 30.)
    actual, _ = _event_correct_divisions(loaded, _events())
    assert actual.parent_track_id.tolist() == expected.parent_track_id.tolist()


@pytest.mark.parametrize('value', ['Inf', '-Infinity'])
def test_native_csv_nonfinite_parent_still_refuses(tmp_path, value):
    from spacr.timelapse import _event_field_inputs
    from spacr.tabular import write_table

    path = tmp_path / 'tracks.csv'
    write_table(pd.DataFrame({'track_id': [1], 'frame': [0], 'x': [10.], 'y': [10.],
                              'parent_track_id': [value]}), path, canonicalise=False)
    loaded, _ = _event_field_inputs(str(path), 30.)
    with pytest.raises(pd.errors.IntCastingNaNError, match='Cannot convert non-finite values'):
        _event_correct_divisions(loaded, _events())


def test_native_csv_out_of_signed64_wrap_remains_unsupported(tmp_path):
    from spacr.timelapse import _event_field_inputs
    from spacr.tabular import write_table

    path = tmp_path / 'tracks.csv'
    write_table(pd.DataFrame({'track_id': [1, 1], 'frame': [0, 1], 'x': [10.] * 2,
                              'y': [10.] * 2, 'parent_track_id': [None, str(2**63)]}),
                path, canonicalise=False)
    loaded, _ = _event_field_inputs(str(path), 30.)
    corrected, _ = _event_correct_divisions(loaded, _events())
    assert corrected.parent_track_id.tolist() == [0, -(2**63)]

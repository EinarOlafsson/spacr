import pickle

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

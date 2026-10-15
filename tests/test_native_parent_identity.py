import pickle

import numpy as np
import pandas as pd
import pytest

from spacr._lineage_trees import _lineage_segments
from spacr.timelapse import _native_lineage_columns


def test_empty_native_table_retains_integer_parent_and_empty_provenance():
    tracks = pd.DataFrame({'track_id': pd.Series([], dtype='int64'),
                           'frame': pd.Series([], dtype='int64'),
                           'x': pd.Series([], dtype=float),
                           'y': pd.Series([], dtype=float)})
    original = pickle.dumps(tracks)
    result = _native_lineage_columns(tracks, {}, 'declared-native')
    assert result.empty
    assert pd.api.types.is_integer_dtype(result.parent_track_id.dtype)
    assert result.parent_track_id_source.empty
    assert pickle.dumps(tracks) == original


@pytest.mark.parametrize('storage', ['integer', 'numpy_integer', 'integer_string'])
@pytest.mark.parametrize('declare_root', [False, True])
def test_native_known_parent_and_actual_lineage_keep_exact_signed_identity(
        storage, declare_root):
    mother = 2**53 + 1
    daughter = mother + 2
    parent = {'integer': mother, 'numpy_integer': np.int64(mother),
              'integer_string': str(mother)}[storage]
    tracks = pd.DataFrame({'track_id': [mother, mother, daughter, daughter],
                           'frame': [0, 1, 2, 3], 'x': [10., 10., 11., 11.],
                           'y': [10.] * 4})
    parents = {daughter: [parent, parent]}
    if declare_root:
        parents[mother] = 0
    original = pickle.dumps(tracks)
    original_parents = pickle.dumps(parents)
    result = _native_lineage_columns(tracks, parents, 'declared-native')
    assert result.parent_track_id.tolist() == [0, 0, mother, mother]
    assert result.parent_track_id_source.tolist() == [
        'declared-native' if declare_root else '',
        'declared-native' if declare_root else '',
        'declared-native', 'declared-native']
    lineage = _lineage_segments(result)
    child = lineage.loc[lineage.track_id.eq(daughter)].iloc[0]
    actual_parent = lineage.loc[lineage.segment_id.eq(child.parent_segment_id)].iloc[0]
    assert actual_parent.track_id == mother
    assert pickle.dumps(tracks) == original
    assert pickle.dumps(parents) == original_parents


@pytest.mark.parametrize('parent', [1.5, np.float64(1.5)])
def test_native_fractional_parent_retains_existing_refusal_without_mutation(parent):
    tracks = pd.DataFrame({'track_id': [1, 2], 'frame': [0, 1],
                           'x': [10., 11.], 'y': [10., 10.]})
    parents = {2: parent}
    original = pickle.dumps(tracks)
    original_parents = pickle.dumps(parents)
    with pytest.raises(ValueError, match='non-integer parent for track 2'):
        _native_lineage_columns(tracks, parents, 'declared-native')
    assert pickle.dumps(tracks) == original
    assert pickle.dumps(parents) == original_parents

"""Explicit known ancestry cannot depend on the order of imported track rows."""
import json
from pathlib import Path

import pandas as pd
import pytest

from spacr import _lineage_trees as lineage

SOURCE = 'Track {track} has conflicting known parent IDs in the lineage table.'


@pytest.fixture(autouse=True)
def english(monkeypatch):
    monkeypatch.setenv('SPACR_LANGUAGE', 'en')


def metadata(values, *, large=False, dtype=object):
    a, b, child = (2**53 + 1, 2**53 + 3, 2**53 + 5) if large else (1, 2, 3)
    frame = pd.DataFrame({'track_id': pd.Series([child] * len(values), dtype='int64'),
                          'parent_track_id': pd.Series(values, dtype=dtype)})
    spans = pd.DataFrame({'start': [0, 0, 3], 'end': [2, 2, 5]},
                         index=pd.Index([a, b, child], dtype='int64'))
    return frame, spans, child


@pytest.mark.parametrize('storage', ['object', 'Int64', 'strings'])
@pytest.mark.parametrize('reverse', [False, True])
def test_conflicting_known_parents_refuse_without_lossy_large_identity_coercion(storage, reverse):
    a, b = 2**53 + 1, 2**53 + 3
    values = [None, str(a), str(b), str(b)] if storage == 'strings' else [None, a, b, b]
    frame, spans, child = metadata(values, large=True, dtype=object if storage == 'strings' else storage)
    if reverse:
        frame = frame.iloc[::-1].copy()
    before = frame.copy(deep=True)
    with pytest.raises(ValueError) as error:
        lineage._lineage_explicit_parents(frame, spans)
    assert str(error.value) == SOURCE.format(track=child)
    pd.testing.assert_frame_equal(frame, before)


def actual_tracks(reverse=False):
    rows = []
    for track in (1, 2):
        for frame in (0, 1, 2):
            rows.append(dict(track_id=track, frame=frame, x=10 * track, y=10,
                             parent_track_id=0, parent_track_id_source='trackastra'))
    for frame, parent in [(3, 1), (4, 2), (5, 2)]:
        rows.append(dict(track_id=3, frame=frame, x=15, y=10,
                         parent_track_id=parent, parent_track_id_source='trackastra'))
    result = pd.DataFrame(rows)
    return result.iloc[::-1].copy() if reverse else result


@pytest.mark.parametrize('reverse', [False, True])
def test_real_segment_boundary_refuses_instead_of_selecting_a_row_order_parent(reverse):
    frame = actual_tracks(reverse)
    before = frame.copy(deep=True)
    with pytest.raises(ValueError) as error:
        lineage._lineage_segments(frame)
    assert str(error.value) == SOURCE.format(track=3)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize('parents,expected', [
    ([1, 1, 1], {3: 1}), ([0, 0, 0], {}), ([None, None, None], {}),
    ([None, 1, 1], {3: 1}), ([99, 1, 1], {}), ([0, 1, 1], {}),
    ([3, 1, 1], {}), ([-1, 1, 1], {}), ([1, 1.25, 1], {3: 1}),
    (['1', 'not-an-id', '1'], {3: 1}),
])
def test_existing_first_parent_sentinel_and_identical_known_semantics_remain(parents, expected):
    frame, spans, _ = metadata(parents)
    before = frame.copy(deep=True)
    assert lineage._lineage_explicit_parents(frame, spans) == expected
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize('storage', ['object', 'Int64', 'strings'])
def test_large_identical_known_parent_preserves_exact_identity_with_missing_rows(storage):
    parent = 2**53 + 1
    values = [None, str(parent), str(parent)] if storage == 'strings' else [None, parent, parent]
    frame, spans, child = metadata(values, large=True, dtype=object if storage == 'strings' else storage)
    before = frame.copy(deep=True)
    assert lineage._lineage_explicit_parents(frame, spans) == {child: parent}
    pd.testing.assert_frame_equal(frame, before)


def test_absent_parent_column_still_has_no_explicit_links():
    frame, spans, _ = metadata([1, 1])
    assert lineage._lineage_explicit_parents(frame.drop(columns='parent_track_id'), spans) == {}


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is'])
def test_installed_reviewed_language_at_actual_segment_boundary(language, monkeypatch):
    review = Path(__file__).parents[1] / 'docs/i18n/reviewed/runtime' / language / '2026-10-15-lineage-parent-identity.json'
    record = json.loads(review.read_text(encoding="utf-8"))['records'][0]
    frame = actual_tracks()
    before = frame.copy(deep=True)
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    with pytest.raises(ValueError) as error:
        lineage._lineage_segments(frame)
    assert str(error.value) == record['translation'].format(track=3)
    assert str(error.value) != SOURCE.format(track=3)
    pd.testing.assert_frame_equal(frame, before)

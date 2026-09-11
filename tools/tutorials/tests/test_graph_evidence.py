"""Synthetic fixtures verify the evidence checks, not application results."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_evidence import check_points, check_histogram, check_brush


def test_points_preserve_duplicates_and_actual_values():
    expected = [(1, 2), (1, 2), (3, 4)]
    assert check_points(expected, list(reversed(expected))) == 3
    with pytest.raises(ValueError, match='multiplicities'):
        check_points(expected, [(1, 2), (3, 4)])
    with pytest.raises(ValueError, match='coordinates'):
        check_points(expected, [(1, 2), (1, 2), (3, 5)])


def test_histogram_counts_include_rightmost_endpoint():
    assert check_histogram([0, 1, 2, 2], [0, 1, 2], [1, 3]) == [1, 3]
    with pytest.raises(ValueError, match='bar counts'):
        check_histogram([0, 1, 2, 2], [0, 1, 2], [2, 2])
    with pytest.raises(ValueError, match='omit'):
        check_histogram([0, 3], [0, 1, 2], [1, 0])


def test_publication_does_not_prove_handoff():
    assert not check_brush(['a', 'b'], ['b', 'a'], 0, False)['annotation_handoff_works']
    with pytest.raises(ValueError, match='identities'):
        check_brush(['a', 'b'], ['a', 'c'], 0, False)
    with pytest.raises(ValueError, match='changed'):
        check_brush(['a', 'b'], ['a', 'b'], 2, True)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_missing_values_are_not_silently_dropped(bad):
    with pytest.raises(ValueError, match='Nonfinite'):
        check_points([(1, bad)], [(1, bad)])

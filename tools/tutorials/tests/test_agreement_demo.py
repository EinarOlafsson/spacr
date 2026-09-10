"""Pin the teaching labels and prove disagreement checks inspect their values."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agreement_demo import COLUMNS, LABELS, expected, verify


@pytest.fixture
def rows():
    return [dict(png_path=f'crop_{index}.png', **dict(zip(COLUMNS, row)))
            for index, row in enumerate(LABELS)]


def test_independent_two_and_three_pass_results(rows):
    two = expected(rows, COLUMNS[:2])
    assert [two[k] for k in ('n_rows','n_complete','n_partial','n_unlabelled','n_disagreements')] == [16,12,3,1,3]
    assert two['percent_agreement'] == .75
    assert two['overall_kappa'] == .5
    assert two['pairs'][0]['confusion'] == [[5,1],[2,4]]
    three = expected(rows, COLUMNS)
    assert three['overall_kappa'] == pytest.approx(.6625)
    assert three['n_complete'] == 12 and three['n_disagreements'] == 4
    assert verify(copy.deepcopy(two), two)
    assert verify(copy.deepcopy(three), three)


@pytest.mark.parametrize('key,value', [('n_complete',13),('n_partial',0),
    ('overall_kappa',.75),('percent_agreement',1.0),('disagreement_paths',[])])
def test_report_values_must_match(rows, key, value):
    reference = expected(rows, COLUMNS[:2])
    assert verify(copy.deepcopy(reference), reference)
    changed = copy.deepcopy(reference)
    changed[key] = value
    with pytest.raises(ValueError, match=key):
        verify(changed, reference)


def test_each_pair_and_confusion_matrix_is_checked(rows):
    reference = expected(rows, COLUMNS)
    assert verify(copy.deepcopy(reference), reference)
    changed = copy.deepcopy(reference)
    changed['pairs'][1]['confusion'][0][0] += 1
    with pytest.raises(ValueError, match='confusion'):
        verify(changed, reference)
    changed = copy.deepcopy(reference)
    changed['pairs'].pop()
    with pytest.raises(ValueError, match='pair count'):
        verify(changed, reference)
    changed = copy.deepcopy(reference)
    changed['pairs'].append(copy.deepcopy(changed['pairs'][0]))
    with pytest.raises(ValueError, match='pair count'):
        verify(changed, reference)

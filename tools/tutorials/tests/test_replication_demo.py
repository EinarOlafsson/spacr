"""Positive counterparts and corrupt-output checks for tutorial evidence only."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replication_demo import expected, summarize, verify_records


def test_distribution_keeps_off_ladder_and_overflow_counts_separate():
    result = summarize([1, 2, 3, 32])
    assert result['n_vacuoles'] == 4 and result['n_parasites'] == 38
    assert result['n_non_power_of_two'] == result['n_gt16'] == 1
    assert result['frac_non_power_of_two'] == result['frac_gt16'] == .25
    assert result['qc_flag_non_power_of_two'] is True
    assert result['n_power_of_two'] == 3
    assert result['median_doublings'] == 1
    assert result['mean_parasites_per_vacuole'] == pytest.approx(35 / 3)
    assert result['mean_fraction_of_vacuoles'] == .75


def test_two_vacuoles_in_the_same_host_are_not_merged():
    template = dict(plateID='plate1', rowID='r1', columnID='c1', fieldID='f1',
                    prcf='plate1_r1_c1_f1', cell_id=1, pathogen_area=300)
    rows = [dict(template, vacuole_id=key) for key in ['a', 'a', 'b', 'b', 'b', 'b']]
    result = expected(rows)
    assert len(result['vacuole_counts.csv'][1]) == 2
    well = result['well_distribution.csv'][1]['plate1_r1_c1']
    assert well['n_vacuoles'] == 2 and well['n_parasites'] == 6
    assert well['n_2'] == well['n_4'] == 1
    assert well['n_non_power_of_two'] == 0
    assert well['qc_flag_non_power_of_two'] is False


@pytest.fixture
def reference():
    return {'well_a': {'n_vacuoles': 40, 'frac_non_power_of_two': .05,
                       'qc_flag_non_power_of_two': False, 'condition': 'DEMO'}}


def as_records(reference):
    return [dict(values, prc=key) for key, values in reference.items()]


@pytest.mark.parametrize('name,value', [('n_vacuoles', 41),
    ('frac_non_power_of_two', .5), ('qc_flag_non_power_of_two', True),
    ('condition', 'wrong')])
def test_changed_values_are_rejected(reference, name, value):
    actual = as_records(reference)
    assert verify_records(actual, 'prc', reference) == 4
    changed = copy.deepcopy(actual)
    changed[0][name] = value
    with pytest.raises(ValueError, match=name):
        verify_records(changed, 'prc', reference)


@pytest.mark.parametrize('mode', ['missing', 'extra', 'duplicate'])
def test_missing_extra_and_duplicate_identities_are_rejected(reference, mode):
    actual = as_records(reference)
    assert verify_records(actual, 'prc', reference) == 4
    changed = copy.deepcopy(actual)
    if mode == 'missing':
        changed = []
    elif mode == 'extra':
        changed.append(dict(changed[0], prc='unexpected'))
    else:
        changed.append(copy.deepcopy(changed[0]))
    with pytest.raises(ValueError, match='identities'):
        verify_records(changed, 'prc', reference)

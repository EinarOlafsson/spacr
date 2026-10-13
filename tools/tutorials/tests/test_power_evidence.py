"""Positive and deliberately wrong captured curves for tutorial acceptance."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from power_evidence import check_curves, check_recommendation


def records():
    result = {}
    for prefix, column in [('cells', 'imaging_n_cells_per_well_mu'), ('wells', 'n_wells_per_screen')]:
        result[prefix + '_scan'] = [
            {column: 120, 'status': 'ok', 'model_auroc': 0.8},
            {column: 120, 'status': 'not_converged', 'model_auroc': None},
        ]
        result[prefix + '_curve'] = [{'value': 120, 'n_replicates': 2,
                                     'n_detected': 1, 'power': 0.5, 'n_ok': 1,
                                     'n_not_converged': 1, 'n_failed': 0}]
    return result


def test_nonconverged_high_score_is_not_a_detection_but_is_in_denominator():
    data = records()
    data['cells_scan'][1]['model_auroc'] = 0.99
    assert check_curves(data, 0.8, 2) == 4


def test_same_score_can_be_a_detection_once_the_fit_is_usable():
    data = records()
    data['cells_scan'][1].update(status='ok', model_auroc=0.99)
    data['cells_curve'][0].update(n_detected=2, power=1, n_ok=2, n_not_converged=0)
    assert check_curves(data, 0.8, 2) == 4


def test_dropping_a_nonconverged_fit_cannot_raise_reported_power():
    data = records()
    data['cells_scan'].pop()
    data['cells_curve'][0].update(n_replicates=1, power=1, n_not_converged=0)
    with pytest.raises(ValueError, match='denominator'):
        check_curves(data, 0.8, 2)


def test_high_score_from_an_unusable_fit_cannot_raise_reported_power():
    data = records()
    data['cells_scan'][1]['model_auroc'] = 0.99
    data['cells_curve'][0].update(n_detected=2, power=1)
    with pytest.raises(ValueError, match='only usable'):
        check_curves(data, 0.8, 2)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'invented'])
def test_curve_cannot_hide_duplicate_or_invent_simulated_points(mutation):
    data = records()
    curves = data['cells_curve']
    if mutation == 'missing':
        curves.clear()
    elif mutation == 'duplicate':
        curves.append(deepcopy(curves[0]))
    else:
        curves[0]['value'] = 999
    with pytest.raises(ValueError, match='axes'):
        check_curves(data, 0.8, 2)


def test_bad_status_counts_are_not_accepted_even_when_power_matches():
    data = records()
    data['cells_curve'][0].update(n_failed=1, n_not_converged=0)
    with pytest.raises(ValueError, match='status counts'):
        check_curves(data, 0.8, 2)


def test_a_different_well_count_is_not_the_unchanged_design_defect():
    assert check_recommendation('Going to 192 wells would take that to 100%.', 96) is None


def test_actual_unchanged_well_count_recommendation_is_rejected():
    with pytest.raises(ValueError, match='unchanged well count'):
        check_recommendation('Going to 96 wells would take that to 100%.', 96)

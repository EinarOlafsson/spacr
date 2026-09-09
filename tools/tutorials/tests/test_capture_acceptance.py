"""A finished worker alone is not proof of a successful tutorial example."""
import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / 'capture_acceptance.py'
spec = importlib.util.spec_from_file_location('capture_acceptance', MODULE)
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


def test_accepts_real_completion_with_figures_and_an_ordinary_qc_warning():
    result = acceptance.assess_pipeline(
        {'finished': True, 'ok': True, 'errors': []},
        ['QC: WARN - size_outliers; inspect this field.\nSuccessfully completed run\n✓ Finished'], 9)
    assert result == {'accepted': True, 'reasons': [], 'figure_count': 9,
                      'worker_finished': True, 'worker_ok': True}


@pytest.mark.parametrize('marker', ['RUN INCOMPLETE', 'ARTIFACTS FROM THIS RUN ARE INCOMPLETE'])
def test_rejects_partial_output_even_when_gui_and_worker_say_finished(marker):
    result = acceptance.assess_pipeline(
        {'finished': True, 'ok': True, 'errors': []},
        [f'{marker} — overlay plotting\nSuccessfully completed run\n✓ Finished'], 9)
    assert result['accepted'] is False
    assert result['worker_ok'] is True
    assert 'partial artifacts' in result['reasons'][0]


@pytest.mark.parametrize('outcome,count', [
    ({'finished': False, 'ok': True}, 9),
    ({'finished': True, 'ok': False}, 9),
    ({'finished': True, 'ok': True, 'errors': ['failed']}, 9),
    ({'finished': True, 'ok': True}, 0),
])
def test_refuses_incomplete_failed_or_figureless_runs(outcome, count):
    result = acceptance.assess_pipeline(outcome, ['✓ Finished'], count)
    assert result['accepted'] is False
    assert result['reasons']

"""Empty score selections fail before aggregation diagnostics or transformation."""
import numpy as np
import pandas as pd
import pytest

from spacr.ml import process_scores


def _scores(values, wells=None):
    """Build one plate of score rows with explicit well metadata."""
    return pd.DataFrame({'plateID': ['p1'] * len(values),
                         'rowID': ['r1'] * len(values),
                         'columnID': wells or ['c1'] * len(values),
                         'pred': values})


@pytest.mark.parametrize('transform', [None, 'log', 'sqrt'])
def test_minimum_cells_explains_the_stage_counts_and_action(transform):
    """Every well below the threshold reports source and well counts before sklearn."""
    frame = _scores([0.2] * 8, ['c1'] * 4 + ['c2'] * 4)
    with pytest.raises(ValueError) as caught:
        process_scores(frame, 'pred', None, min_cells_per_well=25, transform=transform)
    message = str(caught.value)
    for text in ('0 of 2 wells', '8 input score rows', 'min_cells_per_well=25',
                 'largest well has 4', 'Lower'):
        assert text in message
    assert 'FunctionTransformer' not in message


def test_empty_input_is_reported_before_metadata_or_normality(monkeypatch):
    """An upstream selection with no rows gets an actionable score-source message."""
    from spacr import ml

    monkeypatch.setattr(ml, 'check_normality', lambda *a: pytest.fail('normality reached'))
    with pytest.raises(ValueError, match='0 input score rows.*score source'):
        process_scores(_scores([]), 'pred', None, transform='log')


@pytest.mark.parametrize('values', [[np.nan, np.nan], [np.inf, -np.inf], ['bad', 'bad']])
def test_no_finite_response_is_reported_before_aggregation(values):
    """All invalid values cannot become an empty transform or a fabricated zero count."""
    with pytest.raises(ValueError, match='0 finite.*pred.*2 input score rows'):
        process_scores(_scores(values), 'pred', None, min_cells_per_well=1,
                       regression_type='poisson', transform='log')


def test_reciprocal_inversion_explains_why_all_rows_disappeared():
    """Zero-only responses fail at inversion and identify the controlling setting."""
    with pytest.raises(ValueError) as caught:
        process_scores(_scores([0.0, 0.0]), 'pred', None, min_cells_per_well=1,
                       invert_dependent_variable=-1, transform='log')
    message = str(caught.value)
    assert '0 of 2 input score rows' in message
    assert 'invert_dependent_variable=-1' in message
    assert 'zero' in message


def test_all_invalid_aggregates_stop_before_transformation():
    """Nonfinite well aggregates are reported even if some raw rows were finite."""
    with pytest.raises(ValueError, match='0 finite.*after aggregation'):
        process_scores(_scores([1.0, np.inf]), 'pred', None,
                       min_cells_per_well=1, transform='log')


def test_surviving_wells_keep_the_existing_aggregation_and_transform():
    """Valid scores retain their exact well mean and logarithm after a small well is dropped."""
    frame = _scores([1.0, 3.0, 10.0], ['c1', 'c1', 'c2'])
    result, response = process_scores(frame, 'pred', None,
                                     min_cells_per_well=2, transform='log')
    assert list(result['prc']) == ['p1_r1_c1']
    assert result['pred'].iloc[0] == 2.0
    assert result['cell_count'].iloc[0] == 2
    assert response == 'log_pred'
    assert np.isclose(result[response].iloc[0], np.log1p(2.0))


def test_real_zero_counts_remain_valid():
    """Finite zero labels still produce legitimate zero-count wells."""
    result, response = process_scores(_scores([0.0, 0.0]), 'pred', None,
                                     min_cells_per_well=1, regression_type='poisson')
    assert response == 'pred'
    assert result['pred'].iloc[0] == 0
    assert result['cell_count'].iloc[0] == 2

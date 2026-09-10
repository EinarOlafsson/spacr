"""A passing chart capture must check identities, numbers and refusal guards."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from control_chart_evidence import FIXTURE, expected_points, verify_points, verify_export


def test_positive_independent_calculation_matches_actual_engine():
    import pandas as pd
    from spacr.qt.widgets.control_chart import ControlChartSpec, control_chart
    for level, baseline in (('neg', 20), ('pos', 20), ('neg', 12)):
        result = control_chart(pd.read_csv(FIXTURE), ControlChartSpec(
            plate='plateID', order='run_date', value='signal',
            control_column='well_type', control_levels=(level,),
            baseline_n=baseline, rules=(1,)))
        checks = verify_points(result.points_frame().to_dict('records'),
                               expected_points(FIXTURE, level, baseline))
        assert checks['checked_fields'] == 390
        assert checks['rule_one_independently_checked']


@pytest.mark.parametrize('key,value', [('plate', 'P99'), ('order', 'different'),
    ('value', -1000), ('n', 9), ('subgroup_sd', 800), ('centre', 0),
    ('lower', 0), ('upper', 0), ('sigma', 0), ('z', 800),
    ('in_baseline', False), ('flagged', True), ('rules', '8')])
def test_each_wrong_identity_statistic_or_flag_is_rejected(key, value):
    expected = expected_points(FIXTURE)
    actual = deepcopy(expected)
    assert actual[0][key] != value
    actual[0][key] = value
    with pytest.raises(ValueError, match=key):
        verify_points(actual, expected)


def test_missing_points_and_columns_fail():
    expected = expected_points(FIXTURE)
    with pytest.raises(ValueError, match='point count'):
        verify_points(expected[:-1], expected)
    actual = deepcopy(expected)
    del actual[0]['sigma']
    with pytest.raises(ValueError, match='Missing chart column'):
        verify_points(actual, expected)


def test_positive_csv_roundtrip_and_reordered_identity_failure(tmp_path):
    import pandas as pd
    from control_chart_evidence import COLUMNS
    expected = expected_points(FIXTURE)
    file = tmp_path / 'points.csv'
    pd.DataFrame(expected)[list(COLUMNS)].to_csv(file, index=False)
    assert verify_export(file, expected)['checked_fields'] == 390
    pd.DataFrame(expected[::-1])[list(COLUMNS)].to_csv(file, index=False)
    with pytest.raises(ValueError, match='plate'):
        verify_export(file, expected)

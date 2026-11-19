"""Missing regression input tables are explained before a GUI worker starts."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from spacr import validate


@pytest.mark.parametrize('settings', [
    {}, {'paired_data': []}, {'paired_data': [{'score': 's.csv'}]},
    {'paired_data': [{'count': 'c.csv'}]},
    {'score_data': ' ', 'count_data': 'c.csv'},
])
def test_missing_inputs_explain_the_regression_controls(settings):
    """Blank or one-sided inputs name the UI controls and supported formats."""
    original = deepcopy(settings)
    errors = [p for p in validate.validate_settings(settings, 'regression')
              if p.setting == 'paired_data' and p.is_error]
    assert len(errors) == 1
    message = str(errors[0])
    for text in ('Regression', 'Paired data', 'Score', 'Count', 'Parquet', 'database'):
        assert text in message
    assert settings == original


@pytest.mark.parametrize('settings', [
    {'paired_data': [{'score': 's.parquet'}, {'count': 'c.db', 'count_table': 'reads'}]},
    {'score_data': ['s1.csv', 's2.csv'], 'count_data': 'c.csv'},
    {'score_data': 's.csv', 'count_data': ['c1.csv', 'c2.csv']},
    {'screen_type': 'drug_screen', 'drug_data': 'wells.csv'},
])
def test_existing_input_patterns_remain_accepted(settings):
    """Split rows, unequal legacy lists and drug screens retain their contract."""
    assert not [p for p in validate.validate_settings(settings, 'regression')
                if p.setting == 'paired_data']


def test_pipeline_uses_the_same_actionable_missing_input_message():
    """Direct Python users get instructions before any input read."""
    from spacr.ml import normalize_regression_input_pairs

    with pytest.raises(ValueError, match='Paired data.*Score.*Count'):
        normalize_regression_input_pairs({})


def test_gui_refuses_missing_inputs_before_claiming_or_starting_a_run(monkeypatch):
    """Run displays the preflight finding without creating a job or a project lock."""
    from spacr.qt.screens import app_screen

    shown = []
    monkeypatch.setattr(app_screen, 'resolve_pipeline_entry', lambda key: lambda settings: None)
    monkeypatch.setattr(app_screen.QMessageBox, 'warning', lambda *args: shown.append(args))
    screen = SimpleNamespace(
        app_key='regression', _wait_for_the_entry_import=lambda override: False,
        _settings_model=SimpleNamespace(collect=lambda: {}),
        _annotation_labels_named=lambda settings: pytest.fail('preflight was skipped'),
    )
    app_screen.AppScreen._on_run(screen)
    assert len(shown) == 1
    assert 'Paired data' in shown[0][-1]

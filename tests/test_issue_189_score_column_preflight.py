"""Regression rejects an absent response from input headers before starting work."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sqlite3

import pandas as pd
import pytest

from spacr import validate


def _settings(tmp_path, kind='csv', column='cell_area'):
    """Write one small score table and select it as the regression response source."""
    frame = pd.DataFrame({'prcfo': ['p1_r1_c1_f1_o1'], column: [0.5]})
    path = tmp_path / ('scores.' + ('db' if kind == 'sqlite' else kind))
    pair = {'score': str(path), 'count': str(tmp_path / 'counts.csv')}
    if kind == 'sqlite':
        with sqlite3.connect(path) as conn:
            frame.to_sql('scores', conn, index=False)
        pair['score_table'] = 'scores'
    elif kind == 'parquet':
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return {'paired_data': [pair], 'dependent_variable': 'pred'}


def _response_errors(settings):
    """Return only blocking response-column findings from the public preflight."""
    return [problem for problem in validate.validate_settings(settings, 'regression')
            if problem.setting == 'dependent_variable' and problem.is_error]


@pytest.mark.parametrize('kind', ['csv', 'parquet', 'sqlite'])
def test_measurements_without_pred_explain_the_two_explicit_choices(tmp_path, kind):
    """Header checks give score-output or explicit measured-response instructions."""
    settings = _settings(tmp_path, kind)
    original = deepcopy(settings)
    errors = _response_errors(settings)
    assert len(errors) == 1
    message = str(errors[0])
    for text in ('pred', 'scores.', 'cell_area', 'Run Classify', 'dependent_variable'):
        assert text in message
    assert 'choose' in message.lower()
    assert settings == original


def test_score_headers_are_read_without_loading_rows(tmp_path, monkeypatch):
    """A large CSV preflight requests the schema only, never a data sample."""
    from spacr import tabular

    settings = _settings(tmp_path)
    original = tabular.read_table
    reads = []

    def read_header(source, **kwargs):
        """Record and enforce the zero-row CSV read."""
        reads.append(kwargs)
        assert kwargs.get('nrows') == 0
        return original(source, **kwargs)

    monkeypatch.setattr(tabular, 'read_table', read_header)
    assert _response_errors(settings)
    assert reads


def test_each_selected_score_table_must_supply_each_response(tmp_path):
    """A score column on one plate cannot conceal its absence from another."""
    settings = _settings(tmp_path, column='pred')
    second = tmp_path / 'second.csv'
    pd.DataFrame({'prcfo': ['p2_r1_c1_f1_o1'], 'cell_area': [0.5]}).to_csv(second, index=False)
    settings['paired_data'].append({'score': str(second)})
    errors = _response_errors(settings)
    assert len(errors) == 1
    assert 'second.csv' in str(errors[0])


def test_existing_response_and_in_memory_handoff_are_accepted(tmp_path, monkeypatch):
    """An offered frame is authoritative even when its backing file is absent."""
    from spacr import frame_handoff, tabular

    settings = _settings(tmp_path)
    path = settings['paired_data'][0]['score']
    frame = pd.DataFrame({'pred': [0.5], 'cell_area': [3.0]})
    frame_handoff.hold(path, frame)
    Path(path).unlink()
    monkeypatch.setattr(tabular, 'table_columns', lambda *a, **k: pytest.fail('disk read'))
    settings['dependent_variable'] = ['pred', 'cell_area']
    try:
        assert _response_errors(settings) == []
    finally:
        frame_handoff.release(path)


def test_missing_source_gets_an_actionable_file_error_without_creating_it(tmp_path):
    """A missing SQLite source is not opened into an accidental empty database."""
    path = tmp_path / 'missing.db'
    settings = {'paired_data': [{'score': str(path), 'score_table': 'scores', 'count': 'c.csv'}]}
    errors = [p for p in validate.validate_settings(settings, 'regression')
              if p.setting == 'score_data' and p.is_error]
    assert len(errors) == 1
    assert 'Score source' in str(errors[0])
    assert not path.exists()


def test_gui_stops_wrong_score_table_before_worker_setup(tmp_path, monkeypatch):
    """Run surfaces the absent response while its settings remain editable."""
    from spacr.qt.screens import app_screen

    shown = []
    monkeypatch.setattr(app_screen, 'resolve_pipeline_entry', lambda key: lambda settings: None)
    monkeypatch.setattr(app_screen.QMessageBox, 'warning', lambda *args: shown.append(args))
    screen = SimpleNamespace(
        app_key='regression', _wait_for_the_entry_import=lambda override: False,
        _settings_model=SimpleNamespace(collect=lambda: _settings(tmp_path), with_explicit_record=lambda s: s),
        _annotation_labels_named=lambda settings: pytest.fail('worker setup reached'),
    )
    app_screen.AppScreen._on_run(screen)
    assert len(shown) == 1
    assert 'pred' in shown[0][-1]


@pytest.mark.parametrize('source, table', [
    ('postgresql://localhost/test', 'scores'),
    ('https://example.invalid/scores.csv', None),
    ('/data/scores.feather', None),
])
def test_gui_defers_unbounded_reads_to_worker(source, table, monkeypatch):
    """Remote and Feather sources never download or load a full table on Run."""
    from spacr import tabular

    monkeypatch.setattr(tabular, 'table_columns', lambda *a, **k: pytest.fail('GUI I/O'))
    settings = {'paired_data': [{'score': source, 'score_table': table}]}
    assert validate._check_regression_score_columns(settings, local_only=True) == []


def test_worker_checks_postgres_schema_without_a_local_file(monkeypatch):
    """PostgreSQL source names reach the shared table schema reader unchanged."""
    from spacr import tabular

    source = 'postgresql://localhost/test'
    seen = []

    def columns(path, *, table):
        """Record the named remote schema request."""
        seen.append((path, table))
        return ('prcfo', 'cell_area')

    monkeypatch.setattr(tabular, 'table_columns', columns)
    settings = {'paired_data': [{'score': source, 'score_table': 'scores'}]}
    errors = validate._check_regression_score_columns(settings)
    assert len(errors) == 1
    assert seen == [(source, 'scores')]
    assert "'pred' is not a column" in str(errors[0])


def test_direct_pipeline_checks_header_before_reading_inputs(tmp_path, monkeypatch):
    """Python entry points stop at the same schema check before loading tables."""
    from spacr import ml

    settings = _settings(tmp_path)
    monkeypatch.setattr(ml, 'load_regression_input_pairs',
                        lambda pairs: pytest.fail('input rows loaded'))
    with pytest.raises(ValueError, match="'pred' is not a column"):
        ml.perform_regression(settings)

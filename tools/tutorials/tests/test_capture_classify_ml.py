"""Positive and deliberately invalid saved outputs for the tutorial check."""
import csv
import importlib.util
from pathlib import Path
import sqlite3

import pytest

spec = importlib.util.spec_from_file_location(
    'capture_classify_ml', Path(__file__).resolve().parents[1] / 'capture_classify_ml.py')
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


def example(tmp_path, change=None):
    database, output = tmp_path / 'measurements.db', tmp_path / 'results.csv'
    records = [('one', 'p', 'r1', 'c1'), ('two', 'p', 'r1', 'c2'),
               ('three', 'p', 'r2', 'c1'), ('four', 'p', 'r2', 'c2')]
    rows = [dict(prcfo=record[0], data_usage='train' if i < 2 else 'test',
                 data_usage_group_by='well', split_requested_fraction='0.1', predictions='1',
                 prediction_probability_class_0='0.25',
                 prediction_probability_class_1='0.75') for i, record in enumerate(records)]
    if change:
        change(records, rows)
    with sqlite3.connect(database) as con:
        con.execute('CREATE TABLE png_list (prcfo,plateID,rowID,columnID)')
        con.executemany('INSERT INTO png_list VALUES (?,?,?,?)', records)
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return recorder.inspect_output(database, output)


def test_real_disjoint_wells_and_finite_predictions_pass(tmp_path):
    result = example(tmp_path)
    assert result['accepted']
    assert result['sample_counts'] == {'train': 2, 'test': 2, 'not_used': 0}


def test_actual_well_overlap_rejected_despite_unique_objects(tmp_path):
    def change(records, rows):
        records[2] = ('three', 'p', 'r1', 'c1')
    result = example(tmp_path, change)
    assert not result['accepted']
    assert result['overlapping_actual_wells'] == [('p', 'r1', 'c1')]


def test_missing_test_label_rejected(tmp_path):
    result = example(tmp_path, lambda records, rows: rows[3].update(data_usage='not_used'))
    assert not result['accepted']
    assert 'Both plate-column' in result['reasons'][0]


@pytest.mark.parametrize('field,value,message', [
    ('prcfo', 'absent', 'Duplicate or unknown'),
    ('prcfo', 'two', 'Duplicate or unknown'),
    ('data_usage_group_by', 'cell', 'well-level'),
    ('split_requested_fraction', '0.5', 'Saved split request'),
    ('prediction_probability_class_1', 'nan', 'Non-finite'),
    ('prediction_probability_class_1', '0.8', 'sum to one'),
    ('predictions', '0', 'maximum class probability'),
])
def test_invalid_output_refused(tmp_path, field, value, message):
    with pytest.raises(ValueError, match=message):
        example(tmp_path, lambda records, rows: rows[0].update({field: value}))


def test_bounded_demo_does_not_enable_supervised_pruning_or_cv():
    settings = recorder.bounded_settings()
    assert settings['prune_features'] is False
    assert settings['cross_validation'] is False
    assert settings['cv_group_by'] == 'well'
    assert settings['n_jobs'] == 2


def metric_example(tmp_path, *, bad_accuracy=False, bad_database=False):
    example(tmp_path)
    database = tmp_path / 'measurements.db'
    with sqlite3.connect(database) as con:
        con.execute('ALTER TABLE png_list ADD COLUMN predictions')
        con.execute('ALTER TABLE png_list ADD COLUMN ml_pred')
        con.execute('UPDATE png_list SET predictions=1,ml_pred=0.75')
        if bad_database:
            con.execute("UPDATE png_list SET ml_pred=0.9 WHERE prcfo='one'")
    metrics = tmp_path / 'metrics.csv'
    with metrics.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['','precision','recall','f1-score','support'])
        writer.writerows([['0',0,0,0,1], ['1',0.5,1,2/3,1],
                          ['accuracy',0.9 if bad_accuracy else 0.5,0.5,0.5,0.5]])
    return recorder.independent_metrics(database, tmp_path / 'results.csv', metrics)


def test_saved_metrics_and_database_have_positive_counterpart(tmp_path):
    result = metric_example(tmp_path)
    assert result['accuracy'] == 0.5
    assert result['metrics_compared'] == 9
    assert result['database_predictions_checked'] == 4


def test_wrong_saved_accuracy_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='Saved holdout metrics'):
        metric_example(tmp_path, bad_accuracy=True)


def test_database_score_not_just_csv_is_checked(tmp_path):
    with pytest.raises(ValueError, match='Database predictions'):
        metric_example(tmp_path, bad_database=True)

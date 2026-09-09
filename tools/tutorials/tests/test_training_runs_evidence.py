"""Tiny CSV/comparison guards; no spaCR, Qt, app or real-input imports."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_training_runs import _read_curve, _same_rows, _require_visible_identifier


def rows():
    return [{'epoch': 1.0, 'accuracy': 0.5, 'loss': 0.8},
            {'epoch': 2.0, 'accuracy': 0.75, 'loss': 0.4}]


def test_read_and_compare_matching_normalized_csv_without_mutation(tmp_path):
    path = tmp_path / 'train.csv'
    text = ', Epoch ,accuracy,Accuracy,loss\n0,1,0.5,0.5,0.8\n0,2,0.75,0.75,0.4\n'
    path.write_text(text)
    actual, expected = _read_curve(path), rows()
    before = deepcopy(actual)
    assert actual == expected
    _same_rows(actual, expected, 'positive')
    assert actual == before and expected == rows()
    assert path.read_text() == text


@pytest.mark.parametrize('actual', [rows()[:1], rows() + rows()[:1]])
def test_rejects_missing_or_extra_rows(actual):
    with pytest.raises(RuntimeError, match='row count'):
        _same_rows(actual, rows(), 'row-count corruption')


@pytest.mark.parametrize('change', ['missing', 'extra'])
def test_rejects_missing_or_extra_columns(change):
    actual = rows()
    if change == 'missing':
        del actual[0]['accuracy']
    else:
        actual[0]['unrecorded_metric'] = 0.9
    with pytest.raises(RuntimeError, match='columns'):
        _same_rows(actual, rows(), 'column corruption')


@pytest.mark.parametrize('key,value', [
    ('epoch', 3.0), ('accuracy', 0.9), ('loss', 0.81),
    ('accuracy', float('nan')), ('accuracy', float('inf')),
    ('accuracy', -float('inf')),
])
def test_rejects_wrong_or_nonfinite_returned_values(key, value):
    actual = rows()
    actual[0][key] = value
    with pytest.raises(RuntimeError, match='differs from the source CSV'):
        _same_rows(actual, rows(), 'value corruption')


@pytest.mark.parametrize('value', ['nan', 'inf', '-inf'])
def test_reader_rejects_nonfinite_source_values(tmp_path, value):
    path = tmp_path / 'train.csv'
    path.write_text(f'epoch,accuracy\n1,{value}\n')
    with pytest.raises(RuntimeError, match='Non-finite'):
        _read_curve(path)


@pytest.mark.parametrize('body', ['', '1,0.5\n3,0.75\n', '1,0.5\n1,0.75\n'])
def test_reader_rejects_empty_gapped_or_duplicate_epochs(tmp_path, body):
    path = tmp_path / 'train.csv'
    path.write_text('epoch,accuracy\n' + body)
    with pytest.raises(RuntimeError, match='consecutive epoch rows'):
        _read_curve(path)


def test_full_run_identifier_inside_wide_native_viewport():
    _require_visible_identifier('maxvit_t/0_1/epochs_20',
                                'maxvit_t/0_1/epochs_20 · 20 epochs · …',
                                (28, 20, 450, 32), (0, 0, 916, 900))


@pytest.mark.parametrize('text,rect', [
    ('maxvit_t/0_1/epochs_2…', (28, 20, 450, 32)),
    ('maxvit_t/0_1/epochs_25 · 25 epochs', (28, 20, 450, 32)),
    ('maxvit_t/0_1/epochs_20 · 20 epochs', (500, 20, 450, 32)),
    ('maxvit_t/0_1/epochs_20 · 20 epochs', (-1, 20, 450, 32)),
    ('maxvit_t/0_1/epochs_20 · 20 epochs', (28, 880, 450, 32)),
    ('maxvit_t/0_1/epochs_20 · 20 epochs', (28, 20, 0, 32)),
])
def test_rejects_elided_wrong_or_clipped_identifiers(text, rect):
    with pytest.raises(RuntimeError, match='identifier'):
        _require_visible_identifier('maxvit_t/0_1/epochs_20', text, rect, (0, 0, 916, 900))

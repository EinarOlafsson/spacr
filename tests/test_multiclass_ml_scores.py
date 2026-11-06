"""The feature classifier trains three or more classes, and two exactly as before.

A recruitment screen scores objects as recruitment, no recruitment or
artefact. The binary fit could only pool those into two arms, so the
classes are listed in ``ml_classes`` and each gets its own probability,
precision, recall and confusion-matrix row. The two-class output is compared
with a copy written by the binary code before multiclass support existed.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import spacr.ml as ML


FEATURES = [
    'cell_channel_3_mean_intensity',
    'cell_channel_3_percentile_75',
    'nucleus_channel_3_mean_intensity',
    'cytoplasm_channel_3_mean_intensity',
    'pathogen_channel_3_mean_intensity',
    'cell_channel_3_std_intensity',
]

REFERENCE = os.path.join(os.path.dirname(__file__), 'data',
                         'ml_two_class_reference')

COMMON = dict(channel_of_interest=3, location_column='columnID',
              n_repeats=1, n_jobs=1, remove_highly_correlated_features=False,
              test_size=0.25, cross_validation=False, verbose=False,
              split_by='cell')


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def feature_frame(locations, per_class=30, seed=8, spread=0.9):
    """Per-object features whose class centres are 0.3, 0.9 and 0.5."""
    rng = np.random.default_rng(seed)
    rows, index = [], []
    for location, centre in zip(locations, (0.3, 0.9, 0.5)):
        for _ in range(per_class):
            row = {'columnID': location, 'plateID': 'plate1'}
            for name in FEATURES:
                row[name] = float(rng.normal(centre, spread))
            rows.append(row)
            index.append(f'plate1_r1_{location}_f1_o{len(index)}')
    return pd.DataFrame(rows, index=index)


def three_class_frame():
    """Three well-separated classes, so every class is learnt."""
    rng = np.random.default_rng(3)
    rows, index = [], []
    for location, centre in (('c1', 0.0), ('c2', 3.0), ('c3', -3.0)):
        for _ in range(40):
            row = {'columnID': location, 'plateID': 'plate1'}
            for position, name in enumerate(FEATURES):
                shift = centre if position % 2 == 0 else -centre / 2
                row[name] = float(rng.normal(shift, 0.6))
            rows.append(row)
            index.append(f'plate1_r1_{location}_f1_o{len(index)}')
    return pd.DataFrame(rows, index=index)


@pytest.mark.parametrize('model_type', ['logistic_regression', 'shrinkage_lda'])
@pytest.mark.parametrize('classes', [None, ['c1', 'c2']])
def test_two_classes_give_the_saved_binary_output(model_type, classes):
    """The scores, probabilities, columns and metrics match the saved copy."""
    output, _ = ML.ml_analysis(
        feature_frame(('c1', 'c2')), positive_control='c2',
        negative_control='c1', model_type=model_type, classes=classes,
        **COMMON)
    scored, metrics = output[0], output[8]

    expected_columns = pd.read_csv(os.path.join(REFERENCE, 'columns.csv'))
    assert list(scored.columns) == expected_columns['column'].tolist()

    expected = pd.read_csv(os.path.join(REFERENCE, f'{model_type}_scores.csv'),
                           index_col=0)
    got = scored[expected.columns]
    assert list(got.index) == list(expected.index)
    assert got['predictions'].tolist() == expected['predictions'].tolist()
    assert got['data_usage'].tolist() == expected['data_usage'].tolist()
    for column in ('prediction_probability_class_0',
                   'prediction_probability_class_1'):
        np.testing.assert_allclose(got[column].to_numpy(),
                                   expected[column].to_numpy(),
                                   rtol=1e-9, atol=1e-12)

    expected_metrics = pd.read_csv(
        os.path.join(REFERENCE, f'{model_type}_metrics.csv'), index_col=0)
    assert list(metrics.columns) == list(expected_metrics.columns)
    assert [str(i) for i in metrics.index] == [str(i) for i in expected_metrics.index]
    numeric = expected_metrics.select_dtypes(include=[np.number]).columns
    np.testing.assert_allclose(metrics[numeric].to_numpy(dtype=float),
                               expected_metrics[numeric].to_numpy(dtype=float),
                               rtol=1e-9)
    assert not hasattr(output[3], 'spacr_confusion_matrix_')


@pytest.mark.parametrize('model_type', ['logistic_regression', 'shrinkage_lda'])
def test_three_classes_give_per_class_probabilities_and_metrics(model_type):
    """Each class has a probability column, a metrics row and a matrix row."""
    labels = ['no recruitment', 'recruitment', 'artefact']
    frame = three_class_frame()
    frame['columnID'] = frame['columnID'].map(
        {'c1': labels[0], 'c2': labels[1], 'c3': labels[2]})

    output, _ = ML.ml_analysis(frame, model_type=model_type, classes=labels,
                               **COMMON)
    scored, model, metrics = output[0], output[3], output[8]

    probabilities = scored[[f'prediction_probability_class_{i}'
                            for i in range(3)]]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
    assert set(scored['predicted_class']) <= set(labels)
    assert (scored['predicted_class']
            == [labels[i] for i in scored['predictions']]).all()

    for index, label in enumerate(labels):
        row = metrics.loc[str(index)]
        assert row['class_label'] == label
        assert row['precision'] > 0.9
        assert row['recall'] > 0.9

    confusion = model.spacr_confusion_matrix_
    assert list(confusion.index) == [f'true_{label}' for label in labels]
    assert list(confusion.columns) == [f'predicted_{label}' for label in labels]
    assert int(confusion.to_numpy().sum()) == len(output[7])
    assert np.trace(confusion.to_numpy()) >= 0.9 * len(output[7])
    assert model.spacr_class_labels_ == labels


def test_cross_validation_sums_the_confusion_matrix_over_folds():
    """Under cross-validation every labelled object is tested once."""
    output, _ = ML.ml_analysis(
        three_class_frame(), model_type='logistic_regression',
        classes=['c1', 'c2', 'c3'],
        **{**COMMON, 'cross_validation': True})
    confusion = output[3].spacr_confusion_matrix_
    assert int(confusion.to_numpy().sum()) == 120
    assert set(output[8].loc[['0', '1', '2'], 'class_label']) == {'c1', 'c2', 'c3'}


def test_pooled_values_form_one_class():
    """A list entry pools several values into one class, labelled with +."""
    frame = three_class_frame()
    extra = frame[frame['columnID'] == 'c3'].copy()
    extra['columnID'] = 'c4'
    extra.index = [f'plate1_r1_c4_f1_o{1000 + i}' for i in range(len(extra))]
    frame = pd.concat([frame, extra])
    output, _ = ML.ml_analysis(
        frame, model_type='logistic_regression',
        classes=['c1', 'c2', ['c3', 'c4']], **COMMON)
    assert output[3].spacr_class_labels_ == ['c1', 'c2', 'c3+c4']
    assert (output[0].loc[output[0]['columnID'].isin(['c3', 'c4']),
                          'data_usage'] != 'not_used').all()


@pytest.mark.parametrize('text, expected', [
    (None, None), ('', None), ('None', None), ([], None),
    ("['a', 'b', 'c']", ['a', 'b', 'c']),
    ('a, b, c', ['a', 'b', 'c']),
    (('a', ('b', 'c'), 'd'), ['a', ['b', 'c'], 'd']),
])
def test_class_lists_are_read_from_settings_text(text, expected):
    """A list typed into the GUI or stored in a CSV reads as the same list."""
    assert ML._class_arms(text) == expected


def test_a_class_matching_nothing_is_named():
    """A missing class is refused with its name and the values present."""
    with pytest.raises(ValueError, match="'c9'"):
        ML.ml_analysis(three_class_frame(), model_type='logistic_regression',
                       classes=['c1', 'c2', 'c9'], **COMMON)


@pytest.mark.parametrize('bad', [['c1'], ['c1', 'c1', 'c2']])
def test_too_few_or_repeated_classes_are_refused(bad):
    with pytest.raises(ValueError):
        ML._class_arms(bad)

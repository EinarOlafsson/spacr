"""Refuse invalid bag features before training without changing valid rows."""
import warnings

import numpy as np
import pandas as pd
import pytest

from spacr.embeddings import _mil_bags


@pytest.fixture(autouse=True)
def english_errors(monkeypatch):
    monkeypatch.setenv("SPACR_LANGUAGE", "en")


def frame():
    return pd.DataFrame({
        'well': np.repeat(['p1_A01', 'p1_A02', 'p1_A03', 'p1_A04'], 2),
        'label': np.repeat([0, 0, 1, 1], 2),
        'embedding_0': np.arange(8, dtype=np.float64),
        'embedding_1': np.linspace(-1, 1, 8),
    })


@pytest.mark.parametrize('invalid', [np.inf, -np.inf, np.finfo(np.float64).max, -np.finfo(np.float64).max])
def test_nonfinite_or_float32_overflow_refuses_before_bags(invalid):
    source = frame()
    source.loc[3, 'embedding_0'] = invalid
    before = source.copy(deep=True)
    with warnings.catch_warnings(record=True) as emitted:
        with pytest.raises(ValueError, match='finite'):
            _mil_bags(source, label_column='label', well_column='well', feature_columns=['embedding_0', 'embedding_1'])
    assert not [warning for warning in emitted if issubclass(warning.category, RuntimeWarning)]
    pd.testing.assert_frame_equal(source, before)


def test_valid_float32_extremes_and_existing_nan_drop_are_preserved():
    source = frame()
    source.loc[0, 'embedding_0'] = np.finfo(np.float32).max
    source.loc[1, 'embedding_0'] = -np.finfo(np.float32).max
    source.loc[3, 'embedding_1'] = np.nan
    before = source.copy(deep=True)
    bags, labels, wells, rows, kept = _mil_bags(
        source, label_column='label', well_column='well', feature_columns=['embedding_0', 'embedding_1'])
    assert wells == ['p1_A01', 'p1_A02', 'p1_A03', 'p1_A04']
    np.testing.assert_array_equal(labels, [0, 0, 1, 1])
    assert [len(bag) for bag in bags] == [2, 1, 2, 2]
    assert len(kept) == 7 and all(bag.dtype == np.float32 and np.isfinite(bag).all() for bag in bags)
    assert bags[0][0, 0] == np.finfo(np.float32).max
    assert bags[0][1, 0] == -np.finfo(np.float32).max
    assert np.concatenate(rows).tolist() == list(range(7))
    pd.testing.assert_frame_equal(source, before)


def test_nan_drop_does_not_hide_an_infinite_retained_feature():
    source = frame()
    source.loc[0, 'embedding_0'] = np.nan
    source.loc[1, 'embedding_1'] = np.inf
    before = source.copy(deep=True)
    with pytest.raises(ValueError, match='finite'):
        _mil_bags(source, label_column='label', well_column='well', feature_columns=['embedding_0', 'embedding_1'])
    pd.testing.assert_frame_equal(source, before)


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is'])
def test_actual_finite_refusal_uses_installed_reviewed_language(language, monkeypatch):
    import json
    from pathlib import Path

    review = Path(__file__).parents[1] / 'docs/i18n/reviewed/runtime' / language / '2026-10-15-mil-finite-features.json'
    expected = json.loads(review.read_text())['records'][0]['translation']
    source = frame()
    source.loc[0, 'embedding_0'] = np.inf
    before = source.copy(deep=True)
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    with pytest.raises(ValueError) as error:
        _mil_bags(source, label_column='label', well_column='well', feature_columns=['embedding_0', 'embedding_1'])
    assert str(error.value) == expected
    pd.testing.assert_frame_equal(source, before)

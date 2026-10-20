"""MIL refuses known incompatible embedding coordinates before training."""
import numpy as np
import pandas as pd
import pytest

from spacr.embeddings import EmbeddingSpec, _mil_bags


@pytest.fixture(autouse=True)
def english_errors(monkeypatch):
    monkeypatch.setenv('SPACR_LANGUAGE', 'en')


def frame():
    return pd.DataFrame({'well': np.repeat(['p_A1', 'p_A2', 'p_A3', 'p_A4'], 2),
                         'well_label': np.repeat([0, 0, 1, 1], 2),
                         'emb_0': np.arange(8, dtype=float),
                         'emb_1': np.linspace(-1, 1, 8),
                         'measurement': np.arange(8, dtype=float)})


def bags_preserving(source, **kwargs):
    before = source.copy(deep=True)
    try:
        return _mil_bags(source, **kwargs)
    finally:
        pd.testing.assert_frame_equal(source, before)


@pytest.mark.parametrize('column,first,second', [
    ('_embedding_fingerprint', EmbeddingSpec(normalize=True).fingerprint(), EmbeddingSpec(normalize=False).fingerprint()),
    ('_embedding_fingerprint', 'opaque-model-one', 'opaque-model-two'),
    ('_embedding_fingerprint', 'CustomEncoder', 'customencoder'),
    ('_embedding_weights_sha256', 'a' * 64, 'b' * 64),
])
def test_different_known_embedding_coordinates_refuse(column, first, second):
    source = frame()
    source[column] = [first] * 4 + [second] * 4
    with pytest.raises(ValueError, match='conflicting known encoder'):
        bags_preserving(source)


@pytest.mark.parametrize('column,value', [
    ('_embedding_fingerprint', '0123456789abcdef'),
    ('_embedding_weights_sha256', 'a' * 64),
])
def test_case_and_surrounding_space_do_not_invent_digest_conflicts(column, value):
    source = frame()
    source[column] = [value, ' ' + value.upper() + ' '] * 4
    bags, labels, wells, rows, kept = bags_preserving(source)
    assert [len(bag) for bag in bags] == [2] * 4
    np.testing.assert_array_equal(labels, [0, 0, 1, 1])
    assert kept[column].tolist() == source[column].tolist()


@pytest.mark.parametrize('column', ['_embedding_fingerprint', '_embedding_weights_sha256'])
@pytest.mark.parametrize('unknown', [None, np.nan, '', '  ', 'unknown', ' UnKnOwN '])
def test_legacy_missing_or_unknown_identity_remains_unverified_and_unchanged(column, unknown):
    source = frame()
    source[column] = ['a' * (16 if column.endswith('fingerprint') else 64)] * 4 + [unknown] * 4
    bags, labels, wells, rows, kept = bags_preserving(source)
    assert len(bags) == 4 and len(kept) == 8
    pd.testing.assert_series_equal(kept[column], source[column])


def test_unverified_weight_markers_are_not_treated_as_actual_digests():
    source = frame()
    source['_embedding_weights_sha256'] = ['loader_metadata', 'not-a-checksum'] * 4
    bags, _, _, _, kept = bags_preserving(source)
    assert len(bags) == 4 and len(kept) == 8


def test_only_retained_rows_participate_in_coordinate_consistency():
    source = frame()
    source['_embedding_fingerprint'] = ['model-one'] * 7 + ['model-two']
    source.loc[7, 'emb_1'] = np.nan
    bags, _, _, _, kept = bags_preserving(source)
    assert [len(bag) for bag in bags] == [2, 2, 2, 1]
    assert kept['_embedding_fingerprint'].eq('model-one').all()


def test_measurement_only_selection_does_not_compare_unused_embedding_metadata():
    source = frame()
    source['_embedding_fingerprint'] = ['model-one'] * 4 + ['model-two'] * 4
    source['_embedding_weights_sha256'] = ['a' * 64] * 4 + ['b' * 64] * 4
    bags, _, _, _, kept = bags_preserving(source, feature_columns=['measurement'])
    assert [bag.shape for bag in bags] == [(2, 1)] * 4 and len(kept) == 8


def test_explicit_mixed_embedding_measurement_selection_still_compares_embedding_identity():
    source = frame()
    source['_embedding_fingerprint'] = ['model-one'] * 4 + ['model-two'] * 4
    with pytest.raises(ValueError, match='conflicting known encoder'):
        bags_preserving(source, feature_columns=['measurement', 'emb_0'])


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is'])
def test_coordinate_conflict_uses_installed_reviewed_language(language, monkeypatch):
    import json
    from pathlib import Path

    review = Path(__file__).parents[1] / 'docs/i18n/reviewed/runtime' / language / '2026-10-15-mil-coordinate-identity.json'
    record = json.loads(review.read_text(encoding="utf-8"))['records'][0]
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    source = frame()
    source['_embedding_fingerprint'] = ['model-one'] * 4 + ['model-two'] * 4
    with pytest.raises(ValueError) as error:
        bags_preserving(source)
    assert str(error.value) == record['translation']
    assert str(error.value) != record['source']

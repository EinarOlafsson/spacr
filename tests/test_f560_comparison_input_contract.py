"""Comparison helpers refuse misaligned crops before any model is invoked."""
import types

import numpy as np
import pytest

from spacr import embeddings as emb

SOURCE = 'Provide a four-dimensional crop stack and one label per crop.'


@pytest.mark.parametrize('shape,labels', [
    ((4, 2, 2, 1), ['a', 'b', 'a']),
    ((4, 2, 2, 1), ['a', 'b', 'a', 'b', 'extra']),
    ((4, 2), ['a', 'b', 'a', 'b']),
    ((4, 2, 2), ['a', 'b', 'a', 'b']),
    ((4, 2, 2, 1, 1), ['a', 'b', 'a', 'b']),
])
def test_invalid_comparison_refuses_before_encoder_or_scorer(monkeypatch, shape, labels):
    crops = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    before = crops.copy()
    calls = []

    def encode(array, spec):
        calls.append('encode')
        return types.SimpleNamespace(values=np.zeros((len(array), 2)), columns=['x', 'y'])

    def score(*args, **kwargs):
        calls.append('score')
        return {'n': 4.}

    monkeypatch.setattr(emb, 'embed_array', encode)
    monkeypatch.setattr(emb, '_retrieval_scorecard', score)
    monkeypatch.setenv('SPACR_LANGUAGE', 'en')
    with pytest.raises(emb.EmbeddingError, match=SOURCE):
        emb._backbone_scorecards(crops, labels, {'first': emb.EmbeddingSpec(normalize=False)})
    assert calls == []
    np.testing.assert_array_equal(crops, before)


def test_valid_comparison_retains_blank_filtering_and_real_metrics(monkeypatch):
    import pandas as pd

    crops = np.arange(6*2*2, dtype=np.uint16).reshape(6, 2, 2, 1)
    before = crops.copy()
    labels = ['a', 'b', 'a', 'b', '', None]
    features = np.array([[1, 0], [0, 1], [1.1, 0], [0, 1.1], [.5, .5], [.6, .6]])
    calls = []

    def encode(array, spec):
        calls.append(array.copy())
        return types.SimpleNamespace(values=features.copy(), columns=['x', 'y'])

    monkeypatch.setattr(emb, 'embed_array', encode)
    card = emb._backbone_scorecards(crops, labels, {
        name: emb.EmbeddingSpec(normalize=False) for name in ('one', 'two')}, k=1)
    expected = emb._retrieval_scorecard(
        pd.DataFrame(features, index=[f'crop{i}' for i in range(6)], columns=['x', 'y']),
        {f'crop{i}': value for i, value in enumerate(labels)}, k=1)
    for name in ('one', 'two'):
        assert card.loc[name, 'dims'] == 2
        for metric, value in expected.items():
            assert card.loc[name, metric] == value
    assert len(calls) == 2
    assert all(np.array_equal(array, crops) for array in calls)
    np.testing.assert_array_equal(crops, before)

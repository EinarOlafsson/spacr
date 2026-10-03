"""The embedding UMAP and the linear-classifier scorecard."""

import numpy as np
import pandas as pd
import pytest

from spacr.embeddings import (_embedding_classifier_scorecard,
                              _embedding_umap)


def _two_groups(n=20, dims=6, seed=0):
    rng = np.random.default_rng(seed)
    values = np.vstack([rng.normal(0, 1, (n, dims)),
                        rng.normal(6, 1, (n, dims))])
    labels = {str(i): ("a" if i < n else "b") for i in range(2 * n)}
    return pd.DataFrame(values), labels


def test_a_separable_set_classifies_well_above_chance():
    frame, labels = _two_groups()
    card = _embedding_classifier_scorecard(frame, labels, folds=4)
    assert card["accuracy"] > 0.9
    assert card["chance"] == 0.5
    assert card["n"] == 40 and card["classes"] == 2 and card["folds"] == 4


def test_the_classifier_refuses_a_single_class():
    frame, _ = _two_groups()
    with pytest.raises(ValueError, match="two classes"):
        _embedding_classifier_scorecard(frame, {"0": "a", "1": "a"})


def test_the_umap_keeps_the_rows_and_is_repeatable():
    pytest.importorskip("umap")
    frame, _ = _two_groups(n=10)
    first = _embedding_umap(frame, n_neighbors=5)
    again = _embedding_umap(frame, n_neighbors=5)
    assert list(first.columns) == ["umap_1", "umap_2"]
    assert list(first.index) == list(frame.index)
    np.testing.assert_allclose(first.to_numpy(), again.to_numpy())
    with pytest.raises(ValueError, match="three"):
        _embedding_umap(frame.iloc[:2])

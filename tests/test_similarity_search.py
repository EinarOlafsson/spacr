"""Find cells like this: the similarity index behind Annotate's Like this.

The index is exact cosine search over standardised features, so these
tests pin it against a brute-force answer, check that a labelled set's
neighbours share its label far above chance (the rare class most of all),
and time one query over a million vectors on the CPU.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from spacr import active_learning as al


def _clusters(n_per=(400, 400, 20), dims=24, seed=0):
    """Labelled crops in three well-separated clusters, one of them rare."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 4, size=(len(n_per), dims))
    rows, labels = [], []
    for label, (count, centre) in enumerate(zip(n_per, centres)):
        rows.append(centre + rng.normal(0, 1, size=(count, dims)))
        labels += [label] * count
    matrix = np.vstack(rows)
    keys = [f"/data/crop_{i}.png" for i in range(len(matrix))]
    frame = pd.DataFrame(matrix, index=keys,
                         columns=[f"cell_f{j}" for j in range(dims)])
    return frame, dict(zip(keys, labels))


def _brute(frame, key, k):
    """The expected neighbours, computed the long way."""
    x = frame.to_numpy(dtype=np.float64)
    x = (x - np.median(x, axis=0)) / x.std(axis=0)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    at = list(frame.index).index(key)
    scores = x @ x[at]
    scores[at] = -np.inf
    order = np.argsort(-scores)[:k]
    return [frame.index[i] for i in order], scores[order]


def test_the_numpy_search_is_exact_and_leaves_out_the_query():
    frame, _ = _clusters()
    index = al._SimilarityIndex(frame, backend="numpy", block=97)
    key = frame.index[5]
    hits = index.like(key, 15)
    expected, scores = _brute(frame, key, 15)
    assert list(hits["key"]) == expected
    assert np.allclose(hits["similarity"], scores, atol=1e-5)
    assert list(hits["rank"]) == list(range(1, 16))
    assert key not in set(hits["key"])


def test_exclude_drops_crops_and_still_returns_k():
    frame, _ = _clusters()
    index = al._SimilarityIndex(frame, backend="numpy")
    key = frame.index[0]
    first = index.like(key, 10)
    later = index.like(key, 10, exclude=first["key"][:4])
    assert len(later) == 10
    assert not set(first["key"][:4]) & set(later["key"])
    assert list(later["key"][:6]) == list(first["key"][4:])


def test_missing_values_and_constant_columns_are_handled():
    frame, _ = _clusters(n_per=(50, 50, 5), dims=6)
    frame["flat"] = 3.0
    frame["empty"] = np.nan
    frame.iloc[3, 0] = np.nan
    frame.iloc[4, 1] = np.inf
    index = al._SimilarityIndex(frame, backend="numpy")
    assert "flat" not in index.columns and "empty" not in index.columns
    assert np.isfinite(index.vector(frame.index[3])).all()
    assert len(index.like(frame.index[4], 5)) == 5
    with pytest.raises(KeyError):
        index.vector("/not/indexed.png")
    with pytest.raises(ValueError):
        al._SimilarityIndex(pd.DataFrame({"a": [1.0, 1.0]}, index=["x", "y"]))


def test_position_and_bookkeeping_columns_are_not_compared():
    kept = al._similarity_columns([
        "cell_area", "rowID", "fieldID", "object_label", "cell_centroid-0",
        "cell_bbox-2", "cell_id", "time", "pathogen_mean_intensity"])
    assert kept == ["cell_area", "pathogen_mean_intensity"]


def test_neighbours_share_the_label_far_above_chance():
    frame, labels = _clusters()
    index = al._SimilarityIndex(frame, backend="numpy")
    table = al._similarity_agreement(index, labels, k=10).set_index("class")
    assert table.loc["all", "precision_at_k"] > 0.95
    assert table.loc["2", "precision_at_k"] > 0.9
    assert table.loc["2", "lift"] > 20
    with pytest.raises(ValueError):
        al._similarity_agreement(index, {frame.index[0]: 1})


def test_the_index_filters_by_image_type():
    frame, _ = _clusters(n_per=(30, 30, 4), dims=5)
    frame.index = [p.replace("crop_1", "cell_1") for p in frame.index]
    index = al._similarity_index("unused.db", features=frame,
                                 image_type="cell_")
    assert len(index) == sum("cell_" in k for k in frame.index)


def test_an_unknown_backend_or_a_missing_faiss_says_so():
    frame, _ = _clusters(n_per=(10, 10, 2), dims=4)
    with pytest.raises(ValueError):
        al._SimilarityIndex(frame, backend="annoy")
    try:
        import faiss  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="faiss-cpu"):
            al._SimilarityIndex(frame, backend="faiss")
        assert al._SimilarityIndex(frame).backend == "numpy"
    else:
        index = al._SimilarityIndex(frame, backend="faiss")
        ref = al._SimilarityIndex(frame, backend="numpy")
        key = frame.index[0]
        assert list(index.like(key, 5)["key"]) == list(ref.like(key, 5)["key"])


def test_one_query_over_a_million_crops_returns_in_under_a_second():
    rng = np.random.default_rng(3)
    n = 1_000_000
    frame = pd.DataFrame(rng.standard_normal((n, 32), dtype=np.float32),
                         index=pd.RangeIndex(n).astype(str))
    index = al._SimilarityIndex(frame, backend="numpy")
    del frame
    index.like("7", 100)
    started = time.perf_counter()
    hits = index.like("123456", 100)
    elapsed = time.perf_counter() - started
    assert len(hits) == 100 and "123456" not in set(hits["key"])
    assert hits["similarity"].is_monotonic_decreasing
    assert elapsed < 1.0, elapsed


def test_the_index_reads_a_measured_database(tmp_path):
    from tests.test_cov_active_learning_rounds import _make_project

    project = _make_project(tmp_path, per_well=10)
    index = al._similarity_index(project["db"])
    assert len(index) == len(project["crops"])
    assert "cell_area" in index.columns
    assert not {"rowID", "object_label"} & set(index.columns)
    labels = {c["png_path"]: c["_class"] for c in project["crops"]}
    table = al._similarity_agreement(index, labels, k=5).set_index("class")
    assert table.loc["all", "lift"] > 1.5

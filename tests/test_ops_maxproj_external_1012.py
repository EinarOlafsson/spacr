"""Opt-in external feature views preserve the signed OPS path and calibrate gene nulls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from spacr import ops_maxproj as om
from spacr import ops_reference as reference

FIXTURES = Path(__file__).parent / "data" / "release_contracts"
LEGACY_MAXPROJ_SHA256 = "ab334a89c63b499b4bb1ae1cc255f8b05b69584743907692efa4912a2fcf88dd"
LEGACY_EMBEDDING_SHA256 = "cd8b47fe3b651b75cf499922783c4f3e0fd6e603d5a939f8a3c77d36aaf18520"


@pytest.fixture(autouse=True)
def _one_blas_thread():
    """Keep nested numerical libraries within the scheduled CPU allocation."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        yield


@pytest.fixture
def legacy():
    """Load hash-bound original scoring and embedding code in this numerical environment."""
    sources = []
    for filename, digest in (("1012_maxproj_legacy_817e64b2ce.py.txt", LEGACY_MAXPROJ_SHA256),
                             ("1012_embedding_legacy_817e64b2ce.py.txt", LEGACY_EMBEDDING_SHA256)):
        raw = (FIXTURES / filename).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        sources.append(raw.decode())
    contract = json.loads((FIXTURES / "1012_maxproj_legacy_817e64b2ce.json").read_text())
    assert contract["base"] == "817e64b2ce2b770dbc0f8453053bbcaeef3c9556"
    assert contract["maxproj_source_sha256"] == LEGACY_MAXPROJ_SHA256
    assert contract["embedding_helpers_sha256"] == LEGACY_EMBEDDING_SHA256
    original = ModuleType("spacr._original_maxproj_1012")
    original.__package__ = "spacr"
    original.__file__ = str(FIXTURES / "1012_maxproj_legacy_817e64b2ce.py.txt")
    exec(compile(sources[0], original.__file__, "exec"), original.__dict__)
    embedding = {"__name__": "spacr._original_embedding_1012", "np": np}
    exec(compile(sources[1], str(FIXTURES / "1012_embedding_legacy_817e64b2ce.py.txt"), "exec"), embedding)
    return original, embedding["embed_gene_profiles"]


def _ops_inputs():
    """Build an OPS profile exercising guide aggregation and control TVN in both views."""
    rng = np.random.default_rng(921)
    names = np.array([f"gene_{i}" for i in range(25)] + [f"nontargeting_{i}" for i in range(10)])
    guide_genes = np.repeat(names, [3] * 25 + [1] * 10)
    guide_rows = rng.normal(size=(len(guide_genes), 8))
    guide_cells = rng.integers(20, 100, len(guide_genes)).astype(float)
    rows = np.array([guide_cells[guide_genes == name] @ guide_rows[guide_genes == name]
                     / guide_cells[guide_genes == name].sum() for name in names])
    counts = np.array([guide_cells[guide_genes == name].sum() for name in names])
    return rows, names, counts, guide_rows, guide_genes, guide_cells, [np.arange(4), np.arange(4, 8)]


def _assert_result_bytes(before, after):
    """Require identical result keys, array shapes, dtypes and bytes for every output field."""
    assert before.keys() == after.keys()
    for key in before:
        left = before[key] if isinstance(before[key], list) else [before[key]]
        right = after[key] if isinstance(after[key], list) else [after[key]]
        assert len(left) == len(right)
        for original, current in zip(left, right):
            assert original.shape == current.shape, key
            assert original.dtype == current.dtype, key
            assert original.tobytes() == current.tobytes(), key


@pytest.mark.parametrize("similarity", ["cosine", "shrunk"])
@pytest.mark.parametrize("seed", [0, 17])
def test_default_ops_outputs_are_byte_identical_to_frozen_original_code(legacy, monkeypatch, similarity, seed):
    """Preserve every original OPS result byte with TVN, both similarities and distinct seeds."""
    original, embedding = legacy
    inputs = _ops_inputs()
    with monkeypatch.context() as patch:
        patch.setattr(reference, "embed_gene_profiles", embedding)
        expected = original.maxproj_scores(*inputs, similarity=similarity, calibration=3, null=5, seed=seed)
    observed = om.maxproj_scores(*inputs, similarity=similarity, calibration=3, null=5, seed=seed)
    _assert_result_bytes(expected, observed)


def test_column_null_shuffles_columns_independently_within_covered_genes():
    """Break cross-column structure without relabelling rows or changing uncovered values."""
    rows = np.arange(40.0).reshape(20, 2)
    covered = np.arange(20) < 15
    first = om._column_rows(rows, covered, 732, 0)
    second = om._column_rows(rows, covered, 732, 1)
    assert np.array_equal(first[~covered], rows[~covered])
    for column in range(2):
        assert np.array_equal(np.sort(first[covered, column]), np.sort(rows[covered, column]))
    assert not np.array_equal(first[covered, 0] // 2, (first[covered, 1] - 1) // 2)
    assert not np.array_equal(first, second)
    assert np.array_equal(first, om._column_rows(rows, covered, 732, 0))
    assert not np.array_equal(first, om._column_rows(rows, covered, 733, 0))


def _external(seed=3, n=80, columns=12):
    """Return unstructured gene measurements with no guides or non-targeting rows."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, columns)), [f"G{i}" for i in range(n)]


@pytest.mark.parametrize("packed", [False, True])
def test_coverage_excludes_unmeasured_pairs_from_each_view_and_every_null(packed):
    """No-view pairs have score -inf, view -1 and p/q one; null denominators use measured pairs."""
    rows, names = _external(n=30, columns=8)
    coverage = np.zeros((30, 2), bool)
    coverage[:15, 0] = True
    coverage[10:29, 1] = True
    result = om.maxproj_scores(rows, names, views_ix=[np.arange(4), np.arange(4, 8)],
                               null_mode="columns", control=None, coverage=coverage,
                               packed=packed, calibration=3, null=5, block_size=7)
    iu = np.triu_indices(30, 1)
    for index, score in enumerate(result["scores"]):
        valid = coverage[:, index, None] & coverage[None, :, index]
        observed = score if packed else score[iu]
        assert np.isneginf(observed[~valid[iu]]).all()
        assert result["calibration_counts"][index].sum() == 3 * int(valid[iu].sum())
    valid = ((coverage @ coverage.T) > 0)[iu]
    for key in ("projected", "view", "p", "q"):
        observed = result[key] if packed else result[key][iu]
        if key == "projected":
            assert np.isneginf(observed[~valid]).all()
        else:
            assert (observed[~valid] == (-1 if key == "view" else 1)).all()
    assert result["null_counts"].sum() == 5 * int(valid.sum())
    expected_bins = om.pair_bins(om.cell_bins(coverage.sum(axis=1), 3), 3, packed=packed)
    assert np.array_equal(result["pair_bins"], expected_bins)


@pytest.mark.parametrize("similarity", ["cosine", "shrunk"])
def test_float32_packed_scores_agree_with_float64_square_results(similarity):
    """Precision and storage options preserve pair ordering and calibrated probabilities."""
    rows, names = _external()
    keywords = dict(views_ix=[np.arange(6), np.arange(6, 12)], null_mode="columns", control=None,
                    calibration=5, null=8, similarity=similarity, seed=19)
    square = om.maxproj_scores(rows, names, **keywords)
    packed = om.maxproj_scores(rows, names, **keywords, packed=True, dtype="float32", block_size=13)
    iu = np.triu_indices(len(rows), 1)
    assert packed["n_genes"] == len(rows) and packed["packed"] is True
    for key in ("projected", "p", "q"):
        assert packed[key].dtype == np.float32
        np.testing.assert_allclose(packed[key], square[key][iu], atol=2e-3, rtol=2e-3)
    for a, b in zip(packed["scores"], square["scores"]):
        assert a.dtype == np.float32
        np.testing.assert_allclose(a, b[iu], atol=2e-3, rtol=2e-3)


def test_packed_pipeline_does_not_allocate_global_upper_triangle_indices(monkeypatch):
    """Keep the external path free of the two quadratic integer-index arrays."""
    rows, names = _external(n=35, columns=8)

    def refused(*args, **kwargs):
        """Reject a global triangular-index allocation during packed scoring."""
        raise AssertionError("packed scoring requested global upper-triangle indices")

    monkeypatch.setattr(om.np, "triu_indices", refused)
    out = om.maxproj_scores(rows, names, views_ix=[np.arange(4), np.arange(4, 8)],
                            null_mode="columns", packed=True, dtype="float32",
                            calibration=3, null=5, block_size=9)
    assert out["p"].shape == (35 * 34 // 2,)


def test_packed_projection_streams_views_and_selects_the_first_eligible_tie(monkeypatch):
    """Consume a generator without a view stack and preserve coverage-aware tie handling."""
    scores = [np.array([2., -np.inf, -np.inf, 1.]), np.array([2., -np.inf, 3., 0.])]
    valid = [np.array([True, False, False, False]), np.array([True, True, True, False])]

    def refused(*args, **kwargs):
        """Reject stacking full view arrays in the packed projection path."""
        raise AssertionError("packed projection stacked views")

    monkeypatch.setattr(om.np, "stack", refused)
    projected, view = om.max_projection(iter(scores), valid=iter(valid))
    assert np.array_equal(projected, [2., -np.inf, 3., -np.inf])
    assert np.array_equal(view, [0, 1, 1, -1])


def test_optional_missingness_cutoff_reuses_embedding_and_preserves_ops_default(legacy):
    """Retain nonempty external columns while keeping the old OPS embedding bytes."""
    rng = np.random.default_rng(733)
    complete = rng.normal(size=(40, 8))
    np.testing.assert_array_equal(reference.embed_gene_profiles(complete), legacy[1](complete))
    sparse = complete.copy()
    sparse[rng.random(sparse.shape) < .5] = np.nan
    assert np.all(np.isnan(sparse).mean(axis=0) >= .05)
    with pytest.raises(ValueError):
        reference.embed_gene_profiles(sparse)
    embedded = reference.embed_gene_profiles(sparse, max_missing=1.0)
    assert embedded.shape[0] == len(sparse) and np.isfinite(embedded).all()
    out = om.maxproj_scores(sparse, [f"G{i}" for i in range(40)],
                            views_ix=[np.arange(8)], null_mode="columns", control=None,
                            packed=True, calibration=3, null=5)
    assert np.isfinite(out["p"]).all()


def test_shrunk_similarity_without_controls_uses_all_covered_rows():
    """Compute the explicit all-covered shrinkage constant through the existing embedding."""
    rows, names = _external(n=30, columns=6)
    covered = np.arange(30) < 20
    embedded = reference.embed_gene_profiles(rows[covered], .99, 7)
    norm = (embedded ** 2).sum(axis=1)
    scale = np.sqrt(np.clip(norm + norm.mean(), 1e-24, None))
    expected = np.clip((embedded @ embedded.T) / np.outer(scale, scale), -1, 1)
    observed = om.view_similarity(rows, None, .99, 7, "shrunk", coverage=covered)
    np.testing.assert_array_equal(observed[np.ix_(covered, covered)], expected)
    assert np.isneginf(observed[~covered]).all()


def test_column_null_calibrates_unstructured_gene_profiles():
    """Check pooled p-value uniformity across independent structure-free profiles."""
    from scipy.stats import kstest

    probabilities = []
    for seed in range(4):
        rows, names = _external(seed=seed, n=100)
        out = om.maxproj_scores(rows, names, views_ix=[np.arange(4), np.arange(4, 8), np.arange(8, 12)],
                                null_mode="columns", control=None, packed=True,
                                calibration=8, null=16, seed=seed)
        probabilities.append(out["p"])
    p = np.concatenate(probabilities)
    assert .03 < np.mean(p < .05) < .07
    assert abs(p.mean() - .5) < .035
    assert kstest(p, "uniform").statistic < .05


def test_column_null_recovers_small_planted_complexes():
    """Recover strong complexes in distinct views above unplanted pair scores."""
    from sklearn.metrics import average_precision_score

    rows, names = _external(seed=65, n=160)
    for first, columns in ((0, np.arange(6)), (8, np.arange(6, 12))):
        direction = np.random.default_rng(first + 12).normal(size=6)
        direction /= np.linalg.norm(direction)
        rows[np.ix_(np.arange(first, first + 8), columns)] += 8 * direction
    out = om.maxproj_scores(rows, names, views_ix=[np.arange(6), np.arange(6, 12)],
                            null_mode="columns", similarity="shrunk", control=None,
                            packed=True, calibration=8, null=16, seed=29)
    a, b = np.triu_indices(len(rows), 1)
    positive = ((a < 8) & (b < 8)) | ((a >= 8) & (a < 16) & (b >= 8) & (b < 16))
    assert average_precision_score(positive, -out["p"]) > .7
    assert np.mean(out["p"][positive] < .05) > .8
    assert np.mean(out["view"][positive & (a < 8)] == 0) > .8
    assert np.mean(out["view"][positive & (a >= 8)] == 1) > .8


def test_column_draws_keep_identical_bytes_across_worker_counts():
    """Use fixed per-view and per-draw seeds independent of process completion order."""
    rows, names = _external(n=30, columns=8)
    keywords = dict(views_ix=[np.arange(5), np.arange(3, 8)], null_mode="columns",
                    control=None, packed=True, dtype="float32", calibration=3, null=5, seed=31)
    first = om.maxproj_scores(rows, names, **keywords, workers=1)
    second = om.maxproj_scores(rows, names, **keywords, workers=2)
    for key in ("scores", "projected", "view", "p", "q", "pair_bins", "calibration_counts", "null_counts", "control"):
        a = first[key] if isinstance(first[key], list) else [first[key]]
        b = second[key] if isinstance(second[key], list) else [second[key]]
        assert all(x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
                   for x, y in zip(a, b)), key


@pytest.fixture
def legacy_knn():
    """Load the exact original shared KNN owner to verify default graph bytes."""
    from spacr import ops_clusters

    path = FIXTURES / "1012_knn_legacy_b9af1a272b.py.txt"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "4f5782c64893738d2330c0fc3e6d7b1a3ef4fbd8b450b8b3bd62c3c3ee18d4ac"
    namespace = {"np": np, "_optional": ops_clusters._optional}
    exec(compile(raw, str(path), "exec"), namespace)
    return namespace["_knn_graph"]


def _graph_bytes(graph):
    """Return the exact vertex count, ordered edges and binary edge weights."""
    return (graph.vcount(), graph.get_edgelist(), np.asarray(graph.es["weight"], float).tobytes())


@pytest.mark.parametrize("ties", [False, True])
def test_default_dense_graph_and_labels_match_the_original_owners(legacy, legacy_knn, monkeypatch, ties):
    """Preserve the dense graph edge order, weights and view-label tables exactly."""
    from pandas.testing import assert_frame_equal
    from spacr import ops_clusters

    rng = np.random.default_rng(330)
    n = 18
    probabilities = np.full((n, n), 0.5) if ties else rng.uniform(1e-5, 1.0, (n, n))
    probabilities = (probabilities + probabilities.T) / 2
    np.fill_diagonal(probabilities, 1)
    original = legacy[0]
    with monkeypatch.context() as patch:
        patch.setattr(ops_clusters, "_knn_graph", legacy_knn)
        expected = original.projected_graph(probabilities, 4, 0.2)
    observed = om.projected_graph(probabilities, 4, 0.2)
    assert _graph_bytes(expected) == _graph_bytes(observed)
    labels = np.arange(n) // 6
    view = rng.integers(0, 2, (n, n))
    scores = [rng.normal(size=(n, n)), rng.normal(size=(n, n))]
    views = [{"name": "whole", "block": "W", "columns": list(range(20))},
             {"name": "thin", "block": "P2", "columns": list(range(3))}]
    counts = np.ones((1, len(om.SCORE_GRID) - 1), np.int64)
    expected_tables = original.label_clusters_by_view(labels, expected, view, views, scores, counts, 20)
    observed_tables = om.label_clusters_by_view(labels, observed, view, views, scores, counts, 20)
    for before, after in zip(expected_tables, observed_tables):
        assert_frame_equal(before, after, check_exact=True)


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_packed_graph_matches_dense_without_global_pair_or_square_helpers(monkeypatch, ties, dtype):
    """Match graph edges and weights while streaming the shared KNN rows."""
    rng = np.random.default_rng(921)
    n = 24
    values = np.full((n, n), 0.4) if ties else rng.uniform(1e-5, 1, (n, n))
    probabilities = ((values + values.T) / 2).astype(dtype)
    np.fill_diagonal(probabilities, 1)
    expected = om.projected_graph(probabilities, 5, 0.2)
    packed = probabilities[np.triu_indices(n, 1)]

    def no_global_matrix(*args, **kwargs):
        """Reject matrix-only helpers while the packed graph is constructed."""
        raise AssertionError("packed graph must not request global square or pair indices")

    monkeypatch.setattr(np, "eye", no_global_matrix)
    monkeypatch.setattr(np, "triu_indices", no_global_matrix)
    observed = om.projected_graph(packed, 5, 0.2, n_genes=n)
    assert _graph_bytes(observed) == _graph_bytes(expected)


def test_graph_and_view_labels_exclude_pairs_without_covered_support():
    """Keep unsupported genes isolated and exclude unmeasured edges from view summaries."""
    from pandas.testing import assert_frame_equal

    rng = np.random.default_rng(19)
    n = 15
    values = rng.uniform(1e-4, 1, (n, n))
    probabilities = (values + values.T) / 2
    np.fill_diagonal(probabilities, 1)
    valid = np.ones((n, n), bool)
    valid[-1, :] = valid[:, -1] = False
    np.fill_diagonal(valid, False)
    indices = np.triu_indices(n, 1)
    square = om.projected_graph(probabilities, 3, 0.05, valid=valid)
    packed = om.projected_graph(probabilities[indices], 3, 0.05, n_genes=n, valid=valid[indices])
    assert _graph_bytes(square) == _graph_bytes(packed)
    assert square.degree(n - 1) == packed.degree(n - 1) == 0
    square.add_edge(0, n - 1)
    labels = np.zeros(n, int)
    view = np.where(valid, 0, -1).astype(np.int16)
    scores = [np.where(valid, 2.0, -np.inf)]
    views = [{"name": "measured", "block": "W", "columns": list(range(10))}]
    counts = np.ones((1, len(om.SCORE_GRID) - 1), np.int64)
    dense_tables = om.label_clusters_by_view(labels, square, view, views, scores, counts, 10)
    packed_tables = om.label_clusters_by_view(labels, square, view[indices], views,
                                            [scores[0][indices]], counts, 10)
    for before, after in zip(dense_tables, packed_tables):
        assert_frame_equal(before, after, check_exact=True)
    assert int(dense_tables[1].iloc[0]["edges"]) == square.ecount() - 1
    assert dense_tables[0].iloc[0]["share"] == 1.0
    assert dense_tables[0].iloc[0]["mean_score"] == 2.0

"""CPU-only coverage for the feature/stats block of ``spacr.utils``.

Covers the branches that the rest of the suite never reaches:

* ``filter_dataframe_features`` -- the ``exclude`` list / str branches.
* ``find_non_overlapping_position`` -- exhausting ``max_attempts``.
* ``search_reduction_and_clustering`` -- fractional ``n_neighbors``, the
  ``n_neighbors <= 1`` clamp, and both ``ValueError`` guards.
* ``load_image`` / ``extract_features`` -- the ResNet embedding helpers,
  driven with a tiny local network so nothing is downloaded.

Everything here is offline and runs in well under a second; the heavy
reducers are short-circuited by passing a pre-computed ``embedding``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

torch = pytest.importorskip("torch")
PIL_Image = pytest.importorskip("PIL.Image")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# filter_dataframe_features -- the `exclude` branches
# ---------------------------------------------------------------------------

def _plain_feature_df(n=30, rng=None):
    """Feature frame with no _id/count columns so nothing is pre-dropped."""
    rng = rng or np.random.default_rng(7)
    return pd.DataFrame({
        "cell_channel_0_mean_intensity": rng.normal(10, 2, n),
        "cell_channel_1_mean_intensity": rng.normal(20, 3, n),
        "cell_area": rng.normal(500, 50, n),
        "cell_perimeter": rng.normal(90, 5, n),
        "label_text": ["a"] * n,          # non-numeric -> never a feature
    })


def test_filter_dataframe_features_exclude_list_drops_named_features():
    """A list `exclude` removes exactly those features and keeps the rest."""
    from spacr.utils import filter_dataframe_features
    df = _plain_feature_df()
    excluded = ["cell_area", "cell_channel_1_mean_intensity"]

    filtered, features = filter_dataframe_features(
        df.copy(), channel_of_interest=None, exclude=excluded,
        remove_low_variance_features=False,
        remove_highly_correlated_features=False)

    assert set(excluded).isdisjoint(features)
    assert set(excluded).isdisjoint(filtered.columns)
    # every other numeric column survived, the string column never entered
    assert set(features) == {"cell_channel_0_mean_intensity", "cell_perimeter"}
    assert list(filtered.columns) == features
    assert len(filtered) == len(df)


def test_filter_dataframe_features_exclude_str_removes_single_feature():
    """A bare-string `exclude` removes that one feature via list.remove."""
    from spacr.utils import filter_dataframe_features
    df = _plain_feature_df()

    filtered, features = filter_dataframe_features(
        df.copy(), channel_of_interest=None, exclude="cell_perimeter",
        remove_low_variance_features=False,
        remove_highly_correlated_features=False)

    assert "cell_perimeter" not in features
    assert "cell_perimeter" not in filtered.columns
    assert set(features) == {"cell_channel_0_mean_intensity",
                             "cell_channel_1_mean_intensity", "cell_area"}


def test_filter_dataframe_features_exclude_str_unknown_raises():
    """`features.remove` is exact-match, so an unknown name is a ValueError."""
    from spacr.utils import filter_dataframe_features
    with pytest.raises(ValueError):
        filter_dataframe_features(
            _plain_feature_df(), channel_of_interest=None,
            exclude="not_a_column",
            remove_low_variance_features=False,
            remove_highly_correlated_features=False)


# ---------------------------------------------------------------------------
# find_non_overlapping_position -- attempt budget exhausted
# ---------------------------------------------------------------------------

def test_find_non_overlapping_position_gives_up_and_returns_original():
    """With a threshold nothing can escape, the original point comes back."""
    from spacr.utils import find_non_overlapping_position, check_overlap
    occupied = [(0.0, 0.0)]
    x, y = find_non_overlapping_position(
        3.0, -4.0, occupied, threshold=10_000.0, max_attempts=5)
    assert (x, y) == (3.0, -4.0)
    # and the returned point genuinely still overlaps -> it really gave up
    assert check_overlap((x, y), occupied, threshold=10_000.0) is True


def test_find_non_overlapping_position_finds_a_free_spot():
    """With a tiny threshold the first jitter already succeeds."""
    from spacr.utils import find_non_overlapping_position, check_overlap
    occupied = [(0.0, 0.0), (50.0, 50.0)]
    x, y = find_non_overlapping_position(
        0.0, 0.0, occupied, threshold=1e-9, max_attempts=10)
    assert (x, y) != (0.0, 0.0)
    assert abs(x) <= 10.0 and abs(y) <= 10.0     # inside the offset range
    assert check_overlap((x, y), occupied, threshold=1e-9) is False


def test_find_non_overlapping_position_zero_attempts_short_circuits():
    """max_attempts=0 skips the loop entirely and echoes the input."""
    from spacr.utils import find_non_overlapping_position
    assert find_non_overlapping_position(
        1.5, 2.5, [(1.5, 2.5)], threshold=0.1, max_attempts=0) == (1.5, 2.5)


# ---------------------------------------------------------------------------
# search_reduction_and_clustering
# ---------------------------------------------------------------------------

def _two_blob_embedding(n=20):
    """Pre-computed 2-D embedding with two well separated blobs."""
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 0.05, (n // 2, 2))
    b = rng.normal(8.0, 0.05, (n - n // 2, 2))
    return np.vstack([a, b])


def _numeric_data(n=20, d=4):
    rng = np.random.default_rng(4)
    return rng.normal(size=(n, d))


def test_search_reduction_float_n_neighbors_is_scaled_by_sample_count(capsys):
    """A float n_neighbors is read as a fraction of the row count."""
    pytest.importorskip("umap")
    from spacr.utils import search_reduction_and_clustering

    captured = {}
    import spacr.utils as U

    class _SpyUMAP:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit_transform(self, data):
            raise AssertionError("embedding was supplied; must not fit")

    real_umap = U.umap
    U.umap = type("_M", (), {"UMAP": _SpyUMAP})
    try:
        emb = _two_blob_embedding()
        out_emb, labels = search_reduction_and_clustering(
            _numeric_data(n=20), n_neighbors=0.5, min_dist=0.1,
            metric="euclidean", eps=1.0, min_samples=3, clustering="dbscan",
            reduction_method="umap", verbose=True, embedding=emb, n_jobs=1)
    finally:
        U.umap = real_umap

    # 0.5 * 20 rows -> 10, and no clamp message was printed
    assert captured["n_neighbors"] == 10
    assert "less than 2" not in capsys.readouterr().out
    assert out_emb is emb
    # the two blobs are recovered as two dbscan clusters
    assert set(labels) == {0, 1}
    assert len(labels) == 20


def test_search_reduction_clamps_tiny_n_neighbors_to_two(capsys):
    """A fraction that rounds to 0 is clamped up to 2 with a warning."""
    from spacr.utils import search_reduction_and_clustering

    captured = {}
    import spacr.utils as U

    class _SpyTSNE:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit_transform(self, data):
            raise AssertionError("embedding was supplied; must not fit")

    real_tsne = U.TSNE
    U.TSNE = _SpyTSNE
    try:
        emb = _two_blob_embedding()
        _, labels = search_reduction_and_clustering(
            _numeric_data(n=20), n_neighbors=0.01, min_dist=0.1,
            metric="euclidean", eps=1.0, min_samples=2, clustering="kmeans",
            reduction_method="tsne", verbose=False, embedding=emb, n_jobs=1)
    finally:
        U.TSNE = real_tsne

    assert captured["perplexity"] == 2
    # (message wording is loosely matched: the product string has a typo)
    assert "less than 2" in capsys.readouterr().out
    # kmeans with min_samples=2 -> exactly two labels, one per blob
    assert len(set(labels)) == 2
    assert len(set(labels[:10])) == 1 and len(set(labels[10:])) == 1


def test_search_reduction_int_n_neighbors_one_is_also_clamped(capsys):
    """The clamp is not float-only: an int 1 is bumped to 2 as well."""
    from spacr.utils import search_reduction_and_clustering
    import spacr.utils as U

    captured = {}

    class _SpyTSNE:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    real_tsne = U.TSNE
    U.TSNE = _SpyTSNE
    try:
        search_reduction_and_clustering(
            _numeric_data(n=20), n_neighbors=1, min_dist=0.1,
            metric="euclidean", eps=1.0, min_samples=2, clustering="kmeans",
            reduction_method="tsne", verbose=False,
            embedding=_two_blob_embedding(), n_jobs=1)
    finally:
        U.TSNE = real_tsne

    assert captured["perplexity"] == 2
    assert "less than 2" in capsys.readouterr().out


def test_search_reduction_fits_the_reducer_when_no_embedding_given():
    """Without a pre-computed embedding the reducer is fitted on the data."""
    from spacr.utils import search_reduction_and_clustering
    import spacr.utils as U

    data = _numeric_data(n=20, d=4)
    emb = _two_blob_embedding(n=20)
    seen = {}

    class _SpyUMAP:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def fit_transform(self, numeric_data):
            seen["fitted_on"] = numeric_data
            return emb

    real_umap = U.umap
    U.umap = type("_M", (), {"UMAP": _SpyUMAP})
    try:
        out_emb, labels = search_reduction_and_clustering(
            data, n_neighbors=5, min_dist=0.1, metric="euclidean",
            eps=1.0, min_samples=3, clustering="dbscan",
            reduction_method="umap", verbose=False, n_jobs=1)
    finally:
        U.umap = real_umap

    assert seen["fitted_on"] is data
    assert seen["kwargs"]["min_dist"] == 0.1
    assert out_emb is emb
    assert set(labels) == {0, 1}


def test_search_reduction_rejects_unknown_reduction_method():
    from spacr.utils import search_reduction_and_clustering
    with pytest.raises(ValueError, match="Unsupported reduction method: pca"):
        search_reduction_and_clustering(
            _numeric_data(), n_neighbors=5, min_dist=0.1, metric="euclidean",
            eps=1.0, min_samples=3, clustering="dbscan",
            reduction_method="pca", verbose=False,
            embedding=_two_blob_embedding(), n_jobs=1)


def test_search_reduction_rejects_unknown_clustering_method():
    from spacr.utils import search_reduction_and_clustering
    import spacr.utils as U

    class _SpyTSNE:
        def __init__(self, **kwargs):
            pass

    real_tsne = U.TSNE
    U.TSNE = _SpyTSNE
    try:
        with pytest.raises(ValueError,
                           match="Unsupported clustering method: hdbscan"):
            search_reduction_and_clustering(
                _numeric_data(), n_neighbors=5, min_dist=0.1,
                metric="euclidean", eps=1.0, min_samples=3,
                clustering="hdbscan", reduction_method="tsne", verbose=False,
                embedding=_two_blob_embedding(), n_jobs=1)
    finally:
        U.TSNE = real_tsne


def test_search_reduction_strips_conflicting_reduction_params():
    """Duplicated reducer kwargs are filtered out before construction."""
    from spacr.utils import search_reduction_and_clustering
    import spacr.utils as U

    captured = {}

    class _SpyTSNE:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    real_tsne = U.TSNE
    U.TSNE = _SpyTSNE
    try:
        search_reduction_and_clustering(
            _numeric_data(), n_neighbors=5, min_dist=0.1, metric="euclidean",
            eps=1.0, min_samples=2, clustering="kmeans",
            reduction_method="tsne", verbose=False,
            reduction_param={"perplexity": 99, "n_neighbors": 99,
                             "min_dist": 0.9, "metric": "cosine",
                             "method": "exact", "random_state": 42},
            embedding=_two_blob_embedding(), n_jobs=1)
    finally:
        U.TSNE = real_tsne

    # the conflicting keys were dropped, the innocent one survived
    assert captured["perplexity"] == 5
    assert captured["metric"] == "euclidean"
    assert captured["random_state"] == 42
    assert "min_dist" not in captured and "method" not in captured


# ---------------------------------------------------------------------------
# load_image / extract_features
# ---------------------------------------------------------------------------

def _write_png(path, value=255, size=(37, 53)):
    arr = np.full((size[1], size[0]), value, dtype=np.uint8)
    PIL_Image.fromarray(arr, mode="L").save(path)
    return str(path)


def test_load_image_resizes_normalises_and_batches(tmp_path):
    """A grey-scale PNG becomes a normalised 1x3x224x224 float tensor."""
    from spacr.utils import load_image
    p = _write_png(tmp_path / "white.png", value=255)

    tensor = load_image(p)

    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    # pure white -> ToTensor gives 1.0, then (1 - mean) / std per channel
    means = torch.tensor([0.485, 0.456, 0.406])
    stds = torch.tensor([0.229, 0.224, 0.225])
    expected = (1.0 - means) / stds
    got = tensor[0].reshape(3, -1).mean(dim=1)
    assert torch.allclose(got, expected, atol=1e-3)


def test_load_image_black_image_matches_negative_mean_over_std(tmp_path):
    from spacr.utils import load_image
    tensor = load_image(_write_png(tmp_path / "black.png", value=0))
    expected = -torch.tensor([0.485, 0.456, 0.406]) / torch.tensor(
        [0.229, 0.224, 0.225])
    got = tensor[0].reshape(3, -1).mean(dim=1)
    assert torch.allclose(got, expected, atol=1e-3)


def test_load_image_missing_file_raises(tmp_path):
    from spacr.utils import load_image
    with pytest.raises(FileNotFoundError):
        load_image(str(tmp_path / "does_not_exist.png"))


class _TinyResNet(torch.nn.Module):
    """Stand-in for torchvision resnet: a trunk plus a classifier head."""

    def __init__(self, n_feat=5):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Conv2d(3, n_feat, kernel_size=3, stride=4, padding=1),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = torch.nn.Linear(n_feat, 2)

    def forward(self, x):
        return self.fc(torch.flatten(self.trunk(x), 1))


def test_extract_features_stacks_per_image_embeddings(tmp_path):
    """The classifier head is stripped and one vector per image is returned."""
    from spacr.utils import extract_features
    paths = [_write_png(tmp_path / "a.png", value=255),
             _write_png(tmp_path / "b.png", value=0),
             _write_png(tmp_path / "c.png", value=128)]

    calls = []

    def _factory(pretrained=False):
        calls.append(pretrained)
        torch.manual_seed(0)
        return _TinyResNet(n_feat=5)

    feats = extract_features(paths, resnet=_factory)

    assert calls == [True], "pretrained weights must be requested once"
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (3, 5)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()
    # different images -> different embeddings
    assert not np.allclose(feats[0], feats[1])


def test_extract_features_empty_list_returns_empty_array():
    from spacr.utils import extract_features

    def _factory(pretrained=False):
        return _TinyResNet()

    feats = extract_features([], resnet=_factory)
    assert isinstance(feats, np.ndarray)
    assert feats.size == 0


def test_extract_features_runs_without_grad(tmp_path):
    """Embeddings must be detached -- numpy() fails on a grad-tracking tensor."""
    from spacr.utils import extract_features
    p = _write_png(tmp_path / "g.png", value=200)

    seen = {}

    class _GradSpy(_TinyResNet):
        def forward(self, x):
            return super().forward(x)

    def _factory(pretrained=False):
        torch.manual_seed(1)
        m = _GradSpy(n_feat=4)
        for param in m.parameters():
            param.requires_grad_(True)
        return m

    feats = extract_features([p], resnet=_factory)
    seen["shape"] = feats.shape
    # if torch.no_grad() were missing, .numpy() on the grad-tracking output
    # would have raised RuntimeError before we got here
    assert seen["shape"] == (1, 4)


# ---------------------------------------------------------------------------
# statistics helpers -- value-level assertions
# ---------------------------------------------------------------------------

def _clustered_df(n=60):
    rng = np.random.default_rng(11)
    half = n // 2
    return pd.DataFrame({
        "signal": np.concatenate([rng.normal(0, 1, half),
                                  rng.normal(6, 1, n - half)]),
        "noise": rng.normal(0, 1, n),
        "cluster": [0] * half + [1] * (n - half),
    })


def test_check_normality_rejects_a_uniform_sample():
    from spacr.utils import check_normality
    rng = np.random.default_rng(2)
    assert check_normality(pd.Series(rng.normal(0, 1, 400))) is True
    assert check_normality(pd.Series(rng.uniform(0, 1, 400))) is False


def test_random_forest_feature_importance_ranks_the_separating_feature():
    from spacr.utils import random_forest_feature_importance
    out = random_forest_feature_importance(_clustered_df(), cluster_col="cluster")
    assert list(out.columns) == ["Feature", "Importance"]
    assert set(out["Feature"]) == {"signal", "noise"}
    assert out.iloc[0]["Feature"] == "signal"
    assert out["Importance"].is_monotonic_decreasing
    assert pytest.approx(out["Importance"].sum(), abs=1e-6) == 1.0


def test_perform_statistical_tests_splits_normal_and_non_normal():
    from spacr.utils import perform_statistical_tests
    rng = np.random.default_rng(5)
    n = 60
    df = pd.DataFrame({
        "gaussian": rng.normal(0, 1, n),
        "skewed": rng.pareto(0.7, n),          # very much not normal
        "cluster": [0] * 30 + [1] * 30,
    })
    anova_df, kruskal_df = perform_statistical_tests(df, cluster_col="cluster")

    assert list(anova_df.columns) == ["Feature", "ANOVA_Statistic", "ANOVA_pValue"]
    assert list(kruskal_df.columns) == ["Feature", "Kruskal_Statistic",
                                        "Kruskal_pValue"]
    assert "gaussian" in set(anova_df["Feature"])
    assert "skewed" in set(kruskal_df["Feature"])
    assert "cluster" not in set(anova_df["Feature"]) | set(kruskal_df["Feature"])


def test_combine_results_left_joins_on_feature():
    from spacr.utils import combine_results
    rf = pd.DataFrame({"Feature": ["a", "b"], "Importance": [0.7, 0.3]})
    anova = pd.DataFrame({"Feature": ["a"], "ANOVA_Statistic": [4.0],
                          "ANOVA_pValue": [0.04]})
    kruskal_ = pd.DataFrame({"Feature": ["b"], "Kruskal_Statistic": [2.0],
                             "Kruskal_pValue": [0.15]})
    out = combine_results(rf, anova, kruskal_)

    assert len(out) == 2
    assert list(out["Feature"]) == ["a", "b"]
    assert out.loc[out["Feature"] == "a", "ANOVA_pValue"].iloc[0] == 0.04
    assert np.isnan(out.loc[out["Feature"] == "a", "Kruskal_pValue"].iloc[0])
    assert out.loc[out["Feature"] == "b", "Kruskal_Statistic"].iloc[0] == 2.0
    assert np.isnan(out.loc[out["Feature"] == "b", "ANOVA_Statistic"].iloc[0])


def test_cluster_feature_analysis_end_to_end_columns():
    from spacr.utils import cluster_feature_analysis
    out = cluster_feature_analysis(_clustered_df(), cluster_col="cluster")
    assert isinstance(out, pd.DataFrame)
    assert {"Feature", "Importance"} <= set(out.columns)
    assert set(out["Feature"]) == {"signal", "noise"}
    # every feature landed in exactly one of the two test tables
    for _, row in out.iterrows():
        in_anova = "ANOVA_pValue" in out.columns and not pd.isna(
            row.get("ANOVA_pValue", np.nan))
        in_kruskal = "Kruskal_pValue" in out.columns and not pd.isna(
            row.get("Kruskal_pValue", np.nan))
        assert in_anova != in_kruskal

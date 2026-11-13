"""Table-driven tests for the embedding / clustering units of :mod:`spacr.utils`:
``plot_grid``, ``plot_clusters``, ``plot_embedding``,
``reduction_and_clustering`` and ``search_reduction_and_clustering``.

Companion of ``test_utils_feature_tables_and_plots.py``. Tiny inputs, stubbed
reducers where the old tests stubbed them, CPU only; figures and rcParams are
restored after every test.
"""
from __future__ import annotations

import importlib.util

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.legend import Legend  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox  # noqa: E402
from PIL import Image  # noqa: E402

from spacr import utils as U  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    saved = matplotlib.rcParams.copy()
    yield
    plt.close("all")
    matplotlib.rcParams.update(saved)


def _count_annotation_boxes(ax):
    return sum(isinstance(c, AnnotationBbox) for c in ax.get_children())


def _write_gray_pngs(tmp_path, n, size=8):
    paths = []
    for i in range(n):
        arr = np.zeros((size, size), np.uint8)
        arr[2:6, 2:6] = 50 + 10 * i
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(arr, mode="L").save(p)
        paths.append(str(p))
    return paths


# ---------------------------------------------------------------------------
# plot_grid
# ---------------------------------------------------------------------------

def _tiny(val):
    return np.full((4, 4), val, dtype=np.uint8)


def test_plot_grid_draws_one_coloured_column_per_cluster_within_the_size_cap(subtests, capsys):
    """One panel per cluster with an inset per image, its colour, a label, width <= 200 in."""
    red, green, blue = (1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)
    cases = [
        ("single string cluster", {"alpha": [_tiny(10), _tiny(200)]}, [red], 2, False, True),
        ("three int clusters clamped", {0: [_tiny(17)], 1: [_tiny(17)], 2: [_tiny(17)]},
         [red, green, blue], 100, True, False),
        ("single cluster clamped", {0: [np.full((6, 6), 100, np.uint8)] * 3},
         np.array([[0.2, 0.4, 0.6, 1.0]]), 500, True, False),
        ("two string clusters verbose", {"alpha": [np.full((5, 5), 30, np.uint8)],
                                         "beta": [np.full((5, 5), 30, np.uint8)] * 2},
         np.array([[0.1, 0.1, 0.1, 1.0], [0.7, 0.7, 0.7, 1.0]]), 2, False, True),
        ("two string clusters quiet", {"alpha": [np.zeros((4, 4), np.uint8)],
                                       "beta": [np.ones((4, 4), np.uint8)]},
         [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)], 4, False, False),
        ("two string clusters loud", {"alpha": [np.zeros((4, 4), np.uint8)],
                                      "beta": [np.ones((4, 4), np.uint8)]},
         [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)], 4, False, True),
    ]
    for name, images, colors, size, black, verbose in cases:
        with subtests.test(case=name):
            capsys.readouterr()
            fig = U.plot_grid(images, colors, figuresize=size, black_background=black,
                              verbose=verbose)
            printed = capsys.readouterr().out
            n = len(images)
            per = size if size * n <= 200 else 200 / n
            assert fig.get_size_inches().tolist() == pytest.approx([per * n, per])
            assert len(fig.axes) == n
            labels = list(images)
            for idx, (ax, label) in enumerate(zip(fig.axes, labels)):
                assert len(ax.child_axes) == len(images[label])
                firsts = [int(c.images[0].get_array().flat[0]) for c in ax.child_axes]
                assert firsts == [int(im.flat[0]) for im in images[label]]
                assert ax.axison is False
                assert all(c.axison is False for c in ax.child_axes)
                assert tuple(ax.patches[0].get_facecolor()) == pytest.approx(to_rgba(colors[idx]))
            assert [t.get_text() for t in fig.texts] == [f"Cluster {lab}" for lab in labels]
            assert len(fig.patches) == n
            assert all(t.get_color() == ("white" if black else "black") for t in fig.texts)
            for idx, label in enumerate(labels):
                line = f"Lable: {label} index: {idx}"
                if verbose and isinstance(label, str):
                    assert line in printed
                else:
                    assert "Lable:" not in printed
            plt.close(fig)


# ---------------------------------------------------------------------------
# plot_clusters
# ---------------------------------------------------------------------------

def _blob(n, seed):
    return np.random.RandomState(seed).uniform(-1.0, 1.0, size=(n, 2)).astype(float)


def _collinear(n):
    return np.column_stack([np.arange(n, dtype=float), np.arange(n, dtype=float)])


def _two_blobs():
    rng = np.random.default_rng(2)
    return np.vstack([rng.normal(0, 1, (30, 2)), rng.normal(8, 1, (30, 2))])


def test_plot_clusters_draws_points_hulls_centroids_and_a_themed_legend(subtests):
    """One scatter per cluster, a hull per cluster when asked and possible, labels, legend."""
    from scipy.spatial import ConvexHull

    hull7 = _blob(14, 7)
    two = _two_blobs()
    six = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [4.0, 4.0], [5.0, 4.0], [4.0, 5.0]])
    one_color = [(0.5, 0.2, 0.3, 1.0)]
    cases = [
        dict(name="smooth outline is a 100-sample spline", pts=_blob(14, 5), outlines=True,
             points=True, smooth=True, lines=1, line_len=100, kw=dict(figuresize=10, dot_size=20)),
        dict(name="smooth outline on collinear points is skipped", pts=_collinear(6),
             outlines=True, points=True, smooth=True, lines=0),
        dict(name="convex hull draws one line per simplex", pts=hull7, outlines=True, points=True,
             smooth=False, lines=len(ConvexHull(hull7).simplices)),
        dict(name="convex hull on collinear points is skipped", pts=_collinear(7), outlines=True,
             points=True, smooth=False, lines=0),
        dict(name="tiny cluster has no hull", pts=np.array([[0.0, 0.0], [1.0, 1.0]]),
             outlines=True, points=True, smooth=True, lines=0),
        dict(name="points off -> invisible scatter", pts=_blob(10, 9), outlines=False,
             points=False, smooth=False, lines=0, kw=dict(figuresize=8, dot_size=33),
             invisible=33),
        dict(name="two clusters, outlines off", pts=two, labels=np.array([0] * 30 + [1] * 30),
             colors=[(0.2, 0.4, 0.6), (0.6, 0.2, 0.4)], centers=[(0.0, 0.0), (8.0, 8.0)],
             outlines=False, points=True, smooth=True, lines=0),
        dict(name="two clusters, smooth outlines", pts=two, labels=np.array([0] * 30 + [1] * 30),
             colors=[(0.2, 0.4, 0.6), (0.6, 0.2, 0.4)], centers=[(0.0, 0.0), (8.0, 8.0)],
             outlines=True, points=True, smooth=True, lines=2),
        dict(name="hex colours", pts=six, labels=np.array([0, 0, 0, 1, 1, 1]),
             colors=["#1f77b4", "#ff7f0e"], centers=[(0.3, 0.3), (4.3, 4.3)],
             outlines=False, points=True, smooth=False, lines=0),
    ]
    for case in cases:
        with subtests.test(case=case["name"]):
            pts = case["pts"]
            labels = case.get("labels", np.zeros(len(pts), dtype=int))
            colors = case.get("colors", one_color)
            centers = case.get("centers", [pts.mean(axis=0)])
            fig, ax = plt.subplots()
            U.plot_clusters(ax, pts, labels, colors, centers, plot_outlines=case["outlines"],
                            plot_points=case["points"], smooth_lines=case["smooth"],
                            **case.get("kw", {}))
            n_clusters = len(np.unique(labels))
            assert len(ax.lines) == case["lines"]
            assert all(line.get_linewidth() == 1 for line in ax.lines)
            if "line_len" in case:
                assert ax.lines[0].get_xdata().shape == (case["line_len"],)
            assert len(ax.collections) == n_clusters
            assert sorted(t.get_text() for t in ax.texts) == [str(c) for c in range(n_clusters)]
            assert ax.get_xlabel() == "UMAP Dimension 1"
            assert ax.get_ylabel() == "UMAP Dimension 2"
            if "invisible" in case:
                assert ax.collections[0].get_alpha() == 0
                assert ax.collections[0].get_sizes().tolist() == [case["invisible"]]
            legend = ax.get_legend()
            assert legend is not None
            assert legend.get_frame().get_facecolor()[:3] == pytest.approx(ax.get_facecolor()[:3])
            plt.close(fig)

    with subtests.test(case="legend repainted in theme; missing legend keeps labels"):
        rng = np.random.default_rng(3)
        pts = rng.normal(0.0, 1.0, (12, 2))
        labels = np.zeros(len(pts), dtype=int)
        fig, (themed, legendless) = plt.subplots(1, 2)
        empty_fig, empty_ax = plt.subplots()
        assert isinstance(empty_ax.legend(), Legend)
        for axes in (themed, legendless):
            axes.set_facecolor("#202020")
            axes.xaxis.label.set_color("#ff0000")
        legendless.legend = lambda *args, **kwargs: None
        for axes in (themed, legendless):
            U.plot_clusters(axes, pts, labels, [(0.3, 0.5, 0.7, 1.0)], [pts.mean(axis=0)],
                            plot_outlines=False, plot_points=True, smooth_lines=False)
        drawn = themed.get_legend()
        assert drawn is not None
        assert drawn.get_frame().get_facecolor()[:3] == pytest.approx(themed.get_facecolor()[:3])
        assert [t.get_color() for t in drawn.get_texts()] == ["#ff0000"]
        assert legendless.get_legend() is None
        assert legendless.get_xlabel() == "UMAP Dimension 1"
        assert legendless.get_ylabel() == "UMAP Dimension 2"
        assert len(legendless.collections) == len(themed.collections) == 1
        plt.close(fig)
        plt.close(empty_fig)


# ---------------------------------------------------------------------------
# plot_embedding
# ---------------------------------------------------------------------------

_WATCHED_RC = ("text.color", "figure.facecolor", "axes.facecolor",
               "axes.labelcolor", "xtick.color", "ytick.color")


def test_plot_embedding_draws_clusters_and_optional_images_without_leaking_theme(subtests, tmp_path):
    """A scatter per cluster, an image per cluster if paths exist, own theme, rcParams intact."""
    two_colors = np.array([[0.9, 0.1, 0.1, 1.0], [0.1, 0.1, 0.9, 1.0]])
    six = np.array([[0.0, 0.0], [0.4, 0.1], [0.1, 0.5], [4.0, 4.0], [4.3, 4.1], [4.1, 4.4]])
    four = np.array([[0.0, 0.0], [1.0, 0.2], [0.2, 1.0], [5.0, 5.0]])
    cases = [
        dict(name="with images", emb=six, paths=_write_gray_pngs(tmp_path, 6),
             labels=np.array([0, 0, 0, 1, 1, 1]), by_cluster=True, outlines=True, images=True,
             black=False, size=5, dot=20, canvas=False, boxes=2),
        dict(name="no paths skips overlay", emb=four, paths=None,
             labels=np.array([0, 0, 0, 1]), by_cluster=False, outlines=False, images=True,
             black=True, size=4, dot=10, canvas=True, boxes=0),
        dict(name="dark, images off", emb=four, paths=None, labels=np.array([0, 0, 0, 1]),
             by_cluster=False, outlines=False, images=False, black=True, size=4, dot=10,
             canvas=True, boxes=0),
    ]
    for case in cases:
        with subtests.test(case=case["name"]):
            before = {k: matplotlib.rcParams[k] for k in _WATCHED_RC}
            fig = U.plot_embedding(
                case["emb"], case["paths"], case["labels"], image_nr=1, img_zoom=0.3,
                colors=two_colors, plot_by_cluster=case["by_cluster"],
                plot_outlines=case["outlines"], plot_points=True, plot_images=case["images"],
                smooth_lines=False, black_background=case["black"], figuresize=case["size"],
                dot_size=case["dot"], remove_image_canvas=case["canvas"], verbose=False)
            assert isinstance(fig, plt.Figure)
            ax = fig.axes[0]
            assert len(ax.collections) == 2
            assert _count_annotation_boxes(ax) == case["boxes"]
            assert sorted(t.get_text() for t in ax.texts) == ["0", "1"]
            if case["black"]:
                assert fig.get_facecolor()[:3] == (0.0, 0.0, 0.0)
            assert {k: matplotlib.rcParams[k] for k in _WATCHED_RC} == before
            plt.close(fig)


# ---------------------------------------------------------------------------
# reduction_and_clustering
# ---------------------------------------------------------------------------

def _numeric_blob(n=30, d=4, seed=1):
    rs = np.random.RandomState(seed)
    return np.vstack([rs.normal(0.0, 1.0, size=(n // 2, d)),
                      rs.normal(6.0, 1.0, size=(n - n // 2, d))])


class _StubReducer:
    """Minimal 'already fitted' reducer usable in transform mode."""

    def __init__(self):
        self.seen = None

    def transform(self, x):
        self.seen = np.asarray(x)
        return np.asarray(x)[:, :2].astype(float)


class _GpuArray:
    """Stands in for a cuML device array."""

    def __init__(self, values):
        self._values = np.asarray(values)

    def get(self):
        return self._values


class _SealedReducer:
    """Returns a device array and accepts no new attributes (slotted cuML estimator)."""

    __slots__ = ()

    def fit_transform(self, values):
        return _GpuArray(np.zeros((len(values), 2), float))


class _FittedDeviceModel:
    def transform(self, values):
        return _GpuArray(np.zeros((len(values), 2), float))


def _rc(data, n_neighbors=5, eps=0.5, min_samples=2, clustering="kmeans", **kw):
    return U.reduction_and_clustering(data, n_neighbors=n_neighbors, min_dist=0.1,
                                      metric="euclidean", eps=eps, min_samples=min_samples,
                                      clustering=clustering, **kw)


def test_reduction_and_clustering_embeds_to_two_dimensions_and_labels_every_row(subtests, monkeypatch, capsys):
    """Fit or transform yields a host (n, 2) embedding, n labels and the reducer used."""
    from sklearn.manifold import TSNE

    from spacr import accelerator, gpu_reduce

    stub = _StubReducer()
    transform_data = _numeric_blob(n=20, d=4, seed=3)
    fitted = _FittedDeviceModel()
    rng6 = np.random.default_rng(0).normal(size=(6, 3))

    def metal(m):
        m.setenv("SPACR_USE_RAPIDS", "0")
        m.setattr(accelerator, "resolve", lambda: accelerator.Accelerator(
            kind="mps", device="mps", label="Declared Metal", detected=True, usable=True))

    def sealed(m):
        m.setattr(gpu_reduce, "make_reducer",
                  lambda name, prefer_gpu=False, **kwargs: (_SealedReducer(), "cpu"))

    cases = [
        dict(name="float n_neighbors is a fraction of n", data=_numeric_blob(30),
             kw=dict(n_neighbors=0.3, reduction_method="tsne"), perplexity=9, labels={0, 1}),
        dict(name="small int n_neighbors clamped to 2", data=_numeric_blob(20),
             kw=dict(n_neighbors=1, reduction_method="tsne"), perplexity=2),
        dict(name="tiny fraction clamped to 2", data=_numeric_blob(10),
             kw=dict(n_neighbors=0.05, reduction_method="tsne"), perplexity=2),
        dict(name="verbose fit reports progress", data=_numeric_blob(20),
             kw=dict(eps=5.0, clustering="dbscan", reduction_method="tsne", verbose=True, n_jobs=1),
             printed=["Trained and fit reducer", "Embedding shape: (20, 2)"]),
        dict(name="umap uses requested neighbours", data=_numeric_blob(30, seed=11),
             kw=dict(reduction_method="umap", n_jobs=1), labels={0, 1},
             attrs=dict(n_neighbors=5, n_components=2)),
        dict(name="transform reuses the model", data=transform_data,
             kw=dict(eps=1.0, min_samples=3, clustering="dbscan", reduction_method="umap",
                     verbose=True, mode="transform", model=stub, n_jobs=1),
             printed=["Fit data to reducer", "Embedding shape: (20, 2)"], reducer=stub,
             embedding=transform_data[:, :2], found_cluster=True),
        dict(name="device embedding brought to host", data=rng6,
             kw=dict(clustering="dbscan", reduction_method="umap"), patch=sealed,
             reducer_type=_SealedReducer),
        dict(name="device embedding from saved model", data=rng6,
             kw=dict(clustering="dbscan", mode="transform", model=fitted), reducer=fitted),
        dict(name="perplexity lowered to fit the data",
             data=np.random.default_rng(0).normal(size=(5, 3)),
             kw=dict(clustering="dbscan", reduction_method="tsne", verbose=True,
                     reducer_options={"perplexity": 30.0}),
             printed=["Adjusted t-SNE perplexity from 30 to 4"]),
        dict(name="spectral rbf takes no neighbours",
             data=np.random.default_rng(1).random((24, 3)),
             kw=dict(reduction_method="spectral", reducer_options={"affinity": "rbf"}),
             labels={0, 1}, attrs=dict(affinity="rbf"), no_neighbors_param=True),
        dict(name="metal t-SNE runs on the CPU", data=_numeric_blob(20),
             kw=dict(n_neighbors=3, clustering="dbscan", reduction_method="tsne",
                     prefer_gpu=True, n_jobs=1, reducer_options={"max_iter": 250}),
             patch=metal, reducer_type=TSNE, backend="cpu",
             printed=["tsne uses the CPU on Metal; PCA and KMeans can use the GPU."]),
    ]
    for case in cases:
        with subtests.test(case=case["name"]):
            data = case["data"]
            before = data.copy()
            capsys.readouterr()
            with monkeypatch.context() as m:
                if "patch" in case:
                    case["patch"](m)
                embedding, labels, reducer = _rc(data, **case["kw"])
            printed = capsys.readouterr().out
            n = len(data)
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (n, 2)
            assert np.isfinite(embedding).all()
            assert len(labels) == n
            np.testing.assert_array_equal(data, before)
            if "perplexity" in case:
                assert reducer.perplexity == case["perplexity"]
            if "labels" in case:
                assert set(np.unique(labels)) == case["labels"]
            for attr, value in case.get("attrs", {}).items():
                assert getattr(reducer, attr) == value
            if case.get("no_neighbors_param"):
                assert reducer.get_params().get("n_neighbors") is None
            for fragment in case.get("printed", []):
                assert fragment in printed
            if "reducer" in case:
                assert reducer is case["reducer"]
            if "reducer_type" in case:
                assert isinstance(reducer, case["reducer_type"])
            if "backend" in case:
                assert reducer._spacr_backend == case["backend"]
            if "embedding" in case:
                np.testing.assert_allclose(embedding, case["embedding"])
                assert stub.seen.shape == data.shape
            if case.get("found_cluster"):
                assert set(np.unique(labels)) - {-1}


def test_reduction_and_clustering_refuses_settings_it_cannot_honour(subtests, monkeypatch):
    """Unknown methods, a missing/unfit model, bad perplexity and GPU-without-cuML all refuse."""
    rng6 = np.random.default_rng(0).normal(size=(6, 3))
    cases = [
        ("unknown reduction", np.zeros((6, 3)),
         dict(n_neighbors=3, clustering="dbscan", reduction_method="bogus"),
         ValueError, "Unsupported reduction method: bogus", None),
        ("transform without model", np.zeros((5, 3)),
         dict(n_neighbors=3, clustering="dbscan", mode="transform", model=None),
         ValueError, "Model is None", None),
        ("transform with default model", np.zeros((5, 3)),
         dict(n_neighbors=3, clustering="dbscan", mode="transform"), ValueError, None, None),
        ("unknown clustering", _numeric_blob(n=8),
         dict(n_neighbors=3, clustering="hdbscan", reduction_method="tsne"), ValueError, None, None),
        ("zero perplexity", rng6,
         dict(clustering="dbscan", reduction_method="tsne", reducer_options={"perplexity": 0}),
         ValueError, "perplexity", None),
        ("t-SNE cannot transform", rng6,
         dict(clustering="dbscan", reduction_method="tsne", mode="transform", model=object()),
         ValueError, "embedding_by_controls", None),
    ]
    if importlib.util.find_spec("cuml") is None:
        cases.append(("GPU requested without cuML", np.random.default_rng(0).random((20, 4)),
                      dict(min_samples=3, reduction_method="tsne", prefer_gpu=True),
                      RuntimeError, "GPU was requested", {"SPACR_DEVICE": "cpu"}))
    for name, data, kw, exc, match, env in cases:
        with subtests.test(case=name):
            with monkeypatch.context() as m:
                for key, value in (env or {}).items():
                    m.setenv(key, value)
                with pytest.raises(exc, match=match):
                    _rc(data, **kw)


# ---------------------------------------------------------------------------
# search_reduction_and_clustering
# ---------------------------------------------------------------------------

def _two_blob_embedding(n=20):
    rng = np.random.default_rng(3)
    return np.vstack([rng.normal(0.0, 0.05, (n // 2, 2)), rng.normal(8.0, 0.05, (n - n // 2, 2))])


def _numeric_data(n=20, d=4):
    return np.random.default_rng(4).normal(size=(n, d))


def _spy_reducer(record, embedding=None):
    class _Spy:
        def __init__(self, **kwargs):
            record["kwargs"] = kwargs

        def fit_transform(self, data):
            if embedding is None:
                raise AssertionError("embedding was supplied; must not fit")
            record["fitted_on"] = data
            return embedding

    return _Spy


def test_search_reduction_and_clustering_sizes_the_reducer_and_clusters_the_embedding(subtests, monkeypatch, capsys):
    """Neighbour count scaled/clamped, conflicting params stripped, embedding clustered."""
    data = _numeric_data()
    cases = [
        dict(name="float n_neighbors scaled by rows", method="umap", n_neighbors=0.5,
             clustering="dbscan", min_samples=3, verbose=True, embedding=True,
             kwargs={"n_neighbors": 10}, clamp_msg=False, labels={0, 1}),
        dict(name="tiny fraction clamped", method="tsne", n_neighbors=0.01, clustering="kmeans",
             min_samples=2, embedding=True, kwargs={"perplexity": 2}, clamp_msg=True,
             split_blobs=True),
        dict(name="int one clamped", method="tsne", n_neighbors=1, clustering="kmeans",
             min_samples=2, embedding=True, kwargs={"perplexity": 2}, clamp_msg=True),
        dict(name="fits the reducer without an embedding", method="umap", n_neighbors=5,
             clustering="dbscan", min_samples=3, embedding=False, kwargs={"min_dist": 0.1},
             labels={0, 1}, fitted=True),
        dict(name="conflicting params stripped", method="tsne", n_neighbors=5,
             clustering="kmeans", min_samples=2, embedding=True,
             extra=dict(reduction_param={"perplexity": 99, "n_neighbors": 99, "min_dist": 0.9,
                                         "metric": "cosine", "method": "exact",
                                         "random_state": 42}),
             kwargs={"perplexity": 5, "metric": "euclidean", "random_state": 42},
             missing=["min_dist", "method"]),
    ]
    for case in cases:
        with subtests.test(case=case["name"]):
            record = {}
            emb = _two_blob_embedding()
            spy = _spy_reducer(record, embedding=None if case["embedding"] else emb)
            capsys.readouterr()
            with monkeypatch.context() as m:
                if case["method"] == "umap":
                    m.setattr(U, "umap", type("_M", (), {"UMAP": spy}))
                else:
                    m.setattr(U, "TSNE", spy)
                out_emb, labels = U.search_reduction_and_clustering(
                    data, n_neighbors=case["n_neighbors"], min_dist=0.1, metric="euclidean",
                    eps=1.0, min_samples=case["min_samples"], clustering=case["clustering"],
                    reduction_method=case["method"], verbose=case.get("verbose", False),
                    embedding=emb if case["embedding"] else None, n_jobs=1,
                    **case.get("extra", {}))
            printed = capsys.readouterr().out
            assert out_emb is emb
            assert len(labels) == len(data)
            for key, value in case["kwargs"].items():
                assert record["kwargs"][key] == value, key
            for key in case.get("missing", []):
                assert key not in record["kwargs"]
            if "clamp_msg" in case:
                assert ("less than 2" in printed) is case["clamp_msg"]
            if "labels" in case:
                assert set(labels) == case["labels"]
            if case.get("split_blobs"):
                assert len(set(labels)) == 2
                assert len(set(labels[:10])) == 1 and len(set(labels[10:])) == 1
            if case.get("fitted"):
                assert record["fitted_on"] is data

    refusals = [
        ("unknown reduction", "pca", "dbscan", "Unsupported reduction method: pca"),
        ("unknown clustering", "tsne", "hdbscan", "Unsupported clustering method: hdbscan"),
    ]
    for name, method, clustering, match in refusals:
        with subtests.test(case=name):
            with monkeypatch.context() as m:
                m.setattr(U, "TSNE", _spy_reducer({}))
                with pytest.raises(ValueError, match=match):
                    U.search_reduction_and_clustering(
                        data, n_neighbors=5, min_dist=0.1, metric="euclidean", eps=1.0,
                        min_samples=3, clustering=clustering, reduction_method=method,
                        verbose=False, embedding=_two_blob_embedding(), n_jobs=1)

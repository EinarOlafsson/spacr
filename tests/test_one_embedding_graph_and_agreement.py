"""One embedder, one PHATE graph, one Leiden call and one label agreement count.

Duplicate audit D6, D10 and B19. The three standardise-then-PCA embedders
(``ops_reference._GeneEmbedding``, ``gene_measurement_compare._RowEmbedder``
and ``ops_clusters.embed_profiles``), the per-batch TVN standardise and the
``'c0'`` profile recipe all take their column mean and SD from
``ops_tvn.batch_mean_sd``; the three PHATE graph builders share
``ops_clusters._phate_graph`` and the Leiden calls share
``ops_clusters.leiden_partitions``; the label-image agreement of
``ops_large_sites`` and ``model_compare`` counts pairs with
``measure._label_overlaps``.

Each rebased caller is checked against a verbatim copy of the code it
replaced, on fixed fixtures: bit-identical where the arithmetic is the same,
and within a stated tolerance where float32 sums became float64 sums.
"""
import numpy as np
import pandas as pd
import pytest

from spacr import gene_measurement_compare as gmc
from spacr import model_compare as mc
from spacr import ops_clusters as oc
from spacr import ops_large_sites as ols
from spacr import ops_reference as orf
from spacr import ops_tvn


def _profiles(seed=0, n=80, d=14):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)) @ rng.normal(size=(d, d)) + rng.uniform(-5, 5, d)
    x[rng.random(x.shape) < 0.02] = np.nan
    x[:, 3] = 2.5
    x[: n // 5, 5] = np.nan
    x[4, 7] = np.inf
    return x


# ---------------------------------------------------------------- D6 -------

def _old_gene_embedding(values, variance=0.99, seed=0):
    from sklearn.decomposition import PCA

    x = np.asarray(values, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    keep = np.isnan(x).mean(axis=0) < 0.05
    x = x[:, keep]
    fill = np.nanmedian(x, axis=0)
    x = np.where(np.isnan(x), fill, x)
    spread = x.std(axis=0)
    varying = spread > 0
    mean, spread = x[:, varying].mean(axis=0), spread[varying]
    pca = PCA(n_components=variance, random_state=seed)
    rows = pca.fit_transform((x[:, varying] - mean) / spread)

    def transform(other):
        y = np.asarray(other, dtype=float)[:, keep]
        y = np.where(np.isfinite(y), y, fill)
        return pca.transform((y[:, varying] - mean) / spread)

    return rows, transform


def _old_embed_profiles(frame, features, *, variance=0.99, seed=0):
    from sklearn.decomposition import PCA

    x = frame[list(features)].to_numpy(float)
    x = np.where(np.isfinite(x), x, np.nan)
    keep = np.isnan(x).mean(axis=0) < 0.05
    x = x[:, keep]
    x = np.where(np.isnan(x), np.nanmedian(x, axis=0), x)
    spread = x.std(axis=0)
    x = (x[:, spread > 0] - x[:, spread > 0].mean(axis=0)) / spread[spread > 0]
    pca = PCA(n_components=variance, random_state=seed).fit(x)
    return pca.transform(x), int(pca.n_components_), int(keep.sum())


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("seed", range(4))
def test_gene_embedding_and_row_embedder_are_bit_identical_to_the_old_copies(seed, order):
    x = np.asarray(_profiles(seed), order=order)
    guides = _profiles(seed + 10, n=30)
    rows, transform = _old_gene_embedding(x, 0.9, seed)
    new = orf._GeneEmbedding(x, 0.9, seed)
    np.testing.assert_array_equal(new.rows, rows)
    np.testing.assert_array_equal(new.transform(guides), transform(guides))
    np.testing.assert_array_equal(orf.embed_gene_profiles(x, 0.9, seed), rows)
    row_embedder = gmc._RowEmbedder(x, variance=0.9, seed=seed)
    np.testing.assert_array_equal(row_embedder.rows, rows)
    np.testing.assert_array_equal(row_embedder(guides), transform(guides))


@pytest.mark.parametrize("seed", range(4))
def test_embed_profiles_is_bit_identical_to_the_old_copy(seed):
    x = _profiles(seed)
    columns = [f"f{i}" for i in range(x.shape[1])]
    frame = pd.DataFrame(x, columns=columns)
    old = _old_embed_profiles(frame, columns, seed=seed)
    new = oc.embed_profiles(frame, columns, seed=seed)
    np.testing.assert_array_equal(new[0], old[0])
    assert new[1:] == old[1:]


def test_the_embedders_take_their_column_statistics_from_batch_mean_sd(monkeypatch):
    seen = []
    real = ops_tvn.batch_mean_sd

    def spy(*args, **kwargs):
        seen.append(kwargs.get("zero_sd"))
        return real(*args, **kwargs)

    monkeypatch.setattr(ops_tvn, "batch_mean_sd", spy)
    x = _profiles(1)
    orf._GeneEmbedding(x)
    gmc._RowEmbedder(x)
    oc.embed_profiles(pd.DataFrame(x), list(range(x.shape[1])))
    assert seen == ["keep", "keep", "keep"]
    with pytest.raises(ValueError):
        orf._GeneEmbedding(x, rows_from="nope")


def test_column_mean_sd_policies():
    x = _profiles(2)[:, :8]
    x = np.where(np.isfinite(x), x, 0.0)
    x[:, 3] = 2.5
    mean, sd = ops_tvn._column_mean_sd(x, zero_sd="keep")
    np.testing.assert_array_equal(mean, x.mean(0))
    np.testing.assert_array_equal(sd, x.std(0))
    assert sd[3] == 0
    assert ops_tvn._column_mean_sd(x, zero_sd="exact")[1][3] == 1.0
    rows = np.arange(0, len(x), 3)
    mean, sd = ops_tvn._column_mean_sd(x, rows, zero_sd="keep")
    np.testing.assert_array_equal(mean, x[rows].mean(0))
    np.testing.assert_array_equal(sd, x[rows].std(0))
    column_major = np.asfortranarray(x)
    mean, sd = ops_tvn._column_mean_sd(column_major, zero_sd="keep")
    np.testing.assert_array_equal(mean, column_major.mean(0))
    np.testing.assert_array_equal(sd, column_major.std(0))
    with pytest.raises(ValueError):
        ops_tvn.batch_mean_sd(x, np.arange(len(x)), np.zeros(len(x)), zero_sd="nope")


def _old_standardise(self, values, reference=None):
    x = np.array(values, dtype=np.float32)[:, self.columns]
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = np.broadcast_to(self.medians, x.shape)[bad]
    fit = x if reference is None else x[np.asarray(reference, bool)]
    mean, spread = fit.mean(0), fit.std(0)
    spread[spread == 0] = 1
    return (x - mean) / spread


def _tvn_wells(seed=0, wells=4, n=150, d=12):
    rng = np.random.default_rng(seed)
    data = {}
    for w in range(wells):
        x = (rng.normal(size=(n, d)) * rng.uniform(0.5, 3, d) + rng.uniform(-20, 200, d)).astype(np.float32)
        x[rng.random(x.shape) < 0.01] = np.nan
        control = np.zeros(n, bool)
        control[::4] = True
        use = rng.random(n) < 0.9
        use[control] = True
        data[f"w{w}"] = (x, control, use)
    return data


def _component_agreement(old, new):
    """Smallest per-component |corr|, largest absolute difference, and that
    difference over the largest component SD."""
    corr = [abs(np.corrcoef(old[:, j], new[:, j])[0, 1]) for j in range(old.shape[1])]
    diff = float(np.abs(old.astype(np.float64) - new).max())
    return min(corr), diff, diff / float(old.astype(np.float64).std(0).max())


def test_streamed_tvn_on_the_shared_standardise_is_within_tolerance(monkeypatch):
    data = _tvn_wells()

    def run():
        model = ops_tvn.fit_tvn_streamed(list(data), lambda w: data[w], pca_sample=400)
        return model, np.vstack([model.transform(data[w][0], w, data[w][2]) for w in data])

    new_model, new = run()
    monkeypatch.setattr(ops_tvn.TVNModel, "standardise", _old_standardise)
    old_model, old = run()
    assert new.shape == old.shape
    assert new_model.pca.n_components_ == old_model.pca.n_components_
    worst_corr, max_abs, relative = _component_agreement(old, new)
    print(f"TVN components: min |corr| {worst_corr:.10f}, max abs diff {max_abs:.3g}"
          f" ({relative:.3g} of the largest component SD)")
    assert worst_corr >= 0.999999
    assert relative < 2e-3


def _old_c0_stats(values, rows=None, *, zero_sd, chunk=65536):
    x = values if rows is None else values[rows]
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1
    return mu, sd


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_c0_profile_embedding_on_the_shared_standardise(monkeypatch, dtype):
    data = _tvn_wells(1, n=300)
    metas = [pd.DataFrame({"batch_values": [w] * len(v[0])}) for w, v in data.items()]

    def values():
        return [np.nan_to_num(v[0], nan=0.0).astype(dtype) for v in data.values()]

    recipe = {"pca": "c0"}
    new, k_new = ops_tvn._profile_embedding(values(), metas, recipe)
    monkeypatch.setattr(ops_tvn, "_column_mean_sd", _old_c0_stats)
    old, k_old = ops_tvn._profile_embedding(values(), metas, recipe)
    assert k_new == k_old
    if dtype is np.float64:
        np.testing.assert_array_equal(new, old)
    else:
        worst_corr, max_abs, relative = _component_agreement(old, new)
        print(f"c0 float32 components: min |corr| {worst_corr:.10f}, max abs diff {max_abs:.3g}"
              f" ({relative:.3g} of the largest component SD)")
        assert worst_corr >= 0.999999
        assert relative < 1e-4


# ---------------------------------------------------------------- D10 ------

def _old_phate_leiden(components, resolutions, *, knn=10, metric="cosine", seed=42, n_jobs=1):
    import igraph
    import leidenalg
    import phate

    model = phate.PHATE(random_state=seed, n_jobs=n_jobs, knn=knn, knn_dist=metric, verbose=False)
    model.fit(components)
    w = model.graph.diff_op
    w = (w + w.T) / 2
    if hasattr(w, "tocoo"):
        w = w.tocoo()
    else:
        from scipy.sparse import coo_matrix
        w = coo_matrix(w)
    upper = w.row < w.col
    graph = igraph.Graph(n=components.shape[0], edges=list(zip(w.row[upper], w.col[upper])),
                         directed=False)
    graph.es["weight"] = w.data[upper].tolist()
    out = {}
    for resolution in resolutions:
        partition = leidenalg.find_partition(
            graph, leidenalg.RBConfigurationVertexPartition, weights="weight",
            n_iterations=-1, seed=seed, resolution_parameter=float(resolution))
        out[resolution] = np.array(partition.membership)
    return out, graph


def _old_fused_phate_graph(fused, seed, *, knn=10):
    import igraph
    import phate
    from scipy.sparse import coo_matrix

    distance = fused.max() - fused
    np.fill_diagonal(distance, 0)
    model = phate.PHATE(random_state=int(seed), n_jobs=1, knn=int(knn),
                        knn_dist="precomputed_distance", verbose=False)
    model.fit(distance)
    w = model.graph.diff_op
    w = (w + w.T) / 2
    w = w.tocoo() if hasattr(w, "tocoo") else coo_matrix(w)
    upper = w.row < w.col
    return igraph.Graph(n=distance.shape[0], edges=np.stack([w.row[upper], w.col[upper]], 1).tolist(),
                        directed=False), w.data[upper]


def _old_leiden_short(graph, resolution, seed, passes):
    import leidenalg

    return np.asarray(leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition, weights="weight", n_iterations=int(passes),
        seed=int(seed), resolution_parameter=float(resolution)).membership)


def _clustered(seed=0, n=90, d=6):
    pytest.importorskip("phate")
    pytest.importorskip("leidenalg")
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(4, d)) * 4
    return centres[rng.integers(0, 4, n)] + rng.normal(size=(n, d))


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_phate_leiden_is_identical_to_the_old_block(metric):
    x = _clustered()
    resolutions = [0.5, 1.0, 2.0]
    old, old_graph = _old_phate_leiden(x, resolutions, knn=8, metric=metric, seed=3)
    new = oc.phate_leiden(x, resolutions, knn=8, metric=metric, seed=3)
    assert list(new) == list(old)
    for r in resolutions:
        np.testing.assert_array_equal(new[r], old[r])
    graph = oc.gene_graph(x, "phate", seed=3, knn=8, n_jobs=1, metric=metric)
    assert graph.get_edgelist() == old_graph.get_edgelist()
    assert graph.es["weight"] == old_graph.es["weight"]
    swept = oc.leiden_partitions(graph, resolutions, [3])
    for r in resolutions:
        np.testing.assert_array_equal(swept[(3, r)], old[r])


def test_fused_graph_and_short_leiden_are_identical_to_the_old_blocks():
    x = _clustered(1, n=60)
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    fused = unit @ unit.T + 1.0
    old_graph, old_weights = _old_fused_phate_graph(fused.copy(), 5, knn=7)
    graph, weights = oc.fused_phate_graph(fused.copy(), 5, knn=7)
    assert graph.get_edgelist() == old_graph.get_edgelist()
    np.testing.assert_array_equal(weights, old_weights)
    graph.es["weight"] = weights.tolist()
    for passes in (1, 2, -1):
        np.testing.assert_array_equal(oc.leiden_short(graph, 1.0, 9, passes),
                                      _old_leiden_short(graph, 1.0, 9, passes))


# ---------------------------------------------------------------- B19 ------

def _blobs(seed, shape=(64, 64), n=14, shift=(0, 0)):
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, np.int32)
    for k in range(1, n + 1):
        y, x = rng.integers(0, shape[0] - 8), rng.integers(0, shape[1] - 8)
        h, w = rng.integers(4, 14, 2)
        out[y + shift[0]:y + shift[0] + h, x + shift[1]:x + shift[1] + w] = rng.integers(1, 40)
    return out


def _fixture_pairs():
    return [(_blobs(s), _blobs(s, shift=(1, 2)) if s % 2 else _blobs(s + 100)) for s in range(6)] + [
        (np.zeros((16, 16), np.int32), _blobs(7, (16, 16), 3)),
        (np.zeros((16, 16), np.int32), np.zeros((16, 16), np.int32)),
    ]


def _old_pair_overlaps(first, second):
    a, b = np.asarray(first).ravel().astype(np.int64), np.asarray(second).ravel().astype(np.int64)
    area_a, area_b = np.bincount(a), np.bincount(b)
    both = (a > 0) & (b > 0)
    pairs, counts = np.unique(np.stack([a[both], b[both]]), axis=1, return_counts=True)
    return a, area_a, area_b, pairs, counts


def _old_mask_agreement(labels, reference, *, iou=0.5, drop_edge=True):
    labels, reference = np.asarray(labels), np.asarray(reference)
    if drop_edge:
        labels, _ = ols.apply_edge_policy(labels, "exclude")
        reference, _ = ols.apply_edge_policy(reference, "exclude")
    a, b = labels.ravel().astype(np.int64), reference.ravel().astype(np.int64)
    ids_a, ids_b = np.unique(a[a > 0]), np.unique(b[b > 0])
    area_a, area_b = np.bincount(a), np.bincount(b)
    both = (a > 0) & (b > 0)
    pairs, counts = np.unique(np.stack([a[both], b[both]]), axis=1, return_counts=True)
    scores = (counts / (area_a[pairs[0]] + area_b[pairs[1]] - counts)) if counts.size else np.array([])
    hit = scores >= iou
    matched = int(hit.sum())
    precision = matched / ids_a.size if ids_a.size else float("nan")
    recall = matched / ids_b.size if ids_b.size else float("nan")
    f1 = (2 * matched / (ids_a.size + ids_b.size)) if (ids_a.size + ids_b.size) else float("nan")
    return {"n_spacr": int(ids_a.size), "n_reference": int(ids_b.size), "matched": matched,
            "precision": precision, "recall": recall, "f1": f1,
            "median_iou": float(np.median(scores[hit])) if matched else float("nan"),
            "count_ratio": ids_a.size / ids_b.size if ids_b.size else float("nan")}


def _old_split_merge(labels, reference):
    labels, _ = ols.apply_edge_policy(labels, "exclude")
    reference, _ = ols.apply_edge_policy(reference, "exclude")
    a, b = labels.ravel().astype(np.int64), reference.ravel().astype(np.int64)
    area_a, area_b = np.bincount(a), np.bincount(b)
    both = (a > 0) & (b > 0)
    pairs, counts = np.unique(np.stack([a[both], b[both]]), axis=1, return_counts=True)
    a_major = counts > 0.5 * area_a[pairs[0]]
    b_major = counts > 0.5 * area_b[pairs[1]]
    per_ref = np.bincount(pairs[1][a_major], minlength=area_b.size)
    per_mine = np.bincount(pairs[0][b_major], minlength=area_a.size)
    return {"splits": int((per_ref >= 2).sum()), "merges": int((per_mine >= 2).sum())}


def _old_object_overlap(mask_a, mask_b):
    a, b = mc._as_labels(mask_a), mc._as_labels(mask_b)
    values_a, index_a = np.unique(a.ravel(), return_inverse=True)
    values_b, index_b = np.unique(b.ravel(), return_inverse=True)
    n_rows, n_cols = values_a.size, values_b.size
    table = np.bincount(index_a * n_cols + index_b, minlength=n_rows * n_cols).reshape(n_rows, n_cols)
    row0 = 1 if (n_rows and values_a[0] == 0) else 0
    col0 = 1 if (n_cols and values_b[0] == 0) else 0
    overlap = table[row0:, col0:]
    areas_a = table[row0:, :].sum(axis=1)
    areas_b = table[:, col0:].sum(axis=0)
    return {"labels_a": values_a[row0:], "labels_b": values_b[col0:], "areas_a": areas_a,
            "areas_b": areas_b, "overlap": overlap, "n_pixels": int(a.size),
            "union_foreground": int(areas_a.sum() + areas_b.sum() - overlap.sum())}


def _same(new, old):
    assert new.keys() == old.keys()
    for key in old:
        a, b = new[key], old[key]
        if isinstance(b, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape, key
            np.testing.assert_array_equal(a, b)
        elif isinstance(b, float) and np.isnan(b):
            assert np.isnan(a), key
        else:
            assert a == b and type(a) is type(b), key


@pytest.mark.parametrize("index", range(8))
def test_pair_overlaps_mask_agreement_and_split_merge_are_identical(index):
    first, second = _fixture_pairs()[index]
    for new, old in zip(ols._pair_overlaps(first, second), _old_pair_overlaps(first, second)):
        assert new.dtype == old.dtype
        np.testing.assert_array_equal(new, old)
    for drop_edge in (True, False):
        for iou in (0.3, 0.5):
            _same(ols.mask_agreement(first, second, iou=iou, drop_edge=drop_edge),
                  _old_mask_agreement(first, second, iou=iou, drop_edge=drop_edge))
    _same(ols._split_merge(first, second), _old_split_merge(first, second))


@pytest.mark.parametrize("index", range(8))
def test_object_overlap_is_identical_to_the_dense_table(index):
    first, second = _fixture_pairs()[index]
    _same(mc.object_overlap(first, second), _old_object_overlap(first, second))
    _same(mc.object_overlap(second, first), _old_object_overlap(second, first))
    assert mc.match_objects(first, second)["matches"] == mc.match_objects(first, second)["matches"]


def test_object_overlap_handles_a_field_without_background():
    a = np.full((16, 16), 3, np.int32)
    a[:, 7:] = 8
    b = _blobs(2, (16, 16), 2)
    _same(mc.object_overlap(a, b), _old_object_overlap(a, b))


def test_tiling_control_counts_pairs_as_before(monkeypatch):
    rng = np.random.default_rng(4)
    image = rng.random((96, 96)).astype(np.float32)

    def segment(tile):
        from scipy import ndimage

        return ndimage.label(ndimage.uniform_filter(tile, 5) > 0.52)[0]

    new = ols.tiling_control(image, segment, tile_px=64, overlap_px=16)
    monkeypatch.setattr(ols, "_pair_overlaps", _old_pair_overlaps)
    old = ols.tiling_control(image, segment, tile_px=64, overlap_px=16)
    _same(new, old)


# ------------------------------------------------- follow-up decisions -----

def test_leiden_partitions_gives_every_seed_its_partitions_from_a_generator():
    x = _clustered(2, n=60)
    graph = oc.gene_graph(x, "phate", seed=1, knn=8, n_jobs=1)
    resolutions = [0.5, 1.0, 2.0]
    found = oc.leiden_partitions(graph, (r for r in resolutions), [1, 2, 3])
    assert sorted(found) == sorted((s, r) for s in (1, 2, 3) for r in resolutions)
    listed = oc.leiden_partitions(graph, resolutions, [1, 2, 3])
    for key in listed:
        np.testing.assert_array_equal(found[key], listed[key])


def _benchmark(monkeypatch=None, control_n_init=None):
    from sklearn import cluster

    from tests.test_ops_clusters import _screen

    seen = []
    real = cluster.KMeans

    def recording(k, n_init=None, **kwargs):
        seen.append(n_init)
        if control_n_init is not None and len(seen) % 2 == 0:
            n_init = control_n_init
        return real(k, n_init=n_init, **kwargs)

    if monkeypatch is not None:
        monkeypatch.setattr(cluster, "KMeans", recording)
    table, cols, corum, kegg, string, _ = _screen(per=12)
    out = oc.benchmark_clusters(table, cols, corum=corum, kegg=kegg, string=string,
                                resolutions=[1, 2], target=6, baselines=(6, 9), shuffles=2)
    return out["table"], seen


def test_the_shuffled_kmeans_control_is_fitted_with_the_same_n_init(monkeypatch):
    pytest.importorskip("phate")
    pytest.importorskip("leidenalg")
    _, seen = _benchmark(monkeypatch)
    assert seen and len(seen) % 2 == 0
    assert set(seen) == {4}


def test_how_much_the_control_n_init_moves_the_benchmark(monkeypatch):
    pytest.importorskip("phate")
    pytest.importorskip("leidenalg")
    now, _ = _benchmark()
    before, _ = _benchmark(monkeypatch, control_n_init=2)
    rows = now.profile.str.contains("k-means shuffled")
    numeric = now.select_dtypes("number").columns
    shift = (now.loc[rows, numeric] - before.loc[rows, numeric]).abs().max()
    print("k-means shuffled-control shift, n_init 2 -> 4:",
          {k: round(float(v), 4) for k, v in shift.items() if v})
    pd.testing.assert_frame_equal(now.loc[~rows], before.loc[~rows])


def _old_style_stats(values, rows=None, *, zero_sd, chunk=65536):
    x = np.asarray(values) if rows is None else np.asarray(values)[rows]
    mean, sd = x.mean(0), x.std(0)
    if zero_sd == "exact":
        sd[sd == 0] = 1
    return mean, sd


def test_remove_generic_axes_and_fusion_profiles_keep_their_numbers(monkeypatch):
    rng = np.random.default_rng(7)
    genes = [f"g{i}" for i in range(70)] + [f"nontargeting_{i}" for i in range(10)]
    covariates = pd.DataFrame({"gene": genes, "cells_per_guide": rng.integers(20, 400, 80),
                               "shift_G1": rng.normal(size=80), "shift_SG2": rng.normal(size=80),
                               "shift_M": rng.normal(size=80)})
    values = rng.normal(size=(80, 9))
    axes = ["log_cells_per_guide", "shift_G1", "shift_M"]
    blocks = {name: pd.DataFrame({"gene": genes[:60], "n_cells": rng.integers(5, 50, 60),
                                  **{f"PC_{j}": rng.normal(size=60) * (j + 1) for j in range(6)}})
              for name in ("a", "b")}
    blocks["b"]["PC_5"] = 3.0
    new_axes = gmc.remove_generic_axes(values, genes, covariates, axes=axes)
    new_fused = gmc.fusion_profiles(blocks)
    monkeypatch.setattr(ops_tvn, "_column_mean_sd", _old_style_stats)
    old_axes = gmc.remove_generic_axes(values, genes, covariates, axes=axes)
    old_fused = gmc.fusion_profiles(blocks)
    np.testing.assert_array_equal(new_axes[0], old_axes[0])
    assert new_axes[1] == old_axes[1]
    pd.testing.assert_frame_equal(new_fused, old_fused, check_exact=True)


def _old_fit_control_stage(model, nt, nt_batch, *, components, whiten_epsilon, coral_ridge):
    model.control_mean, model.control_scale = nt.mean(0), nt.std(0)
    model.control_scale[model.control_scale == 0] = 1
    if whiten_epsilon > 0:
        model.control_scale = np.sqrt(model.control_scale ** 2 + float(whiten_epsilon))
    nt = (nt - model.control_mean) / model.control_scale
    model.control_pca = ops_tvn._control_rotation(nt, components)
    nt = model.control_pca.transform(nt)
    names = np.unique(nt_batch)
    for name in names:
        rows = nt_batch == name
        mean, spread = nt[rows].mean(0), nt[rows].std(0)
        spread[spread == 0] = 1
        scale = np.sqrt(spread ** 2 + float(whiten_epsilon)) if whiten_epsilon > 0 else spread
        model.batch_stats[str(name)] = (mean, scale)
    whitened = np.vstack([(nt[nt_batch == n] - model.batch_stats[str(n)][0])
                          / model.batch_stats[str(n)][1] for n in names])
    d = whitened.shape[1]
    ridge = float(coral_ridge) * np.eye(d)
    target = ops_tvn._matrix_power(np.cov(whitened, rowvar=False, ddof=1) + ridge, 0.5)
    skipped = []
    for name in names:
        v = (nt[nt_batch == name] - model.batch_stats[str(name)][0]) / model.batch_stats[str(name)][1]
        if len(v) < d:
            model.coral[str(name)] = None
            skipped.append(str(name))
            continue
        source = ops_tvn._matrix_power(np.cov(v, rowvar=False, ddof=1) + ridge, -0.5)
        if np.all(np.isfinite(source)):
            model.coral[str(name)] = source @ target
        else:
            model.coral[str(name)] = None
            skipped.append(str(name))
    return skipped, d


def test_streamed_tvn_control_stage_and_in_memory_screen_within_tolerance(monkeypatch):
    data = _tvn_wells(2)

    def streamed():
        model = ops_tvn.fit_tvn_streamed(list(data), lambda w: data[w], pca_sample=400)
        return np.vstack([model.transform(data[w][0], w, data[w][2]) for w in data])

    x = np.vstack([v[0] for v in data.values()])
    x[:, 4] = 7.25
    wells = np.concatenate([[w] * len(v[0]) for w, v in data.items()])
    control = np.concatenate([v[1] for v in data.values()])

    def in_memory():
        return ops_tvn.tvn_in_memory(x, wells, control, pca_sample=500)

    new_streamed, (new_mem, new_info) = streamed(), in_memory()
    monkeypatch.setattr(ops_tvn, "_column_mean_sd", _old_style_stats)
    monkeypatch.setattr(ops_tvn, "_fit_control_stage", _old_fit_control_stage)
    old_streamed, (old_mem, old_info) = streamed(), in_memory()
    for label, old, new in (("streamed", old_streamed, new_streamed), ("in memory", old_mem, new_mem)):
        assert old.shape == new.shape
        worst_corr, max_abs, relative = _component_agreement(old, new)
        print(f"TVN {label}: min |corr| {worst_corr:.10f}, max abs diff {max_abs:.3g}"
              f" ({relative:.3g} of the largest component SD)")
        assert worst_corr >= 0.999999
        assert relative < 2e-3
    assert new_info["columns_kept"] == old_info["columns_kept"]


def _old_match_fraction(reference, labels, iou=0.5):
    from spacr.seg_qc import _as_labels

    a = _as_labels(reference).ravel().astype(np.int64)
    b = _as_labels(labels).ravel().astype(np.int64)
    area_a = np.bincount(a)
    area_b = np.bincount(b)
    n_ref = int(np.count_nonzero(area_a[1:]))
    if n_ref == 0:
        return 1.0 if not np.count_nonzero(area_b[1:]) else float("nan")
    both = (a > 0) & (b > 0)
    if not both.any():
        return 0.0
    width = int(b.max()) + 1
    pairs, overlap = np.unique(a[both] * width + b[both], return_counts=True)
    ia, ib = pairs // width, pairs % width
    ratio = overlap / (area_a[ia] + area_b[ib] - overlap)
    return float(np.unique(ia[ratio >= iou]).size) / n_ref


def _old_plane_iou(prev_plane, cur_plane):
    prev_ids = np.unique(prev_plane)
    prev_ids = prev_ids[prev_ids > 0]
    cur_ids = np.unique(cur_plane)
    cur_ids = cur_ids[cur_ids > 0]
    if prev_ids.size == 0 or cur_ids.size == 0:
        return prev_ids, cur_ids, np.zeros((prev_ids.size, cur_ids.size), float)
    overlap = np.zeros((prev_ids.size, cur_ids.size), dtype=np.int64)
    both = (prev_plane > 0) & (cur_plane > 0)
    if np.any(both):
        rows = np.searchsorted(prev_ids, prev_plane[both])
        cols = np.searchsorted(cur_ids, cur_plane[both])
        counts = np.bincount(rows * cur_ids.size + cols, minlength=prev_ids.size * cur_ids.size)
        overlap = counts.reshape(prev_ids.size, cur_ids.size)
    prev_counts = np.bincount(prev_plane.ravel())
    cur_counts = np.bincount(cur_plane.ravel())
    prev_area = prev_counts[prev_ids].astype(np.int64)
    cur_area = cur_counts[cur_ids].astype(np.int64)
    union = prev_area[:, None] + cur_area[None, :] - overlap
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, overlap / union, 0.0)
    return prev_ids, cur_ids, iou


@pytest.mark.parametrize("index", range(8))
def test_seg_qc_match_fraction_and_zstack_plane_iou_are_identical(index):
    from spacr import seg_qc, zstack

    first, second = _fixture_pairs()[index]
    for iou in (0.25, 0.5, 0.75):
        new, old = seg_qc._match_fraction(first, second, iou), _old_match_fraction(first, second, iou)
        assert (np.isnan(new) and np.isnan(old)) or new == old
    for a, b in ((first, second), (second, first)):
        for new, old in zip(zstack._plane_iou(a, b), _old_plane_iou(a, b)):
            assert new.dtype == old.dtype and new.shape == old.shape
            np.testing.assert_array_equal(new, old)

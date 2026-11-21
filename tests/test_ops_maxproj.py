"""Max-projection clustering over feature views (item 992, alpha).

The planted toy has a thin group (strong on 4 of 60 columns) and a broad group (weak on 40);
max projection over views must find both where the whole profile alone misses the thin group.
The null toy has no structure, so the max-null p values must be uniform where the naive minimum
over views is not.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr import ops_clusters as oc
from spacr import ops_maxproj as om

pytest.importorskip("igraph")
pytest.importorskip("leidenalg")


@pytest.fixture(autouse=True)
def _one_blas_thread():
    """Small PCAs run far faster on one BLAS thread than on a shared machine's many."""
    threadpoolctl = pytest.importorskip("threadpoolctl")
    with threadpoolctl.threadpool_limits(1):
        yield


def _toy(seed=0, n_genes=200, n_nt=40, guides=3, cols=60, thin=0.8, broad=0.35):
    """Gene and guide tables: genes 0-9 shifted by ``thin`` on columns 0-3, genes 10-29 by ``broad`` on 20-59."""
    rng = np.random.default_rng(seed)
    names = [f"G{i:03d}" for i in range(n_genes)] + [f"nontargeting_{i:03d}" for i in range(n_nt)]
    shift = np.zeros((len(names), cols))
    shift[np.ix_(range(10), range(4))] = thin
    shift[np.ix_(range(10, 30), range(20, cols))] = broad
    rows, genes, cells = [], [], []
    for i, gene in enumerate(names):
        for _ in range(1 if gene.startswith("nontargeting") else guides):
            c = int(rng.integers(40, 120))
            rows.append(shift[i] + rng.normal(0, np.sqrt(40 / c), cols))
            genes.append(gene)
            cells.append(c)
    rows, genes, cells = np.array(rows), np.array(genes), np.array(cells, float)
    columns = [f"f{c:02d}" for c in range(cols)]
    gene_rows = np.array([cells[genes == g] @ rows[genes == g] / cells[genes == g].sum() for g in names])
    gene_table = pd.DataFrame(gene_rows, columns=columns)
    gene_table.insert(0, "n_cells", [cells[genes == g].sum() for g in names])
    gene_table.insert(0, "gene", names)
    guide_table = pd.DataFrame(rows, columns=columns)
    guide_table.insert(0, "n_cells", cells)
    guide_table.insert(0, "gene", genes)
    guide_table.insert(0, "guide", [f"{g}_{i}" for i, g in enumerate(genes)])
    return gene_table, guide_table


def _views(path, spans, cols=60):
    """A views file of ``(name, block, start, stop)`` column spans."""
    columns = [f"f{c:02d}" for c in range(cols)]
    Path(path).write_text(json.dumps([{"name": n, "block": b, "columns": columns[a:z]}
                                      for n, b, a, z in spans]), encoding="utf-8")
    return str(path)


PLANTED_VIEWS = [("all", "W", 0, 60), ("t", "P2", 0, 5), ("n1", "P2", 5, 20),
                 ("b", "P1", 20, 60)]


def _settings(views, **extra):
    """Fast phase settings."""
    return {"ops_maxproj_enabled": True, "ops_maxproj_views": views,
            "ops_maxproj_calibration_permutations": 10, "ops_maxproj_null_permutations": 20,
            "ops_maxproj_k": 12, **extra}


def _together(labels, members):
    """The share of member pairs that share a cluster."""
    lab = labels[members]
    return float(np.mean([lab[i] == lab[j] for i in range(len(lab)) for j in range(i + 1, len(lab))]))


def _arrays(gene, guide, spans):
    """maxproj_scores inputs for column spans."""
    columns = [c for c in gene.columns if c.startswith("f")]
    return (gene[columns].to_numpy(float), gene.gene.to_numpy(), gene.n_cells.to_numpy(float),
            guide[columns].to_numpy(float), guide.gene.to_numpy(), guide.n_cells.to_numpy(float),
            [np.arange(a, z) for a, z in spans])


# 1 (with 'shrunk'; with the default 'cosine' the max projection does not beat the whole profile on
# this toy, the amendment proposed for the maintainer)
def test_max_projection_finds_the_thin_and_the_broad_group_the_whole_profile_misses(tmp_path):
    gene, guide = _toy(thin=0.7)
    found = {}
    for tag, spans in (("views", PLANTED_VIEWS), ("whole", PLANTED_VIEWS[:1])):
        views = _views(tmp_path / f"{tag}.json", spans)
        out = om.cluster_maxproj_phase(_settings(views, ops_maxproj_similarity="shrunk"), gene, guide,
                                       str(tmp_path / tag))
        part = pd.read_csv(out["outputs"]["maxproj_partition"])
        found[tag] = part
    views, whole = found["views"], found["whole"]
    assert _together(views.cluster.to_numpy(), range(10)) >= 0.7
    assert _together(views.cluster.to_numpy(), range(10, 30)) >= 0.8
    assert _together(whole.cluster.to_numpy(), range(10)) <= 0.4
    thin_cluster = np.bincount(views.cluster.to_numpy()[:10]).argmax()
    assert views.label[views.cluster == thin_cluster].iloc[0] == "thin"


# 2
def test_the_max_null_is_calibrated_and_the_naive_minimum_is_not():
    from scipy.stats import norm

    kept, naive = [], []
    for seed in range(20):
        gene, guide = _toy(seed=seed, n_genes=60, n_nt=30, cols=20, thin=0.0, broad=0.0)
        out = om.maxproj_scores(*_arrays(gene, guide, [(0, 10), (10, 20)]), calibration=10, null=20,
                                seed=seed)
        iu = np.triu_indices(len(gene), 1)
        kept.append(np.mean(out["p"][iu] < 0.05))
        naive.append(np.mean(np.minimum(*[norm.sf(s[iu]) for s in out["scores"]]) < 0.05))
    assert 0.03 <= np.mean(kept) <= 0.07
    assert np.mean(naive) > 0.07


# 3
def test_one_view_is_the_whole_profile_calibrated_graph():
    gene, guide = _toy(n_genes=80, n_nt=40)
    rows, names, n_cells, g_rows, g_genes, g_cells, views = _arrays(gene, guide, [(0, 60)])
    out = om.maxproj_scores(rows, names, n_cells, g_rows, g_genes, g_cells, views, calibration=5, null=10)
    control = oc._is_control_row(names, "nontargeting")
    direct = om.calibrate_view(om.view_similarity(rows, control, 0.99, om.DEFAULT_SEED),
                               out["calibration_counts"][0], out["pair_bins"])
    assert np.array_equal(out["projected"], direct) and not out["view"].any()
    graph = om.projected_graph(om.max_null_pvalues(direct, out["null_counts"], out["pair_bins"]), 10, 0.05)
    assert graph.get_edgelist() == om.projected_graph(out["p"], 10, 0.05).get_edgelist()


# 4
def test_the_view_order_does_not_change_the_projection():
    gene, guide = _toy(n_genes=80, n_nt=40)
    spans = [(0, 5), (5, 20), (20, 60)]
    first = om.maxproj_scores(*_arrays(gene, guide, spans), calibration=5, null=10)
    second = om.maxproj_scores(*_arrays(gene, guide, spans[::-1]), calibration=5, null=10)
    assert np.array_equal(first["projected"], second["projected"])
    assert np.array_equal(first["p"], second["p"])


def test_ties_go_to_the_lower_view():
    a = np.array([[0.0, 2.0], [2.0, 0.0]])
    projected, view = om.max_projection([a, a.copy()])
    assert np.array_equal(projected, a) and not view.any()


def _files(folder):
    """Every output file's bytes, the summary without its timing."""
    out = {}
    for path in sorted(Path(folder).iterdir()):
        data = path.read_bytes()
        if path.name == "maxproj_summary.json":
            record = json.loads(data)
            record.pop("seconds")
            data = json.dumps(record, sort_keys=True).encode()
        out[path.name] = gzip.decompress(data) if path.suffix == ".gz" else data
    return out


# 5
def test_the_worker_count_does_not_change_the_outputs(tmp_path):
    gene, guide = _toy(n_genes=80, n_nt=40)
    views = _views(tmp_path / "v.json", PLANTED_VIEWS)
    for workers in (1, 3):
        om.cluster_maxproj_phase(_settings(views, ops_maxproj_workers=workers, ops_maxproj_k=6), gene, guide,
                                 str(tmp_path / f"w{workers}"))
    one, three = (_files(tmp_path / f"w{w}" / "results" / "ops_maxproj") for w in (1, 3))
    summary_one, summary_three = (json.loads(f.pop("maxproj_summary.json")) for f in (one, three))
    assert one == three
    assert {k: v for k, v in summary_one.items() if k != "settings"} == {
        k: v for k, v in summary_three.items() if k != "settings"}


# 6
def test_the_saved_settings_rerun_byte_identically(tmp_path):
    gene, guide = _toy(n_genes=80, n_nt=40)
    views = _views(tmp_path / "v.json", PLANTED_VIEWS)
    first = om.cluster_maxproj_phase(_settings(views, ops_maxproj_k=6), gene, guide, str(tmp_path / "a"))
    saved = json.loads(Path(first["outputs"]["maxproj_summary"]).read_text())["settings"]
    om.cluster_maxproj_phase(saved, gene, guide, str(tmp_path / "b"))
    assert _files(tmp_path / "a" / "results" / "ops_maxproj") == _files(tmp_path / "b" / "results" / "ops_maxproj")
    names = set(_files(tmp_path / "a" / "results" / "ops_maxproj"))
    assert names == {"views.csv", "calibration_bins.parquet", "pairs_projected.parquet", "per_view_scores.npz",
                     "leiden_grid.csv", "maxproj_partition.csv", "cluster_views.csv", "maxproj_summary.json",
                     "maxproj.png"}


# 7
def _bootstrap_before(rows, names, guide_rows, guide_genes, guide_cells, seed, prefix="nontargeting"):
    """guide_bootstrap_profiles as it was before the _guide_groups / _guide_shift split."""
    names = np.asarray(names).astype(str)
    guide_rows = np.asarray(guide_rows, float)
    cells = np.asarray(guide_cells, float)
    members = {}
    for i, gene in enumerate(np.asarray(guide_genes).astype(str)):
        members.setdefault(gene, []).append(i)
    control = oc._is_control_row(names, prefix)
    empty = np.array([], int)
    groups = [np.array(members[g], int) if g in members and not control[i] else empty
              for i, g in enumerate(names)]
    rng = np.random.default_rng(seed)
    drawn = [rng.choice(ix, len(ix), replace=True) if len(ix) else ix for ix in groups]
    shift = np.zeros((len(names), guide_rows.shape[1]))
    for i, (ix, bx) in enumerate(zip(groups, drawn)):
        if len(ix):
            shift[i] = ((cells[bx] @ guide_rows[bx]) / cells[bx].sum()
                        - (cells[ix] @ guide_rows[ix]) / cells[ix].sum())
    return np.asarray(rows, float) + shift


def _knn_edges_before(rows, k):
    """gene_graph(kind='knn')'s edges and weights as they were before _knn_graph."""
    unit = rows / np.clip(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12, None)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = np.argpartition(-similarity, k, axis=1)[:, :k]
    edges = {}
    for i in range(len(unit)):
        for j in nearest[i]:
            a, b = (i, int(j)) if i < j else (int(j), i)
            edges[(a, b)] = max(float(similarity[a, b]), 1e-6)
    return list(edges), list(edges.values())


def test_the_refactored_bootstrap_and_knn_graph_are_bit_identical():
    gene, guide = _toy(n_genes=80, n_nt=40)
    rows, names, _n, g_rows, g_genes, g_cells, _v = _arrays(gene, guide, [(0, 60)])
    for seed in (0, 7, 8000):
        assert np.array_equal(oc.guide_bootstrap_profiles(rows, names, g_rows, g_genes, g_cells, seed),
                              _bootstrap_before(rows, names, g_rows, g_genes, g_cells, seed))
    for k in (4, 10):
        graph = oc.gene_graph(rows, "knn", k=k)
        edges, weights = _knn_edges_before(rows, k)
        assert graph.get_edgelist() == edges and graph.es["weight"] == weights


def test_the_consensus_hook_builds_each_bootstrap_graph():
    gene, guide = _toy(n_genes=40, n_nt=30)
    rows = gene[[c for c in gene.columns if c.startswith("f")]].to_numpy(float)
    seen = []

    def graph_of(boot, seed):
        """Record the seed and build the k-NN graph."""
        seen.append(seed)
        return oc.gene_graph(boot, "knn", k=5)

    out = oc.guide_bootstrap_consensus([rows, rows + 0.0], 2, 4, seed=3, graph_of=graph_of, match=False)
    assert seen == [3 * oc.CONSENSUS_SEED_STRIDE, 3 * oc.CONSENSUS_SEED_STRIDE + 1]
    assert len(out["labels"]) == len(rows)


def test_projected_consensus_writes_a_partition(tmp_path):
    gene, guide = _toy(n_genes=60, n_nt=40)
    views = _views(tmp_path / "v.json", PLANTED_VIEWS)
    out = om.cluster_maxproj_phase(_settings(views, ops_maxproj_consensus=True, ops_consensus_bootstraps=2,
                                             ops_maxproj_k=5, ops_maxproj_calibration_permutations=4,
                                             ops_maxproj_null_permutations=6),
                                   gene, guide, str(tmp_path))
    part = pd.read_csv(out["outputs"]["maxproj_partition"])
    assert part.cluster.notna().all() and len(part) == len(gene)


# 8
def test_the_default_settings_leave_the_sweep_outputs_identical(monkeypatch, tmp_path):
    from spacr.ops_engine import run_ops
    from spacr.ops_settings import OPS_DEFAULTS
    from spacr.settings import ALPHA_FEATURES
    from tests.test_ops_consensus_bench_928_929 import _phase_settings
    from tests.test_ops_leiden_sweep_916 import _install

    _install(monkeypatch)
    keys = ALPHA_FEATURES[992]["settings"]
    assert len(keys) == 19 and OPS_DEFAULTS["ops_maxproj_enabled"] is False
    base = _phase_settings(tmp_path, ops_cluster_knn=4, ops_cluster_seeds="42",
                           ops_cluster_sweep_resolutions={"knn": [1], "phate": [3]}, ops_cluster_target_clusters=4)
    for key in ("ops_consensus_bootstraps", "ops_consensus_k", "ops_consensus_guides"):
        base.pop(key)
    outputs = {}
    for tag, extra in (("legacy", {}), ("defaults", {k: OPS_DEFAULTS[k] for k in keys})):
        settings = {**base, **extra, "dst_root": str(tmp_path / tag)}
        run_ops(settings, phases=["cluster_sweep"])
        folder = tmp_path / tag / "results" / "ops_clusters"
        outputs[tag] = {p.name: gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
                        for p in sorted(folder.iterdir()) if p.suffix != ".png"}
        assert not (tmp_path / tag / "results" / "ops_maxproj").exists()
    assert outputs["legacy"] == outputs["defaults"]


def test_the_views_file_is_checked(tmp_path):
    columns = ["a", "b"]
    for body, match in (([], "no view"), ([{"name": "x", "columns": []}], "no column"),
                        ([{"name": "x", "columns": ["a"]}, {"name": "x", "columns": ["b"]}], "two views"),
                        ([{"name": "x", "columns": ["c"]}], "lacks")):
        path = tmp_path / "v.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            om.read_views(str(path), columns)
    path.write_text(json.dumps([{"name": "x", "columns": ["b", "a"]}]), encoding="utf-8")
    views, md5 = om.read_views(str(path), columns)
    assert views == [{"name": "x", "block": "x", "columns": ["b", "a"]}] and len(md5) == 32


def test_bad_settings_are_refused(tmp_path):
    gene, guide = _toy(n_genes=20, n_nt=10)
    views = _views(tmp_path / "v.json", PLANTED_VIEWS)
    for extra, match in (({"ops_maxproj_views": None}, "ops_maxproj_views"),
                         ({"ops_maxproj_graph": "tree"}, "ops_maxproj_graph"),
                         ({"ops_maxproj_similarity": "dot"}, "ops_maxproj_similarity"),
                         ({"ops_maxproj_edge_p": 0}, "ops_maxproj_edge_p")):
        with pytest.raises(ValueError, match=match):
            om.cluster_maxproj_phase(_settings(views, **extra), gene, guide, str(tmp_path))
    with pytest.raises(ValueError, match="gene and guide rows"):
        om.cluster_maxproj_phase(_settings(views), None, None, str(tmp_path))


def test_the_new_settings_are_alpha_with_short_tooltips():
    from spacr.ops_settings import OPS_CATEGORIES, OPS_DEFAULTS, OPS_TOOLTIPS
    from spacr.settings import (ALPHA_FEATURES, DEFAULT_CHANGES, SETTINGS_FORMAT_VERSIONS,
                                SETTINGS_SCHEMA_VERSION)

    keys = ALPHA_FEATURES[992]["settings"]
    assert set(keys) == {k for k in OPS_DEFAULTS if k.startswith("ops_maxproj_")}
    assert set(keys) == set(OPS_CATEGORIES["Max-projection clustering"])
    for key in keys:
        assert len(OPS_TOOLTIPS[key]) <= 600 and "Alpha" in OPS_TOOLTIPS[key]
    assert OPS_DEFAULTS["ops_maxproj_seed"] == om.DEFAULT_SEED and OPS_DEFAULTS["ops_maxproj_similarity"] == "cosine"
    assert SETTINGS_SCHEMA_VERSION >= 951 and 951 in SETTINGS_FORMAT_VERSIONS and 951 not in DEFAULT_CHANGES


def test_the_cluster_labels_follow_the_card_rule():
    import igraph

    views = [{"name": "W", "block": "W", "columns": list("abcdefgh")},
             {"name": "p1", "block": "P1", "columns": list("a")},
             {"name": "t", "block": "P2", "columns": list("a")},
             {"name": "q", "block": "P2", "columns": list("ab")}]
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    edges = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (6, 7), (6, 8), (7, 8)]
    graph = igraph.Graph(n=9, edges=edges)
    view = np.zeros((9, 9), int)
    for (a, b), v in zip(edges, [0, 0, 2, 1, 1, 2, 2, 3, 0]):
        view[a, b] = view[b, a] = v
    scores = [np.full((9, 9), float(v)) for v in range(4)]
    null = np.zeros((1, len(om.SCORE_GRID) - 1), np.int64)
    null[0, 100] = 1
    shares, kinds = om.label_clusters_by_view(labels, graph, view, views, scores, null, 8)
    assert list(kinds.label) == ["broad", "thin", "thin"]
    assert list(kinds.views) == ["W;t", "p1;t", "W;t;q"]
    assert shares.loc[(shares.cluster == 0) & (shares.view == "t"), "mean_score"].item() == 2.0

"""PHATE + Leiden clustering of the OPS benchmark, with stand-in modules.

PHATE, igraph and leidenalg are the optional ``clustering`` extra and are not
installed on CI, so ``test_ops_clusters.py``'s end-to-end tests skip there.
These tests put small, deterministic stand-ins for the three modules in
``sys.modules`` and check what spaCR's own code does around them: the graph
handed to Leiden is PHATE's diffusion operator made symmetric, one edge per
upper-triangle entry carrying its weight; every resolution is passed with the
seed; the benchmark scores real clusters, the shuffled-PC control, the k-means
baseline and the reference; consensus runs cycle the resolutions; and the
aggregate phase writes every table it names in its summary.
"""
from __future__ import annotations

import json
import sys
import types

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spacr import ops_clusters as oc
from tests.test_ops_clusters import _screen

#: Every call the stand-in Leiden received, for the assertions below.
CALLS: list = []


def _diffusion(x: np.ndarray, knn: int) -> np.ndarray:
    """A row-stochastic kNN Gaussian affinity, PHATE's diffusion operator in
    miniature: asymmetric, so the symmetrising step has work to do."""
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    k = min(knn, len(x) - 1)
    sigma = np.sort(d, axis=1)[:, k][:, None] + 1e-9
    w = np.exp(-(d / sigma) ** 2)
    far = d > np.sort(d, axis=1)[:, k][:, None]
    w[far] = 0.0
    np.fill_diagonal(w, 0.0)
    return w / w.sum(axis=1, keepdims=True)


def _install(monkeypatch, *, as_sparse: bool):
    """Put stand-in phate, igraph and leidenalg modules in sys.modules."""
    CALLS.clear()

    class PHATE:
        def __init__(self, **options):
            self.options = options

        def fit(self, x):
            op = _diffusion(np.asarray(x, float), self.options["knn"])
            self.graph = types.SimpleNamespace(
                diff_op=sparse.csr_matrix(op) if as_sparse else op)
            return self

    class Graph:
        def __init__(self, n, edges, directed):
            assert directed is False
            self.n, self.edges = n, list(edges)
            self.es = {}

    class Partition:
        pass

    def find_partition(graph, kind, *, weights, n_iterations, seed, resolution_parameter):
        from sklearn.cluster import AgglomerativeClustering

        CALLS.append({"kind": kind, "weights": weights, "n_iterations": n_iterations,
                      "seed": seed, "resolution": resolution_parameter, "graph": graph})
        w = np.zeros((graph.n, graph.n))
        for (i, j), value in zip(graph.edges, graph.es[weights]):
            w[i, j] = w[j, i] = value
        k = max(2, min(int(resolution_parameter) * 3, graph.n - 1))
        labels = AgglomerativeClustering(k, linkage="average", metric="precomputed").fit_predict(
            1.0 - w / w.max())
        return types.SimpleNamespace(membership=labels.tolist())

    monkeypatch.setitem(sys.modules, "phate", types.SimpleNamespace(PHATE=PHATE))
    monkeypatch.setitem(sys.modules, "igraph", types.SimpleNamespace(Graph=Graph))
    monkeypatch.setitem(sys.modules, "leidenalg", types.SimpleNamespace(
        find_partition=find_partition, RBConfigurationVertexPartition=Partition))
    return Partition


@pytest.mark.parametrize("as_sparse", [False, True])
def test_leiden_gets_the_symmetrised_diffusion_graph_and_every_resolution(monkeypatch, as_sparse):
    partition = _install(monkeypatch, as_sparse=as_sparse)
    table, cols, *_rest, modules = _screen(per=10)
    pcs, _n, _f = oc.embed_profiles(table, cols)
    out = oc.phate_leiden(pcs, [1, 2], knn=6, seed=7)

    assert sorted(out) == [1, 2]
    assert [c["resolution"] for c in CALLS] == [1.0, 2.0]
    assert {c["seed"] for c in CALLS} == {7}
    assert {c["n_iterations"] for c in CALLS} == {-1}
    assert all(c["kind"] is partition and c["weights"] == "weight" for c in CALLS)

    graph = CALLS[0]["graph"]
    op = _diffusion(pcs, 6)
    sym = (op + op.T) / 2
    assert graph.n == len(pcs)
    assert all(i < j for i, j in graph.edges)
    rows, columns = np.nonzero(np.triu(sym, 1))
    assert sorted(graph.edges) == sorted(zip(rows.tolist(), columns.tolist()))
    for (i, j), value in zip(graph.edges, graph.es["weight"]):
        assert value == pytest.approx(sym[i, j])

    from sklearn.metrics import adjusted_rand_score
    keep = modules >= 0
    assert adjusted_rand_score(modules[keep], out[2][keep]) > 0.8
    assert len(out[1]) == len(pcs)


def test_the_benchmark_scores_every_arm_and_its_controls(monkeypatch):
    _install(monkeypatch, as_sparse=True)
    table, cols, corum, kegg, string, modules = _screen(per=10)
    ref = pd.DataFrame({"gene": table.gene[modules >= 0], "cluster": modules[modules >= 0]})
    out = oc.benchmark_clusters(table, cols, corum=corum, kegg=kegg, string=string,
                                resolutions=[1, 2], target=6, reference=ref,
                                baselines=(4, 10 ** 6), shuffles=2, label="demo")
    t = out["table"]
    assert set(t.arm) == {"demo"}
    leiden = t[t.method == "PHATE+Leiden"]
    assert list(leiden.resolution) == [1, 1, 2, 2]
    assert leiden.profile.str.contains("shuffled").tolist() == [False, True, False, True]
    summary = out["summary"]
    headline = summary["headline_resolution"]
    assert headline == 2 and summary["cluster_counts"][2] == 6
    # The headline count and the extra baseline; a count >= the rows is skipped.
    assert sorted(set(t.method) - {"PHATE+Leiden", "reference"}) == ["k-means k=4",
                                                                       "k-means k=6"]
    assert (t.method == "reference").sum() == 1
    real = leiden[~leiden.profile.str.contains("shuffled")]
    shuffled = leiden[leiden.profile.str.contains("shuffled")]
    assert real.STRING_precision.max() > shuffled.STRING_precision.max()
    assert summary["reference_ARI"] > summary["reference_ARI_shuffled_p95"]
    assert summary["string_AUPRC_cosine"] > summary["string_AUPRC_shuffled_mean"]
    assert summary["rows_clustered"] == len(table) and summary["genes"] == int((modules >= 0).sum())
    assert list(out["assignments"].columns) == ["gene", "n_cells", "leiden_r1", "leiden_r2"]


def test_no_shuffles_leaves_the_shuffled_auprc_empty(monkeypatch):
    _install(monkeypatch, as_sparse=False)
    table, cols, corum, kegg, string, _modules = _screen(per=6)
    out = oc.benchmark_clusters(table, cols, corum=corum, kegg=kegg, string=string,
                                resolutions=[1], target=3, shuffles=0)
    assert out["summary"]["string_AUPRC_shuffled_mean"] is None
    assert "reference_ARI" not in out["summary"]
    assert (out["table"].method == "reference").sum() == 0


def _guides(table, cols, seed=0):
    """Three guides per gene around its profile, controls named nontargeting."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in table.iterrows():
        control = r.gene.startswith("nontargeting")
        for k in range(3):
            rows.append({"guide": f"{r.gene}_{k}", "gene": "nontargeting" if control else r.gene,
                         "n_cells": 30, **{c: r[c] + rng.normal(0, 0.05) for c in cols}})
    guides = pd.DataFrame(rows)
    return guides, guides.groupby("gene", as_index=False).n_cells.sum()


def test_consensus_runs_cycle_resolutions_and_leave_out_a_share(monkeypatch):
    _install(monkeypatch, as_sparse=True)
    table, cols, *_rest, _modules = _screen(per=8)
    guides, genes = _guides(table, cols)
    out = oc.consensus_clustering(genes, guides, cols, runs=3, resolutions=[2, 3],
                                  subsample=0.75)
    assert [r["resolution"] for r in out["runs"]] == [2, 3, 2]
    assert [r["run"] for r in out["runs"]] == [0, 1, 2]
    labels = out["labels"]
    assert labels.shape == (3, len(out["genes"]))
    n = labels.shape[1]
    assert ((labels == -1).sum(axis=1) == n - int(0.75 * n)).all()
    # Each run draws its own seed, so the stand-in Leiden saw three seeds.
    assert len({c["seed"] for c in CALLS}) == 3
    p = oc.co_membership(labels)
    names = list(out["genes"])
    within = [p[names.index("M0G0"), names.index(f"M0G{k}")] for k in range(1, 8)]
    across = [p[names.index("M0G0"), names.index(f"M3G{k}")] for k in range(8)]
    assert np.nanmean(within) > 0.7 > np.nanmean(across)


def test_zero_consensus_runs_give_an_empty_label_matrix(monkeypatch):
    _install(monkeypatch, as_sparse=True)
    table, cols, *_rest = _screen(per=4)
    guides, genes = _guides(table, cols)
    out = oc.consensus_clustering(genes, guides, cols, runs=0)
    assert out["labels"].shape == (0, len(out["genes"])) and out["runs"] == []


def _benchmark_files(tmp_path, corum, kegg, string):
    """The three benchmark tables as Brieflow-style TSVs."""
    paths = {}
    for name, frame in (("corum", corum), ("kegg", kegg), ("string", string)):
        paths[name] = tmp_path / f"{name}.tsv"
        frame.to_csv(paths[name], sep="\t", index=False)
    return paths


@pytest.mark.parametrize("with_reference", [False, True])
def test_the_aggregate_benchmark_writes_every_table_it_reports(monkeypatch, tmp_path,
                                                               with_reference):
    _install(monkeypatch, as_sparse=True)
    table, cols, corum, kegg, string, modules = _screen(per=8)
    guides, genes = _guides(table, cols)
    genes = guides.groupby("gene", as_index=False).agg(
        n_cells=("n_cells", "sum"), **{c: (c, "median") for c in cols})
    paths = _benchmark_files(tmp_path, corum, kegg, string)
    settings = {f"ops_benchmark_{k}": str(v) for k, v in paths.items()}
    settings.update(ops_cluster_resolutions="1,2", ops_cluster_target=6, ops_cluster_seed=0,
                    ops_consensus_runs=2)
    if with_reference:
        ref = tmp_path / "reference.csv"
        pd.DataFrame({"gene_symbol": table.gene[modules >= 0],
                      "interphase_cluster": modules[modules >= 0]}).to_csv(ref, index=False)
        settings["ops_reference_clusters"] = str(ref)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    summary = oc._aggregate_benchmark(guides, genes, cols, settings, str(out_dir), "P1_tvn")

    outputs = summary["outputs"]
    assert set(outputs) == {"cluster_benchmark", "cluster_assignments", "string_pr_curve",
                            "consensus_partitions"}
    bench = pd.read_csv(outputs["cluster_benchmark"])
    assert outputs["cluster_benchmark"].endswith("ops_cluster_benchmark_P1_tvn.csv")
    # Consensus runs cycle their own default sweep, not the benchmark's.
    assert summary["seed"] == 0
    assert [(r["run"], r["resolution"]) for r in summary["consensus_runs"]] == [(0, 3), (1, 5)]
    with np.load(outputs["consensus_partitions"]) as saved:
        assert saved["labels"].shape[0] == 2
        assert len(saved["genes"]) == saved["labels"].shape[1]
    written = json.loads((out_dir / "ops_cluster_summary_P1_tvn.json").read_text())
    assert written["benchmark_files"] == {k: str(v) for k, v in paths.items()}
    if with_reference:
        # The reference scored on the clustered genes, and on all of its own.
        assert (bench.method == "reference").sum() == 2
        assert bench.profile.iloc[-1] == "reference clusters (all reference genes)"
        assert pd.isna(bench.resolution.iloc[-1])
        assert written["reference_clusters"] == settings["ops_reference_clusters"]
    else:
        assert (bench.method == "reference").sum() == 0
        assert written["reference_clusters"] is None


def test_the_aggregate_benchmark_names_a_missing_table(tmp_path):
    table, cols, corum, kegg, string, _modules = _screen(per=4)
    paths = _benchmark_files(tmp_path, corum, kegg, string)
    settings = {"ops_benchmark_corum": str(paths["corum"]),
                "ops_benchmark_kegg": str(tmp_path / "absent.tsv"),
                "ops_benchmark_string": str(paths["string"])}
    with pytest.raises(ValueError, match="ops_benchmark_kegg"):
        oc._aggregate_benchmark(table, table, cols, settings, str(tmp_path), "P1")


def test_a_reference_column_the_caller_names_must_exist(tmp_path):
    path = tmp_path / "ref.tsv"
    pd.DataFrame({"gene": ["A", "B"], "cluster": [1, 2]}).to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="no gene column 'symbol'"):
        oc.reference_clusters(str(path), gene="symbol")
    out = oc.reference_clusters(str(path), gene="gene", cluster="cluster")
    assert out.to_dict("list") == {"gene": ["A", "B"], "cluster": [1, 2]}

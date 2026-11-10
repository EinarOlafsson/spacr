"""Leiden resolution sweep of OPS gene profiles (item 916, alpha).

PHATE, igraph and leidenalg are the optional ``clustering`` extra and are not
installed on CI, so most tests put small stand-ins in ``sys.modules``: an
igraph that keeps its edges and a Leiden that returns the graph's connected
components. The planted profiles are blocks of near-identical rows, so the
right answer is known: every block is one cluster. One test runs the real
extra when it is installed.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spacr import ops_clusters as oc
from spacr.ops_engine import run_ops

CALLS: list = []


def _blocks(n_blocks=6, size=15, dims=12, noise=0.01, seed=0):
    """Planted gene profiles: ``n_blocks`` tight groups along separate axes."""
    rng = np.random.default_rng(seed)
    rows, names, block = [], [], []
    for b in range(n_blocks):
        centre = np.zeros(dims)
        centre[b % dims] = 1.0
        centre[(b + 1) % dims] = 0.3 * (b // dims + 1)
        for i in range(size):
            rows.append(centre + noise * rng.standard_normal(dims))
            names.append(f"G{b}_{i}")
            block.append(b)
    return np.array(names), np.vstack(rows), np.array(block)


def _components(n, edges, weights, floor):
    """Connected components over the edges whose weight exceeds ``floor``."""
    parent = list(range(n))

    def root(i):
        """The component root of ``i``."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (a, b), w in zip(edges, weights):
        if w > floor:
            parent[root(a)] = root(b)
    roots = [root(i) for i in range(n)]
    order = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return [order[r] for r in roots]


def _install(monkeypatch, *, as_sparse=True):
    """Stand-in phate, igraph and leidenalg modules."""
    CALLS.clear()

    class Graph:
        def __init__(self, n, edges, directed):
            assert directed is False
            self.n, self.edges = n, list(edges)
            self.es = {}

    class Partition:
        pass

    class PHATE:
        def __init__(self, **options):
            self.options = options

        def fit(self, x):
            unit = x / np.linalg.norm(x, axis=1, keepdims=True)
            sim = np.clip(unit @ unit.T, 0, None)
            np.fill_diagonal(sim, 0)
            sim[sim < 0.9] = 0
            op = sim / sim.sum(1, keepdims=True)
            self.graph = types.SimpleNamespace(diff_op=sparse.csr_matrix(op) if as_sparse else op)
            return self

    def find_partition(graph, kind, *, weights, n_iterations, seed, resolution_parameter):
        CALLS.append({"seed": seed, "resolution": resolution_parameter, "kind": kind,
                      "n_iterations": n_iterations})
        return types.SimpleNamespace(membership=_components(graph.n, graph.edges, graph.es[weights], 0.0))

    monkeypatch.setitem(sys.modules, "phate", types.SimpleNamespace(PHATE=PHATE))
    monkeypatch.setitem(sys.modules, "igraph", types.SimpleNamespace(Graph=Graph))
    monkeypatch.setitem(sys.modules, "leidenalg", types.SimpleNamespace(
        find_partition=find_partition, RBConfigurationVertexPartition=Partition))
    return Partition


def _references(folder, names, block):
    """Brieflow-format CORUM, KEGG and STRING tables whose groups are the blocks."""
    folder.mkdir(parents=True, exist_ok=True)
    groups = pd.DataFrame({"gene_name": names, "group": [f"complex{b}" for b in block]})
    groups.to_csv(folder / "corum_group_benchmark.tsv", sep="\t", index=False)
    groups.assign(group=[f"pathway{b // 2}" for b in block]).to_csv(
        folder / "kegg_group_benchmark.tsv", sep="\t", index=False)
    pairs = []
    for b in np.unique(block):
        members = names[block == b]
        for i in range(0, len(members) - 1, 2):
            pairs += [{"gene_name": members[i], "pair": f"p{b}_{i}"},
                      {"gene_name": members[i + 1], "pair": f"p{b}_{i}"}]
    pd.DataFrame(pairs).to_csv(folder / "string_pair_benchmark.tsv", sep="\t", index=False)
    return folder


def test_the_knn_graph_joins_only_rows_of_one_block(monkeypatch):
    _install(monkeypatch)
    names, rows, block = _blocks()
    graph = oc.gene_graph(rows, "knn", k=4)
    assert graph.n == len(rows)
    assert all(block[a] == block[b] and a < b for a, b in graph.edges)
    assert len(set(graph.edges)) == len(graph.edges)
    assert min(graph.es["weight"]) >= 1e-6 and max(graph.es["weight"]) <= 1.0 + 1e-9


@pytest.mark.parametrize("as_sparse", [True, False])
def test_the_phate_graph_is_the_symmetrised_upper_triangle(monkeypatch, as_sparse):
    _install(monkeypatch, as_sparse=as_sparse)
    _names, rows, block = _blocks(n_blocks=3, size=6)
    graph = oc.gene_graph(rows, "phate", seed=7)
    assert graph.edges and all(a < b and block[a] == block[b] for a, b in graph.edges)
    assert all(w > 0 for w in graph.es["weight"])


def test_leiden_runs_every_seed_and_resolution_in_order(monkeypatch):
    partition = _install(monkeypatch)
    _names, rows, block = _blocks(n_blocks=4, size=8)
    graph = oc.gene_graph(rows, "knn", k=3)
    found = oc.leiden_partitions(graph, [0.5, 2], [42, 1])
    assert list(found) == [(42, 0.5), (42, 2), (1, 0.5), (1, 2)]
    assert [(c["seed"], c["resolution"]) for c in CALLS] == [(42, 0.5), (42, 2.0), (1, 0.5), (1, 2.0)]
    assert all(c["kind"] is partition and c["n_iterations"] == -1 for c in CALLS)
    labels = found[(42, 0.5)]
    assert len(np.unique(labels)) == 4
    assert all(len(np.unique(labels[block == b])) == 1 for b in range(4))


def test_an_unknown_graph_is_refused_before_any_import():
    with pytest.raises(ValueError, match="'phate' or 'knn'"):
        oc.gene_graph(np.eye(3), "umap")


def test_a_missing_extra_names_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "igraph", None)
    with pytest.raises(ImportError, match=r"spacr\[clustering\]"):
        oc.gene_graph(np.eye(4), "knn", k=2)
    monkeypatch.setitem(sys.modules, "leidenalg", None)
    with pytest.raises(ImportError, match="leidenalg"):
        oc.leiden_partitions(object(), [1], [1])


def test_complex_metrics_count_recovered_report_complexes():
    genes = ["a", "b", "c", "d", "e", "f", "g"]
    labels = np.array([0, 0, 1, 2, 2, 2, 3])
    groups = {"A": ["a", "b", "c", "x"], "B": ["d", "e", "f"], "C": ["g", "a"], "D": ["c", "y"]}
    out = oc.complex_metrics(genes, labels, groups, {"A": 1, "B": 1, "C": 0, "D": 1})
    assert out["REPORT_complexes"] == 2
    assert out["complexes_recovered_pct"] == 100.0
    assert out["CORUM_partner_in_cluster_pct"] == pytest.approx(100 * 5 / 7)
    out = oc.complex_metrics(genes, np.arange(7), groups, {"A": 1, "B": 1})
    assert out["complexes_recovered_pct"] == 0.0 and out["CORUM_partner_in_cluster_pct"] == 0.0
    empty = oc.complex_metrics(["a"], np.array([0]), {"A": ["a", "z"]}, {"A": 1})
    assert empty["REPORT_complexes"] == 0 and np.isnan(empty["complexes_recovered_pct"])
    assert np.isnan(empty["CORUM_partner_in_cluster_pct"])


def test_planted_blocks_score_high_and_their_shuffles_near_zero(tmp_path):
    names, _rows, block = _blocks(n_blocks=8, size=12)
    refs = oc.sweep_references(str(_references(tmp_path / "ref", names, block)))
    assert set(refs["halves"].values()) <= {0, 1} and len(refs["groups"]) == 8
    lines = []
    table = oc.score_partitions(names, {("knn12", 42, 1.0): block, ("knn12", 42, 2.0): block}, refs,
                                shuffles=3, report=lines.append)
    assert len(table) == 8 and len(lines) == 2
    real = table[table.control == "real"]
    shuffled = table[table.control == "shuffled gene labels"]
    assert (real.CORUM_pct_enriched_may2025 == 100.0).all()
    assert (real.complexes_recovered_pct == 100.0).all()
    assert shuffled.CORUM_pct_enriched_may2025.mean() < 15
    assert shuffled.STRING_precision.mean() < real.STRING_precision.min()
    p, r = real.iloc[0].STRING_precision, real.iloc[0].STRING_recall
    assert real.iloc[0].STRING_f1 == 2 * p * r / (p + r)
    assert list(table.columns[-4:]) == ["graph", "seed", "resolution", "control"]
    again = oc.score_partitions(names, {("knn12", 42, 1.0): block, ("knn12", 42, 2.0): block}, refs,
                                shuffles=3)
    pd.testing.assert_frame_equal(table, again)


def test_scoring_a_named_gene_list_ignores_other_rows(tmp_path):
    names, _rows, block = _blocks(n_blocks=4, size=10)
    refs = oc.sweep_references(str(_references(tmp_path / "ref", names, block)))
    rows = np.r_[names, ["nontargeting_1", "nontargeting_2"]]
    labels = np.r_[block, [0, 1]]
    whole = oc.score_partitions(rows, {("phate", 1, 5): labels}, refs, shuffles=0)
    assert whole.genes.iloc[0] == 40 and whole.seed.iloc[0] == 1 and whole.resolution.iloc[0] == 5.0
    picked = oc.score_partitions(rows, {("phate", 1, 5): labels}, refs, shuffles=0, genes=names[:20])
    assert picked.genes.iloc[0] == 20 and picked.clusters.iloc[0] == 2


def test_the_summary_flags_the_resolution_nearest_the_target():
    rows = pd.DataFrame({"graph": ["phate"] * 6 + ["knn12"] * 2,
                         "resolution": [1.0, 1.0, 5.0, 5.0, 9.0, 9.0, 1.0, 1.0],
                         "seed": [1, 2] * 4, "control": ["real"] * 8,
                         "clusters": [10, 14, 200, 210, 400, 420, 30, 32]})
    for column in oc.SWEEP_METRICS[1:]:
        rows[column] = np.arange(len(rows), dtype=float)
    summary = oc.sweep_summary(rows, target=227)
    flagged = summary[summary.nearest_227 == True]  # noqa: E712
    assert flagged[["graph", "resolution"]].values.tolist() == [["knn12", 1.0], ["phate", 5.0]]
    assert summary.loc[(summary.graph == "phate") & (summary.resolution == 5.0), "clusters mean"].iloc[0] == 205
    assert summary.loc[(summary.graph == "phate") & (summary.resolution == 9.0), "clusters max"].iloc[0] == 420


def _sweep_settings(tmp_path, names, rows, block, **extra):
    """Settings of a cluster_sweep run on the planted rows."""
    table = pd.DataFrame(rows, columns=[f"c{j}" for j in range(rows.shape[1])])
    table.insert(0, "gene", names)
    table.insert(1, "n_cells", 50)
    table.to_parquet(tmp_path / "rows.parquet", index=False)
    return {"dst_root": str(tmp_path / "run"), "ops_cluster_rows": str(tmp_path / "rows.parquet"),
            "ops_cluster_reference_dir": str(_references(tmp_path / "ref", names, block)),
            "ops_cluster_knn": 4, "ops_cluster_seeds": "42,1",
            "ops_cluster_sweep_resolutions": {"knn": [1, 2], "phate": [3]},
            "ops_cluster_target_clusters": 8, **extra}


def test_the_phase_writes_rows_summary_assignments_and_figure(monkeypatch, tmp_path):
    _install(monkeypatch)
    names, rows, block = _blocks(n_blocks=8, size=12)
    settings = _sweep_settings(tmp_path, names, rows, block)
    out = run_ops(settings, phases=["cluster_sweep"])
    folder = tmp_path / "run" / "results" / "ops_clusters"
    table = pd.read_csv(folder / "sweep_rows.csv")
    assert len(table) == 2 * (2 * 2 + 2 * 1)
    assert set(table.graph) == {"phate", "knn4"}
    assert (table[table.control == "real"].clusters == 8).all()
    summary = pd.read_csv(folder / "sweep_summary.csv")
    assert summary.nearest_8.eq(True).sum() == 4
    assignments = pd.read_csv(folder / "assignments_knn4_s1.csv.gz")
    assert list(assignments.columns) == ["gene", "leiden_r1", "leiden_r2"]
    assert (folder / "sweep.png").stat().st_size > 1000
    assert out["outputs"]["sweep_rows"].endswith("sweep_rows.csv")
    record = (tmp_path / "run" / "ops_profile" / "A" / "runtime.json").read_text()
    assert "ops_cluster_knn" in record


def test_the_phase_embeds_the_profile_phases_gene_rows(monkeypatch, tmp_path):
    _install(monkeypatch)
    names, rows, block = _blocks(n_blocks=4, size=12, dims=6)
    settings = _sweep_settings(tmp_path, names, rows, block, ops_cluster_rows=None,
                               ops_cluster_graph="knn", ops_cluster_seeds=[3],
                               ops_reference_dir=None)
    genes = pd.DataFrame(rows, columns=[f"PC_{j}" for j in range(rows.shape[1])])
    genes.insert(0, "gene", names)
    genes.insert(1, "n_cells", [50] * (len(names) - 1) + [5])
    folder = tmp_path / "run" / "ops_profile" / "A"
    folder.mkdir(parents=True)
    genes.to_parquet(folder / "genes.parquet", index=False)
    run_ops(settings, phases=["cluster_sweep"])
    assignments = pd.read_csv(tmp_path / "run" / "results" / "ops_clusters" / "assignments_knn4_s3.csv.gz")
    assert len(assignments) == len(names) - 1


def test_the_phase_refuses_what_it_cannot_run(monkeypatch, tmp_path):
    _install(monkeypatch)
    names, rows, block = _blocks(n_blocks=2, size=6)
    settings = _sweep_settings(tmp_path, names, rows, block)
    with pytest.raises(ValueError, match="phate and/or knn"):
        oc.cluster_sweep_phase({**settings, "ops_cluster_graph": ["tsne"]}, None, str(tmp_path))
    with pytest.raises(ValueError, match="ops_cluster_reference_dir"):
        oc.cluster_sweep_phase({**settings, "ops_cluster_reference_dir": None}, None, str(tmp_path))
    with pytest.raises(ValueError, match="needs gene rows"):
        oc.cluster_sweep_phase({**settings, "ops_cluster_rows": None}, None, str(tmp_path))


def test_the_sweep_settings_are_alpha_with_short_tooltips():
    from spacr.ops_settings import OPS_DEFAULTS, OPS_TOOLTIPS
    from spacr.settings import ALPHA_FEATURES

    keys = ALPHA_FEATURES[916]["settings"]
    assert "ops_cluster_graph" in keys and "ops_cluster_gene_list" in keys
    for key in keys:
        assert key in OPS_DEFAULTS and len(OPS_TOOLTIPS[key]) <= 600 and "Alpha" in OPS_TOOLTIPS[key]
    assert OPS_DEFAULTS["ops_cluster_seeds"] == [42, 1, 2]
    assert OPS_DEFAULTS["ops_cluster_target_clusters"] == 227


def test_real_leiden_recovers_planted_blocks_at_a_matching_resolution():
    pytest.importorskip("leidenalg")
    pytest.importorskip("igraph")
    names, rows, block = _blocks(n_blocks=6, size=15)
    graph = oc.gene_graph(rows, "knn", k=8)
    found = oc.leiden_partitions(graph, [0.5], [42])
    labels = found[(42, 0.5)]
    assert len(np.unique(labels)) == 6
    assert all(len(np.unique(labels[block == b])) == 1 for b in range(6))

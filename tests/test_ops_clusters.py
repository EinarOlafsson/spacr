"""Gene-cluster benchmarks (item 855).

The scoring functions are checked against the plate A pilot's own code,
kept here verbatim as the reference (the phenotype agent's
``funk_benchmark.py`` group enrichment and STRING pair scores, and
``cluster_bench.py``'s precision-recall curve). The PHATE + Leiden steps need
the optional ``clustering`` extra and are skipped without it.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from spacr import ops_clusters as oc


# --- the pilot's code, verbatim -------------------------------------------

def _ref_group_enrichment(groups, genes, clusters, restrict=True, ctrl="nontargeting"):
    bg = set(genes)
    keep = ~pd.Series(genes).str.lower().str.startswith(ctrl).to_numpy()
    g2s = {g: (set(d.gene_name) & bg if restrict else set(d.gene_name))
           for g, d in groups.groupby("group")}
    g2s = {g: s for g, s in g2s.items() if s}
    names = list(g2s)
    sizes = np.array([len(g2s[n]) for n in names])
    n_bg = len(bg)
    table = []
    cl = pd.DataFrame({"gene": np.asarray(genes)[keep], "cluster": clusters[keep]})

    @lru_cache(maxsize=None)
    def fisher(a, k, n):
        b, c = k - a, n - a
        return fisher_exact([[a, b], [c, max(n_bg - (a + b + c), 0)]])[1]

    member = {n: g2s[n] for n in names}
    for cid, d in cl.groupby("cluster"):
        cg = set(d.gene)
        n = len(cg)
        a = [len(cg & member[nm]) for nm in names]
        p = np.array([fisher(int(ai), int(k), n) for ai, k in zip(a, sizes)])
        q = multipletests(p, method="fdr_bh")[1]
        enriched = [names[i] for i in np.flatnonzero(q < 0.05)]
        table.append({"cluster": int(cid), "n_genes": n, "n_enriched": len(enriched),
                      "enriched": "; ".join(enriched[:10])})
    t = pd.DataFrame(table)
    return float((t.n_enriched > 0).mean()), t


def _ref_string_pr(pairs, genes, clusters, ctrl="nontargeting"):
    keep = ~pd.Series(genes).str.lower().str.startswith(ctrl).to_numpy()
    g = np.asarray(genes)[keep]
    c = clusters[keep]
    pos = set()
    for _, d in pairs.groupby("pair"):
        u = sorted(d.gene_name.unique())
        pos |= {(u[i], u[j]) for i in range(len(u)) for j in range(i + 1, len(u))}
    tp = fp = 0
    for cid in np.unique(c):
        m = sorted(g[c == cid])
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                if (m[i], m[j]) in pos:
                    tp += 1
                else:
                    fp += 1
    g2c = dict(zip(g, c))
    fn = sum(1 for a, b in pos if a in g2c and b in g2c and g2c[a] != g2c[b])
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def _ref_pr_curve(pcs, names, pos, rng, shuffle=False):
    from sklearn.metrics import average_precision_score
    keep = ~pd.Series(names).str.lower().str.startswith("nontargeting").to_numpy()
    x = pcs[keep]
    g = np.asarray(names)[keep]
    if shuffle:
        g = rng.permutation(g)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    s = x @ x.T
    iu = np.triu_indices(len(g), 1)
    sc = s[iu]
    a, b = g[iu[0]], g[iu[1]]
    lab = np.fromiter(((min(p, q), max(p, q)) in pos for p, q in zip(a, b)), bool, len(sc))
    o = np.argsort(-sc, kind="stable")
    lab = lab[o]
    return float(average_precision_score(lab, -np.arange(len(lab)))), float(lab.mean())


# --- data -----------------------------------------------------------------

def _screen(seed=0, modules=6, per=8, noise=0.3, features=12):
    """Genes in modules sharing a profile, CORUM/KEGG groups and STRING pairs
    that follow the modules, and control rows."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 2, (modules, features))
    names, rows, module_of = [], [], []
    for m in range(modules):
        for k in range(per):
            names.append(f"M{m}G{k}")
            rows.append(centres[m] + rng.normal(0, noise, features))
            module_of.append(m)
    for k in range(10):
        names.append(f"nontargeting_{k}")
        rows.append(rng.normal(0, noise, features))
        module_of.append(-1)
    cols = [f"f{j}" for j in range(features)]
    table = pd.DataFrame(np.vstack(rows), columns=cols)
    table.insert(0, "gene", names)
    table.insert(1, "n_cells", 100)
    corum = pd.DataFrame([(f"M{m}G{k}", f"complex{m}") for m in range(modules)
                          for k in range(per // 2)], columns=["gene_name", "group"])
    kegg = pd.DataFrame([(f"M{m}G{k}", f"path{m // 2}") for m in range(modules)
                         for k in range(per)], columns=["gene_name", "group"])
    string = pd.DataFrame([(f"M{m}G{k}", m * 100 + k // 2) for m in range(modules)
                           for k in range(per)], columns=["gene_name", "pair"])
    return table, cols, corum, kegg, string, np.array(module_of)


# --- scoring, against the pilot ------------------------------------------

@pytest.mark.parametrize("restrict", [True, False])
def test_group_enrichment_is_the_pilots(restrict):
    table, _cols, corum, kegg, _s, modules = _screen()
    rng = np.random.default_rng(1)
    for clusters in (np.where(modules < 0, 99, modules), rng.integers(0, 5, len(modules))):
        for groups in (corum, kegg):
            ours = oc.group_enrichment(groups, table.gene.to_numpy(), clusters, restrict=restrict)
            ref = _ref_group_enrichment(groups, table.gene.to_numpy(), clusters, restrict=restrict)
            assert ours[0] == ref[0]
            pd.testing.assert_frame_equal(ours[1], ref[1], check_dtype=False)


def test_head_and_may2025_differ_only_by_members_outside_the_background():
    table, _cols, corum, _k, _s, modules = _screen()
    extra = pd.DataFrame({"gene_name": ["NOT_SCREENED"] * 30, "group": "complex0"})
    groups = pd.concat([corum, extra], ignore_index=True)
    clusters = np.where(modules < 0, 99, modules)
    head, _ = oc.group_enrichment(groups, table.gene.to_numpy(), clusters, restrict=True)
    may, _ = oc.group_enrichment(groups, table.gene.to_numpy(), clusters, restrict=False)
    assert head >= may


def test_string_pairs_are_the_pilots():
    table, _c, _co, _k, string, modules = _screen()
    rng = np.random.default_rng(2)
    positives = oc.string_positives(string)
    for clusters in (np.where(modules < 0, 99, modules), rng.integers(0, 4, len(modules))):
        ours = oc.string_cluster_pairs(positives, table.gene.to_numpy(), clusters)
        assert ours == _ref_string_pr(string, table.gene.to_numpy(), clusters)


def test_string_auprc_is_the_pilots_and_its_shuffle_sits_at_the_base_rate():
    table, cols, _co, _k, string, _m = _screen()
    pcs, _n, _f = oc.embed_profiles(table, cols)
    positives = oc.string_positives(string)
    ap, base, curve = oc.string_auprc(pcs, table.gene.to_numpy(), positives)
    ref_ap, ref_base = _ref_pr_curve(pcs, table.gene.to_numpy(), positives, None)
    assert (ap, base) == (ref_ap, ref_base)
    assert ap > 5 * base
    shuffled = [oc.string_auprc(pcs, table.gene.to_numpy(), positives, shuffle_seed=s)[0]
                for s in range(5)]
    assert shuffled[0] == _ref_pr_curve(pcs, table.gene.to_numpy(), positives,
                                        np.random.default_rng(0), shuffle=True)[0]
    assert np.mean(shuffled) < 3 * base
    assert curve["recall"].iloc[-1] == 1.0 and curve["rank"].is_monotonic_increasing


def test_score_row_has_every_benchmark_column():
    table, _c, corum, kegg, string, modules = _screen()
    row = oc.score_clusters("x", table.gene.to_numpy(), np.where(modules < 0, 99, modules),
                            corum=corum, kegg=kegg, positives=oc.string_positives(string))
    assert row["clusters"] == 6 and row["genes"] == 48
    for key in ("CORUM_pct_enriched_head", "CORUM_pct_enriched_may2025",
                "KEGG_pct_enriched_head", "KEGG_pct_enriched_may2025",
                "STRING_precision", "STRING_recall"):
        assert key in row
    assert row["STRING_recall"] == 1.0


def test_benchmark_rows_keep_control_guides_as_rows():
    genes = pd.DataFrame({"gene": ["A", "B", "nontargeting"], "n_cells": [50, 5, 900],
                          "f": [1.0, 2.0, 0.0]})
    guides = pd.DataFrame({"guide": ["a1", "nt1", "nt2"], "gene": ["A", "nontargeting",
                                                                   "nontargeting"],
                           "n_cells": [50, 400, 8], "f": [1.0, 0.1, -0.1]})
    rows = oc.benchmark_rows(genes, guides, ["f"])
    assert list(rows.gene) == ["A", "nontargeting_nt1"]


def test_embedding_drops_sparse_and_constant_columns():
    table, cols, *_ = _screen()
    table["sparse"] = np.where(np.arange(len(table)) % 5 == 0, np.nan, 1.0)
    table["constant"] = 3.0
    pcs, n, used = oc.embed_profiles(table, [*cols, "sparse", "constant"])
    assert used == len(cols) + 1
    assert pcs.shape == (len(table), n)


def test_shuffled_columns_keep_their_values():
    x = np.arange(20, dtype=float).reshape(10, 2)
    y = oc.shuffle_columns(x, seed=0)
    assert sorted(y[:, 0]) == sorted(x[:, 0]) and not np.array_equal(x, y)


def test_ari_is_one_against_itself_and_near_zero_shuffled():
    genes = np.array([f"g{i}" for i in range(60)])
    clusters = np.repeat(np.arange(6), 10)
    ref = pd.DataFrame({"gene": genes, "cluster": clusters})
    out = oc.adjusted_rand_vs(ref, genes, clusters)
    assert out["ari"] == 1.0 and out["ari_shuffled_p95"] < 0.2 and out["genes"] == 60


def test_closest_resolution_breaks_ties_by_order():
    assert oc.closest_resolution({5: 200, 10: 250, 12: 252}, 251) == 10


def test_co_membership_is_a_probability_over_runs_that_saw_both():
    labels = np.array([[0, 0, 1, -1],
                       [0, 1, 1, 1],
                       [2, 2, -1, 2]])
    p = oc.co_membership(labels)
    assert p[0, 1] == pytest.approx(2 / 3)
    assert p[1, 2] == pytest.approx(1 / 2)
    assert p[0, 3] == pytest.approx(1 / 2)
    assert np.allclose(np.diag(p), 1)
    assert np.allclose(p, p.T, equal_nan=True)


def test_consensus_rows_group_controls_into_pseudo_genes():
    guides = pd.DataFrame({"guide": [f"nt{i}" for i in range(9)] + ["a1", "a2", "b1"],
                           "gene": ["nontargeting"] * 9 + ["A", "A", "B"],
                           "n_cells": 20, "PC_0": np.arange(12.0)})
    genes = pd.DataFrame({"gene": ["A", "B"], "n_cells": [40, 20]})
    rows = oc.consensus_rows(genes, guides, ["PC_0"])
    # 9 control guides make two full pseudo-genes of 4; the ninth is left out.
    assert [g for g in rows.gene if g.startswith("NTpseudo")] == ["NTpseudo_00", "NTpseudo_01"]
    assert rows.set_index("gene").loc["A", "PC_0"] == 9.5
    again = oc.consensus_rows(genes, guides, ["PC_0"], rng=np.random.default_rng(0))
    assert list(again.gene) == list(rows.gene)


def test_reference_clusters_find_funk_style_columns(tmp_path):
    path = tmp_path / "funk.csv"
    pd.DataFrame({"gene_symbol": ["A", "B", "C"], "interphase_cluster": [1.0, None, 3.0]}
                 ).to_csv(path, index=False)
    ref = oc.reference_clusters(str(path))
    assert list(ref.gene) == ["A", "C"] and ref.cluster.dtype.kind == "i"
    with pytest.raises(ValueError, match="cluster"):
        oc.reference_clusters(str(path), cluster="nope")


def test_benchmark_tables_are_checked_for_their_columns(tmp_path):
    good = tmp_path / "corum.tsv"
    pd.DataFrame({"gene_name": ["A"], "group": ["x"]}).to_csv(good, sep="\t", index=False)
    assert len(oc.read_benchmark(str(good))) == 1
    bad = tmp_path / "bad.tsv"
    pd.DataFrame({"gene": ["A"]}).to_csv(bad, sep="\t", index=False)
    with pytest.raises(ValueError, match="gene_name"):
        oc.read_benchmark(str(bad))


def test_resolution_settings_parse():
    assert oc._resolutions("5, 10,12.5") == [5, 10, 12.5]
    assert oc._resolutions(None) == list(oc.DEFAULT_RESOLUTIONS)


def test_missing_clustering_extra_says_how_to_install(monkeypatch):
    import importlib

    real = importlib.import_module

    def fake(name, *a, **k):
        if name == "phate":
            raise ImportError("no phate")
        return real(name, *a, **k)

    monkeypatch.setattr(oc, "import_module", fake)
    with pytest.raises(ImportError, match=r"spacr\[clustering\]"):
        oc.phate_leiden(np.zeros((5, 2)), [1])


# --- PHATE + Leiden (optional extra) -------------------------------------

def test_phate_leiden_recovers_modules_and_is_reproducible():
    pytest.importorskip("phate")
    pytest.importorskip("leidenalg")
    table, cols, corum, kegg, string, modules = _screen(per=12)
    pcs, _n, _f = oc.embed_profiles(table, cols)
    a = oc.phate_leiden(pcs, [1, 2])
    b = oc.phate_leiden(pcs, [1, 2])
    for r in (1, 2):
        np.testing.assert_array_equal(a[r], b[r])
    from sklearn.metrics import adjusted_rand_score
    keep = modules >= 0
    assert adjusted_rand_score(modules[keep], a[1][keep]) > 0.8


def test_benchmark_beats_its_shuffled_control():
    pytest.importorskip("phate")
    pytest.importorskip("leidenalg")
    table, cols, corum, kegg, string, modules = _screen(per=12)
    ref = pd.DataFrame({"gene": table.gene[modules >= 0], "cluster": modules[modules >= 0]})
    out = oc.benchmark_clusters(table, cols, corum=corum, kegg=kegg, string=string,
                                resolutions=[1, 2], target=6, reference=ref,
                                baselines=(6,), shuffles=2)
    t = out["table"]
    real = t[(t.method == "PHATE+Leiden") & ~t.profile.str.contains("shuffled")]
    shuf = t[(t.method == "PHATE+Leiden") & t.profile.str.contains("shuffled")]
    assert real.STRING_precision.max() > shuf.STRING_precision.max()
    assert out["summary"]["reference_ARI"] > out["summary"]["reference_ARI_shuffled_p95"]
    assert out["summary"]["string_AUPRC_cosine"] > out["summary"]["string_AUPRC_shuffled_mean"]
    assert set(out["assignments"].columns) == {"gene", "n_cells", "leiden_r1", "leiden_r2"}


def test_consensus_clustering_gives_high_co_membership_within_modules():
    pytest.importorskip("phate")
    pytest.importorskip("leidenalg")
    rng = np.random.default_rng(0)
    table, cols, *_rest, modules = _screen(per=12)
    rows = []
    for _, r in table.iterrows():
        for k in range(3):
            rows.append({"guide": f"{r.gene}_{k}", "gene": "nontargeting" if r.gene.startswith(
                "nontargeting") else r.gene, "n_cells": 30,
                **{c: r[c] + rng.normal(0, 0.1) for c in cols}})
    guides = pd.DataFrame(rows)
    genes = guides.groupby("gene", as_index=False).n_cells.sum()
    out = oc.consensus_clustering(genes, guides, cols, runs=4, resolutions=[1, 2])
    p = oc.co_membership(out["labels"])
    names = list(out["genes"])
    i, j, k = names.index("M0G0"), names.index("M0G1"), names.index("M3G0")
    assert np.nanmean([p[i, j]]) > 0.7 > p[i, k]
    assert [r["resolution"] for r in out["runs"]] == [1, 2, 1, 2]


# --- the aggregate phase -------------------------------------------------

def test_the_aggregate_phase_asks_for_the_benchmark_tables(tmp_path):
    from spacr import ops_engine
    from tests.test_ops_aggregate_blockwise import PLATE, _plate, _settings

    db, wells, _ = _plate(tmp_path, n=200)
    settings = _settings(tmp_path, ops_bootstrap=0, ops_cluster_benchmark=True)
    with pytest.raises(ValueError, match="ops_benchmark_corum"):
        ops_engine._ph_aggregate(db, PLATE, wells, settings, str(tmp_path))


def test_the_aggregate_phase_benchmarks_the_tvn_profiles(tmp_path):
    pytest.importorskip("phate")
    pytest.importorskip("leidenalg")
    from spacr import ops_engine
    from tests.test_ops_aggregate_blockwise import GENES, PLATE, _plate, _settings

    db, wells, _ = _plate(tmp_path, n=600)
    names = sorted(set(GENES))
    for name, frame in (("corum", pd.DataFrame({"gene_name": names, "group": "g"})),
                        ("kegg", pd.DataFrame({"gene_name": names, "group": "k"})),
                        ("string", pd.DataFrame({"gene_name": names, "pair": 1}))):
        frame.to_csv(tmp_path / f"{name}.tsv", sep="\t", index=False)
    report = ops_engine._ph_aggregate(db, PLATE, wells, _settings(
        tmp_path, ops_bootstrap=0, ops_profile_normalisation="tvn", ops_cluster_benchmark=True,
        ops_benchmark_corum=str(tmp_path / "corum.tsv"),
        ops_benchmark_kegg=str(tmp_path / "kegg.tsv"),
        ops_benchmark_string=str(tmp_path / "string.tsv"),
        ops_cluster_resolutions="1", ops_cluster_target=2), str(tmp_path))
    assert report["clusters"]["outputs"]["cluster_benchmark"].endswith(f"{PLATE}_tvn.csv")

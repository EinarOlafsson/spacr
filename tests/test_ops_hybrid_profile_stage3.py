"""OPS stage 3: the Hybrid gene profile (Brieflow cell recipe + one fitted gene-level space) and its trim option."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import trim_mean

from spacr import ops_clusters as oc
from spacr import ops_reference as R
from spacr import ops_tvn as T


def _tables(seed=0, genes=30, guides=4, dims=12):
    """A gene table and its guide table in one feature space, with a few control rows and a constant column."""
    rng = np.random.default_rng(seed)
    pcs = [f"PC_{j}" for j in range(dims)]
    names = [f"G{g:02d}" for g in range(genes)] + [f"nontargeting_nt{k}" for k in range(5)]
    rows, gene_rows = [], []
    for name in names:
        centre = rng.standard_normal(dims) * (0.3 if name.startswith("non") else 2.0)
        count = 1 if name.startswith("non") else guides
        members = centre + rng.standard_normal((count, dims)) * 0.5
        cells = rng.integers(15, 60, count)
        for k in range(count):
            label = name.split("_", 1)[1] if name.startswith("non") else f"{name}_g{k}"
            rows.append((label, name, int(cells[k]), *members[k]))
        gene_rows.append((name, int(cells.sum()), *np.median(members, axis=0)))
    genes_t = pd.DataFrame(gene_rows, columns=["gene", "n_cells", *pcs])
    guides_t = pd.DataFrame(rows, columns=["guide", "gene", "n_cells", *pcs])
    genes_t["PC_3"] = 1.0
    guides_t["PC_3"] = 1.0
    return genes_t, guides_t, pcs


def test_hybrid_rows_are_the_evaluate_embedding():
    genes, guides, pcs = _tables()
    rows, guide_rows = R.hybrid_profile_rows(genes, guides)
    E = rows[[c for c in rows.columns if c.startswith("E_")]].to_numpy()
    assert np.array_equal(E, R.embed_gene_profiles(genes[pcs].to_numpy(float)))
    assert rows.gene.tolist() == genes.gene.tolist() and rows.n_cells.tolist() == genes.n_cells.tolist()
    assert guide_rows[["guide", "gene", "n_cells"]].equals(guides[["guide", "gene", "n_cells"]])
    assert [c for c in guide_rows.columns if c.startswith("E_")] == [c for c in rows.columns if c.startswith("E_")]


def test_guide_rows_go_through_the_fitted_transform():
    genes, guides, pcs = _tables(seed=1)
    _rows, guide_rows = R.hybrid_profile_rows(genes, guides)
    x = genes[pcs].to_numpy(float)
    keep = x.std(axis=0) > 0
    mean, spread = x[:, keep].mean(axis=0), x[:, keep].std(axis=0)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=0.99, random_state=0).fit((x[:, keep] - mean) / spread)
    want = pca.transform((guides[pcs].to_numpy(float)[:, keep] - mean) / spread)
    got = guide_rows[[c for c in guide_rows.columns if c.startswith("E_")]].to_numpy()
    assert np.abs(got - want).max() < 1e-12


def test_a_gap_in_a_guide_row_takes_the_gene_rows_fill_value():
    genes, guides, pcs = _tables(seed=2)
    guides.loc[0, "PC_1"] = np.nan
    filled = guides.copy()
    filled.loc[0, "PC_1"] = float(np.median(genes["PC_1"]))
    a = R.hybrid_profile_rows(genes, guides)[1]
    b = R.hybrid_profile_rows(genes, filled)[1]
    assert np.array_equal(a.filter(like="E_").to_numpy(), b.filter(like="E_").to_numpy())


def test_without_guides_and_with_a_missing_column():
    genes, guides, pcs = _tables(seed=3)
    rows, none = R.hybrid_profile_rows(genes)
    assert none is None and len(rows) == len(genes)
    with pytest.raises(ValueError, match="lacks the profile columns"):
        R.hybrid_profile_rows(genes, guides.drop(columns="PC_2"))


def test_a_guide_bootstrap_in_the_hybrid_space_is_the_embedded_raw_bootstrap():
    genes, guides, pcs = _tables(seed=4)
    rows, guide_rows = R.hybrid_profile_rows(genes, guides)
    names = genes.gene.to_numpy()
    raw = oc.guide_bootstrap_profiles(genes[pcs].to_numpy(float), names, guides[pcs].to_numpy(float),
                                      guides.gene, guides.n_cells, 7)
    fitted = R._GeneEmbedding(genes[pcs].to_numpy(float))
    hybrid = oc.guide_bootstrap_profiles(rows.filter(like="E_").to_numpy(), names,
                                         guide_rows.filter(like="E_").to_numpy(), guides.gene, guides.n_cells, 7)
    assert np.abs(hybrid - fitted.transform(raw)).max() < 1e-9


def test_recipe_takes_the_trimmed_mean_and_brieflow_tvn():
    assert T.profile_recipe({"agg": "gene_trim"})["agg"] == "gene_trim"
    assert T.PROFILE_TRIM_FRACTION == 0.1
    assert T.profile_recipe()["tvn"] == "spacr" and T.profile_recipe({"tvn": "brieflow"})["tvn"] == "brieflow"
    with pytest.raises(ValueError, match="must be one of"):
        T.profile_recipe({"tvn": "pycytominer"})


def test_brieflow_rotation_is_sklearns_pca_of_the_control_cells():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(9)
    x = (rng.standard_normal((400, 5)) @ rng.standard_normal((5, 5))).astype(np.float32)
    control = rng.random(400) < 0.4
    got, info = T.tvn_on_controls(x.copy(), control, None, whiten_epsilon=0.0, rotation="brieflow")
    y = StandardScaler().fit(x[control]).transform(x)
    y = PCA().fit(y[control]).transform(y)
    want = StandardScaler().fit(y[control]).transform(y)
    assert got.dtype == np.float32 and np.array_equal(got, want)
    default, _ = T.tvn_on_controls(x.copy(), control, None, whiten_epsilon=0.0)
    assert np.allclose(np.abs(default), np.abs(got), atol=1e-3)
    with pytest.raises(ValueError, match="rotation"):
        T.tvn_on_controls(x.copy(), control, None, rotation="svd")


def _cells(seed=0, dims=6):
    """Per-well cell matrices and metadata as Brieflow's filter step writes them."""
    rng = np.random.default_rng(seed)
    values, metas = [], []
    for well in ("A1", "A2"):
        genes = [f"G{g}" for g in range(8) for _ in range(30)] + [f"nontargeting_nt{k}" for k in range(4)
                                                                    for _ in range(40)]
        guides = [f"G{g}_g{k % 3}" for g in range(8) for k in range(30)] + [f"NT{k}" for k in range(4)
                                                                             for _ in range(40)]
        shift = np.array([int(g[1:]) if g.startswith("G") else 0 for g in genes], float)[:, None] * 0.3
        x = (rng.standard_normal((len(genes), dims)) + shift).astype(np.float32)
        values.append(x)
        metas.append(pd.DataFrame({"gene_symbol_0": genes, "cell_barcode_0": guides,
                                   "batch_values": f"A_{well}"}))
    return values, metas


@pytest.mark.parametrize("agg", ["gene", "gene_trim"])
def test_cells_to_profiles_reduces_cells_by_the_recipe(agg):
    values, metas = _cells()
    recipe = T.profile_recipe({"pca": "brieflow", "agg": agg, "nt": "barcode", "tvn": "brieflow"})
    genes, guides, info = T.cells_to_profiles([v.copy() for v in values], metas, recipe, whiten_epsilon=0.0,
                                              keep_cells=True)
    E, meta = info["kept_cells"]
    pcs = [c for c in genes.columns if c.startswith("PC_")]
    reduce = (lambda x: trim_mean(x, 0.1, axis=0)) if agg == "gene_trim" else (lambda x: np.median(x, axis=0))
    for name in ("G3", "nontargeting_nt2"):
        want = reduce(E[meta.gene_symbol_0.to_numpy() == name])
        assert np.array_equal(genes.set_index("gene").loc[name, pcs].to_numpy(np.float32), want.astype(np.float32))
    guide = guides.set_index("guide").loc["G3_g1"]
    rows = (meta.gene_symbol_0 == "G3").to_numpy() & (meta.cell_barcode_0 == "G3_g1").to_numpy()
    assert np.array_equal(guide[pcs].to_numpy(np.float32), reduce(E[rows]).astype(np.float32))
    assert guide["gene"] == "G3" and guides.set_index("guide").loc["nt2", "gene"] == "nontargeting_nt2"
    assert info["cells"] == len(meta) and info["tvn_dims"] == len(pcs)


def test_trimmed_means_follow_scipy_in_cell_order():
    rng = np.random.default_rng(5)
    cells = pd.DataFrame(rng.standard_normal((50, 3)).astype(np.float32), columns=["PC_0", "PC_1", "PC_2"])
    cells["gene"] = rng.choice(["a", "b", "c"], 50)
    out = T._trimmed_means(cells.groupby("gene"), cells, ["PC_0", "PC_1", "PC_2"])
    for name in "abc":
        want = trim_mean(cells.loc[cells.gene == name, ["PC_0", "PC_1", "PC_2"]].to_numpy(), 0.1, axis=0)
        assert np.array_equal(out.loc[name].to_numpy(), want)
    assert out.dtypes.tolist() == [np.float32] * 3


def test_a_universe_orders_genes_then_controls_and_drops_other_guides():
    values, metas = _cells(seed=1)
    recipe = T.profile_recipe({"pca": "brieflow", "agg": "gene", "nt": "barcode"})
    genes, guides, _ = T.cells_to_profiles(values, metas, recipe, whiten_epsilon=0.0, universe=["G5", "G1", "G9"])
    assert genes.gene.tolist() == ["G5", "G1"] + [f"nontargeting_nt{k}" for k in range(4)]
    assert set(guides.gene) == set(genes.gene)
    assert values == []


def test_the_consensus_phase_clusters_profile_tables_in_the_hybrid_space():
    genes, guides, pcs = _tables(seed=6)
    data = oc._guide_inputs({}, genes, guides)
    rows, guide_rows = R.hybrid_profile_rows(genes, guides, pcs)
    assert np.array_equal(data["rows"], rows.filter(like="E_").to_numpy(np.float32).astype(float))
    assert np.array_equal(data["guide_rows"], guide_rows.filter(like="E_").to_numpy(np.float32).astype(float))
    draw = oc._draw(data, 8000)
    want = oc.guide_bootstrap_profiles(data["rows"], data["names"], data["guide_rows"], data["guide_genes"],
                                       data["guide_cells"], 8000)[data["position"]]
    assert np.array_equal(draw, want)
    assert np.array_equal(oc._draw(data, None), data["rows"][data["position"]])


def test_the_previous_space_refits_the_embedding_on_every_draw():
    genes, guides, pcs = _tables(seed=7)
    data = oc._guide_inputs({"ops_consensus_space": "refit"}, genes, guides)
    assert np.array_equal(data["rows"], genes[pcs].to_numpy(float))
    assert data["transform"] is R.embed_gene_profiles
    raw = oc.guide_bootstrap_profiles(data["rows"], data["names"], data["guide_rows"], data["guide_genes"],
                                      data["guide_cells"], 8001)[data["position"]]
    assert np.array_equal(oc._draw(data, 8001), R.embed_gene_profiles(raw))
    assert oc._guide_inputs({}, genes, guides)["transform"] is None
    with pytest.raises(ValueError, match="ops_consensus_space"):
        oc._guide_inputs({"ops_consensus_space": "pca"}, genes, guides)


def test_ready_rows_are_clustered_as_given_in_either_space(tmp_path):
    genes, guides, pcs = _tables(seed=8)
    genes.to_parquet(tmp_path / "rows.parquet")
    guides.to_parquet(tmp_path / "guides.parquet")
    for space in ("hybrid", "refit"):
        data = oc._guide_inputs({"ops_cluster_rows": str(tmp_path / "rows.parquet"),
                                 "ops_consensus_guides": str(tmp_path / "guides.parquet"),
                                 "ops_consensus_space": space}, None, None)
        assert data["transform"] is None
        assert np.array_equal(data["rows"], genes[pcs].to_numpy(float))


def test_blas_threads_default_and_pin():
    from threadpoolctl import threadpool_info

    from spacr.ops_settings import OPS_DEFAULTS, ops_defaults

    assert OPS_DEFAULTS["ops_blas_threads"] == 2 and ops_defaults({})["ops_blas_threads"] == 2
    assert T._blas_threads({}) == 2 and T._blas_threads({"ops_blas_threads": 3}) == 3
    with pytest.raises(ValueError, match="at least 1"):
        T._blas_threads({"ops_blas_threads": 0})
    with T._pinned_blas(True, 2) as how:
        assert how == "threadpoolctl"
        assert max(pool["num_threads"] for pool in threadpool_info()) <= 2
    with T._pinned_blas(False, 2) as how:
        assert how is None


def test_the_profile_path_runs_at_ops_blas_threads(monkeypatch, tmp_path):
    from threadpoolctl import threadpool_info

    seen = {}

    def fake_state(settings, phases, state, classifier, folder, suffix, run, outputs, say):
        """Record the BLAS pool sizes the profile phase ran under."""
        seen["threads"] = {pool["num_threads"] for pool in threadpool_info()}
        return {}

    monkeypatch.setattr(T, "_profile_state", fake_state)
    out = T.run_profile_phases({"dst_root": str(tmp_path), "ops_blas_threads": 3}, phases=("profile",))
    assert max(seen["threads"]) <= 3 and out["info"]["blas_thread_count"] == 3
    out = T.run_profile_phases({"dst_root": str(tmp_path)}, phases=("profile",))
    assert max(seen["threads"]) <= 2 and out["info"]["blas_thread_count"] == 2
    out = T.run_profile_phases({"dst_root": str(tmp_path), "ops_deterministic_blas": False}, phases=("profile",))
    assert out["info"]["blas_threads"] is None and out["info"]["blas_thread_count"] is None


def test_older_settings_files_keep_one_thread_and_the_refit_space():
    from spacr.settings import DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION, _fold_older_defaults

    assert DEFAULT_CHANGES[936] == {"ops_blas_threads": 1, "ops_consensus_space": "refit"}
    assert SETTINGS_SCHEMA_VERSION >= 936
    old = {"plate": "P1"}
    applied = _fold_older_defaults(old, 931, known={"plate", "ops_blas_threads", "ops_consensus_space"})
    assert applied == {"ops_blas_threads": (1, 936), "ops_consensus_space": ("refit", 936)}
    assert old["ops_blas_threads"] == 1 and old["ops_consensus_space"] == "refit"
    current = {"plate": "P1"}
    assert _fold_older_defaults(current, SETTINGS_SCHEMA_VERSION) == {}
    kept = {"ops_blas_threads": 4}
    assert "ops_blas_threads" not in _fold_older_defaults(kept, 0, known={"ops_blas_threads"})

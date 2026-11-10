"""Item 882: specificity metrics for gene profiles, and the evaluation that runs them (880 + 881 + 882)."""
import numpy as np
import pandas as pd
import pytest

from spacr import ops_reference as R
from spacr import sp_stats as S


@pytest.fixture(autouse=True)
def _fresh_cache():
    R.clear_cache()
    yield
    R.clear_cache()


def _complexes(n_complex=8, size=6, n_free=30, n_nt=6, seed=0, d=12):
    """Genes in complexes that share a direction, free genes and NT rows."""
    rng = np.random.default_rng(seed)
    names, groups, rows = [], {}, []
    for c in range(n_complex):
        centre = rng.standard_normal(d) * 2
        members = [f"C{c}G{k}" for k in range(size)]
        groups[f"cx{c:02d}"] = members
        for m in members:
            names.append(m)
            rows.append(centre + rng.standard_normal(d))
    for k in range(n_free):
        names.append(f"F{k}")
        rows.append(rng.standard_normal(d) * 1.5)
    for k in range(n_nt):
        names.append(f"nontargeting_{k}")
        rows.append(rng.standard_normal(d) * 0.3)
    names = np.array(names)
    cpg = {n: float(50 + 400 * rng.random()) for n in names if not n.startswith("non")}
    return names, np.array(rows), groups, cpg


def test_fitness_bins_and_decoys():
    cpg = np.array([1, 10, 100, 1000, 5, 50, 500, 5000, 0.5, 7.0])
    controls = np.zeros(10, bool)
    controls[-1] = True
    bins = S.fitness_bins(cpg, controls, n_bins=3)
    assert bins[-1] == -1 and set(bins[:-1]) == {0, 1, 2}
    assert (S.fitness_bins(cpg, n_bins=2) >= 0).all()
    pos = np.zeros((6, 6), bool)
    pos[0, 1] = pos[1, 0] = True
    pos[2, 3] = pos[3, 2] = True
    decoys = S.fitness_matched_positives(pos, np.array([0, 1, 0, 1, 1, 0]), seed=0)
    assert (decoys == decoys.T).all() and not decoys.diagonal().any()
    assert not (decoys & pos).any()
    assert decoys[0].sum() == 1 and decoys[2].sum() >= 1
    stuck = S.fitness_matched_positives(np.array([[0, 1], [1, 0]], bool), np.array([0, 1]))
    assert not stuck.any()


def test_fitness_matched_retrieval_rewards_partner_specific_signal():
    names, X, groups, cpg = _complexes()
    sims = S.profile_cosine(X)
    np.fill_diagonal(sims, -np.inf)
    pos = R.positive_matrix(names, R._group_pairs(groups))
    controls = R.is_control(names)
    bins = S.fitness_bins([cpg.get(n, 100.0) for n in names], controls, n_bins=3)
    out = S.fitness_matched_retrieval(sims, pos, bins, controls=controls, decoys=3)
    assert out["queries"] == 48 and out["mAP"] > 0.9 > out["expected_mAP_random"]
    assert out["excess_over_random"] == pytest.approx(out["mAP"] - out["expected_mAP_random"])
    assert out["mAP_lo"] <= out["mAP"] <= out["mAP_hi"]
    assert S.fitness_matched_retrieval(sims, pos, bins, decoys=3)["queries"] == 48
    alone = np.zeros((3, 3), bool)
    alone[0, 1] = alone[1, 0] = True
    empty = S.fitness_matched_retrieval(np.eye(3), alone, np.array([0, 0, 1]))
    assert empty["queries"] == 0 and np.isnan(empty["mAP"]) and np.isnan(empty["excess_over_random"])


def test_pair_auprc_with_and_without_resampling():
    names, X, groups, _ = _complexes()
    sims = S.profile_cosine(X)
    pos = R.positive_matrix(names, R._group_pairs(groups))
    controls = R.is_control(names)
    out = S.pair_auprc(sims, pos, controls=controls, n_boot=20, seed=1)
    targeting = int((~controls).sum())
    assert out["pairs"] == targeting * (targeting - 1) // 2
    assert out["AUPRC"] > 5 * out["base_rate"]
    assert out["AUPRC_lo"] <= out["AUPRC_hi"]
    inner = S.pair_auprc(sims, pos, universe=~controls)
    assert inner["AUPRC"] == out["AUPRC"] and "AUPRC_lo" not in inner


def test_heldout_members_and_complex_identification():
    names, X, groups, cpg = _complexes()
    sims = np.nan_to_num(S.profile_cosine(X))
    controls = R.is_control(names)
    bins = S.fitness_bins([cpg.get(n, 100.0) for n in names], controls, n_bins=3)
    out = S.heldout_complex_members(sims, names, bins, groups, controls=controls, decoys=4, n_boot=20)
    assert out["complexes"] == 8 and out["hidden"] == 48 and out["AUROC"] > 0.9
    assert out["AP_random"] == pytest.approx(out["hidden"] / (out["hidden"] + out["decoys"]))
    folds = S.heldout_complex_members(sims, names, bins, groups, folds=7, min_members=6, n_boot=5)
    assert folds["hidden"] == 48
    none = S.heldout_complex_members(sims, names, bins, {"x": ["C0G0"]})
    assert none["complexes"] == 0 and np.isnan(none["AUROC"])
    sims0 = sims.copy()
    np.fill_diagonal(sims0, 0.0)
    ident = S.complex_identification(sims0, names, groups, controls=controls)
    assert ident["complexes"] == 8 and ident["members"] == 48 and ident["top1"] > 0.9
    assert ident["random_top1"] == pytest.approx(1 / 8) and ident["margin"] > 0
    few = S.complex_identification(sims0, names, dict(list(groups.items())[:3]))
    assert few["members"] == 0 and np.isnan(few["top1"])


def test_time_split_calls_and_control():
    names = np.array(["A", "B", "C", "D", "A&X", "nontargeting_1"])
    sims = np.array([[0, .9, .8, .1, .95, .5], [.9, 0, .7, .2, .3, .5], [.8, .7, 0, .6, .1, .5],
                     [.1, .2, .6, 0, .1, .5], [.95, .3, .1, .1, 0, .5], [.5] * 5 + [0]])
    bins = np.array([0, 0, 0, 0, 0, -1])
    controls = R.is_control(names)
    rows = S.time_split_pairs(sims, names, bins, {("A", "B")}, {("A", "C"), ("C", "D")},
                              controls=controls, top=(2, 50), seed=0)
    assert [r["calls"] for r in rows] == [2, 50]
    assert rows[0]["confirmed_later"] == 0.5 and rows[0]["confirmed_count"] == 1
    assert 0 <= rows[0]["p_binomial_vs_control"] <= 1 and rows[0]["enrichment"] > 0
    assert rows[1]["confirmed_count"] == 3
    lonely = S.time_split_pairs(sims[:2, :2], names[:2], np.array([0, 1]), {("A", "B")}, set(), top=(3,))
    assert np.isnan(lonely[0]["p_binomial_vs_control"])
    single = S.time_split_pairs(sims[:2, :2], names[:2], np.array([0, 1]), set(), set(), top=(1,))
    assert single[0]["confirmed_later_fitness_matched_random (control)"] == 0.0
    many = np.array([f"G{i}" for i in range(200)] + ["Z"])
    close = np.zeros((201, 201))
    close[0, 200] = close[200, 0] = 1.0
    stuck = S.time_split_pairs(close, many, np.r_[np.zeros(200, int), 1],
                               {(f"G{i}", "Z") for i in range(1, 200)}, {("G0", "Z")}, top=(1,))
    assert stuck[0]["confirmed_later"] == 1.0
    assert stuck[0]["confirmed_later_fitness_matched_random (control)"] == pytest.approx(0.2)


def _reference_dir(folder, names, groups):
    rows = [(g, c) for c, members in groups.items() for g in members]
    pd.DataFrame(rows, columns=["gene_name", "group"]).to_csv(folder / "corum_group_benchmark.tsv", sep="\t",
                                                               index=False)
    pd.DataFrame(rows[::2], columns=["gene_name", "group"]).to_csv(folder / "kegg_group_benchmark.tsv",
                                                                    sep="\t", index=False)
    pairs = sorted(R._group_pairs(groups))
    pd.DataFrame([(g, k) for k, p in enumerate(pairs[::3]) for g in p], columns=["gene_name", "pair"]).to_csv(
        folder / "string_pair_benchmark.tsv", sep="\t", index=False)
    targeting = [n for n in names if not n.startswith("non")]
    extra = [tuple(sorted((targeting[i], targeting[i + 1]))) for i in range(0, len(targeting) - 1, 5)]
    frame = pd.DataFrame(pairs + extra, columns=["a", "b"])
    frame["combined_score"] = np.where(np.arange(len(frame)) % 2 == 0, 900, 500)
    frame.to_parquet(folder / "string_v12_ge400.parquet", index=False)
    frame[["a", "b"]].iloc[::2].to_parquet(folder / "biogrid_physical.parquet", index=False)
    old, new = frame.iloc[:20][["a", "b"]], frame.iloc[20:60][["a", "b"]]
    for name in ("string_v11.5_ge700", "string_v11.5_ge400", "biogrid_4.4.214_physical", "corum3_pairs"):
        old.to_parquet(folder / f"{name}.parquet", index=False)
    for name in ("string_new_v12_ge700", "biogrid_new_after_4.4.214", "corum_new_after_3.0"):
        new.to_parquet(folder / f"{name}.parquet", index=False)


def test_evaluation_rows_are_the_metrics_called_directly(tmp_path):
    names, X, groups, cpg = _complexes(n_free=40)
    _reference_dir(tmp_path, names, groups)
    pcs = R.embed_gene_profiles(X)
    settings = {"ops_eval_null_size": 200, "ops_eval_time_split": True, "ops_eval_decoys": 4}
    metrics, per_gene = R.evaluate_gene_profiles(names, pcs, cpg, tmp_path, settings, arm="toy",
                                                 null_cache=str(tmp_path / "nulls"))
    assert metrics.metric_set.tolist() == (["retrieval"] * 23 + ["NT control"] + ["pair AUPRC"] * 3
                                           + ["held-out CORUM"] * 2 + ["time split"] * 2)
    assert metrics.reference.tolist()[:4] == ["CORUM", "CORUM", "CORUM", "STRING>=0.7"]
    assert (metrics.arm == "toy").all() and (metrics.genes == 88).all()
    corum = per_gene[per_gene.reference == "CORUM"].drop(columns="reference").reset_index(drop=True)
    sims = S.profile_cosine(pcs)
    np.fill_diagonal(sims, -np.inf)
    direct = S.pair_map(sims, names, R.positive_matrix(names, R.reference_pairs("CORUM", tmp_path)),
                        controls=R.is_control(names), null_size=200)
    pd.testing.assert_frame_equal(corum, direct)
    assert metrics.iloc[0]["mAP"] == pytest.approx(direct.ap.mean())
    assert per_gene[per_gene.reference == "CORUM@SELECT"].p.isna().all()
    assert metrics.loc[1, "mAP"] < metrics.loc[0, "mAP"]
    extra = R.specificity_metrics(names, pcs, cpg, tmp_path, settings, arm="toy")
    assert extra.metric_set.tolist() == (["specificity: fitness-matched non-partners"] * 3
                                         + ["complex identification"] * 2 + ["time split"] * 2)
    assert extra.loc[3, "top1"] > extra.loc[4, "top1"]
    half, _ = R.evaluate_gene_profiles(names, pcs, cpg, tmp_path,
                                       {"ops_eval_split": "select", "ops_eval_references": "CORUM",
                                        "ops_eval_null_size": 100}, arm="toy")
    assert half.reference.tolist()[0] == "CORUM@SELECT" and "time split" not in set(half.metric_set)
    assert R.specificity_metrics(names, pcs, cpg, tmp_path).metric_set.tolist()[-1] == "complex identification"


def test_evaluation_settings_and_degenerate_profiles(tmp_path):
    options = R.evaluation_options({"ops_eval_references": "KEGG", "ops_eval_null_size": "50",
                                    "ops_eval_decoys": None, "ops_eval_split": "SELECT"})
    assert options["ops_eval_references"] == ["KEGG"] and options["ops_eval_null_size"] == 50
    assert options["ops_eval_decoys"] == 10 and options["ops_eval_split"] == "select"
    assert R.evaluation_options()["ops_eval_references"] == list(R.MAIN_REFERENCES)
    with pytest.raises(ValueError, match="unknown references"):
        R.evaluation_options({"ops_eval_references": ["GO"]})
    with pytest.raises(ValueError, match="at least 1"):
        R.evaluation_options({"ops_eval_heldout_folds": 0})
    with pytest.raises(ValueError):
        R.evaluation_options({"ops_eval_split": "half"})
    names = np.array([f"G{i}" for i in range(10)])
    with pytest.raises(ValueError, match="zero vectors"):
        R.evaluate_gene_profiles(names, np.zeros((10, 3)), {}, tmp_path)
    with pytest.raises(ValueError, match="tied"):
        R.evaluate_gene_profiles(np.array([f"G{i}" for i in range(400)]), np.ones((400, 3)), {}, tmp_path)
    cpg = R._fitness(np.array(["A", "nontargeting_1", "Q"]), {"A": 4.0, "B": 8.0}, fill=False)
    assert cpg[0] == 4.0 and cpg[1] == 6.0 and np.isnan(cpg[2])
    assert R._fitness(np.array(["Q", "A"]), {"A": 4.0}, fill=True).tolist() == [4.0, 4.0]


def test_embedding_drops_sparse_and_constant_columns():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((50, 6))
    X[:, 1] = 3.0
    X[:10, 2] = np.nan
    X[0, 3] = np.inf
    pcs = R.embed_gene_profiles(X, variance=0.999)
    assert pcs.shape[0] == 50 and pcs.shape[1] <= 4 and np.isfinite(pcs).all()

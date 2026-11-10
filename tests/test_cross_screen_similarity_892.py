"""Item 892: cross-screen gene similarity (an OPS screen against JUMP consensus profiles)."""
import json

import numpy as np
import pandas as pd
import pytest

from spacr import sp_stats as S


def _screens(seed=0, n=60, d=8, shared=0.8):
    """Two screens whose gene profiles share structure, plus controls and annotations."""
    rng = np.random.default_rng(seed)
    genes = [f"G{i:02d}" for i in range(n - 3)] + ["RPL5", "TUBB", "FLAG1"]
    latent = rng.standard_normal((n, 4))
    a = latent @ rng.standard_normal((4, d)) * shared + rng.standard_normal((n, d))
    b = latent @ rng.standard_normal((4, d + 22)) * shared + rng.standard_normal((n, d + 22))
    ops = pd.DataFrame(a, columns=[f"PC_{j}" for j in range(d)])
    ops.insert(0, "gene", genes)
    ops.insert(1, "n_cells", rng.integers(20, 500, n))
    ops.loc[3, "PC_0"] = np.inf
    ops = pd.concat([ops, pd.DataFrame({"gene": ["NTpseudo_00", "nontargeting"], "n_cells": [100, 90],
                                        **{f"PC_{j}": [0.1, 0.2] for j in range(d)}})], ignore_index=True)
    crispr = pd.DataFrame(b, index=pd.Index(genes, name="gene"), columns=[f"f{j}" for j in range(d + 22)])
    orf = pd.DataFrame(-b[5:] + rng.standard_normal((n - 5, d + 22)) * 0.1, index=pd.Index(genes[5:], name="gene"),
                       columns=[f"f{j}" for j in range(d + 22)])
    nt = {m: pd.DataFrame(rng.standard_normal((30, d + 22)) * 0.5, columns=crispr.columns,
                          index=[f"NT_{m}_{k}" for k in range(30)]) for m in ("crispr", "orf")}
    return ops, crispr, orf, nt, genes


def _references(tmp_path, genes):
    pairs = [(genes[i], genes[i + 1]) for i in range(0, 50, 2)]
    string = pd.DataFrame(pairs, columns=["a", "b"])
    string["combined_score"] = [900 if k % 2 else 500 for k in range(len(string))]
    string.to_parquet(tmp_path / "string.parquet", index=False)
    pd.DataFrame(pairs[:5], columns=["a", "b"]).to_parquet(tmp_path / "biogrid.parquet", index=False)
    pd.DataFrame(pairs[5:9], columns=["x", "y"]).to_parquet(tmp_path / "corum.parquet", index=False)
    pd.DataFrame({"gene": ["FLAG1", "G01"], "specificity_flag": [1, 0]}).to_csv(tmp_path / "flag.csv", index=False)


def test_concordance_finds_shared_structure_and_its_nulls_do_not():
    rng = np.random.default_rng(3)
    n = 40
    latent = rng.standard_normal((n, 3))
    a = latent @ rng.standard_normal((3, 6)) + 0.3 * rng.standard_normal((n, 6))
    b = latent @ rng.standard_normal((3, 9)) + 0.3 * rng.standard_normal((n, 9))
    genes = [f"G{i}" for i in range(n)]
    sa, sb = S._cross_cosine(a), S._cross_cosine(b)
    known = {frozenset((genes[0], genes[1]))}
    rows, draws = S.cross_screen_concordance(sa, sb, rng.random(n), np.linalg.norm(b, axis=1), rng.integers(0, 9, n),
                                             known, genes, n_perm=20, seed=0)
    real = rows[rows.group == "real"].set_index("statistic").value
    assert real["Mantel Spearman r"] > 0.3 and real["neighbour overlap@10"] > 0.3
    shuffled = rows[(rows.group == "control: shuffled gene labels")].set_index("statistic")
    assert shuffled.loc["Mantel Spearman r", "p_perm"] == pytest.approx(1 / 21)
    assert abs(shuffled.loc["Mantel Spearman r", "value"]) < 0.1
    assert len(rows) == 25 and len(draws) == 80 and set(draws.group) == set(rows.group) - {"real"}
    p = S.stratified_permutation(np.array([0, 0, 1, 1, 1, 2]), np.random.default_rng(1))
    assert sorted(p[:2]) == [0, 1] and sorted(p[2:5]) == [2, 3, 4] and p[5] == 5
    assert np.allclose(S.remove_generic_axes(np.eye(3), [[1, 0, 0]]), np.diag([0, 1, 1]))


def test_top_pairs_are_annotated():
    sims_a = np.array([[1, .9, .1, .2], [.9, 1, .3, .1], [.1, .3, 1, .8], [.2, .1, .8, 1]])
    sims_b = np.array([[1, .7, .2, .1], [.7, 1, .1, .2], [.2, .1, 1, .6], [.1, .2, .6, 1]])
    genes = ["A", "B", "RPL5", "D"]
    out = S.cross_screen_pairs(sims_a, sims_b, genes, top_fraction=0.4, nt_threshold=0.65,
                               string_scores={frozenset(("A", "B")): 900}, biogrid={frozenset(("RPL5", "D"))},
                               flagged={"D"}, fitness=pd.Series({"A": 0.05, "B": 0.05, "RPL5": 0.5, "D": 0.9}))
    assert out[["gene_a", "gene_b"]].values.tolist() == [["A", "B"], ["RPL5", "D"]]
    first, second = out.iloc[0], out.iloc[1]
    assert first.string_v12 == 0.9 and first.known and first.both_essential_decile and first.jump_cos_above_nt99
    assert second.biogrid_physical and second.guide_flag and second.ribo_or_tubulin and not second.jump_cos_above_nt99
    assert not out.new_clean.any() and (out.hub_max_degree == 1).all()
    bare = S.cross_screen_pairs(sims_a, sims_b, genes, top_fraction=0.4, exclude=None)
    assert not bare.ribo_or_tubulin.any() and bare.fitness_a.isna().all() and bare.new_clean.all()


def test_the_whole_comparison_from_settings(tmp_path):
    ops, crispr, orf, nt, genes = _screens()
    _references(tmp_path, genes)
    ops.to_parquet(tmp_path / "ops.parquet", index=False)
    pairs = [{"item": "G00 vs G01", "gene_a": "G00", "partners": "G01"},
             {"gene_a": ["G10"], "partners": ["G11", "G12", "G13"]},
             {"item": "absent", "gene_a": "NONE", "partners": ["G11"]}]
    settings = {"cross_screen_nt_b": {"crispr": nt["crispr"], "orf": nt["orf"]},
                "cross_screen_string": str(tmp_path / "string.parquet"),
                "cross_screen_biogrid": str(tmp_path / "biogrid.parquet"),
                "cross_screen_corum": str(tmp_path / "corum.parquet"),
                "cross_screen_flagged": str(tmp_path / "flag.csv"), "cross_screen_generic_axes": [False, True],
                "cross_screen_permutations": 5, "cross_screen_pairs": pairs, "cross_screen_pair_draws": 50,
                "cross_screen_compounds": pd.DataFrame(np.ones((2, 3)), index=["c1", "c2"],
                                                       columns=["X_0", "X_1", "X_2"]),
                "cross_screen_compound_targets": pd.DataFrame({"target_list": ["G11|G12", "G11|A|B|C|D|E"]},
                                                              index=["c1", "c2"]),
                "cross_screen_compound_nt": pd.DataFrame(np.ones((4, 3)), columns=["X_0", "X_1", "X_2"])}
    result = S.cross_screen_similarity(str(tmp_path / "ops.parquet"), {"crispr": crispr, "orf": orf}, settings)
    conc = result["concordance"]
    assert set(conc.generic_axes_removed) == {False, True} and (conc.n_genes == 57).all()
    assert (conc.modality == "crispr").all()
    summary = result["pairs_summary"]
    assert summary[["modality", "generic_axes_removed"]].values.tolist() == [
        ["crispr", False], ["orf", False], ["crispr", True], ["orf", True]]
    assert summary.n_genes.tolist() == [60, 55, 60, 55] and np.isinf(summary.jump_nt99[2])
    named = result["named_pairs"]
    assert named.item.tolist() == ["G00 vs G01", "G10 vs 3 partners", "absent"]
    assert named.representations[0] == "crispr:G00" and named.n_partner_reps[1] == 8
    assert 0 < named.p_nt[1] <= 1 and named.n_rep_pairs[0] == 1 and named.n_rep_pairs[2] == 0
    assert result["record"]["profiles_a"] is not None and result["record"]["biogrid"] is not None
    written = S.write_cross_screen(result, str(tmp_path / "out"))
    summary_json = json.loads(open(written["summary.json"]).read())
    assert len(summary_json["real"]) == 10 and len(summary_json["controls"]) == 40
    single = S.cross_screen_similarity(ops, crispr, {"cross_screen_permutations": 3, "cross_screen_nt_b": nt["crispr"],
                                                     "cross_screen_exclude": None, "cross_screen_pair_draws": 20,
                                                     "cross_screen_pairs": [{"gene_a": "G00", "partners": "G01"}]})
    assert len(single["named_pairs"]) == 1 and (single["concordance"].n_genes == 60).all()
    unnamed = S.cross_screen_similarity(ops, crispr, {"cross_screen_permutations": 3,
                                                      "cross_screen_nt_b": nt["crispr"],
                                                      "cross_screen_exclude": None, "cross_screen_pairs": None})
    assert unnamed["named_pairs"].empty
    pd.testing.assert_frame_equal(unnamed["concordance"], single["concordance"])
    with pytest.raises(ValueError, match="cosine"):
        S.cross_screen_similarity(ops, crispr, {"cross_screen_metric": "pearson"})
    with pytest.raises(ValueError, match="cross_screen_modality"):
        S.cross_screen_similarity(ops, crispr, {"cross_screen_modality": "orf"})


def test_representations_compare_named_features_across_modalities():
    a = pd.DataFrame(np.eye(3, 25), index=["A", "B", "C"], columns=[f"f{j}" for j in range(25)])
    anonymous = pd.DataFrame(np.eye(3, 25), index=["A", "B", "C"], columns=[f"X_{j}" for j in range(25)])
    few = pd.DataFrame(np.eye(3, 5), index=["A", "B", "C"], columns=[f"f{j}" for j in range(5)])
    reps = S._Representations({"crispr": a, "orf": anonymous, "compound": few}, {"crispr": a, "orf": anonymous,
                                                                                "compound": few})
    assert reps.common("crispr", "orf") == [] and reps.common("crispr", "compound") == []
    assert len(reps.common("crispr", "crispr")) == 25
    reps = S._Representations({"crispr": a, "orf": a.iloc[:2]}, {"crispr": a, "orf": a},
                              fitness=pd.Series({"A": 0.1, "B": 0.15, "C": 0.9}))
    assert reps.reps("A") == [("crispr", "A"), ("orf", "A")] and reps.reps("C") == [("crispr", "C")]
    assert list(reps.pool("crispr", "A", "fitness")) == ["A", "B"]
    assert list(reps.pool("crispr", "Q", "fitness")) == ["A", "B", "C"]
    value, count = S._pair_values(reps, [("crispr", "A"), ("orf", "A")], [("orf", "B", np.array([1]))], ["G"])
    assert count == 2 and value[0] == 0.0
    nothing, zero = S._pair_values(reps, [("crispr", "A")], [("crispr", "A", np.array([0]))], ["G"])
    assert nothing is None and zero == 0

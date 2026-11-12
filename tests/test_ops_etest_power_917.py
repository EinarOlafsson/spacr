"""Per-gene E-distance and E-test against non-targeting cells, with a power curve (item 917, alpha).

The cells are planted: non-targeting cells and null genes are standard
normal, one gene is shifted. The shifted gene must be called with many cells
and missed with ten; null genes and held-out control pseudo-genes must be
called about as often as the test level says.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import gene_measurement_compare as G
from spacr.ops_engine import run_ops


def _planted(seed=0, dims=4, shift=0.45, nt_guides=30, nt_cells=60, null_genes=12, gene_cells=400):
    """Planted per-cell embedding, gene labels and guides."""
    rng = np.random.default_rng(seed)
    blocks, labels, guides = [], [], []
    for k in range(nt_guides):
        blocks.append(rng.standard_normal((nt_cells, dims)))
        labels += [f"nontargeting_nt{k}"] * nt_cells
        guides += [f"NT{k}"] * nt_cells
    blocks.append(rng.standard_normal((gene_cells, dims)) + shift)
    labels += ["SHIFT"] * gene_cells
    guides += ["S1"] * gene_cells
    for g in range(null_genes):
        blocks.append(rng.standard_normal((gene_cells, dims)))
        labels += [f"NULL{g}"] * gene_cells
        guides += [f"N{g}"] * gene_cells
    blocks.append(rng.standard_normal((2, dims)))
    labels += ["TINY"] * 2
    guides += ["T1"] * 2
    return np.vstack(blocks), np.array(labels), np.array(guides)


def _test(**options):
    """An EnergyTest on the planted cells with a small null."""
    cells, labels, guides = _planted()
    kwargs = dict(n_ref=500, n_null=399, n_holdout=200)
    kwargs.update(options)
    return G.EnergyTest(cells, labels, guides, **kwargs)


def test_the_e_distance_of_a_set_to_itself_is_near_zero_and_grows_with_a_shift():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 3))
    assert abs(G.energy_distance(x, x)) < 0.02
    assert abs(G.energy_distance(x, x, "sqeuclidean")) < 0.05
    near = G.energy_distance(x + 0.2, x)
    far = G.energy_distance(x + 1.0, x)
    assert 0 < near < far
    with pytest.raises(ValueError, match="E-distance metric"):
        G.energy_distance(x, x, "cosine")


def test_the_energy_statistic_is_the_textbook_formula():
    rng = np.random.default_rng(2)
    x, y = rng.standard_normal((7, 2)), rng.standard_normal((9, 2)) + 1
    d = lambda a, b: np.linalg.norm(a[:, None] - b[None], axis=2)  # noqa: E731
    within = lambda a: d(a, a).sum() / (len(a) * (len(a) - 1))  # noqa: E731
    expected = 2 * d(x, y).mean() - within(x) - within(y)
    assert G.EnergyReference(y).energy(x)[1] == pytest.approx(expected, rel=1e-7)
    sq = lambda a, b: d(a, b) ** 2  # noqa: E731
    expected = 2 * sq(x, y).mean() - sq(x, x).sum() / 42 - sq(y, y).sum() / 72
    assert G.EnergyReference(y).energy(x)[0] == pytest.approx(expected, rel=1e-10)


def test_a_shifted_gene_is_called_with_many_cells_and_missed_with_ten():
    test = _test()
    full = test.score(None)
    table = full["table"].set_index("gene")
    assert table.loc["SHIFT", "q_e_euclidean"] < 0.05 and table.loc["SHIFT", "cells"] == 400
    assert "TINY" not in table.index
    assert table.loc[[f"NULL{g}" for g in range(12)], "q_e_euclidean"].min() > 0.05
    assert {r["null_cells"] for r in full["summary"]} == {400}
    small = test.score(10, 0)["table"].set_index("gene")
    assert small.loc["SHIFT", "cells"] == 10 and small.loc["SHIFT", "q_e_euclidean"] >= 0.05
    assert small.loc["NTguide_NT0", "cells"] == 10 and np.isnan(small.loc["NTguide_NT0", "q_e_euclidean"])
    assert list(full["table"].columns) == ["gene", "cells", "e_sqeuclidean", "e_euclidean", "p_e_sqeuclidean",
                                           "q_e_sqeuclidean", "p_e_euclidean", "q_e_euclidean"]


def test_control_pseudo_genes_are_called_at_about_the_test_level():
    summary = _test(n_holdout=400).score(50, 0)["summary"]
    for row in summary:
        assert 1.0 <= row["nt_pseudo_p05_pct"] <= 10.0
        assert row["nt_guides"] == 30 and row["genes"] == 13
        assert 0.0 <= row["nt_guide_p05_pct"] <= 20.0


def test_the_same_seeds_give_the_same_tables():
    a = _test().score(25, 1)
    b = _test().score(25, 1)
    pd.testing.assert_frame_equal(a["table"], b["table"])
    c = _test().score(25, 2)
    assert not a["table"]["e_euclidean"].equals(c["table"]["e_euclidean"])


def test_the_permutation_mode_gives_valid_p_values():
    test = _test(mode="permutation", permutations=19, n_ref=300)
    out = test.score(30, 0)
    p = out["table"]["p_e_euclidean"]
    assert ((p >= 1 / 20) & (p <= 1)).all()
    assert np.isnan(out["summary"][0]["nt_pseudo_p05_pct"])
    assert out["table"].set_index("gene").loc["NULL0", "p_e_euclidean"] > 0.05


def test_the_guide_null_draws_four_guide_pseudo_genes_from_each_half():
    test = _test(mode="guide_null", n_holdout=100)
    first, second = test._halves
    assert len(first) == 15 and len(second) == 15 and not set(first) & set(second)
    out = test.score(50, 0)
    row = out["summary"][1]
    assert 0.0 <= row["nt_pseudo_p05_pct"] <= 15.0
    assert out["table"].set_index("gene").loc["NULL0", "p_e_euclidean"] > 0.01
    draw = np.random.default_rng(0)
    cells = test._pseudo_gene(draw, first, 10000)
    assert 0 < len(cells) <= 4 * 60 and len(np.unique(cells)) == len(cells)
    assert np.isin(cells, test.rest).all()
    assert len(test._pseudo_gene(np.random.default_rng(0), first, 5)) == 5


def test_bad_options_are_refused():
    cells, labels, guides = _planted()
    with pytest.raises(ValueError, match="empirical_null, guide_null or permutation"):
        G.EnergyTest(cells, labels, guides, mode="bootstrap")
    with pytest.raises(ValueError, match="more than 5000 control cells"):
        G.EnergyTest(cells, labels, guides, n_ref=5000)


def test_the_power_curve_scores_every_cap_and_seed():
    cells, labels, guides = _planted()
    power = G.subsample_power(cells, labels, [10, None], [0, 1], guides=guides, n_ref=500, n_null=399,
                              n_holdout=50)
    assert list(power["tables"]) == [("10", 0), ("10", 1), ("all", 0)]
    hits = power["hits"]
    assert len(hits) == 6 and list(hits["n"]) == ["10"] * 4 + ["all"] * 2
    assert hits.loc[hits.n == "all", "hits_q05"].min() >= 1
    one = G.energy_distance_test(cells, labels, n_ref=500, n_null=399, n_holdout=50, n_cells=10,
                                 guides=guides)
    pd.testing.assert_frame_equal(one["table"], power["tables"][("10", 0)])
    calls = []
    custom = G.subsample_power(cells, labels, [5, None], [3, 4],
                               scorer=lambda n, s: calls.append((n, s)) or {"table": None, "summary": []})
    assert calls == [(5, 3), (5, 4), (None, 3)] and custom["hits"].empty


def test_subsampled_profiles_are_medians_of_at_most_n_cells():
    cells, labels, guides = _planted()
    genes, guide_rows = G.subsample_profiles(cells.astype(np.float32), labels, guides, 50, 0)
    row = genes.set_index("gene").loc["SHIFT"]
    assert row["n_cells"] == 50
    assert genes.set_index("gene").loc["nontargeting_nt3", "n_cells"] == 60
    assert "TINY" not in set(genes.gene)
    assert genes.filter(like="PC_").dtypes.unique().tolist() == [np.dtype("float32")]
    nt = guide_rows.set_index("guide").loc["nt3"]
    assert nt["gene"] == "nontargeting" and nt["n_cells"] == 60
    expected = np.median(cells[labels == "nontargeting_nt3"].astype(np.float32), axis=0)
    assert np.array_equal(nt[[f"PC_{j}" for j in range(4)]].to_numpy(np.float32), expected)
    again = G.subsample_profiles(cells.astype(np.float32), labels, guides, 50, 0)[0]
    pd.testing.assert_frame_equal(genes, again)
    capped, _ = G.subsample_profiles(cells, labels, guides, 2, 0, min_cells=10)
    assert "TINY" in set(capped.gene)


def _cells_folder(tmp_path):
    """A stored per-cell embedding in the etest phase's layout."""
    cells, labels, guides = _planted()
    folder = tmp_path / "cells"
    folder.mkdir()
    np.save(folder / "cells_E.npy", cells.astype(np.float32))
    pd.DataFrame({"gene_symbol_0": labels, "cell_barcode_0": guides}).to_parquet(
        folder / "cells_meta.parquet", index=False)
    return folder


def test_the_etest_phase_writes_tables_power_and_figure(tmp_path):
    folder = _cells_folder(tmp_path)
    settings = {"dst_root": str(tmp_path / "run"), "ops_etest_cells": str(folder),
                "ops_etest_reference_cells": 500, "ops_etest_null": 99, "ops_etest_holdout": 50,
                "ops_power_cells": "10,all", "ops_power_seeds": [0, 1], "ops_etest_metric": "euclidean"}
    out = run_ops(settings, phases=["etest"])
    results = tmp_path / "run" / "results" / "ops_etest"
    table = pd.read_csv(results / "edist_per_gene_nall_s0.csv.gz")
    assert "e_sqeuclidean" not in table and "q_e_euclidean" in table
    hits = pd.read_csv(results / "power_hits.csv")
    assert set(hits.distance) == {"euclidean"} and len(hits) == 3
    assert (results / "power_hits.png").stat().st_size > 1000
    assert out["outputs"]["power_hits"].endswith("power_hits.csv")
    with pytest.raises(ValueError, match="ops_etest_metric"):
        run_ops({**settings, "ops_etest_metric": "cosine"}, phases=["etest"])
    with pytest.raises(ValueError, match="cells_E.npy"):
        run_ops({**settings, "ops_etest_cells": str(tmp_path)}, phases=["etest"])
    with pytest.raises(ValueError, match="ops_reference_dir"):
        run_ops({**settings, "ops_power_retrieval": True, "ops_power_cells": [10]}, phases=["etest"])


def _complex_cells(seed=0, dims=8, complexes=10, members=4, cells=200, nt_guides=30, nt_cells=60,
                   effect=1.2):
    """Planted cells: the genes of one complex share a direction; control cells are noise."""
    rng = np.random.default_rng(seed)
    blocks, labels, guides, groups = [], [], [], []
    for k in range(nt_guides):
        blocks.append(rng.standard_normal((nt_cells, dims)))
        labels += [f"nontargeting_nt{k}"] * nt_cells
        guides += [f"NT{k}"] * nt_cells
    for c in range(complexes):
        direction = rng.standard_normal(dims)
        direction *= effect / np.linalg.norm(direction)
        for m in range(members):
            gene = f"C{c}G{m}"
            groups.append((gene, f"complex{c}"))
            for guide in range(2):
                blocks.append(rng.standard_normal((cells // 2, dims)) + direction)
                labels += [gene] * (cells // 2)
                guides += [f"{gene}_{guide}"] * (cells // 2)
    return np.vstack(blocks).astype(np.float32), np.array(labels), np.array(guides), groups


def _retrieval_references(folder, groups):
    """CORUM, STRING and BioGRID tables whose partners are the planted complexes."""
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(groups, columns=["gene_name", "group"]).to_csv(
        folder / "corum_group_benchmark.tsv", sep="\t", index=False)
    by = {}
    for gene, group in groups:
        by.setdefault(group, []).append(gene)
    pairs = [(a, b) for members in by.values() for i, a in enumerate(sorted(members))
             for b in sorted(members)[i + 1:]]
    pd.DataFrame({"a": [p[0] for p in pairs], "b": [p[1] for p in pairs],
                  "combined_score": 900}).to_parquet(folder / "string_v12_ge400.parquet", index=False)
    pd.DataFrame({"a": [p[0] for p in pairs], "b": [p[1] for p in pairs]}).to_parquet(
        folder / "biogrid_physical.parquet", index=False)
    return folder


def _reference_ap(sims, positives, candidates):
    """retrieval.py's average precision (ic_lib._ap_rows), written out again."""
    out = np.full(len(sims), np.nan)
    columns = np.flatnonzero(candidates)
    for i in range(len(sims)):
        c = columns[columns != i]
        lab = positives[i, c]
        if not lab.any():
            continue
        order = np.lexsort((~lab, -sims[i, c].astype(np.float32)))
        hits = lab[order]
        out[i] = float((np.cumsum(hits) / np.arange(1, hits.size + 1) * hits).sum() / lab.sum())
    return out


def test_power_retrieval_is_the_reference_retrieval_on_one_gene_list(tmp_path):
    from scipy.stats import binomtest, wilcoxon

    from spacr import ops_reference

    _cells, _labels, _guides, groups = _complex_cells()
    refs = _retrieval_references(tmp_path / "refs", groups)
    rng = np.random.default_rng(4)
    genes = sorted({g for g, _ in groups})
    base = rng.standard_normal((len(genes), 6))
    arms = {"first": (np.array(genes + ["nontargeting_x"]), np.vstack([base, rng.standard_normal(6)])),
            "noisy": (np.array(genes[::-1]), (base + rng.normal(0, 1.5, base.shape))[::-1]),
            "short": (np.array(genes[1:]), base[1:] + rng.normal(0, 0.1, (len(genes) - 1, 6)))}
    out = G.power_retrieval(arms, str(refs))
    gene_list = np.array(genes[1:], dtype=object)
    assert out["genes"]["gene"].tolist() == list(gene_list)
    summary = out["summary"]
    assert summary[["arm", "reference"]].values.tolist() == [
        [a, r] for a in arms for r in G.POWER_RETRIEVAL_REFERENCES]
    shuffled = gene_list[np.random.default_rng(3).permutation(len(gene_list))]
    for arm, (names, rows) in arms.items():
        x = pd.DataFrame(rows, index=names).loc[gene_list].to_numpy(float)
        unit = x / np.linalg.norm(x, axis=1, keepdims=True)
        sims = (unit @ unit.T).astype(np.float32).astype(np.float64)
        np.fill_diagonal(sims, -np.inf)
        order = np.argsort(-sims, axis=1)[:, :10]
        for reference in G.POWER_RETRIEVAL_REFERENCES:
            pairs = ops_reference.reference_pairs(reference, str(refs))
            pos = ops_reference.positive_matrix(gene_list, pairs)
            ap = _reference_ap(sims, pos, pos.any(1))
            keep = np.isfinite(ap)
            hits = np.take_along_axis(pos[keep], order[keep], 1)
            spos = ops_reference.positive_matrix(shuffled, pairs)
            sap = _reference_ap(sims, spos, spos.any(1))
            row = summary[(summary.arm == arm) & (summary.reference == reference)].iloc[0]
            assert row["queries"] == keep.sum()
            assert row["mAP"] == pytest.approx(ap[keep].mean(), abs=1e-12)
            assert row["shuffled_mAP"] == pytest.approx(np.nanmean(sap), abs=1e-12)
            assert row["top10_partner_pct"] == pytest.approx(100 * hits.any(1).mean())
            assert row["p_at_10"] == pytest.approx(hits.mean(1).mean())
    tests = out["tests"]
    assert len(tests) == 2 * 2 * len(G.POWER_RETRIEVAL_REFERENCES)
    assert set(tests["vs"]) == {"first"} and set(tests["arm"]) == {"noisy", "short"}
    per_gene = out["per_gene"]
    corum = per_gene[per_gene.reference == "CORUM@REPORT"].pivot(index="gene", columns="arm", values="ap")
    pair = corum[["first", "noisy"]].dropna()
    row = tests[(tests.reference == "CORUM@REPORT") & (tests.arm == "noisy") & (tests.test == "Wilcoxon AP")]
    assert row["p"].iloc[0] == wilcoxon(pair["noisy"], pair["first"]).pvalue
    hit = per_gene[per_gene.reference == "CORUM@REPORT"].pivot(index="gene", columns="arm", values="top10_hit")
    mine = hit.loc[pair.index, "noisy"].astype(bool)
    theirs = hit.loc[pair.index, "first"].astype(bool)
    only, other = int((mine & ~theirs).sum()), int((~mine & theirs).sum())
    mcnemar = tests[(tests.reference == "CORUM@REPORT") & (tests.arm == "noisy")
                    & (tests.test == "McNemar top-10")]
    assert mcnemar["p"].iloc[0] == (binomtest(only, only + other, 0.5).pvalue if only + other else 1.0)
    assert (tests["p_holm"] >= tests["p"]).all() and (tests["p_holm"] <= 1).all()
    assert out["summary"].equals(G.power_retrieval(arms, str(refs))["summary"])


def test_power_retrieval_refuses_arms_without_a_shared_gene(tmp_path):
    _cells, _labels, _guides, groups = _complex_cells()
    refs = _retrieval_references(tmp_path / "refs", groups)
    with pytest.raises(ValueError, match="at least one arm"):
        G.power_retrieval({}, str(refs))
    with pytest.raises(ValueError, match="share no targeting gene"):
        G.power_retrieval({"a": (np.array(["C0G0", "C0G1"]), np.eye(2)),
                           "b": (np.array(["C1G0", "C1G1"]), np.eye(2))}, str(refs))
    single = G.power_retrieval({"a": (np.array(["C0G0", "C0G1", "C1G0"]), np.eye(3))}, str(refs))
    assert single["tests"].empty and "p_holm" in single["tests"].columns


def test_power_profile_rows_build_the_profile_phases_final_rows():
    from spacr import ops_reference

    cells, labels, guides, _groups = _complex_cells(dims=10, effect=2.0)
    genes, guide_rows = G.subsample_profiles(cells, labels, guides, len(labels), 0)
    names, rows = G.power_profile_rows(genes, guide_rows, {})
    pcs = [c for c in genes.columns if c.startswith("PC_")]
    assert list(names) == genes["gene"].tolist()
    np.testing.assert_array_equal(rows, ops_reference.embed_gene_profiles(genes[pcs].to_numpy(float)))
    listed = [f"C{c}G{m}" for c in range(10) for m in range(4)][::-1][:-2]
    settings = {"ops_profile_weighting": "contrastive_wiener", "ops_guide_efficacy": "jacks_blend",
                "ops_profile_gene_universe": listed + ["MISSING"]}
    names, rows = G.power_profile_rows(genes, guide_rows, settings)
    assert list(names[:38]) == listed and all(n.startswith("nontargeting_") for n in names[38:])
    renamed = guide_rows.copy()
    nt = renamed["gene"] == "nontargeting"
    renamed.loc[nt, "gene"] = "nontargeting_" + renamed.loc[nt, "guide"]
    order = listed + [g for g in genes["gene"] if g.startswith("nontargeting")]
    kept = genes.set_index("gene").loc[order].reset_index()
    weighted, weighted_guides, _info = G.contrastive_weighting(
        kept, renamed[renamed["gene"].isin(order)].reset_index(drop=True), pcs, min_cells=10)
    blended, _e = G.guide_efficacy_profiles(weighted, weighted_guides)
    w = [c for c in blended.columns if c.startswith("w_")]
    np.testing.assert_array_equal(rows, blended[w].to_numpy(np.float32))
    with pytest.raises(ValueError, match="weighted space"):
        G.power_profile_rows(genes, guide_rows, {"ops_guide_efficacy": "jacks_blend"})
    with pytest.raises(ValueError, match="ops_profile_weighting must be"):
        G.power_profile_rows(genes, guide_rows, {"ops_profile_weighting": "pca"})
    with pytest.raises(ValueError, match="ops_guide_efficacy must be"):
        G.power_profile_rows(genes, guide_rows, {"ops_guide_efficacy": "mean"})


def _complex_folder(tmp_path, **options):
    """The planted complex cells in the etest phase's layout, and their reference folder."""
    cells, labels, guides, groups = _complex_cells(**options)
    folder = tmp_path / "cells"
    folder.mkdir()
    np.save(folder / "cells_E.npy", cells)
    pd.DataFrame({"gene_symbol_0": labels, "cell_barcode_0": guides}).to_parquet(
        folder / "cells_meta.parquet", index=False)
    return folder, _retrieval_references(tmp_path / "refs", groups)


def test_the_retrieval_curve_scores_every_draw_on_one_gene_list(tmp_path):
    folder, refs = _complex_folder(tmp_path)
    pd.DataFrame({"gene": [f"C{c}G{m}" for c in range(10) for m in range(4)][2:]}).to_csv(
        tmp_path / "genes.csv", index=False)
    settings = {"dst_root": str(tmp_path / "run"), "ops_etest_cells": str(folder),
                "ops_etest_reference_cells": 500, "ops_etest_null": 19, "ops_etest_holdout": 10,
                "ops_power_cells": [10, 50, None], "ops_power_seeds": [0, 1], "ops_power_retrieval": True,
                "ops_reference_dir": str(refs), "ops_power_gene_list": str(tmp_path / "genes.csv")}
    run_ops(settings, phases=["etest"])
    results = tmp_path / "run" / "results" / "ops_etest"
    genes = pd.read_csv(results / "power_retrieval_genes.csv")
    assert len(genes) == 38 and "C0G0" not in set(genes.gene)
    summary = pd.read_csv(results / "power_retrieval_summary.csv")
    assert list(dict.fromkeys(summary.arm)) == ["all", "n10_s0", "n10_s1", "n50_s0", "n50_s1"]
    corum = summary[summary.reference == "CORUM@REPORT"].set_index("arm")
    assert corum.loc["all", "mAP"] > corum.loc["n10_s0", "mAP"]
    assert corum.loc["all", "mAP"] > 3 * corum.loc["all", "shuffled_mAP"]
    tests = pd.read_csv(results / "power_retrieval_tests.csv")
    assert set(tests.vs) == {"all"} and len(tests) == 4 * 2 * 3
    assert (results / "power_retrieval_pergene.csv.gz").is_file()
    assert (results / "power_profiles" / "genes_n10_s1.parquet").is_file()
    run_ops({**settings, "dst_root": str(tmp_path / "again")}, phases=["etest"])
    again = pd.read_csv(tmp_path / "again" / "results" / "ops_etest" / "power_retrieval_summary.csv")
    pd.testing.assert_frame_equal(summary, again)


def test_ready_reference_rows_are_the_first_arm(tmp_path):
    folder, refs = _complex_folder(tmp_path)
    cells = np.load(folder / "cells_E.npy")
    meta = pd.read_parquet(folder / "cells_meta.parquet")
    genes, guide_rows = G.subsample_profiles(cells, meta.gene_symbol_0, meta.cell_barcode_0, len(meta), 0)
    names, rows = G.power_profile_rows(genes, guide_rows, {})
    np.savez(tmp_path / "final.npz", names=names, pcs=rows.astype(np.float32))
    settings = {"dst_root": str(tmp_path / "run"), "ops_etest_cells": str(folder),
                "ops_etest_reference_cells": 500, "ops_etest_null": 19, "ops_etest_holdout": 10,
                "ops_power_cells": [25], "ops_power_seeds": [0], "ops_power_retrieval": True,
                "ops_reference_dir": str(refs), "ops_power_reference_rows": str(tmp_path / "final.npz")}
    run_ops(settings, phases=["etest"])
    results = tmp_path / "run" / "results" / "ops_etest"
    summary = pd.read_csv(results / "power_retrieval_summary.csv")
    assert list(dict.fromkeys(summary.arm)) == ["reference", "all", "n25_s0"]
    first = summary[summary.arm == "reference"].reset_index(drop=True)
    same = summary[summary.arm == "all"].reset_index(drop=True)
    np.testing.assert_allclose(first["mAP"], same["mAP"], atol=1e-6)
    tests = pd.read_csv(results / "power_retrieval_tests.csv")
    assert set(tests.vs) == {"reference"}
    table = pd.DataFrame(rows, columns=[f"c{j}" for j in range(rows.shape[1])])
    table.insert(0, "gene", names)
    table.to_csv(tmp_path / "final.csv", index=False)
    read_names, read_rows = G._power_rows_file(str(tmp_path / "final.csv"))
    assert list(read_names) == list(names) and np.allclose(read_rows, rows)


def test_the_profile_phase_keeps_the_cells_the_etest_reads(tmp_path):
    from tests.test_ops_profile_pipeline_879 import _cell_tables

    cells = _cell_tables(tmp_path / "cells", wells=("A1",))
    settings = {"dst_root": str(tmp_path), "ops_profile_cells": str(cells), "ops_tvn_whiten_epsilon": 0,
                "ops_profile_min_cells": 5}
    run_ops(settings, phases=("profile",))
    folder = tmp_path / "ops_profile" / "A"
    assert not (folder / "cells_E.npy").exists()
    run_ops({**settings, "ops_etest": True}, phases=("profile",))
    embedding = np.load(folder / "cells_E.npy")
    meta = pd.read_parquet(folder / "cells_meta.parquet")
    assert embedding.dtype == np.float32 and len(embedding) == len(meta)
    assert {"gene_symbol_0", "cell_barcode_0"} <= set(meta.columns)


def test_the_etest_settings_are_alpha_with_short_tooltips():
    from spacr.ops_settings import OPS_DEFAULTS, OPS_TOOLTIPS
    from spacr.settings import ALPHA_FEATURES

    keys = ALPHA_FEATURES[917]["settings"]
    assert "ops_etest" in keys and "ops_power_retrieval" in keys
    assert "ops_power_gene_list" in keys and "ops_power_reference_rows" in keys
    for key in keys:
        assert key in OPS_DEFAULTS and len(OPS_TOOLTIPS[key]) <= 600 and "Alpha" in OPS_TOOLTIPS[key]
    assert OPS_DEFAULTS["ops_power_cells"] == [10, 25, 50, 100, 200, 500, None]
    assert OPS_DEFAULTS["ops_etest"] is False

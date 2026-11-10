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


def test_the_retrieval_curve_scores_each_draw_and_tests_it(tmp_path, monkeypatch):
    from spacr import ops_reference

    def evaluate(names, pcs, fitness, reference_dir, settings, arm=""):
        genes = [n for n in names if not str(n).startswith("nontargeting")]
        ap = np.linspace(0.1, 0.9, len(genes)) * (1 + 0.1 * int(arm.split("_")[0][1:]) / 50)
        per_gene = pd.DataFrame({"gene": genes, "ap": ap, "reference": "CORUM"})
        return pd.DataFrame({"metric": ["mAP"], "value": [float(ap.mean())], "arm": [arm]}), per_gene

    monkeypatch.setattr(ops_reference, "evaluate_gene_profiles", evaluate)
    folder = _cells_folder(tmp_path)
    run_ops({"dst_root": str(tmp_path / "run"), "ops_etest_cells": str(folder),
             "ops_etest_reference_cells": 500, "ops_etest_null": 49, "ops_etest_holdout": 20,
             "ops_power_cells": [10, 50], "ops_power_seeds": [0, 1], "ops_power_retrieval": True,
             "ops_reference_dir": str(tmp_path), "ops_profile_min_cells": 5}, phases=["etest"])
    results = tmp_path / "run" / "results" / "ops_etest"
    summary = pd.read_csv(results / "power_retrieval_summary.csv")
    assert sorted(set(zip(summary.n, summary.seed))) == [(10, 0), (10, 1), (50, 0), (50, 1)]
    tests = pd.read_csv(results / "power_retrieval_tests.csv")
    assert list(tests.n) == [10, 10] and (tests.vs_n == 50).all() and (tests.p_wilcoxon < 0.05).all()
    assert (results / "power_profiles" / "genes_n10_s1.parquet").is_file()


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
    for key in keys:
        assert key in OPS_DEFAULTS and len(OPS_TOOLTIPS[key]) <= 600 and "Alpha" in OPS_TOOLTIPS[key]
    assert OPS_DEFAULTS["ops_power_cells"] == [10, 25, 50, 100, 200, 500, None]
    assert OPS_DEFAULTS["ops_etest"] is False

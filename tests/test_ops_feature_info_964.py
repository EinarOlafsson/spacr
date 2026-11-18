"""Item 964: the feature-information screen on Gaussian data with known answers."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from spacr import ops_feature_info as ofi

P, Q = 4, 3


def _cond(s, target, given):
    """The conditional covariance of ``target`` given ``given``."""
    return (s[np.ix_(target, target)] - s[np.ix_(target, given)]
            @ np.linalg.solve(s[np.ix_(given, given)], s[np.ix_(given, target)]))


def _data(genes=300, per=400, wells=4, seed=0):
    """Gene means and cell noise of known covariance; the analytic answer."""
    rng = np.random.default_rng(seed)
    between = rng.normal(size=(P + Q, P + Q)) * 0.3
    noise = rng.normal(size=(P + Q, P + Q)) * 0.6 + np.eye(P + Q)
    sw = noise @ noise.T
    mu = rng.multivariate_normal(np.zeros(P + Q), between @ between.T, size=genes)
    gene = np.repeat(np.arange(genes), per)
    x = mu[gene] + rng.multivariate_normal(np.zeros(P + Q), sw, size=len(gene))
    st = np.cov(mu.T, bias=True) + sw
    old_i, new_i = list(range(P)), list(range(P, P + Q))
    truth = {f"n{j}": 0.5 * np.log2(_cond(st, [P + j], old_i)[0, 0]
                                    / _cond(sw, [P + j], old_i)[0, 0])
             for j in range(Q)}
    truth["block"] = 0.5 * np.log2(np.linalg.det(_cond(st, new_i, old_i))
                                   / np.linalg.det(_cond(sw, new_i, old_i)))
    well = rng.integers(0, wells, size=len(gene))
    old = pd.DataFrame(x[:, :P], columns=[f"o{i}" for i in range(P)]).assign(
        cell=np.arange(len(gene)), gene=[f"G{g}" for g in gene], well=well,
        site=well * 100 + rng.integers(0, 10, size=len(gene)),
        sgRNA=[f"G{g}_{k}" for g, k in zip(gene, rng.integers(0, 2, len(gene)))])
    new = pd.DataFrame(x[:, P:], columns=[f"n{j}" for j in range(Q)]).assign(
        cell=np.arange(len(gene)),
        lin=x[:, :P] @ np.array([1.0, -2.0, 0.5, 0.0]),
        dup=x[:, 0], noise=rng.normal(size=len(gene)))
    return old, new, truth


@pytest.fixture(scope="module")
def result():
    old, new, truth = _data()
    out = ofi.screen(old, new, id_columns=["cell"],
                     families={"block": ["n0", "n1", "n2"]}, permutations=40,
                     device="cpu")
    return out, truth


def test_added_bits_match_the_analytic_conditional_information(result):
    out, truth = result
    table = out["table"].set_index("name")
    for name, value in truth.items():
        assert table.loc[name, "added_bits"] == pytest.approx(value, rel=0.01)


def test_linear_combinations_duplicates_and_noise_add_nothing(result):
    out, _truth = result
    table = out["table"].set_index("name")
    for name in ("lin", "dup"):
        assert abs(table.loc[name, "added_bits"]) < 1e-5
        assert table.loc[name, "unique_fraction"] < 1e-5
    noise = table.loc["noise"]
    assert noise["null_low"] <= noise["added_bits"] <= noise["null_high"]
    assert noise["unique_fraction"] == pytest.approx(1.0, abs=0.01)
    assert table.loc["n1", "added_bits"] > table.loc["n1", "null_high"]


def test_the_report_carries_the_null_ridge_heldout_guides_and_greedy(result):
    out, _truth = result
    table = out["table"].set_index("name")
    assert out["null"].shape == (40, len(table))
    assert out["summary"]["ridge"] == 1e-6 and out["summary"]["ridge_absolute"] > 0
    assert out["summary"]["heldout_split"] == "well"
    assert 0 < table.loc["block", "heldout_bits"] < table.loc["block", "added_bits"]
    assert table.loc["block", "guide_bits"] == pytest.approx(
        table.loc["block", "added_bits"], rel=0.05)
    assert table.loc["noise", "guide_bits"] < 0.01
    assert [g["item"] for g in out["greedy"]] == ["block"]
    assert table.loc["block", "greedy_rank"] == 1


def test_the_torch_path_gives_the_numpy_numbers():
    pytest.importorskip("torch")
    old, new, _truth = _data(genes=60, per=100)
    kwargs = dict(id_columns=["cell"], families={"block": ["n0", "n1", "n2"]},
                  permutations=5)
    a = ofi.screen(old, new, device="cpu", **kwargs)
    b = ofi.screen(old, new, device="torch:cpu", **kwargs)
    numeric = a["table"].select_dtypes("number").columns
    np.testing.assert_allclose(a["table"][numeric].to_numpy(float),
                               b["table"][numeric].to_numpy(float),
                               rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(a["null"], b["null"], rtol=1e-6, atol=1e-9)


def test_cell_permutations_and_a_one_well_table_split_by_field():
    old, new, _truth = _data(genes=40, per=50, wells=1)
    out = ofi.screen(old, new, id_columns=["cell"], permutations=3,
                     perm_unit="cell", device="cpu")
    assert out["summary"]["units"] == len(old)
    assert out["summary"]["heldout_split"] == "site"


def test_the_phase_writes_the_table_chart_and_summary(tmp_path):
    from spacr.ops_engine import run_ops

    old, new, _truth = _data(genes=40, per=50)
    old.to_parquet(tmp_path / "old.parquet")
    new.to_parquet(tmp_path / "new.parquet")
    (tmp_path / "fam.json").write_text(json.dumps({"block": ["n0", "n1"]}))
    out = run_ops({"ops_feature_screen_old": str(tmp_path / "old.parquet"),
                   "ops_feature_screen_new": str(tmp_path / "new.parquet"),
                   "ops_feature_screen_id_columns": "cell",
                   "ops_feature_screen_families": str(tmp_path / "fam.json"),
                   "ops_feature_screen_permutations": 4,
                   "ops_feature_screen_device": "cpu",
                   "dst_root": str(tmp_path)},
                  phases=("ops_feature_screen",))["ops_feature_screen"]
    folder = tmp_path / "ops_feature_screen"
    table = pd.read_csv(folder / "feature_information.csv")
    assert {"block", "all_new", "n0", "noise"} <= set(table["name"])
    assert (folder / "feature_information.png").is_file()
    assert json.loads((folder / "feature_information.json").read_text())[
        "permutations"] == 4
    assert out["new_columns"] == 6
    with pytest.raises(ValueError, match="runs on its own"):
        run_ops({}, phases=("ops_feature_screen", "score"))
    with pytest.raises(ValueError, match="needs ops_feature_screen_old"):
        run_ops({}, phases=("ops_feature_screen",))


def test_families_come_from_a_csv_too(tmp_path):
    pd.DataFrame({"column": ["n0", "n1"], "family": ["f", "f"]}).to_csv(
        tmp_path / "fam.csv", index=False)
    assert ofi._families(str(tmp_path / "fam.csv")) == {"f": ["n0", "n1"]}
    assert ofi._families(None) is None

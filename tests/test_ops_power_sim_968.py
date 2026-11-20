"""Item 968: the OPS power simulation (spacr.power_ops), on a small synthetic run."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from spacr import power_ops as po


def _run_folder(folder, genes=60, dims=12, seed=0):
    """A synthetic genotyped run: four guides per gene, NT guides, a funnel table."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(genes):
        for j in range(4):
            rows += [(f"g{g}", f"g{g}_{j}")] * int(rng.integers(20, 60))
    for j in range(40):
        rows += [("nontargeting", f"nt{j}")] * 120
    meta = pd.DataFrame(rows, columns=["gene_symbol_0", "cell_barcode_0"])
    meta["well"] = rng.choice(["A1", "A2"], len(meta))
    os.makedirs(folder, exist_ok=True)
    meta.to_parquet(os.path.join(folder, "cells_meta.parquet"))
    np.save(os.path.join(folder, "cells_E.npy"), rng.standard_normal((len(meta), dims)).astype(np.float32))
    pd.DataFrame({"well": ["A1"], "linked": [1000], "with_guide": [770]}).to_csv(
        os.path.join(folder, "funnel_A_A1.csv"), index=False)
    return folder


@pytest.fixture(scope="module")
def calibration(tmp_path_factory):
    folder = _run_folder(str(tmp_path_factory.mktemp("run")))
    out = os.path.join(folder, "calibration.json")
    po.calibrate_from_run(folder, out, fdr_pct=0.1, fdr_source="test")
    return out


def test_calibration_records_every_value_and_its_pool(calibration):
    with open(calibration, encoding="utf-8") as handle:
        values = json.load(handle)
    assert values["n_genes"] == 60 and values["nt_guides"] == 40
    assert values["call_rate"]["value"] == pytest.approx(0.77)
    assert values["call_rate"]["source"].startswith("funnel")
    assert values["fdr_pct"] == {"value": 0.1, "source": "test"}
    assert values["dims"] == 12 and len(values["nt_cov"]) == 12
    assert values["guide_alpha"] > 0 and values["pool_cells"] == 40 * 120
    assert "real_etest" in values and len(values["between_guide_var"]) == 12
    loaded = po.load_calibration(calibration)
    assert loaded["_pool"][0].shape == (4800, 12)


def test_a_changed_pool_is_refused(calibration, tmp_path):
    with open(calibration, encoding="utf-8") as handle:
        values = json.load(handle)
    values["pool_sha256"] = "0" * 64
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        po.load_calibration(str(bad))


def test_the_same_design_and_seed_draw_the_same_screen(calibration):
    cal = po.load_calibration(calibration)
    design = {"n_genes": 30, "cells_per_gene": 40, "hit_fraction": 0.3}
    a, b = po.simulate_ops_screen(cal, design, 3), po.simulate_ops_screen(cal, design, 3)
    assert np.array_equal(a["cells"], b["cells"]) and np.array_equal(a["labels"], b["labels"])
    assert a["truth"].equals(b["truth"])
    c = po.simulate_ops_screen(cal, design, 4)
    assert not np.array_equal(a["cells"][:100], c["cells"][:100])
    assert int(a["truth"]["hit"].sum()) == 9


@pytest.mark.parametrize("mode", ["empirical", "parametric"])
def test_a_strong_effect_is_found_and_nothing_is_found_without_one(calibration, mode):
    """The simulator's recovery check, scored with the pooled null.

    The synthetic run has no spread between guides, so the pooled null is
    valid here, and its 40 control guides are too few for whole-guide
    permutations to reach the p-values a 60-gene BH needs; the guide null's
    calibration is tested in test_guide_null_calibration_978.
    """
    cal = po.load_calibration(calibration)
    strong = po.simulate_ops_screen(cal, {"n_genes": 60, "cells_per_gene": 200, "effect_size": 1.0,
                                          "effect_dims": 5, "penetrance": 1.0, "hit_fraction": 0.2,
                                          "mode": mode, "etest_mode": "empirical_null"}, 0)
    row = po.score_ops_screen(strong, ["etest"], 0)
    assert row["etest_power"] >= 0.95
    null = po.simulate_ops_screen(cal, {"n_genes": 60, "cells_per_gene": 200, "hit_fraction": 0.0,
                                        "mode": mode, "etest_mode": "empirical_null"}, 1)
    row = po.score_ops_screen(null, ["etest"], 1)
    assert row["etest_fpr_null_genes"] <= 0.05


def test_a_lower_call_rate_leaves_fewer_cells_per_gene(calibration):
    cal = po.load_calibration(calibration)
    full = po.simulate_ops_screen(cal, {"n_genes": 20, "cells_per_gene": 100}, 0)
    half = po.simulate_ops_screen(cal, {"n_genes": 20, "cells_per_gene": 100, "call_rate": 0.385}, 0)
    assert half["truth"]["cells"].sum() == pytest.approx(full["truth"]["cells"].sum() / 2, rel=0.05)


def test_conditional_information_is_zero_for_a_redundant_column_and_positive_for_gene_signal():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(50), 200)
    old = rng.standard_normal((len(labels), 4)) + rng.standard_normal((50, 4))[labels]
    redundant = old @ rng.standard_normal((4, 1))
    assert abs(po.conditional_information(old, redundant + 1e-3 * rng.standard_normal(redundant.shape),
                                          labels)) < 0.02
    signal = rng.standard_normal((50, 1))[labels] + rng.standard_normal((len(labels), 1))
    assert po.conditional_information(old, signal, labels) > 0.3


def test_unknown_design_keys_and_scorers_are_refused(calibration):
    cal = po.load_calibration(calibration)
    with pytest.raises(ValueError, match="design keys"):
        po.resolve_design(cal, {"cells_per_guide": 3})
    with pytest.raises(ValueError, match="effect_type"):
        po.resolve_design(cal, {"effect_type": "shape"})
    screen = po.simulate_ops_screen(cal, {"n_genes": 10, "cells_per_gene": 20}, 0)
    with pytest.raises(ValueError, match="scorers"):
        po.score_ops_screen(screen, ["etest", "magic"])


def test_power_summary_and_minimum_detectable_effect():
    rows = pd.DataFrame({"status": "ok", "design_effect_size": [0.1, 0.1, 0.3, 0.3, 0.5, 0.5],
                         "seed": [0, 1] * 3, "etest_detected": [1, 1, 5, 7, 10, 10],
                         "etest_planted": [10] * 6, "etest_fpr_null_genes": [0.0] * 6,
                         "nt_pseudo_p05_pct": [5.0] * 6, "nt_guide_p05_pct": [5.0] * 6})
    summary = po.power_summary(rows)
    assert summary["power"].tolist() == [0.1, 0.6, 1.0]
    assert (summary["power_low"] <= summary["power"]).all() and (summary["power_high"] >= summary["power"]).all()
    mde = po.minimum_detectable_effect(summary)
    assert mde["mde"].iloc[0] == pytest.approx(0.3 + 0.2 * (0.8 - 0.6) / 0.4)


def test_the_power_sim_phase_reruns_identically_and_resumes(calibration, tmp_path):
    from spacr.ops_engine import run_ops

    settings = {"dst_root": str(tmp_path), "ops_power_sim": True, "ops_power_calibration_run": calibration,
                "ops_power_grid": {"n_genes": [20], "cells_per_gene": [40], "effect_size": [0.3, 1.0]},
                "ops_power_sim_seeds": [0, 1], "ops_power_effect_types": ["mean"],
                "ops_power_scorers": ["etest"], "ops_power_sim_workers": 1}
    first = run_ops(settings)["power_sim"]["outputs"]
    tables = {k: open(first[k], "rb").read() for k in ("rows", "summary", "mde")}
    assert len(pd.read_csv(first["rows"])) == 4
    second = run_ops(dict(settings), phases=["power_sim"])["power_sim"]["outputs"]
    assert {k: open(second[k], "rb").read() for k in tables} == tables
    assert len(pd.read_csv(first["designs"], sep="\t")) == 4
    other = tmp_path / "fresh"
    third = run_ops({**settings, "dst_root": str(other), "ops_power_sim_workers": 2})["power_sim"]["outputs"]
    assert {k: open(third[k], "rb").read() for k in tables} == tables


def test_the_switch_off_leaves_run_ops_on_its_image_path(tmp_path):
    from spacr.ops_engine import run_ops
    from spacr.ops_settings import ops_defaults

    assert ops_defaults({})["ops_power_sim"] is False
    with pytest.raises(ValueError, match="genotype_source"):
        run_ops({"dst_root": str(tmp_path), "ops_power_sim": False})
    with pytest.raises(ValueError, match="on its own"):
        run_ops({"dst_root": str(tmp_path)}, phases=["power_sim", "stitch"])


def test_clustering_recovers_planted_complexes(calibration):
    pytest.importorskip("phate")
    pytest.importorskip("leidenalg")
    cal = po.load_calibration(calibration)
    screen = po.simulate_ops_screen(cal, {"n_genes": 60, "cells_per_gene": 200, "effect_size": 1.5,
                                          "hit_fraction": 0.4, "complex_fraction": 1.0, "complex_noise": 0.1,
                                          "cluster_resolution": 1.0}, 0)
    row = po.score_ops_screen(screen, ["clustering"], 0)
    assert row["clustering_status"] == "ok"
    assert row["complex_recovery"] > row["complex_recovery_shuffled"]


def test_guide_variance_fools_the_pooled_null_but_not_the_guide_null(calibration):
    cal = dict(po.load_calibration(calibration))
    cal["between_guide_var"] = [0.05] * cal["dims"]
    design = {"n_genes": 60, "cells_per_gene": 200, "hit_fraction": 0.0, "nt_guides": 40}
    screen = po.simulate_ops_screen(cal, design, 2)
    pooled = po.score_ops_screen({**screen, "design": {**screen["design"], "etest_mode": "empirical_null"}},
                                 ["etest"], 2)
    guided = po.score_ops_screen({**screen, "design": {**screen["design"], "etest_mode": "guide_null"}},
                                 ["etest"], 2)
    assert pooled["etest_fpr_null_genes"] > 0.3
    assert guided["etest_fpr_null_genes"] <= 0.1


def test_the_information_null_permutes_whole_guides(calibration):
    cal = dict(po.load_calibration(calibration))
    cal["between_guide_var"] = [0.05] * cal["dims"]
    design = {"n_genes": 60, "cells_per_gene": 200, "hit_fraction": 0.0, "info_new_dims": 4,
              "info_permutations": 40}
    screen = po.simulate_ops_screen(cal, design, 4)
    by_guide = po.score_ops_screen(screen, ["information"], 4)
    by_cell = po.score_ops_screen({**screen, "design": {**screen["design"], "info_null": "cell"}},
                                  ["information"], 4)
    assert by_cell["info_detected"] and not by_guide["info_detected"]

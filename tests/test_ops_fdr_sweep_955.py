"""Item 955: the matched-FDR sweep of prefix decodes (own and external reads).

Every method's reads go through one matcher against the whitelist with decoys and against
shuffled whitelists. For each FDR target each method gets the configuration with the most
genotyped cells at pooled decoy FDR no higher than the target (ties keep the stricter one),
and every chosen row reports the decoy FDR and the barcode-shuffle call rate together, with
site-bootstrap intervals and the paired difference between methods. The phase is off unless
asked for.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from spacr import ops_prefix as P
from tests.test_ops_spacer_prefix_decode_920 import SPACERS, library_table

SETTINGS = {"ops_prefix_decoy_k": 4, "ops_prefix_decoy_seed": 1,
            "ops_fdr_sweep_targets": [1.0, 50.0], "ops_fdr_sweep_max_distance": [0],
            "ops_fdr_sweep_min_reads": [1], "ops_fdr_sweep_rules": ["maxcount"],
            "ops_fdr_sweep_bootstrap": 50, "ops_fdr_sweep_shuffle_seeds": [0, 1]}


def _library(tmp_path):
    path = tmp_path / "lib.csv"
    library_table(SPACERS).to_csv(path, index=False)
    return P.spacer_prefix_library(str(path), prefix_length=10, n_cycles=12)


def _keys(tmp_path):
    library = _library(tmp_path)
    joint = P.prefix_decoy_library(library, 4, 1)
    controls = {seed: set(P.shuffled_library(library, seed).keys) for seed in (0, 1)}
    every = controls[0] | controls[1]
    decoys = [k for k, v in joint.callable_keys.items() if v[2] == "decoy" and k not in every]
    real = [k for k in library.keys if k in library.callable_keys and k not in every]
    shuffled = {seed: sorted(k for k in keys if k not in joint.guides
                             and k not in controls[1 - seed]) for seed, keys in controls.items()}
    return library, real, decoys, shuffled


def _reads(rows):
    return pd.DataFrame(rows, columns=["site", "cell", "read"])


def test_known_decoy_and_shuffle_counts_give_the_expected_fdr_and_rate(tmp_path):
    library, real, decoys, shuffled = _keys(tmp_path)
    rows = [(1, c, real[c % len(real)]) for c in range(1, 41)]
    rows += [(2, c, real[c % len(real)]) for c in range(1, 41)]
    rows += [(1, 41, decoys[0]), (2, 41, decoys[1])]
    rows += [(1, 50 + i, key) for i, key in enumerate(shuffled[0][:3])]
    rows += [(2, 60 + i, key) for i, key in enumerate(shuffled[1][:3])]
    settings = {**SETTINGS, "ops_fdr_sweep_targets": [0.5, 50.0]}
    out = P.fdr_sweep(_reads(rows), {1: 100, 2: 100}, library, settings)
    entry = out["targets"][1]
    assert entry["label"] == "<= 50 %"
    pick = entry["methods"]["spacr"]
    assert pick["real"] == 80 and pick["decoy"] == 2 and pick["cells"] == 200
    assert pick["fdr"] == pytest.approx(100 / 4 * 2 / 80)
    assert pick["share"] == pytest.approx(40.0)
    assert pick["shuffle_per_seed_pct"] == {0: pytest.approx(1.5), 1: pytest.approx(1.5)}
    assert pick["shuffle_call_rate_pct"] == pytest.approx(1.5)
    assert out["targets"][0]["methods"]["spacr"] is None
    assert out["targets"][-1]["label"] == "lowest common"
    assert out["targets"][-1]["target_fdr_pct"] == pytest.approx(0.625)
    row = out["table"].iloc[0]
    assert {"decoy_fdr_pct", "shuffle_call_rate_pct", "genotyped_pct"} <= set(out["table"].columns)
    assert row["decoy_fdr_pct"] == pytest.approx(0.625) and row["shuffle_call_rate_pct"] == pytest.approx(1.5)


def test_ties_keep_the_stricter_configuration(tmp_path):
    library, real, _, _ = _keys(tmp_path)
    rows = [(1, c, real[c % len(real)]) for c in range(1, 21)]
    settings = {**SETTINGS, "ops_fdr_sweep_targets": [5.0], "ops_fdr_sweep_max_distance": [0, 1],
                "ops_fdr_sweep_rules": ["maxcount", "purity50", "purity100"]}
    out = P.fdr_sweep(_reads(rows), {1: 20}, library, settings)
    pick = out["targets"][0]["methods"]["spacr"]
    assert pick["real"] == 20
    assert (pick["max_d"], pick["min_reads"], pick["rule"]) == (0, 1, "purity100")


def test_the_rules_and_the_read_floor(tmp_path):
    library, real, _, _ = _keys(tmp_path)
    a, b = real[0], real[1]
    rows = [(1, 1, a), (1, 1, a), (1, 1, b), (1, 1, b),
            (1, 2, a), (1, 2, a), (1, 2, b),
            (1, 3, a)]
    settings = {**SETTINGS, "ops_fdr_sweep_targets": [100.0], "ops_fdr_sweep_min_reads": [1, 2],
                "ops_fdr_sweep_rules": list(P.FDR_SWEEP_RULES), "ops_fdr_sweep_quality_quantiles": [0.5]}
    out = P.fdr_sweep(_reads(rows), {1: 10}, library, settings)
    pooled = out["pooled"].set_index(["min_reads", "rule"])["real"]
    assert pooled[(1, "maxcount")] == 2 and pooled[(1, "purity50")] == 2
    assert pooled[(1, "purity100")] == 1 and pooled[(2, "maxcount")] == 1
    assert out["quality_cuts"]["spacr"] == [-np.inf]


def test_quality_cuts_come_from_the_method_reads(tmp_path):
    library, real, _, _ = _keys(tmp_path)
    frame = _reads([(1, c, real[c % len(real)]) for c in range(1, 11)])
    frame["quality"] = np.linspace(0.0, 0.9, 10)
    settings = {**SETTINGS, "ops_fdr_sweep_targets": [100.0], "ops_fdr_sweep_quality_quantiles": [0.5]}
    out = P.fdr_sweep({"spacr": frame, "other": frame.drop(columns="quality")}, {1: 10},
                      library, settings)
    assert out["quality_cuts"]["spacr"] == [-np.inf, pytest.approx(0.45)]
    assert out["quality_cuts"]["other"] == [-np.inf]
    cut = out["pooled"][(out["pooled"]["method"] == "spacr") & (out["pooled"]["quality_index"] == 1)]
    assert int(cut["real"].iloc[0]) == 5


def test_matched_difference_and_bootstrap_draw_order(tmp_path):
    library, real, decoys, _ = _keys(tmp_path)
    one, two = [], []
    for site in range(1, 7):
        one += [(site, c, real[c % len(real)]) for c in range(1, 11 + site)]
        two += [(site, c, real[c % len(real)]) for c in range(1, 6 + site)]
        two += [(site, 90, decoys[site])]
    cells = {s: 30 for s in range(1, 7)}
    settings = {**SETTINGS, "ops_fdr_sweep_targets": [100.0, 50.0], "ops_fdr_sweep_seed": 3}
    out = P.fdr_sweep({"a": _reads(one), "b": _reads(two)}, cells, library, settings)
    sites = out["sites"]
    rng = np.random.default_rng(3)
    first = rng.integers(0, 6, (50, 6))
    second = rng.integers(0, 6, (50, 6))
    real_a = np.array([10 + s for s in sites])
    real_b = np.array([5 + s for s in sites])
    for entry, draws in zip(out["targets"][:2], (first, second)):
        diff = 100 * real_a[draws].sum(1) / 180 - 100 * real_b[draws].sum(1) / 180
        got = entry["diffs"]["a - b"]
        assert got["diff_ci95"] == [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]
        assert got["diff_share_pts"] == pytest.approx(100 * (real_a.sum() - real_b.sum()) / 180)
    pick = out["targets"][0]["methods"]["b"]
    boot = 100 / 4 * 1.0 * np.ones(50) * 6 / real_b[first].sum(1)
    assert pick["fdr_ci95"] == pytest.approx([np.quantile(boot, 0.025), np.quantile(boot, 0.975)])
    row = out["table"][(out["table"]["target"] == "<= 100 %") & (out["table"]["method"] == "b")].iloc[0]
    assert row["paired_diff_vs"] == "a" and row["paired_diff_pts"] < 0


def test_selection_sites_choose_and_the_rest_report(tmp_path):
    library, real, decoys, _ = _keys(tmp_path)
    rows = [(s, c, real[c % len(real)]) for s in (1, 2, 3, 4) for c in range(1, 11)]
    rows += [(1, 20, decoys[0]), (3, 20, decoys[1]), (3, 21, decoys[2])]
    settings = {**SETTINGS, "ops_fdr_sweep_targets": [100.0]}
    out = P.fdr_sweep(_reads(rows), {s: 10 for s in (1, 2, 3, 4)}, library, settings,
                      selection_sites=[1, 2])
    assert out["selection_sites"] == [1, 2] and out["report_sites"] == [3, 4]
    pick = out["targets"][0]["methods"]["spacr"]
    assert pick["selection"]["decoy"] == 1 and pick["decoy"] == 2 and pick["real"] == 20
    with pytest.raises(ValueError, match="selection_sites"):
        P.fdr_sweep(_reads(rows), {s: 10 for s in (1, 2, 3, 4)}, library, settings,
                    selection_sites=[1, 2, 3, 4])


def test_invalid_settings_are_refused(tmp_path):
    library, real, _, _ = _keys(tmp_path)
    reads = _reads([(1, 1, real[0])])
    with pytest.raises(ValueError, match="decoy"):
        P.fdr_sweep(reads, {1: 1}, library, {**SETTINGS, "ops_prefix_decoy_k": 0})
    with pytest.raises(ValueError, match="rules"):
        P.fdr_sweep(reads, {1: 1}, library, {**SETTINGS, "ops_fdr_sweep_rules": "best"})
    with pytest.raises(ValueError, match="cell count"):
        P.fdr_sweep(reads, {2: 1}, library, SETTINGS)
    with pytest.raises(ValueError, match="lack"):
        P.fdr_sweep(reads.drop(columns="cell"), {1: 1}, library, SETTINGS)


def test_the_phase_reads_files_and_writes_its_outputs(tmp_path):
    from spacr.ops_engine import _CPG_PHASES, run_ops
    from spacr.ops_settings import OPS_DEFAULTS
    from spacr.settings import ALPHA_FEATURES

    assert _CPG_PHASES[:3] == ("cpg_index", "cpg_merged", "prefix_decode")
    assert set(ALPHA_FEATURES[955]["settings"]) == {k for k in OPS_DEFAULTS
                                                     if k.startswith("ops_fdr_sweep_")}
    assert OPS_DEFAULTS["ops_fdr_sweep_inputs"] is None
    library_table(SPACERS).to_csv(tmp_path / "lib.csv", index=False)
    library, real, decoys, _ = _keys(tmp_path)
    folder = tmp_path / "run" / "results" / "ops_prefix"
    folder.mkdir(parents=True)
    own = _reads([(1, c, real[c % len(real)]) for c in range(1, 9)] + [(1, 0, real[0])])
    own["quality"] = 1.0
    own.to_parquet(folder / "reads_P_W1.parquet")
    other = pd.DataFrame({"Parent_Cells": range(1, 6),
                          "Barcode_BarcodeCalled": [real[c % len(real)] for c in range(1, 6)]})
    other.to_csv(tmp_path / "foci_1.csv", index=False)
    pd.DataFrame({"site": [1], "Count_Cells": [10]}).to_csv(tmp_path / "count.csv", index=False)
    settings = {"dst_root": str(tmp_path / "run"), "ops_library": str(tmp_path / "lib.csv"),
                "ops_prefix_length": 10, "ops_sbs_cycles": 12, **SETTINGS,
                "ops_fdr_sweep_targets": [100.0],
                "ops_fdr_sweep_inputs": json.dumps({
                    "methods": {"spacr": str(folder / "reads_*.parquet"),
                                "periscope": {"reads": str(tmp_path / "foci_{site}.csv")}},
                    "cells": str(tmp_path / "count.csv"), "sites": [1]})}
    out = run_ops(settings, phases=["prefix_fdr_sweep"])["prefix_fdr_sweep"]
    assert out["cells_source"].endswith(":Count_Cells")
    saved = json.loads((folder / "fdr_sweep_result.json").read_text(encoding="utf-8"))
    picks = saved["targets"][0]["methods"]
    assert picks["spacr"]["real"] == 8 and picks["periscope"]["real"] == 5
    assert picks["spacr"]["cells"] == 10
    for name in ("table", "pooled", "grid"):
        assert (folder / f"fdr_sweep_{name}.csv").is_file()
    without = {key: value for key, value in settings.items() if key != "ops_fdr_sweep_inputs"}
    alone = run_ops(without, phases=["prefix_fdr_sweep"])["prefix_fdr_sweep"]
    assert alone["methods"] == ["spacr"] and alone["cells_source"] == "cells with any read"

"""Item 942: decoy FDR in the spacer-prefix decode and one shared tuner of detection and calling.

The shared tuner (spacr.ops_decode.tune_thresholds) reproduces item 939's choice when asked for
939's rule, falls back to the lowest FDR reached, maximises an F-score on request and marks the
genotyped-vs-FDR front. The prefix decode matches decoys jointly with the library, never
genotypes a decoy call, counts real and decoy cells exactly as its cell calls do, writes per-site
and per-well FDRs, tunes the read threshold, read quality and mismatches per well from decoys
only, and with every item-942 setting off writes what it wrote before. Settings saved before
schema 938 keep 939's auto mode.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from spacr import ops_decode as D
from spacr import ops_prefix as P
from spacr.tabular import read_table
from tests.test_ops_spacer_prefix_decode_920 import (SPACERS, library_table, synthetic_site,
                                                     write_cpg_site)


def _well(fields=16, seed=0):
    """Counts over (read thresholds, floors, 2): looser settings call more and leak more."""
    rng = np.random.default_rng(seed)
    nuclei = rng.integers(900, 1100, fields)
    counts = np.zeros((fields, 3, 2, 2), np.int64)
    for a in range(3):
        for b in range(2):
            real = rng.binomial(nuclei, 0.8 - 0.1 * a - 0.05 * b)
            counts[:, a, b, 0] = real
            counts[:, a, b, 1] = rng.poisson(real * (0.004 - 0.0015 * a - 0.0005 * b))
    return counts, nuclei, np.arange(fields) * 2 + 5


AXES = [("read_threshold", [100.0, 200.0, 300.0]), ("floor", [0.0, 0.1])]


def test_the_shared_tuner_reproduces_939_on_a_singleton_axis():
    counts, nuclei, sites = _well()
    two = counts[:, :, :, None, :]
    legacy = D.choose_well_thresholds(two[:, 1], nuclei, sites, [0.0, 0.1], [0.5],
                                      k=4, scale=1.268, fdr_max=0.2, seed=3)
    shared = D.tune_thresholds(two[:, 1:2], nuclei, sites,
                               [("read_threshold", [200.0]), ("single_read_quality", [0.0, 0.1]),
                                ("pixel_quality", [0.5])],
                               k=4, scale=1.268, fdr_max=0.2, seed=3, unreachable="strictest")
    assert shared["chosen"]["single_read_quality"] == legacy["single_read_quality"]
    for part in ("tune", "report", "all_fields"):
        assert shared[part] == legacy[part]
    assert shared["tune_sites"] == legacy["tune_sites"]
    assert [r["real"] for r in shared["grid"]] == [r["real"] for r in legacy["grid"]]


def test_joint_tuning_trades_detection_against_calling_under_the_fdr_target():
    counts, nuclei, sites = _well()
    got = D.tune_thresholds(counts, nuclei, sites, AXES, k=4, scale=1.0, fdr_max=0.25)
    assert got["auto_met"] and got["tune"]["fdr_ci95"][1] <= 0.25 or got["tune"]["decoy"] == 0
    passing = [r for r in got["grid"] if r["passes"]]
    assert got["tune"]["real"] == max(r["real"] for r in passing)
    chosen = [r for r in got["grid"] if r["chosen"]]
    assert len(chosen) == 1 and chosen[0]["read_threshold"] == got["chosen"]["read_threshold"]
    assert all(r["passes"] for r in chosen)
    assert set(got["tune_sites"]).isdisjoint(got["report_sites"])
    loose = D.tune_thresholds(counts, nuclei, sites, AXES, k=4, scale=1.0, fdr_max=100.0)
    assert loose["chosen"] == {"read_threshold": 100.0, "floor": 0.0}
    front = [r for r in got["grid"] if r["pareto"]]
    for row in got["grid"]:
        assert not any(o["fdr_bound"] <= row["fdr_bound"] and o["real"] > row["real"]
                       for o in got["grid"]) or not row["pareto"]
    assert front


def test_an_unreachable_target_uses_the_lowest_fdr_or_the_strictest_setting():
    counts, nuclei, sites = _well()
    low = D.tune_thresholds(counts, nuclei, sites, AXES, k=4, scale=1.0, fdr_max=1e-6)
    assert low["auto_met"] is False
    picked = [r for r in low["grid"] if r["chosen"]][0]
    assert picked["fdr_bound"] == min(r["fdr_bound"] for r in low["grid"]) == low["fdr_lowest"]
    strict = D.tune_thresholds(counts, nuclei, sites, AXES, k=4, scale=1.0, fdr_max=1e-6,
                               unreachable="strictest")
    assert strict["chosen"] == {"read_threshold": 300.0, "floor": 0.1}
    with pytest.raises(ValueError, match="objective"):
        D.tune_thresholds(counts, nuclei, sites, AXES, k=4, scale=1.0, objective="best")
    with pytest.raises(ValueError, match="unreachable"):
        D.tune_thresholds(counts, nuclei, sites, AXES, k=4, scale=1.0, unreachable="x")
    with pytest.raises(ValueError, match="axis lengths"):
        D.tune_thresholds(counts[:, :2], nuclei, sites, AXES, k=4, scale=1.0)


def test_the_f_score_objective_and_the_curve_files(tmp_path):
    counts, nuclei, sites = _well()
    got = D.tune_thresholds(counts, nuclei, sites, AXES, k=4, scale=1.0, objective="f_score")
    best = max(r["f_score"] for r in got["grid"])
    assert [r for r in got["grid"] if r["chosen"]][0]["f_score"] == best
    assert got["auto_met"]
    precise = D.tune_thresholds(counts, nuclei, sites, AXES, k=4, scale=1.0,
                                objective="f_score", f_beta=1e-3)
    assert precise["tune"]["fdr_per_100"] <= got["tune"]["fdr_per_100"]
    paths = D.write_tuning_curve(got, str(tmp_path), "pr_curve_W1", title="W1")
    table = pd.read_csv(paths["csv"])
    assert len(table) == 6 and table["chosen"].sum() == 1
    assert {"fdr_ci_low", "fdr_ci_high", "pareto", "f_score"} <= set(table.columns)
    assert os.path.getsize(paths["png"]) > 0
    empty = D.tune_thresholds(counts[:1], nuclei[:1], sites[:1], AXES, k=4, scale=1.0)
    assert empty["grid"] == [] and "png" not in D.write_tuning_curve(
        empty, str(tmp_path), "none")


def test_threshold_settings_carry_the_joint_grid_and_the_objective():
    rule = D.decode_rule(D.resolve_settings({"ops_decode_preset": "rule_a_s2"}))
    auto = D.threshold_settings({"ops_decode_threshold_mode": "auto"}, rule)
    assert auto["spot_thresholds"] == list(D.AUTO_SPOT_GRID)
    assert auto["objective"] == "recall_at_fdr" and auto["unreachable"] == "lowest_fdr"
    old = D.threshold_settings({"ops_decode_threshold_mode": "auto",
                                "ops_decode_auto_spot_grid": [],
                                "ops_decode_auto_unreachable": "strictest"}, rule)
    assert old["spot_thresholds"] == [] and old["unreachable"] == "strictest"
    assert D._spot_grid("400, 150", "k") == [150.0, 400.0]
    for bad in ({"ops_decode_auto_objective": "x"}, {"ops_decode_auto_f_beta": 0},
                {"ops_decode_auto_unreachable": "x"}, {"ops_decode_auto_spot_grid": [-1]}):
        with pytest.raises(ValueError):
            D.threshold_settings({"ops_decode_threshold_mode": "auto", **bad}, rule)


def test_settings_saved_before_938_keep_939_auto_mode():
    from spacr.settings import (ALPHA_FEATURES, DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION,
                                _fold_older_defaults)
    from spacr.ops_settings import OPS_DEFAULTS, OPS_TOOLTIPS

    assert SETTINGS_SCHEMA_VERSION >= 938
    assert DEFAULT_CHANGES[938] == {"ops_decode_auto_spot_grid": [],
                                    "ops_decode_auto_unreachable": "strictest"}
    older = _fold_older_defaults({}, 937)
    assert older["ops_decode_auto_spot_grid"][0] == []
    assert older["ops_decode_auto_unreachable"][0] == "strictest"
    assert "ops_decode_auto_spot_grid" not in _fold_older_defaults({}, SETTINGS_SCHEMA_VERSION)
    for key in ALPHA_FEATURES[942]["settings"]:
        assert key in OPS_DEFAULTS and OPS_TOOLTIPS[key].endswith("Alpha.")
    assert OPS_DEFAULTS["ops_decode_threshold_mode"] == "manual"
    assert OPS_DEFAULTS["ops_prefix_decoy_k"] == 0
    assert OPS_DEFAULTS["ops_prefix_threshold_tuning"] == "none"


def test_decoys_join_the_matcher_and_decoy_cells_are_never_genotyped():
    lib = P.spacer_prefix_library(library_table(SPACERS))
    combined = P.prefix_decoy_library(lib, 4, 1)
    decoys = [k for k, v in combined.callable_keys.items() if v[1] == P.DECOY_GENE]
    assert len(decoys) == combined.summary["decoys"] > 0 and not set(decoys) & set(lib.keys)
    assert P.prefix_decoy_library(lib, 4, 1).keys == combined.keys
    reads = P.match_prefix_reads([decoys[0], lib.keys[0]], combined, keyed=True)
    assert list(reads["gene"]) == [P.DECOY_GENE, combined.callable_keys[lib.keys[0]][1]]
    frame = pd.concat([reads.assign(cell=[1, 2]), reads.iloc[:1].assign(cell=1)])
    frame = frame.assign(plate="P", well="W1", site=1).reset_index(drop=True)
    cells = P._flag_decoy_cells(P.prefix_cell_calls(frame))
    one = cells.set_index("cell")
    assert one.loc[1, "decoy"] and one.loc[1, "gene"] is None and one.loc[1, "prefix"] == decoys[0]
    assert not one.loc[2, "decoy"] and one.loc[2, "gene"] == "GENE1"
    assert P._called_counts(frame, "spacr") == (1, 1)


@pytest.mark.parametrize("method", ["spacr", "recipe_simple"])
def test_tuning_counts_equal_the_cell_calls(method):
    rng = np.random.default_rng(1)
    lib = P.spacer_prefix_library(library_table(SPACERS))
    combined = P.prefix_decoy_library(lib, 2, 1)
    keys = list(combined.keys)
    reads = ["".join(rng.choice(list("ACGT"), 10)) if rng.random() < 0.3
             else keys[rng.integers(len(keys))] for _ in range(400)]
    matched = P.match_prefix_reads(reads, combined, keyed=True)
    frame = matched.assign(cell=rng.integers(0, 40, len(reads)), plate="P", well="W1", site=1,
                           quality=rng.random(len(reads)).astype(np.float32))
    for quality, distance in ((0.0, 1), (0.3, 1), (0.0, 0)):
        kept = P._at_setting(frame, quality, distance)
        cells = P._flag_decoy_cells(P.prefix_cell_calls(kept, method=method))
        hit = cells[cells["quality"].isin(("perfect", "great"))]
        expect = (int((~hit["decoy"]).sum()), int(hit["decoy"].sum()))
        assert P._called_counts(kept, method) == expect
    exact = P.match_prefix_reads(reads, combined, keyed=True, max_distance=0)
    restated = P._at_setting(matched.assign(cell=1, quality=1.0), 0.0, 0)
    assert list(restated["status"]) == list(exact["status"])


def test_spot_candidates_keep_detect_spots_unchanged():
    stack, _labels = synthetic_site()
    peaks, values = P.detect_spots(stack, threshold=10.0)
    every, strength, all_values = P._spot_candidates(stack, floor=2.0)
    keep = strength > 10.0
    assert np.array_equal(every[keep], peaks) and np.array_equal(all_values[keep], values)


def _run(tmp_path, name, **extra):
    stack, labels = synthetic_site()
    root = tmp_path / "data"
    if not root.exists():
        for site in (1, 2, 3, 4):
            write_cpg_site(root, site, stack, labels)
        library_table(SPACERS).to_csv(tmp_path / "Barcodes.csv", index=False)
    settings = {"ops_layout": "cpg_cropped", "ops_library_mode": "spacer_prefix",
                "cpg_root": str(root / "images_corrected_cropped"),
                "cpg_phenotype_channels": ["CorrDNA"], "ops_library": str(tmp_path / "Barcodes.csv"),
                "ops_prefix_masks": str(root / "masks" / "{plate}_{well}_{site}_cells.tiff"),
                "ops_read_threshold": 10.0, "n_workers": 1,
                "dst_root": str(tmp_path / name), **extra}
    return P.run_prefix_decode(settings)


def test_off_switch_writes_what_920_wrote(tmp_path):
    base = _run(tmp_path, "base")
    off = _run(tmp_path, "off", ops_prefix_decoy_k=0, ops_prefix_threshold_tuning="none",
               ops_prefix_min_read_quality=0)
    assert set(base) == set(off) == {"folder", "library", "sites", "totals", "cells",
                                    "called_cells", "seconds"}
    for name in sorted(os.listdir(base["folder"])):
        with open(os.path.join(base["folder"], name), "rb") as one, \
                open(os.path.join(off["folder"], name), "rb") as two:
            assert one.read() == two.read(), name


def test_decoy_fdr_and_per_well_tuning_outputs(tmp_path):
    decoy = _run(tmp_path, "decoy", ops_prefix_decoy_k=4)
    summary = read_table(os.path.join(decoy["folder"], "site_summary.csv"),
                         canonicalise=False, report=None)
    assert {"cells_in_labels", "cells_real", "cells_decoy", "fdr_pct"} <= set(summary.columns)
    assert list(summary["cells_real"]) == [3, 3, 3, 3] and summary["cells_in_labels"].eq(4).all()
    report = json.load(open(os.path.join(decoy["folder"], "fdr_CP9_W1.json")))
    assert report["assigned"] == 12 and report["nuclei"] == 16 and report["decoy_library_k"] == 4
    with pytest.raises(ValueError, match="ops_prefix_decoy_k"):
        _run(tmp_path, "bad", ops_prefix_threshold_tuning="per_well_decoy")
    grid = {"ops_read_threshold": [10, 5000], "ops_prefix_min_read_quality": [0, 0.5]}
    tuned = _run(tmp_path, "tuned", ops_prefix_decoy_k=4,
                 ops_prefix_threshold_tuning="per_well_decoy", ops_prefix_tuning_grid=grid,
                 ops_prefix_fdr_target_pct=50.0)
    choice = tuned["tuning"]["CP9_W1"]
    assert choice["chosen"]["ops_read_threshold"] == 10.0
    assert choice["chosen"]["ops_prefix_max_distance"] == 1.0
    folder = tuned["folder"]
    stem = os.path.join(folder, "threshold_tuning_CP9_W1")
    table = pd.read_csv(stem + ".csv")
    assert len(table) == 2 * 2 * 1 and table["chosen"].sum() == 1
    assert os.path.isfile(stem + ".png") and json.load(open(stem + ".json"))["well"] == "W1"
    assert tuned["called_cells"] == decoy["called_cells"]
    again = _run(tmp_path, "tuned_again", ops_prefix_decoy_k=4,
                 ops_prefix_threshold_tuning="per_well_decoy", ops_prefix_tuning_grid=grid,
                 ops_prefix_fdr_target_pct=50.0)
    for name in ("reads_CP9_W1.parquet", "cells_CP9_W1.csv", "site_summary.csv"):
        pd.testing.assert_frame_equal(
            read_table(os.path.join(folder, name), canonicalise=False, report=None),
            read_table(os.path.join(again["folder"], name), canonicalise=False, report=None))
    assert "seconds_per_well" in tuned and tuned["fdr"]["CP9_W1"]["assigned"] == 12

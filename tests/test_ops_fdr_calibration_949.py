"""Item 949: near-miss decoys and the barcode-shuffle calibration of the decoy FDR scale.

Near-miss decoys sit one edit from exactly one library prefix and from no other key, are
called as their own decoy class and never counted as real calls. The calibration raises the
FDR scale until the shuffle call rate is at most 1.5 times the FDR on the tuning fields, never
lowers it, and reads only decoy and shuffle calls. With both settings off the prefix decode
writes what item 942 wrote; with them on it reports the near-miss FDR, the shuffle call rate
and the calibrated scale. The OPS decode's auto mode takes the same calibration.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from spacr import ops_decode as D
from spacr import ops_prefix as P
from spacr.tabular import read_table
from tests.test_ops_autotune_942 import _run
from tests.test_ops_spacer_prefix_decode_920 import SPACERS, library_table


def _library(tmp_path):
    path = tmp_path / "lib.csv"
    library_table(SPACERS).to_csv(path, index=False)
    return P.spacer_prefix_library(str(path), prefix_length=10, n_cycles=12)


def _distance(a, b):
    return sum(x != y for x, y in zip(a, b))


def test_nearmiss_decoys_sit_one_edit_from_exactly_one_prefix(tmp_path):
    library = _library(tmp_path)
    far = P.prefix_decoy_library(library, 4, 1)
    near = P.prefix_nearmiss_decoys(far, 3, 1)
    added = [k for k, v in near.callable_keys.items() if v[2] == "nearmiss"]
    assert added and near.summary["nearmiss_decoys"] == len(added)
    assert near.summary["nearmiss_k"] == 3 and near.summary["decoys"] == far.summary["decoys"]
    others = set(far.keys)
    for key in added:
        assert key not in others
        assert sum(_distance(key, k) == 1 for k in others) == 1
        assert sum(_distance(key, k) == 1 for k in library.keys) == 1
        assert near.callable_keys[key][1] == P.NEARMISS_GENE
    assert all(_distance(a, b) > 1 for i, a in enumerate(added) for b in added[i + 1:])
    per_prefix = [sum(_distance(key, p) == 1 for key in added) for p in library.keys]
    assert max(per_prefix) <= 3
    again = P.prefix_nearmiss_decoys(far, 3, 1)
    assert again.keys == near.keys
    with pytest.raises(ValueError, match="nearmiss"):
        P.prefix_nearmiss_decoys(far, 0, 1)


def test_nearmiss_calls_are_decoys_and_never_real(tmp_path):
    library = _library(tmp_path)
    near = P.prefix_nearmiss_decoys(P.prefix_decoy_library(library, 4, 1), 2, 1)
    decoy = next(k for k, v in near.callable_keys.items() if v[2] == "nearmiss")
    real = next(k for k, v in near.callable_keys.items() if v[2] == "guide")
    matched = P.match_prefix_reads([decoy, decoy, real, real], near, keyed=True, max_distance=0)
    assert list(matched["status"]) == ["exact"] * 4
    matched = matched.assign(cell=[1, 1, 2, 2], quality=1.0)
    assert P._called_counts(matched, "spacr") == (1, 0)
    assert P._called_counts(matched, "spacr", nearmiss=True) == (1, 0, 1)
    cells = P.prefix_cell_calls(matched.assign(plate="P", well="W", site=1), method="spacr")
    flagged = P._flag_decoy_cells(cells, nearmiss=True)
    assert list(flagged["decoy"]) == [True, False]
    assert list(flagged["decoy_class"]) == ["nearmiss", None]
    assert flagged["gene"].isna().tolist() == [True, False]
    assert "decoy_class" not in P._flag_decoy_cells(cells).columns


def test_shuffle_call_rate_and_interval():
    rate = D.shuffle_call_rate([1, 3, 2, 2], [100, 100, 100, 100])
    assert rate["shuffle_called"] == 8 and rate["cells"] == 400
    assert rate["shuffle_call_rate_pct"] == pytest.approx(2.0)
    low, high = rate["shuffle_ci95"]
    assert low <= 2.0 <= high
    assert D.shuffle_call_rate([1], [50])["shuffle_ci95"] is None


def _calibration_well(fields=20, seed=0, leak=0.02):
    """(fields, 2 floors, 2): the looser floor calls more; shuffle calls a share of the cells."""
    rng = np.random.default_rng(seed)
    nuclei = rng.integers(900, 1100, fields)
    counts = np.zeros((fields, 2, 2), np.int64)
    shuffled = np.zeros((fields, 2))
    for b, (share, decoy, shuffle) in enumerate(((0.7, 0.002, leak), (0.6, 0.0005, leak / 4))):
        real = rng.binomial(nuclei, share)
        counts[:, b, 0] = real
        counts[:, b, 1] = rng.poisson(real * decoy)
        shuffled[:, b] = rng.poisson(nuclei * shuffle)
    return counts, shuffled, nuclei, np.arange(fields) + 1


AXES_949 = [("floor", [0.0, 0.2])]


@pytest.mark.parametrize("ci", [True, False])
def test_the_calibrated_scale_brings_the_shuffle_rate_under_one_and_a_half_fdr(ci):
    counts, shuffled, nuclei, sites = _calibration_well()
    choice = D.calibrate_fdr_scale(counts, shuffled, nuclei, sites, AXES_949, k=4, scale=1.0,
                                   ci=ci, fdr_max=5.0, seed=0)
    calibration = choice["calibration"]
    assert calibration["converged"] and choice["fdr_scale"] == calibration["scale"] > 1.0
    scales = [r["scale"] for r in calibration["rounds"]]
    assert scales == sorted(scales)
    tune = calibration["tune"]
    assert tune["passes"]
    assert tune["shuffle_call_rate_pct"] <= tune["shuffle_limit_pct"] * (1 + 1e-9)
    assert tune["fdr_per_100"] == choice["tune"]["fdr_per_100"]
    plain = D.tune_thresholds(counts, nuclei, sites, AXES_949, k=4, scale=choice["fdr_scale"],
                              ci=ci, fdr_max=5.0, seed=0)
    assert plain["chosen"] == choice["chosen"] and plain["report"] == choice["report"]


def test_no_shuffle_calls_keep_the_entered_scale():
    counts, shuffled, nuclei, sites = _calibration_well(leak=0.0)
    choice = D.calibrate_fdr_scale(counts, shuffled, nuclei, sites, AXES_949, k=4, scale=1.268,
                                   fdr_max=5.0, seed=0)
    plain = D.tune_thresholds(counts, nuclei, sites, AXES_949, k=4, scale=1.268, fdr_max=5.0,
                              seed=0)
    assert choice["fdr_scale"] == 1.268 and len(choice["calibration"]["rounds"]) == 1
    assert {k: v for k, v in choice.items() if k not in ("fdr_scale", "calibration")} == plain
    with pytest.raises(ValueError, match="shuffled"):
        D.calibrate_fdr_scale(counts, shuffled[:, :1], nuclei, sites, AXES_949, k=4, scale=1.0)


def test_the_needed_scale_judges_no_decoys_as_three():
    needed = D._needed_scale(np.array([1000, 1000]), np.array([0, 0]), np.array([3, 3]),
                             np.array([1000, 1000]), 4, ci=False)
    fdr1 = 100.0 / 4 * 3 / 2000
    assert needed == pytest.approx(0.3 / (1.5 * fdr1))


def test_settings_are_checked():
    # No decoy library: what a direct call fell back to before one OPS default
    # table (merge 1); OPS_DEFAULTS now gives ops_prefix_decoy_k 4.
    base = {"ops_library": "lib.csv", "ops_prefix_decoy_k": 0}
    chosen = P.prefix_settings(base)
    assert chosen["ops_prefix_nearmiss_decoy_k"] == 0
    assert chosen["ops_prefix_fdr_calibration"] == "none"
    with pytest.raises(ValueError, match="ops_prefix_decoy_k"):
        P.prefix_settings({**base, "ops_prefix_fdr_calibration": "shuffle"})
    with pytest.raises(ValueError, match="ops_prefix_fdr_calibration"):
        P.prefix_settings({**base, "ops_prefix_fdr_calibration": "reference"})
    with pytest.raises(ValueError, match="ops_prefix_nearmiss_decoy_k"):
        P.prefix_settings({**base, "ops_prefix_nearmiss_decoy_k": -1})
    rule = {"decoy_k": 4, "pixel_quality": 0.1, "pixels": True}
    auto = {"ops_decode_threshold_mode": "auto"}
    assert "fdr_calibration" not in D.threshold_settings(auto, rule)
    assert D.threshold_settings({**auto, "ops_decode_fdr_calibration": "shuffle"},
                                rule)["fdr_calibration"] == "shuffle"
    with pytest.raises(ValueError, match="ops_decode_fdr_calibration"):
        D.threshold_settings({**auto, "ops_decode_fdr_calibration": "x"}, rule)


def test_shuffled_barcodes_keep_lengths_and_composition():
    library = ["ACGTAC", "GGTTAA", "CCCAAG"]
    shuffled = D._shuffled_barcodes(library, 0)
    assert [sorted(s) for s in shuffled] == [sorted(b) for b in sorted(library)]
    assert shuffled == D._shuffled_barcodes(library, 0) != D._shuffled_barcodes(library, 1)
    first = D._shuffle_matchers(library, [0, 1])
    assert first is D._shuffle_matchers(library, [0, 1]) and len(first) == 2


def test_threshold_report_carries_the_calibration(tmp_path):
    counts, shuffled, nuclei, sites = _calibration_well()
    choice = D.calibrate_fdr_scale(counts[:, None, :, None], shuffled[:, None, :, None], nuclei,
                                   sites, [("read_threshold", [315.0])] + AXES_949
                                   + [("pixel_quality", [0.1])], k=4, scale=1.268,
                                   fdr_max=5.0, seed=0)
    choice = {"single_read_quality": choice["chosen"]["floor"], "pixel_quality": 0.1,
              "read_threshold": 315.0, **{k: v for k, v in choice.items()
                                          if k not in ("chosen", "index")}}
    db = tmp_path / "measurements.db"
    paths = D._write_threshold_report(str(db), "P", "A1", choice)
    row = read_table(paths["plate_table"], canonicalise=False, report=None).iloc[0]
    assert row["fdr_scale"] == pytest.approx(choice["fdr_scale"])
    assert row["shuffle_call_rate_pct"] == pytest.approx(
        choice["calibration"]["all_fields"]["shuffle_call_rate_pct"])


def test_off_switch_writes_what_942_wrote(tmp_path):
    for extra in ({}, {"ops_prefix_decoy_k": 4}):
        base = _run(tmp_path, f"base{len(extra)}", **extra)
        off = _run(tmp_path, f"off{len(extra)}", ops_prefix_nearmiss_decoy_k=0,
                   ops_prefix_fdr_calibration="none", **extra)
        assert set(base) == set(off)
        for name in sorted(os.listdir(base["folder"])):
            with open(os.path.join(base["folder"], name), "rb") as one, \
                    open(os.path.join(off["folder"], name), "rb") as two:
                assert one.read() == two.read(), name


def test_nearmiss_and_calibration_outputs(tmp_path):
    with pytest.raises(ValueError, match="ops_prefix_decoy_k"):
        _run(tmp_path, "bad", ops_prefix_nearmiss_decoy_k=2)
    on = _run(tmp_path, "on", ops_prefix_decoy_k=4, ops_prefix_nearmiss_decoy_k=2,
              ops_prefix_fdr_calibration="shuffle")
    summary = read_table(os.path.join(on["folder"], "site_summary.csv"),
                         canonicalise=False, report=None)
    assert {"cells_decoy_nearmiss", "fdr_nearmiss_pct", "shuffle_called",
            "shuffle_call_rate_pct", "fdr_scale"} <= set(summary.columns)
    assert list(summary["cells_real"]) == [3, 3, 3, 3]
    scale = on["calibration"]["CP9_W1"]["fdr_scale"]
    assert summary["fdr_scale"].eq(scale).all() and scale >= 1.0
    report = json.load(open(os.path.join(on["folder"], "fdr_CP9_W1.json")))
    assert report["fdr_scale"] == scale and report["fdr_calibration"] == "shuffle"
    for key in ("fdr_ci95", "assigned_decoy_nearmiss", "fdr_nearmiss_per_100",
                "shuffle_call_rate_pct", "shuffle_ci95", "shuffle_limit_pct", "calibration"):
        assert key in report
    cells = read_table(os.path.join(on["folder"], "cells_CP9_W1.csv"), canonicalise=False,
                       report=None)
    assert "decoy_class" in cells.columns
    classes = json.load(open(os.path.join(on["folder"], "quality_classes.json")))
    assert classes["settings"]["ops_prefix_fdr_calibration"] == "shuffle"
    grid = {"ops_read_threshold": [10, 5000], "ops_prefix_min_read_quality": [0, 0.5]}
    tuned = _run(tmp_path, "tuned", ops_prefix_decoy_k=4, ops_prefix_fdr_calibration="shuffle",
                 ops_prefix_threshold_tuning="per_well_decoy", ops_prefix_tuning_grid=grid,
                 ops_prefix_fdr_target_pct=50.0)
    well = tuned["tuning"]["CP9_W1"]
    assert well["fdr_scale"] == tuned["calibration"]["CP9_W1"]["fdr_scale"]
    assert well["calibration"]["rounds"]
    saved = json.load(open(os.path.join(tuned["folder"], "threshold_tuning_CP9_W1.json")))
    assert saved["fdr_scale"] == well["fdr_scale"]

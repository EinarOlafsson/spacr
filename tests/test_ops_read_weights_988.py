"""Item 988: neighbour-bleed discount and distance-aware read weights in the guide call (alpha, off).

The spacer-prefix decode gains a third cell rule, ``'weighted_majority'``:
each barcode's reads are summed with a weight (quality term times distance
term), a minority barcode that is a touching cell's top barcode can be
discounted, and the top barcode is called when it holds the share. The OPS
barcode decode's read vote gains the same distance term and discount. Every
new setting is off by default, and off writes what the decode wrote before.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from spacr import ops_prefix as P
from tests.test_ops_read_mode_952 import _prefix, _same_files


def _read(cell, prefix, n, *, gap=0.0, quality=0.5, gene=None, site=1):
    """``n`` exact reads of one barcode in one cell."""
    return [{"plate": "P", "well": "W", "site": site, "cell": cell, "y": 0, "x": 0,
             "status": "exact", "key": "ACGT", "distance": 0, "prefix": prefix,
             "guide": prefix, "gene": gene or f"G{prefix}", "level": "guide",
             "quality": quality, "nuc_gap_px": gap} for _ in range(n)]


def _reads():
    """Cells built so that each 988 lever changes one call.

    Cell 1: A 3 reads, B 2 reads, and B is the top barcode of touching cell 2.
    Cell 2: B 4 reads.
    Cell 3: C 2 reads near the nucleus, D 2 reads far from it (a count tie).
    Cell 4: E 5 reads, a clean cell.
    Cell 5: a decoy top barcode.
    """
    rows = (_read(1, "A", 3) + _read(1, "B", 2) + _read(2, "B", 4)
            + _read(3, "C", 2, gap=0.0) + _read(3, "D", 2, gap=20.0) + _read(4, "E", 5)
            + _read(5, "X", 3, gene=P.DECOY_GENE) + _read(5, "A", 1))
    rows.append({**rows[0], "cell": 0})
    return pd.DataFrame(rows)


def _neighbours():
    return pd.DataFrame({"plate": "P", "well": "W", "site": 1, "cell": [1, 2],
                         "neighbour": [2, 1]})


def test_the_weighted_majority_calls_by_weight_share():
    reads = _reads()
    calls = P.prefix_cell_calls(reads, method="weighted_majority", weighted_quality=False)
    assert list(calls["quality"]) == ["great", "perfect", "ambiguous", "perfect", "great"]
    assert calls.loc[0, "prefix"] == "A" and calls.loc[0, "weight_top"] == 3.0
    assert calls.loc[0, "weight_total"] == 5.0
    strict = P.prefix_cell_calls(reads, method="weighted_majority", weighted_quality=False,
                                 weighted_min_share=0.7)
    assert strict.loc[0, "quality"] == "ambiguous"
    assert set(P._WEIGHTED_COLUMNS) <= set(calls.columns)
    flagged = P._flag_decoy_cells(calls)
    assert bool(flagged.loc[4, "decoy"]) and flagged.loc[4, "gene"] is None


def test_the_neighbour_discount_rescues_bleed_from_a_touching_cell():
    reads = _reads()
    kw = dict(method="weighted_majority", weighted_quality=False, weighted_min_share=0.7)
    with pytest.raises(ValueError, match="touching cells"):
        P.prefix_cell_calls(reads, neighbour_discount=0.25, **kw)
    calls = P.prefix_cell_calls(reads, neighbour_discount=0.25, neighbours=_neighbours(), **kw)
    assert calls.loc[0, "quality"] == "great" and calls.loc[0, "prefix"] == "A"
    assert calls.loc[0, "weight_total"] == 3.5 and calls.loc[0, "bleed_discounted"] == 1
    assert calls.loc[1, "bleed_discounted"] == 0


def test_the_distance_term_breaks_a_count_tie():
    reads = _reads()
    calls = P.prefix_cell_calls(reads, method="weighted_majority", weighted_quality=False,
                                distance_weight="decay", decay_px=4.0)
    assert calls.loc[2, "quality"] == "great" and calls.loc[2, "prefix"] == "C"
    assert calls.loc[2, "d_weight_top"] == 2.0 and calls.loc[2, "g_median_top"] == 0.0
    near = P._read_weights(np.array([0.5]), np.array([20.0]), quality_term=False,
                           decay_px=4.0)
    assert near[0] == pytest.approx(max(np.exp(-5.0), 0.05))
    assert P._read_weights(np.array([0.5]), np.array([100.0]), quality_term=True,
                           decay_px=4.0)[0] == pytest.approx(0.55 * 0.05)
    without = reads.drop(columns="nuc_gap_px")
    with pytest.raises(ValueError, match="nuc_gap_px"):
        P.prefix_cell_calls(without, method="weighted_majority", distance_weight="decay")


def test_the_tuner_counter_matches_the_cell_table():
    rng = np.random.default_rng(3)
    rows = []
    for cell in range(1, 60):
        for barcode in rng.choice(list("ABCDEF"), rng.integers(1, 4), replace=False):
            rows += _read(cell, str(barcode), int(rng.integers(1, 5)),
                          gap=float(rng.uniform(0, 30)), quality=float(rng.uniform(0, 1)),
                          gene=P.DECOY_GENE if barcode == "F" else None)
    reads = pd.DataFrame(rows)
    pairs = np.array([[c, c + 1] for c in range(1, 59)] + [[c + 1, c] for c in range(1, 59)])
    table = P._neighbour_table(pairs, "P", "W", 1)
    decay, discount, quality = [4.0, 16.0, 0.0], [0.0, 0.5, 1.0], [0.0, 1.0]
    grid = P._weighted_counts(reads, neighbours=pairs, mode="decay", decay=decay,
                              discount=discount, quality=quality, share=0.6,
                              interior_px=0.0, min_weight=0.05)
    for a, length in enumerate(decay):
        for b, value in enumerate(discount):
            for c, q_on in enumerate(quality):
                calls = P._flag_decoy_cells(P.prefix_cell_calls(
                    reads, method="weighted_majority",
                    distance_weight="decay" if length else "off", decay_px=length or 8.0,
                    weighted_quality=bool(q_on), neighbour_discount=value, neighbours=table))
                hit = calls[calls["quality"].isin(("perfect", "great"))]
                assert tuple(grid[a, b, c]) == (int((~hit["decoy"]).sum()),
                                                int(hit["decoy"].sum()))


def test_off_changes_no_call():
    reads = _reads()
    for method in ("spacr", "recipe_simple"):
        base = P.prefix_cell_calls(reads, method=method)
        pd.testing.assert_frame_equal(base, P.prefix_cell_calls(
            reads, method=method, distance_weight="off", neighbour_discount=1.0,
            neighbours=_neighbours()))
    with pytest.raises(ValueError, match="alternatives"):
        P.prefix_cell_calls(reads, method="weighted_majority", minority_min_reads=3)


def test_geometry_of_touching_cells_and_nucleus_distance():
    cells = np.zeros((9, 12), np.int32)
    cells[1:8, 1:6], cells[1:8, 6:11] = 1, 2
    nuclei = np.zeros_like(cells)
    nuclei[3:6, 2:5] = 1
    pairs = P._touching_cells(cells)
    assert pairs.tolist() == [[1, 2], [2, 1]]
    gap, fallback = P._read_nucleus_gaps(np.array([4, 4, 4, 0]), np.array([3, 5, 9, 0]),
                                         np.array([1, 1, 2, 0]), cells, nuclei)
    assert gap[0] == 0.0 and gap[1] == 1.0 and not fallback[:2].any()
    assert fallback[2] and gap[2] == pytest.approx(max(np.hypot(4 - 4, 9 - 8) - np.sqrt(9 / np.pi), 0))
    assert np.isnan(gap[3]) and not fallback[3]
    with pytest.raises(ValueError, match="nucleus labels"):
        P._read_nucleus_gaps(np.array([1]), np.array([1]), np.array([1]), cells, nuclei[:5])


def test_the_settings_are_alpha_off_and_checked():
    from spacr.ops_settings import OPS_DEFAULTS, OPS_TOOLTIPS
    from spacr.settings import ALPHA_FEATURES

    keys = ALPHA_FEATURES[988]["settings"]
    assert len(keys) == 14 and all("Alpha." in OPS_TOOLTIPS[k] for k in keys)
    assert OPS_DEFAULTS["ops_prefix_distance_weight"] == "off"
    assert OPS_DEFAULTS["ops_read_distance_weight"] == "off"
    assert OPS_DEFAULTS["ops_read_neighbour_discount"] == 1.0
    assert OPS_DEFAULTS["ops_keep_multi_guide_cells"] is False
    base = {"ops_library": "x.csv"}
    chosen = P.prefix_settings(base)
    assert chosen["ops_prefix_weighted_min_share"] == 0.6
    assert chosen["ops_prefix_distance_decay_px"] == 8.0
    for bad, match in (({"ops_prefix_distance_weight": "decay"}, "weighted majority"),
                       ({"ops_read_neighbour_discount": 0.5}, "weighted majority"),
                       ({"ops_prefix_quality_method": "weighted_majority",
                         "ops_prefix_minority_min_reads": 3}, "alternatives"),
                       ({"ops_prefix_quality_method": "weighted_majority",
                         "ops_prefix_distance_weight": "near"}, "ops_prefix_distance_weight"),
                       ({"ops_prefix_quality_method": "weighted_majority",
                         "ops_read_neighbour_discount": 2}, "ops_read_neighbour_discount")):
        with pytest.raises(ValueError, match=match):
            P.prefix_settings({**base, **bad})


def test_the_weighted_tuning_grid_keeps_off_in_every_weight_axis():
    chosen = P.prefix_settings({"ops_library": "x.csv",
                                "ops_prefix_quality_method": "weighted_majority",
                                "ops_prefix_threshold_tuning": "per_well_decoy",
                                "ops_prefix_decoy_k": 4})
    axes = dict(P._tuning_axes(chosen))
    assert axes["ops_read_threshold"][:3] == [20.0, 30.0, 40.0]
    assert axes["ops_prefix_distance_decay_px"] == [4.0, 8.0, 16.0, 32.0, 0.0]
    assert axes["ops_read_neighbour_discount"] == [0.0, 0.25, 0.5, 1.0]
    assert axes["ops_prefix_weighted_quality"] == [0.0, 1.0]
    given = dict(P._tuning_axes({**chosen, "ops_prefix_tuning_grid": {
        "ops_prefix_distance_decay_px": [8], "ops_read_neighbour_discount": [0.25]}}))
    assert given["ops_prefix_distance_decay_px"] == [8.0, 0.0]
    assert given["ops_read_neighbour_discount"] == [0.25, 1.0]
    assert given["ops_prefix_weighted_quality"] == [1.0]
    plain = P.prefix_settings({"ops_library": "x.csv"})
    assert dict(P._tuning_axes(plain))["ops_read_threshold"][0] == 40.0
    with pytest.raises(ValueError, match="ops_prefix_tuning_grid"):
        P._tuning_axes({**plain, "ops_prefix_tuning_grid": {"ops_read_neighbour_discount": [0]}})
    best = {"ops_prefix_distance_decay_px": 0.0, "ops_read_neighbour_discount": 0.25,
            "ops_prefix_weighted_quality": 1.0}
    args = P._weight_arguments(chosen, best)
    assert args["distance_weight"] == "off" and args["neighbour_discount"] == 0.25
    only = {**chosen, "ops_prefix_distance_weight": "distance_only"}
    assert P._weight_arguments(only, best)["weighted_quality"] is False


def test_guide_multiplicity_and_multi_guide_cells():
    rows = (_read(1, "A", 3) + _read(1, "B", 2) + _read(2, "B", 4) + _read(3, "C", 2)
            + _read(3, "D", 2) + _read(4, "E", 1) + _read(4, "X", 1, gene=P.DECOY_GENE))
    reads = pd.DataFrame(rows)
    pairs = np.array([[1, 2], [2, 1]])
    counts, multi = P._guide_multiplicity(reads, pairs, 2, ["plate", "well", "site", "cell"])
    assert counts["all_1"] == 2 and counts["all_2"] == 2 and counts["all_decoy"] == 1
    assert counts["min_1"] == 1 and counts["min_2"] == 2 and counts["min_decoy"] == 0
    assert counts["bleed_tested"] == 1 and counts["bleed_neighbour"] == 1
    assert multi["cell"].tolist() == [3] and multi.loc[0, "guides"] == "C|D"
    table = P._multiplicity_table(counts, 10, 2)
    assert table.loc[0, "pct_0"] == 60.0 and table.loc[0, "bleed_pct"] == 100.0


def test_the_funk_vote_takes_the_distance_term_and_the_discount():
    from spacr.ops_sbs import LibraryMatcher, _disc_neighbours, vote_mapped_reads

    matcher = LibraryMatcher(["AAAA", "CCCC"])
    objects = np.array([1, 1, 1, 1, 1, 2, 2, 2])
    barcodes = ["AAAA"] * 3 + ["CCCC"] * 2 + ["CCCC"] * 3
    quality = np.full(8, 0.5)
    data = np.zeros((8, 4, 4))
    margin = np.ones((8, 4))
    base = vote_mapped_reads(objects, barcodes, quality, data, margin, matcher,
                             min_fraction=0.7)
    same = vote_mapped_reads(objects, barcodes, quality, data, margin, matcher,
                             min_fraction=0.7, gaps=np.zeros(8), neighbour_discount=1.0)
    assert base == same and 1 not in base
    found = _disc_neighbours(np.array([1, 2, 3]), np.array([[0, 0], [0, 12], [0, 80]]),
                             np.array([np.pi * 25] * 3), 10.0)
    assert found == {1: {2}, 2: {1}}
    rescued = vote_mapped_reads(objects, barcodes, quality, data, margin, matcher,
                                min_fraction=0.7, neighbours=found, neighbour_discount=0.25)
    assert rescued[1]["guide"] == "AAAA" and rescued[1]["vote_total"] == 3.5
    gaps = np.array([0, 0, 0, 30, 30, 0, 0, 0], float)
    near = vote_mapped_reads(objects, barcodes, quality, data, margin, matcher,
                             min_fraction=0.7, gaps=gaps, distance_weight="decay", decay_px=4.0)
    assert near[1]["guide"] == "AAAA"
    with pytest.raises(ValueError, match="distance"):
        vote_mapped_reads(objects, barcodes, quality, data, margin, matcher,
                          distance_weight="decay")


def test_the_decode_rule_carries_the_weighting_only_when_on():
    from spacr.ops_decode import decode_rule, resolve_settings

    rule = decode_rule(resolve_settings({"ops_decode_preset": "rule_a_s2"}))
    assert "read_weight" not in rule
    on = decode_rule(resolve_settings({"ops_decode_preset": "rule_a_s2",
                                       "ops_read_neighbour_discount": 0.25}))
    assert on["read_weight"]["neighbour_discount"] == 0.25
    assert on["read_weight"]["distance_weight"] == "off"
    with pytest.raises(ValueError, match="decode rules"):
        decode_rule({"ops_read_distance_weight": "decay"})
    with pytest.raises(ValueError, match="ops_read_distance_weight"):
        decode_rule({"ops_decode_preset": "rule_a_s2", "ops_read_distance_weight": "near"})


@pytest.mark.parametrize("extra", [{}, {"ops_prefix_decoy_k": 4,
                                        "ops_prefix_quality_method": "recipe_simple"}])
def test_off_writes_what_the_decode_wrote(tmp_path, extra):
    base = _prefix(tmp_path, "base", **extra)
    off = _prefix(tmp_path, "off", ops_prefix_distance_weight="off",
                  ops_read_neighbour_discount=1.0, ops_keep_multi_guide_cells=False,
                  ops_prefix_weighted_quality=True, ops_prefix_distance_decay_px=8.0, **extra)
    _same_files(base["folder"], off["folder"])
    saved = json.load(open(os.path.join(off["folder"], "quality_classes.json")))
    assert not set(P._ITEM_988_KEYS) & set(saved["settings"])
    assert "multi_guide" not in off


def test_the_weighted_decode_runs_tuned_with_distances_and_the_multi_guide_table(tmp_path):
    root = tmp_path / "data"
    _prefix(tmp_path, "seed")
    masks = str(root / "masks" / "{plate}_{well}_{site}_cells.tiff")
    tuned = _prefix(tmp_path, "tuned", ops_prefix_decoy_k=4,
                    ops_prefix_quality_method="weighted_majority",
                    ops_prefix_nuclei_masks=masks, ops_keep_multi_guide_cells=True,
                    ops_prefix_threshold_tuning="per_well_decoy", ops_read_mode="log",
                    ops_prefix_fdr_target_pct=50.0,
                    ops_prefix_tuning_grid={"ops_read_threshold": [10],
                                            "ops_prefix_distance_decay_px": [8],
                                            "ops_read_neighbour_discount": [0.25]})
    folder = tuned["folder"]
    table = pd.read_csv(os.path.join(folder, "threshold_tuning_CP9_W1.csv"))
    assert len(table) == 1 * 1 * 1 * 2 * 2 * 1
    assert {"ops_prefix_distance_decay_px", "ops_read_neighbour_discount"} <= set(table.columns)
    reads = pd.read_parquet(os.path.join(folder, "reads_CP9_W1.parquet"))
    inside = reads[reads["cell"] > 0]
    assert (inside["nuc_gap_px"] == 0).all() and not inside["nuc_gap_fallback"].any()
    cells = pd.read_csv(os.path.join(folder, "cells_CP9_W1.csv"))
    assert set(P._WEIGHTED_COLUMNS) <= set(cells.columns)
    multi = pd.read_csv(os.path.join(folder, "guide_multiplicity_CP9_W1.csv"))
    assert len(multi) == 2 and tuned["multi_guide"]["CP9_W1"]["multi_guide_cells"] >= 0
    saved = json.load(open(os.path.join(folder, "quality_classes.json")))
    assert saved["method"] == "weighted_majority" and saved["classes"] == P.WEIGHTED_CLASSES
    assert saved["settings"]["ops_keep_multi_guide_cells"] is True
    fixed = _prefix(tmp_path, "fixed", ops_prefix_decoy_k=4,
                    ops_prefix_quality_method="weighted_majority",
                    ops_prefix_distance_weight="decay", ops_prefix_nuclei_masks=masks,
                    ops_read_neighbour_discount=0.5)
    assert fixed["called_cells"] >= 1
    with pytest.raises(ValueError, match="ops_prefix_nuclei_masks"):
        _prefix(tmp_path, "nonuclei", ops_prefix_decoy_k=4,
                ops_prefix_quality_method="weighted_majority",
                ops_prefix_distance_weight="decay")

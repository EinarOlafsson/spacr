"""Stricter genotype calls by default (item 1023, settings schema 966).

A nucleus is called only when its top guide holds at least
``ops_min_top_share`` of its weighted read votes, by its reads or by the
pixel fallback; reads count within ``ops_footprint`` (3 px) of a nucleus;
the neighbour discount keeps its own reach (``ops_read_neighbour_reach``,
10 px). Files stamped before 966 keep the old values. Every well report
carries the split-read concordance: the agreement of the call of a nucleus with
an independent read of it in an overlapping tile.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from spacr import ops_decode
from spacr.settings import DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION, _fold_older_defaults

GUIDES = ["GTACGT", "CATGCA", "TGCATG"]
A, B, C = GUIDES


def _block(codes):
    values = np.full((len(codes), 6, 4), 10.0, np.float32)
    for i, code in enumerate(codes):
        for c, letter in enumerate(code):
            values[i, c, "GTAC".index(letter)] = 1000.0
    return values


def _field():
    """Reads and pixels of five owned nuclei and one another field owns.

    21: nine reads A, one B (share 0.9). 22: four A, one B (0.8). 23: four
    A, one B, three pixels A. 24: one read A. 25: no read, three pixels C.
    31, owned elsewhere: two reads B strictly inside it, one outside.
    """
    reads = ([(21, A)] * 9 + [(21, B)] + [(22, A)] * 4 + [(22, B)]
             + [(23, A)] * 4 + [(23, B)] + [(24, A)])
    n = len(reads)
    gaps = [[0.0, 30.0]] * n
    owned = [True] * n
    reads += [(31, B), (31, B), (31, C)]
    gaps += [[-2.0, 30.0], [-1.0, 30.0], [1.0, 30.0]]
    owned += [False] * 3
    pixels = [(23, A)] * 3 + [(25, C)] * 3
    m = len(pixels)
    return {
        "strength": np.full(len(reads), 500.0), "values": _block([c for _, c in reads]),
        "gap": np.array(gaps, np.float32),
        "id0": np.array([o for o, _ in reads]), "owned0": np.array(owned),
        "owned_ids": np.array([21, 22, 23, 24, 25]),
        "pix_values": _block([c for _, c in pixels]),
        "pix_gap": np.tile(np.array([[1, 9]], np.float16), (m, 1)),
        "pix_id0": np.array([o for o, _ in pixels]), "pix_owned0": np.ones(m, bool),
    }


def _rule(share=None, gate=False):
    settings = {"ops_decode_pixel_fallback": True, "ops_pixel_read_gate": gate,
                "ops_read_neighbour_discount": 1.0,
                "ops_pixel_ownership": "nearest_nucleus"}
    if share is not None:
        settings["ops_min_top_share"] = share
    return ops_decode.decode_rule(settings)


def _calls(share=None, gate=False, single_read_quality=0.0, field=None):
    matcher, decoys = ops_decode.matcher_for(GUIDES, ())
    return ops_decode.field_calls(field or _field(), _rule(share, gate), GUIDES, matcher,
                                  decoys, threshold=315.0, footprint=3.0, max_distance=0,
                                  min_fraction=0.6, single_read_quality=single_read_quality,
                                  crosstalk="none")


def test_the_share_keeps_only_nuclei_whose_top_guide_holds_it():
    old = _calls(0.0)
    assert {o: h["source"] for o, h in old["calls"].items()} == {
        21: "primary", 22: "primary", 23: "primary", 24: "primary", 25: "pixel_fallback"}
    new = _calls(0.9)
    assert {o: h["source"] for o, h in new["calls"].items()} == {
        21: "primary", 24: "primary", 25: "pixel_fallback"}
    assert new["calls"][21]["fraction"] == pytest.approx(0.9)
    assert _rule()["min_top_share"] == 0.9


@pytest.mark.parametrize("gate", [False, True])
def test_a_nucleus_below_the_share_gets_no_pixel_call(gate):
    lone = _calls(0.9, gate=gate, single_read_quality=2.0)
    assert 23 not in lone["calls"] and 22 not in lone["calls"]
    assert lone["calls"][25]["source"] == "pixel_fallback"


def test_the_share_is_checked():
    with pytest.raises(ValueError, match="ops_min_top_share"):
        _rule(1.5)


@pytest.mark.parametrize("share", [0.0, 0.9])
@pytest.mark.parametrize("gate", [False, True])
def test_the_auto_candidates_and_the_controls_see_the_same_rule(share, gate):
    matcher, decoys = ops_decode.matcher_for(GUIDES, ())
    rule = _rule(share, gate)
    candidates = ops_decode._joint_field_candidates(
        _field(), rule, GUIDES, matcher, decoys, spot_thresholds=[315.0],
        floors=[0.0, 2.0], pixel_qualities=[rule["pixel_quality"]],
        footprint=3.0, max_distance=0, min_fraction=0.6, crosstalk="none")
    for i, floor in enumerate((0.0, 2.0)):
        direct = _calls(share, gate, single_read_quality=floor)
        rebuilt = ops_decode._candidate_calls(candidates["by_spot"][0], floor,
                                              rule["pixel_quality"], decoys)
        assert {o: h["source"] for o, h in rebuilt["calls"].items()} == {
            o: h["source"] for o, h in direct["calls"].items()}
        assert candidates["counts"][0, i, 0].tolist() == [direct["assigned_real"],
                                                       direct["assigned_decoy"]]
    shuffled = ops_decode._field_shuffle_counts(
        _field(), rule, GUIDES, seeds=[1], spot_thresholds=[315.0], floors=[0.0],
        pixel_qualities=[rule["pixel_quality"]], footprint=3.0, max_distance=0,
        min_fraction=0.6, crosstalk="none")
    assert shuffled.shape == (1, 1, 1)


def test_the_new_defaults_and_old_files():
    from spacr.ops_settings import OPS_DEFAULTS, OPS_TOOLTIPS

    assert SETTINGS_SCHEMA_VERSION >= 966
    assert DEFAULT_CHANGES[966] == {"ops_min_top_share": 0.0, "ops_footprint": 10.0,
                                    "ops_read_neighbour_reach": None}
    assert OPS_DEFAULTS["ops_min_top_share"] == 0.9
    assert OPS_DEFAULTS["ops_footprint"] == 3.0
    assert OPS_DEFAULTS["ops_read_neighbour_reach"] == 10.0
    assert ops_decode.RULE_A_S2["ops_footprint"] == 3.0
    assert ops_decode.RULE_A_S2["ops_min_top_share"] == 0.9
    for key in DEFAULT_CHANGES[966]:
        assert "966" in OPS_TOOLTIPS[key]
    known = {"plate", *DEFAULT_CHANGES[966]}
    old_file = {"plate": "P1"}
    applied = _fold_older_defaults(old_file, 965, known=known)
    assert applied == {key: (value, 966) for key, value in DEFAULT_CHANGES[966].items()}
    assert _fold_older_defaults({"plate": "P1"}, 966, known=known) == {}
    resolved = ops_decode.resolve_settings({**OPS_DEFAULTS, **old_file,
                                            "ops_decode_preset": "rule_a_s2"})
    assert resolved["ops_footprint"] == 10.0 and resolved["ops_min_top_share"] == 0.0
    rule = ops_decode.decode_rule(resolved)
    assert rule["min_top_share"] == 0.0 and rule["neighbour_reach"] is None
    new = ops_decode.decode_rule(ops_decode.resolve_settings(
        {**OPS_DEFAULTS, "ops_decode_preset": "rule_a_s2"}))
    assert new["min_top_share"] == 0.9 and new["neighbour_reach"] == 10.0


def test_the_neighbour_reach_is_apart_from_the_footprint():
    assert ops_decode._neighbour_reach({"neighbour_reach": 10.0, "footprint": 3.0}, 3.0) == 10.0
    assert ops_decode._neighbour_reach({"neighbour_reach": None, "footprint": 3.0}, 7.0) == 3.0
    assert ops_decode._neighbour_reach({}, 7.0) == 7.0
    radius = np.sqrt(100.0 / np.pi)
    evidence = {"nuc_ids": np.array([1, 2]), "nuc_areas": np.array([100.0, 100.0]),
                "nuc_centroids": np.array([[0.0, 0.0], [0.0, 2 * radius + 6.0]])}
    rule = {"read_weight": {"neighbour_discount": 0.5}}
    far = ops_decode._rule_with_neighbours({**rule, "neighbour_reach": 10.0, "footprint": 3.0},
                                           evidence, 3.0)
    near = ops_decode._rule_with_neighbours({**rule, "neighbour_reach": None, "footprint": 3.0},
                                            evidence, 3.0)
    assert far["neighbours"] == {1: {2}, 2: {1}}
    assert near["neighbours"] == {}
    with pytest.raises(ValueError, match="ops_read_neighbour_reach"):
        ops_decode.decode_rule({"ops_read_neighbour_reach": -1})
    assert ops_decode.decode_rule({"ops_read_neighbour_reach": ""})["neighbour_reach"] is None


def test_a_field_keeps_its_reads_of_nuclei_other_fields_own():
    out = _calls(0.9)
    assert out["split"] == {31: (2, 1, B)}


def test_split_read_concordance():
    calls = pd.DataFrame({"object_id": [1, 2, 3, 4, 5], "site": [0, 0, 1, 1, 2],
                          "mapped_guide": [A, B, A, A, C]})
    split = pd.DataFrame({"object_id": [1, 2, 3, 4, 4, 5, 6],
                          "n": [1, 2, 3, 1, 1, 2, 5], "nguides": [1, 1, 1, 1, 1, 2, 1],
                          "guide": [A, A, A, A, A, A, A], "other_site": [1, 1, 0, 0, 2, 0, 0]})
    out = ops_decode.split_read_concordance(calls, split, [0, 1, 2], reps=200, seed=3)
    one = out["min_reads_1"]
    assert (one["nuclei"], one["agree"]) == (3, 2)
    assert one["pct"] == pytest.approx(100 * 2 / 3)
    assert one["ci95"][0] <= one["pct"] <= one["ci95"][1]
    two = out["min_reads_2"]
    assert (two["nuclei"], two["agree"]) == (2, 1)
    assert out["seed"] == 3 and out["reps"] == 200 and out["tiles"] == 3
    again = ops_decode.split_read_concordance(calls, split, [0, 1, 2], reps=200, seed=3)
    assert again == out
    report = ops_decode.split_read_concordance(calls, split, [0], reps=50)
    assert report["min_reads_1"]["nuclei"] == 2
    empty = ops_decode.split_read_concordance(calls.iloc[:0], split.iloc[:0], [0])
    assert empty["min_reads_1"] == {"nuclei": 0, "agree": 0, "pct": None, "ci95": [None, None]}
    assert ops_decode.SPLIT_READ_SEED == 20261125 and ops_decode.SPLIT_READ_REPS == 2000


def test_the_well_concordance_scores_the_report_fields():
    def field(site, calls, split):
        return {"site": site, "rule_a": {"nuclei": 5, "split": split, "calls": {
            o: {"guide": g, "decoy": False} for o, g in calls.items()}}}

    decoded = [field(0, {1: A}, {2: (1, 1, A)}), field(1, {2: A}, {1: (1, 1, B)})]
    whole = ops_decode._well_split_concordance(decoded, {})
    assert whole["tile_set"] == "all" and whole["min_reads_1"]["agree"] == 1
    half = ops_decode._well_split_concordance(
        decoded, {"threshold_choice": {"tune_sites": [0]}})
    assert half["tile_set"] == "report" and half["tiles"] == 1
    assert (half["min_reads_1"]["nuclei"], half["min_reads_1"]["agree"]) == (1, 1)


def test_decode_cached_well_replays_old_and_new_settings(tmp_path):
    folder = tmp_path / "decode_cache"
    folder.mkdir()
    field = _field()
    for site in (0, 1):
        np.savez(folder / f"site{site:04d}.npz", **field)
    settings = {"ops_decode_preset": "rule_a_s2", "ops_decode_library_calibration": False,
                "ops_decoy_library_k": 0, "ops_crosstalk": "none",
                "ops_read_neighbour_discount": 1.0}
    kwargs = dict(read_threshold=315.0, single_read_quality=0.0, pixel_quality=0.1)
    old = ops_decode.decode_cached_well(str(folder), {**settings, "spacr_settings_version": 965},
                                        GUIDES, **kwargs)
    new = ops_decode.decode_cached_well(str(folder), {**settings, "spacr_settings_version": 966},
                                        GUIDES, tune_sites=[1], **kwargs)
    assert (old["rule"]["footprint"], old["rule"]["min_top_share"]) == (10.0, 0.0)
    assert (new["rule"]["footprint"], new["rule"]["min_top_share"]) == (3.0, 0.9)
    assert [r["site"] for r in new["fields"]] == [0, 1]
    assert set(old["fields"][0]["rule_a"]["calls"]) == {21, 22, 23, 24, 25}
    assert set(new["fields"][0]["rule_a"]["calls"]) == {21, 24, 25}
    assert new["assigned_real"] == 6 and new["nuclei"] == 10
    assert new["split_read_concordance"]["tile_set"] == "report"
    json.dumps(new["split_read_concordance"])

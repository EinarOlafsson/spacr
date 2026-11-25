"""The pixel fallback calls only nuclei the reads left open (settings schema 964).

With ``ops_pixel_read_gate`` on, a pixel vote becomes a call only for a
nucleus whose reads named no guide, or named the pixel guide alone; read
ties, read refusals of two or more guides and a lone read guide that the
pixels contradict stay uncalled. Settings files from before schema 964 keep
the old behaviour. The decoy FDR report counts the pixel-fallback calls on
their own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from spacr import ops_decode
from spacr.settings import DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION, _fold_older_defaults

GUIDES = ["GTACGT", "CATGCA", "TGCATG"]


def _block(codes):
    values = np.full((len(codes), 6, 4), 10.0, np.float32)
    for i, code in enumerate(codes):
        for c, letter in enumerate(code):
            values[i, c, "GTAC".index(letter)] = 1000.0
    return values


def _field():
    """Reads and pixels of five nuclei.

    11: two reads GTACGT, a read call. 12: reads tie GTACGT and CATGCA,
    pixels TGCATG. 13: one read CATGCA, pixels CATGCA. 14: one read GTACGT,
    pixels TGCATG. 15: no read, pixels TGCATG.
    """
    reads = [(11, "GTACGT"), (11, "GTACGT"), (12, "GTACGT"), (12, "CATGCA"),
             (13, "CATGCA"), (14, "GTACGT")]
    pixels = [(12, "TGCATG")] * 3 + [(13, "CATGCA")] * 3 + [(14, "TGCATG")] * 3 \
        + [(15, "TGCATG")] * 3
    n, m = len(reads), len(pixels)
    return {
        "strength": np.full(n, 500.0), "values": _block([c for _, c in reads]),
        "gap": np.tile(np.array([[0.0, 30.0]], np.float32), (n, 1)),
        "id0": np.array([o for o, _ in reads]), "owned0": np.ones(n, bool),
        "owned_ids": np.array([11, 12, 13, 14, 15]),
        "pix_values": _block([c for _, c in pixels]),
        "pix_gap": np.tile(np.array([[1, 9]], np.float16), (m, 1)),
        "pix_id0": np.array([o for o, _ in pixels]), "pix_owned0": np.ones(m, bool),
    }


def _rule(gate):
    settings = {"ops_decode_pixel_fallback": True}
    if gate is not None:
        settings["ops_pixel_read_gate"] = gate
    return ops_decode.decode_rule(settings)


def _calls(gate, single_read_quality=2.0):
    matcher, decoys = ops_decode.matcher_for(GUIDES, ())
    return ops_decode.field_calls(_field(), _rule(gate), GUIDES, matcher, decoys,
                                  threshold=315.0, footprint=10.0, max_distance=0,
                                  min_fraction=0.6,
                                  single_read_quality=single_read_quality,
                                  crosstalk="none")


def test_the_gate_keeps_only_nuclei_the_reads_left_open():
    old = _calls(False)
    assert {o: h["source"] for o, h in old["calls"].items()} == {
        11: "primary", 12: "pixel_fallback", 13: "pixel_fallback",
        14: "pixel_fallback", 15: "pixel_fallback"}
    new = _calls(True)
    assert {o: h["source"] for o, h in new["calls"].items()} == {
        11: "primary", 13: "pixel_fallback", 15: "pixel_fallback"}
    assert new["calls"][13]["guide"] == "CATGCA"
    assert new["calls"][15]["guide"] == "TGCATG"
    assert (new["assigned_real"], new["assigned_pixel_real"]) == (3, 2)
    assert (old["assigned_real"], old["assigned_pixel_real"]) == (5, 4)


def test_the_gate_is_on_by_default_and_old_files_keep_it_off():
    assert _rule(None)["pixel_read_gate"] is True
    assert set(_calls(None)["calls"]) == {11, 13, 15}
    assert SETTINGS_SCHEMA_VERSION >= 964
    assert DEFAULT_CHANGES[964] == {"ops_pixel_read_gate": False}
    old_file = {"plate": "P1"}
    applied = _fold_older_defaults(old_file, 963,
                                   known={"plate", "ops_pixel_read_gate"})
    assert applied["ops_pixel_read_gate"] == (False, 964)
    assert old_file["ops_pixel_read_gate"] is False
    assert _fold_older_defaults({"plate": "P1"}, 964,
                                known={"plate", "ops_pixel_read_gate"}) == {}
    from spacr.ops_settings import OPS_DEFAULTS

    assert OPS_DEFAULTS["ops_pixel_read_gate"] is True


def test_a_read_call_keeps_its_pixel_support():
    out = _calls(True, single_read_quality=0.0)
    assert {o: h["source"] for o, h in out["calls"].items()} == {
        11: "primary", 13: "primary", 14: "primary", 15: "pixel_fallback"}
    assert out["calls"][13]["n_pixels"] == 3
    assert out["calls"][14]["n_pixels"] == 3
    assert out["calls"][14]["guide"] == "GTACGT"


@pytest.mark.parametrize("gate", [False, True])
def test_the_auto_candidates_rebuild_the_same_calls(gate):
    matcher, decoys = ops_decode.matcher_for(GUIDES, ())
    rule = _rule(gate)
    candidates = ops_decode._joint_field_candidates(
        _field(), rule, GUIDES, matcher, decoys, spot_thresholds=[200.0, 315.0],
        floors=[0.0, 2.0], pixel_qualities=[rule["pixel_quality"]],
        footprint=10.0, max_distance=0, min_fraction=0.6, crosstalk="none")
    for (i, floor), spot in [((i, f), s) for i, f in enumerate((0.0, 2.0)) for s in (0, 1)]:
        direct = _calls(gate, single_read_quality=floor)
        rebuilt = ops_decode._candidate_calls(candidates["by_spot"][spot], floor,
                                              rule["pixel_quality"], decoys)
        assert set(rebuilt["calls"]) == set(direct["calls"])
        assert {o: h["source"] for o, h in rebuilt["calls"].items()} == {
            o: h["source"] for o, h in direct["calls"].items()}
        assert rebuilt["assigned_pixel_real"] == direct["assigned_pixel_real"]
        assert candidates["counts"][spot, i, 0].tolist() == [direct["assigned_real"],
                                                          direct["assigned_decoy"]]
        sources = {row["object_id"]: row["source"] for row in rebuilt["evidence"]}
        assert sources == {o: h["source"] for o, h in direct["calls"].items()}


def test_pixel_decoy_calls_are_counted_apart():
    matcher, decoys = ops_decode.matcher_for(GUIDES[:2], ["TGCATG"])
    field = _field()
    out = ops_decode.field_calls(field, _rule(True), GUIDES[:2], matcher, decoys,
                                 threshold=315.0, footprint=10.0, max_distance=0,
                                 min_fraction=0.6, single_read_quality=2.0,
                                 crosstalk="none")
    assert out["calls"][15]["decoy"] is True
    assert out["assigned_pixel_decoy"] == 1 and out["assigned_decoy"] == 1
    assert 14 not in out["calls"]


def test_the_fdr_report_counts_the_pixel_fallback_on_its_own():
    stats = pd.DataFrame({"nuclei": [100, 100], "assigned_real": [60, 40],
                          "assigned_decoy": [2, 1], "assigned_pixel_real": [5, 5],
                          "assigned_pixel_decoy": [1, 1]})
    report = ops_decode.fdr_report(stats, plate="P", well="A1", k=4, scale=1.0)
    assert report["assigned"] == 100 and report["assigned_decoy"] == 3
    assert report["assigned_primary"] == 90 and report["assigned_primary_decoy"] == 1
    assert report["assigned_pixel_fallback"] == 10
    assert report["assigned_pixel_fallback_decoy"] == 2
    assert report["fdr_per_100_primary"] == pytest.approx(
        round(ops_decode.fdr(90, 1, 4, 1.0), 3))
    assert report["fdr_per_100_pixel_fallback"] == pytest.approx(
        round(ops_decode.fdr(10, 2, 4, 1.0), 3))
    plain = ops_decode.fdr_report(stats[["nuclei", "assigned_real", "assigned_decoy"]],
                                  plate="P", well="A1", k=4, scale=1.0)
    assert "assigned_pixel_fallback" not in plain

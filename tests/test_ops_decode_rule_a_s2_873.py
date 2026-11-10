"""Item 873: the decode rule (a) + S2 as OPS engine settings.

Each arm of the rule is pinned by a small synthetic case where turning it
on changes the outcome; the engine is run end to end on the synthetic well
of tests/test_the_ops_engine_runs_a_well_end_to_end.py with the
``rule_a_s2`` preset, which writes the decoy FDR report and logs an
unplaced site. Validation against the study's own outputs (plate A and B
field caches, fields A1/74 and A1/190 from the raw tiles) is recorded in
the item's feature file.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("tifffile")
pytest.importorskip("pyarrow")

from spacr import ops_decode, ops_engine, ops_sbs
from spacr.ops_cycles import align_field
from spacr.settings import ALPHA_FEATURES


def test_decoys_are_shuffles_two_edits_from_everything():
    library = ["GTACGTACGTA", "TTAGGCCAATG", "ACGTTGCAGCA", "CAGTCAGTCAG"]
    decoys = ops_sbs.decoy_library(library, k=3, seed=1)
    assert 0 < len(decoys) <= 12
    assert decoys == ops_sbs.decoy_library(list(reversed(library)), k=3, seed=1)
    pool = library + decoys
    for d in decoys:
        assert sorted(d) in [sorted(g) for g in library]
        others = [e for e in pool if e != d]
        assert min(sum(a != b for a, b in zip(d, e)) for e in others) >= 2


def test_the_matcher_is_unique_closest_with_n_as_a_wildcard():
    matcher = ops_sbs.LibraryMatcher(["AAAA", "AACC", "GGGG"])
    assert matcher.match("AAAA", 0) == ("AAAA", 0)
    assert matcher.match("AAAT", 0) == (None, -1)
    assert matcher.match("AAAT", 1) == ("AAAA", 1)
    assert matcher.match("AACA", 1) == (None, -1)
    assert matcher.match("GGNG", 0) == ("GGGG", 0)
    assert matcher.match("AANN", 0) == (None, -1)
    assert matcher.match("AAA", 1) == (None, -1)
    assert matcher.match("AAAT", 1) == ("AAAA", 1)
    assert ops_sbs.LibraryMatcher([]).match("AAAA", 1) == (None, -1)


def test_the_runner_up_rule_needs_an_uncertain_cycle_naming_the_guide():
    data = np.array([[1.0, 0.0, 0.0, 0.0], [0.6, 0.5, 0.0, 0.0]])
    margin = np.array([1.0, 0.05])
    assert ops_sbs.runner_up_consistent(data, margin, "GG", "GT", 0.1)
    assert not ops_sbs.runner_up_consistent(data, margin, "GG", "GA", 0.1)
    assert not ops_sbs.runner_up_consistent(data, np.array([1.0, 0.5]), "GG", "GT", 0.1)
    assert ops_sbs.runner_up_consistent(data, margin, "GN", "GA", 0.1)


def _vote(objects, codes, quality, guides, **kw):
    n = len(codes)
    cycles = len(codes[0])
    data = np.zeros((n, cycles, 4), np.float32)
    margin = np.ones((n, cycles), np.float32)
    for i, code in enumerate(codes):
        for c, letter in enumerate(code):
            if letter != "N":
                data[i, c, "GTAC".index(letter)] = 1.0
    return ops_sbs.vote_mapped_reads(np.asarray(objects), codes,
                                     np.asarray(quality, np.float32), data,
                                     margin, ops_sbs.LibraryMatcher(guides), **kw)


def test_quality_weighting_lets_one_clean_read_outvote_two_doubtful_ones():
    guides = ["GTAC", "CATG"]
    args = ([7, 7, 7], ["GTAC", "GTAC", "CATG"], [0.0, 0.0, 0.9], guides)
    assert _vote(*args)[7]["guide"] == "GTAC"
    weighted = _vote(*args, quality_weighted=True)
    assert weighted[7]["guide"] == "CATG"
    assert weighted[7]["n_support"] == 1


def test_one_read_needs_the_single_read_quality():
    guides = ["GTAC"]
    assert _vote([3], ["GTAC"], [0.05], guides, single_read_quality=0.06) == {}
    hit = _vote([3], ["GTAC"], [0.07], guides, single_read_quality=0.06)[3]
    assert hit["barcode"] == "GTAC" and hit["n_exact"] == 1


def test_a_read_with_a_lost_cycle_matches_only_when_imputed():
    guides = ["GTAC", "CATG"]
    assert _vote([1], ["GTNC"], [0.5], guides)[1]["guide"] == "GTAC"
    assert _vote([1], ["GTNC"], [0.5], guides, impute_missing=False) == {}


def test_pixels_vote_by_count_and_need_two_agreeing():
    matcher = ops_sbs.LibraryMatcher(["GTAC", "CATG"])
    out = ops_sbs.vote_pixels(np.array([1, 1, 1, 2, 3, 3]),
                              ["GTAC", "GTAC", "CATG", "GTAC", "GTAC", "CATG"],
                              np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), matcher)
    assert out[1]["guide"] == "GTAC" and out[1]["n_pixels"] == 2
    assert 2 not in out and 3 not in out
    weak = ops_sbs.vote_pixels(np.array([1, 1]), ["GTAC", "GTAC"],
                               np.array([0.05, 0.5]), matcher)
    assert weak == {}


def test_nearest_boundaries_rank_by_gap_not_by_centre():
    gap, index = ops_sbs.nearest_boundaries(
        np.array([[0.0, 0.0]]), np.array([[0.0, 5.0], [0.0, 8.0]]),
        np.array([math.pi * 1.0, math.pi * 49.0]))
    assert index[0].tolist() == [1, 0]
    assert gap[0].tolist() == pytest.approx([1.0, 4.0])
    gap, index = ops_sbs.nearest_boundaries(np.array([[0.0, 0.0]]),
                                            np.array([[0.0, 5.0]]),
                                            np.array([math.pi]))
    assert index[0].tolist() == [0, -1] and np.isinf(gap[0, 1])


def _library_reads(rng, n_guides=40, reads_per=60, cycles=6, bleed_cycle=2):
    guides = sorted({"".join(rng.choice(list("GTAC"), cycles)) for _ in range(n_guides)})
    codes = [g for g in guides for _ in range(reads_per)]
    values = np.full((len(codes), cycles, 4), 50.0, np.float32)
    for i, code in enumerate(codes):
        for c, letter in enumerate(code):
            values[i, c, "GTAC".index(letter)] += 1000.0
    values += rng.normal(0, 20, values.shape).astype(np.float32)
    # In one cycle the G dye reads mostly in the T channel, brighter than a
    # real T: a per-cycle distortion one plate-wide unmixing cannot undo.
    is_g = np.array([code[bleed_cycle] == "G" for code in codes])
    values[is_g, bleed_cycle, 0] = 500 + rng.normal(0, 20, is_g.sum())
    values[is_g, bleed_cycle, 1] = 1500 + rng.normal(0, 20, is_g.sum())
    return guides, codes, values


def test_library_calibration_learns_a_cycle_the_global_unmixing_misses():
    rng = np.random.default_rng(0)
    guides, codes, values = _library_reads(rng)
    _, plain, _ = ops_sbs.called_bases(values, compensate=True,
                                       method="argmax_median", gpu=False)
    _, calibrated, margin = ops_sbs.library_calibrated_bases(values, guides)
    plain_ok = np.mean([a == b for a, b in zip(plain, codes)])
    calibrated_ok = np.mean([a == b for a, b in zip(calibrated, codes)])
    assert plain_ok < 0.9
    assert calibrated_ok > 0.99
    assert margin.shape == (len(codes), 6)
    empty = ops_sbs.library_calibrated_bases(np.zeros((0, 6, 4)), guides)
    assert empty[1] == []


def _spot_field(rng, shift=(0, 0), size=160):
    texture = np.full((size, size), 300.0)
    yy, xx = np.mgrid[:size, :size]
    texture += 200 * np.exp(-((yy - 80) ** 2 + (xx - 80) ** 2) / 3000.0)
    spots = [(int(y), int(x)) for y, x in rng.integers(20, size - 20, (60, 2))]
    calls = rng.integers(0, 4, (2, len(spots)))
    planes = {}
    for cycle in (1, 2):
        planes[cycle] = []
        for k in range(4):
            plane = texture + rng.normal(0, 3, texture.shape)
            dy, dx = shift if (cycle == 2 and k == 1) else (0, 0)
            for (y, x), base in zip(spots, calls[cycle - 1]):
                if base == k:
                    plane[y + dy - 1:y + dy + 2, x + dx - 1:x + dx + 2] += 3000
            planes[cycle].append(plane.astype(np.float32))
    return planes


def test_spots_refine_corrects_a_residual_shift_spots_keeps():
    planes = _spot_field(np.random.default_rng(4), shift=(2, 0))
    kept = align_field(planes, reference=1, gpu=False, channel_mode="spots")
    refined = align_field(planes, reference=1, gpu=False,
                          channel_mode="spots_refine")
    assert (2, 1) not in kept.spot_replaced
    assert (2, 1) in refined.spot_replaced
    assert refined.channel_shifts[2][1] == (-2, 0)
    strict = align_field(planes, reference=1, gpu=False,
                         channel_mode="spots_refine", spot_min_peak=1e9)
    assert strict.spot_replaced == ()
    with pytest.raises(ValueError, match="channel_mode"):
        align_field(planes, reference=1, gpu=False, channel_mode="spot")


def test_the_preset_fills_defaults_and_keeps_what_was_set():
    resolved = ops_decode.resolve_settings({"ops_decode_preset": "rule_a_s2",
                                            "ops_single_read_quality": 0.03,
                                            "ops_pixel_quality": 0.2})
    assert resolved["ops_single_read_quality"] == 0.06
    assert resolved["ops_pixel_quality"] == 0.2
    assert resolved["ops_channel_registration"] == "spots_refine"
    assert ops_decode.resolve_settings({"x": 1}) == {"x": 1}
    with pytest.raises(ValueError, match="ops_decode_preset"):
        ops_decode.resolve_settings({"ops_decode_preset": "rule_z"})
    rule = ops_decode.decode_rule(resolved)
    assert rule["active"] and rule["decoy_k"] == 4 and rule["pixels"]
    assert not ops_decode.decode_rule({})["active"]
    assert ops_decode.decode_rule({"ops_decode_runner_up": "false"})["active"] is False
    with pytest.raises(ValueError, match="ops_decoy_library_k"):
        ops_decode.decode_rule({"ops_decoy_library_k": -1})
    with pytest.raises(ValueError, match="ops_pixel_min_fraction"):
        ops_decode.decode_rule({"ops_pixel_min_fraction": 0})
    with pytest.raises(ValueError, match="must be a number"):
        ops_decode.decode_rule({"ops_tie": "near"})


def test_the_fdr_is_scaled_per_decoy_and_never_credited():
    assert ops_decode.fdr(1000, 4, 4, 1.268) == pytest.approx(0.1268)
    assert ops_decode.fdr(1000, 4, 4, 0.5) == pytest.approx(0.1)
    assert math.isnan(ops_decode.fdr(1000, 4, 0, 1.268))


def _field(owned_ids=(11, 12, 13)):
    """A cache-layout field: object 11 has two exact reads, 12 one read that
    lies ambiguously between 12 and 13, 13 only pixels."""
    guides = ["GTACGT", "CATGCA", "TGCATG"]
    codes = ["GTACGT", "GTACGT", "CATGCA"]
    values = np.full((3, 6, 4), 10.0, np.float32)
    for i, code in enumerate(codes):
        for c, letter in enumerate(code):
            values[i, c, "GTAC".index(letter)] = 1000.0
    pix_codes = ["TGCATG"] * 3
    pix = np.full((3, 6, 4), 10.0, np.float32)
    for i, code in enumerate(pix_codes):
        for c, letter in enumerate(code):
            pix[i, c, "GTAC".index(letter)] = 1000.0
    evidence = {
        "strength": np.array([500.0, 500.0, 500.0]), "values": values,
        "gap": np.array([[0.0, 30.0], [1.0, 20.0], [2.0, 2.5]], np.float32),
        "id0": np.array([11, 11, 12]), "owned0": np.array([True, True, True]),
        "owned_ids": np.array(owned_ids), "pix_values": pix,
        "pix_gap": np.array([[1, 9], [2, 9], [3, 9]], np.float16),
        "pix_id0": np.array([13, 13, 13]), "pix_owned0": np.array([True] * 3),
    }
    return guides, evidence


def _calls(rule_settings, evidence, guides, decoys=()):
    rule = ops_decode.decode_rule(rule_settings)
    matcher, decoy_set = ops_decode.matcher_for(guides, decoys)
    return ops_decode.field_calls(evidence, rule, guides, matcher, decoy_set,
                                  threshold=315.0, footprint=10.0,
                                  max_distance=0, min_fraction=0.6,
                                  single_read_quality=0.0, crosstalk="none")


def test_territory_and_the_pixel_fallback_each_add_a_nucleus():
    guides, evidence = _field()
    plain = _calls({}, evidence, guides)
    assert set(plain["calls"]) == {11}
    territory = _calls({"ops_assign_by_cell_territory": True}, evidence, guides)
    assert set(territory["calls"]) == {11, 12}
    both = _calls({"ops_assign_by_cell_territory": True,
                   "ops_decode_pixel_fallback": True}, evidence, guides)
    assert both["calls"][13]["source"] == "pixel_fallback"
    assert both["calls"][13]["guide"] == "TGCATG"
    assert both["assigned_real"] == 3 and both["nuclei"] == 3
    assert both["calls"][11]["n_reads"] == 2


def test_a_decoy_call_is_counted_and_not_owned_calls_are_dropped():
    guides, evidence = _field(owned_ids=(11, 13))
    out = _calls({"ops_decode_pixel_fallback": True}, evidence, guides[:2],
                 decoys=["TGCATG"])
    assert out["calls"][13]["decoy"] is True
    assert out["assigned_decoy"] == 1 and out["assigned_real"] == 1
    rows = ops_decode.barcode_rows("P", "A1", [{"rule_a": out, "n_cycles": 6}])
    assert [r["object_id"] for r in rows] == [11]


def test_every_873_setting_is_alpha_and_declared():
    from spacr import ops_settings

    entry = ALPHA_FEATURES[873]
    for key in entry["settings"]:
        assert key in ops_settings.OPS_DEFAULTS
        assert key in ops_settings.OPS_TYPES
        assert "Alpha." in ops_settings.OPS_TOOLTIPS[key]
        assert any(key in keys for keys in ops_settings.OPS_CATEGORIES.values())
    assert entry["choices"] == {"ops_channel_registration": ("spots_refine",)}
    for key in ops_decode.RULE_A_S2:
        assert key in ops_settings.OPS_DEFAULTS
    assert os.path.isfile(os.path.join(
        os.path.dirname(__file__), os.pardir, "features", "feature_list",
        "873_ops_decode_rule_a_s2.txt"))


@pytest.fixture(scope="module")
def rule_a_run(tmp_path_factory):
    """The synthetic well, stitched and segmented, decoded under the preset."""
    import tests.test_the_ops_engine_runs_a_well_end_to_end as base

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ops_engine, "_RASTER_OVERLAP", base.OVERLAP)
    monkeypatch.setattr(ops_engine, "_WINDOW", 256)
    monkeypatch.setattr(ops_engine, "_WINDOW_OVERLAP", 48)
    monkeypatch.setattr(ops_engine, "_cellpose_model",
                        lambda settings, gpu: base._ThresholdModel())
    root = tmp_path_factory.mktemp("ops_rule_a")
    raw = root / "raw"
    raw.mkdir()
    nuclei, codes, centre, tops = base._plate(raw)
    settings = {"genotype_source": str(raw), "dst_root": str(root / "out"),
                "plate": "synthetic", "ops_gpu": False, "n_workers": 1}
    library = sorted(set(codes)) + ["TTTT", "GGGG"]
    plain = ops_engine.run_ops(settings, library=library)
    for cycle in range(2, base.CYCLES + 1):
        for name in ("CY3", "A594", "CY5", "CY7"):
            shutil.copy(raw / f"c{cycle}" / f"10X_c{cycle}_A1_{name}_Site-0.tif",
                        raw / f"c{cycle}" / f"10X_c{cycle}_A1_{name}_Site-9.tif")
    preset = {**settings, "ops_decode_preset": "rule_a_s2",
              "ops_store_field_cache": True}
    result = ops_engine.run_ops(preset, phases=("decode",), library=library)
    yield plain, result, codes, nuclei, library, preset
    monkeypatch.undo()


def test_the_preset_decodes_the_well_and_writes_its_fdr_report(rule_a_run):
    plain, result, codes, nuclei, library, preset = rule_a_run
    decode = result["wells"]["A1"]["decode"]
    assert decode["cell_rule"] == "mapped"
    assert decode["channel_registration"] == "spots_refine"
    assert decode["skipped_unplaced_sites"] == [9]
    fdr = decode["fdr"]
    assert fdr["skipped_sites"] == [9] and fdr["decoy_library_k"] == 4
    assert fdr["assigned"] == decode["objects_assigned"] > 0
    with open(fdr["paths"]["fdr"], encoding="utf-8") as handle:
        assert json.load(handle)["fdr_per_100"] == fdr["fdr_per_100"]
    with open(fdr["paths"]["plate_summary"], encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["assigned"] == fdr["assigned"]
    assert summary["per_well"]["A1"]["pct"] == fdr["pct"]
    import pandas as pd

    stats = pd.read_csv(fdr["paths"]["field_stats"])
    assert list(stats.columns) == ["plate", "well", "site", "nuclei",
                                   "assigned_real", "assigned_decoy"]
    assert stats["assigned_real"].sum() == fdr["assigned"]
    assert len(decode["field_cache"]) == decode["fields_decoded"]
    cached = np.load(decode["field_cache"][0])
    assert {"peaks", "values", "gap", "pix_values"} <= set(cached.files)
    with sqlite3.connect(result["db"]) as conn:
        rows = conn.execute("SELECT b.mapped_guide, o.centroid_y, o.centroid_x "
                            "FROM ops_barcodes b JOIN ops_objects o USING "
                            "(plate, well, object_id)").fetchall()
    assert len(rows) == fdr["assigned"]
    plain_rows = plain["wells"]["A1"]["decode"]["objects_mapped"]
    assert len(rows) >= plain_rows
    margin = 20
    for guide, y, x in rows:
        index = int(np.argmin([math.hypot(y - (a - margin), x - (b - margin))
                               for a, b in nuclei]))
        assert guide == codes[index]


def test_an_unplaced_site_stops_the_well_when_skipping_is_off(rule_a_run):
    preset, library = rule_a_run[5], rule_a_run[4]
    with pytest.raises(ValueError, match="placed no position for sites"):
        ops_engine.run_ops({**preset, "ops_decode_skip_unplaced_sites": False},
                           phases=("decode",), library=library)


def test_the_new_rules_fall_back_without_the_mapped_rule(rule_a_run, capsys):
    preset, library = rule_a_run[5], rule_a_run[4]
    result = ops_engine.run_ops({**preset, "ops_cell_rule": "agree"},
                                phases=("decode",), library=library)
    assert "fdr" not in result["wells"]["A1"]["decode"]
    assert "need" in capsys.readouterr().out


def test_evidence_rows_carry_the_support_behind_each_call():
    guides, evidence = _field()
    rule = ops_decode.decode_rule({"ops_assign_by_cell_territory": True,
                                   "ops_decode_pixel_fallback": True,
                                   "ops_decode_quality_weighted": True})
    matcher, decoy_set = ops_decode.matcher_for(guides[:2], ["TGCATG"])
    out = ops_decode.field_calls(evidence, rule, guides[:2], matcher, decoy_set,
                                 threshold=315.0, footprint=10.0, max_distance=0,
                                 min_fraction=0.6, single_read_quality=0.0,
                                 crosstalk="none", evidence_rows=True)
    rows = {row["object_id"]: row for row in out["evidence"]}
    assert set(rows) == {11, 12, 13}
    assert set(rows[11]) == set(ops_decode.EVIDENCE_COLUMNS)
    assert rows[11]["source"] == "primary" and rows[11]["n_exact"] == 2
    assert rows[11]["vote_frac"] == 1.0 and rows[11]["runner_margin"] == 1.0
    assert rows[11]["strength_max"] == 500.0 and rows[11]["n_reads"] == 2
    assert rows[12]["n_amb"] == 1
    assert rows[13]["source"] == "pixel_fallback" and rows[13]["is_decoy"]
    assert rows[13]["pix_n"] == 3 and rows[13]["pix_frac"] == 1.0
    assert math.isnan(rows[13]["vote_frac"]) and math.isnan(rows[13]["q_best"])
    assert rows[11]["field_reads_per_nucleus"] == 1.0

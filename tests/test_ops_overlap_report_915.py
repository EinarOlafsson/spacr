"""Read stage and overlap-strip match rates in the OPS run report (item 915, alpha).

Two 100 x 100 fields overlap by 30 columns. One physical cell sits in the
overlap and is segmented in both fields; only field 0's copy is linked. Per
segmented copy the strip's match rate is therefore 50 %, per physical cell
100 % -- the spec's own example of why both units are reported.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

from spacr import ops_decode, ops_engine
from spacr.ops_store import write_table

PLATE = "P1"


def _placements(offsets):
    """``ops_phenotype`` rows for fields at the given x offsets."""
    return pd.DataFrame([{"plate": PLATE, "well": "A1", "site": site, "centre_y": 50.0,
                          "centre_x": ox + 50.0, "is_anchor": 1, "scale": 1.0, "rotation_deg": 0.0,
                          "offset_y": 0.0, "offset_x": ox, "field_height": 100, "field_width": 100,
                          "transform_source": "anchor", "refined": 0}
                         for site, ox in enumerate(offsets)])


def _run_folder(tmp_path, *, ownership=True, links=((0, 1), (0, 2), (1, 2))):
    """Two overlapping fields: mask pass, placements, links and ownership.

    Field 0: cell 1 at (50, 90), in the overlap; cell 2 at (20, 20), core.
    Field 1: cell 1 at (50, 20), the same physical cell as field 0's cell 1;
    cell 2 at (60, 60), core. Field 0's cell 3 at (80, 40), core, unlinked.
    """
    db = str(tmp_path / "measurements.db")
    write_table(db, "ops_phenotype", _placements([0.0, 70.0]))
    cells = pd.DataFrame({
        "plate": PLATE, "well": "A1", "site": [0, 0, 0, 1, 1], "label": [1, 2, 3, 1, 2],
        "cell_cy": [50.0, 20.0, 80.0, 50.0, 60.0], "cell_cx": [90.0, 20.0, 40.0, 20.0, 60.0],
        "nucleus_cy": [50.0, 20.0, 80.0, 50.0, 60.0], "nucleus_cx": [91.0, 21.0, 41.0, 21.0, 61.0],
        "cell_px": 100, "touches_field_edge": False})
    cells.to_parquet(ops_engine._ph_maskpass_path(str(tmp_path), "A1"), index=False)
    write_table(db, "ops_merge", pd.DataFrame([
        {"plate": PLATE, "well": "A1", "site": s, "label": lab, "object_id": 10 * s + lab,
         "distance": 1.0, "refined": 0} for s, lab in links]))
    if ownership:
        pd.DataFrame({"plate": PLATE, "well": "A1", "site": [0, 0, 0, 1, 1], "label": [1, 2, 3, 1, 2],
                      "area": 100, "component": [0, 1, 2, 0, 3], "n_copies": [2, 1, 1, 2, 1],
                      "owner_field": [1, 0, 0, 1, 1], "owner_label": [1, 2, 3, 1, 2],
                      "is_owner": [False, True, True, True, True], "owned_area_share": 1.0,
                      "touches_field_edge": False}).to_parquet(
            ops_engine._ph_cross_field_path(str(tmp_path), "A1"), index=False)
    return db


def _report(tmp_path, **settings):
    """Run the report phase and return its folder."""
    ops_engine.run_ops({"dst_root": str(tmp_path), "plate": PLATE, **settings}, wells=["A1"],
                       phases=("report",))
    return tmp_path / "results" / "ops_report"


def test_a_cell_copied_into_two_fields_is_half_matched_per_copy_and_whole_per_cell(tmp_path):
    _run_folder(tmp_path)
    folder = _report(tmp_path, ops_report_overlap_zones=True)
    zones = pd.read_csv(folder / "overlap_match_by_zone.csv").set_index(["unit", "zone"])
    assert zones.loc[("copy", "overlap strip"), ["cells", "matched"]].tolist() == [2, 1]
    assert zones.loc[("copy", "overlap strip"), "match_pct"] == 50.0
    assert zones.loc[("copy", "core"), ["cells", "matched"]].tolist() == [3, 2]
    assert zones.loc[("physical_cell", "overlap strip"), ["cells", "matched"]].tolist() == [1, 1]
    assert zones.loc[("physical_cell", "overlap strip"), "match_pct"] == 100.0
    assert zones.loc[("physical_cell", "core"), ["cells", "matched"]].tolist() == [3, 2]
    edges = pd.read_csv(folder / "overlap_match_by_edge_distance.csv").set_index("edge_dist_bin_px")
    assert edges.loc[0, "cells"] == 5 and edges.loc[0, "strip_share"] == 0.4
    fields = pd.read_csv(folder / "overlap_match_per_field.csv")
    assert fields.empty
    assert not (folder / "funnel_per_well.csv").exists()


def test_the_edge_bins_follow_the_bin_width(tmp_path):
    _run_folder(tmp_path)
    folder = _report(tmp_path, ops_report_overlap_zones=True, ops_report_edge_bin_px=10)
    edges = pd.read_csv(folder / "overlap_match_by_edge_distance.csv")
    assert sorted(edges.edge_dist_bin_px) == [0, 10, 20, 30]
    assert edges.cells.sum() == 5
    with pytest.raises(ValueError, match="must be positive"):
        _report(tmp_path, ops_report_overlap_zones=True, ops_report_edge_bin_px=0)


def test_without_ownership_only_copies_are_reported(tmp_path):
    _run_folder(tmp_path, ownership=False)
    zones = pd.read_csv(_report(tmp_path, ops_report_overlap_zones=True) / "overlap_match_by_zone.csv")
    assert set(zones.unit) == {"copy"}


def test_an_external_pipeline_is_compared_on_the_same_physical_cells(tmp_path):
    _run_folder(tmp_path)
    external = pd.DataFrame({"well": "A1", "field": [0, 0, 1, 1, 5], "label": [7, 8, 9, 10, 1],
                             "y": [50.0, 20.0, 50.0, 79.0, 1.0], "x": [89.0, 20.0, 21.0, 41.0, 1.0],
                             "matched": [False, False, True, True, True]})
    external.to_parquet(tmp_path / "external.parquet", index=False)
    folder = _report(tmp_path, ops_report_overlap_zones=True,
                     ops_report_external_cells=str(tmp_path / "external.parquet"))
    paired = pd.read_csv(folder / "overlap_external_paired.csv").set_index("zone")
    assert paired.loc["overlap strip", ["physical_cells", "both", "spacr_only", "external_only"]].tolist() \
        == [1, 1, 0, 0]
    assert paired.loc["core", ["physical_cells", "both", "spacr_only", "external_only"]].tolist() == [1, 0, 1, 0]
    assert paired.loc["core", "mcnemar_exact_p"] == 1.0
    info = json.loads((folder / "overlap_external_paired.json").read_text())
    assert "same physical cells" in info["matching"]
    assert info["wells"]["A1"]["assigned"] == 3 and info["wells"]["A1"]["external_cells"] == 5
    zones = pd.read_csv(folder / "overlap_match_by_zone.csv").set_index(["unit", "zone"])
    assert zones.loc[("external_copy", "overlap strip"), ["cells", "matched"]].tolist() == [2, 1]
    assert zones.loc[("external_copy", "core"), ["cells", "matched"]].tolist() == [2, 1]


def test_external_cells_need_the_ownership_table(tmp_path):
    _run_folder(tmp_path, ownership=False)
    pd.DataFrame({"field": [0], "label": [1], "y": [50.0], "x": [90.0], "matched": [True]}).to_parquet(
        tmp_path / "external.parquet", index=False)
    with pytest.raises(ValueError, match="ph_ownership"):
        _report(tmp_path, ops_report_overlap_zones=True,
                ops_report_external_cells=str(tmp_path / "external.parquet"))


def test_the_overlap_report_needs_the_mask_pass(tmp_path):
    write_table(str(tmp_path / "measurements.db"), "ops_phenotype", _placements([0.0]))
    with pytest.raises(ValueError, match="mask pass"):
        _report(tmp_path, ops_report_overlap_zones=True)


def test_external_cells_in_unknown_fields_or_far_away_stay_unassigned():
    copies = pd.DataFrame({"site": [0, 0], "label": [1, 2], "cell_cy": [10.0, np.nan],
                           "cell_cx": [10.0, np.nan], "nucleus_cy": [np.nan, np.nan],
                           "nucleus_cx": [np.nan, np.nan]})
    external = pd.DataFrame({"field": [0, 0, 3], "label": [1, 2, 3], "y": [11.0, 90.0, 1.0],
                             "x": [10.0, 90.0, 1.0], "matched": [True, True, True]})
    out = ops_engine._ph_external_assign(external, copies, 8.0)
    assert out.spacr_label.tolist() == [1, -1]
    empty = ops_engine._ph_external_assign(external[external.field == 3], copies, 8.0)
    assert empty.empty and {"site", "spacr_label"} <= set(empty.columns)


def _evidence():
    """One field's decode evidence: three nuclei, two owned, reads of mixed strength."""
    return {"strength": np.array([400.0, 500.0, 100.0, 600.0, 700.0, 800.0]),
            "id0": np.array([1, 1, 2, -1, 3, 2]), "owned0": np.array([True, True, True, False, False, True]),
            "owned_ids": np.array([1, 2])}


def test_a_nucleus_counts_once_however_many_reads_it_holds():
    assert ops_decode.nuclei_with_read(_evidence(), 315) == 2
    assert ops_decode.nuclei_with_read(_evidence(), 750) == 1
    assert ops_decode.nuclei_with_read(_evidence(), 900) == 0


def test_the_decode_writes_the_read_stage_only_when_asked(tmp_path):
    db = str(tmp_path / "measurements.db")
    write_table(db, "ops_phenotype", _placements([0.0]))
    decoded = [{"site": s, "rule_a": {"nuclei": 10, "assigned_real": 4, "assigned_decoy": 0,
                                       "nuclei_with_read": 7 + s}} for s in (1, 0)]
    ops_decode.write_well_reports(db, PLATE, "A1", decoded, {"decoy_k": 0, "fdr_scale": 1.0}, [])
    stats = pd.read_csv(tmp_path / "results" / "ops_decode" / "field_stats_A1.csv")
    assert "nuclei_with_read" not in stats
    rule = {"decoy_k": 0, "fdr_scale": 1.0, "read_stage": True}
    ops_decode.write_well_reports(db, PLATE, "A1", decoded, rule, [])
    stats = pd.read_csv(tmp_path / "results" / "ops_decode" / "field_stats_A1.csv")
    assert stats.nuclei_with_read.tolist() == [7, 8]
    assert ops_engine._ph_read_stage(db, ["A1", "B1"]) == {"A1": (20, 15)}
    folder = _report(tmp_path, ops_report_read_stage=True)
    funnel = pd.read_csv(folder / "funnel_per_well.csv")
    assert funnel[["well", "sbs_nuclei", "sbs_nuclei_with_read", "sbs_nuclei_with_read_pct"]].values.tolist() \
        == [["A1", 20, 15, 75.0]]


def test_the_read_stage_joins_the_funnel(tmp_path):
    db = str(tmp_path / "measurements.db")
    write_table(db, "ops_phenotype", _placements([0.0]))
    decoded = [{"site": 0, "rule_a": {"nuclei": 8, "assigned_real": 4, "assigned_decoy": 0,
                                       "nuclei_with_read": 6}}]
    ops_decode.write_well_reports(db, PLATE, "A1", decoded, {"decoy_k": 0, "fdr_scale": 1.0,
                                                             "read_stage": True}, [])
    pd.DataFrame({"plate": PLATE, "well": ["A1", "A2"], "cells_segmented": [10, 12],
                  "owned_whole": [9, 11], "linked": [8, 9], "with_guide": [7, 8],
                  "interphase": [6, 7]}).to_parquet(tmp_path / f"ops_funnel_{PLATE}.parquet")
    folder = _report(tmp_path, ops_report_read_stage=True)
    funnel = pd.read_csv(folder / "funnel_per_well.csv").set_index("well")
    assert funnel.loc["A1", "sbs_nuclei_with_read"] == 6 and funnel.loc["A1", "sbs_nuclei_with_read_pct"] == 75.0
    assert np.isnan(funnel.loc["A2", "sbs_nuclei_with_read"])
    assert list(funnel.columns).index("sbs_nuclei") < list(funnel.columns).index("aggregate_cells_segmented")
    plain = pd.read_csv(_report(tmp_path) / "funnel_per_well.csv")
    assert "sbs_nuclei" not in plain


def test_the_decode_rule_carries_the_read_stage_only_when_on():
    from spacr.ops_decode import decode_rule
    from spacr.ops_settings import OPS_DEFAULTS

    assert "read_stage" not in decode_rule(dict(OPS_DEFAULTS))
    assert decode_rule({**OPS_DEFAULTS, "ops_report_read_stage": True})["read_stage"] is True


def test_the_report_settings_are_alpha_with_short_tooltips():
    from spacr.ops_settings import OPS_DEFAULTS, OPS_TOOLTIPS
    from spacr.settings import ALPHA_FEATURES

    keys = ALPHA_FEATURES[915]["settings"]
    assert set(keys) == {"ops_report_read_stage", "ops_report_overlap_zones", "ops_report_edge_bin_px",
                         "ops_report_external_cells", "ops_report_external_max_px"}
    for key in keys:
        assert key in OPS_DEFAULTS and len(OPS_TOOLTIPS[key]) <= 600 and "Alpha" in OPS_TOOLTIPS[key]
    assert OPS_DEFAULTS["ops_report_edge_bin_px"] == 100 and OPS_DEFAULTS["ops_report_external_max_px"] == 8.0


def test_the_field_calls_count_the_read_stage_only_when_asked():
    from tests.test_ops_decode_rule_a_s2_873 import _calls, _field

    guides, evidence = _field()
    plain = _calls({}, evidence, guides)
    assert "nuclei_with_read" not in plain
    counted = _calls({"ops_report_read_stage": True}, evidence, guides)
    assert counted["nuclei_with_read"] == 2
    assert counted["calls"].keys() == plain["calls"].keys()

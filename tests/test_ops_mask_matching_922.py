"""Item 922: cross-pipeline cell matching by mask overlap.

Synthetic label pairs with known overlaps (shift, split, merge, ties), the
PERISCOPE site key, the Feldman reads-to-cell rule, concordance categories,
and the compare_calls phase in mask_overlap mode end to end, including the
self-match control. Validation on PERISCOPE CP186A Well1 is recorded in the
item's feature file.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
tifffile = pytest.importorskip("tifffile")

from spacr import ops_decode
from spacr.ops_engine import run_ops
from spacr.settings import ALPHA_FEATURES
from spacr.tabular import read_table, write_table


def _square(img, label, y, x, h, w):
    img[y:y + h, x:x + w] = label
    return img


def test_site_key_parser():
    assert ops_decode.parse_site_key("CP186A-Well3-75") == ("CP186A", "Well3", 75)
    assert ops_decode.parse_site_key("my-plate-Well1-2") == ("my-plate", "Well1", 2)
    with pytest.raises(ValueError):
        ops_decode.parse_site_key("CP186A-Well3")
    with pytest.raises(ValueError):
        ops_decode.parse_site_key("CP186A-Well3-x")


def test_contingency_counts_every_overlapping_pair_once():
    a = np.zeros((10, 10), int)
    b = np.zeros((10, 10), int)
    _square(a, 1, 2, 2, 4, 4)
    _square(b, 7, 2, 4, 4, 4)
    pairs, area_a, area_b = ops_decode.label_contingency(a, b)
    assert pairs[["label_a", "label_b", "overlap"]].values.tolist() == [[1, 7, 8]]
    assert pairs["iou"].iloc[0] == pytest.approx(8 / 24)
    assert area_a.to_dict() == {1: 16} and area_b.to_dict() == {7: 16}
    with pytest.raises(ValueError):
        ops_decode.label_contingency(a, b[:5])


def test_shift_split_and_merge_give_the_expected_f1():
    a = np.zeros((40, 40), int)
    b = np.zeros((40, 40), int)
    _square(a, 1, 5, 5, 10, 10)
    _square(b, 1, 5, 6, 10, 10)
    _square(a, 2, 5, 20, 10, 10)
    _square(b, 2, 5, 20, 10, 4)
    _square(b, 3, 5, 24, 10, 3)
    _square(b, 5, 5, 27, 10, 3)
    _square(a, 3, 22, 5, 10, 3)
    _square(a, 4, 22, 8, 10, 7)
    _square(b, 4, 22, 5, 10, 10)
    result = ops_decode.match_masks(a, b, iou=0.5)
    counts = result["counts"]
    assert (counts["tp"], counts["fp"], counts["fn"]) == (2, 2, 3)
    assert counts["f1"] == pytest.approx(4 / 9)
    assert counts["median_iou"] == pytest.approx((90 / 110 + 0.7) / 2)
    m = result["matches"].set_index("spacr_id")
    assert m.loc[1, "matched"] and m.loc[1, "matched_reference_id"] == 1
    assert m.loc[2, "reference_id"] == 2 and m.loc[2, "iou"] == pytest.approx(0.4) and not m.loc[2, "matched"]
    assert m.loc[3, "reference_id"] == 4 and m.loc[3, "iou"] == pytest.approx(0.3) and not m.loc[3, "matched"]
    assert m.loc[4, "matched_reference_id"] == 4
    loose = ops_decode.match_masks(a, b, iou=0.3 - 1e-9)
    assert loose["counts"]["tp"] == 3
    assert loose["matches"].set_index("spacr_id").loc[3, "matched_reference_id"] == 0


def test_ties_go_to_higher_iou_then_lower_label():
    a = np.zeros((10, 20), int)
    b = np.zeros((10, 20), int)
    _square(a, 1, 0, 4, 2, 8)
    _square(b, 9, 0, 0, 2, 8)
    _square(b, 5, 0, 8, 2, 8)
    pairs, _, _ = ops_decode.label_contingency(a, b)
    assert ops_decode.best_overlaps(pairs)["label_b"].tolist() == [5]
    _square(b, 9, 0, 0, 2, 7)
    _square(b, 0, 0, 0, 2, 1)
    pairs, _, _ = ops_decode.label_contingency(a, b)
    best = ops_decode.best_overlaps(pairs)
    assert best["label_b"].tolist() == [9]


def test_hungarian_matching_below_half_iou_is_one_to_one():
    pairs = pd.DataFrame({"label_a": [1, 1, 2], "label_b": [1, 2, 1], "iou": [0.4, 0.35, 0.39]})
    matched = ops_decode.one_to_one(pairs, threshold=0.3)
    assert sorted(zip(matched["label_a"], matched["label_b"])) == [(1, 2), (2, 1)]
    assert ops_decode.one_to_one(pairs, threshold=0.9).empty


def test_edge_objects_are_counted_apart():
    a = np.zeros((20, 20), int)
    _square(a, 1, 0, 0, 5, 5)
    _square(a, 2, 8, 8, 5, 5)
    b = a.copy()
    kept = ops_decode.match_masks(a, b, edge_policy="exclude")["counts"]
    assert (kept["tp"], kept["n_spacr"], kept["edge_spacr"], kept["edge_reference"]) == (1, 1, 1, 1)
    every = ops_decode.match_masks(a, b, edge_policy="keep")["counts"]
    assert every["tp"] == 2 and every["f1"] == 1.0
    with pytest.raises(ValueError):
        ops_decode.match_masks(a, b, edge_policy="drop")


def test_centroid_control_and_empty_images():
    a = np.zeros((20, 20), int)
    _square(a, 1, 4, 4, 4, 4)
    b = np.zeros((20, 20), int)
    _square(b, 3, 4, 5, 4, 4)
    found = ops_decode.centroid_matches(a, b, max_px=2)
    assert found["centroid_reference_id"].tolist() == [3]
    assert ops_decode.centroid_matches(a, b, max_px=0.5)["centroid_reference_id"].tolist() == [0]
    assert ops_decode.centroid_matches(a, np.zeros_like(b))["centroid_reference_id"].tolist() == [0]
    assert ops_decode.centroid_matches(np.zeros_like(a), b).empty


def test_reference_calls_parse_the_periscope_columns():
    table = pd.DataFrame({"Metadata_Foci_site": ["CP186A-Well1-2", "CP186A-Well1-2"],
                          "Metadata_Cells_ObjectNumber": [4, 5],
                          "Metadata_Foci_Barcode_MatchedTo_Barcode": ["AAA", ""],
                          "Metadata_Foci_Barcode_MatchedTo_GeneCode": ["G1", "G2"],
                          "Metadata_Foci_Cell_Quality": ["Perfect", "Great"]})
    calls = ops_decode.reference_cell_calls(table)
    assert calls[["plate", "well", "site", "object"]].values.tolist() == [["CP186A", "Well1", 2, 4],
                                                                          ["CP186A", "Well1", 2, 5]]
    assert calls["barcode"].tolist() == ["AAA", None]
    mapped = ops_decode.reference_cell_calls(
        table.assign(P="X", W="A1", S=3),
        {"site": "S", "plate": "P", "well": "W", "object": "Metadata_Cells_ObjectNumber",
         "barcode": "Metadata_Foci_Barcode_MatchedTo_Barcode",
         "gene": "Metadata_Foci_Barcode_MatchedTo_GeneCode"})
    assert mapped["site"].tolist() == [3, 3] and set(mapped["quality"]) == {"all"}
    with pytest.raises(ValueError):
        ops_decode.reference_cell_calls(table.drop(columns="Metadata_Cells_ObjectNumber"))


def test_reads_become_one_call_per_cell_and_ties_are_no_call():
    reads = pd.DataFrame({"well": "A1", "tile": 1,
                          "cell": [1, 1, 1, 2, 2, 0, 3],
                          "barcode": ["AA", "AA", "CC", "GG", "TT", "AA", "CC"]})
    calls = ops_decode.cell_calls_from_reads(reads).set_index("cell")
    assert calls.loc[1, "barcode"] == "AA" and calls.loc[1, "reads"] == 2 and calls.loc[1, "total_reads"] == 3
    assert calls.loc[2, "barcode"] is None
    assert calls.loc[3, "barcode"] == "CC"
    assert 0 not in calls.index


@pytest.mark.parametrize("args, expected", [
    (("A", "g1", "A", "g1"), "same_guide"),
    (("A", "g1", "B", "g1"), "same_gene_other_guide"),
    (("A", "g1", "B", "g2"), "disagree"),
    (("A", "g1", None, None), "spacr_only"),
    ((None, None, "B", "g2"), "reference_only"),
    ((None, None, None, None), "neither"),
])
def test_concordance_categories(args, expected):
    assert ops_decode.concordance_category(*args) == expected


def test_concordance_table_and_shuffle_keep_sites_apart():
    pairs = pd.DataFrame({"spacr_guide": ["A", "B", None], "spacr_gene": ["g", "h", None],
                          "reference_guide": ["A", "C", "D"], "reference_gene": ["g", "k", "d"],
                          "quality": ["Perfect", "Perfect", "Great"]})
    table = ops_decode.call_concordance(pairs)
    row = table[(table["quality"] == "all") & (table["category"] == "both_called")].iloc[0]
    assert row["n"] == 2 and row["pct"] == 50.0
    assert table[(table["quality"] == "Great") & (table["category"] == "reference_only")]["n"].item() == 1
    calls = pd.DataFrame({"plate": "P", "well": "W", "site": [1, 1, 1, 2],
                          "object": [1, 2, 3, 1], "barcode": ["a", "b", "c", "z"],
                          "gene": ["a", "b", "c", "z"], "quality": "all"})
    shuffled = ops_decode.shuffle_calls_within_site(calls, seed=0)
    assert sorted(shuffled.loc[shuffled["site"] == 1, "barcode"]) == ["a", "b", "c"]
    assert shuffled.loc[shuffled["site"] == 2, "barcode"].tolist() == ["z"]
    assert shuffled.equals(ops_decode.shuffle_calls_within_site(calls, seed=0))


def _plate(tmp_path):
    """Two sites of reference TIFF masks, spaCR npz masks and both call tables."""
    rng = np.random.default_rng(3)
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    rows, spacr_rows = [], []
    for site in (1, 2):
        cells = np.zeros((60, 60), np.uint16)
        nuclei = np.zeros((60, 60), np.uint16)
        for k in range(9):
            y, x = 3 + (k // 3) * 19, 3 + (k % 3) * 19
            _square(cells, k + 1, y, x, 15, 15)
            _square(nuclei, k + 1, y + 4, x + 4, 7, 7)
            gene = f"G{k % 4}"
            rows.append({"Metadata_Foci_site": f"P1-Well1-{site}", "Metadata_Cells_ObjectNumber": k + 1,
                         "Metadata_Foci_Barcode_MatchedTo_Barcode": f"{gene}_g{k % 2}",
                         "Metadata_Foci_Barcode_MatchedTo_GeneCode": gene,
                         "Metadata_Foci_Cell_Quality": "Perfect" if k % 3 else "Great"})
        for obj, mask in (("Cells", cells), ("Nuclei", nuclei)):
            tifffile.imwrite(ref_dir / f"Plate_P1_Well_Well1_Site_{site}_{obj}_Objects.tiff", mask)
        mine = np.roll(cells.astype(np.uint32) * 0 + np.where(cells > 0, 10 + cells, 0), 1, axis=1)
        mine_nuclei = np.where(nuclei > 0, 10 + nuclei, 0).astype(np.uint32)
        folder = tmp_path / "run" / "Well1" / "phenotype_masks"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f"site-{site}.npz", cells=mine, nuclei=mine_nuclei)
        for k in range(9):
            gene = f"G{k % 4}"
            guide = f"{gene}_g{k % 2}" if k != 4 else f"{gene}_g9"
            spacr_rows.append({"plate": "P1", "well": "Well1", "site": site, "label": 11 + k,
                               "sgRNA": guide if k != 8 else None, "gene": gene})
    rows.append({"Metadata_Foci_site": "P1-Well1-1", "Metadata_Cells_ObjectNumber": 99,
                 "Metadata_Foci_Barcode_MatchedTo_Barcode": "X", "Metadata_Foci_Barcode_MatchedTo_GeneCode": "X",
                 "Metadata_Foci_Cell_Quality": "Perfect"})
    write_table(pd.DataFrame(rows), str(tmp_path / "ref_calls.csv"), canonicalise=False)
    write_table(pd.DataFrame(spacr_rows), str(tmp_path / "run" / "ops_cells_P1.parquet"), canonicalise=False)
    return {"dst_root": str(tmp_path / "run"), "plate": "P1",
            "ops_compare_match": "mask_overlap",
            "ops_compare_reference_masks": str(ref_dir / "Plate_{plate}_Well_{well}_Site_{site}_{object}_Objects.tiff"),
            "ops_compare_reference_calls": str(tmp_path / "ref_calls.csv")}


def test_mask_overlap_phase_end_to_end(tmp_path):
    settings = _plate(tmp_path)
    out = run_ops(settings, phases=["compare_calls"])["compare_calls"]
    assert out["fields"] == 2 and out["reference_ids_unresolved"] == 1
    folder = os.path.join(settings["dst_root"], "results", "ops_compare")
    agreement = read_table(os.path.join(folder, "mask_agreement.csv"), canonicalise=False, report=None)
    whole = agreement[(agreement["site"] == "all") & (agreement["object"] == "Cells")].iloc[0]
    assert (whole["tp"], whole["fp"], whole["fn"]) == (18, 0, 0)
    assert whole["median_iou"] == pytest.approx(14 / 16)
    nuclei = agreement[(agreement["site"] == "all") & (agreement["object"] == "Nuclei")].iloc[0]
    assert nuclei["f1"] == 1.0 and nuclei["median_iou"] == 1.0
    concord = read_table(os.path.join(folder, "call_concordance.csv"), canonicalise=False, report=None)
    observed = concord[(concord["control"] == "observed") & (concord["quality"] == "all")].set_index("category")
    assert observed.loc["same_guide", "n"] == 14
    assert observed.loc["same_gene_other_guide", "n"] == 2
    assert observed.loc["reference_only", "n"] == 2
    assert out["same_guide_pct_of_both_called"]["observed"] == pytest.approx(100 * 14 / 16)
    assert "shuffled_seed0" in set(concord["control"])
    matches = read_table(os.path.join(folder, "match_cells.parquet"), canonicalise=False, report=None)
    assert len(matches) == 36 and matches["matched"].all()
    centroid = read_table(os.path.join(folder, "centroid_vs_overlap.csv"), canonicalise=False, report=None)
    assert centroid["same_reference_object"].sum() == 36
    with open(os.path.join(folder, "compare_masks.json"), encoding="utf-8") as handle:
        assert "same IoU threshold" in json.load(handle)["matching"]
    again = run_ops(settings, phases=["compare_calls"])["compare_calls"]
    assert again["agreement"] == out["agreement"]


def test_self_match_control_is_perfect(tmp_path):
    settings = _plate(tmp_path)
    settings.update({"ops_compare_spacr_masks": settings["ops_compare_reference_masks"],
                     "ops_compare_spacr_calls": "reference", "ops_compare_edge_policy": "keep"})
    out = run_ops(settings, phases=["compare_calls"])["compare_calls"]
    assert all(r["f1"] == 1.0 and r["median_iou"] == 1.0 for r in out["agreement"])
    assert out["same_guide_pct_of_both_called"]["observed"] == 100.0
    assert out["same_guide_pct_of_both_called"]["shuffled_seed0"] < 100.0


def test_mask_overlap_refuses_what_it_cannot_compare(tmp_path):
    settings = _plate(tmp_path)
    with pytest.raises(ValueError, match="runs on its own"):
        run_ops(settings, phases=["compare_calls", "decode"])
    with pytest.raises(ValueError, match="dst_root"):
        ops_decode.run_compare_masks({**settings, "dst_root": ""})
    with pytest.raises(ValueError, match="ops_compare_reference_masks"):
        ops_decode.run_compare_masks({**settings, "ops_compare_reference_masks": None})
    with pytest.raises(ValueError, match="ops_compare_fields"):
        ops_decode.run_compare_masks({**settings, "ops_compare_reference_calls": None})
    with pytest.raises(ValueError, match="needs"):
        ops_decode.run_compare_masks({**settings, "ops_compare_fields": "Well1:7"})
    small = tmp_path / "small"
    (small / "Well1" / "phenotype_masks").mkdir(parents=True)
    for site in (1, 2):
        np.savez_compressed(small / "Well1" / "phenotype_masks" / f"site-{site}.npz",
                            cells=np.ones((30, 30), np.uint32), nuclei=np.ones((30, 30), np.uint32))
    with pytest.raises(ValueError, match="ops_compare_mask_scale"):
        ops_decode.run_compare_masks({**settings, "dst_root": str(small)})
    scaled = ops_decode.run_compare_masks({**settings, "dst_root": str(small), "ops_compare_mask_scale": 2,
                                           "ops_compare_spacr_calls": None, "ops_compare_edge_policy": "keep"})
    assert scaled["agreement"][0]["n_spacr"] == 2 and "call_concordance" not in scaled["paths"]
    with pytest.raises(ValueError, match="not"):
        ops_decode.run_compare_masks({**settings, "dst_root": str(small), "ops_compare_mask_scale": 3})
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, other=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="holds no"):
        ops_decode._read_labels(str(bad), "Cells")
    write_table(pd.DataFrame({"well": ["Well1"], "site": [1]}), str(tmp_path / "x.csv"), canonicalise=False)
    with pytest.raises(ValueError, match="sgRNA"):
        ops_decode._spacr_cell_calls(str(tmp_path / "x.csv"), "P1")


def test_registered_as_alpha_922():
    keys = ALPHA_FEATURES[922]["settings"]
    assert "ops_compare_match" in keys and "ops_compare_reference_masks" in keys
    from spacr.ops_settings import ops_defaults

    defaults = ops_defaults({})
    assert defaults["ops_compare_match"] == "centroid"
    assert defaults["ops_compare_iou"] == 0.5
    assert defaults["ops_compare_reference_objects"] == ["Cells", "Nuclei"]

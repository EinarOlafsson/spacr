"""Item 1024 (alpha): nucleus fragments and cell-less nuclei in the one pairing.

Both options live in the shared ``spacr.measure._assign_nuclei_to_cells``;
Measure, OPS, the large-site preset and the time-lapse summary reach them
through it. With both off (the default) the pairing is exactly what it was.

Second build (2026-11-25), after the paper session evaluated the first on
LaC024A A1: union solidity 0.94 by default, pairwise-only merging (no chain
merges), the contact, smoothed-valley and relative-area tests (off by
default), binucleate cells keep both nuclei and are flagged, a nucleus given
to the nearest cell is bridged to it, and the small-piece merge of the
large-site preset runs through the same routine.
"""
import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
from skimage.measure import label as connected

from spacr import measure, ops_sample
from spacr.measure import (_assign_nuclei_to_cells, _fragment_pair_test,
                           _nucleus_pairing_active, _nucleus_pairing_options)

ROOT = Path(__file__).resolve().parents[1]

NAMES = ("nucleus_fragment_merge", "nucleus_fragment_max_gap_px",
         "nucleus_fragment_min_solidity", "nucleus_fragment_max_dip",
         "nucleus_fragment_min_contact_px", "nucleus_fragment_max_valley",
         "nucleus_fragment_max_area_rel", "cellless_nuclei",
         "cellless_nucleus_max_distance_px")


def _disc(shape, centre, radius):
    rows, cols = np.ogrid[:shape[0], :shape[1]]
    return (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2 <= radius ** 2


def _field():
    """Four cells and three nuclei without one.

    cell 1: one round nucleus cut in two touching pieces (1 and 2);
    cell 2: two separate round nuclei (3 and 4);
    cell 3: one plain nucleus (5);
    cell 4: two round nuclei that touch side by side (8 and 9);
    nucleus 6: two pixels right of cell 3 (a straight bridge);
    nucleus 10: diagonally below cell 3 (a bridge with corners);
    nucleus 7: far from every cell.
    """
    shape = (90, 190)
    cells = np.zeros(shape, np.int32)
    nuclei = np.zeros(shape, np.int32)
    dna = np.zeros(shape, np.float32)
    columns = np.arange(shape[1])[None, :]
    cells[5:45, 5:45] = 1
    whole = _disc(shape, (25, 25), 9)
    nuclei[whole & (columns <= 25)] = 1
    nuclei[whole & (columns > 25)] = 2
    dna[whole] = 100.0
    cells[5:45, 50:100] = 2
    for label, centre in ((3, (25, 63)), (4, (25, 87))):
        disc = _disc(shape, centre, 7)
        nuclei[disc] = label
        dna[disc] = 100.0
    cells[50:75, 5:45] = 3
    nuclei[_disc(shape, (62, 25), 6)] = 5
    dna[nuclei == 5] = 100.0
    nuclei[58:66, 47:53] = 6
    nuclei[77:81, 47:51] = 10
    nuclei[60:68, 120:128] = 7
    cells[5:45, 140:185] = 4
    for label, centre in ((8, (25, 154)), (9, (25, 164))):
        disc = _disc(shape, centre, 9) & (nuclei == 0)
        nuclei[disc] = label
        dna[disc] = 100.0
    return nuclei, cells, dna


def _on(**changes):
    options = _nucleus_pairing_options()
    options.update(changes)
    return options


def _run(options, nuclei=None, cells=None, dna=None):
    field = _field()
    nuclei = field[0] if nuclei is None else nuclei
    cells = field[1] if cells is None else cells
    dna = field[2] if dna is None else dna
    record = {}
    n_out, c_out, *_ = _assign_nuclei_to_cells(
        nuclei.copy(), cells.copy(), options=options, dna=dna, record=record)
    return n_out, c_out, record


# -- default-off identity ---------------------------------------------------

def test_defaults_are_off_and_the_rule_takes_the_paper_values():
    options = _nucleus_pairing_options()
    assert options["nucleus_fragment_merge"] == "off"
    assert options["cellless_nuclei"] == "drop"
    assert options["nucleus_fragment_min_solidity"] == 0.94
    assert options["nucleus_fragment_max_dip"] == 0.2
    assert options["nucleus_fragment_max_gap_px"] == 2.0
    assert options["nucleus_fragment_min_contact_px"] == 0.0
    assert options["nucleus_fragment_max_valley"] is None
    assert options["nucleus_fragment_max_area_rel"] is None
    assert options["nucleus_small_piece_frac"] == 0.0
    assert not _nucleus_pairing_active(options)
    assert not _nucleus_pairing_active(None)
    assert _nucleus_pairing_options({"ops_ph_cellless_nuclei": "nucleus_only"},
                                    prefix="ops_ph_")["cellless_nuclei"] == "nucleus_only"
    assert _nucleus_pairing_options(
        {"nucleus_fragment_max_valley": "None"})["nucleus_fragment_max_valley"] is None
    assert _nucleus_pairing_options(
        {"nucleus_fragment_max_area_rel": 2.5})["nucleus_fragment_max_area_rel"] == 2.5
    with pytest.raises(ValueError):
        _nucleus_pairing_options({"cellless_nuclei": "keep"})


def test_default_options_give_the_base_pairing_byte_for_byte():
    nuclei, cells, dna = _field()
    base = _assign_nuclei_to_cells(nuclei.copy(), cells.copy())
    off = _assign_nuclei_to_cells(nuclei.copy(), cells.copy(),
                                  options=_nucleus_pairing_options(), dna=dna)
    recorded = _assign_nuclei_to_cells(nuclei.copy(), cells.copy(),
                                       options=_nucleus_pairing_options(),
                                       dna=dna, record={})
    for other in (off, recorded):
        for a, b in zip(base[:2], other[:2]):
            assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
        assert base[2:] == other[2:]


def test_ops_default_pairing_is_unchanged():
    nuclei, cells, _ = _field()
    public = ops_sample.pair_nuclei_with_cells(nuclei, cells)
    private = ops_sample._pair_nuclei_with_cells(
        nuclei, cells, _nucleus_pairing_options())
    assert private[3] is None
    for a, b in zip(public[:2], private[:2]):
        assert a.tobytes() == b.tobytes()
    assert public[2] == private[2]
    # the status quo: extra nuclei and the three cell-less nuclei dropped
    assert public[2]["paired"] == 4
    assert public[2]["extra_nuclei_in_cell"] == 3
    assert public[2]["nuclei_without_cell"] == 3
    assert set(public[2]) == {"nuclei", "cells", "paired",
                              "nuclei_without_cell", "cells_without_nucleus",
                              "extra_nuclei_in_cell", "merged_cells"}


def test_ops_segment_field_default_does_not_take_the_alpha_path():
    from spacr import ops_engine
    from spacr.ops_settings import ops_defaults

    assert ops_engine._ph_pairing_options(ops_defaults()) is None
    assert ops_engine._ph_pairing_options(
        {"ops_ph_nucleus_fragment_merge": "rule"})["nucleus_fragment_merge"] == "rule"
    picked = ops_engine._ph_pairing_options(
        {"ops_ph_nucleus_fragment_merge": "rule",
         "ops_ph_nucleus_fragment_min_contact_px": 3,
         "ops_ph_nucleus_fragment_max_valley": 0.3,
         "ops_ph_nucleus_fragment_max_area_rel": 2.5})
    assert picked["nucleus_fragment_min_contact_px"] == 3.0
    assert picked["nucleus_fragment_max_valley"] == 0.3
    assert picked["nucleus_fragment_max_area_rel"] == 2.5


def test_the_large_site_preset_default_is_unchanged():
    from spacr import ops_large_sites as ols

    assert ols._site_pairing_options(ols.large_site_settings({})) is None
    nuclei, cells, _ = _field()
    public = ops_sample.pair_nuclei_with_cells(nuclei, cells)
    routed = ols._site_pairing(nuclei, cells, None, None)
    for a, b in zip(public, routed):
        if isinstance(a, np.ndarray):
            assert a.tobytes() == b.tobytes()
        else:
            assert a == b


# -- fragment merge ---------------------------------------------------------

def test_merge_joins_a_cut_nucleus_and_keeps_two_true_nuclei():
    nuclei, _cells, _dna = _field()
    out, _c, record = _run(_on(nucleus_fragment_merge="rule"))
    assert record["fragments"] == [(1, 2)]
    # cell 2 (far apart) and cell 4 (touching, a waisted union) stay binucleate
    assert record["binucleate"] == [2, 4]
    assert record["nuclei_per_cell"][1] == 1
    assert record["nuclei_per_cell"][2] == record["nuclei_per_cell"][4] == 2
    assert not (out == 2).any()
    assert (out == 1).sum() == ((nuclei == 1) | (nuclei == 2)).sum()
    for label in (3, 4, 8, 9):
        assert (out == label).any()


def test_two_round_nuclei_side_by_side_fail_the_solidity_test():
    nuclei, _cells, dna = _field()
    eight, nine = nuclei == 8, nuclei == 9
    assert not (eight & nine).any()
    passes, solidity = _fragment_pair_test(eight, nine, dna, _on())
    assert not passes and solidity < 0.94
    # the first build's 0.9 would have merged them
    assert solidity >= 0.9


def test_dip_refuses_two_touching_pieces_with_a_dark_seam():
    nuclei, cells, dna = _field()
    dna[:, 24:27] = 5.0
    _n, _c, record = _run(_on(nucleus_fragment_merge="rule"), dna=dna)
    assert record["fragments"] == []
    assert 1 in record["binucleate"]
    record = {}
    _assign_nuclei_to_cells(nuclei.copy(), cells.copy(),
                            options=_on(nucleus_fragment_merge="rule"),
                            record=record)
    assert record["fragments"] == [(1, 2)]


def test_gap_threshold():
    nuclei, cells, dna = _field()
    nuclei[nuclei == 1] = 0
    nuclei[(_disc(nuclei.shape, (25, 25), 9)) & (np.arange(nuclei.shape[1])[None, :] < 25)] = 1
    loose = _on(nucleus_fragment_merge="rule", nucleus_fragment_min_solidity=0.0)
    assert (1, 2) not in _run(dict(loose, nucleus_fragment_max_gap_px=0.0),
                              nuclei=nuclei)[2]["fragments"]
    assert (1, 2) in _run(dict(loose, nucleus_fragment_max_gap_px=2.0),
                          nuclei=nuclei)[2]["fragments"]


def test_contact_counts_the_pixels_along_the_cut():
    edge = np.zeros((20, 20), bool)
    edge[5:15, 2:10] = True
    other = np.zeros_like(edge)
    other[5:15, 10:18] = True
    corner = np.zeros_like(edge)
    corner[15:19, 10:14] = True
    contact = _on(nucleus_fragment_min_contact_px=5.0, nucleus_fragment_min_solidity=0.0)
    assert _fragment_pair_test(edge, other, None, contact)[0]
    assert not _fragment_pair_test(edge, corner, None, contact)[0]
    assert _fragment_pair_test(edge, corner, None,
                               dict(contact, nucleus_fragment_min_contact_px=0.0))[0]


def test_smoothed_valley_sits_next_to_the_raw_dip():
    first = np.zeros((20, 30), bool)
    first[5:15, 3:15] = True
    second = np.zeros_like(first)
    second[5:15, 15:27] = True
    flat = np.full(first.shape, 100.0, np.float32)
    band = flat.copy()
    band[:, 13:18] = 60.0
    from scipy.ndimage import uniform_filter

    valley = _on(nucleus_fragment_max_dip=1.0, nucleus_fragment_max_valley=0.2)
    for stain, passes in ((flat, True), (band, False)):
        smoothed = uniform_filter(stain, size=3)
        assert _fragment_pair_test(first, second, stain, valley,
                                   smoothed=smoothed)[0] is passes
    # off (None) by default: the same band passes on the raw dip of 1.0
    assert _fragment_pair_test(first, second, band,
                               _on(nucleus_fragment_max_dip=1.0),
                               smoothed=uniform_filter(band, size=3))[0]
    # and the raw dip, kept as built, refuses it at its default 0.2
    assert not _fragment_pair_test(first, second, band, _on())[0]


def test_merging_is_pairwise_never_by_a_chain():
    """Three pieces in a row: A+B and B+C each pass, A+B+C is too large.

    The first build joined every passing pair by union-find and merged all
    three; now A and B merge, the union is re-tested against C and fails,
    so C stays a nucleus of its own."""
    nuclei = np.zeros((40, 90), np.int32)
    cells = np.zeros_like(nuclei)
    cells[2:20, 2:40] = 1
    nuclei[5:15, 5:15] = 1
    nuclei[5:15, 15:25] = 2
    nuclei[5:15, 25:35] = 3
    cells[22:38, 2:40] = 2
    nuclei[25:35, 5:15] = 4
    cells[2:20, 45:85] = 3
    nuclei[5:15, 50:60] = 5
    dna = (nuclei > 0).astype(np.float32) * 100.0
    options = _on(nucleus_fragment_merge="rule", nucleus_fragment_max_area_rel=2.5)
    out, _c, record = _run(options, nuclei=nuclei, cells=cells, dna=dna)
    assert record["fragments"] == [(1, 2)]
    assert record["binucleate"] == [1]
    assert (out == 3).sum() == 100
    # with the area test off all three pass pairwise as one rectangle
    _o, _c, record = _run(_on(nucleus_fragment_merge="rule"),
                          nuclei=nuclei, cells=cells, dna=dna)
    assert record["fragments"] == [(1, 2), (1, 3)]


def test_without_cells_the_rule_merges_neighbouring_pieces_only():
    nuclei, _cells, dna = _field()
    record = {}
    out = _assign_nuclei_to_cells(nuclei.copy(), None,
                                  options=_on(nucleus_fragment_merge="rule"),
                                  dna=dna, record=record)[0]
    assert record["fragments"] == [(1, 2)]
    assert (out == 3).any() and (out == 4).any() and (out == 8).any()


# -- cell-less nuclei -------------------------------------------------------

def test_nearest_cell_attaches_within_reach_and_bridges_the_gap():
    nuclei, cells, _dna = _field()
    n_out, c_out, record = _run(_on(cellless_nuclei="nearest_cell"))
    assert record["attached"] == {6: 3, 10: 3}
    assert record["cellless"] == [7]
    assert record["bridged"] > 0
    assert (c_out[n_out == 6] == 3).all() and (c_out[n_out == 10] == 3).all()
    assert (n_out == nuclei).all()
    # the grown cell is ONE object, 4-connected, not a cell plus islands
    assert connected(c_out == 3, connectivity=1).max() == 1
    # no other cell lost a pixel
    for label in (1, 2, 4):
        assert ((c_out == label) == (cells == label)).all()


def test_nucleus_only_keeps_cellless_nuclei_in_ops():
    nuclei, cells, dna = _field()
    n_out, c_out, counts, record = ops_sample._pair_nuclei_with_cells(
        nuclei, cells, _on(cellless_nuclei="nucleus_only"), dna=dna)
    assert counts["paired"] == 4
    assert counts["nucleus_only"] == 3
    assert record["nucleus_only"] == [5, 6, 7]
    for key in record["nucleus_only"]:
        assert (n_out == key).any()
        assert not (c_out == key).any()
    paired = (n_out > 0) & (n_out <= 4)
    assert (n_out[paired] == c_out[paired]).all()


def test_nearest_cell_in_ops_gives_the_nucleus_a_cell():
    nuclei, cells, dna = _field()
    _n, _c, counts, _record = ops_sample._pair_nuclei_with_cells(
        nuclei, cells, _on(cellless_nuclei="nearest_cell"), dna=dna)
    assert counts["attached_cellless"] == 2
    assert counts["nuclei_without_cell"] == 1


# -- binucleate cells and OPS keying ----------------------------------------

def test_ops_keeps_both_nuclei_of_a_binucleate_cell():
    nuclei, cells, dna = _field()
    n_out, c_out, counts, record = ops_sample._pair_nuclei_with_cells(
        nuclei, cells, _on(nucleus_fragment_merge="rule"), dna=dna)
    assert counts["paired"] == 4
    assert counts["binucleate_cells"] == 2
    assert counts["binucleate_kept"] == 2
    # the paired keys are those of the default pairing
    base_n, base_c, _ = ops_sample.pair_nuclei_with_cells(nuclei, cells)
    assert (c_out == base_c).all()
    seconds = dict(record["second_nuclei"])
    assert sorted(seconds) == [5, 6]
    for key, cell in seconds.items():
        assert (n_out == key).any()
        assert (c_out[n_out == key] == cell).all()
        assert (n_out == cell).any()
    # cells 2 and 4 of the input carry the two binucleate keys
    assert set(seconds.values()) == {int(base_c[25, 70]), int(base_c[25, 160])}


def test_ops_keys_a_merged_nucleus_by_its_lowest_piece():
    nuclei, cells, dna = _field()
    n_out, _c, counts, record = ops_sample._pair_nuclei_with_cells(
        nuclei, cells, _on(nucleus_fragment_merge="rule",
                           cellless_nuclei="nucleus_only"), dna=dna)
    assert counts["merged_fragments"] == 1
    (key, kept, piece), = record["fragments"]
    assert (kept, piece) == (1, 2)
    whole = (nuclei == 1) | (nuclei == 2)
    assert set(np.unique(n_out[whole])) == {key}
    rows, cols = np.nonzero(n_out == key)
    assert abs(rows.mean() - 25) < 0.6 and abs(cols.mean() - 25) < 0.6
    # paired 1..4, then the two second nuclei, then the nucleus-only ones
    assert [k for k, _ in record["second_nuclei"]] == [5, 6]
    assert record["nucleus_only"] == [7, 8, 9]


def test_ops_fragment_table_and_columns(tmp_path):
    import pandas as pd
    from spacr import ops_engine

    nuclei, cells, dna = _field()
    n_out, c_out, _counts, record = ops_sample._pair_nuclei_with_cells(
        nuclei, cells, _on(nucleus_fragment_merge="rule",
                           cellless_nuclei="nucleus_only"), dna=dna)
    path = ops_engine._ph_fragments_path(str(tmp_path), 3)
    ops_engine._ph_write_fragments(path, "A1", 3, record)
    stored = pd.read_parquet(path)
    assert list(stored.columns) == ["well", "site", "label", "kept_label",
                                    "merged_label", "cell_label"]
    labels = np.unique(n_out[n_out > 0])
    table = ops_engine._ph_pairing_columns(
        pd.DataFrame({"label": labels}), c_out, path)
    row = table.set_index("label")
    merged_key = record["fragments"][0][0]
    assert row.loc[merged_key, "nucleus_fragment_count"] == 2
    assert bool(row.loc[merged_key, "nucleus_merged"])
    assert row.loc[merged_key, "nucleus_count"] == 1
    for key, cell in record["second_nuclei"]:
        assert row.loc[key, "cell_label"] == cell
        assert bool(row.loc[key, "has_cell"])
        assert row.loc[key, "nucleus_count"] == 2 and bool(row.loc[key, "binucleate"])
        assert row.loc[cell, "nucleus_count"] == 2 and bool(row.loc[cell, "binucleate"])
    for key in record["nucleus_only"]:
        assert not bool(row.loc[key, "has_cell"])
        assert row.loc[key, "cell_label"] == -1
        assert row.loc[key, "nucleus_count"] == 0
    gathered = ops_engine._ph_collect_fragments(str(tmp_path), str(tmp_path),
                                                "A1", [3, 4])
    assert len(pd.read_parquet(gathered)) == len(stored)


def test_nucleus_only_and_second_nuclei_measure_with_empty_cell_columns():
    nuclei, cells, dna = _field()
    n_out, c_out, _counts, record = ops_sample._pair_nuclei_with_cells(
        nuclei, cells, _on(nucleus_fragment_merge="rule",
                           cellless_nuclei="nucleus_only"), dna=dna)
    image = np.stack([dna, dna], axis=-1)
    table = ops_sample.measure_phenotype_field(n_out, c_out, image,
                                               ["dapi", "gfp"], texture=False)
    extra = [k for k, _ in record["second_nuclei"]] + record["nucleus_only"]
    alone = table[table["label"].isin(extra)]
    assert len(alone) == len(extra)
    assert alone["nucleus_area"].notna().all()
    assert alone["cell_area"].isna().all()
    assert alone[["centroid_row", "centroid_col"]].notna().all().all()


# -- point 8: can a nucleus-only object receive SBS reads? ------------------

def test_a_nucleus_only_object_can_receive_sbs_reads_through_the_link():
    """Reads are never attributed to phenotype masks: the decode gives them
    to the sequencing-side objects (``ops_objects``, by
    :func:`spacr.ops_sbs.attribute_reads`), and the merge phase links each
    phenotype row to one of those objects by its NUCLEUS centroid
    (``centroid_row``/``centroid_col``, :func:`spacr.ops_merge.merge_field`).
    A nucleus-only row has that centroid, so it links and inherits the
    object's barcode like any paired cell."""
    from spacr.ops_merge import merge_field
    from spacr.ops_sbs import assign_reads_to_objects, attribute_reads

    nuclei, cells, dna = _field()
    n_out, c_out, _counts, record = ops_sample._pair_nuclei_with_cells(
        nuclei, cells, _on(cellless_nuclei="nucleus_only"), dna=dna)
    table = ops_sample.measure_phenotype_field(
        n_out, c_out, np.stack([dna, dna], axis=-1), ["dapi", "gfp"],
        texture=False)
    key = record["nucleus_only"][-1]
    assert not (c_out == key).any()
    row = table.set_index("label").loc[key]
    centre = np.array([[row["centroid_row"], row["centroid_col"]]])
    # one sequencing nucleus under it (identity transform) and four reads
    objects = centre + 0.4
    owner, _ambiguous = attribute_reads(centre + [[1, 0], [0, 1], [-1, 0], [0, -1]],
                                        objects, np.array([60.0]))
    assert (owner == 0).all()
    calls = assign_reads_to_objects(owner + 1, ["ACGT"] * 4)
    assert calls[1]["barcode"] == "ACGT"
    identity = (1.0, np.eye(2), np.zeros(2))
    pairs = merge_field(centre, objects, identity, threshold=4.0)
    assert pairs.shape[0] == 1 and int(pairs[0, 1]) == 0


# -- Measure: one table column set, off by default --------------------------

def test_measure_flags_only_when_on():
    import pandas as pd

    frame = pd.DataFrame({"label": [1, 3, 7]})
    morphology = {"nucleus": frame.copy(), "cell": frame.copy()}
    measure._flag_shared_nuclei(morphology, {"nucleus": set(), "cell": set()})
    assert "nucleus_merged" not in morphology["nucleus"]
    assert "binucleate" not in morphology["cell"]
    morphology = {"nucleus": frame.copy(), "cell": pd.DataFrame({"label": [1, 2]})}
    measure._flag_shared_nuclei(morphology, {
        "nucleus": set(), "cell": set(),
        "pairing": {"fragments": [(1, 2)], "cellless": [7],
                    "nuclei_per_cell": {1: 1, 2: 2}}})
    out = morphology["nucleus"].set_index("label")
    assert out.loc[1, "nucleus_fragment_count"] == 2
    assert bool(out.loc[1, "nucleus_merged"])
    assert not bool(out.loc[7, "has_cell"])
    assert bool(out.loc[3, "has_cell"])
    cell = morphology["cell"].set_index("label")
    assert cell.loc[2, "nucleus_count"] == 2 and bool(cell.loc[2, "binucleate"])
    assert cell.loc[1, "nucleus_count"] == 1 and not bool(cell.loc[1, "binucleate"])


# -- the time-lapse callers -------------------------------------------------

def test_time_lapse_passes_the_options_to_the_pairing():
    from spacr import timelapse

    source = inspect.getsource(timelapse._process_merged_group)
    assert "options=pairing" in source
    driver = inspect.getsource(timelapse.automated_motility_assay)
    assert "_nucleus_pairing_options(settings)" in driver
    stack = np.arange(2 * 4 * 5 * 3, dtype=np.float32).reshape(2, 4, 5, 3)
    assert timelapse._pairing_frame_stain(stack, 1, 2).tobytes() == stack[1, ..., 2].tobytes()
    assert timelapse._pairing_frame_stain(stack, 0, None) is None
    assert timelapse._pairing_frame_stain(stack, 0, 7) is None


# -- one implementation -----------------------------------------------------

def _calls(module_path, function):
    tree = ast.parse((ROOT / module_path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == function)
    return {c.func.id if isinstance(c.func, ast.Name) else getattr(c.func, "attr", "")
            for c in ast.walk(node) if isinstance(c, ast.Call)}


def test_measure_ops_and_large_sites_call_the_same_pairing_function():
    assert "_assign_nuclei_to_cells" in _calls("spacr/measure.py", "_measure_crop_core")
    assert "_assign_nuclei_to_cells" in _calls("spacr/ops_sample.py",
                                               "_pair_nuclei_with_cells")
    assert "_pair_nuclei_with_cells" in _calls("spacr/ops_sample.py",
                                               "pair_nuclei_with_cells")
    assert "_pair_nuclei_with_cells" in _calls("spacr/ops_engine.py",
                                               "_ph_segment_field")
    assert "_assign_nuclei_to_cells" in _calls("spacr/ops_large_sites.py",
                                               "_pair_before_cells")
    assert "_pair_before_cells" in _calls("spacr/ops_large_sites.py",
                                          "merge_small_nuclei")
    assert "_assign_nuclei_to_cells" in _calls("spacr/timelapse.py",
                                               "_process_merged_group")
    # the merges and the cell-less handling exist only in measure.py
    for path in ("spacr/ops_sample.py", "spacr/ops_engine.py",
                 "spacr/ops_large_sites.py", "spacr/timelapse.py"):
        source = (ROOT / path).read_text(encoding="utf-8")
        for name in ("_merge_nucleus_fragments", "_attach_cellless_nuclei",
                     "_merge_small_nucleus_pieces", "_fragment_pair_test"):
            assert f"def {name}" not in source
    large = inspect.getsource(__import__("spacr.ops_large_sites",
                                         fromlist=["merge_small_nuclei"]).merge_small_nuclei)
    assert "lexsort" not in large and "bincount" not in large
    for name in ("_merge_nucleus_fragments", "_merge_small_nucleus_pieces",
                 "_attach_cellless_nuclei"):
        assert name in inspect.getsource(measure._assign_nuclei_to_cells)


def test_the_spanning_rule_is_untouched():
    """Einar decides about the spanning merge separately: item 1024 never
    changes it."""
    source = inspect.getsource(measure._merge_spanning_children)
    assert "max(percentages) >= 90" in source
    assert "options" not in inspect.signature(measure._merge_spanning_children).parameters


def test_settings_registered_off_with_schema_row():
    from spacr.settings import (ALPHA_FEATURES, DEFAULT_CHANGES,
                                SETTINGS_SCHEMA_VERSION, get_measure_crop_settings)
    from spacr.ops_settings import OPS_DEFAULTS, OPS_TOOLTIPS

    assert SETTINGS_SCHEMA_VERSION >= 967
    assert DEFAULT_CHANGES[967] == {
        "nucleus_fragment_merge": "off", "cellless_nuclei": "drop",
        "ops_ph_nucleus_fragment_merge": "off", "ops_ph_cellless_nuclei": "drop",
        "nucleus_fragment_min_contact_px": 0.0,
        "nucleus_fragment_max_valley": None,
        "nucleus_fragment_max_area_rel": None,
        "ops_ph_nucleus_fragment_min_contact_px": 0.0,
        "ops_ph_nucleus_fragment_max_valley": None,
        "ops_ph_nucleus_fragment_max_area_rel": None}
    assert tuple(measure._NUCLEUS_PAIRING_DEFAULTS) == NAMES
    measure_defaults = get_measure_crop_settings({})
    for name, default in measure._NUCLEUS_PAIRING_DEFAULTS.items():
        assert measure_defaults[name] == default
        assert OPS_DEFAULTS["ops_ph_" + name] == default
        tip = OPS_TOOLTIPS["ops_ph_" + name]
        assert len(tip) <= 600 and tip.endswith("Alpha.") and "Default" in tip
    assert len(ALPHA_FEATURES[1024]["settings"]) == 2 * len(NAMES)

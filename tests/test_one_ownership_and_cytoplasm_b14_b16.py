"""One ownership and one cytoplasm for every module (duplicate audit B14 and B16).

The nucleus and cell ownership rule (maintainer decision 2026-11-23: a nucleus
split under 90 % in each of several cells merges them; at 90 % or more its
pixels in the others are removed) and the cytoplasm (the cell minus the
nucleus and every other owned object) each have one implementation in
:mod:`spacr.measure`. These tests pin the rule on small fixtures and check
that every module reaches it through that one function.
"""
import ast
import pathlib

import numpy as np
import pandas as pd
import pytest

from spacr import measure as M

SPACR = pathlib.Path(__file__).resolve().parents[1] / "spacr"


def _calls(path, function):
    """The names called in one top-level function of a spaCR module."""
    tree = ast.parse((SPACR / path).read_text(encoding="utf-8"))
    node, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function]
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            names.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return names


def test_overlaps_handle_sparse_labels_empty_fields_and_shape_errors():
    child = np.zeros((4, 4), np.int64)
    parent = np.zeros((4, 4), np.int64)
    child[0, :2] = 10 ** 9
    parent[0, :3] = 7
    kids, parents, overlap = M._label_overlaps(child, parent)
    assert kids.tolist() == [10 ** 9] and parents.tolist() == [7] and overlap.tolist() == [2]
    empty = M._label_overlaps(np.zeros((3, 3), int), np.zeros((3, 3), int))
    assert all(a.size == 0 for a in empty)
    with pytest.raises(ValueError):
        M._label_overlaps(np.zeros((3, 3)), np.zeros((3, 4)))


def test_an_exact_tie_goes_to_the_lowest_parent():
    kids, parents, overlap = M._best_parent([1, 1, 2, 2], [9, 4, 5, 6], [3, 3, 1, 2])
    assert kids.tolist() == [1, 2] and parents.tolist() == [4, 6]


def _two_cells():
    cells = np.zeros((10, 20), np.int32)
    cells[:, :10] = 3
    cells[:, 10:] = 7
    return cells


def test_a_nucleus_under_90_percent_in_each_cell_merges_them():
    cells = _two_cells()
    nuclei = np.zeros_like(cells)
    nuclei[2:6, 6:14] = 1
    nuclei, cells, shared_n, shared_c, merged = M._assign_nuclei_to_cells(nuclei, cells)
    assert merged == 1 and set(np.unique(cells)) == {3}
    assert (nuclei == 1).sum() == 32 and not shared_n


def test_a_nucleus_at_90_percent_in_one_cell_loses_its_pixels_in_the_other():
    cells = _two_cells()
    nuclei = np.zeros_like(cells)
    nuclei[0:10, 0:10] = 1
    nuclei[0, 10] = 1
    nuclei, cells, _n, _c, merged = M._assign_nuclei_to_cells(nuclei, cells)
    assert merged == 0 and set(np.unique(cells)) == {3, 7}
    assert nuclei[0, 10] == 0 and (nuclei == 1).sum() == 100


def test_a_merge_is_seen_by_the_nuclei_examined_after_it():
    """Nucleus 1 (half in cell 3, half in 5) merges 5 into 3. Nucleus 2, 60 %
    in 3, 35 % in 5 and 5 % in 7, is then 95 % in the merged cell, so its
    pixel in 7 is removed and cell 7 survives: the objects are examined one
    at a time in label order, as the original loop did."""
    cells = np.zeros((20, 30), np.int32)
    cells[:, :10] = 3
    cells[:, 10:17] = 5
    cells[:, 17:] = 7
    nuclei = np.zeros_like(cells)
    nuclei[0:2, 8:12] = 1
    nuclei[10, 0:10] = 2
    nuclei[11, 8:18] = 2
    nuclei, cells, _n, _c, merged = M._assign_nuclei_to_cells(nuclei, cells)
    assert merged == 1 and set(np.unique(cells)) == {3, 7}
    assert nuclei[11, 17] == 0 and (nuclei == 2).sum() == 19


def test_disconnected_pieces_keep_only_the_piece_in_the_owning_cell():
    cells = _two_cells()
    nuclei = np.zeros_like(cells)
    nuclei[1:4, 1:4] = 5
    nuclei[1:3, 15:17] = 5
    nuclei, _cells, shared_n, shared_c, merged = M._assign_nuclei_to_cells(nuclei, cells)
    assert merged == 0 and shared_n == {5} and shared_c == {3, 7}
    assert (nuclei[:, 10:] == 0).all() and (nuclei == 5).sum() == 9


def test_child_to_parent_map_keeps_children_over_no_parent():
    child = np.zeros((5, 5), np.uint16)
    child[0, 0] = 2
    child[4, 4] = 3
    parent = np.zeros((5, 5), np.uint16)
    parent[0, :2] = 8
    frame = M._map_child_to_parent(child, parent)
    assert frame.to_dict("list") == {"organelle": [2, 3], "cell": [8, 0]}
    assert M._map_child_to_parent(np.zeros((2, 2), int), parent[:2, :2]).empty


def test_cytoplasm_excludes_every_owned_object_and_keeps_the_cell_dtype():
    cell = np.full((4, 4), 2, np.uint16)
    nucleus = np.zeros((4, 4), np.uint16)
    nucleus[0, 0] = 1
    pathogen = np.zeros((4, 4), np.int32)
    pathogen[1, 1] = 4
    near = np.zeros((4, 4), bool)
    near[2, 2] = True
    cyto = M._derive_cytoplasm(cell, [nucleus, None, pathogen, near])
    assert cyto.dtype == np.uint16 and int((cyto == 0).sum()) == 3
    assert np.array_equal(M._derive_cytoplasm(cell), cell)


def test_the_settings_check_cytoplasm_excludes_organelles_as_the_run_does(monkeypatch):
    data = np.zeros((6, 6, 4), np.int32)
    data[..., 0] = 1
    data[..., 1] = 1
    data[0, 0, 2] = 1
    data[5, 5, 3] = 1
    seen = {}

    def fake(masks, channels, settings, **kwargs):
        seen.update(masks)
        return {"morphology": {}, "intensity": {}}

    monkeypatch.setattr(M, "_measure_field_objects", fake)
    monkeypatch.setattr(M, "_field_role_tables", lambda *a, **k: {})
    M._field_tables_for_check(data, {"cell_mask_dim": 1, "nucleus_mask_dim": 2,
                                     "pathogen_mask_dim": None, "organelle_mask_dim": 3,
                                     "cytoplasm": True, "channels": [0]})
    assert seen["cytoplasm"][0, 0] == 0 and seen["cytoplasm"][5, 5] == 0
    assert int(seen["cytoplasm"].sum()) == 34


def test_ops_cytoplasm_excludes_the_fields_further_objects():
    from spacr import ops_sample

    cells = np.zeros((40, 40), np.int32)
    cells[5:35, 5:35] = 1
    nuclei = np.zeros_like(cells)
    nuclei[15:25, 15:25] = 1
    pathogen = np.zeros_like(cells)
    pathogen[6:10, 6:10] = 1
    rng = np.random.default_rng(3)
    image = rng.random((40, 40, 2)).astype(np.float32) * 100
    plain = ops_sample.measure_phenotype_field(nuclei, cells, image, ["a", "b"], texture=False)
    owned = ops_sample._measure_phenotype_field(nuclei, cells, image, ["a", "b"], texture=False,
                                                objects={"pathogen": pathogen})
    same = ops_sample._measure_phenotype_field(nuclei, cells, image, ["a", "b"], texture=False)
    pd.testing.assert_frame_equal(plain, same)
    assert plain["cytoplasm_area"].iloc[0] == 900 - 100
    assert owned["cytoplasm_area"].iloc[0] == 900 - 100 - 16
    assert owned["cell_area"].iloc[0] == plain["cell_area"].iloc[0]


def test_timelapse_children_go_to_one_parent():
    from spacr.timelapse import _compute_parent_child_overlaps

    cells = np.zeros((2, 6, 6), np.int32)
    cells[:, :, :3] = 1
    cells[:, :, 3:] = 2
    pathogen = np.zeros_like(cells)
    pathogen[1, 0, 1:5] = 9
    pathogen[1, 1, 1:3] = 9
    frame = _compute_parent_child_overlaps(cells, pathogen, "track_id", "pathogen_label")
    assert frame.to_dict("list") == {"frame": [1], "track_id": [1], "pathogen_label": [9]}


def test_majority_parent_breaks_an_exact_tie_to_the_lowest_label():
    from spacr.host_pathogen import _majority_parent

    a = np.array([[1, 1]])
    b = np.array([[6, 4]])
    assert _majority_parent(a, b, 1).tolist() == [0, 4]


@pytest.mark.parametrize("path, function, owner", [
    ("measure.py", "_measure_crop_core", "_assign_nuclei_to_cells"),
    ("measure.py", "_measure_crop_core", "_derive_cytoplasm"),
    ("measure.py", "_field_tables_for_check", "_derive_cytoplasm"),
    # 1024: pair_nuclei_with_cells delegates to _pair_nuclei_with_cells,
    # which also carries the alpha pairing options.
    ("ops_sample.py", "_pair_nuclei_with_cells", "_assign_nuclei_to_cells"),
    ("ops_sample.py", "_pair_nuclei_with_cells", "_best_parent"),
    # 1024: the small-piece merge of the large-site preset (921/965) runs
    # in the shared pairing, called without cells.
    ("ops_large_sites.py", "_pair_before_cells", "_assign_nuclei_to_cells"),
    ("ops_large_sites.py", "merge_small_nuclei", "_pair_before_cells"),
    ("ops_sample.py", "pair_nuclei_with_cells", "_pair_nuclei_with_cells"),
    ("ops_engine.py", "_ph_segment_field", "_pair_nuclei_with_cells"),
    ("ops_sample.py", "_measure_phenotype_field", "_derive_cytoplasm"),
    ("ops_engine.py", "_ph_merged_field", "_derive_cytoplasm"),
    ("ops_flatfield.py", "mask_pass", "_derive_cytoplasm"),
    ("ops_flatfield.py", "field_object_means", "_derive_cytoplasm"),
    ("timelapse.py", "_process_merged_group", "_assign_nuclei_to_cells"),
    ("timelapse.py", "_process_merged_group", "_derive_cytoplasm"),
    ("timelapse.py", "_compute_parent_child_overlaps", "_label_overlaps"),
    ("host_pathogen.py", "_majority_parent", "_best_parent"),
    ("host_pathogen.py", "vacuole_links", "_label_overlaps"),
    ("host_pathogen.py", "translocation_cells", "_derive_cytoplasm"),
    ("filters.py", "_object_overlap_rows", "_label_overlaps"),
    ("ops_decode.py", "label_contingency", "_label_overlaps"),
    ("utils.py", "_merge_overlapping_objects", "_merge_spanning_children"),
    ("utils.py", "_relabel_parent_with_child_labels", "_label_overlaps"),
    ("utils.py", "generate_cytoplasm_mask", "_derive_cytoplasm"),
])
def test_every_module_reaches_the_one_implementation(path, function, owner):
    assert owner in _calls(path, function)


def test_no_module_derives_a_cytoplasm_or_counts_overlaps_by_hand():
    """No ``np.where(<nucleus> > 0, 0, <cell>)`` and no ``np.unique(a * w + b)``
    pair encoding outside the owner. ``host_pathogen`` and ``qt`` boolean
    regions are other operations (audit B16)."""
    offenders = []
    for path in sorted(SPACR.rglob("*.py")):
        rel = path.relative_to(SPACR).as_posix()
        if rel == "measure.py" or rel.startswith("qt/"):
            continue
        text = path.read_text(encoding="utf-8")
        for needle in ("np.where(nuclei > 0, 0, cells)", "np.where(nucleus_mask != 0, 0, cell_mask)",
                       "cytoplasm_masks[nucleus_masks > 0] = 0"):
            if needle in text:
                offenders.append((rel, needle))
    assert not offenders, offenders

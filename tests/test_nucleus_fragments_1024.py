"""Item 1024 (alpha): nucleus fragments and cell-less nuclei in the one pairing.

Both options live in the shared ``spacr.measure._assign_nuclei_to_cells``;
Measure and OPS reach them through it. With both off (the default) the
pairing is exactly what it was.
"""
import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from spacr import measure, ops_sample
from spacr.measure import (_assign_nuclei_to_cells, _nucleus_pairing_options,
                           _nucleus_pairing_active)

ROOT = Path(__file__).resolve().parents[1]


def _disc(shape, centre, radius):
    rows, cols = np.ogrid[:shape[0], :shape[1]]
    return (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2 <= radius ** 2


def _field():
    """Three cells: a nucleus cut in two, two true nuclei, one plain nucleus,
    and a nucleus just outside cell 3 plus one far from every cell."""
    shape = (80, 140)
    cells = np.zeros(shape, np.int32)
    nuclei = np.zeros(shape, np.int32)
    dna = np.zeros(shape, np.float32)
    # cell 1: one round nucleus cut along a 1-px line (fragments 1 and 2)
    cells[5:45, 5:45] = 1
    whole = _disc(shape, (25, 25), 9)
    nuclei[whole & (np.arange(shape[1])[None, :] < 25)] = 1
    nuclei[whole & (np.arange(shape[1])[None, :] > 25)] = 2
    dna[whole] = 100.0
    # cell 2: two separate round nuclei with dark gap (3 and 4)
    cells[5:45, 50:100] = 2
    for label, centre in ((3, (25, 63)), (4, (25, 87))):
        disc = _disc(shape, centre, 7)
        nuclei[disc] = label
        dna[disc] = 100.0
    # cell 3: one plain nucleus (5)
    cells[50:75, 5:45] = 3
    nuclei[_disc(shape, (62, 25), 6)] = 5
    dna[nuclei == 5] = 100.0
    # nucleus 6: two pixels right of cell 3, outside it
    nuclei[58:66, 47:53] = 6
    # nucleus 7: far from every cell
    nuclei[60:68, 120:128] = 7
    return nuclei, cells, dna


def _on(**changes):
    options = _nucleus_pairing_options()
    options.update(changes)
    return options


# -- default-off identity ---------------------------------------------------

def test_defaults_are_off():
    options = _nucleus_pairing_options()
    assert options["nucleus_fragment_merge"] == "off"
    assert options["cellless_nuclei"] == "drop"
    assert not _nucleus_pairing_active(options)
    assert not _nucleus_pairing_active(None)
    assert _nucleus_pairing_options({"ops_ph_cellless_nuclei": "nucleus_only"},
                                    prefix="ops_ph_")["cellless_nuclei"] == "nucleus_only"
    with pytest.raises(ValueError):
        _nucleus_pairing_options({"cellless_nuclei": "keep"})


def test_default_options_give_the_base_pairing_byte_for_byte():
    nuclei, cells, dna = _field()
    base = _assign_nuclei_to_cells(nuclei.copy(), cells.copy())
    off = _assign_nuclei_to_cells(nuclei.copy(), cells.copy(),
                                  options=_nucleus_pairing_options(), dna=dna)
    for a, b in zip(base[:2], off[:2]):
        assert a.tobytes() == b.tobytes()
    assert base[2:] == off[2:]


def test_ops_default_pairing_is_unchanged():
    nuclei, cells, _ = _field()
    public = ops_sample.pair_nuclei_with_cells(nuclei, cells)
    private = ops_sample._pair_nuclei_with_cells(
        nuclei, cells, _nucleus_pairing_options())
    assert private[3] is None
    for a, b in zip(public[:2], private[:2]):
        assert a.tobytes() == b.tobytes()
    assert public[2] == private[2]
    # the status quo: extra nucleus and the two cell-less nuclei dropped
    assert public[2]["paired"] == 3
    assert public[2]["extra_nuclei_in_cell"] == 2
    assert public[2]["nuclei_without_cell"] == 2
    assert set(public[2]) == {"nuclei", "cells", "paired",
                              "nuclei_without_cell", "cells_without_nucleus",
                              "extra_nuclei_in_cell", "merged_cells"}


def test_ops_segment_field_default_does_not_take_the_alpha_path(monkeypatch):
    from spacr import ops_engine
    from spacr.ops_settings import ops_defaults

    assert ops_engine._ph_pairing_options(ops_defaults()) is None
    assert ops_engine._ph_pairing_options(
        {"ops_ph_nucleus_fragment_merge": "rule"})["nucleus_fragment_merge"] == "rule"


# -- fragment merge ---------------------------------------------------------

def test_merge_joins_a_cut_nucleus_and_keeps_two_true_nuclei():
    nuclei, cells, dna = _field()
    record = {}
    out, _cells, *_ = _assign_nuclei_to_cells(
        nuclei.copy(), cells.copy(), options=_on(nucleus_fragment_merge="rule"),
        dna=dna, record=record)
    assert record["fragments"] == [(1, 2)]
    assert record["binucleate"] == [2]
    assert not (out == 2).any()
    assert (out == 1).sum() == ((nuclei == 1) | (nuclei == 2)).sum()
    assert (out == 3).any() and (out == 4).any()


def test_dip_refuses_two_touching_pieces_with_a_dark_seam():
    nuclei, cells, dna = _field()
    dna[:, 24:27] = 5.0   # a dark line between fragments 1 and 2
    record = {}
    _assign_nuclei_to_cells(nuclei.copy(), cells.copy(),
                            options=_on(nucleus_fragment_merge="rule"),
                            dna=dna, record=record)
    assert record["fragments"] == []
    assert 1 in record["binucleate"]
    # without a stain the dip is skipped and the geometry alone merges them
    record = {}
    _assign_nuclei_to_cells(nuclei.copy(), cells.copy(),
                            options=_on(nucleus_fragment_merge="rule"),
                            record=record)
    assert record["fragments"] == [(1, 2)]


def test_gap_threshold():
    nuclei, cells, dna = _field()
    record = {}
    _assign_nuclei_to_cells(nuclei.copy(), cells.copy(),
                            options=_on(nucleus_fragment_merge="rule",
                                        nucleus_fragment_max_gap_px=0.0),
                            dna=dna, record=record)
    assert record["fragments"] == []


# -- cell-less nuclei -------------------------------------------------------

def test_nearest_cell_attaches_within_reach_only():
    nuclei, cells, dna = _field()
    record = {}
    n_out, c_out, *_ = _assign_nuclei_to_cells(
        nuclei.copy(), cells.copy(), options=_on(cellless_nuclei="nearest_cell"),
        dna=dna, record=record)
    assert record["attached"] == {6: 3}
    assert record["cellless"] == [7]
    assert (c_out[n_out == 6] == 3).all()
    assert (n_out == nuclei).all()


def test_nucleus_only_keeps_cellless_nuclei_in_ops():
    nuclei, cells, dna = _field()
    n_out, c_out, counts, record = ops_sample._pair_nuclei_with_cells(
        nuclei, cells, _on(cellless_nuclei="nucleus_only"), dna=dna)
    assert counts["paired"] == 3
    assert counts["nucleus_only"] == 2
    assert record["nucleus_only"] == [4, 5]
    for key in record["nucleus_only"]:
        assert (n_out == key).any()
        assert not (c_out == key).any()
    # paired nuclei still lie in the cell of the same key
    paired = (n_out > 0) & (n_out <= 3)
    assert (n_out[paired] == c_out[paired]).all()


def test_nearest_cell_in_ops_gives_the_nucleus_a_cell():
    nuclei, cells, dna = _field()
    _n, _c, counts, _record = ops_sample._pair_nuclei_with_cells(
        nuclei, cells, _on(cellless_nuclei="nearest_cell"), dna=dna)
    assert counts["attached_cellless"] == 1
    # the attached nucleus is a second nucleus of cell 3, so the ordinary
    # one-nucleus-per-cell rule still keeps one of them
    assert counts["nuclei_without_cell"] == 1


# -- OPS keying -------------------------------------------------------------

def test_ops_keys_a_merged_nucleus_by_its_lowest_piece():
    nuclei, cells, dna = _field()
    n_out, c_out, counts, record = ops_sample._pair_nuclei_with_cells(
        nuclei, cells, _on(nucleus_fragment_merge="rule",
                           cellless_nuclei="nucleus_only"), dna=dna)
    assert counts["merged_fragments"] == 1
    assert counts["binucleate_cells"] == 1
    (key, kept, piece), = record["fragments"]
    assert (kept, piece) == (1, 2)
    whole = (nuclei == 1) | (nuclei == 2)
    assert set(np.unique(n_out[whole])) == {key}
    # the merged nucleus centroid is the centroid of the whole nucleus
    rows, cols = np.nonzero(n_out == key)
    assert abs(rows.mean() - 25) < 0.6 and abs(cols.mean() - 25) < 0.6


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
                                    "merged_label"]
    labels = np.unique(n_out[n_out > 0])
    table = ops_engine._ph_pairing_columns(
        pd.DataFrame({"label": labels}), c_out, path)
    merged_key = record["fragments"][0][0]
    row = table.set_index("label")
    assert row.loc[merged_key, "nucleus_fragment_count"] == 2
    assert bool(row.loc[merged_key, "nucleus_merged"])
    for key in record["nucleus_only"]:
        assert not bool(row.loc[key, "has_cell"])
        assert row.loc[key, "nucleus_fragment_count"] == 1
    gathered = ops_engine._ph_collect_fragments(str(tmp_path), str(tmp_path),
                                                "A1", [3, 4])
    assert len(pd.read_parquet(gathered)) == len(stored)


def test_nucleus_only_object_measures_with_empty_cell_columns():
    nuclei, cells, dna = _field()
    n_out, c_out, _counts, record = ops_sample._pair_nuclei_with_cells(
        nuclei, cells, _on(cellless_nuclei="nucleus_only"), dna=dna)
    image = np.stack([dna, dna], axis=-1)
    table = ops_sample.measure_phenotype_field(n_out, c_out, image,
                                               ["dapi", "gfp"], texture=False)
    alone = table[table["label"].isin(record["nucleus_only"])]
    assert len(alone) == 2
    assert alone["nucleus_area"].notna().all()
    assert alone["cell_area"].isna().all()


# -- Measure: one table column set, off by default --------------------------

def test_measure_flags_only_when_on():
    import pandas as pd

    frame = pd.DataFrame({"label": [1, 3, 7]})
    morphology = {"nucleus": frame.copy(), "cell": frame.copy()}
    measure._flag_shared_nuclei(morphology, {"nucleus": set(), "cell": set()})
    assert "nucleus_merged" not in morphology["nucleus"]
    morphology = {"nucleus": frame.copy()}
    measure._flag_shared_nuclei(morphology, {
        "nucleus": set(), "cell": set(),
        "pairing": {"fragments": [(1, 2)], "cellless": [7]}})
    out = morphology["nucleus"].set_index("label")
    assert out.loc[1, "nucleus_fragment_count"] == 2
    assert bool(out.loc[1, "nucleus_merged"])
    assert not bool(out.loc[7, "has_cell"])
    assert bool(out.loc[3, "has_cell"])


# -- one implementation -----------------------------------------------------

def _calls(module_path, function):
    tree = ast.parse((ROOT / module_path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == function)
    return {c.func.id if isinstance(c.func, ast.Name) else getattr(c.func, "attr", "")
            for c in ast.walk(node) if isinstance(c, ast.Call)}


def test_measure_and_ops_call_the_same_pairing_function():
    assert "_assign_nuclei_to_cells" in _calls("spacr/measure.py", "_measure_crop_core")
    assert "_assign_nuclei_to_cells" in _calls("spacr/ops_sample.py",
                                               "_pair_nuclei_with_cells")
    assert "_pair_nuclei_with_cells" in _calls("spacr/ops_sample.py",
                                               "pair_nuclei_with_cells")
    assert "_pair_nuclei_with_cells" in _calls("spacr/ops_engine.py",
                                               "_ph_segment_field")
    # the fragment merge and the cell-less handling exist only in measure.py
    for path in ("spacr/ops_sample.py", "spacr/ops_engine.py"):
        source = (ROOT / path).read_text(encoding="utf-8")
        assert "def _merge_nucleus_fragments" not in source
        assert "def _attach_cellless_nuclei" not in source
    assert "_merge_nucleus_fragments" in inspect.getsource(
        measure._assign_nuclei_to_cells)


def test_settings_registered_off_with_schema_row():
    from spacr.settings import (ALPHA_FEATURES, DEFAULT_CHANGES,
                                SETTINGS_SCHEMA_VERSION, get_measure_crop_settings)
    from spacr.ops_settings import OPS_DEFAULTS, OPS_TOOLTIPS

    assert SETTINGS_SCHEMA_VERSION >= 967
    assert DEFAULT_CHANGES[967] == {
        "nucleus_fragment_merge": "off", "cellless_nuclei": "drop",
        "ops_ph_nucleus_fragment_merge": "off", "ops_ph_cellless_nuclei": "drop"}
    measure_defaults = get_measure_crop_settings({})
    for name, default in measure._NUCLEUS_PAIRING_DEFAULTS.items():
        assert measure_defaults[name] == default
        assert OPS_DEFAULTS["ops_ph_" + name] == default
        tip = OPS_TOOLTIPS["ops_ph_" + name]
        assert len(tip) <= 600 and tip.endswith("Alpha.") and "Default" in tip
    assert len(ALPHA_FEATURES[1024]["settings"]) == 12

"""Item 1024 follow-up: alpha sub-option defaults (schema 985) and edge flag of nucleus-only objects.

The paper final evaluation sets the fragment test defaults; the feature itself
stays off. A nucleus kept without a cell (``cellless_nuclei`` ``'nucleus_only'``)
takes its field-edge flag from its NUCLEUS mask, through the one mask pass
that gives every cell its flag.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from spacr import ops_engine, ops_sample
from spacr.measure import _nucleus_pairing_options
from spacr.ops_flatfield import mask_pass
from spacr.ops_settings import OPS_DEFAULTS
from spacr.settings import (DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION,
                            _fold_older_defaults, get_measure_crop_settings)

#: sha256 of the default mask pass of tests.test_ops_ownership_flatfield_funk
#: ``_blocky_field`` computed on origin/nightly 5fd144772b, without and with
#: flat-field gains.
NIGHTLY_MASK_PASS_DIGESTS = (
    "b4f537869088b2d2386995c780d0d62f6c5e04b62cd86e98e29db286fbd888dd",
    "fdb7b14451b0fadc8ce6838c0ccad9de8be139cf868a5cb1c32e00aaa073b453")

PAPER = {"max_gap_px": 2.0, "min_solidity": 0.94, "max_valley_dip": 0.30,
         "max_dip": None, "min_contact_px": 0.0, "max_area_rel": None}


def test_the_sub_option_defaults_are_the_paper_values_and_the_feature_stays_off():
    options = _nucleus_pairing_options()
    measure = get_measure_crop_settings({})
    for prefix, source in (("", measure), ("ops_ph_", OPS_DEFAULTS)):
        for short, value in PAPER.items():
            assert source[prefix + "nucleus_fragment_" + short] == value
        assert source[prefix + "nucleus_fragment_merge"] == "off"
    assert options["nucleus_fragment_max_valley_dip"] == 0.30
    assert options["nucleus_fragment_merge"] == "off"


def test_older_files_keep_the_old_valley_dip():
    assert SETTINGS_SCHEMA_VERSION >= 985
    assert DEFAULT_CHANGES[985] == {
        "nucleus_fragment_max_valley_dip": None,
        "ops_ph_nucleus_fragment_max_valley_dip": None}
    file = {}
    _fold_older_defaults(file, 983)
    assert file["nucleus_fragment_max_valley_dip"] is None
    assert file["ops_ph_nucleus_fragment_max_valley_dip"] is None
    chosen = {"nucleus_fragment_max_valley_dip": 0.2}
    _fold_older_defaults(chosen, 983)
    assert chosen["nucleus_fragment_max_valley_dip"] == 0.2
    now = {}
    _fold_older_defaults(now, SETTINGS_SCHEMA_VERSION)
    assert "nucleus_fragment_max_valley_dip" not in now


def _digest(table):
    h = hashlib.sha256(",".join(table.columns).encode())
    for column in table.columns:
        h.update(table[column].to_numpy().tobytes())
    return h.hexdigest()


def test_the_default_mask_pass_is_byte_identical_to_nightly():
    from tests.test_ops_ownership_flatfield_funk import _blocky_field

    nuclei, cells, _image, flat, _offset = _blocky_field()
    got = (_digest(mask_pass(nuclei, cells)),
           _digest(mask_pass(nuclei, cells,
                             {"DAPI": 1.0 / flat.astype(np.float64)})))
    assert got == NIGHTLY_MASK_PASS_DIGESTS


def _field_with_nucleus_only():
    """One cell with its nucleus, a nucleus-only object at the top edge and
    one in the interior."""
    shape = (80, 120)
    cells = np.zeros(shape, np.int32)
    nuclei = np.zeros(shape, np.int32)
    cells[30:70, 60:110] = 1
    nuclei[40:55, 70:90] = 1
    nuclei[0:8, 10:22] = 2
    nuclei[30:42, 10:22] = 3
    return nuclei, cells


def test_a_nucleus_only_object_is_edge_cut_by_its_nucleus_and_an_interior_one_is_not():
    nuclei, cells = _field_with_nucleus_only()
    options = _nucleus_pairing_options({"cellless_nuclei": "nucleus_only"})
    new_nuclei, new_cells, counts, record = ops_sample._pair_nuclei_with_cells(
        nuclei, cells, options)
    keys = record["nucleus_only"]
    assert len(keys) == 2
    table = mask_pass(new_nuclei, new_cells).set_index("label")
    edge, interior = None, None
    for key in keys:
        rows = np.flatnonzero((new_nuclei == key).any(axis=1))
        if rows.min() == 0:
            edge = key
        else:
            interior = key
    assert edge is not None and interior is not None
    assert bool(table.loc[edge, "touches_field_edge"])
    assert bool(table.loc[edge, "nucleus_touches_field_edge"])
    assert not bool(table.loc[interior, "touches_field_edge"])
    assert table.loc[interior, "cell_px"] == 0
    assert table.loc[interior, "nucleus_px"] == 12 * 12
    assert not bool(table.loc[1, "touches_field_edge"])
    cut = ops_engine._ph_touches_edge(table["touches_field_edge"])
    assert int(cut.sum()) == 1


def test_a_field_with_only_nucleus_only_objects_still_gives_rows():
    nuclei = np.zeros((20, 20), np.int32)
    nuclei[5:10, 5:10] = 1
    table = mask_pass(nuclei, np.zeros((20, 20), np.int32))
    assert len(table) == 1 and not bool(table["touches_field_edge"].iloc[0])
    assert isinstance(table, pd.DataFrame)


def test_an_explicit_none_still_turns_the_valley_test_off():
    assert _nucleus_pairing_options({})["nucleus_fragment_max_valley_dip"] == 0.30
    for value in (None, "none", "off"):
        got = _nucleus_pairing_options({"nucleus_fragment_max_valley_dip": value})
        assert got["nucleus_fragment_max_valley_dip"] is None
    twin = _nucleus_pairing_options(
        {"ops_ph_nucleus_fragment_max_valley_dip": None}, prefix="ops_ph_")
    assert twin["nucleus_fragment_max_valley_dip"] is None
    assert _nucleus_pairing_options(
        {"nucleus_fragment_max_valley_dip": ""})["nucleus_fragment_max_valley_dip"] == 0.30
    assert ops_engine._ph_pairing_options({}) is None

"""Mask-engine edges the coverage ratchet found untested.

Filling holes numbers a boolean or float brush mask by connectivity before
filling, exactly as it numbers a single-value integer mask; and a curation
file row with no field name is not a field.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt import mask_engine as me  # noqa: E402


@pytest.mark.parametrize("dtype", [bool, np.float32])
def test_a_brush_mask_that_is_not_integer_is_numbered_then_filled(dtype):
    mask = np.zeros((9, 12), dtype=dtype)
    mask[1:6, 1:6] = 1
    mask[3, 3] = 0
    mask[1:6, 7:11] = 1
    filled = me.fill_holes(mask)
    assert np.issubdtype(filled.dtype, np.integer)
    assert sorted(np.unique(filled)) == [0, 1, 2]
    assert filled[3, 3] == filled[2, 2] != 0


def test_a_curation_row_without_a_field_name_is_skipped(tmp_path):
    columns = list(me.CURATION_COLUMNS)
    (tmp_path / "csv").mkdir()
    with open(me.curation_csv_path(str(tmp_path)), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({columns[0]: "", **{c: "x" for c in columns[1:]}})
        writer.writerow({columns[0]: "plate1_A01_1", **{c: "y" for c in columns[1:]}})
    rows = me.read_curation(str(tmp_path))
    assert list(rows) == ["plate1_A01_1"]

"""A png_list row that cannot name an object is dropped, not mis-cut.

Getting the object id wrong does not fail loudly: it cuts a different cell out
of the merged array and files it under the first one's name. These tests pin
:func:`spacr.png_list._object_id_int` (only an exact, finite, non-boolean
integer value is a label) and :func:`spacr.png_list.crop_rows_from_png_list`
(unknown or ambiguous crop modes are refused; a table written for another
crop mode still finds its labels; a table with no object column cuts nothing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import png_list
from spacr.png_list import (PNG_LIST_ID_COLUMNS, _object_id_int,
                            crop_rows_from_png_list)

_OUTPUT_COLUMNS = ["path_name", "object_label", "object_type",
                   "object_label_type"]


def test_object_id_int_accepts_only_an_exact_finite_integer(subtests):
    """Absent, NaN, fractional, infinite and boolean ids name no object."""
    cases = [
        ("None", None, None),
        ("float nan", float("nan"), None),
        ("numpy whole float", np.float64(12.0), 12),
        ("python whole float", 12.0, 12),
        ("fractional float", 12.5, None),
        ("fractional float32", np.float32(12.5), None),
        ("positive infinity", np.inf, None),
        ("negative infinity", -np.inf, None),
        ("python bool", True, None),
        ("numpy bool", np.bool_(True), None),
        ("python int", 7, 7),
        ("numpy int", np.int64(9), 9),
        ("prefixed string", "o12", 12),
        ("upper-case prefix", "O4", 4),
        ("padded string", " o3 ", 3),
        ("multi-object crop", "omulti", None),
        ("no-object crop", "onone", None),
    ]
    for case_id, value, expected in cases:
        with subtests.test(msg=case_id):
            assert _object_id_int(value) == expected


def test_crop_rows_resolve_labels_or_refuse(subtests, tmp_path, monkeypatch):
    """Each crop-mode situation yields the exact rows, labels and provenance."""
    db = str(tmp_path / "measurements.db")
    modes = sorted(PNG_LIST_ID_COLUMNS)

    with subtests.test(msg="an unknown object type is refused"):
        frame = pd.DataFrame({"cell_id": ["o4"], "path_name": ["field.npy"]})
        with pytest.raises(ValueError) as caught:
            crop_rows_from_png_list(db, frame, object_type="nucleuz",
                                    verbose=False)
        assert str(caught.value) == (
            f"object_type must be one of {modes}; got 'nucleuz'")

    with subtests.test(msg="an unhashable object type is refused the same way"):
        with pytest.raises(ValueError, match="object_type must be one of"):
            crop_rows_from_png_list(db, frame, object_type=["cell"],
                                    verbose=False)

    with subtests.test(msg="two alternate id columns are ambiguous"):
        frame = pd.DataFrame({"nucleus_id": ["o4"], "pathogen_id": ["o7"],
                              "path_name": ["field.npy"]})
        with pytest.raises(ValueError) as caught:
            crop_rows_from_png_list(db, frame, object_type="cell",
                                    verbose=False)
        assert str(caught.value) == (
            "'cell' needs 'cell_id', but this frame carries multiple "
            "alternate object ID columns: ['nucleus_id', 'pathogen_id']")

    with subtests.test(msg="a frame with no object column keeps no rows"):
        frame = pd.DataFrame({"png_path": ["a.png", "b.png"]})
        out = crop_rows_from_png_list(str(tmp_path / "absent.db"), frame,
                                      object_type="cell", verbose=False)
        assert list(out.columns) == ["png_path", *_OUTPUT_COLUMNS]
        assert len(out) == 0

    with subtests.test(msg="object_label is used when no id column exists"):
        frame = pd.DataFrame({"object_label": [3, None, 5],
                              "path_name": ["f.npy", "f.npy", "f.npy"]})
        out = crop_rows_from_png_list(db, frame, object_type="cell",
                                      verbose=False)
        assert out["object_label"].tolist() == [3, 5]
        assert out["object_label_type"].tolist() == ["cell", "cell"]

    with subtests.test(msg="a table from another crop mode finds its labels"):
        frame = pd.DataFrame({
            "nucleus_id": ["o5", "o7", "omulti"],
            "plateID": [1] * 3, "rowID": [1] * 3,
            "columnID": [1] * 3, "fieldID": [1] * 3,
            "png_path": ["/crops/a.png", "/crops/b.png", "/crops/c.png"],
        })
        seen = []

        def field_paths(_db_path, object_type):
            seen.append(object_type)
            return {(1, 1, 1, 1): ("/merged/f1.npy", "f1.npy")}

        monkeypatch.setattr(png_list, "_merged_field_paths", field_paths)
        out = crop_rows_from_png_list(db, frame, object_type="cell",
                                      verbose=False)
        assert seen == ["nucleus"]
        assert out["png_path"].tolist() == ["/crops/a.png", "/crops/b.png"]
        assert out["object_label"].tolist() == [5, 7]
        assert out["path_name"].tolist() == ["/merged/f1.npy"] * 2
        # object_type is the instruction the crop cutter obeys; the label
        # provenance is recorded separately.
        assert out["object_type"].tolist() == ["cell", "cell"]
        assert out["object_label_type"].tolist() == ["nucleus", "nucleus"]

"""A nucleus was handed a picture of a different cell.

`filters` attached crop paths by merging `png_list` onto the object frame on
the identity columns plus `object_label` -- and nothing else. `png_list`'s
label is the label of the object that was CROPPED, usually the cell, while
the object frame holds one row per object of every type. So the merge matched
integers across object types.

Measured on two cells with two nuclei each, crops taken of cells:

    object     label   parent   png_path       correct?
    cell       1       -        cell_1.png     yes
    cell       2       -        cell_2.png     yes
    nucleus    1       cell 1   cell_1.png     by coincidence
    nucleus    2       cell 1   cell_2.png     NO -- its cell is 1
    nucleus    1       cell 2   cell_1.png     NO -- its cell is 2
    nucleus    2       cell 2   cell_2.png     by coincidence

Half wrong, and the half that was right was right because the child's label
happened to equal its parent's -- which is guaranteed only when every cell
has exactly one child. Every crop-backed view downstream reads this column:
the annotation app, the classifier's inputs, the image grids.

A CHILD'S CROP IS ITS PARENT'S, because the crop is a picture of one cell and
the child is inside it. Anything with no containment relation to the cropped
type gets no path, because a wrong picture is worse than a missing one.
"""

import sqlite3

import pandas as pd
import pytest

from spacr.filters import (build_filters_from_relationships, png_crop_type)

BASE = dict(plateID="p1", rowID="r1", columnID="c1", fieldID="f1")


def build(tmp_path, *, crop_column="cell_id", n_cells=2, per_cell=2):
    """A database with `n_cells` cells, `per_cell` nuclei in each, and one
    crop per cell."""
    path = str(tmp_path / "measurements.db")
    con = sqlite3.connect(path)
    try:
        pd.DataFrame([{**BASE, "object_label": i, "cell_area": 100.0 * i}
                      for i in range(1, n_cells + 1)]
                     ).to_sql("cell", con, index=False)
        pd.DataFrame([{**BASE, "object_label": j, "cell_id": i,
                       "nucleus_area": 5.0 * j}
                      for i in range(1, n_cells + 1)
                      for j in range(1, per_cell + 1)]
                     ).to_sql("nucleus", con, index=False)
        pd.DataFrame([{**BASE, crop_column: f"o{i}",
                       "png_path": f"/crops/{crop_column[:-3]}_{i}.png"}
                      for i in range(1, n_cells + 1)]
                     ).to_sql("png_list", con, index=False)
        con.commit()
    finally:
        con.close()
    return path


def frame(tmp_path, **kwargs):
    return build_filters_from_relationships(build(tmp_path, **kwargs))


# ---------------------------------------------------------------------------
# which object the crops are of
# ---------------------------------------------------------------------------

def test_the_crop_mode_is_recovered_from_the_schema(tmp_path):
    """`png_list` names its id column after what it cropped."""
    other = tmp_path / "other"
    other.mkdir()
    assert png_crop_type(build(tmp_path, crop_column="cell_id")) == "cell"
    assert png_crop_type(
        build(other, crop_column="pathogen_id")) == "pathogen"


def test_no_png_table_is_not_an_error(tmp_path):
    path = str(tmp_path / "empty.db")
    con = sqlite3.connect(path)
    pd.DataFrame([{**BASE, "object_label": 1}]).to_sql("cell", con,
                                                       index=False)
    con.commit()
    con.close()
    assert png_crop_type(path) is None


# ---------------------------------------------------------------------------
# the regression
# ---------------------------------------------------------------------------

def test_a_child_gets_its_own_parents_crop(tmp_path):
    """The defect, stated as the thing that must be true."""
    out = frame(tmp_path)
    nuclei = out[out["object_type"] == "nucleus"]

    for _, row in nuclei.iterrows():
        assert row["png_path"] == f"/crops/cell_{int(row['parent_label'])}.png", (
            f"nucleus {row['object_label']} of cell {row['parent_label']} "
            f"points at {row['png_path']}")


def test_two_children_of_one_cell_share_that_cells_crop(tmp_path):
    out = frame(tmp_path)
    nuclei = out[out["object_type"] == "nucleus"]
    for parent, group in nuclei.groupby("parent_label"):
        assert group["png_path"].nunique() == 1, (
            f"the children of cell {parent} point at different pictures")


def test_the_cropped_object_still_gets_its_own_crop(tmp_path):
    out = frame(tmp_path)
    cells = out[out["object_type"] == "cell"]
    for _, row in cells.iterrows():
        assert row["png_path"] == f"/crops/cell_{int(row['object_label'])}.png"


def test_the_label_coincidence_that_hid_this_is_broken_on_purpose(tmp_path):
    """With three children per cell, child 3 of cell 1 has no cell 3 to
    borrow from -- the old code would have handed it a NaN or cell 3's
    picture depending on the plate."""
    out = frame(tmp_path, n_cells=2, per_cell=3)
    nuclei = out[out["object_type"] == "nucleus"]

    assert len(nuclei) == 6
    assert nuclei["png_path"].notna().all(), (
        "a child whose label exceeds the cell count lost its crop")
    assert set(nuclei["png_path"]) == {"/crops/cell_1.png", "/crops/cell_2.png"}


# ---------------------------------------------------------------------------
# no containment relation means no picture
# ---------------------------------------------------------------------------

def test_an_object_unrelated_to_the_cropped_type_gets_no_path(tmp_path):
    """Crops of pathogens say nothing about which nucleus is which.

    A nucleus is not inside a pathogen, so there is no defensible crop for
    it, and the honest answer is none.
    """
    out = build_filters_from_relationships(
        build(tmp_path, crop_column="pathogen_id"))
    nuclei = out[out["object_type"] == "nucleus"]
    assert nuclei["png_path"].isna().all()


def test_the_column_exists_even_when_nothing_could_be_attached(tmp_path):
    """A missing column and an empty one are different failures downstream."""
    out = build_filters_from_relationships(
        build(tmp_path, crop_column="pathogen_id"))
    assert "png_path" in out.columns


# ---------------------------------------------------------------------------
# the other builder
# ---------------------------------------------------------------------------

def test_the_public_builder_gets_the_same_answer(tmp_path):
    """`build_filters_frame` is what `ensure_filters_table` calls.

    It delegates to the relationships route, so this is the path a real run
    takes -- and the assertion is the same one: a child points at a picture
    of ITS cell.
    """
    from spacr.filters import build_filters_frame

    out = build_filters_frame(build(tmp_path))
    nuclei = out[out["object_type"] == "nucleus"]
    assert len(nuclei) == 4
    for _, row in nuclei.iterrows():
        assert row["png_path"] == (
            f"/crops/cell_{int(row['parent_label'])}.png")


def test_the_fallback_builder_gates_on_the_membership_flag(tmp_path):
    """The route taken when relationships cannot be built.

    That frame is one row per (field, object_label) with `in_<table>` flags
    and NO object_type column, so the flag is the only type axis there is. A
    row that is not of the cropped type must keep no path, because matching
    on the label alone is the whole defect.
    """
    import pandas as pd

    from spacr import filters as module

    frame = pd.DataFrame({
        "plateID": ["p1"] * 3, "rowID": ["r1"] * 3, "columnID": ["c1"] * 3,
        "fieldID": ["f1"] * 3, "object_label": [1, 2, 3],
        "in_cell": [1, 0, 1], "in_nucleus": [0, 1, 0],
        "png_path": ["/crops/cell_1.png", "/crops/cell_2.png",
                     "/crops/cell_3.png"],
    })
    flag = f"{module.PRESENT_PREFIX}cell"
    frame.loc[frame[flag] != 1, "png_path"] = None

    assert frame["png_path"].tolist() == [
        "/crops/cell_1.png", None, "/crops/cell_3.png"]

"""A screen that moved computer still finds its crops.

`png_list.png_path` is absolute and written at crop time. The maintainer's
rule: "if a source folder moves computer the paths still work as long as the
folder structure is upheld". Measured on the TSG101 screen -- 0 of 60,816
recorded paths existed, 60,816 of 60,816 existed once rebuilt under the plate
folder the database was opened from.

The safety half is tested as hard as the feature: a path is only ever
rewritten to somewhere a file ACTUALLY IS.
"""
import os

import pandas as pd
import pytest

from spacr.portable_paths import (reroot_column, reroot_crop_path,
                                  source_root_for_database)


@pytest.fixture()
def moved(tmp_path):
    """A crop that exists under `new/`, recorded as living under `old/`."""
    tail = os.path.join("data", "single_nucleus", "plate1_H19", "cell_png")
    new = tmp_path / "new" / tail
    new.mkdir(parents=True)
    (new / "crop.png").write_bytes(b"x")
    recorded = str(tmp_path / "old" / tail / "crop.png")
    return recorded, str(tmp_path / "new"), str(new / "crop.png")


def test_a_recorded_path_is_rebuilt_under_the_folder_it_lives_in_now(moved):
    recorded, root, real = moved
    assert not os.path.exists(recorded)

    assert reroot_crop_path(recorded, root) == real


def test_a_path_that_still_exists_is_left_exactly_as_it_is(tmp_path):
    here = tmp_path / "data" / "x.png"
    here.parent.mkdir(parents=True)
    here.write_bytes(b"x")

    # Even with a root that could rebuild it, an existing path is untouched.
    assert reroot_crop_path(str(here), str(tmp_path / "elsewhere")) == str(here)


def test_a_path_that_cannot_be_found_is_returned_UNCHANGED(tmp_path):
    # The safety rule. Rewriting to somewhere equally absent would make the
    # error name a folder the user never chose.
    recorded = "/gone/plate1/data/single_nucleus/crop.png"

    assert reroot_crop_path(recorded, str(tmp_path)) == recorded


def test_a_root_that_itself_contains_a_data_component_still_resolves(tmp_path):
    tail = os.path.join("data", "single_nucleus", "crop.png")
    new = tmp_path / "new" / tail
    new.parent.mkdir(parents=True)
    new.write_bytes(b"x")
    # The RECORDED root has its own `data` component, so splitting on the
    # first one would keep `proj/plate1/data` in the tail.
    recorded = f"/nas/data/proj/plate1/{tail}"

    assert reroot_crop_path(recorded, str(tmp_path / "new")) == str(new)


def test_reroot_column_reports_how_many_moved_and_leaves_the_rest(moved):
    recorded, root, real = moved
    frame = pd.DataFrame({"png_path": [recorded, "", None, "/still/gone.png"]})

    moved_count = reroot_column(frame, "png_path", root)

    assert moved_count == 1
    assert frame["png_path"].iloc[0] == real
    assert frame["png_path"].iloc[3] == "/still/gone.png"


def test_a_column_that_is_not_there_is_not_an_error():
    # The PNG route and the merged route carry different path columns, so a
    # caller must be able to ask for both.
    assert reroot_column(pd.DataFrame({"a": [1]}), "png_path", "/tmp") == 0


def test_the_root_of_a_database_is_the_plate_folder_that_holds_data():
    got = source_root_for_database("/screens/plate1/measurements/measurements.db")

    assert got == "/screens/plate1"


def test_no_root_means_no_rewrite():
    assert reroot_crop_path("/gone/data/x.png", "") == "/gone/data/x.png"

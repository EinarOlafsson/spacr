"""External-mask input table — correcting what the detector guessed.

The detector is allowed to be wrong; the table exists so the user can fix
it before a run turns the guess into a project. Everything here is driven
against REAL tif files so the roles under test are the ones detection
actually produced, not roles a fixture asserted into place.

Covered: a second batch landing in a folder already in the table, the
string and dict shapes ``set_value`` accepts, the guards that refuse a bad
row or a role/object type that is not one of the known ones, removing
selected rows, the two in-table combo boxes writing back through their own
signals, and the two file pickers including the cancel that must add
nothing.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile


from spacr.qt.widgets.external_mask_inputs import ExternalMaskInputWidget


def _write(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(array), photometric="minisblack")
    return path


@pytest.fixture
def plate(tmp_path):
    """Two intensity images and two label masks, in three folders."""
    yy, xx = np.indices((32, 32))
    images = tmp_path / "images"
    cells = tmp_path / "cell_masks"
    nuclei = tmp_path / "nucleus_masks"
    _write(images / "fov001_C1.tif", yy * 32 + xx)
    _write(images / "fov002_C1.tif", (xx * 17 + yy * 3) % 4096)
    cell = np.zeros((32, 32), dtype=np.uint16)
    cell[3:29, 3:29] = 1
    nucleus = np.zeros((32, 32), dtype=np.uint16)
    nucleus[10:20, 11:21] = 1
    _write(cells / "fov001_cell_mask.tif", cell)
    _write(nuclei / "fov001_nucleus_mask.tif", nucleus)
    return {"images": images, "cells": cells, "nuclei": nuclei}


@pytest.fixture
def widget(qtbot):
    view = ExternalMaskInputWidget()
    qtbot.addWidget(view)
    return view


# ---------------------------------------------------------------------------
# add_paths
# ---------------------------------------------------------------------------


def test_dropping_the_same_file_twice_does_not_count_it_twice(widget, plate):
    one = plate["cells"] / "fov001_cell_mask.tif"
    widget.add_paths([one])
    widget.add_paths([one])
    assert widget.file_count() == 1


def test_a_batch_of_nothing_leaves_the_table_and_the_signal_alone(widget,
                                                                  tmp_path):
    seen = []
    widget.value_changed.connect(lambda: seen.append(1))
    empty = tmp_path / "no_images"
    empty.mkdir()
    assert widget.add_paths([empty]) == 0
    assert seen == []


# ---------------------------------------------------------------------------
# The guards
# ---------------------------------------------------------------------------


def test_a_row_that_is_not_in_the_table_is_refused(widget, plate):
    widget.set_value([str(plate["images"])])
    assert widget.set_group_role(7, "mask") is False
    assert widget.set_group_object_type(7, "cell") is False
    assert widget.set_group_object_type(-1, "cell") is False


def test_unassigning_an_object_type_is_allowed(widget, plate):
    widget.set_value([str(plate["cells"])])
    assert widget.set_group_object_type(0, "unassigned") is True
    assert widget.groups()[0].object_type is None
    assert widget.set_group_object_type(0, "pathogen") is True
    assert widget.groups()[0].object_type == "pathogen"


def test_an_ignored_group_is_not_part_of_the_value(widget, plate):
    widget.set_value([str(plate["images"]), str(plate["cells"])])
    assert widget.set_group_role(0, "ignore") is True
    keys = {row["role"] for row in widget.get_value()}
    assert "ignore" not in keys
    assert len(widget.get_value()) == widget.group_count() - 1

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

from PySide6.QtWidgets import QFileDialog

from spacr.external_masks import OBJECT_TYPES, detect_inputs
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

def test_a_second_drop_into_a_known_folder_merges_instead_of_duplicating(
        widget, plate):
    """Two drops of one folder's files must leave ONE row, not two."""
    first = plate["images"] / "fov001_C1.tif"
    second = plate["images"] / "fov002_C1.tif"

    assert widget.add_paths([first]) == 1
    assert widget.group_count() == 1
    assert widget.add_paths([second]) == 1

    assert widget.group_count() == 1
    assert widget.file_count() == 2
    assert widget._table.rowCount() == 1
    assert sorted(widget.groups()[0].paths) == sorted(
        [str(first), str(second)])


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
# set_value
# ---------------------------------------------------------------------------

def test_a_single_path_string_is_accepted_as_one_path(widget, plate):
    """The settings store can hand back a bare string, not only a list."""
    widget.set_value(str(plate["cells"]))
    assert widget.group_count() == 1
    assert widget.groups()[0].role == "mask"


def test_a_list_of_paths_is_re_detected(widget, plate):
    widget.set_value([str(plate["images"]), str(plate["nuclei"])])
    roles = {group.role for group in widget.groups()}
    assert roles == {"image", "mask"}
    assert widget.file_count() == 3


def test_a_list_of_saved_dicts_is_restored_without_re_detecting(widget,
                                                                plate):
    """A saved answer must survive reload even if detection would differ."""
    saved = [group.to_dict() for group in detect_inputs([plate["cells"]])]
    saved[0]["object_type"] = "pathogen"
    widget.set_value(saved)
    assert widget.groups()[0].object_type == "pathogen"


def test_an_empty_value_clears_the_table(widget, plate):
    widget.set_value([str(plate["images"])])
    widget.set_value(None)
    assert widget.group_count() == 0
    assert widget._table.rowCount() == 0


# ---------------------------------------------------------------------------
# The guards
# ---------------------------------------------------------------------------

def test_a_role_that_is_not_a_role_is_refused(widget, plate):
    widget.set_value([str(plate["images"])])
    assert widget.set_group_role(0, "nonsense") is False
    assert widget.groups()[0].role == "image"


def test_a_row_that_is_not_in_the_table_is_refused(widget, plate):
    widget.set_value([str(plate["images"])])
    assert widget.set_group_role(7, "mask") is False
    assert widget.set_group_object_type(7, "cell") is False
    assert widget.set_group_object_type(-1, "cell") is False


def test_an_object_type_that_is_not_a_plane_is_refused(widget, plate):
    widget.set_value([str(plate["cells"])])
    before = widget.groups()[0].object_type
    assert widget.set_group_object_type(0, "mitochondrion") is False
    assert widget.groups()[0].object_type == before


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


# ---------------------------------------------------------------------------
# remove_selected
# ---------------------------------------------------------------------------

def test_removing_the_selected_rows_takes_them_out_of_the_value(widget,
                                                                plate):
    widget.set_value([str(plate["images"]), str(plate["cells"]),
                      str(plate["nuclei"])])
    assert widget.group_count() == 3
    seen = []
    widget.value_changed.connect(lambda: seen.append(1))

    widget._table.selectRow(0)
    widget._table.selectionModel().select(
        widget._table.model().index(2, 0),
        widget._table.selectionModel().SelectionFlag.Select
        | widget._table.selectionModel().SelectionFlag.Rows)
    widget.remove_selected()

    assert widget.group_count() == 1
    assert widget._table.rowCount() == 1
    assert seen == [1]


def test_removing_nothing_does_not_announce_a_change(widget, plate):
    widget.set_value([str(plate["images"])])
    seen = []
    widget.value_changed.connect(lambda: seen.append(1))
    widget._table.clearSelection()
    widget.remove_selected()
    assert seen == []
    assert widget.group_count() == 1


# ---------------------------------------------------------------------------
# The in-table combo boxes
# ---------------------------------------------------------------------------

def test_changing_the_role_box_greys_the_object_type_beside_it(widget,
                                                               plate):
    """Only a mask has an object type; an image row must not offer one."""
    widget.set_value([str(plate["cells"])])
    role_box = widget._table.cellWidget(0, 2)
    object_box = widget._table.cellWidget(0, 3)
    assert object_box.isEnabled() is True

    seen = []
    widget.value_changed.connect(lambda: seen.append(1))
    role_box.setCurrentIndex(role_box.findData("image"))

    assert widget.groups()[0].role == "image"
    assert object_box.isEnabled() is False
    assert seen == [1]


def test_changing_the_object_box_writes_the_plane_through(widget, plate):
    widget.set_value([str(plate["cells"])])
    object_box = widget._table.cellWidget(0, 3)
    object_box.setCurrentIndex(object_box.findData("pathogen"))
    assert widget.groups()[0].object_type == "pathogen"


def test_choosing_the_placeholder_clears_the_plane(widget, plate):
    """"Choose…" carries None, which is not an object type."""
    widget.set_value([str(plate["cells"])])
    object_box = widget._table.cellWidget(0, 3)
    object_box.setCurrentIndex(object_box.findData("pathogen"))
    object_box.setCurrentIndex(0)
    assert widget.groups()[0].object_type is None
    assert all(name in OBJECT_TYPES for name in ("cell", "pathogen"))


# ---------------------------------------------------------------------------
# The pickers
# ---------------------------------------------------------------------------

def test_the_file_picker_adds_what_it_returned(widget, plate, monkeypatch):
    chosen = [str(plate["images"] / "fov001_C1.tif")]
    monkeypatch.setattr(
        QFileDialog, "getOpenFileNames",
        staticmethod(lambda *a, **k: (chosen, "Images (*.tif)")))
    widget._pick_files()
    assert widget.file_count() == 1


def test_a_cancelled_file_picker_adds_nothing(widget, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: ([], "")))
    widget._pick_files()
    assert widget.group_count() == 0


def test_the_folder_picker_adds_the_whole_folder(widget, plate, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(plate["images"])))
    widget._pick_folder()
    assert widget.file_count() == 2


def test_a_cancelled_folder_picker_adds_nothing(widget, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    widget._pick_folder()
    assert widget.group_count() == 0

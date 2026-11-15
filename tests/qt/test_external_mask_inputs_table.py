"""The External-Masks input table: values in, guards, in-table combos and pickers.

The detector is allowed to guess wrong; this table is where the user fixes the
guess before a run turns it into a project.  Groups are built from REAL tif
files so the roles under test are the ones detection produced, or from saved
group dicts where only index arithmetic is under test.  Rows and groups are
allowed to drift apart (a removed row's combo box stays alive and wired), so
every stale-index case is paired with the live row that still writes through.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QWidget

from spacr.external_masks import OBJECT_TYPES, ROLES, detect_inputs
from spacr.qt.widgets.external_mask_inputs import ExternalMaskInputWidget


def _write(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(array), photometric="minisblack")
    return path


@pytest.fixture
def plate(tmp_path):
    """Two intensity images and two label masks, in three folders."""
    yy, xx = np.indices((16, 16))
    images = tmp_path / "images"
    cells = tmp_path / "cell_masks"
    nuclei = tmp_path / "nucleus_masks"
    _write(images / "fov001_C1.tif", (yy * 16 + xx).astype(np.uint16))
    _write(images / "fov002_C1.tif", ((xx * 7 + yy) % 4096).astype(np.uint16))
    cell = np.zeros((16, 16), dtype=np.uint16)
    cell[3:13, 3:13] = 1
    nucleus = np.zeros((16, 16), dtype=np.uint16)
    nucleus[5:10, 5:10] = 1
    _write(cells / "fov001_cell_mask.tif", cell)
    _write(nuclei / "fov001_nucleus_mask.tif", nucleus)
    return {"images": images, "cells": cells, "nuclei": nuclei}


@pytest.fixture
def make_widget(qtbot):
    def build(**kwargs):
        view = ExternalMaskInputWidget(**kwargs)
        qtbot.addWidget(view)
        return view
    return build


def _group_dict(key, role, object_type=None, count=1):
    """One InputGroup as the mapping ``set_value`` accepts from a saved run."""
    return {"key": key, "root": f"/data/{key}",
            "paths": [f"/data/{key}/fov{n:03d}.tif" for n in range(1, count + 1)],
            "role": role, "object_type": object_type, "confidence": 0.75,
            "reason": f"restored {key}"}


def _announcements(widget):
    seen = []
    widget.value_changed.connect(lambda: seen.append(1))
    return seen


def test_set_value_and_add_paths_accept_every_saved_shape(make_widget, plate,
                                                         subtests):
    """A bare string, a path list, saved dicts and None each land as the right groups."""
    with subtests.test(msg="second-drop-into-a-known-folder-merges"):
        widget = make_widget()
        first = plate["images"] / "fov001_C1.tif"
        second = plate["images"] / "fov002_C1.tif"
        assert widget.add_paths([first]) == 1
        assert widget.add_paths([second]) == 1
        assert (widget.group_count(), widget.file_count(),
                widget._table.rowCount()) == (1, 2, 1)
        assert sorted(widget.groups()[0].paths) == sorted([str(first), str(second)])
    with subtests.test(msg="single-path-string"):
        widget = make_widget()
        widget.set_value(str(plate["cells"]))
        assert [g.role for g in widget.groups()] == ["mask"]
    with subtests.test(msg="list-of-paths-is-re-detected"):
        widget = make_widget()
        widget.set_value([str(plate["images"]), str(plate["nuclei"])])
        assert sorted(g.role for g in widget.groups()) == ["image", "mask"]
        assert widget.file_count() == 3
    with subtests.test(msg="saved-dicts-restore-without-re-detecting"):
        widget = make_widget()
        saved = [g.to_dict() for g in detect_inputs([plate["cells"]])]
        saved[0]["object_type"] = "pathogen"
        widget.set_value(saved)
        assert widget.groups()[0].object_type == "pathogen"
    with subtests.test(msg="unknown-saved-object-type-restores-unassigned"):
        widget = make_widget()
        widget.set_value([_group_dict("mitochondria", "mask", "mitochondrion")])
        box = widget._table.cellWidget(0, 3)
        assert widget.groups()[0].object_type is None
        assert (box.currentData(), box.currentText()) == (None, "Choose…")
        assert widget._table.item(0, 1).text() == "mask · unassigned"
        assert widget.get_value()[0]["object_type"] is None
    with subtests.test(msg="empty-value-clears"):
        widget = make_widget()
        widget.set_value([str(plate["images"])])
        widget.set_value(None)
        assert (widget.group_count(), widget._table.rowCount()) == (0, 0)


def test_rows_map_to_groups_and_stale_indices_change_nothing(make_widget, plate,
                                                            subtests):
    """Unknown roles/planes and out-of-range or stale rows are refused; live rows still write."""
    with subtests.test(msg="drawn-rows-map-to-their-groups"):
        widget = make_widget()
        widget.set_value([str(plate["images"]), str(plate["cells"]),
                          str(plate["nuclei"])])
        assert [widget._group_of_row(r) for r in range(3)] == [0, 1, 2]
        assert [widget._table.item(r, 0).data(Qt.UserRole)
                for r in range(3)] == [0, 1, 2]
    with subtests.test(msg="unknown-role-and-plane-refused"):
        widget = make_widget()
        widget.set_value([str(plate["cells"])])
        before = widget.groups()[0].object_type
        assert widget.set_group_role(0, "nonsense") is False
        assert widget.set_group_object_type(0, "mitochondrion") is False
        assert (widget.groups()[0].role, widget.groups()[0].object_type) == (
            "mask", before)
    with subtests.test(msg="negative-stored-index-refused"):
        widget = make_widget()
        widget.set_value([str(plate["images"]), str(plate["cells"])])
        widget._table.item(0, 0).setData(Qt.UserRole, -1)
        assert widget._group_of_row(0) == -1
        widget._table.selectRow(0)
        widget.remove_selected()
        assert [g.role for g in widget.groups()] == ["image", "mask"]
    with subtests.test(msg="role-for-a-row-past-the-end"):
        widget = make_widget()
        widget.set_value([_group_dict("cells", "mask", "cell")])
        seen = _announcements(widget)
        widget._role_changed(4, "ignore")
        widget._role_changed(-1, "ignore")
        assert (widget.groups()[0].role, widget.groups()[0].object_type,
                seen) == ("mask", "cell", [])
        widget._role_changed(0, "image")
        assert (widget.groups()[0].role, seen) == ("image", [1])
    with subtests.test(msg="stale-role-box-from-a-removed-row"):
        widget = make_widget()
        widget.set_value([str(plate["images"]), str(plate["cells"])])
        stale_box = widget._table.cellWidget(1, 2)
        widget._table.selectRow(1)
        widget.remove_selected()
        seen = _announcements(widget)
        stale_box.setCurrentIndex(stale_box.findData("ignore"))
        assert (widget.groups()[0].role, seen) == ("image", [])
        live_box = widget._table.cellWidget(0, 2)
        live_box.setCurrentIndex(live_box.findData("ignore"))
        assert (widget.groups()[0].role, seen) == ("ignore", [1])
        assert "ignore" in ROLES
    with subtests.test(msg="stale-object-box-from-a-removed-row"):
        widget = make_widget()
        widget.set_value([str(plate["cells"]), str(plate["nuclei"])])
        stale_box = widget._table.cellWidget(1, 3)
        widget._table.selectRow(1)
        widget.remove_selected()
        kept = widget.groups()[0].object_type
        seen = _announcements(widget)
        stale_box.setCurrentIndex(stale_box.findData("pathogen"))
        assert (widget.groups()[0].object_type, seen) == (kept, [])
        live_box = widget._table.cellWidget(0, 3)
        live_box.setCurrentIndex(live_box.findData("pathogen"))
        assert (widget.groups()[0].object_type, seen) == ("pathogen", [1])
        assert "pathogen" in OBJECT_TYPES


def test_remove_selected_takes_out_only_selected_drawn_rows(make_widget, plate,
                                                           subtests):
    """Removing two rows announces once; removing nothing announces nothing."""
    with subtests.test(msg="two-selected-rows"):
        widget = make_widget()
        widget.set_value([str(plate["images"]), str(plate["cells"]),
                          str(plate["nuclei"])])
        seen = _announcements(widget)
        widget._table.selectRow(0)
        model = widget._table.selectionModel()
        model.select(widget._table.model().index(2, 0),
                     model.SelectionFlag.Select | model.SelectionFlag.Rows)
        widget.remove_selected()
        assert (widget.group_count(), widget._table.rowCount(), seen) == (1, 1, [1])
        assert [g.role for g in widget.groups()] == ["mask"]
    with subtests.test(msg="nothing-selected"):
        widget = make_widget()
        widget.set_value([str(plate["images"])])
        seen = _announcements(widget)
        widget._table.clearSelection()
        widget.remove_selected()
        assert (widget.group_count(), seen) == (1, [])


def test_the_in_table_combo_boxes_write_through_and_grey_the_object_box(
        make_widget, plate, subtests):
    """Only a mask row offers an object type; the role lands even when its neighbour box is gone."""
    with subtests.test(msg="role-box-greys-and-re-enables-the-object-box"):
        widget = make_widget()
        widget.set_value([str(plate["cells"])])
        role_box = widget._table.cellWidget(0, 2)
        object_box = widget._table.cellWidget(0, 3)
        assert object_box.isEnabled() is True
        seen = _announcements(widget)
        role_box.setCurrentIndex(role_box.findData("image"))
        assert (widget.groups()[0].role, object_box.isEnabled(), seen) == (
            "image", False, [1])
        role_box.setCurrentIndex(role_box.findData("mask"))
        assert (widget.groups()[0].role, object_box.isEnabled()) == ("mask", True)
    with subtests.test(msg="role-lands-when-the-object-box-is-gone"):
        widget = make_widget()
        widget.set_value([str(plate["cells"])])
        widget._table.removeCellWidget(0, 3)
        assert widget._table.cellWidget(0, 3) is None
        seen = _announcements(widget)
        role_box = widget._table.cellWidget(0, 2)
        role_box.setCurrentIndex(role_box.findData("image"))
        assert (widget.groups()[0].role, seen) == ("image", [1])
        assert widget.get_value()[0]["role"] == "image"
    with subtests.test(msg="role-lands-when-the-object-cell-is-empty"):
        widget = make_widget()
        widget.set_value([_group_dict("cells", "mask", "cell"),
                          _group_dict("nuclei", "mask", "nucleus")])
        widget._table.removeCellWidget(0, 3)
        seen = _announcements(widget)
        widget._role_changed(0, "image")
        assert (widget.groups()[0].role, seen) == ("image", [1])
        surviving = widget._table.cellWidget(1, 3)
        widget._role_changed(1, "ignore")
        assert surviving.isEnabled() is False
        assert [g.role for g in widget.groups()] == ["image", "ignore"]
        assert [(item["key"], item["role"]) for item in widget.get_value()] == [
            ("cells", "image")]
    with subtests.test(msg="object-box-writes-the-plane-and-placeholder-clears"):
        widget = make_widget()
        widget.set_value([str(plate["cells"])])
        object_box = widget._table.cellWidget(0, 3)
        object_box.setCurrentIndex(object_box.findData("pathogen"))
        assert widget.groups()[0].object_type == "pathogen"
        object_box.setCurrentIndex(0)
        assert widget.groups()[0].object_type is None


def test_the_pickers_and_test_data_import_add_files(make_widget, plate, qtbot,
                                                   monkeypatch, subtests):
    """The file and folder pickers add what they return, cancel adds nothing; test data fills an empty destination only."""
    chosen = [str(plate["images"] / "fov001_C1.tif")]
    pickers = {
        "file-picker": ("getOpenFileNames", (chosen, "Images (*.tif)"),
                        "_pick_files", 1, 1),
        "file-picker-cancelled": ("getOpenFileNames", ([], ""), "_pick_files", 0, 0),
        "folder-picker": ("getExistingDirectory", str(plate["images"]),
                          "_pick_folder", 1, 2),
        "folder-picker-cancelled": ("getExistingDirectory", "", "_pick_folder", 0, 0),
    }
    for case, (dialog, answer, method, groups, files) in pickers.items():
        with subtests.test(msg=case):
            widget = make_widget()
            monkeypatch.setattr(QFileDialog, dialog,
                                staticmethod(lambda *a, _v=answer, **k: _v))
            getattr(widget, method)()
            assert (widget.group_count(), widget.file_count()) == (groups, files)

    class SettingsModel:
        def __init__(self, existing):
            self._widgets = {"dst": object()}
            self.existing = existing
            self.values = {}

        def _read_widget(self, widget):
            assert widget is self._widgets["dst"]
            return self.existing

        def set_value_for_key(self, key, value):
            self.values[key] = value

    imports = {
        "fills-an-empty-destination": (
            "", {"images": plate["images"],
                 "masks": {"cell": plate["cells"] / "fov001_cell_mask.tif"}},
            3, {"dst": str(plate["images"]) + "_spacr"}),
        "keeps-an-existing-destination": (
            "/work/keep-this-output", {"images": plate["images"], "masks": {}},
            2, {}),
    }
    for case, (existing, inputs, added, written) in imports.items():
        with subtests.test(msg=case):
            screen = QWidget()
            screen._settings_model = SettingsModel(existing)
            qtbot.addWidget(screen)
            view = make_widget(parent=QWidget(screen))
            assert view._use_test_data(inputs) == added
            assert view.file_count() == added
            assert screen._settings_model.values == written

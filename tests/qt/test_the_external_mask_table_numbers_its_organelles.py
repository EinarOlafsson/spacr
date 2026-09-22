"""The External Masks table calls an organelle slot by its number.

The maintainer, 2026-09-02: "i dont want to see organelle a b c d anywhere,
organelle number should always be controlled by number of organelles".
Measured 2026-09-19 (item 76): the Mask type box listed all 705 object types,
the slots captioned by ``str.title()`` -- ``Organelleb``, ``Organellee``,
``Organellezz`` -- and the Detected as column printed ``mask · organellee``.
Import Project and the settings forms already said ``Organelle 2``.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from spacr.qt.widgets.external_mask_inputs import ExternalMaskInputWidget


def _write(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(array), photometric="minisblack")


@pytest.fixture
def folders(tmp_path):
    yy, xx = np.indices((16, 16))
    images = tmp_path / "images"
    _write(images / "fov001_C1.tif", (yy * 16 + xx).astype(np.uint16))
    made = [images]
    for name in ("cell", "organelle_1", "organelle_5"):
        mask = np.zeros((16, 16), dtype=np.uint16)
        mask[3:9, 3:9] = 1
        folder = tmp_path / f"{name}_masks"
        _write(folder / f"fov001_{name}_mask.tif", mask)
        made.append(folder)
    return made


@pytest.fixture
def widget(qtbot):
    view = ExternalMaskInputWidget()
    qtbot.addWidget(view)
    return view


def _row_of(widget, object_type):
    for row in range(widget._table.rowCount()):
        index = widget._group_of_row(row)
        if widget.groups()[index].object_type == object_type:
            return row
    raise AssertionError(f"no row holds {object_type}")


def test_the_mask_type_box_numbers_the_slots_and_lists_only_a_few(
        widget, folders):
    widget.add_paths([str(folder) for folder in folders])
    box = widget._table.cellWidget(_row_of(widget, "organellee"), 3)
    captions = [box.itemText(index) for index in range(box.count())]

    assert captions[:4] == ["Choose…", "Cell", "Nucleus", "Pathogen"]
    assert captions[4:] == [f"Organelle {n}" for n in range(1, 6)]
    assert box.currentText() == "Organelle 5"
    assert not any(caption.startswith("Organelle") and
                   not caption.split()[-1].isdigit() for caption in captions)


def test_the_detected_column_says_organelle_5(widget, folders):
    widget.add_paths([str(folder) for folder in folders])
    row = _row_of(widget, "organellee")
    assert widget._table.item(row, 1).text() == "mask · Organelle 5"


def test_picking_a_slot_in_the_box_stores_the_role(widget, folders):
    widget.add_paths([str(folder) for folder in folders])
    row = _row_of(widget, "organelle")
    box = widget._table.cellWidget(row, 3)

    box.setCurrentIndex(box.findText("Organelle 4"))

    stored = widget.groups()[widget._group_of_row(row)].object_type
    assert stored == "organelled"
    assert "organelled" in [group["object_type"]
                            for group in widget.get_value()]

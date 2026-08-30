"""The External-Masks table when its rows and its groups drift apart.

The table draws one row per detected :class:`InputGroup`, but the row and
the group are two different things and they are allowed to disagree. The
table sorts, so the third row is not the third group after a header click;
that is why each source cell carries the group index in ``Qt.UserRole``
instead of the widget trusting ``row``. And Qt keeps the per-row combo
boxes alive after the row that held them is gone -- a removed row's role
box and object-type box are still live objects still wired to
``_role_changed``/``_object_changed`` with the index they were built for.

Every test here drives one of those disagreements. What breaks if the
mapping regresses is not cosmetic: a stale row index reaching
``del self._groups[i]`` deletes somebody else's folder from the run, and a
stale index reaching ``self._groups[i].role`` raises ``IndexError`` out of
a Qt signal handler, which in PySide6 tears down the whole settings page
rather than raising anywhere the user can see.

Covered here: the ``-1`` answer ``_group_of_row`` gives for a row it
cannot map (an undrawn row, and a row whose stored index outlived its
group), the bounds guards in both combo-box handlers, and the role handler
finishing its job when the object-type box beside it has been torn down.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from PySide6.QtCore import Qt

from spacr.external_masks import OBJECT_TYPES, ROLES
from spacr.qt.widgets.external_mask_inputs import ExternalMaskInputWidget


def _write(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(array), photometric="minisblack")
    return path


@pytest.fixture
def plate(tmp_path):
    """Real tifs in three folders, so detection produces the real roles."""
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
def widget(qtbot):
    view = ExternalMaskInputWidget()
    qtbot.addWidget(view)
    return view


# ---------------------------------------------------------------------------
# _group_of_row -- the row/group mapping and the two ways it can fail
# ---------------------------------------------------------------------------

def test_a_drawn_row_maps_to_the_group_it_was_built_from(widget, plate):
    """The mapping has to be right before its failure answer means anything.

    Every drawn row stamps its own group index into the source cell. If
    that stamp were ever dropped or written to the wrong column, removal
    and both combo boxes would silently address the wrong folder, and a
    user who greys out one mask folder would find a different one gone.
    """
    widget.set_value([str(plate["images"]), str(plate["cells"]),
                      str(plate["nuclei"])])

    assert widget._table.rowCount() == 3
    assert [widget._group_of_row(row) for row in range(3)] == [0, 1, 2]
    assert [widget._table.item(row, 0).data(Qt.UserRole)
            for row in range(3)] == [0, 1, 2]


def test_a_row_the_table_has_not_drawn_yet_deletes_nothing(widget, plate):
    """A half-built row must not be read as "delete group number N".

    ``_rebuild`` grows the table with ``setRowCount`` before it fills the
    new rows, so a row with no source item is a state the table really
    passes through. If such a row answered with a row number instead of
    ``-1``, "Remove selected" reaching it would delete whichever group
    happens to sit at that index -- a folder the user never selected.
    """
    widget.set_value([str(plate["images"]), str(plate["cells"])])
    phantom = widget._table.rowCount()
    widget._table.setRowCount(phantom + 1)

    assert widget._table.item(phantom, 0) is None
    assert widget._group_of_row(phantom) == -1

    seen = []
    widget.value_changed.connect(lambda: seen.append(1))
    widget._table.selectRow(phantom)
    widget.remove_selected()

    # The undrawn row removed nothing and announced nothing ...
    assert widget.group_count() == 2
    assert seen == []

    # ... while a drawn row in the same table still removes its own group.
    widget._table.selectRow(0)
    widget.remove_selected()
    assert widget.group_count() == 1
    assert widget.groups()[0].role == "mask"
    assert seen == [1]


def test_a_row_still_pointing_at_a_group_that_is_gone_is_ignored(widget,
                                                                 plate):
    """An index that outlived its group must not be clamped into range.

    The source cell stores a group index, not a row, so nothing stops it
    from naming an index past the end once groups have been removed. The
    guard is ``0 <= index < len(groups)``: without the upper bound the
    widget would either raise ``IndexError`` out of "Remove selected" or,
    worse, wrap and delete the last group in the run.
    """
    widget.set_value([str(plate["images"]), str(plate["cells"])])
    stale = widget._table.item(0, 0)
    stale.setData(Qt.UserRole, 9)

    assert widget._group_of_row(0) == -1

    seen = []
    widget.value_changed.connect(lambda: seen.append(1))
    widget._table.selectRow(0)
    widget.remove_selected()

    assert widget.group_count() == 2
    assert seen == []

    # Restoring a legal index makes the very same row removable again.
    stale.setData(Qt.UserRole, 0)
    assert widget._group_of_row(0) == 0
    widget._table.selectRow(0)
    widget.remove_selected()
    assert widget.group_count() == 1
    assert seen == [1]


def test_a_negative_stored_index_is_refused_as_well(widget, plate):
    """The lower bound matters on its own: ``-1`` is a legal Python index.

    Python would happily accept ``self._groups[-1]``, so a source cell
    holding a negative index would delete the LAST group in the table
    while the user believed they had selected the first one.
    """
    widget.set_value([str(plate["images"]), str(plate["cells"])])
    widget._table.item(0, 0).setData(Qt.UserRole, -1)

    assert widget._group_of_row(0) == -1

    widget._table.selectRow(0)
    widget.remove_selected()
    roles = [group.role for group in widget.groups()]
    assert roles == ["image", "mask"]


def test_a_missing_stamp_is_not_read_as_group_zero(widget, plate):
    """An empty ``UserRole`` reads as ``None``, which is not an index.

    ``item.data(Qt.UserRole)`` on a cell that never got the stamp returns
    ``None``. Coercing that with ``int()`` would raise ``TypeError``, and
    treating it as falsy would make it group 0 -- so the first folder in
    the table would be the one that gets deleted by any unstamped row.
    """
    widget.set_value([str(plate["cells"]), str(plate["nuclei"])])
    widget._table.item(1, 0).setData(Qt.UserRole, None)

    assert widget._table.item(1, 0).data(Qt.UserRole) is None
    assert widget._group_of_row(1) == -1
    assert widget._group_of_row(0) == 0

    widget._table.selectRow(1)
    widget.remove_selected()
    assert widget.group_count() == 2


# ---------------------------------------------------------------------------
# _role_changed -- a combo box that outlived the row it was built for
# ---------------------------------------------------------------------------

def test_a_role_box_left_over_from_a_removed_row_writes_nowhere(widget,
                                                                plate):
    """Qt keeps the removed row's combo box alive and still connected.

    Shrinking the table does not disconnect the lambda that captured
    ``r=1``; the box is still a live object bound to an index that no
    longer exists. Any late change on it -- a queued signal, a programmatic
    reset -- reaches ``_role_changed(1, ...)`` after ``_groups`` has one
    entry. Without the bounds guard that is ``IndexError`` raised inside a
    Qt slot, which aborts the settings page instead of being reported.
    """
    widget.set_value([str(plate["images"]), str(plate["cells"])])
    stale_box = widget._table.cellWidget(1, 2)

    widget._table.selectRow(1)
    widget.remove_selected()
    assert widget.group_count() == 1

    seen = []
    widget.value_changed.connect(lambda: seen.append(1))
    stale_box.setCurrentIndex(stale_box.findData("ignore"))

    # The surviving group kept the role detection gave it, and the widget
    # did not tell the settings page anything had changed.
    assert widget.groups()[0].role == "image"
    assert seen == []

    # The row that IS drawn still writes through, so the guard is a bounds
    # check and not a dead handler.
    live_box = widget._table.cellWidget(0, 2)
    live_box.setCurrentIndex(live_box.findData("ignore"))
    assert widget.groups()[0].role == "ignore"
    assert seen == [1]
    assert "ignore" in ROLES


def test_a_role_change_still_lands_when_the_object_box_is_gone(widget,
                                                               plate):
    """Greying the neighbouring box is a courtesy, not a precondition.

    ``_role_changed`` greys the object-type box beside the row it changed,
    but that box is a child of a cell the table can tear down at any time.
    If the handler assumed it were always there, a role change arriving
    after the teardown would raise ``AttributeError: 'NoneType' object has
    no attribute 'setEnabled'`` and the user's choice of "image" would be
    lost even though nothing about the group was wrong.
    """
    widget.set_value([str(plate["cells"])])
    assert widget.groups()[0].role == "mask"
    widget._table.removeCellWidget(0, 3)
    assert widget._table.cellWidget(0, 3) is None

    seen = []
    widget.value_changed.connect(lambda: seen.append(1))
    role_box = widget._table.cellWidget(0, 2)
    role_box.setCurrentIndex(role_box.findData("image"))

    assert widget.groups()[0].role == "image"
    assert seen == [1]
    assert widget.get_value()[0]["role"] == "image"


def test_the_object_box_is_greyed_when_the_row_stops_being_a_mask(widget,
                                                                  plate):
    """The branch the torn-down case skips has to do its job when it runs.

    An image row has no object type, so leaving its object-type box
    clickable invites the user to pick "nucleus" for a plane that will
    never be written -- and enabling it again is what makes a row usable
    after they change their mind back.
    """
    widget.set_value([str(plate["cells"])])
    role_box = widget._table.cellWidget(0, 2)
    object_box = widget._table.cellWidget(0, 3)
    assert object_box.isEnabled() is True

    role_box.setCurrentIndex(role_box.findData("image"))
    assert object_box.isEnabled() is False

    role_box.setCurrentIndex(role_box.findData("mask"))
    assert object_box.isEnabled() is True
    assert widget.groups()[0].role == "mask"


# ---------------------------------------------------------------------------
# _object_changed -- the same drift on the object-type box
# ---------------------------------------------------------------------------

def test_an_object_box_left_over_from_a_removed_row_writes_nowhere(widget,
                                                                   plate):
    """A stale object-type box must not stamp a plane onto a live group.

    The object type decides which ``masks/<object>_mask_stack`` folder a
    run writes, so a stale box that landed on the surviving group would
    silently relabel the user's cell masks as nuclei -- and Measure would
    then extract nucleus features from cell outlines without complaining.
    """
    widget.set_value([str(plate["cells"]), str(plate["nuclei"])])
    stale_box = widget._table.cellWidget(1, 3)

    widget._table.selectRow(1)
    widget.remove_selected()
    assert widget.group_count() == 1
    kept = widget.groups()[0].object_type

    seen = []
    widget.value_changed.connect(lambda: seen.append(1))
    stale_box.setCurrentIndex(stale_box.findData("pathogen"))

    assert widget.groups()[0].object_type == kept
    assert seen == []

    # The drawn row's box still writes through and still announces itself.
    live_box = widget._table.cellWidget(0, 3)
    live_box.setCurrentIndex(live_box.findData("pathogen"))
    assert widget.groups()[0].object_type == "pathogen"
    assert seen == [1]
    assert "pathogen" in OBJECT_TYPES

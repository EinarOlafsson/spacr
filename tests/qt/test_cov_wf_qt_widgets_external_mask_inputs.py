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


# ---------------------------------------------------------------------------
# The same three guards again, reached without touching the disk.
#
# Everything above builds its groups by running detect_inputs over real tifs,
# so every one of those tests carries the whole detection heuristic, tifffile
# and a tmp_path with it. The guards below are pure index arithmetic and do
# not need any of that: these tests hand the widget the group dicts directly,
# which is both the second supported shape of set_value and a route to the
# guards that cannot be broken by a change in how folders are classified.
# ---------------------------------------------------------------------------

def _group_dict(key, role, object_type=None, count=1):
    """One InputGroup as the mapping ``set_value`` accepts from a saved run."""
    return {
        "key": key,
        "root": f"/data/{key}",
        "paths": [f"/data/{key}/fov{n:03d}.tif" for n in range(1, count + 1)],
        "role": role,
        "object_type": object_type,
        "confidence": 0.75,
        "reason": f"restored {key}",
    }


def test_a_synthetic_row_with_no_source_cell_maps_to_no_group(widget):
    """Rows outnumbering groups is a state the table really passes through.

    ``_rebuild`` calls ``setRowCount`` first and fills the cells afterwards,
    so between those two statements the table holds rows whose source cell
    is still ``None``. "Remove selected" firing on such a row -- a queued
    click, a selection restored by the settings page -- must resolve to no
    group at all. If it resolved to its own row number instead, the widget
    would delete whichever folder happened to sit at that index, and the
    user would lose a mask folder they never selected from the run.
    """
    widget.set_value([_group_dict("images", "image", count=2),
                      _group_dict("cells", "mask", "cell")])
    undrawn = widget._table.rowCount()
    widget._table.setRowCount(undrawn + 2)

    assert widget._table.item(undrawn, 0) is None
    assert widget._group_of_row(undrawn) == -1
    assert widget._group_of_row(undrawn + 1) == -1
    # The rows that were drawn still resolve, so this is a mapping failure
    # and not a handler that has stopped answering.
    assert [widget._group_of_row(row) for row in range(undrawn)] == [0, 1]

    announced = []
    widget.value_changed.connect(lambda: announced.append(1))
    widget._table.clearSelection()
    widget._table.selectRow(undrawn)
    widget.remove_selected()

    assert widget.group_count() == 2
    assert announced == []
    assert [group.key for group in widget.groups()] == ["images", "cells"]


def test_a_role_arriving_for_a_row_past_the_end_changes_nothing(widget):
    """A role box outlives its row, and its captured index outlives its group.

    The lambda wired to each role box captures the row it was built for, and
    Qt keeps that box alive after the table has shrunk under it. So
    ``_role_changed`` is genuinely called with indices that no longer name a
    group. Without the bounds check the very next statement is
    ``self._groups[row].role = role`` -- an ``IndexError`` raised inside a Qt
    slot, which in PySide6 takes down the settings page instead of surfacing
    anywhere the user can act on.
    """
    widget.set_value([_group_dict("cells", "mask", "cell")])

    announced = []
    widget.value_changed.connect(lambda: announced.append(1))
    widget._role_changed(4, "ignore")
    widget._role_changed(-1, "ignore")

    # Nothing was written and nothing was announced for either bad index ...
    assert widget.groups()[0].role == "mask"
    assert widget.groups()[0].object_type == "cell"
    assert announced == []

    # ... while the one row that does exist still writes through, so the
    # guard is a bounds check rather than a handler that has stopped working.
    widget._role_changed(0, "image")
    assert widget.groups()[0].role == "image"
    assert announced == [1]


def test_a_role_still_lands_when_the_object_cell_holds_no_widget(widget):
    """Greying the neighbouring box is a courtesy, not a precondition.

    The object-type box lives in a cell the table owns and can tear down --
    a rebuild, a sort, a removed row. ``_role_changed`` looks it up fresh
    every time precisely because it may be gone by then. If the handler
    assumed the box were always there it would raise ``AttributeError`` on
    ``None.setEnabled`` and the user's choice of "image" would be dropped,
    even though there was nothing wrong with the group itself.
    """
    widget.set_value([_group_dict("cells", "mask", "cell"),
                      _group_dict("nuclei", "mask", "nucleus")])
    widget._table.removeCellWidget(0, 3)
    assert widget._table.cellWidget(0, 3) is None

    announced = []
    widget.value_changed.connect(lambda: announced.append(1))
    widget._role_changed(0, "image")

    assert widget.groups()[0].role == "image"
    assert announced == [1]

    # The row that still has its box gets it greyed out, so the skipped
    # branch is doing real work whenever the cell is populated.
    surviving = widget._table.cellWidget(1, 3)
    assert surviving.isEnabled() is True
    widget._role_changed(1, "ignore")
    assert surviving.isEnabled() is False
    assert [group.role for group in widget.groups()] == ["image", "ignore"]
    # The ignored row drops out of the saved value; the row whose role
    # landed without its box is still there to be run.
    saved = widget.get_value()
    assert [item["key"] for item in saved] == ["cells"]
    assert saved[0]["role"] == "image"


def test_an_object_type_arriving_for_a_row_past_the_end_changes_nothing(
        widget):
    """A stale object-type box must not relabel a group that is still live.

    The object type decides which ``masks/<object>_mask_stack`` folder the
    run writes, so a leftover box landing on whatever group now sits at its
    captured index would silently retype the user's cell masks as nuclei --
    and Measure would then extract nucleus features from cell outlines
    without any complaint the user could notice.
    """
    widget.set_value([_group_dict("cells", "mask", "cell")])

    announced = []
    widget.value_changed.connect(lambda: announced.append(1))
    widget._object_changed(3, "nucleus")
    widget._object_changed(-2, "nucleus")

    assert widget.groups()[0].object_type == "cell"
    assert announced == []

    # The live row accepts the same value, so the bad indices were refused
    # for being out of range and not for naming an unknown object type.
    widget._object_changed(0, "nucleus")
    assert widget.groups()[0].object_type == "nucleus"
    assert announced == [1]
    assert "nucleus" in OBJECT_TYPES

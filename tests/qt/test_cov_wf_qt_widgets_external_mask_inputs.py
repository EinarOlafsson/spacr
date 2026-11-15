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

from spacr.external_masks import OBJECT_TYPES
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



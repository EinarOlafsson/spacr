"""The plate-map picker's press-drag-release gesture, regions and layout dialog.

``select_region`` once existed with nothing calling it from the mouse, so the
wiring is driven with real Qt mouse events at real well buttons as well as
through the drag API directly.  Every case asserts the exact set of wells
chosen afterwards.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QInputDialog, QWidget

from spacr.qt.widgets.plate_map_picker import PlateMapPicker, _Well
from spacr.well_spec import DEFAULT_LAYOUT, LAYOUTS


@pytest.fixture
def make_picker(qapp):
    """Build fresh 96-well pickers, closed again when the test ends."""
    made = []

    def build():
        dialog = PlateMapPicker(layout=96)
        dialog.show()
        qapp.processEvents()
        made.append(dialog)
        return dialog

    yield build
    for dialog in made:
        dialog.close()
        dialog.deleteLater()


def _rect(rows, cols):
    return {(r, c) for r in rows for c in cols}


def _press(well, modifiers=Qt.NoModifier):
    centre = well.rect().center()
    QApplication.sendEvent(well, QMouseEvent(
        QMouseEvent.MouseButtonPress, centre, well.mapToGlobal(centre),
        Qt.LeftButton, Qt.LeftButton, modifiers))


def _move_to(source, target):
    """A move over ``target`` delivered to ``source``, which holds the grab."""
    centre = target.rect().center()
    QApplication.sendEvent(source, QMouseEvent(
        QMouseEvent.MouseMove,
        source.mapFromGlobal(target.mapToGlobal(centre)),
        target.mapToGlobal(centre), Qt.NoButton, Qt.LeftButton,
        Qt.NoModifier))


def _release(well):
    centre = well.rect().center()
    event = QMouseEvent(QMouseEvent.MouseButtonRelease, centre,
                        well.mapToGlobal(centre), Qt.LeftButton,
                        Qt.NoButton, Qt.NoModifier)
    QApplication.sendEvent(well, event)
    return event


def test_real_mouse_events_anchor_preview_and_finish_the_drag(
        make_picker, qapp, subtests):
    """Press anchors (Ctrl makes it add), a move previews, a still click toggles."""
    with subtests.test(msg="ctrl-press-adds"):
        picker = make_picker()
        holder = QWidget(picker)
        well = _Well(3, 4, holder)
        picker.set_selection({(8, 11)})
        _press(well, modifiers=Qt.ControlModifier)
        assert (picker._anchor, picker._adding, picker._before) == (
            (3, 4), True, {(8, 11)})
    with subtests.test(msg="move-while-pressed-previews"):
        picker = make_picker()
        well = _Well(3, 4, QWidget(picker))
        _press(well)
        assert (picker._anchor, picker._adding) == ((3, 4), False)
        _move_to(well, picker._wells[(5, 6)])
        assert picker.selection() == _rect((3, 4, 5), (4, 5, 6))
    with subtests.test(msg="drag-across-the-real-grid"):
        picker = make_picker()
        start = picker._wells[(2, 2)]
        _press(start)
        _move_to(start, picker._wells[(4, 5)])
        _release(start)
        qapp.processEvents()
        assert picker.selection() == _rect((2, 3, 4), (2, 3, 4, 5))
        assert picker._anchor is None
    with subtests.test(msg="press-and-release-without-moving-clicks"):
        picker = make_picker()
        well = picker._wells[(6, 6)]
        assert not well.isChecked()
        _press(well)
        _release(well)
        qapp.processEvents()
        assert well.isChecked()
        assert picker.selection() == {(6, 6)}


def test_the_drag_api_previews_from_the_press_state_and_reports_a_drag(
        make_picker, subtests):
    """Plain drags replace, Ctrl drags add; moves off the grid, over the anchor or unpressed do nothing."""
    with subtests.test(msg="plain-drag-replaces"):
        picker = make_picker()
        picker.set_selection({(8, 11)})
        picker.begin_drag(1, 1)
        picker.drag_to(picker._wells[(2, 2)].mapToGlobal(QPoint(2, 2)))
        assert picker.finish_drag() is True
        assert picker.selection() == _rect((1, 2), (1, 2))
    with subtests.test(msg="ctrl-drag-adds"):
        picker = make_picker()
        picker.set_selection({(8, 11)})
        picker.begin_drag(1, 1, Qt.ControlModifier)
        picker.drag_to(picker._wells[(2, 2)].mapToGlobal(QPoint(2, 2)))
        picker.finish_drag()
        assert picker.selection() == _rect((1, 2), (1, 2)) | {(8, 11)}
    with subtests.test(msg="same-well-move-does-not-redraw"):
        picker = make_picker()
        picker.begin_drag(2, 2)
        point = picker._wells[(2, 4)].mapToGlobal(QPoint(2, 2))
        picker.drag_to(point)
        assert picker.selection() == _rect((2,), (2, 3, 4))
        picker.set_selection(set())
        picker.drag_to(point)
        assert picker.selection() == set()
    with subtests.test(msg="never-leaving-the-anchor-is-no-drag"):
        picker = make_picker()
        picker.set_selection({(8, 11)})
        picker.begin_drag(2, 2)
        picker.drag_to(picker._wells[(2, 2)].mapToGlobal(QPoint(3, 3)))
        assert picker.selection() == {(8, 11)}
        assert picker.finish_drag() is False
    with subtests.test(msg="move-off-the-grid-keeps-the-preview"):
        picker = make_picker()
        picker.begin_drag(2, 2)
        picker.drag_to(picker._wells[(2, 4)].mapToGlobal(QPoint(2, 2)))
        picker.drag_to(picker.mapToGlobal(QPoint(-500, -500)))
        assert picker.selection() == _rect((2,), (2, 3, 4))
    with subtests.test(msg="move-with-no-press-does-nothing"):
        picker = make_picker()
        picker.set_selection({(1, 1)})
        picker._anchor = None
        picker.drag_to(picker._wells[(3, 3)].mapToGlobal(QPoint(2, 2)))
        assert picker.selection() == {(1, 1)}


def test_select_region_is_corner_order_free_clipped_and_inverts_by_default(
        make_picker, subtests):
    """Any corner order, clipped to the plate; with no verdict the start well decides."""
    cases = {
        "reversed-corners": ([((5, 6), (3, 4), True)], _rect((3, 4, 5), (4, 5, 6))),
        "runs-off-the-plate": ([((7, 11), (20, 40), True)],
                               _rect((7, 8), (11, 12))),
        "no-verdict-selects-from-an-empty-well": (
            [((1, 1), (2, 2), None)], _rect((1, 2), (1, 2))),
        "no-verdict-clears-from-a-chosen-well": (
            [((1, 1), (2, 2), None), ((1, 1), (2, 2), None)], set()),
    }
    for case, (calls, expected) in cases.items():
        with subtests.test(msg=case):
            picker = make_picker()
            for start, end, choosing in calls:
                if choosing is None:
                    picker.select_region(start, end)
                else:
                    picker.select_region(start, end, choosing=choosing)
            assert picker.selection() == expected


def test_the_layout_dialog_offers_every_layout_and_reports_dropped_wells(
        make_picker, monkeypatch, subtests):
    """The dialog lists every layout opened on the current one; cancel keeps it; a change says what it dropped."""
    sizes = [str(size) for size in sorted(LAYOUTS)]
    with subtests.test(msg="offers-every-layout"):
        picker = make_picker()
        seen = {}

        def _ask(parent, title, label, items, current, editable):
            seen.update(items=list(items), current=current)
            return "384", True

        monkeypatch.setattr(QInputDialog, "getItem", staticmethod(_ask))
        assert picker.ask_for_layout() == 384
        assert seen == {"items": sizes, "current": sizes.index("96")}
    with subtests.test(msg="cancel-keeps-the-layout"):
        picker = make_picker()
        monkeypatch.setattr(QInputDialog, "getItem",
                            staticmethod(lambda *a, **k: ("384", False)))
        assert picker.ask_for_layout() == 96
        assert picker._layout_size == 96
    with subtests.test(msg="unknown-current-opens-on-the-default"):
        picker = make_picker()
        seen = {}

        def _ask_first(parent, title, label, items, current, editable):
            seen["current"] = current
            return items[0], True

        monkeypatch.setattr(QInputDialog, "getItem", staticmethod(_ask_first))
        picker._layout_size = 999
        picker.ask_for_layout()
        assert seen["current"] == sorted(LAYOUTS).index(DEFAULT_LAYOUT)
    with subtests.test(msg="a-change-says-what-it-dropped"):
        picker = make_picker()
        picker.set_selection({(1, 1), (8, 12)})
        picker.ask_for_layout(6)
        assert picker.selection() == {(1, 1)}
        assert "1 well(s) from the previous selection" in picker._caption.text()
        assert "6-well plate (2 x 3)" in picker._caption.text()

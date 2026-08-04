"""``B16`` — the synchronised comparison grid.

:mod:`tests.test_canvas_link` covers the shared-window model with no Qt at
all. What is left here is the grid: that a wheel over one cell moves every
other cell to the same place, that unequal cells stay at one magnification
rather than one field of view, that unlocking one panel really does leave the
others alone, and that a click in one panel reaches the others — and every
other view in the app — through the shared selection.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from spacr.layers import FieldKey, LayerError, LayerStack, Spacing
from spacr.qt import comparison_grid as cg
from spacr.qt.linked_selection import LinkedSelection, Selection


PLATE_KEY_VALUES = ("plate1", "A", "1", "1")


def _field_key():
    return FieldKey(values=dict(zip(FieldKey.columns(), PLATE_KEY_VALUES)))


def _channel(seed=0, *, with_mask=True, label=17, size=64):
    """One channel of the field, optionally with the shared mask over it."""
    stack = LayerStack()
    rng = np.random.default_rng(seed)
    stack.add_image(rng.integers(0, 4000, (size, size)).astype(np.uint16),
                    name=f"ch{seed}", contrast_limits=(0.0, 4000.0))
    if with_mask:
        mask = np.zeros((size, size), np.int32)
        mask[20:30, 20:30] = label
        stack.add_labels(mask, name="mask", field=_field_key(), opacity=0.5)
    return stack


def _grid(qtbot, panels=None, **kwargs):
    grid = cg.ComparisonGrid(
        panels if panels is not None
        else {f"ch{i}": _channel(i) for i in range(4)}, **kwargs)
    qtbot.addWidget(grid)
    grid.resize(600, 600)
    grid.show()
    qtbot.waitExposed(grid)
    return grid


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def test_the_panels_land_in_a_square_grid_with_their_own_stacks(
        qtbot, qt_theme_applied):
    grid = _grid(qtbot)
    assert list(grid.panels) == ["ch0", "ch1", "ch2", "ch3"]
    positions = {grid.grid.getItemPosition(i)[:2]
                 for i in range(grid.grid.count())}
    assert positions == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert grid.panels["ch0"].stack is not grid.panels["ch1"].stack
    assert "4 panel(s)" in grid.status.text()


def test_the_column_count_can_be_chosen(qtbot, qt_theme_applied):
    grid = _grid(qtbot, columns=4)
    rows = {grid.grid.getItemPosition(i)[0] for i in range(grid.grid.count())}
    assert rows == {0}


def test_panels_can_be_given_as_pairs_with_captions(qtbot, qt_theme_applied):
    grid = _grid(qtbot, [("dapi", _channel(0)), ("phalloidin", _channel(1))],
                 titles={"dapi": "DNA"})
    assert grid.panels["dapi"].caption.text() == "DNA"
    assert grid.panels["phalloidin"].caption.text() == "phalloidin"


def test_an_empty_grid_says_it_is_empty(qtbot, qt_theme_applied):
    grid = _grid(qtbot, {})
    assert grid.status.text() == "No panels"
    assert grid.panels == {}


# ---------------------------------------------------------------------------
# The shared window
# ---------------------------------------------------------------------------

def test_every_panel_starts_at_one_magnification(qtbot, qt_theme_applied):
    grid = _grid(qtbot)
    steps = {panel.canvas.canvas.step for panel in grid.panels.values()}
    assert len(steps) == 1


def test_a_wheel_over_one_cell_zooms_every_cell_to_the_same_place(
        qtbot, qt_theme_applied):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QWheelEvent

    grid = _grid(qtbot)
    driver = grid.panels["ch0"].canvas
    anchor = driver.canvas.world_at(20.0, 30.0)
    moved = []
    grid.view_changed.connect(moved.append)

    driver.wheelEvent(QWheelEvent(
        QPointF(31.0, 21.0), driver.mapToGlobal(QPoint(31, 21)),
        QPoint(0, 120), QPoint(0, 120), Qt.NoButton, Qt.NoModifier,
        Qt.NoScrollPhase, False))

    assert moved == ["ch0"]
    steps = {panel.canvas.canvas.step for panel in grid.panels.values()}
    assert len(steps) == 1, "the panels ended up at different magnifications"
    for key, panel in grid.panels.items():
        assert panel.canvas.canvas.pixel_at(anchor) == pytest.approx(
            (20.0, 30.0), abs=1e-6), f"{key} slid out from under the cursor"


def test_dragging_one_cell_pans_every_cell(qtbot, qt_theme_applied):
    grid = _grid(qtbot)
    driver = grid.panels["ch0"].canvas
    before = grid.panels["ch3"].canvas.canvas.origin

    QTest.mousePress(driver, Qt.LeftButton, Qt.ShiftModifier, QPoint(60, 60))
    driver.mouseMoveEvent(_move(driver, 40.0, 30.0))
    QTest.mouseRelease(driver, Qt.LeftButton, Qt.ShiftModifier, QPoint(40, 30))

    after = grid.panels["ch3"].canvas.canvas.origin
    assert after != before
    assert after == pytest.approx(grid.panels["ch0"].canvas.canvas.origin)


def _move(widget, x, y):
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent
    return QMouseEvent(QEvent.MouseMove, QPointF(x, y),
                       widget.mapToGlobal(QPoint(int(x), int(y))),
                       Qt.NoButton, Qt.LeftButton, Qt.ShiftModifier)


def test_unequal_cells_share_the_magnification_not_the_field_of_view(
        qtbot, qt_theme_applied):
    """A narrower cell shows less of the sample, not the same sample smaller."""
    grid = _grid(qtbot)
    narrow = grid.panels["ch1"].canvas
    wide = grid.panels["ch0"].canvas
    narrow.setFixedWidth(wide.width() // 2)
    qtbot.waitUntil(lambda: narrow.width() < wide.width())
    narrow._ensure_canvas()

    assert narrow.canvas.step == pytest.approx(wide.canvas.step), (
        "the narrower cell changed magnification, so the two pictures are no "
        "longer comparable")
    assert narrow.canvas.shape[1] < wide.canvas.shape[1]


def test_fit_puts_every_linked_panel_back_on_the_whole_field(qtbot,
                                                             qt_theme_applied):
    def visible_width(panel):
        canvas = panel.canvas.canvas
        return canvas.step[1] * canvas.shape[1]

    grid = _grid(qtbot)
    driver = grid.panels["ch0"].canvas
    driver._canvas = driver.canvas.zoomed(6.0)
    driver.view_changed.emit()
    assert visible_width(grid.panels["ch2"]) < 64.0, "the zoom did not travel"

    grid.fit_button.click()

    # The whole 64-unit field is in view again, in every panel, at one scale.
    assert len({p.canvas.canvas.step for p in grid.panels.values()}) == 1
    for panel in grid.panels.values():
        assert visible_width(panel) >= 64.0


# ---------------------------------------------------------------------------
# Letting one panel go
# ---------------------------------------------------------------------------

def test_unchecking_a_panel_leaves_it_where_it_is(qtbot, qt_theme_applied):
    grid = _grid(qtbot)
    grid.panels["ch2"].lock_box.setChecked(False)
    parked = grid.panels["ch2"].canvas.canvas.origin
    assert "1 free (ch2)" in grid.status.text()

    driver = grid.panels["ch0"].canvas
    driver._canvas = driver.canvas.panned(10, 10)
    driver.view_changed.emit()

    assert grid.panels["ch2"].canvas.canvas.origin == pytest.approx(parked)
    assert grid.panels["ch1"].canvas.canvas.origin == pytest.approx(
        driver.canvas.origin)


def test_a_free_panel_can_be_moved_without_taking_the_others(qtbot,
                                                             qt_theme_applied):
    grid = _grid(qtbot)
    grid.panels["ch2"].lock_box.setChecked(False)
    others = grid.panels["ch0"].canvas.canvas.origin

    free = grid.panels["ch2"].canvas
    free._canvas = free.canvas.panned(25, 25)
    free.view_changed.emit()

    assert grid.panels["ch0"].canvas.canvas.origin == pytest.approx(others)


def test_link_all_brings_every_panel_back(qtbot, qt_theme_applied):
    grid = _grid(qtbot)
    grid.panels["ch2"].lock_box.setChecked(False)
    free = grid.panels["ch2"].canvas
    free._canvas = free.canvas.panned(25, 25)
    free.view_changed.emit()

    grid.lock_all_button.click()

    assert "all linked" in grid.status.text()
    assert grid.panels["ch2"].lock_box.isChecked()
    origins = {tuple(p.canvas.canvas.origin) for p in grid.panels.values()}
    assert len(origins) == 1


# ---------------------------------------------------------------------------
# Selection across the panels
# ---------------------------------------------------------------------------

def test_a_click_in_one_panel_selects_the_object_in_all_of_them(
        qtbot, qt_theme_applied):
    link = LinkedSelection()
    grid = _grid(qtbot)
    grid.link_selection(cg.GRID_LINK_SOURCE, link=link)
    picked = []
    grid.object_picked.connect(picked.append)

    canvas = grid.panels["ch0"].canvas
    row, column = canvas.canvas.pixel_at({"y": 25.0, "x": 25.0})
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier,
                     QPoint(int(round(column)) + 1, int(round(row)) + 1))

    assert picked == ["plate1_A_1_1_17"]
    for panel in grid.panels.values():
        assert panel.stack["mask"].selected_label == 17
    assert list(link.selection.keys) == ["plate1_A_1_1_17"]


def test_a_selection_from_elsewhere_reaches_every_panel(qtbot,
                                                        qt_theme_applied):
    link = LinkedSelection()
    grid = _grid(qtbot)
    grid.link_selection(cg.GRID_LINK_SOURCE, link=link)

    link.set_selection(Selection(keys=["plate1_A_1_1_17"], source="umap"))

    assert all(panel.stack["mask"].selected_label == 17
               for panel in grid.panels.values())
    assert "in 4 of 4 panel(s)" in grid.status.text()


def test_an_object_only_some_panels_hold_is_reported_honestly(qtbot,
                                                              qt_theme_applied):
    grid = _grid(qtbot, {"has": _channel(0, label=17),
                         "hasnt": _channel(1, label=18),
                         "none": _channel(2, with_mask=False)})
    assert grid.highlight("plate1_A_1_1_17") == ["has"]
    assert "in 1 of 3 panel(s)" in grid.status.text()


def test_clicking_the_background_publishes_nothing(qtbot, qt_theme_applied):
    link = LinkedSelection()
    grid = _grid(qtbot, {"a": _channel(0)})
    grid.link_selection(cg.GRID_LINK_SOURCE, link=link)
    picked = []
    grid.object_picked.connect(picked.append)

    canvas = grid.panels["a"].canvas
    row, column = canvas.canvas.pixel_at({"y": 2.0, "x": 2.0})
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier,
                     QPoint(int(round(column)) + 1, int(round(row)) + 1))

    assert picked == []
    assert link.selection.keys is None


def test_a_multi_object_selection_is_not_a_place_to_go(qtbot, qt_theme_applied):
    link = LinkedSelection()
    grid = _grid(qtbot, {"a": _channel(0)})
    grid.link_selection(cg.GRID_LINK_SOURCE, link=link)
    link.set_selection(Selection(keys=["plate1_A_1_1_17", "plate1_A_1_1_18"],
                                 source="umap"))
    assert grid.panels["a"].stack["mask"].selected_label == 0


# ---------------------------------------------------------------------------
# Adding, removing and teardown
# ---------------------------------------------------------------------------

def test_a_panel_added_to_a_zoomed_grid_starts_where_the_others_are(
        qtbot, qt_theme_applied):
    grid = _grid(qtbot, {"a": _channel(0), "b": _channel(1)})
    driver = grid.panels["a"].canvas
    driver._canvas = driver.canvas.zoomed(4.0)
    driver.view_changed.emit()

    grid.add_panel("c", _channel(2))
    qtbot.waitUntil(lambda: grid.panels["c"].canvas.canvas is not None)

    assert grid.panels["c"].canvas.canvas.step == pytest.approx(
        driver.canvas.step)
    assert grid.panels["c"].canvas.canvas.origin == pytest.approx(
        driver.canvas.origin)


def test_a_panel_key_is_used_once(qtbot, qt_theme_applied):
    grid = _grid(qtbot, {"a": _channel(0)})
    with pytest.raises(LayerError, match="already in this grid"):
        grid.add_panel("a", _channel(1))


def test_removing_a_panel_takes_it_out_of_the_grid_and_the_link(
        qtbot, qt_theme_applied):
    grid = _grid(qtbot)
    grid.remove_panel("ch1")
    assert list(grid.panels) == ["ch0", "ch2", "ch3"]
    assert "ch1" not in grid.canvas_link
    assert "3 panel(s)" in grid.status.text()
    with pytest.raises(LayerError, match="no panel"):
        grid.remove_panel("ch1")


def test_the_canvas_link_is_not_the_selection_link(qtbot, qt_theme_applied):
    """Two different links; confusing them is how a view talks to itself."""
    from spacr.layers import CanvasLink
    from spacr.qt.linked_selection import LinkedSelection as _Sel

    grid = _grid(qtbot, {"a": _channel(0)})
    assert isinstance(grid.canvas_link, CanvasLink)
    assert isinstance(grid.link, _Sel)


def test_the_grid_lets_go_of_everything_when_it_closes(qtbot,
                                                       qt_theme_applied):
    link = LinkedSelection()
    grid = _grid(qtbot, {"a": _channel(0)})
    grid.link_selection(cg.GRID_LINK_SOURCE, link=link)
    stack = grid.panels["a"].stack
    grid.close()

    assert not grid.is_linked
    # The canvas unsubscribed: the model no longer drives a closed widget.
    assert stack._listeners == [] or all(
        getattr(l, "__self__", None) is not grid.panels["a"].canvas
        for l in stack._listeners)

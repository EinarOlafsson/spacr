"""``B15`` — the orthogonal-view widget and its dimension sliders.

:mod:`tests.test_ortho_views` covers the geometry with no Qt at all. What is
left here is the widget: that the sliders are in world units and land on
planes that exist, that clicking a side panel moves the crosshair the top
panel is showing, and that a click also reaches the shared selection.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from spacr.layers import FieldKey, LayerStack, OrthoViews, Spacing
from spacr.qt import ortho_view as ov
from spacr.qt.linked_selection import LinkedSelection, Selection


CONFOCAL = Spacing.from_map({"z": 2.0, "y": 0.65, "x": 0.65}, units="um")
PLATE_KEY_VALUES = ("plate1", "A", "1", "1")


def _field_key():
    return FieldKey(values=dict(zip(FieldKey.columns(), PLATE_KEY_VALUES)))


def _volume(shape=(10, 64, 64), spacing=CONFOCAL, with_mask=False):
    stack = LayerStack()
    data = np.zeros(shape, np.uint16)
    data[shape[0] // 2, 30:34, 30:34] = 4000
    stack.add_image(data, name="volume", spacing=spacing,
                    contrast_limits=(0.0, 4000.0))
    if with_mask:
        mask = np.zeros(shape, np.int32)
        mask[shape[0] // 2, 30:34, 30:34] = 17
        stack.add_labels(mask, name="mask", spacing=spacing,
                         field=_field_key())
    return stack


def _view(qtbot, stack=None, **kwargs):
    view = ov.OrthoView(stack if stack is not None else _volume(), **kwargs)
    qtbot.addWidget(view)
    view.resize(520, 520)
    return view


# ---------------------------------------------------------------------------
# The panels
# ---------------------------------------------------------------------------

def test_the_three_panels_are_painted_at_one_scale(qtbot, qt_theme_applied):
    view = _view(qtbot, width=128)
    steps = {panel.canvas.step for panel in view.panels.values()}
    assert len(steps) == 1
    assert view.panels["zx"].canvas.shape[0] == 62, (
        "the side panel is one pixel per slice; the z voxel size was ignored")
    assert view.panels["xy"].name == "xy"


def test_a_2d_field_says_it_has_no_second_plane(qtbot, qt_theme_applied):
    stack = LayerStack()
    stack.add_image(np.zeros((32, 32), np.uint16), name="field")
    view = _view(qtbot, stack)
    assert view.views is None
    assert "no second plane" in view.status.text()
    assert view.panels["xy"].canvas is None
    # The sliders went with it rather than pointing at nothing.
    assert view._sliders == {}
    # And driving it anyway does nothing rather than raising.
    view.move_to(z=1.0)
    view.zoom_in()
    view._on_panel_clicked("xy", 1.0, 1.0)


def test_the_panels_paint_without_raising(qtbot, qt_theme_applied):
    view = _view(qtbot, width=96)
    view.show()
    qtbot.waitExposed(view)
    for panel in view.panels.values():
        panel.repaint()


# ---------------------------------------------------------------------------
# The sliders
# ---------------------------------------------------------------------------

def test_there_is_a_slider_per_axis_labelled_in_world_units(qtbot,
                                                            qt_theme_applied):
    view = _view(qtbot, width=128)
    assert sorted(view._sliders) == ["x", "y", "z"]
    readout = view._sliders["z"].property("readout").text()
    assert "um" in readout and "slice" in readout


def test_the_slider_lands_on_planes_that_exist(qtbot, qt_theme_applied):
    """Not between two of them, and never one past the last."""
    view = _view(qtbot, width=128)
    seen = []
    view.point_changed.connect(seen.append)

    for tick, expected in ((0, 0), (500, 4), (1000, 9)):
        view._sliders["z"].setValue(tick)
        assert view.slice_index("z") == expected
        assert view.views.point["z"] == pytest.approx(expected * 2.0)
    assert len(seen) == 3
    assert seen[-1]["z"] == pytest.approx(18.0)


def test_moving_the_crosshair_moves_the_slider_back(qtbot, qt_theme_applied):
    view = _view(qtbot, width=128)
    view.move_to(z=12.0)
    assert view.slice_index("z") == 6
    assert view._sliders["z"].value() == pytest.approx(
        int(round((12.0 + 1.0) / 20.0 * 1000)), abs=1)
    assert "12 um" in view._sliders["z"].property("readout").text()


def test_a_crosshair_asked_for_off_the_end_lands_on_the_last_slice(
        qtbot, qt_theme_applied):
    view = _view(qtbot, width=128)
    view.move_to(z=1e6)
    assert view.slice_index("z") == 9
    view.move_to(z=-1e6)
    assert view.slice_index("z") == 0


def test_an_uncalibrated_stack_reads_as_slice_numbers(qtbot, qt_theme_applied):
    view = _view(qtbot, _volume(spacing=Spacing.isotropic(3, 1.0)), width=64)
    view.move_to(z=3.0)
    readout = view._sliders["z"].property("readout").text()
    assert readout.startswith("3 px") and readout.endswith("slice 3")


def test_a_time_slider_appears_only_when_there_are_frames(qtbot,
                                                          qt_theme_applied):
    plain = _view(qtbot, width=96)
    assert not hasattr(plain, "frame_slider")

    view = _view(qtbot, width=96, frames=5)
    frames = []
    view.frame_changed.connect(frames.append)
    view.frame_slider.setValue(3)
    assert frames == [3]
    assert view.frame_slider.maximum() == 4


# ---------------------------------------------------------------------------
# Clicking
# ---------------------------------------------------------------------------

def test_clicking_the_side_panel_moves_the_top_panel_s_slice(qtbot,
                                                             qt_theme_applied):
    view = _view(qtbot, width=128)
    before = view.views.point["z"]
    panel = view.panels["zx"]
    panel.resize(200, 80)
    QTest.mouseClick(panel, Qt.LeftButton, Qt.NoModifier, QPoint(60, 10))

    assert view.views.point["z"] != before
    assert view.panels["xy"].canvas.depth["z"] == view.views.point["z"]
    assert view.views.point["z"] % 2.0 == pytest.approx(0.0), (
        "the click landed between two slices")


def test_clicking_an_object_publishes_it_to_every_other_view(qtbot,
                                                             qt_theme_applied):
    link = LinkedSelection()
    view = _view(qtbot, _volume(with_mask=True), width=128)
    view.link_selection(ov.ORTHO_LINK_SOURCE, link=link)
    picked = []
    view.object_picked.connect(picked.append)

    # The object is at slice 5 (z = 10 µm), rows/cols 30-33.
    view.move_to(z=10.0)
    canvas = view.views.xy
    row, column = canvas.pixel_at({"y": 31 * 0.65, "x": 31 * 0.65})
    view._on_panel_clicked("xy", row, column)

    assert picked == ["plate1_A_1_1_17"]
    assert list(link.selection.keys) == ["plate1_A_1_1_17"]
    assert view.stack["mask"].selected_label == 17


def test_clicking_the_background_publishes_nothing(qtbot, qt_theme_applied):
    link = LinkedSelection()
    view = _view(qtbot, _volume(with_mask=True), width=128)
    view.link_selection(ov.ORTHO_LINK_SOURCE, link=link)
    picked = []
    view.object_picked.connect(picked.append)
    view._on_panel_clicked("xy", 2.0, 2.0)
    assert picked == []
    assert link.selection.keys is None


def test_a_selection_elsewhere_moves_the_crosshair_onto_the_object(
        qtbot, qt_theme_applied):
    """The other half: the UMAP selects a cell, this view goes to its slice."""
    link = LinkedSelection()
    view = _view(qtbot, _volume(with_mask=True), width=128)
    view.link_selection(ov.ORTHO_LINK_SOURCE, link=link)
    view.move_to(z=0.0)

    link.set_selection(Selection(keys=["plate1_A_1_1_17"], source="umap"))

    assert view.slice_index("z") == 5, "the crosshair did not move to the object"
    assert view.stack["mask"].selected_label == 17
    # A selection of many objects is not a place to move to.
    view.move_to(z=0.0)
    link.set_selection(Selection(keys=["plate1_A_1_1_17", "plate1_A_1_1_18"],
                                 source="umap"))
    assert view.slice_index("z") == 0


def test_a_selection_this_view_does_not_hold_moves_nothing(qtbot,
                                                           qt_theme_applied):
    link = LinkedSelection()
    view = _view(qtbot, _volume(with_mask=True), width=128)
    view.link_selection(ov.ORTHO_LINK_SOURCE, link=link)
    view.move_to(z=0.0)
    link.set_selection(Selection(keys=["plate9_Z_9_9_99"], source="umap"))
    assert view.slice_index("z") == 0


# ---------------------------------------------------------------------------
# Zoom and teardown
# ---------------------------------------------------------------------------

def test_zooming_moves_every_panel_together(qtbot, qt_theme_applied):
    view = _view(qtbot, width=128)
    before = view.views.scale
    view.zoom_in()
    assert view.views.scale < before
    assert len({p.canvas.step for p in view.panels.values()}) == 1
    view.zoom_out()
    assert view.views.scale == pytest.approx(before)


def test_the_wheel_zooms_and_fit_puts_it_back(qtbot, qt_theme_applied):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QWheelEvent

    view = _view(qtbot, width=128)
    before = view.views.scale
    view.wheelEvent(QWheelEvent(
        QPointF(10.0, 10.0), view.mapToGlobal(QPoint(10, 10)),
        QPoint(0, 120), QPoint(0, 120), Qt.NoButton, Qt.NoModifier,
        Qt.NoScrollPhase, False))
    assert view.views.scale < before
    view.reset_view()
    assert view.views.scale == pytest.approx(before)


def test_showing_another_volume_rebuilds_the_sliders(qtbot, qt_theme_applied):
    view = _view(qtbot, width=128)
    assert view.views.n_slices("z") == 10
    view.set_stack(_volume(shape=(4, 32, 32)))
    assert view.views.n_slices("z") == 4
    assert sorted(view._sliders) == ["x", "y", "z"]


def test_the_view_leaves_the_shared_selection_when_it_closes(qtbot,
                                                            qt_theme_applied):
    link = LinkedSelection()
    view = _view(qtbot, width=96)
    view.link_selection(ov.ORTHO_LINK_SOURCE, link=link)
    assert view.is_linked
    view.close()
    assert not view.is_linked

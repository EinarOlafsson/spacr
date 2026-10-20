"""The True 3D view's mouse, wheel and magnifier cube, and its quiet refusals (N673).

The mouse on the 3-D canvas goes through the stage's event filter: the wheel
sizes and zooms the magnifier cube, the middle button (or Shift+left) pans,
Pick chooses the object under the pointer, Paint and Erase stroke the slice
plane, and a pointer moving with no button held carries the cube. Every
path is driven here with real Qt events on the real canvas. The view also
declines, without an error, whatever it cannot draw yet: no axes, no layers,
a camera it cannot invert, a box smaller than two pixels.

No GPU and no network: the small synthetic stack of test_673_napari_true3d.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent

from spacr.layers import LayerEvent

from .test_673_napari_true3d import _screen_of, screen, view

__all__ = ["screen", "view"]


def _canvas(stage):
    return stage._panel.canvas._canvas


def _widget_pos(stage, x, y):
    """Matplotlib display pixels (y up) as the canvas widget's position (y down)."""
    canvas = _canvas(stage)
    ratio = canvas.device_pixel_ratio
    return QPointF(x / ratio, canvas.figure.bbox.height / ratio - y / ratio)


def _mouse(stage, kind, x, y, button=Qt.LeftButton, buttons=None, modifiers=Qt.NoModifier):
    pos = _widget_pos(stage, x, y)
    held = button if buttons is None else buttons
    event = QMouseEvent(kind, pos, pos, button, held, modifiers)
    return stage.eventFilter(_canvas(stage), event)


def _wheel(stage, notches, modifiers=Qt.NoModifier):
    event = QWheelEvent(QPointF(20, 20), QPointF(20, 20), QPoint(0, 0),
                        QPoint(0, 120 * notches), Qt.NoButton, modifiers,
                        Qt.NoScrollPhase, False)
    return stage.eventFilter(_canvas(stage), event)


def _top_view(stage):
    canvas = stage._panel.canvas
    canvas._view_angles = (89.0, -90.0, 0.0)
    canvas.render_now()
    canvas._canvas.draw()


def _edge_on(stage):
    canvas = stage._panel.canvas
    canvas._view_angles = (0.0, -90.0, 0.0)
    canvas.render_now()
    canvas._canvas.draw()


def _centre(stage):
    return np.asarray([sum(pair) / 2.0 for pair in stage.session.limits], float)


# ---------------------------------------------------------------------------
# The wheel and the magnifier cube
# ---------------------------------------------------------------------------

def test_the_wheel_sizes_and_zooms_the_cube_only_with_a_modifier(view):
    stage = view
    assert not _wheel(stage, 1, Qt.ShiftModifier)
    assert stage.set_magnifier(True)
    side, zoom = stage._cube.side, stage._cube.zoom
    assert not _wheel(stage, 1)
    assert stage._cube.side == side and stage._cube.zoom == zoom
    assert _wheel(stage, 1, Qt.ShiftModifier)
    assert stage._cube.side > side and stage._cube.zoom == zoom
    assert _wheel(stage, -1, Qt.ControlModifier)
    assert stage._cube.zoom < zoom


def test_turning_the_cube_on_twice_keeps_one_checked_button(view):
    stage = view
    assert stage.set_magnifier(True)
    assert stage._cube_button.isChecked()
    assert stage.set_magnifier(True)
    assert stage._cube_button.isChecked() and stage._cube.on
    cubes = [a for a in stage._view_axes().collections if a.get_gid() == "VolumeMagnifierCube"]
    assert len(cubes) == 1 and cubes[0].get_visible()


def test_the_cube_follows_a_pointer_with_no_button_and_stays_inside(view):
    stage = view
    _top_view(stage)
    stage.set_magnifier(True)
    sx, sy, sz = stage.session.xyz_spacing
    target = (4.5 * sx, 5.5 * sy, (stage._cursor[2] + 0.5) * sz)
    x, y = _screen_of(stage, target)[:2]
    assert not _mouse(stage, QEvent.MouseMove, x, y, button=Qt.NoButton, buttons=Qt.NoButton)
    np.testing.assert_allclose(stage._cube.centre[:2], target[:2], atol=0.6 * max(sx, sy))
    kept = np.array(stage._cube.centre, copy=True)
    far = _screen_of(stage, (stage.session.limits[0][1] * 3, stage.session.limits[1][1] * 3,
                             target[2]))
    assert stage._cube_point(far[0], far[1]) is None
    assert not stage._move_cube(far[0], far[1])
    np.testing.assert_array_equal(stage._cube.centre, kept)


def test_an_edge_on_slice_puts_the_cube_halfway_along_the_line_of_sight(view):
    stage = view
    _edge_on(stage)
    stage.set_magnifier(True)
    centre = _centre(stage)
    x, y = _screen_of(stage, centre)[:2]
    point = stage._cube_point(x, y)
    assert point is not None
    np.testing.assert_allclose(point, centre, atol=0.6 * max(stage.session.xyz_spacing))
    (x0, x1), (y0, y1), (z0, z1) = stage.session.limits
    above = _screen_of(stage, (centre[0], centre[1], z1 + 4 * (z1 - z0)))
    assert stage._cube_point(above[0], above[1]) is None
    beside = _screen_of(stage, (x1 + 4 * (x1 - x0), centre[1], centre[2]))
    assert stage._cube_point(beside[0], beside[1]) is None


def test_the_cube_has_nowhere_to_go_without_a_camera(view, monkeypatch):
    stage = view
    stage.set_magnifier(True)
    monkeypatch.setattr(stage, "_projection", lambda ax: None)
    assert stage._cube_point(10.0, 10.0) is None
    assert not stage._move_cube(10.0, 10.0)
    assert stage._plane_point(10.0, 10.0) is None


# ---------------------------------------------------------------------------
# Pan, Pick, Paint through the event filter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("button, modifiers", [
    (Qt.MiddleButton, Qt.NoModifier), (Qt.LeftButton, Qt.ShiftModifier)])
def test_a_middle_or_shift_drag_pans_the_view(view, button, modifiers):
    stage = view
    start = stage._view_axes().get_position(original=True).x0
    assert _mouse(stage, QEvent.MouseButtonPress, 200, 200, button, modifiers=modifiers)
    assert stage._pan_from == pytest.approx((200.0, 200.0))
    assert _mouse(stage, QEvent.MouseMove, 260, 200, button, modifiers=modifiers)
    assert stage._view_axes().get_position(original=True).x0 > start
    assert _mouse(stage, QEvent.MouseButtonRelease, 260, 200, button, buttons=Qt.NoButton)
    assert stage._pan_from is None
    assert not _mouse(stage, QEvent.MouseButtonRelease, 260, 200, button, buttons=Qt.NoButton)


def test_a_plain_click_without_a_3d_tool_is_left_to_the_view(view):
    stage = view
    assert stage._tool3d is None
    assert not _mouse(stage, QEvent.MouseButtonPress, 200, 200)
    stage._tool_buttons["pick"].click()
    assert not _mouse(stage, QEvent.MouseButtonPress, 200, 200, Qt.RightButton)
    assert not _mouse(stage, QEvent.MouseMove, 200, 200, Qt.NoButton, buttons=Qt.NoButton)
    assert not stage._view_event(QEvent(QEvent.Enter))


def test_a_pick_click_chooses_the_object_under_the_pointer(view, screen):
    stage = view
    stage._tool_buttons["pick"].click()
    state = stage._render_state
    x0, x1, y0, y1 = state.rect
    rows, columns = state.picked.shape
    row, column = np.argwhere(state.picked == 2)[0]
    x = x0 + (column + .5) * (x1 - x0) / columns
    y = y1 - (row + .5) * (y1 - y0) / rows
    assert _mouse(stage, QEvent.MouseButtonPress, x, y)
    assert stage._selected == 2 and stage._labels.currentData() == 2


def test_a_paint_drag_is_one_stroke_written_on_release(view, screen, qtbot):
    stage = view
    _top_view(stage)
    stage.set_slice(2)
    original = stage.session.mask.copy()
    stage._tool_buttons["paint"].click()
    stage.canvas.brush_radius = 1
    sx, sy, sz = stage.session.xyz_spacing
    points = [_screen_of(stage, ((column + .5) * sx, 4.5 * sy, 2.5 * sz)) for column in (3, 6)]
    assert _mouse(stage, QEvent.MouseButtonPress, *points[0][:2])
    assert _mouse(stage, QEvent.MouseMove, *points[1][:2])
    assert np.array_equal(stage.session.mask, original)
    assert _mouse(stage, QEvent.MouseButtonRelease, *points[1][:2], buttons=Qt.NoButton)
    changed = np.argwhere(stage.session.mask != original)
    assert changed.size and set(changed[:, 0]) == {2}
    assert stage.session.mask[2, 4, 4] == stage._selected
    assert stage.step_history(False) is not None
    assert np.array_equal(stage.session.mask, original)


def test_a_stroke_off_the_section_paints_nothing(view, screen):
    stage = view
    _top_view(stage)
    stage._tool_buttons["paint"].click()
    (x0, x1), (y0, y1), _z = stage.session.limits
    off = _screen_of(stage, (x1 + 3 * (x1 - x0), y1 + 3 * (y1 - y0),
                             (stage._cursor[2] + 0.5) * stage.session.xyz_spacing[2]))
    original = stage.session.mask.copy()
    assert stage._plane_point(off[0], off[1]) is None
    stage._stroke_to(off[0], off[1])
    assert stage._stroke is not None and stage._stroke.last is None
    assert not stage._stroke.pixels.any()
    assert stage._finish_stroke() == 0
    assert stage._stroke is None and np.array_equal(stage.session.mask, original)


def test_an_eraser_stroke_over_empty_voxels_changes_nothing_but_says_so(view, screen):
    stage = view
    _top_view(stage)
    stage.set_slice(0)
    stage._tool_buttons["erase"].click()
    stage.canvas.brush_radius = 1
    sx, sy, sz = stage.session.xyz_spacing
    point = _screen_of(stage, (25.5 * sx, 1.5 * sy, 0.5 * sz))
    assert stage.session.mask[0, 1, 25] == 0
    stage._stroke_to(point[0], point[1])
    before = stage.session.mask.copy()
    assert stage._finish_stroke() == 0
    assert np.array_equal(stage.session.mask, before)
    assert "Updated 0 voxels" in screen._status_label.text()


def test_a_stroke_the_volume_refuses_says_why(view, screen, monkeypatch):
    stage = view
    stage._stroke = SimpleNamespace(pixels=np.ones((2, 2), bool), last=None, path=[])

    def refuse(*_args, **_kwargs):
        raise ValueError("that slice is locked")

    monkeypatch.setattr(stage.session, "paint_plane", refuse)
    assert stage._finish_stroke() == 0
    assert screen._status_label.text() == "that slice is locked"


def test_finishing_a_stroke_without_a_3d_view_draws_nothing(view, monkeypatch):
    stage = view
    stage._stroke = None
    panel, stage._panel = stage._panel, None
    try:
        assert stage._finish_stroke() == 0
    finally:
        stage._panel = panel


# ---------------------------------------------------------------------------
# Pick without a frame, outside it, or for an object the list lacks
# ---------------------------------------------------------------------------

def test_pick_says_no_object_without_a_frame_or_outside_it(view, screen):
    stage = view
    state = stage._render_state
    x0, x1, y0, y1 = state.rect
    assert stage.pick_at(x1 + 50, y1 + 50) == 0
    assert "No object" in screen._status_label.text()
    stage._render_state = None
    screen._status_label.setText("")
    assert stage.pick_at((x0 + x1) / 2, (y0 + y1) / 2) == 0
    assert "No object" in screen._status_label.text()


def test_pick_returns_an_object_the_list_does_not_hold_without_moving_it(view):
    stage = view
    state = stage._render_state
    x0, x1, y0, y1 = state.rect
    picked = np.full_like(state.picked, 77)
    stage._render_state = SimpleNamespace(**{**vars(state), "picked": picked})
    current = stage._labels.currentIndex()
    assert stage.pick_at((x0 + x1) / 2, (y0 + y1) / 2) == 77
    assert stage._labels.currentIndex() == current


# ---------------------------------------------------------------------------
# What the view declines to draw
# ---------------------------------------------------------------------------

def test_a_camera_that_cannot_be_inverted_gives_no_projection(view):
    stage = view
    ax = stage._view_axes()
    assert stage._projection(SimpleNamespace(M=None)) is None
    flat = SimpleNamespace(M=np.zeros((4, 4)), transData=ax.transData)
    with np.errstate(all="ignore"):
        assert stage._projection(flat) is None


def test_no_render_is_asked_for_what_cannot_be_drawn(view, monkeypatch):
    stage = view
    origin, matrix, screen0 = stage._projection(stage._view_axes())
    monkeypatch.setattr(stage, "_projection", lambda ax: (origin, matrix * 1e-3, screen0))
    assert stage._render_request("preview") is None
    assert stage._submit_render("preview") is False
    monkeypatch.setattr(stage, "_projection", lambda ax: None)
    assert stage._render_request("preview") is None
    monkeypatch.undo()
    layers, stage._layers = stage._layers, None
    try:
        assert stage._render_request("preview") is None
    finally:
        stage._layers = layers
    assert stage._submit_render("preview") is True


def test_the_slices_view_neither_renders_nor_redraws_for_the_grid(view, monkeypatch):
    stage = view
    redraws = []
    monkeypatch.setattr(stage._panel.canvas, "render_now", lambda: redraws.append(1))
    assert stage.set_view("default")
    assert stage._submit_render("preview") is False
    stage._grid_toggled(True)
    assert stage._show_grid and redraws == []


def test_a_failed_render_is_said_on_the_status_line(view, screen):
    stage = view
    request = SimpleNamespace(session=stage.session)
    stage._take_render((request, None, RuntimeError("out of memory")))
    assert screen._status_label.text() == "out of memory"


def test_without_its_3d_panel_the_stage_has_no_axes_to_dress(view, monkeypatch):
    stage = view
    panel, stage._panel = stage._panel, None
    try:
        assert stage._view_axes() is None
        assert stage._submit_render("preview") is False
        stage._dress_view()
        stage._place_slice_plane()
        stage._place_cube()
    finally:
        stage._panel = panel
    monkeypatch.setattr(stage, "_view_axes", lambda: None)
    pan = stage._pan
    stage._pan_by(40, 0)
    assert stage._pan != pan
    stage._dress_view()
    stage._place_cube()
    stage._stroke = None
    stage._stroke_to(1.0, 1.0)
    assert stage._stroke.last is None


def test_a_box_with_no_volume_is_left_alone(view):
    stage = view
    ax = stage._view_axes()
    limits = ax.get_xlim3d()
    session, stage.session = stage.session, None
    try:
        stage._fit_box(ax)
        stage._place_slice_plane()
    finally:
        stage.session = session
    assert ax.get_xlim3d() == limits


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

def test_an_empty_contrast_range_and_a_non_picture_event_change_nothing(view, monkeypatch):
    stage = view
    name = next(iter(stage._layers)).name
    before = stage._layers[name].contrast_limits()

    class _Spin:
        def __init__(self, value):
            self._value = value

        def value(self):
            return self._value

    stage._layer_contrast(name, _Spin(50.0), _Spin(50.0), 0)
    assert stage._layers[name].contrast_limits() == before
    changed = []
    monkeypatch.setattr(stage, "_view_changed", lambda: changed.append(1))
    stage._on_layer_event(LayerEvent("selected"))
    stage._on_layer_event(LayerEvent("renamed"))
    assert changed == []
    stage._on_layer_event(LayerEvent("changed"))
    assert changed == [1]


def test_a_flat_channel_gets_a_one_unit_contrast_range(view):
    stage = view
    flat = np.full((3, 4, 5), 7, np.uint16)
    session = SimpleNamespace(channels=[flat], mask=np.zeros(flat.shape, np.uint16))
    stack = stage._volume_layers(session)
    image = list(stack)[0]
    assert tuple(map(float, image.contrast_limits())) == (7.0, 8.0)
    assert image.metadata["range"] == (7.0, 8.0)


def test_the_layer_list_is_empty_without_layers(view):
    stage = view
    layers, stage._layers = stage._layers, None
    try:
        stage._fill_layer_list()
        assert stage._layer_rows == {}
        assert stage._layer_list.layout().count() == 0
    finally:
        stage._layers = layers
        stage._fill_layer_list()
    assert stage._layer_rows

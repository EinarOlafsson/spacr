"""``B14`` — the half of the ROI that needs a mouse.

:mod:`tests.test_roi` covers the geometry and the delivery route with no Qt at
all. What is left for this file is the part a widget is required for: that a
click becomes a vertex at the world coordinate under the cursor (not at the
widget pixel, which is a different number at every zoom), that the half-drawn
outline is a real shape in the model rather than a rubber band drawn beside
it, and that the panel's one button is the same ``enable_roi_filter`` call the
non-Qt tests exercise.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from spacr import measure_hooks as mh
from spacr import roi as R
from spacr.layers import LayerStack, Spacing
from spacr.qt import layer_viewer as lv
from spacr.qt import roi_tool as rt


@pytest.fixture(autouse=True)
def clean_hooks():
    """No registration and no environment variable outlives a test here."""
    import os

    saved = {name: os.environ.get(name)
             for name in (mh.HOOKS_ENV_VAR, R.ROI_ENV_VAR,
                          R.ON_MISSING_ENV_VAR)}
    mh.clear_measurement_hooks()
    yield
    mh.clear_measurement_hooks()
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _stack(size=64, step=1.0, units="px"):
    stack = LayerStack()
    stack.add_image(np.arange(size * size, dtype=np.uint16).reshape(size, size),
                    name="image",
                    spacing=Spacing.isotropic(2, step, units=units))
    return stack


def _canvas(qtbot, stack, width=200, height=200):
    canvas = lv.LayerCanvas(stack)
    qtbot.addWidget(canvas)
    canvas.resize(width, height)
    canvas._ensure_canvas()
    return canvas


def _panel(qtbot, canvas, tmp_path):
    panel = rt.RoiPanel(canvas, roi_path=str(tmp_path / "roi.json"))
    qtbot.addWidget(panel)
    return panel


# ---------------------------------------------------------------------------
# The pen
# ---------------------------------------------------------------------------

def test_a_click_places_a_vertex_at_the_world_point_under_the_cursor(
        qtbot, qt_theme_applied, tmp_path):
    canvas = _canvas(qtbot, _stack())
    panel = _panel(qtbot, canvas, tmp_path)
    pen = panel.start_drawing()

    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(40, 30))

    expected = canvas.canvas.world_at(30 - 1, 40 - 1)
    placed = pen.pending[0]
    assert placed[0] == pytest.approx(expected["y"])
    assert placed[1] == pytest.approx(expected["x"])


def test_the_same_click_at_a_different_zoom_is_the_same_world_point(
        qtbot, qt_theme_applied, tmp_path):
    """The whole reason vertices are not widget pixels."""
    canvas = _canvas(qtbot, _stack())
    panel = _panel(qtbot, canvas, tmp_path)
    pen = panel.start_drawing()

    world = canvas.canvas.world_at(29, 39)
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(40, 30))
    canvas._canvas = canvas.canvas.zoomed(4.0)
    row, column = canvas.canvas.pixel_at(world)
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier,
                     QPoint(int(round(column)) + 1, int(round(row)) + 1))

    np.testing.assert_allclose(pen.pending[0], pen.pending[1], atol=0.51)


def test_the_half_drawn_outline_is_a_shape_in_the_model(qtbot,
                                                        qt_theme_applied,
                                                        tmp_path):
    canvas = _canvas(qtbot, _stack())
    panel = _panel(qtbot, canvas, tmp_path)
    pen = panel.start_drawing()

    pen.add_world({"y": 4.0, "x": 4.0})
    assert len(pen.layer) == 0, "one point is not an outline yet"
    pen.add_world({"y": 4.0, "x": 40.0})
    preview, = pen.layer.shapes
    assert preview.kind == "path" and not preview.is_closed
    pen.add_world({"y": 40.0, "x": 40.0})
    assert len(pen.layer) == 1, "the outline was added again instead of replaced"

    index = pen.close_shape()
    assert index == 0
    closed, = pen.layer.shapes
    assert closed.kind == "polygon" and closed.is_closed
    assert pen.pending.shape == (0, 2)


def test_a_double_click_closes_the_polygon(qtbot, qt_theme_applied, tmp_path):
    canvas = _canvas(qtbot, _stack())
    panel = _panel(qtbot, canvas, tmp_path)
    pen = panel.start_drawing()
    finished = []
    pen.roi_finished.connect(finished.append)

    for point in (QPoint(20, 20), QPoint(20, 80), QPoint(80, 80)):
        QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, point)
    QTest.mouseDClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(80, 20))

    assert finished == [0]
    shape, = pen.layer.shapes
    assert shape.kind == "polygon"
    # The double click is not also a vertex: three clicks, three vertices.
    assert len(shape.data) == 3


def test_a_rectangle_closes_itself_on_the_second_corner(qtbot,
                                                        qt_theme_applied,
                                                        tmp_path):
    canvas = _canvas(qtbot, _stack())
    panel = _panel(qtbot, canvas, tmp_path)
    panel.kind_combo.setCurrentText("rectangle")
    pen = panel.pen or panel.start_drawing()

    pen.add_world({"y": 4.0, "x": 4.0})
    assert len(pen.layer) == 0
    pen.add_world({"y": 30.0, "x": 30.0})

    shape, = pen.layer.shapes
    assert shape.kind == "rectangle"
    assert pen.pending.shape == (0, 2)


def test_backspace_undoes_and_escape_abandons(qtbot, qt_theme_applied,
                                              tmp_path):
    canvas = _canvas(qtbot, _stack())
    panel = _panel(qtbot, canvas, tmp_path)
    pen = panel.start_drawing()
    for point in (QPoint(20, 20), QPoint(20, 80), QPoint(80, 80)):
        QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, point)

    QTest.keyClick(canvas, Qt.Key_Backspace)
    assert len(pen.pending) == 2
    QTest.keyClick(canvas, Qt.Key_Escape)
    assert len(pen.pending) == 0
    assert len(pen.layer) == 0, "the abandoned outline stayed in the model"


def test_a_right_click_takes_the_last_vertex_back(qtbot, qt_theme_applied,
                                                  tmp_path):
    canvas = _canvas(qtbot, _stack())
    panel = _panel(qtbot, canvas, tmp_path)
    pen = panel.start_drawing()
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(20, 20))
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(20, 80))
    QTest.mouseClick(canvas, Qt.RightButton, Qt.NoModifier, QPoint(80, 80))
    assert len(pen.pending) == 1


def test_return_closes_and_a_stray_close_does_nothing(qtbot, qt_theme_applied,
                                                      tmp_path):
    canvas = _canvas(qtbot, _stack())
    panel = _panel(qtbot, canvas, tmp_path)
    pen = panel.start_drawing()
    assert pen.close_shape() == -1, "an empty canvas closed a shape"
    for point in ({"y": 4.0, "x": 4.0}, {"y": 4.0, "x": 40.0},
                  {"y": 40.0, "x": 40.0}):
        pen.add_world(point)
    QTest.keyClick(canvas, Qt.Key_Return)
    assert len(pen.layer) == 1


def test_the_pen_takes_the_mouse_and_gives_it_back(qtbot, qt_theme_applied,
                                                   tmp_path):
    """A tool consumes the click; without one the canvas still picks."""
    canvas = _canvas(qtbot, _stack())
    panel = _panel(qtbot, canvas, tmp_path)
    picked = []
    canvas.picked.connect(lambda *args: picked.append(args))

    panel.draw_button.setChecked(True)
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(40, 40))
    assert picked == [], "the pen was attached and the canvas still picked"

    panel.draw_button.setChecked(False)
    assert canvas.tool is None
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(40, 40))
    assert len(picked) == 1


def test_shift_dragging_still_pans_while_the_pen_is_attached(
        qtbot, qt_theme_applied, tmp_path):
    canvas = _canvas(qtbot, _stack())
    panel = _panel(qtbot, canvas, tmp_path)
    pen = panel.start_drawing()

    QTest.mousePress(canvas, Qt.LeftButton, Qt.ShiftModifier, QPoint(50, 50))
    assert canvas._drag is not None, "shift-drag stopped panning"
    QTest.mouseRelease(canvas, Qt.LeftButton, Qt.ShiftModifier, QPoint(60, 60))

    assert len(pen.pending) == 0, "a pan placed a vertex"


def test_a_pen_needs_a_shapes_layer_and_a_closed_kind(qtbot, qt_theme_applied):
    from spacr.layers import LayerError, ShapesLayer

    with pytest.raises(LayerError, match="ShapesLayer"):
        rt.RoiPen(object())
    with pytest.raises(LayerError, match="closed shape"):
        rt.RoiPen(ShapesLayer(name="roi", ndim=2), kind="path")


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

def test_the_roi_layer_inherits_the_stack_s_spacing(qtbot, qt_theme_applied,
                                                    tmp_path):
    """An ROI drawn over a µm image is stored in µm, not in pixels."""
    stack = _stack(step=0.65, units="um")
    panel = _panel(qtbot, _canvas(qtbot, stack), tmp_path)
    layer = panel.roi_layer()
    assert layer.spacing.units == "um"
    assert layer.spacing.scale == (0.65, 0.65)
    assert layer.name == rt.ROI_LAYER_NAME
    # Asked for twice, added once.
    assert panel.roi_layer() is layer


def test_an_empty_stack_still_gets_a_pixel_spaced_roi_layer(qtbot,
                                                            qt_theme_applied,
                                                            tmp_path):
    panel = _panel(qtbot, _canvas(qtbot, LayerStack()), tmp_path)
    assert panel.roi_layer().spacing.units == "px"


def test_the_panel_button_installs_the_filter_the_workers_will_read(
        qtbot, qt_theme_applied, tmp_path):
    canvas = _canvas(qtbot, _stack(size=128))
    panel = _panel(qtbot, canvas, tmp_path)
    states = []
    panel.filter_changed.connect(states.append)

    pen = panel.start_drawing()
    pen.add_world({"y": 0.0, "x": 0.0})
    pen.add_world({"y": 64.0, "x": 128.0})
    panel.kind_combo.setCurrentText("rectangle")  # rebuilds the pen
    panel.pen.add_world({"y": 0.0, "x": 0.0})
    panel.pen.add_world({"y": 64.0, "x": 128.0})

    assert panel.enable() is True
    assert states == [True]

    entry, = mh.region_filter_hooks()
    assert entry.name == R.HOOK_NAME
    assert entry.source == "env", "the filter would not reach a spawn worker"
    saved = R.RoiSet.load(panel.roi_path)
    assert len(saved) == 1 and saved.units == "px"
    ok, _message = R.worker_delivery_status("spawn")
    assert ok is True

    assert panel.disable() is True
    assert mh.region_filter_hooks() == ()
    assert states == [True, False]


def test_the_panel_carries_the_rule_the_user_chose(qtbot, qt_theme_applied,
                                                   tmp_path):
    canvas = _canvas(qtbot, _stack(size=128))
    panel = _panel(qtbot, canvas, tmp_path)
    panel.mode_combo.setCurrentText("overlap")
    assert panel.overlap_spin.isEnabled()
    panel.overlap_spin.setValue(0.8)
    panel.invert_check.setChecked(True)
    panel.field_edit.setText("plate1_A01_F001, plate1_A01_F002")

    layer = panel.roi_layer()
    layer.add_rectangle([0.0, 0.0], [64.0, 128.0])
    roi_set = panel.roi_set()

    assert roi_set.mode == "overlap"
    assert roi_set.min_overlap == pytest.approx(0.8)
    assert roi_set.invert is True
    assert sorted(roi_set.fields) == ["plate1_A01_F001", "plate1_A01_F002"]


def test_enabling_with_nothing_drawn_says_so_instead_of_raising(
        qtbot, qt_theme_applied, tmp_path):
    panel = _panel(qtbot, _canvas(qtbot, _stack()), tmp_path)
    assert panel.enable() is False
    assert "no ROI has been drawn" in panel.status.text()
    assert panel.status.objectName() == "RoiStatusWarning"
    assert mh.region_filter_hooks() == ()


def test_clearing_removes_every_drawn_roi(qtbot, qt_theme_applied, tmp_path):
    panel = _panel(qtbot, _canvas(qtbot, _stack()), tmp_path)
    layer = panel.roi_layer()
    layer.add_rectangle([0.0, 0.0], [10.0, 10.0])
    layer.add_rectangle([20.0, 20.0], [30.0, 30.0])
    assert panel.clear_rois() == 2
    assert len(layer) == 0
    assert panel.clear_rois() == 0


def test_the_status_line_says_whole_fields_until_the_filter_is_on(
        qtbot, qt_theme_applied, tmp_path):
    panel = _panel(qtbot, _canvas(qtbot, _stack(size=128)), tmp_path)
    assert "whole fields are measured" in panel.status.text()

    panel.roi_layer().add_rectangle([0.0, 0.0], [64.0, 128.0])
    panel.enable()
    assert "install it themselves" in panel.status.text()
    assert panel.status.objectName() == "RoiStatus"


def test_the_status_warns_when_the_filter_would_reach_no_worker(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    panel = _panel(qtbot, _canvas(qtbot, _stack(size=128)), tmp_path)
    panel.roi_layer().add_rectangle([0.0, 0.0], [64.0, 128.0])
    panel.enable()
    monkeypatch.setattr(rt, "worker_delivery_status",
                        lambda *a, **k: (False, "it would reach nothing"))
    panel._refresh_status()
    assert panel.status.objectName() == "RoiStatusWarning"


def test_the_panel_lets_go_of_the_canvas_when_it_closes(qtbot,
                                                        qt_theme_applied,
                                                        tmp_path):
    canvas = _canvas(qtbot, _stack())
    panel = _panel(qtbot, canvas, tmp_path)
    panel.start_drawing()
    panel.close()
    assert canvas.tool is None
    assert canvas.focusPolicy() == Qt.NoFocus


def test_setting_the_file_is_where_the_workers_read_it(qtbot, qt_theme_applied,
                                                        tmp_path):
    panel = _panel(qtbot, _canvas(qtbot, _stack(size=128)), tmp_path)
    target = panel.set_roi_path(str(tmp_path / "elsewhere" / "roi.json"))
    panel.roi_layer().add_rectangle([0.0, 0.0], [64.0, 128.0])
    panel.enable()
    import os

    assert os.path.isfile(target)
    assert os.environ[R.ROI_ENV_VAR] == target

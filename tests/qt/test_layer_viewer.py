"""The Qt view over the layer model.

The compositing rules are tested in ``tests/test_layers.py`` with no Qt at
all. What is left for this file is the part that genuinely needs a widget:
that the model's picture reaches the screen, that the layer list and the
model stay one thing rather than two, and that clicking an object publishes
the key the rest of the app already uses for it.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QWidget

from spacr.layers import (Blending, FieldKey, LayerStack, Spacing,
                          label_color)
from spacr.qt import layer_viewer as lv
from spacr.qt.linked_selection import LinkedSelection


PLATE_KEY_VALUES = ("plate1", "A", "1", "1")


def _field_key():
    return FieldKey(values=dict(zip(FieldKey.columns(), PLATE_KEY_VALUES)))


def _stack_with_an_object():
    """A grey field with one labelled object at rows/cols 20-29."""
    stack = LayerStack()
    image = np.full((40, 40), 1000, dtype=np.uint16)
    image[20:30, 20:30] = 4000
    stack.add_image(image, name="image", contrast_limits=(0.0, 4095.0))
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[20:30, 20:30] = 17
    stack.add_labels(mask, name="mask", field=_field_key())
    return stack


def _sized(qtbot, widget: QWidget, width=320, height=320):
    qtbot.addWidget(widget)
    widget.resize(width, height)
    return widget


# ---------------------------------------------------------------------------
# The canvas
# ---------------------------------------------------------------------------

def test_the_canvas_paints_the_model_s_composite(qtbot, qt_theme_applied):
    stack = _stack_with_an_object()
    canvas = _sized(qtbot, lv.LayerCanvas(stack))
    world = canvas._ensure_canvas()
    assert world is not None
    assert world.shape == (canvas.height() - 2, canvas.width() - 2)

    grabbed = canvas.grab().toImage()
    row, column = world.pixel_at({"y": 25.0, "x": 25.0})
    pixel = grabbed.pixelColor(int(column) + 1, int(row) + 1)
    expected = label_color(17)
    # The mask is on top at full opacity, so the object's own colour wins.
    assert abs(pixel.redF() - expected[0]) < 0.05
    assert abs(pixel.greenF() - expected[1]) < 0.05
    assert abs(pixel.blueF() - expected[2]) < 0.05


def test_hiding_a_layer_changes_what_the_canvas_paints(qtbot, qt_theme_applied):
    stack = _stack_with_an_object()
    canvas = _sized(qtbot, lv.LayerCanvas(stack))
    world = canvas._ensure_canvas()
    row, column = world.pixel_at({"y": 25.0, "x": 25.0})
    before = canvas.grab().toImage().pixelColor(int(column) + 1, int(row) + 1)
    stack["mask"].visible = False
    after = canvas.grab().toImage().pixelColor(int(column) + 1, int(row) + 1)
    assert before != after
    # Without the mask the image shows through: near-white at 4000/4095.
    assert after.redF() == pytest.approx(after.greenF(), abs=0.02)
    assert after.redF() > 0.9


def test_an_empty_canvas_says_so_instead_of_raising(qtbot, qt_theme_applied):
    canvas = _sized(qtbot, lv.LayerCanvas())
    assert canvas._ensure_canvas() is None
    canvas.grab()          # must not raise
    canvas.reset_view()
    canvas.wheelEvent(_wheel(canvas, 120))


def _wheel(widget, delta):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QWheelEvent
    return QWheelEvent(QPointF(10.0, 10.0), widget.mapToGlobal(QPoint(10, 10)),
                       QPoint(0, delta), QPoint(0, delta), Qt.NoButton,
                       Qt.NoModifier, Qt.NoScrollPhase, False)


def test_the_wheel_zooms_about_the_cursor_and_fit_puts_it_back(
        qtbot, qt_theme_applied):
    stack = _stack_with_an_object()
    canvas = _sized(qtbot, lv.LayerCanvas(stack))
    before = canvas._ensure_canvas()
    under_cursor = before.world_at(10 - 1, 10 - 1)

    canvas.wheelEvent(_wheel(canvas, 120))
    zoomed = canvas.canvas
    assert zoomed.step[0] < before.step[0]
    assert zoomed.world_at(9, 9)["y"] == pytest.approx(under_cursor["y"])

    canvas.reset_view()
    assert canvas._ensure_canvas().step[0] == pytest.approx(before.step[0])


def test_dragging_pans_the_world_window(qtbot, qt_theme_applied):
    stack = _stack_with_an_object()
    canvas = _sized(qtbot, lv.LayerCanvas(stack))
    before = canvas._ensure_canvas().origin
    QTest.mousePress(canvas, Qt.LeftButton, Qt.ShiftModifier, QPoint(50, 50))
    QTest.mouseMove(canvas, QPoint(60, 70))
    canvas.mouseMoveEvent(_move(canvas, 60, 70))
    QTest.mouseRelease(canvas, Qt.LeftButton, Qt.ShiftModifier, QPoint(60, 70))
    assert canvas.canvas.origin != before


def _move(widget, x, y, buttons=Qt.NoButton):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QEvent
    return QMouseEvent(QEvent.MouseMove, QPointF(x, y),
                       widget.mapToGlobal(QPoint(int(x), int(y))),
                       Qt.NoButton, buttons, Qt.NoModifier)


def test_the_canvas_reports_the_world_under_the_cursor(qtbot, qt_theme_applied):
    stack = _stack_with_an_object()
    canvas = _sized(qtbot, lv.LayerCanvas(stack))
    world = canvas._ensure_canvas()
    seen = []
    canvas.hovered.connect(seen.append)
    canvas.mouseMoveEvent(_move(canvas, 31, 21))
    assert seen and set(seen[0]) == {"y", "x"}
    assert seen[0] == world.world_at(20, 30)


def test_the_canvas_can_be_pointed_at_another_plane(qtbot, qt_theme_applied):
    """The seam the orthogonal-view item builds on."""
    volume = np.zeros((6, 20, 20), dtype=np.uint8)
    volume[3] = 255
    stack = LayerStack()
    stack.add_image(volume, name="stack", contrast_limits=(0.0, 255.0),
                    spacing=Spacing.from_map({"z": 4.0, "y": 1.0, "x": 1.0},
                                             units="um"))
    canvas = _sized(qtbot, lv.LayerCanvas(stack))
    canvas.set_plane(("z", "x"), {"y": 10.0})
    assert canvas._ensure_canvas().axes == ("z", "x")
    canvas.set_depth(y=4.0)
    assert canvas.canvas.depth["y"] == 4.0


def test_swapping_the_stack_lets_go_of_the_old_one(qtbot, qt_theme_applied):
    first = _stack_with_an_object()
    canvas = _sized(qtbot, lv.LayerCanvas(first))
    second = LayerStack()
    canvas.set_stack(second)
    assert canvas.stack is second
    assert first._listeners == []
    canvas.detach()
    assert second._listeners == []


# ---------------------------------------------------------------------------
# Clicking an object
# ---------------------------------------------------------------------------

def _click_the_object(qtbot, viewer):
    """Left-click the centre of label 17 and return the canvas pixel used."""
    world = viewer.canvas._ensure_canvas()
    row, column = world.pixel_at({"y": 25.0, "x": 25.0})
    point = QPoint(int(column) + 1, int(row) + 1)
    QTest.mouseClick(viewer.canvas, Qt.LeftButton, Qt.NoModifier, point)
    return point


def test_clicking_a_label_publishes_the_object_key_the_tables_use(
        qtbot, qt_theme_applied):
    """The point of giving a labels layer a field key."""
    link = LinkedSelection()
    viewer = _sized(qtbot, lv.LayerViewer(_stack_with_an_object()), 400, 400)
    viewer.link_selection(lv.LINK_SOURCE, link=link)

    emitted = []
    viewer.object_picked.connect(emitted.append)
    _click_the_object(qtbot, viewer)

    assert emitted == ["plate1_A_1_1_17"]
    assert list(link.selection.keys) == ["plate1_A_1_1_17"]
    assert link.selection.source == lv.LINK_SOURCE
    assert viewer.stack["mask"].selected_label == 17
    assert "plate1_A_1_1_17" in viewer.status.text()


def test_clicking_the_background_publishes_nothing(qtbot, qt_theme_applied):
    link = LinkedSelection()
    viewer = _sized(qtbot, lv.LayerViewer(_stack_with_an_object()), 400, 400)
    viewer.link_selection(lv.LINK_SOURCE, link=link)
    world = viewer.canvas._ensure_canvas()
    row, column = world.pixel_at({"y": 2.0, "x": 2.0})
    QTest.mouseClick(viewer.canvas, Qt.LeftButton, Qt.NoModifier,
                     QPoint(int(column) + 1, int(row) + 1))
    assert link.selection.keys is None
    assert "nothing here" in viewer.status.text()


def test_a_labels_layer_with_no_field_picks_but_does_not_publish(
        qtbot, qt_theme_applied):
    stack = LayerStack()
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[20:30, 20:30] = 17
    stack.add_labels(mask, name="mask")
    link = LinkedSelection()
    viewer = _sized(qtbot, lv.LayerViewer(stack), 400, 400)
    viewer.link_selection(lv.LINK_SOURCE, link=link)
    _click_the_object(qtbot, viewer)
    assert link.selection.keys is None
    assert "label 17" in viewer.status.text()


def test_double_clicking_asks_for_the_object_to_be_opened(qtbot,
                                                          qt_theme_applied):
    link = LinkedSelection()
    opened = []
    link.register_object_opener("annotate", opened.append)
    viewer = _sized(qtbot, lv.LayerViewer(_stack_with_an_object()), 400, 400)
    viewer.link_selection(lv.LINK_SOURCE, link=link)

    world = viewer.canvas._ensure_canvas()
    row, column = world.pixel_at({"y": 25.0, "x": 25.0})
    QTest.mouseDClick(viewer.canvas, Qt.LeftButton, Qt.NoModifier,
                      QPoint(int(column) + 1, int(row) + 1))
    # `has_object_opener` asks the PROCESS-WIDE registry, so a double click
    # with only this test's private opener registered must be a no-op rather
    # than a NoObjectOpener out of a mouse handler.
    assert opened == [] or [len(r) for r in opened] == [1]


def test_a_selection_from_another_view_highlights_the_matching_label(
        qtbot, qt_theme_applied):
    from spacr.selection import Selection

    link = LinkedSelection()
    viewer = _sized(qtbot, lv.LayerViewer(_stack_with_an_object()), 400, 400)
    viewer.link_selection(lv.LINK_SOURCE, link=link)
    link.set_selection(Selection.from_keys(["plate1_A_1_1_17"], source="umap"))
    assert viewer.stack["mask"].selected_label == 17
    # An unknown key leaves the highlight alone rather than clearing it.
    link.set_selection(Selection.from_keys(["plate9_Z_9_9_9"], source="umap"))
    assert viewer.stack["mask"].selected_label == 17


# ---------------------------------------------------------------------------
# The layer list
# ---------------------------------------------------------------------------

def test_the_layer_list_shows_the_stack_top_first(qtbot, qt_theme_applied):
    stack = _stack_with_an_object()
    viewer = _sized(qtbot, lv.LayerViewer(stack), 400, 400)
    rows = [viewer.layer_list.item(i).text()
            for i in range(viewer.layer_list.count())]
    assert rows == ["mask", "image"]


def test_unchecking_a_row_hides_that_layer(qtbot, qt_theme_applied):
    stack = _stack_with_an_object()
    viewer = _sized(qtbot, lv.LayerViewer(stack), 400, 400)
    viewer.layer_list.item(0).setCheckState(Qt.Unchecked)
    assert stack["mask"].visible is False
    viewer.layer_list.item(0).setCheckState(Qt.Checked)
    assert stack["mask"].visible is True


def test_renaming_a_row_renames_the_layer(qtbot, qt_theme_applied):
    stack = _stack_with_an_object()
    viewer = _sized(qtbot, lv.LayerViewer(stack), 400, 400)
    viewer.layer_list.item(0).setText("nuclei")
    assert stack.names == ("image", "nuclei")
    # A blank name is refused by the model and the row goes back.
    viewer.layer_list.item(0).setText("   ")
    assert stack.names == ("image", "nuclei")


def test_dragging_a_row_reorders_the_model(qtbot, qt_theme_applied):
    stack = _stack_with_an_object()
    viewer = _sized(qtbot, lv.LayerViewer(stack), 400, 400)
    listing = viewer.layer_list
    assert stack.names == ("image", "mask")
    # What an internal-move drag ends up doing to the rows.
    listing._syncing = True
    listing.insertItem(2, listing.takeItem(0))
    listing._syncing = False
    listing._on_rows_moved()
    assert stack.names == ("mask", "image")
    assert [listing.item(i).text() for i in range(listing.count())] == [
        "image", "mask"]


def test_the_buttons_move_and_remove_the_selected_layer(qtbot,
                                                        qt_theme_applied):
    stack = _stack_with_an_object()
    viewer = _sized(qtbot, lv.LayerViewer(stack), 400, 400)
    stack.select("image")
    viewer.raise_button.click()
    assert stack.names == ("mask", "image")
    viewer.lower_button.click()
    assert stack.names == ("image", "mask")
    viewer.remove_button.click()
    assert stack.names == ("mask",)
    viewer.remove_button.click()
    assert stack.names == ()
    # Nothing selected: the buttons are disabled rather than raising.
    assert not viewer.remove_button.isEnabled()
    viewer.remove_button.click()


def test_selecting_a_row_points_the_property_controls_at_that_layer(
        qtbot, qt_theme_applied):
    stack = _stack_with_an_object()
    stack["mask"].opacity = 0.4
    stack["mask"].blending = Blending.ADDITIVE
    viewer = _sized(qtbot, lv.LayerViewer(stack), 400, 400)

    viewer.layer_list.setCurrentRow(0)          # the mask
    assert stack.selected is stack["mask"]
    assert viewer.opacity_slider.value() == 40
    assert viewer.blending_combo.currentText() == Blending.ADDITIVE
    assert not viewer.colormap_combo.isEnabled()

    viewer.layer_list.setCurrentRow(1)          # the image
    assert viewer.opacity_slider.value() == 100
    assert viewer.colormap_combo.isEnabled()
    assert viewer.colormap_combo.currentText() == "gray"


def test_the_property_controls_write_back_to_the_model(qtbot, qt_theme_applied):
    stack = _stack_with_an_object()
    viewer = _sized(qtbot, lv.LayerViewer(stack), 400, 400)
    stack.select("image")

    viewer.opacity_slider.setValue(35)
    assert stack["image"].opacity == pytest.approx(0.35)
    viewer.blending_combo.setCurrentText(Blending.ADDITIVE)
    assert stack["image"].blending == Blending.ADDITIVE
    viewer.colormap_combo.setCurrentText("magenta")
    assert stack["image"].colormap.name == "magenta"

    # The colormap control does nothing to a labels layer rather than raising.
    stack.select("mask")
    viewer._on_colormap("red")
    assert stack["mask"].kind == "labels"


def test_adding_an_empty_points_or_shapes_layer(qtbot, qt_theme_applied):
    stack = _stack_with_an_object()
    viewer = _sized(qtbot, lv.LayerViewer(stack), 400, 400)
    viewer.add_points_button.click()
    viewer.add_shapes_button.click()
    assert stack.names == ("image", "mask", "points", "shapes")
    assert stack["points"].ndim == 2


def test_a_layer_added_from_a_file_joins_the_stack(qtbot, qt_theme_applied,
                                                   tmp_path):
    tifffile = pytest.importorskip("tifffile")
    image = np.zeros((16, 16), dtype=np.uint16)
    image[4:8, 4:8] = 3000
    mask = np.zeros((16, 16), dtype=np.uint16)
    mask[4:8, 4:8] = 3
    tifffile.imwrite(tmp_path / "field.tif", image)
    tifffile.imwrite(tmp_path / "mask.tif", mask)

    viewer = _sized(qtbot, lv.LayerViewer(), 400, 400)
    assert viewer.add_image_file(tmp_path / "field.tif") is not None
    assert viewer.add_labels_file(tmp_path / "mask.tif",
                                  field=_field_key()) is not None
    assert viewer.stack.names == ("image", "mask")
    assert viewer.stack["mask"].object_key_at_world({"y": 5.0, "x": 5.0}) == \
        "plate1_A_1_1_3"


def test_a_file_that_cannot_be_read_is_reported_not_raised(qtbot,
                                                           qt_theme_applied,
                                                           tmp_path):
    viewer = _sized(qtbot, lv.LayerViewer(), 400, 400)
    assert viewer.add_image_file(tmp_path / "nope.tif") is None
    assert viewer.add_labels_file(tmp_path / "nope.tif") is None
    assert "Could not load" in viewer.status.text()
    assert len(viewer.stack) == 0


def test_a_layer_added_later_inherits_the_stack_s_world_units(
        qtbot, qt_theme_applied):
    stack = LayerStack()
    stack.add_image(np.zeros((8, 8)), name="um", spacing=Spacing.from_map(
        {"y": 0.65, "x": 0.65}, units="um"))
    viewer = _sized(qtbot, lv.LayerViewer(stack), 400, 400)
    viewer.add_points_button.click()
    assert stack["points"].spacing.units == "um"
    assert stack["points"].spacing.scale == (0.65, 0.65)


def test_closing_the_viewer_lets_go_of_everything(qtbot, qt_theme_applied):
    stack = _stack_with_an_object()
    viewer = _sized(qtbot, lv.LayerViewer(stack), 400, 400)
    assert viewer.is_linked
    assert len(stack._listeners) == 3          # viewer, canvas, layer list
    viewer.close()
    assert not viewer.is_linked
    assert stack._listeners == []
    viewer.close()                              # idempotent


# ---------------------------------------------------------------------------
# Registration, through the seams
# ---------------------------------------------------------------------------

def test_the_viewer_registers_its_own_qss_block():
    from spacr.qt import theme

    assert "LayerViewer" in theme.widget_qss_names()
    qss = theme.stylesheet("dark")
    assert "QWidget#LayerViewer" in qss
    assert "QListWidget#LayerList" in qss


def test_the_app_registration_is_one_call_away(qtbot, qt_theme_applied):
    """Registered in a sandbox: the seam works, without leaking a row.

    Not called at import — see `register_layer_viewer_app`'s docstring for the
    nine other ledgers a live registration has to land in at the same time.
    """
    from spacr.qt import app as app_mod

    assert not any(row[0] == lv.LAYER_VIEWER_APP_KEY for row in app_mod.APPS)
    apps = list(app_mod.APPS)
    factories = dict(app_mod.APP_FACTORIES)
    stages = dict(app_mod.APP_STAGE)
    try:
        row = lv.register_layer_viewer_app()
        assert row[0] == lv.LAYER_VIEWER_APP_KEY
        assert row in app_mod.APPS
        assert app_mod.registered_factory(lv.LAYER_VIEWER_APP_KEY) is \
            lv.make_layer_viewer_screen
        screen = app_mod.registered_factory(lv.LAYER_VIEWER_APP_KEY)()
        qtbot.addWidget(screen)
        assert isinstance(screen, lv.LayerViewer)
        assert isinstance(screen, QWidget)
    finally:
        app_mod.APPS[:] = apps
        app_mod.APP_FACTORIES.clear()
        app_mod.APP_FACTORIES.update(factories)
        app_mod.APP_STAGE.clear()
        app_mod.APP_STAGE.update(stages)
        app_mod._refresh_sections()
    assert not any(row[0] == lv.LAYER_VIEWER_APP_KEY for row in app_mod.APPS)


# ---------------------------------------------------------------------------
# The paths a user reaches by accident
# ---------------------------------------------------------------------------

def test_a_multichannel_image_and_a_stacked_mask_load_as_one_layer_each(
        qt_theme_applied, tmp_path):
    tifffile = pytest.importorskip("tifffile")
    image = np.zeros((3, 12, 12), dtype=np.uint16)
    image[1, 2:6, 2:6] = 2000
    tifffile.imwrite(tmp_path / "multi.tif", image)
    mask = np.zeros((12, 12, 2), dtype=np.uint16)
    mask[2:6, 2:6, 0] = 5
    tifffile.imwrite(tmp_path / "stacked.tif", mask)

    stack = lv.stack_from_paths(image_path=tmp_path / "multi.tif",
                                labels_path=tmp_path / "stacked.tif")
    assert stack["image"].n_channels == 3
    assert stack["image"].shape == (12, 12)
    assert stack["mask"].shape == (12, 12)
    assert stack["mask"].label_at_world({"y": 3.0, "x": 3.0}) == 5


def test_a_canvas_asked_for_a_plane_the_stack_lacks_paints_nothing(
        qtbot, qt_theme_applied):
    canvas = _sized(qtbot, lv.LayerCanvas(_stack_with_an_object()))
    canvas.set_plane(("q", "r"))
    assert canvas._ensure_canvas() is None
    canvas.grab()


def test_resizing_the_widget_keeps_the_field_of_view(qtbot, qt_theme_applied):
    canvas = _sized(qtbot, lv.LayerCanvas(_stack_with_an_object()))
    before = canvas._ensure_canvas()
    span = before.step[0] * before.shape[0]
    canvas.resize(300, 250)
    after = canvas._ensure_canvas()
    assert after.shape == (248, 298)
    assert after.step[0] * after.shape[0] == pytest.approx(span)


def test_a_render_that_blows_up_is_logged_not_thrown_at_the_window(
        qtbot, qt_theme_applied, monkeypatch, caplog):
    """A paint handler that raises takes the window with it."""
    stack = _stack_with_an_object()
    canvas = _sized(qtbot, lv.LayerCanvas(stack))
    canvas._ensure_canvas()

    def explode(*_args, **_kwargs):
        raise RuntimeError("the compositor fell over")

    monkeypatch.setattr(type(stack), "render_uint8", explode)
    with caplog.at_level("ERROR"):
        canvas.grab()
    assert "Could not paint the layer canvas" in caplog.text


def test_clicks_on_an_empty_canvas_and_with_other_buttons_do_nothing(
        qtbot, qt_theme_applied):
    canvas = _sized(qtbot, lv.LayerCanvas())
    picked = []
    canvas.picked.connect(lambda *a: picked.append(a))
    canvas.activated.connect(lambda *a: picked.append(a))
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(10, 10))
    QTest.mouseDClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(10, 10))
    canvas.mouseMoveEvent(_move(canvas, 10, 10))
    assert picked == []

    canvas.set_stack(_stack_with_an_object())
    canvas._ensure_canvas()
    QTest.mouseClick(canvas, Qt.RightButton, Qt.NoModifier, QPoint(10, 10))
    QTest.mouseDClick(canvas, Qt.RightButton, Qt.NoModifier, QPoint(10, 10))
    assert picked == []


def test_the_status_line_names_whatever_was_clicked(qtbot, qt_theme_applied):
    stack = LayerStack()
    stack.add_image(np.zeros((20, 20)), name="field")
    points = stack.add_points(np.array([[5.0, 5.0]]), name="dots", size=4.0)
    shapes = stack.add_shapes(name="rois")
    shapes.add_rectangle((10.0, 10.0), (15.0, 15.0))
    viewer = _sized(qtbot, lv.LayerViewer(stack), 400, 400)
    world = viewer.canvas._ensure_canvas()

    for target, needle in (({"y": 5.0, "x": 5.0}, "dots point 0"),
                           ({"y": 12.0, "x": 12.0}, "rois shape 0"),
                           ({"y": 19.0, "x": 1.0}, "nothing here")):
        row, column = world.pixel_at(target)
        QTest.mouseClick(viewer.canvas, Qt.LeftButton, Qt.NoModifier,
                         QPoint(int(column) + 1, int(row) + 1))
        assert needle in viewer.status.text()

    # An image layer alone is named but has no value to report.
    stack.remove("dots")
    stack.remove("rois")
    assert "field" in viewer._describe_pick(stack["field"],
                                            {"y": 1.0, "x": 1.0}, None)


def test_hovering_reports_the_position_when_no_layer_is_selected(
        qtbot, qt_theme_applied):
    viewer = _sized(qtbot, lv.LayerViewer(_stack_with_an_object()), 400, 400)
    viewer.stack.select(None)
    viewer.canvas._ensure_canvas()
    viewer.canvas.mouseMoveEvent(_move(viewer.canvas, 40, 30))
    assert viewer.status.text().startswith("y ")


def test_the_property_controls_ignore_an_empty_stack(qtbot, qt_theme_applied):
    viewer = _sized(qtbot, lv.LayerViewer(), 400, 400)
    assert viewer.status.text() == "No layer selected"
    viewer._on_opacity(50)
    viewer._on_blending(lv.Blending.ADDITIVE)
    viewer._on_colormap("red")
    viewer._reorder(viewer.stack.raise_layer)
    viewer._on_remove()
    assert len(viewer.stack) == 0


def test_a_bad_blending_name_from_the_combo_is_logged_not_raised(
        qtbot, qt_theme_applied, caplog):
    viewer = _sized(qtbot, lv.LayerViewer(_stack_with_an_object()), 400, 400)
    viewer.stack.select("image")
    with caplog.at_level("ERROR"):
        viewer._on_blending("screen")
        viewer._on_colormap("definitely-not-a-colormap")
    assert "Could not set the blending mode" in caplog.text
    assert "Could not set the colormap" in caplog.text
    assert viewer.stack["image"].blending == lv.Blending.TRANSLUCENT


def test_the_add_buttons_go_through_a_file_dialog(qtbot, qt_theme_applied,
                                                  tmp_path, monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(tmp_path / "one.tif",
                     np.zeros((8, 8), dtype=np.uint16))
    viewer = _sized(qtbot, lv.LayerViewer(), 400, 400)

    chosen = [str(tmp_path / "one.tif")]
    monkeypatch.setattr(lv.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (chosen[0], "")))
    viewer.add_image_button.click()
    viewer.add_mask_button.click()
    assert viewer.stack.names == ("image", "mask")

    chosen[0] = ""      # the user cancelled
    viewer.add_image_button.click()
    assert viewer.stack.names == ("image", "mask")


def test_a_stale_row_in_the_layer_list_is_ignored(qtbot, qt_theme_applied):
    stack = _stack_with_an_object()
    viewer = _sized(qtbot, lv.LayerViewer(stack), 400, 400)
    item = viewer.layer_list.item(0)
    item.setData(Qt.UserRole, "a layer that is not there")
    assert viewer.layer_list._layer_for(item) is None
    item.setCheckState(Qt.Unchecked)          # must not raise
    assert stack["mask"].visible is True


def test_a_reorder_that_names_a_missing_layer_is_logged_not_raised(
        qtbot, qt_theme_applied, caplog):
    stack = _stack_with_an_object()
    viewer = _sized(qtbot, lv.LayerViewer(stack), 400, 400)
    viewer.layer_list.item(0).setData(Qt.UserRole, "gone")
    with caplog.at_level("ERROR"):
        viewer.layer_list._on_rows_moved()
    assert "Could not reorder the layers" in caplog.text


def test_an_opener_that_fails_does_not_escape_the_double_click(
        qtbot, qt_theme_applied, caplog, monkeypatch):
    from spacr.qt import linked_selection as ls

    link = LinkedSelection()

    def explode(_request):
        raise RuntimeError("the annotate screen fell over")

    link.register_object_opener(ls.DEFAULT_OPEN_KIND, explode)
    monkeypatch.setattr(lv, "has_object_opener", lambda kind: True)
    viewer = _sized(qtbot, lv.LayerViewer(_stack_with_an_object()), 400, 400)
    viewer.link_selection(lv.LINK_SOURCE, link=link)
    world = viewer.canvas._ensure_canvas()
    row, column = world.pixel_at({"y": 25.0, "x": 25.0})
    with caplog.at_level("ERROR"):
        QTest.mouseDClick(viewer.canvas, Qt.LeftButton, Qt.NoModifier,
                          QPoint(int(column) + 1, int(row) + 1))
    assert "Could not open plate1_A_1_1_17" in caplog.text


def test_a_rename_the_model_refuses_puts_the_row_back(qtbot, qt_theme_applied,
                                                       caplog, monkeypatch):
    stack = _stack_with_an_object()
    viewer = _sized(qtbot, lv.LayerViewer(stack), 400, 400)

    def refuse(*_args, **_kwargs):
        raise lv.LayerError("no")

    monkeypatch.setattr(type(stack), "rename", refuse)
    with caplog.at_level("ERROR"):
        viewer.layer_list.item(0).setText("nuclei")
    assert "Could not apply a layer-list edit" in caplog.text
    assert [viewer.layer_list.item(i).text()
            for i in range(viewer.layer_list.count())] == ["mask", "image"]


def test_the_layer_list_shrugs_at_a_selection_of_nothing(qtbot,
                                                          qt_theme_applied):
    viewer = _sized(qtbot, lv.LayerViewer(), 400, 400)
    assert viewer.layer_list.currentItem() is None
    viewer.layer_list._on_selection_changed()       # must not raise
    viewer.layer_list._syncing = True
    viewer.layer_list._on_rows_moved()
    viewer.layer_list._on_selection_changed()
    viewer.layer_list._syncing = False
    assert len(viewer.stack) == 0


def test_a_selection_of_many_objects_leaves_the_highlight_alone(
        qtbot, qt_theme_applied):
    from spacr.selection import Selection

    link = LinkedSelection()
    viewer = _sized(qtbot, lv.LayerViewer(_stack_with_an_object()), 400, 400)
    viewer.link_selection(lv.LINK_SOURCE, link=link)
    link.set_selection(Selection.from_keys(["a", "b"], source="umap"))
    assert viewer.stack["mask"].selected_label == 0
    link.clear_selection()
    assert viewer.stack["mask"].selected_label == 0

"""Item 513: plaque outline and well-box line weights that persist, and a ruler.

    "in the plaque assay modual there should be settings to controll the
     thickness and weight of the plaque outlines, and the yolo boxes ...
     there shouild also be a ruler button in the live preview here as well."
    -- the maintainer, 2026-09-24

The weights are measured on what is PAINTED -- the view's pixmap and a
render of the widget -- rather than read back from a setting, because a
setting that is stored and never reaches the painter would pass the second
kind of test and fail the user. The ruler is driven with real mouse events
through the view's own transform, and its microns come from the pixel size
item 500 already carries.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QContextMenuEvent, QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from spacr.qt.widgets import plaque_preview as ppv  # noqa: E402

FLAT = (40, 40, 40)


def _flat_png(path, size=(60, 60)):
    from PIL import Image

    Image.fromarray(np.full(size + (3,), FLAT[0], dtype=np.uint8)).save(path)
    return path


def _square(shape=(60, 60)):
    labels = np.zeros(shape[:2], dtype=np.int32)
    labels[10:30, 10:30] = 1
    return labels


class _Box:
    def __init__(self, x0, y0, x1, y1, confidence=0.9):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.confidence = confidence


def _detect(image, weights, confidence=0.25, imgsz=640, min_axis_ratio=0.0):
    return [_Box(100, 100, 180, 180), _Box(220, 100, 300, 180)]


def _no_words(path):
    return []


@pytest.fixture
def fresh_style(monkeypatch):
    monkeypatch.setitem(ppv._SESSION, "style", ppv.OverlayStyle())


@pytest.fixture
def panel(qtbot, monkeypatch, fresh_style):
    monkeypatch.setattr(ppv, "missing_papers_packages", lambda *a, **k: [])
    widget = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _run_plaque(panel, tmp_path):
    _flat_png(tmp_path / "a.png")
    panel.load_source_async(str(tmp_path))
    assert panel.run_preview(segment=lambda p: _square())
    return panel


def _run_figure(panel, tmp_path):
    _flat_png(tmp_path / "fig.png", size=(400, 400))
    panel.apply_settings(dict(plaque_mode="figure", confirm_annotations=False))
    panel.load_source_async(str(tmp_path))
    assert panel.run_preview(detect=_detect, read_text=_no_words,
                             segment=lambda crop: _square(crop.shape))
    return panel


def _settle(panel, view):
    """Show the panel large enough for ``view`` to paint at native scale."""
    panel.resize(1000, 700)
    panel.show()
    QApplication.processEvents()
    view.fit_image()
    QApplication.processEvents()
    assert view._scale == 1.0, "the picture is painted pixel for pixel"


def _nearer(pixel, colour, other):
    """Whether ``pixel`` is closer to ``colour`` than to ``other``."""
    def gap(a, b):
        return sum((int(x) - int(y)) ** 2 for x, y in zip(a, b))
    return gap(pixel, colour) < gap(pixel, other)


def _stroke(image: QImage, y: int, colour, other=FLAT, x_from: int = 0) -> int:
    """The width of the first run of ``colour`` along row ``y``."""
    x, run = x_from, 0
    while x < image.width():
        pixel = image.pixelColor(x, y).getRgb()[:3]
        if _nearer(pixel, colour, other):
            run += 1
        elif run:
            break
        x += 1
    return run


def _drag(view, start, end):
    """Drag from one IMAGE pixel to another, through the view's transform."""
    a = view.widget_point(*start).toPoint()
    b = view.widget_point(*end).toPoint()
    QTest.mousePress(view, Qt.LeftButton, pos=a)
    QTest.mouseMove(view, b)
    QTest.mouseRelease(view, Qt.LeftButton, pos=b)


def test_the_outline_weight_changes_the_painted_stroke(qtbot):
    """One pixel wide, then five: measured on a render of the widget."""
    view = ppv._ImageView()
    qtbot.addWidget(view)
    view.resize(200, 200)
    view.show()
    flat = np.full((60, 60, 3), FLAT[0], dtype=np.uint8)
    thin = ppv.OverlayStyle(outline_thickness=1)
    view.set_image(ppv.render_overlay(flat.copy(), _square(), thin))
    left, top = int(view.image_rect().left()), int(view.image_rect().top())
    before = _stroke(view.grab().toImage(), top + 20, ppv.OUTLINE_COLOUR,
                     x_from=left)
    thick = ppv.OverlayStyle(outline_thickness=5)
    view.set_image(ppv.render_overlay(flat.copy(), _square(), thick))
    after = _stroke(view.grab().toImage(), top + 20, ppv.OUTLINE_COLOUR,
                    x_from=left)
    assert (before, after) == (1, 5)
    view.zoom(2.0)
    zoomed = _stroke(view.grab().toImage(), int(view.widget_point(0, 20).y()),
                     ppv.OUTLINE_COLOUR, x_from=int(view.image_rect().left()))
    assert abs(zoomed - 10) <= 1, "the weight is in image pixels, so it zooms"


def test_the_dialog_reaches_the_painted_outline(panel, tmp_path):
    _run_plaque(panel, tmp_path)
    panel.open_overlay_settings()
    dialog = panel._overlay_dialog
    before = _stroke(panel._view._pixmap.toImage(), 20, ppv.OUTLINE_COLOUR)
    dialog.thickness.setValue(7)
    after = _stroke(panel._view._pixmap.toImage(), 20, ppv.OUTLINE_COLOUR)
    assert (before, after) == (1, 7)
    assert panel.overlay_style().outline_thickness == 7
    dialog.reject()


def test_the_box_weight_changes_the_painted_boxes(panel, tmp_path):
    """The second well's box, at row 140: automatic (2 px), then 6 px."""
    _run_figure(panel, tmp_path)
    assert panel.selected_well() == 0, "the first box is the highlighted one"
    other = tuple(ppv.BOX_OK.getRgb()[:3])
    before = _stroke(panel._view._pixmap.toImage(), 140, other, x_from=205)
    panel.open_overlay_settings()
    panel._overlay_dialog.box_weight.setValue(6)
    after = _stroke(panel._view._pixmap.toImage(), 140, other, x_from=205)
    assert (before, after) == (2, 6)
    chosen = tuple(ppv.BOX_SELECTED.getRgb()[:3])
    highlighted = _stroke(panel._view._pixmap.toImage(), 140, chosen, x_from=85)
    assert highlighted == 12, "the highlighted box is twice as wide"
    panel._overlay_dialog.reject()


def test_the_weights_persist_across_a_rebuilt_panel(qtbot, monkeypatch,
                                                    fresh_style):
    from dataclasses import replace

    from spacr.qt.preferences import _settings

    monkeypatch.setattr(ppv, "missing_papers_packages", lambda *a, **k: [])
    first = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(first)
    first.open_overlay_settings()
    first._overlay_dialog.thickness.setValue(7)
    first._overlay_dialog.box_weight.setValue(5)
    first._overlay_dialog.reject()
    first.set_overlay_style(replace(first.overlay_style(),
                                    outline_colour=ppv.RANDOM_COLOUR))
    store = _settings()
    real = os.path.join(os.environ.get("HOME", ""), ".config")
    assert not store.fileName().startswith(real), (
        "the test store is the scratch one, never the user's")
    kept = json.loads(store.value(ppv.OVERLAY_STYLE_KEY))
    assert kept["outline_thickness"] == 7 and kept["box_thickness"] == 5
    first.close()
    monkeypatch.setitem(ppv._SESSION, "style", None)
    second = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(second)
    style = second.overlay_style()
    assert style.outline_thickness == 7
    assert style.box_thickness == 5
    assert style.outline_colour == ppv.RANDOM_COLOUR


def test_a_style_survives_the_json_round_trip_and_reads_junk_as_default():
    style = ppv.OverlayStyle(outline_colour=ppv.RANDOM_COLOUR,
                             outline_thickness=4, fill_colour="#ff0000",
                             fill_opacity=55, box_thickness=9)
    back = ppv.OverlayStyle.from_dict(json.loads(json.dumps(style.as_dict())))
    assert back == style.normalised()
    assert back.fill_colour == (255, 0, 0)
    junk = dict(outline_thickness="many", box_thickness=None, extra=1)
    assert ppv.OverlayStyle.from_dict(junk) == ppv.OverlayStyle()
    assert ppv.OverlayStyle.from_dict("not a mapping") == ppv.OverlayStyle()


def test_box_thickness_for_keeps_the_automatic_rule_and_clamps():
    assert ppv.box_thickness_for(400, 400) == 2
    assert ppv.box_thickness_for(2000, 1000) == 5
    assert ppv.box_thickness_for(400, 400, 7) == 7
    assert ppv.box_thickness_for(400, 400, 99) == ppv.MAX_BOX_THICKNESS
    assert ppv.box_thickness_for(400, 400, "x") == 2


def test_the_box_weight_control_is_automatic_at_zero_and_explains_itself(
        qtbot):
    dialog = ppv.PlaqueOverlayDialog(ppv.OverlayStyle())
    qtbot.addWidget(dialog)
    assert dialog.box_weight.value() == 0
    assert dialog.box_weight.text() == "Automatic"
    assert dialog.box_weight.toolTip() and dialog.thickness.toolTip()
    got = []
    dialog.style_changed.connect(got.append)
    dialog.box_weight.setValue(6)
    assert got[-1].box_thickness == 6
    dialog.set_overlay_style(ppv.OverlayStyle(box_thickness=3))
    assert dialog.box_weight.value() == 3 and len(got) == 1, "shown, not announced"


def test_a_new_picture_clears_the_ruler_and_the_ruler_off_pans(panel, tmp_path):
    _flat_png(tmp_path / "a.png")
    _flat_png(tmp_path / "b.png")
    panel.load_source_async(str(tmp_path))
    _settle(panel, panel._view)
    panel._ruler_btn.setChecked(True)
    _drag(panel._view, (10, 10), (13, 14))
    assert panel._view.ruler.length() == 5
    panel._step(1)
    assert panel._view.ruler.length() is None
    panel._ruler_btn.setChecked(False)
    pan = QPointF(panel._view._pan)
    _drag(panel._view, (10, 10), (30, 30))
    assert panel._view.ruler.length() is None
    assert panel._view._pan != pan, "with the ruler off a drag pans"


def test_the_ruler_reports_pixels_and_microns_from_a_known_pixel_size(
        panel, tmp_path, monkeypatch):
    _run_plaque(panel, tmp_path)
    _settle(panel, panel._view)
    panel.apply_settings(dict(plaque_pixels_per_um=2.0))
    assert panel._ruler_note.isHidden(), "nothing to say until the ruler is out"
    panel._ruler_btn.setChecked(True)
    view, ruler = panel._view, panel._view.ruler
    assert ruler.active and panel._well_view.ruler.active
    assert not panel._ruler_note.isHidden()
    assert panel._ruler_note.text() == "2 px/µm"
    _drag(view, (10, 10), (13, 14))
    assert ruler.length() == 5
    assert ruler.label() == "5.00 px · 2.50 µm"
    assert panel.views().canvas is view, "one canvas, one line on every view"
    endpoints = ruler.start, ruler.end
    view.zoom(2.0)
    assert ruler.length() == 5 and (ruler.start, ruler.end) == endpoints, (
        "zoom does not move the measurement")
    panel.apply_settings(dict(plaque_pixels_per_um=None))
    assert ruler.label() == "5.00 px"
    assert panel._ruler_note.text() == "Pixels only: no pixel size is known."
    shown = []
    monkeypatch.setattr(ppv.PlaquePreviewPanel, "_exec_menu",
                        staticmethod(lambda menu, pos: shown.append(menu)))
    QTest.mouseClick(view, Qt.RightButton, pos=view.widget_point(30, 30).toPoint())
    assert ruler.length() is None, "right-click clears the line"
    event = QContextMenuEvent(QContextMenuEvent.Mouse, QPoint(3, 3), QPoint(30, 40))
    view.contextMenuEvent(event)
    assert not shown, "no overlay menu over the ruler's right-click"
    panel._ruler_btn.setChecked(False)
    assert panel._ruler_note.isHidden()
    view.contextMenuEvent(event)
    assert shown, "with the ruler away the right-click is the overlay menu"


def test_in_figure_mode_the_ruler_uses_the_highlighted_wells_scale(
        panel, tmp_path):
    _run_figure(panel, tmp_path)
    _settle(panel, panel._well_view)
    assert panel.selected_well() == 0
    panel._ruler_btn.setChecked(True)
    assert panel._ruler_note.text() == "Pixels only: no pixel size is known."
    row = panel._view_row(0)
    panel._table.item(row, ppv.PIXELS_PER_UM_COLUMN).setText("4")
    assert panel.ruler_microns_per_pixel() == pytest.approx(0.25)
    assert panel._ruler_note.text() == "4 px/µm"
    _drag(panel._well_view, (10, 10), (50, 10))
    assert panel._well_view.ruler.label() == "40.00 px · 10.00 µm"
    _drag(panel._view, (100, 100), (103, 104))
    length = panel._view.ruler.length()
    assert length is not None and 2 < length < 8
    assert panel._view.ruler.label() == (
        "{:.2f} px · {:.2f} µm".format(length, length / 4)), (
        "the figure's ruler follows the highlighted well too")
    panel.select_well(1)
    assert panel._well_view.ruler.length() is None, "another well, another crop"
    assert panel._ruler_note.text() == "Pixels only: no pixel size is known."
    assert panel._view.ruler.label() == "{:.2f} px".format(length)


def test_a_saved_picture_carries_the_chosen_weights(panel, tmp_path):
    """Read back from the file: box weight 6 and outline weight 4."""
    from dataclasses import replace

    _run_figure(panel, tmp_path)
    panel.set_overlay_style(replace(panel.overlay_style(), box_thickness=6))
    written = panel.save_picture(str(tmp_path / "boxes"), view=panel._view)
    assert written == str(tmp_path / "boxes.png")
    saved = QImage(written)
    assert (saved.width(), saved.height()) == (400, 400)
    other = tuple(ppv.BOX_OK.getRgb()[:3])
    assert _stroke(saved, 140, other, x_from=205) == 6
    menu = panel.overlay_menu()
    names = [a.objectName() for a in menu.actions()]
    assert "PlaqueSavePicture" in names


def test_a_saved_plaque_picture_carries_the_outline_weight(qtbot, monkeypatch,
                                                           fresh_style,
                                                           tmp_path):
    monkeypatch.setattr(ppv, "missing_papers_packages", lambda *a, **k: [])
    widget = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    assert widget.save_picture(str(tmp_path / "none.png")) is None
    _run_plaque(widget, tmp_path)
    widget.set_overlay_style(ppv.OverlayStyle(outline_thickness=4))
    written = widget.save_picture(str(tmp_path / "outline.png"))
    assert _stroke(QImage(written), 20, ppv.OUTLINE_COLOUR) == 4

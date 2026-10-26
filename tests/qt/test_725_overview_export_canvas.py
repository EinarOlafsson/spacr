"""Round-5 Overview export, Custom: the canvas model, its PDF writer and the composer.

Item 725, maintainer feedback 2026-10-25 round 5, point 8: "the user should see
a canvas whith a grid they should be able to draw text boxes on this grid or
drag and drop images to this grid. then export the canvas as pdf."
"""
import json
import re

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QMimeData, QPointF, QUrl
from pypdf import PdfReader

from spacr.qt.widgets import picture_export as export
from spacr.qt.widgets.picture_export import _CanvasComposer, _CanvasLayout, _canvas_pdf


def _pixels(rows=517, columns=733):
    """A deterministic RGB image with no repeated rows."""
    generator = np.random.default_rng(7)
    return (generator.random((rows, columns, 3)) * 255).astype(np.uint8)


def _image_placements(page):
    """``[(x, y, width, height)]`` in points of every image drawn on a PDF page.

    matplotlib draws an image as ``1 0 0 1 x y cm`` then ``w 0 0 -h 0 h cm``
    before ``Do``, so its rows are stored bottom first; ``y`` is measured
    from the bottom of the page.
    """
    content = page.get_contents().get_data().decode("latin-1")
    pattern = re.compile(r"1 0 0 1 ([\d.]+) ([\d.]+) cm ([\d.]+) 0 0 -([\d.]+) 0 [\d.]+ cm\s*/I\d+ Do")
    return [tuple(float(value) for value in match.groups()) for match in pattern.finditer(content)]


def test_boxes_snap_to_the_grid_when_added_moved_and_resized():
    layout = _CanvasLayout(grid=18)
    box = layout.add_box(0, "text", 40, 25, 100, 30, text="Title")
    assert (box["x"], box["y"], box["w"], box["h"]) == (36, 18, 108, 36)
    layout.move_boxes([box["id"]], 10, 30)
    assert (box["x"], box["y"]) == (54, 54)
    layout.resize_box(box["id"], 70, 10)
    assert (box["w"], box["h"]) == (72, 18)
    layout.set_grid(snap=False)
    layout.move_boxes([box["id"]], 1.5, 2.5)
    assert (box["x"], box["y"]) == (55.5, 56.5)


def test_boxes_stay_on_the_page_and_never_collapse():
    layout = _CanvasLayout(grid=10)
    box = layout.add_box(0, "text", 580, 830, 2, 2)
    width, height = layout.page_points
    assert box["w"] >= export._CANVAS_MIN_SIDE and box["h"] >= export._CANVAS_MIN_SIDE
    assert box["x"] + box["w"] <= width and box["y"] + box["h"] <= height
    layout.move_boxes([box["id"]], -5000, -5000)
    assert (box["x"], box["y"]) == (0, 0)


def test_images_keep_their_aspect_ratio_until_it_is_unlocked():
    layout = _CanvasLayout(grid=9)
    box = layout.add_box(0, "image", 18, 18, 200, 50, aspect=2.0, source={"type": "file", "path": "a.png"})
    assert box["w"] == 198 and box["h"] == pytest.approx(99)
    layout.resize_box(box["id"], 300, 10)
    assert box["h"] == pytest.approx(box["w"] / 2)
    layout.update_box(box["id"], lock_aspect=False)
    layout.resize_box(box["id"], 90, 270)
    assert (box["w"], box["h"]) == (90, 270)
    layout.update_box(box["id"], lock_aspect=True)
    assert box["h"] == pytest.approx(45)


def test_text_and_image_boxes_carry_their_own_properties():
    layout = _CanvasLayout()
    text = layout.add_box(0, "text", 0, 0, 90, 36, text="A", font_size=14, bold=True, italic=True, align="right")
    image = layout.add_box(0, "image", 0, 54, 90, 90, source={"type": "file", "path": "x.tif"},
                           pixel_size_um=.5, scale_bar=True, aspect=1.0)
    assert (text["font_size"], text["bold"], text["italic"], text["align"]) == (14, True, True, "right")
    assert image["pixel_size_um"] == .5 and image["scale_bar"] and image["lock_aspect"]
    assert layout.add_box(0, "text", 0, 0, 9, 9, align="sideways")["align"] == "left"


def test_z_order_moves_boxes_up_and_down_their_page():
    layout = _CanvasLayout()
    a, b, c = (layout.add_box(0, "text", 0, 0, 36, 36, text=name) for name in "abc")
    order = lambda: [box["text"] for box in layout.pages[0]]
    layout.restack([a["id"]], "front")
    assert order() == ["b", "c", "a"]
    layout.restack([a["id"]], "backward")
    assert order() == ["b", "a", "c"]
    layout.restack([a["id"], c["id"]], "back")
    assert order() == ["a", "c", "b"]
    layout.restack([c["id"]], "forward")
    assert order() == ["a", "b", "c"]


def test_pages_are_added_removed_and_reordered():
    layout = _CanvasLayout()
    layout.add_box(0, "text", 0, 0, 36, 36, text="first")
    layout.add_page()
    layout.add_box(1, "text", 0, 0, 36, 36, text="second")
    layout.add_page(at=0)
    assert [len(page) for page in layout.pages] == [0, 1, 1]
    layout.move_page(2, 0)
    assert layout.pages[0][0]["text"] == "second"
    layout.remove_page(1)
    assert [page[0]["text"] for page in layout.pages] == ["second", "first"]
    layout.remove_page(0)
    layout.remove_page(0)
    assert layout.pages == [[]]


def test_every_change_can_be_undone_and_redone():
    layout = _CanvasLayout(grid=18)
    box = layout.add_box(0, "text", 36, 36, 72, 36, text="x")
    layout.move_boxes([box["id"]], 36, 0)
    layout.delete_boxes([box["id"]])
    assert layout.pages == [[]]
    assert layout.undo() and layout.pages[0][0]["x"] == 72
    assert layout.undo() and layout.pages[0][0]["x"] == 36
    assert layout.redo() and layout.pages[0][0]["x"] == 72
    assert layout.undo() and layout.undo() and layout.pages == [[]]
    assert not layout.undo()
    layout.redo()
    layout.add_page()
    assert not layout.can_redo()


def test_a_saved_layout_reopens_exactly(tmp_path):
    layout = _CanvasLayout("Letter", "landscape", grid=12, show_grid=False, snap=False)
    layout.add_box(0, "text", 10, 20, 100, 40, text="Line one\nline two", font_size=11, italic=True)
    layout.add_page()
    layout.add_box(1, "image", 30, 40, 120, 90, source={"type": "file", "path": "cells.tif"},
                   aspect=4 / 3, native=[400, 300], pixel_size_um=.65, scale_bar=True, label="cells")
    path = layout.save(tmp_path / "results" / "layout.json")
    data = json.loads(open(path, encoding="utf-8").read())
    assert data["schema"] == "spacr.canvas_layout/1"
    reopened = _CanvasLayout.load(path)
    assert reopened.to_dict() == layout.to_dict()
    assert reopened.page_points == (792.0, 612.0)
    assert reopened.add_box(0, "text", 0, 0, 9, 9)["id"] == 3
    with pytest.raises(ValueError):
        _CanvasLayout.from_dict({"pages": []})


def test_the_pdf_has_one_page_per_canvas_page_at_the_page_size(tmp_path):
    layout = _CanvasLayout("Letter", "landscape")
    layout.add_page()
    layout.add_page()
    path, errors = _canvas_pdf(layout, tmp_path / "out.pdf", lambda box: _pixels())
    reader = PdfReader(path)
    assert errors == [] and len(reader.pages) == 3
    assert [float(value) for value in reader.pages[0].mediabox] == [0, 0, 792, 612]


def test_pdf_images_sit_where_the_boxes_are_and_keep_every_pixel(tmp_path):
    layout = _CanvasLayout(grid=9)
    pixels = _pixels()
    box = layout.add_box(0, "image", 72, 108, 360, 1, aspect=733 / 517, source={"type": "test"})
    path, _ = _canvas_pdf(layout, tmp_path / "image.pdf", lambda _box: pixels)
    page = PdfReader(path).pages[0]
    (x, y, width, height), = _image_placements(page)
    page_height = layout.page_points[1]
    assert (x, width) == pytest.approx((box["x"], box["w"]), abs=1e-3)
    assert height == pytest.approx(box["h"], abs=1e-3)
    assert page_height - y - height == pytest.approx(box["y"], abs=1e-3)
    embedded, = page.images
    assert embedded.image.size == (733, 517)
    stored = np.asarray(embedded.image.convert("RGB"))
    assert np.array_equal(stored[::-1], pixels)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_canvas_text_uses_print_ink_on_its_white_page(theme, monkeypatch):
    from matplotlib.colors import to_rgba
    from spacr.figures import style
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "resolve_effective_theme", lambda: theme)
    monkeypatch.setattr(preferences, "get_figure_color_tokens", lambda: ("auto", "auto"))
    layout = _CanvasLayout()
    layout.add_box(0, "text", 36, 36, 160, 80, text="Cell measurements")
    figure, images, records, errors = export._canvas_page_figure(layout, 0, lambda box: _pixels())
    try:
        assert figure.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
        assert to_rgba(figure.texts[0].get_color()) == to_rgba(style.INK_PRINT)
        assert figure.texts[0].get_text() == "Cell measurements"
        figure.canvas.draw()
        assert np.any(np.asarray(figure.canvas.buffer_rgba())[..., :3] < 128)
        assert not images and not errors and records[0]["text"] == "Cell measurements"
    finally:
        figure.clear()


def test_pdf_text_is_vector_text_wrapped_inside_its_box(tmp_path):
    layout = _CanvasLayout()
    layout.add_box(0, "text", 36, 36, 160, 80, font_size=12, bold=True,
                   text="Infected HeLa cells imaged twenty four hours after infection")
    path, _ = _canvas_pdf(layout, tmp_path / "text.pdf", lambda box: _pixels())
    page = PdfReader(path).pages[0]
    lines = [line for line in page.extract_text().splitlines() if line.strip()]
    assert " ".join(lines) == "Infected HeLa cells imaged twenty four hours after infection"
    assert len(lines) > 1
    assert len(page.images) == 0
    content = page.get_contents().get_data().decode("latin-1")
    assert "BT" in content and "/F1 12 Tf" in content


def test_a_scale_bar_is_drawn_when_the_pixel_size_is_known(tmp_path):
    layout = _CanvasLayout()
    layout.add_box(0, "image", 36, 36, 300, 200, aspect=733 / 517, scale_bar=True, pixel_size_um=.65,
                   source={"type": "test"})
    layout.add_box(0, "image", 36, 300, 300, 200, aspect=733 / 517, scale_bar=True, source={"type": "test"})
    path, _ = _canvas_pdf(layout, tmp_path / "bar.pdf", lambda box: _pixels())
    text = PdfReader(path).pages[0].extract_text()
    assert text.count("µm") == 1 and "100 µm" in text
    assert export._canvas_scale_length(733 * .65) == 100
    assert export._canvas_scale_length(40) == 10
    assert export._canvas_scale_length(7) == 1


def test_a_fitted_image_is_centred_without_stretching(tmp_path):
    layout = _CanvasLayout(snap=False)
    layout.add_box(0, "image", 36, 36, 400, 100, fit=True, lock_aspect=False, source={"type": "test"})
    path, _ = _canvas_pdf(layout, tmp_path / "fit.pdf", lambda box: _pixels(100, 100))
    (x, _y, width, height), = _image_placements(PdfReader(path).pages[0])
    assert width == pytest.approx(height) == pytest.approx(100)
    assert x == pytest.approx(36 + 150)


def test_an_unreadable_image_is_reported_and_the_rest_is_written(tmp_path):
    layout = _CanvasLayout()
    layout.add_box(0, "image", 36, 36, 90, 90, label="gone", source={"type": "file", "path": str(tmp_path / "x.png")})
    layout.add_box(0, "text", 36, 200, 200, 30, text="still here")
    path, errors = _canvas_pdf(layout, tmp_path / "partial.pdf",
                               lambda box: export._canvas_box_pixels(box, None, None))
    assert [name for name, _error in errors] == ["gone"]
    assert "still here" in PdfReader(path).pages[0].extract_text()


def test_a_cancelled_export_writes_nothing(tmp_path):
    import threading

    cancelled = threading.Event()
    cancelled.set()
    path, _ = _canvas_pdf(_CanvasLayout(), tmp_path / "never.pdf", lambda box: _pixels(), cancelled)
    assert path is None and not (tmp_path / "never.pdf").exists()
    assert [entry.name for entry in tmp_path.iterdir()] == []


def test_image_files_are_read_at_full_size_or_fitted(tmp_path):
    from PIL import Image

    rgb = _pixels(40, 60)
    Image.fromarray(rgb).save(tmp_path / "a.png")
    tifffile.imwrite(tmp_path / "b.tif", (np.arange(30 * 50, dtype=np.uint16).reshape(30, 50) * 7))
    tifffile.imwrite(tmp_path / "c.tif", np.stack([rgb[..., 0]] * 3))
    assert np.array_equal(export._canvas_file_pixels(tmp_path / "a.png"), rgb)
    grey = export._canvas_file_pixels(tmp_path / "b.tif")
    assert grey.shape == (30, 50, 3) and grey.dtype == np.uint8 and grey.max() == 255 and grey.min() == 0
    assert export._canvas_file_pixels(tmp_path / "c.tif").shape == (40, 60, 3)
    assert export._canvas_file_pixels(tmp_path / "a.png", edge=20).shape[:2] == (14, 20)


@pytest.fixture
def composer(qtbot):
    widget = _CanvasComposer()
    qtbot.addWidget(widget)
    widget.resize(1200, 820)
    widget.show()
    return widget


def test_the_composer_draws_text_boxes_with_the_text_tool(composer, qtbot):
    from PySide6.QtCore import Qt

    composer.set_tool("text")
    view = composer.view
    start = view.mapFromScene(QPointF(40, 40))
    end = view.mapFromScene(QPointF(200, 90))
    qtbot.mousePress(view.viewport(), Qt.LeftButton, pos=start)
    qtbot.mouseMove(view.viewport(), pos=end)
    qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=end)
    box, = composer.canvas.pages[0]
    assert box["kind"] == "text" and box["x"] % 18 == 0 and box["w"] >= 144
    assert composer.selected_ids() == [box["id"]] and composer._tool == "select"
    composer.text_edit.setPlainText("Typed caption")
    composer.font_size.setValue(18)
    composer.bold.setChecked(True)
    composer.align.setCurrentIndex(composer.align.findData("center"))
    assert (box["text"], box["font_size"], box["bold"], box["align"]) == ("Typed caption", 18, True, "center")
    composer.undo()
    composer.undo()
    composer.undo()
    assert composer.canvas.pages[0][0]["font_size"] == 10 and composer.canvas.pages[0][0]["text"] == "Typed caption"
    composer.undo()
    assert composer.canvas.pages[0][0]["text"] == ""


def test_dragging_a_box_moves_it_onto_the_grid(composer, qtbot):
    from PySide6.QtCore import Qt

    box = composer.add_text(36, 36, 108, 36, text="drag me")
    view = composer.view
    start = view.mapFromScene(QPointF(60, 50))
    end = view.mapFromScene(QPointF(60 + 95, 50 + 40))
    qtbot.mousePress(view.viewport(), Qt.LeftButton, pos=start)
    qtbot.mouseMove(view.viewport(), pos=view.mapFromScene(QPointF(80, 60)))
    qtbot.mouseMove(view.viewport(), pos=end)
    qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=end)
    qtbot.waitUntil(lambda: composer.canvas.find(box["id"])[1]["x"] != 36)
    moved = composer.canvas.find(box["id"])[1]
    assert moved["x"] % 18 == 0 and moved["y"] % 18 == 0
    assert moved["x"] == pytest.approx(126, abs=18) and moved["y"] == pytest.approx(72, abs=18)


def test_image_files_drop_onto_the_page_and_multi_select_deletes(composer, qtbot, tmp_path):
    from PIL import Image

    for name in ("a.png", "b.png"):
        Image.fromarray(_pixels(60, 90)).save(tmp_path / name)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(tmp_path / "a.png")), QUrl.fromLocalFile(str(tmp_path / "b.png")),
                  QUrl.fromLocalFile(str(tmp_path / "notes.txt"))])
    images = export._canvas_dropped_images(mime)
    assert [image["label"] for image in images] == ["a.png", "b.png"]
    boxes = composer.add_images(images, QPointF(36, 300))
    assert all(box["kind"] == "image" for box in boxes) and boxes[1]["x"] > boxes[0]["x"]
    qtbot.waitUntil(lambda: len(composer._previews) == 2, timeout=10000)
    qtbot.waitUntil(lambda: all(box["aspect"] for box in composer.canvas.pages[0]), timeout=10000)
    assert composer.canvas.pages[0][0]["h"] == pytest.approx(composer.canvas.pages[0][0]["w"] * 60 / 90)
    composer.select([box["id"] for box in boxes])
    assert len(composer.selected_ids()) == 2
    composer.restack("front")
    composer.delete_selected()
    assert composer.canvas.pages == [[]]


def test_overview_drag_data_round_trips_through_the_drop(composer):
    images = [{"source": {"type": "overview_field", "key": ["", "p1", "A01", "1", "", ""]},
               "label": "p1 / A01 / 1", "pixel_size_um": .65}]
    dropped = export._canvas_dropped_images(export._canvas_mime(images))
    assert dropped == images
    box, = composer.add_images(dropped)
    assert box["source"]["key"][2] == "A01" and box["pixel_size_um"] == .65
    composer.select([box["id"]])
    assert composer.scale_bar.isEnabled()


def test_composer_pages_toolbar_save_open_and_export(composer, qtbot, tmp_path):
    composer.add_text(36, 36, 200, 36, text="page one")
    composer.add_page()
    composer.add_text(36, 36, 200, 36, text="page two")
    assert composer.pages.count() == 2 and composer.page_index == 1
    composer.move_page(-1)
    assert composer.canvas.pages[0][0]["text"] == "page two"
    composer.orientation.setCurrentIndex(composer.orientation.findData("landscape"))
    composer.page_size.setCurrentIndex(composer.page_size.findData("Letter"))
    assert composer.canvas.page_points == (792.0, 612.0)
    composer.grid_spin.setValue(24)
    composer.show_grid.setChecked(False)
    assert composer.canvas.grid == 24 and not composer.canvas.show_grid
    path = composer.save_layout(tmp_path / "results" / "canvas_layout.json")
    composer.remove_page()
    assert len(composer.canvas.pages) == 1
    assert composer.open_layout(path) and len(composer.canvas.pages) == 2
    assert not composer.open_layout(tmp_path / "missing.json")
    composer.export_pdf(str(tmp_path / "composed.pdf"))
    qtbot.waitUntil(lambda: composer.last_export is not None, timeout=30000)
    reader = PdfReader(composer.last_export)
    assert len(reader.pages) == 2 and "page two" in reader.pages[0].extract_text()
    assert "composed.pdf" in composer.status.text()


def test_nudging_and_page_switching_follow_the_grid(composer):
    box = composer.add_text(36, 36, 72, 36)
    composer.nudge(1, 0)
    composer.nudge(0, 1)
    assert (composer.canvas.find(box["id"])[1]["x"], composer.canvas.find(box["id"])[1]["y"]) == (54, 54)
    composer.add_page()
    composer.show_page(0)
    assert composer.page_index == 0 and composer.selected_ids() == []
    composer.redo()
    composer.undo()
    assert len(composer.canvas.pages) == 1

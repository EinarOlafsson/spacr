"""Saved boxes come back exactly after Next, a negative export and Prev (lesson 14 YOLO tour).

The capture host stopped the YOLO tour with "Navigation did not reload the
exact saved annotations". Offscreen, with the real driver and both load
paths (a field decoded in the click and one decoded on the loader thread),
the tour passes: Make Masks neither loses nor reorders saved boxes. The
host clicks with real input, which can return before the click is handled,
and the driver read the field before Prev had landed. These pin the editor
side, and the driver now waits for the field it asked for.
"""
from __future__ import annotations

import ast
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import MODE_BOX, MakeMasksScreen

pytestmark = pytest.mark.qt

DRIVER = Path(__file__).resolve().parents[2] / "tools" / "tutorials" / "capture_make_masks.py"


@pytest.fixture(params=["in_the_click", "on_the_loader"])
def opened(request, qtbot, qt_theme_applied, tmp_path, monkeypatch):
    folder = tmp_path / "fields"
    (folder / "masks").mkdir(parents=True)
    rng = np.random.default_rng(1)
    for index in range(2):
        image = rng.integers(0, 60000, (96, 128), dtype=np.uint16)
        labels = np.zeros((96, 128), np.uint16)
        labels[20:40, 30:60] = 1
        imageio.imwrite(folder / f"field_{index}.tif", image)
        imageio.imwrite(folder / "masks" / f"field_{index}.tif", labels)
    if request.param == "on_the_loader":
        monkeypatch.setattr(MakeMasksScreen, "_should_background_load",
                            staticmethod(lambda _path: True))
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget.warnings = []
    widget._warn = lambda title, text: widget.warnings.append((title, text))
    assert widget._open_folder(str(folder))
    qtbot.waitUntil(lambda: not widget._loading and widget._canvas.image is not None,
                    timeout=20000)
    yield widget, folder
    widget._magnifier.close()
    widget.close_folded()


def _landed(qtbot, widget, name):
    qtbot.waitUntil(lambda: not widget._loading
                    and widget._image_files[widget._current_index] == name,
                    timeout=20000)


def test_prev_after_a_negative_export_reloads_the_exact_saved_boxes(qtbot, opened, tmp_path):
    widget, folder = opened
    first, second = widget._image_files
    widget._set_mode(MODE_BOX)
    assert widget._on_add_box_class("demonstration") == 1
    widget._canvas.boxes = [(1, 40, 30, 90, 70)]
    widget._canvas.boxes_changed.emit()
    widget._canvas.boxes.append((1, 50, 40, 70, 60))
    widget._canvas.boxes_changed.emit()
    widget._canvas.boxes = [(1, 40, 30, 90, 70)]
    widget._canvas.boxes_changed.emit()
    widget._on_undo()
    saved = [list(box) for box in widget._canvas.boxes]
    assert saved == [[1, 40, 30, 90, 70], [1, 50, 40, 70, 60]]
    assert widget._on_save_boxes() is not None
    assert widget._on_export_yolo_boxes(str(folder / "field_0.txt"))
    widget._btn_next.click()
    _landed(qtbot, widget, second)
    assert widget._canvas.boxes == []
    negative = folder / "field_1.txt"
    assert widget._on_export_yolo_boxes(str(negative))
    assert negative.read_bytes() == b""
    widget._btn_prev.click()
    _landed(qtbot, widget, first)
    assert [list(box) for box in widget._canvas.boxes] == saved
    assert widget._box_classes == ["object", "demonstration"]
    assert not widget._boxes_dirty and widget.warnings == []
    stored = engine.load_yolo_boxes(folder, first, widget._canvas.image.shape)
    assert [list(box) for box in stored["boxes"]] == saved


def test_the_driver_waits_for_the_field_a_navigation_asked_for():
    """``ready`` takes the expected field, and both navigations pass one."""
    tree = ast.parse(DRIVER.read_text())
    tour = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "record_yolo_boxes")
    ready = next(node for node in ast.walk(tour)
                 if isinstance(node, ast.FunctionDef) and node.name == "ready")
    assert [arg.arg for arg in ready.args.args] == ["expected"]
    calls = [node for node in ast.walk(tour)
             if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ready"]
    assert sum(1 for call in calls if call.args) == 2

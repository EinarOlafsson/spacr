"""Time mode on a crop of the real T98G timelapse, as the maintainer traced it.

The fixture is fourteen frames (source frames 15 to 28) of the published
mammalian movie around one cell, in the CTC layout the 712 build reads: track
61 ends, its lone daughter 107 carries on (a founder after conversion) and
107 divides into 89 and 90; cell 69 sits right beside them. The movie is
built with ``tools/build_timelapse_example_datasets.py`` and opened through
the test-data opener, then traced with real Qt events.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from spacr.qt import make_masks_demo as demo
from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests" / "data" / "708_t98g_division_crop"


def _build(dest):
    spec = importlib.util.spec_from_file_location(
        "build_timelapse_example_datasets",
        ROOT / "tools" / "build_timelapse_example_datasets.py")
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)
    movie = dataclasses.replace(builder.movie("timelapse_mammalian"), window=(0, 14))
    summary = builder.build_movie(movie, SOURCE, dest)
    assert (summary["divisions"], summary["lone_daughters"]) == (1, 1)
    return dest / movie.folder


@pytest.fixture
def movie(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    folder = _build(tmp_path)
    screen = mm.MakeMasksScreen()
    monkeypatch.setattr(screen, "_confirm", lambda *args: True)
    qtbot.addWidget(screen)
    screen.resize(900, 700)
    screen.show()
    assert demo._open_timelapse(screen, folder)
    screen._canvas.resize(480, 680)
    screen._canvas.refresh()
    screen._btn_save_on_navigation.setChecked(False)
    yield screen, folder
    screen.close()


def idle(qtbot, screen):
    qtbot.waitUntil(lambda: not screen._loading and not screen._label_time_queue
                    and screen._label_time_pending is None, timeout=15000)


def _position(screen, point):
    return QPointF(screen._canvas._image_to_canvas(point[0] + .5, point[1] + .5))


def move(screen, point):
    pos = _position(screen, point)
    QApplication.sendEvent(screen._canvas, QMouseEvent(
        QEvent.MouseMove, pos, screen._canvas.mapToGlobal(pos.toPoint()),
        Qt.NoButton, Qt.NoButton, Qt.NoModifier))


def scroll(qtbot, screen, point, steps=1):
    pos = _position(screen, point)
    QApplication.sendEvent(screen._canvas, QWheelEvent(
        pos, screen._canvas.mapToGlobal(pos.toPoint()), QPoint(),
        QPoint(0, -120 * steps), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    idle(qtbot, screen)


def centre(mask, label):
    ys, xs = np.nonzero(mask == label)
    return int(round(xs.mean())), int(round(ys.mean()))


def owner(screen, point):
    label = int(screen._canvas.mask[point[1], point[0]])
    mapping = screen._lineage["frames"][screen._lineage_frame()]
    return label, screen._lineage["cells"][mapping[str(label)]]["name"]


def sweep(screen, start, end):
    for k in range(11):
        move(screen, (start[0] + (end[0] - start[0]) * k // 10,
                      start[1] + (end[1] - start[1]) * k // 10))


def test_typed_61_follows_its_cell_through_107_into_daughter_89(qtbot, movie):
    screen, folder = movie
    screen._on_next()
    idle(qtbot, screen)
    screen._btn_label_time.click()
    assert screen._time_label_enabled and screen._btn_label_time.isChecked()
    screen._label_value.clear()
    qtbot.keyClicks(screen._label_value, "61")
    assert screen._lineage["cells"][screen._selected_cell]["name"] == "61"
    chosen = screen._selected_cell
    canvas = screen._canvas
    neighbour = centre(canvas.mask, 69)
    point = centre(canvas.mask, 61)
    before = canvas.mask.copy()
    sweep(screen, neighbour, point)
    np.testing.assert_array_equal(canvas.mask, before)
    for frame in range(2, 12):
        scroll(qtbot, screen, point)
        assert screen._current_index == frame
        assert owner(screen, point) == (61, "61")
        point = centre(canvas.mask, 61)
    scroll(qtbot, screen, point)
    assert screen._current_index == 12
    mask = canvas.mask
    daughter, sister, neighbour = centre(mask, 89), centre(mask, 90), centre(mask, 69)
    sweep(screen, neighbour, daughter)
    assert owner(screen, neighbour) == (69, "69")
    assert owner(screen, daughter) == (89, "107a")
    scroll(qtbot, screen, daughter)
    assert screen._current_index == 13
    assert owner(screen, daughter) == (61, "61")
    assert owner(screen, centre(canvas.mask, 69)) == (69, "69")
    assert screen._selected_cell == chosen
    arrived = canvas.mask.copy()
    screen._on_undo()
    assert owner(screen, daughter) == (89, "107a")
    screen._on_redo()
    np.testing.assert_array_equal(canvas.mask, arrived)
    scroll(qtbot, screen, daughter, -1)
    assert screen._current_index == 12
    assert owner(screen, daughter) == (61, "61")
    assert owner(screen, sister)[0] == 90
    cells = screen._lineage["cells"]
    branch = screen._lineage["frames"][screen._lineage_frame()]["90"]
    assert cells[branch]["name"] == "107b"
    assert cells[cells[branch]["parent_uuid"]]["name"] == "107"
    screen._save_mask()
    for sidecar in Path(folder).rglob("*.json"):
        sidecar.unlink()
    fresh = mm.MakeMasksScreen()
    qtbot.addWidget(fresh)
    assert fresh._open_folder(str(folder), masks_dir=str(folder / "masks"))
    for index in (12, 13):
        name = f"t98g_t{index:03d}.tif"
        saved = imageio.imread(engine.mask_save_path(str(folder), name,
                                                     str(folder / "masks")))
        assert saved.dtype == np.uint32 and int(saved[daughter[1], daughter[0]]) == 61
        mapping = fresh._lineage["frames"][str(folder / name)]
        assert mapping["61"] == chosen and fresh._lineage["cells"][mapping["90"]]["name"] == "107b"
    fresh.close()


def test_picking_61_from_the_cell_table_traces_the_same_way(qtbot, movie):
    screen, _folder = movie
    screen._on_next()
    idle(qtbot, screen)
    screen._btn_label_time.click()
    model = screen._lineage_model
    row = next(index for index, key in enumerate(model.keys)
               if screen._lineage["cells"][key]["name"] == "61")
    screen._lineage_table.selectRow(
        screen._lineage_table.model().mapFromSource(model.index(row, 0)).row())
    assert screen._label_value.text() == "61"
    canvas = screen._canvas
    point = centre(canvas.mask, 61)
    for _frame in range(2, 13):
        scroll(qtbot, screen, point)
        point = centre(canvas.mask, 61) if np.any(canvas.mask == 61) else point
    daughter = centre(canvas.mask, 89)
    sweep(screen, centre(canvas.mask, 69), daughter)
    scroll(qtbot, screen, daughter)
    assert screen._current_index == 13
    assert owner(screen, daughter) == (61, "61")
    assert owner(screen, centre(canvas.mask, 69)) == (69, "69")

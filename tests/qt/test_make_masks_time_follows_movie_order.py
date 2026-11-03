"""Time-mode wheel steps follow the movie's time order, not the curation queue.

A queue built with ``--order prob`` or ``--order easy`` offers the frames by
score, and a resumed queue leaves the done frames out. A Time step must still
land on the adjacent time point and give the selected identity to that frame,
never to the next queue item.
"""
from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent

from spacr.curation_queue import build_queue, mark_state
from spacr.qt.screens import make_masks as mm

FRAMES = [f"t{index:03d}" for index in range(5)]


@pytest.fixture
def movie(tmp_path):
    folder = tmp_path / "movie"
    (folder / "masks").mkdir(parents=True)
    mask = np.zeros((40, 40), dtype=np.uint16)
    mask[5:15, 5:15] = 7
    mask[25:35, 25:35] = 11
    for index, stem in enumerate(FRAMES):
        imageio.imwrite(folder / f"{stem}.tif", np.full(mask.shape, index + 1, np.uint8))
        imageio.imwrite(folder / "masks" / f"{stem}.tif", mask)
    return folder


@pytest.fixture
def screen(qtbot, qt_theme_applied, monkeypatch):
    widget = mm.MakeMasksScreen()
    monkeypatch.setattr(widget, "_confirm", lambda *args: True)
    qtbot.addWidget(widget)
    yield widget
    widget.close()


def _idle(qtbot, screen):
    qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                    and screen._load_worker is None and screen._canvas.mask is not None
                    and not screen._label_time_queue
                    and screen._label_time_pending is None, timeout=20000)


def _open(qtbot, screen, queue):
    assert screen.open_queue(queue)
    _idle(qtbot, screen)
    screen._canvas.resize(400, 400)
    screen._canvas.refresh()
    screen._btn_label_time.setChecked(True)
    assert screen._canvas.mode == mm.MODE_LABEL_TIME
    screen._label_value.setText("7")


def _wheel(qtbot, screen, steps):
    canvas = screen._canvas
    pos = QPointF(canvas._image_to_canvas(10.5, 10.5))
    canvas.wheelEvent(QWheelEvent(pos, pos, QPoint(), QPoint(0, -120 * steps), Qt.NoButton,
                                  Qt.NoModifier, Qt.NoScrollPhase, False))
    _idle(qtbot, screen)


def _shown(screen):
    return Path(screen._image_files[screen._current_index]).stem


def _traced(screen, folder):
    identity = screen._selected_cell
    frames = screen._lineage["frames"]
    return sorted(Path(frame).stem for frame, mapping in frames.items()
                  if identity in mapping.values()
                  and os.path.dirname(frame) == os.path.abspath(folder))


@pytest.mark.parametrize("order", ["prob", "easy"])
def test_time_steps_follow_time_in_a_score_ordered_queue(qtbot, screen, movie, order):
    probs = {"t003": .9, "t000": .8, "t004": .7, "t001": .6, "t002": .5}
    counts = dict.fromkeys(FRAMES, 2)
    queue = build_queue(movie, order=order, probs=probs, counts=counts,
                        announce=lambda _message: None)
    assert [item.stem for item in queue.items] == ["t003", "t000", "t004", "t001", "t002"]
    _open(qtbot, screen, queue)
    assert _shown(screen) == "t003"
    _wheel(qtbot, screen, 1)
    assert _shown(screen) == "t004"
    _wheel(qtbot, screen, -2)
    assert _shown(screen) == "t002"
    assert _traced(screen, movie) == ["t002", "t003", "t004"]
    assert [Path(name).stem for name in screen._image_files] == [
        "t003", "t000", "t004", "t001", "t002"]


def test_time_steps_reach_done_frames_on_resume(qtbot, screen, movie):
    for stem in ("t001", "t002"):
        mark_state(movie, stem, "done")
    queue = build_queue(movie, order="name", announce=lambda _message: None)
    assert [item.stem for item in queue.items] == ["t000", "t003", "t004"]
    _open(qtbot, screen, queue)
    _wheel(qtbot, screen, 1)
    assert _shown(screen) == "t001"
    assert [Path(name).stem for name in screen._image_files] == [
        "t000", "t001", "t003", "t004"]
    screen._on_next()
    _idle(qtbot, screen)
    assert _shown(screen) == "t003"
    _wheel(qtbot, screen, -1)
    assert _shown(screen) == "t002"
    assert [Path(name).stem for name in screen._image_files] == FRAMES
    _wheel(qtbot, screen, 3)
    assert _shown(screen) == "t004"
    assert _traced(screen, movie) == FRAMES

"""A curation queue opened from Make Masks' folder dialog can mark a field skip.

The folder dialog used to open every folder as a plain one, so a queue that
already keeps a ``curate_status.csv`` lost its record: Skip was disabled and
a save recorded nothing. A folder with a record now opens as that queue, and
Skip writes exactly the row the queue's own command-line helper writes -- the
same ``spacr.curation_queue.mark_state`` call against the same file.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr import cli_make_masks
from spacr.curation_queue import (STATUS_FILENAME, build_queue, mark_state,
                                  read_status)
from spacr.qt.screens.make_masks import MakeMasksScreen


@pytest.fixture
def queue_folder(tmp_path: Path) -> Path:
    """Four fields under ``masks/``, ``f_00`` already done in the record."""
    folder = tmp_path / "queue"
    (folder / "masks").mkdir(parents=True)
    rng = np.random.default_rng(11)
    for index in range(4):
        imageio.imwrite(folder / f"f_{index:02d}.tif",
                        rng.integers(0, 4000, (24, 24), dtype=np.uint16))
    mark_state(folder, "f_00", "done", n_objects=3,
               updated="2026-11-01T10:00:00")
    return folder


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    cli_make_masks.hand_over(None)
    made = MakeMasksScreen()
    qtbot.addWidget(made)
    yield made
    cli_make_masks.hand_over(None)


def _settle(qtbot, screen):
    qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                    and screen._load_worker is None)


def test_skip_writes_the_row_the_queue_helper_writes(qtbot, screen,
                                                     queue_folder, tmp_path):
    twin = tmp_path / "twin"
    shutil.copytree(queue_folder, twin)

    assert screen._open_folder(str(queue_folder)) is True
    _settle(qtbot, screen)
    assert screen._queue is not None
    assert screen._image_files == ["f_00.tif", "f_01.tif", "f_02.tif",
                                   "f_03.tif"]
    assert screen._btn_skip.isEnabled()
    screen._current_index = screen._image_files.index("f_02.tif")
    screen._load_current()
    _settle(qtbot, screen)

    screen._btn_skip.click()

    rows = read_status(queue_folder)
    assert rows["f_02"].state == "skip"
    assert rows["f_00"].state == "done"
    mark_state(twin, "f_02", "skip", updated=rows["f_02"].updated)
    assert ((queue_folder / STATUS_FILENAME).read_bytes()
            == (twin / STATUS_FILENAME).read_bytes())
    assert not (queue_folder / "masks" / "f_02.tif").exists()
    assert [item.stem for item in build_queue(queue_folder, order="name").items] \
        == ["f_01", "f_03"]


def test_a_folder_without_a_record_still_opens_plain(qtbot, screen, tmp_path):
    folder = tmp_path / "plain"
    folder.mkdir()
    imageio.imwrite(folder / "a.tif", np.zeros((16, 16), dtype=np.uint16))

    assert screen._open_folder(str(folder)) is True
    _settle(qtbot, screen)

    assert screen._queue is None
    assert not screen._btn_skip.isEnabled()
    assert not (folder / STATUS_FILENAME).exists()


def test_an_unreadable_record_falls_back_to_a_plain_folder(qtbot, screen,
                                                           queue_folder):
    (queue_folder / STATUS_FILENAME).write_text("not,a,status\n1,2,3\n",
                                                encoding="utf-8")

    assert screen._open_folder(str(queue_folder)) is True
    _settle(qtbot, screen)

    assert screen._queue is None
    assert len(screen._image_files) == 4

"""Item 448: the Keep and Discard buttons in Make Masks.

The CSV itself is tested at the engine in
tests/test_a_field_is_kept_or_discarded_and_the_csv_says_so.py. What is
held here is the part only the screen can answer: that the buttons write
the field that is open, count the objects as they stand at the press, and
show a field's existing verdict when it is opened again.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm

IMG_N = 64


@pytest.fixture
def two_fields(tmp_path: Path) -> Path:
    """Two fields, each with a mask holding a known number of objects."""
    folder = tmp_path / "images"
    (folder / "masks").mkdir(parents=True)
    for name, objects in (("a.tif", 2), ("b.tif", 1)):
        imageio.imwrite(folder / name,
                        np.zeros((IMG_N, IMG_N), dtype=np.uint16))
        mask = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
        for index in range(objects):
            mask[4 + index * 8:8 + index * 8, 4:8] = index + 1
        imageio.imwrite(folder / "masks" / name, mask)
    return folder


@pytest.fixture
def screen(qtbot, qt_theme_applied, two_fields: Path):
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    assert made._open_folder(str(two_fields))
    yield made
    made._magnifier.close()
    made.close_folded()


def _rows(folder):
    with open(engine.curation_csv_path(str(folder)), encoding="utf-8",
              newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_two_buttons_are_there(screen):
    assert screen._btn_keep.text() == "Keep"
    assert screen._btn_discard.text() == "Discard"


def test_keep_writes_the_field_that_is_open(screen, two_fields):
    screen._btn_keep.click()
    rows = _rows(two_fields)
    assert len(rows) == 1
    assert rows[0]["image path"] == os.path.join(str(two_fields), "a.tif")
    assert rows[0]["mask path"] == os.path.join(str(two_fields), "masks",
                                                "a.tif")
    assert rows[0]["keep"] == "true"
    assert rows[0]["object count"] == "2"


def test_discard_writes_false_and_deletes_nothing(screen, two_fields):
    screen._btn_discard.click()
    assert _rows(two_fields)[0]["keep"] == "false"
    assert (two_fields / "a.tif").is_file()
    assert (two_fields / "masks" / "a.tif").is_file()


def test_the_count_is_read_at_the_press_not_at_the_load(screen, two_fields):
    """A field curated after hand editing records what the user saw."""
    assert screen._objects_now() == 2
    mask = np.asarray(screen._canvas.mask).copy()
    mask[mask == 2] = 0
    screen._canvas.mask = mask
    assert screen._objects_now() == 1
    screen._btn_keep.click()
    assert _rows(two_fields)[0]["object count"] == "1"


def test_a_field_opened_again_shows_the_verdict_it_has(screen, qtbot):
    screen._btn_discard.click()
    assert screen._btn_discard.isChecked()
    assert not screen._btn_keep.isChecked()
    screen._on_next()
    qtbot.waitUntil(lambda: not screen._loading, timeout=5000)
    assert not screen._btn_keep.isChecked()
    assert not screen._btn_discard.isChecked(), (
        "the second field has no verdict and must not wear the first's")
    screen._on_prev()
    qtbot.waitUntil(lambda: not screen._loading, timeout=5000)
    assert screen._btn_discard.isChecked()


def test_changing_the_verdict_replaces_the_row(screen, two_fields):
    screen._btn_keep.click()
    screen._btn_discard.click()
    rows = _rows(two_fields)
    assert len(rows) == 1 and rows[0]["keep"] == "false"
    assert screen._btn_discard.isChecked()
    assert not screen._btn_keep.isChecked()

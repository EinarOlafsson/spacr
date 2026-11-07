"""Annotate's touching-pair review writes one merge verdict per pair.

The set it builds is an ordinary Annotate source: pair PNGs under
``data/merge_pairs`` and a ``png_list`` whose ``merge`` column holds 1 (one
object cut in two), 2 (two objects) or 3 (cannot tell). These tests read the
database the way a labelling check script reads it -- the basename of
``png_path`` against the verdict -- so the rows themselves are the outcome.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tifffile                                                    # noqa: E402
from PIL import Image                                              # noqa: E402
from PySide6.QtCore import Qt                                      # noqa: E402
from PySide6.QtTest import QTest                                   # noqa: E402

from spacr import _merge_pairs as pairs                            # noqa: E402
from spacr.qt.screens import annotate as annotate_mod              # noqa: E402


def _field() -> np.ndarray:
    """Labels 1|2 cut down the middle, 3 and 4 side by side, 5 on the edge."""
    mask = np.zeros((160, 160), dtype=np.uint16)
    mask[30:70, 30:50] = 1
    mask[30:70, 50:70] = 2
    mask[100:130, 90:100] = 3
    mask[100:130, 100:104] = 4
    mask[0:20, 120:140] = 5
    mask[0:20, 140:150] = 6
    return mask


@pytest.fixture
def stacks(tmp_path):
    """Two fields stored planes-first as uint16 TIFFs: three image planes, then labels."""
    folder = tmp_path / "masks"
    folder.mkdir()
    rng = np.random.default_rng(3)
    for stem in ("P1_A01_1", "P1_B02_4"):
        image = rng.integers(100, 4000, size=(3, 160, 160)).astype(np.uint16)
        tifffile.imwrite(folder / f"{stem}.tif",
                         np.concatenate([image, _field()[None]], 0),
                         photometric="minisblack")
    return folder


def _verdicts(db_path) -> dict:
    """``{png basename: merge}``, read as a labelling check script reads it."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        return {os.path.basename(p): v for p, v in con.execute(
            'SELECT png_path, "merge" FROM png_list')}
    finally:
        con.close()


def _dialog(qtbot, stacks, tmp_path):
    dialog = annotate_mod._PairReviewDialog()
    qtbot.addWidget(dialog)
    dialog._masks.setText(str(stacks))
    dialog._images.setText(str(stacks))
    dialog._mask_plane.setValue(-1)
    dialog._object.setText("cell")
    dialog._target.setText(str(tmp_path / "set"))
    return dialog


def test_keys_write_one_verdict_row_per_pair(qtbot, stacks, tmp_path):
    dialog = _dialog(qtbot, stacks, tmp_path)

    assert dialog.build_set() == 4

    db = tmp_path / "set" / "measurements" / "measurements.db"
    assert sorted(_verdicts(db)) == [
        "P1_A01_1_cell1-2.png", "P1_A01_1_cell3-4.png",
        "P1_B02_4_cell1-2.png", "P1_B02_4_cell3-4.png"]
    order = [row["file_name"] for row in dialog._pairs]
    dialog.show()
    dialog.activateWindow()
    QTest.keyClick(dialog, Qt.Key_1)
    QTest.keyClick(dialog, Qt.Key_2)
    QTest.keyClick(dialog, Qt.Key_3)

    assert _verdicts(db) == {order[0]: 1, order[1]: 2, order[2]: 3,
                             order[3]: None}
    assert dialog._index == 3


def test_zero_clears_and_the_arrows_move(qtbot, stacks, tmp_path):
    dialog = _dialog(qtbot, stacks, tmp_path)
    dialog.build_set()
    db = tmp_path / "set" / "measurements" / "measurements.db"
    first = dialog._pairs[0]["file_name"]
    dialog.record(1)
    QTest.keyClick(dialog, Qt.Key_Left)
    assert dialog._index == 0

    QTest.keyClick(dialog, Qt.Key_0)

    assert _verdicts(db)[first] is None
    assert dialog._index == 0
    QTest.keyClick(dialog, Qt.Key_Right)
    assert dialog._index == 1


def test_a_set_is_reopened_not_rebuilt_and_keeps_its_verdicts(qtbot, stacks,
                                                              tmp_path):
    dialog = _dialog(qtbot, stacks, tmp_path)
    dialog.build_set()
    dialog.record(2)
    judged = dialog._pairs[0]["file_name"]

    again = _dialog(qtbot, stacks, tmp_path)
    assert again.build_set() == 4

    db = tmp_path / "set" / "measurements" / "measurements.db"
    assert _verdicts(db)[judged] == 2
    assert again._index == 1, "a reopened set starts at its first open pair"
    assert "already holds a pair set" in again._status.text()


def test_the_pictures_outline_a_white_and_b_yellow(stacks, tmp_path):
    pairs._build_set(str(stacks), str(tmp_path / "set"), mask_plane=-1,
                     image_folder=str(stacks))
    picture = np.asarray(Image.open(
        tmp_path / "set" / "data" / "merge_pairs" / "P1_A01_1_cell1-2.png"))
    colours = {tuple(px) for px in picture.reshape(-1, 3)}

    assert pairs.COLOUR_A in colours and pairs.COLOUR_B in colours
    assert picture.shape[0] >= 96 and picture.shape[1] >= 96


def test_pairs_on_the_field_edge_are_left_out_and_the_rule_is_recorded(
        stacks, tmp_path):
    pairs._build_set(str(stacks), str(tmp_path / "set"), mask_plane=-1)
    rows = {row["file_name"]: row for row in pairs._read_set(str(tmp_path / "set"))}

    assert not any("cell5-6" in name for name in rows)
    halves = rows["P1_A01_1_cell1-2.png"]
    assert halves["merged_by_default_rule"] == 1
    assert halves["source_image"] == ""
    assert (halves["label_a"], halves["label_b"]) == (1, 2)


def test_a_plain_label_folder_and_a_limit(tmp_path):
    folder = tmp_path / "labels"
    folder.mkdir()
    np.save(folder / "f1.npy", _field())
    Image.fromarray(_field().astype(np.uint8)).save(folder / "f2.png")

    report = pairs._build_set(str(folder), str(tmp_path / "set"),
                              object_name="nucleus", limit=3)

    assert report["pairs"] == 3
    names = sorted(_verdicts(report["db"]))
    assert names == ["f1_nucleus1-2.png", "f1_nucleus3-4.png",
                     "f2_nucleus1-2.png"]


def test_an_empty_folder_says_why_nothing_was_built(qtbot, tmp_path):
    (tmp_path / "empty").mkdir()
    dialog = annotate_mod._PairReviewDialog()
    qtbot.addWidget(dialog)
    dialog._masks.setText(str(tmp_path / "empty"))
    dialog._target.setText(str(tmp_path / "set"))

    assert dialog.build_set() == 0
    assert "no label images" in dialog._status.text()
    assert not dialog._verdict_buttons[1].isEnabled()
    assert dialog.record(1) is False


def test_a_verdict_outside_the_three_is_refused(stacks, tmp_path):
    pairs._build_set(str(stacks), str(tmp_path / "set"), mask_plane=-1)
    row = pairs._read_set(str(tmp_path / "set"))[0]

    with pytest.raises(ValueError):
        pairs._write_verdict(str(tmp_path / "set"), row["png_path"], 7)


def test_a_wrong_plane_is_reported(qtbot, stacks, tmp_path):
    dialog = _dialog(qtbot, stacks, tmp_path)
    dialog._mask_plane.setValue(9)

    assert dialog.build_set() == 0
    assert "plane 9" in dialog._status.text()


def test_the_setup_menu_opens_the_review_on_a_set_source(qtbot, stacks,
                                                         tmp_path):
    pairs._build_set(str(stacks), str(tmp_path / "set"), mask_plane=-1)
    screen = annotate_mod.AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.src = str(tmp_path / "set")

    dialog = screen._on_pair_review()

    assert len(dialog._pairs) == 4
    assert dialog._caption.text().startswith("1/4")
    assert not dialog._image.pixmap().isNull()
    dialog.close()

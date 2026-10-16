"""Make Masks' 3D volume view, MitoNet and histogram window at their guards.

Each test drives a path the coverage gate named on ``make_masks.py`` after
N692 and item 691 (MitoNet): a stroke or edit that arrives while a 3-D edit
runs, a stroke that changes nothing, a shape the True 3D view cannot
voxelize, a volume that refuses to be left, a MitoNet run without a ticket
or inversion, a model fetched this session and a magnifier drag without a
usable start point.

2026-10-12: the separate 3D mask editor window (its Draw slices and 3D
anchors tabs) was retired; a volume opens in the Make Masks view. The
volume guards below now drive that view through the screen.

No GPU and no network. Waits are condition-based with generous timeouts so a
heavily loaded machine slows the tests down rather than failing them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
from shiboken6 import delete

from spacr import model_zoo as mz
from spacr.qt.screens import make_masks as mm
from spacr.qt.screens.make_masks import MakeMasksScreen

WAIT = 60000


def _settle(rounds=4):
    for _ in range(rounds):
        QApplication.processEvents()


@pytest.fixture
def discard(monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Discard)


@pytest.fixture
def screen(qtbot, discard):
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    yield widget
    stage = widget._mask_stage
    qtbot.waitUntil(lambda: stage._worker.idle() and stage._mesher.idle(), timeout=WAIT)
    stage.dirty = False


def _stack(tmp_path, name="stack.tif"):
    image = np.random.default_rng(5).integers(0, 255, (6, 12, 12), dtype=np.uint8)
    tifffile.imwrite(tmp_path / name, image)
    return tmp_path / name


@pytest.fixture
def slices(screen, tmp_path):
    """A 6 x 12 x 12 stack open in the Make Masks view."""
    assert screen.open_volume(str(_stack(tmp_path))) is screen._mask_stage
    _settle()
    return screen._mask_stage


def test_a_stroke_that_changes_no_pixel_makes_no_undo_step(slices, screen):
    screen._canvas.stroke_finished.emit()
    _settle(2)
    assert not slices.session._undo and not slices.dirty
    assert not screen._act_undo.isEnabled()


def test_open_volume_asks_for_a_file_and_a_cancelled_picker_opens_nothing(screen, tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *args, **kwargs: (asked.append(args), ("", ""))[1])
    assert screen.open_volume() is None
    assert asked and screen._mask_stage.session is None
    path = _stack(tmp_path)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(path), ""))
    stage = screen.open_volume()
    assert stage is screen._mask_stage
    assert stage.session.image.shape == (6, 12, 12)


def test_a_flat_image_is_refused_on_the_status_line(screen, tmp_path):
    tifffile.imwrite(tmp_path / "flat.tif", np.zeros((12, 12), dtype=np.uint8))
    assert screen.open_volume(str(tmp_path / "flat.tif")) is None
    assert screen._mask_stage.session is None
    assert screen._status_label.text()


def test_a_save_after_the_source_changed_is_refused_and_stays_dirty(slices, screen):
    slices.session.paint_plane("Z", 0, 1, paint=np.ones((12, 12), dtype=bool))
    slices.dirty = True
    source = Path(slices.session.source_path)
    tifffile.imwrite(source, np.zeros((6, 12, 12), dtype=np.uint8))
    assert slices.save() is False
    assert "source volume changed" in screen._status_label.text()
    assert slices.dirty and not Path(slices.session.mask_path).exists()


def test_mitonet_without_inversion_or_ticket_gets_the_raw_crop(monkeypatch):
    seen = {}

    def fake(image, **kwargs):
        seen["image"] = np.asarray(image)
        seen.update(kwargs)
        kwargs["on_tile"](1, 2)
        kwargs["on_tile"](2, 2)
        return np.ones(np.asarray(image).shape, np.int32)

    monkeypatch.setattr(mz, "segment_mitonet", fake)
    crop = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    request = mm._MagnifierRequest(
        key=(), crop=crop, box=(0, 0, 8, 8), shape=(8, 8), mode="mitonet",
        sensitivity=1.0, bright=True, min_area=0, model_name="cpsam",
        diameter=30, colour=(0, 0, 0), invert=False, ticket=None)
    labels = mm._mitonet_segmenter(request)
    assert labels.shape == (8, 8)
    np.testing.assert_array_equal(seen["image"], crop)
    assert seen["min_size"] == 1


def test_the_histogram_window_resizes_after_its_layout_is_gone(qtbot):
    dialog = mm._OtsuHistogramDialog(np.array([1., 3., 2.]), np.array([0., 1., 2., 3.]),
                                     [1.5], "Otsu")
    qtbot.addWidget(dialog)
    dialog.show()
    _settle()
    minimum = dialog.minimumHeight()
    delete(dialog.layout())
    assert dialog.layout() is None
    dialog.resize(dialog.width() + 80, dialog.height() + 40)
    QApplication.sendPostedEvents(dialog, QEvent.Resize)
    _settle()
    assert dialog.minimumHeight() == minimum


def test_a_model_fetched_this_session_is_listed_by_where_it_landed(screen, tmp_path, monkeypatch):
    weights = tmp_path / "landed" / "zoo_cp"
    weights.parent.mkdir()
    weights.write_bytes(b"weights")
    entry = mz.ModelEntry(key="zoo_cp", name="zoo_cp", kind="cellpose", source="remote",
                          uri="file:///nowhere/zoo_cp", sha256="0" * 64, size_bytes=7)
    monkeypatch.setattr(mm, "_zoo_cellpose_models", lambda: [("zoo_cp", None, entry)])
    screen._cp_fetched["zoo_cp"] = str(weights)
    screen._fill_zoo_models()
    box = screen._cp_model
    row = box.findData(str(weights))
    assert row >= 0 and box.itemText(row) == "zoo_cp"
    assert box.itemData(row, mm._ZOO_PENDING_ROLE) is None


def _drag(screen, outcome):
    screen._magnifier.drag_ready.emit((outcome, True))
    _settle(2)


@pytest.fixture
def field(screen):
    image = np.random.default_rng(3).integers(0, 4000, (64, 64), dtype=np.uint16)
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[2:14, 2:14] = 4
    screen._canvas.set_image_and_mask(image, mask)
    return screen


def _stroke_labels():
    labels = np.zeros((14, 14), dtype=np.int32)
    labels[1:13, 1:13] = 1
    return labels


def test_a_drag_outcome_without_provenance_pastes_a_new_object(field):
    from spacr.qt._magnifier_drag import _StrokeOutcome

    assert field._detect_min_area() <= 144
    _drag(field, _StrokeOutcome(_stroke_labels(), (30, 30), 1, 1, 1))
    mask = field._canvas.mask
    assert set(np.unique(mask[31:43, 31:43])) == {5}
    assert np.all(mask[2:14, 2:14] == 4)


def test_a_drag_starting_off_the_field_does_not_extend_an_object(field):
    from spacr.qt._magnifier_drag import _StrokeOutcome

    _drag(field, _StrokeOutcome(_stroke_labels(), (30, 30), 1, 1, 1,
                                provenance={"path": [[-3, 90], [31, 31]]}))
    mask = field._canvas.mask
    assert set(np.unique(mask[31:43, 31:43])) == {5}
    assert np.count_nonzero(mask == 4) == 144

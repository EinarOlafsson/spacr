"""A secondary mask's pairing with its primary is checked at every step.

Pins three guards around growing secondary objects from a primary mask in
Make Masks:

* a primary snapshot that belongs to another field is refused before any
  object is grown from it;
* once the primary has changed, a later edit is still recorded as paired
  with the primary it was grown from, without claiming the new primary's
  object relationships;
* a save onto the file the pairing was grown from is refused even after the
  primary selection has been cleared, and that file is left untouched.
"""
from __future__ import annotations

import dataclasses
import os

import imageio.v3 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QPointF

from spacr.qt import cpu_modes
from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm


@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    images, primaries = tmp_path / "images", tmp_path / "primary"
    images.mkdir()
    primaries.mkdir()
    field = np.zeros((96, 96), np.uint16)
    field[16:80, 16:80] = 200
    primary = np.zeros_like(field)
    primary[42:54, 42:54] = 900
    field[primary > 0] = 0
    for name in ("a.tif", "b.tif"):
        imageio.imwrite(images / name, field)
        imageio.imwrite(primaries / name, primary)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    made.warnings = []
    monkeypatch.setattr(made, "_warn",
                        lambda *args: made.warnings.append(args))
    assert made._open_folder(str(images))
    made._min_area.setValue(1)
    made._detect_normalized.setChecked(False)
    made._primary_selector.path.setText(str(primaries))
    made._primary_selector._source_changed()
    qtbot.waitUntil(lambda: made._primary_selector.snapshot is not None)
    made._mag_mode.setCurrentIndex(
        made._mag_mode.findData(cpu_modes.SECONDARY))
    made._secondary_widgets["propagate_sigma"].setValue(0)
    made._secondary_widgets["propagate_stop"].setCurrentIndex(
        made._secondary_widgets["propagate_stop"].findData("absolute"))
    made._secondary_widgets["propagate_stop_value"].setValue(100)
    made._combine_mode.setCurrentIndex(made._combine_mode.findData("replace"))
    made._magnifier.size = 96
    made._magnifier._cursor = (48, 48)
    made._magnifier._anchor = QPointF(48, 48)
    made._magnifier.enabled = True
    made._magnifier.exclude_border = False
    monkeypatch.setattr(made._magnifier._worker, "submit",
                        lambda req, **kw: None)
    monkeypatch.setattr(made._magnifier._image_worker, "submit",
                        lambda req: None)
    yield made
    made.close()


def test_a_primary_of_another_field_is_refused(screen, tmp_path):
    source = screen._primary_selector.snapshot
    other = os.path.realpath(str(tmp_path / "images" / "b.tif"))
    screen._primary_selector.snapshot = dataclasses.replace(
        source, image_path=other)
    before = screen._canvas.mask.copy()
    screen._on_detect_otsu()
    assert screen.warnings[-1] == (
        "Detect failed",
        "The primary mask belongs to a different field. Reload it for "
        "this image.")
    np.testing.assert_array_equal(screen._canvas.mask, before)


def test_an_edit_after_the_primary_changed_keeps_the_old_pairing(
        screen, qtbot):
    screen._on_detect_otsu()
    assert not screen.warnings
    paired = dict(screen._paired_source)
    screen._primary_selector.primary_class.setEditText("different primary")
    qtbot.waitUntil(lambda: screen._primary_selector.snapshot is not None)
    assert screen._primary_selector.snapshot.provenance()["primary_class"] \
        == "different primary"
    screen._grow_step.setValue(1)
    screen._on_dilate()
    detail = screen._log.edits[-1].detail
    assert detail["preserve_ids"] is True
    assert detail["primary_source"] == paired
    assert "primary_secondary" not in detail


def test_a_save_onto_the_paired_primary_is_refused_after_it_was_cleared(
        screen, qtbot, monkeypatch):
    screen._on_detect_otsu()
    primary_path = screen._paired_source["path"]
    before = open(primary_path, "rb").read()
    screen._primary_selector.path.setText("")
    screen._primary_selector._source_changed()
    assert screen._primary_selector.snapshot is None
    monkeypatch.setattr(engine, "mask_save_path",
                        lambda *a, **k: primary_path)
    screen._on_save()
    assert screen.warnings[-1] == (
        "Save failed",
        "Primary and secondary masks must be saved to different files.")
    assert open(primary_path, "rb").read() == before

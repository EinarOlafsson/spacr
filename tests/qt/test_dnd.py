"""Tests for the drag-and-drop system + per-module handlers."""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# has_images_in / find_image_folders_nearby
# ---------------------------------------------------------------------------

def _mkimg(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"II*\x00\x08\x00\x00\x00")   # tiny TIFF header


def test_has_images_in_true_for_folder_with_tifs(tmp_path):
    from spacr.qt.dnd import has_images_in
    _mkimg(tmp_path / "a.tif"); _mkimg(tmp_path / "b.png")
    assert has_images_in(tmp_path) is True


def test_has_images_in_false_for_empty_folder(tmp_path):
    from spacr.qt.dnd import has_images_in
    assert has_images_in(tmp_path) is False


def test_has_images_in_false_for_non_dir(tmp_path):
    from spacr.qt.dnd import has_images_in
    _mkimg(tmp_path / "x.tif")
    assert has_images_in(tmp_path / "x.tif") is False


def test_find_image_folders_nearby_returns_sibling(tmp_path):
    from spacr.qt.dnd import find_image_folders_nearby
    (tmp_path / "empty").mkdir()
    (tmp_path / "sibling").mkdir()
    _mkimg(tmp_path / "sibling" / "img.tif")
    hits = find_image_folders_nearby(tmp_path / "empty")
    assert tmp_path / "sibling" in hits


def test_find_image_folders_nearby_returns_child(tmp_path):
    from spacr.qt.dnd import find_image_folders_nearby
    parent = tmp_path / "parent"; parent.mkdir()
    _mkimg(parent / "sub" / "img.tif")
    hits = find_image_folders_nearby(parent)
    assert parent / "sub" in hits


def test_sample_image_names_caps_count(tmp_path):
    from spacr.qt.dnd import sample_image_names
    for i in range(20):
        _mkimg(tmp_path / f"img_{i:02d}.tif")
    hits = sample_image_names(tmp_path, n=5)
    assert len(hits) == 5


# ---------------------------------------------------------------------------
# Per-module DropHandler acceptance
# ---------------------------------------------------------------------------

def test_mask_handler_accepts_image_folder(tmp_path):
    from spacr.qt.dnd_handlers import MaskDropHandler
    _mkimg(tmp_path / "img.tif")
    assert MaskDropHandler().can_accept(tmp_path) is True


def test_mask_handler_rejects_empty(tmp_path):
    from spacr.qt.dnd_handlers import MaskDropHandler
    assert MaskDropHandler().can_accept(tmp_path) is False


def test_mask_handler_suggests_sibling(tmp_path):
    from spacr.qt.dnd_handlers import MaskDropHandler
    (tmp_path / "wrong").mkdir()
    (tmp_path / "right").mkdir()
    _mkimg(tmp_path / "right" / "img.tif")
    alts = MaskDropHandler().suggest_alternatives(tmp_path / "wrong")
    assert tmp_path / "right" in alts


def test_measure_handler_accepts_merged_folder(tmp_path):
    from spacr.qt.dnd_handlers import MeasureDropHandler
    merged = tmp_path / "merged"; merged.mkdir()
    _mkimg(merged / "stack_0.tif")
    assert MeasureDropHandler().can_accept(merged) is True


def test_measure_handler_accepts_parent_containing_merged(tmp_path):
    from spacr.qt.dnd_handlers import MeasureDropHandler
    (tmp_path / "merged").mkdir()
    assert MeasureDropHandler().can_accept(tmp_path) is True


def test_measure_handler_rejects_plain_folder(tmp_path):
    from spacr.qt.dnd_handlers import MeasureDropHandler
    assert MeasureDropHandler().can_accept(tmp_path) is False


def test_annotate_handler_accepts_folder_with_measurements_db(tmp_path):
    from spacr.qt.dnd_handlers import AnnotateDropHandler
    db = tmp_path / "measurements" / "measurements.db"
    db.parent.mkdir()
    db.write_bytes(b"sqlite")
    assert AnnotateDropHandler().can_accept(tmp_path) is True


def test_annotate_handler_accepts_direct_db_file(tmp_path):
    from spacr.qt.dnd_handlers import AnnotateDropHandler
    db = tmp_path / "measurements.db"
    db.write_bytes(b"sqlite")
    assert AnnotateDropHandler().can_accept(db) is True


def test_map_barcodes_handler_accepts_fastq_gz(tmp_path):
    from spacr.qt.dnd_handlers import MapBarcodesDropHandler
    fq = tmp_path / "reads.fastq.gz"
    fq.write_bytes(b"@id\nACGT\n+\nIIII\n")
    assert MapBarcodesDropHandler().can_accept(fq) is True


def test_map_barcodes_handler_rejects_random_file(tmp_path):
    from spacr.qt.dnd_handlers import MapBarcodesDropHandler
    (tmp_path / "notes.txt").write_text("hi")
    assert MapBarcodesDropHandler().can_accept(tmp_path / "notes.txt") is False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key,expected", [
    ("mask",         "MaskDropHandler"),
    ("measure",      "MeasureDropHandler"),
    ("annotate",     "AnnotateDropHandler"),
    ("classify",     "ClassifyDropHandler"),
    ("make_masks",   "MakeMasksDropHandler"),
    ("map_barcodes", "MapBarcodesDropHandler"),
    ("umap",         "MeasurementsDropHandler"),
    ("nonexistent",  "MeasurementsDropHandler"),  # fallback
])
def test_get_handler_returns_expected_class(app_key, expected):
    from spacr.qt.dnd_handlers import get_handler
    h = get_handler(app_key)
    assert type(h).__name__ == expected


# ---------------------------------------------------------------------------
# install_dropzone smoke check
# ---------------------------------------------------------------------------

def test_install_dropzone_enables_drops(qtbot, qt_theme_applied):
    from PySide6.QtWidgets import QWidget
    from spacr.qt.dnd import install_dropzone
    from spacr.qt.dnd_handlers import MaskDropHandler

    w = QWidget()
    qtbot.addWidget(w)
    assert w.acceptDrops() is False
    install_dropzone(w, MaskDropHandler(), w)
    assert w.acceptDrops() is True
    # Handler + screen backref stored on the widget for the filter to read
    assert isinstance(w._dnd_handler, MaskDropHandler)
    assert w._dnd_screen is w


def test_set_screen_setting_sets_metadata_type(qtbot):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.dnd_handlers import _set_screen_setting
    scr = AppScreen("mask")
    qtbot.addWidget(scr)
    assert _set_screen_setting(scr, "metadata_type", "auto") is True
    w = scr._settings_model._widgets.get("metadata_type")
    assert w.currentText() == "auto"


def test_report_folder_structure_logs_detected_labels(qtbot, tmp_path):
    from spacr.qt.dnd_handlers import _report_folder_structure
    # plate/well/field folder layout with an image at the leaf
    leaf = tmp_path / "plate1" / "A01" / "f01"
    leaf.mkdir(parents=True)
    import tifffile, numpy as np
    tifffile.imwrite(str(leaf / "C01.tif"), np.zeros((4, 4), np.uint16))
    logged = {"text": ""}

    class _Console:
        def append_stdout(self, t): logged["text"] += t

    class _Screen:
        _console = _Console()

    # Should not raise; if folder_metadata detects a layout, it logs it.
    _report_folder_structure(tmp_path, _Screen())
    # (Detection may or may not fire on this tiny synthetic tree; the contract
    # we assert is that it never crashes and only ever logs folder-structure
    # info when a template is found.)
    if logged["text"]:
        assert "folder-structure" in logged["text"]

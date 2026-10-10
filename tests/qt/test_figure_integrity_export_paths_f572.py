"""Figure integrity (F572) on the GUI's own export paths.

The right-click picture save and the cell montage save check what they
write when the guard is on, and neither ever fails a save because of the
check: a picture too small to check, a check that raises, a sidecar that
cannot be written, and crop sources that vanish or change after loading all
leave the saved file in place.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from spacr import plot  # noqa: E402
from spacr.qt.widgets import cell_montage_view as montage  # noqa: E402
from spacr.qt.widgets import picture_export  # noqa: E402

from .test_cells_behind_the_dot_tab import GENE_KEY, _view  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _checked(monkeypatch):
    monkeypatch.setenv(plot._INTEGRITY_ENV, "1")
    monkeypatch.delenv(plot._INDEX_ENV, raising=False)


def _picture(width, height):
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(40, 90, 160))
    return pixmap


def test_a_picture_too_small_to_check_is_saved_without_a_sidecar(app, tmp_path):
    path = tmp_path / "tiny.png"
    assert picture_export.save_picture(_picture(8, 8), str(path)) is True
    assert path.exists()
    assert not (tmp_path / "tiny.png.provenance.json").exists()


def test_a_failing_check_still_saves_the_picture(app, tmp_path, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("check failed")

    monkeypatch.setattr(plot, "_integrity_report", broken)
    path = tmp_path / "field.png"
    assert picture_export.save_picture(_picture(64, 64), str(path)) is True
    assert QImage(str(path)).text(plot._PNG_PROVENANCE_KEY) == ""


def test_a_sidecar_that_cannot_be_written_does_not_fail_the_save(
        app, tmp_path, monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(plot, "_finish_integrity", broken)
    path = tmp_path / "field.pdf"
    assert picture_export.save_picture(_picture(64, 64), str(path)) is True
    assert path.exists()


def test_cut_records_only_png_and_merged_sources_that_stay_put(tmp_path):
    from PIL import Image

    from spacr.crops import read_crop_png

    path = tmp_path / "crop.png"
    Image.fromarray(np.full((24, 28, 3), 32, dtype=np.uint8)).save(path)

    class Other:
        kind = "other"

        def get_many(self, rows):
            return [np.zeros((4, 4))] * len(rows)

    class Vanishing:
        kind = "png"
        db_path = None

        def resolve(self, _row):
            return str(path)

        def get_many(self, rows):
            pixels = [read_crop_png(str(path)) for _row in rows]
            os.remove(path)
            return pixels

    plan = SimpleNamespace(rows=lambda: [{"montage_source_root": "a"},
                                         {"montage_source_root": "b"}])
    records = [None, None]
    crops = montage._cut(plan, {"a": SimpleNamespace(source=Other()),
                                "b": SimpleNamespace(source=Vanishing())},
                         None, [], provenance=records)
    assert len(crops) == 2 and records == [None, None]
    crops = montage._cut(plan, {"a": SimpleNamespace(source=Other())}, None,
                         [])
    assert crops[0] is not None and crops[1] is None


def test_montage_save_tags_only_crops_whose_source_is_unchanged(
        qtbot, tmp_path):
    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=True)
    view.set_coefficient(GENE_KEY)
    view.build()
    images = [list(crops) for crops in view._images]
    records = [list(found) for found in view._crop_sources]
    images[0][1] = None
    records[0][2] = None
    records[0][3] = dict(records[0][3], path=str(tmp_path / "gone.png"))
    view._images = tuple(tuple(crops) for crops in images)
    view._crop_sources = tuple(tuple(found) for found in records)
    written = view.save(str(tmp_path / "montage.png"))
    with open(written + ".provenance.json", encoding="utf-8") as handle:
        report = json.load(handle)
    traced = [panel for panel in report["panels"] if panel["source"]]
    assert len(traced) == sum(crop is not None and record is not None
                              and record["path"] != str(tmp_path / "gone.png")
                              for crop, record in zip(images[0], records[0]))

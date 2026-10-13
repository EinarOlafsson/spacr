"""Branches in simulation sizing, crop paths, LIMS paging, the layer viewer and the dock."""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QHBoxLayout, QWidget  # noqa: E402


def test_simulation_sizing_falls_back_on_unreadable_sweeps():
    from spacr.sim import _simulation_unit_bytes

    assert _simulation_unit_bytes({"nr_plates": ["two"]}) == int(
        1 * 384 * 100 * (2 + 8) * 8)

    class _Broken(dict):
        def get(self, key, default=None):
            raise RuntimeError("no settings")

    assert _simulation_unit_bytes(_Broken()) == 0


def _objects(**extra):
    return pd.DataFrame({"plateID": ["p1"], "rowID": ["r1"], "columnID": ["c1"],
                         "fieldID": ["f1"], "object_label": [1], **extra})


def test_rows_that_already_have_crops_are_not_looked_up(tmp_path):
    from spacr.png_list import _attach_object_crop_paths

    frame = _objects(png_path=["a.png"])
    assert _attach_object_crop_paths(tmp_path / "none.db", frame, "cell") is frame


def test_a_png_list_without_the_crop_identity_is_ignored(tmp_path, caplog):
    from spacr.png_list import _attach_object_crop_paths

    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE png_list (png_path TEXT, plateID TEXT)")
    frame = _objects()
    with caplog.at_level("INFO"):
        assert _attach_object_crop_paths(db, frame, "cell") is frame
    assert any("lacks the complete" in r.getMessage() for r in caplog.records)


class _Reply:
    def __init__(self, body, url):
        self.body, self.url = body, url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size):
        return self.body

    def geturl(self):
        return self.url


def test_an_oversized_lims_page_is_refused(monkeypatch):
    import urllib.request

    from spacr import plate_qc

    class _Opener:
        def open(self, request, timeout):
            return _Reply(b"x" * (plate_qc._LIMS_MAX_PAGE_BYTES + 1),
                          request.full_url)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *h: _Opener())
    with pytest.raises(ValueError, match="page byte limit"):
        plate_qc._default_lims_fetch("https://lims.example/p", {})


def test_a_lims_redirect_back_to_a_visited_page_is_refused():
    from spacr import plate_qc

    first = []

    def fetch(url, headers):
        if not first:
            first.append(url)
            return plate_qc._LimsPage(
                b'{"records": [], "next": "https://lims.example/p2"}', url)
        return plate_qc._LimsPage(b"[]", first[0])

    with pytest.raises(ValueError, match="redirect repeats"):
        plate_qc._lims_records("https://lims.example/plates", ["B1"],
                               fetch=fetch)


def test_a_lims_redirect_to_another_service_is_refused():
    from spacr import plate_qc

    def fetch(url, headers):
        return plate_qc._LimsPage(b"[]", "https://elsewhere.example/data")

    with pytest.raises(ValueError, match="original service origin"):
        plate_qc._lims_records("https://lims.example/plates", ["B1"],
                               fetch=fetch)


def test_layer_viewer_test_data_with_no_images_says_so(qtbot, tmp_path):
    from spacr.qt.layer_viewer import LayerViewer

    viewer = LayerViewer()
    qtbot.addWidget(viewer)
    viewer._use_test_data({"images": tmp_path, "masks": {}})
    assert viewer.status.text() == "The test data has no images."


def test_layer_viewer_opens_one_field_and_its_existing_masks(qtbot, tmp_path):
    import numpy as np
    import tifffile

    from spacr.qt.layer_viewer import LayerViewer

    images = tmp_path / "images" / "plate1"
    images.mkdir(parents=True)
    for name in ("A01_f1_ch1.tif", "A01_f1_ch2.tif", "A01_f2_ch1.tif"):
        tifffile.imwrite(images / name, np.zeros((8, 8), np.uint16))
    masks = tmp_path / "masks" / "plate1"
    masks.mkdir(parents=True)
    tifffile.imwrite(masks / "A01_f1_ch1.tif", np.ones((8, 8), np.uint16))
    opened, labels = [], []
    viewer = LayerViewer()
    qtbot.addWidget(viewer)
    viewer.add_image_file = opened.append
    viewer.add_labels_file = labels.append
    viewer._use_test_data({"images": tmp_path / "images",
                           "masks": {"cell": tmp_path / "masks",
                                     "nucleus": tmp_path / "none"}})
    assert [p.rsplit("/", 1)[-1] for p in opened] == ["A01_f1_ch1.tif",
                                                      "A01_f1_ch2.tif"]
    assert len(labels) == 1


@pytest.fixture
def edge(qtbot):
    from spacr.qt.widgets.dock import Dock, DockEdge

    host = QWidget()
    qtbot.addWidget(host)
    dock = Dock([("__home__", "Home", "Home", ""),
                 ("measure", "Measure", "Objects", "")])
    row = QHBoxLayout(host)
    row.addWidget(dock)
    widget = DockEdge(dock, host)
    row.addWidget(widget)
    host.show()
    widget.keep_host_alive = host
    yield widget


def test_other_keys_on_the_dock_edge_do_not_toggle_it(edge):
    before = edge._collapsed
    edge.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier))
    assert edge._collapsed == before


def test_a_tiny_drag_does_not_resize_the_dock(edge):
    def mouse(kind, x):
        point = QPointF(x, 5)
        return QMouseEvent(kind, point, point, Qt.LeftButton, Qt.LeftButton,
                           Qt.NoModifier)

    width = edge._dock.width()
    edge.mousePressEvent(mouse(QEvent.MouseButtonPress, 10))
    edge.mouseMoveEvent(mouse(QEvent.MouseMove, 12))
    assert edge._dragged is False and edge._dock.width() == width


def test_leaving_a_dock_row_cancels_its_pending_help(edge):
    dock = edge._dock
    dock._on_row_hovered("measure", True)
    assert dock._hover_help_delay._anchor is not None
    dock._on_row_hovered("measure", False)
    assert dock._hover_help_delay._anchor is None

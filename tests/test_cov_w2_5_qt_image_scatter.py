"""A point in the image scatter is the cell it stands for, and stays so.

The index of a point is its identity: it indexes the key list, the crop
paths, and the ring the shared selection draws. So the paths worth driving
are the ones where something is missing — a point with no coordinate, a
table with no object keys, a key with no crop — because each is a chance for
the plot to renumber itself and start opening the object next door.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("PIL")

from PIL import Image                                            # noqa: E402
from PySide6.QtGui import QPixmap                                # noqa: E402
from PySide6.QtWidgets import QFileDialog                        # noqa: E402

from spacr.qt.crop_thumbs import CropThumbnails                  # noqa: E402
from spacr.qt.linked_selection import (                          # noqa: E402
    register_object_opener, unregister_object_opener)
from spacr.qt.screens import image_scatter as isc                # noqa: E402
from spacr.selection import Selection                            # noqa: E402


@pytest.fixture
def plate(tmp_path):
    """A measurements database with four objects and four real crops."""
    crops = tmp_path / "crops"
    crops.mkdir()
    rows = []
    for index in range(4):
        path = crops / f"object{index}.png"
        Image.fromarray(
            np.full((16, 16, 3), 30 * (index + 1), dtype=np.uint8)).save(path)
        rows.append({
            "plateID": "plate1", "rowID": "r1", "columnID": "c1",
            "fieldID": "f1", "object_label": index + 1,
            "cell_area": 100.0 + index, "cell_intensity": 5.0 - index,
            "png_path": str(path),
        })
    frame = pd.DataFrame(rows)
    db_path = tmp_path / "measurements.db"
    connection = sqlite3.connect(db_path)
    try:
        frame.drop(columns=["png_path"]).to_sql("cell", connection,
                                                index=False)
        frame[["plateID", "rowID", "columnID", "fieldID", "object_label",
               "png_path"]].rename(columns={"object_label": "cell_id"}).to_sql(
            "png_list", connection, index=False)
    finally:
        connection.close()
    return {
        "db": str(db_path),
        "frame": frame.drop(columns=["png_path"]),
        "keys": [f"plate1_r1_c1_f1_cell{i + 1}" for i in range(4)],
        "paths": {f"plate1_r1_c1_f1_cell{i + 1}": str(crops / f"object{i}.png")
                  for i in range(4)},
    }


@pytest.fixture
def screen(qtbot, plate):
    """A screen whose reads run inline, sized so points really project."""
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.resize(420, 320)
    widget.canvas.resize(300, 260)
    widget._db.setText(plate["db"])
    widget._thumbs = CropThumbnails(plate["db"])
    widget.set_frame(plate["frame"], keys=plate["keys"], paths=plate["paths"],
                     x="cell_area", y="cell_intensity")
    return widget


@pytest.fixture
def opener():
    """A real registered opener that records the requests it is handed."""
    received = []
    register_object_opener("annotate", received.append)
    try:
        yield received
    finally:
        unregister_object_opener("annotate", received.append)


# ---------------------------------------------------------------------------
# reading the database
# ---------------------------------------------------------------------------

def test_a_path_that_is_not_a_database_lists_no_tables(tmp_path):
    """A wrong path is an empty list, so the picker stays honestly empty."""
    assert isc.list_tables("") == []
    assert isc.list_tables(str(tmp_path / "nope.db")) == []


def test_the_tables_come_back_sorted(plate):
    """Every table in the file, in a stable order."""
    assert isc.list_tables(plate["db"]) == ["cell", "png_list"]


# ---------------------------------------------------------------------------
# the canvas
# ---------------------------------------------------------------------------

def test_an_empty_canvas_has_nothing_plottable_and_hits_nothing(qtbot):
    """No points is a real state, not a crash waiting for a click."""
    canvas = isc.ScatterCanvas()
    qtbot.addWidget(canvas)

    assert len(canvas) == 0
    assert canvas.plottable.tolist() == []
    assert canvas.index_at(10.0, 10.0) == -1
    assert canvas.point_position(0) is None


def test_a_selection_of_the_wrong_length_rings_nothing(qtbot):
    """A stale mask must not ring points it was not computed for."""
    canvas = isc.ScatterCanvas()
    qtbot.addWidget(canvas)
    canvas.set_points([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    canvas.set_selected([True, False])

    assert canvas._selected.tolist() == [False, False, False]


def test_the_canvas_paints_rings_labels_and_a_hover(qtbot):
    """Every decoration is drawn in one pass, and none of them raises."""
    canvas = isc.ScatterCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(220, 200)
    canvas.set_points([1.0, 2.0, 3.0], [1.0, 4.0, 9.0],
                      x_label="area", y_label="intensity")
    canvas.set_selected([True, False, True])
    canvas._set_hover(1)

    picture = canvas.grab()

    assert isinstance(picture, QPixmap)
    assert not picture.isNull()
    assert canvas.hovered == 1


def test_an_empty_canvas_says_there_are_no_points(qtbot):
    """The blank state is labelled rather than left as an empty rectangle."""
    canvas = isc.ScatterCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(120, 100)

    assert not canvas.grab().isNull()
    assert len(canvas) == 0


# ---------------------------------------------------------------------------
# choosing the source
# ---------------------------------------------------------------------------

def test_choosing_a_database_lists_its_tables(qtbot, plate, monkeypatch):
    """The dialog's answer is taken and the listing starts at once."""
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (plate["db"], "")))

    widget._choose_db()

    assert widget.database() == plate["db"]
    assert [widget._table.itemText(i) for i in range(widget._table.count())] \
        == ["cell", "png_list"]
    assert "2 table(s)" in widget.status.text()


def test_cancelling_the_dialog_leaves_the_box_alone(qtbot, plate,
                                                    monkeypatch):
    """An empty answer must not clear a path the user typed."""
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._db.setText(plate["db"])
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))

    widget._choose_db()

    assert widget.database() == plate["db"]
    assert widget._table.count() == 0


def test_plotting_with_no_table_chosen_says_to_choose_one(qtbot, plate):
    """The button explains itself instead of reading nothing."""
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._db.setText(plate["db"])

    widget.load_table()

    assert widget.status.text() == "Choose a table first."


def test_an_empty_payload_leaves_the_plot_alone(screen):
    """A worker that produced nothing must not blank a good plot."""
    before = len(screen._frame)

    screen._on_loaded({})

    assert len(screen._frame) == before


def test_a_failed_read_is_reported_in_the_status_line(screen):
    """The reason lands on the label, coloured as an error."""
    screen._on_job_failed("no such table: nucleus")

    assert screen.status.text() == "no such table: nucleus"
    assert "color:" in screen.status.styleSheet()


# ---------------------------------------------------------------------------
# the frame
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# hovering
# ---------------------------------------------------------------------------


def test_resting_on_nothing_decodes_nothing(screen):
    """The timer firing with no point under it is not a decode."""
    screen._pending_hover = -1

    screen._show_hovered_crop()                  # must not raise

    assert screen.caption.text() == ""


# ---------------------------------------------------------------------------
# opening a point
# ---------------------------------------------------------------------------


def test_an_opener_that_refuses_is_reported_not_raised(screen, caplog):
    """A failing destination becomes a status line naming the object."""
    def refuse(request):
        raise RuntimeError("annotate is closing")

    register_object_opener("annotate", refuse)
    try:
        with caplog.at_level("ERROR",
                             logger="spacr.qt.screens.image_scatter"):
            assert screen.open_point(1) is None
    finally:
        unregister_object_opener("annotate", refuse)

    assert screen.key_at(1) in screen.status.text()
    assert "Could not open" in screen.status.text()


def test_opening_an_index_with_no_key_routes_nothing(screen, opener):
    """An index past the end is not an object."""
    assert screen.open_point(99) is None
    assert opener == []


# ---------------------------------------------------------------------------
# the shared selection
# ---------------------------------------------------------------------------


def test_a_cleared_selection_rings_nothing(screen):
    """Returning to rest un-rings every point rather than keeping the last."""
    screen.on_linked_selection_changed(
        Selection(keys=pd.Index([screen.key_at(0)]), source="elsewhere"))
    assert screen.canvas._selected.any()

    screen.on_linked_selection_changed(Selection(keys=None, source=""))

    assert not screen.canvas._selected.any()


def test_a_keyless_plot_rings_nothing_whatever_is_selected(qtbot):
    """Without keys there is nothing to match a selection against."""
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))

    widget.on_linked_selection_changed(
        Selection(keys=pd.Index(["plate1_r1_c1_f1_cell1"]), source="x"))

    assert not widget.canvas._selected.any()


def test_hovering_the_same_point_twice_emits_once(qtbot):
    """The hover signal reports changes, not every mouse move."""
    canvas = isc.ScatterCanvas()
    qtbot.addWidget(canvas)
    canvas.set_points([1.0, 2.0], [1.0, 2.0])
    seen = []
    canvas.hover_changed.connect(seen.append)

    canvas._set_hover(1)
    canvas._set_hover(1)

    assert seen == [1]


def test_the_canvas_gets_a_surface_of_its_own(qapp):
    """The panel behind shows between the points rather than under a slab."""
    from spacr.qt.theme import active_palette

    block = isc._image_scatter_qss(active_palette(), 0.5)

    assert f"QFrame#{isc.CANVAS_OBJECT}" in block
    assert "border-radius" in block


def test_the_factory_builds_the_screen_and_no_row_comes_with_it(qtbot):
    """The factory outlived the registration it was written for.

    It used to be the ``factory=`` of a registry row and this test
    asserted the row was there. The row is gone -- Image Scatter is a
    button on Image UMAP -- and the factory is still the one constructor
    the module offers, which is what Image UMAP's builder calls, so what
    is asserted is that it builds the screen and adds no tile.
    """
    from spacr.qt.app import APPS

    made = isc.make_image_scatter_screen()
    qtbot.addWidget(made)

    assert isinstance(made, isc.ImageScatterScreen)
    assert not any(row[0] == isc.APP_KEY for row in APPS)
    made.close()

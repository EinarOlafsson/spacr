"""Image Scatter: a point is the object it stands for, from read to click.

The index of a point is its identity: it indexes the key list, the crop
paths and the ring the shared selection draws.  So the behaviours pinned here
are the ones where something is missing -- a point with no coordinate, a
table with no object keys, a key with no crop, a database with no tables --
because each is a chance for the plot to renumber itself and open the object
next door.  Every answer is an exact status line, caption or value.

Offscreen; reads run inline (``threaded=False``).
"""
from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("PIL")

from PIL import Image                                            # noqa: E402
from PySide6.QtCore import QEvent, QPointF, Qt                   # noqa: E402
from PySide6.QtGui import QMouseEvent                            # noqa: E402

from spacr.qt.crop_thumbs import CropThumbnails                  # noqa: E402
from spacr.qt.linked_selection import (                          # noqa: E402
    register_object_opener, unregister_object_opener)
from spacr.qt.screens import image_scatter as isc                # noqa: E402
from spacr.selection import Selection                            # noqa: E402

pytestmark = pytest.mark.qt

KEYLESS_CLICK = (
    "This table has no object keys, so a point cannot be opened. Plot a "
    "table that carries plateID/rowID/columnID/fieldID/object_label.")


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
    keys = [f"plate1_r1_c1_f1_cell{i + 1}" for i in range(4)]
    return {
        "db": str(db_path),
        "frame": frame.drop(columns=["png_path"]),
        "keys": keys,
        "paths": {key: str(crops / f"object{i}.png")
                  for i, key in enumerate(keys)},
    }


@pytest.fixture
def make_screen(qtbot):
    """Build an inline screen, sized so its points really project."""
    def _build():
        widget = isc.ImageScatterScreen(threaded=False)
        qtbot.addWidget(widget)
        widget.resize(420, 320)
        widget.canvas.resize(300, 260)
        return widget
    return _build


@pytest.fixture
def screen(make_screen, plate):
    """A screen showing the four-object plate, crops resolvable."""
    widget = make_screen()
    widget._db.setText(plate["db"])
    widget._thumbs = CropThumbnails(plate["db"])
    widget.set_frame(plate["frame"], keys=plate["keys"], paths=plate["paths"],
                     x="cell_area", y="cell_intensity")
    return widget


@pytest.fixture
def opener():
    """A real registered Annotate opener that records what it is handed."""
    received = []
    register_object_opener("annotate", received.append)
    try:
        yield received
    finally:
        unregister_object_opener("annotate", received.append)


@pytest.fixture
def canvas(qtbot):
    """A bare canvas at its minimum size, 240 x 200."""
    widget = isc.ScatterCanvas()
    qtbot.addWidget(widget)
    widget.resize(240, 200)
    return widget


def _mouse(kind, x, y, button):
    return QMouseEvent(kind, QPointF(x, y), button, button, Qt.NoModifier)


def _click(canvas, x, y, button=Qt.LeftButton):
    canvas.mousePressEvent(_mouse(QEvent.MouseButtonPress, x, y, button))


def _move(canvas, x, y):
    canvas.mouseMoveEvent(_mouse(QEvent.MouseMove, x, y, Qt.NoButton))


def test_load_scatter_frame_reads_one_table_or_refuses_plainly(
        subtests, plate, tmp_path):
    """A missing database raises with its path; a table comes back stamped."""
    for case_id, path in (("absent file", str(tmp_path / "gone.db")),
                          ("blank path", "")):
        with subtests.test(msg=case_id):
            with pytest.raises(FileNotFoundError) as caught:
                isc.load_scatter_frame(path, "cell")
            assert str(caught.value) == f"no measurements database at {path!r}"

    with subtests.test(msg="the cell table, stamped"):
        frame = isc.load_scatter_frame(plate["db"], "cell")
        assert list(frame.columns) == list(plate["frame"].columns) + [
            "object_type"]
        assert frame["object_type"].tolist() == ["cell"] * 4
        assert frame["cell_area"].tolist() == [100.0, 101.0, 102.0, 103.0]

    with subtests.test(msg="the row cap"):
        assert len(isc.load_scatter_frame(plate["db"], "cell", limit=2)) == 2


def test_scatter_canvas_keeps_every_index_and_draws_only_finite_points(
        subtests, canvas):
    """Non-finite rows keep their place; mismatched columns are refused."""
    # A lone finite point has no range to scale by, so it sits mid-canvas.
    cases = (
        ("one point of three is finite", [1.0, np.nan, 3.0],
         [1.0, 2.0, np.nan], 1, [True, False, False],
         [(120.0, 100.0), None, None], 0),
        ("nothing finite", [np.nan, np.nan], [np.nan, np.nan], 0,
         [False, False], [None, None], -1),
        ("empty", [], [], 0, [], [], -1),
    )
    for case_id, xs, ys, drawn, plottable, where, centre_hit in cases:
        with subtests.test(msg=case_id):
            assert canvas.set_points(xs, ys) == drawn
            assert len(canvas) == len(xs)
            assert canvas.plottable.tolist() == plottable
            assert [canvas.point_position(i) for i in range(len(xs))] == where
            assert canvas.point_position(len(xs)) is None
            assert canvas.index_at(120.0, 100.0) == centre_hit
            assert canvas.hovered == -1
            assert not canvas.grab().isNull()      # the paint really runs

    with subtests.test(msg="one y per x"):
        with pytest.raises(ValueError) as caught:
            canvas.set_points([1.0, 2.0, 3.0], [1.0, 2.0])
        assert str(caught.value) == "a scatter needs one y per x, got 3 and 2"

    with subtests.test(msg="a stale selection mask rings nothing"):
        canvas.set_points([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        canvas.set_selected([True, False])
        assert canvas._selected.tolist() == [False, False, False]


def test_scatter_canvas_hover_and_click_follow_the_left_button(subtests,
                                                               canvas):
    """The hover tracks the cursor; only a left click on a point emits."""
    canvas.set_points([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    hovers, clicks = [], []
    canvas.hover_changed.connect(hovers.append)
    canvas.point_clicked.connect(clicks.append)
    first = canvas.point_position(0)
    second = canvas.point_position(1)
    assert first == (18.0, 182.0)
    assert second == (120.0, 100.0)
    assert canvas.point_position(2) == (222.0, 18.0)

    steps = (
        ("move onto point 1", lambda: _move(canvas, *second), 1, [1], []),
        ("move into empty space", lambda: _move(canvas, 2.0, 2.0), -1,
         [1, -1], []),
        ("hover point 2 then leave",
         lambda: (canvas._set_hover(2),
                  canvas.leaveEvent(QEvent(QEvent.Leave))), -1,
         [1, -1, 2, -1], []),
        ("the same hover twice emits once",
         lambda: (canvas._set_hover(-1)), -1, [1, -1, 2, -1], []),
        ("right click on point 0", lambda: _click(canvas, *first,
                                                  Qt.RightButton), -1,
         [1, -1, 2, -1], []),
        ("left click in empty space", lambda: _click(canvas, 150.0, 4.0), -1,
         [1, -1, 2, -1], []),
        ("left click on point 0", lambda: _click(canvas, *first), 0,
         [1, -1, 2, -1, 0], [0]),
    )
    for case_id, act, hovered, seen_hovers, seen_clicks in steps:
        with subtests.test(msg=case_id):
            act()
            assert canvas.hovered == hovered
            assert hovers == seen_hovers
            assert clicks == seen_clicks


def test_scatter_canvas_logs_a_paint_that_raises_instead_of_raising(
        canvas, monkeypatch, caplog):
    """A broken palette is one logged failure, not a dead window."""
    canvas.set_points([1.0, 2.0], [1.0, 2.0])

    def explode():
        raise RuntimeError("no palette")

    monkeypatch.setattr(isc, "active_palette", explode)
    with caplog.at_level(logging.ERROR,
                         logger="spacr.qt.screens.image_scatter"):
        canvas.grab()                               # must not raise

    messages = [record.getMessage() for record in caplog.records
                if record.name == "spacr.qt.screens.image_scatter"]
    assert messages == ["Could not paint the image scatter"]
    assert caplog.records[-1].exc_info[1].args == ("no palette",)


def test_image_scatter_source_lists_tables_and_plots_the_chosen_one(
        subtests, make_screen, plate, tmp_path):
    """Seeding, an empty database and Plot each say exactly what happened."""
    with subtests.test(msg="a blank path keeps the typed one"):
        widget = make_screen()
        widget._db.setText(plate["db"])
        assert widget.set_database("   ") is False
        assert widget.set_database(None) is False
        assert widget.database() == plate["db"]
        assert widget._table.count() == 0

    with subtests.test(msg="a seeded path is taken and listed"):
        widget = make_screen()
        assert widget.set_database(f"  {plate['db']} ") is True
        assert widget.database() == plate["db"]
        assert [widget._table.itemText(i)
                for i in range(widget._table.count())] == ["cell", "png_list"]
        assert widget.status.text() == "2 table(s). Choose one and select Plot."

    with subtests.test(msg="a database with no tables"):
        empty = tmp_path / "empty.db"
        sqlite3.connect(empty).close()
        widget = make_screen()
        assert widget.set_database(str(empty)) is True
        assert widget._table.count() == 0
        assert widget.status.text() == (
            "No tables — is that a measurements database?")

    with subtests.test(msg="Plot reads the frame, keys and crops together"):
        widget = make_screen()
        widget.set_database(plate["db"])
        widget._table.setCurrentText("cell")
        widget.load_table()
        assert widget._frame["cell_area"].tolist() == [
            100.0, 101.0, 102.0, 103.0]
        assert widget._keys == plate["keys"]
        assert [widget.path_at(i) for i in range(4)] == [
            plate["paths"][key] for key in plate["keys"]]
        assert widget._x_choice.currentText() == "object_label"
        assert widget._y_choice.currentText() == "cell_area"
        assert len(widget.canvas) == 4
        assert widget.status.text() == "4 row(s) from cell · 4 with a crop"


def test_image_scatter_set_frame_derives_keys_and_crop_paths(
        subtests, make_screen, plate):
    """Keys come from the key columns, crops from a path column, else nothing."""
    loaded = isc.load_scatter_frame(plate["db"], "cell")
    with_paths = loaded.copy()
    with_paths["png_path"] = [plate["paths"][key] for key in plate["keys"]]
    keyless = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    cases = (
        ("keys derived from the frame", loaded, plate["keys"], {},
         "4 row(s) · 0 with a crop", 4),
        ("crop path column used", with_paths, plate["keys"], plate["paths"],
         "4 row(s) · 4 with a crop", 4),
        ("no key columns", keyless, [], {},
         "2 row(s) · no object keys, so points cannot be opened", 2),
        ("not a frame", None, [], {},
         "0 row(s) · no object keys, so points cannot be opened", 0),
    )
    for case_id, frame, keys, paths, status, points in cases:
        with subtests.test(msg=case_id):
            widget = make_screen()
            widget.set_frame(frame)
            assert widget._keys == keys
            assert widget._paths == paths
            assert widget.status.text() == status
            assert len(widget.canvas) == points
            assert widget.key_at(-1) == ""
            assert widget.key_at(99) == ""
            assert widget.path_at(99) == ""

    with subtests.test(msg="a click on a keyless plot names the columns"):
        widget = make_screen()
        widget.set_frame(keyless)
        widget._on_click(0)
        assert widget.status.text() == KEYLESS_CLICK


def test_image_scatter_hover_preview_waits_shows_or_says_why_not(
        subtests, screen, make_screen, plate):
    """Uncached crops wait for the timer; cached ones and misses are at once."""
    with subtests.test(msg="an uncached crop waits for the cursor to rest"):
        screen._on_hover(1)
        assert screen._hover_timer.isActive()
        assert screen._hover_timer.interval() == isc.HOVER_DELAY_MS
        assert screen._pending_hover == 1
        assert screen.preview.text() == "Hover a point"

    with subtests.test(msg="the rested crop is decoded and captioned"):
        shown = []
        screen.crop_shown.connect(shown.append)
        screen._show_hovered_crop()
        assert screen.caption.text() == plate["keys"][1]
        assert not screen.preview.pixmap().isNull()
        assert shown == [plate["keys"][1]]

    with subtests.test(msg="a decoded crop appears with no timer"):
        screen._on_hover(-1)
        screen._on_hover(1)
        assert not screen._hover_timer.isActive()
        assert screen.caption.text() == plate["keys"][1]

    with subtests.test(msg="leaving the plot clears the preview"):
        screen._on_hover(0)
        screen._on_hover(-1)
        assert screen.caption.text() == ""
        assert screen.preview.text() == "Hover a point"
        assert not screen._open_button.isEnabled()
        assert not screen._hover_timer.isActive()
        assert screen._pending_hover == -1

    with subtests.test(msg="resting on nothing decodes nothing"):
        screen._show_hovered_crop()
        assert screen.caption.text() == ""

    with subtests.test(msg="a key with no crop says so"):
        widget = make_screen()
        widget._thumbs = CropThumbnails(plate["db"])
        widget.set_frame(plate["frame"], keys=plate["keys"], paths={},
                         x="cell_area", y="cell_intensity")
        widget._on_hover(0)
        widget._show_hovered_crop()
        assert widget.preview.text() == "no crop for this object"
        assert widget.caption.text() == plate["keys"][0]

    with subtests.test(msg="a point with no key is named by its index"):
        widget = make_screen()
        widget.set_frame(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
        widget._show_crop(1, None)
        assert widget.preview.text() == "no object key"
        assert widget.caption.text() == "point 1"
        assert not widget._open_button.isEnabled()


def test_image_scatter_click_publishes_opens_and_rings(subtests, screen,
                                                       opener, plate):
    """A click rings the object everywhere and routes it to Annotate."""
    expected_context = {"x": "cell_area", "y": "cell_intensity",
                        "db_path": plate["db"]}

    with subtests.test(msg="a click publishes and opens"):
        screen._on_click(2)
        assert len(opener) == 1
        request = opener[0]
        assert list(request.keys) == [plate["keys"][2]]
        assert request.reason == (
            "clicked in the scatter of cell_intensity against cell_area")
        assert request.source == isc.LINK_SOURCE
        assert request.context == {**expected_context, "index": 2}
        assert list(screen.link.selection.keys) == [plate["keys"][2]]

    with subtests.test(msg="the side button opens the hovered point"):
        screen.canvas._set_hover(3)
        screen._open_hovered()
        assert len(opener) == 2
        assert list(opener[1].keys) == [plate["keys"][3]]
        assert opener[1].context == {**expected_context, "index": 3}

    with subtests.test(msg="an index with no key opens nothing"):
        assert screen.open_point(99) is None
        assert len(opener) == 2

    with subtests.test(msg="another view's selection rings these points"):
        screen.on_linked_selection_changed(
            Selection(keys=pd.Index([plate["keys"][1], plate["keys"][3]]),
                      source="somewhere else"))
        assert screen.canvas._selected.tolist() == [False, True, False, True]

    with subtests.test(msg="a cleared selection rings nothing"):
        screen.on_linked_selection_changed(Selection(keys=None, source=""))
        assert screen.canvas._selected.tolist() == [False] * 4

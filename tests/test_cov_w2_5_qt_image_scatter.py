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
from PySide6.QtCore import QEvent, QPointF, Qt                   # noqa: E402
from PySide6.QtGui import QMouseEvent, QPixmap                   # noqa: E402
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


def _click(canvas, x, y, button=Qt.LeftButton):
    event = QMouseEvent(QEvent.MouseButtonPress, QPointF(x, y), button,
                        button, Qt.NoModifier)
    canvas.mousePressEvent(event)


def _move(canvas, x, y):
    event = QMouseEvent(QEvent.MouseMove, QPointF(x, y), Qt.NoButton,
                        Qt.NoButton, Qt.NoModifier)
    canvas.mouseMoveEvent(event)


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


def test_a_missing_database_says_so_rather_than_plotting_nothing(tmp_path):
    """An empty plot from a wrong path looks exactly like an empty table."""
    with pytest.raises(FileNotFoundError) as caught:
        isc.load_scatter_frame(str(tmp_path / "gone.db"), "cell")

    assert "no measurements database" in str(caught.value)


def test_the_frame_is_stamped_with_the_table_it_came_from(plate):
    """A cell 1 and a nucleus 1 in one field must not share a key."""
    frame = isc.load_scatter_frame(plate["db"], "cell")

    assert len(frame) == 4
    assert set(frame["object_type"]) == {"cell"}


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


def test_a_point_with_no_coordinate_keeps_its_index(qtbot):
    """Dropping it would renumber every point after it."""
    canvas = isc.ScatterCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(200, 200)

    drawn = canvas.set_points([1.0, np.nan, 3.0], [1.0, 2.0, np.nan])

    assert len(canvas) == 3
    assert drawn == 1
    assert canvas.plottable.tolist() == [True, False, False]
    assert canvas.point_position(1) is None


def test_nothing_finite_projects_to_nothing_without_raising(qtbot):
    """An all-NaN table draws an empty plot, not a division by zero."""
    canvas = isc.ScatterCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(200, 200)

    assert canvas.set_points([np.nan, np.nan], [np.nan, np.nan]) == 0
    assert canvas.index_at(100.0, 100.0) == -1
    canvas.grab()                                # must not raise


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


def test_a_paint_that_raises_does_not_take_the_window_with_it(qtbot,
                                                              monkeypatch,
                                                              caplog):
    """A broken palette is a logged failure, not a dead application."""
    canvas = isc.ScatterCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(200, 180)
    canvas.set_points([1.0, 2.0], [1.0, 2.0])

    def explode():
        raise RuntimeError("no palette")

    monkeypatch.setattr(isc, "active_palette", explode)

    with caplog.at_level("ERROR", logger="spacr.qt.screens.image_scatter"):
        canvas.grab()                            # must not raise

    assert "Could not paint the image scatter" in caplog.text


def test_moving_over_a_point_and_off_it_again_moves_the_hover(qtbot):
    """The hover follows the cursor and is cleared when it leaves."""
    canvas = isc.ScatterCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(220, 200)
    canvas.set_points([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    seen = []
    canvas.hover_changed.connect(seen.append)
    where = canvas.point_position(1)

    _move(canvas, where[0], where[1])
    assert canvas.hovered == 1

    _move(canvas, 2.0, 2.0)
    assert canvas.hovered == -1

    canvas._set_hover(2)
    canvas.leaveEvent(QEvent(QEvent.Leave))

    assert canvas.hovered == -1
    assert seen == [1, -1, 2, -1]


def test_a_right_click_is_not_a_click_on_a_point(qtbot):
    """Only the left button opens an object."""
    canvas = isc.ScatterCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(220, 200)
    canvas.set_points([1.0, 2.0], [1.0, 2.0])
    clicked = []
    canvas.point_clicked.connect(clicked.append)
    where = canvas.point_position(0)

    _click(canvas, where[0], where[1], Qt.RightButton)

    assert clicked == []

    _click(canvas, where[0], where[1], Qt.LeftButton)

    assert clicked == [0]


def test_clicking_empty_space_selects_nothing(qtbot):
    """A miss is a miss; the previous point is not re-emitted."""
    canvas = isc.ScatterCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(300, 300)
    canvas.set_points([1.0, 2.0], [1.0, 2.0])
    clicked = []
    canvas.point_clicked.connect(clicked.append)

    _click(canvas, 150.0, 4.0)

    assert clicked == []


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


def test_an_empty_path_is_ignored_rather_than_clearing_the_box(qtbot, plate):
    """"No path known" must not throw away a path already there."""
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._db.setText(plate["db"])

    assert widget.set_database("   ") is False
    assert widget.database() == plate["db"]


def test_a_seeded_path_is_taken_and_listed(qtbot, plate):
    """The seam Image UMAP uses to hand this screen the same database."""
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)

    assert widget.set_database(plate["db"]) is True
    assert widget.database() == plate["db"]
    assert widget._table.count() == 2


def test_a_database_with_no_tables_asks_whether_it_is_one(qtbot, tmp_path):
    """An empty listing is a question, not a blank picker."""
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)

    widget.set_database(str(empty))

    assert "is that a measurements database?" in widget.status.text()


def test_plotting_with_no_table_chosen_says_to_choose_one(qtbot, plate):
    """The button explains itself instead of reading nothing."""
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._db.setText(plate["db"])

    widget.load_table()

    assert widget.status.text() == "Choose a table first."


def test_loading_a_table_plots_it_with_its_crops(qtbot, plate):
    """One job reads the frame, its keys and its crop paths together."""
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_database(plate["db"])
    widget._table.setCurrentText("cell")

    widget.load_table()

    assert len(widget._frame) == 4
    assert widget._keys == plate["keys"]
    assert widget.path_at(0) == plate["paths"][plate["keys"][0]]
    assert "4 row(s) from cell" in widget.status.text()
    assert "4 with a crop" in widget.status.text()


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

def test_a_table_with_no_object_keys_still_plots_and_says_why(qtbot):
    """The plot draws; a click cannot open anything and says so."""
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.resize(400, 300)
    widget.canvas.resize(300, 250)

    widget.set_frame(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))

    assert widget._keys == []
    assert "no object keys" in widget.status.text()
    assert len(widget.canvas) == 2


def test_a_click_on_a_keyless_table_names_the_columns_it_wants(qtbot):
    """The message says which columns would make the point openable."""
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))

    widget._on_click(0)

    assert "no object keys" in widget.status.text()
    assert "object_label" in widget.status.text()


def test_something_that_is_not_a_frame_is_an_empty_plot(qtbot):
    """A caller handing over nothing gets an empty plot, not an exception."""
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)

    widget.set_frame(None)

    assert widget._frame.empty
    assert len(widget.canvas) == 0


def test_the_keys_are_derived_from_the_frame_when_they_are_not_given(qtbot,
                                                                     plate):
    """A frame carrying the key columns needs no separate key list."""
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)
    frame = isc.load_scatter_frame(plate["db"], "cell")

    widget.set_frame(frame)

    assert widget._keys == plate["keys"]


def test_a_crop_path_already_in_the_frame_is_used(qtbot, plate):
    """A crop table carries the path; going back to the database is waste."""
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)
    frame = isc.load_scatter_frame(plate["db"], "cell")
    frame["png_path"] = [plate["paths"][key] for key in plate["keys"]]

    widget.set_frame(frame)

    assert widget.path_at(2) == plate["paths"][plate["keys"][2]]


def test_an_index_outside_the_table_has_no_key_and_no_crop(screen):
    """Out of range is empty, not an IndexError from a stray hover."""
    assert screen.key_at(-1) == ""
    assert screen.key_at(99) == ""
    assert screen.path_at(99) == ""


# ---------------------------------------------------------------------------
# hovering
# ---------------------------------------------------------------------------

def test_leaving_the_plot_clears_the_preview(screen):
    """No point under the cursor means no crop and no caption."""
    screen._on_hover(0)
    screen._show_hovered_crop()
    assert screen.caption.text()

    screen._on_hover(-1)

    assert screen.caption.text() == ""
    assert screen.preview.text() == "Hover a point"
    assert not screen._open_button.isEnabled()
    assert not screen._hover_timer.isActive()


def test_an_uncached_crop_waits_for_the_cursor_to_rest(screen):
    """The plot must not decode a crop for every pixel the cursor crosses."""
    screen._on_hover(1)

    assert screen._hover_timer.isActive()
    assert screen._pending_hover == 1


def test_a_crop_already_decoded_appears_at_once(screen):
    """A second hover of the same point costs no decode and no timer."""
    screen._on_hover(1)
    screen._show_hovered_crop()

    screen._on_hover(1)

    assert not screen._hover_timer.isActive()
    assert screen.caption.text() == screen.key_at(1)


def test_resting_on_nothing_decodes_nothing(screen):
    """The timer firing with no point under it is not a decode."""
    screen._pending_hover = -1

    screen._show_hovered_crop()                  # must not raise

    assert screen.caption.text() == ""


def test_a_point_with_no_crop_says_so_rather_than_showing_the_last_one(qtbot,
                                                                       plate):
    """A stale crop under a new caption is worse than an empty box."""
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._thumbs = CropThumbnails(plate["db"])
    widget.set_frame(plate["frame"], keys=plate["keys"], paths={},
                     x="cell_area", y="cell_intensity")

    widget._on_hover(0)
    widget._show_hovered_crop()

    assert widget.preview.text() == "no crop for this object"
    assert widget.caption.text() == plate["keys"][0]


def test_a_point_with_no_key_is_captioned_by_its_index(qtbot):
    """Something has to name the point, and the index is what there is."""
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))

    widget._show_crop(1, None)

    assert widget.preview.text() == "no object key"
    assert widget.caption.text() == "point 1"


# ---------------------------------------------------------------------------
# opening a point
# ---------------------------------------------------------------------------

def test_a_click_publishes_the_object_and_opens_its_crop(screen, opener):
    """Two acts, and both happen: the ring everywhere and the crop here."""
    screen._on_click(2)

    assert len(opener) == 1
    assert list(opener[0].keys) == [screen.key_at(2)]
    assert "scatter of" in opener[0].reason
    assert opener[0].context["index"] == 2
    assert list(screen.link.selection.keys) == [screen.key_at(2)]


def test_the_side_button_opens_whatever_is_hovered(screen, opener):
    """The button is a second route to the same act."""
    screen.canvas._set_hover(3)

    screen._open_hovered()

    assert list(opener[0].keys) == [screen.key_at(3)]


def test_with_nowhere_to_open_a_point_nothing_is_routed(screen):
    """No registered opener means None, not an exception on a click."""
    assert screen.open_point(0) is None


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

def test_a_selection_from_another_view_rings_these_points(screen):
    """Brushing elsewhere lights up the same objects here."""
    screen.on_linked_selection_changed(
        Selection(keys=pd.Index([screen.key_at(1), screen.key_at(3)]),
                  source="somewhere else"))

    assert screen.canvas._selected.tolist() == [False, True, False, True]


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


def test_a_scatter_needs_one_y_per_x(qtbot):
    """Mismatched columns are refused rather than silently truncated."""
    canvas = isc.ScatterCanvas()
    qtbot.addWidget(canvas)

    with pytest.raises(ValueError) as caught:
        canvas.set_points([1.0, 2.0, 3.0], [1.0, 2.0])

    assert "one y per x" in str(caught.value)


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


def test_a_narrowed_population_is_stated_and_re_plotted(screen):
    """A filter hides rows, so the plot is redrawn and the label says why."""
    from spacr.selection import CategoryFilter, DataFilter

    narrowed = DataFilter([CategoryFilter("rowID", ("r1",))])

    screen.on_linked_filter_changed(narrowed)

    assert screen.status.text().startswith("Filter: ")
    assert len(screen.canvas) == 4


def test_a_filter_with_nothing_plotted_changes_nothing(qtbot):
    """A filter arriving before a table is not a status line."""
    from spacr.selection import CategoryFilter, DataFilter

    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)

    widget.on_linked_filter_changed(DataFilter([CategoryFilter("rowID",
                                                              ("r1",))]))

    assert widget.status.text() == ""


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

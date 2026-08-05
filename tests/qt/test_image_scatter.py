"""``V3`` — the scatter where a point is the cell it stands for.

Three claims are worth pinning. That a click reaches the registered opener
with the *right* object and not the one next to it. That hovering decodes a
crop once and then never again, because the alternative is a plot that lags
the cursor. And that a point whose coordinate is missing is still the same
point — dropping it would renumber every index and make clicks open the wrong
object, which is the failure that looks like working software.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("PIL")

from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from spacr.qt.crop_thumbs import CropThumbnails, crop_paths_for_keys
from spacr.qt.linked_selection import (register_object_opener,
                                       unregister_object_opener)
from spacr.qt.screens import image_scatter as isc


# ---------------------------------------------------------------------------
# Fixtures — a tiny plate with four objects and four real crop PNGs
# ---------------------------------------------------------------------------

@pytest.fixture
def plate(tmp_path):
    """A measurements.db with a `cell` table and a `png_list` beside it."""
    crops = tmp_path / "data" / "crops"
    crops.mkdir(parents=True)
    rows = []
    for index in range(4):
        path = crops / f"object{index}.png"
        Image.fromarray(
            np.full((24, 24, 3), 20 * (index + 1), dtype=np.uint8)).save(path)
        rows.append({
            "plateID": "plate1", "rowID": "r1", "columnID": "c1",
            "fieldID": "f1", "object_label": index + 1,
            "cell_area": 100.0 + index, "cell_intensity": 5.0 - index,
            "png_path": str(path),
        })
    frame = pd.DataFrame(rows)
    db_dir = tmp_path / "measurements"
    db_dir.mkdir()
    db_path = db_dir / "measurements.db"
    connection = sqlite3.connect(db_path)
    try:
        frame.drop(columns=["png_path"]).to_sql("cell", connection,
                                                index=False)
        frame[["plateID", "rowID", "columnID", "fieldID", "object_label",
               "png_path"]].rename(
            columns={"object_label": "cell_id"}).to_sql(
            "png_list", connection, index=False)
    finally:
        connection.close()
    # `cell`-typed keys, because `load_scatter_frame` stamps the table it
    # read: without the type a cell 1 and a nucleus 1 in this field would
    # publish the same string, and clicking one would open whichever crop
    # png_list happened to list first.
    return {"db": str(db_path), "frame": frame,
            "keys": [f"plate1_r1_c1_f1_cell{i + 1}" for i in range(4)],
            "paths": {f"plate1_r1_c1_f1_cell{i + 1}":
                      str(crops / f"object{i}.png") for i in range(4)}}


@pytest.fixture
def screen(qtbot, qt_theme_applied, plate):
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.resize(420, 320)
    widget.canvas.resize(300, 260)
    widget._db.setText(plate["db"])
    widget._thumbs = CropThumbnails(plate["db"])
    widget.set_frame(plate["frame"].drop(columns=["png_path"]),
                     keys=plate["keys"], paths=plate["paths"],
                     x="cell_area", y="cell_intensity")
    return widget


@pytest.fixture
def opener():
    received = []
    register_object_opener("annotate", received.append)
    try:
        yield received
    finally:
        unregister_object_opener("annotate", received.append)


# ---------------------------------------------------------------------------
# The canvas
# ---------------------------------------------------------------------------

def test_points_are_projected_into_the_widget_and_hit_tested(qtbot):
    canvas = isc.ScatterCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(200, 200)
    assert canvas.set_points([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]) == 3
    canvas._build_cloud()

    for index in range(3):
        x, y = canvas.point_position(index)
        assert canvas.index_at(x, y) == index


def test_a_click_far_from_every_point_hits_nothing(qtbot):
    canvas = isc.ScatterCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(200, 200)
    canvas.set_points([0.0, 1.0], [0.0, 1.0])
    canvas._build_cloud()
    # Dead centre of a two-point diagonal is far from both.
    assert canvas.index_at(100.0, 100.0) == -1


def test_a_point_with_no_coordinate_keeps_its_index_rather_than_renumbering(
        qtbot):
    canvas = isc.ScatterCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(200, 200)
    plotted = canvas.set_points([0.0, np.nan, 2.0], [0.0, 5.0, 2.0])

    assert plotted == 2
    assert len(canvas) == 3
    assert not canvas.plottable[1]
    assert canvas.point_position(1) is None
    # The last point is still index 2, not index 1.
    assert canvas.index_at(*canvas.point_position(2)) == 2


def test_a_constant_axis_centres_rather_than_dividing_by_zero(qtbot):
    canvas = isc.ScatterCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(200, 200)
    canvas.set_points([1.0, 1.0, 1.0], [0.0, 1.0, 2.0])
    canvas._build_cloud()
    assert np.all(np.isfinite(canvas._px))
    assert len(set(np.round(canvas._px, 3))) == 1


def test_x_and_y_of_different_lengths_is_refused(qtbot):
    canvas = isc.ScatterCanvas()
    qtbot.addWidget(canvas)
    with pytest.raises(ValueError, match="one y per x"):
        canvas.set_points([1.0, 2.0], [1.0])


def test_hovering_a_point_emits_once_and_leaving_clears_it(qtbot):
    canvas = isc.ScatterCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(200, 200)
    canvas.set_points([0.0, 1.0], [0.0, 1.0])
    canvas._build_cloud()
    seen = []
    canvas.hover_changed.connect(seen.append)

    canvas._set_hover(1)
    canvas._set_hover(1)          # the same point is not a new hover
    canvas._set_hover(-1)
    assert seen == [1, -1]


# ---------------------------------------------------------------------------
# The crop cache
# ---------------------------------------------------------------------------

def test_a_crop_is_decoded_once_however_often_it_is_hovered(qtbot, plate,
                                                            qt_theme_applied):
    cache = CropThumbnails(plate["db"])
    path = plate["paths"]["plate1_r1_c1_f1_cell1"]

    assert cache.peek(path) is None          # nothing decoded yet
    first = cache.pixmap(path)
    for _ in range(20):
        cache.pixmap(path)

    assert first is not None and not first.isNull()
    assert cache.decodes == 1
    assert cache.hits == 20
    assert path in cache


def test_a_missing_crop_is_remembered_as_missing_rather_than_retried(
        tmp_path, qt_theme_applied):
    cache = CropThumbnails()
    ghost = str(tmp_path / "not-here.png")
    assert cache.pixmap(ghost) is None
    assert cache.pixmap(ghost) is None
    assert cache.decodes == 1
    assert cache.failures == 1


def test_the_cache_forgets_the_oldest_rather_than_growing(plate,
                                                          qt_theme_applied):
    cache = CropThumbnails(plate["db"], capacity=2)
    for path in list(plate["paths"].values()):
        cache.pixmap(path)
    assert len(cache) == 2


def test_a_rewritten_crop_is_re_read_rather_than_served_stale(tmp_path,
                                                              qt_theme_applied):
    path = tmp_path / "crop.png"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(path)
    cache = CropThumbnails()
    cache.pixmap(str(path))
    assert cache.decodes == 1

    os.utime(path, (0, 0))                   # a re-run rewrote it
    cache.pixmap(str(path))
    assert cache.decodes == 2


def test_crop_paths_resolve_in_one_pass_when_every_key_is_present(plate):
    resolved = crop_paths_for_keys(plate["db"], plate["keys"])
    assert resolved == plate["paths"]


def test_crop_paths_still_line_up_when_some_keys_have_no_crop(plate):
    keys = [plate["keys"][0], "plate1_r1_c1_f1_cell999", plate["keys"][3]]
    resolved = crop_paths_for_keys(plate["db"], keys)
    assert resolved == {keys[0]: plate["paths"][keys[0]],
                        keys[2]: plate["paths"][keys[2]]}


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------

def test_the_axes_offer_the_numeric_columns(screen):
    options = [screen._x_choice.itemText(i)
               for i in range(screen._x_choice.count())]
    assert "cell_area" in options and "cell_intensity" in options
    assert screen._x_choice.currentText() == "cell_area"


def test_hovering_shows_that_point_s_crop_and_names_it(screen, plate):
    screen._on_hover(2)
    screen._show_hovered_crop()

    assert screen.caption.text() == plate["keys"][2]
    assert screen.preview.pixmap() is not None
    assert not screen.preview.pixmap().isNull()


def test_a_cached_crop_is_shown_without_waiting_for_the_debounce(screen):
    screen._on_hover(1)
    screen._show_hovered_crop()               # decode it once
    screen._on_hover(-1)

    screen._on_hover(1)
    assert not screen._hover_timer.isActive()   # answered from the cache
    assert screen.caption.text() == screen.key_at(1)


def test_sweeping_across_points_decodes_only_what_the_cursor_rests_on(screen):
    for index in (0, 1, 2, 3):
        screen._on_hover(index)               # a sweep: no timer has fired
    assert screen._thumbs.decodes == 0

    screen._show_hovered_crop()               # the cursor stopped on point 3
    assert screen._thumbs.decodes == 1
    assert screen.caption.text() == screen.key_at(3)


def test_leaving_the_plot_clears_the_preview(screen):
    screen._on_hover(0)
    screen._show_hovered_crop()
    screen._on_hover(-1)
    assert screen.caption.text() == ""
    assert not screen._open_button.isEnabled()


# ---------------------------------------------------------------------------
# Routing — a click reaches the opener with the right key
# ---------------------------------------------------------------------------

def test_a_click_reaches_the_registered_opener_with_that_point_s_key(
        screen, plate, opener):
    screen._on_click(2)

    assert len(opener) == 1
    assert list(opener[0].keys) == [plate["keys"][2]]
    assert opener[0].source == "image_scatter"
    assert "cell_intensity" in opener[0].reason
    assert opener[0].context["index"] == 2


def test_a_click_on_the_canvas_opens_the_point_under_the_cursor(
        screen, plate, opener, qtbot):
    x, y = screen.canvas.point_position(1)
    target = QPoint(int(x), int(y))
    QTest.mouseClick(screen.canvas, Qt.LeftButton, Qt.NoModifier, target)

    assert [list(request.keys) for request in opener] == [[plate["keys"][1]]]


def test_a_click_publishes_the_selection_as_well_as_opening_it(
        screen, plate, opener):
    screen._on_click(0)
    selection = screen.link.selection
    assert list(selection.keys) == [plate["keys"][0]]
    assert selection.source == "image_scatter"


def test_a_table_with_no_object_keys_says_so_rather_than_doing_nothing(
        qtbot, qt_theme_applied, opener):
    widget = isc.ImageScatterScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
    widget._on_click(0)

    assert opener == []
    assert "no object keys" in widget.status.text()


def test_with_nothing_registered_a_click_opens_nothing_and_does_not_raise(
        screen):
    assert screen.open_point(0) is None


def test_the_open_button_opens_the_point_under_the_cursor(screen, plate,
                                                          opener):
    screen.canvas._set_hover(3)
    screen._on_hover(3)
    screen._show_hovered_crop()
    assert screen._open_button.isEnabled()
    screen._open_button.click()
    assert list(opener[0].keys) == [plate["keys"][3]]


# ---------------------------------------------------------------------------
# The shared selection
# ---------------------------------------------------------------------------

def test_a_selection_made_elsewhere_rings_the_matching_points(screen, plate):
    from spacr.selection import Selection

    screen.link.set_selection(
        Selection.from_keys([plate["keys"][1], plate["keys"][3]],
                            source="umap"))
    assert list(screen.canvas._selected) == [False, True, False, True]


def test_the_resting_selection_rings_nothing(screen, plate):
    from spacr.selection import Selection

    screen.link.set_selection(Selection.from_keys([plate["keys"][0]],
                                                  source="umap"))
    screen.link.clear_selection()
    assert not screen.canvas._selected.any()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_the_table_list_comes_out_of_the_database(plate):
    assert set(isc.list_tables(plate["db"])) == {"cell", "png_list"}


def test_loading_a_table_reads_it_and_resolves_every_crop(plate):
    payload = isc.ImageScatterScreen._read(plate["db"], "cell")
    assert len(payload["frame"]) == 4
    assert payload["keys"] == plate["keys"]
    assert payload["paths"] == plate["paths"]


def test_a_missing_database_says_so_rather_than_plotting_nothing(tmp_path):
    with pytest.raises(FileNotFoundError, match="no measurements database"):
        isc.load_scatter_frame(str(tmp_path / "nope.db"), "cell")


def test_the_screen_registers_itself_into_the_app_registry():
    from spacr.qt.app import APPS

    isc.register()
    assert any(row[0] == isc.APP_KEY for row in APPS)
    assert isc.register() is None      # idempotent

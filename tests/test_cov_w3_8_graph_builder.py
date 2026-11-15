"""The graph builder's shelf, its drop zones, and the draws they lead to.

The canvas is rendered for real -- a matplotlib figure on the offscreen
platform -- and the assertions are about the artists that came out of it, so a
branch that draws nothing cannot pass by not raising.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from PySide6.QtCore import QMimeData, QPointF, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent,
)

from spacr.qt.linked_selection import LinkedSelection  # noqa: E402
from spacr.qt.widgets import graph_builder as gb  # noqa: E402
from spacr.qt.widgets.graph_spec import (  # noqa: E402
    HISTOGRAM, GraphSpec,
)

pytestmark = pytest.mark.qt


def _frame(rows=60):
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "plateID": [f"p{i % 2 + 1}" for i in range(rows)],
        "rowID": [f"r{i % 3 + 1}" for i in range(rows)],
        "columnID": [f"c{i % 4 + 1}" for i in range(rows)],
        "area": rng.normal(100, 10, rows),
        "intensity": rng.normal(size=rows),
        "gene": [f"g{i % 3}" for i in range(rows)],
        "cell_count": [i % 5 for i in range(rows)],
        # The object key columns, so the chart can join the linked selection
        # -- without them `set_frame` marks the table unkeyed and a brush
        # publishes nothing.
        "fieldID": [f"f{i % 2 + 1}" for i in range(rows)],
        "object_label": list(range(rows)),
    })


@pytest.fixture()
def canvas(qtbot):
    view = gb.GraphCanvas(link=LinkedSelection(), source="w3_8")
    qtbot.addWidget(view)
    view.set_frame(_frame())
    return view


# ---------------------------------------------------------------------------
# Theme-derived helpers
# ---------------------------------------------------------------------------

def test_a_palette_that_cannot_be_read_is_treated_as_dark(monkeypatch):
    """The series steps must still be chosen when the surface is unreadable."""
    monkeypatch.setattr(gb, "active_palette", lambda: {"surface": "not a hue"})
    assert gb._is_light_surface() is False
    assert len(gb.categorical_colours()) == 8


def test_a_light_surface_and_a_dark_one_are_told_apart(monkeypatch):
    monkeypatch.setattr(gb, "active_palette", lambda: {"surface": "#ffffff"})
    assert gb._is_light_surface() is True
    monkeypatch.setattr(gb, "active_palette", lambda: {"surface": "#0d0e10"})
    assert gb._is_light_surface() is False


def test_the_orientation_keyword_follows_the_installed_matplotlib(monkeypatch):
    """Matplotlib 3.10 renamed ``vert`` to ``orientation``."""
    import matplotlib

    monkeypatch.setattr(matplotlib, "__version__", "3.10.1")
    assert gb._orientation(True) == {"orientation": "vertical"}
    assert gb._orientation(False) == {"orientation": "horizontal"}
    monkeypatch.setattr(matplotlib, "__version__", "3.8.4")
    assert gb._orientation(True) == {"vert": True}
    assert gb._orientation(False) == {"vert": False}


def test_the_page_alpha_falls_back_when_preferences_cannot_be_read(
        monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "spacr.qt.preferences", None)
    value = gb.page_alpha()
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# The column shelf
# ---------------------------------------------------------------------------

def test_the_shelf_empties_when_the_table_goes(qtbot):
    shelf = gb.ColumnWell()
    qtbot.addWidget(shelf)
    shelf.set_frame(_frame())
    assert shelf.columns()
    assert "no table loaded" not in shelf._count.text()

    shelf.set_frame(None)
    assert shelf.columns() == ()
    assert shelf.visible_columns() == []
    assert shelf._count.text() == "no table loaded"


def test_the_search_box_narrows_the_shelf_without_losing_the_rest(qtbot):
    shelf = gb.ColumnWell()
    qtbot.addWidget(shelf)
    shelf.set_frame(_frame())
    total = len(shelf.columns())
    shelf._search.setText("plate")
    assert shelf.visible_columns() == ["plateID"]
    assert shelf._count.text() == f"1 of {total} columns"
    shelf._search.setText("")
    assert len(shelf.visible_columns()) == total
    assert shelf._count.text() == f"{total} columns"


def test_a_dragged_column_carries_its_name_as_text_as_well(qtbot):
    shelf = gb.ColumnWell()
    qtbot.addWidget(shelf)
    shelf.set_frame(_frame())
    item = shelf._list.item(0)
    payload = shelf._list.mimeData([item])
    name = item.data(Qt.UserRole)
    assert bytes(payload.data(gb.COLUMN_MIME)).decode("utf-8") == name
    assert payload.text() == name


def test_dragging_nothing_carries_nothing(qtbot):
    shelf = gb.ColumnWell()
    qtbot.addWidget(shelf)
    payload = shelf._list.mimeData([])
    assert not payload.hasFormat(gb.COLUMN_MIME)
    assert payload.text() == ""


# ---------------------------------------------------------------------------
# The drop zones
# ---------------------------------------------------------------------------

def test_a_channel_the_builder_does_not_have_is_refused():
    with pytest.raises(ValueError, match="unknown channel"):
        gb.DropZone("depth")


def _column_mime(name):
    mime = QMimeData()
    mime.setData(gb.COLUMN_MIME, name.encode("utf-8"))
    return mime


def test_a_column_dragged_over_a_zone_lights_it_and_drops_into_it(qtbot):
    zone = gb.DropZone(gb.CHANNELS[0])
    qtbot.addWidget(zone)
    seen = []
    zone.column_changed.connect(lambda channel, column: seen.append(
        (channel, column)))
    payload = _column_mime("area")
    where = QPointF(zone.rect().center())

    enter = QDragEnterEvent(where.toPoint(), Qt.CopyAction, payload,
                            Qt.LeftButton, Qt.NoModifier)
    enter.ignore()
    zone.dragEnterEvent(enter)
    assert enter.isAccepted()
    assert zone.property("hovered") is True

    move = QDragMoveEvent(where.toPoint(), Qt.CopyAction, payload,
                          Qt.LeftButton, Qt.NoModifier)
    move.ignore()
    zone.dragMoveEvent(move)
    assert move.isAccepted()

    drop = QDropEvent(where, Qt.CopyAction, payload, Qt.LeftButton,
                      Qt.NoModifier)
    zone.dropEvent(drop)
    assert zone.property("hovered") is False
    assert seen == [(gb.CHANNELS[0], "area")]


def test_a_drag_that_leaves_puts_the_zone_out_again(qtbot):
    zone = gb.DropZone(gb.CHANNELS[0])
    qtbot.addWidget(zone)
    payload = _column_mime("area")
    where = QPointF(zone.rect().center())
    enter = QDragEnterEvent(where.toPoint(), Qt.CopyAction, payload,
                            Qt.LeftButton, Qt.NoModifier)
    zone.dragEnterEvent(enter)
    zone.dragLeaveEvent(QDragLeaveEvent())
    assert zone.property("hovered") is False


def test_a_zone_refuses_a_payload_that_is_not_a_column(qtbot):
    zone = gb.DropZone(gb.CHANNELS[0])
    qtbot.addWidget(zone)
    seen = []
    zone.column_changed.connect(lambda *args: seen.append(args))
    plain = QMimeData()
    plain.setText("area")
    where = QPointF(zone.rect().center())

    enter = QDragEnterEvent(where.toPoint(), Qt.CopyAction, plain,
                            Qt.LeftButton, Qt.NoModifier)
    enter.accept()
    zone.dragEnterEvent(enter)
    assert not enter.isAccepted()

    move = QDragMoveEvent(where.toPoint(), Qt.CopyAction, plain,
                          Qt.LeftButton, Qt.NoModifier)
    move.accept()
    zone.dragMoveEvent(move)
    assert not move.isAccepted()

    drop = QDropEvent(where, Qt.CopyAction, plain, Qt.LeftButton,
                      Qt.NoModifier)
    drop.accept()
    zone.dropEvent(drop)
    assert not drop.isAccepted()
    assert seen == []


# ---------------------------------------------------------------------------
# The canvas
# ---------------------------------------------------------------------------


def test_a_log_axis_keeps_a_floor_above_zero(canvas):
    limits = canvas._limits_for
    assert limits("log", (-5.0, 100.0)) == (100.0 / 1e6, 100.0)
    assert limits("log", (1.0, 100.0)) == (1.0, 100.0)
    assert limits("log", (-5.0, -1.0)) == (-5.0, -1.0)
    assert limits("linear", (-5.0, 100.0)) == (-5.0, 100.0)


def test_a_selection_is_drawn_over_the_histogram_it_belongs_to(canvas):
    canvas.set_spec(GraphSpec(x="area", kind=HISTOGRAM, bins=8),
                    immediate=True)
    plain = len(canvas.figure().axes[0].patches)
    mask = pd.Series(canvas._frame["area"] > canvas._frame["area"].median())
    canvas._draw_histogram(canvas.figure().axes[0], canvas._frame,
                           mask.to_numpy(), gb.active_palette())
    assert len(canvas.figure().axes[0].patches) > plain


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------

def test_a_canvas_whose_link_is_already_gone_still_closes(canvas):
    def explode():
        raise RuntimeError("the link is gone")

    canvas.unlink_selection = explode
    canvas.close()
    assert not canvas._debounce.isActive()


def test_a_redraw_after_the_timer_is_gone_is_discarded(canvas):
    """A queued redraw can arrive after Qt has destroyed the canvas timer."""
    class Dead:
        def isActive(self):
            raise RuntimeError("wrapped C/C++ object has been deleted")

        def start(self, _ms):
            raise RuntimeError("wrapped C/C++ object has been deleted")

        def stop(self):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    surface = canvas._canvas
    surface._spacr_draw_timer = Dead()
    surface._draw_pending = True
    surface.draw_idle()
    assert surface._draw_pending is False

    surface._draw_pending = True
    surface.cancel_pending_draw()
    assert surface._draw_pending is False


def test_a_draw_on_a_dead_surface_is_discarded(canvas):
    surface = canvas._canvas

    def explode():
        raise RuntimeError("wrapped C/C++ object has been deleted")

    surface.draw = explode
    surface._draw_pending = True
    surface._spacr_draw()
    assert surface._draw_pending is False
    surface._spacr_draw()

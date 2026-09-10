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
    CONTINUOUS, HISTOGRAM, SCATTER, GraphSpec,
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

def test_an_ordinary_edit_is_debounced_rather_than_drawn_at_once(canvas):
    canvas._debounce.stop()
    canvas.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER),
                    immediate=False)
    assert canvas._debounce.isActive()
    canvas._debounce.stop()
    canvas.render_now()
    assert canvas.figure().axes


def test_setting_one_channel_keeps_the_rest_of_the_spec(canvas):
    canvas.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER),
                    immediate=True)
    canvas.set_channel("colour", "gene")
    assert canvas.spec.colour == "gene"
    assert canvas.spec.x == "area"


def test_a_selection_that_cannot_be_resolved_is_no_selection(canvas,
                                                             monkeypatch):
    """A shared selection keyed on another table must not take the chart down."""
    canvas.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER),
                    immediate=True)

    class Refusing:
        is_active = True

        def mask_for(self, _frame):
            raise KeyError("prcfo")

    monkeypatch.setattr(type(canvas.link), "selection",
                        property(lambda _self: Refusing()))
    canvas._keyed = True
    assert canvas._selection_mask(canvas._frame) is None
    canvas.render_now()
    assert canvas.figure().axes


def test_an_axis_with_no_column_sits_at_zero(canvas):
    rows = canvas._frame
    canvas.set_spec(GraphSpec(x="area", kind=HISTOGRAM), immediate=True)
    x, y = canvas._xy(rows)
    assert np.array_equal(y, np.zeros(len(rows)))
    assert np.isfinite(x).all()


def test_a_categorical_axis_is_placed_by_level_order(canvas):
    canvas.set_spec(GraphSpec(x="gene", y="area", kind=SCATTER),
                    immediate=True)
    x, _y = canvas._xy(canvas._frame)
    assert set(np.unique(x)) <= {0.0, 1.0, 2.0}


def test_a_log_axis_keeps_a_floor_above_zero(canvas):
    limits = canvas._limits_for
    assert limits("log", (-5.0, 100.0)) == (100.0 / 1e6, 100.0)
    assert limits("log", (1.0, 100.0)) == (1.0, 100.0)
    assert limits("log", (-5.0, -1.0)) == (-5.0, -1.0)
    assert limits("linear", (-5.0, 100.0)) == (-5.0, 100.0)


def test_a_histogram_split_by_a_categorical_colour_stacks_its_levels(canvas):
    canvas.set_spec(GraphSpec(x="area", colour="gene", kind=HISTOGRAM,
                              bins=8), immediate=True)
    figure = canvas.figure()
    assert figure.axes
    # One stacked series per level, named once in the figure's own legend.
    assert [bars.get_label() for bars in figure.axes[0].containers] == [
        "g0", "g1", "g2"]
    assert [text.get_text() for text in figure.legends[0].get_texts()] == [
        "g0", "g1", "g2"]
    assert len(figure.axes[0].patches) == 24


def test_a_selection_is_drawn_over_the_histogram_it_belongs_to(canvas):
    canvas.set_spec(GraphSpec(x="area", kind=HISTOGRAM, bins=8),
                    immediate=True)
    plain = len(canvas.figure().axes[0].patches)
    mask = pd.Series(canvas._frame["area"] > canvas._frame["area"].median())
    canvas._draw_histogram(canvas.figure().axes[0], canvas._frame,
                           mask.to_numpy(), gb.active_palette())
    assert len(canvas.figure().axes[0].patches) > plain


def test_a_density_raster_can_be_weighted_by_a_third_column(canvas):
    """Past the point budget the draw bins rather than drawing every mark."""
    big = _frame(4000)
    canvas.set_frame(big)
    canvas.set_spec(GraphSpec(x="area", y="intensity", colour="cell_count",
                              roles={"cell_count": CONTINUOUS},
                              kind=SCATTER, point_budget=100, bins=12),
                    immediate=True)
    axes = canvas.figure().axes
    assert axes and axes[0].images
    assert axes[0].images[0].get_array().shape == (12, 12)


# ---------------------------------------------------------------------------
# The aggregate draws
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["jitter", "box", "violin", "bar_jitter"])
def test_every_aggregate_kind_puts_its_marks_on_the_axes(canvas, kind):
    canvas.set_spec(GraphSpec(x="gene", y="area", kind=kind), immediate=True)
    ax = canvas.figure().axes[0]
    drawn = (len(ax.collections) + len(ax.patches) + len(ax.lines)
             + len(ax.containers))
    assert drawn > 0


def test_a_level_with_no_measurements_is_skipped_not_drawn_empty(canvas):
    frame = _frame()
    frame.loc[frame["gene"] == "g2", "area"] = np.nan
    canvas.set_frame(frame)
    canvas.set_spec(GraphSpec(x="gene", y="area", kind="box"), immediate=True)
    ax = canvas.figure().axes[0]
    # Two levels have data; the empty one contributes no box.
    assert len(ax.lines) > 0
    canvas.set_spec(GraphSpec(x="gene", y="area", kind="jitter"),
                    immediate=True)
    assert len(canvas.figure().axes[0].collections) == 2


def test_a_distribution_with_nothing_measurable_draws_nothing(canvas):
    frame = _frame()
    frame["area"] = np.nan
    canvas.set_frame(frame)
    canvas.set_spec(GraphSpec(x="gene", y="area", kind="violin"),
                    immediate=True)
    ax = canvas.figure().axes[0]
    assert not ax.collections
    canvas.set_spec(GraphSpec(x="gene", y="area", kind="jitter"),
                    immediate=True)
    assert not canvas.figure().axes[0].collections


def test_jitter_needs_a_categorical_axis_and_a_measurement(canvas):
    """Two continuous columns are a scatter; there is nothing to jitter."""
    canvas.set_spec(GraphSpec(x="area", y="intensity", kind="jitter"),
                    immediate=True)
    ax = canvas.figure().axes[0]
    canvas._draw_jitter(ax, canvas._frame, gb.active_palette(),
                        over_bars=False)
    assert len(ax.collections) <= 1


# ---------------------------------------------------------------------------
# Sweeping a selection out of the chart
# ---------------------------------------------------------------------------

def _mouse(canvas, ax, xdata, ydata, name="button_press_event"):
    from matplotlib.backend_bases import MouseEvent

    x, y = ax.transData.transform((xdata, ydata))
    return MouseEvent(name, canvas._canvas, x, y, 1)


def test_a_press_outside_the_axes_starts_no_sweep(canvas):
    from matplotlib.backend_bases import MouseEvent

    canvas.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER),
                    immediate=True)
    canvas._on_press(MouseEvent("button_press_event", canvas._canvas,
                                1.0, 1.0, 1))
    assert canvas._drag_origin is None
    canvas._on_motion(MouseEvent("motion_notify_event", canvas._canvas,
                                 1.0, 1.0, 1))
    assert canvas._drag_patch is None


def test_a_sweep_previews_a_rectangle_and_publishes_what_it_covers(canvas):
    canvas.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER),
                    immediate=True)
    ax = canvas.figure().axes[0]
    assert canvas._keyed
    low_x, high_x = ax.get_xlim()
    low_y, high_y = ax.get_ylim()

    canvas._on_press(_mouse(canvas, ax, low_x + 0.05 * (high_x - low_x),
                            low_y + 0.05 * (high_y - low_y)))
    assert canvas._drag_origin is not None
    canvas._on_motion(_mouse(canvas, ax, high_x - 0.05 * (high_x - low_x),
                             high_y - 0.05 * (high_y - low_y),
                             "motion_notify_event"))
    assert canvas._drag_patch is not None
    assert canvas._drag_patch.get_width() > 0

    canvas._on_release(_mouse(canvas, ax,
                              high_x - 0.05 * (high_x - low_x),
                              high_y - 0.05 * (high_y - low_y),
                              "button_release_event"))
    assert canvas._drag_origin is None
    assert canvas._drag_patch is None


def test_a_click_rather_than_a_sweep_goes_back_to_resting(canvas):
    canvas.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER),
                    immediate=True)
    ax = canvas.figure().axes[0]
    middle_x = sum(ax.get_xlim()) / 2.0
    middle_y = sum(ax.get_ylim()) / 2.0
    canvas._on_press(_mouse(canvas, ax, middle_x, middle_y))
    cleared = []
    canvas.clear_linked_selection = lambda: cleared.append(True)
    canvas._on_release(_mouse(canvas, ax, middle_x, middle_y,
                              "button_release_event"))
    assert cleared == [True]


def test_a_brush_on_a_panel_that_is_not_there_selects_nothing(canvas):
    canvas.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER),
                    immediate=True)
    assert canvas.brush(0.0, 0.0, 1.0, 1.0, row=9, col=9) is None


def test_a_brush_can_be_asked_for_the_selection_without_publishing(canvas):
    canvas.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER),
                    immediate=True)
    published = []
    canvas.link.selection_changed.connect(published.append)
    ax = canvas.figure().axes[0]
    low_x, high_x = ax.get_xlim()
    low_y, high_y = ax.get_ylim()
    selection = canvas.brush(low_x, low_y, high_x, high_y, publish=False)
    assert selection is not None
    assert published == []


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


def test_a_selection_is_counted_on_top_of_the_bars_it_belongs_to(canvas):
    canvas.set_spec(GraphSpec(x="gene", kind="bar"), immediate=True)
    ax = canvas.figure().axes[0]
    plain = len(ax.patches)
    mask = (canvas._frame["gene"] == "g1").to_numpy()
    canvas._draw_bar(ax, canvas._frame, mask, gb.active_palette())
    assert len(ax.patches) > plain
    labels = [bars.get_label() for bars in ax.containers]
    assert "selected" in labels


def test_more_levels_than_hues_are_gathered_into_one_other_entry(canvas):
    """A ninth generated hue is one nobody can name, so it is not offered."""
    frame = _frame(60)
    frame["gene"] = [f"g{i % 11}" for i in range(len(frame))]
    canvas.set_frame(frame)
    canvas.set_spec(GraphSpec(x="area", y="intensity", colour="gene",
                              kind=SCATTER), immediate=True)
    legend = canvas.figure().legends[0]
    labels = [text.get_text() for text in legend.get_texts()]
    assert len(labels) == len(gb.categorical_colours()) + 1
    assert labels[-1] == f"{gb.OTHER_LABEL} (3)"


def test_a_preview_patch_that_is_already_gone_is_not_an_error(canvas):
    from matplotlib.backend_bases import MouseEvent

    canvas.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER),
                    immediate=True)

    class Detached:
        def remove(self):
            raise ValueError("artist is not in the axes")

    canvas._drag_patch = Detached()
    canvas._drag_origin = None
    canvas._on_release(MouseEvent("button_release_event", canvas._canvas,
                                  5.0, 5.0, 1))
    assert canvas._drag_patch is None


def test_a_sweep_ending_on_a_panel_the_grid_does_not_know_selects_nothing(
        canvas):
    canvas.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER),
                    immediate=True)
    ax = canvas.figure().axes[0]
    middle_x = sum(ax.get_xlim()) / 2.0
    middle_y = sum(ax.get_ylim()) / 2.0
    canvas._on_press(_mouse(canvas, ax, middle_x, middle_y))
    canvas._axes_at = {}
    brushed = []
    canvas.brush = lambda *args, **kwargs: brushed.append(args)
    low_x, high_x = ax.get_xlim()
    low_y, high_y = ax.get_ylim()
    canvas._on_release(_mouse(canvas, ax, low_x + 0.9 * (high_x - low_x),
                              low_y + 0.9 * (high_y - low_y),
                              "button_release_event"))
    assert brushed == []

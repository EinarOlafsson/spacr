"""The edges seven small widgets only reach when something is missing.

Every case below is one of "the thing this code usually has is not there
this time", and each is a real situation rather than a hypothetical:

* **Field fade** — a paint event for a field whose painter Qt refuses to
  begin (nothing to paint on), and an uninstall after the QApplication has
  already gone.
* **EdgeDrawer** — arming over a drawer that is already open, a panel the
  locked dock has re-parented away, and a close timeout that lands after
  the pin went on.
* **Timelapse movie** — a filmstrip holding a layout item that is not a
  widget, a canvas laid out too small to scale into, and collapsing the
  strip.
* **Trellis** — a grid whose panels all have no rows, and a brush on a grid
  faceted one way only.
* **Pivot builder** — a drop carrying an empty column name, a cell above
  the low-n mark, and clearing the table without losing the axes.
* **Toggle** — a press-and-twitch that is a tap, not a drag.
* **LivePreviewContract** — the busy state on a panel whose buttons are not
  built yet.

The one arc these cannot reach is proved rather than driven: see
``test_a_panel_of_missing_values_still_tops_its_count_axis``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt
from PySide6.QtGui import QDropEvent, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QWidget
from shiboken6 import isValid


# ---------------------------------------------------------------------------
# Field fade: a painter that never began, and an app that went first
# ---------------------------------------------------------------------------

def test_a_painter_that_could_not_begin_neither_paints_nor_ends(qapp,
                                                                monkeypatch):
    """A paint event can arrive for a field Qt will not give a painter for.

    ``QPainter(widget)`` fails — quietly, returning an inactive painter —
    whenever the widget has no usable paint device: a field being torn
    down, or one that is not on a backing store. The filter must then paint
    nothing and, above all, must not call ``end()`` on a painter that never
    began, and it must still return False so the field draws its own text.

    The fake painter is the same device ``test_field_fade`` uses for this
    class: it makes "Qt declined to begin" an ordinary object rather than a
    state that can only be produced by tearing a widget in half.
    """
    from spacr.qt.widgets import field_fade as ff

    painted, ended = [], []

    class _Painter:
        began = False

        def __init__(self, device):
            self.device = device

        def isActive(self):  # noqa: N802 - QPainter contract
            return type(self).began

        def end(self):
            ended.append(self.device)

    monkeypatch.setattr(ff, "QPainter", _Painter)
    monkeypatch.setattr(ff, "paint_field_fade",
                        lambda obj, _painter: painted.append(obj))
    monkeypatch.setattr(ff, "field_fade_enabled", lambda: True)

    field = QLineEdit()
    assert ff.fades(field), "the fixture must be a field the filter paints"
    # The filter class rather than the installed singleton: this is one
    # event, not an app-wide install, and the module global is shared with
    # every other test in the process.
    event_filter = ff._FieldFadeFilter()

    handled = event_filter.eventFilter(field, QEvent(QEvent.Type.Paint))

    assert handled is False, "the field still has to draw its own text"
    assert painted == [], "nothing can be painted through a dead painter"
    assert ended == [], "end() on a painter that never began is a crash"

    # And the same event with a painter Qt did begin: the ramp is painted
    # and the painter is closed explicitly, before the field paints on top.
    _Painter.began = True
    assert event_filter.eventFilter(field, QEvent(QEvent.Type.Paint)) is False
    assert painted == [field]
    assert ended == [field]


def test_the_hook_is_forgotten_when_the_application_went_first(qapp,
                                                               monkeypatch):
    """Uninstalling after the QApplication has gone must not wedge the hook.

    At interpreter shutdown the application can be destroyed before
    anything gets round to removing the filter. ``uninstall_field_fade``
    then has nothing to remove it *from*, and the one thing it must still
    do is drop its own reference — otherwise the next application starts
    with a filter the module thinks is already installed, and no field is
    ever painted again.
    """
    from spacr.qt.widgets import field_fade as ff

    saved = ff._filter
    ff.uninstall_field_fade(qapp)
    try:
        assert ff.install_field_fade(qapp) is True
        orphan = ff._filter
        assert orphan is not None

        monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))
        assert ff.uninstall_field_fade() is True
        assert ff._filter is None

        # Not wedged: a later install builds a fresh filter rather than
        # reporting the dead one as already there.
        monkeypatch.undo()
        assert ff.install_field_fade(qapp) is True
        assert ff._filter is not orphan
    finally:
        # The orphan was never removed from the app — that is the whole
        # point — so remove it here rather than leaving it filtering every
        # paint event for the rest of the session.
        qapp.removeEventFilter(orphan)
        ff.uninstall_field_fade(qapp)
        ff._filter = saved
        if saved is not None:
            qapp.installEventFilter(saved)


# ---------------------------------------------------------------------------
# EdgeDrawer
# ---------------------------------------------------------------------------

@pytest.fixture
def drawer(qtbot):
    """A drawer over an 800×600 host, with a 220 px panel."""
    from spacr.qt.widgets.drawer import EdgeDrawer

    host = QWidget()
    host.resize(800, 600)
    qtbot.addWidget(host)
    panel = QWidget()
    panel.resize(220, 400)
    widget = EdgeDrawer(host, panel)
    qtbot.addWidget(widget)
    return widget


def test_arming_over_an_open_drawer_cancels_the_close_but_does_not_re_arm(
        drawer):
    """Re-entering the strip of an open drawer must not restart the dwell.

    The pointer crossing the hot strip on its way back into an open panel
    is the commonest event this widget sees. It has to cancel the pending
    close — otherwise the panel shuts under the pointer — and it must not
    start the open timer, which would fire :meth:`open` a second time on a
    drawer that is already open and restart the slide.
    """
    drawer.open()
    drawer.schedule_close()
    assert drawer._close_timer.isActive(), "the grace period must be running"

    drawer.arm()

    assert drawer.is_open()
    assert not drawer._close_timer.isActive(), "re-entry cancels the close"
    assert not drawer._open_timer.isActive(), (
        "an open drawer has nothing to dwell for")

    # Closed, the same gesture is the one that opens it: the dwell starts.
    drawer.close()
    drawer.arm()
    assert drawer._open_timer.isActive()
    drawer.disarm()


def test_a_panel_the_dock_took_away_is_not_resized_by_the_drawer(drawer,
                                                                 qtbot):
    """Once the locked dock owns the sidebar, the drawer stops sizing it.

    The locked dock re-parents the same panel into the window's layout.
    A drawer that kept resizing it would fight that layout on every open —
    the column snapping to the drawer's width and back again.
    """
    elsewhere = QWidget()
    elsewhere.resize(300, 300)
    qtbot.addWidget(elsewhere)
    panel = drawer._panel
    panel.setParent(elsewhere)
    panel.resize(111, 77)

    drawer.relayout_for_open()

    assert panel.size().toTuple() == (111, 77), (
        "a panel in somebody else's layout keeps the size that layout gave it")
    assert drawer.size().toTuple() == (220, 600), "the drawer still re-fits"

    # Taken back, it is the drawer's to size again.
    drawer.adopt(panel)
    panel.resize(111, 77)
    drawer.relayout_for_open()
    assert panel.size().toTuple() == (220, 600)


def test_a_close_timeout_that_lands_after_the_pin_leaves_the_drawer_open(
        drawer):
    """A timeout already in flight must not close a drawer that got pinned.

    :meth:`hold` stops the close timer, so the only way into this guard is
    a timeout that had already been emitted when the pin went on — a click
    landing in the panel in the last milliseconds of the grace period. The
    signal is emitted directly here because that race is not something a
    test can win by waiting for it.
    """
    drawer.open()
    drawer.schedule_close()
    drawer.hold(True)

    drawer._close_timer.timeout.emit()

    assert drawer.is_open(), "a pinned drawer stays open"
    assert drawer.is_held()

    # Unpinned, the very same timeout closes it.
    drawer.hold(False)
    drawer._close_timer.timeout.emit()
    assert not drawer.is_open()


# ---------------------------------------------------------------------------
# The timelapse movie
# ---------------------------------------------------------------------------

def _sequence(frames: int = 3, size: int = 48) -> np.ndarray:
    images = np.zeros((frames, size, size), np.uint16)
    for t in range(frames):
        images[t, 10:20, 6 + 3 * t:16 + 3 * t] = 900
    return images


@pytest.fixture
def movie(qtbot):
    from spacr.qt.widgets.timelapse_movie import FovMovie

    widget = FovMovie("field 1")
    qtbot.addWidget(widget)
    widget.set_sequence(_sequence())
    return widget


def test_the_filmstrip_clears_a_layout_item_that_is_not_a_widget(qtbot):
    """Emptying the strip must survive a spacer as well as a cell.

    The row is a public layout on a public widget, so what it holds is not
    only the cells this class put there. Taking an item and calling
    ``deleteLater`` on ``item.widget()`` without checking would raise on
    the first spacer and leave half a strip on screen.
    """
    from spacr.qt.widgets.timelapse_movie import FilmStrip

    strip = FilmStrip()
    qtbot.addWidget(strip)
    red, blue = QPixmap(20, 20), QPixmap(20, 20)
    red.fill(Qt.red)
    blue.fill(Qt.blue)
    strip.set_frames([red, blue])

    row = strip.widget().layout()
    assert row.count() == 3, "two cells and the trailing stretch"
    old_cells = [row.itemAt(i).widget() for i in range(2)]
    row.insertSpacing(0, 12)

    strip.set_frames([blue])

    assert row.count() == 2, "one cell and the stretch — the spacer is gone"
    assert row.itemAt(0).widget() is not None
    assert row.itemAt(0).widget() not in old_cells
    # The widget items were not merely dropped from the layout: they were
    # deleted, which is the branch the spacer skips.
    qtbot.waitUntil(lambda: not any(isValid(cell) for cell in old_cells))


def test_a_canvas_too_small_to_scale_into_shows_the_frame_unscaled(movie):
    """A collapsed pane must not ask for a zero-sized scale.

    A splitter dragged shut, or a panel measured before its first layout,
    leaves the canvas a few pixels across. Scaling into that produces a
    pixmap with no picture in it, so below the floor the frame goes up at
    its own size and the label clips it.
    """
    # The canvas' minimum height is what a laid-out panel gets; lifting it
    # is the only way to produce the geometry a collapsed pane really has.
    movie._canvas.setMinimumSize(0, 0)
    movie._canvas.resize(4, 4)

    movie.show_frame(1)

    assert movie._canvas.pixmap().size().toTuple() == (48, 48), (
        "the frame goes up at its own size rather than scaled into nothing")

    # Given room, the same frame is scaled to fit it.
    movie._canvas.resize(300, 300)
    movie.show_frame(1)
    assert movie._canvas.pixmap().size().toTuple() == (300, 300)


def test_collapsing_the_strip_does_not_re_ring_a_frame(movie):
    """Only opening the strip highlights; closing it just goes away."""
    rung = []
    original = movie._strip.highlight
    movie._strip.highlight = lambda index: rung.append(index) or original(index)

    movie.show_frame(2)
    rung.clear()

    movie.set_strip_open(True)
    assert movie.strip_is_open()
    assert rung == [2], "opening rings the frame the movie is on"

    movie.set_strip_open(False)
    assert not movie.strip_is_open()
    assert rung == [2], "a hidden strip has nothing to ring"


# ---------------------------------------------------------------------------
# The trellis
# ---------------------------------------------------------------------------

def _hand() -> pd.DataFrame:
    return pd.DataFrame({
        "plateID": ["p1", "p1", "p1", "p2", "p2", "p2"],
        "gene": ["a", "a", "b", "a", "a", "a"],
        "area": [10.0, 20.0, 100.0, 30.0, 40.0, 50.0],
    })


def _rows_only_spec():
    from spacr.qt.widgets.graph_spec import CONTINUOUS, GraphSpec
    from spacr.qt.widgets.trellis_spec import TrellisSpec

    return TrellisSpec(graph=GraphSpec(x="area", facet_row="plateID", bins=2,
                                       roles={"area": CONTINUOUS}))


def test_a_grid_where_every_panel_is_empty_says_nothing_about_n():
    """"n per panel 0–0" would be a fact about nothing.

    A grid drawn from ``levels_source`` keeps a panel for a level that drew
    no rows — that is the whole point of it — so a filter that removed
    everything leaves every panel empty. The summary has to say that they
    are empty and then stop, because there is no n to spread.
    """
    from spacr.qt.widgets.trellis_spec import trellis

    hand = _hand()
    spec = _rows_only_spec()

    empty = trellis(hand.iloc[0:0], spec, levels_source=hand)

    assert empty.n_range() is None
    assert empty.summary() == "2 × 1 = 2 panels · 2 with no rows"

    # The same grid over the rows themselves does carry the spread.
    assert "n = 3 each" in trellis(hand, spec).summary()


def test_a_brush_on_a_row_only_facet_keeps_the_other_row_out():
    """Sweeping one panel is a predicate on that panel's rows, not the axes.

    With only one facet set the other level is ``None`` and there is no
    column to test it against; membership then rests on the row facet
    alone. Getting that wrong selects every row in the x range, from every
    panel — which is how a brush in one small multiple lights up the rest.
    """
    from spacr.qt.widgets.trellis_spec import trellis

    hand = _hand()
    result = trellis(hand, _rows_only_spec())
    assert result.shape == (2, 1)

    # 5..45 covers p1's 10 and 20 as well as p2's 30 and 40. Swept on the
    # p2 panel it must take p2's two rows and neither of p1's.
    mask = result.brush(5.0, -1e9, 45.0, 1e9, row=1, col=0)

    assert mask.tolist() == [False, False, False, True, True, False]
    assert hand.loc[mask, "area"].tolist() == [30.0, 40.0]


def test_a_panel_of_missing_values_still_tops_its_count_axis():
    """A panel whose category is missing everywhere counts as one level.

    ``_level_series`` folds NaN into ``MISSING_LEVEL`` rather than dropping
    it, so "no gene recorded" is a bar like any other and the shared count
    axis has to clear it.

    PROOF, and why the ``return 0.0`` at ``trellis_spec.py:451`` is
    unreachable: ``_panel_top`` returns at line 438 unless
    ``panel.occupied and panel.n != 0``, and ``TrellisPanel.n`` is
    ``len(self.index)`` (line 284), so ``rows = frame.iloc[panel.index]``
    at 439 is never empty. Its only caller is ``trellis()`` at line 560,
    which passes the panels ``_place`` built over that same frame. For
    ``kind == BAR`` the value counted is ``_level_series(rows,
    column).value_counts()``, and ``_level_series``
    (``graph_spec.py:531``) maps *every* row to a string — NaN included —
    so a non-empty ``rows`` always yields at least one count and the
    ``if len(counts):`` at 449 cannot be false. The assertion below is that
    fact measured: were the missing-level panel to fall through to 0.0, the
    shared limit would be 1 × 1.08 rather than 2 × 1.08.
    """
    from spacr.qt.widgets.graph_spec import BAR, CATEGORICAL, GraphSpec
    from spacr.qt.widgets.trellis_spec import TrellisSpec, trellis

    frame = pd.DataFrame({"plateID": ["p1", "p1", "p2", "p2"],
                          "gene": ["a", "b", None, None]})
    spec = TrellisSpec(graph=GraphSpec(x="gene", facet_row="plateID",
                                       kind=BAR,
                                       roles={"gene": CATEGORICAL}))

    result = trellis(frame, spec)

    assert [panel.n for panel in result.panels] == [2, 2]
    assert result.panels[1].scales.count_limit == pytest.approx(2 * 1.08)


# ---------------------------------------------------------------------------
# The pivot builder
# ---------------------------------------------------------------------------

def _pivot_frame() -> pd.DataFrame:
    """Eight objects on p1 and two on p2 — one cell either side of low-n."""
    return pd.DataFrame({
        "plateID": ["p1"] * 8 + ["p2"] * 2,
        "gene": ["a"] * 8 + ["b"] * 2,
        "area": [float(v) for v in range(10, 90, 10)] + [5.0, 7.0],
    })


@pytest.fixture
def pivot_panel(qtbot):
    from spacr.qt.widgets.pivot_builder import PivotPanel

    widget = PivotPanel()
    qtbot.addWidget(widget)
    widget.set_frame(_pivot_frame())
    return widget


def _drop(well, payload: str) -> QDropEvent:
    """Drop ``payload`` on a well exactly as the drag-and-drop does."""
    from spacr.qt.widgets.graph_builder import COLUMN_MIME

    data = QMimeData()
    data.setData(COLUMN_MIME, payload.encode("utf-8"))
    event = QDropEvent(QPointF(4, 4), Qt.CopyAction, data,
                       Qt.LeftButton, Qt.NoModifier)
    well._list.dropEvent(event)
    return event


def test_a_drop_carrying_no_column_name_is_still_consumed(pivot_panel):
    """An empty payload adds no axis, and does not fall through to Qt.

    The mime type is right — it came from the column list — so the well is
    the widget that owns this drop and has to accept it. Leaving it
    unaccepted hands the drop to whatever is underneath, which is the page.
    """
    from spacr.qt.widgets.pivot_builder import AXIS_ROWS

    well = pivot_panel.wells[AXIS_ROWS]

    event = _drop(well, "")

    assert well.columns() == (), "an unnamed column is not an axis"
    assert event.isAccepted(), "the well still owns the drop"

    # A named column on the same well does become an axis.
    named = _drop(well, "plateID")
    assert well.columns() == ("plateID",)
    assert named.isAccepted()


def test_only_a_cell_at_or_below_the_low_n_mark_is_called_an_anecdote(
        pivot_panel):
    """The warning belongs on the cells it is about and nowhere else.

    A caveat printed on every cell is a caveat nobody reads. p1 aggregates
    eight objects and p2 two, so one tooltip carries the sentence and the
    other must not.
    """
    from spacr.qt.widgets.pivot_builder import AXIS_ROWS, AXIS_VALUES

    _drop(pivot_panel.wells[AXIS_ROWS], "plateID")
    _drop(pivot_panel.wells[AXIS_VALUES], "area")
    pivot_panel.recompute()
    result = pivot_panel.result
    assert result.row_levels[0] == ("p1",) and result.sizes[0, 0] == 8
    offset = len(result.row_keys)

    healthy = pivot_panel.table.item(0, offset).toolTip()
    thin = pivot_panel.table.item(1, offset).toolTip()

    assert "8 source row(s)" in healthy
    assert "anecdote" not in healthy
    assert "anecdote" in thin, "two objects is exactly what the mark is for"


def test_clearing_the_table_keeps_axes_a_different_table_would_lose(
        pivot_panel):
    """No frame is "not yet", not "these columns are wrong".

    Axes are dropped when the *new* table cannot group by them — a pivot
    half-resolved against the wrong frame would group by fewer keys than
    the wells claim. Clearing the panel is a different thing: the columns
    have to still be there when a table comes back.
    """
    from spacr.qt.widgets.pivot_builder import AXIS_COLS, AXIS_ROWS

    _drop(pivot_panel.wells[AXIS_ROWS], "plateID")
    _drop(pivot_panel.wells[AXIS_COLS], "gene")

    pivot_panel.set_frame(None)

    assert pivot_panel.wells[AXIS_ROWS].columns() == ("plateID",)
    assert pivot_panel.wells[AXIS_COLS].columns() == ("gene",)
    assert pivot_panel.result is None

    # A table without ``gene`` does drop it, which is the case the guard
    # above is not.
    pivot_panel.set_frame(_pivot_frame().drop(columns=["gene"]))
    assert pivot_panel.wells[AXIS_COLS].columns() == ()
    assert pivot_panel.wells[AXIS_ROWS].columns() == ("plateID",)


# ---------------------------------------------------------------------------
# The toggle
# ---------------------------------------------------------------------------

def test_a_twitch_under_three_pixels_is_a_tap_and_not_a_drag(qtbot):
    """A pointer that moves two pixels while pressed has not dragged.

    Every press moves a little. Treating the first pixel of movement as a
    drag would leave the knob stranded wherever the pointer happened to be
    when the button came up, instead of flipping the switch.
    """
    from spacr.qt.widgets.toggle import Toggle

    widget = Toggle()
    qtbot.addWidget(widget)
    widget.resize(60, 24)
    widget.show()
    qtbot.waitExposed(widget)
    resting = widget._knob_pos
    start = int(widget._minimum_knob_x() + 6)
    middle = widget.height() // 2

    QTest.mousePress(widget, Qt.LeftButton, pos=QPoint(start, middle))
    QTest.mouseMove(widget, QPoint(start + 2, middle))

    assert widget._dragging is False, "two pixels is a hand, not a gesture"
    assert widget._knob_pos == resting, "the knob has not left its side"

    # Three pixels is the threshold, and past it the knob follows.
    QTest.mouseMove(widget, QPoint(start + 20, middle))
    assert widget._dragging is True
    assert widget._knob_pos > resting


# ---------------------------------------------------------------------------
# The live-preview contract
# ---------------------------------------------------------------------------

def test_the_busy_state_skips_buttons_a_panel_has_not_built_yet(qtbot):
    """A panel can be marked busy before, or without, its buttons.

    ``LivePreviewContract`` is mixed into a QWidget that supplies the
    buttons itself, so between the mixin's methods becoming callable and
    the panel's ``_build`` running there is a window in which neither
    button exists — and a live view that never offers cancellation has no
    cancel button at all. Marking such a panel busy must not invent the
    attributes, and must not raise.
    """
    from spacr.qt.widgets.preview_contract import LivePreviewContract

    class _Panel(QWidget, LivePreviewContract):
        pass

    panel = _Panel()
    qtbot.addWidget(panel)

    panel.set_preview_busy(True)

    assert not hasattr(panel, "_run_btn")
    assert not hasattr(panel, "_cancel_btn")

    # Once the panel has built them, the same call drives exactly one of
    # the two: run off while a pass is in flight, cancel on.
    panel._run_btn = QPushButton("Run preview", panel)
    panel._cancel_btn = QPushButton("Cancel", panel)

    panel.set_preview_busy(True)
    assert panel._run_btn.isEnabled() is False
    assert panel._cancel_btn.isEnabled() is True

    panel.set_preview_busy(False)
    assert panel._run_btn.isEnabled() is True
    assert panel._cancel_btn.isEnabled() is False

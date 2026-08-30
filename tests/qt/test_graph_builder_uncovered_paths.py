"""The canvas when there is nothing, or not enough, to draw with.

``matplotlib`` renamed ``vert`` to ``orientation`` in 3.10, so the canvas
reads the installed version to decide which keyword the box and violin calls
take. A version string that does not parse must not take the render down with
it — the argument is spelling, not science, and the newer spelling is the one
to guess at.

``EMPTY`` is a member of :data:`~spacr.qt.widgets.graph_spec.PLOT_KINDS` that
has no marks of its own: it means "nothing has been dropped yet". The render
path turns it into a message before it reaches the panel drawing, so the
drawing itself has to be a no-op for it rather than fall through into
whichever branch happens to be last.

The rest are the states around an empty chart: a table taken away, a spec
dropped before a table arrives, a filter that leaves no rows, a bar whose
other channel holds no numbers to average, and the axes of a canvas that
deliberately does not rescale when a filter narrows it — the Gate Editor's
rule, because rescaling moves the ground out from under a drawn gate.

Two more are about what the widget puts on screen rather than in the figure:
the page surface painted under a transparent figure, and the single preview
rectangle a sweep resizes rather than re-adds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("matplotlib")

from matplotlib.figure import Figure                             # noqa: E402

from spacr.qt.linked_selection import LinkedSelection            # noqa: E402
from spacr.selection import DataFilter, RangeFilter              # noqa: E402
from spacr.qt.widgets import graph_builder as gb                 # noqa: E402
from spacr.qt.widgets.graph_spec import (                        # noqa: E402
    EMPTY, PLOT_KINDS, SCATTER, GraphSpec,
)

pytestmark = pytest.mark.qt


def _frame(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "plateID": [f"p{i % 2 + 1}" for i in range(rows)],
        "rowID": [f"r{i % 3 + 1}" for i in range(rows)],
        "columnID": [f"c{i % 4 + 1}" for i in range(rows)],
        "fieldID": [f"f{i % 2 + 1}" for i in range(rows)],
        "object_label": list(range(rows)),
        "area": rng.normal(100.0, 10.0, rows),
        "intensity": rng.normal(size=rows),
    })


@pytest.fixture
def canvas(qtbot):
    view = gb.GraphCanvas(link=LinkedSelection(), source="builders")
    qtbot.addWidget(view)
    view.set_frame(_frame())
    view.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER))
    return view


@pytest.mark.parametrize("version", ["3", "unreleased", "3.x", ""])
def test_a_matplotlib_version_that_does_not_parse_is_read_as_the_new_spelling(
        monkeypatch, version):
    """An unparseable version picks ``orientation``, and does not raise.

    Guessing the modern keyword is the safe half of the choice: the old one
    is deprecated, so guessing it on a new matplotlib prints a warning for
    every panel of every redraw.
    """
    import matplotlib

    monkeypatch.setattr(matplotlib, "__version__", version)

    assert gb._orientation(True) == {"orientation": "vertical"}
    assert gb._orientation(False) == {"orientation": "horizontal"}


def test_a_parseable_version_is_still_read_as_the_version_it_is(monkeypatch):
    """The fallback did not swallow the comparison it is a fallback for."""
    import matplotlib

    monkeypatch.setattr(matplotlib, "__version__", "3.9.4")
    assert gb._orientation(True) == {"vert": True}
    monkeypatch.setattr(matplotlib, "__version__", "3.10.0")
    assert gb._orientation(True) == {"orientation": "vertical"}


def test_the_empty_kind_is_a_plot_kind_that_has_no_marks(canvas):
    """Why the panel drawing needs a branch for a kind nobody can pick.

    ``EMPTY`` sits in :data:`PLOT_KINDS` beside kinds that all draw
    something, so the claim its name makes is a claim about every other
    member too: given rows every kind can handle — a categorical column
    against a continuous one — it is the only one that puts nothing on the
    axes.
    """
    assert EMPTY in PLOT_KINDS
    canvas.set_spec(GraphSpec(x="rowID", y="area", kind=SCATTER))
    data = canvas.render_data
    rows = data.frame
    mask = np.zeros(len(rows), dtype=bool)

    bare = set()
    for kind in PLOT_KINDS:
        ax = Figure().add_subplot(111)
        canvas._draw_panel_marks(ax, rows, mask, kind, data,
                                 gb.active_palette())
        artists = (list(ax.collections) + list(ax.lines) + list(ax.patches)
                   + list(ax.images) + list(ax.texts))
        if not artists:
            bare.add(kind)

    assert bare == {EMPTY}


def test_drawing_a_panel_of_the_empty_kind_leaves_the_axes_untouched(canvas):
    """No artists, and no highlight updater for the renderer to hold on to.

    Falling through to another kind's branch would draw a scatter under the
    "drag a column onto X or Y" message.
    """
    data = canvas.render_data
    assert data is not None and not data.frame.empty
    rows = data.frame
    mask = np.zeros(len(rows), dtype=bool)
    ax = Figure().add_subplot(111)

    updater = canvas._draw_panel_marks(ax, rows, mask, EMPTY, data,
                                       gb.active_palette())

    assert updater is None
    assert list(ax.collections) == []
    assert list(ax.lines) == []
    assert list(ax.patches) == []
    assert list(ax.images) == []
    assert list(ax.texts) == []


def test_drawing_the_same_rows_as_a_scatter_does_put_marks_on_the_axes(canvas):
    """The contrast that makes the empty case an assertion about behaviour."""
    data = canvas.render_data
    rows = data.frame
    mask = np.zeros(len(rows), dtype=bool)
    ax = Figure().add_subplot(111)

    updater = canvas._draw_panel_marks(ax, rows, mask, SCATTER, data,
                                       gb.active_palette())

    assert list(ax.collections)
    assert callable(updater)


def test_a_panel_with_no_rows_says_so_instead_of_drawing_an_empty_axes(canvas):
    """An empty facet is labelled, not silently blank."""
    data = canvas.render_data
    rows = data.frame.iloc[0:0]
    ax = Figure().add_subplot(111)

    updater = canvas._draw_panel_marks(ax, rows, None, SCATTER, data,
                                       gb.active_palette())

    assert updater is None
    assert [t.get_text() for t in ax.texts] == ["no rows"]


def test_taking_the_table_away_leaves_the_invitation_to_load_one(qtbot):
    """A canvas whose table is removed goes back to its empty state.

    Not to the last chart it drew: the panels, the shared scales and the
    column kinds all belonged to a table that is gone, and leaving them on
    screen shows a picture of data the user has just closed.
    """
    view = gb.GraphCanvas(link=LinkedSelection(), source="builders")
    qtbot.addWidget(view)
    view.set_frame(_frame())
    view.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER))
    assert view.panel_axes()

    view.set_frame(None)

    assert view.panel_axes() == {}
    assert view.kinds == {}
    assert [t.get_text() for t in view.figure().axes[0].texts] == [
        "Load a table, then drag a column onto X or Y."]


def test_a_spec_dropped_on_a_canvas_with_no_table_is_kept_but_draws_nothing(
        qtbot):
    """The spec survives, so it applies the moment a table is loaded."""
    view = gb.GraphCanvas(link=LinkedSelection(), source="builders")
    qtbot.addWidget(view)

    view.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER))

    assert view.spec.x == "area"
    assert view.kinds == {}
    assert view.panel_axes() == {}
    assert [t.get_text() for t in view.figure().axes[0].texts] == [
        "Load a table, then drag a column onto X or Y."]

    view.set_frame(_frame())

    assert view.panel_axes()
    assert view.kinds["area"]


def test_a_canvas_that_does_not_rescale_keeps_the_whole_tables_axes(qtbot):
    """The Gate Editor's rule: a filter must not move the ground.

    Rescaling to the rows a gate kept reads as the plot zooming into the
    gate, and it moves the axes out from under the outline still drawn on
    them.
    """
    link = LinkedSelection()
    view = gb.GraphCanvas(link=link, source="builders")
    qtbot.addWidget(view)
    frame = _frame()
    frame["area"] = np.linspace(0.0, 100.0, len(frame))
    view.set_frame(frame)
    view.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER))
    unfiltered = view.axes_at(0, 0).get_xlim()

    link.set_filter(DataFilter().add(RangeFilter("area", low=80.0)))
    view.render_now()
    rescaled = view.axes_at(0, 0).get_xlim()

    view.RESCALE_ON_FILTER = False
    view.render_now()
    kept = view.axes_at(0, 0).get_xlim()

    assert rescaled[0] > unfiltered[0] + 50.0
    assert kept == unfiltered
    assert view.render_data.frame["area"].min() >= 80.0


def test_the_canvas_paints_the_page_surface_under_the_figure(qtbot,
                                                            monkeypatch):
    """The figure patch is transparent, so something has to draw the panel."""
    painted = []
    real_paint_panel = gb.paint_panel

    def recording(painter, widget, **kwargs):
        painted.append(kwargs)
        return real_paint_panel(painter, widget, **kwargs)

    monkeypatch.setattr(gb, "paint_panel", recording)
    inner = gb._canvas_class()(Figure(figsize=(2.0, 1.5)))
    qtbot.addWidget(inner)
    inner.resize(120, 90)

    pixmap = inner.grab()

    assert inner._spacr_panel is True
    assert not pixmap.isNull()
    assert painted and painted[0]["role"] == "surface"


def test_a_canvas_already_sitting_on_a_panel_does_not_paint_a_second_one(
        qtbot, monkeypatch):
    """Two stacked surfaces reach an opacity no slider position can."""
    painted = []
    monkeypatch.setattr(gb, "paint_panel",
                        lambda *args, **kwargs: painted.append(kwargs))
    inner = gb._canvas_class()(Figure(figsize=(2.0, 1.5)), panel=False)
    qtbot.addWidget(inner)
    inner.resize(120, 90)

    pixmap = inner.grab()

    assert inner._spacr_panel is False
    assert not pixmap.isNull()
    assert painted == []


class _MouseEvent:
    """A matplotlib mouse event with the fields the drag handlers read."""

    def __init__(self, ax, x_data, y_data):
        self.inaxes = ax
        self.xdata, self.ydata = x_data, y_data
        self.button = 1


def test_a_bar_against_a_column_with_no_numbers_in_it_counts_instead_of_averaging(
        qtbot):
    """A mean of nothing is not a bar; the honest answer is the count.

    The other channel is numeric only when the user put a measurement there.
    Averaging a column that is entirely missing would draw three bars at NaN
    — three gaps where the reader expects three groups.
    """
    view = gb.GraphCanvas(link=LinkedSelection(), source="builders")
    qtbot.addWidget(view)
    frame = _frame(24)
    frame["gene"] = [f"g{i % 3}" for i in range(24)]
    frame["blank"] = [float("nan")] * 24
    view.set_frame(frame)

    view.set_spec(GraphSpec(x="gene", y="blank", kind="bar"))

    heights = sorted(patch.get_height()
                     for patch in view.axes_at(0, 0).patches)
    assert heights == [8.0, 8.0, 8.0]


def test_a_bar_against_a_measurement_draws_the_mean_of_that_measurement(qtbot):
    """The contrast: with numbers to average, the bars are means."""
    view = gb.GraphCanvas(link=LinkedSelection(), source="builders")
    qtbot.addWidget(view)
    frame = _frame(24)
    frame["gene"] = ["g0"] * 12 + ["g1"] * 12
    frame["area"] = [10.0] * 12 + [30.0] * 12
    view.set_frame(frame)

    view.set_spec(GraphSpec(x="gene", y="area", kind="bar"))

    heights = sorted(patch.get_height()
                     for patch in view.axes_at(0, 0).patches)
    assert heights == [10.0, 30.0]


def test_categorical_axes_are_ticked_but_not_pinned_when_sharing_is_off(qtbot):
    """Ticks name the levels either way; only sharing fixes the limits.

    A panel that is not sharing its axes is allowed to frame its own rows,
    and forcing the full level range onto it would leave empty margins in
    every panel that has fewer.
    """
    view = gb.GraphCanvas(link=LinkedSelection(), source="builders")
    qtbot.addWidget(view)
    frame = _frame(24)
    frame["gene"] = [f"g{i % 3}" for i in range(24)]
    view.set_frame(frame)

    view.set_spec(GraphSpec(x="gene", y="area", kind="box",
                            shared_x=False, shared_y=False))
    loose = view.axes_at(0, 0)
    loose_ticks = [t.get_text() for t in loose.get_xticklabels()]
    loose_limits = loose.get_xlim()

    view.set_spec(GraphSpec(x="gene", y="area", kind="box",
                            shared_x=True, shared_y=True))
    shared = view.axes_at(0, 0)

    assert loose_ticks == ["g0", "g1", "g2"]
    assert [t.get_text() for t in shared.get_xticklabels()] == loose_ticks
    assert shared.get_xlim() == (-0.6, 2.6)
    assert loose_limits != shared.get_xlim()


def test_a_single_panel_emptied_by_a_filter_says_no_rows_and_carries_no_count(
        qtbot):
    """An unfaceted chart with nothing left is labelled, not headed "n=0"."""
    link = LinkedSelection()
    view = gb.GraphCanvas(link=link, source="builders")
    qtbot.addWidget(view)
    frame = _frame(24)
    frame["area"] = np.linspace(0.0, 100.0, 24)
    view.set_frame(frame)
    view.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER))

    link.set_filter(DataFilter().add(RangeFilter("area", low=1e9)))
    view.render_now()
    ax = view.axes_at(0, 0)

    assert [t.get_text() for t in ax.texts] == ["no rows"]
    assert ax.get_title(loc="right") == ""
    assert ax.get_title() == ""
    assert "no rows" in view.notice()


def test_dragging_on_reuses_the_one_preview_rectangle_it_already_added(qtbot):
    """One patch per sweep, resized — not a new one per mouse move.

    A patch added on every motion event leaves a trail of rectangles on the
    axes, and only the last of them is ever removed on release.
    """
    view = gb.GraphCanvas(link=LinkedSelection(), source="builders")
    qtbot.addWidget(view)
    view.set_frame(_frame(24))
    view.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER))
    ax = view.axes_at(0, 0)
    before = len(ax.patches)

    view._on_press(_MouseEvent(ax, 90.0, -1.0))
    view._on_motion(_MouseEvent(ax, 100.0, 0.0))
    first = view._drag_patch
    view._on_motion(_MouseEvent(ax, 110.0, 1.0))

    assert first is not None
    assert view._drag_patch is first
    assert len(ax.patches) == before + 1
    assert first.get_width() == pytest.approx(20.0)
    assert first.get_height() == pytest.approx(2.0)

    view._on_release(_MouseEvent(ax, 110.0, 1.0))
    assert view._drag_patch is None
    assert len(ax.patches) == before


def test_a_categorical_y_is_ticked_but_not_pinned_when_sharing_is_off(qtbot):
    """The vertical twin of the shared-axis rule.

    The level names still label the ticks; only sharing fixes the limits, so
    a panel that is framing its own rows is left to frame them.
    """
    view = gb.GraphCanvas(link=LinkedSelection(), source="builders")
    qtbot.addWidget(view)
    frame = _frame(24)
    frame["gene"] = [f"g{i % 3}" for i in range(24)]
    view.set_frame(frame)

    view.set_spec(GraphSpec(x="area", y="gene", kind=SCATTER, shared_y=False))
    loose = view.axes_at(0, 0)
    loose_ticks = [t.get_text() for t in loose.get_yticklabels()]
    loose_limits = loose.get_ylim()

    view.set_spec(GraphSpec(x="area", y="gene", kind=SCATTER, shared_y=True))
    shared = view.axes_at(0, 0)

    assert loose_ticks == ["g0", "g1", "g2"]
    assert shared.get_ylim() == (-0.6, 2.6)
    assert loose_limits != shared.get_ylim()


def test_the_plot_picker_stays_on_automatic_for_a_kind_it_does_not_offer(
        qtbot):
    """``empty`` is inferred, never picked, so it is not in the menu.

    A spec restored with it must leave the picker showing what it showed
    rather than take ``findData``'s "not found" as an index: -1 is Qt's
    no-selection, and the plot menu would go blank on a spec that is simply
    not pinned to anything.
    """
    panel = gb.GraphBuilderPanel(link=LinkedSelection(), source="builders")
    qtbot.addWidget(panel)
    picker = panel.findChild(gb.QComboBox, "GraphKindPicker")
    assert picker is not None
    offered = [picker.itemData(i) for i in range(picker.count())]
    assert EMPTY not in offered

    panel.set_spec(GraphSpec(kind="bar", x="plateID"))
    assert picker.currentData() == "bar"

    panel.set_spec(GraphSpec(kind=EMPTY))

    assert picker.currentData() == "bar"
    assert panel.canvas.spec.kind == EMPTY

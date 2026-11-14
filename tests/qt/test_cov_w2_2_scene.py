"""Every artist the translator carries, and every one it refuses.

A generated figure is written twice over: once as the matplotlib page it has
always been, and once as a pyqtgraph scene that looks like the screen. The
whole value of the second is that it is either FAITHFUL or ABSENT -- an
incomplete translation must fall back rather than write a page missing its
bars, its labels or its colour bar. So the interesting behaviour is all in
the branches: the artist with no data in it, the colour that resolves to
nothing, the formula the translator will not guess at, the exporter that
writes no file.

Every figure here is a real matplotlib figure with real artists on it, drawn
and then read, because the thing under test is a translation between two
libraries and a stand-in artist would only be a translation of the stand-in.
"""

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt          # noqa: E402

from spacr.figures import scene as sc    # noqa: E402
from spacr.figures.scene import (
    SceneReport,
    _alpha,
    _anchor,
    _clamp,
    _dash,
    _hex,
    _looks_numeric,
    build_scene,
    export_scene,
    pyqtgraph_ready,
    requested_renderer,
)


@pytest.fixture
def closed_figures():
    """Close whatever a test drew, whichever way it ends."""
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def renderer_environment_pinned(monkeypatch):
    """`SPACR_FIGURE_RENDERER` is process-wide; pin it away from the tests."""
    monkeypatch.delenv("SPACR_FIGURE_RENDERER", raising=False)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# choosing a renderer
# ---------------------------------------------------------------------------

def test_a_misspelt_environment_variable_does_not_lose_the_figures(
        monkeypatch):
    """An unrecognised value is 'auto', not an error."""
    monkeypatch.setenv("SPACR_FIGURE_RENDERER", "pyqtgrpah")
    assert requested_renderer() == "auto"

    monkeypatch.setenv("SPACR_FIGURE_RENDERER", "  MATPLOTLIB ")
    assert requested_renderer() == "matplotlib"


# ---------------------------------------------------------------------------
# colour
# ---------------------------------------------------------------------------

def test_a_colour_that_is_not_one_leaves_the_page_blank_there():
    """Inventing a colour puts ink on a page the panel deliberately left off."""
    assert _hex(None) is None
    assert _hex("none") is None
    assert _hex("not a colour") is None
    assert _hex((0.0, 0.0, 0.0, 0.0)) is None
    assert _hex("#FF0000") == "#FF0000"
    assert _hex((1.0, 0.0, 0.0)) == "#FF0000"


def test_an_unreadable_colour_is_fully_opaque_rather_than_invisible():
    """The alpha of something that is not a colour is not zero.

    Zero would silently erase an artist whose colour merely failed to parse.
    """
    assert _alpha("not a colour") == 255
    assert _alpha("#FF000080") == 128
    assert _alpha("#FF0000", 0.5) == 128
    assert _alpha("#FF0000", "half") == 255
    assert _alpha("#FF0000", 4.0) == 255
    assert _alpha("#FF0000", -1.0) == 0


def test_a_dash_pattern_comes_from_either_spelling():
    """matplotlib carries both the string styles and (offset, sequence)."""
    assert _dash("-") is None
    assert _dash("--") == [4, 3]
    assert _dash(":") == [1, 2]
    assert _dash((0, (3, 2))) == [3.0, 2.0]
    assert _dash((0, ())) is None
    assert _dash(object()) is None
    assert _dash((0, (0.0, 1.0))) == [0.1, 1.0]


# ---------------------------------------------------------------------------
# geometry and text
# ---------------------------------------------------------------------------

def test_a_range_is_held_inside_its_bounds():
    """A panel's limits cannot leave the page they are drawn on."""
    assert _clamp(-5.0, 500.0, 0.0, 100.0) == (0.0, 100.0)
    assert _clamp(10.0, 20.0, 0.0, 100.0) == (10.0, 20.0)


def test_an_alignment_pair_becomes_an_anchor(closed_figures):
    """pyqtgraph anchors from 0 to 1; matplotlib names its alignments."""
    figure, axes = plt.subplots()
    left = axes.text(0.1, 0.1, "a", ha="left", va="top")
    middle = axes.text(0.2, 0.2, "b", ha="center", va="center")
    right = axes.text(0.3, 0.3, "c", ha="right", va="bottom")

    assert _anchor(left) == (0.0, 0.0)
    assert _anchor(middle) == (0.5, 0.5)
    assert _anchor(right) == (1.0, 1.0)


def test_a_tick_label_is_numeric_only_when_it_is_a_number():
    """A categorical axis writes plate1, plate2 -- not 0, 1, 2."""
    assert _looks_numeric("3.5") is True
    assert _looks_numeric("−1") is True         # matplotlib's minus sign
    assert _looks_numeric("plate1") is False
    assert _looks_numeric("") is False


# ---------------------------------------------------------------------------
# translating a real figure
# ---------------------------------------------------------------------------

@pytest.fixture
def qt_ready(qapp):
    ready, why = pyqtgraph_ready()
    if not ready:
        pytest.skip(why)
    return True


def _built(figure):
    widget, report = build_scene(figure)
    try:
        return report
    finally:
        widget.deleteLater()


def test_a_colour_bar_is_placed_beside_what_it_keys(qt_ready,
                                                    closed_figures):
    """It has no cell of its own on the page it came from."""
    figure, axes = plt.subplots()
    image = axes.imshow(np.arange(64, dtype=float).reshape(8, 8))
    figure.colorbar(image, ax=axes, label="counts")
    figure.canvas.draw()

    positions = sc._positions(figure)
    assert len(positions) == 2
    keyed, bar = positions
    assert keyed[0] == bar[0], "the colour bar left its panel's row"
    assert bar[1] == keyed[1] + 1
    assert keyed[2] == 1 and bar[2] == 1
    assert sc._columns(figure) == 2

    report = _built(figure)
    assert report.complete, report.reason()


# ---------------------------------------------------------------------------
# writing it out
# ---------------------------------------------------------------------------

def test_a_scene_is_written_by_the_exporter_the_tabs_use(qt_ready, tmp_path,
                                                         closed_figures):
    """A second exporter here would recreate the same duplication one level
    down."""
    figure, axes = plt.subplots()
    axes.plot([0, 1, 2], [1, 2, 3])
    figure.canvas.draw()

    widget, report = build_scene(figure)
    try:
        written = export_scene(widget, str(tmp_path / "out" / "panel.png"))
        assert written and os.path.isfile(written)
    finally:
        widget.deleteLater()


# ---------------------------------------------------------------------------
# the shapes an artist can be in when it is not the clean case
# ---------------------------------------------------------------------------

def test_an_axes_with_no_place_on_the_grid_falls_back_to_its_order(
        qt_ready, closed_figures):
    """`add_axes` has no subplotspec, and a page still has to be laid out."""
    figure = plt.figure()
    figure.add_axes([0.1, 0.1, 0.3, 0.8])
    figure.add_axes([0.5, 0.1, 0.3, 0.8])

    assert sc._grid_position(figure.axes[0], 0) == (0, 0)
    assert sc._grid_position(figure.axes[1], 1) == (0, 1)
    assert sc._columns(figure) == 4


def test_a_colour_bar_whose_mappable_has_no_axes_is_left_where_it_is(
        qt_ready, closed_figures):
    """The reposition needs to find the panel the bar keys."""
    figure, axes = plt.subplots()
    image = axes.imshow(np.arange(16, dtype=float).reshape(4, 4))
    bar = figure.colorbar(image, ax=axes)
    bar.mappable = object()
    figure.canvas.draw()

    positions = sc._positions(figure)
    assert len(positions) == 2
    assert all(span == 2 for _, _, span in positions)


def test_a_scene_can_be_written_as_a_vector_file(qt_ready, tmp_path,
                                                 closed_figures):
    """A PDF full of little bitmaps of a dot is a PDF that is not vector."""
    figure, axes = plt.subplots()
    axes.plot([0, 1, 2], [1, 2, 3])
    axes.scatter([0, 1], [1, 2])
    figure.canvas.draw()

    widget, report = build_scene(figure)
    try:
        assert report.complete, report.reason()
        for suffix in (".pdf", ".svg"):
            written = export_scene(widget, str(tmp_path / f"panel{suffix}"))
            assert written and os.path.isfile(written)
    finally:
        widget.deleteLater()


def test_a_data_colour_nobody_can_see_on_the_page_is_named(capsys,
                                                           monkeypatch):
    """Named, not replaced: the panel chose it and the reader is told."""
    from spacr import figure_style

    report = SceneReport()
    report.data_colours.extend(["#FFFFF0", "#FFFFFE"])

    monkeypatch.setattr(
        figure_style, "saved_figure_appearance",
        lambda: type("L", (), {"flip": True, "ground": "#FFFFFF"})())

    sc._warn_about_data_colours(report)
    printed = capsys.readouterr().out
    assert printed.strip(), "an illegible data colour was not reported"


def test_nothing_is_said_about_colours_on_a_page_that_was_not_flipped(
        capsys, monkeypatch):
    """A figure saved in the screen's own colours needs no warning."""
    from spacr import figure_style

    report = SceneReport()
    report.data_colours.append("#FFFFF0")
    monkeypatch.setattr(
        figure_style, "saved_figure_appearance",
        lambda: type("L", (), {"flip": False, "ground": "#FFFFFF"})())

    sc._warn_about_data_colours(report)
    assert capsys.readouterr().out == ""

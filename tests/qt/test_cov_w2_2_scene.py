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
from spacr.figures.scene import (RENDERERS, SceneReport, _alpha, _anchor,
                                 _clamp, _dash, _hex, _looks_numeric,
                                 _plain_text, build_scene, export_scene,
                                 pyqtgraph_ready, render_figure,
                                 requested_renderer, scene_renderer,
                                 write_figure)


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

def test_a_complete_report_has_no_reason_to_give():
    """"Why does this not look like the others" is asked of the fallback."""
    report = SceneReport()
    assert report.complete is True
    assert report.reason() == ""


def test_an_incomplete_report_counts_what_it_could_not_carry():
    """The names and their counts, so one missing artist is not read as many."""
    report = SceneReport()
    report.missing.extend(["Quiver", "Quiver", "Arc"])
    assert report.complete is False
    assert report.reason() == "pyqtgraph cannot yet carry Arc x1, Quiver x2"


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


def test_asking_for_matplotlib_says_that_is_why(monkeypatch):
    """The reason is never empty for matplotlib."""
    chosen, why = scene_renderer("matplotlib")
    assert chosen == "matplotlib"
    assert why == "matplotlib was asked for"

    monkeypatch.setenv("SPACR_FIGURE_RENDERER", "matplotlib")
    assert scene_renderer()[0] == "matplotlib"


def test_a_forced_value_that_is_not_a_renderer_falls_back_to_auto():
    """A caller passing nonsense gets the automatic answer, not an error."""
    chosen, why = scene_renderer("crayons")
    assert chosen in RENDERERS
    if chosen == "matplotlib":
        assert why


def test_a_machine_without_pyqtgraph_writes_the_page_it_always_wrote(
        monkeypatch):
    """And says why, because that is the question a user asks of it."""
    monkeypatch.setattr(sc, "pyqtgraph_ready",
                        lambda: (False, "pyqtgraph is unavailable here: no Qt"))
    monkeypatch.setattr(sc, "_the_gallery_could_not_show_it", lambda: "")

    chosen, why = scene_renderer("pyqtgraph")
    assert chosen == "matplotlib"
    assert "unavailable" in why


def test_a_gallery_that_could_not_show_it_sends_auto_to_matplotlib(
        monkeypatch):
    """'auto' means "pyqtgraph if this process can use the result"."""
    monkeypatch.setattr(sc, "_the_gallery_could_not_show_it",
                        lambda: "the gallery shows matplotlib figures here")
    chosen, why = scene_renderer("auto")
    assert chosen == "matplotlib"
    assert "gallery" in why


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


def test_the_mathtext_the_translator_knows_becomes_plain_unicode():
    """A panel writes its labels for a reader, so the set is deliberately
    short."""
    body, understood = _plain_text("plain label")
    assert (body, understood) == ("plain label", True)

    body, understood = _plain_text("")
    assert understood is True

    _body, understood = _plain_text(r"$\frac{\partial f}{\partial x}$")
    assert understood is False, "the translator guessed at a formula"


def test_a_tick_label_is_numeric_only_when_it_is_a_number():
    """A categorical axis writes plate1, plate2 -- not 0, 1, 2."""
    assert _looks_numeric("3.5") is True
    assert _looks_numeric("−1") is True         # matplotlib's minus sign
    assert _looks_numeric("plate1") is False
    assert _looks_numeric("") is False


# ---------------------------------------------------------------------------
# translating a real figure
# ---------------------------------------------------------------------------

pytestmark_qt = pytest.mark.skipif(not pyqtgraph_ready()[0],
                                   reason="pyqtgraph is not usable here")


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


def test_a_reference_line_is_carried_as_an_infinite_line(qt_ready,
                                                         closed_figures):
    """axhline and axvline draw in a blended transform, and that is the test.

    It is the only way to tell a zero line from a fitted curve, both of which
    are Line2D, and it survives a user restyle.
    """
    figure, axes = plt.subplots()
    axes.plot([0, 1, 2], [1, 2, 3])
    axes.axhline(0.0, color="#888888", linestyle="--")
    axes.axvline(1.0, color="#888888", linestyle=":")
    figure.canvas.draw()

    report = _built(figure)
    assert report.complete, report.reason()
    assert report.items >= 3


def test_a_line_with_markers_and_no_line_becomes_points(qt_ready,
                                                        closed_figures):
    """"marker only" is a scatter drawn as a Line2D, and it has to survive."""
    figure, axes = plt.subplots()
    axes.plot([0, 1, 2], [1, 2, 3], linestyle="None", marker="o",
              markerfacecolor="#3355AA", markersize=6)
    figure.canvas.draw()

    report = _built(figure)
    assert report.complete, report.reason()
    assert report.items >= 1


def test_a_line_with_no_data_adds_nothing(qt_ready, closed_figures):
    """An empty series is not an error and is not a mark."""
    figure, axes = plt.subplots()
    axes.plot([], [])
    figure.canvas.draw()

    report = _built(figure)
    assert report.complete, report.reason()
    assert report.items == 0


def test_a_scatter_is_converted_by_area_not_by_scale(qt_ready,
                                                     closed_figures):
    """matplotlib's `s` is an area and pyqtgraph's size is a diameter.

    An s=18 scatter drawn at diameter 18 is a panel of overlapping blobs.
    """
    figure, axes = plt.subplots()
    axes.scatter([0, 1, 2], [1, 2, 3], s=[18, 36, 72], c="#AA3333")
    axes.scatter([], [])
    figure.canvas.draw()

    report = _built(figure)
    assert report.complete, report.reason()
    assert report.items >= 1


def test_vlines_become_one_item_not_one_per_segment(qt_ready,
                                                    closed_figures):
    """A stem plot on a real screen is one segment per well."""
    figure, axes = plt.subplots()
    axes.vlines(range(40), 0, np.linspace(0.1, 1.0, 40), color="#444444")
    figure.canvas.draw()

    report = _built(figure)
    assert report.complete, report.reason()
    assert report.items == 1


def test_a_polygon_is_carried_as_a_filled_shape(qt_ready, closed_figures):
    """The translator's own polygon path, driven by a PolyCollection."""
    from matplotlib.collections import PolyCollection

    figure, axes = plt.subplots()
    axes.add_collection(PolyCollection(
        [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],      # a triangle
         [(0.0, 0.0), (1.0, 1.0)]],                 # a line, not a polygon
        closed=False, facecolors=["#66AACC", "#66AACC"], alpha=0.4))
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    figure.canvas.draw()

    report = _built(figure)
    assert report.complete, report.reason()
    assert report.items == 1, "the two-vertex path was drawn as a polygon"


def test_a_polygon_with_no_colour_is_not_drawn(qt_ready, closed_figures):
    """A face of 'none' is a shape the panel deliberately left unfilled."""
    from matplotlib.collections import PolyCollection

    figure, axes = plt.subplots()
    axes.add_collection(PolyCollection(
        [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]], facecolors="none"))
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    figure.canvas.draw()

    report = _built(figure)
    assert report.complete, report.reason()
    assert report.items == 0


@pytest.mark.xfail(strict=True, reason=(
    "the artist dispatch in _translate_axes keys on type(artist).__name__, "
    "and matplotlib 3.10+ returns a FillBetweenPolyCollection from "
    "fill_between. It IS a PolyCollection, so _add_poly_collection would "
    "carry it, but the name test does not match and the whole figure is "
    "reported incomplete -- so every panel with a confidence band silently "
    "loses the pyqtgraph renderer. _is_chrome_artist in the same module "
    "records this exact lesson: check by TYPE, not by class NAME."))
def test_a_confidence_band_does_not_cost_the_figure_its_renderer(
        qt_ready, closed_figures):
    """`fill_between` is the commonest band in the suite, and it is a polygon.

    The translator has a polygon path and the artist is a PolyCollection; the
    only thing standing between them is the spelling of its class name.
    """
    figure, axes = plt.subplots()
    x = np.linspace(0, 1, 20)
    axes.fill_between(x, x - 0.1, x + 0.1, color="#66AACC", alpha=0.4)
    figure.canvas.draw()

    report = _built(figure)
    assert report.complete, report.reason()


def test_bars_are_carried_as_one_item(qt_ready, closed_figures):
    """A Patch's transform is not the data transform, and testing it against
    `ax.transData` silently dropped every bar in the suite."""
    figure, axes = plt.subplots()
    axes.hist(np.random.default_rng(0).normal(size=200), bins=20,
              color="#8888CC", edgecolor="white")
    figure.canvas.draw()

    report = _built(figure)
    assert report.complete, report.reason()
    assert report.items >= 1


def test_a_title_an_axis_label_and_an_annotation_are_all_carried(
        qt_ready, closed_figures):
    """Chrome is chrome: it goes through the print-colour resolver."""
    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])
    axes.set_title("A panel")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.text(0.05, 0.9, "n = 200", transform=axes.transAxes,
              bbox={"facecolor": "white", "edgecolor": "#333333"})
    axes.text(0.5, 0.5, "in data coordinates")
    axes.text(0.5, 0.4, "")
    figure.suptitle("The whole page")
    figure.canvas.draw()

    report = _built(figure)
    assert report.complete, report.reason()
    assert report.items >= 3


def test_a_formula_the_translator_will_not_guess_at_stops_the_scene(
        qt_ready, closed_figures):
    """An incomplete translation must fall back, not write a page missing a
    label."""
    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])
    axes.text(0.5, 0.5, r"$\frac{\partial f}{\partial x}$")
    figure.canvas.draw()

    report = _built(figure)
    assert report.complete is False
    assert "mathtext" in report.reason()


def test_an_annotation_with_an_arrow_records_that_the_arrow_was_dropped(
        qt_ready, closed_figures):
    """The note is how a caller knows the picture is not quite the same."""
    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])
    axes.annotate("here", xy=(0.5, 0.5), xytext=(0.7, 0.2),
                  arrowprops={"arrowstyle": "->"})
    figure.canvas.draw()

    widget, report = build_scene(figure)
    try:
        assert any("arrow" in note for note in report.notes), report.notes
    finally:
        widget.deleteLater()


def test_an_image_keeps_its_colour_map(qt_ready, closed_figures):
    """The map encodes data values and must not be inverted by print styling."""
    figure, axes = plt.subplots()
    axes.imshow(np.arange(64, dtype=float).reshape(8, 8), cmap="viridis")
    figure.canvas.draw()

    report = _built(figure)
    assert report.complete, report.reason()
    assert report.items >= 1


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


def test_a_legend_is_carried_as_one_item(qt_ready, closed_figures):
    """A legend nobody can read is a panel nobody can read."""
    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1], label="up", color="#3366AA")
    axes.plot([0, 1], [1, 0], label="down", color="#AA3366")
    axes.legend()
    figure.canvas.draw()

    report = _built(figure)
    assert report.complete, report.reason()


def test_a_categorical_axis_keeps_the_labels_the_panel_wrote(qt_ready,
                                                             closed_figures):
    """A renderer that re-derives ticks draws 0, 1, 2 where the panel wrote
    plate1, plate2, plate3 -- a different figure, silently."""
    figure, axes = plt.subplots()
    axes.bar(["plate1", "plate2", "plate3"], [1, 2, 3])
    figure.canvas.draw()

    widget, report = build_scene(figure)
    try:
        assert report.complete, report.reason()
        plot = widget.getItem(0, 0)
        ticks = plot.getAxis("bottom")._tickLevels
        written = {label for _, label in (ticks[0] if ticks else [])}
        assert {"plate1", "plate2", "plate3"} <= written
    finally:
        widget.deleteLater()


def test_a_log_axis_leaves_its_ticks_to_pyqtgraph(qt_ready, closed_figures):
    """A log axis writes its ticks in mathtext, and carrying those across put
    `$\\mathdefault{10^{-1}}$` down the side of the panel."""
    figure, axes = plt.subplots()
    axes.plot([1, 10, 100], [1e-3, 1e-2, 1e-1])
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.text(0.5, 0.5, "corner", transform=axes.transAxes)
    figure.canvas.draw()

    widget, report = build_scene(figure)
    try:
        assert report.complete, report.reason()
        for name in ("bottom", "left"):
            levels = widget.getItem(0, 0).getAxis(name)._tickLevels
            for _, label in (levels[0] if levels else []):
                assert "mathdefault" not in label
    finally:
        widget.deleteLater()


def test_an_artist_nobody_has_taught_the_translator_is_named(qt_ready,
                                                             closed_figures):
    """The fallback is not a failure, but it has to say what caused it."""
    figure, axes = plt.subplots()
    axes.quiver([0, 1], [0, 1], [1, 1], [1, -1])
    figure.canvas.draw()

    report = _built(figure)
    assert report.complete is False
    assert "Quiver" in report.reason()


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


def test_a_figure_that_cannot_be_translated_writes_the_matplotlib_page(
        tmp_path, closed_figures, monkeypatch):
    """The picture is never the thing that is lost."""
    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])
    axes.text(0.5, 0.5, r"$\frac{a}{b}$")
    figure.canvas.draw()

    written, renderer, why = write_figure(figure, str(tmp_path / "panel"),
                                          fmt="png", announce=False)
    assert renderer == "matplotlib"
    assert why
    assert written and os.path.isfile(str(written))


def test_a_renderer_that_is_not_available_never_raises(tmp_path,
                                                       closed_figures,
                                                       monkeypatch):
    """Losing an hour's fit to a renderer is the worst trade in this module."""
    monkeypatch.setattr(sc, "pyqtgraph_ready",
                        lambda: (False, "pyqtgraph is unavailable here"))

    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])

    written, report = render_figure(figure, str(tmp_path / "panel"),
                                    fmt="png", announce=False)
    assert written is None
    assert "pyqtgraph" in report.missing
    assert report.notes


def test_an_exception_inside_the_translation_is_reported_not_raised(
        tmp_path, closed_figures, monkeypatch):
    """Nothing here raises: the report carries the failure instead."""
    monkeypatch.setattr(sc, "pyqtgraph_ready", lambda: (True, ""))

    def explode(_figure, mode=None):
        raise RuntimeError("the scene could not be laid out")

    monkeypatch.setattr(sc, "build_scene", explode)

    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])

    written, report = render_figure(figure, str(tmp_path / "panel"),
                                    fmt="png", announce=False)
    assert written is None
    assert "RuntimeError" in report.missing
    assert "could not be laid out" in report.notes[-1]


def test_an_exporter_that_writes_nothing_is_a_fallback_not_a_success(
        tmp_path, closed_figures, monkeypatch):
    """A manifest entry pointing at a file that is not there is worse than
    none."""
    monkeypatch.setattr(sc, "pyqtgraph_ready", lambda: (True, ""))
    monkeypatch.setattr(sc, "build_scene",
                        lambda figure, mode=None: (None, SceneReport()))
    monkeypatch.setattr(sc, "export_scene", lambda widget, path: None)

    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])

    written, report = render_figure(figure, str(tmp_path / "panel"),
                                    fmt="png", announce=False)
    assert written is None
    assert "the exporter wrote nothing" in report.missing

    written, renderer, why = write_figure(figure, str(tmp_path / "panel2"),
                                          fmt="png", announce=False)
    assert renderer == "matplotlib"
    assert why


def test_a_run_with_no_gallery_still_gets_its_file(tmp_path, closed_figures):
    """`announce=False` writes the file without publishing a tile."""
    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])

    written, renderer, _why = write_figure(
        figure, str(tmp_path / "quiet"), fmt="png", announce=False,
        renderer="matplotlib")
    assert renderer == "matplotlib"
    assert os.path.isfile(str(written))


# ---------------------------------------------------------------------------
# the shapes an artist can be in when it is not the clean case
# ---------------------------------------------------------------------------

def test_a_bar_with_no_size_is_not_drawn(qt_ready, closed_figures):
    """A zero-by-zero rectangle is not a bar and not an error."""
    figure, axes = plt.subplots()
    axes.bar([0.0], [0.0], width=0.0)
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    figure.canvas.draw()

    report = _built(figure)
    assert report.complete, report.reason()
    assert report.items == 0


def test_a_bar_entirely_outside_the_view_is_clamped_away(qt_ready,
                                                         closed_figures):
    """A ViewBox does not clip its children and matplotlib does clip a patch.

    Clamping the geometry rather than clipping the item is what keeps a
    caption placed below the axes on the page.
    """
    figure, axes = plt.subplots()
    axes.bar([5.0], [1.0], width=0.4)
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    figure.canvas.draw()

    report = _built(figure)
    assert report.complete, report.reason()
    assert report.items == 0


def test_a_backdrop_drawn_in_axes_fraction_is_converted(qt_ready,
                                                        closed_figures):
    """A skipped tile with no backdrop stops looking skipped."""
    from matplotlib.patches import Rectangle

    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])
    axes.add_patch(Rectangle((0.1, 0.1), 0.8, 0.8, transform=axes.transAxes,
                             facecolor="#DDDDDD"))
    figure.canvas.draw()

    report = _built(figure)
    assert report.complete, report.reason()
    assert report.items >= 1


def test_a_text_whose_transform_cannot_be_read_keeps_its_own_position(
        qt_ready, closed_figures):
    """A Text mid-teardown raises when asked what it is drawn in."""
    figure, axes = plt.subplots()
    text = axes.text(0.5, 0.5, "a caption")

    def explode():
        raise RuntimeError("this artist is being removed")

    text.get_transform = explode
    assert sc._in_data_coordinates(text, axes) == (0.5, 0.5)


def test_a_text_that_cannot_be_placed_is_left_off(qt_ready, closed_figures):
    """An infinite position is nowhere, and nowhere is not a place to draw."""
    figure, axes = plt.subplots()
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    text = axes.text(float("inf"), 0.5, "a caption",
                     transform=axes.transAxes)
    assert sc._in_data_coordinates(text, axes) is None


def test_a_text_with_no_colour_is_not_drawn(qt_ready, closed_figures):
    """A caption painted in 'none' is one the panel left off."""
    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])
    axes.text(0.5, 0.5, "invisible", color="none")
    figure.canvas.draw()

    widget, report = build_scene(figure)
    try:
        assert report.complete, report.reason()
        assert report.items == 1, "an invisible caption was drawn"
    finally:
        widget.deleteLater()


def test_an_image_with_no_data_is_not_drawn(qt_ready, closed_figures):
    """An empty array is an axes with nothing in it."""
    figure, axes = plt.subplots()
    axes.imshow(np.zeros((0, 0)))
    figure.canvas.draw()

    report = _built(figure)
    assert report.items == 0


def test_a_legend_with_no_entries_adds_nothing(qt_ready, closed_figures):
    """An empty legend is not a legend."""
    import pyqtgraph as pg

    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])
    widget = pg.GraphicsLayoutWidget()
    try:
        plot = widget.addPlot(row=0, col=0)

        class _Empty:
            def get_texts(self):
                return []

        assert sc._add_legend(plot, _Empty(), sc._Look(None)) == 0
    finally:
        widget.deleteLater()


def test_a_legend_handle_that_refuses_every_colour_still_gets_a_sample(
        qt_ready, closed_figures):
    """A handle with no readable colour is drawn in the neutral grey."""
    import pyqtgraph as pg

    class _Text:
        def get_text(self):
            return "an entry"

        def get_color(self):
            return "#333333"

    class _Handle:
        def get_color(self):
            raise RuntimeError("this handle has been removed")

        def get_facecolor(self):
            raise RuntimeError("this handle has been removed")

        def get_edgecolor(self):
            raise RuntimeError("this handle has been removed")

    class _Legend:
        legend_handles = [_Handle()]

        def get_texts(self):
            return [_Text()]

    widget = pg.GraphicsLayoutWidget()
    try:
        plot = widget.addPlot(row=0, col=0)
        assert sc._add_legend(plot, _Legend(), sc._Look(None)) == 1
    finally:
        widget.deleteLater()


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


def test_a_scene_with_no_ground_paints_none(qt_ready, closed_figures,
                                            monkeypatch):
    """A figure saved with no background is transparent, not white."""
    look = sc._Look(None)
    monkeypatch.setattr(type(look.look), "ground", property(lambda self: None),
                        raising=False)

    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])
    figure.canvas.draw()

    widget, report = build_scene(figure)
    try:
        assert report.complete, report.reason()
    finally:
        widget.deleteLater()


def test_laying_out_a_widget_that_refuses_is_not_fatal(qt_ready,
                                                       closed_figures):
    """Asking for the layout is best-effort; the export still happens."""
    class _Refusing:
        @property
        def ci(self):
            raise RuntimeError("this widget has gone")

    sc._lay_out(_Refusing())                  # must not raise


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


def test_a_complete_scene_is_written_by_pyqtgraph_and_says_nothing_is_wrong(
        qt_ready, tmp_path, closed_figures):
    """The success path, end to end: no reason, because it is the screen's."""
    figure, axes = plt.subplots()
    axes.plot([0, 1, 2], [1, 2, 3], color="#3366AA")
    axes.set_title("A panel")
    figure.canvas.draw()

    written, renderer, why = write_figure(
        figure, str(tmp_path / "panel"), fmt="png", renderer="pyqtgraph",
        announce=False)
    assert renderer == "pyqtgraph"
    assert why == ""
    assert os.path.isfile(str(written))


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

"""The rest of the right-click restyle menu, and the SVG that never worked.

Instruction 128 O. The maintainer asked twice, and the second ask is the one
that names what was still missing:

    "when right clicking a graph i should also be able to choose its
     dimensions the cmap of the datapoints and their shape, font size color,
     line color and width (sometimes not applicable on certain graph types).
     as well as graph type when applicable"

    "axis limits, axis labels, aspect ratio, cmap (choose any column), font
     size, point shape (choose any column)"

Point size, point colour, opacity, axis labels, font size, grid, legend, the
mark type, the baseline, the compartment, the effect-size cut and the p-value
axis already shipped. What lands here is AXIS LIMITS with a way back to auto,
ASPECT RATIO, FIGURE DIMENSIONS, FONT COLOUR, LINE COLOUR AND WIDTH, and the
two that are not restyling at all -- a COLOUR SCALE over any numeric column
and a POINT SHAPE over any low-cardinality one, which MAP A COLUMN onto a
visual channel rather than setting one value.

THE DESIGN CONSTRAINT IS THE MAINTAINER'S OWN PARENTHETICAL, "(sometimes not
applicable on certain graph types)", and instruction 106 already answered it
for settings: an entry that cannot do anything is GREYED OUT AND SAYS WHY. Not
silently absent, which leaves a user hunting for a control they were told
about; not present-but-inert, which leaves them concluding the plot is broken.

AND THE EXPORT. pyqtgraph's own SVGExporter raises on every plot in this
module, so "Vector (*.svg)" was a format that always failed. It is fixed here
by not using it -- Qt paints vector devices, and the PDF path had already
proved that -- and the same change turned out to fix a defect nobody had
noticed in the PDF: its scatter points were fifty little JPEGs.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


N = 60


def _frame(seed: int = 0, missing: int = 0) -> pd.DataFrame:
    """A coefficient table with one column of every kind the menu sorts on."""
    rng = np.random.default_rng(seed)
    effects = rng.normal(0, 0.5, N)
    well_count = rng.integers(4, 96, N).astype("float64")
    if missing:
        well_count[:missing] = np.nan
    return pd.DataFrame({
        "feature": [f"fraction:grna[{i}_1]" for i in range(N)],
        "coefficient": effects,
        "p_value": rng.uniform(0.001, 0.99, N),
        "well_count": well_count,
        "n_guides": rng.integers(1, 4, N),
        "condition": list(rng.choice(["nc", "pc", "other"], N,
                                     p=[0.1, 0.1, 0.8])),
    })


@pytest.fixture()
def volcano(qtbot):
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_frame(), effect_threshold=0.6)
    return plot


@pytest.fixture()
def histogram(qtbot):
    from spacr.qt.widgets.fast_plots import PValueHistogram

    plot = PValueHistogram()
    qtbot.addWidget(plot)
    plot.set_p_values(np.random.default_rng(1).random(200))
    return plot


@pytest.fixture()
def controls(qtbot):
    from spacr.qt.widgets.fast_plots import ControlSeparation

    plot = ControlSeparation()
    qtbot.addWidget(plot)
    rng = np.random.default_rng(2)
    plot.set_groups({"negative": rng.normal(-1, 0.3, 40),
                     "positive": rng.normal(1, 0.3, 35)})
    return plot


def _entries(plot):
    return [action.text() for action in plot.build_style_menu().actions()]


def _joined(plot):
    return " ".join(_entries(plot))


def _action(plot, fragment):
    for action in plot.build_style_menu().actions():
        if fragment in action.text():
            return action
    raise AssertionError(f"no entry containing {fragment!r}: {_entries(plot)}")


# --------------------------------------------------------------------------- #
#  Everything the maintainer named is on the menu
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("wanted", [
    "Axis limits", "Aspect ratio", "Font colour", "Line colour and width",
    "Colour by a column", "Shape by a column", "Size on screen",
    "Exported page size",
])
def test_the_menu_offers_everything_that_was_asked_for_by_name(volcano,
                                                               wanted):
    """The second ask listed six things and five of them did not exist. A
    feature that ships without an entry on the menu the user right-clicked is
    a feature nobody will find."""
    assert wanted in _joined(volcano), _entries(volcano)


def test_the_two_dimension_entries_say_which_dimension_each_one_moves():
    """A single "Dimensions…" would be the misleading version. On a live plot
    it is the widget's size; on a saved figure it is the page. A user who sets
    "dimensions" and then finds the exported PDF unchanged has been misled by
    the control rather than helped by it, so the two are named apart."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    plot.set_results(_frame())
    entries = _entries(plot)

    assert "Size on screen…" in entries, entries
    assert "Exported page size…" in entries, entries
    assert not any(text.strip() == "Dimensions…" for text in entries), (
        "one entry called 'Dimensions' cannot say which of the two it moves")


def test_there_is_a_way_back_from_every_control_that_pins_something(volcano):
    """A control that can only be set is a trap: pin x to the wrong decade,
    or fix the widget at 200 px, and the only way out is reloading the run."""
    entries = _joined(volcano)

    assert "Axis limits: back to automatic" in entries
    assert "Size on screen: back to automatic" in entries


# --------------------------------------------------------------------------- #
#  Axis limits
# --------------------------------------------------------------------------- #

def test_a_typed_axis_limit_is_what_the_plot_actually_shows(volcano):
    volcano.set_axis_limits(x=(-1.0, 1.0), y=(0.0, 3.0))

    (x_from, x_to), (y_from, y_to) = volcano.axis_limits()
    assert (round(x_from, 6), round(x_to, 6)) == (-1.0, 1.0)
    assert (round(y_from, 6), round(y_to, 6)) == (0.0, 3.0)


def test_a_typed_limit_survives_the_next_redraw(volcano):
    """pyqtgraph re-fits the view to the data unless auto-range is switched
    off, so a limit the user typed would hold until the first recolour and
    then spring back -- which reads as the control not working."""
    volcano.set_axis_limits(x=(-0.5, 0.5))

    volcano.set_results(_frame(), effect_threshold=0.6)

    x_range, _ = volcano.axis_limits()
    assert (round(x_range[0], 6), round(x_range[1], 6)) == (-0.5, 0.5), (
        f"the redraw threw the typed limit away: {x_range}")


def test_only_the_pinned_axis_stops_following_the_data(volcano):
    """Pinning x must not freeze y. Half the point of typing one limit is
    that the other axis still fits whatever is drawn next."""
    volcano.set_axis_limits(x=(-0.5, 0.5))
    before = volcano.axis_limits()[1]

    frame = _frame()
    frame["p_value"] = np.linspace(1e-12, 0.9, N)
    volcano.set_results(frame)

    assert volcano.axis_limits()[1] != before, (
        "the y axis stopped following the data when x was pinned")


def test_back_to_automatic_gives_both_axes_back_to_the_data(volcano):
    original = volcano.axis_limits()
    volcano.set_axis_limits(x=(-99.0, 99.0), y=(-99.0, 99.0))

    volcano.auto_range_axes()

    assert volcano.axis_limits() == original


def test_the_axis_limit_dialog_writes_the_four_numbers_it_collected(
        volcano, monkeypatch):
    """Driving the menu action proves it is wired, not merely populated."""
    from PySide6.QtWidgets import QInputDialog

    answers = iter([(-2.0, True), (2.0, True), (0.0, True), (5.0, True)])
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: next(answers)))

    _action(volcano, "Axis limits…").trigger()

    assert volcano.axis_limits() == ((-2.0, 2.0), (0.0, 5.0))


def test_cancelling_part_way_through_leaves_the_axes_exactly_as_they_were(
        volcano, monkeypatch):
    """A user three numbers in who changes their mind must not be left with a
    half-pinned axis showing a range nobody chose."""
    from PySide6.QtWidgets import QInputDialog

    before = volcano.axis_limits()
    answers = iter([(-2.0, True), (2.0, True), (0.0, False)])
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: next(answers)))

    _action(volcano, "Axis limits…").trigger()

    assert volcano.axis_limits() == before


# --------------------------------------------------------------------------- #
#  Aspect ratio
# --------------------------------------------------------------------------- #

def test_locking_the_aspect_ratio_is_reported_back(volcano):
    """A Q-Q's 45-degree diagonal only means anything when the axes share a
    scale, which is the case this control exists for."""
    volcano.set_aspect_ratio(1.0)

    assert volcano.aspect_ratio() == 1.0


def test_the_aspect_lock_can_be_released_again(volcano):
    volcano.set_aspect_ratio(2.0)

    volcano.set_aspect_ratio(None)

    assert volcano.aspect_ratio() is None


def test_asking_for_no_ratio_unlocks_rather_than_locking_to_nothing(
        volcano, monkeypatch):
    """0 in the dialog means "let the plot fill its box". Locking an aspect of
    zero would collapse the view instead."""
    from PySide6.QtWidgets import QInputDialog

    volcano.set_aspect_ratio(3.0)
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: (0.0, True)))

    _action(volcano, "Aspect ratio").trigger()

    assert volcano.aspect_ratio() is None


# --------------------------------------------------------------------------- #
#  Dimensions: the widget, and the page, and which is which
# --------------------------------------------------------------------------- #

def test_the_screen_size_survives_the_layout_pass_that_undoes_a_resize(qtbot):
    """MEASURED, not assumed, and it is why this uses a fixed size rather than
    `resize`. These plots live in splitters, which own their children's
    geometry: a `resize` holds until the next layout pass and is then thrown
    away, so the control would appear to work and silently come undone the
    first time the user touched the divider or the window. The same finding is
    written on `FastPlot.snapshot`, where it produced a blank tile from a
    widget that reported the size it had been asked for."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QLabel, QSplitter

    from spacr.qt.widgets.fast_plots import VolcanoPlot

    split = QSplitter(Qt.Vertical)
    qtbot.addWidget(split)
    plot = VolcanoPlot()
    plot.set_results(_frame())
    split.addWidget(plot)
    split.addWidget(QLabel("under"))
    split.resize(900, 700)
    split.show()
    qtbot.waitExposed(split)

    plot.resize(400, 300)
    split.resize(901, 701)                  # the divider moves, or the window
    QApplication.processEvents()
    assert plot.size().width() != 400, (
        "resize() now survives a layout pass in a splitter; the fixed size "
        "below is then heavier than it needs to be and should be revisited")

    plot.set_screen_size(400, 300)
    split.resize(902, 702)
    QApplication.processEvents()

    assert (plot.size().width(), plot.size().height()) == (400, 300)


def test_going_back_to_automatic_keeps_the_floor_the_panel_set(qtbot):
    """`RegressionResultsPanel` gives the volcano `setMinimumHeight(240)` so a
    splitter cannot collapse it to a sliver. A restyle that released the
    widget to nothing would silently drop that floor, and the plot would
    vanish the first time the user dragged the divider."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.setMinimumHeight(240)
    plot.set_screen_size(400, 300)

    plot.clear_screen_size()

    assert plot.minimumHeight() == 240
    assert plot.maximumHeight() > 300, "the widget is still capped at 300 px"


def test_the_exported_page_is_the_size_that_was_asked_for(volcano, tmp_path):
    """THE HALF THAT MAKES THE CONTROL HONEST. A "dimensions" entry that moved
    the widget and left the PDF at its old page would be exactly the failure
    instruction 128 O names."""
    volcano.set_export_size(90.0, 60.0)

    volcano.export(str(tmp_path / "small.svg"))

    header = (tmp_path / "small.svg").read_text()[:400]
    assert 'width="89.9' in header or 'width="90' in header, header[:200]
    assert 'height="59.9' in header or 'height="60' in header, header[:200]


def test_the_page_height_follows_the_plot_unless_one_is_given(volcano,
                                                              tmp_path):
    """Nothing is stretched by default: a page whose height was invented
    would squash the point cloud in a direction nobody chose."""
    volcano.set_export_size(120.0)

    volcano.export(str(tmp_path / "auto.svg"))

    header = (tmp_path / "auto.svg").read_text()[:400]
    assert 'width="119.9' in header or 'width="120' in header, header[:200]
    assert 'height="90' not in header, "the height was pinned, not derived"


def test_setting_the_screen_size_does_not_move_the_saved_page(volcano,
                                                              tmp_path):
    """The two are different quantities, and the menu says so. This is the
    assertion behind that sentence."""
    before = volcano.export_size()

    volcano.set_screen_size(320, 240)

    assert volcano.export_size() == before


# --------------------------------------------------------------------------- #
#  Font colour, and the font size that only ever reached half the text
# --------------------------------------------------------------------------- #

def test_the_font_colour_reaches_the_ticks_and_the_labels(volcano):
    volcano.set_font_colour("#ff0000")

    axis = volcano.plot.getAxis("bottom")
    assert axis.textPen().color().name() == "#ff0000"
    assert axis.labelStyle.get("color") == "#ff0000"


def test_the_font_size_reaches_the_tick_numbers_not_only_the_axis_label(
        volcano):
    """THE BUG THIS REPLACED. The old handler passed `tickFont=None`, which
    asks for pyqtgraph's default rather than for a size, so "Font size: 20"
    enlarged two strings and left about twenty tick numbers where they were.
    A font control that moves the labels and not the numbers is the half of
    the figure a reader actually reads, left behind."""
    volcano.set_font_size(20)

    tick_font = volcano.plot.getAxis("bottom").style.get("tickFont")
    assert tick_font is not None, "the tick numbers still have no font"
    assert tick_font.pointSize() == 20


def test_setting_the_size_does_not_wipe_the_colour_and_the_other_way_round(
        volcano):
    """They come from two menu entries and each has to leave the other's
    choice standing, or the user concludes one of the two is broken."""
    volcano.set_font_colour("#00ff00")
    volcano.set_font_size(16)

    axis = volcano.plot.getAxis("bottom")
    assert axis.labelStyle.get("color") == "#00ff00"
    assert axis.labelStyle.get("font-size") == "16pt"


def test_a_theme_switch_does_not_undo_a_chosen_font_colour(volcano):
    """`restyle` exists to move every open plot's ink when the theme changes.
    It must not treat a colour the user picked as ink it owns."""
    volcano.set_font_colour("#ff00ff")

    volcano.restyle(background="none", foreground="#123456")

    assert volcano.plot.getAxis("bottom").textPen().color().name() == "#ff00ff"


def test_an_unlabelled_axis_is_left_alone_by_the_font_controls(controls):
    """The control panel's x-axis is deliberately unlabelled -- its ticks
    already name the groups. `setLabel` calls `showLabel`, so restyling the
    empty string would grow a blank strip under the plot."""
    axis = controls.plot.getAxis("bottom")
    assert not axis.labelText

    controls.set_font_size(18)

    assert not axis.label.isVisible(), (
        "changing the font gave the unlabelled axis a blank label")


def test_picking_a_font_colour_off_the_menu_applies_it(volcano, monkeypatch):
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog

    monkeypatch.setattr(QColorDialog, "getColor",
                        staticmethod(lambda *a, **k: QColor("#abcdef")))

    _action(volcano, "Font colour").trigger()

    assert volcano.font_colour() == "#abcdef"


def test_a_cancelled_colour_dialog_changes_nothing(volcano, monkeypatch):
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog

    monkeypatch.setattr(QColorDialog, "getColor",
                        staticmethod(lambda *a, **k: QColor()))

    _action(volcano, "Font colour").trigger()

    assert volcano.font_colour() is None


# --------------------------------------------------------------------------- #
#  Line colour and width
# --------------------------------------------------------------------------- #

def test_the_reference_and_threshold_lines_are_what_the_control_reaches(
        volcano):
    """A volcano carries the p=0.05 line and both effect-size cuts."""
    assert len(volcano.line_items()) == 3

    assert volcano.set_line_style("#00ff00", 4.0) == 3


def test_recolouring_the_lines_leaves_their_dashes_alone(volcano):
    """The dash pattern is what tells a reader which line is a threshold and
    which is the data's own trend. Rebuilding each pen from scratch would
    flatten that distinction on every restyle."""
    from PySide6.QtCore import Qt

    before = [volcano._pen_of(item).style() for item in volcano.line_items()]
    assert Qt.DashLine in before

    volcano.set_line_style("#00ff00", 4.0)

    after = [volcano._pen_of(item).style() for item in volcano.line_items()]
    assert after == before


def test_the_line_colour_and_width_are_both_applied(volcano):
    volcano.set_line_style("#00ff00", 4.0)

    for item in volcano.line_items():
        pen = volcano._pen_of(item)
        assert pen.color().name() == "#00ff00"
        assert pen.widthF() == 4.0


def test_the_threshold_caption_moves_with_the_line_it_names(volcano):
    """"p=0.05" is drawn by the line that carries it, in a colour given at
    construction. A red word beside a green line is the two-idioms failure
    this module warns about, on the one mark that names a threshold."""
    volcano.set_line_style("#00ff00", 2.0)

    labelled = [item for item in volcano.line_items()
                if getattr(item, "label", None) is not None]
    assert labelled, "the volcano lost the caption on its p-value line"
    assert labelled[0].label.color.name() == "#00ff00"


def test_the_summary_line_across_a_group_is_a_line_this_control_reaches(
        controls):
    """The maintainer said "line color and width" without qualification. A
    control that reached the volcano's thresholds and not the control panel's
    median lines would be a control the user cannot predict."""
    assert len(controls.line_items()) == 2

    assert controls.set_line_style("#123456", 3.0) == 2


def test_the_selection_ring_is_not_treated_as_a_line(volcano):
    """It is a cursor, not data. Recolouring it to match the thresholds would
    make the selection invisible against them."""
    volcano.highlight_key(_frame()["feature"].iloc[0])
    assert volcano._highlight is not None

    assert volcano._highlight not in volcano.line_items()


def test_a_line_restyle_keeps_the_width_even_when_the_colour_is_cancelled(
        volcano, monkeypatch):
    """The user answered one question and declined the other. Throwing away
    the answer they gave makes the dialog feel like it lost their input."""
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog, QInputDialog

    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: (6.0, True)))
    monkeypatch.setattr(QColorDialog, "getColor",
                        staticmethod(lambda *a, **k: QColor()))

    _action(volcano, "Line colour and width").trigger()

    assert volcano._pen_of(volcano.line_items()[0]).widthF() == 6.0


# --------------------------------------------------------------------------- #
#  A COLUMN onto a visual channel: the two that are not restyling
# --------------------------------------------------------------------------- #

def test_the_column_list_is_the_plots_own_table(volcano):
    """"cmap (choose any column)" means the columns of THIS table, not a
    second list built somewhere else that can disagree with it."""
    assert volcano.frame() is not None

    assert set(volcano.numeric_columns()) == {"coefficient", "p_value",
                                              "well_count", "n_guides"}


def test_only_a_continuous_column_is_offered_a_colour_scale(volcano):
    """A cmap on a nominal category puts an order into the picture that the
    data does not have -- the mistake the house style warns about."""
    assert "condition" not in volcano.numeric_columns()
    assert "feature" not in volcano.numeric_columns()


def test_only_a_low_cardinality_column_is_offered_shapes(volcano):
    """DTYPE IS NOT THE TEST. `n_guides` is an integer column with four values
    and is exactly what shapes are for; `feature` is a string column with
    sixty and is exactly what they are not."""
    assert set(volcano.shape_columns()) == {"n_guides", "condition"}


def test_colouring_by_a_column_uses_the_whole_scale(volcano):
    """One brush for everything would be the picture not changing at all."""
    painted = volcano.colour_by_column("coefficient", "viridis")

    assert painted == N
    brushes = list(volcano._scatter_items()[0].data["brush"])
    assert len({brush.color().rgba() for brush in brushes}) > 20


def test_the_scale_range_is_written_where_the_reader_can_see_it(volcano):
    """A continuous colouring with no range is a picture nobody can read a
    number off. The range IS the legend."""
    volcano.colour_by_column("well_count", "magma")

    said = volcano._status.text()
    assert "Coloured by well_count" in said, said
    assert "magma" in said, said


def test_a_row_with_no_value_is_grey_and_counted_rather_than_drawn_dark(
        volcano):
    """A NaN painted at the bottom of a viridis scale is a made-up
    measurement, and it is indistinguishable from a real small one."""
    volcano.set_results(_frame(missing=7), effect_threshold=0.6)

    volcano.colour_by_column("well_count")

    said = volcano._status.text()
    assert "7 points have no well_count and are grey" in said, said


def test_the_colour_key_survives_a_click_on_a_point(volcano):
    """A click writes the clicked row's detail into the status line. If that
    overwrote the scale's range, reading a point would destroy the legend of
    the picture the point is in."""
    volcano.colour_by_column("coefficient")

    volcano.set_status_note("fraction:grna[3_1]")

    said = volcano._status.text()
    assert "Coloured by coefficient" in said, said
    assert "fraction:grna[3_1]" in said, said


def test_the_colour_key_survives_a_redraw(volcano):
    """The headline is rewritten on every redraw. The key must not be part of
    it, or recolouring twice would leave a picture with no scale."""
    volcano.colour_by_column("coefficient")

    volcano.set_status("60 coefficients. Click a point for detail.")

    assert "Coloured by coefficient" in volcano._status.text()


def test_a_colour_scale_on_a_category_is_refused_out_loud(volcano):
    """Loudly, in the same spirit as `set_mark`: the callers are this class's
    own menu and a test, so a silent fallback would only ever make a mistake
    look like a working option."""
    with pytest.raises(ValueError, match="not a continuous column"):
        volcano.colour_by_column("condition")


def test_an_unknown_colour_scale_names_the_ones_that_exist(volcano):
    with pytest.raises(ValueError, match="unknown colormap"):
        volcano.colour_by_column("coefficient", "jet")


def test_shaping_by_a_column_gives_each_value_its_own_marker(volcano):
    shaped = volcano.shape_by_column("condition")

    assert shaped == N
    drawn = {point.symbol() for point in volcano._scatter_items()[0].points()}
    assert len(drawn) == 3, drawn


def test_the_shape_key_names_which_value_is_which(volcano):
    """Shapes with no key are a picture the reader has to guess at."""
    volcano.shape_by_column("condition")

    said = volcano._status.text()
    assert "nc is a circle" in said, said
    assert "other is a square" in said, said


def test_a_column_with_more_values_than_shapes_is_refused(volcano):
    """Reusing the circle for the ninth value would draw two different things
    identically, which is worse than not offering the column at all."""
    with pytest.raises(ValueError, match="shapes are distinguishable"):
        volcano.shape_by_column("feature")


def test_a_column_with_one_value_is_refused_because_it_says_nothing(qtbot):
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    frame = _frame()
    frame["condition"] = "other"
    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(frame)

    with pytest.raises(ValueError, match="one value"):
        plot.shape_by_column("condition")


def test_the_two_mappings_compose_rather_than_replacing_each_other(volcano):
    """Colour by effect and shape by condition is one picture with two
    channels, which is the whole reason they are separate controls."""
    volcano.colour_by_column("coefficient")
    volcano.shape_by_column("condition")

    item = volcano._scatter_items()[0]
    assert len({brush.color().rgba() for brush in item.data["brush"]}) > 20
    assert len({point.symbol() for point in item.points()}) == 3


def test_going_back_restores_the_plots_own_colouring(qtbot):
    """The compartment split, the single-guide genes, the influential wells:
    the brushes a plot was BUILT with are the only record of the sentence it
    was making. A restore that recomputed them would need this class to know
    every subclass's rule and would quietly swap one sentence for another."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_frame(), category_column="condition")
    item = plot._scatter_items()[0]
    before = [brush.color().name() for brush in item.data["brush"]]
    assert len(set(before)) == 3, "the category colouring did not happen"

    plot.colour_by_column("coefficient")
    plot.shape_by_column("condition")
    plot.clear_column_mapping()

    after = [brush.color().name() for brush in item.data["brush"]]
    assert after == before
    assert {point.symbol() for point in item.points()} == {"o"}


def test_the_way_back_only_appears_once_there_is_something_to_go_back_from(
        volcano):
    assert "own colouring" not in _joined(volcano)

    volcano.colour_by_column("coefficient")

    assert "own colouring" in _joined(volcano)


def test_the_guide_support_plot_can_map_a_column_too(qtbot):
    """It holds a table per gene -- guides, concordance, the gene p -- and
    "colour the genes by their p-value" is exactly the question this control
    is for."""
    from spacr.qt.widgets.fast_plots import GuideAgreementPlot

    rng = np.random.default_rng(3)
    plot = GuideAgreementPlot()
    qtbot.addWidget(plot)
    plot.set_support(pd.DataFrame({
        "feature": [f"gene_fraction:gene[{i}]" for i in range(50)],
        "n_guides": rng.integers(1, 5, 50),
        "concordance": rng.random(50),
        "gene_p": rng.random(50),
        "single_guide": rng.random(50) < 0.2,
    }))

    assert plot.colour_by_column("gene_p") > 0


def test_the_column_mapping_dialog_is_wired_to_the_menu(volcano, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    answers = iter([("well_count", True), ("magma", True)])
    monkeypatch.setattr(QInputDialog, "getItem",
                        staticmethod(lambda *a, **k: next(answers)))

    _action(volcano, "Colour by a column").trigger()

    assert "Coloured by well_count (magma)" in volcano._status.text()


def test_the_shape_dialog_is_wired_to_the_menu(volcano, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    monkeypatch.setattr(QInputDialog, "getItem",
                        staticmethod(lambda *a, **k: ("n_guides", True)))

    _action(volcano, "Shape by a column").trigger()

    assert "Shaped by n_guides" in volcano._status.text()


def test_cancelling_the_column_picker_leaves_the_plot_alone(volcano,
                                                            monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    monkeypatch.setattr(QInputDialog, "getItem",
                        staticmethod(lambda *a, **k: ("well_count", False)))

    _action(volcano, "Colour by a column").trigger()

    assert "Coloured by" not in volcano._status.text()


# --------------------------------------------------------------------------- #
#  Instruction 106: greyed out, and SAYING why
# --------------------------------------------------------------------------- #

def test_a_histogram_greys_the_point_controls_rather_than_hiding_them(
        histogram):
    """A p-value histogram is bars. "Point size" on it is the plainest case
    of a control that looks live and does nothing -- and hiding it instead
    would leave a user hunting a menu for an entry they have seen elsewhere,
    with nothing saying where it went."""
    menu = histogram.build_style_menu()
    greyed = {action.text() for action in menu.actions()
              if not action.isEnabled() and not action.isSeparator()}

    assert any("Point size" in text for text in greyed), greyed
    assert any("Shape by a column" in text for text in greyed), greyed


def test_the_greyed_entry_is_disabled_AND_says_why(histogram):
    """Either half alone is a failure. Disabled with no reason is a dead
    control; a reason on a live control is a lie."""
    action = _action(histogram, "Point size")

    assert not action.isEnabled()
    assert "nothing on this plot is drawn as points" in action.text()


def test_the_reason_is_on_the_tooltip_as_well_as_in_the_label(histogram):
    """Long entries elide on some themes; the tooltip is the fallback."""
    action = _action(histogram, "Point size")

    assert action.toolTip() == "nothing on this plot is drawn as points"


def test_the_menu_shows_action_tooltips_so_the_fallback_is_reachable(
        histogram):
    """Qt hides action tooltips unless the menu asks for them. Without this
    the fallback would exist and never be seen."""
    assert histogram.build_style_menu().toolTipsVisible()


def test_a_plot_with_no_table_says_that_is_why_it_cannot_map_a_column(
        controls):
    """The control panel is handed arrays of effects, not a frame. "Choose any
    column" has no answer there, and the honest one is to say so."""
    action = _action(controls, "Colour by a column")

    assert not action.isEnabled()
    assert "holds no table" in action.text()


def test_a_plot_whose_table_has_no_numeric_column_says_that_instead(qtbot):
    """A different reason from "no table", and the user needs to know which:
    one is fixed by loading a different run, the other never can be."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(pd.DataFrame({
        "feature": [f"fraction:grna[{i}_1]" for i in range(6)],
        "condition": ["nc", "pc", "other", "other", "nc", "pc"],
    }))

    assert plot.colour_map_reason() == (
        "no column here is a number a colour scale could read")


def test_a_plot_with_no_lines_greys_the_line_control(qtbot):
    """"a line width on a plot with no lines" is the third example instruction
    128 O gives by name."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)

    action = _action(plot, "Line colour and width")
    assert not action.isEnabled()
    assert "no lines on it" in action.text()


def test_a_plot_that_has_lines_offers_the_control_plainly(volcano):
    """The other half of the rule: an applicable control carries no excuse."""
    action = _action(volcano, "Line colour and width")

    assert action.isEnabled()
    assert action.text() == "Line colour and width…"


# --------------------------------------------------------------------------- #
#  The SVG that never worked
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("suffix", ["pdf", "svg", "png"])
def test_every_offered_format_actually_writes_a_file(volcano, tmp_path,
                                                     suffix):
    """SVG was on the Export dialog's filter and raised every single time it
    was chosen. A format that always fails must not be offered, and the way
    to keep offering it is to make it work."""
    path = tmp_path / f"plot.{suffix}"

    assert volcano.export(str(path)) == str(path)
    assert path.stat().st_size > 0


@pytest.mark.parametrize("build", ["volcano", "histogram", "controls"])
def test_every_plot_in_this_module_can_be_written_as_svg(request, tmp_path,
                                                         build):
    """The failure was not specific to the volcano: it was the ViewBox frame
    and every round scatter marker, so it hit all of them."""
    plot = request.getfixturevalue(build)

    plot.export(str(tmp_path / f"{build}.svg"))

    assert (tmp_path / f"{build}.svg").stat().st_size > 0


def test_the_saved_vector_page_draws_its_points_rather_than_pasting_them(
        volcano, tmp_path):
    """A ScatterPlotItem draws its markers from a cached pixmap atlas -- that
    cache is why 1,215 points pan at no cost -- and a plain scene render
    copies those pixmaps straight into the vector file. Measured before this
    change: 50 `<image>` elements and ONE `<path>`, i.e. fifty little bitmaps
    of a dot in a file that claims to be vector."""
    volcano.export(str(tmp_path / "v.svg"))

    svg = (tmp_path / "v.svg").read_text()
    assert svg.count("<image") == 0, "the points were pasted in as bitmaps"
    assert svg.count("<path") >= N, (
        f"only {svg.count('<path')} paths for {N} points")


def test_the_text_stays_text_in_the_saved_page(volcano, tmp_path):
    """The reason to want vector at all: a tick number a reader can select,
    and an axis that survives being scaled into a figure panel."""
    volcano.export(str(tmp_path / "v.svg"))

    assert (tmp_path / "v.svg").read_text().count("<text") > 5


def test_pyqtgraphs_own_svg_exporter_still_raises_on_this_scene(volcano,
                                                                tmp_path):
    """A TRIPWIRE, deliberately. `correctCoordinates` parses a path's `d`
    attribute by splitting on spaces and unpacking each token as `x,y`; a
    closepath token is the single letter `Z`, which has no comma. Every closed
    shape ends in one, so a round scatter marker and the ViewBox's own frame
    both trip it, and there is no scene-level workaround -- a round point IS a
    closed path, and clearing the ViewBox border changed nothing when it was
    measured.

    WHEN THIS TEST FAILS, pyqtgraph has fixed it and `_export_svg` can be
    reconsidered. Until then it is the evidence that routing around the
    library was the right call rather than a preference.
    """
    from pyqtgraph import exporters

    with pytest.raises(ValueError, match="not enough values to unpack"):
        exporters.SVGExporter(volcano.plot.plotItem).export(
            str(tmp_path / "upstream.svg"))


# --------------------------------------------------------------------------- #
#  The edges: what each control does when it has nothing to work on
# --------------------------------------------------------------------------- #

def test_a_plot_with_no_table_offers_no_columns_at_all(controls):
    """`shape_columns` on a plot with no frame must be an empty list rather
    than a crash: the menu asks it on every right-click, including on the
    panels that were handed bare arrays."""
    assert controls.frame() is None
    assert controls.numeric_columns() == []
    assert controls.shape_columns() == []


def test_the_font_size_is_reported_back_so_the_dialog_can_open_on_it(volcano):
    """A dialog that reopens at 10 after the user set 20 makes them retype
    the answer they already gave."""
    assert volcano.font_size() is None

    volcano.set_font_size(14)

    assert volcano.font_size() == 14


def test_the_font_size_dialog_is_wired_to_the_menu(volcano, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    monkeypatch.setattr(QInputDialog, "getInt",
                        staticmethod(lambda *a, **k: (22, True)))

    _action(volcano, "Font size").trigger()

    assert volcano.font_size() == 22


def test_a_cancelled_font_size_leaves_the_plot_alone(volcano, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    monkeypatch.setattr(QInputDialog, "getInt",
                        staticmethod(lambda *a, **k: (22, False)))

    _action(volcano, "Font size").trigger()

    assert volcano.font_size() is None


def test_without_pyqtgraph_there_are_simply_no_lines_to_restyle(volcano,
                                                                monkeypatch):
    """The widget still constructs when the optional library is absent -- that
    is the whole of `_Absorbs` -- so every control has to answer "nothing"
    rather than reach into a plot that was never built."""
    from spacr.qt.widgets import fast_plots

    monkeypatch.setattr(fast_plots, "HAVE_PYQTGRAPH", False)

    assert volcano.line_items() == []
    assert volcano.line_reason() == "this plot has no lines on it"


def test_colouring_a_plot_that_holds_no_table_is_refused_out_loud(controls):
    with pytest.raises(ValueError, match="holds no table"):
        controls.colour_by_column("anything")


def test_shaping_a_plot_that_holds_no_table_is_refused_out_loud(controls):
    with pytest.raises(ValueError, match="holds no table"):
        controls.shape_by_column("anything")


def test_a_column_that_is_not_there_is_answered_with_the_ones_that_are(
        volcano):
    """A typo in a column name is the commonest way to reach this, and the
    answer a user needs is the list they were choosing from."""
    with pytest.raises(ValueError, match="no column 'coefficnet'"):
        volcano.colour_by_column("coefficnet")
    with pytest.raises(ValueError, match="no column 'conditon'"):
        volcano.shape_by_column("conditon")


def test_points_that_carry_no_row_are_skipped_rather_than_mis_joined(volcano):
    """A scatter whose points do not say which row they came from cannot be
    mapped, and guessing -- taking them in drawing order -- is the exact
    failure `add_scatter` exists to prevent: something lights up, in the
    direction nobody questions, and it is the wrong guide."""
    import pyqtgraph as pg

    stray = pg.ScatterPlotItem(x=[0.0, 0.1], y=[1.0, 1.1],
                               data=[None, None], size=6)
    volcano.plot.addItem(stray)
    assert stray in volcano._scatter_items()

    painted = volcano.colour_by_column("coefficient")

    assert painted == N, "the stray scatter was mapped from its drawing order"
    assert volcano.shape_by_column("condition") == N


def test_an_item_with_no_pen_at_all_does_not_break_the_line_restyle():
    """`_pen_of` is asked about every item on the plot. One that has no pen is
    a real answer, and the restyle gives it the house's muted ink rather than
    raising in the middle of a loop over the others."""
    from spacr.qt.widgets.fast_plots import FastPlot

    assert FastPlot._pen_of(object()) is None


def test_going_back_to_automatic_size_does_nothing_when_none_was_set(qtbot):
    """The entry is always on the menu, so it is always clickable -- including
    by a user who never fixed the size. It must not then release the floors
    the panel set."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.setMinimumHeight(240)

    plot.clear_screen_size()

    assert plot.minimumHeight() == 240


def test_an_item_with_nothing_drawn_on_it_writes_no_page(tmp_path):
    """A zero-sized page is not a figure; it is a file that opens blank and
    looks like a corrupted export. Writing nothing is the honest answer."""
    from PySide6.QtCore import QRectF

    from spacr.qt.widgets.fast_plots import FastPlot

    class _Nothing:
        def scene(self):
            return None

        def boundingRect(self):
            return QRectF(0, 0, 0, 0)

    assert FastPlot._page_source(_Nothing()) == (None, 0.0)

    FastPlot._export_pdf(_Nothing(), tmp_path / "empty.pdf")
    FastPlot._export_svg(_Nothing(), tmp_path / "empty.svg")

    assert not (tmp_path / "empty.pdf").exists()
    assert not (tmp_path / "empty.svg").exists()


def test_the_screen_size_dialog_is_wired_to_the_menu(volcano, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    answers = iter([(640, True), (480, True)])
    monkeypatch.setattr(QInputDialog, "getInt",
                        staticmethod(lambda *a, **k: next(answers)))

    _action(volcano, "Size on screen…").trigger()

    assert (volcano.width(), volcano.height()) == (640, 480)


@pytest.mark.parametrize("answers", [
    [(640, False)],                      # cancelled on the width
    [(640, True), (480, False)],         # cancelled on the height
])
def test_a_cancelled_screen_size_leaves_the_widget_free(volcano, monkeypatch,
                                                        answers):
    from PySide6.QtWidgets import QInputDialog

    replies = iter(answers)
    monkeypatch.setattr(QInputDialog, "getInt",
                        staticmethod(lambda *a, **k: next(replies)))

    _action(volcano, "Size on screen…").trigger()

    assert volcano.maximumWidth() > 640, "the widget was pinned anyway"


def test_the_page_size_dialog_is_wired_to_the_menu(volcano, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    answers = iter([(85.0, True), (60.0, True)])
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: next(answers)))

    _action(volcano, "Exported page size").trigger()

    assert volcano.export_size() == (85.0, 60.0)


def test_a_page_height_of_nothing_means_follow_the_plot(volcano, monkeypatch):
    """The dialog says so in words -- "0 follows the plot's own shape" -- and
    this is that sentence as behaviour: no height stored, so the aspect is
    taken from the plot at export time."""
    from PySide6.QtWidgets import QInputDialog

    answers = iter([(85.0, True), (0.0, True)])
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: next(answers)))

    _action(volcano, "Exported page size").trigger()

    assert volcano.export_size() == (85.0, None)


@pytest.mark.parametrize("answers", [
    [(85.0, False)],
    [(85.0, True), (60.0, False)],
])
def test_a_cancelled_page_size_leaves_the_page_alone(volcano, monkeypatch,
                                                     answers):
    from PySide6.QtWidgets import QInputDialog

    before = volcano.export_size()
    replies = iter(answers)
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: next(replies)))

    _action(volcano, "Exported page size").trigger()

    assert volcano.export_size() == before


def test_a_cancelled_line_width_never_reaches_the_colour_question(
        volcano, monkeypatch):
    """Declining the first question ends the whole gesture. Asking for a
    colour after the user has said no is a dialog that will not take no."""
    from PySide6.QtWidgets import QColorDialog, QInputDialog

    asked = []
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: (6.0, False)))
    monkeypatch.setattr(QColorDialog, "getColor",
                        staticmethod(lambda *a, **k: asked.append(1)))
    before = volcano._pen_of(volcano.line_items()[0]).widthF()

    _action(volcano, "Line colour and width").trigger()

    assert not asked, "the colour dialog opened after the user cancelled"
    assert volcano._pen_of(volcano.line_items()[0]).widthF() == before


def test_only_the_colour_or_only_the_width_can_be_changed(volcano):
    """Both arguments are optional, and a caller that passes one must not
    silently reset the other to a default it never asked for."""
    volcano.set_line_style("#111111", 5.0)

    volcano.set_line_style(colour="#222222")
    assert volcano._pen_of(volcano.line_items()[0]).widthF() == 5.0

    volcano.set_line_style(width=2.0)
    assert volcano._pen_of(
        volcano.line_items()[0]).color().name() == "#222222"


def test_a_scatter_that_was_never_mapped_is_left_alone_by_the_restore(volcano):
    """"Back to this plot's own colouring" walks every scatter. One the
    mapping never reached has no saved original, and inventing one would
    overwrite whatever it was actually drawn with."""
    import pyqtgraph as pg

    volcano.colour_by_column("coefficient")
    stray = pg.ScatterPlotItem(x=[0.0], y=[1.0], data=[None], size=6)
    volcano.plot.addItem(stray)

    assert volcano.clear_column_mapping() == 1


def test_a_scatter_with_no_points_at_all_reports_no_rows():
    """`_rows_of` is asked about every scatter on the plot, and an empty one
    is a real state -- a compartment nothing matched draws exactly that."""
    import pyqtgraph as pg

    from spacr.qt.widgets.fast_plots import FastPlot

    assert FastPlot._rows_of(pg.ScatterPlotItem(x=[], y=[])) is None


def test_a_mark_whose_data_is_not_a_scatters_data_reports_no_rows():
    """The mapping walks whatever is on the plot. Something carrying a `data`
    attribute that is not pyqtgraph's per-point record array is not a scatter
    this can map, and saying so beats raising in the middle of the loop."""
    from spacr.qt.widgets.fast_plots import FastPlot

    class _Odd:
        data = [1, 2, 3]

    assert FastPlot._rows_of(_Odd()) is None

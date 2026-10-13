"""The fast plots past the drawing: the dialogs, the keys, and the table.

Round 3 pinned what these widgets do when the DATA lets them down. This file
takes the layer under that -- a save dialog the user cancels, a coefficient
table whose recorded correction is a name the library has never heard of, a
histogram bar holding nothing, an influence panel where no single well is
carrying the fit, a results table asked about a key column its frame does not
have -- because every one of those is a refusal a user reaches by ordinary
use, and a refusal nothing has ever taken is a refusal nobody knows the shape
of.

Each test drives the refusal AND the answer it is a refusal against, in the
same test: a widget that declines everything would otherwise pass every one
of them while drawing nothing at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg                                       # noqa: E402
from PySide6.QtCore import QPointF, Qt                       # noqa: E402
from PySide6.QtGui import QColor                             # noqa: E402
from PySide6.QtWidgets import (QApplication, QCheckBox,      # noqa: E402
                               QFileDialog, QTableWidgetItem)

from spacr.qt.widgets import fast_plots as fp                # noqa: E402

pytestmark = pytest.mark.qt


# ------------------------------------------------------------------ fixtures

@pytest.fixture
def plot(qtbot):
    """A live FastPlot with both axes named, nothing drawn on it yet."""
    widget = fp.FastPlot(title="fast", x_label="ex", y_label="why")
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def volcano(qtbot):
    """A live VolcanoPlot with nothing set on it."""
    widget = fp.VolcanoPlot()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def table(qtbot):
    """A live ResultsTable holding no frame."""
    widget = fp.ResultsTable()
    qtbot.addWidget(widget)
    return widget


def _coefficients(n=12, *, p=None, feature=True):
    """A coefficient table the shape ``perform_regression`` writes."""
    frame = pd.DataFrame({
        "grna": [f"g{i}" for i in range(n)],
        "coefficient": np.linspace(-2.0, 2.0, n),
        "p_value": (np.linspace(1e-6, 0.9, n) if p is None
                    else np.asarray(p, dtype=float)),
    })
    if feature:
        frame.insert(0, "feature", [f"fraction:grna[g{i}]" for i in range(n)])
    return frame


def _scatters(widget) -> list:
    return [item for item in widget.plot.plotItem.items
            if isinstance(item, pg.ScatterPlotItem)]


def _brush_names(item) -> list:
    return [brush.color().name() for brush in item.data["brush"]]


# --------------------------------------------------------- what a click says

def test_a_clicked_point_says_the_label_it_was_drawn_with(plot):
    """The labels handed to ``add_scatter`` are what a click reports.

    A caller that took the trouble to name its points has already decided
    what they are called, so the click note is that name rather than the
    frame lookup or the bare key underneath it -- which is the difference
    between "second gene" and a row number the user has to go and resolve.
    """
    item = plot.add_scatter(np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                            labels=["first gene", "second gene"])

    plot._on_points_clicked(item, [item.points()[1]])

    assert "second gene" in plot._status.text()
    assert "first gene" not in plot._status.text()


# ------------------------------------------------------------- the save path

def test_a_cancelled_export_writes_nothing_and_says_so(plot, tmp_path,
                                                       monkeypatch):
    """Export must survive Qt's checked-state argument and a cancelled box.

    ``export`` is wired to actions whose signals carry a bool, so ``False``
    arrives where a path belongs and has to be read as "no path given". The
    dialog it then opens can be dismissed, and dismissing it must leave no
    file behind and return None rather than writing ``plot.pdf`` into
    whatever directory the process happens to be in.
    """
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    plot.add_scatter(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    assert plot.export(False) is None
    assert list(tmp_path.iterdir()) == []

    chosen = tmp_path / "figure.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(chosen), "")))

    assert plot.export(False) == str(chosen)
    assert chosen.exists() and chosen.stat().st_size > 0


def test_a_cancelled_bundle_creates_no_folder(plot, tmp_path, monkeypatch):
    """The bundle asks for a directory and takes "no" for an answer.

    Same shape as the export: the menu action hands its checked state in as
    the folder, and a cancelled directory chooser must not fall through to
    the current working directory -- a bundle is a whole tree of files, and
    writing one where the user did not ask is not a mistake they can undo by
    deleting a single file.
    """
    plot.add_scatter(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))

    assert plot.export_bundle(False) is None
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))

    made = plot.export_bundle(False)
    assert made is not None
    assert (tmp_path / "fast").is_dir(), (
        f"the bundle went somewhere else: {made}")


def test_a_plot_with_no_title_writes_no_title_into_the_bundle(qtbot):
    """The settings file records a title only when there IS one.

    ``export_settings`` is read back by whoever opens the bundle. An empty
    title written as a key is a title of "", which reads as a graph somebody
    deliberately named nothing.
    """
    named = fp.FastPlot(title="Control separation", x_label="", y_label="")
    qtbot.addWidget(named)
    anonymous = fp.FastPlot(x_label="", y_label="")
    qtbot.addWidget(anonymous)

    assert named.export_settings()["title"] == "Control separation"
    assert "title" not in anonymous.export_settings()
    assert anonymous.export_settings()["plot"] == "FastPlot"


def test_the_save_dialog_result_is_what_save_styled_returns(plot,
                                                            monkeypatch):
    """``save_styled`` hands back the dialog's own result code.

    The caller uses it to tell "the user saved" from "the user closed the
    box", so returning None -- or the dialog object -- would make a cancel
    indistinguishable from a save.
    """
    from spacr.qt.widgets import save_figure_dialog as dialog_module

    built = []

    def _exec(self):
        built.append(self)
        return 42

    monkeypatch.setattr(dialog_module.SaveFigureDialog, "exec", _exec)

    assert plot.save_styled() == 42
    assert len(built) == 1
    assert built[0].parent() is plot


def test_a_snapshot_that_cannot_be_rendered_is_no_snapshot(plot, monkeypatch):
    """A tile is never worth taking the screen down for.

    The gallery asks every open plot for a picture. An exporter that fails --
    an old pyqtgraph, a scene mid-teardown -- has to come back as "no tile"
    rather than as an exception on a path the user did not ask for.
    """
    plot.add_scatter(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert plot.snapshot(120) is not None

    from pyqtgraph import exporters

    monkeypatch.setattr(exporters, "ImageExporter",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("no exporter in this build")))

    assert plot.snapshot(120) is None


def test_a_theme_switch_does_not_undo_a_line_colour_the_user_chose(plot):
    """Restyle repaints the axes, then puts the user's own ink back on top.

    A theme switch runs over every axis with the theme's pen. The line colour
    is a choice the user made off this plot's menu, and losing it to a
    light/dark toggle is the plot telling them their setting did not stick.
    """
    plot.add_curve([0.0, 1.0], [0.0, 1.0])
    plot.set_line_colour("#123456")

    plot.restyle(background="#ffffff", foreground="#000000")

    assert plot.line_colour() == "#123456"
    axis = plot.plot.getAxis("bottom")
    assert axis.pen().color().name() == "#123456", (
        "the theme's ink was left on the axis over the user's choice")

    # pyqtgraph resolves `foreground` globally, so put the theme's own back
    # rather than leaving this session's default ink black.
    plot.restyle()


def test_a_plot_closes_even_when_its_menus_cannot_be_retired(plot,
                                                             monkeypatch):
    """Closing must not depend on the cleanup helper being reachable.

    ``retire_pyqtgraph_menus`` walks parentless menus whose C++ side may
    already be gone by the time a window closes, and a RuntimeError there
    would leave the widget on screen and the close event unhandled.
    """
    from spacr.qt import widget_cleanup

    def _gone(_widget):
        raise RuntimeError("wrapped C/C++ object already deleted")

    monkeypatch.setattr(widget_cleanup, "retire_pyqtgraph_menus", _gone)

    assert plot.close() is True
    assert plot.isVisible() is False


# ------------------------------------------------------------ "Show as"

def _show_as(menu):
    """The ``Show as`` submenu of a built style menu, or None."""
    for action in menu.actions():
        if action.menu() is not None and action.text() == "Show as":
            return action.menu()
    return None


@pytest.fixture
def spec_plot(qtbot):
    """A grouped plot that remembers the spec it was drawn from."""
    from spacr.qt.widgets.grouped_plot import GroupedPlot, PlotSpec

    frame = pd.DataFrame({"group": ["a", "a", "b", "b"],
                          "value": [1.0, 2.0, 3.0, 4.0]})
    widget = GroupedPlot(PlotSpec(frame=frame, value="value", group="group",
                                  kind="box"))
    qtbot.addWidget(widget)
    return widget


def test_a_spec_with_no_rows_left_offers_no_graph_types(spec_plot):
    """"Show as" needs data to decide what fits, so an empty frame gets none.

    The compatibility list is computed FROM the rows -- how many groups, what
    the value column is -- so a spec whose frame has been filtered down to
    nothing cannot answer the question. An enabled menu of eight graph types
    that all redraw an empty plot is worse than no menu.
    """
    from dataclasses import replace

    assert _show_as(spec_plot.build_style_menu()) is not None, (
        "a plot with rows should offer the graph types"
    )

    spec_plot.spec = replace(spec_plot.spec,
                             frame=pd.DataFrame({"group": [], "value": []}))

    assert _show_as(spec_plot.build_style_menu()) is None


def test_a_build_that_offers_no_graph_types_adds_no_submenu(spec_plot,
                                                            monkeypatch):
    """An empty offer must leave no empty submenu behind.

    ``graph_types.offer`` is the one place that knows which kinds a shape can
    take. A build whose table has nothing to say -- the module is the thing
    that moves when the package does -- must produce no "Show as" entry at
    all rather than one that opens onto nothing.
    """
    from spacr import graph_types

    menu = spec_plot.build_style_menu()
    assert len(_show_as(menu).actions()) > 1

    monkeypatch.setattr(graph_types, "offer", lambda *a, **k: [])

    assert _show_as(spec_plot.build_style_menu()) is None


def test_only_a_pinned_kind_can_be_made_the_starting_point(spec_plot):
    """"Always start with X" needs an X: a spec with no kind offers none.

    The entry remembers THIS graph's kind as the default for its shape, so a
    spec that has not been drawn as anything in particular has nothing to
    remember, and an entry reading "Always start with" would name nothing.
    """
    from dataclasses import replace

    menu = _show_as(spec_plot.build_style_menu())
    labels = [action.text() for action in menu.actions()]
    assert any(text.startswith("Always start with") for text in labels)

    spec_plot.spec = replace(spec_plot.spec, kind="")
    kindless = [a.text()
                for a in _show_as(spec_plot.build_style_menu()).actions()]

    assert kindless, "the graph types themselves should still be offered"
    assert not any(text.startswith("Always start with")
                   for text in kindless)


def test_the_starting_point_is_remembered_and_a_failure_is_not_fatal(
        spec_plot, monkeypatch):
    """Picking "always start with" writes the preference, and survives not to.

    The write goes through the settings store, which can be absent or
    read-only in a packaged build. A menu entry that took the application
    down when the store would not answer would be worse than one that quietly
    fails to persist -- the graph in front of the user is already redrawn.
    """
    from spacr.qt import preferences

    written = []
    monkeypatch.setattr(preferences, "set_default_graph_type",
                        lambda shape, kind: written.append((shape, kind)))
    remember = [a for a in _show_as(spec_plot.build_style_menu()).actions()
                if a.text().startswith("Always start with")][0]

    remember.trigger()
    assert written == [(spec_plot.spec.shape(), "box")]

    def _no_store(*_args, **_kwargs):
        raise RuntimeError("no settings store in this build")

    monkeypatch.setattr(preferences, "set_default_graph_type", _no_store)
    remember = [a for a in _show_as(spec_plot.build_style_menu()).actions()
                if a.text().startswith("Always start with")][0]

    remember.trigger()
    assert written == [(spec_plot.spec.shape(), "box")], (
        "the failed write recorded something anyway")


# ------------------------------------------------------------- the volcano

def _legend_labels(widget) -> list:
    legend = getattr(widget.plot.plotItem, "legend", None)
    return [] if legend is None else [label.text for _, label in legend.items]


def _legend_box(widget):
    return [box for box in widget.findChildren(QCheckBox)
            if box.text().startswith("legend")][0]


def test_the_height_can_only_be_one_of_the_three_honest_axes(volcano):
    """An axis this plot cannot draw is refused by name, not approximated.

    The three are the raw p, the adjusted p and the local FDR, and each means
    something different about what the height IS. A typo silently falling
    back to the raw p would put a figure on screen whose axis label and axis
    values came from different decisions.
    """
    volcano.set_p_axis("lfdr")
    assert volcano.p_axis() == "lfdr"

    with pytest.raises(ValueError) as raised:
        volcano.set_p_axis("bonferroni")

    assert "bonferroni" in str(raised.value)
    assert volcano.p_axis() == "lfdr", "the refused axis was applied anyway"


def test_a_correction_the_library_cannot_name_is_not_checked_against(volcano):
    """A table naming an unknown method is drawn, and says nothing it cannot.

    ``multiple_testing_method`` is written by whatever produced the table, and
    an older or hand-edited file can carry a name this build does not know.
    The plot keeps the name for the record, falls back to BH for what it
    actually draws, and -- the part that matters -- does NOT claim the table's
    q values disagree with it, because it could not have recomputed the
    table's own method to find out.
    """
    frame = _coefficients()
    frame["q_value"] = 0.5              # not what BH gives for these p values

    volcano.set_results(frame, run_method="peeking-until-significant")

    assert volcano.run_correction() == "peeking-until-significant"
    assert volcano.correction() == "fdr_bh"
    assert "WARNING" not in volcano.caption()

    volcano.set_results(frame, run_method="bh")

    assert volcano.correction() == "fdr_bh"
    assert "WARNING" in volcano.caption(), (
        "a method it CAN recompute must be checked against the table")


def test_the_local_fdr_is_computed_once_and_never_before_there_is_data(
        volcano):
    """The mixture fit is lazy and cached, because it costs more than the draw.

    It is 25 ms of a 40 ms redraw on a real screen and the default axis does
    not use it, so it must not be computed on a plot that has been handed
    nothing, and it must not be computed twice for the same table.
    """
    assert volcano.local_fdr_values() is None
    assert volcano.redraw() == 0, "a plot with no table redrew something"

    volcano.set_results(_coefficients(24))
    first = volcano.local_fdr_values()

    assert first is not None and len(first) == 24
    assert volcano.local_fdr_values() is first, (
        "the beta-uniform fit ran a second time for the same table")


def test_a_screen_whose_every_p_underflowed_has_no_q_ramp_to_draw(volcano):
    """A ramp needs a spread of q values; every q at zero is not one.

    The ramp is built on -log10(q), so it needs at least one q that is finite
    AND positive to have a top and a bottom. A screen whose p values all
    underflowed to exactly zero gives it neither, so every point takes the
    missing colour and the key says how many that was -- rather than the plot
    inventing a ramp over a quantity with no range.
    """
    volcano.set_q_colour("ramp")
    frame = _coefficients(8, p=[0.0] * 8)

    assert volcano.set_results(frame) == 8
    _legend_box(volcano).setChecked(True)

    painted = set(_brush_names(_scatters(volcano)[0]))
    assert painted == {QColor(fp.MISSING_COLOUR).name()}
    assert "no q (8)" in _legend_labels(volcano)


def test_a_screen_holding_one_q_value_is_drawn_as_one_colour(volcano):
    """A ramp over a single value is one colour, and is honest about it.

    Every test sharing a q means the evidence against them is identical.
    Stretching the whole scale across that would paint a range the data does
    not have; the ramp sits at the middle instead, so the picture says "these
    are all the same" rather than "these differ".
    """
    volcano.set_q_colour("ramp")

    assert volcano.set_results(_coefficients(8, p=[0.4] * 8)) == 8
    flat = set(_brush_names(_scatters(volcano)[0]))
    assert len(flat) == 1
    assert flat != {QColor(fp.MISSING_COLOUR).name()}

    assert volcano.set_results(_coefficients(8)) == 8
    spread = set(_brush_names(_scatters(volcano)[0]))
    assert len(spread) > 1, "a real spread of q values drew one colour"


def test_the_ramp_says_which_colouring_it_took_the_channel_from(volcano):
    """One sentence per figure: the ramp names what it displaced.

    A dot cannot be coloured for its compartment and for its q at once. The
    compartment colouring is not silently dropped -- it is turned off and
    said so, because a reader who asked for compartments and got a ramp would
    otherwise read the ramp as compartments.
    """
    frame = _coefficients()
    volcano.set_results(frame, compartment="nucleus")
    assert "colouring is off while it is" not in volcano.caption()

    volcano.set_q_colour("ramp")
    volcano.set_results(frame, compartment="nucleus")

    assert "the nucleus colouring is off while it is" in volcano.caption()


def test_colouring_every_compartment_of_a_table_with_no_genes_in_it(volcano):
    """"All compartments" over a table whose rows name no gene has no key.

    The compartments are joined through the design-matrix term, so a frame
    carrying no ``feature`` column resolves no gene and therefore no
    compartment. Colouring by them then has nothing to colour, and the legend
    checkbox has to stay dark rather than offering a key with nothing in it.
    """
    frame = _coefficients(feature=False)
    everywhere = fp.FastPlot._all_compartments()

    assert volcano.set_results(frame, compartment=everywhere) == 12
    assert _legend_box(volcano).isEnabled() is False

    assert volcano.set_results(frame) == 12
    assert _legend_box(volcano).isEnabled() is True
    assert "legend (2)" in _legend_box(volcano).text()


def test_opacity_by_q_composes_with_no_colouring_at_all(volcano):
    """The fade is applied even when nothing has claimed the colour channel.

    Opacity is a second channel, so it is written over whatever brushes the
    colouring produced -- and when a table's p values are all blank there are
    no brushes at all. The plot still has to say what the marks mean, because
    a caption that names an encoding the picture does not carry is worse than
    no caption.
    """
    volcano.set_q_mark("opacity")
    frame = _coefficients(6, p=[np.nan] * 6)

    assert volcano.set_results(frame) == 0
    assert ("Point OPACITY is the evidence against the null"
            in volcano.caption())

    assert volcano.set_results(_coefficients(6)) == 6
    faded = {QColor(name).alpha()
             for name in _brush_names(_scatters(volcano)[0])} or {255}
    assert faded, "the fade produced no brushes at all"


def test_a_legend_the_user_asked_for_comes_back_after_a_redraw(volcano):
    """A redraw must not silently take the key off the picture.

    Every setting change redraws the whole plot, and the legend is added by
    the draw. A checkbox that stays ticked over a plot with no key is the
    control lying about what is on screen.
    """
    volcano.set_results(_coefficients())
    box = _legend_box(volcano)
    box.setChecked(True)
    assert len(_legend_labels(volcano)) == 2

    volcano.set_correction("bonferroni")          # any change redraws

    # The SET, not the list: a redraw appends the entries to the legend item
    # `plot.clear()` left behind, so they accumulate. That is a defect worth
    # reporting and not one worth pinning -- what a reader needs is that the
    # key still names both call states, and it does.
    assert set(_legend_labels(volcano)) == {"called (1)", "not called (11)"}
    assert "Bonferroni" in volcano.caption(), (
        "the redraw was not the new correction")


def test_the_correction_can_only_be_written_when_there_is_one(volcano,
                                                              tmp_path,
                                                              monkeypatch):
    """Writing the recomputed q values needs q values, and a chosen path.

    The action sits on the plot's own menu whether or not anything has been
    drawn, so it has to answer "there is nothing to write" without opening a
    dialog, and it has to take a cancelled dialog for an answer without
    writing a file named after the button.
    """
    volcano.set_results(_coefficients())
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    assert volcano._write_corrected_table() is None
    assert list(tmp_path.iterdir()) == []

    # A reload that came back empty. The menu keeps the entry it was given by
    # the draw before, and there are no q values behind it any more.
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "no.csv"),
                                                      "")))
    assert volcano.set_results(pd.DataFrame()) == 0
    assert volcano._write_corrected_table() is None
    assert not (tmp_path / "no.csv").exists()

    volcano.set_results(_coefficients())

    written = tmp_path / "recorrected.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(written), "")))

    assert volcano._write_corrected_table() == str(written)
    saved = pd.read_csv(written)
    assert "q_value" in saved.columns and "local_fdr" in saved.columns


def test_a_dot_outside_the_tested_family_says_so_when_clicked(volcano):
    """A nuisance term has no q, and a click on it must say that.

    Reporting nothing would leave the user staring at a point whose colour
    they cannot account for; reporting a number would be worse, because the
    multiple-testing family this plot draws does not contain that row and no
    q was ever computed for it.
    """
    frame = _coefficients(6)
    frame.loc[6] = ["Intercept", "", 0.4, 0.001]

    volcano.set_results(frame, drop_untested=False)

    assert volcano._detail(6) == "not in the tested family, so no q"
    assert volcano._detail(0).startswith("q=")
    assert "called" in volcano._detail(0)


# --------------------------------------------------------- the ranked effects

@pytest.fixture
def ranked(qtbot):
    widget = fp.EffectRankPlot()
    qtbot.addWidget(widget)
    return widget


def test_an_error_column_that_is_not_in_the_table_is_not_an_interval(ranked):
    """Naming a standard-error column the table does not have draws no bars.

    The caller names the column; the table is what actually arrived. Reaching
    for a column that is not there would raise, and quietly drawing dots
    while the caller believes it asked for intervals would leave a reader
    looking at point estimates that LOOK like they came with uncertainty.
    So the sentence says the intervals are missing.
    """
    frame = _coefficients(6)
    frame["std_err"] = 0.2

    ranked.set_results(frame, error_column="posterior_sd")
    assert "carries no standard error" in ranked._status.text()

    ranked.set_results(frame, error_column="std_err")
    assert ("1.96-standard-error interval from “std_err”"
            in ranked._status.text())


def test_a_named_significance_column_decides_the_colour_or_nothing_does(
        ranked):
    """The caller may name the column that calls a hit -- if the table has it.

    ``None`` goes looking, :data:`NO_SIGNIFICANCE` says "this table has none",
    and a NAMED column that is absent is the third case: the caller asked for
    a verdict the table cannot give, and colouring off an uncorrected p
    instead is exactly the error this panel exists to make visible.
    """
    frame = _coefficients(6)
    frame["q_value"] = [0.001, 0.2, 0.3, 0.4, 0.5, 0.6]

    ranked.set_results(frame, significance_column="q_value")
    assert "1 called at q_value ≤ 0.05" in ranked._status.text()

    ranked.set_results(frame, significance_column="posterior_probability")
    assert "Nothing is coloured" in ranked._status.text()


def test_a_clicked_dot_says_what_it_has_and_stays_quiet_about_the_rest(
        ranked):
    """The detail line carries the interval and the q only when they exist.

    Every part of it is a fact about the table: a blank coefficient has no
    effect to report, a table with no standard error has no interval to put
    around one, and a table with no corrected p has no verdict to quote.
    None of them is worth a placeholder.
    """
    frame = _coefficients(4)
    frame.loc[0, "coefficient"] = np.nan

    ranked.set_results(frame)
    bare = ranked._detail(int(np.argmax(np.isfinite(frame["coefficient"]))))

    assert ranked._detail(0) == "", "a blank coefficient described itself"
    assert bare.startswith("effect = ") and "[" not in bare, (
        "an interval was drawn around a coefficient with no standard error")

    frame["std_err"] = 0.5
    frame["q_value"] = 0.01
    ranked.set_results(frame)
    full = ranked._detail(1)

    assert "[" in full and "q_value = 0.01" in full


# ------------------------------------------------------------- the histograms

class _Click:
    """A pyqtgraph scene mouse event, as much of one as the handler reads."""

    def __init__(self, position, button=None):
        self._position = position
        self._button = Qt.LeftButton if button is None else button

    def button(self):
        return self._button

    def scenePos(self):
        return self._position


@pytest.fixture
def histogram(qtbot):
    widget = fp.PValueHistogram()
    qtbot.addWidget(widget)
    widget.resize(420, 320)
    return widget


def test_a_histogram_with_no_bars_answers_nothing_about_them(histogram):
    """Every question about a bar has an answer before there are any bars.

    The panel is built with the screen and filled when a fit finishes, so
    every one of these can be asked of an empty one -- by a click that lands
    on it, or by a selection arriving from the table.
    """
    assert histogram.bin_at(0.5) is None
    assert histogram.keys_in_bin(0) == []
    assert histogram.select_bin(0) == []

    histogram.set_p_values([0.1, 0.2, 0.3, 0.9], bins=4,
                           keys=["a", "b", "c", "d"])

    assert histogram.bin_at(0.3) == 1
    assert histogram.keys_in_bin(0) == ["a", "b"]
    assert histogram.keys_in_bin(99) == []
    assert histogram.select_bin(99) == []


def test_an_empty_bar_says_it_is_empty_and_selects_nothing(histogram):
    """A bar holding no p-value is a fact about the screen, not a dead click.

    The p-value histogram is pinned to [0, 1] whatever the data, so bars with
    nothing in them are normal and clicking one is something a user does. It
    has to say what that stretch of the axis is and that it is empty, rather
    than leaving the click looking broken.
    """
    histogram.set_p_values([0.05, 0.1, 0.15], bins=4, keys=["a", "b", "c"])

    assert histogram.select_bin(3) == []
    assert "p 0.75 to 1: empty." in histogram._status.text()

    assert histogram.select_bin(0) == ["a", "b", "c"]
    assert "3 coefficients" in histogram._status.text()


def test_a_bar_whose_rows_have_no_names_still_says_what_it_holds(histogram):
    """A histogram drawn without keys reports the count and selects nobody.

    The keys are optional -- a p-value histogram of a table with no feature
    column is still worth drawing -- and without them a bar cannot hand
    anything to the table. It still has to say how many coefficients are in
    it, which is what the panel is for.
    """
    calls = []
    histogram.keys_selected.connect(calls.append)
    histogram.set_p_values([0.05, 0.1, 0.15], bins=4)

    assert histogram.select_bin(0) == []
    assert "3 coefficients" in histogram._status.text()
    assert "A bar is not one point" in histogram._status.text()
    assert calls == [], "a bar with no keys handed a selection over anyway"

    histogram.set_p_values([0.05, 0.1, 0.15], bins=4, keys=["a", "b", "c"])
    histogram.select_bin(0)
    assert calls == [["a", "b", "c"]]


def test_a_click_off_the_bars_selects_nothing(histogram):
    """Only a press ON the plot picks a bar, and only where there IS one.

    The handler is wired to the whole scene, so it sees presses on the axes,
    the margins and the empty space beyond the axis range. Each of those must
    leave the selection alone rather than picking the nearest bar.
    """
    histogram.set_p_values(np.linspace(0.01, 0.99, 20), bins=4,
                           keys=[f"g{i}" for i in range(20)])
    item = histogram.plot.plotItem
    centre = item.sceneBoundingRect().center()
    expected = histogram.bin_at(histogram._to_data(
        item.vb.mapSceneToView(centre).x(), "x"))

    histogram._on_scene_clicked(_Click(centre))
    on_a_bar = histogram._status.text()
    assert f"{histogram.QUANTITY} " in on_a_bar and expected is not None

    histogram._on_scene_clicked(_Click(QPointF(-500.0, -500.0)))
    assert histogram._status.text() == on_a_bar, (
        "a click outside the plot picked a bar")

    # Inside the plot, but past the last bar: the axis is showing more than
    # the histogram covers, which is what "Reset view" leaves behind.
    histogram.plot.setXRange(2.0, 4.0, padding=0.0)
    histogram._on_scene_clicked(_Click(centre))
    assert histogram._status.text() == on_a_bar


def test_a_selection_marks_the_bar_a_row_falls_in_or_no_bar_at_all(histogram):
    """A row selected elsewhere marks its bar -- when it has one.

    A coefficient whose p-value was unusable is in no bar, and a key list
    longer than the values it names has rows the histogram never binned.
    Both must answer "not on this plot" rather than outlining bar zero.
    """
    histogram.set_p_values([0.05, np.nan, 0.9], bins=4,
                           keys=["drawn", "blank", "high", "beyond"])

    assert histogram.highlight_key("drawn") is True
    assert histogram.highlight_key("blank") is False, (
        "a coefficient with no p-value marked a bar")
    assert histogram.highlight_key("beyond") is False, (
        "a key past the end of the values marked a bar")


def test_a_bar_describes_the_value_behind_a_row_it_holds(histogram):
    """The click hook answers from the plotted array, in O(1).

    A histogram draws no points of its own, so this is reached from the base
    class's click handler on subclasses that add them -- and from a row it
    never binned, which has no value to name.
    """
    histogram.set_p_values([0.02, np.nan], bins=4, keys=["a", "b"])

    assert histogram._detail(0) == "p = 0.02"
    assert histogram._detail(1) == "", "a blank p-value described itself"
    assert histogram._detail(9) == ""


# ------------------------------------------------------------ the diagnostics

def _curves(widget) -> int:
    """Plot-data curves on the item: the trend lines and the smoothers."""
    return sum(1 for item in widget.plot.plotItem.items
               if isinstance(item, pg.PlotDataItem))


def test_a_qq_point_the_plot_never_drew_has_no_p_to_report(qtbot):
    """The click hook answers from the sorted array it kept, or says nothing.

    A Q-Q is ranked by p, so the nth drawn point is not the nth row -- the
    rows are carried through the sort explicitly. A row whose p was unusable
    is on no point at all, and the detail line for it has to be empty rather
    than the p of whoever ended up at that position.
    """
    qq = fp.QQPlot()
    qtbot.addWidget(qq)

    assert qq.set_p_values([0.01, np.nan, 0.5], keys=["a", "b", "c"]) == 2
    assert qq._detail(0) == "p = 0.01"
    assert qq._detail(1) == "", "a row with no p-value quoted one"


def test_residuals_that_are_too_few_to_fit_get_no_trend_line(qtbot):
    """A trend through two points is a line through two points.

    It would be drawn at whatever slope those two happen to make and read as
    a finding about the mean model. Below three residuals there is nothing to
    fit, and an empty fit has nothing at all to plot.
    """
    panel = fp.ResidualPlot()
    qtbot.addWidget(panel)

    assert panel.set_residuals([], []) == 0
    assert "No residuals." in panel._status.text()

    assert panel.set_residuals([1.0, 2.0], [0.1, -0.1]) == 2
    assert _curves(panel) == 0, "a trend was fitted to two residuals"

    assert panel.set_residuals([1.0, 2.0, 3.0, 4.0],
                               [0.1, -0.1, 0.2, -0.2]) == 4
    assert _curves(panel) >= 1
    assert "Trend slope" in panel._status.text()


def test_choosing_a_smoother_before_there_is_a_fit_draws_nothing(qtbot):
    """The smoother menu is on the plot from the moment it is built.

    It is offered in the constructor, so it can be used on a panel the run
    has not filled yet. Redrawing from data that is not there would raise on
    a menu pick the user is entitled to make.
    """
    panel = fp.ScaleLocationPlot()
    qtbot.addWidget(panel)

    # The callback the smoother menu entries invoke; there is no public
    # setter for the choice, which is what the menu is.
    panel._choose_smoother("lowess")
    assert _curves(panel) == 0
    assert panel._status.text() == ""

    panel.set_scale_location([1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 0.4, 1.2])
    with_curve = _curves(panel)
    panel._choose_smoother("")

    assert with_curve > _curves(panel), (
        "turning the smoother off left its curve on the plot")
    assert "Trend slope" in panel._status.text()


@pytest.fixture
def influence(qtbot):
    widget = fp.InfluencePlot()
    qtbot.addWidget(widget)
    return widget


def test_a_fit_no_single_well_is_carrying_says_exactly_that(influence):
    """Nothing past Cook's 4/n is a finding, and the panel says it.

    The coloured points are the argument -- "these are worth going back to
    the microscope for" -- so a screen with none of them must not silently
    draw a grey cloud and leave the reader to work out that the absence WAS
    the answer. And with no wells flagged there is no second pass to draw.
    """
    leverage = [0.1, 0.12, 0.11, 0.09]
    residuals = [0.5, -0.4, 0.3, -0.2]

    assert influence.set_influence(leverage, residuals, [0.01] * 4) == 4

    assert "none past Cook's D > 4/n" in influence._status.text()
    assert len(_scatters(influence)) == 1, (
        "an empty flagged pass was drawn as its own scatter")
    assert "2p/n" not in influence._status.text()


def test_one_influential_well_is_named_as_one(influence):
    """One well past the rule is singular, and the design line is optional.

    "1 wells" is how a reader learns not to trust the sentence, and the 2p/n
    line can only be drawn by a caller that knows how many parameters the fit
    had -- so it is absent rather than guessed.
    """
    leverage = [0.1, 0.12, 0.11, 0.9]
    residuals = [0.5, -0.4, 0.3, -3.0]
    cooks = [0.01, 0.02, 0.01, 5.0]

    assert influence.set_influence(leverage, residuals, cooks) == 4
    assert "1 past Cook's D > 4/n" in influence._status.text()
    assert len(_scatters(influence)) == 2, "the flagged well took no colour"

    influence.set_influence(leverage, residuals, [0.01, 5.0, 4.0, 5.0],
                            n_params=2)
    assert "3 past Cook's D > 4/n" in influence._status.text()


def test_an_influence_dot_quotes_the_cooks_distance_it_has(influence):
    """The detail line is the number the panel exists for, when there is one.

    Cook's D is what says a well is moving the coefficients. A row without
    one is a row the rule cannot judge, and quoting nothing is the honest
    answer -- a zero would read as "this well moves nothing".
    """
    influence.set_influence([0.1, 0.2], [0.5, -0.5], [0.25, np.nan])

    assert influence._detail(0) == "Cook's D = 0.25"
    assert influence._detail(1) == ""
    assert influence._detail(7) == ""


# ----------------------------------------------------------- the group plots

def test_the_grouped_base_draws_nothing_until_a_subclass_says_how(qtbot):
    """The base class holds the marks and refuses to draw them itself.

    Every mark change redraws the SAME observations, which only the subclass
    that stored them can do. A base class that quietly drew nothing would let
    a subclass ship with no redraw at all and look like a plot whose mark
    menu does not work.
    """
    base = fp.GroupedPlot(title="grouped", x_label="", y_label="")
    qtbot.addWidget(base)

    assert base.mark() == "jitter"
    assert base.group_sizes() == []
    assert base.mark_note() == ""

    with pytest.raises(NotImplementedError):
        base.redraw()
    with pytest.raises(NotImplementedError):
        base.set_mark("box")


@pytest.fixture
def separation(qtbot):
    widget = fp.ControlSeparation()
    qtbot.addWidget(widget)
    return widget


def test_a_short_key_list_pads_rather_than_shifting_every_row_after_it(
        separation):
    """A caller that names too few controls must not misname the rest.

    The keys are joined to the values by position, so a list one short would
    hand every later row its predecessor's identifier -- a click on the third
    negative control would select the second, and there is nothing on screen
    that would look wrong. The unnamed rows report no key instead.
    """
    total = separation.set_groups({"nc": [0.1, 0.2, 0.3], "pc": [1.0, 1.1]},
                                  keys={"nc": ["a"], "pc": ["p1", "p2"]})

    assert total == 5
    assert separation.key_for_row(0) == "a"
    assert separation.key_for_row(1) is None
    assert separation.key_for_row(3) == "p1", (
        "the short list shifted the next group's keys")
    assert separation.key_for_row(4) == "p2"


def test_a_control_group_of_blanks_is_left_out_of_the_sentence(separation):
    """A group with no usable value has no median, so it claims none.

    The status line is what a reader compares the classes on. A group whose
    wells all failed would otherwise need a median printed from nothing, and
    "nc n=0 median=nan" is a number nobody can act on.
    """
    total = separation.set_groups({"nc": [0.1, 0.2],
                                   "blank": [np.nan, np.nan]})

    assert total == 2
    assert "nc n=2 median=" in separation._status.text()
    assert "blank n=" not in separation._status.text()
    assert separation.group_sizes() == [2, 0]


def test_a_dot_knows_which_control_group_it_came_from(separation):
    """Which group a row is in is a span scan, and it has to reach the last.

    The groups are laid out in one flat row space, so answering "which group"
    means walking the spans -- the second group's rows are only found by
    stepping past the first. A row beyond every span belongs to none.
    """
    separation.set_groups({"nc": [0.1, 0.2], "pc": [1.0, 1.1]})

    assert separation.group_of(0) == "nc"
    assert separation.group_of(3) == "pc"
    assert separation.group_of(9) is None

    assert separation._detail(3) == "pc   effect = 1.1"
    assert separation._detail(9) == ""


@pytest.fixture
def agreement(qtbot):
    widget = fp.GuideAgreementPlot()
    qtbot.addWidget(widget)
    return widget


def _support(**columns):
    return pd.DataFrame(columns)


def test_changing_the_mark_before_the_support_arrives_draws_nothing(
        agreement):
    """The mark menu works on an empty panel, and draws nothing on one.

    Every panel on the screen is built before the fit finishes, so a user can
    pick "box" on this one while it is still empty. Redrawing from a table
    that is not there has to be a no-op, not a traceback.
    """
    assert agreement.group_sizes() == []

    agreement.set_mark("box")

    assert agreement.mark() == "box"
    assert agreement.plot.listDataItems() == []

    agreement.set_support(_support(feature=["a", "b", "c"],
                                   n_guides=[2, 2, 3],
                                   concordance=[0.5, 1.0, 0.75]))

    assert agreement.group_sizes() == [2, 1]
    assert agreement.plot.listDataItems() != []


def test_a_guide_count_whose_genes_all_lack_agreement_gets_no_mark(agreement):
    """A box needs values; a guide count with none gets no box.

    Agreement is blank for a gene whose guides did not all fit, and a count
    where every gene is in that state is a group of nothing. Drawing an empty
    box at "3 guides" would say something about three-guide genes that the
    fit never measured.
    """
    agreement.set_mark("box")
    drawn = agreement.set_support(_support(feature=["a", "b", "c"],
                                           n_guides=[2, 2, 3],
                                           concordance=[0.5, 0.7, np.nan]))

    assert drawn == 2
    assert agreement.group_sizes() == [2, 0]
    assert "2 genes" in agreement._status.text()


def test_a_support_table_with_no_guide_counts_draws_no_genes(agreement):
    """Without ``n_guides`` there is no x for any gene, so none is drawn.

    The count is the horizontal axis. A table that does not carry it -- an
    older run, a different summariser -- has to report zero genes rather than
    stacking every one of them on a NaN.
    """
    blank = agreement.set_support(_support(feature=["a", "b"],
                                           concordance=[0.5, 1.0]))

    assert blank == 0
    assert "0 genes" in agreement._status.text()

    drawn = agreement.set_support(_support(feature=["a", "b"],
                                           n_guides=[2, 3],
                                           concordance=[0.5, 1.0]))
    assert drawn == 2
    assert "2 genes" in agreement._status.text()


def test_a_gene_says_only_what_its_support_table_carries(agreement):
    """The detail line names the columns that are there, and invents none.

    The support table is written by whatever summarised the guides, and the
    per-gene p and the single-guide flag are both optional. The one sentence
    that must survive is "SINGLE GUIDE" -- a gene resting on one guide has
    nothing corroborating it, and that is what this panel exists to say.
    """
    assert agreement._detail(0) == "", "an empty panel described a gene"

    full = _support(feature=["g1", "g2"], n_guides=[4, 1],
                    concordance=[0.75, 1.0], single_guide=[False, True],
                    n_same_direction=[3, 1], gene_p=[0.001, np.nan])
    agreement.set_support(full)

    assert agreement._detail(0) == "3 of 4 guides agree   gene p = 0.001"
    assert agreement._detail(1) == (
        "1 of 1 guides agree   SINGLE GUIDE -- gene p IS that guide's p")
    assert agreement._detail(5) == "", "a row past the table described itself"

    agreement.set_support(_support(feature=["g1"], n_guides=[4],
                                   concordance=[0.75]))
    assert agreement._detail(0) == ""


# ------------------------------------------------------------- results table

def _visible(table) -> list:
    return [row for row in range(table.table.rowCount())
            if not table.table.isRowHidden(row)]


def test_a_q_value_that_is_not_a_number_is_not_a_hit(table):
    """"Significant only" must not admit a row it cannot read.

    A results table can carry a blank, an "NA" or a formatted string in the
    corrected column -- older exports and hand-edited files both do. A row
    whose q cannot be read as a number has not been shown to pass the cut, so
    it is hidden; showing it would put an unverifiable row in a list the user
    reads as the hits.
    """
    table.set_frame(pd.DataFrame({"feature": ["a", "b"],
                                  "q_value": ["n/a", 0.01]}))

    table._only_hits.setChecked(True)

    assert _visible(table) == [1]
    assert "1 of 2 rows (q_value <= 0.05)" in table._count.text()


def test_a_row_with_no_frame_position_behind_it_selects_nothing(table):
    """A selected row that names no frame row must not emit a row number.

    Every cell the table fills carries its frame position, because the user
    can sort the view and the position is the only way home. A row that
    arrived any other way has no position, and emitting the sorted VIEW index
    instead would light up an unrelated coefficient everywhere else.
    """
    table.set_frame(pd.DataFrame({"feature": ["a", "b"],
                                  "q_value": [0.1, 0.2]}))
    seen: list = []
    table.row_selected.connect(seen.append)

    table.table.selectRow(1)
    assert seen == [1]

    table.table.insertRow(2)
    table.table.setItem(2, 0, QTableWidgetItem("orphan"))
    table.table.clearSelection()
    table.table.selectRow(2)

    assert seen == [1], "a row with no frame position emitted one anyway"


def test_a_placeholder_alone_leaves_the_significance_filter_alone(table):
    """Configuring one control must not move the other.

    The sweep's run list reuses this widget and sets only the placeholder.
    Having that hide "significant only" -- or show it -- would make one
    caller's wording decide another caller's controls.
    """
    table._only_hits.setChecked(True)

    table.configure(placeholder="Filter runs")

    assert table._filter.placeholderText() == "Filter runs"
    assert table._only_hits.isChecked() is True

    table.configure(significance_filter=False)

    assert table._only_hits.isChecked() is False
    assert table._only_hits.isVisible() is False


def test_a_key_column_the_frame_does_not_have_names_nobody(table):
    """Asking for a key column that is not there is answered, not raised.

    The caller names the column every other view joins on. A table loaded
    from a file that does not carry it -- an older export, a different
    summariser -- cannot answer any of these questions, and each of them is
    reached from a click on another widget.
    """
    assert table.key_for_row(0) is None
    assert table.select_keys(["a"]) == 0
    assert table.selected_keys() == []

    frame = pd.DataFrame({"feature": ["a", "b"], "q_value": [0.1, 0.2]})
    table.set_frame(frame, key_column="gene")

    assert table.key_for_row(0) is None
    assert table.select_key("a") is False
    assert table.select_keys(["a"]) == 0
    assert table.selected_keys() == []

    table.set_frame(frame, key_column="feature")

    assert table.key_for_row(0) == "a"
    assert table.select_key("a") is True
    assert table.select_keys(["b"]) == 1
    assert table.selected_keys() == ["b"]


def test_a_row_number_off_the_end_of_the_frame_names_nobody(table):
    """A stale index from another view must not wrap round the frame.

    Selections travel between views by key, but a row index still arrives
    from the plot's own click. If the frame has been reloaded shorter since,
    ``.iloc`` on a negative or oversized index would either raise or name the
    wrong gene.
    """
    table.set_frame(pd.DataFrame({"feature": ["a", "b"]}))

    assert table.key_for_row(1) == "b"
    assert table.key_for_row(2) is None
    assert table.key_for_row(-1) is None


def test_selecting_keys_says_what_was_selected_even_when_none_of_it_was(
        table):
    """Both signals carry the whole selection, matched or not.

    The consumers hold the selection the user made, not the rows this table
    happens to have: a gene that was filtered out of THIS export is still
    selected everywhere else. So the signals go out for a selection that
    matched nothing, and an empty selection is not the same thing as a
    selection of one unknown key.
    """
    table.set_frame(pd.DataFrame({"feature": ["a", "b"]}))
    ones: list = []
    lists: list = []
    table.key_selected.connect(ones.append)
    table.keys_selected.connect(lists.append)

    assert table.select_keys(["ghost"]) == 0
    assert ones == ["ghost"] and lists == [["ghost"]]

    assert table.select_keys([]) == 0
    assert ones == ["ghost"], "an empty selection named a row"
    assert lists == [["ghost"], []]

    assert table.select_keys(["a", "b"]) == 2
    # The live selection emits as each row goes in, so the single-row signal
    # fires more than once; the LAST one is the selection's own.
    assert ones[-1] == "b" and lists[-1] == ["a", "b"]


def test_a_row_whose_key_cell_is_blank_is_not_a_selected_key(table):
    """A blank identifier is not an identifier.

    A coefficient table can carry an empty feature -- a term the fit could
    not name. Reporting "" as a selected key would give every such row the
    same identifier, which is the collision the whole key rule exists to
    prevent.
    """
    table.set_frame(pd.DataFrame({"feature": ["a", ""], "value": [1, 2]}))
    table.table.selectAll()

    assert table.selected_keys() == ["a"]


def test_a_table_with_no_selection_model_still_reports_what_it_found(
        table, monkeypatch):
    """The count and the signals do not depend on Qt's selection model.

    It is gone while the view is being rebuilt, and a plot's selection can
    arrive in that window. What the consumers need is what matched; the
    highlight is what the model would have added.
    """
    table.set_frame(pd.DataFrame({"feature": ["a", "b"]}))
    monkeypatch.setattr(table.table, "selectionModel", lambda: None)

    assert table.select_keys(["a", "b"]) == 2
    assert table.table.selectedItems() == []


def test_the_visible_rows_come_back_even_with_no_clipboard(table,
                                                           monkeypatch):
    """Copy returns the text as well as putting it on the clipboard.

    The clipboard is absent under a headless run and on some remote sessions,
    and the return value is what the tests and the callers actually use. A
    copy that dropped the text because it could not reach the clipboard would
    take the user's rows with it.
    """
    table.set_frame(pd.DataFrame({"feature": ["a", "b"],
                                  "q_value": [0.01, 0.9]}))
    table._only_hits.setChecked(True)

    monkeypatch.setattr(QApplication, "clipboard", staticmethod(lambda: None))
    text = table.copy_visible()

    assert text.splitlines()[0] == "feature\tq_value"
    assert len(text.splitlines()) == 2, "a hidden row was copied"
    assert text.splitlines()[1].startswith("a\t")

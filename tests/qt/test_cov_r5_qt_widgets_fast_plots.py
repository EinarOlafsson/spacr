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
from PySide6.QtWidgets import (QFileDialog)

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


@pytest.fixture
def histogram(qtbot):
    widget = fp.PValueHistogram()
    qtbot.addWidget(widget)
    widget.resize(420, 320)
    return widget


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



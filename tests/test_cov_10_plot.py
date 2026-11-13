"""Figure-writing fallbacks, and the plots drawn from data that is not there.

Two kinds of branch live here. The first is what the figure writer does when
something it does not control misbehaves: a preference holding a format spaCR
cannot write, an artist whose colour cannot be read or set, a restore that
fails on the way out of an export. None of these may lose a figure, and none
may leave the on-screen plot changed after a save.

The second is what the plotters draw when the data is thinner than the
picture assumes: a merged field with no object masks, a plate table with no
well grid, a single measurement column asked for as a line, a paired
comparison whose two arms are different lengths, and a proportion table with
no replication unit in it. Each of these has a defined, visible answer, and
the alternative in every case is a figure that looks like a result.

Two parametrised tests state the contracts; a row is a helper that does the setup
and returns what it observed, next to the value the contract promises.
"""
from __future__ import annotations

import inspect
import os
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import spacr.plot as P


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Figure output preferences
# ---------------------------------------------------------------------------

def _case_a_format_spacr_cannot_write_falls_back_to_the_default(monkeypatch):
    """A stale or hand-edited preference must not make every save raise.
    spaCR writes PNG, PDF, SVG and TIFF; anything else becomes the default
    rather than an extension no writer accepts."""
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_figure_format", lambda: "bmp")
    monkeypatch.setattr(preferences, "get_figure_png_dpi", lambda: 200)
    fmt, dpi = P.figure_output_preferences()
    return (fmt == P.DEFAULT_FIGURE_FORMAT, dpi)


def _case_a_dpi_of_zero_falls_back_to_the_default(monkeypatch):
    """Zero DPI produces a zero-pixel image. The stored preference is
    replaced by the shipped default instead of writing an empty file."""
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_figure_format", lambda: "png")
    monkeypatch.setattr(preferences, "get_figure_png_dpi", lambda: 0)
    fmt, dpi = P.figure_output_preferences()
    return (fmt, dpi == P.DEFAULT_FIGURE_DPI)


def _case_an_unknown_forced_format_falls_back_to_the_preference():
    """``figure_path`` is what a non-matplotlib renderer asks before it
    exports. A typo in the format must give the preferred extension, not a
    name no renderer branches on."""
    return P.figure_path("volcano", fmt="jpeg2000").endswith(
        f".{P.figure_output_preferences()[0]}")


def _case_save_figure_writes_the_preferred_format_for_an_unknown_one(tmp_path):
    """The same fallback on the writing path: a run must not lose a figure
    over a typo in the format argument."""
    fig = plt.figure()
    fig.add_subplot().plot([0, 1], [0, 1])
    written = P.save_figure(fig, str(tmp_path / "fig"), fmt="jpeg2000",
                            announce_colours=False)
    assert os.path.isfile(written)
    return written.endswith(f".{P.figure_output_preferences()[0]}")


# ---------------------------------------------------------------------------
# Which artists are chrome, and which are data
# ---------------------------------------------------------------------------

def _case_a_figure_level_legend_is_chrome_and_is_restyled():
    """A legend attached to the figure rather than to an axes is still the
    frame around the claim. Missing it leaves a dark legend box on a white
    page after a print-mode save."""
    fig = plt.figure()
    ax = fig.add_subplot()
    line, = ax.plot([0, 1], [0, 1], label="series")
    without = len(list(P._chrome(fig)))
    fig.legend([line], ["series"])
    with_legend = list(P._chrome(fig))
    legend_texts = [artist for kind, artist, _g, _s in with_legend
                    if kind == "text" and getattr(artist, "get_text", None)
                    and artist.get_text() == "series"]
    return (len(with_legend) > without, len(legend_texts))


def _case_a_colour_that_cannot_be_read_is_skipped_not_fatal():
    """``data_colours`` reports what a reader will see. An artist whose
    colour accessor raises contributes nothing, and must not take down the
    legibility check that a save depends on."""
    fig = plt.figure()
    ax = fig.add_subplot()
    collection = ax.scatter([0, 1], [0, 1])
    bar = ax.bar([0], [1])[0]

    def _raise(*args, **kwargs):
        raise RuntimeError("this artist will not say")

    collection.get_facecolor = _raise
    collection.get_edgecolor = _raise
    bar.get_facecolor = _raise

    # A figure-shaped object whose axes list holds something that is not an
    # Axes: a caller may hand over any figure-like, and one bad entry must
    # not stop the rest being read.
    stub = types.SimpleNamespace(axes=[object(), ax])
    return P.data_colours(stub)


def _case_an_artist_that_will_not_report_its_colour_is_left_alone():
    """One unreadable artist must not abort a print-ready export; the rest
    of the figure is still restyled."""
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1])
    title = fig.suptitle("headline")
    title.set_color("#FFFFFF")

    def _raise(*args, **kwargs):
        raise RuntimeError("no colour here")

    title.get_color = _raise
    with P.print_ready(fig, mode="print", announce=False) as look:
        flipped = look.flip
        restyled = ax.title.get_color() != "#FFFFFF"
    return (flipped, restyled)


def _case_an_artist_that_refuses_a_new_colour_is_not_recorded_for_restore():
    """A setter that raises leaves that artist as it was. Recording it for
    restore anyway would try to set it a second time on the way out."""
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1])
    title = fig.suptitle("headline")
    title.set_color("#FFFFFF")
    original = title.get_color()

    def _raise(*args, **kwargs):
        raise RuntimeError("this artist will not change")

    title.set_color = _raise
    with P.print_ready(fig, mode="print", announce=False):
        pass
    return original


def _case_a_restore_that_fails_does_not_escape_the_export():
    """The context manager's whole promise is that the on-screen figure is
    the same afterwards. When one artist cannot be put back, the remaining
    artists must still be restored rather than the exception unwinding the
    loop."""
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1])
    title = fig.suptitle("headline")
    title.set_color("#FFFFFF")
    real_set = title.set_color
    calls = []

    def _flaky(value):
        calls.append(value)
        if len(calls) > 1:
            raise RuntimeError("cannot put this back")
        real_set(value)

    title.set_color = _flaky
    with P.print_ready(fig, mode="print", announce=False):
        pass
    return (len(calls),                         # applied, then restore tried
            ax.title.get_color() == "black"
            or ax.title.get_color() != "#FFFFFF")


# ---------------------------------------------------------------------------
# Plotters drawing data that is not there
# ---------------------------------------------------------------------------

def _case_a_field_with_no_object_masks_draws_an_empty_named_panel(tmp_path):
    """With no cell, nucleus or pathogen channel there is nothing to
    combine. The last panel says "no objects" rather than raising on
    ``outlines[0]`` or drawing the previous field's mask."""
    stack = np.zeros((32, 32, 3), dtype=np.uint16)
    stack[..., 0] = 100
    path = str(tmp_path / "fov.npy")
    np.save(path, stack)

    fig = P.plot_image_mask_overlay_magenta_outlines(
        path, [0, 1, 2], cell_channel=None, nucleus_channel=None,
        pathogen_channel=None, figuresize=2, thickness=1, save_pdf=False)
    combined = fig.axes[-1].images[0].get_array()
    return ([a.get_title() for a in fig.axes][-1],
            float(np.asarray(combined).max()))


def _case_a_plate_panel_that_cannot_be_drawn_says_why(capsys):
    """A table with no well grid produces no heatmap. Printing the reason is
    what stops a silently blank results folder."""
    df = pd.DataFrame({"prc": [], "recruitment": []})
    P.plot_plates(df, "recruitment", grouping="mean", min_max="allq",
                  cmap=None, verbose=True)
    return "No plate heatmap drawn" in capsys.readouterr().out


def _case_a_line_across_groups_honours_the_requested_group_order(tmp_path):
    """The order argument is how a user puts the control first. A line chart
    that ignored it would put the groups in alphabetical order while the bar
    chart of the same data obeyed it."""
    df = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 4,
                       "y": [1.0, 2, 3, 4, 10, 11, 12, 13]})
    figure, results = P.create_grouped_plot(
        df, "g", "y", graph_type="line", order=["b", "a"],
        output_dir=str(tmp_path), save=False)
    labels = [t.get_text() for t in figure.axes[0].get_xticklabels()]
    return (labels[:2], not results.empty)


def _case_a_figure_that_refuses_the_replot_recipe_is_still_returned(
        monkeypatch, tmp_path):
    """The recipe is a convenience for the right-click menu. A figure object
    that will not carry it must still be handed back with its statistics --
    losing the plot over an attribute is the worse failure."""
    real_gcf = plt.gcf

    class _Frozen:
        def __init__(self, real):
            object.__setattr__(self, "_real", real)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_real"), name)

        def __setattr__(self, name, value):
            raise AttributeError("this figure carries no recipe")

    monkeypatch.setattr(plt, "gcf", lambda: _Frozen(real_gcf()))
    df = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 4,
                       "y": [1.0, 2, 3, 4, 10, 11, 12, 13]})
    figure, results = P.create_grouped_plot(
        df, "g", "y", graph_type="bar", output_dir=str(tmp_path), save=False)
    return (isinstance(figure, _Frozen), not results.empty)


def _case_paired_arms_of_different_lengths_are_refused_by_name():
    """A paired test matches observation to observation. Two arms of
    different sizes cannot be matched, and the row says so instead of
    reporting a p-value from whatever scipy did with the mismatch."""
    df = pd.DataFrame({"g": ["a"] * 3 + ["b"] * 4,
                       "y": [1.0, 2, 3, 4, 5, 6, 7]})
    graph = P.spacrGraph(df, "g", "y", graph_type="bar", paired=True)
    graph.create_plot()
    results = graph.get_results()
    refused = results[results["Test Name"] == "not testable"]
    return (len(refused),
            "cannot be matched up" in str(refused.iloc[0]["Why This Test"]))


def _case_one_measurement_column_asked_for_as_a_line_draws_across_the_groups():
    """A line needs two axes. With a single data column there is no second
    column to put on x, so the group becomes the x axis and the point on each
    group is the same summary the bar chart would draw -- the two pictures
    then agree about the data."""
    df = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 5,
                       "y": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    graph = P.spacrGraph(df, "g", "y", graph_type="line",
                         summary_func="mean")
    graph.create_plot()
    axes = graph.get_figure().axes[0]
    return ([t.get_text() for t in axes.get_xticklabels()],
            axes.get_xlabel(), axes.get_ylabel(),
            list(graph.summary_df["centre"]))


def _case_a_line_across_groups_takes_the_log_scale_it_was_asked_for():
    """``log_y`` is set on the object, not on the axes the caller sees, so
    the across-groups line has to apply it itself. Drawing a linear axis
    while the object says log makes the same figure mean two things."""
    df = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 5,
                       "y": [1.0, 2, 3, 4, 5, 60, 70, 80, 90, 100]})
    graph = P.spacrGraph(df, "g", "y", graph_type="line", log_y=True)
    graph.create_plot()
    return graph.get_figure().axes[0].get_yscale()


# ---------------------------------------------------------------------------
# Proportion tables
# ---------------------------------------------------------------------------

def _case_a_bin_named_like_the_unit_column_does_not_collide():
    """``unstack`` turns bin VALUES into column names. When one of them is
    spelled like the unit column the reset would raise "cannot insert prc,
    already exists"; the unit column has to survive as the unit."""
    df = pd.DataFrame({
        "g": ["a", "a", "b", "b"],
        "prc": ["w1", "w1", "w2", "w2"],
        "bin": ["prc", "other", "prc", "other"],
    })
    out = P.proportions_per_unit(df, "g", "bin", "prc")
    assert "other" in out.columns
    return (sorted(out["prc"].tolist()), out["other"].tolist())


def _case_a_single_group_has_no_glm_contrast_and_says_so():
    """One group gives a design with no contrast column. The row is still
    reported, with NaN rather than a statistic, so the table has one line per
    bin whatever the design turns out to be."""
    df = pd.DataFrame({
        "g": ["a"] * 4,
        "prc": ["w1", "w1", "w2", "w2"],
        "bin": ["hit", "miss", "hit", "miss"],
    })
    out = P.proportion_mixed_model(df, "g", "bin", "prc")
    return (sorted(out["bin"].tolist()), bool(out["p_value"].isna().all()),
            out["unit"].unique().tolist())


def _case_a_table_with_no_replication_unit_says_the_p_value_is_too_small(
        capsys):
    """Without a well column the only test possible is over objects, which
    are not independent. The caller has to be told that the number is
    smaller than the experiment supports."""
    df = pd.DataFrame({
        "g": ["a", "a", "b", "b", "a", "b"],
        "bin": ["hit", "miss", "hit", "miss", "hit", "miss"],
    })
    results, _pairwise, fig = P.plot_proportion_stacked_bars(
        {"verbose": False}, df, "g", "bin", prc_column="prc", level="object")
    printed = capsys.readouterr().out
    assert fig is not None
    return ("no 'prc' column" in printed,
            "smaller than the experiment" in printed,
            results["unit"].tolist())


def _call(request, helper):
    """Run a case helper, giving it the fixtures named in its signature."""
    names = inspect.signature(helper).parameters
    return helper(**{n: request.getfixturevalue(n) for n in names})


@pytest.mark.parametrize("helper, expected", [
    pytest.param(_case_a_format_spacr_cannot_write_falls_back_to_the_default,
                 (True, 200), id="unwritable-format-falls-back-to-default"),
    pytest.param(_case_a_dpi_of_zero_falls_back_to_the_default,
                 ("png", True), id="zero-dpi-falls-back-to-default"),
    pytest.param(_case_an_unknown_forced_format_falls_back_to_the_preference,
                 True, id="figure-path-unknown-format-uses-preference"),
    pytest.param(
        _case_save_figure_writes_the_preferred_format_for_an_unknown_one,
        True, id="save-figure-unknown-format-uses-preference"),
    pytest.param(_case_a_figure_level_legend_is_chrome_and_is_restyled,
                 (True, 1), id="figure-level-legend-is-chrome"),
    pytest.param(_case_a_colour_that_cannot_be_read_is_skipped_not_fatal,
                 [], id="unreadable-colour-is-skipped"),
    pytest.param(_case_an_artist_that_will_not_report_its_colour_is_left_alone,
                 (True, True), id="artist-without-colour-left-alone"),
    pytest.param(
        _case_an_artist_that_refuses_a_new_colour_is_not_recorded_for_restore,
        "#FFFFFF", id="artist-refusing-new-colour-not-recorded"),
    pytest.param(_case_a_restore_that_fails_does_not_escape_the_export,
                 (2, True), id="failed-restore-does-not-escape"),
])
def test_figure_writing_and_plotting_fall_back_to_a_defined_answer(
        request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    pytest.param(_case_a_field_with_no_object_masks_draws_an_empty_named_panel,
                 ("no objects", 0.0), id="no-masks-draws-no-objects-panel"),
    pytest.param(_case_a_plate_panel_that_cannot_be_drawn_says_why,
                 True, id="plate-without-grid-says-why"),
    pytest.param(_case_a_line_across_groups_honours_the_requested_group_order,
                 (["b", "a"], True), id="line-honours-group-order"),
    pytest.param(_case_a_figure_that_refuses_the_replot_recipe_is_still_returned,
                 (True, True), id="figure-refusing-recipe-still-returned"),
    pytest.param(_case_paired_arms_of_different_lengths_are_refused_by_name,
                 (1, True), id="unequal-paired-arms-refused-by-name"),
    pytest.param(
        _case_one_measurement_column_asked_for_as_a_line_draws_across_the_groups,
        (["a", "b"], "g", "y", [3.0, 8.0]),
        id="single-column-line-draws-across-groups"),
    pytest.param(_case_a_line_across_groups_takes_the_log_scale_it_was_asked_for,
                 "log", id="line-across-groups-takes-log-scale"),
    pytest.param(_case_a_bin_named_like_the_unit_column_does_not_collide,
                 (["w1", "w2"], [0.5, 0.5]),
                 id="bin-named-like-unit-column-does-not-collide"),
    pytest.param(_case_a_single_group_has_no_glm_contrast_and_says_so,
                 (["hit", "miss"], True, ["prc"]),
                 id="single-group-glm-reports-nan-per-bin"),
    pytest.param(
        _case_a_table_with_no_replication_unit_says_the_p_value_is_too_small,
        (True, True, ["object"]), id="no-replication-unit-warns-p-too-small"),
])
def test_plotters_give_a_defined_answer_when_the_data_is_thin(
        request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected

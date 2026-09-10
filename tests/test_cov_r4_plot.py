"""The branches of ``spacr.plot`` that only fire on an unusual input.

Everything here is CPU-only, offline and deterministic. Each test drives one
path that the rest of the suite never reaches, and each one asserts on what
the code actually produced -- a written file, a printed line, the arrays
handed to ``imshow``, the artists on an axis -- rather than on the absence of
an exception.

Where the point of a test is that something is NOT done (no directory made,
no legend removed, no column split, no bracket drawn), the same test also
drives the input that makes it happen, because an assertion about an absence
that no present case is measured against is an assertion about nothing.

What is pinned, and why each one matters:

* ``save_figure`` into the working directory -- a bare filename has no parent
  to create, and calling ``os.makedirs('')`` would raise on every such save.
* the overlay renderer on a float stack -- the uint->float cast is a
  normalisation, not a requirement, and a caller holding already-normalised
  data must get the same picture.
* ``_filter_objects_in_plot`` with only a nucleus plane -- the three summary
  lines are printed per object type, and a run with a disabled type must not
  report counts it never computed.
* ``_plot_recruitment`` where the strain names repeat the condition names --
  seaborn draws no legend for a redundant hue, and the panel loop must not
  try to remove one that was never made.
* ``spacrGraph``'s drawing-time guards -- the y-limit forms, the missing
  legend, and the four ways a comparison bracket is refused. A bracket over
  the wrong pair, or one drawn for a comparison that was never made, reads
  exactly like a real result.
* ``plot_data_from_db`` / ``plot_data_from_csv`` / ``plot_region`` -- the
  per-source database list, the frame that already names its wells, the
  ``keep_groups`` value that is neither a string nor a list, and the overlay
  figure that came back empty.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

pytest.importorskip("seaborn")
pytest.importorskip("statsmodels")


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate across tests."""
    plt.close("all")
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------

def test_a_figure_saved_by_bare_name_lands_in_the_working_directory(
        tmp_path, monkeypatch):
    """``os.path.dirname('fig.png')`` is ``''``, and ``os.makedirs('')``
    raises ``FileNotFoundError``. Every save that names no directory -- which
    is what a notebook user types -- would die on the mkdir if the guard were
    not there. The nested path in the same test is what shows the guard is
    skipping an empty directory rather than skipping the mkdir entirely."""
    from spacr.plot import save_figure

    monkeypatch.chdir(tmp_path)
    fig = plt.figure()
    fig.add_subplot().plot([0.0, 1.0], [0.0, 1.0])

    bare = save_figure(fig, "bare_name.png", fmt="png")

    assert os.path.dirname(bare) == ""
    assert (tmp_path / "bare_name.png").is_file()
    assert (tmp_path / "bare_name.png").stat().st_size > 0

    nested = save_figure(fig, os.path.join("made", "up", "deep.png"),
                         fmt="png")

    assert os.path.dirname(nested) == os.path.join("made", "up")
    assert (tmp_path / "made" / "up" / "deep.png").is_file()


# ---------------------------------------------------------------------------
# plot_image_mask_overlay_magenta_outlines: the dtype normalisation
# ---------------------------------------------------------------------------

_SHAPE = (32, 32)


def _disc(centres):
    """Label image with one disc per ``(cy, cx, r)``, labelled from 1."""
    yy, xx = np.mgrid[:_SHAPE[0], :_SHAPE[1]]
    out = np.zeros(_SHAPE, dtype=np.int32)
    for index, (cy, cx, radius) in enumerate(centres, start=1):
        out[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius] = index
    return out


def _write_stack(path, intensity, masks, dtype):
    planes = [np.asarray(intensity, dtype=np.float64)]
    planes += [np.asarray(mask, dtype=np.float64) for mask in masks]
    stack = np.stack(planes, axis=-1).astype(dtype)
    np.save(path, stack)
    return stack


def _painted(fig, index, colour=(1.0, 0.0, 1.0)):
    """Which pixels of one panel carry ``colour`` exactly."""
    panel = np.asarray(fig.axes[index].images[0].get_array())
    return np.all(np.isclose(panel, colour, atol=1e-6), axis=-1)


def test_an_already_normalised_float_stack_is_overlaid_like_a_uint_one(
        tmp_path):
    """The uint16/uint8 cast is a normalisation, not an entry requirement.

    A caller who has already scaled their field of view to ``[0, 1]`` -- which
    a uint16 array cannot represent at all -- has a stack whose dtype the cast
    branch does not recognise, and it has to reach the same picture as the raw
    camera data does. The uint16 stack of the same geometry is drawn in the
    same test, because "the float one rendered" only means something beside a
    picture known to be right."""
    from spacr.plot import plot_image_mask_overlay_magenta_outlines

    cells = _disc([(10, 10, 6), (24, 24, 5)])
    raw = np.linspace(200.0, 3800.0, _SHAPE[0] * _SHAPE[1]).reshape(_SHAPE)

    float_dir = tmp_path / "asfloat" / "stack"
    uint_dir = tmp_path / "asuint" / "stack"
    float_dir.mkdir(parents=True)
    uint_dir.mkdir(parents=True)
    # The float stack carries intensities in [0, 1]; a uint16 array would
    # round every one of them to 0.
    float_stack = _write_stack(float_dir / "fov.npy", raw / 4095.0, [cells],
                               np.float32)
    _write_stack(uint_dir / "fov.npy", raw, [cells], np.uint16)

    assert float_stack.dtype == np.float32
    assert float_stack[..., 0].max() < 1.0

    as_float = plot_image_mask_overlay_magenta_outlines(
        str(float_dir / "fov.npy"), [0], cell_channel=0, nucleus_channel=None,
        pathogen_channel=None, figuresize=2, thickness=1, save_pdf=False)
    as_uint = plot_image_mask_overlay_magenta_outlines(
        str(uint_dir / "fov.npy"), [0], cell_channel=0, nucleus_channel=None,
        pathogen_channel=None, figuresize=2, thickness=1, save_pdf=False)

    float_outline = _painted(as_float, 0)
    assert float_outline.any(), "the float stack drew no outline at all"
    assert np.array_equal(float_outline, _painted(as_uint, 0))
    # The outline follows the mask plane read off the tail of the stack.
    assert float_outline[cells > 0].any()
    # The panel underneath is a real normalised image, not a flat frame.
    base = np.asarray(as_float.axes[0].images[0].get_array())
    assert base.min() == pytest.approx(0.0, abs=1e-6)
    assert base.max() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# plot_masks
# ---------------------------------------------------------------------------

def _label_mask(size=16, n=3):
    mask = np.zeros((size, size), dtype=np.int32)
    band = size // (n + 1)
    for label in range(1, n + 1):
        mask[label * band:label * band + 2, 2:size - 2] = label
    return mask


def test_a_mask_panel_can_be_drawn_without_its_object_numbers():
    """The numbers are an annotation, and on a crowded field they are ink
    over the objects they name. Turning them off must leave the mask panel
    itself untouched -- same image, same title -- so both are asserted
    against a run with them on."""
    from spacr.plot import plot_masks

    batch = np.zeros((1, 16, 16, 1), dtype=np.float32)
    batch[0, ..., 0] = np.linspace(0.0, 1.0, 256).reshape(16, 16)
    mask = _label_mask(16, n=3)
    flow = np.zeros((16, 16, 3), dtype=np.float32)

    plot_masks(batch, [mask], [flow], figuresize=2, nr=1,
               print_object_number=True)
    numbered = plt.figure(plt.get_fignums()[-1])
    assert sorted(t.get_text() for t in numbered.axes[1].texts) == \
        ["1", "2", "3"]

    plt.close("all")
    plot_masks(batch, [mask], [flow], figuresize=2, nr=1,
               print_object_number=False)
    plain = plt.figure(plt.get_fignums()[-1])

    assert list(plain.axes[1].texts) == []
    assert [ax.get_title() for ax in plain.axes] == \
        ["Image - Channel0", "Mask", "Flow"]
    np.testing.assert_array_equal(
        np.asarray(plain.axes[1].images[0].get_array()), mask)


# ---------------------------------------------------------------------------
# _filter_objects_in_plot
# ---------------------------------------------------------------------------

def _nucleus_only_stack():
    """Two intensity planes and one nucleus plane holding two objects.

    Object 1 is 4x4 = 16 px, object 2 is 2x2 = 4 px, so an area floor
    between the two keeps exactly one of them.
    """
    stack = np.zeros((24, 24, 3), dtype=np.int32)
    stack[..., 0] = 100
    nucleus = np.zeros((24, 24), dtype=np.int32)
    nucleus[2:6, 2:6] = 1
    nucleus[16:18, 16:18] = 2
    stack[..., 1] = nucleus
    return stack


def test_a_run_with_only_a_nucleus_mask_reports_only_the_nucleus(capsys):
    """The three summary lines are printed per object type, and each one
    reads its own counters. With the cell and pathogen planes disabled those
    counters were never assigned, so printing their line would raise
    ``UnboundLocalError`` and take the whole browse down -- and a line
    reporting objects the run never looked at would be worse than the crash.
    The filtering itself is asserted so the counts are known to be real."""
    from spacr.plot import _filter_objects_in_plot

    stack = _nucleus_only_stack()
    assert sorted(np.unique(stack[..., 1]).tolist()) == [0, 1, 2]

    out = _filter_objects_in_plot(
        stack.copy(), cell_mask_dim=None, nucleus_mask_dim=1,
        pathogen_mask_dim=None, mask_dims=[1],
        # ROLE order: [cell, nucleus, pathogen]. The nucleus range is the
        # second entry, and an area floor of 10 keeps only the 16-px object.
        filter_min_max=[[0, 10 ** 8], [10, 10 ** 8], [0, 10 ** 8]],
        nuclei_limit=True, pathogen_limit=True)

    assert sorted(np.unique(out[..., 1]).tolist()) == [0, 1]
    assert np.array_equal(out[..., 0], stack[..., 0])

    printed = capsys.readouterr().out
    assert "removed 1 nucleus, nucleus size from" in printed
    assert "cells, cell size" not in printed
    assert "pathogens, pathogen size" not in printed


# ---------------------------------------------------------------------------
# _plot_cropped_arrays
# ---------------------------------------------------------------------------

def test_a_one_dimensional_crop_matches_no_panel_layout():
    """``_plot_cropped_arrays`` lays out either one panel (2D) or one per
    channel (3D). A 1D array matches neither, so the function falls through
    to ``return fig`` with ``fig`` never assigned -- which is what its
    docstring says it does. Pinning it here means a later refactor that
    turns this into a silent empty figure has to change the docstring too.
    The 2D array in the same test shows the function is otherwise working."""
    from spacr.plot import _plot_cropped_arrays

    plane = np.arange(64, dtype=np.float32).reshape(8, 8)
    fig = _plot_cropped_arrays(plane, "plane.npy", figuresize=2, threshold=0)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 1

    with pytest.raises(UnboundLocalError):
        _plot_cropped_arrays(np.arange(8, dtype=np.float32), "line.npy",
                             figuresize=2)


# ---------------------------------------------------------------------------
# _plot_recruitment
# ---------------------------------------------------------------------------

_RECRUITMENT_EXTRA = [
    "pathogen_cytoplasm_mean_mean",
    "pathogen_cytoplasm_q75_mean",
    "pathogen_periphery_cytoplasm_mean_mean",
    "pathogen_outside_cytoplasm_mean_mean",
    "pathogen_outside_cytoplasm_q75_mean",
]


def _recruitment_df(strains, channel=1, n=24, seed=0):
    """A recruitment frame whose ``pathogen`` column holds ``strains``."""
    rng = np.random.default_rng(seed)
    data = {
        "condition": ["ctrl", "trt"] * (n // 2),
        "pathogen": list(strains) * (n // 2),
    }
    for comp in ("cell", "nucleus", "cytoplasm", "pathogen"):
        data[f"{comp}_channel_{channel}_mean_intensity"] = rng.uniform(10, 100, n)
    for col in _RECRUITMENT_EXTRA:
        data[col] = rng.uniform(2, 50, n)
    return pd.DataFrame(data)


@pytest.fixture
def _restore_rcparams():
    saved = matplotlib.rcParams.copy()
    yield
    matplotlib.rcParams.update(saved)


def test_a_strain_column_that_repeats_the_condition_gets_no_legend_to_remove(
        _restore_rcparams):
    """seaborn draws no legend when the hue only restates the x axis, and the
    recruitment grid then has nothing to remove. Calling ``ax.legend_.remove()``
    unguarded on that axis is an ``AttributeError`` on ``None`` -- the whole
    figure lost for a screen where the strain happens to be the condition.

    The frame with two real strain names is drawn in the same test: there
    seaborn DOES build a legend, which is what makes the guard a guard."""
    from spacr.plot import _plot_recruitment

    _plot_recruitment(_recruitment_df(("wt", "mut")), "test", 1, figuresize=4)
    distinct_intensity, distinct_grid = [plt.figure(n)
                                         for n in plt.get_fignums()]
    assert {t.get_text()
            for t in distinct_intensity.axes[3].get_legend().get_texts()} == \
        {"wt", "mut"}
    assert all(ax.get_legend() is None for ax in distinct_grid.axes[:5])

    plt.close("all")
    # The strain names now repeat the condition names exactly.
    _plot_recruitment(_recruitment_df(("ctrl", "trt")), "test", 1,
                      figuresize=4)
    same_intensity, same_grid = [plt.figure(n) for n in plt.get_fignums()]

    assert same_intensity.axes[3].get_legend().get_texts() == []
    assert len(same_grid.axes) == 6
    assert [ax.get_ylabel() for ax in same_grid.axes[:5]] == _RECRUITMENT_EXTRA
    assert all(ax.get_legend() is None for ax in same_grid.axes[:5])
    assert all(ax.patches for ax in same_grid.axes[:5]), \
        "the panels must still be drawn, not merely left legend-free"


# ---------------------------------------------------------------------------
# _finite_p_value
# ---------------------------------------------------------------------------

def test_a_p_value_that_is_not_a_number_has_no_answer():
    """A skipped test records ``None`` and a refused one records ``nan``, and
    both mean the comparison was never made. Coercing either to a float and
    comparing it against 0.05 is how "no test" becomes "not significant" on a
    figure."""
    from spacr.plot import _finite_p_value

    assert _finite_p_value(0.01) == pytest.approx(0.01)
    assert _finite_p_value("0.02") == pytest.approx(0.02)
    assert _finite_p_value(None) is None            # TypeError
    assert _finite_p_value("not testable") is None  # ValueError
    assert _finite_p_value(float("nan")) is None
    assert _finite_p_value(float("inf")) is None


# ---------------------------------------------------------------------------
# spacrGraph
# ---------------------------------------------------------------------------

def _group(loc, n=10):
    """A normal group with no sampling noise, so every verdict is fixed."""
    from scipy.stats import norm

    return norm.ppf(np.linspace(0.05, 0.95, n), loc=loc, scale=1.0)


def _frame(groups, n=10):
    rows = []
    for name, loc in groups:
        for value in _group(loc, n):
            rows.append({"grp": name, "val": float(value),
                         "val2": float(value) * 2.0})
    return pd.DataFrame(rows)


def _brackets(ax):
    """The four-point bracket polylines drawn on ``ax``, as x-pair tuples."""
    out = []
    for line in ax.lines:
        xs = list(line.get_xdata())
        if len(xs) == 4 and xs[0] == xs[1] and xs[2] == xs[3]:
            out.append((float(xs[0]), float(xs[2])))
    return out


def test_a_y_limit_of_one_number_is_a_floor_and_anything_else_is_ignored():
    """``y_lim`` is documented as two numbers and accepted as one (a floor).
    A list of any other length is neither, and pinning the axis from it would
    put the plot in a window the caller never asked for. All three forms are
    driven here so the ignored case is measured against the ones that act."""
    from spacr.plot import spacrGraph

    frame = _frame([("a", 0.0), ("b", 3.0)])

    both = spacrGraph(frame.copy(), "grp", "val", graph_type="bar",
                      y_lim=[-5.0, 12.0])
    both.create_plot()
    assert both.fig.axes[0].get_ylim() == (-5.0, 12.0)

    floor = spacrGraph(frame.copy(), "grp", "val", graph_type="bar",
                       y_lim=[-5.0])
    floor.create_plot()
    assert floor.fig.axes[0].get_ylim()[0] == -5.0

    three = spacrGraph(frame.copy(), "grp", "val", graph_type="bar",
                       y_lim=[-5.0, 12.0, 99.0])
    three.create_plot()
    limits = three.fig.axes[0].get_ylim()
    assert limits != (-5.0, 12.0)
    assert limits[0] != -5.0
    # The plot is still a plot: only the window was left alone.
    assert three.fig.axes[0].patches


def test_an_order_naming_no_group_in_the_frame_leaves_no_legend_to_remove():
    """``order`` selects which groups are drawn. When it names none of them
    seaborn draws nothing and attaches no legend, and the tidy-up that strips
    the redundant legend then has nothing to strip -- ``legend.remove()`` on
    ``None`` is an ``AttributeError`` that loses the figure. The same frame
    with a real order is plotted here so the missing legend is known to be a
    property of the order, not of the code never making one."""
    from spacr.plot import spacrGraph

    frame = _frame([("a", 0.0), ("b", 3.0)])

    real = spacrGraph(frame.copy(), "grp", "val", graph_type="bar",
                      order=["a", "b"])
    real.create_plot()
    real_ax = real.fig.axes[0]
    assert real_ax.patches
    assert [text.get_text() for text in real_ax.get_xticklabels()] == \
        ["a", "b"]

    missing = spacrGraph(frame.copy(), "grp", "val", graph_type="bar",
                         order=["not_a_group"])
    missing.create_plot()
    missing_ax = missing.fig.axes[0]

    assert missing_ax.get_legend() is None
    assert list(missing_ax.patches) == []
    assert [text.get_text() for text in missing_ax.get_xticklabels()] == \
        ["not_a_group"]
    # The statistics still ran on the frame that was handed in.
    assert not missing.results_df.empty


def test_removing_outliers_says_nothing_when_there_are_none():
    """The trim is announced because the reader has to know the plot and the
    p-values used different points. When nothing was trimmed there is no such
    gap, and printing "0 of 20 points are hidden" is a warning about a
    difference that does not exist. A frame with a real outlier is trimmed in
    the same test, so the silence is known to be conditional."""
    from spacr.plot import spacrGraph

    clean = _frame([("a", 0.0), ("b", 3.0)])
    graph = spacrGraph(clean.copy(), "grp", "val", graph_type="bar",
                       remove_outliers=True)
    graph.create_plot()

    assert len(graph.df) == len(clean)
    assert bool(graph.results_df["outliers_removed_from_plot_only"].all())

    spiked = clean.copy()
    spiked.loc[0, "val"] = 5000.0
    trimmed = spacrGraph(spiked, "grp", "val", graph_type="bar",
                         remove_outliers=True)
    trimmed.create_plot()

    assert len(trimmed.df) == len(spiked) - 1
    assert trimmed.df["val"].max() < 5000.0


def test_a_wide_panel_stops_growing_with_the_number_of_groups():
    """The figure side is ``max(6, 2n)/4`` with a floor of 10 inches, so it
    is pinned at 10 up to nineteen groups and grows from twenty on. Without
    the floor a two-group plot would be a 3-inch square with unreadable
    labels; without the growth a fifty-group plot would be 10 inches of
    overlapping ticks. Both sides of the hinge are measured here."""
    from spacr.plot import spacrGraph

    few = spacrGraph(_frame([("a", 0.0), ("b", 3.0)], n=6), "grp", "val",
                     graph_type="bar")
    few.create_plot()
    assert tuple(few.fig.axes[0].figure.get_size_inches()) == (10.0, 10.0)

    many_names = [(f"g{i:02d}", float(i)) for i in range(24)]
    many = spacrGraph(_frame(many_names, n=6), "grp", "val", graph_type="bar")
    many.create_plot()
    side = many.fig.axes[0].figure.get_size_inches()[0]

    assert side == pytest.approx(24 * 2 / 4)
    assert side > 10.0
    assert many.fig.axes[0].get_xlim() == (-0.5, 23.5)


def test_a_results_table_with_no_pairwise_row_names_no_pair():
    """``results_df`` mixes normality rows, comparison rows and post-hoc
    rows, and only the last two name two groups. A table that holds neither
    -- a graph built but never plotted, or a table with no ``Comparison``
    column at all -- has no pair to bracket, and reading one out of a
    normality row is how the annotation pass used to die on its first row."""
    from spacr.plot import spacrGraph

    graph = spacrGraph(_frame([("a", 0.0), ("b", 3.0)]), "grp", "val",
                       graph_type="bar")

    # Nothing has been plotted yet, so the table is empty.
    assert graph.results_df.empty
    assert graph._comparison_pairs() == []

    # A table with rows but no 'Comparison' column is equally unreadable.
    graph.results_df = pd.DataFrame([{"Test Name": "Shapiro-Wilk",
                                      "p-value": 0.4}])
    assert graph._comparison_pairs() == []

    # And once a real comparison is there, pairs come back out.
    graph.create_plot()
    pairs = graph._comparison_pairs()
    assert pairs
    assert all(len(pair) == 3 for pair in pairs)
    assert ("a", "b") in [(first, second) for first, second, _p in pairs]


def test_a_table_of_normality_rows_alone_draws_no_bracket(capsys):
    """Every row of ``results_df`` can be a per-group normality row -- one
    data column, two groups, and a comparison the engine refused to name.
    There is no pair in that table, and the annotation pass has to say so and
    draw nothing rather than bracket the first two labels it can find."""
    from spacr.plot import spacrGraph

    graph = spacrGraph(_frame([("a", 0.0), ("b", 3.0)]), "grp", "val",
                       graph_type="bar", annotate_stats=True)
    graph.create_plot()
    ax = graph.fig.axes[0]
    assert _brackets(ax), "the run itself must draw brackets"

    for line in list(ax.lines):
        line.remove()
    graph.results_df = pd.DataFrame([
        {"Comparison": "Normality test for a on val", "p-value": 0.4},
        {"Comparison": "Normality test for b on val", "p-value": 0.6}])

    assert graph._draw_comparison_lines(ax) == 0
    assert _brackets(ax) == []
    assert "No comparisons available to annotate." in capsys.readouterr().out


def test_several_measurements_on_one_axis_are_not_bracketed():
    """With more than one data column the x axis is one position per
    (column, group) pair and the symbol table underneath says which is which.
    A bracket has no unambiguous pair of ends to sit on there, so the pass
    refuses -- while the single-column plot of the same data draws them."""
    from spacr.plot import spacrGraph

    frame = _frame([("a", 0.0), ("b", 3.0)])

    one = spacrGraph(frame.copy(), "grp", "val", graph_type="bar",
                     annotate_stats=True)
    one.create_plot()
    assert len(_brackets(one.fig.axes[0])) >= 1

    two = spacrGraph(frame.copy(), "grp", ["val", "val2"], graph_type="bar",
                     annotate_stats=True)
    two.create_plot()
    ax = two.fig.axes[0]

    assert two._comparison_pairs(), "there are pairs; they are just not drawn"
    assert two._draw_comparison_lines(ax) == 0
    assert _brackets(ax) == []


def test_a_bracket_stack_on_an_axis_with_no_data_starts_at_the_view_top():
    """The stack is placed above the data, read off ``ax.dataLim``. An axis
    that has had nothing added to it reports ``-inf`` there, and ``min(-inf,
    top)`` would put every bracket at negative infinity -- off the figure,
    which is indistinguishable from a comparison nobody made. The fallback is
    the top of the view."""
    from spacr.plot import spacrGraph

    graph = spacrGraph(_frame([("a", 0.0), ("b", 3.0)]), "grp", "val",
                       graph_type="bar")
    graph.create_plot()
    plt.close(graph.fig)

    fig, ax = plt.subplots()
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["a", "b"])
    assert not np.isfinite(ax.dataLim.y1), "the axis must carry no data yet"
    graph.results_df = pd.DataFrame([{"Comparison": "a vs b",
                                      "p-value": 0.001}])

    assert graph._draw_comparison_lines(ax) == 1

    drawn = _brackets(ax)
    assert drawn == [(0.0, 1.0)]
    ys = [y for line in ax.lines for y in line.get_ydata()]
    assert ys and all(np.isfinite(y) for y in ys)
    assert min(ys) > 0.0, "the stack sits above the view's bottom, not at -inf"


# ---------------------------------------------------------------------------
# plot_data_from_db
# ---------------------------------------------------------------------------

ROWS = ["r1", "r2"]
COLUMNS = [f"c{i}" for i in range(1, 11)]
N_FIELDS = 3


def _well_grid():
    return [("plate1", row, col, f"f{f + 1}")
            for row in ROWS for col in COLUMNS for f in range(N_FIELDS)]


def _saliency_frame(rng):
    grid = _well_grid()
    n = len(grid)
    offset = np.array([0.0 if r == "r1" else 1.5 for _, r, _, _ in grid])
    return pd.DataFrame({
        "plateID": [g[0] for g in grid],
        "rowID": [g[1] for g in grid],
        "columnID": [g[2] for g in grid],
        "fieldID": [g[3] for g in grid],
        "object_label": np.arange(1, n + 1),
        "saliency_correlation": rng.normal(0.0, 0.25, n) + offset,
    })


def _make_saliency_db(dirpath, rng, name="measurements.db"):
    meas = os.path.join(str(dirpath), "measurements")
    os.makedirs(meas, exist_ok=True)
    db = os.path.join(meas, name)
    con = sqlite3.connect(db)
    try:
        _saliency_frame(rng).to_sql("saliency_image_correlations", con,
                                    index=False)
    finally:
        con.close()
    return db


def _db_settings(src, **over):
    settings = {
        "src": src,
        "database": "measurements.db",
        "table_names": "saliency_image_correlations",
        "data_column": "saliency_correlation",
        "grouping_column": "rowID",
        "graph_name": "Fig",
        "graph_type": "jitter",
        "cell_types": ["HeLa", "U2OS"],
        "cell_plate_metadata": [["r1"], ["r2"]],
        "representation": "well",
        "theme": "deep",
        "save": False,
        "verbose": False,
    }
    settings.update(over)
    return settings


def test_each_source_can_name_its_own_database_file(tmp_path, rng):
    """``database`` is broadcast to one name per source only when it is a
    single string. A caller who already listed one file per source is naming
    files that differ, and overwriting that list with the first name would
    read the same plate twice -- or, as here, look for a file that is not
    there. Both forms are run so the untouched list is measured against the
    broadcast one."""
    from spacr.plot import plot_data_from_db

    shared_a = tmp_path / "shared_a"
    shared_b = tmp_path / "shared_b"
    shared_a.mkdir()
    shared_b.mkdir()
    _make_saliency_db(shared_a, rng)
    _make_saliency_db(shared_b, rng)

    broadcast = _db_settings([str(shared_a), str(shared_b)])
    fig, results_df, df = plot_data_from_db(broadcast)
    assert broadcast["database"] == ["measurements.db", "measurements.db"]
    assert len(df) == 2 * len(ROWS) * len(COLUMNS) * N_FIELDS

    named_a = tmp_path / "named_a"
    named_b = tmp_path / "named_b"
    named_a.mkdir()
    named_b.mkdir()
    _make_saliency_db(named_a, rng, name="first.db")
    # The second source's file has a DIFFERENT name, so a broadcast list
    # would look for 'first.db' in named_b and find nothing.
    _make_saliency_db(named_b, rng, name="second.db")

    listed = _db_settings([str(named_a), str(named_b)],
                          database=["first.db", "second.db"],
                          graph_name="Listed")
    fig2, results2, df2 = plot_data_from_db(listed)

    assert listed["database"] == ["first.db", "second.db"]
    assert len(df2) == 2 * len(ROWS) * len(COLUMNS) * N_FIELDS
    assert isinstance(fig2, matplotlib.figure.Figure)
    assert not results2.empty
    assert isinstance(fig, matplotlib.figure.Figure)
    assert not results_df.empty


# ---------------------------------------------------------------------------
# plot_data_from_csv
# ---------------------------------------------------------------------------

def _csv_settings(src, **over):
    settings = {
        "src": src,
        "data_column": "value",
        "grouping_column": "group",
        "graph_name": "CsvFig",
        "graph_type": "box",
        "save": True,
        "y_lim": None,
        "log_y": False,
        "log_x": False,
        "representation": "object",
        "theme": "deep",
        "remove_outliers": False,
        "verbose": False,
    }
    settings.update(over)
    return settings


def _csv_stem(settings):
    return (f"{settings['graph_name']}_{settings['data_column']}"
            f"_{settings['grouping_column']}_{settings['graph_type']}")


def _prc_frame(rng, with_well_columns, groups=("ctrl", "treat"), n_wells=8):
    rows = []
    for group_index, group in enumerate(groups):
        for well in range(n_wells):
            prc = f"realplate_r{group_index + 1}_c{well + 1}"
            row = {"prc": prc, "group": group,
                   "value": float(rng.normal(10.0 + 4.0 * group_index, 1.0))}
            if with_well_columns:
                row.update({"plate": "named_by_hand", "rowID": "handrow",
                            "columnID": "handcol"})
            rows.append(row)
    return pd.DataFrame(rows)


def test_a_frame_that_already_names_its_wells_keeps_the_names_it_came_with(
        tmp_path, rng):
    """``prc`` is split into plateID/rowID/columnID only when the frame does
    not already carry those columns. Splitting regardless would overwrite the
    caller's own well identities with whatever the composite key happens to
    hold, and every grouping downstream would then be computed on the wrong
    keys -- silently, because both spellings look like well ids.

    The frame WITHOUT those columns is run in the same test, because "the
    names survived" only means something beside a run where they are
    replaced."""
    from spacr.plot import plot_data_from_csv

    already = tmp_path / "already_named.csv"
    _prc_frame(rng, with_well_columns=True).to_csv(already, index=False)
    settings = _csv_settings(str(already), graph_name="Kept")
    plot_data_from_csv(settings)

    kept = pd.read_csv(tmp_path / "results" / "Kept" /
                       f"{_csv_stem(settings)}_data.csv")
    assert set(kept["rowID"]) == {"handrow"}
    assert set(kept["columnID"]) == {"handcol"}
    # plateID was synthesised before the split was skipped, so it is the
    # placeholder rather than the plate named inside prc.
    assert set(kept["plateID"]) == {"plate1"}

    bare = tmp_path / "bare.csv"
    _prc_frame(rng, with_well_columns=False).to_csv(bare, index=False)
    settings_bare = _csv_settings(str(bare), graph_name="Split")
    plot_data_from_csv(settings_bare)

    split = pd.read_csv(tmp_path / "results" / "Split" /
                        f"{_csv_stem(settings_bare)}_data.csv")
    assert set(split["plateID"]) == {"realplate"}
    assert set(split["rowID"]) == {"r1", "r2"}


def test_a_keep_groups_that_is_neither_a_name_nor_a_list_filters_nothing(
        tmp_path, rng):
    """``keep_groups`` is a name or a list of names. A tuple, a set or a
    number is neither, and filtering on it would either raise or -- worse --
    quietly keep a subset nobody asked for. It is left alone, and the list
    form in the same test shows the filter does work when it is given
    something it understands."""
    from spacr.plot import plot_data_from_csv

    csv = tmp_path / "three_groups.csv"
    _prc_frame(rng, with_well_columns=False,
               groups=("ctrl", "treat", "drop_me")).to_csv(csv, index=False)

    listed = _csv_settings(str(csv), keep_groups=["ctrl", "treat"],
                           graph_name="Listed")
    plot_data_from_csv(listed)
    filtered = pd.read_csv(tmp_path / "results" / "Listed" /
                           f"{_csv_stem(listed)}_data.csv")
    assert set(filtered["group"]) == {"ctrl", "treat"}

    odd = _csv_settings(str(csv), keep_groups=("ctrl", "treat"),
                        graph_name="Tuple")
    plot_data_from_csv(odd)

    assert odd["keep_groups"] == ("ctrl", "treat"), "the value is untouched"
    unfiltered = pd.read_csv(tmp_path / "results" / "Tuple" /
                             f"{_csv_stem(odd)}_data.csv")
    assert set(unfiltered["group"]) == {"ctrl", "treat", "drop_me"}


# ---------------------------------------------------------------------------
# plot_region
# ---------------------------------------------------------------------------

FOV_NAME = "plate1_r1_c1_f1"


def _write_png(path, rng, size=16):
    from PIL import Image

    arr = rng.integers(0, 255, size=(size, size, 3)).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
    return str(path)


def _make_region_src(tmp_path, rng, n_crops=3):
    src = tmp_path / "region_src"
    (src / "merged").mkdir(parents=True)
    (src / "measurements").mkdir(parents=True)
    png_dir = src / "data" / "cell_png"
    png_dir.mkdir(parents=True)
    act_dir = src / "datasets" / "activation" / "saliency_image"
    act_dir.mkdir(parents=True)

    stack = np.zeros((32, 32, 3), dtype=np.uint16)
    stack[..., 0] = rng.integers(100, 4000, (32, 32))
    stack[..., 1] = rng.integers(100, 4000, (32, 32))
    cell = np.zeros((32, 32), np.uint16)
    cell[4:14, 4:14] = 1
    stack[..., 2] = cell
    np.save(src / "merged" / f"{FOV_NAME}.npy", stack)

    png_paths, act_paths = [], []
    for i in range(n_crops):
        png_paths.append(_write_png(png_dir / f"{FOV_NAME}_o{i + 1}.png", rng))
        act_paths.append(_write_png(act_dir / f"{FOV_NAME}_o{i + 1}.png", rng))

    meta = {
        "plateID": ["plate1"] * n_crops,
        "rowID": ["r1"] * n_crops,
        "columnID": ["c1"] * n_crops,
        "fieldID": ["f1"] * n_crops,
        "cell_id": [f"o{i + 1}" for i in range(n_crops)],
    }
    con = sqlite3.connect(src / "measurements" / "measurements.db")
    try:
        pd.DataFrame({**meta, "png_path": png_paths}).to_sql(
            "png_list", con, index=False)
    finally:
        con.close()
    con = sqlite3.connect(src / "measurements" / "activation.db")
    try:
        pd.DataFrame({**meta, "png_path": act_paths}).to_sql(
            "saliency_image_list", con, index=False)
    finally:
        con.close()
    return src


def _region_settings(src, **over):
    settings = {
        "src": str(src),
        "name": f"{FOV_NAME}.npy",
        "channels": [0, 1],
        "cell_channel": 0,
        "nucleus_channel": None,
        "pathogen_channel": None,
        "percentiles": (2, 98),
        "activation_mode": "saliency_image",
        "activation_db": "activation.db",
        "mode": "outlines",
        "export_tiffs": False,
    }
    settings.update(over)
    return settings


def test_an_overlay_that_came_back_empty_costs_no_file_and_no_crash(
        tmp_path, rng, monkeypatch):
    """``plot_region`` saves each of its three figures only if there is one.
    The mask overlay is the one that can come back as ``None`` -- a field of
    view whose merged stack the renderer refused -- and handing ``None`` to
    the saver is an ``AttributeError`` that loses the two figures that DID
    render. The unstubbed run in the same test shows all three being written,
    so the missing file is known to be the overlay's absence and not the
    saving being broken."""
    import spacr.plot as P
    from spacr.plot import plot_region

    src = _make_region_src(tmp_path, rng)
    dst = src / "results" / FOV_NAME

    real = P.plot_image_mask_overlay

    def _small(*args, **kwargs):
        fig = real(*args, **kwargs)
        if fig is not None:
            fig.set_size_inches(4, 1)
        return fig

    monkeypatch.setattr(P, "plot_image_mask_overlay", _small)
    fig_1, fig_2, fig_3 = plot_region(_region_settings(src))
    assert isinstance(fig_1, matplotlib.figure.Figure)
    assert (dst / f"{FOV_NAME}_mask_overlay.pdf").is_file()
    assert (dst / f"{FOV_NAME}_png_grid.pdf").is_file()
    assert (dst / f"{FOV_NAME}_activation_grid.pdf").is_file()

    for name in ("mask_overlay", "png_grid", "activation_grid"):
        (dst / f"{FOV_NAME}_{name}.pdf").unlink()
    monkeypatch.setattr(P, "plot_image_mask_overlay",
                        lambda *args, **kwargs: None)

    empty_1, empty_2, empty_3 = plot_region(_region_settings(src))

    assert empty_1 is None
    assert isinstance(empty_2, matplotlib.figure.Figure)
    assert isinstance(empty_3, matplotlib.figure.Figure)
    assert not (dst / f"{FOV_NAME}_mask_overlay.pdf").exists()
    assert (dst / f"{FOV_NAME}_png_grid.pdf").is_file()
    assert (dst / f"{FOV_NAME}_activation_grid.pdf").is_file()

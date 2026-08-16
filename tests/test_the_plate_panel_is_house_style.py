"""The plate heatmaps, in the house style, as one panel with square wells.

Instruction 124 A, and the loudest of the figure complaints:

    "the lpates look super small on the collected figure please"

The plates were laid out four-per-row on a 40 x 5 inch figure. Letterboxed
into a 230 px tile of the figure grid that is a 230 x 28 strip -- wells 2.7
by 1.8 pixels -- which is what "super small" describes. Three things are
pinned here, all of them measured on the real screen before they were
changed:

1. **The wells are square.** They were 0.275 x 0.241 inches, a 1.14:1
   rectangle. A plate heatmap with rectangular wells is not a heatmap of a
   plate; positional artefacts, the whole reason to look at one, stop being
   visible. That is what instruction 117 exists for.

2. **One colour scale across the plates.** Each plate was scaled on its own,
   so on the tsg101 screen the top of the scale was 0.243 on plate 1 and
   0.281 on plate 3 -- the same blue, adjacent, meaning different numbers,
   with four separate colour bars.

3. **A well that was never measured is not a measurement of zero.**
   ``generate_plate_heatmap`` ends in ``.fillna(0)``; 155 of a tsg101
   plate's 384 wells carry data, so 54% of every panel was a solid block of
   the lowest colour there is -- and, through ``min_max='allq'``, those
   invented zeros SET the bottom of the scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spacr.figures.plates import (build_plates, full_plate_grid, plate_names,
                                  plate_ramp, shared_limits,
                                  small_multiple_layout, well_matrices)
from spacr.figures.style import Palette


@pytest.fixture(autouse=True)
def _no_figure_leak():
    plt.close("all")
    yield
    plt.close("all")


def _screen(n_plates=4, rows=16, columns=24, first_column=4, seed=0,
            per_well=3):
    """A screen shaped like the real one: a 384 plate whose first three
    columns were never used, and roughly half its wells measured."""
    rng = np.random.default_rng(seed)
    records = []
    for plate in range(1, n_plates + 1):
        for row in range(1, rows + 1):
            for column in range(first_column, columns + 1):
                if rng.random() < 0.45:
                    continue
                for _ in range(per_well):
                    records.append({
                        "prc": f"plate{plate}_r{row}_c{column}",
                        "pred": float(rng.uniform(0.02, 0.6)),
                    })
    return pd.DataFrame.from_records(records)


def _plate_axes(figure):
    return [ax for ax in figure.axes if ax.images]


def _well_size(figure, ax):
    """The drawn size of one well, in inches."""
    box = ax.get_window_extent(figure.canvas.get_renderer())
    n_rows, n_columns = ax.images[0].get_array().shape
    return (box.width / figure.dpi / n_columns,
            box.height / figure.dpi / n_rows)


# --------------------------------------------------------------------------- #
#  The non-negotiable
# --------------------------------------------------------------------------- #

def test_the_wells_are_square():
    """Measured through the function the pipeline actually calls."""
    from spacr.plot import plot_plates

    figure = plot_plates(_screen(), "pred", "mean", "allq", "viridis",
                         min_count=0, verbose=False, dst=None)
    figure.canvas.draw()
    axes = _plate_axes(figure)
    assert len(axes) == 4
    for ax in axes:
        width, height = _well_size(figure, ax)
        assert width == pytest.approx(height, rel=1e-6), (
            f"{ax.get_title()} draws {width:.4f} x {height:.4f} inch wells; "
            f"a plate heatmap with rectangular wells is not a heatmap of a "
            f"plate")


def test_the_panel_is_not_a_strip():
    """40 x 5 inches is 8:1. In a 230 px tile that is 28 pixels tall."""
    from spacr.plot import plot_plates

    figure = plot_plates(_screen(), "pred", "mean", "allq", "viridis",
                         min_count=0, verbose=False, dst=None)
    width, height = figure.get_size_inches()
    assert width / height < 2.0, (
        f"the four plates are still a {width / height:.1f}:1 strip")


def test_four_plates_are_a_small_multiple_and_not_a_row():
    """The choice made: one figure, plates stacked, not four across."""
    from spacr.plot import plot_plates

    figure = plot_plates(_screen(), "pred", "mean", "allq", "viridis",
                         min_count=0, verbose=False, dst=None)
    figure.canvas.draw()
    lefts = {round(ax.get_position().x0, 3) for ax in _plate_axes(figure)}
    tops = {round(ax.get_position().y0, 3) for ax in _plate_axes(figure)}
    assert len(lefts) == 2 and len(tops) == 2, (
        f"expected a 2x2 small multiple, got {len(lefts)} columns and "
        f"{len(tops)} rows")


# --------------------------------------------------------------------------- #
#  One scale, because the point is comparing plates
# --------------------------------------------------------------------------- #

def test_every_plate_is_on_the_same_colour_scale():
    from spacr.plot import plot_plates

    figure = plot_plates(_screen(), "pred", "mean", "allq", "viridis",
                         min_count=0, verbose=False, dst=None)
    scales = {(round(ax.images[0].norm.vmin, 9),
               round(ax.images[0].norm.vmax, 9))
              for ax in _plate_axes(figure)}
    assert len(scales) == 1, (
        f"four plates, {len(scales)} colour scales: {sorted(scales)}")


def test_there_is_one_colour_bar_for_the_whole_panel():
    from spacr.plot import plot_plates

    figure = plot_plates(_screen(), "pred", "mean", "allq", "viridis",
                         min_count=0, verbose=False, dst=None)
    bars = [ax for ax in figure.axes if not ax.images]
    assert len(bars) == 1, f"{len(bars)} colour bars for one scale"


# --------------------------------------------------------------------------- #
#  A well that was never measured
# --------------------------------------------------------------------------- #

def test_an_unmeasured_well_is_masked_rather_than_drawn_as_zero():
    frame = pd.DataFrame({
        "prc": ["p1_r1_c1", "p1_r1_c1", "p1_r2_c2", "p1_r3_c3"],
        "value": [4.0, 6.0, 7.0, 9.0],
    })
    _names, matrices, _grid = well_matrices(frame, "value")
    block = matrices[0]
    assert block[0, 0] == 5.0
    assert np.isnan(block[0, 1]), (
        "a well with no rows came back as a number; it is indistinguishable "
        "from a well that measured that number")
    assert int(np.isfinite(block).sum()) == 3


def test_the_invented_zeros_do_not_set_the_colour_scale():
    """The failure this reproduces, on the real screen: the drawn range was
    0.000-0.243 where the range of the wells that exist is 0.060-0.273."""
    frame = _screen(n_plates=1)
    _names, matrices, _grid = well_matrices(frame, "pred")
    low, high = shared_limits(matrices, "allq")
    assert low > 0.0, "the scale still starts at an invented zero"
    finite = matrices[0][np.isfinite(matrices[0])]
    assert low == pytest.approx(float(np.quantile(finite, 0.02)))
    assert high == pytest.approx(float(np.quantile(finite, 0.98)))


def test_a_well_with_rows_but_no_readable_number_is_not_a_zero_either():
    """One step further in than ``.fillna(0)``.

    The presence map counts ROWS, and ``generate_plate_heatmap`` coerces the
    variable with ``errors='coerce'`` -- so a well whose every row holds
    nothing numeric aggregates to NaN, is filled with 0, and, having a row
    count above zero, survives the mask as a measurement of zero. It then
    sets the bottom of the shared scale, which is the whole defect again.
    """
    frame = pd.DataFrame({
        "prc": ["p1_r1_c1", "p1_r2_c1", "p1_r2_c2", "p1_r3_c1"],
        "value": [4.0, np.nan, 7.0, 8.0],
    })
    _names, matrices, _grid = well_matrices(frame, "value")
    block = matrices[0]
    assert block[0, 0] == 4.0
    assert np.isnan(block[1, 0]), (
        "a well whose every measurement is missing was drawn as a "
        "measurement of zero")
    assert int(np.isfinite(block).sum()) == 3
    assert shared_limits(matrices, "all") == (4.0, 8.0), (
        "the invented zero is still setting the colour scale")


def test_a_well_that_is_only_unreadable_text_is_absent_too():
    """``errors='coerce'`` is what turns 'n/a' into a NaN, and a column of
    strings is what a real results table hands a plotter."""
    frame = pd.DataFrame({
        "prc": ["p1_r1_c1", "p1_r2_c1", "p1_r2_c2"],
        "value": ["4.0", "n/a", "7.0"],
    })
    _names, matrices, _grid = well_matrices(frame, "value")
    assert np.isnan(matrices[0][1, 0])
    assert int(np.isfinite(matrices[0]).sum()) == 2


def test_a_well_dropped_by_min_count_reads_as_absent_not_as_zero():
    frame = pd.DataFrame({
        "prc": ["p1_r1_c1"] * 3 + ["p1_r2_c2"] + ["p1_r3_c3"] * 3,
        "value": [4.0, 5.0, 6.0, 100.0, 1.0, 2.0, 3.0],
    })
    _names, matrices, _grid = well_matrices(frame, "value", min_count=2)
    block = matrices[0]
    assert np.isnan(block[1, 1]), "the singleton well became a zero"
    assert np.nanmax(block) == 5.0


# --------------------------------------------------------------------------- #
#  The plate is the plate, not the bounding box of its used wells
# --------------------------------------------------------------------------- #

def test_a_screen_that_never_used_column_one_still_draws_it():
    """THE EDGE HAS TO BE THE EDGE. Pivoting only the wells that carry data
    puts the first measured column hard against the left spine, and every
    edge effect then reads one plate position out."""
    _names, matrices, grid = well_matrices(_screen(), "pred")
    assert grid == (16, 24)
    assert np.isnan(matrices[0][:, :3]).all(), (
        "columns 1-3 were never measured and must not be invented")


@pytest.mark.parametrize("rows,columns,expected", [
    ([1, 8], [1, 12], (8, 12)),
    ([1, 16], [4, 21], (16, 24)),
    ([1, 32], [1, 48], (32, 48)),
    ([1, 40], [1, 60], (40, 60)),      # no standard format: the bounding box
])
def test_the_grid_is_the_plate_the_wells_sit_on(rows, columns, expected):
    assert full_plate_grid(rows, columns) == expected


# --------------------------------------------------------------------------- #
#  The layout
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("count,expected", [
    (1, (1, 1)),
    (2, (2, 1)),      # 0.75:1 stacked, against 3:1 side by side
    (4, (2, 2)),      # 1.5:1, against 6:1 in a row
    (6, (3, 2)),      # 1:1, against 2.25:1 three across
    (8, (3, 3)),
])
def test_the_small_multiple_is_the_arrangement_nearest_square(count, expected):
    """Four plates in a row is a 6:1 composite; 2 x 2 is 1.5:1. Both hold the
    same picture and only one of them fills a tile."""
    assert small_multiple_layout(count, 24 / 16) == expected


def test_no_plates_is_an_answer_and_not_an_exception():
    """``plt.subplots(0, 4)`` raised. A frame with no readable plate is a
    real state -- a filter that removed everything -- and must come back
    saying so."""
    figure, panel = build_plates(pd.DataFrame({"prc": [], "pred": []}), "pred")
    assert not panel.drawn
    assert panel.reason


# --------------------------------------------------------------------------- #
#  The house style
# --------------------------------------------------------------------------- #

def test_the_style_does_not_leak_into_the_process():
    """spaCR draws from a long-lived GUI: a global rcParams change restyles
    every later figure in the session."""
    from spacr.plot import plot_plates

    before = dict(plt.rcParams)
    plot_plates(_screen(n_plates=2), "pred", "mean", "allq", "viridis",
                min_count=0, verbose=False, dst=None)
    changed = {key for key in before
               if str(before[key]) != str(plt.rcParams[key])}
    assert not changed, f"these rcParams were left changed: {sorted(changed)}"


def test_there_are_no_gridlines():
    from spacr.plot import plot_plates

    figure = plot_plates(_screen(n_plates=2), "pred", "mean", "allq",
                         "viridis", min_count=0, verbose=False, dst=None)
    for ax in _plate_axes(figure):
        assert not any(line.get_visible()
                       for line in ax.get_xgridlines() + ax.get_ygridlines())


def test_the_legacy_viridis_literal_becomes_the_house_ramp():
    """Every internal call site passed ``cmap='viridis'``; viridis is not in
    the palette and was never a choice. Any OTHER colormap is honoured."""
    from spacr.plot import plot_plates

    figure = plot_plates(_screen(n_plates=1), "pred", "mean", "allq",
                         "viridis", min_count=0, verbose=False, dst=None)
    assert _plate_axes(figure)[0].images[0].get_cmap().name.startswith(
        "spacr_plate")

    plt.close(figure)
    figure = plot_plates(_screen(n_plates=1), "pred", "mean", "allq",
                         "magma", min_count=0, verbose=False, dst=None)
    assert _plate_axes(figure)[0].images[0].get_cmap().name == "magma"


def test_the_ramp_is_a_single_blue_hue_and_ends_short_of_black_on_screen():
    """"Sequential encodings use a single-hue blue ramp, light→dark." On the
    dark theme the print ramp's NAVY end is within a hair of the ground, and
    a well at the top of the scale must not be confusable with a well that
    has no measurement."""
    from matplotlib.colors import to_rgb

    printed = plate_ramp("print")(1.0)[:3]
    screen = plate_ramp("screen")(1.0)[:3]
    assert printed == pytest.approx(to_rgb(Palette.NAVY), abs=0.01)
    assert screen == pytest.approx(to_rgb(Palette.BLUE), abs=0.01)
    # Light to dark, monotonically, at both ends of both ramps.
    for ramp in (plate_ramp("print"), plate_ramp("screen")):
        levels = [sum(ramp(value)[:3]) for value in np.linspace(0, 1, 9)]
        assert all(b <= a + 1e-9 for a, b in zip(levels, levels[1:])), levels


# --------------------------------------------------------------------------- #
#  What a run leaves on disk
# --------------------------------------------------------------------------- #

def test_the_file_is_named_for_what_it_draws_and_is_rewritten(tmp_path):
    """The old loop took the first free ``plate_heatmap_<n>.pdf`` and never
    overwrote, so the real screen's results folder holds twelve
    byte-identical copies of one figure and the grid showed all twelve."""
    from spacr.plot import plot_plates

    frame = _screen(n_plates=2)
    for _ in range(3):
        plot_plates(frame, "pred", "mean", "allq", "viridis", min_count=0,
                    verbose=False, dst=str(tmp_path))
    written = sorted(path.name for path in tmp_path.iterdir())
    assert written == ["plate_heatmap_pred.pdf"], written
    assert (tmp_path / "plate_heatmap_pred.pdf").stat().st_size > 0


def test_two_measurements_get_two_files(tmp_path):
    from spacr.plot import plot_plates

    frame = _screen(n_plates=2)
    frame["log_pred"] = np.log1p(frame["pred"])
    for column in ("pred", "log_pred"):
        plot_plates(frame, column, "mean", "allq", "viridis", min_count=0,
                    verbose=False, dst=str(tmp_path))
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "plate_heatmap_log_pred.pdf", "plate_heatmap_pred.pdf"]


# --------------------------------------------------------------------------- #
#  The caller's frame, and the option that stays open
# --------------------------------------------------------------------------- #

def test_the_callers_frame_is_not_written_on():
    """``generate_plate_heatmap`` adds plateID/rowID/columnID to the frame it
    is handed. A plotter must not leave columns behind in the caller's
    table."""
    frame = _screen(n_plates=1)
    before = list(frame.columns)
    build_plates(frame, "pred")
    assert list(frame.columns) == before


def test_one_plate_per_figure_can_still_share_the_scale():
    """Instruction 124 offers one plate per slot as the alternative. It is
    only correct if the four figures share one scale, which is what
    ``limits=`` is for -- without it each figure would silently rescale to
    its own plate, which is the defect this work removed."""
    frame = _screen()
    names, matrices, _grid = well_matrices(frame, "pred")
    limits = shared_limits(matrices, "allq")
    scales = set()
    for name in names:
        figure, panel = build_plates(frame, "pred", plates=[name],
                                     limits=limits)
        assert panel.drawn
        scales.add((round(_plate_axes(figure)[0].images[0].norm.vmin, 9),
                    round(_plate_axes(figure)[0].images[0].norm.vmax, 9)))
        plt.close(figure)
    assert len(scales) == 1, sorted(scales)


def test_the_caption_says_how_many_wells_were_never_measured():
    """The legend is generated from what was drawn, not written twice."""
    figure, panel = build_plates(_screen(), "pred")
    assert panel.drawn
    assert "share one colour scale" in panel.caption
    assert "wells were measured" in panel.caption
    assert "16x24" in panel.caption


def test_plate_names_reads_the_plate_out_of_prc():
    frame = pd.DataFrame({"prc": ["p2_r1_c1", "p1_r1_c1", "p2_r2_c2"],
                          "value": [1.0, 2.0, 3.0]})
    assert plate_names(frame) == ["p2", "p1"]
    assert plate_names(pd.DataFrame({"value": [1.0]})) == []

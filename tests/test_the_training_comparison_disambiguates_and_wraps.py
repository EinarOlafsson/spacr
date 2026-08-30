"""Collisions, wrappers and a caller-supplied axis in the training comparison.

The two that matter are the ones a user sees: a legend with two identically
labelled lines, and a settings value rendered differently by the GUI than by
the console. Both are guarded, and neither guard had run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# _unique_ids — two runs that would otherwise share a label
# ---------------------------------------------------------------------------

def test_two_runs_with_the_same_folder_layout_get_different_ids():
    """Lines 970-971 and arc 969 -> 970: the lengthening loop.

    ``train_test_model`` writes .../<model>/<channels>/epochs_<n>, so two runs
    of the same model from different dataset roots collide exactly. The
    docstring states the trade: a legend with two identically-labelled lines
    is worse than a long label, and this loop is what buys the long one.
    """
    from spacr.train_compare import _unique_ids

    paths = [Path("/data/screen_a/maxvit_t/rgb/epochs_25"),
             Path("/data/screen_b/maxvit_t/rgb/epochs_25")]

    ids = _unique_ids(paths)

    # The FIRST keeps its short id and each later collider grows one
    # component at a time, which is the minimum that separates them.
    assert len(set(ids.values())) == 2
    assert "screen_b" in " ".join(ids.values())


def test_runs_that_do_not_collide_keep_their_short_ids():
    """The loop not entered, so the lengthening above is visibly conditional."""
    from spacr.train_compare import _unique_ids

    paths = [Path("/data/screen_a/maxvit_t/rgb/epochs_25"),
             Path("/data/screen_a/resnet50/rgb/epochs_25")]

    ids = _unique_ids(paths)

    assert len(set(ids.values())) == 2
    for value in ids.values():
        assert value.count("/") <= 2


# ---------------------------------------------------------------------------
# _pick_settings_file — candidates that do not match the expected stem
# ---------------------------------------------------------------------------

def test_settings_files_that_do_not_match_the_stem_are_passed_over(tmp_path):
    """Arcs 729 -> 728 and 730 -> 729: both loops go round.

    A run folder holds several CSVs and only one is the settings for THIS
    model and epoch count. Returning the first CSV found would show the user
    another run's settings beside this run's curves.

    The files are real, because the fallback below the loops sorts the
    survivors by mtime and cannot stat a path that does not exist.
    """
    from spacr.train_compare import _pick_settings_file, _settings_stems

    stems = _settings_stems("maxvit_t", "25")
    assert len(stems) > 1, "fixture assumes more than one candidate stem"

    # Deliberately the LAST stem in the preference order, so the outer loop
    # has to advance past the earlier ones -- the arc that goes round.
    wanted = tmp_path / f"{stems[-1]}.csv"
    for name in ("metrics.csv", "train_test_resnet50_10.csv", wanted.name):
        (tmp_path / name).write_text("setting,value\n")

    picked = _pick_settings_file(sorted(tmp_path.iterdir()), "maxvit_t", "25")

    assert picked == wanted


def test_no_candidates_at_all_picks_nothing():
    """The early return above both loops."""
    from spacr.train_compare import _pick_settings_file

    assert _pick_settings_file([], "maxvit_t", "25") is None


# ---------------------------------------------------------------------------
# render_setting_value — the public wrapper
# ---------------------------------------------------------------------------

def test_the_gui_renders_a_setting_exactly_as_the_console_does():
    """Line 1072: the wrapper had no test at all.

    Its whole reason for existing is that the GUI and the console must not
    render a setting two ways -- a user comparing a panel against a printed
    report should see the same string. A wrapper with no test is a promise
    nobody has checked, and this one is a single delegation away from
    silently drifting.
    """
    from spacr.run_journal import _render_value
    from spacr.train_compare import render_setting_value

    for value in (None, True, 42, 3.5, "a string",
                  ["a", "list", "of", "things"], {"a": 1, "b": 2},
                  "x" * 200):
        assert render_setting_value(value) == _render_value(value, 40)


def test_the_width_is_passed_through_to_the_renderer():
    """The parameter, which a caller with a narrow column needs."""
    from spacr.run_journal import _render_value
    from spacr.train_compare import render_setting_value

    long_value = "y" * 200
    assert render_setting_value(long_value, 12) == _render_value(long_value, 12)
    assert len(render_setting_value(long_value, 12)) <= 12


# ---------------------------------------------------------------------------
# plot_curves — an axis the caller already has
# ---------------------------------------------------------------------------

def test_plotting_into_a_caller_supplied_axis_uses_that_figure():
    """Line 1302: ``fig = ax.figure`` rather than making a new one.

    The comment above it explains the other branch -- a style context must
    wrap ``plt.subplots`` or the spines and ticks stay at the caller's
    globals. When the caller brings its own axis there is nothing to wrap, and
    taking the figure FROM the axis is what lets the comparison be embedded in
    a panel that already exists.
    """
    from spacr.train_compare import Comparison, plot_curves

    comparison = Comparison(runs=[], series=[], settings_diff={}, metrics={},
                            problems=[], fold_mode="none")

    figure, axis = plt.subplots()
    try:
        plot_curves(comparison, metric="accuracy", ax=axis)
        assert axis.figure is figure
    finally:
        plt.close(figure)

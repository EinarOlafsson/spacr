"""Three small readers, each on the input it declines to use.

A channel that is not a number, a table with no coefficient column, a panel
that was never drawn. In each case the caller carries on with less rather than
stopping -- and each "less" is a different, deliberate answer.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# _v1_v2_bridge.v2_channels_from_settings
# ---------------------------------------------------------------------------

def test_a_channel_that_is_not_a_number_is_passed_over():
    """The ``except (TypeError, ValueError): continue``.

    A settings CSV round trip turns a cleared channel into '' and a mistyped
    one into a word. Either would raise on int(), and the run would stop
    before reading an image -- over a channel the user did not mean to set.
    """
    from spacr._v1_v2_bridge import _CHANNEL_KEYS, v2_channels_from_settings

    keys = [key for key, _human in _CHANNEL_KEYS]
    settings = {keys[0]: 0, keys[1]: "not a number"}

    channels, names = v2_channels_from_settings(settings)

    assert channels == [0]
    assert len(names) == 1


def test_an_unset_channel_is_skipped_without_reaching_the_int():
    """The ``if v is None: continue`` above it, which is the common case."""
    from spacr._v1_v2_bridge import _CHANNEL_KEYS, v2_channels_from_settings

    keys = [key for key, _human in _CHANNEL_KEYS]
    channels, _names = v2_channels_from_settings({keys[0]: 1, keys[1]: None})

    assert channels == [1]


def test_a_top_level_channels_list_is_the_fallback():
    """The documented fallback for a user who set ``channels`` instead."""
    from spacr._v1_v2_bridge import v2_channels_from_settings

    channels, names = v2_channels_from_settings({"channels": [0, 2]})

    assert channels == [0, 2]
    assert len(names) == 2


def test_no_channels_anywhere_falls_back_to_a_default_four():
    """Both routes empty, which a settings file for another module produces.

    The answer is a default set rather than nothing: a run with no channel
    settings still has an image, and returning ([], []) would leave the
    pipeline with nothing to read from a stack that plainly has planes.
    """
    from spacr._v1_v2_bridge import v2_channels_from_settings

    channels, names = v2_channels_from_settings({})

    assert channels == [0, 1, 2, 3]
    assert len(names) == len(channels)


# ---------------------------------------------------------------------------
# baseline.resolve
# ---------------------------------------------------------------------------

def test_a_table_with_no_effect_column_falls_back_to_zero_and_says_why():
    """The reason field, which is what makes the fallback auditable.

    The caption still reads "measured from zero", and the REASON records that
    the column was absent rather than that the user chose zero. Without it a
    figure and a deliberate choice look identical.
    """
    from spacr.baseline import CONTROLS, resolve

    frame = pd.DataFrame({"gene": ["a", "b"]})

    baseline = resolve(frame, CONTROLS, column="coefficient")

    assert baseline.shift == 0.0
    assert baseline.reason and "coefficient" in baseline.reason
    assert "measured from zero" in baseline.sentence


def test_asking_for_zero_needs_no_column_at_all():
    """The first return, which is why the column check comes second."""
    from spacr.baseline import ZERO, resolve

    baseline = resolve(pd.DataFrame(), ZERO)

    assert baseline.shift == 0.0
    assert not baseline.reason
    assert "no dose-response" in baseline.sentence


# ---------------------------------------------------------------------------
# figures.sheet.attach
# ---------------------------------------------------------------------------

def test_attaching_no_panel_leaves_the_figure_alone():
    """The ``if panel is None: return``.

    A panel that could not be drawn is None, and the sheet still calls attach.
    Labelling the figure with a panel that does not exist would put a caption
    on a blank axis.
    """
    from spacr.figures.sheet import attach

    figure = plt.figure()
    try:
        attach(figure, None)
        assert figure.get_label() == ""
        assert not hasattr(figure, "_spacr_title")
    finally:
        plt.close(figure)


def test_attaching_a_panel_labels_the_figure_for_the_journey_ahead():
    """The private attributes, and the docstring's reason for them.

    "the figure is handed through the Qt bridge, the queue, a spill file and
    back, and a wrapper would be lost at the first of those" -- so the title
    rides on the Figure itself.
    """
    from spacr.figures.panels import Panel
    from spacr.figures.sheet import attach

    figure = plt.figure()
    try:
        attach(figure, Panel("volcano", "volcano plot", caption="A caption."))

        assert figure.get_label() == "volcano plot"
        assert figure._spacr_title == "volcano plot"
        assert figure._spacr_caption == "A caption."
    finally:
        plt.close(figure)


def test_a_panel_with_no_title_is_labelled_with_its_key():
    """The ``or panel.key`` fallback, so a figure is never nameless."""
    from spacr.figures.panels import Panel
    from spacr.figures.sheet import attach

    figure = plt.figure()
    try:
        attach(figure, Panel("volcano", ""))
        assert figure.get_label() == "volcano"
    finally:
        plt.close(figure)


def test_a_panel_with_no_data_attaches_none():
    """The ``if getattr(panel, "data", None) is not None`` guard.

    A panel that drew from a computed array carries no rows to export, and
    attaching an empty frame would offer the user a download of nothing.
    """
    from spacr.figures.panels import Panel
    from spacr.figures.sheet import attach

    figure = plt.figure()
    try:
        attach(figure, Panel("volcano", "volcano plot"))
        assert not hasattr(figure, "_spacr_data")
    finally:
        plt.close(figure)

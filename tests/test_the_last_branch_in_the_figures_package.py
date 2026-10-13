"""Four more last branches: three in spacr/figures and one in the family list.

Each is the side of a decision where the code declines to say something --
declines to report a skew it cannot compute, declines to call a threshold
control-based, declines to letter a panel with no caption, declines to offer a
family that cannot be fitted. Declining quietly is the behaviour that a
regression would turn into a confident wrong statement, which is why these
arcs are worth a test rather than a pragma.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# figures.distributions.response — arc 436 -> 440, too few wells to compare
# ---------------------------------------------------------------------------

def test_the_skew_before_a_log_is_omitted_when_too_few_wells_survive():
    """The ``if raw_values.size >= 3:`` branch not taken.

    The annotation exists to say what the log transform bought. A skew
    computed from one or two wells is not a shape, it is an artefact, and
    printing it beside the after-value would invite exactly the comparison it
    cannot support. Three is the floor, and below it the sentence is dropped
    rather than filled with a number.
    """
    from spacr.figures.distributions import response

    # Ten wells so the histogram itself is drawable, but the RAW column is
    # almost entirely missing -- which is what a partially computed table
    # looks like, and it leaves fewer than three wells to take a skew from.
    frame = pd.DataFrame({
        "prc": [f"A{i:03d}" for i in range(10)],
        "value": [1.0, 2.0] + [float("nan")] * 8,
        "log_value": np.linspace(0.0, 2.0, 10),
    })
    fig, ax = plt.subplots()
    try:
        panel = response(ax, frame, column="log_value", well="prc")
        text = " ".join(t.get_text() for t in ax.texts)
        assert "skew before log" not in text
        assert panel.drawn
    finally:
        plt.close(fig)


def test_the_skew_before_a_log_is_reported_when_enough_wells_survive():
    """The taken side, so the omission above is visibly a decision."""
    from spacr.figures.distributions import response

    rng = np.random.default_rng(0)
    raw = np.abs(rng.lognormal(0.0, 1.0, 40)) + 0.1
    frame = pd.DataFrame({
        "prc": [f"A{i:03d}" for i in range(40)],
        "value": raw,
        "log_value": np.log(raw),
    })
    fig, ax = plt.subplots()
    try:
        response(ax, frame, column="log_value", well="prc")
        text = " ".join(t.get_text() for t in ax.texts)
        assert "skew before log" in text
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# figures.panels.control_threshold — arc 184 -> 190, controls with no spread
# ---------------------------------------------------------------------------

def test_identical_controls_do_not_become_a_threshold_of_zero():
    """The ``if spread > 0:`` branch not taken.

    Controls that are all the same value give a MAD of exactly zero, and a
    threshold of zero would mark EVERY coefficient a hit -- the most damaging
    possible failure of this function, and a silent one, because zero is a
    number and the panel would draw it without complaint. Falling through to
    the all-guide MAD, under a different name, is the protection.
    """
    from spacr.figures.panels import control_threshold, MIN_CONTROLS

    n = MIN_CONTROLS + 2
    frame = pd.DataFrame({
        "condition": ["control"] * n + ["treated"] * n,
        "coefficient": [0.5] * n + list(np.linspace(-2.0, 2.0, n)),
    })
    label, value = control_threshold(frame)

    assert value is None or value > 0
    assert "controls" not in (label or "")     # never claimed as control-based


def test_controls_with_spread_do_give_a_control_based_threshold():
    """The taken side: the same shape of table, controls that actually vary."""
    from spacr.figures.panels import control_threshold, MIN_CONTROLS

    n = MIN_CONTROLS + 2
    frame = pd.DataFrame({
        "condition": ["control"] * n + ["treated"] * n,
        "coefficient": list(np.linspace(-1.0, 1.0, n))
                       + list(np.linspace(-2.0, 2.0, n)),
    })
    label, value = control_threshold(frame)

    assert value is not None and value > 0
    assert "controls" in label


# ---------------------------------------------------------------------------
# figures.sheet.Sheet.legend — arc 60 -> 59, a panel with no caption
# ---------------------------------------------------------------------------

def test_a_panel_without_a_caption_takes_no_letter_in_the_legend():
    """The ``if panel.caption:`` branch not taken.

    An empty caption would otherwise produce "(B) " -- a letter pointing at
    nothing, in a published figure legend. Skipping it is right, and it means
    the letters in the legend are those of the panels that HAVE something to
    say, which is what a reader matches against the figure.
    """
    from spacr.figures.panels import Panel
    from spacr.figures.sheet import Sheet

    sheet = Sheet(figure=None,
                  panels=[Panel("a", "first", caption="The first thing."),
                          Panel("b", "second", caption=""),
                          Panel("c", "third", caption="The third thing.")],
                  skipped=[])
    legend = sheet.legend()

    assert "(A) The first thing." in legend
    assert "(B)" not in legend
    assert "(C) The third thing." in legend


def test_skipped_panels_are_named_at_the_end_of_the_legend():
    """The other half of the same method, and the reason skipping is safe."""
    from spacr.figures.panels import Panel
    from spacr.figures.sheet import Sheet

    sheet = Sheet(figure=None,
                  panels=[Panel("a", "first", caption="The first thing.")],
                  skipped=[Panel("z", "volcano", drawn=False,
                                 reason="no p-values in the table")])
    legend = sheet.legend()

    assert "Not shown: volcano (no p-values in the table)." in legend


# ---------------------------------------------------------------------------
# regression_families.regression_family_choices — arc 163 -> 162
# ---------------------------------------------------------------------------

def test_a_family_that_cannot_be_fitted_is_not_offered(monkeypatch):
    """The ``if name in fittable:`` branch not taken.

    The list this builds is what the settings panel offers. A family whose
    dependency is missing must not appear in it: choosing it would fail at fit
    time, long after the user made the choice and with an error that names a
    package rather than the setting. The filter is the whole point of the
    function, and the case where it actually removes something had never run.
    """
    from spacr import regression_families as rf

    everything = rf._fittable()
    assert "beta" in everything, "fixture assumes beta is normally fittable"
    monkeypatch.setattr(rf, "_fittable",
                        lambda: tuple(n for n in everything if n != "beta"))

    offered = [name for name, _label in rf.regression_family_choices()]

    assert "beta" not in offered
    assert offered, "the rest of the families are still offered"


def test_every_fittable_family_is_offered_with_a_label():
    """The taken side, and the contract the panel depends on."""
    from spacr import regression_families as rf

    offered = rf.regression_family_choices()

    assert offered
    for name, label in offered:
        assert name in rf._fittable()
        assert label and isinstance(label, str)

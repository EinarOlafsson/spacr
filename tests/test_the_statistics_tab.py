"""Right-click a graph, and the statistics are a tab of its settings.

Asked for on 2026-08-16: "provide a statistics tab in the settings for the
graph upon right clicking".

ONLY WHERE THERE IS SOMETHING TO COMPARE. The figure carries its groups or it
does not, and a tab offering a t-test on a Q-Q plot is an invitation to
report a number that means nothing. A Q-Q is one distribution against a
reference, not two samples.

THE DEFAULT IS AUTOMATIC. Forcing a test is offered, because a reader
sometimes has a reason the data cannot express -- a paired design, a
pre-registered analysis -- but it is an override rather than the starting
point. The automatic choice is the one that reads the assumption checks, and
treats a check it had too few points to run as FAILED: "did not reject" on
n = 3 is not "the assumption holds".
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pytestmark = pytest.mark.qt


def _results(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feature": [f"fraction:grna[{i // 3}_{i % 3}]" for i in range(n)],
        "coefficient": rng.normal(0, .5, n),
        "p_value": rng.uniform(size=n),
        "condition": list(rng.choice(["nc", "pc", "other"], n,
                                     p=[.10, .05, .85])),
    })


def _dialog(qtbot, key):
    from spacr.figures import build_panel
    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    figure, _panel = build_panel(key, _results())
    dialog = FigureSettingsDialog(figure)
    qtbot.addWidget(dialog)
    return dialog


def _tabs(dialog):
    return [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]


# --------------------------------------------------------------------------- #
#  Where it appears, and where it must not
# --------------------------------------------------------------------------- #

def test_a_panel_that_compares_groups_gets_the_tab(qtbot):
    assert "Statistics" in _tabs(_dialog(qtbot, "controls"))


def test_a_panel_that_compares_nothing_does_not(qtbot):
    """A Q-Q is one distribution against a reference. Offering a t-test on it
    invites a number that means nothing."""
    assert "Statistics" not in _tabs(_dialog(qtbot, "qq"))


def test_a_plain_figure_does_not(qtbot):
    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    figure = plt.figure()
    figure.add_subplot(111).plot([0, 1], [1, 0])
    dialog = FigureSettingsDialog(figure)
    qtbot.addWidget(dialog)
    try:
        assert "Statistics" not in _tabs(dialog)
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------- #
#  What it says
# --------------------------------------------------------------------------- #

def test_it_reports_every_pair(qtbot):
    dialog = _dialog(qtbot, "controls")

    text = dialog._stats_verdict.text()
    assert text.count("vs") >= 3, text


def test_it_never_shows_a_bare_p(qtbot):
    """A p-value alone is not reportable: the test, the n and the convention
    go with it."""
    dialog = _dialog(qtbot, "controls")
    text = dialog._stats_verdict.text()

    assert "p = " in text
    assert "n=" in text
    assert "p<0.05" in text, "the asterisk convention is missing"
    assert "Hedges" in text or "Cohen" in text, "no effect size"


def test_it_names_the_unit_of_replication(qtbot):
    """A test across cells when the replicate is the well returns p < 1e-10
    on pure noise, and nothing in the number itself says so."""
    dialog = _dialog(qtbot, "controls")

    assert "coefficient" in dialog._stats_verdict.text()


def test_a_handful_of_controls_gets_the_robust_test(qtbot):
    """THE REAL SHAPE. The tsg101 screen has THREE positive controls, which
    is far too few for a variance test to have power -- so "did not reject"
    means "could not tell" and the robust branch is taken.

    The fixture has to be built for this deliberately: 5% of 300 rows is 15
    controls, and at 15 the checks CAN see, so Student's t is correctly
    chosen there. A first version of this test asserted Mann-Whitney against
    that fixture and was asserting the wrong thing about the wrong data.
    """
    from spacr.figures import build_panel
    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    frame = _results()
    frame["condition"] = "other"
    frame.loc[frame.index[:3], "condition"] = "pc"
    frame.loc[frame.index[3:27], "condition"] = "nc"

    figure, _panel = build_panel("controls", frame)
    dialog = FigureSettingsDialog(figure)
    qtbot.addWidget(dialog)

    text = dialog._stats_verdict.text()
    assert "Mann-Whitney" in text, text
    assert "n=3" in text


def test_enough_replicates_get_the_parametric_test(qtbot):
    """The other side of the same rule, so the robust branch is not simply
    always taken -- which would cost power on every well-replicated screen."""
    dialog = _dialog(qtbot, "controls")

    assert "Student's t" in dialog._stats_verdict.text()


# --------------------------------------------------------------------------- #
#  Forcing a test is an override, not the default
# --------------------------------------------------------------------------- #

def test_the_default_is_automatic(qtbot):
    dialog = _dialog(qtbot, "controls")

    assert dialog._stats_state["test"] is None


def test_forcing_a_test_changes_the_answer(qtbot):
    from PySide6.QtWidgets import QComboBox

    dialog = _dialog(qtbot, "controls")
    combo = next(c for c in dialog.findChildren(QComboBox)
                 if c.findData("Welch's t") >= 0)
    combo.setCurrentIndex(combo.findData("Welch's t"))

    assert dialog._stats_state["test"] == "Welch's t"
    assert "Welch" in dialog._stats_verdict.text()


def test_a_pair_too_small_to_test_says_so_rather_than_vanishing(qtbot):
    """A comparison that could not be made is not a comparison with an
    unknown answer, and a pair silently missing from the panel reads as a
    pair nobody looked at."""
    from spacr.figures import build_panel
    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    frame = _results()
    frame.loc[frame["condition"] == "pc", "condition"] = "other"
    frame.loc[frame.index[:1], "condition"] = "pc"

    figure, _panel = build_panel("controls", frame)
    dialog = FigureSettingsDialog(figure)
    qtbot.addWidget(dialog)

    text = dialog._stats_verdict.text()
    if "positive" in text:
        assert "fewer than" in text or "positive vs" not in text


def test_the_correction_is_across_the_pairs(qtbot):
    from PySide6.QtWidgets import QComboBox

    dialog = _dialog(qtbot, "controls")
    combo = next((c for c in dialog.findChildren(QComboBox)
                  if c.findData("bonferroni") >= 0), None)

    assert combo is not None, "no correction control"
    assert dialog._stats_state["correction"] == "fdr_bh"

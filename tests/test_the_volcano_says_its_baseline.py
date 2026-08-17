"""Right-click the volcano and choose what the effects are measured from.

The arithmetic is in :mod:`spacr.baseline` and pinned by
tests/test_an_effect_says_what_it_is_measured_from.py. This file pins that the
choice reaches the plot, that the panel SAYS which baseline is in force, and
that a baseline is offered as a different kind of thing from a re-fit --
moving where zero is drawn on a fit that already happened is not replacing
the fit.
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


def _frame(control_effect=1.5, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"feature": f"fraction:grna[000000_{i}]",
             "coefficient": control_effect + rng.normal(0, .2),
             "p_value": rng.uniform(), "condition": "nc"} for i in range(24)]
    rows += [{"feature": f"fraction:grna[{411000 + i}_1]",
              "coefficient": rng.normal(0, .5),
              "p_value": rng.uniform(), "condition": "other"}
             for i in range(200)]
    return pd.DataFrame(rows)


def _panel(qtbot, frame=None):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(_frame() if frame is None else frame)
    return panel


def _items(plot):
    return ["|" if a.isSeparator() else a.text()
            for a in plot.build_style_menu().actions()]


def test_the_volcano_offers_the_baselines(qtbot):
    items = _items(_panel(qtbot).volcano)

    assert any("non-targeting" in item for item in items), items
    assert any("zero" in item for item in items), items


def test_a_baseline_is_not_a_refit(qtbot):
    """They must be tellable apart: one moves where zero is drawn on a fit
    that has already happened, the other replaces the fit."""
    items = _items(_panel(qtbot).volcano)

    baseline = next(i for i, t in enumerate(items) if "non-targeting" in t)
    refit = next(i for i, t in enumerate(items) if "Re-fit" in t)
    assert "|" in items[baseline:refit], items


def test_the_chosen_one_is_ticked(qtbot):
    """A menu of baselines that does not say which is in force is a menu
    that cannot answer the only question a reader has about the axis."""
    panel = _panel(qtbot)
    panel.set_baseline("controls")

    ticked = {a.text() for a in panel.volcano.build_style_menu().actions()
              if a.isCheckable() and a.isChecked()}
    assert any("non-targeting" in t for t in ticked), ticked


def test_choosing_it_says_so_on_the_panel(qtbot):
    panel = _panel(qtbot)
    panel.set_baseline("controls")

    said = panel.status_text()
    assert "non-targeting controls" in said
    assert "24" in said, "the number of controls it used is not stated"


def test_the_default_is_the_fits_own_zero(qtbot):
    """Nothing is re-expressed unless asked. A panel that silently centred on
    the controls would report effects nobody chose."""
    panel = _panel(qtbot)

    assert panel._baseline == (None, None)


def test_the_points_actually_move(qtbot):
    panel = _panel(qtbot)
    before = np.array(panel.volcano._row_xy and
                      [xy[0] for xy in panel.volcano._row_xy.values()])
    panel.set_baseline("controls")
    after = np.array([xy[0] for xy in panel.volcano._row_xy.values()])

    assert before.size and after.size
    assert not np.allclose(np.sort(before), np.sort(after)), (
        "choosing the control baseline left every point where it was")


def test_the_shift_is_the_control_median(qtbot):
    """Not the mean: `000000_22`, a non-targeting control, is the strongest
    effect in this screen at +4.37, and a mean baseline would shift every
    effect in the screen by it."""
    frame = _frame()
    frame.loc[frame.index[0], "coefficient"] = 4.37
    panel = _panel(qtbot, frame)
    panel.set_baseline("controls")

    controls = frame.loc[frame["condition"] == "nc", "coefficient"]
    moved = np.array([xy[0] for xy in panel.volcano._row_xy.values()])
    expected = frame["coefficient"] - controls.median()
    assert np.allclose(np.sort(moved), np.sort(expected.to_numpy()))


def test_the_run_s_own_table_is_not_shifted(qtbot):
    """The coefficient table sits beside the volcano showing the same rows.
    Shifting in place makes the two disagree with nothing saying why."""
    frame = _frame()
    before = frame["coefficient"].copy()
    panel = _panel(qtbot, frame)
    panel.set_baseline("controls")

    pd.testing.assert_series_equal(panel._frame["coefficient"], before)


def test_a_baseline_that_cannot_be_honoured_says_why(qtbot):
    """Falling back to zero in silence is a user who believes they are
    reading control-relative effects and is not."""
    frame = _frame().drop(columns=["condition"])
    panel = _panel(qtbot, frame)
    panel.set_baseline("controls")

    said = panel.status_text()
    assert "not knowable" in said or "Asked for" in said, said


def test_the_selection_survives_a_baseline_change(qtbot):
    """Same rule as the colouring: the ring the user was reading must not
    vanish and leave them to find their guide again."""
    panel = _panel(qtbot)
    key = panel._frame["feature"].iloc[5]
    panel._select_key(key)
    panel.set_baseline("controls")

    assert panel.volcano._selected_key == key

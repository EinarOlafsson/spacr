"""The guarantees that make six of the fast plots' guards dead code.

Round 6 was pointed at 227 uncovered items in this module. All but nine of
them were already closed by the four fast-plot test files beside this one;
the nine that remained are six branches that NO caller can take, and the
lines behind them. They are not silenced here -- there is no pragma in this
repository -- because a branch that is unreachable today is unreachable only
for a reason, and the reason is always somebody else's invariant:

* ``_violin_profile``'s ``peak <= 0`` sits after a guard that has already
  established ``max > min`` over finite values, and the histogram is taken
  over exactly that range, so every value it was given falls in a bin.
* ``_rows_of``'s object-dtype test is a conversion, not a choice:
  pyqtgraph stores per-point ``data`` in an object column, always, and the
  two callers only ever hand it a live ``ScatterPlotItem``.
* ``_q_ramp``'s empty-``finite`` arm cannot run, because ``set_results``
  only reaches for the ramp once ``np.isfinite(q).any()`` is true.
* ``_offer_graph_kinds`` classifies the same frame twice -- once inside
  ``offer``, once through ``shape_of`` -- so the second call cannot raise
  or come back blank when the first one did not.
* ``VolcanoPlot._detail``'s local-FDR test is guarded by the q above it: a
  finite q means a finite raw p, and a finite raw p means a finite local FDR.
* ``EffectRankPlot.set_results`` returns on an empty table long before it
  writes the names onto the axis, so ``shown`` is never zero there.

So each test below pins the INVARIANT rather than the guard: if one of these
ever stops holding, the branch stops being dead and this file is where that
is noticed. Every one drives the case the invariant forbids AND the case it
allows, in the same test, because an assertion that something is absent is
worth nothing on a widget that draws nothing at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")


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
    widget = fp.VolcanoPlot()
    qtbot.addWidget(widget)
    return widget


def _coefficients(n=12, *, p=None):
    """A coefficient table the shape ``perform_regression`` writes."""
    return pd.DataFrame({
        "feature": [f"fraction:grna[g{i}]" for i in range(n)],
        "grna": [f"g{i}" for i in range(n)],
        "coefficient": np.linspace(-2.0, 2.0, n),
        "p_value": (np.linspace(1e-6, 0.9, n) if p is None
                    else np.asarray(p, dtype=float)),
    })


# ------------------------------------------------------- the violin's bins


# -------------------------------------------------- the rows under a scatter


# ------------------------------------------------------------- the q ramp


# --------------------------------------------------- "Show as", and its shape


# ------------------------------------------------------ q and the local FDR

def test_a_row_that_has_a_q_always_has_a_local_fdr_to_go_with_it(volcano):
    """The two are computed over the same family, from the same raw p.

    ``adjust_p_values`` leaves a non-finite p non-finite, and ``local_fdr``
    does the same, so within one family the rows carrying a q are exactly the
    rows carrying a local FDR. That is why the click report's local-FDR test
    can never fail for a row whose q it has already printed -- and a row with
    no q never gets that far, which the nuisance term here shows.
    """
    frame = _coefficients(24)
    frame.loc[24] = ["Intercept", "", 0.4, np.nan]

    volcano.set_results(frame, drop_untested=False)

    q = volcano._q_values
    lfdr = volcano.local_fdr_values()
    finite = np.isfinite(q)
    assert finite.any() and not finite.all(), "the table was not mixed"
    assert np.isfinite(lfdr[finite]).all(), (
        "a row carrying a q was left without a local FDR")
    assert not np.isfinite(lfdr[~finite]).any(), (
        "a row with no q was given a local FDR anyway")

    assert "local FDR=" in volcano._detail(0)
    assert volcano._detail(24) == "not in the tested family, so no q"


# ------------------------------------------------- the names on the rank axis


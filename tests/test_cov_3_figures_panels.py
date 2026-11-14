"""A panel that cannot be drawn says which column it needed.

Every panel on the result sheet takes the same coefficient table, and the
tables real backends produce differ: a penalised fit has no p-value, a
gene-level export has no per-guide rows, an arbitrary CSV has no effect
column at all. A panel handed one of those must come back undrawn with the
missing column named, because a blank tile on a sheet is indistinguishable
from a panel that found nothing.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import pandas as pd                                            # noqa: E402
import pytest                                                  # noqa: E402

from spacr.figures import panels                               # noqa: E402


@pytest.fixture()
def ax():
    figure = plt.figure()
    try:
        yield figure.add_subplot(111)
    finally:
        plt.close(figure)


def _no_effect_frame():
    """A table that names a p-value but no fitted effect."""
    return pd.DataFrame({"p_value": [0.01, 0.4], "gene": ["A", "B"]})


# ---------------------------------------------------------------------------
# Panels that cannot be drawn
# ---------------------------------------------------------------------------

def test_a_threshold_cannot_be_estimated_without_an_effect_column():
    """The caller prints the reason next to the volcano, so an empty string
    here would leave the sheet claiming a threshold it never computed."""
    rule, threshold = panels.control_threshold(_no_effect_frame())

    assert rule == "no effect column"
    assert threshold is None


@pytest.mark.parametrize("panel_fn,key,needed", [
    (panels.effect_rank, "effect_rank", "coefficient"),
    (panels.effect_distribution, "effect_distribution", "coefficient"),
])
def test_a_panel_with_no_effect_column_names_what_it_needed(
        ax, panel_fn, key, needed):
    """`drawn=False` plus the reason is what turns a blank tile into a
    sentence the user can act on."""
    panel = panel_fn(ax, _no_effect_frame())

    for field in ("caption", "drawn", "reason", "needs", "data", "groups"):
        assert f":param {field}:" in (panels.Panel.__doc__ or "")
    assert panel.drawn is False
    assert panel.key == key
    assert panel.reason == "no effect column"
    assert needed in panel.needs


def test_guide_agreement_needs_per_guide_rows_not_just_effects(ax):
    """A gene-level export has effects but no `feature`, so there is nothing
    to ask whether the guides agree about."""
    frame = pd.DataFrame({"coefficient": [0.4, -0.2], "gene": ["A", "B"]})

    panel = panels.guide_agreement(ax, frame)

    assert panel.drawn is False
    assert panel.reason == "no per-guide coefficients"
    assert "feature" in panel.needs


# ---------------------------------------------------------------------------
# The volcano's labels and highlight
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Asking before drawing
# ---------------------------------------------------------------------------

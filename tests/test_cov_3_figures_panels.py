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
import numpy as np                                             # noqa: E402
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

def _called_frame():
    """Five significant, large-effect coefficients, one of them unnamed."""
    return pd.DataFrame({
        "gene": ["GRA14", "ROP18", float("nan"), "MYR1", "ASP5"],
        "coefficient": [2.0, -2.2, 2.4, 1.8, -1.9],
        "p_value": [1e-8, 1e-9, 1e-10, 1e-7, 1e-6],
    })


def test_a_coefficient_with_no_name_is_not_labelled_nan(ax, monkeypatch):
    """Half the rows of a real coefficient table carry no gene, and a
    volcano that labels them 'nan' puts the same meaningless word on its
    strongest points."""
    frame = _called_frame()
    # Pandas 3 can preserve the float NaN in a nominally string-like Series.
    # Force that cross-version shape even when this test runs on pandas 2.
    monkeypatch.setattr(
        panels, "label_series", lambda _frame: _frame["gene"].astype(object))

    panel = panels.volcano(ax, frame, effect_threshold=None,
                           label_top=5)

    assert panel.drawn is True
    labels = [text.get_text() for text in ax.texts]
    assert "nan" not in [label.lower() for label in labels]
    assert "GRA14" in labels


def test_a_highlighted_coefficient_is_outlined(ax):
    """The highlight is how a user finds their gene on a volcano of
    thousands; silently drawing nothing for a name that IS present would be
    read as 'this gene was not tested'."""
    before = len(ax.collections)

    panel = panels.volcano(ax, _called_frame(), effect_threshold=None,
                           label_top=0, highlight="ROP18")

    assert panel.drawn is True
    outlines = [c for c in ax.collections[before:]
                if len(c.get_offsets()) == 1]
    assert outlines, "no single-point outline was drawn for the highlight"


def test_a_highlight_that_names_nothing_draws_no_outline(ax):
    """The contrast: an outline drawn at an arbitrary point would tell the
    user their gene is somewhere it is not."""
    before = len(ax.collections)

    panels.volcano(ax, _called_frame(), effect_threshold=None, label_top=0,
                   highlight="not_in_this_screen")

    outlines = [c for c in ax.collections[before:]
                if len(c.get_offsets()) == 1]
    assert outlines == []


# ---------------------------------------------------------------------------
# Asking before drawing
# ---------------------------------------------------------------------------

def test_the_sheet_can_be_asked_which_panels_a_table_supports():
    """The sheet asks before it builds, so a table missing a column leaves
    that tile out rather than putting an undrawn one on the page."""
    frame = pd.DataFrame({
        "feature": [f"fraction:grna[g{i}]" for i in range(30)],
        "gene": [f"G{i // 3}" for i in range(30)],
        "coefficient": np.linspace(-2.0, 2.0, 30),
        "p_value": np.linspace(1e-6, 0.9, 30),
    })

    caller_figure = plt.figure()
    figures_before = set(plt.get_fignums())
    try:
        answer = panels.available(frame)

        assert set(answer) == set(panels.SHEET_ORDER)
        assert all(isinstance(value, bool) for value in answer.values())
        assert answer["volcano"] is True
        assert set(plt.get_fignums()) == figures_before, (
            "available() leaked a figure or closed one it did not own")
    finally:
        plt.close(caller_figure)


def test_a_table_with_no_effect_column_supports_no_effect_panel():
    """Every panel that needs the effect answers False, and the answer is
    reached without raising out of the panel that could not draw."""
    answer = panels.available(_no_effect_frame())

    assert answer["volcano"] is False
    assert answer["effect_rank"] is False
    assert answer["effect_distribution"] is False


def test_a_panel_that_raises_is_reported_unavailable_not_propagated(
        monkeypatch):
    """A panel builder that throws must not take the whole sheet down; the
    sheet leaves that tile out and draws the rest."""
    def explode(_ax, _frame):
        raise RuntimeError("this panel is broken")

    monkeypatch.setitem(panels.REGISTRY, "qq", explode)
    frame = pd.DataFrame({
        "gene": ["A", "B", "C"],
        "coefficient": [1.0, -1.0, 0.2],
        "p_value": [0.001, 0.002, 0.5],
    })

    caller_figure = plt.figure()
    figures_before = set(plt.get_fignums())
    try:
        answer = panels.available(frame)

        assert answer["qq"] is False
        assert answer["volcano"] is True, "one broken panel disabled the others"
        assert set(plt.get_fignums()) == figures_before, (
            "available() leaked a figure or closed one it did not own")
    finally:
        plt.close(caller_figure)

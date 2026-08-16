"""Clicking a results row lights up THAT guide's point.

Instruction 119 section B: "this table should be clickable so the user can
click a row and that rows datapoint in the regression graph gets highlighted",
and the warning it carries -- the row and the point must be joined on a KEY,
not on a position:

    "A regression table sorted by effect and a scatter drawn in input order
     are the same points in two orders, and joining them by index highlights
     the wrong guide -- silently, and in exactly the direction a user would
     not question, because SOMETHING lights up."

Which is what the link did. It carried an integer, and the two frames stopped
being the same frame the moment the volcano stopped plotting nuisance terms.

So every test here SORTS THE TABLE DIFFERENTLY FROM THE PLOT ORDER first. A
position-joined link passes none of them; it cannot, because under a sort the
right answer and the index are different numbers.

The key is `feature`, checked against the real screen rather than assumed:
1,213 rows, 1,213 distinct values. `gene` (389 distinct) and `grna` (823) are
not keys and joining on either selects an arbitrary guide of that gene.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


@pytest.fixture()
def results():
    """A screen shaped like the real one: an intercept, guides and genes."""
    rng = np.random.default_rng(7)
    n = 400
    frame = pd.DataFrame({
        "feature": [f"fraction:grna[{i // 4}_{i % 4}]" for i in range(n)],
        "coefficient": rng.normal(size=n),
        "p_value": rng.uniform(size=n),
        "gene": [f"{i // 4}" for i in range(n)],
        "condition": rng.choice(["nc", "pc", "other"], n, p=[.05, .05, .9]),
    })
    # The nuisance term the fit writes and the volcano must not plot.
    intercept = pd.DataFrame([{
        "feature": "Intercept", "coefficient": 0.19, "p_value": 3.1e-46,
        "gene": None, "condition": "other"}])
    return pd.concat([intercept, frame], ignore_index=True)


@pytest.fixture()
def panel(qtbot, results):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)
    widget.set_frame(results, source="results.csv")
    # SORT THE TABLE. This is the whole point of the file: after this, a row's
    # screen position and its frame position are different numbers, and a link
    # that carries either one of them alone is wrong about the other.
    widget.table.table.sortItems(1)          # by coefficient
    return widget


def _selected_feature(panel):
    items = panel.table.table.selectedItems()
    if not items:
        return None
    return panel.table.table.item(items[0].row(), 0).text()


# --------------------------------------------------------------------------- #
#  Row -> point
# --------------------------------------------------------------------------- #

def test_selecting_a_sorted_row_rings_that_guides_point(panel, results):
    """The feature the instruction asked for, under the condition that breaks
    the naive version of it."""
    table = panel.table.table
    table.selectRow(3)                       # third-smallest coefficient
    shown = table.item(3, 0).text()

    assert panel.volcano._selected_key == shown, (
        f"row 3 of the sorted table is {shown} but the plot rang "
        f"{panel.volcano._selected_key}")
    assert panel.volcano._highlight is not None, "nothing was drawn"


def test_the_ring_is_at_that_guides_coordinates(panel, results):
    """Not merely present -- in the right place.

    A highlight drawn at the wrong point is the failure this file exists to
    catch, and 'an item exists' does not catch it.
    """
    table = panel.table.table
    table.selectRow(11)
    shown = table.item(11, 0).text()

    row = results.index[results["feature"] == shown][0]
    expected_x = float(results["coefficient"].iloc[row])
    expected_y = -np.log10(float(results["p_value"].iloc[row]))

    spots = panel.volcano._highlight.getData()
    assert float(spots[0][0]) == pytest.approx(expected_x, abs=1e-9)
    assert float(spots[1][0]) == pytest.approx(expected_y, abs=1e-6)


# --------------------------------------------------------------------------- #
#  Point -> row
# --------------------------------------------------------------------------- #

def test_clicking_a_point_selects_that_guides_row_not_that_position(panel,
                                                                    results):
    """The other direction, and the one where the frames genuinely differ:
    the volcano dropped the Intercept, so every plot position is one less
    than the table position it used to mean."""
    plotted = panel.volcano._keys
    assert "Intercept" not in plotted, "the volcano is plotting a nuisance term"

    wanted = plotted[30]
    panel.volcano.key_selected.emit(wanted)

    assert _selected_feature(panel) == wanted


def test_a_position_join_would_have_picked_a_different_guide(panel, results):
    """Proves the tests above are actually testing something.

    If the old integer link were still in place it would select the row at
    plot position 30, which after the sort and the dropped intercept is a
    different guide. Naming it here means a future 'simplification' back to
    positions fails loudly instead of silently.
    """
    table = panel.table.table
    wanted = panel.volcano._keys[30]
    by_position = table.item(30, 0).text()

    assert by_position != wanted, (
        "the fixture no longer distinguishes the two joins -- change the sort")


# --------------------------------------------------------------------------- #
#  It survives a redraw
# --------------------------------------------------------------------------- #

def test_the_selection_survives_a_settings_change(panel):
    """"Highlighting should also survive a re-draw: change a setting, and the
    selected point stays selected." A redraw clears the scene, so the marker
    has to be put back or the user loses their place on every recolour."""
    panel.table.table.selectRow(5)
    chosen = panel.volcano._selected_key
    assert chosen

    panel._redraw_volcano()

    assert panel.volcano._selected_key == chosen
    assert panel.volcano._highlight is not None, (
        "the redraw dropped the selection")


def test_a_new_table_drops_the_old_selection(panel, results):
    """The other half of the same rule: a different run is a different
    experiment, and carrying a ring over would mark a point that means
    something else now."""
    panel.table.table.selectRow(5)
    assert panel.volcano._selected_key

    panel.set_frame(results.iloc[:50].copy(), source="other.csv")

    assert panel._selected_key is None
    assert panel.volcano._highlight is None


# --------------------------------------------------------------------------- #
#  Honest about a miss
# --------------------------------------------------------------------------- #

def test_selecting_the_intercept_says_it_is_not_on_the_plot(panel):
    """It is in the table and deliberately not in the plot. Saying so beats
    ringing whichever point happens to be nearest."""
    assert panel.table.select_key("Intercept")

    assert panel.volcano._highlight is None
    assert "not on this plot" in panel.volcano._status.text()


def test_a_key_that_is_nowhere_is_a_false_answer_not_a_crash(panel):
    assert panel.volcano.highlight_key("no_such_guide") is False
    assert panel.table.select_key("no_such_guide") is False


def test_a_filtered_out_row_is_unhidden_to_select_it(panel):
    """Clicking a point whose row the filter box excludes must not look like
    a dead click."""
    table = panel.table
    wanted = panel.volcano._keys[12]
    table._filter.setText("zzzz_matches_nothing")
    assert all(table.table.isRowHidden(r) for r in range(table.table.rowCount()))

    assert table.select_key(wanted)
    assert _selected_feature(panel) == wanted


# --------------------------------------------------------------------------- #
#  Duplicate keys are refused as keys, not guessed at
# --------------------------------------------------------------------------- #

def test_a_non_unique_column_is_not_used_as_a_key(qtbot, results):
    """`gene` repeats across a gene's four guides. Selecting on it would ring
    an arbitrary one of them and look correct."""
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    assert not results["gene"].is_unique
    assert RegressionResultsPanel._key_column(results) == "feature"

    without = results.drop(columns=["feature"])
    assert RegressionResultsPanel._key_column(without) is None


def test_a_duplicated_feature_column_is_refused(qtbot, results):
    """If `feature` itself ever repeats it stops being a key, and the panel
    must notice rather than trust the name."""
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    duped = results.copy()
    duped.loc[duped.index[-1], "feature"] = duped["feature"].iloc[0]

    assert RegressionResultsPanel._key_column(duped) is None

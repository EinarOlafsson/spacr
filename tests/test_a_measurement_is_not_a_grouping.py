"""Pressing Rank produced thousands of lines and no ranking.

    0.735573 vs 0.778142: these groups have fewer than 2 usable
    observations and cannot be tested: ['0.735573', '0.778142']
    0.735573 vs 0.779132: ...
    (thousands more)

The group labels are FLOATING-POINT MEASUREMENTS. A column of measurements
had been handed in as the grouping, so every "group" held one row, every
pair was untestable, and the pair count is quadratic in the number of rows.

Two defects, and both are fixed here:

  a. a continuous column is not a grouping. It is refused at the door and
     the column is NAMED, rather than discovered one impossible pair at a
     time;
  b. even with a real grouping, one line per untestable pair is not a
     report -- it is the same sentence a quadratic number of times, and it
     buries whatever else the run said. Said once, with a count.

Nothing is hidden either way: every pair is in the results table, marked
'not testable'.
"""
from __future__ import annotations

import contextlib
import io

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from spacr.plot import create_grouped_plot


def _draw(frame, grouping):
    """Draw and return everything printed while drawing."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        create_grouped_plot(df=frame, grouping_column=grouping,
                            data_column="value", graph_type="bar", save=False)
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def test_a_column_of_measurements_is_refused_by_name():
    """The thing that actually happened, and the message that helps."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"measurement": np.round(rng.random(60), 6),
                          "value": rng.random(60)})

    with pytest.raises(ValueError, match="is not a grouping"):
        create_grouped_plot(df=frame, grouping_column="measurement",
                            data_column="value", graph_type="bar", save=False)


def test_the_refusal_names_the_column_and_says_what_to_do():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"cell_area": np.round(rng.random(40), 6),
                          "value": rng.random(40)})

    with pytest.raises(ValueError) as refused:
        create_grouped_plot(df=frame, grouping_column="cell_area",
                            data_column="value", graph_type="bar", save=False)

    said = str(refused.value)
    assert "cell_area" in said
    assert "category" in said, "it must say what a grouping IS"


def test_a_real_grouping_with_many_levels_is_not_refused():
    """A 30-well plate is a grouping, however many levels it has."""
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"well": np.repeat([f"w{i}" for i in range(30)], 4),
                          "value": rng.random(120)})

    said = _draw(frame, "well")

    # Not refused, and every pair was testable: four wells per group is
    # enough, and 30 levels is a plate, not a continuous column.
    assert "is not a grouping" not in said
    assert "could not be tested" not in said


def test_one_thin_group_produces_one_line_not_one_per_pair():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 10 + ["c"],
                          "value": rng.random(21)})

    said = _draw(frame, "g")

    lines = [l for l in said.splitlines() if "could not be tested" in l]
    assert len(lines) == 1
    assert "2 of 3 comparison(s)" in lines[0]


def test_the_summary_names_only_the_thin_group():
    """A pair fails because ONE side is small; naming both blames a
    healthy group."""
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 10 + ["c"],
                          "value": rng.random(21)})

    said = _draw(frame, "g")
    line = [l for l in said.splitlines() if "could not be tested" in l][0]

    assert "1 group(s)" in line
    assert line.rstrip().count("a,") == 0, "a healthy group was blamed"


def test_a_clean_grouping_says_nothing_about_untestable_pairs():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 10,
                          "value": rng.random(20)})

    assert "could not be tested" not in _draw(frame, "g")


def test_the_untestable_pairs_are_still_in_the_table():
    """Quieter, not hidden."""
    rng = np.random.default_rng(6)
    frame = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 10 + ["c"],
                          "value": rng.random(21)})

    _figure, results = create_grouped_plot(
        df=frame, grouping_column="g", data_column="value",
        graph_type="bar", save=False)

    assert (results["Test Name"] == "not testable").sum() == 2

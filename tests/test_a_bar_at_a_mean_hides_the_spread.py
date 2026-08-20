"""Instruction 139 B — a boxplot with jitter, not a bar with jitter.

    "the bargraphs with jutter plot backgrounds should be boxplots with
     jutter"

THIS IS A STATISTICAL CORRECTION AND NOT A PREFERENCE, which is why it is
worth a test rather than a style rule. A bar drawn at a mean with points
behind it shows ONE number and hides the shape of the data: two groups with
the same mean and completely different spreads draw THE SAME BAR. The box
shows the median, the quartiles and the whiskers, so the reader sees the
distribution the points already imply — and the jitter stays, because the box
summarises and the points are the evidence.

The first test below demonstrates the defect rather than asserting the fix, so
that anyone changing the default back can see what it costs.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

pytest.importorskip("matplotlib")


def _two_groups_same_mean():
    """Same mean, wildly different spread. A bar cannot tell them apart."""
    rng = np.random.default_rng(0)
    tight = rng.normal(10.0, 0.2, 60)
    wide = rng.normal(10.0, 4.0, 60)
    return tight - tight.mean() + 10.0, wide - wide.mean() + 10.0


def test_a_bar_at_the_mean_draws_the_same_picture_for_both():
    """The defect, demonstrated. This is why the default moved."""
    tight, wide = _two_groups_same_mean()

    assert np.isclose(tight.mean(), wide.mean())
    # A bar is the mean and an error bar is often the SEM, which shrinks with
    # n and says nothing about spread either.
    assert abs(tight.std() - wide.std()) > 3.0, "the fixture is not a fixture"


def test_the_box_tells_them_apart():
    tight, wide = _two_groups_same_mean()

    def iqr(values):
        return float(np.percentile(values, 75) - np.percentile(values, 25))

    assert iqr(wide) > iqr(tight) * 5


@pytest.mark.parametrize("target", ["create_grouped_plot", "spacrGraph"])
def test_the_default_is_the_box(target):
    import spacr.plot as plot

    obj = getattr(plot, target)
    signature = inspect.signature(obj if not isinstance(obj, type)
                                  else obj.__init__)
    assert signature.parameters["graph_type"].default == "jitter_box"


def test_bar_is_still_available_for_anyone_who_wants_it():
    """A default is not a removal."""
    import spacr.plot as plot

    source = inspect.getsource(plot.spacrGraph)
    assert "'bar'" in source or '"bar"' in source


def test_the_reason_is_written_where_the_default_is():
    """A default nobody can find a reason for gets changed back."""
    import spacr.plot as plot

    doc = inspect.getdoc(plot.create_grouped_plot) or ""
    assert "STATISTICAL CORRECTION" in doc
    assert "same mean" in doc
    # The jitter is kept, and the docstring says why: the box summarises and
    # the points are the evidence.
    assert "the jitter stays" in doc
    assert "points are the evidence" in doc


def test_every_module_default_agrees_with_the_function_default():
    """Three modules set this key; three answers is three different figures."""
    import re
    from pathlib import Path

    settings = Path(inspect.getfile(
        __import__("spacr.settings", fromlist=["settings"])))
    defaults = re.findall(r"setdefault\(\s*'graph_type'\s*,\s*'([a-z_]+)'",
                          settings.read_text())
    assert defaults, "no module sets graph_type any more"
    assert set(defaults) == {"jitter_box"}, (
        f"modules disagree about the default graph type: {sorted(set(defaults))}")

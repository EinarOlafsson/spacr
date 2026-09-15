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


#: The settings functions that hand `create_grouped_plot` its graph type.
GRAPH_TYPE_DEFAULTERS = (
    "set_default_plot_data_from_db",
    "set_graph_importance_defaults",
    "get_plot_data_from_csv_default_settings",
)


def _resolved_graph_types(monkeypatch, preference):
    """What the three settings functions put in `graph_type`.

    :param monkeypatch: the pytest fixture, used to state the preference.
    :param preference: the graph-types mark the user has chosen, or ``""``
        for a user who has chosen nothing.
    :returns: a mapping from settings-function name to the resolved value.

    THE PREFERENCE IS STATED RATHER THAN INHERITED. Since item 293 these
    defaults consult the saved preference store, so calling them plainly
    would assert on whatever THIS MACHINE has saved -- green on a fresh
    runner and red for a developer who once picked bars. Replacing
    `chosen_for` also keeps `spacr.qt.preferences`, and PySide6 behind it,
    out of a process that is not testing Qt.
    """
    from spacr import graph_types, settings as settings_module

    monkeypatch.setattr(graph_types, "chosen_for", lambda shape: preference)
    return {
        name: getattr(settings_module, name)({})["graph_type"]
        for name in GRAPH_TYPE_DEFAULTERS
    }


def test_a_user_who_chose_nothing_gets_the_function_default(monkeypatch):
    """Three modules set this key; three answers is three different figures.

    THIS ASKS THE FUNCTIONS RATHER THAN READING THE FILE. It used to grep
    settings.py for `setdefault('graph_type', '...')` and collect the string
    literals, which stopped working the moment those three sites started
    COMPUTING the value instead of spelling it -- the defaults were still
    identical and still correct, and the test said "no module sets graph_type
    any more".

    A source grep can only see defaults that are written down. It cannot see
    a computed one at all, and it cannot see a written-down one that is
    right in the file and wrong by the time the settings dict is built. What
    matters to the user is the value in the dict, so that is what is read.
    """
    from spacr import plot

    function_default = inspect.signature(
        plot.create_grouped_plot).parameters["graph_type"].default

    resolved = _resolved_graph_types(monkeypatch, "")

    assert set(resolved.values()) == {function_default}, (
        f"modules disagree with create_grouped_plot's own default "
        f"{function_default!r}: {resolved}")


def test_a_user_who_chose_bars_gets_bars_from_all_three(monkeypatch):
    """The three agree with EACH OTHER once a preference exists, too.

    This is the half that proves the test above is measuring anything. If
    `chosen_for` were not actually reached -- a patch on the wrong name, a
    value read at import time and cached -- the test above would still pass,
    because it asserts on exactly the value the code would produce with the
    preference ignored. A clean result is only evidence when the probe
    reached the thing being measured.

    It also pins item 293's behaviour at the settings layer: a saved choice
    reaches the defaults, translated out of the graph-types vocabulary
    (`bar_jitter`) into the one `create_grouped_plot` speaks (`jitter_bar`).
    """
    resolved = _resolved_graph_types(monkeypatch, "bar_jitter")

    assert set(resolved.values()) == {"jitter_bar"}, resolved

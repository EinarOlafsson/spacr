"""Right-click a grouped live plot and change what its groups are drawn as.

Instruction 128 F, in the maintainer's words:

    "for the live plots id like to be able to right click and change the plot
     type like show guide support as a violin, box, bar, jitter plot"

The right-click menu already restyles, offers baselines and offers a re-fit.
This adds the MARK, and it follows the same `offer_*` shape as those, because
a fifth way of building the same menu is how the entries start disagreeing
about where they live.

THE HARD PART IS NOT DRAWING FIVE MARKS. It is that four of them are wrong for
the data these panels usually hold. The house rule: with eight or fewer points
per group the individual points ARE the figure, a box plot hides n, and a
violin draws a density through points that never described one. The menu could
have hidden them -- but a menu that hides an option the user asked for by name
cannot explain why. So all five are offered, and the panel SAYS what the chosen
one hides, measured on the groups actually on screen.
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


@pytest.fixture()
def controls(qtbot):
    from spacr.qt.widgets.fast_plots import ControlSeparation

    widget = ControlSeparation()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture()
def agreement(qtbot):
    from spacr.qt.widgets.fast_plots import GuideAgreementPlot

    widget = GuideAgreementPlot()
    qtbot.addWidget(widget)
    return widget


def _groups(sizes=(40, 35, 400), seed=0):
    rng = np.random.default_rng(seed)
    centres = (-1.0, 1.0, 0.0)
    names = ("negative", "positive", "other")
    return {name: rng.normal(centre, 0.3, size)
            for name, centre, size in zip(names, centres, sizes)}


def _keys(groups):
    return {name: [f"{name}_{i}" for i in range(len(values))]
            for name, values in groups.items()}


def _support(n=120, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feature": [f"gene_fraction:gene[{i}]" for i in range(n)],
        "n_guides": rng.integers(1, 5, n),
        "concordance": rng.random(n),
        "single_guide": rng.random(n) < 0.2,
        "n_same_direction": rng.integers(1, 3, n),
        "gene_p": rng.random(n),
    })


def _menu_entries(plot):
    return [action.text() for action in plot.build_style_menu().actions()]


# --------------------------------------------------------------------------- #
#  The gesture
# --------------------------------------------------------------------------- #

def test_the_right_click_menu_offers_every_mark_by_name(controls):
    """Named in the request: "a violin, box, bar, jitter plot"."""
    controls.set_groups(_groups())

    entries = _menu_entries(controls)

    assert "Draw as" in entries, f"no mark section on the menu: {entries}"
    for wanted in ("Jittered points", "Box plot", "Violin plot", "Bar chart"):
        assert wanted in entries, f"{wanted} is not offered: {entries}"


def test_the_menu_ticks_the_mark_that_is_actually_drawn(controls):
    """A menu that does not show the current state is a menu the user has to
    guess at, and they will pick the one already selected to find out."""
    controls.set_groups(_groups())
    controls.set_mark("violin")

    ticked = [action.text()
              for action in controls.build_style_menu().actions()
              if action.isCheckable() and action.isChecked()]

    assert "Violin plot" in ticked
    assert "Box plot" not in ticked


def test_picking_a_mark_off_the_menu_redraws_the_plot(controls):
    """Driving the action proves the menu is wired, not merely populated."""
    controls.set_groups(_groups())
    assert controls.mark() == "jitter"

    action = next(a for a in controls.build_style_menu().actions()
                  if a.text() == "Bar chart")
    action.trigger()

    assert controls.mark() == "bar"


def test_a_plot_with_no_groups_is_not_offered_a_violin(qtbot):
    """A volcano's x is an effect size. "Draw this as a violin" is not a
    question that has an answer there, and offering it would produce either a
    dead entry or a wrong picture."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    volcano = VolcanoPlot()
    qtbot.addWidget(volcano)
    volcano.set_results(pd.DataFrame({
        "feature": [f"fraction:grna[{i}_1]" for i in range(50)],
        "coefficient": np.linspace(-2, 2, 50),
        "p_value": np.linspace(0.001, 0.9, 50),
    }))

    assert "Draw as" not in _menu_entries(volcano)


# --------------------------------------------------------------------------- #
#  Every mark actually draws
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kind", ["points", "jitter", "box", "violin", "bar"])
def test_every_offered_mark_draws_something_on_the_controls(controls, kind):
    groups = _groups()
    controls.set_groups(groups, keys=_keys(groups))

    controls.set_mark(kind)

    assert len(controls.plot.plotItem.items) >= len(groups), (
        f"{kind} left the plot with nothing on it")


@pytest.mark.parametrize("kind", ["points", "jitter", "box", "violin", "bar"])
def test_every_offered_mark_draws_something_on_guide_support(agreement, kind):
    """The plot named in the request."""
    agreement.set_support(_support())

    agreement.set_mark(kind)

    assert len(agreement.plot.plotItem.items) > 1, (
        f"{kind} left the guide-support plot with nothing on it")


def test_switching_marks_shows_the_same_observations_each_time(controls):
    """The failure mode of recomputing per mark type: a bar of one population
    and points of another, with nothing saying they differ."""
    groups = _groups()
    controls.set_groups(groups, keys=_keys(groups))

    for kind in ("box", "violin", "bar", "points", "jitter"):
        controls.set_mark(kind)
        assert "negative n=40" in controls._status.text()
        assert "positive n=35" in controls._status.text()
        assert "other n=400" in controls._status.text()


def test_a_mark_this_module_cannot_draw_is_refused_loudly(controls):
    """A silent fallback would make a typo look like a working option."""
    controls.set_groups(_groups())

    with pytest.raises(ValueError, match="unknown mark"):
        controls.set_mark("pie")


# --------------------------------------------------------------------------- #
#  The panel says when the mark misleads for THIS data
# --------------------------------------------------------------------------- #

def test_a_bar_over_four_points_says_it_hides_them(controls):
    """The house rule as a sentence the user reads, not a rule that silently
    refuses. n <= 8 per group is individual points, never a bar."""
    groups = _groups(sizes=(4, 5, 6))
    controls.set_groups(groups, keys=_keys(groups))

    controls.set_mark("bar")

    said = controls._status.text()
    assert "hides every observation" in said, said
    assert "the smallest 4" in said, said


def test_a_box_over_few_points_says_it_hides_n(controls):
    controls.set_groups(_groups(sizes=(4, 5, 6)))

    controls.set_mark("box")

    assert "A box plot hides n" in controls._status.text()


def test_a_violin_over_few_points_says_the_density_is_not_there(controls):
    controls.set_groups(_groups(sizes=(4, 5, 6)))

    controls.set_mark("violin")

    said = controls._status.text()
    assert "draws a density that is not there" in said, said


def test_points_and_jitter_are_never_accused_of_hiding_anything(controls):
    """They show every observation. A warning on the honest mark is noise, and
    noise is what makes the real warnings unreadable."""
    controls.set_groups(_groups(sizes=(3, 3)))

    for kind in ("points", "jitter"):
        controls.set_mark(kind)
        said = controls._status.text()
        assert "hides" not in said and "not there" not in said, said


def test_a_mark_that_does_not_mislead_for_this_n_is_not_warned_about(controls):
    """The advice is measured on the data in front of it. Four hundred wells
    per group is exactly what a box plot is for, and a panel that nagged about
    every box would be one nobody reads."""
    groups = _groups(sizes=(40, 35, 400))
    controls.set_groups(groups)

    controls.set_mark("box")

    said = controls._status.text()
    assert "hides n" not in said, said


def test_the_advice_counts_how_many_groups_are_too_small(controls):
    """One thin group among many is a different report from all of them being
    thin, and the user needs to know which group to look at."""
    from spacr.qt.widgets.fast_plots import mark_advice

    assert "the smallest group has 3" in mark_advice("bar", [3, 40, 400])
    assert "2 of 3 groups have 8 or fewer" in mark_advice("bar", [3, 5, 400])
    assert mark_advice("bar", [40, 35, 400]) == ""
    assert mark_advice("points", [1, 1]) == ""


# --------------------------------------------------------------------------- #
#  Clicking a point is part of what a mark costs
# --------------------------------------------------------------------------- #

def test_a_bar_says_that_nothing_on_it_can_be_clicked(controls):
    """Every one of these plots selects a coefficient when a point is clicked.
    A bar stands for four hundred rows and cannot honestly pick one, so the
    ability goes -- and the panel says where it went."""
    groups = _groups()
    controls.set_groups(groups, keys=_keys(groups))
    assert "Click a point for its coefficient" in controls._status.text()

    controls.set_mark("bar")

    said = controls._status.text()
    assert "Click a point for its coefficient" not in said
    assert "nothing on it can be clicked" in said, said


def test_a_box_keeps_its_outliers_clickable_and_says_only_those(controls):
    """The rows a reader of a box plot actually wants to name are the ones
    outside the whiskers, and those ARE individual observations."""
    groups = _groups()
    controls.set_groups(groups, keys=_keys(groups))

    controls.set_mark("box")

    said = controls._status.text()
    assert "Only the outliers" in said, said

    scatters = [item for item in controls.plot.listDataItems()
                if hasattr(item, "points") and len(item.points())]
    assert scatters, "a box plot drew no outlier points at all"
    picked = []
    controls.key_selected.connect(picked.append)
    point = scatters[0].points()[0]
    scatters[0].sigClicked.emit(scatters[0], [point], None)

    assert picked, "clicking an outlier selected no coefficient"
    assert picked[0] in sum(_keys(groups).values(), []), (
        f"an outlier reported {picked[0]!r}, which is not one of the guides")


def test_going_back_to_points_restores_clicking(controls):
    """The user has to be able to undo the trade, and the sentence tells them
    how."""
    groups = _groups()
    controls.set_groups(groups, keys=_keys(groups))
    controls.set_mark("violin")

    controls.set_mark("jitter")

    assert "Click a point for its coefficient" in controls._status.text()


# --------------------------------------------------------------------------- #
#  The default picture does not move
# --------------------------------------------------------------------------- #

def test_the_default_mark_is_what_these_panels_already_drew(controls,
                                                            agreement):
    """A default that changed the picture on upgrade would be this feature
    breaking the two plots it was added to."""
    assert controls.mark() == "jitter"
    assert agreement.mark() == "jitter"


def test_the_control_panel_s_summary_line_stays_the_median(controls):
    """Its status quotes medians and the reader compares those two lines. A
    line drawn at the MEAN beside a number that says median is worse than no
    line at all."""
    groups = {"negative": np.array([0.0, 0.0, 0.0, 12.0])}
    controls.set_groups(groups)

    controls.set_mark("points")

    assert "median=0" in controls._status.text()
    lines = [item for item in controls.plot.plotItem.items
             if hasattr(item, "getData") and not hasattr(item, "points")]
    drawn = [item.getData()[1] for item in lines if item.getData()[1] is not None]
    assert any(np.allclose(y, 0.0) for y in drawn), (
        f"no summary line at the median; drew {drawn}")


def test_guide_support_keeps_its_single_guide_colouring_as_points(agreement):
    """The house rule: everything grey except what the sentence is about, and
    the sentence is "these genes rest on a single guide". A box plot holds
    both kinds at once and cannot carry it -- which is why the point marks
    keep their own path rather than one path drawing a compromise."""
    from spacr.qt.widgets.fast_plots import GuideAgreementPlot

    agreement.set_support(_support())

    assert agreement.mark() == "jitter"
    assert len(agreement.plot.listDataItems()) == 2, (
        "the corroborated and single-guide genes are not drawn apart")
    agreement.set_mark("box")
    assert "rest on a single guide" in agreement._status.text(), (
        "the panel stopped saying its own sentence when the mark changed")

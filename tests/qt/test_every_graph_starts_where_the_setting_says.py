"""Instruction 293 — every graph, not only Regression, starts where the
DEFAULT GRAPH TYPE setting says.

    "Today Regression has most of this: it draws bar-with-jitter and the
     user can right-click to change it. The right-click menu is right; the
     starting point is not. The setting must dictate what is drawn from the
     start. THE SAME BEHAVIOUR FOR EVERY GRAPH IN spaCR."

`tests/test_the_default_graph_type_starts_every_graph.py` tests the
mechanism in `spacr.graph_types`. This file tests the WIDGETS: that a live
regression panel, the guide-support plot and the Measurement Compare panel
each open on the saved choice, that right-click still changes one
afterwards, and that a saved choice the data or the panel cannot honour
falls back somewhere named and says so.

THE STORE IS THE REAL ONE. `tests/conftest.py` gives every test its own
QSettings directory, so these go through `set_default_graph_type` rather
than a stand-in -- which is the half a monkeypatched reader cannot prove:
that what the settings dialog writes is what a plot reads back.
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

SHAPE = "categorical_continuous"


@pytest.fixture
def prefer(qapp):
    """Save a DEFAULT GRAPH TYPE, the way the settings dialog does."""
    from spacr.qt import preferences

    def choose(kind, shape=SHAPE):
        preferences.set_default_graph_type(shape, kind)
        assert preferences.get_default_graph_type(shape) == kind
    return choose


def _groups(sizes=(40, 35, 400), seed=0):
    rng = np.random.default_rng(seed)
    names = ("negative", "positive", "other")
    return {name: rng.normal(centre, 0.3, size)
            for name, centre, size in zip(names, (-1.0, 1.0, 0.0), sizes)}


def _support(n=120, seed=1, guides=(1, 5)):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feature": [f"gene_fraction:gene[{i}]" for i in range(n)],
        "n_guides": rng.integers(guides[0], guides[1], n),
        "concordance": rng.random(n),
        "single_guide": rng.random(n) < 0.2,
        "n_same_direction": rng.integers(1, 3, n),
        "gene_p": rng.random(n),
    })


def _controls(qtbot):
    from spacr.qt.widgets.fast_plots import ControlSeparation

    widget = ControlSeparation()
    qtbot.addWidget(widget)
    return widget


def _agreement(qtbot):
    from spacr.qt.widgets.fast_plots import GuideAgreementPlot

    widget = GuideAgreementPlot()
    qtbot.addWidget(widget)
    return widget


# --------------------------------------------------------------------------- #
#  The two vocabularies have to keep saying the same thing
# --------------------------------------------------------------------------- #

class TestTheTablesAgree:
    """The translation lives in two files, so it is checked in one test."""

    def test_the_mark_table_is_the_same_table_in_both_modules(self):
        from spacr.graph_types import MARKS
        from spacr.qt.widgets.grouped_plot import MARKS as GROUPED

        assert MARKS == GROUPED

    def test_every_mark_a_graph_type_names_is_a_mark_that_exists(self):
        from spacr.graph_types import MARKS
        from spacr.qt.widgets.fast_plots import MARK_TYPES

        assert set(MARKS.values()) == {name for name, _label in MARK_TYPES}

    def test_the_house_rule_is_one_number_in_both_modules(self):
        from spacr.graph_types import MIN_N_FOR_DISTRIBUTION as HEADLESS
        from spacr.qt.widgets.fast_plots import MIN_N_FOR_DISTRIBUTION as LIVE

        assert HEADLESS == LIVE


# --------------------------------------------------------------------------- #
#  The live regression panels
# --------------------------------------------------------------------------- #

class TestALivePanelOpensOnTheChoice:
    """Control separation and guide support are the panels the request calls
    Regression's graphs."""

    def test_with_nothing_chosen_the_panel_keeps_its_own_mark(self, qtbot):
        controls = _controls(qtbot)
        controls.set_groups(_groups())
        assert controls.mark() == "jitter"

    @pytest.mark.parametrize("chosen,mark", [
        ("violin", "violin"),
        ("box", "box"),
        ("bar", "bar"),
        ("bar_jitter", "jitter_bar"),
        ("box_jitter", "jitter_box"),
    ])
    def test_the_saved_choice_is_what_is_drawn_first(self, qtbot, prefer,
                                                     chosen, mark):
        prefer(chosen)

        controls = _controls(qtbot)
        controls.set_groups(_groups())

        assert controls.mark() == mark

    def test_it_is_the_picture_that_changes_not_only_the_name(self, qtbot,
                                                              prefer):
        """A mark the panel reports but never draws is the same bug in a
        different place, so this reads the items on the scene: a violin is
        three curves, a jitter is three scatters."""
        plain = _controls(qtbot)
        plain.set_groups(_groups())
        assert {type(i).__name__ for i in plain.plot.plotItem.items} >= \
            {"ScatterPlotItem"}

        prefer("violin")
        chosen = _controls(qtbot)
        drawn = chosen.set_groups(_groups())

        drew = [type(i).__name__ for i in chosen.plot.plotItem.items]
        assert drawn == 40 + 35 + 400
        assert drew.count("PlotCurveItem") == 3
        assert "ScatterPlotItem" not in drew

    def test_the_menu_ticks_what_the_setting_chose(self, qtbot, prefer):
        from spacr.qt.widgets.fast_plots import menu_entries

        prefer("violin")
        controls = _controls(qtbot)
        controls.set_groups(_groups())

        ticked = [a.text() for a in menu_entries(controls.build_style_menu())
                  if a.isCheckable() and a.isChecked()]

        assert "Violin plot" in ticked
        assert "Jittered points" not in ticked

    def test_guide_support_starts_there_too(self, qtbot, prefer):
        """"The same behaviour for every graph", so the other live panel is
        not a special case of the first one."""
        prefer("box")
        agreement = _agreement(qtbot)

        agreement.set_support(_support())

        assert agreement.mark() == "box"

    def test_right_click_still_changes_it_afterwards(self, qtbot, prefer):
        prefer("violin")
        controls = _controls(qtbot)
        controls.set_groups(_groups())

        assert controls.set_mark("bar") is True

        assert controls.mark() == "bar"

    def test_what_the_user_picked_survives_the_next_load(self, qtbot, prefer):
        """New data must not put the setting back over a choice made since."""
        prefer("violin")
        controls = _controls(qtbot)
        controls.set_groups(_groups())
        controls.set_mark("bar")

        controls.set_groups(_groups(seed=3))

        assert controls.mark() == "bar"

    def test_a_mark_picked_before_the_data_arrives_is_not_overruled(
            self, qtbot, prefer):
        """The panel exists before its data does, so a user can right-click
        one of these empty and then load a run into it. The size rule is
        allowed to talk the SETTING out of a violin; it is not allowed to
        talk the user out of one."""
        prefer("violin")
        controls = _controls(qtbot)
        assert controls.mark() == "violin"

        controls.set_mark("bar")
        controls.set_groups(_groups(sizes=(3, 3, 4)))

        assert controls.mark() == "bar"


# --------------------------------------------------------------------------- #
#  A choice the data cannot support
# --------------------------------------------------------------------------- #

class TestTheFallbackIsSaidOutLoud:
    """"A violin of two points" -- the fallback is explicit, not silent."""

    def test_a_violin_over_three_point_groups_draws_the_points(self, qtbot,
                                                               prefer):
        prefer("violin")
        controls = _controls(qtbot)

        controls.set_groups(_groups(sizes=(3, 3, 4)))

        assert controls.mark() == "jitter"

    def test_the_plot_says_why_it_is_not_the_violin(self, qtbot, prefer):
        prefer("violin")
        controls = _controls(qtbot)

        controls.set_groups(_groups(sizes=(3, 3, 4)))

        said = controls._status.text()
        assert "Violin" in said
        assert "Jitter" in said
        assert "Right-click" in said

    def test_the_same_choice_over_real_groups_is_honoured(self, qtbot,
                                                          prefer):
        prefer("violin")
        controls = _controls(qtbot)

        controls.set_groups(_groups())

        assert controls.mark() == "violin"
        assert "Right-click to draw" not in controls._status.text()

    def test_asking_for_it_by_hand_still_draws_it(self, qtbot, prefer):
        """The fallback governs the STARTING point. A user who right-clicks
        a violin onto four points has asked for that one, and gets it with
        the advice the mark already carried."""
        prefer("violin")
        controls = _controls(qtbot)
        controls.set_groups(_groups(sizes=(3, 3, 4)))

        controls.set_mark("violin")

        assert controls.mark() == "violin"
        assert "Right-click to draw" not in controls._status.text()
        assert "too few to have a shape" in controls._status.text()

    def test_a_choice_that_keeps_its_points_is_never_talked_out_of(
            self, qtbot, prefer):
        """A box WITH the observations on it shows the reader the n."""
        prefer("box_jitter")
        controls = _controls(qtbot)

        controls.set_groups(_groups(sizes=(3, 3, 4)))

        assert controls.mark() == "jitter_box"

    def test_the_other_live_panel_is_measured_too(self, qtbot, prefer):
        """The size rule is not a Control-separation special case. Guide
        support counts genes per distinct guide count, and those groups can
        be as thin as any other -- so it has to be measured where its own
        sizes exist, which is at draw time and nowhere earlier."""
        from spacr.qt.widgets.fast_plots import MIN_N_FOR_DISTRIBUTION

        prefer("violin")
        agreement = _agreement(qtbot)
        assert agreement.mark() == "violin"

        agreement.set_support(_support(n=6, guides=(1, 3)))

        sizes = agreement.group_sizes()
        assert sizes and min(sizes) <= MIN_N_FOR_DISTRIBUTION
        assert agreement.mark() == "jitter"
        assert "Right-click" in agreement._status.text()

    def test_a_second_table_in_the_same_panel_is_measured_again(
            self, qtbot, prefer):
        """ONE PANEL DRAWS MANY TABLES. `RegressionResults.refresh_views`
        re-draws these from a narrowed frame, and a fit run twice is two
        tables in one widget, so the group sizes change under a panel that
        was built once. Deciding only on the first table would leave the
        jitter a three-point run fell back to drawn over four hundred
        points, with the sentence naming the three still under it -- a
        caption contradicting the group sizes printed beside it."""
        prefer("violin")
        controls = _controls(qtbot)
        controls.set_groups(_groups(sizes=(3, 3, 4)))
        assert controls.mark() == "jitter"

        controls.set_groups(_groups(sizes=(400, 350, 400)))

        assert controls.mark() == "violin"
        assert "smallest here has 3" not in controls._status.text()


# --------------------------------------------------------------------------- #
#  Measurement Compare
# --------------------------------------------------------------------------- #

class TestTheComparisonPanelOpensOnTheChoice:
    """The panel draws five of the eight types, which makes it the one that
    has to refuse a choice it cannot honour."""

    @pytest.fixture
    def rows(self):
        rows = pd.DataFrame({
            "plateID": ["p1"] * 8, "rowID": ["r1"] * 8,
            "columnID": ["c1"] * 4 + ["c2"] * 4,
            "grna": ["g1"] * 4 + ["g2"] * 4,
            "cell_area": np.arange(1.0, 9.0)})
        rows["prcfo"] = [f"p1_r1_c{1 + i // 4}_f1_o{i}" for i in range(8)]
        return rows

    def _panel(self, qtbot, rows):
        from spacr.qt.widgets.measurement_compare_dialog import (
            MeasurementComparePanel)

        widget = MeasurementComparePanel(rows, {"a": list(rows.index[:4])})
        qtbot.addWidget(widget)
        return widget

    def test_with_nothing_chosen_it_opens_on_its_own_first_plot(self, qtbot,
                                                               rows):
        panel = self._panel(qtbot, rows)
        assert panel.kind.currentData() == "jitter_box"

    @pytest.mark.parametrize("chosen,plot", [
        ("violin", "violin"),
        ("box", "box"),
        ("bar", "bar"),
        ("jitter", "jitter"),
    ])
    def test_the_saved_choice_is_what_the_box_opens_on(self, qtbot, rows,
                                                       prefer, chosen, plot):
        """``box_jitter`` is deliberately not in this list: it is also the
        panel's own first plot, so that case passes whether the setting was
        read or ignored and would prove nothing. What it maps to is asserted
        in `test_the_panel_draws_the_box_its_own_label_promises`."""
        prefer(chosen)

        panel = self._panel(qtbot, rows)

        assert panel.kind.currentData() == plot

    def test_a_choice_this_panel_cannot_draw_leaves_it_alone(self, qtbot,
                                                             rows, prefer):
        """It offers no bar-with-jitter. Opening on the nearest thing and
        calling it the choice would be the silent substitution this
        instruction is about."""
        from spacr.gene_measurement_compare import PLOTS

        prefer("bar_jitter")
        assert "bar_jitter" not in {value for value, _label in PLOTS}

        panel = self._panel(qtbot, rows)

        assert panel.kind.currentData() == "jitter_box"

    def test_changing_the_box_still_changes_this_comparison(self, qtbot, rows,
                                                            prefer):
        prefer("violin")
        panel = self._panel(qtbot, rows)

        panel.kind.setCurrentIndex(panel.kind.findData("bar"))

        assert panel.kind.currentData() == "bar"

    def test_the_panel_draws_the_box_its_own_label_promises(self, qtbot,
                                                            rows):
        """`jitter_box` used to be mapped to `bar_jitter`, so the label said
        "jitter over box" and the live plot drew a bar."""
        from spacr.qt.widgets.measurement_compare_dialog import _SPEC_KINDS

        assert _SPEC_KINDS["jitter_box"] == "box_jitter"

        from spacr.graph_types import MARKS

        assert MARKS[_SPEC_KINDS["jitter_box"]] == "jitter_box"


# --------------------------------------------------------------------------- #
#  The whole loop
# --------------------------------------------------------------------------- #

def test_remembering_a_mark_makes_the_next_plot_open_on_it(qtbot):
    """"Always start with ..." on one plot's menu is the same setting the
    settings dialog writes, and the next plot built reads it."""
    from spacr.qt import preferences

    first = _controls(qtbot)
    first.set_groups(_groups())
    assert first.mark() == "jitter"

    first._remember_default_kind(SHAPE, "violin")
    assert preferences.get_default_graph_type(SHAPE) == "violin"

    second = _controls(qtbot)
    second.set_groups(_groups())

    assert second.mark() == "violin"

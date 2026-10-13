"""The cell table's bar carries the error the user picks (instruction 204).

"for the cell table graphs if bar is chosen the user should be able to choose
SD, Var, or SEM error bars."

THE CELL TABLE GRAPHS ARE THE pyqtgraph ONES -- the Compare panel's
`GroupedPlot`, not the Matplotlib canvas the graph builder uses. The whisker
there was a hard-coded SEM with nothing on the figure saying so, and the
three quantities are not interchangeable: SD describes the cells, SEM
describes the confidence in their mean, and at n=300 they differ by more than
seventeen-fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "group": ["a"] * 300 + ["b"] * 300,
        "value": np.concatenate([rng.normal(10, 2, 300),
                                 rng.normal(12, 2, 300)])})


def _whiskers(plot):
    """The vertical spans drawn at a group's position, tallest first."""
    import pyqtgraph as pg

    out = []
    for item in plot.plot.listDataItems():
        if not isinstance(item, pg.PlotDataItem):
            continue
        x, y = item.getData()
        if x is not None and len(x) == 2 and x[0] == x[1]:
            out.append(float(y[1] - y[0]))
    return sorted(out, reverse=True)


def _plot(frame, qtbot, **kwargs):
    from spacr.qt.widgets.grouped_plot import GroupedPlot, PlotSpec

    widget = GroupedPlot(PlotSpec(frame=frame, value="value", group="group",
                                  **kwargs))
    qtbot.addWidget(widget)
    return widget


class TestTheDrawnWhiskerChanges:
    """"Choosing Bar offers SD, variance and SEM, and the drawn whisker
    changes"."""

    def test_the_three_draw_three_different_whiskers(self, frame, qtbot):
        drawn = {kind: _whiskers(_plot(frame, qtbot, kind="bar", spread=kind))
                 for kind in ("sd", "sem", "var")}
        assert len({round(v[0], 6) for v in drawn.values()}) == 3, drawn

    def test_sd_and_sem_differ_by_root_n(self, frame, qtbot):
        """Hand-computed, not merely different from each other: the whisker
        is drawn from the mean to +/- the quantity, so its span is twice the
        quantity, and SD/SEM is sqrt(300)."""
        sd = _whiskers(_plot(frame, qtbot, kind="bar", spread="sd"))[0]
        sem = _whiskers(_plot(frame, qtbot, kind="bar", spread="sem"))[0]
        assert sd / sem == pytest.approx(np.sqrt(300), rel=1e-3)

    def test_the_span_is_twice_the_hand_computed_sd(self, frame, qtbot):
        by_group = [np.std(part["value"].to_numpy(), ddof=1)
                    for _, part in frame.groupby("group")]
        drawn = _whiskers(_plot(frame, qtbot, kind="bar", spread="sd"))
        assert drawn == pytest.approx(sorted((2 * s for s in by_group),
                                             reverse=True), rel=1e-6)

    def test_the_variance_is_the_square(self, frame, qtbot):
        sd = _whiskers(_plot(frame, qtbot, kind="bar", spread="sd"))[0]
        var = _whiskers(_plot(frame, qtbot, kind="bar", spread="var"))[0]
        assert var == pytest.approx(sd ** 2 / 2, rel=1e-6)

    def test_none_draws_no_whisker(self, frame, qtbot):
        assert _whiskers(_plot(frame, qtbot, kind="bar", spread="none")) == []

    def test_it_defaults_to_the_sem_it_always_drew(self, frame, qtbot):
        plain = _whiskers(_plot(frame, qtbot, kind="bar"))
        named = _whiskers(_plot(frame, qtbot, kind="bar", spread="sem"))
        assert plain == pytest.approx(named)


class TestTheFigureSaysWhichOneItDrew:
    """"An error bar with an unnamed spread is not readable": a reader cannot
    tell a SEM from an SD without being told."""

    @pytest.mark.parametrize("spread,said", [
        ("sd", "SD"), ("sem", "SEM"), ("var", "variance")])
    def test_the_caption_names_it(self, frame, qtbot, spread, said):
        plot = _plot(frame, qtbot, kind="bar", spread=spread)
        assert said in plot._status.text()

    def test_a_graph_with_no_whisker_says_nothing_about_one(self, frame,
                                                            qtbot):
        text = _plot(frame, qtbot, kind="box", spread="sd")._status.text()
        assert "SD" not in text and "±" not in text


class TestTheControlIsAbsentRatherThanInert:
    """"The control is absent, not inert, for graph types that have no error
    bar" -- 106's rule, on a control that would otherwise be drawn and do
    nothing."""

    @pytest.fixture
    def panel(self, frame, qtbot):
        from PySide6.QtWidgets import QApplication

        from spacr.qt.widgets.measurement_compare_dialog import (
            MeasurementComparePanel)

        QApplication.instance() or QApplication([])
        rows = pd.DataFrame({
            "plateID": ["p1"] * 8, "rowID": ["r1"] * 8,
            "columnID": ["c1"] * 4 + ["c2"] * 4,
            "grna": ["g1"] * 4 + ["g2"] * 4,
            "cell_area": np.arange(1.0, 9.0)})
        rows["prcfo"] = [f"p1_r1_c{1 + i // 4}_f1_o{i}" for i in range(8)]
        widget = MeasurementComparePanel(rows, {"a": list(rows.index[:4])})
        qtbot.addWidget(widget)
        return widget

    def test_the_three_are_offered(self, panel):
        offered = [panel.spread.itemData(i)
                   for i in range(panel.spread.count())]
        for wanted in ("sd", "sem", "var"):
            assert wanted in offered

    def test_it_is_there_for_the_bar(self, panel):
        panel.kind.setCurrentIndex(panel.kind.findData("bar"))
        assert panel._has_an_error_bar()
        assert not panel.spread.isHidden()

    def test_it_is_gone_for_a_box(self, panel):
        panel.kind.setCurrentIndex(panel.kind.findData("box"))
        assert not panel._has_an_error_bar()
        assert panel.spread.isHidden()

    def test_it_is_gone_for_a_jitter(self, panel):
        panel.kind.setCurrentIndex(panel.kind.findData("jitter"))
        assert panel.spread.isHidden()

    def test_the_panel_hands_the_choice_to_the_plot(self, panel):
        """The whole point: the box the user moves reaches the drawing."""
        panel.kind.setCurrentIndex(panel.kind.findData("bar"))
        panel.spread.setCurrentIndex(panel.spread.findData("sd"))
        panel.refresh()
        canvas = panel._canvas
        if canvas is None or not hasattr(canvas, "spec"):
            pytest.skip("this build drew the fallback canvas")
        assert canvas.spec.spread == "sd"

    def test_a_type_with_no_whisker_is_drawn_without_one(self, panel):
        panel.kind.setCurrentIndex(panel.kind.findData("box"))
        panel.spread.setCurrentIndex(panel.spread.findData("sd"))
        panel.refresh()
        canvas = panel._canvas
        if canvas is None or not hasattr(canvas, "spec"):
            pytest.skip("this build drew the fallback canvas")
        assert canvas.spec.spread == "none"

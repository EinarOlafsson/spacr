"""Instruction 179: a Graph tab, and measurements that combine.

    "there should be a new tab beside summary in the cell tab after
    generating a montage that says graph ... looking at any measurement or an
    interaction between 2 measurements (one mes minus, plus, multiplied by or
    devided by another mes)."

177 F built the engine, the plots, the statistics and the one-folder save.
What is new here is a TAB rather than a window, and the four operators.
"""
import sys

import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_compare import (OPERATORS, build, combine,
                                            with_statistics)

sys.path.insert(0, "tests/qt")


@pytest.fixture()
def objects():
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame({
        "pathogen_area": rng.normal(10.0, 2.0, n),
        "cell_area": rng.normal(50.0, 5.0, n),
        "plateID": np.repeat(["p1", "p2", "p3", "p4"], n // 4),
        "rowID": rng.choice(["r1", "r2"], n),
        "columnID": rng.choice(["c1", "c2"], n),
    })
    return frame


# ------------------------------------------------------------- the operators


def test_all_four_operators_are_offered():
    assert [op for op, _label in OPERATORS] == ["", "+", "-", "*", "/"]


@pytest.mark.parametrize("operator", ["+", "-", "*", "/"])
def test_each_operator_combines(objects, operator):
    values, name, _dropped = combine(objects, "pathogen_area", operator,
                                     "cell_area")

    assert name == f"pathogen_area {operator} cell_area"
    assert np.isfinite(values.dropna()).all()


def test_the_name_is_the_expression(objects):
    """So a saved table, a figure legend and the settings file all say the
    same thing and none of them needs a key to decode."""
    _values, name, _dropped = combine(objects, "pathogen_area", "/",
                                      "cell_area")

    assert name == "pathogen_area / cell_area"


def test_the_arithmetic_is_the_arithmetic(objects):
    values, _name, _dropped = combine(objects, "pathogen_area", "-",
                                      "cell_area")

    expected = objects["pathogen_area"] - objects["cell_area"]
    assert np.allclose(values.to_numpy(), expected.to_numpy())


def test_dividing_by_zero_drops_the_row_and_counts_it(objects):
    """A zero denominator is not an error in the data and not a number in the
    result. Turning it into infinity invents an extreme observation; turning
    it into zero invents an ordinary one."""
    objects = objects.copy()
    objects.loc[objects.index[:20], "cell_area"] = 0.0

    values, _name, dropped = combine(objects, "pathogen_area", "/",
                                     "cell_area")

    assert dropped == 20
    assert values.isna().sum() == 20
    assert np.isfinite(values.dropna()).all()


def test_the_dropped_rows_are_said(objects):
    """A comparison quietly computed on fewer rows than the user thinks is
    the kind of result that survives review and is wrong."""
    objects = objects.copy()
    objects.loc[objects.index[:20], "cell_area"] = 0.0

    out = build(objects, "pathogen_area", groups={"g": list(objects.index[:60])},
                level="well", operator="/", second="cell_area")

    assert "20" in out.note
    assert "denominator" in out.note


def test_an_unknown_operator_is_refused(objects):
    with pytest.raises(ValueError):
        combine(objects, "pathogen_area", "^", "cell_area")


def test_a_missing_second_measurement_says_which(objects):
    out = build(objects, "pathogen_area", groups={}, level="well",
                operator="/", second="no_such_column")

    assert not len(out.frame)
    assert "no_such_column" in out.note


def test_a_combined_measurement_is_still_testable(objects):
    out = with_statistics(
        build(objects, "pathogen_area",
              groups={"g": list(objects.index[:60])}, level="well",
              operator="/", second="cell_area"))

    assert out.statistics
    assert out.statistics[0]["Measurement"] == "pathogen_area / cell_area"


# ----------------------------------------------------------------- the panel


def test_the_second_chooser_waits_for_an_operator(objects, qtbot):
    """A second measurement is only meaningful with something to do to it."""
    from spacr.qt.widgets.measurement_compare_dialog import (
        MeasurementComparePanel)

    panel = MeasurementComparePanel(objects, {"g": list(objects.index[:60])})
    qtbot.addWidget(panel)

    assert not panel.second.isEnabled()
    panel.operator.setCurrentIndex(4)          # divided by
    assert panel.second.isEnabled()


def test_the_panel_combines_when_asked(objects, qtbot):
    from spacr.qt.widgets.measurement_compare_dialog import (
        MeasurementComparePanel)

    panel = MeasurementComparePanel(objects, {"g": list(objects.index[:60])})
    qtbot.addWidget(panel)
    panel.operator.setCurrentIndex(4)
    panel.second.setCurrentIndex(panel.second.findData("cell_area"))

    assert panel.comparison().measurement == "pathogen_area / cell_area"


def test_the_window_is_the_panel(objects, qtbot):
    """One implementation. Two copies of a comparison would be two answers to
    the same question the first time either was edited."""
    from spacr.qt.widgets.measurement_compare_dialog import (
        MeasurementComparePanel, MeasurementCompareDialog)

    dialog = MeasurementCompareDialog(objects,
                                      {"g": list(objects.index[:60])})
    qtbot.addWidget(dialog)

    assert isinstance(dialog.panel, MeasurementComparePanel)
    assert dialog.comparison() is dialog.panel.comparison()


def test_the_panel_keeps_the_users_choice_across_a_rerun(objects, qtbot):
    """The Graph tab is re-pointed rather than rebuilt, so a chosen
    measurement survives a second montage."""
    from spacr.qt.widgets.measurement_compare_dialog import (
        MeasurementComparePanel)

    panel = MeasurementComparePanel(objects, {"g": list(objects.index[:60])})
    qtbot.addWidget(panel)
    panel.measurement.setCurrentIndex(panel.measurement.findData("cell_area"))

    panel.set_data(objects, {"g": list(objects.index[:40])})

    assert panel.measurement.currentData() == "cell_area"


# ------------------------------------------------------------------- the tab


def test_the_graph_tab_appears_after_a_montage(qtbot, tmp_path):
    import test_cells_behind_the_dot_tab as T

    root, db, csv = T._screen(tmp_path, with_png=True)
    view = T.CellMontageView(frame_provider=lambda: pd.read_csv(csv),
                             results_provider=lambda: csv,
                             database_provider=lambda: T._rows(db),
                             threaded=False)
    qtbot.addWidget(view)

    before = [view._tabs.tabText(i) for i in range(view._tabs.count())]
    assert "Graph" not in before, (
        "a Graph tab before a montage offers to graph nothing")

    view.set_coefficient(T.GENE_KEY)
    view.build()

    after = [view._tabs.tabText(i) for i in range(view._tabs.count())]
    assert after[0] == "Summary"
    assert after[1] == "Graph", (
        "the Graph tab must sit beside Summary, before the well tabs")


def test_the_graph_tab_is_not_rebuilt_on_a_second_run(qtbot, tmp_path):
    import test_cells_behind_the_dot_tab as T

    root, db, csv = T._screen(tmp_path, with_png=True)
    view = T.CellMontageView(frame_provider=lambda: pd.read_csv(csv),
                             results_provider=lambda: csv,
                             database_provider=lambda: T._rows(db),
                             threaded=False)
    qtbot.addWidget(view)
    view.set_coefficient(T.GENE_KEY)
    view.build()
    first = view._graph_panel

    view.build()

    assert view._graph_panel is first
    assert [view._tabs.tabText(i)
            for i in range(view._tabs.count())].count("Graph") == 1

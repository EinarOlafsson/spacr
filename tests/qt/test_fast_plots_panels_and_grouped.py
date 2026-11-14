"""The p-value histogram, results table, effect ranking, style files and grouped plot.

Each is a refusal a user reaches by ordinary use -- an empty bar, a q value
that is not a number, an error column the table does not have, a cancelled
style load, a spec whose frame has no rows -- driven together with the answer
it is a refusal against.  One general test per unit, each walking a case
table with pytest's ``subtests`` fixture.
"""
from __future__ import annotations

import dataclasses
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QFileDialog, QMenu, QTableWidgetItem,
)

from spacr.qt.widgets import fast_plots as fp  # noqa: E402
from spacr.qt.widgets.grouped_plot import GroupedPlot, PlotSpec  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def make(qtbot):
    """Build any widget class and hand it to qtbot for teardown."""
    def build(cls, *args, **kwargs):
        widget = cls(*args, **kwargs)
        qtbot.addWidget(widget)
        return widget
    return build


def _coefficients(n=6):
    return pd.DataFrame({
        "feature": [f"fraction:grna[g{i}]" for i in range(n)],
        "grna": [f"g{i}" for i in range(n)],
        "coefficient": np.linspace(-2.0, 2.0, n),
        "p_value": np.linspace(1e-6, 0.9, n),
    })


class _Click:
    """A pyqtgraph scene mouse event, as much of one as the handler reads."""

    def __init__(self, position):
        self._position = position

    def button(self):
        return Qt.LeftButton

    def scenePos(self):
        return self._position


def test_the_p_value_histogram_answers_about_bars_it_has(make, subtests):
    """bin_at / keys_in_bin / select_bin / highlight_key / _detail per bar."""
    def histogram():
        widget = make(fp.PValueHistogram)
        widget.resize(420, 320)
        return widget

    with subtests.test(msg="no_bars_answer_nothing_then_four_bars_answer"):
        h = histogram()
        assert (h.bin_at(0.5), h.keys_in_bin(0), h.select_bin(0)) == (None, [], [])
        h.set_p_values([0.1, 0.2, 0.3, 0.9], bins=4, keys=["a", "b", "c", "d"])
        assert (h.bin_at(0.3), h.keys_in_bin(0), h.keys_in_bin(99), h.select_bin(99)) == (
            1, ["a", "b"], [], [])
    with subtests.test(msg="an_empty_bar_says_so"):
        h = histogram()
        h.set_p_values([0.05, 0.1, 0.15], bins=4, keys=["a", "b", "c"])
        assert h.select_bin(3) == []
        assert "p 0.75 to 1: empty." in h._status.text()
        assert h.select_bin(0) == ["a", "b", "c"]
        assert "3 coefficients" in h._status.text()
    with subtests.test(msg="a_bar_without_keys_counts_and_selects_nobody"):
        h = histogram()
        calls: list = []
        h.keys_selected.connect(calls.append)
        h.set_p_values([0.05, 0.1, 0.15], bins=4)
        assert h.select_bin(0) == []
        assert "3 coefficients" in h._status.text()
        assert "A bar is not one point" in h._status.text()
        assert calls == []
        h.set_p_values([0.05, 0.1, 0.15], bins=4, keys=["a", "b", "c"])
        h.select_bin(0)
        assert calls == [["a", "b", "c"]]
    with subtests.test(msg="a_click_off_the_bars_selects_nothing"):
        h = histogram()
        h.set_p_values(np.linspace(0.01, 0.99, 20), bins=4, keys=[f"g{i}" for i in range(20)])
        item = h.plot.plotItem
        centre = item.sceneBoundingRect().center()
        expected = h.bin_at(h._to_data(item.vb.mapSceneToView(centre).x(), "x"))
        h._on_scene_clicked(_Click(centre))
        on_a_bar = h._status.text()
        assert expected is not None and f"{h.QUANTITY} " in on_a_bar
        h._on_scene_clicked(_Click(QPointF(-500.0, -500.0)))
        assert h._status.text() == on_a_bar
        h.plot.setXRange(2.0, 4.0, padding=0.0)
        h._on_scene_clicked(_Click(centre))
        assert h._status.text() == on_a_bar
    with subtests.test(msg="a_selection_marks_a_bar_only_for_a_binned_row"):
        h = histogram()
        h.set_p_values([0.05, np.nan, 0.9], bins=4, keys=["drawn", "blank", "high", "beyond"])
        assert [h.highlight_key(k) for k in ("drawn", "blank", "beyond")] == [True, False, False]
    with subtests.test(msg="a_bar_describes_the_value_behind_a_row"):
        h = histogram()
        h.set_p_values([0.02, np.nan], bins=4, keys=["a", "b"])
        assert [h._detail(i) for i in (0, 1, 9)] == ["p = 0.02", "", ""]


def test_the_results_table_reports_only_rows_it_can_read(make, monkeypatch, subtests):
    """Hits filter, row emission, key reading, selection and copy of ResultsTable."""
    def visible(table):
        return [row for row in range(table.table.rowCount())
                if not table.table.isRowHidden(row)]

    with subtests.test(msg="an_unreadable_q_is_not_a_hit"):
        table = make(fp.ResultsTable)
        table.set_frame(pd.DataFrame({"feature": ["a", "b"], "q_value": ["n/a", 0.01]}))
        table._only_hits.setChecked(True)
        assert visible(table) == [1]
        assert "1 of 2 rows (q_value <= 0.05)" in table._count.text()
    with subtests.test(msg="a_row_with_no_frame_position_emits_nothing"):
        table = make(fp.ResultsTable)
        table.set_frame(pd.DataFrame({"feature": ["a", "b"], "q_value": [0.1, 0.2]}))
        seen: list = []
        table.row_selected.connect(seen.append)
        table.table.selectRow(1)
        table.table.insertRow(2)
        table.table.setItem(2, 0, QTableWidgetItem("orphan"))
        table.table.clearSelection()
        table.table.selectRow(2)
        assert seen == [1]
    with subtests.test(msg="a_blank_key_cell_is_not_a_key"):
        table = make(fp.ResultsTable)
        table.set_frame(pd.DataFrame({"feature": ["a", ""], "value": [1, 2]}))
        table.table.selectAll()
        assert table.selected_keys() == ["a"]
    with subtests.test(msg="no_selection_model_still_counts_matches"):
        table = make(fp.ResultsTable)
        table.set_frame(pd.DataFrame({"feature": ["a", "b"]}))
        with monkeypatch.context() as mp:
            mp.setattr(table.table, "selectionModel", lambda: None)
            assert table.select_keys(["a", "b"]) == 2
        assert table.table.selectedItems() == []
    with subtests.test(msg="copy_returns_visible_rows_without_a_clipboard"):
        table = make(fp.ResultsTable)
        table.set_frame(pd.DataFrame({"feature": ["a", "b"], "q_value": [0.01, 0.9]}))
        table._only_hits.setChecked(True)
        with monkeypatch.context() as mp:
            mp.setattr(QApplication, "clipboard", staticmethod(lambda: None))
            lines = table.copy_visible().splitlines()
        assert lines[0] == "feature\tq_value"
        assert len(lines) == 2 and lines[1].startswith("a\t")


def test_the_effect_ranking_states_its_intervals_and_calls(make, subtests):
    """EffectRankPlot names missing error / significance columns and bare axes."""
    frame = _coefficients()
    frame["std_err"] = 0.2
    frame["q_value"] = [0.001, 0.2, 0.3, 0.4, 0.5, 0.6]
    cases = [
        ("an_absent_error_column", dict(error_column="posterior_sd"),
         "carries no standard error"),
        ("a_present_error_column", dict(error_column="std_err"),
         "1.96-standard-error interval from “std_err”"),
        ("a_present_significance_column", dict(significance_column="q_value"),
         "1 called at q_value ≤ 0.05"),
        ("an_absent_significance_column", dict(significance_column="posterior_probability"),
         "Nothing is coloured"),
    ]
    for case_id, kwargs, fragment in cases:
        with subtests.test(msg=case_id):
            ranked = make(fp.EffectRankPlot)
            ranked.set_results(frame, **kwargs)
            assert fragment in ranked._status.text()
    with subtests.test(msg="the_rank_axis_is_named_only_once_a_row_survives"):
        ranked = make(fp.EffectRankPlot)
        empty = pd.DataFrame({"feature": [], "coefficient": [], "p_value": []})
        assert ranked.set_results(empty) == 0
        assert "No coefficients to rank" in ranked._status.text()
        assert not ranked.plot.getAxis("left")._tickLevels
        assert ranked.set_results(_coefficients(1)) == 1
        levels = ranked.plot.getAxis("left")._tickLevels
        assert len(levels[0]) == 1
        low, high = ranked.plot.getViewBox().viewRange()[1]
        assert low < -0.5 and high > 0.5


@dataclasses.dataclass
class ScaleLocationStyle:
    """A style with one field of every shape the menu has to sort out."""

    y_lim: tuple = (0.0, 1.0)
    point_colour: str = "#112233"
    dpi: int = 300
    marker: str = dataclasses.field(default="o", metadata={"choices": ("o", "s", "d")})
    corners: tuple = (0, 0, 1, 1)

    CHOICES = {"point_colour": ()}


def _entry(actions, fragment):
    return next(action for action in actions if fragment in action.text())


def test_the_style_file_entries_save_load_and_clear_honestly(tmp_path, monkeypatch, subtests):
    """add_style_file_entries: a save says where, a failure says why, cancel and clear do nothing."""
    menus: list = []

    def entries(style, note):
        menus.append(QMenu())       # kept alive: the actions belong to it
        return fp.add_style_file_entries(menus[-1], style, note=note)

    with subtests.test(msg="saving_says_where_it_went"):
        target = tmp_path / "house.json"
        said: list = []
        with monkeypatch.context() as mp:
            mp.setattr(QFileDialog, "getSaveFileName",
                       staticmethod(lambda *a, **k: (str(target), "")))
            actions = entries(ScaleLocationStyle(), said.append)
            _entry(actions, "Save style").trigger()
        assert target.is_file()
        assert any(str(target) in message for message in said)
    with subtests.test(msg="an_unwritable_style_names_the_reason"):
        def read_only(*_a, **_k):
            raise OSError("Read-only file system")

        said = []
        with monkeypatch.context() as mp:
            mp.setattr(QFileDialog, "getSaveFileName",
                       staticmethod(lambda *a, **k: (str(tmp_path / "s.json"), "")))
            mp.setattr(fp, "save_style", read_only)
            actions = entries(ScaleLocationStyle(), said.append)
            _entry(actions, "Save style").trigger()
        assert any("Could not write the style" in m and "Read-only" in m for m in said)
    with subtests.test(msg="a_cancelled_load_changes_nothing"):
        style, said = ScaleLocationStyle(), []
        with monkeypatch.context() as mp:
            mp.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", "")))
            actions = entries(style, said.append)
            _entry(actions, "Load style").trigger()
        assert (style, said) == (ScaleLocationStyle(), [])
    with subtests.test(msg="clear_is_greyed_with_no_default"):
        clear = _entry(entries(ScaleLocationStyle(), [].append), "Clear the default")
        assert clear.isEnabled() is False
        assert "scale_location" in clear.toolTip()


def test_a_plot_spec_groups_only_columns_its_frame_has(subtests):
    """PlotSpec.groups: one series per level, the value alone, or nothing."""
    frame = pd.DataFrame({"well": ["a", "a", "b", "b"], "area": [1.0, 2.0, 3.0, 4.0]})
    cases = [
        ("grouped_by_a_present_column", dict(frame=frame, value="area", group="well"),
         {"a": [1.0, 2.0], "b": [3.0, 4.0]}),
        ("a_value_column_that_is_gone", dict(frame=frame, value="perimeter", group="well"), {}),
        ("no_frame_at_all", dict(frame=None, value="area"), {}),
        ("no_group_column_is_one_series", dict(frame=frame, value="area"),
         {"area": [1.0, 2.0, 3.0, 4.0]}),
        ("a_stale_group_column_is_one_series", dict(frame=frame, value="area", group="plate"),
         {"area": [1.0, 2.0, 3.0, 4.0]}),
    ]
    for case_id, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            groups = PlotSpec(**kwargs).groups()
            assert {k: list(np.asarray(v, float)) for k, v in groups.items()} == expected


def _show_as(menu):
    for action in menu.actions():
        if action.menu() is not None and action.text() == "Show as":
            return action.menu()
    return None


def _labels(menu):
    sub = _show_as(menu)
    return None if sub is None else [a.text() for a in sub.actions()]


def test_a_grouped_plot_keeps_its_spec_to_redraw_offer_and_export(make, monkeypatch, subtests):
    """GroupedPlot: frame/export follow the spec; 'Show as' needs rows, offers and a kind."""
    from spacr import graph_types
    from spacr.graph_types import shape_of

    frame = pd.DataFrame({"group": ["a", "a", "b", "b"], "value": [1.0, 2.0, 3.0, 4.0]})

    def boxed():
        return make(GroupedPlot, PlotSpec(frame=frame, value="value", group="group", kind="box"))

    with subtests.test(msg="the_exported_frame_is_the_drawn_one"):
        assert boxed().frame() is frame
    for case_id, spec, unit in (
            ("export_with_a_unit", PlotSpec(frame=frame.rename(columns={"group": "treatment"}),
                                            group="treatment", value="value", unit="cell"),
             "cell"),
            ("export_without_a_unit", PlotSpec(frame=frame, value="value", group="group"), None)):
        with subtests.test(msg=case_id):
            plot = make(GroupedPlot)
            plot.show_spec(spec)
            out = plot.export_settings()
            assert (out["group"], out["value"]) == (spec.group, "value")
            assert out["kind"] != "" and out["shape"] != ""
            if unit:
                assert out["unit"] == unit
    with subtests.test(msg="a_line_joins_its_points_in_x_order"):
        plot = make(GroupedPlot)
        dose = pd.DataFrame({"dose": [3.0, 1.0, 2.0], "response": [30.0, 10.0, 20.0]})
        assert plot.show_spec(PlotSpec(frame=dose, value="response", group="dose",
                                       kind="line")) == 1
        curves = [i for i in plot.plot.listDataItems() if i.xData is not None]
        assert list(curves[0].xData) == [1.0, 2.0, 3.0]
    with subtests.test(msg="no_rows_left_no_graph_types"):
        plot = boxed()
        assert _show_as(plot.build_style_menu()) is not None
        plot.spec = replace(plot.spec, frame=pd.DataFrame({"group": [], "value": []}))
        assert _show_as(plot.build_style_menu()) is None
    with subtests.test(msg="an_empty_offer_adds_no_submenu"):
        plot = boxed()
        assert len(_labels(plot.build_style_menu())) > 1
        with monkeypatch.context() as mp:
            mp.setattr(graph_types, "offer", lambda *a, **k: [])
            assert _show_as(plot.build_style_menu()) is None
    with subtests.test(msg="only_a_pinned_kind_can_be_the_starting_point"):
        plot = boxed()
        assert any(t.startswith("Always start with") for t in _labels(plot.build_style_menu()))
        plot.spec = replace(plot.spec, kind="")
        kindless = _labels(plot.build_style_menu())
        assert kindless != []
        assert not any(t.startswith("Always start with") for t in kindless)
    shapes = {
        "categorical_continuous": (pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 2.0, 3.0]}), "g"),
        "continuous_continuous": (pd.DataFrame({"g": [2.0, 1.0, 2.0], "v": [1.0, 2.0, 3.0]}), "g"),
        "ordered_continuous": (pd.DataFrame({"g": [1.0, 2.0, 3.0], "v": [1.0, 2.0, 3.0]}), "g"),
        "continuous_only": (pd.DataFrame({"v": [1.0, 2.0, 3.0]}), ""),
    }
    for expected, (shape_frame, group) in shapes.items():
        with subtests.test(msg=f"every_shape_can_be_remembered_{expected}"):
            plot = boxed()
            plot.spec = replace(plot.spec, frame=shape_frame, group=group, value="v", kind="box")
            assert shape_of(shape_frame, group, "v") == expected
            entry = next(a for a in _show_as(plot.build_style_menu()).actions()
                         if a.text().startswith("Always start with"))
            assert entry.toolTip().startswith("Draw this kind first")

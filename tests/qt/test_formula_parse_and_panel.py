"""A computed column either means something exact or says why it does not.

The formula language turns a sentence a user typed into a column the rest of
the analysis treats as data, and the Formula panel is where that sentence is
validated before it may be added.  Malformed text must raise with the sentence
that fits it, evaluation must give exact values (NaN rather than 0 when nothing
was measured), and the panel must keep Add disabled and its status line exact
for every half-typed, unparseable, inapplicable or valid formula.

One general test per behaviour family, each walking a case table with pytest's
``subtests`` fixture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402

from spacr.qt.widgets.formula import (  # noqa: E402
    ColumnFormula, FormulaError, evaluate, parse,
)
from spacr.qt.widgets.formula_editor import PREVIEW_ROWS, FormulaPanel  # noqa: E402

pytestmark = pytest.mark.qt


def _frame():
    return pd.DataFrame({"area": [10.0, 20.0, 30.0, 40.0],
                         "flag": [True, False, True, False],
                         "gene": ["a", "b", "a", "b"]})


def _table():
    return pd.DataFrame({"area": [10.0, 20.0, 30.0],
                         "perimeter": [4.0, 8.0, 12.0]})


_BAD_NAME = ("'2 bad name' is not a usable column name — letters, digits and "
             "underscores, not starting with a digit. Anything else has to be "
             "quoted everywhere it is used")


def test_parse_refuses_each_malformed_expression_with_its_own_sentence(subtests):
    """Every incomplete or misplaced token raises a FormulaError naming the problem exactly."""
    cases = {
        "empty": ("", "the formula is empty. Write something like "
                      "cell_area / cell_perimeter ** 2"),
        "ends_early": ("area +", "the formula ends early — something is "
                                 "missing after '+'"),
        "ends_early_trailing_space": ("area + ", "the formula ends early — "
                                                 "something is missing after '+'"),
        "leading_and": ("and area", "'and' at position 1 needs something "
                                    "before it; it joins two expressions"),
        "trailing_not": ("area not", "unexpected 'not' at position 6 — 'not' "
                                     "joins two comparisons; the expression "
                                     "already ended before it"),
        "lone_paren": (")", "unexpected ')' at position 1"),
        "lone_comma": (",", "unexpected ',' at position 1"),
        "unclosed_call": ("mean(area", "missing ')' after the arguments of mean()"),
    }
    for case_id, (expression, message) in cases.items():
        with subtests.test(msg=case_id):
            with pytest.raises(FormulaError) as caught:
                parse(expression)
            assert str(caught.value) == message


def test_evaluate_gives_exact_values_and_nan_where_nothing_was_measured(subtests):
    """Signs, booleans, constants, spreads, aggregates and ranks evaluate to exact values."""
    frame = _frame()
    one = pd.DataFrame({"area": [5.0]})
    missing = pd.DataFrame({"area": [np.nan, np.nan]})
    with_gap = pd.DataFrame({"area": [30.0, np.nan, 10.0, 20.0]})

    with subtests.test(msg="leading_plus_is_identity"):
        assert list(evaluate(parse("+area"), frame)) == [10.0, 20.0, 30.0, 40.0]
    with subtests.test(msg="boolean_is_one_and_zero"):
        assert list(evaluate(parse("flag + 0"), frame)) == [1.0, 0.0, 1.0, 0.0]
    with subtests.test(msg="aggregate_of_a_constant"):
        assert evaluate(parse("mean(3)"), frame) == 3.0
    with subtests.test(msg="function_failure_names_the_function"):
        with pytest.raises(FormulaError) as caught:
            evaluate(parse("round(area, log(-1))"), frame)
        assert str(caught.value).startswith("round() could not be computed: ")
    for expression in ("std(area)", "var(area)"):
        with subtests.test(msg=f"one_row_{expression}"):
            assert np.isnan(evaluate(parse(expression), one))
    with subtests.test(msg="one_row_zscore"):
        assert np.isnan(np.asarray(evaluate(parse("zscore(area)"), one))).all()
    for expression in ("mean(area)", "sum(area)", "quantile(area, 0.5)"):
        with subtests.test(msg=f"all_missing_{expression}"):
            assert np.isnan(evaluate(parse(expression), missing))
    with subtests.test(msg="rank_is_one_based_and_skips_missing"):
        ranks = np.asarray(evaluate(parse("rank(area)"), with_gap))
        np.testing.assert_array_equal(ranks, [3.0, np.nan, 1.0, 2.0])
        assert np.nanmax(ranks) == evaluate(parse("count(area)"), with_gap) == 3


def test_formula_panel_status_line_and_add_button_follow_what_is_typed(qtbot, subtests):
    """Each state of the two boxes gives an exact status sentence and the right Add state."""
    long_table = pd.DataFrame({"area": range(PREVIEW_ROWS * 3),
                               "perimeter": range(1, PREVIEW_ROWS * 3 + 1)})
    n = f"{PREVIEW_ROWS:,}"
    cases = {
        # case: (table, name, expression, status, add enabled)
        "no_table_parses": (None, "ratio", "area / perimeter",
                            "parses — load a table to see the values", True),
        "unparseable": (_table(), "ratio", "area / /",
                        "unexpected '/' at position 8", False),
        "valid_preview": (_table(), "ratio", "area / perimeter",
                          "ratio: 3 of 3 rows have a finite value", True),
        "whole_table": (_table(), "share", "area / sum(area)",
                        "share: 3 of 3 rows have a finite value · uses the "
                        "whole table, so this column changes if the table does",
                        True),
        "long_table_preview": (long_table, "ratio", "area / perimeter",
                               f"ratio: {n} of {n} rows have a finite value "
                               f"(previewed on the first {n} rows)", True),
        "invalid_name": (_table(), "2 bad name", "area", _BAD_NAME, False),
        "name_without_expression": (_table(), "ratio", "",
                                    "type an expression, then press Enter", False),
    }
    for case_id, (table, name, expression, status, enabled) in cases.items():
        with subtests.test(msg=case_id):
            panel = FormulaPanel()
            qtbot.addWidget(panel)
            panel.set_frame(table)
            panel._name.setText(name)
            panel._expression.setText(expression)
            panel._validate()
            assert panel.status() == status
            assert panel._add.isEnabled() is enabled
            if not enabled:
                assert panel.commit() is False
    with subtests.test(msg="half_filled_is_no_formula"):
        panel = FormulaPanel()
        qtbot.addWidget(panel)
        panel._name.setText("ratio")
        assert panel._current_formula() is None
    with subtests.test(msg="invalid_name_never_constructs"):
        with pytest.raises(FormulaError) as caught:
            ColumnFormula("2 bad name", "area")
        assert str(caught.value) == _BAD_NAME


def test_formula_panel_adds_refuses_removes_and_reapplies_formulas(qtbot, subtests):
    """Adding, refusing, removing and reloading change the formula set and computed frame exactly."""
    table = _table()

    def make():
        panel = FormulaPanel()
        qtbot.addWidget(panel)
        panel.set_frame(table)
        return panel

    with subtests.test(msg="missing_column_refused_with_its_name"):
        panel = make()
        assert panel.add_formula(ColumnFormula("bad", "area / girth")) is False
        assert panel.status() == ("bad: there is no column called 'girth' in "
                                  "this table (it has 2 columns)")
        pd.testing.assert_frame_equal(panel.computed_frame(), table)
        assert panel.formulas().names == ()
    with subtests.test(msg="parses_but_cannot_apply_on_commit"):
        panel = make()
        panel._name.setText("ratio")
        panel._expression.setText("area / girth")
        assert panel.commit() is False
        assert panel.status() == ("ratio: there is no column called 'girth' "
                                  "in this table (it has 2 columns)")
    with subtests.test(msg="valid_commit_clears_boxes"):
        panel = make()
        panel._name.setText("compactness")
        panel._expression.setText("area / perimeter ** 2")
        panel._replace.setChecked(True)
        assert panel.commit() is True
        assert (panel._name.text(), panel._expression.text()) == ("", "")
        assert panel._replace.isChecked() is False
        assert panel.status() == "added compactness"
        assert list(panel.computed_frame()["compactness"]) == pytest.approx(
            [10.0 / 16.0, 20.0 / 64.0, 30.0 / 144.0])
    with subtests.test(msg="remove_with_nothing_selected"):
        panel = make()
        panel.add_formula(ColumnFormula("ratio", "area / perimeter"))
        panel._list.setCurrentItem(None)
        panel.remove_selected()
        assert panel.formulas().names == ("ratio",)
    with subtests.test(msg="remove_unknown_name"):
        panel = make()
        panel.add_formula(ColumnFormula("ratio", "area / perimeter"))
        with qtbot.assertNotEmitted(panel.formulas_changed):
            panel.remove("never_defined")
        assert panel.formulas().names == ("ratio",)
    with subtests.test(msg="remove_selected_drops_the_column"):
        panel = make()
        panel.add_formula(ColumnFormula("ratio", "area / perimeter"))
        panel._list.setCurrentRow(0)
        assert panel._list.currentItem().data(Qt.UserRole) == "ratio"
        with qtbot.waitSignal(panel.formulas_changed):
            panel.remove_selected()
        assert panel.formulas().names == ()
        assert list(panel.computed_frame().columns) == ["area", "perimeter"]
    with subtests.test(msg="no_table_keeps_formulas_drops_values"):
        panel = make()
        panel.add_formula(ColumnFormula("ratio", "area / perimeter"))
        panel.set_frame(None)
        assert panel.computed_frame() is None
        assert panel.results() == []
        assert panel.formulas().names == ("ratio",)
    with subtests.test(msg="kept_formula_on_new_table_names_missing_column"):
        panel = make()
        panel.add_formula(ColumnFormula("ratio", "area / perimeter"))
        panel.set_frame(pd.DataFrame({"area": [1.0, 2.0]}))
        assert panel.status() == ("ratio: there is no column called "
                                  "'perimeter' in this table (it has 1 columns)")
        assert panel.results() == []

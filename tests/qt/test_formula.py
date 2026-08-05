"""B7 — the column formula language, worked out by hand before the code runs.

The frame every test below uses is four rows, so the whole answer can be
written down first::

    area  perimeter  gene  count
      10          5   a        1
      20          5   b        2
      30         10   a        3
      40          0   b        4

``ratio = area / perimeter ** 2`` is therefore exactly

    10/25 = 0.4     20/25 = 0.8     30/100 = 0.3     40/0 = inf

and that last cell is the point of the example: a division by zero is a real
answer about one object, it is kept, and it is counted out loud.

The row that matters most for precedence is the first. ``area / perimeter ** 2``
is 0.4 if ``**`` binds tighter than ``/`` and 4.0 if it does not, so one
assertion separates the two readings.

Safety is tested by *outcome*, not by reading the source: the hostile
expressions below are asserted to raise, and the file they would have written
is asserted not to exist afterwards.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.formula import (
    AGGREGATE_FUNCTIONS, FUNCTIONS, MAX_DEPTH, MAX_NODES,
    Binary, Call, Column, ColumnFormula, FormulaError, FormulaSet, Number,
    Unary, compute, evaluate, parse, referenced_columns, tokenize, unparse,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "area": [10.0, 20.0, 30.0, 40.0],
        "perimeter": [5.0, 5.0, 10.0, 0.0],
        "gene": ["a", "b", "a", "b"],
        "count": [1, 2, 3, 4],
    })


def values_of(frame: pd.DataFrame, expression: str) -> np.ndarray:
    """The expression evaluated over the frame, as a length-4 array."""
    result = evaluate(parse(expression), frame)
    return np.asarray(result if np.ndim(result) else
                      np.full(len(frame), result))


# ---------------------------------------------------------------------------
# The worked example
# ---------------------------------------------------------------------------

def test_ratio_is_exactly_the_hand_computed_column(frame):
    computed, results = compute(
        frame, [ColumnFormula("ratio", "area / perimeter ** 2")])
    got = computed["ratio"].to_numpy()
    assert got[0] == pytest.approx(0.4)
    assert got[1] == pytest.approx(0.8)
    assert got[2] == pytest.approx(0.3)
    assert math.isinf(got[3]) and got[3] > 0
    # The division by zero is counted, not hidden.
    assert results[0].n_nonfinite == 1
    assert "3 of 4 rows have a finite value" in results[0].notice
    assert "division by zero" in results[0].notice


def test_power_binds_tighter_than_division(frame):
    """0.4 and 4.0 are the two readings; only one of them is arithmetic."""
    assert values_of(frame, "area / perimeter ** 2")[0] == pytest.approx(0.4)
    assert values_of(frame, "(area / perimeter) ** 2")[0] == pytest.approx(4.0)


def test_power_is_right_associative(frame):
    """2 ** 3 ** 2 is 2**9 = 512, not (2**3)**2 = 64."""
    assert values_of(frame, "2 ** 3 ** 2")[0] == pytest.approx(512.0)


def test_unary_minus_applies_to_the_power(frame):
    """-area ** 2 is -(area ** 2) = -100, as in Python and in arithmetic."""
    assert values_of(frame, "-area ** 2")[0] == pytest.approx(-100.0)


def test_arithmetic_precedence_and_parentheses(frame):
    assert values_of(frame, "1 + 2 * 3")[0] == pytest.approx(7.0)
    assert values_of(frame, "(1 + 2) * 3")[0] == pytest.approx(9.0)
    assert values_of(frame, "7 % 4")[0] == pytest.approx(3.0)
    assert values_of(frame, "7 // 4")[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Aggregates reduce over the whole table
# ---------------------------------------------------------------------------

def test_aggregates_are_one_number_broadcast(frame):
    """mean(area) is 25 for every row, so area - mean(area) is the deviation."""
    assert values_of(frame, "mean(area)").tolist() == [25.0] * 4
    np.testing.assert_allclose(
        values_of(frame, "area - mean(area)"), [-15.0, -5.0, 5.0, 15.0])


def test_zscore_uses_the_sample_sd(frame):
    """sd of (10,20,30,40) with ddof=1 is sqrt(500/3); z0 = -15/that."""
    sd = math.sqrt(500.0 / 3.0)
    np.testing.assert_allclose(
        values_of(frame, "zscore(area)"),
        [-15.0 / sd, -5.0 / sd, 5.0 / sd, 15.0 / sd])
    assert values_of(frame, "std(area)")[0] == pytest.approx(sd)


def test_count_counts_finite_values_only():
    frame = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    assert values_of(frame, "count(x)")[0] == pytest.approx(2.0)
    assert values_of(frame, "mean(x)")[0] == pytest.approx(2.0)


def test_min_max_are_aggregates_and_minimum_maximum_are_not(frame):
    assert values_of(frame, "min(area)").tolist() == [10.0] * 4
    np.testing.assert_allclose(
        values_of(frame, "minimum(area, 25)"), [10.0, 20.0, 25.0, 25.0])
    assert set(AGGREGATE_FUNCTIONS) >= {"min", "max", "mean", "count"}
    assert "minimum" not in AGGREGATE_FUNCTIONS


def test_quantile_rejects_a_fraction_outside_zero_one(frame):
    assert values_of(frame, "quantile(area, 0.5)")[0] == pytest.approx(25.0)
    with pytest.raises(FormulaError, match=r"\[0, 1\]"):
        values_of(frame, "quantile(area, 75)")


def test_table_dependence_is_reported(frame):
    """zscore and rank do not reduce, but they still read every row."""
    assert ColumnFormula("a", "area / mean(area)").uses_whole_table()
    assert not ColumnFormula("b", "area / perimeter").uses_whole_table()
    assert ColumnFormula("c", "zscore(area)").uses_whole_table()
    assert ColumnFormula("d", "rank(area)").uses_whole_table()
    assert "uses the whole table" in ColumnFormula(
        "a", "zscore(area)").describe()


def test_aggregates_see_the_whole_table_not_a_filtered_view(frame):
    """The reason compute() is handed the loaded frame, stated as a test.

    zscore over the top two rows is a different column from zscore over four,
    and a column that moves when a slider moves is not a column.
    """
    whole, _ = compute(frame, [ColumnFormula("z", "zscore(area)")])
    half, _ = compute(frame.iloc[:2], [ColumnFormula("z", "zscore(area)")])
    assert whole["z"].iloc[0] != pytest.approx(half["z"].iloc[0])


# ---------------------------------------------------------------------------
# Booleans
# ---------------------------------------------------------------------------

def test_comparison_gives_a_boolean_column(frame):
    computed, results = compute(frame, [ColumnFormula("big", "area > 20")])
    assert computed["big"].tolist() == [False, False, True, True]
    assert results[0].is_boolean
    assert "2 of 4 rows true" in results[0].notice


def test_boolean_operators(frame):
    np.testing.assert_array_equal(
        values_of(frame, "area > 15 and perimeter > 5"),
        [False, False, True, False])
    np.testing.assert_array_equal(
        values_of(frame, "not (area > 20)"), [True, True, False, False])
    np.testing.assert_array_equal(
        values_of(frame, "area < 15 or area > 35"),
        [True, False, False, True])


def test_where_picks_per_row(frame):
    np.testing.assert_allclose(
        values_of(frame, "where(perimeter > 0, area / perimeter, 0)"),
        [2.0, 4.0, 3.0, 0.0])


def test_not_binds_looser_than_comparison(frame):
    """`not a > b` is `not (a > b)` — the same reading Python has."""
    np.testing.assert_array_equal(
        values_of(frame, "not area > 20"),
        values_of(frame, "not (area > 20)"))


# ---------------------------------------------------------------------------
# Errors name the column and the problem
# ---------------------------------------------------------------------------

def test_unknown_column_is_named_and_a_neighbour_suggested(frame):
    with pytest.raises(FormulaError) as excinfo:
        values_of(frame, "aera * 2")
    message = str(excinfo.value)
    assert "'aera'" in message
    assert "area" in message           # the suggestion


def test_text_column_says_so_and_points_at_the_filter(frame):
    with pytest.raises(FormulaError) as excinfo:
        values_of(frame, "gene + 1")
    message = str(excinfo.value)
    assert "'gene'" in message
    assert "text" in message
    assert "Local Data Filter" in message


def test_numbers_stored_as_text_still_work():
    frame = pd.DataFrame({"n": ["1", "2", "x"]})
    np.testing.assert_allclose(values_of(frame, "n * 2"), [2.0, 4.0, np.nan])


def test_unknown_function_lists_the_ones_that_exist(frame):
    with pytest.raises(FormulaError) as excinfo:
        parse("logg(area)")
    message = str(excinfo.value)
    assert "logg()" in message
    assert "log10" in message           # the function list is printed


def test_wrong_arity_names_the_function(frame):
    with pytest.raises(FormulaError, match=r"clip\(\) takes 3 argument"):
        parse("clip(area, 1)")


def test_chained_comparison_is_refused_with_the_rewrite(frame):
    with pytest.raises(FormulaError) as excinfo:
        parse("0 < area < 5")
    assert "chained comparison" in str(excinfo.value)
    assert "'and'" in str(excinfo.value)


def test_unbalanced_parenthesis_says_which_one(frame):
    with pytest.raises(FormulaError, match="never closed"):
        parse("(area + 1")


def test_empty_formula_suggests_one():
    with pytest.raises(FormulaError, match="empty"):
        parse("   ")


def test_a_failed_formula_adds_no_columns(frame):
    with pytest.raises(FormulaError):
        compute(frame, [ColumnFormula("ok", "area * 2"),
                        ColumnFormula("bad", "nope * 2")])
    assert "ok" not in frame.columns          # the caller's frame is untouched


def test_replacing_an_existing_column_needs_saying_so(frame):
    with pytest.raises(FormulaError, match="already has a column"):
        compute(frame, [ColumnFormula("area", "area * 2")])
    computed, _ = compute(
        frame, [ColumnFormula("area", "area * 2", replace=True)])
    assert computed["area"].tolist() == [20.0, 40.0, 60.0, 80.0]


def test_a_formula_cannot_refer_to_a_column_that_does_not_exist_yet(frame):
    with pytest.raises(FormulaError, match="refers to itself"):
        compute(frame, [ColumnFormula("x", "x + 1", replace=True)])


def test_rescaling_a_real_column_in_place_is_allowed_and_idempotent(frame):
    """`area = area * 2` reads the measured column; re-applying does not
    double twice, because compute() always starts from the loaded table."""
    formulas = [ColumnFormula("area", "area * 2", replace=True)]
    once, _ = compute(frame, formulas)
    twice, _ = compute(frame, formulas)
    assert once["area"].tolist() == twice["area"].tolist() == [
        20.0, 40.0, 60.0, 80.0]


def test_a_bad_name_is_refused_at_construction():
    for name in ("2big", "has space", "", "log"):
        with pytest.raises(FormulaError):
            ColumnFormula(name, "1")


# ---------------------------------------------------------------------------
# Hostile input cannot execute anything
# ---------------------------------------------------------------------------

HOSTILE = [
    "__import__('os').system('touch /tmp/spacr_formula_pwned')",
    "().__class__.__bases__[0].__subclasses__()",
    "open('/tmp/spacr_formula_pwned', 'w')",
    "eval('1+1')",
    "exec('import os')",
    "compile('1', '<s>', 'eval')",
    "globals()",
    "area.__class__",
    "area[0]",
    "lambda: 1",
    "area if area else 1",
    "x = 1",
    "import os",
    "area; area",
    "getattr(area, 'x')",
    "1 & 2",
    "area @ area",
]


@pytest.mark.parametrize("expression", HOSTILE)
def test_hostile_expressions_never_reach_an_interpreter(frame, expression):
    """Every one of these fails to tokenise, to parse, or to resolve a name.

    None of them is on a blacklist. ``__import__`` is refused because it is not
    a column of the frame; ``.`` and ``[`` are refused because the grammar has
    no such tokens.
    """
    with pytest.raises(FormulaError):
        values_of(frame, expression)


def test_the_hostile_expressions_wrote_nothing(frame, tmp_path):
    """The outcome, not the mechanism: the side effect never happened."""
    target = "/tmp/spacr_formula_pwned"
    if os.path.exists(target):           # pragma: no cover - a dirty box
        os.unlink(target)
    for expression in HOSTILE:
        with pytest.raises(FormulaError):
            values_of(frame, expression)
    assert not os.path.exists(target)


def test_the_error_for_a_dot_explains_the_language(frame):
    with pytest.raises(FormulaError) as excinfo:
        parse("area.mean")
    assert "attribute access" in str(excinfo.value)


def test_quotes_point_at_the_filter_panel(frame):
    with pytest.raises(FormulaError, match="Local Data Filter"):
        parse("gene == 'a'")


def test_ampersand_suggests_the_word(frame):
    with pytest.raises(FormulaError, match="'and'"):
        parse("area > 1 & area < 5")


def test_assignment_suggests_the_comparison(frame):
    with pytest.raises(FormulaError, match="=="):
        parse("area = 5")


def test_a_giant_integer_power_cannot_hang_the_gui(frame):
    """9 ** 9 ** 9 with Python ints allocates a 300-million-digit number.

    Literals are floats, so it is ``inf`` immediately.
    """
    got = values_of(frame, "9 ** 9 ** 9")
    assert math.isinf(got[0])


def test_a_pathological_expression_is_refused_before_it_is_evaluated(frame):
    with pytest.raises(FormulaError, match="more than"):
        parse("(" * (MAX_DEPTH + 2) + "1" + ")" * (MAX_DEPTH + 2))
    with pytest.raises(FormulaError, match="more than"):
        parse("+".join(["1"] * (MAX_NODES + 5)))


def test_an_over_long_formula_is_refused():
    with pytest.raises(FormulaError, match="characters"):
        tokenize("1" * 5000)


def test_no_dynamic_execution_in_the_module_source():
    """A guard against the one-line shortcut being reintroduced later.

    Read with Python's own parser rather than by grepping, so the module
    docstring — which quotes the ``eval`` line it exists to avoid — cannot
    make the test pass or fail for the wrong reason.
    """
    import ast as _ast
    import inspect

    from spacr.qt.widgets import formula as module

    forbidden = {"eval", "exec", "compile", "__import__", "getattr",
                 "setattr", "globals", "locals", "vars", "open"}
    tree = _ast.parse(inspect.getsource(module))
    called = {node.func.id for node in _ast.walk(tree)
              if isinstance(node, _ast.Call)
              and isinstance(node.func, _ast.Name)}
    assert not (called & forbidden), f"dynamic execution reintroduced: {called & forbidden}"
    # `re.compile` is fine and is the only attribute call worth allowing;
    # nothing else may reach an interpreter through an attribute either.
    attribute_calls = {
        f"{node.func.value.id}.{node.func.attr}" for node in _ast.walk(tree)
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Attribute)
        and isinstance(node.func.value, _ast.Name)}
    assert not ({c for c in attribute_calls
                 if c.split(".")[-1] in forbidden} - {"re.compile"})


# ---------------------------------------------------------------------------
# The AST, and the round trip
# ---------------------------------------------------------------------------

def test_the_tree_is_the_tree_the_grammar_describes():
    node = parse("area / perimeter ** 2")
    assert isinstance(node, Binary) and node.op == "/"
    assert node.left == Column("area")
    assert isinstance(node.right, Binary) and node.right.op == "**"
    assert node.right.left == Column("perimeter")
    assert node.right.right == Number(2.0)


def test_backticks_reach_a_column_whose_name_is_not_an_identifier():
    frame = pd.DataFrame({"cell area": [2.0, 4.0]})
    np.testing.assert_allclose(values_of(frame, "`cell area` * 2"), [4.0, 8.0])
    assert referenced_columns(parse("`cell area` + 1")) == ("cell area",)


def test_unparse_round_trips_through_the_parser():
    for text in ("area / perimeter ** 2", "-area ** 2", "2 ** 3 ** 2",
                 "not (area > 20)", "a > 1 and b < 2 or c == 3",
                 "where(a > 0, log(a), 0)", "`odd name` + 1"):
        once = parse(text)
        assert parse(unparse(once)) == once


def test_referenced_columns_are_in_first_appearance_order():
    assert referenced_columns(parse("b + a * b + c")) == ("b", "a", "c")


def test_the_ast_is_frozen():
    node = parse("area + 1")
    with pytest.raises(Exception):
        node.op = "-"


# ---------------------------------------------------------------------------
# Serialisation, and the set
# ---------------------------------------------------------------------------

def test_a_formula_round_trips_through_json():
    formula = ColumnFormula("ratio", "area / perimeter ** 2")
    again = ColumnFormula.from_json(formula.to_json())
    assert again == formula
    assert again.ast == formula.ast


def test_a_formula_set_round_trips_and_keeps_its_order(frame):
    formulas = FormulaSet()
    formulas.add(ColumnFormula("d", "count / area"))
    formulas.add(ColumnFormula("ld", "log(d)"))
    again = FormulaSet.from_json(formulas.to_json())
    assert again.names == ("d", "ld")
    computed, results = again.apply(frame)
    # d = count/area = 0.1, 0.1, 0.1, 0.1 -> log(0.1) for every row
    np.testing.assert_allclose(computed["d"].to_numpy(), [0.1] * 4)
    np.testing.assert_allclose(computed["ld"].to_numpy(),
                               [math.log(0.1)] * 4)
    assert len(results) == 2


def test_a_later_formula_sees_an_earlier_column_and_not_the_reverse(frame):
    with pytest.raises(FormulaError, match="no column called 'ld'"):
        compute(frame, [ColumnFormula("d", "log(ld)"),
                        ColumnFormula("ld", "area")])


def test_adding_a_formula_twice_replaces_it(frame):
    formulas = FormulaSet()
    formulas.add(ColumnFormula("x", "area"))
    formulas.add(ColumnFormula("x", "area * 2"))
    assert len(formulas) == 1
    computed, _ = formulas.apply(frame)
    assert computed["x"].tolist() == [20.0, 40.0, 60.0, 80.0]
    formulas.remove("x")
    assert formulas.is_empty
    assert "no computed columns" == formulas.describe()


def test_apply_never_mutates_the_source_frame(frame):
    before = list(frame.columns)
    FormulaSet([ColumnFormula("x", "area")]).apply(frame)
    assert list(frame.columns) == before


# ---------------------------------------------------------------------------
# The new column participates in everything else
# ---------------------------------------------------------------------------

def test_a_computed_column_is_classified_like_any_other():
    from spacr.qt.widgets.data_filter_panel import classify_columns
    from spacr.qt.widgets.graph_spec import (
        CATEGORICAL, CONTINUOUS, column_kinds, plottable_columns)

    # Wider than the four-row fixture on purpose: `classify_columns` calls a
    # numeric column with twelve or fewer distinct values a *category*, which
    # is the right rule and would make a four-row ratio a tick list.
    wide = pd.DataFrame({"area": np.arange(1.0, 41.0),
                         "perimeter": np.arange(40.0, 0.0, -1.0)})
    computed, _ = compute(wide, [
        ColumnFormula("ratio", "area / (perimeter + 1)"),
        ColumnFormula("big", "area > 20"),
    ])
    # A boolean lands as a tick list, a ratio as a range -- so the filter
    # panel offers the right control for each without knowing they are derived.
    kinds = classify_columns(computed)
    assert kinds["big"] == "category"
    assert column_kinds(computed)["big"] == CATEGORICAL
    assert column_kinds(computed)["ratio"] == CONTINUOUS
    assert "ratio" in plottable_columns(computed)
    assert "big" in plottable_columns(computed)


def test_a_computed_column_can_be_filtered_and_plotted(frame):
    from spacr.selection import DataFilter, RangeFilter
    from spacr.qt.widgets.graph_spec import GraphSpec, facet_grid

    computed, _ = compute(frame, [ColumnFormula("half", "area / 2")])
    kept = DataFilter([RangeFilter("half", low=10.0)]).apply(computed)
    assert kept["area"].tolist() == [20.0, 30.0, 40.0]

    grid = facet_grid(computed, GraphSpec(x="half", facet_col="gene"))
    assert grid.shape == (1, 2)


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

@pytest.fixture
def panel(qtbot, frame):
    from spacr.qt.widgets.formula_editor import FormulaPanel
    widget = FormulaPanel()
    qtbot.addWidget(widget)
    widget.set_frame(frame)
    return widget


def type_formula(panel, name: str, expression: str) -> None:
    panel._name.setText(name)
    panel._expression.setText(expression)
    panel._validate()


def test_the_panel_adds_a_column_and_announces_it(panel, frame):
    seen = []
    panel.formulas_changed.connect(lambda: seen.append(1))
    type_formula(panel, "ratio", "area / perimeter ** 2")
    assert panel._add.isEnabled()
    assert panel.commit()
    assert seen == [1]
    assert panel.formulas().names == ("ratio",)
    assert panel.computed_frame()["ratio"].iloc[0] == pytest.approx(0.4)
    # the loaded frame is untouched
    assert "ratio" not in panel.frame().columns
    assert "ratio" not in frame.columns


def test_a_typo_disables_add_and_says_which_column(panel):
    type_formula(panel, "ratio", "area / perimter")
    assert not panel._add.isEnabled()
    assert "perimter" in panel.status()
    assert "perimeter" in panel.status()          # the suggestion


def test_the_preview_reports_the_infinities_before_the_column_is_added(panel):
    type_formula(panel, "ratio", "area / perimeter")
    assert "3 of 4 rows have a finite value" in panel.status()


def test_an_aggregate_formula_says_it_depends_on_the_table(panel):
    type_formula(panel, "z", "zscore(area)")
    assert "uses the whole table" in panel.status()


def test_removing_a_formula_removes_its_column(panel):
    type_formula(panel, "half", "area / 2")
    panel.commit()
    assert "half" in panel.computed_frame().columns
    panel.remove("half")
    assert "half" not in panel.computed_frame().columns
    assert panel.formulas().is_empty


def test_formulas_chain_in_the_panel(panel):
    type_formula(panel, "d", "count / area")
    panel.commit()
    type_formula(panel, "ld", "d * 10")
    assert panel._add.isEnabled()
    panel.commit()
    np.testing.assert_allclose(
        panel.computed_frame()["ld"].to_numpy(), [1.0] * 4)


def test_set_formulas_restores_a_saved_analysis(panel):
    saved = FormulaSet.from_json(
        FormulaSet([ColumnFormula("half", "area / 2")]).to_json())
    panel.set_formulas(saved.formulas)
    assert panel.computed_frame()["half"].tolist() == [5.0, 10.0, 15.0, 20.0]


def test_the_panel_survives_a_table_that_lacks_the_columns(panel):
    """A formula kept across a table change fails loudly, not silently."""
    type_formula(panel, "half", "area / 2")
    panel.commit()
    panel.set_frame(pd.DataFrame({"other": [1.0, 2.0]}))
    assert "no column called 'area'" in panel.status()
    # And the frame handed on is still usable — the original, without the
    # column that could not be computed.
    assert list(panel.computed_frame().columns) == ["other"]

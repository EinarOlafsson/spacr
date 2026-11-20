"""Formula and PCA refusals retain diagnostics and explain how to correct the input."""
import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets import formula, pca_model

pytestmark = pytest.mark.qt


@pytest.mark.parametrize("expression, detail, advice", [
    ("@", "position 1", "Replace the unsupported character"),
    ("``", "empty column name", "Put an existing column name between the backticks"),
    ("1 2", "expression already ended", "Remove the extra token"),
    ("(1 + 2", "never closed", "Add the closing parenthesis"),
    ("unknown(1)", "no function called unknown", "supported names listed above"),
    ("1 +", "formula ends early", "Complete the expression"),
])
def test_formula_syntax_errors_keep_location_and_next_step(expression, detail, advice):
    """Malformed formulas remain FormulaErrors with the original diagnostic and a correction."""
    with pytest.raises(formula.FormulaError) as caught:
        formula.parse(expression)
    message = str(caught.value)
    assert detail in message
    assert advice in message


@pytest.mark.parametrize("name, advice", [
    ("", "Enter a name for the new column"),
    ("1value", "Rename the new column to a bare name"),
    ("mean", "Choose a column name that is not a function or keyword"),
])
def test_computed_column_name_errors_offer_a_valid_name(name, advice):
    """Invalid names are refused when constructing the formula, before any table changes."""
    with pytest.raises(formula.FormulaError) as caught:
        formula.ColumnFormula(name, "1 + 2")
    assert advice in str(caught.value)


@pytest.mark.parametrize("values, detail, advice", [
    ({"a": [1.0], "b": [2.0]}, "at least two objects", "Load a table with at least two measured objects"),
    ({"a": [1.0, 1.0], "b": [2.0, 2.0]}, "every selected feature is constant", "Select measurement columns that vary"),
    ({"a": [1.0, 2.0], "b": [2.0, 2.0]}, "only 1 feature varies", "Select at least two features that vary"),
    ({"a": [1.0, np.nan], "b": [2.0, 3.0]}, "only 1 object has a value", "Select features measured in at least two objects"),
])
def test_pca_refusals_explain_data_requirements_without_mutation(values, detail, advice):
    """Insufficient or constant PCA data remains an error and the input frame stays intact."""
    frame = pd.DataFrame(values)
    before = frame.copy(deep=True)
    spec = pca_model.PCASpec(features=("a", "b"), nan_policy=pca_model.NAN_COMPLETE)
    with pytest.raises(pca_model.PCAError) as caught:
        pca_model.pca(frame, spec)
    message = str(caught.value)
    assert detail in message
    assert advice in message
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("settings, detail, advice", [
    ({"n_components": 0}, "n_components must be at least 1", "positive whole number"),
    ({"structural_missing": 2}, "structural_missing is a fraction", "use 0.5 for 50 percent"),
])
def test_pca_setting_errors_explain_valid_values(settings, detail, advice):
    """Invalid PCA settings keep their values in the error and explain the accepted range."""
    with pytest.raises(pca_model.PCAError) as caught:
        pca_model.PCASpec(**settings)
    message = str(caught.value)
    assert detail in message
    assert advice in message

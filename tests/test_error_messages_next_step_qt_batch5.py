"""Dose-response and outlier failures retain their guards and explain valid inputs."""
from types import SimpleNamespace

import pandas as pd
import pytest

from spacr.qt.widgets import dose_response as dose
from spacr.qt.widgets import outlier_model as outlier

pytestmark = pytest.mark.qt


@pytest.mark.parametrize("settings, advice", [
    ({"response": "signal"}, "Choose the concentration and response columns"),
    ({"concentration": "missing", "response": "signal"}, "column names shown above"),
    ({"concentration": "dose", "response": "signal", "group": "missing"}, "Choose an existing grouping column"),
])
def test_curve_column_errors_explain_selection_without_changing_data(settings, advice):
    """A missing fit column remains a DoseResponseError and leaves the loaded table intact."""
    frame = pd.DataFrame({"dose": [1.0, 2.0], "signal": [3.0, 4.0]})
    before = frame.copy(deep=True)
    with pytest.raises(dose.DoseResponseError) as caught:
        dose.fit_frame(frame, dose.DoseResponseSpec(**settings))
    assert advice in str(caught.value)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("concentrations, responses, detail, advice", [
    ([1, 2], [3], "pair up one to one", "Provide one measured response for each concentration"),
    ([1, 2], [3, 4], "distinct positive", "Use a measured series with at least the required number"),
    ([1, 2, 4, 8, 16, 32], [3] * 6, "every response is 3", "report that this series has no estimable midpoint"),
])
def test_curve_guards_explain_measurements_needed(concentrations, responses, detail, advice):
    """Unpaired, insufficient, and flat measurements are still refused before fitting."""
    with pytest.raises(dose.DoseResponseError) as caught:
        dose.fit_dose_response(concentrations, responses)
    message = str(caught.value)
    assert detail in message
    assert advice in message


@pytest.mark.parametrize("a, b, advice", [
    ([1, 2], [1, 2], "includes single-agent wells for each agent"),
    ([1, 0], [0, 1], "includes combination wells"),
])
def test_checkerboard_refusals_explain_missing_wells(a, b, advice):
    """Missing single-agent or combination wells remain errors without altering the input table."""
    frame = pd.DataFrame({"a": a, "b": b, "signal": [1.0, 2.0]})
    before = frame.copy(deep=True)
    with pytest.raises(dose.DoseResponseError) as caught:
        dose.checkerboard_from_frame(frame, dose_a="a", dose_b="b", response="signal")
    assert advice in str(caught.value)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("plate, control, advice", [
    ("", "condition", "Choose the column containing the plate identifier"),
    ("plate", "", "Choose the column containing the control labels"),
])
def test_plate_control_spec_explains_required_column(plate, control, advice):
    """Control normalization still requires both identifying columns before construction succeeds."""
    with pytest.raises(dose.DoseResponseError) as caught:
        dose.PlateSpec(plate=plate, control=control, positive=("pos",), negative=("neg",))
    assert advice in str(caught.value)


def test_empty_outlier_table_explains_loading_or_filter_review():
    """An empty object table remains an OutlierError and points to source or filter selection."""
    frame = pd.DataFrame({"area": []})
    with pytest.raises(outlier.OutlierError) as caught:
        outlier.detect_outliers(frame)
    message = str(caught.value)
    assert "no objects" in message
    assert "Load a nonempty measurement table" in message
    assert "filters that removed all rows" in message
    assert frame.empty


@pytest.mark.parametrize("settings, advice", [
    ({"k": 0}, "Enter a positive value for k"),
    ({"alpha": 2}, "using a fraction rather than a percentage"),
])
def test_outlier_threshold_errors_explain_valid_values(settings, advice):
    """Invalid outlier thresholds keep their validation error and explain the accepted units."""
    with pytest.raises(outlier.OutlierError) as caught:
        outlier.OutlierSpec(**settings)
    assert advice in str(caught.value)


def test_flags_for_another_table_require_new_detection():
    """A stale flag result cannot filter a different table and explains how to recompute it."""
    frame = pd.DataFrame({"area": [1.0, 2.0]})
    before = frame.copy(deep=True)
    result = SimpleNamespace(n_rows_in=3)
    with pytest.raises(outlier.OutlierError) as caught:
        outlier.OutlierResult.filtered(result, frame)
    message = str(caught.value)
    assert "computed on 3 objects" in message and "frame has 2" in message
    assert "Run outlier detection again on the current table" in message
    pd.testing.assert_frame_equal(frame, before)

"""Table-driven test of ``spacr.permutation_qc.position_effect``.

The position effect says how much residual variance one plate-position column
explains; it decides whether shuffling across positions is exchangeable.  Too
few finite wells is "not measurable" (NaN), a flat block or a single level is
"no effect", and unpaired inputs are refused with the lengths named.
"""
from __future__ import annotations

import math

import pytest

from spacr.permutation_qc import position_effect


def test_position_effect_measures_between_position_variance(subtests):
    """Exact eta/omega/p for real blocks, NaN when unmeasurable, refusals when unpaired."""
    cases = [
        ("two wells are unmeasurable", [1.0, 2.0], ["A", "B"],
         {"eta_squared": math.nan, "levels": 0.0}),
        ("non-finite residuals are dropped before the count",
         [1.0, float("nan"), float("inf"), 2.0], ["A", "B", "A", "B"],
         {"eta_squared": math.nan, "levels": 0.0}),
        ("a flat block has no effect", [1.0, 1.0, 1.0], ["A", "B", "C"],
         {"eta_squared": 0.0, "p_value": 1.0, "levels": 3.0}),
        ("a single level has no effect", [1.0, 2.0, 3.0], ["A", "A", "A"],
         {"eta_squared": 0.0, "p_value": 1.0, "levels": 1.0}),
        ("a strong row effect", [1.0, 1.2, 3.0, 3.2], ["A", "A", "B", "B"],
         {"eta_squared": 0.9900990099009901,
          "omega_squared": 0.980295566502463,
          "p_value": 0.004962809790010869, "levels": 2.0,
          "worst_level": "A", "worst_departure": -1.0}),
        ("one well per level leaves no within spread", [1.0, 2.0, 3.0],
         ["A", "B", "C"],
         {"eta_squared": 1.0, "omega_squared": 0.0, "p_value": 1.0,
          "levels": 3.0, "worst_level": "A", "worst_departure": -1.0}),
    ]
    for case_id, residuals, positions, expected in cases:
        with subtests.test(msg=case_id):
            result = position_effect(residuals, positions)
            assert set(result) == set(expected)
            for key, value in expected.items():
                if isinstance(value, float) and math.isnan(value):
                    assert math.isnan(result[key])
                elif isinstance(value, float):
                    assert result[key] == pytest.approx(value, rel=1e-12)
                else:
                    assert result[key] == value

    for positions in (["A"], ["A", "B", "C"]):
        with subtests.test(msg=f"unpaired: {len(positions)} positions"):
            with pytest.raises(ValueError) as caught:
                position_effect([1.0, 2.0], positions)
            assert str(caught.value) == (
                "residuals and positions must have the same length; got 2 "
                f"and {len(positions)}")

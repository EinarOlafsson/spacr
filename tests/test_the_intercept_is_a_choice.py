"""What the intercept of a screen regression is, as a setting.

A fitted intercept is the response where every predictor is zero, which on
a screen design is a well with no guide in it -- a point that does not
exist. That makes it hard to read, and it is why this was asked for
repeatedly. There are three answers now:

* ``fitted``  -- estimate it, which is what every run did before;
* ``zero``    -- no intercept, so the fit passes through the origin and a
                 coefficient is a whole predicted score;
* ``control`` -- centre the response on the negative controls first, so
                 the intercept IS the control level and a coefficient
                 reads as its distance from the controls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.ml import INTERCEPT_MODES, centre_on_controls, prepare_formula


def test_the_three_modes_are_the_ones_offered():
    assert INTERCEPT_MODES == ("fitted", "zero", "control")


@pytest.mark.parametrize("plate_position", [True, False])
def test_zero_takes_the_term_out_of_every_formula(plate_position):
    """Patsy's own suppression, on each of the shapes this builds."""
    fitted = prepare_formula("pred", level="grna", intercept="fitted",
                             model_plate_position=plate_position)
    origin = prepare_formula("pred", level="grna", intercept="zero",
                             model_plate_position=plate_position)

    assert not fitted.endswith("- 1")
    assert origin.endswith("- 1")
    assert origin.startswith(fitted)


def test_control_leaves_the_formula_alone():
    """It is a shift of the response, not a change of design -- the term
    stays so the fit still has an intercept to BE the control level."""
    assert (prepare_formula("pred", level="grna", intercept="control")
            == prepare_formula("pred", level="grna", intercept="fitted"))


def test_an_unknown_mode_is_refused_by_name():
    with pytest.raises(ValueError) as raised:
        prepare_formula("pred", intercept="middle")

    said = str(raised.value)
    assert "middle" in said
    for mode in INTERCEPT_MODES:
        assert mode in said


def test_centring_moves_the_controls_to_zero():
    frame = pd.DataFrame({
        "grna": ["nc_1", "nc_1", "nc_1", "g2", "g3"],
        "pred": [0.4, 0.6, 0.5, 2.0, 3.0],
    })

    centred, offset = centre_on_controls(frame, "pred", "nc_1")

    assert offset == pytest.approx(0.5)
    controls = centred.loc[centred["grna"] == "nc_1", "pred"]
    assert float(np.median(controls)) == pytest.approx(0.0)
    # And every other row moved by the same amount, so the DIFFERENCES the
    # regression fits are untouched.
    assert float(centred.loc[3, "pred"]) == pytest.approx(1.5)


def test_a_control_that_matches_nothing_changes_nothing():
    """Silently centring on an empty selection would shift by NaN; saying
    so and leaving the response alone is the honest answer."""
    frame = pd.DataFrame({"grna": ["g1", "g2"], "pred": [1.0, 2.0]})

    same, offset = centre_on_controls(frame, "pred", "nc_1")

    assert offset == 0.0
    assert same is frame


def test_it_finds_the_control_by_gene_as_well_as_by_guide():
    """`nc` is read as a gene when it is bare and a guide when it has an
    underscore, which is the rule the rest of the module applies."""
    frame = pd.DataFrame({
        "gene": ["220950", "220950", "other"],
        "pred": [1.0, 3.0, 10.0],
    })

    centred, offset = centre_on_controls(frame, "pred", "220950")

    assert offset == pytest.approx(2.0)
    assert float(centred.loc[2, "pred"]) == pytest.approx(8.0)


def test_the_regression_entry_point_takes_the_setting():
    """The whole point is that it can be set, so the parameter is there."""
    import inspect

    from spacr.ml import regression

    assert "intercept" in inspect.signature(regression).parameters

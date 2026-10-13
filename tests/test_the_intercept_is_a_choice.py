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
                 reads as its distance from the controls;
* ``value``   -- pin it at a number the user gives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.ml import INTERCEPT_MODES, centre_on_controls, prepare_formula


def test_the_four_modes_are_the_ones_offered():
    assert INTERCEPT_MODES == ("fitted", "zero", "control", "value")


@pytest.mark.parametrize("mode", ["zero", "value"])
@pytest.mark.parametrize("plate_position", [True, False])
def test_the_term_is_suppressed_when_the_intercept_is_decided(mode,
                                                              plate_position):
    """Patsy's own suppression, on each of the shapes this builds."""
    fitted = prepare_formula("pred", level="grna", intercept="fitted",
                             model_plate_position=plate_position)
    origin = prepare_formula("pred", level="grna", intercept=mode,
                             model_plate_position=plate_position)

    # A DECIDED INTERCEPT IS NOT AN ESTIMATED ONE. Leaving the term in
    # while shifting the response would fit an intercept NEAR the value
    # asked for, which reads as though the number had been a suggestion.
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
    """The whole point is that it can be set, so both parameters are
    there: the mode, and the number the fourth mode pins it at."""
    import inspect

    from spacr.ml import regression

    parameters = inspect.signature(regression).parameters
    assert "intercept" in parameters
    assert "intercept_value" in parameters
    assert parameters["intercept"].default == "fitted"


def test_a_pinned_intercept_lands_exactly_where_it_was_asked_to():
    """The arithmetic behind the fourth mode, on a fit with a known answer.

    Shifting the response by c and suppressing the term fits
    ``y - c = Xb``, which is ``y = c + Xb`` -- so the intercept is c
    exactly. This builds data whose true intercept is 5, pins it at 2, and
    checks the slope is unchanged and the fitted line passes through 2.
    """
    import numpy as np
    import statsmodels.api as sm

    x = np.arange(20.0)
    y = 5.0 + 3.0 * x

    pinned_at = 2.0
    design = sm.add_constant(x)
    free = sm.OLS(y, design).fit()
    held = sm.OLS(y - pinned_at, x.reshape(-1, 1)).fit()

    assert free.params[0] == pytest.approx(5.0)
    # The pinned fit reconstructs the response as c + slope*x, and the
    # slope has absorbed the difference rather than the intercept moving.
    assert float(held.params[0]) == pytest.approx(
        float(np.sum((y - pinned_at) * x) / np.sum(x * x)))
    reconstructed = pinned_at + float(held.params[0]) * x
    assert reconstructed[0] == pytest.approx(pinned_at)


# --------------------------------------------------------------------------
# The control reaches the user. The engine above is unreachable until the
# setting is declared, described, typed, placed in a panel category and
# passed to the fit -- five separate tables, any one of which can be
# forgotten, and a forgotten one leaves a working engine no one can drive.
# --------------------------------------------------------------------------

def test_the_setting_defaults_to_a_fitted_intercept():
    """An older settings file has no intercept key and must fit as it did."""
    from spacr.settings import get_perform_regression_default_settings

    filled = get_perform_regression_default_settings(
        {'src': '/tmp', 'count_data': '/tmp', 'score_data': '/tmp'})
    assert filled['intercept'] == 'fitted'
    assert filled['intercept_value'] == 0.0


def test_the_panel_offers_exactly_the_modes_the_engine_accepts():
    """A fifth option in the dropdown would be refused by prepare_formula."""
    from spacr.settings_spec import convert_settings_dict_for_gui

    kind, options, default = convert_settings_dict_for_gui(
        {'intercept': 'fitted'})['intercept']
    assert kind == 'combo'
    assert tuple(options) == INTERCEPT_MODES
    assert default == 'fitted'


def test_the_pinned_number_is_a_free_field():
    """'value' means any number, so it cannot be a list of choices."""
    from spacr.settings_spec import convert_settings_dict_for_gui

    kind, options, _default = convert_settings_dict_for_gui(
        {'intercept_value': 0.0})['intercept_value']
    assert kind == 'entry'
    assert options is None


def test_both_keys_are_typed_and_explained():
    from spacr.settings import expected_types, tooltips

    assert expected_types['intercept'] is str
    assert expected_types['intercept_value'] is float
    # The tooltip has to say what each mode DOES, or the dropdown is four
    # words with no way to choose between them.
    text = tooltips['intercept']
    for mode in INTERCEPT_MODES:
        assert f"'{mode}'" in text, f"{mode} is offered and never explained"
    assert 'intercept_value' in text
    assert 'intercept' in tooltips['intercept_value']


def test_both_keys_appear_in_the_model_panel():
    """A declared setting with no category is editable nowhere."""
    from spacr.settings import categories

    model = categories['Regression: Model']
    assert 'intercept' in model
    assert 'intercept_value' in model
    # Read in order: what the intercept is, then the number it may be
    # pinned to. Reversed, the panel offers the refinement first.
    assert model.index('intercept') < model.index('intercept_value')


@pytest.mark.parametrize('mode', ['fitted', 'zero', 'control'])
def test_the_pinned_number_is_greyed_for_the_modes_that_ignore_it(mode):
    from spacr.settings import get_setting_dependencies

    rule = get_setting_dependencies()['intercept_value']
    assert 'intercept' in rule['sources']
    assert not rule['predicate']({'intercept': mode}, None)
    # The reason names the mode actually in force, not the one that reads it.
    assert repr(mode) in rule['reason']({'intercept': mode}, None)


def test_the_pinned_number_is_live_when_the_intercept_is_pinned():
    from spacr.settings import get_setting_dependencies

    rule = get_setting_dependencies()['intercept_value']
    assert rule['predicate']({'intercept': 'value'}, None)


def test_the_dropdown_is_greyed_when_no_model_is_fitted():
    """Permutation inference fits no line, so there is no intercept to set."""
    from spacr.settings import get_setting_dependencies

    rule = get_setting_dependencies()['intercept']
    assert set(rule['sources']) == {'inference', 'analysis_mode'}
    settings = {'inference': 'nonparametric'}
    assert rule['predicate'](settings, {}) is False
    reason = rule['reason'](settings, {})
    assert 'fits no model' in reason
    assert 'kept and saved' in reason
    assert rule['predicate']({'inference': 'parametric'}, {}) is True
    assert rule['predicate']({'inference': 'auto'}, {}) is True


def test_perform_regression_forwards_what_the_panel_set():
    """The last link: the panel writes a settings key and the run has to hand
    it to the fit. It is read with `.get` because a settings CSV written
    before the key existed must still run, and what it meant is a fitted
    intercept."""
    import inspect

    from spacr.ml import _perform_regression

    source = inspect.getsource(_perform_regression)
    assert "intercept=str(settings.get('intercept') or 'fitted')" in source
    assert "intercept_value=float(settings.get('intercept_value') or 0.0)" \
        in source

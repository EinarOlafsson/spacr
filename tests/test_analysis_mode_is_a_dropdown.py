"""`analysis_mode` is a dropdown, and `level` sits beside what decides it.

Instruction 134, asked for on 2026-08-17: "analasys mode should be a dropdown".
Instruction 132's addendum, the same day: "level should be in model and
inference not additional settings".

Both are the same shape of bug. A setting whose entire domain is two strings
was a free-text box, and a setting whose enabled state is decided by
`regression_type` was drawn three sections away from it.
"""

import pytest

from spacr.settings import INFERENCE_MODES, get_setting_dependencies
from spacr.settings_spec import convert_settings_dict_for_gui


def test_analysis_mode_is_a_combo_with_exactly_its_two_values():
    spec = convert_settings_dict_for_gui({"analysis_mode": "regression"})
    kind, options, default = spec["analysis_mode"]
    assert kind == "combo"
    assert options == ["regression", "guide_permutation"]
    assert default == "regression"


def test_the_two_values_are_the_ones_inference_maps_onto():
    """The dropdown cannot offer a mode the resolver would refuse.

    `INFERENCE_MODES` is what turns the readable `inference` into
    `analysis_mode`, so its value set IS the domain. Reading it here rather
    than repeating the two strings is what stops the list drifting when a
    third mode is added.
    """
    _kind, options, _default = convert_settings_dict_for_gui(
        {"analysis_mode": "regression"})["analysis_mode"]
    assert set(options) == {mode for mode in INFERENCE_MODES.values() if mode}


def test_the_qt_front_end_offers_the_same_two():
    """One list, two GUIs. Two lists is how they start disagreeing."""
    from spacr.qt.screens.settings_model import _APP_COMBO_OPTIONS

    assert _APP_COMBO_OPTIONS["regression"]["analysis_mode"] == [
        "regression", "guide_permutation"]


@pytest.mark.parametrize("inference, selects", [
    ("nonparametric", "guide_permutation"),
    ("parametric", "regression"),
])
def test_analysis_mode_is_greyed_out_while_inference_decides_it(inference,
                                                                selects):
    """Instruction 106's rule: greyed out WITH the reason, never inert.

    A user who picks nonparametric and then reads a live `analysis_mode` box
    still saying `regression` is looking at two controls that contradict each
    other, and the one they can edit is the one that loses.
    """
    rule = get_setting_dependencies()["analysis_mode"]
    settings = {"inference": inference, "analysis_mode": "regression"}
    assert rule["sources"] == ("inference",)
    assert rule["predicate"](settings, {}) is False
    reason = rule["reason"](settings, {})
    assert inference in reason
    assert selects in reason
    assert "kept and saved" in reason


def test_choosing_auto_hands_the_control_back():
    rule = get_setting_dependencies()["analysis_mode"]
    assert rule["predicate"]({"inference": "auto"}, {}) is True


def test_an_absent_inference_leaves_the_control_live():
    """A settings dict that predates `inference` must not arrive greyed out."""
    rule = get_setting_dependencies()["analysis_mode"]
    assert rule["predicate"]({}, {}) is True
    assert rule["predicate"]({"inference": None}, {}) is True


# ---------------------------------------------------------------------------
# `level` moves out of the leftovers bucket
# ---------------------------------------------------------------------------

def _regression_sections():
    from spacr.qt.screens.settings_model import _APP_CATEGORY_SPECS

    return dict(_APP_CATEGORY_SPECS["regression"])


def test_level_is_in_model_and_inference():
    assert "level" in _regression_sections()["Model & Inference"]


def test_level_is_beside_the_setting_that_greys_it_out():
    """`regression_type='mixed'` is what disables `level`.

    Being in the same section is the whole point of the move: a control that
    goes grey when its neighbour changes explains itself; the same control
    three sections down reads as broken.
    """
    section = _regression_sections()["Model & Inference"]
    assert "regression_type" in section
    assert get_setting_dependencies()["level"]["sources"][0] == \
        "regression_type"


def test_level_is_in_exactly_one_section():
    """Two sections would draw it twice, each with its own widget."""
    from spacr.qt.screens.settings_model import _APP_CATEGORY_SPECS

    holding = [name for name, keys in _APP_CATEGORY_SPECS["regression"]
               if "level" in keys]
    assert holding == ["Model & Inference"]


def test_level_still_means_object_well_plate_for_the_proportion_plots():
    """ONE KEY, TWO MODULES, and the move must not have crossed them.

    `level` is the regression's grna/gene/both AND the proportion plots'
    object/well/plate. The spec dispatches on the VALUE, so moving the
    regression layout entry cannot reach the other module -- checked rather
    than assumed.
    """
    spec = convert_settings_dict_for_gui({"level": "object"})
    kind, options, default = spec["level"]
    assert kind == "combo"
    assert options == ["object", "well", "plate"]
    assert default == "object"

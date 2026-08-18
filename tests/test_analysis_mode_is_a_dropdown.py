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


# ---------------------------------------------------------------------------
# Instruction 135: the sections the maintainer asked for
# ---------------------------------------------------------------------------

def test_controls_and_filters_is_one_section():
    """Asked for on 2026-08-17: "merge quality & filters in here".

    Controls and filters are one question -- which rows reach the model --
    and they were two sections with the response, the estimator and the
    hit-calling rules between them.
    """
    sections = _regression_sections()
    assert "Controls & Filters" in sections
    assert "Controls & Plate Design" not in sections
    assert "Quality Filters" not in sections
    for key in ("min_cell_count", "min_n", "fraction_threshold",
                "outlier_detection", "filter_column", "negative_control"):
        assert key in sections["Controls & Filters"], key


def test_significance_merged_into_model_and_inference():
    """"merge all of these settings into Model and inference"."""
    sections = _regression_sections()
    assert "Significance & Hit Calling" not in sections
    for key in ("multiple_testing_method", "fdr_alpha", "threshold_method",
                "threshold_multiplier", "toxo"):
        assert key in sections["Model & Inference"], key


def test_cov_type_moved_to_estimator_tuning():
    """"mooveCov type here".

    It is estimator-specific in exactly the way the rest of that section is:
    the penalised, robust and quantile fits have no such estimator and refuse
    it rather than reporting ordinary errors under a robust label.
    """
    sections = _regression_sections()
    assert "cov_type" in sections["Estimator Tuning"]
    assert "cov_type" not in sections["Model & Inference"]


def test_level_is_directly_under_regression_type():
    """"moove level from aditional settings to right under Regression type".

    Adjacency, not just membership. `regression_type='mixed'` is what greys
    `level` out, and a control that goes grey when the row above it changes
    explains itself.
    """
    section = list(_regression_sections()["Model & Inference"])
    assert section[section.index("regression_type") + 1] == "level"


def test_no_setting_was_dropped_by_the_regroup():
    """Merging sections must not lose a key into Additional Settings."""
    sections = _regression_sections()
    everywhere = {key for keys in sections.values() for key in keys}
    for key in ("min_cell_count", "outlier_detection", "cov_type",
                "multiple_testing_method", "fdr_alpha", "toxo",
                "threshold_method", "threshold_multiplier", "tolerance",
                "target_unique_count"):
        assert key in everywhere, key


def test_a_key_is_named_in_exactly_one_section():
    """Two sections would draw the same setting twice, each with a widget."""
    from collections import Counter
    from spacr.qt.screens.settings_model import _APP_CATEGORY_SPECS

    counts = Counter(key for _name, keys in _APP_CATEGORY_SPECS["regression"]
                     for key in keys)
    assert [k for k, n in counts.items() if n > 1] == []


# ---------------------------------------------------------------------------
# Instruction 135: every estimator tooltip names its family
# ---------------------------------------------------------------------------

def _owned():
    from spacr.regression_spec import REGRESSION_SETTINGS_USED

    families = {}
    for family, keys in REGRESSION_SETTINGS_USED.items():
        for key in keys:
            families.setdefault(key, []).append(family)
    return families


def test_every_estimator_setting_names_the_family_that_reads_it():
    """"make sure the tooltips make apparent which regression type each
    setting here is for"."""
    from spacr.settings import tooltips

    for key, owners in _owned().items():
        text = tooltips[key]
        assert "regression_type" in text, key
        for family in owners:
            assert f"'{family}'" in text, (key, family)


def test_the_sentence_is_generated_from_the_same_table_as_the_greying():
    """One table, so the tooltip and the enabled state cannot disagree.

    A hand-written list drifts the first time a family is added; this reads
    REGRESSION_SETTINGS_USED, which is what the dependency rule is built from
    too.
    """
    from spacr.settings import get_setting_dependencies, tooltips

    deps = get_setting_dependencies()
    for key, owners in _owned().items():
        assert deps[key]["sources"] == ("regression_type",), key
        for family in owners:
            assert deps[key]["predicate"]({"regression_type": family}, {}), \
                (key, family)
        assert not deps[key]["predicate"]({"regression_type": "ols"}, {}) \
            or "ols" in owners


def test_a_tooltip_that_already_named_its_family_is_not_told_twice():
    """The project's existing "Read only by regression_type 'x'." convention
    answers the same question in the same place; a second sentence saying it
    again reads as a mistake."""
    from spacr.settings import tooltips

    for key in ("l1_ratio", "quantile", "huber_t", "hinge_threshold"):
        assert "Read only by regression_type" in tooltips[key], key
        assert "Read by regression_type" not in tooltips[key], key


def test_the_generated_sentence_is_added_once():
    """Running the generator again must not append it twice.

    Called directly rather than by reloading the module: `spacr.settings`
    registers categories as a side effect of import, and reloading it drops
    a registration that another test then finds missing. A test that breaks
    a different test is not a test of anything.
    """
    import spacr.settings as S

    S._name_the_family_in_every_estimator_tooltip()
    S._name_the_family_in_every_estimator_tooltip()
    for key in _owned():
        assert S.tooltips[key].count("Read by regression_type") <= 1, key


# ---------------------------------------------------------------------------
# The regression-type combo is the inventory, not a hand-written list
# ---------------------------------------------------------------------------

def _regression_type_options():
    from spacr.settings_spec import convert_settings_dict_for_gui

    _kind, options, _default = convert_settings_dict_for_gui(
        {"regression_type": "mixed"})["regression_type"]
    return options


def test_the_combo_offers_no_family_that_raises():
    """`gls` was on the hand-written list and is in
    UNSUPPORTED_REGRESSION_TYPES, so the Tk panel could pick a type that
    fails after the whole database had been read."""
    from spacr.regression_spec import UNSUPPORTED_REGRESSION_TYPES

    offered = set(_regression_type_options())
    assert not offered & set(UNSUPPORTED_REGRESSION_TYPES)
    assert "gls" not in offered


def test_the_combo_offers_every_family_that_fits():
    """Six were missing: huber, beta, quasi_binomial, elasticnet, hinge and
    horseshoe -- a third of the inventory, unreachable from the Tk panel."""
    from spacr.regression_spec import (REGRESSION_TYPES,
                                       UNSUPPORTED_REGRESSION_TYPES)

    expected = set(REGRESSION_TYPES) - set(UNSUPPORTED_REGRESSION_TYPES)
    assert set(_regression_type_options()) == expected
    for family in ("huber", "beta", "quasi_binomial", "elasticnet", "hinge",
                   "horseshoe"):
        assert family in _regression_type_options(), family


def test_mixed_is_first_and_the_default():
    """It answers the most central question, so it leads the list."""
    from spacr.settings_spec import convert_settings_dict_for_gui

    _kind, options, default = convert_settings_dict_for_gui(
        {"regression_type": "mixed"})["regression_type"]
    assert options[0] == "mixed"
    assert default == "mixed"
    # The rest alphabetically, so a family added to the inventory lands
    # somewhere predictable rather than at the end.
    assert options[1:] == sorted(options[1:])

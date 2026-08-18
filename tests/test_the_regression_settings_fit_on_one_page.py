"""Instruction 135, the settings half: what the regression panel still asks.

The maintainer walked every category of the regression panel on 2026-08-17
and the settings-side result is four changes, each of which is a claim that
can be checked:

* `score_column` was a second name for `dependent_variable` and is gone from
  the regression module, with an old settings CSV migrated rather than
  refused;
* the run now says what "significant" means -- the alpha AND whether the cut
  is on the raw or the adjusted P -- instead of leaving it to a right-click
  on the volcano;
* the plot-styling settings stop being settings, and `regression_qc` does NOT,
  for a measured reason;
* the settings the other halves of 135 read are declared here, because a key
  no defaults factory produces cannot be reached from Tk, Qt or the CLI.

Every assertion drives the real function; none of them reads a comment.
"""

import pytest

import spacr.settings as S
from spacr.settings_spec import convert_settings_dict_for_gui


def defaults(**given):
    """The regression defaults, built exactly as all three dispatchers do."""
    return S.get_perform_regression_default_settings(dict(given))


# ---------------------------------------------------------------------------
# A. one column, not two
# ---------------------------------------------------------------------------

def test_the_regression_no_longer_offers_a_second_name_for_the_response():
    assert "score_column" not in defaults()


def test_an_old_csv_with_only_score_column_still_names_the_response():
    """The pre-135 spelling of "fit this column", carried forward.

    A settings file written by the CLI before today can carry `score_column`
    without `dependent_variable`: `resolve_settings` layers the file OVER the
    defaults, so the key that survives is whichever the file wrote.
    """
    assert defaults(score_column="pathogen_area")["dependent_variable"] == \
        "pathogen_area"


def test_the_two_agreeing_is_the_ordinary_case_and_says_nothing(capsys):
    settings = defaults(score_column="pred", dependent_variable="pred")
    assert settings["dependent_variable"] == "pred"
    assert "score_column" not in capsys.readouterr().out


def test_a_disagreement_is_named_because_it_changed_what_the_old_run_did(
        capsys):
    settings = defaults(score_column="nucleus_area",
                        dependent_variable="pred")
    assert settings["dependent_variable"] == "pred"
    said = capsys.readouterr().out
    assert "nucleus_area" in said and "pred" in said


def test_the_key_itself_is_not_retired_because_explain_cv_still_uses_it():
    """The boundary in instruction 135 A: one name, two modules.

    `interpret_vision_model` and `hit_investigation` mean the CNN score
    column by `score_column`. Retiring the KEY rather than the regression's
    duplicate would take that with it.
    """
    assert S.set_interpret_vision_model_defaults({})["score_column"] == \
        "cv_predictions"
    assert S.expected_types["score_column"] is str
    assert "cv_predictions" in S.tooltips["score_column"]
    homes = [c for c, keys in S.categories.items() if "score_column" in keys]
    assert homes == ["Model Evaluation"], homes


# ---------------------------------------------------------------------------
# B. the run says what "significant" meant
# ---------------------------------------------------------------------------

def test_the_alpha_and_the_kind_of_p_are_settings_now():
    settings = defaults()
    assert settings["p_threshold_alpha"] == 0.05
    assert settings["p_threshold_kind"] == "adjusted"


def test_the_kind_of_p_is_a_dropdown_of_the_two_that_exist():
    """A free-text box here lets a CSV say 'bh' and mean nothing."""
    spec = convert_settings_dict_for_gui(defaults())["p_threshold_kind"]
    assert spec == ("combo", ["adjusted", "raw"], "adjusted")


def test_the_raw_choice_is_accepted_as_readily_as_the_adjusted_one():
    assert defaults(p_threshold_kind="raw")["p_threshold_kind"] == "raw"


@pytest.mark.parametrize("given", ["Adjusted", "RAW"])
def test_the_case_a_user_types_is_accepted(given):
    assert defaults(p_threshold_kind=given)["p_threshold_kind"] == given


@pytest.mark.parametrize("bad", ["bh", "", "fdr", None, 1])
def test_a_p_that_is_neither_raw_nor_adjusted_is_refused_not_defaulted(bad):
    with pytest.raises(ValueError) as raised:
        defaults(p_threshold_kind=bad)
    assert "'adjusted' or 'raw'" in str(raised.value)


@pytest.mark.parametrize("bad", [5, 0, 1, -0.01, 1.5])
def test_an_alpha_that_is_not_a_probability_is_refused(bad):
    """`p_threshold_alpha=5` for "5%" calls every coefficient a hit."""
    with pytest.raises(ValueError) as raised:
        defaults(p_threshold_alpha=bad)
    assert "p_threshold_alpha" in str(raised.value)


@pytest.mark.parametrize("bad", ["0.05", None, True])
def test_an_alpha_that_is_not_a_number_is_refused(bad):
    with pytest.raises(ValueError) as raised:
        defaults(p_threshold_alpha=bad)
    assert "must be a number" in str(raised.value)


# ---------------------------------------------------------------------------
# C. the plot settings stop being settings
# ---------------------------------------------------------------------------

RETIRED_PLOT_SETTINGS = ("log_x", "log_y", "x_lim", "y_lims",
                         "split_axis_lims", "guide_permutation_plot")


@pytest.mark.parametrize("key", RETIRED_PLOT_SETTINGS)
def test_the_plot_styling_settings_are_no_longer_offered(key):
    assert key not in defaults()


@pytest.mark.parametrize("key, was", [("log_x", True), ("log_y", True),
                                      ("x_lim", [0.0, 1.0]),
                                      ("y_lims", [0, 8]),
                                      ("guide_permutation_plot", False)])
def test_an_old_csv_still_loads_and_is_told_what_replaced_its_value(
        key, was, capsys):
    """Accepted and dropped, like `volcano`: every regression settings CSV
    written before today carries all six, and a file that suddenly fails to
    load is worse than a key nothing reads."""
    settings = defaults(**{key: was})
    assert key not in settings
    said = capsys.readouterr().out
    assert key in said, said


@pytest.mark.parametrize("key, agreed", [("log_x", False), ("log_y", False),
                                         ("x_lim", None), ("y_lims", None),
                                         ("split_axis_lims", ""),
                                         ("guide_permutation_plot", True)])
def test_a_value_that_already_agreed_is_dropped_in_silence(key, agreed,
                                                           capsys):
    assert key not in defaults(**{key: agreed})
    assert key not in capsys.readouterr().out


def test_the_retired_plot_keys_are_gone_from_every_regression_category():
    """A category naming a key the module no longer offers is a control the
    panel is asked for and cannot build."""
    regression_categories = {c: keys for c, keys in S.categories.items()
                             if c.startswith("Regression")}
    listed = {k for keys in regression_categories.values() for k in keys}
    assert not listed & set(RETIRED_PLOT_SETTINGS)


def test_guide_permutation_plot_is_not_declared_anywhere_any_more():
    """Hard-coded True, so it is not a setting in any of the three senses:
    no type, no tooltip, no category. `ml` reads it as
    `settings.get('guide_permutation_plot', True)`, which is True once the
    key is gone."""
    assert "guide_permutation_plot" not in S.expected_types
    assert "guide_permutation_plot" not in S.tooltips
    assert not [c for c, keys in S.categories.items()
                if "guide_permutation_plot" in keys]


# ---------------------------------------------------------------------------
# D. regression_qc is the exception, and the sweep is why
# ---------------------------------------------------------------------------

def test_the_qc_suite_is_on_for_a_single_analysis():
    assert defaults()["regression_qc"] is True


def test_hard_coding_the_qc_suite_would_have_cost_the_sweep_its_escape():
    """MEASURED, not assumed. `parameter_sweep._trial_settings` turns the QC
    suite off in the TRIAL'S SETTINGS DICT, and `perform_regression` re-applies
    this very function to that dict before reading it. Forcing True here would
    overwrite the sweep's False and put ~5.8 s and ~19 figures back on every
    one of a hundred trials, so the parameter stays and only the GUI control
    goes."""
    assert defaults(regression_qc=False)["regression_qc"] is False


def test_the_sweep_turns_it_off_through_the_settings_dict(tmp_path):
    """And the defaults function does not take it back, which is the whole
    reason `regression_qc` may not be hard-coded here: `perform_regression`
    re-applies these defaults to the trial's dict before reading it."""
    from spacr.parameter_sweep import _trial_settings

    base = defaults()
    base.pop("regression_qc")
    settings, _folder = _trial_settings(
        base, {"trial_id": 1, "alpha": 1}, str(tmp_path))
    assert settings["regression_qc"] is False
    assert defaults(**settings)["regression_qc"] is False


def test_a_base_dict_that_already_carries_the_key_defeats_that_setdefault(
        tmp_path):
    """MEASURED 2026-08-18, and REPORTED rather than fixed here because
    `spacr/parameter_sweep.py` is not this slice's file.

    `_trial_settings` says `settings.setdefault("regression_qc", False)`,
    which does nothing when the base dict already has the key -- and every
    base dict built by the Tk panel, the Qt panel or `spacr-run` has it,
    because all three build from `get_perform_regression_default_settings`
    and it defaults to True. So the sweep pays ~5.8 s and ~19 figures per
    trial exactly when it is driven from the application. The fix belongs in
    parameter_sweep (assign False unless the caller asked for diagnostics),
    not here: taking the key out of the defaults would remove the only way
    the sweep has of saying no at all.

    Pinned so the day parameter_sweep is fixed, this says so."""
    from spacr.parameter_sweep import _trial_settings

    settings, _folder = _trial_settings(
        defaults(), {"trial_id": 2, "alpha": 1}, str(tmp_path))
    assert settings["regression_qc"] is True


# ---------------------------------------------------------------------------
# E. the settings the other halves of 135 read
# ---------------------------------------------------------------------------

DECLARED_FOR_OTHER_SLICES = {
    "group_lasso_lambda": 0.05,
    "rra_alpha": 0.25,
    "rra_permutations": 10000,
    "count_grna_column": "grna",
    "count_value_column": "count",
    "p_threshold_alpha": 0.05,
    "p_threshold_kind": "adjusted",
}


@pytest.mark.parametrize("key, value", sorted(DECLARED_FOR_OTHER_SLICES.items()))
def test_each_new_setting_is_produced_typed_tooltipped_and_categorised(
        key, value):
    """All four, because three of the four is still a setting no panel can
    render: a key with no default reaches no dispatcher, a key with no
    expected_types entry is DROPPED by check_settings, a key with no tooltip
    hovers blank and an uncategorised key is dumped in "Other"."""
    assert defaults()[key] == value
    assert key in S.expected_types
    assert isinstance(value, S.expected_types[key])
    assert [c for c, keys in S.categories.items() if key in keys]
    tip = S.tooltips[key]
    assert tip.startswith("("), tip
    assert 80 <= len(tip) <= 600, len(tip)
    assert "efault" in tip


def test_the_count_columns_name_what_ml_hard_coded():
    """spacr.ml requires ['rowID', 'columnID', 'grna', 'count'] of the count
    CSV and names none of the columns the file HAS when they are missing."""
    settings = defaults()
    assert settings["count_grna_column"] == "grna"
    assert settings["count_value_column"] == "count"
    assert defaults(count_grna_column="sgRNA")["count_grna_column"] == "sgRNA"


@pytest.mark.parametrize("key", ["count_grna_column", "count_value_column"])
@pytest.mark.parametrize("bad", ["", "   ", None, 3])
def test_a_count_column_that_cannot_be_looked_up_is_refused(key, bad):
    with pytest.raises(ValueError) as raised:
        defaults(**{key: bad})
    assert key in str(raised.value)


@pytest.mark.parametrize("bad", [0, -0.5, 1.5, 25, "0.25", True])
def test_an_rra_alpha_outside_the_top_fraction_is_refused(bad):
    """`rra_alpha=25` for "the top 25%" scores the whole ranking."""
    with pytest.raises(ValueError):
        defaults(rra_alpha=bad)


def test_the_whole_ranking_is_a_legal_rra_alpha():
    assert defaults(rra_alpha=1)["rra_alpha"] == 1


@pytest.mark.parametrize("bad", [0, -1, 1.5, "10000", True])
def test_a_permutation_count_that_cannot_build_a_null_is_refused(bad):
    with pytest.raises(ValueError) as raised:
        defaults(rra_permutations=bad)
    assert "positive integer" in str(raised.value)


@pytest.mark.parametrize("bad", [-0.1, "0.05", None, True])
def test_a_penalty_that_is_not_a_penalty_is_refused(bad):
    with pytest.raises(ValueError) as raised:
        defaults(group_lasso_lambda=bad)
    assert "group_lasso_lambda" in str(raised.value)


def test_an_unpenalised_group_lasso_is_allowed():
    assert defaults(group_lasso_lambda=0)["group_lasso_lambda"] == 0


@pytest.mark.parametrize("key, family", [
    ("group_lasso_lambda", "group_lasso"),
    ("rra_alpha", "rra"),
    ("rra_permutations", "rra"),
])
def test_each_new_estimator_tooltip_names_the_family_that_reads_it(key,
                                                                   family):
    """Instruction 135's Estimator Tuning rule -- "make sure the tooltips
    make apparent which regression type each setting here is for" -- and it
    is GENERATED, not typed: `_name_the_family_in_every_estimator_tooltip`
    reads `regression_spec.REGRESSION_SETTINGS_USED`, which is the same table
    the greying rule comes from, so the sentence and the enabled state cannot
    disagree."""
    assert f"regression_type '{family}'" in S.tooltips[key]


@pytest.mark.parametrize("key", ["group_lasso_lambda", "rra_alpha",
                                 "rra_permutations"])
def test_the_new_defaults_are_the_ones_the_spec_calls_untouched(key):
    """`_reject_unused_settings` compares a posted value against
    `_MODEL_LEVEL_DEFAULTS` to tell "the user asked for this" from "the panel
    posted its default". A default here that differed from the one there
    would make every fresh panel look like a deliberate request for a knob
    the chosen family does not read, and the run would be refused."""
    from spacr.regression_spec import _MODEL_LEVEL_DEFAULTS

    if key not in _MODEL_LEVEL_DEFAULTS:
        pytest.skip("regression_spec does not carry this default yet")
    assert defaults()[key] == _MODEL_LEVEL_DEFAULTS[key]


def test_the_group_lasso_penalty_is_greyed_out_until_that_family_is_chosen():
    rules = S.get_setting_dependencies()
    rule = rules["group_lasso_lambda"]
    assert "regression_type" in rule["sources"]
    assert not rule["predicate"]({"regression_type": "mixed"}, {})
    assert rule["predicate"]({"regression_type": "group_lasso"}, {})
    assert "group_lasso_lambda" in rule["reason"]({"regression_type": "mixed"},
                                                  {})


def test_the_generated_family_rule_wins_when_the_spec_declares_the_family():
    """`setdefault`, so the hand-written rule is a placeholder: the loop over
    REGRESSION_SETTINGS_USED writes the same rule for every key the spec
    claims, and it must not be overwritten by this one."""
    from spacr.regression_spec import REGRESSION_SETTINGS_USED

    generated = {k for keys in REGRESSION_SETTINGS_USED.values() for k in keys}
    if "group_lasso_lambda" not in generated:
        pytest.skip("regression_spec does not declare the family yet")
    rules = S.get_setting_dependencies()
    assert rules["group_lasso_lambda"]["sources"] == ("regression_type",)


# ---------------------------------------------------------------------------
# F. runtime and reliability is not a regression question
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["strict_errors", "max_failure_rate",
                                 "verbose", "random_seed"])
def test_the_run_controls_are_filed_outside_the_regression_categories(key):
    """Instruction 135: "Runtime and reliability should be removed and go to
    prefgerences/general". In the SHARED map they already are -- the heading
    the maintainer saw is the Qt per-app layout's, not this one -- so the
    check that matters here is that no regression heading claims them."""
    homes = [c for c, keys in S.categories.items() if key in keys]
    assert homes, f"{key} is uncategorised"
    assert not any(c.startswith("Regression") for c in homes), homes


def test_the_run_controls_are_still_supplied_to_the_run():
    """Moving the CONTROL is not removing the SETTING: perform_regression
    indexes `verbose` and reads `strict_errors`, and a regression that could
    not start on a missing `verbose` is the failure this defaults function
    was written to end."""
    settings = defaults()
    assert settings["verbose"] is False
    assert settings["strict_errors"] is None
    assert settings["max_failure_rate"] is None


# ---------------------------------------------------------------------------
# G. the other migration in this function, restructured beside the new one
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("old_value", [True, False])
def test_the_toxo_rename_still_migrates_now_that_the_pop_does_the_reading(
        old_value):
    """`toxo` -> `Toxoplasma` (instruction 133) used to read
    `settings['toxo']` behind an `in` guard, which the contract test that
    walks perform_regression's helpers reads as a key needing a default --
    and this function exists to REMOVE that key. One pop says the same thing
    and answers it. What a run does is unchanged, which is what these check."""
    settings = defaults(toxo=old_value)
    assert settings["Toxoplasma"] is old_value
    assert "toxo" not in settings


def test_an_explicit_new_spelling_wins_over_the_old_one():
    settings = defaults(toxo=False, Toxoplasma=True)
    assert settings["Toxoplasma"] is True
    assert "toxo" not in settings

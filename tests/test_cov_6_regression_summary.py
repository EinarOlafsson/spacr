"""Every place the run summary would rather say nothing than guess.

The contract this module enforces is that no field is ever blank and no zero
ever stands in for "not checked". That guarantee is only as good as its
failure paths, and those are what this file drives: a value that is not a
number, a table that cannot be read, an import that is not there, a fit that
kept no design, a helper module that raises. In each case the summary has to
come back with a *sentence*, and the field it belongs to still has to be
present.

The section builders are driven through :class:`spacr.regression_summary._Run`
directly. That is the object ``build_run_summary`` assembles and hands them,
so building one is describing a run -- there is no shorter way to state "this
fit exposes no residual vector" than to write down a fit that does not.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from spacr import regression_summary as RS
from spacr.regression_summary import (
    COMPUTED, NOT_APPLICABLE, RunSummary, SummaryField, SummarySection,
)


def _run(**over):
    """A ``_Run`` with the least a builder needs, overridden per test."""
    base = dict(
        res_folder=None, model=None, settings={}, coef_df=None,
        regression_type="ols", nonparametric=False, penalised=False,
        data=None, data_note="no run folder was given", metrics={},
    )
    base.update(over)
    return RS._Run(**base)


# ---------------------------------------------------------------------------
# SummaryField: the type that refuses a blank
# ---------------------------------------------------------------------------

def test_a_field_with_an_unknown_kind_is_refused():
    """Only the three vocabularies render; a fourth would print as nothing."""
    with pytest.raises(ValueError) as excinfo:
        SummaryField("r_squared", "R2", value="0.4", kind="probably")
    assert "unknown kind" in str(excinfo.value)
    assert "probably" in str(excinfo.value)


def test_a_field_with_both_a_value_and_a_reason_is_refused():
    """Two answers to one question is the blank this type exists to refuse."""
    with pytest.raises(ValueError):
        SummaryField("r_squared", "R2", value="0.4", reason="no model")


# ---------------------------------------------------------------------------
# RunSummary lookups and the contract check
# ---------------------------------------------------------------------------

def test_asking_for_a_field_no_section_carries_gives_none():
    """The lookup walks every section; falling off the end is not an error."""
    summary = RunSummary(sections=[SummarySection(
        name="fitted", title="Fitted",
        fields=[SummaryField("regression_type", "Family", value="ols")])])
    assert summary.field("regression_type") is not None
    assert summary.field("r_squared") is None


def test_a_summary_with_no_sections_reports_the_whole_contract_missing():
    """``missing()`` is the whole-product assertion; it must see an absence."""
    gaps = RunSummary(sections=[]).missing()
    expected = sum(len(names) for names in RS.CONTRACT.values())
    assert len(gaps) == expected
    assert all("." in gap for gap in gaps)


def test_a_section_that_is_present_but_short_reports_only_its_own_gaps():
    """A half-built section must not be reported as an absent one."""
    name = next(iter(RS.CONTRACT))
    first = RS.CONTRACT[name][0]
    summary = RunSummary(sections=[SummarySection(
        name=name, title="whatever",
        fields=[SummaryField(first, RS.LABELS[(name, first)], value="x")])])
    gaps = summary.missing()
    assert f"{name}.{first}" not in gaps
    assert f"{name}.{RS.CONTRACT[name][1]}" in gaps


# ---------------------------------------------------------------------------
# The small formatters
# ---------------------------------------------------------------------------

def test_a_non_finite_raw_value_is_absent_not_the_string_nan():
    """``str(nan)`` is 'nan', and a field printing 'nan' looks like a result."""
    assert RS._raw(float("nan")) is None
    assert RS._raw(float("inf")) is None
    assert RS._raw("none") == "none", "'none' is a real correction method"


def test_a_non_finite_clean_value_is_absent():
    """``_clean`` drops every spelling of absent, the float one included."""
    assert RS._clean(float("nan")) is None
    assert RS._clean("NaN") is None
    assert RS._clean("0.4") == "0.4"


def test_a_non_finite_count_is_absent_rather_than_a_huge_integer():
    """``int(nan)`` raises and ``int(inf)`` overflows; neither is a count."""
    assert RS._count(float("nan")) is None
    assert RS._count(float("inf")) is None
    assert RS._count("12") == 12


def test_a_non_finite_number_has_no_human_form():
    """A field is either a number or a reason; 'inf' is neither."""
    assert RS._number(float("nan")) is None
    assert RS._number(float("-inf")) is None
    assert RS._number(0.5) == "0.5"


def test_floats_of_a_column_that_is_not_there_is_an_empty_array():
    """Callers filter and count the result; ``None`` would need a guard each."""
    frame = pd.DataFrame({"p_value": [0.1, 0.2]})
    assert RS._floats(frame, None).size == 0
    assert RS._floats(None, "p_value").size == 0
    assert RS._floats(frame, "p_value").tolist() == [0.1, 0.2]


# ---------------------------------------------------------------------------
# Imports that may not be there
# ---------------------------------------------------------------------------

def test_the_penalised_family_list_falls_back_when_the_spec_is_unreadable(
        monkeypatch):
    """The three penalised families are named here so the fallback still works.

    ``_penalised_types`` decides whether a run is ranked by P values or by
    bootstrap selection frequency. If the spec module cannot be read, guessing
    "not penalised" would report a lasso run's absent P values as a defect
    rather than as the property of the family.
    """
    import spacr.regression_spec as spec

    monkeypatch.delattr(spec, "NO_P_VALUE_TYPES")
    types = RS._penalised_types()
    for family in ("lasso", "elasticnet", "group_lasso"):
        assert family in types


def test_the_metrics_collector_returns_nothing_when_trial_metrics_is_absent(
        monkeypatch):
    """Every metric is optional; losing the module must lose only the metrics.

    Each field then falls through to its own "not applicable" reason, which is
    the contract. Raising here would lose the whole summary to a module that
    only ever contributes numbers.
    """
    monkeypatch.setitem(sys.modules, "spacr.trial_metrics", None)
    assert RS._collect_metrics(object(), pd.DataFrame(), {}) == {}


def test_the_summary_filename_falls_back_when_ml_cannot_be_read(monkeypatch):
    """The readers accept the fallback name, so a write still lands somewhere."""
    import spacr.ml

    monkeypatch.delattr(spacr.ml, "SUMMARY_FILENAME")
    assert RS._summary_filename() == RS._FALLBACK_SUMMARY_FILENAME


def test_the_hyperparameter_report_says_so_when_the_family_table_is_gone(
        monkeypatch):
    """The field is still answered -- with a reason, which is the contract."""
    import spacr.regression_spec as spec

    monkeypatch.delattr(spec, "REGRESSION_SETTINGS_USED")
    report = RS._hyperparameter_report("ridge", {}, _run())
    assert "reason" in report
    assert "family table" in report["reason"]


# ---------------------------------------------------------------------------
# _read_fitted_table
# ---------------------------------------------------------------------------

def test_a_regression_data_csv_that_cannot_be_read_is_a_note_not_a_crash(
        tmp_path):
    """The four counts it feeds go missing; the rest of the summary must not."""
    folder = tmp_path / "run"
    folder.mkdir()
    path = folder / "regression_data.csv"
    path.write_bytes(b"\x00\x01\x02")
    os.chmod(path, 0o000)
    try:
        frame, note = RS._read_fitted_table(str(folder))
    finally:
        os.chmod(path, 0o644)
    assert frame is None
    assert "could not be read" in note
    assert str(path) in note


def test_a_run_folder_with_no_table_says_which_path_it_looked_at(tmp_path):
    """The note is what a user reads to find out where to put the file."""
    frame, note = RS._read_fitted_table(str(tmp_path))
    assert frame is None
    assert note.endswith("does not exist")


# ---------------------------------------------------------------------------
# _no_residual_reason
# ---------------------------------------------------------------------------

class _NoResiduals:
    """A fit whose ``resid`` is not a numeric vector."""

    resid = ["a", "b", "c"]


def test_a_fit_whose_residuals_are_not_numbers_says_so():
    """Some backends expose a ``resid`` that is not a float vector at all.

    Coercing it raises, and the residual tests then have to report why they
    could not be run rather than a p value computed from nothing.
    """
    reason = RS._no_residual_reason(_run(model=_NoResiduals()))
    assert "exposes no residual vector" in reason
    assert "_NoResiduals" in reason


def test_a_fit_with_too_few_residuals_names_the_count():
    """Eight is the floor; the message has to say how far short this fit is."""

    class _Short:
        resid = np.array([0.1, -0.2, 0.3])

    reason = RS._no_residual_reason(_run(model=_Short()))
    assert "at least 8 finite residuals" in reason
    assert "3" in reason


# ---------------------------------------------------------------------------
# _permutation_count and the fields that depend on it
# ---------------------------------------------------------------------------

def test_a_parametric_run_has_no_permutation_count():
    """The count only exists for the permutation path."""
    assert RS._permutation_count(_run()) is None


def test_a_permutation_run_with_no_recorded_count_says_neither_source_has_it():
    """A count that is not a number must not be replaced by the default.

    ``guide_permutations`` survives a settings-CSV round trip as text, and a
    run whose value came back unreadable has no floor. Filling in 200000 there
    would print "no P below 5e-06 is expressible" about a run that may have
    drawn a hundred.
    """
    run = _run(nonparametric=True, coef_df=pd.DataFrame({"gene": ["a"]}),
               settings={"guide_permutations": "as many as it took"})
    assert RS._permutation_count(run) is None
    assert "records no permutation count" in RS._permutations(run)["reason"]
    assert "1/(n+1)" in RS._finest_p(run)["reason"]
    assert "no permutation P value" in RS._n_at_finest_p(run)["reason"]


def test_a_permutation_run_reports_its_floor_from_the_recorded_count():
    """The floor is the whole point: no p below 1/(n+1) is expressible."""
    run = _run(nonparametric=True,
               coef_df=pd.DataFrame({"permutations": [999, 999]}))
    assert RS._permutations(run)["value"] == "999"
    assert "1/(999+1)" in RS._finest_p(run)["value"]


def test_a_permutation_p_column_with_nothing_finite_in_it_says_so():
    """An all-NaN P column is not "no tests at the floor"; it is no P values."""
    run = _run(nonparametric=True, coef_df=pd.DataFrame({
        "permutations": [99, 99], "permutation_p_value": [np.nan, np.nan]}))
    assert RS._n_at_finest_p(run)["reason"] == \
        "the P value column holds no finite value"


# ---------------------------------------------------------------------------
# _selection_frequency
# ---------------------------------------------------------------------------

def test_a_selection_frequency_column_with_nothing_finite_says_so():
    """An empty column is a defect in the run, not a stability of zero."""
    run = _run(penalised=True, regression_type="lasso",
               coef_df=pd.DataFrame({"selection_frequency": [np.nan, np.nan]}))
    assert RS._selection_frequency(run)["reason"] == \
        "the selection_frequency column holds no finite value"


def test_a_real_selection_frequency_column_is_summarised():
    """The refusal above must not be swallowing the ordinary answer."""
    run = _run(penalised=True, regression_type="lasso",
               coef_df=pd.DataFrame({"selection_frequency": [0.9, 0.1, 0.7]}),
               settings={"lasso_selection_threshold": 0.6,
                         "lasso_n_boot": 50})
    value = RS._selection_frequency(run)["value"]
    assert "2 at or above" in value
    assert "50 bootstrap resamples" in value


# ---------------------------------------------------------------------------
# _pseudo_r_squared
# ---------------------------------------------------------------------------

def test_a_pseudo_r_squared_is_rebuilt_from_the_log_likelihoods():
    """``pseudo_rsquared`` and ``prsquared`` both failing is not the end.

    McFadden's is ``1 - llf/llnull`` by definition, and a fit that kept both
    likelihoods can still answer even when neither convenience accessor does.
    """

    class _Likelihoods:
        llf = -50.0
        llnull = -100.0

        def pseudo_rsquared(self, **kwargs):
            raise RuntimeError("not implemented on this fit")

        prsquared = property(lambda self: 1 / 0)

    out = RS._pseudo_r_squared(_run(model=_Likelihoods()))
    assert out["value"].startswith("0.5")


# ---------------------------------------------------------------------------
# _exclusion_count
# ---------------------------------------------------------------------------

def test_an_exclusion_record_that_is_not_a_number_is_not_counted():
    """"nobody counted" and "zero rows were dropped" are opposite findings."""
    settings = {"_regression_exclusions": {"low_reads": "lots"}}
    assert RS._exclusion_count(settings, "low_reads") is None


def test_an_exclusion_record_that_is_not_a_mapping_is_not_counted():
    """A settings CSV round-trip can leave a string where a dict was."""
    assert RS._exclusion_count({"_regression_exclusions": "3 rows"},
                               "low_reads") is None


def test_a_recorded_exclusion_count_is_read_as_an_integer():
    """The refusals above must not be swallowing a real count."""
    settings = {"_regression_exclusions": {"low_reads": "12"}}
    assert RS._exclusion_count(settings, "low_reads") == 12


# ---------------------------------------------------------------------------
# _median_wells
# ---------------------------------------------------------------------------

def test_the_median_wells_per_guide_needs_both_a_guide_and_a_well_column():
    """Read off the fitted table; a table without those columns has no answer."""
    assert RS._median_wells(_run(data=pd.DataFrame({"x": [1, 2]}))) is None
    assert RS._median_wells(_run(data=pd.DataFrame())) is None


def test_the_median_wells_per_guide_is_read_off_the_fitted_table():
    """It reflects the threshold as APPLIED, so it comes from that table."""
    data = pd.DataFrame({
        "grna": ["g1", "g1", "g1", "g2", "g2"],
        "prc": ["w1", "w2", "w3", "w1", "w1"],
    })
    assert RS._median_wells(_run(data=data)) == 2.0


# ---------------------------------------------------------------------------
# _verbatim
# ---------------------------------------------------------------------------

def test_a_statsmodels_summary_that_will_not_render_is_a_note_not_a_loss():
    """The rest of the summary is worth more than the appended text."""

    class _Unrenderable:
        def summary(self):
            raise RuntimeError("singular design")

    text, note = RS._verbatim(_run(model=_Unrenderable()))
    assert text is None
    assert "could not render its summary" in note
    assert "RuntimeError" in note


# ---------------------------------------------------------------------------
# _recommendations
# ---------------------------------------------------------------------------

def test_advice_that_raises_costs_the_advice_and_nothing_else(monkeypatch):
    """A run that got this far has numbers worth reading; keep them."""
    monkeypatch.setitem(sys.modules, "spacr.run_recommendations", None)
    run = _run(metrics={"r_squared": 0.4})
    assert RS._recommendations(run) == []


def test_the_rendered_summary_survives_the_advice_formatter_being_gone(
        monkeypatch):
    """The sections above the advice are the deliverable, not the advice."""
    monkeypatch.setitem(sys.modules, "spacr.run_recommendations", None)
    summary = RS.build_run_summary(settings={"regression_type": "ols"})
    text = RS.format_run_summary(summary)
    assert "spaCR RUN SUMMARY" in text
    assert summary.missing() == []


# ---------------------------------------------------------------------------
# _fitted_section: what the run says it was
# ---------------------------------------------------------------------------

def _fields(builder, run):
    return {one.name: one for one in builder(run)}


def test_a_run_that_recorded_no_family_says_it_cannot_name_one():
    """Guessing 'ols' would put a family in a methods section nobody fitted."""
    fields = _fields(RS._fitted_section, _run(regression_type=None))
    assert fields["regression_type"].value is None
    assert "cannot be named" in fields["regression_type"].reason


def test_a_parametric_run_reports_the_inference_it_resolved_to():
    """``inference='auto'`` is a resolution, and the summary has to say so."""
    fields = _fields(RS._fitted_section,
                     _run(regression_type=None,
                          settings={"inference": "auto"}))
    value = fields["inference"].value
    assert value.startswith("parametric")
    assert "inference='auto' resolved to it" in value


def test_a_parametric_run_with_no_inference_key_names_the_default():
    """A settings file from before the key existed still gets a sentence."""
    value = _fields(RS._fitted_section,
                    _run(regression_type=None))["inference"].value
    assert "'parametric'" in value


def test_the_backend_line_falls_back_to_the_bare_name(monkeypatch):
    """The spec supplies the human label; without it the key is still true."""
    import spacr.regression_spec as spec

    monkeypatch.delattr(spec, "REGRESSION_BACKENDS")
    fields = _fields(RS._fitted_section,
                     _run(regression_type=None,
                          settings={"regression_backend": "pyfixest"}))
    assert fields["backend"].value == "pyfixest"


def test_the_level_line_names_the_levels_rows_were_actually_written_for():
    """"asked for both" and "got only guide rows" are different runs."""
    frame = pd.DataFrame({"level": ["grna", "gene", "grna"]})
    fields = _fields(RS._fitted_section,
                     _run(regression_type=None, settings={"level": "both"},
                          coef_df=frame))
    assert fields["level"].value == "both (rows written for: gene, grna)"


def test_a_beta_transform_names_the_squeeze_it_applied():
    """A well at exactly 0 or 1 was MOVED, and the reader has to be told."""
    from spacr.ml import BETA_SQUEEZE_NOTE

    fields = _fields(RS._fitted_section,
                     _run(regression_type=None,
                          settings={"transform": "beta"}))
    assert fields["transform"].value == BETA_SQUEEZE_NOTE


def test_plate_position_left_out_of_the_model_is_said_out_loud():
    """A screen fitted without row/column effects is a different analysis."""
    fields = _fields(RS._fitted_section,
                     _run(regression_type=None,
                          settings={"model_plate_position": False}))
    assert "OUT of the model entirely" in fields["plate_position"].value


def test_plate_position_as_random_effects_is_distinguished_from_fixed():
    """Variance components and fixed effects are not the same model."""
    fields = _fields(RS._fitted_section,
                     _run(regression_type=None,
                          settings={"random_row_column_effects": True}))
    assert "VARIANCE COMPONENTS" in fields["plate_position"].value


def test_plate_position_defaults_to_fixed_effects():
    """The default has to be stated, or the two branches above mean nothing."""
    fields = _fields(RS._fitted_section, _run(regression_type=None))
    assert "FIXED effects" in fields["plate_position"].value


# ---------------------------------------------------------------------------
# _formula
# ---------------------------------------------------------------------------

class _Fit:
    """A statsmodels-shaped result: ``.model`` carries the design."""

    def __init__(self, **inner):
        self.model = type("_Inner", (), inner)()


def test_a_formula_kept_by_the_fit_is_reported_as_read_off_it():
    """When the fit knows, nothing is rebuilt and nothing can drift."""
    out = RS._formula(_run(model=_Fit(formula="pred ~ grna + rowID")))
    assert out["value"].startswith("pred ~ grna + rowID")
    assert "read off the fit" in out["value"]


def test_a_backend_with_no_formula_has_one_rebuilt_from_the_settings():
    """The rebuilt formula names the response the ESTIMATOR saw."""
    run = _run(model=_Fit(formula=None, endog_names="log_pred"),
               settings={"level": "both", "regression_type": "ols"})
    out = RS._formula(run)
    assert "log_pred" in out["value"]
    assert "rebuilt from the settings" in out["value"]


def test_a_formula_that_cannot_be_rebuilt_is_a_reason_not_a_blank(monkeypatch):
    """Losing the formula must not lose the field it belongs to."""
    monkeypatch.setitem(sys.modules, "spacr.ml", None)
    out = RS._formula(_run(model=_Fit(formula=None)))
    assert "value" not in out
    assert "could not be rebuilt" in out["reason"]


# ---------------------------------------------------------------------------
# _design_section
# ---------------------------------------------------------------------------

def test_a_design_whose_rank_was_never_reported_says_so():
    """A width without a rank cannot say whether the fit was identifiable."""
    fields = _fields(RS._design_section, _run(metrics={"n_parameters": 12}))
    assert fields["n_parameters"].value == "12"
    assert "reported no rank" in fields["design_rank"].reason
    assert "no parameters to divide" in fields["wells_per_parameter"].reason
    assert "identifiability is the comparison" in fields["identifiable"].reason


def test_a_design_with_a_rank_reports_it_against_the_width():
    """The refusals above must not be swallowing the ordinary answer."""
    fields = _fields(RS._design_section,
                     _run(metrics={"n_parameters": 12, "design_rank": 11,
                                   "wells_per_parameter": 3.5,
                                   "design_identifiable": False,
                                   "non_identifiable_directions": 1}))
    assert fields["design_rank"].value == "11 of 12 columns"
    assert fields["wells_per_parameter"].value == "3.5"
    assert fields["identifiable"].value.startswith("NO — 1 direction")


# ---------------------------------------------------------------------------
# _fit_quality_section: the last R2 fallback
# ---------------------------------------------------------------------------

def test_a_fit_that_reports_no_r_squared_is_not_given_an_invented_one():
    """A number spaCR computed itself would match no other tool's printout."""

    class _Odd:
        resid = np.zeros(20)

    fields = _fields(RS._fit_quality_section,
                     _run(model=_Odd(), regression_type="mystery"))
    assert fields["r_squared"].value is None
    assert "_Odd reports no R2" in fields["r_squared"].reason


# ---------------------------------------------------------------------------
# The assumption helpers
# ---------------------------------------------------------------------------

def test_only_one_heteroscedasticity_test_reports_one_verdict():
    """Breusch-Pagan and White disagree often; each gets its own line.

    White's test squares every column pair, so it is skipped on wide or
    singular designs. The summary has to say it was skipped rather than let
    the reader take one verdict for two.
    """
    out = RS._equal_variance(_run(metrics={"breusch_pagan_p": 0.645}))
    assert "Breusch-Pagan p = 0.645" in out["value"]
    assert "White's test was not run" in out["value"]


def test_a_normality_check_that_raises_is_a_reason_not_a_blank(monkeypatch):
    """The residual shape is optional; the field it fills is not."""
    monkeypatch.setitem(sys.modules, "spacr.regression_qc", None)
    out = RS._normality(_run(model=_Fit(), metrics={}))
    assert "value" not in out
    assert "could not be measured" in out["reason"]


def test_a_fit_with_no_hat_diagonal_has_no_leverage_to_report():
    """Cook's distance is built from it, so both are undefined together."""

    class _Context:
        leverage = np.array([np.nan, np.nan])
        n = 2
        p = 1

    monkeypatched = _run(model=_Fit())
    import spacr.regression_qc as qc

    original = qc.context_from_model
    try:
        qc.context_from_model = lambda *a, **k: _Context()
        out = RS._influence(monkeypatched)
    finally:
        qc.context_from_model = original
    assert "no hat-matrix diagonal" in out["reason"]


def test_an_unscaled_condition_number_is_labelled_unscaled():
    """The 30/100/1000 bands apply to the scaled number, not to this one.

    Printing statsmodels' own condition number beside those bands is how a
    perfectly conditioned design comes to be read as collinear.
    """
    run = _run(model=_Fit(exog=None), metrics={"condition_number": 4.2e5})
    out = RS._multicollinearity(run)
    assert "UNSCALED as statsmodels prints it" in out["value"]


# ---------------------------------------------------------------------------
# _tested_mask and the call section
# ---------------------------------------------------------------------------

def test_a_family_that_cannot_be_identified_is_no_mask_at_all(monkeypatch):
    """Guessing "everything was tested" would inflate the correction's family."""
    monkeypatch.setitem(sys.modules, "spacr.hits", None)
    frame = pd.DataFrame({"feature": ["grna_a"], "p_value": [0.01]})
    assert RS._tested_mask(_run(coef_df=frame)) is None


def test_hits_called_on_the_raw_p_say_so_in_the_alpha_line():
    """A raw-P cut is not a corrected one, and the line has to distinguish them."""
    frame = pd.DataFrame({"feature": ["grna_a", "grna_b"],
                          "p_value": [0.001, 0.4],
                          "coefficient": [1.0, 0.1]})
    run = _run(coef_df=frame,
               settings={"regression_type": "ols", "fdr_alpha": 0.05,
                         "hit_cut": "raw", "hit_p_threshold": 0.01})
    fields = _fields(RS._call_section, run)
    assert "fdr_alpha" in fields


def test_the_correction_method_line_survives_a_missing_label_table(
        monkeypatch):
    """The key is the fact; the human label is a courtesy on top of it."""
    import spacr.multiple_testing as mt

    monkeypatch.delattr(mt, "method_label")
    frame = pd.DataFrame({"feature": ["grna_a"], "p_value": [0.01],
                          "coefficient": [1.0]})
    run = _run(coef_df=frame,
               settings={"regression_type": "ols",
                         "multiple_testing_method": "fdr_bh"})
    fields = _fields(RS._call_section, run)
    assert "fdr_bh" in fields["multiple_testing_method"].value


# ---------------------------------------------------------------------------
# _critical_p
# ---------------------------------------------------------------------------

def test_a_tested_family_with_no_finite_p_has_no_critical_threshold():
    """A threshold located from nothing would be a line nobody's data reached."""
    frame = pd.DataFrame({"p_value": [np.nan, np.nan]})
    tested = np.array([True, True])
    assert RS._critical_p(_run(coef_df=frame), tested)["reason"] == \
        "the tested family carries no finite raw P value"


def test_a_critical_p_that_cannot_be_computed_is_a_reason(monkeypatch):
    """Losing the exact threshold must not lose the field."""
    monkeypatch.setitem(sys.modules, "spacr.multiple_testing", None)
    frame = pd.DataFrame({"p_value": [0.01, 0.2, 0.5]})
    out = RS._critical_p(_run(coef_df=frame), np.array([True, True, True]))
    assert "could not be computed" in out["reason"]


# ---------------------------------------------------------------------------
# _below_effect_size
# ---------------------------------------------------------------------------

def test_an_effect_size_cut_cannot_be_recounted_without_the_p_column():
    """Recounting from a table missing the column would report a made-up zero."""
    frame = pd.DataFrame({"coefficient": [1.0, 0.01],
                          "effect_size_threshold": [0.5, 0.5]})
    out = RS._below_effect_size(_run(coef_df=frame))
    assert "value" not in out
    assert "cannot be recounted" in out["reason"]
    assert "q_value" in out["reason"]


# ---------------------------------------------------------------------------
# write_run_summary
# ---------------------------------------------------------------------------

def test_an_unreadable_previous_summary_does_not_stop_the_write(tmp_path,
                                                                monkeypatch):
    """The recovered statsmodels text is a bonus; the summary is the file.

    The earlier text is read back only so it can be appended unchanged. The
    read is injected to fail here because the file has to stay writable for
    the rest of the call -- a permission that blocked both would test the
    write, not the recovery.
    """
    import builtins

    folder = tmp_path / "run"
    folder.mkdir()
    path = folder / RS._summary_filename()
    path.write_text("an earlier statsmodels summary")

    real_open = builtins.open

    def refuse_reads(file, mode="r", *args, **kwargs):
        if str(file) == str(path) and "r" in mode:
            raise OSError(5, "Input/output error")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", refuse_reads)
    written = RS.write_run_summary(str(folder), settings={})
    monkeypatch.undo()

    assert written == str(path)
    text = path.read_text()
    assert text.startswith("spaCR RUN SUMMARY")
    assert "an earlier statsmodels summary" not in text



# ---------------------------------------------------------------------------
# The fitted section is currently lost to a missing label
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="LABELS has no ('fitted', "
                                       "'hyperparameters') entry, so "
                                       "_fitted_section raises KeyError for "
                                       "every parametric run that recorded a "
                                       "family and the whole section is "
                                       "backfilled with the exception")
def test_a_parametric_run_reports_what_it_fitted_rather_than_a_key_error():
    """WHAT WAS FITTED must carry the run's family, not a traceback.

    ``_fitted_section`` adds a ``hyperparameters`` field for every run that
    named a regression family, but ``LABELS`` carries no label for it. The
    lookup raises, ``build_run_summary`` backfills the entire section with
    "this section could not be built (KeyError: ...)", and every one of the
    eleven fields in the most-read section of the summary -- the family, the
    response, the formula, whether plate position was modelled -- is replaced
    by that sentence. The permutation path is unaffected, so the defect is
    invisible on exactly the mode that was added last.
    """
    summary = RS.build_run_summary(
        settings={"regression_type": "ols", "dependent_variable": "pred"})
    fitted = summary.section("fitted")
    assert fitted.get("regression_type").value == "ols"
    assert fitted.get("dependent_variable").value == "pred"
    assert all("could not be built" not in str(one.reason or "")
               for one in fitted.fields)


# ---------------------------------------------------------------------------
# _multicollinearity: the two condition-number readings
# ---------------------------------------------------------------------------

def test_a_condition_number_that_cannot_be_scaled_falls_back_to_the_raw_one(
        monkeypatch):
    """Losing the scaled reading must not lose the design's conditioning.

    The scaled number is the one the 30 / 100 / 1000 bands are written for.
    When it cannot be computed, the unscaled number is still worth printing --
    labelled as unscaled, so nobody reads it against the wrong bands.
    """
    monkeypatch.setitem(sys.modules, "spacr.regression_qc", None)
    run = _run(model=_Fit(exog=np.eye(4)),
               metrics={"condition_number": 12.0, "max_vif": 1.0})
    out = RS._multicollinearity(run)
    assert "UNSCALED as statsmodels prints it" in out["value"]


def test_a_scaled_condition_number_with_no_verdict_still_prints_the_number(
        monkeypatch):
    """The number is the measurement; the verdict is the gloss on it."""
    import spacr.regression_qc as qc

    def no_verdict(value):
        raise RuntimeError("bands unavailable")

    monkeypatch.setattr(qc, "condition_verdict", no_verdict)
    run = _run(model=_Fit(exog=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])),
               metrics={"max_vif": 1.0})
    out = RS._multicollinearity(run)
    assert "scaled condition number" in out["value"]
    assert "no reading available" in out["value"]


# ---------------------------------------------------------------------------
# Hits called on the raw P
# ---------------------------------------------------------------------------

def _raw_cut_run(frame):
    return _run(coef_df=frame, regression_type=None,
                settings={"p_threshold_kind": "raw",
                          "p_threshold_alpha": 0.01,
                          "fdr_alpha": 0.05})


def test_a_run_that_called_on_the_raw_p_says_raw_and_not_q():
    """"q value below 0.01" about an uncorrected cut is the wrong claim."""
    frame = pd.DataFrame({"p_value": [0.001, 0.4], "coefficient": [1.0, 0.1]})
    mask, note = RS._hit_mask(_raw_cut_run(frame))
    assert mask.tolist() == [True, False]
    assert note == "raw P below 0.01"


def test_the_alpha_line_says_when_hits_were_called_off_the_uncorrected_p():
    """fdr_alpha targets the correction; a raw cut did not use it."""
    frame = pd.DataFrame({"feature": ["grna_a", "grna_b"],
                          "p_value": [0.001, 0.4],
                          "coefficient": [1.0, 0.1]})
    fields = _fields(RS._call_section, _raw_cut_run(frame))
    value = fields["fdr_alpha"].value
    assert "hits were CALLED on the raw P at 0.01" in value
    assert "NOT corrected for multiple testing" in value


# ---------------------------------------------------------------------------
# _excluded_section
# ---------------------------------------------------------------------------

def test_a_recorded_drop_with_no_denominator_still_reports_the_count():
    """"12 rows were dropped" is a finding even without "of how many"."""
    run = _run(settings={"fraction_threshold": 0.05,
                         "_regression_exclusions": {"fraction_threshold": 12}})
    fields = _fields(RS._excluded_section, run)
    value = fields["fraction_threshold"].value
    assert value.startswith("12 gRNA rows were below a well fraction of 0.05")
    assert " of " not in value.split("were below")[0]


def test_a_recorded_drop_with_a_denominator_reports_both():
    """The branch above must not be swallowing the fuller answer."""
    run = _run(settings={"fraction_threshold": 0.05,
                         "_regression_exclusions": {
                             "fraction_threshold": 12,
                             "fraction_threshold_of": 400}})
    fields = _fields(RS._excluded_section, run)
    assert fields["fraction_threshold"].value.startswith(
        "12 of 400 gRNA rows")


# ---------------------------------------------------------------------------
# _median_wells on a table it cannot group
# ---------------------------------------------------------------------------

def test_a_fitted_table_that_cannot_be_grouped_yields_no_median():
    """A guide column of unhashable values makes ``groupby`` raise.

    The median wells per guide is one input to the advice at the end. Losing
    it must cost that one number and nothing else, so the failure is absorbed
    here rather than in the middle of building a recommendation.
    """
    data = pd.DataFrame({"grna": [["g1"], ["g2"]], "prc": ["w1", "w2"]})
    assert RS._median_wells(_run(data=data)) is None


# ---------------------------------------------------------------------------
# _hyperparameter_report
# ---------------------------------------------------------------------------

def test_a_cross_validated_alpha_that_is_not_a_number_is_reported_as_unknown():
    """``alpha_`` off a stub or a partially restored fit may not be a float.

    The run asked for ``alpha='auto'``, so the value that won is the number a
    reader needs. When the object cannot supply one, the line has to say the
    run did not record it -- not print the object's repr as if it were the
    alpha.
    """

    class _StubCV:
        alpha_ = "chosen elsewhere"

    report = RS._hyperparameter_report("ridge", {"alpha": "auto"},
                                       _run(model=_StubCV()))
    assert "alpha=auto — cross-validated" in report["value"]
    assert "did not record the value that won" in report["value"]


def test_a_cross_validated_alpha_that_is_a_number_is_reported():
    """The refusal above must not be swallowing the value that won."""

    class _CV:
        alpha_ = 0.125

    report = RS._hyperparameter_report("ridge", {"alpha": "auto"},
                                       _run(model=_CV()))
    assert "alpha=0.125 (cross-validated, not given)" in report["value"]

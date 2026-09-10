"""Instruction 156: every regression mode gets a summary, not only the two
statsmodels writes one for.

THE CONTRACT IS THE DELIVERABLE, so the contract is what these tests assert.
:func:`spacr.regression_summary.build_run_summary` returns every name in
:data:`spacr.regression_summary.CONTRACT` for every one of the twenty
:data:`spacr.regression_spec.REGRESSION_TYPES` crossed with both inferences,
and each name is answered — with a number, or with a sentence saying why this
mode cannot have one. A blank is not allowed and neither is a zero standing in
for "we did not check": that is the failure this instruction is about, and
:class:`spacr.regression_summary.SummaryField` refuses to represent it.

THE REPORTED CASE. A nonparametric mixed run said

    No summary: this run came back without a fitted model, so there is none
    to summarise.

which is true — ``inference='nonparametric'`` is a plate-blocked permutation
test with no design matrix and no coefficient covariance — and useless,
because the run produced results with properties worth reporting. So the
permutation arm here asserts the things that arm must say: that R2 DOES NOT
EXIST in those words, the permutation resolution in its place, and every
assumption listed as NOT ASSUMED rather than left empty.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr.regression_spec import NO_P_VALUE_TYPES, REGRESSION_TYPES
import spacr.regression_summary as RS


INFERENCES = ("parametric", "nonparametric")
GUIDES = [f"{gene}_{i}" for gene in ("000000", "220950", "gene3", "gene4")
          for i in (1, 2, 3)]


# ---------------------------------------------------------------------------
# Fixtures: a run folder, and the coefficient table each mode writes
# ---------------------------------------------------------------------------

def _fitted_table(n_wells=24, seed=0):
    """``regression_data.csv`` as ``perform_regression`` writes it.

    One row per (well, guide), the response repeated on each of a well's rows
    and ``cell_count`` repeated with it — which is the shape that makes
    ``n_cells`` a sum over DISTINCT WELLS rather than over rows.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_wells):
        prc = f"plate{w % 2 + 1}_r{w // 4 + 1:02d}_c{w % 4 + 1:02d}"
        cells = int(rng.integers(100, 400))
        response = float(rng.normal(0.5, 0.2))
        # THE FRACTIONS VARY WITHIN THE WELL, as the real ones do. Fixed at
        # 1/3 each they sum to a constant over the twelve `fraction:grna`
        # columns, which is EXACTLY collinear with the intercept -- the fit
        # then comes back rank deficient and the identifiability warning fires
        # correctly on a fixture that meant to be healthy. Found by that
        # warning, which is what it is for.
        raw = rng.random(3) + 0.2
        for guide, one in zip(rng.choice(GUIDES, size=3, replace=False),
                              raw / raw.sum()):
            rows.append({
                "plateID": prc.split("_")[0], "rowID": prc.split("_")[1],
                "columnID": prc.split("_")[2], "prc": prc,
                "grna": guide, "gene": guide.rsplit("_", 1)[0],
                "fraction": float(one), "cell_count": cells,
                "pred": response,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def run_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("run")
    _fitted_table().to_csv(folder / "regression_data.csv", index=False)
    return str(folder)


def _parametric_coefficients(regression_type, seed=1):
    """One corrected coefficient table, shaped as ``_call_level_hits`` leaves it."""
    rng = np.random.default_rng(seed)
    features = (["Intercept", "rowID[T.r02]", "columnID[T.c02]"]
                + [f"fraction:grna[{guide}]" for guide in GUIDES])
    p = np.concatenate([[np.nan, 1e-8, 0.4], rng.uniform(0, 1, len(GUIDES))])
    frame = pd.DataFrame({
        "feature": features,
        "coefficient": rng.normal(0, 1, len(features)),
        "p_value": p,
        "level": "grna",
        "condition": (["other"] * 3
                      + ["nc"] * 3 + ["pc"] * 3 + ["other"] * (len(GUIDES) - 6)),
        "n_grna": 3.0, "n_gene": 3.0,
    })
    if regression_type in NO_P_VALUE_TYPES:
        # The penalised branch of `_call_level_hits` returns BEFORE the
        # correction, so its table has no q value and carries the bootstrap
        # frequency instead. Reproducing that here is the point: the summary
        # must report the selection frequency and say plainly that a training
        # R2 is not a fit statistic.
        frame["selection_frequency"] = rng.uniform(0, 1, len(features))
        return frame
    frame["q_value"] = np.clip(frame["p_value"] * 3, 0, 1)
    frame["multiple_testing_method"] = "fdr_bh"
    frame["effect_size_threshold"] = 0.8
    frame["effect_size_rule"] = "3x std of 6 controls = 0.8"
    if regression_type == "mixed":
        frame["term_type"] = "fixed"
        blups = frame.tail(3).copy()
        blups["term_type"] = "random_effect_blup"
        blups["p_value"] = np.nan
        blups["q_value"] = np.nan
        frame = pd.concat([frame, blups], ignore_index=True)
    return frame


def _permutation_coefficients(permutations=1000, seed=2):
    """The primary family, as ``analyse_long_guide_table`` writes it."""
    rng = np.random.default_rng(seed)
    p = np.clip(rng.uniform(0, 1, len(GUIDES)), 1.0 / (permutations + 1), 1.0)
    p[0] = 1.0 / (permutations + 1)                 # one test at the floor
    frame = pd.DataFrame({
        "outcome": "pred",
        "guide": GUIDES,
        "grna": GUIDES,
        "feature": [f"fraction:grna[{guide}]" for guide in GUIDES],
        "wells_with_guide": 6,
        "standardized_marginal_effect": rng.normal(0, 1, len(GUIDES)),
        "permutation_exceedances": 0,
        "permutations": permutations,
        "permutation_p_value": p,
        "block_column": "plateID",
        "nuisance_columns": "",
        "presence_threshold": 0.0,
        "minimum_wells_threshold": 1,
        "tested_guides_in_family": len(GUIDES),
        "multiple_testing_method": "fdr_bh",
    })
    frame["coefficient"] = frame["standardized_marginal_effect"]
    frame["p_value"] = frame["permutation_p_value"]
    frame["adjusted_p_value"] = np.clip(frame["p_value"] * 2, 0, 1)
    frame["q_value"] = frame["adjusted_p_value"]
    frame["significant"] = frame["q_value"] < 0.05
    frame["condition"] = ["nc"] * 3 + ["pc"] * 3 + ["other"] * (len(GUIDES) - 6)
    frame["effect_size_threshold"] = 1.2
    frame["effect_size_rule"] = "3x std of 3 controls = 1.2"
    frame["passes_effect_size"] = frame["coefficient"].abs() >= 1.2
    return frame


def _settings(regression_type, inference):
    settings = {
        "regression_type": regression_type,
        "inference": inference,
        "dependent_variable": "pred",
        "analysis_unit": "well",
        "agg_type": "mean",
        "transform": None,
        "multiple_testing_method": "fdr_bh",
        "fdr_alpha": 0.05,
        "fraction_threshold": 0.01,
        "min_cell_count": 25,
        "level": "both",
        "regression_backend": "statsmodels",
        "model_plate_position": True,
        "random_row_column_effects": False,
        "controls": ["000000_1", "000000_2", "000000_3"],
        "negative_control": "000000",
        "positive_control": "220950",
        "lasso_n_boot": 200,
        "lasso_selection_threshold": 0.6,
    }
    if inference == "nonparametric":
        settings["analysis_mode"] = "guide_permutation"
        settings["guide_permutations"] = 1000
        settings["guide_permutation_block"] = "plateID"
    else:
        settings["analysis_mode"] = "regression"
    return settings


def _summary(regression_type, inference, run_folder, model=None):
    coef = (_permutation_coefficients() if inference == "nonparametric"
            else _parametric_coefficients(regression_type))
    return RS.build_run_summary(
        model=model, settings=_settings(regression_type, inference),
        coef_df=coef, regression_type=regression_type,
        res_folder=run_folder)


@pytest.fixture(scope="module")
def ols_fit():
    """A real statsmodels fit, so the model-reading branches are exercised."""
    import statsmodels.api as sm
    from patsy import dmatrices

    frame = _fitted_table(n_wells=48, seed=5)
    frame["response"] = (frame["pred"]
                         + 0.3 * (frame["grna"] == GUIDES[3]).astype(float))
    y, X = dmatrices("response ~ fraction:grna + rowID + columnID",
                     data=frame, return_type="dataframe")
    return sm.OLS(y, X).fit()


# ---------------------------------------------------------------------------
# 1. THE WHOLE PRODUCT. Every type times two inferences, no hole.
# ---------------------------------------------------------------------------

def _check_every_field_is_answered(summary):
    """No missing name, no blank, and no shrug where a reason belongs."""
    assert summary.missing() == []
    seen = set()
    for section in summary.sections:
        assert section.name in RS.CONTRACT
        for one in section.fields:
            assert (section.name, one.name) not in seen, "duplicated field"
            seen.add((section.name, one.name))
            assert one.text.strip(), f"{section.name}.{one.name} is blank"
            if one.answered:
                assert str(one.value).strip()
            else:
                reason = str(one.reason).strip()
                # A reason is a SENTENCE. "n/a", "unknown" and "-" are the
                # blanks this type exists to refuse, spelled differently.
                assert len(reason) >= 25, f"{one.name}: {reason!r}"
                assert reason.lower() not in ("n/a", "na", "unknown", "none",
                                              "-", "not computed")
    assert set(seen) == {(section, name)
                         for section, names in RS.CONTRACT.items()
                         for name in names}


@pytest.mark.parametrize("regression_type", REGRESSION_TYPES)
@pytest.mark.parametrize("inference", INFERENCES)
def test_every_type_and_inference_answers_every_field(regression_type,
                                                      inference, run_folder):
    """The item's own acceptance test, over all thirty-eight combinations."""
    _check_every_field_is_answered(
        _summary(regression_type, inference, run_folder))


@pytest.mark.parametrize("regression_type", REGRESSION_TYPES)
@pytest.mark.parametrize("inference", INFERENCES)
def test_the_contract_holds_with_a_fitted_model_too(regression_type, inference,
                                                    run_folder, ols_fit):
    """The same product, with a real model attached.

    The model is an OLS fit whatever the type string says, and deliberately
    so: what is being asserted is that the branch matrix — twenty type
    strings against a model that DOES report an R2, a log-likelihood and a
    design matrix — never leaves a field unanswered. Reading a real fit is
    where the exceptions live; the type string is what chooses the sentence.
    """
    _check_every_field_is_answered(
        _summary(regression_type, inference, run_folder, model=ols_fit))


def test_every_field_reaches_the_rendered_text(run_folder):
    """A field the renderer drops is as missing as one never built."""
    summary = _summary("ols", "parametric", run_folder)
    text = summary.text()
    for section in summary.sections:
        assert section.title in text
        for one in section.fields:
            assert one.label in text, f"{one.label} never printed"


# ---------------------------------------------------------------------------
# 2. The nonparametric path, which is the reported case
# ---------------------------------------------------------------------------

def test_a_nonparametric_run_has_a_summary_at_all(run_folder):
    """The sentence the maintainer got must not appear anywhere in it."""
    text = _summary("mixed", "nonparametric", run_folder).text()
    assert "No summary" not in text
    assert "there is none to summarise" not in text
    assert "WHAT WAS FITTED" in text


def test_the_nonparametric_summary_says_r2_does_not_exist_in_those_words(
        run_folder):
    summary = _summary("mixed", "nonparametric", run_folder)
    r2 = summary.field("r_squared")
    assert not r2.answered
    assert "R2 DOES NOT EXIST" in r2.reason
    # And not a zero standing in for it, which is the failure being prevented.
    assert r2.value is None


def test_the_nonparametric_summary_reports_the_permutation_resolution(
        run_folder):
    """1,000 permutations cannot express p < 1e-3, and the summary says so."""
    summary = _summary("mixed", "nonparametric", run_folder)
    assert summary.field("permutations").value == "1,000"
    finest = summary.field("finest_p")
    assert finest.answered
    assert "1/(1,000+1)" in finest.value
    assert f"{1 / 1001:.3g}" in finest.value
    assert "1e-03" in finest.value
    at_floor = summary.field("n_at_finest_p")
    assert at_floor.answered
    # The fixture plants exactly one test on the floor.
    assert at_floor.value.startswith("1 of 12")


def test_a_parametric_run_has_no_permutation_floor_and_says_why(run_folder):
    summary = _summary("ols", "parametric", run_folder)
    for name in ("permutations", "finest_p", "n_at_finest_p"):
        one = summary.field(name)
        assert not one.answered
        assert one.reason


@pytest.mark.parametrize("name", tuple(RS.CONTRACT["assumptions"]))
def test_nonparametric_assumptions_are_not_assumed_never_blank(name,
                                                               run_folder):
    """The POINT of choosing the safer method must not read as less information.

    An empty assumptions block would make the permutation test look like the
    less informative one; each of the five is a stated NOT ASSUMED with the
    reason the permutation buys it.
    """
    one = _summary("mixed", "nonparametric", run_folder).field(name)
    assert one.kind == RS.NOT_ASSUMED
    assert "NOT ASSUMED" in one.text
    assert len(one.reason) >= 40


def test_parametric_assumptions_carry_a_test_and_a_verdict(run_folder,
                                                           ols_fit):
    summary = _summary("ols", "parametric", run_folder, model=ols_fit)
    equal = summary.field("equal_variance")
    assert equal.answered
    assert "Breusch-Pagan" in equal.value
    assert ("REJECTED" in equal.value or "not rejected" in equal.value)
    normal = summary.field("normality")
    assert normal.answered
    assert "skew" in normal.value and "excess kurtosis" in normal.value


# ---------------------------------------------------------------------------
# 3. Fit quality per family: pseudo-R2 NAMED, penalised R2 refused
# ---------------------------------------------------------------------------

def test_a_glm_family_reports_a_pseudo_r_squared_named_as_pseudo(run_folder):
    """It must not be readable as an OLS R2, which is the whole risk."""
    import statsmodels.api as sm

    rng = np.random.default_rng(3)
    X = sm.add_constant(rng.normal(size=(80, 3)))
    y = rng.poisson(np.exp(0.2 + X[:, 1] * 0.3))
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    summary = RS.build_run_summary(
        model=model, settings=_settings("poisson", "parametric"),
        coef_df=_parametric_coefficients("poisson"),
        regression_type="poisson", res_folder=run_folder)
    pseudo = summary.field("pseudo_r_squared")
    assert pseudo.answered
    assert "McFadden" in pseudo.value
    assert "PSEUDO-R2" in pseudo.value
    assert "must not be compared against an OLS R2" in pseudo.value
    r2 = summary.field("r_squared")
    assert not r2.answered
    assert "no R2" in r2.reason


@pytest.mark.parametrize("regression_type", NO_P_VALUE_TYPES)
def test_a_penalised_fit_refuses_a_training_r2_and_gives_the_bootstrap(
        regression_type, run_folder):
    summary = _summary(regression_type, "parametric", run_folder)
    r2 = summary.field("r_squared")
    assert not r2.answered
    assert "not a fit statistic" in r2.reason.lower()
    selection = summary.field("selection_frequency")
    assert selection.answered
    assert "median" in selection.value
    assert "0.6" in selection.value
    correction = summary.field("multiple_testing_method")
    assert correction.answered
    assert "no valid frequentist P" in correction.value


# ---------------------------------------------------------------------------
# 4. The design, and the identifiability warning that stays at the top
# ---------------------------------------------------------------------------

def test_n_cells_is_summed_over_wells_not_over_rows(run_folder):
    """The bug this counter is written to avoid, pinned.

    ``regression_data.csv`` repeats ``cell_count`` once per guide in the well,
    so summing the column multiplies the screen's objects by its guides per
    well — three times over on this fixture.
    """
    frame = pd.read_csv(os.path.join(run_folder, "regression_data.csv"))
    per_well = frame.drop_duplicates("prc")["cell_count"].sum()
    assert frame["cell_count"].sum() == 3 * per_well
    cells = _summary("ols", "parametric", run_folder).field("n_cells")
    assert cells.value.startswith(f"{int(per_well):,} objects")


def test_the_design_counts_come_from_the_run_folder(run_folder):
    summary = _summary("ols", "parametric", run_folder)
    assert summary.field("n_wells").value.startswith("24 distinct")
    assert summary.field("n_guides").value.startswith("12 distinct")
    assert summary.field("n_genes").value.startswith("4 distinct")
    assert summary.field("n_rows_fitted").value.startswith("72 rows")


def test_without_a_run_folder_the_counts_say_why_rather_than_zero(tmp_path):
    summary = RS.build_run_summary(
        settings=_settings("ols", "parametric"),
        coef_df=_parametric_coefficients("ols"), regression_type="ols",
        res_folder=str(tmp_path))
    for name in ("n_wells", "n_guides", "n_genes", "n_cells"):
        one = summary.field(name)
        assert not one.answered
        assert "regression_data.csv" in one.reason
        # NOT a zero. That is the failure this instruction is about.
        assert "0" != str(one.value)


def test_the_identifiability_warning_is_the_first_thing_in_the_summary(
        run_folder):
    """A wide fit prints a full table of standard errors regardless."""
    import statsmodels.api as sm

    rng = np.random.default_rng(4)
    X = sm.add_constant(rng.normal(size=(10, 14)))
    model = sm.OLS(rng.normal(size=10), X).fit()
    summary = RS.build_run_summary(
        model=model, settings=_settings("ols", "parametric"),
        coef_df=_parametric_coefficients("ols"), regression_type="ols",
        res_folder=run_folder)
    assert summary.warnings
    assert "NOT IDENTIFIABLE" in summary.warnings[0]
    text = summary.text()
    assert text.index("NOT IDENTIFIABLE") < text.index("WHAT WAS FITTED")
    assert summary.field("identifiable").value.startswith("NO")


def test_an_identifiable_fit_carries_no_warning(run_folder, ols_fit):
    summary = _summary("ols", "parametric", run_folder, model=ols_fit)
    assert summary.warnings == []
    assert summary.field("identifiable").value == "yes"


# ---------------------------------------------------------------------------
# 5. The call
# ---------------------------------------------------------------------------

def test_the_bh_critical_raw_p_is_the_correction_s_own_threshold(run_folder):
    """Not alpha. Drawing the line at alpha is the mistake it replaces."""
    from spacr.multiple_testing import critical_p_value

    coef = _parametric_coefficients("ols")
    summary = RS.build_run_summary(
        settings=_settings("ols", "parametric"), coef_df=coef,
        regression_type="ols", res_folder=run_folder)
    tested = coef["feature"].str.startswith("fraction:grna") \
        & coef["p_value"].notna()
    expected = critical_p_value(coef.loc[tested, "p_value"].to_numpy(),
                                method="fdr_bh", alpha=0.05)
    one = summary.field("critical_p")
    assert one.answered
    if expected is None:
        assert "no test was called" in one.value
    else:
        assert f"{expected:.4g}" in one.value
        assert "It is NOT alpha" in one.value


def test_the_intercept_and_the_position_terms_are_not_counted_as_tests(
        run_folder):
    summary = _summary("ols", "parametric", run_folder)
    tested = summary.field("n_tested")
    assert tested.answered
    assert tested.value.startswith("12 of 15")
    excluded = summary.field("untested_coefficients")
    assert excluded.value.startswith("3 of 15")


def test_no_correction_is_reported_as_a_correction_not_as_a_gap(run_folder):
    """``multiple_testing_method='none'`` is spaCR's default and a real answer.

    Read through a helper that treats 'none' as absent, this field said
    "neither the results table nor the settings record which correction was
    applied" about a run that recorded it twice — measured on the first real
    end-to-end run of this module.
    """
    settings = _settings("ols", "parametric")
    settings["multiple_testing_method"] = "none"
    coef = _parametric_coefficients("ols")
    coef["multiple_testing_method"] = "none"
    summary = RS.build_run_summary(settings=settings, coef_df=coef,
                                   regression_type="ols",
                                   res_folder=run_folder)
    one = summary.field("multiple_testing_method")
    assert one.answered
    assert one.value.startswith("none — NO correction was applied")


# ---------------------------------------------------------------------------
# 6. The type refuses a blank, and the builder cannot leave a hole
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {},                                         # neither
    {"value": "x", "reason": "y"},              # both
    {"value": ""},                              # blank value
    {"value": "   "},                           # whitespace
    {"reason": ""},                             # blank reason
])
def test_a_field_cannot_be_blank(kwargs):
    with pytest.raises(ValueError):
        RS.SummaryField("n", "label", **kwargs)


def test_a_not_assumed_field_must_carry_its_reason():
    with pytest.raises(ValueError):
        RS.SummaryField("n", "label", value="x", kind=RS.NOT_ASSUMED)


def test_a_builder_that_raises_still_leaves_every_field_answered(monkeypatch,
                                                                 run_folder):
    """A reporting bug must not be able to produce a summary with a hole."""
    def boom(_run):
        raise RuntimeError("planted")

    monkeypatch.setitem(RS._BUILDERS, "assumptions", boom)
    summary = _summary("ols", "parametric", run_folder)
    assert summary.missing() == []
    for one in summary.section("assumptions").fields:
        assert not one.answered
        assert "planted" in one.reason


def test_a_field_added_to_the_contract_is_backfilled_rather_than_dropped(
        monkeypatch, run_folder):
    monkeypatch.setitem(RS.CONTRACT, "call",
                        tuple(RS.CONTRACT["call"]) + ("invented",))
    monkeypatch.setitem(RS.LABELS, ("call", "invented"), "invented")
    summary = _summary("ols", "parametric", run_folder)
    assert summary.missing() == []
    one = summary.field("invented")
    assert not one.answered
    assert "defect in the summary" in one.reason


# ---------------------------------------------------------------------------
# 7. The Runs tab compares runs by columns the summary must report
# ---------------------------------------------------------------------------

def test_the_runs_tab_comparison_columns_all_have_a_summary_field():
    """"If those two disagree, one of them is wrong" -- instruction 156 C."""
    pytest.importorskip("PySide6")
    from spacr.qt.widgets.sweep_runs import PREFERRED_COLUMNS

    every = {name for names in RS.CONTRACT.values() for name in names}
    unanswered = []
    for column in PREFERRED_COLUMNS:
        if column in RS.NOT_A_FIT_PROPERTY:
            continue
        field_name = RS.COMPARISON_FIELDS.get(column)
        if field_name is None or field_name not in every:
            unanswered.append(column)
    assert unanswered == [], (
        f"the Runs tab compares runs by {unanswered}, and no summary field "
        f"reports it")


def test_the_comparison_mapping_names_no_field_that_does_not_exist():
    every = {name for names in RS.CONTRACT.values() for name in names}
    missing = {column: name for column, name in RS.COMPARISON_FIELDS.items()
               if name not in every}
    assert missing == {}


def test_those_comparison_fields_are_answered_on_a_real_run(run_folder,
                                                            ols_fit):
    summary = _summary("ols", "parametric", run_folder, model=ols_fit)
    for column, name in RS.COMPARISON_FIELDS.items():
        one = summary.field(name)
        assert one is not None, f"{column} -> {name} is not in the summary"
        assert one.text.strip()


# ---------------------------------------------------------------------------
# 8. Written with the run, and read back by the tab 153 already taught
# ---------------------------------------------------------------------------

def test_write_run_summary_writes_the_file_the_panel_reads(tmp_path):
    from spacr.ml import SUMMARY_FILENAME

    folder = tmp_path / "ols_1"
    folder.mkdir()
    _fitted_table().to_csv(folder / "regression_data.csv", index=False)
    path = RS.write_run_summary(
        str(folder), model=None, settings=_settings("ols", "nonparametric"),
        coef_df=_permutation_coefficients(), regression_type="ols")
    assert path == str(folder / SUMMARY_FILENAME)
    assert os.path.isfile(path)
    assert "R2 DOES NOT EXIST" in open(path).read()


def test_the_panel_reads_the_summary_back_from_the_run_folder(tmp_path):
    """No GUI change was needed; this is the assertion that says so."""
    pytest.importorskip("PySide6")
    from spacr.qt.widgets.regression_results import (find_summary_file,
                                                     summary_text)

    folder = tmp_path / "guide_permutation"
    folder.mkdir()
    _fitted_table().to_csv(folder / "regression_data.csv", index=False)
    pd.DataFrame({"feature": ["a"], "coefficient": [1.0]}).to_csv(
        folder / "results.csv", index=False)
    written = RS.write_run_summary(
        str(folder), model=None, settings=_settings("mixed", "nonparametric"),
        coef_df=_permutation_coefficients(), regression_type="mixed")

    assert find_summary_file(str(folder)) == written
    text = summary_text(None, "mixed", path=str(folder))
    assert not text.startswith("No summary")
    assert "spaCR RUN SUMMARY" in text
    assert "NOT ASSUMED" in text
    # And from the results CSV the panel is actually pointed at.
    assert "spaCR RUN SUMMARY" in summary_text(
        None, "mixed", path=str(folder / "results.csv"))


def test_the_summary_is_identical_when_it_is_read_back(tmp_path):
    folder = tmp_path / "ols_2"
    folder.mkdir()
    _fitted_table().to_csv(folder / "regression_data.csv", index=False)
    settings = _settings("ols", "nonparametric")
    coef = _permutation_coefficients()
    first = RS.write_run_summary(str(folder), settings=settings,
                                 coef_df=coef, regression_type="ols")
    written = open(first).read()
    again = RS.write_run_summary(str(folder), settings=settings,
                                 coef_df=coef, regression_type="ols")
    assert open(again).read() == written


def test_the_statsmodels_summary_survives_and_stays_verbatim(tmp_path,
                                                             ols_fit):
    """It is appended at the end, unchanged, never replaced."""
    from spacr.ml import SUMMARY_FILENAME, save_summary_to_file

    folder = tmp_path / "ols_3"
    folder.mkdir()
    _fitted_table().to_csv(folder / "regression_data.csv", index=False)
    # Exactly the order perform_regression uses: statsmodels first.
    save_summary_to_file(ols_fit, file_path=str(folder / SUMMARY_FILENAME))
    original = open(folder / SUMMARY_FILENAME).read().strip()

    path = RS.write_run_summary(
        str(folder), model=ols_fit, settings=_settings("ols", "parametric"),
        coef_df=_parametric_coefficients("ols"), regression_type="ols")
    text = open(path).read()
    assert "spaCR RUN SUMMARY" in text
    assert text.index("WHAT WAS FITTED") < text.index("OLS Regression Results")
    # VERBATIM: every line of the statsmodels block, in order, at the end.
    # Compared against the BYTES ON DISK rather than a fresh render -- a new
    # `summary()` differs in its `Time:` header (instruction 153's finding).
    #
    # AND THE CLOCK LINE IS EXCLUDED, because comparing to the bytes on disk
    # is not enough on its own: `write_run_summary` renders the model again,
    # so if the two renders land either side of a second boundary the `Time:`
    # header differs and this test fails. Measured 2026-08-19: one run in
    # five. Every other line is a property of the FIT and must match exactly;
    # the timestamp is a property of when it was printed.
    for line in original.splitlines():
        if "Time:" in line:
            continue
        assert line in text


def test_a_model_that_cannot_render_twice_keeps_the_summary_on_disk(tmp_path):
    """The statsmodels text is recovered from the file rather than dropped."""
    from spacr.ml import SUMMARY_FILENAME

    folder = tmp_path / "beta_1"
    folder.mkdir()
    (folder / SUMMARY_FILENAME).write_text("Beta Regression Results\n"
                                           "=======================\n")
    path = RS.write_run_summary(
        str(folder), model=None, settings=_settings("beta", "parametric"),
        coef_df=_parametric_coefficients("beta"), regression_type="beta")
    text = open(path).read()
    assert "Beta Regression Results" in text
    assert text.index("WHAT WAS FITTED") < text.index("Beta Regression Results")


def test_write_run_summary_without_a_folder_writes_nothing(tmp_path):
    assert RS.write_run_summary(None, settings=_settings("ols", "parametric")) \
        is None


# ---------------------------------------------------------------------------
# 9. One statement of the normality verdict, shared with the QC panel
# ---------------------------------------------------------------------------

def test_the_summary_and_the_qc_panel_report_the_same_normality(run_folder,
                                                                ols_fit):
    """A second copy is how the picture and the prose come to disagree."""
    import matplotlib.pyplot as plt

    import spacr.regression_qc as rq

    ctx = rq.context_from_model(ols_fit, regression_type="ols")
    figure, ax = plt.subplots()
    try:
        stats = rq.draw_panel("residual_distribution", ctx, ax)
    finally:
        plt.close(figure)
    summary = _summary("ols", "parametric", run_folder, model=ols_fit)
    value = summary.field("normality").value
    assert f"{stats['skew']:+.2f}" in value
    assert f"{stats['excess_kurtosis']:+.2f}" in value
    assert f"{stats['normality_p']:.3g}" in value


def test_the_normality_test_refuses_below_eight_and_says_so():
    from spacr.regression_qc import NORMALITY_TOO_FEW, residual_normality

    out = residual_normality(np.arange(5.0))
    assert out["test"] == NORMALITY_TOO_FEW
    assert not np.isfinite(out["normality_p"])
    assert np.isfinite(out["skew"])


def test_a_short_fit_reports_the_shape_without_inventing_a_p_value(run_folder):
    import statsmodels.api as sm

    rng = np.random.default_rng(7)
    X = sm.add_constant(rng.normal(size=(6, 2)))
    model = sm.OLS(rng.normal(size=6), X).fit()
    summary = RS.build_run_summary(
        model=model, settings=_settings("ols", "parametric"),
        coef_df=_parametric_coefficients("ols"), regression_type="ols",
        res_folder=run_folder)
    one = summary.field("normality")
    assert one.answered
    assert "needs n >= 8" in one.value
    assert "so there is no P value" in one.value


def test_the_nonparametric_path_reaches_the_summary_writer():
    """The call site must be BEFORE `guide_permutation`'s early return.

    THE BUG THIS PINS. The summary call was placed at the end of the
    parametric path, and `perform_regression` returns from the
    `analysis_mode == 'guide_permutation'` branch long before it -- so the one
    mode with no statsmodels summary to fall back on was also the one mode
    that wrote no spaCR summary, which is the run the maintainer reported.

    Asserted on the SOURCE rather than by running a fit: a permutation run
    needs a screen, and what is wrong here is an ordering, which the source
    states exactly.
    """
    import inspect
    import re
    from spacr import ml

    # THE BODY. Instruction 161 made `perform_regression` a wrapper that reports
    # failures and delegates, so the public name no longer contains the
    # branches this test is about. The ordering did not move; the place to look
    # for it did -- the same correction test_regression_entry_points.py needed.
    source = inspect.getsource(ml._perform_regression)
    call = source.find("write_run_summary(")
    early_return = re.search(
        r"if settings\.get\('analysis_mode'\) == 'guide_permutation':", source)
    assert early_return is not None, "the permutation branch moved"
    assert call != -1, "perform_regression no longer writes a run summary"
    # The FIRST write_run_summary call has to come after the branch opens and
    # before that branch's own `return output`.
    branch = source[early_return.start():]
    assert "write_run_summary(" in branch.split("return output")[0], (
        "write_run_summary is not reached on the guide_permutation path: it "
        "sits after the early return, which is the bug this test exists for")


# ---------------------------------------------------------------------------
# The exclusion counts are recorded, not only printed (156's trailing note)
# ---------------------------------------------------------------------------

def test_process_reads_records_what_it_dropped(tmp_path):
    """The count has to survive the run, not just the console.

    `process_reads` computed `removed` and printed it. A console scrolls, and
    the run somebody asks about tomorrow needs the number.
    """
    import pandas as pd
    from spacr.ml import process_reads

    path = tmp_path / "counts.csv"
    pd.DataFrame({
        "plateID": ["plate1"] * 4, "rowID": ["r1", "r1", "r2", "r2"],
        "columnID": ["c1", "c1", "c1", "c1"],
        "grna": ["g1", "g2", "g1", "g3"],
        "count": [95, 5, 50, 50],
    }).to_csv(path, index=False)

    record = {}
    out = process_reads(str(path), 0.1, None, record=record)
    assert record["fraction_threshold"] == 1
    assert record["fraction_threshold_of"] == 4
    assert len(out) == 3


def test_the_record_accumulates_across_plates(tmp_path):
    """It runs once per plate, so the count is a sum rather than the last one."""
    import pandas as pd
    from spacr.ml import process_reads

    record = {}
    for plate in ("plate1", "plate2"):
        path = tmp_path / f"{plate}.csv"
        pd.DataFrame({
            "plateID": [plate] * 4, "rowID": ["r1", "r1", "r2", "r2"],
            "columnID": ["c1", "c1", "c1", "c1"],
            "grna": ["g1", "g2", "g1", "g3"],
            "count": [95, 5, 50, 50],
        }).to_csv(path, index=False)
        process_reads(str(path), 0.1, None, record=record)

    assert record["fraction_threshold"] == 2
    assert record["fraction_threshold_of"] == 8


def test_the_summary_reports_a_recorded_count(tmp_path):
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf

    from spacr.regression_summary import write_run_summary

    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"y": rng.normal(0, 1, 200), "x": rng.normal(0, 1, 200)})
    model = smf.ols("y ~ x", frame).fit()
    path = write_run_summary(
        str(tmp_path), model=model,
        settings={"regression_type": "ols", "fdr_alpha": 0.05,
                  "multiple_testing_method": "fdr_bh",
                  "fraction_threshold": 0.1,
                  "_regression_exclusions": {"fraction_threshold": 137,
                                             "fraction_threshold_of": 1945}},
        coef_df=pd.DataFrame({"feature": ["x"], "coefficient": [0.1],
                              "p_value": [0.2]}),
        regression_type="ols")
    text = open(path).read()
    assert "137 of 1,945" in text


def test_nothing_recorded_is_not_reported_as_zero(tmp_path):
    """"No row was dropped" and "nobody counted" are opposite findings.

    A summary that spelled the second as the first would understate what the
    fit was actually given.
    """
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf

    from spacr.regression_summary import write_run_summary

    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"y": rng.normal(0, 1, 200), "x": rng.normal(0, 1, 200)})
    model = smf.ols("y ~ x", frame).fit()
    path = write_run_summary(
        str(tmp_path), model=model,
        settings={"regression_type": "ols", "fdr_alpha": 0.05,
                  "multiple_testing_method": "fdr_bh",
                  "fraction_threshold": 0.1},
        coef_df=pd.DataFrame({"feature": ["x"], "coefficient": [0.1],
                              "p_value": [0.2]}),
        regression_type="ols")
    text = open(path).read()
    assert "did not record it" in text
    assert "0 of" not in text

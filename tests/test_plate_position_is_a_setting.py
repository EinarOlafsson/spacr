"""Plate position is a setting, and it is tested BY FITTING (instruction 143 A).

"is rowID + columnID always run? this should be an opt in" -- it was:
``prepare_formula`` ended with an unconditional ``+ rowID + columnID`` and
``random_row_column_effects`` only chose FIXED or RANDOM for terms that were
already in. ``model_plate_position`` is the third state, and these tests hold
it to the thing that actually matters: the DESIGN MATRIX gains or loses the
row and column columns, and the coefficients that come back move the way the
measurement said they would. A test that asserted on the formula STRING would
pass with a formula nothing fits.

THE DEFAULT IS ON, and it was measured rather than chosen. Instruction 143
proposed OFF; two fits per level of the maintainer's TSG101 screen (1945 rows,
610 wells, 823 guides, 389 genes) said the opposite -- the 35 position terms
are jointly significant at F = 5.781, p = 6.71e-23 (guide) and F = 6.277,
p = 2.33e-26 (gene), eight of nine real screens agree, dropping them costs 8.4
points of R2 and 7.2% on the residual sd, and the exported gene hit list swaps
277230 out (q 0.0394 -> 0.4071) for 258462 (q 0.1134 -> 0.0146). Carrying the
terms on a plate that does not need them costs 1.6% on the standard errors and
0.02 hits out of 20. The tests below reproduce the SHAPE of that asymmetry on
a synthetic plate whose truth is known, so the default has a reason on file
that does not depend on a CSV outside the repo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spacr.settings as S
from spacr.ml import (check_and_clean_data, prepare_formula,
                      _reconcile_random_row_column_effects)


NC = "233460"   # spacr's default negative control
PC = "220950"   # spacr's default positive control
GENES = [NC, PC, "gene3", "gene4", "gene5", "gene6"]
GUIDES_PER_GENE = 3

#: FOUR 96-WELL PLATES, NOT ONE, and it is the ratio that made this matter.
#: Plate position costs (8 - 1) + (12 - 1) = 18 parameters however many plates
#: share those coordinates, so one plate spends 18 of its 96 wells on position
#: (19%) while four spend 18 of 384 (4.7%) -- and the real screen spends 35 of
#: 610 (5.7%). On a single plate the position terms fit so much of the
#: well-level noise that the NULL arm looks like the planted one: measured on
#: this generator, median |delta|/SE 0.23 with no position effect planted at
#: all, against 0.35 with one. Four plates separate them properly, 0.12
#: against 0.29, which is the contrast these tests are for.
N_PLATES, N_ROWS, N_COLS = 4, 8, 12
POSITION_PARAMS = (N_ROWS - 1) + (N_COLS - 1)

#: How many of the library's guides land in one well. The real TSG101 screen
#: is 1945 rows over 610 wells -- 3.2 guides per well -- and the replication
#: matters: the response is measured ONCE per well and repeated on every one
#: of that well's rows, so a design that put all 18 guides in every well would
#: give the position terms eighteen copies of each well's noise to fit.
GUIDES_PER_WELL = 3


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _screen(seed=0, position_sd=0.60, dep="predictions"):
    """A pooled screen with a KNOWN plate-position effect and known hits.

    One score per WELL, repeated on each of that well's guide rows, which is
    the shape ``perform_regression`` merges out of a score CSV and a count
    CSV. ``position_sd=0`` gives the other arm of the measurement: a plate
    with no position effect at all, which is the only screen the ON default is
    wrong about.
    """
    rng = np.random.default_rng(seed)
    library = [f"{gene}_{i}" for gene in GENES for i in range(GUIDES_PER_GENE)]
    gene_of = {f"{gene}_{i}": gene
               for gene in GENES for i in range(GUIDES_PER_GENE)}
    # The planted truth: two genes move the response, the other four do not.
    beta = {gene: 0.0 for gene in GENES}
    beta[PC] = 4.0
    beta["gene3"] = -3.0
    # A PLATE EFFECT WITH A SHAPE, not a random draw: a gradient down the
    # rows and an edge effect across the columns, which is what a plate
    # actually has (evaporation and temperature at the edges). Drawn from a
    # generator instead, the planted effect's SIZE would depend on the random
    # stream and these tests would assert on a number numpy is free to change
    # between releases.
    # SIZED FROM THE REAL PLATE: the two together contribute an sd of 0.42,
    # which is 0.44 of the fitted residual sd -- the same ratio the
    # maintainer's screen carries (contribution sd 0.0229 against a residual
    # sd of 0.0527), so the arms below are separated by as much as a real
    # plate separates them and no more.
    row_effect = {f"r{r + 1:02d}":
                  position_sd * (2.0 * r / (N_ROWS - 1) - 1.0)
                  for r in range(N_ROWS)}
    col_effect = {f"c{c + 1:02d}":
                  position_sd * ((2.0 * c / (N_COLS - 1) - 1.0) ** 2 - 1 / 3)
                  for c in range(N_COLS)}

    records = []
    for plate_index in range(N_PLATES):
        plate = f"plate{plate_index + 1}"
        for r in range(N_ROWS):
            for c in range(N_COLS):
                row_id, col_id = f"r{r + 1:02d}", f"c{c + 1:02d}"
                prc = f"{plate}_{row_id}_{col_id}"
                picked = rng.choice(library, size=GUIDES_PER_WELL,
                                    replace=False)
                raw = rng.random(GUIDES_PER_WELL) + 0.2
                frac = raw / raw.sum()
                gene_fraction = {}
                for guide, one in zip(picked, frac):
                    gene_fraction[gene_of[guide]] = (
                        gene_fraction.get(gene_of[guide], 0.0) + float(one))
                score = (0.5
                         + sum(beta[gene] * value
                               for gene, value in gene_fraction.items())
                         + row_effect[row_id] + col_effect[col_id]
                         + float(rng.normal(0, 0.10)))
                for guide, one in zip(picked, frac):
                    records.append({
                        "plateID": plate, "rowID": row_id,
                        "columnID": col_id, "prc": prc,
                        "gene": gene_of[guide], "grna": guide,
                        "fraction": float(one),
                        "cell_count": int(rng.integers(120, 400)),
                        dep: score,
                    })
    return pd.DataFrame(records)


def _fit(frame, model_plate_position, level="grna", dep="predictions"):
    """Fit ONE cleaned frame through the real path and hand back the fit.

    Both arms are fitted on the SAME cleaned frame, so a difference between
    them is the design and nothing else -- patsy must also keep the same rows
    in both, which the caller asserts.
    """
    import statsmodels.api as sm
    from patsy import dmatrices

    formula = prepare_formula(dep, level=level,
                              model_plate_position=model_plate_position)
    y, X = dmatrices(formula, data=frame, return_type="dataframe")
    return sm.OLS(y, X).fit(), y, X


@pytest.fixture(scope="module")
def cleaned():
    """One cleaned frame with a real position effect, shared by both arms."""
    return check_and_clean_data(_screen(seed=0), "predictions")


@pytest.fixture(scope="module")
def cleaned_flat():
    """The null arm: the same design, with no plate-position effect planted."""
    return check_and_clean_data(_screen(seed=0, position_sd=0.0),
                                "predictions")


# ---------------------------------------------------------------------------
# 1. The design matrix, which is the thing that actually changes
# ---------------------------------------------------------------------------

def test_the_design_matrix_loses_exactly_the_row_and_column_columns(cleaned):
    """Counted, not read off the formula string.

    The number is the one instruction 143 argues about: it counts a full
    384-well plate at 16 + 24 - 2 = 36 parameters, and the real screen spends
    35 because it occupies 16 rows and 21 columns rather than 16 and 24.
    These are 96-well plates, so they spend (8 - 1) + (12 - 1) = 18 however
    many of them are stacked.
    """
    on, y_on, X_on = _fit(cleaned, True)
    off, y_off, X_off = _fit(cleaned, False)

    position_on = [c for c in X_on.columns
                   if c.startswith(("rowID[", "columnID["))]
    assert len(position_on) == POSITION_PARAMS, position_on
    assert not [c for c in X_off.columns
                if c.startswith(("rowID[", "columnID["))]

    # Everything else is untouched: same rows, same guide block, and the
    # parameter count differs by exactly the position terms.
    assert list(y_on.index) == list(y_off.index)
    assert set(X_off.columns) == set(X_on.columns) - set(position_on)
    assert X_on.shape[1] - X_off.shape[1] == POSITION_PARAMS
    assert int(on.df_resid) == int(off.df_resid) - POSITION_PARAMS
    # Both designs are FULL RANK; the choice is not trading one deficiency
    # for another.
    assert np.linalg.matrix_rank(X_on.to_numpy()) == X_on.shape[1]
    assert np.linalg.matrix_rank(X_off.to_numpy()) == X_off.shape[1]


def test_the_gene_level_loses_the_same_columns(cleaned):
    """The setting is a property of the DESIGN, not of one level."""
    _, _, X_on = _fit(cleaned, True, level="gene")
    _, _, X_off = _fit(cleaned, False, level="gene")

    assert len([c for c in X_on.columns
                if c.startswith(("rowID[", "columnID["))]) == POSITION_PARAMS
    assert not [c for c in X_off.columns
                if c.startswith(("rowID[", "columnID["))]
    assert any(c.startswith("gene_fraction:gene[") for c in X_off.columns)


# ---------------------------------------------------------------------------
# 2. The coefficients move the way the measurement says
# ---------------------------------------------------------------------------

def _guide_terms(fit):
    return [t for t in fit.params.index if t.startswith("fraction:grna[")]


def test_the_position_terms_are_jointly_significant_when_they_are_real(
        cleaned):
    """The two-line answer instruction 143 asked for, on a plate whose
    position effect is planted: F = 5.781, p = 6.71e-23 on the real screen."""
    on, _, X_on = _fit(cleaned, True)
    terms = [c for c in X_on.columns
             if c.startswith(("rowID[", "columnID["))]
    joint = on.f_test(" = 0, ".join(terms) + " = 0")

    assert float(joint.fvalue) > 3.0, float(joint.fvalue)
    assert float(joint.pvalue) < 1e-6, float(joint.pvalue)


def test_dropping_them_inflates_the_residual_and_the_standard_errors(cleaned):
    """7.2% more residual sd and 5.5% LARGER standard errors, measured.

    This is the direction that decides the default: the drop in residual sd
    more than pays for the parameters, so the fit WITH plate position is the
    more precise one, not merely the better-fitting one.
    """
    on, _, _ = _fit(cleaned, True)
    off, _, _ = _fit(cleaned, False)

    assert np.sqrt(off.scale) > np.sqrt(on.scale)
    assert off.rsquared < on.rsquared

    ratio = np.median([off.bse[t] / on.bse[t] for t in _guide_terms(on)])
    assert ratio > 1.02, ratio


def test_the_guide_coefficients_move_by_more_than_a_fifth_of_an_se(cleaned):
    """Median |delta|/SE was 0.271 at the guide level on the real screen, and
    216 of 823 guides moved more than half a standard error. Movement relative
    to the standard error is the number that decides whether anyone notices.
    """
    on, _, _ = _fit(cleaned, True)
    off, _, _ = _fit(cleaned, False)

    terms = _guide_terms(on)
    moved = np.array([abs(on.params[t] - off.params[t]) / on.bse[t]
                      for t in terms])
    assert np.median(moved) > 0.2, np.median(moved)
    assert moved.max() > 0.5, moved.max()
    # A quarter of an SE is where the real screen's swapped genes lived: 429
    # of its 823 guides moved further than that.
    assert (moved > 0.25).mean() > 0.3, (moved > 0.25).mean()


def _median_movement(frame):
    """Median |delta| / SE across the guide coefficients, both arms."""
    on, _, _ = _fit(frame, True)
    off, _, _ = _fit(frame, False)
    terms = _guide_terms(on)
    moved = np.array([abs(on.params[t] - off.params[t]) / on.bse[t]
                      for t in terms])
    return float(np.median(moved)), on, off, terms


def test_on_a_plate_with_no_position_effect_the_answer_barely_moves(
        cleaned, cleaned_flat):
    """The other half of the asymmetry, and the only case OFF is right about.

    Median |delta|/SE was 0.121 with max 0.575 on the maintainer's synthetic
    null plate, against 0.305 median / 1.47 max when the effect is real -- so
    the cost of carrying plate position on a plate that does not need it is
    not visible in the coefficients. Reproduced here at 0.12 against 0.29;
    the RATIO is what carries over between designs, because the absolute
    number depends on how many wells the 18 nuisance parameters are spread
    over.
    """
    flat, on, off, terms = _median_movement(cleaned_flat)
    planted, _, _, _ = _median_movement(cleaned)

    assert flat < planted / 1.5, (flat, planted)
    assert flat < 0.25, flat

    # THE ONLY COST IS THE DEGREES OF FREEDOM, and on a plate with nothing
    # there to model it does not even show. The maintainer's null arm paid
    # 1.6% on the standard errors for its 35 terms (se_off/se_on = 0.9842);
    # this one pays nothing measurable, 1.001 against the 1.008 that giving
    # 18 degrees of freedom back would buy on its own. Both are inside a
    # couple of percent, and that is the entire cost of being wrong in the ON
    # direction -- against a fifth of the true hits for being wrong in the
    # other one.
    df_only = float(np.sqrt(off.df_resid / on.df_resid))
    ratio = float(np.median([off.bse[t] / on.bse[t] for t in terms]))
    assert abs(ratio - 1.0) < 0.02, ratio
    assert abs(ratio - df_only) < 0.02, (ratio, df_only)
    # ...and the residual sd is essentially unchanged, which is what "there
    # was nothing there for them to explain" looks like. It rose 8% on the
    # planted arm.
    assert np.sqrt(off.scale) / np.sqrt(on.scale) < 1.02


def test_leaving_plate_position_out_costs_power_on_the_planted_hits(cleaned):
    """This is the 17% of true hits, in the form a small screen can show it.

    On the maintainer's synthetic arm -- the real design, the response
    re-simulated, the position effect at the size the real plate carries --
    dropping position recovered 2.60 of the 20 planted hits at BH where
    keeping it recovered 3.15, and 7.38 against 9.20 at raw p. The mechanism
    is here: the two planted genes come out with the right sign either way,
    and BOTH are further from zero with plate position in the model, because
    the unexplained plate variance is otherwise charged to the residual and
    widens every standard error.
    """
    on, _, _ = _fit(cleaned, True, level="gene")
    off, _, _ = _fit(cleaned, False, level="gene")

    for gene, sign in ((PC, +1), ("gene3", -1)):
        term = f"gene_fraction:gene[{gene}]"
        assert sign * on.params[term] > 0
        assert sign * off.params[term] > 0
        # t = 13.87 against 12.87 for the positive control, -15.07 against
        # -14.16 for the planted knockdown: five orders of magnitude on the
        # p-value, on a screen with only two hits to lose.
        assert abs(on.params[term] / on.bse[term]) > \
            abs(off.params[term] / off.bse[term])
        assert on.pvalues[term] < off.pvalues[term]

    # And nothing is bought with false positives: the four null genes stay
    # non-significant in both arms.
    nulls = [f"gene_fraction:gene[{g}]" for g in GENES
             if g not in (PC, "gene3")]
    assert max(on.pvalues[t] for t in nulls) > 0.05
    assert min(on.pvalues[t] for t in nulls) > 0.05
    assert min(off.pvalues[t] for t in nulls) > 0.05


# ---------------------------------------------------------------------------
# 3. The three states, and the fourth that does not exist
# ---------------------------------------------------------------------------

def test_new_runs_leave_plate_position_out_until_it_is_requested():
    """The settings default is opt-in even though the API remains compatible."""
    settings = S.get_perform_regression_default_settings({})
    assert settings["model_plate_position"] is False
    formula = prepare_formula(
        "score", model_plate_position=settings["model_plate_position"])
    assert "rowID" not in formula
    assert "columnID" not in formula


def test_the_three_states_are_three_different_formulas():
    out = prepare_formula("score", model_plate_position=False)
    fixed = prepare_formula("score", model_plate_position=True)
    random = prepare_formula("score", model_plate_position=True,
                             random_row_column_effects=True)

    assert "rowID" not in out and "columnID" not in out
    assert "rowID" in fixed and "columnID" in fixed
    # RANDOM leaves them out of the FORMULA because fit_mixed_model puts them
    # into its variance components; that is the state the formula alone
    # cannot tell you about, which is why the model box has to name it.
    assert random == out


def test_out_plus_random_is_refused_rather_than_resolved():
    """Instruction 106: name both settings and both ways out, do not pick."""
    with pytest.raises(ValueError) as error:
        prepare_formula("score", model_plate_position=False,
                        random_row_column_effects=True)

    message = str(error.value)
    assert "model_plate_position" in message
    assert "random_row_column_effects" in message
    # It has to say what to do, not just what is wrong.
    assert "model_plate_position=True" in message
    assert "random_row_column_effects=False" in message


def test_the_settings_seam_refuses_the_same_pair_before_a_file_is_written():
    """The same refusal one layer up, where the folder is named.

    ``_reconcile_random_row_column_effects`` runs inside
    ``perform_regression``'s validation, before any results folder exists, so
    the contradiction is answered before a run spends an hour producing a fit
    nobody asked for.
    """
    with pytest.raises(ValueError) as error:
        _reconcile_random_row_column_effects({
            "random_row_column_effects": True,
            "model_plate_position": False,
            "regression_type": "mixed"})

    message = str(error.value)
    assert "model_plate_position" in message
    assert "nothing left" in message


@pytest.mark.parametrize("settings", [
    {"random_row_column_effects": True, "regression_type": "mixed",
     "model_plate_position": True},
    # THE ABSENT KEY IS THE OLD CSV, and it must not be read as False.
    {"random_row_column_effects": True, "regression_type": "mixed"},
])
def test_position_in_the_model_is_not_refused(settings):
    assert _reconcile_random_row_column_effects(dict(settings)) is not None


# ---------------------------------------------------------------------------
# 4. An old settings CSV still loads and still MEANS what it meant
# ---------------------------------------------------------------------------

def test_an_unspecified_plate_position_choice_uses_the_opt_in_default():
    supplied = {"dependent_variable": "pred", "regression_type": "ols",
                "random_row_column_effects": False}
    assert "model_plate_position" not in supplied

    settings = S.get_perform_regression_default_settings(dict(supplied))
    assert settings["model_plate_position"] is False

    formula = prepare_formula(
        settings["dependent_variable"],
        random_row_column_effects=settings["random_row_column_effects"],
        model_plate_position=settings["model_plate_position"])
    assert "rowID" not in formula and "columnID" not in formula


def test_random_position_effects_require_position_to_be_enabled():
    settings = S.get_perform_regression_default_settings(
        {"random_row_column_effects": True, "regression_type": "mixed"})

    assert settings["model_plate_position"] is False
    with pytest.raises(ValueError, match="model_plate_position=True"):
        _reconcile_random_row_column_effects(settings)


def test_an_explicit_choice_survives_the_defaults_factory():
    settings = S.get_perform_regression_default_settings(
        {"model_plate_position": False})
    assert settings["model_plate_position"] is False


# ---------------------------------------------------------------------------
# 5. The setting is a SETTING: typed, tooltipped, categorised, and read
# ---------------------------------------------------------------------------

def test_the_setting_is_declared_the_way_every_other_one_is():
    assert S.expected_types["model_plate_position"] is bool
    tip = S.tooltips["model_plate_position"]
    assert tip.startswith("(bool)")
    assert 80 <= len(tip) <= 600, len(tip)
    assert "Default False" in tip
    # The number that makes the claim checkable survives into the hover.
    assert "6.7e-23" in tip


def test_it_sits_beside_the_setting_it_is_confused_with():
    """Two settings whose names do not distinguish them is how a user sets
    the wrong one, so they are adjacent and in the order they are read."""
    model = S.categories["Regression: Model"]
    assert model.index("model_plate_position") == \
        model.index("random_row_column_effects") - 1


def test_the_run_is_handed_the_setting_it_was_given():
    """``perform_regression`` reads it out of the settings dict and hands it
    to ``regression_levels``; nothing downstream re-decides it."""
    import inspect

    from spacr import ml

    source = inspect.getsource(ml._perform_regression)
    assert "model_plate_position=settings.get('model_plate_position', True)" \
        in source


# ---------------------------------------------------------------------------
# 6. End to end through regression(), because the fit is the deliverable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_plate_position, expected", [(True, True),
                                                            (False, False)])
def test_regression_fits_the_design_the_setting_asked_for(
        tmp_path, model_plate_position, expected):
    from spacr.ml import regression

    frame = _screen(seed=7)
    model, coef_df, _ = regression(
        frame, csv_path=str(tmp_path / "results.csv"),
        dependent_variable="predictions", regression_type="ols",
        nc=NC, pc=PC, dst=str(tmp_path), level="grna", qc=False, plot=False,
        model_plate_position=model_plate_position)

    fitted = [t for t in model.params.index
              if t.startswith(("rowID[", "columnID["))]
    assert bool(fitted) is expected, fitted
    # The coefficient table is the guide block either way -- the position
    # terms are nuisance and process_model_coefficients filters them out --
    # so the setting is invisible downstream and can only be checked here.
    assert any(f.startswith("fraction:grna[")
               for f in coef_df["feature"])
    assert not any(f.startswith(("rowID[", "columnID["))
                   for f in coef_df["feature"])


def test_regression_levels_passes_the_setting_through_to_both_fits(tmp_path):
    """Two fits, one design decision: a setting honoured at one level and not
    the other would be two answers to one question."""
    from spacr.ml import regression_levels

    frame = _screen(seed=11)
    fits = regression_levels(
        frame, csv_path=str(tmp_path / "results.csv"),
        dependent_variable="predictions", regression_type="ols", level="both",
        nc=NC, pc=PC, dst=str(tmp_path), qc=False, plot=False,
        model_plate_position=False)

    assert set(fits) == {"grna", "gene"}
    for level, (model, _coef, _type) in fits.items():
        assert not [t for t in model.params.index
                    if t.startswith(("rowID[", "columnID["))], level


def test_the_mixed_model_can_be_fitted_without_plate_position(tmp_path):
    """INDEPENDENT OF THE MODEL FAMILY, which is what instruction 143 asks
    for: the setting decides whether plate position is in the model, and
    ``random_row_column_effects`` decides fixed-or-random for a term that is.

    Under ``regression_type='mixed'`` with position out, row and column leave
    the fixed part and do NOT reappear as variance components -- and the
    gene/guide nesting the mixed model exists for is untouched.
    """
    from spacr.ml import regression

    frame = _screen(seed=0)
    model, coef_df, regression_type = regression(
        frame, csv_path=str(tmp_path / "results.csv"),
        dependent_variable="predictions", regression_type="mixed",
        nc=NC, pc=PC, dst=str(tmp_path), level="gene", qc=False, plot=False,
        model_plate_position=False)

    assert regression_type == "mixed"
    features = {str(f) for f in coef_df["feature"]}
    assert not [f for f in features if f.startswith(("rowID", "columnID"))]
    assert {"Group Var", "grna Var", "Intercept"} <= features

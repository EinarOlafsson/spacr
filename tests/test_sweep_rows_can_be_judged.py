"""A sweep row must be enough to CHOOSE a configuration, not just count one.

The defect these guard, in the user's words: "a sweep row today carries the
settings, a hit count, and seconds. That is not enough to choose between two
runs: it cannot say whether a fit with more hits had a worse-behaved residual,
a rank-deficient design, or simply threw data away." The nightly run over the
TSG101 screen was the concrete case -- ``min_cell_count=50`` reported MORE hits
than 100 while quietly losing GRA14, and no column in the table said so.

The acceptance test the instruction names is "sorting the sweep table by the
positive control's rank puts the best configuration first", so that is tested
directly rather than by proxy.
"""

import numpy as np
import pandas as pd
import pytest

from spacr.parameter_sweep import (
    SweepSpace, rank_trials, run_sweep, settings_for_trial, summarise_sweep,
)
from spacr.trial_metrics import METRIC_COLUMNS, summarise_trial


# --------------------------------------------------------------------------
# a fit shaped like the one perform_regression returns
# --------------------------------------------------------------------------

def _fitted(n_rows=120, n_guides=12, seed=0, pc="239740", full_rank=True):
    """A real statsmodels fit on a design shaped like a well-level screen."""
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(seed)
    guides = [f"{pc}_1", f"{pc}_2"] + [f"2000{i:02d}_1" for i in range(n_guides - 2)]
    if full_rank:
        values = rng.normal(size=(n_rows, len(guides)))
    else:
        # Guide fractions that sum to 1 in every well: collinear with the
        # intercept, which is how a real fraction design goes rank deficient.
        values = rng.dirichlet(np.ones(len(guides)), size=n_rows)
    design = pd.DataFrame(values, columns=[f"fraction:grna[{g}]" for g in guides])
    design.insert(0, "Intercept", 1.0)
    # The positive control genuinely carries the signal.
    y = values[:, 0] * 3.0 + rng.normal(0, 1, n_rows)
    model = sm.OLS(y, design).fit()
    results = pd.DataFrame({
        "feature": design.columns,
        "coefficient": model.params.to_numpy(),
        "p_value": model.pvalues.to_numpy(),
    })
    results["q_value"] = np.clip(results["p_value"] * len(results), 0, 1)
    results["grna"] = [None] + guides
    results["gene"] = [None] + [g.split("_")[0] for g in guides]
    # spaCR's own control annotation, written by spacr.ml.regression.
    results["condition"] = ["other"] + [
        "pc" if g.startswith(pc) else "other" for g in guides]
    return {"results": results, "model": model,
            "model_data": pd.DataFrame({"prc": [f"w{i}" for i in range(n_rows)],
                                        "grna": rng.choice(guides, n_rows)}),
            "significant": results[results["q_value"] < 0.05]}


SETTINGS = {"positive_control": "239740", "negative_control": "233460",
            "fdr_alpha": 0.05}


# --------------------------------------------------------------------------
class TestTheRowSaysWhetherTheFitDeservesBelief:
    """"10 hits" does not say whether the fit that produced them was sound."""

    def test_a_row_carries_fit_quality_residual_and_design_columns(self):
        """A row with a hit count and nothing else cannot rank two runs.

        Every block the instruction enumerates -- FIT QUALITY, RESIDUALS,
        DESIGN, CONTROLS, CALIBRATION, GUIDE SUPPORT -- has to be on the row,
        or the user is back to opening every trial folder by hand.
        """
        row = summarise_trial(_fitted(), SETTINGS)
        for column in ("r_squared", "aic", "bic", "log_likelihood",
                       "residual_se",                        # fit quality
                       "durbin_watson", "jarque_bera_p",
                       "breusch_pagan_p", "residual_trend_slope",  # residuals
                       "design_rank", "design_identifiable",
                       "n_parameters", "n_wells",             # design
                       "positive_control_rank",               # controls
                       "genomic_inflation", "p_first_bin_excess"):  # calibration
            assert column in row, f"{column} missing from the sweep row"

    def test_a_rank_deficient_design_is_named_as_such(self):
        """A fit with more parameters than wells still returns coefficients.

        statsmodels falls back to a pseudo-inverse rather than refusing, so
        nothing else in the output says the per-guide effects are one of
        infinitely many solutions. If the row does not say it, nobody learns
        it.
        """
        deficient = summarise_trial(_fitted(full_rank=False), SETTINGS)
        assert deficient["design_identifiable"] is False
        assert deficient["non_identifiable_directions"] >= 1
        healthy = summarise_trial(_fitted(full_rank=True), SETTINGS)
        assert healthy["design_identifiable"] is True
        assert healthy["non_identifiable_directions"] == 0

    def test_no_vif_is_reported_for_a_design_that_cannot_support_one(self):
        """VIF off a pseudo-inverse is a number with no meaning.

        Reporting one would put a fabricated collinearity figure in the column
        a user sorts on, which is worse than leaving it blank.
        """
        row = summarise_trial(_fitted(full_rank=False), SETTINGS)
        assert "max_vif" not in row
        assert summarise_trial(_fitted(full_rank=True), SETTINGS)["max_vif"] > 0

    def test_max_vif_agrees_with_the_reference_implementation(self):
        """The cheap VIF must be the same number, not merely a similar one."""
        from spacr.regression_diagnostics import variance_inflation_factors

        output = _fitted(n_rows=200, n_guides=10, full_rank=True)
        row = summarise_trial(output, SETTINGS)
        design = pd.DataFrame(
            np.asarray(output["model"].model.exog)[:, 1:],
            columns=list(output["results"]["feature"])[1:])
        reference = variance_inflation_factors(design)
        assert row["max_vif"] == pytest.approx(
            float(reference["vif"].max()), rel=1e-8)


# --------------------------------------------------------------------------
class TestThePositiveControlIsTheYardstick:
    """The column the instruction says the table must be sortable by."""

    def test_the_row_carries_the_positive_controls_rank_p_and_q(self):
        row = summarise_trial(_fitted(), SETTINGS)
        assert row["positive_control_found"] is True
        assert row["positive_control_rank"] == 1      # it carries the signal
        assert 0.0 <= row["positive_control_percentile"] < 1.0
        assert row["positive_control_p"] < 0.05
        assert "positive_control_q" in row

    def test_an_absent_control_gets_no_rank_rather_than_a_made_up_one(self):
        """The default negative control does not occur in the TSG101 screen.

        A fabricated rank in the column the sweep is judged on is the worst
        possible failure mode, so absence must read as absence.
        """
        row = summarise_trial(_fitted(), SETTINGS)
        assert row["negative_control_found"] is False
        assert "negative_control_rank" not in row
        assert "negative_control_percentile" not in row

    def test_the_rank_is_read_from_spacrs_own_control_annotation(self):
        """The row and the volcano must not disagree about which gene it is.

        spacr.ml.regression writes 'pc' into the condition column. Matching the
        identifier against the feature text a second time is a second chance to
        get a different answer -- here the feature names are opaque and only
        the annotation identifies the control.
        """
        output = _fitted()
        output["results"]["feature"] = [
            "Intercept"] + [f"fraction:grna[opaque_{i}]"
                            for i in range(len(output["results"]) - 1)]
        row = summarise_trial(output, SETTINGS)
        assert row["positive_control_found"] is True
        assert row["positive_control_rank"] == 1

    def test_a_table_with_no_annotation_still_finds_the_control_by_name(self):
        """A permutation run writes no condition column."""
        output = _fitted()
        output["results"] = output["results"].drop(columns=["condition"])
        row = summarise_trial(output, SETTINGS)
        assert row["positive_control_found"] is True
        assert row["positive_control_rank"] == 1

    def test_percentile_is_comparable_where_raw_rank_is_not(self):
        """Rank 3 of 20 and rank 3 of 200 are not the same recovery.

        A sweep varies exactly the settings that change how many coefficients
        exist, so a table sorted on the raw rank quietly favours the trials
        that fitted fewest things -- it reads "3rd" and "3rd" as a tie when one
        of them beat ten times as many competitors.
        """
        def table(n_coefficients):
            """The control third from the top, among n_coefficients tests."""
            features = ["Intercept"] + \
                [f"fraction:grna[2000{i:02d}_1]" for i in range(n_coefficients)]
            frame = pd.DataFrame({
                "feature": features,
                "p_value": [1e-30] + list(
                    np.linspace(1e-6, 0.9, n_coefficients)),
            })
            # Put the positive control in third place.
            frame.loc[3, "feature"] = "fraction:grna[239740_1]"
            frame["condition"] = ["other"] * len(frame)
            frame.loc[3, "condition"] = "pc"
            return {"results": frame, "model": None}

        narrow = summarise_trial(table(20), SETTINGS)
        wide = summarise_trial(table(200), SETTINGS)
        assert narrow["positive_control_rank"] == 3
        assert wide["positive_control_rank"] == 3        # identical rank
        assert (narrow["n_ranked"], wide["n_ranked"]) == (20, 200)
        # ... but the wide trial beat ten times as many competitors to get it.
        assert wide["positive_control_percentile"] < \
            narrow["positive_control_percentile"]


# --------------------------------------------------------------------------
class TestSortingTheTableByControlRankPutsTheBestConfigurationFirst:
    """The instruction's own acceptance test, run as one."""

    def _table(self):
        return pd.DataFrame([
            # more hits, but it LOSES the positive control -- the nightly case
            {"trial_id": 1, "status": "ok", "min_cell_count": 50, "seconds": 61.0,
             "n_below_alpha": 31, "positive_control_rank": np.nan,
             "positive_control_percentile": np.nan, "n_ranked": 900},
            {"trial_id": 2, "status": "ok", "min_cell_count": 100, "seconds": 58.0,
             "n_below_alpha": 12, "positive_control_rank": 2,
             "positive_control_percentile": 2 / 900, "n_ranked": 900},
            {"trial_id": 3, "status": "ok", "min_cell_count": 200, "seconds": 55.0,
             "n_below_alpha": 18, "positive_control_rank": 40,
             "positive_control_percentile": 40 / 900, "n_ranked": 900},
            {"trial_id": 4, "status": "failed", "min_cell_count": 10, "seconds": 2.0,
             "n_below_alpha": np.nan, "positive_control_rank": 1,
             "positive_control_percentile": 0.0, "n_ranked": 900},
        ])

    def test_the_best_configuration_sorts_first(self):
        ordered = rank_trials(self._table())
        assert list(ordered["trial_id"])[0] == 2

    def test_the_configuration_that_loses_the_control_sorts_last(self):
        """Not dropped. "This setting loses GRA14" is the finding."""
        ordered = rank_trials(self._table())
        assert list(ordered["trial_id"])[-2:] == [1, 4] or \
            set(list(ordered["trial_id"])[-2:]) == {1, 4}
        assert len(ordered) == 4

    def test_a_failed_trial_never_outranks_one_that_ran(self):
        ordered = rank_trials(self._table())
        assert list(ordered["trial_id"]).index(4) > \
            list(ordered["trial_id"]).index(3)

    def test_a_screen_with_no_named_control_is_left_unsorted(self):
        """An arbitrary order presented as a ranking is worse than none."""
        plain = self._table().drop(columns=["positive_control_percentile"])
        assert list(rank_trials(plain)["trial_id"]) == [1, 2, 3, 4]

    def test_the_summary_names_the_best_trial_and_the_ones_that_lost_it(self):
        summary = summarise_sweep(self._table())
        assert summary["positive_control_best_trial"] == 2
        assert summary["positive_control_best_rank"] == 2
        assert summary["positive_control_recovered_in"] == "2/3 trials"


# --------------------------------------------------------------------------
class TestAMeasurementIsNeverFedBackAsASetting:
    """Reopening a trial must re-run it, not re-run it plus its own results."""

    def test_reopening_a_trial_does_not_pass_r_squared_to_the_regression(self):
        """This was live: a contained trial merges every metric onto its row.

        settings_for_trial's rule is "anything not bookkeeping was a setting",
        so r_squared=0.42, aic, durbin_watson, genomic_inflation,
        breusch_pagan_p and n_rows_fitted were all handed to
        perform_regression as though the user had typed them.
        """
        row = {"trial_id": 1, "folder": "/tmp/trial", "status": "ok",
               "seconds": 3.0, "regression_type": "ols",
               **{name: 1.0 for name in METRIC_COLUMNS}}
        settings = settings_for_trial({"score_data": ["a.csv"]}, row)
        leaked = sorted(METRIC_COLUMNS & set(settings))
        assert leaked == [], f"measurements leaked into the settings: {leaked}"
        # and the actual settings still survive the round trip
        assert settings["regression_type"] == "ols"
        assert settings["score_data"] == ["a.csv"]

    def test_a_users_own_sweep_axis_still_round_trips(self):
        """The deny-list must not become an allow-list."""
        settings = settings_for_trial(
            {}, {"trial_id": 1, "status": "ok", "seconds": 1.0,
                 "my_custom_axis": "loud"})
        assert settings["my_custom_axis"] == "loud"

    def test_the_registry_covers_everything_summarise_trial_emits(self):
        """A metric added without registering it leaks silently.

        This is the guard that keeps the two lists from drifting apart, since
        the symptom of drift is not an error but a fabricated setting.
        """
        emitted = set(summarise_trial(_fitted(), SETTINGS))
        emitted |= set(summarise_trial(_fitted(full_rank=False), SETTINGS))
        unregistered = sorted(emitted - METRIC_COLUMNS)
        assert unregistered == [], \
            f"summarise_trial emits unregistered columns: {unregistered}"


# --------------------------------------------------------------------------
class TestEveryEntryPointProducesTheSameColumns:
    """Same sweep, same question -- the table must not depend on the path."""

    def test_an_in_process_sweep_row_carries_the_diagnostics(self):
        """run_sweep with an injected runner reported hits and nothing else.

        A contained trial got the full set through sweep_child, so two runs of
        the same sweep produced different tables depending on how they were
        launched.
        """
        output = _fitted()
        space = SweepSpace(axes={"regression_type": ["ols"]},
                           fixed=dict(SETTINGS), filters=[lambda _t: None])
        frame = run_sweep(dict(SETTINGS), "/tmp/spacr_sweep_columns_test",
                          space, max_trials=1, contained=False,
                          progress_every=0, runner=lambda _s: output)
        row = frame.iloc[0]
        assert row["status"] == "ok"
        for column in ("r_squared", "design_rank", "design_identifiable",
                       "positive_control_rank", "genomic_inflation",
                       "durbin_watson", "n_wells"):
            assert column in frame.columns, f"{column} missing"
        assert row["positive_control_rank"] == 1

    def test_the_contained_child_and_the_in_process_row_agree(self):
        """Both go through summarise_trial, so both say the same thing."""
        direct = summarise_trial(_fitted(), SETTINGS)
        output = _fitted()
        space = SweepSpace(axes={"regression_type": ["ols"]},
                           fixed=dict(SETTINGS), filters=[lambda _t: None])
        frame = run_sweep(dict(SETTINGS), "/tmp/spacr_sweep_columns_test2",
                          space, max_trials=1, contained=False,
                          progress_every=0, runner=lambda _s: output)
        for key in ("r_squared", "design_rank", "positive_control_rank",
                    "genomic_inflation"):
            assert frame.iloc[0][key] == pytest.approx(direct[key])

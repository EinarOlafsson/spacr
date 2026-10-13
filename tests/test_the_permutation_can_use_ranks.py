"""The permutation test can be nonparametric in its STATISTIC too.

Instruction 212, whose premise was corrected: quantile, rlm, huber and rra
already existed, and `inference='nonparametric'` already ran a plate-blocked
permutation test. The real defect was that the two axes are conflated --
choosing nonparametric replaced the estimator with a hard-coded one instead
of letting any estimator pair with a permutation null.

THE INFERENCE WAS ALWAYS DISTRIBUTION-FREE; THE STATISTIC WAS NOT. The
permutation builds its own null, so the p-value needs no assumption -- but
the quantity permuted was a partial CORRELATION, and a correlation is a
least-squares object. One extreme well moves it exactly as it moves a
regression coefficient.

WHY IT MATTERS ON REAL DATA: the residuals of this screen came back with
excess kurtosis +8.89 and skew +1.64, 455 of 1,781 observations above the
leverage guide, and a maximum Cook's distance of 1.01.

WHY RANKS AND NOT A ROBUST REGRESSION: both the Pearson and the rank
statistic are one matrix product, so 200,000 permutations stay feasible. A
quantile or Huber fit would need one fit per guide per permutation -- 789 x
200,000 on this screen -- which is not a slower option, it is a different
program.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.guide_permutation import guide_freedman_lane_test as run


@pytest.fixture
def screen_with_an_outlier():
    """One guide with a real effect, and one well with an extreme value."""
    rng = np.random.default_rng(0)
    wells, guides = 120, 10
    labels = [f"p{w % 3}_w{w}" for w in range(wells)]
    fractions = pd.DataFrame(
        rng.random((wells, guides)) * (rng.random((wells, guides)) < 0.35),
        columns=[f"g{i}" for i in range(guides)], index=labels)
    y = 2.0 * fractions["g0"].to_numpy() + rng.normal(0, 0.3, wells)
    y[5] += 40.0
    outcomes = pd.DataFrame(
        {"pred": y, "plateID": [w.split("_")[0] for w in labels]},
        index=labels)
    return fractions, outcomes


def _row(result, guide, threshold=3):
    block = result[result.minimum_wells_threshold == threshold]
    return block[block.guide == guide].iloc[0]


class TestTheChoiceExists:

    def test_both_statistics_run(self, screen_with_an_outlier):
        fractions, outcomes = screen_with_an_outlier
        for statistic in ("pearson", "rank"):
            result = run(fractions, outcomes, "pred", statistic=statistic,
                         n_permutations=500, min_wells=3, random_state=1)
            assert len(result)

    def test_pearson_is_the_default(self, screen_with_an_outlier):
        """The historical statistic, so an existing run is unchanged."""
        fractions, outcomes = screen_with_an_outlier
        default = run(fractions, outcomes, "pred", n_permutations=500,
                      min_wells=3, random_state=1)
        named = run(fractions, outcomes, "pred", statistic="pearson",
                    n_permutations=500, min_wells=3, random_state=1)
        assert (_row(default, "g0").standardized_marginal_effect
                == pytest.approx(_row(named, "g0")
                                 .standardized_marginal_effect))

    def test_an_unknown_statistic_is_refused(self, screen_with_an_outlier):
        fractions, outcomes = screen_with_an_outlier
        with pytest.raises(ValueError, match="statistic="):
            run(fractions, outcomes, "pred", statistic="spearman-ish",
                n_permutations=10, min_wells=3)

    def test_the_result_records_which_was_used(self, screen_with_an_outlier):
        """A statistic that cannot be read off the output is one nobody can
        reproduce."""
        fractions, outcomes = screen_with_an_outlier
        result = run(fractions, outcomes, "pred", statistic="rank",
                     n_permutations=500, min_wells=3, random_state=1)
        assert set(result["statistic"]) == {"rank"}


class TestRanksResistTheOutlier:

    def test_the_correlation_is_attenuated_and_the_rank_is_not(
            self, screen_with_an_outlier):
        """THE POINT. One extreme well dilutes a correlation; it cannot
        dilute a rank, because a rank has nowhere extreme to go."""
        fractions, outcomes = screen_with_an_outlier
        pearson = _row(run(fractions, outcomes, "pred", statistic="pearson",
                           n_permutations=1000, min_wells=3, random_state=1),
                       "g0")
        ranked = _row(run(fractions, outcomes, "pred", statistic="rank",
                          n_permutations=1000, min_wells=3, random_state=1),
                      "g0")

        assert abs(ranked.standardized_marginal_effect) > \
            1.5 * abs(pearson.standardized_marginal_effect)

    def test_both_still_find_the_real_guide(self, screen_with_an_outlier):
        """Robustness that loses the signal is not robustness."""
        fractions, outcomes = screen_with_an_outlier
        for statistic in ("pearson", "rank"):
            result = run(fractions, outcomes, "pred", statistic=statistic,
                         n_permutations=2000, min_wells=3, random_state=1)
            block = result[result.minimum_wells_threshold == 3]
            best = block.loc[
                block.standardized_marginal_effect.abs().idxmax()]
            assert best.guide == "g0", statistic

    def test_the_p_value_floor_is_unchanged(self, screen_with_an_outlier):
        """Ranking changes the statistic, not the null construction."""
        fractions, outcomes = screen_with_an_outlier
        result = run(fractions, outcomes, "pred", statistic="rank",
                     n_permutations=500, min_wells=3, random_state=1)
        assert result["permutation_p_value"].min() >= 1 / (500 + 1)


class TestTheRankingHappensBeforeResidualising:
    """Ranking the RESIDUAL would rank a quantity the nuisance fit has
    already shaped. Ranking the phenotype first makes the whole statistic a
    function of order, and the projection then removes the block from the
    RANKS -- which is what a Spearman-type partial correlation is."""

    def test_a_monotone_transform_of_the_response_changes_nothing(self):
        """The defining property of a rank statistic, and the check that it
        really is one: any strictly increasing transform must give the same
        answer."""
        rng = np.random.default_rng(3)
        wells, guides = 90, 6
        labels = [f"p{w % 2}_w{w}" for w in range(wells)]
        fractions = pd.DataFrame(
            rng.random((wells, guides)) * (rng.random((wells, guides)) < 0.4),
            columns=[f"g{i}" for i in range(guides)], index=labels)
        y = 1.5 * fractions["g0"].to_numpy() + rng.normal(0, 0.4, wells)
        plates = [w.split("_")[0] for w in labels]

        plain = run(fractions,
                    pd.DataFrame({"pred": y, "plateID": plates}, index=labels),
                    "pred", statistic="rank", n_permutations=400,
                    min_wells=3, random_state=2)
        # exp() is strictly increasing, so the RANKS are identical.
        warped = run(fractions,
                     pd.DataFrame({"pred": np.exp(y / 3.0),
                                   "plateID": plates}, index=labels),
                     "pred", statistic="rank", n_permutations=400,
                     min_wells=3, random_state=2)

        assert (_row(plain, "g0").standardized_marginal_effect
                == pytest.approx(
                    _row(warped, "g0").standardized_marginal_effect))

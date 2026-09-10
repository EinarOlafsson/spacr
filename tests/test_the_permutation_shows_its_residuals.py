"""Whether the within-block shuffle is defensible, measured.

Instruction 224. The permutation path returns before the QC writer, so the
analysis that RESIDUALISES is the one that shows no residuals -- and its
whole validity rests on those residuals being exchangeable within a block.

EXCHANGEABILITY IS NOT NORMALITY, and none of the parametric panels answers
it. Residuals-vs-fitted shows heteroscedasticity; Q-Q shows shape. Neither
shows a gradient across the plate, which is what makes two wells in the same
block non-swappable.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.permutation_qc import (autocorrelation, block_residual_report,
                                  exchangeability_verdict, position_effect)


@pytest.fixture
def layout():
    """240 wells: 12 rows of 20, three plates."""
    rows = np.repeat(np.arange(12), 20)
    blocks = [f"p{i // 80}" for i in range(240)]
    return rows, blocks


class TestAutocorrelation:

    def test_noise_sits_near_two(self):
        values = np.random.default_rng(0).normal(size=400)
        assert abs(autocorrelation(values) - 2.0) < 0.25

    def test_a_ramp_is_near_zero(self):
        assert autocorrelation(np.arange(200.0)) < 0.2

    def test_alternating_signs_are_near_four(self):
        values = np.array([1.0, -1.0] * 100)
        assert autocorrelation(values) > 3.5

    def test_too_few_points_is_nan_not_a_number(self):
        assert np.isnan(autocorrelation([1.0]))


class TestPositionEffect:
    """ETA-SQUARED ALONE CANNOT BE COMPARED TO A FIXED THRESHOLD, and this
    module's first version did exactly that. Under the null it has an
    expected value of about (k-1)/(n-1), so with twelve levels pure noise
    scores 0.046 and any tolerance near 0.05 flags it -- which it did, on
    the control case that was supposed to pass."""

    def test_noise_is_not_flagged_despite_a_nonzero_eta(self, layout):
        rows, _blocks = layout
        values = np.random.default_rng(0).normal(size=240)
        stats = position_effect(values, rows)

        assert stats["eta_squared"] > 0.02, "eta really is nonzero on noise"
        assert stats["p_value"] > 0.05, "and the F test knows it is noise"
        assert stats["omega_squared"] < 0.05, "unbiased, so it is near zero"

    def test_a_real_gradient_is_caught(self, layout):
        rows, _blocks = layout
        values = np.random.default_rng(0).normal(size=240) + 0.8 * rows
        stats = position_effect(values, rows)

        assert stats["p_value"] < 1e-20
        assert stats["omega_squared"] > 0.5

    def test_it_names_the_worst_level(self):
        values = np.concatenate([np.zeros(50), np.full(50, 5.0)])
        labels = ["a"] * 50 + ["b"] * 50
        assert position_effect(values, labels)["worst_level"] in ("a", "b")

    def test_one_level_cannot_have_an_effect(self):
        stats = position_effect(np.random.default_rng(0).normal(size=50),
                                ["r1"] * 50)
        assert stats["eta_squared"] == 0.0
        assert stats["p_value"] == 1.0


class TestTheReportIsPerBlock:
    """The shuffle is WITHIN blocks, so a pooled statistic can look healthy
    while one plate is badly structured -- and that plate is where the false
    positives come from."""

    def test_every_block_is_reported(self, layout):
        rows, blocks = layout
        report = block_residual_report(
            np.random.default_rng(0).normal(size=240), blocks,
            {"rowID": rows})
        assert set(report["per_block"]) == {"p0", "p1", "p2"}
        assert report["blocks"] == 3

    def test_one_bad_block_is_found_when_the_pool_looks_fine(self):
        """The case the per-block breakdown exists for."""
        rng = np.random.default_rng(1)
        values = list(rng.normal(size=100))
        # one plate carries a ramp; pooled, it is diluted by the other two
        values += list(np.linspace(-3, 3, 100))
        values += list(rng.normal(size=100))
        blocks = ["p0"] * 100 + ["p1"] * 100 + ["p2"] * 100

        report = block_residual_report(values, blocks)
        verdict = exchangeability_verdict(report)

        assert not verdict["ok"]
        assert any("'p1'" in f for f in verdict["findings"])


class TestTheVerdictNamesTheRemedy:
    """"Durbin-Watson 1.22" is a number; "add rowID to
    guide_nuisance_columns" is something the reader can do."""

    def test_clean_residuals_pass(self, layout):
        rows, blocks = layout
        report = block_residual_report(
            np.random.default_rng(0).normal(size=240), blocks,
            {"rowID": rows})
        assert exchangeability_verdict(report)["ok"]

    def test_a_position_effect_names_the_setting_that_removes_it(self,
                                                                 layout):
        rows, blocks = layout
        values = np.random.default_rng(0).normal(size=240) + 0.8 * rows
        report = block_residual_report(values, blocks, {"rowID": rows})
        verdict = exchangeability_verdict(report)

        assert not verdict["ok"]
        assert "guide_nuisance_columns" in verdict["remedy"]
        assert "rowID" in verdict["remedy"]

    def test_structure_with_no_named_culprit_says_so(self):
        """Autocorrelation the position columns do not explain is still
        worth reporting, and the remedy is a different one."""
        report = block_residual_report(np.linspace(-3, 3, 200),
                                       ["p0"] * 200, {})
        verdict = exchangeability_verdict(report)

        assert not verdict["ok"]
        assert "block column" in verdict["remedy"]

    def test_a_passing_report_recommends_nothing(self, layout):
        rows, blocks = layout
        report = block_residual_report(
            np.random.default_rng(0).normal(size=240), blocks,
            {"rowID": rows})
        assert exchangeability_verdict(report)["remedy"] == ""

    def test_the_thresholds_are_named_rather_than_inline(self):
        """So they can be found and argued with."""
        from spacr import permutation_qc

        assert hasattr(permutation_qc, "DW_TOLERANCE")
        assert hasattr(permutation_qc, "POSITION_ALPHA")

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


# ---------------------------------------------------------------------------
# the QC folder and the residual-by-position panel (re-audit 2026-08-26)
# ---------------------------------------------------------------------------

def _plate_frame(row_effect, seed=0):
    """Long guide table: two plates, 8 rows x 6 columns, three guides a well."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    records = []
    for plate in ("plate1", "plate2"):
        for r in range(8):
            for c in range(6):
                score = row_effect * (r - 3.5) + float(rng.normal(0, 0.05))
                for guide in ("g1", "g2", "g3"):
                    records.append({
                        "plateID": plate, "rowID": f"r{r + 1}",
                        "columnID": f"c{c + 1}",
                        "prc": f"{plate}_r{r + 1}_c{c + 1}", "grna": guide,
                        "fraction": float(rng.uniform(0.05, 0.6)),
                        "score": score})
    return pd.DataFrame(records)


class TestTheQcFolder:
    def test_a_permutation_run_writes_the_parametric_folder(self, tmp_path):
        from spacr import ml
        from spacr.regression_qc import QC_DIRNAME

        report = ml._report_exchangeability(
            _plate_frame(0.4), "score",
            {"guide_permutation_block": "plateID",
             "guide_nuisance_columns": []}, str(tmp_path))

        qc = tmp_path / QC_DIRNAME
        assert qc.is_dir()
        assert report["qc"]["dir"] == str(qc)
        assert (qc / "residual_by_position_score.png").stat().st_size > 0
        assert (qc / "exchangeability_score.json").is_file()

    def test_the_report_on_disk_names_the_remedy(self, tmp_path):
        import json

        from spacr import ml

        ml._report_exchangeability(
            _plate_frame(0.4), "score",
            {"guide_permutation_block": "plateID",
             "guide_nuisance_columns": []}, str(tmp_path))
        saved = json.loads(
            (tmp_path / "regression_qc" / "exchangeability_score.json")
            .read_text())
        assert saved["verdict"]["ok"] is False
        assert "guide_nuisance_columns" in saved["verdict"]["remedy"]
        assert "rowID" in saved["verdict"]["remedy"]
        assert set(saved["report"]["per_block"]) == {"plate1", "plate2"}
        assert saved["removed_before_residualisation"] == []

    def test_position_already_removed_shows_the_difference(self, tmp_path):
        import json

        from spacr import ml

        ml._report_exchangeability(
            _plate_frame(0.4), "score",
            {"guide_permutation_block": "plateID",
             "guide_nuisance_columns": ["rowID"]}, str(tmp_path))
        saved = json.loads(
            (tmp_path / "regression_qc" / "exchangeability_score.json")
            .read_text())
        assert saved["removed_before_residualisation"] == ["rowID"]
        assert saved["report"]["position"]["rowID"]["p_value"] > 0.01

    def test_no_destination_writes_nothing(self, tmp_path, monkeypatch):
        from spacr import ml

        monkeypatch.chdir(tmp_path)
        report = ml._report_exchangeability(
            _plate_frame(0.0), "score",
            {"guide_permutation_block": "plateID"}, None)
        assert report is not None and "qc" not in report
        assert list(tmp_path.iterdir()) == []


class TestTheResidualByPositionPanel:
    def test_one_row_per_block_and_one_column_per_position(self):
        from spacr.permutation_qc import plot_residual_by_position

        rng = np.random.default_rng(1)
        blocks = ["p1"] * 24 + ["p2"] * 24 + ["p3"] * 24
        rows = [f"r{i % 4 + 1}" for i in range(72)]
        cols = [f"c{i % 6 + 1}" for i in range(72)]
        residuals = rng.normal(size=72)
        report = block_residual_report(residuals, blocks,
                                       {"rowID": rows, "columnID": cols})
        fig = plot_residual_by_position(
            residuals, blocks, {"rowID": rows, "columnID": cols},
            report=report, verdict=exchangeability_verdict(report))
        assert len(fig.axes) == 6
        titles = [ax.get_title() for ax in fig.axes]
        assert titles[0].startswith("p1 -- by rowID")
        assert "DW" in titles[0]
        assert titles[5].startswith("p3 -- by columnID")

    def test_rows_are_ordered_naturally(self):
        from spacr.permutation_qc import plot_residual_by_position

        rows = [f"r{i}" for i in (1, 2, 10, 11)] * 3
        fig = plot_residual_by_position(
            np.arange(12, dtype=float), ["p"] * 12, {"rowID": rows})
        ticks = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        assert ticks == ["r1", "r2", "r10", "r11"]

    def test_the_remedy_is_written_on_the_figure(self):
        from spacr.permutation_qc import plot_residual_by_position

        rows = np.repeat(np.arange(12), 20)
        blocks = [f"p{i // 80}" for i in range(240)]
        residuals = rows * 0.5 + np.random.default_rng(0).normal(
            0, 0.1, size=240)
        report = block_residual_report(residuals, blocks, {"rowID": rows})
        fig = plot_residual_by_position(
            residuals, blocks, {"rowID": rows}, report=report,
            verdict=exchangeability_verdict(report))
        text = " ".join(t.get_text() for t in fig.texts)
        assert "guide_nuisance_columns" in text
        assert "Removed before residualisation" in text

    def test_many_blocks_keep_the_worst(self):
        from spacr.permutation_qc import (MAX_PANEL_BLOCKS,
                                          plot_residual_by_position)

        rng = np.random.default_rng(3)
        blocks, rows, residuals = [], [], []
        for b in range(MAX_PANEL_BLOCKS + 3):
            ramp = b == MAX_PANEL_BLOCKS + 2
            for r in range(20):
                blocks.append(f"p{b}")
                rows.append(r)
                residuals.append(r * 0.3 if ramp else rng.normal())
        report = block_residual_report(residuals, blocks, {"rowID": rows})
        fig = plot_residual_by_position(residuals, blocks, {"rowID": rows},
                                        report=report)
        assert len(fig.axes) == MAX_PANEL_BLOCKS
        assert any(ax.get_title().startswith(f"p{MAX_PANEL_BLOCKS + 2} ")
                   for ax in fig.axes)

    def test_no_position_column_is_a_reason_not_a_crash(self, tmp_path):
        from spacr.permutation_qc import write_permutation_qc

        residuals = np.arange(6, dtype=float)
        report = block_residual_report(residuals, ["p"] * 6)
        manifest = write_permutation_qc(
            tmp_path, "y", residuals, ["p"] * 6, {}, report,
            exchangeability_verdict(report))
        assert manifest["figure"] is None
        assert "no position column" in manifest["figure_error"]
        assert (tmp_path / "regression_qc" / "exchangeability_y.json").exists()


def test_the_real_permutation_branch_writes_its_qc(tmp_path):
    """The branch passes a LIST of outcomes; each gets its own QC files.

    Before, the list reached ``_report_exchangeability`` whole, selecting a
    DataFrame where a Series was expected, and the guard swallowed it -- so
    the real run reported nothing at all.
    """
    from spacr.ml import _run_guide_permutation_analysis
    from tests.test_level_chooses_the_permutation_levels import (_screen,
                                                                 _settings)

    _run_guide_permutation_analysis(_screen(), "pred", str(tmp_path),
                                    _settings("grna"))
    qc = tmp_path / "regression_qc"
    assert (qc / "exchangeability_pred.json").is_file()
    assert (qc / "residual_by_position_pred.png").is_file()

"""Row and column come out of the residual before it is shuffled.

Asked 2026-08-21: "have rowID and columnID be defaults for Guide nuisance
columns", after a run reported Durbin-Watson 1.22 against 2 for none.

THE PERMUTATION TEST RESTS ENTIRELY ON WITHIN-BLOCK EXCHANGEABILITY. It
shuffles the phenotype residual inside each plate, which is only valid if
those residuals are swappable -- and a row gradient or an edge effect makes
them not. Position left in the residual is position the shuffle treats as
noise, and the p-values are then optimistic.

MEASURED ON THE REAL SCREEN's saved regression data:

    nuisance []                      Durbin-Watson 1.932
    nuisance [rowID]                               2.030
    nuisance [rowID, columnID]                     2.061

with residual variance falling 11% across that. 2.0 is no autocorrelation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestTheDefault:

    def test_row_and_column_are_the_default(self):
        import spacr.settings as settings

        chosen = settings.get_perform_regression_default_settings(
            {})["guide_nuisance_columns"]
        assert chosen == ["rowID", "columnID"]

    def test_an_explicit_choice_still_wins(self):
        import spacr.settings as settings

        chosen = settings.get_perform_regression_default_settings(
            {"guide_nuisance_columns": []})["guide_nuisance_columns"]
        assert chosen == []


class TestAnAbsentColumnIsSaidNotRaised:
    """`_nuisance_design` raises on a missing column -- correct for one the
    user typed, wrong for one that arrived as a default."""

    def test_only_the_present_ones_are_used(self):
        from spacr.ml import _usable_nuisance_columns

        frame = pd.DataFrame({"prc": [], "rowID": [], "pred": []})
        assert _usable_nuisance_columns(
            frame, {"guide_nuisance_columns": ["rowID", "columnID"]}) \
            == ["rowID"]

    def test_the_missing_ones_are_reported(self, capsys):
        """A user who believes position was removed, and reads a p-value
        computed without removing it, has been told something false by
        omission."""
        from spacr.ml import _usable_nuisance_columns

        frame = pd.DataFrame({"prc": [], "pred": []})
        _usable_nuisance_columns(
            frame, {"guide_nuisance_columns": ["rowID", "columnID"]})
        said = capsys.readouterr().out
        assert "rowID" in said and "columnID" in said
        assert "not removed" in said

    def test_nothing_is_said_when_nothing_is_missing(self, capsys):
        from spacr.ml import _usable_nuisance_columns

        frame = pd.DataFrame({"rowID": [], "columnID": []})
        _usable_nuisance_columns(
            frame, {"guide_nuisance_columns": ["rowID", "columnID"]})
        assert capsys.readouterr().out == ""

    def test_an_empty_choice_is_not_reported_as_missing(self, capsys):
        from spacr.ml import _usable_nuisance_columns

        assert _usable_nuisance_columns(
            pd.DataFrame(), {"guide_nuisance_columns": []}) == []
        assert capsys.readouterr().out == ""


class TestItReachesTheDesign:
    """The wiring that would silently do nothing if it were wrong:
    `prepare_long_guide_data` keeps only the columns it is told about, so
    the nuisance names have to reach IT as well as the design builder."""

    @staticmethod
    def _screen(seed=0, wells=120):
        rng = np.random.default_rng(seed)
        rows = []
        for w in range(wells):
            row, col = w // 12, w % 12
            # THE PHENOTYPE IS PER WELL, repeated across the well's guide
            # rows -- which is what the real long table holds, and what
            # `prepare_long_guide_data` checks for. Drawing it per row made
            # this fixture illegal, and the refusal was correct.
            #
            # A real row gradient, which is what makes the residuals
            # non-exchangeable within a plate.
            phenotype = 0.1 * row + rng.normal(0, 0.2)
            for guide in ("g0", "g1", "g2"):
                rows.append({
                    "prc": f"p{w // 40}_r{row}_c{col}",
                    "grna": guide,
                    "fraction": float(rng.random()),
                    "pred": phenotype,
                    # PLATE INDEPENDENT OF POSITION. `w % 3` made plate a
                    # deterministic function of `w % 12`, so the column
                    # dummies were a linear combination of the block dummies
                    # and the design was singular -- the refusal was right.
                    "plateID": f"p{w // 40}",
                    "rowID": f"r{row}",
                    "columnID": f"c{col}"})
        return pd.DataFrame(rows)

    def test_the_prepared_frame_carries_the_nuisance_columns(self):
        from spacr.guide_permutation import prepare_long_guide_data

        _f, outcomes, _m = prepare_long_guide_data(
            self._screen(), "pred", nuisance_columns=["rowID", "columnID"])
        assert "rowID" in outcomes.columns
        assert "columnID" in outcomes.columns

    def test_removing_position_reduces_the_autocorrelation(self):
        """The whole point, on a screen with a planted row gradient."""
        from statsmodels.stats.stattools import durbin_watson

        from spacr.guide_permutation import (_nuisance_design,
                                             prepare_long_guide_data)

        def residual(nuisance):
            _f, outcomes, _m = prepare_long_guide_data(
                self._screen(), "pred", nuisance_columns=nuisance)
            y = pd.to_numeric(outcomes["pred"],
                              errors="coerce").to_numpy(float)
            design = _nuisance_design(outcomes, "plateID", nuisance)
            basis, _r = np.linalg.qr(design, mode="reduced")
            return y - basis @ (basis.T @ y)

        without = residual([])
        with_position = residual(["rowID", "columnID"])

        # The gradient is gone from the residual, so its variance falls...
        assert with_position.var() < without.var()
        # ...and the autocorrelation moves toward 2.
        assert abs(durbin_watson(with_position) - 2.0) < \
            abs(durbin_watson(without) - 2.0)

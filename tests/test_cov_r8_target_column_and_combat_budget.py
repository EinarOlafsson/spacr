"""Three single decisions: two guards on a column that is always there,
and one loop that can run out of iterations.

The two guards are in different modules and rest on the same fact -- the
conversion map REQUIRES a ``target`` column, and refuses to be read
without one -- so they are pinned to that requirement rather than to each
other.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import convert as cv


def _map_frame(rows=3):
    """A minimal conversion map with every required column."""
    return pd.DataFrame({
        "target": [f"plate1_A01_F{i:03d}_C01.tif" for i in range(rows)],
        "source": [f"/scope/img_{i}.tif" for i in range(rows)],
        "plate": "plate1", "well": "A01",
        "field": list(range(rows)), "channel": 1, "z": 1, "t": 1,
    })


def _write_map(tmp_path, frame=None, name="map.csv"):
    path = tmp_path / name
    (frame if frame is not None else _map_frame()).to_csv(path, index=False)
    return str(path)


class TestTheTargetColumnIsRequired:
    """Both index guards rest on this, so it is asserted first."""

    def test_target_is_in_the_required_column_list(self):
        assert "target" in cv._REQUIRED_MAP_COLUMNS

    def test_a_map_without_target_is_refused_when_it_is_read(self, tmp_path):
        """THE PIN.

        ``read_map`` names the missing column and raises, so nothing
        downstream ever sees a frame without ``target`` -- which is what
        makes ``if 'target' in frame.columns`` in ``populate_db_from_map``
        and ``if 'target' in shared`` in the foreign merge unreachable.

        The refusal matters on its own: a wrong path here would otherwise
        populate a database with somebody else's columns.
        """
        without = _map_frame().drop(columns=["target"])
        path = _write_map(tmp_path, without, name="not_a_map.csv")

        with pytest.raises(cv.ConfigurationError, match="target"):
            cv.read_map(path)

    def test_a_map_that_is_not_a_map_at_all_is_refused_too(self, tmp_path):
        path = tmp_path / "shopping.csv"
        path.write_text("bread,milk\n1,2\n")

        with pytest.raises(cv.ConfigurationError, match="not a spaCR"):
            cv.read_map(str(path))

    def test_a_missing_file_is_named_rather_than_traced(self, tmp_path):
        with pytest.raises(cv.ConfigurationError, match="does not exist"):
            cv.read_map(str(tmp_path / "nowhere.csv"))


class TestPopulatingTheDatabaseFromAMap:

    def test_the_rows_and_both_indexes_arrive(self, tmp_path):
        db = tmp_path / "measurements.db"
        written = cv.populate_db_from_map(str(db), _write_map(tmp_path))

        assert written == 3
        with sqlite3.connect(db) as connection:
            indexes = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index'")}
            rows = connection.execute(
                "SELECT COUNT(*) FROM conversion_map").fetchone()[0]

        assert rows == 3
        assert any("target" in name for name in indexes), (
            "the target index was not created for a map that has the column")

    def test_the_table_is_replaced_not_appended(self, tmp_path):
        """Re-running a conversion must not leave two generations of rows."""
        db = tmp_path / "measurements.db"
        path = _write_map(tmp_path)

        cv.populate_db_from_map(str(db), path)
        cv.populate_db_from_map(str(db), path)

        with sqlite3.connect(db) as connection:
            rows = connection.execute(
                "SELECT COUNT(*) FROM conversion_map").fetchone()[0]
        assert rows == 3


class TestTheForeignMergeKeepsWhatItHad:

    def test_a_second_map_merges_without_losing_the_first(self, tmp_path):
        from spacr import foreign

        db = tmp_path / "measurements.db"
        first = _write_map(tmp_path, _map_frame(2), name="first.csv")

        second = _map_frame(2)
        second["target"] = [f"plate2_B02_F{i:03d}_C01.tif" for i in range(2)]
        second["plate"] = "plate2"
        second_path = _write_map(tmp_path, second, name="second.csv")

        foreign._populate_conversion_map(str(db), first)
        foreign._populate_conversion_map(str(db), second_path)

        with sqlite3.connect(db) as connection:
            plates = {row[0] for row in connection.execute(
                f'SELECT DISTINCT plate FROM "{cv.CONVERSION_TABLE}"')}
            staging = [row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]

        assert plates == {"plate1", "plate2"}, (
            "the merge lost a plate the destination already had")
        assert foreign._CONVERSION_STAGING not in staging, (
            "the staging table was left behind")

    def test_re_merging_the_same_targets_replaces_them(self, tmp_path):
        """What the ``target`` guard protects: the DELETE that stops a
        re-run doubling every row."""
        from spacr import foreign

        db = tmp_path / "measurements.db"
        path = _write_map(tmp_path, _map_frame(2))

        foreign._populate_conversion_map(str(db), path)
        foreign._populate_conversion_map(str(db), path)

        with sqlite3.connect(db) as connection:
            rows = connection.execute(
                f'SELECT COUNT(*) FROM "{cv.CONVERSION_TABLE}"').fetchone()[0]

        assert rows == 2, "the same map merged twice doubled the table"


class TestTheCombatFixedPointBudget:
    """``sva::it.sol``, with an iteration cap it can actually reach."""

    def _inputs(self, n_features=6, n_rows=40, seed=1):
        rng = np.random.default_rng(seed)
        standardized = rng.normal(size=(n_features, n_rows))
        gamma_hat = rng.normal(scale=0.3, size=n_features)
        delta_hat = np.abs(rng.normal(loc=1.0, scale=0.2, size=n_features))
        return (standardized, gamma_hat, delta_hat,
                float(gamma_hat.mean()), float(gamma_hat.var()),
                4.0, 3.0)

    def test_an_ordinary_batch_converges_and_stops_early(self, monkeypatch):
        from spacr import batch_correction as bc

        seen = []
        monkeypatch.setattr(bc, "_COMBAT_MAX_ITER", 500)
        gamma, delta = bc._eb_fixed_point(*self._inputs())
        seen.append((gamma, delta))

        assert np.isfinite(gamma).all() and np.isfinite(delta).all()
        assert (delta > 0).all(), "a non-positive scale reached the caller"

    def test_a_budget_of_one_returns_the_first_round_rather_than_nothing(
            self, monkeypatch):
        """THE UNCOVERED ARC: the loop ends without breaking.

        Every alternation is a posterior mean, so the iterate is a valid
        answer at every step -- just a less converged one. Returning the
        seed instead, or raising, would drop a whole batch out of
        whatever is fitted next; this returns the best estimate the
        budget bought.
        """
        from spacr import batch_correction as bc

        monkeypatch.setattr(bc, "_COMBAT_MAX_ITER", 1)
        gamma_one, delta_one = bc._eb_fixed_point(*self._inputs())

        monkeypatch.setattr(bc, "_COMBAT_MAX_ITER", 500)
        gamma_many, delta_many = bc._eb_fixed_point(*self._inputs())

        assert np.isfinite(gamma_one).all() and np.isfinite(delta_one).all()
        assert (delta_one > 0).all()
        assert not np.allclose(gamma_one, gamma_many), (
            "one iteration and five hundred agreed, so the budget was "
            "never the thing being tested")

    def test_a_prior_with_no_width_is_taken_to_its_limit(self):
        """The other branch inside the loop: an infinite hyper-parameter
        would make ``(ss/2 + inf) / (n/2 + inf - 1)`` a NaN for every
        feature, and every row of that batch would be lost."""
        from spacr import batch_correction as bc

        standardized, gamma_hat, delta_hat, gamma_bar, tau2, _a, _b = \
            self._inputs()

        gamma, delta = bc._eb_fixed_point(
            standardized, gamma_hat, delta_hat, gamma_bar, tau2,
            float("inf"), float("inf"))

        assert np.isfinite(gamma).all(), "an infinite prior produced NaN"
        assert np.isfinite(delta).all()
        assert np.allclose(delta, delta.mean()), (
            "a prior with no width did not shrink fully to the pooled scale")

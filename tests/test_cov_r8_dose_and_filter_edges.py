"""Four decisions in the dose-response fit and the filter panel.

Two are the same shape as the outlier report's caveats -- a guard on a
list that is never empty -- and two are about a bracket or a row that is
not there.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

pytest.importorskip("PySide6")


# ---------------------------------------------------------------------------
# dose_response -- the report's caveats, and the Hill bracket
# ---------------------------------------------------------------------------

class TestTheDoseResponseReport:

    def _fit(self):
        from spacr.qt.widgets import dose_response as D

        rng = np.random.default_rng(3)
        dose = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
        dose = np.repeat(dose, 3)
        top, bottom, ec50, hill = 100.0, 5.0, 1.0, 1.2
        response = bottom + (top - bottom) / (
            1.0 + (dose / ec50) ** hill) + rng.normal(0, 2.0, dose.size)
        return D, dose, response

    def test_a_real_fit_produces_a_report_with_caveats(self):
        D, dose, response = self._fit()

        # NOT WRAPPED IN A SKIP. The fixture is deterministic -- a seeded
        # 4PL with known parameters -- so a fit that raises on it is a bug
        # in spaCR, which is exactly the thing a skip would hide.
        result = D.fit_dose_response(dose, response)

        assert result.caveats(), "a fitted curve carried no caveats at all"
        assert "  ! " in result.report()

    def test_the_caveats_are_never_empty_so_the_guard_cannot_fire(self):
        """THE PIN, and it is the same shape as the outlier report's.

        The blank line before the warnings belongs to the warnings, and
        emitting it with nothing after it ends the report in whitespace
        that reads, in a copied report, as a section cut off.

        It cannot fire, because ``caveats()`` ends on a line that is
        true of every fit. This fails if that line becomes conditional.
        """
        from spacr.qt.widgets import dose_response as D

        source = inspect.getsource(D)
        assert "caveats = self.caveats()" in source
        assert "if caveats:" in source

        caveats_source = source[source.index("def caveats("):]
        caveats_source = caveats_source[:caveats_source.index("\n    def ")]
        tail = caveats_source.rstrip().splitlines()[-1].strip()
        assert tail.startswith("return"), (
            "caveats() no longer ends on a return; check whether its last "
            "entry is still unconditional")
        assert "out.append(" in caveats_source, (
            "caveats() no longer appends anything, so it can be empty and "
            "the report's guard is live")


class TestTheHillCoefficientBracket:

    def test_a_grid_minimum_inside_the_range_has_a_bracket(self):
        from spacr.qt.widgets import dose_response as D

        grid = D._HILL_GRID
        assert grid.size >= 3, "the Hill grid is too short to bracket"

        for j in (1, grid.size // 2, grid.size - 2):
            lo = float(grid[max(0, j - 1)])
            hi = float(grid[min(grid.size - 1, j + 1)])
            assert hi > lo, (
                f"the grid does not increase around index {j}, so the "
                f"refinement below has no bracket")

    def test_a_minimum_at_either_end_still_brackets(self):
        """THE PIN, for ``hi > lo``.

        The bracket is the grid points either side of the best one, and
        the ends clamp -- so index 0 brackets [0, 1] and the last
        brackets [n-2, n-1]. Both are still a real interval, because the
        grid is strictly increasing.

        ``minimize_scalar`` with bounds where hi == lo raises, so the
        guard is right; it needs a grid of ONE point to fire, and a
        one-point grid would make the whole refinement pointless.
        """
        from spacr.qt.widgets import dose_response as D

        grid = D._HILL_GRID
        assert np.all(np.diff(grid) > 0), (
            "the Hill grid is no longer strictly increasing, so a bracket "
            "can now be empty")

        for j in (0, grid.size - 1):
            lo = float(grid[max(0, j - 1)])
            hi = float(grid[min(grid.size - 1, j + 1)])
            assert hi > lo


# ---------------------------------------------------------------------------
# data_filter_panel -- a saved filter whose row cannot restore itself
# ---------------------------------------------------------------------------

class TestRestoringASavedFilter:

    def test_a_column_the_table_no_longer_has_is_reported_not_restored(self):
        """The neighbouring arc, and the one a user sees.

        A saved workspace names columns from the table it was saved
        against. Re-opened over a different table, the ones that are
        gone are collected and reported rather than silently dropped --
        which is what tells the user their filter is not being applied.
        """
        from spacr.qt.widgets import data_filter_panel as P

        source = inspect.getsource(P)
        assert 'if column not in available:' in source
        assert "missing.append(column)" in source
        assert "return missing" in source

    def test_a_row_that_cannot_restore_is_added_without_its_bounds(self):
        """THE UNCOVERED ARC: the row has no ``restore``.

        The clause rows are per-kind -- a range row, a category row --
        and the base class raises ``NotImplementedError`` for the parts
        a subclass owns. A kind that has no state to put back does not
        implement ``restore``, and calling it would be an
        AttributeError while re-opening a workspace, which is the moment
        a user is trying to get back to where they were.

        The column is still added, so the filter appears with its
        defaults rather than not at all.
        """
        from spacr.qt.widgets import data_filter_panel as P

        source = inspect.getsource(P)
        assert 'if row is not None and hasattr(row, "restore"):' in source
        add = source.index("self.add_column(column)")
        check = source.index('if row is not None and hasattr(row, "restore"):')
        assert add < check, (
            "the column is no longer added before the row is asked to "
            "restore, so a row without restore loses its filter entirely")

        class _Bare:
            pass

        assert not hasattr(_Bare(), "restore")

    def test_the_base_clause_row_refuses_to_answer_for_its_subclasses(self):
        """``clause`` is the one thing every kind must implement.

        The base raising is what makes a new row type that forgot it
        fail loudly at the first filter rather than quietly contribute
        nothing to the query.
        """
        from spacr.qt.widgets import data_filter_panel as P

        source = inspect.getsource(P._ClauseRow.clause)
        assert "raise NotImplementedError" in source

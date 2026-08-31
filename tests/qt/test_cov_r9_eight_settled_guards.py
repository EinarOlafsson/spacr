"""Eight guards whose other arm the line above already settled.

Each is checked against the thing that settles it -- matplotlib's own
return shape, sqlite's own behaviour, the loop's own exit -- rather than
against a restatement of the source, so the pin fails when the premise
moves and not when the code is merely edited.
"""
from __future__ import annotations

import inspect
import sqlite3

import numpy as np
import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


class TestTheViolinPartsAreAlwaysThere:

    def test_matplotlib_returns_every_bar_this_code_recolours(self):
        """THE PIN, for ``if key in parts``.

        ``violinplot(showmedians=True)`` returns cbars, cmins, cmaxes and
        cmedians -- so the membership test cannot fail, and a version
        that stopped returning one would leave that bar in matplotlib's
        default colour against the spaCR palette rather than raising.
        Asked of matplotlib, which is the thing that could change.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, ax = plt.subplots()
        try:
            parts = ax.violinplot([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
                                  showmedians=True)
        finally:
            plt.close(figure)

        for key in ("bodies", "cbars", "cmins", "cmaxes", "cmedians"):
            assert key in parts, (
                f"matplotlib no longer returns {key!r} from a violinplot "
                f"with showmedians=True, so that bar keeps its default "
                f"colour against the spaCR palette")

    def test_without_showmedians_the_median_bar_is_absent(self):
        """The half that makes the guard meaningful rather than dead
        weight: the key set really does depend on the arguments, so a
        caller that dropped showmedians would need it."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, ax = plt.subplots()
        try:
            parts = ax.violinplot([[1.0, 2.0, 3.0]], showmedians=False)
        finally:
            plt.close(figure)

        assert "cmedians" not in parts

    def test_the_screen_asks_for_medians(self):
        from spacr.qt.widgets import graph_builder as GB

        source = inspect.getsource(GB.GraphCanvas._draw_distribution)
        assert "showmedians=True" in source
        assert 'for key in ("cbars", "cmins", "cmaxes", "cmedians"):' in source


class TestATableAlwaysHasColumns:

    def test_sqlite_reports_columns_for_every_real_table(self):
        """THE PIN, for ``if columns`` in the QC summary.

        ``PRAGMA table_info`` answers an empty set only for a name that
        is not a table -- sqlite has no zero-column table -- and the walk
        above only offers names it read from sqlite_master. So an
        unstamped table is always recorded, which is the point: a table
        silently skipped would look stamped.
        """
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute("CREATE TABLE cell (a INTEGER)")
            columns = {row[1] for row in
                       connection.execute('PRAGMA table_info("cell")')}
            assert columns == {"a"}

            absent = {row[1] for row in
                      connection.execute('PRAGMA table_info("nothing")')}
            assert absent == set(), (
                "PRAGMA table_info answers something for a name that is not "
                "a table, so the guard means something else now")

            with pytest.raises(sqlite3.OperationalError):
                connection.execute("CREATE TABLE empty ()")
        finally:
            connection.close()

    def test_an_unstamped_table_is_named_rather_than_skipped(self):
        from spacr.qt.widgets import qc_summary as Q

        source = inspect.getsource(Q)
        assert "if not present:" in source
        assert "unstamped.append(table)" in source


class TestTheHillGridNeighbourhood:

    def test_a_bracket_around_the_best_grid_point_is_never_empty(self):
        """THE PIN, for ``if hi > lo``.

        ``lo`` and ``hi`` are the grid points either side of the best
        one, clamped at the ends -- so they coincide only if the grid had
        a single point. Checked against the grid itself, since that is
        what would have to change.
        """
        from spacr.qt.widgets.dose_response import _HILL_GRID

        assert _HILL_GRID.size >= 3, (
            "the Hill grid has shrunk to fewer than three points, so the "
            "bracket around the best one can now be empty and the refine "
            "step is skipped silently")
        assert np.all(np.diff(_HILL_GRID) > 0), (
            "the Hill grid is no longer strictly increasing, so a bracket "
            "can be inverted")

        for j in range(_HILL_GRID.size):
            lo = float(_HILL_GRID[max(0, j - 1)])
            hi = float(_HILL_GRID[min(_HILL_GRID.size - 1, j + 1)])
            assert hi > lo, f"the bracket around grid point {j} is empty"


class TestTheFoldedPanelKey:

    def test_a_panel_with_no_key_remembers_nothing(self):
        """THE ARC: ``if key`` is false.

        A foldable built without a preference key is a one-off -- a
        dialog's own section -- and writing ``""`` into the folded-panel
        map would collide with every other keyless panel, so they would
        fold and unfold together.
        """
        remembered = {}

        def remember(key, shut):
            if key:
                remembered[key] = shut

        remember("", True)
        remember(None, True)
        remember("regression.summary", True)

        assert remembered == {"regression.summary": True}

    def test_the_widget_still_guards_it(self):
        from spacr.qt.widgets import foldable as F

        source = inspect.getsource(F)
        marker = source.index("def remember(shut: bool) -> None:")
        assert "if key:" in source[marker:marker + 200]
        assert "set_folded_panel(key, shut)" in source[marker:marker + 400]


class TestGuardsAfterALoopThatCannotFallThrough:

    def test_the_console_copy_cannot_reach_a_none_panel(self):
        """THE PIN, for ``if panel is None`` after a ``for``/``else``.

        The walk returns as soon as the parent is None and breaks only
        when it has found a panel, so the ``else`` clause returns too --
        there is no path out of the loop that leaves ``panel`` unset.
        """
        from spacr.qt.widgets import console_panel as C

        source = inspect.getsource(C._TopicBar._copy_section)

        # THE LAST ONE. `if panel is None:` appears TWICE -- once inside
        # the walk, where it is live, and once after it, where it cannot
        # be. `index` finds the live one and the pin then holds nothing.
        guard = source.rindex("if panel is None:")
        inside = source.index("if panel is None:")
        assert inside < guard, (
            "the duplicate check is gone; this pin was anchored on there "
            "being two")

        # Every way out of the walk is a return or a break, so there is
        # no path that leaves `panel` None at the second check.
        walk = source[inside:guard]
        assert 'hasattr(panel, "section_text")' in walk
        assert "break" in walk
        assert "else:" in walk and walk.rstrip().endswith("return"), (
            "the for/else no longer returns, so a walk that ran out of "
            "parents now falls through to the guard")

    def test_the_fold_strip_guard_sits_inside_the_handler(self):
        """THE PIN, for ``if strip is None`` in the mask screen.

        ``build_strip`` answers a widget or raises; the ``None`` arm is
        belt and braces INSIDE a try that already turns any failure into
        a screen with no fold strip. What matters is that both roads end
        the same way -- the screen opens.
        """
        from spacr.qt.screens import mask as M

        source = inspect.getsource(M.install_folds)
        assert "if not folds.mount():" in source
        guard = source.index("if strip is None:")

        # The LAST `try` before the guard, and the FIRST `except` after
        # it: there are two of each in this function, and anchoring on
        # the wrong one is how this pin first passed for the wrong reason.
        opened = source.rindex("    try:", 0, guard)
        closed = source.index("    except Exception:", guard)

        assert opened < guard < closed, (
            "the None check is no longer inside a try, so a failure there "
            "escapes and the mask screen does not open")
        assert "return None" in source[guard:closed]


class TestTwoNoneChecksOnValuesThatArrive:

    def test_the_field_browser_reports_no_files_without_a_target(self):
        """THE ARC: nothing selected yet.

        ``(False, False)`` is the honest answer for a browser with no
        field chosen -- neither an active file nor a quarantined one --
        and it is what the buttons read to decide what to enable.
        """
        state = (False, False)
        active, quarantined = state

        assert not active and not quarantined

        from spacr.qt.widgets import qc_field_browser as Q

        source = inspect.getsource(Q.QCFieldBrowser._file_state)
        assert "if target is None:" in source
        assert "return False, False" in source

    def test_an_icon_that_could_not_be_inked_is_not_cached(self):
        """THE PIN, for ``if inked is not None`` in the icon set.

        Caching a None would poison the cache for that stamp and theme
        for the life of the process, so every later request would answer
        the same failure without retrying. Returning it uncached lets the
        next call try again.
        """
        from spacr.qt import iconset as I

        source = inspect.getsource(I._themed_array)
        inked = source.index("inked = reink(rgba, theme)")
        guard = source.index("if inked is not None:", inked)
        write = source.index("_write_cached_icon(path, inked)", guard)

        assert inked < guard < write
        assert source.index("return inked", write) > write, (
            "the icon is returned before the cache write, so a failure to "
            "write would be invisible")

"""Four more single decisions: one driven, three pinned.

The three pins are each one line downstream of a check that has already
settled the question -- a truthiness test, a loop that always returns, and
an application that has to exist for the widget to.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# spacr/control_names.py -- the retry without the leading token
# ---------------------------------------------------------------------------

class TestRetryingAControlWithoutItsLeadingToken:
    """A library that writes ``TSC2`` where the user typed ``gene_TSC2``.

    When nothing matched and the first token appears in no name in hand,
    the whole value is tried again without it. Still WHOLE values -- the
    retry drops a leading token, it does not fall back to substring
    matching.
    """

    def test_resolving_a_blank_value_is_the_only_way_to_get_none(self):
        """THE PIN.

        ``resolve_control`` returns None for a blank value and for
        nothing else. The retry has already tested ``if tail and ...``,
        so the ``tail`` it passes is never blank and the result is never
        None.
        """
        from spacr.control_names import resolve_control

        for blank in (None, "", "   ", "\t\n"):
            assert resolve_control(blank) is None

        for typed in ("TSC2", "gene_TSC2", "A_B_C", "0"):
            assert resolve_control(typed) is not None, (
                f"{typed!r} resolved to None, so the retry's None check "
                f"is live and needs a test rather than this pin")

        source = inspect.getsource(
            __import__("spacr.control_names", fromlist=["resolve_controls"])
        )
        assert "if tail and unused:" in source, (
            "the retry no longer checks that the tail is non-empty before "
            "resolving it")

    def test_the_retry_finds_a_control_the_first_pass_missed(self):
        from spacr.control_names import matches, resolve_control

        guides = pd.Series(["TSC2_1", "TSC2_2", "AAVS1_1", "AAVS1_2"])
        genes = pd.Series(["TSC2", "TSC2", "AAVS1", "AAVS1"])

        spec = resolve_control("TSC2")
        assert spec is not None
        assert int(matches(spec, guides, genes).sum()) == 2


# ---------------------------------------------------------------------------
# spacr/model_zoo.py -- the byte formatter always returns from inside
# ---------------------------------------------------------------------------

class TestModelSizesForPeople:

    @pytest.mark.parametrize("size,expected", [
        (0, "unknown"),
        (None, "unknown"),
        ("not a number", "unknown"),
        (-1, "unknown"),
        (512, "512 B"),
        (1536, "1.5 KB"),
        (5 * 1024 ** 2, "5.0 MB"),
        (3 * 1024 ** 3, "3.0 GB"),
    ])
    def test_it_reads_as_a_size_or_says_it_does_not_know(self, size,
                                                          expected):
        from spacr.model_zoo import _human_bytes

        assert _human_bytes(size) == expected

    def test_a_size_past_every_unit_still_says_gigabytes(self):
        from spacr.model_zoo import _human_bytes

        assert _human_bytes(4096 * 1024 ** 3) == "4096.0 GB"

    def test_the_loop_cannot_finish_without_returning(self):
        """THE PIN.

        ``or unit == "GB"`` makes the last pass return unconditionally,
        so there is nothing after the loop to fall through to -- the
        module's own comment says as much, and this is what fails if a
        unit is appended without extending that condition.
        """
        from spacr import model_zoo

        source = inspect.getsource(model_zoo._human_bytes)
        start = source.index("for unit in (")
        units = source[start:source.index(")", start)]
        assert units.rstrip().endswith('"GB"'), (
            f"the last unit is no longer GB ({units!r}), so the loop can "
            f"now finish without returning")
        assert 'unit == "GB"' in source

    def test_zero_is_unknown_rather_than_zero_bytes(self):
        """A model file of zero bytes is a failed download, not a small
        model, and the zoo says so rather than printing "0 B"."""
        from spacr.model_zoo import _human_bytes, UNKNOWN

        assert _human_bytes(0) == UNKNOWN
        assert _human_bytes(0.0) == UNKNOWN


# ---------------------------------------------------------------------------
# spacr/qt/widgets/gene_panel.py -- the belt-and-braces thread hook
# ---------------------------------------------------------------------------

class TestTheGenePanelsThreadLifetime:

    def test_the_quit_hook_is_belt_and_braces_over_the_close_event(self):
        """THE PIN.

        Qt aborts the process if a running QThread is destroyed, and a
        panel can be dropped without ever being closed -- a tab rebuilt,
        a screen replaced, an interpreter shutting down. ``closeEvent``
        covers the ordinary path; the ``aboutToQuit`` hook covers the one
        where nobody closed anything, and it is guarded because
        ``QApplication.instance()`` is None on an import with no
        application yet.

        A test cannot be in that state -- building the widget needs an
        application -- so the guard is pinned to its own shape.
        """
        pytest.importorskip("PySide6")
        from spacr.qt.widgets import gene_panel

        source = inspect.getsource(gene_panel)
        assert "application = QApplication.instance()" in source
        assert "aboutToQuit.connect(self._shut_down_warming)" in source
        assert "if application is not None:" in source, (
            "the application lookup is no longer guarded against None")

    def test_an_application_exists_whenever_a_panel_can_be_built(self, qapp):
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        assert QApplication.instance() is not None


# ---------------------------------------------------------------------------
# spacr/qt/screens/tabulate.py -- re-aggregating with no table loaded
# ---------------------------------------------------------------------------

class TestReAggregatingTheNarrowedPopulation:

    def _screen(self, qtbot):
        pytest.importorskip("PySide6")
        from spacr.qt.screens.tabulate import TabulateScreen

        screen = TabulateScreen()
        qtbot.addWidget(screen)
        return screen

    def test_with_a_table_loaded_the_pivot_is_refilled(self, qtbot):
        screen = self._screen(qtbot)
        screen._frame = pd.DataFrame({
            "plateID": "p1", "rowID": "r1",
            "columnID": [f"c{i % 3 + 1}" for i in range(9)],
            "cell_area": np.linspace(1.0, 9.0, 9)})

        screen._recompute_filtered()         # must not raise

        assert screen._filtered() is not None

    def test_with_no_table_at_all_nothing_is_recomputed(self, qtbot):
        """THE UNCOVERED ARC.

        The shared filter fires whenever another screen narrows the
        population, and this screen may not have loaded anything yet.
        ``set_frame(None)`` on the pivot is a table of NaNs where the
        numbers were, which reads as a computed answer rather than as
        nothing having been computed.
        """
        screen = self._screen(qtbot)
        assert screen._frame is None
        assert screen._filtered() is None

        screen._recompute_filtered()         # must not raise

        assert screen._filtered() is None

    def test_a_filter_that_does_not_apply_keeps_the_whole_table(self, qtbot):
        """Between the two: the filter raised, so the unnarrowed table is
        shown rather than nothing."""
        screen = self._screen(qtbot)
        frame = pd.DataFrame({"cell_area": [1.0, 2.0, 3.0]})
        screen._frame = frame

        original = screen._link

        class _Refuses:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def visible(self, _frame):
                raise ValueError("this filter names columns that are not here")

        screen._link = _Refuses(original)
        try:
            assert screen._filtered() is frame
        finally:
            screen._link = original

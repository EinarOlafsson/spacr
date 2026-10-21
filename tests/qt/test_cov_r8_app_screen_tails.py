"""AppScreen's optional-import guards and its teardown paths.

Everything here runs only when something is absent, broken, or already
half-destroyed, which is why none of it was reached: in a healthy suite
every optional module imports and every widget is alive.

Three of the five carry a `# pragma: no cover`, which excludes nothing
in this project -- `.coveragerc` sets `exclude_lines =` to an empty list
-- so they were counted and simply untested.
"""
from __future__ import annotations

import pytest

from spacr.qt.screens import app_screen as mod
from spacr.qt.screens.app_screen import AppScreen

pytestmark = pytest.mark.qt


@pytest.fixture()
def screen(qtbot):
    scr = AppScreen("mask")
    qtbot.addWidget(scr)
    return scr


class TestTheLazyCapabilityProbes:
    """`_sweepable` and `_hyperparam_searchable` are import-guarded.

    Both pull heavy scientific modules, so they are imported lazily and
    a screen must still build when the module is not installed.
    """

    def test_a_missing_sweep_module_means_not_sweepable(self, monkeypatch):
        import builtins

        real = builtins.__import__

        def refuse(name, g=None, l=None, fromlist=(), level=0):
            if "parameter_sweep" in name or "sweepable" in (fromlist or ()):
                raise ImportError("the sweep module is not installed")
            return real(name, g, l, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", refuse)
        assert mod._sweepable("mask") is False

    def test_a_missing_hyperparam_module_means_not_searchable(self,
                                                              monkeypatch):
        import builtins

        real = builtins.__import__

        def refuse(name, g=None, l=None, fromlist=(), level=0):
            if "hyperparam" in name or "searchable" in (fromlist or ()):
                raise ImportError("hyperparam is not installed")
            return real(name, g, l, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", refuse)
        assert mod._hyperparam_searchable("mask") is False

    def test_the_probes_answer_for_a_real_app_key(self):
        """Both halves: with the modules present they return a bool."""
        assert isinstance(mod._sweepable("mask"), bool)
        assert isinstance(mod._hyperparam_searchable("mask"), bool)


class TestTearingDownAScreenThatIsAlreadyHalfGone:

    def test_a_flowview_section_whose_c_half_has_gone_is_survived(
            self, screen, qtbot):
        """`closeEvent` closes FlowView explicitly while Qt objects live.

        If the section's C++ half has already been destroyed, shutdown
        raises RuntimeError -- and a close handler that propagated it
        would leave the screen half-closed.
        """
        class _Gone:
            def __init__(self):
                self.asked = 0

            def shutdown(self):
                self.asked += 1
                raise RuntimeError("Internal C++ object already deleted.")

        section = _Gone()
        screen._flowview_section = section
        screen.close()          # must not raise

        # ASSERTED, not merely survived. "It did not raise" passes just
        # as well against a closeEvent that never reaches the section.
        assert section.asked == 1, "the section was never shut down"
        assert screen.isVisible() is False, "the screen did not close"

    def test_a_flowview_section_that_closes_cleanly_is_shut_down(
            self, screen):
        """The ordinary path, so the guard above is visibly a guard."""
        closed = []

        class _Live:
            def shutdown(self):
                closed.append(1)

        screen._flowview_section = _Live()
        screen.close()
        assert closed == [1]


class TestRaisingTheResultsTab:

    def test_a_tab_widget_that_refuses_is_survived(self, screen, caplog):
        """`setCurrentWidget` raises on a deleted page or a wrong type.

        Failing to raise a tab is a blemish; raising out of the slot that
        tries would lose whatever called it.
        """
        class _Refuses:
            def __init__(self):
                self.asked = 0

            def setCurrentWidget(self, _page):   # noqa: N802 - Qt naming
                self.asked += 1
                raise RuntimeError("Internal C++ object already deleted.")

        tabs = _Refuses()
        screen._results_tabs = tabs
        screen._results_page = object()
        with caplog.at_level("DEBUG"):
            screen._raise_the_results_tab()      # must not raise

        assert tabs.asked == 1, "the tab was never asked to be raised"

    def test_a_type_error_is_caught_as_well(self, screen):
        """A page of the wrong type raises TypeError, not RuntimeError."""
        class _Picky:
            def __init__(self):
                self.asked = 0

            def setCurrentWidget(self, _page):   # noqa: N802 - Qt naming
                self.asked += 1
                raise TypeError("not a QWidget")

        tabs = _Picky()
        screen._results_tabs = tabs
        screen._results_page = object()
        screen._raise_the_results_tab()

        assert tabs.asked == 1, "the tab was never asked to be raised"

    def test_with_no_tabs_or_no_page_it_does_nothing(self, screen):
        """The early return, asserted as one.

        Without an assertion this passes against a version that raises
        the wrong tab -- there is nothing here that could fail.
        """
        screen._results_tabs = None
        screen._results_page = None

        screen._raise_the_results_tab()      # must not raise

        assert screen._results_tabs is None, (
            "the slot invented a tab widget on the way past")
        assert screen._results_page is None


class TestTheFormShapeDecision:
    """`_bulk_apply_changes_form_shape` -- does this change need a rebuild?

    Item 356 as the maintainer extended it on 2026-10-17: a settings file
    follows the rule a typed value does. Only `number_of_organelles` adds
    or removes rows; a channel or organelle-slot switch is shown in place.
    """

    def test_an_object_switch_never_reshapes_the_form(self, screen):
        """A changed channel, for a named object or a counted slot."""
        from spacr.organelle_types import NUMBER_OF_ORGANELLES

        current = {NUMBER_OF_ORGANELLES: 2, "nucleus_channel": None,
                   "organelle_channel": 1, "organelleb_channel": None}
        for settings in ({"nucleus_channel": 0}, {"organelle_channel": 3},
                         {"organelleb_channel": 2}):
            assert screen._bulk_apply_changes_form_shape(
                settings, current) is False, settings

    def test_a_new_organelle_count_reshapes_the_form(self, screen):
        from spacr.organelle_types import NUMBER_OF_ORGANELLES

        current = {NUMBER_OF_ORGANELLES: 1, "organelle_channel": 1}
        assert screen._bulk_apply_changes_form_shape(
            {NUMBER_OF_ORGANELLES: 2}, current) is True
        assert screen._bulk_apply_changes_form_shape(
            {NUMBER_OF_ORGANELLES: 1}, current) is False

    def test_a_form_without_an_organelle_count_is_never_reshaped(self, screen):
        from spacr.organelle_types import NUMBER_OF_ORGANELLES

        assert screen._bulk_apply_changes_form_shape(
            {NUMBER_OF_ORGANELLES: 2, "nucleus_channel": 0},
            {"nucleus_channel": None}) is False

    def test_an_identical_settings_dict_needs_no_new_shape(self, screen):
        current = {"cell_diameter": 30}
        assert screen._bulk_apply_changes_form_shape(dict(current),
                                                     current) is False

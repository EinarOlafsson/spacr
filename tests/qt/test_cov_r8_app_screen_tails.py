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
            def shutdown(self):
                raise RuntimeError("Internal C++ object already deleted.")

        screen._flowview_section = _Gone()
        screen.close()          # must not raise

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
            def setCurrentWidget(self, _page):   # noqa: N802 - Qt naming
                raise RuntimeError("Internal C++ object already deleted.")

        screen._results_tabs = _Refuses()
        screen._results_page = object()
        with caplog.at_level("DEBUG"):
            screen._raise_the_results_tab()      # must not raise

    def test_a_type_error_is_caught_as_well(self, screen):
        """A page of the wrong type raises TypeError, not RuntimeError."""
        class _Picky:
            def setCurrentWidget(self, _page):   # noqa: N802 - Qt naming
                raise TypeError("not a QWidget")

        screen._results_tabs = _Picky()
        screen._results_page = object()
        screen._raise_the_results_tab()

    def test_with_no_tabs_or_no_page_it_does_nothing(self, screen):
        screen._results_tabs = None
        screen._results_page = None
        screen._raise_the_results_tab()


class TestTheFormShapeDecision:
    """`_bulk_apply_changes_form_shape` -- does this change need a rebuild?"""

    def test_an_unnumbered_organelle_role_is_skipped(self, screen,
                                                     monkeypatch):
        """`organelle_number` raises ValueError on a role with no number.

        Only `organelle`, `organelleb`, ... `organellez` carry a slot
        number. A role outside that set cannot be governed by the target
        count, so it is skipped -- rather than taking the whole shape
        decision down with a ValueError raised deep inside a loop.

        Driven by naming a role the numberer refuses. `object_of_setting`
        and `object_switch_keys` are module-level here, so the key can be
        routed to that role without inventing a settings vocabulary.
        """
        from spacr.organelle_types import NUMBER_OF_ORGANELLES

        monkeypatch.setattr(mod, "object_of_setting",
                            lambda _key: "not_a_numbered_slot")
        monkeypatch.setattr(mod, "object_switch_keys",
                            lambda _role: {"some_switch"})

        current = {NUMBER_OF_ORGANELLES: 2, "some_switch": "before"}
        settings = {"some_switch": "after"}
        # The value DIFFERS, so without the skip this would report True.
        assert screen._bulk_apply_changes_form_shape(settings, current) is \
            False, ("a role with no slot number decided the form's shape")

    def test_a_numbered_role_over_the_target_count_is_skipped(self, screen,
                                                              monkeypatch):
        """The neighbouring `continue`: a slot beyond the active count.

        A switch for organelle 5 cannot change the shape of a form that
        is showing two organelles.
        """
        from spacr.organelle_types import NUMBER_OF_ORGANELLES

        monkeypatch.setattr(mod, "object_of_setting",
                            lambda _key: "organellee")     # slot 5
        monkeypatch.setattr(mod, "object_switch_keys",
                            lambda _role: {"some_switch"})

        current = {NUMBER_OF_ORGANELLES: 2, "some_switch": "before"}
        settings = {"some_switch": "after"}
        assert screen._bulk_apply_changes_form_shape(settings, current) is \
            False

    def test_a_numbered_role_within_the_count_does_change_the_shape(
            self, screen, monkeypatch):
        """And the positive side, so the two skips are visibly skips."""
        from spacr.organelle_types import NUMBER_OF_ORGANELLES

        monkeypatch.setattr(mod, "object_of_setting",
                            lambda _key: "organelle")      # slot 1
        monkeypatch.setattr(mod, "object_switch_keys",
                            lambda _role: {"some_switch"})

        current = {NUMBER_OF_ORGANELLES: 2, "some_switch": "before"}
        settings = {"some_switch": "after"}
        assert screen._bulk_apply_changes_form_shape(settings, current) is True

    def test_an_identical_settings_dict_needs_no_new_shape(self, screen):
        current = {"cell_diameter": 30}
        assert screen._bulk_apply_changes_form_shape(dict(current),
                                                     current) is False

"""Choosing `cell` sets what it forces, shows it, and greys it.

Instruction 219's remaining half. The refusal already stops the incompatible
combination before anything is read, and it stays as the backstop for a
settings CSV that never passed through a panel -- but REFUSING AT THE SEAM
IS THE FLOOR, NOT THE ANSWER. A user who has to run and read an error to
learn that two controls disagree is being taught by failure.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from spacr.settings_advisor import refusals, requirements_for_unit


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app):
    from spacr.qt.screens.settings_model import SettingsWidgets

    model = SettingsWidgets("regression")
    model.build_sections()
    return model


LOCKED = tuple(requirements_for_unit("cell"))


class TestChoosingCellLocksThem:

    def test_the_values_are_applied(self, panel):
        panel.set_value_for_key("analysis_unit", "cell")
        wanted = requirements_for_unit("cell")
        for key, value in wanted.items():
            assert panel._read_widget(panel._widgets[key]) == value

    @pytest.mark.parametrize("key", LOCKED)
    def test_the_control_is_disabled(self, panel, key):
        panel.set_value_for_key("analysis_unit", "cell")
        assert not panel._widgets[key].isEnabled()

    @pytest.mark.parametrize("key", LOCKED)
    def test_the_control_says_why(self, panel, key):
        """The way the mode-inapplicable picture settings already do.

        NOT ALWAYS NAMING `analysis_unit`. Two of the three already had a
        rule of their own -- `analysis_mode` is greyed because
        inference='parametric' selects it, `agg_type` because the unit is
        not 'well' -- and those reasons are MORE specific than a general
        "the unit fixed this". What every greyed control must carry is the
        house sentence saying the value is kept, which is what tells the
        user the run will use what they can see.
        """
        panel.set_value_for_key("analysis_unit", "cell")
        tip = panel._widgets[key].toolTip()
        assert "kept and saved" in tip or "analysis_unit" in tip, tip

    def test_the_setting_only_this_rule_locks_names_it(self, panel):
        """`inference` had no rule of its own, which is the gap this fills."""
        panel.set_value_for_key("analysis_unit", "cell")
        assert "analysis_unit" in panel._widgets["inference"].toolTip()

    def test_the_shown_value_is_the_one_the_run_will_use(self, panel):
        """A greyed control still showing the old value tells the user the
        run will use that value, and it will not -- which is worse than an
        editable control that disagrees, because it looks settled."""
        panel.set_value_for_key("inference", "nonparametric")
        panel.set_value_for_key("analysis_unit", "cell")
        assert panel._read_widget(panel._widgets["inference"]) == "parametric"


class TestChoosingWellReleasesThem:

    @pytest.mark.parametrize("key", LOCKED)
    def test_the_control_comes_back(self, panel, key):
        """UNLESS ANOTHER RULE STILL GREYS IT, which is the whole point.

        `analysis_mode` is greyed by the inference rule -- it is set for you
        by inference='parametric' -- and that has nothing to do with the
        unit. Releasing it here would enable a control something else is
        still deciding the value of, and the user would change it and be
        ignored. So what is asserted is that THIS rule's lock is gone, not
        that the control is editable.
        """
        panel.set_value_for_key("analysis_unit", "cell")
        panel.set_value_for_key("analysis_unit", "well")
        if key in panel._unit_locked:
            pytest.fail(f"{key} is still locked by the unit")
        tip = panel._widgets[key].toolTip()
        assert "analysis_unit" not in tip or "'well'" in tip

    @pytest.mark.parametrize("key", LOCKED)
    def test_the_reason_goes_with_it(self, panel, key):
        panel.set_value_for_key("analysis_unit", "cell")
        panel.set_value_for_key("analysis_unit", "well")
        assert "analysis_unit" not in panel._widgets[key].toolTip()

    def test_it_releases_every_key_any_unit_locks(self, panel):
        """Refreshing only the CURRENT unit's keys would leave a control
        greyed after the reason for it was withdrawn."""
        panel.set_value_for_key("analysis_unit", "cell")
        assert panel._unit_locked
        panel.set_value_for_key("analysis_unit", "well")
        assert panel._unit_locked == set(), (
            "a key this rule locked is still recorded as locked after the "
            "reason for it was withdrawn")

    def test_a_control_another_rule_greys_stays_greyed(self, panel):
        """Enabling a control another rule disabled is worse than leaving
        one greyed: the user changes it and the run ignores them."""
        panel.set_value_for_key("analysis_unit", "cell")
        panel.set_value_for_key("analysis_unit", "well")
        panel.set_value_for_key("inference", "parametric")
        assert not panel._widgets["analysis_mode"].isEnabled()


class TestThePanelAndThePreflightAgree:
    """The list lives in settings_advisor; a copy in the panel would be a
    second opinion and the run would keep the casting vote."""

    def test_what_the_panel_applies_is_runnable(self, panel):
        panel.set_value_for_key("analysis_unit", "cell")
        settings = dict(panel.collect() or {})
        settings["analysis_unit"] = "cell"
        assert not refusals(settings), refusals(settings)

    def test_the_refusal_remains_as_the_backstop(self):
        """For a settings CSV that never passed through a panel."""
        assert refusals({"analysis_unit": "cell",
                         "analysis_mode": "guide_permutation"})

    def test_a_well_run_is_not_refused_for_this(self):
        assert not any(
            "analysis_unit" in r
            for r in refusals({"analysis_unit": "well",
                               "analysis_mode": "guide_permutation"}))

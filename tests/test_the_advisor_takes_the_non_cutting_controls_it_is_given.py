"""Item 288: the settings advisor and the non-cutting controls it is told about.

When the user names non-cutting control guides, the advisor proposes them
as ``nontargeting_control_grnas`` -- each one, trimmed, blanks dropped --
and says why. When they name none, it leaves the setting undecided and
explains the baseline that implies. Both are what the advisor dialog shows.
"""
from __future__ import annotations

from spacr.settings_advisor import advise
from tests.test_cov_w2_4_settings_advisor import _reading


def test_named_controls_are_proposed_each_one_trimmed():
    advice = advise(_reading(), {"nontargeting_control_grnas":
                                 " NTC_1, ,NTC_2 "})
    assert advice.as_settings()["nontargeting_control_grnas"] == \
        ["NTC_1", "NTC_2"]
    why = advice.why("nontargeting_control_grnas")
    assert "'NTC_1', 'NTC_2'" in why and "non-cutting control" in why


def test_no_controls_named_leaves_the_baseline_undecided():
    advice = advise(_reading(), {})
    assert "nontargeting_control_grnas" not in advice.as_settings()
    assert "measured from zero" in advice.why("nontargeting_control_grnas")

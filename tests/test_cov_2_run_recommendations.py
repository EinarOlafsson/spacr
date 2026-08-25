"""The two diagnostics that only fire on their own: collinearity and thin guides.

Both rules are unconditional on settings, so a run that trips them and gets no
recommendation back has silently lost the only warning it was going to get.
The VIF rule tells the user their coefficients are not separable; the
wells-per-guide rule is blocking, because a median gRNA seen in fewer than
three wells cannot support the families the sweep would resolve.
"""
from __future__ import annotations

from spacr.run_recommendations import VIF_HIGH, recommend


def _settings_by(items):
    return {item.setting for item in items}


def test_a_high_vif_recommends_a_penalised_regression():
    """Above the VIF bar the individual coefficients are not separable.

    The rule has no settings escape hatch: whatever regression type is
    configured, overlapping predictors are a fact about the design, and the
    recommendation names the penalised alternatives.
    """
    items = recommend({"max_vif": VIF_HIGH + 5.0})

    hits = [i for i in items if i.setting == "regression_type"]
    assert len(hits) == 1
    assert "ridge" in hits[0].action
    assert f"{VIF_HIGH + 5.0:.1f}" in hits[0].because
    assert hits[0].severity == "consider"

    # Exactly at the bar is not above it.
    assert "regression_type" not in _settings_by(recommend({"max_vif": VIF_HIGH}))


def test_too_few_wells_per_guide_blocks_and_asks_for_the_sweep():
    """A median gRNA in fewer than three wells cannot resolve support families.

    This one is blocking rather than advisory, because the numbers downstream
    are not interpretable at all -- so it must sort ahead of any advisory
    recommendation the same run produced.
    """
    items = recommend({"median_wells_per_guide": 2.0, "max_vif": VIF_HIGH + 5.0})

    hits = [i for i in items if i.setting == "fraction_threshold"]
    assert len(hits) == 1
    assert hits[0].severity == "blocking"
    assert "2 well(s)" in hits[0].because
    assert items[0] is hits[0], "blocking recommendations sort first"

    # Three wells is enough; the rule stays quiet.
    assert "fraction_threshold" not in _settings_by(
        recommend({"median_wells_per_guide": 3.0}))

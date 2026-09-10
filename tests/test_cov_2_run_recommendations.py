"""The two diagnostics that only fire on their own: collinearity and thin guides.

Both rules are unconditional on settings, so a run that trips them and gets no
recommendation back has silently lost the only warning it was going to get.
The VIF rule tells the user their coefficients are not separable; the
wells-per-guide rule is blocking, because a median gRNA seen in fewer than
three wells cannot support the families the sweep would resolve.
"""
from __future__ import annotations

from spacr.run_recommendations import Recommendation, VIF_HIGH, recommend


def _settings_by(items):
    return {item.setting for item in items}


def test_recommendation_lines_distinguish_blocking_from_advisory_actions():
    """The display marker exposes severity without changing its explanation."""
    advisory = Recommendation("alpha", "raise it", "the fit was noisy")
    blocking = Recommendation(
        "alpha", "raise it", "the fit was noisy", severity="blocking")

    assert advisory.line() == (
        "  - alpha: raise it\n      because the fit was noisy")
    assert blocking.line() == (
        "  ! alpha: raise it\n      because the fit was noisy")


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


def test_every_recommended_setting_is_a_setting_that_exists():
    """Spelled the way the run reads them -- checked, not eyeballed.

    A recommendation naming a key no panel offers and no run reads is worse
    than none: the reader follows it, nothing changes, and the section stops
    being believed. The names are read off the rules themselves rather than
    listed here, so a rule added later is checked the day it lands.
    """
    import ast
    import inspect

    from spacr.settings import expected_types
    import spacr.run_recommendations as module

    tree = ast.parse(inspect.getsource(module))
    named = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "Recommendation"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    # Guards the guard: an empty set would assert nothing at all.
    assert len(named) >= 5, named
    missing = sorted(k for k in named if k not in expected_types)
    assert not missing, f"recommended settings that do not exist: {missing}"


def test_the_settings_a_rule_reads_are_the_settings_it_names():
    """A rule suppressed by what the run already did must read the same key.

    `recommend` skips a recommendation when the run already applied it, and
    it does that by reading the settings mapping. If the key it READS were
    spelled differently from the key it NAMES, the advice would repeat
    itself forever and the suppression would look like a threshold bug.
    """
    from spacr.run_recommendations import recommend

    diagnostics = {"normality_p": 1e-40, "durbin_watson": 1.2}
    named = {item.setting for item in recommend(diagnostics, settings={})}
    assert "inference" in named and "guide_nuisance_columns" in named
    applied = recommend(
        diagnostics,
        settings={"inference": "nonparametric",
                  "guide_nuisance_columns": ["rowID", "columnID"]},
    )
    assert "inference" not in {item.setting for item in applied}

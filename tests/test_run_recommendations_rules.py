"""Which regression-diagnostic rules ``spacr.run_recommendations.recommend`` fires.

Each rule is a fixed threshold on one measured diagnostic. A run that trips a
rule and gets no recommendation back has lost the only warning it was going to
get, while a rule that keeps firing after the run already applied its advice
repeats itself forever. The table below drives each threshold from both sides
and the suppression a run's own settings provide.
"""
from __future__ import annotations

from spacr.run_recommendations import VIF_HIGH, recommend


def _fired(items):
    """``(setting, severity)`` per recommendation, in the order returned."""
    return [(item.setting, item.severity) for item in items]


def test_recommend_fires_each_rule_once_and_sorts_blocking_first(subtests):
    """Every rule fires above its bar, stays quiet at it, and blocking sorts first."""
    cases = [
        ("high VIF", {"max_vif": VIF_HIGH + 5.0}, None,
         [("regression_type", "consider")]),
        ("VIF exactly at the bar", {"max_vif": VIF_HIGH}, None, []),
        ("thin guides", {"median_wells_per_guide": 2.0}, None,
         [("fraction_threshold", "blocking")]),
        ("three wells is enough", {"median_wells_per_guide": 3.0}, None, []),
        ("blocking ahead of advisory",
         {"median_wells_per_guide": 2.0, "max_vif": VIF_HIGH + 5.0}, None,
         [("fraction_threshold", "blocking"), ("regression_type", "consider")]),
        ("non-normal and correlated residuals, nothing applied",
         {"normality_p": 1e-40, "durbin_watson": 1.2}, {},
         [("inference", "blocking"), ("guide_nuisance_columns", "blocking")]),
        ("the run already applied both",
         {"normality_p": 1e-40, "durbin_watson": 1.2},
         {"inference": "nonparametric",
          "guide_nuisance_columns": ["rowID", "columnID"]},
         [("guide_permutation_block", "consider")]),
        ("no diagnostics at all", {}, None, []),
    ]
    for case_id, diagnostics, settings, expected in cases:
        with subtests.test(msg=case_id):
            assert _fired(recommend(diagnostics, settings=settings)) == expected

    # A rule that fires quotes the number that tripped it.
    vif, = recommend({"max_vif": VIF_HIGH + 5.0})
    assert vif.action == "'ridge', 'lasso' or 'elasticnet'"
    assert vif.because == (
        "the largest VIF is 15.0, above 10, so the predictors carry "
        "overlapping information and their individual coefficients are not "
        "separable.")
    wells, = recommend({"median_wells_per_guide": 2.0})
    assert wells.action == "None, to sweep it"
    assert wells.because.startswith(
        "the median gRNA is in 2 well(s) after the threshold")
    assert wells.line().startswith("  ! fraction_threshold: None, to sweep it\n")

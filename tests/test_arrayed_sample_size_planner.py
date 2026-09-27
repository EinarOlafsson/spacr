"""Sample-size planning for arrayed assays from a pilot plate's variance.

The analytic power must equal statsmodels' t-test power for the same
replicate-mean variance, and the planner's recommended design, built from
components estimated on a simulated pilot plate, must reach its target
power when whole experiments of that design are simulated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.sp_stats import (
    _arrayed_power,
    _nested_variance_components,
    _plan_arrayed_design,
    _replicate_mean_variance,
    _simulate_arrayed_power,
)

TRUE = {"replicate": 0.3, "well": 0.2, "field": 0.1, "cell": 4.0,
        "cells_per_field": 20}


def _pilot(seed=1, replicates=5, wells=10, fields=6):
    rng = np.random.default_rng(seed)
    parts = []
    for r in range(replicates):
        rep = rng.normal(0, np.sqrt(TRUE["replicate"]))
        for w in range(wells):
            well = rng.normal(0, np.sqrt(TRUE["well"]))
            for f in range(fields):
                field = rng.normal(0, np.sqrt(TRUE["field"]))
                n = int(rng.integers(12, 30))
                parts.append(pd.DataFrame({
                    "intensity": 5 + rep + well + field
                    + rng.normal(0, np.sqrt(TRUE["cell"]), n),
                    "plateID": f"p{r}", "prc": f"p{r}_r1_c{w}",
                    "fieldID": f}))
    return pd.concat(parts, ignore_index=True)


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("design", [(3, 2, 4), (6, 3, 9), (4, 1, 1)])
def test_power_matches_statsmodels(paired, design):
    from statsmodels.stats.power import TTestIndPower, TTestPower

    replicates, wells, fields = design
    var = _replicate_mean_variance(TRUE, wells, fields, 20, paired=paired)
    expected = (TTestPower().power(1.0 / np.sqrt(2 * var), replicates, 0.05)
                if paired else
                TTestIndPower().power(1.0 / np.sqrt(var), replicates, 0.05))
    got = _arrayed_power(TRUE, 1.0, replicates=replicates, wells=wells,
                         fields=fields, paired=paired)
    assert got == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("paired", [False, True])
def test_power_matches_simulation(paired):
    analytic = _arrayed_power(TRUE, 1.0, replicates=4, wells=2, fields=3,
                              paired=paired)
    simulated = _simulate_arrayed_power(TRUE, 1.0, replicates=4, wells=2,
                                        fields=3, paired=paired, n_sim=3000,
                                        seed=7)
    assert simulated == pytest.approx(analytic, abs=0.03)


def test_pilot_components_recover_the_truth():
    """Tolerances are about three standard errors of each estimate."""
    comps = _nested_variance_components(_pilot(), "intensity",
                                        replicate="plateID")
    assert comps["cell"] == pytest.approx(TRUE["cell"], rel=0.05)
    assert comps["field"] == pytest.approx(TRUE["field"], abs=0.08)
    assert comps["well"] == pytest.approx(TRUE["well"], abs=0.15)
    assert comps["replicate"] == pytest.approx(TRUE["replicate"], abs=0.45)
    assert comps["n_replicates"] == 5 and comps["n_wells"] == 50
    assert all(comps["estimated"].values())


def test_recommendation_reaches_target_in_simulation():
    comps = _nested_variance_components(_pilot(), "intensity",
                                        replicate="plateID")
    designs = _plan_arrayed_design(comps, 1.0, power=0.8)
    assert not designs.empty
    assert (designs["power"] >= 0.8).all()
    assert designs["cost"].is_monotonic_increasing
    best = designs.iloc[0]
    fewer = _arrayed_power(comps, 1.0, replicates=int(best.replicates) - 1,
                           wells=int(best.wells), fields=int(best.fields))
    assert fewer < 0.8 or best.replicates == 2
    simulated = _simulate_arrayed_power(
        comps, 1.0, replicates=int(best.replicates), wells=int(best.wells),
        fields=int(best.fields), n_sim=3000, seed=11)
    assert simulated == pytest.approx(best.power, abs=0.03)


def test_single_replicate_pilot_flags_replicate_variance():
    pilot = _pilot(replicates=1)
    comps = _nested_variance_components(pilot, "intensity")
    assert comps["estimated"]["replicate"] is False
    assert np.isnan(comps["replicate"])
    assert comps["n_replicates"] == 1
    assert _arrayed_power(comps, 1.0, replicates=3, wells=2,
                          fields=3) > 0.05


def test_condition_pooling_ignores_the_treatment_shift():
    pilot = _pilot()
    shifted = pilot.assign(intensity=pilot["intensity"] + 10.0,
                           prc=pilot["prc"] + "_t", treatment="drug")
    both = pd.concat([pilot.assign(treatment="ctrl"), shifted])
    comps = _nested_variance_components(both, "intensity",
                                        replicate="plateID",
                                        condition="treatment")
    assert comps["well"] < 1.0 and comps["replicate"] < 1.0


def test_unreachable_target_and_bad_input():
    assert _plan_arrayed_design(TRUE, 0.01, max_replicates=3, max_wells=2,
                                max_fields=2).empty
    assert np.isnan(_arrayed_power(TRUE, 1.0, replicates=1, wells=1,
                                   fields=1))
    with pytest.raises(KeyError):
        _nested_variance_components(_pilot(), "missing")
    with pytest.raises(ValueError):
        _nested_variance_components(
            pd.DataFrame({"v": [np.nan], "prc": ["a"], "fieldID": [1]}), "v")

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
    _condition_cell_variances,
    _default_cells,
    _nested_variance_components,
    _plan_arrayed_design,
    _replicate_mean_variance,
    _resample_arrayed_power,
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
    # 2026-09-30: a one-condition pilot cannot estimate the
    # replicate-by-condition term added then; every other level it can.
    estimated = dict(comps["estimated"])
    assert estimated.pop("replicate_condition") is False
    assert all(estimated.values())


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


# 2026-09-30: count and proportion readouts, the replicate-by-condition
# term and a model-free resampling check (item 585's audit).

WITH_RC = {**TRUE, "replicate_condition": 0.15, "mean": 5.0}
PROPORTION = {"replicate": 0.002, "replicate_condition": 0.001,
              "well": 0.002, "field": 0.001, "cell": 0.2,
              "cells_per_field": 30, "mean": 0.3}
COUNT = {"replicate": 0.05, "replicate_condition": 0.03, "well": 0.05,
         "field": 0.05, "cell": 6.0, "cells_per_field": 15, "mean": 2.0}


def _two_condition_pilot(seed=3, replicates=8, wells=6, fields=5, rc=0.2):
    """Simulate a pilot with a control and a treated arm on every plate.

    :param seed: random seed.
    :param replicates: plates, one biological replicate each.
    :param wells: wells per condition per plate.
    :param fields: fields per well.
    :param rc: the replicate-by-condition variance.
    :returns: one row per cell.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for r in range(replicates):
        rep = rng.normal(0, np.sqrt(TRUE["replicate"]))
        for arm, shift in (("ctrl", 0.0), ("drug", 2.0)):
            inter = rng.normal(0, np.sqrt(rc))
            for w in range(wells):
                well = rng.normal(0, np.sqrt(TRUE["well"]))
                for f in range(fields):
                    field = rng.normal(0, np.sqrt(TRUE["field"]))
                    parts.append(pd.DataFrame({
                        "intensity": 5 + shift + rep + inter + well + field
                        + rng.normal(0, np.sqrt(TRUE["cell"]), 20),
                        "plateID": f"p{r}", "arm": arm,
                        "prc": f"p{r}_{arm}_c{w}", "fieldID": f}))
    return pd.concat(parts, ignore_index=True)


@pytest.mark.parametrize("readout,comps,effect", [
    ("continuous", WITH_RC, 1.0), ("proportion", PROPORTION, 0.1),
    ("proportion", PROPORTION, -0.1), ("count", COUNT, 0.6)])
@pytest.mark.parametrize("paired", [False, True])
def test_unequal_variance_power_matches_statsmodels(readout, comps, effect,
                                                    paired):
    """Each condition keeps its own replicate-mean variance.

    statsmodels' standardised effect for the same test is the difference
    over the root mean of the two variances (unpaired) or over the root of
    their sum (the paired differences).
    """
    from statsmodels.stats.power import TTestIndPower, TTestPower

    cell0, cell1 = _condition_cell_variances(comps, effect, readout)
    v0, v1 = (_replicate_mean_variance(comps, 2, 3, comps["cells_per_field"],
                                       paired=paired, cell=c)
              for c in (cell0, cell1))
    expected = (TTestPower().power(abs(effect) / np.sqrt(v0 + v1), 5, 0.05)
                if paired else
                TTestIndPower().power(abs(effect) / np.sqrt((v0 + v1) / 2),
                                      5, 0.05))
    got = _arrayed_power(comps, effect, replicates=5, wells=2, fields=3,
                         cells=comps["cells_per_field"], paired=paired,
                         readout=readout)
    assert got == pytest.approx(expected, abs=1e-6)


def test_readout_cell_variances_follow_the_mean():
    upper = sum(PROPORTION[k] for k in ("replicate", "replicate_condition",
                                        "well", "field"))
    c0, c1 = _condition_cell_variances(PROPORTION, 0.2, "proportion")
    assert c0 == pytest.approx(0.3 * 0.7 - upper)
    assert c1 == pytest.approx(0.5 * 0.5 - upper)
    c0, c1 = _condition_cell_variances(COUNT, 1.0, "count")
    assert (c0, c1) == pytest.approx((6.0, 9.0))
    under = {**COUNT, "cell": 1.0}
    assert _condition_cell_variances(under, 1.0, "count") == pytest.approx(
        (2.0, 3.0))
    assert _condition_cell_variances(TRUE, 3.0) == (4.0, 4.0)
    for readout, comps, effect in (("proportion", PROPORTION, 0.8),
                                   ("count", COUNT, -3.0),
                                   ("ratio", COUNT, 1.0)):
        with pytest.raises(ValueError):
            _condition_cell_variances(comps, effect, readout)
    with pytest.raises(ValueError):
        _plan_arrayed_design(PROPORTION, 0.9, readout="proportion")


@pytest.mark.parametrize("readout,comps,effect,design", [
    ("continuous", WITH_RC, 1.0, (4, 2, 3)),
    ("proportion", PROPORTION, 0.1, (4, 2, 3)),
    ("proportion", PROPORTION, -0.1, (4, 2, 3)),
    ("count", COUNT, 0.6, (4, 2, 3)),
    ("count", {**COUNT, "cell": 2.0}, 0.5, (3, 2, 2))])
@pytest.mark.parametrize("paired", [False, True])
def test_every_readout_matches_simulation(readout, comps, effect, design,
                                          paired):
    """Cells drawn as normal, 0/1 or (overdispersed) counts.

    3000 experiments put the Monte Carlo standard error near 0.009.
    """
    replicates, wells, fields = design
    analytic = _arrayed_power(comps, effect, replicates=replicates,
                              wells=wells, fields=fields, paired=paired,
                              readout=readout)
    simulated = _simulate_arrayed_power(
        comps, effect, replicates=replicates, wells=wells, fields=fields,
        paired=paired, readout=readout, n_sim=3000, seed=5)
    assert simulated == pytest.approx(analytic, abs=0.035)


def test_replicate_by_condition_term_is_estimated_and_separated():
    """Over twenty pilots the interaction is recovered, not left in the
    replicate component, and a pilot without it estimates about zero."""
    rc, rep = [], []
    for seed in range(20):
        comps = _nested_variance_components(
            _two_condition_pilot(seed=seed), "intensity",
            replicate="plateID", condition="arm")
        rc.append(comps["replicate_condition"])
        rep.append(comps["replicate"])
        assert comps["estimated"]["replicate_condition"]
        assert comps["n_replicates"] == 8 and comps["n_conditions"] == 2
    assert np.mean(rc) == pytest.approx(0.2, abs=0.06)
    assert np.mean(rep) == pytest.approx(TRUE["replicate"], abs=0.1)
    none = [_nested_variance_components(
        _two_condition_pilot(seed=s, rc=0.0), "intensity",
        replicate="plateID", condition="arm")["replicate_condition"]
        for s in range(10)]
    assert np.mean(none) < 0.05
    single = _nested_variance_components(_pilot(), "intensity",
                                         replicate="plateID")
    assert single["estimated"]["replicate_condition"] is False


def test_the_interaction_does_not_cancel_in_a_paired_design():
    base = {**TRUE, "replicate_condition": 0.0}
    with_rc = {**TRUE, "replicate_condition": 0.3}
    kwargs = dict(replicates=4, wells=2, fields=3, paired=True)
    assert _arrayed_power(with_rc, 1.0, **kwargs) < _arrayed_power(
        base, 1.0, **kwargs)
    more = _plan_arrayed_design(with_rc, 1.0, paired=True, max_wells=2,
                                max_fields=3)
    fewer = _plan_arrayed_design(base, 1.0, paired=True, max_wells=2,
                                 max_fields=3)
    assert more["replicates"].min() > fewer["replicates"].min()


def test_default_cells_is_the_harmonic_mean():
    pilot = _pilot()
    comps = _nested_variance_components(pilot, "intensity",
                                        replicate="plateID")
    sizes = pilot.groupby(["plateID", "prc", "fieldID"]).size()
    assert comps["cells_per_field_effective"] == pytest.approx(
        len(sizes) / (1.0 / sizes).sum())
    assert _default_cells(comps) == comps["cells_per_field_effective"]
    assert _default_cells({"cells_per_field": 7.0}) == 7.0
    assert _default_cells({}) == 1.0


def test_resampling_brackets_the_model_on_a_normal_pilot():
    """A large normal pilot: the null holds its level, and the planned
    power sits between distinct and with-replacement resampling."""
    pilot = _pilot(seed=4, replicates=6, wells=24, fields=12)
    comps = _nested_variance_components(pilot, "intensity",
                                        replicate="plateID")
    shape = dict(replicates=6, wells=4, fields=4)
    null = _resample_arrayed_power(pilot, "intensity", 0.0, **shape,
                                   n_sim=2000, seed=1)
    assert null == pytest.approx(0.05, abs=0.02)
    planned = _arrayed_power(comps, 1.0, **shape)
    distinct = _resample_arrayed_power(pilot, "intensity", 1.0, **shape,
                                       n_sim=2000, seed=2)
    repeated = _resample_arrayed_power(pilot, "intensity", 1.0, **shape,
                                       n_sim=2000, seed=3, replace=True)
    assert repeated - 0.03 <= planned <= distinct + 0.03
    for readout, effect in (("proportion", 0.1), ("count", 0.5)):
        values = ((pilot["intensity"] > 5).astype(float)
                  if readout == "proportion"
                  else np.round(np.clip(pilot["intensity"], 0, None)))
        other = pilot.assign(intensity=values)
        for signed in (effect, -effect):
            assert 0.0 <= _resample_arrayed_power(
                other, "intensity", signed, **shape, readout=readout,
                n_sim=200, seed=4) <= 1.0


def test_resampling_refuses_a_pilot_too_small_for_the_design():
    with pytest.raises(ValueError):
        _resample_arrayed_power(_pilot(), "intensity", 1.0, replicates=3,
                                wells=6, fields=3)
    with pytest.raises(ValueError):
        _resample_arrayed_power(_pilot(), "intensity", 1.0, replicates=3,
                                wells=2, fields=2, readout="ratio")
    assert 0.0 <= _resample_arrayed_power(
        _pilot(), "intensity", 1.0, replicates=3, wells=6, fields=3,
        replace=True, n_sim=100) <= 1.0

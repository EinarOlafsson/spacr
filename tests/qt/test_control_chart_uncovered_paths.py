"""Control chart edges: overflowing z, an empty result, and a rule set of none.

Every case here is a path the ordinary campaign never walks. They are worth
pinning anyway, because each one is a place where the chart could report
something reassuring about a campaign that is not fine: a plate so far outside
the limits that ``value - centre`` overflows, a result assembled with nothing
in it, and a chart run with no rules at all, which must not then claim a
false-alarm budget it does not have.

The arithmetic is worked out beside the data rather than sampled, so there is
no seed and no tolerance to argue about.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets import control_chart as CC
from spacr.qt.widgets.control_chart import (
    ABOVE, ESTIMATOR_MOVING_RANGE, RULES_LIMITS_ONLY, RULE_BEYOND_3_SIGMA,
    ControlChartError, ControlChartSpec, Violation, control_chart,
)


# ---------------------------------------------------------------------------
# The campaigns
# ---------------------------------------------------------------------------

def _quiet_campaign(plates: int = 25) -> pd.DataFrame:
    """Plates alternating 10 and 12 with an explicit run-order column.

    Every moving range is exactly 2, so ``MR-bar = 2`` and
    ``sigma = 2 / 1.128`` exactly, and the centre is exactly 11. One control
    well per plate, so the subgroup sizes do not vary; the order is given, so
    nothing is inferred. Nothing about it trips a rule.
    """
    return pd.DataFrame({
        "plate": [f"P{i:02d}" for i in range(plates)],
        "run": list(range(plates)),
        "signal": [10.0 if i % 2 == 0 else 12.0 for i in range(plates)],
    })


def _overflowing_campaign() -> pd.DataFrame:
    """Eight baseline plates near ``-2.2e307`` and one plate at ``1.79e308``.

    Both ends are ordinary finite floats and every one of them survives the
    ``isfinite`` filter, but their difference does not: ``1.79e308`` minus a
    centre line of ``-2.2e307`` is larger than a float can hold, so the z of
    the last plate comes out infinite. The baseline plates are spread by
    ``1e300``, which is far enough above the relative sigma tolerance that the
    chart is not called degenerate.
    """
    baseline = [-2.2e307 + step * 1.0e300 for step in range(8)]
    values = baseline + [-2.2e307, 1.79e308]
    return pd.DataFrame({
        "plate": [f"P{i:02d}" for i in range(len(values))],
        "run": list(range(len(values))),
        "signal": values,
    })


# ---------------------------------------------------------------------------
# A z that overflows
# ---------------------------------------------------------------------------

def test_a_plate_whose_distance_from_the_centre_overflows_still_has_a_z():
    """The extreme plate's z is infinite, and the chart is not degenerate.

    This is the precondition for the two tests below: if the arithmetic did not
    actually overflow they would be asserting about nothing.
    """
    spec = ControlChartSpec(value="signal", plate="plate", order="run",
                            baseline_n=8, rules=RULES_LIMITS_ONLY)
    with np.errstate(over="ignore"):
        result = control_chart(_overflowing_campaign(), spec)

    assert not result.degenerate
    assert np.isfinite(result.sigma_within)
    assert np.isinf(result.z[9])
    assert np.isfinite(result.z[:9]).all()


def test_zone_puts_an_overflowing_point_in_the_outermost_band_not_the_innermost():
    """A point beyond every limit is zone 3, the band a renderer paints red.

    Reporting zone 0 for it would colour the worst plate of the campaign as
    though it had never left one sigma, and would disagree with rule 1, which
    flags the very same plate.
    """
    spec = ControlChartSpec(value="signal", plate="plate", order="run",
                            baseline_n=8, rules=RULES_LIMITS_ONLY)
    with np.errstate(over="ignore"):
        result = control_chart(_overflowing_campaign(), spec)

    assert result.zone(9) == 3
    assert result.rules_at(9) == (RULE_BEYOND_3_SIGMA,)
    assert bool(result.flagged[9])
    # The finite points are still banded by their own z: plate 4 sits at
    # z = 0.564, inside one sigma.
    assert result.zone(4) == 0


def test_an_overflowing_point_does_not_break_the_points_frame():
    """``points_frame`` still produces one row per plate, infinity included.

    It calls :meth:`ControlChartResult.rules_at` for every point, so a plate
    the zone arithmetic could not band would take the whole CSV down with it.
    """
    spec = ControlChartSpec(value="signal", plate="plate", order="run",
                            baseline_n=8, rules=RULES_LIMITS_ONLY)
    with np.errstate(over="ignore"):
        result = control_chart(_overflowing_campaign(), spec)

    frame = result.points_frame()
    assert len(frame) == 10
    assert frame["rules"].iloc[9] == "1"
    assert np.isinf(frame["z"].iloc[9])


# ---------------------------------------------------------------------------
# A violation and a result with nothing in them
# ---------------------------------------------------------------------------

def test_a_violation_carrying_no_plates_says_so_rather_than_naming_a_blank_span():
    """``where()`` on a point-less violation reads "no plates".

    Without the guard the span arithmetic would still produce a sentence — the
    string ``"plates "`` with nothing after it — and a report would print a
    rule firing on a plate it could not name.
    """
    empty = Violation(rule=RULE_BEYOND_3_SIGMA, start=0, end=0,
                      points=(), plates=(), side=ABOVE)

    assert empty.where() == "no plates"
    assert "no plates" in empty.describe()


def test_a_result_with_no_plates_headlines_that_instead_of_reading_empty_limits():
    """``headline()`` on a plate-less result is a sentence, not an exception.

    :func:`control_chart` refuses to build one, but the result is a public
    frozen dataclass and ``replace`` will make one. Every other branch of the
    headline reaches for ``self.lower.min()`` or ``self.plates[first]``, which
    on an empty chart is a ``ValueError`` out of a method documented to return
    a sentence.
    """
    result = control_chart(
        _quiet_campaign(),
        ControlChartSpec(value="signal", plate="plate", order="run",
                         baseline_n=20))
    emptied = replace(result, plates=())

    assert len(emptied) == 0
    assert emptied.headline() == "no plates."


def test_a_report_with_no_baseline_plates_omits_the_phase_one_line():
    """No baseline means no "Limits estimated from those" claim in the report.

    The rest of the report still prints: dropping the Phase I paragraph must
    not drop the centre, the limits or the headline with it.
    """
    result = control_chart(
        _quiet_campaign(),
        ControlChartSpec(value="signal", plate="plate", order="run",
                         baseline_n=20))
    unbaselined = replace(result, baseline=np.asarray([], dtype=int))

    assert unbaselined.baseline_plates == ()
    text = unbaselined.report()
    assert "Baseline (Phase I)" not in text
    assert "Limits estimated from those" not in text
    assert "Centre 11;" in text
    assert "Run order: the run column." in text


# ---------------------------------------------------------------------------
# A chart with no rules
# ---------------------------------------------------------------------------

def test_a_chart_with_no_rules_selected_claims_no_false_alarm_budget():
    """An empty rule set has a zero alarm rate and no arithmetic to report.

    The caveat exists to say "expect roughly N false alarms even if nothing is
    wrong". With no rule running there are none, and printing the sentence with
    a divide-by-zero in it would be worse than silence.
    """
    spec = ControlChartSpec(value="signal", plate="plate", order="run",
                            baseline_n=20, rules=())
    result = control_chart(_quiet_campaign(), spec)

    assert result.rules == ()
    assert result.false_alarm_rate() == 0.0
    assert not any("false-alarm rate" in caveat
                   for caveat in result.caveats())


def test_a_clean_chart_with_no_rules_has_nothing_to_caveat():
    """Explicit order, a baseline shorter than the campaign, no rules: silence.

    Every caveat this module can raise is about something the reader has to
    know before believing the chart, so a chart with nothing to disclose must
    produce an empty tuple rather than a reassurance.
    """
    spec = ControlChartSpec(value="signal", plate="plate", order="run",
                            baseline_n=20, rules=())
    result = control_chart(_quiet_campaign(), spec)

    assert result.caveats() == ()
    assert result.violations == ()
    assert not result.order_inferred
    assert result.baseline.size == 20 < len(result)


def test_a_report_with_no_caveats_prints_no_caveat_block():
    """The ``!`` lines are absent, while the notes and the headline remain."""
    spec = ControlChartSpec(value="signal", plate="plate", order="run",
                            baseline_n=20, rules=())
    text = control_chart(_quiet_campaign(), spec).report()

    assert "  ! " not in text
    assert "  · estimator chosen automatically" in text
    assert "all in control" in text


# ---------------------------------------------------------------------------
# The estimator the dispatcher has no branch for
# ---------------------------------------------------------------------------

def test_estimate_refuses_an_estimator_it_has_no_branch_for():
    """``_estimate`` names the estimator it could not run rather than falling
    through to ``None``.

    The spec validates against :data:`ESTIMATORS` and
    ``_resolve_estimator`` turns ``"auto"`` into a concrete choice, so nothing
    in the public path arrives here. The guard is what makes a name added to
    :data:`ESTIMATORS` without a matching branch fail loudly instead of
    returning ``None`` and raising a ``TypeError`` three frames later.
    """
    values = np.asarray([10.0, 12.0, 10.0, 12.0], dtype=float)
    sizes = np.ones(4)
    sds = np.full(4, np.nan)
    notes: list[str] = []

    with pytest.raises(ControlChartError) as excinfo:
        CC._estimate(values, sizes, sds, "trimmed_mean", notes)

    assert "trimmed_mean" in str(excinfo.value)
    assert notes == []


def test_estimate_still_runs_the_estimator_it_does_have_a_branch_for():
    """The same call with a real estimator returns the hand-computed pair.

    Guards the test above from passing because ``_estimate`` was renamed or
    its signature moved: MR-bar over ``10, 12, 10, 12`` is exactly 2 and the
    centre exactly 11.
    """
    values = np.asarray([10.0, 12.0, 10.0, 12.0], dtype=float)
    centre, sigma = CC._estimate(values, np.ones(4), np.full(4, np.nan),
                                 ESTIMATOR_MOVING_RANGE, [])

    assert centre == 11.0
    assert sigma == pytest.approx(2.0 / CC.D2_MOVING_RANGE)

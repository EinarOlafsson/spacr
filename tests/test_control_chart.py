"""Control charts: the limits by hand, then one planted series per rule.

Every assertion here is on a number worked out on paper next to the code or on
a behaviour a screener would notice. A control chart that lays out beautifully
and reports "in control" because sigma was taken from the standard deviation of
a drifting series is the failure worth catching, and it is the one that looks
fine.

The exact datasets
------------------

Three of them, all constructed rather than sampled, so there is no seed and no
tolerance worth arguing about.

``FLAT`` — ten plates alternating 10 and 12. Every moving range is exactly 2,
so ``MR-bar = 2`` exactly and ``sigma = 2 / 1.128`` exactly. The centre is
exactly 11. Nothing about it trips any of the eight rules, which is asserted
rather than assumed.

``SUBGROUPS`` — twelve plates of five control wells each, the wells at
``mean + (-2, -1, 0, 1, 2)``. The within-plate sample SD of those offsets is
``sqrt(10/4) = sqrt(2.5)`` exactly and identically on every plate, so
``S-bar = sqrt(2.5)`` exactly and ``sigma_within = sqrt(2.5) / c4(5)``. Plate
means alternate 100.5 and 99.5 over twelve plates, so the centre is exactly 100.

``CYCLE`` — the four values ``0, 3, -1, -2`` repeated. Its moving ranges are
``3, 4, 1, 2`` repeating, so over a whole number of cycles ``MR-bar`` is exactly
2.5 and ``sigma = 2.5 / 1.128``; its mean is exactly 0. It is also, by
construction, clean under all eight rules — never nine on a side, never six
trending, never fourteen alternating, never four of five beyond one sigma,
never fifteen inside one sigma — which is what makes it usable as a baseline to
plant a single rule violation on top of. That property is asserted first, in
:func:`test_the_baseline_fixture_is_clean_under_every_rule`, because every
planted-rule test below depends on it.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.control_chart import (
    ABOVE, BELOW, D2_MOVING_RANGE, DEFAULT_BASELINE, EITHER, ESTIMATOR_AUTO,
    ESTIMATOR_MAD, ESTIMATOR_MOVING_RANGE, ESTIMATOR_ROBUST,
    ESTIMATOR_SUBGROUP_S, MAD_SCALE, MEDIAN_MR_CONSTANT, MIN_BASELINE,
    RULES_ALL, RULES_DEFAULT, RULES_LIMITS_ONLY, RULE_ALARM_RATE,
    RULE_BEYOND_3_SIGMA, RULE_DETECTS, RULE_EIGHT_BEYOND_1,
    RULE_FIFTEEN_WITHIN_1, RULE_FOURTEEN_ALTERNATING,
    RULE_FOUR_OF_FIVE_BEYOND_1, RULE_NAMES, RULE_NINE_ONE_SIDE,
    RULE_SIX_TRENDING, RULE_TWO_OF_THREE_BEYOND_2, ControlChartError,
    ControlChartSpec, c4, candidate_key_columns, candidate_value_columns,
    control_chart, sd_reference_limits, zprime_chart, zprime_frame,
)


# ---------------------------------------------------------------------------
# The fixtures, and the arithmetic that goes with them
# ---------------------------------------------------------------------------

#: Ten plates alternating 10 and 12. Nine moving ranges, every one of them 2.
FLAT = [10.0, 12.0] * 5

#: MR-bar = (2+2+2+2+2+2+2+2+2) / 9 = 2, exactly.
FLAT_MR_BAR = 2.0
#: sigma = MR-bar / d2 = 2 / 1.128 = 1.7730496453900709
FLAT_SIGMA = FLAT_MR_BAR / 1.128
#: centre = (10+12+10+12+10+12+10+12+10+12) / 10 = 11, exactly.
FLAT_CENTRE = 11.0

#: The clean, deterministic baseline every planted rule is grown on: five
#: cycles of (0, 3, -1, -2) plus a final 0, so 21 plates. Moving ranges are
#: 3, 4, 1, 2 repeating -> 20 of them summing to 50 -> MR-bar = 2.5 exactly.
CYCLE = [0.0, 3.0, -1.0, -2.0]
BASELINE = CYCLE * 5 + [0.0]
#: sigma = 2.5 / 1.128 = 2.2163120567375887, and the mean is exactly 0.
S = 2.5 / 1.128


def series_frame(values, *, order=True, plate_prefix="P") -> pd.DataFrame:
    """One row per plate: a plate id, an integer run order, and the value."""
    values = [float(v) for v in values]
    payload = {"plateID": [f"{plate_prefix}{i + 1:02d}"
                           for i in range(len(values))],
               "signal": values}
    if order:
        payload["run_order"] = list(range(len(values)))
    return pd.DataFrame(payload)


def chart(values, *, rules=RULES_ALL, baseline_n=21, order=True, **kwargs):
    """Chart ``values`` as one plate per value, in the order given."""
    return control_chart(
        series_frame(values, order=order),
        ControlChartSpec(value="signal", plate="plateID",
                         order="run_order" if order else None,
                         rules=rules, baseline_n=baseline_n, **kwargs))


def planted(tail, rule):
    """``BASELINE`` followed by ``tail``, charted with only ``rule`` selected.

    One rule at a time on purpose. "Exactly rule k fired" is only a meaningful
    claim about a rule that was actually offered the chance, and a tight
    synthetic tail almost always trips rule 7 or rule 8 as well — which is a
    true statement about the fixture rather than about the rule under test.
    """
    return chart(list(BASELINE) + list(tail), rules=(rule,), baseline_n=21)


def subgroup_frame(plate_means, offsets=(-2.0, -1.0, 0.0, 1.0, 2.0),
                   ) -> pd.DataFrame:
    """One row per control well: ``len(offsets)`` wells on every plate.

    The sample SD of ``(-2, -1, 0, 1, 2)`` is ``sqrt((4+1+0+1+4)/4)`` =
    ``sqrt(2.5)``, identically on every plate, so S-bar is exact.
    """
    rows = []
    for index, mean in enumerate(plate_means):
        for offset in offsets:
            rows.append({"plateID": f"P{index + 1:02d}",
                         "run_order": index,
                         "well_type": "neg",
                         "signal": float(mean) + float(offset)})
    return pd.DataFrame(rows)


#: Twelve plates, means alternating 100.5 / 99.5 -> grand mean exactly 100.
SUBGROUP_MEANS = [100.5, 99.5] * 6


# ---------------------------------------------------------------------------
# The limits, by hand
# ---------------------------------------------------------------------------

def test_the_individuals_limits_are_the_hand_computed_ones():
    """MR-bar / d2, worked out on paper and asserted digit for digit.

    The series is 10, 12, 10, 12, 10, 12, 10, 12, 10, 12.
    The nine moving ranges are |12-10| = 2 each time:

        MR = 2, 2, 2, 2, 2, 2, 2, 2, 2
        MR-bar = 18 / 9 = 2
        sigma  = MR-bar / d2 = 2 / 1.128 = 1.7730496453900709
        centre = 110 / 10 = 11
        UCL    = 11 + 3 x 1.7730496453900709 = 16.319148936170215
        LCL    = 11 - 3 x 1.7730496453900709 =  5.680851063829786
    """
    result = chart(FLAT, baseline_n=10)

    assert result.estimator == ESTIMATOR_MOVING_RANGE
    assert result.centre == pytest.approx(11.0, abs=1e-12)
    assert result.sigma == pytest.approx(2.0 / 1.128, abs=1e-15)
    assert result.sigma == pytest.approx(1.7730496453900709, abs=1e-12)
    assert result.upper[0] == pytest.approx(16.319148936170215, abs=1e-12)
    assert result.lower[0] == pytest.approx(5.680851063829786, abs=1e-12)
    # Constant across the chart, because every plate has one control value.
    assert result.upper == pytest.approx(np.full(10, 16.319148936170215))
    assert result.lower == pytest.approx(np.full(10, 5.680851063829786))
    assert result.sigma_within == pytest.approx(result.sigma)
    # d2 is what it says it is, and it is 1.128 rather than 2/sqrt(pi).
    assert D2_MOVING_RANGE == 1.128
    assert not result.violations


def test_c4_from_the_gamma_function_matches_the_published_constants():
    """The real test of the gamma-function implementation.

    c4(n) = sqrt(2/(n-1)) * Gamma(n/2) / Gamma((n-1)/2). For n = 5 that is
    sqrt(1/2) * Gamma(2.5) / Gamma(2) = 0.7071067811865476 * 1.3293403881791370
    = 0.9399856..., and the table in the back of every control-chart text says
    0.9400.
    """
    assert c4(5) == pytest.approx(0.9400, abs=5e-5)
    assert c4(5) == pytest.approx(
        math.sqrt(0.5) * 1.3293403881791370, abs=1e-12)
    # The other published entries, to the four decimals they are printed to.
    assert c4(2) == pytest.approx(0.7979, abs=5e-5)
    assert c4(3) == pytest.approx(0.8862, abs=5e-5)
    assert c4(4) == pytest.approx(0.9213, abs=5e-5)
    assert c4(10) == pytest.approx(0.9727, abs=5e-5)
    assert c4(25) == pytest.approx(0.9896, abs=5e-5)
    # It is a correction towards 1 and it never overshoots.
    assert all(c4(n) < c4(n + 1) < 1.0 for n in range(2, 40))


def test_c4_refuses_a_subgroup_of_one_and_says_what_to_use_instead():
    with pytest.raises(ControlChartError, match="at least two observations"):
        c4(1)


def test_the_xbar_s_limits_are_the_hand_computed_ones():
    """S-bar / c4(n), worked out on paper.

    Twelve plates, five control wells each at ``mean + (-2, -1, 0, 1, 2)``:

        s_j    = sqrt((4+1+0+1+4)/4) = sqrt(2.5) = 1.5811388300841898, every j
        S-bar  = sqrt(2.5)
        c4(5)  = 0.9399856029866253
        sigma_within = sqrt(2.5) / c4(5) = 1.68208834801344
        the plotted point is the plate MEAN, so its sigma is
        sigma_within / sqrt(5) = sqrt(0.5) / c4(5) = 0.7522527780636751
        centre = (100.5 + 99.5) x 6 / 12 = 100
        UCL    = 100 + 3 x 0.7522527780636751 = 102.25675833419102
    """
    result = control_chart(
        subgroup_frame(SUBGROUP_MEANS),
        ControlChartSpec(value="signal", plate="plateID", order="run_order",
                         baseline_n=12, rules=RULES_ALL))

    assert result.estimator == ESTIMATOR_SUBGROUP_S
    assert result.subgroup_sizes == pytest.approx(np.full(12, 5.0))
    assert result.subgroup_sd == pytest.approx(np.full(12, math.sqrt(2.5)))
    assert result.centre == pytest.approx(100.0, abs=1e-12)
    assert result.sigma_within == pytest.approx(
        math.sqrt(2.5) / c4(5), abs=1e-12)
    assert result.sigma_within == pytest.approx(1.68208834801344, abs=1e-12)
    assert result.sigma == pytest.approx(math.sqrt(0.5) / c4(5), abs=1e-12)
    assert result.sigma == pytest.approx(0.7522527780636751, abs=1e-12)
    assert result.upper[0] == pytest.approx(102.25675833419102, abs=1e-12)
    assert result.lower[0] == pytest.approx(97.74324166580898, abs=1e-12)
    assert not result.violations
    assert "X-bar / S" in result.report()


def test_the_estimator_is_chosen_from_the_subgroup_size_and_said_out_loud():
    one_well = chart(FLAT, baseline_n=10)
    several = control_chart(
        subgroup_frame(SUBGROUP_MEANS),
        ControlChartSpec(value="signal", plate="plateID", order="run_order",
                         baseline_n=12))
    assert one_well.estimator == ESTIMATOR_MOVING_RANGE
    assert several.estimator == ESTIMATOR_SUBGROUP_S
    assert any("chosen automatically" in note for note in one_well.notes)
    assert any("chosen automatically" in note for note in several.notes)
    # And neither result ever reports the placeholder.
    assert ESTIMATOR_AUTO not in (one_well.estimator, several.estimator)


def test_asking_for_xbar_s_without_subgroups_is_refused_with_the_way_out():
    with pytest.raises(ControlChartError, match="two or more control wells"):
        chart(FLAT, baseline_n=10, estimator=ESTIMATOR_SUBGROUP_S)


def test_varying_subgroup_sizes_give_per_plate_limits_and_a_caveat():
    """A plate with fewer control wells has a noisier mean and wider limits."""
    frame = subgroup_frame(SUBGROUP_MEANS)
    thinned = frame.drop(
        frame.index[(frame["plateID"] == "P03")][:3]).reset_index(drop=True)
    result = control_chart(
        thinned, ControlChartSpec(value="signal", plate="plateID",
                                  order="run_order", baseline_n=12))
    third = result.plates.index("P03")
    assert result.subgroup_sizes[third] == 2
    assert result.upper[third] > result.upper[0]
    assert "Subgroup sizes vary" in " ".join(result.caveats())


# ---------------------------------------------------------------------------
# The point of the whole design
# ---------------------------------------------------------------------------

def test_moving_range_limits_catch_a_drift_that_sd_limits_swallow():
    """The asymmetry the module exists for, pinned.

    Thirty plates drifting by exactly 0.5 per plate, from 10.0 to 24.5, with a
    twenty-plate baseline.

        every moving range is 0.5, so MR-bar = 0.5
        sigma  = 0.5 / 1.128 = 0.44326241134751776
        centre = mean(10.0 .. 19.5) = 10 + 0.5 x 9.5 = 14.75
        limits = 14.75 +/- 1.3297872340425533 = [13.4202..., 16.0798...]

    against the SD route over the same thirty numbers:

        mean = 17.25, SD (ddof=1) = 4.401704215414752
        limits = [4.044887353755744, 30.455112646244256]

    Every one of the thirty values lies inside that, so the SD chart reports a
    campaign in control while the control slides from 10 to 24.5. If this
    asymmetry ever stops holding, the reason for the design has gone with it.
    """
    values = [10.0 + 0.5 * i for i in range(30)]
    result = chart(values, rules=RULES_LIMITS_ONLY, baseline_n=20)

    assert result.centre == pytest.approx(14.75, abs=1e-12)
    assert result.sigma == pytest.approx(0.5 / 1.128, abs=1e-15)
    assert result.lower[0] == pytest.approx(13.420212765957446, abs=1e-12)
    assert result.upper[0] == pytest.approx(16.079787234042552, abs=1e-12)

    # The moving-range chart sees it.
    assert int(result.flagged.sum()) == 24
    assert result.rules_at(len(result) - 1) == (RULE_BEYOND_3_SIGMA,)

    # The SD route does not see it at all.
    centre, sigma, low, high = sd_reference_limits(values)
    assert (centre, sigma) == pytest.approx((17.25, 4.401704215414752))
    assert (low, high) == pytest.approx(
        (4.044887353755744, 30.455112646244256))
    assert result.sd_reference == pytest.approx((centre, sigma, low, high))
    assert result.sd_would_flag == 0
    assert min(values) > low and max(values) < high

    # And the result says so, in the caveats, with both intervals in it.
    caveats = " ".join(result.caveats())
    assert "standard deviation of the whole series" in caveats
    assert "0 point(s) outside" in caveats


def test_the_sd_reference_refuses_to_invent_a_spread_from_one_point():
    assert sd_reference_limits([]) == (0.0, 0.0, 0.0, 0.0)
    assert sd_reference_limits([4.0]) == (4.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# One planted series per rule
# ---------------------------------------------------------------------------

def test_the_baseline_fixture_is_clean_under_every_rule():
    """Every planted test below rests on this, so it is asserted first."""
    result = chart(BASELINE, rules=RULES_ALL, baseline_n=21)
    assert result.centre == pytest.approx(0.0, abs=1e-12)
    assert result.sigma == pytest.approx(2.5 / 1.128, abs=1e-15)
    assert result.violations == ()
    # And it stays clean however long the cycle runs, so a planted violation
    # in the tail is the only thing a longer series can be reporting.
    longer = chart(BASELINE + CYCLE * 5, rules=RULES_ALL, baseline_n=21)
    assert longer.violations == ()
    assert longer.sigma == pytest.approx(result.sigma)


def test_rule_1_fires_on_exactly_the_planted_plate():
    """One point beyond 3 sigma: plate 25 is at 3.5 sigma and nothing else is."""
    result = planted([3.0, -1.0, -2.0, 3.5 * S, 0.0, -1.0, -2.0],
                     RULE_BEYOND_3_SIGMA)
    assert len(result.violations) == 1
    violation = result.violations[0]
    assert violation.rule == RULE_BEYOND_3_SIGMA
    assert violation.points == (24,)
    assert violation.plates == ("P25",)
    assert violation.side == ABOVE
    assert not violation.in_baseline
    assert result.rules_at(24) == (RULE_BEYOND_3_SIGMA,)
    assert result.z[24] == pytest.approx(3.5)
    assert result.zone(24) == 3
    assert "plate P25 is beyond the upper limit" in violation.describe()


def test_rule_1_flags_the_low_side_too():
    result = planted([3.0, -1.0, -2.0, -3.5 * S, 0.0], RULE_BEYOND_3_SIGMA)
    assert [v.side for v in result.violations] == [BELOW]
    assert "beyond the lower limit" in result.violations[0].describe()


def test_rule_2_fires_on_exactly_the_nine_planted_plates():
    """Nine in a row on one side — the shift that costs a campaign.

    The run is nine points at +0.5 sigma, indices 24..32 (plates P25..P33).
    The plate before it is at -2 and the plate after it at -1, so the run is
    exactly nine and the maximal span is exactly the planted one.
    """
    result = planted([3.0, -1.0, -2.0] + [0.5 * S] * 9 + [-1.0, -2.0],
                     RULE_NINE_ONE_SIDE)
    assert len(result.violations) == 1
    violation = result.violations[0]
    assert violation.rule == RULE_NINE_ONE_SIDE
    assert violation.points == tuple(range(24, 33))
    assert violation.plates == tuple(f"P{i + 1:02d}" for i in range(24, 33))
    assert violation.side == ABOVE
    assert violation.span == 9
    # Reported once as a span, not as four overlapping nine-point windows.
    assert result.rules_at(28) == (RULE_NINE_ONE_SIDE,)
    assert not result.flagged[23] and not result.flagged[33]
    # Named by number AND in words, with the span and what it means.
    text = result.report()
    assert "rule 2 — nine points in a row on the same side" in text
    assert "plates P25–P33" in text
    assert "a shift, not a spike" in text


def test_rule_3_fires_on_exactly_the_six_rising_plates():
    """Six points in a row steadily increasing: -2.5 to 0.0 in five steps."""
    result = planted(
        [3.0, -1.0, -2.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, -1.0, -2.0],
        RULE_SIX_TRENDING)
    assert len(result.violations) == 1
    violation = result.violations[0]
    assert violation.rule == RULE_SIX_TRENDING
    assert violation.points == tuple(range(24, 30))
    assert violation.span == 6
    assert violation.side == ABOVE          # rising
    assert "moving steadily one way" in violation.describe()


def test_a_repeated_value_breaks_a_trend_because_equal_is_neither_way():
    """Five rises and a flat spot is not six points steadily increasing."""
    result = planted(
        [3.0, -1.0, -2.0, -2.5, -2.0, -1.5, -1.0, -1.0, -0.5, 0.0],
        RULE_SIX_TRENDING)
    assert result.violations == ()


def test_rule_4_fires_on_exactly_the_fourteen_alternating_plates():
    """Fourteen alternating: -3, -4, -3, -4, ... seven times over.

    The step into the run (0 -> -3) is downward and the first step inside it
    (-3 -> -4) is downward too, so there is no turn at the join and the run is
    exactly the fourteen planted plates.
    """
    result = planted([-3.0, -4.0] * 7, RULE_FOURTEEN_ALTERNATING)
    assert len(result.violations) == 1
    violation = result.violations[0]
    assert violation.rule == RULE_FOURTEEN_ALTERNATING
    assert violation.points == tuple(range(21, 35))
    assert violation.span == 14
    assert violation.side == EITHER
    assert "alternate up and down" in violation.describe()


def test_rule_5_flags_the_two_offending_plates_and_not_the_one_between():
    """Two of three beyond 2 sigma: plates 25 and 27, with 26 on the centre.

    Flagging plate 26 would be an accusation against a plate that sat exactly
    on the centre line, which is the reason the k-of-m rules report the
    qualifying points rather than the whole window.
    """
    result = planted([3.0, -1.0, -2.0, 2.5 * S, 0.0, 2.5 * S, -1.0, -2.0],
                     RULE_TWO_OF_THREE_BEYOND_2)
    assert len(result.violations) == 1
    violation = result.violations[0]
    assert violation.rule == RULE_TWO_OF_THREE_BEYOND_2
    assert violation.points == (24, 26)
    assert violation.plates == ("P25", "P27")
    assert violation.side == ABOVE
    assert result.rules_at(25) == ()
    assert not result.flagged[25]


def test_rule_6_flags_the_four_offending_plates_of_the_five():
    """Four of five beyond 1 sigma: 25, 26, 28, 29 with 27 on the centre."""
    result = planted(
        [3.0, -1.0, -2.0, 1.5 * S, 1.5 * S, 0.0, 1.5 * S, 1.5 * S, -1.0, -2.0],
        RULE_FOUR_OF_FIVE_BEYOND_1)
    assert len(result.violations) == 1
    violation = result.violations[0]
    assert violation.rule == RULE_FOUR_OF_FIVE_BEYOND_1
    assert violation.points == (24, 25, 27, 28)
    assert violation.side == ABOVE
    assert not result.flagged[26]


def test_rule_7_fires_on_a_series_that_is_too_good():
    """Fifteen in a row inside 1 sigma — the sigma is probably wrong."""
    result = planted([3.0] + [0.0] * 15, RULE_FIFTEEN_WITHIN_1)
    assert len(result.violations) == 1
    violation = result.violations[0]
    assert violation.rule == RULE_FIFTEEN_WITHIN_1
    assert violation.points == tuple(range(22, 37))
    assert violation.span == 15
    assert "too good" in violation.describe()


def test_rule_8_fires_on_eight_in_a_row_outside_one_sigma():
    result = planted([1.5 * S] * 8 + [0.0, 0.0], RULE_EIGHT_BEYOND_1)
    assert len(result.violations) == 1
    violation = result.violations[0]
    assert violation.rule == RULE_EIGHT_BEYOND_1
    assert violation.points == tuple(range(21, 29))
    assert violation.span == 8
    assert "a mixture of two populations" in violation.describe()


@pytest.mark.parametrize("rule", RULES_ALL)
def test_no_rule_fires_on_a_clean_series(rule):
    """The other half of every planted test: silence on data with nothing in it."""
    assert chart(BASELINE + CYCLE * 6, rules=(rule,),
                 baseline_n=21).violations == ()


def test_a_plate_can_trip_several_rules_at_once_and_all_are_reported():
    """A plate beyond 3 sigma is usually also the end of a 2-of-3 window."""
    result = planted([3.0, -1.0, -2.0, 2.5 * S, 3.5 * S, -1.0, -2.0],
                     RULE_BEYOND_3_SIGMA)
    both = chart(list(BASELINE) + [3.0, -1.0, -2.0, 2.5 * S, 3.5 * S,
                                   -1.0, -2.0],
                 rules=(RULE_BEYOND_3_SIGMA, RULE_TWO_OF_THREE_BEYOND_2),
                 baseline_n=21)
    assert result.rules_at(25) == (RULE_BEYOND_3_SIGMA,)
    assert both.rules_at(25) == (RULE_BEYOND_3_SIGMA,
                                 RULE_TWO_OF_THREE_BEYOND_2)


def test_asking_about_a_point_that_is_not_on_the_chart_is_refused():
    result = chart(BASELINE, baseline_n=21)
    with pytest.raises(ControlChartError, match="there is no point"):
        result.rules_at(999)
    with pytest.raises(ControlChartError, match="there is no point"):
        result.zone(-1)


# ---------------------------------------------------------------------------
# Phase I / Phase II
# ---------------------------------------------------------------------------

def test_the_limits_come_from_the_baseline_and_nothing_after_it():
    """Whatever happens in Phase II must not move the line it is judged by."""
    quiet = chart(BASELINE + [0.0] * 10, baseline_n=21)
    wild = chart(BASELINE + [500.0] * 10, baseline_n=21,
                 rules=RULES_LIMITS_ONLY)
    assert quiet.centre == wild.centre == pytest.approx(0.0, abs=1e-12)
    assert quiet.sigma == wild.sigma == pytest.approx(2.5 / 1.128, abs=1e-15)
    assert wild.baseline_plates == tuple(f"P{i + 1:02d}" for i in range(21))
    # And the excursion is caught, which is what the forward application is for.
    assert int(wild.flagged.sum()) == 10
    assert all(v.start >= 21 for v in wild.violations)


def test_including_the_drift_in_the_baseline_moves_the_centre_off_the_process():
    """The mistake the baseline parameter exists to prevent, shown as numbers.

    Twenty-one stable plates at a level of 0, then twenty drifting up by 1.5 a
    plate to 30. Estimated from Phase I the centre is exactly 0 — the level the
    process actually held — and only the drifted plates are flagged. Estimated
    from everything the centre is ``315 / 41 = 7.68``, a level no plate was ever
    stable at, and the *stable* plates at the start of the campaign come out as
    the ones below the limit. The chart then blames the only part of the
    campaign that was working.
    """
    values = BASELINE + [1.5 * (i + 1) for i in range(20)]
    honest = chart(values, baseline_n=21, rules=RULES_LIMITS_ONLY)
    everything = chart(values, baseline_n=41, rules=RULES_LIMITS_ONLY)

    assert honest.centre == pytest.approx(0.0, abs=1e-12)
    assert everything.centre == pytest.approx(315.0 / 41)
    assert honest.violations and all(v.start >= 21 for v in honest.violations)
    assert any(v.start < 21 for v in everything.violations)
    assert any("all 41 plates are Phase I" in note
               for note in everything.notes)
    assert "cannot be out of limits that it helped set" in " ".join(
        everything.caveats())
    assert "helped set" not in " ".join(honest.caveats())


def test_an_explicit_list_of_baseline_plates_is_honoured():
    result = chart(BASELINE + CYCLE * 3, baseline_n=21,
                   baseline_plates=tuple(f"P{i + 1:02d}" for i in range(1, 21)))
    assert result.baseline_plates == tuple(
        f"P{i + 1:02d}" for i in range(1, 21))
    assert len(result.baseline_plates) == 20


def test_a_baseline_cut_off_on_a_date_column_is_honoured():
    values = BASELINE + CYCLE * 2
    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(
        np.arange(len(values)), unit="D")
    frame = pd.DataFrame({"plateID": [f"P{i + 1:02d}"
                                      for i in range(len(values))],
                          "run_date": dates, "signal": values})
    result = control_chart(frame, ControlChartSpec(
        value="signal", plate="plateID", order="run_date",
        baseline_before="2026-01-22"))
    assert len(result.baseline_plates) == 21        # 1 Jan .. 21 Jan
    assert result.baseline_plates[-1] == "P21"
    assert result.centre == pytest.approx(0.0, abs=1e-12)


def test_a_baseline_cut_off_before_everything_is_refused():
    values = BASELINE + CYCLE
    frame = series_frame(values)
    with pytest.raises(ControlChartError, match="Phase I would be empty"):
        control_chart(frame, ControlChartSpec(
            value="signal", plate="plateID", order="run_order",
            baseline_before=-5))


def test_a_violation_inside_the_baseline_is_reported_as_such():
    """Limits estimated from an out-of-control baseline are not limits."""
    values = list(BASELINE)
    values[12] = 30.0
    result = chart(values + CYCLE * 3, baseline_n=21, rules=RULES_LIMITS_ONLY)

    assert [v.plates for v in result.violations] == [("P13",)]
    assert result.violations[0].in_baseline
    assert result.baseline_violations == result.violations
    assert "(inside the baseline)" in result.violations[0].describe()
    caveats = " ".join(result.caveats())
    assert "baseline itself is out of control" in caveats
    assert "reestimate=True" in caveats


def test_re_estimating_drops_the_flagged_plates_and_says_it_happened():
    values = list(BASELINE)
    values[12] = 30.0
    kept = chart(values + CYCLE * 3, baseline_n=21, rules=RULES_LIMITS_ONLY)
    redone = chart(values + CYCLE * 3, baseline_n=21,
                   rules=RULES_LIMITS_ONLY, reestimate=True)

    assert kept.baseline_excluded == ()
    assert redone.baseline_excluded == ("P13",)
    assert "P13" not in redone.baseline_plates
    assert redone.sigma < kept.sigma / 1.5
    assert redone.centre == pytest.approx(0.0, abs=1e-12)
    assert any("re-estimated" in note for note in redone.notes)
    assert "One pass only" in " ".join(redone.caveats())


def test_re_estimation_refuses_to_shrink_the_baseline_below_the_minimum():
    """Dropping the inconvenient plates has a floor, and it says when it hit it."""
    values = list(BASELINE[:MIN_BASELINE])
    values[3] = 200.0
    result = chart(values + CYCLE * 4, baseline_n=MIN_BASELINE,
                   rules=RULES_LIMITS_ONLY, reestimate=True)
    assert result.baseline_violations              # there is one to drop
    assert result.baseline_excluded == ()          # and it was not dropped
    assert len(result.baseline_plates) == MIN_BASELINE
    assert any("under the 8-plate minimum" in note for note in result.notes)


def test_the_report_names_the_baseline_it_used():
    result = chart(BASELINE + CYCLE * 2, baseline_n=21)
    text = result.report()
    assert "Baseline (Phase I): 21 plate(s), P01–P21" in text
    assert "applied forward" in text


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def test_shuffling_the_rows_changes_nothing_when_an_order_column_is_given():
    """The x axis is run order, so the row order of the table is irrelevant."""
    frame = series_frame(BASELINE + CYCLE * 3)
    shuffled = frame.iloc[
        np.random.default_rng(19).permutation(len(frame))].reset_index(
            drop=True)
    spec = ControlChartSpec(value="signal", plate="plateID",
                            order="run_order", baseline_n=21, rules=RULES_ALL)
    straight = control_chart(frame, spec)
    scrambled = control_chart(shuffled, spec)

    assert straight.plates == scrambled.plates
    assert straight.values == pytest.approx(scrambled.values)
    assert straight.centre == pytest.approx(scrambled.centre)
    assert straight.sigma == pytest.approx(scrambled.sigma)
    assert straight.baseline_plates == scrambled.baseline_plates
    assert straight.violations == scrambled.violations
    assert not straight.order_inferred


def test_an_inferred_order_is_said_loudly_in_every_place_it_matters():
    result = chart(BASELINE + CYCLE * 2, baseline_n=21, order=False)
    assert result.order_inferred
    assert result.order_column is None
    assert "INFERRED" in result.report()
    assert "INFERRED" in " ".join(result.caveats())
    assert "statements about a sequence" in " ".join(result.caveats())


def test_the_inferred_order_puts_plate_2_before_plate_10():
    """A plain string sort puts P10 before P2 and every run rule then rests on
    the wrong sequence."""
    frame = pd.DataFrame({"plateID": [f"P{i}" for i in range(1, 13)],
                          "signal": CYCLE * 3})
    result = control_chart(frame, ControlChartSpec(
        value="signal", plate="plateID", baseline_n=12))
    assert result.plates == tuple(f"P{i}" for i in range(1, 13))
    assert result.order_inferred


def test_an_explicit_order_column_beats_the_plate_id():
    """The plates were run backwards; the chart must follow the run, not the id."""
    values = BASELINE + CYCLE
    frame = pd.DataFrame({
        "plateID": [f"P{i + 1:02d}" for i in range(len(values))],
        "run_order": list(range(len(values)))[::-1],
        "signal": values})
    result = control_chart(frame, ControlChartSpec(
        value="signal", plate="plateID", order="run_order", baseline_n=21))
    assert result.plates[0] == f"P{len(values):02d}"
    assert not result.order_inferred


# ---------------------------------------------------------------------------
# Robust estimators
# ---------------------------------------------------------------------------

def test_one_catastrophic_plate_in_the_baseline_wrecks_the_classical_limits():
    """Which is what the robust variants are for, and it is a large effect."""
    values = list(BASELINE)
    values[5] = 60.0
    spec = ControlChartSpec(value="signal", plate="plateID",
                            order="run_order", baseline_n=21)
    frame = series_frame(values)

    classical = control_chart(frame, spec)
    robust = control_chart(frame, spec.with_estimator(ESTIMATOR_ROBUST))
    mad = control_chart(frame, spec.with_estimator(ESTIMATOR_MAD))

    assert classical.sigma > 2 * robust.sigma
    assert classical.sigma > 4 * mad.sigma
    # The centres, hand-checkable: the median of the cycle is 0 either way.
    assert robust.centre == pytest.approx(0.0, abs=1e-12)
    assert mad.centre == pytest.approx(0.0, abs=1e-12)
    # median(MR) over the clean cycle is 2.5 (five each of 1, 2, 3, 4), and
    # the outlier moves two of the twenty ranges without moving the median.
    assert robust.sigma == pytest.approx(2.5 / MEDIAN_MR_CONSTANT, abs=1e-12)
    # median|x - 0| over the 21 baseline values is exactly 1.
    assert mad.sigma == pytest.approx(MAD_SCALE * 1.0, abs=1e-12)
    assert "robust" in robust.report()
    assert "1.4826" in mad.report()


def test_the_two_robust_constants_are_not_interchangeable():
    """1.4826 is the MAD constant; a moving range needs sqrt(2) x 0.6745.

    Using 1.4826 on a median moving range overestimates sigma by exactly
    sqrt(2) — 41% wider limits, which is the difference between catching a
    drift and not.
    """
    assert MAD_SCALE == pytest.approx(1.4826022185056, abs=1e-12)
    assert MEDIAN_MR_CONSTANT == pytest.approx(0.9538725524089, abs=1e-12)
    assert MAD_SCALE * MEDIAN_MR_CONSTANT == pytest.approx(math.sqrt(2.0))


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------

def test_a_campaign_shorter_than_the_minimum_baseline_is_refused():
    with pytest.raises(ControlChartError,
                       match="smallest defensible baseline is 8"):
        chart([1.0, 2.0, 3.0, 4.0, 5.0], baseline_n=MIN_BASELINE)


def test_a_baseline_selecting_too_few_plates_is_refused():
    with pytest.raises(ControlChartError, match="is the minimum"):
        chart(BASELINE, baseline_n=21,
              baseline_plates=("P01", "P02", "P03"))


def test_a_constant_series_collapses_deliberately_rather_than_flagging_everything():
    """Sigma zero means the limits sit on the centre line, which would put all
    two hundred plates 'beyond' them. That is arithmetic, not a QC finding."""
    result = chart([7.0] * 40, baseline_n=20, rules=RULES_ALL)
    assert result.degenerate
    assert result.sigma == 0.0
    assert result.violations == ()
    assert not result.flagged.any()
    assert result.upper == pytest.approx(np.full(40, 7.0))
    assert result.lower == pytest.approx(np.full(40, 7.0))
    assert "reported the same signal" in result.headline()
    caveats = " ".join(result.caveats())
    assert "Sigma came out zero" in caveats
    assert "wrong column is charted" in caveats
    assert any("no rule was run" in note for note in result.notes)


@pytest.mark.parametrize("kwargs, message", [
    ({"rules": (9,)}, "there is no rule 9"),
    ({"rules": ("two",)}, "not a rule number"),
    ({"estimator": "stdev"}, "unknown estimator"),
    ({"baseline_n": 3}, "not a baseline"),
    ({"control_column": "well_type"}, "which level it is"),
    ({"control_levels": ("neg",)}, "no column to find it in"),
])
def test_a_meaningless_spec_is_refused_where_it_is_built(kwargs, message):
    with pytest.raises(ControlChartError, match=message):
        ControlChartSpec(value="signal", plate="plateID", **kwargs)


@pytest.mark.parametrize("kwargs, message", [
    ({"value": ""}, "no value column chosen"),
    ({"plate": ""}, "no plate column chosen"),
    ({"value": "absent"}, "value column 'absent' is not in this table"),
    ({"plate": "absent"}, "plate column 'absent' is not in this table"),
    ({"order": "absent"}, "order column 'absent' is not in this table"),
    ({"control_column": "absent", "control_levels": ("neg",)},
     "control column 'absent' is not in this table"),
])
def test_a_missing_column_is_refused_by_name(kwargs, message):
    spec_kwargs = {"value": "signal", "plate": "plateID",
                   "order": "run_order", "baseline_n": 21}
    spec_kwargs.update(kwargs)
    with pytest.raises(ControlChartError, match=message):
        control_chart(series_frame(BASELINE),
                      ControlChartSpec(**spec_kwargs))


def test_a_control_level_that_matches_nothing_lists_what_is_there():
    frame = subgroup_frame(SUBGROUP_MEANS)
    with pytest.raises(ControlChartError, match="no row has well_type"):
        control_chart(frame, ControlChartSpec(
            value="signal", plate="plateID", order="run_order",
            control_column="well_type", control_levels=("positive",),
            baseline_n=12))


def test_a_column_with_no_numbers_in_it_is_refused():
    frame = series_frame(BASELINE)
    frame["signal"] = "not a number"
    with pytest.raises(ControlChartError, match="no plate has a finite"):
        control_chart(frame, ControlChartSpec(
            value="signal", plate="plateID", order="run_order",
            baseline_n=21))


def test_plates_with_no_finite_value_are_left_off_and_counted():
    values = list(BASELINE) + CYCLE * 2
    frame = series_frame(values)
    frame.loc[frame.index[3], "signal"] = np.nan
    result = control_chart(frame, ControlChartSpec(
        value="signal", plate="plateID", order="run_order", baseline_n=20))
    assert "P04" not in result.plates
    assert len(result) == len(values) - 1
    assert any("not on the chart" in note for note in result.notes)


def test_a_named_baseline_plate_that_is_not_in_the_table_is_noted_not_fatal():
    result = chart(BASELINE, baseline_n=21,
                   baseline_plates=tuple(f"P{i + 1:02d}"
                                         for i in range(20)) + ("P99",))
    assert "P99" not in result.baseline_plates
    assert any("not in this table" in note for note in result.notes)


# ---------------------------------------------------------------------------
# Saying it out loud
# ---------------------------------------------------------------------------

def test_the_false_alarm_arithmetic_is_in_the_caveats():
    """Eight rules on two hundred plates is not eight times the sensitivity."""
    long_run = chart(BASELINE + CYCLE * 45, rules=RULES_ALL, baseline_n=21)
    assert len(long_run) == 201
    rate = long_run.false_alarm_rate()
    expected = 1.0 - np.prod([1.0 - RULE_ALARM_RATE[r] for r in RULES_ALL])
    assert rate == pytest.approx(expected)
    caveats = " ".join(long_run.caveats())
    assert "8 rule(s) on 201 plates" in caveats
    assert f"about one in {1 / rate:.0f}" in caveats
    assert f"roughly {rate * 201:.1f} false alarm(s)" in caveats
    assert "91.75" in caveats
    # Fewer rules, fewer alarms — the reason the set is selectable.
    quiet = chart(BASELINE + CYCLE * 45, rules=RULES_LIMITS_ONLY,
                  baseline_n=21)
    assert quiet.false_alarm_rate() < rate / 4


def test_a_rule_that_cannot_fire_on_a_campaign_this_short_says_so():
    """Fifteen-point rules on a twenty-one-plate chart can fire; on a nine
    plate one they cannot, and their silence is not evidence."""
    result = chart(CYCLE * 2 + [0.0], rules=RULES_ALL, baseline_n=MIN_BASELINE)
    assert len(result) == 9
    caveats = " ".join(result.caveats())
    assert "need more points than this campaign has" in caveats
    assert "Their silence is not evidence of control" in caveats


def test_the_report_reads_as_prose_and_carries_every_decision():
    values = list(BASELINE) + [3.0, -1.0, -2.0] + [0.5 * S] * 9
    result = chart(values, rules=RULES_DEFAULT, baseline_n=21)
    text = result.report()
    assert text.startswith("Control chart of signal for every row")
    assert "individuals / moving range — sigma = MR-bar / 1.128" in text
    assert "Centre 0; sigma 2.21631" in text
    assert "Baseline (Phase I): 21 plate(s)" in text
    assert "Run order: the run_order column." in text
    assert "rule 2 — nine points in a row" in text
    assert result.headline() in text


def test_the_headline_says_it_is_clean_when_it_is():
    result = chart(BASELINE + CYCLE * 2, baseline_n=21)
    assert "all in control" in result.headline()
    assert "21 baseline plates" in result.headline()


def test_every_rule_carries_a_name_a_failure_mode_and_an_alarm_rate():
    for rule in RULES_ALL:
        assert RULE_NAMES[rule] and RULE_DETECTS[rule]
        assert 0.0 < RULE_ALARM_RATE[rule] < 0.01


# ---------------------------------------------------------------------------
# The frames and the spec
# ---------------------------------------------------------------------------

def test_the_points_frame_is_the_chart_as_a_table():
    values = list(BASELINE) + [3.0, -1.0, -2.0, 3.5 * S]
    result = chart(values, rules=RULES_LIMITS_ONLY, baseline_n=21)
    frame = result.points_frame()

    assert len(frame) == len(result)
    assert list(frame["plate"]) == list(result.plates)
    assert frame["value"].to_numpy() == pytest.approx(result.values)
    assert bool(frame.loc[0, "in_baseline"])
    assert not bool(frame.loc[len(frame) - 1, "in_baseline"])
    assert frame.loc[len(frame) - 1, "rules"] == "1"
    assert frame.loc[0, "rules"] == ""


def test_the_violations_frame_is_the_table_the_screen_shows():
    values = list(BASELINE) + [3.0, -1.0, -2.0] + [0.5 * S] * 9
    result = chart(values, rules=RULES_DEFAULT, baseline_n=21)
    frame = result.violations_frame()
    assert set(frame["rule"]) == {RULE_NINE_ONE_SIDE}
    assert frame.loc[0, "name"] == RULE_NAMES[RULE_NINE_ONE_SIDE]
    assert frame.loc[0, "n_points"] == 9
    assert frame.loc[0, "first_plate"] == "P25"
    assert not bool(frame.loc[0, "in_baseline"])


def test_the_spec_round_trips_through_json_exactly():
    spec = ControlChartSpec(
        value="signal", plate="plateID", order="run_date",
        control_column="well_type", control_levels=("neg", "neg"),
        positive_levels=("pos",), negative_levels=("neg",),
        estimator=ESTIMATOR_ROBUST, rules=(3, 1, 1), baseline_n=12,
        baseline_plates=("P01",), baseline_before="2026-02-01",
        reestimate=True)
    assert ControlChartSpec.from_json(spec.to_json()) == spec
    assert spec.control_levels == ("neg",)      # de-duplicated
    assert spec.rules == (1, 3)                 # sorted, de-duplicated
    assert json.loads(spec.to_json())["rules"] == [1, 3]


def test_a_spec_from_another_build_still_opens():
    spec = ControlChartSpec.from_dict(
        {"value": "signal", "plate": "plateID", "future_option": 12})
    assert spec.value == "signal"
    assert spec.baseline_n == DEFAULT_BASELINE
    assert spec.rules == RULES_DEFAULT


def test_the_spec_edits_return_new_specs():
    spec = ControlChartSpec(value="signal", plate="plateID")
    assert spec.with_rules([2]).rules == (2,)
    assert spec.with_estimator(ESTIMATOR_MAD).estimator == ESTIMATOR_MAD
    assert spec.with_baseline(n=9).baseline_n == 9
    assert spec.with_baseline(before="x").baseline_before == "x"
    assert spec.with_columns(order="run_order").order == "run_order"
    assert spec.with_control("well_type", ["neg"]).control_levels == ("neg",)
    assert spec.rules == RULES_DEFAULT          # the original is untouched


def test_the_spec_describes_itself_for_a_caption():
    spec = ControlChartSpec(value="signal", plate="plateID",
                            control_column="well_type",
                            control_levels=("neg",))
    assert "signal of well_type=neg" in spec.describe()
    assert "order inferred from the plate id" in spec.describe()
    assert f"baseline: first {DEFAULT_BASELINE}" in spec.describe()
    assert ControlChartSpec(baseline_plates=("P1",)).describe().count(
        "1 named plate") == 1


def test_the_column_offers_reuse_the_one_column_classifier():
    frame = subgroup_frame(SUBGROUP_MEANS)
    frame["cell_area"] = np.linspace(500.0, 1500.0, len(frame))
    assert "cell_area" in candidate_value_columns(frame)
    assert "plateID" not in candidate_value_columns(frame)
    # Keys are what the plate and control pickers want, and a plate id is a
    # key: this is the one screen where "identifies rather than describes" is
    # a recommendation rather than a disqualification.
    assert "plateID" in candidate_key_columns(frame)
    assert "well_type" in candidate_key_columns(frame)


# ---------------------------------------------------------------------------
# Z-prime
# ---------------------------------------------------------------------------

def zprime_campaign(plates: int = 20, seed: int = 5) -> pd.DataFrame:
    """A campaign whose assay window closes: the controls' spread grows."""
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(plates):
        spread = 3.0 + 2.2 * max(0, index - 11)
        for _ in range(4):
            rows.append({"plateID": f"P{index + 1:02d}", "day": index,
                         "well_type": "pos",
                         "signal": 100.0 + rng.normal(0.0, spread)})
            rows.append({"plateID": f"P{index + 1:02d}", "day": index,
                         "well_type": "neg",
                         "signal": 20.0 + rng.normal(0.0, spread)})
    return pd.DataFrame(rows)


def zprime_spec(**kwargs) -> ControlChartSpec:
    base = dict(value="signal", plate="plateID", order="day",
                control_column="well_type", control_levels=("pos",),
                positive_levels=("pos",), negative_levels=("neg",),
                baseline_n=10)
    base.update(kwargs)
    return ControlChartSpec(**base)


def test_zprime_is_the_textbook_formula_worked_out_per_plate():
    """Z' = 1 - 3(sd_pos + sd_neg) / |mean_pos - mean_neg|, by hand on plate 1."""
    frame = zprime_campaign()
    computed = zprime_frame(frame, zprime_spec())
    assert len(computed) == 20
    assert list(computed["order_index"]) == list(range(20))

    plate = frame[frame["plateID"] == "P01"]
    positive = plate.loc[plate["well_type"] == "pos", "signal"].to_numpy()
    negative = plate.loc[plate["well_type"] == "neg", "signal"].to_numpy()
    expected = 1.0 - 3.0 * (positive.std(ddof=1) + negative.std(ddof=1)) / abs(
        positive.mean() - negative.mean())
    row = computed[computed["plate"] == "P01"].iloc[0]
    assert row["zprime"] == pytest.approx(expected)
    assert row["n_positive"] == 4 and row["n_negative"] == 4


def test_the_zprime_chart_catches_a_closing_assay_window():
    result = zprime_chart(zprime_campaign(), zprime_spec(rules=RULES_ALL))
    assert result.value_column == "zprime"
    assert result.estimator == ESTIMATOR_MOVING_RANGE
    # The order was resolved once, upstream, so this chart never guesses it.
    assert not result.order_inferred
    assert result.order_column == "order_index"
    flagged = [p for p, f in zip(result.plates, result.flagged) if f]
    assert flagged, "a Z' that halves over a campaign must be caught"
    assert all(int(p[1:]) > 10 for p in flagged)


def test_zprime_needs_both_controls_named():
    frame = zprime_campaign()
    with pytest.raises(ControlChartError, match="both controls named"):
        zprime_frame(frame, zprime_spec(positive_levels=()))
    with pytest.raises(ControlChartError, match="no assay window"):
        zprime_frame(frame, zprime_spec(negative_levels=()))


def test_zprime_refuses_a_campaign_with_one_well_per_control():
    frame = zprime_campaign()
    thin = frame.drop_duplicates(subset=["plateID", "well_type"])
    with pytest.raises(ControlChartError, match="no plate carries at least"):
        zprime_frame(thin, zprime_spec())

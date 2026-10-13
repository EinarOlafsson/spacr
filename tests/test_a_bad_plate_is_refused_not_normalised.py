"""387(3, 4): percent inhibition per plate, and the Z' the package already has.

The instruction's bar for this pair is one sentence -- "a deliberately bad
plate must fail visibly" -- so every test here builds a plate that IS bad in
one specific way and asserts that spaCR says so rather than returning a clean
number. The good-plate tests exist to show the refusals are not simply a
function that always refuses.

The Z' tests matter for a second reason: 387 item 4 says CONSUME the Z' that
`spacr/qt/widgets/control_chart.py` already computes, do not rebuild it,
because "two screens computing Z' two ways would be worse than the gap it
closes". :func:`test_the_zprime_is_the_control_charts_own_number` is what
holds that: it compares the gate's Z' against `zprime_frame` directly, so a
future re-implementation here fails this file.
"""

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.dose_response import (
    STATUS_FITTED, STATUS_REFUSED, ZPRIME_MARGINAL,
    DoseResponseError, PlateSpec, PlateReport,
    normalise_to_controls, plate_reports,
)


def _plate(name, *, pos_mean, neg_mean, noise, n=8, seed=0, doses=(0.1, 1, 10)):
    """One plate: n positive wells, n negative wells, and a dose series."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        rows.append({"plate": name, "role": "pos", "dose": np.nan,
                     "signal": pos_mean + rng.normal(0, noise)})
        rows.append({"plate": name, "role": "neg", "dose": np.nan,
                     "signal": neg_mean + rng.normal(0, noise)})
    for dose in doses:
        rows.append({"plate": name, "role": "sample", "dose": dose,
                     "signal": (neg_mean + (pos_mean - neg_mean) * 0.5
                                + rng.normal(0, noise))})
    return pd.DataFrame(rows)


SPEC = PlateSpec(plate="plate", control="role",
                 positive=("pos",), negative=("neg",))


def test_the_controls_read_zero_and_one_hundred():
    """The scale is defined by its two ends, so they must land on them."""
    frame = _plate("P1", pos_mean=100.0, neg_mean=1000.0, noise=5.0)
    out, reports = normalise_to_controls(frame, SPEC, response="signal")

    assert [r.status for r in reports] == [STATUS_FITTED]
    pos = out.loc[out["role"] == "pos", "percent_inhibition"].mean()
    neg = out.loc[out["role"] == "neg", "percent_inhibition"].mean()
    assert pos == pytest.approx(100.0, abs=1e-9)
    assert neg == pytest.approx(0.0, abs=1e-9)


def test_a_readout_that_runs_the_other_way_normalises_the_same():
    """Direction-agnostic by construction: the positive control reads 100.

    A viability readout falls with dose and a burden readout rises with it.
    If the formula had a hard-coded direction, one of the two would come back
    as negative percent inhibition and the fit downstream would need to be
    told which readout it got.
    """
    falling = _plate("P1", pos_mean=100.0, neg_mean=1000.0, noise=5.0)
    rising = _plate("P1", pos_mean=1000.0, neg_mean=100.0, noise=5.0)

    for frame in (falling, rising):
        out, _ = normalise_to_controls(frame, SPEC, response="signal")
        assert out.loc[out["role"] == "pos",
                       "percent_inhibition"].mean() == pytest.approx(100.0)


def test_two_plates_at_different_gain_agree_after_normalising():
    """The whole point: raw signal is not comparable across plates.

    Plate B is the same experiment read at three times the gain. Raw, its
    sample wells sit nowhere near plate A's. Normalised, they coincide.
    """
    a = _plate("A", pos_mean=100.0, neg_mean=1000.0, noise=1.0, seed=1)
    b = _plate("B", pos_mean=300.0, neg_mean=3000.0, noise=3.0, seed=1)
    frame = pd.concat([a, b], ignore_index=True)

    raw = frame.loc[frame["role"] == "sample"].groupby("plate")["signal"].mean()
    assert abs(raw["A"] - raw["B"]) > 500.0

    out, reports = normalise_to_controls(frame, SPEC, response="signal")
    assert all(r.usable for r in reports)
    scaled = out.loc[out["role"] == "sample"].groupby(
        "plate")["percent_inhibition"].mean()
    assert scaled["A"] == pytest.approx(scaled["B"], abs=2.0)


def test_a_plate_with_no_assay_window_is_refused():
    """Both controls read the same. There is nothing to divide by.

    This is the failure that matters most, because the arithmetic does not
    complain: dividing by a near-zero separation turns well-to-well noise
    into hundreds of percent inhibition and a confident EC50 on a plate that
    measured nothing at all.
    """
    good = _plate("GOOD", pos_mean=100.0, neg_mean=1000.0, noise=5.0, seed=2)
    flat = _plate("FLAT", pos_mean=500.0, neg_mean=500.0, noise=0.0, seed=3)
    frame = pd.concat([good, flat], ignore_index=True)

    out, reports = normalise_to_controls(frame, SPEC, response="signal")
    verdicts = {r.plate: r for r in reports}

    assert verdicts["GOOD"].status == STATUS_FITTED
    assert verdicts["FLAT"].status == STATUS_REFUSED
    assert "no assay window" in verdicts["FLAT"].note
    # and the refusal reaches the data, not only the report
    assert out.loc[out["plate"] == "FLAT", "percent_inhibition"].isna().all()
    assert out.loc[out["plate"] == "GOOD", "percent_inhibition"].notna().all()


def test_a_plate_missing_a_control_is_refused_by_name():
    """Half a scale is not a scale, and the message says which half."""
    frame = _plate("P1", pos_mean=100.0, neg_mean=1000.0, noise=5.0)
    frame = frame.loc[frame["role"] != "pos"].reset_index(drop=True)

    with pytest.raises(DoseResponseError):
        normalise_to_controls(frame, SPEC, response="signal")

    reports = plate_reports(frame, SPEC, response="signal")
    assert reports[0].status == STATUS_REFUSED
    assert "no positive control" in reports[0].note


def test_a_noisy_plate_fails_the_zprime_gate_with_the_number_in_the_note():
    """A plate below the gate is refused, and the refusal quotes its Z'.

    "A fit on a failed plate should be STATUS_REFUSED with the Z' in its
    message, not a clean EC50 with a footnote nobody reads."
    """
    clean = _plate("CLEAN", pos_mean=100.0, neg_mean=1000.0, noise=5.0, seed=4)
    noisy = _plate("NOISY", pos_mean=100.0, neg_mean=1000.0, noise=250.0, seed=5)
    frame = pd.concat([clean, noisy], ignore_index=True)

    gated = PlateSpec(plate="plate", control="role", positive=("pos",),
                      negative=("neg",), min_zprime=ZPRIME_MARGINAL)
    out, reports = normalise_to_controls(frame, gated, response="signal")
    verdicts = {r.plate: r for r in reports}

    assert verdicts["CLEAN"].status == STATUS_FITTED
    assert verdicts["NOISY"].status == STATUS_REFUSED
    assert "fails Z'" in verdicts["NOISY"].note
    assert f"{verdicts['NOISY'].zprime:.3g}" in verdicts["NOISY"].note
    assert out.loc[out["plate"] == "NOISY", "percent_inhibition"].isna().all()


def test_ungated_the_same_noisy_plate_normalises_but_says_so():
    """Without a gate nothing is dropped -- and the caveat is still carried.

    A threshold nobody chose is a threshold nobody can defend, so the default
    gates nothing. What it must not do is stay silent: the report still names
    the Z' and the convention it sits below.
    """
    noisy = _plate("NOISY", pos_mean=100.0, neg_mean=1000.0, noise=250.0, seed=5)
    out, reports = normalise_to_controls(noisy, SPEC, response="signal")

    assert reports[0].status == STATUS_FITTED
    assert reports[0].zprime < ZPRIME_MARGINAL
    assert "below" in reports[0].note
    assert out["percent_inhibition"].notna().any()


def test_the_zprime_is_the_control_charts_own_number():
    """387 item 4: consume `zprime_frame`, do not rebuild it.

    Two screens computing Z' two ways would be worse than the gap this
    closes -- the Control Chart would show 0.62 while Dose-Response refused
    the same plate at 0.48 and no user could say which to believe.
    """
    from spacr.qt.widgets.control_chart import (
        ControlChartSpec, ZPRIME_PLATE, ZPRIME_VALUE, zprime_frame)

    frame = pd.concat([
        _plate("A", pos_mean=100.0, neg_mean=1000.0, noise=20.0, seed=6),
        _plate("B", pos_mean=100.0, neg_mean=1000.0, noise=180.0, seed=7),
    ], ignore_index=True)

    chart = ControlChartSpec(value="signal", plate="plate",
                             control_column="role",
                             control_levels=("pos", "neg"),
                             positive_levels=("pos",),
                             negative_levels=("neg",))
    theirs = {str(row[ZPRIME_PLATE]): float(row[ZPRIME_VALUE])
              for _, row in zprime_frame(frame, chart).iterrows()}
    mine = {r.plate: r.zprime
            for r in plate_reports(frame, SPEC, response="signal")}

    assert set(theirs) == set(mine)
    for plate, value in theirs.items():
        assert mine[plate] == pytest.approx(value)


def test_a_plate_too_small_for_a_zprime_has_none_not_zero():
    """Absence of a number, reported as absence.

    `zprime_frame` leaves a plate with one control well out rather than
    giving it a zero. Inventing one here would undo that -- and a zero would
    read as a terrible plate rather than an unmeasured one.
    """
    frame = _plate("SMALL", pos_mean=100.0, neg_mean=1000.0, noise=5.0, n=1)
    reports = plate_reports(frame, SPEC, response="signal")

    assert reports[0].zprime is None
    assert reports[0].status == STATUS_FITTED
    assert "no Z'" in reports[0].note


def test_a_gate_on_a_plate_with_no_zprime_refuses_rather_than_guesses():
    """Asked to certify a plate it cannot certify, the gate says no.

    The alternative -- letting an ungatable plate through a gate the user
    switched on -- would mean `min_zprime` quietly meant "where available",
    which is the kind of silent exception this module exists not to make.
    """
    frame = _plate("SMALL", pos_mean=100.0, neg_mean=1000.0, noise=5.0, n=1)
    gated = PlateSpec(plate="plate", control="role", positive=("pos",),
                      negative=("neg",), min_zprime=ZPRIME_MARGINAL)

    reports = plate_reports(frame, gated, response="signal")
    assert reports[0].status == STATUS_REFUSED
    assert "cannot be gated" in reports[0].note


def test_a_spec_with_one_control_is_refused_when_it_is_built():
    """At the point the spec is built, not halfway through a plate."""
    with pytest.raises(DoseResponseError, match="two-point scale"):
        PlateSpec(plate="plate", control="role", positive=("pos",))
    with pytest.raises(DoseResponseError, match="identifies the plate"):
        PlateSpec(control="role", positive=("pos",), negative=("neg",))


def test_the_spec_round_trips_through_json():
    """The normalisation behind a figure travels with the fit that used it."""
    spec = PlateSpec(plate="plate", control="role", positive=("pos", "max"),
                     negative=("neg",), min_zprime=0.4)
    assert PlateSpec.from_json(spec.to_json()) == spec


def test_the_plate_table_carries_the_refusals_too():
    """Six curves out of eight plates, and the other two findable."""
    frame = pd.concat([
        _plate("A", pos_mean=100.0, neg_mean=1000.0, noise=5.0, seed=8),
        _plate("B", pos_mean=500.0, neg_mean=500.0, noise=0.0, seed=9),
    ], ignore_index=True)

    rows = [r.summary_row() for r in plate_reports(frame, SPEC,
                                                   response="signal")]
    table = pd.DataFrame(rows)
    assert set(table["plate"]) == {"A", "B"}
    assert set(table["status"]) == {STATUS_FITTED, STATUS_REFUSED}
    assert table.loc[table["plate"] == "B", "note"].iloc[0]


def test_a_second_readout_normalises_into_its_own_column():
    """Host viability and parasite burden, off one plate, side by side.

    This is the pairing 387 item 1 needs: a selectivity index divides two
    fits, and those fits are only comparable if both readouts were scaled to
    the same plate's controls.
    """
    frame = _plate("P1", pos_mean=100.0, neg_mean=1000.0, noise=5.0)
    frame["viability"] = frame["signal"] * 0.5 + 20.0

    out, _ = normalise_to_controls(frame, SPEC, response="signal")
    out, _ = normalise_to_controls(out, SPEC, response="viability",
                                   out="percent_viability")

    assert {"percent_inhibition", "percent_viability"} <= set(out.columns)
    assert out["percent_inhibition"].equals(out["percent_viability"]) is False
    assert out.loc[out["role"] == "pos",
                   "percent_viability"].mean() == pytest.approx(100.0)

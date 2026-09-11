"""387(5): one EC50 across replicate plates, rather than three and an eyeball.

The instruction's bar is stated as an arithmetic fact -- "pooling three
replicates of the same compound must land inside the three individual
intervals" -- and :func:`test_the_pooled_ec50_lands_inside_every_plates_interval`
is that sentence. The rest of the file is 387's own standard applied to
pooling: the useful answer is sometimes "these plates do not support one
number", so plates that genuinely disagree must be refused rather than
averaged into a value none of them measured.
"""

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.dose_response import (
    DEFAULT_CONFIDENCE, MAX_HETEROGENEITY, MIN_PLATES,
    STATUS_FITTED, STATUS_REFUSED,
    DoseResponseError, DoseResponseSpec, PooledFit,
    fit_dose_response, four_parameter_logistic,
    pool_across_plates, pool_frame,
)

DOSES = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])


def _plate_frame(name, ec50, *, noise=2.0, seed=0, replicates=3):
    """A clean inhibition series on one plate, at a stated EC50."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(replicates):
        signal = four_parameter_logistic(DOSES, 0.0, 100.0,
                                         np.log10(ec50), -1.0)
        signal = signal + rng.normal(0.0, noise, size=DOSES.shape)
        for dose, value in zip(DOSES, signal):
            rows.append({"plate": name, "dose": dose, "response": value})
    return pd.DataFrame(rows)


SPEC = DoseResponseSpec(concentration="dose", response="response", unit="uM")


def _fit(name, ec50, **kwargs):
    frame = _plate_frame(name, ec50, **kwargs)
    return fit_dose_response(frame["dose"], frame["response"], SPEC,
                             group=name)


def test_the_pooled_ec50_lands_inside_every_plates_interval():
    """387's stated bar, verbatim.

    Three replicates of the same compound, so the pooled value must sit
    inside each of the three intervals -- a pooled estimate outside them all
    would be an artefact of the weighting, not a summary of the data.
    """
    fits = {name: _fit(name, 1.0, seed=seed)
            for seed, name in enumerate(("A", "B", "C"), start=1)}
    pooled = pool_across_plates(fits)

    assert pooled.status == STATUS_FITTED
    assert pooled.n_used == 3
    for name, result in fits.items():
        low, high = result.log10_ec50_ci
        assert low <= pooled.log10_ec50 <= high, f"outside plate {name}"


def test_pooling_is_tighter_than_any_one_plate_when_they_agree():
    """The gain from pooling, which is the reason to do it at all."""
    fits = {name: _fit(name, 1.0, seed=seed)
            for seed, name in enumerate(("A", "B", "C"), start=1)}
    pooled = pool_across_plates(fits)

    widths = [high - low for low, high in
              (r.log10_ec50_ci for r in fits.values())]
    pooled_width = 2.0 * pooled.log10_se * 1.959963984540054
    assert pooled_width < min(widths)


def test_the_pool_is_geometric_not_arithmetic():
    """1, 10 and 100 uM pool to 10, not to 37.

    An EC50 is estimated on the log10 scale and its interval is symmetric
    there. Averaging concentrations arithmetically would let the highest
    plate drag the answer, which is why a pooled EC50 done in a spreadsheet
    is usually wrong in a direction nobody notices.
    """
    fits = {"A": _fit("A", 0.1, seed=1),
            "B": _fit("B", 1.0, seed=2),
            "C": _fit("C", 10.0, seed=3)}
    # Deliberately heterogeneous, so ask for the number without the gate.
    pooled = pool_across_plates(fits, max_heterogeneity=1.0)

    assert pooled.status == STATUS_FITTED
    assert pooled.ec50 == pytest.approx(1.0, rel=0.3)      # geometric
    assert pooled.ec50 < (0.1 + 1.0 + 10.0) / 3.0          # not arithmetic


def test_plates_that_disagree_are_refused_with_the_spread_named():
    """The disagreement IS the finding, so it must not be averaged away.

    Three tight plates at 0.1, 1 and 10 uM are not three measurements of one
    EC50; something differed between them. A pooled number here would be a
    confident value that none of the three supports.
    """
    fits = {"A": _fit("A", 0.1, noise=0.5, seed=1),
            "B": _fit("B", 1.0, noise=0.5, seed=2),
            "C": _fit("C", 10.0, noise=0.5, seed=3)}
    pooled = pool_across_plates(fits)

    assert pooled.status == STATUS_REFUSED
    assert pooled.ec50 is None
    assert "disagree" in pooled.note
    assert "I-squared" in pooled.note
    assert pooled.i_squared > MAX_HETEROGENEITY
    # the refusal still carries the diagnostics, so the user can act on it
    assert pooled.q > 0 and pooled.q_p < 0.05
    assert pooled.n_used == 3


def test_random_effects_not_fixed_effects():
    """Real plate-to-plate variation widens the answer, not narrows it.

    Under fixed-effect pooling three tight but separated plates produce an
    impossibly narrow interval around a value none of them support. The
    between-plate variance is added to each plate's own, so tau > 0 forces
    the interval open.
    """
    spread = {"A": _fit("A", 0.8, noise=3.0, seed=1),
              "B": _fit("B", 1.0, noise=3.0, seed=2),
              "C": _fit("C", 1.25, noise=3.0, seed=3)}
    pooled = pool_across_plates(spread)
    assert pooled.status == STATUS_FITTED
    assert pooled.tau > 0.0

    fixed_w = np.array([1.0 / ((high - low) / (2 * 1.959963984540054)) ** 2
                        for low, high in
                        (r.log10_ec50_ci for r in spread.values())])
    fixed_se = float(np.sqrt(1.0 / fixed_w.sum()))
    assert pooled.log10_se > fixed_se
    assert "weighted away" in pooled.note


def test_tau_reports_reproducibility_no_average_could():
    """The number a screener wants and three EC50s cannot give.

    tau is the plate-to-plate SD on log10, so a compound reproducible to
    within 10% and one reproducible to within threefold get different
    numbers even when their means coincide.
    """
    tight = pool_across_plates({"A": _fit("A", 1.0, noise=0.5, seed=1),
                                "B": _fit("B", 1.02, noise=0.5, seed=2),
                                "C": _fit("C", 0.98, noise=0.5, seed=3)})
    loose = pool_across_plates({"A": _fit("A", 0.5, noise=0.5, seed=1),
                                "B": _fit("B", 1.0, noise=0.5, seed=2),
                                "C": _fit("C", 2.0, noise=0.5, seed=3)},
                               max_heterogeneity=1.0)
    assert loose.tau > tight.tau
    assert loose.ec50 == pytest.approx(tight.ec50, rel=0.35)


def test_one_plate_is_not_a_replicate():
    """Refused, and the refusal says what to do instead."""
    pooled = pool_across_plates({"A": _fit("A", 1.0, seed=1)})
    assert pooled.status == STATUS_REFUSED
    assert f"at least {MIN_PLATES} plates" in pooled.note
    assert "not a replicate" in pooled.note


def test_a_plate_with_an_unbounded_ec50_is_named_not_dropped_silently():
    """Dropping it quietly would make the interval look better than the run.

    A plate whose top dose never reached the plateau has no variance to
    weight by, so it cannot enter the pool -- but the pooled interval would
    then be reported as if that plate had never been run.
    """
    flat = np.full(DOSES.shape, 100.0) - DOSES * 0.02   # never turns over
    weak = fit_dose_response(DOSES, flat, SPEC, group="C")
    fits = {"A": _fit("A", 1.0, seed=1), "B": _fit("B", 1.0, seed=2),
            "C": weak}

    pooled = pool_across_plates(fits)
    assert pooled.n_plates == 3
    assert pooled.n_used == 2
    assert "C" in pooled.note
    assert "no variance to weight by" in pooled.note


def test_pool_frame_fits_each_plate_then_pools():
    """The convenience path over one table with a plate column."""
    frame = pd.concat([_plate_frame(name, 1.0, seed=seed)
                       for seed, name in enumerate(("A", "B", "C"), start=1)],
                      ignore_index=True)
    pooled = pool_frame(frame, SPEC, plate="plate")

    assert pooled.status == STATUS_FITTED
    assert pooled.n_used == 3
    assert pooled.unit == "uM"
    assert {name for name, _ in pooled.per_plate} == {"A", "B", "C"}


def test_a_plate_that_cannot_be_fitted_at_all_is_reported_not_swallowed():
    """One bad plate must not take the pool down, nor vanish from it."""
    frame = pd.concat([_plate_frame(name, 1.0, seed=seed)
                       for seed, name in enumerate(("A", "B"), start=1)],
                      ignore_index=True)
    broken = pd.DataFrame({"plate": "BAD", "dose": [1.0, 1.0],
                           "response": [50.0, 51.0]})
    pooled = pool_frame(pd.concat([frame, broken], ignore_index=True),
                        SPEC, plate="plate")

    assert pooled.status == STATUS_FITTED
    assert "BAD" in pooled.note
    assert "did not fit at all" in pooled.note


def test_no_plate_column_is_refused_before_any_fitting():
    with pytest.raises(DoseResponseError, match="no replicates to pool"):
        pool_frame(_plate_frame("A", 1.0), SPEC, plate="run")


def test_an_impossible_heterogeneity_gate_is_refused_when_asked_for():
    with pytest.raises(DoseResponseError, match="share of the spread"):
        pool_across_plates({"A": _fit("A", 1.0, seed=1),
                            "B": _fit("B", 1.0, seed=2)},
                           max_heterogeneity=1.5)


def test_the_pooled_row_carries_the_refusal_into_the_table():
    """A refused pool is still a row, so it cannot silently be no row."""
    fits = {"A": _fit("A", 0.1, noise=0.5, seed=1),
            "B": _fit("B", 1.0, noise=0.5, seed=2),
            "C": _fit("C", 10.0, noise=0.5, seed=3)}
    row = pool_across_plates(fits).summary_row()

    assert row["metric"] == "pooled_ec50"
    assert row["status"] == STATUS_REFUSED
    assert np.isnan(row["ec50"])
    assert row["n_plates"] == 3
    assert row["note"]


def test_the_pool_can_always_be_taken_apart_again():
    """Every pooled number keeps the fits it came from."""
    fits = {name: _fit(name, 1.0, seed=seed)
            for seed, name in enumerate(("A", "B", "C"), start=1)}
    pooled = pool_across_plates(fits)

    assert len(pooled.per_plate) == 3
    for name, result in pooled.per_plate:
        assert result.ec50 == pytest.approx(fits[name].ec50)

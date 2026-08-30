"""The two guards that stand between a bad well and a confident guide name.

Both of them are arithmetic seatbelts inside :mod:`spacr.guide_attribution`,
and both matter to whoever reads the annotation table rather than the code:

* :func:`attributable` divides the competing guides' weights by their total
  before it can say whether a guide is callable at all. A total that is not
  positive would turn that division into ``inf``/``nan`` and hand the caller a
  posterior ceiling built out of nonsense -- a guide the preflight then
  promises can be called when nothing about it can.
* :func:`posterior_multivariate` exponentiates a sum of log densities. Option
  C reads hundreds of measurements per cell, so the sum is routinely far below
  ``log(tiny)``; the per-cell maximum is subtracted first precisely so that no
  cell's row collapses to all-zero and gets handed back a uniform "ambiguous".
  These tests pin that invariant down with inputs where every guide really is
  impossible.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.guide_attribution import attributable, posterior_multivariate


class _WeightReadTwice:
    """A weight whose float value changes between reads.

    ``attributable`` calls ``float(w)`` once to test the weight and again to
    store it, so a weight that is not a plain number -- a lazily fetched
    count, a mutable cell, a proxy over a stream -- can pass the test and be
    stored as something else entirely. This stands in for that caller.
    """

    def __init__(self, first: float, then: float) -> None:
        self._values = [float(first), float(then)]
        self.reads = 0

    def __float__(self) -> float:
        value = self._values[min(self.reads, 1)]
        self.reads += 1
        return value


# ---------------------------------------------------------------------------
# attributable: the competing weights must add up to something positive
# ---------------------------------------------------------------------------

def test_a_competitor_weight_that_changes_between_reads_is_refused():
    """A non-positive total must not become a division, it must become "no".

    The line after the guard is ``w * (1 - p) / total``. If a weight sneaks
    past the ``> 0`` filter and the total comes out at or below zero, that
    division produces infinities and the preflight would report a guide as
    reachable with a ceiling computed from them. The honest answer for a well
    whose competition does not add up is "this guide cannot be called".
    """
    unstable = _WeightReadTwice(1.0, -2.0)
    refused = attributable(1.0, 1.0, 0.5, others=[(0.0, unstable)])

    assert refused == (False, 0.0)
    assert unstable.reads == 2                 # read for the test, and again

    # The same call with a weight that means what it says IS answerable, so
    # the refusal above is the guard and not a function that never says yes.
    allowed = attributable(1.0, 1.0, 0.5, others=[(0.0, 1.0)])
    assert allowed[0] is True
    assert 0.5 < allowed[1] <= 1.0


def test_competitors_with_no_weight_leave_one_flat_rival_behind():
    """Dropping a zero-weight guide must not leave the well uncontested.

    Sequencing routinely reports guides at zero reads in a well. If those
    dropped out and nothing replaced them, the candidate guide would be
    compared against an empty field and every guide would look callable.
    """
    dead_competition = attributable(0.8, 1.0, 0.4,
                                    others=[(2.0, 0.0), (-2.0, 0.0)])
    no_competition_given = attributable(0.8, 1.0, 0.4, others=None)

    assert dead_competition == no_competition_given
    assert dead_competition[0] is True
    # And a live competitor with the SAME effect really does change the
    # answer, so the equality above is not two identical no-ops.
    real_competition = attributable(0.8, 1.0, 0.4, others=[(0.8, 0.6)])
    assert real_competition[1] < dead_competition[1]


def test_the_competition_is_judged_by_its_shape_not_its_units():
    """Read counts and read fractions must give the same verdict.

    Callers pass whatever the sequencing table holds -- raw counts in one
    pipeline, normalised fractions in another. The weights are rescaled to the
    rest of the well, so a well described in counts and the same well
    described in fractions have to agree, or the preflight's answer would
    depend on which table a lab happened to load.
    """
    counts = attributable(1.5, 1.0, 0.25, others=[(0.0, 30.0), (1.0, 10.0)])
    fractions = attributable(1.5, 1.0, 0.25, others=[(0.0, 0.75), (1.0, 0.25)])

    assert counts[1] == pytest.approx(fractions[1], rel=1e-12)
    assert counts[0] == fractions[0] is True
    # Change the SHAPE of the competition and the ceiling moves, which is what
    # makes the agreement above meaningful.
    lopsided = attributable(1.5, 1.0, 0.25, others=[(1.5, 30.0), (1.5, 10.0)])
    assert lopsided[1] < counts[1]


def test_a_guide_with_no_reads_and_a_guide_with_every_read():
    """The two certainties are answered without any competition arithmetic.

    A guide absent from a well can never be called and a guide that is the
    whole well is always called; both answers must be exact, because the
    preflight prints them as "hopeless" and "certain" to the user.
    """
    assert attributable(5.0, 1.0, 0.0, others=[(0.0, 1.0)]) == (False, 0.0)
    assert attributable(0.0, 1.0, 1.0, others=[(0.0, 1.0)]) == (True, 1.0)
    # Between those, the answer is neither exact nor trivially yes/no.
    middling = attributable(0.2, 1.0, 0.5, others=[(0.0, 0.5)])
    assert 0.0 < middling[1] < 1.0


# ---------------------------------------------------------------------------
# posterior_multivariate: no cell may come back with an all-zero row
# ---------------------------------------------------------------------------

def _impossible_well(n_columns: int = 40):
    """A well where every guide's expectation is nowhere near the scores.

    Each guide sits 40 sigma away in every one of ``n_columns`` measurements,
    so every density underflows to zero and the summed log density is a few
    ten-thousand nats below zero -- the regime option C exists to survive.
    """
    measurements = np.zeros((3, n_columns), dtype=float)
    measurements[1, :] = 0.5
    measurements[2, :] = -0.5
    effects = {"g1": [40.0] * n_columns, "g2": [45.0] * n_columns}
    priors = {"g1": 0.75, "g2": 0.25}
    return measurements, priors, effects


def test_a_cell_no_guide_can_explain_still_gets_a_real_distribution():
    """Underflow must not turn a row of the posterior into zeros.

    Every guide here is 40 sigma from every measurement, so the raw product of
    densities is zero in double precision. Without the per-cell shift before
    the exponential, each cell's row would be all zeros, the normalisation
    would divide by zero, and the whole well would come back as NaN -- which
    downstream reads as "no guide anywhere" for a plate that merely had a
    badly centred score column.
    """
    measurements, priors, effects = _impossible_well()

    r, guides, report = posterior_multivariate(
        measurements, priors, effects,
        centres=[0.0] * measurements.shape[1],
        scales=[1.0] * measurements.shape[1],
        correct_for_correlation=False)

    assert guides == ("g1", "g2")
    assert r.shape == (3, 2)
    assert np.isfinite(r).all()
    assert r.sum(axis=1) == pytest.approx(np.ones(3))
    assert (r > 0).all()
    assert report["scale_factor"] == 1.0


def test_an_impossible_well_falls_back_to_the_sequencing_fractions():
    """When the scores say nothing, the read fractions must be the answer.

    All guides being equally impossible is not the same as all guides being
    equally likely: the sequencing still says three quarters of the cells
    carry g1. The posterior has to land on the prior, so a user reading the
    attribution of a hopeless well sees the fractions they sequenced rather
    than a coin flip.
    """
    measurements, priors, effects = _impossible_well()

    r, guides, _ = posterior_multivariate(
        measurements, priors, effects,
        centres=[0.0] * measurements.shape[1],
        scales=[1.0] * measurements.shape[1],
        correct_for_correlation=False)

    assert r[:, 0] == pytest.approx(np.full(3, 0.75), abs=1e-9)
    assert r[:, 1] == pytest.approx(np.full(3, 0.25), abs=1e-9)
    # The mass constraint that produces it: each guide's column sums to its
    # fraction of the cells.
    assert r.sum(axis=0) == pytest.approx(np.array([2.25, 0.75]), abs=1e-6)
    # A well the scores CAN separate does not sit on the prior, so the match
    # above is the fallback and not the only thing this function returns.
    separable, _, _ = posterior_multivariate(
        np.array([[3.0], [-3.0], [-3.0]]), priors, {"g1": [3.0], "g2": [-3.0]},
        centres=[0.0], scales=[1.0], correct_for_correlation=False)
    assert separable[0, 0] > 0.9


def test_a_measurement_no_cell_recorded_is_skipped_not_scored_as_zero():
    """A column of NaNs must contribute nothing, not a score of zero.

    Feature tables arrive with columns that failed for a whole plate. Treating
    the missing value as 0.0 would be a real and usually extreme score, and it
    would drag every cell towards whichever guide happens to sit near zero.
    """
    scores = np.array([[2.0], [-2.0]])
    blank = np.full((2, 1), np.nan)
    priors = {"g1": 0.5, "g2": 0.5}
    effects = {"g1": [2.0, 0.0], "g2": [-2.0, 0.0]}

    with_blank, guides, report = posterior_multivariate(
        np.hstack([scores, blank]), priors, effects,
        centres=[0.0, 0.0], scales=[1.0, 1.0],
        correct_for_correlation=False)
    without_blank, _, _ = posterior_multivariate(
        scores, priors, {"g1": [2.0], "g2": [-2.0]},
        centres=[0.0], scales=[1.0], correct_for_correlation=False)

    assert guides == ("g1", "g2")
    assert with_blank == pytest.approx(without_blank, abs=1e-12)
    assert with_blank[0, 0] > 0.9 and with_blank[1, 1] > 0.9
    # The blank column was still counted as a measurement in the report, so
    # the reader can see what was offered as well as what was used.
    assert report["n_measurements"] == 2.0


def test_the_diagnostics_show_how_much_the_correlation_correction_did():
    """Duplicated features must not be allowed to vote twice.

    ``cell_area`` and ``cell_perimeter`` are one measurement wearing two
    names. If the evidence is not scaled down, twelve copies of one column
    make the posterior saturate and the 0.55 assignment threshold stops
    meaning anything. The report has to show the scaling so a reader can tell
    how much of their "785 measurements" were real.
    """
    rng = np.random.default_rng(4)
    one_column = rng.normal(size=(200, 1))
    duplicated = np.repeat(one_column, 8, axis=1)
    priors = {"g1": 0.5, "g2": 0.5}
    effects = {"g1": [1.0] * 8, "g2": [-1.0] * 8}

    corrected, _, report = posterior_multivariate(duplicated, priors, effects)
    uncorrected, _, plain = posterior_multivariate(
        duplicated, priors, effects, correct_for_correlation=False)

    assert report["n_measurements"] == 8.0
    assert report["effective_dimension"] == pytest.approx(1.0, abs=1e-6)
    assert report["scale_factor"] == pytest.approx(0.125, abs=1e-6)
    assert plain["scale_factor"] == 1.0
    # Same data, and the uncorrected run really is the more confident one.
    assert corrected.max() < uncorrected.max()


def test_a_guide_nothing_was_fitted_for_stays_flat():
    """A guide with no effect vector must not be quietly dropped.

    Screens carry guides the regression never fitted. They still hold reads,
    so they still hold prior mass; giving them a flat likelihood keeps the
    cell counts honest instead of handing their share to whichever guide was
    modelled.
    """
    measurements = np.array([[2.0, 2.0], [0.0, 0.0], [-2.0, -2.0]])
    priors = {"fitted": 0.5, "unfitted": 0.5}

    r, guides, _ = posterior_multivariate(
        measurements, priors, {"fitted": [2.0, 2.0]},
        centres=[0.0, 0.0], scales=[1.0, 1.0],
        correct_for_correlation=False)

    assert guides == ("fitted", "unfitted")
    assert np.isfinite(r).all()
    assert r[0, 0] > r[0, 1]          # the high-scoring cell goes to `fitted`
    assert r[2, 1] > r[2, 0]          # the low-scoring one to the flat guide
    assert r.sum(axis=0) == pytest.approx(np.array([1.5, 1.5]), abs=1e-6)


def test_a_short_effect_vector_does_not_reach_past_its_end():
    """A guide fitted on fewer columns than the table has must not crash.

    Effect vectors come from a regression that may have been run on a subset
    of the features. Indexing past the end would raise mid-plate; the missing
    columns have to read as "no effect" so the columns that were fitted still
    count.
    """
    measurements = np.array([[3.0, 0.0], [-3.0, 0.0]])
    priors = {"g1": 0.5, "g2": 0.5}

    r, guides, _ = posterior_multivariate(
        measurements, priors, {"g1": [3.0], "g2": [-3.0]},
        centres=[0.0, 0.0], scales=[1.0, 1.0],
        correct_for_correlation=False)

    assert guides == ("g1", "g2")
    assert r.shape == (2, 2)
    assert r[0, 0] > 0.9
    assert r[1, 1] > 0.9

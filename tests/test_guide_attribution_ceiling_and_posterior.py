"""The posterior ceiling and the multivariate posterior of guide attribution.

Two units of :mod:`spacr.guide_attribution`, each walked over a case table:

* ``attributable`` -- whether a guide can ever reach the assignment threshold
  in a well, and the best posterior it could reach.  The competing weights are
  rescaled to the rest of the well, so only their shape matters, and each
  weight is read exactly once.
* ``posterior_multivariate`` (option C) -- a cell's evidence summed over every
  measurement, shifted per cell before the exponential so no row can underflow
  to zero, scaled down for correlated measurements, and fitted so each guide's
  column sums to its sequencing fraction of the cells.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.guide_attribution import (DEFAULT_THRESHOLD, attributable,
                                     posterior_multivariate)


class _WeightReadTwice:
    """A weight whose float value changes between reads (1.0, then -2.0)."""

    def __init__(self, first: float, then: float) -> None:
        self._values = [float(first), float(then)]
        self.reads = 0

    def __float__(self) -> float:
        value = self._values[min(self.reads, 1)]
        self.reads += 1
        return value


def test_attributable(subtests):
    """Exact (callable, ceiling) answers across priors, rivals and settings."""
    assert DEFAULT_THRESHOLD == 0.55
    cases = [
        ("absent guide is hopeless", (5.0, 1.0, 0.0), {"others": [(0.0, 1.0)]},
         (False, 0.0)),
        ("negative prior clips to absent", (0.0, 1.0, -1.0), {}, (False, 0.0)),
        ("whole-well guide is certain", (0.0, 1.0, 1.0), {"others": [(0.0, 1.0)]},
         (True, 1.0)),
        ("prior above one clips to certain", (0.0, 1.0, 1.5), {}, (True, 1.0)),
        ("one flat rival", (1.0, 1.0, 0.5), {"others": [(0.0, 1.0)]},
         (True, 0.9890130573694067)),
        ("narrower span lowers the ceiling", (1.0, 1.0, 0.5),
         {"others": [(0.0, 1.0)], "span": 1.0}, (True, 0.8175744761936437)),
        ("grid below three is raised to three", (1.0, 1.0, 0.5),
         {"others": [(0.0, 1.0)], "grid": 1}, (True, 0.9890130573694067)),
        ("no rivals given: the rest of the well is one flat rival",
         (0.8, 1.0, 0.4), {"others": None}, (True, 0.9574883294046603)),
        ("zero-weight rivals are replaced by one flat rival", (0.8, 1.0, 0.4),
         {"others": [(2.0, 0.0), (-2.0, 0.0)]}, (True, 0.9574883294046603)),
        ("a negative-weight rival is dropped too", (0.0, 1.0, 0.5),
         {"others": [(0.0, -1.0)]}, (False, 0.5)),
        ("a rival with the same effect caps the ceiling at the prior",
         (0.8, 1.0, 0.4), {"others": [(0.8, 0.6)]}, (False, 0.4)),
        ("read counts", (1.5, 1.0, 0.25), {"others": [(0.0, 30.0), (1.0, 10.0)]},
         (True, 0.9162666299855544)),
        ("the same well as fractions", (1.5, 1.0, 0.25),
         {"others": [(0.0, 0.75), (1.0, 0.25)]}, (True, 0.9162666299855544)),
        ("rivals sharing the effect", (1.5, 1.0, 0.25),
         {"others": [(1.5, 30.0), (1.5, 10.0)]}, (False, 0.25)),
        ("middling", (0.2, 1.0, 0.5), {"others": [(0.0, 0.5)]},
         (True, 0.6942363401080306)),
        ("zero scale resolves to one", (0.2, 0.0, 0.5), {"others": [(0.0, 0.5)]},
         (True, 0.6942363401080306)),
        ("negative scale resolves to one", (0.2, -3.0, 0.5),
         {"others": [(0.0, 0.5)]}, (True, 0.6942363401080306)),
        ("ceiling below a raised threshold", (0.2, 1.0, 0.5),
         {"others": [(0.0, 0.5)], "threshold": 0.7}, (False, 0.6942363401080306)),
        ("ceiling exactly at the threshold counts", (0.2, 1.0, 0.5),
         {"others": [(0.0, 0.5)], "threshold": 0.6942363401080306},
         (True, 0.6942363401080306)),
    ]
    for case_id, args, kwargs, (can, best) in cases:
        with subtests.test(msg=case_id):
            answer = attributable(*args, **kwargs)
            assert answer[0] is can
            assert answer[1] == pytest.approx(best, rel=1e-12, abs=1e-15)

    with subtests.test(msg="each competitor weight is read exactly once"):
        unstable = _WeightReadTwice(1.0, -2.0)
        answer = attributable(1.0, 1.0, 0.5, others=[(0.0, unstable)])
        assert unstable.reads == 1
        assert answer == attributable(1.0, 1.0, 0.5, others=[(0.0, 1.0)])


def _call(measurements, priors, effects, **kwargs):
    kwargs.setdefault("correct_for_correlation", False)
    n = np.asarray(measurements).reshape(len(measurements), -1).shape[1]
    kwargs.setdefault("centres", [0.0] * n)
    kwargs.setdefault("scales", [1.0] * n)
    return posterior_multivariate(np.asarray(measurements, dtype=float),
                                  priors, effects, **kwargs)


def _impossible_well(n_columns=40):
    """Every guide 40+ sigma from every score: every raw density underflows."""
    measurements = np.zeros((3, n_columns))
    measurements[1, :] = 0.5
    measurements[2, :] = -0.5
    return measurements, {"g1": [40.0] * n_columns, "g2": [45.0] * n_columns}


def test_posterior_multivariate(subtests):
    """Exact posteriors and diagnostics; rows sum to one, columns to the priors."""
    even = {"g1": 0.5, "g2": 0.5}
    impossible, far = _impossible_well()
    rng = np.random.default_rng(0)
    shared = rng.normal(0.0, 1.0, (4, 1))
    correlated = np.repeat(shared, 200, axis=1) + rng.normal(0.0, 1e-3, (4, 200))

    cases = [
        ("separable well", [[3.0], [-3.0], [-3.0]], {"g1": 0.75, "g2": 0.25},
         {"g1": [3.0], "g2": [-3.0]}, {},
         [[1.0, 0.0], [0.62499999961027064, 0.37500000038972930],
          [0.62499999961027064, 0.37500000038972930]],
         {"n_measurements": 1.0, "effective_dimension": 1.0, "scale_factor": 1.0}),
        ("an all-NaN column is skipped, not scored as zero",
         [[2.0, np.nan], [-2.0, np.nan]], even,
         {"g1": [2.0, 0.0], "g2": [-2.0, 0.0]}, {},
         [[0.99966464986953363, 3.3535013046647816e-04],
          [3.3535013046647816e-04, 0.99966464986953363]],
         {"n_measurements": 2.0, "effective_dimension": 2.0, "scale_factor": 1.0}),
        ("an unfitted guide keeps a flat likelihood",
         [[2.0, 2.0], [0.0, 0.0], [-2.0, -2.0]], {"fitted": 0.5, "unfitted": 0.5},
         {"fitted": [2.0, 2.0]}, {},
         [[0.9996646498666095, 3.353501333904573e-04],
          [0.4999999978194728, 0.5000000021805273],
          [3.353501275424987e-04, 0.9996646498724575]],
         {"n_measurements": 2.0, "effective_dimension": 2.0, "scale_factor": 1.0}),
        ("a short effect vector reads as no effect past its end",
         [[3.0, 0.0], [-3.0, 0.0]], even, {"g1": [3.0], "g2": [-3.0]}, {},
         [[0.9999999847700205, 1.5229979512760293e-08],
          [1.5229979512760293e-08, 0.9999999847700205]],
         {"n_measurements": 2.0, "effective_dimension": 2.0, "scale_factor": 1.0}),
        ("extreme effects are assigned, not flattened",
         [[0.0], [40.0], [-40.0], [1e4]], even,
         {"g1": np.array([0.0]), "g2": np.array([1e4])},
         {"correct_for_correlation": True},
         [[1.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.0, 1.0]],
         {"n_measurements": 1.0, "effective_dimension": 1.0, "scale_factor": 1.0}),
        ("an impossible well falls back to the sequencing fractions",
         impossible, {"g1": 0.75, "g2": 0.25}, far, {},
         [[0.75, 0.25]] * 3,
         {"n_measurements": 40.0, "effective_dimension": 40.0, "scale_factor": 1.0}),
        ("200 correlated hopeless columns still land on the prior",
         correlated, {"g1": 0.6, "g2": 0.4},
         {"g1": [300.0] * 200, "g2": [-300.0] * 200},
         {"correct_for_correlation": True}, [[0.6, 0.4]] * 4,
         {"n_measurements": 200.0, "effective_dimension": 1.0,
          "scale_factor": 1.0 / 200}),
        ("a 1-D vector is one measurement, default centres and scales",
         [1.0, 2.0, 3.0], even, {"g1": [1.0]},
         {"centres": None, "scales": None},
         [[0.1824255235029469, 0.8175744764970531],
          [0.49999999949142343, 0.5000000005085765],
          [0.8175744758902342, 0.18242552410976584]],
         {"n_measurements": 1.0, "effective_dimension": 1.0, "scale_factor": 1.0}),
    ]
    for case_id, values, priors, effects, kwargs, expected, report in cases:
        with subtests.test(msg=case_id):
            r, guides, diagnostics = _call(values, priors, effects, **kwargs)
            assert guides == tuple(priors)
            np.testing.assert_allclose(r, np.array(expected), rtol=1e-6, atol=1e-9)
            assert np.isfinite(r).all()
            np.testing.assert_allclose(r.sum(axis=1), 1.0, atol=1e-12)
            assert diagnostics.keys() == report.keys()
            for key, value in report.items():
                assert diagnostics[key] == pytest.approx(value, rel=1e-3)

    with subtests.test(msg="impossible rows stay strictly positive and mass-matched"):
        r, _, _ = _call(impossible, {"g1": 0.75, "g2": 0.25}, far)
        assert (r > 0).all()
        np.testing.assert_allclose(r.sum(axis=0), [2.25, 0.75], atol=1e-6)
    with subtests.test(msg="unfitted guide columns match the priors"):
        r, _, _ = _call([[2.0, 2.0], [0.0, 0.0], [-2.0, -2.0]],
                        {"fitted": 0.5, "unfitted": 0.5}, {"fitted": [2.0, 2.0]})
        np.testing.assert_allclose(r.sum(axis=0), [1.5, 1.5], atol=1e-6)

    with subtests.test(msg="duplicated columns do not vote eight times"):
        one_column = np.random.default_rng(4).normal(size=(200, 1))
        duplicated = np.repeat(one_column, 8, axis=1)
        effects = {"g1": [1.0] * 8, "g2": [-1.0] * 8}
        corrected, _, report = posterior_multivariate(duplicated, even, effects)
        plain, _, plain_report = posterior_multivariate(
            duplicated, even, effects, correct_for_correlation=False)
        assert report["n_measurements"] == 8.0
        assert report["effective_dimension"] == pytest.approx(1.0, abs=1e-6)
        assert report["scale_factor"] == pytest.approx(0.125, abs=1e-6)
        assert plain_report == {"n_measurements": 8.0,
                                "effective_dimension": 8.0, "scale_factor": 1.0}
        assert corrected.max() == pytest.approx(0.997742561269308, rel=1e-9)
        assert plain.max() == pytest.approx(1.0, abs=1e-12)

    empties = [
        ("no cells", np.zeros((0, 2)), even, (0, 2), ("g1", "g2")),
        ("no guides", np.zeros((2, 2)), {}, (2, 0), ()),
        ("no measurements", np.zeros((3, 0)), even, (3, 2), ("g1", "g2")),
    ]
    for case_id, values, priors, shape, guides in empties:
        with subtests.test(msg=case_id):
            r, names, report = posterior_multivariate(values, priors, {})
            assert r.shape == shape and not r.any()
            assert names == guides
            assert report == {"n_measurements": float(values.shape[1]),
                              "effective_dimension": 0.0, "scale_factor": 1.0}

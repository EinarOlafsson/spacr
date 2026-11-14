"""Table-driven tests of the control-based annotation QC and screen sizing.

``spacr.annotation_umap_qc`` picks an embedding recipe on held-out controls
and scores each cell's positive-control neighbourhood; ``spacr.annotation_power``
estimates the screen shape a reachable annotation needs.  Every early return
here is a refusal the user has to see: too few or misaligned controls, an
invalid split, a recipe the embedding library rejected, an embedding without
controls, a plate map without wells or guides, a decision threshold no
specificity can meet.  Each must come back as a stated reason or NaN, never
an exception and never a plausible-looking zero.  One general test per unit,
each walking a case table with pytest's ``subtests`` fixture.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from spacr import hyperparam
from spacr.annotation_power import screen_size_for
from spacr.annotation_umap_qc import (NEGATIVE, POSITIVE, fit_on_controls,
                                      neighbour_purity)

RECIPE = [{"n_neighbors": 5}]
TWENTY = np.arange(60, dtype=float).reshape(20, 3)
BALANCED = [POSITIVE] * 10 + [NEGATIVE] * 10


def test_fit_on_controls_states_why_it_cannot_pick_a_recipe(subtests,
                                                            monkeypatch):
    """Each unsplittable or unscorable control set returns its exact reason."""
    cases = [
        ("under_eight_controls", np.arange(12, dtype=float).reshape(4, 3),
         [POSITIVE, NEGATIVE] * 2, {},
         {"error": "too few control cells to split"}),
        ("a_label_per_row_is_required", TWENTY, [POSITIVE] * 19, {},
         {"error": "too few control cells to split"}),
        ("only_one_label", TWENTY, [POSITIVE] * 20, {},
         {"error": "the controls carry only one label"}),
        ("one_group_per_control_cell", TWENTY, BALANCED,
         {"groups": [f"well-{i}" for i in range(19)]},
         {"error": "one group is needed per control cell"}),
    ]
    for case_id, features, labels, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            assert fit_on_controls(features, labels, recipes=RECIPE,
                                   **kwargs) == expected

    for level in ("well", "plate"):
        with subtests.test(msg=f"invalid_holdout_keeps_level:{level}"):
            out = fit_on_controls(TWENTY, BALANCED,
                                  groups=[f"g-{i}" for i in range(20)],
                                  group_by=level, holdout=1.0, recipes=RECIPE)
            assert set(out) == {"error", "split_level"}
            assert out["split_level"] == level
            assert "strictly between 0 and 1" in out["error"]

    with subtests.test(msg="every_rejected_recipe_is_recorded"):
        def explode(*_args, **_kwargs):
            raise ValueError("n_neighbors must be smaller than the sample "
                             "size")

        monkeypatch.setattr(hyperparam, "_default_umap_embed", explode)
        features = np.random.default_rng(0).normal(size=(20, 4))
        out = fit_on_controls(features, BALANCED,
                              recipes=[{"n_neighbors": 500},
                                       {"n_neighbors": 900}])
        assert out == {
            "error": "no recipe produced a scorable embedding",
            "split_level": "cell",
            "trials": [
                {"recipe": {"n_neighbors": n},
                 "error": "ValueError: n_neighbors must be smaller than the "
                          "sample size"}
                for n in (500, 900)],
        }


def test_neighbour_purity_scores_the_positive_share_of_control_neighbours(
        subtests):
    """Exact per-cell purity; NaN where no eligible control neighbour exists."""
    line = np.array([[0.0], [1.0], [2.4], [3.0], [10.0]])
    marks = [POSITIVE, NEGATIVE, None, POSITIVE, None]
    rand = np.random.default_rng(1).normal(size=(6, 2))
    nan = math.nan
    cases = [
        ("nearest_control_only", line, marks, 1, [0.0, 1.0, 1.0, 0.0, 1.0]),
        ("k_capped_at_controls_minus_one", line, marks, 25,
         [0.5, 1.0, 0.5, 0.5, 0.5]),
        ("the_only_control_has_no_neighbour",
         np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
         [POSITIVE, None, None], 25, [nan, 1.0, 1.0]),
        ("no_controls_is_unknown_everywhere", rand, [None] * 6, 25,
         [nan] * 6),
        ("a_label_per_point_is_required", rand, [POSITIVE, None], 25,
         [nan] * 6),
    ]
    for case_id, points, labels, k, expected in cases:
        with subtests.test(msg=case_id):
            out = neighbour_purity(points, labels, k=k)
            np.testing.assert_array_equal(out, np.array(expected))


def test_screen_size_for_answers_honestly_on_degenerate_plate_maps(subtests):
    """Exact estimates, an error for no wells, NaN where no answer exists."""
    pairs = {"A1": {"g1": 0.5, "g2": 0.5}, "A2": {"g1": 0.5, "g3": 0.5}}
    floor = 0.012702078521939969
    nan = math.nan
    cases = [
        ("two_guides_per_well", pairs, {},
         {"floor": floor, "wells_now": 2, "library": 3,
          "guides_per_well_now": 2.0, "wells_per_guide_now": 4 / 3,
          "guides_per_well_needed": 1 / floor,
          "wells_needed": 3 * (4 / 3) * floor,
          "wells_multiplier": 3 * (4 / 3) * floor / 2,
          "library_if_wells_fixed": 2 / floor / (4 / 3), "target": 0.8,
          "specificity_needed_at_current_shape": 0.22272727272727288}),
        ("certainty_admits_no_specificity", pairs,
         {"decision": 1.0, "target": 0.9},
         {"floor": 1.0, "wells_now": 2, "library": 3,
          "guides_per_well_now": 2.0, "wells_per_guide_now": 4 / 3,
          "guides_per_well_needed": 1.0, "wells_needed": 4.0,
          "wells_multiplier": 2.0, "library_if_wells_fixed": 1.5,
          "target": 0.9, "specificity_needed_at_current_shape": nan}),
        ("wells_without_guides", {"A1": {}, "A2": {}}, {},
         {"floor": floor, "wells_now": 2, "library": 0,
          "guides_per_well_now": 0.0, "wells_per_guide_now": 0.0,
          "guides_per_well_needed": 1 / floor, "wells_needed": 0.0,
          "wells_multiplier": 0.0, "library_if_wells_fixed": nan,
          "target": 0.8, "specificity_needed_at_current_shape": nan}),
        ("no_wells", {}, {}, {"error": "no wells"}),
    ]
    for case_id, fractions, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            out = screen_size_for(fractions, sensitivity=0.95,
                                  specificity=0.99, **kwargs)
            assert list(out) == list(expected)
            for key, value in expected.items():
                if isinstance(value, float) and math.isnan(value):
                    assert math.isnan(out[key]), key
                else:
                    assert out[key] == pytest.approx(value), key

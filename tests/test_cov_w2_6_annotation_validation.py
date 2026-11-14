"""Validation that can tell a working method from working guide fractions.

The screens here are the module's own simulations, driven end to end: they
are the only place where cell-level guide truth exists, so a check that
stubbed them out would be checking the stub. Each test names the confound it
isolates -- penetrance, fraction inflation, classifier error, order
dependence -- rather than only that a number came back.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import annotation_validation as av


@pytest.fixture(scope="module")
def small_screen():
    """A clean, small screen: enough wells for a permutation to mean
    something, small enough to build many times."""
    return av.synthesise(wells=6, guides_per_well=2, cells_per_well=12,
                         guides=4, seed=3)


# --------------------------------------------------------------------------
# the simulation
# --------------------------------------------------------------------------

def test_a_screen_knows_how_many_cells_it_has(small_screen):
    assert len(small_screen) == 6 * 12
    assert small_screen.features.shape == (72, 6)
    assert len(small_screen.wells) == len(small_screen.truth) == 72


# --------------------------------------------------------------------------
# the null
# --------------------------------------------------------------------------

def test_a_permutation_keeps_the_cells_and_breaks_the_link(small_screen):
    null = av.permuted(small_screen, seed=1)
    assert null.features is small_screen.features
    assert null.truth == small_screen.truth
    assert sorted(null.fractions) == sorted(small_screen.fractions)
    assert null.meta["permuted"] is True
    assert sorted(null.true_fractions) == sorted(small_screen.true_fractions)


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------

def test_the_majority_baseline_gives_a_well_its_biggest_guide(small_screen):
    called = av.baseline_majority(small_screen)
    assert len(called) == len(small_screen)
    for well in set(small_screen.wells):
        here = {called[i] for i, w in enumerate(small_screen.wells)
                if w == well}
        assert here == {max(small_screen.fractions[well],
                            key=small_screen.fractions[well].get)}


def test_a_well_sequencing_reported_nothing_for_is_left_unannotated(
        small_screen):
    """Inventing a guide for a well with no counts would put cells behind a
    label no sequencing supports."""
    blank = av.Screen(features=small_screen.features,
                      scores=small_screen.scores, wells=small_screen.wells,
                      truth=small_screen.truth,
                      fractions={w: {} for w in small_screen.fractions},
                      true_fractions=small_screen.true_fractions,
                      guides=small_screen.guides)
    assert set(av.baseline_majority(blank)) == {"Non_annotated"}
    assert set(av.baseline_chance(blank)) == {"Non_annotated"}


def test_fractions_that_are_all_zero_leave_the_well_unannotated(small_screen):
    """There is nothing to sample from, and normalising zero counts would
    divide by zero to invent a uniform draw."""
    zeroed = av.Screen(features=small_screen.features,
                       scores=small_screen.scores, wells=small_screen.wells,
                       truth=small_screen.truth,
                       fractions={w: {g: 0.0 for g in f}
                                  for w, f in small_screen.fractions.items()},
                       true_fractions=small_screen.true_fractions,
                       guides=small_screen.guides)
    assert set(av.baseline_chance(zeroed)) == {"Non_annotated"}


def test_the_chance_baseline_only_ever_names_a_guide_that_well_reported(
        small_screen):
    called = av.baseline_chance(small_screen, seed=5)
    for index, well in enumerate(small_screen.wells):
        assert called[index] in small_screen.fractions[well]


# --------------------------------------------------------------------------
# the default scenarios
# --------------------------------------------------------------------------

def test_the_default_scenarios_isolate_the_confounds_they_are_named_for():
    scenes = av.default_scenarios(seed=100)
    assert set(scenes) == {"clean", "no_effect", "penetrance_0.5",
                           "inflated_fractions", "classifier_0.94",
                           "crowded", "realistic"}
    assert scenes["no_effect"].meta["effect"] == 0.0
    assert scenes["penetrance_0.5"].meta["penetrance"] == 0.5
    assert scenes["inflated_fractions"].meta["fraction_bias"] == 1.8
    assert scenes["classifier_0.94"].meta["classifier_accuracy"] == 0.94
    assert len(scenes["crowded"].guides) == 16


# --------------------------------------------------------------------------
# the slope fit and its reading
# --------------------------------------------------------------------------

def test_a_series_at_one_reported_value_yields_no_slope_at_all():
    """Every pair has the same x, so there is no line; NaN says so instead of
    a slope produced by dividing by zero."""
    slope, intercept = av._theil_sen(np.array([0.5, 0.5, 0.5]),
                                     np.array([0.1, 0.2, 0.3]))
    assert np.isnan(slope) and np.isnan(intercept)


def test_a_slope_that_could_not_be_fitted_says_so():
    assert av._read_slope(float("nan")) == "no slope could be fitted"


def test_imaging_finding_more_control_cells_is_read_as_understated():
    assert "understate" in av._read_slope(1.5)

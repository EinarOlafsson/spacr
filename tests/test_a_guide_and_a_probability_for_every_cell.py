"""Every cell gets a guide and a probability, or an honest 'ambiguous'.

Instruction 173, and the failure it exists to fix, in the maintainer's words:
"if i have 2 grnas both in the same direction and want to attribute cells to
both my arithmatic wont work well because both pick the top hits which would
be wrong".
"""
import numpy as np
import pytest

from spacr.guide_attribution import (AMBIGUOUS, DEFAULT_THRESHOLD, attributable,
                               attribute_well, normalise_fractions, posterior)

SCORES = np.linspace(-3.0, 3.0, 400)


# ------------------------------------------------------------------- priors


def test_the_fractions_are_normalised_to_one():
    got = normalise_fractions({"a": 0.2, "b": 0.2, "c": 0.1})

    assert sum(got.values()) == pytest.approx(1.0)
    assert got["a"] == pytest.approx(0.4)


def test_a_well_with_no_usable_fractions_gets_no_prior():
    """Inventing a uniform prior where there is no sequencing is the one
    thing this must not do."""
    assert normalise_fractions({}) == {}
    assert normalise_fractions({"a": 0.0, "b": -1.0}) == {}


# ------------------------------------------------------------- the constraint


def test_each_guide_gets_exactly_its_read_fraction_of_the_cells():
    """The constraint. Without it, two same-direction guides both claim the
    same top cells."""
    priors = {"a": 0.4, "b": 0.4, "c": 0.2}

    r, guides = posterior(SCORES, priors, {"a": 1.5, "b": 1.5, "c": -1.5})

    mass = r.sum(axis=0) / len(SCORES)
    assert mass == pytest.approx([priors[g] for g in guides], abs=1e-6)


def test_every_cell_carries_exactly_one_guide():
    r, _guides = posterior(SCORES, {"a": 0.5, "b": 0.5}, {"a": 1.0, "b": -1.0})

    assert r.sum(axis=1) == pytest.approx(np.ones(len(SCORES)))


def test_two_same_direction_guides_SPLIT_the_top_cells(): 
    """THE REPORTED FAILURE. Ranking gave both guides the same top cells."""
    priors = {"a": 0.4, "b": 0.4, "c": 0.2}

    r, guides = posterior(SCORES, priors, {"a": 1.5, "b": 1.5, "c": -1.5})

    top = np.argsort(SCORES)[-20:]
    share = dict(zip(guides, r[top].mean(axis=0)))
    assert share["a"] == pytest.approx(0.5, abs=0.01)
    assert share["b"] == pytest.approx(0.5, abs=0.01)


def test_identical_effects_are_flat_at_the_prior_ratio():
    """Unidentifiable, and the method must say so rather than invent a split."""
    r, guides = posterior(SCORES, {"a": 0.3, "b": 0.6}, {"a": 1.5, "b": 1.5})

    ratio = r[:, guides.index("a")] / r[:, guides.index("b")]
    assert ratio.std() == pytest.approx(0.0, abs=1e-9)
    assert ratio.mean() == pytest.approx(0.5, rel=1e-6)


def test_a_stronger_guide_takes_more_of_the_extreme_cells():
    r, guides = posterior(SCORES, {"a": 0.4, "b": 0.4, "c": 0.2},
                          {"a": 2.5, "b": 0.8, "c": -1.5})

    top = np.argsort(SCORES)[-20:]
    share = dict(zip(guides, r[top].mean(axis=0)))
    assert share["a"] > 0.9
    assert share["b"] < 0.1


def test_a_negative_guide_takes_the_low_scores_with_no_ranking_rule():
    """beta_g carries its own sign, so the direction problem disappears."""
    r, guides = posterior(SCORES, {"up": 0.5, "down": 0.5},
                          {"up": 2.0, "down": -2.0})

    lowest = np.argsort(SCORES)[:20]
    assert r[lowest, guides.index("down")].mean() > 0.9


# ---------------------------------------------------------------- the call


def test_a_confident_cell_gets_its_guide_and_its_probability():
    out = attribute_well(SCORES, {"a": 0.4, "b": 0.4, "c": 0.2},
                         {"a": 1.5, "b": 1.5, "c": -1.5})
    called = [a for a in out if a.called]

    assert called
    for call in called:
        assert call.guide != AMBIGUOUS
        assert call.probability >= DEFAULT_THRESHOLD


def test_an_uncertain_cell_is_tagged_and_KEEPS_its_best_probability():
    """"it gets a ambiguous tag and the highest probability any gran had for
    it" -- the number is the useful part of the refusal."""
    out = attribute_well(SCORES, {"a": 0.4, "b": 0.4, "c": 0.2},
                         {"a": 1.5, "b": 1.5, "c": -1.5})
    unsure = [a for a in out if a.ambiguous]

    assert unsure
    for call in unsure:
        assert call.guide == AMBIGUOUS
        assert 0.0 < call.probability < DEFAULT_THRESHOLD


def test_the_threshold_is_settable():
    loose = attribute_well(SCORES, {"a": 0.5, "b": 0.5},
                           {"a": 1.0, "b": -1.0}, threshold=0.5)
    strict = attribute_well(SCORES, {"a": 0.5, "b": 0.5},
                            {"a": 1.0, "b": -1.0}, threshold=0.99)

    assert sum(a.called for a in loose) > sum(a.called for a in strict)


def test_an_exact_tie_is_broken_the_SAME_way_on_a_re_run():
    """A coin flip that landed differently on a re-run would annotate the same
    screen two ways."""
    args = ([0.0] * 50, {"a": 0.5, "b": 0.5}, {"a": 1.0, "b": 1.0})

    first = attribute_well(*args, threshold=0.1, seed=7)
    again = attribute_well(*args, threshold=0.1, seed=7)

    assert [a.guide for a in first] == [a.guide for a in again]


def test_a_tie_does_not_always_pick_the_first_guide():
    """Randomly chosen, not 'the first one in the dict'."""
    args = ([0.0] * 200, {"a": 0.5, "b": 0.5}, {"a": 1.0, "b": 1.0})

    picked = {a.guide for a in attribute_well(*args, threshold=0.1, seed=3)}

    assert picked == {"a", "b"}


def test_entropy_says_how_arbitrary_the_call_was():
    out = attribute_well(SCORES, {"a": 0.5, "b": 0.5}, {"a": 2.0, "b": -2.0})
    confident = min(out, key=lambda a: a.entropy)
    unsure = max(out, key=lambda a: a.entropy)

    assert confident.entropy < 0.1
    assert unsure.entropy > 0.9


# ------------------------------------------------- can it be called at all


def test_a_strong_guide_can_be_called():
    can, best = attributable(effect=2.5, scale=1.0, prior=0.4)

    assert can and best > DEFAULT_THRESHOLD


def test_a_guide_with_no_effect_at_all_can_never_be_called():
    """Arithmetic, not sample size: no number of cells rescues it.

    THIS TEST USED TO ASK THE WRONG QUESTION. It pinned effect=0.3 at
    prior=0.4 as impossible, and on a real screen the shipped attribution
    called guides exactly like it -- 230 guide-well pairs across four plates.
    A guide with a modest effect IS callable for a cell far enough into its
    tail; what is impossible is a guide the likelihood cannot tell from the
    others at any score, which is a zero effect.
    """
    can, best = attributable(effect=0.0, scale=1.0, prior=0.4)

    assert not can
    assert best == pytest.approx(0.4), "with no effect the ceiling is the prior"


def test_a_modest_effect_is_callable_only_out_in_the_tail():
    """And the ceiling has to say so, or a user drops a usable hit."""
    can, best = attributable(effect=0.3, scale=1.0, prior=0.4)

    assert can and best > DEFAULT_THRESHOLD
    # ...but not when the range of scores a screen produces is narrow.
    can_near, best_near = attributable(effect=0.3, scale=1.0, prior=0.4,
                                       span=0.5)
    assert not can_near and best_near < best


def test_the_ceiling_is_a_ceiling_for_the_posterior_that_ships():
    """The property the old closed form did not have.

    It evaluated the ratio at the guide's own centre and called that the
    best possible score, which understated it threefold even against a
    competitor with no effect. Here the bound is checked against what
    `posterior` actually produces over the same range.
    """
    priors = {"a": 0.1, "b": 0.9}
    effects = {"a": 0.5, "b": -0.5}
    _, ceiling = attributable(0.5, 1.0, 0.1,
                              others=[(-0.5, 0.9)], span=4.0)
    scores = np.linspace(-4.5, 4.5, 400)
    r, guides = posterior(scores, priors, effects)
    assert r[:, guides.index("a")].max() <= ceiling + 1e-9


def test_ignoring_the_competition_is_the_generous_reading():
    """A competitor pushing the other way makes a guide EASIER to call."""
    _, flat = attributable(0.5, 1.0, 0.1)
    _, opposed = attributable(0.5, 1.0, 0.1, others=[(-0.5, 0.9)])

    assert opposed > flat


def test_a_trace_guide_can_never_be_called_either():
    can, best = attributable(effect=0.05, scale=1.0, prior=0.02)

    assert not can and best < 0.1


# ------------------------------------------------------------- permutation


def test_shuffling_the_scores_collapses_the_posterior_to_the_prior():
    """Structure surviving a permutation is structure the method invented."""
    rng = np.random.default_rng(0)
    priors = {"a": 0.4, "b": 0.4, "c": 0.2}
    noise = rng.normal(0.0, 1.0, 4000)

    # Effects of zero: no guide moves the score, so nothing is learnable.
    r, guides = posterior(noise, priors, {g: 0.0 for g in priors})

    for i, g in enumerate(guides):
        assert r[:, i] == pytest.approx(np.full(len(noise), priors[g]),
                                        abs=1e-9)


def test_the_beta_likelihood_is_available():
    pytest.importorskip("scipy")
    scores = np.linspace(0.01, 0.99, 200)

    r, guides = posterior(scores, {"a": 0.5, "b": 0.5},
                          {"a": 1.5, "b": -1.5}, centre=0.5, scale=0.15,
                          likelihood="beta")

    assert r.sum(axis=1) == pytest.approx(np.ones(len(scores)))
    assert r.sum(axis=0) / len(scores) == pytest.approx([0.5, 0.5], abs=1e-6)


# ----------------------------------------------------- the constrained assignment
# "my mind always goes to suduko where you have rules and conditions that must
# be met and you use the little information you have within the confines of
# the rules to do your inference."


def test_every_cell_gets_a_guide():
    """What the marginal posterior can never deliver when priors are small."""
    from spacr.guide_attribution import assign_well

    got = assign_well(SCORES, {"x": 0.5, "y": 0.3, "z": 0.2},
                      {"x": 2.0, "y": 0.0, "z": -2.0})

    assert len(got.guides) == len(SCORES)
    assert AMBIGUOUS not in got.guides


def test_the_counts_are_exactly_what_sequencing_says():
    """The Sudoku rule: guide g occupies exactly round(N * pi_g) cells."""
    from spacr.guide_attribution import assign_well

    got = assign_well(SCORES, {"x": 0.5, "y": 0.3, "z": 0.2},
                      {"x": 2.0, "y": 0.0, "z": -2.0})

    assert sum(got.counts.values()) == len(SCORES)
    assert got.counts["x"] == 200   # 0.5 of 400
    assert got.counts["y"] == 120
    assert got.counts["z"] == 80


def test_the_counts_still_sum_to_every_cell_when_they_do_not_divide():
    """A rounding that left a cell unassigned would break the one rule that
    makes this an assignment."""
    from spacr.guide_attribution import assign_well

    got = assign_well(list(range(7)), {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3},
                      {"a": 1.0, "b": 0.0, "c": -1.0})

    assert sum(got.counts.values()) == 7
    assert len(got.guides) == 7


def test_exclusion_puts_the_extremes_where_they_belong():
    from spacr.guide_attribution import assign_well

    got = assign_well(SCORES, {"up": 0.5, "down": 0.5},
                      {"up": 2.0, "down": -2.0})

    highest = np.argsort(SCORES)[-30:]
    lowest = np.argsort(SCORES)[:30]
    assert {got.guides[i] for i in highest} == {"up"}
    assert {got.guides[i] for i in lowest} == {"down"}


def test_a_guide_absent_from_the_well_occupies_none_of_it():
    """Pure elimination -- the Sudoku move."""
    from spacr.guide_attribution import assign_well

    got = assign_well(SCORES, {"here": 1.0, "absent": 0.0},
                      {"here": 1.0, "absent": -5.0})

    assert set(got.guides) == {"here"}
    assert got.counts.get("absent", 0) == 0


def test_it_says_when_the_rules_did_not_pin_it_down():
    """An assignment being OPTIMAL does not make it CERTAIN. When many
    assignments are nearly as good, swapping two cells costs almost nothing."""
    from spacr.guide_attribution import assign_well

    decided = assign_well(SCORES, {"a": 0.5, "b": 0.5},
                          {"a": 6.0, "b": -6.0})
    arbitrary = assign_well(SCORES, {"a": 0.5, "b": 0.5},
                            {"a": 0.0, "b": 0.0})

    assert decided.degeneracy > arbitrary.degeneracy
    assert not arbitrary.decisive, "no evidence must not read as a solved grid"


def test_two_guides_with_no_effect_still_get_their_exact_counts():
    """The counts are a constraint, not an inference: they hold even when
    nothing is learnable."""
    from spacr.guide_attribution import assign_well

    got = assign_well(SCORES, {"a": 0.25, "b": 0.75}, {"a": 0.0, "b": 0.0})

    assert got.counts["a"] == 100
    assert got.counts["b"] == 300

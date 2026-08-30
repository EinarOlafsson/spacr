"""Where an annotation strategy stops in the middle and says why.

Round 3 pinned the refusals a run makes on its way IN -- an empty table, a
score that never varies, a hold-out that swallowed the named wells. This
file picks up the ones a strategy can only discover once it is already
running: a self-training round that has nothing left to label, or nothing
confident enough to label it with; a positive-unlabelled split whose
held-out side kept no positive to measure the labelling rate on; two views
seeded with every cell there was; a score whose ties leave whole strata
empty. Each of them ends in a sentence naming the setting the user can
change, and each is checked beside an input that does not trigger it, so a
guard that started refusing everything would fail here rather than pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import regression_annotation as ra


def _screen(*, wells: int = 8, per_well: int = 8, seed: int = 0,
            annotate: str = "none", separation: float = 300.0) -> pd.DataFrame:
    """A small plate: one shape column, one intensity column, a score.

    ``annotate`` is ``"none"``, ``"all"`` or ``"some"`` -- the last leaves
    the later cells of every well without an annotation, which is what a
    part-way-through screen looks like. ``separation`` of zero makes the
    two classes indistinguishable, so no model can be confident about them.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for well in range(wells):
        for index in range(per_well):
            hit = index % 2 == 0
            row = {
                "plateID": "plate1",
                "rowID": f"r{1 + well // 4}",
                "columnID": f"c{1 + well % 4}",
                "fieldID": "f1",
                "cell_area": float(rng.normal(900 + separation * hit, 40)),
                "cell_channel_1_mean_intensity": float(
                    rng.normal(1200 + separation * hit, 60)),
                "pred": float(rng.uniform(0.4, 0.6) + 0.3 * hit),
            }
            if annotate != "none":
                written = annotate == "all" or index < 3
                row["label"] = ("hit" if hit else "control") if written else ""
            rows.append(row)
    return pd.DataFrame(rows)


def _request(frame: pd.DataFrame, **overrides) -> ra.AnnotationRequest:
    """A request over ``frame`` with every well eligible."""
    values = dict(frame=frame, score_column="pred", n_positive=4,
                  holdout_fraction=0.25, seed=1)
    values.update(overrides)
    return ra.AnnotationRequest(**values)


def _stopped(result) -> str:
    """The one note a self-training run writes about why it stopped."""
    return next(note for note in result.notes
                if note.startswith("stopped") or note.startswith("ran the "))


# ---------------------------------------------------------------------------
# Self-training: the two ways a round can have nothing left to do
# ---------------------------------------------------------------------------

def test_self_training_stops_when_every_cell_already_carries_a_label():
    """A confidence of 0.5 accepts every prediction -- the test either way
    of ``p >= c or p <= 1 - c`` -- so the first round pseudo-labels the whole
    selectable pool and the second finds it empty. It has to say so and
    keep the rounds it did run; falling through would ask the model to
    improve on a training set it has already absorbed entirely.
    """
    result = ra.run("self_training",
                    _request(_screen(seed=1), rounds=2, confidence=0.5,
                             seed=1))

    assert "every selectable cell already carries a label" in _stopped(result)
    curve = next(note for note in result.notes
                 if note.startswith("Audit curve"))
    assert curve.count(";") == 1, curve          # two rounds were fitted
    assert result.counts["pseudo_labelled"] > 0


def test_self_training_stops_when_nothing_is_confident_enough_to_accept():
    """The counterpart, driven by the same knob from the other end. With
    features that carry no class difference at all, no unlabelled cell
    reaches 0.999, so the round has nothing to accept -- and a round that
    accepted its own uncertain guesses is exactly how self-training
    reinforces its errors.
    """
    flat = _screen(seed=0, separation=0.0)
    result = ra.run("self_training",
                    _request(flat, rounds=3, confidence=0.999, seed=0))

    assert "no unlabelled cell reached the 1.00 confidence" in _stopped(result)
    assert result.counts["pseudo_labelled"] == 0
    assert result.counts["rounds"] == 1


# ---------------------------------------------------------------------------
# Positive-unlabelled learning: the inner split that estimates c
# ---------------------------------------------------------------------------

def test_the_inner_split_refusal_names_what_it_was_trying_to_estimate(
        monkeypatch):
    """The rate ``c`` is estimated on positives the model did not fit, which
    needs a second split inside the training set. A plain ``ValueError``
    from the splitter says nothing about why a split was being asked for at
    all, so it is re-raised with that sentence in front of it -- and with
    the splitter's own words kept, because they are what says which design
    was refused.
    """
    from spacr import classifier_evaluation as evaluation

    request = _request(_screen(), seed=2)
    setup = ra.prepare(request)          # built before the splitter is broken

    def refuse(*args, **kwargs):
        raise ValueError("one class on the training side")

    monkeypatch.setattr(evaluation, "grouped_split", refuse)
    with pytest.raises(ra.AnnotationStrategyError) as caught:
        ra.run("pu_learning", request, setup)

    assert "labelling rate cannot be estimated on cells the model did not " \
           "fit" in str(caught.value)
    assert "one class on the training side" in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)
    # The same run against the working splitter reaches the rate itself.
    monkeypatch.undo()
    assert ra.run("pu_learning", request, setup).counts["labelling_rate"] > 0


def test_an_inner_split_holding_no_positive_cannot_calibrate_the_rate(
        monkeypatch):
    """``c`` is the model's mean probability over held-out POSITIVES. A
    split that put every positive on the training side leaves that mean
    undefined, and a rescaling divided by an undefined rate is not a
    correction -- it is a number. The refusal has to come before the
    division rather than after it.
    """
    from spacr import classifier_evaluation as evaluation

    request = _request(_screen(), seed=3)
    setup = ra.prepare(request)
    real = evaluation.grouped_split

    def positives_stay_in_training(groups, labels, holdout, **kwargs):
        """The real split, then every positive moved to the training side."""
        train, held, report = real(groups, labels, holdout, **kwargs)
        y = np.asarray(labels)
        kept = np.asarray([i for i in held if y[i] == 0], dtype=int)
        moved = np.asarray([i for i in held if y[i] == 1], dtype=int)
        return (np.sort(np.concatenate([np.asarray(train, dtype=int), moved])),
                kept, report)

    monkeypatch.setattr(evaluation, "grouped_split",
                        positives_stay_in_training)
    with pytest.raises(ra.NotEnoughLabels, match="no labelled positive"):
        ra.run("pu_learning", request, setup)

    # Unpatched, the same request estimates a rate and says so in words.
    monkeypatch.undo()
    result = ra.run("pu_learning", request, setup)
    assert 0.0 < result.counts["labelling_rate"] <= 1.0
    assert any("held-out positive" in note for note in result.notes)


# ---------------------------------------------------------------------------
# Two views, and the seed set that leaves them nothing to disagree about
# ---------------------------------------------------------------------------

def test_two_views_asked_about_a_fully_annotated_screen_have_nothing_to_rank():
    """The seed set is every annotated cell outside the hold-out. When the
    whole screen is annotated that IS the selectable pool, so the two views
    are being asked to disagree about cells they were both fitted on -- a
    queue drawn from it would be a queue of cells already answered.
    """
    request = _request(_screen(annotate="all"), label_column="label")

    with pytest.raises(ra.AnnotationStrategyError,
                       match="nothing for two views to disagree about"):
        ra.run("two_view_disagreement", request)

    # Leave the later cells of each well unannotated and the same strategy
    # has a pool again, which is what says the refusal is about the pool.
    partly = _request(_screen(annotate="some"), label_column="label")
    result = ra.run("two_view_disagreement", partly)
    assert result.counts["queued"] > 0
    assert result.counts["intensity_columns"] >= 1
    assert result.counts["shape_columns"] >= 1


# ---------------------------------------------------------------------------
# Score strata
# ---------------------------------------------------------------------------

def test_a_score_with_one_finite_value_cannot_be_divided_into_strata():
    """Stratifying needs a range, and one number is not one. The labels here
    come from the annotation column, so the run gets as far as the strategy
    before the missing score matters -- which is the case that would
    otherwise divide by an empty quantile.
    """
    frame = _screen(annotate="all")
    values = np.full(len(frame), np.nan)
    values[0] = 0.5
    frame["pred"] = values

    with pytest.raises(ra.AnnotationStrategyError,
                       match="Fewer than two selectable cells carry a finite "
                             "score"):
        ra.run("score_strata", _request(frame, label_column="label"))


def test_strata_nobody_lands_in_are_counted_and_explained():
    """Equal-count edges over a score with two distinct values collapse: the
    quantiles repeat, three of five strata get no cell, and the run must
    report a queue from the two that did rather than index into an empty
    member list. The empty ones are counted in the note, because a user who
    asked for five strata and got two is owed the reason.
    """
    frame = _screen()
    frame["pred"] = np.where(np.arange(len(frame)) % 2 == 0, 0.9, 0.1)

    result = ra.run("score_strata",
                    _request(frame, n_bins=5, n_positive=10))

    counts = result.counts
    assert counts["strata"] == 5
    empty = [name for name, size in counts.items()
             if name.startswith("stratum_") and size == 0]
    assert len(empty) == 3, counts
    assert any(f"{len(empty)} stratum/strata are empty" in note
               for note in result.notes)
    assert counts["queued"] > 0

    # A score that does vary fills every stratum, so the note is about the
    # ties and not about the strategy always reporting empties.
    varied = frame.copy()
    varied["pred"] = np.linspace(0.0, 1.0, len(varied))
    spread = ra.run("score_strata", _request(varied, n_bins=5, n_positive=10))
    assert all(size > 0 for name, size in spread.counts.items()
               if name.startswith("stratum_"))
    assert not any("stratum/strata are empty" in note
                   for note in spread.notes)


# ---------------------------------------------------------------------------
# Neighbour propagation
# ---------------------------------------------------------------------------

def test_propagation_refuses_a_table_it_cannot_measure_a_distance_in():
    """One setup is shared across strategies, and `prepare` relaxes the
    feature requirement for a strategy that fits nothing. Handed that setup,
    propagation must refuse: the radius it reports would be a distance in a
    space of no dimensions.
    """
    bare = _screen(annotate="all")[
        ["plateID", "rowID", "columnID", "fieldID", "pred", "label"]].copy()
    request = _request(bare, label_column="label")
    setup = ra.prepare(request, ra.SCORE_STRATA)
    assert setup.features == ()

    with pytest.raises(ra.AnnotationStrategyError,
                       match="no feature column to measure a distance in"):
        ra.run("neighbour_propagation", request, setup)


def test_a_pool_too_large_to_measure_is_sampled_and_the_run_says_so(
        monkeypatch):
    """The nearest-neighbour search builds a dense matrix, so a screen
    larger than the ceiling is sampled down to it. The sample is a fact
    about the answer -- the radius is a quantile of the distances THIS pool
    produced -- so the run reports the cut it made rather than presenting a
    sampled search as a complete one.
    """
    monkeypatch.setattr(ra, "MAX_POOL_FOR_DISTANCES", 10)
    frame = _screen()

    result = ra.run("neighbour_propagation",
                    _request(frame, distance_quantile=0.5))

    note = next(note for note in result.notes if "dense matrix" in note)
    assert "10 this search builds a dense matrix over" in note
    assert "48 selectable cells is more than" in note
    assert result.counts["propagated"] > 0

    # Raise the ceiling above the screen and the note is gone, which is what
    # says it describes the sampling rather than the strategy.
    monkeypatch.setattr(ra, "MAX_POOL_FOR_DISTANCES", 50_000)
    whole = ra.run("neighbour_propagation",
                   _request(frame, distance_quantile=0.5))
    assert not any("dense matrix" in note for note in whole.notes)
    assert whole.counts["seeds"] == result.counts["seeds"]

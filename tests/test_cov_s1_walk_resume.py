"""A Walk that is interrupted, resumed, or given a fit function that fails.

A hyperparameter Walk is hours of fits, so every one of these is a real
operating condition rather than an edge case: a machine that was rebooted, a
checkpoint written by an older build, a configuration UMAP refuses, a round in
which nothing succeeded. What is pinned here is that none of them costs the
work already done, and none of them ends the search silently.
"""
from __future__ import annotations

import json

import pytest

from spacr import hyperparam as hp
from spacr.hyperparam import Trial, WalkAxis, walk_search


def _checkpoint(tmp_path, *, resume=False, keep_embeddings=False):
    return hp._UmapCheckpoint(str(tmp_path / "walk.json"),
                              {"features": "matrix-a"}, resume,
                              keep_embeddings)


def _axes():
    return [WalkAxis("n_neighbors", step=5.0, minimum=2.0, integer=True),
            WalkAxis("min_dist", step=0.05, minimum=0.0, maximum=1.0)]


# ---------------------------------------------------------------------------
# Resuming
# ---------------------------------------------------------------------------

def test_a_checkpoint_from_the_two_axis_build_still_names_its_centre(tmp_path):
    """1.5.x wrote ``centre_n``/``centre_d`` rather than a ``centre`` map.

    A user who started a search on that build and resumes on this one must
    continue from where they were, not from the starting point they typed
    hours ago.
    """
    store = _checkpoint(tmp_path)
    store.record(Trial(params={"n_neighbors": 20, "min_dist": 0.2},
                       score=0.5, index=0), round_index=0)
    store.finish({"rounds_completed": 1, "centre_n": 20, "centre_d": 0.2,
                  "best_score": 0.5})

    seen = []

    def _fit(params):
        seen.append(dict(params))
        return 0.4

    result = walk_search(_fit, {"n_neighbors": 5, "min_dist": 0.05}, _axes(),
                         n_trials=2, checkpoint=_checkpoint(tmp_path,
                                                            resume=True))

    assert seen, "the resumed walk still has rounds to run"
    assert any(row["n_neighbors"] in (15, 25) for row in seen), \
        "the walk moved on from the stored centre of 20, not from 5"
    assert any("Resumed" in note for note in result.notes)


def test_a_resumed_walk_replays_its_completed_trials_to_the_caller(tmp_path):
    """The progress bar has to account for the work already done, or a
    resumed search looks like it started over."""
    store = _checkpoint(tmp_path)
    for index, neighbours in enumerate((5, 10)):
        store.record(Trial(params={"n_neighbors": neighbours,
                                   "min_dist": 0.05},
                           score=0.1 * index, index=index), round_index=0)
    store.finish({"rounds_completed": 1,
                  "centre": {"n_neighbors": 10, "min_dist": 0.05},
                  "best_score": "not a number"})

    reported = []
    result = walk_search(lambda params: 0.05,
                         {"n_neighbors": 10, "min_dist": 0.05}, _axes(),
                         n_trials=2,
                         on_trial=lambda trial, done, total: reported.append(
                             (trial.params.get("n_neighbors"), done, total)),
                         checkpoint=_checkpoint(tmp_path, resume=True))

    replayed = [row for row in reported if row[0] in (5, 10)]
    assert replayed, "the completed trials are reported before the new ones"
    assert [trial.params["n_neighbors"] for trial in result.trials[:2]] == \
        [5, 10]


# ---------------------------------------------------------------------------
# A fit function that will not cooperate
# ---------------------------------------------------------------------------

def test_a_configuration_the_fit_refuses_is_a_failed_trial_not_a_failed_walk(
        tmp_path):
    """UMAP raises on some parameter combinations. The walk has to record
    that one and keep going, because the point of a walk is to find out which
    combinations work."""
    def _fit(params):
        if params["n_neighbors"] > 10:
            raise RuntimeError("n_neighbors larger than the sample")
        return 0.5

    result = walk_search(_fit, {"n_neighbors": 10, "min_dist": 0.1}, _axes(),
                         n_trials=2)

    failed = [trial for trial in result.trials if trial.error]
    assert failed
    assert failed[0].error.startswith(
        "RuntimeError: n_neighbors larger than the sample")
    assert any(trial.ok for trial in result.trials)


def test_a_fit_that_returns_no_score_says_so_on_the_row(tmp_path):
    """``None`` is not a score and must not rank as one. The row carries the
    reason so the table shows what happened to that configuration."""
    result = walk_search(lambda params: None,
                         {"n_neighbors": 10, "min_dist": 0.1}, _axes(),
                         n_trials=1)

    assert result.trials
    assert all(trial.score is None for trial in result.trials)
    assert all("returned no score" in (trial.error or "")
               for trial in result.trials)


def test_a_round_in_which_nothing_succeeded_stops_the_walk(tmp_path):
    """There is no direction to move in from a neighbourhood with no scores,
    and the checkpoint is updated so a resume does not repeat the round."""
    checkpoint = _checkpoint(tmp_path)

    def _fit(_params):
        raise RuntimeError("every fit fails here")

    result = walk_search(_fit, {"n_neighbors": 10, "min_dist": 0.1}, _axes(),
                         n_trials=5, checkpoint=checkpoint)

    assert result.trials
    assert all(trial.error for trial in result.trials)
    assert len(result.trials) <= 4, "the walk stopped after the first round"


def test_a_walk_that_stops_improving_records_where_it_got_to(tmp_path):
    """The stopping threshold is the whole reason a walk terminates, and the
    round it stopped on has to be on disk so a resume does not redo it."""
    checkpoint = _checkpoint(tmp_path)

    result = walk_search(lambda params: 0.5,
                         {"n_neighbors": 10, "min_dist": 0.1}, _axes(),
                         n_trials=5, checkpoint=checkpoint)

    assert any("stopped because" in note for note in result.notes)
    stored = json.loads((tmp_path / "walk.json").read_text())
    assert stored["meta"]["rounds_completed"] >= 1


def test_a_walk_with_nowhere_left_to_go_says_so_rather_than_spinning():
    """An axis whose bounds pin it to one value has no neighbourhood.

    Every candidate clamps back onto the centre, which has already been
    scored. Without the note the walk would burn its remaining rounds
    producing no trials and finish looking like it converged.
    """
    pinned = WalkAxis("n_neighbors", step=5.0, minimum=15.0, maximum=15.0,
                      integer=True)

    result = walk_search(lambda params: 0.5, {"n_neighbors": 15}, [pinned],
                         n_trials=10)

    assert any("no configuration left to try" in note
               for note in result.notes)
    assert result.trials == []
    assert result.best is None


def test_a_hand_edited_checkpoint_entry_is_skipped_not_fatal(tmp_path):
    """The checkpoint is a JSON document a user can open and break.

    An entry that is not a record cannot be turned into a Trial; reading one
    would end the resume with a TypeError rather than recomputing that single
    configuration.
    """
    store = _checkpoint(tmp_path)
    store.record(Trial(params={"n_neighbors": 5}, score=0.8, index=0))
    store.finish()
    document = json.loads((tmp_path / "walk.json").read_text())
    document["completed"]["hand-edited"] = "not a record"
    (tmp_path / "walk.json").write_text(json.dumps(document))

    loaded = _checkpoint(tmp_path, resume=True).load()

    assert "hand-edited" not in loaded
    assert len(loaded) == 1


def test_a_legacy_centre_that_cannot_be_read_leaves_the_axis_alone(tmp_path):
    """The old two-axis checkpoint wrote bare numbers. One that is not a
    number is one axis lost, not a resume lost."""
    store = _checkpoint(tmp_path)
    store.record(Trial(params={"n_neighbors": 20}, score=0.5, index=0),
                 round_index=0)
    store.finish({"rounds_completed": 1, "centre_n": "twenty",
                  "best_score": 0.5})

    seen = []
    walk_search(lambda params: seen.append(dict(params)) or 0.9,
                {"n_neighbors": 10},
                [WalkAxis("n_neighbors", step=5.0, minimum=2.0,
                          integer=True)],
                n_trials=2, checkpoint=_checkpoint(tmp_path, resume=True))

    assert seen, "the walk still ran"
    assert {row["n_neighbors"] for row in seen} <= {5, 15}, \
        "the unreadable centre left the typed starting point in place"


def test_a_trial_already_scored_for_this_round_is_not_fitted_again(tmp_path):
    """A walk interrupted mid-round has scored some of that round already.

    Refitting those is the cost the checkpoint exists to avoid, and it is
    paid in hours.
    """
    store = _checkpoint(tmp_path)
    store.record(Trial(params={"n_neighbors": 5}, score=0.9, index=0),
                 round_index=1)
    store.finish({"rounds_completed": 1, "centre": {"n_neighbors": 10},
                  "best_score": 0.1})

    fitted = []
    reported = []
    result = walk_search(lambda params: fitted.append(dict(params)) or 0.2,
                         {"n_neighbors": 10},
                         [WalkAxis("n_neighbors", step=5.0, minimum=2.0,
                                   integer=True)],
                         n_trials=2,
                         on_trial=lambda t, done, total: reported.append(done),
                         checkpoint=_checkpoint(tmp_path, resume=True))

    assert {row["n_neighbors"] for row in fitted} == {15}, \
        "n_neighbors=5 was already scored in this round"
    assert any(trial.params["n_neighbors"] == 5 and trial.score == 0.9
               for trial in result.trials)
    assert reported


def test_a_cancelled_walk_is_cancelled_and_not_recorded_as_a_failure(
        tmp_path):
    """``PipelineCancelled`` is the user pressing Stop.

    Catching it as a trial error would turn a deliberate stop into a table
    full of red rows and let the walk carry on to the next configuration.
    """
    from spacr.qt.bridge import PipelineCancelled

    def _stopped(_params):
        raise PipelineCancelled("stopped by the user")

    with pytest.raises(PipelineCancelled):
        walk_search(_stopped, {"n_neighbors": 10},
                    [WalkAxis("n_neighbors", step=5.0, integer=True)],
                    n_trials=2)
